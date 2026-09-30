//! Exact result-cache payload construction, independent of parser and lexer.
//!
//! Choosing a representation does not change cache admission, key identity,
//! eviction, or the words restored on a hit. Count the two sparse candidates
//! together, then allocate only the representation selected by the original
//! byte-cost rule.

use std::sync::Arc;

#[derive(Debug)]
#[cfg_attr(test, derive(PartialEq, Eq))]
pub(super) enum DynamicMaskCachePayload {
    /// The key was observed once, but its result has not been stored. A repeat
    /// upgrades this entry using the unchanged cache-admission policy.
    Probation,
    Dense(Arc<[u32]>),
    SparseZero(Box<[(u32, u32)]>),
    SparseAllOriginal(Box<[(u32, u32)]>),
}

/// A missing baseline suffix denotes zero words, matching cache-hit decoding.
/// A baseline longer than the requested mask contributes nothing beyond it.
#[inline]
fn sparse_counts(mask: &[u32], baseline: &[u32]) -> (usize, usize) {
    let shared = mask.len().min(baseline.len());
    let (mut nonzero, mut different) = (0, 0);
    for (&word, &original) in mask[..shared].iter().zip(baseline) {
        nonzero += usize::from(word != 0);
        different += usize::from(word != original);
    }
    for &word in &mask[shared..] {
        let present = usize::from(word != 0);
        nonzero += present;
        different += present;
    }
    (nonzero, different)
}

/// Preserve the existing representation, tie-breaking and sparse allocation
/// sequence. Only the classification scans are fused: exact pre-reservation
/// changed allocation history and regressed a loaded JavaScript warm tail.
#[inline]
pub(super) fn from_words(mask: &[u32], baseline: &[u32]) -> DynamicMaskCachePayload {
    let (nonzero, different) = sparse_counts(mask, baseline);
    let dense_bytes = mask.len().saturating_mul(std::mem::size_of::<u32>());
    let sparse_zero_bytes = nonzero.saturating_mul(std::mem::size_of::<(u32, u32)>());
    let sparse_baseline_bytes = different.saturating_mul(std::mem::size_of::<(u32, u32)>());
    if sparse_zero_bytes < dense_bytes && sparse_zero_bytes <= sparse_baseline_bytes {
        DynamicMaskCachePayload::SparseZero(
            mask.iter()
                .enumerate()
                .filter_map(|(i, &word)| (word != 0).then_some((i as u32, word)))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        )
    } else if sparse_baseline_bytes < dense_bytes {
        DynamicMaskCachePayload::SparseAllOriginal(
            mask.iter()
                .enumerate()
                .filter_map(|(i, &word)| {
                    (word != baseline.get(i).copied().unwrap_or(0)).then_some((i as u32, word))
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        )
    } else {
        DynamicMaskCachePayload::Dense(Arc::from(mask))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Deliberately retain the original independent two-pass implementation as
    // an oracle for representation, tie-breaking and sparse entry ordering.
    fn original(mask: &[u32], baseline: &[u32]) -> DynamicMaskCachePayload {
        let nonzero = mask.iter().filter(|&&word| word != 0).count();
        let different = mask
            .iter()
            .enumerate()
            .filter(|&(index, &word)| word != baseline.get(index).copied().unwrap_or(0))
            .count();
        let dense_bytes = mask.len().saturating_mul(std::mem::size_of::<u32>());
        let sparse_zero_bytes = nonzero.saturating_mul(std::mem::size_of::<(u32, u32)>());
        let sparse_baseline_bytes = different.saturating_mul(std::mem::size_of::<(u32, u32)>());
        if sparse_zero_bytes < dense_bytes && sparse_zero_bytes <= sparse_baseline_bytes {
            DynamicMaskCachePayload::SparseZero(
                mask.iter()
                    .enumerate()
                    .filter_map(|(i, &word)| (word != 0).then_some((i as u32, word)))
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            )
        } else if sparse_baseline_bytes < dense_bytes {
            DynamicMaskCachePayload::SparseAllOriginal(
                mask.iter()
                    .enumerate()
                    .filter_map(|(i, &word)| {
                        (word != baseline.get(i).copied().unwrap_or(0)).then_some((i as u32, word))
                    })
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            )
        } else {
            DynamicMaskCachePayload::Dense(Arc::from(mask))
        }
    }

    fn check(mask: &[u32], baseline: &[u32]) {
        let actual = from_words(mask, baseline);
        assert_eq!(actual, original(mask, baseline));
        let mut decoded = vec![0; mask.len()];
        match actual {
            DynamicMaskCachePayload::Dense(words) => decoded.copy_from_slice(&words),
            DynamicMaskCachePayload::SparseZero(entries) => {
                for &(i, word) in &entries {
                    decoded[i as usize] = word;
                }
            }
            DynamicMaskCachePayload::SparseAllOriginal(entries) => {
                let shared = decoded.len().min(baseline.len());
                decoded[..shared].copy_from_slice(&baseline[..shared]);
                for &(i, word) in &entries {
                    decoded[i as usize] = word;
                }
            }
            DynamicMaskCachePayload::Probation => {
                panic!("payload construction cannot produce probation")
            }
        }
        assert_eq!(decoded, mask);
    }

    #[test]
    fn every_small_mask_and_baseline_preserves_payload_and_words() {
        let mut words = vec![Vec::new()];
        for len in 1..=4 {
            for encoded in 0..3usize.pow(len) {
                let mut value = encoded;
                let mut word = Vec::new();
                for _ in 0..len {
                    word.push([0, 1, u32::MAX][value % 3]);
                    value /= 3;
                }
                words.push(word);
            }
        }
        for mask in &words {
            for baseline in &words {
                check(mask, baseline);
            }
        }
    }

    #[test]
    fn large_sparse_dense_and_truncated_baselines_match_reference() {
        let mut seed = 0x6088940720u64;
        let mut random = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed as u32
        };
        for len in [8, 63, 64, 65, 127, 128, 1024, 4009, 4096, 8192] {
            for base_len in [0, 1, len / 2, len, len + 17] {
                let baseline = (0..base_len).map(|_| random()).collect::<Vec<_>>();
                for _ in 0..8 {
                    let mask = (0..len)
                        .map(|i| match random() % 4 {
                            0 => 0,
                            1 => baseline.get(i).copied().unwrap_or(0),
                            _ => random(),
                        })
                        .collect::<Vec<_>>();
                    check(&mask, &baseline);
                }
                check(&vec![0; len], &baseline);
                check(&vec![u32::MAX; len], &baseline);
            }
        }
    }

    #[test]
    fn dense_ties_and_sparse_zero_ties_keep_original_preference() {
        assert!(matches!(
            from_words(&[], &[]),
            DynamicMaskCachePayload::Dense(_)
        ));
        assert!(matches!(
            from_words(&[1, 0], &[0, 1]),
            DynamicMaskCachePayload::Dense(_)
        ));
        assert!(matches!(
            from_words(&[1, 0, 0, 0], &[]),
            DynamicMaskCachePayload::SparseZero(_)
        ));
        assert!(matches!(
            from_words(&[1, 1, 1, 1], &[1, 1, 1, 1]),
            DynamicMaskCachePayload::SparseAllOriginal(_)
        ));
    }
}
