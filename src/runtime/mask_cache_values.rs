//! Bounded sharing of exact mask values between independently validated keys.
//!
//! A value is considered only after the caller has computed the entire mask.
//! Reusing it therefore does not assert that two parser states are equivalent.
//! Hash collisions are resolved by full output equality, including sparse gaps.

use super::mask_cache_payload::DynamicMaskCachePayload;
use rustc_hash::FxHasher;
use std::hash::Hasher;
use std::sync::{Arc, Weak};

const MAX_RECENT_VALUES: usize = 64;

struct RecentValue {
    hash: u64,
    word_count: usize,
    payload: Weak<DynamicMaskCachePayload>,
}

/// The index does not own payloads. Evicting the last state entry still frees
/// its value; at most 64 small weak control blocks remain until overwritten.
#[derive(Default)]
pub(super) struct MaskValueIndex {
    values: Vec<RecentValue>,
    next: usize,
}

impl std::fmt::Debug for MaskValueIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MaskValueIndex")
            .field("entries", &self.values.len())
            .field("next", &self.next)
            .finish()
    }
}

#[inline]
pub(super) fn mask_hash(mask: &[u32]) -> u64 {
    // A single serial hash dependency chain costs more than copying a mask.
    // Independent fixed lanes overlap that arithmetic without requiring a CPU
    // extension. This fingerprint is only an index: every hit still checks
    // the complete output below, so a collision cannot admit a wrong token.
    let mut lanes: [FxHasher; 8] = std::array::from_fn(|_| FxHasher::default());
    let mut blocks = mask.chunks_exact(16);
    for words in &mut blocks {
        for lane in 0..8 {
            lanes[lane]
                .write_u64(u64::from(words[lane * 2]) | (u64::from(words[lane * 2 + 1]) << 32));
        }
    }
    let mut result = FxHasher::default();
    result.write_usize(mask.len());
    for lane in &lanes {
        result.write_u64(lane.finish());
    }
    for &word in blocks.remainder() {
        result.write_u32(word);
    }
    result.finish()
}

fn matches_baseline(mask: &[u32], baseline: &[u32], begin: usize, end: usize) -> bool {
    let shared = end.min(baseline.len());
    if begin < shared && mask[begin..shared] != baseline[begin..shared] {
        return false;
    }
    mask[begin.max(shared)..end].iter().all(|&word| word == 0)
}

/// Compare the complete decoded value without allocating a dense temporary.
fn matches_mask(payload: &DynamicMaskCachePayload, mask: &[u32], baseline: &[u32]) -> bool {
    match payload {
        DynamicMaskCachePayload::Probation => false,
        DynamicMaskCachePayload::Dense(words) => words.as_ref() == mask,
        DynamicMaskCachePayload::SparseZero(words) => {
            let mut begin = 0;
            for &(index, value) in words.iter() {
                let index = index as usize;
                if index < begin
                    || index >= mask.len()
                    || mask[index] != value
                    || mask[begin..index].iter().any(|&word| word != 0)
                {
                    return false;
                }
                begin = index + 1;
            }
            mask[begin..].iter().all(|&word| word == 0)
        }
        DynamicMaskCachePayload::SparseAllOriginal(words) => {
            let mut begin = 0;
            for &(index, value) in words.iter() {
                let index = index as usize;
                if index < begin
                    || index >= mask.len()
                    || mask[index] != value
                    || !matches_baseline(mask, baseline, begin, index)
                {
                    return false;
                }
                begin = index + 1;
            }
            matches_baseline(mask, baseline, begin, mask.len())
        }
    }
}

impl MaskValueIndex {
    pub(super) fn get(
        &self,
        hash: u64,
        mask: &[u32],
        baseline: &[u32],
    ) -> Option<Arc<DynamicMaskCachePayload>> {
        self.values.iter().rev().find_map(|entry| {
            if entry.hash != hash || entry.word_count != mask.len() {
                return None;
            }
            let value = entry.payload.upgrade()?;
            matches_mask(&value, mask, baseline).then_some(value)
        })
    }

    pub(super) fn insert(
        &mut self,
        hash: u64,
        word_count: usize,
        value: &Arc<DynamicMaskCachePayload>,
    ) {
        let entry = RecentValue {
            hash,
            word_count,
            payload: Arc::downgrade(value),
        };
        if self.values.len() < MAX_RECENT_VALUES {
            self.values.push(entry);
        } else {
            self.values[self.next] = entry;
            self.next = (self.next + 1) % MAX_RECENT_VALUES;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::mask_cache_payload::from_words;

    #[test]
    fn exact_value_equality_preserves_sparse_gaps_and_baseline_suffixes() {
        let masks = [
            vec![],
            vec![0; 20],
            vec![7; 20],
            vec![0, 3, 0, 0, 0],
            vec![u32::MAX, 0, 7, 0, 1],
            vec![u32::MAX; 20],
        ];
        for mask in &masks {
            for baseline in &masks {
                let payload = from_words(mask, baseline);
                assert!(matches_mask(&payload, mask, baseline));
                for index in 0..mask.len() {
                    let mut changed = mask.clone();
                    changed[index] ^= 1;
                    assert!(!matches_mask(&payload, &changed, baseline));
                }
            }
        }
        assert!(!matches_mask(&DynamicMaskCachePayload::Probation, &[], &[]));
    }

    #[test]
    fn hash_collision_is_not_mask_equality_and_eviction_is_bounded() {
        let mut index = MaskValueIndex::default();
        let first = Arc::new(from_words(&[1, 0, 0, 0], &[]));
        let second = Arc::new(from_words(&[2, 0, 0, 0], &[]));
        index.insert(42, 4, &first);
        index.insert(42, 4, &second);
        assert!(Arc::ptr_eq(
            &index.get(42, &[1, 0, 0, 0], &[]).unwrap(),
            &first
        ));
        assert!(Arc::ptr_eq(
            &index.get(42, &[2, 0, 0, 0], &[]).unwrap(),
            &second
        ));
        assert!(index.get(42, &[3, 0, 0, 0], &[]).is_none());
        assert!(index.get(42, &[1, 0, 0, 0, 0], &[]).is_none());
        drop(first);
        assert!(index.get(42, &[1, 0, 0, 0], &[]).is_none());
        for value in 0..MAX_RECENT_VALUES * 3 {
            let payload = Arc::new(from_words(&[value as u32; 4], &[]));
            index.insert(value as u64, 4, &payload);
            assert!(index.get(value as u64, &[value as u32; 4], &[]).is_some());
        }
        assert_eq!(index.values.len(), MAX_RECENT_VALUES);
        assert!(
            index
                .values
                .iter()
                .all(|entry| entry.payload.strong_count() == 0)
        );
    }

    #[test]
    fn duplicate_output_can_share_without_changing_key_or_payload() {
        let words = vec![u32::MAX; 4008];
        let hash = mask_hash(&words);
        let payload = Arc::new(from_words(&words, &words));
        let mut index = MaskValueIndex::default();
        index.insert(hash, words.len(), &payload);
        let other_key_value = index.get(hash, &words, &words).unwrap();
        assert!(Arc::ptr_eq(&payload, &other_key_value));
        assert_eq!(Arc::strong_count(&payload), 2);
        drop(payload);
        assert!(index.get(hash, &words, &words).is_some());
        drop(other_key_value);
        assert!(index.get(hash, &words, &words).is_none());
    }

    #[test]
    fn fingerprint_handles_every_remainder_and_keeps_order_information() {
        for len in 0..64 {
            let words = (0..len)
                .map(|i| (i as u32).wrapping_mul(0x9e3779b1))
                .collect::<Vec<_>>();
            assert_eq!(mask_hash(&words), mask_hash(&words.clone()));
            for i in 0..len {
                let mut other = words.clone();
                other[i] ^= 1;
                // These fixed fixtures guard accidentally omitted lanes or
                // suffix words. Exact equality, not hash uniqueness, remains
                // authoritative for arbitrary inputs.
                assert_ne!(mask_hash(&words), mask_hash(&other));
            }
        }
    }
}
