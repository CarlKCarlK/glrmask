//! Derived original-token projections for packed non-DWA final weights.
//!
//! The wire pool remains authoritative. Small projections use the same bounded
//! sparse expansion as materialized weights; costly projections cache M(T) and
//! use it only after proving T is contained in the live internal-token set.
//! Otherwise the existing exact intersection/expansion path remains in charge.
use super::*;

const MAX_CACHED_SETS: usize = 16_384;
const MAX_CACHE_BYTES: usize = 8 * 1024 * 1024;
const MAX_SCANNED_ENTRIES: usize = 262_144;
// A separate transient workspace ceiling: retained cache accounting does not
// include the token-work prefix or one currently decoded weight entry vector.
const MAX_WORKSPACE_BYTES: usize = 8 * 1024 * 1024;

fn bounded_count(count: usize, remaining: usize) -> bool {
    count <= remaining && count <= MAX_SCANNED_ENTRIES
}

#[derive(Debug, Clone)]
enum ProjectedMask {
    Dense(Box<[u32]>),
    Sparse(Box<[(u16, u32)]>),
}

#[derive(Debug, Clone)]
enum CachedFinalMask {
    DirectSparse,
    Projected {
        internal: Box<[u64]>,
        output: ProjectedMask,
    },
}

/// Immutable index over the sorted TSID intervals of one wire weight. The
/// builder checks that every declared interval was decoded before publishing
/// an index; no expanded per-TSID row or token set is reconstructed. Holding
/// the pool explicitly keeps token-set IDs scoped.
#[derive(Debug)]
pub(crate) struct PackedWeightIndex {
    pool: Arc<crate::ds::weight::PackedRuntimeWeightPool>,
    entries: Box<[(u32, u32, u32)]>,
}
impl PackedWeightIndex {
    #[inline]
    pub(crate) fn is_full(&self) -> bool {
        // Full weights stay in the original pool; they have no finite index.
        false
    }
    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    #[inline]
    pub(crate) fn token_set_for_tsid(&self, tsid: u32) -> Option<PackedRuntimePoolTokenSetRef<'_>> {
        let index = self.entries.partition_point(|&(_, end, _)| end < tsid);
        let &(start, _, id) = self.entries.get(index)?;
        (start <= tsid).then(|| self.pool.token_set(id)).flatten()
    }
    pub(crate) fn for_each_entry<'a>(
        &'a self,
        mut f: impl FnMut(u32, u32, PackedRuntimePoolTokenSetRef<'a>),
    ) {
        for &(start, end, id) in &self.entries {
            if let Some(tokens) = self.pool.token_set(id) {
                f(start, end, tokens);
            }
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct PackedFinalMaskCache {
    pub(crate) indices: FxHashMap<u32, PackedWeightIndex>,
    sets: FxHashMap<u32, CachedFinalMask>,
    bytes: usize,
    scanned_entries: usize,
    scanned_ranges: usize,
    prefix_entries: usize,
}

impl Constraint {
    /// Rebuild against this constraint's current token mapping. Using make_mut
    /// detaches any shared pool metadata before publishing a mapping-specific
    /// cache; clones whose token mapping is unchanged may share it immutably.
    pub(crate) fn rebuild_packed_final_mask_cache(&mut self) {
        let Some(packed) = &self.packed_non_dwa_weights else {
            return;
        };
        let mut cache = PackedFinalMaskCache::default();
        let buf_words = self.body_mask_len();
        if self.internal_token_dense_words == 0
            || buf_words == 0
            || self.final_mask_mapping.internal_len() != 0
            || buf_words > MAX_CACHE_BYTES / 4
            || self.internal_token_buf_mask_count() >= MAX_WORKSPACE_BYTES / 8
        {
            Arc::make_mut(self.packed_non_dwa_weights.as_mut().unwrap()).final_masks =
                Arc::new(cache);
            return;
        }
        let mut weight_ids = BTreeSet::new();
        for &id in packed
            .parser_top_accept
            .values()
            .chain(packed.parser_top_accept_parts.values().flatten())
            .chain(packed.direct_regular_l1_complete_by_terminal.values())
            .take(MAX_SCANNED_ENTRIES)
        {
            weight_ids.insert(id);
            if weight_ids.len() == MAX_CACHED_SETS {
                break;
            }
        }
        let mut token_ids = BTreeSet::new();
        let mut scanned = 0usize;
        'weights: for id in weight_ids {
            if let Some(weight) = packed.pool.weight(id) {
                let count = weight.entry_count();
                if weight.is_full()
                    || !bounded_count(count, MAX_SCANNED_ENTRIES.saturating_sub(scanned))
                    || count
                        > MAX_WORKSPACE_BYTES
                            / std::mem::size_of::<((u32, u32), PackedRuntimePoolTokenSetRef<'_>)>()
                {
                    continue;
                }
                // entries() reserves its declared count. Check that count
                // above, before allocating or traversing a potentially large
                // wire weight. Partial malformed decodes cannot be indexed.
                let entries = weight.entries();
                scanned += count;
                if entries.len() != count {
                    continue;
                }
                let index_bytes = entries.len().saturating_mul(12).saturating_add(128);
                if index_bytes <= MAX_CACHE_BYTES.saturating_sub(cache.bytes) {
                    cache.indices.insert(
                        id,
                        PackedWeightIndex {
                            pool: Arc::clone(&packed.pool),
                            entries: entries
                                .iter()
                                .map(|&((a, b), tokens)| (a, b, tokens.id()))
                                .collect(),
                        },
                    );
                    cache.bytes += index_bytes;
                }
                for (_, tokens) in entries {
                    token_ids.insert(tokens.id());
                    if token_ids.len() == MAX_CACHED_SETS {
                        break 'weights;
                    }
                }
            }
        }
        let direct_limit = (buf_words / 2).min(2_048) as u64;
        let direct_enabled =
            buf_words <= u16::MAX as usize && Self::direct_sparse_weight_buf_cache_enabled();
        let prefix = self.direct_sparse_work_prefix_current(buf_words);
        let n_internal = prefix.len().saturating_sub(1);
        let mut ranges_seen = 0usize;
        for id in token_ids {
            let Some(tokens) = packed.pool.token_set(id) else {
                continue;
            };
            let count = tokens.range_count();
            if !bounded_count(count, MAX_SCANNED_ENTRIES.saturating_sub(ranges_seen)) {
                continue;
            }
            ranges_seen += count;
            let mut decoded = 0usize;
            let mut cardinality = 0u64;
            let mut work = 0u64;
            tokens.for_each_range(|a, b| {
                decoded += 1;
                cardinality = cardinality.saturating_add(b as u64 - a as u64 + 1);
                let start = (a as usize).min(n_internal);
                let end = (b as usize).saturating_add(1).min(n_internal);
                if start < end {
                    work = work
                        .saturating_add((end - start) as u64)
                        .saturating_add(prefix[end].saturating_sub(prefix[start]));
                }
            });
            if decoded != count {
                continue;
            }
            // Account conservatively for the hash table slot and allocator.
            let overhead = 128usize;
            if cache.bytes.saturating_add(overhead) > MAX_CACHE_BYTES {
                break;
            }
            if direct_enabled && cardinality <= direct_limit && work <= direct_limit {
                cache.sets.insert(id, CachedFinalMask::DirectSparse);
                cache.bytes += overhead;
                continue;
            }
            let upper = self
                .internal_token_dense_words
                .checked_mul(8)
                .and_then(|n| n.checked_add(buf_words.saturating_mul(4)))
                .and_then(|n| n.checked_add(overhead));
            if upper.is_none_or(|n| n > MAX_CACHE_BYTES.saturating_sub(cache.bytes)) {
                continue;
            }
            // This cache owns its membership rows. Fill their final allocation
            // directly instead of constructing an Arc and immediately copying
            // it into a second Box allocation.
            let mut internal = vec![0u64; self.internal_token_dense_words];
            Self::fill_dense_words_from_runtime_token_set(
                &mut internal,
                RuntimeTokenSetRef::PackedPool(tokens),
            );
            let mut output = vec![0u32; buf_words];
            self.or_internal_dense_to_buf(&internal, &mut output, true);
            let output = if buf_words <= u16::MAX as usize {
                let sparse = Self::dense_buf_to_sparse_entries(&output);
                if sparse.len().saturating_mul(2) < output.len() {
                    ProjectedMask::Sparse(sparse)
                } else {
                    ProjectedMask::Dense(output.into_boxed_slice())
                }
            } else {
                ProjectedMask::Dense(output.into_boxed_slice())
            };
            let output_bytes = match &output {
                ProjectedMask::Dense(x) => x.len() * 4,
                ProjectedMask::Sparse(x) => x.len() * 8,
            };
            cache.bytes += internal.len() * 8 + output_bytes + overhead;
            cache.sets.insert(
                id,
                CachedFinalMask::Projected {
                    internal: internal.into_boxed_slice(),
                    output,
                },
            );
        }
        cache.scanned_entries = scanned;
        cache.scanned_ranges = ranges_seen;
        cache.prefix_entries = prefix.len();
        debug_assert!(cache.bytes <= MAX_CACHE_BYTES);
        if std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some() {
            eprintln!(
                "[glrmask/profile][packed_final_masks] sets={} indexed_weights={} accounted_bytes={} ranges={}",
                cache.sets.len(),
                cache.indices.len(),
                cache.bytes,
                ranges_seen
            );
        }
        Arc::make_mut(self.packed_non_dwa_weights.as_mut().unwrap()).final_masks = Arc::new(cache);
    }

    /// `tokens` comes from this constraint's packed non-DWA pool. Returning
    /// true means that the complete contribution has been handled, including
    /// an empty intersection. A cache miss is never an admissibility verdict.
    #[inline]
    pub(crate) fn or_packed_final_tokens_to_buf(
        &self,
        dense: &[u64],
        tokens: PackedRuntimePoolTokenSetRef<'_>,
        buf: &mut [u32],
    ) -> bool {
        if self.final_mask_mapping.internal_len() != 0 {
            return false;
        }
        let Some(pool) = &self.packed_non_dwa_weights else {
            return false;
        };
        let Some(entry) = pool.final_masks.sets.get(&tokens.id()) else {
            return false;
        };
        match entry {
            CachedFinalMask::DirectSparse => self
                .or_runtime_token_set_to_buf_sparse(
                    dense,
                    RuntimeTokenSetRef::PackedPool(tokens),
                    2_048,
                    buf,
                )
                .is_some(),
            CachedFinalMask::Projected { internal, output } => {
                if internal
                    .iter()
                    .enumerate()
                    .any(|(i, &word)| word & !dense.get(i).copied().unwrap_or(0) != 0)
                {
                    return false;
                }
                match output {
                    ProjectedMask::Dense(mask) => or_dense_buf(buf, mask),
                    ProjectedMask::Sparse(mask) => {
                        // All public mask entry points already require the
                        // full word count; retain safety for private callers.
                        for &(word, bits) in mask.iter() {
                            if let Some(slot) = buf.get_mut(word as usize) {
                                *slot |= bits;
                            }
                        }
                    }
                }
                true
            }
        }
    }
}

#[cfg(test)]
mod budget_tests {
    use super::*;

    #[test]
    fn cache_counts_are_checked_before_decoding() {
        assert!(bounded_count(0, 0));
        assert!(bounded_count(MAX_SCANNED_ENTRIES, MAX_SCANNED_ENTRIES));
        assert!(!bounded_count(MAX_SCANNED_ENTRIES + 1, usize::MAX));
        assert!(!bounded_count(1, 0));
        assert!(!bounded_count(u32::MAX as usize, MAX_SCANNED_ENTRIES));
        assert!(!bounded_count(usize::MAX, usize::MAX));
    }
}

/// Exhaustive interval-boundary checks and randomized internal-set projection
/// checks against the wire decoder and original-token fragments, independently
/// of the cache. Used by crate tests and isolated internal release gates only.
#[cfg(any(test, feature = "internal-api"))]
impl Constraint {
    pub(crate) fn check_packed_final_cache_against_uncached(&self, rounds: usize) -> usize {
        let constraint = self;
        let Some(pool) = &constraint.packed_non_dwa_weights else {
            return 0;
        };
        let mut checked = 0usize;
        let ranges = |set: Option<PackedRuntimePoolTokenSetRef<'_>>| {
            let mut result = Vec::new();
            if let Some(set) = set {
                set.for_each_range(|a, b| result.push((a, b)));
            }
            result
        };
        for (&id, index) in &pool.final_masks.indices {
            let wire = pool.pool.weight(id).unwrap();
            assert_eq!(index.is_full(), wire.is_full());
            assert_eq!(index.is_empty(), wire.is_empty());
            let mut queries = BTreeSet::from([0u32, u32::MAX]);
            for &(start, end, _) in &index.entries {
                queries.extend([start.saturating_sub(1), start, end, end.saturating_add(1)]);
            }
            for q in queries {
                assert_eq!(
                    ranges(index.token_set_for_tsid(q)),
                    ranges(wire.token_set_for_tsid(q)),
                    "indexed pool id {id} TSID {q}"
                );
                checked += 1;
            }
        }
        let mut random = 0x693ed391275f192du64;
        let n = constraint.internal_token_count();
        for &id in pool.final_masks.sets.keys() {
            let tokens = pool.pool.token_set(id).unwrap();
            for round in 0..rounds {
                let mut dense = vec![0u64; constraint.internal_token_dense_words];
                for word in &mut dense {
                    random ^= random << 13;
                    random ^= random >> 7;
                    random ^= random << 17;
                    *word = if round == 0 {
                        u64::MAX
                    } else if round == 1 {
                        0
                    } else {
                        random
                    };
                }
                let mut got =
                    vec![if round % 3 == 0 { 0xaaaaaaaa } else { 0 }; constraint.body_mask_len()];
                let mut expected = got.clone();
                if constraint.or_packed_final_tokens_to_buf(&dense, tokens, &mut got) {
                    tokens.for_each_range(|a, b| {
                        let end = (b as usize).min(n.saturating_sub(1));
                        if n == 0 || a as usize > end {
                            return;
                        }
                        for token in a as usize..=end {
                            if dense
                                .get(token / 64)
                                .is_some_and(|word| word & (1u64 << (token % 64)) != 0)
                            {
                                constraint.for_each_internal_token_buf_mask_entry(
                                    token,
                                    |word, bits| {
                                        expected[word as usize] |= bits;
                                    },
                                );
                            }
                        }
                    });
                    assert_eq!(got, expected, "cached final pool id {id}, round {round}");
                    checked += 1;
                } else {
                    assert_eq!(
                        got, expected,
                        "a declined cached projection must not mutate output"
                    );
                }
            }
        }
        assert!(pool.final_masks.bytes <= MAX_CACHE_BYTES);
        assert!(pool.final_masks.sets.len() <= MAX_CACHED_SETS);
        assert!(pool.final_masks.indices.len() <= MAX_CACHED_SETS);
        assert!(pool.final_masks.scanned_entries <= MAX_SCANNED_ENTRIES);
        assert!(pool.final_masks.scanned_ranges <= MAX_SCANNED_ENTRIES);
        assert!(pool.final_masks.prefix_entries <= MAX_WORKSPACE_BYTES / 8);
        checked
    }
}
