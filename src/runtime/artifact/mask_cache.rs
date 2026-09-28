//! Runtime mask and subset caches; sharing never proves state equality.

use super::{
    DirectRegularDynamicFrontierCacheEntry, DirectRegularTerminalSupport, DynamicMaskStateKey,
    DynamicMaskVocab,
};
use crate::ds::bitset::BitSet;
use crate::grammar::flat::TerminalID;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::sync::Arc;
#[derive(Debug)]
pub(super) struct DynamicMaskCacheEntry {
    pub(super) hash: u64,
    pub(super) state: DynamicMaskStateKey,
    pub(super) mask: DynamicMaskCachePayload,
}

#[derive(Debug)]
pub(super) enum DynamicMaskCachePayload {
    /// Exact key observed once, but no mask payload stored yet. A second miss
    /// for the same key upgrades this entry to a real payload. This avoids
    /// paying mask-storage cost for cheap one-off states while preserving
    /// reuse for cheap states that actually recur.
    Probation,
    Dense(Arc<[u32]>),
    SparseZero(Box<[(u32, u32)]>),
    SparseAllOriginal(Box<[(u32, u32)]>),
}

#[derive(Debug, Default)]
pub(super) struct DynamicMaskCache {
    entry_cap: Option<usize>,
    pub(super) entries: Vec<Option<DynamicMaskCacheEntry>>,
    pub(super) by_hash: FxHashMap<u64, SmallVec<[usize; 1]>>,
    pub(super) next_slot: usize,
}

/// Runtime-only lazily determinized subset-state cache for scalar-dispatch mask execution.
/// Canonical subset states are shared across mask calls within one constraint runtime; this
/// is derived acceleration state, never serialized, and reset by
/// `fresh_runtime_instance`.
#[derive(Debug)]
pub(crate) struct DynamicLazyUnionMetadata {
    pub(crate) finalizer_code: u32,
    pub(crate) single_finalizer_continues: u8,
    pub(crate) matched: BitSet,
    pub(crate) futures: BitSet,
}

#[derive(Debug)]
pub(crate) enum DynamicLazyUnionRow {
    Sparse(SmallVec<[(u8, u32); 8]>),
    Dense(Box<[u32; 256]>),
}

impl Default for DynamicLazyUnionRow {
    fn default() -> Self {
        Self::Sparse(SmallVec::new())
    }
}

#[derive(Debug, Default)]
pub(crate) struct DynamicLazyUnionCache {
    pub(crate) base_state_count: u32,
    /// Sparse-on-demand physical scalar-dispatch transition rows. Each row is
    /// allocated only after the first touched byte, and individual cells stay
    /// UNBUILT until requested by the vocabulary walk. This retains O(1)
    /// repeated probes without paying to enumerate all 256 bytes of a newly
    /// encountered physical state.
    pub(crate) base_rows: Vec<Option<Box<[u32; 256]>>>,
    /// Dedicated packed-key index for the overwhelmingly common two-state
    /// derivatives.  Avoids hashing/cloning a SmallVec on every pair lookup.
    pub(crate) state_by_pair: Option<Box<FxHashMap<u64, u32>>>,
    /// Exact union of two *input coordinates*, including virtual subsets.
    /// Distinct from state_by_pair, whose keys contain physical members only.
    /// Bounded, runtime-only, and cleared with every extension-ID reset.
    pub(crate) state_by_union_pair: Option<Box<FxHashMap<u64, u32>>>,
    pub(crate) state_by_subset: FxHashMap<SmallVec<[u32; 8]>, u32>,
    pub(crate) subsets: Vec<SmallVec<[u32; 8]>>,
    /// Lazy virtual-state derivatives. Most derived states see only a few
    /// vocabulary bytes, so keep them sparse and promote only genuinely hot
    /// rows to a dense 256-cell table.
    pub(crate) rows: Vec<DynamicLazyUnionRow>,
    pub(crate) metadata: Vec<Option<DynamicLazyUnionMetadata>>,
    /// Learned output-polarity hint for exact lazy-union/physical root states.
    /// 1 = observed non-dense, 2 = observed lexically dense. Runtime-only and
    /// cleared whenever the lazy-union coordinate space is reset.
    pub(crate) dense_output_hints: FxHashMap<u32, u8>,
}

/// Runtime-only exact deterministic extension for a parser-filtered union of
/// Flat16 mask-tokenizer states. IDs in `rows` start at `base_state_count`;
/// this object is derived, never serialized, and may be reused for every mask
/// whose canonical lexer-root subset matches its cache key.
#[derive(Debug)]
pub(crate) struct DynamicDenseSubset16 {
    pub(crate) root_state: u32,
    pub(crate) base_state_count: u32,
    pub(crate) rows: Vec<Box<[u32; 256]>>,
    pub(crate) finalizer_code: Vec<u32>,
    pub(crate) single_finalizer_continues: Vec<u8>,
    pub(crate) matched: Vec<BitSet>,
    pub(crate) futures: Vec<BitSet>,
    pub(crate) subsets: Vec<SmallVec<[u32; 8]>>,
}

impl DynamicMaskVocab {
    pub(crate) fn set_direct_regular_terminal_support(
        &mut self,
        support: DirectRegularTerminalSupport,
    ) {
        self.direct_regular_terminal_support = Arc::new(support);
    }

    pub(crate) fn direct_regular_terminal_support(&self) -> &DirectRegularTerminalSupport {
        self.direct_regular_terminal_support.as_ref()
    }

    pub(crate) fn cached_direct_regular_wide_frontier_index(&self, key: usize) -> Option<usize> {
        self.direct_regular_wide_frontier_index_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&key)
            .copied()
    }

    pub(crate) fn cache_direct_regular_wide_frontier_index(&self, key: usize, index: usize) {
        self.direct_regular_wide_frontier_index_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(key, index);
    }

    pub(crate) fn cached_pending_guard_blocked_mask(
        &self,
        memories: &[(u32, TerminalID)],
    ) -> Option<Arc<Vec<u32>>> {
        self.pending_guard_blocked_mask_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(memories)
            .cloned()
    }

    pub(crate) fn cache_pending_guard_blocked_mask(
        &self,
        memories: &[(u32, TerminalID)],
        mask: Vec<u32>,
    ) -> Arc<Vec<u32>> {
        // Bound retained derived data independently of corpus behavior. A
        // Llama-sized mask is about 16 KiB, so 256 entries cap this cache at
        // roughly 4 MiB plus map/key overhead.
        const MAX_PENDING_GUARD_MASK_CACHE_ENTRIES: usize = 256;
        let mut cache = self
            .pending_guard_blocked_mask_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(existing) = cache.get(memories) {
            return Arc::clone(existing);
        }
        if cache.len() >= MAX_PENDING_GUARD_MASK_CACHE_ENTRIES {
            cache.clear();
        }
        let mask = Arc::new(mask);
        cache.insert(memories.to_vec(), Arc::clone(&mask));
        mask
    }

    pub(crate) fn cached_residual_slice_contained(
        &self,
        terminal: TerminalID,
        source: u32,
        slice_cache_id: u32,
    ) -> Option<bool> {
        let key = (terminal, source, 0xA000_0000u32 | slice_cache_id);
        self.projected_terminal_partition_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&key)
            .copied()
    }

    pub(crate) fn cache_residual_slice_contained(
        &self,
        terminal: TerminalID,
        source: u32,
        slice_cache_id: u32,
        contained: bool,
    ) {
        let key = (terminal, source, 0xA000_0000u32 | slice_cache_id);
        self.projected_terminal_partition_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(key, contained);
    }

    pub(crate) fn cached_direct_slice_contained(
        &self,
        terminal: TerminalID,
        mask_state: u32,
        slice_cache_id: u32,
    ) -> Option<bool> {
        let key = (terminal, mask_state, 0xC000_0000u32 | slice_cache_id);
        self.projected_terminal_partition_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&key)
            .copied()
    }

    pub(crate) fn cache_direct_slice_contained(
        &self,
        terminal: TerminalID,
        mask_state: u32,
        slice_cache_id: u32,
        contained: bool,
    ) {
        let key = (terminal, mask_state, 0xC000_0000u32 | slice_cache_id);
        self.projected_terminal_partition_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(key, contained);
    }

    pub(crate) fn cached_direct_regular_frontier(
        &self,
        key: usize,
    ) -> Option<DirectRegularDynamicFrontierCacheEntry> {
        self.direct_regular_frontier_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&key)
            .cloned()
    }

    pub(crate) fn cache_direct_regular_frontier(
        &self,
        key: usize,
        entry: DirectRegularDynamicFrontierCacheEntry,
    ) -> DirectRegularDynamicFrontierCacheEntry {
        const MAX_FRONTIER_CACHE_ENTRIES: usize = 1024;
        let mut cache = self
            .direct_regular_frontier_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(existing) = cache.get(&key) {
            return existing.clone();
        }
        if cache.len() >= MAX_FRONTIER_CACHE_ENTRIES {
            // Cache entries retain their source GSS interface, making pointer
            // keys safe. Clearing atomically drops both keys and retained
            // interfaces before any allocator reuse can produce a new key.
            cache.clear();
        }
        cache.insert(key, entry.clone());
        entry
    }

    pub(crate) fn lock_lazy_union_cache(&self) -> std::sync::MutexGuard<'_, DynamicLazyUnionCache> {
        self.lazy_union_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub(crate) fn try_lock_lazy_union_cache(
        &self,
    ) -> Option<std::sync::MutexGuard<'_, DynamicLazyUnionCache>> {
        match self.lazy_union_cache.try_lock() {
            Ok(cache) => Some(cache),
            Err(std::sync::TryLockError::Poisoned(poisoned)) => Some(poisoned.into_inner()),
            Err(std::sync::TryLockError::WouldBlock) => None,
        }
    }

    pub(crate) fn cached_dense_subset16(
        &self,
        root_states: &[u32],
    ) -> Option<Arc<DynamicDenseSubset16>> {
        self.dense_subset16_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(root_states)
            .cloned()
    }

    pub(crate) fn cached_dense_subset16_state_for_subset(
        &self,
        root_states: &[u32],
    ) -> Option<(Arc<DynamicDenseSubset16>, u32)> {
        let cache = self
            .dense_subset16_cache
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        for extension in cache.values() {
            if let Some(index) = extension
                .subsets
                .iter()
                .position(|subset| subset.as_slice() == root_states)
            {
                return Some((
                    Arc::clone(extension),
                    extension.base_state_count + index as u32,
                ));
            }
        }
        None
    }

    pub(crate) fn cache_dense_subset16(
        &self,
        root_states: Vec<u32>,
        extension: DynamicDenseSubset16,
    ) -> Arc<DynamicDenseSubset16> {
        const MAX_DENSE_SUBSET16_CACHE_ENTRIES: usize = 256;
        let mut cache = self
            .dense_subset16_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(existing) = cache.get(root_states.as_slice()) {
            return Arc::clone(existing);
        }
        if cache.len() >= MAX_DENSE_SUBSET16_CACHE_ENTRIES {
            cache.clear();
        }
        let extension = Arc::new(extension);
        cache.insert(root_states, Arc::clone(&extension));
        extension
    }

    pub(crate) fn copy_cached_mask(
        &self,
        state: &DynamicMaskStateKey,
        hash: u64,
        buf: &mut [u32],
    ) -> bool {
        self.copy_cached_mask_with_predicate(hash, |candidate| candidate == state, buf)
    }

    pub(crate) fn copy_cached_mask_with_predicate<F: Fn(&DynamicMaskStateKey) -> bool>(
        &self,
        hash: u64,
        matches: F,
        buf: &mut [u32],
    ) -> bool {
        let cache = self
            .mask_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(slots) = cache.by_hash.get(&hash) else {
            return false;
        };
        let Some(entry) = slots.iter().rev().find_map(|&slot| {
            cache
                .entries
                .get(slot)
                .and_then(Option::as_ref)
                .filter(|entry| matches(&entry.state))
        }) else {
            return false;
        };
        Self::copy_dynamic_mask_cache_payload(self.all_original_token_words(), &entry.mask, buf)
    }

    pub(crate) fn has_cached_mask_with_predicate<F: Fn(&DynamicMaskStateKey) -> bool>(
        &self,
        hash: u64,
        matches: F,
    ) -> bool {
        let cache = self
            .mask_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(slots) = cache.by_hash.get(&hash) else {
            return false;
        };
        slots.iter().rev().any(|&slot| {
            cache
                .entries
                .get(slot)
                .and_then(Option::as_ref)
                .is_some_and(|entry| {
                    matches(&entry.state)
                        && !matches!(entry.mask, DynamicMaskCachePayload::Probation)
                })
        })
    }

    pub(super) fn copy_dynamic_mask_cache_payload(
        baseline: &[u32],
        payload: &DynamicMaskCachePayload,
        buf: &mut [u32],
    ) -> bool {
        match payload {
            DynamicMaskCachePayload::Probation => return false,
            DynamicMaskCachePayload::Dense(mask) => {
                if mask.len() != buf.len() {
                    return false;
                }
                buf.copy_from_slice(mask);
            }
            DynamicMaskCachePayload::SparseZero(words) => {
                buf.fill(0);
                for &(word, value) in words.iter() {
                    let Some(dst) = buf.get_mut(word as usize) else {
                        return false;
                    };
                    *dst = value;
                }
            }
            DynamicMaskCachePayload::SparseAllOriginal(words) => {
                let copy_len = buf.len().min(baseline.len());
                buf[..copy_len].copy_from_slice(&baseline[..copy_len]);
                if copy_len < buf.len() {
                    buf[copy_len..].fill(0);
                }
                for &(word, value) in words.iter() {
                    let Some(dst) = buf.get_mut(word as usize) else {
                        return false;
                    };
                    *dst = value;
                }
            }
        }
        true
    }

    pub(super) fn dynamic_mask_cache_payload(&self, mask: &[u32]) -> DynamicMaskCachePayload {
        // G: the entry budget already charges every payload its full dense
        // size. Direct storage therefore stays within that existing payload
        // bound, at the cost of giving up sparse working-set compression.
        // Avoid the two classification scans plus sparse materialization scan.
        // Exact word-for-word copy; key matching, eviction, and admission are
        // unchanged. Experiment only until representative measurements pass.
        static DIRECT_COPY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *DIRECT_COPY.get_or_init(|| {
            std::env::var_os("GLRMASK_EXPERIMENT_DYNAMIC_MASK_CACHE_DENSE_ONLY").is_some()
        }) {
            return DynamicMaskCachePayload::Dense(Arc::from(mask));
        }
        let baseline = self.all_original_token_words();
        let nonzero_count = mask.iter().filter(|&&word| word != 0).count();
        let baseline_diff_count = mask
            .iter()
            .enumerate()
            .filter(|&(index, &word)| word != baseline.get(index).copied().unwrap_or(0))
            .count();
        let dense_bytes = mask.len().saturating_mul(std::mem::size_of::<u32>());
        let sparse_zero_bytes = nonzero_count.saturating_mul(std::mem::size_of::<(u32, u32)>());
        let sparse_baseline_bytes =
            baseline_diff_count.saturating_mul(std::mem::size_of::<(u32, u32)>());
        if sparse_zero_bytes < dense_bytes && sparse_zero_bytes <= sparse_baseline_bytes {
            DynamicMaskCachePayload::SparseZero(
                mask.iter()
                    .enumerate()
                    .filter_map(|(index, &word)| (word != 0).then_some((index as u32, word)))
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            )
        } else if sparse_baseline_bytes < dense_bytes {
            DynamicMaskCachePayload::SparseAllOriginal(
                mask.iter()
                    .enumerate()
                    .filter_map(|(index, &word)| {
                        (word != baseline.get(index).copied().unwrap_or(0))
                            .then_some((index as u32, word))
                    })
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            )
        } else {
            DynamicMaskCachePayload::Dense(Arc::from(mask))
        }
    }

    pub(crate) fn cache_mask(
        &self,
        state: DynamicMaskStateKey,
        hash: u64,
        mask: &[u32],
        probation_if_absent: bool,
    ) {
        // Keep enough exact states to cover an ordinary generated sequence.
        // A fixed 64-entry limit caused long source-specialized sequences to
        // evict their expensive early masks during the warmup pass, so every
        // measured pass recomputed them. Bound by bytes instead: Llama-sized
        // masks retain about 2k states in 32 MiB, while tiny vocabularies may
        // retain more without material memory cost.
        // Opt-in bounded capacity experiment; default and key semantics unchanged.
        static CACHE_BUDGET_MIB: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        let budget_mib = *CACHE_BUDGET_MIB.get_or_init(|| {
            std::env::var("GLRMASK_EXPERIMENT_DYNAMIC_MASK_CACHE_BUDGET_MIB")
                .ok().and_then(|s| s.parse::<usize>().ok())
                .filter(|n| (1..=64).contains(n)).unwrap_or(32)
        });
        const MIN_MASK_CACHE_ENTRIES: usize = 64;
        const MAX_MASK_CACHE_ENTRIES: usize = 4096;
        let mask_bytes = mask.len().saturating_mul(std::mem::size_of::<u32>()).max(1);
        let max_entries = ((budget_mib * 1024 * 1024) / mask_bytes)
            .clamp(MIN_MASK_CACHE_ENTRIES, MAX_MASK_CACHE_ENTRIES);
        let mut cache = self
            .mask_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let max_entries = max_entries.min(cache.entry_cap.unwrap_or(usize::MAX));
        if let Some(slots) = cache.by_hash.get(&hash).cloned() {
            for slot in slots {
                let matches = cache
                    .entries
                    .get(slot)
                    .and_then(Option::as_ref)
                    .is_some_and(|entry| entry.state == state);
                if !matches {
                    continue;
                }
                let needs_upgrade = cache.entries[slot]
                    .as_ref()
                    .is_some_and(|entry| matches!(entry.mask, DynamicMaskCachePayload::Probation));
                if needs_upgrade {
                    let payload = self.dynamic_mask_cache_payload(mask);
                    cache.entries[slot]
                        .as_mut()
                        .expect("probation cache slot disappeared")
                        .mask = payload;
                }
                return;
            }
        }
        let payload = if probation_if_absent {
            DynamicMaskCachePayload::Probation
        } else {
            self.dynamic_mask_cache_payload(mask)
        };
        let entry = DynamicMaskCacheEntry {
            hash,
            state,
            mask: payload,
        };
        let slot = if cache.entries.len() < max_entries {
            let slot = cache.entries.len();
            cache.entries.push(Some(entry));
            slot
        } else {
            let slot = cache.next_slot % max_entries;
            cache.next_slot = (slot + 1) % max_entries;
            if let Some(previous) = cache.entries[slot].take() {
                let mut remove_hash = false;
                if let Some(slots) = cache.by_hash.get_mut(&previous.hash) {
                    if let Some(index) = slots.iter().position(|&candidate| candidate == slot) {
                        slots.swap_remove(index);
                    }
                    remove_hash = slots.is_empty();
                }
                if remove_hash {
                    cache.by_hash.remove(&previous.hash);
                }
            }
            cache.entries[slot] = Some(entry);
            slot
        };
        cache.by_hash.entry(hash).or_default().push(slot);
        if cache.entries.len() == max_entries && cache.next_slot >= max_entries {
            cache.next_slot = 0;
        }
    }
}


impl DynamicMaskVocab {

    /// Limit one immutable vocabulary's result memo independently of the
    /// ordinary full-vocabulary cache budget. Set only before publishing it.
    pub(crate) fn with_mask_cache_entry_cap(self, cap: usize) -> Self {
        {
            let mut cache = self.mask_cache.lock().unwrap_or_else(|e| e.into_inner());
            assert!(cache.entries.is_empty());
            cache.entry_cap = Some(cap.clamp(1, 4096));
        }
        self
    }
}
