use super::*;
use crate::runtime::artifact::{DynamicLazyUnionRow, DynamicMaskTrieFullWalkOp};
use rustc_hash::FxHashSet;

#[inline(always)]
fn canonicalize_lazy_targets(targets: &mut SmallVec<[u32; 8]>) {
    if targets.windows(2).any(|pair| pair[0] > pair[1]) {
        targets.sort_unstable();
    }
    targets.dedup();
}

#[inline]
fn physical_sole_live_terminal(tokenizer: &Tokenizer, state: u32) -> Option<TerminalID> {
    let mut only = None::<TerminalID>;
    for terminal in tokenizer
        .matched_terminals_slice(state)
        .iter()
        .copied()
        .chain(
            tokenizer
                .possible_future_terminals(state)
                .iter_ones()
                .map(|terminal| terminal as TerminalID),
        )
    {
        match only {
            None => only = Some(terminal),
            Some(existing) if existing == terminal => {}
            Some(_) => return None,
        }
    }
    only
}

#[inline]
fn bitset_sole_live_terminal(matched: &BitSet, futures: &BitSet) -> Option<TerminalID> {
    let mut only = None::<TerminalID>;
    for terminal in matched
        .iter_ones()
        .chain(futures.iter_ones())
        .map(|terminal| terminal as TerminalID)
    {
        match only {
            None => only = Some(terminal),
            Some(existing) if existing == terminal => {}
            Some(_) => return None,
        }
    }
    only
}

trait FullWalkTransitionTable {
    type Cell: Copy;

    fn cell(&self, state: u32, byte: u8) -> Self::Cell;

    fn cell_is_dead(cell: Self::Cell) -> bool;

    fn cell_has_finalizer(cell: Self::Cell) -> bool;

    fn cell_target(cell: Self::Cell) -> u32;

    /// Exact self-transition classes of an already materialized table row.
    /// Lazy tables override this: proving an alphabet must not create a large
    /// collection of speculative derivative states for unused vocabulary bytes.
    fn proven_identity_byte_classes(
        &self,
        _tokenizer: &Tokenizer,
        state: u32,
    ) -> Option<FullWalkIdentityByteClasses> {
        Some(full_walk_identity_byte_classes(state, |byte| {
            let cell = self.cell(state, byte);
            (!Self::cell_is_dead(cell))
                .then(|| (Self::cell_target(cell), Self::cell_has_finalizer(cell)))
        }))
    }

    /// Exact alternatives sharing one parser coordinate. Ordinary rows are
    /// atomic. A lazy union may expose its bounded physical member set, but
    /// never derivative approximations or members from another table.
    #[inline]
    fn continuation_witness_components(&self, state: u32) -> SmallVec<[u32; 8]> {
        smallvec::smallvec![state]
    }

    #[inline(always)]
    fn transition(&self, state: u32, byte: u8) -> u32 {
        let cell = self.cell(state, byte);
        if Self::cell_is_dead(cell) {
            u32::MAX
        } else {
            Self::cell_target(cell)
        }
    }

    #[inline(always)]
    fn state_count(&self, tokenizer: &Tokenizer) -> usize {
        tokenizer.num_states() as usize
    }

    #[inline(always)]
    fn finalizer_code(&self, state: u32, base: &[u32]) -> u32 {
        unsafe { *base.get_unchecked(state as usize) }
    }

    #[inline(always)]
    fn single_finalizer_continues(&self, state: u32, base: &[u8]) -> bool {
        unsafe { *base.get_unchecked(state as usize) != 0 }
    }

    #[inline]
    fn matched_terminals(&self, tokenizer: &Tokenizer, state: u32) -> SmallVec<[TerminalID; 4]> {
        tokenizer.matched_terminals_slice(state).iter().copied().collect()
    }


    #[inline(always)]
    fn future_contains(&self, tokenizer: &Tokenizer, state: u32, terminal: TerminalID) -> bool {
        tokenizer.possible_future_terminals(state).contains(terminal as usize)
    }

    #[inline(always)]
    fn future_intersects(&self, tokenizer: &Tokenizer, state: u32, terminals: &BitSet) -> bool {
        !terminals.is_disjoint(tokenizer.possible_future_terminals(state))
    }

    /// Return the sole matched/future terminal at this exact walk coordinate.
    /// `None` means either no terminal is live or more than one distinct
    /// terminal is live.  Synthetic subset/union coordinates override this to
    /// read their already-materialized exact terminal metadata.
    #[inline]
    fn sole_live_terminal(&self, tokenizer: &Tokenizer, state: u32) -> Option<TerminalID> {
        physical_sole_live_terminal(tokenizer, state)
    }

    #[inline]
    fn union_states(&self, _states: &[u32]) -> Option<u32> {
        None
    }

    #[inline(always)]
    fn dense_output_hint(&self, _state: u32) -> Option<bool> {
        None
    }

    #[inline(always)]
    fn cache_dense_output_hint(&self, _state: u32, _dense: bool) {}
}

#[derive(Clone, Copy)]
struct FullWalkFlat16<'a> {
    transitions: &'a [u16],
}

impl FullWalkTransitionTable for FullWalkFlat16<'_> {
    type Cell = u16;

    #[inline(always)]
    fn cell(&self, state: u32, byte: u8) -> u16 {
        unsafe {
            *self
                .transitions
                .get_unchecked((state as usize).wrapping_mul(256) + byte as usize)
        }
    }

    #[inline(always)]
    fn cell_is_dead(cell: u16) -> bool {
        cell == u16::MAX
    }

    #[inline(always)]
    fn cell_has_finalizer(cell: u16) -> bool {
        cell & 0x8000 != 0
    }

    #[inline(always)]
    fn cell_target(cell: u16) -> u32 {
        u32::from(cell & 0x7fff)
    }

}


struct FullWalkLazyUnion<'a> {
    base_transitions16: Option<&'a [u16]>,
    base_transitions32: Option<&'a [u32]>,
    base_state_count: u32,
    tokenizer: *const Tokenizer,
    overflowed: &'a std::cell::Cell<bool>,
    use_pair_map: bool,
    use_union_pair_memo: bool,
    cache: std::cell::UnsafeCell<std::sync::MutexGuard<'a, DynamicLazyUnionCache>>,
}

impl<'a> FullWalkLazyUnion<'a> {
    const UNBUILT: u32 = u32::MAX - 1;
    const SOFT_MAX_EXTENSION_STATES: usize = 4096;
    const RESERVED_EXTENSION_STATES: usize = 8192;
    const MAX_UNION_PAIR_MEMO: usize = 2048;

    fn new(
        tokenizer: &Tokenizer,
        base_transitions16: Option<&'a [u16]>,
        base_transitions32: Option<&'a [u32]>,
        mut cache: std::sync::MutexGuard<'a, DynamicLazyUnionCache>,
        root_states: &[u32],
        overflowed: &'a std::cell::Cell<bool>,
        use_pair_map: bool,
    ) -> Option<(Self, u32)> {
        if root_states.len() < 2 {
            return None;
        }
        let base_state_count = tokenizer.num_states();
        if base_state_count >= 0x8000_0000u32.saturating_sub(Self::RESERVED_EXTENSION_STATES as u32) {
            return None;
        }
        if cache.base_state_count == 0 {
            cache.base_state_count = base_state_count;
        } else if cache.base_state_count != base_state_count {
            Self::clear_cache(&mut cache);
            cache.base_state_count = base_state_count;
        }
        if cache.subsets.len() >= Self::SOFT_MAX_EXTENSION_STATES {
            // `clear_cache()` also drops the sparse physical transition rows.
            // Perform the soft-limit reset before sizing `base_rows`; doing
            // this in the opposite order leaves `base_rows` empty and makes
            // the unchecked hot-path lookup in `base_cell()` immediately
            // out-of-bounds on the next physical transition.
            Self::clear_cache(&mut cache);
            cache.base_state_count = base_state_count;
        }
        if cache.base_rows.len() != base_state_count as usize {
            cache
                .base_rows
                .resize_with(base_state_count as usize, || None);
        }
        static UNION_PAIR_MEMO: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let use_union_pair_memo = full_walk_acceleration_enabled() && *UNION_PAIR_MEMO.get_or_init(|| {
            !std::env::var("GLRMASK_DISABLE_LAZY_UNION_PAIR_MEMO").ok().is_some_and(|value| {
                matches!(value.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on")
            })
        });
        let table = Self {
            base_transitions16,
            base_transitions32,
            base_state_count,
            tokenizer: tokenizer as *const Tokenizer,
            overflowed,
            use_pair_map,
            use_union_pair_memo,
            cache: std::cell::UnsafeCell::new(cache),
        };
        let root = table.intern_states(root_states)?;
        Some((table, root))
    }

    #[inline]
    fn clear_cache(cache: &mut DynamicLazyUnionCache) {
        cache.base_rows.clear();
        if let Some(pair_map) = cache.state_by_pair.as_mut() {
            pair_map.clear();
        }
        if let Some(pair_map) = cache.state_by_union_pair.as_mut() {
            pair_map.clear();
        }
        cache.state_by_subset.clear();
        cache.subsets.clear();
        cache.rows.clear();
        cache.metadata.clear();
        cache.dense_output_hints.clear();
    }

    fn intern_states(&self, states: &[u32]) -> Option<u32> {
        // Cached physical subsets already have canonical IDs, but discovering
        // that ID used to re-expand and sort both inputs on every call. Exact
        // input-coordinate pairs are stable until clear_cache resets the whole
        // namespace, at which point this memo is cleared too.
        let pair_key = if self.use_union_pair_memo {
            match states {
                [first, second] => Some((u64::from((*first).min(*second)) << 32) | u64::from((*first).max(*second))),
                _ => None,
            }
        } else { None };
        if let Some(key) = pair_key {
            let cache = unsafe { &*self.cache.get() };
            if let Some(result) = cache.state_by_union_pair.as_ref().and_then(|memo| memo.get(&key)) {
                return Some(*result);
            }
        }
        let mut physical = SmallVec::<[u32; 8]>::new();
        {
            let cache = unsafe { &*self.cache.get() };
            for &state in states {
                if state < self.base_state_count {
                    physical.push(state);
                } else {
                    let subset = cache.subsets.get((state - self.base_state_count) as usize)?;
                    physical.extend_from_slice(subset);
                }
            }
        }
        let cache = unsafe { &mut *self.cache.get() };
        let result = Self::intern_physical_inner(self.base_state_count, cache, physical, self.use_pair_map);
        if let (Some(key), Some(result)) = (pair_key, result) {
            let memo = cache.state_by_union_pair.get_or_insert_with(|| Box::new(FxHashMap::default()));
            if memo.len() < Self::MAX_UNION_PAIR_MEMO {
                memo.insert(key, result);
            }
        }
        if result.is_none() {
            self.overflowed.set(true);
        }
        result
    }

    fn intern_physical_inner(
        base_state_count: u32,
        cache: &mut DynamicLazyUnionCache,
        mut states: SmallVec<[u32; 8]>,
        use_pair_map: bool,
    ) -> Option<u32> {
        states.sort_unstable();
        states.dedup();
        Self::intern_sorted_physical_inner(base_state_count, cache, states, use_pair_map)
    }

    fn intern_sorted_physical_inner(
        base_state_count: u32,
        cache: &mut DynamicLazyUnionCache,
        states: SmallVec<[u32; 8]>,
        use_pair_map: bool,
    ) -> Option<u32> {
        debug_assert!(states.windows(2).all(|pair| pair[0] < pair[1]));
        match states.as_slice() {
            [] => return None,
            [state] => return Some(*state),
            _ => {}
        }
        if states.iter().any(|&state| state >= base_state_count) {
            return None;
        }
        let pair_key = if use_pair_map {
            match states.as_slice() {
                [first, second] => Some((u64::from(*first) << 32) | u64::from(*second)),
                _ => None,
            }
        } else {
            None
        };
        if let Some(key) = pair_key {
            if let Some(&state) = cache
                .state_by_pair
                .as_ref()
                .and_then(|pair_map| pair_map.get(&key))
            {
                return Some(state);
            }
        } else if let Some(&state) = cache.state_by_subset.get(&states) {
            return Some(state);
        }
        if cache.subsets.len() >= Self::RESERVED_EXTENSION_STATES {
            return None;
        }
        let id = base_state_count.checked_add(cache.subsets.len() as u32)?;
        if let Some(key) = pair_key {
            cache
                .state_by_pair
                .get_or_insert_with(|| Box::new(FxHashMap::default()))
                .insert(key, id);
        } else {
            cache.state_by_subset.insert(states.clone(), id);
        }
        cache.subsets.push(states);
        cache.rows.push(DynamicLazyUnionRow::default());
        cache.metadata.push(None);
        Some(id)
    }

    #[inline(always)]
    fn extension_index(&self, state: u32) -> usize {
        (state - self.base_state_count) as usize
    }

    #[inline(always)]
    fn checked_extension_index(&self, state: u32) -> Option<usize> {
        if state < self.base_state_count {
            return None;
        }
        let index = self.extension_index(state);
        let valid = unsafe {
            let cache = &*self.cache.get();
            index < cache.subsets.len()
                && index < cache.rows.len()
                && index < cache.metadata.len()
        };
        if !valid {
            if std::env::var_os("GLRMASK_PROFILE_INVALID_LAZY_UNION_STATE").is_some() {
                let cache = unsafe { &*self.cache.get() };
                eprintln!(
                    "[glrmask/profile][invalid_lazy_union_state] state={} base={} index={} subsets={} rows={} metadata={} reserved={}",
                    state,
                    self.base_state_count,
                    index,
                    cache.subsets.len(),
                    cache.rows.len(),
                    cache.metadata.len(),
                    Self::RESERVED_EXTENSION_STATES,
                );
            }
            // Lazy-union execution is an optional bounded acceleration. An
            // out-of-domain state must never become a correctness or memory
            // safety boundary: flag the attempt so the caller discards this
            // partial walk and reruns through the ordinary exact walker.
            self.overflowed.set(true);
            return None;
        }
        Some(index)
    }

    #[inline(always)]
    fn uncached_base_cell(&self, state: u32, byte: u8) -> u32 {
        if let Some(base_transitions) = self.base_transitions16 {
            let cell = unsafe {
                *base_transitions
                    .get_unchecked((state as usize).wrapping_mul(256) + byte as usize)
            };
            return if cell == u16::MAX {
                u32::MAX
            } else {
                u32::from(cell & 0x7fff)
                    | if cell & 0x8000 != 0 { 0x8000_0000 } else { 0 }
            };
        }
        if let Some(base_transitions) = self.base_transitions32 {
            return unsafe {
                *base_transitions
                    .get_unchecked((state as usize).wrapping_mul(256) + byte as usize)
            };
        }
        let tokenizer = unsafe { &*self.tokenizer };
        let target = tokenizer.dynamic_direct_transition(state, byte);
        if target == u32::MAX {
            return u32::MAX;
        }
        debug_assert!(target < 0x8000_0000);
        target
            | if tokenizer.matched_terminal_bitset(target).is_empty() {
                0
            } else {
                0x8000_0000
            }
    }

    fn base_identity_byte_classes(
        &self,
        tokenizer: &Tokenizer,
        state: u32,
    ) -> Option<FullWalkIdentityByteClasses> {
        if self.base_transitions16.is_some() || self.base_transitions32.is_some() {
            // Existing dense rows are cheap read-only evidence. Do not call
            // cell_raw(), which could materialize new union derivatives.
            return Some(full_walk_identity_byte_classes(state, |byte| {
                let cell = self.uncached_base_cell(state, byte);
                (cell != u32::MAX).then(|| (cell & 0x7fff_ffff, cell & 0x8000_0000 != 0))
            }));
        }
        // This query declines virtual/compressed coordinates. Plain physical
        // transitions and their self-loop range query use the same immutable
        // byte-transition semantics as dynamic_direct_transition().
        tokenizer.dynamic_direct_transition_count(state)?;
        let bytes = Lexer::self_loop_bytes(tokenizer, state).to_words();
        let mut result = FullWalkIdentityByteClasses::default();
        if tokenizer.matched_terminal_bitset(state).is_empty() {
            result.ordinary = bytes;
        } else {
            result.finalizing = bytes;
        }
        Some(result)
    }

    #[inline(always)]
    fn base_cell(&self, state: u32, byte: u8) -> u32 {
        if let Some(base_transitions) = self.base_transitions16 {
            let cell = unsafe {
                *base_transitions
                    .get_unchecked((state as usize).wrapping_mul(256) + byte as usize)
            };
            return if cell == u16::MAX {
                u32::MAX
            } else {
                u32::from(cell & 0x7fff)
                    | if cell & 0x8000 != 0 { 0x8000_0000 } else { 0 }
            };
        }
        if let Some(base_transitions) = self.base_transitions32 {
            return unsafe {
                *base_transitions
                    .get_unchecked((state as usize).wrapping_mul(256) + byte as usize)
            };
        }
        // Scalar-dispatch tokenizers can expose thousands of physical states,
        // while one mask normally probes only a few bytes from each newly seen
        // state. Building a complete 256-cell row on the first probe creates a
        // large tail spike; never caching the probe makes common repeated states
        // needlessly expensive. Allocate the row lazily and fill only requested
        // cells, so each exact physical (state, byte) transition is paid once.
        let cached = unsafe {
            let cache = &mut *self.cache.get();
            let slot = cache.base_rows.get_unchecked_mut(state as usize);
            let row = slot.get_or_insert_with(|| Box::new([Self::UNBUILT; 256]));
            *row.get_unchecked(byte as usize)
        };
        if cached != Self::UNBUILT {
            return cached;
        }
        let value = self.uncached_base_cell(state, byte);
        unsafe {
            let cache = &mut *self.cache.get();
            *cache
                .base_rows
                .get_unchecked_mut(state as usize)
                .as_mut()
                .unwrap_unchecked()
                .get_unchecked_mut(byte as usize) = value;
        }
        value
    }

    #[inline(always)]
    fn cached_virtual_cell(&self, index: usize, byte: u8) -> Option<u32> {
        let cache = unsafe { &*self.cache.get() };
        match cache.rows.get(index)? {
            DynamicLazyUnionRow::Sparse(cells) => cells
                .iter()
                .find_map(|&(cached_byte, value)| (cached_byte == byte).then_some(value)),
            DynamicLazyUnionRow::Dense(row) => {
                let value = unsafe { *row.get_unchecked(byte as usize) };
                (value != Self::UNBUILT).then_some(value)
            }
        }
    }

    #[inline(always)]
    fn cache_virtual_cell(&self, index: usize, byte: u8, value: u32) {
        const SPARSE_CELL_LIMIT: usize = 8;
        let cache = unsafe { &mut *self.cache.get() };
        let row = unsafe { cache.rows.get_unchecked_mut(index) };
        match row {
            DynamicLazyUnionRow::Sparse(cells) => {
                if let Some((_, existing)) = cells.iter_mut().find(|(cached_byte, _)| *cached_byte == byte) {
                    *existing = value;
                    return;
                }
                if cells.len() < SPARSE_CELL_LIMIT {
                    cells.push((byte, value));
                    return;
                }
                let mut dense = Box::new([Self::UNBUILT; 256]);
                for &(cached_byte, cached_value) in cells.iter() {
                    dense[cached_byte as usize] = cached_value;
                }
                dense[byte as usize] = value;
                *row = DynamicLazyUnionRow::Dense(dense);
            }
            DynamicLazyUnionRow::Dense(dense) => unsafe {
                *dense.get_unchecked_mut(byte as usize) = value;
            },
        }
    }

    /// After a wide virtual subset has already paid for several independent
    /// byte probes, switch sparse physical components to row-wise construction
    /// only when their cheap packed-row lengths prove that enumerating every
    /// outgoing edge is no more work than another handful of subset probes.
    fn maybe_materialize_sparse_wide_row(&self, index: usize) -> bool {
        const WIDE_SUBSET_MIN: usize = 64;
        const PROBE_AMORTIZATION: usize = 8;
        if self.base_transitions16.is_some() || self.base_transitions32.is_some() {
            return false;
        }
        let members = {
            let cache = unsafe { &*self.cache.get() };
            let Some(DynamicLazyUnionRow::Sparse(cells)) = cache.rows.get(index) else {
                return false;
            };
            if cells.len() < PROBE_AMORTIZATION {
                return false;
            }
            let Some(subset) = cache.subsets.get(index) else {
                return false;
            };
            if subset.len() < WIDE_SUBSET_MIN {
                return false;
            }
            subset.clone()
        };
        let tokenizer = unsafe { &*self.tokenizer };
        let budget = members.len().saturating_mul(PROBE_AMORTIZATION);
        let mut outgoing = 0usize;
        for &member in &members {
            let Some(count) = tokenizer.dynamic_direct_transition_count(member) else {
                return false;
            };
            outgoing = outgoing.saturating_add(count);
            if outgoing > budget {
                return false;
            }
        }
        // Near-one-edge-per-state dispatcher layers are already cheap to probe
        // and do not amortize the fixed row-build machinery. Conversely, dense
        // rows were rejected by `budget` above. Materialize only the middle
        // regime where repeated subset probes dominate but sparse enumeration
        // is still bounded.
        if outgoing.saturating_mul(2) < members.len().saturating_mul(3) {
            return false;
        }

        let mut targets_by_byte: [SmallVec<[u32; 8]>; 256] =
            std::array::from_fn(|_| SmallVec::new());
        let mut touched = SmallVec::<[u8; 32]>::new();
        let mut finalizing = [false; 256];
        for member in members {
            for (byte, target) in tokenizer.transitions_from(member) {
                let targets = unsafe { targets_by_byte.get_unchecked_mut(byte as usize) };
                if targets.is_empty() {
                    touched.push(byte);
                }
                targets.push(target);
                if !finalizing[byte as usize]
                    && !tokenizer.matched_terminal_bitset(target).is_empty()
                {
                    finalizing[byte as usize] = true;
                }
            }
        }
        let mut row = Box::new([u32::MAX; 256]);
        for byte in touched {
            let targets = unsafe { targets_by_byte.get_unchecked_mut(byte as usize) };
            canonicalize_lazy_targets(targets);
            let target = match targets.as_slice() {
                [] => continue,
                [single] => *single,
                _ => {
                    let cache = unsafe { &mut *self.cache.get() };
                    let Some(target) = Self::intern_sorted_physical_inner(
                        self.base_state_count,
                        cache,
                        std::mem::take(targets),
                        self.use_pair_map,
                    ) else {
                        self.overflowed.set(true);
                        return false;
                    };
                    target
                }
            };
            row[byte as usize] = target
                | if finalizing[byte as usize] {
                    0x8000_0000
                } else {
                    0
                };
        }
        unsafe {
            (&mut *self.cache.get()).rows[index] = DynamicLazyUnionRow::Dense(row);
        }
        true
    }

    #[inline(always)]
    fn cell_raw(&self, state: u32, byte: u8) -> u32 {
        if state < self.base_state_count {
            return self.base_cell(state, byte);
        }
        let Some(index) = self.checked_extension_index(state) else {
            return u32::MAX;
        };
        if let Some(cached) = self.cached_virtual_cell(index, byte) {
            return cached;
        }
        if self.maybe_materialize_sparse_wide_row(index) {
            return self
                .cached_virtual_cell(index, byte)
                .unwrap_or(u32::MAX);
        }

        let mut targets = SmallVec::<[u32; 8]>::new();
        let mut finalizer_bits = 0u32;
        // Virtual derivatives cache their own resulting cell, so member probes
        // deliberately bypass the physical-row cache. That makes this scan
        // read-only with respect to `DynamicLazyUnionCache`: borrow the canonical
        // subset in place instead of cloning a potentially heap-backed wide
        // union on every newly observed vocabulary byte.
        {
            let cache = unsafe { &*self.cache.get() };
            for &member in &cache.subsets[index] {
                let cell = self.uncached_base_cell(member, byte);
                if cell == u32::MAX {
                    continue;
                }
                finalizer_bits |= cell & 0x8000_0000;
                targets.push(cell & 0x7fff_ffff);
            }
        }
        canonicalize_lazy_targets(&mut targets);
        let target = match targets.as_slice() {
            [] => u32::MAX,
            [state] => *state,
            _ => {
                let cache = unsafe { &mut *self.cache.get() };
                let Some(target) =
                    Self::intern_sorted_physical_inner(
                        self.base_state_count,
                        cache,
                        targets,
                        self.use_pair_map,
                    )
                else {
                    self.overflowed.set(true);
                    return u32::MAX;
                };
                target
            }
        };
        let value = if target == u32::MAX {
            u32::MAX
        } else {
            target | finalizer_bits
        };
        self.cache_virtual_cell(index, byte, value);
        value
    }

    #[inline(always)]
    fn ensure_virtual_metadata(&self, state: u32) -> bool {
        debug_assert!(state >= self.base_state_count);
        let Some(index) = self.checked_extension_index(state) else {
            return false;
        };
        if unsafe { (&*self.cache.get()).metadata[index].is_none() } {
            self.build_virtual_metadata(index);
        }
        true
    }

    // Every finalizer/future lookup hits the small cached branch above. Keep
    // first-use bitset construction out of those inlined vocabulary-walk
    // sites, while retaining the checked extension-coordinate boundary.
    #[cold]
    #[inline(never)]
    fn build_virtual_metadata(&self, index: usize) {
        let tokenizer = unsafe { &*self.tokenizer };
        let mut matched = BitSet::new(tokenizer.num_terminals() as usize);
        let mut futures = BitSet::new(tokenizer.num_terminals() as usize);
        {
            let cache = unsafe { &*self.cache.get() };
            for &member in &cache.subsets[index] {
                matched.union_with(tokenizer.matched_terminal_bitset(member));
                futures.union_with(tokenizer.possible_future_terminals(member));
            }
        }
        let (first, second) = {
            let mut iter = matched.iter_ones().map(|terminal| terminal as TerminalID);
            (iter.next(), iter.next())
        };
        let finalizer_code = match (first, second) {
            (None, _) => u32::MAX,
            (Some(terminal), None) => terminal,
            _ => u32::MAX - 1,
        };
        let single_finalizer_continues = first
            .filter(|_| second.is_none())
            .is_some_and(|terminal| futures.contains(terminal as usize));
        let metadata = DynamicLazyUnionMetadata {
            finalizer_code,
            single_finalizer_continues: u8::from(single_finalizer_continues),
            matched,
            futures,
        };
        let cache = unsafe { &mut *self.cache.get() };
        if cache.metadata[index].is_none() {
            cache.metadata[index] = Some(metadata);
        }
    }
}

impl FullWalkTransitionTable for FullWalkLazyUnion<'_> {
    type Cell = u32;

    #[inline(always)]
    fn cell(&self, state: u32, byte: u8) -> u32 {
        self.cell_raw(state, byte)
    }

    #[inline(always)] fn cell_is_dead(cell: u32) -> bool { cell == u32::MAX }
    #[inline(always)] fn cell_has_finalizer(cell: u32) -> bool { cell & 0x8000_0000 != 0 }
    #[inline(always)] fn cell_target(cell: u32) -> u32 { cell & 0x7fff_ffff }

    fn continuation_witness_components(&self, state: u32) -> SmallVec<[u32; 8]> {
        if state < self.base_state_count { return smallvec::smallvec![state]; }
        let Some(index) = self.checked_extension_index(state) else {
            return SmallVec::new();
        };
        let cache = unsafe { &*self.cache.get() };
        let members = &cache.subsets[index];
        if members.len() > 8 { return SmallVec::new(); }
        // The owned inline copy releases the UnsafeCell borrow before any
        // later metadata or transition query can grow the lazy-union cache.
        SmallVec::from_slice(members)
    }

    fn proven_identity_byte_classes(
        &self,
        tokenizer: &Tokenizer,
        state: u32,
    ) -> Option<FullWalkIdentityByteClasses> {
        if state < self.base_state_count {
            return self.base_identity_byte_classes(tokenizer, state);
        }
        let index = self.checked_extension_index(state)?;
        let cache = unsafe { &*self.cache.get() };
        let members = &cache.subsets[index];
        if members.len() > 8 { return None; }
        let (&first, rest) = members.split_first()?;
        let mut result = self.base_identity_byte_classes(tokenizer, first)?;
        // Every member staying itself is sufficient for the exact union to
        // stay itself. A union edge finalizes iff any member edge finalizes.
        // No derivative state is interned by this proof.
        for &member in rest {
            result.intersect_union_member(self.base_identity_byte_classes(tokenizer, member)?);
        }
        Some(result)
    }

    #[inline(always)]
    fn state_count(&self, _tokenizer: &Tokenizer) -> usize {
        self.base_state_count as usize + Self::RESERVED_EXTENSION_STATES
    }

    #[inline(always)]
    fn finalizer_code(&self, state: u32, base: &[u32]) -> u32 {
        if state < self.base_state_count {
            if !base.is_empty() {
                return unsafe { *base.get_unchecked(state as usize) };
            }
            let tokenizer = unsafe { &*self.tokenizer };
            return match tokenizer.matched_terminals_slice(state) {
                [] => u32::MAX,
                [terminal] => *terminal,
                _ => u32::MAX - 1,
            };
        }
        if !self.ensure_virtual_metadata(state) {
            return u32::MAX;
        }
        let cache = unsafe { &*self.cache.get() };
        cache.metadata[self.extension_index(state)]
            .as_ref().expect("virtual subset metadata missing").finalizer_code
    }

    #[inline(always)]
    fn single_finalizer_continues(&self, state: u32, base: &[u8]) -> bool {
        if state < self.base_state_count {
            if !base.is_empty() {
                return unsafe { *base.get_unchecked(state as usize) != 0 };
            }
            let tokenizer = unsafe { &*self.tokenizer };
            return match tokenizer.matched_terminals_slice(state) {
                [terminal] => tokenizer
                    .possible_future_terminals(state)
                    .contains(*terminal as usize),
                _ => false,
            };
        }
        if !self.ensure_virtual_metadata(state) {
            return false;
        }
        let cache = unsafe { &*self.cache.get() };
        cache.metadata[self.extension_index(state)]
            .as_ref().expect("virtual subset metadata missing").single_finalizer_continues != 0
    }

    #[inline]
    fn matched_terminals(&self, tokenizer: &Tokenizer, state: u32) -> SmallVec<[TerminalID; 4]> {
        if state < self.base_state_count {
            return tokenizer.matched_terminals_slice(state).iter().copied().collect();
        }
        if !self.ensure_virtual_metadata(state) {
            return SmallVec::new();
        }
        let cache = unsafe { &*self.cache.get() };
        cache.metadata[self.extension_index(state)]
            .as_ref().expect("virtual subset metadata missing").matched
            .iter_ones().map(|terminal| terminal as TerminalID).collect()
    }


    #[inline(always)]
    fn future_contains(&self, tokenizer: &Tokenizer, state: u32, terminal: TerminalID) -> bool {
        if state < self.base_state_count {
            return tokenizer.possible_future_terminals(state).contains(terminal as usize);
        }
        if !self.ensure_virtual_metadata(state) {
            return false;
        }
        let cache = unsafe { &*self.cache.get() };
        cache.metadata[self.extension_index(state)]
            .as_ref().expect("virtual subset metadata missing").futures.contains(terminal as usize)
    }

    #[inline(always)]
    fn future_intersects(&self, tokenizer: &Tokenizer, state: u32, terminals: &BitSet) -> bool {
        if state < self.base_state_count {
            return !terminals.is_disjoint(tokenizer.possible_future_terminals(state));
        }
        if !self.ensure_virtual_metadata(state) {
            return false;
        }
        let cache = unsafe { &*self.cache.get() };
        !terminals.is_disjoint(&cache.metadata[self.extension_index(state)]
            .as_ref().expect("virtual subset metadata missing").futures)
    }

    #[inline]
    fn sole_live_terminal(&self, tokenizer: &Tokenizer, state: u32) -> Option<TerminalID> {
        if state < self.base_state_count {
            return physical_sole_live_terminal(tokenizer, state);
        }
        if !self.ensure_virtual_metadata(state) {
            return None;
        }
        let cache = unsafe { &*self.cache.get() };
        let metadata = cache.metadata[self.extension_index(state)]
            .as_ref()
            .expect("virtual subset metadata missing");
        bitset_sole_live_terminal(&metadata.matched, &metadata.futures)
    }

    #[inline]
    fn union_states(&self, states: &[u32]) -> Option<u32> {
        self.intern_states(states)
    }

    #[inline(always)]
    fn dense_output_hint(&self, state: u32) -> Option<bool> {
        let cache = unsafe { &*self.cache.get() };
        match cache.dense_output_hints.get(&state).copied() {
            Some(2) => Some(true),
            Some(1) => Some(false),
            _ => None,
        }
    }

    #[inline(always)]
    fn cache_dense_output_hint(&self, state: u32, dense: bool) {
        let cache = unsafe { &mut *self.cache.get() };
        cache.dense_output_hints.insert(state, if dense { 2 } else { 1 });
    }
}

struct FullWalkSubset16<'a> {
    base_transitions: &'a [u16],
    base_state_count: u32,
    rows: Vec<Box<[u32; 256]>>,
    finalizer_code: Vec<u32>,
    single_finalizer_continues: Vec<u8>,
    matched: Vec<BitSet>,
    futures: Vec<BitSet>,
}

impl<'a> FullWalkSubset16<'a> {
    const MAX_EXTENSION_STATES: usize = 4096;

    fn build(
        tokenizer: &Tokenizer,
        base_transitions: &'a [u16],
        root_states: &[u32],
        horizon: usize,
    ) -> Option<(Self, u32, Vec<SmallVec<[u32; 8]>>)> {
        let base_state_count = tokenizer.num_states();
        let mut table = Self {
            base_transitions,
            base_state_count,
            rows: Vec::new(),
            finalizer_code: Vec::new(),
            single_finalizer_continues: Vec::new(),
            matched: Vec::new(),
            futures: Vec::new(),
        };
        let mut state_by_subset = FxHashMap::<SmallVec<[u32; 8]>, u32>::default();
        let mut subsets = Vec::<SmallVec<[u32; 8]>>::new();
        let mut depths = Vec::<usize>::new();

        fn intern(
            table: &mut FullWalkSubset16<'_>,
            tokenizer: &Tokenizer,
            state_by_subset: &mut FxHashMap<SmallVec<[u32; 8]>, u32>,
            subsets: &mut Vec<SmallVec<[u32; 8]>>,
            depths: &mut Vec<usize>,
            depth: usize,
            mut states: SmallVec<[u32; 8]>,
        ) -> Option<u32> {
            states.sort_unstable();
            states.dedup();
            match states.as_slice() {
                [] => return None,
                [state] => return Some(*state),
                _ => {}
            }
            if let Some(&state) = state_by_subset.get(&states) {
                return Some(state);
            }
            if subsets.len() >= FullWalkSubset16::MAX_EXTENSION_STATES {
                return None;
            }
            if states.iter().any(|&state| state >= table.base_state_count) {
                return None;
            }
            let id = table.base_state_count + subsets.len() as u32;
            let mut matched = BitSet::new(tokenizer.num_terminals() as usize);
            let mut futures = BitSet::new(tokenizer.num_terminals() as usize);
            for &state in &states {
                matched.union_with(tokenizer.matched_terminal_bitset(state));
                futures.union_with(tokenizer.possible_future_terminals(state));
            }
            let (first, second) = {
                let mut matched_iter = matched.iter_ones().map(|terminal| terminal as TerminalID);
                (matched_iter.next(), matched_iter.next())
            };
            let code = match (first, second) {
                (None, _) => u32::MAX,
                (Some(terminal), None) => terminal,
                _ => u32::MAX - 1,
            };
            let continues = first
                .filter(|_| second.is_none())
                .is_some_and(|terminal| futures.contains(terminal as usize));
            state_by_subset.insert(states.clone(), id);
            subsets.push(states);
            depths.push(depth);
            table.rows.push(Box::new([u32::MAX; 256]));
            table.finalizer_code.push(code);
            table.single_finalizer_continues.push(u8::from(continues));
            table.matched.push(matched);
            table.futures.push(futures);
            Some(id)
        }

        let root = intern(
            &mut table,
            tokenizer,
            &mut state_by_subset,
            &mut subsets,
            &mut depths,
            0,
            SmallVec::from_slice(root_states),
        )?;
        if root < base_state_count {
            return Some((table, root, subsets));
        }

        let mut build_index = 0usize;
        while build_index < subsets.len() {
            let depth = depths[build_index];
            if depth >= horizon {
                build_index += 1;
                continue;
            }
            let members = subsets[build_index].clone();
            let mut row = [u32::MAX; 256];
            for byte in 0..=u8::MAX {
                let mut targets = SmallVec::<[u32; 8]>::new();
                for &state in members.iter() {
                    let cell = unsafe {
                        *base_transitions
                            .get_unchecked((state as usize).wrapping_mul(256) + byte as usize)
                    };
                    if cell != u16::MAX {
                        targets.push(u32::from(cell & 0x7fff));
                    }
                }
                if targets.is_empty() {
                    continue;
                }
                let target = intern(
                    &mut table,
                    tokenizer,
                    &mut state_by_subset,
                    &mut subsets,
                    &mut depths,
                    depth + 1,
                    targets,
                )?;
                let has_finalizer = if target < base_state_count {
                    !tokenizer.matched_terminal_bitset(target).is_empty()
                } else {
                    !table.matched[(target - base_state_count) as usize].is_empty()
                };
                row[byte as usize] = target | if has_finalizer { 0x8000_0000 } else { 0 };
            }
            table.rows[build_index] = Box::new(row);
            build_index += 1;
        }
        Some((table, root, subsets))
    }

    fn into_owned(self, root_state: u32, subsets: Vec<SmallVec<[u32; 8]>>) -> DynamicDenseSubset16 {
        DynamicDenseSubset16 {
            root_state,
            base_state_count: self.base_state_count,
            rows: self.rows,
            finalizer_code: self.finalizer_code,
            single_finalizer_continues: self.single_finalizer_continues,
            matched: self.matched,
            futures: self.futures,
            subsets,
        }
    }

    #[inline(always)]
    fn extension_index(&self, state: u32) -> usize {
        (state - self.base_state_count) as usize
    }
}

impl FullWalkTransitionTable for FullWalkSubset16<'_> {
    type Cell = u32;

    #[inline(always)]
    fn cell(&self, state: u32, byte: u8) -> u32 {
        if state < self.base_state_count {
            let cell = unsafe {
                *self
                    .base_transitions
                    .get_unchecked((state as usize).wrapping_mul(256) + byte as usize)
            };
            if cell == u16::MAX {
                u32::MAX
            } else {
                u32::from(cell & 0x7fff)
                    | if cell & 0x8000 != 0 { 0x8000_0000 } else { 0 }
            }
        } else {
            unsafe {
                *self
                    .rows
                    .get_unchecked(self.extension_index(state))
                    .get_unchecked(byte as usize)
            }
        }
    }

    #[inline(always)]
    fn cell_is_dead(cell: u32) -> bool { cell == u32::MAX }

    #[inline(always)]
    fn cell_has_finalizer(cell: u32) -> bool { cell & 0x8000_0000 != 0 }

    #[inline(always)]
    fn cell_target(cell: u32) -> u32 { cell & 0x7fff_ffff }

    #[inline(always)]
    fn state_count(&self, _tokenizer: &Tokenizer) -> usize {
        self.base_state_count as usize + self.rows.len()
    }

    #[inline(always)]
    fn finalizer_code(&self, state: u32, base: &[u32]) -> u32 {
        if state < self.base_state_count {
            unsafe { *base.get_unchecked(state as usize) }
        } else {
            unsafe { *self.finalizer_code.get_unchecked(self.extension_index(state)) }
        }
    }

    #[inline(always)]
    fn single_finalizer_continues(&self, state: u32, base: &[u8]) -> bool {
        if state < self.base_state_count {
            unsafe { *base.get_unchecked(state as usize) != 0 }
        } else {
            unsafe { *self.single_finalizer_continues.get_unchecked(self.extension_index(state)) != 0 }
        }
    }

    #[inline]
    fn matched_terminals(&self, tokenizer: &Tokenizer, state: u32) -> SmallVec<[TerminalID; 4]> {
        if state < self.base_state_count {
            tokenizer.matched_terminals_slice(state).iter().copied().collect()
        } else {
            self.matched[self.extension_index(state)]
                .iter_ones()
                .map(|terminal| terminal as TerminalID)
                .collect()
        }
    }


    #[inline(always)]
    fn future_contains(&self, tokenizer: &Tokenizer, state: u32, terminal: TerminalID) -> bool {
        if state < self.base_state_count {
            tokenizer.possible_future_terminals(state).contains(terminal as usize)
        } else {
            self.futures[self.extension_index(state)].contains(terminal as usize)
        }
    }

    #[inline(always)]
    fn future_intersects(&self, tokenizer: &Tokenizer, state: u32, terminals: &BitSet) -> bool {
        if state < self.base_state_count {
            !terminals.is_disjoint(tokenizer.possible_future_terminals(state))
        } else {
            !terminals.is_disjoint(&self.futures[self.extension_index(state)])
        }
    }

    #[inline]
    fn sole_live_terminal(&self, tokenizer: &Tokenizer, state: u32) -> Option<TerminalID> {
        if state < self.base_state_count {
            return physical_sole_live_terminal(tokenizer, state);
        }
        let index = self.extension_index(state);
        bitset_sole_live_terminal(&self.matched[index], &self.futures[index])
    }
}


struct FullWalkCachedSubset16<'a> {
    base_transitions: &'a [u16],
    extension: Arc<DynamicDenseSubset16>,
}

impl FullWalkCachedSubset16<'_> {
    #[inline(always)]
    fn extension_index(&self, state: u32) -> usize {
        (state - self.extension.base_state_count) as usize
    }
}

impl FullWalkTransitionTable for FullWalkCachedSubset16<'_> {
    type Cell = u32;

    #[inline(always)]
    fn cell(&self, state: u32, byte: u8) -> u32 {
        if state < self.extension.base_state_count {
            let cell = unsafe {
                *self.base_transitions
                    .get_unchecked((state as usize).wrapping_mul(256) + byte as usize)
            };
            if cell == u16::MAX {
                u32::MAX
            } else {
                u32::from(cell & 0x7fff)
                    | if cell & 0x8000 != 0 { 0x8000_0000 } else { 0 }
            }
        } else {
            unsafe {
                *self.extension.rows
                    .get_unchecked(self.extension_index(state))
                    .get_unchecked(byte as usize)
            }
        }
    }

    #[inline(always)]
    fn cell_is_dead(cell: u32) -> bool { cell == u32::MAX }
    #[inline(always)]
    fn cell_has_finalizer(cell: u32) -> bool { cell & 0x8000_0000 != 0 }
    #[inline(always)]
    fn cell_target(cell: u32) -> u32 { cell & 0x7fff_ffff }

    #[inline(always)]
    fn state_count(&self, _tokenizer: &Tokenizer) -> usize {
        self.extension.base_state_count as usize + self.extension.rows.len()
    }

    #[inline(always)]
    fn finalizer_code(&self, state: u32, base: &[u32]) -> u32 {
        if state < self.extension.base_state_count {
            unsafe { *base.get_unchecked(state as usize) }
        } else {
            unsafe { *self.extension.finalizer_code.get_unchecked(self.extension_index(state)) }
        }
    }

    #[inline(always)]
    fn single_finalizer_continues(&self, state: u32, base: &[u8]) -> bool {
        if state < self.extension.base_state_count {
            unsafe { *base.get_unchecked(state as usize) != 0 }
        } else {
            unsafe { *self.extension.single_finalizer_continues.get_unchecked(self.extension_index(state)) != 0 }
        }
    }

    #[inline]
    fn matched_terminals(&self, tokenizer: &Tokenizer, state: u32) -> SmallVec<[TerminalID; 4]> {
        if state < self.extension.base_state_count {
            tokenizer.matched_terminals_slice(state).iter().copied().collect()
        } else {
            self.extension.matched[self.extension_index(state)]
                .iter_ones().map(|terminal| terminal as TerminalID).collect()
        }
    }


    #[inline(always)]
    fn future_contains(&self, tokenizer: &Tokenizer, state: u32, terminal: TerminalID) -> bool {
        if state < self.extension.base_state_count {
            tokenizer.possible_future_terminals(state).contains(terminal as usize)
        } else {
            self.extension.futures[self.extension_index(state)].contains(terminal as usize)
        }
    }

    #[inline(always)]
    fn future_intersects(&self, tokenizer: &Tokenizer, state: u32, terminals: &BitSet) -> bool {
        if state < self.extension.base_state_count {
            !terminals.is_disjoint(tokenizer.possible_future_terminals(state))
        } else {
            !terminals.is_disjoint(&self.extension.futures[self.extension_index(state)])
        }
    }

    #[inline]
    fn sole_live_terminal(&self, tokenizer: &Tokenizer, state: u32) -> Option<TerminalID> {
        if state < self.extension.base_state_count {
            return physical_sole_live_terminal(tokenizer, state);
        }
        let index = self.extension_index(state);
        bitset_sole_live_terminal(
            &self.extension.matched[index],
            &self.extension.futures[index],
        )
    }
}

#[derive(Clone, Copy)]
struct FullWalkFlat32<'a> {
    transitions: &'a [u32],
}

impl FullWalkTransitionTable for FullWalkFlat32<'_> {
    type Cell = u32;

    #[inline(always)]
    fn cell(&self, state: u32, byte: u8) -> u32 {
        unsafe {
            *self
                .transitions
                .get_unchecked((state as usize).wrapping_mul(256) + byte as usize)
        }
    }

    #[inline(always)]
    fn cell_is_dead(cell: u32) -> bool {
        cell == u32::MAX
    }

    #[inline(always)]
    fn cell_has_finalizer(cell: u32) -> bool {
        cell & 0x8000_0000 != 0
    }

    #[inline(always)]
    fn cell_target(cell: u32) -> u32 {
        cell & 0x7fff_ffff
    }

}

#[derive(Clone, Copy, Debug, Default)]
struct LlgMasterDecision {
    /// Exact whole-token safe-string radius proved at the current lexer/parser
    /// frontier. `u16::MAX` denotes the complete unbounded safe+ language.
    safe_radius: u16,
    whitespace: bool,
}

impl LlgMasterDecision {
    #[inline(always)]
    fn admits_root_class(self, class: u16) -> bool {
        let safe_chars = crate::runtime::dynamic_mask_llg_master_safe_chars(class);
        (safe_chars != 0 && safe_chars <= self.safe_radius)
            || (self.whitespace
                && crate::runtime::dynamic_mask_llg_master_is_whitespace(class))
    }

    #[inline(always)]
    fn is_empty(self) -> bool {
        self.safe_radius == 0 && !self.whitespace
    }
}

/// Prove the exact master safe+/whitespace certificate while every original
/// lexer root still carries its exact source state. This must run before any
/// same-parser lexer-root union, because the union coordinate intentionally
/// discards the one-source provenance used by projected/symbolic proofs.
#[cold]
#[inline(never)]
fn precollapse_master_decision(
    state: &ConstraintState<'_>,
    vocab: &DynamicMaskVocab,
    root_branches: &DynamicBranches,
    lexer_state_count: usize,
) -> Option<LlgMasterDecision> {
    // Profiling flags are intentionally read once per proof attempt rather
    // than inside the hot parser/proof helpers. These knobs are sometimes
    // enabled dynamically by diagnostic harnesses, so unlike ordinary feature
    // flags they should not be process-cached.
    let profile_mask = dynamic_mask_profile_enabled(state.generation);
    let profile_proof = dynamic_mask_proof_profile_enabled() && profile_mask;
    let proof_definitions_available = vocab.llg_master_trie().is_some()
        && vocab.llg_slice_by_cache_id(LLG_SAFE_PLUS_SLICE as u32).is_some()
        && vocab.llg_slice_by_cache_id(LLG_WHITESPACE_SLICE as u32).is_some();
    if profile_mask {
        eprintln!(
            "[glrmask/profile][precollapse_master_entry] roots={} pending_guard={} missing_exact={} definitions={}",
            root_branches.len(),
            root_branches
                .iter()
                .any(|branch| !branch.initial_prune_guard.is_passed()),
            root_branches
                .iter()
                .any(|branch| branch.exact_tokenizer_state.is_none()),
            proof_definitions_available,
        );
    }
    if root_branches.is_empty()
        || root_branches
            .iter()
            .any(|branch| !branch.initial_prune_guard.is_passed())
        || root_branches
            .iter()
            .any(|branch| branch.exact_tokenizer_state.is_none())
        || !proof_definitions_available
    {
        return None;
    }

    let safe_plus_slice = vocab
        .llg_slice_by_cache_id(LLG_SAFE_PLUS_SLICE as u32)
        .expect("safe+ proof slice missing");
    let whitespace_slice = vocab
        .llg_slice_by_cache_id(LLG_WHITESPACE_SLICE as u32)
        .expect("whitespace proof slice missing");
    // Parser admission can be materially more expensive than the cheap
    // lexical support test below, and a failed pre-collapse proof otherwise
    // pays it again when the exact full walk constructs its own parser cache.
    // Every master proof in this function already requires one parser-admitted
    // terminal whose byte support covers the complete slice alphabet. Before
    // asking the parser anything, conservatively test the larger set of all
    // lexically-live terminals. If even that superset has no such terminal,
    // neither safe+/radius nor whitespace can possibly produce a certificate.
    // This is a necessary-condition gate only; it never admits a slice.
    let tokenizer = &state.constraint.tokenizer;
    let lexical_slice_candidate = |slice: &crate::runtime::artifact::DynamicMaskSliceTrie| {
        root_branches.iter().any(|branch| {
            let source = branch
                .exact_tokenizer_state
                .expect("precollapse proof requires exact source");
            tokenizer
                .matched_terminals_slice(source)
                .iter()
                .copied()
                .chain(
                    tokenizer
                        .possible_future_terminals(source)
                        .iter()
                        .map(|terminal| terminal as TerminalID),
                )
                .any(|terminal| {
                    tokenizer
                        .terminal_byte_support(terminal)
                        .is_some_and(|support| slice.slice_token_bytes().is_subset(&support))
                })
        })
    };
    if !lexical_slice_candidate(safe_plus_slice)
        && !lexical_slice_candidate(whitespace_slice)
    {
        if profile_mask {
            eprintln!("[glrmask/profile][precollapse_master] declined=lexical_support");
        }
        return None;
    }

    // These are immutable properties of the two proof DFAs. Compute them once
    // per surviving decision rather than rescanning all 256 bytes for every
    // root and again for the radius fallback. Keep them after the cheap lexical
    // gate above so proof-ineligible states pay none of this work.
    let safe_plus_first_bytes = safe_plus_slice.first_bytes();
    let whitespace_first_bytes = whitespace_slice.first_bytes();

    let (mut parser_cache, root_parser_nodes) =
        FullWalkParserCache::from_roots(root_branches, lexer_state_count, profile_mask);
    let mut outcomes = [false; LLG_PROOF_SLOT_COUNT];
    // When the unbounded safe+ proof fails, the bounded-radius fallback asks
    // exactly the same immutable lexer eligibility question. Retain the
    // already-filtered candidates rather than repeating support/residual work.
    let mut safe_radius_candidates = Vec::<(u32, SmallVec<[TerminalID; 8]>)>::new();
    for slice_index in [LLG_SAFE_PLUS_SLICE, LLG_WHITESPACE_SLICE] {
        let slice = vocab
            .llg_slice_by_cache_id(slice_index as u32)
            .expect("requested proof slice missing");
        let slice_first_bytes = if slice_index == LLG_SAFE_PLUS_SLICE {
            safe_plus_first_bytes
        } else {
            whitespace_first_bytes
        };
        'sources: for (root_index, branch) in root_branches.iter().enumerate() {
            let source = branch
                .exact_tokenizer_state
                .expect("precollapse proof requires exact source");
            let admitted_started = profile_proof.then(std::time::Instant::now);
            let admitted = parser_cache
                .admitted(state.constraint, root_parser_nodes[root_index])
                .clone();
            if let Some(started) = admitted_started {
                eprintln!(
                    "[glrmask/profile][proof_phase] kind=admitted slice={} count={} ms={:.3}",
                    slice_index,
                    admitted.count_ones(),
                    started.elapsed().as_secs_f64() * 1e3,
                );
            }

            // Prepared build-time certificates are exact positive proofs for
            // this source state and slice.  A parser-admitted proving terminal
            // lets us bypass all lexer eligibility filtering and runtime
            // quotient/product work.  Empty/missing rows deliberately fall
            // through to the existing exact proof machinery.
            let prepared_slice_slot = if slice_index == LLG_SAFE_PLUS_SLICE {
                Some(0usize)
            } else if slice_index == LLG_WHITESPACE_SLICE {
                Some(1usize)
            } else {
                None
            };
            if let Some(slice_slot) = prepared_slice_slot {
                if let Some(&terminal) = vocab
                    .prepared_master_provers(source, slice_slot)
                    .iter()
                    .find(|&&terminal| admitted.contains(terminal as usize))
                {
                    if profile_mask {
                        eprintln!(
                            "[glrmask/profile][precollapse_master_proof] slice={} source={} terminal={} proof=prepared",
                            slice_index, source, terminal,
                        );
                    }
                    outcomes[slice_index] = true;
                    break 'sources;
                }
            }

            // Support and physical-residual coverage are immutable lexer facts
            // for this `(source, terminal, slice)`. Derive the candidate list
            // once and reuse it for virtual proof, quotient proof, and (for
            // safe+) the bounded-radius fallback below.
            // Parser admission alone may include hundreds of terminals
            // unrelated to this exact lexer source. Byte support is not a
            // liveness certificate. Refuse impossible candidates BEFORE any
            // per-terminal construction; this only removes an acceleration.
            let filter_source_live =
                std::env::var_os("GLRMASK_DISABLE_PROOF_SOURCE_LIVE_GATE").is_none();
            let eligible = admitted
                .iter_ones()
                .map(|terminal| terminal as TerminalID)
                .filter(|&terminal| {
                    (!filter_source_live || tokenizer.terminal_state_is_live(source, terminal))
                        && tokenizer
                        .terminal_byte_support(terminal)
                        .is_some_and(|support| slice.slice_token_bytes().is_subset(&support))
                        && tokenizer
                            .physical_terminal_residual_covers_first_bytes(
                                source,
                                terminal,
                                slice_first_bytes,
                            )
                            != Some(false)
                })
                .collect::<SmallVec<[TerminalID; 8]>>();
            if profile_proof && slice_index == LLG_SAFE_PLUS_SLICE {
                for &terminal in &eligible {
                    for partition in 0usize..9 {
                        let Some(partition_dfa) = crate::compiler::stages::id_map_and_terminal_dwa::classify::vocab_partition_effective_dfa(partition) else {
                            continue;
                        };
                        let started = std::time::Instant::now();
                        let result = residual_regex_slice_prefix_contained(
                            &state.constraint.tokenizer,
                            source,
                            terminal,
                            partition_dfa.as_ref(),
                            32 * 1024,
                        );
                        eprintln!(
                            "[glrmask/profile][partition_residual_probe] source={} terminal={} partition={} result={:?} us={:.1}",
                            source,
                            terminal,
                            partition,
                            result,
                            started.elapsed().as_secs_f64() * 1e6,
                        );
                        let projected_started = std::time::Instant::now();
                        let projected = vocab.projected_terminal_slice_contained(
                            terminal,
                            source,
                            0x1000 + partition as u32,
                            partition_dfa.as_ref(),
                        );
                        eprintln!(
                            "[glrmask/profile][partition_projected_probe] source={} terminal={} partition={} result={:?} us={:.1}",
                            source,
                            terminal,
                            partition,
                            projected,
                            projected_started.elapsed().as_secs_f64() * 1e6,
                        );
                    }
                }
            }
            if slice_index == LLG_SAFE_PLUS_SLICE {
                safe_radius_candidates.push((source, eligible.clone()));
            }

            for &terminal in &eligible {
                if prepared_slice_slot
                    .and_then(|slot| {
                        vocab.prepared_master_proof_result(source, slot, terminal)
                    })
                    == Some(false)
                {
                    continue;
                }
                let proof_started = profile_proof.then(std::time::Instant::now);
                let virtual_proof = virtual_residual_slice_prefix_contained(
                    &state.constraint.tokenizer,
                    source,
                    terminal,
                    slice.dfa(),
                    1_536,
                );
                if let Some(started) = proof_started {
                    eprintln!(
                        "[glrmask/profile][proof_phase] kind=virtual slice={} terminal={} source={} result={:?} ms={:.3}",
                        slice_index,
                        terminal,
                        source,
                        virtual_proof,
                        started.elapsed().as_secs_f64() * 1e3,
                    );
                }
                if virtual_proof == Some(true) {
                    if profile_mask {
                        eprintln!(
                            "[glrmask/profile][precollapse_master_proof] slice={} source={} terminal={} proof=virtual",
                            slice_index, source, terminal,
                        );
                    }
                    outcomes[slice_index] = true;
                    break 'sources;
                }
            }

            let needs_quotient = eligible.iter().copied().any(|terminal| {
                prepared_slice_slot
                    .and_then(|slot| {
                        vocab.prepared_master_proof_result(source, slot, terminal)
                    })
                    .is_none()
            });
            // A grammar-quotiented dynamic vocabulary has already paid to
            // reduce the model vocabulary to grammar-equivalence representatives.
            // Building a second terminal-projection quotient lazily here can cost
            // orders of magnitude more than simply walking that small trie. Keep
            // consuming explicitly/prepared quotients when present, but do not
            // synthesize them online for the O2 runtime.
            if needs_quotient && !eligible.is_empty() && !vocab.is_grammar_quotiented() {
                if std::env::var_os("GLRMASK_DISABLE_DEMAND_TERMINAL_PROOF").is_none() {
                    // The admitted candidate list is already exact. Preparing all
                    // broad terminals here puts unrelated grammar-wide work on a
                    // single mask's cold path. Reuse the existing per-terminal
                    // cache without changing the containment proof or its scope.
                    for &terminal in &eligible {
                        if prepared_slice_slot.and_then(|slot| {
                            vocab.prepared_master_proof_result(source, slot, terminal)
                        }).is_none() {
                            vocab.prepare_runtime_projected_terminal_quotient(
                                &state.constraint.tokenizer, terminal,
                            );
                        }
                    }
                } else {
                    vocab.prepare_runtime_projected_terminal_quotients(
                        &state.constraint.tokenizer,
                        &safe_plus_slice.slice_token_bytes(),
                        None,
                    );
                }
            }

            for &terminal in &eligible {
                if prepared_slice_slot
                    .and_then(|slot| {
                        vocab.prepared_master_proof_result(source, slot, terminal)
                    })
                    .is_some()
                {
                    continue;
                }
                let proof_started = profile_proof.then(std::time::Instant::now);
                let quotient_proof = vocab.projected_terminal_slice_contained(
                    terminal,
                    source,
                    slice.cache_id(),
                    slice.dfa(),
                );
                if let Some(started) = proof_started {
                    eprintln!(
                        "[glrmask/profile][proof_phase] kind=quotient slice={} terminal={} source={} result={:?} ms={:.3}",
                        slice_index,
                        terminal,
                        source,
                        quotient_proof,
                        started.elapsed().as_secs_f64() * 1e3,
                    );
                }
                if quotient_proof == Some(true) {
                    if profile_mask {
                        eprintln!(
                            "[glrmask/profile][precollapse_master_proof] slice={} source={} terminal={} proof=quotient",
                            slice_index, source, terminal,
                        );
                    }
                    outcomes[slice_index] = true;
                    break 'sources;
                }
            }
        }
    }

    if !outcomes[LLG_SAFE_PLUS_SLICE] {
        let max_vocab_safe_chars = u32::from(vocab.llg_master_max_safe_chars());
        let mut answers = Vec::<Option<u32>>::new();
        for (source, terminals) in &safe_radius_candidates {
                for &terminal in terminals {
                    let projected_started = profile_proof.then(std::time::Instant::now);
                    let projected_radius = vocab
                        .prepared_safe_radius(*source, terminal)
                        .map(u32::from)
                        .or_else(|| {
                            vocab.projected_terminal_slice_repeat_radius(
                                terminal,
                                *source,
                                safe_plus_slice.cache_id(),
                                safe_plus_slice.dfa(),
                                max_vocab_safe_chars,
                                16 * 1024,
                            )
                        });
                    if let Some(started) = projected_started {
                        eprintln!(
                            "[glrmask/profile][proof_phase] kind=projected_radius terminal={} source={} result={:?} ms={:.3}",
                            terminal,
                            source,
                            projected_radius,
                            started.elapsed().as_secs_f64() * 1e3,
                        );
                    }
                    let virtual_started = profile_proof.then(std::time::Instant::now);
                    let virtual_radius = virtual_residual_safe_repeat_radius(
                        &state.constraint.tokenizer,
                        *source,
                        terminal,
                        safe_plus_slice.dfa(),
                        max_vocab_safe_chars,
                        16 * 1024,
                    );
                    if let Some(started) = virtual_started {
                        eprintln!(
                            "[glrmask/profile][proof_phase] kind=virtual_radius terminal={} source={} result={:?} ms={:.3}",
                            terminal,
                            source,
                            virtual_radius,
                            started.elapsed().as_secs_f64() * 1e3,
                        );
                    }
                    let radius = match (projected_radius, virtual_radius) {
                        (Some(left), Some(right)) => Some(left.max(right)),
                        (left, right) => left.or(right),
                    };
                    answers.push(radius);
                }
            }
        let safe_radius = answers
            .into_iter()
            .flatten()
            .filter_map(|radius| u16::try_from(radius).ok())
            .max()
            .unwrap_or(0);
        let decision = LlgMasterDecision {
            safe_radius,
            whitespace: outcomes[LLG_WHITESPACE_SLICE],
        };
        return (!decision.is_empty()).then_some(decision);
    }

    let safe_radius = u16::MAX;
    let decision = LlgMasterDecision {
        safe_radius,
        whitespace: outcomes[LLG_WHITESPACE_SLICE],
    };
    (!decision.is_empty()).then_some(decision)
}

/// Execute a finite projection whose only live epsilon structure is a reset
/// dispatcher directly over its raw scalar component rows. The dispatcher
/// closure and any other multi-state root config are represented by the same
/// exact lazy-union table already used for deterministic same-parser unions.
///
/// This removes compile-time whole-product determinization without changing
/// the walk language. If the bounded derived-state cache cannot represent the
/// observed execution, return `false`; the caller reruns through the ordinary
/// exact NFA configuration walker.
#[allow(clippy::too_many_arguments)]
pub(super) fn try_scalar_dispatch(
    state: &ConstraintState<'_>,
    vocab: &DynamicMaskVocab,
    trie: &DynamicMaskTrie,
    root_branches: &DynamicBranches,
    lexer_scan_cache: &mut DynamicNfaScanCache<'_>,
    buf: &mut [u32],
    transitions16: Option<&[u16]>,
    transitions32: Option<&[u32]>,
    finalizer_code: Option<&[u32]>,
    single_finalizer_continues: Option<&[u8]>,
    ignored_output: Option<&[u32]>,
) -> Result<bool, String> {
    let tokenizer = lexer_scan_cache.tokenizer();
    if !tokenizer.has_scalar_deterministic_dispatch() {
        return Ok(false);
    }
    let profile = dynamic_mask_profile_enabled(state.generation);

    // A pending token-start guard is an exact output filter: its remembered
    // lexer/terminal pairs reject precisely those vocabulary tokens whose
    // bytes produce a blocked match. The predicate is deliberately independent
    // of parser resets during the candidate token. Factor it out of traversal
    // so each correlated root can use the ordinary Passed fast path (including
    // master slicing), then subtract that root's exact blocked-token mask.
    // Root branches are language alternatives, so OR the filtered root masks.
    if root_branches
        .iter()
        .any(|branch| !branch.initial_prune_guard.is_passed())
        && !joint_initial_guard_walk_enabled(root_branches)
    {
        let mut merged = vec![0u32; buf.len()];
        let mut scratch = vec![0u32; buf.len()];
        for branch in root_branches {
            scratch.fill(0);
            let mut one = DynamicBranches::new();
            let blocked = branch
                .initial_prune_guard
                .blocked_output_mask(state.constraint, buf.len())?;
            let mut unguarded = branch.clone();
            unguarded.initial_prune_guard = InitialPruneGuard::Passed;
            one.push(unguarded);
            if !try_scalar_dispatch(
                state,
                vocab,
                trie,
                &one,
                lexer_scan_cache,
                &mut scratch,
                transitions16,
                transitions32,
                finalizer_code,
                single_finalizer_continues,
                ignored_output,
            )? {
                return Ok(false);
            }
            if let Some(blocked) = blocked {
                for (word, &blocked) in scratch.iter_mut().zip(blocked.iter()) {
                    *word &= !blocked;
                }
            }
            for (dst, &word) in merged.iter_mut().zip(&scratch) {
                *dst |= word;
            }
        }
        if profile {
            eprintln!(
                "[glrmask/profile][scalar_dispatch] factored_pending_roots={}",
                root_branches.len(),
            );
        }
        buf.copy_from_slice(&merged);
        return Ok(true);
    }

    // The exact containment proofs below can be much more expensive than a
    // sparse vocabulary walk. Use the already-materialized finite mask
    // projection as a necessary-condition gate: if the union of immediate
    // bytes from the current root configs cannot even cover a slice's
    // first-byte language, that slice cannot help this mask. This gate may
    // conservatively skip an optimization; it never admits a token.
    let master_may_apply = if root_branches.iter().all(|branch| branch.initial_prune_guard.is_passed())
        && vocab.llg_master_trie().is_some()
        && let (Some(safe_plus), Some(whitespace)) = (
            vocab.llg_slice_by_cache_id(LLG_SAFE_PLUS_SLICE as u32),
            vocab.llg_slice_by_cache_id(LLG_WHITESPACE_SLICE as u32),
        )
    {
        let mut root_first_bytes = U8Set::empty();
        let mut physical = SmallVec::<[u32; 8]>::new();
        for branch in root_branches {
            lexer_scan_cache.physical_states_for_config(branch.tokenizer_config, &mut physical)?;
            for &raw_state in &physical {
                if tokenizer.state_has_epsilon_transitions(raw_state) {
                    continue;
                }
                if let Some(transitions) = transitions16 {
                    let row = (raw_state as usize).wrapping_mul(256);
                    if row + 256 > transitions.len() {
                        continue;
                    }
                    for byte in 0u16..=255 {
                        if unsafe { *transitions.get_unchecked(row + byte as usize) } != u16::MAX {
                            root_first_bytes.insert(byte as u8);
                        }
                    }
                } else if let Some(transitions) = transitions32 {
                    let row = (raw_state as usize).wrapping_mul(256);
                    if row + 256 > transitions.len() {
                        continue;
                    }
                    for byte in 0u16..=255 {
                        if unsafe { *transitions.get_unchecked(row + byte as usize) } != u32::MAX {
                            root_first_bytes.insert(byte as u8);
                        }
                    }
                } else {
                    root_first_bytes |= tokenizer.state_first_bytes(raw_state);
                }
            }
        }
        safe_plus
            .first_bytes()
            .is_subset(&root_first_bytes)
            || whitespace.first_bytes().is_subset(&root_first_bytes)
    } else {
        false
    };
    let master_started = profile.then(std::time::Instant::now);
    // A single exact root does not need a pre-collapse certificate: no lexer
    // provenance is about to be lost, and the later single-root master path can
    // prove the same slice directly (including the cheap virtual bounded-radius
    // fast path). Reserve the expensive pre-collapse prover for actual unions.
    let skip_single_virtual_precollapse = master_may_apply
        && root_branches.len() == 1
        && root_branches[0].exact_tokenizer_state.is_some_and(|source| {
            state
                .constraint
                .tokenizer
                .singleton_epsilon_closure(source)
                .into_iter()
                .any(|residual_state| {
                    state
                        .constraint
                        .tokenizer
                        .virtual_residual_terminal_for_state(residual_state)
                        .is_some()
                })
        });
    let precollapse_master_decision = (master_may_apply
        && !skip_single_virtual_precollapse
        && std::env::var_os("GLRMASK_EXPERIMENT_DISABLE_PRECOLLAPSE_MASTER").is_none())
        .then(|| {
            precollapse_master_decision(
                state,
                vocab,
                root_branches,
                tokenizer.num_states() as usize,
            )
        })
        .flatten();
    if let Some(started) = master_started {
        eprintln!(
            "[glrmask/profile][scalar_dispatch_precollapse_master] eligible={} hit={} elapsed_ms={:.3}",
            master_may_apply,
            precollapse_master_decision.is_some(),
            started.elapsed().as_secs_f64() * 1e3,
        );
    }
    // Carry the certificate, not the replacement trie. The inner walk first
    // compares the certified residual work against this ordinary trie and
    // selects the master only if profitable. Replacing it here makes that
    // comparison use the larger master as its own baseline and also leaves
    // the wrong trie in place when the inner gate declines the certificate.
    let Some(reset_states) = tokenizer.sorted_deterministic_dispatch_roots() else {
        return Ok(false);
    };
    let extension = if transitions16.is_some()
        && let Some(cached) = vocab.cached_dense_subset16(&reset_states)
    {
        if profile {
            eprintln!(
                "[glrmask/profile][scalar_dispatch] cache=hit roots={} rows={}",
                reset_states.len(),
                cached.rows.len(),
            );
        }
        cached
    } else {
        // Do not speculatively determinize the complete dispatcher subset
        // graph here. The vocabulary walk observes only a small portion of
        // that graph, while full construction can cost many milliseconds or
        // exceed the dense extension reserve. The exact lazy table below
        // materializes only derivatives actually requested by trie edges.
        let Some(subset_cache) = vocab.try_lock_lazy_union_cache() else {
            return Ok(false);
        };
        let overflowed = std::cell::Cell::new(false);
        let use_pair_map = !std::ptr::eq(tokenizer, state.constraint.tokenizer.as_ref());
        let Some((lazy_transitions, initial_lexer_state)) = FullWalkLazyUnion::new(
            tokenizer,
            transitions16,
            transitions32,
            subset_cache,
            &reset_states,
            &overflowed,
            use_pair_map,
        ) else {
            return Ok(false);
        };

        let mut physical = SmallVec::<[u32; 8]>::new();
        let mut transformed = DynamicBranches::new();
        for branch in root_branches {
            lexer_scan_cache.physical_states_for_config(branch.tokenizer_config, &mut physical)?;
            physical.retain(|tokenizer_state| {
                let tokenizer_state = *tokenizer_state;
                if !tokenizer.state_has_epsilon_transitions(tokenizer_state) {
                    return true;
                }
                debug_assert!(tokenizer.transitions_from(tokenizer_state).next().is_none());
                false
            });
            physical.sort_unstable();
            physical.dedup();
            if physical.is_empty() {
                continue;
            }
            let tokenizer_config = match physical.as_slice() {
                [single] => *single,
                _ => {
                    let Some(state) = lazy_transitions.union_states(&physical) else {
                        return Ok(false);
                    };
                    state
                }
            };
            transformed.push(DynamicBranch {
                tokenizer_config,
                exact_tokenizer_state: branch.exact_tokenizer_state,
                parser_filtered_root: branch.parser_filtered_root,
                parser_filtered_transparent: branch.parser_filtered_transparent,
                residual_continuation_terminal: branch.residual_continuation_terminal,
                gss: branch.gss.clone(),
                shared_root_admission: branch.shared_root_admission.clone(),
                initial_prune_guard: branch.initial_prune_guard.clone(),
            });
        }
        if transformed.is_empty() {
            return Ok(false);
        }

        let mut scratch = vec![0u32; buf.len()];
        let result = if transformed.len() == 1 {
            try_full_walk_mask_with_table_from_initial_in_output_scope::<_, true>(
                state,
                vocab,
                trie,
                precollapse_master_decision,
                &transformed,
                lexer_scan_cache,
                &mut scratch,
                lazy_transitions,
                finalizer_code.unwrap_or(&[]),
                single_finalizer_continues.unwrap_or(&[]),
                initial_lexer_state,
                ignored_output,
            )
        } else {
            try_full_walk_mask_with_table_from_initial_in_output_scope::<_, false>(
                state,
                vocab,
                trie,
                precollapse_master_decision,
                &transformed,
                lexer_scan_cache,
                &mut scratch,
                lazy_transitions,
                finalizer_code.unwrap_or(&[]),
                single_finalizer_continues.unwrap_or(&[]),
                initial_lexer_state,
                ignored_output,
            )
        }?;
        if overflowed.get() {
            let mut cache = vocab.lock_lazy_union_cache();
            FullWalkLazyUnion::clear_cache(&mut cache);
            return Ok(false);
        }
        if !result {
            return Ok(false);
        }
        if profile {
            eprintln!("[glrmask/profile][scalar_dispatch] cache=lazy_walk_success");
        }
        buf.copy_from_slice(&scratch);
        return Ok(true);
    };
    let initial_lexer_state = extension.root_state;
    let table = FullWalkCachedSubset16 {
        base_transitions: transitions16.expect("dense subset cache requires Flat16 base transitions"),
        extension: Arc::clone(&extension),
    };

    let transform_started = profile.then(std::time::Instant::now);
    let mut physical = SmallVec::<[u32; 8]>::new();
    let mut transformed = DynamicBranches::new();
    for branch in root_branches {
        lexer_scan_cache.physical_states_for_config(branch.tokenizer_config, &mut physical)?;
        physical.retain(|tokenizer_state| {
            let tokenizer_state = *tokenizer_state;
            if !tokenizer.state_has_epsilon_transitions(tokenizer_state) {
                return true;
            }
            // Under `has_scalar_deterministic_dispatch`, the only byte-live
            // epsilon source reachable from reset is the dispatcher itself,
            // and it has no consuming row. Its closure members are already in
            // this config, so retaining it would add no byte-language behavior.
            debug_assert!(tokenizer.transitions_from(tokenizer_state).next().is_none());
            false
        });
        physical.sort_unstable();
        physical.dedup();
        if physical.is_empty() {
            continue;
        }
        let tokenizer_config = match physical.as_slice() {
            [single] => *single,
            _ => {
                let Some(index) = extension
                    .subsets
                    .iter()
                    .position(|subset| subset.as_slice() == physical.as_slice())
                else {
                    if dynamic_mask_profile_enabled(state.generation) {
                        eprintln!(
                            "[glrmask/profile][scalar_dispatch] missing_root_subset size={}",
                            physical.len(),
                        );
                    }
                    return Ok(false);
                };
                extension.base_state_count + index as u32
            }
        };
        transformed.push(DynamicBranch {
            tokenizer_config,
            exact_tokenizer_state: branch.exact_tokenizer_state,
            parser_filtered_root: branch.parser_filtered_root,
            parser_filtered_transparent: branch.parser_filtered_transparent,
            residual_continuation_terminal: branch.residual_continuation_terminal,
            gss: branch.gss.clone(),
            shared_root_admission: branch.shared_root_admission.clone(),
            initial_prune_guard: branch.initial_prune_guard.clone(),
        });
    }
    if transformed.is_empty() {
        return Ok(false);
    }
    if let Some(started) = transform_started {
        eprintln!(
            "[glrmask/profile][scalar_dispatch_transform] branches={} elapsed_ms={:.3}",
            transformed.len(),
            started.elapsed().as_secs_f64() * 1e3,
        );
    }

    // Run into scratch so an unexpected structural decline can still fall back
    // to the ordinary NFA walker without exposing a partial mask.
    let walk_started = profile.then(std::time::Instant::now);
    let mut scratch = vec![0u32; buf.len()];
    let result = if transformed.len() == 1 {
        try_full_walk_mask_with_table_from_initial_in_output_scope::<_, true>(
            state,
            vocab,
            trie,
            precollapse_master_decision,
            &transformed,
            lexer_scan_cache,
            &mut scratch,
            table,
            finalizer_code.expect("dense subset cache requires Flat16 finalizer metadata"),
            single_finalizer_continues
                .expect("dense subset cache requires Flat16 continuation metadata"),
            initial_lexer_state,
            ignored_output,
        )
    } else {
        try_full_walk_mask_with_table_from_initial_in_output_scope::<_, false>(
            state,
            vocab,
            trie,
            precollapse_master_decision,
            &transformed,
            lexer_scan_cache,
            &mut scratch,
            table,
            finalizer_code.expect("dense subset cache requires Flat16 finalizer metadata"),
            single_finalizer_continues
                .expect("dense subset cache requires Flat16 continuation metadata"),
            initial_lexer_state,
            ignored_output,
        )
    }?;
    if let Some(started) = walk_started {
        eprintln!(
            "[glrmask/profile][scalar_dispatch_inner_walk] elapsed_ms={:.3}",
            started.elapsed().as_secs_f64() * 1e3,
        );
    }
    if !result {
        if dynamic_mask_profile_enabled(state.generation) {
            eprintln!("[glrmask/profile][scalar_dispatch] walk_declined");
        }
        return Ok(false);
    }
    if profile {
        eprintln!("[glrmask/profile][scalar_dispatch] walk=success");
    }
    buf.copy_from_slice(&scratch);
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn try_flat16<const HOT_SINGLE_ROOT: bool>(
    state: &ConstraintState<'_>,
    vocab: &DynamicMaskVocab,
    trie: &DynamicMaskTrie,
    root_branches: &DynamicBranches,
    lexer_scan_cache: &mut DynamicNfaScanCache<'_>,
    buf: &mut [u32],
    transitions: &[u16],
    finalizer_code: &[u32],
    single_finalizer_continues: &[u8],
) -> Result<bool, String> {
    // A token-start maximal-munch guard is an output filter over the whole
    // candidate model token. Carrying one pending guard inside a multi-root
    // deterministic walk prevents the pre-collapse master proof from applying
    // to every root and can turn an otherwise sliced broad mask into a full
    // vocabulary traversal. Factor each correlated root exactly as in the
    // scalar-dispatch path: evaluate the root with the guard marked Passed,
    // subtract the immutable blocked-token mask, then union root languages.
    if root_branches
        .iter()
        .any(|branch| !branch.initial_prune_guard.is_passed())
        && !joint_initial_guard_walk_enabled(root_branches)
    {
        let profile = dynamic_mask_profile_enabled(state.generation);
        let mut merged = vec![0u32; buf.len()];
        let mut scratch = vec![0u32; buf.len()];
        for branch in root_branches {
            scratch.fill(0);
            let blocked_started = profile.then(std::time::Instant::now);
            let blocked = branch
                .initial_prune_guard
                .blocked_output_mask(state.constraint, buf.len())?;
            if let Some(started) = blocked_started {
                eprintln!(
                    "[glrmask/profile][flat16_pending_filter] build_ms={:.3} words={}",
                    started.elapsed().as_secs_f64() * 1e3,
                    blocked.as_ref().map_or(0, |mask| mask.len()),
                );
            }

            let mut one = DynamicBranches::new();
            let mut unguarded = branch.clone();
            unguarded.initial_prune_guard = InitialPruneGuard::Passed;
            one.push(unguarded);
            let root_started = profile.then(std::time::Instant::now);
            if !try_flat16::<HOT_SINGLE_ROOT>(
                state,
                vocab,
                trie,
                &one,
                lexer_scan_cache,
                &mut scratch,
                transitions,
                finalizer_code,
                single_finalizer_continues,
            )? {
                return Ok(false);
            }
            if let Some(started) = root_started {
                eprintln!(
                    "[glrmask/profile][flat16_pending_filter] root_walk_ms={:.3}",
                    started.elapsed().as_secs_f64() * 1e3,
                );
            }
            if let Some(blocked) = blocked {
                for (word, &blocked) in scratch.iter_mut().zip(blocked.iter()) {
                    *word &= !blocked;
                }
            }
            for (dst, &word) in merged.iter_mut().zip(&scratch) {
                *dst |= word;
            }
        }
        buf.copy_from_slice(&merged);
        return Ok(true);
    }

    let profile_flat16 = dynamic_mask_flat16_profile_enabled();
    let proof_started = profile_flat16.then(std::time::Instant::now);
    // A single root cannot lose exact-source provenance through root
    // collapsing. Let the ordinary one-root proof below use the active dense
    // transition table directly before considering the heavier projected
    // quotient. Pre-collapse is only needed when multiple roots may be merged
    // into a coordinate that no longer identifies one exact source state.
    let precollapse_master_decision = (root_branches.len() >= 2
        && root_branches.iter().all(|branch| branch.initial_prune_guard.is_passed())
        && std::env::var_os("GLRMASK_EXPERIMENT_DISABLE_PRECOLLAPSE_MASTER").is_none()).then(|| {
        precollapse_master_decision(
            state,
            vocab,
            root_branches,
            lexer_scan_cache.tokenizer().num_states() as usize,
        )
    }).flatten();
    if let Some(started) = proof_started {
        eprintln!(
            "[glrmask/profile][flat16_phases] proof_ms={:.3} proof_hit={}",
            started.elapsed().as_secs_f64() * 1e3,
            precollapse_master_decision.is_some(),
        );
    }
    // Preserve the ordinary trie until the inner profitability gate accepts
    // the pre-collapse certificate; see the scalar-dispatch path above.
    if lexer_scan_cache.subset_union_requested
        && root_branches.len() >= 2
        && root_branches.iter().all(|branch| branch.initial_prune_guard.is_passed())
        && root_branches
            .iter()
            .skip(1)
            .all(|branch| branch.gss.ptr_eq(&root_branches[0].gss))
    {
        let mut root_states = root_branches
            .iter()
            .map(|branch| branch.tokenizer_config)
            .collect::<Vec<_>>();
        root_states.sort_unstable();
        root_states.dedup();
        if let Some((extension, root_state)) = vocab.cached_dense_subset16_state_for_subset(&root_states) {
            let subset_transitions = FullWalkCachedSubset16 { base_transitions: transitions, extension };
            let mut collapsed = DynamicBranches::new();
            collapsed.push(DynamicBranch {
                tokenizer_config: root_state,
                exact_tokenizer_state: None,
                parser_filtered_root: false,
                parser_filtered_transparent: false,
                residual_continuation_terminal: None,
                gss: root_branches[0].gss.clone(),
                shared_root_admission: root_branches[0].shared_root_admission.clone(),
                initial_prune_guard: InitialPruneGuard::Passed,
            });
            return try_full_walk_mask_with_table::<_, true>(
                state, vocab, trie, precollapse_master_decision, &collapsed, lexer_scan_cache, buf, subset_transitions, finalizer_code, single_finalizer_continues,
            );
        }
        if (2..=4).contains(&root_states.len()) {
            let Some(subset_cache) = vocab.try_lock_lazy_union_cache() else {
                // The persistent lazy-subset cache is shared by all sequences
                // using this constraint. Never serialize concurrent mask fills
                // behind that derived acceleration state: contention falls back
                // to the ordinary exact multi-root walker.
                return try_full_walk_mask_with_table::<_, HOT_SINGLE_ROOT>(
                    state,
                    vocab,
                    trie,
                    precollapse_master_decision,
                    root_branches,
                    lexer_scan_cache,
                    buf,
                    FullWalkFlat16 { transitions },
                    finalizer_code,
                    single_finalizer_continues,
                );
            };
            let overflowed = std::cell::Cell::new(false);
            if let Some((subset_transitions, root_state)) = FullWalkLazyUnion::new(
                lexer_scan_cache.tokenizer(),
                Some(transitions),
                None,
                subset_cache,
                &root_states,
                &overflowed,
                !std::ptr::eq(lexer_scan_cache.tokenizer(), state.constraint.tokenizer.as_ref()),
            ) {
                let mut collapsed = DynamicBranches::new();
                collapsed.push(DynamicBranch {
                    tokenizer_config: root_state,
                    exact_tokenizer_state: None,
                    parser_filtered_root: false,
                    parser_filtered_transparent: false,
                    residual_continuation_terminal: None,
                    gss: root_branches[0].gss.clone(),
                    shared_root_admission: root_branches[0].shared_root_admission.clone(),
                    initial_prune_guard: InitialPruneGuard::Passed,
                });
                let result = try_full_walk_mask_with_table::<_, true>(
                    state,
                    vocab,
                    trie,
                    precollapse_master_decision,
                    &collapsed,
                    lexer_scan_cache,
                    buf,
                    subset_transitions,
                    finalizer_code,
                    single_finalizer_continues,
                );
                if !overflowed.get() {
                    return result;
                }
                // Do not carry a saturated partial interner into the next
                // mask. The retry below does not use this cache, so clearing
                // here affects only future lazy-union attempts.
                {
                    let mut cache = vocab.lock_lazy_union_cache();
                    FullWalkLazyUnion::clear_cache(&mut cache);
                }
                // The lazy execution extension is deliberately bounded so
                // parser-node boundary caches can stay dense. If one mask
                // discovers more subset states than that execution reserve,
                // discard the partial mask and rerun through the ordinary
                // exact multi-root Flat16 walker instead of making the cache
                // capacity a correctness limit.
                return try_full_walk_mask_with_table::<_, HOT_SINGLE_ROOT>(
                    state,
                    vocab,
                    trie,
                    precollapse_master_decision,
                    root_branches,
                    lexer_scan_cache,
                    buf,
                    FullWalkFlat16 { transitions },
                    finalizer_code,
                    single_finalizer_continues,
                );
            }
        }
        let extension = if let Some(cached) = vocab.cached_dense_subset16(&root_states) {
            Some(cached)
        } else {
            {
                let started = std::time::Instant::now();
                let built = FullWalkSubset16::build(
                    lexer_scan_cache.tokenizer(),
                    transitions,
                    &root_states,
                    vocab.max_token_byte_len(),
                );
                if dynamic_mask_profile_enabled(state.generation) {
                    eprintln!(
                        "[glrmask/profile][dense_subset16_build] roots={} rows={} elapsed_ms={:.3}",
                        root_states.len(),
                        built.as_ref().map_or(0, |(table, _, _)| table.rows.len()),
                        started.elapsed().as_secs_f64() * 1e3,
                    );
                }
                built.map(|(built, root_state, subsets)| {
                    vocab.cache_dense_subset16(
                        root_states.clone(),
                        built.into_owned(root_state, subsets),
                    )
                })
            }
        };
        if let Some(extension) = extension {
            let root_state = extension.root_state;
            let subset_transitions = FullWalkCachedSubset16 {
                base_transitions: transitions,
                extension,
            };
            let mut collapsed = DynamicBranches::new();
            collapsed.push(DynamicBranch {
                tokenizer_config: root_state,
                exact_tokenizer_state: None,
                parser_filtered_root: false,
                parser_filtered_transparent: false,
                residual_continuation_terminal: None,
                gss: root_branches[0].gss.clone(),
                shared_root_admission: root_branches[0].shared_root_admission.clone(),
                initial_prune_guard: InitialPruneGuard::Passed,
            });
            return try_full_walk_mask_with_table::<_, true>(
                state,
                vocab,
                trie,
                None,
                &collapsed,
                lexer_scan_cache,
                buf,
                subset_transitions,
                finalizer_code,
                single_finalizer_continues,
            );
        }
    }
    let walk_started = profile_flat16.then(std::time::Instant::now);
    let result = try_full_walk_mask_with_table::<_, HOT_SINGLE_ROOT>(
        state,
        vocab,
        trie,
        precollapse_master_decision,
        root_branches,
        lexer_scan_cache,
        buf,
        FullWalkFlat16 { transitions },
        finalizer_code,
        single_finalizer_continues,
    );
    if let Some(started) = walk_started {
        eprintln!(
            "[glrmask/profile][flat16_phases] walk_ms={:.3}",
            started.elapsed().as_secs_f64() * 1e3,
        );
    }
    result
}

#[allow(clippy::too_many_arguments)]
pub(super) fn try_flat32<const HOT_SINGLE_ROOT: bool>(
    state: &ConstraintState<'_>,
    vocab: &DynamicMaskVocab,
    trie: &DynamicMaskTrie,
    root_branches: &DynamicBranches,
    lexer_scan_cache: &mut DynamicNfaScanCache<'_>,
    buf: &mut [u32],
    transitions: &[u32],
    finalizer_code: &[u32],
    single_finalizer_continues: &[u8],
) -> Result<bool, String> {
    try_full_walk_mask_with_table::<_, HOT_SINGLE_ROOT>(
        state,
        vocab,
        trie,
        None,
        root_branches,
        lexer_scan_cache,
        buf,
        FullWalkFlat32 { transitions },
        finalizer_code,
        single_finalizer_continues,
    )
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum FullWalkPruneGuard {
    Passed,
    Pending(SmallVec<[(u32, TerminalID); 2]>),
}

impl FullWalkPruneGuard {
    fn from_initial(guard: &InitialPruneGuard, vocab: &DynamicMaskVocab) -> Self {
        match guard {
            InitialPruneGuard::Passed => Self::Passed,
            InitialPruneGuard::Pending { memories } => {
                Self::Pending(
                    memories
                        .iter()
                        .map(|&(state, _, terminal)| (state, terminal))
                        .collect(),
                )
            }
        }
    }

    #[inline(always)]
    fn is_passed(&self) -> bool {
        matches!(self, Self::Passed)
    }

    /// Advance the maximal-munch guard in the same deterministic lexer
    /// coordinate as the direct full walk. This is only exercised by the slow
    /// side branch; the dominant scalar path always has `Passed`.
    fn advance<T: FullWalkTransitionTable>(
        &self,
        tokenizer: &Tokenizer,
        transitions: &T,
        byte: u8,
    ) -> Option<Self> {
        let Self::Pending(memories) = self else {
            return Some(Self::Passed);
        };
        let mut next = SmallVec::<[(u32, TerminalID); 2]>::new();
        for &(lexer_state, terminal) in memories {
            let target = transitions.transition(lexer_state, byte);
            if target == u32::MAX {
                continue;
            }
            if transitions.matched_terminals(tokenizer, target).contains(&terminal) {
                return None;
            }
            if transitions.future_contains(tokenizer, target, terminal)
                && !next.contains(&(target, terminal))
            {
                next.push((target, terminal));
            }
        }
        if next.is_empty() {
            Some(Self::Passed)
        } else {
            Some(Self::Pending(next))
        }
    }

    fn remember_terminal_match<T: FullWalkTransitionTable>(
        &self,
        tokenizer: &Tokenizer,
        transitions: &T,
        lexer_state: u32,
        terminal: TerminalID,
    ) -> Self {
        if !transitions.future_contains(tokenizer, lexer_state, terminal) {
            return self.clone();
        }
        let mut memories = match self {
            Self::Passed => SmallVec::new(),
            Self::Pending(memories) => memories.clone(),
        };
        if !memories.contains(&(lexer_state, terminal)) {
            memories.push((lexer_state, terminal));
        }
        Self::Pending(memories)
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct FullWalkBranch {
    lexer_state: u32,
    parser_node: u32,
    prune_guard: FullWalkPruneGuard,
}

type FullWalkBranches = SmallVec<[FullWalkBranch; 4]>;

#[derive(Clone, Copy, PartialEq, Eq)]
struct FullWalkGuardedPair {
    continuing_lexer: u32,
    continuing_parser: u32,
    pending_parser: u32,
    guard_terminal: TerminalID,
}

impl FullWalkGuardedPair {
    #[inline(always)]
    fn pack(self) -> ((u32, u32), (u32, u32)) {
        (
            (self.continuing_lexer, self.continuing_parser),
            (self.pending_parser, self.guard_terminal),
        )
    }

    #[inline(always)]
    fn unpack(packed: ((u32, u32), (u32, u32))) -> Self {
        Self {
            continuing_lexer: packed.0.0,
            continuing_parser: packed.0.1,
            pending_parser: packed.1.0,
            guard_terminal: packed.1.1,
        }
    }
}

#[inline(always)]
fn full_walk_guarded_pair_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("GLRMASK_DISABLE_GUARDED_BOUND_CONTINUATION").is_none()
    })
}

#[inline(always)]
fn full_walk_guarded_pair_from_branches(
    branches: &FullWalkBranches,
    initial_lexer_state: u32,
) -> Option<FullWalkGuardedPair> {
    if !full_walk_guarded_pair_enabled() {
        return None;
    }
    let [first, second] = branches.as_slice() else {
        return None;
    };
    let (pending, continuing, memories) = match (&first.prune_guard, &second.prune_guard) {
        (FullWalkPruneGuard::Pending(memories), FullWalkPruneGuard::Passed) => {
            (first, second, memories)
        }
        (FullWalkPruneGuard::Passed, FullWalkPruneGuard::Pending(memories)) => {
            (second, first, memories)
        }
        _ => return None,
    };
    let [(guard_lexer, guard_terminal)] = memories.as_slice() else {
        return None;
    };
    if pending.lexer_state != initial_lexer_state || *guard_lexer != continuing.lexer_state {
        return None;
    }
    Some(FullWalkGuardedPair {
        continuing_lexer: continuing.lexer_state,
        continuing_parser: continuing.parser_node,
        pending_parser: pending.parser_node,
        guard_terminal: *guard_terminal,
    })
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum FullWalkManyState {
    Branches(FullWalkBranches),
    ThreeSameParser {
        lexers: (u32, u32, u32),
        parser_node: u32,
    },
}

/// Exact identity alphabet for one scalar parser/lexer pair. Scalar lanes
/// have no pending maximal-munch guard; every other lane uses a different
/// representation. The certificate is learned only from the complete result
/// of the authoritative byte executor and cannot outlive its mask walk.
#[derive(Default)]
struct FullWalkScalarIdentity {
    state: Option<(u32, u32)>,
    bytes: [u64; 4],
    proof_checked: bool,
}

impl FullWalkScalarIdentity {
    #[inline]
    fn remember(&mut self, source: (u32, u32), target: (u32, u32), byte: u8) {
        if source != target {
            return;
        }
        if self.state != Some(source) {
            self.state = Some(source);
            self.bytes = [0; 4];
            self.proof_checked = false;
        }
        self.bytes[byte as usize >> 6] |= 1u64 << (byte & 63);
    }

    #[inline]
    fn alphabet(&self, state: (u32, u32)) -> Option<&[u64; 4]> {
        (self.state == Some(state)).then_some(&self.bytes)
    }
}

/// Lexer-only evidence used to strengthen an already established *whole*
/// parser/lexer identity. Ordinary and finalizing edges are kept separate:
/// the latter require the guarded-pair shortcut's exact terminal condition.
#[derive(Clone, Copy, Default)]
struct FullWalkIdentityByteClasses {
    ordinary: [u64; 4],
    finalizing: [u64; 4],
}

impl FullWalkIdentityByteClasses {
    fn intersect_union_member(&mut self, member: Self) {
        for word in 0..4 {
            let self_any = self.ordinary[word] | self.finalizing[word];
            let member_any = member.ordinary[word] | member.finalizing[word];
            self.ordinary[word] &= member.ordinary[word];
            self.finalizing[word] = (self_any & member_any) & !self.ordinary[word];
        }
    }
}

fn full_walk_identity_byte_classes(
    state: u32,
    mut transition: impl FnMut(u8) -> Option<(u32, bool)>,
) -> FullWalkIdentityByteClasses {
    let mut classes = FullWalkIdentityByteClasses::default();
    for byte in 0..=u8::MAX {
        let Some((target, finalizing)) = transition(byte) else { continue; };
        if target != state { continue; }
        let words = if finalizing { &mut classes.finalizing } else { &mut classes.ordinary };
        words[byte as usize >> 6] |= 1u64 << (byte & 63);
    }
    classes
}

#[derive(Default)]
struct FullWalkIdentityProofCache {
    rows: FxHashMap<u32, Option<FullWalkIdentityByteClasses>>,
}

/// A marker is irrelevant only when *every* original token alias is outside
/// this root's output responsibility. Looking at a canonical representative
/// alone would be unsound when aliases span multiple output words.
fn full_walk_marker_fully_ignored(
    marker: u64,
    ignored: &[u32],
    aliases_ignored: impl FnOnce(u32) -> bool,
) -> bool {
    if marker & DYNAMIC_TOKEN_MARKER_FALLBACK == 0 {
        let word = (marker >> 32) as usize;
        let bits = marker as u32;
        return ignored.get(word).is_some_and(|value| value & bits == bits);
    }
    let Some(canonical) = (marker & !DYNAMIC_TOKEN_MARKER_FALLBACK)
        .checked_sub(1).and_then(|id| u32::try_from(id).ok())
    else {
        return false;
    };
    aliases_ignored(canonical)
}

/// Exact prefix counts in the actual walk's canonical endpoint order. This
/// index is local to one root mask, not keyed by parser states or persisted.
#[cfg(test)]
struct FullWalkOutputScope {
    required_prefix: Vec<u32>,
}

#[cfg(test)]
std::thread_local! {
    pub(super) static TEST_OUTPUT_SCOPE_SKIPPED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}


/// Lazy equivalent of the scope prefix index for preorder marker intervals.
/// The next responsible marker refutes every containing interval without
/// re-scanning it. As starts advance, each marker is inspected at most once;
/// markers already bypassed by an exact lexer/parser proof are never inspected.
/// Backward queries reset the certificate rather than assuming caller order.
#[derive(Default)]
struct FullWalkOutputScopeCursor {
    last_start: usize,
    ignored_through: usize,
    next_required: Option<usize>,
}

impl FullWalkOutputScopeCursor {
    fn covers(
        &mut self,
        markers: &[u64],
        start: usize,
        end: usize,
        mut is_ignored: impl FnMut(u64) -> bool,
    ) -> bool {
        if start > end || end > markers.len() { return false; }
        if start < self.last_start { *self = Self::default(); }
        self.last_start = start;
        if start == end { return true; }
        if let Some(required) = self.next_required {
            if required >= start { return required >= end; }
            self.next_required = None;
        }
        for index in start.max(self.ignored_through)..end {
            if !is_ignored(markers[index]) {
                self.ignored_through = index;
                self.next_required = Some(index);
                return false;
            }
        }
        self.ignored_through = self.ignored_through.max(end);
        true
    }
}

#[cfg(test)]
impl FullWalkOutputScope {
    fn new(markers: &[u64], mut is_ignored: impl FnMut(u64) -> bool) -> Option<Self> {
        const MAX_MARKERS: usize = 1 << 20;
        if markers.len() > MAX_MARKERS { return None; }
        let mut required_prefix = Vec::new();
        required_prefix.try_reserve_exact(markers.len().checked_add(1)?).ok()?;
        let mut required = 0u32;
        required_prefix.push(required);
        for &marker in markers {
            required += u32::from(!is_ignored(marker));
            required_prefix.push(required);
        }
        Some(Self { required_prefix })
    }

    fn covers(&self, start: usize, end: usize) -> bool {
        start <= end
            && self.required_prefix.get(start).zip(self.required_prefix.get(end))
                .is_some_and(|(left, right)| left == right)
    }
}

impl FullWalkIdentityProofCache {
    fn get<T: FullWalkTransitionTable>(
        &mut self,
        transitions: &T,
        tokenizer: &Tokenizer,
        lexer: u32,
    ) -> Option<FullWalkIdentityByteClasses> {
        if let Some(&classes) = self.rows.get(&lexer) { return classes; }
        // At most 64 local row proofs, with no persistent or serialized state.
        if self.rows.len() >= 64 { return None; }
        let classes = transitions.proven_identity_byte_classes(tokenizer, lexer);
        self.rows.insert(lexer, classes);
        classes
    }
}


/// A passed continuing branch survives each self-edge even when finalizers
/// fork other alternatives. Exact physical liveness therefore witnesses every
/// descendant of an alphabet-closed subtree; other branches may change.
#[allow(clippy::too_many_arguments)]
fn full_walk_live_branch_witness<T: FullWalkTransitionTable>(
    branch: &FullWalkBranch,
    alphabet: [u64; 4],
    proofs: &mut FullWalkIdentityProofCache,
    parser_cache: &mut FullWalkParserCache,
    constraint: &Constraint,
    tokenizer: &Tokenizer,
    transitions: &T,
) -> bool {
    if !branch.prune_guard.is_passed() { return false; }
    let Some(classes) = proofs.get(transitions, tokenizer, branch.lexer_state) else {
        return false;
    };
    let self_edges = std::array::from_fn(|word|
        classes.ordinary[word] | classes.finalizing[word]);
    identity_alphabet_covers_subtree(&self_edges, alphabet)
        && parser_cache.physical_token_boundary_allowed(
            constraint, tokenizer, transitions, branch.parser_node, branch.lexer_state,
        )
}

/// Bound proof work independently of whole-frontier identity. No frontier
/// mutation or assumption about missing rows is permitted on decline.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn full_walk_many_live_witness<T: FullWalkTransitionTable>(
    frontier: &FullWalkManyState,
    alphabet: [u64; 4],
    proofs: &mut FullWalkIdentityProofCache,
    parser_cache: &mut FullWalkParserCache,
    constraint: &Constraint,
    tokenizer: &Tokenizer,
    transitions: &T,
) -> bool {
    let mut witness = |branch: &FullWalkBranch| full_walk_live_branch_witness(
        branch, alphabet, proofs, parser_cache, constraint, tokenizer, transitions,
    );
    match frontier {
        FullWalkManyState::Branches(branches) => {
            branches.len() <= 8 && branches.iter().any(&mut witness)
        }
        FullWalkManyState::ThreeSameParser { lexers, parser_node } => {
            [lexers.0, lexers.1, lexers.2].into_iter().any(|lexer_state| witness(&FullWalkBranch {
                lexer_state, parser_node: *parser_node, prune_guard: FullWalkPruneGuard::Passed,
            }))
        }
    }
}

/// A passed union branch is existential, not an intersection: a live physical
/// member can keep matching while its siblings advance or disappear. The
/// continuing branch is retained by the reference executor even on finalizers.
/// Its parser coordinate is unchanged; no pending guard may use this helper.
#[allow(clippy::too_many_arguments)]
fn full_walk_live_component_witness<T: FullWalkTransitionTable>(
    lexer: u32,
    parser: u32,
    alphabet: [u64; 4],
    proofs: &mut FullWalkIdentityProofCache,
    parser_cache: &mut FullWalkParserCache,
    constraint: &Constraint,
    tokenizer: &Tokenizer,
    transitions: &T,
) -> bool {
    transitions.continuation_witness_components(lexer).into_iter().any(|lexer_state| {
        full_walk_live_branch_witness(&FullWalkBranch {
            lexer_state, parser_node: parser, prune_guard: FullWalkPruneGuard::Passed,
        }, alphabet, proofs, parser_cache, constraint, tokenizer, transitions)
    })
}

#[allow(clippy::too_many_arguments)]
fn full_walk_many_component_witness<T: FullWalkTransitionTable>(
    frontier: &FullWalkManyState,
    alphabet: [u64; 4],
    proofs: &mut FullWalkIdentityProofCache,
    parser_cache: &mut FullWalkParserCache,
    constraint: &Constraint,
    tokenizer: &Tokenizer,
    transitions: &T,
) -> bool {
    let mut witness = |lexer, parser| full_walk_live_component_witness(
        lexer, parser, alphabet, proofs, parser_cache, constraint, tokenizer, transitions);
    match frontier {
        FullWalkManyState::Branches(branches) => branches.len() <= 8
            && branches.iter().any(|branch| branch.prune_guard.is_passed()
                && witness(branch.lexer_state, branch.parser_node)),
        FullWalkManyState::ThreeSameParser { lexers, parser_node } =>
            [lexers.0, lexers.1, lexers.2].into_iter().any(|lexer| witness(lexer, *parser_node)),
    }
}

/// Only representations that prove their selected alternatives have Passed
/// guards can enter this view. The pending side of a guarded pair is omitted.
enum FullWalkWitnessFrontier<'a> {
    One(u32, u32),
    Two([(u32, u32); 2]),
    Many(&'a FullWalkManyState),
}

/// Keep bounded proof discovery out of the large byte-dispatch loop. Most
/// edges are too small to amortize a proof, and the ordinary walker remains
/// the exact fallback when any certificate or work-budget check declines.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn full_walk_frontier_live_witness<T: FullWalkTransitionTable>(
    frontier: FullWalkWitnessFrontier<'_>,
    alphabet: [u64; 4],
    proofs: &mut FullWalkIdentityProofCache,
    parser_cache: &mut FullWalkParserCache,
    constraint: &Constraint,
    tokenizer: &Tokenizer,
    transitions: &T,
) -> bool {
    match frontier {
        FullWalkWitnessFrontier::One(lexer, parser) => full_walk_live_component_witness(
            lexer, parser, alphabet, proofs, parser_cache, constraint, tokenizer, transitions),
        FullWalkWitnessFrontier::Two(branches) => branches.into_iter().any(|(lexer, parser)|
            full_walk_live_component_witness(lexer, parser, alphabet, proofs,
                parser_cache, constraint, tokenizer, transitions)),
        FullWalkWitnessFrontier::Many(frontier) => full_walk_many_component_witness(
            frontier, alphabet, proofs, parser_cache, constraint, tokenizer, transitions),
    }
}

/// A cursor into a bounded, mask-local deterministic transition memo. The
/// direct variant preserves the reference executor when storage is exhausted.
/// Cached cursors copy only their ID when a vocabulary edge saves/restores its
/// parent; they do not repeatedly clone the whole ambiguous lexer frontier.
enum FullWalkManyCursor {
    Direct(FullWalkManyState),
    Cached(u16),
}

impl Clone for FullWalkManyCursor {
    fn clone(&self) -> Self {
        match self {
            Self::Direct(state) => Self::Direct(state.clone()),
            Self::Cached(id) => Self::Cached(*id),
        }
    }

    #[inline]
    fn clone_from(&mut self, source: &Self) {
        // Most vocabulary stack restores copy only a memo ID. Do not route
        // that through a temporary large inline frontier and its drop path.
        match (&mut *self, source) {
            (Self::Cached(target), Self::Cached(source)) => *target = *source,
            _ => *self = source.clone(),
        }
    }
}

impl FullWalkManyCursor {
    #[inline]
    fn set_cached(&mut self, source: u16) {
        match self {
            Self::Cached(target) => *target = source,
            _ => *self = Self::Cached(source),
        }
    }

    fn id(&self) -> Option<u16> {
        match self { Self::Cached(id) => Some(*id), Self::Direct(_) => None }
    }
}

/// Store ambiguous frontiers only at depths that actually need them. A scalar
/// walk does not allocate or reserve hundreds of large frontier payloads in its
/// stack frame. The parallel lexer stack remains the authority on which depths
/// contain a Many cursor; stale entries at other depths are never read.
#[derive(Default)]
struct FullWalkManyStack {
    slots: Vec<Option<FullWalkManyCursor>>,
}

impl FullWalkManyStack {
    #[inline]
    fn save(&mut self, depth: usize, cursor: &FullWalkManyCursor) {
        debug_assert!(depth < 256, "full-walk trie depth was checked at entry");
        if self.slots.len() <= depth {
            self.slots.resize_with(depth + 1, || None);
        }
        match &mut self.slots[depth] {
            Some(existing) => existing.clone_from(cursor),
            slot @ None => *slot = Some(cursor.clone()),
        }
    }

    #[inline]
    fn restore(&self, depth: usize) -> &FullWalkManyCursor {
        self.slots[depth]
            .as_ref()
            .expect("a Many lexer-stack entry must have a saved correlated frontier")
    }
}

/// Exact, bounded memoization of complete correlated frontiers in one mask.
/// Parser/lexer IDs are append-only. Full equality, not hashes, establishes
/// identity; exhausted storage retains the authoritative direct executor.
struct FullWalkManyTransitionMemo {
    limit: usize,
    states: Vec<FullWalkManyState>,
    // No references into rows escape a lookup. A contiguous growable table
    // avoids a separate allocation for every memoized frontier; IDs, not
    // pointers, remain the only row handles. Allocation is still lazy/bounded.
    rows: Vec<[u16; 256]>,
    identity_bytes: Vec<[u64; 4]>,
    buckets: FxHashMap<u64, SmallVec<[u16; 2]>>,
    lookups: usize,
    hits: usize,
}

impl FullWalkManyTransitionMemo {
    const UNKNOWN: u16 = u16::MAX;
    const MAX_STATES: usize = 1024;

    fn new(limit: usize) -> Self {
        Self {
            limit: limit.min(Self::MAX_STATES), states: Vec::new(),
            rows: Vec::new(), identity_bytes: Vec::new(),
            buckets: FxHashMap::default(), lookups: 0, hits: 0,
        }
    }

    fn view<'a>(&'a self, cursor: &'a FullWalkManyCursor) -> &'a FullWalkManyState {
        match cursor {
            FullWalkManyCursor::Direct(state) => state,
            FullWalkManyCursor::Cached(id) => &self.states[*id as usize],
        }
    }

    fn hold(&mut self, state: FullWalkManyState) -> FullWalkManyCursor {
        let cacheable = match &state {
            FullWalkManyState::Branches(branches) => branches.len() <= 8
                && branches.iter().all(|branch| match &branch.prune_guard {
                    FullWalkPruneGuard::Passed => true,
                    FullWalkPruneGuard::Pending(memories) => memories.len() <= 4,
                }),
            FullWalkManyState::ThreeSameParser { .. } => true,
        };
        if self.limit == 0 || !cacheable { return FullWalkManyCursor::Direct(state); }
        use std::hash::{Hash, Hasher};
        let mut hasher = rustc_hash::FxHasher::default();
        state.hash(&mut hasher);
        let hash = hasher.finish();
        if let Some(bucket) = self.buckets.get(&hash) {
            for &id in bucket {
                if self.states[id as usize] == state {
                    return FullWalkManyCursor::Cached(id);
                }
            }
        }
        if self.states.len() >= self.limit { return FullWalkManyCursor::Direct(state); }
        let id = self.states.len() as u16;
        self.states.push(state);
        self.rows.push([Self::UNKNOWN; 256]);
        self.identity_bytes.push([0; 4]);
        self.buckets.entry(hash).or_default().push(id);
        FullWalkManyCursor::Cached(id)
    }

    #[inline]
    fn cached_transition(&mut self, cursor: &FullWalkManyCursor, byte: u8)
        -> Option<u16>
    {
        let source = cursor.id()?;
        self.lookups += 1;
        let target = self.rows[source as usize][byte as usize];
        if target == Self::UNKNOWN { return None; }
        self.hits += 1;
        Some(target)
    }

    fn remember_transition(&mut self, source: Option<u16>, byte: u8, target: &FullWalkManyCursor) {
        if let (Some(source), Some(target)) = (source, target.id()) {
            self.rows[source as usize][byte as usize] = target;
            if source == target {
                self.identity_bytes[source as usize][byte as usize >> 6] |= 1u64 << (byte & 63);
            }
        }
    }

    #[inline]
    fn identity_alphabet(&self, cursor: &FullWalkManyCursor) -> Option<&[u64; 4]> {
        Some(&self.identity_bytes[cursor.id()? as usize])
    }
}

/// Cost policy only: shallow single-root frontiers can retain the original
/// byte loop, while genuinely branching parser work uses identity certificates.
/// Admission is monotone because parser nodes are append-only within a walk.
/// Declining never removes a branch or changes a mask bit.
#[inline(always)]
fn full_walk_identity_context_profitable(
    roots: usize, parser_nodes: usize,
) -> bool {
    roots >= 2 || parser_nodes >= 32
}

/// An alphabet of exact self-loops is closed under concatenation. When every
/// byte in a vocabulary subtree belongs to it, all descendant token endpoints
/// see exactly the current full parser/lexer/guard state. Unseen bytes never
/// qualify; this is not a character-class or language approximation.
#[inline]
fn identity_alphabet_covers_subtree(identity: &[u64; 4], subtree: [u64; 4]) -> bool {
    subtree.iter().zip(identity).all(|(&needed, &known)| needed & !known == 0)
}

/// Deferred accounting must still visit every marker. Once polarity is
/// fixed, a constant subtree needs no writes when its verdict already agrees
/// with the initialized output bits. Token aliases then require no work either.
#[inline]
fn identity_subtree_requires_output(deferred: bool, positive: bool, allowed: bool) -> bool {
    deferred || positive == allowed
}

/// A lower-bound hit proves true; an upper-bound miss proves false.
/// The unresolved middle must use the existing exact stack simulation.
#[inline(always)]
fn full_walk_row_liveness_bound(
    guaranteed_live: bool,
    necessary_live: impl FnOnce() -> bool,
) -> Option<bool> {
    if guaranteed_live {
        Some(true)
    } else if !necessary_live() {
        Some(false)
    } else {
        None
    }
}

struct FullWalkParserNode {
    gss: ParserStacks,
    admitted: Option<BitSet>,
    shared_root_admission: SharedRootAdmission,
    admitted_singleton: Option<TerminalID>,
    token_boundary_allowed: Vec<u8>,
    children: SmallVec<[(TerminalID, u32); 16]>,
    last_child_terminal: TerminalID,
    last_child_target: u32,
}

struct FullWalkParserCache {
    row_liveness_enabled: bool,
    nodes: Vec<FullWalkParserNode>,
    lexer_state_count: usize,
    canonicalize: bool,
    identity_proofs_enabled: bool,
    canonical: Option<Box<FullWalkDenseParserCanonical>>,
    profile: bool,
    profile_boundary_calls: usize,
    profile_boundary_misses: usize,
    profile_advance_calls: usize,
    profile_advance_misses: usize,
    profile_inadmissible_finalizers: usize,
}

/// Bounded exact sharing of complete parser-stack languages within one mask.
/// Maximal-munch exclusions remain in correlated lexer branches. Unit GSS
/// accumulators make the semantic key a complete parser-language identity.
struct FullWalkDenseParserCanonical {
    keys: crate::ds::leveled_gss::GssSemanticKeyInterner<u32, ()>,
    nodes: FxHashMap<u32, u32>,
    hits: usize,
}

impl FullWalkParserCache {
    const DEAD: u32 = u32::MAX;

    fn from_roots(
        root_branches: &DynamicBranches,
        lexer_state_count: usize,
        profile: bool,
    ) -> (Self, SmallVec<[u32; 4]>) {
        let mut nodes = Vec::<FullWalkParserNode>::new();
        let mut root_nodes = SmallVec::<[u32; 4]>::new();
        for branch in root_branches {
            if let Some((index, _)) = nodes
                .iter()
                .enumerate()
                .find(|(_, node)| node.gss.ptr_eq(&branch.gss))
            {
                root_nodes.push(index as u32);
                continue;
            }
            let id = nodes.len() as u32;
            nodes.push(FullWalkParserNode {
                gss: branch.gss.clone(),
                admitted: None,
                shared_root_admission: branch.shared_root_admission.clone(),
                admitted_singleton: None,
                token_boundary_allowed: vec![0; lexer_state_count],
                children: SmallVec::new(),
                last_child_terminal: TerminalID::MAX,
                last_child_target: Self::DEAD,
            });
            root_nodes.push(id);
        }
        let canonicalize = full_walk_acceleration_enabled();
        let identity_proofs_enabled = canonicalize
            && full_walk_identity_context_profitable(root_branches.len(), nodes.len());
        (
            Self {
                row_liveness_enabled: {
                    static ENABLED: OnceLock<bool> = OnceLock::new();
                    *ENABLED.get_or_init(|| !env_flag("GLRMASK_DISABLE_ROW_LIVENESS", false))
                },
                nodes,
                lexer_state_count,
                canonicalize,
                identity_proofs_enabled,
                canonical: None,
                profile,
                profile_boundary_calls: 0,
                profile_boundary_misses: 0,
                profile_advance_calls: 0,
                profile_advance_misses: 0,
                profile_inadmissible_finalizers: 0,
            },
            root_nodes,
        )
    }

    fn push_parser_stacks(&mut self, gss: ParserStacks) -> u32 {
        let id = self.nodes.len() as u32;
        self.nodes.push(FullWalkParserNode {
            gss,
            admitted: None,
            shared_root_admission: None,
            admitted_singleton: None,
            token_boundary_allowed: vec![0; self.lexer_state_count],
            children: SmallVec::new(),
            last_child_terminal: TerminalID::MAX,
            last_child_target: Self::DEAD,
        });
        // Nodes are append-only. Update this monotone policy at the mutation,
        // not by re-reading two lengths for every byte in the vocabulary walk.
        self.identity_proofs_enabled |= self.canonicalize && self.nodes.len() >= 32;
        id
    }

    fn intern_parser_stacks(&mut self, gss: ParserStacks) -> u32 {
        if !self.canonicalize || self.nodes.len() < 32 {
            return self.push_parser_stacks(gss);
        }
        if self.canonical.is_none() {
            let mut canonical = FullWalkDenseParserCanonical {
                keys: crate::ds::leveled_gss::GssSemanticKeyInterner::with_budget(
                    16_384, 8_192, 16_384,
                ),
                nodes: FxHashMap::default(),
                hits: 0,
            };
            for (index, node) in self.nodes.iter().enumerate() {
                let key = canonical.keys.key(&node.gss);
                if canonical.keys.is_exhausted() { break; }
                canonical.nodes.entry(key).or_insert(index as u32);
            }
            self.canonical = Some(Box::new(canonical));
        }
        let canonical = self.canonical.as_mut().expect("initialized above");
        if canonical.keys.is_exhausted() {
            return self.push_parser_stacks(gss);
        }
        let key = canonical.keys.key(&gss);
        // Exhaustion returns an internal sentinel, not a language certificate.
        // Never alias on that value; leave the original exact path available.
        if canonical.keys.is_exhausted() {
            return self.push_parser_stacks(gss);
        }
        if let Some(&node) = canonical.nodes.get(&key) {
            canonical.hits += 1;
            return node;
        }
        let node = self.push_parser_stacks(gss);
        self.canonical.as_mut().unwrap().nodes.insert(key, node);
        node
    }

    #[inline(always)]
    fn terminal_not_known_inadmissible(
        &mut self,
        constraint: &Constraint,
        node: u32,
        terminal: TerminalID,
    ) -> bool {
        if Some(terminal) == constraint.ignore_terminal {
            return true;
        }
        // This is deliberately only a free precheck. Do not compute parser
        // admission merely to avoid an `advance`: on many states the existing
        // LR/action fast rejection is cheaper than materializing the complete
        // exact admission set. When another boundary/proof query has already
        // populated that set, however, membership is an exact inexpensive way
        // to skip impossible reset branches.
        let admitted = self.nodes[node as usize]
            .admitted
            .as_ref()
            .is_none_or(|admitted| admitted.contains(terminal as usize));
        if self.profile && !admitted {
            self.profile_inadmissible_finalizers += 1;
        }
        admitted
    }

    #[inline(always)]
    fn advance(
        &mut self,
        constraint: &Constraint,
        node: u32,
        terminal: TerminalID,
    ) -> Option<u32> {
        if self.profile {
            self.profile_advance_calls += 1;
        }
        let node_index = node as usize;
        let (last_child_terminal, last_child_target) = unsafe {
            let cached_node = self.nodes.get_unchecked(node_index);
            (cached_node.last_child_terminal, cached_node.last_child_target)
        };
        if last_child_terminal == terminal {
            let cached = last_child_target;
            return (cached != Self::DEAD).then_some(cached);
        }
        if Some(terminal) == constraint.ignore_terminal {
            return Some(node);
        }
        if let Some(&(_, cached)) = self.nodes[node_index]
            .children
            .iter()
            .find(|&&(candidate, _)| candidate == terminal)
        {
            self.nodes[node_index].last_child_terminal = terminal;
            self.nodes[node_index].last_child_target = cached;
            return (cached != Self::DEAD).then_some(cached);
        }
        // With no zero-width control terminals, a single-top parser frontier
        // whose LR row has no action for this terminal cannot advance. This is
        // exactly the first branch that the generic GLR advance would reject;
        // avoid constructing an empty-accumulator GSS and entering the GLR
        // engine for that overwhelmingly common negative lookup.
        if Some(terminal) != constraint.ignore_terminal
            && !constraint.uses_sparse_direct_regular_runtime()
            && !constraint.uses_compact_segmented_parser_runtime()
            && constraint.table.control_terminals.is_empty()
            && self.nodes[node_index]
                .gss
                .single_top_value()
                .is_some_and(|top| constraint.table.action(top, terminal).is_none())
        {
            self.nodes[node_index].children.push((terminal, Self::DEAD));
            self.nodes[node_index].last_child_terminal = terminal;
            self.nodes[node_index].last_child_target = Self::DEAD;
            return None;
        }
        let next = parser_child(constraint, &self.nodes[node_index].gss, terminal);
        if self.profile {
            self.profile_advance_misses += 1;
        }
        let target = if let Some(gss) = next {
            self.intern_parser_stacks(gss)
        } else {
            Self::DEAD
        };
        self.nodes[node_index].children.push((terminal, target));
        self.nodes[node_index].last_child_terminal = terminal;
        self.nodes[node_index].last_child_target = target;
        (target != Self::DEAD).then_some(target)
    }

    fn admitted(&mut self, constraint: &Constraint, node: u32) -> &BitSet {
        let index = node as usize;
        if self.nodes[index].admitted.is_none() {
            let started = self.profile.then(std::time::Instant::now);
            let admitted = parser_admission_with_root_memo(
                constraint, &self.nodes[index].gss, &self.nodes[index].shared_root_admission,
            );
            self.nodes[index].admitted_singleton = {
                let mut ones = admitted.iter_ones();
                let first = ones.next().map(|terminal| terminal as TerminalID);
                first.filter(|_| ones.next().is_none())
            };
            self.nodes[index].admitted = Some(admitted);
            if let Some(started) = started {
                eprintln!(
                    "[glrmask/profile][parser_admitted] node={} count={} elapsed_ms={:.3}",
                    node,
                    self.nodes[index].admitted.as_ref().map_or(0, BitSet::count_ones),
                    started.elapsed().as_secs_f64() * 1e3,
                );
            }
        }
        self.nodes[index].admitted.as_ref().unwrap()
    }


    /// Exact lower/upper bounds for one plain parser top, with no scoped or
    /// zero-width transitions. The existing table contract is:
    /// unconditional advance <= stack-admissible terminals <= advance row.
    /// Only a definite answer bypasses full stack-dependent admission.
    #[inline]
    fn row_future_allowed<T: FullWalkTransitionTable>(
        &self,
        constraint: &Constraint,
        tokenizer: &Tokenizer,
        transitions: &T,
        parser_node: u32,
        lexer_state: u32,
    ) -> Option<bool> {
        if !self.row_liveness_enabled
            || self.nodes[parser_node as usize].admitted.is_some()
            || constraint.uses_sparse_direct_regular_runtime()
            || constraint.uses_compact_segmented_parser_runtime()
            || !constraint.table.control_terminals.is_empty()
        {
            return None;
        }
        let top = self.nodes[parser_node as usize].gss.single_top_value()?;
        let necessary = constraint.table.advance_row(top)?;
        let guaranteed = constraint.table.unconditional_advance_row(top)?;
        full_walk_row_liveness_bound(
            transitions.future_intersects(tokenizer, lexer_state, guaranteed),
            || transitions.future_intersects(tokenizer, lexer_state, necessary),
        )
    }

    #[inline(always)]
    fn physical_token_boundary_allowed<T: FullWalkTransitionTable>(
        &mut self,
        constraint: &Constraint,
        tokenizer: &Tokenizer,
        transitions: &T,
        parser_node: u32,
        lexer_state: u32,
    ) -> bool {
        if self.profile {
            self.profile_boundary_calls += 1;
        }
        let node = parser_node as usize;
        let lexer = lexer_state as usize;
        let cached = unsafe {
            *self.nodes
                .get_unchecked(node)
                .token_boundary_allowed
                .get_unchecked(lexer)
        };
        if cached != 0 {
            return cached == 2;
        }
        if self.profile {
            self.profile_boundary_misses += 1;
        }
        // Prove liveness directly from the current parser row where possible.
        // Ambiguous reductions and guarded actions still use full exact
        // admission. Its singleton case retains the existing one-bit lookup.
        let parser_future_allowed = if let Some(allowed) = self.row_future_allowed(
            constraint, tokenizer, transitions, parser_node, lexer_state,
        ) {
            allowed
        } else {
            let _ = self.admitted(constraint, parser_node);
            if let Some(terminal) = self.nodes[node].admitted_singleton {
                transitions.future_contains(tokenizer, lexer_state, terminal)
            } else {
                transitions.future_intersects(
                    tokenizer, lexer_state,
                    self.nodes[node].admitted.as_ref().expect("admitted set populated above"),
                )
            }
        };
        let allowed = constraint
            .ignore_terminal
            .is_some_and(|terminal| transitions.future_contains(tokenizer, lexer_state, terminal))
            || parser_future_allowed;
        unsafe {
            *self.nodes
                .get_unchecked_mut(node)
                .token_boundary_allowed
                .get_unchecked_mut(lexer) = if allowed { 2 } else { 1 };
        }
        allowed
    }

    /// Stable pointer to the fixed-size parser/lexer liveness row for one
    /// parser-cache node.  The node vector may move when parser children are
    /// appended, but each row owns a separate allocation whose length never
    /// changes, so the row buffer itself remains stable for the lifetime of
    /// this cache.  The dense hot walker keeps this pointer alongside its
    /// scalar state so a cached liveness hit is one byte load.
    #[inline(always)]
    fn physical_boundary_row_ptr(&self, parser_node: u32) -> *const u8 {
        unsafe {
            self.nodes
                .get_unchecked(parser_node as usize)
                .token_boundary_allowed
                .as_ptr()
        }
    }

    #[inline(always)]
    fn token_boundary_allowed_raw<T: FullWalkTransitionTable>(
        &mut self,
        constraint: &Constraint,
        tokenizer: &Tokenizer,
        transitions: &T,
        initial_lexer_state: u32,
        lexer_state: u32,
        parser_node: u32,
    ) -> bool {
        lexer_state == initial_lexer_state
            || self.physical_token_boundary_allowed(
                constraint,
                tokenizer,
                transitions,
                parser_node,
                lexer_state,
            )
    }

    #[inline(always)]
    fn token_boundary_allowed<T: FullWalkTransitionTable>(
        &mut self,
        constraint: &Constraint,
        tokenizer: &Tokenizer,
        transitions: &T,
        initial_lexer_state: u32,
        branch: &FullWalkBranch,
    ) -> bool {
        self.token_boundary_allowed_raw(
            constraint,
            tokenizer,
            transitions,
            initial_lexer_state,
            branch.lexer_state,
            branch.parser_node,
        )
    }

}

#[inline]
fn full_walk_push_unique(
    branches: &mut FullWalkBranches,
    branch: FullWalkBranch,
) {
    if !branches.contains(&branch) {
        branches.push(branch);
    }
}

enum FullWalkScalarFinalizerOutcome {
    Scalar(FullWalkBranch),
    Two(FullWalkBranch, FullWalkBranch),
    Many(FullWalkBranches),
}

enum FullWalkTwoStepOutcome {
    Dead,
    One((u32, u32)),
    Two((u32, u32), (u32, u32)),
    Many(FullWalkBranches),
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn full_walk_step_two<T: FullWalkTransitionTable>(
    branches: ((u32, u32), (u32, u32)),
    byte: u8,
    initial_lexer_state: u32,
    finalizer_code: &[u32],
    single_finalizer_continues: &[u8],
    tokenizer: &Tokenizer,
    transitions: &T,
    parser_cache: &mut FullWalkParserCache,
    constraint: &Constraint,
) -> FullWalkTwoStepOutcome {
    let first_cell = transitions.cell(branches.0.0, byte);
    let second_cell = transitions.cell(branches.1.0, byte);

    // Dominant correlated-parser case: both deterministic lexer branches stay
    // alive without finalizing, while their exact parser identities differ.
    // Keep the correlation tuple intact and skip the general option/collapse
    // classification below.
    // Dominant two-branch case: neither branch finalizes. A globally live
    // lexer target may nevertheless be dead for this correlated parser branch
    // once all of its remaining terminal futures are parser-inadmissible. The
    // parser-node/lexer-state cache makes this exact test a byte lookup after
    // the first visit and prevents unrelated lexer residuals from keeping the
    // branch alive through the rest of a vocabulary token.
    if !T::cell_has_finalizer(first_cell) && !T::cell_has_finalizer(second_cell) {
        let first = if T::cell_is_dead(first_cell) {
            None
        } else {
            let target = T::cell_target(first_cell);
            parser_cache
                .physical_token_boundary_allowed(constraint, tokenizer, transitions, branches.0.1, target)
                .then_some((target, branches.0.1))
        };
        let second = if T::cell_is_dead(second_cell) {
            None
        } else {
            let target = T::cell_target(second_cell);
            parser_cache
                .physical_token_boundary_allowed(constraint, tokenizer, transitions, branches.1.1, target)
                .then_some((target, branches.1.1))
        };
        return match (first, second) {
            (None, None) => FullWalkTwoStepOutcome::Dead,
            (Some(branch), None) | (None, Some(branch)) => FullWalkTwoStepOutcome::One(branch),
            (Some(first), Some(second)) if first == second => FullWalkTwoStepOutcome::One(first),
            (Some(first), Some(second)) => FullWalkTwoStepOutcome::Two(first, second),
        };
    }

    full_walk_step_two_finalizing::<T>(
        branches,
        first_cell,
        second_cell,
        initial_lexer_state,
        finalizer_code,
        single_finalizer_continues,
        tokenizer,
        transitions,
        parser_cache,
        constraint,
    )
}

#[allow(clippy::too_many_arguments)]
#[cold]
#[inline(never)]
fn full_walk_step_two_finalizing<T: FullWalkTransitionTable>(
    branches: ((u32, u32), (u32, u32)),
    first_cell: T::Cell,
    second_cell: T::Cell,
    initial_lexer_state: u32,
    finalizer_code: &[u32],
    single_finalizer_continues: &[u8],
    tokenizer: &Tokenizer,
    transitions: &T,
    parser_cache: &mut FullWalkParserCache,
    constraint: &Constraint,
) -> FullWalkTwoStepOutcome {
    let mut next = FullWalkBranches::new();
    for (cell, (source_lexer, parser_node)) in
        [(first_cell, branches.0), (second_cell, branches.1)]
    {
        if T::cell_is_dead(cell) {
            continue;
        }
        let target = T::cell_target(cell);
        if !T::cell_has_finalizer(cell) {
            if parser_cache.physical_token_boundary_allowed(
                constraint,
                tokenizer,
                transitions,
                parser_node,
                target,
            ) {
                full_walk_push_unique(
                    &mut next,
                    FullWalkBranch {
                        lexer_state: target,
                        parser_node,
                        prune_guard: FullWalkPruneGuard::Passed,
                    },
                );
            }
            continue;
        }
        let _ = source_lexer;
        match full_walk_scalar_finalizer(
            target,
            parser_node,
            initial_lexer_state,
            finalizer_code,
            single_finalizer_continues,
            tokenizer,
            transitions,
            parser_cache,
            constraint,
        ) {
            FullWalkScalarFinalizerOutcome::Scalar(branch) => {
                full_walk_push_unique(&mut next, branch);
            }
            FullWalkScalarFinalizerOutcome::Two(first, second) => {
                full_walk_push_unique(&mut next, first);
                full_walk_push_unique(&mut next, second);
            }
            FullWalkScalarFinalizerOutcome::Many(branches) => {
                for branch in branches {
                    full_walk_push_unique(&mut next, branch);
                }
            }
        }
    }
    match next.len() {
        0 => FullWalkTwoStepOutcome::Dead,
        1 if next[0].prune_guard.is_passed() => {
            let branch = next.pop().expect("one full-walk branch disappeared");
            FullWalkTwoStepOutcome::One((branch.lexer_state, branch.parser_node))
        }
        2 if next.iter().all(|branch| branch.prune_guard.is_passed()) => {
            let second = next.pop().expect("second full-walk branch disappeared");
            let first = next.pop().expect("first full-walk branch disappeared");
            FullWalkTwoStepOutcome::Two(
                (first.lexer_state, first.parser_node),
                (second.lexer_state, second.parser_node),
            )
        }
        _ => FullWalkTwoStepOutcome::Many(next),
    }
}

#[allow(clippy::too_many_arguments)]
#[cold]
#[inline(never)]
fn full_walk_scalar_finalizer<T: FullWalkTransitionTable>(
    target: u32,
    parser_node: u32,
    initial_lexer_state: u32,
    finalizer_code: &[u32],
    single_finalizer_continues: &[u8],
    tokenizer: &Tokenizer,
    transitions: &T,
    parser_cache: &mut FullWalkParserCache,
    constraint: &Constraint,
) -> FullWalkScalarFinalizerOutcome {
    const MULTI: u32 = u32::MAX - 1;
    let code = transitions.finalizer_code(target, finalizer_code);
    if code != MULTI {
        if parser_cache.terminal_not_known_inadmissible(constraint, parser_node, code)
            && let Some(next_parser) = parser_cache.advance(constraint, parser_node, code)
        {
            let reset = FullWalkBranch {
                lexer_state: initial_lexer_state,
                parser_node: next_parser,
                prune_guard: if Some(code) == constraint.ignore_terminal {
                    FullWalkPruneGuard::Passed
                } else if transitions.single_finalizer_continues(target, single_finalizer_continues) {
                    FullWalkPruneGuard::Pending(smallvec::smallvec![(target, code)])
                } else {
                    FullWalkPruneGuard::Passed
                },
            };
            let continuing = FullWalkBranch {
                lexer_state: target,
                parser_node,
                prune_guard: FullWalkPruneGuard::Passed,
            };
            if reset == continuing {
                return FullWalkScalarFinalizerOutcome::Scalar(continuing);
            }
            return FullWalkScalarFinalizerOutcome::Two(reset, continuing);
        }
        return FullWalkScalarFinalizerOutcome::Scalar(FullWalkBranch {
            lexer_state: target,
            parser_node,
            prune_guard: FullWalkPruneGuard::Passed,
        });
    }

    let mut next = FullWalkBranches::new();
    for terminal in transitions.matched_terminals(tokenizer, target) {
        if !parser_cache.terminal_not_known_inadmissible(constraint, parser_node, terminal) {
            continue;
        }
        if let Some(next_parser) = parser_cache.advance(constraint, parser_node, terminal) {
            full_walk_push_unique(
                &mut next,
                FullWalkBranch {
                    lexer_state: initial_lexer_state,
                    parser_node: next_parser,
                    prune_guard: if Some(terminal) == constraint.ignore_terminal {
                        FullWalkPruneGuard::Passed
                    } else {
                        FullWalkPruneGuard::Passed.remember_terminal_match(
                            tokenizer, transitions, target, terminal,
                        )
                    },
                },
            );
        }
    }

    if next.is_empty() {
        return FullWalkScalarFinalizerOutcome::Scalar(FullWalkBranch {
            lexer_state: target,
            parser_node,
            prune_guard: FullWalkPruneGuard::Passed,
        });
    }
    full_walk_push_unique(
        &mut next,
        FullWalkBranch {
            lexer_state: target,
            parser_node,
            prune_guard: FullWalkPruneGuard::Passed,
        },
    );
    if next.len() == 1 && next[0].prune_guard.is_passed() {
        FullWalkScalarFinalizerOutcome::Scalar(
            next.pop().expect("one full-walk branch disappeared"),
        )
    } else {
        FullWalkScalarFinalizerOutcome::Many(next)
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn full_walk_scalar_finalizer_hot_single<T: FullWalkTransitionTable>(
    target: u32,
    parser_node: u32,
    initial_lexer_state: u32,
    finalizer_code: &[u32],
    single_finalizer_continues: &[u8],
    tokenizer: &Tokenizer,
    transitions: &T,
    parser_cache: &mut FullWalkParserCache,
    constraint: &Constraint,
) -> FullWalkScalarFinalizerOutcome {
    const MULTI: u32 = u32::MAX - 1;
    let code = transitions.finalizer_code(target, finalizer_code);
    if code == MULTI {
        return full_walk_scalar_finalizer(
            target,
            parser_node,
            initial_lexer_state,
            finalizer_code,
            single_finalizer_continues,
            tokenizer,
            transitions,
            parser_cache,
            constraint,
        );
    }
    if parser_cache.terminal_not_known_inadmissible(constraint, parser_node, code)
        && let Some(next_parser) = parser_cache.advance(constraint, parser_node, code)
    {
        let reset = FullWalkBranch {
            lexer_state: initial_lexer_state,
            parser_node: next_parser,
            prune_guard: if Some(code) == constraint.ignore_terminal {
                FullWalkPruneGuard::Passed
            } else if transitions.single_finalizer_continues(target, single_finalizer_continues) {
                FullWalkPruneGuard::Pending(smallvec::smallvec![(target, code)])
            } else {
                FullWalkPruneGuard::Passed
            },
        };
        let continuing = FullWalkBranch {
            lexer_state: target,
            parser_node,
            prune_guard: FullWalkPruneGuard::Passed,
        };
        if reset == continuing {
            FullWalkScalarFinalizerOutcome::Scalar(continuing)
        } else {
            FullWalkScalarFinalizerOutcome::Two(reset, continuing)
        }
    } else {
        FullWalkScalarFinalizerOutcome::Scalar(FullWalkBranch {
            lexer_state: target,
            parser_node,
            prune_guard: FullWalkPruneGuard::Passed,
        })
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn full_walk_try_apply_plain_single_finalizer<T: FullWalkTransitionTable>(
    target: u32,
    parser_node: u32,
    initial_lexer_state: u32,
    finalizer_code: &[u32],
    single_finalizer_continues: &[u8],
    transitions: &T,
    parser_cache: &mut FullWalkParserCache,
    constraint: &Constraint,
    two_distinct_marker: u32,
    scalar_lexer: &mut u32,
    scalar_parser: &mut u32,
    current_two: &mut ((u32, u32), (u32, u32)),
) -> bool {
    const MULTI: u32 = u32::MAX - 1;
    let code = transitions.finalizer_code(target, finalizer_code);
    if code == MULTI
        || (Some(code) != constraint.ignore_terminal
            && transitions.single_finalizer_continues(target, single_finalizer_continues))
    {
        return false;
    }

    if !parser_cache.terminal_not_known_inadmissible(constraint, parser_node, code) {
        *scalar_lexer = target;
        *scalar_parser = parser_node;
        return true;
    }

    let Some(next_parser) = parser_cache.advance(constraint, parser_node, code) else {
        *scalar_lexer = target;
        *scalar_parser = parser_node;
        return true;
    };

    if next_parser != parser_node {
        *scalar_lexer = two_distinct_marker;
        *current_two = (
            (initial_lexer_state, next_parser),
            (target, parser_node),
        );
        return true;
    }

    // If both exact coordinates are identical there is only one branch. When
    // the parser is the same but lexer coordinates differ, leave the rare case
    // to the existing exact union logic below.
    if initial_lexer_state == target {
        *scalar_lexer = target;
        *scalar_parser = parser_node;
        return true;
    }
    false
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn full_walk_step_many<T: FullWalkTransitionTable>(
    branches: &FullWalkBranches,
    byte: u8,
    initial_lexer_state: u32,
    finalizer_code: &[u32],
    tokenizer: &Tokenizer,
    transitions: &T,
    parser_cache: &mut FullWalkParserCache,
    constraint: &Constraint,
) -> FullWalkBranches {
    const NONE: u32 = u32::MAX;
    const MULTI: u32 = u32::MAX - 1;
    let mut next = FullWalkBranches::new();
    for branch in branches {
        let Some(advanced_guard) = branch
            .prune_guard
            .advance(tokenizer, transitions, byte)
        else {
            continue;
        };
        let target = transitions.transition(branch.lexer_state, byte);
        if target == u32::MAX {
            continue;
        }
        let code = transitions.finalizer_code(target, finalizer_code);
        if code == MULTI {
            for terminal in transitions.matched_terminals(tokenizer, target) {
                if !parser_cache.terminal_not_known_inadmissible(
                    constraint,
                    branch.parser_node,
                    terminal,
                ) {
                    continue;
                }
                if let Some(parser_node) = parser_cache.advance(
                    constraint, branch.parser_node, terminal,
                ) {
                    let matched_guard = if Some(terminal) == constraint.ignore_terminal {
                        advanced_guard.clone()
                    } else {
                        advanced_guard.remember_terminal_match(tokenizer, transitions, target, terminal)
                    };
                    full_walk_push_unique(
                        &mut next,
                        FullWalkBranch {
                            lexer_state: initial_lexer_state,
                            parser_node,
                            prune_guard: matched_guard,
                        },
                    );
                }
            }
        } else if code != NONE
            && parser_cache.terminal_not_known_inadmissible(constraint, branch.parser_node, code)
            && let Some(parser_node) = parser_cache.advance(
                constraint, branch.parser_node, code,
            )
        {
            let matched_guard = if Some(code) == constraint.ignore_terminal {
                advanced_guard.clone()
            } else {
                advanced_guard.remember_terminal_match(tokenizer, transitions, target, code)
            };
            full_walk_push_unique(
                &mut next,
                FullWalkBranch {
                    lexer_state: initial_lexer_state,
                    parser_node,
                    prune_guard: matched_guard,
                },
            );
        }
        if parser_cache.physical_token_boundary_allowed(
            constraint,
            tokenizer,
            transitions,
            branch.parser_node,
            target,
        ) {
            full_walk_push_unique(
                &mut next,
                FullWalkBranch {
                    lexer_state: target,
                    parser_node: branch.parser_node,
                    prune_guard: advanced_guard,
                },
            );
        }
    }
    next
}

#[inline]
fn full_walk_projection_union_two(
    vocab: &DynamicMaskVocab,
    cache: &mut FxHashMap<(u32, u32), Option<u32>>,
    first: u32,
    second: u32,
) -> Option<u32> {
    let key = if first <= second {
        (first, second)
    } else {
        (second, first)
    };
    if let Some(&cached) = cache.get(&key) {
        return cached;
    }
    let result = vocab.mask_projection_state_for_projection_states(&[key.0, key.1]);
    cache.insert(key, result);
    result
}

#[inline]
fn full_walk_projection_union_three(
    vocab: &DynamicMaskVocab,
    cache: &mut FxHashMap<(u32, u32, u32), Option<u32>>,
    first: u32,
    second: u32,
    third: u32,
) -> Option<u32> {
    let mut states = [first, second, third];
    states.sort_unstable();
    let key = (states[0], states[1], states[2]);
    if let Some(&cached) = cache.get(&key) {
        return cached;
    }
    let result = vocab.mask_projection_state_for_projection_states(&states);
    cache.insert(key, result);
    result
}

#[inline]
fn full_walk_merge_two_same_parser<T: FullWalkTransitionTable>(
    transitions: &T,
    vocab: &DynamicMaskVocab,
    pair_union_cache: &mut FxHashMap<(u32, u32), Option<u32>>,
    first: (u32, u32),
    second: (u32, u32),
) -> Option<(u32, u32)> {
    if first.1 != second.1 {
        return None;
    }
    transitions
        .union_states(&[first.0, second.0])
        .or_else(|| full_walk_projection_union_two(vocab, pair_union_cache, first.0, second.0))
        .map(|lexer_state| (lexer_state, first.1))
}

#[inline]
fn full_walk_merge_three_same_parser<T: FullWalkTransitionTable>(
    transitions: &T,
    vocab: &DynamicMaskVocab,
    triple_union_cache: &mut FxHashMap<(u32, u32, u32), Option<u32>>,
    lexers: (u32, u32, u32),
    parser_node: u32,
) -> Option<(u32, u32)> {
    transitions
        .union_states(&[lexers.0, lexers.1, lexers.2])
        .or_else(|| full_walk_projection_union_three(
            vocab,
            triple_union_cache,
            lexers.0,
            lexers.1,
            lexers.2,
        ))
        .map(|lexer_state| (lexer_state, parser_node))
}

#[inline]
fn full_walk_merge_branches_same_parser<T: FullWalkTransitionTable>(
    transitions: &T,
    vocab: &DynamicMaskVocab,
    pair_union_cache: &mut FxHashMap<(u32, u32), Option<u32>>,
    triple_union_cache: &mut FxHashMap<(u32, u32, u32), Option<u32>>,
    branches: &FullWalkBranches,
) -> Option<(u32, u32)> {
    let first = branches.first()?;
    if branches.len() < 2
        || !first.prune_guard.is_passed()
        || branches.iter().skip(1).any(|branch| {
            !branch.prune_guard.is_passed() || branch.parser_node != first.parser_node
        })
    {
        return None;
    }
    let lexer_state = match branches.as_slice() {
        [first, second] => transitions
            .union_states(&[first.lexer_state, second.lexer_state])
            .or_else(|| full_walk_projection_union_two(
                vocab,
                pair_union_cache,
                first.lexer_state,
                second.lexer_state,
            )),
        [first, second, third] => transitions
            .union_states(&[first.lexer_state, second.lexer_state, third.lexer_state])
            .or_else(|| full_walk_projection_union_three(
                vocab,
                triple_union_cache,
                first.lexer_state,
                second.lexer_state,
                third.lexer_state,
            )),
        _ => {
            let lexers = branches
                .iter()
                .map(|branch| branch.lexer_state)
                .collect::<SmallVec<[u32; 4]>>();
            transitions
                .union_states(&lexers)
                .or_else(|| vocab.mask_projection_state_for_projection_states(&lexers))
        }
    }?;
    Some((lexer_state, first.parser_node))
}

#[inline]
fn full_walk_many_state_from_branches(branches: FullWalkBranches) -> FullWalkManyState {
    if let [first, second, third] = branches.as_slice()
        && first.prune_guard.is_passed()
        && second.prune_guard.is_passed()
        && third.prune_guard.is_passed()
        && first.parser_node == second.parser_node
        && first.parser_node == third.parser_node
    {
        return FullWalkManyState::ThreeSameParser {
            lexers: (first.lexer_state, second.lexer_state, third.lexer_state),
            parser_node: first.parser_node,
        };
    }
    FullWalkManyState::Branches(branches)
}

#[cold]
#[inline(never)]
fn full_walk_step_guarded_pair_fallback<T: FullWalkTransitionTable>(
    pair: FullWalkGuardedPair,
    byte: u8,
    initial_lexer_state: u32,
    finalizer_code: &[u32],
    tokenizer: &Tokenizer,
    transitions: &T,
    parser_cache: &mut FullWalkParserCache,
    constraint: &Constraint,
) -> FullWalkBranches {
    let mut branches = FullWalkBranches::new();
    branches.push(FullWalkBranch {
        lexer_state: pair.continuing_lexer,
        parser_node: pair.continuing_parser,
        prune_guard: FullWalkPruneGuard::Passed,
    });
    branches.push(FullWalkBranch {
        lexer_state: initial_lexer_state,
        parser_node: pair.pending_parser,
        prune_guard: FullWalkPruneGuard::Pending(smallvec::smallvec![(
            pair.continuing_lexer,
            pair.guard_terminal,
        )]),
    });
    full_walk_step_many(
        &branches,
        byte,
        initial_lexer_state,
        finalizer_code,
        tokenizer,
        transitions,
        parser_cache,
        constraint,
    )
}

enum FullWalkGuardedStepOutcome {
    Dead,
    Scalar(u32, u32),
    Two((u32, u32), (u32, u32)),
    Guarded(FullWalkGuardedPair),
    Many(FullWalkManyState),
}

/// Record only a completed exact identity transition. In particular, both
/// parser coordinates and the pending terminal must agree, not just the lexer.
#[inline]
fn remember_guarded_identity(
    before: FullWalkGuardedPair,
    after: FullWalkGuardedPair,
    byte: u8,
    cached: &mut Option<FullWalkGuardedPair>,
    bytes: &mut [u64; 4],
) -> bool {
    if before != after { return false; }
    if *cached != Some(after) {
        *cached = Some(after);
        *bytes = [0; 4];
    }
    bytes[byte as usize >> 6] |= 1u64 << (byte & 63);
    true
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn full_walk_step_guarded_pair_bound<T: FullWalkTransitionTable>(
    pair: FullWalkGuardedPair,
    byte: u8,
    initial_lexer_state: u32,
    finalizer_code: &[u32],
    single_finalizer_continues: &[u8],
    tokenizer: &Tokenizer,
    transitions: &T,
    parser_cache: &mut FullWalkParserCache,
    constraint: &Constraint,
    vocab: &DynamicMaskVocab,
    pair_union_cache: &mut FxHashMap<(u32, u32), Option<u32>>,
    triple_union_cache: &mut FxHashMap<(u32, u32, u32), Option<u32>>,
    self_loop_state: &mut Option<FullWalkGuardedPair>,
    self_loop_bytes: &mut [u64; 4],
) -> FullWalkGuardedStepOutcome {
    let self_loop_word = byte as usize >> 6;
    let self_loop_bit = 1u64 << (byte & 63);
    if *self_loop_state == Some(pair) && self_loop_bytes[self_loop_word] & self_loop_bit != 0 {
        return FullWalkGuardedStepOutcome::Guarded(pair);
    }

    let cell = transitions.cell(pair.continuing_lexer, byte);
    if !T::cell_is_dead(cell) && T::cell_has_finalizer(cell) {
        let target = T::cell_target(cell);
        if transitions.finalizer_code(target, finalizer_code) == pair.guard_terminal
            && transitions.single_finalizer_continues(target, single_finalizer_continues)
        {
            return FullWalkGuardedStepOutcome::Guarded(FullWalkGuardedPair {
                continuing_lexer: target,
                ..pair
            });
        }
    }

    let next = full_walk_step_guarded_pair_fallback(
        pair,
        byte,
        initial_lexer_state,
        finalizer_code,
        tokenizer,
        transitions,
        parser_cache,
        constraint,
    );
    match next.as_slice() {
        [] => FullWalkGuardedStepOutcome::Dead,
        [branch] if branch.prune_guard.is_passed() => {
            FullWalkGuardedStepOutcome::Scalar(branch.lexer_state, branch.parser_node)
        }
        [first, second] if first.prune_guard.is_passed() && second.prune_guard.is_passed() => {
            if let Some((lexer_state, parser_node)) = full_walk_merge_two_same_parser(
                transitions,
                vocab,
                pair_union_cache,
                (first.lexer_state, first.parser_node),
                (second.lexer_state, second.parser_node),
            ) {
                FullWalkGuardedStepOutcome::Scalar(lexer_state, parser_node)
            } else {
                FullWalkGuardedStepOutcome::Two(
                    (first.lexer_state, first.parser_node),
                    (second.lexer_state, second.parser_node),
                )
            }
        }
        _ => {
            if let Some(guarded) = full_walk_guarded_pair_from_branches(&next, initial_lexer_state)
            {
                if guarded == pair {
                    if *self_loop_state != Some(guarded) {
                        *self_loop_state = Some(guarded);
                        *self_loop_bytes = [0; 4];
                    }
                    self_loop_bytes[self_loop_word] |= self_loop_bit;
                }
                FullWalkGuardedStepOutcome::Guarded(guarded)
            } else if let Some((lexer_state, parser_node)) = full_walk_merge_branches_same_parser(
                transitions,
                vocab,
                pair_union_cache,
                triple_union_cache,
                &next,
            ) {
                FullWalkGuardedStepOutcome::Scalar(lexer_state, parser_node)
            } else {
                FullWalkGuardedStepOutcome::Many(full_walk_many_state_from_branches(next))
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn full_walk_step_many_state<T: FullWalkTransitionTable>(
    state: &FullWalkManyState,
    byte: u8,
    initial_lexer_state: u32,
    finalizer_code: &[u32],
    single_finalizer_continues: &[u8],
    tokenizer: &Tokenizer,
    transitions: &T,
    parser_cache: &mut FullWalkParserCache,
    constraint: &Constraint,
) -> FullWalkManyState {
    match state {
        FullWalkManyState::Branches(branches) => full_walk_many_state_from_branches(
            full_walk_step_many(
                branches,
                byte,
                initial_lexer_state,
                finalizer_code,
                tokenizer,
                transitions,
                parser_cache,
                constraint,
            ),
        ),
        FullWalkManyState::ThreeSameParser {
            lexers,
            parser_node,
        } => {
            let first = transitions.cell(lexers.0, byte);
            let second = transitions.cell(lexers.1, byte);
            let third = transitions.cell(lexers.2, byte);
            if !T::cell_has_finalizer(first)
                && !T::cell_has_finalizer(second)
                && !T::cell_has_finalizer(third)
                && !T::cell_is_dead(first)
                && !T::cell_is_dead(second)
                && !T::cell_is_dead(third)
            {
                let next = (
                    T::cell_target(first),
                    T::cell_target(second),
                    T::cell_target(third),
                );
                if next.0 != next.1
                    && next.0 != next.2
                    && next.1 != next.2
                    && parser_cache.physical_token_boundary_allowed(
                        constraint,
                        tokenizer,
                        transitions,
                        *parser_node,
                        next.0,
                    )
                    && parser_cache.physical_token_boundary_allowed(
                        constraint,
                        tokenizer,
                        transitions,
                        *parser_node,
                        next.1,
                    )
                    && parser_cache.physical_token_boundary_allowed(
                        constraint,
                        tokenizer,
                        transitions,
                        *parser_node,
                        next.2,
                    )
                {
                    return FullWalkManyState::ThreeSameParser {
                        lexers: next,
                        parser_node: *parser_node,
                    };
                }
            }

            let mut branches = FullWalkBranches::new();
            for lexer_state in [lexers.0, lexers.1, lexers.2] {
                branches.push(FullWalkBranch {
                    lexer_state,
                    parser_node: *parser_node,
                    prune_guard: FullWalkPruneGuard::Passed,
                });
            }
            full_walk_many_state_from_branches(full_walk_step_many(
                &branches,
                byte,
                initial_lexer_state,
                finalizer_code,
                tokenizer,
                transitions,
                parser_cache,
                constraint,
            ))
        }
    }
}


#[inline(always)]
fn full_walk_skip_lexically_dead_subtree<'a>(
    vocab: &DynamicMaskVocab,
    trie: &DynamicMaskTrie,
    walk_ops: &'a [DynamicMaskTrieFullWalkOp],
    remaining_ops: &mut std::slice::Iter<'a, DynamicMaskTrieFullWalkOp>,
    token_marker_index: &mut usize,
    buf: &mut [u32],
    clear_dead_tokens: bool,
    deferred_dead_subtrees: &mut Vec<u32>,
    record_deferred: bool,
) -> usize {
    let op_index = walk_ops.len() - remaining_ops.as_slice().len() - 1;
    let (child, subtree_end_op) = trie.full_walk_dead_subtree(op_index);

    // Full-vocabulary walks start with every vocabulary token admitted, so a
    // dead subtree must clear its token bits. Residual-slice walks invert this:
    // the entire residual is bulk-cleared before traversal and surviving token
    // endpoints are set back. In residual mode, a dead-subtree skip is therefore
    // only cursor movement; doing per-token clears here would duplicate work.
    if record_deferred {
        deferred_dead_subtrees.push(child);
    }
    let cleared = if clear_dead_tokens || record_deferred {
        let tokens = vocab.subtree_original_tokens_for(trie, child);
        if clear_dead_tokens {
            for &token_id in tokens {
                clear_mask_bit_known_in_range(buf, token_id);
            }
        }
        tokens.len()
    } else {
        0
    };

    // Token markers follow the same DFS token order as subtree metadata. The
    // root is normally not a token; account for an empty-token vocabulary
    // defensively so the cursor remains aligned after the jump.
    let root_token_offset = usize::from(trie.node(0).token_id.is_some());
    let token_end = trie
        .subtree_token_index_range(child)
        .end
        .saturating_sub(root_token_offset);
    debug_assert!(*token_marker_index <= token_end);
    *token_marker_index = token_end;
    *remaining_ops = walk_ops[subtree_end_op as usize..].iter();
    cleared
}

#[inline(always)]
fn full_walk_maybe_commit_deferred_positive(
    vocab: &DynamicMaskVocab,
    total_original_tokens: usize,
    deferred_output: &mut bool,
    positive_rebuild: &mut bool,
    deferred_negative_mutations: usize,
    deferred_allowed_markers: &mut Vec<u64>,
    deferred_rejected_markers: &mut Vec<u64>,
    deferred_dead_subtrees: &mut Vec<u32>,
    buf: &mut [u32],
) {
    if vocab.is_grammar_quotiented()
        || !*deferred_output
        || deferred_negative_mutations <= total_original_tokens / 2
    {
        return;
    }

    full_walk_commit_deferred_positive(
        vocab,
        deferred_output,
        positive_rebuild,
        deferred_allowed_markers,
        deferred_rejected_markers,
        deferred_dead_subtrees,
        buf,
    );
}

#[cold]
#[inline(never)]
fn full_walk_commit_deferred_positive(
    vocab: &DynamicMaskVocab,
    deferred_output: &mut bool,
    positive_rebuild: &mut bool,
    deferred_allowed_markers: &mut Vec<u64>,
    deferred_rejected_markers: &mut Vec<u64>,
    deferred_dead_subtrees: &mut Vec<u32>,
    buf: &mut [u32],
) {
    // The output buffer is still the deferred-mode zero baseline. Once more
    // than half of the original vocabulary has been rejected, materializing
    // the positive side can no longer require more token mutations than the
    // negative side. Commit every positive endpoint observed so far, then
    // discard rejection metadata and continue writing positives directly.
    for marker in deferred_allowed_markers.drain(..) {
        mark_dynamic_token_marker(vocab, marker, buf);
    }
    deferred_rejected_markers.clear();
    deferred_dead_subtrees.clear();
    *deferred_output = false;
    *positive_rebuild = true;
}

#[inline(always)]
fn dynamic_token_marker_original_count(vocab: &DynamicMaskVocab, marker: u64) -> usize {
    debug_assert_ne!(marker, 0);
    if marker & DYNAMIC_TOKEN_MARKER_FALLBACK == 0 {
        return (marker as u32).count_ones() as usize;
    }
    let canonical_token = ((marker & !DYNAMIC_TOKEN_MARKER_FALLBACK) - 1) as u32;
    vocab
        .token_ids(canonical_token)
        .expect("dynamic vocabulary trie node lacks token ids")
        .len()
}

#[inline(always)]
fn dynamic_token_marker_materialization_cost(vocab: &DynamicMaskVocab, marker: u64) -> usize {
    debug_assert_ne!(marker, 0);
    if marker & DYNAMIC_TOKEN_MARKER_FALLBACK == 0 {
        return 1;
    }
    let canonical_token = ((marker & !DYNAMIC_TOKEN_MARKER_FALLBACK) - 1) as u32;
    vocab.token_word_masks(canonical_token).len()
}

/// Exact direct dynamic-mask path for bounded deterministic lexer coordinates.
///
/// This performs the exact vocabulary walk, skipping only a child subtree once
/// the current lexer branch is already proven dead. It does not use subtree
/// certificates, segment-effect caches, recognizer-state interning, or any
/// speculative admission rule. Unsupported lexer or composition shapes return
/// `false` and use the existing exact fallback.

fn residual_regex_slice_prefix_contained(
    tokenizer: &Tokenizer,
    source: u32,
    terminal: TerminalID,
    slice: &crate::compiler::stages::id_map_and_terminal_dwa::classify::VocabPartitionDfa,
    work_limit: usize,
) -> Option<bool> {
    let coordinates = tokenizer.terminal_residual_coordinates()?;
    let residual_state = coordinates
        .row(source)?
        .iter()
        .find_map(|&(candidate, state)| (candidate == terminal).then_some(state))?;
    let residual = coordinates.terminal_dfa(terminal)?;
    let mut seen = FxHashSet::<(u32, u32)>::default();
    let mut queue = std::collections::VecDeque::from([(slice.start_state(), residual_state)]);
    let mut work = 0usize;
    while let Some((slice_state, regex_state)) = queue.pop_front() {
        if !seen.insert((slice_state, regex_state)) {
            continue;
        }
        for byte in 0u16..=255 {
            let byte = byte as u8;
            let slice_target = slice.step(slice_state, byte);
            if !slice.can_reach_accepting(slice_target) {
                continue;
            }
            work = work.saturating_add(1);
            if work > work_limit {
                return None;
            }
            let Some(regex_target) = residual.step(regex_state, byte) else {
                return Some(false);
            };
            let regex_live = residual.finalizers(regex_target).contains(0)
                || residual.possible_future_group_ids(regex_target).contains(0);
            if !regex_live {
                return Some(false);
            }
            if !seen.contains(&(slice_target, regex_target)) {
                queue.push_back((slice_target, regex_target));
            }
        }
    }
    Some(true)
}


const LLG_SAFE_PLUS_SLICE: usize = 0;
const LLG_WHITESPACE_SLICE: usize = 3;
// Proof cache IDs 0 and 3 are intentionally stable. Slot count only needs to
// cover the largest stable ID; there are no fixed bounded-length proof slots.
const LLG_PROOF_SLOT_COUNT: usize = LLG_WHITESPACE_SLICE + 1;

pub(super) fn virtual_residual_slice_prefix_contained(
    tokenizer: &Tokenizer,
    source: u32,
    terminal: TerminalID,
    slice: &crate::compiler::stages::id_map_and_terminal_dwa::classify::VocabPartitionDfa,
    work_limit: usize,
) -> Option<bool> {
    let mut found = false;
    let mut unknown = false;
    for residual_state in tokenizer.singleton_epsilon_closure(source) {
        if tokenizer.virtual_residual_terminal_for_state(residual_state) != Some(terminal) {
            continue;
        }
        found = true;
        match tokenizer.virtual_residual_parser_transparent_byte_dfa(
            residual_state,
            slice.start_state(),
            slice.class_count(),
            slice.byte_to_class_map(),
            slice.transition_table(),
            slice.can_reach_accepting_map(),
            slice.has_finite_language(),
            work_limit,
        ) {
            Some(true) => return Some(true),
            Some(false) => {}
            None => unknown = true,
        }
    }
    if !found || unknown {
        None
    } else {
        Some(false)
    }
}

pub(super) fn virtual_residual_safe_repeat_radius(
    tokenizer: &Tokenizer,
    source: u32,
    terminal: TerminalID,
    safe_plus: &crate::compiler::stages::id_map_and_terminal_dwa::classify::VocabPartitionDfa,
    max_repetitions: u32,
    work_limit: usize,
) -> Option<u32> {
    let mut best = None;
    let mut found = false;
    for residual_state in tokenizer.singleton_epsilon_closure(source) {
        if tokenizer.virtual_residual_terminal_for_state(residual_state) != Some(terminal) {
            continue;
        }
        found = true;
        if let Some(radius) = tokenizer.virtual_residual_parser_transparent_byte_dfa_repeat_radius(
            residual_state,
            safe_plus.start_state(),
            safe_plus.class_count(),
            safe_plus.byte_to_class_map(),
            safe_plus.transition_table(),
            safe_plus.accepting_map(),
            safe_plus.can_reach_accepting_map(),
            max_repetitions,
            work_limit,
        ) {
            best = Some(best.map_or(radius, |current: u32| current.max(radius)));
        }
    }
    found.then_some(best).flatten()
}
fn direct_slice_prefix_contained<T: FullWalkTransitionTable>(
    transitions: &T,
    tokenizer: &Tokenizer,
    start: u32,
    terminal: TerminalID,
    slice: &crate::compiler::stages::id_map_and_terminal_dwa::classify::VocabPartitionDfa,
    work_limit: usize,
) -> Option<bool> {
    let mut seen = FxHashSet::<(u32, u32)>::default();
    let mut queue = std::collections::VecDeque::from([(slice.start_state(), start)]);
    let mut work = 0usize;
    while let Some((slice_state, lexer_state)) = queue.pop_front() {
        if !seen.insert((slice_state, lexer_state)) {
            continue;
        }
        for byte in 0u16..=255 {
            let byte = byte as u8;
            let slice_target = slice.step(slice_state, byte);
            if !slice.can_reach_accepting(slice_target) {
                continue;
            }
            work = work.saturating_add(1);
            if work > work_limit {
                return None;
            }
            let cell = transitions.cell(lexer_state, byte);
            if T::cell_is_dead(cell) {
                return Some(false);
            }
            let target = T::cell_target(cell);
            let terminal_live = transitions.future_contains(tokenizer, target, terminal)
                || transitions
                    .matched_terminals(tokenizer, target)
                    .contains(&terminal);
            if !terminal_live {
                return Some(false);
            }
            if !seen.contains(&(slice_target, target)) {
                queue.push_back((slice_target, target));
            }
        }
    }
    Some(true)
}

fn debug_direct_slice_prefix_counterexample<T: FullWalkTransitionTable>(
    transitions: &T,
    tokenizer: &Tokenizer,
    start: u32,
    terminal: TerminalID,
    slice: &crate::compiler::stages::id_map_and_terminal_dwa::classify::VocabPartitionDfa,
    work_limit: usize,
) -> (Option<bool>, Vec<u8>, &'static str) {
    let mut seen = FxHashSet::<(u32, u32)>::default();
    let mut queue = std::collections::VecDeque::from([(
        slice.start_state(),
        start,
        Vec::<u8>::new(),
    )]);
    let mut work = 0usize;
    while let Some((slice_state, lexer_state, path)) = queue.pop_front() {
        if !seen.insert((slice_state, lexer_state)) {
            continue;
        }
        for byte in 0u16..=255 {
            let byte = byte as u8;
            let slice_target = slice.step(slice_state, byte);
            if !slice.can_reach_accepting(slice_target) {
                continue;
            }
            work += 1;
            if work > work_limit {
                return (None, path, "budget");
            }
            let mut next_path = path.clone();
            next_path.push(byte);
            let cell = transitions.cell(lexer_state, byte);
            if T::cell_is_dead(cell) {
                return (Some(false), next_path, "dead-transition");
            }
            let target = T::cell_target(cell);
            let terminal_live = transitions.future_contains(tokenizer, target, terminal)
                || transitions
                    .matched_terminals(tokenizer, target)
                    .contains(&terminal);
            if !terminal_live {
                return (Some(false), next_path, "terminal-not-live");
            }
            if next_path.len() < 96 && !seen.contains(&(slice_target, target)) {
                queue.push_back((slice_target, target, next_path));
            }
        }
    }
    (Some(true), Vec::new(), "contained")
}


#[inline]
fn is_transparent_json_string_chunk_terminal(constraint: &Constraint, terminal: TerminalID) -> bool {
    constraint
        .terminal_display_names
        .get(terminal as usize)
        .is_some_and(|name| name.starts_with("json_string_char_"))
}

fn transparent_json_string_terminals(
    parser_cache: &mut FullWalkParserCache,
    constraint: &Constraint,
    parser_node: u32,
) -> BitSet {
    let mut transparent = parser_cache.admitted(constraint, parser_node).clone();
    let terminals = transparent.iter_ones().collect::<Vec<_>>();
    for terminal in terminals {
        if !is_transparent_json_string_chunk_terminal(constraint, terminal as TerminalID) {
            transparent.clear(terminal);
        }
    }
    transparent
}

/// Exact bounded containment through GLRMask's implementation-level JSON-string
/// chunk terminals. Long bounded semantic strings are deliberately lowered into
/// a sequence of `json_string_char_*` parser terminals. llguidance keeps this as
/// one lexeme, so a faithful slice proof must allow those internal finalizations
/// and parser advances while refusing every unrelated grammar terminal.
fn bounded_string_chunk_slice_contained<T: FullWalkTransitionTable>(
    transitions: &T,
    tokenizer: &Tokenizer,
    start_lexer_state: u32,
    reset_lexer_state: u32,
    initial_parser_node: u32,
    initial_guard: FullWalkPruneGuard,
    slice: &crate::compiler::stages::id_map_and_terminal_dwa::classify::VocabPartitionDfa,
    finalizer_code: &[u32],
    parser_cache: &mut FullWalkParserCache,
    constraint: &Constraint,
    work_limit: usize,
) -> Option<bool> {
    const NONE: u32 = u32::MAX;
    const MULTI: u32 = u32::MAX - 1;

    let mut start_branches = FullWalkBranches::new();
    start_branches.push(FullWalkBranch {
        lexer_state: start_lexer_state,
        parser_node: initial_parser_node,
        prune_guard: initial_guard,
    });
    let mut queue = std::collections::VecDeque::from([(
        slice.start_state(),
        start_branches,
    )]);
    let mut seen = FxHashSet::<(u32, FullWalkBranches)>::default();
    let mut work = 0usize;

    while let Some((slice_state, mut branches)) = queue.pop_front() {
        branches.sort_unstable();
        branches.dedup();
        if !seen.insert((slice_state, branches.clone())) {
            continue;
        }

        for byte in 0u16..=255 {
            let byte = byte as u8;
            let slice_target = slice.step(slice_state, byte);
            if !slice.can_reach_accepting(slice_target) {
                continue;
            }
            work = work.saturating_add(1);
            if work > work_limit {
                return None;
            }

            let mut next = FullWalkBranches::new();
            for branch in &branches {
                let Some(advanced_guard) = branch.prune_guard.advance(tokenizer, transitions, byte)
                else {
                    continue;
                };
                let target = transitions.transition(branch.lexer_state, byte);
                if target == u32::MAX {
                    continue;
                }

                let code = transitions.finalizer_code(target, finalizer_code);
                if code == MULTI {
                    for terminal in transitions.matched_terminals(tokenizer, target) {
                        if !is_transparent_json_string_chunk_terminal(constraint, terminal) {
                            continue;
                        }
                        if !parser_cache.terminal_not_known_inadmissible(
                            constraint,
                            branch.parser_node,
                            terminal,
                        ) {
                            continue;
                        }
                        if let Some(parser_node) =
                            parser_cache.advance(constraint, branch.parser_node, terminal)
                        {
                            full_walk_push_unique(
                                &mut next,
                                FullWalkBranch {
                                    lexer_state: reset_lexer_state,
                                    parser_node,
                                    prune_guard: advanced_guard.remember_terminal_match(
                                        tokenizer,
                                        transitions,
                                        target,
                                        terminal,
                                    ),
                                },
                            );
                        }
                    }
                } else if code != NONE
                    && is_transparent_json_string_chunk_terminal(constraint, code)
                    && parser_cache.terminal_not_known_inadmissible(
                        constraint,
                        branch.parser_node,
                        code,
                    )
                    && let Some(parser_node) =
                        parser_cache.advance(constraint, branch.parser_node, code)
                {
                    full_walk_push_unique(
                        &mut next,
                        FullWalkBranch {
                            lexer_state: reset_lexer_state,
                            parser_node,
                            prune_guard: advanced_guard.remember_terminal_match(
                                tokenizer,
                                transitions,
                                target,
                                code,
                            ),
                        },
                    );
                }

                // Continue the current lexical branch only while some parser-
                // admitted transparent string-chunk terminal can still complete.
                let transparent = transparent_json_string_terminals(
                    parser_cache,
                    constraint,
                    branch.parser_node,
                );
                if !transparent.is_empty()
                    && transitions.future_intersects(tokenizer, target, &transparent)
                {
                    full_walk_push_unique(
                        &mut next,
                        FullWalkBranch {
                            lexer_state: target,
                            parser_node: branch.parser_node,
                            prune_guard: advanced_guard,
                        },
                    );
                }
            }

            if next.is_empty() {
                return Some(false);
            }
            next.sort_unstable();
            next.dedup();
            if !seen.contains(&(slice_target, next.clone())) {
                queue.push_back((slice_target, next));
            }
        }
    }
    Some(true)
}

const DENSE_HOT_LEXER_TWO_DISTINCT: u32 = u32::MAX - 3;
const DENSE_HOT_LEXER_TWO: u32 = u32::MAX - 2;
const DENSE_HOT_LEXER_DEAD: u32 = u32::MAX;

enum DenseHotLaneOutcome {
    Dead,
    Scalar(u32, u32),
    Two((u32, u32), (u32, u32)),
    Decline,
}

/// Populate one entry in the tiny scalar transition cache.  The cache sits
/// above `FullWalkTransitionTable`, so every scalar lexer coordinate -- raw
/// Flat16/Flat32, config, or a synthetic lazy-union state -- gets the same
/// `u8 state x byte -> u8 state` hot path.  Finalizing transitions stay cold.
#[cold]
#[inline(never)]
fn dense_hot_transition_miss<T: FullWalkTransitionTable>(
    hot: &mut FullWalkHotScalarCache,
    transitions: &T,
    hot_id: u8,
    byte: u8,
) -> u8 {
    let Some(&source) = hot.hot_to_raw.get(hot_id as usize) else {
        return FULL_WALK_HOT_SLOW;
    };
    let cell = transitions.cell(source, byte);
    let value = if T::cell_is_dead(cell) {
        FULL_WALK_HOT_DEAD
    } else if T::cell_has_finalizer(cell) {
        FULL_WALK_HOT_SLOW
    } else {
        let target = T::cell_target(cell);
        hot.intern(target).unwrap_or(FULL_WALK_HOT_SLOW)
    };
    if value == FULL_WALK_HOT_SLOW {
        hot.slow += 1;
    }
    hot.misses += 1;
    unsafe {
        *hot.rows
            .get_unchecked_mut(hot_id as usize)
            .get_unchecked_mut(byte as usize) = value;
    }
    value
}

/// The active parser node changes rarely.  When it does, refresh the tiny
/// liveness view for every lexer coordinate already interned in the hot cache.
/// The byte-hot loop can then test parser-conditioned liveness by hot id rather
/// than mapping the id back to a wide lexer coordinate on every transition.
#[inline(never)]
fn dense_hot_refresh_liveness(
    hot: &FullWalkHotScalarCache,
    boundary_row: *const u8,
    hot_liveness: &mut [u8; FULL_WALK_HOT_CAPACITY],
) {
    for (id, &lexer_state) in hot.hot_to_raw.iter().enumerate() {
        hot_liveness[id] = unsafe { *boundary_row.add(lexer_state as usize) };
    }
}

#[allow(clippy::too_many_arguments)]
#[cold]
#[inline(never)]
fn dense_hot_scalar_escape<T: FullWalkTransitionTable>(
    source_lexer: u32,
    byte: u8,
    parser_node: u32,
    initial_lexer_state: u32,
    finalizer_code: &[u32],
    single_finalizer_continues: &[u8],
    tokenizer: &Tokenizer,
    transitions: &T,
    parser_cache: &mut FullWalkParserCache,
    constraint: &Constraint,
    vocab: &DynamicMaskVocab,
    pair_union_cache: &mut FxHashMap<(u32, u32), Option<u32>>,
) -> DenseHotLaneOutcome {
    let cell = transitions.cell(source_lexer, byte);
    if T::cell_is_dead(cell) {
        return DenseHotLaneOutcome::Dead;
    }
    let target = T::cell_target(cell);
    if !T::cell_has_finalizer(cell) {
        return DenseHotLaneOutcome::Scalar(target, parser_node);
    }

    let mut lexer = target;
    let mut parser = parser_node;
    let mut two = ((0u32, 0u32), (0u32, 0u32));
    if full_walk_try_apply_plain_single_finalizer(
        target,
        parser_node,
        initial_lexer_state,
        finalizer_code,
        single_finalizer_continues,
        transitions,
        parser_cache,
        constraint,
        DENSE_HOT_LEXER_TWO_DISTINCT,
        &mut lexer,
        &mut parser,
        &mut two,
    ) {
        return if lexer == DENSE_HOT_LEXER_TWO_DISTINCT {
            DenseHotLaneOutcome::Two(two.0, two.1)
        } else {
            DenseHotLaneOutcome::Scalar(lexer, parser)
        };
    }

    match full_walk_scalar_finalizer_hot_single(
        target,
        parser_node,
        initial_lexer_state,
        finalizer_code,
        single_finalizer_continues,
        tokenizer,
        transitions,
        parser_cache,
        constraint,
    ) {
        FullWalkScalarFinalizerOutcome::Scalar(branch) if branch.prune_guard.is_passed() => {
            DenseHotLaneOutcome::Scalar(branch.lexer_state, branch.parser_node)
        }
        FullWalkScalarFinalizerOutcome::Two(first, second)
            if first.prune_guard.is_passed() && second.prune_guard.is_passed() =>
        {
            if let Some((lexer_state, parser_node)) = full_walk_merge_two_same_parser(
                transitions,
                vocab,
                pair_union_cache,
                (first.lexer_state, first.parser_node),
                (second.lexer_state, second.parser_node),
            ) {
                DenseHotLaneOutcome::Scalar(lexer_state, parser_node)
            } else {
                DenseHotLaneOutcome::Two(
                    (first.lexer_state, first.parser_node),
                    (second.lexer_state, second.parser_node),
                )
            }
        }
        FullWalkScalarFinalizerOutcome::Many(mut branches)
            if branches.len() == 2
                && branches.iter().all(|branch| branch.prune_guard.is_passed()) =>
        {
            let second = branches.pop().expect("second dense hot branch disappeared");
            let first = branches.pop().expect("first dense hot branch disappeared");
            if let Some((lexer_state, parser_node)) = full_walk_merge_two_same_parser(
                transitions,
                vocab,
                pair_union_cache,
                (first.lexer_state, first.parser_node),
                (second.lexer_state, second.parser_node),
            ) {
                DenseHotLaneOutcome::Scalar(lexer_state, parser_node)
            } else {
                DenseHotLaneOutcome::Two(
                    (first.lexer_state, first.parser_node),
                    (second.lexer_state, second.parser_node),
                )
            }
        }
        _ => DenseHotLaneOutcome::Decline,
    }
}

#[allow(clippy::too_many_arguments)]
#[cold]
#[inline(never)]
fn dense_hot_two_escape<T: FullWalkTransitionTable>(
    branches: ((u32, u32), (u32, u32)),
    byte: u8,
    initial_lexer_state: u32,
    finalizer_code: &[u32],
    single_finalizer_continues: &[u8],
    tokenizer: &Tokenizer,
    transitions: &T,
    parser_cache: &mut FullWalkParserCache,
    constraint: &Constraint,
    vocab: &DynamicMaskVocab,
    pair_union_cache: &mut FxHashMap<(u32, u32), Option<u32>>,
) -> DenseHotLaneOutcome {
    match full_walk_step_two(
        branches,
        byte,
        initial_lexer_state,
        finalizer_code,
        single_finalizer_continues,
        tokenizer,
        transitions,
        parser_cache,
        constraint,
    ) {
        FullWalkTwoStepOutcome::Dead => DenseHotLaneOutcome::Dead,
        FullWalkTwoStepOutcome::One((lexer_state, parser_node)) => {
            DenseHotLaneOutcome::Scalar(lexer_state, parser_node)
        }
        FullWalkTwoStepOutcome::Two(first, second) => {
            if let Some((lexer_state, parser_node)) = full_walk_merge_two_same_parser(
                transitions,
                vocab,
                pair_union_cache,
                first,
                second,
            ) {
                DenseHotLaneOutcome::Scalar(lexer_state, parser_node)
            } else {
                DenseHotLaneOutcome::Two(first, second)
            }
        }
        FullWalkTwoStepOutcome::Many(_) => DenseHotLaneOutcome::Decline,
    }
}

#[cold]
#[inline(never)]
fn dense_hot_physical_boundary_miss<T: FullWalkTransitionTable>(
    parser_cache: &mut FullWalkParserCache,
    constraint: &Constraint,
    tokenizer: &Tokenizer,
    transitions: &T,
    parser_node: u32,
    lexer_state: u32,
) -> bool {
    parser_cache.physical_token_boundary_allowed(
        constraint,
        tokenizer,
        transitions,
        parser_node,
        lexer_state,
    )
}

#[inline(always)]
fn dense_hot_skip_dead_subtree<'a, const POSITIVE: bool, const OBSERVE_DENSITY: bool>(
    vocab: &DynamicMaskVocab,
    trie: &DynamicMaskTrie,
    walk_ops: &'a [DynamicMaskTrieFullWalkOp],
    remaining_ops: &mut std::slice::Iter<'a, DynamicMaskTrieFullWalkOp>,
    token_marker_index: &mut usize,
    skipped_original_tokens: &mut usize,
    buf: &mut [u32],
) {
    let op_index = walk_ops.len() - remaining_ops.as_slice().len() - 1;
    let (child, subtree_end_op) = trie.full_walk_dead_subtree(op_index);
    if OBSERVE_DENSITY {
        *skipped_original_tokens = skipped_original_tokens
            .saturating_add(vocab.subtree_original_tokens_for(trie, child).len());
    }
    if !POSITIVE {
        for &token_id in vocab.subtree_original_tokens_for(trie, child) {
            clear_mask_bit_known_in_range(buf, token_id);
        }
    }
    let root_token_offset = usize::from(trie.node(0).token_id.is_some());
    *token_marker_index = trie
        .subtree_token_index_range(child)
        .end
        .saturating_sub(root_token_offset);
    *remaining_ops = walk_ops[subtree_end_op as usize..].iter();
}

#[derive(Clone, Copy, Debug, Default)]
struct DenseHotRootProbe {
    prefer_pruning: bool,
    known_live_original_tokens: usize,
    known_dead_original_tokens: usize,
    total_root_original_tokens: usize,
}

#[allow(clippy::too_many_arguments)]
#[cold]
#[inline(never)]
fn dense_hot_root_probe<T: FullWalkTransitionTable>(
    state: &ConstraintState<'_>,
    vocab: &DynamicMaskVocab,
    trie: &DynamicMaskTrie,
    root_lexer: u32,
    root_parser: u32,
    tokenizer: &Tokenizer,
    transitions: &T,
    parser_cache: &mut FullWalkParserCache,
    hot_scalar: &mut FullWalkHotScalarCache,
    certified_byte: Option<u8>,
) -> DenseHotRootProbe {
    if !trie.has_full_walk_root_byte_index() {
        return DenseHotRootProbe {
            prefer_pruning: true,
            ..DenseHotRootProbe::default()
        };
    }

    // Routing only: all selected kernels are logically exact. Parser-dead mass
    // decides whether interior pruning is worth paying for. If pruning is not
    // needed, the same complete root scan also gives a cheap density signal for
    // output polarity: overwhelmingly parser-live root mass predicts a dense
    // final mask without putting any policy branch into the byte-hot loop.
    const DEAD_ORIGINAL_TOKENS_TRIGGER: usize = 256;
    let mut probe = DenseHotRootProbe::default();
    let Some(root_hot_id) = hot_scalar.intern(root_lexer) else {
        probe.prefer_pruning = true;
        return probe;
    };

    // Experiment E: when a single certified live root byte is supplied, probe
    // only it. Every omitted byte is lexically dead at this root lexer
    // coordinate (single-live-root-byte proof), so it only ever populates a
    // DEAD cache entry and can never set `prefer_pruning`. Bound the loop to
    // that single byte so E actually performs one iteration.
    let (raw_byte_start, raw_byte_end) = match certified_byte {
        Some(certified) => (certified as u16, certified as u16),
        None => (0u16, 255u16),
    };
    for raw_byte in raw_byte_start..=raw_byte_end {
        let byte = raw_byte as u8;
        let Some((start_op, _, _)) = trie.full_walk_root_byte_range(byte) else {
            continue;
        };
        let (child, _) = trie.full_walk_dead_subtree(start_op as usize);
        let represented = vocab.subtree_original_tokens_for(trie, child).len();
        probe.total_root_original_tokens =
            probe.total_root_original_tokens.saturating_add(represented);

        let mut next = unsafe {
            *hot_scalar
                .rows
                .get_unchecked(root_hot_id as usize)
                .get_unchecked(byte as usize)
        };
        if next == FULL_WALK_HOT_UNKNOWN {
            next = dense_hot_transition_miss(hot_scalar, transitions, root_hot_id, byte);
        }
        if next == FULL_WALK_HOT_DEAD {
            // Lexical death is already encoded in the universal hot transition
            // cache in every kernel, so it is not evidence for parser pruning
            // and it cannot contribute to a dense output mask.
            continue;
        }
        if next == FULL_WALK_HOT_SLOW {
            // Finalizer/complex root transitions are inconclusive for routing.
            // The cached SLOW result is still reused by the actual walk.
            continue;
        }

        let target = unsafe { *hot_scalar.hot_to_raw.get_unchecked(next as usize) };
        if parser_cache.physical_token_boundary_allowed(
            state.constraint,
            tokenizer,
            transitions,
            root_parser,
            target,
        ) {
            probe.known_live_original_tokens =
                probe.known_live_original_tokens.saturating_add(represented);
        } else {
            probe.known_dead_original_tokens =
                probe.known_dead_original_tokens.saturating_add(represented);
            if probe.known_dead_original_tokens >= DEAD_ORIGINAL_TOKENS_TRIGGER {
                probe.prefer_pruning = true;
                return probe;
            }
        }
    }
    probe
}

/// Narrow scalar/two-state lane for scalar-dispatch masks. Every scalar lexer
/// coordinate is interned into the same tiny u8 transition cache regardless of
/// the underlying transition provider. PRUNE_INTERIOR is compile-time selected
/// so the genuine dense lane pays no parser-liveness branch in its byte loop.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn try_dense_hot_scalar_edges<
    T: FullWalkTransitionTable,
    const POSITIVE: bool,
    const PRUNE_INTERIOR: bool,
    const OBSERVE_DENSITY: bool,
    const BOUNDED_ROOT: bool,
>(
    state: &ConstraintState<'_>,
    vocab: &DynamicMaskVocab,
    trie: &DynamicMaskTrie,
    root_lexer: u32,
    root_parser: u32,
    initial_lexer_state: u32,
    finalizer_code: &[u32],
    single_finalizer_continues: &[u8],
    tokenizer: &Tokenizer,
    transitions: &T,
    parser_cache: &mut FullWalkParserCache,
    mut hot_scalar: FullWalkHotScalarCache,
    root_range: Option<(u32, u32, usize)>,
    buf: &mut [u32],
) -> Result<Option<bool>, String> {
    if !trie.full_walk_all_consume() || trie.full_walk_max_parent_depth() >= 255 {
        return Ok(None);
    }
    // Experiment E: a bounded walk is only ever the positive, non-observing
    // specialization. Negative polarity would leave out-of-range (lexically
    // dead) bits set, and observing density over a partial range would corrupt
    // the global dense-output hint.
    if BOUNDED_ROOT && (!POSITIVE || OBSERVE_DENSITY) {
        return Ok(None);
    }

    let Some(initial_hot_id) = hot_scalar.intern(initial_lexer_state) else {
        return Ok(None);
    };
    let Some(root_hot_id) = hot_scalar.intern(root_lexer) else {
        return Ok(None);
    };
    // Dedicated hot-walker coordinate: 0..252 are scalar hot IDs; the top
    // three byte values are local two-branch/dead sentinels.  Keep the wide
    // underlying lexer coordinate entirely out of the common byte loop.
    const HOT_TWO_DISTINCT: u8 = FULL_WALK_HOT_SLOW;
    const HOT_TWO: u8 = FULL_WALK_HOT_DEAD;
    const HOT_DEAD: u8 = FULL_WALK_HOT_UNKNOWN;
    let mut stack_lexer = [HOT_DEAD; 256];
    let mut stack_parser = [0u32; 256];
    let mut stack_two = [((0u32, 0u32), (0u32, 0u32)); 256];
    stack_lexer[0] = root_hot_id;
    stack_parser[0] = root_parser;

    let mut pair_union_cache = FxHashMap::<(u32, u32), Option<u32>>::default();
    let walk_ops = trie.full_walk_ops();
    let token_markers = vocab.full_walk_token_markers_for(trie);
    let mut token_marker_index = 0usize;
    let mut skipped_original_tokens = 0usize;
    let mut remaining_ops = walk_ops.iter();

    // Experiment E bounded-root setup. This is a compile-time specialization:
    // the legacy BOUNDED_ROOT=false monomorph keeps `remaining_ops =
    // walk_ops.iter()` and pays no range-index branch in the edge loop. For
    // BOUNDED_ROOT=true the iterator keeps the FULL tail (`walk_ops[start..]`,
    // never `[start..end]`) because the skip helpers recover the absolute op
    // index as `walk_ops.len() - remaining_ops.as_slice().len()`; a truncated
    // slice would corrupt that global-index arithmetic.
    let root_range_end: usize = if BOUNDED_ROOT {
        let (start, end, marker_start) = match root_range {
            Some(range) => range,
            None => return Ok(None),
        };
        debug_assert!(start <= end);
        token_marker_index = marker_start;
        remaining_ops = walk_ops[start as usize..].iter();
        end as usize
    } else {
        0
    };
    let mut lexer: u8;
    let mut parser = root_parser;
    let mut boundary_parser = root_parser;
    let mut boundary_row = parser_cache.physical_boundary_row_ptr(root_parser);
    let mut hot_liveness = [0u8; FULL_WALK_HOT_CAPACITY];
    dense_hot_refresh_liveness(&hot_scalar, boundary_row, &mut hot_liveness);
    let mut two = ((0u32, 0u32), (0u32, 0u32));

    'edge_walk: loop {
        if BOUNDED_ROOT {
            // Stop at the certified range end at each outer edge boundary. The
            // global next-op index is derived from the full tail (see setup),
            // and `end` aligns with a starts_edge op, so no per-byte range
            // check is needed.
            let next_op_index = walk_ops.len() - remaining_ops.as_slice().len();
            if next_op_index >= root_range_end {
                break;
            }
        }
        let Some(&first_op) = remaining_ops.next() else {
            break;
        };
        if !first_op.starts_edge() || !first_op.consumes_byte() {
            return Ok(None);
        }
        let parent_depth = first_op.parent_depth() as usize;
        lexer = unsafe { *stack_lexer.get_unchecked(parent_depth) };
        if lexer < HOT_TWO_DISTINCT {
            parser = unsafe { *stack_parser.get_unchecked(parent_depth) };
            if parser != boundary_parser {
                boundary_parser = parser;
                boundary_row = parser_cache.physical_boundary_row_ptr(parser);
                dense_hot_refresh_liveness(&hot_scalar, boundary_row, &mut hot_liveness);
            }
        } else if lexer == HOT_TWO_DISTINCT || lexer == HOT_TWO {
            two = unsafe { *stack_two.get_unchecked(parent_depth) };
        } else {
            return Ok(None);
        }

        let mut op = first_op;
        loop {
            let byte = op.byte();
            if lexer == HOT_DEAD {
                dense_hot_skip_dead_subtree::<POSITIVE, OBSERVE_DENSITY>(
                    vocab,
                    trie,
                    walk_ops,
                    &mut remaining_ops,
                    &mut token_marker_index,
                    &mut skipped_original_tokens,
                    buf,
                );
                continue 'edge_walk;
            }

            let outcome = if lexer < HOT_TWO_DISTINCT {
                let hot_id = lexer;
                let mut next = unsafe {
                    *hot_scalar
                        .rows
                        .get_unchecked(hot_id as usize)
                        .get_unchecked(byte as usize)
                };
                if next == FULL_WALK_HOT_UNKNOWN {
                    next = dense_hot_transition_miss(&mut hot_scalar, transitions, hot_id, byte);
                }
                if next < FULL_WALK_HOT_SLOW {
                    if !PRUNE_INTERIOR {
                        lexer = next;
                        None
                    } else {
                        let live = unsafe { *hot_liveness.get_unchecked(next as usize) };
                        if live == 2 {
                            lexer = next;
                            None
                        } else {
                            let target =
                                unsafe { *hot_scalar.hot_to_raw.get_unchecked(next as usize) };
                            if live == 0
                                && dense_hot_physical_boundary_miss(
                                    parser_cache,
                                    state.constraint,
                                    tokenizer,
                                    transitions,
                                    parser,
                                    target,
                                )
                            {
                                unsafe { *hot_liveness.get_unchecked_mut(next as usize) = 2 };
                                lexer = next;
                                None
                            } else {
                                unsafe { *hot_liveness.get_unchecked_mut(next as usize) = 1 };
                                Some(DenseHotLaneOutcome::Dead)
                            }
                        }
                    }
                } else if next == FULL_WALK_HOT_DEAD {
                    Some(DenseHotLaneOutcome::Dead)
                } else {
                    let source = unsafe { *hot_scalar.hot_to_raw.get_unchecked(hot_id as usize) };
                    Some(dense_hot_scalar_escape(
                        source,
                        byte,
                        parser,
                        initial_lexer_state,
                        finalizer_code,
                        single_finalizer_continues,
                        tokenizer,
                        transitions,
                        parser_cache,
                        state.constraint,
                        vocab,
                        &mut pair_union_cache,
                    ))
                }
            } else if lexer == HOT_TWO_DISTINCT || lexer == HOT_TWO {
                Some(dense_hot_two_escape(
                    two,
                    byte,
                    initial_lexer_state,
                    finalizer_code,
                    single_finalizer_continues,
                    tokenizer,
                    transitions,
                    parser_cache,
                    state.constraint,
                    vocab,
                    &mut pair_union_cache,
                ))
            } else {
                return Ok(None);
            };

            if let Some(outcome) = outcome {
                match outcome {
                    DenseHotLaneOutcome::Dead => {
                        dense_hot_skip_dead_subtree::<POSITIVE, OBSERVE_DENSITY>(
                            vocab,
                            trie,
                            walk_ops,
                            &mut remaining_ops,
                            &mut token_marker_index,
                            &mut skipped_original_tokens,
                            buf,
                        );
                        continue 'edge_walk;
                    }
                    DenseHotLaneOutcome::Scalar(next_lexer, next_parser) => {
                        let Some(next_hot_id) = hot_scalar.intern(next_lexer) else {
                            return Ok(None);
                        };
                        lexer = next_hot_id;
                        parser = next_parser;
                        if parser != boundary_parser {
                            boundary_parser = parser;
                            boundary_row = parser_cache.physical_boundary_row_ptr(parser);
                            dense_hot_refresh_liveness(
                                &hot_scalar,
                                boundary_row,
                                &mut hot_liveness,
                            );
                        } else {
                            hot_liveness[next_hot_id as usize] =
                                unsafe { *boundary_row.add(next_lexer as usize) };
                        }
                    }
                    DenseHotLaneOutcome::Two(first, second) => {
                        lexer = if first.1 != second.1 {
                            HOT_TWO_DISTINCT
                        } else {
                            HOT_TWO
                        };
                        two = (first, second);
                    }
                    DenseHotLaneOutcome::Decline => return Ok(None),
                }
            }

            if op.ends_edge() {
                if op.child_is_token() {
                    let token_marker = unsafe { *token_markers.get_unchecked(token_marker_index) };
                    token_marker_index += 1;
                    let allowed = if lexer == HOT_DEAD {
                        false
                    } else if lexer < HOT_TWO_DISTINCT {
                        if lexer == initial_hot_id {
                            true
                        } else {
                            let live = unsafe { *hot_liveness.get_unchecked(lexer as usize) };
                            if live == 2 {
                                true
                            } else if live == 1 {
                                false
                            } else {
                                let endpoint_lexer = unsafe {
                                    *hot_scalar.hot_to_raw.get_unchecked(lexer as usize)
                                };
                                let allowed = dense_hot_physical_boundary_miss(
                                    parser_cache,
                                    state.constraint,
                                    tokenizer,
                                    transitions,
                                    parser,
                                    endpoint_lexer,
                                );
                                unsafe {
                                    *hot_liveness.get_unchecked_mut(lexer as usize) =
                                        if allowed { 2 } else { 1 };
                                }
                                allowed
                            }
                        }
                    } else if lexer == HOT_TWO_DISTINCT || lexer == HOT_TWO {
                        parser_cache.token_boundary_allowed_raw(
                            state.constraint,
                            tokenizer,
                            transitions,
                            initial_lexer_state,
                            two.0.0,
                            two.0.1,
                        ) || parser_cache.token_boundary_allowed_raw(
                            state.constraint,
                            tokenizer,
                            transitions,
                            initial_lexer_state,
                            two.1.0,
                            two.1.1,
                        )
                    } else {
                        return Ok(None);
                    };
                    if POSITIVE {
                        if allowed {
                            mark_dynamic_token_marker(vocab, token_marker, buf);
                        }
                    } else if !allowed {
                        clear_dynamic_token_marker(vocab, token_marker, buf);
                    }
                }
                unsafe {
                    *stack_lexer.get_unchecked_mut(parent_depth + 1) = lexer;
                    if lexer < HOT_TWO_DISTINCT {
                        *stack_parser.get_unchecked_mut(parent_depth + 1) = parser;
                    } else if lexer == HOT_TWO_DISTINCT || lexer == HOT_TWO {
                        *stack_two.get_unchecked_mut(parent_depth + 1) = two;
                    } else {
                        return Ok(None);
                    }
                }
                break;
            }
            let Some(&next_op) = remaining_ops.next() else {
                return Ok(None);
            };
            if next_op.starts_edge() || !next_op.consumes_byte() {
                return Ok(None);
            }
            op = next_op;
        }
    }

    if OBSERVE_DENSITY {
        let total_original_tokens = vocab.subtree_original_tokens_for(trie, 0).len();
        // Deliberately conservative: only states that lost at most 5% of the
        // represented vocabulary to exact dead-subtree jumps are classified
        // dense. Endpoint-only parser rejection is not credited as dense.
        let dense = total_original_tokens != 0
            && skipped_original_tokens.saturating_mul(100)
                <= total_original_tokens.saturating_mul(5);
        transitions.cache_dense_output_hint(root_lexer, dense);
    }
    Ok(Some(true))
}

#[inline(never)]
fn try_full_walk_mask_with_table<T: FullWalkTransitionTable, const HOT_SINGLE_ROOT: bool>(
    state: &ConstraintState<'_>,
    vocab: &DynamicMaskVocab,
    trie: &DynamicMaskTrie,
    llg_master_decision: Option<LlgMasterDecision>,
    root_branches: &DynamicBranches,
    lexer_scan_cache: &mut DynamicNfaScanCache<'_>,
    buf: &mut [u32],
    transitions: T,
    finalizer_code: &[u32],
    single_finalizer_continues: &[u8],
) -> Result<bool, String> {
    debug_assert!(trie.full_walk_max_parent_depth() < 255);

    let initial_lexer_state = lexer_scan_cache
        .config_for_raw_start(vocab.mask_runtime_state(state.constraint.tokenizer.initial_state()))?;

    try_full_walk_mask_with_table_from_initial::<T, HOT_SINGLE_ROOT>(
        state,
        vocab,
        trie,
        llg_master_decision,
        root_branches,
        lexer_scan_cache,
        buf,
        transitions,
        finalizer_code,
        single_finalizer_continues,
        initial_lexer_state,
    )
}

#[inline(never)]
fn try_full_walk_mask_with_table_from_initial<
    T: FullWalkTransitionTable,
    const HOT_SINGLE_ROOT: bool,
>(
    state: &ConstraintState<'_>,
    vocab: &DynamicMaskVocab,
    trie: &DynamicMaskTrie,
    llg_master_decision: Option<LlgMasterDecision>,
    root_branches: &DynamicBranches,
    lexer_scan_cache: &mut DynamicNfaScanCache<'_>,
    buf: &mut [u32],
    transitions: T,
    finalizer_code: &[u32],
    single_finalizer_continues: &[u8],
    initial_lexer_state: u32,
) -> Result<bool, String> {
    try_full_walk_mask_with_table_from_initial_in_output_scope::<T, HOT_SINGLE_ROOT>(
        state, vocab, trie, llg_master_decision, root_branches, lexer_scan_cache,
        buf, transitions, finalizer_code, single_finalizer_continues,
        initial_lexer_state, None,
    )
}

#[inline(never)]
fn try_full_walk_mask_with_table_from_initial_in_output_scope<
    T: FullWalkTransitionTable,
    const HOT_SINGLE_ROOT: bool,
>(
    state: &ConstraintState<'_>,
    vocab: &DynamicMaskVocab,
    trie: &DynamicMaskTrie,
    llg_master_decision: Option<LlgMasterDecision>,
    root_branches: &DynamicBranches,
    lexer_scan_cache: &mut DynamicNfaScanCache<'_>,
    buf: &mut [u32],
    transitions: T,
    finalizer_code: &[u32],
    single_finalizer_continues: &[u8],
    initial_lexer_state: u32,
    ignored_output: Option<&[u32]>,
) -> Result<bool, String> {
    debug_assert!(trie.full_walk_max_parent_depth() < 255);

    let profile_walk = dynamic_mask_profile_enabled(state.generation);
    let profile_kernel = std::env::var("GLRMASK_PROFILE_DYNAMIC_KERNEL_GENERATION")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        == Some(state.generation);
    let kernel_started = profile_kernel.then(std::time::Instant::now);

    let (mut parser_cache, root_parser_nodes) = FullWalkParserCache::from_roots(
        root_branches,
        transitions.state_count(lexer_scan_cache.tokenizer()),
        profile_walk,
    );
    if profile_walk {
        for (root_index, &parser_node) in root_parser_nodes.iter().enumerate() {
            let lexer_state = root_branches[root_index].tokenizer_config;
            let tokenizer = lexer_scan_cache.tokenizer();
            if lexer_state < tokenizer.num_states() {
                let matched = tokenizer
                    .matched_terminal_bitset(lexer_state)
                    .iter_ones()
                    .map(|terminal| terminal as TerminalID)
                    .collect::<Vec<_>>();
                let futures = tokenizer
                    .possible_future_terminals(lexer_state)
                    .iter_ones()
                    .map(|terminal| terminal as TerminalID)
                    .collect::<Vec<_>>();
                eprintln!(
                    "[glrmask/profile][root_lexer_observation] generation={} root={} lexer={} matched={:?} futures={:?}",
                    state.generation,
                    root_index,
                    lexer_state,
                    matched,
                    futures,
                );
            } else {
                eprintln!(
                    "[glrmask/profile][root_lexer_observation] generation={} root={} lexer={} subset_extension=true",
                    state.generation,
                    root_index,
                    lexer_state,
                );
            }
            let admitted = parser_cache.admitted(state.constraint, parser_node);
            let ids = admitted
                .iter_ones()
                .map(|terminal| terminal as TerminalID)
                .collect::<Vec<_>>();
            eprintln!(
                "[glrmask/profile][root_admitted_terminals] generation={} root={} count={} ids={:?}",
                state.generation,
                root_index,
                ids.len(),
                ids,
            );
            for &terminal in &ids {
                if let Some(expr) = state.constraint.retained_terminal_expr(terminal) {
                    eprintln!(
                        "[glrmask/profile][root_admitted_terminal_expr] generation={} root={} terminal={} expr={:?}",
                        state.generation,
                        root_index,
                        terminal,
                        expr,
                    );
                }
            }
        }
    }

    // With no pending terminal-exclusion guard, a zero-initialized mask is an
    // exact representation for the common singleton-parser frontier: the walk
    // can set surviving token endpoints instead of starting full and clearing
    // every dead subtree. Pending prune guards are deliberately excluded here
    // because their accumulator correlation is not represented by the plain
    // parser-admission singleton test.
    static SINGLETON_POSITIVE_REBUILD_ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let singleton_positive_rebuild = *SINGLETON_POSITIVE_REBUILD_ENABLED
        .get_or_init(|| std::env::var_os("GLRMASK_DISABLE_SINGLETON_POSITIVE_REBUILD").is_none())
        && state.constraint.ignore_terminal.is_none()
        && root_branches
            .iter()
            .all(|branch| branch.initial_prune_guard.is_passed())
        && {
            let mut only_terminal = None::<TerminalID>;
            let mut singleton = true;
            for &parser_node in &root_parser_nodes {
                let mut admitted = parser_cache
                    .admitted(state.constraint, parser_node)
                    .iter_ones()
                    .map(|terminal| terminal as TerminalID);
                let Some(terminal) = admitted.next() else {
                    singleton = false;
                    break;
                };
                if admitted.next().is_some() {
                    singleton = false;
                    break;
                }
                match only_terminal {
                    None => only_terminal = Some(terminal),
                    Some(existing) if existing == terminal => {}
                    Some(_) => {
                        singleton = false;
                        break;
                    }
                }
            }
            singleton && only_terminal.is_some()
        };
    // The parser row can admit many terminals even when the current lexer
    // residual can produce only one of them. In that common narrow-mask shape,
    // requiring parser admission itself to be singleton needlessly forces the
    // negative-polarity path. Use the exact walk coordinate's sole live
    // terminal instead, then verify that terminal against parser admission.
    // This is purely an output-polarity choice: the semantic walk is unchanged.
    static EFFECTIVE_SINGLETON_POSITIVE_REBUILD_ENABLED: std::sync::OnceLock<bool> =
        std::sync::OnceLock::new();
    let effective_singleton_positive_rebuild = *EFFECTIVE_SINGLETON_POSITIVE_REBUILD_ENABLED
        .get_or_init(|| {
            std::env::var_os("GLRMASK_DISABLE_EFFECTIVE_SINGLETON_POSITIVE_REBUILD").is_none()
        })
        && root_branches
            .iter()
            .all(|branch| branch.initial_prune_guard.is_passed())
        && {
            let tokenizer = lexer_scan_cache.tokenizer();
            let mut only_terminal = None::<TerminalID>;
            let mut singleton = true;
            for (root_index, root) in root_branches.iter().enumerate() {
                let lexer_state = root.tokenizer_config;
                let admitted = parser_cache.admitted(state.constraint, root_parser_nodes[root_index]);
                let Some(root_terminal) = transitions.sole_live_terminal(tokenizer, lexer_state)
                else {
                    singleton = false;
                    break;
                };
                if Some(root_terminal) != state.constraint.ignore_terminal
                    && !admitted.contains(root_terminal as usize)
                {
                    singleton = false;
                    break;
                }
                match only_terminal {
                    None => only_terminal = Some(root_terminal),
                    Some(existing) if existing == root_terminal => {}
                    Some(_) => {
                        singleton = false;
                        break;
                    }
                }
            }
            singleton && only_terminal.is_some()
        };
    // The pre-collapse Flat16 path can hand us a master certificate directly.
    // Other deterministic one-root paths still have the same exact lexer/parser
    // information. Before running an exact proof, filter admitted terminals by
    // the necessary byte-support condition for the safe+ language. This keeps
    // the cold proof off ordinary narrow terminals even when a parser state has
    // many alternatives, while broad string terminals remain eligible.
    let mut llg_master_decision = llg_master_decision;
    static GENERIC_MASTER_PROOF_ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let generic_master_proof_enabled = *GENERIC_MASTER_PROOF_ENABLED.get_or_init(|| {
        std::env::var_os("GLRMASK_DISABLE_GENERIC_MASTER_PROOF").is_none()
    });
    let generic_master_candidates = if generic_master_proof_enabled
        && llg_master_decision.is_none()
        && HOT_SINGLE_ROOT
        && root_branches.len() == 1
        && root_branches[0].initial_prune_guard.is_passed()
        && vocab.llg_master_trie().is_some()
    {
        vocab
            .llg_slice_by_cache_id(LLG_SAFE_PLUS_SLICE as u32)
            .and_then(|safe_plus| {
                let admitted = parser_cache.admitted(state.constraint, root_parser_nodes[0]);
                let mut candidates = SmallVec::<[TerminalID; 4]>::new();
                let lexer_state = root_branches[0].tokenizer_config;
                let filter_first_bytes =
                    std::env::var_os("GLRMASK_DISABLE_GENERIC_MASTER_FIRST_BYTES").is_none();
                let first_bytes = filter_first_bytes.then(|| safe_plus.first_bytes());
                let exact_source = root_branches[0].exact_tokenizer_state;
                for terminal in admitted.iter_ones().map(|terminal| terminal as TerminalID) {
                    let lexer_live = transitions.future_contains(
                        lexer_scan_cache.tokenizer(),
                        lexer_state,
                        terminal,
                    ) || transitions
                        .matched_terminals(lexer_scan_cache.tokenizer(), lexer_state)
                        .contains(&terminal);
                    if lexer_live
                        && state
                            .constraint
                            .tokenizer
                            .terminal_byte_support(terminal)
                            .is_some_and(|support| safe_plus.slice_token_bytes().is_subset(&support))
                        // A wide parser row need not imply a wide proof search.
                        // Use the exact source's cheap first-byte necessary
                        // condition BEFORE applying the candidate-count limit.
                        // Unknown/virtual sources remain conservative candidates.
                        && !first_bytes.zip(exact_source).is_some_and(|(bytes, source)| {
                            state.constraint.tokenizer.physical_terminal_residual_covers_first_bytes(
                                source, terminal, bytes,
                            ) == Some(false)
                        })
                    {
                        candidates.push(terminal);
                    }
                }
                // The old <=2 gate protects the expensive wholly-online proof
                // path. Once this source has prepared master-prover rows, the
                // expensive batch work has already been done and the remaining
                // exact direct/quotient fallbacks are the validated wide path.
                // Keep the conservative limit for sources with no prepared row.
                let prepared_wide = root_branches[0]
                    .exact_tokenizer_state
                    .is_some_and(|source| vocab.has_prepared_master_prover_row(source));
                let candidate_limit = if prepared_wide { 8 } else { 2 };
                if std::env::var_os("GLRMASK_DIAG_CANDIDATE_GATES").is_some() {
                    eprintln!("[candidate_gate] generation={} lexer={} exact={:?} candidates={} limit={} first_bytes_gate={}",
                        state.generation, lexer_state, exact_source, candidates.len(), candidate_limit, filter_first_bytes);
                }
                (!candidates.is_empty() && candidates.len() <= candidate_limit)
                    .then_some(candidates)
            })
    } else {
        None
    };
    if let Some(safe_plus_candidates) = generic_master_candidates {
        let parser_node = root_parser_nodes[0];
        let lexer_state = root_branches[0].tokenizer_config;
        let exact_source = root_branches[0].exact_tokenizer_state;
        let admitted = parser_cache.admitted(state.constraint, parser_node).clone();
        let safe_plus = vocab
            .llg_slice_by_cache_id(LLG_SAFE_PLUS_SLICE as u32)
            .expect("dynamic-radius master requires safe+ proof DFA");
        let whitespace = vocab
            .llg_slice_by_cache_id(LLG_WHITESPACE_SLICE as u32)
            .expect("dynamic-radius master requires whitespace proof DFA");
        let whitespace_candidates = admitted
            .iter_ones()
            .map(|terminal| terminal as TerminalID)
            .filter(|&terminal| {
                state
                    .constraint
                    .tokenizer
                    .terminal_byte_support(terminal)
                    .is_some_and(|support| whitespace.slice_token_bytes().is_subset(&support))
            })
            .collect::<SmallVec<[TerminalID; 8]>>();

        let prove_slice = |
            slice: &crate::runtime::artifact::DynamicMaskSliceTrie,
            terminals: &[TerminalID],
        | -> bool {
            terminals.iter().copied().any(|terminal| {
                if let Some(source) = exact_source {
                    let prepared_slot = if slice.cache_id() == LLG_SAFE_PLUS_SLICE as u32 {
                        Some(0usize)
                    } else if slice.cache_id() == LLG_WHITESPACE_SLICE as u32 {
                        Some(1usize)
                    } else {
                        None
                    };
                    if let Some(prepared) = prepared_slot.and_then(|slot| {
                        vocab.prepared_master_proof_result(source, slot, terminal)
                    }) {
                        return prepared;
                    }
                }
                let live = transitions.future_contains(
                    lexer_scan_cache.tokenizer(),
                    lexer_state,
                    terminal,
                ) || transitions
                    .matched_terminals(lexer_scan_cache.tokenizer(), lexer_state)
                    .contains(&terminal);
                if !live {
                    return false;
                }
                if let Some(cached) = vocab.cached_direct_slice_contained(
                    terminal,
                    lexer_state,
                    slice.cache_id(),
                ) {
                    return cached;
                }
                // The active transition table is already an exact mask-runtime
                // coordinate. For one-root states, try the source-local
                // containment proof before materializing any terminal quotient.
                // A positive result is definitive; a negative/unknown result
                // merely falls through to the existing exact quotient and
                // symbolic proofs.
                if direct_slice_prefix_contained(
                    &transitions,
                    lexer_scan_cache.tokenizer(),
                    lexer_state,
                    terminal,
                    slice.dfa(),
                    32 * 1024,
                ) == Some(true)
                {
                    vocab.cache_direct_slice_contained(
                        terminal,
                        lexer_state,
                        slice.cache_id(),
                        true,
                    );
                    return true;
                }
                if let Some(quotient) = exact_source.and_then(|source| {
                    vocab.projected_terminal_slice_contained(
                        terminal,
                        source,
                        slice.cache_id(),
                        slice.dfa(),
                    )
                }) {
                    vocab.cache_direct_slice_contained(
                        terminal,
                        lexer_state,
                        slice.cache_id(),
                        quotient,
                    );
                    return quotient;
                }
                let symbolic = exact_source.and_then(|source| {
                    virtual_residual_slice_prefix_contained(
                        &state.constraint.tokenizer,
                        source,
                        terminal,
                        slice.dfa(),
                        1_536,
                    )
                });
                let proved = symbolic == Some(true);
                vocab.cache_direct_slice_contained(
                    terminal,
                    lexer_state,
                    slice.cache_id(),
                    proved,
                );
                proved
            })
        };

        let compute_safe_radius = || -> Option<u16> {
            let source = exact_source?;
            let max_vocab_safe_chars = u32::from(vocab.llg_master_max_safe_chars());
            Some(
                safe_plus_candidates
                    .iter()
                    .copied()
                    .filter_map(|terminal| {
                        let projected = vocab
                            .prepared_safe_radius(source, terminal)
                            .map(u32::from)
                            .or_else(|| {
                                vocab.projected_terminal_slice_repeat_radius(
                                    terminal,
                                    source,
                                    safe_plus.cache_id(),
                                    safe_plus.dfa(),
                                    max_vocab_safe_chars,
                                    16 * 1024,
                                )
                            });
                        let symbolic = virtual_residual_safe_repeat_radius(
                            &state.constraint.tokenizer,
                            source,
                            terminal,
                            safe_plus.dfa(),
                            max_vocab_safe_chars,
                            16 * 1024,
                        );
                        match (projected, symbolic) {
                            (Some(left), Some(right)) => Some(left.max(right)),
                            (left, right) => left.or(right),
                        }
                    })
                    .filter_map(|radius| u16::try_from(radius).ok())
                    .max()
                    .unwrap_or(0),
            )
        };
        let prefer_bounded_radius = std::env::var_os("GLRMASK_DISABLE_VIRTUAL_RADIUS_FIRST").is_none()
            && exact_source.is_some_and(|source| {
            safe_plus_candidates.iter().copied().any(|terminal| {
                state
                    .constraint
                    .tokenizer
                    .singleton_epsilon_closure(source)
                    .into_iter()
                    .any(|residual_state| {
                        state
                            .constraint
                            .tokenizer
                            .virtual_residual_terminal_for_state(residual_state)
                            == Some(terminal)
                    })
            })
        });
        let radius_before_unbounded = prefer_bounded_radius.then(compute_safe_radius).flatten();
        // For vocabulary masking an exact positive bounded radius is already a
        // complete certificate for every precomputed safe-slice token within
        // that radius. Do not first pay to prove the strictly stronger
        // unbounded safe+ language when the bounded certificate is available.
        let safe_plus_proved = if radius_before_unbounded.is_some_and(|radius| radius != 0) {
            false
        } else {
            prove_slice(safe_plus, safe_plus_candidates.as_slice())
        };
        let safe_radius = if safe_plus_proved {
            u16::MAX
        } else {
            radius_before_unbounded
                .or_else(compute_safe_radius)
                .unwrap_or(0)
        };
        let whitespace_proved = prove_slice(whitespace, whitespace_candidates.as_slice());
        let decision = LlgMasterDecision {
            safe_radius,
            whitespace: whitespace_proved,
        };
        if !decision.is_empty() {
            llg_master_decision = Some(decision);
        }
    }
    // Certify an upper bound separately from the positive safe radius. A
    // successful lower-radius proof alone NEVER licenses rejecting long tokens.
    // Every source in this one exact root's epsilon closure must have a finite
    // body envelope that cannot finish its terminal on a safe-string atom.
    static SAFE_ENVELOPE_UPPER_ENABLED: OnceLock<bool> = OnceLock::new();
    let mut safe_envelope_upper = if *SAFE_ENVELOPE_UPPER_ENABLED.get_or_init(|| {
        std::env::var_os("GLRMASK_DISABLE_SAFE_ENVELOPE_UPPER").is_none()
    }) && HOT_SINGLE_ROOT && root_branches.len() == 1
        && root_branches[0].initial_prune_guard.is_passed()
        && vocab.llg_master_trie().is_some()
        && !vocab.is_grammar_quotiented()
    {
        (|| -> Option<u16> {
            let source = root_branches[0].exact_tokenizer_state?;
            let tok = &state.constraint.tokenizer;
            let slice = vocab.llg_slice_by_cache_id(LLG_SAFE_PLUS_SLICE as u32)?;
            let dfa = slice.dfa();
            let mut upper = None::<u32>;
            for raw in tok.singleton_epsilon_closure(source).iter().copied() {
                // Pure structural epsilon nodes add no consuming behavior.
                if tok.state_has_epsilon_transitions(raw)
                    && tok.transitions_from(raw).next().is_none()
                    && tok.matched_terminal_bitset(raw).is_empty()
                {
                    continue;
                }
                let bound = tok.virtual_residual_safe_atom_length_upper_bound(
                    raw,
                    dfa.start_state(),
                    dfa.class_count(),
                    dfa.byte_to_class_map(),
                    dfa.transition_table(),
                    dfa.accepting_map(),
                    dfa.can_reach_accepting_map(),
                )?;
                upper = Some(upper.map_or(bound, |old| old.max(bound)));
            }
            let upper = u16::try_from(upper?).ok()?;
            // No vocabulary token can exceed the maximum observed safe length.
            if upper >= vocab.llg_master_max_safe_chars() {
                return None;
            }
            if let Some(lower) = llg_master_decision {
                if lower.whitespace || lower.safe_radius > upper {
                    return None;
                }
            }
            Some(upper)
        })()
    } else {
        None
    };
    if safe_envelope_upper.is_some() && llg_master_decision.is_none() {
        llg_master_decision = Some(LlgMasterDecision::default());
    }
    let safe_interval_work = |decision: LlgMasterDecision| -> Option<usize> {
        let base = vocab.llg_master_residual_ops(decision.safe_radius, decision.whitespace)?;
        if let Some(upper) = safe_envelope_upper {
            let above_upper = vocab.llg_master_residual_ops(upper, decision.whitespace)?;
            let all_safe_skipped = vocab.llg_master_residual_ops(u16::MAX, decision.whitespace)?;
            Some(base.checked_sub(above_upper)?.checked_add(all_safe_skipped)?)
        } else {
            Some(base)
        }
    };
    if std::env::var_os("GLRMASK_DIAG_SAFE_ENVELOPE_UPPER").is_some() {
        eprintln!(
            "[safe_envelope_upper] generation={} lower={:?} upper={:?} work={:?}",
            state.generation,
            llg_master_decision.map(|v| v.safe_radius),
            safe_envelope_upper,
            llg_master_decision.and_then(safe_interval_work)
        );
    }

    // Exact master proofs are not automatically profitable. Selecting the
    // partitioned master trie also abandons the optimized ordinary-trie hot
    // lane; for small bounded radii the residual master walk can be comparable
    // to, or even larger than, the ordinary vocabulary walk. Gate the route on
    // exact precomputed strict-walk volume without changing proof semantics.
    if let Some(decision) = llg_master_decision {
        let max_permille = std::env::var("GLRMASK_EXPERIMENT_CONFIG_MASTER_MAX_RESIDUAL_PERMILLE")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(1000);
        let ordinary_ops = trie.full_walk_ops().len();
        let profitable = safe_interval_work(decision)
            .is_some_and(|residual_ops| {
                residual_ops.saturating_mul(1000)
                    <= ordinary_ops.saturating_mul(max_permille)
            });
        if std::env::var_os("GLRMASK_DIAG_CANDIDATE_GATES").is_some() {
            eprintln!("[candidate_gate_master] generation={} radius={} whitespace={} ordinary={} residual={:?} profitable={}",
                state.generation, decision.safe_radius, decision.whitespace, ordinary_ops,
                vocab.llg_master_residual_ops(decision.safe_radius, decision.whitespace), profitable);
        }
        if !profitable {
            llg_master_decision = None;
        }
    }
    if llg_master_decision.is_none() {
        safe_envelope_upper = None;
    }
    let trie = llg_master_decision
        .and_then(|_| vocab.llg_master_trie())
        .map_or(trie, |slice| slice.trie());
    debug_assert!(trie.full_walk_max_parent_depth() < 255);

    // Choose output polarity only after every master/slice proof above has
    // completed. A late master decision changes both the trie being walked and
    // the set of tokens represented by skipped subtrees, so initializing a
    // sparse zero mask before that decision would omit the newly certified
    // admitted slice. The three modes below are exact representations of the
    // same walk result:
    //   * master: seed the already-proved admitted slice and add residual hits;
    //   * singleton: seed zero and add the few surviving endpoints;
    //   * ordinary: seed all tokens and clear rejected subtrees/endpoints.
    // For ordinary passed-guard walks the accepted and rejected output sets
    // are both exact representations of the same logical result.  The legacy
    // path chose one polarity before the walk, which can force O(vocab)
    // mutations for a sparse result (or the reverse for a dense result).
    // Defer ordinary materialization and record compact trie events so the
    // cheaper exact side can be selected after the logical walk.  This is an
    // output-representation choice only: the parser/lexer walk and acceptance
    // decisions are identical for both polarities.
    static FORCE_POSITIVE_REBUILD: OnceLock<bool> = OnceLock::new();
    let force_positive_rebuild = *FORCE_POSITIVE_REBUILD
        .get_or_init(|| std::env::var_os("GLRMASK_EXPERIMENT_FORCE_POSITIVE_REBUILD").is_some())
        && llg_master_decision.is_none()
        && state.constraint.ignore_terminal.is_none()
        && root_branches
            .iter()
            .all(|branch| branch.initial_prune_guard.is_passed())
        && vocab.residual_original_token_words_for(trie).is_none();
    // Grammar-quotiented O2 masks can be extremely dense even when the parser
    // admits only one terminal: one quotient terminal may represent nearly the
    // whole model vocabulary.  Do not let parser singleton-ness force a sparse
    // positive rebuild in that case.  Record the exact walk result and choose
    // the cheaper output polarity after the walk instead.  O1 retains the
    // existing singleton-positive policy.
    let quotient_adaptive_polarity = vocab.is_grammar_quotiented();
    let mut deferred_output = !force_positive_rebuild
        && llg_master_decision.is_none()
        && (quotient_adaptive_polarity
            || (!singleton_positive_rebuild && !effective_singleton_positive_rebuild))
        && state.constraint.ignore_terminal.is_none()
        && root_branches
            .iter()
            .all(|branch| branch.initial_prune_guard.is_passed())
        && vocab.residual_original_token_words_for(trie).is_none();

    let mut positive_rebuild = if let Some(decision) = llg_master_decision {
        let admitted_words = vocab
            .llg_master_admitted_words(decision.safe_radius, decision.whitespace)
            .ok_or_else(|| "dynamic-radius LLG master trie is missing admitted-token words".to_owned())?;
        let copy_len = buf.len().min(admitted_words.len());
        buf[..copy_len].copy_from_slice(&admitted_words[..copy_len]);
        if copy_len < buf.len() {
            buf[copy_len..].fill(0);
        }
        true
    } else if !deferred_output
        && (singleton_positive_rebuild
            || effective_singleton_positive_rebuild
            || force_positive_rebuild)
    {
        buf.fill(0);
        true
    } else if deferred_output {
        // No output mutation occurs until the walk has counted both exact
        // materialization sides.  Initialize defensively; the chosen side is
        // written in full before return.
        buf.fill(0);
        false
    } else {
        let all_words = vocab.all_original_token_words();
        let copy_len = buf.len().min(all_words.len());
        buf[..copy_len].copy_from_slice(&all_words[..copy_len]);
        if copy_len < buf.len() {
            buf[copy_len..].fill(0);
        }
        if let Some(residual_words) = vocab.residual_original_token_words_for(trie) {
            for (target, &residual) in buf.iter_mut().zip(residual_words) {
                *target &= !residual;
            }
            true
        } else {
            false
        }
    };

    // Scalar is overwhelmingly dominant. Encode dead/multi directly in the
    // lexer-state coordinate so the common DFS path needs no separate kind
    // load/store. Full-walk lexer states are bounded far below these u32
    // sentinels by the dense-transition memory budget.
    const FULL_WALK_LEXER_TWO_DISTINCT: u32 = u32::MAX - 4;
    const FULL_WALK_LEXER_TWO: u32 = u32::MAX - 3;
    const FULL_WALK_LEXER_GUARDED_PAIR: u32 = u32::MAX - 2;
    const FULL_WALK_LEXER_MULTI: u32 = u32::MAX - 1;
    const FULL_WALK_LEXER_DEAD: u32 = u32::MAX;

    // Change D: hoist the exact single-live-root-byte scheduler above the
    // hot-lane choice. The same exact proof (a physical root lexer coordinate
    // with exactly one outgoing byte) now both suppresses the hot lane when its
    // root range cannot amortize the 256-byte root probe, and drives the generic
    // walk without recomputation. Opaque / lazy-union / virtual roots stay None
    // and keep their previous hot-lane selection.
    static SINGLE_BYTE_ROOT_WALK_ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let sparse_root_walk_enabled = *SINGLE_BYTE_ROOT_WALK_ENABLED.get_or_init(|| {
        std::env::var_os("GLRMASK_DISABLE_SINGLE_BYTE_ROOT_WALK").is_none()
    });
    let hoisted_root_coord = (root_branches.len() == 1
        && root_branches[0].initial_prune_guard.is_passed())
        .then(|| root_branches[0].tokenizer_config);
    let sparse_root_range = if sparse_root_walk_enabled
        && positive_rebuild
        && !deferred_output
        && llg_master_decision.is_none()
        && let Some(root_lexer) = hoisted_root_coord
        && root_lexer < FULL_WALK_LEXER_TWO_DISTINCT
        && root_lexer < lexer_scan_cache.tokenizer().num_states()
        && trie.node(0).token_id.is_none()
        && trie.has_full_walk_root_byte_index()
    {
        let mut transitions = lexer_scan_cache.tokenizer().transitions_from(root_lexer);
        let first = transitions.next();
        match (first, transitions.next()) {
            (Some((byte, _)), None) => trie.full_walk_root_byte_range(byte),
            _ => None,
        }
    } else {
        None
    };
    // Exact work of the scheduled root range (end - start): the op count the
    // generic scheduler iterates for the single live root byte. It is NOT
    // `remaining_ops.len()`, which also includes later sibling ranges.
    let sparse_root_range_span = sparse_root_range
        .map(|(start, end, _)| end.saturating_sub(start) as u64);
    static DENSE_HOT_MIN_ROOT_OPS: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let dense_hot_min_root_ops = *DENSE_HOT_MIN_ROOT_OPS.get_or_init(|| {
        std::env::var("GLRMASK_EXPERIMENT_DENSE_HOT_MIN_ROOT_OPS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(4096)
    });
    // Legacy scheduling (0) and every non-scheduled root keep the hot lane; only
    // a proved narrow root range hands the walk to the generic scheduler.
    let dense_hot_root_profitable = sparse_root_range_span
        .map_or(true, |span| dense_hot_min_root_ops == 0 || span >= dense_hot_min_root_ops);

    let dense_hot_negative_proof =
        std::env::var_os("GLRMASK_EXPERIMENT_DENSE_HOT_NEGATIVE").is_some()
            && std::env::var_os("GLRMASK_DISABLE_DENSE_HOT_LANE").is_none()
            && !profile_walk
            && !profile_kernel
            && HOT_SINGLE_ROOT
            && root_branches.len() == 1
            && root_branches[0].initial_prune_guard.is_passed()
            && llg_master_decision.is_none()
            && positive_rebuild
            && !deferred_output;
    if dense_hot_negative_proof {
        let all_words = vocab.all_original_token_words();
        let copy_len = buf.len().min(all_words.len());
        buf[..copy_len].copy_from_slice(&all_words[..copy_len]);
        if copy_len < buf.len() {
            buf[copy_len..].fill(0);
        }
        match try_dense_hot_scalar_edges::<_, false, true, false, false>(
            state,
            vocab,
            trie,
            root_branches[0].tokenizer_config,
            root_parser_nodes[0],
            initial_lexer_state,
            finalizer_code,
            single_finalizer_continues,
            lexer_scan_cache.tokenizer(),
            &transitions,
            &mut parser_cache,
            FullWalkHotScalarCache::new(),
            None,
            buf,
        )? {
            Some(true) => {
                update_special_token_mask(state, buf);
                state.clear_late_grammar_placeholder_mask(buf);
                return Ok(true);
            }
            Some(false) | None => {
                // Proof lane is allowed only where the normal lane is a
                // positive rebuild; restore that exact baseline before retry.
                buf.fill(0);
            }
        }
    }

    static PARSER_NARROW_HOT: OnceLock<bool> = OnceLock::new();
    let parser_narrow_root = *PARSER_NARROW_HOT.get_or_init(|| {
        std::env::var_os("GLRMASK_DISABLE_PARSER_NARROW_HOT").is_none()
    })
        && root_branches.len() == 1
        && root_branches[0].initial_prune_guard.is_passed()
        && root_branches[0].exact_tokenizer_state
            == Some(root_branches[0].tokenizer_config)
        && std::ptr::eq(lexer_scan_cache.tokenizer(), state.constraint.tokenizer.as_ref())
        && parser_cache.nodes[root_parser_nodes[0] as usize]
            .admitted_singleton.is_some_and(|terminal| {
                let tokenizer = lexer_scan_cache.tokenizer();
                let source = root_branches[0].tokenizer_config;
                source < tokenizer.num_states()
                    && !tokenizer.state_is_virtual_runtime(source)
                    && !tokenizer.state_has_epsilon_transitions(source)
                    && tokenizer.possible_future_terminals_iter(source)
                        .any(|candidate| candidate != terminal)
            });

    let dense_hot_lane_would = std::env::var_os("GLRMASK_DISABLE_DENSE_HOT_LANE").is_none()
        && !profile_walk
        && !profile_kernel
        && HOT_SINGLE_ROOT
        && root_branches.len() == 1
        && root_branches[0].initial_prune_guard.is_passed()
        && llg_master_decision.is_none()
        && positive_rebuild
        && !deferred_output;
    // Change D profitability gate: same exact eligibility, narrowed by the exact
    // root-range op span. `GLRMASK_DISABLE_DENSE_HOT_LANE` stays authoritative.
    let dense_hot_lane_eligible = dense_hot_lane_would && dense_hot_root_profitable;
    // (dense-hot selection diagnostic relocated below, after `e_active` is known)

    // Experiment E: exact single-root-range traversal INSIDE the dense hot
    // executor, with the root probe restricted to the certified live byte.
    // E is gated on `dense_hot_lane_would` (NOT Change D's span profitability
    // gate) plus a certified single-live-root-byte range. Opt-in only.
    static DENSE_HOT_ROOT_RANGE_ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let dense_hot_root_range_enabled = *DENSE_HOT_ROOT_RANGE_ENABLED.get_or_init(|| {
        std::env::var_os("GLRMASK_EXPERIMENT_DENSE_HOT_ROOT_RANGE").is_some()
    });
    // TODO(root-range-certificate): a conservative `state_has_epsilon_transitions`
    // exclusion of the certified root row was intentionally left OUT pending the
    // parent's exact executed-row proof (Flat16/Flat32 rows copy transitions_from
    // directly; epsilon-only states carry no byte transitions). Re-add here as a
    // single isolated `&& !...epsilon...` term if the proof requires it.
    let dense_hot_lane_eligible_e = dense_hot_root_range_enabled
        && dense_hot_lane_would
        && sparse_root_range.is_some();
    // E's certified live byte is read from the range's first op, which must be a
    // starts_edge / consumes_byte op at parent_depth 0 (the root-byte index is
    // built from non-empty root edges only). Decline E to the legacy path if the
    // invariant is not met.
    let e_certified_byte: Option<u8> = if dense_hot_lane_eligible_e {
        let (start, _, _) = sparse_root_range.expect("E eligibility implies a certified range");
        let first_op = trie.full_walk_ops()[start as usize];
        if first_op.starts_edge() && first_op.consumes_byte() && first_op.parent_depth() == 0 {
            Some(first_op.byte())
        } else {
            None
        }
    } else {
        None
    };
    let e_active = e_certified_byte.is_some();

    // Correctness-only certification assert (off for timing; does not change
    // executor selection). Opt-in via GLRMASK_ASSERT_DENSE_HOT_ROOT_RANGE. This
    // runtime check is NOT a replacement for the reviewer's all-T certificate.
    static DENSE_HOT_ROOT_RANGE_ASSERT_ENABLED: std::sync::OnceLock<bool> =
        std::sync::OnceLock::new();
    let dense_hot_root_range_assert = *DENSE_HOT_ROOT_RANGE_ASSERT_ENABLED.get_or_init(|| {
        std::env::var_os("GLRMASK_ASSERT_DENSE_HOT_ROOT_RANGE").is_some()
    });
    static DENSE_HOT_ROOT_RANGE_ASSERT_GENERATION: std::sync::OnceLock<Option<u64>> =
        std::sync::OnceLock::new();
    let dense_hot_root_range_assert_generation =
        *DENSE_HOT_ROOT_RANGE_ASSERT_GENERATION.get_or_init(|| {
            std::env::var("GLRMASK_ASSERT_DENSE_HOT_ROOT_RANGE_GENERATION")
                .ok()
                .and_then(|value| value.parse::<u64>().ok())
        });
    static ROOT_RANGE_ASSERT_PRINTED: std::sync::atomic::AtomicBool =
        std::sync::atomic::AtomicBool::new(false);
    if e_active && dense_hot_root_range_assert {
        let certified = e_certified_byte.expect("e_active implies a certified byte");
        let root_lexer = root_branches[0].tokenizer_config;
        // Every non-certified byte must be lexically dead at this root lexer
        // (exact executed T-cell check over all 256 bytes).
        for raw in 0u16..=255 {
            let b = raw as u8;
            if b == certified {
                continue;
            }
            let cell = transitions.cell(root_lexer, b);
            assert!(
                T::cell_is_dead(cell),
                "root-range E: non-certified byte {} not dead at root lexer {}",
                b,
                root_lexer
            );
        }
        // Range-bound invariants.
        let (start, end, marker_start) = sparse_root_range.expect("E requires a range");
        let walk_ops = trie.full_walk_ops();
        let token_markers = vocab.full_walk_token_markers_for(trie);
        let start = start as usize;
        let end = end as usize;
        assert!(start < end && end <= walk_ops.len(), "root-range E: bad bounds ({start},{end})");
        let first_op = walk_ops[start];
        assert!(
            first_op.starts_edge() && first_op.consumes_byte() && first_op.parent_depth() == 0,
            "root-range E: bad start op"
        );
        if end < walk_ops.len() {
            assert!(
                walk_ops[end].starts_edge() && walk_ops[end].parent_depth() == 0,
                "root-range E: bad end boundary"
            );
        }
        assert!(marker_start <= token_markers.len(), "root-range E: marker start out of range");
        let should_print = dense_hot_root_range_assert_generation == Some(state.generation)
            || !ROOT_RANGE_ASSERT_PRINTED.swap(true, std::sync::atomic::Ordering::Relaxed);
        if should_print {
            eprintln!(
                "[glrmask/root_range_assert] generation={} root_lexer={} t={} range=({},{},{}) certified={} walk_ops={}",
                state.generation, root_lexer, std::any::type_name::<T>(), start, end, marker_start, certified, walk_ops.len()
            );
        }
    }

    // Extended selection diagnostic (after `e_active` is known). The bounded-vs-
    // negative choice is printed inside the executor block once the probe result
    // is available.
    static DENSE_HOT_SELECTION_GENERATION: std::sync::OnceLock<Option<u64>> =
        std::sync::OnceLock::new();
    let profile_selection = *DENSE_HOT_SELECTION_GENERATION.get_or_init(|| {
        std::env::var("GLRMASK_PROFILE_DENSE_HOT_SELECTION_GENERATION")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
    }) == Some(state.generation);
    if profile_selection {
        eprintln!(
            "[glrmask/profile][dense_hot_selection] generation={} root_lexer={:?} range_span={:?} threshold={} would_hot={} selected_hot={} e_active={}",
            state.generation,
            hoisted_root_coord,
            sparse_root_range_span,
            dense_hot_min_root_ops,
            dense_hot_lane_would,
            dense_hot_lane_eligible,
            e_active,
        );
    }
    if dense_hot_lane_eligible || e_active {
        let root_lexer = root_branches[0].tokenizer_config;
        let root_parser = root_parser_nodes[0];
        let mut hot_scalar = FullWalkHotScalarCache::new();
        // Preserve the dense walker's original hot-ID ordering: the initial
        // lexer state remains ID 0 even though the root probe now seeds root
        // transitions before traversal.
        if hot_scalar.intern(initial_lexer_state).is_none() {
            return Ok(false);
        }
        let mut probe = dense_hot_root_probe(
            state,
            vocab,
            trie,
            root_lexer,
            root_parser,
            lexer_scan_cache.tokenizer(),
            &transitions,
            &mut parser_cache,
            &mut hot_scalar,
            e_certified_byte,
        );
        if parser_narrow_root {
            // A singleton parser frontier with broader lexical futures favors
            // the existing exact interior-pruning executor. No acceptance
            // condition or transition is changed by this routing decision.
            probe.prefer_pruning = true;
        }
        let learned_dense = if probe.prefer_pruning {
            None
        } else {
            transitions.dense_output_hint(root_lexer)
        };
        if profile_selection {
            let selection = if e_active {
                if probe.prefer_pruning {
                    "bounded_prune"
                } else if learned_dense == Some(true) {
                    "negative_full"
                } else {
                    "bounded_positive"
                }
            } else if probe.prefer_pruning {
                "prune"
            } else if learned_dense == Some(true) {
                "negative_full"
            } else if learned_dense == Some(false) {
                "positive_sparse"
            } else {
                "positive_observe"
            };
            eprintln!(
                "[glrmask/profile][dense_hot_selection_choice] generation={} e_active={} selection={}",
                state.generation, e_active, selection
            );
        }

        // The negative executor (learned dense) needs the all-admitted baseline.
        if !probe.prefer_pruning && learned_dense == Some(true) {
            let all_words = vocab.all_original_token_words();
            let copy_len = buf.len().min(all_words.len());
            buf[..copy_len].copy_from_slice(&all_words[..copy_len]);
            if copy_len < buf.len() {
                buf[copy_len..].fill(0);
            }
        }
        let mut hot_scalar = Some(hot_scalar);
        let dense_result = if e_active {
            // E dispatch: bounded POSITIVE traversal over the certified root
            // byte's op range. Dense masks keep the full-vocabulary NEGATIVE
            // executor: a bounded positive path must not leave out-of-range
            // initially admitted bits set, and a partial walk must not
            // reinterpret a global dense hint as range density. E never caches
            // a partial density (OBSERVE_DENSITY=false throughout).
            if probe.prefer_pruning {
                try_dense_hot_scalar_edges::<_, true, true, false, true>(
                    state,
                    vocab,
                    trie,
                    root_lexer,
                    root_parser,
                    initial_lexer_state,
                    finalizer_code,
                    single_finalizer_continues,
                    lexer_scan_cache.tokenizer(),
                    &transitions,
                    &mut parser_cache,
                    hot_scalar.take().unwrap(),
                    sparse_root_range,
                    buf,
                )?
            } else if learned_dense == Some(true) {
                try_dense_hot_scalar_edges::<_, false, false, false, false>(
                    state,
                    vocab,
                    trie,
                    root_lexer,
                    root_parser,
                    initial_lexer_state,
                    finalizer_code,
                    single_finalizer_continues,
                    lexer_scan_cache.tokenizer(),
                    &transitions,
                    &mut parser_cache,
                    hot_scalar.take().unwrap(),
                    None,
                    buf,
                )?
            } else {
                try_dense_hot_scalar_edges::<_, true, false, false, true>(
                    state,
                    vocab,
                    trie,
                    root_lexer,
                    root_parser,
                    initial_lexer_state,
                    finalizer_code,
                    single_finalizer_continues,
                    lexer_scan_cache.tokenizer(),
                    &transitions,
                    &mut parser_cache,
                    hot_scalar.take().unwrap(),
                    sparse_root_range,
                    buf,
                )?
            }
        } else if probe.prefer_pruning {
            try_dense_hot_scalar_edges::<_, true, true, false, false>(
                state,
                vocab,
                trie,
                root_lexer,
                root_parser,
                initial_lexer_state,
                finalizer_code,
                single_finalizer_continues,
                lexer_scan_cache.tokenizer(),
                &transitions,
                &mut parser_cache,
                hot_scalar.take().unwrap(),
                None,
                buf,
            )?
        } else if learned_dense == Some(true) {
            try_dense_hot_scalar_edges::<_, false, false, false, false>(
                state,
                vocab,
                trie,
                root_lexer,
                root_parser,
                initial_lexer_state,
                finalizer_code,
                single_finalizer_continues,
                lexer_scan_cache.tokenizer(),
                &transitions,
                &mut parser_cache,
                hot_scalar.take().unwrap(),
                None,
                buf,
            )?
        } else if learned_dense == Some(false) {
            try_dense_hot_scalar_edges::<_, true, false, false, false>(
                state,
                vocab,
                trie,
                root_lexer,
                root_parser,
                initial_lexer_state,
                finalizer_code,
                single_finalizer_continues,
                lexer_scan_cache.tokenizer(),
                &transitions,
                &mut parser_cache,
                hot_scalar.take().unwrap(),
                None,
                buf,
            )?
        } else {
            try_dense_hot_scalar_edges::<_, true, false, true, false>(
                state,
                vocab,
                trie,
                root_lexer,
                root_parser,
                initial_lexer_state,
                finalizer_code,
                single_finalizer_continues,
                lexer_scan_cache.tokenizer(),
                &transitions,
                &mut parser_cache,
                hot_scalar.take().unwrap(),
                None,
                buf,
            )?
        };
        match dense_result {
            Some(true) => {
                update_special_token_mask(state, buf);
                state.clear_late_grammar_placeholder_mask(buf);
                return Ok(true);
            }
            Some(false) | None => {
                // The narrow lane may have mutated either output polarity
                // before discovering an unsupported correlated state. The
                // general positive-rebuild walker recomputes the exact mask.
                buf.fill(0);
            }
        }
    }

    let mut stack_lexer = [FULL_WALK_LEXER_DEAD; 256];
    let mut stack_parser = [0u32; 256];
    let mut stack_two = [((0u32, 0u32), (0u32, 0u32)); 256];
    let empty_guarded_pair = FullWalkGuardedPair {
        continuing_lexer: 0,
        continuing_parser: 0,
        pending_parser: 0,
        guard_terminal: 0,
    };
    // Profiling-only observation for the fragment experiment: whether the
    // vocabulary prefix reaching this depth has already crossed a terminal
    // finalizer and therefore required a parser effect. This deliberately
    // tracks only the certified scalar lane; ambiguity is conservatively
    // treated as parser-dependent below.
    let mut stack_parser_effect_seen = [false; 256];
    // Multi-branch frontiers embed a large SmallVec. Even an array of empty
    // Options would enlarge the frame of every scalar fast-path invocation;
    // allocate depth storage only after an actual ambiguous frontier appears.
    let accelerated = full_walk_acceleration_enabled();
    let mut many_transition_memo = FullWalkManyTransitionMemo::new(
        if accelerated { FullWalkManyTransitionMemo::MAX_STATES } else { 0 },
    );
    let mut stack_many = FullWalkManyStack::default();
    let mut pair_union_cache = FxHashMap::<(u32, u32), Option<u32>>::default();
    let mut triple_union_cache = FxHashMap::<(u32, u32, u32), Option<u32>>::default();
    if root_branches.len() == 1 && root_branches[0].initial_prune_guard.is_passed() {
        stack_lexer[0] = root_branches[0].tokenizer_config;
        stack_parser[0] = root_parser_nodes[0];
    } else if root_branches.len() == 2
        && root_branches.iter().all(|root| root.initial_prune_guard.is_passed())
    {
        let first = (root_branches[0].tokenizer_config, root_parser_nodes[0]);
        let second = (root_branches[1].tokenizer_config, root_parser_nodes[1]);
        if let Some((lexer_state, parser_node)) =
            full_walk_merge_two_same_parser(&transitions, vocab, &mut pair_union_cache, first, second)
        {
            stack_lexer[0] = lexer_state;
            stack_parser[0] = parser_node;
        } else {
            stack_lexer[0] = if first.1 != second.1 {
                FULL_WALK_LEXER_TWO_DISTINCT
            } else {
                FULL_WALK_LEXER_TWO
            };
            stack_two[0] = (first, second);
        }
    } else {
        stack_lexer[0] = FULL_WALK_LEXER_MULTI;
        let mut roots = FullWalkBranches::new();
        for (root_index, root) in root_branches.iter().enumerate() {
            full_walk_push_unique(
                &mut roots,
                FullWalkBranch {
                    lexer_state: root.tokenizer_config,
                    parser_node: root_parser_nodes[root_index],
                    prune_guard: FullWalkPruneGuard::from_initial(&root.initial_prune_guard, vocab),
                },
            );
        }
        if let Some(guarded) = full_walk_guarded_pair_from_branches(&roots, initial_lexer_state) {
            stack_lexer[0] = FULL_WALK_LEXER_GUARDED_PAIR;
            stack_two[0] = guarded.pack();
        } else if let Some((lexer_state, parser_node)) = full_walk_merge_branches_same_parser(
            &transitions,
            vocab,
            &mut pair_union_cache,
            &mut triple_union_cache,
            &roots,
        )
        {
            stack_lexer[0] = lexer_state;
            stack_parser[0] = parser_node;
        } else {
            stack_many.save(0, &many_transition_memo.hold(full_walk_many_state_from_branches(roots)));
        }
    }

    let walk_ops = trie.full_walk_ops();
    let token_markers = vocab.full_walk_token_markers_for(trie);
    let mut token_marker_index = 0usize;
    let tokenizer = lexer_scan_cache.tokenizer();

    let mut scalar_lexer = FULL_WALK_LEXER_DEAD;
    let mut scalar_parser = 0u32;
    let mut parser_effect_seen = false;
    let mut current_two = ((0u32, 0u32), (0u32, 0u32));
    let mut current_guarded_pair = empty_guarded_pair;
    // Exact one-state bound-continuation cache for the dedicated guarded-pair
    // lane. A byte is recorded only after the reference generic executor has
    // processed `(state, byte)` and returned the identical compact state.
    let mut guarded_self_loop_state = None::<FullWalkGuardedPair>;
    let mut guarded_self_loop_bytes = [0u64; 4];
    let mut guarded_outer_cache_hits = 0usize;
    let mut identity_proofs = FullWalkIdentityProofCache::default();
    let mut witness_proofs = FullWalkIdentityProofCache::default();
    static LIVE_BRANCH_WITNESS: OnceLock<bool> = OnceLock::new();
    let live_branch_witness_enabled = *LIVE_BRANCH_WITNESS.get_or_init(||
        !env_flag("GLRMASK_DISABLE_LIVE_BRANCH_WITNESS", false));
    let mut live_witness_subtrees = 0usize;
    let mut guarded_proof_state = None::<FullWalkGuardedPair>;
    let mut scalar_identity = FullWalkScalarIdentity::default();
    let mut identity_subtrees_skipped = 0usize;
    let mut identity_subtree_ops_skipped = 0usize;
    let mut identity_subtree_tokens = 0usize;
    let mut current_many = FullWalkManyCursor::Direct(FullWalkManyState::Branches(FullWalkBranches::new()));
    let mut partition_root_slot = 0usize;
    // `sparse_root_range` was computed above (Change D) from the same exact
    // single-live-root-byte proof and is reused unchanged here. It stays a
    // `walk_ops[start..]` iterator with `range_end` stopping logic because the
    // skip helpers derive absolute offsets from `walk_ops.len() - remaining.len()`.
    let sparse_root_range_end = sparse_root_range.map_or(0, |range| range.1);
    let mut remaining_ops = if let Some((start, _, marker_start)) = sparse_root_range {
        token_marker_index = marker_start;
        walk_ops[start as usize..].iter()
    } else {
        walk_ops.iter()
    };
    let mut profile_ops = 0usize;
    let mut profile_bytes = 0usize;
    let mut profile_token_endpoints = 0usize;
    let mut profile_finalizing_bytes = 0usize;
    let mut profile_direct_finalizers = 0usize;
    let mut profile_pre_effect_byte_ops = 0usize;
    let mut profile_pre_effect_token_endpoints = 0usize;
    let mut profile_first_effect_frontiers = 0usize;
    let mut profile_scalar_lane_bytes = 0usize;
    let mut profile_two_distinct_lane_bytes = 0usize;
    let mut profile_two_same_lane_bytes = 0usize;
    let mut profile_multi_lane_bytes = 0usize;
    let mut profile_multi_three_same_bytes = 0usize;
    let mut profile_multi_branches_2 = 0usize;
    let mut profile_multi_branches_3 = 0usize;
    let mut profile_multi_branches_4plus = 0usize;
    let mut profile_multi_two_same_parser = 0usize;
    let mut profile_multi_two_both_passed = 0usize;
    let mut profile_multi_two_one_pending = 0usize;
    let mut profile_multi_two_both_pending = 0usize;
    let mut profile_multi_two_pending_mem1 = 0usize;
    let mut profile_multi_two_pending_mem2 = 0usize;
    let mut profile_multi_two_pending_mem3plus = 0usize;
    let mut profile_multi_two_guard_eq_passed_lexer = 0usize;
    let mut profile_multi_two_pending_lexer_is_initial = 0usize;
    let mut profile_dead_tokens_cleared = 0usize;
    let mut deferred_allowed_markers = Vec::<u64>::new();
    let mut deferred_rejected_markers = Vec::<u64>::new();
    let mut deferred_dead_subtrees = Vec::<u32>::new();
    let mut deferred_positive_mutations = 0usize;
    let mut deferred_negative_mutations = 0usize;
    let mut deferred_positive_work = 0usize;
    let mut deferred_negative_work = 0usize;
    let total_original_tokens = if deferred_output {
        vocab.subtree_original_tokens_for(trie, 0).len()
    } else {
        0
    };
    let mut output_scope_cursor = FullWalkOutputScopeCursor::default();
    let mut output_scope_attempted = false;
    let mut output_scope_visited = 0usize;
    let mut output_scope_skips = 0usize;
    let mut output_scope_saved_ops = 0usize;
    let scope_build_after = if cfg!(test) { 0 } else { 1024 };
    let walk_started = profile_kernel.then(std::time::Instant::now);
    'walk: loop {
        if sparse_root_range.is_some() {
            let current_op = walk_ops.len() - remaining_ops.as_slice().len();
            if current_op >= sparse_root_range_end as usize {
                break;
            }
        }
        let Some(&op) = remaining_ops.next() else {
            break;
        };
        let identity_active = parser_cache.identity_proofs_enabled;
        // A surviving scalar non-finalizing byte is retained only after
        // `physical_token_boundary_allowed()` has proved the exact
        // `(parser_node, target_lexer)` coordinate live.  When that same op is
        // also a model-token endpoint, repeating `token_boundary_allowed_raw`
        // is therefore redundant.  Keep the ordinary endpoint query for
        // zero-byte edges, finalizer transitions, and all correlated lanes.
        let mut scalar_endpoint_known_allowed = false;
        if profile_walk {
            profile_ops += 1;
            profile_bytes += usize::from(op.consumes_byte());
            profile_token_endpoints += usize::from(op.ends_edge() && op.child_is_token());
        }
        let parent_depth = op.parent_depth() as usize;
        if op.starts_edge() {
            if profile_walk {
                parser_effect_seen = unsafe {
                    *stack_parser_effect_seen.get_unchecked(parent_depth)
                };
            }
            scalar_lexer = unsafe { *stack_lexer.get_unchecked(parent_depth) };
            if scalar_lexer < FULL_WALK_LEXER_TWO_DISTINCT {
                scalar_parser = unsafe { *stack_parser.get_unchecked(parent_depth) };
            } else if scalar_lexer == FULL_WALK_LEXER_TWO_DISTINCT || scalar_lexer == FULL_WALK_LEXER_TWO {
                current_two = unsafe { *stack_two.get_unchecked(parent_depth) };
            } else if scalar_lexer == FULL_WALK_LEXER_GUARDED_PAIR {
                current_guarded_pair = FullWalkGuardedPair::unpack(unsafe {
                    *stack_two.get_unchecked(parent_depth)
                });
            } else if scalar_lexer == FULL_WALK_LEXER_MULTI {
                current_many.clone_from(stack_many.restore(parent_depth));
            }

            if parent_depth == 0 && !op.consumes_byte() {
                let root_slot = partition_root_slot;
                partition_root_slot += 1;
                if let Some(class) = trie.root_layout_class(root_slot) {
                    let master_skip = llg_master_decision.is_some_and(|decision| {
                        decision.admits_root_class(class)
                            || safe_envelope_upper.is_some_and(|upper| {
                                let count =
                                    crate::runtime::dynamic_mask_llg_master_safe_chars(class);
                                count != 0 && count > upper
                            })
                    });
                    if master_skip {
                        super::full_walk_skip_admitted_subtree_generic(
                            trie,
                            walk_ops,
                            &mut remaining_ops,
                            &mut token_marker_index,
                        );
                        continue;
                    }
                }
            }
        }

        if let Some(ignored) = ignored_output {
            output_scope_visited += 1;
            if !output_scope_attempted && output_scope_visited >= scope_build_after {
                output_scope_attempted = true;
            }
            if op.starts_edge() && op.consumes_byte()
                && output_scope_attempted
            {
                let op_index = walk_ops.len() - remaining_ops.as_slice().len() - 1;
                let (child, end_op) = trie.full_walk_dead_subtree(op_index);
                let root_offset = usize::from(trie.node(0).token_id.is_some());
                let end_token = trie.subtree_token_index_range(child).end.saturating_sub(root_offset);
                let covered = output_scope_cursor.covers(
                    token_markers, token_marker_index, end_token, |marker| {
                        full_walk_marker_fully_ignored(marker, ignored, |canonical| {
                            vocab.token_ids(canonical).is_some_and(|aliases| {
                                aliases.iter().all(|&id| ignored.get(id as usize / 32)
                                    .is_some_and(|word| word & (1u32 << (id % 32)) != 0))
                            })
                        })
                    },
                );
                if covered {
                    output_scope_skips += 1;
                    #[cfg(test)]
                    TEST_OUTPUT_SCOPE_SKIPPED.with(|count| count.set(count.get() + 1));
                    output_scope_saved_ops += end_op as usize - op_index;
                    token_marker_index = end_token;
                    remaining_ops = walk_ops[end_op as usize..].iter();
                    // No transition/acceptance result is fabricated or cached.
                    // The next preorder edge restores an existing ancestor;
                    // skipped frames and don't-care output bits are not read.
                    continue 'walk;
                }
            }
        }

        if op.consumes_byte() {
            let byte = op.byte();
            if profile_walk {
                if scalar_lexer < FULL_WALK_LEXER_TWO_DISTINCT {
                    profile_scalar_lane_bytes += 1;
                } else if scalar_lexer == FULL_WALK_LEXER_TWO_DISTINCT {
                    profile_two_distinct_lane_bytes += 1;
                } else if scalar_lexer == FULL_WALK_LEXER_TWO {
                    profile_two_same_lane_bytes += 1;
                } else if scalar_lexer == FULL_WALK_LEXER_GUARDED_PAIR {
                    profile_multi_lane_bytes += 1;
                    profile_multi_branches_2 += 1;
                    profile_multi_two_one_pending += 1;
                    profile_multi_two_pending_mem1 += 1;
                    profile_multi_two_guard_eq_passed_lexer += 1;
                    profile_multi_two_pending_lexer_is_initial += 1;
                } else if scalar_lexer == FULL_WALK_LEXER_MULTI {
                    profile_multi_lane_bytes += 1;
                    match many_transition_memo.view(&current_many) {
                        FullWalkManyState::ThreeSameParser { .. } => {
                            profile_multi_three_same_bytes += 1;
                        }
                        FullWalkManyState::Branches(branches) => match branches.len() {
                            2 => {
                                profile_multi_branches_2 += 1;
                                if branches[0].parser_node == branches[1].parser_node {
                                    profile_multi_two_same_parser += 1;
                                }
                                match (
                                    branches[0].prune_guard.is_passed(),
                                    branches[1].prune_guard.is_passed(),
                                ) {
                                    (true, true) => profile_multi_two_both_passed += 1,
                                    (false, false) => profile_multi_two_both_pending += 1,
                                    _ => {
                                        profile_multi_two_one_pending += 1;
                                        let pending = if branches[0].prune_guard.is_passed() {
                                            &branches[1].prune_guard
                                        } else {
                                            &branches[0].prune_guard
                                        };
                                        let FullWalkPruneGuard::Pending(memories) = pending else {
                                            unreachable!("one-pending profile state must have a pending guard");
                                        };
                                        match memories.len() {
                                            1 => {
                                                profile_multi_two_pending_mem1 += 1;
                                                let (pending_branch, passed_branch) = if branches[0]
                                                    .prune_guard
                                                    .is_passed()
                                                {
                                                    (&branches[1], &branches[0])
                                                } else {
                                                    (&branches[0], &branches[1])
                                                };
                                                if memories[0].0 == passed_branch.lexer_state {
                                                    profile_multi_two_guard_eq_passed_lexer += 1;
                                                }
                                                if pending_branch.lexer_state == initial_lexer_state {
                                                    profile_multi_two_pending_lexer_is_initial += 1;
                                                }
                                            }
                                            2 => profile_multi_two_pending_mem2 += 1,
                                            _ => profile_multi_two_pending_mem3plus += 1,
                                        }
                                    }
                                }
                            }
                            3 => profile_multi_branches_3 += 1,
                            4.. => profile_multi_branches_4plus += 1,
                            _ => {}
                        },
                    }
                }
            }
            if profile_walk && !parser_effect_seen {
                profile_pre_effect_byte_ops += 1;
            }
            if scalar_lexer == FULL_WALK_LEXER_DEAD {
            } else if scalar_lexer < FULL_WALK_LEXER_TWO_DISTINCT {
                let scalar_identity_source = (scalar_lexer, scalar_parser);
                let mut scalar_ordinary_identity = false;
                let cell = transitions.cell(scalar_lexer, byte);
                if T::cell_is_dead(cell) {
                    scalar_lexer = FULL_WALK_LEXER_DEAD;
                    let rejected = full_walk_skip_lexically_dead_subtree(
                        vocab,
                        trie,
                        walk_ops,
                        &mut remaining_ops,
                        &mut token_marker_index,
                        buf,
                        !positive_rebuild && !deferred_output,
                        &mut deferred_dead_subtrees,
                        deferred_output,
                    );
                    profile_dead_tokens_cleared += rejected;
                    if deferred_output {
                        deferred_negative_mutations += rejected;
                        deferred_negative_work += rejected;
                        full_walk_maybe_commit_deferred_positive(
                            vocab,
                            total_original_tokens,
                            &mut deferred_output,
                            &mut positive_rebuild,
                            deferred_negative_mutations,
                            &mut deferred_allowed_markers,
                            &mut deferred_rejected_markers,
                            &mut deferred_dead_subtrees,
                            buf,
                        );
                    }
                    continue;
                } else {
                    let target = T::cell_target(cell);
                    if !T::cell_has_finalizer(cell) {
                        // A globally live lexer state can still be impossible
                        // for this correlated parser branch when every terminal
                        // it could eventually produce is parser-inadmissible.
                        // Kill that entire vocabulary subtree at the first
                        // impossible prefix. Genuine near-full walks use the
                        // dedicated dense hot lane above instead, where paying
                        // this parser-conditioned check on every byte loses.
                        if parser_cache.physical_token_boundary_allowed(
                            state.constraint,
                            tokenizer,
                            &transitions,
                            scalar_parser,
                            target,
                        ) {
                            scalar_ordinary_identity = scalar_lexer == target;
                            scalar_lexer = target;
                            scalar_endpoint_known_allowed = true;
                        } else {
                            scalar_lexer = FULL_WALK_LEXER_DEAD;
                            let rejected = full_walk_skip_lexically_dead_subtree(
                                vocab,
                                trie,
                                walk_ops,
                                &mut remaining_ops,
                                &mut token_marker_index,
                                buf,
                                !positive_rebuild && !deferred_output,
                                &mut deferred_dead_subtrees,
                                deferred_output,
                            );
                            profile_dead_tokens_cleared += rejected;
                            if deferred_output {
                                deferred_negative_mutations += rejected;
                                deferred_negative_work += rejected;
                                full_walk_maybe_commit_deferred_positive(
                                    vocab,
                                    total_original_tokens,
                                    &mut deferred_output,
                                    &mut positive_rebuild,
                                    deferred_negative_mutations,
                                    &mut deferred_allowed_markers,
                                    &mut deferred_rejected_markers,
                                    &mut deferred_dead_subtrees,
                                    buf,
                                );
                            }
                            continue;
                        }
                    } else {
                        if profile_walk {
                            profile_finalizing_bytes += 1;
                            if !parser_effect_seen {
                                profile_first_effect_frontiers += 1;
                                parser_effect_seen = true;
                            }
                        }
                        let direct_applied = HOT_SINGLE_ROOT
                            && full_walk_try_apply_plain_single_finalizer(
                                target,
                                scalar_parser,
                                initial_lexer_state,
                                finalizer_code,
                                single_finalizer_continues,
                                &transitions,
                                &mut parser_cache,
                                state.constraint,
                                FULL_WALK_LEXER_TWO_DISTINCT,
                                &mut scalar_lexer,
                                &mut scalar_parser,
                                &mut current_two,
                            );
                        if profile_walk && direct_applied {
                            profile_direct_finalizers += 1;
                        }
                        if !direct_applied {
                            let outcome = if HOT_SINGLE_ROOT {
                                full_walk_scalar_finalizer_hot_single(
                                    target,
                                    scalar_parser,
                                    initial_lexer_state,
                                    finalizer_code,
                                    single_finalizer_continues,
                                    tokenizer,
                                    &transitions,
                                    &mut parser_cache,
                                    state.constraint,
                                )
                            } else {
                                full_walk_scalar_finalizer(
                                    target,
                                    scalar_parser,
                                    initial_lexer_state,
                                    finalizer_code,
                                    single_finalizer_continues,
                                    tokenizer,
                                    &transitions,
                                    &mut parser_cache,
                                    state.constraint,
                                )
                            };
                            match outcome {
                            FullWalkScalarFinalizerOutcome::Scalar(branch) => {
                                scalar_lexer = branch.lexer_state;
                                scalar_parser = branch.parser_node;
                            }
                            FullWalkScalarFinalizerOutcome::Two(first, second) => {
                                if first.prune_guard.is_passed() && second.prune_guard.is_passed() {
                                    if let Some((lexer_state, parser_node)) =
                                        full_walk_merge_two_same_parser(
                                            &transitions,
                                            vocab,
                                            &mut pair_union_cache,
                                            (first.lexer_state, first.parser_node),
                                            (second.lexer_state, second.parser_node),
                                        )
                                    {
                                        scalar_lexer = lexer_state;
                                        scalar_parser = parser_node;
                                    } else {
                                        scalar_lexer = if first.parser_node != second.parser_node {
                                            FULL_WALK_LEXER_TWO_DISTINCT
                                        } else {
                                            FULL_WALK_LEXER_TWO
                                        };
                                        current_two = (
                                            (first.lexer_state, first.parser_node),
                                            (second.lexer_state, second.parser_node),
                                        );
                                    }
                                } else {
                                    let mut next = FullWalkBranches::new();
                                    next.push(first);
                                    next.push(second);
                                    if let Some(guarded) =
                                        full_walk_guarded_pair_from_branches(&next, initial_lexer_state)
                                    {
                                        scalar_lexer = FULL_WALK_LEXER_GUARDED_PAIR;
                                        current_guarded_pair = guarded;
                                    } else {
                                        scalar_lexer = FULL_WALK_LEXER_MULTI;
                                        current_many = many_transition_memo.hold(full_walk_many_state_from_branches(next));
                                    }
                                }
                            }
                            FullWalkScalarFinalizerOutcome::Many(next) => {
                                if let [first, second] = next.as_slice() {
                                    if first.prune_guard.is_passed() && second.prune_guard.is_passed() {
                                        if let Some((lexer_state, parser_node)) =
                                            full_walk_merge_two_same_parser(
                                                &transitions,
                                                vocab,
                                                &mut pair_union_cache,
                                                (first.lexer_state, first.parser_node),
                                                (second.lexer_state, second.parser_node),
                                            )
                                        {
                                            scalar_lexer = lexer_state;
                                            scalar_parser = parser_node;
                                        } else {
                                            scalar_lexer = if first.parser_node != second.parser_node {
                                                FULL_WALK_LEXER_TWO_DISTINCT
                                            } else {
                                                FULL_WALK_LEXER_TWO
                                            };
                                            current_two = (
                                                (first.lexer_state, first.parser_node),
                                                (second.lexer_state, second.parser_node),
                                            );
                                        }
                                    } else {
                                        if let Some(guarded) = full_walk_guarded_pair_from_branches(
                                            &next,
                                            initial_lexer_state,
                                        ) {
                                            scalar_lexer = FULL_WALK_LEXER_GUARDED_PAIR;
                                            current_guarded_pair = guarded;
                                        } else {
                                            scalar_lexer = FULL_WALK_LEXER_MULTI;
                                            current_many = many_transition_memo.hold(full_walk_many_state_from_branches(next));
                                        }
                                    }
                                } else if let Some((lexer_state, parser_node)) =
                                    full_walk_merge_branches_same_parser(
                                        &transitions,
                                        vocab,
                                        &mut pair_union_cache,
                                        &mut triple_union_cache,
                                        &next,
                                    )
                                {
                                    scalar_lexer = lexer_state;
                                    scalar_parser = parser_node;
                                } else {
                                    if let Some(guarded) =
                                        full_walk_guarded_pair_from_branches(&next, initial_lexer_state)
                                    {
                                        scalar_lexer = FULL_WALK_LEXER_GUARDED_PAIR;
                                        current_guarded_pair = guarded;
                                    } else {
                                        scalar_lexer = FULL_WALK_LEXER_MULTI;
                                        current_many = many_transition_memo.hold(full_walk_many_state_from_branches(next));
                                    }
                                }
                            }
                            }
                        }
                    }
                }
                if identity_active {
                    let source = scalar_identity_source;
                    scalar_identity.remember(source, (scalar_lexer, scalar_parser), byte);
                    if scalar_ordinary_identity
                        && scalar_identity.state == Some(source)
                        && !scalar_identity.proof_checked
                    {
                        scalar_identity.proof_checked = true;
                        // Admission succeeded at this exact (lexer, parser)
                        // pair. All non-finalizing self-edges have the same
                        // target and parser-conditioned liveness result.
                        if let Some(classes) = identity_proofs.get(&transitions, tokenizer, source.0) {
                            for (known, exact) in scalar_identity.bytes.iter_mut().zip(classes.ordinary) {
                                *known |= exact;
                            }
                        }
                    }
                }
            } else if scalar_lexer == FULL_WALK_LEXER_TWO_DISTINCT {
                if profile_walk {
                    parser_effect_seen = true;
                }
                // Distinct parser contexts require parser-conditioned liveness
                // even while both lexer cells are non-finalizing. The old
                // lexical-only shortcut kept parser-dead branches alive across
                // almost the whole vocabulary on common JSON states.
                match full_walk_step_two::<T>(
                    current_two,
                    byte,
                    initial_lexer_state,
                    finalizer_code,
                    single_finalizer_continues,
                    tokenizer,
                    &transitions,
                    &mut parser_cache,
                    state.constraint,
                ) {
                        FullWalkTwoStepOutcome::Dead => {
                            scalar_lexer = FULL_WALK_LEXER_DEAD;
                            let rejected = full_walk_skip_lexically_dead_subtree(
                                vocab,
                                trie,
                                walk_ops,
                                &mut remaining_ops,
                                &mut token_marker_index,
                                buf,
                                !positive_rebuild && !deferred_output,
                                &mut deferred_dead_subtrees,
                                deferred_output,
                            );
                            profile_dead_tokens_cleared += rejected;
                            if deferred_output {
                                deferred_negative_mutations += rejected;
                                deferred_negative_work += rejected;
                                full_walk_maybe_commit_deferred_positive(
                                    vocab,
                                    total_original_tokens,
                                    &mut deferred_output,
                                    &mut positive_rebuild,
                                    deferred_negative_mutations,
                                    &mut deferred_allowed_markers,
                                    &mut deferred_rejected_markers,
                                    &mut deferred_dead_subtrees,
                                    buf,
                                );
                            }
                            continue;
                        }
                        FullWalkTwoStepOutcome::One((lexer, parser)) => {
                            scalar_lexer = lexer;
                            scalar_parser = parser;
                        }
                        FullWalkTwoStepOutcome::Two(first, second) => {
                            if let Some((lexer_state, parser_node)) =
                                full_walk_merge_two_same_parser(
                                    &transitions,
                                    vocab,
                                    &mut pair_union_cache,
                                    first,
                                    second,
                                )
                            {
                                scalar_lexer = lexer_state;
                                scalar_parser = parser_node;
                            } else {
                                scalar_lexer = if first.1 != second.1 {
                                    FULL_WALK_LEXER_TWO_DISTINCT
                                } else {
                                    FULL_WALK_LEXER_TWO
                                };
                                current_two = (first, second);
                            }
                        }
                        FullWalkTwoStepOutcome::Many(next) => {
                            if let Some((lexer_state, parser_node)) =
                                full_walk_merge_branches_same_parser(
                                            &transitions,
                                            vocab,
                                            &mut pair_union_cache,
                                            &mut triple_union_cache,
                                            &next,
                                        )
                            {
                                scalar_lexer = lexer_state;
                                scalar_parser = parser_node;
                            } else {
                                if let Some(guarded) =
                                    full_walk_guarded_pair_from_branches(&next, initial_lexer_state)
                                {
                                    scalar_lexer = FULL_WALK_LEXER_GUARDED_PAIR;
                                    current_guarded_pair = guarded;
                                } else {
                                    scalar_lexer = FULL_WALK_LEXER_MULTI;
                                    current_many = many_transition_memo.hold(full_walk_many_state_from_branches(next));
                                }
                            }
                        }
                }
            } else if scalar_lexer == FULL_WALK_LEXER_TWO {
                if profile_walk {
                    parser_effect_seen = true;
                }
                match full_walk_step_two::<T>(
                        current_two,
                        byte,
                        initial_lexer_state,
                        finalizer_code,
                        single_finalizer_continues,
                        tokenizer,
                        &transitions,
                        &mut parser_cache,
                        state.constraint,
                    ) {
                        FullWalkTwoStepOutcome::Dead => {
                            scalar_lexer = FULL_WALK_LEXER_DEAD;
                            let rejected = full_walk_skip_lexically_dead_subtree(
                                vocab,
                                trie,
                                walk_ops,
                                &mut remaining_ops,
                                &mut token_marker_index,
                                buf,
                                !positive_rebuild && !deferred_output,
                                &mut deferred_dead_subtrees,
                                deferred_output,
                            );
                            profile_dead_tokens_cleared += rejected;
                            if deferred_output {
                                deferred_negative_mutations += rejected;
                                deferred_negative_work += rejected;
                                full_walk_maybe_commit_deferred_positive(
                                    vocab,
                                    total_original_tokens,
                                    &mut deferred_output,
                                    &mut positive_rebuild,
                                    deferred_negative_mutations,
                                    &mut deferred_allowed_markers,
                                    &mut deferred_rejected_markers,
                                    &mut deferred_dead_subtrees,
                                    buf,
                                );
                            }
                            continue;
                        }
                        FullWalkTwoStepOutcome::One((lexer, parser)) => {
                            scalar_lexer = lexer;
                            scalar_parser = parser;
                        }
                        FullWalkTwoStepOutcome::Two(first, second) => {
                            if let Some((lexer_state, parser_node)) =
                                full_walk_merge_two_same_parser(
                                    &transitions,
                                    vocab,
                                    &mut pair_union_cache,
                                    first,
                                    second,
                                )
                            {
                                scalar_lexer = lexer_state;
                                scalar_parser = parser_node;
                            } else {
                                scalar_lexer = if first.1 != second.1 {
                                    FULL_WALK_LEXER_TWO_DISTINCT
                                } else {
                                    FULL_WALK_LEXER_TWO
                                };
                                current_two = (first, second);
                            }
                        }
                        FullWalkTwoStepOutcome::Many(next) => {
                            if let Some((lexer_state, parser_node)) =
                                full_walk_merge_branches_same_parser(
                                            &transitions,
                                            vocab,
                                            &mut pair_union_cache,
                                            &mut triple_union_cache,
                                            &next,
                                        )
                            {
                                scalar_lexer = lexer_state;
                                scalar_parser = parser_node;
                            } else {
                                if let Some(guarded) =
                                    full_walk_guarded_pair_from_branches(&next, initial_lexer_state)
                                {
                                    scalar_lexer = FULL_WALK_LEXER_GUARDED_PAIR;
                                    current_guarded_pair = guarded;
                                } else {
                                    scalar_lexer = FULL_WALK_LEXER_MULTI;
                                    current_many = many_transition_memo.hold(full_walk_many_state_from_branches(next));
                                }
                            }
                        }
                    }
            } else if scalar_lexer == FULL_WALK_LEXER_GUARDED_PAIR {
                if profile_walk {
                    parser_effect_seen = true;
                }
                // The inner bound helper is intentionally out of line and has
                // a large outcome. Avoid that call entirely on an already
                // proven identity byte; endpoints still use the unchanged pair.
                if accelerated
                    && guarded_self_loop_state == Some(current_guarded_pair)
                    && guarded_self_loop_bytes[byte as usize >> 6] & (1u64 << (byte & 63)) != 0
                {
                    guarded_outer_cache_hits += 1;
                } else {
                match full_walk_step_guarded_pair_bound(
                    current_guarded_pair,
                    byte,
                    initial_lexer_state,
                    finalizer_code,
                    single_finalizer_continues,
                    tokenizer,
                    &transitions,
                    &mut parser_cache,
                    state.constraint,
                    vocab,
                    &mut pair_union_cache,
                    &mut triple_union_cache,
                    &mut guarded_self_loop_state,
                    &mut guarded_self_loop_bytes,
                ) {
                    FullWalkGuardedStepOutcome::Dead => {
                        scalar_lexer = FULL_WALK_LEXER_DEAD;
                        let rejected = full_walk_skip_lexically_dead_subtree(
                            vocab,
                            trie,
                            walk_ops,
                            &mut remaining_ops,
                            &mut token_marker_index,
                            buf,
                            !positive_rebuild && !deferred_output,
                            &mut deferred_dead_subtrees,
                            deferred_output,
                        );
                        profile_dead_tokens_cleared += rejected;
                        if deferred_output {
                            deferred_negative_mutations += rejected;
                            deferred_negative_work += rejected;
                            full_walk_maybe_commit_deferred_positive(
                                vocab,
                                total_original_tokens,
                                &mut deferred_output,
                                &mut positive_rebuild,
                                deferred_negative_mutations,
                                &mut deferred_allowed_markers,
                                &mut deferred_rejected_markers,
                                &mut deferred_dead_subtrees,
                                buf,
                            );
                        }
                        continue;
                    }
                    FullWalkGuardedStepOutcome::Scalar(lexer, parser) => {
                        scalar_lexer = lexer;
                        scalar_parser = parser;
                    }
                    FullWalkGuardedStepOutcome::Two(first, second) => {
                        scalar_lexer = if first.1 != second.1 {
                            FULL_WALK_LEXER_TWO_DISTINCT
                        } else {
                            FULL_WALK_LEXER_TWO
                        };
                        current_two = (first, second);
                    }
                    FullWalkGuardedStepOutcome::Guarded(guarded) => {
                        if accelerated {
                            // The helper's single-finalizer shortcut returns
                            // before its old identity recorder. Cover that exact
                            // shortcut too, rather than relearning it per byte.
                            remember_guarded_identity(
                                current_guarded_pair, guarded, byte,
                                &mut guarded_self_loop_state, &mut guarded_self_loop_bytes,
                            );
                            if identity_active
                                && guarded == current_guarded_pair
                                && guarded_proof_state != Some(guarded)
                            {
                                guarded_proof_state = Some(guarded);
                                // This is precisely the authoritative helper's
                                // single-finalizer shortcut, for every byte in
                                // the certified self-transition class.
                                if transitions.finalizer_code(guarded.continuing_lexer, finalizer_code)
                                    == guarded.guard_terminal
                                    && transitions.single_finalizer_continues(
                                        guarded.continuing_lexer, single_finalizer_continues)
                                    && let Some(classes) = identity_proofs.get(
                                        &transitions, tokenizer, guarded.continuing_lexer)
                                {
                                    for (known, exact) in guarded_self_loop_bytes.iter_mut().zip(classes.finalizing) {
                                        *known |= exact;
                                    }
                                }
                            }
                        }
                        current_guarded_pair = guarded;
                    }
                    FullWalkGuardedStepOutcome::Many(next) => {
                        scalar_lexer = FULL_WALK_LEXER_MULTI;
                        current_many = many_transition_memo.hold(next);
                    }
                }
                }
            } else if scalar_lexer == FULL_WALK_LEXER_MULTI {
                if profile_walk {
                    parser_effect_seen = true;
                }
                'advance_many: {
                let source_id = current_many.id();
                if let Some(cached) = many_transition_memo.cached_transition(&current_many, byte) {
                    current_many.set_cached(cached);
                    break 'advance_many;
                }
                let next = full_walk_step_many_state(
                    many_transition_memo.view(&current_many),
                    byte,
                    initial_lexer_state,
                    finalizer_code,
                    single_finalizer_continues,
                    tokenizer,
                    &transitions,
                    &mut parser_cache,
                    state.constraint,
                );
                match next {
                    FullWalkManyState::Branches(next) => {
                        match next.as_slice() {
                            [] => {
                                scalar_lexer = FULL_WALK_LEXER_DEAD;
                                let rejected = full_walk_skip_lexically_dead_subtree(
                                    vocab,
                                    trie,
                                    walk_ops,
                                    &mut remaining_ops,
                                    &mut token_marker_index,
                                    buf,
                                    !positive_rebuild && !deferred_output,
                                    &mut deferred_dead_subtrees,
                                    deferred_output,
                                );
                                profile_dead_tokens_cleared += rejected;
                                if deferred_output {
                                    deferred_negative_mutations += rejected;
                                    deferred_negative_work += rejected;
                                    full_walk_maybe_commit_deferred_positive(
                                        vocab,
                                        total_original_tokens,
                                        &mut deferred_output,
                                        &mut positive_rebuild,
                                        deferred_negative_mutations,
                                        &mut deferred_allowed_markers,
                                        &mut deferred_rejected_markers,
                                        &mut deferred_dead_subtrees,
                                        buf,
                                    );
                                }
                                continue 'walk;
                            }
                            [branch] if branch.prune_guard.is_passed() => {
                                scalar_lexer = branch.lexer_state;
                                scalar_parser = branch.parser_node;
                            }
                            [first, second]
                                if first.prune_guard.is_passed() && second.prune_guard.is_passed() =>
                            {
                                if let Some((lexer_state, parser_node)) =
                                    full_walk_merge_two_same_parser(
                                        &transitions,
                                        vocab,
                                        &mut pair_union_cache,
                                        (first.lexer_state, first.parser_node),
                                        (second.lexer_state, second.parser_node),
                                    )
                                {
                                    scalar_lexer = lexer_state;
                                    scalar_parser = parser_node;
                                } else {
                                    scalar_lexer = if first.parser_node != second.parser_node {
                                        FULL_WALK_LEXER_TWO_DISTINCT
                                    } else {
                                        FULL_WALK_LEXER_TWO
                                    };
                                    current_two = (
                                        (first.lexer_state, first.parser_node),
                                        (second.lexer_state, second.parser_node),
                                    );
                                }
                            }
                            _ => {
                                if accelerated
                                    && let Some(guarded) = full_walk_guarded_pair_from_branches(&next, initial_lexer_state)
                                {
                                    // Ambiguity can shrink back to the exact
                                    // specialized pair. Do not strand that
                                    // shape in generic multi-branch execution.
                                    // The existing constructor checks both
                                    // parser IDs and the complete pending guard.
                                    scalar_lexer = FULL_WALK_LEXER_GUARDED_PAIR;
                                    current_guarded_pair = guarded;
                                } else if let Some((lexer_state, parser_node)) =
                                    full_walk_merge_branches_same_parser(
                                        &transitions,
                                        vocab,
                                        &mut pair_union_cache,
                                        &mut triple_union_cache,
                                        &next,
                                    )
                                {
                                    scalar_lexer = lexer_state;
                                    scalar_parser = parser_node;
                                } else {
                                    scalar_lexer = FULL_WALK_LEXER_MULTI;
                                    current_many = many_transition_memo.hold(FullWalkManyState::Branches(next));
                                }
                            }
                        }
                    }
                    next @ FullWalkManyState::ThreeSameParser { lexers, parser_node } => {
                        if let Some((lexer_state, parser_node)) =
                            full_walk_merge_three_same_parser(
                                &transitions,
                                vocab,
                                &mut triple_union_cache,
                                lexers,
                                parser_node,
                            )
                        {
                            scalar_lexer = lexer_state;
                            scalar_parser = parser_node;
                        } else {
                            scalar_lexer = FULL_WALK_LEXER_MULTI;
                            current_many = many_transition_memo.hold(next);
                        }
                    }
                }
                if scalar_lexer == FULL_WALK_LEXER_MULTI {
                    many_transition_memo.remember_transition(source_id, byte, &current_many);
                }
                }
            }
        }

        if op.ends_edge() {
            if identity_active {
                let identity = if scalar_lexer < FULL_WALK_LEXER_TWO_DISTINCT
                {
                    scalar_identity.alphabet((scalar_lexer, scalar_parser))
                } else if scalar_lexer == FULL_WALK_LEXER_GUARDED_PAIR
                    && guarded_self_loop_state == Some(current_guarded_pair)
                {
                    Some(&guarded_self_loop_bytes)
                } else if scalar_lexer == FULL_WALK_LEXER_MULTI {
                    many_transition_memo.identity_alphabet(&current_many)
                } else {
                    None
                };
                if identity.is_some()
                    || (live_branch_witness_enabled && scalar_lexer != FULL_WALK_LEXER_DEAD)
                {
                    let op_index = walk_ops.len() - remaining_ops.as_slice().len() - 1;
                    let (child, subtree_end) = trie.full_walk_dead_subtree(op_index);
                    let root_offset = usize::from(trie.node(0).token_id.is_some());
                    let token_end = trie.subtree_token_index_range(child).end.saturating_sub(root_offset);
                    let saved_ops = subtree_end as usize - op_index - 1;
                    let token_count = token_end.saturating_sub(token_marker_index);
                    let identical = saved_ops >= 8 && token_count >= 4
                        && identity.is_some_and(|identity|
                            identity_alphabet_covers_subtree(identity, trie.subtree_bytes(child)));
                    let witnessed = if !identical && live_branch_witness_enabled
                        && saved_ops >= 64 && token_count >= 16 {
                        let frontier = if scalar_lexer < FULL_WALK_LEXER_TWO_DISTINCT {
                            Some(FullWalkWitnessFrontier::One(scalar_lexer, scalar_parser))
                        } else if scalar_lexer == FULL_WALK_LEXER_TWO_DISTINCT
                            || scalar_lexer == FULL_WALK_LEXER_TWO {
                            Some(FullWalkWitnessFrontier::Two([current_two.0, current_two.1]))
                        } else if scalar_lexer == FULL_WALK_LEXER_GUARDED_PAIR {
                            Some(FullWalkWitnessFrontier::One(current_guarded_pair.continuing_lexer,
                                current_guarded_pair.continuing_parser))
                        } else if scalar_lexer == FULL_WALK_LEXER_MULTI {
                            Some(FullWalkWitnessFrontier::Many(many_transition_memo.view(&current_many)))
                        } else { None };
                        frontier.is_some_and(|frontier| full_walk_frontier_live_witness(
                            frontier, trie.subtree_bytes(child), &mut witness_proofs,
                            &mut parser_cache, state.constraint, tokenizer, &transitions))
                    } else { false };
                    if identical || witnessed {
                        live_witness_subtrees += usize::from(witnessed);
                        let allowed = witnessed || if scalar_lexer == FULL_WALK_LEXER_DEAD {
                            false
                        } else if scalar_lexer < FULL_WALK_LEXER_TWO_DISTINCT {
                            scalar_endpoint_known_allowed
                                || parser_cache.token_boundary_allowed_raw(
                                    state.constraint,
                                    tokenizer,
                                    &transitions,
                                    initial_lexer_state,
                                    scalar_lexer,
                                    scalar_parser,
                                )
                        } else if scalar_lexer == FULL_WALK_LEXER_TWO_DISTINCT || scalar_lexer == FULL_WALK_LEXER_TWO {
                            parser_cache.token_boundary_allowed_raw(
                                state.constraint,
                                tokenizer,
                                &transitions,
                                initial_lexer_state,
                                current_two.0.0,
                                current_two.0.1,
                            ) || parser_cache.token_boundary_allowed_raw(
                                state.constraint,
                                tokenizer,
                                &transitions,
                                initial_lexer_state,
                                current_two.1.0,
                                current_two.1.1,
                            )
                        } else if scalar_lexer == FULL_WALK_LEXER_GUARDED_PAIR {
                            // The pending branch is at the reset lexer state, so a
                            // model token may end here using the most recent accepted
                            // terminal without consulting parser/lexer liveness again.
                            true
                        } else if scalar_lexer == FULL_WALK_LEXER_MULTI {
                            match many_transition_memo.view(&current_many) {
                                FullWalkManyState::Branches(branches) => branches.iter().any(|branch| {
                                    parser_cache.token_boundary_allowed(
                                        state.constraint,
                                        tokenizer,
                                        &transitions,
                                        initial_lexer_state,
                                        branch,
                                    )
                                }),
                                FullWalkManyState::ThreeSameParser {
                                    lexers,
                                    parser_node,
                                } => parser_cache.token_boundary_allowed_raw(
                                    state.constraint,
                                    tokenizer,
                                    &transitions,
                                    initial_lexer_state,
                                    lexers.0,
                                    *parser_node,
                                ) || parser_cache.token_boundary_allowed_raw(
                                    state.constraint,
                                    tokenizer,
                                    &transitions,
                                    initial_lexer_state,
                                    lexers.1,
                                    *parser_node,
                                ) || parser_cache.token_boundary_allowed_raw(
                                    state.constraint,
                                    tokenizer,
                                    &transitions,
                                    initial_lexer_state,
                                    lexers.2,
                                    *parser_node,
                                ),
                            }
                        } else {
                            false
                        };
                        // Emit the same canonical markers and maintain the
                        // existing deferred/adaptive output polarity. A marker
                        // may represent several original vocabulary aliases.
                        // No descendants' parser/lexer transitions are needed.
                        if identity_subtree_requires_output(deferred_output, positive_rebuild, allowed) {
                            for &token_marker in &token_markers[token_marker_index..token_end] {
                                if deferred_output {
                                    let mutations = dynamic_token_marker_original_count(vocab, token_marker);
                                    let marker_work = dynamic_token_marker_materialization_cost(vocab, token_marker);
                                    if allowed {
                                        deferred_positive_mutations =
                                            deferred_positive_mutations.saturating_add(mutations);
                                        deferred_positive_work = deferred_positive_work.saturating_add(marker_work);
                                        deferred_allowed_markers.push(token_marker);
                                    } else {
                                        deferred_negative_mutations =
                                            deferred_negative_mutations.saturating_add(mutations);
                                        deferred_negative_work = deferred_negative_work.saturating_add(marker_work);
                                        deferred_rejected_markers.push(token_marker);
                                        full_walk_maybe_commit_deferred_positive(
                                            vocab,
                                            total_original_tokens,
                                            &mut deferred_output,
                                            &mut positive_rebuild,
                                            deferred_negative_mutations,
                                            &mut deferred_allowed_markers,
                                            &mut deferred_rejected_markers,
                                            &mut deferred_dead_subtrees,
                                            buf,
                                        );
                                    }
                                } else if positive_rebuild {
                                    if allowed {
                                        mark_dynamic_token_marker(vocab, token_marker, buf);
                                    }
                                } else if !allowed {
                                    clear_dynamic_token_marker(vocab, token_marker, buf);
                                }
                            }
                        }
                        identity_subtrees_skipped += 1;
                        identity_subtree_ops_skipped += saved_ops;
                        identity_subtree_tokens += token_end - token_marker_index;
                        token_marker_index = token_end;
                        remaining_ops = walk_ops[subtree_end as usize..].iter();
                        // The next preorder edge restores its unchanged parent
                        // from an ancestor slot, so no skipped DFS frame is read.
                        continue 'walk;
                    }
                }
            }
            if op.child_is_token() {
                if profile_walk && !parser_effect_seen {
                    profile_pre_effect_token_endpoints += 1;
                }
                let token_marker = unsafe { *token_markers.get_unchecked(token_marker_index) };
                token_marker_index += 1;
                let allowed = if scalar_lexer == FULL_WALK_LEXER_DEAD {
                    false
                } else if scalar_lexer < FULL_WALK_LEXER_TWO_DISTINCT {
                    scalar_endpoint_known_allowed
                        || parser_cache.token_boundary_allowed_raw(
                            state.constraint,
                            tokenizer,
                            &transitions,
                            initial_lexer_state,
                            scalar_lexer,
                            scalar_parser,
                        )
                } else if scalar_lexer == FULL_WALK_LEXER_TWO_DISTINCT || scalar_lexer == FULL_WALK_LEXER_TWO {
                    parser_cache.token_boundary_allowed_raw(
                        state.constraint,
                        tokenizer,
                        &transitions,
                        initial_lexer_state,
                        current_two.0.0,
                        current_two.0.1,
                    ) || parser_cache.token_boundary_allowed_raw(
                        state.constraint,
                        tokenizer,
                        &transitions,
                        initial_lexer_state,
                        current_two.1.0,
                        current_two.1.1,
                    )
                } else if scalar_lexer == FULL_WALK_LEXER_GUARDED_PAIR {
                    // The pending branch is at the reset lexer state, so a
                    // model token may end here using the most recent accepted
                    // terminal without consulting parser/lexer liveness again.
                    true
                } else if scalar_lexer == FULL_WALK_LEXER_MULTI {
                    match many_transition_memo.view(&current_many) {
                        FullWalkManyState::Branches(branches) => branches.iter().any(|branch| {
                            parser_cache.token_boundary_allowed(
                                state.constraint,
                                tokenizer,
                                &transitions,
                                initial_lexer_state,
                                branch,
                            )
                        }),
                        FullWalkManyState::ThreeSameParser {
                            lexers,
                            parser_node,
                        } => parser_cache.token_boundary_allowed_raw(
                            state.constraint,
                            tokenizer,
                            &transitions,
                            initial_lexer_state,
                            lexers.0,
                            *parser_node,
                        ) || parser_cache.token_boundary_allowed_raw(
                            state.constraint,
                            tokenizer,
                            &transitions,
                            initial_lexer_state,
                            lexers.1,
                            *parser_node,
                        ) || parser_cache.token_boundary_allowed_raw(
                            state.constraint,
                            tokenizer,
                            &transitions,
                            initial_lexer_state,
                            lexers.2,
                            *parser_node,
                        ),
                    }
                } else {
                    false
                };
                if deferred_output {
                    let mutations = dynamic_token_marker_original_count(vocab, token_marker);
                    let marker_work = dynamic_token_marker_materialization_cost(vocab, token_marker);
                    if allowed {
                        deferred_positive_mutations =
                            deferred_positive_mutations.saturating_add(mutations);
                        deferred_positive_work = deferred_positive_work.saturating_add(marker_work);
                        deferred_allowed_markers.push(token_marker);
                    } else {
                        deferred_negative_mutations =
                            deferred_negative_mutations.saturating_add(mutations);
                        deferred_negative_work = deferred_negative_work.saturating_add(marker_work);
                        deferred_rejected_markers.push(token_marker);
                        full_walk_maybe_commit_deferred_positive(
                            vocab,
                            total_original_tokens,
                            &mut deferred_output,
                            &mut positive_rebuild,
                            deferred_negative_mutations,
                            &mut deferred_allowed_markers,
                            &mut deferred_rejected_markers,
                            &mut deferred_dead_subtrees,
                            buf,
                        );
                    }
                } else if positive_rebuild {
                    if allowed {
                        mark_dynamic_token_marker(vocab, token_marker, buf);
                    }
                } else if !allowed {
                    clear_dynamic_token_marker(vocab, token_marker, buf);
                }
            }

            unsafe {
                *stack_lexer.get_unchecked_mut(parent_depth + 1) = scalar_lexer;
                if profile_walk {
                    *stack_parser_effect_seen.get_unchecked_mut(parent_depth + 1) =
                        parser_effect_seen;
                }
                if scalar_lexer == FULL_WALK_LEXER_DEAD {
                } else if scalar_lexer < FULL_WALK_LEXER_TWO_DISTINCT {
                    *stack_parser.get_unchecked_mut(parent_depth + 1) = scalar_parser;
                } else if scalar_lexer == FULL_WALK_LEXER_TWO_DISTINCT || scalar_lexer == FULL_WALK_LEXER_TWO {
                    *stack_two.get_unchecked_mut(parent_depth + 1) = current_two;
                } else if scalar_lexer == FULL_WALK_LEXER_GUARDED_PAIR {
                    *stack_two.get_unchecked_mut(parent_depth + 1) = current_guarded_pair.pack();
                } else if scalar_lexer == FULL_WALK_LEXER_MULTI {
                    stack_many.save(parent_depth + 1, &current_many);
                }
            }
        }
    }

    // Diagnostics must not be included in the measured traversal interval.
    let walk_elapsed = walk_started.map(|start| start.elapsed());
    if profile_kernel {
        eprintln!("[glrmask/profile][live_branch_witness] generation={} subtrees={} rows={}", state.generation, live_witness_subtrees, witness_proofs.rows.len());
        eprintln!("[glrmask/profile][full_walk_acceleration] generation={} enabled={} memo_states={} memo_lookups={} memo_hits={} outer_hits={} identity_subtrees={} identity_ops={} identity_tokens={} canonical_hits={}",
            state.generation, accelerated, many_transition_memo.states.len(),
            many_transition_memo.lookups, many_transition_memo.hits,
            guarded_outer_cache_hits, identity_subtrees_skipped,
            identity_subtree_ops_skipped, identity_subtree_tokens,
            parser_cache.canonical.as_ref().map_or(0, |c| c.hits));
    }
    if let (Some(kernel_started), Some(walk_started), Some(walk_elapsed)) =
        (kernel_started, walk_started, walk_elapsed)
    {
        eprintln!(
            "[glrmask/profile][dynamic_kernel_phases] generation={} setup_us={:.1} walk_us={:.1}",
            state.generation,
            walk_started.duration_since(kernel_started).as_secs_f64() * 1e6,
            walk_elapsed.as_secs_f64() * 1e6,
        );
    }

    if deferred_output {
        if profile_kernel {
            eprintln!(
                "[glrmask/profile][dynamic_output_deferred] generation={} allowed_markers={} rejected_markers={} dead_subtrees={} positive_mutations={} negative_mutations={} positive_work={} negative_work={}",
                state.generation,
                deferred_allowed_markers.len(),
                deferred_rejected_markers.len(),
                deferred_dead_subtrees.len(),
                deferred_positive_mutations,
                deferred_negative_mutations,
                deferred_positive_work,
                deferred_negative_work,
            );
        }
        let choose_positive = if quotient_adaptive_polarity {
            deferred_positive_work <= deferred_negative_work
        } else {
            deferred_positive_mutations <= deferred_negative_mutations
        };
        if choose_positive {
            buf.fill(0);
            for marker in deferred_allowed_markers {
                mark_dynamic_token_marker(vocab, marker, buf);
            }
        } else {
            let all_words = vocab.all_original_token_words();
            let copy_len = buf.len().min(all_words.len());
            buf[..copy_len].copy_from_slice(&all_words[..copy_len]);
            if copy_len < buf.len() {
                buf[copy_len..].fill(0);
            }
            for child in deferred_dead_subtrees {
                for &token_id in vocab.subtree_original_tokens_for(trie, child) {
                    clear_mask_bit_known_in_range(buf, token_id);
                }
            }
            for marker in deferred_rejected_markers {
                clear_dynamic_token_marker(vocab, marker, buf);
            }
        }
    }
    if profile_kernel && ignored_output.is_some() {
        eprintln!("[glrmask/profile][root_output_scope] generation={} visited={} skips={} saved_ops={}",
            state.generation, output_scope_visited, output_scope_skips,
            output_scope_saved_ops);
    }
    if profile_walk {
        eprintln!(
                "[glrmask/profile][full_walk_volume] generation={} ops={} byte_ops={} token_endpoints={} finalizing_bytes={} direct_finalizers={} pre_effect_byte_ops={} pre_effect_token_endpoints={} first_effect_frontiers={} scalar_lane_bytes={} two_distinct_lane_bytes={} two_same_lane_bytes={} multi_lane_bytes={} multi_three_same_bytes={} multi_branches_2={} multi_branches_3={} multi_branches_4plus={} multi_two_same_parser={} multi_two_both_passed={} multi_two_one_pending={} multi_two_both_pending={} multi_two_pending_mem1={} multi_two_pending_mem2={} multi_two_pending_mem3plus={} multi_two_guard_eq_passed_lexer={} multi_two_pending_lexer_is_initial={} boundary_calls={} boundary_misses={} parser_advance_calls={} parser_advance_misses={} inadmissible_finalizers={} parser_nodes={} dead_tokens_cleared={} total_ops={} total_tokens={}",
            state.generation,
            profile_ops,
            profile_bytes,
            profile_token_endpoints,
            profile_finalizing_bytes,
            profile_direct_finalizers,
            profile_pre_effect_byte_ops,
            profile_pre_effect_token_endpoints,
            profile_first_effect_frontiers,
            profile_scalar_lane_bytes,
            profile_two_distinct_lane_bytes,
            profile_two_same_lane_bytes,
            profile_multi_lane_bytes,
            profile_multi_three_same_bytes,
            profile_multi_branches_2,
            profile_multi_branches_3,
            profile_multi_branches_4plus,
            profile_multi_two_same_parser,
            profile_multi_two_both_passed,
            profile_multi_two_one_pending,
            profile_multi_two_both_pending,
            profile_multi_two_pending_mem1,
            profile_multi_two_pending_mem2,
            profile_multi_two_pending_mem3plus,
            profile_multi_two_guard_eq_passed_lexer,
            profile_multi_two_pending_lexer_is_initial,
            parser_cache.profile_boundary_calls,
            parser_cache.profile_boundary_misses,
            parser_cache.profile_advance_calls,
            parser_cache.profile_advance_misses,
            parser_cache.profile_inadmissible_finalizers,
            parser_cache.nodes.len(),
            profile_dead_tokens_cleared,
            walk_ops.len(),
            token_markers.len(),
        );
    }
    // Ordinary vocabulary bytes and exact special-token-ID paths are a union.
    // The strict walk above computes the byte-language contribution for every
    // model token; the existing special-token routine then ORs in token-ID-only
    // paths. This also handles a token ID that is valid through both routes.
    update_special_token_mask(state, buf);
    state.clear_late_grammar_placeholder_mask(buf);
    Ok(true)
}

#[cfg(test)]
mod full_walk_acceleration_tests {
    use super::*;

    #[test]
    fn live_branch_witness_survives_finalizers_and_changing_alternatives() {
        use crate::{DynamicConstraint, Grammar, Vocab};
        let vocab = Vocab::new(["a", "b", "aa", "ab", " "].into_iter().enumerate()
            .map(|(id, text)| (id as u32, text.as_bytes().to_vec())).collect());
        let dynamic = DynamicConstraint::compile(Grammar::glrm(
            "start s; ignore WS; t WS ::= ' '+; t A ::= 'a'+; t B ::= 'b'; nt s ::= A B;"), &vocab).unwrap();
        let constraint = &dynamic.inner;
        let tokenizer = &constraint.tokenizer;
        let states = tokenizer.num_states() as usize;
        let mut rows = vec![u32::MAX; states * 256];
        let finals = (0..states).map(|state| match tokenizer.matched_terminals_slice(state as u32) {
            [] => u32::MAX,
            [terminal] => *terminal,
            _ => u32::MAX - 1,
        }).collect::<Vec<_>>();
        for state in 0..states {
            for (byte, target) in tokenizer.transitions_from(state as u32) {
                rows[state * 256 + byte as usize] = target
                    | if finals[target as usize] != u32::MAX { 0x8000_0000 } else { 0 };
            }
        }
        let transitions = FullWalkFlat32 { transitions: &rows };
        let stacks = constraint.start().state.entries[0].1.apply(|_| ());
        let (mut cache, _) = FullWalkParserCache::from_roots(&DynamicBranches::new(), states, false);
        let parser = cache.push_parser_stacks(stacks);
        let mut proofs = FullWalkIdentityProofCache::default();
        let alphabet = U8Set::single(b'a').to_words();
        let lexer = (0..states as u32).find(|&lexer| {
            let cell = transitions.cell(lexer, b'a');
            !FullWalkFlat32::cell_is_dead(cell)
                && FullWalkFlat32::cell_target(cell) == lexer
                && FullWalkFlat32::cell_has_finalizer(cell)
                && cache.physical_token_boundary_allowed(constraint, tokenizer, &transitions, parser, lexer)
        }).expect("fixture must exercise a live finalizing self-edge");
        let witness = FullWalkBranch { lexer_state: lexer, parser_node: parser,
            prune_guard: FullWalkPruneGuard::Passed };
        assert!(full_walk_live_branch_witness(&witness, alphabet, &mut proofs,
            &mut cache, constraint, tokenizer, &transitions));
        let mut current = smallvec::smallvec![witness.clone(), FullWalkBranch {
            lexer_state: tokenizer.start_state(), parser_node: parser,
            prune_guard: FullWalkPruneGuard::Passed,
        }];
        let original = current.clone();
        for _ in 0..6 {
            current = full_walk_step_many(&current, b'a', tokenizer.start_state(), &finals,
                tokenizer, &transitions, &mut cache, constraint);
            assert!(current.contains(&witness), "reference step must retain the witness");
            assert!(current.iter().any(|branch| cache.token_boundary_allowed(
                constraint, tokenizer, &transitions, tokenizer.start_state(), branch)));
        }
        assert!(current != original, "proof must work without whole-frontier identity");
        let mut guarded = witness.clone();
        guarded.prune_guard = FullWalkPruneGuard::Pending(smallvec::smallvec![(lexer, finals[lexer as usize])]);
        assert!(!full_walk_live_branch_witness(&guarded, alphabet, &mut proofs,
            &mut cache, constraint, tokenizer, &transitions));
        assert!(!full_walk_live_branch_witness(&witness, U8Set::single(b'b').to_words(), &mut proofs,
            &mut cache, constraint, tokenizer, &transitions));

        // Proof-cache exhaustion declines without constructing more rows or
        // weakening the physical parser-liveness condition.
        let mut exhausted = FullWalkIdentityProofCache::default();
        for id in 0..64 {
            exhausted.rows.insert(u32::MAX - id, None);
        }
        assert!(!full_walk_live_branch_witness(&witness, alphabet, &mut exhausted,
            &mut cache, constraint, tokenizer, &transitions));
        assert_eq!(exhausted.rows.len(), 64);

        let many = FullWalkManyState::Branches(smallvec::smallvec![guarded, witness.clone()]);
        assert!(full_walk_many_live_witness(&many, alphabet, &mut proofs,
            &mut cache, constraint, tokenizer, &transitions));
        let over_budget = FullWalkManyState::Branches(
            (0..9).map(|_| witness.clone()).collect());
        let mut unused = FullWalkIdentityProofCache::default();
        assert!(!full_walk_many_live_witness(&over_budget, alphabet, &mut unused,
            &mut cache, constraint, tokenizer, &transitions));
        assert!(unused.rows.is_empty(), "branch-work rejection must precede row proofs");

        // The whole union changes on 'a': its reset member advances into A.
        // The continuing A member nevertheless witnesses every a+ endpoint.
        let union_storage = std::sync::Mutex::new(DynamicLazyUnionCache::default());
        let overflowed = std::cell::Cell::new(false);
        let (union, union_root) = FullWalkLazyUnion::new(
            tokenizer, None, Some(&rows), union_storage.lock().unwrap(),
            &[lexer, tokenizer.start_state()], &overflowed, false,
        ).unwrap();
        let (mut union_parser_cache, _) = FullWalkParserCache::from_roots(
            &DynamicBranches::new(), union.state_count(tokenizer), false);
        let union_parser = union_parser_cache.push_parser_stacks(cache.nodes[parser as usize].gss.clone());
        let union_branch = FullWalkBranch {
            lexer_state: union_root, parser_node: union_parser,
            prune_guard: FullWalkPruneGuard::Passed,
        };
        let mut union_proofs = FullWalkIdentityProofCache::default();
        assert!(!full_walk_live_branch_witness(&union_branch, alphabet, &mut union_proofs,
            &mut union_parser_cache, constraint, tokenizer, &union));
        assert!(full_walk_live_component_witness(union_root, union_parser, alphabet,
            &mut union_proofs, &mut union_parser_cache, constraint, tokenizer, &union));
        let mut union_current = smallvec::smallvec![union_branch];
        for _ in 0..6 {
            union_current = full_walk_step_many(&union_current, b'a', tokenizer.start_state(),
                &finals, tokenizer, &union, &mut union_parser_cache, constraint);
            assert!(union_current.iter().any(|branch| branch.prune_guard.is_passed()
                && branch.parser_node == union_parser
                && union.continuation_witness_components(branch.lexer_state).contains(&lexer)));
            assert!(union_current.iter().any(|branch| union_parser_cache.token_boundary_allowed(
                constraint, tokenizer, &union, tokenizer.start_state(), branch)));
        }
        assert!(!overflowed.get());
        let guarded_union = FullWalkManyState::Branches(smallvec::smallvec![FullWalkBranch {
            lexer_state: union_root, parser_node: union_parser,
            prune_guard: FullWalkPruneGuard::Pending(smallvec::smallvec![(lexer, finals[lexer as usize])]),
        }]);
        assert!(!full_walk_many_component_witness(&guarded_union, alphabet, &mut union_proofs,
            &mut union_parser_cache, constraint, tokenizer, &union));

        let dead_parser = cache.push_parser_stacks(ParserStacks::empty());
        let dead = FullWalkBranch { parser_node: dead_parser, ..witness };
        assert!(!full_walk_live_branch_witness(&dead, alphabet, &mut proofs,
            &mut cache, constraint, tokenizer, &transitions));
        assert!(full_walk_frontier_live_witness(
            FullWalkWitnessFrontier::One(lexer, parser), alphabet, &mut proofs,
            &mut cache, constraint, tokenizer, &transitions));
        assert!(full_walk_frontier_live_witness(
            FullWalkWitnessFrontier::Two([(lexer, dead_parser), (lexer, parser)]),
            alphabet, &mut proofs, &mut cache, constraint, tokenizer, &transitions));
        assert!(!full_walk_frontier_live_witness(
            FullWalkWitnessFrontier::Many(&guarded_union), alphabet, &mut union_proofs,
            &mut union_parser_cache, constraint, tokenizer, &union));
    }

    #[test]
    fn row_liveness_bounds_are_exact_for_all_small_terminal_sets() {
        let mut proofs = 0;
        let mut unresolved = 0;
        for upper in 0u32..16 {
            for lower in 0u32..16 {
                if lower & !upper != 0 { continue; }
                for admitted in 0u32..16 {
                    if lower & !admitted != 0 || admitted & !upper != 0 { continue; }
                    for futures in 0u32..16 {
                        match full_walk_row_liveness_bound(lower & futures != 0, || upper & futures != 0) {
                            Some(value) => { assert_eq!(value, admitted & futures != 0); proofs += 1; }
                            None => { assert_eq!(lower & futures, 0); assert_ne!(upper & futures, 0); unresolved += 1; }
                        }
                    }
                }
            }
        }
        assert!(proofs > 1000 && unresolved > 100);
        assert_eq!(full_walk_row_liveness_bound(true, || panic!("proven positive must not query upper bound")), Some(true));
    }

    #[test]
    fn row_liveness_bounds_match_compiled_and_loaded_parser_frontiers() {
        use crate::{DynamicConstraint, Grammar, Vocab};
        let vocab = Vocab::new([
            "a", "b", "c", "aa", "ab", "ba", "bb", " ", "acb",
        ].into_iter().enumerate()
            .map(|(id, text)| (id as u32, text.as_bytes().to_vec())).collect());
        let grammars = [
            "start s; t A ::= 'a'; t B ::= 'b'; t C ::= 'c'; nt s ::= A s B | C;",
            "start s; t A ::= 'a'; t B ::= 'b'; t C ::= 'c'; nt s ::= A s B | A A | A B | C;",
            "start s; ignore WS; t WS ::= ' '+; t A ::= 'a'+; t B ::= 'ab'; t C ::= 'b'+; nt s ::= A s C | B | A C;",
        ];
        let mut positive = 0;
        let mut negative = 0;
        let mut compared = 0;
        for grammar in grammars {
            let compiled = DynamicConstraint::compile(Grammar::glrm(grammar), &vocab).unwrap();
            let loaded = DynamicConstraint::load_with_vocab(&compiled.save(), &vocab).unwrap();
            for dynamic in [&compiled, &loaded] {
                let constraint = &dynamic.inner;
                // These predicates read terminal metadata, never byte cells.
                let transitions = FullWalkFlat32 { transitions: &[] };
                let mut pending = constraint.start().state.entries.iter()
                    .map(|(_, gss)| gss.apply(|_| ())).collect::<Vec<_>>();
                let mut seen = std::collections::HashSet::new();
                for _ in 0..5 {
                    let mut next = Vec::new();
                    for stacks in pending {
                        let mut key = stacks.to_stacks(4096).expect("small grammar path bound");
                        key.sort();
                        if !seen.insert(key) { continue; }
                        let (mut cache, _) = FullWalkParserCache::from_roots(
                            &DynamicBranches::new(), constraint.tokenizer.num_states() as usize, false,
                        );
                        cache.row_liveness_enabled = true;
                        let node = cache.push_parser_stacks(stacks.clone());
                        let admitted = exact_parser_admission_for_stacks(constraint, &stacks);
                        for lexer in 0..constraint.tokenizer.num_states() {
                            let exact = transitions.future_intersects(&constraint.tokenizer, lexer, &admitted);
                            if let Some(answer) = cache.row_future_allowed(
                                constraint, &constraint.tokenizer, &transitions, node, lexer,
                            ) {
                                assert_eq!(answer, exact, "grammar={grammar} lexer={lexer}");
                                positive += usize::from(answer);
                                negative += usize::from(!answer);
                            }
                            compared += 1;
                        }
                        // Independently check the complete cached predicate,
                        // including ignore, positive/negative hits, and disabled fallback.
                        for lexer in 0..constraint.tokenizer.num_states() {
                            let exact = transitions.future_intersects(&constraint.tokenizer, lexer, &admitted)
                                || constraint.ignore_terminal.is_some_and(|terminal|
                                    transitions.future_contains(&constraint.tokenizer, lexer, terminal));
                            assert_eq!(cache.physical_token_boundary_allowed(
                                constraint, &constraint.tokenizer, &transitions, node, lexer), exact);
                            assert_eq!(cache.physical_token_boundary_allowed(
                                constraint, &constraint.tokenizer, &transitions, node, lexer), exact);
                        }
                        cache.row_liveness_enabled = false;
                        assert!(cache.row_future_allowed(constraint, &constraint.tokenizer,
                            &transitions, node, 0).is_none());
                        for terminal in 0..constraint.table.num_terminals {
                            if let Some(child) = parser_child(constraint, &stacks, terminal) {
                                next.push(child);
                            }
                        }
                    }
                    pending = next;
                }
            }
        }
        assert!(compared > 100, "test must cover actual parser/lexer combinations");
        assert!(positive > 0 && negative > 0, "both proof directions must execute");
    }

    #[test]
    fn identity_output_noop_preserves_deferred_and_both_polarities() {
        for (deferred, positive, allowed, expected) in [
            (false, false, false, true),
            (false, false, true, false),
            (false, true, false, false),
            (false, true, true, true),
            (true, false, false, true),
            (true, false, true, true),
            (true, true, false, true),
            (true, true, true, true),
        ] {
            assert_eq!(identity_subtree_requires_output(deferred, positive, allowed), expected);
        }
    }

    #[test]
    fn cached_identity_policy_changes_only_at_the_exact_node_threshold() {
        let (mut cache, _) = FullWalkParserCache::from_roots(&DynamicBranches::new(), 0, false);
        cache.canonicalize = true;
        assert!(!cache.identity_proofs_enabled);
        for count in 1..=48 {
            cache.push_parser_stacks(ParserStacks::from_single_stack(vec![count], ()));
            assert_eq!(cache.identity_proofs_enabled,
                full_walk_identity_context_profitable(0, count as usize));
        }

        let (mut disabled, _) = FullWalkParserCache::from_roots(&DynamicBranches::new(), 0, false);
        disabled.canonicalize = false;
        for value in 0..40 {
            disabled.push_parser_stacks(ParserStacks::from_single_stack(vec![value], ()));
        }
        assert!(!disabled.identity_proofs_enabled);
    }

    #[test]
    fn many_frontier_stack_is_lazy_and_restores_exact_depths() {
        let mut stack = FullWalkManyStack::default();
        assert_eq!(stack.slots.capacity(), 0);
        let root = FullWalkManyCursor::Direct(guarded_state(7, 2));
        let child = FullWalkManyCursor::Direct(guarded_state(8, 3));
        stack.save(0, &root);
        stack.save(17, &child);
        let mut memo = FullWalkManyTransitionMemo::new(4);
        assert!(memo.view(stack.restore(0)) == memo.view(&root));
        assert!(memo.view(stack.restore(17)) == memo.view(&child));
        assert_eq!(stack.slots.len(), 18);
        assert!(stack.slots[1..17].iter().all(Option::is_none));

        let cached = memo.hold(guarded_state(19, 5));
        stack.save(17, &cached);
        assert_eq!(stack.restore(17).id(), cached.id());
        assert!(memo.view(stack.restore(17)) == &guarded_state(19, 5));
        assert!(memo.view(stack.restore(0)) == memo.view(&root));
        stack.save(17, &child);
        assert!(stack.restore(17).id().is_none());
        assert!(memo.view(stack.restore(17)) == memo.view(&child));

        stack.save(255, &cached);
        assert_eq!(stack.slots.len(), 256);
        assert_eq!(stack.restore(255).id(), cached.id());
        assert!(memo.view(stack.restore(17)) == memo.view(&child));
    }

    #[test]
    fn cached_frontier_copies_and_direct_replacements_preserve_correlations() {
        let mut memo = FullWalkManyTransitionMemo::new(4);
        let first = memo.hold(guarded_state(7, 2));
        let second = memo.hold(guarded_state(8, 3));
        let direct = FullWalkManyCursor::Direct(guarded_state(9, 4));
        let mut cursor = direct.clone();
        for source in [&first, &second, &direct, &direct, &first] {
            cursor.clone_from(source);
            assert!(memo.view(&cursor) == memo.view(source));
        }
        cursor.set_cached(second.id().unwrap());
        assert!(memo.view(&cursor) == memo.view(&second));
        cursor.clone_from(&direct);
        cursor.set_cached(first.id().unwrap());
        assert!(memo.view(&cursor) == memo.view(&first));
        assert!(memo.view(&direct) == &guarded_state(9, 4));
    }

    #[test]
    fn contiguous_frontier_rows_preserve_transitions_across_growth() {
        let mut memo = FullWalkManyTransitionMemo::new(256);
        let first = memo.hold(guarded_state(7, 2));
        memo.remember_transition(first.id(), 0, &first);
        memo.remember_transition(first.id(), 255, &first);
        for value in 8..263 {
            let next = memo.hold(guarded_state(value, 2));
            memo.remember_transition(next.id(), 127, &first);
        }
        assert_eq!(memo.rows.len(), 256);
        assert_eq!(memo.cached_transition(&first, 0), first.id());
        assert_eq!(memo.cached_transition(&first, 255), first.id());
        assert!(memo.cached_transition(&first, 127).is_none());
        for id in 1..256 {
            let cursor = FullWalkManyCursor::Cached(id);
            assert_eq!(memo.cached_transition(&cursor, 127), first.id());
            assert!(memo.cached_transition(&cursor, 255).is_none());
        }
    }

    #[test]
    fn many_transition_memo_bounds_allocations_and_keeps_exact_direct_states() {
        let mut disabled = FullWalkManyTransitionMemo::new(0);
        let state = guarded_state(7, 2);
        let cursor = disabled.hold(state.clone());
        assert!(cursor.id().is_none());
        assert!(disabled.view(&cursor) == &state);
        assert!(disabled.states.is_empty());
        assert!(disabled.rows.is_empty());

        let mut bounded = FullWalkManyTransitionMemo::new(usize::MAX);
        assert_eq!(bounded.limit, FullWalkManyTransitionMemo::MAX_STATES);
        for parser in 0..FullWalkManyTransitionMemo::MAX_STATES + 3 {
            let state = guarded_state(parser as u32, 2);
            let cursor = bounded.hold(state.clone());
            assert_eq!(cursor.id().is_some(), parser < FullWalkManyTransitionMemo::MAX_STATES);
            assert!(bounded.view(&cursor) == &state);
        }
        assert_eq!(bounded.rows.len(), FullWalkManyTransitionMemo::MAX_STATES);
        assert_eq!(bounded.states.len(), bounded.rows.len());
        assert_eq!(bounded.identity_bytes.len(), bounded.rows.len());
        assert!(bounded.hold(guarded_state(7, 2)).id().is_some(),
            "existing exact entries remain reusable after capacity is exhausted");
    }

    #[test]
    fn root_output_scope_marker_checks_all_aliases_across_output_words() {
        let direct = (1u64 << 32) | 0b101;
        assert!(full_walk_marker_fully_ignored(direct, &[0, 0b111], |_| panic!("direct")));
        assert!(!full_walk_marker_fully_ignored(direct, &[0, 0b001], |_| panic!("direct")));
        assert!(!full_walk_marker_fully_ignored(direct, &[u32::MAX], |_| panic!("direct")));
        let aliased = DYNAMIC_TOKEN_MARKER_FALLBACK | 4;
        for mask in [vec![1, 0], vec![0, 1], vec![1, 1]] {
            let covered = full_walk_marker_fully_ignored(aliased, &mask, |id| {
                assert_eq!(id, 3);
                [0_u32, 32].iter().all(|id| mask[*id as usize / 32] & (1 << (*id % 32)) != 0)
            });
            assert_eq!(covered, mask == [1, 1]);
        }
        assert!(!full_walk_marker_fully_ignored(DYNAMIC_TOKEN_MARKER_FALLBACK, &[], |_| panic!("invalid")));
        assert!(!full_walk_marker_fully_ignored(u64::MAX, &[], |_| panic!("invalid")));
    }

    #[test]
    fn identity_context_policy_is_monotone_and_only_changes_admission() {
        for roots in 0..10 {
            let mut admitted = false;
            for parser_nodes in 0..256 {
                let now = full_walk_identity_context_profitable(roots, parser_nodes);
                assert_eq!(now, roots >= 2 || parser_nodes >= 32);
                assert!(!admitted || now);
                admitted |= now;
            }
        }
    }


    #[test]
    fn root_output_scope_cursor_matches_all_intervals_and_scans_each_marker_once() {
        let markers: Vec<u64> = (0..8).collect();
        for ignored in 0_u32..256 {
            let mut cursor = FullWalkOutputScopeCursor::default();
            let mut visits = [0_usize; 8];
            for start in 0..=markers.len() {
                for end in (start..=markers.len()).rev() {
                    let expected = markers[start..end].iter().all(|&m| ignored & (1 << m) != 0);
                    let actual = cursor.covers(&markers, start, end, |m| {
                        visits[m as usize] += 1;
                        ignored & (1 << m) != 0
                    });
                    assert_eq!(actual, expected, "ignored={ignored} interval={start}..{end}");
                }
            }
            assert!(visits.iter().all(|&n| n <= 1), "repeated scans: {visits:?}");
            // A caller restoring an earlier traversal must not reuse a later
            // ignored-prefix certificate. Test nonmonotone order too.
            for start in (0..=markers.len()).rev() {
                for end in start..=markers.len() {
                    let expected = markers[start..end].iter().all(|&m| ignored & (1 << m) != 0);
                    assert_eq!(cursor.covers(&markers, start, end, |m| ignored & (1 << m) != 0), expected);
                }
            }
            assert!(!cursor.covers(&markers, 3, 2, |_| panic!("invalid interval")));
            assert!(!cursor.covers(&markers, 0, 9, |_| panic!("invalid interval")));
        }
        let mut cursor = FullWalkOutputScopeCursor::default();
        assert!(cursor.covers(&[], 0, 0, |_| panic!("empty range")));
    }

    #[test]
    fn root_output_scope_prefix_ranges_match_literal_marker_responsibility() {
        let markers = [0, 1, 2, 3, 4, 5, 6, 7];
        for ignored in 0..256u32 {
            let scope = FullWalkOutputScope::new(&markers, |marker| ignored & (1 << marker) != 0).unwrap();
            for start in 0..=8 {
                for end in start..=8 {
                    assert_eq!(scope.covers(start, end),
                        (start..end).all(|marker| ignored & (1 << marker) != 0));
                }
            }
            assert!(!scope.covers(4, 3));
            assert!(!scope.covers(0, 9));
            assert!(!scope.covers(usize::MAX, usize::MAX));
        }
        assert!(FullWalkOutputScope::new(&[], |_| false).unwrap().covers(0, 0));
    }

    #[test]
    fn root_output_scope_dont_care_algebra_preserves_guarded_union() {
        for accepted in 0..16u32 {
            for blocked in 0..16u32 {
                let ignored = accepted | blocked;
                for root in 0..16u32 {
                    for arbitrary in 0..16u32 {
                        let partial = (root & !ignored) | (arbitrary & ignored);
                        assert_eq!(accepted | (partial & !blocked), accepted | (root & !blocked));
                    }
                }
            }
        }
    }

    #[test]
    fn structural_identity_classes_separate_finalizers_and_never_include_dead_or_changed_targets() {
        let state = 37;
        let mut calls = 0;
        let classes = full_walk_identity_byte_classes(state, |byte| {
            calls += 1;
            match byte {
                0 | 255 => Some((state, false)),
                b'a'..=b'z' => Some((state, true)),
                b'A'..=b'Z' => Some((state + 1, false)),
                b'0'..=b'9' => Some((state + 1, true)),
                _ => None,
            }
        });
        assert_eq!(calls, 256);
        for byte in 0..=u8::MAX {
            let bit = 1u64 << (byte & 63);
            assert_eq!(classes.ordinary[byte as usize >> 6] & bit != 0,
                       byte == 0 || byte == 255);
            assert_eq!(classes.finalizing[byte as usize >> 6] & bit != 0,
                       byte.is_ascii_lowercase());
        }
        let shifted = full_walk_identity_byte_classes(state + 1, |byte| {
            Some((state, byte % 2 == 0))
        });
        assert_eq!(shifted.ordinary, [0; 4]);
        assert_eq!(shifted.finalizing, [0; 4]);
    }

    #[test]
    fn scalar_identity_structural_proof_is_scoped_to_the_complete_pair() {
        let mut cache = FullWalkScalarIdentity::default();
        cache.remember((7, 11), (7, 11), b'a');
        cache.proof_checked = true;
        cache.bytes = [u64::MAX; 4];
        cache.remember((7, 11), (7, 11), b'b');
        assert!(cache.proof_checked);
        assert!(cache.alphabet((7, 12)).is_none());
        cache.remember((7, 12), (7, 12), b'a');
        assert!(!cache.proof_checked);
        assert_eq!(cache.bytes.iter().map(|word| word.count_ones()).sum::<u32>(), 1);
        cache.proof_checked = true;
        cache.remember((8, 12), (8, 12), 255);
        assert!(!cache.proof_checked);
        assert_eq!(cache.alphabet((8, 12)), Some(&[0, 0, 0, 1u64 << 63]));
    }

    #[test]
    fn identity_classes_of_union_require_every_member_and_keep_finalizer_or() {
        let mut left = full_walk_identity_byte_classes(1, |byte| match byte {
            0 | b'a' | b'b' => Some((1, false)),
            b'c' | 255 => Some((1, true)),
            _ => None,
        });
        let right = full_walk_identity_byte_classes(2, |byte| match byte {
            b'b' | b'c' => Some((2, false)),
            b'a' | 255 => Some((2, true)),
            _ => None,
        });
        left.intersect_union_member(right);
        assert_eq!(left.ordinary, U8Set::single(b'b').to_words());
        assert_eq!(left.finalizing, U8Set::from_bytes(&[b'a', b'c', 255]).to_words());
        left.intersect_union_member(FullWalkIdentityByteClasses::default());
        assert_eq!(left.ordinary, [0; 4]);
        assert_eq!(left.finalizing, [0; 4]);
    }

    #[test]
    fn structural_identity_proof_cache_is_bounded_and_keeps_exact_row_keys() {
        let tokenizer = crate::automata::lexer::tokenizer::arbitrary_flat32_test_tokenizer();
        let mut rows = vec![u16::MAX; 65 * 256];
        for state in 0..65 {
            rows[state * 256 + b'a' as usize] = state as u16;
        }
        let table = FullWalkFlat16 { transitions: &rows };
        let mut cache = FullWalkIdentityProofCache::default();
        for state in 0..64 {
            let classes = cache.get(&table, &tokenizer, state).unwrap();
            assert_eq!(classes.ordinary, U8Set::single(b'a').to_words());
            assert_eq!(classes.finalizing, [0; 4]);
        }
        assert_eq!(cache.rows.len(), 64);
        assert!(cache.get(&table, &tokenizer, 64).is_none());
        assert!(cache.get(&table, &tokenizer, 0).is_some());
        assert_eq!(cache.rows.len(), 64);
    }

    #[test]
    fn scalar_identity_alphabet_requires_both_coordinates_and_resets_on_replacement() {
        let mut cache = FullWalkScalarIdentity::default();
        let source = (37, 91);
        assert!(cache.alphabet(source).is_none());
        cache.remember(source, (38, 91), b'a');
        cache.remember(source, (37, 92), b'b');
        assert!(cache.alphabet(source).is_none());
        cache.remember(source, source, 0);
        cache.remember(source, source, 255);
        assert_eq!(cache.alphabet(source), Some(&[1, 0, 0, 1u64 << 63]));
        assert!(cache.alphabet((38, 91)).is_none());
        assert!(cache.alphabet((37, 92)).is_none());
        let other = (0x8000_0100, 92);
        cache.remember(other, other, 64);
        assert!(cache.alphabet(source).is_none());
        assert_eq!(cache.alphabet(other), Some(&[0, 1, 0, 0]));
    }

    #[test]
    fn identity_subtree_alphabet_requires_every_byte_and_exact_memo_self_edges() {
        let mut identity = [0_u64; 4];
        identity[b'a' as usize >> 6] |= 1 << (b'a' & 63);
        identity[3] |= 1 << 63;
        assert!(identity_alphabet_covers_subtree(&identity, identity));
        assert!(identity_alphabet_covers_subtree(&identity, [0; 4]));
        for byte in 0_u16..=255 {
            let mut alphabet = [0_u64; 4];
            alphabet[byte as usize >> 6] = 1 << (byte & 63);
            assert_eq!(identity_alphabet_covers_subtree(&identity, alphabet),
                byte == u16::from(b'a') || byte == 255);
        }
        let mut memo = FullWalkManyTransitionMemo::new(4);
        let source = memo.hold(guarded_state(7, 2));
        let other = memo.hold(guarded_state(8, 2));
        memo.remember_transition(source.id(), b'a', &source);
        memo.remember_transition(source.id(), b'b', &other);
        let known = memo.identity_alphabet(&source).unwrap();
        assert_eq!(known[b'a' as usize >> 6] & (1 << (b'a' & 63)), 1 << (b'a' & 63));
        assert_eq!(known[b'b' as usize >> 6] & (1 << (b'b' & 63)), 0);
        assert_eq!(memo.identity_alphabet(&other).unwrap(), &[0; 4]);
    }

    #[test]
    fn guarded_identity_covers_fast_returns_without_aliasing_other_coordinates() {
        let pair = FullWalkGuardedPair {
            continuing_lexer: 3, continuing_parser: 4, pending_parser: 5, guard_terminal: 6,
        };
        let mut cached = None;
        let mut bytes = [0_u64; 4];
        assert!(remember_guarded_identity(pair, pair, 255, &mut cached, &mut bytes));
        assert!(cached == Some(pair));
        assert_eq!(bytes, [0, 0, 0, 1u64 << 63]);
        for different in [
            FullWalkGuardedPair { continuing_lexer: 7, ..pair },
            FullWalkGuardedPair { continuing_parser: 7, ..pair },
            FullWalkGuardedPair { pending_parser: 7, ..pair },
            FullWalkGuardedPair { guard_terminal: 7, ..pair },
        ] {
            let old_bytes = bytes;
            assert!(!remember_guarded_identity(pair, different, 0, &mut cached, &mut bytes));
            assert!(cached == Some(pair));
            assert_eq!(bytes, old_bytes);
        }
        let other = FullWalkGuardedPair { pending_parser: 8, ..pair };
        assert!(remember_guarded_identity(other, other, 0, &mut cached, &mut bytes));
        assert!(cached == Some(other));
        assert_eq!(bytes, [1, 0, 0, 0]);
    }

    #[test]
    fn guarded_reentry_requires_the_complete_correlated_pair_shape() {
        let passed = FullWalkBranch {
            lexer_state: 9, parser_node: 17, prune_guard: FullWalkPruneGuard::Passed,
        };
        let pending = FullWalkBranch {
            lexer_state: 3, parser_node: 29,
            prune_guard: FullWalkPruneGuard::Pending(smallvec::smallvec![(9, 6)]),
        };
        let expected = FullWalkGuardedPair {
            continuing_lexer: 9, continuing_parser: 17, pending_parser: 29, guard_terminal: 6,
        };
        for branches in [smallvec::smallvec![passed.clone(), pending.clone()], smallvec::smallvec![pending.clone(), passed.clone()]] {
            assert!(full_walk_guarded_pair_from_branches(&branches, 3) == Some(expected));
        }
        for other in [
            FullWalkBranch { lexer_state: 4, ..pending.clone() },
            FullWalkBranch { prune_guard: FullWalkPruneGuard::Pending(smallvec::smallvec![(8, 6)]), ..pending.clone() },
            FullWalkBranch { prune_guard: FullWalkPruneGuard::Pending(smallvec::smallvec![(9, 6), (9, 7)]), ..pending.clone() },
        ] {
            assert!(full_walk_guarded_pair_from_branches(&smallvec::smallvec![passed.clone(), other], 3).is_none());
        }
        assert!(full_walk_guarded_pair_from_branches(&smallvec::smallvec![passed.clone(), pending, passed], 3).is_none());
    }


    #[test]
    fn many_transition_memo_preserves_complete_state_and_byte_keys() {
        let mut memo = FullWalkManyTransitionMemo::new(8);
        let state = guarded_state(7, 2);
        let source = memo.hold(state.clone());
        let duplicate = memo.hold(state.clone());
        let target = memo.hold(guarded_state(8, 2));
        let other_guard = memo.hold(guarded_state(7, 3));
        assert_eq!(source.id(), duplicate.id());
        assert_ne!(source.id(), target.id());
        assert_ne!(source.id(), other_guard.id());
        assert!(memo.cached_transition(&source, b'a').is_none());
        memo.remember_transition(source.id(), b'a', &target);
        let hit = memo.cached_transition(&duplicate, b'a').unwrap();
        assert!(memo.view(&FullWalkManyCursor::Cached(hit)) == &guarded_state(8, 2));
        assert!(memo.cached_transition(&source, b'b').is_none());
        assert!(memo.cached_transition(&other_guard, b'a').is_none());
        assert!(memo.cached_transition(&target, b'a').is_none());
    }

    #[test]
    fn many_transition_memo_limit_retains_exact_uncached_fallback() {
        let mut memo = FullWalkManyTransitionMemo::new(1);
        let source = memo.hold(guarded_state(7, 2));
        let uncached = memo.hold(guarded_state(8, 2));
        assert!(source.id().is_some());
        assert!(uncached.id().is_none());
        assert!(memo.view(&uncached) == &guarded_state(8, 2));
        memo.remember_transition(source.id(), 255, &uncached);
        assert!(memo.cached_transition(&source, 255).is_none());
        memo.remember_transition(source.id(), b'a', &source);
        let repeated = memo.hold(guarded_state(7, 2));
        assert_eq!(source.id(), repeated.id());
        assert_eq!(Some(memo.cached_transition(&repeated, b'a').unwrap()), source.id());
        assert_eq!(memo.states.len(), 1);
        let mut next_mask = FullWalkManyTransitionMemo::new(1);
        let fresh = next_mask.hold(guarded_state(7, 2));
        assert!(next_mask.cached_transition(&fresh, b'a').is_none());
    }

    #[test]
    fn many_transition_memo_checks_equality_after_hash_collisions() {
        use std::hash::{Hash, Hasher};
        let mut memo = FullWalkManyTransitionMemo::new(8);
        let source = memo.hold(guarded_state(7, 2));
        let other = guarded_state(8, 2);
        let mut hash = rustc_hash::FxHasher::default();
        other.hash(&mut hash);
        // Force a non-equal candidate into this state's bucket.
        memo.buckets.insert(hash.finish(), smallvec::smallvec![source.id().unwrap()]);
        let target = memo.hold(other.clone());
        assert_ne!(source.id(), target.id());
        assert!(memo.view(&target) == &other);
        assert_eq!(memo.hold(other).id(), target.id());
    }

    #[test]
    fn many_transition_memo_bounds_large_guard_storage() {
        let large = FullWalkManyState::Branches(smallvec::smallvec![FullWalkBranch {
            lexer_state: 3,
            parser_node: 9,
            prune_guard: FullWalkPruneGuard::Pending((0..5).map(|x| (x, x)).collect()),
        }]);
        let mut memo = FullWalkManyTransitionMemo::new(8);
        let cursor = memo.hold(large.clone());
        assert!(cursor.id().is_none());
        assert!(memo.states.is_empty());
        assert!(memo.view(&cursor) == &large);
    }


    #[test]
    fn dense_parser_canonical_preserves_complete_stack_languages() {
        let (mut cache, _) = FullWalkParserCache::from_roots(&DynamicBranches::new(), 9, false);
        cache.canonicalize = true;
        let first = ParserStacks::from_stacks(&[
            (vec![0, 1, 7], ()), (vec![0, 2, 7], ()),
        ]);
        let id = cache.push_parser_stacks(first.clone());
        // Pass the cost gate without requiring a grammar-specific shape.
        for value in 1..32 {
            cache.push_parser_stacks(ParserStacks::from_single_stack(vec![value + 100], ()));
        }
        let same = ParserStacks::merge_many([
            ParserStacks::from_single_stack(vec![0, 2, 7], ()),
            ParserStacks::from_single_stack(vec![0, 1, 7], ()),
        ]);
        assert_eq!(cache.intern_parser_stacks(same), id);
        let different = ParserStacks::from_single_stack(vec![0, 3, 7], ());
        let different_id = cache.intern_parser_stacks(different.clone());
        assert_ne!(different_id, id, "equal parser tops are not an equivalence");
        assert_eq!(cache.intern_parser_stacks(different), different_id);
        let empty_word = first.merge(&ParserStacks::from_single_stack(Vec::new(), ()));
        assert_ne!(cache.intern_parser_stacks(empty_word), id, "empty-stack acceptance is part of the key");
        assert!(cache.canonical.as_ref().unwrap().hits >= 2);
    }

    #[test]
    fn dense_parser_canonical_budget_failure_falls_back_without_aliasing() {
        let (mut cache, _) = FullWalkParserCache::from_roots(&DynamicBranches::new(), 9, false);
        cache.canonicalize = true;
        for value in 0..32 {
            cache.push_parser_stacks(ParserStacks::from_single_stack(vec![value], ()));
        }
        cache.canonical = Some(Box::new(FullWalkDenseParserCanonical {
            keys: crate::ds::leveled_gss::GssSemanticKeyInterner::with_budget(1, 1, 1),
            nodes: FxHashMap::from_iter([(0, 0)]),
            hits: 0,
        }));
        let gss = ParserStacks::from_single_stack(vec![88, 99], ());
        let first = cache.intern_parser_stacks(gss.clone());
        let second = cache.intern_parser_stacks(gss.clone());
        assert_ne!(first, 0);
        assert_ne!(first, second);
        assert!(cache.nodes[first as usize].gss.ptr_eq(&gss));
        assert!(cache.nodes[second as usize].gss.ptr_eq(&gss));
        assert!(cache.canonical.as_ref().unwrap().keys.is_exhausted());
    }

    fn guarded_state(parser: u32, terminal: TerminalID) -> FullWalkManyState {
        FullWalkManyState::Branches(smallvec::smallvec![
            FullWalkBranch {
                lexer_state: 5,
                parser_node: parser,
                prune_guard: FullWalkPruneGuard::Passed,
            },
            FullWalkBranch {
                lexer_state: 0,
                parser_node: 9,
                prune_guard: FullWalkPruneGuard::Pending(smallvec::smallvec![(5, terminal)]),
            },
        ])
    }


}

#[cfg(test)]
mod wide_scalar_dispatch_tests {
    use super::*;

    #[test]
    fn lazy_union_pair_memo_preserves_virtual_members_aliases_and_bounds() {
        let tokenizer = crate::automata::lexer::tokenizer::arbitrary_flat32_test_tokenizer();
        let cache = std::sync::Mutex::new(DynamicLazyUnionCache::default());
        let overflowed = std::cell::Cell::new(false);
        let (mut table, first) = FullWalkLazyUnion::new(
            &tokenizer, None, None, cache.lock().unwrap(), &[0, 32_768], &overflowed, false,
        ).unwrap();
        table.use_union_pair_memo = true;
        let second = table.intern_states(&[1, 32_768]).unwrap();
        let inputs = [0, 1, 2, 32_768, first, second];
        for &left in &inputs {
            for &right in &inputs {
                let result = table.intern_states(&[left, right]).unwrap();
                let reversed = table.intern_states(&[right, left]).unwrap();
                assert_eq!(result, reversed);
                let cache = unsafe { &*table.cache.get() };
                let members = |state: u32| -> Vec<u32> {
                    if state < table.base_state_count { vec![state] }
                    else { cache.subsets[(state - table.base_state_count) as usize].to_vec() }
                };
                let mut expected = members(left);
                expected.extend(members(right));
                expected.sort_unstable(); expected.dedup();
                assert_eq!(members(result), expected, "the full physical union is the oracle");
            }
        }
        let invalid = table.base_state_count + FullWalkLazyUnion::RESERVED_EXTENSION_STATES as u32;
        assert_eq!(table.intern_states(&[0, invalid]), None);
        assert!(table.continuation_witness_components(invalid).is_empty());
        assert_eq!(table.continuation_witness_components(first).as_slice(), &[0, 32_768]);
        let wide = table.intern_states(&(0..9).collect::<Vec<_>>()).unwrap();
        assert!(table.continuation_witness_components(wide).is_empty(),
            "oversized exact unions must fall back rather than truncate their member set");
        let before = unsafe { (&*table.cache.get()).state_by_union_pair.as_ref().unwrap().len() };
        // Fill only the memo's bounded index with valid known results. Once
        // full, a new union still executes normally but is not retained.
        {
            let cache = unsafe { &mut *table.cache.get() };
            let memo = cache.state_by_union_pair.as_mut().unwrap();
            for i in 1..=(FullWalkLazyUnion::MAX_UNION_PAIR_MEMO - before) {
                memo.insert((1u64 << 63) + i as u64, first);
            }
            assert_eq!(memo.len(), FullWalkLazyUnion::MAX_UNION_PAIR_MEMO);
        }
        let joined = table.intern_states(&[10, 11]).unwrap();
        let cache = unsafe { &*table.cache.get() };
        assert_eq!(cache.subsets[(joined - table.base_state_count) as usize].as_slice(), &[10, 11]);
        assert_eq!(cache.state_by_union_pair.as_ref().unwrap().len(), FullWalkLazyUnion::MAX_UNION_PAIR_MEMO);
    }

    #[test]
    fn lazy_scalar_dispatch_rows_support_states_beyond_flat16() {
        let tokenizer =
            crate::automata::lexer::tokenizer::arbitrary_flat32_test_tokenizer();
        let cache = std::sync::Mutex::new(DynamicLazyUnionCache::default());
        let guard = cache.lock().expect("lazy-union cache lock");
        let overflowed = std::cell::Cell::new(false);
        let (table, root) = FullWalkLazyUnion::new(
            &tokenizer,
            None,
            None,
            guard,
            &[0, 32_768],
            &overflowed,
            false,
        )
        .expect("wide lazy scalar-dispatch table");

        assert!(root >= tokenizer.num_states());
        assert_eq!(table.transition(0, b'a'), 32_768);
        assert_eq!(table.transition(0, b'b'), u32::MAX);
        assert!(!overflowed.get());
    }

    #[test]
    fn lazy_scalar_dispatch_soft_reset_reinitializes_physical_rows() {
        let tokenizer =
            crate::automata::lexer::tokenizer::arbitrary_flat32_test_tokenizer();
        let mut seeded = DynamicLazyUnionCache::default();
        seeded.base_state_count = tokenizer.num_states();
        seeded.state_by_union_pair = Some(Box::new(FxHashMap::from_iter([(17, u32::MAX)])));
        seeded
            .base_rows
            .resize_with(tokenizer.num_states() as usize, || None);
        seeded.subsets.resize_with(
            FullWalkLazyUnion::SOFT_MAX_EXTENSION_STATES,
            || SmallVec::from_slice(&[0, 32_768]),
        );

        let cache = std::sync::Mutex::new(seeded);
        let guard = cache.lock().expect("lazy-union cache lock");
        let overflowed = std::cell::Cell::new(false);
        let (table, _root) = FullWalkLazyUnion::new(
            &tokenizer,
            None,
            None,
            guard,
            &[0, 32_768],
            &overflowed,
            false,
        )
        .expect("lazy table after soft-limit reset");

        {
            let cache = unsafe { &*table.cache.get() };
            assert_eq!(cache.base_rows.len(), tokenizer.num_states() as usize);
            assert!(!cache.state_by_union_pair.as_ref().unwrap().contains_key(&17),
                "old extension coordinates must not survive a soft reset");
        }
        assert_eq!(table.transition(0, b'a'), 32_768);
        assert!(!overflowed.get());
    }

    #[test]
    fn lazy_scalar_dispatch_repeated_soft_resets_keep_physical_rows_live() {
        let tokenizer =
            crate::automata::lexer::tokenizer::arbitrary_flat32_test_tokenizer();
        let cache = std::sync::Mutex::new(DynamicLazyUnionCache::default());

        for _ in 0..3 {
            {
                let mut seeded = cache.lock().expect("lazy-union cache lock");
                seeded.base_state_count = tokenizer.num_states();
                seeded
                    .base_rows
                    .resize_with(tokenizer.num_states() as usize, || None);
                seeded.subsets.resize_with(
                    FullWalkLazyUnion::SOFT_MAX_EXTENSION_STATES,
                    || SmallVec::from_slice(&[0, 32_768]),
                );
            }

            let guard = cache.lock().expect("lazy-union cache lock");
            let overflowed = std::cell::Cell::new(false);
            let (table, _root) = FullWalkLazyUnion::new(
                &tokenizer,
                None,
                None,
                guard,
                &[0, 32_768],
                &overflowed,
                false,
            )
            .expect("lazy table after repeated soft-limit reset");

            {
                let cache = unsafe { &*table.cache.get() };
                assert_eq!(cache.base_rows.len(), tokenizer.num_states() as usize);
            }
            assert_eq!(table.transition(0, b'a'), 32_768);
            assert!(!overflowed.get());
        }
    }
}
