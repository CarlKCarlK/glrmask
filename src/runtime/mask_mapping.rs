use rayon::prelude::*;

pub type InternalTokenBufMasks = Vec<(u32, u32)>;

#[derive(Clone, Copy, Debug, Default)]
struct SparseEntry {
    word_idx: u32,
    mask: u32,
}

impl SparseEntry {
    #[inline(always)]
    fn new(word_idx: u32, mask: u32) -> Self {
        Self { word_idx, mask }
    }

    #[inline(always)]
    fn word_idx(self) -> u32 {
        self.word_idx
    }

    #[inline(always)]
    fn mask(self) -> u32 {
        self.mask
    }
}

#[derive(Default, Clone, Copy)]
pub(crate) struct DenseToBufProfileStats {
    pub dense_words_visited: u64,
    pub normal_full_word_hits: u64,
    pub normal_group_complement_hits: u64,
    pub complement_full_word_hits: u64,
    pub complement_full_byte_groups: u64,
    pub complement_full_nibble_groups: u64,
    pub complement_remaining_bits: u64,
    pub normal_token_iterations: u64,
    pub complement_token_iterations: u64,
    pub normal_sparse_entries: u64,
    pub normal_group_complement_sparse_entries: u64,
    pub complement_sparse_entries: u64,
    pub complement_heavy_dense_clears: u64,
    pub complement_max_sparse_span: u64,
    pub group_or_sparse_entries: u64,
    pub group_andnot_sparse_entries: u64,
    pub complement_path_used: bool,
}

#[derive(Default, Clone, Copy)]
pub(crate) struct DeltaReplayProfileStats {
    pub added_word_group_hits: u64,
    pub added_word_group_entries: u64,
    pub removed_word_group_hits: u64,
    pub removed_word_group_entries: u64,
    pub added_byte_group_hits: u64,
    pub added_byte_group_entries: u64,
    pub removed_byte_group_hits: u64,
    pub removed_byte_group_entries: u64,
    pub added_token_iterations: u64,
    pub added_token_entries: u64,
    pub removed_token_iterations: u64,
    pub removed_token_entries: u64,
}

#[derive(Default, Clone, Copy)]
struct DenseSelectionStats {
    n_set: usize,
    selected_entry_cost: usize,
}

#[derive(Debug, Clone, Copy, Default)]
struct EntryRange {
    start: u32,
    len: u32,
}

#[derive(Debug, Clone, Default)]
struct EntryTable {
    ranges: Box<[EntryRange]>,
    entries: Box<[SparseEntry]>,
}

impl EntryTable {
    fn from_groups(groups: Vec<Box<[SparseEntry]>>) -> Self {
        let mut ranges = Vec::with_capacity(groups.len());
        let total_entries = groups.iter().map(|group| group.len()).sum();
        let mut entries = Vec::with_capacity(total_entries);
        for group in groups {
            let start = entries.len();
            entries.extend(group.iter().copied());
            ranges.push(EntryRange {
                start: start as u32,
                len: entries.len().saturating_sub(start) as u32,
            });
        }
        Self {
            ranges: ranges.into_boxed_slice(),
            entries: entries.into_boxed_slice(),
        }
    }

    fn len(&self) -> usize {
        self.ranges.len()
    }

    fn get(&self, index: usize) -> Option<&[SparseEntry]> {
        let range = *self.ranges.get(index)?;
        let start = range.start as usize;
        let end = start + range.len as usize;
        Some(&self.entries[start..end])
    }
}

#[derive(Debug, Clone, Default)]
pub struct FinalMaskMapping {
    token_entries: EntryTable,
    token_entry_lens: Vec<usize>,
    word_entry_len_sums: Vec<usize>,
    quad_pattern_entries: EntryTable,
    quad_pattern_dense: Vec<Option<Box<[u32]>>>,
    quad_group_entries: EntryTable,
    quad_group_dense: Vec<Option<Box<[u32]>>>,
    byte_group_entries: EntryTable,
    byte_group_dense: Vec<Option<Box<[u32]>>>,
    halfword_group_entries: EntryTable,
    halfword_group_dense: Vec<Option<Box<[u32]>>>,
    word32_group_entries: EntryTable,
    word32_group_dense: Vec<Option<Box<[u32]>>>,
    word_group_entries: EntryTable,
    word_group_dense: Vec<Option<Box<[u32]>>>,
    word_group_entry_prefix: Vec<usize>,
    word_group_dense_prefix: Vec<Box<[u32]>>,
    all_tokens_mask: Box<[u32]>,
    buf_words: usize,
}

impl FinalMaskMapping {
    pub fn new(internal_to_original: &[Vec<u32>], buf_words: usize) -> Self {
        let token_entries = compute_token_entries(internal_to_original, buf_words);
        let token_entry_lens = token_entries
            .iter()
            .map(|entries| entries.len())
            .collect::<Vec<_>>();
        let word_entry_len_sums = token_entry_lens
            .chunks(64)
            .map(|chunk| chunk.iter().sum())
            .collect::<Vec<_>>();
        let quad_pattern_entries = compute_quad_pattern_entries(&token_entries, buf_words);
        let quad_group_entries = compute_block_entries(&token_entries, buf_words, 4);
        let byte_group_entries = compute_block_entries(&token_entries, buf_words, 8);
        let halfword_group_entries = compute_block_entries(&token_entries, buf_words, 16);
        let word32_group_entries = compute_block_entries(&token_entries, buf_words, 32);
        let word_group_entries = compute_block_entries(&token_entries, buf_words, 64);
        let dense_group_threshold = dense_group_threshold(buf_words);
        let quad_pattern_dense =
            compute_heavy_group_dense(&quad_pattern_entries, buf_words, dense_group_threshold);
        let quad_group_dense =
            compute_heavy_group_dense(&quad_group_entries, buf_words, dense_group_threshold);
        let byte_group_dense =
            compute_heavy_group_dense(&byte_group_entries, buf_words, dense_group_threshold);
        let halfword_group_dense =
            compute_heavy_group_dense(&halfword_group_entries, buf_words, dense_group_threshold);
        let word32_group_dense =
            compute_heavy_group_dense(&word32_group_entries, buf_words, dense_group_threshold);
        let word_group_dense =
            compute_heavy_group_dense(&word_group_entries, buf_words, dense_group_threshold);
        let word_group_entry_prefix = compute_entry_prefix(&word_group_entries);
        let word_group_dense_prefix = compute_dense_prefix(&word_group_entries, buf_words);
        let all_tokens_mask = word_group_dense_prefix
            .last()
            .cloned()
            .unwrap_or_else(|| vec![0u32; buf_words].into_boxed_slice());

        Self {
            token_entries: EntryTable::from_groups(token_entries),
            token_entry_lens,
            word_entry_len_sums,
            quad_pattern_entries: EntryTable::from_groups(quad_pattern_entries),
            quad_pattern_dense,
            quad_group_entries: EntryTable::from_groups(quad_group_entries),
            quad_group_dense,
            byte_group_entries: EntryTable::from_groups(byte_group_entries),
            byte_group_dense,
            halfword_group_entries: EntryTable::from_groups(halfword_group_entries),
            halfword_group_dense,
            word32_group_entries: EntryTable::from_groups(word32_group_entries),
            word32_group_dense,
            word_group_entries: EntryTable::from_groups(word_group_entries),
            word_group_dense,
            word_group_entry_prefix,
            word_group_dense_prefix,
            all_tokens_mask,
            buf_words,
        }
    }

    pub fn buf_words(&self) -> usize {
        self.buf_words
    }

    pub fn internal_len(&self) -> usize {
        self.token_entries.len()
    }

    pub fn fill_internal_ids(&self, internal_ids: &[u32], out: &mut [u32]) {
        let n_internal = self.token_entries.len();
        let selected = internal_ids.len().min(n_internal);
        let missing = n_internal.saturating_sub(selected);

        if selected >= n_internal || (selected * 5 >= n_internal * 4 && missing <= 128) {
            copy_dense(out, &self.all_tokens_mask);
            self.andnot_missing_ids(internal_ids, out);
        } else {
            self.fill_internal_ids_by_runs(internal_ids, out);
        }
    }

    pub fn fill_dense_words(&self, dense: &[u64], out: &mut [u32]) {
        self.or_dense_to_buf_fast(dense, out, true);
    }

    pub fn fill_dense_words_complement(&self, dense: &[u64], out: &mut [u32]) {
        copy_dense(out, &self.all_tokens_mask);
        self.andnot_missing_dense_fast(dense, out);
    }

    pub(crate) fn or_dense_to_buf_fast(&self, dense: &[u64], buf: &mut [u32], buf_zeroed: bool) {
        let n_internal = self.token_entries.len();
        if n_internal == 0 || dense.is_empty() {
            return;
        }

        let n_set = self.count_dense_set_bits(dense);
        if n_set >= n_internal && !self.all_tokens_mask.is_empty() {
            if buf_zeroed {
                copy_dense(buf, &self.all_tokens_mask);
            } else {
                or_dense(buf, &self.all_tokens_mask);
            }
            return;
        }
        if n_set == 0 {
            return;
        }

        let n_missing = n_internal - n_set;
        // The complement path is destructive: it seeds from ALL and then
        // clears missing-token bits. That is only equivalent to OR-ing the
        // selected image when the destination is known zero. With an existing
        // partial mask, clearing a missing token can erase a bit admitted by a
        // different parser path.
        if buf_zeroed
            && !self.all_tokens_mask.is_empty()
            && n_set.saturating_mul(5) >= n_internal.saturating_mul(3)
            && n_missing <= 800
        {
            let selection = self.dense_selection_stats(dense);
            let clear_cost = self.estimate_missing_dense_clear_cost(dense);
            let scattered_dense_selected =
                n_set.saturating_mul(3) >= n_internal.saturating_mul(2)
                    && selection.selected_entry_cost > self.buf_words.saturating_mul(3);
            if !scattered_dense_selected
                && self.buf_words + clear_cost >= selection.selected_entry_cost
            {
                self.or_selected_dense_fast(dense, buf, buf_zeroed);
                return;
            }
            copy_dense(buf, &self.all_tokens_mask);
            self.andnot_missing_dense_fast(dense, buf);
            return;
        }

        self.or_selected_dense_fast(dense, buf, buf_zeroed);
    }

    pub(crate) fn estimate_dense_to_buf_cost(&self, dense: &[u64]) -> u64 {
        let n_internal = self.token_entries.len();
        if n_internal == 0 || dense.is_empty() {
            return 0;
        }
        let n_set = self.count_dense_set_bits(dense);
        if n_set >= n_internal && !self.all_tokens_mask.is_empty() {
            return self.buf_words as u64;
        }
        if n_set == 0 {
            return 0;
        }
        let n_missing = n_internal - n_set;
        if !self.all_tokens_mask.is_empty()
            && n_set.saturating_mul(5) >= n_internal.saturating_mul(3)
            && n_missing <= 800
        {
            let stats = self.dense_selection_stats(dense);
            let clear_cost = self.estimate_missing_dense_clear_cost(dense);
            let scattered_dense_selected =
                n_set.saturating_mul(3) >= n_internal.saturating_mul(2)
                    && stats.selected_entry_cost > self.buf_words.saturating_mul(3);
            if scattered_dense_selected || self.buf_words + clear_cost < stats.selected_entry_cost {
                return (self.buf_words + clear_cost) as u64;
            }
        }
        self.internal_ids_from_dense(dense)
            .iter()
            .map(|&id| self.token_entries.get(id as usize).map_or(0, |entries| entries.len()))
            .sum::<usize>() as u64
    }

    pub(crate) fn or_dense_to_buf(
        &self,
        dense: &[u64],
        buf: &mut [u32],
        buf_zeroed: bool,
    ) -> DenseToBufProfileStats {
        let mut stats = DenseToBufProfileStats::default();
        let n_internal = self.token_entries.len();
        if n_internal == 0 || dense.is_empty() {
            return stats;
        }

        let n_set = self.count_dense_set_bits(dense);
        if n_set >= n_internal && !self.all_tokens_mask.is_empty() {
            if buf_zeroed {
                copy_dense(buf, &self.all_tokens_mask);
            } else {
                or_dense(buf, &self.all_tokens_mask);
            }
            return stats;
        }
        if n_set == 0 {
            return stats;
        }

        let n_missing = n_internal - n_set;
        // Complement-and-clear is only an OR-equivalent conversion when the
        // destination is known zero.
        if buf_zeroed
            && !self.all_tokens_mask.is_empty()
            && n_set.saturating_mul(5) >= n_internal.saturating_mul(3)
            && n_missing <= 800
        {
            let selection = self.dense_selection_stats(dense);
            let clear_cost = self.estimate_missing_dense_clear_cost(dense);
            let scattered_dense_selected =
                n_set.saturating_mul(3) >= n_internal.saturating_mul(2)
                    && selection.selected_entry_cost > self.buf_words.saturating_mul(3);
            if !scattered_dense_selected
                && self.buf_words + clear_cost >= selection.selected_entry_cost
            {
                self.or_selected_dense(dense, buf, buf_zeroed, &mut stats);
                return stats;
            }
            stats.complement_path_used = true;
            copy_dense(buf, &self.all_tokens_mask);
            self.andnot_missing_dense(dense, buf, &mut stats);
            return stats;
        }

        self.or_selected_dense(dense, buf, buf_zeroed, &mut stats);
        stats
    }

    fn internal_ids_from_dense(&self, dense: &[u64]) -> Vec<u32> {
        let n_internal = self.token_entries.len();
        let mut internal_ids = Vec::new();
        for (wi, &word) in dense.iter().enumerate() {
            let base = wi * 64;
            if base >= n_internal {
                break;
            }
            let valid_bits = (n_internal - base).min(64);
            let valid_mask = if valid_bits == 64 {
                !0u64
            } else {
                (1u64 << valid_bits) - 1
            };
            let mut bits = word & valid_mask;
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                internal_ids.push((base + bit) as u32);
                bits &= bits - 1;
            }
        }
        internal_ids
    }

    fn count_dense_set_bits(&self, dense: &[u64]) -> usize {
        let n_internal = self.token_entries.len();
        let mut n_set = 0usize;
        for (wi, &word) in dense.iter().enumerate() {
            let base = wi * 64;
            if base >= n_internal {
                break;
            }
            let valid_bits = (n_internal - base).min(64);
            let valid_mask = if valid_bits == 64 {
                !0u64
            } else {
                (1u64 << valid_bits) - 1
            };
            n_set += (word & valid_mask).count_ones() as usize;
        }
        n_set
    }

    fn missing_dense_entry_cost(&self, dense: &[u64]) -> usize {
        let n_internal = self.token_entries.len();
        let mut cost = 0usize;
        for (wi, &word) in dense.iter().enumerate() {
            let base = wi * 64;
            if base >= n_internal {
                break;
            }
            let valid_bits = (n_internal - base).min(64);
            let valid_mask = if valid_bits == 64 {
                !0u64
            } else {
                (1u64 << valid_bits) - 1
            };
            let mut missing = !word & valid_mask;
            while missing != 0 {
                let bit = missing.trailing_zeros() as usize;
                cost += self.token_entry_lens.get(base + bit).copied().unwrap_or(0);
                missing &= missing - 1;
            }
        }
        cost
    }

    fn estimate_missing_dense_clear_cost(&self, dense: &[u64]) -> usize {
        let n_internal = self.token_entries.len();
        let mut cost = 0usize;
        let mut wi = 0usize;
        while wi < dense.len() {
            let word = dense[wi];
            let base = wi * 64;
            if base >= n_internal {
                break;
            }
            let valid_bits = (n_internal - base).min(64);
            let valid_mask = if valid_bits == 64 {
                !0u64
            } else {
                (1u64 << valid_bits) - 1
            };
            let missing = !word & valid_mask;
            if missing == 0 {
                wi += 1;
                continue;
            }
            if valid_bits == 64 && missing == !0u64 {
                let run_start = wi;
                wi += 1;
                while wi < dense.len() {
                    let base = wi * 64;
                    if base >= n_internal || n_internal - base < 64 || dense[wi] != 0 {
                        break;
                    }
                    wi += 1;
                }
                cost += self.group_entries_in(run_start, wi).min(self.buf_words);
                continue;
            }
            cost += self.estimate_andnot_bits_cost(base, missing);
            wi += 1;
        }
        cost
    }

    fn estimate_andnot_bits_cost(&self, base: usize, mut bits: u64) -> usize {
        let mut cost = 0usize;
        let word32_base = base / 32;
        for lane in 0..2 {
            let shift = lane * 32;
            let mask = 0xffff_ffffu64 << shift;
            if bits & mask == mask {
                cost += self
                    .word32_group_entries
                    .get(word32_base + lane)
                    .map_or(0, |entries| entries.len());
                bits &= !mask;
            }
        }

        let halfword_base = base / 16;
        for lane in 0..4 {
            let shift = lane * 16;
            let mask = 0xffffu64 << shift;
            if bits & mask == mask {
                cost += self
                    .halfword_group_entries
                    .get(halfword_base + lane)
                    .map_or(0, |entries| entries.len());
                bits &= !mask;
            }
        }

        let byte_base = base / 8;
        for lane in 0..8 {
            let shift = lane * 8;
            let mask = 0xffu64 << shift;
            if bits & mask == mask {
                cost += self
                    .byte_group_entries
                    .get(byte_base + lane)
                    .map_or(0, |entries| entries.len());
                bits &= !mask;
            }
        }

        let quad_base = base / 4;
        for lane in 0..16 {
            let shift = lane * 4;
            let pattern = ((bits >> shift) & 0x0f) as usize;
            if pattern != 0 {
                cost += self
                    .quad_pattern_entries
                    .get((quad_base + lane) * 16 + pattern)
                    .map_or(0, |entries| entries.len());
                bits &= !(0x0fu64 << shift);
            }
        }

        cost + self.entry_cost_for_bits(base, bits)
    }

    fn dense_selection_stats(&self, dense: &[u64]) -> DenseSelectionStats {
        let n_internal = self.token_entries.len();
        let mut stats = DenseSelectionStats::default();
        for (wi, &word) in dense.iter().enumerate() {
            let base = wi * 64;
            if base >= n_internal {
                break;
            }
            let remaining = n_internal - base;
            let valid_bits = remaining.min(64);
            let valid_mask = if valid_bits == 64 { !0u64 } else { (1u64 << valid_bits) - 1 };
            let selected = word & valid_mask;
            let missing = !word & valid_mask;
            let selected_count = selected.count_ones() as usize;
            let missing_count = valid_bits - selected_count;
            stats.n_set += selected_count;

            let word_sum = self.word_entry_len_sums.get(wi).copied().unwrap_or(0);
            let missing_cost = self.entry_cost_for_bits(base, missing);
            let selected_cost = if selected_count <= missing_count {
                self.entry_cost_for_bits(base, selected)
            } else {
                word_sum.saturating_sub(missing_cost)
            };
            stats.selected_entry_cost += selected_cost;
        }
        stats
    }

    fn entry_cost_for_bits(&self, base: usize, mut bits: u64) -> usize {
        let mut cost = 0usize;
        while bits != 0 {
            let bit = bits.trailing_zeros() as usize;
            cost += self.token_entry_lens.get(base + bit).copied().unwrap_or(0);
            bits &= bits - 1;
        }
        cost
    }

    fn fill_internal_ids_by_runs(&self, internal_ids: &[u32], out: &mut [u32]) {
        let mut idx = 0usize;
        let mut wrote = false;
        while idx < internal_ids.len() {
            let run_start = internal_ids[idx] as usize;
            let mut idx_end = idx + 1;
            let mut run_end = run_start + 1;
            while idx_end < internal_ids.len() && internal_ids[idx_end] as usize == run_end {
                idx_end += 1;
                run_end += 1;
            }

            if run_end > self.token_entries.len() {
                for &internal_id in &internal_ids[idx..idx_end] {
                    wrote |= self.or_token(internal_id as usize, out);
                }
                idx = idx_end;
                continue;
            }

            wrote |= self.or_internal_run(run_start, run_end, out, !wrote);
            idx = idx_end;
        }
    }

    fn or_selected_dense(
        &self,
        dense: &[u64],
        out: &mut [u32],
        buf_zeroed: bool,
        stats: &mut DenseToBufProfileStats,
    ) {
        let n_internal = self.token_entries.len();
        let seed_full_run = if buf_zeroed {
            self.first_full_dense_run(dense)
        } else {
            None
        };
        if let Some((run_start, run_end)) = seed_full_run {
            self.or_full_group_run(run_start, run_end, out, true);
            stats.normal_full_word_hits += (run_end - run_start) as u64;
        }
        let mut wi = 0usize;
        let mut wrote = seed_full_run.is_some();
        while wi < dense.len() {
            let word = dense[wi];
            let base = wi * 64;
            if base >= n_internal {
                break;
            }
            if let Some((run_start, run_end)) = seed_full_run
                && wi == run_start
            {
                stats.dense_words_visited += (run_end - run_start) as u64;
                wi = run_end;
                continue;
            }
            stats.dense_words_visited += 1;
            let remaining = n_internal - base;
            let valid_bits = remaining.min(64);
            let valid_mask = if valid_bits == 64 { !0u64 } else { (1u64 << valid_bits) - 1 };
            let bits = word & valid_mask;
            if bits == 0 {
                wi += 1;
                continue;
            }
            if bits == valid_mask && valid_bits == 64 {
                let run_start = wi;
                wi += 1;
                while wi < dense.len() {
                    let base = wi * 64;
                    if base >= n_internal {
                        break;
                    }
                    let remaining = n_internal - base;
                    if remaining < 64 || dense[wi] != !0u64 {
                        break;
                    }
                    stats.dense_words_visited += 1;
                    wi += 1;
                }
                self.or_full_group_run(run_start, wi, out, buf_zeroed && !wrote);
                wrote = true;
                stats.normal_full_word_hits += (wi - run_start) as u64;
                continue;
            }
            self.or_bits(base, bits, out, buf_zeroed, stats);
            wrote = true;
            wi += 1;
        }
    }

    fn or_selected_dense_fast(&self, dense: &[u64], out: &mut [u32], buf_zeroed: bool) {
        let n_internal = self.token_entries.len();
        let seed_full_run = if buf_zeroed {
            self.first_full_dense_run(dense)
        } else {
            None
        };
        if let Some((run_start, run_end)) = seed_full_run {
            self.or_full_group_run(run_start, run_end, out, true);
        }
        let mut wi = 0usize;
        let mut wrote = seed_full_run.is_some();
        while wi < dense.len() {
            let word = dense[wi];
            let base = wi * 64;
            if base >= n_internal {
                break;
            }
            if let Some((run_start, run_end)) = seed_full_run
                && wi == run_start
            {
                wi = run_end;
                continue;
            }
            let remaining = n_internal - base;
            let valid_bits = remaining.min(64);
            let valid_mask = if valid_bits == 64 {
                !0u64
            } else {
                (1u64 << valid_bits) - 1
            };
            let bits = word & valid_mask;
            if bits == 0 {
                wi += 1;
                continue;
            }
            if bits == valid_mask && valid_bits == 64 {
                let run_start = wi;
                wi += 1;
                while wi < dense.len() {
                    let base = wi * 64;
                    if base >= n_internal {
                        break;
                    }
                    let remaining = n_internal - base;
                    if remaining < 64 || dense[wi] != !0u64 {
                        break;
                    }
                    wi += 1;
                }
                self.or_full_group_run(run_start, wi, out, buf_zeroed && !wrote);
                wrote = true;
                continue;
            }
            self.or_bits_fast(base, bits, out, buf_zeroed);
            wrote = true;
            wi += 1;
        }
    }

    fn first_full_dense_run(&self, dense: &[u64]) -> Option<(usize, usize)> {
        let n_internal = self.token_entries.len();
        let mut wi = 0usize;
        while wi < dense.len() {
            let base = wi * 64;
            if base >= n_internal {
                break;
            }
            let remaining = n_internal - base;
            if remaining >= 64 && dense[wi] == !0u64 {
                let run_start = wi;
                wi += 1;
                while wi < dense.len() {
                    let base = wi * 64;
                    if base >= n_internal {
                        break;
                    }
                    let remaining = n_internal - base;
                    if remaining < 64 || dense[wi] != !0u64 {
                        break;
                    }
                    wi += 1;
                }
                return Some((run_start, wi));
            }
            wi += 1;
        }
        None
    }

    fn andnot_missing_dense(
        &self,
        dense: &[u64],
        out: &mut [u32],
        stats: &mut DenseToBufProfileStats,
    ) {
        let n_internal = self.token_entries.len();
        let mut wi = 0usize;
        while wi < dense.len() {
            let word = dense[wi];
            let base = wi * 64;
            if base >= n_internal {
                break;
            }
            stats.dense_words_visited += 1;
            let remaining = n_internal - base;
            let valid_bits = remaining.min(64);
            let valid_mask = if valid_bits == 64 { !0u64 } else { (1u64 << valid_bits) - 1 };
            let missing = !word & valid_mask;
            if missing == 0 {
                wi += 1;
                continue;
            }
            if valid_bits == 64 && missing == !0u64 {
                let run_start = wi;
                wi += 1;
                while wi < dense.len() {
                    let base = wi * 64;
                    if base >= n_internal || n_internal - base < 64 || dense[wi] != 0 {
                        break;
                    }
                    stats.dense_words_visited += 1;
                    wi += 1;
                }
                if self.andnot_word_group_run(run_start, wi, out) {
                    stats.complement_full_word_hits += (wi - run_start) as u64;
                }
                continue;
            }
            self.andnot_bits(base, missing, out, stats);
            wi += 1;
        }
    }

    fn andnot_missing_dense_fast(&self, dense: &[u64], out: &mut [u32]) {
        let n_internal = self.token_entries.len();
        let mut wi = 0usize;
        while wi < dense.len() {
            let word = dense[wi];
            let base = wi * 64;
            if base >= n_internal {
                break;
            }
            let remaining = n_internal - base;
            let valid_bits = remaining.min(64);
            let valid_mask = if valid_bits == 64 {
                !0u64
            } else {
                (1u64 << valid_bits) - 1
            };
            let missing = !word & valid_mask;
            if valid_bits == 64 && missing == !0u64 {
                let run_start = wi;
                wi += 1;
                while wi < dense.len() {
                    let base = wi * 64;
                    if base >= n_internal || n_internal - base < 64 || dense[wi] != 0 {
                        break;
                    }
                    wi += 1;
                }
                self.andnot_word_group_run(run_start, wi, out);
                continue;
            }
            if missing != 0 {
                self.andnot_bits_fast(base, missing, out);
            }
            wi += 1;
        }
    }

    fn or_internal_run(&self, mut start: usize, end: usize, out: &mut [u32], can_copy: bool) -> bool {
        let mut wrote = false;
        while start < end && !start.is_multiple_of(4) {
            wrote |= self.or_token(start, out);
            start += 1;
        }
        while start + 4 <= end && !start.is_multiple_of(8) {
            wrote |= self.or_quad_group(start / 4, out);
            start += 4;
        }
        while start < end && !start.is_multiple_of(8) {
            wrote |= self.or_token(start, out);
            start += 1;
        }
        while start + 8 <= end && !start.is_multiple_of(64) {
            wrote |= self.or_byte_group(start / 8, out);
            start += 8;
        }
        while start < end && !start.is_multiple_of(64) {
            wrote |= self.or_token(start, out);
            start += 1;
        }
        let group_start = start / 64;
        let group_end = end / 64;
        if group_start < group_end {
            wrote |= self.or_full_group_run(group_start, group_end, out, can_copy && !wrote);
            start = group_end * 64;
        }
        while start + 8 <= end {
            wrote |= self.or_byte_group(start / 8, out);
            start += 8;
        }
        while start + 4 <= end {
            wrote |= self.or_quad_group(start / 4, out);
            start += 4;
        }
        while start < end {
            wrote |= self.or_token(start, out);
            start += 1;
        }
        wrote
    }

    fn or_full_group_run(&self, start: usize, end: usize, out: &mut [u32], can_copy: bool) -> bool {
        if start >= end {
            return false;
        }
        if end == start + 1
            && let Some(Some(dense)) = self.word_group_dense.get(start)
        {
            if can_copy {
                copy_dense(out, dense);
            } else {
                or_dense(out, dense);
            }
            return true;
        }
        let sparse_entries = self.group_entries_in(start, end);
        if sparse_entries > self.buf_words / 4 && end < self.word_group_dense_prefix.len() {
            let before = &self.word_group_dense_prefix[start];
            let after = &self.word_group_dense_prefix[end];
            if can_copy {
                copy_prefix_diff(out, before, after);
            } else {
                or_prefix_diff(out, before, after);
            }
            return true;
        }
        let mut wrote = false;
        for group_id in start..end {
            if let Some(entries) = self.word_group_entries.get(group_id) {
                or_sparse(out, entries);
                wrote |= !entries.is_empty();
            }
        }
        wrote
    }

    fn group_entries_in(&self, start: usize, end: usize) -> usize {
        self.word_group_entry_prefix
            .get(end)
            .zip(self.word_group_entry_prefix.get(start))
            .map(|(end, start)| end - start)
            .unwrap_or_else(|| {
                (start..end)
                    .map(|group_id| {
                        self.word_group_entries
                            .get(group_id)
                            .map_or(0, |entries| entries.len())
                    })
                    .sum()
            })
    }

    #[inline(always)]
    fn or_token(&self, internal_id: usize, out: &mut [u32]) -> bool {
        let Some(entries) = self.token_entries.get(internal_id) else {
            return false;
        };
        or_sparse(out, entries);
        !entries.is_empty()
    }

    #[inline(always)]
    fn andnot_token(&self, internal_id: usize, out: &mut [u32]) {
        let Some(entries) = self.token_entries.get(internal_id) else {
            return;
        };
        andnot_sparse(out, entries);
    }

    #[inline(always)]
    fn andnot_quad_group(&self, group_id: usize, out: &mut [u32]) -> bool {
        if let Some(Some(dense)) = self.quad_group_dense.get(group_id) {
            andnot_dense(out, dense);
            return true;
        }
        let Some(entries) = self.quad_group_entries.get(group_id) else {
            return false;
        };
        andnot_sparse(out, entries);
        !entries.is_empty()
    }

    #[inline(always)]
    fn andnot_byte_group(&self, group_id: usize, out: &mut [u32]) -> bool {
        if let Some(Some(dense)) = self.byte_group_dense.get(group_id) {
            andnot_dense(out, dense);
            return true;
        }
        let Some(entries) = self.byte_group_entries.get(group_id) else {
            return false;
        };
        andnot_sparse(out, entries);
        !entries.is_empty()
    }

    #[inline(always)]
    fn andnot_halfword_group(&self, group_id: usize, out: &mut [u32]) -> bool {
        if let Some(Some(dense)) = self.halfword_group_dense.get(group_id) {
            andnot_dense(out, dense);
            return true;
        }
        let Some(entries) = self.halfword_group_entries.get(group_id) else {
            return false;
        };
        andnot_sparse(out, entries);
        !entries.is_empty()
    }

    #[inline(always)]
    fn andnot_word32_group(&self, group_id: usize, out: &mut [u32]) -> bool {
        if let Some(Some(dense)) = self.word32_group_dense.get(group_id) {
            andnot_dense(out, dense);
            return true;
        }
        let Some(entries) = self.word32_group_entries.get(group_id) else {
            return false;
        };
        andnot_sparse(out, entries);
        !entries.is_empty()
    }

    fn andnot_word_group(&self, group_id: usize, out: &mut [u32]) -> bool {
        let Some(entries) = self.word_group_entries.get(group_id) else {
            return false;
        };
        if let Some(Some(dense)) = self.word_group_dense.get(group_id) {
            andnot_dense(out, dense);
            return true;
        }
        if entries.len() > self.buf_words / 4 && group_id + 1 < self.word_group_dense_prefix.len()
        {
            let before = &self.word_group_dense_prefix[group_id];
            let after = &self.word_group_dense_prefix[group_id + 1];
            andnot_prefix_diff(out, before, after);
            return true;
        }
        andnot_sparse(out, entries);
        !entries.is_empty()
    }

    fn andnot_word_group_run(&self, start: usize, end: usize, out: &mut [u32]) -> bool {
        if start >= end {
            return false;
        }
        if end == start + 1 {
            return self.andnot_word_group(start, out);
        }
        let sparse_entries = self.group_entries_in(start, end);
        if sparse_entries > self.buf_words / 4 && end < self.word_group_dense_prefix.len() {
            let before = &self.word_group_dense_prefix[start];
            let after = &self.word_group_dense_prefix[end];
            andnot_prefix_diff(out, before, after);
            return true;
        }
        let mut cleared = false;
        for group_id in start..end {
            if let Some(entries) = self.word_group_entries.get(group_id) {
                andnot_sparse(out, entries);
                cleared |= !entries.is_empty();
            }
        }
        cleared
    }

    #[inline(always)]
    fn or_quad_pattern(&self, group_id: usize, pattern: usize, out: &mut [u32]) -> bool {
        let idx = group_id * 16 + pattern;
        if let Some(Some(dense)) = self.quad_pattern_dense.get(idx) {
            or_dense(out, dense);
            return true;
        }
        let Some(entries) = self.quad_pattern_entries.get(group_id * 16 + pattern) else {
            return false;
        };
        or_sparse(out, entries);
        !entries.is_empty()
    }

    #[inline(always)]
    fn andnot_quad_pattern(&self, group_id: usize, pattern: usize, out: &mut [u32]) -> bool {
        let idx = group_id * 16 + pattern;
        if let Some(Some(dense)) = self.quad_pattern_dense.get(idx) {
            andnot_dense(out, dense);
            return true;
        }
        let Some(entries) = self.quad_pattern_entries.get(group_id * 16 + pattern) else {
            return false;
        };
        andnot_sparse(out, entries);
        !entries.is_empty()
    }

    #[inline(always)]
    fn or_quad_group(&self, group_id: usize, out: &mut [u32]) -> bool {
        if let Some(Some(dense)) = self.quad_group_dense.get(group_id) {
            or_dense(out, dense);
            return true;
        }
        let Some(entries) = self.quad_group_entries.get(group_id) else {
            return false;
        };
        or_sparse(out, entries);
        !entries.is_empty()
    }

    #[inline(always)]
    fn or_byte_group(&self, group_id: usize, out: &mut [u32]) -> bool {
        if let Some(Some(dense)) = self.byte_group_dense.get(group_id) {
            or_dense(out, dense);
            return true;
        }
        let Some(entries) = self.byte_group_entries.get(group_id) else {
            return false;
        };
        or_sparse(out, entries);
        !entries.is_empty()
    }

    fn andnot_missing_ids(&self, internal_ids: &[u32], out: &mut [u32]) {
        let mut expected = 0usize;
        for &raw in internal_ids {
            let selected = raw as usize;
            if selected > self.token_entries.len() {
                break;
            }
            while expected < selected {
                self.andnot_token(expected, out);
                expected += 1;
            }
            expected = selected.saturating_add(1);
        }
        while expected < self.token_entries.len() {
            self.andnot_token(expected, out);
            expected += 1;
        }
    }

    fn or_bits(
        &self,
        base: usize,
        mut bits: u64,
        out: &mut [u32],
        allow_complement_clear: bool,
        stats: &mut DenseToBufProfileStats,
    ) {
        if allow_complement_clear
            && let Some((group_id, missing)) = self.group_complement_bits(base, bits)
        {
            if self.or_full_group_run(group_id, group_id + 1, out, false) {
                stats.normal_group_complement_hits += 1;
                stats.normal_group_complement_sparse_entries += self
                    .word_group_entries
                    .get(group_id)
                    .map_or(0, |entries| entries.len()) as u64;
            }
            self.andnot_bits(base, missing, out, stats);
            return;
        }

        let byte_base = base / 8;
        for lane in 0..8 {
            let shift = lane * 8;
            let mask = 0xffu64 << shift;
            if bits & mask == mask {
                if self.or_byte_group(byte_base + lane, out) {
                    stats.group_or_sparse_entries += self
                        .byte_group_entries
                        .get(byte_base + lane)
                        .map_or(0, |entries| entries.len()) as u64;
                }
                bits &= !mask;
            }
        }

        let quad_base = base / 4;
        for lane in 0..16 {
            let shift = lane * 4;
            let pattern = ((bits >> shift) & 0x0f) as usize;
            if pattern != 0 {
                if self.or_quad_pattern(quad_base + lane, pattern, out) {
                    stats.group_or_sparse_entries += self
                        .quad_pattern_entries
                        .get((quad_base + lane) * 16 + pattern)
                        .map_or(0, |entries| entries.len()) as u64;
                }
                bits &= !(0x0fu64 << shift);
            }
        }

        while bits != 0 {
            stats.normal_token_iterations += 1;
            let bit = bits.trailing_zeros() as usize;
            if self.or_token(base + bit, out) {
                stats.normal_sparse_entries += self
                    .token_entries
                    .get(base + bit)
                    .map_or(0, |entries| entries.len()) as u64;
            }
            bits &= bits - 1;
        }
    }

    fn or_bits_fast(
        &self,
        base: usize,
        mut bits: u64,
        out: &mut [u32],
        allow_complement_clear: bool,
    ) {
        if allow_complement_clear
            && let Some((group_id, missing)) = self.group_complement_bits(base, bits)
        {
            self.or_full_group_run(group_id, group_id + 1, out, false);
            self.andnot_bits_fast(base, missing, out);
            return;
        }

        let byte_base = base / 8;
        for lane in 0..8 {
            let shift = lane * 8;
            let mask = 0xffu64 << shift;
            if bits & mask == mask {
                self.or_byte_group(byte_base + lane, out);
                bits &= !mask;
            }
        }

        let quad_base = base / 4;
        for lane in 0..16 {
            let shift = lane * 4;
            let pattern = ((bits >> shift) & 0x0f) as usize;
            if pattern != 0 {
                self.or_quad_pattern(quad_base + lane, pattern, out);
                bits &= !(0x0fu64 << shift);
            }
        }

        while bits != 0 {
            let bit = bits.trailing_zeros() as usize;
            self.or_token(base + bit, out);
            bits &= bits - 1;
        }
    }

    fn group_complement_bits(&self, base: usize, bits: u64) -> Option<(usize, u64)> {
        let remaining = self.token_entries.len().checked_sub(base)?;
        if remaining < 64 {
            return None;
        }
        let selected_count = bits.count_ones() as usize;
        if selected_count < 48 {
            return None;
        }
        let missing = !bits;
        let selected_cost = self.entry_cost_for_bits(base, bits);
        let missing_cost = self.entry_cost_for_bits(base, missing);
        let group_id = base / 64;
        let group_sparse_cost = self
            .word_group_entries
            .get(group_id)
            .map_or(0, |entries| entries.len());
        let group_or_cost = if group_sparse_cost > self.buf_words / 4 {
            self.buf_words
        } else {
            group_sparse_cost
        };
        if group_or_cost + missing_cost < selected_cost {
            Some((group_id, missing))
        } else {
            None
        }
    }

    fn andnot_bits(
        &self,
        base: usize,
        mut bits: u64,
        out: &mut [u32],
        stats: &mut DenseToBufProfileStats,
    ) {
        if bits == !0u64 {
            if self.andnot_word_group(base / 64, out) {
                stats.complement_full_word_hits += 1;
                stats.group_andnot_sparse_entries += self
                    .word_group_entries
                    .get(base / 64)
                    .map_or(0, |entries| entries.len()) as u64;
            }
            return;
        }

        let word32_base = base / 32;
        for lane in 0..2 {
            let shift = lane * 32;
            let mask = 0xffff_ffffu64 << shift;
            if bits & mask == mask {
                if self.andnot_word32_group(word32_base + lane, out) {
                    stats.group_andnot_sparse_entries += self
                        .word32_group_entries
                        .get(word32_base + lane)
                        .map_or(0, |entries| entries.len()) as u64;
                }
                bits &= !mask;
            }
        }

        let halfword_base = base / 16;
        for lane in 0..4 {
            let shift = lane * 16;
            let mask = 0xffffu64 << shift;
            if bits & mask == mask {
                if self.andnot_halfword_group(halfword_base + lane, out) {
                    stats.group_andnot_sparse_entries += self
                        .halfword_group_entries
                        .get(halfword_base + lane)
                        .map_or(0, |entries| entries.len()) as u64;
                }
                bits &= !mask;
            }
        }

        let byte_base = base / 8;
        for lane in 0..8 {
            let shift = lane * 8;
            let mask = 0xffu64 << shift;
            if bits & mask == mask {
                if self.andnot_byte_group(byte_base + lane, out) {
                    stats.group_andnot_sparse_entries += self
                        .byte_group_entries
                        .get(byte_base + lane)
                        .map_or(0, |entries| entries.len()) as u64;
                }
                stats.complement_full_byte_groups += 1;
                bits &= !mask;
            }
        }

        let quad_base = base / 4;
        for lane in 0..16 {
            let shift = lane * 4;
            let pattern = ((bits >> shift) & 0x0f) as usize;
            if pattern != 0 {
                if self.andnot_quad_pattern(quad_base + lane, pattern, out) {
                    stats.group_andnot_sparse_entries += self
                        .quad_pattern_entries
                        .get((quad_base + lane) * 16 + pattern)
                        .map_or(0, |entries| entries.len()) as u64;
                }
                if pattern == 0x0f {
                    stats.complement_full_nibble_groups += 1;
                }
                bits &= !(0x0fu64 << shift);
            }
        }

        while bits != 0 {
            stats.complement_token_iterations += 1;
            let bit = bits.trailing_zeros() as usize;
            self.andnot_token(base + bit, out);
            stats.complement_sparse_entries += self
                .token_entries
                .get(base + bit)
                .map_or(0, |entries| entries.len()) as u64;
            bits &= bits - 1;
        }
    }

    fn andnot_bits_fast(&self, base: usize, mut bits: u64, out: &mut [u32]) {
        if bits == !0u64 {
            self.andnot_word_group(base / 64, out);
            return;
        }

        let word32_base = base / 32;
        for lane in 0..2 {
            let shift = lane * 32;
            let mask = 0xffff_ffffu64 << shift;
            if bits & mask == mask {
                self.andnot_word32_group(word32_base + lane, out);
                bits &= !mask;
            }
        }

        let halfword_base = base / 16;
        for lane in 0..4 {
            let shift = lane * 16;
            let mask = 0xffffu64 << shift;
            if bits & mask == mask {
                self.andnot_halfword_group(halfword_base + lane, out);
                bits &= !mask;
            }
        }

        let byte_base = base / 8;
        for lane in 0..8 {
            let shift = lane * 8;
            let mask = 0xffu64 << shift;
            if bits & mask == mask {
                self.andnot_byte_group(byte_base + lane, out);
                bits &= !mask;
            }
        }

        let quad_base = base / 4;
        for lane in 0..16 {
            let shift = lane * 4;
            let pattern = ((bits >> shift) & 0x0f) as usize;
            if pattern != 0 {
                self.andnot_quad_pattern(quad_base + lane, pattern, out);
                bits &= !(0x0fu64 << shift);
            }
        }

        while bits != 0 {
            let bit = bits.trailing_zeros() as usize;
            self.andnot_token(base + bit, out);
            bits &= bits - 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::FinalMaskMapping;

    #[test]
    fn dense_or_matches_set_union_exhaustively_on_small_mappings() {
        let mappings = [
            vec![vec![0], vec![1], vec![2], vec![3]],
            vec![vec![0, 4], vec![1], vec![2, 5], vec![3]],
            vec![vec![0, 2], vec![1, 4], vec![3, 5]],
        ];

        for internal_to_original in mappings {
            let mapping = FinalMaskMapping::new(&internal_to_original, 1);
            let internal_count = internal_to_original.len();
            for selected in 0u64..(1u64 << internal_count) {
                let dense = [selected];
                let selected_image = internal_to_original
                    .iter()
                    .enumerate()
                    .filter(|(internal, _)| selected & (1u64 << internal) != 0)
                    .flat_map(|(_, originals)| originals.iter().copied())
                    .fold(0u32, |mask, original| mask | (1u32 << original));

                for initial in 0u32..=0xff {
                    let expected = initial | selected_image;
                    let buf_zeroed = initial == 0;

                    let mut profiled = [initial];
                    mapping.or_dense_to_buf(&dense, &mut profiled, buf_zeroed);
                    assert_eq!(
                        profiled,
                        [expected],
                        "profiled OR mismatch: mapping={internal_to_original:?} selected={selected:#b} initial={initial:#010b}"
                    );

                    let mut fast = [initial];
                    mapping.or_dense_to_buf_fast(&dense, &mut fast, buf_zeroed);
                    assert_eq!(
                        fast,
                        [expected],
                        "fast OR mismatch: mapping={internal_to_original:?} selected={selected:#b} initial={initial:#010b}"
                    );
                }
            }
        }
    }
}

fn compute_token_entries(
    internal_to_original: &[Vec<u32>],
    buf_words: usize,
) -> Vec<Box<[SparseEntry]>> {
    internal_to_original
        .iter()
        .map(|original_ids| {
            let mut entries = Vec::<SparseEntry>::new();
            for &original in original_ids {
                let word_idx = (original / 32) as usize;
                if word_idx >= buf_words {
                    continue;
                }
                let mask = 1u32 << (original & 31);
                let word_idx = word_idx as u32;
                if let Some(existing) = entries
                    .iter_mut()
                    .find(|entry| entry.word_idx() == word_idx)
                {
                    existing.mask |= mask;
                } else {
                    entries.push(SparseEntry::new(word_idx, mask));
                }
            }
            entries.into_boxed_slice()
        })
        .collect()
}

fn compute_block_entries(
    token_entries: &[Box<[SparseEntry]>],
    buf_words: usize,
    block_size: usize,
) -> Vec<Box<[SparseEntry]>> {
    let n_groups = token_entries.len().div_ceil(block_size);
    let build = |group_id: usize| {
        let start = group_id * block_size;
        let end = (start + block_size).min(token_entries.len());
        let mut dense = vec![0u32; buf_words];
        let mut touched = Vec::<usize>::new();
        for entries in &token_entries[start..end] {
            for &entry in entries.iter() {
                let word_idx = entry.word_idx() as usize;
                if dense[word_idx] == 0 {
                    touched.push(word_idx);
                }
                dense[word_idx] |= entry.mask();
            }
        }
        touched.sort_unstable();
        touched.dedup();
            touched
                .into_iter()
                .map(|word_idx| SparseEntry::new(word_idx as u32, dense[word_idx]))
                .collect::<Vec<_>>()
                .into_boxed_slice()
    };
    if rayon::current_num_threads() == 1 {
        (0..n_groups).map(build).collect()
    } else {
        (0..n_groups).into_par_iter().map(build).collect()
    }
}

fn compute_quad_pattern_entries(
    token_entries: &[Box<[SparseEntry]>],
    buf_words: usize,
) -> Vec<Box<[SparseEntry]>> {
    let n_groups = token_entries.len().div_ceil(4);
    let build_group = |group_id: usize| {
        let start = group_id * 4;
        let end = (start + 4).min(token_entries.len());
        let mut groups = Vec::with_capacity(16);
        groups.push(Vec::<SparseEntry>::new().into_boxed_slice());
        for pattern in 1usize..16 {
            let mut dense = vec![0u32; buf_words];
            let mut touched = Vec::<usize>::new();
            for bit in 0..4 {
                if pattern & (1usize << bit) == 0 {
                    continue;
                }
                let token_idx = start + bit;
                if token_idx >= end {
                    continue;
                }
                for &entry in token_entries[token_idx].iter() {
                    let word_idx = entry.word_idx() as usize;
                    if dense[word_idx] == 0 {
                        touched.push(word_idx);
                    }
                    dense[word_idx] |= entry.mask();
                }
            }
            touched.sort_unstable();
            touched.dedup();
            groups.push(
                touched
                    .into_iter()
                    .map(|word_idx| SparseEntry::new(word_idx as u32, dense[word_idx]))
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            );
        }
        groups
    };

    if rayon::current_num_threads() == 1 {
        (0..n_groups).flat_map(build_group).collect()
    } else {
        (0..n_groups)
            .into_par_iter()
            .map(build_group)
            .collect::<Vec<_>>()
            .into_iter()
            .flatten()
            .collect()
    }
}

fn compute_entry_prefix(group_entries: &[Box<[SparseEntry]>]) -> Vec<usize> {
    let mut prefix = Vec::with_capacity(group_entries.len() + 1);
    let mut total = 0usize;
    prefix.push(total);
    for group in group_entries {
        total += group.len();
        prefix.push(total);
    }
    prefix
}

fn compute_dense_prefix(group_entries: &[Box<[SparseEntry]>], buf_words: usize) -> Vec<Box<[u32]>> {
    let mut prefixes = Vec::with_capacity(group_entries.len() + 1);
    let mut current = vec![0u32; buf_words];
    prefixes.push(current.clone().into_boxed_slice());
    for group in group_entries {
        for &entry in group.iter() {
            current[entry.word_idx() as usize] |= entry.mask();
        }
        prefixes.push(current.clone().into_boxed_slice());
    }
    prefixes
}

fn dense_group_threshold(buf_words: usize) -> usize {
    (buf_words / 4).max(64)
}

fn compute_heavy_group_dense(
    group_entries: &[Box<[SparseEntry]>],
    buf_words: usize,
    threshold: usize,
) -> Vec<Option<Box<[u32]>>> {
    group_entries
        .iter()
        .map(|entries| {
            if entries.len() <= threshold {
                return None;
            }
            let mut dense = vec![0u32; buf_words];
            for &entry in entries.iter() {
                dense[entry.word_idx() as usize] |= entry.mask();
            }
            Some(dense.into_boxed_slice())
        })
        .collect()
}

#[inline(always)]
fn or_sparse(out: &mut [u32], entries: &[SparseEntry]) {
    let out_ptr = out.as_mut_ptr();
    unsafe {
        for &entry in entries.iter() {
            let word = out_ptr.add(entry.word_idx() as usize);
            *word |= entry.mask();
        }
    }
}

#[inline(always)]
fn andnot_sparse(out: &mut [u32], entries: &[SparseEntry]) {
    let out_ptr = out.as_mut_ptr();
    unsafe {
        for &entry in entries.iter() {
            let word = out_ptr.add(entry.word_idx() as usize);
            *word &= !entry.mask();
        }
    }
}

#[inline(always)]
fn or_dense(out: &mut [u32], mask: &[u32]) {
    let n = out.len().min(mask.len());
    let n_pairs = n / 2;
    unsafe {
        let out_ptr = out.as_mut_ptr();
        let mask_ptr = mask.as_ptr();
        for i in 0..n_pairs {
            let offset = i * 2;
            let b = std::ptr::read_unaligned(out_ptr.add(offset) as *const u64);
            let m = std::ptr::read_unaligned(mask_ptr.add(offset) as *const u64);
            std::ptr::write_unaligned(out_ptr.add(offset) as *mut u64, b | m);
        }
        for i in (n_pairs * 2)..n {
            *out_ptr.add(i) |= *mask_ptr.add(i);
        }
    }
}

#[inline(always)]
fn andnot_dense(out: &mut [u32], mask: &[u32]) {
    let n = out.len().min(mask.len());
    let n_pairs = n / 2;
    unsafe {
        let out_ptr = out.as_mut_ptr();
        let mask_ptr = mask.as_ptr();
        for i in 0..n_pairs {
            let offset = i * 2;
            let b = std::ptr::read_unaligned(out_ptr.add(offset) as *const u64);
            let m = std::ptr::read_unaligned(mask_ptr.add(offset) as *const u64);
            std::ptr::write_unaligned(out_ptr.add(offset) as *mut u64, b & !m);
        }
        for i in (n_pairs * 2)..n {
            *out_ptr.add(i) &= !*mask_ptr.add(i);
        }
    }
}

#[inline(always)]
fn copy_dense(out: &mut [u32], dense: &[u32]) {
    let n = out.len().min(dense.len());
    out[..n].copy_from_slice(&dense[..n]);
}

#[inline(always)]
fn or_prefix_diff(out: &mut [u32], before: &[u32], after: &[u32]) {
    let n = out.len().min(before.len()).min(after.len());
    let n_pairs = n / 2;
    unsafe {
        let out_ptr = out.as_mut_ptr();
        let before_ptr = before.as_ptr();
        let after_ptr = after.as_ptr();
        for i in 0..n_pairs {
            let offset = i * 2;
            let b = std::ptr::read_unaligned(out_ptr.add(offset) as *const u64);
            let s = std::ptr::read_unaligned(before_ptr.add(offset) as *const u64);
            let e = std::ptr::read_unaligned(after_ptr.add(offset) as *const u64);
            std::ptr::write_unaligned(out_ptr.add(offset) as *mut u64, b | (e & !s));
        }
        for i in (n_pairs * 2)..n {
            *out_ptr.add(i) |= *after_ptr.add(i) & !*before_ptr.add(i);
        }
    }
}

#[inline(always)]
fn copy_prefix_diff(out: &mut [u32], before: &[u32], after: &[u32]) {
    let n = out.len().min(before.len()).min(after.len());
    let n_pairs = n / 2;
    unsafe {
        let out_ptr = out.as_mut_ptr();
        let before_ptr = before.as_ptr();
        let after_ptr = after.as_ptr();
        for i in 0..n_pairs {
            let offset = i * 2;
            let s = std::ptr::read_unaligned(before_ptr.add(offset) as *const u64);
            let e = std::ptr::read_unaligned(after_ptr.add(offset) as *const u64);
            std::ptr::write_unaligned(out_ptr.add(offset) as *mut u64, e & !s);
        }
        for i in (n_pairs * 2)..n {
            *out_ptr.add(i) = *after_ptr.add(i) & !*before_ptr.add(i);
        }
    }
}

#[inline(always)]
fn andnot_prefix_diff(out: &mut [u32], before: &[u32], after: &[u32]) {
    let n = out.len().min(before.len()).min(after.len());
    let n_pairs = n / 2;
    unsafe {
        let out_ptr = out.as_mut_ptr();
        let before_ptr = before.as_ptr();
        let after_ptr = after.as_ptr();
        for i in 0..n_pairs {
            let offset = i * 2;
            let b = std::ptr::read_unaligned(out_ptr.add(offset) as *const u64);
            let s = std::ptr::read_unaligned(before_ptr.add(offset) as *const u64);
            let e = std::ptr::read_unaligned(after_ptr.add(offset) as *const u64);
            std::ptr::write_unaligned(out_ptr.add(offset) as *mut u64, b & !(e & !s));
        }
        for i in (n_pairs * 2)..n {
            *out_ptr.add(i) &= !(*after_ptr.add(i) & !*before_ptr.add(i));
        }
    }
}

#[cfg(test)]
#[path = "mask_mapping/coordinate_tests.rs"]
mod coordinate_tests;
