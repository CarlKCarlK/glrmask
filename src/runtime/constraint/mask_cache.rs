//! Construction of token-mask caches in internal and original vocabulary coordinates.

use crate::runtime::artifact::Constraint;
use crate::runtime::artifact::DenseBufMaskRows;
use crate::runtime::artifact::DenseWeightBufMaskCache;
use crate::runtime::artifact::RangeFinalTokenSetCache;
use crate::runtime::artifact::InternalTokenBufMasks;
use crate::runtime::artifact::PackedInternalTokenBufMask;
use crate::runtime::artifact::SparseWeightBufMaskCache;
use range_set_blaze::RangeSetBlaze;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use rustc_hash::FxHashSet;
use std::collections::BTreeMap;
use std::sync::Arc;
use super::mask_replay::pack_internal_token_buf_entry;

#[derive(Default)]
pub(super) struct FinalTokenSetPlan {
    pub(super) eligible: RangeFinalTokenSetCache,
    pub(super) fallback: Vec<(usize, Arc<RangeSetBlaze<u32>>)>,
}


/// Finalization-local view of the parser-DWA token sets.  The final and
/// transition cache builders share it so finalization traverses the DWA once.
pub(super) struct WeightTokenSetInventory {
    pub(super) final_sets: Vec<(usize, Arc<RangeSetBlaze<u32>>)>,
    pub(super) transition_sets: FxHashMap<usize, Arc<RangeSetBlaze<u32>>>,
    pub(super) transition_word_spans: Option<FxHashMap<usize, u32>>,
}


pub(crate) struct InternalTokenMaskPrebuild {
    internal_token_buf_masks: Vec<InternalTokenBufMasks>,
}


impl InternalTokenMaskPrebuild {
    pub(crate) fn build(
        original_to_internal: &[u32],
        internal_to_tokens: &[Vec<u32>],
    ) -> Self {
        Self {
            internal_token_buf_masks: build_internal_token_buf_masks_from_maps(
                original_to_internal,
                internal_to_tokens,
            ),
        }
    }

    pub(crate) fn install(self, constraint: &mut Constraint) {
        debug_assert_eq!(
            self.internal_token_buf_masks.len(),
            constraint.internal_token_to_tokens.len(),
            "base token-mask prebuild must match final internal-token coordinate",
        );
        constraint.internal_token_buf_masks = self.internal_token_buf_masks;
    }
}


fn build_internal_token_buf_masks_from_maps(
    original_to_internal: &[u32],
    internal_to_tokens: &[Vec<u32>],
) -> Vec<InternalTokenBufMasks> {
    let grouped = std::env::var("GLRMASK_GROUPED_INTERNAL_TOKEN_MASKS")
        .map(|value| {
            let value = value.trim();
            value.is_empty() || (value != "0" && !value.eq_ignore_ascii_case("false"))
        })
        .unwrap_or(true);
    if !grouped && !original_to_internal.is_empty() {
        let mut masks = vec![Vec::<(u32, u32)>::new(); internal_to_tokens.len()];
        for (original, &internal) in original_to_internal.iter().enumerate() {
            if internal == u32::MAX {
                continue;
            }
            let Some(mask) = masks.get_mut(internal as usize) else {
                continue;
            };
            let word = original as u32 / 32;
            let bit = original as u32 % 32;
            if let Some((last_word, last_mask)) = mask.last_mut()
                && *last_word == word
            {
                *last_mask |= 1u32 << bit;
                continue;
            }
            mask.push((word, 1u32 << bit));
        }
        masks
    } else {
        internal_to_tokens
            .iter()
            .map(|originals| Constraint::build_internal_token_buf_mask(originals))
            .collect()
    }
}


pub(crate) struct TokenMaskCachePrebuild {
    mask_words: usize,
    internal_token_buf_masks: Vec<InternalTokenBufMasks>,
    word_group_buf_masks: Vec<Box<[u32]>>,
    pair_word_group_buf_masks: DenseBufMaskRows,
    quad_word_group_buf_masks: DenseBufMaskRows,
    super_word_group_buf_masks: DenseBufMaskRows,
    mega_word_group_buf_masks: DenseBufMaskRows,
    giga_word_group_buf_masks: DenseBufMaskRows,
    word_group_sparse_masks: Vec<InternalTokenBufMasks>,
    word_group_prefix_buf_masks: DenseBufMaskRows,
    word_group_sparse_prefix_entries: Vec<usize>,
    quad_group_sparse_masks: Vec<InternalTokenBufMasks>,
    quad_group_dense_masks: Vec<Option<Box<[u32]>>>,
    byte_group_sparse_masks: Vec<InternalTokenBufMasks>,
    byte_group_dense_masks: Vec<Option<Box<[u32]>>>,
    word_group_sparse_total_entries: usize,
    word_group_sparse_max_entries: usize,
    all_tokens_buf_mask: Box<[u32]>,
    heavy_token_dense_masks: Vec<Option<Box<[u32]>>>,
    internal_token_buf_flat: Box<[PackedInternalTokenBufMask]>,
    internal_token_buf_offsets: Box<[u32]>,
    total_internal_buf_cost: usize,
    heavy_token_indices: Vec<usize>,
    heavy_total_cost: usize,
    light_avg_cost_x256: usize,
    internal_token_buf_op_costs: Vec<usize>,
    word_group_buf_op_costs: Vec<usize>,
}


impl TokenMaskCachePrebuild {
    pub(crate) fn build(
        original_to_internal: &[u32],
        internal_to_tokens: &[Vec<u32>],
        mask_words: usize,
    ) -> Self {
        let internal_token_buf_masks = build_internal_token_buf_masks_from_maps(
            original_to_internal,
            internal_to_tokens,
        );

        let build_blocks = |block_size: usize| {
            if internal_token_buf_masks.is_empty() {
                return (Vec::new(), 0usize, 0usize);
            }
            let n_groups = if block_size == 64 {
                internal_token_buf_masks.len().div_ceil(block_size)
            } else {
                internal_token_buf_masks.len() / block_size
            };
            let mut groups = Vec::with_capacity(n_groups);
            for group_id in 0..n_groups {
                let group_start = group_id * block_size;
                let group_end =
                    (group_start + block_size).min(internal_token_buf_masks.len());
                let mut dense = vec![0u32; mask_words];
                let mut touched = Vec::<u32>::new();
                for token_masks in &internal_token_buf_masks[group_start..group_end] {
                    for &(word_idx, mask) in token_masks {
                        let slot = &mut dense[word_idx as usize];
                        if *slot == 0 {
                            touched.push(word_idx);
                        }
                        *slot |= mask;
                    }
                }
                touched.sort_unstable();
                groups.push(
                    touched
                        .into_iter()
                        .map(|word_idx| (word_idx, dense[word_idx as usize]))
                        .collect::<InternalTokenBufMasks>(),
                );
            }
            let total_entries = groups.iter().map(Vec::len).sum();
            let max_entries = groups.iter().map(Vec::len).max().unwrap_or(0);
            (groups, total_entries, max_entries)
        };

        let skip_small_group_caches = std::env::var("GLRMASK_SKIP_SMALL_GROUP_MASK_CACHES")
            .map(|value| {
                let value = value.trim();
                value.is_empty() || (value != "0" && !value.eq_ignore_ascii_case("false"))
            })
            .unwrap_or(true);
        let (word_group_sparse_masks, word_group_sparse_total_entries, word_group_sparse_max_entries) =
            build_blocks(64);
        let quad_group_sparse_masks = if skip_small_group_caches {
            Vec::new()
        } else {
            build_blocks(4).0
        };
        let byte_group_sparse_masks = if skip_small_group_caches {
            Vec::new()
        } else {
            build_blocks(8).0
        };

        let prefix_rows = word_group_sparse_masks.len() + 1;
        let mut prefix_dense = vec![0u32; mask_words];
        let word_group_prefix_buf_masks = if DenseBufMaskRows::prefer_flat(prefix_rows, mask_words) {
            let mut flat = Vec::with_capacity(prefix_rows.saturating_mul(mask_words));
            flat.extend_from_slice(&prefix_dense);
            for group in &word_group_sparse_masks {
                for &(word_idx, mask) in group {
                    prefix_dense[word_idx as usize] |= mask;
                }
                flat.extend_from_slice(&prefix_dense);
            }
            DenseBufMaskRows::from_flat(flat.into_boxed_slice(), prefix_rows, mask_words)
                .expect("word-group prefix dimensions should match construction")
        } else {
            let mut rows = Vec::with_capacity(prefix_rows);
            rows.push(prefix_dense.clone().into_boxed_slice());
            for group in &word_group_sparse_masks {
                for &(word_idx, mask) in group {
                    prefix_dense[word_idx as usize] |= mask;
                }
                rows.push(prefix_dense.clone().into_boxed_slice());
            }
            DenseBufMaskRows::from_rows(rows)
                .expect("word-group prefix rows should have uniform dimensions")
        };
        let mut word_group_sparse_prefix_entries =
            Vec::with_capacity(word_group_sparse_masks.len() + 1);
        let mut prefix_entries = 0usize;
        word_group_sparse_prefix_entries.push(0);
        for group in &word_group_sparse_masks {
            prefix_entries += group.len();
            word_group_sparse_prefix_entries.push(prefix_entries);
        }

        let build_dense_groups = |groups: &[InternalTokenBufMasks]| {
            groups
                .iter()
                .map(|group| {
                    if !Constraint::prefer_dense_buf_scan(mask_words, group.len()) {
                        return None;
                    }
                    let mut dense = vec![0u32; mask_words];
                    for &(word_idx, mask) in group {
                        dense[word_idx as usize] |= mask;
                    }
                    Some(dense.into_boxed_slice())
                })
                .collect::<Vec<_>>()
        };
        let quad_group_dense_masks = build_dense_groups(&quad_group_sparse_masks);
        let byte_group_dense_masks = build_dense_groups(&byte_group_sparse_masks);

        let build_sliding = |word_group_len: usize| {
            if word_group_prefix_buf_masks.is_empty() || word_group_len == 0 {
                return DenseBufMaskRows::default();
            }
            let n_word_groups = word_group_prefix_buf_masks.len() - 1;
            let n_windows = if n_word_groups < word_group_len {
                0
            } else {
                n_word_groups - word_group_len + 1
            };
            if n_windows == 0 {
                return DenseBufMaskRows::default();
            }
            let mut flat = vec![0u32; n_windows.saturating_mul(mask_words)];
            for word_group_start in 0..n_windows {
                let before = &word_group_prefix_buf_masks[word_group_start];
                let through = &word_group_prefix_buf_masks[word_group_start + word_group_len];
                // Internal-token groups partition original model-token ids, so
                // their output bits are disjoint. The OR-prefix therefore has
                // an exact inverse over a window: P[b] & !P[a].
                let row = &mut flat
                    [word_group_start * mask_words..(word_group_start + 1) * mask_words];
                for ((slot, &end), &start) in
                    row.iter_mut().zip(through.iter()).zip(before.iter())
                {
                    *slot = end & !start;
                }
            }
            DenseBufMaskRows::from_flat(flat.into_boxed_slice(), n_windows, mask_words)
                .expect("sliding dense-mask dimensions should match construction")
        };
        let pair_word_group_buf_masks = build_sliding(2);
        let quad_word_group_buf_masks = build_sliding(4);
        let super_word_group_buf_masks = build_sliding(8);
        let mega_word_group_buf_masks = build_sliding(16);
        let giga_word_group_buf_masks = build_sliding(32);

        let mut all_tokens_buf_mask = vec![0u32; mask_words];
        for group in &word_group_sparse_masks {
            for &(word_idx, mask) in group {
                all_tokens_buf_mask[word_idx as usize] |= mask;
            }
        }

        let threshold = mask_words / 4;
        let heavy_token_dense_masks = internal_token_buf_masks
            .iter()
            .map(|sparse| {
                if sparse.len() <= threshold || mask_words == 0 {
                    return None;
                }
                let mut dense = vec![0u32; mask_words];
                for &(word_idx, mask) in sparse {
                    dense[word_idx as usize] |= mask;
                }
                Some(dense.into_boxed_slice())
            })
            .collect::<Vec<_>>();
        let (internal_token_buf_flat, internal_token_buf_offsets) =
            Constraint::compute_flat_buf_masks(&internal_token_buf_masks);
        let total_internal_buf_cost = Constraint::compute_total_internal_buf_cost(
            &internal_token_buf_offsets,
            &heavy_token_dense_masks,
            mask_words,
        );
        let heavy_token_indices = heavy_token_dense_masks
            .iter()
            .enumerate()
            .filter_map(|(index, mask)| mask.is_some().then_some(index))
            .collect::<Vec<_>>();
        let heavy_total_cost = heavy_token_indices.len() * mask_words;
        let internal_token_buf_op_costs = Constraint::compute_internal_token_buf_op_costs(
            &internal_token_buf_offsets,
            &heavy_token_dense_masks,
            mask_words,
        );
        let word_group_buf_op_costs =
            Constraint::compute_word_group_buf_op_costs(&internal_token_buf_op_costs);
        let n_light = internal_token_buf_masks
            .len()
            .saturating_sub(heavy_token_indices.len());
        let light_total = total_internal_buf_cost.saturating_sub(heavy_total_cost);
        let light_avg_cost_x256 = if n_light > 0 {
            (light_total * 256) / n_light
        } else {
            0
        };

        Self {
            mask_words,
            internal_token_buf_masks,
            word_group_buf_masks: Vec::new(),
            pair_word_group_buf_masks,
            quad_word_group_buf_masks,
            super_word_group_buf_masks,
            mega_word_group_buf_masks,
            giga_word_group_buf_masks,
            word_group_sparse_masks,
            word_group_prefix_buf_masks,
            word_group_sparse_prefix_entries,
            quad_group_sparse_masks,
            quad_group_dense_masks,
            byte_group_sparse_masks,
            byte_group_dense_masks,
            word_group_sparse_total_entries,
            word_group_sparse_max_entries,
            all_tokens_buf_mask: all_tokens_buf_mask.into_boxed_slice(),
            heavy_token_dense_masks,
            internal_token_buf_flat,
            internal_token_buf_offsets,
            total_internal_buf_cost,
            heavy_token_indices,
            heavy_total_cost,
            light_avg_cost_x256,
            internal_token_buf_op_costs,
            word_group_buf_op_costs,
        }
    }

    pub(crate) fn matches_constraint(&self, constraint: &Constraint) -> bool {
        self.mask_words == constraint.body_mask_len()
            && self.internal_token_buf_masks.len() == constraint.internal_token_count()
    }

    pub(crate) fn install(self, constraint: &mut Constraint) {
        constraint.internal_token_buf_masks = self.internal_token_buf_masks;
        constraint.word_group_buf_masks = self.word_group_buf_masks;
        constraint.pair_word_group_buf_masks = self.pair_word_group_buf_masks;
        constraint.quad_word_group_buf_masks = self.quad_word_group_buf_masks;
        constraint.super_word_group_buf_masks = self.super_word_group_buf_masks;
        constraint.mega_word_group_buf_masks = self.mega_word_group_buf_masks;
        constraint.giga_word_group_buf_masks = self.giga_word_group_buf_masks;
        constraint.word_group_sparse_masks = self.word_group_sparse_masks;
        constraint.word_group_prefix_buf_masks = self.word_group_prefix_buf_masks;
        constraint.word_group_sparse_prefix_entries = self.word_group_sparse_prefix_entries;
        constraint.quad_group_sparse_masks = self.quad_group_sparse_masks;
        constraint.quad_group_dense_masks = self.quad_group_dense_masks;
        constraint.byte_group_sparse_masks = self.byte_group_sparse_masks;
        constraint.byte_group_dense_masks = self.byte_group_dense_masks;
        constraint.word_group_sparse_total_entries = self.word_group_sparse_total_entries;
        constraint.word_group_sparse_max_entries = self.word_group_sparse_max_entries;
        constraint.all_tokens_buf_mask = self.all_tokens_buf_mask;
        constraint.heavy_token_dense_masks = self.heavy_token_dense_masks;
        constraint.internal_token_buf_flat = self.internal_token_buf_flat;
        constraint.backed_internal_token_buf_flat = None;
        constraint.internal_token_buf_offsets = self.internal_token_buf_offsets;
        constraint.total_internal_buf_cost = self.total_internal_buf_cost;
        constraint.heavy_token_indices = self.heavy_token_indices;
        constraint.heavy_total_cost = self.heavy_total_cost;
        constraint.light_avg_cost_x256 = self.light_avg_cost_x256;
        constraint.internal_token_buf_op_costs = self.internal_token_buf_op_costs;
        constraint.word_group_buf_op_costs = self.word_group_buf_op_costs;
    }
}

impl Constraint {


    pub(super) fn compute_buf_masks(&self) -> Vec<InternalTokenBufMasks> {
        let Some(internal_token_to_tokens) = self.internal_token_groups() else {
            return Vec::new();
        };

        let grouped = std::env::var("GLRMASK_GROUPED_INTERNAL_TOKEN_MASKS")
            .map(|value| {
                let value = value.trim();
                value.is_empty() || (value != "0" && !value.eq_ignore_ascii_case("false"))
            })
            .unwrap_or(true);
        if !grouped && self.has_original_token_map() {
            let original_token_to_internal = self.original_token_map();
            let mut masks = vec![Vec::<(u32, u32)>::new(); internal_token_to_tokens.len()];
            for (original, &internal) in original_token_to_internal.iter().enumerate() {
                if internal == u32::MAX {
                    continue;
                }
                let internal = internal as usize;
                let Some(mask) = masks.get_mut(internal) else {
                    continue;
                };
                let word = original as u32 / 32;
                let bit = original as u32 % 32;
                if let Some((last_word, last_mask)) = mask.last_mut() {
                    if *last_word == word {
                        *last_mask |= 1u32 << bit;
                        continue;
                    }
                }
                mask.push((word, 1u32 << bit));
            }
            return masks;
        }

        if rayon::current_num_threads() == 1 {
            internal_token_to_tokens
                .iter()
                .map(|originals| Self::build_internal_token_buf_mask(originals))
                .collect()
        } else {
            internal_token_to_tokens
                .par_iter()
                .map(|originals| Self::build_internal_token_buf_mask(originals))
                .collect()
        }
    }

    pub(super) fn compute_token_block_sparse_masks(&self, block_size: usize) -> (Vec<InternalTokenBufMasks>, usize, usize) {
        let internal_count = self.internal_token_buf_mask_count();
        if internal_count == 0 {
            return (Vec::new(), 0, 0);
        }
        // Byte/quad subgroup caches are only consulted when the entire subgroup
        // is valid; a trailing partial subgroup can never be selected.  A
        // 64-token word group is different: runtime also uses its final partial
        // group under the word's exact valid-bit mask.
        let n_groups = if block_size == 64 {
            internal_count.div_ceil(block_size)
        } else {
            internal_count / block_size
        };
        let mask_words = self.body_mask_len();
        let build_group = |group_id: usize| {
                let group_start = group_id * block_size;
                let group_end = (group_start + block_size).min(internal_count);
                let mut dense = vec![0u32; mask_words];
                let mut touched = Vec::<u32>::new();
                for internal_token in group_start..group_end {
                    self.for_each_internal_token_buf_mask_entry(
                        internal_token,
                        |word_idx, mask| {
                        let slot = &mut dense[word_idx as usize];
                        if *slot == 0 {
                            touched.push(word_idx);
                        }
                        *slot |= mask;
                        },
                    );
                }
                touched.sort_unstable();
                touched
                    .into_iter()
                    .map(|word_idx| (word_idx, dense[word_idx as usize]))
                    .collect()
            };
        let groups: Vec<InternalTokenBufMasks> = if rayon::current_num_threads() == 1 {
            (0..n_groups).map(build_group).collect()
        } else {
            (0..n_groups).into_par_iter().map(build_group).collect()
        };
        let total_entries = groups.iter().map(Vec::len).sum();
        let max_entries = groups.iter().map(Vec::len).max().unwrap_or(0);
        (groups, total_entries, max_entries)
    }

    pub(super) fn compute_heavy_group_dense_masks(
        groups: &[InternalTokenBufMasks],
        mask_words: usize,
    ) -> Vec<Option<Box<[u32]>>> {
        if mask_words == 0 {
            return Vec::new();
        }
        let build_group = |group: &InternalTokenBufMasks| {
            if !Self::prefer_dense_buf_scan(mask_words, group.len()) {
                return None;
            }
            let mut dense = vec![0u32; mask_words];
            for &(word_idx, mask) in group {
                dense[word_idx as usize] |= mask;
            }
            Some(dense.into_boxed_slice())
        };
        if rayon::current_num_threads() == 1 {
            groups.iter().map(build_group).collect()
        } else {
            groups.par_iter().map(build_group).collect()
        }
    }

    pub(super) fn compute_sliding_word_group_dense_masks(&self, word_group_len: usize) -> DenseBufMaskRows {
        if self.word_group_prefix_buf_masks.is_empty() || word_group_len == 0 {
            return DenseBufMaskRows::default();
        }
        let n_word_groups = self.word_group_prefix_buf_masks.len() - 1;
        // Runtime only consults a dense sliding window when `remaining >=
        // word_group_len`.  The old builder nevertheless materialized a dense
        // mask for every start position, including truncated suffix windows and
        // entire window tiers wider than the constraint.  Those entries were
        // unreachable.
        let n_windows = if n_word_groups < word_group_len {
            0
        } else {
            n_word_groups - word_group_len + 1
        };
        if n_windows == 0 {
            return DenseBufMaskRows::default();
        }
        let row_len = self.word_group_prefix_buf_masks.row_len();
        if row_len == 0 {
            return DenseBufMaskRows::from_flat(Vec::new().into_boxed_slice(), n_windows, 0)
                .expect("zero-width sliding dense-mask dimensions should match");
        }
        let total_values = n_windows
            .checked_mul(row_len)
            .expect("sliding dense-mask dimensions fit usize");
        let mut flat = Vec::<std::mem::MaybeUninit<u32>>::with_capacity(total_values);
        // SAFETY: `MaybeUninit<u32>` may be uninitialized. Every slot is
        // written exactly once below before the allocation is reinterpreted.
        unsafe {
            flat.set_len(total_values);
        }
        let build_group = |word_group_start: usize, dense: &mut [std::mem::MaybeUninit<u32>]| {
            let before = &self.word_group_prefix_buf_masks[word_group_start];
            let through = &self.word_group_prefix_buf_masks[word_group_start + word_group_len];
            debug_assert_eq!(dense.len(), row_len);
            for ((slot, &end), &start) in dense.iter_mut().zip(through.iter()).zip(before.iter()) {
                slot.write(end & !start);
            }
        };
        if rayon::current_num_threads() == 1 {
            for (word_group_start, dense) in flat.chunks_mut(row_len).enumerate() {
                build_group(word_group_start, dense);
            }
        } else {
            flat.par_chunks_mut(row_len)
                .enumerate()
                .for_each(|(word_group_start, dense)| build_group(word_group_start, dense));
        }
        let flat = flat.into_boxed_slice();
        // SAFETY: all `MaybeUninit<u32>` elements were initialized above and
        // `MaybeUninit<u32>` has the same layout/alignment as `u32`.
        let flat = unsafe { Box::from_raw(Box::into_raw(flat) as *mut [u32]) };
        DenseBufMaskRows::from_flat(flat, n_windows, row_len)
            .expect("sliding dense-mask dimensions should match construction")
    }

    fn compute_all_sliding_word_group_dense_masks(
        &self,
    ) -> (
        DenseBufMaskRows,
        DenseBufMaskRows,
        DenseBufMaskRows,
        DenseBufMaskRows,
        DenseBufMaskRows,
    ) {
        if self.word_group_prefix_buf_masks.is_empty() {
            return Default::default();
        }
        let n_word_groups = self.word_group_prefix_buf_masks.len() - 1;
        let row_len = self.word_group_prefix_buf_masks.row_len();
        let lengths = [2usize, 4, 8, 16, 32];
        let windows = lengths.map(|len| {
            if n_word_groups >= len {
                n_word_groups - len + 1
            } else {
                0
            }
        });
        if row_len == 0 {
            let empty = |rows| {
                DenseBufMaskRows::from_flat(Vec::new().into_boxed_slice(), rows, 0)
                    .expect("zero-width sliding dense-mask dimensions should match")
            };
            return (
                empty(windows[0]),
                empty(windows[1]),
                empty(windows[2]),
                empty(windows[3]),
                empty(windows[4]),
            );
        }

        let allocate = |rows: usize| {
            let total = rows
                .checked_mul(row_len)
                .expect("sliding dense-mask dimensions fit usize");
            let mut values = Vec::<std::mem::MaybeUninit<u32>>::with_capacity(total);
            // SAFETY: MaybeUninit elements may be uninitialized; every slot is
            // written exactly once below before reinterpretation.
            unsafe {
                values.set_len(total);
            }
            values
        };
        let mut pair = allocate(windows[0]);
        let mut quad = allocate(windows[1]);
        let mut super_group = allocate(windows[2]);
        let mut mega = allocate(windows[3]);
        let mut giga = allocate(windows[4]);
        let ptrs = [
            pair.as_mut_ptr() as usize,
            quad.as_mut_ptr() as usize,
            super_group.as_mut_ptr() as usize,
            mega.as_mut_ptr() as usize,
            giga.as_mut_ptr() as usize,
        ];
        let build_start = |start: usize| {
            let before = &self.word_group_prefix_buf_masks[start];
            for word in 0..row_len {
                let base = before[word];
                for tier in 0..lengths.len() {
                    if start >= windows[tier] {
                        continue;
                    }
                    let end = self.word_group_prefix_buf_masks[start + lengths[tier]][word];
                    // SAFETY: each `start` owns one disjoint row in every tier;
                    // `word` selects one disjoint element within that row.
                    unsafe {
                        ((ptrs[tier] as *mut std::mem::MaybeUninit<u32>)
                            .add(start * row_len + word))
                        .write(std::mem::MaybeUninit::new(end & !base));
                    }
                }
            }
        };
        if rayon::current_num_threads() == 1 || n_word_groups < 8 {
            for start in 0..n_word_groups {
                build_start(start);
            }
        } else {
            (0..n_word_groups).into_par_iter().for_each(build_start);
        }

        let finish = |values: Vec<std::mem::MaybeUninit<u32>>, rows: usize| {
            let values = values.into_boxed_slice();
            // SAFETY: all slots corresponding to the `rows` output were
            // initialized above and MaybeUninit<u32> has u32's layout.
            let values = unsafe { Box::from_raw(Box::into_raw(values) as *mut [u32]) };
            DenseBufMaskRows::from_flat(values, rows, row_len)
                .expect("fused sliding dense-mask dimensions should match")
        };
        (
            finish(pair, windows[0]),
            finish(quad, windows[1]),
            finish(super_group, windows[2]),
            finish(mega, windows[3]),
            finish(giga, windows[4]),
        )
    }

    pub(crate) fn rebuild_sliding_word_group_dense_masks(&mut self) {
        let (pair, quad, super_group, mega, giga) =
            self.compute_all_sliding_word_group_dense_masks();
        self.pair_word_group_buf_masks = pair;
        self.quad_word_group_buf_masks = quad;
        self.super_word_group_buf_masks = super_group;
        self.mega_word_group_buf_masks = mega;
        self.giga_word_group_buf_masks = giga;
    }

    pub(crate) fn rebuild_word_group_prefix_and_sliding_dense_masks(&mut self) {
        self.word_group_prefix_buf_masks = self.compute_word_group_prefix_buf_masks();
        self.rebuild_sliding_word_group_dense_masks();
    }

    pub(super) fn compute_all_tokens_buf_mask(&self) -> Box<[u32]> {
        let buf_words = self.body_mask_len();
        let mut combined = vec![0u32; buf_words];
        for group in &self.word_group_sparse_masks {
            for &(word_idx, mask) in group {
                combined[word_idx as usize] |= mask;
            }
        }
        combined.into_boxed_slice()
    }

    pub(super) fn compute_word_group_prefix_buf_masks(&self) -> DenseBufMaskRows {
        let buf_words = self.body_mask_len();
        let rows = self.word_group_sparse_masks.len() + 1;
        let mut current = vec![0u32; buf_words];
        if DenseBufMaskRows::prefer_flat(rows, buf_words) {
            let mut flat = Vec::with_capacity(rows.saturating_mul(buf_words));
            flat.extend_from_slice(&current);
            for group in &self.word_group_sparse_masks {
                for &(word_idx, mask) in group {
                    current[word_idx as usize] |= mask;
                }
                flat.extend_from_slice(&current);
            }
            DenseBufMaskRows::from_flat(flat.into_boxed_slice(), rows, buf_words)
                .expect("word-group prefix dimensions should match construction")
        } else {
            let mut dense_rows = Vec::with_capacity(rows);
            dense_rows.push(current.clone().into_boxed_slice());
            for group in &self.word_group_sparse_masks {
                for &(word_idx, mask) in group {
                    current[word_idx as usize] |= mask;
                }
                dense_rows.push(current.clone().into_boxed_slice());
            }
            DenseBufMaskRows::from_rows(dense_rows)
                .expect("word-group prefix rows should have uniform dimensions")
        }
    }

    pub(super) fn compute_sparse_entry_prefix(groups: &[InternalTokenBufMasks]) -> Vec<usize> {
        let mut prefix = Vec::with_capacity(groups.len() + 1);
        let mut total = 0usize;
        prefix.push(0);
        for group in groups {
            total += group.len();
            prefix.push(total);
        }
        prefix
    }

    fn range_final_token_sets_enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| {
            // Retain the existing diagnostic switch for cache-plan comparisons.
            std::env::var("GLRMASK_DIRECT_SPARSE_WEIGHT_BUF_CACHE")
                .map(|value| {
                    let value = value.trim();
                    !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
                })
                .unwrap_or(true)
        })
    }





    fn token_set_cardinality_at_most(tokens: &RangeSetBlaze<u32>, limit: u64) -> bool {
        tokens.len() <= limit
    }

    pub(super) fn direct_sparse_work_prefix(
        internal_token_buf_masks: &[InternalTokenBufMasks],
        buf_words: usize,
    ) -> Vec<u64> {
        // Direct sparse replay scans each selected internal token and then ORs
        // its image into the original-token mask. Internal cardinality alone is
        // not a runtime-cost bound: one internal token can represent thousands
        // of original LLM token IDs. Mirror or_internal_token_to_buf_fast's
        // heavy-token choice and prefix-sum the worst-case replay work so a
        // RangeSetBlaze can be costed by ranges rather than token-by-token.
        let heavy_threshold = buf_words / 4;
        let mut prefix = Vec::with_capacity(internal_token_buf_masks.len() + 1);
        prefix.push(0u64);
        for mask in internal_token_buf_masks {
            let mask_work = if mask.len() > heavy_threshold {
                buf_words as u64
            } else {
                mask.len() as u64
            };
            let next = prefix
                .last()
                .copied()
                .unwrap_or_default()
                .saturating_add(mask_work);
            prefix.push(next);
        }
        prefix
    }

    fn direct_sparse_work_prefix_current(&self, buf_words: usize) -> Vec<u64> {
        let heavy_threshold = buf_words / 4;
        let count = self.internal_token_buf_mask_count();
        let mut prefix = Vec::with_capacity(count + 1);
        prefix.push(0u64);
        for internal_token in 0..count {
            let mask_len = self.internal_token_buf_mask_len(internal_token);
            let mask_work = if mask_len > heavy_threshold {
                buf_words as u64
            } else {
                mask_len as u64
            };
            prefix.push(
                prefix
                    .last()
                    .copied()
                    .unwrap_or_default()
                    .saturating_add(mask_work),
            );
        }
        prefix
    }

    pub(super) fn direct_sparse_expanded_work(tokens: &RangeSetBlaze<u32>, work_prefix: &[u64]) -> u64 {
        let n_internal = work_prefix.len().saturating_sub(1);
        let mut work = 0u64;
        for range in tokens.ranges() {
            let start = (*range.start() as usize).min(n_internal);
            let end_exclusive = (*range.end() as usize)
                .saturating_add(1)
                .min(n_internal);
            if start >= end_exclusive {
                continue;
            }
            work = work
                .saturating_add((end_exclusive - start) as u64)
                .saturating_add(
                    work_prefix[end_exclusive].saturating_sub(work_prefix[start]),
                );
        }
        work
    }

    fn direct_sparse_expanded_work_at_most(
        tokens: &RangeSetBlaze<u32>,
        work_prefix: &[u64],
        limit: u64,
    ) -> Option<u64> {
        let n_internal = work_prefix.len().saturating_sub(1);
        let mut work = 0u64;
        for range in tokens.ranges() {
            let start = (*range.start() as usize).min(n_internal);
            let end_exclusive = (*range.end() as usize)
                .saturating_add(1)
                .min(n_internal);
            if start >= end_exclusive {
                continue;
            }
            work = work
                .saturating_add((end_exclusive - start) as u64)
                .saturating_add(
                    work_prefix[end_exclusive].saturating_sub(work_prefix[start]),
                );
            if work > limit {
                return None;
            }
        }
        Some(work)
    }

    pub(super) fn weight_token_set_inventory_with_packed(
        &self,
        packed: Option<crate::automata::weighted::dwa::PackedDwaTokenSetInventory>,
    ) -> WeightTokenSetInventory {
        if self.packed_non_dwa_weights.is_some()
            && self
                .packed_parser_dwa
                .as_ref()
                .is_some_and(|dwa| dwa.materialized_token_sets_with_word_spans().is_none())
        {
            return WeightTokenSetInventory {
                final_sets: Vec::new(),
                transition_sets: FxHashMap::default(),
                transition_word_spans: None,
            };
        }
        let profile = std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some();
        let total_started = profile.then(std::time::Instant::now);
        #[derive(Default)]
        struct InventoryBatch {
            final_sets: FxHashMap<usize, Arc<RangeSetBlaze<u32>>>,
            transition_sets: FxHashMap<usize, Arc<RangeSetBlaze<u32>>>,
        }

        impl InventoryBatch {
            fn add_state(&mut self, state: &crate::automata::weighted::dwa::DWAState) {
                for (_, weight) in state.transitions.values() {
                    for (_tsid_range, token_set) in weight.raw_range_values() {
                        let key = Arc::as_ptr(token_set) as usize;
                        self.transition_sets
                            .entry(key)
                            .or_insert_with(|| Arc::clone(token_set));
                    }
                }
                let Some(final_weight) = &state.final_weight else {
                    return;
                };
                if final_weight.is_full() || final_weight.is_empty() {
                    return;
                }
                for (_tsid_range, token_set) in final_weight.raw_range_values() {
                    let key = Arc::as_ptr(token_set) as usize;
                    self.final_sets
                        .entry(key)
                        .or_insert_with(|| Arc::clone(token_set));
                }
            }

            fn merge_from(&mut self, other: Self) {
                self.final_sets.extend(other.final_sets);
                self.transition_sets.extend(other.transition_sets);
            }
        }

        let (mut inventory, transition_word_spans) = if let Some(packed_dwa) = &self.packed_parser_dwa {
            if let Some((sets, spans)) = packed_dwa.materialized_token_sets_with_word_spans() {
                let mut transition_sets = FxHashMap::default();
                let mut transition_spans = FxHashMap::default();
                transition_sets.reserve(sets.len());
                transition_spans.reserve(sets.len());
                for (tokens, &word_spans) in sets.iter().zip(spans) {
                    let key = Arc::as_ptr(tokens) as usize;
                    transition_sets.insert(key, Arc::clone(tokens));
                    transition_spans.insert(key, word_spans);
                }
                (
                    InventoryBatch {
                        final_sets: FxHashMap::default(),
                        transition_sets,
                    },
                    Some(transition_spans),
                )
            } else {
                // Loaded packed DWAs already execute directly from their flat
                // token ranges. Avoid rebuilding RangeSetBlaze solely to seed
                // pointer-keyed dense caches.
                (InventoryBatch::default(), None)
            }
        } else if let Some(packed) = packed {
            (
                InventoryBatch {
                    final_sets: packed.final_sets,
                    transition_sets: packed.transition_sets,
                },
                Some(packed.transition_word_spans),
            )
        } else if rayon::current_num_threads() > 1 && self.parser_dwa.states().len() >= 4_096 {
            (
                self.parser_dwa
                    .states()
                    .par_iter()
                    .fold(InventoryBatch::default, |mut batch, state| {
                        batch.add_state(state);
                        batch
                    })
                    .reduce(InventoryBatch::default, |mut left, right| {
                        left.merge_from(right);
                        left
                    }),
                None,
            )
        } else {
            let mut batch = InventoryBatch::default();
            for state in self.parser_dwa.states() {
                batch.add_state(state);
            }
            (batch, None)
        };

        let mut seen_final_weights = FxHashSet::<usize>::default();
        seen_final_weights.reserve(
            self.parser_top_accept
                .len()
                .saturating_add(self.parser_top_accept_parts.len())
                .saturating_add(self.direct_regular_l1_complete_by_terminal.len()),
        );
        let top_started = profile.then(std::time::Instant::now);
        for final_weight in self.parser_top_accept.values() {
            if final_weight.is_full() || final_weight.is_empty() {
                continue;
            }
            if !seen_final_weights.insert(final_weight.ptr_key()) {
                continue;
            }
            for (_tsid_range, token_set) in final_weight.raw_range_values() {
                let key = Arc::as_ptr(token_set) as usize;
                inventory
                    .final_sets
                    .entry(key)
                    .or_insert_with(|| Arc::clone(token_set));
            }
        }
        let top_ms = top_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

        let parts_started = profile.then(std::time::Instant::now);
        for final_weight in self.parser_top_accept_parts.values().flatten() {
            if final_weight.is_full() || final_weight.is_empty() {
                continue;
            }
            if !seen_final_weights.insert(final_weight.ptr_key()) {
                continue;
            }
            for (_tsid_range, token_set) in final_weight.raw_range_values() {
                let key = Arc::as_ptr(token_set) as usize;
                inventory
                    .final_sets
                    .entry(key)
                    .or_insert_with(|| Arc::clone(token_set));
            }
        }
        let parts_ms = parts_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

        let direct_started = profile.then(std::time::Instant::now);
        for final_weight in self.direct_regular_l1_complete_by_terminal.values() {
            if final_weight.is_full() || final_weight.is_empty() {
                continue;
            }
            if !seen_final_weights.insert(final_weight.ptr_key()) {
                continue;
            }
            for (_tsid_range, token_set) in final_weight.raw_range_values() {
                let key = Arc::as_ptr(token_set) as usize;
                inventory
                    .final_sets
                    .entry(key)
                    .or_insert_with(|| Arc::clone(token_set));
            }
        }
        let direct_ms = direct_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

        if let Some(started) = total_started {
            eprintln!(
                "[glrmask/profile][weight_token_inventory] top_ms={top_ms:.3} parts_ms={parts_ms:.3} direct_ms={direct_ms:.3} top_weights={} part_weights={} direct_weights={} final_sets={} transition_sets={} total_ms={:.3}",
                self.parser_top_accept.len(),
                self.parser_top_accept_parts.values().map(Vec::len).sum::<usize>(),
                self.direct_regular_l1_complete_by_terminal.len(),
                inventory.final_sets.len(),
                inventory.transition_sets.len(),
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }

        WeightTokenSetInventory {
            final_sets: inventory.final_sets.into_iter().collect(),
            transition_sets: inventory.transition_sets,
            transition_word_spans,
        }
    }

    pub(super) fn weight_token_set_inventory(&self) -> WeightTokenSetInventory {
        self.weight_token_set_inventory_with_packed(None)
    }

    /// Plan bounded final token sets without prebuilding dense/output masks.
    /// Fresh owned runtime weights union these sets before output expansion;
    /// artifact-backed weights may replay their intersection directly instead.
    /// The same conservative scan/expansion budget bounds both representations
    /// without adding compile-time cache work for the replay policy.
    pub(super) fn plan_final_token_sets(
        &self,
        final_token_sets: &[(usize, Arc<RangeSetBlaze<u32>>) ],
    ) -> FinalTokenSetPlan {
        if final_token_sets.is_empty() {
            return FinalTokenSetPlan::default();
        }
        let profile = std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some();
        let total_started = profile.then(std::time::Instant::now);
        let buf_words = self.body_mask_len();
        let direct_sparse = buf_words != 0
            && buf_words <= u16::MAX as usize
            && Self::range_final_token_sets_enabled()
            && self.final_mask_mapping.internal_len() == 0;
        let direct_token_limit = ((buf_words / 2).min(2048)) as u64;
        let direct_work_limit = direct_token_limit;
        let prefix_started = profile.then(std::time::Instant::now);
        let work_prefix = self.direct_sparse_work_prefix_current(buf_words);
        let prefix_ms = prefix_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let classify_started = profile.then(std::time::Instant::now);

        #[derive(Default)]
        struct SparseBatch {
            eligible: Vec<usize>,
            fallback: Vec<(usize, Arc<RangeSetBlaze<u32>>)>,
            small_cardinality: usize,
            expanded_work_max: u64,
        }

        impl SparseBatch {
            fn merge_from(&mut self, mut other: Self) {
                self.eligible.append(&mut other.eligible);
                self.fallback.append(&mut other.fallback);
                self.small_cardinality += other.small_cardinality;
                self.expanded_work_max = self.expanded_work_max.max(other.expanded_work_max);
            }
        }

        let build_one = |batch: &mut SparseBatch,
                         (key, token_set): &(usize, Arc<RangeSetBlaze<u32>>)| {
            if direct_sparse
                && Self::token_set_cardinality_at_most(token_set.as_ref(), direct_token_limit)
            {
                batch.small_cardinality += 1;
                if let Some(expanded_work) = Self::direct_sparse_expanded_work_at_most(
                    token_set.as_ref(),
                    &work_prefix,
                    direct_work_limit,
                ) {
                    batch.expanded_work_max = batch.expanded_work_max.max(expanded_work);
                    batch.eligible.push(*key);
                } else {
                    // Profiling only needs to indicate that at least one set
                    // exceeded the cap; avoid rescanning just to recover the
                    // exact over-limit total.
                    batch.expanded_work_max = batch
                        .expanded_work_max
                        .max(direct_work_limit.saturating_add(1));
                    batch.fallback.push((*key, Arc::clone(token_set)));
                }
            } else {
                batch.fallback.push((*key, Arc::clone(token_set)));
            }
        };

        let batch = if rayon::current_num_threads() == 1 {
            let mut batch = SparseBatch::default();
            for entry in final_token_sets {
                build_one(&mut batch, entry);
            }
            batch
        } else {
            final_token_sets
                .par_iter()
                .fold(SparseBatch::default, |mut batch, entry| {
                    build_one(&mut batch, entry);
                    batch
                })
                .reduce(SparseBatch::default, |mut left, right| {
                    left.merge_from(right);
                    left
                })
        };
        let classify_ms = classify_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

        if std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
            || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
        {
            let total_ranges = final_token_sets
                .iter()
                .map(|(_, tokens)| tokens.ranges().len())
                .sum::<usize>();
            let cardinality_fallback_sets = batch
                .eligible
                .len()
                .saturating_add(batch.fallback.len())
                .saturating_sub(batch.small_cardinality);
            let expanded_work_fallback_sets = batch
                .small_cardinality
                .saturating_sub(batch.eligible.len());
            eprintln!(
                "[glrmask/profile][runtime_direct_sparse] final_sets={} total_ranges={} direct_sets={} fallback_sets={} cardinality_fallback_sets={} expanded_work_fallback_sets={} expanded_work_max={} prefix_ms={:.3} classify_ms={:.3} total_ms={:.3}",
                batch.eligible.len() + batch.fallback.len(),
                total_ranges,
                batch.eligible.len(),
                batch.fallback.len(),
                cardinality_fallback_sets,
                expanded_work_fallback_sets,
                batch.expanded_work_max,
                prefix_ms,
                classify_ms,
                total_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
            );
        }
        FinalTokenSetPlan {
            eligible: batch.eligible.into_iter().collect(),
            fallback: batch.fallback,
        }
    }

    /// Build output masks only for final sets selected for materialization.
    /// Range-replayed sets need neither a dense internal mask nor an output mask.
    pub(super) fn compute_final_output_mask_caches(
        &self,
        mut prebuilt: FinalTokenSetPlan,
    ) -> (
        DenseWeightBufMaskCache,
        SparseWeightBufMaskCache,
        RangeFinalTokenSetCache,
    ) {
        #[derive(Default)]
        struct CacheBatch {
            dense: Vec<(usize, Box<[u32]>)>,
            sparse: Vec<(usize, Box<[(u32, u32)]>)>,
        }

        impl CacheBatch {
            fn merge_from(&mut self, mut other: Self) {
                self.dense.append(&mut other.dense);
                self.sparse.append(&mut other.sparse);
            }
        }

        let buf_words = self.body_mask_len();
        if buf_words == 0 {
            return (
                FxHashMap::default(),
                FxHashMap::default(),
                prebuilt.eligible,
            );
        }

        let can_store_sparse = buf_words <= u16::MAX as usize;
        let sparse_cost_limit = (buf_words / 2) as u64;
        let dense_masks = &self.weight_token_dense_masks;

        let build_one = |batch: &mut CacheBatch,
                         (key, _token_set): (usize, Arc<RangeSetBlaze<u32>>)| {
            let Some(dense) = dense_masks.get(&key) else {
                return;
            };
            let estimated_cost = self.estimate_internal_dense_to_buf_cost(dense);
            if estimated_cost == 0 {
                return;
            }

            let try_sparse = can_store_sparse && estimated_cost < sparse_cost_limit;
            let mut buf = vec![0u32; buf_words];
            self.or_internal_dense_to_buf(dense, &mut buf, true);

            if try_sparse {
                let sparse = Self::dense_buf_to_sparse_entries(&buf);
                if sparse.len() < buf_words / 2 {
                    batch.sparse.push((key, sparse));
                    return;
                }
            }

            batch.dense.push((key, buf.into_boxed_slice()));
        };

        let fallback = std::mem::take(&mut prebuilt.fallback);
        let batch = if rayon::current_num_threads() == 1 {
            let mut batch = CacheBatch::default();
            for entry in fallback {
                build_one(&mut batch, entry);
            }
            batch
        } else {
            fallback
                .into_par_iter()
                .fold(CacheBatch::default, |mut batch, entry| {
                    build_one(&mut batch, entry);
                    batch
                })
                .reduce(CacheBatch::default, |mut left, right| {
                    left.merge_from(right);
                    left
                })
        };

        (
            batch.dense.into_iter().collect(),
            batch.sparse.into_iter().collect(),
            prebuilt.eligible,
        )
    }

    fn dense_buf_to_sparse_entries(buf: &[u32]) -> Box<[(u32, u32)]> {
        buf.iter()
            .enumerate()
            .filter_map(|(idx, &word)| {
                if word == 0 {
                    None
                } else {
                    Some((idx as u32, word))
                }
            })
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }

    /// Build dense buf masks for internal tokens with many sparse entries.
    /// A token with >THRESHOLD entries benefits from a sequential 16KB scan
    /// instead of thousands of indexed read-modify-writes.
    pub(super) fn compute_heavy_token_dense_masks(&self) -> Vec<Option<Box<[u32]>>> {
        let buf_words = self.body_mask_len();
        if buf_words == 0 {
            return Vec::new();
        }
        // Threshold: use dense when sparse entries are large enough that a
        // sequential scan beats many indexed read-modify-writes.
        // Dense OR costs ~buf_words ops; sparse OR costs ~n_entries ops.
        // With buf in L1 cache (≤16KB), sparse random writes are fast,
        // so we only go dense when entries exceed half the buffer size.
        let threshold = buf_words / 4;
        let build = |internal_token: usize| {
            if self.internal_token_buf_mask_len(internal_token) > threshold {
                let mut dense = vec![0u32; buf_words];
                self.for_each_internal_token_buf_mask_entry(internal_token, |word_idx, mask| {
                    dense[word_idx as usize] |= mask;
                });
                Some(dense.into_boxed_slice())
            } else {
                None
            }
        };
        if rayon::current_num_threads() == 1 {
            (0..self.internal_token_buf_mask_count()).map(build).collect()
        } else {
            (0..self.internal_token_buf_mask_count()).into_par_iter().map(build).collect()
        }
    }

    pub(crate) fn rebuild_heavy_token_dense_masks(&mut self) {
        self.heavy_token_dense_masks = self.compute_heavy_token_dense_masks();
    }

    pub(crate) fn rebuild_heavy_and_sliding_token_mask_caches(&mut self) {
        let n_word_groups = self.word_group_prefix_buf_masks.len().saturating_sub(1);
        let buf_words = self.body_mask_len();
        let sliding_useful = [2usize, 4, 8, 16, 32].into_iter().any(|len| {
            n_word_groups >= len
                && (0..=n_word_groups - len).any(|start| {
                    Self::prefer_dense_buf_scan(
                        buf_words,
                        self.sparse_word_group_entries_in(start, len),
                    )
                })
        });
        let build_heavy = || self.compute_heavy_token_dense_masks();
        let build_sliding = || {
            if sliding_useful {
                self.compute_all_sliding_word_group_dense_masks()
            } else {
                (
                    DenseBufMaskRows::default(),
                    DenseBufMaskRows::default(),
                    DenseBufMaskRows::default(),
                    DenseBufMaskRows::default(),
                    DenseBufMaskRows::default(),
                )
            }
        };
        let (heavy, (pair, quad, super_group, mega, giga)) = if rayon::current_num_threads() == 1 {
            (build_heavy(), build_sliding())
        } else {
            rayon::join(build_heavy, build_sliding)
        };
        self.heavy_token_dense_masks = heavy;
        self.pair_word_group_buf_masks = pair;
        self.quad_word_group_buf_masks = quad;
        self.super_word_group_buf_masks = super_group;
        self.mega_word_group_buf_masks = mega;
        self.giga_word_group_buf_masks = giga;
    }

    pub(crate) fn rebuild_token_mask_cache_stats(&mut self) {
        self.word_group_sparse_prefix_entries =
            Self::compute_sparse_entry_prefix(&self.word_group_sparse_masks);
        self.word_group_sparse_total_entries =
            self.word_group_sparse_masks.iter().map(Vec::len).sum();
        self.word_group_sparse_max_entries = self
            .word_group_sparse_masks
            .iter()
            .map(Vec::len)
            .max()
            .unwrap_or(0);
        self.all_tokens_buf_mask = self
            .word_group_prefix_buf_masks
            .last()
            .map(Box::<[u32]>::from)
            .unwrap_or_default();

        let buf_len = self.body_mask_len();
        self.total_internal_buf_cost = Self::compute_total_internal_buf_cost(
            &self.internal_token_buf_offsets,
            &self.heavy_token_dense_masks,
            buf_len,
        );
        self.heavy_token_indices = self
            .heavy_token_dense_masks
            .iter()
            .enumerate()
            .filter_map(|(index, mask)| mask.is_some().then_some(index))
            .collect();
        self.heavy_total_cost = self.heavy_token_indices.len() * buf_len;
        self.internal_token_buf_op_costs = Self::compute_internal_token_buf_op_costs(
            &self.internal_token_buf_offsets,
            &self.heavy_token_dense_masks,
            buf_len,
        );
        self.word_group_buf_op_costs =
            Self::compute_word_group_buf_op_costs(&self.internal_token_buf_op_costs);
        let n_internal = self.internal_token_buf_offsets.len().saturating_sub(1);
        let n_light = n_internal.saturating_sub(self.heavy_token_indices.len());
        let light_total = self.total_internal_buf_cost.saturating_sub(self.heavy_total_cost);
        self.light_avg_cost_x256 = if n_light > 0 {
            (light_total * 256) / n_light
        } else {
            0
        };
    }

    /// Flatten all per-token sparse entries into a single contiguous array
    /// with an offset table. Improves cache locality during convert phase.
    pub(super) fn compute_flat_buf_masks(
        masks: &[InternalTokenBufMasks],
    ) -> (Box<[PackedInternalTokenBufMask]>, Box<[u32]>) {
        let total: usize = masks.iter().map(|m| m.len()).sum();
        let mut flat = Vec::with_capacity(total);
        let mut offsets = Vec::with_capacity(masks.len() + 1);
        for m in masks {
            offsets.push(flat.len() as u32);
            flat.extend(
                m.iter()
                    .map(|&(word_idx, mask)| pack_internal_token_buf_entry(word_idx, mask)),
            );
        }
        offsets.push(flat.len() as u32);
        (flat.into_boxed_slice(), offsets.into_boxed_slice())
    }

    /// Pre-compute total cost for all internal tokens (sum of entry counts,
    /// with heavy tokens counted at buf_len).
    pub(super) fn compute_total_internal_buf_cost(
        offsets: &[u32],
        heavy: &[Option<Box<[u32]>>],
        buf_len: usize,
    ) -> usize {
        let n_internal = if offsets.len() > 1 { offsets.len() - 1 } else { 0 };
        let mut total: usize = 0;
        for idx in 0..n_internal {
            if idx < heavy.len() && heavy[idx].is_some() {
                total += buf_len;
            } else {
                total += (offsets[idx + 1] - offsets[idx]) as usize;
            }
        }
        total
    }

    pub(super) fn compute_internal_token_buf_op_costs(
        offsets: &[u32],
        heavy: &[Option<Box<[u32]>>],
        buf_len: usize,
    ) -> Vec<usize> {
        let n_internal = if offsets.len() > 1 { offsets.len() - 1 } else { 0 };
        (0..n_internal)
            .map(|idx| {
                if idx < heavy.len() && heavy[idx].is_some() {
                    buf_len
                } else {
                    (offsets[idx + 1] - offsets[idx]) as usize
                }
            })
            .collect()
    }

    pub(super) fn compute_word_group_buf_op_costs(costs: &[usize]) -> Vec<usize> {
        costs
            .chunks(64)
            .map(|chunk| chunk.iter().copied().sum())
            .collect()
    }
    /// Build precomputed bitmask fragments for each internal token.
    pub(crate) fn build_buf_masks(&mut self) {
        self.internal_token_buf_masks = self.compute_buf_masks();
        self.word_group_buf_masks = Vec::new();
        let (word_group_sparse_masks, word_group_sparse_total_entries, word_group_sparse_max_entries) =
            self.compute_token_block_sparse_masks(64);
        let (quad_group_sparse_masks, _, _) = self.compute_token_block_sparse_masks(4);
        let (byte_group_sparse_masks, _, _) = self.compute_token_block_sparse_masks(8);
        self.word_group_sparse_masks = word_group_sparse_masks;
        self.word_group_prefix_buf_masks = self.compute_word_group_prefix_buf_masks();
        self.word_group_sparse_prefix_entries =
            Self::compute_sparse_entry_prefix(&self.word_group_sparse_masks);
        self.quad_group_sparse_masks = quad_group_sparse_masks;
        self.byte_group_sparse_masks = byte_group_sparse_masks;
        self.quad_group_dense_masks = Self::compute_heavy_group_dense_masks(
            &self.quad_group_sparse_masks,
            self.body_mask_len(),
        );
        self.byte_group_dense_masks = Self::compute_heavy_group_dense_masks(
            &self.byte_group_sparse_masks,
            self.body_mask_len(),
        );
        self.word_group_sparse_total_entries = word_group_sparse_total_entries;
        self.word_group_sparse_max_entries = word_group_sparse_max_entries;
        self.pair_word_group_buf_masks = self.compute_sliding_word_group_dense_masks(2);
        self.quad_word_group_buf_masks = self.compute_sliding_word_group_dense_masks(4);
        self.super_word_group_buf_masks = self.compute_sliding_word_group_dense_masks(8);
        self.mega_word_group_buf_masks = self.compute_sliding_word_group_dense_masks(16);
        self.giga_word_group_buf_masks = self.compute_sliding_word_group_dense_masks(32);
        self.all_tokens_buf_mask = self.compute_all_tokens_buf_mask();
        self.heavy_token_dense_masks = self.compute_heavy_token_dense_masks();
        let flat_ready = self.internal_token_buf_offsets.len()
            == self.internal_token_count().saturating_add(1)
            && self
                .internal_token_buf_offsets
                .last()
                .is_some_and(|&end| end as usize == self.internal_token_buf_flat_len());
        if !flat_ready {
            let (flat, offsets) = Self::compute_flat_buf_masks(&self.internal_token_buf_masks);
            self.internal_token_buf_flat = flat;
            self.backed_internal_token_buf_flat = None;
            self.internal_token_buf_offsets = offsets;
        }
        self.internal_token_buf_op_costs = Self::compute_internal_token_buf_op_costs(
            &self.internal_token_buf_offsets,
            &self.heavy_token_dense_masks,
            self.body_mask_len(),
        );
        self.word_group_buf_op_costs =
            Self::compute_word_group_buf_op_costs(&self.internal_token_buf_op_costs);
    }

    fn build_internal_token_buf_mask(originals: &[u32]) -> InternalTokenBufMasks {
        let mut result = Vec::<(u32, u32)>::new();
        let mut current_word = None::<u32>;
        let mut current_mask = 0u32;
        for &original in originals {
            let word = original / 32;
            let bit = original % 32;
            match current_word {
                None => {
                    current_word = Some(word);
                    current_mask = 1u32 << bit;
                }
                Some(current) if current == word => {
                    current_mask |= 1u32 << bit;
                }
                Some(current) if current < word => {
                    result.push((current, current_mask));
                    current_word = Some(word);
                    current_mask = 1u32 << bit;
                }
                Some(_) => {
                    return Self::build_internal_token_buf_mask_unsorted(originals);
                }
            }
        }
        if let Some(word) = current_word {
            result.push((word, current_mask));
        }
        result
    }

    fn build_internal_token_buf_mask_unsorted(originals: &[u32]) -> InternalTokenBufMasks {
        let mut word_map = BTreeMap::<u32, u32>::new();
        for &original in originals {
            let word = original / 32;
            let bit = original % 32;
            *word_map.entry(word).or_default() |= 1u32 << bit;
        }
        word_map.into_iter().collect()
    }
}
