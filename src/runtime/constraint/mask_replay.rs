//! Cost-directed replay of internal token sets into original-token mask buffers.

use crate::runtime::artifact::Constraint;
use crate::runtime::artifact::PackedInternalTokenBufMask;
use range_set_blaze::RangeSetBlaze;
use std::sync::Arc;
use super::DeltaReplayProfileStats;
use super::DenseToBufProfileStats;

/// Dense buf OR: `buf[i] |= mask[i]` for all i in min(buf.len(), mask.len()).
/// Processes u64 chunks for reduced loop overhead and better throughput.
#[inline(always)]
fn or_dense_buf(buf: &mut [u32], mask: &[u32]) {
    let n = buf.len().min(mask.len());
    let n_pairs = n / 2;
    unsafe {
        let buf_ptr = buf.as_mut_ptr();
        let mask_ptr = mask.as_ptr();
        for i in 0..n_pairs {
            let offset = i * 2;
            let b = std::ptr::read_unaligned(buf_ptr.add(offset) as *const u64);
            let m = std::ptr::read_unaligned(mask_ptr.add(offset) as *const u64);
            std::ptr::write_unaligned(buf_ptr.add(offset) as *mut u64, b | m);
        }
        for i in (n_pairs * 2)..n {
            *buf_ptr.add(i) |= *mask_ptr.add(i);
        }
    }
}


/// Dense buf AND-NOT: `buf[i] &= !mask[i]` for all i in min(buf.len(), mask.len()).
/// Processes u64 chunks for reduced loop overhead and better throughput.
#[inline(always)]
fn andnot_dense_buf(buf: &mut [u32], mask: &[u32]) {
    let n = buf.len().min(mask.len());
    let n_pairs = n / 2;
    unsafe {
        let buf_ptr = buf.as_mut_ptr();
        let mask_ptr = mask.as_ptr();
        for i in 0..n_pairs {
            let offset = i * 2;
            let b = std::ptr::read_unaligned(buf_ptr.add(offset) as *const u64);
            let m = std::ptr::read_unaligned(mask_ptr.add(offset) as *const u64);
            std::ptr::write_unaligned(buf_ptr.add(offset) as *mut u64, b & !m);
        }
        for i in (n_pairs * 2)..n {
            *buf_ptr.add(i) &= !*mask_ptr.add(i);
        }
    }
}


#[inline(always)]
fn copy_dense_buf(buf: &mut [u32], mask: &[u32]) {
    let n = buf.len().min(mask.len());
    unsafe {
        std::ptr::copy_nonoverlapping(mask.as_ptr(), buf.as_mut_ptr(), n);
    }
}


#[inline(always)]
pub(super) fn or_sparse_buf_entries(buf: &mut [u32], entries: &[(u32, u32)]) {
    for &(word_idx, mask) in entries {
        unsafe {
            let slot = buf.get_unchecked_mut(word_idx as usize);
            *slot |= mask;
        }
    }
}


#[inline(always)]
pub(super) fn andnot_sparse_buf_entries(buf: &mut [u32], entries: &[(u32, u32)]) {
    for &(word_idx, mask) in entries {
        unsafe {
            let slot = buf.get_unchecked_mut(word_idx as usize);
            *slot &= !mask;
        }
    }
}


#[inline(always)]
pub(super) fn pack_internal_token_buf_entry(word_idx: u32, mask: u32) -> PackedInternalTokenBufMask {
    PackedInternalTokenBufMask {
        word_idx,
        mask,
    }
}


#[inline(always)]
fn unpack_internal_token_buf_entry(entry: PackedInternalTokenBufMask) -> (u32, u32) {
    (entry.word_idx, entry.mask)
}


#[inline(always)]
fn or_packed_sparse_buf_entries(buf: &mut [u32], entries: &[PackedInternalTokenBufMask]) {
    for &entry in entries {
        let (word_idx, mask) = unpack_internal_token_buf_entry(entry);
        unsafe {
            let slot = buf.get_unchecked_mut(word_idx as usize);
            *slot |= mask;
        }
    }
}


#[inline(always)]
fn andnot_packed_sparse_buf_entries(buf: &mut [u32], entries: &[PackedInternalTokenBufMask]) {
    for &entry in entries {
        let (word_idx, mask) = unpack_internal_token_buf_entry(entry);
        unsafe {
            let slot = buf.get_unchecked_mut(word_idx as usize);
            *slot &= !mask;
        }
    }
}


#[inline(always)]
fn group_buf_mask_cost(sparse: &[(u32, u32)], dense: Option<&[u32]>) -> usize {
    dense.map_or(sparse.len(), <[u32]>::len)
}


#[inline(always)]
pub(super) fn or_group_buf_mask(
    buf: &mut [u32],
    sparse: &[(u32, u32)],
    dense: Option<&[u32]>,
) -> usize {
    if let Some(dense) = dense {
        or_dense_buf(buf, dense);
    } else {
        or_sparse_buf_entries(buf, sparse);
    }
    group_buf_mask_cost(sparse, dense)
}


#[inline(always)]
pub(super) fn andnot_group_buf_mask(
    buf: &mut [u32],
    sparse: &[(u32, u32)],
    dense: Option<&[u32]>,
) -> usize {
    if let Some(dense) = dense {
        andnot_dense_buf(buf, dense);
    } else {
        andnot_sparse_buf_entries(buf, sparse);
    }
    group_buf_mask_cost(sparse, dense)
}


#[inline(always)]
fn count_complement_subgroups(missing: u64, valid_mask: u64) -> (u32, u32, u32) {
    let mut byte_groups = 0u32;
    let mut nibble_groups = 0u32;
    let mut remaining_bits = 0u32;

    for byte_idx in 0..8 {
        let shift = byte_idx * 8;
        let byte_valid = ((valid_mask >> shift) & 0xff) as u8;
        if byte_valid == 0 {
            continue;
        }

        let byte_missing = ((missing >> shift) & 0xff) as u8;
        if byte_valid == 0xff && byte_missing == 0xff {
            byte_groups += 1;
            continue;
        }

        for nibble_idx in 0..2 {
            let nibble_shift = nibble_idx * 4;
            let nibble_valid = (byte_valid >> nibble_shift) & 0x0f;
            if nibble_valid == 0 {
                continue;
            }

            let nibble_missing = (byte_missing >> nibble_shift) & 0x0f;
            if nibble_valid == 0x0f && nibble_missing == 0x0f {
                nibble_groups += 1;
            } else {
                remaining_bits += nibble_missing.count_ones();
            }
        }
    }

    (byte_groups, nibble_groups, remaining_bits)
}

impl Constraint {


    pub(crate) fn internal_token_materialization_cost(&self, internal_token: usize) -> u64 {
        if internal_token < self.heavy_token_dense_masks.len()
            && self.heavy_token_dense_masks[internal_token].is_some()
        {
            return self.body_mask_len() as u64;
        }
        if internal_token + 1 >= self.internal_token_buf_offsets.len() {
            return 0;
        }
        (self.internal_token_buf_offsets[internal_token + 1]
            - self.internal_token_buf_offsets[internal_token]) as u64
    }

    pub(crate) fn estimate_internal_dense_to_buf_cost(&self, dense: &[u64]) -> u64 {
        if self.final_mask_mapping.internal_len() > 0 {
            return self.final_mask_mapping.estimate_dense_to_buf_cost(dense);
        }

        let all_mask = &self.all_tokens_buf_mask;
        let sparse_word_groups = &self.word_group_sparse_masks;
        let offsets = &self.internal_token_buf_offsets;
        let n_internal = if offsets.len() > 1 { offsets.len() - 1 } else { 0 };
        if n_internal == 0 || dense.is_empty() {
            return 0;
        }

        let n_set: usize = dense.iter().map(|w| w.count_ones() as usize).sum();
        let buf_len = self.body_mask_len();
        if n_set >= n_internal && !all_mask.is_empty() {
            return buf_len as u64;
        }
        if n_set == 0 {
            return 0;
        }

        let n_missing = n_internal - n_set;

        let dense_complement_fast_path = n_set.saturating_mul(5) >= n_internal.saturating_mul(4)
            && n_missing <= 128;

        if !all_mask.is_empty() && dense_complement_fast_path {
            let mut cost = buf_len as u64;
            for (wi, &w) in dense.iter().enumerate() {
                if wi * 64 >= n_internal {
                    break;
                }
                let remaining = n_internal - wi * 64;
                let valid_mask = if remaining >= 64 { !0u64 } else { (1u64 << remaining) - 1 };
                let missing = !w & valid_mask;
                if missing == 0 {
                    continue;
                }
                if missing == valid_mask {
                    if let Some(group_mask) = sparse_word_groups.get(wi) {
                        cost += group_mask.len() as u64;
                        continue;
                    }
                }
                cost += self.internal_bits_grouped_buf_op_cost(wi, missing, valid_mask, buf_len)
                    as u64;
            }
            cost
        } else {
            let mut cost = 0u64;
            for (wi, &w) in dense.iter().enumerate() {
                if wi * 64 >= n_internal {
                    break;
                }
                let remaining = n_internal - wi * 64;
                let valid_mask = if remaining >= 64 { !0u64 } else { (1u64 << remaining) - 1 };
                let valid_bits = w & valid_mask;
                if valid_bits == 0 {
                    continue;
                }
                if valid_bits == valid_mask {
                    if let Some(group_mask) = sparse_word_groups.get(wi) {
                        cost += group_mask.len() as u64;
                        continue;
                    }
                }
                cost += self.internal_bits_grouped_buf_op_cost(wi, valid_bits, valid_mask, buf_len)
                    as u64;
            }
            cost
        }
    }

    pub(crate) fn apply_internal_dense_delta_to_buf(
        &self,
        previous_dense: &[u64],
        current_dense: &[u64],
        buf: &mut [u32],
    ) -> DeltaReplayProfileStats {
        let mut stats = DeltaReplayProfileStats::default();
        let offsets = &self.internal_token_buf_offsets;
        let heavy = &self.heavy_token_dense_masks;
        let n_internal = if offsets.len() > 1 { offsets.len() - 1 } else { 0 };

        if n_internal == 0 {
            return stats;
        }

        let word_len = previous_dense.len().max(current_dense.len());
        for wi in 0..word_len {
            if wi * 64 >= n_internal {
                break;
            }
            let remaining = n_internal - wi * 64;
            let valid_mask = if remaining >= 64 { !0u64 } else { (1u64 << remaining) - 1 };
            let previous = previous_dense.get(wi).copied().unwrap_or(0) & valid_mask;
            let current = current_dense.get(wi).copied().unwrap_or(0) & valid_mask;

            let mut added = current & !previous;
            if added == valid_mask {
                if let Some(group_mask) = self.word_group_sparse_masks.get(wi) {
                    stats.added_word_group_hits += 1;
                    stats.added_word_group_entries += group_mask.len() as u64;
                    or_sparse_buf_entries(buf, group_mask);
                    continue;
                }
            }
            for byte_idx in 0..8 {
                let shift = byte_idx * 8;
                let byte_valid = (valid_mask >> shift) & 0xff;
                let byte_bits = (added >> shift) & 0xff;
                if byte_valid == 0xff && byte_bits == 0xff {
                    let group_idx = wi * 8 + byte_idx;
                    if let Some(group_mask) = self.byte_group_sparse_masks.get(group_idx) {
                        let dense_mask = self
                            .byte_group_dense_masks
                            .get(group_idx)
                            .and_then(Option::as_deref);
                        stats.added_byte_group_hits += 1;
                        stats.added_byte_group_entries +=
                            or_group_buf_mask(buf, group_mask, dense_mask) as u64;
                        added &= !(0xffu64 << shift);
                    }
                }
            }
            for quad_idx in 0..16 {
                let shift = quad_idx * 4;
                let quad_valid = (valid_mask >> shift) & 0x0f;
                let quad_bits = (added >> shift) & 0x0f;
                if quad_valid == 0x0f && quad_bits == 0x0f {
                    let group_idx = wi * 16 + quad_idx;
                    if let Some(group_mask) = self.quad_group_sparse_masks.get(group_idx) {
                        let dense_mask = self
                            .quad_group_dense_masks
                            .get(group_idx)
                            .and_then(Option::as_deref);
                        // DeltaReplayProfileStats historically combines byte
                        // and quad subgroup activity in these counters.
                        stats.added_byte_group_hits += 1;
                        stats.added_byte_group_entries +=
                            or_group_buf_mask(buf, group_mask, dense_mask) as u64;
                        added &= !(0x0fu64 << shift);
                    }
                }
            }
            while added != 0 {
                stats.added_token_iterations += 1;
                let bit = added.trailing_zeros() as usize;
                let internal_token = wi * 64 + bit;
                if internal_token < heavy.len() {
                    if let Some(ref dense_mask) = heavy[internal_token] {
                        stats.added_token_entries += dense_mask.len() as u64;
                        or_dense_buf(buf, dense_mask);
                        added &= added - 1;
                        continue;
                    }
                }
                let start = offsets[internal_token] as usize;
                let end = offsets[internal_token + 1] as usize;
                stats.added_token_entries += (end - start) as u64;
                self.or_internal_token_buf_range(start, end, buf);
                added &= added - 1;
            }

        }

        for wi in 0..word_len {
            if wi * 64 >= n_internal {
                break;
            }
            let remaining = n_internal - wi * 64;
            let valid_mask = if remaining >= 64 { !0u64 } else { (1u64 << remaining) - 1 };
            let previous = previous_dense.get(wi).copied().unwrap_or(0) & valid_mask;
            let current = current_dense.get(wi).copied().unwrap_or(0) & valid_mask;

            let mut removed = previous & !current;
            if removed == valid_mask {
                if let Some(group_mask) = self.word_group_sparse_masks.get(wi) {
                    stats.removed_word_group_hits += 1;
                    stats.removed_word_group_entries += group_mask.len() as u64;
                    andnot_sparse_buf_entries(buf, group_mask);
                    continue;
                }
            }
            for byte_idx in 0..8 {
                let shift = byte_idx * 8;
                let byte_valid = (valid_mask >> shift) & 0xff;
                let byte_bits = (removed >> shift) & 0xff;
                if byte_valid == 0xff && byte_bits == 0xff {
                    let group_idx = wi * 8 + byte_idx;
                    if let Some(group_mask) = self.byte_group_sparse_masks.get(group_idx) {
                        let dense_mask = self
                            .byte_group_dense_masks
                            .get(group_idx)
                            .and_then(Option::as_deref);
                        stats.removed_byte_group_hits += 1;
                        stats.removed_byte_group_entries +=
                            andnot_group_buf_mask(buf, group_mask, dense_mask) as u64;
                        removed &= !(0xffu64 << shift);
                    }
                }
            }
            for quad_idx in 0..16 {
                let shift = quad_idx * 4;
                let quad_valid = (valid_mask >> shift) & 0x0f;
                let quad_bits = (removed >> shift) & 0x0f;
                if quad_valid == 0x0f && quad_bits == 0x0f {
                    let group_idx = wi * 16 + quad_idx;
                    if let Some(group_mask) = self.quad_group_sparse_masks.get(group_idx) {
                        let dense_mask = self
                            .quad_group_dense_masks
                            .get(group_idx)
                            .and_then(Option::as_deref);
                        // See the added-side note above: these profile counters
                        // intentionally retain their historical combined shape.
                        stats.removed_byte_group_hits += 1;
                        stats.removed_byte_group_entries +=
                            andnot_group_buf_mask(buf, group_mask, dense_mask) as u64;
                        removed &= !(0x0fu64 << shift);
                    }
                }
            }
            while removed != 0 {
                stats.removed_token_iterations += 1;
                let bit = removed.trailing_zeros() as usize;
                let internal_token = wi * 64 + bit;
                if internal_token < heavy.len() {
                    if let Some(ref dense_mask) = heavy[internal_token] {
                        stats.removed_token_entries += dense_mask.len() as u64;
                        andnot_dense_buf(buf, dense_mask);
                        removed &= removed - 1;
                        continue;
                    }
                }
                let start = offsets[internal_token] as usize;
                let end = offsets[internal_token + 1] as usize;
                stats.removed_token_entries += (end - start) as u64;
                self.andnot_internal_token_buf_range(start, end, buf);
                removed &= removed - 1;
            }
        }

        stats
    }

    pub(crate) fn token_mask_caches_ready(&self) -> bool {
        let count = self.internal_token_count();
        self.internal_token_buf_mask_count() == count
            && self.internal_token_buf_offsets.len() == count.saturating_add(1)
            && self.word_group_sparse_masks.len() == count.div_ceil(64)
            && self.all_tokens_buf_mask.len() == self.body_mask_len()
    }

    #[inline]
    pub(super) fn internal_token_buf_mask_count(&self) -> usize {
        if self.internal_token_buf_offsets.len() > 1
            && self.internal_token_buf_offsets.last().copied().map(|end| end as usize)
                == Some(self.internal_token_buf_flat_len())
        {
            self.internal_token_buf_offsets.len() - 1
        } else {
            self.internal_token_buf_masks.len()
        }
    }

    #[inline]
    fn internal_token_buf_packed_slice(
        &self,
        internal_token: usize,
    ) -> Option<&[PackedInternalTokenBufMask]> {
        if let (Some(&start), Some(&end)) = (
            self.internal_token_buf_offsets.get(internal_token),
            self.internal_token_buf_offsets.get(internal_token + 1),
        ) && let Some(mask) = self.internal_token_buf_flat.get(start as usize..end as usize)
        {
            return Some(mask);
        }
        if let (Some(backed), Some(&start), Some(&end)) = (
            self.backed_internal_token_buf_flat.as_ref(),
            self.internal_token_buf_offsets.get(internal_token),
            self.internal_token_buf_offsets.get(internal_token + 1),
        ) {
            return backed.slice(start as usize, end as usize);
        }
        None
    }

    #[inline]
    pub(crate) fn internal_token_buf_flat_len(&self) -> usize {
        if !self.internal_token_buf_flat.is_empty() {
            self.internal_token_buf_flat.len()
        } else {
            self.backed_internal_token_buf_flat
                .as_ref()
                .map_or(0, |backed| backed.len())
        }
    }

    #[inline]
    pub(super) fn internal_token_buf_mask_len(&self, internal_token: usize) -> usize {
        if let (Some(&start), Some(&end)) = (
            self.internal_token_buf_offsets.get(internal_token),
            self.internal_token_buf_offsets.get(internal_token + 1),
        ) {
            let flat_len = self.internal_token_buf_flat_len();
            if start <= end && end as usize <= flat_len {
                return (end - start) as usize;
            }
        }
        self.internal_token_buf_packed_slice(internal_token)
            .map(<[PackedInternalTokenBufMask]>::len)
            .unwrap_or_else(|| {
                self.internal_token_buf_masks
                    .get(internal_token)
                    .map(Vec::len)
                    .unwrap_or(0)
            })
    }

    #[inline]
    pub(super) fn for_each_internal_token_buf_mask_entry(
        &self,
        internal_token: usize,
        mut visit: impl FnMut(u32, u32),
    ) {
        if let Some(mask) = self.internal_token_buf_packed_slice(internal_token) {
            for &entry in mask {
                let (word, bits) = unpack_internal_token_buf_entry(entry);
                visit(word, bits);
            }
            return;
        }
        if let (Some(backed), Some(&start), Some(&end)) = (
            self.backed_internal_token_buf_flat.as_ref(),
            self.internal_token_buf_offsets.get(internal_token),
            self.internal_token_buf_offsets.get(internal_token + 1),
        ) {
            backed.for_each_range(start as usize, end as usize, visit);
            return;
        }
        self.internal_token_buf_masks
            .get(internal_token)
            .into_iter()
            .flatten()
            .for_each(|&(word, bits)| visit(word, bits));
    }

    #[inline(always)]
    fn or_internal_token_buf_range(&self, start: usize, end: usize, buf: &mut [u32]) {
        if let Some(entries) = self.internal_token_buf_flat.get(start..end) {
            or_packed_sparse_buf_entries(buf, entries);
        } else if let Some(backed) = self.backed_internal_token_buf_flat.as_ref() {
            backed.for_each_range(start, end, |word_idx, mask| unsafe {
                let slot = buf.get_unchecked_mut(word_idx as usize);
                *slot |= mask;
            });
        }
    }

    #[inline(always)]
    fn andnot_internal_token_buf_range(&self, start: usize, end: usize, buf: &mut [u32]) {
        if let Some(entries) = self.internal_token_buf_flat.get(start..end) {
            andnot_packed_sparse_buf_entries(buf, entries);
        } else if let Some(backed) = self.backed_internal_token_buf_flat.as_ref() {
            backed.for_each_range(start, end, |word_idx, mask| unsafe {
                let slot = buf.get_unchecked_mut(word_idx as usize);
                *slot &= !mask;
            });
        }
    }

    /// Replay an already cached output mask only when its complete internal
    /// token set is admissible. A cache miss must leave the output untouched.
    #[inline(always)]
    pub(crate) fn try_replay_cached_final_mask(
        &self,
        dense: &[u64],
        token_set: &Arc<RangeSetBlaze<u32>>,
        buf: &mut [u32],
    ) -> bool {
        let key = Arc::as_ptr(token_set) as usize;
        let sparse_mask = self.weight_token_sparse_buf_masks.get(&key);
        let dense_mask = self.weight_token_buf_masks.get(&key);
        if sparse_mask.is_none() && dense_mask.is_none() {
            return false;
        }
        let Some(token_dense) = self.weight_token_dense_masks.get(&key) else {
            return false;
        };

        for (i, &token_word) in token_dense.iter().enumerate() {
            let dense_word = dense.get(i).copied().unwrap_or(0);
            if token_word & !dense_word != 0 {
                return false;
            }
        }

        if let Some(sparse_mask) = sparse_mask {
            or_sparse_buf_entries(buf, sparse_mask);
        } else {
            or_dense_buf(buf, dense_mask.expect("cache presence checked"));
        }
        true
    }

    pub(super) fn sparse_word_group_entries_in(&self, start: usize, len: usize) -> usize {
        let end = start + len;
        if end < self.word_group_sparse_prefix_entries.len() {
            self.word_group_sparse_prefix_entries[end] - self.word_group_sparse_prefix_entries[start]
        } else {
            self.word_group_sparse_masks[start..end]
                .iter()
                .map(Vec::len)
                .sum()
        }
    }

    #[inline(always)]
    pub(super) fn prefer_dense_buf_scan(buf_words: usize, sparse_entries: usize) -> bool {
        sparse_entries > buf_words / 4
    }

    #[inline(always)]
    fn or_word_group_prefix_diff_to_buf(&self, start: usize, end: usize, buf: &mut [u32]) {
        let Some(start_mask) = self.word_group_prefix_buf_masks.get(start) else {
            return;
        };
        let Some(end_mask) = self.word_group_prefix_buf_masks.get(end) else {
            return;
        };
        let n = buf.len().min(start_mask.len()).min(end_mask.len());
        let n_pairs = n / 2;
        unsafe {
            let buf_ptr = buf.as_mut_ptr();
            let start_ptr = start_mask.as_ptr();
            let end_ptr = end_mask.as_ptr();
            for i in 0..n_pairs {
                let offset = i * 2;
                let b = std::ptr::read_unaligned(buf_ptr.add(offset) as *const u64);
                let s = std::ptr::read_unaligned(start_ptr.add(offset) as *const u64);
                let e = std::ptr::read_unaligned(end_ptr.add(offset) as *const u64);
                std::ptr::write_unaligned(buf_ptr.add(offset) as *mut u64, b | (e & !s));
            }
            for i in (n_pairs * 2)..n {
                *buf_ptr.add(i) |= *end_ptr.add(i) & !*start_ptr.add(i);
            }
        }
    }

    #[inline(always)]
    fn andnot_word_group_prefix_diff_from_buf(&self, start: usize, end: usize, buf: &mut [u32]) {
        let Some(start_mask) = self.word_group_prefix_buf_masks.get(start) else {
            return;
        };
        let Some(end_mask) = self.word_group_prefix_buf_masks.get(end) else {
            return;
        };
        let n = buf.len().min(start_mask.len()).min(end_mask.len());
        let n_pairs = n / 2;
        unsafe {
            let buf_ptr = buf.as_mut_ptr();
            let start_ptr = start_mask.as_ptr();
            let end_ptr = end_mask.as_ptr();
            for i in 0..n_pairs {
                let offset = i * 2;
                let b = std::ptr::read_unaligned(buf_ptr.add(offset) as *const u64);
                let s = std::ptr::read_unaligned(start_ptr.add(offset) as *const u64);
                let e = std::ptr::read_unaligned(end_ptr.add(offset) as *const u64);
                std::ptr::write_unaligned(buf_ptr.add(offset) as *mut u64, b & !(e & !s));
            }
            for i in (n_pairs * 2)..n {
                *buf_ptr.add(i) &= !(*end_ptr.add(i) & !*start_ptr.add(i));
            }
        }
    }

    fn or_full_internal_word_run_to_buf<const PROFILE: bool>(
        &self,
        mut wi: usize,
        end: usize,
        buf: &mut [u32],
        stats: &mut DenseToBufProfileStats,
    ) {
        let run_len = end.saturating_sub(wi);
        if run_len > 0
            && end < self.word_group_prefix_buf_masks.len()
            && Self::prefer_dense_buf_scan(buf.len(), self.sparse_word_group_entries_in(wi, run_len))
        {
            if PROFILE {
                stats.normal_full_word_hits += run_len as u64;
                stats.group_or_sparse_entries += buf.len() as u64;
            }
            self.or_word_group_prefix_diff_to_buf(wi, end, buf);
            return;
        }

        while wi < end {
            let remaining = end - wi;
            let block = if remaining >= 32
                && self
                    .giga_word_group_buf_masks
                    .get(wi)
                    .is_some_and(|dense| Self::prefer_dense_buf_scan(dense.len(), self.sparse_word_group_entries_in(wi, 32)))
            {
                Some((32, &self.giga_word_group_buf_masks[wi]))
            } else if remaining >= 16
                && self
                    .mega_word_group_buf_masks
                    .get(wi)
                    .is_some_and(|dense| Self::prefer_dense_buf_scan(dense.len(), self.sparse_word_group_entries_in(wi, 16)))
            {
                Some((16, &self.mega_word_group_buf_masks[wi]))
            } else if remaining >= 8
                && self
                    .super_word_group_buf_masks
                    .get(wi)
                    .is_some_and(|dense| Self::prefer_dense_buf_scan(dense.len(), self.sparse_word_group_entries_in(wi, 8)))
            {
                Some((8, &self.super_word_group_buf_masks[wi]))
            } else if remaining >= 4
                && self
                    .quad_word_group_buf_masks
                    .get(wi)
                    .is_some_and(|dense| Self::prefer_dense_buf_scan(dense.len(), self.sparse_word_group_entries_in(wi, 4)))
            {
                Some((4, &self.quad_word_group_buf_masks[wi]))
            } else if remaining >= 2
                && self
                    .pair_word_group_buf_masks
                    .get(wi)
                    .is_some_and(|dense| Self::prefer_dense_buf_scan(dense.len(), self.sparse_word_group_entries_in(wi, 2)))
            {
                Some((2, &self.pair_word_group_buf_masks[wi]))
            } else {
                None
            };

            if let Some((block_len, dense_mask)) = block {
                if PROFILE {
                    stats.normal_full_word_hits += block_len as u64;
                    stats.group_or_sparse_entries += dense_mask.len() as u64;
                }
                or_dense_buf(buf, dense_mask);
                wi += block_len;
                continue;
            }

            if let Some(group_mask) = self.word_group_sparse_masks.get(wi) {
                if PROFILE {
                    stats.normal_full_word_hits += 1;
                }
                if Self::prefer_dense_buf_scan(buf.len(), group_mask.len())
                    && wi + 1 < self.word_group_prefix_buf_masks.len()
                {
                    if PROFILE {
                        stats.group_or_sparse_entries += buf.len() as u64;
                    }
                    self.or_word_group_prefix_diff_to_buf(wi, wi + 1, buf);
                } else {
                    if PROFILE {
                        stats.group_or_sparse_entries += group_mask.len() as u64;
                    }
                    or_sparse_buf_entries(buf, group_mask);
                }
            }
            wi += 1;
        }
    }

    fn andnot_full_internal_word_run_from_buf<const PROFILE: bool>(
        &self,
        mut wi: usize,
        end: usize,
        buf: &mut [u32],
        stats: &mut DenseToBufProfileStats,
    ) {
        let run_len = end.saturating_sub(wi);
        if run_len > 0
            && end < self.word_group_prefix_buf_masks.len()
            && Self::prefer_dense_buf_scan(buf.len(), self.sparse_word_group_entries_in(wi, run_len))
        {
            if PROFILE {
                stats.complement_full_word_hits += run_len as u64;
                stats.group_andnot_sparse_entries += buf.len() as u64;
            }
            self.andnot_word_group_prefix_diff_from_buf(wi, end, buf);
            return;
        }

        while wi < end {
            let remaining = end - wi;
            let block = if remaining >= 32
                && self
                    .giga_word_group_buf_masks
                    .get(wi)
                    .is_some_and(|dense| Self::prefer_dense_buf_scan(dense.len(), self.sparse_word_group_entries_in(wi, 32)))
            {
                Some((32, &self.giga_word_group_buf_masks[wi]))
            } else if remaining >= 16
                && self
                    .mega_word_group_buf_masks
                    .get(wi)
                    .is_some_and(|dense| Self::prefer_dense_buf_scan(dense.len(), self.sparse_word_group_entries_in(wi, 16)))
            {
                Some((16, &self.mega_word_group_buf_masks[wi]))
            } else if remaining >= 8
                && self
                    .super_word_group_buf_masks
                    .get(wi)
                    .is_some_and(|dense| Self::prefer_dense_buf_scan(dense.len(), self.sparse_word_group_entries_in(wi, 8)))
            {
                Some((8, &self.super_word_group_buf_masks[wi]))
            } else if remaining >= 4
                && self
                    .quad_word_group_buf_masks
                    .get(wi)
                    .is_some_and(|dense| Self::prefer_dense_buf_scan(dense.len(), self.sparse_word_group_entries_in(wi, 4)))
            {
                Some((4, &self.quad_word_group_buf_masks[wi]))
            } else if remaining >= 2
                && self
                    .pair_word_group_buf_masks
                    .get(wi)
                    .is_some_and(|dense| Self::prefer_dense_buf_scan(dense.len(), self.sparse_word_group_entries_in(wi, 2)))
            {
                Some((2, &self.pair_word_group_buf_masks[wi]))
            } else {
                None
            };

            if let Some((block_len, dense_mask)) = block {
                if PROFILE {
                    stats.complement_full_word_hits += block_len as u64;
                    stats.group_andnot_sparse_entries += dense_mask.len() as u64;
                }
                andnot_dense_buf(buf, dense_mask);
                wi += block_len;
                continue;
            }

            if let Some(group_mask) = self.word_group_sparse_masks.get(wi) {
                if PROFILE {
                    stats.complement_full_word_hits += 1;
                }
                if Self::prefer_dense_buf_scan(buf.len(), group_mask.len())
                    && wi + 1 < self.word_group_prefix_buf_masks.len()
                {
                    if PROFILE {
                        stats.group_andnot_sparse_entries += buf.len() as u64;
                    }
                    self.andnot_word_group_prefix_diff_from_buf(wi, wi + 1, buf);
                } else {
                    if PROFILE {
                        stats.group_andnot_sparse_entries += group_mask.len() as u64;
                    }
                    andnot_sparse_buf_entries(buf, group_mask);
                }
            }
            wi += 1;
        }
    }

    #[inline(always)]
    fn internal_token_buf_op_cost(&self, internal_token: usize, buf_len: usize) -> usize {
        if let Some(&cost) = self.internal_token_buf_op_costs.get(internal_token) {
            return cost;
        }
        if internal_token < self.heavy_token_dense_masks.len()
            && self.heavy_token_dense_masks[internal_token].is_some()
        {
            buf_len
        } else {
            (self.internal_token_buf_offsets[internal_token + 1]
                - self.internal_token_buf_offsets[internal_token]) as usize
        }
    }

    #[inline(always)]
    fn internal_bits_buf_op_cost(&self, wi: usize, mut bits: u64, buf_len: usize) -> usize {
        let mut cost = 0usize;
        while bits != 0 {
            let bit = bits.trailing_zeros() as usize;
            let internal_token = wi * 64 + bit;
            cost += self.internal_token_buf_op_cost(internal_token, buf_len);
            bits &= bits - 1;
        }
        cost
    }

    #[inline(always)]
    pub(crate) fn internal_bits_grouped_buf_op_cost(
        &self,
        wi: usize,
        mut bits: u64,
        valid_mask: u64,
        buf_len: usize,
    ) -> usize {
        let mut cost = 0usize;
        for byte_idx in 0..8 {
            let shift = byte_idx * 8;
            let byte_valid = (valid_mask >> shift) & 0xff;
            let byte_bits = (bits >> shift) & 0xff;
            if byte_valid == 0xff && byte_bits == 0xff {
                let group_idx = wi * 8 + byte_idx;
                if let Some(group_mask) = self.byte_group_sparse_masks.get(group_idx) {
                    let dense_mask = self
                        .byte_group_dense_masks
                        .get(group_idx)
                        .and_then(Option::as_deref);
                    cost += group_buf_mask_cost(group_mask, dense_mask);
                    bits &= !(0xffu64 << shift);
                }
            }
        }

        for quad_idx in 0..16 {
            let shift = quad_idx * 4;
            let quad_valid = (valid_mask >> shift) & 0x0f;
            let quad_bits = (bits >> shift) & 0x0f;
            if quad_valid == 0x0f && quad_bits == 0x0f {
                let group_idx = wi * 16 + quad_idx;
                if let Some(group_mask) = self.quad_group_sparse_masks.get(group_idx) {
                    let dense_mask = self
                        .quad_group_dense_masks
                        .get(group_idx)
                        .and_then(Option::as_deref);
                    cost += group_buf_mask_cost(group_mask, dense_mask);
                    bits &= !(0x0fu64 << shift);
                }
            }
        }

        cost + self.internal_bits_buf_op_cost(wi, bits, buf_len)
    }

    fn full_internal_word_run_buf_op_cost(
        &self,
        mut wi: usize,
        end: usize,
        buf_len: usize,
    ) -> usize {
        let run_len = end.saturating_sub(wi);
        if run_len > 0
            && end < self.word_group_prefix_buf_masks.len()
            && Self::prefer_dense_buf_scan(
                buf_len,
                self.sparse_word_group_entries_in(wi, run_len),
            )
        {
            return buf_len;
        }

        let mut cost = 0usize;
        while wi < end {
            let remaining = end - wi;
            let block = if remaining >= 32
                && self
                    .giga_word_group_buf_masks
                    .get(wi)
                    .is_some_and(|dense| {
                        Self::prefer_dense_buf_scan(
                            dense.len(),
                            self.sparse_word_group_entries_in(wi, 32),
                        )
                    })
            {
                Some((32, self.giga_word_group_buf_masks[wi].len()))
            } else if remaining >= 16
                && self
                    .mega_word_group_buf_masks
                    .get(wi)
                    .is_some_and(|dense| {
                        Self::prefer_dense_buf_scan(
                            dense.len(),
                            self.sparse_word_group_entries_in(wi, 16),
                        )
                    })
            {
                Some((16, self.mega_word_group_buf_masks[wi].len()))
            } else if remaining >= 8
                && self
                    .super_word_group_buf_masks
                    .get(wi)
                    .is_some_and(|dense| {
                        Self::prefer_dense_buf_scan(
                            dense.len(),
                            self.sparse_word_group_entries_in(wi, 8),
                        )
                    })
            {
                Some((8, self.super_word_group_buf_masks[wi].len()))
            } else if remaining >= 4
                && self
                    .quad_word_group_buf_masks
                    .get(wi)
                    .is_some_and(|dense| {
                        Self::prefer_dense_buf_scan(
                            dense.len(),
                            self.sparse_word_group_entries_in(wi, 4),
                        )
                    })
            {
                Some((4, self.quad_word_group_buf_masks[wi].len()))
            } else if remaining >= 2
                && self
                    .pair_word_group_buf_masks
                    .get(wi)
                    .is_some_and(|dense| {
                        Self::prefer_dense_buf_scan(
                            dense.len(),
                            self.sparse_word_group_entries_in(wi, 2),
                        )
                    })
            {
                Some((2, self.pair_word_group_buf_masks[wi].len()))
            } else {
                None
            };

            if let Some((block_len, dense_cost)) = block {
                cost = cost.saturating_add(dense_cost);
                wi += block_len;
                continue;
            }

            if let Some(group_mask) = self.word_group_sparse_masks.get(wi) {
                cost = cost.saturating_add(
                    if Self::prefer_dense_buf_scan(buf_len, group_mask.len())
                        && wi + 1 < self.word_group_prefix_buf_masks.len()
                    {
                        buf_len
                    } else {
                        group_mask.len()
                    },
                );
            }
            wi += 1;
        }
        cost
    }

    fn internal_dense_buf_replay_cost(
        &self,
        dense: &[u64],
        n_internal: usize,
        buf_len: usize,
        complement: bool,
    ) -> usize {
        let mut cost = 0usize;
        let mut wi = 0usize;
        while wi < dense.len() && wi * 64 < n_internal {
            let remaining = n_internal - wi * 64;
            let valid_mask = if remaining >= 64 {
                !0u64
            } else {
                (1u64 << remaining) - 1
            };
            let bits = if complement {
                !dense[wi] & valid_mask
            } else {
                dense[wi] & valid_mask
            };
            if bits == 0 {
                wi += 1;
                continue;
            }
            if bits == valid_mask {
                let run_start = wi;
                wi += 1;
                while wi < dense.len() && wi * 64 < n_internal {
                    let remaining = n_internal - wi * 64;
                    if remaining < 64
                        || if complement {
                            dense[wi] != 0
                        } else {
                            dense[wi] != !0u64
                        }
                    {
                        break;
                    }
                    wi += 1;
                }
                cost = cost.saturating_add(self.full_internal_word_run_buf_op_cost(
                    run_start,
                    wi,
                    buf_len,
                ));
                continue;
            }
            cost = cost.saturating_add(self.internal_bits_grouped_buf_op_cost(
                wi,
                bits,
                valid_mask,
                buf_len,
            ));
            wi += 1;
        }
        cost
    }

    #[inline(always)]
    pub(super) fn or_internal_token_to_buf_fast<const PROFILE: bool>(
        &self,
        internal_token: usize,
        buf: &mut [u32],
        stats_entries: &mut u64,
    ) {
        if internal_token < self.heavy_token_dense_masks.len() {
            if let Some(ref dense_mask) = self.heavy_token_dense_masks[internal_token] {
                if PROFILE {
                    *stats_entries += dense_mask.len() as u64;
                }
                or_dense_buf(buf, dense_mask);
                return;
            }
        }
        let start = self.internal_token_buf_offsets[internal_token] as usize;
        let end = self.internal_token_buf_offsets[internal_token + 1] as usize;
        if PROFILE {
            *stats_entries += end.saturating_sub(start) as u64;
        }
        self.or_internal_token_buf_range(start, end, buf);
    }

    #[inline(always)]
    fn andnot_internal_token_from_buf_fast<const PROFILE: bool>(
        &self,
        internal_token: usize,
        buf: &mut [u32],
        stats_entries: &mut u64,
    ) {
        if internal_token < self.heavy_token_dense_masks.len() {
            if let Some(ref dense_mask) = self.heavy_token_dense_masks[internal_token] {
                if PROFILE {
                    *stats_entries += dense_mask.len() as u64;
                }
                andnot_dense_buf(buf, dense_mask);
                return;
            }
        }
        let start = self.internal_token_buf_offsets[internal_token] as usize;
        let end = self.internal_token_buf_offsets[internal_token + 1] as usize;
        if PROFILE {
            *stats_entries += end.saturating_sub(start) as u64;
        }
        self.andnot_internal_token_buf_range(start, end, buf);
    }

    fn or_internal_bits_to_buf_grouped<const PROFILE: bool>(
        &self,
        wi: usize,
        mut bits: u64,
        valid_mask: u64,
        buf: &mut [u32],
        stats: &mut DenseToBufProfileStats,
    ) {
        for byte_idx in 0..8 {
            let shift = byte_idx * 8;
            let byte_valid = (valid_mask >> shift) & 0xff;
            let byte_bits = (bits >> shift) & 0xff;
            if byte_valid == 0xff && byte_bits == 0xff {
                let group_idx = wi * 8 + byte_idx;
                if let Some(group_mask) = self.byte_group_sparse_masks.get(group_idx) {
                    let dense_mask = self
                        .byte_group_dense_masks
                        .get(group_idx)
                        .and_then(Option::as_deref);
                    let replay_cost = or_group_buf_mask(buf, group_mask, dense_mask);
                    if PROFILE {
                        stats.group_or_sparse_entries += replay_cost as u64;
                    }
                    bits &= !(0xffu64 << shift);
                }
            }
        }

        for quad_idx in 0..16 {
            let shift = quad_idx * 4;
            let quad_valid = (valid_mask >> shift) & 0x0f;
            let quad_bits = (bits >> shift) & 0x0f;
            if quad_valid == 0x0f && quad_bits == 0x0f {
                let group_idx = wi * 16 + quad_idx;
                if let Some(group_mask) = self.quad_group_sparse_masks.get(group_idx) {
                    let dense_mask = self
                        .quad_group_dense_masks
                        .get(group_idx)
                        .and_then(Option::as_deref);
                    let replay_cost = or_group_buf_mask(buf, group_mask, dense_mask);
                    if PROFILE {
                        stats.group_or_sparse_entries += replay_cost as u64;
                    }
                    bits &= !(0x0fu64 << shift);
                }
            }
        }

        while bits != 0 {
            if PROFILE {
                stats.normal_token_iterations += 1;
            }
            let bit = bits.trailing_zeros() as usize;
            let internal_token = wi * 64 + bit;
            if internal_token < self.internal_token_buf_offsets.len().saturating_sub(1) {
                self.or_internal_token_to_buf_fast::<PROFILE>(
                    internal_token,
                    buf,
                    &mut stats.normal_sparse_entries,
                );
            }
            bits &= bits - 1;
        }
    }

    fn andnot_internal_bits_from_buf_grouped<const PROFILE: bool>(
        &self,
        wi: usize,
        mut bits: u64,
        valid_mask: u64,
        buf: &mut [u32],
        stats: &mut DenseToBufProfileStats,
    ) {
        for byte_idx in 0..8 {
            let shift = byte_idx * 8;
            let byte_valid = (valid_mask >> shift) & 0xff;
            let byte_bits = (bits >> shift) & 0xff;
            if byte_valid == 0xff && byte_bits == 0xff {
                let group_idx = wi * 8 + byte_idx;
                if let Some(group_mask) = self.byte_group_sparse_masks.get(group_idx) {
                    let dense_mask = self
                        .byte_group_dense_masks
                        .get(group_idx)
                        .and_then(Option::as_deref);
                    let replay_cost = andnot_group_buf_mask(buf, group_mask, dense_mask);
                    if PROFILE {
                        stats.complement_full_byte_groups += 1;
                        stats.group_andnot_sparse_entries += replay_cost as u64;
                    }
                    bits &= !(0xffu64 << shift);
                }
            }
        }

        for quad_idx in 0..16 {
            let shift = quad_idx * 4;
            let quad_valid = (valid_mask >> shift) & 0x0f;
            let quad_bits = (bits >> shift) & 0x0f;
            if quad_valid == 0x0f && quad_bits == 0x0f {
                let group_idx = wi * 16 + quad_idx;
                if let Some(group_mask) = self.quad_group_sparse_masks.get(group_idx) {
                    let dense_mask = self
                        .quad_group_dense_masks
                        .get(group_idx)
                        .and_then(Option::as_deref);
                    let replay_cost = andnot_group_buf_mask(buf, group_mask, dense_mask);
                    if PROFILE {
                        stats.complement_full_nibble_groups += 1;
                        stats.group_andnot_sparse_entries += replay_cost as u64;
                    }
                    bits &= !(0x0fu64 << shift);
                }
            }
        }

        while bits != 0 {
            if PROFILE {
                stats.complement_token_iterations += 1;
            }
            let bit = bits.trailing_zeros() as usize;
            let internal_token = wi * 64 + bit;
            if internal_token < self.internal_token_buf_offsets.len().saturating_sub(1) {
                self.andnot_internal_token_from_buf_fast::<PROFILE>(
                    internal_token,
                    buf,
                    &mut stats.complement_sparse_entries,
                );
            }
            bits &= bits - 1;
        }
    }

    fn fill_internal_dense_complement_to_buf<const PROFILE: bool>(
        &self,
        dense: &[u64],
        n_internal: usize,
        buf: &mut [u32],
        stats: &mut DenseToBufProfileStats,
    ) {
        copy_dense_buf(buf, &self.all_tokens_buf_mask);
        let mut wi = 0usize;
        while wi < dense.len() {
            if wi * 64 >= n_internal {
                break;
            }
            if PROFILE {
                stats.dense_words_visited += 1;
            }
            let w = dense[wi];
            let remaining = n_internal - wi * 64;
            let valid_mask = if remaining >= 64 {
                !0u64
            } else {
                (1u64 << remaining) - 1
            };
            let missing = !w & valid_mask;
            if missing == 0 {
                wi += 1;
                continue;
            }
            if missing == valid_mask {
                let run_start = wi;
                wi += 1;
                while wi < dense.len() && wi * 64 < n_internal {
                    let remaining = n_internal - wi * 64;
                    if remaining < 64 || dense[wi] != 0 {
                        break;
                    }
                    if PROFILE {
                        stats.dense_words_visited += 1;
                    }
                    wi += 1;
                }
                self.andnot_full_internal_word_run_from_buf::<PROFILE>(
                    run_start,
                    wi,
                    buf,
                    stats,
                );
                continue;
            }
            self.andnot_internal_bits_from_buf_grouped::<PROFILE>(
                wi,
                missing,
                valid_mask,
                buf,
                stats,
            );
            wi += 1;
        }
    }

    /// Convert a merged internal token dense bitmap to the output buffer.
    /// Uses a contiguous flat entry array for cache-friendly sequential access,
    /// with word_group fast paths for fully-set 64-bit words and heavy token
    /// dense masks for tokens with many buf entries.
    fn or_internal_dense_to_buf_impl<const PROFILE: bool>(
        &self,
        dense: &[u64],
        buf: &mut [u32],
        buf_zeroed: bool,
        mut dirty_complement_scratch: Option<&mut Vec<u32>>,
    ) -> DenseToBufProfileStats {
        if self.final_mask_mapping.internal_len() > 0 {
            if PROFILE {
                return self
                    .final_mask_mapping
                    .or_dense_to_buf(dense, buf, buf_zeroed);
            }
            self.final_mask_mapping
                .or_dense_to_buf_fast(dense, buf, buf_zeroed);
            return DenseToBufProfileStats::default();
        }

        let mut stats = DenseToBufProfileStats::default();
        let all_mask = &self.all_tokens_buf_mask;
        let sparse_word_groups = &self.word_group_sparse_masks;
        let offsets = &self.internal_token_buf_offsets;
        let n_internal = if offsets.len() > 1 { offsets.len() - 1 } else { 0 };

        if n_internal == 0 || dense.is_empty() {
            return stats;
        }

        // Count set bits to choose path.
        let n_set: usize = dense.iter().map(|w| w.count_ones() as usize).sum();

        // Super-fast path: all internal tokens set → OR all_tokens_buf_mask.
        if n_set >= n_internal && !all_mask.is_empty() {
            if buf_zeroed {
                copy_dense_buf(buf, all_mask);
            } else {
                or_dense_buf(buf, all_mask);
            }
            return stats;
        }

        if n_set == 0 {
            return stats;
        }

        let buf_len = buf.len();
        let n_missing = n_internal - n_set;

        let dense_complement_fast_path =
            n_set.saturating_mul(5) >= n_internal.saturating_mul(4) && n_missing <= 128;

        let dirty_complement_fast_path = if !buf_zeroed
            && dirty_complement_scratch.is_some()
            && !all_mask.is_empty()
        {
            let selected_cost =
                self.internal_dense_buf_replay_cost(dense, n_internal, buf_len, false);
            let dense_pass_cost = buf_len.saturating_mul(2);
            if selected_cost <= dense_pass_cost {
                false
            } else {
                let missing_cost =
                    self.internal_dense_buf_replay_cost(dense, n_internal, buf_len, true);
                dense_pass_cost.saturating_add(missing_cost) < selected_cost
            }
        } else {
            false
        };

        // Complement conversion seeds ALL and then clears missing-token bits.
        // It is only an OR-equivalent conversion when `buf` is known zero;
        // otherwise the clears can erase bits produced by another parser path.
        if !all_mask.is_empty()
            && ((buf_zeroed && dense_complement_fast_path)
                || (!buf_zeroed && dirty_complement_fast_path))
        {
            if PROFILE {
                stats.complement_path_used = true;
            }
            if buf_zeroed {
                self.fill_internal_dense_complement_to_buf::<PROFILE>(
                    dense,
                    n_internal,
                    buf,
                    &mut stats,
                );
            } else if let Some(scratch) = dirty_complement_scratch.as_deref_mut() {
                scratch.resize(buf.len(), 0);
                self.fill_internal_dense_complement_to_buf::<PROFILE>(
                    dense,
                    n_internal,
                    scratch,
                    &mut stats,
                );
                or_dense_buf(buf, scratch);
            }
        } else {
            // Normal path: process sparse light tokens and dense heavy tokens.
            let mut wi = 0usize;
            while wi < dense.len() {
                if wi * 64 >= n_internal {
                    break;
                }
                if PROFILE {
                    stats.dense_words_visited += 1;
                }
                let w = dense[wi];
                let remaining = n_internal - wi * 64;
                let valid_mask = if remaining >= 64 { !0u64 } else { (1u64 << remaining) - 1 };
                let valid_bits = w & valid_mask;
                if valid_bits == 0 {
                    wi += 1;
                    continue;
                }
                if valid_bits == valid_mask {
                    let run_start = wi;
                    wi += 1;
                    while wi < dense.len() && wi * 64 < n_internal {
                        let remaining = n_internal - wi * 64;
                        if remaining < 64 || dense[wi] != !0u64 {
                            break;
                        }
                        if PROFILE {
                            stats.dense_words_visited += 1;
                        }
                        wi += 1;
                    }
                    self.or_full_internal_word_run_to_buf::<PROFILE>(
                        run_start,
                        wi,
                        buf,
                        &mut stats,
                    );
                    continue;
                }
                let missing_bits = !valid_bits & valid_mask;
                if missing_bits != 0 {
                    if let Some(group_mask) = sparse_word_groups.get(wi) {
                        let selected_cost = self.internal_bits_buf_op_cost(wi, valid_bits, buf_len);
                        let missing_cost = self
                            .word_group_buf_op_costs
                            .get(wi)
                            .copied()
                            .unwrap_or_else(|| selected_cost + self.internal_bits_buf_op_cost(wi, missing_bits, buf_len))
                            .saturating_sub(selected_cost);
                        if buf_zeroed && group_mask.len() + missing_cost < selected_cost {
                            if PROFILE {
                                stats.normal_group_complement_hits += 1;
                            }
                            if Self::prefer_dense_buf_scan(buf_len, group_mask.len())
                                && wi + 1 < self.word_group_prefix_buf_masks.len()
                            {
                                if PROFILE {
                                    stats.group_or_sparse_entries += buf_len as u64;
                                }
                                self.or_word_group_prefix_diff_to_buf(wi, wi + 1, buf);
                            } else {
                                if PROFILE {
                                    stats.group_or_sparse_entries += group_mask.len() as u64;
                                }
                                or_sparse_buf_entries(buf, group_mask);
                            }
                            let mut missing_stats = DenseToBufProfileStats::default();
                            self.andnot_internal_bits_from_buf_grouped::<PROFILE>(
                                wi,
                                missing_bits,
                                valid_mask,
                                buf,
                                &mut missing_stats,
                            );
                            if PROFILE {
                                stats.normal_group_complement_sparse_entries +=
                                    missing_stats.group_andnot_sparse_entries
                                        + missing_stats.complement_sparse_entries;
                                stats.complement_full_byte_groups +=
                                    missing_stats.complement_full_byte_groups;
                                stats.complement_full_nibble_groups +=
                                    missing_stats.complement_full_nibble_groups;
                            }
                            wi += 1;
                            continue;
                        }
                    }
                }

                self.or_internal_bits_to_buf_grouped::<PROFILE>(
                    wi,
                    valid_bits,
                    valid_mask,
                    buf,
                    &mut stats,
                );
                wi += 1;
            }
        }

        stats
    }

    pub(crate) fn or_internal_dense_to_buf(
        &self,
        dense: &[u64],
        buf: &mut [u32],
        buf_zeroed: bool,
    ) -> DenseToBufProfileStats {
        self.or_internal_dense_to_buf_impl::<true>(dense, buf, buf_zeroed, None)
    }

    pub(crate) fn or_internal_dense_to_buf_fast(
        &self,
        dense: &[u64],
        buf: &mut [u32],
        buf_zeroed: bool,
    ) {
        let _ = self.or_internal_dense_to_buf_impl::<false>(dense, buf, buf_zeroed, None);
    }

    pub(crate) fn or_internal_dense_to_buf_fast_with_scratch(
        &self,
        dense: &[u64],
        buf: &mut [u32],
        buf_zeroed: bool,
        dirty_complement_scratch: &mut Vec<u32>,
    ) {
        let _ = self.or_internal_dense_to_buf_impl::<false>(
            dense,
            buf,
            buf_zeroed,
            Some(dirty_complement_scratch),
        );
    }


}
