//! Packed mask caches and model/internal token-coordinate maps.

use rayon::prelude::*;
use super::{BackedInternalTokenBufMasks, Constraint, DenseBufMaskRows, Deserialize, InternalTokenBufMasks, PackedInternalTokenBufMask, Serialize};

#[derive(Deserialize)]
pub(super) struct TokenMaskCacheTail {
    pub(super) guarded_shift_index: Vec<rustc_hash::FxHashMap<
        crate::grammar::flat::TerminalID,
        crate::compiler::glr::table::GuardedShiftCellIndex,
    >>,
    pub(super) seed_terminal_dense: crate::runtime::artifact::SeedTerminalDenseMasks,
    pub(super) seed_universe_dense: Vec<u64>,
    pub(super) word_group_sparse_masks: Vec<InternalTokenBufMasks>,
    pub(super) word_group_sparse_prefix_entries: Vec<usize>,
    pub(super) quad_group_sparse_masks: Vec<InternalTokenBufMasks>,
    pub(super) quad_group_dense_masks: Vec<Option<Box<[u32]>>>,
    pub(super) byte_group_sparse_masks: Vec<InternalTokenBufMasks>,
    pub(super) byte_group_dense_masks: Vec<Option<Box<[u32]>>>,
    pub(super) word_group_sparse_total_entries: usize,
    pub(super) word_group_sparse_max_entries: usize,
    pub(super) all_tokens_buf_mask: Box<[u32]>,
    pub(super) total_internal_buf_cost: usize,
    pub(super) heavy_token_indices: Vec<usize>,
    pub(super) heavy_total_cost: usize,
    pub(super) light_avg_cost_x256: usize,
    pub(super) internal_token_buf_op_costs: Vec<usize>,
    pub(super) word_group_buf_op_costs: Vec<usize>,
}

#[derive(Serialize)]
pub(super) struct TokenMaskCacheIrregularRef<'a> {
    pub(super) guarded_shift_index: &'a [rustc_hash::FxHashMap<
        crate::grammar::flat::TerminalID,
        crate::compiler::glr::table::GuardedShiftCellIndex,
    >],
    pub(super) seed_terminal_dense: &'a crate::runtime::artifact::SeedTerminalDenseMasks,
    pub(super) seed_universe_dense: &'a [u64],
    pub(super) quad_group_sparse_masks: &'a [InternalTokenBufMasks],
    pub(super) quad_group_dense_masks: &'a [Option<Box<[u32]>>],
    pub(super) byte_group_sparse_masks: &'a [InternalTokenBufMasks],
    pub(super) byte_group_dense_masks: &'a [Option<Box<[u32]>>],
}

#[derive(Deserialize)]
pub(super) struct TokenMaskCacheIrregular {
    pub(super) guarded_shift_index: Vec<rustc_hash::FxHashMap<
        crate::grammar::flat::TerminalID,
        crate::compiler::glr::table::GuardedShiftCellIndex,
    >>,
    pub(super) seed_terminal_dense: crate::runtime::artifact::SeedTerminalDenseMasks,
    pub(super) seed_universe_dense: Vec<u64>,
    pub(super) quad_group_sparse_masks: Vec<InternalTokenBufMasks>,
    pub(super) quad_group_dense_masks: Vec<Option<Box<[u32]>>>,
    pub(super) byte_group_sparse_masks: Vec<InternalTokenBufMasks>,
    pub(super) byte_group_dense_masks: Vec<Option<Box<[u32]>>>,
}

#[derive(Serialize, Deserialize)]
pub(super) struct SeedTerminalDenseCompact {
    pub(super) masks: Vec<crate::runtime::artifact::DenseWords>,
    pub(super) entries: Vec<(u32, crate::grammar::flat::TerminalID, u32)>,
}

impl SeedTerminalDenseCompact {
    pub(super) fn from_map(map: &crate::runtime::artifact::SeedTerminalDenseMasks) -> Self {
        let mut ordered = map.iter().collect::<Vec<_>>();
        ordered.sort_unstable_by_key(|&(&key, _)| key);
        let mut mask_ids = rustc_hash::FxHashMap::<
            crate::runtime::artifact::DenseWords,
            u32,
        >::default();
        let mut masks = Vec::new();
        let mut entries = Vec::with_capacity(ordered.len());
        for (&(state, terminal), mask) in ordered {
            let mask_id = if let Some(&mask_id) = mask_ids.get(mask) {
                mask_id
            } else {
                let mask_id = masks.len() as u32;
                let shared = std::sync::Arc::clone(mask);
                mask_ids.insert(std::sync::Arc::clone(&shared), mask_id);
                masks.push(shared);
                mask_id
            };
            entries.push((state, terminal, mask_id));
        }
        Self { masks, entries }
    }

    pub(super) fn into_map(self) -> Result<crate::runtime::artifact::SeedTerminalDenseMasks, String> {
        let mut result = crate::runtime::artifact::SeedTerminalDenseMasks::default();
        result.reserve(self.entries.len());
        for (state, terminal, mask_id) in self.entries {
            let mask = self
                .masks
                .get(mask_id as usize)
                .ok_or_else(|| "token-mask seed mask id is out of range".to_owned())?;
            if result
                .insert((state, terminal), std::sync::Arc::clone(mask))
                .is_some()
            {
                return Err("duplicate token-mask seed key".to_owned());
            }
        }
        Ok(result)
    }
}

#[derive(Serialize)]
pub(super) struct TokenMaskCacheIrregularV5Ref<'a> {
    pub(super) guarded_shift_index: &'a [rustc_hash::FxHashMap<
        crate::grammar::flat::TerminalID,
        crate::compiler::glr::table::GuardedShiftCellIndex,
    >],
    pub(super) seed_terminal_dense: &'a SeedTerminalDenseCompact,
    pub(super) seed_universe_dense: &'a [u64],
    pub(super) quad_group_sparse_masks: &'a [InternalTokenBufMasks],
    pub(super) quad_group_dense_masks: &'a [Option<Box<[u32]>>],
    pub(super) byte_group_sparse_masks: &'a [InternalTokenBufMasks],
    pub(super) byte_group_dense_masks: &'a [Option<Box<[u32]>>],
}

#[derive(Deserialize)]
pub(super) struct TokenMaskCacheIrregularV5 {
    pub(super) guarded_shift_index: Vec<rustc_hash::FxHashMap<
        crate::grammar::flat::TerminalID,
        crate::compiler::glr::table::GuardedShiftCellIndex,
    >>,
    pub(super) seed_terminal_dense: SeedTerminalDenseCompact,
    pub(super) seed_universe_dense: Vec<u64>,
    pub(super) quad_group_sparse_masks: Vec<InternalTokenBufMasks>,
    pub(super) quad_group_dense_masks: Vec<Option<Box<[u32]>>>,
    pub(super) byte_group_sparse_masks: Vec<InternalTokenBufMasks>,
    pub(super) byte_group_dense_masks: Vec<Option<Box<[u32]>>>,
}

impl TokenMaskCacheIrregularV5 {
    pub(super) fn into_irregular(self) -> Result<TokenMaskCacheIrregular, String> {
        Ok(TokenMaskCacheIrregular {
            guarded_shift_index: self.guarded_shift_index,
            seed_terminal_dense: self.seed_terminal_dense.into_map()?,
            seed_universe_dense: self.seed_universe_dense,
            quad_group_sparse_masks: self.quad_group_sparse_masks,
            quad_group_dense_masks: self.quad_group_dense_masks,
            byte_group_sparse_masks: self.byte_group_sparse_masks,
            byte_group_dense_masks: self.byte_group_dense_masks,
        })
    }
}

pub(super) enum TokenMaskCacheArtifact {
    Full {
        tail: TokenMaskCacheTail,
        word_group_prefix_buf_masks: DenseBufMaskRows,
    },
    Fast {
        irregular: TokenMaskCacheIrregular,
        word_group_sparse_masks: Vec<InternalTokenBufMasks>,
        word_group_prefix_buf_masks: DenseBufMaskRows,
    },
    WordSparse(Vec<InternalTokenBufMasks>),
}

pub(super) fn encode_word_sparse_token_mask_cache(constraint: &Constraint) -> Vec<u8> {
    const MAGIC: &[u8; 4] = b"TWS2";
    const MAX_BYTES: usize = 512 * 1024;
    let expected_groups = constraint.internal_token_count().div_ceil(64);
    if constraint.word_group_sparse_masks.len() != expected_groups {
        return Vec::new();
    }
    let entry_count = constraint
        .word_group_sparse_masks
        .iter()
        .map(Vec::len)
        .sum::<usize>();
    let encoded_len = 12usize
        .saturating_add((expected_groups + 1).saturating_mul(4))
        .saturating_add(entry_count.saturating_mul(8));
    if encoded_len > MAX_BYTES {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(encoded_len);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(expected_groups as u32).to_le_bytes());
    out.extend_from_slice(&(entry_count as u32).to_le_bytes());
    let mut end = 0u32;
    out.extend_from_slice(&end.to_le_bytes());
    for group in &constraint.word_group_sparse_masks {
        end = end.saturating_add(group.len() as u32);
        out.extend_from_slice(&end.to_le_bytes());
    }
    for group in &constraint.word_group_sparse_masks {
        for &(word, bits) in group {
            out.extend_from_slice(&word.to_le_bytes());
            out.extend_from_slice(&bits.to_le_bytes());
        }
    }
    debug_assert_eq!(out.len(), encoded_len);
    out
}

pub(super) fn decode_word_sparse_token_mask_cache(input: &[u8]) -> Result<Vec<InternalTokenBufMasks>, String> {
    const HEADER_LEN: usize = 12;
    if input.len() < HEADER_LEN || !input.starts_with(b"TWS2") {
        return Err("invalid sparse word-group cache header".to_owned());
    }
    let group_count = u32::from_le_bytes(input[4..8].try_into().unwrap()) as usize;
    let entry_count = u32::from_le_bytes(input[8..12].try_into().unwrap()) as usize;
    let offsets_bytes = (group_count + 1)
        .checked_mul(4)
        .ok_or_else(|| "sparse word-group cache offsets overflow".to_owned())?;
    let entries_bytes = entry_count
        .checked_mul(8)
        .ok_or_else(|| "sparse word-group cache entries overflow".to_owned())?;
    let expected = HEADER_LEN
        .checked_add(offsets_bytes)
        .and_then(|n| n.checked_add(entries_bytes))
        .ok_or_else(|| "sparse word-group cache length overflow".to_owned())?;
    if input.len() != expected {
        return Err("invalid sparse word-group cache length".to_owned());
    }
    let offsets_body = &input[HEADER_LEN..HEADER_LEN + offsets_bytes];
    let mut offsets = Vec::<u32>::with_capacity(group_count + 1);
    for bytes in offsets_body.chunks_exact(4) {
        offsets.push(u32::from_le_bytes(bytes.try_into().unwrap()));
    }
    if offsets.first().copied() != Some(0)
        || offsets.last().copied() != Some(entry_count as u32)
        || offsets.windows(2).any(|pair| pair[0] > pair[1])
    {
        return Err("invalid sparse word-group cache offsets".to_owned());
    }
    let entries = &input[HEADER_LEN + offsets_bytes..];
    let mut groups = Vec::with_capacity(group_count);
    for group in 0..group_count {
        let start = offsets[group] as usize;
        let end = offsets[group + 1] as usize;
        let mut decoded = Vec::with_capacity(end - start);
        for entry in start..end {
            let pos = entry * 8;
            decoded.push((
                u32::from_le_bytes(entries[pos..pos + 4].try_into().unwrap()),
                u32::from_le_bytes(entries[pos + 4..pos + 8].try_into().unwrap()),
            ));
        }
        groups.push(decoded);
    }
    Ok(groups)
}

#[inline]
pub(super) fn append_cache_u32s(out: &mut Vec<u8>, values: &[u32]) {
    if cfg!(target_endian = "little") {
        let bytes = unsafe {
            std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * 4)
        };
        out.extend_from_slice(bytes);
    } else {
        for &value in values {
            out.extend_from_slice(&value.to_le_bytes());
        }
    }
}

pub(super) fn decode_cache_u32_rows(
    input: &[u8],
    pos: &mut usize,
    rows: usize,
    row_len: usize,
) -> Result<DenseBufMaskRows, String> {
    if !DenseBufMaskRows::prefer_flat(rows, row_len) {
        let row_bytes = row_len
            .checked_mul(4)
            .ok_or_else(|| "token-mask prefix row byte length overflow".to_owned())?;
        let mut decoded_rows = Vec::with_capacity(rows);
        for _ in 0..rows {
            let end = pos
                .checked_add(row_bytes)
                .ok_or_else(|| "token-mask prefix row offset overflow".to_owned())?;
            let bytes = input
                .get(*pos..end)
                .ok_or_else(|| "truncated token-mask prefix row".to_owned())?;
            let mut row = Vec::<u32>::with_capacity(row_len);
            if cfg!(target_endian = "little") {
                unsafe {
                    row.set_len(row_len);
                    std::ptr::copy_nonoverlapping(
                        bytes.as_ptr(),
                        row.as_mut_ptr().cast::<u8>(),
                        row_bytes,
                    );
                }
            } else {
                row.extend(
                    bytes
                        .chunks_exact(4)
                        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap())),
                );
            }
            *pos = end;
            decoded_rows.push(row.into_boxed_slice());
        }
        return DenseBufMaskRows::from_rows(decoded_rows);
    }
    let values = rows
        .checked_mul(row_len)
        .ok_or_else(|| "token-mask prefix dimensions overflow".to_owned())?;
    let byte_len = values
        .checked_mul(4)
        .ok_or_else(|| "token-mask prefix byte length overflow".to_owned())?;
    let end = pos
        .checked_add(byte_len)
        .ok_or_else(|| "token-mask prefix offset overflow".to_owned())?;
    let bytes = input
        .get(*pos..end)
        .ok_or_else(|| "truncated token-mask prefix".to_owned())?;
    let mut flat = Vec::<u32>::with_capacity(values);
    if cfg!(target_endian = "little") {
        unsafe {
            flat.set_len(values);
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), flat.as_mut_ptr().cast::<u8>(), byte_len);
        }
    } else {
        flat.extend(
            bytes
                .chunks_exact(4)
                .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap())),
        );
    }
    *pos = end;
    DenseBufMaskRows::from_flat(flat.into_boxed_slice(), rows, row_len)
}

pub(super) fn encode_token_mask_cache(constraint: &Constraint) -> Vec<u8> {
    const HEADER_LEN: usize = 24;
    // JS's exact prefix matrix is ~1.3 MiB. Keeping the old 1 MiB cutoff made
    // every load throw that useful runtime-native cache away and rebuild it.
    const MAX_PREFIX_BYTES: usize = 2 * 1024 * 1024;
    const MAX_CACHE_BYTES: usize = 4 * 1024 * 1024;
    if !constraint.token_mask_caches_ready() {
        return Vec::new();
    }
    let mask_words = constraint.body_mask_len();
    let prefix_rows = constraint.word_group_prefix_buf_masks.len();
    let word_groups = constraint.word_group_sparse_masks.len();
    let word_entries = constraint
        .word_group_sparse_masks
        .iter()
        .map(Vec::len)
        .sum::<usize>();
    let prefix_bytes = prefix_rows
        .saturating_mul(mask_words)
        .saturating_mul(std::mem::size_of::<u32>());
    if prefix_bytes > MAX_PREFIX_BYTES {
        return encode_word_sparse_token_mask_cache(constraint);
    }
    if prefix_rows != word_groups.saturating_add(1)
        || constraint
            .word_group_prefix_buf_masks
            .iter()
            .any(|row| row.len() != mask_words)
    {
        return Vec::new();
    }
    let profile = std::env::var_os("GLRMASK_PROFILE_SERIALIZATION").is_some();
    let tail_started = profile.then(std::time::Instant::now);
    let mut tail = Vec::with_capacity(32 * 1024);
    // Deduplicating tiny seed maps costs more bookkeeping than it saves. Large
    // seed tables, however, commonly repeat the same dense token mask across
    // hundreds of tokenizer-state/terminal keys; the compact encoding pools those immutable
    // masks once and shares the Arc again after load.
    let compact_seed = constraint.seed_terminal_dense.len() >= 64;
    if compact_seed {
        let seed_terminal_dense = SeedTerminalDenseCompact::from_map(&constraint.seed_terminal_dense);
        bincode::serialize_into(
            &mut tail,
            &TokenMaskCacheIrregularV5Ref {
                guarded_shift_index: &constraint.table.guarded_shift_index,
                seed_terminal_dense: &seed_terminal_dense,
                seed_universe_dense: &constraint.seed_universe_dense,
                quad_group_sparse_masks: &constraint.quad_group_sparse_masks,
                quad_group_dense_masks: &constraint.quad_group_dense_masks,
                byte_group_sparse_masks: &constraint.byte_group_sparse_masks,
                byte_group_dense_masks: &constraint.byte_group_dense_masks,
            },
        )
        .expect("token-mask cache serialization should succeed");
    } else {
        bincode::serialize_into(
            &mut tail,
            &TokenMaskCacheIrregularRef {
                guarded_shift_index: &constraint.table.guarded_shift_index,
                seed_terminal_dense: &constraint.seed_terminal_dense,
                seed_universe_dense: &constraint.seed_universe_dense,
                quad_group_sparse_masks: &constraint.quad_group_sparse_masks,
                quad_group_dense_masks: &constraint.quad_group_dense_masks,
                byte_group_sparse_masks: &constraint.byte_group_sparse_masks,
                byte_group_dense_masks: &constraint.byte_group_dense_masks,
            },
        )
        .expect("token-mask cache serialization should succeed");
    }
    let tail_ms = tail_started
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let word_offsets_bytes = (word_groups + 1).saturating_mul(4);
    let word_entries_bytes = word_entries.saturating_mul(8);
    // TMC8/9 align the dense u32 matrix so an owned load can retain it directly
    // from the artifact rather than copying ~1.3 MiB merely for alignment.
    let prefix_unaligned_start = HEADER_LEN
        .saturating_add(tail.len())
        .saturating_add(word_offsets_bytes)
        .saturating_add(word_entries_bytes);
    let prefix_padding = (4 - (prefix_unaligned_start & 3)) & 3;
    let total_len = HEADER_LEN
        .saturating_add(tail.len())
        .saturating_add(word_offsets_bytes)
        .saturating_add(word_entries_bytes)
        .saturating_add(prefix_padding)
        .saturating_add(prefix_bytes);
    if total_len > MAX_CACHE_BYTES {
        return encode_word_sparse_token_mask_cache(constraint);
    }
    let prefix_started = profile.then(std::time::Instant::now);
    let mut out = Vec::with_capacity(total_len);
    out.extend_from_slice(if compact_seed { b"TMC9" } else { b"TMC8" });
    for value in [tail.len(), mask_words, word_groups, word_entries, prefix_rows] {
        out.extend_from_slice(
            &u32::try_from(value)
                .expect("token-mask cache dimension fits u32")
                .to_le_bytes(),
        );
    }
    out.extend_from_slice(&tail);
    let mut end = 0u32;
    out.extend_from_slice(&end.to_le_bytes());
    for group in &constraint.word_group_sparse_masks {
        end = end.saturating_add(group.len() as u32);
        out.extend_from_slice(&end.to_le_bytes());
    }
    for group in &constraint.word_group_sparse_masks {
        for &(word, bits) in group {
            out.extend_from_slice(&word.to_le_bytes());
            out.extend_from_slice(&bits.to_le_bytes());
        }
    }
    out.resize(out.len() + prefix_padding, 0);
    if let Some(flat) = constraint.word_group_prefix_buf_masks.as_contiguous() {
        append_cache_u32s(&mut out, flat);
    } else {
        for row in &constraint.word_group_prefix_buf_masks {
            append_cache_u32s(&mut out, row);
        }
    }
    debug_assert_eq!(out.len(), total_len);
    if let Some(started) = prefix_started {
        eprintln!(
            "[glrmask/profile][token_mask_cache_encode] tail_ms={tail_ms:.3} body_ms={:.3} tail_bytes={} word_sparse_bytes={} prefix_bytes={} total_bytes={}",
            started.elapsed().as_secs_f64() * 1000.0,
            tail.len(),
            word_offsets_bytes + word_entries_bytes,
            prefix_bytes,
            out.len(),
        );
    }
    out
}

pub(super) fn decode_token_mask_cache(input: &[u8]) -> Result<TokenMaskCacheArtifact, String> {
    decode_token_mask_cache_impl(input, None)
}

pub(super) fn decode_token_mask_cache_backed(
    input: &[u8],
    backing: std::sync::Arc<Vec<u8>>,
    section_start: usize,
) -> Result<TokenMaskCacheArtifact, String> {
    decode_token_mask_cache_impl(input, Some((backing, section_start)))
}

pub(super) fn decode_token_mask_cache_impl(
    input: &[u8],
    backing: Option<(std::sync::Arc<Vec<u8>>, usize)>,
) -> Result<TokenMaskCacheArtifact, String> {
    // The cache may have up to three leading zero bytes to align its dense
    // matrix within the enclosing artifact. Standalone sections need no padding.
    let has_known_magic = |bytes: &[u8]| {
        bytes.starts_with(b"TWS2")
            || bytes.starts_with(b"TMC3")
            || bytes.starts_with(b"TMC4")
            || bytes.starts_with(b"TMC5")
            || bytes.starts_with(b"TMC8")
            || bytes.starts_with(b"TMC9")
    };
    let leading_padding = if has_known_magic(input) {
        0
    } else {
        (1..=3)
            .find(|&padding| {
                input
                    .get(..padding)
                    .is_some_and(|prefix| prefix.iter().all(|&byte| byte == 0))
                    && input.get(padding..).is_some_and(has_known_magic)
            })
            .unwrap_or(0)
    };
    let input = &input[leading_padding..];
    let backing = backing.map(|(backing, section_start)| {
        (
            backing,
            section_start
                .checked_add(leading_padding)
                .expect("token-mask cache section offset cannot overflow"),
        )
    });
    const MAGIC: &[u8; 4] = b"TMC3";
    const HEADER_LEN: usize = 16;
    if input.starts_with(b"TWS2") {
        return decode_word_sparse_token_mask_cache(input).map(TokenMaskCacheArtifact::WordSparse);
    }
    if input.starts_with(b"TMC9")
        || input.starts_with(b"TMC8")
        || input.starts_with(b"TMC5")
        || input.starts_with(b"TMC4")
    {
        let compact_seed = input.starts_with(b"TMC9") || input.starts_with(b"TMC5");
        let aligned_prefix = input.starts_with(b"TMC9") || input.starts_with(b"TMC8");
        const FAST_HEADER_LEN: usize = 24;
        if input.len() < FAST_HEADER_LEN {
            return Err("invalid fast token-mask cache header".to_owned());
        }
        let read = |offset: usize| {
            u32::from_le_bytes(input[offset..offset + 4].try_into().unwrap()) as usize
        };
        let tail_len = read(4);
        let mask_words = read(8);
        let word_groups = read(12);
        let word_entries = read(16);
        let prefix_rows = read(20);
        if prefix_rows != word_groups.saturating_add(1) {
            return Err("fast token-mask prefix row count mismatch".to_owned());
        }
        let tail_end = FAST_HEADER_LEN
            .checked_add(tail_len)
            .ok_or_else(|| "fast token-mask cache tail overflow".to_owned())?;
        let offsets_bytes = (word_groups + 1)
            .checked_mul(4)
            .ok_or_else(|| "fast token-mask sparse offsets overflow".to_owned())?;
        let entries_bytes = word_entries
            .checked_mul(8)
            .ok_or_else(|| "fast token-mask sparse entries overflow".to_owned())?;
        let prefix_bytes = prefix_rows
            .checked_mul(mask_words)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| "fast token-mask prefix bytes overflow".to_owned())?;
        let prefix_unaligned_start = tail_end
            .checked_add(offsets_bytes)
            .and_then(|n| n.checked_add(entries_bytes))
            .ok_or_else(|| "fast token-mask cache prefix offset overflow".to_owned())?;
        let prefix_padding = if aligned_prefix {
            (4 - (prefix_unaligned_start & 3)) & 3
        } else {
            0
        };
        let prefix_start = prefix_unaligned_start
            .checked_add(prefix_padding)
            .ok_or_else(|| "fast token-mask cache prefix padding overflow".to_owned())?;
        let expected = prefix_start
            .checked_add(prefix_bytes)
            .ok_or_else(|| "fast token-mask cache length overflow".to_owned())?;
        if expected != input.len() {
            return Err("invalid fast token-mask cache length".to_owned());
        }
        let offsets_start = tail_end;
        let entries_start = offsets_start + offsets_bytes;
        if aligned_prefix
            && input
                .get(prefix_unaligned_start..prefix_start)
                .is_none_or(|padding| padding.iter().any(|&byte| byte != 0))
        {
            return Err("invalid fast token-mask prefix padding".to_owned());
        }
        let profile = std::env::var_os("GLRMASK_PROFILE_SERIALIZATION").is_some();
        let tail_started = profile.then(std::time::Instant::now);
        let tail_bytes = input
            .get(FAST_HEADER_LEN..tail_end)
            .ok_or_else(|| "truncated fast token-mask cache tail".to_owned())?;
        let irregular = if compact_seed {
            bincode::deserialize::<TokenMaskCacheIrregularV5>(tail_bytes)
                .map_err(|err| err.to_string())?
                .into_irregular()?
        } else {
            bincode::deserialize::<TokenMaskCacheIrregular>(tail_bytes)
                .map_err(|err| err.to_string())?
        };
        let tail_ms = tail_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let body_started = profile.then(std::time::Instant::now);
        let mut offsets = Vec::<u32>::with_capacity(word_groups + 1);
        for bytes in input[offsets_start..entries_start].chunks_exact(4) {
            offsets.push(u32::from_le_bytes(bytes.try_into().unwrap()));
        }
        if offsets.first().copied() != Some(0)
            || offsets.last().copied() != Some(word_entries as u32)
            || offsets.windows(2).any(|pair| pair[0] > pair[1])
        {
            return Err("invalid fast token-mask sparse offsets".to_owned());
        }
        let entries = &input[entries_start..prefix_start];
        let decode_group = |group: usize| -> Result<InternalTokenBufMasks, String> {
            let start = offsets[group] as usize;
            let end = offsets[group + 1] as usize;
            let mut decoded = Vec::with_capacity(end - start);
            if cfg!(target_endian = "little") {
                // `expected == input.len()` above proves every 8-byte record is
                // present. Read the packed fields directly instead of doing
                // two independently bounds-checked slices + `try_into()` per
                // sparse entry. The wire is intentionally unaligned.
                let base = entries.as_ptr();
                for entry in start..end {
                    let ptr = unsafe { base.add(entry * 8) };
                    let word = unsafe { std::ptr::read_unaligned(ptr.cast::<u32>()) };
                    if word as usize >= mask_words {
                        return Err("fast token-mask sparse word out of range".to_owned());
                    }
                    let bits = unsafe { std::ptr::read_unaligned(ptr.add(4).cast::<u32>()) };
                    decoded.push((word, bits));
                }
            } else {
                for entry in start..end {
                    let pos = entry * 8;
                    let word = u32::from_le_bytes(entries[pos..pos + 4].try_into().unwrap());
                    if word as usize >= mask_words {
                        return Err("fast token-mask sparse word out of range".to_owned());
                    }
                    decoded.push((
                        word,
                        u32::from_le_bytes(entries[pos + 4..pos + 8].try_into().unwrap()),
                    ));
                }
            }
            Ok(decoded)
        };
        let word_group_sparse_masks = if word_groups >= 128 && rayon::current_num_threads() > 1 {
            (0..word_groups)
                .into_par_iter()
                .map(decode_group)
                .collect::<Result<Vec<_>, _>>()?
        } else {
            (0..word_groups)
                .map(decode_group)
                .collect::<Result<Vec<_>, _>>()?
        };
        let mut pos = prefix_start;
        let materialize_prefix = std::env::var_os("GLRMASK_MATERIALIZE_TMC_PREFIX").is_some();
        let word_group_prefix_buf_masks = if aligned_prefix && !materialize_prefix {
            if let Some((backing, section_start)) = backing.as_ref() {
                let absolute_start = section_start
                    .checked_add(prefix_start)
                    .ok_or_else(|| "fast token-mask backed prefix offset overflow".to_owned())?;
                match DenseBufMaskRows::from_backed(
                    std::sync::Arc::clone(backing),
                    absolute_start,
                    prefix_rows,
                    mask_words,
                ) {
                    Ok(rows) => {
                        pos = pos
                            .checked_add(prefix_bytes)
                            .ok_or_else(|| "fast token-mask backed prefix overflow".to_owned())?;
                        rows
                    }
                    Err(_) => decode_cache_u32_rows(input, &mut pos, prefix_rows, mask_words)?,
                }
            } else {
                decode_cache_u32_rows(input, &mut pos, prefix_rows, mask_words)?
            }
        } else {
            decode_cache_u32_rows(input, &mut pos, prefix_rows, mask_words)?
        };
        debug_assert_eq!(pos, input.len());
        if let Some(started) = body_started {
            eprintln!(
                "[glrmask/profile][token_mask_cache_decode] tail_ms={tail_ms:.3} body_ms={:.3} tail_bytes={} word_sparse_bytes={} prefix_bytes={}",
                started.elapsed().as_secs_f64() * 1000.0,
                tail_len,
                offsets_bytes + entries_bytes,
                prefix_bytes,
            );
        }
        return Ok(TokenMaskCacheArtifact::Fast {
            irregular,
            word_group_sparse_masks,
            word_group_prefix_buf_masks,
        });
    }
    if input.len() < HEADER_LEN || !input.starts_with(MAGIC) {
        return Err("invalid token-mask cache header".to_owned());
    }
    let read = |offset: usize| {
        u32::from_le_bytes(input[offset..offset + 4].try_into().unwrap()) as usize
    };
    let tail_len = read(4);
    let mask_words = read(8);
    let prefix_rows = read(12);
    let tail_end = HEADER_LEN
        .checked_add(tail_len)
        .ok_or_else(|| "token-mask cache tail overflow".to_owned())?;
    let profile = std::env::var_os("GLRMASK_PROFILE_SERIALIZATION").is_some();
    let tail_started = profile.then(std::time::Instant::now);
    let tail = bincode::deserialize::<TokenMaskCacheTail>(
        input
            .get(HEADER_LEN..tail_end)
            .ok_or_else(|| "truncated token-mask cache tail".to_owned())?,
    )
    .map_err(|err| err.to_string())?;
    let tail_ms = tail_started
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let prefix_started = profile.then(std::time::Instant::now);
    let mut pos = tail_end;
    let word_group_prefix_buf_masks =
        decode_cache_u32_rows(input, &mut pos, prefix_rows, mask_words)?;
    if pos != input.len() {
        return Err("trailing bytes in token-mask cache".to_owned());
    }
    if let Some(started) = prefix_started {
        eprintln!(
            "[glrmask/profile][token_mask_cache_decode] tail_ms={tail_ms:.3} prefix_ms={:.3} tail_bytes={} prefix_bytes={}",
            started.elapsed().as_secs_f64() * 1000.0,
            tail_len,
            input.len().saturating_sub(tail_end),
        );
    }
    Ok(TokenMaskCacheArtifact::Full {
        tail,
        word_group_prefix_buf_masks,
    })
}

pub(super) fn install_token_mask_cache(
    constraint: &mut Constraint,
    cache: TokenMaskCacheArtifact,
) -> Result<(), String> {
    match cache {
        TokenMaskCacheArtifact::WordSparse(groups) => {
            constraint.word_group_sparse_masks = groups;
            Ok(())
        }
        TokenMaskCacheArtifact::Fast {
            irregular,
            word_group_sparse_masks,
            word_group_prefix_buf_masks,
        } => {
            constraint.table.guarded_shift_index = irregular.guarded_shift_index;
            constraint.seed_terminal_dense = irregular.seed_terminal_dense;
            constraint.seed_universe_dense = irregular.seed_universe_dense.into();
            constraint.word_group_sparse_masks = word_group_sparse_masks;
            constraint.word_group_prefix_buf_masks = word_group_prefix_buf_masks;
            constraint.quad_group_sparse_masks = irregular.quad_group_sparse_masks;
            constraint.quad_group_dense_masks = irregular.quad_group_dense_masks;
            constraint.byte_group_sparse_masks = irregular.byte_group_sparse_masks;
            constraint.byte_group_dense_masks = irregular.byte_group_dense_masks;
            constraint.rebuild_heavy_and_sliding_token_mask_caches();
            constraint.rebuild_token_mask_cache_stats();
            if constraint.token_mask_caches_ready() {
                Ok(())
            } else {
                Err("fast token-mask cache section does not match constraint dimensions".to_owned())
            }
        }
        TokenMaskCacheArtifact::Full {
            tail: cache,
            word_group_prefix_buf_masks,
        } => {
            constraint.table.guarded_shift_index = cache.guarded_shift_index;
            constraint.seed_terminal_dense = cache.seed_terminal_dense;
            constraint.seed_universe_dense = cache.seed_universe_dense.into();
            constraint.word_group_sparse_masks = cache.word_group_sparse_masks;
            constraint.word_group_prefix_buf_masks = word_group_prefix_buf_masks;
            constraint.word_group_sparse_prefix_entries = cache.word_group_sparse_prefix_entries;
            constraint.quad_group_sparse_masks = cache.quad_group_sparse_masks;
            constraint.quad_group_dense_masks = cache.quad_group_dense_masks;
            constraint.byte_group_sparse_masks = cache.byte_group_sparse_masks;
            constraint.byte_group_dense_masks = cache.byte_group_dense_masks;
            constraint.word_group_sparse_total_entries = cache.word_group_sparse_total_entries;
            constraint.word_group_sparse_max_entries = cache.word_group_sparse_max_entries;
            constraint.all_tokens_buf_mask = cache.all_tokens_buf_mask;
            constraint.total_internal_buf_cost = cache.total_internal_buf_cost;
            constraint.heavy_token_indices = cache.heavy_token_indices;
            constraint.heavy_total_cost = cache.heavy_total_cost;
            constraint.light_avg_cost_x256 = cache.light_avg_cost_x256;
            constraint.internal_token_buf_op_costs = cache.internal_token_buf_op_costs;
            constraint.word_group_buf_op_costs = cache.word_group_buf_op_costs;
            constraint.rebuild_heavy_and_sliding_token_mask_caches();
            let rebuilt_heavy_indices = constraint
                .heavy_token_dense_masks
                .iter()
                .enumerate()
                .filter_map(|(index, mask)| mask.is_some().then_some(index))
                .collect::<Vec<_>>();
            if rebuilt_heavy_indices != constraint.heavy_token_indices {
                return Err("token-mask cache heavy-token index mismatch".to_owned());
            }
            if constraint.token_mask_caches_ready() {
                Ok(())
            } else {
                Err("token-mask cache section does not match constraint dimensions".to_owned())
            }
        }
    }
}

pub(super) fn encode_internal_token_buf_masks(constraint: &Constraint) -> Vec<u8> {
    const MAGIC: &[u8; 4] = b"IBM3";
    const ENTRY_BYTES: usize = std::mem::size_of::<PackedInternalTokenBufMask>();
    let flat_len = constraint.internal_token_buf_flat_len();
    let packed_ready = constraint.internal_token_buf_offsets.len()
        == constraint.internal_token_count().saturating_add(1)
        && constraint
            .internal_token_buf_offsets
            .last()
            .is_some_and(|&end| end as usize == flat_len);
    let group_count = if packed_ready {
        constraint.internal_token_buf_offsets.len().saturating_sub(1)
    } else {
        constraint.internal_token_buf_masks.len()
    };
    let entry_count = if packed_ready {
        flat_len
    } else {
        constraint
            .internal_token_buf_masks
            .iter()
            .map(Vec::len)
            .sum::<usize>()
    };
    let mut out = Vec::with_capacity(
        12usize
            .saturating_add((group_count + 1).saturating_mul(4))
            .saturating_add(entry_count.saturating_mul(ENTRY_BYTES)),
    );
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(group_count as u32).to_le_bytes());
    out.extend_from_slice(&(entry_count as u32).to_le_bytes());
    if packed_ready {
        for &offset in constraint.internal_token_buf_offsets.iter() {
            out.extend_from_slice(&offset.to_le_bytes());
        }
        if let Some(backed) = constraint.backed_internal_token_buf_flat.as_ref() {
            backed.append_wire_bytes(&mut out);
        } else if cfg!(target_endian = "little") {
            let byte_len = entry_count * ENTRY_BYTES;
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    constraint.internal_token_buf_flat.as_ptr().cast::<u8>(),
                    byte_len,
                )
            };
            out.extend_from_slice(bytes);
        } else {
            for entry in constraint.internal_token_buf_flat.iter() {
                out.extend_from_slice(&entry.word_idx.to_le_bytes());
                out.extend_from_slice(&entry.mask.to_le_bytes());
            }
        }
    } else {
        let mut end = 0u32;
        out.extend_from_slice(&end.to_le_bytes());
        for mask in &constraint.internal_token_buf_masks {
            end = end.saturating_add(mask.len() as u32);
            out.extend_from_slice(&end.to_le_bytes());
        }
        for mask in &constraint.internal_token_buf_masks {
            for &(word, bits) in mask {
                out.extend_from_slice(&word.to_le_bytes());
                out.extend_from_slice(&bits.to_le_bytes());
            }
        }
    }
    out
}

pub(super) struct DecodedInternalTokenBufMasks {
    pub(super) flat: Box<[PackedInternalTokenBufMask]>,
    pub(super) backed: Option<BackedInternalTokenBufMasks>,
    pub(super) offsets: Box<[u32]>,
}

pub(super) enum DecodedOriginalTokenMap {
    Materialized(Vec<u32>),
    Packed(std::sync::Arc<
        crate::runtime::artifact::original_token_map_artifact_serde::PackedOriginalTokenMap,
    >),
}

pub(super) fn decode_internal_token_buf_masks(
    input: &[u8],
    mut backing: Option<(std::sync::Arc<Vec<u8>>, usize)>,
) -> Result<DecodedInternalTokenBufMasks, String> {
    const FIXED_MAGIC: &[u8; 4] = b"IBM3";
    let mut input = input;
    if !input.starts_with(FIXED_MAGIC) {
        let leading_padding = (1..std::mem::align_of::<PackedInternalTokenBufMask>())
            .find(|&padding| {
                input.len() >= padding + FIXED_MAGIC.len()
                    && input[..padding].iter().all(|&byte| byte == 0)
                    && input[padding..].starts_with(FIXED_MAGIC)
            });
        if let Some(padding) = leading_padding {
            input = &input[padding..];
            if let Some((_, section_start)) = backing.as_mut() {
                *section_start = section_start
                    .checked_add(padding)
                    .ok_or_else(|| "internal-token buffer-mask backing offset overflow".to_owned())?;
            }
        }
    }
    if input.len() < 12 || !input.starts_with(FIXED_MAGIC) {
        return Err("invalid internal-token buffer-mask section".to_owned());
    }
    let group_count = u32::from_le_bytes(input[4..8].try_into().unwrap()) as usize;
    let entry_count = u32::from_le_bytes(input[8..12].try_into().unwrap()) as usize;
    let offsets_bytes = (group_count + 1)
        .checked_mul(4)
        .ok_or_else(|| "internal-token buffer-mask offsets overflow".to_owned())?;
    let entry_width = std::mem::size_of::<PackedInternalTokenBufMask>();
    let entries_bytes = entry_count
        .checked_mul(entry_width)
        .ok_or_else(|| "internal-token buffer-mask entries overflow".to_owned())?;
    let expected = 12usize
        .checked_add(offsets_bytes)
        .and_then(|n| n.checked_add(entries_bytes))
        .ok_or_else(|| "internal-token buffer-mask section length overflow".to_owned())?;
    if expected != input.len() {
        return Err("invalid internal-token buffer-mask section length".to_owned());
    }
    let offsets_body = &input[12..12 + offsets_bytes];
    let mut offsets = Vec::<u32>::with_capacity(group_count + 1);
    if cfg!(target_endian = "little") {
        unsafe {
            offsets.set_len(group_count + 1);
            std::ptr::copy_nonoverlapping(
                offsets_body.as_ptr(),
                offsets.as_mut_ptr().cast::<u8>(),
                offsets_bytes,
            );
        }
    } else {
        offsets.extend(
            offsets_body
                .chunks_exact(4)
                .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap())),
        );
    }
    if offsets.first().copied() != Some(0)
        || offsets.last().copied().map(|end| end as usize) != Some(entry_count)
    {
        return Err("invalid internal-token buffer-mask offsets".to_owned());
    }
    if offsets.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err("non-monotonic internal-token buffer-mask offsets".to_owned());
    }
    let entries_start = 12 + offsets_bytes;
    let entries = &input[entries_start..];
    let backed = backing
        .map(|(backing, section_start)| {
            let absolute_start = section_start.checked_add(entries_start)
                .ok_or_else(|| "internal-token buffer-mask backing offset overflow".to_owned())?;
            BackedInternalTokenBufMasks::new(backing, absolute_start, entry_count)
        })
        .transpose()?;
    let mut flat = Vec::<PackedInternalTokenBufMask>::with_capacity(if backed.is_some() {
        0
    } else {
        entry_count
    });
    if backed.is_some() {
        // The retained artifact is the runtime storage; offsets remain owned
        // because they are tiny and hot to index.
    } else if cfg!(target_endian = "little") {
        unsafe {
            flat.set_len(entry_count);
            std::ptr::copy_nonoverlapping(
                entries.as_ptr(),
                flat.as_mut_ptr().cast::<u8>(),
                entries_bytes,
            );
        }
    } else {
        // The section length was validated above, so every record is present.
        // IBM3 stores a little-endian u32 word index and u32 mask, without padding.
        unsafe {
            flat.set_len(entry_count);
            let src = entries.as_ptr();
            let dst = flat.as_mut_ptr();
            for entry in 0..entry_count {
                let pos = entry * entry_width;
                let word = u32::from_le(std::ptr::read_unaligned(src.add(pos).cast::<u32>()));
                let bits = u32::from_le(std::ptr::read_unaligned(
                    src.add(pos + 4).cast::<u32>(),
                ));
                std::ptr::write(
                    dst.add(entry),
                    PackedInternalTokenBufMask {
                        word_idx: word,
                        mask: bits,
                    },
                );
            }
        }
    }
    Ok(DecodedInternalTokenBufMasks {
        flat: flat.into_boxed_slice(),
        backed,
        offsets: offsets.into_boxed_slice(),
    })
}

#[cfg(test)]
mod output_coordinate_tests;
