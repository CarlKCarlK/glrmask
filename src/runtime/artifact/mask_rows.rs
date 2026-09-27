//! Dense, sparse, and artifact-backed token-mask row representations.

use crate::grammar::flat::TerminalID;
use rustc_hash::FxHashMap;
use rustc_hash::FxHashSet;
use std::collections::BTreeMap;
use std::sync::Arc;
#[derive(Debug)]
pub(crate) struct PackedNonDwaWeights {
    pub(crate) pool: Arc<crate::ds::weight::PackedRuntimeWeightPool>,
    pub(crate) parser_top_accept: BTreeMap<i32, u32>,
    pub(crate) parser_top_accept_parts: BTreeMap<i32, Vec<u32>>,
    pub(crate) direct_regular_l1_complete_by_terminal: BTreeMap<TerminalID, u32>,
    pub(crate) possible_matches: BTreeMap<TerminalID, u32>,
}

pub(crate) type DenseWords = Arc<[u64]>;

/// Exact dense acceptance indexed directly by internal tokenizer-state ID.
///
/// `row_kinds` uses 0 for empty, 1 for an ordinary row in `rows`, and 2 for the
/// shared all-token row. Keeping all ordinary rows in one flat allocation avoids
/// tens of thousands of per-state `Arc` allocations during finalization and
/// makes hot-path lookup a bounds check plus one slice operation.
#[derive(Debug, Clone, Default)]
pub(crate) struct DenseAcceptanceRows {
    pub(super) words_per_row: usize,
    pub(super) rows: Arc<[u64]>,
    pub(super) row_kinds: Arc<[u8]>,
    pub(super) full_dense: DenseWords,
}

impl DenseAcceptanceRows {
    pub(crate) fn new(
        words_per_row: usize,
        rows: Vec<u64>,
        row_kinds: Vec<u8>,
        full_dense: DenseWords,
    ) -> Self {
        debug_assert_eq!(rows.len(), words_per_row.saturating_mul(row_kinds.len()));
        Self {
            words_per_row,
            rows: rows.into(),
            row_kinds: row_kinds.into(),
            full_dense,
        }
    }

    #[inline]
    pub(crate) fn get(&self, tsid: u32) -> Option<&[u64]> {
        let tsid = tsid as usize;
        match self.row_kinds.get(tsid).copied()? {
            0 => None,
            2 => Some(self.full_dense.as_ref()),
            _ => {
                let start = tsid.checked_mul(self.words_per_row)?;
                self.rows.get(start..start + self.words_per_row)
            }
        }
    }
}

pub(crate) fn empty_dense_words() -> DenseWords {
    Arc::<[u64]>::from(Vec::<u64>::new().into_boxed_slice())
}

pub(crate) type InternalTokenBufMasks = Vec<(u32, u32)>;

/// Runtime-native sparse output-mask entry. Every u32 model token ID has a
/// u32 word coordinate; narrowing it to u16 would wrap IDs at or above 2^21.
/// The two u32 fields occupy the same eight bytes as the old padded entry and
/// preserve direct, naturally aligned replay from current artifact backing.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct PackedInternalTokenBufMask {
    pub(crate) word_idx: u32,
    pub(crate) mask: u32,
}

const _: () = assert!(std::mem::size_of::<PackedInternalTokenBufMask>() == 8);

#[derive(Debug, Clone)]
pub(crate) struct BackedInternalTokenBufMasks {
    pub(super) backing: Arc<Vec<u8>>,
    pub(super) entries_start: usize,
    pub(super) len: usize,
    pub(super) aligned_base_addr: Option<usize>,
}

impl BackedInternalTokenBufMasks {
    pub(crate) fn new(
        backing: Arc<Vec<u8>>,
        entries_start: usize,
        len: usize,
    ) -> Result<Self, String> {
        let bytes = len
            .checked_mul(std::mem::size_of::<PackedInternalTokenBufMask>())
            .ok_or_else(|| "backed internal-token buffer-mask length overflow".to_owned())?;
        let end = entries_start
            .checked_add(bytes)
            .ok_or_else(|| "backed internal-token buffer-mask range overflow".to_owned())?;
        if end > backing.len() {
            return Err("backed internal-token buffer-mask range is outside artifact".to_owned());
        }
        let ptr = unsafe { backing.as_ptr().add(entries_start) };
        let aligned_base_addr = (cfg!(target_endian = "little")
            && ptr.align_offset(std::mem::align_of::<PackedInternalTokenBufMask>()) == 0)
            .then_some(ptr as usize);
        Ok(Self {
            backing,
            entries_start,
            len,
            aligned_base_addr,
        })
    }

    #[inline(always)]
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn append_wire_bytes(&self, out: &mut Vec<u8>) {
        let byte_len = self.len * std::mem::size_of::<PackedInternalTokenBufMask>();
        out.extend_from_slice(&self.backing[self.entries_start..self.entries_start + byte_len]);
    }

    #[inline(always)]
    pub(crate) fn slice(&self, start: usize, end: usize) -> Option<&[PackedInternalTokenBufMask]> {
        if start > end || end > self.len {
            return None;
        }
        let base = self.aligned_base_addr? as *const PackedInternalTokenBufMask;
        // SAFETY: `new` validated the complete backing range and natural
        // alignment, and the retained Arc keeps the allocation alive.
        Some(unsafe { std::slice::from_raw_parts(base.add(start), end - start) })
    }

    #[inline(always)]
    pub(crate) fn for_each_range(&self, start: usize, end: usize, mut visit: impl FnMut(u32, u32)) {
        debug_assert!(start <= end && end <= self.len);
        if start > end || end > self.len {
            return;
        }
        if let Some(entries) = self.slice(start, end) {
            for &entry in entries {
                visit(entry.word_idx, entry.mask);
            }
            return;
        }
        let entry_bytes = std::mem::size_of::<PackedInternalTokenBufMask>();
        let base = unsafe {
            self.backing
                .as_ptr()
                .add(self.entries_start + start * entry_bytes)
        };
        for index in 0..(end - start) {
            let entry = unsafe {
                std::ptr::read_unaligned(
                    base.add(index * entry_bytes)
                        .cast::<PackedInternalTokenBufMask>(),
                )
            };
            if cfg!(target_endian = "little") {
                visit(entry.word_idx, entry.mask);
            } else {
                visit(u32::from_le(entry.word_idx), u32::from_le(entry.mask));
            }
        }
    }
}

/// Contiguous dense-mask matrix used by the word-group prefix cache. The old
/// `Vec<Box<[u32]>>` representation allocated one heap object per row; this
/// keeps the same row-slice API while requiring one aligned allocation.
pub(super) const DENSE_BUF_MASK_ROWS_FLAT_MIN_BYTES: usize = 512 * 1024;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(super) enum DenseBufMaskRowsStorage {
    Rows(Vec<Box<[u32]>>),
    Flat(Box<[u32]>),
    #[serde(skip)]
    Backed {
        backing: Arc<Vec<u8>>,
        /// Address of the first u32 in `backing`. `from_backed` validates the
        /// complete range and alignment once, so hot row lookup does not need
        /// to redo checked byte-offset arithmetic for every prefix row.
        base_addr: usize,
    },
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct DenseBufMaskRows {
    pub(super) storage: DenseBufMaskRowsStorage,
    pub(super) rows: usize,
    pub(super) row_len: usize,
}

impl Default for DenseBufMaskRows {
    fn default() -> Self {
        Self {
            storage: DenseBufMaskRowsStorage::Rows(Vec::new()),
            rows: 0,
            row_len: 0,
        }
    }
}

impl DenseBufMaskRows {
    #[inline]
    pub(crate) fn prefer_flat(rows: usize, row_len: usize) -> bool {
        rows.checked_mul(row_len)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u32>()))
            .is_some_and(|bytes| bytes >= DENSE_BUF_MASK_ROWS_FLAT_MIN_BYTES)
    }

    pub(crate) fn from_flat(flat: Box<[u32]>, rows: usize, row_len: usize) -> Result<Self, String> {
        let expected = rows
            .checked_mul(row_len)
            .ok_or_else(|| "dense mask row dimensions overflow".to_owned())?;
        if flat.len() != expected {
            return Err("dense mask flat length does not match row dimensions".to_owned());
        }
        Ok(Self {
            storage: DenseBufMaskRowsStorage::Flat(flat),
            rows,
            row_len,
        })
    }

    /// Retain a current-format little-endian dense matrix directly in the
    /// artifact allocation. The byte range must be naturally aligned because
    /// callers consume rows as ordinary `&[u32]` slices on the hot path.
    pub(crate) fn from_backed(
        backing: Arc<Vec<u8>>,
        start: usize,
        rows: usize,
        row_len: usize,
    ) -> Result<Self, String> {
        if !cfg!(target_endian = "little") {
            return Err("backed dense mask rows require little-endian host".to_owned());
        }
        let values = rows
            .checked_mul(row_len)
            .ok_or_else(|| "backed dense mask dimensions overflow".to_owned())?;
        let bytes = values
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| "backed dense mask byte length overflow".to_owned())?;
        let end = start
            .checked_add(bytes)
            .ok_or_else(|| "backed dense mask range overflow".to_owned())?;
        if end > backing.len() {
            return Err("backed dense mask range is outside artifact".to_owned());
        }
        let ptr = unsafe { backing.as_ptr().add(start) };
        if ptr.align_offset(std::mem::align_of::<u32>()) != 0 {
            return Err("backed dense mask range is not u32-aligned".to_owned());
        }
        Ok(Self {
            storage: DenseBufMaskRowsStorage::Backed {
                backing,
                base_addr: ptr as usize,
            },
            rows,
            row_len,
        })
    }

    pub(crate) fn from_rows(rows: Vec<Box<[u32]>>) -> Result<Self, String> {
        let row_count = rows.len();
        let row_len = rows.first().map_or(0, |row| row.len());
        if rows.iter().any(|row| row.len() != row_len) {
            return Err("dense mask rows have inconsistent lengths".to_owned());
        }
        Ok(Self {
            storage: DenseBufMaskRowsStorage::Rows(rows),
            rows: row_count,
            row_len,
        })
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.rows
    }

    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.rows == 0
    }

    #[inline]
    pub(crate) fn row_len(&self) -> usize {
        self.row_len
    }

    /// Return the complete dense matrix as one contiguous runtime slice when
    /// storage is already flat/backed. This is the same memory consumed by hot
    /// row lookups; serialization can therefore copy it without rebuilding
    /// row objects.
    #[inline]
    pub(crate) fn as_contiguous(&self) -> Option<&[u32]> {
        match &self.storage {
            DenseBufMaskRowsStorage::Rows(_) => None,
            DenseBufMaskRowsStorage::Flat(flat) => Some(flat),
            DenseBufMaskRowsStorage::Backed {
                backing: _,
                base_addr,
            } => {
                let values = self.rows.checked_mul(self.row_len)?;
                let ptr = *base_addr as *const u32;
                // SAFETY: `from_backed` validated the complete byte range and
                // alignment, and the retained Arc keeps the allocation alive.
                Some(unsafe { std::slice::from_raw_parts(ptr, values) })
            }
        }
    }

    #[inline]
    pub(crate) fn get(&self, row: usize) -> Option<&[u32]> {
        if row >= self.rows {
            return None;
        }
        match &self.storage {
            DenseBufMaskRowsStorage::Rows(rows) => rows.get(row).map(Box::as_ref),
            DenseBufMaskRowsStorage::Flat(flat) => {
                let start = row * self.row_len;
                flat.get(start..start + self.row_len)
            }
            DenseBufMaskRowsStorage::Backed {
                backing: _,
                base_addr,
            } => {
                // `row < self.rows` and `from_backed` validated the complete
                // rows*row_len slab, so this multiplication and pointer offset
                // are within the retained allocation.
                let value_start = row * self.row_len;
                let ptr = unsafe { (*base_addr as *const u32).add(value_start) };
                // SAFETY: `from_backed` validated the full range and alignment;
                // row boundaries advance by a multiple of four bytes.
                Some(unsafe { std::slice::from_raw_parts(ptr, self.row_len) })
            }
        }
    }

    #[inline]
    pub(crate) fn last(&self) -> Option<&[u32]> {
        self.rows.checked_sub(1).and_then(|row| self.get(row))
    }

    #[inline]
    pub(crate) fn iter(&self) -> DenseBufMaskRowsIter<'_> {
        DenseBufMaskRowsIter {
            rows: self,
            next: 0,
        }
    }
}

impl std::ops::Index<usize> for DenseBufMaskRows {
    type Output = [u32];

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).expect("dense mask row index out of bounds")
    }
}

pub(crate) struct DenseBufMaskRowsIter<'a> {
    pub(super) rows: &'a DenseBufMaskRows,
    pub(super) next: usize,
}

impl<'a> Iterator for DenseBufMaskRowsIter<'a> {
    type Item = &'a [u32];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let row = self.rows.get(self.next)?;
        self.next += 1;
        Some(row)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.rows.len().saturating_sub(self.next);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for DenseBufMaskRowsIter<'_> {}

impl<'a> IntoIterator for &'a DenseBufMaskRows {
    type Item = &'a [u32];
    type IntoIter = DenseBufMaskRowsIter<'a>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

pub(crate) type DenseWeightMaskCache = FxHashMap<usize, DenseWords>;

/// Dense masks for selected packed-DWA token sets.
///
/// Keep the rows in one contiguous slab rather than one `Arc<[u64]>` per
/// token set. Besides reducing allocator traffic, this makes the cache cheap
/// to persist and restore as two flat arrays while preserving O(1) lookup by
/// packed token-set id.
#[derive(Debug, Clone, Default)]
pub(crate) struct PackedDwaDenseWeightMaskCache {
    pub(super) words_per_row: usize,
    pub(super) row_by_token_set: Box<[u32]>,
    pub(super) token_set_ids: Box<[u32]>,
    pub(super) rows: DenseWords,
}

impl PackedDwaDenseWeightMaskCache {
    pub(super) const MISSING_ROW: u32 = u32::MAX;

    pub(crate) fn from_rows(
        token_set_count: usize,
        words_per_row: usize,
        mut rows: Vec<(u32, DenseWords)>,
    ) -> Result<Self, String> {
        rows.sort_unstable_by_key(|(id, _)| *id);
        let mut row_by_token_set = vec![Self::MISSING_ROW; token_set_count];
        let mut token_set_ids = Vec::with_capacity(rows.len());
        let mut flat = Vec::with_capacity(rows.len().saturating_mul(words_per_row));
        for (row_index, (id, words)) in rows.into_iter().enumerate() {
            let slot = row_by_token_set
                .get_mut(id as usize)
                .ok_or_else(|| format!("packed DWA dense-mask token-set id {id} out of bounds"))?;
            if *slot != Self::MISSING_ROW {
                return Err(format!("duplicate packed DWA dense-mask token-set id {id}"));
            }
            if words.len() != words_per_row {
                return Err(format!(
                    "packed DWA dense-mask row has {} words; expected {words_per_row}",
                    words.len(),
                ));
            }
            *slot = u32::try_from(row_index)
                .map_err(|_| "too many packed DWA dense-mask rows".to_owned())?;
            token_set_ids.push(id);
            flat.extend_from_slice(words.as_ref());
        }
        Ok(Self {
            words_per_row,
            row_by_token_set: row_by_token_set.into_boxed_slice(),
            token_set_ids: token_set_ids.into_boxed_slice(),
            rows: Arc::from(flat.into_boxed_slice()),
        })
    }

    pub(crate) fn from_flat(
        token_set_count: usize,
        words_per_row: usize,
        token_set_ids: Vec<u32>,
        rows: Vec<u64>,
    ) -> Result<Self, String> {
        if rows.len() != token_set_ids.len().saturating_mul(words_per_row) {
            return Err(format!(
                "packed DWA dense-mask slab has {} words for {} rows of width {words_per_row}",
                rows.len(),
                token_set_ids.len(),
            ));
        }
        let mut row_by_token_set = vec![Self::MISSING_ROW; token_set_count];
        for (row_index, &id) in token_set_ids.iter().enumerate() {
            let slot = row_by_token_set
                .get_mut(id as usize)
                .ok_or_else(|| format!("packed DWA dense-mask token-set id {id} out of bounds"))?;
            if *slot != Self::MISSING_ROW {
                return Err(format!("duplicate packed DWA dense-mask token-set id {id}"));
            }
            *slot = u32::try_from(row_index)
                .map_err(|_| "too many packed DWA dense-mask rows".to_owned())?;
        }
        Ok(Self {
            words_per_row,
            row_by_token_set: row_by_token_set.into_boxed_slice(),
            token_set_ids: token_set_ids.into_boxed_slice(),
            rows: Arc::from(rows.into_boxed_slice()),
        })
    }

    #[inline]
    pub(crate) fn get(&self, id: u32) -> Option<&[u64]> {
        let row = *self.row_by_token_set.get(id as usize)?;
        if row == Self::MISSING_ROW {
            return None;
        }
        let start = (row as usize).checked_mul(self.words_per_row)?;
        self.rows.get(start..start + self.words_per_row)
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.token_set_ids.len()
    }

    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.token_set_ids.is_empty()
    }

    pub(crate) fn clear(&mut self) {
        *self = Self::default();
    }

    #[inline]
    pub(crate) fn words_per_row(&self) -> usize {
        self.words_per_row
    }

    #[inline]
    pub(crate) fn token_set_count(&self) -> usize {
        self.row_by_token_set.len()
    }

    #[inline]
    pub(crate) fn token_set_ids(&self) -> &[u32] {
        &self.token_set_ids
    }

    #[inline]
    pub(crate) fn flat_rows(&self) -> &[u64] {
        self.rows.as_ref()
    }
}

pub(crate) type DenseWeightBufMaskCache = FxHashMap<usize, Box<[u32]>>;

pub(crate) type SparseWeightBufMaskCache = FxHashMap<usize, Box<[(u32, u32)]>>;

pub(crate) type RangeFinalTokenSetCache = FxHashSet<usize>;

pub(crate) type SeedTerminalDenseMasks = FxHashMap<(u32, TerminalID), DenseWords>;
