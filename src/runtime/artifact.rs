use glrmask_artifact::CommitTemplateDfas;
use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, OnceLock};
use rayon::prelude::*;

use rustc_hash::{FxHashMap, FxHashSet, FxHasher};
use smallvec::SmallVec;

use crate::automata::lexer::{
    DFA as LexerDfa, Lexer,
    tokenizer::{TerminalProjectedQuotient, Tokenizer},
};
use crate::automata::lexer::runtime_repeat_product::VirtualBinaryRepeatIntersectionMaskProjection;
use crate::automata::lexer::tokenizer::VirtualResidualMaskProjection;
use crate::automata::lexer::runtime_unit_repeat::VirtualZeroMinUnitRepeatMaskProjection;
use crate::automata::regex::Expr;
use crate::automata::unweighted_u32::dfa::DFA as UnweightedDfa;
use crate::automata::weighted::dwa::{DWA, DwaTransitionMap};
use crate::compiler::glr::labels::DEFAULT_LABEL;
use crate::compiler::glr::parser::{
    ParserComponentTableSource, ParserGSS, ScopedSubgrammarLink,
};
use crate::compiler::glr::table::GLRTable;
use crate::compiler::stages::id_map_and_terminal_dwa::classify::{
    VocabPartitionDfa, classify_vocab_char_type,
};
use crate::compiler::stages::templates::characterize::TerminalCharacterization;
use crate::ds::vocab_prefix_tree::{VocabPrefixTree, VocabPrefixTreeNode};
use crate::ds::weight::Weight;
use crate::grammar::flat::{DirectRegularAutomaton, TerminalID};
use crate::ds::bitset::BitSet;
use crate::ds::u8set::U8Set;

use super::mask_mapping::FinalMaskMapping;

pub(crate) type PossibleMatchesByTerminal = BTreeMap<TerminalID, Weight>;

/// Compile-time detail requested for the reusable dynamic-boundary trigger.
///
/// This setting is orthogonal to whether ordinary component masking is static
/// or dynamic. `None` is the zero-cost default; `Tokens` adds only a
/// parser-state-independent candidate set; `Exact` builds the full
/// GSS-sensitive trigger Parser DWA when the component supports it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum BoundaryTriggerDetail {
    #[default]
    None,
    Tokens,
    Exact,
}

/// Optional reusable hint for dynamic composition boundaries. This is an
/// accelerator only: `None` means the composition runtime must conservatively
/// assume that any model token may cross a component boundary.
#[derive(Debug, Clone, Default)]
pub(crate) enum BoundaryTrigger {
    #[default]
    None,
    /// Conservative original model-token IDs that may contain the first
    /// internal boundary crossing. Parser-state independent.
    Tokens(Arc<[u32]>),
    /// Exact component-local parser DWA. Its coordinate is deliberately
    /// independent of the ordinary parser-DWA quotient: parser labels are
    /// local LR-state IDs, weight TSIDs are raw local tokenizer-state IDs, and
    /// weight token IDs are original/model token IDs. Proper-prefix boundary
    /// behavior is a stronger observation than the ordinary whole-token mask
    /// language, so reusing the normal TSID/internal-token quotient would need
    /// a separate equivalence proof.
    Exact(Arc<DWA>),
}

impl BoundaryTrigger {
    pub(crate) fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }

    pub(crate) fn token_summary(&self) -> Option<&[u32]> {
        match self {
            Self::Tokens(tokens) => Some(tokens),
            Self::None | Self::Exact(_) => None,
        }
    }
}

/// Stable identity of the inputs that make a boundary-candidate summary
/// reusable.  The four pieces are intentionally separate so diagnostics can
/// distinguish stale component semantics, interface changes, and vocabulary
/// mismatches instead of treating every miss as an opaque hash failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct BoundaryCandidateFingerprint {
    pub(crate) algorithm_version: u16,
    pub(crate) component_semantics: [u8; 32],
    pub(crate) public_interface: [u8; 32],
    pub(crate) vocabulary: [u8; 32],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SummaryPrecision {
    RegularUpperBound,
    ContextRefinedUpperBound,
    BudgetWidenedUpperBound,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SummaryUnavailable {
    Disabled,
    Deferred,
    LegacyArtifact,
    MissingGrammarMetadata,
    UnsupportedFiniteLexer,
    InvalidatedBinding,
    FingerprintMismatch,
    MalformedMetadata,
}

/// Original/model-token ID set used by boundary summaries.  Sparse is the
/// canonical wire form; Dense is useful when a component legitimately retains
/// a large irregular set.  `AllByteTokensAtLeastTwo` is the safe top element
/// for the proper-prefix question: one-byte tokens can never contain a
/// non-empty proper byte prefix.
#[derive(Debug, Clone)]
pub(crate) enum OriginalTokenSet {
    Empty,
    Sparse(Arc<[u32]>),
    Dense(Arc<BitSet>),
    AllByteTokensAtLeastTwo,
}

impl OriginalTokenSet {
    pub(crate) fn from_sorted_unique(ids: Vec<u32>, max_token_id: u32) -> Self {
        if ids.is_empty() {
            return Self::Empty;
        }
        // Dense becomes cheaper only when the set is genuinely dense. Keep a
        // deliberately conservative crossover because Sparse is also the
        // canonical persisted representation.
        let dense_words = (max_token_id as usize + 64) / 64;
        if ids.len() > dense_words.saturating_mul(3) {
            let mut bits = BitSet::new(max_token_id as usize + 1);
            for id in ids {
                bits.set(id as usize);
            }
            Self::Dense(Arc::new(bits))
        } else {
            Self::Sparse(Arc::from(ids.into_boxed_slice()))
        }
    }

    pub(crate) fn contains(&self, token_id: u32, bytes: &[u8]) -> bool {
        match self {
            Self::Empty => false,
            Self::Sparse(ids) => ids.binary_search(&token_id).is_ok(),
            Self::Dense(bits) => {
                (token_id as usize) < bits.len() && bits.get(token_id as usize)
            }
            Self::AllByteTokensAtLeastTwo => bytes.len() >= 2,
        }
    }

    pub(crate) fn canonical_ids<'a>(
        &'a self,
        tokens: impl Iterator<Item = (u32, &'a [u8])>,
    ) -> Vec<u32> {
        match self {
            Self::Empty => Vec::new(),
            Self::Sparse(ids) => ids.to_vec(),
            Self::Dense(bits) => bits.iter_ones().map(|id| id as u32).collect(),
            Self::AllByteTokensAtLeastTwo => tokens
                .filter_map(|(id, bytes)| (bytes.len() >= 2).then_some(id))
                .collect(),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) enum BoundaryCandidateSummary {
    Unknown { reason: SummaryUnavailable },
    Known {
        fingerprint: BoundaryCandidateFingerprint,
        tokens: OriginalTokenSet,
        precision: SummaryPrecision,
    },
}

impl Default for BoundaryCandidateSummary {
    fn default() -> Self {
        Self::Unknown {
            reason: SummaryUnavailable::Deferred,
        }
    }
}

impl BoundaryCandidateSummary {
    pub(crate) fn known_tokens_for(
        &self,
        fingerprint: &BoundaryCandidateFingerprint,
    ) -> Option<&OriginalTokenSet> {
        match self {
            Self::Known {
                fingerprint: actual,
                tokens,
                ..
            } if actual == fingerprint => Some(tokens),
            _ => None,
        }
    }

    pub(crate) fn is_known(&self) -> bool {
        matches!(self, Self::Known { .. })
    }
}

/// Small composition-time grammar summary retained with a compiled component.
///
/// For a nonnullable child, substituting the child's language for a parent
/// placeholder needs only:
/// * terminal adjacency (`allowed_follows`),
/// * FIRST/LAST of the component root, and
/// * root nullability.
///
/// Keeping this summary in the outer artifact envelope lets the linker compose
/// grammar legality algebraically instead of rebuilding FIRST/FOLLOW over the
/// fully merged rule graph.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub(crate) struct CompositionGrammarSummary {
    pub(crate) allowed_follows: Vec<BitSet>,
    pub(crate) root_first: BitSet,
    pub(crate) root_last: BitSet,
    pub(crate) root_nullable: bool,
}

#[derive(Debug)]
pub(crate) struct PackedNonDwaWeights {
    pub(crate) pool: Arc<crate::ds::weight::PackedRuntimeWeightPool>,
    pub(crate) parser_top_accept: BTreeMap<i32, u32>,
    pub(crate) parser_top_accept_parts: BTreeMap<i32, Vec<u32>>,
    pub(crate) direct_regular_l1_complete_by_terminal: BTreeMap<TerminalID, u32>,
    pub(crate) possible_matches: BTreeMap<TerminalID, u32>,
}

#[derive(Debug, Clone)]
pub(crate) struct DirectRegularWideFrontierAcceptance {
    /// Pointer identities of immutable replace-target or StackShifts slices in the live table
    /// that all produce this exact frontier. Runtime-only and rebuilt after
    /// deserialization.
    pub(crate) action_origins: Vec<usize>,
    pub(crate) state_count: usize,
    pub(crate) actionable_terminals: crate::ds::bitset::BitSet,
    pub(crate) frontier_states: Arc<[u32]>,
    pub(crate) empty_acc_frontier: ParserGSS,
    pub(crate) acceptance_parts: Arc<[Weight]>,
    pub(crate) dense_by_tsid: Arc<DenseAcceptanceRows>,
    pub(crate) advance_by_terminal: Arc<[(TerminalID, Arc<[u32]>)]>,
}

#[derive(Debug, Clone)]
pub(crate) struct DirectRegularDynamicHotFrontier {
    pub(crate) frontier_states: Arc<[u32]>,
    pub(crate) empty_acc_frontier: ParserGSS,
    pub(crate) actionable_terminals: crate::ds::bitset::BitSet,
    pub(crate) advance_by_terminal: Arc<[(TerminalID, Arc<[u32]>)]>,
}

#[derive(Debug, Clone)]
pub(crate) struct DirectRegularParserStateAcceptance {
    pub(crate) parser_state: u32,
    pub(crate) acceptance_parts: Arc<[Weight]>,
    pub(crate) dense_by_tsid: Arc<DenseAcceptanceRows>,
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
    words_per_row: usize,
    rows: Arc<[u64]>,
    row_kinds: Arc<[u8]>,
    full_dense: DenseWords,
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

pub(crate) type InternalTokenBufMasks = Vec<(u16, u32)>;
/// Runtime-native fixed-width form of one sparse output-mask entry. The two-byte
/// pad makes the layout exactly eight bytes while keeping the hot fields at
/// their natural offsets; current artifacts can therefore bulk-copy the slab
/// without making commit pay bit shifts on every sparse replay.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct PackedInternalTokenBufMask {
    pub(crate) word_idx: u16,
    pub(crate) _pad: u16,
    pub(crate) mask: u32,
}
const _: () = assert!(std::mem::size_of::<PackedInternalTokenBufMask>() == 8);

#[derive(Debug, Clone)]
pub(crate) struct BackedInternalTokenBufMasks {
    backing: Arc<Vec<u8>>,
    entries_start: usize,
    len: usize,
    aligned_base_addr: Option<usize>,
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
    pub(crate) fn slice(
        &self,
        start: usize,
        end: usize,
    ) -> Option<&[PackedInternalTokenBufMask]> {
        if start > end || end > self.len {
            return None;
        }
        let base = self.aligned_base_addr? as *const PackedInternalTokenBufMask;
        // SAFETY: `new` validated the complete backing range and natural
        // alignment, and the retained Arc keeps the allocation alive.
        Some(unsafe { std::slice::from_raw_parts(base.add(start), end - start) })
    }

    #[inline(always)]
    pub(crate) fn for_each_range(
        &self,
        start: usize,
        end: usize,
        mut visit: impl FnMut(u16, u32),
    ) {
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
        let base = unsafe { self.backing.as_ptr().add(self.entries_start + start * entry_bytes) };
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
                visit(u16::from_le(entry.word_idx), u32::from_le(entry.mask));
            }
        }
    }
}

/// Contiguous dense-mask matrix used by the word-group prefix cache. The old
/// `Vec<Box<[u32]>>` representation allocated one heap object per row; this
/// keeps the same row-slice API while requiring one aligned allocation.
const DENSE_BUF_MASK_ROWS_FLAT_MIN_BYTES: usize = 512 * 1024;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
enum DenseBufMaskRowsStorage {
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
    storage: DenseBufMaskRowsStorage,
    rows: usize,
    row_len: usize,
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

    pub(crate) fn from_flat(
        flat: Box<[u32]>,
        rows: usize,
        row_len: usize,
    ) -> Result<Self, String> {
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
    rows: &'a DenseBufMaskRows,
    next: usize,
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
    words_per_row: usize,
    row_by_token_set: Box<[u32]>,
    token_set_ids: Box<[u32]>,
    rows: DenseWords,
}

impl PackedDwaDenseWeightMaskCache {
    const MISSING_ROW: u32 = u32::MAX;

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
pub(crate) type SparseWeightBufMaskCache = FxHashMap<usize, Box<[(u16, u32)]>>;
pub(crate) type DirectSparseWeightTokenSetCache = FxHashSet<usize>;
pub(crate) type SeedTerminalDenseMasks = FxHashMap<(u32, TerminalID), DenseWords>;
const INLINE_DWA_TRANSITION_LIMIT: usize = 8;

#[derive(Debug, Clone)]
pub(crate) enum FastDwaTransitionRow {
    Inline(SmallVec<[(i32, (u32, Weight)); 4]>),
    Hash(FxHashMap<i32, (u32, Weight)>),
    Packed(DwaTransitionMap),
}

impl FastDwaTransitionRow {
    pub(crate) fn from_entries(
        entries: impl IntoIterator<Item = (i32, (u32, Weight))>,
    ) -> Self {
        let entries = entries.into_iter().collect::<SmallVec<[_; 4]>>();
        if entries.len() <= INLINE_DWA_TRANSITION_LIMIT {
            Self::Inline(entries)
        } else {
            Self::Hash(entries.into_iter().collect())
        }
    }

    pub(crate) fn from_exact_entries(
        len: usize,
        entries: impl IntoIterator<Item = (i32, (u32, Weight))>,
    ) -> Self {
        if len <= INLINE_DWA_TRANSITION_LIMIT {
            Self::Inline(entries.into_iter().collect())
        } else {
            let mut map = FxHashMap::default();
            map.reserve(len);
            map.extend(entries);
            Self::Hash(map)
        }
    }

    pub(crate) fn from_packed(row: DwaTransitionMap) -> Self {
        debug_assert!(row.is_packed());
        Self::Packed(row)
    }

    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        match self {
            Self::Inline(entries) => entries.is_empty(),
            Self::Hash(entries) => entries.is_empty(),
            Self::Packed(row) => row.is_empty(),
        }
    }

    #[inline]
    pub(crate) fn get(&self, label: &i32) -> Option<(u32, &Weight)> {
        match self {
            Self::Inline(entries) => entries
                .iter()
                .find_map(|(candidate, (target, weight))| {
                    (candidate == label).then_some((*target, weight))
                }),
            Self::Hash(entries) => entries.get(label).map(|(target, weight)| (*target, weight)),
            Self::Packed(row) => row.get_entry(label),
        }
    }
}
#[derive(Debug, Clone)]
pub(crate) enum FastDwaTransitions {
    Direct(Vec<FastDwaTransitionRow>),
    Shared {
        rows: Vec<FastDwaTransitionRow>,
        state_rows: Vec<u32>,
    },
}

impl Default for FastDwaTransitions {
    fn default() -> Self {
        Self::Direct(Vec::new())
    }
}

impl FastDwaTransitions {
    pub(crate) fn direct(rows: Vec<FastDwaTransitionRow>) -> Self {
        Self::Direct(rows)
    }

    pub(crate) fn shared(rows: Vec<FastDwaTransitionRow>, state_rows: Vec<u32>) -> Self {
        Self::Shared { rows, state_rows }
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Direct(rows) => rows.len(),
            Self::Shared { state_rows, .. } => state_rows.len(),
        }
    }

    #[inline]
    pub(crate) fn get(&self, state: usize) -> Option<&FastDwaTransitionRow> {
        match self {
            Self::Direct(rows) => rows.get(state),
            Self::Shared { rows, state_rows } => state_rows
                .get(state)
                .and_then(|&row| rows.get(row as usize)),
        }
    }
}

impl std::ops::Index<usize> for FastDwaTransitions {
    type Output = FastDwaTransitionRow;

    #[inline]
    fn index(&self, state: usize) -> &Self::Output {
        match self {
            Self::Direct(rows) => &rows[state],
            Self::Shared { rows, state_rows } => &rows[state_rows[state] as usize],
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) enum IndexedDagDenseMask {
    Full,
    Dense {
        words: DenseWords,
        start: usize,
        end: usize,
    },
    Empty,
}

#[derive(Debug, Clone)]
pub(crate) struct IndexedDagDenseTransition {
    pub(crate) target: u32,
    pub(crate) masks: IndexedDagDenseTransitionMasks,
}

const INLINE_INDEXED_DAG_TSID_LIMIT: usize = 8;

#[derive(Debug, Clone)]
pub(crate) enum IndexedDagDenseTransitionMasks {
    Full,
    Inline(SmallVec<[(u32, IndexedDagDenseMask); 2]>),
    Hash(FxHashMap<u32, IndexedDagDenseMask>),
}

static INDEXED_DAG_FULL_MASK: IndexedDagDenseMask = IndexedDagDenseMask::Full;
static INDEXED_DAG_EMPTY_MASK: IndexedDagDenseMask = IndexedDagDenseMask::Empty;

impl IndexedDagDenseTransitionMasks {
    pub(crate) fn from_entries(
        entries: impl IntoIterator<Item = (u32, IndexedDagDenseMask)>,
    ) -> Self {
        let entries = entries.into_iter().collect::<SmallVec<[_; 2]>>();
        if entries.len() <= INLINE_INDEXED_DAG_TSID_LIMIT {
            Self::Inline(entries)
        } else {
            Self::Hash(entries.into_iter().collect())
        }
    }

    #[inline]
    pub(crate) fn get(&self, tsid: u32) -> &IndexedDagDenseMask {
        match self {
            Self::Full => &INDEXED_DAG_FULL_MASK,
            Self::Inline(entries) => entries
                .iter()
                .find_map(|(candidate, mask)| (*candidate == tsid).then_some(mask))
                .unwrap_or(&INDEXED_DAG_EMPTY_MASK),
            Self::Hash(entries) => entries.get(&tsid).unwrap_or(&INDEXED_DAG_EMPTY_MASK),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) enum IndexedDagDenseTransitionRow {
    Inline(SmallVec<[(i32, IndexedDagDenseTransition); 4]>),
    Hash(FxHashMap<i32, IndexedDagDenseTransition>),
}

impl IndexedDagDenseTransitionRow {
    pub(crate) fn from_entries(
        entries: impl IntoIterator<Item = (i32, IndexedDagDenseTransition)>,
    ) -> Self {
        let entries = entries.into_iter().collect::<SmallVec<[_; 4]>>();
        if entries.len() <= INLINE_DWA_TRANSITION_LIMIT {
            Self::Inline(entries)
        } else {
            Self::Hash(entries.into_iter().collect())
        }
    }

    #[inline]
    pub(crate) fn get(&self, label: &i32) -> Option<&IndexedDagDenseTransition> {
        match self {
            Self::Inline(entries) => entries
                .iter()
                .find_map(|(candidate, transition)| (candidate == label).then_some(transition)),
            Self::Hash(entries) => entries.get(label),
        }
    }
}

pub(crate) type IndexedDagDenseTransitions = Vec<IndexedDagDenseTransitionRow>;

#[derive(Debug, Clone)]
pub(crate) enum FastTokenizerTransitions {
    Dense(Vec<Box<[u32; 256]>>),
    Flat(Arc<[u32]>),
    /// Compact exact dense DFA rows for tokenizers with fewer than 32,768
    /// states. The high bit marks a target state with at least one finalizer;
    /// u16::MAX is the dead-transition sentinel.
    Flat16 {
        transitions: Arc<[u16]>,
        finalizer_code: Arc<[u32]>,
        single_finalizer_continues: Arc<[u8]>,
    },
    /// Exact dense DFA rows for the strict full-vocabulary walker when the
    /// tokenizer no longer fits the 15-bit Flat16 coordinate. The high bit
    /// marks a target state with at least one finalizer; u32::MAX is dead.
    Flat32 {
        transitions: Arc<[u32]>,
        finalizer_code: Arc<[u32]>,
        single_finalizer_continues: Arc<[u8]>,
    },
    /// Runtime tokenizer already owns an allocation-light exact transition
    /// table; call through instead of rebuilding a second dense table.
    Fallback(usize),
    Hybrid {
        state_to_dense_row: Vec<u32>,
        dense_rows: Vec<Box<[u32; 256]>>,
    },
}

impl Default for FastTokenizerTransitions {
    fn default() -> Self {
        Self::Dense(Vec::new())
    }
}

impl FastTokenizerTransitions {
    fn full_walk_finalizer_metadata(tokenizer: &Tokenizer) -> (Arc<[u32]>, Arc<[u8]>) {
        const NONE: u32 = u32::MAX;
        const MULTI: u32 = u32::MAX - 1;
        let num_states = tokenizer.num_states();
        let state_metadata = |state: u32| {
            let finalizers = tokenizer.matched_terminals_slice(state);
            let code = match finalizers {
                [] => NONE,
                [terminal] => *terminal,
                _ => MULTI,
            };
            let continues = match finalizers {
                [terminal]
                    if tokenizer
                        .possible_future_terminals(state)
                        .contains(*terminal as usize) => 1u8,
                _ => 0u8,
            };
            (code, continues)
        };
        let metadata = if num_states >= 1_024 && rayon::current_num_threads() > 1 {
            (0..num_states)
                .into_par_iter()
                .map(state_metadata)
                .collect::<Vec<_>>()
        } else {
            (0..num_states).map(state_metadata).collect::<Vec<_>>()
        };
        let (finalizer_code, single_finalizer_continues): (Vec<_>, Vec<_>) =
            metadata.into_iter().unzip();
        (
            Arc::from(finalizer_code),
            Arc::from(single_finalizer_continues),
        )
    }

    pub(crate) fn flat16_for(tokenizer: &Tokenizer) -> Option<Self> {
        let num_states = tokenizer.num_states();
        if num_states >= 0x8000 {
            return None;
        }
        // Build finalizer metadata once per state before populating transition
        // cells. The old implementation called `matched_terminals_slice()` for
        // every edge merely to set the high-bit hint, then scanned every state
        // again below to construct these exact metadata arrays. Fresh compiler
        // DFAs make that per-edge lookup materially more expensive than loaded
        // packed tokenizers.
        let (finalizer_code, single_finalizer_continues) =
            Self::full_walk_finalizer_metadata(tokenizer);
        let mut flat = vec![u16::MAX; num_states as usize * 256];
        let fill_row = |(state, row): (usize, &mut [u16])| {
            for (byte, target) in tokenizer.transitions_from(state as u32) {
                let mut encoded = u16::try_from(target)
                    .expect("flat16 tokenizer target exceeds 15-bit state coordinate");
                if finalizer_code[target as usize] != u32::MAX {
                    encoded |= 0x8000;
                }
                row[byte as usize] = encoded;
            }
        };
        if num_states >= 1_024 && rayon::current_num_threads() > 1 {
            flat.par_chunks_mut(256).enumerate().for_each(fill_row);
        } else {
            flat.chunks_mut(256).enumerate().for_each(fill_row);
        }
        Some(Self::Flat16 {
            transitions: Arc::from(flat),
            finalizer_code,
            single_finalizer_continues,
        })
    }

    /// Dense u16 transition slab for ordinary tokenizer execution. Unlike the
    /// strict full-vocabulary walker, commit/scan only asks this object for the
    /// target state, so constructing per-state finalizer certificates (and
    /// probing finalizers for every transition target) is pure load-time waste.
    pub(crate) fn flat16_transitions_only_for(tokenizer: &Tokenizer) -> Option<Self> {
        let num_states = tokenizer.num_states();
        if num_states >= 0x8000 {
            return None;
        }
        let mut flat = vec![u16::MAX; num_states as usize * 256];
        let fill_row = |(state, row): (usize, &mut [u16])| {
            for (byte, target) in tokenizer.transitions_from(state as u32) {
                row[byte as usize] = u16::try_from(target)
                    .expect("flat16 tokenizer target exceeds 15-bit state coordinate");
            }
        };
        if num_states >= 1_024 && rayon::current_num_threads() > 1 {
            flat.par_chunks_mut(256).enumerate().for_each(fill_row);
        } else {
            flat.chunks_mut(256).enumerate().for_each(fill_row);
        }
        Some(Self::Flat16 {
            transitions: Arc::from(flat),
            finalizer_code: Arc::from([]),
            single_finalizer_continues: Arc::from([]),
        })
    }

    pub(crate) fn flat32_for(tokenizer: &Tokenizer) -> Option<Self> {
        let num_states = tokenizer.num_states();
        if num_states >= 0x8000_0000 {
            return None;
        }
        let (finalizer_code, single_finalizer_continues) =
            Self::full_walk_finalizer_metadata(tokenizer);
        let cell_count = (num_states as usize).checked_mul(256)?;
        let mut flat = vec![u32::MAX; cell_count];
        let fill_row = |(state, row): (usize, &mut [u32])| {
            for (byte, target) in tokenizer.transitions_from(state as u32) {
                debug_assert!(target < 0x8000_0000);
                let mut encoded = target;
                if finalizer_code[target as usize] != u32::MAX {
                    encoded |= 0x8000_0000;
                }
                row[byte as usize] = encoded;
            }
        };
        if num_states >= 1_024 && rayon::current_num_threads() > 1 {
            flat.par_chunks_mut(256).enumerate().for_each(fill_row);
        } else {
            flat.chunks_mut(256).enumerate().for_each(fill_row);
        }
        Some(Self::Flat32 {
            transitions: Arc::from(flat),
            finalizer_code,
            single_finalizer_continues,
        })
    }

    /// Build the dense transition representation used only by the strict full
    /// vocabulary walker. Flat16 is preferred whenever possible; Flat32 is
    /// available for larger deterministic mask coordinates. `max_bytes`
    /// bounds only transition-cell storage, not the small metadata arrays.
    pub(crate) fn full_walk_dense_for(tokenizer: &Tokenizer, max_bytes: usize) -> Option<Self> {
        let num_states = tokenizer.num_states() as usize;
        let flat16_bytes = num_states
            .checked_mul(256)?
            .checked_mul(std::mem::size_of::<u16>())?;
        if num_states < 0x8000 && flat16_bytes <= max_bytes {
            return Self::flat16_for(tokenizer);
        }
        let flat32_bytes = num_states
            .checked_mul(256)?
            .checked_mul(std::mem::size_of::<u32>())?;
        if num_states < 0x8000_0000 && flat32_bytes <= max_bytes {
            return Self::flat32_for(tokenizer);
        }
        None
    }

    #[inline]
    pub(crate) fn transition(
        &self,
        tokenizer: &Tokenizer,
        state: u32,
        byte: u8,
    ) -> u32 {
        match self {
            Self::Dense(rows) => rows
                .get(state as usize)
                .map_or_else(
                    || tokenizer.get_transition(state, byte),
                    |row| row[byte as usize],
                ),
            Self::Flat(flat) => state
                .try_into()
                .ok()
                .and_then(|state: usize| state.checked_mul(256))
                .and_then(|offset| offset.checked_add(byte as usize))
                .and_then(|index| flat.get(index))
                .copied()
                .unwrap_or_else(|| tokenizer.get_transition(state, byte)),
            Self::Flat16 { transitions, .. } => state
                .try_into()
                .ok()
                .and_then(|state: usize| state.checked_mul(256))
                .and_then(|offset| offset.checked_add(byte as usize))
                .and_then(|index| transitions.get(index))
                .copied()
                .map(|target| {
                    if target == u16::MAX {
                        u32::MAX
                    } else {
                        u32::from(target & 0x7fff)
                    }
                })
                .unwrap_or_else(|| tokenizer.get_transition(state, byte)),
            Self::Flat32 { transitions, .. } => state
                .try_into()
                .ok()
                .and_then(|state: usize| state.checked_mul(256))
                .and_then(|offset| offset.checked_add(byte as usize))
                .and_then(|index| transitions.get(index))
                .copied()
                .map(|target| {
                    if target == u32::MAX {
                        u32::MAX
                    } else {
                        target & 0x7fff_ffff
                    }
                })
                .unwrap_or_else(|| tokenizer.get_transition(state, byte)),
            Self::Fallback(_) => tokenizer.get_transition(state, byte),
            Self::Hybrid {
                state_to_dense_row,
                dense_rows,
            } => {
                let dense = state_to_dense_row
                    .get(state as usize)
                    .copied()
                    .unwrap_or(u32::MAX);
                if dense == u32::MAX {
                    tokenizer.get_transition(state, byte)
                } else {
                    dense_rows[dense as usize][byte as usize]
                }
            }
        }
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Dense(rows) => rows.len(),
            Self::Flat(flat) => flat.len() / 256,
            Self::Flat16 { transitions, .. } => transitions.len() / 256,
            Self::Flat32 { transitions, .. } => transitions.len() / 256,
            Self::Fallback(len) => *len,
            Self::Hybrid {
                state_to_dense_row,
                ..
            } => state_to_dense_row.len(),
        }
    }

    /// Reuse the consumed parent's fast transition rows and append rebased
    /// child rows. Compressed child states remain sparse and fall back to the
    /// merged tokenizer, whose compressed segments have already been rebased.
    pub(crate) fn append_rebased_children(
        self,
        children: &[(&FastTokenizerTransitions, u32)],
    ) -> Option<Self> {
        fn flat_rows(flat: &[u32]) -> Option<Vec<Box<[u32; 256]>>> {
            let chunks = flat.chunks_exact(256);
            if !chunks.remainder().is_empty() {
                return None;
            }
            chunks
                .map(|chunk| {
                    let row: &[u32; 256] = chunk.try_into().ok()?;
                    Some(Box::new(*row))
                })
                .collect()
        }

        fn rebased_row(row: &[u32; 256], offset: u32) -> Box<[u32; 256]> {
            let mut rebased = Box::new(*row);
            for target in rebased.iter_mut() {
                if *target != u32::MAX {
                    *target = target.checked_add(offset)
                        .expect("composed tokenizer fast-transition target overflow");
                }
            }
            rebased
        }

        let all_dense = children
            .iter()
            .all(|(child, _)| matches!(child, FastTokenizerTransitions::Dense(_)));
        match self {
            Self::Dense(mut rows) if all_dense => {
                for (child, offset) in children {
                    if *offset as usize != rows.len() {
                        return None;
                    }
                    let Self::Dense(child_rows) = child else { unreachable!() };
                    rows.extend(child_rows.iter().map(|row| rebased_row(row, *offset)));
                }
                Some(Self::Dense(rows))
            }
            parent => {
                let (mut state_to_dense_row, mut dense_rows) = match parent {
                    Self::Dense(rows) => {
                        let state_to_dense_row = (0..rows.len() as u32).collect::<Vec<_>>();
                        (state_to_dense_row, rows)
                    }
                    Self::Flat(flat) => {
                        let rows = flat_rows(&flat)?;
                        let state_to_dense_row = (0..rows.len() as u32).collect::<Vec<_>>();
                        (state_to_dense_row, rows)
                    }
                    Self::Flat16 { .. } | Self::Flat32 { .. } | Self::Fallback(_) => return None,
                    Self::Hybrid {
                        state_to_dense_row,
                        dense_rows,
                    } => (state_to_dense_row, dense_rows),
                };
                for (child, offset) in children {
                    if *offset as usize != state_to_dense_row.len() {
                        return None;
                    }
                    match child {
                        Self::Flat16 { .. } | Self::Flat32 { .. } | Self::Fallback(_) => return None,
                        Self::Dense(rows) => {
                            for row in rows {
                                let dense = dense_rows.len() as u32;
                                dense_rows.push(rebased_row(row, *offset));
                                state_to_dense_row.push(dense);
                            }
                        }
                        Self::Flat(flat) => {
                            let rows = flat_rows(flat)?;
                            for row in rows {
                                let dense = dense_rows.len() as u32;
                                dense_rows.push(rebased_row(&row, *offset));
                                state_to_dense_row.push(dense);
                            }
                        }
                        Self::Hybrid {
                            state_to_dense_row: child_mapping,
                            dense_rows: child_rows,
                        } => {
                            for &child_dense in child_mapping {
                                if child_dense == u32::MAX {
                                    state_to_dense_row.push(u32::MAX);
                                } else {
                                    let row = child_rows.get(child_dense as usize)?;
                                    let dense = dense_rows.len() as u32;
                                    dense_rows.push(rebased_row(row, *offset));
                                    state_to_dense_row.push(dense);
                                }
                            }
                        }
                    }
                }
                Some(Self::Hybrid {
                    state_to_dense_row,
                    dense_rows,
                })
            }
        }
    }
}
pub(crate) type TemplateDfasByTerminal = Vec<Option<Arc<CommitTemplateDfas>>>;
pub(crate) type FastTemplateDfasByTerminal = Vec<Option<Arc<FastCommitTemplateDfas>>>;

const INLINE_TEMPLATE_TRANSITION_LIMIT: usize = 8;

#[derive(Debug, Clone, Default)]
pub(crate) enum FastTemplateTransitionRow {
    #[default]
    Empty,
    Inline(SmallVec<[(i32, u32); 4]>),
    Hash(FxHashMap<i32, u32>),
}

impl FastTemplateTransitionRow {
    fn from_entries(entries: impl IntoIterator<Item = (i32, u32)>) -> Self {
        let entries = entries.into_iter().collect::<SmallVec<[_; 4]>>();
        match entries.len() {
            0 => Self::Empty,
            len if len <= INLINE_TEMPLATE_TRANSITION_LIMIT => Self::Inline(entries),
            _ => Self::Hash(entries.into_iter().collect()),
        }
    }

    #[inline]
    pub(crate) fn get(&self, label: i32) -> Option<u32> {
        match self {
            Self::Empty => None,
            Self::Inline(entries) => entries
                .iter()
                .find_map(|(candidate, target)| (*candidate == label).then_some(*target)),
            Self::Hash(entries) => entries.get(&label).copied(),
        }
    }

    #[inline]
    pub(crate) fn for_each(&self, mut f: impl FnMut(i32, u32)) {
        match self {
            Self::Empty => {}
            Self::Inline(entries) => {
                for &(label, target) in entries {
                    f(label, target);
                }
            }
            Self::Hash(entries) => {
                for (&label, &target) in entries {
                    f(label, target);
                }
            }
        }
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct FastTemplateDfaState {
    pub(crate) is_accepting: bool,
    pub(crate) default_target: Option<u32>,
    pub(crate) transitions: FastTemplateTransitionRow,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct FastTemplateDfa {
    pub(crate) states: Vec<FastTemplateDfaState>,
    pub(crate) start_state: u32,
}

impl FastTemplateDfa {
    fn from_dfa(dfa: &UnweightedDfa) -> Self {
        Self {
            states: dfa
                .states
                .iter()
                .map(|state| FastTemplateDfaState {
                    is_accepting: state.is_accepting,
                    default_target: state.transitions.get(&DEFAULT_LABEL).copied(),
                    transitions: FastTemplateTransitionRow::from_entries(
                        state
                            .transitions
                            .iter()
                            .filter(|(label, _)| **label != DEFAULT_LABEL)
                            .map(|(&label, &target)| (label, target)),
                    ),
                })
                .collect(),
            start_state: dfa.start_state,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct FastCommitTemplateDfas {
    pub(crate) pop: FastTemplateDfa,
    pub(crate) read: FastTemplateDfa,
    pub(crate) push: FastTemplateDfa,
    pub(crate) pop_to_read: Vec<Option<u32>>,
    pub(crate) pop_to_push: Vec<Option<u32>>,
    pub(crate) read_to_push: Vec<Option<u32>>,
}

impl FastCommitTemplateDfas {
    pub(crate) fn from_template(template: &CommitTemplateDfas) -> Self {
        Self {
            pop: FastTemplateDfa::from_dfa(&template.pop),
            read: FastTemplateDfa::from_dfa(&template.read),
            push: FastTemplateDfa::from_dfa(&template.push),
            pop_to_read: template.pop_to_read.clone(),
            pop_to_push: template.pop_to_push.clone(),
            read_to_push: template.read_to_push.clone(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct SpecialTokenTerminal {
    pub(crate) terminal_id: TerminalID,
    pub(crate) token_id: u32,
}

/// Compact runtime-only vocabulary trie. It deliberately stores only the
/// information dynamic mask traversal consumes: compressed byte edges, child
/// ranges, and canonical token leaves.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct DynamicMaskTrieNode {
    pub(crate) token_id: Option<u32>,
    pub(crate) first_child: u32,
    pub(crate) child_len: u32,
    /// Canonical token ids below this node occupy one contiguous range in
    /// `DynamicMaskTrie::subtree_tokens`.
    pub(crate) subtree_token_start: u32,
    pub(crate) subtree_token_end: u32,
    /// Union of every byte on every edge strictly below this node.
    pub(crate) subtree_bytes: [u64; 4],
    /// Bytes that may occur next on some non-empty token suffix below this
    /// node. Unlike `subtree_bytes`, this records only the first consumed byte;
    /// zero-byte structural layout edges transparently inherit their child's
    /// first-byte set. Dynamic masking uses it to reject whole vocabulary
    /// layout classes before entering the radix walk when the current lexer
    /// configuration cannot consume any of their first bytes.
    pub(crate) subtree_first_bytes: [u64; 4],
    /// Number of vocabulary bytes consumed from the global trie root to reach
    /// this node. Structural partition edges have length zero and therefore do
    /// not affect it. This lets finite-horizon root-state certificates be
    /// reused only for subtrees whose *complete token strings* fit the proof
    /// horizon.
    pub(crate) prefix_byte_len: u32,
    /// Maximum number of token bytes still reachable strictly below this node.
    /// This is a runtime-only certificate aid: dynamic masking can prove that
    /// a lexer configuration stays safely live for this bounded horizon and
    /// accept the whole subtree without walking every token edge.
    pub(crate) subtree_max_byte_len: u32,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct DynamicMaskTrieEdge {
    pub(crate) byte_start: u32,
    pub(crate) byte_len: u32,
    pub(crate) child: u32,
}

/// One radix edge in depth-first preorder. `subtree_end` is the first walk
/// entry after the child subtree, so a failed edge or accepted whole subtree
/// can be skipped with one index assignment.
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct DynamicMaskTrieWalkEdge {
    pub(crate) byte_start: u32,
    pub(crate) child: u32,
    pub(crate) subtree_end: u32,
    pub(crate) byte_len: u16,
    pub(crate) parent_depth: u16,
}

/// One sequential operation in the strict full-vocabulary byte walk. Every
/// radix edge contributes at least one op; non-empty edges contribute one op
/// per consumed byte. This is purely a flatter view of the vocabulary trie: it
/// does not omit or summarize any edge.
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct DynamicMaskTrieFullWalkOp {
    meta: u32,
}

const _: () = assert!(std::mem::size_of::<DynamicMaskTrieFullWalkOp>() == 4);

impl DynamicMaskTrieFullWalkOp {
    const CONSUME: u32 = 1 << 24;
    const START: u32 = 1 << 25;
    const END: u32 = 1 << 26;
    const TOKEN: u32 = 1 << 27;

    #[inline(always)]
    pub(crate) fn byte(&self) -> u8 { self.meta as u8 }
    #[inline(always)]
    pub(crate) fn parent_depth(&self) -> u16 { ((self.meta >> 8) & 0xffff) as u16 }
    #[inline(always)]
    pub(crate) fn consumes_byte(&self) -> bool { self.meta & Self::CONSUME != 0 }
    #[inline(always)]
    pub(crate) fn starts_edge(&self) -> bool { self.meta & Self::START != 0 }
    #[inline(always)]
    pub(crate) fn ends_edge(&self) -> bool { self.meta & Self::END != 0 }
    #[inline(always)]
    pub(crate) fn child_is_token(&self) -> bool { self.meta & Self::TOKEN != 0 }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct DynamicMaskTrie {
    pub(crate) nodes: Vec<DynamicMaskTrieNode>,
    pub(crate) edges: Vec<DynamicMaskTrieEdge>,
    edge_bytes: Vec<u8>,
    subtree_tokens: Vec<u32>,
    walk_edges: Vec<DynamicMaskTrieWalkEdge>,
    full_walk_ops: Vec<DynamicMaskTrieFullWalkOp>,
    /// Edge index owning each strict full-walk op. Read only after death.
    full_walk_op_edges: Vec<u32>,
    /// Starting full-walk op index for each DFS-preorder walk edge, plus one
    /// sentinel at the end. Combined with `DynamicMaskTrieWalkEdge::subtree_end`
    /// this gives an O(1) exact jump past a dead child subtree.
    full_walk_edge_op_starts: Vec<u32>,
    /// For an ordinary, unpartitioned radix root, map each possible first byte
    /// directly to the first strict-walk op for its root child. `u32::MAX`
    /// means the vocabulary has no token beginning with that byte. Structural
    /// zero-byte roots deliberately leave this unavailable.
    #[serde(with = "optional_u32_256_serde")]
    full_walk_root_byte_op_starts: Option<Box<[u32; 256]>>,
    full_walk_token_nodes: Vec<u32>,
    /// Maximum structural radix-edge parent depth encoded by `full_walk_ops`.
    /// This is deliberately distinct from token byte length: one long
    /// compressed edge consumes many bytes without requiring additional DFS
    /// stack slots in the strict full walker.
    full_walk_max_parent_depth: u16,
    /// Declared regular-language partition id for each zero-byte structural
    /// child of the true root. An empty table disables partition certificates.
    root_layout_classes: Vec<u16>,
    /// True iff every complete token in the corresponding structural root
    /// class is valid UTF-8. Logical-scalar subtree proofs require this exact
    /// vocabulary property before treating non-ASCII bytes as UTF-8 scalars.
    root_layout_all_valid_utf8: Vec<bool>,
}

mod optional_u32_256_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(
        value: &Option<Box<[u32; 256]>>,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        value.as_deref().map(|row| row.as_slice()).serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Box<[u32; 256]>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Option::<Vec<u32>>::deserialize(deserializer)?;
        value
            .map(|row| {
                row.try_into()
                    .map(Box::new)
                    .map_err(|row: Vec<u32>| serde::de::Error::custom(format!(
                        "expected 256 root-byte entries, got {}",
                        row.len()
                    )))
            })
            .transpose()
    }
}

/// Stable structural class used by the dynamic-mask radix trie.
///
/// This is deliberately *only* the declared regular-language partition id.
/// Older revisions refined it with arbitrary first-byte/character flags; that
/// made one structural root cease to correspond to a first-class language and
/// therefore prevented exact language-containment proofs. Any further runtime
/// acceleration must be represented as an explicit regular language instead.
pub(crate) fn dynamic_mask_vocab_layout_class(base_partition: u8, _bytes: &[u8]) -> u16 {
    u16::from(base_partition)
}

pub(crate) const DYNAMIC_MASK_LLG_MASTER_CACHE_ID: u32 = 0x200;
const DYNAMIC_MASK_LLG_MASTER_WHITESPACE_BIT: u16 = 1 << 15;
const DYNAMIC_MASK_LLG_MASTER_SAFE_LEN_MASK: u16 = DYNAMIC_MASK_LLG_MASTER_WHITESPACE_BIT - 1;

/// Structural class for the dynamic-radius LLG vocabulary trie. The low bits
/// are the exact number of Unicode scalar values in a whole token matching the
/// regex-defined safe-string language (`0` means not in that language); the high
/// bit records whole-token whitespace-regex membership. These are language
/// properties only. The trie deliberately does not mark them as compiler
/// partition languages, so generic partition certificates cannot reinterpret
/// the encoding.
pub(crate) fn dynamic_mask_llg_master_layout_class(
    safe_chars: u16,
    whitespace: bool,
) -> u16 {
    debug_assert!(safe_chars <= DYNAMIC_MASK_LLG_MASTER_SAFE_LEN_MASK);
    safe_chars | if whitespace { DYNAMIC_MASK_LLG_MASTER_WHITESPACE_BIT } else { 0 }
}

#[inline(always)]
pub(crate) fn dynamic_mask_llg_master_safe_chars(class: u16) -> u16 {
    class & DYNAMIC_MASK_LLG_MASTER_SAFE_LEN_MASK
}

#[inline(always)]
pub(crate) fn dynamic_mask_llg_master_is_whitespace(class: u16) -> bool {
    class & DYNAMIC_MASK_LLG_MASTER_WHITESPACE_BIT != 0
}

impl DynamicMaskTrie {
    pub(crate) fn new() -> Self {
        Self {
            nodes: vec![DynamicMaskTrieNode::default()],
            edges: Vec::new(),
            edge_bytes: Vec::new(),
            subtree_tokens: Vec::new(),
            walk_edges: Vec::new(),
            full_walk_ops: Vec::new(),
            full_walk_op_edges: Vec::new(),
            full_walk_edge_op_starts: Vec::new(),
            full_walk_root_byte_op_starts: None,
            full_walk_token_nodes: Vec::new(),
            full_walk_max_parent_depth: 0,
            root_layout_classes: Vec::new(),
            root_layout_all_valid_utf8: Vec::new(),
        }
    }


    #[inline]
    pub(crate) fn root_layout_class(&self, root_slot: usize) -> Option<u16> {
        self.root_layout_classes.get(root_slot).copied()
    }

    #[inline]
    pub(crate) fn root_layout_all_valid_utf8(&self, root_slot: usize) -> bool {
        self.root_layout_all_valid_utf8
            .get(root_slot)
            .copied()
            .unwrap_or(false)
    }
    #[inline]
    pub(crate) fn node(&self, node: u32) -> &DynamicMaskTrieNode {
        &self.nodes[node as usize]
    }

    #[inline]
    pub(crate) fn node_count(&self) -> usize {
        self.nodes.len()
    }

    #[inline]
    pub(crate) fn children(&self, node: u32) -> &[DynamicMaskTrieEdge] {
        let node = self.node(node);
        let start = node.first_child as usize;
        let end = start + node.child_len as usize;
        &self.edges[start..end]
    }

    #[inline]
    pub(crate) fn edge_bytes(&self, edge: &DynamicMaskTrieEdge) -> &[u8] {
        let start = edge.byte_start as usize;
        let end = start + edge.byte_len as usize;
        &self.edge_bytes[start..end]
    }

    #[inline]
    pub(crate) fn walk_edges(&self) -> &[DynamicMaskTrieWalkEdge] {
        &self.walk_edges
    }

    #[inline]
    pub(crate) fn walk_edge_bytes(&self, edge: &DynamicMaskTrieWalkEdge) -> &[u8] {
        let start = edge.byte_start as usize;
        let end = start + edge.byte_len as usize;
        &self.edge_bytes[start..end]
    }

    #[inline]
    pub(crate) fn full_walk_ops(&self) -> &[DynamicMaskTrieFullWalkOp] {
        &self.full_walk_ops
    }

    /// Exact strict-walk operation count after skipping structural root classes
    /// accepted by `admitted`.  Each admitted class still costs its one
    /// zero-byte structural root op; the remainder of that root subtree is
    /// skipped in O(1) by the runtime walker.
    pub(crate) fn full_walk_ops_after_root_class_skips(
        &self,
        mut admitted: impl FnMut(u16) -> bool,
    ) -> Option<usize> {
        if self.root_layout_classes.is_empty() {
            return None;
        }
        let mut root_slot = 0usize;
        let mut work = 0usize;
        for (edge_index, edge) in self.walk_edges.iter().copied().enumerate() {
            if edge.parent_depth != 0 {
                continue;
            }
            let class = *self.root_layout_classes.get(root_slot)?;
            root_slot += 1;
            let start = *self.full_walk_edge_op_starts.get(edge_index)? as usize;
            let end = *self.full_walk_edge_op_starts.get(edge.subtree_end as usize)? as usize;
            work = work.saturating_add(if admitted(class) { 1 } else { end.saturating_sub(start) });
        }
        (root_slot == self.root_layout_classes.len()).then_some(work)
    }

    /// Every full-walk op consumes one byte iff there are no synthetic
    /// empty-edge ops. Non-empty radix edges contribute exactly one op per
    /// byte; empty edges contribute one op and zero bytes.
    #[inline(always)]
    pub(crate) fn full_walk_all_consume(&self) -> bool {
        self.full_walk_ops.len() == self.edge_bytes.len()
    }

    #[inline(always)]
    pub(crate) fn full_walk_dead_subtree(&self, op_index: usize) -> (u32, u32) {
        debug_assert!(op_index < self.full_walk_op_edges.len());
        let edge_index = unsafe { *self.full_walk_op_edges.get_unchecked(op_index) } as usize;
        let edge = unsafe { *self.walk_edges.get_unchecked(edge_index) };
        let subtree_end_edge = edge.subtree_end as usize;
        debug_assert!(subtree_end_edge < self.full_walk_edge_op_starts.len());
        let subtree_end_op =
            unsafe { *self.full_walk_edge_op_starts.get_unchecked(subtree_end_edge) };
        (edge.child, subtree_end_op)
    }

    #[inline]
    pub(crate) fn full_walk_token_nodes(&self) -> &[u32] {
        &self.full_walk_token_nodes
    }

    #[inline]
    pub(crate) fn full_walk_max_parent_depth(&self) -> u16 {
        self.full_walk_max_parent_depth
    }

    #[inline(always)]
    pub(crate) fn has_full_walk_root_byte_index(&self) -> bool {
        self.full_walk_root_byte_op_starts.is_some()
    }

    #[inline(always)]
    pub(crate) fn full_walk_root_byte_range(&self, byte: u8) -> Option<(u32, u32, usize)> {
        let starts = self.full_walk_root_byte_op_starts.as_ref()?;
        let start_op = starts[byte as usize];
        if start_op == u32::MAX {
            return None;
        }
        let (child, end_op) = self.full_walk_dead_subtree(start_op as usize);
        let root_token_offset = usize::from(self.node(0).token_id.is_some());
        let marker_start = self
            .subtree_token_index_range(child)
            .start
            .saturating_sub(root_token_offset);
        Some((start_op, end_op, marker_start))
    }

    #[inline]
    pub(crate) fn subtree_tokens(&self, node: u32) -> &[u32] {
        let node = self.node(node);
        &self.subtree_tokens
            [node.subtree_token_start as usize..node.subtree_token_end as usize]
    }

    #[inline]
    pub(crate) fn subtree_token_index_range(&self, node: u32) -> std::ops::Range<usize> {
        let node = self.node(node);
        node.subtree_token_start as usize..node.subtree_token_end as usize
    }

    #[inline]
    pub(crate) fn all_subtree_tokens(&self) -> &[u32] {
        &self.subtree_tokens
    }

    #[inline]
    pub(crate) fn subtree_bytes(&self, node: u32) -> [u64; 4] {
        self.node(node).subtree_bytes
    }

    #[inline]
    pub(crate) fn subtree_first_bytes(&self, node: u32) -> [u64; 4] {
        self.node(node).subtree_first_bytes
    }

    #[inline]
    pub(crate) fn subtree_max_byte_len(&self, node: u32) -> u32 {
        self.node(node).subtree_max_byte_len
    }

    #[inline]
    pub(crate) fn subtree_max_total_byte_len(&self, node: u32) -> u32 {
        let node = self.node(node);
        node.prefix_byte_len.saturating_add(node.subtree_max_byte_len)
    }

    pub(crate) fn push_edge_bytes(&mut self, bytes: &[u8]) -> (u32, u32) {
        let start = self.edge_bytes.len() as u32;
        self.edge_bytes.extend_from_slice(bytes);
        (start, bytes.len() as u32)
    }

    #[inline]
    pub(crate) fn edge_bytes_len(&self) -> usize {
        self.edge_bytes.len()
    }

    fn collect_subtree_metadata(
        &mut self,
        node_id: u32,
        prefix_byte_len: u32,
    ) -> ([u64; 4], [u64; 4], u32) {
        self.nodes[node_id as usize].prefix_byte_len = prefix_byte_len;
        let start = self.subtree_tokens.len() as u32;
        if let Some(token_id) = self.nodes[node_id as usize].token_id {
            self.subtree_tokens.push(token_id);
        }

        let first_child = self.nodes[node_id as usize].first_child as usize;
        let child_len = self.nodes[node_id as usize].child_len as usize;
        let mut subtree_bytes = [0u64; 4];
        let mut subtree_first_bytes = [0u64; 4];
        let mut subtree_max_byte_len = 0u32;
        for edge_index in first_child..first_child + child_len {
            // Copy the compact edge fields before recursing so no borrow of
            // `self.edges` remains live across the mutable recursive call.
            let edge = self.edges[edge_index].clone();
            let byte_start = edge.byte_start as usize;
            let byte_end = byte_start + edge.byte_len as usize;
            for &byte in &self.edge_bytes[byte_start..byte_end] {
                subtree_bytes[byte as usize >> 6] |= 1u64 << (byte & 63);
            }
            let child_prefix_byte_len = prefix_byte_len
                .checked_add(edge.byte_len)
                .expect("dynamic mask trie token byte length exceeds u32");
            let (child_bytes, child_first_bytes, child_max_byte_len) =
                self.collect_subtree_metadata(edge.child, child_prefix_byte_len);
            for (target, child) in subtree_bytes.iter_mut().zip(child_bytes) {
                *target |= child;
            }
            if edge.byte_len == 0 {
                for (target, child) in subtree_first_bytes.iter_mut().zip(child_first_bytes) {
                    *target |= child;
                }
            } else {
                let first = self.edge_bytes[byte_start];
                subtree_first_bytes[first as usize >> 6] |= 1u64 << (first & 63);
            }
            subtree_max_byte_len = subtree_max_byte_len.max(
                edge.byte_len
                    .checked_add(child_max_byte_len)
                    .expect("dynamic mask trie token byte length exceeds u32"),
            );
        }

        let end = self.subtree_tokens.len() as u32;
        let node = &mut self.nodes[node_id as usize];
        node.subtree_token_start = start;
        node.subtree_token_end = end;
        node.subtree_bytes = subtree_bytes;
        node.subtree_first_bytes = subtree_first_bytes;
        node.subtree_max_byte_len = subtree_max_byte_len;
        (subtree_bytes, subtree_first_bytes, subtree_max_byte_len)
    }

    pub(crate) fn finalize_subtree_metadata(&mut self) {
        self.subtree_tokens.clear();
        self.subtree_tokens.reserve(self.nodes.len());
        if !self.nodes.is_empty() {
            self.collect_subtree_metadata(0, 0);
        }
        self.finalize_walk_edges();
    }

    fn append_walk_edges(&mut self, node_id: u32, parent_depth: u16) {
        let first_child = self.nodes[node_id as usize].first_child as usize;
        let child_len = self.nodes[node_id as usize].child_len as usize;
        for edge_index in first_child..first_child + child_len {
            let edge = self.edges[edge_index].clone();
            let byte_len = u16::try_from(edge.byte_len)
                .expect("dynamic mask trie radix edge exceeds u16 length");
            let entry_index = self.walk_edges.len();
            self.walk_edges.push(DynamicMaskTrieWalkEdge {
                byte_start: edge.byte_start,
                child: edge.child,
                subtree_end: 0,
                byte_len,
                parent_depth,
            });
            self.append_walk_edges(
                edge.child,
                parent_depth
                    .checked_add(1)
                    .expect("dynamic mask trie depth exceeds u16"),
            );
            self.walk_edges[entry_index].subtree_end = self.walk_edges.len() as u32;
        }
    }

    fn finalize_walk_edges(&mut self) {
        self.walk_edges.clear();
        self.walk_edges.reserve(self.edges.len());
        if !self.nodes.is_empty() {
            self.append_walk_edges(0, 0);
        }
        debug_assert_eq!(self.walk_edges.len(), self.edges.len());

        self.full_walk_ops.clear();
        self.full_walk_op_edges.clear();
        self.full_walk_edge_op_starts.clear();
        self.full_walk_root_byte_op_starts = None;
        self.full_walk_token_nodes.clear();
        self.full_walk_max_parent_depth = self
            .walk_edges
            .iter()
            .map(|edge| edge.parent_depth)
            .max()
            .unwrap_or(0);
        self.full_walk_ops.reserve(self.edge_bytes.len().max(self.walk_edges.len()));
        self.full_walk_op_edges
            .reserve(self.edge_bytes.len().max(self.walk_edges.len()));
        self.full_walk_edge_op_starts.reserve(self.walk_edges.len() + 1);
        self.full_walk_token_nodes.reserve(self.walk_edges.len());
        for (edge_index, edge) in self.walk_edges.iter().copied().enumerate() {
            self.full_walk_edge_op_starts
                .push(self.full_walk_ops.len() as u32);
            let start = edge.byte_start as usize;
            let end = start + edge.byte_len as usize;
            let bytes = &self.edge_bytes[start..end];
            let depth = u32::from(edge.parent_depth) << 8;
            let child_is_token = self.nodes[edge.child as usize].token_id.is_some();
            if bytes.is_empty() {
                let mut meta = depth
                    | DynamicMaskTrieFullWalkOp::START
                    | DynamicMaskTrieFullWalkOp::END;
                if child_is_token { meta |= DynamicMaskTrieFullWalkOp::TOKEN; }
                self.full_walk_ops.push(DynamicMaskTrieFullWalkOp { meta });
                self.full_walk_op_edges.push(edge_index as u32);
                if child_is_token { self.full_walk_token_nodes.push(edge.child); }
                continue;
            }
            for (index, &byte) in bytes.iter().enumerate() {
                let mut meta = u32::from(byte) | depth | DynamicMaskTrieFullWalkOp::CONSUME;
                if index == 0 { meta |= DynamicMaskTrieFullWalkOp::START; }
                let last = index + 1 == bytes.len();
                if last {
                    meta |= DynamicMaskTrieFullWalkOp::END;
                    if child_is_token { meta |= DynamicMaskTrieFullWalkOp::TOKEN; }
                }
                self.full_walk_ops.push(DynamicMaskTrieFullWalkOp { meta });
                self.full_walk_op_edges.push(edge_index as u32);
                if last && child_is_token { self.full_walk_token_nodes.push(edge.child); }
            }
        }
        self.full_walk_edge_op_starts
            .push(self.full_walk_ops.len() as u32);
        debug_assert_eq!(self.full_walk_op_edges.len(), self.full_walk_ops.len());
        debug_assert_eq!(
            self.full_walk_edge_op_starts.len(),
            self.walk_edges.len() + 1
        );

        // Ordinary radix tries have one non-empty edge per distinct first byte
        // below the true root. Index those DFS ranges once so sparse lexer
        // roots can enter only byte-live vocabulary subtrees. Partitioned
        // tries have zero-byte structural root edges and intentionally decline
        // this optimization rather than conflating layout with byte language.
        let mut root_byte_starts = Box::new([u32::MAX; 256]);
        let mut root_byte_index_valid = true;
        for (edge_index, edge) in self.walk_edges.iter().copied().enumerate() {
            if edge.parent_depth != 0 {
                continue;
            }
            let bytes = self.walk_edge_bytes(&edge);
            let Some(&first) = bytes.first() else {
                root_byte_index_valid = false;
                break;
            };
            let slot = &mut root_byte_starts[first as usize];
            if *slot != u32::MAX {
                root_byte_index_valid = false;
                break;
            }
            *slot = self.full_walk_edge_op_starts[edge_index];
        }
        if root_byte_index_valid {
            self.full_walk_root_byte_op_starts = Some(root_byte_starts);
        }
    }

    fn flatten_vocab_node(node: &VocabPrefixTreeNode, output: &mut Self) -> u32 {
        let node_id = output.nodes.len() as u32;
        output.nodes.push(DynamicMaskTrieNode {
            token_id: node.has_token().then_some(node.token_id() as u32),
            first_child: 0,
            child_len: 0,
            subtree_token_start: 0,
            subtree_token_end: 0,
            subtree_bytes: [0; 4],
            subtree_first_bytes: [0; 4],
            prefix_byte_len: 0,
            subtree_max_byte_len: 0,
        });

        let children = node.children();
        if children.is_empty() {
            return node_id;
        }

        let first_child = output.edges.len() as u32;
        output
            .edges
            .resize_with(output.edges.len() + children.len(), DynamicMaskTrieEdge::default);
        output.nodes[node_id as usize].first_child = first_child;
        output.nodes[node_id as usize].child_len = children.len() as u32;

        for (offset, (segment, child)) in node.iter_children().enumerate() {
            let child_id = Self::flatten_vocab_node(child, output);
            let (byte_start, byte_len) = output.push_edge_bytes(segment);
            output.edges[first_child as usize + offset] = DynamicMaskTrieEdge {
                byte_start,
                byte_len,
                child: child_id,
            };
        }

        node_id
    }

    fn from_vocab_prefix_tree_node(node: &VocabPrefixTreeNode) -> Self {
        let mut output = Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            edge_bytes: Vec::new(),
            subtree_tokens: Vec::new(),
            walk_edges: Vec::new(),
            full_walk_ops: Vec::new(),
            full_walk_op_edges: Vec::new(),
            full_walk_edge_op_starts: Vec::new(),
            full_walk_root_byte_op_starts: None,
            full_walk_token_nodes: Vec::new(),
            full_walk_max_parent_depth: 0,
            root_layout_classes: Vec::new(),
            root_layout_all_valid_utf8: Vec::new(),
        };
        let root = Self::flatten_vocab_node(node, &mut output);
        debug_assert_eq!(root, 0);
        output.finalize_subtree_metadata();
        output
    }

    pub(crate) fn from_vocab_prefix_tree(tree: &VocabPrefixTree) -> Self {
        // Root children are disjoint lexical subtrees. Flattening them in
        // parallel is safe, then the compact fragments are stitched with fixed
        // index offsets. This keeps the runtime representation lean without
        // making finalization wait on a single 140k-node recursive walk.
        let root = &tree.root;
        let root_children = root.children();
        if rayon::current_num_threads() == 1 || root_children.len() < 8 {
            return Self::from_vocab_prefix_tree_node(root);
        }

        let root_prefix_len = root.prefix().len();
        let mut fragments: Vec<(Box<[u8]>, Self)> = root_children
            .par_iter()
            .map(|child| {
                let edge = child.prefix()[root_prefix_len..].to_vec().into_boxed_slice();
                (edge, Self::from_vocab_prefix_tree_node(child))
            })
            .collect();
        let node_capacity = 1 + fragments.iter().map(|(_, fragment)| fragment.nodes.len()).sum::<usize>();
        let edge_capacity = root_children.len()
            + fragments.iter().map(|(_, fragment)| fragment.edges.len()).sum::<usize>();
        let byte_capacity = fragments
            .iter()
            .map(|(edge, fragment)| edge.len() + fragment.edge_bytes.len())
            .sum::<usize>();
        let mut output = Self {
            nodes: Vec::with_capacity(node_capacity),
            edges: Vec::with_capacity(edge_capacity),
            edge_bytes: Vec::with_capacity(byte_capacity),
            subtree_tokens: Vec::with_capacity(node_capacity),
            walk_edges: Vec::with_capacity(edge_capacity),
            full_walk_ops: Vec::with_capacity(byte_capacity.max(edge_capacity)),
            full_walk_op_edges: Vec::with_capacity(byte_capacity.max(edge_capacity)),
            full_walk_edge_op_starts: Vec::with_capacity(edge_capacity + 1),
            full_walk_root_byte_op_starts: None,
            full_walk_token_nodes: Vec::with_capacity(edge_capacity),
            full_walk_max_parent_depth: 0,
            root_layout_classes: Vec::new(),
            root_layout_all_valid_utf8: Vec::new(),
        };
        output.nodes.push(DynamicMaskTrieNode {
            token_id: root.has_token().then_some(root.token_id() as u32),
            first_child: 0,
            child_len: root_children.len() as u32,
            subtree_token_start: 0,
            subtree_token_end: 0,
            subtree_bytes: [0; 4],
            subtree_first_bytes: [0; 4],
            prefix_byte_len: 0,
            subtree_max_byte_len: 0,
        });
        output
            .edges
            .resize_with(root_children.len(), DynamicMaskTrieEdge::default);

        for (root_slot, (root_edge, mut fragment)) in fragments.drain(..).enumerate() {
            let node_base = output.nodes.len() as u32;
            let edge_base = output.edges.len() as u32;
            let byte_base = output.edge_bytes.len() as u32;
            output.edge_bytes.extend_from_slice(&fragment.edge_bytes);
            for node in &mut fragment.nodes {
                if node.child_len != 0 {
                    node.first_child += edge_base;
                }
            }
            for edge in &mut fragment.edges {
                edge.byte_start += byte_base;
                edge.child += node_base;
            }
            output.nodes.append(&mut fragment.nodes);
            output.edges.append(&mut fragment.edges);
            let (byte_start, byte_len) = output.push_edge_bytes(&root_edge);
            output.edges[root_slot] = DynamicMaskTrieEdge {
                byte_start,
                byte_len,
                child: node_base,
            };
        }

        output.finalize_subtree_metadata();
        output
    }

    /// Build the same flat runtime radix-trie representation, but place one
    /// zero-byte structural node above each caller-supplied vocabulary layout
    /// class. `entries` must be ordered by `(class, token_bytes)` and token byte
    /// strings must already be canonical/deduplicated.
    ///
    /// The structural edges are a layout device only: they consume no input
    /// and are invisible to lexer semantics. Their purpose is to keep token
    /// families with different byte behaviour from contaminating one another's
    /// subtree metadata, so the generic runtime subtree certificates can skip
    /// large groups without any partition-specific masking logic.
    pub(crate) fn from_partitioned_token_refs(entries: &[(u16, usize, &[u8])]) -> Self {
        static DIRECT_FLAT: OnceLock<bool> = OnceLock::new();
        if *DIRECT_FLAT.get_or_init(|| std::env::var("GLRMASK_DIRECT_FLAT_MASK_TRIE")
            .map_or(true, |value| matches!(value.as_str(), "1" | "true")))
        {
            return Self::from_partitioned_sorted_direct(entries);
        }
        Self::from_partitioned_token_refs_reference(entries)
    }

    fn from_partitioned_token_refs_reference(entries: &[(u16, usize, &[u8])]) -> Self {
        let mut output = Self::new();
        if entries.is_empty() {
            return output;
        }

        // Empty-token aliases are canonicalized before this stage, so at most
        // one canonical empty byte string may exist. Keep it on the true root.
        let mut start = 0usize;
        if entries[0].2.is_empty() {
            output.nodes[0].token_id = Some(entries[0].1 as u32);
            start = 1;
        }

        let mut groups = Vec::<Self>::new();
        let mut group_classes = Vec::<u16>::new();
        let mut group_all_valid_utf8 = Vec::<bool>::new();
        let mut index = start;
        while index < entries.len() {
            let class = entries[index].0;
            let group_start = index;
            index += 1;
            while index < entries.len() && entries[index].0 == class {
                index += 1;
            }
            let refs = entries[group_start..index]
                .iter()
                .map(|(_, token_id, bytes)| (*token_id, *bytes))
                .collect::<Vec<_>>();
            debug_assert!(refs.windows(2).all(|pair| pair[0].1 <= pair[1].1));
            group_classes.push(class);
            group_all_valid_utf8.push(
                entries[group_start..index]
                    .iter()
                    .all(|(_, _, bytes)| std::str::from_utf8(bytes).is_ok()),
            );
            let tree = VocabPrefixTree::build_presorted(&refs);
            groups.push(Self::from_vocab_prefix_tree_node(&tree.root));
        }

        let root_child_count = groups.len();
        output.edges.resize_with(root_child_count, DynamicMaskTrieEdge::default);
        output.nodes[0].first_child = 0;
        output.nodes[0].child_len = root_child_count as u32;

        for (root_slot, mut fragment) in groups.into_iter().enumerate() {
            let node_base = output.nodes.len() as u32;
            let edge_base = output.edges.len() as u32;
            let byte_base = output.edge_bytes.len() as u32;

            output.edge_bytes.extend_from_slice(&fragment.edge_bytes);
            for node in &mut fragment.nodes {
                if node.child_len != 0 {
                    node.first_child += edge_base;
                }
            }
            for edge in &mut fragment.edges {
                edge.byte_start += byte_base;
                edge.child += node_base;
            }
            output.nodes.append(&mut fragment.nodes);
            output.edges.append(&mut fragment.edges);

            // Structural class edge: no lexer byte is consumed here.
            let (byte_start, byte_len) = output.push_edge_bytes(&[]);
            output.edges[root_slot] = DynamicMaskTrieEdge {
                byte_start,
                byte_len,
                child: node_base,
            };
        }

        output.finalize_subtree_metadata();
        output.root_layout_classes = group_classes;
        output.root_layout_all_valid_utf8 = group_all_valid_utf8;
        output
    }
}

impl DynamicMaskTrie {
    /// Direct flat construction from sorted canonical entries. The iterative
    /// worklist preserves the reference's preorder nodes/postorder byte ranges
    /// without transient prefix trees, copied whole prefixes or range-set
    /// metadata. Existing subtree/walk metadata finalization is unchanged.
    fn from_partitioned_sorted_direct(entries: &[(u16, usize, &[u8])]) -> Self {
        enum Task {
            Node { lo: usize, hi: usize, depth: usize, edge_from: usize, parent_edge: usize },
            FinishEdge { entry: usize, from: usize, to: usize, edge: usize, child: u32 },
        }
        let mut output = Self::new();
        if entries.is_empty() { return output; }
        let mut index = 0;
        if entries[0].2.is_empty() {
            output.nodes[0].token_id = Some(entries[0].1 as u32);
            index = 1;
        }
        let mut groups = Vec::new();
        while index < entries.len() {
            let start = index; let class = entries[index].0; index += 1;
            while index < entries.len() && entries[index].0 == class { index += 1; }
            groups.push((start,index));
        }
        output.nodes.reserve(entries.len().saturating_mul(2));
        output.edges.reserve(entries.len().saturating_mul(2));
        output.edge_bytes.reserve(entries.iter().map(|e|e.2.len()).sum());
        output.edges.resize_with(groups.len(), DynamicMaskTrieEdge::default);
        output.nodes[0].child_len = groups.len() as u32;
        for &(lo,hi) in &groups {
            output.root_layout_classes.push(entries[lo].0);
            output.root_layout_all_valid_utf8.push(entries[lo..hi].iter().all(|e|std::str::from_utf8(e.2).is_ok()));
        }
        let mut work = Vec::with_capacity(64);
        for (parent_edge,&(lo,hi)) in groups.iter().enumerate().rev() {
            work.push(Task::Node {lo,hi,depth:0,edge_from:0,parent_edge});
        }
        while let Some(task) = work.pop() {
            match task {
                Task::FinishEdge {entry,from,to,edge,child} => {
                    let (byte_start,byte_len)=output.push_edge_bytes(&entries[entry].2[from..to]);
                    output.edges[edge]=DynamicMaskTrieEdge {byte_start,byte_len,child};
                }
                Task::Node {mut lo,hi,depth,edge_from,parent_edge} => {
                    let node=output.nodes.len() as u32;
                    output.nodes.push(DynamicMaskTrieNode::default());
                    work.push(Task::FinishEdge {entry:lo,from:edge_from,to:depth,edge:parent_edge,child:node});
                    if entries[lo].2.len()==depth {
                        output.nodes[node as usize].token_id=Some(entries[lo].1 as u32);
                        lo+=1;
                    }
                    if lo==hi {continue;}
                    let child_count=1+entries[lo..hi].windows(2)
                        .filter(|p|p[0].2[depth]!=p[1].2[depth]).count();
                    let first=output.edges.len();
                    output.edges.resize_with(first+child_count,DynamicMaskTrieEdge::default);
                    output.nodes[node as usize].first_child=first as u32;
                    output.nodes[node as usize].child_len=child_count as u32;
                    let mut end=hi;let mut slot=child_count;
                    while end>lo {
                        let byte=entries[end-1].2[depth];
                        let mut start=end-1;
                        while start>lo && entries[start-1].2[depth]==byte {start-=1;}
                        let a=entries[start].2;let b=entries[end-1].2;
                        let mut common=depth+1;
                        while common<a.len().min(b.len()) && a[common]==b[common] {common+=1;}
                        slot-=1;
                        work.push(Task::Node {lo:start,hi:end,depth:common,edge_from:depth,parent_edge:first+slot});
                        end=start;
                    }
                }
            }
        }
        output.finalize_subtree_metadata();
        output
    }
}


impl Default for DynamicMaskTrie {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) enum PackedDynamicMaskTokenAliases {
    Single(u32),
    Many(Box<[u32]>),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) enum DynamicMaskAliasStore {
    Ordered(Arc<Vec<Vec<u32>>>),
    /// Canonical alias groups already flattened into the runtime hot layout.
    /// Offsets has `canonical_count + 1` entries into `originals`.
    Flat {
        offsets: Arc<Vec<u32>>,
        originals: Arc<Vec<u32>>,
    },
    Packed(Arc<Vec<Option<PackedDynamicMaskTokenAliases>>>),
}

#[derive(Debug)]
struct DynamicMaskCacheEntry {
    hash: u64,
    state: DynamicMaskStateKey,
    mask: DynamicMaskCachePayload,
}

#[derive(Debug)]
enum DynamicMaskCachePayload {
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
struct DynamicMaskCache {
    entry_cap: Option<usize>,
    entries: Vec<Option<DynamicMaskCacheEntry>>,
    by_hash: FxHashMap<u64, SmallVec<[usize; 1]>>,
    next_slot: usize,
}

#[inline]
pub(crate) fn dynamic_mask_state_key_hash(state: &DynamicMaskStateKey) -> u64 {
    let mut hasher = FxHasher::default();
    state.hash(&mut hasher);
    hasher.finish()
}


#[derive(Debug, Clone, Copy)]
enum DirectRegularSupportNode {
    Leaf(u64),
    Branch(u32, u32),
}

#[derive(Debug, Clone, Copy)]
struct DirectRegularSmallSupport {
    len: u8,
    terminals: [u16; 4],
}

impl DirectRegularSmallSupport {
    const UNAVAILABLE: u8 = u8::MAX;

    fn unavailable() -> Self {
        Self {
            len: Self::UNAVAILABLE,
            terminals: [0; 4],
        }
    }

    fn from_leaf(mut value: u64) -> Self {
        if value.count_ones() > 4 {
            return Self::unavailable();
        }
        let mut result = Self {
            len: 0,
            terminals: [0; 4],
        };
        while value != 0 {
            result.terminals[result.len as usize] = value.trailing_zeros() as u16;
            result.len += 1;
            value &= value - 1;
        }
        result
    }

    fn combine(left: Self, right: Self, right_offset: usize) -> Self {
        if left.len == Self::UNAVAILABLE
            || right.len == Self::UNAVAILABLE
            || usize::from(left.len) + usize::from(right.len) > 4
            || right_offset > u16::MAX as usize
        {
            return Self::unavailable();
        }
        let mut result = Self {
            len: left.len + right.len,
            terminals: [0; 4],
        };
        result.terminals[..left.len as usize]
            .copy_from_slice(&left.terminals[..left.len as usize]);
        for (index, &terminal) in right.terminals[..right.len as usize].iter().enumerate() {
            let Some(terminal) = usize::from(terminal).checked_add(right_offset) else {
                return Self::unavailable();
            };
            let Ok(terminal) = u16::try_from(terminal) else {
                return Self::unavailable();
            };
            result.terminals[left.len as usize + index] = terminal;
        }
        result
    }

    fn terminals(&self) -> Option<&[u16]> {
        (self.len != Self::UNAVAILABLE).then(|| &self.terminals[..self.len as usize])
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct DirectRegularTerminalSupport {
    roots: Vec<u32>,
    nodes: Vec<DirectRegularSupportNode>,
    node_counts: Vec<u16>,
    node_small_support: Vec<DirectRegularSmallSupport>,
    dense_state_rows: FxHashMap<u32, Arc<[u64]>>,
    zero: Vec<u32>,
    levels: u8,
    num_terminals: usize,
}

struct DirectRegularTerminalSupportBuilder {
    nodes: Vec<DirectRegularSupportNode>,
    node_counts: Vec<u16>,
    node_small_support: Vec<DirectRegularSmallSupport>,
    leaf_intern: FxHashMap<u64, u32>,
    branch_intern: Vec<FxHashMap<(u32, u32), u32>>,
    union_memo: Vec<FxHashMap<(u32, u32), u32>>,
    zero: Vec<u32>,
}

impl DirectRegularTerminalSupportBuilder {
    fn new(levels: usize) -> Self {
        let mut builder = Self {
            nodes: Vec::new(),
            node_counts: Vec::new(),
            node_small_support: Vec::new(),
            leaf_intern: FxHashMap::default(),
            branch_intern: (0..=levels).map(|_| FxHashMap::default()).collect(),
            union_memo: (0..=levels).map(|_| FxHashMap::default()).collect(),
            zero: Vec::with_capacity(levels + 1),
        };
        let leaf = builder.intern_leaf(0);
        builder.zero.push(leaf);
        for level in 1..=levels {
            let child = builder.zero[level - 1];
            let root = builder.intern_branch(level, child, child);
            builder.zero.push(root);
        }
        builder
    }

    fn intern_leaf(&mut self, value: u64) -> u32 {
        if let Some(&id) = self.leaf_intern.get(&value) {
            return id;
        }
        let id = self.nodes.len() as u32;
        self.nodes.push(DirectRegularSupportNode::Leaf(value));
        self.node_counts.push(value.count_ones() as u16);
        self.node_small_support
            .push(DirectRegularSmallSupport::from_leaf(value));
        self.leaf_intern.insert(value, id);
        id
    }

    fn intern_branch(&mut self, level: usize, left: u32, right: u32) -> u32 {
        if let Some(&id) = self.branch_intern[level].get(&(left, right)) {
            return id;
        }
        let id = self.nodes.len() as u32;
        self.nodes
            .push(DirectRegularSupportNode::Branch(left, right));
        self.node_counts.push(
            self.node_counts[left as usize].saturating_add(self.node_counts[right as usize]),
        );
        let right_offset = 64usize << (level - 1);
        self.node_small_support.push(DirectRegularSmallSupport::combine(
            self.node_small_support[left as usize],
            self.node_small_support[right as usize],
            right_offset,
        ));
        self.branch_intern[level].insert((left, right), id);
        id
    }

    fn union(&mut self, level: usize, left: u32, right: u32) -> u32 {
        if left == right {
            return left;
        }
        if left == self.zero[level] {
            return right;
        }
        if right == self.zero[level] {
            return left;
        }
        let key = if left < right {
            (left, right)
        } else {
            (right, left)
        };
        if let Some(&id) = self.union_memo[level].get(&key) {
            return id;
        }
        let result = if level == 0 {
            let DirectRegularSupportNode::Leaf(left_value) = self.nodes[left as usize] else {
                unreachable!()
            };
            let DirectRegularSupportNode::Leaf(right_value) = self.nodes[right as usize] else {
                unreachable!()
            };
            self.intern_leaf(left_value | right_value)
        } else {
            let DirectRegularSupportNode::Branch(left_a, left_b) = self.nodes[left as usize] else {
                unreachable!()
            };
            let DirectRegularSupportNode::Branch(right_a, right_b) = self.nodes[right as usize]
            else {
                unreachable!()
            };
            let a = self.union(level - 1, left_a, right_a);
            let b = self.union(level - 1, left_b, right_b);
            self.intern_branch(level, a, b)
        };
        self.union_memo[level].insert(key, result);
        result
    }

    fn singleton(&mut self, levels: usize, terminal: usize) -> u32 {
        let word = terminal / 64;
        let mut node = self.intern_leaf(1u64 << (terminal % 64));
        for level in 1..=levels {
            let zero = self.zero[level - 1];
            node = if ((word >> (level - 1)) & 1) == 0 {
                self.intern_branch(level, node, zero)
            } else {
                self.intern_branch(level, zero, node)
            };
        }
        node
    }
}

impl DirectRegularTerminalSupport {
    pub(crate) fn build(automaton: &DirectRegularAutomaton, num_terminals: usize) -> Self {
        if automaton.states.is_empty() || num_terminals == 0 {
            return Self::default();
        }
        let word_count = num_terminals.div_ceil(64).next_power_of_two();
        let levels = word_count.trailing_zeros() as usize;
        let mut builder = DirectRegularTerminalSupportBuilder::new(levels);
        let singletons = (0..num_terminals)
            .map(|terminal| builder.singleton(levels, terminal))
            .collect::<Vec<_>>();

        let mut parents = vec![Vec::<u32>::new(); automaton.states.len()];
        let mut remaining_children = Vec::<u32>::with_capacity(automaton.states.len());
        let mut queue = VecDeque::<u32>::new();
        for (source, state) in automaton.states.iter().enumerate() {
            remaining_children.push(state.epsilons.len() as u32);
            if state.epsilons.is_empty() {
                queue.push_back(source as u32);
            }
            for &child in &state.epsilons {
                parents[child as usize].push(source as u32);
            }
        }

        let mut roots = vec![builder.zero[levels]; automaton.states.len()];
        let mut processed = 0usize;
        while let Some(raw) = queue.pop_front() {
            let state = &automaton.states[raw as usize];
            let mut root = builder.zero[levels];
            for &terminal in state.transitions.keys() {
                if (terminal as usize) < num_terminals {
                    root = builder.union(levels, root, singletons[terminal as usize]);
                }
            }
            for &child in &state.epsilons {
                root = builder.union(levels, root, roots[child as usize]);
            }
            roots[raw as usize] = root;
            processed += 1;
            for &parent in &parents[raw as usize] {
                let remaining = &mut remaining_children[parent as usize];
                *remaining -= 1;
                if *remaining == 0 {
                    queue.push_back(parent);
                }
            }
        }
        if processed != automaton.states.len() {
            return Self::default();
        }
        let mut support = Self {
            roots,
            nodes: builder.nodes,
            node_counts: builder.node_counts,
            node_small_support: builder.node_small_support,
            dense_state_rows: FxHashMap::default(),
            zero: builder.zero,
            levels: levels as u8,
            num_terminals,
        };
        let dense_word_count = num_terminals.div_ceil(64);
        for &raw_state in &automaton.start_states {
            let mut words = vec![0u64; dense_word_count];
            support.or_state_into(raw_state, &mut words);
            support
                .dense_state_rows
                .insert(raw_state, Arc::from(words));
        }
        support
    }

    pub(crate) fn is_initialized(&self) -> bool {
        !self.roots.is_empty()
    }

    pub(crate) fn for_each_small_state_terminal(
        &self,
        raw_state: u32,
        mut visit: impl FnMut(TerminalID),
    ) -> bool {
        let Some(root) = self.root_id(raw_state) else {
            return false;
        };
        let Some(terminals) = self.node_small_support[root as usize].terminals() else {
            return false;
        };
        for &terminal in terminals {
            let terminal = TerminalID::from(terminal);
            if (terminal as usize) < self.num_terminals {
                visit(terminal);
            }
        }
        true
    }

    #[inline]
    pub(crate) fn contains(&self, raw_state: u32, terminal: TerminalID) -> bool {
        let terminal = terminal as usize;
        if terminal >= self.num_terminals {
            return false;
        }
        let Some(&mut_node) = self.roots.get(raw_state as usize) else {
            return false;
        };
        let mut node = mut_node;
        let mut level = self.levels as usize;
        let word = terminal / 64;
        while level != 0 {
            let DirectRegularSupportNode::Branch(left, right) = self.nodes[node as usize] else {
                return false;
            };
            node = if ((word >> (level - 1)) & 1) == 0 {
                left
            } else {
                right
            };
            level -= 1;
        }
        let DirectRegularSupportNode::Leaf(value) = self.nodes[node as usize] else {
            return false;
        };
        value & (1u64 << (terminal % 64)) != 0
    }

    fn or_node(&self, node: u32, level: usize, word_base: usize, output: &mut [u64]) {
        if node == self.zero[level] {
            return;
        }
        if level == 0 {
            let DirectRegularSupportNode::Leaf(value) = self.nodes[node as usize] else {
                return;
            };
            if let Some(word) = output.get_mut(word_base) {
                *word |= value;
            }
            return;
        }
        let DirectRegularSupportNode::Branch(left, right) = self.nodes[node as usize] else {
            return;
        };
        let half = 1usize << (level - 1);
        self.or_node(left, level - 1, word_base, output);
        self.or_node(right, level - 1, word_base + half, output);
    }

    pub(crate) fn or_state_into(&self, raw_state: u32, output: &mut [u64]) {
        if let Some(words) = self.dense_state_rows.get(&raw_state) {
            for (target, source) in output.iter_mut().zip(words.iter()) {
                *target |= *source;
            }
            return;
        }
        if let Some(&root) = self.roots.get(raw_state as usize) {
            self.or_node(root, self.levels as usize, 0, output);
        }
    }

    #[inline]
    pub(crate) fn root_id(&self, raw_state: u32) -> Option<u32> {
        self.roots.get(raw_state as usize).copied()
    }

    #[inline]
    pub(crate) fn state_terminal_count(&self, raw_state: u32) -> Option<u16> {
        let root = *self.roots.get(raw_state as usize)?;
        self.node_counts.get(root as usize).copied()
    }

    pub(crate) fn singleton_terminal(&self, raw_state: u32) -> Option<TerminalID> {
        let root = self.root_id(raw_state)?;
        let terminals = self.node_small_support[root as usize].terminals()?;
        let [terminal] = terminals else {
            return None;
        };
        Some(TerminalID::from(*terminal))
    }

    fn intersects_node(
        &self,
        node: u32,
        level: usize,
        word_base: usize,
        terminals: &[u64],
    ) -> bool {
        if node == self.zero[level] {
            return false;
        }
        if level == 0 {
            let DirectRegularSupportNode::Leaf(value) = self.nodes[node as usize] else {
                return false;
            };
            return terminals
                .get(word_base)
                .is_some_and(|word| (*word & value) != 0);
        }
        let DirectRegularSupportNode::Branch(left, right) = self.nodes[node as usize] else {
            return false;
        };
        let half = 1usize << (level - 1);
        self.intersects_node(left, level - 1, word_base, terminals)
            || self.intersects_node(right, level - 1, word_base + half, terminals)
    }

    pub(crate) fn intersects(&self, raw_state: u32, terminals: &[u64]) -> bool {
        if let Some(words) = self.dense_state_rows.get(&raw_state) {
            return words
                .iter()
                .zip(terminals)
                .any(|(left, right)| (*left & *right) != 0);
        }
        self.roots.get(raw_state as usize).is_some_and(|&root| {
            self.intersects_node(root, self.levels as usize, 0, terminals)
        })
    }
}

#[derive(Debug, Clone)]
pub(crate) struct DirectRegularDynamicFrontierCacheEntry {
    /// Retain the source interface so its pointer-derived key cannot be reused
    /// while this cache entry exists.
    pub(crate) source: ParserGSS,
    pub(crate) actionable_terminals: crate::ds::bitset::BitSet,
    pub(crate) advance_by_terminal: Arc<[(TerminalID, Arc<[u32]>)]>,
}

/// Canonical semantic snapshot of a dynamic-mask residual. Flattening the GSS
/// deliberately removes representation-only Arc identities and accumulator
/// node organization, so equivalent residuals reached after different token
/// commits share one exact cached mask.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum DynamicMaskLexerStateKey {
    Exact(u32),
    /// Exact scoped tokenizer coordinate used by recursive/provider-native
    /// composition. Keep it distinct from the ordinary tokenizer coordinate:
    /// both are u32 IDs but belong to different state spaces.
    RecursiveExact(u32),
    /// Exact lexer coordinate consumed by dynamic mask execution. Distinct
    /// source/runtime lexer states may map here when their complete one-model-
    /// token continuation languages are identical. Preserve whether the source
    /// state is the true lexer initial state because reset semantics can make
    /// that distinction observable outside the projected coordinate itself.
    MaskProjection { state: u32, initial: bool },
    TerminalObservation { terminal: TerminalID, class: u32, initial: bool },
    /// Sound pre-minimization coordinate of a virtual bounded-code residual's
    /// finite one-model-token projection. This gives the dynamic cache the
    /// useful count collapse of the finite projection without constructing its
    /// potentially large DFA.
    VirtualDenseProjection { runtime: u32, state: u32, initial: bool },
}

/// Paths are top-first and sorted lexicographically with their correlated
/// exclusions. The orientation is private to this in-memory cache; reversing
/// every path is a bijection, so it changes neither key equivalence nor masks.
pub(crate) type DynamicMaskStateKey = Vec<(
    DynamicMaskLexerStateKey,
    Vec<(Vec<u32>, Vec<(u32, Vec<TerminalID>)>)>,
)>;

#[derive(Debug, Clone)]
pub(crate) struct DynamicConfigSubtreeCertificate {
    pub(crate) node: u32,
    /// Lexer NFA configuration in the mask-tokenizer quotient coordinate.
    pub(crate) projected_config: Arc<[u32]>,
    /// Every vocabulary token below `node`, when entered in
    /// `projected_config`, reaches token boundary without another lexer
    /// finalization while retaining at least one of these terminals as a
    /// possible future.  Runtime needs only one terminal to be parser-admissible.
    pub(crate) common_future_terminals: Arc<[TerminalID]>,
}

/// Exact vocabulary-relative continuation row after one lexer terminal has
/// finalized inside a model token and the lexer has reset.  Every token in
/// `tokens` reaches token boundary without a second lexer finalization and is
/// live for at least one terminal in `terminals`.  `terminals` are grouped by
/// exact equality of their fused-token set, so runtime normally tests only a
/// handful of rows even when many grammar terminals share the same lexical
/// continuation language.
#[derive(Debug, Clone)]
pub(crate) struct DynamicFirstMatchPostRow {
    pub(crate) terminals: Arc<[TerminalID]>,
    pub(crate) tokens: Arc<[u32]>,
    /// Prepacked token mask for broad rows.  Sparse rows leave this empty and
    /// are cheaper to apply by setting their handful of token IDs directly.
    pub(crate) dense_mask: Arc<[u32]>,
}

/// Second-finalization continuation from a first-match one-step projection.
/// `terminal` is consumed on the post-first parser stack.  Exact-end tokens
/// become immediately valid after that parser advance; `post_rows` describe
/// residual lexer futures after the second reset for branches that reach token
/// boundary without a third finalization.
#[derive(Debug, Clone)]
pub(crate) struct DynamicFirstMatchSecondRow {
    pub(crate) terminal: TerminalID,
    pub(crate) exact_end_tokens: Arc<[u32]>,
    pub(crate) post_rows: Arc<[DynamicFirstMatchPostRow]>,
    /// Additional terminal finalizations after this terminal resets the lexer.
    /// The row type is recursive so a short vocabulary-relative lexical-effect
    /// program can represent arbitrarily many in-token finalizations without
    /// returning to byte-wise trie traversal.
    pub(crate) next_rows: Arc<[DynamicFirstMatchSecondRow]>,
}

#[derive(Debug, Clone)]
pub(crate) struct DynamicSelfLoopProjection {
    pub(crate) source_state: u32,
    /// Exact possible-future terminal set at `source_state`. Projection token
    /// leaves are certified only when they restore this complete set, making
    /// the projection independent of parser context. Runtime needs only one of
    /// these terminals to be parser-admissible for the continuing witness.
    pub(crate) future_terminals: Arc<[TerminalID]>,
    pub(crate) safe_no_match_mask: Arc<[u32]>,
    pub(crate) safe_subtrees: Arc<[u8]>,
    /// Nodes whose complete suffix language is safe when entered with
    /// `source_state` itself. `safe_subtrees` is relative to the state reached
    /// by consuming the node's root prefix during projection construction; an
    /// intermediate runtime walk may only reuse a projection at nodes where
    /// that reached state has returned to the projection source.
    pub(crate) source_reentry_safe_subtrees: Arc<[u8]>,
    /// For a projection rooted at `source_state`, row `node` is a bitmask over
    /// `future_terminals`: bit i is set iff that terminal remains a live
    /// no-finalization continuation for every vocabulary token below `node`.
    /// This is stronger than the historical exact-future-set projection for
    /// accepting+continuing lexer states: unrelated finalizers/futures may
    /// churn as long as one common continuing terminal witnesses the subtree.
    pub(crate) common_future_masks: Arc<[u64]>,
    /// Sparse trie nodes that are provably useless while following the
    /// no-finalization path from `source_state`: every token below the node
    /// dies before any lexer terminal can match and no token can end with a
    /// live residual lexer state.  This certificate is parser-independent.
    pub(crate) pre_match_dead_words: Arc<[u64]>,
    /// Sparse trie nodes whose incoming radix edge reaches the first lexer
    /// terminal match from `source_state`.  The dead-node certificate above is
    /// no longer applicable below these nodes because parser-dependent reset
    /// branches become possible there.
    pub(crate) pre_match_frontier_words: Arc<[u64]>,
    /// Experimental exact subset for tokens that first finalize the sole
    /// future terminal from one concrete full tokenizer state and whose
    /// post-reset byte suffix is itself an ordinary vocabulary token.
    ///
    /// Runtime validates only the suffix-token candidates after advancing the
    /// parser once on `future_terminals[0]`; an accepted suffix then certifies
    /// the corresponding fused original token.  This is deliberately a
    /// one-sided baseline: tokens not represented here still go through the
    /// ordinary exact dynamic walk.
    pub(crate) first_match_fusion_source_state: u32,
    pub(crate) first_match_fusion_match_state: u32,
    pub(crate) first_match_fusion_candidate_mask: Arc<[u32]>,
    /// One bit per dynamic vocabulary-trie node: set iff the subtree contains
    /// at least one suffix token from `first_match_fusion_candidate_mask`.
    pub(crate) first_match_fusion_candidate_subtrees: Arc<[u64]>,
    /// `(fused_original_token, suffix_original_token)` pairs.
    pub(crate) first_match_fusions: Arc<[(u32, u32)]>,
    /// Experimental exact one-finalization decomposition for a concrete full
    /// tokenizer state.  It is deliberately vocabulary-relative: tokens with
    /// more than one possible first-match width or any second finalization
    /// after reset are listed in `first_match_step_unknown_tokens` and are
    /// validated by the ordinary exact dynamic walker.
    pub(crate) first_match_step_source_state: u32,
    pub(crate) first_match_step_root_live_tokens: Arc<[u32]>,
    pub(crate) first_match_step_exact_end_tokens: Arc<[u32]>,
    pub(crate) first_match_step_post_rows: Arc<[DynamicFirstMatchPostRow]>,
    pub(crate) first_match_step_second_rows: Arc<[DynamicFirstMatchSecondRow]>,
    pub(crate) first_match_step_unknown_tokens: Arc<[u32]>,
    /// One bit per runtime vocabulary-trie node, set iff that subtree contains
    /// at least one token from `first_match_step_unknown_tokens`.
    pub(crate) first_match_step_unknown_subtrees: Arc<[u64]>,
    /// General vocabulary-relative lexical-effect program rooted directly at
    /// a concrete tokenizer state. Unlike `first_match_step_*`, this does not
    /// require a sole first terminal: residual no-finalization futures live in
    /// `root_effect_post_rows`, while `root_effect_rows` encode arbitrary
    /// terminal-finalization/reset sequences. Runtime executes only the parser
    /// effects; unresolved depth-limited tokens fall back to the exact walker.
    pub(crate) root_effect_source_state: u32,
    pub(crate) root_effect_post_rows: Arc<[DynamicFirstMatchPostRow]>,
    pub(crate) root_effect_rows: Arc<[DynamicFirstMatchSecondRow]>,
    pub(crate) root_effect_unknown_tokens: Arc<[u32]>,
    pub(crate) root_effect_unknown_subtrees: Arc<[u64]>,
    /// Sparse post-finalization certificates discovered from repeated reset-NFA
    /// configurations below this projection's first-match frontier.
    pub(crate) config_subtree_certificates: Arc<[DynamicConfigSubtreeCertificate]>,
}

impl DynamicSelfLoopProjection {
    #[inline]
    pub(crate) fn subtree_is_safe(&self, node: u32) -> bool {
        self.safe_subtrees
            .get(node as usize)
            .is_some_and(|&safe| safe != 0)
    }

    #[inline]
    pub(crate) fn subtree_is_safe_from_source(&self, node: u32) -> bool {
        self.source_reentry_safe_subtrees
            .get(node as usize)
            .is_some_and(|&safe| safe != 0)
    }

    #[inline]
    pub(crate) fn subtree_common_future_mask(&self, node: u32) -> u64 {
        self.common_future_masks
            .get(node as usize)
            .copied()
            .unwrap_or(0)
    }

    #[inline]
    pub(crate) fn pre_match_subtree_is_dead(&self, node: u32) -> bool {
        let word = node as usize >> 6;
        let bit = node & 63;
        self.pre_match_dead_words
            .get(word)
            .is_some_and(|bits| bits & (1u64 << bit) != 0)
    }

    #[inline]
    pub(crate) fn pre_match_subtree_is_frontier(&self, node: u32) -> bool {
        let word = node as usize >> 6;
        let bit = node & 63;
        self.pre_match_frontier_words
            .get(word)
            .is_some_and(|bits| bits & (1u64 << bit) != 0)
    }

    #[inline]
    pub(crate) fn has_pre_match_dead_subtrees(&self) -> bool {
        self.pre_match_dead_words.iter().any(|&word| word != 0)
    }

    #[inline]
    pub(crate) fn has_first_match_fusions_from(&self, full_source_state: u32) -> bool {
        self.first_match_fusion_source_state == full_source_state
            && self.first_match_fusion_match_state != u32::MAX
            && !self.first_match_fusions.is_empty()
    }

    #[inline]
    pub(crate) fn has_root_effect_from(&self, full_source_state: u32) -> bool {
        self.root_effect_source_state == full_source_state
    }

    #[inline]
    pub(crate) fn has_first_match_step_from(&self, full_source_state: u32) -> bool {
        self.first_match_step_source_state == full_source_state
            && self.future_terminals.len() == 1
            && (!self.first_match_step_root_live_tokens.is_empty()
                || !self.first_match_step_exact_end_tokens.is_empty()
                || !self.first_match_step_post_rows.is_empty()
                || !self.first_match_step_unknown_tokens.is_empty())
    }

    #[inline]
    pub(crate) fn config_subtree_certificates_for_node(
        &self,
        node: u32,
    ) -> &[DynamicConfigSubtreeCertificate] {
        let certificates = self.config_subtree_certificates.as_ref();
        let start = certificates.partition_point(|certificate| certificate.node < node);
        let end = start
            + certificates[start..]
                .partition_point(|certificate| certificate.node == node);
        &certificates[start..end]
    }
}

#[derive(Debug, Clone)]
pub(crate) struct DynamicMaskVocabSource {
    pub(crate) trie: Arc<VocabPrefixTree>,
    pub(crate) token_aliases: Arc<Vec<Vec<u32>>>,
}

/// Compact parser-independent master-slice proof sidecar. All arrays are CSR
/// over `(exact tokenizer source, proof slot)` rows. `positive_*` stores true
/// proofs; `coverage_*` stores every terminal for which the build solved the
/// exact residual product, so covered-but-not-positive is an exact false.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct PreparedMasterProofArtifact {
    pub(crate) positive_row_ids: Vec<u32>,
    pub(crate) positive_offsets: Vec<u32>,
    pub(crate) positive_terminals: Vec<TerminalID>,
    pub(crate) coverage_row_ids: Vec<u32>,
    pub(crate) coverage_offsets: Vec<u32>,
    pub(crate) coverage_terminals: Vec<TerminalID>,
    /// Safe+ terminals for which the direct residual solver is complete over
    /// every exact source. For these terminals, absence from a source coverage
    /// row means that terminal has no residual coordinate at that source and is
    /// therefore an exact negative rather than an unknown requiring a quotient.
    pub(crate) safe_plus_complete_terminals: Vec<TerminalID>,
    pub(crate) safe_radius_row_ids: Vec<u32>,
    pub(crate) safe_radius_offsets: Vec<u32>,
    pub(crate) safe_radius_entries: Vec<(TerminalID, u16)>,
}

impl PreparedMasterProofArtifact {
    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.positive_row_ids.is_empty()
            && self.coverage_row_ids.is_empty()
            && self.safe_plus_complete_terminals.is_empty()
            && self.safe_radius_row_ids.is_empty()
    }
}

#[inline]
fn interned_terminal_row<'a>(
    row_ids: &[u32],
    offsets: &[u32],
    entries: &'a [TerminalID],
    row: usize,
) -> &'a [TerminalID] {
    let row_id = row_ids[row] as usize;
    &entries[offsets[row_id] as usize..offsets[row_id + 1] as usize]
}

fn intern_terminal_rows(
    mut rows: Vec<SmallVec<[TerminalID; 4]>>,
) -> (Vec<u32>, Vec<u32>, Vec<TerminalID>) {
    let mut intern = FxHashMap::<Vec<TerminalID>, u32>::default();
    let mut unique_rows = Vec::<Vec<TerminalID>>::new();
    let mut row_ids = Vec::<u32>::with_capacity(rows.len());
    for row in &mut rows {
        row.sort_unstable();
        row.dedup();
        let slice = row.as_slice();
        let id = if let Some(&id) = intern.get(slice) {
            id
        } else {
            let owned = slice.to_vec();
            let id = unique_rows.len() as u32;
            intern.insert(owned.clone(), id);
            unique_rows.push(owned);
            id
        };
        row_ids.push(id);
    }
    let mut offsets = Vec::<u32>::with_capacity(unique_rows.len() + 1);
    let mut entries = Vec::<TerminalID>::new();
    offsets.push(0);
    for row in unique_rows {
        entries.extend(row);
        offsets.push(entries.len() as u32);
    }
    (row_ids, offsets, entries)
}

fn intern_radius_rows(
    mut rows: Vec<SmallVec<[(TerminalID, u16); 4]>>,
) -> (Vec<u32>, Vec<u32>, Vec<(TerminalID, u16)>) {
    let mut intern = FxHashMap::<Vec<(TerminalID, u16)>, u32>::default();
    let mut unique_rows = Vec::<Vec<(TerminalID, u16)>>::new();
    let mut row_ids = Vec::<u32>::with_capacity(rows.len());
    for row in &mut rows {
        row.sort_unstable_by_key(|&(terminal, _)| terminal);
        row.dedup_by_key(|entry| entry.0);
        let slice = row.as_slice();
        let id = if let Some(&id) = intern.get(slice) {
            id
        } else {
            let owned = slice.to_vec();
            let id = unique_rows.len() as u32;
            intern.insert(owned.clone(), id);
            unique_rows.push(owned);
            id
        };
        row_ids.push(id);
    }
    let mut offsets = Vec::<u32>::with_capacity(unique_rows.len() + 1);
    let mut entries = Vec::<(TerminalID, u16)>::new();
    offsets.push(0);
    for row in unique_rows {
        entries.extend(row);
        offsets.push(entries.len() as u32);
    }
    (row_ids, offsets, entries)
}

#[derive(Debug, Clone, Default)]
pub(crate) struct DynamicBoundedObservationSets {
    pool: Arc<[U8Set]>,
    horizon16: Arc<[u32]>,
    horizon64: Arc<[u32]>,
}

impl DynamicBoundedObservationSets {
    pub(crate) fn from_raw(horizon16: Box<[U8Set]>, horizon64: Box<[U8Set]>) -> Self {
        debug_assert_eq!(horizon16.len(), horizon64.len());
        let mut ids = FxHashMap::<U8Set, u32>::default();
        let mut pool = Vec::<U8Set>::new();
        let mut intern = |set: U8Set| -> u32 {
            if let Some(&id) = ids.get(&set) {
                return id;
            }
            let id = pool.len() as u32;
            pool.push(set);
            ids.insert(set, id);
            id
        };
        let horizon16 = horizon16
            .iter()
            .copied()
            .map(&mut intern)
            .collect::<Vec<_>>();
        let horizon64 = horizon64
            .iter()
            .copied()
            .map(&mut intern)
            .collect::<Vec<_>>();
        Self {
            pool: Arc::from(pool),
            horizon16: Arc::from(horizon16),
            horizon64: Arc::from(horizon64),
        }
    }

    #[inline]
    pub(crate) fn safe_bytes(&self, state: u32, required_horizon: u32) -> Option<U8Set> {
        let ids = if required_horizon <= 16 {
            self.horizon16.as_ref()
        } else if required_horizon <= 64 {
            self.horizon64.as_ref()
        } else {
            return None;
        };
        let id = *ids.get(state as usize)? as usize;
        self.pool.get(id).copied()
    }

    #[inline]
    pub(crate) fn state_count(&self) -> usize {
        self.horizon16.len()
    }

    #[inline]
    pub(crate) fn unique_set_count(&self) -> usize {
        self.pool.len()
    }
}


#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct DynamicMaskVocabArtifact {
    trie: DynamicMaskTrie,
    token_aliases: DynamicMaskAliasStore,
    llg_slice_leftovers: Vec<Arc<DynamicMaskSliceTrie>>,
    mask_tokenizer: Option<Tokenizer>,
    full_to_mask_state: Vec<u32>,
    grammar_quotiented: bool,
    source_vocab_digest: Option<[u8; 32]>,
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

/// One overlapping runtime slice language and the residual vocabulary trie
/// to walk after that language has been proved contained. The slice language
/// itself is not a compiler partition and may overlap/nest with other slices.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct DynamicMaskSliceTrie {
    cache_id: u32,
    dfa: Arc<VocabPartitionDfa>,
    trie: Arc<DynamicMaskTrie>,
    full_walk_token_markers: Arc<Vec<u64>>,
    subtree_original_token_offsets: Arc<Vec<u32>>,
    subtree_original_tokens: Arc<Vec<u32>>,
    slice_original_token_words: Arc<Vec<u32>>,
    /// Conservative byte family containing every byte of every current-vocab
    /// token in this slice. Proving this whole family parser-transparent through
    /// `slice_max_token_byte_len` is sufficient to skip every slice token.
    slice_token_bytes: U8Set,
    slice_max_token_byte_len: u32,
    /// First-byte set of the slice DFA start state. Immutable per slice;
    /// cached because the dynamic pre-collapse gate needs it on every mask.
    #[serde(default, skip)]
    first_bytes: OnceLock<U8Set>,
}

impl DynamicMaskSliceTrie {
    #[inline(always)]
    pub(crate) fn cache_id(&self) -> u32 {
        self.cache_id
    }

    #[inline(always)]
    pub(crate) fn dfa(&self) -> &VocabPartitionDfa {
        self.dfa.as_ref()
    }

    /// First-byte set of the slice DFA start state: the exact first-byte
    /// language of this proof slice. Only bytes whose one-byte derivative
    /// can still reach an accepting slice word are relevant here. This is
    /// deliberately computed from the DFA rather than from the bytes that
    /// occur anywhere inside whole slice tokens (UTF-8 continuation bytes,
    /// for example, are not valid first bytes). Immutable per slice;
    /// computed once because the dynamic pre-collapse gate needs it on
    /// every mask.
    pub(crate) fn first_bytes(&self) -> U8Set {
        *self.first_bytes.get_or_init(|| {
            let dfa = self.dfa.as_ref();
            let start = dfa.start_state();
            let mut result = U8Set::empty();
            for byte in 0u16..=255 {
                let byte = byte as u8;
                if dfa.can_reach_accepting(dfa.step(start, byte)) {
                    result.insert(byte);
                }
            }
            result
        })
    }

    #[inline(always)]
    pub(crate) fn trie(&self) -> &DynamicMaskTrie {
        self.trie.as_ref()
    }

    #[inline(always)]
    fn full_walk_token_markers(&self) -> &[u64] {
        self.full_walk_token_markers.as_ref()
    }

    #[inline(always)]
    fn subtree_original_tokens(&self, node: u32) -> &[u32] {
        let canonical_range = self.trie.subtree_token_index_range(node);
        let start = self.subtree_original_token_offsets[canonical_range.start] as usize;
        let end = self.subtree_original_token_offsets[canonical_range.end] as usize;
        &self.subtree_original_tokens[start..end]
    }

    #[inline(always)]
    pub(crate) fn slice_original_token_words(&self) -> &[u32] {
        self.slice_original_token_words.as_ref()
    }

    #[inline(always)]
    pub(crate) fn slice_token_bytes(&self) -> U8Set {
        self.slice_token_bytes
    }

    #[inline(always)]
    pub(crate) fn slice_max_token_byte_len(&self) -> u32 {
        self.slice_max_token_byte_len
    }
}

/// Runtime-only vocabulary data for direct dynamic mask generation.
#[derive(Debug, Clone)]
pub(crate) struct DynamicMaskVocab {
    pub(crate) trie: Arc<DynamicMaskTrie>,
    token_aliases: DynamicMaskAliasStore,
    canonical_original_token_offsets: Arc<Vec<u32>>,
    canonical_original_tokens: Arc<Vec<u32>>,
    canonical_original_word_offsets: Arc<Vec<u32>>,
    canonical_original_word_masks: Arc<Vec<(u32, u32)>>,
    node_token_markers: Arc<Vec<u64>>,
    /// Token markers in the exact order token endpoints are encountered by
    /// `full_walk_ops`. This removes an extra node-id indirection from the
    /// strict walk's very hot endpoint path without changing which endpoints
    /// are visited.
    full_walk_token_markers: Arc<Vec<u64>>,
    subtree_original_token_offsets: Arc<Vec<u32>>,
    subtree_original_tokens: Arc<Vec<u32>>,
    all_original_token_words: Arc<Vec<u32>>,
    llg_slice_leftovers: Arc<Vec<Arc<DynamicMaskSliceTrie>>>,
    /// Cumulative admitted-token bitsets for the dynamic-radius master trie.
    /// Index = `safe_radius * 2 + whitespace_proved`; each row contains exactly
    /// the whole tokens that can be accepted without walking the trie. Shared by
    /// every constraint using the same model vocabulary.
    llg_master_admitted_words: Arc<Vec<Vec<u32>>>,
    /// Exact strict-walk operation count remaining after admitting each master
    /// `(safe_radius, whitespace)` class combination. Indexed identically to
    /// `llg_master_admitted_words`. This is runtime-only derived metadata and
    /// is rebuilt from the master trie after construction/load.
    llg_master_residual_ops: Arc<Vec<u32>>,
    llg_master_max_safe_chars: u16,
    /// Positive-only parser-independent proof rows for the two overlapping
    /// master slice languages. Row = `source_tsid * 2 + slice_slot`, where
    /// slot 0 is safe+ and slot 1 is whitespace. Offsets index the flattened
    /// terminal list. A terminal in a row certifies that the entire slice
    /// language remains inside that terminal's exact projected residual from
    /// the source TSID. Missing rows/terminals simply fall back to runtime
    /// proof; they never imply rejection or admission.
    prepared_master_prover_row_ids: Arc<[u32]>,
    prepared_master_prover_offsets: Arc<[u32]>,
    prepared_master_prover_terminals: Arc<[TerminalID]>,
    /// Exact coverage rows corresponding to `prepared_master_prover_*`.
    /// A terminal present here means the build-time proof solved this
    /// `(source, slice, terminal)` residual completely. Membership in the
    /// positive row is therefore `true`; coverage without positive membership
    /// is an exact `false`. This lets compact direct-residual proofs retain
    /// exact negative answers without carrying a projected-terminal quotient.
    prepared_master_coverage_row_ids: Arc<[u32]>,
    prepared_master_coverage_offsets: Arc<[u32]>,
    prepared_master_coverage_terminals: Arc<[TerminalID]>,
    /// Global safe+ terminal completeness marker for direct residual proofs.
    /// Sorted and unique. See `PreparedMasterProofArtifact`.
    prepared_safe_plus_complete_terminals: Arc<[TerminalID]>,
    /// Exact positive bounded safe-slice rows. Row = exact source TSID; entries
    /// are `(terminal, max safe Unicode-scalar radius)` for every prepared
    /// projected terminal residual. Radius is capped at the largest safe-token
    /// scalar count in this model vocabulary, because larger values are
    /// observationally identical for one-token masking.
    prepared_safe_radius_row_ids: Arc<[u32]>,
    prepared_safe_radius_offsets: Arc<[u32]>,
    prepared_safe_radius_entries: Arc<[(TerminalID, u16)]>,
    pending_source: Option<DynamicMaskVocabSource>,
    initialized: bool,
    /// True only when trie endpoints represent grammar-proven vocabulary
    /// equivalence classes rather than byte-identical token aliases.
    grammar_quotiented: bool,
    /// Strong identity of the complete original model vocabulary from which a
    /// grammar quotient was built. Ordinary dynamic vocabularies leave this
    /// unset; compiler-created O2 quotients carry it into transfer metadata.
    source_vocab_digest: Option<[u8; 32]>,
    mask_cache: Arc<Mutex<DynamicMaskCache>>,
    dense_subset16_cache: Arc<Mutex<FxHashMap<Vec<u32>, Arc<DynamicDenseSubset16>>>>,
    lazy_union_cache: Arc<Mutex<DynamicLazyUnionCache>>,
    direct_regular_frontier_cache:
        Arc<Mutex<FxHashMap<usize, DirectRegularDynamicFrontierCacheEntry>>>,
    direct_regular_wide_frontier_index_cache: Arc<Mutex<FxHashMap<usize, usize>>>,
    direct_regular_terminal_support: Arc<DirectRegularTerminalSupport>,
    self_loop_projections: Arc<Vec<DynamicSelfLoopProjection>>,
    projection_by_source: Arc<[u32]>,
    projection_alias_vocab: Arc<[u32]>,
    projection_alias_h64: Arc<[u32]>,
    bounded_observation_sets: Arc<DynamicBoundedObservationSets>,
    terminal_observation_classes: Arc<[(TerminalID, Arc<[u32]>)]>,
    projected_terminal_quotients: Arc<[(TerminalID, Arc<TerminalProjectedQuotient>)]>,
    /// Runtime-only lazy quotient analysis. Dynamic compilation keeps quotient
    /// construction off the build/first-mask path until a master-slice proof
    /// actually needs it. The persisted/prepared sidecar above still takes
    /// precedence when present.
    runtime_projected_terminal_quotients:
        Arc<OnceLock<Arc<[(TerminalID, Arc<TerminalProjectedQuotient>)]>>>,
    /// Incremental runtime-only quotient cache used by hot-path containment
    /// proofs. Unlike `runtime_projected_terminal_quotients`, this can prepare
    /// only the terminal actually requested by the current proof instead of
    /// materializing every safe-alphabet candidate at once. `None` is a cached
    /// exact "no quotient available" result for that terminal.
    runtime_projected_terminal_quotient_cache:
        Arc<Mutex<FxHashMap<TerminalID, Option<Arc<TerminalProjectedQuotient>>>>>,
    /// True once the exact projected-terminal analysis has run, including
    /// when it proved that no quotient is worth retaining.  This distinguishes
    /// a legitimate empty result from an unprepared legacy/runtime value.
    projected_terminal_quotients_prepared: bool,
    /// Parser-independent exact projected-text proof results.  The key is the
    /// terminal residual coordinate plus the vocabulary alphabet being proved.
    /// Sharing this across sequences is safe because parser admission only
    /// decides whether a proof is queried; the proof result itself depends
    /// solely on immutable lexer/vocabulary data.
    projected_terminal_text_cache: Arc<Mutex<FxHashMap<(TerminalID, u32, U8Set, bool), bool>>>,
    /// Exact regular-language containment results for named proof slices.
    /// High key bits distinguish projected, symbolic-residual, and finite-direct
    /// proof namespaces over the same immutable terminal/source coordinates.
    projected_terminal_partition_cache:
        Arc<Mutex<FxHashMap<(TerminalID, u32, u32), bool>>>,
    /// Exact bounded repetition radii for regular-language proof slices. The
    /// extra key component is the caller's maximum relevant repetition count;
    /// for model-token masking this is the largest safe-token scalar length.
    projected_terminal_radius_cache:
        Arc<Mutex<FxHashMap<(TerminalID, u32, u32, u32), u32>>>,
    /// Conservative bounded certificates over exact physical source coordinates.
    /// Never shared with a different grammar's fresh runtime vocabulary instance.
    raw_terminal_radius_cache:
        Arc<Mutex<FxHashMap<(TerminalID, u32, u32, u32, usize), u32>>>,
    /// Exact original-token masks rejected by a token-start maximal-munch
    /// guard. Keys are the canonical sorted `(mask lexer state, terminal)`
    /// memories carried by `InitialPruneGuard`. The result depends only on the
    /// immutable lexer/vocabulary coordinate, so sequences may safely share it.
    pending_guard_blocked_mask_cache:
        Arc<Mutex<FxHashMap<Vec<(u32, TerminalID)>, Arc<Vec<u32>>>>>,
    /// Optional mask-only finite-token quotient. Commit continues to use the
    /// exact tokenizer stored on `Constraint`; dynamic mask projections may be
    /// built in this smaller coordinate and indexed from exact runtime states
    /// through `full_to_mask_state`.
    mask_tokenizer: Option<Arc<Tokenizer>>,
    /// Optional deterministic derivative of `mask_tokenizer` used only by
    /// mask generation. This is particularly useful for finite one-token
    /// projections of lazy/virtual lexers whose compact serialized projection
    /// still contains epsilon fan-in. Commit never uses this coordinate.
    mask_determinized_tokenizer: Option<Arc<Tokenizer>>,
    /// Dense map from the serialized/base mask-tokenizer state coordinate to
    /// `mask_determinized_tokenizer`. Empty when no second-stage
    /// determinization is active.
    mask_projection_to_determinized: Arc<[u32]>,
    mask_tokenizer_fast_transitions: Option<FastTokenizerTransitions>,
    full_to_mask_state: Arc<[u32]>,
    /// Derived exact subset provenance for the dense mask tokenizer. Keys are
    /// epsilon-closed source-tokenizer state sets and values are the already
    /// materialized deterministic mask states representing those sets.
    /// Runtime mask roots may use this only when every source state in the set
    /// carries the same parser object by identity.
    mask_state_source_subsets: Arc<[Arc<[u32]>]>,
    mask_source_subset_to_state: Arc<FxHashMap<Arc<[u32]>, u32>>,
    virtual_unit_repeat_projection: Option<VirtualZeroMinUnitRepeatMaskProjection>,
    virtual_repeat_intersection_projections: Vec<VirtualBinaryRepeatIntersectionMaskProjection>,
    virtual_residual_projections: Vec<VirtualResidualMaskProjection>,
}

impl DynamicMaskVocab {
    const FULL_WALK_DENSE_TRANSITION_BYTES: usize = 64 * 1024 * 1024;
    pub(crate) const PREPARED_PROOF_SLOT_COUNT: usize = 2;
    const PREPARED_SAFE_PLUS_SLOT: usize = 0;
    const PREPARED_WHITESPACE_SLOT: usize = 1;

    #[inline]
    pub(crate) fn max_token_byte_len(&self) -> usize {
        self.trie
            .nodes
            .first()
            .map_or(0, |root| root.subtree_max_byte_len as usize)
    }

    fn build_full_walk_fast_transitions(tokenizer: &Tokenizer) -> Option<FastTokenizerTransitions> {
        // Scalar-dispatch projections deliberately execute through the lazy
        // physical-row/subset cache. A complete Flat16 slab is pure derived
        // acceleration and is too expensive to materialize before the first
        // mask; the lazy executor builds only rows actually reached by vocab
        // traversal. Ordinary deterministic tokenizers still use the dense
        // table below.
        if tokenizer.has_any_virtual_runtime() || tokenizer.has_epsilon_transitions() {
            return None;
        }
        FastTokenizerTransitions::full_walk_dense_for(
            tokenizer,
            Self::FULL_WALK_DENSE_TRANSITION_BYTES,
        )
    }

    /// Prepare the dense transition cells used exclusively by the strict full
    /// vocabulary walker. The mask projection, when present, is the exact
    /// coordinate masking advances; otherwise use the source runtime tokenizer.
    /// These tables are derived runtime data and are deliberately not serialized.
    pub(crate) fn prepare_full_walk_fast_transitions(&mut self, source: &Tokenizer) {
        // Every operation that changes the active mask tokenizer either builds
        // this derived table for the new coordinate or explicitly clears it.
        // Dynamic finalization calls this preparation step again after choosing
        // the final coordinate, so rebuilding an already-present table here is
        // duplicate work (and is material for large finite residual projections).
        if self.mask_tokenizer_fast_transitions.is_some() {
            return;
        }
        let tokenizer = self
            .mask_determinized_tokenizer
            .as_deref()
            .or(self.mask_tokenizer.as_deref())
            .unwrap_or(source);
        self.mask_tokenizer_fast_transitions = Self::build_full_walk_fast_transitions(tokenizer);
    }

    /// Prepare a deterministic execution representation for the exact finite
    /// lexer coordinate used by mask generation. The source constraint lexer
    /// remains unchanged; this is analogous to choosing Flat16 versus Flat32
    /// for the strict mask walk. If the derived representation does not fit,
    /// the strict walker executes the epsilon-NFA coordinate directly.
    pub(crate) fn prepare_mask_execution(
        &mut self,
        source_tokenizer: &Tokenizer,
        horizon: usize,
    ) -> bool {
        // Bound eager work by the same memory budget that decides whether the
        // result can use Flat32. This avoids introducing an independent
        // state-count determinization policy on top of lexer compilation.
        const CELL_BYTES: usize = std::mem::size_of::<u32>();
        const ALPHABET: usize = 256;
        let state_limit = Self::FULL_WALK_DENSE_TRANSITION_BYTES / (CELL_BYTES * ALPHABET);
        let transition_limit = state_limit.saturating_mul(ALPHABET);
        let source = self.mask_tokenizer.as_deref().unwrap_or(source_tokenizer);
        if source.has_any_virtual_runtime() {
            return false;
        }
        if !source.has_epsilon_transitions() {
            self.mask_determinized_tokenizer = None;
            self.mask_projection_to_determinized = Arc::from(Vec::<u32>::new());
            self.mask_tokenizer_fast_transitions = Self::build_full_walk_fast_transitions(source);
            return self.mask_tokenizer_fast_transitions.is_some();
        }
        let Some((built, source_to_determinized)) = source
            .try_reusing_horizon_determinization_all_starts(
                horizon,
                state_limit,
                transition_limit,
            )
            .or_else(|| {
                source.try_horizon_determinization_all_starts(
                    horizon,
                    state_limit,
                    transition_limit,
                )
            })
        else {
            return false;
        };
        let source_subsets = built.source_subsets;
        let deterministic = built.tokenizer;
        let Some(fast) = Self::build_full_walk_fast_transitions(&deterministic) else {
            return false;
        };
        if self.mask_tokenizer.is_none() {
            // Directly-derived source execution coordinate. Represent it as
            // the ordinary source->mask quotient so all existing runtime
            // metadata sees the same coordinate, and retain exact subset
            // provenance for root/branch unions.
            self.set_mask_tokenizer_quotient(deterministic, source_to_determinized);
            self.set_mask_tokenizer_source_subsets(source_subsets);
            return true;
        }
        self.mask_determinized_tokenizer = Some(Arc::new(deterministic));
        self.mask_projection_to_determinized = Arc::from(source_to_determinized);
        self.mask_tokenizer_fast_transitions = Some(fast);
        // The derivative determinizer subsets are expressed in the finite
        // projection coordinate. Retain them so multiple exact lexer states
        // carrying one parser object can be coalesced into an already-built
        // execution subset at the beginning of a mask walk.
        self.set_mask_tokenizer_source_subsets(source_subsets);
        true
    }

    pub(crate) fn from_compiler_artifacts(
        trie: Arc<VocabPrefixTree>,
        token_aliases: Arc<Vec<Vec<u32>>>,
    ) -> Self {
        Self::from_source(DynamicMaskVocabSource { trie, token_aliases })
    }

    pub(crate) fn from_compiler_artifacts_materialized(
        trie: Arc<VocabPrefixTree>,
        token_aliases: Arc<Vec<Vec<u32>>>,
    ) -> Self {
        let mut vocab = Self::from_compiler_artifacts(trie, token_aliases);
        let materialized = vocab.materialize_pending_source();
        debug_assert!(materialized);
        vocab
    }

    pub(crate) fn from_materialized_ordered(
        trie: Arc<DynamicMaskTrie>,
        token_aliases: Arc<Vec<Vec<u32>>>,
    ) -> Self {
        Self::from_materialized_ordered_with_all_original_token_words(
            trie,
            token_aliases,
            None,
        )
    }

    pub(crate) fn from_materialized_ordered_with_all_original_token_words(
        trie: Arc<DynamicMaskTrie>,
        token_aliases: Arc<Vec<Vec<u32>>>,
        prepared_all_original_token_words: Option<Arc<Vec<u32>>>,
    ) -> Self {
        // Grammar quotients can contain one overwhelmingly large alias group
        // (for example the permanently-dead class of a singleton language).
        // When this Arc is uniquely owned and its first group already reserved
        // enough capacity for the complete flattened list, reuse that Vec as
        // the canonical runtime storage instead of copying ~the whole model
        // vocabulary into a second allocation. Small/shared callers retain the
        // established grouped representation.
        let (token_aliases, preflattened) = match Arc::try_unwrap(token_aliases) {
            Ok(groups) => {
                let total = groups.iter().map(Vec::len).sum::<usize>();
                let can_reuse_first = groups
                    .first()
                    .is_some_and(|first| first.len() >= 4_096 && first.capacity() >= total);
                if can_reuse_first {
                    let mut offsets = Vec::with_capacity(groups.len() + 1);
                    offsets.push(0);
                    for group in &groups {
                        offsets.push(offsets.last().copied().unwrap_or(0) + group.len() as u32);
                    }
                    let mut iter = groups.into_iter();
                    let mut originals = iter.next().unwrap_or_default();
                    for group in iter {
                        originals.extend_from_slice(&group);
                    }
                    let offsets = Arc::new(offsets);
                    let originals = Arc::new(originals);
                    (
                        DynamicMaskAliasStore::Flat {
                            offsets: Arc::clone(&offsets),
                            originals: Arc::clone(&originals),
                        },
                        Some((offsets, originals)),
                    )
                } else {
                    (DynamicMaskAliasStore::Ordered(Arc::new(groups)), None)
                }
            }
            Err(token_aliases) => (DynamicMaskAliasStore::Ordered(token_aliases), None),
        };
        let (canonical_original_token_offsets, canonical_original_tokens) =
            preflattened.unwrap_or_else(|| Self::flatten_canonical_original_tokens(&token_aliases));
        Self::from_ordered_alias_parts(
            trie, token_aliases, canonical_original_token_offsets,
            canonical_original_tokens, prepared_all_original_token_words,
        )
    }

    /// Build the same owned runtime vocabulary from preflattened, sorted aliases.
    /// Each offsets interval corresponds to one canonical trie token.
    pub(crate) fn from_materialized_flat_ordered(
        trie: Arc<DynamicMaskTrie>,
        offsets: Arc<Vec<u32>>,
        originals: Arc<Vec<u32>>,
    ) -> Self {
        assert_eq!(offsets.first(), Some(&0));
        assert_eq!(offsets.last().copied().map(|n| n as usize), Some(originals.len()));
        assert!(offsets.windows(2).all(|p| p[0] <= p[1]));
        debug_assert!(offsets.windows(2).all(|p| originals[p[0] as usize..p[1] as usize]
            .windows(2).all(|q| q[0] <= q[1])));
        let token_aliases = DynamicMaskAliasStore::Flat {
            offsets: Arc::clone(&offsets), originals: Arc::clone(&originals),
        };
        Self::from_ordered_alias_parts(trie, token_aliases, offsets, originals, None)
    }

    fn from_ordered_alias_parts(
        trie: Arc<DynamicMaskTrie>,
        token_aliases: DynamicMaskAliasStore,
        canonical_original_token_offsets: Arc<Vec<u32>>,
        canonical_original_tokens: Arc<Vec<u32>>,
        prepared_all_original_token_words: Option<Arc<Vec<u32>>>,
    ) -> Self {
        let build_words = || {
            Self::build_canonical_original_word_masks_sorted(
                &canonical_original_token_offsets,
                &canonical_original_tokens,
            )
        };
        let build_markers = || {
            let node_token_markers = Self::build_node_token_markers(
                trie.as_ref(),
                &canonical_original_token_offsets,
                &canonical_original_tokens,
            );
            let full_walk_token_markers =
                Self::build_full_walk_token_markers(trie.as_ref(), &node_token_markers);
            (node_token_markers, full_walk_token_markers)
        };
        let build_subtree = || {
            // `from_materialized_ordered` receives canonical tokens in lexical
            // trie order. `DynamicMaskTrie::all_subtree_tokens()` therefore
            // enumerates canonical IDs as 0..N, so the canonical flattened
            // aliases are already exactly the subtree-order flattened aliases.
            // Reuse the same immutable buffers instead of copying every model
            // token ID a second time. Fall back defensively if a future caller
            // violates the ordered-coordinate contract.
            let subtree_identity = trie
                .all_subtree_tokens()
                .iter()
                .copied()
                .enumerate()
                .all(|(index, canonical)| index as u32 == canonical);
            if subtree_identity {
                (
                    Arc::clone(&canonical_original_token_offsets),
                    Arc::clone(&canonical_original_tokens),
                )
            } else {
                Self::flatten_subtree_original_tokens(
                    trie.as_ref(),
                    &canonical_original_token_offsets,
                    &canonical_original_tokens,
                )
            }
        };
        let build_all_words = || {
            prepared_all_original_token_words
                .clone()
                .unwrap_or_else(|| Self::build_all_original_token_words(&canonical_original_tokens))
        };

        let (
            (canonical_original_word_offsets, canonical_original_word_masks),
            (node_token_markers, full_walk_token_markers),
            (subtree_original_token_offsets, subtree_original_tokens),
            all_original_token_words,
        ) = if canonical_original_tokens.len() >= 4_096 && rayon::current_num_threads() > 1 {
            let ((word_masks, markers), (subtree, all_words)) = rayon::join(
                || rayon::join(build_words, build_markers),
                || rayon::join(build_subtree, build_all_words),
            );
            (word_masks, markers, subtree, all_words)
        } else {
            (
                build_words(),
                build_markers(),
                build_subtree(),
                build_all_words(),
            )
        };
        Self {
            trie,
            token_aliases,
            canonical_original_token_offsets,
            canonical_original_tokens,
            canonical_original_word_offsets,
            canonical_original_word_masks,
            node_token_markers,
            full_walk_token_markers,
            subtree_original_token_offsets,
            subtree_original_tokens,
            all_original_token_words,
            llg_slice_leftovers: Arc::new(Vec::new()),
            llg_master_admitted_words: Arc::new(Vec::new()),
            llg_master_residual_ops: Arc::new(Vec::new()),
            llg_master_max_safe_chars: 0,
            prepared_master_prover_row_ids: Arc::from(Vec::<u32>::new()),
            prepared_master_prover_offsets: Arc::from(Vec::<u32>::new()),
            prepared_master_prover_terminals: Arc::from(Vec::<TerminalID>::new()),
            prepared_master_coverage_row_ids: Arc::from(Vec::<u32>::new()),
            prepared_master_coverage_offsets: Arc::from(Vec::<u32>::new()),
            prepared_master_coverage_terminals: Arc::from(Vec::<TerminalID>::new()),
            prepared_safe_plus_complete_terminals: Arc::from(Vec::<TerminalID>::new()),
            prepared_safe_radius_row_ids: Arc::from(Vec::<u32>::new()),
            prepared_safe_radius_offsets: Arc::from(Vec::<u32>::new()),
            prepared_safe_radius_entries: Arc::from(Vec::<(TerminalID, u16)>::new()),
            pending_source: None,
            initialized: true,
            grammar_quotiented: false,
            source_vocab_digest: None,
            mask_cache: Arc::new(Mutex::new(DynamicMaskCache::default())),
            dense_subset16_cache: Arc::new(Mutex::new(FxHashMap::default())),
            lazy_union_cache: Arc::new(Mutex::new(DynamicLazyUnionCache::default())),
            direct_regular_frontier_cache: Arc::new(Mutex::new(FxHashMap::default())),
            direct_regular_wide_frontier_index_cache: Arc::new(Mutex::new(FxHashMap::default())),
            direct_regular_terminal_support: Arc::new(DirectRegularTerminalSupport::default()),
            self_loop_projections: Arc::new(Vec::new()),
            projection_by_source: Arc::from(Vec::<u32>::new()),
            projection_alias_vocab: Arc::from(Vec::<u32>::new()),
            projection_alias_h64: Arc::from(Vec::<u32>::new()),
            bounded_observation_sets: Arc::new(DynamicBoundedObservationSets::default()),
            terminal_observation_classes: Arc::from(Vec::<(TerminalID, Arc<[u32]>)>::new()),
            projected_terminal_quotients: Arc::from(Vec::<(TerminalID, Arc<TerminalProjectedQuotient>)>::new()),
            runtime_projected_terminal_quotients: Arc::new(OnceLock::new()),
            runtime_projected_terminal_quotient_cache: Arc::new(Mutex::new(FxHashMap::default())),
            projected_terminal_quotients_prepared: false,
            projected_terminal_text_cache: Arc::new(Mutex::new(FxHashMap::default())),
            projected_terminal_partition_cache: Arc::new(Mutex::new(FxHashMap::default())),
            projected_terminal_radius_cache: Arc::new(Mutex::new(FxHashMap::default())),
            raw_terminal_radius_cache: Arc::new(Mutex::new(FxHashMap::default())),
            pending_guard_blocked_mask_cache: Arc::new(Mutex::new(FxHashMap::default())),
            mask_tokenizer: None,
            mask_determinized_tokenizer: None,
            mask_projection_to_determinized: Arc::from(Vec::<u32>::new()),
            mask_tokenizer_fast_transitions: None,
            full_to_mask_state: Arc::from(Vec::<u32>::new()),
            mask_state_source_subsets: Arc::from(Vec::<Arc<[u32]>>::new()),
            mask_source_subset_to_state: Arc::new(FxHashMap::default()),
            virtual_unit_repeat_projection: None,
            virtual_repeat_intersection_projections: Vec::new(),
            virtual_residual_projections: Vec::new(),
        }
    }

    /// Create a constraint-local runtime value from a fully initialized,
    /// vocabulary-only template.
    ///
    /// The immutable trie and token indexes are shared. Every cache or
    /// accelerator whose contents can depend on parser, lexer, or constraint
    /// state is recreated empty, so repeated schema builds cannot inherit
    /// schema-derived runtime state.
    /// Minimal producer-side representation for an external-vocabulary
    /// transfer artifact. The worker never executes masks with this value: the
    /// execution parent reconstructs all runtime-only indexes from trie+aliases
    /// after load. Keeping those indexes lazy here avoids doing the same O(vocab)
    /// work twice per schema.
    pub(crate) fn from_materialized_ordered_for_transfer(
        trie: Arc<DynamicMaskTrie>,
        token_aliases: Arc<Vec<Vec<u32>>>,
    ) -> Self {
        let mut vocab = Self::default();
        vocab.trie = trie;
        vocab.token_aliases = DynamicMaskAliasStore::Ordered(token_aliases);
        vocab.pending_source = None;
        vocab.initialized = true;
        vocab
    }

    pub(crate) fn fresh_runtime_instance(&self) -> Self {
        debug_assert!(self.initialized);
        debug_assert!(self.pending_source.is_none());
        Self {
            trie: Arc::clone(&self.trie),
            token_aliases: self.token_aliases.clone(),
            canonical_original_token_offsets: Arc::clone(
                &self.canonical_original_token_offsets,
            ),
            canonical_original_tokens: Arc::clone(&self.canonical_original_tokens),
            canonical_original_word_offsets: Arc::clone(&self.canonical_original_word_offsets),
            canonical_original_word_masks: Arc::clone(&self.canonical_original_word_masks),
            node_token_markers: Arc::clone(&self.node_token_markers),
            full_walk_token_markers: Arc::clone(&self.full_walk_token_markers),
            subtree_original_token_offsets: Arc::clone(
                &self.subtree_original_token_offsets,
            ),
            subtree_original_tokens: Arc::clone(&self.subtree_original_tokens),
            all_original_token_words: Arc::clone(&self.all_original_token_words),
            llg_slice_leftovers: Arc::clone(&self.llg_slice_leftovers),
            llg_master_admitted_words: Arc::clone(&self.llg_master_admitted_words),
            llg_master_residual_ops: Arc::clone(&self.llg_master_residual_ops),
            llg_master_max_safe_chars: self.llg_master_max_safe_chars,
            prepared_master_prover_row_ids: Arc::from(Vec::<u32>::new()),
            prepared_master_prover_offsets: Arc::from(Vec::<u32>::new()),
            prepared_master_prover_terminals: Arc::from(Vec::<TerminalID>::new()),
            prepared_master_coverage_row_ids: Arc::from(Vec::<u32>::new()),
            prepared_master_coverage_offsets: Arc::from(Vec::<u32>::new()),
            prepared_master_coverage_terminals: Arc::from(Vec::<TerminalID>::new()),
            prepared_safe_plus_complete_terminals: Arc::from(Vec::<TerminalID>::new()),
            prepared_safe_radius_row_ids: Arc::from(Vec::<u32>::new()),
            prepared_safe_radius_offsets: Arc::from(Vec::<u32>::new()),
            prepared_safe_radius_entries: Arc::from(Vec::<(TerminalID, u16)>::new()),
            pending_source: None,
            initialized: true,
            grammar_quotiented: self.grammar_quotiented,
            source_vocab_digest: self.source_vocab_digest,
            mask_cache: Arc::new(Mutex::new(DynamicMaskCache::default())),
            dense_subset16_cache: Arc::new(Mutex::new(FxHashMap::default())),
            lazy_union_cache: Arc::new(Mutex::new(DynamicLazyUnionCache::default())),
            direct_regular_frontier_cache: Arc::new(Mutex::new(FxHashMap::default())),
            direct_regular_wide_frontier_index_cache: Arc::new(Mutex::new(
                FxHashMap::default(),
            )),
            direct_regular_terminal_support: Arc::new(
                DirectRegularTerminalSupport::default(),
            ),
            self_loop_projections: Arc::new(Vec::new()),
            projection_by_source: Arc::from(Vec::<u32>::new()),
            projection_alias_vocab: Arc::from(Vec::<u32>::new()),
            projection_alias_h64: Arc::from(Vec::<u32>::new()),
            bounded_observation_sets: Arc::new(DynamicBoundedObservationSets::default()),
            terminal_observation_classes: Arc::from(Vec::<(TerminalID, Arc<[u32]>)>::new()),
            projected_terminal_quotients: Arc::from(Vec::<(TerminalID, Arc<TerminalProjectedQuotient>)>::new()),
            runtime_projected_terminal_quotients: Arc::new(OnceLock::new()),
            runtime_projected_terminal_quotient_cache: Arc::new(Mutex::new(FxHashMap::default())),
            projected_terminal_quotients_prepared: false,
            projected_terminal_text_cache: Arc::new(Mutex::new(FxHashMap::default())),
            projected_terminal_partition_cache: Arc::new(Mutex::new(FxHashMap::default())),
            projected_terminal_radius_cache: Arc::new(Mutex::new(FxHashMap::default())),
            raw_terminal_radius_cache: Arc::new(Mutex::new(FxHashMap::default())),
            pending_guard_blocked_mask_cache: Arc::new(Mutex::new(FxHashMap::default())),
            mask_tokenizer: None,
            mask_determinized_tokenizer: None,
            mask_projection_to_determinized: Arc::from(Vec::<u32>::new()),
            mask_tokenizer_fast_transitions: None,
            full_to_mask_state: Arc::from(Vec::<u32>::new()),
            mask_state_source_subsets: Arc::from(Vec::<Arc<[u32]>>::new()),
            mask_source_subset_to_state: Arc::new(FxHashMap::default()),
            virtual_unit_repeat_projection: None,
            virtual_repeat_intersection_projections: Vec::new(),
            virtual_residual_projections: Vec::new(),
        }
    }

    fn from_source(source: DynamicMaskVocabSource) -> Self {
        Self {
            trie: Arc::new(DynamicMaskTrie::new()),
            token_aliases: DynamicMaskAliasStore::Packed(Arc::new(Vec::new())),
            canonical_original_token_offsets: Arc::new(vec![0]),
            canonical_original_tokens: Arc::new(Vec::new()),
            canonical_original_word_offsets: Arc::new(vec![0]),
            canonical_original_word_masks: Arc::new(Vec::new()),
            node_token_markers: Arc::new(vec![0]),
            full_walk_token_markers: Arc::new(Vec::new()),
            subtree_original_token_offsets: Arc::new(vec![0]),
            subtree_original_tokens: Arc::new(Vec::new()),
            all_original_token_words: Arc::new(Vec::new()),
            llg_slice_leftovers: Arc::new(Vec::new()),
            llg_master_admitted_words: Arc::new(Vec::new()),
            llg_master_residual_ops: Arc::new(Vec::new()),
            llg_master_max_safe_chars: 0,
            prepared_master_prover_row_ids: Arc::from(Vec::<u32>::new()),
            prepared_master_prover_offsets: Arc::from(Vec::<u32>::new()),
            prepared_master_prover_terminals: Arc::from(Vec::<TerminalID>::new()),
            prepared_master_coverage_row_ids: Arc::from(Vec::<u32>::new()),
            prepared_master_coverage_offsets: Arc::from(Vec::<u32>::new()),
            prepared_master_coverage_terminals: Arc::from(Vec::<TerminalID>::new()),
            prepared_safe_plus_complete_terminals: Arc::from(Vec::<TerminalID>::new()),
            prepared_safe_radius_row_ids: Arc::from(Vec::<u32>::new()),
            prepared_safe_radius_offsets: Arc::from(Vec::<u32>::new()),
            prepared_safe_radius_entries: Arc::from(Vec::<(TerminalID, u16)>::new()),
            pending_source: Some(source),
            initialized: false,
            grammar_quotiented: false,
            source_vocab_digest: None,
            mask_cache: Arc::new(Mutex::new(DynamicMaskCache::default())),
            dense_subset16_cache: Arc::new(Mutex::new(FxHashMap::default())),
            lazy_union_cache: Arc::new(Mutex::new(DynamicLazyUnionCache::default())),
            direct_regular_frontier_cache: Arc::new(Mutex::new(FxHashMap::default())),
            direct_regular_wide_frontier_index_cache: Arc::new(Mutex::new(FxHashMap::default())),
            direct_regular_terminal_support: Arc::new(DirectRegularTerminalSupport::default()),
            self_loop_projections: Arc::new(Vec::new()),
            projection_by_source: Arc::from(Vec::<u32>::new()),
            projection_alias_vocab: Arc::from(Vec::<u32>::new()),
            projection_alias_h64: Arc::from(Vec::<u32>::new()),
            bounded_observation_sets: Arc::new(DynamicBoundedObservationSets::default()),
            terminal_observation_classes: Arc::from(Vec::<(TerminalID, Arc<[u32]>)>::new()),
            projected_terminal_quotients: Arc::from(Vec::<(TerminalID, Arc<TerminalProjectedQuotient>)>::new()),
            runtime_projected_terminal_quotients: Arc::new(OnceLock::new()),
            runtime_projected_terminal_quotient_cache: Arc::new(Mutex::new(FxHashMap::default())),
            projected_terminal_quotients_prepared: false,
            projected_terminal_text_cache: Arc::new(Mutex::new(FxHashMap::default())),
            projected_terminal_partition_cache: Arc::new(Mutex::new(FxHashMap::default())),
            projected_terminal_radius_cache: Arc::new(Mutex::new(FxHashMap::default())),
            raw_terminal_radius_cache: Arc::new(Mutex::new(FxHashMap::default())),
            pending_guard_blocked_mask_cache: Arc::new(Mutex::new(FxHashMap::default())),
            mask_tokenizer: None,
            mask_determinized_tokenizer: None,
            mask_projection_to_determinized: Arc::from(Vec::<u32>::new()),
            mask_tokenizer_fast_transitions: None,
            full_to_mask_state: Arc::from(Vec::<u32>::new()),
            mask_state_source_subsets: Arc::from(Vec::<Arc<[u32]>>::new()),
            mask_source_subset_to_state: Arc::new(FxHashMap::default()),
            virtual_unit_repeat_projection: None,
            virtual_repeat_intersection_projections: Vec::new(),
            virtual_residual_projections: Vec::new(),
        }
    }

    pub(crate) fn mark_grammar_quotiented(&mut self) {
        self.grammar_quotiented = true;
    }

    pub(crate) fn set_source_vocab_digest(&mut self, digest: [u8; 32]) {
        self.source_vocab_digest = Some(digest);
    }

    pub(crate) fn source_vocab_digest(&self) -> Option<[u8; 32]> {
        self.source_vocab_digest
    }

    pub(crate) fn is_grammar_quotiented(&self) -> bool {
        self.grammar_quotiented
    }

    pub(crate) fn from_packed(
        trie: Arc<DynamicMaskTrie>,
        token_aliases: Arc<Vec<Option<PackedDynamicMaskTokenAliases>>>,
    ) -> Self {
        Self::from_alias_store(trie, DynamicMaskAliasStore::Packed(token_aliases))
    }

    fn from_alias_store(trie: Arc<DynamicMaskTrie>, token_aliases: DynamicMaskAliasStore) -> Self {
        Self::from_alias_store_with_slices(trie, token_aliases, None)
    }

    fn from_alias_store_with_slices(
        trie: Arc<DynamicMaskTrie>,
        token_aliases: DynamicMaskAliasStore,
        slices: Option<&[Arc<DynamicMaskSliceTrie>]>,
    ) -> Self {
        let (canonical_original_token_offsets, canonical_original_tokens) =
            Self::flatten_canonical_original_tokens(&token_aliases);
        let master_layout_classes = slices.and_then(|slices| {
            Self::llg_master_layout_classes_from_slices(
                slices,
                canonical_original_token_offsets.len().saturating_sub(1),
            )
        });
        let (
            canonical_original_word_offsets,
            canonical_original_word_masks,
            fused_master_admission,
        ) = if let Some(classes) = master_layout_classes.as_deref() {
            Self::build_canonical_original_word_masks_and_master_admission(
                &canonical_original_token_offsets,
                &canonical_original_tokens,
                classes,
            )
        } else {
            let (offsets, masks) = Self::build_canonical_original_word_masks(
                &canonical_original_token_offsets,
                &canonical_original_tokens,
            );
            (offsets, masks, None)
        };
        let node_token_markers = Self::build_node_token_markers(
            trie.as_ref(),
            &canonical_original_token_offsets,
            &canonical_original_tokens,
        );
        let full_walk_token_markers =
            Self::build_full_walk_token_markers(trie.as_ref(), &node_token_markers);
        let (subtree_original_token_offsets, subtree_original_tokens) =
            Self::flatten_subtree_original_tokens(
                trie.as_ref(),
                &canonical_original_token_offsets,
                &canonical_original_tokens,
            );
        let all_original_token_words =
            Self::build_all_original_token_words(&subtree_original_tokens);
        Self {
            trie,
            token_aliases,
            canonical_original_token_offsets,
            canonical_original_tokens,
            canonical_original_word_offsets,
            canonical_original_word_masks,
            node_token_markers,
            full_walk_token_markers,
            subtree_original_token_offsets,
            subtree_original_tokens,
            all_original_token_words,
            llg_slice_leftovers: Arc::new(Vec::new()),
            llg_master_admitted_words: fused_master_admission
                .as_ref()
                .map_or_else(|| Arc::new(Vec::new()), |(_, words)| Arc::clone(words)),
            llg_master_residual_ops: Arc::new(Vec::new()),
            llg_master_max_safe_chars: fused_master_admission.map_or(0, |(max_safe_chars, _)| max_safe_chars),
            prepared_master_prover_row_ids: Arc::from(Vec::<u32>::new()),
            prepared_master_prover_offsets: Arc::from(Vec::<u32>::new()),
            prepared_master_prover_terminals: Arc::from(Vec::<TerminalID>::new()),
            prepared_master_coverage_row_ids: Arc::from(Vec::<u32>::new()),
            prepared_master_coverage_offsets: Arc::from(Vec::<u32>::new()),
            prepared_master_coverage_terminals: Arc::from(Vec::<TerminalID>::new()),
            prepared_safe_plus_complete_terminals: Arc::from(Vec::<TerminalID>::new()),
            prepared_safe_radius_row_ids: Arc::from(Vec::<u32>::new()),
            prepared_safe_radius_offsets: Arc::from(Vec::<u32>::new()),
            prepared_safe_radius_entries: Arc::from(Vec::<(TerminalID, u16)>::new()),
            pending_source: None,
            initialized: true,
            grammar_quotiented: false,
            source_vocab_digest: None,
            mask_cache: Arc::new(Mutex::new(DynamicMaskCache::default())),
            dense_subset16_cache: Arc::new(Mutex::new(FxHashMap::default())),
            lazy_union_cache: Arc::new(Mutex::new(DynamicLazyUnionCache::default())),
            direct_regular_frontier_cache: Arc::new(Mutex::new(FxHashMap::default())),
            direct_regular_wide_frontier_index_cache: Arc::new(Mutex::new(FxHashMap::default())),
            direct_regular_terminal_support: Arc::new(DirectRegularTerminalSupport::default()),
            self_loop_projections: Arc::new(Vec::new()),
            projection_by_source: Arc::from(Vec::<u32>::new()),
            projection_alias_vocab: Arc::from(Vec::<u32>::new()),
            projection_alias_h64: Arc::from(Vec::<u32>::new()),
            bounded_observation_sets: Arc::new(DynamicBoundedObservationSets::default()),
            terminal_observation_classes: Arc::from(Vec::<(TerminalID, Arc<[u32]>)>::new()),
            projected_terminal_quotients: Arc::from(Vec::<(TerminalID, Arc<TerminalProjectedQuotient>)>::new()),
            runtime_projected_terminal_quotients: Arc::new(OnceLock::new()),
            runtime_projected_terminal_quotient_cache: Arc::new(Mutex::new(FxHashMap::default())),
            projected_terminal_quotients_prepared: false,
            projected_terminal_text_cache: Arc::new(Mutex::new(FxHashMap::default())),
            projected_terminal_partition_cache: Arc::new(Mutex::new(FxHashMap::default())),
            projected_terminal_radius_cache: Arc::new(Mutex::new(FxHashMap::default())),
            raw_terminal_radius_cache: Arc::new(Mutex::new(FxHashMap::default())),
            pending_guard_blocked_mask_cache: Arc::new(Mutex::new(FxHashMap::default())),
            mask_tokenizer: None,
            mask_determinized_tokenizer: None,
            mask_projection_to_determinized: Arc::from(Vec::<u32>::new()),
            mask_tokenizer_fast_transitions: None,
            full_to_mask_state: Arc::from(Vec::<u32>::new()),
            mask_state_source_subsets: Arc::from(Vec::<Arc<[u32]>>::new()),
            mask_source_subset_to_state: Arc::new(FxHashMap::default()),
            virtual_unit_repeat_projection: None,
            virtual_repeat_intersection_projections: Vec::new(),
            virtual_residual_projections: Vec::new(),
        }
    }

    pub(crate) fn is_initialized(&self) -> bool {
        self.initialized
    }

    pub(crate) fn materialize_pending_source(&mut self) -> bool {
        let Some(source) = self.pending_source.take() else {
            return false;
        };
        self.trie = Arc::new(DynamicMaskTrie::from_vocab_prefix_tree(source.trie.as_ref()));
        self.token_aliases = DynamicMaskAliasStore::Ordered(source.token_aliases);
        (self.canonical_original_token_offsets, self.canonical_original_tokens) =
            Self::flatten_canonical_original_tokens(&self.token_aliases);
        (self.canonical_original_word_offsets, self.canonical_original_word_masks) =
            Self::build_canonical_original_word_masks(
                &self.canonical_original_token_offsets,
                &self.canonical_original_tokens,
            );
        self.node_token_markers = Self::build_node_token_markers(
            self.trie.as_ref(),
            &self.canonical_original_token_offsets,
            &self.canonical_original_tokens,
        );
        self.full_walk_token_markers =
            Self::build_full_walk_token_markers(self.trie.as_ref(), &self.node_token_markers);
        (self.subtree_original_token_offsets, self.subtree_original_tokens) =
            Self::flatten_subtree_original_tokens(
                self.trie.as_ref(),
                &self.canonical_original_token_offsets,
                &self.canonical_original_tokens,
            );
        self.all_original_token_words =
            Self::build_all_original_token_words(&self.subtree_original_tokens);
        self.initialized = true;
        true
    }

    fn flatten_canonical_original_tokens(
        token_aliases: &DynamicMaskAliasStore,
    ) -> (Arc<Vec<u32>>, Arc<Vec<u32>>) {
        if let DynamicMaskAliasStore::Flat { offsets, originals } = token_aliases {
            return (Arc::clone(offsets), Arc::clone(originals));
        }
        let alias_slots = match token_aliases {
            DynamicMaskAliasStore::Ordered(aliases) => aliases.len(),
            DynamicMaskAliasStore::Flat { .. } => unreachable!("flat alias store returned above"),
            DynamicMaskAliasStore::Packed(aliases) => aliases.len(),
        };
        let mut offsets = Vec::with_capacity(alias_slots + 1);
        let mut originals = Vec::new();
        offsets.push(0);
        for canonical_token in 0..alias_slots {
            match token_aliases {
                DynamicMaskAliasStore::Ordered(aliases) => {
                    originals.extend_from_slice(&aliases[canonical_token]);
                }
                DynamicMaskAliasStore::Flat { .. } => unreachable!("flat alias store returned above"),
                DynamicMaskAliasStore::Packed(aliases) => {
                    if let Some(alias) = aliases[canonical_token].as_ref() {
                        match alias {
                            PackedDynamicMaskTokenAliases::Single(token_id) => {
                                originals.push(*token_id);
                            }
                            PackedDynamicMaskTokenAliases::Many(token_ids) => {
                                originals.extend_from_slice(token_ids);
                            }
                        }
                    }
                }
            }
            offsets.push(originals.len() as u32);
        }
        (Arc::new(offsets), Arc::new(originals))
    }

    fn llg_master_layout_classes_from_slices(
        slices: &[Arc<DynamicMaskSliceTrie>],
        canonical_count: usize,
    ) -> Option<Vec<u16>> {
        let master = slices
            .iter()
            .find(|slice| slice.cache_id() == DYNAMIC_MASK_LLG_MASTER_CACHE_ID)?;
        let trie = master.trie();
        let mut classes = vec![0u16; canonical_count];
        let mut seen = vec![false; canonical_count];
        if let Some(root_token) = trie.node(0).token_id {
            let slot = root_token as usize;
            if slot >= canonical_count {
                return None;
            }
            seen[slot] = true;
        }
        let all_tokens = trie.all_subtree_tokens();
        for (root_slot, edge) in trie.children(0).iter().enumerate() {
            let class = trie.root_layout_class(root_slot)?;
            let canonical_tokens = all_tokens.get(trie.subtree_token_index_range(edge.child))?;
            for &canonical in canonical_tokens {
                let slot = canonical as usize;
                if slot >= canonical_count {
                    return None;
                }
                if seen[slot] && classes[slot] != class {
                    return None;
                }
                classes[slot] = class;
                seen[slot] = true;
            }
        }
        seen.iter().all(|&value| value).then_some(classes)
    }

    fn build_canonical_original_word_masks_and_master_admission(
        canonical_offsets: &[u32],
        canonical_original_tokens: &[u32],
        master_layout_classes: &[u16],
    ) -> (
        Arc<Vec<u32>>,
        Arc<Vec<(u32, u32)>>,
        Option<(u16, Arc<Vec<Vec<u32>>>)>,
    ) {
        let canonical_count = canonical_offsets.len().saturating_sub(1);
        if master_layout_classes.len() != canonical_count {
            let (offsets, masks) = Self::build_canonical_original_word_masks(
                canonical_offsets,
                canonical_original_tokens,
            );
            return (offsets, masks, None);
        }
        let word_len = canonical_original_tokens
            .iter()
            .copied()
            .max()
            .map_or(0, |token| token as usize / 32 + 1);
        let max_safe_chars = master_layout_classes
            .iter()
            .copied()
            .map(dynamic_mask_llg_master_safe_chars)
            .max()
            .unwrap_or(0);
        let mut exact_safe_words =
            vec![vec![0u32; word_len]; usize::from(max_safe_chars) + 1];
        let mut whitespace_words = vec![0u32; word_len];
        let mut scratch = vec![0u32; word_len];
        let mut touched = Vec::<u32>::new();
        let mut offsets = Vec::<u32>::with_capacity(canonical_count + 1);
        let mut masks = Vec::<(u32, u32)>::new();
        offsets.push(0);
        for canonical in 0..canonical_count {
            let class = master_layout_classes[canonical];
            let safe_chars = dynamic_mask_llg_master_safe_chars(class);
            let is_whitespace = dynamic_mask_llg_master_is_whitespace(class);
            let start = canonical_offsets[canonical] as usize;
            let end = canonical_offsets[canonical + 1] as usize;
            for &token_id in &canonical_original_tokens[start..end] {
                let word = token_id / 32;
                let word_slot = word as usize;
                let bit = 1u32 << (token_id % 32);
                let slot = unsafe { scratch.get_unchecked_mut(word_slot) };
                if *slot == 0 {
                    touched.push(word);
                }
                *slot |= bit;
                if safe_chars != 0 {
                    unsafe {
                        *exact_safe_words
                            .get_unchecked_mut(usize::from(safe_chars))
                            .get_unchecked_mut(word_slot) |= bit;
                    }
                }
                if is_whitespace {
                    unsafe { *whitespace_words.get_unchecked_mut(word_slot) |= bit; }
                }
            }
            for word in touched.drain(..) {
                let bits = unsafe { *scratch.get_unchecked(word as usize) };
                debug_assert_ne!(bits, 0);
                masks.push((word, bits));
                unsafe { *scratch.get_unchecked_mut(word as usize) = 0; }
            }
            offsets.push(masks.len() as u32);
        }

        let mut admitted_words =
            Vec::<Vec<u32>>::with_capacity((usize::from(max_safe_chars) + 1) * 2);
        let mut safe_prefix = vec![0u32; word_len];
        for radius in 0..=usize::from(max_safe_chars) {
            if radius != 0 {
                for (target, &source) in safe_prefix.iter_mut().zip(&exact_safe_words[radius]) {
                    *target |= source;
                }
            }
            admitted_words.push(safe_prefix.clone());
            let mut with_whitespace = safe_prefix.clone();
            for (target, &source) in with_whitespace.iter_mut().zip(&whitespace_words) {
                *target |= source;
            }
            admitted_words.push(with_whitespace);
        }
        (
            Arc::new(offsets),
            Arc::new(masks),
            Some((max_safe_chars, Arc::new(admitted_words))),
        )
    }

    fn build_canonical_original_word_masks(
        canonical_offsets: &[u32],
        canonical_original_tokens: &[u32],
    ) -> (Arc<Vec<u32>>, Arc<Vec<(u32, u32)>>) {
        let canonical_count = canonical_offsets.len().saturating_sub(1);
        let word_len = canonical_original_tokens
            .iter()
            .copied()
            .max()
            .map_or(0, |token| token as usize / 32 + 1);
        let mut scratch = vec![0u32; word_len];
        let mut touched = Vec::<u32>::new();
        let mut offsets = Vec::<u32>::with_capacity(canonical_count + 1);
        let mut masks = Vec::<(u32, u32)>::new();
        offsets.push(0);
        for canonical in 0..canonical_count {
            let start = canonical_offsets[canonical] as usize;
            let end = canonical_offsets[canonical + 1] as usize;
            for &token_id in &canonical_original_tokens[start..end] {
                let word = token_id / 32;
                let slot = unsafe { scratch.get_unchecked_mut(word as usize) };
                if *slot == 0 {
                    touched.push(word);
                }
                *slot |= 1u32 << (token_id % 32);
            }
            for word in touched.drain(..) {
                let bits = unsafe { *scratch.get_unchecked(word as usize) };
                debug_assert_ne!(bits, 0);
                masks.push((word, bits));
                unsafe { *scratch.get_unchecked_mut(word as usize) = 0; }
            }
            offsets.push(masks.len() as u32);
        }
        (Arc::new(offsets), Arc::new(masks))
    }

    /// Faster exact mask construction for the ordered vocabulary coordinate.
    /// Each canonical alias list is sorted by original token ID before this
    /// constructor is called, so equal 32-token words are contiguous.  Build
    /// sparse word masks in one linear pass rather than clearing/probing a
    /// model-vocabulary-sized scratch bitset for every canonical class.
    fn build_canonical_original_word_masks_sorted(
        canonical_offsets: &[u32],
        canonical_original_tokens: &[u32],
    ) -> (Arc<Vec<u32>>, Arc<Vec<(u32, u32)>>) {
        let canonical_count = canonical_offsets.len().saturating_sub(1);
        let mut offsets = Vec::<u32>::with_capacity(canonical_count + 1);
        let mut masks = Vec::<(u32, u32)>::new();
        offsets.push(0);
        for canonical in 0..canonical_count {
            let start = canonical_offsets[canonical] as usize;
            let end = canonical_offsets[canonical + 1] as usize;
            let aliases = &canonical_original_tokens[start..end];
            debug_assert!(aliases.windows(2).all(|pair| pair[0] <= pair[1]));
            let mut current_word = u32::MAX;
            let mut bits = 0u32;
            for &token_id in aliases {
                let word = token_id / 32;
                if word != current_word {
                    if current_word != u32::MAX {
                        masks.push((current_word, bits));
                    }
                    current_word = word;
                    bits = 0;
                }
                bits |= 1u32 << (token_id % 32);
            }
            if current_word != u32::MAX {
                masks.push((current_word, bits));
            }
            offsets.push(masks.len() as u32);
        }
        (Arc::new(offsets), Arc::new(masks))
    }

    fn build_all_original_token_words(originals: &[u32]) -> Arc<Vec<u32>> {
        let word_len = originals
            .iter()
            .copied()
            .max()
            .map_or(0, |token| token as usize / 32 + 1);
        let mut words = vec![0u32; word_len];
        for &token in originals {
            words[token as usize / 32] |= 1u32 << (token % 32);
        }
        Arc::new(words)
    }

    #[inline]
    pub(crate) fn all_original_token_words(&self) -> &[u32] {
        self.all_original_token_words.as_ref()
    }

    pub(crate) fn all_original_token_words_arc(&self) -> Arc<Vec<u32>> {
        Arc::clone(&self.all_original_token_words)
    }

    /// Reuse only vocabulary-global proof languages and admitted-token masks.
    /// The master walk trie itself is deliberately excluded: a grammar quotient
    /// can merge tokens that differ in safe-string length/whitespace class, so
    /// sharing the full-vocabulary master trie would both lose the O2 runtime
    /// reduction and couple the wrong canonical coordinate to this constraint.
    pub(crate) fn inherit_llg_vocab_acceleration_without_master_from(&mut self, source: &Self) {
        self.llg_slice_leftovers = Arc::new(
            source
                .llg_slice_leftovers
                .iter()
                .filter(|slice| slice.cache_id() != DYNAMIC_MASK_LLG_MASTER_CACHE_ID)
                .cloned()
                .collect(),
        );
        self.llg_master_admitted_words = Arc::clone(&source.llg_master_admitted_words);
        self.llg_master_residual_ops = Arc::new(Vec::new());
        self.llg_master_max_safe_chars = source.llg_master_max_safe_chars;
    }

    pub(crate) fn set_llg_slice_leftovers(
        &mut self,
        slices: Vec<(
            u32,
            Arc<VocabPartitionDfa>,
            Arc<DynamicMaskTrie>,
            Arc<Vec<u32>>,
            U8Set,
            u32,
        )>,
    ) {
        let mut built = Vec::with_capacity(slices.len());
        for (
            cache_id,
            dfa,
            trie,
            slice_original_token_words,
            slice_token_bytes,
            slice_max_token_byte_len,
        ) in slices
        {
            let slice_node_token_markers = Self::build_node_token_markers(
                trie.as_ref(),
                &self.canonical_original_token_offsets,
                &self.canonical_original_tokens,
            );
            let full_walk_token_markers =
                Self::build_full_walk_token_markers(trie.as_ref(), &slice_node_token_markers);
            let (subtree_original_token_offsets, subtree_original_tokens) =
                Self::flatten_subtree_original_tokens(
                    trie.as_ref(),
                    &self.canonical_original_token_offsets,
                    &self.canonical_original_tokens,
                );
            built.push(Arc::new(DynamicMaskSliceTrie {
                cache_id,
                dfa,
                trie,
                full_walk_token_markers,
                subtree_original_token_offsets,
                subtree_original_tokens,
                slice_original_token_words,
                slice_token_bytes,
                slice_max_token_byte_len,
                first_bytes: OnceLock::new(),
            }));
        }
        self.llg_slice_leftovers = Arc::new(built);
        self.rebuild_llg_master_residual_ops_from_master_trie();
    }

    pub(crate) fn set_llg_master_admitted_words(
        &mut self,
        max_safe_chars: u16,
        admitted_words: Vec<Vec<u32>>,
    ) {
        debug_assert_eq!(admitted_words.len(), (usize::from(max_safe_chars) + 1) * 2);
        self.llg_master_max_safe_chars = max_safe_chars;
        self.llg_master_admitted_words = Arc::new(admitted_words);
    }

    #[inline(always)]
    pub(crate) fn llg_master_admitted_words(
        &self,
        safe_radius: u16,
        whitespace: bool,
    ) -> Option<&[u32]> {
        if self.llg_master_admitted_words.is_empty() {
            return None;
        }
        let radius = safe_radius.min(self.llg_master_max_safe_chars) as usize;
        self.llg_master_admitted_words
            .get(radius * 2 + usize::from(whitespace))
            .map(Vec::as_slice)
    }

    #[inline(always)]
    pub(crate) fn llg_master_residual_ops(
        &self,
        safe_radius: u16,
        whitespace: bool,
    ) -> Option<usize> {
        if self.llg_master_residual_ops.is_empty() {
            return None;
        }
        let radius = safe_radius.min(self.llg_master_max_safe_chars) as usize;
        self.llg_master_residual_ops
            .get(radius * 2 + usize::from(whitespace))
            .copied()
            .map(|ops| ops as usize)
    }

    #[inline(always)]
    pub(crate) fn llg_master_max_safe_chars(&self) -> u16 {
        self.llg_master_max_safe_chars
    }

    fn rebuild_llg_master_residual_ops_from_master_trie(&mut self) {
        let Some(master) = self.llg_master_trie() else {
            self.llg_master_residual_ops = Arc::new(Vec::new());
            return;
        };
        let max_safe_chars = master
            .trie()
            .children(0)
            .iter()
            .enumerate()
            .filter_map(|(slot, _)| master.trie().root_layout_class(slot))
            .map(dynamic_mask_llg_master_safe_chars)
            .max()
            .unwrap_or(0);
        let mut rows = Vec::with_capacity((usize::from(max_safe_chars) + 1) * 2);
        for radius in 0..=max_safe_chars {
            for whitespace in [false, true] {
                let work = master
                    .trie()
                    .full_walk_ops_after_root_class_skips(|class| {
                        let safe_chars = dynamic_mask_llg_master_safe_chars(class);
                        (safe_chars != 0 && safe_chars <= radius)
                            || (whitespace && dynamic_mask_llg_master_is_whitespace(class))
                    })
                    .unwrap_or_else(|| master.trie().full_walk_ops().len());
                rows.push(u32::try_from(work).unwrap_or(u32::MAX));
            }
        }
        self.llg_master_residual_ops = Arc::new(rows);
    }

    /// Reconstruct cumulative admitted-token bitsets from the already-transferred
    /// master slice trie. Each root subtree has one exact `(safe_chars, whitespace)`
    /// language class, so this is linear in original-token membership plus the
    /// small number of cumulative rows and avoids serializing those dense rows.
    fn rebuild_llg_master_admitted_words_from_master_trie(&mut self) {
        let Some(master) = self.llg_master_trie() else {
            self.llg_master_admitted_words = Arc::new(Vec::new());
            self.llg_master_residual_ops = Arc::new(Vec::new());
            self.llg_master_max_safe_chars = 0;
            return;
        };
        let root_children = master.trie().children(0);
        let max_safe_chars = root_children
            .iter()
            .enumerate()
            .filter_map(|(slot, _)| master.trie().root_layout_class(slot))
            .map(dynamic_mask_llg_master_safe_chars)
            .max()
            .unwrap_or(0);
        let word_len = self.all_original_token_words.len();
        let mut exact_safe_words =
            vec![vec![0u32; word_len]; usize::from(max_safe_chars) + 1];
        let mut whitespace_words = vec![0u32; word_len];
        for (slot, edge) in root_children.iter().enumerate() {
            let Some(class) = master.trie().root_layout_class(slot) else {
                continue;
            };
            let safe_chars = dynamic_mask_llg_master_safe_chars(class);
            let whitespace = dynamic_mask_llg_master_is_whitespace(class);
            if safe_chars == 0 && !whitespace {
                continue;
            }
            for &token in master.subtree_original_tokens(edge.child) {
                let word = token as usize / 32;
                if word >= word_len {
                    continue;
                }
                let bit = 1u32 << (token % 32);
                if safe_chars != 0 {
                    exact_safe_words[usize::from(safe_chars)][word] |= bit;
                }
                if whitespace {
                    whitespace_words[word] |= bit;
                }
            }
        }
        let mut admitted_words =
            Vec::<Vec<u32>>::with_capacity((usize::from(max_safe_chars) + 1) * 2);
        let mut safe_prefix = vec![0u32; word_len];
        for radius in 0..=usize::from(max_safe_chars) {
            if radius != 0 {
                for (target, &source) in safe_prefix.iter_mut().zip(&exact_safe_words[radius]) {
                    *target |= source;
                }
            }
            admitted_words.push(safe_prefix.clone());
            let mut with_whitespace = safe_prefix.clone();
            for (target, &source) in with_whitespace.iter_mut().zip(&whitespace_words) {
                *target |= source;
            }
            admitted_words.push(with_whitespace);
        }
        self.llg_master_max_safe_chars = max_safe_chars;
        self.llg_master_admitted_words = Arc::new(admitted_words);
        self.rebuild_llg_master_residual_ops_from_master_trie();
    }

    #[inline(always)]
    pub(crate) fn prepared_master_provers(
        &self,
        source: u32,
        slice_slot: usize,
    ) -> &[TerminalID] {
        if slice_slot >= Self::PREPARED_PROOF_SLOT_COUNT {
            return &[];
        }
        let row = source as usize * Self::PREPARED_PROOF_SLOT_COUNT + slice_slot;
        let Some(&row_id) = self.prepared_master_prover_row_ids.get(row) else {
            return &[];
        };
        let row_id = row_id as usize;
        let Some((&start, &end)) = self
            .prepared_master_prover_offsets
            .get(row_id)
            .zip(self.prepared_master_prover_offsets.get(row_id + 1))
        else {
            return &[];
        };
        self.prepared_master_prover_terminals
            .get(start as usize..end as usize)
            .unwrap_or(&[])
    }

    pub(crate) fn prepared_master_proof_artifact(&self) -> PreparedMasterProofArtifact {
        PreparedMasterProofArtifact {
            positive_row_ids: self.prepared_master_prover_row_ids.as_ref().to_vec(),
            positive_offsets: self.prepared_master_prover_offsets.as_ref().to_vec(),
            positive_terminals: self.prepared_master_prover_terminals.as_ref().to_vec(),
            coverage_row_ids: self.prepared_master_coverage_row_ids.as_ref().to_vec(),
            coverage_offsets: self.prepared_master_coverage_offsets.as_ref().to_vec(),
            coverage_terminals: self.prepared_master_coverage_terminals.as_ref().to_vec(),
            safe_plus_complete_terminals: self
                .prepared_safe_plus_complete_terminals
                .as_ref()
                .to_vec(),
            safe_radius_row_ids: self.prepared_safe_radius_row_ids.as_ref().to_vec(),
            safe_radius_offsets: self.prepared_safe_radius_offsets.as_ref().to_vec(),
            safe_radius_entries: self.prepared_safe_radius_entries.as_ref().to_vec(),
        }
    }

    pub(crate) fn restore_prepared_master_proof_artifact(
        &mut self,
        artifact: PreparedMasterProofArtifact,
        source_state_count: usize,
    ) -> Result<(), String> {
        let expected_rows = source_state_count
            .checked_mul(Self::PREPARED_PROOF_SLOT_COUNT)
            .ok_or_else(|| "prepared master proof row count overflow".to_owned())?;
        let validate_interned =
            |label: &str, row_ids: &[u32], offsets: &[u32], terminals: &[TerminalID], rows: usize| {
                if row_ids.is_empty() {
                    return if offsets.is_empty() && terminals.is_empty() {
                        Ok(())
                    } else {
                        Err(format!("{label} has table data without row ids"))
                    };
                }
                if row_ids.len() != rows
                    || offsets.is_empty()
                    || offsets.first().copied() != Some(0)
                    || offsets.windows(2).any(|pair| pair[0] > pair[1])
                    || offsets.last().copied().map(|value| value as usize) != Some(terminals.len())
                {
                    return Err(format!("{label} has invalid interned row framing"));
                }
                let unique_rows = offsets.len() - 1;
                if row_ids.iter().any(|&row| row as usize >= unique_rows) {
                    return Err(format!("{label} has an out-of-range row id"));
                }
                for row in 0..unique_rows {
                    let start = offsets[row] as usize;
                    let end = offsets[row + 1] as usize;
                    if terminals[start..end].windows(2).any(|pair| pair[0] >= pair[1]) {
                        return Err(format!("{label} unique row {row} is not sorted and unique"));
                    }
                }
                Ok(())
            };
        validate_interned(
            "prepared master positive rows",
            &artifact.positive_row_ids,
            &artifact.positive_offsets,
            &artifact.positive_terminals,
            expected_rows,
        )?;
        validate_interned(
            "prepared master coverage rows",
            &artifact.coverage_row_ids,
            &artifact.coverage_offsets,
            &artifact.coverage_terminals,
            expected_rows,
        )?;
        if !artifact.coverage_row_ids.is_empty() {
            if artifact.positive_row_ids.is_empty() {
                return Err("prepared master coverage exists without positive row framing".to_owned());
            }
            for row in 0..expected_rows {
                let coverage = interned_terminal_row(
                    &artifact.coverage_row_ids,
                    &artifact.coverage_offsets,
                    &artifact.coverage_terminals,
                    row,
                );
                if interned_terminal_row(
                    &artifact.positive_row_ids,
                    &artifact.positive_offsets,
                    &artifact.positive_terminals,
                    row,
                )
                .iter()
                .any(|terminal| coverage.binary_search(terminal).is_err())
                {
                    return Err(format!(
                        "prepared master positive row {row} is not a subset of coverage"
                    ));
                }
            }
        }
        if artifact
            .safe_plus_complete_terminals
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        {
            return Err(
                "prepared safe+ complete terminal list is not sorted and unique".to_owned(),
            );
        }
        if !artifact.safe_radius_row_ids.is_empty() {
            if artifact.safe_radius_row_ids.len() != source_state_count
                || artifact.safe_radius_offsets.is_empty()
                || artifact.safe_radius_offsets.first().copied() != Some(0)
                || artifact.safe_radius_offsets.windows(2).any(|pair| pair[0] > pair[1])
                || artifact.safe_radius_offsets.last().copied().map(|value| value as usize)
                    != Some(artifact.safe_radius_entries.len())
            {
                return Err("prepared safe-radius rows have invalid interned framing".to_owned());
            }
            let unique_rows = artifact.safe_radius_offsets.len() - 1;
            if artifact
                .safe_radius_row_ids
                .iter()
                .any(|&row| row as usize >= unique_rows)
            {
                return Err("prepared safe-radius rows have an out-of-range row id".to_owned());
            }
            for row in 0..unique_rows {
                let start = artifact.safe_radius_offsets[row] as usize;
                let end = artifact.safe_radius_offsets[row + 1] as usize;
                if artifact.safe_radius_entries[start..end]
                    .windows(2)
                    .any(|pair| pair[0].0 >= pair[1].0)
                {
                    return Err(format!(
                        "prepared safe-radius unique row {row} is not sorted and unique"
                    ));
                }
            }
        } else if !artifact.safe_radius_offsets.is_empty() || !artifact.safe_radius_entries.is_empty()
        {
            return Err("prepared safe-radius table exists without row ids".to_owned());
        }
        self.prepared_master_prover_row_ids = Arc::from(artifact.positive_row_ids);
        self.prepared_master_prover_offsets = Arc::from(artifact.positive_offsets);
        self.prepared_master_prover_terminals = Arc::from(artifact.positive_terminals);
        self.prepared_master_coverage_row_ids = Arc::from(artifact.coverage_row_ids);
        self.prepared_master_coverage_offsets = Arc::from(artifact.coverage_offsets);
        self.prepared_master_coverage_terminals = Arc::from(artifact.coverage_terminals);
        self.prepared_safe_plus_complete_terminals =
            Arc::from(artifact.safe_plus_complete_terminals);
        self.prepared_safe_radius_row_ids = Arc::from(artifact.safe_radius_row_ids);
        self.prepared_safe_radius_offsets = Arc::from(artifact.safe_radius_offsets);
        self.prepared_safe_radius_entries = Arc::from(artifact.safe_radius_entries);
        Ok(())
    }

    /// Whether build/runtime preparation installed exact master-prover rows for
    /// this tokenizer source. An empty positive-terminal row is still a valid
    /// prepared row, so checking `prepared_master_provers()` itself is not
    /// sufficient.
    #[inline(always)]
    pub(crate) fn has_prepared_master_prover_row(&self, source: u32) -> bool {
        let row = source as usize * Self::PREPARED_PROOF_SLOT_COUNT;
        self.prepared_master_prover_row_ids
            .get(row + Self::PREPARED_PROOF_SLOT_COUNT - 1)
            .is_some()
    }

    /// Whether the compact prepared-master artifact contains row framing for
    /// every exact tokenizer source.  Transfer artifacts restore these rows
    /// verbatim; when they are complete, rebuilding the terminal projection
    /// quotients and solving the same source×slice products again on load is
    /// redundant.
    #[inline]
    pub(crate) fn has_complete_prepared_master_prover_rows(
        &self,
        source_state_count: usize,
    ) -> bool {
        source_state_count
            .checked_mul(Self::PREPARED_PROOF_SLOT_COUNT)
            .and_then(|rows| rows.checked_add(1))
            .is_some_and(|expected_offsets| {
                self.prepared_master_prover_offsets.len() == expected_offsets
            })
    }

    /// Exact build-time master-slice answer when this `(terminal, source)` is
    /// represented by an exact projected terminal quotient.
    ///
    /// `prepare_master_provers_all_sources` solves the complete
    /// quotientA-slice product for every retained quotient residual. Therefore
    /// once the compact row table is present, absence from a row is an exact
    /// negative result for a terminal whose quotient contains `source`.
    /// Missing rows or missing quotients remain `None` and must use the normal
    /// runtime proof path.
    #[inline]
    pub(crate) fn prepared_master_proof_result(
        &self,
        source: u32,
        slice_slot: usize,
        terminal: TerminalID,
    ) -> Option<bool> {
        if slice_slot >= Self::PREPARED_PROOF_SLOT_COUNT
            || self.prepared_master_prover_row_ids.is_empty()
        {
            return None;
        }
        let row = source as usize * Self::PREPARED_PROOF_SLOT_COUNT + slice_slot;
        self.prepared_master_prover_row_ids.get(row)?;
        if !self.prepared_master_coverage_row_ids.is_empty() {
            let coverage_id = *self.prepared_master_coverage_row_ids.get(row)? as usize;
            let (&start, &end) = self
                .prepared_master_coverage_offsets
                .get(coverage_id)
                .zip(self.prepared_master_coverage_offsets.get(coverage_id + 1))?;
            if self.prepared_master_coverage_terminals[start as usize..end as usize]
                .binary_search(&terminal)
                .is_err()
            {
                if slice_slot == Self::PREPARED_SAFE_PLUS_SLOT
                    && self
                        .prepared_safe_plus_complete_terminals
                        .binary_search(&terminal)
                        .is_ok()
                {
                    return Some(false);
                }
                return None;
            }
        } else if self.projected_terminal_quotient(terminal, source).is_none() {
            return None;
        }
        Some(
            self.prepared_master_provers(source, slice_slot)
                .binary_search(&terminal)
                .is_ok(),
        )
    }

    #[inline(always)]
    pub(crate) fn prepared_safe_radii(&self, source: u32) -> &[(TerminalID, u16)] {
        let Some(&row_id) = self.prepared_safe_radius_row_ids.get(source as usize) else {
            return &[];
        };
        let row_id = row_id as usize;
        let Some((&start, &end)) = self
            .prepared_safe_radius_offsets
            .get(row_id)
            .zip(self.prepared_safe_radius_offsets.get(row_id + 1))
        else {
            return &[];
        };
        self.prepared_safe_radius_entries
            .get(start as usize..end as usize)
            .unwrap_or(&[])
    }




    #[inline]
    pub(crate) fn prepared_safe_radius(
        &self,
        source: u32,
        terminal: TerminalID,
    ) -> Option<u16> {
        let row = self.prepared_safe_radii(source);
        if let Ok(index) = row.binary_search_by_key(&terminal, |&(candidate, _)| candidate) {
            return Some(row[index].1);
        }

        // Prepared radius rows intentionally omit zero radii to keep the
        // transfer compact. When the safe+ master-proof coverage row contains
        // this exact `(source, terminal)`, the radius solver was also run for
        // the same residual and an absent radius entry therefore means the
        // exact answer is zero, not "unprepared". Returning None here would
        // incorrectly launch the lazy projected-terminal quotient builder
        // during masking.
        self.prepared_master_proof_result(
            source,
            Self::PREPARED_SAFE_PLUS_SLOT,
            terminal,
        )
        .map(|_| 0)
    }

    /// Exact all-start-state master-slice proof directly over an independently
    /// compiled terminal DFA retained by the partitioned lexer builder.
    ///
    /// The slice DFA already supplies an exact byte partition. For each
    /// terminal state we group live outgoing targets by slice class and require
    /// complete byte coverage for every class that can remain on an accepting
    /// slice prefix. This avoids materializing a dense global tokenizer-state
    /// quotient while preserving exact containment semantics.
    fn terminal_dfa_partition_all_transparent_states(
        dfa: &LexerDfa,
        group: u32,
        partition: &VocabPartitionDfa,
    ) -> Option<(Vec<bool>, usize, usize)> {
        if dfa.has_epsilon_transitions() {
            return None;
        }
        let q_count = dfa.num_states();
        let p_count = partition.state_count();
        let class_count = partition.class_count();
        if q_count == 0 || p_count == 0 || class_count == 0 {
            return Some((vec![false; q_count], 0, 0));
        }

        let mut class_sizes = vec![0usize; class_count];
        let mut class_representatives = vec![u8::MAX; class_count];
        for raw in 0u16..=255 {
            let byte = raw as u8;
            let class = partition.byte_class(byte) as usize;
            class_sizes[class] += 1;
            if class_representatives[class] == u8::MAX {
                class_representatives[class] = byte;
            }
        }

        let relevant_classes = (0..p_count as u32)
            .map(|p| {
                if !partition.can_reach_accepting(p) {
                    return SmallVec::<[(usize, u32); 8]>::new();
                }
                class_representatives
                    .iter()
                    .enumerate()
                    .filter_map(|(class, &byte)| {
                        let target = partition.step(p, byte);
                        partition
                            .can_reach_accepting(target)
                            .then_some((class, target))
                    })
                    .collect::<SmallVec<[(usize, u32); 8]>>()
            })
            .collect::<Vec<_>>();

        let pair_count = p_count.saturating_mul(q_count);
        let pair_index = |p: u32, q: u32| p as usize * q_count + q as usize;
        let mut predecessors = vec![SmallVec::<[u32; 8]>::new(); pair_count];
        let mut bad = vec![false; pair_count];
        let mut queue = VecDeque::<u32>::new();
        let mut edge_count = 0usize;

        let state_live = |state: u32| {
            dfa.finalizers(state).contains(group as usize)
                || dfa
                    .possible_future_group_ids(state)
                    .contains(group as usize)
        };

        for q in 0..q_count as u32 {
            if !state_live(q) {
                for p in 0..p_count as u32 {
                    if !partition.can_reach_accepting(p) {
                        continue;
                    }
                    let current = pair_index(p, q);
                    bad[current] = true;
                    queue.push_back(current as u32);
                }
                continue;
            }

            let mut covered = vec![0usize; class_count];
            let mut targets = (0..class_count)
                .map(|_| SmallVec::<[u32; 4]>::new())
                .collect::<Vec<_>>();
            for (byte, target) in dfa.transitions(q) {
                if !state_live(target) {
                    continue;
                }
                let class = partition.byte_class(byte) as usize;
                covered[class] += 1;
                if !targets[class].contains(&target) {
                    targets[class].push(target);
                }
            }

            for p in 0..p_count as u32 {
                if !partition.can_reach_accepting(p) {
                    continue;
                }
                let current = pair_index(p, q);
                let relevant = &relevant_classes[p as usize];
                if relevant
                    .iter()
                    .any(|&(class, _)| covered[class] != class_sizes[class])
                {
                    bad[current] = true;
                    queue.push_back(current as u32);
                    continue;
                }
                for &(class, p_target) in relevant {
                    for &q_target in &targets[class] {
                        let target = pair_index(p_target, q_target);
                        predecessors[target].push(current as u32);
                        edge_count = edge_count.saturating_add(1);
                    }
                }
            }
        }

        while let Some(target) = queue.pop_front() {
            for &pred in &predecessors[target as usize] {
                if !bad[pred as usize] {
                    bad[pred as usize] = true;
                    queue.push_back(pred);
                }
            }
        }

        let start = partition.start_state();
        let mut transparent = vec![false; q_count];
        if partition.can_reach_accepting(start) {
            for q in 0..q_count as u32 {
                transparent[q as usize] = !bad[pair_index(start, q)];
            }
        } else {
            for q in 0..q_count as u32 {
                transparent[q as usize] = state_live(q);
            }
        }
        Some((transparent, pair_count, edge_count))
    }

    /// Exact safe-slice transparency and bounded radius for every residual state
    /// of one retained terminal DFA. Both answers are determined by the same
    /// shortest-counterexample product graph, so compute that graph once instead
    /// of separately solving containment and radius.
    fn terminal_dfa_partition_all_transparency_and_repeat_radii(
        dfa: &LexerDfa,
        group: u32,
        slice: &VocabPartitionDfa,
        max_repetitions: u32,
    ) -> Option<(Vec<bool>, Vec<u32>, usize, usize)> {
        if dfa.has_epsilon_transitions() {
            return None;
        }
        let q_count = dfa.num_states();
        let p_count = slice.state_count();
        let class_count = slice.class_count();
        if q_count == 0 || p_count == 0 || class_count == 0 {
            return Some((vec![false; q_count], vec![0; q_count], 0, 0));
        }

        let live_states = (0..q_count as u32)
            .map(|state| {
                dfa.finalizers(state).contains(group as usize)
                    || dfa
                        .possible_future_group_ids(state)
                        .contains(group as usize)
            })
            .collect::<Vec<_>>();
        let state_live = |state: u32| live_states[state as usize];

        let mut class_sizes = vec![0usize; class_count];
        let mut class_representatives = vec![u8::MAX; class_count];
        for raw in 0u16..=255 {
            let byte = raw as u8;
            let class = slice.byte_class(byte) as usize;
            class_sizes[class] += 1;
            if class_representatives[class] == u8::MAX {
                class_representatives[class] = byte;
            }
        }

        let relevant_classes = (0..p_count as u32)
            .map(|p| {
                if !slice.can_reach_accepting(p) {
                    return SmallVec::<[(usize, u32, u8); 8]>::new();
                }
                class_representatives
                    .iter()
                    .enumerate()
                    .filter_map(|(class, &byte)| {
                        let target = slice.step(p, byte);
                        slice.can_reach_accepting(target).then_some((
                            class,
                            target,
                            u8::from(slice.is_accepting(target)),
                        ))
                    })
                    .collect::<SmallVec<[(usize, u32, u8); 8]>>()
            })
            .collect::<Vec<_>>();

        // Minimum completed-atom cost from each slice state to an accepting
        // state. Every byte in one slice class has the same target, so one
        // representative per class is exact here.
        let mut slice_reverse = vec![SmallVec::<[(u32, u8); 8]>::new(); p_count];
        for p in 0..p_count as u32 {
            let mut seen = SmallVec::<[u32; 16]>::new();
            for &byte in &class_representatives {
                let target = slice.step(p, byte);
                if target as usize >= p_count || seen.contains(&target) {
                    continue;
                }
                seen.push(target);
                slice_reverse[target as usize]
                    .push((p, u8::from(slice.is_accepting(target))));
            }
        }
        let mut min_to_accept = vec![u32::MAX; p_count];
        let mut zero_one = VecDeque::<u32>::new();
        for p in 0..p_count as u32 {
            if slice.is_accepting(p) {
                min_to_accept[p as usize] = 0;
                zero_one.push_back(p);
            }
        }
        while let Some(target) = zero_one.pop_front() {
            let base = min_to_accept[target as usize];
            for &(source, cost) in &slice_reverse[target as usize] {
                let candidate = base.saturating_add(u32::from(cost));
                if candidate < min_to_accept[source as usize] {
                    min_to_accept[source as usize] = candidate;
                    if cost == 0 {
                        zero_one.push_front(source);
                    } else {
                        zero_one.push_back(source);
                    }
                }
            }
        }

        let mut covered_by_q = Vec::<Vec<usize>>::with_capacity(q_count);
        let mut targets_by_q = Vec::<Vec<SmallVec<[u32; 4]>>>::with_capacity(q_count);
        for q in 0..q_count as u32 {
            let mut covered = vec![0usize; class_count];
            let mut targets = (0..class_count)
                .map(|_| SmallVec::<[u32; 4]>::new())
                .collect::<Vec<_>>();
            if state_live(q) {
                for (byte, target) in dfa.transitions(q) {
                    if !state_live(target) {
                        continue;
                    }
                    let class = slice.byte_class(byte) as usize;
                    covered[class] += 1;
                    if !targets[class].contains(&target) {
                        targets[class].push(target);
                    }
                }
            }
            covered_by_q.push(covered);
            targets_by_q.push(targets);
        }

        let pair_count = p_count.saturating_mul(q_count);
        let index = |p: u32, q: u32| p as usize * q_count + q as usize;
        let mut reverse = vec![SmallVec::<[(u32, u8); 8]>::new(); pair_count];
        let mut distance = vec![u32::MAX; pair_count];
        let mut max_seed_distance = 0u32;
        let mut edge_count = 0usize;

        for p in 0..p_count as u32 {
            if !slice.can_reach_accepting(p) {
                continue;
            }
            for q in 0..q_count as u32 {
                let current = index(p, q);
                if !state_live(q) {
                    distance[current] = 0;
                    continue;
                }
                for &(class, p_target, enter_cost) in &relevant_classes[p as usize] {
                    if covered_by_q[q as usize][class] != class_sizes[class] {
                        let completion = min_to_accept[p_target as usize];
                        if completion != u32::MAX {
                            let candidate = u32::from(enter_cost).saturating_add(completion);
                            if candidate < distance[current] {
                                distance[current] = candidate;
                                max_seed_distance = max_seed_distance.max(candidate);
                            }
                        }
                    }
                    for &q_target in &targets_by_q[q as usize][class] {
                        let target = index(p_target, q_target);
                        reverse[target].push((current as u32, enter_cost));
                        edge_count = edge_count.saturating_add(1);
                    }
                }
            }
        }

        // Product edges are 0/1-weighted. When the counterexample seeds are too,
        // 0-1 BFS avoids heap traffic; larger seed distances still use Dijkstra.
        if max_seed_distance <= 1 {
            let mut queue = VecDeque::<(u32, u32)>::new();
            for (state, &dist) in distance.iter().enumerate() {
                match dist {
                    0 => queue.push_front((state as u32, 0)),
                    1 => queue.push_back((state as u32, 1)),
                    _ => {}
                }
            }
            while let Some((target, dist)) = queue.pop_front() {
                if distance[target as usize] != dist {
                    continue;
                }
                for &(pred, cost) in &reverse[target as usize] {
                    let candidate = dist.saturating_add(u32::from(cost));
                    if candidate < distance[pred as usize] {
                        distance[pred as usize] = candidate;
                        if cost == 0 {
                            queue.push_front((pred, candidate));
                        } else {
                            queue.push_back((pred, candidate));
                        }
                    }
                }
            }
        } else {
            let mut heap = BinaryHeap::<(Reverse<u32>, u32)>::new();
            for (state, &dist) in distance.iter().enumerate() {
                if dist != u32::MAX {
                    heap.push((Reverse(dist), state as u32));
                }
            }
            while let Some((Reverse(dist), target)) = heap.pop() {
                if distance[target as usize] != dist {
                    continue;
                }
                for &(pred, cost) in &reverse[target as usize] {
                    let candidate = dist.saturating_add(u32::from(cost));
                    if candidate < distance[pred as usize] {
                        distance[pred as usize] = candidate;
                        heap.push((Reverse(candidate), pred));
                    }
                }
            }
        }

        let start = slice.start_state();
        let start_can_accept = slice.can_reach_accepting(start);
        let transparent = (0..q_count as u32)
            .map(|q| {
                if start_can_accept {
                    state_live(q) && distance[index(start, q)] == u32::MAX
                } else {
                    state_live(q)
                }
            })
            .collect::<Vec<_>>();
        let radii = (0..q_count as u32)
            .map(|q| {
                if !state_live(q) {
                    0
                } else {
                    match distance[index(start, q)] {
                        u32::MAX => max_repetitions,
                        first_counterexample => first_counterexample
                            .saturating_sub(1)
                            .min(max_repetitions),
                    }
                }
            })
            .collect::<Vec<_>>();
        Some((transparent, radii, pair_count, edge_count))
    }

    /// Build exact compact master-slice and bounded-radius certificates from
    /// the partitioned lexer's retained terminal-residual coordinates. The
    /// result contains no terminal quotient transition matrix and is therefore
    /// cheap to transfer and restore.
    pub(crate) fn prepare_master_provers_from_residual_coordinates(
        &mut self,
        tokenizer: &Tokenizer,
        source_state_count: usize,
        safe_plus: &VocabPartitionDfa,
        whitespace: &VocabPartitionDfa,
        safe_slice_token_bytes: U8Set,
        max_safe_chars: u16,
    ) -> Option<(usize, usize, usize)> {
        let coordinates = tokenizer.terminal_residual_coordinates()?;
        if source_state_count == 0 || coordinates.len() == 0 {
            return Some((0, 0, 0));
        }
        let required_bytes = |dfa: &VocabPartitionDfa| {
            let mut bytes = U8Set::empty();
            for raw in 0u16..=255 {
                let byte = raw as u8;
                let used = (0..dfa.state_count() as u32).any(|state| {
                    dfa.can_reach_accepting(state)
                        && dfa.can_reach_accepting(dfa.step(state, byte))
                });
                if used {
                    bytes.insert(byte);
                }
            }
            bytes
        };
        let safe_plus_required = required_bytes(safe_plus);
        let whitespace_required = required_bytes(whitespace);

        struct DirectPreparedPair {
            terminal: TerminalID,
            slice_slot: usize,
            transparent: Vec<bool>,
            product_pairs: usize,
            edges: usize,
        }

        struct DirectSafePreparedPair {
            terminal: TerminalID,
            transparent: Option<Vec<bool>>,
            radii: Option<Vec<u32>>,
            product_pairs: usize,
            edges: usize,
        }

        let candidates = (0..tokenizer.num_terminals())
            .filter(|&terminal| {
                tokenizer
                    .terminal_byte_support(terminal)
                    .is_some_and(|support| safe_slice_token_bytes.is_subset(&support))
                    && coordinates.terminal_dfa_and_group(terminal).is_some()
            })
            .collect::<Vec<_>>();
        let verify_combined =
            std::env::var_os("GLRMASK_VERIFY_DIRECT_RESIDUAL_MASTER_PROVERS").is_some();
        let build_whitespace = || {
            candidates
                .par_iter()
                .copied()
                .filter(|&terminal| {
                    tokenizer
                        .terminal_byte_support(terminal)
                        .is_some_and(|support| whitespace_required.is_subset(&support))
                })
                .filter_map(|terminal| {
                    let (dfa, group) = coordinates.terminal_dfa_and_group(terminal)?;
                    let (transparent, product_pairs, edges) =
                        Self::terminal_dfa_partition_all_transparent_states(dfa, group, whitespace)?;
                    Some(DirectPreparedPair {
                        terminal,
                        slice_slot: Self::PREPARED_WHITESPACE_SLOT,
                        transparent,
                        product_pairs,
                        edges,
                    })
                })
                .collect::<Vec<_>>()
        };
        let build_safe = || {
            candidates
                .par_iter()
                .copied()
                .filter_map(|terminal| {
                    let (dfa, group) = coordinates.terminal_dfa_and_group(terminal)?;
                    let safe_supported = tokenizer
                        .terminal_byte_support(terminal)
                        .is_some_and(|support| safe_plus_required.is_subset(&support));
                    if max_safe_chars == 0 {
                        let (transparent, product_pairs, edges) =
                            Self::terminal_dfa_partition_all_transparent_states(
                                dfa, group, safe_plus,
                            )?;
                        return Some(DirectSafePreparedPair {
                            terminal,
                            transparent: safe_supported.then_some(transparent),
                            radii: None,
                            product_pairs,
                            edges,
                        });
                    }
                    let (transparent, radii, product_pairs, edges) =
                        Self::terminal_dfa_partition_all_transparency_and_repeat_radii(
                            dfa,
                            group,
                            safe_plus,
                            u32::from(max_safe_chars),
                        )?;
                    if verify_combined && safe_supported {
                        let (reference, _, _) = Self::terminal_dfa_partition_all_transparent_states(
                            dfa, group, safe_plus,
                        )?;
                        assert_eq!(
                            transparent, reference,
                            "combined direct residual safe+ solver disagrees for terminal {terminal}"
                        );
                    }
                    Some(DirectSafePreparedPair {
                        terminal,
                        transparent: safe_supported.then_some(transparent),
                        radii: Some(radii),
                        product_pairs,
                        edges,
                    })
                })
                .collect::<Vec<_>>()
        };
        let (proofs, safe_jobs) = rayon::join(build_whitespace, build_safe);

        let terminal_count = tokenizer.num_terminals();
        let mut by_terminal_slot = (0..terminal_count)
            .map(|_| [None::<Vec<bool>>, None::<Vec<bool>>])
            .collect::<Vec<_>>();
        let mut product_pairs = 0usize;
        let mut edges = 0usize;
        for proof in proofs {
            product_pairs = product_pairs.saturating_add(proof.product_pairs);
            edges = edges.saturating_add(proof.edges);
            by_terminal_slot[proof.terminal as usize][proof.slice_slot] = Some(proof.transparent);
        }
        let mut radius_by_terminal =
            (0..terminal_count).map(|_| None::<Vec<u32>>).collect::<Vec<_>>();
        for proof in safe_jobs {
            product_pairs = product_pairs.saturating_add(proof.product_pairs);
            edges = edges.saturating_add(proof.edges);
            if let Some(transparent) = proof.transparent {
                by_terminal_slot[proof.terminal as usize][Self::PREPARED_SAFE_PLUS_SLOT] =
                    Some(transparent);
            }
            if let Some(radii) = proof.radii {
                radius_by_terminal[proof.terminal as usize] = Some(radii);
            }
        }
        let mut safe_plus_complete_terminals = candidates
            .iter()
            .copied()
            .filter(|terminal| {
                by_terminal_slot[*terminal as usize][Self::PREPARED_SAFE_PLUS_SLOT].is_some()
                    && radius_by_terminal[*terminal as usize].is_some()
            })
            .collect::<Vec<_>>();
        safe_plus_complete_terminals.sort_unstable();
        safe_plus_complete_terminals.dedup();

        let row_count = source_state_count * Self::PREPARED_PROOF_SLOT_COUNT;
        let mut positive_rows = vec![SmallVec::<[TerminalID; 4]>::new(); row_count];
        let mut coverage_rows = vec![SmallVec::<[TerminalID; 4]>::new(); row_count];
        let mut radius_rows = vec![SmallVec::<[(TerminalID, u16); 4]>::new(); source_state_count];
        let source_limit = source_state_count.min(coordinates.len());
        positive_rows[..source_limit * Self::PREPARED_PROOF_SLOT_COUNT]
            .par_chunks_mut(Self::PREPARED_PROOF_SLOT_COUNT)
            .zip(
                coverage_rows[..source_limit * Self::PREPARED_PROOF_SLOT_COUNT]
                    .par_chunks_mut(Self::PREPARED_PROOF_SLOT_COUNT),
            )
            .zip(radius_rows[..source_limit].par_iter_mut())
            .enumerate()
            .for_each(|(source, ((positive_rows, coverage_rows), radius_row))| {
                let Some(entries) = coordinates.row(source as u32) else {
                    return;
                };
                for &(terminal, residual) in entries {
                    let proof_slots = &by_terminal_slot[terminal as usize];
                    for slice_slot in 0..Self::PREPARED_PROOF_SLOT_COUNT {
                        let Some(transparent) = proof_slots[slice_slot].as_ref() else {
                            continue;
                        };
                        coverage_rows[slice_slot].push(terminal);
                        if transparent.get(residual as usize).copied().unwrap_or(false) {
                            positive_rows[slice_slot].push(terminal);
                        }
                    }
                    if let Some(radii) = radius_by_terminal[terminal as usize].as_ref() {
                        let radius = radii
                            .get(residual as usize)
                            .copied()
                            .unwrap_or(0)
                            .min(u32::from(max_safe_chars)) as u16;
                        if radius != 0 {
                            radius_row.push((terminal, radius));
                        }
                    }
                }
            });

        let positive_entry_count = positive_rows.iter().map(SmallVec::len).sum::<usize>();
        let parallel_intern = candidates.len() >= 64 && rayon::current_num_threads() > 1;
        let ((positive, coverage), radius) = if parallel_intern {
            rayon::join(
                || {
                    rayon::join(
                        || intern_terminal_rows(positive_rows),
                        || intern_terminal_rows(coverage_rows),
                    )
                },
                || intern_radius_rows(radius_rows),
            )
        } else {
            (
                (
                    intern_terminal_rows(positive_rows),
                    intern_terminal_rows(coverage_rows),
                ),
                intern_radius_rows(radius_rows),
            )
        };
        let (positive_row_ids, positive_offsets, positive_terminals) = positive;
        let (coverage_row_ids, coverage_offsets, coverage_terminals) = coverage;
        let (radius_row_ids, radius_offsets, radius_entries) = radius;

        self.prepared_master_prover_row_ids = Arc::from(positive_row_ids);
        self.prepared_master_prover_offsets = Arc::from(positive_offsets);
        self.prepared_master_prover_terminals = Arc::from(positive_terminals);
        self.prepared_master_coverage_row_ids = Arc::from(coverage_row_ids);
        self.prepared_master_coverage_offsets = Arc::from(coverage_offsets);
        self.prepared_master_coverage_terminals = Arc::from(coverage_terminals);
        self.prepared_safe_plus_complete_terminals = Arc::from(safe_plus_complete_terminals);
        self.prepared_safe_radius_row_ids = Arc::from(radius_row_ids);
        self.prepared_safe_radius_offsets = Arc::from(radius_offsets);
        self.prepared_safe_radius_entries = Arc::from(radius_entries);
        Some((positive_entry_count, product_pairs, edges))
    }

    /// Build positive-only parser-independent master-slice certificates for
    /// every exact source TSID represented by the currently prepared terminal
    /// quotients. One `(terminal quotient, slice)` product graph is solved once
    /// for all quotient residual states; source TSIDs are then fanned out
    /// through the quotient's exact source->residual mapping.
    pub(crate) fn prepare_master_provers_all_sources(
        &mut self,
        tokenizer: &Tokenizer,
        source_state_count: usize,
        include_safe_radii: bool,
    ) -> (usize, usize) {
        let Some(safe_plus) = self
            .llg_slice_leftovers
            .iter()
            .find(|slice| slice.cache_id() == 0)
            .cloned()
        else {
            return (0, 0);
        };
        let Some(whitespace) = self
            .llg_slice_leftovers
            .iter()
            .find(|slice| slice.cache_id() == 3)
            .cloned()
        else {
            return (0, 0);
        };
        let mut proof_languages = Vec::<(usize, Arc<VocabPartitionDfa>, U8Set)>::new();
        let required_bytes = |dfa: &VocabPartitionDfa| {
            let mut bytes = U8Set::empty();
            for raw in 0u16..=255 {
                let byte = raw as u8;
                let used = (0..dfa.state_count() as u32).any(|state| {
                    dfa.can_reach_accepting(state)
                        && dfa.can_reach_accepting(dfa.step(state, byte))
                });
                if used {
                    bytes.insert(byte);
                }
            }
            bytes
        };
        proof_languages.push((
            Self::PREPARED_SAFE_PLUS_SLOT,
            Arc::clone(&safe_plus.dfa),
            required_bytes(safe_plus.dfa()),
        ));
        proof_languages.push((
            Self::PREPARED_WHITESPACE_SLOT,
            Arc::clone(&whitespace.dfa),
            required_bytes(whitespace.dfa()),
        ));
        let quotients = self.active_projected_terminal_quotients();
        if quotients.is_empty() || source_state_count == 0 {
            self.prepared_master_prover_row_ids = Arc::from(Vec::<u32>::new());
            self.prepared_master_prover_offsets = Arc::from(Vec::<u32>::new());
            self.prepared_master_prover_terminals = Arc::from(Vec::<TerminalID>::new());
            self.prepared_master_coverage_row_ids = Arc::from(Vec::<u32>::new());
            self.prepared_master_coverage_offsets = Arc::from(Vec::<u32>::new());
            self.prepared_master_coverage_terminals = Arc::from(Vec::<TerminalID>::new());
            self.prepared_safe_plus_complete_terminals = Arc::from(Vec::<TerminalID>::new());
            self.prepared_safe_radius_row_ids = Arc::from(Vec::<u32>::new());
            self.prepared_safe_radius_offsets = Arc::from(Vec::<u32>::new());
            self.prepared_safe_radius_entries = Arc::from(Vec::<(TerminalID, u16)>::new());
            return (0, 0);
        }

        struct PreparedPair {
            terminal: TerminalID,
            slice_slot: usize,
            certified_sources: Vec<u32>,
            product_pairs: usize,
            elapsed_ns: u64,
            verified: usize,
            mismatches: usize,
            unknown: usize,
        }
        let verify = std::env::var_os("GLRMASK_VERIFY_PREPARED_MASTER_PROVERS").is_some();
        let jobs = quotients
            .par_iter()
            .flat_map_iter(|(terminal, quotient)| {
                proof_languages
                    .iter()
                    .filter_map(move |(slot, slice, required)| {
                        tokenizer
                            .terminal_byte_support(*terminal)
                            .is_some_and(|support| required.is_subset(&support))
                            .then_some((*terminal, Arc::clone(quotient), *slot, Arc::clone(slice)))
                    })
            })
            .map(|(terminal, quotient, slice_slot, slice)| {
                let started = std::time::Instant::now();
                let (transparent, product_pairs) =
                    Self::terminal_partition_all_transparent_states(quotient.as_ref(), slice.as_ref());
                let mut verified = 0usize;
                let mut mismatches = 0usize;
                let mut unknown = 0usize;
                if verify {
                    for (&source, &projected) in quotient
                        .projected_source_states()
                        .iter()
                        .zip(quotient.projected_states_for_sources())
                    {
                        let batch = transparent.get(projected as usize).copied().unwrap_or(false);
                        match Self::terminal_partition_product_is_transparent(
                            quotient.as_ref(),
                            slice.as_ref(),
                            source,
                            200_000,
                        ) {
                            Some(reference) => {
                                verified += 1;
                                mismatches += usize::from(reference != batch);
                            }
                            None => unknown += 1,
                        }
                    }
                }
                let certified_sources = quotient
                    .projected_source_states()
                    .iter()
                    .copied()
                    .zip(quotient.projected_states_for_sources().iter().copied())
                    .filter_map(|(source, projected)| {
                        ((source as usize) < source_state_count
                            && transparent.get(projected as usize).copied().unwrap_or(false))
                        .then_some(source)
                    })
                    .collect::<Vec<_>>();
                PreparedPair {
                    terminal,
                    slice_slot,
                    certified_sources,
                    product_pairs,
                    elapsed_ns: started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64,
                    verified,
                    mismatches,
                    unknown,
                }
            })
            .collect::<Vec<_>>();

        let mut rows = vec![
            SmallVec::<[TerminalID; 4]>::new();
            source_state_count * Self::PREPARED_PROOF_SLOT_COUNT
        ];
        let mut product_pairs = 0usize;
        let mut pair_work_ns = 0u64;
        let mut max_pair_ns = 0u64;
        let mut verified = 0usize;
        let mut mismatches = 0usize;
        let mut unknown = 0usize;
        for job in jobs {
            product_pairs = product_pairs.saturating_add(job.product_pairs);
            pair_work_ns = pair_work_ns.saturating_add(job.elapsed_ns);
            max_pair_ns = max_pair_ns.max(job.elapsed_ns);
            verified += job.verified;
            mismatches += job.mismatches;
            unknown += job.unknown;
            for source in job.certified_sources {
                rows[source as usize * Self::PREPARED_PROOF_SLOT_COUNT + job.slice_slot]
                    .push(job.terminal);
            }
        }
        if std::env::var_os("GLRMASK_PROFILE_PREPARED_MASTER_PROVERS").is_some() {
            eprintln!(
                "[glrmask/profile][prepared_master_pair_work] jobs={} work_ms={:.3} max_pair_ms={:.3}",
                quotients.len() * 2,
                pair_work_ns as f64 / 1e6,
                max_pair_ns as f64 / 1e6,
            );
        }
        if verify {
            eprintln!(
                "[glrmask/profile][prepared_master_verify] verified={} mismatches={} unknown={}",
                verified, mismatches, unknown,
            );
            assert_eq!(mismatches, 0, "batched master proof disagrees with per-source exact proof");
        }

        let entry_count = rows.iter().map(SmallVec::len).sum::<usize>();
        let (row_ids, offsets, terminals) = intern_terminal_rows(rows);
        if include_safe_radii {
            let max_radius = self.llg_master_max_safe_chars;
            let radius_jobs = quotients
                .par_iter()
                .map(|(terminal, quotient)| {
                    let radii = Self::terminal_partition_all_repeat_radii(
                        quotient.as_ref(),
                        safe_plus.dfa(),
                        u32::from(max_radius),
                    );
                    (*terminal, Arc::clone(quotient), radii)
                })
                .collect::<Vec<_>>();
            let mut radius_rows =
                vec![SmallVec::<[(TerminalID, u16); 4]>::new(); source_state_count];
            for (terminal, quotient, radii) in radius_jobs {
                for (&source, &projected) in quotient
                    .projected_source_states()
                    .iter()
                    .zip(quotient.projected_states_for_sources())
                {
                    if source as usize >= source_state_count {
                        continue;
                    }
                    let radius = radii
                        .get(projected as usize)
                        .copied()
                        .unwrap_or(0)
                        .min(u32::from(max_radius)) as u16;
                    if radius != 0 {
                        radius_rows[source as usize].push((terminal, radius));
                    }
                }
            }
            let (radius_row_ids, radius_offsets, radius_entries) = intern_radius_rows(radius_rows);
            self.prepared_safe_radius_row_ids = Arc::from(radius_row_ids);
            self.prepared_safe_radius_offsets = Arc::from(radius_offsets);
            self.prepared_safe_radius_entries = Arc::from(radius_entries);
        } else {
            self.prepared_safe_radius_row_ids = Arc::from(Vec::<u32>::new());
            self.prepared_safe_radius_offsets = Arc::from(Vec::<u32>::new());
            self.prepared_safe_radius_entries = Arc::from(Vec::<(TerminalID, u16)>::new());
        }
        // Keep the borrow of the active quotient sidecar alive through the
        // optional radius analysis above, then publish the compact prepared
        // rows only after all quotient reads are complete.
        self.prepared_master_prover_row_ids = Arc::from(row_ids);
        self.prepared_master_prover_offsets = Arc::from(offsets);
        self.prepared_master_prover_terminals = Arc::from(terminals);
        self.prepared_master_coverage_row_ids = Arc::from(Vec::<u32>::new());
        self.prepared_master_coverage_offsets = Arc::from(Vec::<u32>::new());
        self.prepared_master_coverage_terminals = Arc::from(Vec::<TerminalID>::new());
        self.prepared_safe_plus_complete_terminals = Arc::from(Vec::<TerminalID>::new());
        (entry_count, product_pairs)
    }

    /// Exact all-start-state analogue of `projected_terminal_slice_repeat_radius`.
    /// `distance[p,q]` is the minimum number of additional completed slice
    /// atoms in a completed slice word whose prefix first leaves the live
    /// terminal residual. Solving this reverse shortest-path problem once gives
    /// the bounded radius for every quotient residual state simultaneously.
    fn terminal_partition_all_repeat_radii(
        quotient: &TerminalProjectedQuotient,
        slice: &VocabPartitionDfa,
        max_repetitions: u32,
    ) -> Vec<u32> {
        let q_count = quotient.projected_state_count();
        let p_count = slice.state_count();
        if q_count == 0 || p_count == 0 || max_repetitions == 0 {
            return vec![0; q_count];
        }

        let mut representatives = SmallVec::<[(u8, u8, u8); 32]>::new();
        for raw in 0u16..=255 {
            let byte = raw as u8;
            let p_class = slice.byte_class(byte);
            let q_class = quotient.projected_byte_class(byte);
            if !representatives
                .iter()
                .any(|&(p, q, _)| p == p_class && q == q_class)
            {
                representatives.push((p_class, q_class, byte));
            }
        }

        // Minimum completed-atom cost from each slice state to some accepting
        // state. Edge cost is one exactly when the target state completes an
        // atom; UTF-8 continuation transitions therefore cost zero.
        let mut slice_reverse = vec![SmallVec::<[(u32, u8); 8]>::new(); p_count];
        for p in 0..p_count as u32 {
            let mut seen = SmallVec::<[u32; 16]>::new();
            for &(_, _, byte) in &representatives {
                let target = slice.step(p, byte);
                if target as usize >= p_count || seen.contains(&target) {
                    continue;
                }
                seen.push(target);
                slice_reverse[target as usize]
                    .push((p, u8::from(slice.is_accepting(target))));
            }
        }
        let mut min_to_accept = vec![u32::MAX; p_count];
        let mut zero_one = VecDeque::<u32>::new();
        for p in 0..p_count as u32 {
            if slice.is_accepting(p) {
                min_to_accept[p as usize] = 0;
                zero_one.push_back(p);
            }
        }
        while let Some(target) = zero_one.pop_front() {
            let base = min_to_accept[target as usize];
            for &(source, cost) in &slice_reverse[target as usize] {
                let candidate = base.saturating_add(u32::from(cost));
                if candidate < min_to_accept[source as usize] {
                    min_to_accept[source as usize] = candidate;
                    if cost == 0 {
                        zero_one.push_front(source);
                    } else {
                        zero_one.push_back(source);
                    }
                }
            }
        }

        let pair_count = p_count * q_count;
        let index = |p: u32, q: u32| p as usize * q_count + q as usize;
        let mut reverse = vec![SmallVec::<[(u32, u8); 8]>::new(); pair_count];
        let mut distance = vec![u32::MAX; pair_count];
        let mut heap = BinaryHeap::<(Reverse<u32>, u32)>::new();

        for p in 0..p_count as u32 {
            if !slice.can_reach_accepting(p) {
                continue;
            }
            for q in 0..q_count as u32 {
                let current = index(p, q);
                let q_live = quotient.projected_state_is_accepting(q)
                    || quotient.projected_state_has_future(q);
                if !q_live {
                    distance[current] = 0;
                    heap.push((Reverse(0), current as u32));
                    continue;
                }
                for &(_, q_class, byte) in &representatives {
                    let p_target = slice.step(p, byte);
                    if !slice.can_reach_accepting(p_target) {
                        continue;
                    }
                    let enter_cost = u32::from(slice.is_accepting(p_target));
                    let q_target = quotient.projected_step_class(q, q_class);
                    let target_live = q_target.is_some_and(|target| {
                        quotient.projected_state_is_accepting(target)
                            || quotient.projected_state_has_future(target)
                    });
                    if !target_live {
                        let completion = min_to_accept[p_target as usize];
                        if completion != u32::MAX {
                            let candidate = enter_cost.saturating_add(completion);
                            if candidate < distance[current] {
                                distance[current] = candidate;
                                heap.push((Reverse(candidate), current as u32));
                            }
                        }
                        continue;
                    }
                    let target = index(p_target, q_target.expect("live target exists"));
                    reverse[target].push((current as u32, enter_cost as u8));
                }
            }
        }

        while let Some((Reverse(dist), target)) = heap.pop() {
            if distance[target as usize] != dist {
                continue;
            }
            for &(pred, cost) in &reverse[target as usize] {
                let candidate = dist.saturating_add(u32::from(cost));
                if candidate < distance[pred as usize] {
                    distance[pred as usize] = candidate;
                    heap.push((Reverse(candidate), pred));
                }
            }
        }

        let start = slice.start_state();
        (0..q_count as u32)
            .map(|q| {
                if !quotient.projected_state_is_accepting(q)
                    && !quotient.projected_state_has_future(q)
                {
                    0
                } else {
                    match distance[index(start, q)] {
                        u32::MAX => max_repetitions,
                        first_counterexample => first_counterexample
                            .saturating_sub(1)
                            .min(max_repetitions),
                    }
                }
            })
            .collect()
    }

    /// Exact all-start-state version of `terminal_partition_product_is_transparent`.
    /// A product pair `(p,q)` is bad when some byte that keeps `P` on an
    /// accepting-prefix path either has no live `Q` successor or reaches a bad
    /// product pair. This computes the least backwards closure of those bad
    /// pairs once, yielding the answer for every quotient state simultaneously.
    fn terminal_partition_all_transparent_states(
        quotient: &TerminalProjectedQuotient,
        partition: &VocabPartitionDfa,
    ) -> (Vec<bool>, usize) {
        let q_count = quotient.projected_state_count();
        let p_count = partition.state_count();
        if q_count == 0 || p_count == 0 {
            return (vec![false; q_count], 0);
        }

        // Refine the two exact byte partitions once. One representative is
        // sufficient for each `(P class, Q class)` pair because both automata
        // transition identically for every byte in that pair.
        let mut representatives = SmallVec::<[(u8, u8, u8); 32]>::new();
        for raw in 0u16..=255 {
            let byte = raw as u8;
            let p_class = partition.byte_class(byte);
            let q_class = quotient.projected_byte_class(byte);
            if !representatives
                .iter()
                .any(|&(p, q, _)| p == p_class && q == q_class)
            {
                representatives.push((p_class, q_class, byte));
            }
        }

        let pair_count = p_count.saturating_mul(q_count);
        let pair_index = |p: u32, q: u32| p as usize * q_count + q as usize;
        let mut predecessors = vec![SmallVec::<[u32; 8]>::new(); pair_count];
        let mut bad = vec![false; pair_count];
        let mut queue = VecDeque::<u32>::new();

        for p in 0..p_count as u32 {
            if !partition.can_reach_accepting(p) {
                continue;
            }
            for q in 0..q_count as u32 {
                let current = pair_index(p, q);
                if !quotient.projected_state_is_accepting(q)
                    && !quotient.projected_state_has_future(q)
                {
                    bad[current] = true;
                    queue.push_back(current as u32);
                    continue;
                }

                let mut immediate_bad = false;
                for &(_, q_class, byte) in &representatives {
                    let p_target = partition.step(p, byte);
                    if !partition.can_reach_accepting(p_target) {
                        continue;
                    }
                    let Some(q_target) = quotient.projected_step_class(q, q_class) else {
                        immediate_bad = true;
                        break;
                    };
                    if !quotient.projected_state_is_accepting(q_target)
                        && !quotient.projected_state_has_future(q_target)
                    {
                        immediate_bad = true;
                        break;
                    }
                    let target = pair_index(p_target, q_target);
                    predecessors[target].push(current as u32);
                }
                if immediate_bad {
                    bad[current] = true;
                    queue.push_back(current as u32);
                }
            }
        }

        while let Some(target) = queue.pop_front() {
            for &pred in &predecessors[target as usize] {
                if !bad[pred as usize] {
                    bad[pred as usize] = true;
                    queue.push_back(pred);
                }
            }
        }

        let start = partition.start_state();
        let mut transparent = vec![false; q_count];
        if partition.can_reach_accepting(start) {
            for q in 0..q_count as u32 {
                transparent[q as usize] = !bad[pair_index(start, q)];
            }
        } else {
            for q in 0..q_count as u32 {
                transparent[q as usize] = quotient.projected_state_is_accepting(q)
                    || quotient.projected_state_has_future(q);
            }
        }
        (transparent, pair_count)
    }

    #[inline]
    fn active_projected_terminal_quotients(
        &self,
    ) -> &[(TerminalID, Arc<TerminalProjectedQuotient>)] {
        if self.projected_terminal_quotients_prepared {
            self.projected_terminal_quotients.as_ref()
        } else {
            self.runtime_projected_terminal_quotients
                .get()
                .map(Arc::as_ref)
                .unwrap_or(&[])
        }
    }

    #[inline(always)]
    pub(crate) fn llg_master_trie(&self) -> Option<&DynamicMaskSliceTrie> {
        // O2 already walks a grammar quotient that is orders of magnitude
        // smaller than the model vocabulary. Switching to a separately
        // structured master trie loses that quotient and is unnecessary; the
        // exact ordinary quotient walk is both the semantic authority and the
        // cheaper coordinate here.
        if self.grammar_quotiented {
            return None;
        }
        self.llg_slice_by_cache_id(DYNAMIC_MASK_LLG_MASTER_CACHE_ID)
    }

    #[inline(always)]
    pub(crate) fn llg_slice_leftovers(&self) -> &[Arc<DynamicMaskSliceTrie>] {
        self.llg_slice_leftovers.as_ref()
    }

    #[inline(always)]
    pub(crate) fn has_llg_slice_leftovers(&self) -> bool {
        !self.llg_slice_leftovers.is_empty()
    }

    #[inline(always)]
    pub(crate) fn llg_slice_by_cache_id(&self, cache_id: u32) -> Option<&DynamicMaskSliceTrie> {
        self.llg_slice_leftovers
            .iter()
            .find(|slice| slice.cache_id() == cache_id)
            .map(Arc::as_ref)
    }

    #[inline(always)]
    pub(crate) fn residual_original_token_words_for(
        &self,
        trie: &DynamicMaskTrie,
    ) -> Option<&[u32]> {
        self.llg_slice_leftovers
            .iter()
            .find(|slice| {
                slice.cache_id() & 0x100 != 0 && std::ptr::eq(trie, slice.trie())
            })
            .map(|slice| slice.slice_original_token_words())
    }

    #[inline(always)]
    pub(crate) fn full_walk_token_markers_for(&self, trie: &DynamicMaskTrie) -> &[u64] {
        if std::ptr::eq(trie, self.trie.as_ref()) {
            return self.full_walk_token_markers();
        }
        if let Some(slice) = self
            .llg_slice_leftovers
            .iter()
            .find(|slice| std::ptr::eq(trie, slice.trie()))
        {
            return slice.full_walk_token_markers();
        }
        debug_assert!(false, "unknown dynamic-mask walk trie");
        &[]
    }

    #[inline(always)]
    pub(crate) fn subtree_original_tokens_for(
        &self,
        trie: &DynamicMaskTrie,
        node: u32,
    ) -> &[u32] {
        if std::ptr::eq(trie, self.trie.as_ref()) {
            return self.subtree_original_tokens(node);
        }
        if let Some(slice) = self
            .llg_slice_leftovers
            .iter()
            .find(|slice| std::ptr::eq(trie, slice.trie()))
        {
            return slice.subtree_original_tokens(node);
        }
        debug_assert!(false, "unknown dynamic-mask walk trie");
        &[]
    }


    fn flatten_subtree_original_tokens(
        trie: &DynamicMaskTrie,
        canonical_offsets: &[u32],
        canonical_original_tokens: &[u32],
    ) -> (Arc<Vec<u32>>, Arc<Vec<u32>>) {
        let subtree_canonical_tokens = trie.all_subtree_tokens();
        let mut offsets = Vec::with_capacity(subtree_canonical_tokens.len() + 1);
        let mut originals = Vec::new();
        offsets.push(0);
        for &canonical_token in subtree_canonical_tokens {
            let index = canonical_token as usize;
            let start = canonical_offsets[index] as usize;
            let end = canonical_offsets[index + 1] as usize;
            originals.extend_from_slice(&canonical_original_tokens[start..end]);
            offsets.push(originals.len() as u32);
        }
        (Arc::new(offsets), Arc::new(originals))
    }

    fn build_node_token_markers(
        trie: &DynamicMaskTrie,
        canonical_offsets: &[u32],
        canonical_original_tokens: &[u32],
    ) -> Arc<Vec<u64>> {
        const FALLBACK_TAG: u64 = 1u64 << 63;
        let mut markers = Vec::with_capacity(trie.nodes.len());
        for node in &trie.nodes {
            let Some(canonical_token) = node.token_id else {
                markers.push(0);
                continue;
            };
            let index = canonical_token as usize;
            let start = canonical_offsets[index] as usize;
            let end = canonical_offsets[index + 1] as usize;
            let aliases = &canonical_original_tokens[start..end];
            let Some(&first_token) = aliases.first() else {
                markers.push(FALLBACK_TAG | (canonical_token as u64 + 1));
                continue;
            };
            let word = first_token / 32;
            let mut bits = 0u32;
            let mut one_word = true;
            for &token_id in aliases {
                if token_id / 32 != word {
                    one_word = false;
                    break;
                }
                bits |= 1u32 << (token_id % 32);
            }
            if one_word {
                debug_assert_ne!(bits, 0);
                debug_assert!(word < (1u32 << 31));
                markers.push((u64::from(word) << 32) | u64::from(bits));
            } else {
                markers.push(FALLBACK_TAG | (canonical_token as u64 + 1));
            }
        }
        Arc::new(markers)
    }

    fn build_full_walk_token_markers(
        trie: &DynamicMaskTrie,
        node_token_markers: &[u64],
    ) -> Arc<Vec<u64>> {
        Arc::new(
            trie.full_walk_token_nodes()
                .iter()
                .map(|&node| {
                    debug_assert!((node as usize) < node_token_markers.len());
                    unsafe { *node_token_markers.get_unchecked(node as usize) }
                })
                .collect(),
        )
    }

    #[inline]
    pub(crate) fn subtree_original_tokens(&self, node: u32) -> &[u32] {
        let canonical_range = self.trie.subtree_token_index_range(node);
        let start = self.subtree_original_token_offsets[canonical_range.start] as usize;
        let end = self.subtree_original_token_offsets[canonical_range.end] as usize;
        &self.subtree_original_tokens[start..end]
    }

    #[inline]
    pub(crate) fn canonical_token_count(&self) -> usize {
        let indexed = self.canonical_original_token_offsets.len().saturating_sub(1);
        if indexed != 0 {
            return indexed;
        }
        match &self.token_aliases {
            DynamicMaskAliasStore::Ordered(aliases) => aliases.len(),
            DynamicMaskAliasStore::Flat { offsets, .. } => offsets.len().saturating_sub(1),
            DynamicMaskAliasStore::Packed(aliases) => aliases.len(),
        }
    }

    pub(crate) fn token_ids(&self, canonical_token_id: u32) -> Option<&[u32]> {
        let index = canonical_token_id as usize;
        let end_index = index.checked_add(1)?;
        let (&start, &end) = self
            .canonical_original_token_offsets
            .get(index)
            .zip(self.canonical_original_token_offsets.get(end_index))?;
        (start != end).then(|| {
            &self.canonical_original_tokens[start as usize..end as usize]
        })
    }

    #[inline(always)]
    pub(crate) fn token_word_masks(&self, canonical_token_id: u32) -> &[(u32, u32)] {
        let index = canonical_token_id as usize;
        let start = unsafe { *self.canonical_original_word_offsets.get_unchecked(index) } as usize;
        let end = unsafe { *self.canonical_original_word_offsets.get_unchecked(index + 1) } as usize;
        unsafe { self.canonical_original_word_masks.get_unchecked(start..end) }
    }

    #[inline(always)]
    pub(crate) fn node_token_marker(&self, node: u32) -> u64 {
        debug_assert!((node as usize) < self.node_token_markers.len());
        unsafe { *self.node_token_markers.get_unchecked(node as usize) }
    }

    #[inline(always)]
    pub(crate) fn full_walk_token_markers(&self) -> &[u64] {
        self.full_walk_token_markers.as_ref()
    }

    pub(crate) fn set_direct_regular_terminal_support(
        &mut self,
        support: DirectRegularTerminalSupport,
    ) {
        self.direct_regular_terminal_support = Arc::new(support);
    }

    pub(crate) fn direct_regular_terminal_support(&self) -> &DirectRegularTerminalSupport {
        self.direct_regular_terminal_support.as_ref()
    }

    pub(crate) fn cached_direct_regular_wide_frontier_index(
        &self,
        key: usize,
    ) -> Option<usize> {
        self.direct_regular_wide_frontier_index_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&key)
            .copied()
    }

    pub(crate) fn cache_direct_regular_wide_frontier_index(
        &self,
        key: usize,
        index: usize,
    ) {
        self.direct_regular_wide_frontier_index_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(key, index);
    }

    pub(crate) fn set_self_loop_projections(
        &mut self,
        projections: Vec<DynamicSelfLoopProjection>,
    ) {
        let state_count = self.mask_tokenizer.as_ref().map_or_else(
            || {
                projections
                    .iter()
                    .map(|projection| projection.source_state as usize + 1)
                    .max()
                    .unwrap_or(0)
            },
            |tokenizer| tokenizer.num_states() as usize,
        );
        let mut by_source = vec![u32::MAX; state_count];
        for (index, projection) in projections.iter().enumerate() {
            if let Some(slot) = by_source.get_mut(projection.source_state as usize) {
                *slot = index as u32;
            }
        }
        self.projection_by_source = Arc::from(by_source);
        self.self_loop_projections = Arc::new(projections);
    }

    pub(crate) fn set_projection_alias_vocab(&mut self, aliases: Vec<u32>) {
        self.projection_alias_vocab = Arc::from(aliases);
    }

    pub(crate) fn set_projection_alias_h64(&mut self, aliases: Vec<u32>) {
        self.projection_alias_h64 = Arc::from(aliases);
    }

    pub(crate) fn set_mask_tokenizer_quotient(
        &mut self,
        tokenizer: Tokenizer,
        full_to_mask_state: Vec<u32>,
    ) {
        debug_assert!(!full_to_mask_state.is_empty());
        debug_assert!(full_to_mask_state
            .iter()
            .all(|&state| state < tokenizer.num_states()));
        self.mask_tokenizer_fast_transitions = Self::build_full_walk_fast_transitions(&tokenizer);
        self.mask_tokenizer = Some(Arc::new(tokenizer));
        self.mask_determinized_tokenizer = None;
        self.mask_projection_to_determinized = Arc::from(Vec::<u32>::new());
        self.full_to_mask_state = Arc::from(full_to_mask_state);
        self.mask_state_source_subsets = Arc::from(Vec::<Arc<[u32]>>::new());
        self.mask_source_subset_to_state = Arc::new(FxHashMap::default());
        self.virtual_unit_repeat_projection = None;
        self.virtual_repeat_intersection_projections.clear();
        self.virtual_residual_projections.clear();
    }

    pub(crate) fn set_mask_tokenizer_source_subsets(
        &mut self,
        source_subsets: Vec<Box<[u32]>>,
    ) {
        let Some(tokenizer) = self.mask_runtime_tokenizer() else {
            self.mask_state_source_subsets = Arc::from(Vec::<Arc<[u32]>>::new());
            self.mask_source_subset_to_state = Arc::new(FxHashMap::default());
            return;
        };
        if source_subsets.len() != tokenizer.num_states() as usize {
            self.mask_state_source_subsets = Arc::from(Vec::<Arc<[u32]>>::new());
            self.mask_source_subset_to_state = Arc::new(FxHashMap::default());
            return;
        }
        let by_state = source_subsets
            .into_iter()
            .map(Arc::<[u32]>::from)
            .collect::<Vec<_>>();
        let mut by_subset = FxHashMap::default();
        by_subset.reserve(by_state.len());
        for (state, subset) in by_state.iter().enumerate() {
            by_subset.insert(Arc::clone(subset), state as u32);
        }
        self.mask_state_source_subsets = Arc::from(by_state);
        self.mask_source_subset_to_state = Arc::new(by_subset);
    }

    #[inline]
    pub(crate) fn has_mask_tokenizer_source_subsets(&self) -> bool {
        !self.mask_source_subset_to_state.is_empty()
            // A derivative of an intermediate mask projection records subsets
            // in projection-state coordinates, not exact source-tokenizer
            // coordinates. It remains useful for unioning execution states in
            // the strict walker, but must not be queried with ConstraintState
            // source lexer states.
            && !(self.mask_tokenizer.is_some() && self.mask_determinized_tokenizer.is_some())
    }

    #[inline]
    pub(crate) fn has_mask_subset_provenance(&self) -> bool {
        !self.mask_source_subset_to_state.is_empty()
    }

    /// Return an already-materialized mask execution state for the union of
    /// exact constraint lexer states. For a direct determinization the retained
    /// subsets are in exact-source coordinates. For a second-stage derivative
    /// of a finite virtual projection, first map each exact state into that
    /// projection and close it there; the inverse subset table is keyed in that
    /// intermediate coordinate.
    pub(crate) fn mask_runtime_state_for_source_states(
        &self,
        source_tokenizer: &Tokenizer,
        source_states: &[u32],
    ) -> Option<u32> {
        if source_states.is_empty() || self.mask_source_subset_to_state.is_empty() {
            return None;
        }
        if self.mask_tokenizer.is_some() && self.mask_determinized_tokenizer.is_some() {
            let projection = self.mask_tokenizer.as_deref()?;
            let mut subset = SmallVec::<[u32; 32]>::new();
            for &state in source_states {
                let projected = self.mask_projection_state(state);
                subset.extend_from_slice(&projection.singleton_epsilon_closure(projected));
            }
            subset.sort_unstable();
            subset.dedup();
            return self.mask_source_subset_to_state.get(subset.as_slice()).copied();
        }
        self.mask_projection_state_for_source_states(source_tokenizer, source_states)
    }

    pub(crate) fn mask_projection_state_for_source_states(
        &self,
        source_tokenizer: &Tokenizer,
        source_states: &[u32],
    ) -> Option<u32> {
        if source_states.is_empty() || self.mask_source_subset_to_state.is_empty() {
            return None;
        }
        let mut subset = SmallVec::<[u32; 16]>::new();
        for &state in source_states {
            let closure = source_tokenizer.singleton_epsilon_closure(state);
            subset.extend_from_slice(&closure);
        }
        subset.sort_unstable();
        subset.dedup();
        self.mask_source_subset_to_state
            .get(subset.as_slice())
            .copied()
    }

    /// Return an already-materialized deterministic mask state whose exact
    /// source-state subset is the union of `projection_states`.
    ///
    /// This never creates a DFA state at runtime. It is only an inverse lookup
    /// into source-subset provenance retained from compile-time determinization.
    pub(crate) fn mask_projection_state_for_projection_states(
        &self,
        projection_states: &[u32],
    ) -> Option<u32> {
        if projection_states.is_empty()
            || self.mask_state_source_subsets.is_empty()
            || self.mask_source_subset_to_state.is_empty()
        {
            return None;
        }
        let mut subset = SmallVec::<[u32; 32]>::new();
        for &state in projection_states {
            let source = self.mask_state_source_subsets.get(state as usize)?;
            subset.extend_from_slice(source);
        }
        subset.sort_unstable();
        subset.dedup();
        self.mask_source_subset_to_state
            .get(subset.as_slice())
            .copied()
    }

    pub(crate) fn set_virtual_unit_repeat_mask_projection(
        &mut self,
        tokenizer: Tokenizer,
        projection: VirtualZeroMinUnitRepeatMaskProjection,
    ) {
        debug_assert_eq!(
            tokenizer.num_states(),
            projection.mask_state_count(),
        );
        self.mask_tokenizer_fast_transitions = Self::build_full_walk_fast_transitions(&tokenizer);
        self.mask_tokenizer = Some(Arc::new(tokenizer));
        self.mask_determinized_tokenizer = None;
        self.mask_projection_to_determinized = Arc::from(Vec::<u32>::new());
        self.full_to_mask_state = Arc::from(Vec::<u32>::new());
        self.mask_state_source_subsets = Arc::from(Vec::<Arc<[u32]>>::new());
        self.mask_source_subset_to_state = Arc::new(FxHashMap::default());
        self.virtual_unit_repeat_projection = Some(projection);
        self.virtual_repeat_intersection_projections.clear();
        self.virtual_residual_projections.clear();
    }

    pub(crate) fn set_virtual_repeat_intersection_mask_projection(
        &mut self,
        tokenizer: Tokenizer,
        projection: VirtualBinaryRepeatIntersectionMaskProjection,
    ) {
        self.set_virtual_repeat_intersections_mask_projection(tokenizer, vec![projection]);
    }

    pub(crate) fn set_virtual_repeat_intersections_mask_projection(
        &mut self,
        tokenizer: Tokenizer,
        projections: Vec<VirtualBinaryRepeatIntersectionMaskProjection>,
    ) {
        debug_assert!(!projections.is_empty());
        self.mask_tokenizer_fast_transitions = Self::build_full_walk_fast_transitions(&tokenizer);
        self.mask_tokenizer = Some(Arc::new(tokenizer));
        self.mask_determinized_tokenizer = None;
        self.mask_projection_to_determinized = Arc::from(Vec::<u32>::new());
        self.full_to_mask_state = Arc::from(Vec::<u32>::new());
        self.mask_state_source_subsets = Arc::from(Vec::<Arc<[u32]>>::new());
        self.mask_source_subset_to_state = Arc::new(FxHashMap::default());
        self.virtual_unit_repeat_projection = None;
        self.virtual_repeat_intersection_projections = projections;
        self.virtual_residual_projections.clear();
    }

    pub(crate) fn set_virtual_residuals_mask_projection(
        &mut self,
        tokenizer: Tokenizer,
        projections: Vec<VirtualResidualMaskProjection>,
    ) {
        debug_assert!(!projections.is_empty());
        self.mask_tokenizer_fast_transitions = Self::build_full_walk_fast_transitions(&tokenizer);
        self.mask_tokenizer = Some(Arc::new(tokenizer));
        self.mask_determinized_tokenizer = None;
        self.mask_projection_to_determinized = Arc::from(Vec::<u32>::new());
        self.full_to_mask_state = Arc::from(Vec::<u32>::new());
        self.mask_state_source_subsets = Arc::from(Vec::<Arc<[u32]>>::new());
        self.mask_source_subset_to_state = Arc::new(FxHashMap::default());
        self.virtual_unit_repeat_projection = None;
        self.virtual_repeat_intersection_projections.clear();
        self.virtual_residual_projections = projections;
    }

    pub(crate) fn virtual_residual_mask_projection_parts(
        &self,
    ) -> Option<(&Tokenizer, &[VirtualResidualMaskProjection])> {
        if self.virtual_residual_projections.is_empty() {
            return None;
        }
        Some((
            self.mask_tokenizer.as_deref()?,
            self.virtual_residual_projections.as_slice(),
        ))
    }

    /// Preserve lexer-derived dynamic-mask metadata when a deferred vocabulary
    /// placeholder is replaced by its fully materialized runtime trie.
    ///
    /// These quotients/projections are constraint/lexer derived while the trie
    /// is vocabulary derived, so deferred dynamic compilation may construct
    /// them at different times. Sharing the immutable Arc-backed metadata
    /// avoids cloning it during that handoff.
    pub(crate) fn inherit_dynamic_lexer_metadata_from(&mut self, source: &Self) {
        self.mask_tokenizer = source.mask_tokenizer.clone();
        self.mask_determinized_tokenizer = source.mask_determinized_tokenizer.clone();
        self.mask_projection_to_determinized =
            Arc::clone(&source.mask_projection_to_determinized);
        self.mask_tokenizer_fast_transitions = source.mask_tokenizer_fast_transitions.clone();
        self.full_to_mask_state = Arc::clone(&source.full_to_mask_state);
        self.mask_state_source_subsets = Arc::clone(&source.mask_state_source_subsets);
        self.mask_source_subset_to_state = Arc::clone(&source.mask_source_subset_to_state);
        self.terminal_observation_classes = Arc::clone(&source.terminal_observation_classes);
        self.projected_terminal_quotients = Arc::clone(&source.projected_terminal_quotients);
        self.runtime_projected_terminal_quotients =
            Arc::clone(&source.runtime_projected_terminal_quotients);
        self.runtime_projected_terminal_quotient_cache =
            Arc::clone(&source.runtime_projected_terminal_quotient_cache);
        self.projected_terminal_quotients_prepared =
            source.projected_terminal_quotients_prepared;
        self.prepared_master_prover_row_ids =
            Arc::clone(&source.prepared_master_prover_row_ids);
        self.prepared_master_prover_offsets =
            Arc::clone(&source.prepared_master_prover_offsets);
        self.prepared_master_prover_terminals =
            Arc::clone(&source.prepared_master_prover_terminals);
        self.prepared_master_coverage_row_ids =
            Arc::clone(&source.prepared_master_coverage_row_ids);
        self.prepared_master_coverage_offsets =
            Arc::clone(&source.prepared_master_coverage_offsets);
        self.prepared_master_coverage_terminals =
            Arc::clone(&source.prepared_master_coverage_terminals);
        self.prepared_safe_plus_complete_terminals =
            Arc::clone(&source.prepared_safe_plus_complete_terminals);
        self.prepared_safe_radius_row_ids =
            Arc::clone(&source.prepared_safe_radius_row_ids);
        self.prepared_safe_radius_offsets =
            Arc::clone(&source.prepared_safe_radius_offsets);
        self.prepared_safe_radius_entries =
            Arc::clone(&source.prepared_safe_radius_entries);
        self.virtual_unit_repeat_projection = source.virtual_unit_repeat_projection;
        self.virtual_repeat_intersection_projections =
            source.virtual_repeat_intersection_projections.clone();
        self.virtual_residual_projections = source.virtual_residual_projections.clone();
    }

    pub(crate) fn mask_tokenizer_quotient_for_transfer(&self) -> Option<(Tokenizer, Vec<u32>)> {
        if self.virtual_unit_repeat_projection.is_some()
            || !self.virtual_repeat_intersection_projections.is_empty()
            || !self.virtual_residual_projections.is_empty()
        {
            // This compact structural projection is rebuilt from the exact
            // virtual tokenizer and bound vocabulary after load. The legacy
            // transfer tuple can only express a dense full-state vector.
            return None;
        }
        self.mask_tokenizer.as_ref().map(|tokenizer| {
            ((**tokenizer).clone(), self.full_to_mask_state.as_ref().to_vec())
        })
    }

    #[inline]
    pub(crate) fn mask_projection_tokenizer(&self) -> Option<&Tokenizer> {
        self.mask_tokenizer.as_deref()
    }

    /// Exact tokenizer coordinate used by dynamic mask generation. A finite
    /// serialized projection may have a second-stage deterministic derivative
    /// for the hot walk while static artifact tables remain keyed by the base
    /// projection above.
    #[inline]
    pub(crate) fn mask_runtime_tokenizer(&self) -> Option<&Tokenizer> {
        self.mask_determinized_tokenizer
            .as_deref()
            .or(self.mask_tokenizer.as_deref())
    }

    #[inline]
    pub(crate) fn has_dense_mask_tokenizer_projection(&self) -> bool {
        self.mask_tokenizer.is_some() && !self.full_to_mask_state.is_empty()
    }

    #[inline]
    pub(crate) fn mask_projection_fast_transitions(&self) -> Option<&FastTokenizerTransitions> {
        self.mask_tokenizer_fast_transitions.as_ref()
    }

    pub(crate) fn clear_mask_projection_fast_transitions(&mut self) {
        self.mask_tokenizer_fast_transitions = None;
    }

    #[cfg(test)]
    pub(crate) fn disable_prepared_mask_execution_for_test(&mut self) {
        self.mask_tokenizer = None;
        self.mask_determinized_tokenizer = None;
        self.mask_projection_to_determinized = Arc::from(Vec::<u32>::new());
        self.mask_tokenizer_fast_transitions = None;
        self.full_to_mask_state = Arc::from(Vec::<u32>::new());
        self.mask_state_source_subsets = Arc::from(Vec::<Arc<[u32]>>::new());
        self.mask_source_subset_to_state = Arc::new(FxHashMap::default());
    }

    #[inline]
    pub(crate) fn mask_projection_state(&self, full_state: u32) -> u32 {
        if !self.virtual_residual_projections.is_empty() {
            for projection in &self.virtual_residual_projections {
                if let Some(projected) = projection.project(full_state) {
                    return projected;
                }
            }
            // Ordinary physical states retain their IDs in the residual mask
            // tokenizer; only exact virtual states require an owning projection.
            if self
                .virtual_residual_projections
                .iter()
                .all(|projection| full_state < projection.physical_state_count())
            {
                return full_state;
            }
            panic!(
                "exact residual tokenizer state {full_state} has no owning finite-mask projection"
            );
        }
        if !self.virtual_repeat_intersection_projections.is_empty() {
            for projection in &self.virtual_repeat_intersection_projections {
                if let Some(projected) = projection.project(full_state) {
                    return projected;
                }
            }
            panic!(
                "exact virtual tokenizer state {full_state} has no owning finite-mask projection"
            );
        }
        if let Some(projection) = self.virtual_unit_repeat_projection {
            return projection.project(full_state).unwrap_or_else(|| {
                panic!(
                    "exact arithmetic tokenizer state {full_state} has no owning finite-mask projection"
                )
            });
        }
        self.full_to_mask_state
            .get(full_state as usize)
            .copied()
            .unwrap_or(full_state)
    }

    #[inline]
    pub(crate) fn mask_runtime_state(&self, full_state: u32) -> u32 {
        let projected = self.mask_projection_state(full_state);
        self.mask_projection_to_determinized
            .get(projected as usize)
            .copied()
            .unwrap_or(projected)
    }

    pub(crate) fn mask_projection_state_multiplicities(&self) -> Option<Vec<usize>> {
        let tokenizer = self.mask_tokenizer.as_ref()?;
        if !self.virtual_residual_projections.is_empty()
            || !self.virtual_repeat_intersection_projections.is_empty() {
            // The exact virtual state domain is populated lazily, so no finite
            // global full-state multiplicity table exists. Optimizations that
            // require such a table must simply decline.
            return None;
        }
        if let Some(projection) = self.virtual_unit_repeat_projection {
            let counts = projection.multiplicities();
            debug_assert_eq!(counts.len(), tokenizer.num_states() as usize);
            return Some(counts);
        }
        let mut counts = vec![0usize; tokenizer.num_states() as usize];
        for &state in self.full_to_mask_state.iter() {
            if let Some(count) = counts.get_mut(state as usize) {
                *count += 1;
            }
        }
        Some(counts)
    }

    /// Exact full-tokenizer preimage for quotient states that have exactly one
    /// runtime source.  Non-unique and unreachable quotient states are
    /// represented by `u32::MAX`.
    pub(crate) fn mask_projection_unique_full_states(&self) -> Option<Vec<u32>> {
        let tokenizer = self.mask_tokenizer.as_ref()?;
        if !self.virtual_repeat_intersection_projections.is_empty() {
            return None;
        }
        if let Some(projection) = self.virtual_unit_repeat_projection {
            let unique = projection.unique_full_states();
            debug_assert_eq!(unique.len(), tokenizer.num_states() as usize);
            return Some(unique);
        }
        let mut unique = vec![u32::MAX; tokenizer.num_states() as usize];
        let mut duplicate = vec![false; tokenizer.num_states() as usize];
        for (full_state, &mask_state) in self.full_to_mask_state.iter().enumerate() {
            let index = mask_state as usize;
            if index >= unique.len() {
                continue;
            }
            if unique[index] == u32::MAX && !duplicate[index] {
                unique[index] = full_state as u32;
            } else {
                duplicate[index] = true;
                unique[index] = u32::MAX;
            }
        }
        Some(unique)
    }

    /// Lookup by a state in the active mask-tokenizer coordinate. Callers that
    /// hold an exact committed tokenizer state must project it first.
    pub(crate) fn self_loop_projection(
        &self,
        source_state: u32,
    ) -> Option<&DynamicSelfLoopProjection> {
        let index = *self.projection_by_source.get(source_state as usize)?;
        if index == u32::MAX {
            return None;
        }
        self.self_loop_projections.get(index as usize)
    }

    /// H64 alias lookup in the active mask-tokenizer coordinate.
    pub(crate) fn self_loop_projection_alias_h64(
        &self,
        source_state: u32,
    ) -> Option<&DynamicSelfLoopProjection> {
        let index = *self.projection_alias_h64.get(source_state as usize)?;
        if index == u32::MAX {
            return None;
        }
        self.self_loop_projections.get(index as usize)
    }

    /// Vocabulary alias lookup in the active mask-tokenizer coordinate.
    pub(crate) fn self_loop_projection_alias_vocab(
        &self,
        source_state: u32,
    ) -> Option<&DynamicSelfLoopProjection> {
        let index = *self.projection_alias_vocab.get(source_state as usize)?;
        if index == u32::MAX {
            return None;
        }
        self.self_loop_projections.get(index as usize)
    }

    #[inline]
    pub(crate) fn has_self_loop_projections(&self) -> bool {
        !self.self_loop_projections.is_empty()
    }

    pub(crate) fn set_bounded_observation_sets(
        &mut self,
        sets: DynamicBoundedObservationSets,
    ) {
        self.bounded_observation_sets = Arc::new(sets);
    }

    #[inline]
    pub(crate) fn bounded_observation_safe_bytes(
        &self,
        source: u32,
        required_horizon: u32,
    ) -> Option<U8Set> {
        self.bounded_observation_sets
            .safe_bytes(source, required_horizon)
    }

    #[inline]
    pub(crate) fn bounded_observation_set_counts(&self) -> (usize, usize) {
        (
            self.bounded_observation_sets.state_count(),
            self.bounded_observation_sets.unique_set_count(),
        )
    }

    pub(crate) fn set_terminal_observation_classes(
        &mut self,
        mut classes: Vec<(TerminalID, Arc<[u32]>)>,
    ) {
        classes.sort_unstable_by_key(|(terminal, _)| *terminal);
        classes.dedup_by_key(|(terminal, _)| *terminal);
        self.terminal_observation_classes = Arc::from(classes);
    }

    #[inline]
    pub(crate) fn terminal_observation_class(
        &self,
        terminal: TerminalID,
        state: u32,
    ) -> Option<u32> {
        let index = self
            .terminal_observation_classes
            .binary_search_by_key(&terminal, |(candidate, _)| *candidate)
            .ok()?;
        self.terminal_observation_classes[index]
            .1
            .get(state as usize)
            .copied()
            .filter(|&class| class != 0)
    }

    #[inline]
    pub(crate) fn has_terminal_observation_classes(&self) -> bool {
        !self.terminal_observation_classes.is_empty()
    }

    /// Cheap necessary condition for the exact parser-relative observation
    /// certificate: at least one prepared terminal quotient places both lexer
    /// source states in the same nonzero class. This is not sufficient for
    /// parser-relative equivalence, but it cheaply avoids arming the expensive
    /// checkpoint for unrelated source pairs.
    #[inline]
    pub(crate) fn shares_terminal_observation_class(
        &self,
        left_state: u32,
        right_state: u32,
    ) -> bool {
        let left = left_state as usize;
        let right = right_state as usize;
        self.terminal_observation_classes.iter().any(|(_, classes)| {
            let Some(&left_class) = classes.get(left) else {
                return false;
            };
            left_class != 0 && classes.get(right).copied() == Some(left_class)
        })
    }

    /// Prove equality of every parser-admitted terminal observation that is
    /// live at either lexer source, using only the selectively prepared exact
    /// per-terminal quotient rows. Returns the number of relevant live
    /// terminals on success; missing quotient coverage is a conservative
    /// decline.
    ///
    /// Iterate the tiny prepared-class sidecar (normally one or two terminals)
    /// rather than all parser-admitted terminals. The live-count pass is word
    /// based, so the common <=64-terminal grammar costs a handful of integer
    /// operations instead of repeated bitset probes and binary searches.
    #[inline]
    pub(crate) fn terminal_observation_equivalent_for_live_admitted(
        &self,
        left_state: u32,
        right_state: u32,
        admitted: &BitSet,
        left_matched: &BitSet,
        left_future: &BitSet,
        right_matched: &BitSet,
        right_future: &BitSet,
    ) -> Option<usize> {
        if self.terminal_observation_classes.is_empty() {
            return None;
        }

        let mut live_count = 0usize;
        for (word_index, &admitted_word) in admitted.words().iter().enumerate() {
            let word = |set: &BitSet| set.words().get(word_index).copied().unwrap_or(0);
            let live = admitted_word
                & (word(left_matched)
                    | word(left_future)
                    | word(right_matched)
                    | word(right_future));
            live_count += live.count_ones() as usize;
        }
        if live_count == 0 {
            return None;
        }

        let mut covered = 0usize;
        for (terminal, classes) in self.terminal_observation_classes.iter() {
            let terminal = *terminal as usize;
            if !admitted.contains(terminal)
                || !(left_matched.contains(terminal)
                    || left_future.contains(terminal)
                    || right_matched.contains(terminal)
                    || right_future.contains(terminal))
            {
                continue;
            }
            covered += 1;
            let left = classes.get(left_state as usize).copied().unwrap_or(0);
            let right = classes.get(right_state as usize).copied().unwrap_or(0);
            if left == 0 || left != right {
                return None;
            }
        }
        (covered == live_count).then_some(live_count)
    }

    pub(crate) fn terminal_observation_classes_cloned(
        &self,
    ) -> Vec<(TerminalID, Arc<[u32]>)> {
        self.terminal_observation_classes
            .iter()
            .map(|(terminal, classes)| (*terminal, Arc::clone(classes)))
            .collect()
    }

    pub(crate) fn terminal_observation_classes_for_artifact(
        &self,
    ) -> Vec<(TerminalID, Vec<u32>)> {
        self.terminal_observation_classes
            .iter()
            .map(|(terminal, classes)| (*terminal, classes.as_ref().to_vec()))
            .collect()
    }

    pub(crate) fn set_projected_terminal_quotients(
        &mut self,
        mut quotients: Vec<(TerminalID, TerminalProjectedQuotient)>,
    ) {
        quotients.sort_unstable_by_key(|(terminal, _)| *terminal);
        quotients.dedup_by_key(|(terminal, _)| *terminal);
        self.projected_terminal_quotients = Arc::from(
            quotients
                .into_iter()
                .map(|(terminal, quotient)| (terminal, Arc::new(quotient)))
                .collect::<Vec<_>>(),
        );
        self.runtime_projected_terminal_quotients = Arc::new(OnceLock::new());
        self.runtime_projected_terminal_quotient_cache =
            Arc::new(Mutex::new(FxHashMap::default()));
        self.projected_terminal_quotients_prepared = true;
        self.projected_terminal_text_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
        self.projected_terminal_partition_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
        self.projected_terminal_radius_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
    }

    pub(crate) fn projected_terminal_quotients_for_artifact(
        &self,
    ) -> Vec<(TerminalID, TerminalProjectedQuotient)> {
        self.projected_terminal_quotients
            .iter()
            .map(|(terminal, quotient)| (*terminal, quotient.as_ref().clone()))
            .collect()
    }

    #[inline]
    pub(crate) fn projected_terminal_quotient(
        &self,
        terminal: TerminalID,
        source: u32,
    ) -> Option<Arc<TerminalProjectedQuotient>> {
        let quotients = self.active_projected_terminal_quotients();
        if let Ok(index) = quotients
            .binary_search_by_key(&terminal, |(candidate, _)| *candidate)
        {
            let quotient = Arc::clone(&quotients[index].1);
            return quotient.contains_source(source).then_some(quotient);
        }
        if self.projected_terminal_quotients_prepared
            || self.runtime_projected_terminal_quotients.get().is_some()
        {
            return None;
        }
        let quotient = self
            .runtime_projected_terminal_quotient_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&terminal)
            .cloned()
            .flatten()?;
        quotient.contains_source(source).then_some(quotient)
    }

    /// Prepare exactly one runtime containment quotient. This is the hot-path
    /// counterpart to `prepare_runtime_projected_terminal_quotients`: callers
    /// that already know the parser/lexer-admitted terminal should not pay to
    /// materialize every terminal whose byte support happens to cover the
    /// generic safe-string alphabet.
    pub(crate) fn prepare_runtime_projected_terminal_quotient(
        &self,
        source: &Tokenizer,
        terminal: TerminalID,
    ) {
        if self.projected_terminal_quotients_prepared
            || self.runtime_projected_terminal_quotients.get().is_some()
        {
            return;
        }
        {
            let cache = self
                .runtime_projected_terminal_quotient_cache
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if cache.contains_key(&terminal) {
                return;
            }
        }

        let state_cap = std::env::var("GLRMASK_EXPERIMENT_DEMAND_PROOF_STATE_CAP")
            .ok().and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(2048);
        let built = if std::env::var_os("GLRMASK_DISABLE_DEMAND_PROOF_CAP").is_none() {
            let edge_cap = std::env::var("GLRMASK_EXPERIMENT_DEMAND_PROOF_EDGE_CAP")
                .ok().and_then(|value| value.parse::<usize>().ok())
                .unwrap_or_else(|| state_cap.saturating_mul(128));
            let quotient = source.build_terminal_projected_quotient_for_containment_bounded(
                terminal, state_cap, edge_cap,
            );
            if std::env::var_os("GLRMASK_DIAG_DEMAND_PROOF_CAP").is_some() {
                eprintln!("[demand_proof_cap] terminal={} state_cap={} edge_cap={} built={}",
                    terminal, state_cap, edge_cap, quotient.is_some());
            }
            quotient.map(Arc::new)
        } else {
            source.build_terminal_projected_quotients_for_containment_candidates(&[terminal])
                .into_iter().find_map(|(candidate, quotient)| {
                    (candidate == terminal).then(|| Arc::new(quotient))
                })
        };
        self.runtime_projected_terminal_quotient_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .entry(terminal)
            .or_insert(built);
    }

    pub(crate) fn prepare_runtime_projected_terminal_quotients(
        &self,
        source: &Tokenizer,
        safe_slice_bytes: &U8Set,
        component_state_cap: Option<usize>,
    ) {
        if self.projected_terminal_quotients_prepared
            || self.runtime_projected_terminal_quotients.get().is_some()
        {
            return;
        }
        let _ = self.runtime_projected_terminal_quotients.get_or_init(|| {
            let candidates = (0..source.num_terminals())
                .filter(|&terminal| {
                    source
                        .terminal_byte_support(terminal)
                        .is_some_and(|support| safe_slice_bytes.is_subset(&support))
                })
                .collect::<Vec<_>>();
            let mut quotients = match component_state_cap {
                Some(cap) => source.build_terminal_projected_quotients_for_containment_candidates_bounded(
                    &candidates, cap, cap.saturating_mul(256),
                ),
                None => source.build_terminal_projected_quotients_for_containment_candidates(&candidates),
            };
            quotients.sort_unstable_by_key(|(terminal, _)| *terminal);
            quotients.dedup_by_key(|(terminal, _)| *terminal);
            Arc::from(
                quotients
                    .into_iter()
                    .map(|(terminal, quotient)| (terminal, Arc::new(quotient)))
                    .collect::<Vec<_>>(),
            )
        });
    }

    #[inline]
    pub(crate) fn has_projected_terminal_quotients(&self) -> bool {
        if self.projected_terminal_quotients_prepared {
            !self.projected_terminal_quotients.is_empty()
        } else {
            self.runtime_projected_terminal_quotients
                .get()
                .is_some_and(|quotients| !quotients.is_empty())
                || self
                    .runtime_projected_terminal_quotient_cache
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .values()
                    .any(Option::is_some)
        }
    }

    #[inline]
    pub(crate) fn projected_terminal_quotients_prepared(&self) -> bool {
        self.projected_terminal_quotients_prepared
    }

    pub(crate) fn projected_terminal_text_liveness(
        &self,
        terminal: TerminalID,
        source: u32,
        alphabet: U8Set,
        include_utf8_scalars: bool,
    ) -> Option<bool> {
        let quotient = self.projected_terminal_quotient(terminal, source)?;
        let key = (terminal, source, alphabet, include_utf8_scalars);
        if let Some(&cached) = self
            .projected_terminal_text_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&key)
        {
            return Some(cached);
        }
        let certified = quotient
            .text_liveness_closed_bounded(
                source,
                alphabet,
                1_024,
                200_000,
                include_utf8_scalars,
            )
            .unwrap_or(false);
        self.projected_terminal_text_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(key, certified);
        Some(certified)
    }
    pub(crate) fn projected_terminal_slice_contained(
        &self,
        terminal: TerminalID,
        source: u32,
        slice_cache_id: u32,
        slice_dfa: &VocabPartitionDfa,
    ) -> Option<bool> {
        let key = (terminal, source, 0x8000_0000u32 | slice_cache_id);
        if let Some(&cached) = self
            .projected_terminal_partition_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&key)
        {
            return Some(cached);
        }
        let quotient = self.projected_terminal_quotient(terminal, source)?;
        let certified = Self::terminal_partition_product_is_transparent(
            quotient.as_ref(), slice_dfa, source, 200_000,
        )
        .unwrap_or(false);
        self.projected_terminal_partition_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(key, certified);
        Some(certified)
    }

    /// Prove a bounded safe-atom prefix radius directly in the physical lexer.
    /// Unlike the projected proof this explores only the requested horizon,
    /// without first building/minimizing the terminal's entire residual DFA.
    /// The slice must mark each completed atom by entering an accepting state,
    /// as the safe+ UTF-8 slice does. The shortest failing accepted word gives
    /// a universal safe lower bound for all strictly shorter atom layers.
    /// Every traversed prefix must retain the selected terminal. Epsilon or
    /// nonphysical coordinates and exhausted work budgets fail closed.
    pub(crate) fn raw_terminal_slice_repeat_radius(
        &self,
        tokenizer: &Tokenizer,
        terminal: TerminalID,
        source: u32,
        slice_cache_id: u32,
        slice: &VocabPartitionDfa,
        max_repetitions: u32,
        work_limit: usize,
    ) -> Option<u32> {
        if max_repetitions == 0
            || source >= tokenizer.num_states()
            || tokenizer.state_is_virtual_runtime(source)
            || tokenizer.state_has_epsilon_transitions(source)
            || slice.accepting_map().get(slice.start_state() as usize).copied()?
        {
            return None;
        }
        // A dedicated cache cannot alias projected/symbolic certificate keys.
        // Slice IDs refer to immutable languages within this vocabulary instance.
        // Include budget so an earlier cheap failed attempt cannot mask a later
        // larger-budget certificate. Cached zero is always conservative.
        let key = (terminal, source, slice_cache_id, max_repetitions, work_limit);
        if let Some(&cached) = self.raw_terminal_radius_cache.lock()
            .unwrap_or_else(|p| p.into_inner()).get(&key)
        {
            return Some(cached);
        }
        static PROFILE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let diagnostic = *PROFILE.get_or_init(|| {
            std::env::var_os("GLRMASK_PROFILE_RAW_TERMINAL_RADIUS").is_some()
        });
        static BATCH: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        static VERIFY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let batch_rows = *BATCH.get_or_init(|| {
            std::env::var_os("GLRMASK_DISABLE_RAW_RADIUS_BATCH_ROWS").is_none()
        });
        let verify_rows = *VERIFY.get_or_init(|| {
            std::env::var_os("GLRMASK_ASSERT_RAW_RADIUS_BATCH_ROWS").is_some()
        });
        let started = diagnostic.then(std::time::Instant::now);
        let mut work = 0usize;
        let mut explored = 0usize;
        let result = (|| {
            let slice_count = slice.accepting_map().len();
            let mut reverse = vec![Vec::<(u32, u8)>::new(); slice_count];
            for state in 0..slice_count as u32 {
                let mut seen = FxHashSet::default();
                for byte in 0u16..=255 {
                    let target = slice.step(state, byte as u8);
                    if (target as usize) < slice_count && seen.insert(target) {
                        reverse[target as usize].push((state, u8::from(slice.accepting_map()[target as usize])));
                    }
                }
            }
            let mut min_to_accept = vec![u32::MAX; slice_count];
            let mut distances = VecDeque::new();
            for (state, &accepting) in slice.accepting_map().iter().enumerate() {
                if accepting { min_to_accept[state] = 0; distances.push_back(state as u32); }
            }
            while let Some(target) = distances.pop_front() {
                for &(state, cost) in &reverse[target as usize] {
                    let next = min_to_accept[target as usize].saturating_add(u32::from(cost));
                    if next < min_to_accept[state as usize] {
                        min_to_accept[state as usize] = next;
                        if cost == 0 { distances.push_front(state); } else { distances.push_back(state); }
                    }
                }
            }
            let mut best = FxHashMap::<(u32, u32), u32>::default();
            let mut queue = VecDeque::from([(slice.start_state(), source, 0u32)]);
            best.insert((slice.start_state(), source), 0);
            let mut first_bad = max_repetitions.saturating_add(1);
            while let Some((slice_state, lexer_state, completed)) = queue.pop_front() {
                if best.get(&(slice_state, lexer_state)).copied() != Some(completed)
                    || completed >= first_bad || completed > max_repetitions
                { continue; }
                if lexer_state >= tokenizer.num_states()
                    || tokenizer.state_is_virtual_runtime(lexer_state)
                    || tokenizer.state_has_epsilon_transitions(lexer_state)
                { return None; }
                explored += 1;
                // The guarded physical, epsilon-free row is exactly step().
                // Materialize once per product node; no row survives the proof.
                let mut row = [None; 256];
                if batch_rows {
                    for (byte, target) in tokenizer.transitions_from(lexer_state) {
                        row[byte as usize] = Some(target);
                    }
                    if verify_rows {
                        for byte in 0u16..=255 {
                            assert_eq!(row[byte as usize], tokenizer.step(lexer_state, byte as u8));
                        }
                    }
                }
                let mut previous_edge = None;
                for byte in 0u16..=255 {
                    let target_slice = slice.step(slice_state, byte as u8);
                    if target_slice as usize >= slice_count || !slice.can_reach_accepting(target_slice) { continue; }
                    let next_completed = completed.saturating_add(u32::from(slice.accepting_map()[target_slice as usize]));
                    let rest = min_to_accept[target_slice as usize];
                    if rest == u32::MAX { continue; }
                    let shortest_word = next_completed.saturating_add(rest);
                    if shortest_word > max_repetitions || shortest_word >= first_bad { continue; }
                    work += 1;
                    if work > work_limit { return None; }
                    let target = if batch_rows {
                        row[byte as usize]
                    } else {
                        tokenizer.step(lexer_state, byte as u8)
                    };
                    // Equal product targets have identical atom costs, liveness
                    // and successors. The original eligible-byte budget has
                    // ALREADY been charged above, including duplicate edges.
                    if batch_rows {
                        let edge = (target_slice, target);
                        if previous_edge == Some(edge) { continue; }
                        previous_edge = Some(edge);
                    }
                    if target.is_some_and(|t| tokenizer.state_is_virtual_runtime(t)) {
                        return None;
                    }
                    let live = target.is_some_and(|t| t < tokenizer.num_states()
                        && (tokenizer.possible_future_terminals(t).contains(terminal as usize)
                            || tokenizer.matched_terminals_slice(t).contains(&terminal)));
                    if !live {
                        first_bad = first_bad.min(shortest_word);
                        continue;
                    }
                    let target = target.expect("live target exists");
                    if tokenizer.state_has_epsilon_transitions(target) { return None; }
                    if next_completed >= first_bad { continue; }
                    let state_key = (target_slice, target);
                    if next_completed < best.get(&state_key).copied().unwrap_or(u32::MAX) {
                        best.insert(state_key, next_completed);
                        if slice.accepting_map()[target_slice as usize] {
                            queue.push_back((target_slice, target, next_completed));
                        } else { queue.push_front((target_slice, target, next_completed)); }
                    }
                }
            }
            Some(first_bad.saturating_sub(1).min(max_repetitions))
        })();
        // Zero is always a sound lower bound. Remember failures so an expensive
        // source cannot repeatedly spend the same bounded proof budget.
        let radius = result.unwrap_or(0);
        self.raw_terminal_radius_cache.lock().unwrap_or_else(|p| p.into_inner()).insert(key, radius);
        if diagnostic {
            eprintln!("[raw_terminal_radius] source={} terminal={} radius={} complete={} work={} states={} max={} ms={:.3}",
                source, terminal, radius, result.is_some(), work, explored, max_repetitions,
                started.map_or(0.0, |s| s.elapsed().as_secs_f64()*1000.0));
        }
        Some(radius)
    }


    /// Return the largest completed-atom count `r <= max_repetitions` such
    /// that every word in the regular `slice+` language with at most `r`
    /// completed atoms remains a live prefix of this exact projected terminal.
    ///
    /// The slice DFA must mark completion of one atom by entering an accepting
    /// state (the safe+ UTF-8 DFA has exactly this property). This is the finite
    /// quotient analogue of the symbolic residual repeat-radius proof: find the
    /// shortest slice word that reaches a dead/missing quotient transition, and
    /// admit every strictly shorter completed-atom layer.
    pub(crate) fn projected_terminal_slice_repeat_radius(
        &self,
        terminal: TerminalID,
        source: u32,
        slice_cache_id: u32,
        slice: &VocabPartitionDfa,
        max_repetitions: u32,
        work_limit: usize,
    ) -> Option<u32> {
        if max_repetitions == 0 || slice.accepting_map().get(slice.start_state() as usize).copied()? {
            return None;
        }
        let key = (terminal, source, slice_cache_id, max_repetitions);
        if let Some(&cached) = self
            .projected_terminal_radius_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&key)
        {
            return Some(cached);
        }
        let quotient = self.projected_terminal_quotient(terminal, source)?;
        let quotient_start = quotient.projected_state_for_source(source)?;
        if !quotient.projected_state_is_accepting(quotient_start)
            && !quotient.projected_state_has_future(quotient_start)
        {
            return Some(0);
        }

        let slice_state_count = slice.accepting_map().len();
        if slice_state_count == 0 {
            return None;
        }

        // Exact representatives for the common refinement of the slice and
        // quotient byte partitions.
        let mut representatives = Vec::<u8>::new();
        let mut seen_classes = FxHashSet::<(u8, u8)>::default();
        for byte in 0u16..=255 {
            let byte = byte as u8;
            let pair = (slice.byte_class(byte), quotient.projected_byte_class(byte));
            if seen_classes.insert(pair) {
                representatives.push(byte);
            }
        }

        // Minimum additional completed atoms needed to reach slice acceptance
        // from every slice state. Entering an accepting state completes one
        // atom; UTF-8 continuation states therefore have zero-cost edges.
        let mut reverse = vec![Vec::<(u32, u8)>::new(); slice_state_count];
        for source_state in 0..slice_state_count as u32 {
            let mut seen_targets = FxHashSet::<u32>::default();
            for &byte in &representatives {
                let target = slice.step(source_state, byte);
                if target as usize >= slice_state_count || !seen_targets.insert(target) {
                    continue;
                }
                reverse[target as usize].push((
                    source_state,
                    u8::from(slice.accepting_map()[target as usize]),
                ));
            }
        }
        let mut min_to_accept = vec![u32::MAX; slice_state_count];
        let mut distance_queue = VecDeque::<u32>::new();
        for (state, &accepting) in slice.accepting_map().iter().enumerate() {
            if accepting {
                min_to_accept[state] = 0;
                distance_queue.push_back(state as u32);
            }
        }
        while let Some(target) = distance_queue.pop_front() {
            let target_distance = min_to_accept[target as usize];
            for &(source_state, cost) in &reverse[target as usize] {
                let candidate = target_distance.saturating_add(u32::from(cost));
                if candidate < min_to_accept[source_state as usize] {
                    min_to_accept[source_state as usize] = candidate;
                    if cost == 0 {
                        distance_queue.push_front(source_state);
                    } else {
                        distance_queue.push_back(source_state);
                    }
                }
            }
        }

        let mut best = FxHashMap::<(u32, u32), u32>::default();
        let mut queue = VecDeque::<(u32, u32, u32)>::new();
        let slice_start = slice.start_state();
        best.insert((slice_start, quotient_start), 0);
        queue.push_back((slice_start, quotient_start, 0));
        let mut work = 0usize;
        let mut first_counterexample = max_repetitions.saturating_add(1);

        while let Some((slice_state, quotient_state, completed)) = queue.pop_front() {
            if best.get(&(slice_state, quotient_state)).copied() != Some(completed)
                || completed >= first_counterexample
                || completed > max_repetitions
            {
                continue;
            }
            for &byte in &representatives {
                let slice_target = slice.step(slice_state, byte);
                if slice_target as usize >= slice_state_count
                    || !slice.can_reach_accepting(slice_target)
                {
                    continue;
                }
                let completed_target = completed.saturating_add(u32::from(
                    slice.accepting_map()[slice_target as usize],
                ));
                let completion_cost = min_to_accept[slice_target as usize];
                if completion_cost == u32::MAX {
                    continue;
                }
                let shortest_complete_word = completed_target.saturating_add(completion_cost);
                if shortest_complete_word > max_repetitions {
                    continue;
                }
                work = work.saturating_add(1);
                if work > work_limit {
                    return None;
                }
                let quotient_class = quotient.projected_byte_class(byte);
                let quotient_target = quotient.projected_step_class(quotient_state, quotient_class);
                let target_live = quotient_target.is_some_and(|target| {
                    quotient.projected_state_is_accepting(target)
                        || quotient.projected_state_has_future(target)
                });
                if !target_live {
                    first_counterexample = first_counterexample.min(shortest_complete_word);
                    continue;
                }
                let quotient_target = quotient_target.expect("live quotient target must exist");
                if completed_target >= first_counterexample
                    || completed_target > max_repetitions
                {
                    continue;
                }
                let state_key = (slice_target, quotient_target);
                if completed_target < best.get(&state_key).copied().unwrap_or(u32::MAX) {
                    best.insert(state_key, completed_target);
                    if slice.accepting_map()[slice_target as usize] {
                        queue.push_back((slice_target, quotient_target, completed_target));
                    } else {
                        queue.push_front((slice_target, quotient_target, completed_target));
                    }
                }
            }
        }

        let radius = first_counterexample.saturating_sub(1).min(max_repetitions);
        self.projected_terminal_radius_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(key, radius);
        Some(radius)
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

    fn terminal_partition_product_is_transparent(
        quotient: &TerminalProjectedQuotient,
        partition: &VocabPartitionDfa,
        source: u32,
        work_limit: usize,
    ) -> Option<bool> {
        let quotient_start = quotient.projected_state_for_source(source)?;
        if !quotient.projected_state_is_accepting(quotient_start)
            && !quotient.projected_state_has_future(quotient_start)
        {
            return Some(false);
        }
        let partition_start = partition.start_state();
        if !partition.can_reach_accepting(partition_start) {
            return Some(true);
        }

        // Refine the two exact global byte partitions. A byte representative is
        // interchangeable only when both automata put it in the same class.
        let mut representatives = Vec::<(u8, u8, u8)>::new();
        for byte in 0u16..=255 {
            let byte = byte as u8;
            let partition_class = partition.byte_class(byte);
            let quotient_class = quotient.projected_byte_class(byte);
            if !representatives.iter().any(|&(p, q, _)| {
                p == partition_class && q == quotient_class
            }) {
                representatives.push((partition_class, quotient_class, byte));
            }
        }

        let mut seen = FxHashSet::<(u32, u32)>::default();
        let mut queue = VecDeque::from([(partition_start, quotient_start)]);
        let mut work = 0usize;
        while let Some((partition_state, quotient_state)) = queue.pop_front() {
            if !seen.insert((partition_state, quotient_state)) {
                continue;
            }
            for &(_, quotient_class, byte) in &representatives {
                work = work.saturating_add(1);
                if work > work_limit {
                    return None;
                }
                let partition_target = partition.step(partition_state, byte);
                if !partition.can_reach_accepting(partition_target) {
                    continue;
                }
                let Some(quotient_target) =
                    quotient.projected_step_class(quotient_state, quotient_class)
                else {
                    return Some(false);
                };
                if !quotient.projected_state_is_accepting(quotient_target)
                    && !quotient.projected_state_has_future(quotient_target)
                {
                    return Some(false);
                }
                if !seen.contains(&(partition_target, quotient_target)) {
                    queue.push_back((partition_target, quotient_target));
                }
            }
        }
        Some(true)
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

    pub(crate) fn lock_lazy_union_cache(
        &self,
    ) -> std::sync::MutexGuard<'_, DynamicLazyUnionCache> {
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
        let cache = self.dense_subset16_cache.lock().unwrap_or_else(|p| p.into_inner());
        for extension in cache.values() {
            if let Some(index) = extension.subsets.iter().position(|subset| subset.as_slice() == root_states) {
                return Some((Arc::clone(extension), extension.base_state_count + index as u32));
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

    fn copy_dynamic_mask_cache_payload(
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

    fn dynamic_mask_cache_payload(&self, mask: &[u32]) -> DynamicMaskCachePayload {
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
                    .filter_map(|(index, &word)| {
                        (word != 0).then_some((index as u32, word))
                    })
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
    fn to_artifact_impl(&self, include_mask_quotient: bool) -> Option<DynamicMaskVocabArtifact> {
        if !self.initialized || self.pending_source.is_some() {
            return None;
        }
        let mask_quotient = include_mask_quotient
            .then(|| self.mask_tokenizer_quotient_for_transfer())
            .flatten();
        Some(DynamicMaskVocabArtifact {
            trie: self.trie.as_ref().clone(),
            token_aliases: self.token_aliases.clone(),
            llg_slice_leftovers: self.llg_slice_leftovers.as_ref().clone(),
            mask_tokenizer: mask_quotient.as_ref().map(|(tokenizer, _)| tokenizer.clone()),
            full_to_mask_state: mask_quotient
                .map(|(_, full_to_mask_state)| full_to_mask_state)
                .unwrap_or_default(),
            grammar_quotiented: self.grammar_quotiented,
            source_vocab_digest: self.source_vocab_digest,
        })
    }

    pub(crate) fn to_artifact(&self) -> Option<DynamicMaskVocabArtifact> {
        self.to_artifact_impl(true)
    }

    /// Preserve the exact O2 execution coordinate while omitting accelerators
    /// determined solely by the caller's separately supplied model vocabulary.
    pub(crate) fn to_template_external_vocab_artifact(&self) -> Option<DynamicMaskVocabArtifact> {
        let mut artifact = self.to_artifact_impl(true)?;
        if artifact.grammar_quotiented {
            artifact.llg_slice_leftovers.clear();
        }
        Some(artifact)
    }

    /// Serialize only vocabulary-derived runtime data. Constraint-specific
    /// mask-tokenizer quotients and projections are reconstructed after load.
    pub(crate) fn to_vocab_artifact(&self) -> Option<DynamicMaskVocabArtifact> {
        self.to_artifact_impl(false)
    }

    /// External-vocabulary transfer can omit model-vocabulary-only slice/master
    /// acceleration for grammar quotients. `load_with_vocab` reattaches the
    /// already-prepared full-vocabulary structure in the parent process.
    pub(crate) fn to_external_vocab_artifact(&self) -> Option<DynamicMaskVocabArtifact> {
        let mut artifact = self.to_artifact_impl(false)?;
        if artifact.grammar_quotiented {
            artifact.llg_slice_leftovers.clear();
        }
        Some(artifact)
    }

    pub(crate) fn from_artifact(artifact: DynamicMaskVocabArtifact) -> Result<Self, String> {
        Self::from_artifact_impl(artifact, None)
    }

    pub(crate) fn from_external_vocab_artifact(
        artifact: DynamicMaskVocabArtifact,
        full_vocab_template: &DynamicMaskVocab,
    ) -> Result<Self, String> {
        Self::from_artifact_impl(artifact, Some(full_vocab_template))
    }

    fn from_artifact_impl(
        artifact: DynamicMaskVocabArtifact,
        full_vocab_template: Option<&DynamicMaskVocab>,
    ) -> Result<Self, String> {
        if artifact.trie.nodes.is_empty() {
            return Err("dynamic-mask vocabulary artifact has no trie root".to_owned());
        }
        let llg_slice_leftovers = artifact.llg_slice_leftovers;
        let external_acceleration = artifact.grammar_quotiented
            && llg_slice_leftovers.is_empty()
            && full_vocab_template.is_some();
        let mut result = if external_acceleration {
            DynamicMaskVocab::from_alias_store(
                Arc::new(artifact.trie),
                artifact.token_aliases,
            )
        } else {
            DynamicMaskVocab::from_alias_store_with_slices(
                Arc::new(artifact.trie),
                artifact.token_aliases,
                Some(&llg_slice_leftovers),
            )
        };
        if external_acceleration {
            result.inherit_llg_vocab_acceleration_without_master_from(
                full_vocab_template.expect("external acceleration template checked above"),
            );
        } else {
            result.llg_slice_leftovers = Arc::new(llg_slice_leftovers);
            if result.llg_master_admitted_words.is_empty() {
                result.rebuild_llg_master_admitted_words_from_master_trie();
            } else {
                result.rebuild_llg_master_residual_ops_from_master_trie();
            }
        }
        result.grammar_quotiented = artifact.grammar_quotiented;
        result.source_vocab_digest = artifact.source_vocab_digest;
        match artifact.mask_tokenizer {
            Some(tokenizer) => {
                if artifact.full_to_mask_state.is_empty()
                    || artifact
                        .full_to_mask_state
                        .iter()
                        .any(|&state| state >= tokenizer.num_states())
                {
                    return Err(
                        "dynamic-mask vocabulary artifact has an invalid mask-tokenizer quotient"
                            .to_owned(),
                    );
                }
                result.set_mask_tokenizer_quotient(tokenizer, artifact.full_to_mask_state);
            }
            None if !artifact.full_to_mask_state.is_empty() => {
                return Err(
                    "dynamic-mask vocabulary artifact has a quotient map without a tokenizer"
                        .to_owned(),
                );
            }
            None => {}
        }
        Ok(result)
    }
}

impl DynamicMaskVocab {
    pub(crate) fn restore_root_layout_metadata_from_token_bytes(
        &mut self,
        token_bytes: &BTreeMap<u32, Vec<u8>>,
    ) {
        let mut canonical_meta = Vec::<Option<(u16, bool)>>::new();
        canonical_meta.reserve(self.canonical_token_count());
        for canonical in 0..self.canonical_token_count() as u32 {
            let Some(originals) = self.token_ids(canonical) else {
                return;
            };
            let Some(first_original) = originals.first() else {
                return;
            };
            let Some(bytes) = token_bytes.get(first_original).map(Vec::as_slice) else {
                return;
            };
            canonical_meta.push((!bytes.is_empty()).then(|| {
                (
                    dynamic_mask_vocab_layout_class(classify_vocab_char_type(bytes), bytes),
                    std::str::from_utf8(bytes).is_ok(),
                )
            }));
        }

        let mut root_layout_classes = Vec::with_capacity(self.trie.children(0).len());
        let mut root_layout_all_valid_utf8 = Vec::with_capacity(self.trie.children(0).len());
        for edge in self.trie.children(0) {
            let mut class = None::<u16>;
            let mut all_valid_utf8 = true;
            let mut saw_token = false;
            for &canonical in self.trie.subtree_tokens(edge.child) {
                let Some(Some((token_class, valid_utf8))) = canonical_meta.get(canonical as usize)
                else {
                    return;
                };
                if class.is_some_and(|existing| existing != *token_class) {
                    return;
                }
                class = Some(*token_class);
                all_valid_utf8 &= *valid_utf8;
                saw_token = true;
            }
            let Some(class) = class.filter(|_| saw_token) else {
                return;
            };
            root_layout_classes.push(class);
            root_layout_all_valid_utf8.push(all_valid_utf8);
        }

        let trie = Arc::make_mut(&mut self.trie);
        trie.root_layout_classes = root_layout_classes;
        trie.root_layout_all_valid_utf8 = root_layout_all_valid_utf8;
    }

    /// Verify that this vocabulary-only runtime index represents exactly the
    /// supplied original token-id -> byte mapping. This is used when loading a
    /// self-contained dynamic artifact: the persisted trie is an accelerator,
    /// never an independent source of vocabulary semantics.
    pub(crate) fn matches_token_bytes_exact(&self, token_bytes: &BTreeMap<u32, Vec<u8>>) -> bool {
        if self.grammar_quotiented {
            return self.matches_grammar_quotiented_token_bytes(token_bytes);
        }
        if self.canonical_original_tokens.len() != token_bytes.len() {
            return false;
        }

        // Ordinary dynamic vocabularies canonicalize only byte-identical model
        // tokens. Keep the historical strong validation for those artifacts.
        let mut sorted_tokens = token_bytes
            .iter()
            .map(|(&token_id, bytes)| (token_id, bytes.as_slice()))
            .collect::<Vec<_>>();
        let sort_tokens = |left: &(u32, &[u8]), right: &(u32, &[u8])| {
            left.1.cmp(right.1).then_with(|| left.0.cmp(&right.0))
        };
        if rayon::current_num_threads() == 1 {
            sorted_tokens.sort_unstable_by(sort_tokens);
        } else {
            sorted_tokens.par_sort_unstable_by(sort_tokens);
        }
        let mut canonical_bytes = Vec::<&[u8]>::with_capacity(self.canonical_token_count());
        let mut start = 0usize;
        while start < sorted_tokens.len() {
            let bytes = sorted_tokens[start].1;
            let mut end = start + 1;
            while end < sorted_tokens.len() && sorted_tokens[end].1 == bytes {
                end += 1;
            }
            let canonical = canonical_bytes.len() as u32;
            let Some(originals) = self.token_ids(canonical) else {
                return false;
            };
            if originals.len() != end - start
                || !originals
                    .iter()
                    .copied()
                    .eq(sorted_tokens[start..end].iter().map(|(token_id, _)| *token_id))
            {
                return false;
            }
            canonical_bytes.push(bytes);
            start = end;
        }
        if canonical_bytes.len() != self.canonical_token_count() {
            return false;
        }

        struct Frame {
            node: u32,
            next_child: usize,
            prefix_len: usize,
        }

        let mut canonical_seen = vec![false; canonical_bytes.len()];
        let mut prefix = Vec::<u8>::new();
        let mut frames = vec![Frame {
            node: 0,
            next_child: 0,
            prefix_len: 0,
        }];
        while !frames.is_empty() {
            let frame_index = frames.len() - 1;
            let node_id = frames[frame_index].node;
            if frames[frame_index].next_child == 0 {
                if let Some(canonical) = self.trie.node(node_id).token_id {
                    let canonical = canonical as usize;
                    if canonical >= canonical_seen.len()
                        || canonical_seen[canonical]
                        || canonical_bytes[canonical] != prefix.as_slice()
                    {
                        return false;
                    }
                    canonical_seen[canonical] = true;
                }
            }

            let children = self.trie.children(node_id);
            if frames[frame_index].next_child < children.len() {
                let edge = children[frames[frame_index].next_child].clone();
                frames[frame_index].next_child += 1;
                let prefix_len = prefix.len();
                prefix.extend_from_slice(self.trie.edge_bytes(&edge));
                frames.push(Frame {
                    node: edge.child,
                    next_child: 0,
                    prefix_len,
                });
            } else {
                let prefix_len = frames[frame_index].prefix_len;
                frames.pop();
                prefix.truncate(prefix_len);
            }
        }

        canonical_seen.into_iter().all(|seen| seen)
    }

    fn matches_grammar_quotiented_token_bytes(
        &self,
        token_bytes: &BTreeMap<u32, Vec<u8>>,
    ) -> bool {
        if self.canonical_original_tokens.len() != token_bytes.len() {
            return false;
        }
        let mut covered = FxHashSet::<u32>::default();
        for &token_id in self.canonical_original_tokens.iter() {
            if !token_bytes.contains_key(&token_id) || !covered.insert(token_id) {
                return false;
            }
        }
        if covered.len() != token_bytes.len() {
            return false;
        }

        struct Frame {
            node: u32,
            next_child: usize,
            prefix_len: usize,
        }
        let mut canonical_seen = vec![false; self.canonical_token_count()];
        let mut prefix = Vec::<u8>::new();
        let mut frames = vec![Frame {
            node: 0,
            next_child: 0,
            prefix_len: 0,
        }];
        while !frames.is_empty() {
            let frame_index = frames.len() - 1;
            let node_id = frames[frame_index].node;
            if frames[frame_index].next_child == 0 {
                if let Some(canonical) = self.trie.node(node_id).token_id {
                    let canonical = canonical as usize;
                    if canonical >= canonical_seen.len() || canonical_seen[canonical] {
                        return false;
                    }
                    let Some(originals) = self.token_ids(canonical as u32) else {
                        return false;
                    };
                    let Some(representative) = originals.first() else {
                        return false;
                    };
                    if token_bytes
                        .get(representative)
                        .is_none_or(|bytes| bytes.as_slice() != prefix.as_slice())
                    {
                        return false;
                    }
                    canonical_seen[canonical] = true;
                }
            }
            let children = self.trie.children(node_id);
            if frames[frame_index].next_child < children.len() {
                let edge = children[frames[frame_index].next_child].clone();
                frames[frame_index].next_child += 1;
                let prefix_len = prefix.len();
                prefix.extend_from_slice(self.trie.edge_bytes(&edge));
                frames.push(Frame {
                    node: edge.child,
                    next_child: 0,
                    prefix_len,
                });
            } else {
                let prefix_len = frames[frame_index].prefix_len;
                frames.pop();
                prefix.truncate(prefix_len);
            }
        }
        canonical_seen.into_iter().all(|seen| seen)
    }
}

impl Default for DynamicMaskVocab {
    fn default() -> Self {
        Self {
            trie: Arc::new(DynamicMaskTrie::new()),
            token_aliases: DynamicMaskAliasStore::Packed(Arc::new(Vec::new())),
            canonical_original_token_offsets: Arc::new(vec![0]),
            canonical_original_tokens: Arc::new(Vec::new()),
            canonical_original_word_offsets: Arc::new(vec![0]),
            canonical_original_word_masks: Arc::new(Vec::new()),
            node_token_markers: Arc::new(vec![0]),
            full_walk_token_markers: Arc::new(Vec::new()),
            subtree_original_token_offsets: Arc::new(vec![0]),
            subtree_original_tokens: Arc::new(Vec::new()),
            all_original_token_words: Arc::new(Vec::new()),
            llg_slice_leftovers: Arc::new(Vec::new()),
            llg_master_admitted_words: Arc::new(Vec::new()),
            llg_master_residual_ops: Arc::new(Vec::new()),
            llg_master_max_safe_chars: 0,
            prepared_master_prover_row_ids: Arc::from(Vec::<u32>::new()),
            prepared_master_prover_offsets: Arc::from(Vec::<u32>::new()),
            prepared_master_prover_terminals: Arc::from(Vec::<TerminalID>::new()),
            prepared_master_coverage_row_ids: Arc::from(Vec::<u32>::new()),
            prepared_master_coverage_offsets: Arc::from(Vec::<u32>::new()),
            prepared_master_coverage_terminals: Arc::from(Vec::<TerminalID>::new()),
            prepared_safe_plus_complete_terminals: Arc::from(Vec::<TerminalID>::new()),
            prepared_safe_radius_row_ids: Arc::from(Vec::<u32>::new()),
            prepared_safe_radius_offsets: Arc::from(Vec::<u32>::new()),
            prepared_safe_radius_entries: Arc::from(Vec::<(TerminalID, u16)>::new()),
            pending_source: None,
            initialized: false,
            grammar_quotiented: false,
            source_vocab_digest: None,
            mask_cache: Arc::new(Mutex::new(DynamicMaskCache::default())),
            dense_subset16_cache: Arc::new(Mutex::new(FxHashMap::default())),
            lazy_union_cache: Arc::new(Mutex::new(DynamicLazyUnionCache::default())),
            direct_regular_frontier_cache: Arc::new(Mutex::new(FxHashMap::default())),
            direct_regular_wide_frontier_index_cache: Arc::new(Mutex::new(FxHashMap::default())),
            direct_regular_terminal_support: Arc::new(DirectRegularTerminalSupport::default()),
            self_loop_projections: Arc::new(Vec::new()),
            projection_by_source: Arc::from(Vec::<u32>::new()),
            projection_alias_vocab: Arc::from(Vec::<u32>::new()),
            projection_alias_h64: Arc::from(Vec::<u32>::new()),
            bounded_observation_sets: Arc::new(DynamicBoundedObservationSets::default()),
            terminal_observation_classes: Arc::from(Vec::<(TerminalID, Arc<[u32]>)>::new()),
            projected_terminal_quotients: Arc::from(Vec::<(TerminalID, Arc<TerminalProjectedQuotient>)>::new()),
            runtime_projected_terminal_quotients: Arc::new(OnceLock::new()),
            runtime_projected_terminal_quotient_cache: Arc::new(Mutex::new(FxHashMap::default())),
            projected_terminal_quotients_prepared: false,
            projected_terminal_text_cache: Arc::new(Mutex::new(FxHashMap::default())),
            projected_terminal_partition_cache: Arc::new(Mutex::new(FxHashMap::default())),
            projected_terminal_radius_cache: Arc::new(Mutex::new(FxHashMap::default())),
            raw_terminal_radius_cache: Arc::new(Mutex::new(FxHashMap::default())),
            pending_guard_blocked_mask_cache: Arc::new(Mutex::new(FxHashMap::default())),
            mask_tokenizer: None,
            mask_determinized_tokenizer: None,
            mask_projection_to_determinized: Arc::from(Vec::<u32>::new()),
            mask_tokenizer_fast_transitions: None,
            full_to_mask_state: Arc::from(Vec::<u32>::new()),
            mask_state_source_subsets: Arc::from(Vec::<Arc<[u32]>>::new()),
            mask_source_subset_to_state: Arc::new(FxHashMap::default()),
            virtual_unit_repeat_projection: None,
            virtual_repeat_intersection_projections: Vec::new(),
            virtual_residual_projections: Vec::new(),
        }
    }
}

/// Version-scoped serde for the inverse token-id map. Current sectioned
/// artifacts carry only `original_token_to_internal`; the inverse is exactly
/// derivable from it and is rebuilt after the core section is decoded. Older
/// artifact versions leave this mode disabled and retain their historical wire
/// shape.
pub(crate) mod internal_token_inverse_artifact_serde {
    use std::cell::Cell;

    use serde::{Deserialize, Serialize};

    thread_local! {
        static OMIT_INVERSE: Cell<bool> = const { Cell::new(false) };
    }

    pub(crate) fn set_omit(enabled: bool) -> bool {
        OMIT_INVERSE.with(|mode| mode.replace(enabled))
    }

    pub fn serialize<S>(value: &[Vec<u32>], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        if OMIT_INVERSE.with(Cell::get) {
            return ().serialize(serializer);
        }
        value.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<Vec<u32>>, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        if OMIT_INVERSE.with(Cell::get) {
            <()>::deserialize(deserializer)?;
            return Ok(Vec::new());
        }
        Vec::<Vec<u32>>::deserialize(deserializer)
    }
}

/// Current-core serialization can omit the tokenizer-state inverse when it is
/// exactly derivable from the scalar state -> internal-TSID map. The default
/// mode preserves the historical `Vec<Vec<u32>>` bincode wire, so legacy
/// artifact decoding is unchanged.
pub(crate) mod internal_tsid_inverse_artifact_serde {
    use std::cell::Cell;

    use serde::{Deserialize, Serialize};

    thread_local! {
        static OMIT_INVERSE: Cell<bool> = const { Cell::new(false) };
    }

    pub(crate) fn set_omit(enabled: bool) -> bool {
        OMIT_INVERSE.with(|mode| mode.replace(enabled))
    }

    pub fn serialize<S>(value: &[Vec<u32>], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        if OMIT_INVERSE.with(Cell::get) {
            return ().serialize(serializer);
        }
        value.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<Vec<u32>>, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        if OMIT_INVERSE.with(Cell::get) {
            <()>::deserialize(deserializer)?;
            return Ok(Vec::new());
        }
        Vec::<Vec<u32>>::deserialize(deserializer)
    }
}

/// Compact v14+ wire form for the dense original-token -> internal-token map.
/// Internal IDs are normally only a few thousand wide even for 128k-token
/// vocabularies, so fixed-width u32 storage wastes roughly half this field.
/// Zero encodes the historical `u32::MAX` sentinel; ordinary IDs are stored as
/// `id + 1` varints. Older artifact versions leave this mode disabled.
pub(crate) mod original_token_map_artifact_serde {
    use std::cell::Cell;
    use std::sync::Arc;

    use serde::{Deserialize, Serialize};

    const VARINT_MAGIC: &[u8; 4] = b"OTM1";
    const FIXED_MAGIC: &[u8; 4] = b"OTM2";
    const FIXED_HEADER_LEN: usize = FIXED_MAGIC.len() + 1 + 4;

    #[derive(Debug)]
    pub(crate) struct PackedOriginalTokenMap {
        backing: Arc<Vec<u8>>,
        payload_start: usize,
        count: usize,
        width: usize,
    }

    impl PackedOriginalTokenMap {
        pub(crate) fn parse_backed(
            backing: Arc<Vec<u8>>,
            start: usize,
            len: usize,
        ) -> Result<Self, String> {
            let end = start
                .checked_add(len)
                .ok_or_else(|| "fixed original-token map range overflows".to_owned())?;
            let input = backing
                .get(start..end)
                .ok_or_else(|| "fixed original-token map is outside artifact backing".to_owned())?;
            if input.len() < FIXED_HEADER_LEN || !input.starts_with(FIXED_MAGIC) {
                return Err("invalid fixed original-token map header".to_owned());
            }
            let width = input[FIXED_MAGIC.len()] as usize;
            if !matches!(width, 1 | 2 | 4) {
                return Err("invalid fixed original-token map width".to_owned());
            }
            let count_start = FIXED_MAGIC.len() + 1;
            let count = u32::from_le_bytes(
                input[count_start..count_start + 4]
                    .try_into()
                    .expect("fixed original-token count has fixed width"),
            ) as usize;
            let payload_len = count
                .checked_mul(width)
                .ok_or_else(|| "fixed original-token map payload overflows".to_owned())?;
            if FIXED_HEADER_LEN
                .checked_add(payload_len)
                .is_none_or(|expected| expected != input.len())
            {
                return Err("invalid fixed original-token map length".to_owned());
            }
            Ok(Self {
                backing,
                payload_start: start + FIXED_HEADER_LEN,
                count,
                width,
            })
        }

        #[inline]
        pub(crate) fn len(&self) -> usize {
            self.count
        }

        #[inline]
        pub(crate) fn is_empty(&self) -> bool {
            self.count == 0
        }

        #[inline]
        pub(crate) fn get(&self, index: usize) -> Option<u32> {
            if index >= self.count {
                return None;
            }
            let start = self.payload_start + index * self.width;
            let bytes = self.backing.get(start..start + self.width)?;
            Some(match self.width {
                1 => {
                    let value = bytes[0];
                    if value == u8::MAX { u32::MAX } else { value as u32 }
                }
                2 => {
                    let value = u16::from_le_bytes([bytes[0], bytes[1]]);
                    if value == u16::MAX { u32::MAX } else { value as u32 }
                }
                4 => u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
                _ => unreachable!(),
            })
        }

        pub(crate) fn materialize(&self) -> Vec<u32> {
            (0..self.count)
                .map(|index| self.get(index).expect("validated packed original-token map index"))
                .collect()
        }
    }

    thread_local! {
        static PACKED: Cell<bool> = const { Cell::new(false) };
        static EXTERNAL: Cell<bool> = const { Cell::new(false) };
    }

    pub(crate) fn set_packed(enabled: bool) -> bool {
        PACKED.with(|mode| mode.replace(enabled))
    }

    pub(crate) fn set_external(enabled: bool) -> bool {
        EXTERNAL.with(|mode| mode.replace(enabled))
    }

    pub(crate) fn to_fast_bytes(value: &[u32]) -> Vec<u8> {
        pack_fixed(value)
    }

    pub(crate) fn from_fast_bytes(input: &[u8]) -> Result<Vec<u32>, String> {
        unpack(input)
    }

    #[inline]
    fn put_var_u32(out: &mut Vec<u8>, mut value: u32) {
        while value >= 0x80 {
            out.push((value as u8) | 0x80);
            value >>= 7;
        }
        out.push(value as u8);
    }

    #[inline]
    fn take_var_u32(input: &[u8], pos: &mut usize) -> Result<u32, String> {
        let mut value = 0u32;
        let mut shift = 0u32;
        for _ in 0..5 {
            let byte = *input
                .get(*pos)
                .ok_or_else(|| "truncated packed original-token map".to_owned())?;
            *pos += 1;
            if shift == 28 && byte > 0x0f {
                return Err("overflowing packed original-token map".to_owned());
            }
            value |= ((byte & 0x7f) as u32) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
            shift += 7;
        }
        Err("overflowing packed original-token map".to_owned())
    }

    fn pack_varint(value: &[u32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(value.len().saturating_mul(2));
        out.extend_from_slice(VARINT_MAGIC);
        put_var_u32(
            &mut out,
            u32::try_from(value.len()).expect("token vocabulary should fit u32"),
        );
        for &internal in value {
            let encoded = if internal == u32::MAX {
                0
            } else {
                internal
                    .checked_add(1)
                    .expect("u32::MAX is reserved as the unmapped-token sentinel")
            };
            put_var_u32(&mut out, encoded);
        }
        out
    }

    fn unpack_varint(input: &[u8]) -> Result<Vec<u32>, String> {
        if !input.starts_with(VARINT_MAGIC) {
            return Err("invalid varint original-token map header".to_owned());
        }
        let mut pos = VARINT_MAGIC.len();
        let count = take_var_u32(input, &mut pos)? as usize;
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            let encoded = take_var_u32(input, &mut pos)?;
            out.push(if encoded == 0 { u32::MAX } else { encoded - 1 });
        }
        if pos != input.len() {
            return Err("trailing bytes in packed original-token map".to_owned());
        }
        Ok(out)
    }

    fn pack_fixed(value: &[u32]) -> Vec<u8> {
        let max_internal = value
            .iter()
            .copied()
            .filter(|&internal| internal != u32::MAX)
            .max()
            .unwrap_or(0);
        let width = if max_internal < u8::MAX as u32 {
            1u8
        } else if max_internal < u16::MAX as u32 {
            2u8
        } else {
            4u8
        };
        let payload_len = value
            .len()
            .checked_mul(width as usize)
            .expect("original-token map payload should fit usize");
        let mut out = Vec::with_capacity(FIXED_HEADER_LEN + payload_len);
        out.extend_from_slice(FIXED_MAGIC);
        out.push(width);
        out.extend_from_slice(
            &u32::try_from(value.len())
                .expect("token vocabulary should fit u32")
                .to_le_bytes(),
        );
        match width {
            1 => {
                for &internal in value {
                    out.push(if internal == u32::MAX {
                        u8::MAX
                    } else {
                        internal as u8
                    });
                }
            }
            2 => {
                for &internal in value {
                    let encoded = if internal == u32::MAX {
                        u16::MAX
                    } else {
                        internal as u16
                    };
                    out.extend_from_slice(&encoded.to_le_bytes());
                }
            }
            4 => {
                for &internal in value {
                    out.extend_from_slice(&internal.to_le_bytes());
                }
            }
            _ => unreachable!(),
        }
        out
    }

    fn unpack_fixed(input: &[u8]) -> Result<Vec<u32>, String> {
        if input.len() < FIXED_HEADER_LEN || !input.starts_with(FIXED_MAGIC) {
            return Err("invalid fixed original-token map header".to_owned());
        }
        let width = input[FIXED_MAGIC.len()] as usize;
        if !matches!(width, 1 | 2 | 4) {
            return Err("invalid fixed original-token map width".to_owned());
        }
        let count_start = FIXED_MAGIC.len() + 1;
        let count = u32::from_le_bytes(
            input[count_start..count_start + 4]
                .try_into()
                .expect("fixed original-token count has fixed width"),
        ) as usize;
        let payload_len = count
            .checked_mul(width)
            .ok_or_else(|| "fixed original-token map payload overflows".to_owned())?;
        if FIXED_HEADER_LEN
            .checked_add(payload_len)
            .is_none_or(|expected| expected != input.len())
        {
            return Err("invalid fixed original-token map length".to_owned());
        }
        let payload = &input[FIXED_HEADER_LEN..];
        let mut out = Vec::with_capacity(count);
        match width {
            1 => out.extend(payload.iter().map(|&encoded| {
                if encoded == u8::MAX {
                    u32::MAX
                } else {
                    encoded as u32
                }
            })),
            2 => out.extend(payload.chunks_exact(2).map(|bytes| {
                let encoded = u16::from_le_bytes([bytes[0], bytes[1]]);
                if encoded == u16::MAX {
                    u32::MAX
                } else {
                    encoded as u32
                }
            })),
            4 => out.extend(payload.chunks_exact(4).map(|bytes| {
                u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
            })),
            _ => unreachable!(),
        }
        Ok(out)
    }

    fn unpack(input: &[u8]) -> Result<Vec<u32>, String> {
        if input.starts_with(FIXED_MAGIC) {
            unpack_fixed(input)
        } else if input.starts_with(VARINT_MAGIC) {
            unpack_varint(input)
        } else {
            Err("invalid packed original-token map header".to_owned())
        }
    }

    pub fn serialize<S>(value: &[u32], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        if EXTERNAL.with(Cell::get) {
            return 0u8.serialize(serializer);
        }
        if !PACKED.with(Cell::get) {
            return value.serialize(serializer);
        }
        let profile = std::env::var_os("GLRMASK_PROFILE_SERIALIZATION").is_some();
        let total_started = profile.then(std::time::Instant::now);
        let pack_started = profile.then(std::time::Instant::now);
        let packed = pack_fixed(value);
        let pack_ms = pack_started.map_or(0.0, |s| s.elapsed().as_secs_f64() * 1000.0);
        let wire_bytes = packed.len();
        let result = packed.serialize(serializer);
        if let Some(started) = total_started {
            eprintln!(
                "[glrmask/profile][original_token_map_encode] entries={} wire_bytes={} pack_ms={:.3} total_ms={:.3}",
                value.len(),
                wire_bytes,
                pack_ms,
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
        result
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<u32>, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        if EXTERNAL.with(Cell::get) {
            let marker = u8::deserialize(deserializer)?;
            if marker != 0 {
                return Err(serde::de::Error::custom(
                    "invalid external original-token map placeholder",
                ));
            }
            return Ok(Vec::new());
        }
        if !PACKED.with(Cell::get) {
            return Vec::<u32>::deserialize(deserializer);
        }
        let profile = std::env::var_os("GLRMASK_PROFILE_SERIALIZATION").is_some();
        let total_started = profile.then(std::time::Instant::now);
        let packed = Vec::<u8>::deserialize(deserializer)?;
        let packed_len = packed.len();
        let result = unpack(&packed).map_err(serde::de::Error::custom);
        if let Some(started) = total_started {
            eprintln!(
                "[glrmask/profile][original_token_map_decode] wire_bytes={} ms={:.3}",
                packed_len,
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
        result
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn fixed_original_token_map_roundtrips_all_widths_and_legacy() {
            for values in [
                vec![0, 12, u32::MAX, 254],
                vec![0, 255, 4096, u32::MAX, 65534],
                vec![0, 65535, 1_000_000, u32::MAX],
            ] {
                let packed = pack_fixed(&values);
                assert_eq!(unpack(&packed).unwrap(), values);

                let legacy = pack_varint(&values);
                assert_eq!(unpack(&legacy).unwrap(), values);
            }
        }
    }
}

/// Compact v14+ wire encoding for the immutable model-token byte vocabulary.
/// Ordinary LLM vocabs are dense in token id, so the historical
/// `BTreeMap<u32, Vec<u8>>` representation spends more bytes on map keys and
/// per-Vec lengths than on useful token data. The packed form stores a dense
/// sequence of varint lengths followed by token bytes, with a sparse fallback
/// for unusual vocabularies. Deserialization reconstructs the exact historical
/// in-memory BTreeMap, so compiler/composition/runtime APIs do not change.
pub(crate) mod token_bytes_artifact_serde {
    use std::cell::{Cell, RefCell};
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use serde::{Deserialize, Serialize};

    const LEGACY_MAGIC: &[u8; 4] = b"TBP1";
    const INDEXED_MAGIC: &[u8; 4] = b"TBP2";
    const INDEXED_HEADER_LEN: usize = INDEXED_MAGIC.len() + 1 + 4;

    thread_local! {
        static PACKED: Cell<bool> = const { Cell::new(false) };
        static DEFER_UNPACK: Cell<bool> = const { Cell::new(false) };
        static EXTERNAL: Cell<bool> = const { Cell::new(false) };
        static DEFERRED: RefCell<Option<Arc<PackedTokenBytes>>> = const { RefCell::new(None) };
    }

    pub(crate) fn set_packed(enabled: bool) -> bool {
        PACKED.with(|mode| mode.replace(enabled))
    }

    pub(crate) fn set_defer_unpack(enabled: bool) -> bool {
        DEFER_UNPACK.with(|mode| mode.replace(enabled))
    }

    pub(crate) fn set_external(enabled: bool) -> bool {
        EXTERNAL.with(|mode| mode.replace(enabled))
    }

    pub(crate) fn take_deferred() -> Option<Arc<PackedTokenBytes>> {
        DEFERRED.with(|slot| slot.borrow_mut().take())
    }

    #[derive(Debug)]
    pub(crate) struct PackedTokenBytes {
        wire: Arc<Vec<u8>>,
        wire_start: usize,
        wire_len: usize,
        indexed: Option<PackedTokenBytesIndexed>,
        spans: Box<[(u32, u32)]>,
        sparse_ids: Option<Box<[u32]>>,
    }

    #[derive(Debug, Clone, Copy)]
    struct PackedTokenBytesIndexed {
        count: usize,
        sparse_ids_start: Option<usize>,
        offsets_start: usize,
        data_start: usize,
    }

    impl PackedTokenBytes {
        pub(crate) fn from_runtime_entries(value: &BTreeMap<u32, Vec<u8>>) -> Result<Self, String> {
            // The indexed representation is useful runtime state in its own
            // right: token lookup/iteration reads it directly. Build it once
            // when a compiler-created Constraint is finalized rather than
            // rebuilding the same index inside every save().
            Self::parse(pack(value))
        }

        fn parse(wire: Vec<u8>) -> Result<Self, String> {
            let wire_len = wire.len();
            Self::parse_backed(Arc::new(wire), 0, wire_len)
        }

        pub(crate) fn parse_backed(
            wire: Arc<Vec<u8>>,
            wire_start: usize,
            wire_len: usize,
        ) -> Result<Self, String> {
            let wire_end = wire_start
                .checked_add(wire_len)
                .ok_or_else(|| "overflowing packed token-byte backing range".to_owned())?;
            let input = wire
                .get(wire_start..wire_end)
                .ok_or_else(|| "packed token-byte backing range is out of bounds".to_owned())?;
            if input.starts_with(INDEXED_MAGIC) {
                return Self::parse_indexed_backed(wire, wire_start, wire_len);
            }
            if !input.starts_with(LEGACY_MAGIC) {
                return Err("invalid packed token-byte header".to_owned());
            }
            let mut pos = LEGACY_MAGIC.len();
            let sparse = match input.get(pos).copied() {
                Some(0) => false,
                Some(1) => true,
                _ => return Err("invalid packed token-byte mode".to_owned()),
            };
            pos += 1;
            let count = take_var_u32(input, &mut pos)? as usize;
            let mut spans = Vec::with_capacity(count);
            let mut sparse_ids = sparse.then(|| Vec::with_capacity(count));
            let mut previous_end = 0u64;
            for dense_id in 0..count {
                let id = if sparse {
                    let gap = take_var_u32(input, &mut pos)? as u64;
                    let id = previous_end
                        .checked_add(gap)
                        .ok_or_else(|| "overflowing packed token id".to_owned())?;
                    let id = u32::try_from(id)
                        .map_err(|_| "overflowing packed token id".to_owned())?;
                    previous_end = id as u64 + 1;
                    sparse_ids.as_mut().expect("sparse ids enabled").push(id);
                    id
                } else {
                    u32::try_from(dense_id)
                        .map_err(|_| "dense packed token id exceeds u32".to_owned())?
                };
                let _ = id;
                let len = take_var_u32(input, &mut pos)? as usize;
                let start = pos;
                let end = start
                    .checked_add(len)
                    .ok_or_else(|| "overflowing packed token-byte length".to_owned())?;
                if end > input.len() {
                    return Err("truncated packed token bytes".to_owned());
                }
                spans.push((
                    u32::try_from(start)
                        .map_err(|_| "packed token byte offset exceeds u32".to_owned())?,
                    u32::try_from(len)
                        .map_err(|_| "packed token byte length exceeds u32".to_owned())?,
                ));
                pos = end;
            }
            if pos != input.len() {
                return Err("trailing bytes in packed token-byte vocabulary".to_owned());
            }
            Ok(Self {
                wire,
                wire_start,
                wire_len,
                indexed: None,
                spans: spans.into_boxed_slice(),
                sparse_ids: sparse_ids.map(Vec::into_boxed_slice),
            })
        }

        fn parse_indexed_backed(
            wire: Arc<Vec<u8>>,
            wire_start: usize,
            wire_len: usize,
        ) -> Result<Self, String> {
            let wire_end = wire_start
                .checked_add(wire_len)
                .ok_or_else(|| "overflowing indexed token-byte backing range".to_owned())?;
            let input = wire
                .get(wire_start..wire_end)
                .ok_or_else(|| "indexed token-byte backing range is out of bounds".to_owned())?;
            if input.len() < INDEXED_HEADER_LEN || !input.starts_with(INDEXED_MAGIC) {
                return Err("invalid indexed token-byte header".to_owned());
            }
            let sparse = match input[INDEXED_MAGIC.len()] {
                0 => false,
                1 => true,
                _ => return Err("invalid indexed token-byte mode".to_owned()),
            };
            let count_start = INDEXED_MAGIC.len() + 1;
            let count = u32::from_le_bytes(
                input[count_start..count_start + 4]
                    .try_into()
                    .expect("indexed token count has fixed width"),
            ) as usize;
            let sparse_ids_start = sparse.then_some(INDEXED_HEADER_LEN);
            let ids_bytes = if sparse {
                count
                    .checked_mul(4)
                    .ok_or_else(|| "indexed token-id table overflows".to_owned())?
            } else {
                0
            };
            let offsets_start = INDEXED_HEADER_LEN
                .checked_add(ids_bytes)
                .ok_or_else(|| "indexed token-byte offsets start overflows".to_owned())?;
            let offsets_bytes = count
                .checked_add(1)
                .and_then(|count| count.checked_mul(4))
                .ok_or_else(|| "indexed token-byte offset table overflows".to_owned())?;
            let data_start = offsets_start
                .checked_add(offsets_bytes)
                .ok_or_else(|| "indexed token-byte data start overflows".to_owned())?;
            if data_start > input.len() {
                return Err("truncated indexed token-byte tables".to_owned());
            }
            let first_offset = read_u32_at(input, offsets_start)? as usize;
            let final_offset = read_u32_at(input, offsets_start + count * 4)? as usize;
            if first_offset != 0 || final_offset != input.len() - data_start {
                return Err("invalid indexed token-byte offset bounds".to_owned());
            }
            Ok(Self {
                wire,
                wire_start,
                wire_len,
                indexed: Some(PackedTokenBytesIndexed {
                    count,
                    sparse_ids_start,
                    offsets_start,
                    data_start,
                }),
                spans: Box::new([]),
                sparse_ids: None,
            })
        }

        #[inline]
        pub(crate) fn wire(&self) -> &[u8] {
            &self.wire[self.wire_start..self.wire_start + self.wire_len]
        }

        pub(crate) fn whole_wire_arc(&self) -> Option<Arc<Vec<u8>>> {
            (self.wire_start == 0 && self.wire_len == self.wire.len())
                .then(|| Arc::clone(&self.wire))
        }

        #[inline]
        pub(crate) fn len(&self) -> usize {
            self.indexed.map_or(self.spans.len(), |indexed| indexed.count)
        }

        #[inline]
        pub(crate) fn get(&self, token_id: u32) -> Option<&[u8]> {
            if let Some(indexed) = self.indexed {
                let index = match indexed.sparse_ids_start {
                    None => usize::try_from(token_id)
                        .ok()
                        .filter(|&index| index < indexed.count)?,
                    Some(_) => self.indexed_sparse_token_index(indexed, token_id)?,
                };
                return self.indexed_bytes_at(indexed, index);
            }
            let index = match &self.sparse_ids {
                None => usize::try_from(token_id).ok().filter(|&index| index < self.spans.len())?,
                Some(ids) => ids.binary_search(&token_id).ok()?,
            };
            let (start, len) = self.spans[index];
            let start = start as usize;
            self.wire().get(start..start + len as usize)
        }

        pub(crate) fn iter(&self) -> impl Iterator<Item = (u32, &[u8])> + '_ {
            (0..self.len()).map(|index| {
                let token_id = self
                    .token_id_at(index)
                    .expect("validated packed token index should have an id");
                let bytes = self
                    .bytes_at_index(index)
                    .expect("validated packed token index should have bytes");
                (token_id, bytes)
            })
        }

        pub(crate) fn max_token_id(&self) -> Option<u32> {
            if self.indexed.is_some() {
                return self.len().checked_sub(1).and_then(|index| self.token_id_at(index));
            }
            match &self.sparse_ids {
                Some(ids) => ids.last().copied(),
                None => self.spans.len().checked_sub(1).map(|id| id as u32),
            }
        }

        pub(crate) fn materialize(&self) -> Arc<BTreeMap<u32, Vec<u8>>> {
            Arc::new(
                self.iter()
                    .map(|(token_id, bytes)| (token_id, bytes.to_vec()))
                    .collect(),
            )
        }

        fn token_id_at(&self, index: usize) -> Option<u32> {
            if let Some(indexed) = self.indexed {
                if index >= indexed.count {
                    return None;
                }
                return indexed.sparse_ids_start.map_or_else(
                    || u32::try_from(index).ok(),
                    |start| read_u32_at(self.wire(), start + index * 4).ok(),
                );
            }
            self.sparse_ids
                .as_ref()
                .map_or_else(|| u32::try_from(index).ok(), |ids| ids.get(index).copied())
        }

        fn bytes_at_index(&self, index: usize) -> Option<&[u8]> {
            if let Some(indexed) = self.indexed {
                return self.indexed_bytes_at(indexed, index);
            }
            let &(start, len) = self.spans.get(index)?;
            let start = start as usize;
            self.wire().get(start..start + len as usize)
        }

        fn indexed_bytes_at(
            &self,
            indexed: PackedTokenBytesIndexed,
            index: usize,
        ) -> Option<&[u8]> {
            if index >= indexed.count {
                return None;
            }
            let wire = self.wire();
            let start = read_u32_at(wire, indexed.offsets_start + index * 4).ok()? as usize;
            let end = read_u32_at(wire, indexed.offsets_start + (index + 1) * 4).ok()? as usize;
            if start > end {
                return None;
            }
            wire.get(indexed.data_start + start..indexed.data_start + end)
        }

        fn indexed_sparse_token_index(
            &self,
            indexed: PackedTokenBytesIndexed,
            token_id: u32,
        ) -> Option<usize> {
            let start = indexed.sparse_ids_start?;
            let wire = self.wire();
            let mut lo = 0usize;
            let mut hi = indexed.count;
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                let candidate = read_u32_at(wire, start + mid * 4).ok()?;
                match candidate.cmp(&token_id) {
                    std::cmp::Ordering::Less => lo = mid + 1,
                    std::cmp::Ordering::Greater => hi = mid,
                    std::cmp::Ordering::Equal => return Some(mid),
                }
            }
            None
        }
    }

    #[inline]
    fn read_u32_at(input: &[u8], pos: usize) -> Result<u32, String> {
        let end = pos
            .checked_add(4)
            .ok_or_else(|| "packed token-byte u32 offset overflows".to_owned())?;
        let bytes = input
            .get(pos..end)
            .ok_or_else(|| "truncated packed token-byte u32".to_owned())?;
        Ok(u32::from_le_bytes(
            bytes.try_into().expect("packed token u32 has fixed width"),
        ))
    }

    #[inline]
    fn put_var_u32(out: &mut Vec<u8>, mut value: u32) {
        while value >= 0x80 {
            out.push((value as u8) | 0x80);
            value >>= 7;
        }
        out.push(value as u8);
    }

    #[inline]
    fn take_var_u32(input: &[u8], pos: &mut usize) -> Result<u32, String> {
        let mut value = 0u32;
        let mut shift = 0u32;
        for _ in 0..5 {
            let byte = *input
                .get(*pos)
                .ok_or_else(|| "truncated packed token-byte varint".to_owned())?;
            *pos += 1;
            if shift == 28 && byte > 0x0f {
                return Err("overflowing packed token-byte varint".to_owned());
            }
            value |= ((byte & 0x7f) as u32) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
            shift += 7;
        }
        Err("overflowing packed token-byte varint".to_owned())
    }

    fn pack_legacy(value: &BTreeMap<u32, Vec<u8>>) -> Vec<u8> {
        let dense = value
            .keys()
            .copied()
            .enumerate()
            .all(|(expected, actual)| actual as usize == expected);
        let mut out = Vec::new();
        out.extend_from_slice(LEGACY_MAGIC);
        out.push(u8::from(!dense));
        put_var_u32(
            &mut out,
            u32::try_from(value.len()).expect("token vocabulary should fit u32"),
        );
        let mut previous_end = 0u64;
        for (&id, bytes) in value {
            if !dense {
                let gap = (id as u64)
                    .checked_sub(previous_end)
                    .expect("token ids are sorted");
                put_var_u32(
                    &mut out,
                    u32::try_from(gap).expect("token-id gap should fit u32"),
                );
                previous_end = id as u64 + 1;
            }
            put_var_u32(
                &mut out,
                u32::try_from(bytes.len()).expect("token byte length should fit u32"),
            );
            out.extend_from_slice(bytes);
        }
        out
    }

    fn pack(value: &BTreeMap<u32, Vec<u8>>) -> Vec<u8> {
        let dense = value
            .keys()
            .copied()
            .enumerate()
            .all(|(expected, actual)| actual as usize == expected);
        let count = value.len();
        let data_len = value.values().try_fold(0usize, |total, bytes| {
            total.checked_add(bytes.len())
        });
        let Some(data_len) = data_len.filter(|&len| u32::try_from(len).is_ok()) else {
            return pack_legacy(value);
        };
        let ids_len = if dense { 0 } else { count.saturating_mul(4) };
        let Some(offsets_len) = count.checked_add(1).and_then(|count| count.checked_mul(4)) else {
            return pack_legacy(value);
        };
        let capacity = INDEXED_HEADER_LEN
            .checked_add(ids_len)
            .and_then(|len| len.checked_add(offsets_len))
            .and_then(|len| len.checked_add(data_len));
        let Some(capacity) = capacity else {
            return pack_legacy(value);
        };
        // Reserve the complete indexed representation up front and fill the
        // id/offset/data regions in one BTreeMap traversal.  The previous
        // implementation walked the pointer-heavy map once for offsets and a
        // second time for bytes; on 100k+ token vocabularies that dominated a
        // genuinely fresh Constraint::save().
        let mut out = vec![0u8; capacity];
        out[..INDEXED_MAGIC.len()].copy_from_slice(INDEXED_MAGIC);
        out[INDEXED_MAGIC.len()] = u8::from(!dense);
        out[INDEXED_MAGIC.len() + 1..INDEXED_HEADER_LEN].copy_from_slice(
            &u32::try_from(count)
                .expect("token vocabulary should fit u32")
                .to_le_bytes(),
        );

        let ids_start = INDEXED_HEADER_LEN;
        let offsets_start = ids_start + ids_len;
        let data_start = offsets_start + offsets_len;
        out[offsets_start..offsets_start + 4].copy_from_slice(&0u32.to_le_bytes());

        let mut offset = 0u32;
        let mut data_pos = data_start;
        for (index, (&token_id, bytes)) in value.iter().enumerate() {
            if !dense {
                let id_pos = ids_start + index * 4;
                out[id_pos..id_pos + 4].copy_from_slice(&token_id.to_le_bytes());
            }
            let next_offset = offset
                .checked_add(bytes.len() as u32)
                .expect("indexed token-byte data length was prevalidated");
            let offset_pos = offsets_start + (index + 1) * 4;
            out[offset_pos..offset_pos + 4].copy_from_slice(&next_offset.to_le_bytes());
            let data_end = data_pos + bytes.len();
            out[data_pos..data_end].copy_from_slice(bytes);
            data_pos = data_end;
            offset = next_offset;
        }
        debug_assert_eq!(data_pos, capacity);
        out
    }

    pub(crate) fn pack_external(value: &BTreeMap<u32, Vec<u8>>) -> Vec<u8> {
        pack(value)
    }

    fn unpack(input: &[u8]) -> Result<Arc<BTreeMap<u32, Vec<u8>>>, String> {
        if input.starts_with(INDEXED_MAGIC) {
            return PackedTokenBytes::parse(input.to_vec()).map(|packed| packed.materialize());
        }
        if !input.starts_with(LEGACY_MAGIC) {
            return Err("invalid packed token-byte header".to_owned());
        }
        let mut pos = LEGACY_MAGIC.len();
        let sparse = match input.get(pos).copied() {
            Some(0) => false,
            Some(1) => true,
            _ => return Err("invalid packed token-byte mode".to_owned()),
        };
        pos += 1;
        let count = take_var_u32(input, &mut pos)? as usize;
        let mut map = BTreeMap::new();
        let mut previous_end = 0u64;
        for dense_id in 0..count {
            let id = if sparse {
                let gap = take_var_u32(input, &mut pos)? as u64;
                let id = previous_end
                    .checked_add(gap)
                    .ok_or_else(|| "overflowing packed token id".to_owned())?;
                let id = u32::try_from(id)
                    .map_err(|_| "overflowing packed token id".to_owned())?;
                previous_end = id as u64 + 1;
                id
            } else {
                u32::try_from(dense_id)
                    .map_err(|_| "dense packed token id exceeds u32".to_owned())?
            };
            let len = take_var_u32(input, &mut pos)? as usize;
            let end = pos
                .checked_add(len)
                .ok_or_else(|| "overflowing packed token-byte length".to_owned())?;
            let bytes = input
                .get(pos..end)
                .ok_or_else(|| "truncated packed token bytes".to_owned())?
                .to_vec();
            pos = end;
            if map.insert(id, bytes).is_some() {
                return Err("duplicate packed token id".to_owned());
            }
        }
        if pos != input.len() {
            return Err("trailing bytes in packed token-byte vocabulary".to_owned());
        }
        Ok(Arc::new(map))
    }

    pub fn serialize<S>(
        value: &Arc<BTreeMap<u32, Vec<u8>>>,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        if !PACKED.with(Cell::get) {
            return value.serialize(serializer);
        }
        if EXTERNAL.with(Cell::get) {
            return Vec::<u8>::new().serialize(serializer);
        }
        pack(value).serialize(serializer)
    }

    pub fn deserialize<'de, D>(
        deserializer: D,
    ) -> Result<Arc<BTreeMap<u32, Vec<u8>>>, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        if !PACKED.with(Cell::get) {
            return Arc::<BTreeMap<u32, Vec<u8>>>::deserialize(deserializer);
        }
        let profile = std::env::var_os("GLRMASK_PROFILE_SERIALIZATION").is_some();
        let total = profile.then(std::time::Instant::now);
        let packed_started = profile.then(std::time::Instant::now);
        let packed = Vec::<u8>::deserialize(deserializer)?;
        let packed_len = packed.len();
        let packed_ms = packed_started.map_or(0.0, |s| s.elapsed().as_secs_f64() * 1000.0);
        if EXTERNAL.with(Cell::get) {
            if !packed.is_empty() {
                return Err(serde::de::Error::custom(
                    "external packed token-byte placeholder must be empty",
                ));
            }
            return Ok(Arc::new(BTreeMap::new()));
        }
        let unpack_started = profile.then(std::time::Instant::now);
        let result = if DEFER_UNPACK.with(Cell::get) {
            let deferred = PackedTokenBytes::parse(packed)
                .map(Arc::new)
                .map_err(serde::de::Error::custom)?;
            DEFERRED.with(|slot| *slot.borrow_mut() = Some(deferred));
            Ok(Arc::new(BTreeMap::new()))
        } else {
            unpack(&packed).map_err(serde::de::Error::custom)
        };
        if let Some(total) = total {
            eprintln!(
                "[glrmask/profile][token_bytes_decode] wire_bytes={} vec_ms={packed_ms:.3} unpack_ms={:.3} total_ms={:.3}",
                packed_len,
                unpack_started.map_or(0.0, |s| s.elapsed().as_secs_f64() * 1000.0),
                total.elapsed().as_secs_f64() * 1000.0,
            );
        }
        result
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn check_roundtrip(value: BTreeMap<u32, Vec<u8>>) {
            let packed = pack(&value);
            assert!(packed.starts_with(INDEXED_MAGIC));
            let view = PackedTokenBytes::parse(packed).unwrap();
            assert_eq!(view.len(), value.len());
            assert_eq!(
                view.iter()
                    .map(|(id, bytes)| (id, bytes.to_vec()))
                    .collect::<BTreeMap<_, _>>(),
                value
            );
            for (&id, bytes) in &value {
                assert_eq!(view.get(id), Some(bytes.as_slice()));
            }
            assert_eq!(view.max_token_id(), value.keys().next_back().copied());

            let legacy = pack_legacy(&value);
            let legacy_view = PackedTokenBytes::parse(legacy).unwrap();
            assert_eq!(
                legacy_view
                    .iter()
                    .map(|(id, bytes)| (id, bytes.to_vec()))
                    .collect::<BTreeMap<_, _>>(),
                value
            );
        }

        #[test]
        fn indexed_token_bytes_roundtrip_dense_and_sparse() {
            check_roundtrip(BTreeMap::from([
                (0, b"a".to_vec()),
                (1, b"bc".to_vec()),
                (2, Vec::new()),
            ]));
            check_roundtrip(BTreeMap::from([
                (2, b"a".to_vec()),
                (9, b"bc".to_vec()),
                (1000, b"xyz".to_vec()),
            ]));
        }
    }
}

#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize,
)]
pub(crate) enum ConstraintRuntimeBackend {
    #[default]
    Static,
    Dynamic,
}

pub(crate) type SegmentedParserLink = ScopedSubgrammarLink;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RecursiveParserLeafLayout {
    pub(crate) state_offset: u32,
    pub(crate) state_count: u32,
    /// Immediate wrapper of the composition whose layout was requested. Inner
    /// leaf changes within one nested component deliberately keep this owner.
    pub(crate) top_component: u32,
    /// Immediate-component path from the requested composition root to this
    /// intact LR table.
    pub(crate) component_path: Vec<u32>,
}

const RECURSIVE_VIRTUAL_TOKENIZER_STATE_LIMIT: u32 = 1 << 31;

#[derive(Debug, Default)]
struct RecursiveVirtualTokenizerStateStore {
    next_state: u32,
    scoped_by_local: FxHashMap<(u32, u32), u32>,
    local_by_scoped: FxHashMap<u32, (u32, u32)>,
}

#[derive(Debug, Default)]
pub(crate) struct RecursiveVirtualTokenizerStates {
    store: Mutex<RecursiveVirtualTokenizerStateStore>,
}

impl RecursiveVirtualTokenizerStates {
    pub(crate) fn scoped_state(
        &self,
        physical_state_count: u32,
        leaf_index: usize,
        local_state: u32,
    ) -> Option<u32> {
        let leaf_index = u32::try_from(leaf_index).ok()?;
        let mut store = self.store.lock().ok()?;
        if let Some(&scoped) = store.scoped_by_local.get(&(leaf_index, local_state)) {
            return Some(scoped);
        }
        if store.next_state < physical_state_count {
            store.next_state = physical_state_count;
        }
        if store.next_state >= RECURSIVE_VIRTUAL_TOKENIZER_STATE_LIMIT {
            return None;
        }
        let scoped = store.next_state;
        store.next_state = store.next_state.checked_add(1)?;
        store.scoped_by_local.insert((leaf_index, local_state), scoped);
        store.local_by_scoped.insert(scoped, (leaf_index, local_state));
        Some(scoped)
    }

    pub(crate) fn local_state(&self, scoped_state: u32) -> Option<(usize, u32)> {
        let store = self.store.lock().ok()?;
        let &(leaf_index, local_state) = store.local_by_scoped.get(&scoped_state)?;
        Some((leaf_index as usize, local_state))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RecursiveParserLayout {
    pub(crate) component_offsets: Vec<u32>,
    pub(crate) leaves: Vec<RecursiveParserLeafLayout>,
    pub(crate) leaf_state_offsets: Vec<u32>,
    /// Disjoint-union tokenizer-state coordinate over the same intact leaves.
    /// Parser and tokenizer leaves have identical ordering/component paths, but
    /// independent offsets because their local state counts differ.
    pub(crate) leaf_tokenizer_state_offsets: Vec<u32>,
    pub(crate) total_tokenizer_states: u32,
    /// Disjoint-union terminal coordinate over the same intact leaves. Runtime
    /// byte terminals are encoded after the outer/materialized terminal range,
    /// so the `u32` is self-describing while byte scanning no longer needs a
    /// leaf-local -> outer -> leaf-local round trip.
    pub(crate) leaf_terminal_offsets: Vec<u32>,
    pub(crate) total_leaf_terminals: u32,
    /// Size of the transitional outer/global terminal prefix at layout-build
    /// time. Live recursive terminal IDs use this immutable tag boundary and
    /// therefore do not need to consult the materialized outer table again.
    pub(crate) outer_terminal_count: u32,
    /// Lazily mapped future-terminal support in the live runtime terminal
    /// coordinate for each scoped leaf tokenizer state.
    pub(crate) tokenizer_future_scoped: Vec<OnceLock<BitSet>>,
    /// Linker controls rewritten only into the leaf-component coordinate.
    pub(crate) links: Vec<SegmentedParserLink>,
    /// Outer/global terminal -> intact leaf-local terminals. Ordinary byte
    /// commit no longer consumes this relation; it remains for compiler-side
    /// static-B materialization, compatibility/reference evaluation, and outer
    /// special-token routing.
    pub(crate) terminal_targets: Vec<SmallVec<[(u32, TerminalID); 4]>>,
    pub(crate) total_states: u32,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct StaticDynamicOverlayMetadata {
    /// Global terminal-id offsets for the transported composition components.
    pub(crate) terminal_offsets: Vec<u32>,
    /// Global raw-tokenizer-state offsets for those same components. State zero
    /// is the merged reset dispatcher and deliberately belongs to no component.
    pub(crate) tokenizer_state_offsets: Vec<u32>,
    /// Terminals whose composed parser template has behavior absent from the
    /// transported component parser artifacts (including scoped-ignore repair
    /// and conservative unsafe terminals).
    pub(crate) repair_terminals: Vec<bool>,
    /// Composed LR states which belong to one or more child components but not
    /// to the parent component. Runtime lookahead-return factoring is useful
    /// only while the concrete top state is still inside such a child-owned
    /// region; ordinary parent reductions must not pay for that machinery.
    #[serde(default)]
    pub(crate) non_parent_only_parser_states: Vec<bool>,
    /// Exact segmented parser backend. Each retained source constraint keeps
    /// its own parser/token coordinate and is projected from the composed
    /// tokenizer/LR coordinates at mask time. Current artifacts flatten static
    /// components into one parser DWA but retain dynamic/hybrid components.
    #[serde(skip, default)]
    pub(crate) segmented_parser_components: Vec<SegmentedParserComponent>,
    /// Explicit local-coordinate calls between the wrappers above. This is the
    /// semantic relation consumed by the compact parser-action provider; it is
    /// intentionally scoped to this composition level only.
    #[serde(skip, default)]
    pub(crate) segmented_parser_links: Vec<SegmentedParserLink>,
    /// Legacy v24 prefix offsets for the old immediate-component compact parser
    /// coordinate. New recursive runtimes derive wrapper/leaf intervals from
    /// `segmented_parser_components` and leave this empty; retained only for
    /// loading older current-version artifacts and legacy experiment paths.
    #[serde(skip, default)]
    pub(crate) segmented_parser_state_offsets: Vec<u32>,
    /// Runtime-derived endpoint parser view. The semantic component tree stays
    /// literal; this cache contains only the flat leaf coordinate required by
    /// the existing `LeveledGSS<u32, ...>` action provider.
    #[serde(skip, default)]
    pub(crate) recursive_parser_layout: OnceLock<Arc<RecursiveParserLayout>>,
    /// Exact packed flattened parser table retained only for future compiler
    /// rebinding. Live recursive execution never reads it. The coordinator's
    /// ordinary `table` field is reduced to a grammar shell after compilation;
    /// late composition materializes this blob into a temporary compiler clone
    /// and detaches it again before publishing runtime components.
    #[serde(skip, default)]
    pub(crate) recursive_compiler_table: OnceLock<Arc<[u8]>>,
    /// Exact endpoint lexical relation: recursive leaf tokenizer state -> this
    /// constraint's internal TSID set. Runtime-product components can make the
    /// image genuinely set-valued, so this must not be collapsed to one TSID.
    #[serde(skip, default)]
    pub(crate) recursive_tokenizer_internal_tsids: OnceLock<Arc<Vec<Vec<u32>>>>,
    /// Lazily allocated outer scoped IDs for exact virtual tokenizer states in
    /// retained recursive leaves. Physical states keep their contiguous layout
    /// IDs; only actually reached virtual states enter this runtime-only map.
    #[serde(skip, default)]
    pub(crate) recursive_virtual_tokenizer_states: Arc<RecursiveVirtualTokenizerStates>,
    /// The segmented A/B factorization is the masking implementation for this
    /// constraint, rather than an optional validation view of a flattened
    /// parser DWA. Current serialization preserves this split explicitly.
    #[serde(skip, default)]
    pub(crate) segmented_mask_authoritative: bool,
    /// A serialized hybrid may flatten only its static component contribution
    /// into `Constraint::parser_dwa` while retaining dynamic components below.
    /// When set, segmented masking starts from that static parser-DWA mask and
    /// ORs the retained dynamic components plus boundary B.
    #[serde(skip, default)]
    pub(crate) segmented_static_baseline: bool,
    /// Compressed deterministic union root for `segmented_parser_components`.
    /// Entry `g` is the unique component selected by composed LR state `g`, or
    /// `u32::MAX` when no component has a root transition on that state. When
    /// non-empty, the component collection is one deterministic parser DWA in
    /// segmented storage: a synthetic root followed by one cached component
    /// body. No runtime parser-NWA branching is involved.
    #[serde(skip, default)]
    pub(crate) segmented_component_union_root_dispatch: Vec<u32>,
    /// Authoritative cross-component accelerators, optionally restricted to
    /// model tokens whose parser execution starts in one component.  This is
    /// the new composition model: boundary acceleration is selected per
    /// starting component, while the retained component constraints remain
    /// the ordinary A contribution.  `start_component == None` is the exact
    /// legacy/global fallback used while older artifacts and compiler paths
    /// are migrated.
    #[serde(skip, default)]
    pub(crate) segmented_boundary_shards: Vec<SegmentedBoundaryShard>,
    /// Legacy/global v22 storage.  New live compositions also publish an Arc
    /// to the same backend through `segmented_boundary_shards`; these fields
    /// remain until the partitioned boundary wire format is versioned.
    #[serde(skip, default)]
    pub(crate) segmented_boundary_parser: Option<Arc<SegmentedBoundaryParser>>,
    #[serde(skip, default)]
    pub(crate) segmented_boundary_terminal_trie: Option<Arc<SegmentedBoundaryTerminalTrie>>,
}

#[derive(Debug, Clone)]
pub(crate) struct SegmentedBoundaryShard {
    /// Component in which the model token starts.
    pub(crate) start_component: u32,
    /// Exact composed LR top states that can represent this starting
    /// component. A state may intentionally occur in more than one shard when
    /// the composed relation is non-functional; evaluating both shards is then
    /// conservative and exact under OR.
    pub(crate) start_parser_states: BitSet,
    /// Whether an empty composed parser stack belongs to this shard. Only the
    /// outer/root component normally owns that coordinate.
    pub(crate) accepts_empty_stack: bool,
    /// Optional conservative model-token summary for the first internal
    /// component crossing. `None` means no trigger information is available;
    /// it must never be interpreted as "no crossings".
    pub(crate) candidate_tokens: Option<Arc<[u32]>>,
    /// Runtime-only immutable vocabulary scoped to this binding. Never key a
    /// global cache by model IDs alone: IDs can spell different bytes and LR
    /// coordinates can differ in another binding. Rebuilt on artifact load.
    pub(crate) mask_vocabulary: Arc<OnceLock<Result<super::dynamic_mask::PreparedMaskVocabulary, String>>>,
    pub(crate) backend: SegmentedBoundaryShardBackend,
}

#[derive(Debug, Clone)]
pub(crate) enum SegmentedBoundaryShardBackend {
    StaticParser(Arc<SegmentedBoundaryParser>),
    /// Preserved terminal-language/NWA crossing accelerator used by historical
    /// artifacts and the explicit experimental runtime path. Normal current
    /// dynamic composition publishes `DynamicDirect` instead.
    DynamicTerminalTrie(Arc<SegmentedBoundaryTerminalTrie>),
    /// Exact dynamic crossing with no required composition-specific B artifact.
    /// Materialized runtimes use the strict dynamic full-vocabulary walker;
    /// recursive runtimes use the scoped shared-prefix vocabulary walker.
    DynamicDirect,
}

#[derive(Debug, Clone)]
pub(crate) struct SegmentedParserComponent {
    pub(crate) constraint: Arc<Constraint>,
    /// Composition-specific crossing backend for model tokens that start in
    /// this component. This lives on the wrapper, not on the reusable
    /// underlying constraint.
    pub(crate) boundary: Option<SegmentedBoundaryShard>,
    pub(crate) tokenizer_state_offset: u32,
    pub(crate) terminal_offset: u32,
    /// Outer composed terminals which are canonical aliases for a local
    /// terminal of this component. Direct interval mapping remains implicit;
    /// this stores only exceptional many-scope aliases (currently globally
    /// equivalent ignore terminals).
    pub(crate) global_terminal_aliases: Vec<(u32, u32)>,
    /// Component-local Static TSID -> composed Static TSID relation retained
    /// for lazy exact tokenizer states that are allocated after link time.
    pub(crate) local_tsid_to_global_tsids: Vec<Vec<u32>>,
    /// Legacy v22 compatibility metadata. New authoritative A+B compositions
    /// leave this `None`: component parser DWAs keep their standalone semantics
    /// unchanged, and scope/link behavior lives in the composed parser view/B.
    pub(crate) root_disallowed_terminal: Option<u32>,
    pub(crate) global_to_local_parser_state: Vec<u32>,
}


#[derive(Clone, Copy)]
pub(crate) struct SegmentedParserComponentTables<'a> {
    components: &'a [SegmentedParserComponent],
}

impl<'a> SegmentedParserComponentTables<'a> {
    #[inline]
    pub(crate) fn new(components: &'a [SegmentedParserComponent]) -> Self {
        Self { components }
    }
}

impl ParserComponentTableSource for SegmentedParserComponentTables<'_> {
    #[inline]
    fn component_count(&self) -> usize {
        self.components.len()
    }

    #[inline]
    fn component_table(&self, component: u32) -> Option<&GLRTable> {
        self.components
            .get(component as usize)
            .map(|component| &*component.constraint.table)
    }

    #[inline]
    fn component_ignore_terminal(&self, component: u32) -> Option<u32> {
        let component = self.components.get(component as usize)?;
        // A composed component table already owns its scoped ignores in its
        // rows (`skip_terminals` provenance); provider-level Identity would
        // shadow those rows. Only raw component tables need it here.
        let ignore = component.constraint.ignore_terminal?;
        (!component.constraint.table.skip_terminals.contains(&ignore)).then_some(ignore)
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct BoundaryTerminalTrieNode {
    pub(crate) children: Vec<(u32, u32)>,
    /// Legacy v21 representation: private boundary token classes accepted at
    /// this node after expanding the TSID dimension during construction.
    pub(crate) outputs: Vec<u32>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct BoundaryTerminalNwaTransition {
    pub(crate) terminal: u32,
    pub(crate) target: u32,
    pub(crate) weight: Weight,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct BoundaryTerminalNwaNode {
    pub(crate) final_weight: Option<Weight>,
    pub(crate) transitions: Vec<BoundaryTerminalNwaTransition>,
    pub(crate) epsilons: Vec<(u32, Weight)>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct BoundaryTerminalNwa {
    pub(crate) nodes: Vec<BoundaryTerminalNwaNode>,
    pub(crate) start_states: Vec<u32>,
    /// A topological order of `nodes`, validated when the artifact is loaded.
    /// Runtime evaluation uses it to coalesce equal token domains at converged
    /// NWA states without materializing the exponentially large prefix trie.
    pub(crate) topological_order: Vec<u32>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct SegmentedBoundaryTerminalTrie {
    pub(crate) nodes: Vec<BoundaryTerminalTrieNode>,
    pub(crate) root_by_tsid: Vec<u32>,
    pub(crate) tokenizer_state_to_tsid: Vec<u32>,
    pub(crate) internal_token_to_originals: Vec<Vec<u32>>,
    /// Current representation. The legacy v21 serde shape above is retained so
    /// old artifacts still decode; v22 persists this DAG explicitly.
    #[serde(skip, default)]
    pub(crate) symbolic_nwa: Option<BoundaryTerminalNwa>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct SegmentedBoundaryParser {
    /// Generic wire/reference representation. Compact in-memory boundary
    /// parsers leave this as the empty one-state DWA and use
    /// `compact_parser_dwa`; V17 serialization materializes the compact machine
    /// back into this exact generic wire shape.
    pub(crate) parser_dwa: DWA,
    #[serde(skip, default)]
    pub(crate) compact_parser_dwa:
        Option<crate::compiler::stages::parser_dwa::SmallBoundaryDwa>,
    /// Provider-native static boundary parser in the recursive leaf parser-state
    /// coordinate. New v25 compositions compile and serialize this coordinate
    /// directly. Legacy v24 static boundaries retain only `parser_dwa` and stay
    /// on their materialized runtime when loaded.
    #[serde(skip, default)]
    pub(crate) recursive_parser_dwa: Option<DWA>,
    /// New static boundary shards are compiled directly in the authoritative
    /// composed constraint TSID coordinate. They therefore need no private
    /// raw-tokenizer-state map; runtime reads `Constraint::state_to_internal_tsid`
    /// instead. Legacy/global boundary artifacts keep this false and retain the
    /// explicit private map below.
    #[serde(skip, default)]
    pub(crate) uses_composed_tsid_coordinate: bool,
    pub(crate) tokenizer_state_to_tsid: Vec<u32>,
    pub(crate) internal_token_to_originals: Vec<Vec<u32>>,
}

#[derive(Debug, Clone)]
pub(crate) enum DeferredTerminalExprBytes {
    Owned(Arc<[u8]>),
    Backed {
        backing: Arc<Vec<u8>>,
        start: usize,
        len: usize,
    },
    /// Dynamic worker-transfer artifacts keep the terminal-expression section
    /// compressed until a later composition actually asks for source
    /// expressions. Ordinary mask/commit execution never needs these trees.
    CompressedOwned(Arc<[u8]>),
    /// Same compressed representation, retained directly inside an owned
    /// sectioned transfer backing allocation.
    CompressedBacked {
        backing: Arc<Vec<u8>>,
        start: usize,
        len: usize,
    },
}

impl DeferredTerminalExprBytes {
    #[inline]
    pub(crate) fn as_slice(&self) -> &[u8] {
        match self {
            Self::Owned(bytes) => bytes,
            Self::Backed {
                backing,
                start,
                len,
            } => &backing[*start..*start + *len],
            Self::CompressedOwned(bytes) => bytes,
            Self::CompressedBacked {
                backing,
                start,
                len,
            } => &backing[*start..*start + *len],
        }
    }

    pub(crate) fn decode_exprs(&self) -> Result<Vec<Expr>, String> {
        let raw;
        let bytes = match self {
            Self::Owned(bytes) => bytes.as_ref(),
            Self::Backed {
                backing,
                start,
                len,
            } => &backing[*start..*start + *len],
            Self::CompressedOwned(bytes) => {
                raw = zstd::stream::decode_all(bytes.as_ref()).map_err(|err| err.to_string())?;
                raw.as_slice()
            }
            Self::CompressedBacked {
                backing,
                start,
                len,
            } => {
                raw = zstd::stream::decode_all(&backing[*start..*start + *len])
                    .map_err(|err| err.to_string())?;
                raw.as_slice()
            }
        };
        bincode::deserialize(bytes).map_err(|err| err.to_string())
    }

    /// Append the canonical uncompressed bincode expression section used by
    /// the self-contained Constraint artifact format. A transfer-loaded
    /// compressed blob therefore remains lazy until either composition or a
    /// later explicit save requests it.
    pub(crate) fn append_raw_serialized(&self, out: &mut Vec<u8>) -> Result<(), String> {
        match self {
            Self::Owned(bytes) => out.extend_from_slice(bytes),
            Self::Backed {
                backing,
                start,
                len,
            } => out.extend_from_slice(&backing[*start..*start + *len]),
            Self::CompressedOwned(bytes) => {
                zstd::stream::copy_decode(bytes.as_ref(), out).map_err(|err| err.to_string())?;
            }
            Self::CompressedBacked {
                backing,
                start,
                len,
            } => {
                zstd::stream::copy_decode(&backing[*start..*start + *len], out)
                    .map_err(|err| err.to_string())?;
            }
        }
        Ok(())
    }
}

/// Opaque current-format composition metadata retained without eagerly
/// rebuilding the large parser-template/characterization graphs. Ordinary
/// runtime masking never needs these bytes; constraint composition materializes
/// them on demand.
#[derive(Debug, Clone)]
pub(crate) enum DeferredCompositionMetadataBytes {
    Owned(Arc<[u8]>),
    Backed {
        backing: Arc<Vec<u8>>,
        start: usize,
        len: usize,
    },
}

impl DeferredCompositionMetadataBytes {
    #[inline]
    pub(crate) fn as_slice(&self) -> &[u8] {
        match self {
            Self::Owned(bytes) => bytes,
            Self::Backed {
                backing,
                start,
                len,
            } => &backing[*start..*start + *len],
        }
    }
}

/// Compact persisted metadata for an unresolved external-grammar slot.
///
/// The terminal ID is an internal linker coordinate, never a public/model
/// token ID. Keeping this record beside the compiled parser artifact lets a
/// loaded parent be linked without parsing or recompiling its source grammar.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct LateGrammarSlot {
    pub(crate) name: String,
    pub(crate) terminal_id: TerminalID,
}

/// Fully compiled, immutable grammar constraint.
///
/// A `Constraint` is intended to be reused across generated sequences. Call
/// [`Constraint::start`] to create a mutable per-sequence state.
#[derive(Debug, Clone)]
pub struct Constraint {
    /// Final generation policy, not part of the embeddable compiled body.
    /// The outer root artifact stores these separately from compiler data.
    pub(crate) end_tokens: Arc<[u32]>,
    pub(crate) runtime_backend: ConstraintRuntimeBackend,
    pub(crate) static_dynamic_overlay: Option<StaticDynamicOverlayMetadata>,
    /// Reusable component-local trigger metadata for dynamic composition.
    /// Ordinary static/dynamic compilation leaves this at `None` so there is
    /// no trigger construction cost unless explicitly requested in the future.
    pub(crate) boundary_trigger: BoundaryTrigger,
    /// Grammar-aware proper-prefix model-token summary used only by static
    /// composition boundary preparation.  It is intentionally independent of
    /// `boundary_trigger`: the latter is a dynamic-runtime accelerator with a
    /// different observation contract.  The lock permits lazy preparation by
    /// composition without mutating the immutable constraint API.
    pub(crate) boundary_candidate_summary: OnceLock<BoundaryCandidateSummary>,
    /// Named, compiler-generated linker terminals for unresolved
    /// `extern grammar` declarations. The token IDs backing these terminals
    /// are deliberately private and outside the model vocabulary; callers
    /// address slots only by `name` through the late-binding API.
    pub(crate) late_grammar_slots: Vec<LateGrammarSlot>,
    /// Runtime-only public vocabulary reconstructed from this immutable
    /// constraint for late subgrammar binding. Keeping it here lets repeated
    /// binds share `Vocab`'s pure derived-artifact cache (adjacent-byte index,
    /// tries, etc.) instead of rebuilding those structures for every child.
    pub(crate) late_bind_vocab: OnceLock<crate::Vocab>,
    /// Runtime-derived exact original-token sets for `Skip` terminals in a
    /// composed grammar. Each token is wholly in `L(skip)+`: it can be
    /// consumed as one or more complete instances of that scoped-ignore
    /// terminal with a lexer reset between instances. This is deliberately
    /// not serialized; it is cheap to rebuild from the retained terminal
    /// expression and vocabulary and therefore does not change artifact wire
    /// compatibility.
    pub(crate) scoped_ignore_only_tokens: Vec<(TerminalID, Box<[u32]>)>,
    /// Exact byte-token fusions `(fused, suffix)` grouped by scoped Skip. The
    /// fused token begins with one or more complete instances of the Skip
    /// language and the remaining bytes equal `suffix` exactly. If `suffix`
    /// is admitted by the ordinary static mask, `fused` is therefore admitted
    /// as well. Runtime-only for the same wire-compatibility reason as above.
    pub(crate) scoped_ignore_prefix_fusions: Vec<(TerminalID, Box<[(u32, u32)]>)>,
    pub(crate) parser_dwa: DWA,
    /// Current-format loaded constraints retain the immutable parser DWA in
    /// its compact canonical pools instead of reconstructing RangeSet/Weight
    /// objects. Compiler-created and legacy-loaded constraints leave this
    /// empty and use `parser_dwa` directly.
    pub(crate) packed_parser_dwa:
        Option<Arc<crate::automata::weighted::dwa::PackedRuntimeDwa>>,
    /// Runtime-only override for the parser-DWA start final. Composition uses
    /// this to suppress a component's standalone globally-erased ignore at the
    /// union root without materializing or mutating the full parser DWA.
    pub(crate) parser_start_final_override: Option<Weight>,
    /// Exact depth-one parser acceptance kept separate from the deeper parser
    /// DWA. Keys are encoded parser-state labels; values are already the
    /// transition/final-weight intersection for accepting after that one
    /// stack symbol.
    pub(crate) parser_top_accept: BTreeMap<i32, Weight>,
    /// Uncombined exact depth-one acceptance parts. Direct-regular grammars
    /// retain terminal completion weights separately to avoid constructing one
    /// large union weight per parser state at compile time.
    pub(crate) parser_top_accept_parts: BTreeMap<i32, Vec<Weight>>,
    /// Immediate-completion L1 terminal weights for direct-regular parsers.
    /// Kept once per grammar terminal rather than duplicated across every
    /// epsilon-closed parser row.
    pub(crate) direct_regular_l1_complete_by_terminal: BTreeMap<TerminalID, Weight>,
    pub(crate) packed_non_dwa_weights: Option<Arc<PackedNonDwaWeights>>,
    /// Runtime-derived exact acceptance summaries for wide direct-regular
    /// replace-top frontiers. Rebuilt after compile/load from the table and
    /// parser-top acceptance artifacts.
    pub(crate) direct_regular_wide_frontier_acceptance:
        Vec<DirectRegularWideFrontierAcceptance>,
    /// Runtime-only exact transition maps for the direct automaton's initial
    /// frontier and its single widest successor frontier. Dynamic masking
    /// repeatedly queries these two frontiers at token boundaries.
    pub(crate) direct_regular_dynamic_hot_frontiers:
        Vec<DirectRegularDynamicHotFrontier>,
    /// Runtime-derived exact dense acceptance for the broadest direct-regular
    /// parser row(s). This avoids replaying thousands of L1 terminal weights on
    /// every mask while keeping the cached result source-state exact.
    pub(crate) direct_regular_parser_state_acceptance:
        Vec<DirectRegularParserStateAcceptance>,
    /// Sparse terminal-level automaton retained for exact direct-regular
    /// runtime indexes. Static artifact format versioning covers this field.
    pub(crate) direct_regular_automaton: Option<DirectRegularAutomaton>,
    pub(crate) table: crate::runtime::parser_backend::ParserTableStorage,
    pub(crate) terminal_display_names: Vec<String>,
    pub(crate) tokenizer: Arc<Tokenizer>,
    pub(crate) boundary_completion_index: Option<Arc<crate::compiler::boundary_precomputed_completion::PreparedCompletion>>,
    /// Cached tokenizer topology flag. `Tokenizer::has_epsilon_transitions()`
    /// scans every tokenizer state, so runtime dispatch must not recompute it.
    pub(crate) tokenizer_has_epsilon_transitions: bool,
    pub(crate) ignore_terminal: Option<TerminalID>,
    pub(crate) special_token_terminals: Vec<SpecialTokenTerminal>,

    /// Runtime-only vocabulary data for direct dynamic masking.
    pub(crate) dynamic_mask_vocab: DynamicMaskVocab,
    /// Lazily materialized static-mode fallback vocabulary. Ordinary static
    /// masking never touches this; it is initialized only if an empty
    /// possible-matches table encounters a token-start exclusion.
    pub(crate) lazy_dynamic_mask_vocab: OnceLock<DynamicMaskVocab>,
    /// Original IDs with an empty byte spelling. Derived at compile/load,
    /// never by constructing the full vocabulary trie on a mask call. Shared
    /// with clones; special-token admission is checked against the live state.
    pub(crate) empty_byte_token_ids: Arc<[u32]>,

    /// possible_matches keyed by grammar terminal id.
    ///
    /// An empty table may represent deferred possible-match construction in
    /// legacy code only.
    ///
    /// IMPORTANT: the dynamic possible-matches fallback is intentionally
    /// terrible and is planned for removal. New compiler paths MUST construct
    /// complete exact possible matches and MUST NOT set
    /// `possible_matches_complete` to false as an implementation shortcut.
    /// DO NOT REMOVE OR WEAKEN THIS COMMENT.
    ///
    /// Each Weight maps final shared internal tokenizer-state ids to token sets
    /// in the final shared constraint-internal vocab space. Parser-DWA weights
    /// and possible_matches weights are reconciled into this same space during
    /// compilation.
    pub(crate) possible_matches: PossibleMatchesByTerminal,
    /// Whether `possible_matches` is a complete table. New static constraints
    /// must set this to true. False exists only for legacy dynamic/deferred
    /// construction and is not permitted as a fallback strategy for new
    /// compiler features.
    pub(crate) possible_matches_complete: bool,
    pub(crate) state_to_internal_tsid: Vec<u32>,
    pub(crate) internal_tsid_to_states: Vec<Vec<u32>>,
    /// Ordinary tokenizers have one internal TSID per physical state, making
    /// `internal_tsid_to_states` the exact bucket inverse of
    /// `state_to_internal_tsid`. Current artifacts can omit that redundant
    /// allocation and reconstruct it only for composition/debug paths.
    pub(crate) deferred_internal_tsid_to_states: OnceLock<Vec<Vec<u32>>>,
    /// Composition-preparation cache: row `t` lists original model-token IDs
    /// which, from this component's lexer reset, complete terminal `t` exactly
    /// at the end of the model token.  This is not part of the historical inner
    /// `Constraint` bincode layout; artifact V13 stores it in the outer
    /// envelope so V12 constraints remain loadable unchanged.
    pub(crate) composition_reset_tokens_by_terminal: Vec<Vec<u32>>,
    /// Named unresolved `extern grammar` slots retained by a compiled parent.
    /// Values are parent-local hidden placeholder terminal IDs. Stored in the
    /// outer composition metadata so cached parents can be rebound after load.
    pub(crate) unbound_grammar_placeholders: BTreeMap<String, TerminalID>,
    /// Composition-time parser stack-effect templates retained from the
    /// original compile. These are the unspecialized per-terminal DFAs used to
    /// build parser DWAs, so a later linker can transport unchanged component
    /// behavior instead of re-characterizing the component LR table.
    /// Stored in the outer versioned artifact envelope for compatibility with
    /// older inner `Constraint` bincode layouts.
    pub(crate) composition_parser_templates_by_terminal: Vec<Option<UnweightedDfa>>,
    /// Composition-time symbolic parser characterizations retained from the
    /// original compile. A later linker can append only the boundary-induced
    /// reductions/rereductions and recompile affected terminal templates,
    /// rather than re-solving the component's reduction closure from scratch.
    pub(crate) composition_parser_characterizations_by_terminal:
        Vec<Option<TerminalCharacterization>>,
    /// Composition-time grammar adjacency summary. Stored in the outer
    /// versioned artifact envelope so older inner `Constraint` layouts remain
    /// loadable unchanged.
    pub(crate) composition_grammar_summary: Option<CompositionGrammarSummary>,
    /// Runtime-only inverse lexer-metadata index used by compiled-constraint
    /// composition. Row `t` lists exactly the raw tokenizer states whose
    /// epsilon closure has terminal `t` matched or still reachable.
    pub(crate) terminal_live_states: Vec<Vec<u32>>,
    /// Runtime-only CSR view of the exact state -> internal-TSID relation.
    /// Ordinary tokenizers have one entry per state. A fully determinized
    /// runtime lexer may represent several old lexer states and therefore
    /// several independent TSID lanes in one physical state.
    pub(crate) state_internal_tsid_offsets: Vec<u32>,
    pub(crate) state_internal_tsids: Vec<u32>,
    /// Final-runtime subset states followed by an exact copy of the source
    /// tokenizer. `runtime_source_state_offset` is the boundary between the
    /// two coordinates. Empty metadata means no runtime-only determinization.
    pub(crate) runtime_source_state_offset: Option<u32>,
    /// CSR offsets for product-state -> exact source-state subset. There is one
    /// row per product state and therefore `product_state_count + 1` offsets.
    pub(crate) runtime_product_source_offsets: Vec<u32>,
    pub(crate) runtime_product_source_states: Vec<u32>,
    /// Scalar source representative for product states that are exactly one
    /// source state's epsilon closure; `u32::MAX` otherwise.
    pub(crate) runtime_product_exact_source_states: Vec<u32>,
    /// Runtime-only inverse used to re-coalesce a uniform source frontier.
    pub(crate) runtime_product_state_by_source_subset: FxHashMap<Box<[u32]>, u32>,
    pub(crate) template_dfas_by_terminal: TemplateDfasByTerminal,
    /// Runtime-only compact transition view for commit template products.
    pub(crate) fast_template_dfas_by_terminal: FastTemplateDfasByTerminal,
    pub(crate) template_parser: Option<Arc<crate::runtime::parser_backend::TemplateParser>>,
    /// Original token -> final shared constraint-internal token id.
    ///
    /// This is not necessarily equal to the parser-DWA compaction vocab map
    /// produced before possible-match reconciliation. It may contain additional
    /// splits required by possible_matches.
    pub(crate) original_token_to_internal: Vec<u32>,
    /// Current-format loads retain the fixed-width original-token map inside
    /// the owned artifact instead of expanding all model-token entries to
    /// `u32`. Ordinary static mask/commit performs direct packed lookups; only
    /// composition/debug-style bulk access materializes the vector lazily.
    pub(crate) packed_original_token_to_internal:
        Option<Arc<original_token_map_artifact_serde::PackedOriginalTokenMap>>,
    pub(crate) deferred_original_token_to_internal: OnceLock<Vec<u32>>,
    /// Final shared constraint-internal token id -> original token ids.
    ///
    /// Parser-DWA weights and Constraint.possible_matches bitmaps both use these
    /// final internal token ids.
    pub(crate) internal_token_to_tokens: Vec<Vec<u32>>,
    /// Current-format loads can defer reconstructing the explicit inverse of
    /// `original_token_to_internal`. Static mask/commit only needs the internal
    /// token count and the already-serialized mask fragments; composition and
    /// token-space expansion materialize this inverse on first use.
    pub(crate) deferred_internal_token_to_tokens: OnceLock<Vec<Vec<u32>>>,
    pub(crate) token_bytes: Arc<BTreeMap<u32, Vec<u8>>>,
    /// Indexed immutable vocabulary used directly by runtime token lookup and
    /// iteration. Loaded constraints can point into artifact backing; compiled
    /// constraints own the same indexed representation alongside the source
    /// map used by compiler/composition code.
    pub(crate) packed_token_bytes: Option<Arc<token_bytes_artifact_serde::PackedTokenBytes>>,
    // Compiler-side scratch/result metadata only. No runtime or composition
    // path reads this field; composition rebuilds the map for its result when
    // needed. Persisting it duplicated token bytes inside every constraint.
    pub(crate) internal_token_bytes: BTreeMap<u32, Vec<u8>>,
    pub(crate) token_bytes_dense: Vec<Option<Box<[u8]>>>,

    /// Precomputed bitmask fragments for each internal token.
    /// `internal_token_buf_masks[i]` contains (word_index, or_mask) pairs
    /// for all original tokens that map to internal token `i`.
    pub(crate) internal_token_buf_masks: Vec<InternalTokenBufMasks>,
    /// Precomputed combined buf output for each group of 64 internal tokens.
    /// `word_group_buf_masks[w]` is the combined mask for internal tokens [w*64 .. (w+1)*64).
    /// Used as a fast path in `or_to_buf` when a dense word is all-ones (!0u64).
    pub(crate) word_group_buf_masks: Vec<Box<[u32]>>,
    /// Precomputed dense output masks for groups of 128 internal tokens.
    pub(crate) pair_word_group_buf_masks: DenseBufMaskRows,
    /// Precomputed dense output masks for groups of 256 internal tokens.
    pub(crate) quad_word_group_buf_masks: DenseBufMaskRows,
    /// Precomputed dense output masks for groups of 512 internal tokens.
    pub(crate) super_word_group_buf_masks: DenseBufMaskRows,
    /// Precomputed dense output masks for groups of 1024 internal tokens.
    pub(crate) mega_word_group_buf_masks: DenseBufMaskRows,
    /// Precomputed dense output masks for groups of 2048 internal tokens.
    pub(crate) giga_word_group_buf_masks: DenseBufMaskRows,
    /// Sparse OR-union for each 64-token internal word group.
    pub(crate) word_group_sparse_masks: Vec<InternalTokenBufMasks>,
    /// Dense prefix-unions of 64-token internal word groups.
    ///
    /// `word_group_prefix_buf_masks[i]` is the OR-union of word groups
    /// `[0, i)`. Internal-token groups are disjoint in original-token space,
    /// so `prefix[end] & !prefix[start]` is the exact dense mask for a full
    /// internal-word run `[start, end)`.
    pub(crate) word_group_prefix_buf_masks: DenseBufMaskRows,
    /// Prefix sums of `word_group_sparse_masks[i].len()`.
    pub(crate) word_group_sparse_prefix_entries: Vec<usize>,
    pub(crate) quad_group_sparse_masks: Vec<InternalTokenBufMasks>,
    /// Dense output masks for quad groups whose sparse replay is more
    /// expensive than a sequential output-buffer scan.
    pub(crate) quad_group_dense_masks: Vec<Option<Box<[u32]>>>,
    pub(crate) byte_group_sparse_masks: Vec<InternalTokenBufMasks>,
    /// Dense output masks for byte groups whose sparse replay is more
    /// expensive than a sequential output-buffer scan.
    pub(crate) byte_group_dense_masks: Vec<Option<Box<[u32]>>>,
    pub(crate) word_group_sparse_total_entries: usize,
    pub(crate) word_group_sparse_max_entries: usize,
    /// Precomputed buf output for the full internal token universe (OR of all word_group_buf_masks).
    pub(crate) all_tokens_buf_mask: Box<[u32]>,
    pub(crate) internal_token_dense_words: usize,
    pub(crate) weight_token_dense_masks: DenseWeightMaskCache,
    /// Dense masks for wide token sets retained in a current-format packed
    /// parser DWA. Unlike `weight_token_dense_masks`, these are keyed by the
    /// packed token-set id and can therefore be rebuilt directly from the
    /// artifact without materializing RangeSet/Weight objects.
    pub(crate) packed_dwa_token_dense_masks: PackedDwaDenseWeightMaskCache,
    pub(crate) weight_token_buf_masks: DenseWeightBufMaskCache,
    pub(crate) weight_token_sparse_buf_masks: SparseWeightBufMaskCache,
    /// Final-weight token sets eligible for the direct sparse-intersection
    /// path. Their full output masks are intentionally not materialized: the
    /// runtime intersects them with the current dense state on every use.
    pub(crate) direct_sparse_weight_token_sets: DirectSparseWeightTokenSetCache,
    /// Precomputed dense bitmask for the seed phase: for each (tokenizer_state, terminal_id),
    /// the dense bitmap of internal tokens that terminal covers in that state.
    pub(crate) seed_terminal_dense: SeedTerminalDenseMasks,
    /// Exact masks lazily materialized for delayed-exclusion pairs that are not
    /// represented by `possible_matches`. Shared across sequence states cloned
    /// from this immutable constraint.
    pub(crate) seed_terminal_dense_fallback: Arc<Mutex<SeedTerminalDenseMasks>>,
    /// Dense bitmap of the full internal token universe.
    pub(crate) seed_universe_dense: DenseWords,
    /// Fast DWA transition lookup (FxHashMap instead of BTreeMap).
    /// Built from parser_dwa.states at load/build time.
    pub(crate) dwa_fast_transitions: FastDwaTransitions,
    /// Runtime-only readiness marker for caches derived from the final parser
    /// DWA and final internal-token coordinate. Composition may build these at
    /// the final parser-union boundary so generic post-link finalization does
    /// not rescan the same parser artifact.
    pub(crate) parser_runtime_caches_prebuilt: bool,
    /// Runtime-only parser-DWA transitions with exact dense masks materialized
    /// for the final internal tokenizer states present in each transition
    /// weight; absent states are implicitly empty. Indexed-DAG masking uses
    /// this table directly instead of hashing a transition tuple and lazily
    /// rebuilding the same dense transition record at runtime.
    pub(crate) indexed_dag_dense_transitions: IndexedDagDenseTransitions,
    /// Runtime-only exact dense final weights, indexed by parser-DWA state.
    /// This is the final-weight analogue of `indexed_dag_dense_transitions`:
    /// absent tokenizer states are empty, and full final weights stay implicit.
    pub(crate) indexed_dag_dense_finals: Vec<IndexedDagDenseTransitionMasks>,
    /// Dense tokenizer transition lookup for commit-time byte scans.
    pub(crate) tokenizer_fast_transitions: FastTokenizerTransitions,
    /// Dense buf masks for "heavy" internal tokens (those with many buf entries).
    /// Indexed by internal token ID; None for light tokens.
    pub(crate) heavy_token_dense_masks: Vec<Option<Box<[u32]>>>,
    /// Flattened contiguous array of all internal token buf mask entries.
    /// All tokens' (word_index, or_mask) pairs concatenated in token order.
    /// Improves cache locality vs separate Vec allocations per token.
    pub(crate) internal_token_buf_flat: Box<[PackedInternalTokenBufMask]>,
    /// Current IBM2 loads can retain the runtime-native flat sparse-mask slab
    /// directly inside the owned artifact instead of copying ~0.5-1 MiB.
    pub(crate) backed_internal_token_buf_flat: Option<BackedInternalTokenBufMasks>,
    /// Offsets into `internal_token_buf_flat` for each internal token.
    /// `internal_token_buf_flat[offsets[i]..offsets[i+1]]` gives token i's entries.
    /// Length = n_internal + 1 (sentinel at end).
    pub(crate) internal_token_buf_offsets: Box<[u32]>,
    /// Pre-computed total cost (sum of entry counts) for all internal tokens.
    /// Used to avoid O(n_internal) cost analysis in the convert phase.
    pub(crate) total_internal_buf_cost: usize,
    /// Indices of heavy tokens for fast iteration. Length == n_heavy_tokens.
    pub(crate) heavy_token_indices: Vec<usize>,
    /// Total cost of all heavy tokens combined (n_heavy Ã— buf_len).
    pub(crate) heavy_total_cost: usize,
    /// Average cost per light token: (total_cost - heavy_total) / n_light.
    /// Pre-multiplied by 256 for fixed-point arithmetic to avoid float.
    pub(crate) light_avg_cost_x256: usize,
    /// Exact materialization cost per internal token, after heavy-token dense masks
    /// have been chosen.
    pub(crate) internal_token_buf_op_costs: Vec<usize>,
    /// Exact materialization cost per 64-token internal word group.
    pub(crate) word_group_buf_op_costs: Vec<usize>,
    /// Self-contained final internal-token -> original-token bitset materializer.
    pub(crate) final_mask_mapping: FinalMaskMapping,
    /// Optional exact quotient of positive parser-state labels used by composed
    /// parser DWAs. Entry `s` is a synthetic fallback label for parser state
    /// `s`; `i32::MAX` means no component-local fallback. Concrete parser-state
    /// transitions always take precedence, followed by this label, then the
    /// ordinary global DEFAULT. Empty for ordinary non-composed constraints.
    pub(crate) parser_state_domain_labels: Vec<i32>,
    /// Exact source expression for the globally erasable ignore terminal.
    ///
    /// Tokenizer source expressions are compile-time data and are normally
    /// omitted from artifacts. Retaining this one expression lets a loaded
    /// compiled constraint participate in later subgrammar composition without
    /// conservatively degrading an identical global ignore into scoped skips.
    pub(crate) ignore_expr: Option<Expr>,
    /// Exact current-format artifact backing for an unchanged loaded
    /// constraint. Runtime cache rebuilds do not alter serialized semantics,
    /// so resave can return a single bulk copy instead of rediscovering and
    /// re-encoding the same canonical pools.
    pub(crate) serialized_artifact_cache: Option<Arc<Vec<u8>>>,
    /// Current-format terminal source expressions can be retained as their
    /// canonical bincode payload instead of recursively rebuilding every Expr
    /// node during an ordinary static load. Composition materializes the list
    /// lazily through `retained_terminal_exprs` when it actually needs source
    /// language proofs.
    pub(crate) deferred_terminal_exprs_blob: Option<DeferredTerminalExprBytes>,
    pub(crate) deferred_terminal_exprs: OnceLock<Arc<[Expr]>>,
    /// Serialized composition-only metadata (reset-token rows, parser template
    /// cache, symbolic characterizations, and grammar summary). Current-format
    /// loads keep this cold section backed by the artifact and materialize it
    /// only if the constraint is later used as a composition component.
    pub(crate) deferred_composition_metadata_blob: Option<DeferredCompositionMetadataBytes>,
    /// Runtime-only marker distinguishing "the lightweight linking metadata
    /// has already been decoded" from "the heavy static compiler-cache blob is
    /// still deferred". Dynamic A+B intentionally keeps the latter deferred.
    pub(crate) composition_link_metadata_materialized: bool,
    /// Large current-format GLR rule vectors are composition metadata rather
    /// than runtime parser data. Keep their canonical payload undecoded during
    /// ordinary load; composition materializes it lazily through
    /// `retained_table_rules`.
    pub(crate) deferred_table_rules_blob:
        Option<crate::compiler::glr::table::artifact_serde::DeferredRuleBytes>,
    pub(crate) deferred_table_rules: OnceLock<Arc<[crate::grammar::flat::Rule]>>,
}

// Private Serde definition used only by the versioned artifact encoder/decoder.
// Keeping this remote definition separate prevents `Constraint` itself from
// implementing Serde, so `Constraint::save`/`Constraint::load` remain the only
// public persistence contract.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(remote = "Constraint")]
pub(crate) struct ConstraintSerde {
    #[serde(skip, default)]
    pub(crate) end_tokens: Arc<[u32]>,
    #[serde(default)]
    pub(crate) runtime_backend: ConstraintRuntimeBackend,
    #[serde(skip, default)]
    pub(crate) static_dynamic_overlay: Option<StaticDynamicOverlayMetadata>,
    #[serde(skip, default)]
    pub(crate) boundary_trigger: BoundaryTrigger,
    #[serde(skip, default)]
    pub(crate) boundary_candidate_summary: OnceLock<BoundaryCandidateSummary>,
    #[serde(skip, default)]
    pub(crate) late_grammar_slots: Vec<LateGrammarSlot>,
    #[serde(skip, default)]
    pub(crate) late_bind_vocab: OnceLock<crate::Vocab>,
    /// Runtime-derived exact original-token sets for `Skip` terminals in a
    /// composed grammar. Each token is wholly in `L(skip)+`: it can be
    /// consumed as one or more complete instances of that scoped-ignore
    /// terminal with a lexer reset between instances. This is deliberately
    /// not serialized; it is cheap to rebuild from the retained terminal
    /// expression and vocabulary and therefore does not change artifact wire
    /// compatibility.
    #[serde(skip, default)]
    pub(crate) scoped_ignore_only_tokens: Vec<(TerminalID, Box<[u32]>)>,
    /// Exact byte-token fusions `(fused, suffix)` grouped by scoped Skip. The
    /// fused token begins with one or more complete instances of the Skip
    /// language and the remaining bytes equal `suffix` exactly. If `suffix`
    /// is admitted by the ordinary static mask, `fused` is therefore admitted
    /// as well. Runtime-only for the same wire-compatibility reason as above.
    #[serde(skip, default)]
    pub(crate) scoped_ignore_prefix_fusions: Vec<(TerminalID, Box<[(u32, u32)]>)>,
    pub(crate) parser_dwa: DWA,
    /// Current-format loaded constraints retain the immutable parser DWA in
    /// its compact canonical pools instead of reconstructing RangeSet/Weight
    /// objects. Compiler-created and legacy-loaded constraints leave this
    /// empty and use `parser_dwa` directly.
    #[serde(skip, default)]
    pub(crate) packed_parser_dwa:
        Option<Arc<crate::automata::weighted::dwa::PackedRuntimeDwa>>,
    #[serde(skip, default)]
    pub(crate) parser_start_final_override: Option<Weight>,
    /// Exact depth-one parser acceptance kept separate from the deeper parser
    /// DWA. Keys are encoded parser-state labels; values are already the
    /// transition/final-weight intersection for accepting after that one
    /// stack symbol.
    #[serde(default)]
    pub(crate) parser_top_accept: BTreeMap<i32, Weight>,
    /// Uncombined exact depth-one acceptance parts. Direct-regular grammars
    /// retain terminal completion weights separately to avoid constructing one
    /// large union weight per parser state at compile time.
    #[serde(default)]
    pub(crate) parser_top_accept_parts: BTreeMap<i32, Vec<Weight>>,
    /// Immediate-completion L1 terminal weights for direct-regular parsers.
    /// Kept once per grammar terminal rather than duplicated across every
    /// epsilon-closed parser row.
    #[serde(default)]
    pub(crate) direct_regular_l1_complete_by_terminal: BTreeMap<TerminalID, Weight>,
    #[serde(skip, default)]
    pub(crate) packed_non_dwa_weights: Option<Arc<PackedNonDwaWeights>>,
    /// Runtime-derived exact acceptance summaries for wide direct-regular
    /// replace-top frontiers. Rebuilt after compile/load from the table and
    /// parser-top acceptance artifacts.
    #[serde(skip, default)]
    pub(crate) direct_regular_wide_frontier_acceptance:
        Vec<DirectRegularWideFrontierAcceptance>,
    /// Runtime-only exact transition maps for the direct automaton's initial
    /// frontier and its single widest successor frontier. Dynamic masking
    /// repeatedly queries these two frontiers at token boundaries.
    #[serde(skip, default)]
    pub(crate) direct_regular_dynamic_hot_frontiers:
        Vec<DirectRegularDynamicHotFrontier>,
    /// Runtime-derived exact dense acceptance for the broadest direct-regular
    /// parser row(s). This avoids replaying thousands of L1 terminal weights on
    /// every mask while keeping the cached result source-state exact.
    #[serde(skip, default)]
    pub(crate) direct_regular_parser_state_acceptance:
        Vec<DirectRegularParserStateAcceptance>,
    /// Sparse terminal-level automaton retained for exact direct-regular
    /// runtime indexes. Static artifact format versioning covers this field.
    #[serde(default)]
    pub(crate) direct_regular_automaton: Option<DirectRegularAutomaton>,
    #[serde(with = "crate::runtime::parser_backend::table_serde")]
    pub(crate) table: crate::runtime::parser_backend::ParserTableStorage,
    #[serde(default)]
    pub(crate) terminal_display_names: Vec<String>,
    #[serde(with = "crate::runtime::artifact::immutable_tokenizer_serde")]
    pub(crate) tokenizer: Arc<Tokenizer>,
    #[serde(skip)]
    pub(crate) boundary_completion_index: Option<Arc<crate::compiler::boundary_precomputed_completion::PreparedCompletion>>,
    /// Cached tokenizer topology flag. `Tokenizer::has_epsilon_transitions()`
    /// scans every tokenizer state, so runtime dispatch must not recompute it.
    #[serde(skip, default)]
    pub(crate) tokenizer_has_epsilon_transitions: bool,
    #[serde(default)]
    pub(crate) ignore_terminal: Option<TerminalID>,
    #[serde(default)]
    pub(crate) special_token_terminals: Vec<SpecialTokenTerminal>,

    /// Runtime-only vocabulary data for direct dynamic masking.
    #[serde(skip, default)]
    pub(crate) dynamic_mask_vocab: DynamicMaskVocab,
    /// Lazily materialized static-mode fallback vocabulary. Ordinary static
    /// masking never touches this; it is initialized only if an empty
    /// possible-matches table encounters a token-start exclusion.
    #[serde(skip, default)]
    pub(crate) lazy_dynamic_mask_vocab: OnceLock<DynamicMaskVocab>,
    #[serde(skip, default)]
    pub(crate) empty_byte_token_ids: Arc<[u32]>,

    /// possible_matches keyed by grammar terminal id.
    ///
    /// An empty table may represent deferred possible-match construction in
    /// legacy code only.
    ///
    /// IMPORTANT: the dynamic possible-matches fallback is intentionally
    /// terrible and is planned for removal. New compiler paths MUST construct
    /// complete exact possible matches and MUST NOT set
    /// `possible_matches_complete` to false as an implementation shortcut.
    /// DO NOT REMOVE OR WEAKEN THIS COMMENT.
    ///
    /// Each Weight maps final shared internal tokenizer-state ids to token sets
    /// in the final shared constraint-internal vocab space. Parser-DWA weights
    /// and possible_matches weights are reconciled into this same space during
    /// compilation.
    pub(crate) possible_matches: PossibleMatchesByTerminal,
    /// Whether `possible_matches` is a complete table. New static constraints
    /// must set this to true. False exists only for legacy dynamic/deferred
    /// construction and is not permitted as a fallback strategy for new
    /// compiler features.
    #[serde(default)]
    pub(crate) possible_matches_complete: bool,
    pub(crate) state_to_internal_tsid: Vec<u32>,
    #[serde(default, with = "internal_tsid_inverse_artifact_serde")]
    pub(crate) internal_tsid_to_states: Vec<Vec<u32>>,
    /// Ordinary tokenizers have one internal TSID per physical state, making
    /// `internal_tsid_to_states` the exact bucket inverse of
    /// `state_to_internal_tsid`. Current artifacts can omit that redundant
    /// allocation and reconstruct it only for composition/debug paths.
    #[serde(skip, default)]
    pub(crate) deferred_internal_tsid_to_states: OnceLock<Vec<Vec<u32>>>,
    /// Composition-preparation cache: row `t` lists original model-token IDs
    /// which, from this component's lexer reset, complete terminal `t` exactly
    /// at the end of the model token.  This is not part of the historical inner
    /// `Constraint` bincode layout; artifact V13 stores it in the outer
    /// envelope so V12 constraints remain loadable unchanged.
    #[serde(skip, default)]
    pub(crate) composition_reset_tokens_by_terminal: Vec<Vec<u32>>,
    /// Named unresolved `extern grammar` slots retained by a compiled parent.
    /// Values are parent-local hidden placeholder terminal IDs. Stored in the
    /// outer composition metadata so cached parents can be rebound after load.
    #[serde(skip, default)]
    pub(crate) unbound_grammar_placeholders: BTreeMap<String, TerminalID>,
    /// Composition-time parser stack-effect templates retained from the
    /// original compile. These are the unspecialized per-terminal DFAs used to
    /// build parser DWAs, so a later linker can transport unchanged component
    /// behavior instead of re-characterizing the component LR table.
    /// Stored in the outer versioned artifact envelope for compatibility with
    /// older inner `Constraint` bincode layouts.
    #[serde(skip, default)]
    pub(crate) composition_parser_templates_by_terminal: Vec<Option<UnweightedDfa>>,
    /// Composition-time symbolic parser characterizations retained from the
    /// original compile. A later linker can append only the boundary-induced
    /// reductions/rereductions and recompile affected terminal templates,
    /// rather than re-solving the component's reduction closure from scratch.
    #[serde(skip, default)]
    pub(crate) composition_parser_characterizations_by_terminal:
        Vec<Option<TerminalCharacterization>>,
    /// Composition-time grammar adjacency summary. Stored in the outer
    /// versioned artifact envelope so older inner `Constraint` layouts remain
    /// loadable unchanged.
    #[serde(skip, default)]
    pub(crate) composition_grammar_summary: Option<CompositionGrammarSummary>,
    /// Runtime-only inverse lexer-metadata index used by compiled-constraint
    /// composition. Row `t` lists exactly the raw tokenizer states whose
    /// epsilon closure has terminal `t` matched or still reachable.
    #[serde(skip, default)]
    pub(crate) terminal_live_states: Vec<Vec<u32>>,
    /// Runtime-only CSR view of the exact state -> internal-TSID relation.
    /// Ordinary tokenizers have one entry per state. A fully determinized
    /// runtime lexer may represent several old lexer states and therefore
    /// several independent TSID lanes in one physical state.
    #[serde(skip, default)]
    pub(crate) state_internal_tsid_offsets: Vec<u32>,
    #[serde(skip, default)]
    pub(crate) state_internal_tsids: Vec<u32>,
    /// Final-runtime subset states followed by an exact copy of the source
    /// tokenizer. `runtime_source_state_offset` is the boundary between the
    /// two coordinates. Empty metadata means no runtime-only determinization.
    #[serde(default)]
    pub(crate) runtime_source_state_offset: Option<u32>,
    /// CSR offsets for product-state -> exact source-state subset. There is one
    /// row per product state and therefore `product_state_count + 1` offsets.
    #[serde(default)]
    pub(crate) runtime_product_source_offsets: Vec<u32>,
    #[serde(default)]
    pub(crate) runtime_product_source_states: Vec<u32>,
    /// Scalar source representative for product states that are exactly one
    /// source state's epsilon closure; `u32::MAX` otherwise.
    #[serde(default)]
    pub(crate) runtime_product_exact_source_states: Vec<u32>,
    /// Runtime-only inverse used to re-coalesce a uniform source frontier.
    #[serde(skip, default)]
    pub(crate) runtime_product_state_by_source_subset: FxHashMap<Box<[u32]>, u32>,
    #[serde(with = "crate::runtime::parser_backend::wire::core_programs")]
    pub(crate) template_dfas_by_terminal: TemplateDfasByTerminal,
    /// Runtime-only compact transition view for commit template products.
    #[serde(skip, default)]
    pub(crate) fast_template_dfas_by_terminal: FastTemplateDfasByTerminal,
    #[serde(skip, default)]
    pub(crate) template_parser: Option<Arc<crate::runtime::parser_backend::TemplateParser>>,
    /// Original token -> final shared constraint-internal token id.
    ///
    /// This is not necessarily equal to the parser-DWA compaction vocab map
    /// produced before possible-match reconciliation. It may contain additional
    /// splits required by possible_matches.
    #[serde(default, with = "original_token_map_artifact_serde")]
    pub(crate) original_token_to_internal: Vec<u32>,
    /// Current-format loads retain the fixed-width original-token map inside
    /// the owned artifact instead of expanding all model-token entries to
    /// `u32`. Ordinary static mask/commit performs direct packed lookups; only
    /// composition/debug-style bulk access materializes the vector lazily.
    #[serde(skip, default)]
    pub(crate) packed_original_token_to_internal:
        Option<Arc<original_token_map_artifact_serde::PackedOriginalTokenMap>>,
    #[serde(skip, default)]
    pub(crate) deferred_original_token_to_internal: OnceLock<Vec<u32>>,
    /// Final shared constraint-internal token id -> original token ids.
    ///
    /// Parser-DWA weights and Constraint.possible_matches bitmaps both use these
    /// final internal token ids.
    #[serde(default, with = "internal_token_inverse_artifact_serde")]
    pub(crate) internal_token_to_tokens: Vec<Vec<u32>>,
    /// Current-format loads can defer reconstructing the explicit inverse of
    /// `original_token_to_internal`. Static mask/commit only needs the internal
    /// token count and the already-serialized mask fragments; composition and
    /// token-space expansion materialize this inverse on first use.
    #[serde(skip, default)]
    pub(crate) deferred_internal_token_to_tokens: OnceLock<Vec<Vec<u32>>>,
    #[serde(with = "token_bytes_artifact_serde")]
    pub(crate) token_bytes: Arc<BTreeMap<u32, Vec<u8>>>,
    /// Indexed immutable vocabulary used directly by runtime token lookup and
    /// iteration. Loaded constraints can point into artifact backing; compiled
    /// constraints own the same indexed representation alongside the source
    /// map used by compiler/composition code.
    #[serde(skip, default)]
    pub(crate) packed_token_bytes: Option<Arc<token_bytes_artifact_serde::PackedTokenBytes>>,
    // Compiler-side scratch/result metadata only. No runtime or composition
    // path reads this field; composition rebuilds the map for its result when
    // needed. Persisting it duplicated token bytes inside every constraint.
    #[serde(skip, default)]
    pub(crate) internal_token_bytes: BTreeMap<u32, Vec<u8>>,
    #[serde(skip)]
    pub(crate) token_bytes_dense: Vec<Option<Box<[u8]>>>,

    /// Precomputed bitmask fragments for each internal token.
    /// `internal_token_buf_masks[i]` contains (word_index, or_mask) pairs
    /// for all original tokens that map to internal token `i`.
    #[serde(skip)]
    pub(crate) internal_token_buf_masks: Vec<InternalTokenBufMasks>,
    /// Precomputed combined buf output for each group of 64 internal tokens.
    /// `word_group_buf_masks[w]` is the combined mask for internal tokens [w*64 .. (w+1)*64).
    /// Used as a fast path in `or_to_buf` when a dense word is all-ones (!0u64).
    #[serde(skip)]
    pub(crate) word_group_buf_masks: Vec<Box<[u32]>>,
    /// Precomputed dense output masks for groups of 128 internal tokens.
    #[serde(skip)]
    pub(crate) pair_word_group_buf_masks: DenseBufMaskRows,
    /// Precomputed dense output masks for groups of 256 internal tokens.
    #[serde(skip)]
    pub(crate) quad_word_group_buf_masks: DenseBufMaskRows,
    /// Precomputed dense output masks for groups of 512 internal tokens.
    #[serde(skip)]
    pub(crate) super_word_group_buf_masks: DenseBufMaskRows,
    /// Precomputed dense output masks for groups of 1024 internal tokens.
    #[serde(skip)]
    pub(crate) mega_word_group_buf_masks: DenseBufMaskRows,
    /// Precomputed dense output masks for groups of 2048 internal tokens.
    #[serde(skip)]
    pub(crate) giga_word_group_buf_masks: DenseBufMaskRows,
    /// Sparse OR-union for each 64-token internal word group.
    #[serde(skip)]
    pub(crate) word_group_sparse_masks: Vec<InternalTokenBufMasks>,
    /// Dense prefix-unions of 64-token internal word groups.
    ///
    /// `word_group_prefix_buf_masks[i]` is the OR-union of word groups
    /// `[0, i)`. Internal-token groups are disjoint in original-token space,
    /// so `prefix[end] & !prefix[start]` is the exact dense mask for a full
    /// internal-word run `[start, end)`.
    #[serde(skip)]
    pub(crate) word_group_prefix_buf_masks: DenseBufMaskRows,
    /// Prefix sums of `word_group_sparse_masks[i].len()`.
    #[serde(skip)]
    pub(crate) word_group_sparse_prefix_entries: Vec<usize>,
    #[serde(skip)]
    pub(crate) quad_group_sparse_masks: Vec<InternalTokenBufMasks>,
    /// Dense output masks for quad groups whose sparse replay is more
    /// expensive than a sequential output-buffer scan.
    #[serde(skip)]
    pub(crate) quad_group_dense_masks: Vec<Option<Box<[u32]>>>,
    #[serde(skip)]
    pub(crate) byte_group_sparse_masks: Vec<InternalTokenBufMasks>,
    /// Dense output masks for byte groups whose sparse replay is more
    /// expensive than a sequential output-buffer scan.
    #[serde(skip)]
    pub(crate) byte_group_dense_masks: Vec<Option<Box<[u32]>>>,
    pub(crate) word_group_sparse_total_entries: usize,
    #[serde(skip)]
    pub(crate) word_group_sparse_max_entries: usize,
    /// Precomputed buf output for the full internal token universe (OR of all word_group_buf_masks).
    #[serde(skip)]
    pub(crate) all_tokens_buf_mask: Box<[u32]>,
    #[serde(skip)]
    pub(crate) internal_token_dense_words: usize,
    #[serde(skip)]
    pub(crate) weight_token_dense_masks: DenseWeightMaskCache,
    /// Dense masks for wide token sets retained in a current-format packed
    /// parser DWA. Unlike `weight_token_dense_masks`, these are keyed by the
    /// packed token-set id and can therefore be rebuilt directly from the
    /// artifact without materializing RangeSet/Weight objects.
    #[serde(skip, default)]
    pub(crate) packed_dwa_token_dense_masks: PackedDwaDenseWeightMaskCache,
    #[serde(skip)]
    pub(crate) weight_token_buf_masks: DenseWeightBufMaskCache,
    #[serde(skip)]
    pub(crate) weight_token_sparse_buf_masks: SparseWeightBufMaskCache,
    /// Final-weight token sets eligible for the direct sparse-intersection
    /// path. Their full output masks are intentionally not materialized: the
    /// runtime intersects them with the current dense state on every use.
    #[serde(skip)]
    pub(crate) direct_sparse_weight_token_sets: DirectSparseWeightTokenSetCache,
    /// Precomputed dense bitmask for the seed phase: for each (tokenizer_state, terminal_id),
    /// the dense bitmap of internal tokens that terminal covers in that state.
    #[serde(skip)]
    pub(crate) seed_terminal_dense: SeedTerminalDenseMasks,
    /// Exact masks lazily materialized for delayed-exclusion pairs that are not
    /// represented by `possible_matches`. Shared across sequence states cloned
    /// from this immutable constraint.
    #[serde(skip, default)]
    pub(crate) seed_terminal_dense_fallback: Arc<Mutex<SeedTerminalDenseMasks>>,
    /// Dense bitmap of the full internal token universe.
    #[serde(skip, default = "empty_dense_words")]
    pub(crate) seed_universe_dense: DenseWords,
    /// Fast DWA transition lookup (FxHashMap instead of BTreeMap).
    /// Built from parser_dwa.states at load/build time.
    #[serde(skip)]
    pub(crate) dwa_fast_transitions: FastDwaTransitions,
    /// Runtime-only readiness marker for caches derived from the final parser
    /// DWA and final internal-token coordinate. Composition may build these at
    /// the final parser-union boundary so generic post-link finalization does
    /// not rescan the same parser artifact.
    #[serde(skip, default)]
    pub(crate) parser_runtime_caches_prebuilt: bool,
    /// Runtime-only parser-DWA transitions with exact dense masks materialized
    /// for the final internal tokenizer states present in each transition
    /// weight; absent states are implicitly empty. Indexed-DAG masking uses
    /// this table directly instead of hashing a transition tuple and lazily
    /// rebuilding the same dense transition record at runtime.
    #[serde(skip, default)]
    pub(crate) indexed_dag_dense_transitions: IndexedDagDenseTransitions,
    /// Runtime-only exact dense final weights, indexed by parser-DWA state.
    /// This is the final-weight analogue of `indexed_dag_dense_transitions`:
    /// absent tokenizer states are empty, and full final weights stay implicit.
    #[serde(skip, default)]
    pub(crate) indexed_dag_dense_finals: Vec<IndexedDagDenseTransitionMasks>,
    /// Dense tokenizer transition lookup for commit-time byte scans.
    #[serde(skip)]
    pub(crate) tokenizer_fast_transitions: FastTokenizerTransitions,
    /// Dense buf masks for "heavy" internal tokens (those with many buf entries).
    /// Indexed by internal token ID; None for light tokens.
    #[serde(skip)]
    pub(crate) heavy_token_dense_masks: Vec<Option<Box<[u32]>>>,
    /// Flattened contiguous array of all internal token buf mask entries.
    /// All tokens' (word_index, or_mask) pairs concatenated in token order.
    /// Improves cache locality vs separate Vec allocations per token.
    #[serde(skip)]
    pub(crate) internal_token_buf_flat: Box<[PackedInternalTokenBufMask]>,
    /// Current IBM2 loads can retain the runtime-native flat sparse-mask slab
    /// directly inside the owned artifact instead of copying ~0.5-1 MiB.
    #[serde(skip, default)]
    pub(crate) backed_internal_token_buf_flat: Option<BackedInternalTokenBufMasks>,
    /// Offsets into `internal_token_buf_flat` for each internal token.
    /// `internal_token_buf_flat[offsets[i]..offsets[i+1]]` gives token i's entries.
    /// Length = n_internal + 1 (sentinel at end).
    #[serde(skip)]
    pub(crate) internal_token_buf_offsets: Box<[u32]>,
    /// Pre-computed total cost (sum of entry counts) for all internal tokens.
    /// Used to avoid O(n_internal) cost analysis in the convert phase.
    #[serde(skip)]
    pub(crate) total_internal_buf_cost: usize,
    /// Indices of heavy tokens for fast iteration. Length == n_heavy_tokens.
    #[serde(skip)]
    pub(crate) heavy_token_indices: Vec<usize>,
    /// Total cost of all heavy tokens combined (n_heavy Ã— buf_len).
    #[serde(skip)]
    pub(crate) heavy_total_cost: usize,
    /// Average cost per light token: (total_cost - heavy_total) / n_light.
    /// Pre-multiplied by 256 for fixed-point arithmetic to avoid float.
    #[serde(skip)]
    pub(crate) light_avg_cost_x256: usize,
    /// Exact materialization cost per internal token, after heavy-token dense masks
    /// have been chosen.
    #[serde(skip)]
    pub(crate) internal_token_buf_op_costs: Vec<usize>,
    /// Exact materialization cost per 64-token internal word group.
    #[serde(skip)]
    pub(crate) word_group_buf_op_costs: Vec<usize>,
    /// Self-contained final internal-token -> original-token bitset materializer.
    #[serde(skip)]
    pub(crate) final_mask_mapping: FinalMaskMapping,
    /// Optional exact quotient of positive parser-state labels used by composed
    /// parser DWAs. Entry `s` is a synthetic fallback label for parser state
    /// `s`; `i32::MAX` means no component-local fallback. Concrete parser-state
    /// transitions always take precedence, followed by this label, then the
    /// ordinary global DEFAULT. Empty for ordinary non-composed constraints.
    #[serde(skip, default)]
    pub(crate) parser_state_domain_labels: Vec<i32>,
    /// Exact source expression for the globally erasable ignore terminal.
    ///
    /// Tokenizer source expressions are compile-time data and are normally
    /// omitted from artifacts. Retaining this one expression lets a loaded
    /// compiled constraint participate in later subgrammar composition without
    /// conservatively degrading an identical global ignore into scoped skips.
    #[serde(skip, default)]
    pub(crate) ignore_expr: Option<Expr>,
    /// Exact current-format artifact backing for an unchanged loaded
    /// constraint. Runtime cache rebuilds do not alter serialized semantics,
    /// so resave can return a single bulk copy instead of rediscovering and
    /// re-encoding the same canonical pools.
    #[serde(skip, default)]
    pub(crate) serialized_artifact_cache: Option<Arc<Vec<u8>>>,
    /// Current-format terminal source expressions can be retained as their
    /// canonical bincode payload instead of recursively rebuilding every Expr
    /// node during an ordinary static load. Composition materializes the list
    /// lazily through `retained_terminal_exprs` when it actually needs source
    /// language proofs.
    #[serde(skip, default)]
    pub(crate) deferred_terminal_exprs_blob: Option<DeferredTerminalExprBytes>,
    #[serde(skip, default)]
    pub(crate) deferred_terminal_exprs: OnceLock<Arc<[Expr]>>,
    /// Serialized composition-only metadata (reset-token rows, parser template
    /// cache, symbolic characterizations, and grammar summary). Current-format
    /// loads keep this cold section backed by the artifact and materialize it
    /// only if the constraint is later used as a composition component.
    #[serde(skip, default)]
    pub(crate) deferred_composition_metadata_blob: Option<DeferredCompositionMetadataBytes>,
    #[serde(skip, default)]
    pub(crate) composition_link_metadata_materialized: bool,
    /// Large current-format GLR rule vectors are composition metadata rather
    /// than runtime parser data. Keep their canonical payload undecoded during
    /// ordinary load; composition materializes it lazily through
    /// `retained_table_rules`.
    #[serde(skip, default)]
    pub(crate) deferred_table_rules_blob:
        Option<crate::compiler::glr::table::artifact_serde::DeferredRuleBytes>,
    #[serde(skip, default)]
    pub(crate) deferred_table_rules: OnceLock<Arc<[crate::grammar::flat::Rule]>>,
}


#[cfg(test)]
mod dynamic_mask_vocab_cache_boundary_tests {
    use super::*;

    #[test]
    fn full_walk_flat32_covers_tokenizers_beyond_flat16_state_space() {
        const TARGET: u32 = 32_768;
        let tokenizer = crate::automata::lexer::tokenizer::arbitrary_flat32_test_tokenizer();

        assert!(FastTokenizerTransitions::flat16_for(&tokenizer).is_none());
        let flat32 = FastTokenizerTransitions::flat32_for(&tokenizer)
            .expect("32769-state tokenizer must fit the Flat32 full-walk coordinate");
        assert_eq!(flat32.len(), TARGET as usize + 1);
        assert_eq!(flat32.transition(&tokenizer, 0, b'a'), TARGET);
        assert_eq!(flat32.transition(&tokenizer, 0, b'b'), u32::MAX);
        let FastTokenizerTransitions::Flat32 { transitions, .. } = flat32 else {
            panic!("wide tokenizer unexpectedly used a non-Flat32 representation");
        };
        let encoded = transitions[b'a' as usize];
        assert_ne!(encoded & 0x8000_0000, 0, "finalizer bit was not encoded");
        assert_eq!(encoded & 0x7fff_ffff, TARGET);
    }

    #[test]
    fn packed_dwa_dense_mask_cache_rejects_malformed_flat_layouts() {
        assert!(PackedDwaDenseWeightMaskCache::from_flat(4, 2, vec![1], vec![7]).is_err());
        assert!(
            PackedDwaDenseWeightMaskCache::from_flat(4, 2, vec![1, 1], vec![1, 2, 3, 4])
                .is_err()
        );
        assert!(
            PackedDwaDenseWeightMaskCache::from_flat(2, 1, vec![2], vec![7]).is_err()
        );
    }

    #[test]
    fn vocab_only_artifact_rejects_missing_trie_root() {
        let vocab = DynamicMaskVocab::from_materialized_ordered(
            Arc::new(DynamicMaskTrie::new()),
            Arc::new(Vec::new()),
        );
        let mut artifact = vocab
            .to_vocab_artifact()
            .expect("initialized vocabulary should serialize");
        assert!(artifact.mask_tokenizer.is_none());
        assert!(artifact.full_to_mask_state.is_empty());
        artifact.trie.nodes.clear();
        let error = DynamicMaskVocab::from_artifact(artifact).unwrap_err();
        assert!(error.contains("no trie root"));
    }

    #[test]
    fn dense_mask_projection_union_lookup_matches_exact_source_subset_union() {
        let source =
            crate::automata::lexer::tokenizer::arbitrary_epsilon_l1_test_tokenizer();
        let (built, full_to_mask_state) = source
            .try_full_determinization_all_starts(256, 16_384)
            .expect("small epsilon tokenizer should determinize from every raw start");
        let source_subsets = built.source_subsets.clone();
        let state_count = built.tokenizer.num_states();

        let mut vocab = DynamicMaskVocab::from_materialized_ordered(
            Arc::new(DynamicMaskTrie::new()),
            Arc::new(Vec::new()),
        );
        vocab.set_mask_tokenizer_quotient(built.tokenizer, full_to_mask_state);
        vocab.set_mask_tokenizer_source_subsets(source_subsets.clone());

        for left in 0..state_count {
            for right in 0..state_count {
                let mut union = source_subsets[left as usize].to_vec();
                union.extend_from_slice(&source_subsets[right as usize]);
                union.sort_unstable();
                union.dedup();
                let expected = source_subsets
                    .iter()
                    .position(|subset| subset.as_ref() == union.as_slice())
                    .map(|state| state as u32);
                assert_eq!(
                    vocab.mask_projection_state_for_projection_states(&[left, right]),
                    expected,
                    "projection-state union mismatch for ({left}, {right})",
                );
            }
        }
    }

    #[test]
    fn full_artifact_round_trips_dense_mask_tokenizer_quotient() {
        let source =
            crate::automata::lexer::tokenizer::arbitrary_epsilon_l1_test_tokenizer();
        let (built, full_to_mask_state) = source
            .try_full_determinization_all_starts(256, 16_384)
            .expect("small epsilon tokenizer should determinize from every raw start");
        let expected_states = built.tokenizer.num_states();

        let mut vocab = DynamicMaskVocab::from_materialized_ordered(
            Arc::new(DynamicMaskTrie::new()),
            Arc::new(Vec::new()),
        );
        vocab.set_mask_tokenizer_quotient(built.tokenizer, full_to_mask_state.clone());

        let artifact = vocab.to_artifact().expect("full runtime artifact should serialize");
        assert_eq!(artifact.full_to_mask_state, full_to_mask_state);
        assert_eq!(
            artifact.mask_tokenizer.as_ref().map(Tokenizer::num_states),
            Some(expected_states),
        );

        let loaded = DynamicMaskVocab::from_artifact(artifact).unwrap();
        assert_eq!(loaded.full_to_mask_state.as_ref(), full_to_mask_state.as_slice());
        assert_eq!(
            loaded.mask_projection_tokenizer().map(Tokenizer::num_states),
            Some(expected_states),
        );
        assert!(matches!(
            loaded.mask_projection_fast_transitions(),
            Some(FastTokenizerTransitions::Flat16 { .. }),
        ));
    }

    #[test]
    fn vocab_artifact_restores_root_layout_metadata_from_token_bytes() {
        let token_bytes = BTreeMap::from([
            (0u32, b"abc".to_vec()),
            (1u32, "é".as_bytes().to_vec()),
        ]);
        let mut entries = token_bytes
            .iter()
            .enumerate()
            .map(|(canonical, (_, bytes))| {
                (
                    dynamic_mask_vocab_layout_class(classify_vocab_char_type(bytes), bytes),
                    canonical,
                    bytes.as_slice(),
                )
            })
            .collect::<Vec<_>>();
        entries.sort_unstable_by(|left, right| {
            left.0
                .cmp(&right.0)
                .then_with(|| left.2.cmp(right.2))
                .then_with(|| left.1.cmp(&right.1))
        });
        let trie = DynamicMaskTrie::from_partitioned_token_refs(&entries);
        let expected_classes = trie.root_layout_classes.clone();
        let expected_utf8 = trie.root_layout_all_valid_utf8.clone();
        let vocab = DynamicMaskVocab::from_materialized_ordered(
            Arc::new(trie),
            Arc::new(vec![vec![0], vec![1]]),
        );
        let artifact = vocab.to_vocab_artifact().unwrap();
        let loaded = DynamicMaskVocab::from_artifact(artifact).unwrap();
        assert_eq!(loaded.trie.root_layout_classes, expected_classes);
        assert_eq!(loaded.trie.root_layout_all_valid_utf8, expected_utf8);
    }

    #[test]
    fn full_walk_all_consume_exactly_rejects_empty_edges() {
        fn one_edge(bytes: &[u8]) -> DynamicMaskTrie {
            let mut trie = DynamicMaskTrie::new();
            trie.nodes.push(DynamicMaskTrieNode {
                token_id: Some(0),
                ..DynamicMaskTrieNode::default()
            });
            let (byte_start, byte_len) = trie.push_edge_bytes(bytes);
            trie.edges.push(DynamicMaskTrieEdge {
                byte_start,
                byte_len,
                child: 1,
            });
            trie.nodes[0].first_child = 0;
            trie.nodes[0].child_len = 1;
            trie.finalize_subtree_metadata();
            trie
        }

        let nonempty = one_edge(b"abc");
        assert_eq!(nonempty.full_walk_ops().len(), 3);
        assert!(nonempty.full_walk_all_consume());

        let empty = one_edge(b"");
        assert_eq!(empty.full_walk_ops().len(), 1);
        assert!(!empty.full_walk_all_consume());
    }

    #[test]
    fn vocab_runtime_exactly_checks_original_token_bytes() {
        let mut trie = DynamicMaskTrie::new();
        trie.nodes.push(DynamicMaskTrieNode {
            token_id: Some(0),
            ..DynamicMaskTrieNode::default()
        });
        let (byte_start, byte_len) = trie.push_edge_bytes(b"a");
        trie.edges.push(DynamicMaskTrieEdge {
            byte_start,
            byte_len,
            child: 1,
        });
        trie.nodes[0].first_child = 0;
        trie.nodes[0].child_len = 1;
        trie.finalize_subtree_metadata();
        let vocab = DynamicMaskVocab::from_materialized_ordered(
            Arc::new(trie),
            Arc::new(vec![vec![7]]),
        );
        assert!(vocab.matches_token_bytes_exact(&BTreeMap::from([(7, b"a".to_vec())])));
        assert!(!vocab.matches_token_bytes_exact(&BTreeMap::from([(7, b"b".to_vec())])));
    }

    #[test]
    fn lazy_union_cache_try_lock_is_nonblocking_when_in_use() {
        let vocab = DynamicMaskVocab::from_materialized_ordered(
            Arc::new(DynamicMaskTrie::new()),
            Arc::new(Vec::new()),
        );
        let _owner = vocab.lock_lazy_union_cache();
        assert!(vocab.try_lock_lazy_union_cache().is_none());
    }

    #[test]
    fn fresh_runtime_instance_shares_only_vocab_derived_data() {
        let template = DynamicMaskVocab::from_materialized_ordered(
            Arc::new(DynamicMaskTrie::new()),
            Arc::new(Vec::new()),
        );
        let fresh = template.fresh_runtime_instance();

        assert!(Arc::ptr_eq(&template.trie, &fresh.trie));
        match (&template.token_aliases, &fresh.token_aliases) {
            (DynamicMaskAliasStore::Ordered(left), DynamicMaskAliasStore::Ordered(right)) => {
                assert!(Arc::ptr_eq(left, right));
            }
            _ => panic!("materialized ordered vocabulary changed alias representation"),
        }
        assert!(Arc::ptr_eq(
            &template.canonical_original_token_offsets,
            &fresh.canonical_original_token_offsets,
        ));
        assert!(Arc::ptr_eq(
            &template.canonical_original_tokens,
            &fresh.canonical_original_tokens,
        ));
        assert!(Arc::ptr_eq(
            &template.node_token_markers,
            &fresh.node_token_markers,
        ));
        assert!(Arc::ptr_eq(
            &template.subtree_original_token_offsets,
            &fresh.subtree_original_token_offsets,
        ));
        assert!(Arc::ptr_eq(
            &template.subtree_original_tokens,
            &fresh.subtree_original_tokens,
        ));

        assert!(!Arc::ptr_eq(&template.mask_cache, &fresh.mask_cache));
        assert!(!Arc::ptr_eq(
            &template.direct_regular_frontier_cache,
            &fresh.direct_regular_frontier_cache,
        ));
        assert!(!Arc::ptr_eq(
            &template.direct_regular_wide_frontier_index_cache,
            &fresh.direct_regular_wide_frontier_index_cache,
        ));
        assert!(!Arc::ptr_eq(
            &template.direct_regular_terminal_support,
            &fresh.direct_regular_terminal_support,
        ));
        assert!(!Arc::ptr_eq(
            &template.self_loop_projections,
            &fresh.self_loop_projections,
        ));
    }
}

/// Arc ownership is a runtime/lifecycle detail; keep the existing tokenizer
/// artifact bytes unchanged. A retained observer can pin this same allocation,
/// and all subsequent mutation must use copy-on-write on the Constraint side.
pub(crate) mod immutable_tokenizer_serde {
    use super::{Arc, Tokenizer};
    use serde::{Deserializer, Serializer};
    pub(crate) fn serialize<S: Serializer>(tokenizer: &Arc<Tokenizer>, serializer: S) -> Result<S::Ok,S::Error> {
        crate::automata::lexer::tokenizer::artifact_serde::serialize(tokenizer.as_ref(), serializer)
    }
    pub(crate) fn deserialize<'de,D: Deserializer<'de>>(deserializer: D) -> Result<Arc<Tokenizer>,D::Error> {
        crate::automata::lexer::tokenizer::artifact_serde::deserialize(deserializer).map(Arc::new)
    }
}

#[cfg(test)]
mod immutable_tokenizer_tests {
    use super::*;
    use crate::automata::lexer::ast::{bytes, choice, plus};
    use crate::automata::lexer::compile::{build_regex_monolithic, build_regex_partitioned};
    use serde::{Deserialize, Serialize};
    #[derive(Serialize)]
    struct OwnedRef<'a>(#[serde(with = "crate::automata::lexer::tokenizer::artifact_serde")] &'a Tokenizer);
    #[derive(Serialize, Deserialize)]
    struct Shared(#[serde(with = "super::immutable_tokenizer_serde")] Arc<Tokenizer>);
    #[test]
    fn shared_tokenizer_retains_exact_owned_wire_and_byte_behavior() {
        let expressions=vec![choice(vec![bytes(b"abc"),bytes(b"abd")]),plus(bytes(b" "))];
        for tokenizer in [build_regex_monolithic(&expressions).into_tokenizer(2,None),build_regex_partitioned(&expressions,&[0,1]).into_tokenizer(2,None)] {
            let owned_wire=bincode::serialize(&OwnedRef(&tokenizer)).unwrap();let shared=Shared(Arc::new(tokenizer));
            let wire=bincode::serialize(&shared).unwrap();assert_eq!(owned_wire,wire);
            let loaded:Shared=bincode::deserialize(&wire).unwrap();assert_eq!(bincode::serialize(&loaded).unwrap(),owned_wire);
            for q in 0..shared.0.num_states(){for b in 0..=255u8{assert_eq!(shared.0.step_all(&[q],b),loaded.0.step_all(&[q],b));}}
        }
    }
    #[test]
    fn pinned_tokenizer_is_unchanged_after_copy_on_write_mutation() {
        let expressions=vec![Expr::Epsilon,bytes(b"a")];
        let mut owner=Arc::new(build_regex_monolithic(&expressions).into_tokenizer(2,None));let pinned=Arc::clone(&owner);
        let original=bincode::serialize(&OwnedRef(pinned.as_ref())).unwrap();assert!(Arc::ptr_eq(&owner,&pinned));
        let drained=Arc::make_mut(&mut owner).isolate_start_state_and_drain_nullable_terminals();
        assert!(drained.contains(&0),"fixture must exercise a real nullable-state mutation");assert!(!Arc::ptr_eq(&owner,&pinned));
        assert_eq!(bincode::serialize(&OwnedRef(pinned.as_ref())).unwrap(),original);
        assert_ne!(bincode::serialize(&OwnedRef(owner.as_ref())).unwrap(),original);
        assert!(!Arc::make_mut(&mut owner).isolate_start_state_and_drain_nullable_terminals().contains(&0));
    }
    #[test]
    fn constraint_clone_shares_the_same_immutable_lexer() {
        let vocab=crate::Vocab::new(vec![(1,b"a".to_vec()),(2,b"b".to_vec())]);
        let source=Constraint::from_glrm_grammar(r#"start x; nt x ::= "a" | "b";"#,&vocab).unwrap();
        let copy=source.clone();assert!(Arc::ptr_eq(&source.tokenizer,&copy.tokenizer));assert_eq!(source.save(),copy.save());
    }
}

#[cfg(test)]
mod raw_terminal_radius_tests {
    use super::*;
    use crate::{Constraint, Grammar, Vocab};

    /// Exhaustively enumerate the finite test alphabet, independent of the
    /// product/shortest-distance algorithm being checked. No quotient proofs.
    fn reference_radius(
        tokenizer: &Tokenizer, terminal: TerminalID, source: u32,
        atoms: &[&[u8]], horizon: u32,
    ) -> u32 {
        let mut frontier = vec![source];
        for count in 1..=horizon {
            let mut next = Vec::new();
            for state in frontier {
                for atom in atoms {
                    let mut end = state;
                    for &byte in *atom {
                        let Some(target) = tokenizer.step(end, byte) else { return count - 1; };
                        if target >= tokenizer.num_states()
                            || (!tokenizer.possible_future_terminals(target).contains(terminal as usize)
                                && !tokenizer.matched_terminals_slice(target).contains(&terminal))
                        {
                            return count - 1;
                        }
                        end = target;
                    }
                    next.push(end);
                }
            }
            frontier = next;
        }
        horizon
    }

    #[test]
    fn raw_terminal_radius_matches_exhaustive_ascii_and_multibyte_words() {
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec()),
                                    (2, "é".as_bytes().to_vec())]);
        let fixtures: [(&str, &str, Vec<&[u8]>); 3] = [
            (r"[ab]{1,4}", r"[ab]+", vec![b"a", b"b"]),
            (r"a{1,4}", r"[ab]+", vec![b"a", b"b"]),
            (r"(?:a|é){1,4}", r"(?:a|\xC3\xA9)+", vec![b"a", "é".as_bytes()]),
        ];
        let mut positive = 0usize;
        let mut checked = 0usize;
        for (pattern, slice_pattern, atoms) in fixtures {
            let grammar = format!("glrm 1; start s; t WORD = /{pattern}/; nt s = WORD;");
            let constraint = Constraint::compile(Grammar::glrm(&grammar), &vocab).unwrap();
            let terminal = constraint.terminal_display_names.iter()
                .position(|name| name == "WORD").unwrap() as TerminalID;
            let slice = VocabPartitionDfa::compile_byte_regex("test-atoms", slice_pattern).unwrap();
            for source in 0..constraint.tokenizer.num_states() {
                if constraint.tokenizer.state_has_epsilon_transitions(source)
                    || constraint.tokenizer.state_is_virtual_runtime(source) { continue; }
                let expected = reference_radius(&constraint.tokenizer, terminal, source, &atoms, 6);
                let actual = constraint.dynamic_mask_vocab.raw_terminal_slice_repeat_radius(
                    &constraint.tokenizer, terminal, source, 901, &slice, 6, 65_536,
                );
                assert_eq!(actual, Some(expected), "pattern={pattern} source={source}");
                checked += 1;
                positive += usize::from(expected > 0);
                // Budget exhaustion never creates a certificate, and its key
                // cannot poison a subsequent larger-budget query.
                assert_eq!(constraint.dynamic_mask_vocab.raw_terminal_slice_repeat_radius(
                    &constraint.tokenizer, terminal, source, 901, &slice, 6, 0,
                ), Some(0));
                assert_eq!(constraint.dynamic_mask_vocab.raw_terminal_slice_repeat_radius(
                    &constraint.tokenizer, terminal, source, 901, &slice, 6, 65_536,
                ), actual);
            }
            let empty = VocabPartitionDfa::compile_byte_regex("nullable", r"a*").unwrap();
            assert_eq!(constraint.dynamic_mask_vocab.raw_terminal_slice_repeat_radius(
                &constraint.tokenizer, terminal, 0, 902, &empty, 6, 65_536,
            ), None);
            assert_eq!(constraint.dynamic_mask_vocab.raw_terminal_slice_repeat_radius(
                &constraint.tokenizer, terminal, u32::MAX, 901, &slice, 6, 65_536,
            ), None);
        }
        assert!(checked > 4 && positive > 0, "must exercise real positive and negative certificates");
    }
}

#[cfg(test)]
#[test]
fn direct_flat_mask_trie_preserves_all_reference_runtime_fields() {
    fn check(entries: &[(u16, usize, &[u8])]) {
        let reference = DynamicMaskTrie::from_partitioned_token_refs_reference(entries);
        let direct = DynamicMaskTrie::from_partitioned_sorted_direct(entries);
        assert_eq!(bincode::serialize(&reference).unwrap(), bincode::serialize(&direct).unwrap(),
            "all nodes, edges, byte layout, metadata, walk ops and language classes must agree");
    }
    check(&[]);
    let mut random = 7u64;
    for case in 0..512 {
        let mut next = || {random ^= random << 13; random ^= random >> 7; random ^= random << 17; random};
        let mut words = vec![(0u16, Vec::<u8>::new())];
        for _ in 0..case % 150 {
            let class = (next() % 5) as u16;
            let len = (next() % 32) as usize;
            words.push((class, (0..len).map(|_|(next() % 256) as u8).collect()));
        }
        words.sort(); words.dedup();
        let entries = words.iter().enumerate().map(|(i,(class,word))|
            (*class,i*17,word.as_slice())).collect::<Vec<_>>();
        check(&entries);
    }
    for depth in [1,16,64,256,512] {
        let words = (0..depth).map(|n|vec![b'a';n]).collect::<Vec<_>>();
        let entries = words.iter().enumerate().map(|(i,word)|(0u16,i,word.as_slice())).collect::<Vec<_>>();
        check(&entries);
    }
}
