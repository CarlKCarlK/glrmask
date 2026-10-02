mod boundary_reset_support;
pub(crate) mod packed_final_masks;
mod scoped_admission_support;
pub(crate) use scoped_admission_support::ScopedAdmissionSupport;

use crate::automata::lexer::{
    tokenizer::{Tokenizer, TokenizerStateSet},
    Lexer,
};
use crate::automata::regex::Expr;
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex, OnceLock};

use range_set_blaze::RangeSetBlaze;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use rayon::prelude::*;

use crate::automata::weighted::dwa::{
    DWA, PackedRuntimeTokenSetRef, PackedRuntimeWeightRef,
};
use crate::compiler::glr::accumulator::TerminalsDisallowed;
use crate::compiler::glr::labels::{encode_positive_label, DEFAULT_LABEL};
use crate::compiler::glr::parser::{
    DisjointComponentActionProvider, ParserComponentTableSource, ParserGSS, ScopedParserSymbol,
    advance_provider_control_closed_stacks, close_provider_control_stacks,
    materialize_control_eliminated_scoped_provider_table,
    stack_may_advance_on_with_provider, stack_may_advance_on_any_with_provider,
    for_each_admitted_symbol_with_provider, find_admitted_symbol_with_provider,
    stacks_finished_with_provider,
};
use crate::compiler::glr::table::{Action, GLRTable, TableAmbiguity, subgrammar_child_return_pop};
use crate::compiler::stages::id_map_and_terminal_dwa::classify::{
    VocabPartitionDfa, classify_vocab_char_type,
};
use crate::ds::bitset::BitSet;
use crate::ds::u8set::U8Set;
use crate::ds::weight::{PackedRuntimePoolTokenSetRef, PackedRuntimePoolWeightRef, Weight};
use crate::grammar::flat::TerminalID;

use super::artifact::{
    dynamic_mask_vocab_layout_class, empty_dense_words, DenseAcceptanceRows,
    DirectRegularDynamicFrontierCacheEntry,
    DirectRegularDynamicHotFrontier, DirectRegularTerminalSupport,
    DirectRegularParserStateAcceptance, DirectRegularWideFrontierAcceptance,
    DenseBufMaskRows, DenseWeightBufMaskCache,
    DenseWeightMaskCache,
    DenseWords,
    DynamicFirstMatchPostRow,
    DynamicFirstMatchSecondRow,
    DynamicBoundedObservationSets, DynamicConfigSubtreeCertificate,
    DynamicSelfLoopProjection,
    DirectSparseWeightTokenSetCache,
    PackedDynamicMaskTokenAliases,
    RecursiveParserLayout, RecursiveParserLeafLayout,
    DynamicMaskTrie,
    DynamicMaskTrieEdge,
    DynamicMaskVocab,
    FastCommitTemplateDfas, FastDwaTransitionRow, FastDwaTransitions,
    FastTemplateDfasByTerminal, FastTokenizerTransitions,
    IndexedDagDenseMask, IndexedDagDenseTransition, IndexedDagDenseTransitionMasks,
    IndexedDagDenseTransitionRow, IndexedDagDenseTransitions,
    InternalTokenBufMasks, PackedDwaDenseWeightMaskCache, PackedInternalTokenBufMask,
    SeedTerminalDenseMasks,
    SparseWeightBufMaskCache,
};
pub use super::artifact::Constraint;
pub(crate) use super::mask_mapping::{DeltaReplayProfileStats, DenseToBufProfileStats};
use super::mask_mapping::FinalMaskMapping;
use super::state::ConstraintState;

struct RecursiveSegmentedParserTables<'a> {
    root: &'a Constraint,
    layout: &'a RecursiveParserLayout,
}

impl RecursiveSegmentedParserTables<'_> {
    #[inline]
    fn leaf_constraint(&self, component: u32) -> Option<&Constraint> {
        let leaf = self.layout.leaves.get(component as usize)?;
        self.root
            .constraint_at_recursive_component_path(&leaf.component_path)
    }
}

impl ParserComponentTableSource for RecursiveSegmentedParserTables<'_> {
    #[inline]
    fn component_count(&self) -> usize {
        self.layout.leaves.len()
    }

    #[inline]
    fn component_table(&self, component: u32) -> Option<&GLRTable> {
        self.leaf_constraint(component).map(|constraint| &*constraint.table)
    }

    #[inline]
    fn component_ignore_terminal(&self, component: u32) -> Option<TerminalID> {
        let constraint = self.leaf_constraint(component)?;
        // Same idempotence contract: composed leaf tables own their scoped
        // ignores in-row; only raw leaves need provider-level Identity.
        let ignore = constraint.ignore_terminal?;
        (!constraint.parser_skip_terminals().contains(&ignore)).then_some(ignore)
    }
}

/// Slow recursive oracle: inspect immutable component relations directly,
/// without prepared shift shortcuts or control-admission certificates.
struct RecursiveTemplateReference<'a> {
    parser: &'a super::parser_backend::TemplateParser,
    composition: &'a super::parser_backend::composition::TemplateComposition,
}

impl RecursiveTemplateReference<'_> {
    fn advance_view(
        view: &super::parser_backend::scoped_program::ScopedProgram,
        stack: &ParserGSS,
    ) -> ParserGSS {
        let input = if view.guard_owner {
            let mut input = ParserGSS::empty();
            for top in stack.peek_values() {
                if view.local(top).is_some() {
                    input = input.merge(&stack.isolate(Some(top)));
                }
            }
            input
        } else {
            stack.clone()
        };
        let output = super::commit::template_advance::advance_with_template_coordinate(
            &view.source, input, Some((view.offset, view.symbols)),
        );
        match view.append_push {
            Some(symbol) => output.push(symbol),
            None => output,
        }
    }
}

impl crate::compiler::glr::parser::ParserActionProvider for RecursiveTemplateReference<'_> {
    type Symbol = u32;

    fn action(&self, _: u32, _: u32) -> Option<crate::compiler::glr::parser::ProvidedAction<'_>> {
        unreachable!("native recursive reference cannot inspect LR actions")
    }
    fn scope_state(&self, _: u32, _: u32) -> Option<u32> {
        unreachable!("native recursive reference already uses scoped symbols")
    }
    fn goto_target(&self, _: u32, _: u32, _: u32) -> Option<(u32, bool)> {
        unreachable!("native recursive reference cannot inspect LR gotos")
    }
    fn state_count_hint(&self) -> usize { self.parser.state_count as usize }
    fn control_symbols(&self, _: u32, out: &mut SmallVec<[u32; 4]>) {
        // Enumerate every explicit crossing; the exact interpreter decides
        // feasibility, independently of the production top certificates.
        out.extend((self.composition.control_start..self.composition.views.len() as u32)
            .map(|index| self.parser.terminal_count + index));
    }
    fn advance_relation(&self, stack: &ParserGSS, terminal: u32) -> Option<ParserGSS> {
        let view = if terminal < self.parser.terminal_count {
            self.composition.outer_views.get(terminal as usize)
        } else if terminal != crate::compiler::glr::analysis::EOF {
            self.composition.views.get((terminal - self.parser.terminal_count) as usize)
        } else { None };
        Some(view.map_or_else(ParserGSS::empty, |view| Self::advance_view(view, stack)))
    }
    fn relation_finished(&self, stack: &ParserGSS, terminal: u32) -> Option<bool> {
        Some(terminal == crate::compiler::glr::analysis::EOF
            && self.composition.completion_view.as_ref()
                .is_some_and(|view| !Self::advance_view(view, stack).is_empty()))
    }
}

#[derive(Default)]
struct DirectSparseWeightBufCaches {
    eligible: DirectSparseWeightTokenSetCache,
    fallback: Vec<(usize, Arc<RangeSetBlaze<u32>>)>,
}

/// Finalization-local view of the parser-DWA token sets.  The final and
/// transition cache builders share it so finalization traverses the DWA once.
struct WeightTokenSetInventory {
    final_sets: Vec<(usize, Arc<RangeSetBlaze<u32>>)>,
    transition_sets: FxHashMap<usize, Arc<RangeSetBlaze<u32>>>,
    transition_word_spans: Option<FxHashMap<usize, u32>>,
}

#[derive(Clone, Copy)]
pub(crate) enum RuntimeTokenSetRef<'a> {
    Materialized(&'a Arc<RangeSetBlaze<u32>>),
    PackedDwa(PackedRuntimeTokenSetRef<'a>),
    PackedPool(PackedRuntimePoolTokenSetRef<'a>),
}

impl<'a> RuntimeTokenSetRef<'a> {
    #[inline]
    pub(crate) fn is_empty(self) -> bool {
        match self {
            Self::Materialized(tokens) => tokens.is_empty(),
            Self::PackedDwa(tokens) => tokens.range_count() == 0,
            Self::PackedPool(tokens) => tokens.is_empty(),
        }
    }

    #[inline]
    pub(crate) fn materialized_key(self) -> Option<usize> {
        match self {
            Self::Materialized(tokens) => Some(Arc::as_ptr(tokens) as usize),
            Self::PackedDwa(_) | Self::PackedPool(_) => None,
        }
    }

    #[inline]
    pub(crate) fn packed_id(self) -> Option<u32> {
        match self {
            Self::Materialized(_) => None,
            Self::PackedDwa(tokens) => Some(tokens.id()),
            Self::PackedPool(_) => None,
        }
    }

    #[inline]
    pub(crate) fn packed_pool_id(self) -> Option<u32> {
        match self {
            Self::PackedPool(tokens) => Some(tokens.id()),
            _ => None,
        }
    }

    #[inline]
    pub(crate) fn for_each_range(self, mut f: impl FnMut(u32, u32)) {
        match self {
            Self::Materialized(tokens) => {
                for range in tokens.ranges() {
                    f(*range.start(), *range.end());
                }
            }
            Self::PackedDwa(tokens) => {
                tokens.for_each_range(f);
            }
            Self::PackedPool(tokens) => {
                tokens.for_each_range(f);
            }
        }
    }

    #[inline]
    pub(crate) fn word_spans(self) -> Option<u32> {
        match self {
            Self::Materialized(_) => None,
            Self::PackedDwa(tokens) => Some(tokens.word_spans()),
            Self::PackedPool(_) => None,
        }
    }

    pub(crate) fn to_range_set(self) -> RangeSetBlaze<u32> {
        match self {
            Self::Materialized(tokens) => tokens.as_ref().clone(),
            packed => {
                let mut ranges = Vec::new();
                packed.for_each_range(|start, end| ranges.push(start..=end));
                RangeSetBlaze::from_iter(ranges)
            }
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum RuntimeWeightRef<'a> {
    Materialized(&'a Weight),
    PackedDwa(PackedRuntimeWeightRef<'a>),
    PackedPool(PackedRuntimePoolWeightRef<'a>),
    IndexedPool(&'a packed_final_masks::PackedWeightIndex),
}

impl<'a> RuntimeWeightRef<'a> {
    #[inline]
    pub(crate) fn is_full(self) -> bool {
        match self {
            Self::Materialized(weight) => weight.is_full(),
            Self::PackedDwa(weight) => weight.is_full(),
            Self::PackedPool(weight) => weight.is_full(),
            Self::IndexedPool(weight) => weight.is_full(),
        }
    }

    #[inline]
    pub(crate) fn is_empty(self) -> bool {
        match self {
            Self::Materialized(weight) => weight.is_empty(),
            Self::PackedDwa(weight) => weight.is_empty(),
            Self::PackedPool(weight) => weight.is_empty(),
            Self::IndexedPool(weight) => weight.is_empty(),
        }
    }

    #[inline]
    pub(crate) fn token_set_for_tsid(self, tsid: u32) -> Option<RuntimeTokenSetRef<'a>> {
        match self {
            Self::Materialized(weight) => weight
                .token_set_for_tsid_ref(tsid)
                .map(RuntimeTokenSetRef::Materialized),
            Self::PackedDwa(weight) => weight.token_set_for_tsid(tsid).map(|tokens| {
                tokens.materialized_arc().map_or(
                    RuntimeTokenSetRef::PackedDwa(tokens),
                    RuntimeTokenSetRef::Materialized,
                )
            }),
            Self::PackedPool(weight) => weight
                .token_set_for_tsid(tsid)
                .map(RuntimeTokenSetRef::PackedPool),
            Self::IndexedPool(weight) => weight
                .token_set_for_tsid(tsid)
                .map(RuntimeTokenSetRef::PackedPool),
        }
    }

    #[inline]
    pub(crate) fn to_weight(self) -> Weight {
        if self.is_full() {
            return Weight::all();
        }
        if self.is_empty() {
            return Weight::empty();
        }
        let mut entries = Vec::<(u32, RangeSetBlaze<u32>)>::new();
        self.for_each_entry(|start, end, tokens| {
            let tokens = tokens.to_range_set();
            entries.extend((start..=end).map(|tsid| (tsid, tokens.clone())));
        });
        Weight::from_per_tsid_token_sets(entries)
    }

    pub(crate) fn for_each_entry(
        self,
        mut f: impl FnMut(u32, u32, RuntimeTokenSetRef<'a>),
    ) {
        match self {
            Self::Materialized(weight) => {
                if weight.is_full() {
                    return;
                }
                for (range, tokens) in weight.raw_range_values() {
                    f(
                        *range.start(),
                        *range.end(),
                        RuntimeTokenSetRef::Materialized(tokens),
                    );
                }
            }
            Self::PackedDwa(weight) => {
                for ((start, end), tokens) in weight.entries() {
                    let tokens = tokens.materialized_arc().map_or(
                        RuntimeTokenSetRef::PackedDwa(tokens),
                        RuntimeTokenSetRef::Materialized,
                    );
                    f(start, end, tokens);
                }
            }
            Self::PackedPool(weight) => {
                for ((start, end), tokens) in weight.entries() {
                    f(start, end, RuntimeTokenSetRef::PackedPool(tokens));
                }
            }
            Self::IndexedPool(weight) => weight.for_each_entry(|start, end, tokens| {
                f(start, end, RuntimeTokenSetRef::PackedPool(tokens));
            }),
        }
    }
}

impl<'a> From<&'a Weight> for RuntimeWeightRef<'a> {
    #[inline]
    fn from(weight: &'a Weight) -> Self {
        Self::Materialized(weight)
    }
}

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
fn or_sparse_buf_entries(buf: &mut [u32], entries: &[(u16, u32)]) {
    for &(word_idx, mask) in entries {
        unsafe {
            let slot = buf.get_unchecked_mut(word_idx as usize);
            *slot |= mask;
        }
    }
}

#[inline(always)]
fn andnot_sparse_buf_entries(buf: &mut [u32], entries: &[(u16, u32)]) {
    for &(word_idx, mask) in entries {
        unsafe {
            let slot = buf.get_unchecked_mut(word_idx as usize);
            *slot &= !mask;
        }
    }
}

#[inline(always)]
fn pack_internal_token_buf_entry(word_idx: u16, mask: u32) -> PackedInternalTokenBufMask {
    PackedInternalTokenBufMask {
        word_idx,
        _pad: 0,
        mask,
    }
}

#[inline(always)]
fn unpack_internal_token_buf_entry(entry: PackedInternalTokenBufMask) -> (u16, u32) {
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
fn group_buf_mask_cost(sparse: &[(u16, u32)], dense: Option<&[u32]>) -> usize {
    dense.map_or(sparse.len(), <[u32]>::len)
}

#[inline(always)]
fn or_group_buf_mask(
    buf: &mut [u32],
    sparse: &[(u16, u32)],
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
fn andnot_group_buf_mask(
    buf: &mut [u32],
    sparse: &[(u16, u32)],
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

const INITIAL_COMMIT_PRIME_MAX_TOKENS: usize = 16;

fn initial_commit_prime_token_ids(mask: &[u32]) -> Option<Vec<u32>> {
    let mut token_ids = Vec::new();
    for (word_index, &word) in mask.iter().enumerate() {
        let mut remaining = word;
        while remaining != 0 {
            if token_ids.len() == INITIAL_COMMIT_PRIME_MAX_TOKENS {
                return None;
            }
            let bit = remaining.trailing_zeros() as usize;
            token_ids.push((word_index * 32 + bit) as u32);
            remaining &= remaining - 1;
        }
    }
    Some(token_ids)
}

/// Select at most the existing priming budget from the actual initial mask.
/// A wide initial language does not prevent bounded template-kernel priming.
/// This selects only allowed token IDs; it never changes the initial state or
/// the tokens admitted by the compiled constraint.
fn template_initial_prime_token_ids(mask: &[u32]) -> Vec<u32> {
    let mut token_ids = Vec::new();
    for (word_index, &word) in mask.iter().enumerate() {
        let mut remaining = word;
        while remaining != 0 {
            if token_ids.len() == INITIAL_COMMIT_PRIME_MAX_TOKENS {
                return token_ids;
            }
            let bit = remaining.trailing_zeros() as usize;
            token_ids.push((word_index * 32 + bit) as u32);
            remaining &= remaining - 1;
        }
    }
    token_ids
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
        let mut masks = vec![Vec::<(u16, u32)>::new(); internal_to_tokens.len()];
        for (original, &internal) in original_to_internal.iter().enumerate() {
            if internal == u32::MAX {
                continue;
            }
            let Some(mask) = masks.get_mut(internal as usize) else {
                continue;
            };
            let word = (original as u32 / 32) as u16;
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
                let mut touched = Vec::<u16>::new();
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

#[derive(Default)]
struct TokenMaskCacheBuildProfile {
    word_block_ms: f64,
    quad_block_ms: f64,
    byte_block_ms: f64,
    block_ms: f64,
    pair_ms: f64,
    quad_ms: f64,
    super_ms: f64,
    mega_ms: f64,
    giga_ms: f64,
    all_tokens_ms: f64,
    heavy_ms: f64,
    flat_ms: f64,
    costs_ms: f64,
    derived_ms: f64,
}

fn group_dynamic_effect_post_rows(
    by_terminal: BTreeMap<TerminalID, Vec<u32>>,
) -> Vec<DynamicFirstMatchPostRow> {
    let mut terminals_by_tokens = BTreeMap::<Vec<u32>, Vec<TerminalID>>::new();
    for (terminal, mut tokens) in by_terminal {
        tokens.sort_unstable();
        tokens.dedup();
        if !tokens.is_empty() {
            terminals_by_tokens.entry(tokens).or_default().push(terminal);
        }
    }
    let mut rows = terminals_by_tokens
        .into_iter()
        .map(|(tokens, mut terminals)| {
            terminals.sort_unstable();
            terminals.dedup();
            dynamic_effect_post_row(terminals, tokens)
        })
        .collect::<Vec<_>>();
    rows.sort_unstable_by(|left, right| {
        std::cmp::Reverse(left.tokens.len())
            .cmp(&std::cmp::Reverse(right.tokens.len()))
            .then_with(|| left.terminals.as_ref().cmp(right.terminals.as_ref()))
    });
    rows
}

fn dynamic_effect_post_row(
    terminals: Vec<TerminalID>,
    tokens: Vec<u32>,
) -> DynamicFirstMatchPostRow {
    // Above a few cache lines of token IDs, applying the row wordwise is both
    // cheaper and more predictable.  Keep sparse rows sparse so the common
    // tiny-mask case does not pay a 16 KiB dense-row scan.
    const DENSE_MIN_TOKENS: usize = 1_024;
    let dense_mask = if tokens.len() >= DENSE_MIN_TOKENS {
        let words = tokens
            .last()
            .copied()
            .map_or(0usize, |token| token as usize / 32 + 1);
        let mut dense = vec![0u32; words];
        for &token in &tokens {
            dense[token as usize / 32] |= 1u32 << (token % 32);
        }
        dense
    } else {
        Vec::new()
    };
    DynamicFirstMatchPostRow {
        terminals: Arc::from(terminals),
        tokens: Arc::from(tokens),
        dense_mask: Arc::from(dense_mask),
    }
}

/// Build a reset-rooted lexical-effect program for concrete vocabulary
/// suffixes.  Each row consumes one lexer terminal and then recursively
/// describes the suffix after the lexer reset.  Residual no-finalization paths
/// are returned separately as parser-admissibility rows at the current parser
/// state.  Tokens whose match sequence exceeds `depth_left` are accumulated in
/// `unresolved` for exact runtime fallback.
fn build_dynamic_reset_effect_rows(
    tokenizer: &Tokenizer,
    mut entries: Vec<(u32, Vec<u8>)>,
    depth_left: usize,
    unresolved: &mut BTreeSet<u32>,
) -> (Vec<DynamicFirstMatchPostRow>, Vec<DynamicFirstMatchSecondRow>) {
    entries.sort_unstable();
    entries.dedup();
    let mut residual_by_terminal = BTreeMap::<TerminalID, Vec<u32>>::new();
    let mut exact_by_terminal = BTreeMap::<TerminalID, Vec<u32>>::new();
    let mut children_by_terminal = BTreeMap::<TerminalID, Vec<(u32, Vec<u8>)>>::new();

    for (token_id, bytes) in entries {
        if bytes.is_empty() {
            unresolved.insert(token_id);
            continue;
        }
        let execution = tokenizer.execute_from_state_all_widths(&bytes, tokenizer.start_state());
        let mut futures = BitSet::new(tokenizer.num_terminals() as usize);
        for &end_state in &execution.end_state {
            futures.union_with_prefix(tokenizer.possible_future_terminals(end_state));
        }
        for future_terminal in futures.iter_ones() {
            residual_by_terminal
                .entry(future_terminal as TerminalID)
                .or_default()
                .push(token_id);
        }

        let mut branches = execution
            .matches
            .iter()
            .map(|matched| (matched.id, matched.width))
            .collect::<Vec<_>>();
        branches.sort_unstable();
        branches.dedup();
        for (terminal, width) in branches {
            if width == 0 || width > bytes.len() {
                unresolved.insert(token_id);
            } else if width == bytes.len() {
                exact_by_terminal.entry(terminal).or_default().push(token_id);
            } else if depth_left == 0 {
                unresolved.insert(token_id);
            } else {
                children_by_terminal
                    .entry(terminal)
                    .or_default()
                    .push((token_id, bytes[width..].to_vec()));
            }
        }
    }

    let post_rows = group_dynamic_effect_post_rows(residual_by_terminal);
    let mut terminals = BTreeSet::<TerminalID>::new();
    terminals.extend(exact_by_terminal.keys().copied());
    terminals.extend(children_by_terminal.keys().copied());
    let mut rows = Vec::<DynamicFirstMatchSecondRow>::new();
    for terminal in terminals {
        let mut exact_end_tokens = exact_by_terminal.remove(&terminal).unwrap_or_default();
        exact_end_tokens.sort_unstable();
        exact_end_tokens.dedup();
        let child_entries = children_by_terminal.remove(&terminal).unwrap_or_default();
        let (child_post_rows, next_rows) = if child_entries.is_empty() {
            (Vec::new(), Vec::new())
        } else {
            build_dynamic_reset_effect_rows(
                tokenizer,
                child_entries,
                depth_left.saturating_sub(1),
                unresolved,
            )
        };
        if !exact_end_tokens.is_empty() || !child_post_rows.is_empty() || !next_rows.is_empty() {
            rows.push(DynamicFirstMatchSecondRow {
                terminal,
                exact_end_tokens: Arc::from(exact_end_tokens),
                post_rows: Arc::from(child_post_rows),
                next_rows: Arc::from(next_rows),
            });
        }
    }
    rows.sort_unstable_by_key(|row| row.terminal);
    (post_rows, rows)
}



impl Constraint {
    /// Build the parser-state-independent trigger level used by dynamic
    /// composition. This is deliberately optional: ordinary Constraint and
    /// DynamicConstraint compilation never calls it, preserving zero trigger
    /// build cost by default.
    ///
    /// The scan starts from every tokenizer state, so the resulting original
    /// model-token set is conservative for any runtime TSID. A token is kept
    /// when a proper prefix can complete a terminal that may either finish the
    /// component or immediately precede one of its unresolved grammar slots.
    pub(crate) fn build_boundary_token_trigger(&mut self) -> Result<(), String> {
        self.materialize_composition_link_metadata_for_compilation()?;

        let mut relevant = BitSet::new(self.parser_terminal_count() as usize);
        if let Some(summary) = self.composition_grammar_summary.as_ref() {
            relevant.union_with(&summary.root_last);
            let placeholders = self
                .unbound_grammar_placeholders
                .values()
                .copied()
                .chain(self.late_grammar_slots.iter().map(|slot| slot.terminal_id));
            for placeholder in placeholders {
                for (terminal, follows) in summary.allowed_follows.iter().enumerate() {
                    if follows.contains(placeholder as usize) {
                        relevant.set(terminal);
                    }
                }
            }
            // Scoped ignore can occur immediately before an entry/finish point
            // without appearing in grammar adjacency. A model token may
            // therefore consume one or more ignore lexemes and then cross the
            // component boundary internally. Tokens is intentionally an
            // overapproximation, so treating any matched ignore as a possible
            // boundary precursor is the sound parser-state-independent choice.
            if let Some(ignore) = self.ignore_terminal {
                relevant.set(ignore as usize);
            }
        } else {
            // Older/stripped artifacts may lack the grammar summary. The
            // Tokens trigger is only a pruning accelerator, so falling back to
            // every terminal preserves exactness.
            relevant = BitSet::all(self.parser_terminal_count() as usize);
        }

        if relevant.is_empty() {
            self.boundary_trigger =
                crate::runtime::BoundaryTrigger::Tokens(Arc::from([]));
            self.serialized_artifact_cache = None;
            return Ok(());
        }

        let all_states = (0..self.tokenizer.num_states()).collect::<Vec<_>>();
        let tokens = self.token_bytes_iter().collect::<Vec<_>>();
        let mut candidates = tokens
            .par_iter()
            .filter_map(|(token_id, bytes)| {
                if bytes.len() < 2 {
                    return None;
                }
                let mut states = TokenizerStateSet::from_iter(all_states.iter().copied());
                for &byte in &bytes[..bytes.len() - 1] {
                    states = self.tokenizer.step_all(states.as_slice(), byte);
                    if states.is_empty() {
                        return None;
                    }
                    let mut matched_any = false;
                    let mut matched_relevant = false;
                    for state in states.iter().copied() {
                        for terminal in self.tokenizer.matched_terminals_iter(state) {
                            matched_any = true;
                            matched_relevant |= relevant.contains(terminal as usize);
                        }
                    }
                    if matched_relevant {
                        return Some(*token_id);
                    }
                    if matched_any {
                        // A real lexer commits only according to its exact
                        // longest-match policy. Forking a reset continuation at
                        // every match is deliberately broader: it preserves all
                        // real multi-lexeme paths (including ignore -> reset ->
                        // terminal -> boundary) while admitting only harmless
                        // false-positive trigger tokens.
                        let reset = self.runtime_commit_initial_state();
                        if !states.contains(&reset) {
                            states.push(reset);
                            states.sort_unstable();
                        }
                    }
                }
                None
            })
            .collect::<Vec<_>>();
        candidates.sort_unstable();
        candidates.dedup();
        self.boundary_trigger = crate::runtime::BoundaryTrigger::Tokens(Arc::from(
            candidates.into_boxed_slice(),
        ));
        self.serialized_artifact_cache = None;
        Ok(())
    }

    /// Build reusable boundary-trigger metadata at the requested detail level.
    ///
    /// `None` is a no-op and preserves the zero-cost default. `Tokens` builds
    /// only the conservative parser-state-independent token set. `Exact` first
    /// builds that set as a candidate prefilter, then upgrades to a full local
    /// Parser DWA when this component supports exact trigger compilation.
    pub fn build_boundary_trigger(
        &mut self,
        detail: crate::runtime::BoundaryTriggerDetail,
    ) -> Result<(), String> {
        match detail {
            crate::runtime::BoundaryTriggerDetail::None => Ok(()),
            crate::runtime::BoundaryTriggerDetail::Tokens => self.build_boundary_token_trigger(),
            crate::runtime::BoundaryTriggerDetail::Exact => self.build_exact_boundary_trigger(),
        }
    }

    /// Build the full GSS-sensitive proper-prefix boundary trigger Parser DWA.
    ///
    /// This is an optional accelerator for dynamic composition and is
    /// deliberately separate from ordinary constraint compilation. The trigger
    /// uses raw local tokenizer-state IDs and original model-token IDs rather
    /// than the component's whole-token TSID/token quotient. Recursive
    /// coordinators build a temporary lexical union and query their immutable
    /// scoped template relations. No parser table is reconstructed.
    pub fn build_exact_boundary_trigger(&mut self) -> Result<(), String> {
        if matches!(self.boundary_trigger, crate::runtime::BoundaryTrigger::Exact(_)) {
            return Ok(());
        }

        // The trigger is an optional query artifact. Components retain their
        // own lexer and template coordinates throughout its construction.
        self.build_boundary_token_trigger()?;
        let candidates = if self.uses_compact_segmented_parser_runtime() {
            // The coordinator retains its root lexer, so its leaf-only token
            // trigger cannot certify the whole recursive interface. A checked
            // component summary may prune; otherwise widen conservatively.
            crate::compiler::boundary_candidates::persisted_boundary_candidate_ids(self,
                self.late_bind_vocab.get().ok_or("exact recursive trigger requires its vocabulary")?)?
                .unwrap_or_else(|| self.token_bytes_iter().filter(|(_, bytes)| bytes.len() >= 2)
                    .map(|(id, _)| id).collect())
        } else {
            self.boundary_trigger.token_summary().map(|tokens| tokens.to_vec()).unwrap_or_default()
        };
        let dwa = crate::compiler::constraint_compose::build_exact_component_boundary_trigger(self, &candidates)?;
        let Some(dwa) = dwa else {
            return Err(
                "exact boundary trigger construction could not characterize the component parser"
                    .to_owned(),
            );
        };
        self.boundary_trigger = crate::runtime::BoundaryTrigger::Exact(Arc::new(dwa));
        self.serialized_artifact_cache = None;
        Ok(())
    }

    /// Whether the stored internal-TSID inverse is redundant with the scalar
    /// state -> TSID map. Runtime-product tokenizers can have several TSID
    /// lanes per physical state and therefore must retain their explicit
    /// inverse/relation metadata.
    pub(crate) fn can_defer_internal_tsid_inverse(&self) -> bool {
        if self.runtime_source_state_offset.is_some()
            || self.state_to_internal_tsid.len() != self.tokenizer.num_states() as usize
            || self.state_to_internal_tsid.iter().any(|&tsid| tsid == u32::MAX)
        {
            return false;
        }
        let Some(count) = self
            .state_to_internal_tsid
            .iter()
            .copied()
            .max()
            .map(|max| max as usize + 1)
        else {
            return self.internal_tsid_to_states.is_empty();
        };
        // A previously loaded omitted inverse is already certified by the
        // current-format writer. The complete scalar map is sufficient to
        // reconstruct it exactly.
        if self.internal_tsid_to_states.is_empty() {
            return true;
        }
        if self.internal_tsid_to_states.len() != count {
            return false;
        }
        let mut entries = 0usize;
        for (tsid, states) in self.internal_tsid_to_states.iter().enumerate() {
            entries = entries.saturating_add(states.len());
            for &state in states {
                if self
                    .state_to_internal_tsid
                    .get(state as usize)
                    .copied()
                    != Some(tsid as u32)
                {
                    return false;
                }
            }
        }
        entries == self.state_to_internal_tsid.len()
    }

    pub(crate) fn internal_tsid_count(&self) -> usize {
        if !self.internal_tsid_to_states.is_empty() {
            return self.internal_tsid_to_states.len();
        }
        if let Some(groups) = self.deferred_internal_tsid_to_states.get() {
            return groups.len();
        }
        self.state_to_internal_tsid
            .iter()
            .copied()
            .filter(|&tsid| tsid != u32::MAX)
            .max()
            .map_or(0, |max| max as usize + 1)
    }

    pub(crate) fn internal_tsid_groups(&self) -> &[Vec<u32>] {
        if !self.internal_tsid_to_states.is_empty() {
            return &self.internal_tsid_to_states;
        }
        let groups = self.deferred_internal_tsid_to_states.get_or_init(|| {
            let mut groups = vec![Vec::new(); self.internal_tsid_count()];
            for (state, &tsid) in self.state_to_internal_tsid.iter().enumerate() {
                if tsid != u32::MAX
                    && let Some(states) = groups.get_mut(tsid as usize)
                {
                    states.push(state as u32);
                }
            }
            groups
        });
        groups.as_slice()
    }

    /// Terminal source expressions retained for later compiled-constraint
    /// composition. Fresh constraints keep them directly on the tokenizer;
    /// current loaded artifacts may keep only the canonical serialized blob so
    /// ordinary static load/mask/commit does not rebuild expression trees.
    pub(crate) fn retained_terminal_exprs(&self) -> Option<&[Expr]> {
        if let Some(exprs) = self.tokenizer.terminal_exprs() {
            return Some(exprs);
        }
        if let Some(exprs) = self.deferred_terminal_exprs.get() {
            return Some(exprs.as_ref());
        }
        let decoded = self.deferred_terminal_exprs_blob.as_ref()?.decode_exprs().ok()?;
        if decoded.len() != self.tokenizer.num_terminals() as usize {
            return None;
        }
        let decoded = Arc::<[Expr]>::from(decoded.into_boxed_slice());
        let _ = self.deferred_terminal_exprs.set(decoded);
        self.deferred_terminal_exprs.get().map(Arc::as_ref)
    }

    #[inline]
    pub(crate) fn retained_terminal_expr(&self, terminal: TerminalID) -> Option<&Expr> {
        self.retained_terminal_exprs()?.get(terminal as usize)
    }

    /// Return the complete source grammar rules for composition. Ordinary
    /// runtime execution never consults these rules, so large current-format
    /// artifacts may retain their canonical bincode payload and decode it only
    /// when a later composition actually needs grammar structure.
    pub(crate) fn retained_table_rules(&self) -> Result<&[crate::grammar::flat::Rule], String> {
        if let Some(parser) = &self.template_parser {
            return parser.link_grammar.as_ref().map(|grammar| grammar.rules.as_ref())
                .ok_or_else(|| "template parser has no source grammar metadata".to_owned());
        }
        if self.deferred_table_rules_blob.is_none() {
            return Ok(&self.table.rules);
        }
        if let Some(rules) = self.deferred_table_rules.get() {
            return Ok(rules.as_ref());
        }
        let blob = self
            .deferred_table_rules_blob
            .as_ref()
            .ok_or_else(|| "missing deferred GLR rules payload".to_owned())?;
        let decoded = bincode::deserialize::<Vec<crate::grammar::flat::Rule>>(blob.as_slice())
            .map_err(|err| err.to_string())?;
        if decoded.len() != self.table.num_rules as usize {
            return Err("deferred GLR rule count does not match table num_rules".to_owned());
        }
        if decoded.first() != self.table.rules.first() {
            return Err("deferred GLR augmented-start rule mismatch".to_owned());
        }
        let decoded = Arc::<[crate::grammar::flat::Rule]>::from(decoded.into_boxed_slice());
        let _ = self.deferred_table_rules.set(decoded);
        self.deferred_table_rules
            .get()
            .map(Arc::as_ref)
            .ok_or_else(|| "failed to install deferred GLR rules".to_owned())
    }

    /// Exact root nullability as seen by a later linker.
    ///
    /// A composed child's descendants can change whether its root language is
    /// nullable, so the intact root leaf table is not sufficient here. The
    /// retained composition grammar summary is the exact wrapper-language
    /// fact when available. Older/transitional artifacts fall back to the
    /// materialized compiler-oracle table; ordinary constraints use that table
    /// directly as before.
    pub(crate) fn composition_start_nullable(&self) -> Result<bool, String> {
        if let Some(parser) = &self.template_parser {
            return parser.embedding.as_ref().map(|embedding| embedding.nullable)
                .ok_or_else(|| "template parser has no embedding nullability".to_owned());
        }
        if self.uses_compact_segmented_parser_runtime() {
            if let Some(summary) = self.composition_grammar_summary.as_ref() {
                return Ok(summary.root_nullable);
            }
        }
        Ok(self.table.embedded_start_nullable())
    }

    /// Reconstruct the compiler-only flattened tokenizer of a recursive
    /// composition directly from its intact leaf tokenizers.
    ///
    /// Live recursive execution never needs the old outer union tokenizer, but
    /// late composition still has compiler analyses which consume one flat raw
    /// tokenizer-state coordinate. The recursive layout deliberately uses the
    /// same owned-parent state ordering: root leaf states begin at zero and all
    /// remaining leaves are appended contiguously. Rebuilding in that order
    /// therefore preserves `recursive_tokenizer_internal_tsids` exactly; no
    /// state relation is transported or approximated here.
    fn rebuild_recursive_compiler_tokenizer(&self) -> Result<Tokenizer, String> {
        let layout = self
            .recursive_parser_layout_ref()
            .ok_or_else(|| "recursive composition layout is unavailable".to_owned())?;
        let root_leaf = layout
            .leaves
            .first()
            .ok_or_else(|| "recursive composition has no tokenizer leaves".to_owned())?;
        let root = self
            .constraint_at_recursive_component_path(&root_leaf.component_path)
            .ok_or_else(|| "recursive tokenizer root leaf does not resolve".to_owned())?;

        fn terminal_base_for_path(
            root: &Constraint,
            path: &[u32],
        ) -> Result<u32, String> {
            let mut constraint = root;
            let mut base = 0u32;
            for &component_index in path {
                let overlay = constraint.static_dynamic_overlay.as_ref().ok_or_else(|| {
                    format!(
                        "recursive tokenizer path {path:?} enters a non-composed constraint"
                    )
                })?;
                let component = overlay
                    .segmented_parser_components
                    .get(component_index as usize)
                    .ok_or_else(|| {
                        format!(
                            "recursive tokenizer path {path:?} references missing component {component_index}"
                        )
                    })?;
                base = base
                    .checked_add(component.terminal_offset)
                    .ok_or_else(|| "recursive tokenizer terminal offset overflow".to_owned())?;
                constraint = component.constraint.as_ref();
            }
            Ok(base)
        }

        let root_terminal_base = terminal_base_for_path(self, &root_leaf.component_path)?;
        if root_terminal_base != 0 {
            return Err(format!(
                "recursive tokenizer root leaf terminal base is {root_terminal_base}, expected zero"
            ));
        }

        let mut child_inputs = Vec::<(&Tokenizer, u32)>::with_capacity(layout.leaves.len().saturating_sub(1));
        for leaf in layout.leaves.iter().skip(1) {
            let constraint = self
                .constraint_at_recursive_component_path(&leaf.component_path)
                .ok_or_else(|| {
                    format!(
                        "recursive tokenizer leaf path {:?} does not resolve",
                        leaf.component_path,
                    )
                })?;
            child_inputs.push((
                &constraint.tokenizer,
                terminal_base_for_path(self, &leaf.component_path)?,
            ));
        }

        let (mut tokenizer, state_offsets) = Tokenizer::disjoint_union_with_owned_parent(
            root.tokenizer.as_ref().clone(),
            root_terminal_base,
            &child_inputs,
        );
        if state_offsets != layout.leaf_tokenizer_state_offsets {
            return Err(format!(
                "rebuilt recursive compiler tokenizer state offsets {state_offsets:?} disagree with live layout {:?}",
                layout.leaf_tokenizer_state_offsets,
            ));
        }

        // Preserve the exact canonical terminal chosen at every wrapper level.
        // `global_terminal_aliases` is directed metadata: `(canonical, local)`
        // means the component's direct `terminal_offset + local` terminal was
        // folded into `canonical`. Nested wrappers therefore form alias chains.
        // Collapse those chains once in the root terminal coordinate before
        // mutating the rebuilt tokenizer.
        fn collect_alias_edges(
            constraint: &Constraint,
            base: u32,
            edges: &mut Vec<(u32, u32)>,
        ) -> Result<(), String> {
            let Some(overlay) = constraint.static_dynamic_overlay.as_ref() else {
                return Ok(());
            };
            for component in &overlay.segmented_parser_components {
                let component_base = base
                    .checked_add(component.terminal_offset)
                    .ok_or_else(|| "recursive tokenizer terminal alias offset overflow".to_owned())?;
                for &(canonical, local) in &component.global_terminal_aliases {
                    let canonical = base.checked_add(canonical).ok_or_else(|| {
                        "recursive tokenizer canonical terminal overflow".to_owned()
                    })?;
                    let alias = component_base.checked_add(local).ok_or_else(|| {
                        "recursive tokenizer alias terminal overflow".to_owned()
                    })?;
                    if canonical != alias {
                        edges.push((canonical, alias));
                    }
                }
                collect_alias_edges(component.constraint.as_ref(), component_base, edges)?;
            }
            Ok(())
        }

        let mut edges = Vec::<(u32, u32)>::new();
        collect_alias_edges(self, 0, &mut edges)?;
        let mut parent = BTreeMap::<u32, u32>::new();
        for (canonical, alias) in edges {
            if canonical >= tokenizer.num_terminals() || alias >= tokenizer.num_terminals() {
                return Err(format!(
                    "recursive tokenizer terminal alias {alias}->{canonical} lies outside rebuilt domain {}",
                    tokenizer.num_terminals(),
                ));
            }
            if let Some(previous) = parent.insert(alias, canonical)
                && previous != canonical
            {
                return Err(format!(
                    "recursive tokenizer terminal alias {alias} has conflicting canonicals {previous} and {canonical}"
                ));
            }
        }
        let resolve = |start: u32, parent: &BTreeMap<u32, u32>| -> Result<u32, String> {
            let mut current = start;
            let mut steps = 0usize;
            while let Some(&next) = parent.get(&current) {
                current = next;
                steps += 1;
                if steps > parent.len() {
                    return Err("recursive tokenizer terminal alias cycle".to_owned());
                }
            }
            Ok(current)
        };
        let mut aliases_by_root = BTreeMap::<u32, Vec<u32>>::new();
        for &alias in parent.keys() {
            let root = resolve(alias, &parent)?;
            if alias != root {
                aliases_by_root.entry(root).or_default().push(alias);
            }
        }
        for (canonical, aliases) in aliases_by_root {
            tokenizer.canonicalize_terminal_aliases(canonical, &aliases);
        }
        Ok(tokenizer)
    }

    /// Ensure a compiler-owned recursive component has the flat tokenizer
    /// coordinate required by current late-composition analyses. Current v25
    /// artifacts still carry that tokenizer eagerly, so this is normally a
    /// no-op. It becomes the exact lazy reconstruction path once the outer
    /// compiler tokenizer is omitted from a future wire artifact.
    pub(crate) fn prepare_recursive_compiler_tokenizer_for_composition(
        &mut self,
    ) -> Result<bool, String> {
        if !self.uses_compact_segmented_parser_runtime() {
            return Ok(false);
        }
        let layout = self
            .recursive_parser_layout_for_pending_root()?
            .ok_or_else(|| "recursive composition layout is unavailable".to_owned())?;
        let expected_states = layout.total_tokenizer_states as usize;
        if self.tokenizer.num_states() as usize == expected_states
            && self.state_to_internal_tsid.len() == expected_states
        {
            return Ok(false);
        }
        let overlay = self
            .static_dynamic_overlay
            .as_ref()
            .ok_or_else(|| "recursive compiler tokenizer reconstruction requires overlay".to_owned())?;
        let provider_native = overlay
            .recursive_compiler_table
            .get()
            .is_some_and(|blob| blob.is_empty());
        let relation = overlay.recursive_tokenizer_internal_tsids.get().cloned();
        if !provider_native {
            let relation = relation.as_ref().ok_or_else(|| {
                "recursive compiler tokenizer reconstruction has no persisted state/TSID relation"
                    .to_owned()
            })?;
            if relation.len() != expected_states {
                return Err(format!(
                    "recursive compiler tokenizer TSID relation has {} rows for {expected_states} states",
                    relation.len(),
                ));
            }
        }
        let tokenizer = self.rebuild_recursive_compiler_tokenizer()?;
        if tokenizer.num_states() as usize != expected_states {
            return Err(format!(
                "rebuilt recursive compiler tokenizer has {} states, expected {expected_states}",
                tokenizer.num_states(),
            ));
        }

        let (state_to_internal_tsid, internal_tsid_to_states, state_internal_tsid_offsets, state_internal_tsids) =
            if provider_native {
                // The fast DynamicDirect coordinator intentionally stores no
                // outer quotient because live recursive execution never uses
                // it. A later static/compiler analysis can safely over-refine
                // to the exact raw-state identity coordinate on demand.
                (
                    (0..expected_states as u32).collect::<Vec<_>>(),
                    Vec::new(),
                    vec![u32::MAX],
                    Vec::new(),
                )
            } else {
                let relation = relation
                    .as_ref()
                    .expect("non-provider-native recursive compiler tokenizer requires relation");
                let tsid_count = self.internal_tsid_count();
                let mut state_to_internal_tsid = Vec::with_capacity(expected_states);
                let mut internal_tsid_to_states = vec![Vec::<u32>::new(); tsid_count];
                let mut state_internal_tsid_offsets = Vec::with_capacity(expected_states + 1);
                let mut state_internal_tsids = Vec::<u32>::new();
                state_internal_tsid_offsets.push(0);
                for (state, row) in relation.iter().enumerate() {
                    let Some(&primary) = row.first() else {
                        return Err(format!(
                            "recursive compiler tokenizer state {state} has no internal TSID"
                        ));
                    };
                    state_to_internal_tsid.push(primary);
                    for &tsid in row {
                        if tsid as usize >= tsid_count {
                            return Err(format!(
                                "recursive compiler tokenizer state {state} references TSID {tsid}/{tsid_count}"
                            ));
                        }
                        internal_tsid_to_states[tsid as usize].push(state as u32);
                        state_internal_tsids.push(tsid);
                    }
                    state_internal_tsid_offsets.push(state_internal_tsids.len() as u32);
                }
                (
                    state_to_internal_tsid,
                    internal_tsid_to_states,
                    state_internal_tsid_offsets,
                    state_internal_tsids,
                )
            };

        self.tokenizer = tokenizer.into();
        self.state_to_internal_tsid = state_to_internal_tsid;
        self.internal_tsid_to_states = internal_tsid_to_states;
        self.deferred_internal_tsid_to_states = OnceLock::new();
        self.state_internal_tsid_offsets = state_internal_tsid_offsets;
        self.state_internal_tsids = state_internal_tsids;
        // The rebuilt tokenizer is a direct union of exact leaf states, not a
        // runtime subset/product expansion of the old outer tokenizer.
        self.runtime_source_state_offset = None;
        self.runtime_product_source_offsets.clear();
        self.runtime_product_source_states.clear();
        self.runtime_product_exact_source_states.clear();
        self.runtime_product_state_by_source_subset.clear();
        self.tokenizer_has_epsilon_transitions = self.tokenizer.has_epsilon_transitions();
        self.terminal_live_states = self.compute_terminal_live_states();
        self.tokenizer_fast_transitions = Self::compute_tokenizer_fast_transitions_for(&self.tokenizer);
        Ok(true)
    }


    /// Reconstruct the exact compiler-only flattened table from the retained
    /// recursive component tree. Provider-native DynamicDirect coordinators
    /// intentionally omit their historical flattened compiler blob at build
    /// time; a later static/legacy composition is allowed to pay this cost on
    /// demand without changing the live recursive runtime representation.
    fn rebuild_recursive_compiler_table_from_components(&self) -> Result<GLRTable, String> {
        let overlay = self
            .static_dynamic_overlay
            .as_ref()
            .ok_or_else(|| "recursive compiler-table reconstruction requires overlay".to_owned())?;
        if overlay.segmented_parser_components.is_empty() {
            return Err("recursive compiler-table reconstruction has no components".to_owned());
        }

        let mut prepared = Vec::<Constraint>::with_capacity(overlay.segmented_parser_components.len());
        for component in &overlay.segmented_parser_components {
            let mut constraint = component.constraint.as_ref().clone();
            constraint.prepare_recursive_compiler_table_for_composition()?;
            prepared.push(constraint);
        }

        let mut slots_by_child = vec![Vec::<u32>::new(); prepared.len()];
        for (link_index, link) in overlay.segmented_parser_links.iter().enumerate() {
            if link.parent_component != 0 {
                return Err(format!(
                    "recursive compiler-table reconstruction link {link_index} has non-root parent component {}",
                    link.parent_component,
                ));
            }
            let child = link.child_component as usize;
            if child == 0 || child >= prepared.len() {
                return Err(format!(
                    "recursive compiler-table reconstruction link {link_index} references invalid child {child}",
                ));
            }
            slots_by_child[child].push(link.slot_terminal);
        }
        for slots in &mut slots_by_child {
            slots.sort_unstable();
            slots.dedup();
        }

        let mut child_rules = Vec::with_capacity(prepared.len().saturating_sub(1));
        for child in prepared.iter().skip(1) {
            child_rules.push(child.retained_table_rules()?);
        }
        let parent_rules = prepared[0].retained_table_rules()?;
        let mut table_inputs = Vec::with_capacity(prepared.len().saturating_sub(1));
        for component_index in 1..prepared.len() {
            let slots = &slots_by_child[component_index];
            let Some((&placeholder_terminal, additional_placeholder_terminals)) = slots.split_first()
            else {
                return Err(format!(
                    "recursive compiler-table reconstruction component {component_index} has no linker slot",
                ));
            };
            let child = &prepared[component_index];
            table_inputs.push(crate::compiler::glr::table::SubgrammarTableInput {
                placeholder_terminal,
                additional_placeholder_terminals,
                table: &child.table,
                ignore_terminal: child.ignore_terminal,
                start_nullable: child.composition_start_nullable()?,
            });
        }

        let composed = crate::compiler::glr::table::compose_subgrammar_tables_explicit_with_rules(
            &prepared[0].table,
            parent_rules,
            prepared[0].ignore_terminal,
            &table_inputs,
            &child_rules,
        )?;
        let mut table = composed.table;
        if table.num_terminals != self.parser_terminal_count() {
            return Err(format!(
                "rebuilt recursive compiler table has {} terminals, grammar shell has {}",
                table.num_terminals, self.parser_terminal_count(),
            ));
        }
        table.set_embedded_end_token_ids(&self.table.embedded_end_token_ids());
        Ok(table)
    }

    /// Materialize the exact flattened parser table used only by later
    /// composition. Recursive runtime execution never consults this table: its
    /// parser coordinate is the disjoint union of intact leaf tables plus
    /// CALL/RETURN provider controls. We nevertheless retain the exact compiled
    /// table in packed form because grammar rules alone do not encode all LR
    /// conflict/precedence decisions.
    pub(crate) fn prepare_recursive_compiler_table_for_composition(
        &mut self,
    ) -> Result<bool, String> {
        if !self.uses_compact_segmented_parser_runtime() {
            return Ok(false);
        }
        if self.parser_symbol_count() != 0 && !self.table.action.is_empty() {
            return Ok(false);
        }
        let blob = self
            .static_dynamic_overlay
            .as_ref()
            .and_then(|overlay| overlay.recursive_compiler_table.get())
            .cloned()
            .ok_or_else(|| {
                "recursive composition has no packed compiler table".to_owned()
            })?;
        let provider_native = blob.is_empty();
        let table = if provider_native {
            self.rebuild_recursive_compiler_table_from_components()?
        } else {
            crate::compiler::glr::table::artifact_serde::from_compact_bytes(blob.as_ref())?
        };
        if table.num_terminals != self.parser_terminal_count() {
            return Err(format!(
                "recursive compiler table has {} terminals, grammar shell has {}",
                table.num_terminals, self.parser_terminal_count(),
            ));
        }
        if !provider_native && table.num_rules != self.table.num_rules {
            return Err(format!(
                "recursive compiler table has {} rules, grammar shell has {}",
                table.num_rules, self.table.num_rules,
            ));
        }
        self.table = table.into();
        self.deferred_table_rules_blob = None;
        self.deferred_table_rules = OnceLock::new();
        Ok(true)
    }

    /// Replace the live recursive coordinator's flattened LR machine with a
    /// grammar shell while retaining an exact packed compiler copy for future
    /// rebinding. The shell deliberately keeps terminal/rule/nonterminal
    /// metadata because those are semantic grammar facts; only executable
    /// materialized parser-state machinery is discarded.
    pub(crate) fn detach_recursive_outer_table(&mut self) -> Result<bool, String> {
        if !self.uses_compact_segmented_parser_runtime() {
            return Ok(false);
        }
        let overlay = self
            .static_dynamic_overlay
            .as_ref()
            .ok_or_else(|| "recursive runtime is missing overlay metadata".to_owned())?;
        if overlay.recursive_compiler_table.get().is_none() {
            if self.parser_symbol_count() == 0 || self.table.action.is_empty() {
                return Err(
                    "recursive grammar shell has no packed compiler table".to_owned(),
                );
            }
            let packed = Arc::<[u8]>::from(
                crate::compiler::glr::table::artifact_serde::to_compact_bytes(&self.table),
            );
            overlay
                .recursive_compiler_table
                .set(packed)
                .map_err(|_| "recursive compiler table initialized twice".to_owned())?;
        }
        if self.parser_symbol_count() == 0 && self.table.action.is_empty() && self.table.goto.is_empty() {
            return Ok(false);
        }
        self.table.action.clear();
        self.table.goto.clear();
        self.table.advance.clear();
        self.table.unconditional_advance.clear();
        self.table.forwarded_shifts.clear();
        self.table.control_terminals.clear();
        self.table.skip_terminals.clear();
        self.table.guarded_shift_index.clear();
        self.table.direct_regular_wide_frontiers.clear();
        self.table.num_states = 0;
        Ok(true)
    }

    /// Drop the redundant flattened union tokenizer from a live recursive
    /// coordinator after compilation has finished. Recursive execution scans
    /// the intact leaf tokenizers directly; the outer `Constraint::tokenizer`
    /// field is therefore only a structural placeholder for this runtime kind.
    /// Keep the root leaf tokenizer there so generic/debug code still sees a
    /// valid tokenizer object. If this constraint is composed again later,
    /// `prepare_recursive_compiler_tokenizer_for_composition` reconstructs the
    /// exact temporary flat compiler view from the leaf tree and the persisted
    /// recursive state/TSID relation.
    pub(crate) fn detach_recursive_outer_tokenizer(&mut self) -> Result<bool, String> {
        if !self.uses_compact_segmented_parser_runtime() {
            return Ok(false);
        }
        let layout = self
            .recursive_parser_layout_for_pending_root()?
            .ok_or_else(|| "recursive composition layout is unavailable".to_owned())?;
        let root_leaf = layout
            .leaves
            .first()
            .ok_or_else(|| "recursive composition has no tokenizer leaves".to_owned())?;
        let root = self
            .constraint_at_recursive_component_path(&root_leaf.component_path)
            .ok_or_else(|| "recursive tokenizer root leaf does not resolve".to_owned())?;
        if self.tokenizer.num_states() == root.tokenizer.num_states()
            && self.tokenizer.num_terminals() == root.tokenizer.num_terminals()
        {
            return Ok(false);
        }
        let tokenizer = root.tokenizer.clone();
        let tokenizer_fast_transitions = root.tokenizer_fast_transitions.clone();
        let tokenizer_has_epsilon_transitions = root.tokenizer_has_epsilon_transitions;
        self.tokenizer = tokenizer.into();
        self.tokenizer_fast_transitions = tokenizer_fast_transitions;
        self.tokenizer_has_epsilon_transitions = tokenizer_has_epsilon_transitions;
        self.terminal_live_states.clear();
        Ok(true)
    }

    /// Exact zero-width pop depth for returning from this constraint when it is
    /// used as an opaque subgrammar by a later composition.
    ///
    /// Descendant bindings do not change the augmented-root goto of the
    /// outermost intact grammar frame. Recursive runtimes therefore derive the
    /// invariant from the first/root leaf instead of the materialized composed
    /// table; ordinary constraints keep the historical direct derivation.
    pub(crate) fn composition_child_return_pop(&self) -> Result<u32, String> {
        if let Some(parser) = &self.template_parser {
            return parser.embedding.as_ref().map(|embedding| embedding.return_pop)
                .ok_or_else(|| "native parser lacks a finite return convention".to_owned());
        }
        if !self.uses_compact_segmented_parser_runtime() {
            return subgrammar_child_return_pop(&self.table, self.retained_table_rules()?);
        }
        let layout = self
            .recursive_parser_layout_ref()
            .ok_or_else(|| "recursive composition layout is unavailable".to_owned())?;
        let root_leaf = layout
            .leaves
            .first()
            .ok_or_else(|| "recursive composition has no root leaf".to_owned())?;
        let root = self
            .constraint_at_recursive_component_path(&root_leaf.component_path)
            .ok_or_else(|| "recursive composition root leaf does not resolve".to_owned())?;
        subgrammar_child_return_pop(&root.table, root.retained_table_rules()?)
    }

    pub(crate) fn embedded_end_token_ids(&self) -> Vec<u32> {
        self.template_parser.as_ref().and_then(|parser| parser.link_grammar.as_ref())
            .map(|grammar| grammar.embedded_end_token_ids.clone())
            .expect("native parser is missing composition grammar metadata")
    }

    pub(crate) fn set_composition_end_token_ids(&mut self, ids: &[u32]) {
        if let Some(parser) = self.template_parser.as_mut().and_then(Arc::get_mut)
            && let Some(grammar) = parser.link_grammar.as_mut() {
            Arc::make_mut(grammar).embedded_end_token_ids = ids.to_vec();
        }
    }

    /// Preserve source nullability after compiler-only normalization. This
    /// updates interface metadata and the finite nullable return relation;
    /// it never reconstructs or materializes an LR table.
    pub(crate) fn set_composition_start_nullable(&mut self, nullable: bool) {
        let parser = self.template_parser.as_mut()
            .and_then(Arc::get_mut).expect("source nullability must be set before sharing the native parser");
        if let Some(grammar) = parser.link_grammar.as_mut() {
            Arc::make_mut(grammar).root_nullable = nullable;
        }
        if let Some(embedding) = parser.embedding.as_mut() {
            let embedding = Arc::make_mut(embedding);
            if nullable && !embedding.nullable {
                embedding.finish = Arc::new(crate::runtime::parser_backend::link_program::compile(&[
                    crate::runtime::parser_backend::link_program::action_nfa(&embedding.finish)
                        .expect("validated finite finish relation"),
                    crate::runtime::parser_backend::link_program::nullable_return(0),
                ]).expect("nullable return must remain finite"));
                embedding.finish_view = crate::runtime::parser_backend::scoped_program::ScopedProgram::prepare(
                    Arc::clone(&embedding.finish), embedding.finish_view.symbols)
                    .expect("nullable FINISH must remain within its component");
            }
            embedding.nullable = nullable;
        }
    }

    /// Finite tokenizer coordinate used by Static composition/link analysis.
    ///
    /// Static constraints with an exact virtual residual lexer commit in the
    /// exact symbolic coordinate, but their parser DWA / TSID tables were
    /// compiled against the finite vocabulary-horizon observation tokenizer.
    /// Composition is likewise a whole-model-token analysis, so it must use
    /// that same finite coordinate. The retained component constraint remains
    /// exact and is used unchanged by the segmented runtime.
    pub(crate) fn composition_tokenizer(&self) -> &Tokenizer {
        if !self.uses_dynamic_runtime() && self.tokenizer.has_virtual_residual_runtime() {
            self.dynamic_mask_vocab
                .mask_projection_tokenizer()
                .expect("Static virtual-residual constraint requires its finite observation tokenizer")
        } else {
            &self.tokenizer
        }
    }

    pub(crate) fn token_bytes_match_vocab(&self, vocab: &crate::Vocab) -> bool {
        let vocab_entries = vocab.entries_arc();
        if Arc::ptr_eq(&self.token_bytes, &vocab_entries) {
            return true;
        }
        if let Some(packed) = &self.packed_token_bytes {
            // Current artifacts and Vocab's prepared packed-token artifact use
            // the same canonical TBP2 encoding. Comparing those byte strings is
            // an exact vocabulary equality proof and lets a 128k-token bind use
            // one contiguous memcmp instead of 128k map/offset lookups. The
            // packed Vocab artifact is owned by the caller's Vocab-derived
            // cache, not a process-global load cache, so this remains honest
            // model-vocabulary preparation and does not make later constraints
            // cheaper merely because an earlier constraint was loaded.
            if let Some(vocab_packed) =
                crate::compiler::compile::prepared_vocab_packed_token_bytes(vocab)
                && packed.wire() == vocab_packed.wire()
            {
                return true;
            }
            // Legacy/noncanonical packed encodings can still describe the same
            // vocabulary. Preserve the exact structural fallback rather than
            // treating unequal wire encodings as unequal vocabularies.
            return packed.len() == vocab.entries_map().len()
                && packed.iter().eq(
                    vocab
                        .entries_map()
                        .iter()
                        .map(|(&token_id, bytes)| (token_id, bytes.as_slice())),
                );
        }
        self.token_bytes.as_ref() == vocab.entries_map()
    }

    #[inline]
    pub(crate) fn token_bytes_for_id(&self, token_id: u32) -> Option<&[u8]> {
        self.packed_token_bytes
            .as_ref()
            .and_then(|packed| packed.get(token_id))
            .or_else(|| self.token_bytes.get(&token_id).map(Vec::as_slice))
    }

    #[inline]
    pub(crate) fn token_bytes_count(&self) -> usize {
        self.packed_token_bytes
            .as_ref()
            .map_or_else(|| self.token_bytes.len(), |packed| packed.len())
    }

    #[inline]
    pub(crate) fn max_token_byte_len(&self) -> usize {
        self.packed_token_bytes.as_ref().map_or_else(
            || self.token_bytes.values().map(Vec::len).max().unwrap_or(0),
            |packed| packed.iter().map(|(_, bytes)| bytes.len()).max().unwrap_or(0),
        )
    }

    pub(crate) fn token_bytes_iter(&self) -> Box<dyn Iterator<Item = (u32, &[u8])> + '_> {
        if let Some(packed) = &self.packed_token_bytes {
            Box::new(packed.iter())
        } else {
            Box::new(
                self.token_bytes
                    .iter()
                    .map(|(&token_id, bytes)| (token_id, bytes.as_slice())),
            )
        }
    }

    /// Bind a loaded constraint to an exact model vocabulary once.
    ///
    /// The deep byte-map equality check is paid at load/bind time; successful
    /// binding then lets repeated composition prove compatibility by `Arc` identity.
    #[doc(hidden)]
    pub(crate) fn bind_vocab_exact(&mut self, vocab: &crate::Vocab) -> Result<(), String> {
        if let Some(existing) = self.late_bind_vocab.get()
            && !existing.same_model_vocab(vocab)
        {
            return Err(
                "constraint was not compiled for the supplied exact vocabulary".to_string(),
            );
        }
        let entries = vocab.entries_arc();
        if Arc::ptr_eq(&self.token_bytes, &entries) {
            self.late_bind_vocab = OnceLock::from(vocab.clone());
        } else {
            if !self.token_bytes_match_vocab(vocab) {
                return Err("constraint was not compiled for the supplied vocabulary".to_string());
            }
            self.token_bytes = entries;
            // A successful exact bind establishes the precise public vocabulary
            // for every later late-subgrammar bind as well. Keep the caller's
            // already-built `Vocab` (and its pure derived-artifact cache) instead
            // of reconstructing the same bytes again in `constraint_vocab()`.
            self.late_bind_vocab = OnceLock::from(vocab.clone());
        }
        if let Some(overlay) = self.static_dynamic_overlay.as_mut() {
            for component in &mut overlay.segmented_parser_components {
                Arc::make_mut(&mut component.constraint).bind_vocab_exact(vocab)?;
            }
        }
        Ok(())
    }

    #[inline]
    pub(crate) fn parser_state_domain_label(&self, parser_state: u32) -> Option<i32> {
        self.parser_state_domain_labels
            .get(parser_state as usize)
            .copied()
            .filter(|&label| label != i32::MAX)
    }

    #[inline]
    pub(crate) fn fast_parser_dwa_transition<'a>(
        &self,
        row: &'a FastDwaTransitionRow,
        parser_state: u32,
    ) -> Option<(u32, &'a Weight)> {
        let positive = encode_positive_label(parser_state);
        row.get(&positive)
            .or_else(|| self.parser_state_domain_label(parser_state).and_then(|label| row.get(&label)))
            .or_else(|| row.get(&DEFAULT_LABEL))
    }

    #[inline]
    pub(crate) fn runtime_parser_dwa_state_count(&self) -> usize {
        self.packed_parser_dwa
            .as_ref()
            .map_or_else(|| self.parser_dwa.states().len(), |dwa| dwa.state_count())
    }

    #[inline]
    pub(crate) fn runtime_parser_dwa_start_state(&self) -> u32 {
        self.packed_parser_dwa
            .as_ref()
            .map_or_else(|| self.parser_dwa.start_state(), |dwa| dwa.start_state())
    }

    #[inline]
    pub(crate) fn runtime_parser_dwa_final_weight(
        &self,
        dwa_state: u32,
    ) -> Option<RuntimeWeightRef<'_>> {
        if dwa_state == self.runtime_parser_dwa_start_state()
            && let Some(weight) = self.parser_start_final_override.as_ref()
        {
            return (!weight.is_empty()).then_some(RuntimeWeightRef::Materialized(weight));
        }
        if let Some(dwa) = &self.packed_parser_dwa {
            return dwa
                .final_weight(dwa_state)
                .map(RuntimeWeightRef::PackedDwa);
        }
        self.parser_dwa
            .states()
            .get(dwa_state as usize)?
            .final_weight
            .as_ref()
            .map(RuntimeWeightRef::Materialized)
    }

    #[inline]
    pub(crate) fn runtime_parser_dwa_transition(
        &self,
        dwa_state: u32,
        parser_state: u32,
    ) -> Option<(u32, RuntimeWeightRef<'_>)> {
        let positive = encode_positive_label(parser_state);
        if let Some(dwa) = &self.packed_parser_dwa {
            return dwa
                .transition(dwa_state, positive)
                .or_else(|| {
                    self.parser_state_domain_label(parser_state)
                        .and_then(|label| dwa.transition(dwa_state, label))
                })
                .or_else(|| dwa.transition(dwa_state, DEFAULT_LABEL))
                .map(|(target, weight)| (target, RuntimeWeightRef::PackedDwa(weight)));
        }
        let row = self.dwa_fast_transitions.get(dwa_state as usize)?;
        self.fast_parser_dwa_transition(row, parser_state)
            .map(|(target, weight)| (target, RuntimeWeightRef::Materialized(weight)))
    }

    #[inline]
    pub(crate) fn runtime_parser_dwa_row_is_empty(&self, dwa_state: u32) -> bool {
        if let Some(dwa) = &self.packed_parser_dwa {
            dwa.row_is_empty(dwa_state)
        } else {
            self.dwa_fast_transitions
                .get(dwa_state as usize)
                .is_none_or(FastDwaTransitionRow::is_empty)
        }
    }

    #[inline]
    fn runtime_pooled_weight(&self, id: u32) -> Option<RuntimeWeightRef<'_>> {
        let packed = self.packed_non_dwa_weights.as_ref()?;
        if let Some(index) = packed.final_masks.indices.get(&id) {
            return Some(RuntimeWeightRef::IndexedPool(index));
        }
        packed.pool.weight(id).map(RuntimeWeightRef::PackedPool)
    }

    #[inline]
    pub(crate) fn runtime_parser_top_accept(
        &self,
        label: i32,
    ) -> Option<RuntimeWeightRef<'_>> {
        if let Some(packed) = &self.packed_non_dwa_weights {
            let id = packed
                .parser_top_accept
                .get(&label)
                .or_else(|| packed.parser_top_accept.get(&DEFAULT_LABEL))?;
            return self.runtime_pooled_weight(*id);
        }
        self.parser_top_accept
            .get(&label)
            .or_else(|| self.parser_top_accept.get(&DEFAULT_LABEL))
            .map(RuntimeWeightRef::Materialized)
    }

    pub(crate) fn runtime_parser_top_accept_parts(
        &self,
        label: i32,
    ) -> SmallVec<[RuntimeWeightRef<'_>; 4]> {
        if let Some(packed) = &self.packed_non_dwa_weights {
            let Some(ids) = packed
                .parser_top_accept_parts
                .get(&label)
                .or_else(|| packed.parser_top_accept_parts.get(&DEFAULT_LABEL))
            else {
                return SmallVec::new();
            };
            return ids
                .iter()
                .filter_map(|&id| self.runtime_pooled_weight(id))
                .collect();
        }
        self.parser_top_accept_parts
            .get(&label)
            .or_else(|| self.parser_top_accept_parts.get(&DEFAULT_LABEL))
            .into_iter()
            .flatten()
            .map(RuntimeWeightRef::Materialized)
            .collect()
    }

    #[inline]
    pub(crate) fn runtime_direct_regular_l1_complete(
        &self,
        terminal: TerminalID,
    ) -> Option<RuntimeWeightRef<'_>> {
        if let Some(packed) = &self.packed_non_dwa_weights {
            return packed
                .direct_regular_l1_complete_by_terminal
                .get(&terminal)
                .and_then(|&id| self.runtime_pooled_weight(id));
        }
        self.direct_regular_l1_complete_by_terminal
            .get(&terminal)
            .map(RuntimeWeightRef::Materialized)
    }

    #[inline]
    pub(crate) fn runtime_possible_match_weight(
        &self,
        terminal: TerminalID,
    ) -> Option<RuntimeWeightRef<'_>> {
        if let Some(packed) = &self.packed_non_dwa_weights {
            return packed
                .possible_matches
                .get(&terminal)
                .and_then(|&id| self.runtime_pooled_weight(id));
        }
        self.possible_matches
            .get(&terminal)
            .map(RuntimeWeightRef::Materialized)
    }

    pub(crate) fn runtime_possible_match_terminals(
        &self,
    ) -> Box<dyn Iterator<Item = TerminalID> + '_> {
        if let Some(packed) = &self.packed_non_dwa_weights {
            Box::new(packed.possible_matches.keys().copied())
        } else {
            Box::new(self.possible_matches.keys().copied())
        }
    }

    #[inline]
    pub(crate) fn runtime_direct_regular_l1_is_empty(&self) -> bool {
        self.packed_non_dwa_weights.as_ref().map_or_else(
            || self.direct_regular_l1_complete_by_terminal.is_empty(),
            |packed| packed.direct_regular_l1_complete_by_terminal.is_empty(),
        )
    }

    pub(crate) fn runtime_direct_regular_l1_terminals(
        &self,
    ) -> Box<dyn Iterator<Item = TerminalID> + '_> {
        if let Some(packed) = &self.packed_non_dwa_weights {
            Box::new(
                packed
                    .direct_regular_l1_complete_by_terminal
                    .keys()
                    .copied(),
            )
        } else {
            Box::new(self.direct_regular_l1_complete_by_terminal.keys().copied())
        }
    }

    #[inline]
    pub(crate) fn indexed_parser_dwa_transition<'a>(
        &self,
        row: &'a IndexedDagDenseTransitionRow,
        parser_state: u32,
    ) -> Option<&'a IndexedDagDenseTransition> {
        let positive = encode_positive_label(parser_state);
        row.get(&positive)
            .or_else(|| self.parser_state_domain_label(parser_state).and_then(|label| row.get(&label)))
            .or_else(|| row.get(&DEFAULT_LABEL))
    }

    #[inline]
    pub(crate) fn uses_dynamic_runtime(&self) -> bool {
        matches!(
            self.runtime_backend,
            super::artifact::ConstraintRuntimeBackend::Dynamic
        )
    }

    #[inline]
    pub(crate) fn has_recursive_segmented_parser_tree(&self) -> bool {
        self.static_dynamic_overlay.as_ref().is_some_and(|overlay| {
            !overlay.segmented_parser_components.is_empty()
                && !overlay.segmented_parser_links.is_empty()
        })
    }

    /// Width of the endpoint parser-state coordinate when this constraint is
    /// embedded as one component. Nested compositions contribute the disjoint
    /// union of their intact descendants, not the size of their transitional
    /// materialized composed table.
    pub(crate) fn recursive_parser_state_span(&self) -> Result<u32, String> {
        if !self.uses_compact_segmented_parser_runtime() {
            return Ok(self.parser_symbol_count());
        }
        let overlay = self
            .static_dynamic_overlay
            .as_ref()
            .expect("recursive segmented parser tree requires overlay");
        let mut total = 0u32;
        for component in &overlay.segmented_parser_components {
            total = total
                .checked_add(component.constraint.recursive_parser_state_span()?)
                .ok_or_else(|| "recursive parser-state coordinate overflow".to_owned())?;
        }
        Ok(total)
    }

    pub(crate) fn constraint_at_recursive_component_path(&self, path: &[u32]) -> Option<&Constraint> {
        let mut current = self;
        for &component_index in path {
            let overlay = current.static_dynamic_overlay.as_ref()?;
            current = overlay
                .segmented_parser_components
                .get(component_index as usize)?
                .constraint
                .as_ref();
        }
        Some(current)
    }

    fn append_recursive_terminal_targets(
        &self,
        terminal: TerminalID,
        expand_this: bool,
        component_path: &mut Vec<u32>,
        leaves: &[RecursiveParserLeafLayout],
        out: &mut SmallVec<[(u32, TerminalID); 4]>,
    ) -> Result<(), String> {
        if !expand_this {
            if terminal >= self.parser_terminal_count() {
                return Ok(());
            }
            let leaf_index = leaves
                .iter()
                .position(|leaf| leaf.component_path == *component_path)
                .ok_or_else(|| "recursive parser terminal target has no leaf layout".to_owned())?;
            let target = (leaf_index as u32, terminal);
            if !out.contains(&target) {
                out.push(target);
            }
            return Ok(());
        }

        let overlay = self
            .static_dynamic_overlay
            .as_ref()
            .expect("recursive segmented parser tree requires overlay");
        for (component_index, component) in overlay.segmented_parser_components.iter().enumerate() {
            let offset = component.terminal_offset;
            let end = offset.saturating_add(component.constraint.parser_terminal_count());
            if terminal >= offset && terminal < end {
                component_path.push(component_index as u32);
                component.constraint.append_recursive_terminal_targets(
                    terminal - offset,
                    component.constraint.uses_compact_segmented_parser_runtime(),
                    component_path,
                    leaves,
                    out,
                )?;
                component_path.pop();
            }
            for &(alias, local_terminal) in &component.global_terminal_aliases {
                if alias == terminal {
                    component_path.push(component_index as u32);
                    component.constraint.append_recursive_terminal_targets(
                        local_terminal,
                        component.constraint.uses_compact_segmented_parser_runtime(),
                        component_path,
                        leaves,
                        out,
                    )?;
                    component_path.pop();
                }
            }
        }
        Ok(())
    }

    fn append_recursive_parser_layout(
        &self,
        base: u32,
        top_component: u32,
        expand_this: bool,
        component_path: &mut Vec<u32>,
        leaves: &mut Vec<RecursiveParserLeafLayout>,
        links: &mut Vec<super::artifact::SegmentedParserLink>,
    ) -> Result<(u32, u32), String> {
        if !expand_this {
            let leaf_index = u32::try_from(leaves.len())
                .map_err(|_| "recursive parser leaf index overflow".to_owned())?;
            leaves.push(RecursiveParserLeafLayout {
                state_offset: base,
                state_count: self.parser_symbol_count(),
                top_component,
                component_path: component_path.clone(),
            });
            let next = base
                .checked_add(self.parser_symbol_count())
                .ok_or_else(|| "recursive parser-state coordinate overflow".to_owned())?;
            return Ok((next, leaf_index));
        }
        let overlay = self
            .static_dynamic_overlay
            .as_ref()
            .expect("recursive segmented parser tree requires overlay");
        let mut next = base;
        let mut component_root_leaves = Vec::with_capacity(overlay.segmented_parser_components.len());
        for (component_index, component) in overlay.segmented_parser_components.iter().enumerate() {
            component_path.push(component_index as u32);
            let (component_next, component_root_leaf) = component.constraint.append_recursive_parser_layout(
                next,
                top_component,
                component.constraint.uses_compact_segmented_parser_runtime(),
                component_path,
                leaves,
                links,
            )?;
            component_path.pop();
            next = component_next;
            component_root_leaves.push(component_root_leaf);
        }

        self.append_recursive_node_links(
            component_path,
            leaves,
            &component_root_leaves,
            links,
        )?;

        let root_leaf = *component_root_leaves
            .first()
            .ok_or_else(|| "recursive parser composition has no root component".to_owned())?;
        Ok((next, root_leaf))
    }

    fn append_recursive_node_links(
        &self,
        component_path: &mut Vec<u32>,
        leaves: &[RecursiveParserLeafLayout],
        component_root_leaves: &[u32],
        links: &mut Vec<super::artifact::SegmentedParserLink>,
    ) -> Result<(), String> {
        let overlay = self
            .static_dynamic_overlay
            .as_ref()
            .expect("recursive segmented parser tree requires overlay");
        for (link_index, link) in overlay.segmented_parser_links.iter().enumerate() {
            let parent_component = overlay
                .segmented_parser_components
                .get(link.parent_component as usize)
                .ok_or_else(|| format!("recursive parser link {link_index} has missing parent component"))?;
            let child_root_leaf = *component_root_leaves
                .get(link.child_component as usize)
                .ok_or_else(|| format!("recursive parser link {link_index} has missing child component"))?;
            if link.child_start != 0 {
                return Err(format!(
                    "recursive parser link {link_index} has unsupported non-root child start {}",
                    link.child_start,
                ));
            }

            component_path.push(link.parent_component);
            let mut parent_targets = SmallVec::<[(u32, TerminalID); 4]>::new();
            parent_component.constraint.append_recursive_terminal_targets(
                link.slot_terminal,
                parent_component
                    .constraint
                    .uses_compact_segmented_parser_runtime(),
                component_path,
                leaves,
                &mut parent_targets,
            )?;
            component_path.pop();
            let [(parent_leaf, local_slot)] = parent_targets.as_slice() else {
                return Err(format!(
                    "recursive parser link {link_index} slot terminal {} resolves to {} leaf targets, expected exactly one",
                    link.slot_terminal,
                    parent_targets.len(),
                ));
            };
            links.push(super::artifact::SegmentedParserLink {
                parent_component: *parent_leaf,
                slot_terminal: *local_slot,
                child_component: child_root_leaf,
                child_start: 0,
                return_pop: link.return_pop,
                child_start_nullable: link.child_start_nullable,
            });
        }
        Ok(())
    }

    fn build_recursive_parser_layout_root_expanded(
        &self,
    ) -> Result<Arc<RecursiveParserLayout>, String> {
        let overlay = self
            .static_dynamic_overlay
            .as_ref()
            .expect("recursive segmented parser tree requires overlay");
        if let Some(layout) = overlay.recursive_parser_layout.get() {
            return Ok(Arc::clone(layout));
        }
        let mut component_offsets = Vec::with_capacity(overlay.segmented_parser_components.len());
        let mut leaves = Vec::new();
        let mut links = Vec::new();
        let mut component_path = Vec::new();
        let mut component_root_leaves = Vec::with_capacity(overlay.segmented_parser_components.len());
        let mut next = 0u32;
        for (component_index, component) in overlay.segmented_parser_components.iter().enumerate() {
            component_offsets.push(next);
            component_path.push(component_index as u32);
            let (component_next, component_root_leaf) = component.constraint.append_recursive_parser_layout(
                next,
                component_index as u32,
                component.constraint.uses_compact_segmented_parser_runtime(),
                &mut component_path,
                &mut leaves,
                &mut links,
            )?;
            component_path.pop();
            next = component_next;
            component_root_leaves.push(component_root_leaf);
        }
        self.append_recursive_node_links(
            &mut component_path,
            &leaves,
            &component_root_leaves,
            &mut links,
        )?;
        let leaf_state_offsets = leaves.iter().map(|leaf| leaf.state_offset).collect();
        let mut leaf_tokenizer_state_offsets = Vec::with_capacity(leaves.len());
        let mut leaf_terminal_offsets = Vec::with_capacity(leaves.len());
        let mut next_tokenizer_state = 0u32;
        let mut next_leaf_terminal = 0u32;
        let mut has_virtual_residuals = false;
        for leaf in &leaves {
            let constraint = self
                .constraint_at_recursive_component_path(&leaf.component_path)
                .ok_or_else(|| {
                    format!(
                        "recursive tokenizer leaf path {:?} does not resolve to a constraint",
                        leaf.component_path,
                    )
                })?;
            has_virtual_residuals |= constraint.tokenizer.has_virtual_residual_runtime();
            leaf_tokenizer_state_offsets.push(next_tokenizer_state);
            next_tokenizer_state = next_tokenizer_state
                .checked_add(constraint.tokenizer.num_states())
                .ok_or_else(|| "recursive tokenizer-state coordinate overflow".to_owned())?;
            leaf_terminal_offsets.push(next_leaf_terminal);
            next_leaf_terminal = next_leaf_terminal
                .checked_add(constraint.parser_terminal_count())
                .ok_or_else(|| "recursive terminal coordinate overflow".to_owned())?;
        }
        let mut terminal_targets = Vec::with_capacity(self.parser_terminal_count() as usize);
        for terminal in 0..self.parser_terminal_count() {
            let mut targets = SmallVec::<[(u32, TerminalID); 4]>::new();
            self.append_recursive_terminal_targets(
                terminal,
                true,
                &mut Vec::new(),
                &leaves,
                &mut targets,
            )?;
            terminal_targets.push(targets);
        }
        let layout = Arc::new(RecursiveParserLayout {
            component_offsets,
            leaves,
            has_virtual_residuals,
            leaf_state_offsets,
            leaf_tokenizer_state_offsets,
            total_tokenizer_states: next_tokenizer_state,
            leaf_terminal_offsets,
            total_leaf_terminals: next_leaf_terminal,
            outer_terminal_count: self.parser_terminal_count(),
            tokenizer_future_scoped: (0..next_tokenizer_state)
                .map(|_| OnceLock::new())
                .collect(),
            links,
            terminal_targets,
            total_states: next,
        });
        let _ = overlay.recursive_parser_layout.set(Arc::clone(&layout));
        Ok(overlay
            .recursive_parser_layout
            .get()
            .cloned()
            .unwrap_or(layout))
    }

    /// Derived recursive layout for the live endpoint parser coordinate.
    /// Immediate wrappers occupy contiguous intervals. Descendant leaves
    /// inside the same wrapper retain the same `top_component`.
    pub(crate) fn recursive_parser_layout(
        &self,
    ) -> Result<Option<Arc<RecursiveParserLayout>>, String> {
        if !self.uses_compact_segmented_parser_runtime() {
            return Ok(None);
        }
        self.build_recursive_parser_layout_root_expanded().map(Some)
    }

    fn recursive_parser_layout_ref(&self) -> Option<&RecursiveParserLayout> {
        if !self.uses_compact_segmented_parser_runtime() {
            return None;
        }
        let initialized = self
            .static_dynamic_overlay
            .as_ref()?
            .recursive_parser_layout
            .get()
            .is_some();
        if !initialized {
            self.build_recursive_parser_layout_root_expanded().ok()?;
        }
        self.static_dynamic_overlay
            .as_ref()?
            .recursive_parser_layout
            .get()
            .map(Arc::as_ref)
    }

    /// Compiler/load bridge used while a static boundary is still stored in
    /// the materialized composed-table coordinate. The root is expanded
    /// unconditionally; descendants are expanded only when their own runtime
    /// has already migrated to the recursive coordinate.
    pub(crate) fn recursive_parser_layout_for_pending_root(
        &self,
    ) -> Result<Option<Arc<RecursiveParserLayout>>, String> {
        if !self.has_recursive_segmented_parser_tree() {
            return Ok(None);
        }
        self.build_recursive_parser_layout_root_expanded().map(Some)
    }

    #[inline]
    pub(crate) fn recursive_tokenizer_leaf_state(
        &self,
        scoped_state: u32,
    ) -> Option<(usize, u32)> {
        let layout = self.recursive_parser_layout().ok().flatten()?;
        if scoped_state < layout.total_tokenizer_states {
            let leaf_index = layout
                .leaf_tokenizer_state_offsets
                .partition_point(|&offset| offset <= scoped_state)
                .checked_sub(1)?;
            let offset = *layout.leaf_tokenizer_state_offsets.get(leaf_index)?;
            let local_state = scoped_state.checked_sub(offset)?;
            let leaf = layout.leaves.get(leaf_index)?;
            let constraint = self.constraint_at_recursive_component_path(&leaf.component_path)?;
            return (local_state < constraint.tokenizer.num_states())
                .then_some((leaf_index, local_state));
        }
        self.static_dynamic_overlay
            .as_ref()?
            .recursive_virtual_tokenizer_states
            .local_state(scoped_state)
    }

    #[inline]
    pub(crate) fn recursive_tokenizer_scoped_state(
        &self,
        leaf_index: usize,
        local_state: u32,
    ) -> Option<u32> {
        let layout = self.recursive_parser_layout().ok().flatten()?;
        let leaf = layout.leaves.get(leaf_index)?;
        let constraint = self.constraint_at_recursive_component_path(&leaf.component_path)?;
        if local_state < constraint.tokenizer.num_states() {
            return layout
                .leaf_tokenizer_state_offsets
                .get(leaf_index)?
                .checked_add(local_state);
        }
        if !constraint.tokenizer.contains_runtime_state(local_state) {
            return None;
        }
        self.static_dynamic_overlay
            .as_ref()?
            .recursive_virtual_tokenizer_states
            .scoped_state(layout.total_tokenizer_states, leaf_index, local_state)
    }

    /// Project one tokenizer state from this composition's recursive leaf
    /// coordinate into an immediate component's own live tokenizer coordinate.
    /// Intact components receive a raw local state; recursively composed
    /// components receive the corresponding state in their own leaf union.
    pub(crate) fn recursive_tokenizer_state_for_component(
        &self,
        component_index: usize,
        scoped_state: u32,
    ) -> Option<u32> {
        let (leaf_index, local_state) = self.recursive_tokenizer_leaf_state(scoped_state)?;
        let layout = self.recursive_parser_layout_ref()?;
        let leaf = layout.leaves.get(leaf_index)?;
        let (&owner, descendant_path) = leaf.component_path.split_first()?;
        if owner as usize != component_index {
            return None;
        }

        let overlay = self.static_dynamic_overlay.as_ref()?;
        let component = overlay.segmented_parser_components.get(component_index)?;
        let component_constraint = component.constraint.as_ref();
        if !component_constraint.uses_compact_segmented_parser_runtime() {
            if !descendant_path.is_empty()
                || !component_constraint.tokenizer.contains_runtime_state(local_state)
            {
                return None;
            }
            return Some(local_state);
        }

        let component_layout = component_constraint.recursive_parser_layout_ref()?;
        let component_leaf_index = component_layout
            .leaves
            .iter()
            .position(|candidate| candidate.component_path.as_slice() == descendant_path)?;
        component_constraint.recursive_tokenizer_scoped_state(component_leaf_index, local_state)
    }

    /// Exact internal-TSID image of one live tokenizer state. Recursive
    /// compositions use their persisted/derived leaf-state relation; intact
    /// constraints use the ordinary tokenizer-state relation. Dynamic intact
    /// constraints without a TSID quotient use raw tokenizer state as TSID.
    pub(crate) fn runtime_internal_tsids_for_tokenizer_state(
        &self,
        tokenizer_state: u32,
    ) -> Option<SmallVec<[u32; 4]>> {
        if self.static_dynamic_overlay.as_ref().is_some_and(|overlay| overlay.recursive_static_observation.is_some()) {
            return self.static_exclusion_state(tokenizer_state).map(|key| SmallVec::from_slice(&[key]));
        }
        if self.uses_compact_segmented_parser_runtime() {
            let layout = self.recursive_parser_layout_ref()?;
            if tokenizer_state < layout.total_tokenizer_states {
                let overlay = self.static_dynamic_overlay.as_ref()?;
                let provider_native_compiler_view = overlay
                    .recursive_compiler_table
                    .get()
                    .is_some_and(|blob| blob.is_empty())
                    && self.state_to_internal_tsid.len() == layout.total_tokenizer_states as usize;
                if provider_native_compiler_view {
                    return Some(
                        self.internal_tsids_for_state(tokenizer_state)
                            .iter()
                            .copied()
                            .collect(),
                    );
                }
                return overlay
                    .recursive_tokenizer_internal_tsids
                    .get()?
                    .get(tokenizer_state as usize)
                    .map(|row| row.iter().copied().collect());
            }
            let (leaf_index, _) = self.recursive_tokenizer_leaf_state(tokenizer_state)?;
            let leaf = layout.leaves.get(leaf_index)?;
            let owner = *leaf.component_path.first()? as usize;
            let overlay = self.static_dynamic_overlay.as_ref()?;
            let component = overlay.segmented_parser_components.get(owner)?;
            let component_state =
                self.recursive_tokenizer_state_for_component(owner, tokenizer_state)?;
            let local_tsids = component
                .constraint
                .runtime_internal_tsids_for_tokenizer_state(component_state)?;
            let mut global = SmallVec::<[u32; 4]>::new();
            for local_tsid in local_tsids {
                global.extend(
                    component
                        .local_tsid_to_global_tsids
                        .get(local_tsid as usize)?
                        .iter()
                        .copied(),
                );
            }
            global.sort_unstable();
            global.dedup();
            return (!global.is_empty()).then_some(global);
        }
        if self.state_to_internal_tsid.is_empty() && self.internal_tsid_to_states.is_empty() {
            return (tokenizer_state < self.tokenizer.num_states())
                .then(|| smallvec::smallvec![tokenizer_state]);
        }
        if self.tokenizer.has_virtual_residual_runtime() {
            return self
                .tokenizer
                .contains_runtime_state(tokenizer_state)
                .then(|| self.internal_tsids_for_state(tokenizer_state).iter().copied().collect());
        }
        (tokenizer_state < self.tokenizer.num_states())
            .then(|| self.internal_tsids_for_state(tokenizer_state).iter().copied().collect())
    }

    pub(crate) fn install_recursive_tokenizer_internal_tsids(
        &mut self,
        mut relation: Vec<Vec<u32>>,
    ) -> Result<(), String> {
        let layout = self
            .recursive_parser_layout_for_pending_root()?
            .ok_or_else(|| "recursive tokenizer TSID relation requires recursive runtime".to_owned())?;
        if relation.len() != layout.total_tokenizer_states as usize {
            return Err(format!(
                "recursive tokenizer TSID relation has {} rows for {} scoped tokenizer states",
                relation.len(),
                layout.total_tokenizer_states,
            ));
        }
        let tsid_count = self.internal_tsid_count();
        for (state, row) in relation.iter_mut().enumerate() {
            row.sort_unstable();
            row.dedup();
            if row.is_empty() {
                return Err(format!(
                    "recursive tokenizer state {state} has no internal TSID image"
                ));
            }
            if let Some(&bad) = row.iter().find(|&&tsid| tsid as usize >= tsid_count) {
                return Err(format!(
                    "recursive tokenizer state {state} references out-of-range internal TSID {bad}/{tsid_count}"
                ));
            }
        }
        let overlay = self
            .static_dynamic_overlay
            .as_mut()
            .ok_or_else(|| "recursive tokenizer TSID relation requires overlay".to_owned())?;
        if let Some(existing) = overlay.recursive_tokenizer_internal_tsids.get() {
            if existing.as_ref() != &relation {
                return Err("recursive tokenizer TSID relation disagrees with existing view".to_owned());
            }
            return Ok(());
        }
        overlay
            .recursive_tokenizer_internal_tsids
            .set(Arc::new(relation))
            .map_err(|_| "recursive tokenizer TSID relation initialized twice".to_owned())
    }

    #[inline]
    pub(crate) fn recursive_tokenizer_reset_state(&self, leaf_index: usize) -> Option<u32> {
        let layout = self.recursive_parser_layout().ok().flatten()?;
        let leaf = layout.leaves.get(leaf_index)?;
        let constraint = self.constraint_at_recursive_component_path(&leaf.component_path)?;
        self.recursive_tokenizer_scoped_state(
            leaf_index,
            constraint.runtime_commit_initial_state(),
        )
    }

    pub(crate) fn recursive_tokenizer_is_reset_state(&self, scoped_state: u32) -> bool {
        let Some(layout) = self.recursive_parser_layout_ref() else {
            return false;
        };
        if scoped_state >= layout.total_tokenizer_states {
            return false;
        }
        let Some(leaf_index) = layout
            .leaf_tokenizer_state_offsets
            .partition_point(|&offset| offset <= scoped_state)
            .checked_sub(1)
        else {
            return false;
        };
        let offset = layout.leaf_tokenizer_state_offsets[leaf_index];
        let local_state = scoped_state - offset;
        let Some(leaf) = layout.leaves.get(leaf_index) else {
            return false;
        };
        let Some(constraint) = self.constraint_at_recursive_component_path(&leaf.component_path)
        else {
            return false;
        };
        local_state == constraint.runtime_commit_initial_state()
    }

    #[inline]
    pub(crate) fn recursive_runtime_terminal_count(&self) -> Option<usize> {
        let layout = self.recursive_parser_layout_ref()?;
        usize::try_from(
            layout
                .outer_terminal_count
                .checked_add(layout.total_leaf_terminals)?,
        )
        .ok()
    }

    #[inline]
    pub(crate) fn recursive_terminal_scoped_id(
        &self,
        leaf_index: usize,
        local_terminal: TerminalID,
    ) -> Option<u32> {
        let layout = self.recursive_parser_layout_ref()?;
        let leaf = layout.leaves.get(leaf_index)?;
        let leaf_constraint = self.constraint_at_recursive_component_path(&leaf.component_path)?;
        if local_terminal >= leaf_constraint.parser_terminal_count() {
            return None;
        }
        layout
            .outer_terminal_count
            .checked_add(*layout.leaf_terminal_offsets.get(leaf_index)?)?
            .checked_add(local_terminal)
    }

    #[inline]
    pub(crate) fn recursive_terminal_leaf_local(
        &self,
        runtime_terminal: u32,
    ) -> Option<(usize, TerminalID)> {
        let layout = self.recursive_parser_layout_ref()?;
        let scoped = runtime_terminal.checked_sub(layout.outer_terminal_count)?;
        if scoped >= layout.total_leaf_terminals {
            return None;
        }
        let leaf_index = layout
            .leaf_terminal_offsets
            .partition_point(|&offset| offset <= scoped)
            .checked_sub(1)?;
        let local_terminal = scoped.checked_sub(layout.leaf_terminal_offsets[leaf_index])?;
        let leaf = layout.leaves.get(leaf_index)?;
        let leaf_constraint = self.constraint_at_recursive_component_path(&leaf.component_path)?;
        (local_terminal < leaf_constraint.parser_terminal_count())
            .then_some((leaf_index, local_terminal))
    }

    pub(crate) fn recursive_terminal_for_component(
        &self,
        component_index: usize,
        runtime_terminal: u32,
    ) -> Option<u32> {
        let layout = self.recursive_parser_layout_ref()?;
        let (leaf_index, local_terminal) = self.recursive_terminal_leaf_local(runtime_terminal)?;
        let leaf = layout.leaves.get(leaf_index)?;
        let (&owner, descendant_path) = leaf.component_path.split_first()?;
        if owner as usize != component_index {
            return None;
        }
        let component = self
            .static_dynamic_overlay
            .as_ref()?
            .segmented_parser_components
            .get(component_index)?;
        let component_constraint = component.constraint.as_ref();
        if !component_constraint.uses_compact_segmented_parser_runtime() {
            if !descendant_path.is_empty()
                || local_terminal >= component_constraint.parser_terminal_count()
            {
                return None;
            }
            return Some(local_terminal);
        }
        let component_layout = component_constraint.recursive_parser_layout_ref()?;
        let component_leaf_index = component_layout
            .leaves
            .iter()
            .position(|candidate| candidate.component_path.as_slice() == descendant_path)?;
        component_constraint.recursive_terminal_scoped_id(component_leaf_index, local_terminal)
    }

    pub(crate) fn recursive_terminal_is_ignore(&self, runtime_terminal: u32) -> bool {
        let Some((leaf_index, local_terminal)) =
            self.recursive_terminal_leaf_local(runtime_terminal)
        else {
            return false;
        };
        self.recursive_leaf_constraint(leaf_index)
            .is_some_and(|leaf| leaf.ignore_terminal == Some(local_terminal))
    }

    pub(crate) fn recursive_tokenizer_future_scoped_terminals(
        &self,
        scoped_state: u32,
    ) -> Option<Cow<'_, BitSet>> {
        let layout = self.recursive_parser_layout_ref()?;
        let (leaf_index, local_state) = self.recursive_tokenizer_leaf_state(scoped_state)?;
        let leaf = layout.leaves.get(leaf_index)?;
        let leaf_constraint = self.constraint_at_recursive_component_path(&leaf.component_path)?;
        let build_scoped = || {
            let local_future = leaf_constraint
                .tokenizer
                .possible_future_terminals(local_state);
            let mut scoped = BitSet::new(
                self.recursive_runtime_terminal_count()
                    .expect("recursive terminal layout must fit usize"),
            );
            for local_terminal in local_future.iter_ones() {
                if let Some(runtime_terminal) =
                    self.recursive_terminal_scoped_id(leaf_index, local_terminal as u32)
                {
                    scoped.set(runtime_terminal as usize);
                }
            }
            scoped
        };
        if scoped_state < layout.total_tokenizer_states {
            let cache = layout.tokenizer_future_scoped.get(scoped_state as usize)?;
            return Some(Cow::Borrowed(cache.get_or_init(build_scoped)));
        }
        Some(Cow::Owned(build_scoped()))
    }

    pub(crate) fn recursive_leaf_constraint(&self, leaf_index: usize) -> Option<&Constraint> {
        let layout = self.recursive_parser_layout().ok().flatten()?;
        let leaf = layout.leaves.get(leaf_index)?;
        self.constraint_at_recursive_component_path(&leaf.component_path)
    }

    #[inline]
    pub(crate) fn tokenizer_requires_exact_liveness(&self) -> bool {
        if self.uses_compact_segmented_parser_runtime() {
            self.recursive_parser_layout_ref().is_some_and(|layout| layout.has_virtual_residuals)
        } else { self.tokenizer.has_virtual_residual_runtime() }
    }

    pub(crate) fn exact_runtime_tokenizer_state_has_future(&self, state: u32) -> Result<bool, String> {
        let (body, local) = if self.uses_compact_segmented_parser_runtime() {
            let (leaf, local) = self.recursive_tokenizer_leaf_state(state)
                .ok_or("exact residual liveness received an unknown scoped tokenizer state")?;
            (self.recursive_leaf_constraint(leaf).ok_or("missing residual tokenizer leaf")?, local)
        } else { (self, state) };
        if local == body.runtime_commit_initial_state() || !body.tokenizer.has_virtual_residual_runtime() {
            return Ok(true);
        }
        body.tokenizer.exact_dynamic_state_has_future(local)
    }

    #[inline]
    pub(crate) fn recursive_parser_leaf_state(
        &self,
        scoped_state: u32,
    ) -> Option<(usize, u32)> {
        let layout = self.recursive_parser_layout().ok().flatten()?;
        if scoped_state >= layout.total_states {
            return None;
        }
        let leaf_index = layout
            .leaf_state_offsets
            .partition_point(|&offset| offset <= scoped_state)
            .checked_sub(1)?;
        let leaf = layout.leaves.get(leaf_index)?;
        let local_state = scoped_state.checked_sub(leaf.state_offset)?;
        (local_state < leaf.state_count).then_some((leaf_index, local_state))
    }

    /// Split a recursive parser language by the leaf owning each live stack
    /// top. `LeveledGSS::isolate` prunes only the selected top branch, so the
    /// original lower-stack and accumulator correlations remain intact. This is
    /// the exact bridge needed when a zero-width CALL/RETURN closure changes the
    /// active lexer scope after one terminal commits.
    pub(crate) fn partition_recursive_parser_gss_by_active_leaf(
        &self,
        gss: &ParserGSS,
    ) -> Option<SmallVec<[(usize, ParserGSS); 4]>> {
        if !self.uses_compact_segmented_parser_runtime() {
            return None;
        }
        if gss.is_empty() {
            return Some(SmallVec::new());
        }
        let tops = gss.peek_values();
        if tops.is_empty() {
            // A non-empty parser language with an empty stack has no active
            // component to own a tokenizer continuation. Parser runtime stacks
            // are rooted, so treat this as an invalid recursive frontier rather
            // than guessing a scope.
            return None;
        }
        let mut partitions = SmallVec::<[(usize, ParserGSS); 4]>::new();
        for top in tops {
            let (leaf_index, _) = self.recursive_parser_leaf_state(top)?;
            let branch = gss.isolate(Some(top));
            if branch.is_empty() {
                continue;
            }
            if let Some((_, existing)) = partitions
                .iter_mut()
                .find(|(candidate, _)| *candidate == leaf_index)
            {
                *existing = existing.merge(&branch);
            } else {
                partitions.push((leaf_index, branch));
            }
        }
        partitions.sort_unstable_by_key(|(leaf_index, _)| *leaf_index);
        Some(partitions)
    }

    /// Project the exact scoped one-step relations, retaining controls as
    /// private labels. Exact leaf terminal IDs occupy their existing runtime
    /// namespace after the outer (possibly aliased) terminal coordinate.
    pub(crate) fn recursive_explicit_control_parser_table(&self) -> Result<GLRTable, String> {
        let layout = self.recursive_parser_layout_for_pending_root()?
            .ok_or_else(|| "missing recursive parser layout".to_owned())?;
        let tables = RecursiveSegmentedParserTables { root: self, layout: &layout };
        let provider = DisjointComponentActionProvider::with_state_offsets(
            &tables, &layout.links, &layout.leaf_state_offsets,
        )?;
        let mut symbols = layout.terminal_targets.iter().map(|targets| {
            targets.iter().map(|&(component, terminal)| ScopedParserSymbol::Terminal {
                component, terminal,
            }).collect::<SmallVec<[ScopedParserSymbol; 4]>>()
        }).collect::<Vec<_>>();
        for (component, leaf) in layout.leaves.iter().enumerate() {
            let constraint = self.constraint_at_recursive_component_path(&leaf.component_path)
                .ok_or_else(|| "composition leaf path no longer resolves".to_owned())?;
            for terminal in 0..constraint.parser_terminal_count() {
                symbols.push(smallvec::smallvec![ScopedParserSymbol::Terminal {
                    component: component as u32, terminal,
                }]);
            }
        }
        crate::compiler::glr::parser::materialize_scoped_provider_table(
            &provider, &symbols, Some(ScopedParserSymbol::Terminal {
                component: 0, terminal: crate::compiler::glr::analysis::EOF,
            }),
        )
    }

    /// Exact ordinary-terminal GLR table for compiler-side analyses that still
    /// consume a table rather than a `ParserActionProvider`. Its parser-state
    /// alphabet is the live recursive leaf coordinate; CALL/RETURN are first
    /// encoded as private controls and then eliminated exactly.
    pub(crate) fn recursive_control_eliminated_parser_table(
        &self,
    ) -> Result<Option<Arc<GLRTable>>, String> {
        let Some(layout) = self.recursive_parser_layout_for_pending_root()? else {
            return Ok(None);
        };
        let tables = RecursiveSegmentedParserTables {
            root: self,
            layout: &layout,
        };
        let provider = DisjointComponentActionProvider::with_state_offsets(
            &tables,
            &layout.links,
            &layout.leaf_state_offsets,
        )?;
        let terminal_symbols = layout
            .terminal_targets
            .iter()
            .map(|targets| {
                targets
                    .iter()
                    .map(|&(component, terminal)| ScopedParserSymbol::Terminal {
                        component,
                        terminal,
                    })
                    .collect::<SmallVec<[ScopedParserSymbol; 4]>>()
            })
            .collect::<Vec<_>>();
        let table = materialize_control_eliminated_scoped_provider_table(
            &provider,
            &terminal_symbols,
        )?;
        Ok(Some(Arc::new(table)))
    }

    /// Drop the materialized-table start-state ownership sets once recursive
    /// wrapper ownership is authoritative. v23 requires those bitsets on the
    /// wire; v24 recursive runtimes do not need them after load/build.
    pub(crate) fn clear_recursive_legacy_boundary_start_states(&mut self) {
        if !self.uses_compact_segmented_parser_runtime() {
            return;
        }
        let Some(overlay) = self.static_dynamic_overlay.as_mut() else {
            return;
        };
        for component in &mut overlay.segmented_parser_components {
            if let Some(shard) = component.boundary.as_mut() {
                shard.start_parser_states = BitSet::new(0);
            }
        }
        overlay.segmented_boundary_shards = overlay
            .segmented_parser_components
            .iter()
            .filter_map(|component| component.boundary.clone())
            .collect();
    }

    /// Drop the materialized composed-table -> immediate-component parser
    /// projections after every runtime consumer has moved to recursive wrapper
    /// ownership. v25 stores recursive B directly; v24 legacy static B remains
    /// on the materialized runtime and therefore retains these projections.
    pub(crate) fn clear_recursive_legacy_parser_state_projections(&mut self) {
        if !self.uses_compact_segmented_parser_runtime() {
            return;
        }
        let Some(overlay) = self.static_dynamic_overlay.as_mut() else {
            return;
        };
        for component in &mut overlay.segmented_parser_components {
            component.global_to_local_parser_state.clear();
        }
    }

    fn recursive_parser_symbols_for_global_terminal(
        &self,
        layout: &RecursiveParserLayout,
        global_terminal: TerminalID,
        out: &mut SmallVec<[ScopedParserSymbol; 8]>,
    ) -> bool {
        out.clear();
        let Some(targets) = layout.terminal_targets.get(global_terminal as usize) else {
            return false;
        };
        for &(component, terminal) in targets {
            let symbol = ScopedParserSymbol::Terminal {
                component,
                terminal,
            };
            if !out.contains(&symbol) {
                out.push(symbol);
            }
        }
        !out.is_empty()
    }

    fn recursive_parser_symbols_for_runtime_terminal(
        &self,
        layout: &RecursiveParserLayout,
        runtime_terminal: TerminalID,
        out: &mut SmallVec<[ScopedParserSymbol; 8]>,
    ) -> bool {
        out.clear();
        if let Some((leaf_index, local_terminal)) =
            self.recursive_terminal_leaf_local(runtime_terminal)
        {
            out.push(ScopedParserSymbol::Terminal {
                component: leaf_index as u32,
                terminal: local_terminal,
            });
            return true;
        }
        self.recursive_parser_symbols_for_global_terminal(layout, runtime_terminal, out)
    }

    /// Correctness/reference entry point for the recursive endpoint parser
    /// coordinate. This deliberately does not switch the live masking/commit
    /// path yet: component A projection and dynamic-trigger routing must move
    /// to the same recursive ownership model atomically.
    pub(crate) fn close_recursive_segmented_parser_reference(
        &self,
        stack: &ParserGSS,
    ) -> Result<Option<ParserGSS>, String> {
        let Some(layout) = self.recursive_parser_layout()? else {
            return Ok(None);
        };
        if let Some(parser) = &self.template_parser {
            let composition = parser.composition.as_deref()
                .ok_or("native recursive parser has no scoped relations")?;
            return Ok(Some(close_provider_control_stacks(
                &RecursiveTemplateReference { parser, composition }, stack,
            )));
        }
        let tables = RecursiveSegmentedParserTables {
            root: self,
            layout: &layout,
        };
        let provider = DisjointComponentActionProvider::with_state_offsets(
            &tables,
            &layout.links,
            &layout.leaf_state_offsets,
        )?;
        for (component, leaf) in layout.leaves.iter().enumerate() {
            if provider.scoped_state(component as u32, 0) != Some(leaf.state_offset) {
                return Err(format!(
                    "recursive parser leaf {component} offset disagrees with provider coordinate",
                ));
            }
        }
        Ok(Some(close_provider_control_stacks(&provider, stack)))
    }

    /// Advance one composed/global terminal through the recursive leaf view.
    /// Global terminal IDs remain only an outer migration coordinate here;
    /// each provider action itself sees a leaf-local terminal.
    pub(crate) fn advance_recursive_segmented_parser_reference(
        &self,
        stack: &ParserGSS,
        global_terminal: TerminalID,
    ) -> Result<Option<ParserGSS>, String> {
        let Some(layout) = self.recursive_parser_layout()? else {
            return Ok(None);
        };
        if let Some(parser) = &self.template_parser {
            let composition = parser.composition.as_deref()
                .ok_or("native recursive parser has no scoped relations")?;
            let provider = RecursiveTemplateReference { parser, composition };
            return Ok(Some(advance_provider_control_closed_stacks(
                &provider, stack, global_terminal,
            )));
        }
        let tables = RecursiveSegmentedParserTables {
            root: self,
            layout: &layout,
        };
        let provider = DisjointComponentActionProvider::with_state_offsets(
            &tables,
            &layout.links,
            &layout.leaf_state_offsets,
        )?;
        let mut symbols = SmallVec::<[ScopedParserSymbol; 8]>::new();
        if !self.recursive_parser_symbols_for_global_terminal(&layout, global_terminal, &mut symbols)
        {
            return Ok(Some(ParserGSS::empty()));
        }
        let mut advanced = ParserGSS::empty();
        for symbol in symbols {
            let branch = advance_provider_control_closed_stacks(&provider, stack, symbol);
            if advanced.is_empty() {
                advanced = branch;
            } else if !branch.is_empty() {
                advanced = advanced.merge(&branch);
            }
        }
        Ok(Some(advanced))
    }

    pub(crate) fn recursive_segmented_parser_is_finished_reference(
        &self,
        stack: &ParserGSS,
    ) -> Result<Option<bool>, String> {
        let Some(layout) = self.recursive_parser_layout()? else {
            return Ok(None);
        };
        if let Some(parser) = &self.template_parser {
            let composition = parser.composition.as_deref()
                .ok_or("native recursive parser has no scoped relations")?;
            return Ok(Some(stacks_finished_with_provider(
                &RecursiveTemplateReference { parser, composition }, stack,
                crate::compiler::glr::analysis::EOF,
            )));
        }
        let tables = RecursiveSegmentedParserTables {
            root: self,
            layout: &layout,
        };
        let provider = DisjointComponentActionProvider::with_state_offsets(
            &tables,
            &layout.links,
            &layout.leaf_state_offsets,
        )?;
        Ok(Some(stacks_finished_with_provider(
            &provider,
            stack,
            ScopedParserSymbol::Terminal {
                component: 0,
                terminal: crate::compiler::glr::analysis::EOF,
            },
        )))
    }

    /// Whether this live, source-built composition can use the compact parser
    /// coordinate directly in the ordinary `ParserGSS<u32>`. Serialized legacy
    /// overlays intentionally omit links/offsets and therefore stay on the old
    /// composed-table coordinate until the wire format is versioned.
    #[inline]
    pub(crate) fn uses_compact_segmented_parser_runtime(&self) -> bool {
        self.static_dynamic_overlay.as_ref().is_some_and(|overlay| {
            let boundary_is_provider_native = overlay.segmented_parser_components.iter().all(|component| {
                match component.boundary.as_ref().map(|shard| &shard.backend) {
                    Some(super::artifact::SegmentedBoundaryShardBackend::DynamicDirect) => true,
                    Some(super::artifact::SegmentedBoundaryShardBackend::StaticParser(boundary)) => {
                        boundary.recursive_parser_dwa.is_some()
                    }
                    None => true,
                    _ => false,
                }
            });
            overlay.segmented_mask_authoritative
                && !overlay.segmented_parser_components.is_empty()
                && !overlay.segmented_parser_links.is_empty()
                && boundary_is_provider_native
        })
    }

    pub(crate) fn close_compact_segmented_parser(
        &self,
        stack: &ParserGSS,
    ) -> Option<ParserGSS> {
        if let Some(provider) = self.template_composition_provider() {
            return Some(close_provider_control_stacks(&provider, stack));
        }
        if !self.uses_compact_segmented_parser_runtime() {
            return None;
        }
        let layout = self
            .recursive_parser_layout()
            .expect("validated recursive compact parser metadata")?;
        let tables = RecursiveSegmentedParserTables {
            root: self,
            layout: &layout,
        };
        let provider = DisjointComponentActionProvider::with_state_offsets(
            &tables,
            &layout.links,
            &layout.leaf_state_offsets,
        )
        .expect("validated compact segmented parser metadata");
        Some(close_provider_control_stacks(&provider, stack))
    }

    pub(crate) fn advance_compact_segmented_parser(
        &self,
        stack: &ParserGSS,
        global_terminal: u32,
    ) -> Option<ParserGSS> {
        if let Some(provider) = self.template_composition_provider() {
            return Some(advance_provider_control_closed_stacks(&provider, stack, global_terminal));
        }
        if !self.uses_compact_segmented_parser_runtime() {
            return None;
        }
        let layout = self
            .recursive_parser_layout()
            .expect("validated recursive compact parser metadata")?;
        let tables = RecursiveSegmentedParserTables {
            root: self,
            layout: &layout,
        };
        let provider = DisjointComponentActionProvider::with_state_offsets(
            &tables,
            &layout.links,
            &layout.leaf_state_offsets,
        )
        .expect("validated compact segmented parser metadata");
        let mut symbols = SmallVec::<[ScopedParserSymbol; 8]>::new();
        if !self.recursive_parser_symbols_for_runtime_terminal(
            &layout,
            global_terminal,
            &mut symbols,
        ) {
            return Some(ParserGSS::empty());
        }
        let mut advanced = ParserGSS::empty();
        for symbol in symbols {
            let branch = advance_provider_control_closed_stacks(&provider, stack, symbol);
            if advanced.is_empty() {
                advanced = branch;
            } else if !branch.is_empty() {
                advanced = advanced.merge(&branch);
            }
        }
        Some(advanced)
    }

    pub(crate) fn compact_segmented_parser_may_advance_on(
        &self,
        stack: &ParserGSS,
        global_terminal: u32,
    ) -> Option<bool> {
        if let Some(provider) = self.template_composition_provider() {
            return Some(stack_may_advance_on_with_provider(&provider, stack, global_terminal));
        }
        if !self.uses_compact_segmented_parser_runtime() {
            return None;
        }
        let layout = self
            .recursive_parser_layout()
            .expect("validated recursive compact parser metadata")?;
        let tables = RecursiveSegmentedParserTables {
            root: self,
            layout: &layout,
        };
        let provider = DisjointComponentActionProvider::with_state_offsets(
            &tables,
            &layout.links,
            &layout.leaf_state_offsets,
        )
        .expect("validated compact segmented parser metadata");
        let mut symbols = SmallVec::<[ScopedParserSymbol; 8]>::new();
        if !self.recursive_parser_symbols_for_runtime_terminal(
            &layout,
            global_terminal,
            &mut symbols,
        ) {
            return Some(false);
        }
        Some(
            symbols
                .into_iter()
                .any(|symbol| stack_may_advance_on_with_provider(&provider, stack, symbol)),
        )
    }

    pub(crate) fn compact_segmented_parser_may_advance_on_any(
        &self,
        stack: &ParserGSS,
        terminals: &BitSet,
    ) -> Option<bool> {
        if let Some(provider) = self.template_composition_provider() {
            return Some(stack_may_advance_on_any_with_provider(&provider, stack,
                terminals.iter_ones().map(|terminal| terminal as u32)));
        }
        if !self.uses_compact_segmented_parser_runtime() {
            return None;
        }
        let layout = self.recursive_parser_layout()
            .expect("validated recursive compact parser metadata")?;
        let tables = RecursiveSegmentedParserTables { root: self, layout: &layout };
        let provider = DisjointComponentActionProvider::with_state_offsets(
            &tables, &layout.links, &layout.leaf_state_offsets,
        ).expect("validated compact segmented parser metadata");
        let symbols = terminals.iter_ones().flat_map(|terminal| {
            let mut symbols = SmallVec::<[ScopedParserSymbol; 8]>::new();
            self.recursive_parser_symbols_for_runtime_terminal(
                &layout, terminal as u32, &mut symbols,
            );
            symbols
        });
        Some(stack_may_advance_on_any_with_provider(&provider, stack, symbols))
    }

    pub(crate) fn compact_segmented_parser_admitted_terminals(
        &self,
        stack: &ParserGSS,
        candidates: &BitSet,
    ) -> Option<BitSet> {
        if self.template_composition_provider().is_some() {
            return Some(self.template_parser.as_ref().unwrap().admitted(stack, candidates));
        }
        if !self.uses_compact_segmented_parser_runtime() {
            return None;
        }
        let layout = self.recursive_parser_layout()
            .expect("validated recursive compact parser metadata")?;
        let tables = RecursiveSegmentedParserTables { root: self, layout: &layout };
        let provider = DisjointComponentActionProvider::with_state_offsets(
            &tables, &layout.links, &layout.leaf_state_offsets,
        ).expect("validated compact segmented parser metadata");
        let symbols = candidates.iter_ones().flat_map(|terminal| {
            let mut symbols = SmallVec::<[ScopedParserSymbol; 8]>::new();
            self.recursive_parser_symbols_for_runtime_terminal(
                &layout, terminal as u32, &mut symbols,
            );
            symbols.into_iter().map(move |symbol| (terminal, symbol))
        });
        let mut admitted = BitSet::new(candidates.len());
        for_each_admitted_symbol_with_provider(&provider, stack, symbols,
            |terminal| admitted.set(terminal));
        Some(admitted)
    }

    /// Exact existential intersection of parser admission and a caller's
    /// terminal predicate. Candidate supports may over-approximate lexical
    /// liveness: only a terminal passing BOTH checks can establish success.
    pub(crate) fn compact_segmented_parser_may_advance_on_any_matching(
        &self,
        stack: &ParserGSS,
        candidates: impl IntoIterator<Item = u32>,
        mut matches: impl FnMut(u32) -> bool,
    ) -> Option<bool> {
        if let Some(provider) = self.template_composition_provider() {
            return Some(find_admitted_symbol_with_provider(&provider, stack,
                candidates.into_iter().map(|terminal| (terminal, terminal)),
                |&terminal| matches(terminal)).is_some());
        }
        if !self.uses_compact_segmented_parser_runtime() {
            return None;
        }
        let layout = self.recursive_parser_layout()
            .expect("validated recursive compact parser metadata")?;
        let tables = RecursiveSegmentedParserTables { root: self, layout: &layout };
        let provider = DisjointComponentActionProvider::with_state_offsets(
            &tables, &layout.links, &layout.leaf_state_offsets,
        ).expect("validated compact segmented parser metadata");
        let symbols = candidates.into_iter().flat_map(|terminal| {
            let mut symbols = SmallVec::<[ScopedParserSymbol; 8]>::new();
            self.recursive_parser_symbols_for_runtime_terminal(
                &layout, terminal, &mut symbols,
            );
            symbols.into_iter().map(move |symbol| (terminal, symbol))
        });
        Some(find_admitted_symbol_with_provider(
            &provider, stack, symbols, |&terminal| matches(terminal),
        ).is_some())
    }

    pub(crate) fn compact_segmented_parser_is_finished(
        &self,
        stack: &ParserGSS,
    ) -> Option<bool> {
        if let Some(provider) = self.template_composition_provider() {
            return Some(stacks_finished_with_provider(&provider, stack,
                crate::compiler::glr::analysis::EOF));
        }
        if !self.uses_compact_segmented_parser_runtime() {
            return None;
        }
        let layout = self
            .recursive_parser_layout()
            .expect("validated recursive compact parser metadata")?;
        let tables = RecursiveSegmentedParserTables {
            root: self,
            layout: &layout,
        };
        let provider = DisjointComponentActionProvider::with_state_offsets(
            &tables,
            &layout.links,
            &layout.leaf_state_offsets,
        )
        .expect("validated compact segmented parser metadata");
        Some(stacks_finished_with_provider(
            &provider,
            stack,
            ScopedParserSymbol::Terminal {
                component: 0,
                terminal: crate::compiler::glr::analysis::EOF,
            },
        ))
    }

    #[inline]
    pub(crate) fn compact_segmented_parser_local_state(
        &self,
        component_index: usize,
        scoped_state: u32,
    ) -> Option<u32> {
        if !self.uses_compact_segmented_parser_runtime() {
            return None;
        }
        let layout = self
            .recursive_parser_layout()
            .expect("validated recursive compact parser metadata")?;
        let offset = *layout.component_offsets.get(component_index)?;
        let end = layout
            .component_offsets
            .get(component_index + 1)
            .copied()
            .unwrap_or(layout.total_states);
        if scoped_state < offset || scoped_state >= end {
            return None;
        }
        let local = scoped_state.checked_sub(offset)?;
        Some(local)
    }

    #[inline]
    pub(crate) fn compact_segmented_parser_component(
        &self,
        scoped_state: u32,
    ) -> Option<(usize, u32)> {
        if !self.uses_compact_segmented_parser_runtime() {
            return None;
        }
        let layout = self
            .recursive_parser_layout()
            .expect("validated recursive compact parser metadata")?;
        if scoped_state >= layout.total_states {
            return None;
        }
        let leaf_index = layout
            .leaf_state_offsets
            .partition_point(|&offset| offset <= scoped_state)
            .checked_sub(1)?;
        let component_index = layout.leaves.get(leaf_index)?.top_component as usize;
        let local = scoped_state.checked_sub(*layout.component_offsets.get(component_index)?)?;
        Some((component_index, local))
    }

    #[inline]
    pub(crate) fn uses_sparse_direct_regular_runtime(&self) -> bool {
        if self.has_template_parser() { return false; }
        self.direct_regular_automaton.is_some()
            && self.table.num_rules == 0
            && self.table.action.is_empty()
    }

    #[cold]
    pub(crate) fn prime_initial_commit_hot_path(&self) {
        let mut state = ConstraintState {
            terminated: false,
            constraint: self,
            state: self.initial_state_map(),
            buffers: Default::default(),
            generation: 0,
            mask_cache: Mutex::new(None),
            mask_scratch: Arc::new(Mutex::new(crate::runtime::state::MaskScratch::for_constraint(self))),
        };
        state.prefill_mask_cache();
        state.reserve_linear_stack_hot_path();

        let token_ids = {
            let cache = state.mask_cache.lock().unwrap();
            let Some(mask) = cache.as_ref().map(|cache| cache.mask.as_slice()) else {
                return;
            };
            if self.has_template_parser() {
                template_initial_prime_token_ids(mask)
            } else {
                let Some(token_ids) = initial_commit_prime_token_ids(mask) else {
                    return;
                };
                token_ids
            }
        };

        let initial_state = &state.state;
        let buffers = &mut state.buffers;
        super::commit::prime_initial_commits(self, initial_state, buffers, &token_ids);
    }

    /// Attach the derived lexer artifacts used only by exact dynamic masking.
    ///
    /// In particular, finite projections of symbolic residual lexers are a
    /// mask-execution optimization, not part of compilation semantics. They
    /// can cost several milliseconds to construct for a constraint that may
    /// never be asked for a mask, so callers should invoke this only for the
    /// runtime vocabulary selected by `dynamic_mask_vocab_for_runtime` (or for
    /// an already-serialized projection that merely needs its derived table
    /// rebuilt after load).
    #[inline]
    fn dynamic_runtime_max_token_byte_len(&self, vocab: &DynamicMaskVocab) -> usize {
        // A materialized DynamicMaskVocab is built from the exact runtime token
        // vocabulary and stores the maximum token length in its trie root. Use
        // that O(1) metadata instead of rescanning every model token through
        // Constraint::max_token_byte_len(). The fallback is needed only for
        // unmaterialized transfer/compiler values.
        if vocab.is_initialized() {
            return vocab.max_token_byte_len();
        }
        if let Some(bound_vocab) = self.late_bind_vocab.get() {
            return bound_vocab.max_token_byte_len();
        }
        self.max_token_byte_len()
    }

    pub(crate) fn prepare_dynamic_virtual_residual_mask_projection(
        &self,
        vocab: &mut DynamicMaskVocab,
    ) {
        let max_token_len = self.dynamic_runtime_max_token_byte_len(vocab);
        if max_token_len > 0
            && vocab.mask_projection_tokenizer().is_none()
            && self.tokenizer.has_any_virtual_runtime()
        {
            let virtual_residual_mask_projection_enabled = std::env::var(
                "GLRMASK_DYNAMIC_VIRTUAL_RESIDUAL_MASK_PROJECTION",
            )
            .ok()
            .is_none_or(|value| !matches!(value.trim(), "0" | "false" | "no" | "off"));

            // RESOURCE budget, not a corpus-tuned state threshold. The dense
            // estimate charges one u32 index per candidate coordinate before
            // the projection retains only reachable sparse states.
            const DEFAULT_VIRTUAL_RESIDUAL_PROJECTION_MAX_DENSE_STATES: usize = 1024 * 1024;
            let max_dense_states = std::env::var(
                "GLRMASK_DYNAMIC_VIRTUAL_RESIDUAL_MASK_PROJECTION_MAX_DENSE_STATES",
            )
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(DEFAULT_VIRTUAL_RESIDUAL_PROJECTION_MAX_DENSE_STATES);
            let projection_work = self
                .tokenizer
                .virtual_residual_mask_projection_dense_state_work(max_token_len);
            let within_budget = projection_work.is_some_and(|work| work <= max_dense_states);
            if virtual_residual_mask_projection_enabled
                && within_budget
                && let Some((mask_tokenizer, projections)) = self
                    .tokenizer
                    .virtual_residuals_mask_tokenizer_with_vocab(
                        max_token_len,
                        self.late_bind_vocab.get(),
                    )
            {
                vocab.set_virtual_residuals_mask_projection(mask_tokenizer, projections);
            }
        }
    }

    pub(crate) fn prepare_dynamic_mask_runtime_artifacts(&self, vocab: &mut DynamicMaskVocab) {
        let profile_runtime_mask = std::env::var_os("GLRMASK_PROFILE_DYNAMIC_MASK").is_some();
        let max_token_started = profile_runtime_mask.then(std::time::Instant::now);
        let max_token_len = self.dynamic_runtime_max_token_byte_len(vocab);
        if let Some(started) = max_token_started {
            eprintln!(
                "[glrmask/profile][dynamic_mask_runtime_prepare] max_token_len_ms={:.3} value={}",
                started.elapsed().as_secs_f64() * 1e3,
                max_token_len,
            );
        }
        let projection_started = profile_runtime_mask.then(std::time::Instant::now);
        self.prepare_dynamic_virtual_residual_mask_projection(vocab);
        if max_token_len > 0
            && vocab.mask_projection_tokenizer().is_none()
            && self.tokenizer.has_any_virtual_runtime()
        {
            if let Some((mask_tokenizer, projections)) = self
                .tokenizer
                .virtual_binary_repeat_intersections_mask_tokenizer(max_token_len)
            {
                vocab.set_virtual_repeat_intersections_mask_projection(mask_tokenizer, projections);
            } else if let Some((mask_tokenizer, projection)) =
                self.tokenizer.virtual_unit_repeat_mask_tokenizer(max_token_len)
            {
                vocab.set_virtual_unit_repeat_mask_projection(mask_tokenizer, projection);
            }
        }
        if let Some(started) = projection_started {
            eprintln!(
                "[glrmask/profile][dynamic_mask_runtime_prepare] virtual_residual_projection_ms={:.3}",
                started.elapsed().as_secs_f64() * 1e3,
            );
        }

        let mask_execution_source = vocab.mask_runtime_tokenizer().unwrap_or(&self.tokenizer);
        let eager_mask_execution = std::env::var("GLRMASK_DYNAMIC_EAGER_MASK_EXECUTION")
            .ok()
            .is_none_or(|value| !matches!(value.trim(), "0" | "false" | "no" | "off"));
        // A finite projection whose only epsilon structure is its reset
        // dispatcher executes directly over raw scalar component rows. The
        // 0x8000 bound is the Flat16 representation boundary.
        let scalar_started = profile_runtime_mask.then(std::time::Instant::now);
        let scalar_dispatch = mask_execution_source.has_scalar_deterministic_dispatch();
        if let Some(started) = scalar_started {
            eprintln!(
                "[glrmask/profile][dynamic_mask_runtime_prepare] scalar_dispatch_proof_ms={:.3} result={}",
                started.elapsed().as_secs_f64() * 1e3,
                scalar_dispatch,
            );
        }
        if eager_mask_execution
            && mask_execution_source.has_epsilon_transitions()
            && !mask_execution_source.has_any_virtual_runtime()
            && !scalar_dispatch
        {
            let _ = vocab.prepare_mask_execution(&self.tokenizer, max_token_len);
        }
        let full_walk_started = profile_runtime_mask.then(std::time::Instant::now);
        vocab.prepare_full_walk_fast_transitions(&self.tokenizer);
        if let Some(started) = full_walk_started {
            eprintln!(
                "[glrmask/profile][dynamic_mask_runtime_prepare] full_walk_fast_ms={:.3}",
                started.elapsed().as_secs_f64() * 1e3,
            );
        }
        let master_slice_started = profile_runtime_mask.then(std::time::Instant::now);
        let max_safe_chars = u32::from(vocab.llg_master_max_safe_chars());
        for &slice_id in &[0u32, 3u32] {
            if let Some(slice) = vocab.llg_slice_by_cache_id(slice_id) {
                self.tokenizer.prepare_virtual_residual_master_slice_artifacts(
                    slice.dfa().start_state(),
                    slice.dfa().class_count(),
                    slice.dfa().byte_to_class_map(),
                    slice.dfa().transition_table(),
                    slice.dfa().accepting_map(),
                    slice.dfa().can_reach_accepting_map(),
                    max_safe_chars,
                );
            }
        }
        if let Some(started) = master_slice_started {
            eprintln!(
                "[glrmask/profile][dynamic_mask_runtime_prepare] master_slice_artifacts_ms={:.3}",
                started.elapsed().as_secs_f64() * 1e3,
            );
        }

        // Projected-terminal containment quotients are proof accelerators, not
        // prerequisites for exact masking. Build them only once a master-slice
        // proof survives the cheap runtime eligibility gates.
    }

    /// Return the direct-dynamic vocabulary, materializing it only when a
    /// dynamic mask is actually requested. Static constraints with complete
    /// possible-matches tables never pay this cost; deferred-PM constraints
    /// pay it on their first exact fallback instead of during compile/load.
    /// Symbolic lexer finite projections are likewise deferred to first mask:
    /// they are execution accelerators and must not inflate dynamic compile
    /// latency for constraints that are never sampled.
    pub(crate) fn dynamic_mask_vocab_for_runtime(&self) -> &DynamicMaskVocab {
        let needs_runtime_projection = self.tokenizer.has_any_virtual_runtime()
            && self.dynamic_mask_vocab.mask_projection_tokenizer().is_none();
        if self.dynamic_mask_vocab.is_initialized() && !needs_runtime_projection {
            return &self.dynamic_mask_vocab;
        }
        self.lazy_dynamic_mask_vocab.get_or_init(|| {
            let profile_runtime_mask = std::env::var_os("GLRMASK_PROFILE_DYNAMIC_MASK").is_some();
            let total_started = profile_runtime_mask.then(std::time::Instant::now);
            let mut vocab = self.dynamic_mask_vocab.clone();
            let materialize_started = profile_runtime_mask.then(std::time::Instant::now);
            let _ = vocab.materialize_pending_source();
            if !vocab.is_initialized() {
                vocab = self.build_dynamic_mask_vocab();
            }
            if let Some(started) = materialize_started {
                eprintln!(
                    "[glrmask/profile][dynamic_mask_first_use] vocab_materialize_ms={:.3}",
                    started.elapsed().as_secs_f64() * 1e3,
                );
            }
            // A loaded/static fallback may have had only the compact vocab
            // source serialized. Rebuild vocabulary-only slice metadata before
            // any lazy lexer quotient tries to consume those proof languages.
            let slice_started = profile_runtime_mask.then(std::time::Instant::now);
            // Recursive masking uses scoped leaf lexers, not this transitional
            // outer tokenizer. Its source-only projection/slice proofs cannot
            // be used by the provider and must not be built on its first mask.
            let recursive_provider = self.uses_compact_segmented_parser_runtime();
            if !recursive_provider {
                self.prepare_llg_slice_leftovers(&mut vocab);
            }
            if let Some(started) = slice_started {
                eprintln!(
                    "[glrmask/profile][dynamic_mask_first_use] slice_prepare_ms={:.3}",
                    started.elapsed().as_secs_f64() * 1e3,
                );
            }
            let runtime_started = profile_runtime_mask.then(std::time::Instant::now);
            if !recursive_provider {
                self.prepare_dynamic_mask_runtime_artifacts(&mut vocab);
            }
            if let Some(started) = runtime_started {
                eprintln!(
                    "[glrmask/profile][dynamic_mask_first_use] lexer_runtime_prepare_ms={:.3}",
                    started.elapsed().as_secs_f64() * 1e3,
                );
            }
            if let Some(started) = total_started {
                eprintln!(
                    "[glrmask/profile][dynamic_mask_first_use] total_ms={:.3}",
                    started.elapsed().as_secs_f64() * 1e3,
                );
            }
            vocab
        })
    }

    /// Return an already-materialized dynamic-mask vocabulary without causing
    /// first-use runtime work.  Callers that only want to consult an optional
    /// memoization cache must use this instead of `dynamic_mask_vocab_for_runtime`.
    #[inline]
    fn initialized_dynamic_mask_vocab_for_runtime(&self) -> Option<&DynamicMaskVocab> {
        if let Some(vocab) = self.lazy_dynamic_mask_vocab.get() {
            return Some(vocab);
        }
        self.dynamic_mask_vocab.is_initialized().then_some(&self.dynamic_mask_vocab)
    }

    fn build_dynamic_mask_vocab(&self) -> DynamicMaskVocab {
        let profile = std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
            || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some();
        let total_started_at = profile.then(std::time::Instant::now);
        let collect_started_at = profile.then(std::time::Instant::now);
        // `token_bytes` is id-sorted. Runtime traversal needs byte-sorted,
        // duplicate-collapsed leaves instead. Borrow the source byte slices
        // until trie construction is complete: this avoids cloning every token
        // once into a BTreeMap and again into VocabPrefixTree::build_owned.
        let mut sorted_tokens = self.token_bytes_iter().collect::<Vec<_>>();
        let collect_ms = collect_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let sort_started_at = profile.then(std::time::Instant::now);
        let sort_tokens = |left: &(u32, &[u8]), right: &(u32, &[u8])| {
            left.1.cmp(right.1).then_with(|| left.0.cmp(&right.0))
        };
        if rayon::current_num_threads() == 1 {
            sorted_tokens.sort_unstable_by(sort_tokens);
        } else {
            sorted_tokens.par_sort_unstable_by(sort_tokens);
        }

        let sort_ms = sort_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let aliases_started_at = profile.then(std::time::Instant::now);
        // Match the compiler's OrderedVocab coordinate system exactly: trie
        // token ids are dense byte-order ids, and each id maps back to one or
        // more original model token ids. Using an original token id as the trie
        // id is incorrect when lexical byte order differs from token-id order.
        let mut token_aliases = Vec::with_capacity(sorted_tokens.len());
        let mut trie_entries = Vec::with_capacity(sorted_tokens.len());

        let mut start = 0usize;
        while start < sorted_tokens.len() {
            let bytes = sorted_tokens[start].1;
            let mut end = start + 1;
            while end < sorted_tokens.len() && sorted_tokens[end].1 == bytes {
                end += 1;
            }

            let ordered_id = token_aliases.len() as u32;
            let aliases = if end == start + 1 {
                PackedDynamicMaskTokenAliases::Single(sorted_tokens[start].0)
            } else {
                PackedDynamicMaskTokenAliases::Many(
                    sorted_tokens[start..end]
                        .iter()
                        .map(|(token_id, _)| *token_id)
                        .collect::<Vec<_>>()
                        .into_boxed_slice(),
                )
            };
            token_aliases.push(Some(aliases));
            trie_entries.push((ordered_id as usize, bytes));
            start = end;
        }

        let aliases_ms = aliases_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let trie_started_at = profile.then(std::time::Instant::now);
        // The runtime trie remains one flat radix-trie arena, but construction
        // inserts one zero-byte structural child per character-type vocabulary
        // partition. This deliberately separates lexically different token
        // families without introducing a second trie type or per-partition
        // dense masks. The ordinary walker treats these as no-op edges and can
        // certify an entire partition subtree in one bounded-continuation test.
        let trie = Self::build_dynamic_mask_trie_partitioned(&trie_entries);
        let trie_ms = trie_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        if let Some(total_started_at) = total_started_at {
            let alias_groups = token_aliases.iter().flatten().count();
            let alias_many = token_aliases
                .iter()
                .flatten()
                .filter(|aliases| matches!(aliases, PackedDynamicMaskTokenAliases::Many(_)))
                .count();
            eprintln!(
                "[glrmask/profile][runtime_dynamic_vocab] tokens={} unique_bytes={} aliases={} alias_many={} collect_ms={:.3} sort_ms={:.3} aliases_ms={:.3} trie_ms={:.3} trie_nodes={} trie_edges={} trie_bytes={} total_ms={:.3}",
                self.token_bytes_count(),
                trie_entries.len(),
                alias_groups,
                alias_many,
                collect_ms,
                sort_ms,
                aliases_ms,
                trie_ms,
                trie.nodes.len(),
                trie.edges.len(),
                trie.edge_bytes_len(),
                total_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        DynamicMaskVocab::from_packed(Arc::new(trie), Arc::new(token_aliases))
    }

    fn prepare_llg_slice_leftovers(&self, vocab: &mut DynamicMaskVocab) {
        if vocab.has_llg_slice_leftovers() {
            return;
        }

        const SAFE_PLUS_CACHE_ID: u32 = 0;
        const WHITESPACE_CACHE_ID: u32 = 3;
        const ASCII_WORD_CACHE_ID: u32 = 0x40;
        let safe_plus = Arc::new(
            VocabPartitionDfa::compile_utf8_regex(
                "llg-safe+",
                r#"[^"\\\x00-\x1F\x7F]+"#,
            )
            .expect("safe-string slice regex must compile"),
        );
        let whitespace = Arc::new(
            VocabPartitionDfa::compile_utf8_regex("llg-whitespace", r"[\x20\x0A\x0D\x09]+")
                .expect("whitespace slice regex must compile"),
        );
        let max_token_byte_len = vocab.max_token_byte_len();
        let ascii_slice_max_token_byte_len = std::env::var(
            "GLRMASK_EXPERIMENT_DIRECT_RESIDUAL_ASCII_WORD_SLICE_MAX_BYTES",
        )
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|&value| value != 0)
        .map_or(max_token_byte_len, |value| value.min(max_token_byte_len));
        let ascii_word = (max_token_byte_len != 0
            && std::env::var_os("GLRMASK_EXPERIMENT_DIRECT_RESIDUAL_ASCII_WORD_SLICE").is_some())
        .then(|| {
            Arc::new(
                VocabPartitionDfa::compile_utf8_regex(
                    "dynamic-optional-space-nonspace-safe",
                    &format!(r#" ?[^\s"\\\x00-\x1F\x7F]{{1,{ascii_slice_max_token_byte_len}}}"#),
                )
                .expect("bounded optional-space nonspace-safe slice regex must compile"),
            )
        });
        let word_len = vocab.all_original_token_words().len();
        let mut safe_words = vec![0u32; word_len];
        let mut whitespace_words = vec![0u32; word_len];
        let mut ascii_word_words = vec![0u32; word_len];
        let mut safe_token_bytes = U8Set::empty();
        let mut whitespace_token_bytes = U8Set::empty();
        let mut ascii_word_token_bytes = U8Set::empty();
        let mut safe_max_token_byte_len = 0u32;
        let mut whitespace_max_token_byte_len = 0u32;
        let mut ascii_word_max_token_byte_len = 0u32;
        let mut ascii_word_residual_entries =
            Vec::<(u16, usize, &[u8])>::with_capacity(vocab.canonical_token_count());
        let mut entries = Vec::<(u16, usize, &[u8])>::with_capacity(vocab.canonical_token_count());
        let mut max_safe_chars = 0u16;

        for canonical in 0..vocab.canonical_token_count() as u32 {
            let Some(originals) = vocab.token_ids(canonical) else { continue; };
            let Some(&first) = originals.first() else { continue; };
            let Some(bytes) = self.token_bytes_for_id(first) else { continue; };
            let is_safe = safe_plus.is_match(bytes);
            let safe_chars = if is_safe {
                let chars = std::str::from_utf8(bytes)
                    .expect("safe-string UTF-8 regex matched invalid UTF-8")
                    .chars()
                    .count();
                u16::try_from(chars).expect("model token exceeds u16 Unicode-scalar count")
            } else {
                0
            };
            let is_whitespace = whitespace.is_match(bytes);
            let is_ascii_word = ascii_word.as_ref().is_some_and(|slice| slice.is_match(bytes));
            max_safe_chars = max_safe_chars.max(safe_chars);
            entries.push((
                crate::runtime::dynamic_mask_llg_master_layout_class(safe_chars, is_whitespace),
                canonical as usize,
                bytes,
            ));
            if is_safe {
                safe_max_token_byte_len = safe_max_token_byte_len.max(bytes.len() as u32);
                for &byte in bytes {
                    safe_token_bytes.insert(byte);
                }
            }
            if is_whitespace {
                whitespace_max_token_byte_len = whitespace_max_token_byte_len.max(bytes.len() as u32);
                for &byte in bytes {
                    whitespace_token_bytes.insert(byte);
                }
            }
            if is_ascii_word {
                ascii_word_max_token_byte_len = ascii_word_max_token_byte_len.max(bytes.len() as u32);
                for &byte in bytes {
                    ascii_word_token_bytes.insert(byte);
                }
            } else {
                ascii_word_residual_entries.push((0, canonical as usize, bytes));
            }
            for &token in originals {
                let word = token as usize / 32;
                if word >= word_len {
                    continue;
                }
                let bit = 1u32 << (token % 32);
                if is_safe {
                    safe_words[word] |= bit;
                }
                if is_whitespace {
                    whitespace_words[word] |= bit;
                }
                if is_ascii_word {
                    ascii_word_words[word] |= bit;
                }
            }
        }

        entries.sort_unstable_by(|left, right| {
            left.2
                .is_empty()
                .cmp(&right.2.is_empty())
                .reverse()
                .then_with(|| left.0.cmp(&right.0))
                .then_with(|| left.2.cmp(right.2))
                .then_with(|| left.1.cmp(&right.1))
        });
        let master_trie = DynamicMaskTrie::from_partitioned_token_refs(&entries);
        ascii_word_residual_entries.sort_unstable_by(|left, right| {
            left.2.cmp(right.2).then_with(|| left.1.cmp(&right.1))
        });
        let ascii_word_residual_trie =
            DynamicMaskTrie::from_partitioned_token_refs(&ascii_word_residual_entries);
        let mut exact_safe_words = vec![vec![0u32; word_len]; usize::from(max_safe_chars) + 1];
        for &(class, canonical, _) in &entries {
            let safe_chars = crate::runtime::dynamic_mask_llg_master_safe_chars(class);
            if safe_chars == 0 {
                continue;
            }
            let Some(originals) = vocab.token_ids(canonical as u32) else { continue; };
            for &token in originals {
                let word = token as usize / 32;
                if word < word_len {
                    exact_safe_words[usize::from(safe_chars)][word] |= 1u32 << (token % 32);
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
        vocab.set_llg_master_admitted_words(max_safe_chars, admitted_words);
        let mut slices = vec![
            (
                SAFE_PLUS_CACHE_ID,
                Arc::clone(&safe_plus),
                Arc::new(DynamicMaskTrie::new()),
                Arc::new(safe_words),
                safe_token_bytes,
                safe_max_token_byte_len,
            ),
            (
                WHITESPACE_CACHE_ID,
                Arc::clone(&whitespace),
                Arc::new(DynamicMaskTrie::new()),
                Arc::new(whitespace_words),
                whitespace_token_bytes,
                whitespace_max_token_byte_len,
            ),
            (
                crate::runtime::DYNAMIC_MASK_LLG_MASTER_CACHE_ID,
                safe_plus,
                Arc::new(master_trie),
                Arc::new(Vec::new()),
                U8Set::empty(),
                0,
            ),
        ];
        if let Some(ascii_word) = ascii_word {
            slices.push((
                ASCII_WORD_CACHE_ID,
                ascii_word,
                Arc::new(ascii_word_residual_trie),
                Arc::new(ascii_word_words),
                ascii_word_token_bytes,
                ascii_word_max_token_byte_len,
            ));
        }
        vocab.set_llg_slice_leftovers(slices);
    }

    fn dynamic_self_loop_projection_candidates(
        &self,
        vocab: &DynamicMaskVocab,
        tokenizer: &Tokenizer,
        bounded64: Option<&[U8Set]>,
    ) -> Vec<(u32, Vec<TerminalID>, bool)> {
        if let Ok(value) = std::env::var("GLRMASK_DYNAMIC_SELF_LOOP_PROJECTION_FORCE_STATES") {
            let mut states = value
                .split(',')
                .filter_map(|value| value.trim().parse::<u32>().ok())
                .map(|state| vocab.mask_projection_state(state))
                .filter(|&state| state < tokenizer.num_states())
                .collect::<Vec<_>>();
            states.sort_unstable();
            states.dedup();
            let forced = states
                .into_iter()
                .filter_map(|state| {
                    let futures = tokenizer
                        .possible_future_terminals_iter(state)
                        .collect::<Vec<_>>();
                    (!futures.is_empty()).then_some((state, futures, false))
                })
                .collect::<Vec<_>>();
            if !forced.is_empty() {
                return forced;
            }
        }
        if let Ok(value) = std::env::var("GLRMASK_DYNAMIC_SELF_LOOP_PROJECTION_FORCE_STATE")
            && let Ok(state) = value.trim().parse::<u32>()
        {
            let state = vocab.mask_projection_state(state);
            if state >= tokenizer.num_states() {
                return Vec::new();
            }
            let futures = tokenizer
                .possible_future_terminals_iter(state)
                .collect::<Vec<_>>();
            if !futures.is_empty() {
                return vec![(state, futures, false)];
            }
        }
        if let Some(bounded64) = bounded64 {
            let quotient_multiplicity = vocab.mask_projection_state_multiplicities();
            let mut parser_rows_by_terminal = vec![0usize; self.parser_terminal_count() as usize];
            for row in (0..self.parser_symbol_count()).filter_map(|top| self.parser_advance_row(top)) {
                for terminal in row.iter() {
                    if let Some(count) = parser_rows_by_terminal.get_mut(terminal) {
                        *count += 1;
                    }
                }
            }
            let mut ranked =
                Vec::<(usize, usize, usize, usize, u32, Vec<TerminalID>)>::new();
            for state in 0..tokenizer.num_states() {
                let safe64_len = bounded64
                    .get(state as usize)
                    .map_or(0, |safe64| safe64.len());
                let futures = tokenizer
                    .possible_future_terminals_iter(state)
                    .collect::<Vec<_>>();
                if futures.is_empty() || futures.len() > 64 {
                    continue;
                }
                let parser_rows = futures
                    .iter()
                    .filter_map(|&terminal| parser_rows_by_terminal.get(terminal as usize).copied())
                    .max()
                    .unwrap_or(0);
                if parser_rows == 0 {
                    continue;
                }
                let multiplicity = quotient_multiplicity
                    .as_ref()
                    .and_then(|counts| counts.get(state as usize))
                    .copied()
                    .unwrap_or(1);
                // Synthesized residual roots added only to make the quotient
                // proof total need not correspond to any exact runtime state.
                // They can never be queried through full_to_mask_state.
                if multiplicity == 0 {
                    continue;
                }
                ranked.push((
                    parser_rows,
                    multiplicity,
                    safe64_len,
                    tokenizer.transitions_from(state).count(),
                    state,
                    futures,
                ));
            }
            // A global parser-row-frequency rank badly starves structurally
            // rare parser states that are nevertheless the hot generation
            // state (large bounded-repeat/string interiors are a common
            // example).  A projection is only useful for its own future
            // terminal signature, so first preserve representation across
            // signatures.  Within a signature, prefer a quotient state that
            // represents many exact runtime states, then broad bounded-byte
            // behavior and transition fanout.
            let mut by_futures = BTreeMap::<
                Vec<TerminalID>,
                Vec<(usize, usize, usize, usize, u32, Vec<TerminalID>)>,
            >::new();
            for candidate in ranked.drain(..) {
                by_futures
                    .entry(candidate.5.clone())
                    .or_default()
                    .push(candidate);
            }
            let mut groups = by_futures
                .into_iter()
                .map(|(futures, mut candidates)| {
                    let total_multiplicity = candidates
                        .iter()
                        .map(|candidate| candidate.1)
                        .sum::<usize>();
                    candidates.sort_unstable_by(|left, right| {
                        (
                            std::cmp::Reverse(left.3),
                            std::cmp::Reverse(left.2),
                            std::cmp::Reverse(left.1),
                            left.4,
                        )
                            .cmp(&(
                                std::cmp::Reverse(right.3),
                                std::cmp::Reverse(right.2),
                                std::cmp::Reverse(right.1),
                                right.4,
                            ))
                    });
                    (total_multiplicity, futures, candidates)
                })
                .collect::<Vec<_>>();
            groups.sort_unstable_by(|left, right| {
                (std::cmp::Reverse(left.0), &left.1)
                    .cmp(&(std::cmp::Reverse(right.0), &right.1))
            });
            if std::env::var_os("GLRMASK_PROFILE_DYNAMIC_PROJECTION_CANDIDATES").is_some() {
                if let Ok(value) = std::env::var("GLRMASK_PROFILE_DYNAMIC_FUTURE_QUOTIENT_STATES") {
                    let requested = value
                        .split(',')
                        .filter_map(|value| value.trim().parse::<u32>().ok())
                        .map(|state| vocab.mask_projection_state(state))
                        .collect::<BTreeSet<_>>();
                    for (_, futures, candidates) in &groups {
                        for (parser_rows, multiplicity, safe64, transitions, state, _) in candidates {
                            if requested.contains(state) {
                                eprintln!(
                                    "[glrmask/profile][dynamic_projection_requested_candidate] state={} parser_rows={} multiplicity={} safe64={} transitions={} futures={:?} family_candidates={}",
                                    state,
                                    parser_rows,
                                    multiplicity,
                                    safe64,
                                    transitions,
                                    futures,
                                    candidates.len(),
                                );
                            }
                        }
                    }
                }
                for parser_row_threshold in [1usize, 2, 4, 8, 16] {
                    let eligible_groups = groups
                        .iter()
                        .filter(|(_, futures, candidates)| {
                            futures.len() == 1
                                && candidates.len() <= 64
                                && candidates.first().map_or(0, |candidate| candidate.0)
                                    >= parser_row_threshold
                        })
                        .count();
                    let eligible_states = groups
                        .iter()
                        .filter(|(_, futures, candidates)| {
                            futures.len() == 1
                                && candidates.len() <= 64
                                && candidates.first().map_or(0, |candidate| candidate.0)
                                    >= parser_row_threshold
                        })
                        .map(|(_, _, candidates)| candidates.len())
                        .sum::<usize>();
                    eprintln!(
                        "[glrmask/profile][dynamic_projection_probe_budget] parser_rows_min={} groups={} states={}",
                        parser_row_threshold, eligible_groups, eligible_states,
                    );
                }
                for (group_rank, (total_multiplicity, futures, candidates)) in
                    groups.iter().take(64).enumerate()
                {
                    let (parser_rows, multiplicity, safe64_len, transitions, state, _) =
                        &candidates[0];
                    eprintln!(
                        "[glrmask/profile][dynamic_quotient_projection_group] rank={} state={} parser_rows={} multiplicity={} group_multiplicity={} safe64={} transitions={} futures={:?} candidates={}",
                        group_rank + 1,
                        state,
                        parser_rows,
                        multiplicity,
                        total_multiplicity,
                        safe64_len,
                        transitions,
                        futures,
                        candidates.len(),
                    );
                }
            }
            let max_projections = std::env::var("GLRMASK_DYNAMIC_QUOTIENT_PROJECTIONS")
                .ok()
                .and_then(|value| value.trim().parse::<usize>().ok())
                .filter(|&value| value > 0)
                .unwrap_or(16);
            let mut selected = Vec::with_capacity(max_projections);
            if std::env::var_os("GLRMASK_DYNAMIC_FUTURE_ALIAS_H64").is_some() {
                // H64 aliases make it worthwhile to seed several structurally
                // distinct sources within a large single-future family.  A
                // long bounded-repeat chain often has thousands of states with
                // the same 146-byte row, followed by much smaller 145/144-byte
                // boundary phases.  Picking the first N raw states therefore
                // misses exactly the phases that dominate p85/p90.
                //
                // Keep this strictly a *selection* heuristic: correctness of
                // sharing is established later by the exact H64 terminal-
                // liveness partition.  Here we only ensure that large runtime
                // state families seed one projection for each broad outgoing-
                // row shape before spending slots on tiny signatures.
                let mut active = groups
                    .iter()
                    .enumerate()
                    .filter(|(_, (multiplicity, futures, _))| {
                        *multiplicity >= 1_024 && futures.len() == 1
                    })
                    .map(|(index, _)| index)
                    .collect::<Vec<_>>();
                active.sort_unstable_by_key(|&index| std::cmp::Reverse(groups[index].0));

                let alias_seed_budget = max_projections.saturating_mul(3) / 4;
                let mut seen_transition_counts = vec![BTreeSet::<usize>::new(); groups.len()];
                // One strongest representative per large single-future family.
                for &group_index in &active {
                    if selected.len() >= alias_seed_budget.max(active.len()).min(max_projections) {
                        break;
                    }
                    let (_, _, candidates) = &groups[group_index];
                    let (_, _, _, transitions, state, futures) = &candidates[0];
                    selected.push((*state, futures.clone(), false));
                    seen_transition_counts[group_index].insert(*transitions);
                }
                // Then round-robin over distinct outgoing-row widths.  The H64
                // equivalence check may merge them further, but never aliases
                // states merely because their widths happen to match.
                while selected.len() < alias_seed_budget {
                    let mut added = false;
                    for &group_index in &active {
                        if selected.len() == alias_seed_budget {
                            break;
                        }
                        let (_, _, candidates) = &groups[group_index];
                        let candidate = candidates.iter().find(|candidate| {
                            !seen_transition_counts[group_index].contains(&candidate.3)
                        });
                        let Some((_, _, _, transitions, state, futures)) = candidate else {
                            continue;
                        };
                        seen_transition_counts[group_index].insert(*transitions);
                        selected.push((*state, futures.clone(), false));
                        added = true;
                    }
                    if !added {
                        break;
                    }
                }

                // Preserve some exact-projection coverage for smaller or
                // multi-future signatures as well.
                for (_, _, candidates) in &groups {
                    if selected.len() == max_projections {
                        break;
                    }
                    let (_, _, _, _, state, futures) = &candidates[0];
                    if !selected.iter().any(|(seen, _, _)| seen == state) {
                        selected.push((*state, futures.clone(), false));
                    }
                }
                // Small but highly parser-relevant single-terminal families are
                // cheap enough to probe exhaustively.  Their raw-state count
                // can be tiny even when they dominate runtime token boundaries
                // (for example a short boundary stencil around a huge repeat).
                // Build-time vocabulary projection equivalence will collapse
                // these probes back to a few useful classes; until then, keep
                // every source so the experiment measures the attainable
                // runtime effect without trace-derived state selection.
                for (_, futures, candidates) in &groups {
                    let parser_rows = candidates.first().map_or(0, |candidate| candidate.0);
                    if futures.len() != 1 || parser_rows < 16 || candidates.len() > 64 {
                        continue;
                    }
                    for (_, _, _, _, state, candidate_futures) in candidates {
                        if !selected.iter().any(|(seen, _, _)| seen == state) {
                            selected.push((*state, candidate_futures.clone(), true));
                        }
                    }
                }
                return selected;
            }
            // First pass: one representative from every future signature, in
            // descending exact-state coverage order.
            for (_, _, candidates) in &groups {
                if selected.len() == max_projections {
                    break;
                }
                let (_, _, _, _, state, futures) = &candidates[0];
                selected.push((*state, futures.clone(), false));
            }
            // Fill remaining slots round-robin so one long counter family
            // cannot consume every projection while still allowing important
            // families to receive more than one representative.
            let mut offset = 1usize;
            while selected.len() < max_projections {
                let mut added = false;
                for (_, _, candidates) in &groups {
                    if selected.len() == max_projections {
                        break;
                    }
                    let Some((_, _, _, _, state, futures)) = candidates.get(offset) else {
                        continue;
                    };
                    selected.push((*state, futures.clone(), false));
                    added = true;
                }
                if !added {
                    break;
                }
                offset += 1;
            }
            return selected;
        }

        let max_projections = if tokenizer.num_states() < 50_000 { 2 } else { 3 };
        const MAX_BASE_PROJECTIONS: usize = 2;
        // Trie-aware projections are cheap and useful for small/medium
        // tokenizers, especially broad string states whose UTF-8 paths defeat
        // byte-union certificates. Very large synthesized tokenizers are the
        // opposite: walking the whole vocabulary through their transition
        // product can cost hundreds of milliseconds, while the local bounded
        // continuation certificate handles their counter chains cheaply.
        // Keep both limits structural and overrideable for benchmarking.
        let min_projection_tokenizer_states = std::env::var(
            "GLRMASK_DYNAMIC_SELF_LOOP_PROJECTION_MIN_STATES",
        )
        .ok()
        .and_then(|value| value.trim().parse::<u32>().ok())
        .unwrap_or(128);
        let max_projection_tokenizer_states = std::env::var(
            "GLRMASK_DYNAMIC_SELF_LOOP_PROJECTION_MAX_STATES",
        )
        .ok()
        .and_then(|value| value.trim().parse::<u32>().ok())
        .unwrap_or(8_192);
        if std::env::var_os("GLRMASK_PROFILE_DYNAMIC_PROJECTION_CANDIDATES").is_some() {
            let started = std::time::Instant::now();
            let mut parser_rows_by_terminal = vec![0usize; self.parser_terminal_count() as usize];
            for row in (0..self.parser_symbol_count()).filter_map(|top| self.parser_advance_row(top)) {
                for terminal in row.iter() {
                    if let Some(count) = parser_rows_by_terminal.get_mut(terminal) {
                        *count += 1;
                    }
                }
            }
            let mut ranked = Vec::<(usize, usize, usize, u32, Vec<TerminalID>)>::new();
            for state in 0..tokenizer.num_states() {
                let Some(safe64) = vocab.bounded_observation_safe_bytes(state, 64) else {
                    continue;
                };
                if safe64.len() < 64 {
                    continue;
                }
                let futures = tokenizer
                    .possible_future_terminals_iter(state)
                    .collect::<Vec<_>>();
                if futures.is_empty() || futures.len() > 64 {
                    continue;
                }
                let transitions = tokenizer.transitions_from(state).count();
                let parser_rows = futures
                    .iter()
                    .filter_map(|&terminal| parser_rows_by_terminal.get(terminal as usize).copied())
                    .max()
                    .unwrap_or(0);
                ranked.push((parser_rows, safe64.len(), transitions, state, futures));
            }
            ranked.sort_unstable_by(|left, right| {
                (
                    std::cmp::Reverse(left.0),
                    std::cmp::Reverse(left.1),
                    std::cmp::Reverse(left.2),
                    left.3,
                )
                    .cmp(&(
                        std::cmp::Reverse(right.0),
                        std::cmp::Reverse(right.1),
                        std::cmp::Reverse(right.2),
                        right.3,
                    ))
            });
            eprintln!(
                "[glrmask/profile][dynamic_projection_candidates] broad_states={} scan_ms={:.3}",
                ranked.len(),
                started.elapsed().as_secs_f64() * 1000.0,
            );
            let mut groups = BTreeMap::<(usize, Vec<TerminalID>), usize>::new();
            for (parser_rows, _, _, _, futures) in &ranked {
                *groups.entry((*parser_rows, futures.clone())).or_default() += 1;
            }
            let mut groups = groups.into_iter().collect::<Vec<_>>();
            groups.sort_unstable_by(|left, right| {
                (std::cmp::Reverse((left.0).0), std::cmp::Reverse(left.1), &(left.0).1)
                    .cmp(&(
                        std::cmp::Reverse((right.0).0),
                        std::cmp::Reverse(right.1),
                        &(right.0).1,
                    ))
            });
            for ((parser_rows, futures), count) in groups.into_iter().take(16) {
                eprintln!(
                    "[glrmask/profile][dynamic_projection_candidate_group] parser_rows={} count={} futures={:?}",
                    parser_rows,
                    count,
                    futures,
                );
            }
            for (rank, (parser_rows, safe64, transitions, state, futures)) in
                ranked.iter().take(64).enumerate()
            {
                eprintln!(
                    "[glrmask/profile][dynamic_projection_candidate] rank={} state={} parser_rows={} safe64={} transitions={} futures={:?}",
                    rank + 1,
                    state,
                    parser_rows,
                    safe64,
                    transitions,
                    futures,
                );
            }
        }
        if tokenizer.num_states() < min_projection_tokenizer_states
            || tokenizer.num_states() > max_projection_tokenizer_states
        {
            return Vec::new();
        }
        // Keep at most one strong source per exact future-terminal signature.
        // This avoids spending projection slots on long counter chains whose
        // states are lexer-observationally equivalent for our purpose.
        let mut best_by_futures = BTreeMap::<Vec<TerminalID>, (usize, u32)>::new();
        for state in 0..tokenizer.num_states() {
            if tokenizer.transitions_from(state).count() < 100 {
                continue;
            }
            let futures = tokenizer
                .possible_future_terminals_iter(state)
                .collect::<Vec<_>>();
            if futures.is_empty() {
                continue;
            }
            let loop_len = tokenizer.self_loop_bytes(state).len();
            if loop_len < 64 {
                continue;
            }
            let entry = best_by_futures.entry(futures).or_insert((loop_len, state));
            if (loop_len, std::cmp::Reverse(state))
                > (entry.0, std::cmp::Reverse(entry.1))
            {
                *entry = (loop_len, state);
            }
        }
        let mut ranked = best_by_futures
            .into_iter()
            .map(|(futures, (loop_len, state))| (loop_len, state, futures))
            .collect::<Vec<_>>();
        ranked.sort_unstable_by(|left, right| {
            (std::cmp::Reverse(left.0), left.1, &left.2)
                .cmp(&(std::cmp::Reverse(right.0), right.1, &right.2))
        });
        ranked.truncate(MAX_BASE_PROJECTIONS);
        let mut selected = ranked
            .into_iter()
            .map(|(_, state, futures)| (state, futures, false))
            .collect::<Vec<_>>();

        if selected.len() < max_projections {
            let selected_targets = selected
                .iter()
                .map(|(state, futures, _)| (*state, futures.clone()))
                .collect::<FxHashMap<u32, Vec<TerminalID>>>();
            let root_byte_sizes = vocab
                .trie
                .children(0)
                .iter()
                .filter_map(|edge| {
                    vocab
                        .trie
                        .edge_bytes(edge)
                        .first()
                        .copied()
                        .map(|byte| (byte, vocab.trie.subtree_tokens(edge.child).len()))
                })
                .collect::<FxHashMap<u8, usize>>();
            let mut predecessors = Vec::<(usize, u32, Vec<TerminalID>)>::new();
            for state in 0..tokenizer.num_states() {
                if selected
                    .iter()
                    .any(|&(selected_state, _, _)| selected_state == state)
                {
                    continue;
                }
                let futures = tokenizer
                    .possible_future_terminals_iter(state)
                    .collect::<Vec<_>>();
                if futures.is_empty() {
                    continue;
                }
                let mut transitions = tokenizer.transitions_from(state);
                let Some((byte, target)) = transitions.next() else {
                    continue;
                };
                if transitions.next().is_some()
                    || selected_targets.get(&target) != Some(&futures)
                {
                    continue;
                }
                let score = root_byte_sizes.get(&byte).copied().unwrap_or(0);
                predecessors.push((score, state, futures));
            }
            predecessors.sort_unstable_by(|left, right| {
                (std::cmp::Reverse(left.0), left.1, &left.2)
                    .cmp(&(std::cmp::Reverse(right.0), right.1, &right.2))
            });
            for (_, state, futures) in predecessors {
                selected.push((state, futures, false));
                if selected.len() == max_projections {
                    break;
                }
            }
        }
        selected
    }

    fn build_dynamic_self_loop_projections(
        &self,
        vocab: &DynamicMaskVocab,
        full_fast_transitions: &FastTokenizerTransitions,
    ) -> (Vec<DynamicSelfLoopProjection>, Vec<u32>) {
        fn build_one(
            constraint: &Constraint,
            tokenizer: &Tokenizer,
            vocab: &DynamicMaskVocab,
            fast_transitions: &FastTokenizerTransitions,
            source_state: u32,
            source_future_terminals: &[TerminalID],
        ) -> DynamicSelfLoopProjection {
            let trie = vocab.trie.as_ref();
            let mut safe_subtrees = vec![0u8; trie.node_count()];
            let mut source_reentry_safe_subtrees = vec![0u8; trie.node_count()];
            let mut common_future_masks = vec![0u64; trie.node_count()];
            let mut safe_no_match_mask = vec![0u32; constraint.body_mask_len()];
            fn advance_no_match_deterministic(
                tokenizer: &Tokenizer,
                fast_transitions: &FastTokenizerTransitions,
                mut state: u32,
                bytes: &[u8],
            ) -> Option<u32> {
                if tokenizer.state_has_epsilon_transitions(state) {
                    return None;
                }
                for &byte in bytes {
                    let target = fast_transitions.transition(tokenizer, state, byte);
                    if target == u32::MAX {
                        return None;
                    }
                    if tokenizer.state_has_epsilon_transitions(target) {
                        return None;
                    }
                    state = target;
                }
                Some(state)
            }

            fn mark_token_ids(mask: &mut [u32], token_ids: &[u32]) {
                for &token_id in token_ids {
                    let word = token_id as usize / 32;
                    if let Some(mask_word) = mask.get_mut(word) {
                        *mask_word |= 1u32 << (token_id % 32);
                    }
                }
            }

            fn visit(
                constraint: &Constraint,
                tokenizer: &Tokenizer,
                vocab: &DynamicMaskVocab,
                fast_transitions: &FastTokenizerTransitions,
                trie: &DynamicMaskTrie,
                node: u32,
                tokenizer_state: u32,
                source_state: u32,
                source_loop_bytes: U8Set,
                source_future_terminals: &[TerminalID],
                safe_subtrees: &mut [u8],
                source_reentry_safe_subtrees: &mut [u8],
                safe_no_match_mask: &mut [u32],
            ) -> bool {
                let state_remains_live = tokenizer
                    .possible_future_terminals_iter(tokenizer_state)
                    .eq(source_future_terminals.iter().copied());
                let node_data = trie.node(node);
                let token_safe = node_data.token_id.is_none() || state_remains_live;
                if state_remains_live {
                    if tokenizer_state == source_state
                        && U8Set::from_words(trie.subtree_bytes(node))
                            .is_subset(&source_loop_bytes)
                    {
                        mark_token_ids(safe_no_match_mask, vocab.subtree_original_tokens(node));
                        safe_subtrees[node as usize] = 1;
                        source_reentry_safe_subtrees[node as usize] = 1;
                        return true;
                    }
                    if let Some(canonical_token) = node_data.token_id
                        && let Some(token_ids) = vocab.token_ids(canonical_token)
                    {
                        mark_token_ids(safe_no_match_mask, token_ids);
                    }
                }

                let mut all_children_safe = true;
                for edge in trie.children(node) {
                    let child_safe = advance_no_match_deterministic(
                        tokenizer,
                        fast_transitions,
                        tokenizer_state,
                        trie.edge_bytes(edge),
                    )
                    .is_some_and(|end_state| {
                        visit(
                            constraint,
                            tokenizer,
                            vocab,
                            fast_transitions,
                            trie,
                            edge.child,
                            end_state,
                            source_state,
                            source_loop_bytes,
                            source_future_terminals,
                            safe_subtrees,
                            source_reentry_safe_subtrees,
                            safe_no_match_mask,
                        )
                    });
                    all_children_safe &= child_safe;
                }
                let subtree_safe = token_safe && all_children_safe;
                safe_subtrees[node as usize] = u8::from(subtree_safe);
                if tokenizer_state == source_state && subtree_safe {
                    source_reentry_safe_subtrees[node as usize] = 1;
                }
                subtree_safe
            }

            fn visit_common_futures(
                tokenizer: &Tokenizer,
                vocab: &DynamicMaskVocab,
                fast_transitions: &FastTokenizerTransitions,
                trie: &DynamicMaskTrie,
                node: u32,
                tokenizer_state: u32,
                source_future_terminals: &[TerminalID],
                common_future_masks: &mut [u64],
            ) -> u64 {
                if source_future_terminals.is_empty() || source_future_terminals.len() > 64 {
                    return 0;
                }
                let all = if source_future_terminals.len() == 64 {
                    u64::MAX
                } else {
                    (1u64 << source_future_terminals.len()) - 1
                };
                let state_future_mask = || {
                    let futures = tokenizer.possible_future_terminals(tokenizer_state);
                    source_future_terminals
                        .iter()
                        .enumerate()
                        .fold(0u64, |mask, (index, &terminal)| {
                            mask | (u64::from(futures.contains(terminal as usize)) << index)
                        })
                };

                let mut common = if trie.node(node).token_id.is_some() {
                    state_future_mask()
                } else {
                    all
                };
                for edge in trie.children(node) {
                    let child_common = advance_no_match_deterministic(
                        tokenizer,
                        fast_transitions,
                        tokenizer_state,
                        trie.edge_bytes(edge),
                    )
                    .map_or(0, |end_state| {
                        visit_common_futures(
                            tokenizer,
                            vocab,
                            fast_transitions,
                            trie,
                            edge.child,
                            end_state,
                            source_future_terminals,
                            common_future_masks,
                        )
                    });
                    common &= child_common;
                    if common == 0 {
                        // Descendants still need their own rows for runtime
                        // subtree checks, so do not prune the traversal here.
                    }
                }
                common_future_masks[node as usize] = common;
                let _ = vocab;
                common
            }

            let source_loop_bytes = tokenizer.self_loop_bytes(source_state);
            visit(
                constraint,
                tokenizer,
                vocab,
                fast_transitions,
                trie,
                0,
                source_state,
                source_state,
                source_loop_bytes,
                source_future_terminals,
                &mut safe_subtrees,
                &mut source_reentry_safe_subtrees,
                &mut safe_no_match_mask,
            );
            visit_common_futures(
                tokenizer,
                vocab,
                fast_transitions,
                trie,
                0,
                source_state,
                source_future_terminals,
                &mut common_future_masks,
            );
            fn visit_candidate_frontier(
                tokenizer: &Tokenizer,
                fast_transitions: &FastTokenizerTransitions,
                trie: &DynamicMaskTrie,
                node: u32,
                tokenizer_state: u32,
                dead_nodes: &mut Vec<u32>,
                frontier_nodes: &mut Vec<u32>,
                live_nodes: &mut usize,
                match_frontier_tokens: &mut usize,
                reset_child_states: &mut FxHashMap<u32, (usize, usize)>,
                reset_child_configs: &mut FxHashMap<Vec<u32>, (usize, usize)>,
                reset_config_frontiers: &mut Vec<(u32, Vec<u32>)>,
                reset_edge_second_matches: &mut usize,
                reset_second_match_tokens: &mut usize,
                reset_edge_unknown: &mut usize,
                reset_unknown_tokens: &mut usize,
            ) -> bool {
                if tokenizer.state_has_epsilon_transitions(tokenizer_state) {
                    *live_nodes += 1;
                    return true;
                }
                let node_data = trie.node(node);
                let mut any = node_data.token_id.is_some()
                    && (tokenizer
                        .possible_future_terminals(tokenizer_state)
                        .iter()
                        .next()
                        .is_some()
                        || tokenizer
                            .matched_terminals_iter(tokenizer_state)
                            .next()
                            .is_some());
                for edge in trie.children(node) {
                    let mut state = tokenizer_state;
                    let mut alive = true;
                    let mut matched = false;
                    let mut unknown = false;
                    let edge_bytes = trie.edge_bytes(edge);
                    let mut match_offset = None::<usize>;
                    for (offset, &byte) in edge_bytes.iter().enumerate() {
                        let target = fast_transitions.transition(tokenizer, state, byte);
                        if target == u32::MAX {
                            alive = false;
                            break;
                        }
                        if tokenizer.state_has_epsilon_transitions(target) {
                            unknown = true;
                            break;
                        }
                        state = target;
                        if tokenizer.matched_terminals_iter(state).next().is_some() {
                            matched = true;
                            match_offset = Some(offset);
                            break;
                        }
                    }
                    let child_any = if matched || unknown {
                        frontier_nodes.push(edge.child);
                        if matched {
                            *match_frontier_tokens += trie.subtree_tokens(edge.child).len();
                            if let Some(match_offset) = match_offset {
                                let reset_execution = tokenizer.execute_from_state_all_widths(
                                    &edge_bytes[match_offset + 1..],
                                    tokenizer.start_state(),
                                );
                                if !reset_execution.matches.is_empty() {
                                    *reset_edge_second_matches += 1;
                                    *reset_second_match_tokens +=
                                        trie.subtree_tokens(edge.child).len();
                                } else if let [reset_state] = reset_execution.end_state.as_slice() {
                                    let entry = reset_child_states.entry(*reset_state).or_default();
                                    entry.0 += 1;
                                    entry.1 += trie.subtree_tokens(edge.child).len();
                                } else if !reset_execution.end_state.is_empty() {
                                    let mut config = reset_execution.end_state.to_vec();
                                    config.sort_unstable();
                                    config.dedup();
                                    reset_config_frontiers.push((edge.child, config.clone()));
                                    let entry = reset_child_configs.entry(config).or_default();
                                    entry.0 += 1;
                                    entry.1 += trie.subtree_tokens(edge.child).len();
                                } else {
                                    *reset_edge_unknown += 1;
                                    *reset_unknown_tokens +=
                                        trie.subtree_tokens(edge.child).len();
                                }
                            } else {
                                *reset_edge_unknown += 1;
                                *reset_unknown_tokens += trie.subtree_tokens(edge.child).len();
                            }
                        }
                        true
                    } else if alive {
                        let child_any = visit_candidate_frontier(
                            tokenizer,
                            fast_transitions,
                            trie,
                            edge.child,
                            state,
                            dead_nodes,
                            frontier_nodes,
                            live_nodes,
                            match_frontier_tokens,
                            reset_child_states,
                            reset_child_configs,
                            reset_config_frontiers,
                            reset_edge_second_matches,
                            reset_second_match_tokens,
                            reset_edge_unknown,
                            reset_unknown_tokens,
                        );
                        if !child_any {
                            dead_nodes.push(edge.child);
                        }
                        child_any
                    } else {
                        dead_nodes.push(edge.child);
                        false
                    };
                    any |= child_any;
                }
                if any {
                    *live_nodes += 1;
                }
                any
            }
            let mut pre_match_dead_nodes = Vec::<u32>::new();
            let mut pre_match_frontier_nodes = Vec::<u32>::new();
            let mut live_nodes = 0usize;
            let mut match_frontier_tokens = 0usize;
            let mut reset_child_states = FxHashMap::<u32, (usize, usize)>::default();
            let mut reset_child_configs = FxHashMap::<Vec<u32>, (usize, usize)>::default();
            let mut reset_config_frontiers = Vec::<(u32, Vec<u32>)>::new();
            let mut reset_edge_second_matches = 0usize;
            let mut reset_second_match_tokens = 0usize;
            let mut reset_edge_unknown = 0usize;
            let mut reset_unknown_tokens = 0usize;
            let _ = visit_candidate_frontier(
                tokenizer,
                fast_transitions,
                trie,
                0,
                source_state,
                &mut pre_match_dead_nodes,
                &mut pre_match_frontier_nodes,
                &mut live_nodes,
                &mut match_frontier_tokens,
                &mut reset_child_states,
                &mut reset_child_configs,
                &mut reset_config_frontiers,
                &mut reset_edge_second_matches,
                &mut reset_second_match_tokens,
                &mut reset_edge_unknown,
                &mut reset_unknown_tokens,
            );
            pre_match_dead_nodes.sort_unstable();
            pre_match_dead_nodes.dedup();
            pre_match_frontier_nodes.sort_unstable();
            pre_match_frontier_nodes.dedup();
            let pack_nodes = |nodes: &[u32]| {
                let mut words = vec![0u64; trie.node_count().div_ceil(64)];
                for &node in nodes {
                    let word = node as usize >> 6;
                    let bit = node & 63;
                    words[word] |= 1u64 << bit;
                }
                words
            };
            let pre_match_dead_words = pack_nodes(&pre_match_dead_nodes);
            let pre_match_frontier_words = pack_nodes(&pre_match_frontier_nodes);

            let config_certificate_requested = std::env::var(
                "GLRMASK_DYNAMIC_CONFIG_SUBTREE_CERT_STATES",
            )
            .ok()
            .is_some_and(|value| {
                value
                    .split(',')
                    .filter_map(|value| value.trim().parse::<u32>().ok())
                    .map(|state| vocab.mask_projection_state(state))
                    .any(|state| state == source_state)
            });
            let mut config_subtree_certificates = Vec::<DynamicConfigSubtreeCertificate>::new();
            if config_certificate_requested
                && source_future_terminals.len() == 1
                && match_frontier_tokens >= 4_096
                && let Some((dominant_config, &(_, dominant_tokens))) = reset_child_configs
                    .iter()
                    .max_by_key(|entry| (entry.1).1)
                && dominant_config.len() <= 16
                && dominant_tokens.saturating_mul(4)
                    >= match_frontier_tokens.saturating_mul(3)
            {
                fn config_futures_for_certificate(
                    tokenizer: &Tokenizer,
                    config: &[u32],
                ) -> BitSet {
                    let mut futures = BitSet::new(tokenizer.num_terminals() as usize);
                    for &state in config {
                        futures.union_with_prefix(tokenizer.possible_future_terminals(state));
                    }
                    futures
                }

                fn collect_config_certificates(
                    tokenizer: &Tokenizer,
                    trie: &DynamicMaskTrie,
                    node: u32,
                    config: &[u32],
                ) -> (BitSet, Vec<DynamicConfigSubtreeCertificate>) {
                    let mut common = if trie.node(node).token_id.is_some() {
                        config_futures_for_certificate(tokenizer, config)
                    } else {
                        BitSet::all(tokenizer.num_terminals() as usize)
                    };
                    let mut descendant_certificates = Vec::<DynamicConfigSubtreeCertificate>::new();
                    for edge in trie.children(node) {
                        let mut next = TokenizerStateSet::from_iter(config.iter().copied());
                        let mut matched = false;
                        for &byte in trie.edge_bytes(edge) {
                            next = tokenizer.step_all(next.as_slice(), byte);
                            if next.is_empty() {
                                break;
                            }
                            if next.iter().any(|&state| {
                                tokenizer.matched_terminals_iter(state).next().is_some()
                            }) {
                                matched = true;
                                break;
                            }
                        }
                        if matched || next.is_empty() {
                            common.clear_all();
                            continue;
                        }
                        let (child_common, mut child_certificates) = collect_config_certificates(
                            tokenizer,
                            trie,
                            edge.child,
                            next.as_slice(),
                        );
                        common.intersect_with(&child_common);
                        descendant_certificates.append(&mut child_certificates);
                    }
                    if !common.is_empty() {
                        let mut projected_config = config.to_vec();
                        projected_config.sort_unstable();
                        projected_config.dedup();
                        let future_terminals = common
                            .iter_ones()
                            .map(|terminal| terminal as TerminalID)
                            .collect::<Vec<_>>();
                        // Keep the current maximal certificate, but do not
                        // throw away more-specific descendants whose common
                        // future set has grown.  Runtime may reject the broad
                        // parent set for the current parser stack while a
                        // narrower vocabulary region exposes an additional
                        // admissible terminal.  Descendants with the *same*
                        // terminal set are redundant: if the parent is
                        // rejected they are rejected too, and if it is
                        // accepted the whole parent subtree is already done.
                        descendant_certificates.retain(|certificate| {
                            certificate.common_future_terminals.as_ref()
                                != future_terminals.as_slice()
                        });
                        descendant_certificates.push(DynamicConfigSubtreeCertificate {
                            node,
                            projected_config: Arc::from(projected_config),
                            common_future_terminals: Arc::from(future_terminals),
                        });
                        return (
                            common,
                            descendant_certificates,
                        );
                    }
                    (common, descendant_certificates)
                }

                for (node, config) in &reset_config_frontiers {
                    if config != dominant_config {
                        continue;
                    }
                    let (_, mut certificates) = collect_config_certificates(
                        tokenizer,
                        trie,
                        *node,
                        dominant_config,
                    );
                    config_subtree_certificates.append(&mut certificates);
                }
                config_subtree_certificates.sort_unstable_by(|left, right| {
                    left.node
                        .cmp(&right.node)
                        .then_with(|| left.projected_config.as_ref().cmp(right.projected_config.as_ref()))
                });
                config_subtree_certificates.dedup_by(|left, right| {
                    left.node == right.node
                        && left.projected_config.as_ref() == right.projected_config.as_ref()
                        && left.common_future_terminals.as_ref()
                            == right.common_future_terminals.as_ref()
                });
                if std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
                    || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
                {
                    let covered = config_subtree_certificates
                        .iter()
                        .map(|certificate| trie.subtree_tokens(certificate.node).len())
                        .sum::<usize>();
                    eprintln!(
                        "[glrmask/profile][dynamic_config_subtree_certificates] source_state={} dominant_config_states={} dominant_tokens={} certificates={} covered_tokens={}",
                        source_state,
                        dominant_config.len(),
                        dominant_tokens,
                        config_subtree_certificates.len(),
                        covered,
                    );
                }
            }
            if std::env::var_os("GLRMASK_PROFILE_DYNAMIC_PROJECTION_CANDIDATES").is_some() {
                eprintln!(
                    "[glrmask/profile][dynamic_projection_candidate_frontier] source_state={} futures={:?} live_nodes={} dead_subtrees={} frontier_nodes={} match_frontier_tokens={} reset_child_states={} reset_child_configs={} reset_edge_second_matches={} reset_second_match_tokens={} reset_edge_unknown={} reset_unknown_tokens={} reset_child_top={:?} reset_config_top={:?}",
                    source_state,
                    source_future_terminals,
                    live_nodes,
                    pre_match_dead_nodes.len(),
                    pre_match_frontier_nodes.len(),
                    match_frontier_tokens,
                    reset_child_states.len(),
                    reset_child_configs.len(),
                    reset_edge_second_matches,
                    reset_second_match_tokens,
                    reset_edge_unknown,
                    reset_unknown_tokens,
                    {
                        let mut rows = reset_child_states
                            .iter()
                            .map(|(&state, &(nodes, tokens))| (tokens, nodes, state))
                            .collect::<Vec<_>>();
                        rows.sort_unstable_by(|left, right| right.cmp(left));
                        rows.truncate(12);
                        rows
                    },
                    {
                        let mut rows = reset_child_configs
                            .iter()
                            .map(|(config, &(nodes, tokens))| (tokens, nodes, config.len()))
                            .collect::<Vec<_>>();
                        rows.sort_unstable_by(|left, right| right.cmp(left));
                        rows.truncate(12);
                        rows
                    },
                );

                if let Some((dominant_config, &(_, dominant_tokens))) = reset_child_configs
                    .iter()
                    .max_by_key(|entry| (entry.1).1)
                {
                    fn config_futures(tokenizer: &Tokenizer, config: &[u32]) -> BitSet {
                        let mut futures = BitSet::new(tokenizer.num_terminals() as usize);
                        for &state in config {
                            futures.union_with_prefix(tokenizer.possible_future_terminals(state));
                        }
                        futures
                    }

                    fn project_config_common_futures(
                        tokenizer: &Tokenizer,
                        trie: &DynamicMaskTrie,
                        node: u32,
                        config: &[u32],
                    ) -> (BitSet, usize, usize, usize) {
                        let all = BitSet::all(tokenizer.num_terminals() as usize);
                        let mut common = if trie.node(node).token_id.is_some() {
                            config_futures(tokenizer, config)
                        } else {
                            all
                        };
                        let mut descendant_covered = 0usize;
                        let mut descendant_safe_nodes = 0usize;
                        let mut second_match_tokens = 0usize;
                        for edge in trie.children(node) {
                            let mut next = TokenizerStateSet::from_iter(config.iter().copied());
                            let mut matched = false;
                            for &byte in trie.edge_bytes(edge) {
                                next = tokenizer.step_all(next.as_slice(), byte);
                                if next.is_empty() {
                                    break;
                                }
                                if next.iter().any(|&state| {
                                    tokenizer.matched_terminals_iter(state).next().is_some()
                                }) {
                                    matched = true;
                                    break;
                                }
                            }
                            if matched {
                                common.clear_all();
                                second_match_tokens += trie.subtree_tokens(edge.child).len();
                                continue;
                            }
                            if next.is_empty() {
                                common.clear_all();
                                continue;
                            }
                            let (child_common, child_covered, child_safe, child_second) =
                                project_config_common_futures(
                                    tokenizer,
                                    trie,
                                    edge.child,
                                    next.as_slice(),
                                );
                            common.intersect_with(&child_common);
                            descendant_covered += child_covered;
                            descendant_safe_nodes += child_safe;
                            second_match_tokens += child_second;
                        }
                        if !common.is_empty() {
                            (common, trie.subtree_tokens(node).len(), 1, second_match_tokens)
                        } else {
                            (
                                common,
                                descendant_covered,
                                descendant_safe_nodes,
                                second_match_tokens,
                            )
                        }
                    }

                    let mut covered = 0usize;
                    let mut safe_nodes = 0usize;
                    let mut second_match_tokens = 0usize;
                    let mut root_common_sizes = Vec::<usize>::new();
                    let mut roots = 0usize;
                    for (node, config) in &reset_config_frontiers {
                        if config != dominant_config {
                            continue;
                        }
                        roots += 1;
                        let (common, root_covered, root_safe, root_second) =
                            project_config_common_futures(
                                tokenizer,
                                trie,
                                *node,
                                dominant_config,
                            );
                        root_common_sizes.push(common.count_ones());
                        covered += root_covered;
                        safe_nodes += root_safe;
                        second_match_tokens += root_second;
                    }
                    root_common_sizes.sort_unstable_by(|a, b| b.cmp(a));
                    eprintln!(
                        "[glrmask/profile][dynamic_reset_config_projection] source_state={} config_states={} frontier_roots={} dominant_tokens={} common_future_covered={} safe_nodes={} second_match_tokens={} root_common_sizes={:?}",
                        source_state,
                        dominant_config.len(),
                        roots,
                        dominant_tokens,
                        covered,
                        safe_nodes,
                        second_match_tokens,
                        root_common_sizes,
                    );

                    // The common-intersection certificate above is sufficient
                    // but often much stronger than necessary.  For a fixed
                    // parser admissible-terminal set A, a no-finalization
                    // subtree is valid exactly when every token leaf's future
                    // set F intersects A.  Represent that monotone condition
                    // as the inclusion-minimal antichain of leaf future sets:
                    // supersets add no constraint.  Measure how compact that
                    // CNF is before committing to a runtime representation.
                    let profile_cnf = std::env::var(
                        "GLRMASK_PROFILE_DYNAMIC_FUTURE_QUOTIENT_STATES",
                    )
                    .ok()
                    .is_some_and(|value| {
                        value
                            .split(',')
                            .filter_map(|value| value.trim().parse::<u32>().ok())
                            .map(|state| vocab.mask_projection_state(state))
                            .any(|state| state == source_state)
                    });
                    if profile_cnf {
                        const THRESHOLDS: [usize; 6] = [1, 2, 4, 8, 16, 32];

                        fn insert_minimal_clause(clauses: &mut Vec<BitSet>, clause: BitSet) {
                            if clause.is_empty()
                                || clauses.iter().any(|existing| existing.is_subset(&clause))
                            {
                                return;
                            }
                            clauses.retain(|existing| !clause.is_subset(existing));
                            clauses.push(clause);
                        }

                        struct CnfAnalysis {
                            complete: bool,
                            clauses: Vec<BitSet>,
                            covered: [usize; THRESHOLDS.len()],
                            certificates: [usize; THRESHOLDS.len()],
                            complete_clause_histogram: BTreeMap<usize, usize>,
                            complete_token_histogram: BTreeMap<usize, usize>,
                            max_clause_count: usize,
                        }

                        fn analyze_config_cnf(
                            tokenizer: &Tokenizer,
                            trie: &DynamicMaskTrie,
                            node: u32,
                            config: &[u32],
                        ) -> CnfAnalysis {
                            let mut complete = true;
                            let mut clauses = Vec::<BitSet>::new();
                            let mut covered = [0usize; THRESHOLDS.len()];
                            let mut certificates = [0usize; THRESHOLDS.len()];
                            let mut complete_clause_histogram = BTreeMap::<usize, usize>::new();
                            let mut complete_token_histogram = BTreeMap::<usize, usize>::new();
                            let mut max_clause_count = 0usize;

                            if trie.node(node).token_id.is_some() {
                                let future = config_futures(tokenizer, config);
                                if future.is_empty() {
                                    complete = false;
                                } else {
                                    insert_minimal_clause(&mut clauses, future);
                                }
                            }

                            let mut child_analyses = Vec::<CnfAnalysis>::new();
                            for edge in trie.children(node) {
                                let mut next = TokenizerStateSet::from_iter(config.iter().copied());
                                let mut matched = false;
                                for &byte in trie.edge_bytes(edge) {
                                    next = tokenizer.step_all(next.as_slice(), byte);
                                    if next.is_empty() {
                                        break;
                                    }
                                    if next.iter().any(|&state| {
                                        tokenizer.matched_terminals_iter(state).next().is_some()
                                    }) {
                                        matched = true;
                                        break;
                                    }
                                }
                                if matched || next.is_empty() {
                                    complete = false;
                                    continue;
                                }
                                let child = analyze_config_cnf(
                                    tokenizer,
                                    trie,
                                    edge.child,
                                    next.as_slice(),
                                );
                                if !child.complete {
                                    complete = false;
                                }
                                for clause in child.clauses.iter().cloned() {
                                    insert_minimal_clause(&mut clauses, clause);
                                }
                                max_clause_count = max_clause_count.max(child.max_clause_count);
                                for (&count, &nodes) in &child.complete_clause_histogram {
                                    *complete_clause_histogram.entry(count).or_default() += nodes;
                                }
                                for (&count, &tokens) in &child.complete_token_histogram {
                                    *complete_token_histogram.entry(count).or_default() += tokens;
                                }
                                child_analyses.push(child);
                            }

                            if complete {
                                let clause_count = clauses.len();
                                let tokens = trie.subtree_tokens(node).len();
                                max_clause_count = max_clause_count.max(clause_count);
                                *complete_clause_histogram.entry(clause_count).or_default() += 1;
                                *complete_token_histogram.entry(clause_count).or_default() += tokens;
                                for (index, threshold) in THRESHOLDS.iter().copied().enumerate() {
                                    if clause_count <= threshold {
                                        covered[index] = tokens;
                                        certificates[index] = 1;
                                    } else {
                                        covered[index] = child_analyses
                                            .iter()
                                            .map(|child| child.covered[index])
                                            .sum();
                                        certificates[index] = child_analyses
                                            .iter()
                                            .map(|child| child.certificates[index])
                                            .sum();
                                    }
                                }
                            } else {
                                for index in 0..THRESHOLDS.len() {
                                    covered[index] = child_analyses
                                        .iter()
                                        .map(|child| child.covered[index])
                                        .sum();
                                    certificates[index] = child_analyses
                                        .iter()
                                        .map(|child| child.certificates[index])
                                        .sum();
                                }
                            }

                            CnfAnalysis {
                                complete,
                                clauses,
                                covered,
                                certificates,
                                complete_clause_histogram,
                                complete_token_histogram,
                                max_clause_count,
                            }
                        }

                        let mut total_covered = [0usize; THRESHOLDS.len()];
                        let mut total_certificates = [0usize; THRESHOLDS.len()];
                        let mut clause_histogram = BTreeMap::<usize, usize>::new();
                        let mut token_histogram = BTreeMap::<usize, usize>::new();
                        let mut max_clause_count = 0usize;
                        let mut analyzed_roots = 0usize;
                        for (node, config) in &reset_config_frontiers {
                            if config != dominant_config {
                                continue;
                            }
                            analyzed_roots += 1;
                            let analysis = analyze_config_cnf(
                                tokenizer,
                                trie,
                                *node,
                                dominant_config,
                            );
                            max_clause_count = max_clause_count.max(analysis.max_clause_count);
                            for index in 0..THRESHOLDS.len() {
                                total_covered[index] += analysis.covered[index];
                                total_certificates[index] += analysis.certificates[index];
                            }
                            for (count, nodes) in analysis.complete_clause_histogram {
                                *clause_histogram.entry(count).or_default() += nodes;
                            }
                            for (count, tokens) in analysis.complete_token_histogram {
                                *token_histogram.entry(count).or_default() += tokens;
                            }
                        }
                        let coverage = THRESHOLDS
                            .iter()
                            .enumerate()
                            .map(|(index, &threshold)| {
                                (threshold, total_covered[index], total_certificates[index])
                            })
                            .collect::<Vec<_>>();
                        let histogram = clause_histogram
                            .iter()
                            .take(24)
                            .map(|(&clauses, &nodes)| {
                                (
                                    clauses,
                                    nodes,
                                    token_histogram.get(&clauses).copied().unwrap_or(0),
                                )
                            })
                            .collect::<Vec<_>>();
                        eprintln!(
                            "[glrmask/profile][dynamic_reset_config_cnf] source_state={} config_states={} roots={} dominant_tokens={} coverage={:?} max_clause_count={} clause_histogram={:?}",
                            source_state,
                            dominant_config.len(),
                            analyzed_roots,
                            dominant_tokens,
                            coverage,
                            max_clause_count,
                            histogram,
                        );
                    }
                }
                if let [terminal] = source_future_terminals
                    && std::env::var("GLRMASK_PROFILE_DYNAMIC_FUTURE_QUOTIENT_STATES")
                        .ok()
                        .is_some_and(|value| {
                            value
                                .split(',')
                                .filter_map(|value| value.trim().parse::<u32>().ok())
                                .map(|state| vocab.mask_projection_state(state))
                                .any(|state| state == source_state)
                        })
                {
                    let vocab_bytes = constraint
                        .token_bytes
                        .values()
                        .map(Vec::as_slice)
                        .collect::<BTreeSet<_>>();
                    let mut matched_tokens = 0usize;
                    let mut continuation_live = 0usize;
                    let mut matched_and_continuation_live = 0usize;
                    let mut empty_suffix = 0usize;
                    let mut suffix_is_vocab = 0usize;
                    let mut any_match_suffix_is_vocab = 0usize;
                    let mut all_match_suffix_pairs = 0usize;
                    let mut suffix_counts = FxHashMap::<Vec<u8>, usize>::default();
                    let mut full_match_states = FxHashMap::<u32, usize>::default();
                    for bytes in constraint.token_bytes.values() {
                        let execution = tokenizer.execute_from_state_all_widths(bytes, source_state);
                        let first_match_width = execution
                            .matches
                            .iter()
                            .filter(|matched| matched.id == *terminal)
                            .map(|matched| matched.width)
                            .min();
                        let live = execution.end_state.iter().any(|&end_state| {
                            tokenizer
                                .possible_future_terminals(end_state)
                                .contains(*terminal as usize)
                        });
                        continuation_live += usize::from(live);
                        if let Some(width) = first_match_width {
                            matched_tokens += 1;
                            matched_and_continuation_live += usize::from(live);
                            let suffix = &bytes[width..];
                            empty_suffix += usize::from(suffix.is_empty());
                            suffix_is_vocab += usize::from(vocab_bytes.contains(suffix));
                            *suffix_counts.entry(suffix.to_vec()).or_default() += 1;
                        }
                        let mut token_has_vocab_suffix = false;
                        for matched in execution.matches.iter().filter(|matched| matched.id == *terminal) {
                            let suffix = &bytes[matched.width..];
                            if !suffix.is_empty() && vocab_bytes.contains(suffix) {
                                all_match_suffix_pairs += 1;
                                token_has_vocab_suffix = true;
                            }
                        }
                        any_match_suffix_is_vocab += usize::from(token_has_vocab_suffix);
                    }
                    if source_state < constraint.tokenizer.num_states()
                        && constraint
                            .tokenizer
                            .possible_future_terminals(source_state)
                            .contains(*terminal as usize)
                    {
                        for bytes in constraint.token_bytes.values() {
                            let execution = constraint
                                .tokenizer
                                .execute_from_state_all_widths(bytes, source_state);
                            let Some(first_width) = execution
                                .matches
                                .iter()
                                .filter(|matched| matched.id == *terminal)
                                .map(|matched| matched.width)
                                .min()
                            else {
                                continue;
                            };
                            for matched in execution.matches.iter().filter(|matched| {
                                matched.id == *terminal && matched.width == first_width
                            }) {
                                *full_match_states.entry(matched.end_state).or_default() += 1;
                            }
                        }
                    }
                    let mut top_suffixes = suffix_counts
                        .iter()
                        .map(|(suffix, &count)| (count, suffix.len()))
                        .collect::<Vec<_>>();
                    top_suffixes.sort_unstable_by(|left, right| right.cmp(left));
                    top_suffixes.truncate(12);
                    eprintln!(
                        "[glrmask/profile][dynamic_projection_first_match_suffixes] source_state={} terminal={} matched_tokens={} unique_suffixes={} continuation_live={} matched_and_continuation_live={} empty_suffix={} suffix_is_vocab={} any_match_suffix_is_vocab={} all_match_suffix_pairs={} full_match_states={} top_count_len={:?}",
                        source_state,
                        terminal,
                        matched_tokens,
                        suffix_counts.len(),
                        continuation_live,
                        matched_and_continuation_live,
                        empty_suffix,
                        suffix_is_vocab,
                        any_match_suffix_is_vocab,
                        all_match_suffix_pairs,
                        full_match_states.len(),
                        top_suffixes,
                    );
                }
            }
            if std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
                || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
            {
                let safe_tokens = safe_no_match_mask
                    .iter()
                    .map(|word| word.count_ones() as usize)
                    .sum::<usize>();
                let safe_subtree_count = safe_subtrees
                    .iter()
                    .filter(|&&safe| safe != 0)
                    .count();
                let source_reentry_safe_subtree_count = source_reentry_safe_subtrees
                    .iter()
                    .filter(|&&safe| safe != 0)
                    .count();
                let common_future_subtree_count = common_future_masks
                    .iter()
                    .filter(|&&mask| mask != 0)
                    .count();
                let common_future_fingerprint = common_future_masks.iter().fold(
                    0xcbf29ce484222325u64,
                    |hash, &mask| {
                        (hash ^ mask)
                            .wrapping_mul(0x100000001b3)
                    },
                );
                eprintln!(
                    "[glrmask/profile][dynamic_self_loop_projection_build] source_state={} futures={:?} loop_bytes={} safe_tokens={} safe_subtrees={} source_reentry_safe_subtrees={} common_future_subtrees={} common_future_fingerprint={:016x}",
                    source_state,
                    source_future_terminals,
                    source_loop_bytes.len(),
                    safe_tokens,
                    safe_subtree_count,
                    source_reentry_safe_subtree_count,
                    common_future_subtree_count,
                    common_future_fingerprint,
                );
            }
            DynamicSelfLoopProjection {
                source_state,
                future_terminals: Arc::from(source_future_terminals),
                safe_no_match_mask: Arc::from(safe_no_match_mask),
                safe_subtrees: Arc::from(safe_subtrees),
                source_reentry_safe_subtrees: Arc::from(source_reentry_safe_subtrees),
                common_future_masks: Arc::from(common_future_masks),
                pre_match_dead_words: Arc::from(pre_match_dead_words),
                pre_match_frontier_words: Arc::from(pre_match_frontier_words),
                first_match_fusion_source_state: u32::MAX,
                first_match_fusion_match_state: u32::MAX,
                first_match_fusion_candidate_mask: Arc::from(Vec::<u32>::new()),
                first_match_fusion_candidate_subtrees: Arc::from(Vec::<u64>::new()),
                first_match_fusions: Arc::from(Vec::<(u32, u32)>::new()),
                first_match_step_source_state: u32::MAX,
                first_match_step_root_live_tokens: Arc::from(Vec::<u32>::new()),
                first_match_step_exact_end_tokens: Arc::from(Vec::<u32>::new()),
                first_match_step_post_rows: Arc::from(Vec::<DynamicFirstMatchPostRow>::new()),
                first_match_step_second_rows: Arc::from(Vec::<DynamicFirstMatchSecondRow>::new()),
                first_match_step_unknown_tokens: Arc::from(Vec::<u32>::new()),
                first_match_step_unknown_subtrees: Arc::from(Vec::<u64>::new()),
                root_effect_source_state: u32::MAX,
                root_effect_post_rows: Arc::from(Vec::<DynamicFirstMatchPostRow>::new()),
                root_effect_rows: Arc::from(Vec::<DynamicFirstMatchSecondRow>::new()),
                root_effect_unknown_tokens: Arc::from(Vec::<u32>::new()),
                root_effect_unknown_subtrees: Arc::from(Vec::<u64>::new()),
                config_subtree_certificates: Arc::from(
                    config_subtree_certificates.into_boxed_slice(),
                ),
            }
        }

        let projection_tokenizer = vocab
            .mask_projection_tokenizer()
            .unwrap_or(&self.tokenizer);
        if let Ok(value) = std::env::var("GLRMASK_PROFILE_DYNAMIC_FUTURE_QUOTIENT_STATES") {
            let requested = value
                .split(',')
                .filter_map(|value| value.trim().parse::<u32>().ok())
                .map(|state| vocab.mask_projection_state(state))
                .filter(|&state| state < projection_tokenizer.num_states())
                .collect::<Vec<_>>();
            let mut by_terminal = BTreeMap::<TerminalID, Vec<u32>>::new();
            for &state in &requested {
                let futures = projection_tokenizer
                    .possible_future_terminals_iter(state)
                    .collect::<Vec<_>>();
                if let [terminal] = futures.as_slice() {
                    by_terminal.entry(*terminal).or_default().push(state);
                }
            }
            for (terminal, states) in by_terminal {
                let horizon = std::env::var("GLRMASK_PROFILE_DYNAMIC_FUTURE_QUOTIENT_HORIZON")
                    .ok()
                    .and_then(|value| value.trim().parse::<u8>().ok())
                    .filter(|&value| value > 0)
                    .unwrap_or(64);
                let started = std::time::Instant::now();
                let classes = projection_tokenizer
                    .bounded_terminal_future_partition(terminal, horizon);
                let class_count = classes.iter().copied().max().unwrap_or(0) as usize + 1;
                let mut single_future_class_counts = FxHashMap::<u32, usize>::default();
                for state in 0..projection_tokenizer.num_states() {
                    if projection_tokenizer
                        .possible_future_terminals_iter(state)
                        .eq(std::iter::once(terminal))
                    {
                        let class = classes[state as usize];
                        *single_future_class_counts.entry(class).or_default() += 1;
                    }
                }
                let requested_class_counts = single_future_class_counts.clone();
                let mut single_future_classes = single_future_class_counts
                    .into_iter()
                    .collect::<Vec<_>>();
                single_future_classes.sort_unstable_by(|left, right| {
                    (std::cmp::Reverse(left.1), left.0)
                        .cmp(&(std::cmp::Reverse(right.1), right.0))
                });
                eprintln!(
                    "[glrmask/profile][dynamic_future_quotient] terminal={} horizon={} states={} classes={} single_future_classes={} single_future_states={} largest_single_future_classes={:?} elapsed_ms={:.3}",
                    terminal,
                    horizon,
                    projection_tokenizer.num_states(),
                    class_count,
                    single_future_classes.len(),
                    single_future_classes.iter().map(|(_, count)| count).sum::<usize>(),
                    single_future_classes.iter().take(16).collect::<Vec<_>>(),
                    started.elapsed().as_secs_f64() * 1000.0,
                );
                for state in states {
                    let class = classes.get(state as usize).copied().unwrap_or(0);
                    eprintln!(
                        "[glrmask/profile][dynamic_future_quotient_state] terminal={} state={} class={} class_states={}",
                        terminal,
                        state,
                        class,
                        requested_class_counts.get(&class).copied().unwrap_or(0),
                    );
                }
            }
        }
        let quotient_bounded64 = vocab.mask_projection_tokenizer().map(|tokenizer| {
            let (_, bounded64) = tokenizer.precompute_bounded_observation_safe_byte_sets();
            bounded64
        });
        let projection_fast_owned = vocab
            .mask_projection_tokenizer()
            .map(Self::compute_tokenizer_fast_transitions_for);
        let fast_transitions = projection_fast_owned
            .as_ref()
            .unwrap_or(full_fast_transitions);
        let candidates = self.dynamic_self_loop_projection_candidates(
            vocab,
            projection_tokenizer,
            quotient_bounded64.as_deref(),
        );
        let mut built = if candidates.len() <= 1 {
            candidates
                .into_iter()
                .map(|(source_state, future_terminals, probe)| {
                    (
                        build_one(
                            self,
                            projection_tokenizer,
                            vocab,
                            fast_transitions,
                            source_state,
                            &future_terminals,
                        ),
                        probe,
                    )
                })
                .collect::<Vec<_>>()
        } else {
            std::thread::scope(|scope| {
            candidates
                .into_iter()
                .map(|(source_state, future_terminals, probe)| {
                    scope.spawn(move || {
                        (
                            build_one(
                                self,
                                projection_tokenizer,
                                vocab,
                                fast_transitions,
                                source_state,
                                &future_terminals,
                            ),
                            probe,
                        )
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| {
                    handle
                        .join()
                        .expect("dynamic self-loop projection worker panicked")
                })
                .collect()
            })
        };

        // Experimental exact first-match/suffix fusion subset.  This is
        // intentionally opt-in while its runtime value and construction cost
        // are measured.  Only projection states with exactly one full-state
        // preimage are eligible, so every remembered matched lexer state is in
        // the exact runtime tokenizer coordinate rather than a quotient alias.
        let fusion_states = std::env::var("GLRMASK_DYNAMIC_FIRST_MATCH_FUSION_STATES")
            .ok()
            .map(|value| {
                value
                    .split(',')
                    .filter_map(|value| value.trim().parse::<u32>().ok())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        if !fusion_states.is_empty() {
            let multiplicities = vocab.mask_projection_state_multiplicities();
            let special_tokens = self
                .special_token_terminals
                .iter()
                .map(|special| special.token_id)
                .collect::<BTreeSet<_>>();
            let mut ordinary_token_by_bytes = FxHashMap::<&[u8], u32>::default();
            for (&token, bytes) in self.token_bytes.iter() {
                if !special_tokens.contains(&token) {
                    ordinary_token_by_bytes
                        .entry(bytes.as_slice())
                        .or_insert(token);
                }
            }

            for full_source_state in fusion_states {
                if full_source_state >= self.tokenizer.num_states() {
                    continue;
                }
                let projection_state = vocab.mask_projection_state(full_source_state);
                if multiplicities
                    .as_ref()
                    .and_then(|counts| counts.get(projection_state as usize))
                    .copied()
                    .unwrap_or(1)
                    != 1
                {
                    continue;
                }
                let Some((projection, _)) = built
                    .iter_mut()
                    .find(|(projection, _)| projection.source_state == projection_state)
                else {
                    continue;
                };
                let [terminal] = projection.future_terminals.as_ref() else {
                    continue;
                };

                let started = std::time::Instant::now();
                let mut fusions = Vec::<(u32, u32)>::new();
                let mut all_suffix_entries = Vec::<(usize, &[u8])>::new();
                let mut common_match_state = None::<u32>;
                let mut conflicting_match_state = false;
                for (&fused_token, bytes) in self.token_bytes.iter() {
                    if bytes.is_empty() {
                        continue;
                    }
                    let execution = self
                        .tokenizer
                        .execute_from_state_all_widths(bytes, full_source_state);
                    let Some(first_width) = execution
                        .matches
                        .iter()
                        .filter(|matched| matched.id == *terminal)
                        .map(|matched| matched.width)
                        .min()
                    else {
                        continue;
                    };
                    if first_width >= bytes.len() {
                        // The empty-suffix case is tiny and the ordinary walk
                        // already handles it; keeping this subset to nonempty
                        // suffix tokens simplifies candidate validation.
                        continue;
                    }
                    let mut match_states = execution
                        .matches
                        .iter()
                        .filter(|matched| matched.id == *terminal && matched.width == first_width)
                        .map(|matched| matched.end_state);
                    let Some(match_state) = match_states.next() else {
                        continue;
                    };
                    if match_states.any(|state| state != match_state) {
                        continue;
                    }
                    match common_match_state {
                        None => common_match_state = Some(match_state),
                        Some(existing) if existing == match_state => {}
                        Some(_) => {
                            conflicting_match_state = true;
                            break;
                        }
                    }
                    let suffix = &bytes[first_width..];
                    all_suffix_entries.push((fused_token as usize, suffix));
                    let Some(&suffix_token) = ordinary_token_by_bytes.get(suffix) else {
                        continue;
                    };
                    fusions.push((fused_token, suffix_token));
                }
                if conflicting_match_state || fusions.is_empty() {
                    continue;
                }
                let mut suffix_byte_set = BTreeSet::<&[u8]>::new();
                let suffixes_unique = all_suffix_entries
                    .iter()
                    .all(|(_, suffix)| suffix_byte_set.insert(*suffix));
                let suffix_trie_profile = suffixes_unique.then(|| {
                    Self::build_dynamic_mask_trie_partitioned(&all_suffix_entries)
                });
                fusions.sort_unstable();
                fusions.dedup();
                let mut candidate_mask = vec![0u32; self.body_mask_len()];
                for &(_, suffix_token) in &fusions {
                    let word = suffix_token as usize / 32;
                    if let Some(bits) = candidate_mask.get_mut(word) {
                        *bits |= 1u32 << (suffix_token % 32);
                    }
                }
                let ordered = vocab.trie.all_subtree_tokens();
                let mut candidate_prefix = Vec::<u32>::with_capacity(ordered.len() + 1);
                candidate_prefix.push(0);
                let mut candidate_count = 0u32;
                for &canonical_token in ordered {
                    let is_candidate = vocab.token_ids(canonical_token).is_some_and(|aliases| {
                        aliases.iter().any(|&token_id| {
                            let word = token_id as usize / 32;
                            let bit = token_id % 32;
                            candidate_mask
                                .get(word)
                                .is_some_and(|bits| bits & (1u32 << bit) != 0)
                        })
                    });
                    candidate_count += u32::from(is_candidate);
                    candidate_prefix.push(candidate_count);
                }
                let mut candidate_subtrees =
                    vec![0u64; vocab.trie.node_count().div_ceil(64)];
                for node in 0..vocab.trie.node_count() as u32 {
                    let range = vocab.trie.subtree_token_index_range(node);
                    if candidate_prefix[range.start] != candidate_prefix[range.end] {
                        candidate_subtrees[node as usize >> 6] |= 1u64 << (node & 63);
                    }
                }
                projection.first_match_fusion_source_state = full_source_state;
                projection.first_match_fusion_match_state = common_match_state.unwrap_or(u32::MAX);
                projection.first_match_fusion_candidate_mask = Arc::from(candidate_mask);
                projection.first_match_fusion_candidate_subtrees = Arc::from(candidate_subtrees);
                projection.first_match_fusions = Arc::from(fusions.into_boxed_slice());
                if std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
                    || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
                {
                    eprintln!(
                        "[glrmask/profile][dynamic_first_match_fusions] full_source={} projection_source={} terminal={} match_state={} fusions={} suffix_candidates={} elapsed_ms={:.3}",
                        full_source_state,
                        projection_state,
                        terminal,
                        projection.first_match_fusion_match_state,
                        projection.first_match_fusions.len(),
                        projection
                            .first_match_fusion_candidate_mask
                            .iter()
                            .map(|word| word.count_ones() as usize)
                            .sum::<usize>(),
                        started.elapsed().as_secs_f64() * 1000.0,
                    );
                    if let Some(suffix_trie) = suffix_trie_profile.as_ref() {
                        eprintln!(
                            "[glrmask/profile][dynamic_first_match_suffix_trie] full_source={} suffixes={} trie_nodes={} trie_edges={} trie_bytes={} max_depth_bytes={}",
                            full_source_state,
                            all_suffix_entries.len(),
                            suffix_trie.nodes.len(),
                            suffix_trie.edges.len(),
                            suffix_trie.edge_bytes_len(),
                            suffix_trie.subtree_max_byte_len(0),
                        );
                    } else {
                        eprintln!(
                            "[glrmask/profile][dynamic_first_match_suffix_trie] full_source={} suffixes={} duplicate_suffixes=true",
                            full_source_state,
                            all_suffix_entries.len(),
                        );
                    }
                }
            }
        }

        // Exact one-finalization decomposition.  This is a stronger and more
        // general version of the suffix-token fusion experiment above: suffix
        // bytes do not need to be a vocabulary token themselves.  Instead we
        // classify the concrete model token by the residual lexer terminals
        // that remain possible after the first finalization/reset.  Any token
        // that sees a second finalization (or has ambiguous first-match width)
        // is left in `unknown` and therefore still goes through the ordinary
        // exact dynamic walker at runtime.
        let first_match_step_states =
            std::env::var("GLRMASK_DYNAMIC_FIRST_MATCH_ONE_STEP_STATES")
                .ok()
                .map(|value| {
                    value
                        .split(',')
                        .filter_map(|value| value.trim().parse::<u32>().ok())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
        if !first_match_step_states.is_empty() {
            let multiplicities = vocab.mask_projection_state_multiplicities();
            let special_tokens = self
                .special_token_terminals
                .iter()
                .map(|special| special.token_id)
                .collect::<BTreeSet<_>>();

            for full_source_state in first_match_step_states {
                if full_source_state >= self.tokenizer.num_states() {
                    continue;
                }
                let projection_state = vocab.mask_projection_state(full_source_state);
                if multiplicities
                    .as_ref()
                    .and_then(|counts| counts.get(projection_state as usize))
                    .copied()
                    .unwrap_or(1)
                    != 1
                {
                    continue;
                }
                let Some((projection, _)) = built
                    .iter_mut()
                    .find(|(projection, _)| projection.source_state == projection_state)
                else {
                    continue;
                };
                let [terminal] = projection.future_terminals.as_ref() else {
                    continue;
                };
                if Some(*terminal) == self.ignore_terminal {
                    continue;
                }

                let started = std::time::Instant::now();
                let mut root_live_tokens = Vec::<u32>::new();
                let mut exact_end_tokens = Vec::<u32>::new();
                let mut unknown_tokens = Vec::<u32>::new();
                let mut post_tokens_by_terminal =
                    BTreeMap::<TerminalID, Vec<u32>>::new();
                let mut known_reject = 0usize;

                for (&token_id, bytes) in self.token_bytes.iter() {
                    if special_tokens.contains(&token_id) || bytes.is_empty() {
                        continue;
                    }
                    let execution = self
                        .tokenizer
                        .execute_from_state_all_widths(bytes, full_source_state);

                    // A no-finalization continuation is already a complete
                    // witness whenever the sole root terminal is parser-live.
                    // Prefer it over the matched branch so the token need not
                    // enter the unknown set merely because some *other* lexer
                    // path also finalizes inside the token.
                    let root_live = execution.end_state.iter().any(|&end_state| {
                        self.tokenizer
                            .possible_future_terminals(end_state)
                            .contains(*terminal as usize)
                    });
                    if root_live {
                        root_live_tokens.push(token_id);
                        continue;
                    }

                    if execution.matches.iter().any(|matched| matched.id != *terminal) {
                        unknown_tokens.push(token_id);
                        continue;
                    }
                    let mut widths = execution
                        .matches
                        .iter()
                        .filter(|matched| matched.id == *terminal)
                        .map(|matched| matched.width)
                        .collect::<Vec<_>>();
                    widths.sort_unstable();
                    widths.dedup();
                    let [first_width] = widths.as_slice() else {
                        if widths.is_empty() {
                            known_reject += 1;
                        } else {
                            unknown_tokens.push(token_id);
                        }
                        continue;
                    };
                    if *first_width == bytes.len() {
                        exact_end_tokens.push(token_id);
                        continue;
                    }

                    let suffix = &bytes[*first_width..];
                    let reset_execution = self
                        .tokenizer
                        .execute_from_state_all_widths(suffix, self.tokenizer.start_state());
                    if !reset_execution.matches.is_empty() {
                        unknown_tokens.push(token_id);
                        continue;
                    }
                    let mut futures = BitSet::new(self.tokenizer.num_terminals() as usize);
                    for &end_state in &reset_execution.end_state {
                        futures.union_with_prefix(self.tokenizer.possible_future_terminals(end_state));
                    }
                    if futures.is_empty() {
                        known_reject += 1;
                        continue;
                    }
                    for post_terminal in futures.iter_ones() {
                        post_tokens_by_terminal
                            .entry(post_terminal as TerminalID)
                            .or_default()
                            .push(token_id);
                    }
                }

                root_live_tokens.sort_unstable();
                root_live_tokens.dedup();
                exact_end_tokens.sort_unstable();
                exact_end_tokens.dedup();
                unknown_tokens.sort_unstable();
                unknown_tokens.dedup();

                // Compile the remaining reset-suffix language into a short
                // recursive lexical-effect program.  Each row consumes one
                // parser terminal, then either ends the model token, reaches a
                // residual future-terminal set at token boundary, or descends
                // to another row after the lexer resets again.  The program is
                // entirely vocabulary-derived; runtime still performs the real
                // parser advances.  A bounded recursion depth keeps the
                // construction predictable, with any unresolved token falling
                // back to the exact byte-wise walker.
                let first_step_unknown_count = unknown_tokens.len();

                fn grouped_post_rows(
                    by_terminal: BTreeMap<TerminalID, Vec<u32>>,
                ) -> Vec<DynamicFirstMatchPostRow> {
                    let mut terminals_by_tokens =
                        BTreeMap::<Vec<u32>, Vec<TerminalID>>::new();
                    for (terminal, mut tokens) in by_terminal {
                        tokens.sort_unstable();
                        tokens.dedup();
                        if !tokens.is_empty() {
                            terminals_by_tokens.entry(tokens).or_default().push(terminal);
                        }
                    }
                    let mut rows = terminals_by_tokens
                        .into_iter()
                        .map(|(tokens, mut terminals)| {
                            terminals.sort_unstable();
                            terminals.dedup();
                            dynamic_effect_post_row(terminals, tokens)
                        })
                        .collect::<Vec<_>>();
                    rows.sort_unstable_by(|left, right| {
                        std::cmp::Reverse(left.tokens.len())
                            .cmp(&std::cmp::Reverse(right.tokens.len()))
                            .then_with(|| {
                                left.terminals.as_ref().cmp(right.terminals.as_ref())
                            })
                    });
                    rows
                }

                fn build_reset_effect_rows(
                    tokenizer: &Tokenizer,
                    mut entries: Vec<(u32, Vec<u8>)>,
                    depth_left: usize,
                    unresolved: &mut BTreeSet<u32>,
                ) -> (Vec<DynamicFirstMatchPostRow>, Vec<DynamicFirstMatchSecondRow>) {
                    entries.sort_unstable();
                    entries.dedup();
                    let mut residual_by_terminal =
                        BTreeMap::<TerminalID, Vec<u32>>::new();
                    let mut exact_by_terminal = BTreeMap::<TerminalID, Vec<u32>>::new();
                    let mut children_by_terminal =
                        BTreeMap::<TerminalID, Vec<(u32, Vec<u8>)>>::new();

                    for (token_id, bytes) in entries {
                        if bytes.is_empty() {
                            unresolved.insert(token_id);
                            continue;
                        }
                        let execution = tokenizer.execute_from_state_all_widths(
                            &bytes,
                            tokenizer.start_state(),
                        );
                        let mut futures = BitSet::new(tokenizer.num_terminals() as usize);
                        for &end_state in &execution.end_state {
                            futures.union_with_prefix(tokenizer.possible_future_terminals(end_state));
                        }
                        for future_terminal in futures.iter_ones() {
                            residual_by_terminal
                                .entry(future_terminal as TerminalID)
                                .or_default()
                                .push(token_id);
                        }

                        let mut branches = execution
                            .matches
                            .iter()
                            .map(|matched| (matched.id, matched.width))
                            .collect::<Vec<_>>();
                        branches.sort_unstable();
                        branches.dedup();
                        for (terminal, width) in branches {
                            if width == 0 || width > bytes.len() {
                                unresolved.insert(token_id);
                            } else if width == bytes.len() {
                                exact_by_terminal.entry(terminal).or_default().push(token_id);
                            } else if depth_left == 0 {
                                unresolved.insert(token_id);
                            } else {
                                children_by_terminal
                                    .entry(terminal)
                                    .or_default()
                                    .push((token_id, bytes[width..].to_vec()));
                            }
                        }
                    }

                    let post_rows = grouped_post_rows(residual_by_terminal);
                    let mut terminals = BTreeSet::<TerminalID>::new();
                    terminals.extend(exact_by_terminal.keys().copied());
                    terminals.extend(children_by_terminal.keys().copied());
                    let mut rows = Vec::<DynamicFirstMatchSecondRow>::new();
                    for terminal in terminals {
                        let mut exact_end_tokens =
                            exact_by_terminal.remove(&terminal).unwrap_or_default();
                        exact_end_tokens.sort_unstable();
                        exact_end_tokens.dedup();
                        let child_entries =
                            children_by_terminal.remove(&terminal).unwrap_or_default();
                        let (child_post_rows, next_rows) = if child_entries.is_empty() {
                            (Vec::new(), Vec::new())
                        } else {
                            build_reset_effect_rows(
                                tokenizer,
                                child_entries,
                                depth_left.saturating_sub(1),
                                unresolved,
                            )
                        };
                        if !exact_end_tokens.is_empty()
                            || !child_post_rows.is_empty()
                            || !next_rows.is_empty()
                        {
                            rows.push(DynamicFirstMatchSecondRow {
                                terminal,
                                exact_end_tokens: Arc::from(exact_end_tokens),
                                post_rows: Arc::from(child_post_rows),
                                next_rows: Arc::from(next_rows),
                            });
                        }
                    }
                    rows.sort_unstable_by_key(|row| row.terminal);
                    (post_rows, rows)
                }

                let mut reset_entries = Vec::<(u32, Vec<u8>)>::new();
                let mut unresolved = BTreeSet::<u32>::new();
                for &token_id in &unknown_tokens {
                    let Some(bytes) = self.token_bytes.get(&token_id) else {
                        unresolved.insert(token_id);
                        continue;
                    };
                    let execution = self
                        .tokenizer
                        .execute_from_state_all_widths(bytes, full_source_state);
                    if execution.matches.iter().any(|matched| matched.id != *terminal) {
                        unresolved.insert(token_id);
                        continue;
                    }
                    let mut widths = execution
                        .matches
                        .iter()
                        .filter(|matched| matched.id == *terminal)
                        .map(|matched| matched.width)
                        .collect::<Vec<_>>();
                    widths.sort_unstable();
                    widths.dedup();
                    let [first_width] = widths.as_slice() else {
                        unresolved.insert(token_id);
                        continue;
                    };
                    if *first_width >= bytes.len() {
                        unresolved.insert(token_id);
                        continue;
                    }
                    reset_entries.push((token_id, bytes[*first_width..].to_vec()));
                }

                // The measured vocabulary needs at most five reset
                // finalizations below the first match. Six leaves one level of
                // safety margin while keeping pathological grammars bounded.
                let (extra_post_rows, second_rows) = build_reset_effect_rows(
                    &self.tokenizer,
                    reset_entries,
                    6,
                    &mut unresolved,
                );
                for row in extra_post_rows {
                    for &post_terminal in row.terminals.iter() {
                        post_tokens_by_terminal
                            .entry(post_terminal)
                            .or_default()
                            .extend(row.tokens.iter().copied());
                    }
                }
                unknown_tokens = unresolved.into_iter().collect();

                let mut unknown_mask = vec![0u32; self.body_mask_len()];
                for &token_id in &unknown_tokens {
                    let word = token_id as usize / 32;
                    if let Some(bits) = unknown_mask.get_mut(word) {
                        *bits |= 1u32 << (token_id % 32);
                    }
                }
                let ordered = vocab.trie.all_subtree_tokens();
                let mut unknown_prefix = Vec::<u32>::with_capacity(ordered.len() + 1);
                unknown_prefix.push(0);
                let mut unknown_count = 0u32;
                for &canonical_token in ordered {
                    let is_unknown = vocab.token_ids(canonical_token).is_some_and(|aliases| {
                        aliases.iter().any(|&token_id| {
                            let word = token_id as usize / 32;
                            let bit = token_id % 32;
                            unknown_mask
                                .get(word)
                                .is_some_and(|bits| bits & (1u32 << bit) != 0)
                        })
                    });
                    unknown_count += u32::from(is_unknown);
                    unknown_prefix.push(unknown_count);
                }
                let mut unknown_subtrees =
                    vec![0u64; vocab.trie.node_count().div_ceil(64)];
                for node in 0..vocab.trie.node_count() as u32 {
                    let range = vocab.trie.subtree_token_index_range(node);
                    if unknown_prefix[range.start] != unknown_prefix[range.end] {
                        unknown_subtrees[node as usize >> 6] |= 1u64 << (node & 63);
                    }
                }

                // Many schema terminals have different parser identities but
                // exactly the same concrete-vocabulary residual language.
                // Collapse them by their fused-token row so runtime performs
                // one admitted-terminal intersection per distinct row rather
                // than one test per grammar terminal.
                let mut terminals_by_tokens =
                    BTreeMap::<Vec<u32>, Vec<TerminalID>>::new();
                for (post_terminal, mut tokens) in post_tokens_by_terminal {
                    tokens.sort_unstable();
                    tokens.dedup();
                    if !tokens.is_empty() {
                        terminals_by_tokens
                            .entry(tokens)
                            .or_default()
                            .push(post_terminal);
                    }
                }
                let mut post_rows = terminals_by_tokens
                    .into_iter()
                    .map(|(tokens, mut terminals)| {
                        terminals.sort_unstable();
                        terminals.dedup();
                        dynamic_effect_post_row(terminals, tokens)
                    })
                    .collect::<Vec<_>>();
                post_rows.sort_unstable_by(|left, right| {
                    std::cmp::Reverse(left.tokens.len())
                        .cmp(&std::cmp::Reverse(right.tokens.len()))
                        .then_with(|| left.terminals.as_ref().cmp(right.terminals.as_ref()))
                });

                projection.first_match_step_source_state = full_source_state;
                projection.first_match_step_root_live_tokens = Arc::from(root_live_tokens);
                projection.first_match_step_exact_end_tokens = Arc::from(exact_end_tokens);
                projection.first_match_step_post_rows = Arc::from(post_rows);
                projection.first_match_step_second_rows = Arc::from(second_rows);
                projection.first_match_step_unknown_tokens = Arc::from(unknown_tokens);
                projection.first_match_step_unknown_subtrees = Arc::from(unknown_subtrees);

                if std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
                    || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
                {
                    let mut unknown_second_pairs =
                        FxHashMap::<Vec<(TerminalID, usize)>, usize>::default();
                    let mut unknown_second_terminals = FxHashMap::<TerminalID, usize>::default();
                    let mut unknown_multi_first = 0usize;
                    let mut unknown_second_match = 0usize;
                    let mut unknown_other = 0usize;
                    let mut second_branches = 0usize;
                    let mut second_exact_end_branches = 0usize;
                    let mut second_residual_only_branches = 0usize;
                    let mut third_match_branches = 0usize;
                    let mut tokens_with_third_match = BTreeSet::<u32>::new();
                    let mut remaining_depth_histogram = BTreeMap::<usize, usize>::new();
                    let mut depth_memo = FxHashMap::<Vec<u8>, usize>::default();
                    fn max_reset_finalization_depth(
                        tokenizer: &Tokenizer,
                        bytes: &[u8],
                        memo: &mut FxHashMap<Vec<u8>, usize>,
                    ) -> usize {
                        if bytes.is_empty() {
                            return 0;
                        }
                        if let Some(&cached) = memo.get(bytes) {
                            return cached;
                        }
                        let execution = tokenizer.execute_from_state_all_widths(
                            bytes,
                            tokenizer.start_state(),
                        );
                        let mut branches = execution
                            .matches
                            .iter()
                            .map(|matched| (matched.id, matched.width))
                            .collect::<Vec<_>>();
                        branches.sort_unstable();
                        branches.dedup();
                        let mut depth = 0usize;
                        for (_, width) in branches {
                            let child = if width < bytes.len() {
                                max_reset_finalization_depth(tokenizer, &bytes[width..], memo)
                            } else {
                                0
                            };
                            depth = depth.max(1 + child);
                        }
                        memo.insert(bytes.to_vec(), depth);
                        depth
                    }
                    for &token_id in projection.first_match_step_unknown_tokens.iter() {
                        let Some(bytes) = self.token_bytes.get(&token_id) else {
                            continue;
                        };
                        let execution = self
                            .tokenizer
                            .execute_from_state_all_widths(bytes, full_source_state);
                        let mut widths = execution
                            .matches
                            .iter()
                            .filter(|matched| matched.id == *terminal)
                            .map(|matched| matched.width)
                            .collect::<Vec<_>>();
                        widths.sort_unstable();
                        widths.dedup();
                        let [first_width] = widths.as_slice() else {
                            unknown_multi_first += 1;
                            continue;
                        };
                        if *first_width >= bytes.len() {
                            unknown_other += 1;
                            continue;
                        }
                        let reset_execution = self.tokenizer.execute_from_state_all_widths(
                            &bytes[*first_width..],
                            self.tokenizer.start_state(),
                        );
                        let mut pairs = reset_execution
                            .matches
                            .iter()
                            .map(|matched| (matched.id, matched.width))
                            .collect::<Vec<_>>();
                        pairs.sort_unstable();
                        pairs.dedup();
                        if pairs.is_empty() {
                            unknown_other += 1;
                            continue;
                        }
                        unknown_second_match += 1;
                        for &(second_terminal, _) in &pairs {
                            *unknown_second_terminals.entry(second_terminal).or_default() += 1;
                        }
                        *unknown_second_pairs.entry(pairs).or_default() += 1;

                        // Classify each distinct second-finalization branch by
                        // what remains after its reset.  This measures the
                        // depth needed by the next exact static layer.
                        let suffix = &bytes[*first_width..];
                        let remaining_depth = max_reset_finalization_depth(
                            &self.tokenizer,
                            suffix,
                            &mut depth_memo,
                        );
                        *remaining_depth_histogram.entry(remaining_depth).or_default() += 1;
                        let mut branches = reset_execution
                            .matches
                            .iter()
                            .map(|matched| (matched.id, matched.width))
                            .collect::<Vec<_>>();
                        branches.sort_unstable();
                        branches.dedup();
                        for (_second_terminal, second_width) in branches {
                            second_branches += 1;
                            if second_width == suffix.len() {
                                second_exact_end_branches += 1;
                                continue;
                            }
                            let third_execution = self.tokenizer.execute_from_state_all_widths(
                                &suffix[second_width..],
                                self.tokenizer.start_state(),
                            );
                            if third_execution.matches.is_empty() {
                                second_residual_only_branches += 1;
                            } else {
                                third_match_branches += 1;
                                tokens_with_third_match.insert(token_id);
                            }
                        }
                    }
                    let mut second_terminal_summary = unknown_second_terminals
                        .into_iter()
                        .map(|(terminal, count)| (count, terminal))
                        .collect::<Vec<_>>();
                    second_terminal_summary.sort_unstable_by(|a, b| b.cmp(a));
                    second_terminal_summary.truncate(20);
                    let mut second_pair_summary = unknown_second_pairs
                        .into_iter()
                        .map(|(pairs, count)| (count, pairs))
                        .collect::<Vec<_>>();
                    second_pair_summary.sort_unstable_by(|a, b| b.0.cmp(&a.0));
                    second_pair_summary.truncate(20);
                    let row_summary = projection
                        .first_match_step_post_rows
                        .iter()
                        .take(16)
                        .map(|row| (row.tokens.len(), row.terminals.len(), row.terminals.to_vec()))
                        .collect::<Vec<_>>();
                    eprintln!(
                        "[glrmask/profile][dynamic_first_match_one_step] full_source={} projection_source={} terminal={} root_live={} exact_end={} post_rows={} second_rows={} first_unknown={} fallback_unknown={} known_reject={} rows={:?} elapsed_ms={:.3}",
                        full_source_state,
                        projection_state,
                        terminal,
                        projection.first_match_step_root_live_tokens.len(),
                        projection.first_match_step_exact_end_tokens.len(),
                        projection.first_match_step_post_rows.len(),
                        projection.first_match_step_second_rows.len(),
                        first_step_unknown_count,
                        projection.first_match_step_unknown_tokens.len(),
                        known_reject,
                        row_summary,
                        started.elapsed().as_secs_f64() * 1000.0,
                    );
                    eprintln!(
                        "[glrmask/profile][dynamic_first_match_one_step_unknown] full_source={} unknown={} multi_first={} second_match={} other={} second_branches={} second_exact_end={} second_residual_only={} third_match_branches={} tokens_with_third_match={} remaining_depth_histogram={:?} second_terminals={:?} second_pairs={:?}",
                        full_source_state,
                        projection.first_match_step_unknown_tokens.len(),
                        unknown_multi_first,
                        unknown_second_match,
                        unknown_other,
                        second_branches,
                        second_exact_end_branches,
                        second_residual_only_branches,
                        third_match_branches,
                        tokens_with_third_match.len(),
                        remaining_depth_histogram,
                        second_terminal_summary,
                        second_pair_summary,
                    );
                }
            }
        }

        // General root lexical-effect program.  Unlike the single-future
        // experiment above, this preserves every residual future terminal at
        // the source state and every possible first terminal finalization.
        // It is therefore applicable to the multi-future lexer states that now
        // dominate the median after the single-future paths were accelerated.
        let mut root_effect_states = std::env::var("GLRMASK_DYNAMIC_ROOT_EFFECT_STATES")
            .ok()
            .map(|value| {
                value
                    .split(',')
                    .filter_map(|value| value.trim().parse::<u32>().ok())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        if let Some(auto_limit) = std::env::var("GLRMASK_DYNAMIC_ROOT_EFFECT_AUTO_NARROW_LIMIT")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|&limit| limit > 0)
            && let Some(unique_full_states) = vocab.mask_projection_unique_full_states()
        {
            let max_first_byte_tokens = std::env::var(
                "GLRMASK_DYNAMIC_ROOT_EFFECT_AUTO_NARROW_MAX_FIRST_BYTE_TOKENS",
            )
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(4_096);
            let mut first_byte_tokens = [0usize; 256];
            for bytes in self.token_bytes.values() {
                if let Some(&byte) = bytes.first() {
                    first_byte_tokens[byte as usize] += 1;
                }
            }
            let mut parser_rows_by_terminal = vec![0usize; self.parser_terminal_count() as usize];
            for row in (0..self.parser_symbol_count()).filter_map(|top| self.parser_advance_row(top)) {
                for terminal in row.iter() {
                    if let Some(count) = parser_rows_by_terminal.get_mut(terminal) {
                        *count += 1;
                    }
                }
            }
            let already_selected = root_effect_states.iter().copied().collect::<BTreeSet<_>>();
            let mut ranked = Vec::<(usize, usize, usize, u32)>::new();
            for projection_state in 0..projection_tokenizer.num_states() {
                let Some(&full_source_state) = unique_full_states.get(projection_state as usize)
                else {
                    continue;
                };
                if full_source_state == u32::MAX
                    || already_selected.contains(&full_source_state)
                    || self
                        .tokenizer
                        .matched_terminals_iter(full_source_state)
                        .next()
                        .is_some()
                {
                    continue;
                }
                let futures = self
                    .tokenizer
                    .possible_future_terminals_iter(full_source_state)
                    .collect::<SmallVec<[TerminalID; 8]>>();
                if futures.is_empty() || futures.len() > 8 {
                    continue;
                }
                let mut transition_bytes = SmallVec::<[u8; 4]>::new();
                for (byte, _) in self.tokenizer.transitions_from(full_source_state) {
                    if !transition_bytes.contains(&byte) {
                        transition_bytes.push(byte);
                    }
                    if transition_bytes.len() > 3 {
                        break;
                    }
                }
                if transition_bytes.is_empty() || transition_bytes.len() > 3 {
                    continue;
                }
                let first_byte_support = transition_bytes
                    .iter()
                    .map(|&byte| first_byte_tokens[byte as usize])
                    .sum::<usize>();
                if first_byte_support == 0 || first_byte_support > max_first_byte_tokens {
                    continue;
                }
                let parser_rows = futures
                    .iter()
                    .filter_map(|&terminal| {
                        parser_rows_by_terminal.get(terminal as usize).copied()
                    })
                    .max()
                    .unwrap_or(0);
                if parser_rows == 0 {
                    continue;
                }
                let score = parser_rows.saturating_mul(first_byte_support);
                ranked.push((score, parser_rows, first_byte_support, full_source_state));
            }
            ranked.sort_unstable_by(|left, right| {
                (
                    std::cmp::Reverse(left.0),
                    std::cmp::Reverse(left.1),
                    std::cmp::Reverse(left.2),
                    left.3,
                )
                    .cmp(&(
                        std::cmp::Reverse(right.0),
                        std::cmp::Reverse(right.1),
                        std::cmp::Reverse(right.2),
                        right.3,
                    ))
            });
            let available = ranked.len();
            root_effect_states.extend(
                ranked
                    .into_iter()
                    .take(auto_limit)
                    .map(|(_, _, _, full_source_state)| full_source_state),
            );
            if std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
                || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
            {
                eprintln!(
                    "[glrmask/profile][dynamic_root_effect_auto_narrow] limit={} available={} selected={} max_first_byte_tokens={}",
                    auto_limit,
                    available,
                    root_effect_states.len().min(auto_limit),
                    max_first_byte_tokens,
                );
            }
        }
        root_effect_states.sort_unstable();
        root_effect_states.dedup();
        if !root_effect_states.is_empty() {
            let multiplicities = vocab.mask_projection_state_multiplicities();
            let special_tokens = self
                .special_token_terminals
                .iter()
                .map(|special| special.token_id)
                .collect::<BTreeSet<_>>();

            for full_source_state in root_effect_states {
                if full_source_state >= self.tokenizer.num_states() {
                    continue;
                }
                let projection_state = vocab.mask_projection_state(full_source_state);
                if multiplicities
                    .as_ref()
                    .and_then(|counts| counts.get(projection_state as usize))
                    .copied()
                    .unwrap_or(1)
                    != 1
                {
                    continue;
                }

                if !built
                    .iter()
                    .any(|(projection, _)| projection.source_state == projection_state)
                {
                    let futures = projection_tokenizer
                        .possible_future_terminals_iter(projection_state)
                        .collect::<Vec<_>>();
                    if futures.is_empty() {
                        continue;
                    }
                    built.push((
                        build_one(
                            self,
                            projection_tokenizer,
                            vocab,
                            fast_transitions,
                            projection_state,
                            &futures,
                        ),
                        false,
                    ));
                }
                let Some((projection, _)) = built
                    .iter_mut()
                    .find(|(projection, _)| projection.source_state == projection_state)
                else {
                    continue;
                };

                let started = std::time::Instant::now();
                let mut residual_by_terminal = BTreeMap::<TerminalID, Vec<u32>>::new();
                let mut exact_by_terminal = BTreeMap::<TerminalID, Vec<u32>>::new();
                let mut children_by_terminal =
                    BTreeMap::<TerminalID, Vec<(u32, Vec<u8>)>>::new();
                let mut unresolved = BTreeSet::<u32>::new();
                let mut classified_tokens = 0usize;

                // Execute the source lexer over the vocabulary radix trie
                // rather than restarting from the source state for every
                // complete model token.  All descendants of a trie node share
                // the same consumed byte prefix, lexer configuration, and
                // accumulated terminal finalizations, so this preserves the
                // exact all-match-width semantics while avoiding repeated
                // scans of common token prefixes.
                struct RootEffectBuild<'a> {
                    tokenizer: &'a Tokenizer,
                    vocab: &'a DynamicMaskVocab,
                    special_tokens: &'a BTreeSet<u32>,
                    residual_by_terminal: &'a mut BTreeMap<TerminalID, Vec<u32>>,
                    exact_by_terminal: &'a mut BTreeMap<TerminalID, Vec<u32>>,
                    children_by_terminal:
                        &'a mut BTreeMap<TerminalID, Vec<(u32, Vec<u8>)>>,
                    unresolved: &'a mut BTreeSet<u32>,
                    classified_tokens: &'a mut usize,
                }

                fn visit_root_effect_trie(
                    build: &mut RootEffectBuild<'_>,
                    node: u32,
                    states: &TokenizerStateSet,
                    matches: &[(TerminalID, usize)],
                    prefix: &mut Vec<u8>,
                ) {
                    // If the source lexer has died before any terminal
                    // finalization, no longer token sharing this prefix can
                    // ever become relevant.  This is especially important
                    // for narrow one-transition residual states: without the
                    // cutoff we still walk the whole 128k-token vocabulary
                    // merely to discover that almost every branch is dead.
                    if states.is_empty() && matches.is_empty() {
                        return;
                    }
                    let trie = build.vocab.trie.as_ref();
                    if let Some(canonical_token) = trie.node(node).token_id
                        && let Some(token_ids) = build.vocab.token_ids(canonical_token)
                    {
                        let mut futures = BitSet::new(build.tokenizer.num_terminals() as usize);
                        for &end_state in states {
                            futures.union_with(
                                build.tokenizer.possible_future_terminals(end_state),
                            );
                        }
                        let mut branches = matches.to_vec();
                        branches.sort_unstable();
                        branches.dedup();
                        let classified = !futures.is_empty() || !branches.is_empty();

                        for &token_id in token_ids {
                            if build.special_tokens.contains(&token_id) || prefix.is_empty() {
                                continue;
                            }
                            if classified {
                                *build.classified_tokens += 1;
                            }
                            for future_terminal in futures.iter_ones() {
                                build
                                    .residual_by_terminal
                                    .entry(future_terminal as TerminalID)
                                    .or_default()
                                    .push(token_id);
                            }
                            for &(terminal, width) in &branches {
                                if width == 0 || width > prefix.len() {
                                    build.unresolved.insert(token_id);
                                } else if width == prefix.len() {
                                    build
                                        .exact_by_terminal
                                        .entry(terminal)
                                        .or_default()
                                        .push(token_id);
                                } else {
                                    build
                                        .children_by_terminal
                                        .entry(terminal)
                                        .or_default()
                                        .push((token_id, prefix[width..].to_vec()));
                                }
                            }
                        }
                    }

                    for edge in trie.children(node) {
                        let old_prefix_len = prefix.len();
                        let mut next_states = states.clone();
                        let mut next_matches = matches.to_vec();
                        for &byte in trie.edge_bytes(edge) {
                            prefix.push(byte);
                            if next_states.is_empty() {
                                continue;
                            }
                            next_states = build.tokenizer.step_all(next_states.as_slice(), byte);
                            if next_states.is_empty() {
                                continue;
                            }
                            let width = prefix.len();
                            for &state in &next_states {
                                next_matches.extend(
                                    build
                                        .tokenizer
                                        .matched_terminals_iter(state)
                                        .map(|terminal| (terminal, width)),
                                );
                            }
                        }
                        visit_root_effect_trie(
                            build,
                            edge.child,
                            &next_states,
                            &next_matches,
                            prefix,
                        );
                        prefix.truncate(old_prefix_len);
                    }
                }

                let mut prefix = Vec::<u8>::with_capacity(
                    vocab.trie.subtree_max_byte_len(0) as usize,
                );
                let initial_states = TokenizerStateSet::from_iter([full_source_state]);
                let mut root_build = RootEffectBuild {
                    tokenizer: &self.tokenizer,
                    vocab,
                    special_tokens: &special_tokens,
                    residual_by_terminal: &mut residual_by_terminal,
                    exact_by_terminal: &mut exact_by_terminal,
                    children_by_terminal: &mut children_by_terminal,
                    unresolved: &mut unresolved,
                    classified_tokens: &mut classified_tokens,
                };
                visit_root_effect_trie(
                    &mut root_build,
                    0,
                    &initial_states,
                    &[],
                    &mut prefix,
                );

                let root_post_rows = group_dynamic_effect_post_rows(residual_by_terminal);
                let mut terminals = BTreeSet::<TerminalID>::new();
                terminals.extend(exact_by_terminal.keys().copied());
                terminals.extend(children_by_terminal.keys().copied());
                let mut root_rows = Vec::<DynamicFirstMatchSecondRow>::new();
                for terminal in terminals {
                    let mut exact_end_tokens = exact_by_terminal.remove(&terminal).unwrap_or_default();
                    exact_end_tokens.sort_unstable();
                    exact_end_tokens.dedup();
                    let children = children_by_terminal.remove(&terminal).unwrap_or_default();
                    let (post_rows, next_rows) = if children.is_empty() {
                        (Vec::new(), Vec::new())
                    } else {
                        build_dynamic_reset_effect_rows(
                            &self.tokenizer,
                            children,
                            6,
                            &mut unresolved,
                        )
                    };
                    if !exact_end_tokens.is_empty() || !post_rows.is_empty() || !next_rows.is_empty()
                    {
                        root_rows.push(DynamicFirstMatchSecondRow {
                            terminal,
                            exact_end_tokens: Arc::from(exact_end_tokens),
                            post_rows: Arc::from(post_rows),
                            next_rows: Arc::from(next_rows),
                        });
                    }
                }
                root_rows.sort_unstable_by_key(|row| row.terminal);
                let unknown_tokens = unresolved.into_iter().collect::<Vec<_>>();

                let mut unknown_mask = vec![0u32; self.body_mask_len()];
                for &token_id in &unknown_tokens {
                    let word = token_id as usize / 32;
                    if let Some(bits) = unknown_mask.get_mut(word) {
                        *bits |= 1u32 << (token_id % 32);
                    }
                }
                let ordered = vocab.trie.all_subtree_tokens();
                let mut unknown_prefix = Vec::<u32>::with_capacity(ordered.len() + 1);
                unknown_prefix.push(0);
                let mut unknown_count = 0u32;
                for &canonical_token in ordered {
                    let is_unknown = vocab.token_ids(canonical_token).is_some_and(|aliases| {
                        aliases.iter().any(|&token_id| {
                            let word = token_id as usize / 32;
                            let bit = token_id % 32;
                            unknown_mask
                                .get(word)
                                .is_some_and(|bits| bits & (1u32 << bit) != 0)
                        })
                    });
                    unknown_count += u32::from(is_unknown);
                    unknown_prefix.push(unknown_count);
                }
                let mut unknown_subtrees =
                    vec![0u64; vocab.trie.node_count().div_ceil(64)];
                for node in 0..vocab.trie.node_count() as u32 {
                    let range = vocab.trie.subtree_token_index_range(node);
                    if unknown_prefix[range.start] != unknown_prefix[range.end] {
                        unknown_subtrees[node as usize >> 6] |= 1u64 << (node & 63);
                    }
                }

                projection.root_effect_source_state = full_source_state;
                projection.root_effect_post_rows = Arc::from(root_post_rows);
                projection.root_effect_rows = Arc::from(root_rows);
                projection.root_effect_unknown_tokens = Arc::from(unknown_tokens);
                projection.root_effect_unknown_subtrees = Arc::from(unknown_subtrees);

                if std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
                    || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
                {
                    eprintln!(
                        "[glrmask/profile][dynamic_root_effect] full_source={} projection_source={} futures={} classified_tokens={} post_rows={} root_rows={} fallback_unknown={} elapsed_ms={:.3}",
                        full_source_state,
                        projection_state,
                        projection.future_terminals.len(),
                        classified_tokens,
                        projection.root_effect_post_rows.len(),
                        projection.root_effect_rows.len(),
                        projection.root_effect_unknown_tokens.len(),
                        started.elapsed().as_secs_f64() * 1000.0,
                    );
                }
            }
        }

        // Exhaustively probed small parser-relevant families are only an
        // analysis device. Collapse them by the exact vocabulary-relative
        // common-future projection they produced, retaining one canonical
        // projection for each broad useful class and aliasing duplicate source
        // states to it. Ordinary selected projections remain exact and are
        // never discarded by this pass.
        let profile_projection_classes = std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
            || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some();
        let built_count = built.len();
        let probe_count = built.iter().filter(|(_, probe)| *probe).count();
        let mut projections = Vec::<DynamicSelfLoopProjection>::new();
        let mut vocab_aliases = vec![u32::MAX; projection_tokenizer.num_states() as usize];
        let broad_min_nodes = vocab.trie.node_count().div_ceil(4);
        for (mut projection, probe) in built {
            if !probe {
                projections.push(projection);
                continue;
            }
            let common_nodes = projection
                .common_future_masks
                .iter()
                .filter(|&&mask| mask != 0)
                .count();
            if common_nodes < broad_min_nodes {
                // A narrow probe may still be valuable for median masks when
                // it proves branches dead before the first terminal match.
                // Keep only that compact certificate: retaining the full
                // per-node safe/common-future arrays for every narrow probe
                // would cost hundreds of MB on a 128k-token vocabulary.
                if projection.has_pre_match_dead_subtrees()
                    || projection.first_match_step_source_state != u32::MAX
                    || projection.root_effect_source_state != u32::MAX
                {
                    projection.safe_no_match_mask = Arc::from(Vec::<u32>::new());
                    projection.safe_subtrees = Arc::from(Vec::<u8>::new());
                    projection.source_reentry_safe_subtrees = Arc::from(Vec::<u8>::new());
                    projection.common_future_masks = Arc::from(Vec::<u64>::new());
                    projections.push(projection);
                }
                continue;
            }
            // Source-specific lexical-effect programs are exact in the full
            // tokenizer coordinate.  Do not collapse their owning projection
            // into a vocabulary/common-future alias: two source states may
            // have identical ordinary projection rows while still requiring
            // distinct root effect programs (and runtime must be able to find
            // the program by the concrete source state).
            if projection.first_match_step_source_state != u32::MAX
                || projection.root_effect_source_state != u32::MAX
            {
                projections.push(projection);
                continue;
            }
            let existing = projections.iter().position(|candidate| {
                candidate.future_terminals.as_ref() == projection.future_terminals.as_ref()
                    && candidate.common_future_masks.as_ref()
                        == projection.common_future_masks.as_ref()
            });
            if let Some(index) = existing {
                vocab_aliases[projection.source_state as usize] = index as u32;
            } else {
                projections.push(projection);
            }
        }
        if profile_projection_classes {
            eprintln!(
                "[glrmask/profile][dynamic_projection_classes] built={} probes={} retained={} vocab_aliases={}",
                built_count,
                probe_count,
                projections.len(),
                vocab_aliases.iter().filter(|&&index| index != u32::MAX).count(),
            );
        }
        (projections, vocab_aliases)
    }

    fn build_dynamic_projection_alias_h64(
        &self,
        vocab: &DynamicMaskVocab,
        projections: &[DynamicSelfLoopProjection],
    ) -> Vec<u32> {
        let tokenizer = vocab
            .mask_projection_tokenizer()
            .unwrap_or(&self.tokenizer);
        let mut aliases = vec![u32::MAX; tokenizer.num_states() as usize];
        let mut by_terminal = BTreeMap::<TerminalID, Vec<usize>>::new();
        for (index, projection) in projections.iter().enumerate() {
            if let [terminal] = projection.future_terminals.as_ref() {
                by_terminal.entry(*terminal).or_default().push(index);
            }
        }
        // Exact vocabulary-relative probing handles small boundary stencils
        // more precisely and more cheaply than a whole-tokenizer H64
        // refinement. Reserve H64 for genuinely large single-terminal
        // residual families where exhaustive vocab probing would be expensive.
        let mut single_future_counts = vec![0usize; self.parser_terminal_count() as usize];
        for state in 0..tokenizer.num_states() {
            let mut futures = tokenizer.possible_future_terminals_iter(state);
            let Some(terminal) = futures.next() else {
                continue;
            };
            if futures.next().is_none()
                && let Some(count) = single_future_counts.get_mut(terminal as usize)
            {
                *count += 1;
            }
        }
        let families = by_terminal
            .into_iter()
            .filter(|(terminal, _)| {
                single_future_counts
                    .get(*terminal as usize)
                    .copied()
                    .unwrap_or(0)
                    >= 1_024
            })
            .collect::<Vec<_>>();
        let profile = std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
            || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some();
        let results = families
            .into_par_iter()
            .map(|(terminal, indices)| {
                let started = profile.then(std::time::Instant::now);
                let classes = tokenizer.bounded_terminal_future_partition(terminal, 64);
                let mut projection_by_class = FxHashMap::<u32, u32>::default();
                for index in indices {
                    let source = projections[index].source_state as usize;
                    let class = classes.get(source).copied().unwrap_or(0);
                    if class != 0 {
                        projection_by_class.entry(class).or_insert(index as u32);
                    }
                }
                let mut mapped = Vec::<(usize, u32)>::new();
                if !projection_by_class.is_empty() {
                    mapped.reserve(classes.len() / 8);
                    for (state, &class) in classes.iter().enumerate() {
                        if let Some(&projection) = projection_by_class.get(&class) {
                            mapped.push((state, projection));
                        }
                    }
                }
                if let Some(started) = started {
                    eprintln!(
                        "[glrmask/profile][dynamic_projection_alias_h64] terminal={} family_states={} aliases={} elapsed_ms={:.3}",
                        terminal,
                        single_future_counts.get(terminal as usize).copied().unwrap_or(0),
                        mapped.len(),
                        started.elapsed().as_secs_f64() * 1000.0,
                    );
                }
                mapped
            })
            .collect::<Vec<_>>();
        for mapped in results {
            for (state, projection) in mapped {
                aliases[state] = projection;
            }
        }
        aliases
    }

    pub(crate) fn build_dynamic_terminal_observation_classes(
        &self,
    ) -> Vec<(TerminalID, Arc<[u32]>)> {
        let classes = if std::env::var_os("GLRMASK_DISABLE_EARLY_OBSERVATION_FILTER").is_some() {
            self.build_dynamic_terminal_observation_classes_filtered::<false>()
        } else {
            self.build_dynamic_terminal_observation_classes_filtered::<true>()
        };
        if std::env::var_os("GLRMASK_ASSERT_EARLY_OBSERVATION_FILTER").is_some() {
            assert_eq!(
                classes,
                self.build_dynamic_terminal_observation_classes_filtered::<false>(),
                "early future filtering changed exact terminal-observation rows",
            );
        }
        classes
    }

    // Only reorder necessary, immutable selector predicates. The historical
    // selector remains available as an exact same-binary differential oracle.
    fn build_dynamic_terminal_observation_classes_filtered<const EARLY_FUTURES: bool>(
        &self,
    ) -> Vec<(TerminalID, Arc<[u32]>)> {
        if std::env::var_os("GLRMASK_DISABLE_DYNAMIC_TERMINAL_OBSERVATION_CACHE").is_some() {
            return Vec::new();
        }
        if self.tokenizer.has_any_virtual_runtime() {
            // Virtual runtimes leave the finite physical DFA domain after
            // their physical proxy/root. An observation quotient over only raw DFA
            // states cannot certify those lazily-created virtual states.
            return Vec::new();
        }

        // Preserve the historical selector exactly on its successful path:
        // broad literal-self-loop residuals whose futures contain a terminal
        // admitted by at least one singleton LR row. This sidecar was already
        // paid for by production before parser-relative commit reuse consumed it.
        let mut singleton_rows = vec![0usize; self.parser_terminal_count() as usize];
        for row in (0..self.parser_symbol_count()).filter_map(|top| self.parser_advance_row(top)) {
            let mut terminals = row.iter();
            let Some(terminal) = terminals.next() else {
                continue;
            };
            if terminals.next().is_none()
                && let Some(count) = singleton_rows.get_mut(terminal)
            {
                *count += 1;
            }
        }
        let mut best_by_futures = BTreeMap::<Vec<TerminalID>, (usize, u32)>::new();
        for state in 0..self.tokenizer.num_states() {
            // Singleton/empty and >8-terminal futures can never participate.
            // Check that cheap metadata before scanning potentially 256 byte
            // transitions twice. Nine entries are enough to reject >8 exactly;
            // a retained 2..=8 signature is complete, ordered and unchanged.
            let early_futures = if EARLY_FUTURES {
                let futures = self.tokenizer.possible_future_terminals_iter(state)
                    .take(9).collect::<SmallVec<[TerminalID; 8]>>();
                if !(2..=8).contains(&futures.len()) {
                    continue;
                }
                Some(futures)
            } else {
                None
            };
            if self.tokenizer.transitions_from(state).count() < 100 {
                continue;
            }
            let loop_len = self.tokenizer.self_loop_bytes(state).len();
            if loop_len < 64 {
                continue;
            }
            let futures = early_futures.map(SmallVec::into_vec).unwrap_or_else(|| {
                self.tokenizer.possible_future_terminals_iter(state).collect::<Vec<_>>()
            });
            if !(2..=8).contains(&futures.len()) {
                continue;
            }
            let entry = best_by_futures
                .entry(futures)
                .or_insert((loop_len, state));
            if (loop_len, std::cmp::Reverse(state))
                > (entry.0, std::cmp::Reverse(entry.1))
            {
                *entry = (loop_len, state);
            }
        }
        let mut ranked = best_by_futures
            .into_iter()
            .map(|(futures, (loop_len, state))| (loop_len, state, futures))
            .collect::<Vec<_>>();
        ranked.sort_unstable_by(|left, right| {
            (std::cmp::Reverse(left.0), left.1, &left.2)
                .cmp(&(std::cmp::Reverse(right.0), right.1, &right.2))
        });
        ranked.truncate(2);

        let mut candidates = BTreeSet::<TerminalID>::new();
        for (_, _, futures) in ranked {
            let best = futures
                .iter()
                .copied()
                .filter_map(|terminal| {
                    let count = singleton_rows
                        .get(terminal as usize)
                        .copied()
                        .unwrap_or(0);
                    (count != 0).then_some((count, std::cmp::Reverse(terminal)))
                })
                .max()
                .map(|(_, std::cmp::Reverse(terminal))| terminal);
            if let Some(terminal) = best {
                candidates.insert(terminal);
            }
        }

        // Only when the historical selector found nothing, look for the second
        // proven shape: a high-degree mixed residual signature repeated across
        // many raw states, with one of its terminals participating in a small
        // parser row. This captures bounded/string-prefix chains such as o9818
        // without adding any selection work to the existing sidecar population.
        let mut fallback_candidate = None;
        if candidates.is_empty() {
            const SMALL_PARSER_ROW_LIMIT: usize = 8;
            const REPEATED_MIXED_FAMILY_MIN_STATES: usize = 8;
            let mut small_rows = vec![0usize; self.parser_terminal_count() as usize];
            for row in (0..self.parser_symbol_count()).filter_map(|top| self.parser_advance_row(top)) {
                let row_len = row.iter().count();
                if (1..=SMALL_PARSER_ROW_LIMIT).contains(&row_len) {
                    for terminal in row.iter() {
                        if let Some(count) = small_rows.get_mut(terminal) {
                            *count += 1;
                        }
                    }
                }
            }
            let mut repeated_mixed =
                BTreeMap::<Vec<TerminalID>, (usize, usize, u32)>::new();
            for state in 0..self.tokenizer.num_states() {
                let early_futures = if EARLY_FUTURES {
                    let futures = self.tokenizer.possible_future_terminals_iter(state)
                        .take(9).collect::<SmallVec<[TerminalID; 8]>>();
                    if !(2..=8).contains(&futures.len())
                        || !futures.iter().any(|&terminal| {
                            small_rows.get(terminal as usize).copied().unwrap_or(0) != 0
                        })
                    {
                        continue;
                    }
                    Some(futures)
                } else {
                    None
                };
                let transitions = self.tokenizer.transitions_from(state).count();
                if transitions < 100 {
                    continue;
                }
                let futures = early_futures.map(SmallVec::into_vec).unwrap_or_else(|| {
                    self.tokenizer.possible_future_terminals_iter(state).collect::<Vec<_>>()
                });
                if !(2..=8).contains(&futures.len())
                    || !futures.iter().any(|&terminal| {
                        small_rows.get(terminal as usize).copied().unwrap_or(0) != 0
                    })
                {
                    continue;
                }
                let entry = repeated_mixed
                    .entry(futures)
                    .or_insert((0, transitions, state));
                entry.0 += 1;
                if (transitions, std::cmp::Reverse(state))
                    > (entry.1, std::cmp::Reverse(entry.2))
                {
                    entry.1 = transitions;
                    entry.2 = state;
                }
            }
            let fallback = repeated_mixed
                .into_iter()
                .filter(|(_, (count, _, _))| *count >= REPEATED_MIXED_FAMILY_MIN_STATES)
                .max_by(|left, right| {
                    let (_, (left_count, left_transitions, left_state)) = left;
                    let (_, (right_count, right_transitions, right_state)) = right;
                    (*left_count, *left_transitions, std::cmp::Reverse(*left_state))
                        .cmp(&(*right_count, *right_transitions, std::cmp::Reverse(*right_state)))
                });
            if let Some((futures, _)) = fallback {
                fallback_candidate = futures
                    .iter()
                    .copied()
                    .filter_map(|terminal| {
                        let count = small_rows
                            .get(terminal as usize)
                            .copied()
                            .unwrap_or(0);
                        (count != 0).then_some((count, std::cmp::Reverse(terminal)))
                    })
                    .max()
                    .map(|(_, std::cmp::Reverse(terminal))| terminal);
                if let Some(terminal) = fallback_candidate {
                    candidates.insert(terminal);
                }
            }
        }

        let profile = std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
            || std::env::var_os("GLRMASK_PROFILE_DYNAMIC_TERMINAL_OBSERVATION_CACHE").is_some();
        candidates
            .into_par_iter()
            .filter_map(|terminal| {
                let started = std::time::Instant::now();
                let has_alias = |classes: &[u32]| {
                    let mut seen = BTreeSet::<u32>::new();
                    classes
                        .iter()
                        .copied()
                        .filter(|&class| class != 0)
                        .any(|class| !seen.insert(class))
                };
                let coordinate_classes = self
                    .tokenizer
                    .terminal_residual_coordinate_observation_partition(terminal)
                    .filter(|(_, _, useful_alias)| *useful_alias);
                let (classes, configs, rounds) = if let Some((classes, distinct, _)) = coordinate_classes {
                    (classes, distinct, 0)
                } else {
                    self.tokenizer
                        .exact_terminal_observation_partition(terminal, 100_000, 20_000_000)?
                };

                let useful = has_alias(&classes);
                if profile {
                    eprintln!(
                        "[glrmask/profile][dynamic_terminal_observation_cache_build] terminal={} singleton_rows={} fallback={} configs={} rounds={} useful={} elapsed_ms={:.3}",
                        terminal,
                        singleton_rows.get(terminal as usize).copied().unwrap_or(0),
                        fallback_candidate == Some(terminal),
                        configs,
                        rounds,
                        useful,
                        started.elapsed().as_secs_f64() * 1000.0,
                    );
                }
                useful.then_some((terminal, Arc::from(classes)))
            })
            .collect()
    }

    fn dynamic_terminal_observation_classes_enabled(
        &self,
        vocab: &DynamicMaskVocab,
    ) -> bool {
        match std::env::var("GLRMASK_DYNAMIC_TERMINAL_OBSERVATION_CLASSES") {
            Ok(value) => !matches!(value.trim(), "0" | "false" | "no" | "off"),
            // O2 already collapses the model vocabulary to the exact grammar
            // quotient. On the canonical runtime cohort this additional lexer
            // observation quotient costs build time without improving TBM, so
            // leave it off by default for grammar-quotiented constraints. An
            // explicit truthy env value remains an opt-in for experiments.
            Err(_) => !vocab.is_grammar_quotiented(),
        }
    }

    pub(crate) fn prepare_dynamic_terminal_observation_classes_for_artifact(&mut self) {
        if self.dynamic_mask_vocab.has_terminal_observation_classes() {
            return;
        }
        if !self.dynamic_terminal_observation_classes_enabled(&self.dynamic_mask_vocab) {
            return;
        }
        let classes = self.build_dynamic_terminal_observation_classes();
        self.dynamic_mask_vocab
            .set_terminal_observation_classes(classes);
    }

    pub(crate) fn prepare_dynamic_virtual_residual_mask_projections_for_artifact(&mut self) {
        if self
            .dynamic_mask_vocab
            .virtual_residual_mask_projection_parts()
            .is_some()
            || !self.tokenizer.has_any_virtual_runtime()
        {
            return;
        }
        if std::env::var("GLRMASK_DYNAMIC_TRANSFER_VIRTUAL_RESIDUAL_PROJECTIONS")
            .ok()
            .is_some_and(|value| matches!(value.trim(), "0" | "false" | "no" | "off"))
        {
            return;
        }
        let mut dynamic_mask_vocab = std::mem::take(&mut self.dynamic_mask_vocab);
        self.prepare_dynamic_virtual_residual_mask_projection(&mut dynamic_mask_vocab);
        self.dynamic_mask_vocab = dynamic_mask_vocab;
    }

    /// Preserve the exact projected-terminal proof coordinate for future mask
    /// acceleration work without making it part of ordinary runtime
    /// finalization. This used to be eagerly prepared for the retired general
    /// walker; callers should opt into the cost explicitly if/when the strict
    /// full walker learns to consume it again.
    pub(crate) fn prepare_dynamic_projected_terminal_quotients_for_artifact(&mut self) {
        if self
            .dynamic_mask_vocab
            .projected_terminal_quotients_prepared()
        {
            return;
        }
        if std::env::var("GLRMASK_DYNAMIC_PROJECTED_TERMINAL_QUOTIENTS")
            .ok()
            .is_some_and(|value| matches!(value.trim(), "0" | "false" | "no" | "off"))
        {
            self.dynamic_mask_vocab
                .set_projected_terminal_quotients(Vec::new());
            return;
        }
        let quotients = self
            .tokenizer
            .build_shared_component_terminal_projected_quotients(256);
        self.dynamic_mask_vocab
            .set_projected_terminal_quotients(quotients);
    }

    /// Materialize the exact H16/H64 observation-stability certificates that
    /// powered bounded subtree accelerators in the previous dynamic walker.
    /// Kept as dormant infrastructure for later strict-full-walk P50 work;
    /// ordinary runtime finalization deliberately does not call this today.
    pub(crate) fn prepare_dynamic_bounded_observation_sets(&mut self) {
        let observation_tokenizer = self
            .dynamic_mask_vocab
            .mask_projection_tokenizer()
            .unwrap_or(&self.tokenizer);
        let (bounded16, bounded64) =
            observation_tokenizer.precompute_bounded_observation_safe_byte_sets();
        self.dynamic_mask_vocab
            .set_bounded_observation_sets(DynamicBoundedObservationSets::from_raw(
                bounded16, bounded64,
            ));
    }

    pub(crate) fn rebuild_dynamic_runtime_caches(&mut self) {
        self.tokenizer_has_epsilon_transitions = self.tokenizer.has_epsilon_transitions();
        if !self.has_template_parser() && (self.table.unconditional_advance.len() != self.parser_symbol_count() as usize
            || self.table.unconditional_advance.iter()
                .any(|row| row.len() != self.parser_terminal_count() as usize))
        {
            self.table.rebuild_unconditional_advance_rows();
        }
        let profile = std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
            || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some();
        let total_started_at = profile.then(std::time::Instant::now);
        // `terminal_live_states` is a legacy/static derived cache. Dynamic mask
        // and commit paths do not consume it, so rebuilding it here only adds
        // dynamic compile/load latency and allocations.
        self.terminal_live_states.clear();
        let started_at = profile.then(std::time::Instant::now);
        if !self.has_template_parser() && self.table.guarded_shift_index.len() != self.parser_symbol_count() as usize {
            if self.table.has_guarded_stack_shifts() {
                self.table.rebuild_guarded_shift_index();
            } else {
                self.table.guarded_shift_index.clear();
            }
        }
        let guarded_shift_ms = started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let mut dynamic_mask_vocab = std::mem::take(&mut self.dynamic_mask_vocab);
        let small_ready_runtime = dynamic_mask_vocab.is_initialized()
            && self.tokenizer.num_states() <= 64
            && self.parser_symbol_count() <= 32
            && !self.uses_sparse_direct_regular_runtime();
        let build_vocab = || {
            let started_at = profile.then(std::time::Instant::now);
            if !dynamic_mask_vocab.is_initialized() {
                let _ = dynamic_mask_vocab.materialize_pending_source();
            }
            if !dynamic_mask_vocab.is_initialized() {
                let mut materialized = self.build_dynamic_mask_vocab();
                materialized.inherit_dynamic_lexer_metadata_from(&dynamic_mask_vocab);
                dynamic_mask_vocab = materialized;
            }
            let elapsed = started_at
                .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
            (dynamic_mask_vocab, elapsed)
        };
        let build_fast = || {
            let started_at = profile.then(std::time::Instant::now);
            let transitions = self.compute_tokenizer_fast_transitions();
            let elapsed = started_at
                .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
            (transitions, elapsed)
        };
        let build_support = || {
            let started_at = profile.then(std::time::Instant::now);
            let support = if self.uses_sparse_direct_regular_runtime() {
                self.direct_regular_automaton
                    .as_ref()
                    .map_or_else(DirectRegularTerminalSupport::default, |automaton| {
                        DirectRegularTerminalSupport::build(
                            automaton,
                            self.parser_terminal_count() as usize,
                        )
                    })
            } else {
                DirectRegularTerminalSupport::default()
            };
            let elapsed = started_at
                .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
            (support, elapsed)
        };
        let (
            ((mut dynamic_mask_vocab, dynamic_vocab_ms), (tokenizer_fast_transitions, tokenizer_fast_ms)),
            (direct_regular_terminal_support, support_ms),
        ) = if rayon::current_num_threads() == 1 || small_ready_runtime {
            ((build_vocab(), build_fast()), build_support())
        } else {
            rayon::join(|| rayon::join(build_vocab, build_fast), build_support)
        };
        dynamic_mask_vocab.set_direct_regular_terminal_support(
            direct_regular_terminal_support,
        );
        let slice_leftovers_started_at = profile.then(std::time::Instant::now);
        self.prepare_llg_slice_leftovers(&mut dynamic_mask_vocab);
        let slice_leftovers_ms = slice_leftovers_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        // Master-slice prover preparation remains available for O1 experiments
        // and for artifacts that actually carry a master trie. O2 deliberately
        // keeps its grammar quotient as the walk coordinate, so its master trie
        // accessor returns None and none of this eager prover work is charged to
        // the O2 build path.
        let o2_prepared_master = dynamic_mask_vocab.is_grammar_quotiented()
            && dynamic_mask_vocab.llg_master_trie().is_some()
            && std::env::var_os("GLRMASK_DISABLE_O2_PREPARED_MASTER_PROVERS").is_none();
        let restored_complete_master = dynamic_mask_vocab
            .has_complete_prepared_master_prover_rows(self.tokenizer.num_states() as usize);
        // Pay bounded scalar-proof setup before serving rather than during a
        // cold mask. This is a performance policy only; skipped or failed proof
        // candidates retain the exact vocabulary walk. O2 keeps its own policy.
        let ordinary_eager = self.uses_dynamic_runtime()
            && !dynamic_mask_vocab.is_grammar_quotiented()
            && std::env::var_os("GLRMASK_DISABLE_EAGER_CONTAINMENT_QUOTIENTS").is_none();
        let eager_containment_quotients = o2_prepared_master || ordinary_eager
            || std::env::var_os("GLRMASK_EXPERIMENT_EAGER_CONTAINMENT_QUOTIENTS").is_some();
        let eager_component_state_cap = ordinary_eager.then(|| {
            std::env::var("GLRMASK_EAGER_CONTAINMENT_MAX_STATES")
                .ok().and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(3_000)
        });
        if eager_containment_quotients {
            let started = std::time::Instant::now();
            let force_partition_provers =
                std::env::var_os("GLRMASK_EXPERIMENT_PREPARED_PARTITION_PROVERS").is_some();
            if force_partition_provers {
                let candidates = (0..self.tokenizer.num_terminals()).collect::<Vec<_>>();
                let quotients = self
                    .tokenizer
                    .build_terminal_projected_quotients_for_containment_candidates(&candidates);
                dynamic_mask_vocab.set_projected_terminal_quotients(quotients);
            } else if !restored_complete_master {
                let safe_plus = dynamic_mask_vocab
                    .llg_slice_by_cache_id(0)
                    .expect("safe+ slice prepared before containment quotient construction");
                dynamic_mask_vocab.prepare_runtime_projected_terminal_quotients(
                    &self.tokenizer,
                    &safe_plus.slice_token_bytes(),
                    eager_component_state_cap,
                );
            }
            if profile || std::env::var_os("GLRMASK_PROFILE_PREPARED_MASTER_PROVERS").is_some() {
                eprintln!(
                    "[glrmask/profile][eager_containment_quotients] prepared={} elapsed_ms={:.3}",
                    dynamic_mask_vocab.has_projected_terminal_quotients(),
                    started.elapsed().as_secs_f64() * 1e3,
                );
            }
            let prepare_master = o2_prepared_master
                || std::env::var_os("GLRMASK_EXPERIMENT_PREPARED_MASTER_PROVERS").is_some();
            if prepare_master {
                let proof_started = std::time::Instant::now();
                let force_master =
                    std::env::var_os("GLRMASK_EXPERIMENT_PREPARED_MASTER_PROVERS").is_some();
                let (entries, product_pairs) = if restored_complete_master && !force_master {
                    (0, 0)
                } else {
                    let include_safe_radii =
                        std::env::var_os("GLRMASK_EXPERIMENT_PREPARED_SAFE_RADII").is_some();
                    dynamic_mask_vocab.prepare_master_provers_all_sources(
                        &self.tokenizer,
                        self.tokenizer.num_states() as usize,
                        include_safe_radii,
                    )
                };
                if profile
                    || std::env::var_os("GLRMASK_PROFILE_PREPARED_MASTER_PROVERS").is_some()
                {
                    eprintln!(
                        "[glrmask/profile][prepared_master_provers] entries={} product_pairs={} restored={} elapsed_ms={:.3}",
                        entries,
                        product_pairs,
                        restored_complete_master && !force_master,
                        proof_started.elapsed().as_secs_f64() * 1e3,
                    );
                }
                let max_safe_chars = u32::from(dynamic_mask_vocab.llg_master_max_safe_chars());
                for &slice_id in &[0u32, 3u32] {
                    if let Some(slice) = dynamic_mask_vocab.llg_slice_by_cache_id(slice_id) {
                        self.tokenizer.prepare_virtual_residual_master_slice_artifacts(
                            slice.dfa().start_state(),
                            slice.dfa().class_count(),
                            slice.dfa().byte_to_class_map(),
                            slice.dfa().transition_table(),
                            slice.dfa().accepting_map(),
                            slice.dfa().can_reach_accepting_map(),
                            max_safe_chars,
                        );
                    }
                }
            }
        }
        // Mask-runtime acceleration derived solely from the compiled
        // constraint belongs to build/finalization, not to the first token.
        // In particular, symbolic/virtual residual master-slice preparation can
        // cost milliseconds on pathological schemas. Deferring it would turn
        // build work into an unaccounted first-mask latency spike and make
        // warmup semantics affect measured TBM.
        self.prepare_dynamic_mask_runtime_artifacts(&mut dynamic_mask_vocab);
        let has_dense_mask_projection =
            dynamic_mask_vocab.has_dense_mask_tokenizer_projection();
        let terminal_observation_enabled =
            self.dynamic_terminal_observation_classes_enabled(&dynamic_mask_vocab);
        let terminal_observation_started_at = profile.then(std::time::Instant::now);
        let terminal_observation_classes = if !terminal_observation_enabled {
            Vec::new()
        } else if has_dense_mask_projection {
            dynamic_mask_vocab.terminal_observation_classes_cloned()
        } else if dynamic_mask_vocab.has_terminal_observation_classes() {
            dynamic_mask_vocab.terminal_observation_classes_cloned()
        } else {
            self.build_dynamic_terminal_observation_classes()
        };
        dynamic_mask_vocab.set_terminal_observation_classes(terminal_observation_classes);
        let terminal_observation_ms = terminal_observation_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let hot_frontier_started_at = profile.then(std::time::Instant::now);
        self.direct_regular_dynamic_hot_frontiers = self
            .compute_direct_regular_dynamic_hot_frontiers(
                dynamic_mask_vocab.direct_regular_terminal_support(),
            );
        let hot_frontier_ms = hot_frontier_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let hot_frontier_count = self.direct_regular_dynamic_hot_frontiers.len();
        self.dynamic_mask_vocab = dynamic_mask_vocab;
        self.tokenizer_fast_transitions = tokenizer_fast_transitions;
        if let Some(total_started_at) = total_started_at {
            eprintln!(
                "[glrmask/profile][dynamic_runtime_finalize] guarded_shift_ms={:.3} dynamic_vocab_ms={:.3} tokenizer_fast_ms={:.3} direct_regular_support_ms={:.3} slice_leftovers_ms={:.3} terminal_observation_ms={:.3} hot_frontier_ms={:.3} hot_frontiers={} total_ms={:.3}",
                guarded_shift_ms,
                dynamic_vocab_ms,
                tokenizer_fast_ms,
                support_ms,
                slice_leftovers_ms,
                terminal_observation_ms,
                hot_frontier_ms,
                hot_frontier_count,
                total_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }

    #[inline]
    fn dynamic_mask_lcp_len(left: &[u8], right: &[u8], from: usize) -> usize {
        let max_len = left.len().min(right.len());
        let mut index = from;
        while index < max_len && left[index] == right[index] {
            index += 1;
        }
        index
    }

    fn build_dynamic_mask_trie_children(
        entries: &[(usize, &[u8])],
        parent_prefix_len: usize,
        parent_node_id: u32,
        trie: &mut DynamicMaskTrie,
    ) {
        let child_edges = Self::build_dynamic_mask_trie_child_edges(
            entries,
            parent_prefix_len,
            trie,
        );

        if !child_edges.is_empty() {
            let first_child = trie.edges.len() as u32;
            let child_len = child_edges.len() as u32;
            trie.edges.extend(child_edges);
            let parent = &mut trie.nodes[parent_node_id as usize];
            parent.first_child = first_child;
            parent.child_len = child_len;
        }
    }

    fn build_dynamic_mask_trie_child_edges(
        entries: &[(usize, &[u8])],
        parent_prefix_len: usize,
        trie: &mut DynamicMaskTrie,
    ) -> SmallVec<[DynamicMaskTrieEdge; 4]> {
        let mut child_edges = SmallVec::<[DynamicMaskTrieEdge; 4]>::new();
        let mut index = 0usize;
        while index < entries.len() {
            let group_start = index;
            let next_byte = entries[index].1[parent_prefix_len];
            index += 1;
            while index < entries.len() && entries[index].1[parent_prefix_len] == next_byte {
                index += 1;
            }
            let group = &entries[group_start..index];
            let (child, child_prefix_len) =
                Self::build_dynamic_mask_trie_node(group, parent_prefix_len, trie);
            let (byte_start, byte_len) =
                trie.push_edge_bytes(&group[0].1[parent_prefix_len..child_prefix_len]);
            child_edges.push(DynamicMaskTrieEdge {
                byte_start,
                byte_len,
                child,
            });
        }
        child_edges
    }

    fn build_dynamic_mask_trie_node(
        entries: &[(usize, &[u8])],
        parent_prefix_len: usize,
        trie: &mut DynamicMaskTrie,
    ) -> (u32, usize) {
        debug_assert!(!entries.is_empty());
        let prefix_len = Self::dynamic_mask_lcp_len(
            entries.first().expect("nonempty entries").1,
            entries.last().expect("nonempty entries").1,
            parent_prefix_len,
        );
        let has_token = entries[0].1.len() == prefix_len;
        let node_id = trie.nodes.len() as u32;
        trie.nodes.push(super::artifact::DynamicMaskTrieNode {
            token_id: has_token.then_some(entries[0].0 as u32),
            first_child: 0,
            child_len: 0,
            subtree_token_start: 0,
            subtree_token_end: 0,
            subtree_bytes: [0; 4],
            subtree_first_bytes: [0; 4],
            prefix_byte_len: 0,
            subtree_max_byte_len: 0,
        });

        let child_entries = if has_token { &entries[1..] } else { entries };
        if !child_entries.is_empty() {
            Self::build_dynamic_mask_trie_children(child_entries, prefix_len, node_id, trie);
        }

        (node_id, prefix_len)
    }

    fn build_dynamic_mask_trie(entries: &[(usize, &[u8])]) -> DynamicMaskTrie {
        let mut trie = DynamicMaskTrie::new();
        if entries.is_empty() {
            return trie;
        }

        let mut start = 0usize;
        if entries[0].1.is_empty() {
            trie.nodes[0].token_id = Some(entries[0].0 as u32);
            start = 1;
        }
        if start != entries.len() {
            Self::build_dynamic_mask_trie_children(&entries[start..], 0, 0, &mut trie);
        }

        trie.finalize_subtree_metadata();
        trie
    }

    pub(crate) fn build_dynamic_mask_trie_partitioned(
        entries: &[(usize, &[u8])],
    ) -> DynamicMaskTrie {
        // Character-type classification is substantially more expensive than
        // comparing the resulting small integer. Do it once per vocabulary
        // entry rather than O(log N) times from the sort comparator.
        let mut classified_entries = entries
            .iter()
            .map(|&(token_id, bytes)| {
                (
                    dynamic_mask_vocab_layout_class(classify_vocab_char_type(bytes), bytes),
                    token_id,
                    bytes,
                )
            })
            .collect::<Vec<_>>();
        classified_entries.sort_unstable_by(|left, right| {
            right
                .2
                .is_empty()
                .cmp(&left.2.is_empty())
                .then_with(|| left.0.cmp(&right.0))
                .then_with(|| left.2.cmp(right.2))
                .then_with(|| left.1.cmp(&right.1))
        });
        let ordered_entries = classified_entries
            .into_iter()
            .map(|(_, token_id, bytes)| (token_id, bytes))
            .collect::<Vec<_>>();
        let entries = ordered_entries.as_slice();
        let mut trie = DynamicMaskTrie::new();
        if entries.is_empty() {
            return trie;
        }

        // Empty byte strings live directly on the global root. There can be at
        // most one canonical entry after byte-string alias collapsing.
        let mut start = 0usize;
        if entries[0].1.is_empty() {
            trie.nodes[0].token_id = Some(entries[0].0 as u32);
            start = 1;
        }

        let mut partition_edges = SmallVec::<[DynamicMaskTrieEdge; 16]>::new();
        let mut index = start;
        while index < entries.len() {
            let partition = dynamic_mask_vocab_layout_class(
                classify_vocab_char_type(entries[index].1),
                entries[index].1,
            );
            let partition_start = index;
            index += 1;
            while index < entries.len()
                && dynamic_mask_vocab_layout_class(
                    classify_vocab_char_type(entries[index].1),
                    entries[index].1,
                ) == partition
            {
                index += 1;
            }
            let partition_entries = &entries[partition_start..index];

            // Structural partition node. Its incoming zero-byte edge consumes
            // no vocabulary byte; below it the existing compressed radix-trie
            // builder is used unchanged.
            let partition_node = trie.nodes.len() as u32;
            trie.nodes.push(super::artifact::DynamicMaskTrieNode {
                token_id: None,
                first_child: 0,
                child_len: 0,
                subtree_token_start: 0,
                subtree_token_end: 0,
                subtree_bytes: [0; 4],
                subtree_first_bytes: [0; 4],
                prefix_byte_len: 0,
                subtree_max_byte_len: 0,
            });
            Self::build_dynamic_mask_trie_children(
                partition_entries,
                0,
                partition_node,
                &mut trie,
            );
            let (byte_start, byte_len) = trie.push_edge_bytes(&[]);
            partition_edges.push(DynamicMaskTrieEdge {
                byte_start,
                byte_len,
                child: partition_node,
            });
        }

        if !partition_edges.is_empty() {
            let first_child = trie.edges.len() as u32;
            let child_len = partition_edges.len() as u32;
            trie.edges.extend(partition_edges);
            trie.nodes[0].first_child = first_child;
            trie.nodes[0].child_len = child_len;
        }

        trie.finalize_subtree_metadata();
        trie
    }

    pub(crate) fn table_ambiguous_actions(&self) -> Vec<TableAmbiguity> {
        self.table.ambiguous_actions()
    }

    pub(crate) fn table_has_ambiguity(&self) -> bool {
        if self.has_template_parser() { return true; }
        self.table.has_ambiguity()
    }

    pub(crate) fn terminal_display_names(&self) -> &[String] {
        &self.terminal_display_names
    }

    pub(crate) fn terminal_display_name(&self, terminal_id: TerminalID) -> Option<&str> {
        self.terminal_display_names
            .get(terminal_id as usize)
            .map(String::as_str)
    }

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

    fn direct_regular_frontier_advances(
        &self,
        parser_states: &Arc<[u32]>,
    ) -> Arc<[(TerminalID, Arc<[u32]>)]> {
        let Some(automaton) = self.direct_regular_automaton.as_ref() else {
            return Arc::from(Vec::<(TerminalID, Arc<[u32]>)>::new());
        };
        let mut seen = vec![false; automaton.states.len()];
        let mut stack = Vec::with_capacity(parser_states.len());
        for &parser_state in parser_states.iter() {
            let Some(raw_state) = parser_state.checked_sub(1) else {
                return Arc::from(Vec::<(TerminalID, Arc<[u32]>)>::new());
            };
            if raw_state as usize >= automaton.states.len() {
                return Arc::from(Vec::<(TerminalID, Arc<[u32]>)>::new());
            }
            stack.push(raw_state);
        }

        let mut targets_by_terminal = BTreeMap::<TerminalID, Vec<u32>>::new();
        while let Some(raw_state) = stack.pop() {
            let index = raw_state as usize;
            if seen[index] {
                continue;
            }
            seen[index] = true;
            let state = &automaton.states[index];
            stack.extend(state.epsilons.iter().copied());
            for (&terminal, targets) in &state.transitions {
                let entry = targets_by_terminal.entry(terminal).or_default();
                entry.extend(targets.iter().map(|target| target + 1));
            }
        }

        targets_by_terminal
            .into_iter()
            .filter_map(|(terminal, mut targets)| {
                targets.sort_unstable();
                targets.dedup();
                (!targets.is_empty()).then(|| {
                    let targets: Arc<[u32]> = if targets.as_slice() == parser_states.as_ref() {
                        Arc::clone(parser_states)
                    } else {
                        targets.into()
                    };
                    (terminal, targets)
                })
            })
            .collect::<Vec<_>>()
            .into()
    }

    fn direct_regular_dynamic_hot_frontier_summary(
        &self,
        frontier_states: Arc<[u32]>,
        advance_by_terminal: Arc<[(TerminalID, Arc<[u32]>)]>,
    ) -> DirectRegularDynamicHotFrontier {
        let mut actionable_terminals =
            crate::ds::bitset::BitSet::new(self.parser_terminal_count() as usize);
        for &(terminal, _) in advance_by_terminal.iter() {
            actionable_terminals.set(terminal as usize);
        }
        let empty_acc_frontier = ParserGSS::from_sorted_unique_single_value_stacks(
            &frontier_states,
            TerminalsDisallowed::new(),
        );
        DirectRegularDynamicHotFrontier {
            frontier_states,
            empty_acc_frontier,
            actionable_terminals,
            advance_by_terminal,
        }
    }

    fn compute_direct_regular_dynamic_hot_frontiers(
        &self,
        support: &DirectRegularTerminalSupport,
    ) -> Vec<DirectRegularDynamicHotFrontier> {
        const MIN_AUTOMATON_STATES: usize = 16_384;
        const MIN_PRIMARY_WORK: u64 = 64;
        const MIN_SECONDARY_FRONTIER_STATES: usize = 64;
        if !self.uses_sparse_direct_regular_runtime() {
            return Vec::new();
        }
        let Some(automaton) = self.direct_regular_automaton.as_ref() else {
            return Vec::new();
        };
        if automaton.states.len() < MIN_AUTOMATON_STATES {
            return Vec::new();
        }

        let mut parents = vec![Vec::<u32>::new(); automaton.states.len()];
        let mut remaining_children = Vec::<u32>::with_capacity(automaton.states.len());
        let mut queue = std::collections::VecDeque::<u32>::new();
        for (source, state) in automaton.states.iter().enumerate() {
            remaining_children.push(state.epsilons.len() as u32);
            if state.epsilons.is_empty() {
                queue.push_back(source as u32);
            }
            for &child in &state.epsilons {
                parents[child as usize].push(source as u32);
            }
        }
        let mut transition_work = vec![0u64; automaton.states.len()];
        let mut processed = 0usize;
        while let Some(raw) = queue.pop_front() {
            let state = &automaton.states[raw as usize];
            let own = state
                .transitions
                .values()
                .map(|targets| targets.len() as u64)
                .sum::<u64>();
            transition_work[raw as usize] = state.epsilons.iter().fold(own, |work, child| {
                work.saturating_add(transition_work[*child as usize])
            });
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
            return Vec::new();
        }
        let Some((primary_raw, primary_work)) = transition_work
            .iter()
            .copied()
            .enumerate()
            .max_by_key(|&(state, work)| {
                (
                    support.state_terminal_count(state as u32).unwrap_or(0),
                    work,
                    state,
                )
            })
        else {
            return Vec::new();
        };
        if primary_work < MIN_PRIMARY_WORK {
            return Vec::new();
        }
        let primary_states: Arc<[u32]> = Arc::from([primary_raw as u32 + 1]);
        let primary_advances = self.direct_regular_frontier_advances(&primary_states);
        let widest_frontier = primary_advances
            .iter()
            .map(|(_, targets)| Arc::clone(targets))
            .filter(|targets| targets.len() >= MIN_SECONDARY_FRONTIER_STATES)
            .max_by_key(|targets| targets.len());

        let mut summaries = Vec::with_capacity(2);
        summaries.push(self.direct_regular_dynamic_hot_frontier_summary(
            Arc::clone(&primary_states),
            primary_advances,
        ));
        if let Some(widest_frontier) = widest_frontier
            && widest_frontier.as_ref() != primary_states.as_ref()
        {
            let advances = self.direct_regular_frontier_advances(&widest_frontier);
            summaries.push(self.direct_regular_dynamic_hot_frontier_summary(
                widest_frontier,
                advances,
            ));
        }
        if std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
            || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
        {
            eprintln!(
                "[glrmask/profile][dynamic_hot_frontiers] primary_state={} primary_support={} primary_work={} summaries={} widths={:?}",
                primary_raw + 1,
                support.state_terminal_count(primary_raw as u32).unwrap_or(0),
                primary_work,
                summaries.len(),
                summaries
                    .iter()
                    .map(|summary| summary.frontier_states.len())
                    .collect::<Vec<_>>(),
            );
        }
        summaries
    }

    fn direct_regular_dynamic_hot_frontier_for_gss(
        &self,
        gss: &ParserGSS,
    ) -> Option<&DirectRegularDynamicHotFrontier> {
        if self.has_template_parser() { return None; }

        if self.direct_regular_dynamic_hot_frontiers.is_empty() || gss.max_depth() != 1 {
            return None;
        }
        let top_count = gss.top_value_count();
        if top_count == 1 {
            let state = gss.single_top_value()?;
            return self
                .direct_regular_dynamic_hot_frontiers
                .iter()
                .find(|summary| summary.frontier_states.as_ref() == [state]);
        }
        if let Some(lower_id) = gss.single_interface_lower_id()
            && let Some(summary) = self
                .direct_regular_dynamic_hot_frontiers
                .iter()
                .find(|summary| {
                    summary.frontier_states.len() == top_count
                        && summary.empty_acc_frontier.single_interface_lower_id() == Some(lower_id)
                })
        {
            return Some(summary);
        }
        if !self
            .direct_regular_dynamic_hot_frontiers
            .iter()
            .any(|summary| summary.frontier_states.len() == top_count)
        {
            return None;
        }
        let mut top_values = gss.peek_values();
        top_values.sort_unstable();
        self.direct_regular_dynamic_hot_frontiers
            .iter()
            .find(|summary| summary.frontier_states.as_ref() == top_values.as_slice())
    }

    fn compute_direct_regular_wide_frontier_acceptance(
        &self,
    ) -> Vec<DirectRegularWideFrontierAcceptance> {
        if self.has_template_parser() { return Vec::new(); }
        // Loaded current-format constraints can execute exact acceptance
        // directly from packed Weight ids. These summaries are only an
        // optimization over materialized Weight objects; rebuilding them would
        // defeat the packed-load path.
        if self.packed_non_dwa_weights.is_some() {
            return Vec::new();
        }
        const MIN_FRONTIER_STATES: usize = 64;
        if self.uses_dynamic_runtime() || self.table.num_rules != 0 {
            return Vec::new();
        }

        let mut seen_frontiers = FxHashMap::<Vec<u32>, usize>::default();
        let mut parts_cache = FxHashMap::<Vec<usize>, Arc<[Weight]>>::default();
        let mut summaries = Vec::<DirectRegularWideFrontierAcceptance>::new();
        for descriptor in &self.table.direct_regular_wide_frontiers {
            let action = self
                .table
                .action(descriptor.source_state, descriptor.terminal);
            let action_origin_and_states = action.and_then(|action| match action {
                Action::ReplaceShifts(targets) => {
                    Some((targets.as_ptr() as usize, targets.to_vec()))
                }
                Action::StackShifts(shifts)
                    if shifts
                        .iter()
                        .all(|shift| shift.pop == 1 && shift.pushes.len() == 1) =>
                {
                    Some((
                        shifts.as_ptr() as usize,
                        shifts.iter().map(|shift| shift.pushes[0]).collect(),
                    ))
                }
                _ => None,
            });
            if action.is_some() && action_origin_and_states.is_none() {
                continue;
            }
            let action_origin = action_origin_and_states.as_ref().map(|(origin, _)| *origin);

            let mut states = descriptor.target_states.clone();
            states.sort_unstable();
            states.dedup();
            if states.len() < MIN_FRONTIER_STATES {
                continue;
            }
            if let Some((_, mut action_states)) = action_origin_and_states {
                action_states.sort_unstable();
                action_states.dedup();
                debug_assert_eq!(
                    states,
                    action_states,
                    "direct-regular frontier descriptor drifted from the live table action",
                );
            } else if !self.uses_sparse_direct_regular_runtime() {
                continue;
            }

            if let Some(&summary_index) = seen_frontiers.get(&states) {
                if let Some(action_origin) = action_origin {
                    summaries[summary_index].action_origins.push(action_origin);
                }
                continue;
            }
            seen_frontiers.insert(states.clone(), summaries.len());

            let mut weights = Vec::<Weight>::new();
            for &state in &states {
                let label = encode_positive_label(state);
                if let Some(weight) = self
                    .parser_top_accept
                    .get(&label)
                    .or_else(|| self.parser_top_accept.get(&DEFAULT_LABEL))
                {
                    weights.push(weight.clone());
                }
                if let Some(parts) = self
                    .parser_top_accept_parts
                    .get(&label)
                    .or_else(|| self.parser_top_accept_parts.get(&DEFAULT_LABEL))
                {
                    weights.extend(parts.iter().cloned());
                }
            }
            let mut actionable_terminals =
                crate::ds::bitset::BitSet::new(self.parser_terminal_count() as usize + 1);
            for &state in &states {
                if let Some(row) = self.table.advance.get(state as usize) {
                    actionable_terminals.union_with(row);
                }
            }
            for terminal in actionable_terminals.iter_ones() {
                if let Some(weight) = self
                    .direct_regular_l1_complete_by_terminal
                    .get(&(terminal as TerminalID))
                {
                    weights.push(weight.clone());
                }
            }
            weights.sort_unstable_by_key(Weight::ptr_key);
            weights.dedup_by_key(|weight| weight.ptr_key());
            let parts_key = weights.iter().map(Weight::ptr_key).collect::<Vec<_>>();
            let acceptance_parts = if let Some(cached) = parts_cache.get(&parts_key) {
                Arc::clone(cached)
            } else {
                let parts: Arc<[Weight]> = weights.into();
                parts_cache.insert(parts_key, Arc::clone(&parts));
                parts
            };
            let state_count = states.len();
            let frontier_states: Arc<[u32]> = states.into();
            let empty_acc_frontier = ParserGSS::from_sorted_unique_single_value_stacks(
                &frontier_states,
                TerminalsDisallowed::new(),
            );
            let advance_by_terminal = self.direct_regular_frontier_advances(&frontier_states);
            summaries.push(DirectRegularWideFrontierAcceptance {
                action_origins: action_origin.into_iter().collect(),
                state_count,
                actionable_terminals,
                frontier_states,
                empty_acc_frontier,
                acceptance_parts,
                dense_by_tsid: Arc::new(DenseAcceptanceRows::default()),
                advance_by_terminal,
            });
        }
        summaries
    }

    fn compute_direct_regular_parser_state_acceptance(
        &self,
    ) -> Vec<DirectRegularParserStateAcceptance> {
        if self.has_template_parser() { return Vec::new(); }
        if self.packed_non_dwa_weights.is_some() {
            return Vec::new();
        }
        const MIN_L1_TERMINALS: usize = 64;
        if self.uses_dynamic_runtime()
            || self.table.num_rules != 0
            || self.direct_regular_l1_complete_by_terminal.is_empty()
        {
            return Vec::new();
        }

        let mut l1_terminals =
            crate::ds::bitset::BitSet::new(self.parser_terminal_count() as usize + 1);
        for &terminal in self.direct_regular_l1_complete_by_terminal.keys() {
            l1_terminals.set(terminal as usize);
        }
        let l1_count = |row: &crate::ds::bitset::BitSet| {
            row.words()
                .iter()
                .zip(l1_terminals.words())
                .map(|(left, right)| (left & right).count_ones() as usize)
                .sum::<usize>()
        };
        let max_l1_terminals = self
            .table
            .advance
            .iter()
            .map(l1_count)
            .max()
            .unwrap_or(0);
        if max_l1_terminals < MIN_L1_TERMINALS {
            return Vec::new();
        }

        let mut parts_cache = FxHashMap::<Vec<usize>, Arc<[Weight]>>::default();
        let mut summaries = Vec::new();
        for (parser_state, row) in self.table.advance.iter().enumerate() {
            if l1_count(row) != max_l1_terminals {
                continue;
            }

            let label = encode_positive_label(parser_state as u32);
            let mut weights = Vec::<Weight>::with_capacity(max_l1_terminals + 4);
            if let Some(weight) = self
                .parser_top_accept
                .get(&label)
                .or_else(|| self.parser_top_accept.get(&DEFAULT_LABEL))
            {
                weights.push(weight.clone());
            }
            if let Some(parts) = self
                .parser_top_accept_parts
                .get(&label)
                .or_else(|| self.parser_top_accept_parts.get(&DEFAULT_LABEL))
            {
                weights.extend(parts.iter().cloned());
            }
            for terminal in row.iter_ones() {
                if let Some(weight) = self
                    .direct_regular_l1_complete_by_terminal
                    .get(&(terminal as TerminalID))
                {
                    weights.push(weight.clone());
                }
            }
            weights.sort_unstable_by_key(Weight::ptr_key);
            weights.dedup_by_key(|weight| weight.ptr_key());
            let parts_key = weights.iter().map(Weight::ptr_key).collect::<Vec<_>>();
            let acceptance_parts = if let Some(cached) = parts_cache.get(&parts_key) {
                Arc::clone(cached)
            } else {
                let parts: Arc<[Weight]> = weights.into();
                parts_cache.insert(parts_key, Arc::clone(&parts));
                parts
            };
            summaries.push(DirectRegularParserStateAcceptance {
                parser_state: parser_state as u32,
                acceptance_parts,
                dense_by_tsid: Arc::new(DenseAcceptanceRows::default()),
            });
        }
        summaries
    }

    fn materialize_acceptance_parts_dense(
        acceptance_parts: &[Weight],
        dense_word_count: usize,
        tsid_count: usize,
        full_dense: &DenseWords,
        dense_cache: &DenseWeightMaskCache,
    ) -> Arc<DenseAcceptanceRows> {
        let dense_cells = tsid_count
            .checked_mul(dense_word_count)
            .expect("acceptance dense matrix size must fit usize");
        // This used to allocate one Vec per tokenizer state. Large tokenizers
        // therefore performed tens of thousands of small allocations merely to
        // materialize one direct-regular acceptance summary. Keep the same
        // exact row-major matrix in one allocation and freeze only nonempty rows
        // into the sparse runtime representation below.
        let mut by_tsid = vec![0u64; dense_cells];
        let mut row_kinds = vec![0u8; tsid_count];
        for weight in acceptance_parts {
            if weight.is_full() {
                row_kinds.fill(2);
                continue;
            }
            if weight.is_empty() {
                continue;
            }
            for (tsid_range, token_set) in weight.raw_range_values() {
                let key = Arc::as_ptr(token_set) as usize;
                let dense = dense_cache.get(&key).cloned().unwrap_or_else(|| {
                    Self::dense_words_from_internal_set_with_words(
                        token_set.as_ref(),
                        dense_word_count,
                    )
                });
                for tsid in *tsid_range.start()..=*tsid_range.end() {
                    let tsid = tsid as usize;
                    if tsid >= tsid_count {
                        continue;
                    }
                    if row_kinds[tsid] == 2 {
                        continue;
                    }
                    let start = tsid * dense_word_count;
                    let dst = &mut by_tsid[start..start + dense_word_count];
                    for (dst_word, src_word) in dst.iter_mut().zip(dense.iter()) {
                        *dst_word |= *src_word;
                    }
                    row_kinds[tsid] = 1;
                }
            }
        }

        Arc::new(DenseAcceptanceRows::new(
            dense_word_count,
            by_tsid,
            row_kinds,
            Arc::clone(full_dense),
        ))
    }

    fn materialize_direct_regular_acceptance_rows(
        acceptance_parts: &[Arc<[Weight]>],
        dense_word_count: usize,
        tsid_count: usize,
        full_dense: &DenseWords,
        dense_cache: &DenseWeightMaskCache,
    ) -> Vec<Arc<DenseAcceptanceRows>> {
        if dense_word_count == 0 {
            return (0..acceptance_parts.len())
                .map(|_| Arc::new(DenseAcceptanceRows::default()))
                .collect();
        }
        let mut materialized = FxHashMap::<usize, Arc<DenseAcceptanceRows>>::default();
        acceptance_parts
            .iter()
            .map(|parts| {
            let parts_key = Arc::as_ptr(parts) as *const Weight as usize;
            if let Some(cached) = materialized.get(&parts_key) {
                return Arc::clone(cached);
            }
            let entries = Self::materialize_acceptance_parts_dense(
                parts,
                dense_word_count,
                tsid_count,
                full_dense,
                dense_cache,
            );
            materialized.insert(parts_key, Arc::clone(&entries));
            entries
        })
        .collect()
    }

    pub(crate) fn compute_composition_reset_tokens_by_terminal(&self) -> Vec<Vec<u32>> {
        let terminal_count = self.tokenizer.num_terminals() as usize;
        let empty = || (0..terminal_count).map(|_| Vec::<u32>::new()).collect::<Vec<_>>();
        if terminal_count == 0 || self.token_bytes.is_empty() {
            return empty();
        }
        let start = self.tokenizer.start_state();
        let mut rows = self
            .token_bytes
            .par_iter()
            .fold(empty, |mut rows, (&token_id, bytes)| {
                if bytes.is_empty() {
                    return rows;
                }
                let (_, matches) = self.tokenizer.execute_summary_from_state(bytes, start);
                for (terminal, width) in matches {
                    if width == bytes.len() {
                        if let Some(row) = rows.get_mut(terminal as usize) {
                            row.push(token_id);
                        }
                    }
                }
                rows
            })
            .reduce(empty, |mut left, mut right| {
                for (left_row, right_row) in left.iter_mut().zip(&mut right) {
                    left_row.append(right_row);
                }
                left
            });
        for row in &mut rows {
            row.sort_unstable();
            row.dedup();
        }
        rows
    }

    pub(crate) fn ensure_composition_reset_tokens_by_terminal(&mut self) {
        if self.composition_reset_tokens_by_terminal.len()
            != self.tokenizer.num_terminals() as usize
        {
            self.composition_reset_tokens_by_terminal =
                self.compute_composition_reset_tokens_by_terminal();
        }
    }

    fn compute_terminal_live_states(&self) -> Vec<Vec<u32>> {
        let terminal_count = self.tokenizer.num_terminals() as usize;
        if terminal_count == 0 {
            return Vec::new();
        }
        let closures = self.tokenizer.all_singleton_epsilon_closures();
        let state_count = self.tokenizer.num_states() as usize;
        let build_state = |mut rows: Vec<Vec<u32>>, state: usize| {
            // Rows are sorted and deduplicated once after the parallel transpose.
            // Stream observations directly instead of sorting/deduplicating a
            // temporary terminal list for every runtime tokenizer state.
            for &closure_state in closures[state].iter() {
                for terminal in self.tokenizer.matched_terminals_iter(closure_state) {
                    if let Some(row) = rows.get_mut(terminal as usize) {
                        row.push(state as u32);
                    }
                }
                for terminal in self.tokenizer.possible_future_terminals_iter(closure_state) {
                    if let Some(row) = rows.get_mut(terminal as usize) {
                        row.push(state as u32);
                    }
                }
            }
            rows
        };
        let empty = || (0..terminal_count).map(|_| Vec::<u32>::new()).collect::<Vec<_>>();
        let mut rows = if rayon::current_num_threads() == 1 || state_count < 4096 {
            (0..state_count).fold(empty(), build_state)
        } else {
            (0..state_count)
                .into_par_iter()
                .fold(empty, build_state)
                .reduce(empty, |mut left, mut right| {
                    for (left_row, right_row) in left.iter_mut().zip(&mut right) {
                        left_row.append(right_row);
                    }
                    left
                })
        };
        for row in &mut rows {
            row.sort_unstable();
            row.dedup();
        }
        rows
    }

    pub(crate) fn token_mask_caches_ready(&self) -> bool {
        let count = self.internal_token_count();
        self.internal_token_buf_mask_count() == count
            && self.internal_token_buf_offsets.len() == count.saturating_add(1)
            && self.word_group_sparse_masks.len() == count.div_ceil(64)
            && self.all_tokens_buf_mask.len() == self.body_mask_len()
    }

    #[inline]
    fn internal_token_buf_mask_count(&self) -> usize {
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
    fn internal_token_buf_mask_len(&self, internal_token: usize) -> usize {
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
    fn for_each_internal_token_buf_mask_entry(
        &self,
        internal_token: usize,
        mut visit: impl FnMut(u16, u32),
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

    pub(crate) fn prebuild_token_mask_caches(&mut self) {
        self.internal_token_buf_masks = self.compute_buf_masks();
        let _ = self.rebuild_token_mask_derived_caches(false);
    }

    fn rebuild_token_mask_derived_caches(
        &mut self,
        profile: bool,
    ) -> TokenMaskCacheBuildProfile {
        let skip_load_dense_group_caches = self.packed_parser_dwa.is_some()
            && std::env::var_os("GLRMASK_SKIP_LOAD_DENSE_GROUP_CACHES").is_some();
        let skip_load_sliding_dense_caches = self.packed_parser_dwa.is_some()
            && std::env::var_os("GLRMASK_SKIP_LOAD_SLIDING_DENSE_CACHES").is_some();
        self.word_group_buf_masks = Vec::new();
        let block_started_at = profile.then(std::time::Instant::now);
        let skip_small_group_caches = std::env::var("GLRMASK_SKIP_SMALL_GROUP_MASK_CACHES")
            .map(|value| {
                let value = value.trim();
                value.is_empty() || (value != "0" && !value.eq_ignore_ascii_case("false"))
            })
            .unwrap_or(true);
        let expected_word_groups = self.internal_token_buf_mask_count().div_ceil(64);
        let prebuilt_word_blocks =
            (self.word_group_sparse_masks.len() == expected_word_groups).then(|| {
                let groups = std::mem::take(&mut self.word_group_sparse_masks);
                let total_entries = groups.iter().map(Vec::len).sum::<usize>();
                let max_entries = groups.iter().map(Vec::len).max().unwrap_or(0);
                (groups, total_entries, max_entries)
            });
        let build_word_blocks = || {
            let started = profile.then(std::time::Instant::now);
            let reused = prebuilt_word_blocks.is_some();
            let result = prebuilt_word_blocks
                .unwrap_or_else(|| self.compute_token_block_sparse_masks(64));
            let ms = if reused {
                0.0
            } else {
                started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0)
            };
            (result, ms)
        };
        let build_quad_blocks = || {
            let started = profile.then(std::time::Instant::now);
            let result = if skip_small_group_caches {
                (Vec::new(), 0, 0)
            } else {
                self.compute_token_block_sparse_masks(4)
            };
            let ms = started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
            (result, ms)
        };
        let build_byte_blocks = || {
            let started = profile.then(std::time::Instant::now);
            let result = if skip_small_group_caches {
                (Vec::new(), 0, 0)
            } else {
                self.compute_token_block_sparse_masks(8)
            };
            let ms = started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
            (result, ms)
        };
        let ((word_blocks, word_block_ms), (quad_blocks, quad_block_ms), (byte_blocks, byte_block_ms)) =
            if rayon::current_num_threads() == 1 {
                (build_word_blocks(), build_quad_blocks(), build_byte_blocks())
            } else {
                let (word, (quad, byte)) = rayon::join(
                    build_word_blocks,
                    || rayon::join(build_quad_blocks, build_byte_blocks),
                );
                (word, quad, byte)
            };
        let block_masks = (word_blocks, quad_blocks, byte_blocks);
        let (
            (word_group_sparse_masks, word_group_sparse_total_entries, word_group_sparse_max_entries),
            (quad_group_sparse_masks, _, _),
            (byte_group_sparse_masks, _, _),
        ) = block_masks;
        self.word_group_sparse_masks = word_group_sparse_masks;
        self.word_group_prefix_buf_masks = if skip_load_dense_group_caches {
            DenseBufMaskRows::default()
        } else {
            self.compute_word_group_prefix_buf_masks()
        };
        self.word_group_sparse_prefix_entries =
            Self::compute_sparse_entry_prefix(&self.word_group_sparse_masks);
        self.quad_group_sparse_masks = quad_group_sparse_masks;
        self.byte_group_sparse_masks = byte_group_sparse_masks;
        let mask_words = self.body_mask_len();
        let (quad_group_dense_masks, byte_group_dense_masks) =
            if rayon::current_num_threads() == 1 {
                (
                    Self::compute_heavy_group_dense_masks(
                        &self.quad_group_sparse_masks,
                        mask_words,
                    ),
                    Self::compute_heavy_group_dense_masks(
                        &self.byte_group_sparse_masks,
                        mask_words,
                    ),
                )
            } else {
                rayon::join(
                    || {
                        Self::compute_heavy_group_dense_masks(
                            &self.quad_group_sparse_masks,
                            mask_words,
                        )
                    },
                    || {
                        Self::compute_heavy_group_dense_masks(
                            &self.byte_group_sparse_masks,
                            mask_words,
                        )
                    },
                )
            };
        self.quad_group_dense_masks = quad_group_dense_masks;
        self.byte_group_dense_masks = byte_group_dense_masks;
        self.word_group_sparse_total_entries = word_group_sparse_total_entries;
        self.word_group_sparse_max_entries = word_group_sparse_max_entries;
        let block_ms = block_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let derived_started_at = profile.then(std::time::Instant::now);
        let build_sliding = |len: usize| {
            let started = profile.then(std::time::Instant::now);
            let result = if skip_load_dense_group_caches || skip_load_sliding_dense_caches {
                DenseBufMaskRows::default()
            } else {
                self.compute_sliding_word_group_dense_masks(len)
            };
            let ms = started
                .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
            (result, ms)
        };
        let (
            ((pair_word_group_buf_masks, pair_ms), (quad_word_group_buf_masks, quad_ms)),
            (
                (super_word_group_buf_masks, super_ms),
                ((mega_word_group_buf_masks, mega_ms), (giga_word_group_buf_masks, giga_ms)),
            ),
        ) = if rayon::current_num_threads() == 1 {
            (
                (build_sliding(2), build_sliding(4)),
                (build_sliding(8), (build_sliding(16), build_sliding(32))),
            )
        } else {
            rayon::join(
                || rayon::join(|| build_sliding(2), || build_sliding(4)),
                || {
                    rayon::join(
                        || build_sliding(8),
                        || rayon::join(|| build_sliding(16), || build_sliding(32)),
                    )
                },
            )
        };
        self.pair_word_group_buf_masks = pair_word_group_buf_masks;
        self.quad_word_group_buf_masks = quad_word_group_buf_masks;
        self.super_word_group_buf_masks = super_word_group_buf_masks;
        self.mega_word_group_buf_masks = mega_word_group_buf_masks;
        self.giga_word_group_buf_masks = giga_word_group_buf_masks;
        let derived_piece_started_at = profile.then(std::time::Instant::now);
        self.all_tokens_buf_mask = self.compute_all_tokens_buf_mask();
        let all_tokens_ms = derived_piece_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let derived_piece_started_at = profile.then(std::time::Instant::now);
        self.heavy_token_dense_masks = self.compute_heavy_token_dense_masks();
        let heavy_ms = derived_piece_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let derived_piece_started_at = profile.then(std::time::Instant::now);
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
        let flat_ms = if flat_ready {
            0.0
        } else {
            derived_piece_started_at
                .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0)
        };
        let derived_piece_started_at = profile.then(std::time::Instant::now);
        self.total_internal_buf_cost = Self::compute_total_internal_buf_cost(
            &self.internal_token_buf_offsets,
            &self.heavy_token_dense_masks,
            self.body_mask_len(),
        );

        // Precompute heavy token stats for fast path decision in convert.
        let buf_len = self.body_mask_len();
        let n_internal = if self.internal_token_buf_offsets.len() > 1 {
            self.internal_token_buf_offsets.len() - 1
        } else {
            0
        };
        self.heavy_token_indices = self.heavy_token_dense_masks.iter().enumerate().filter_map(|(i, m)| if m.is_some() { Some(i) } else { None }).collect();
        self.heavy_total_cost = self.heavy_token_indices.len() * buf_len;
        self.internal_token_buf_op_costs = Self::compute_internal_token_buf_op_costs(
            &self.internal_token_buf_offsets,
            &self.heavy_token_dense_masks,
            buf_len,
        );
        self.word_group_buf_op_costs =
            Self::compute_word_group_buf_op_costs(&self.internal_token_buf_op_costs);
        let costs_ms = derived_piece_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let n_light = n_internal.saturating_sub(self.heavy_token_indices.len());
        let light_total = self.total_internal_buf_cost.saturating_sub(self.heavy_total_cost);
        self.light_avg_cost_x256 = if n_light > 0 { (light_total * 256) / n_light } else { 0 };
        let derived_ms = derived_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

        TokenMaskCacheBuildProfile {
            word_block_ms,
            quad_block_ms,
            byte_block_ms,
            block_ms,
            pair_ms,
            quad_ms,
            super_ms,
            mega_ms,
            giga_ms,
            all_tokens_ms,
            heavy_ms,
            flat_ms,
            costs_ms,
            derived_ms,
        }
    }

    fn compute_scoped_ignore_runtime_tokens(
        &self,
    ) -> (
        Vec<(TerminalID, Box<[u32]>)>,
        Vec<(TerminalID, Box<[(u32, u32)]>)>,
    ) {
        if self.static_dynamic_overlay.is_none() || self.parser_skip_terminals().is_empty() {
            return (Vec::new(), Vec::new());
        }

        let special_tokens = self
            .special_token_terminals
            .iter()
            .map(|special| special.token_id)
            .collect::<std::collections::BTreeSet<_>>();
        let mut tokens_by_bytes = FxHashMap::<&[u8], SmallVec<[u32; 2]>>::default();
        for (&token, bytes) in self.token_bytes.iter() {
            // A suffix is replayed as ordinary bytes inside a larger model
            // token. Do not use an exact-special-token-only identity as the
            // witness for that byte suffix.
            if !special_tokens.contains(&token) {
                tokens_by_bytes.entry(bytes.as_slice()).or_default().push(token);
            }
        }

        let rows = self
            .table
            .skip_terminals
            .par_iter()
            .filter_map(|&terminal| {
                let expr = self.tokenizer.terminal_expr(terminal)?;
                let dfa = crate::automata::lexer::compile::compile_terminal_expr_dfa(expr);
                if dfa.has_epsilon_transitions() || dfa.finalizers(0).contains(0) {
                    return None;
                }

                let mut tokens = Vec::<u32>::new();
                let mut fusions = Vec::<(u32, u32)>::new();
                for (&token, bytes) in self.token_bytes.iter() {
                    if bytes.is_empty() {
                        continue;
                    }
                    let mut states = SmallVec::<[u32; 8]>::new();
                    states.push(0);
                    let mut valid_end = false;
                    for (index, &byte) in bytes.iter().enumerate() {
                        let mut next = SmallVec::<[u32; 8]>::new();
                        for &state in &states {
                            if let Some(target) = dfa.step(state, byte)
                                && !next.contains(&target)
                            {
                                next.push(target);
                            }
                        }
                        if next.is_empty() {
                            valid_end = false;
                            states.clear();
                            break;
                        }
                        let completed_here = next
                            .iter()
                            .any(|&state| dfa.finalizers(state).contains(0));
                        valid_end = completed_here
                            || next.iter().any(|&state| {
                                dfa.possible_future_group_ids(state).contains(0)
                            });

                        let prefix_end = index + 1;
                        if completed_here && prefix_end < bytes.len() {
                            let suffix_bytes = &bytes[prefix_end..];
                            if let Some(suffix_tokens) = tokens_by_bytes.get(suffix_bytes) {
                                // This is deliberately a permissive candidate
                                // relation, not an acceptance proof. Runtime
                                // correlates the suffix with the exact parser
                                // branch and then sends the fused token through
                                // the exact dynamic recognizer before admitting
                                // it. Keeping unfinished suffixes here is what
                                // recovers model tokens such as " (" or " &&\n".
                                fusions.extend(
                                    suffix_tokens
                                        .iter()
                                        .copied()
                                        .map(|suffix| (token, suffix)),
                                );
                            }
                            if !next.contains(&0) {
                                // The completed Skip may reset before the next
                                // byte, while the current accepting DFA state
                                // remains live for a possible longer match.
                                next.push(0);
                            }
                        }
                        states = next;
                    }
                    if valid_end && !states.is_empty() {
                        tokens.push(token);
                    }
                }
                tokens.sort_unstable();
                tokens.dedup();
                fusions.sort_unstable_by_key(|&(fused, suffix)| {
                    (
                        self.token_bytes.get(&fused).map_or(usize::MAX, Vec::len),
                        fused,
                        suffix,
                    )
                });
                fusions.dedup();
                Some((
                    terminal,
                    tokens.into_boxed_slice(),
                    fusions.into_boxed_slice(),
                ))
            })
            .collect::<Vec<_>>();

        let mut tokens = Vec::with_capacity(rows.len());
        let mut fusions = Vec::with_capacity(rows.len());
        for (terminal, token_row, fusion_row) in rows {
            if !token_row.is_empty() {
                tokens.push((terminal, token_row));
            }
            if !fusion_row.is_empty() {
                fusions.push((terminal, fusion_row));
            }
        }
        (tokens, fusions)
    }

    /// Materialize non-DWA weights retained in the compact current-artifact
    /// pool. Compiler transformations still have a few mutable/map-oriented
    /// consumers, so one-time composition preparation reconstructs those maps
    /// rather than mixing packed and materialized sources of truth.
    pub(crate) fn materialize_non_dwa_weights_for_compilation(&mut self) -> Result<(), String> {
        let Some(packed) = self.packed_non_dwa_weights.take() else {
            return Ok(());
        };
        let weights = crate::ds::weight::unpack_pooled_weights(packed.pool.packed_bytes())?;
        let weight = |id: u32| -> Result<Weight, String> {
            weights
                .get(id as usize)
                .cloned()
                .ok_or_else(|| format!("packed non-DWA Weight id {id} is out of range"))
        };

        self.parser_top_accept = packed
            .parser_top_accept
            .iter()
            .map(|(&label, &id)| Ok((label, weight(id)?)))
            .collect::<Result<_, String>>()?;
        self.parser_top_accept_parts = packed
            .parser_top_accept_parts
            .iter()
            .map(|(&label, ids)| {
                let parts = ids
                    .iter()
                    .map(|&id| weight(id))
                    .collect::<Result<Vec<_>, String>>()?;
                Ok((label, parts))
            })
            .collect::<Result<_, String>>()?;
        self.direct_regular_l1_complete_by_terminal = packed
            .direct_regular_l1_complete_by_terminal
            .iter()
            .map(|(&terminal, &id)| Ok((terminal, weight(id)?)))
            .collect::<Result<_, String>>()?;
        self.possible_matches = packed
            .possible_matches
            .iter()
            .map(|(&terminal, &id)| Ok((terminal, weight(id)?)))
            .collect::<Result<_, String>>()?;
        self.serialized_artifact_cache = None;
        Ok(())
    }

    /// Compiler-side escape hatch for transformations that genuinely require
    /// a mutable ordinary parser DWA. Ordinary load/mask/commit and the
    /// segmented late-binding composition path deliberately keep
    /// `packed_parser_dwa` zero-copy.
    pub(crate) fn materialize_parser_dwa_for_compilation(&mut self) -> Result<(), String> {
        if let Some(packed) = self.packed_parser_dwa.take() {
            self.parser_dwa = packed.to_dwa()?;
            self.serialized_artifact_cache = None;
            self.parser_runtime_caches_prebuilt = false;
            self.packed_dwa_token_dense_masks.clear();
            self.dwa_fast_transitions = Default::default();
            self.indexed_dag_dense_transitions.clear();
            self.indexed_dag_dense_finals.clear();
        }
        if let Some(override_weight) = self.parser_start_final_override.take() {
            let start = self.parser_dwa.start_state() as usize;
            self.parser_dwa.states_mut()[start].final_weight =
                (!override_weight.is_empty()).then_some(override_weight);
            self.serialized_artifact_cache = None;
            self.parser_runtime_caches_prebuilt = false;
            self.dwa_fast_transitions = Default::default();
            self.indexed_dag_dense_transitions.clear();
            self.indexed_dag_dense_finals.clear();
        }
        Ok(())
    }

    pub(crate) fn rebuild_scoped_ignore_runtime_tokens(&mut self) {
        // This cache is consumed only by the opt-in exact scoped-ignore mask
        // overlay. Building it eagerly can require a vocabulary-wide fusion
        // scan (tens of milliseconds on selected10) even when that runtime
        // path is disabled. Keep ordinary/static constraints at zero cost.
        if self.static_dynamic_overlay.is_none()
            || std::env::var_os("GLRMASK_EXPERIMENT_SCOPED_IGNORE_EXACT_OVERLAY").is_none()
        {
            self.scoped_ignore_only_tokens.clear();
            self.scoped_ignore_prefix_fusions.clear();
            return;
        }
        let (tokens, fusions) = self.compute_scoped_ignore_runtime_tokens();
        self.scoped_ignore_only_tokens = tokens;
        self.scoped_ignore_prefix_fusions = fusions;
    }

    /// Build the expensive caches whose source of truth is the final parser
    /// DWA. Composition can do this at the final parser-union boundary; later
    /// generic finalization then reuses them instead of rescanning the same
    /// completed automaton.
    pub(crate) fn prebuild_parser_runtime_caches(&mut self) {
        debug_assert!(
            self.token_mask_caches_ready(),
            "parser runtime cache prebuild requires internal-token output masks",
        );
        self.final_mask_mapping = FinalMaskMapping::default();
        let (fast_transitions, (prebuilt_sparse, dense_words, dense_masks)) = rayon::join(
            || self.compute_fast_transitions(),
            || {
                let inventory = self.weight_token_set_inventory();
                let prebuilt_sparse = self.compute_direct_sparse_weight_token_buf_masks(
                    &inventory.final_sets,
                );
                let (dense_words, dense_masks) =
                    self.compute_dense_token_masks_excluding_direct_final(
                        &prebuilt_sparse.eligible,
                        inventory,
                    );
                (prebuilt_sparse, dense_words, dense_masks)
            },
        );
        self.internal_token_dense_words = dense_words;
        self.weight_token_dense_masks = dense_masks;
        self.packed_dwa_token_dense_masks = self.compute_packed_dwa_dense_token_masks();
        self.dwa_fast_transitions = fast_transitions;
        let (weight_token_buf_masks, weight_token_sparse_buf_masks, direct_sparse_weight_token_sets) =
            self.compute_weight_token_buf_mask_caches_with_prebuilt_sparse(prebuilt_sparse);
        self.weight_token_buf_masks = weight_token_buf_masks;
        self.weight_token_sparse_buf_masks = weight_token_sparse_buf_masks;
        self.direct_sparse_weight_token_sets = direct_sparse_weight_token_sets;
        self.parser_runtime_caches_prebuilt = true;
    }

    pub(crate) fn rebuild_runtime_caches_impl(
        &mut self,
        preserve_packed_dwa_dense_masks: bool,
        prepared_template_indices: Option<FastTemplateDfasByTerminal>,
    ) {
        let mut packed_weight_token_sets =
            crate::automata::weighted::dwa::take_packed_decode_token_set_inventory();
        self.tokenizer_has_epsilon_transitions = self.tokenizer.has_epsilon_transitions();
        if !self.has_template_parser() { self.table.rebuild_unconditional_advance_rows(); }
        let profile = std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
            || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some();
        let total_started_at = profile.then(std::time::Instant::now);
        let terminal_live_started_at = profile.then(std::time::Instant::now);
        if self.terminal_live_states.len() != self.tokenizer.num_terminals() as usize {
            self.terminal_live_states = self.compute_terminal_live_states();
        }
        let terminal_live_ms = terminal_live_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let prepare_composition_cache = std::env::var_os(
            "GLRMASK_EXPERIMENT_COMPONENT_ONLY_BUILD_DELTA_METADATA",
        )
        .is_some()
            || std::env::var_os("GLRMASK_PREPARE_COMPOSITION_CACHE").is_some();
        if prepare_composition_cache {
            let reset_started_at = profile.then(std::time::Instant::now);
            self.ensure_composition_reset_tokens_by_terminal();
            if profile {
                let token_pairs = self
                    .composition_reset_tokens_by_terminal
                    .iter()
                    .map(Vec::len)
                    .sum::<usize>();
                eprintln!(
                    "[glrmask/profile][composition_reset_tokens] terminals={} token_pairs={} ms={:.3}",
                    self.composition_reset_tokens_by_terminal
                        .iter()
                        .filter(|row| !row.is_empty())
                        .count(),
                    token_pairs,
                    reset_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
                );
            }
        }
        let scoped_ignore_started_at = profile.then(std::time::Instant::now);
        self.rebuild_scoped_ignore_runtime_tokens();
        let scoped_ignore_ms = scoped_ignore_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        if profile && !self.scoped_ignore_only_tokens.is_empty() {
            eprintln!(
                "[glrmask/profile][scoped_ignore_only_tokens] terminals={} tokens={} fusions={} ms={:.3}",
                self.scoped_ignore_only_tokens.len(),
                self.scoped_ignore_only_tokens
                    .iter()
                    .map(|(_, tokens)| tokens.len())
                    .sum::<usize>(),
                self.scoped_ignore_prefix_fusions
                    .iter()
                    .map(|(_, pairs)| pairs.len())
                    .sum::<usize>(),
                scoped_ignore_ms,
            );
        }
        if self.uses_sparse_direct_regular_runtime() {
            let support = self
                .direct_regular_automaton
                .as_ref()
                .map_or_else(DirectRegularTerminalSupport::default, |automaton| {
                    DirectRegularTerminalSupport::build(
                        automaton,
                        self.parser_terminal_count() as usize,
                    )
                });
            self.dynamic_mask_vocab
                .set_direct_regular_terminal_support(support);
        }

        let wide_frontier_started_at = profile.then(std::time::Instant::now);
        self.direct_regular_wide_frontier_acceptance =
            self.compute_direct_regular_wide_frontier_acceptance();
        self.direct_regular_parser_state_acceptance =
            self.compute_direct_regular_parser_state_acceptance();
        let wide_frontier_ms = wide_frontier_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        if profile && !self.direct_regular_wide_frontier_acceptance.is_empty() {
            eprintln!(
                "[glrmask/profile][wide_frontier_acceptance] summaries={} max_states={} ms={:.3}",
                self.direct_regular_wide_frontier_acceptance.len(),
                self.direct_regular_wide_frontier_acceptance
                    .iter()
                    .map(|summary| summary.state_count)
                    .max()
                    .unwrap_or(0),
                wide_frontier_ms,
            );
        }

        if self.tokenizer.has_virtual_residual_runtime()
            && self.dynamic_mask_vocab.mask_projection_tokenizer().is_none()
        {
            let max_token_len = self.max_token_byte_len();
            let projection_started_at = profile.then(std::time::Instant::now);
            if let Some((mask_tokenizer, projections)) =
                self.tokenizer.virtual_residuals_mask_tokenizer(max_token_len)
            {
                if profile {
                    eprintln!(
                        "[glrmask/profile][static_runtime_finalize] mask_lexer=virtual_residual components={} mask_states={} horizon={} ms={:.3}",
                        projections.len(),
                        mask_tokenizer.num_states(),
                        max_token_len,
                        projection_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
                    );
                }
                self.dynamic_mask_vocab
                    .set_virtual_residuals_mask_projection(mask_tokenizer, projections);
            }
        }
        // Ordinary static masking executes through the parser DWA, not the
        // dynamic/full-walk lexer coordinate. Do not speculatively determinize
        // every epsilon-NFA start here: the result is derived acceleration data,
        // is not serialized, and the exact epsilon-NFA path remains available
        // anywhere a dynamic full walk is explicitly requested. DynamicConstraint
        // prepares its own execution coordinate in rebuild_dynamic_runtime_caches().
        let guarded_shift_started_at = profile.then(std::time::Instant::now);
        if !self.has_template_parser() && self.table.guarded_shift_index.len() != self.parser_symbol_count() as usize {
            if self.table.num_rules == 0 {
                self.table.guarded_shift_index =
                    vec![FxHashMap::default(); self.parser_symbol_count() as usize];
            } else {
                self.table.rebuild_guarded_shift_index();
            }
        }
        let guarded_index_ms = guarded_shift_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let state_relation_started_at = profile.then(std::time::Instant::now);
        let state_count = if self.tokenizer.has_virtual_residual_runtime() {
            self.dynamic_mask_vocab
                .mask_projection_tokenizer()
                .map_or(self.tokenizer.num_states(), |tokenizer| tokenizer.num_states())
        } else {
            self.tokenizer.num_states()
        } as usize;
        let singleton_state_relation_ready = self.state_internal_tsid_offsets.as_slice()
            == [u32::MAX]
            && self.state_internal_tsids.is_empty()
            && self.state_to_internal_tsid.len() == state_count;
        let deferred_singleton_state_relation_ready = self.runtime_source_state_offset.is_none()
            && self.state_internal_tsid_offsets.is_empty()
            && self.state_internal_tsids.is_empty()
            && self.state_to_internal_tsid.len() == state_count
            && self.state_to_internal_tsid.iter().all(|&tsid| tsid != u32::MAX);
        let state_relation_ready = singleton_state_relation_ready
            || deferred_singleton_state_relation_ready
            || (self.state_internal_tsid_offsets.len() == state_count + 1
                && self
                    .state_internal_tsid_offsets
                    .last()
                    .is_some_and(|&end| end as usize == self.state_internal_tsids.len()));
        if !state_relation_ready {
            self.rebuild_state_internal_tsid_relation();
        }
        self.rebuild_runtime_product_state_lookup();
        let state_relation_ms = state_relation_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let fast_template_started_at = profile.then(std::time::Instant::now);
        // The template loader may hand back the exact indices it just built
        // from this constraint's validated immutable program. Keep those
        // objects instead of constructing and discarding a second copy.
        // Ordinary compile/mutation rebuilds supply None and remain unchanged.
        let fast_template_dfas_by_terminal = prepared_template_indices
            .unwrap_or_else(|| self.compute_fast_template_dfas());
        let fast_template_ms = fast_template_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let guarded_shift_ms = guarded_shift_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        if profile {
            eprintln!(
                "[glrmask/profile][runtime_finalize_guarded_split] guarded_index_ms={guarded_index_ms:.3} state_relation_ms={state_relation_ms:.3} fast_template_ms={fast_template_ms:.3} guarded_cells={}",
                self.table.as_lr().map_or(0, |table| table.guarded_shift_index.iter().map(|row| row.len()).sum::<usize>()),
            );
        }
        // This mapping is a derived cache. Reset it before scheduling the
        // independent cache builders so the direct sparse weight-cache branch
        // observes the same default mapping as the historical serial path.
        self.final_mask_mapping = FinalMaskMapping::default();
        // Static cache finalization never constructs the direct-dynamic
        // vocabulary. Deferred possible-match fallback materializes it lazily
        // on the first state that actually requires a dynamic mask.
        let dynamic_vocab_reused = false;
        let dynamic_vocab_ms = 0.0;
        let token_mask_caches_prebuilt = self.token_mask_caches_ready()
            || std::env::var_os("GLRMASK_SKIP_TOKEN_MASK_REBUILD_FOR_PROFILE").is_some();
        // v13+ cache artifacts can persist the portable per-internal-token
        // output fragments without persisting every derived aggregate cache.
        // Reuse those fragments independently; the remaining derived caches
        // are still rebuilt unless their own readiness invariant is satisfied.
        let internal_token_count = self.internal_token_count();
        let flat_internal_token_buf_masks_prebuilt =
            self.internal_token_buf_offsets.len() == internal_token_count.saturating_add(1)
                && self
                    .internal_token_buf_offsets
                    .last()
                    .is_some_and(|&end| end as usize == self.internal_token_buf_flat_len());
        let mut prebuilt_internal_token_buf_masks =
            (self.internal_token_buf_masks.len() == internal_token_count)
                .then(|| std::mem::take(&mut self.internal_token_buf_masks));
        let parser_runtime_caches_prebuilt = self.parser_runtime_caches_prebuilt;
        let mut prebuilt_parser_dense_masks = parser_runtime_caches_prebuilt.then(|| {
            (
                self.internal_token_dense_words,
                std::mem::take(&mut self.weight_token_dense_masks),
            )
        });
        let mut prebuilt_parser_fast_transitions = parser_runtime_caches_prebuilt
            .then(|| std::mem::take(&mut self.dwa_fast_transitions));
        let mut prebuilt_parser_weight_buf_caches = parser_runtime_caches_prebuilt.then(|| {
            (
                std::mem::take(&mut self.weight_token_buf_masks),
                std::mem::take(&mut self.weight_token_sparse_buf_masks),
                std::mem::take(&mut self.direct_sparse_weight_token_sets),
            )
        });
        let primary_started_at = profile.then(std::time::Instant::now);
        let mut prebuilt_tokenizer_fast_transitions =
            (self.tokenizer_fast_transitions.len() == self.tokenizer.num_states() as usize)
                .then(|| std::mem::take(&mut self.tokenizer_fast_transitions));
        let (
            internal_token_buf_masks,
            internal_token_buf_masks_ms,
            tokenizer_fast_transitions,
            tokenizer_fast_transitions_ms,
            (dense_mask_words, dense_masks),
            dense_token_masks_ms,
            fast_transitions,
            dwa_fast_transitions_ms,
            prebuilt_weight_caches,
            prebuilt_weight_sparse_ms,
        ) = if rayon::current_num_threads() == 1 {
            let started = profile.then(std::time::Instant::now);
            let reused_internal_token_buf_masks = prebuilt_internal_token_buf_masks.is_some()
                || flat_internal_token_buf_masks_prebuilt;
            let internal_token_buf_masks = prebuilt_internal_token_buf_masks.take().unwrap_or_else(|| {
                if flat_internal_token_buf_masks_prebuilt {
                    Vec::new()
                } else {
                    self.compute_buf_masks()
                }
            });
            let internal_token_buf_masks_ms = if reused_internal_token_buf_masks {
                0.0
            } else {
                started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0)
            };
            let started = profile.then(std::time::Instant::now);
            let parser_dense_prebuilt = prebuilt_parser_dense_masks.take();
            let (dense_masks, prebuilt_weight_caches, prebuilt_weight_sparse_ms, dense_token_masks_ms) =
                if let Some(dense_masks) = parser_dense_prebuilt {
                    (dense_masks, DirectSparseWeightBufCaches::default(), 0.0, 0.0)
                } else {
                    let weight_token_sets = self
                        .weight_token_set_inventory_with_packed(packed_weight_token_sets.take());
                    let prebuilt_weight_caches = self.compute_direct_sparse_weight_token_buf_masks(
                        &weight_token_sets.final_sets,
                    );
                    let prebuilt_weight_sparse_ms = started
                        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
                    let dense_started = profile.then(std::time::Instant::now);
                    let dense_masks = self.compute_dense_token_masks_excluding_direct_final(
                        &prebuilt_weight_caches.eligible,
                        weight_token_sets,
                    );
                    let dense_token_masks_ms = dense_started
                        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
                    (
                        dense_masks,
                        prebuilt_weight_caches,
                        prebuilt_weight_sparse_ms,
                        dense_token_masks_ms,
                    )
                };
            let started = profile.then(std::time::Instant::now);
            let reused_tokenizer_fast_transitions = prebuilt_tokenizer_fast_transitions.is_some();
            let tokenizer_fast_transitions = prebuilt_tokenizer_fast_transitions
                .take()
                .unwrap_or_else(|| self.compute_tokenizer_fast_transitions());
            let tokenizer_fast_transitions_ms = if reused_tokenizer_fast_transitions {
                0.0
            } else {
                started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0)
            };
            let started = profile.then(std::time::Instant::now);
            let reused_parser_fast_transitions = prebuilt_parser_fast_transitions.is_some();
            let fast_transitions = prebuilt_parser_fast_transitions
                .take()
                .unwrap_or_else(|| self.compute_fast_transitions());
            let dwa_fast_transitions_ms = if reused_parser_fast_transitions {
                0.0
            } else {
                started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0)
            };
            (
                internal_token_buf_masks,
                internal_token_buf_masks_ms,
                tokenizer_fast_transitions,
                tokenizer_fast_transitions_ms,
                dense_masks,
                dense_token_masks_ms,
                fast_transitions,
                dwa_fast_transitions_ms,
                prebuilt_weight_caches,
                prebuilt_weight_sparse_ms,
            )
        } else {
            let (
                ((tokenizer_fast_transitions, tokenizer_fast_transitions_ms), (fast_transitions, dwa_fast_transitions_ms)),
                (((internal_token_buf_masks, internal_token_buf_masks_ms), ((dense_mask_words, dense_masks), dense_token_masks_ms)), (prebuilt_weight_caches, prebuilt_weight_sparse_ms)),
            ) = rayon::join(
                || {
                    let build_tokenizer_fast_transitions = || {
                        let started = profile.then(std::time::Instant::now);
                        if let Some(prebuilt) = prebuilt_tokenizer_fast_transitions.take() {
                            return (prebuilt, 0.0);
                        }
                        let result = self.compute_tokenizer_fast_transitions();
                        let ms = started
                            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
                        (result, ms)
                    };
                    let build_dwa_fast_transitions = || {
                        let started = profile.then(std::time::Instant::now);
                        let reused = prebuilt_parser_fast_transitions.is_some();
                        let result = prebuilt_parser_fast_transitions
                            .take()
                            .unwrap_or_else(|| self.compute_fast_transitions());
                        let ms = if reused {
                            0.0
                        } else {
                            started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0)
                        };
                        (result, ms)
                    };
                    rayon::join(build_tokenizer_fast_transitions, build_dwa_fast_transitions)
                },
                || {
                    let started = profile.then(std::time::Instant::now);
                    let reused_internal_token_buf_masks = prebuilt_internal_token_buf_masks.is_some()
                        || flat_internal_token_buf_masks_prebuilt;
                    let internal_token_buf_masks = prebuilt_internal_token_buf_masks.take().unwrap_or_else(|| {
                        if flat_internal_token_buf_masks_prebuilt {
                            Vec::new()
                        } else {
                            self.compute_buf_masks()
                        }
                    });
                    let internal_token_buf_masks_ms = if reused_internal_token_buf_masks {
                        0.0
                    } else {
                        started.map_or(0.0, |started| {
                            started.elapsed().as_secs_f64() * 1000.0
                        })
                    };
                    let started = profile.then(std::time::Instant::now);
                    let (dense_masks, prebuilt_weight_caches, prebuilt_weight_sparse_ms, dense_token_masks_ms) =
                        if let Some(dense_masks) = prebuilt_parser_dense_masks.take() {
                            (dense_masks, DirectSparseWeightBufCaches::default(), 0.0, 0.0)
                        } else {
                            let weight_token_sets = self
                                .weight_token_set_inventory_with_packed(packed_weight_token_sets.take());
                            let prebuilt_weight_caches = self.compute_direct_sparse_weight_token_buf_masks(
                                &weight_token_sets.final_sets,
                            );
                            let prebuilt_weight_sparse_ms = started
                                .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
                            let dense_started = profile.then(std::time::Instant::now);
                            let dense_masks = self.compute_dense_token_masks_excluding_direct_final(
                                &prebuilt_weight_caches.eligible,
                                weight_token_sets,
                            );
                            let dense_token_masks_ms = dense_started
                                .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
                            (
                                dense_masks,
                                prebuilt_weight_caches,
                                prebuilt_weight_sparse_ms,
                                dense_token_masks_ms,
                            )
                        };
                    (
                        ((internal_token_buf_masks, internal_token_buf_masks_ms), (dense_masks, dense_token_masks_ms)),
                        (prebuilt_weight_caches, prebuilt_weight_sparse_ms),
                    )
                },
            );
            (
                internal_token_buf_masks,
                internal_token_buf_masks_ms,
                tokenizer_fast_transitions,
                tokenizer_fast_transitions_ms,
                (dense_mask_words, dense_masks),
                dense_token_masks_ms,
                fast_transitions,
                dwa_fast_transitions_ms,
                prebuilt_weight_caches,
                prebuilt_weight_sparse_ms,
            )
        };
        let primary_ms = primary_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        self.internal_token_buf_masks = internal_token_buf_masks;
        let token_mask_profile = if token_mask_caches_prebuilt {
            TokenMaskCacheBuildProfile::default()
        } else {
            self.rebuild_token_mask_derived_caches(profile)
        };
        let TokenMaskCacheBuildProfile {
            word_block_ms,
            quad_block_ms,
            byte_block_ms,
            block_ms,
            pair_ms,
            quad_ms,
            super_ms,
            mega_ms,
            giga_ms,
            all_tokens_ms,
            heavy_ms,
            flat_ms,
            costs_ms,
            derived_ms,
        } = token_mask_profile;

        self.token_bytes_dense = Vec::new();
        self.internal_token_dense_words = dense_mask_words;
        self.weight_token_dense_masks = dense_masks;
        if !preserve_packed_dwa_dense_masks {
            self.packed_dwa_token_dense_masks = self.compute_packed_dwa_dense_token_masks();
        }
        let full_dense = Self::dense_words_from_internal_set_with_words(
            &self.internal_token_universe(),
            self.internal_token_dense_words,
        );
        let tsid_count = self.internal_tsid_count().max(1);
        let wide_parts = self
            .direct_regular_wide_frontier_acceptance
            .iter()
            .map(|summary| Arc::clone(&summary.acceptance_parts))
            .collect::<Vec<_>>();
        let parser_parts = self
            .direct_regular_parser_state_acceptance
            .iter()
            .map(|summary| Arc::clone(&summary.acceptance_parts))
            .collect::<Vec<_>>();
        let dense_cache = &self.weight_token_dense_masks;
        let direct_wide_dense_started_at = profile.then(std::time::Instant::now);
        let direct_parser_dense_started_at = profile.then(std::time::Instant::now);
        let build_wide = || {
            Self::materialize_direct_regular_acceptance_rows(
                &wide_parts,
                self.internal_token_dense_words,
                tsid_count,
                &full_dense,
                dense_cache,
            )
        };
        let build_parser = || {
            Self::materialize_direct_regular_acceptance_rows(
                &parser_parts,
                self.internal_token_dense_words,
                tsid_count,
                &full_dense,
                dense_cache,
            )
        };
        let (wide_dense, parser_dense) = if wide_parts.is_empty() && parser_parts.is_empty() {
            (Vec::new(), Vec::new())
        } else if rayon::current_num_threads() == 1 {
            (build_wide(), build_parser())
        } else {
            rayon::join(build_wide, build_parser)
        };
        let direct_wide_dense_ms = direct_wide_dense_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let direct_parser_dense_ms = direct_parser_dense_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        for (summary, dense) in self
            .direct_regular_wide_frontier_acceptance
            .iter_mut()
            .zip(wide_dense)
        {
            summary.dense_by_tsid = dense;
        }
        for (summary, dense) in self
            .direct_regular_parser_state_acceptance
            .iter_mut()
            .zip(parser_dense)
        {
            summary.dense_by_tsid = dense;
        }
        let derived_piece_started_at = profile.then(std::time::Instant::now);
        let (
            weight_token_buf_masks,
            weight_token_sparse_buf_masks,
            direct_sparse_weight_token_sets,
        ) = prebuilt_parser_weight_buf_caches
            .take()
            .unwrap_or_else(|| {
                self.compute_weight_token_buf_mask_caches_with_prebuilt_sparse(
                    prebuilt_weight_caches,
                )
            });
        self.weight_token_buf_masks = weight_token_buf_masks;
        let weight_buf_ms = if parser_runtime_caches_prebuilt {
            0.0
        } else {
            derived_piece_started_at
                .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0)
        };
        self.weight_token_sparse_buf_masks = weight_token_sparse_buf_masks;
        self.direct_sparse_weight_token_sets = direct_sparse_weight_token_sets;
        let weight_sparse_ms = 0.0;
        self.rebuild_packed_final_mask_cache();
        self.dwa_fast_transitions = fast_transitions;
        self.parser_runtime_caches_prebuilt = true;
        let indexed_dag_dense_started_at = profile.then(std::time::Instant::now);
        let (indexed_dag_dense_transitions, indexed_dag_dense_finals) =
            self.compute_indexed_dag_dense_tables();
        self.indexed_dag_dense_transitions = indexed_dag_dense_transitions;
        self.indexed_dag_dense_finals = indexed_dag_dense_finals;
        if let Some(started) = indexed_dag_dense_started_at {
            let transitions = self
                .parser_dwa
                .states()
                .iter()
                .map(|state| state.transitions.len())
                .sum::<usize>();
            eprintln!(
                "[glrmask/profile][indexed_dag_dense_transitions] ms={:.3} states={} transitions={} internal_tsids={}",
                started.elapsed().as_secs_f64() * 1000.0,
                self.parser_dwa.states().len(),
                transitions,
                self.internal_tsid_count(),
            );
        }
        self.fast_template_dfas_by_terminal = fast_template_dfas_by_terminal;
        self.tokenizer_fast_transitions = tokenizer_fast_transitions;
        let seed_started_at = profile.then(std::time::Instant::now);
        let seed_dense_prebuilt = self.seed_universe_dense.len() == self.internal_token_dense_words
            && self
                .seed_terminal_dense
                .values()
                .all(|mask| mask.len() == self.internal_token_dense_words);
        if !seed_dense_prebuilt {
            self.build_seed_dense_masks();
        }
        let seed_ms = if seed_dense_prebuilt {
            0.0
        } else {
            seed_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0)
        };
        // The bounded tokenizer scanner used by every allocation-free commit
        // reads this constraint-level cache. Materialize it during compile/load
        // finalization rather than charging its one-time allocations to the
        // first decoding commit.
        let tokenizer_closures_started_at = profile.then(std::time::Instant::now);
        if !self.tokenizer.has_packed_runtime_metadata() {
            let tokenizer_closures = self.tokenizer.all_singleton_epsilon_closures();
            if tokenizer_closures
                .get(self.tokenizer.initial_state() as usize)
                .is_some_and(|closure| closure.len() > 64)
            {
                let _ = self.tokenizer.initial_byte_frontiers();
            }
        }
        let tokenizer_closures_ms = tokenizer_closures_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let initial_commit_prime_started_at = profile.then(std::time::Instant::now);
        // Freshly compiled constraints may still choose to pay this one-time
        // warm-up before decoding starts. A current-format disk load already
        // has an explicit latency target, and warming an otherwise lazy cache
        // is not part of reconstructing its semantics. Ordinary LR packed
        // loads retain their existing lazy policy. A table-free static parser
        // uses at most 16 real initial tokens to prepare the same shared commit
        // kernel; its measured load cost includes this work. Dynamic parsers
        // retain their existing initialization policy.
        if self.packed_parser_dwa.is_none()
            || (self.has_template_parser() && !self.uses_dynamic_runtime())
        {
            self.prime_initial_commit_hot_path();
        }
        let initial_commit_prime_ms = initial_commit_prime_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        if let Some(total_started_at) = total_started_at {
            eprintln!(
                "[glrmask/profile][runtime_finalize_derived] pair_ms={:.3} quad_ms={:.3} super_ms={:.3} mega_ms={:.3} giga_ms={:.3} all_tokens_ms={:.3} heavy_ms={:.3} flat_ms={:.3} costs_ms={:.3} direct_wide_dense_ms={:.3} direct_parser_dense_ms={:.3} prebuilt_weight_sparse_ms={:.3} weight_buf_ms={:.3} weight_sparse_ms={:.3} final_weight_sets={} final_weight_sparse_sets={} direct_sparse_weight_sets={}",
                pair_ms,
                quad_ms,
                super_ms,
                mega_ms,
                giga_ms,
                all_tokens_ms,
                heavy_ms,
                flat_ms,
                costs_ms,
                direct_wide_dense_ms,
                direct_parser_dense_ms,
                prebuilt_weight_sparse_ms,
                weight_buf_ms,
                weight_sparse_ms,
                self.weight_token_buf_masks.len(),
                self.weight_token_sparse_buf_masks.len(),
                self.direct_sparse_weight_token_sets.len(),
            );
            eprintln!(
                "[glrmask/profile][runtime_finalize] terminal_live_ms={:.3} guarded_shift_ms={:.3} dynamic_mask_vocab_ms={:.3} dynamic_mask_vocab_reused={} internal_token_buf_masks_ms={:.3} tokenizer_fast_transitions_ms={:.3} dense_token_masks_ms={:.3} dwa_fast_transitions_ms={:.3} primary_ms={:.3} word_block_masks_ms={:.3} quad_word_block_masks_ms={:.3} byte_block_masks_ms={:.3} block_masks_ms={:.3} derived_masks_ms={:.3} seed_dense_ms={:.3} tokenizer_closures_ms={:.3} initial_commit_prime_ms={:.3} total_ms={:.3}",
                terminal_live_ms,
                guarded_shift_ms,
                dynamic_vocab_ms,
                dynamic_vocab_reused,
                internal_token_buf_masks_ms,
                tokenizer_fast_transitions_ms,
                dense_token_masks_ms,
                dwa_fast_transitions_ms,
                primary_ms,
                word_block_ms,
                quad_block_ms,
                byte_block_ms,
                block_ms,
                derived_ms,
                seed_ms,
                tokenizer_closures_ms,
                initial_commit_prime_ms,
                total_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }

    fn compute_tokenizer_fast_transitions(&self) -> FastTokenizerTransitions {
        Self::compute_tokenizer_fast_transitions_for(&self.tokenizer)
    }

    fn compute_tokenizer_fast_transitions_for(tokenizer: &Tokenizer) -> FastTokenizerTransitions {
        let num_states = tokenizer.num_states();
        // A tokenizer with lazy virtual runtimes can transition from one of
        // the physical rows below into a runtime state whose id lies outside
        // `0..num_states`. The compact physical-only slabs cannot represent
        // that target (and cannot service its subsequent row), so route this
        // family through the generic tokenizer transition API instead.
        if tokenizer.has_any_virtual_runtime() {
            return FastTokenizerTransitions::Fallback(num_states as usize);
        }
        // A borrowed packed tokenizer is the canonical storage, not a reason
        // to discard a bounded runtime acceleration that fresh constraints
        // receive. The generic packed lookup is measurably slower in commit
        // scanning even though the separate dynamic mask projection is fast.
        // Restore the same exact <=8 MiB Flat16 cache for loaded and freshly
        // compiled tokenizers. Above the old 8,192-state cutoff, fresh dense
        // u32 rows were twice as large while reload had no corresponding
        // acceleration. Bound the cache by bytes, independently of parser kind.
        // Virtual states remain excluded above; larger or nonrepresentable
        // tokenizers retain the existing dense/packed fallback policy.
        const MAX_EXACT_TRANSITION_BYTES: usize = 8 * 1024 * 1024;
        if (num_states as usize) <= MAX_EXACT_TRANSITION_BYTES / (256 * std::mem::size_of::<u16>())
            && let Some(flat16) = FastTokenizerTransitions::flat16_transitions_only_for(tokenizer)
        {
            return flat16;
        }
        if tokenizer.has_packed_runtime_transitions() {
            return FastTokenizerTransitions::Fallback(num_states as usize);
        }
        let has_compressed =
            (0..num_states).any(|state| tokenizer.has_compressed_transition_state(state));
        if !has_compressed {
            let build = |state| tokenizer.transition_row(state);
            let rows = if rayon::current_num_threads() == 1 {
                (0..num_states).map(build).collect()
            } else {
                (0..num_states).into_par_iter().map(build).collect()
            };
            return FastTokenizerTransitions::Dense(rows);
        }

        let dense_states = (0..num_states)
            .filter(|&state| !tokenizer.has_compressed_transition_state(state))
            .collect::<Vec<_>>();
        let dense_rows = if rayon::current_num_threads() == 1 {
            dense_states
                .iter()
                .map(|&state| tokenizer.transition_row(state))
                .collect::<Vec<_>>()
        } else {
            dense_states
                .par_iter()
                .map(|&state| tokenizer.transition_row(state))
                .collect::<Vec<_>>()
        };
        let mut state_to_dense_row = vec![u32::MAX; num_states as usize];
        for (row, &state) in dense_states.iter().enumerate() {
            state_to_dense_row[state as usize] = row as u32;
        }
        FastTokenizerTransitions::Hybrid {
            state_to_dense_row,
            dense_rows,
        }
    }

    /// Exhaustively compare derived direct-byte transitions with the canonical
    /// tokenizer. Regression support only: never run this before a timing pass.
    #[cfg(any(test, feature = "internal-api"))]
    pub(crate) fn check_tokenizer_fast_transitions_against_uncached(&self) -> usize {
        if self.tokenizer.has_any_virtual_runtime()
            || matches!(
                &self.tokenizer_fast_transitions,
                FastTokenizerTransitions::Fallback(_)
            )
        {
            return 0;
        }
        let mut checked = 0;
        for state in 0..self.tokenizer.num_states() {
            for byte in 0..=u8::MAX {
                assert_eq!(
                    self.tokenizer_fast_transitions
                        .transition(&self.tokenizer, state, byte),
                    self.tokenizer.get_transition(state, byte),
                    "cached tokenizer state {state}, byte {byte}"
                );
                checked += 1;
            }
        }
        checked
    }

    fn compute_buf_masks(&self) -> Vec<InternalTokenBufMasks> {
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
            let mut masks = vec![Vec::<(u16, u32)>::new(); internal_token_to_tokens.len()];
            for (original, &internal) in original_token_to_internal.iter().enumerate() {
                if internal == u32::MAX {
                    continue;
                }
                let internal = internal as usize;
                let Some(mask) = masks.get_mut(internal) else {
                    continue;
                };
                let word = (original as u32 / 32) as u16;
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

    fn compute_token_block_sparse_masks(&self, block_size: usize) -> (Vec<InternalTokenBufMasks>, usize, usize) {
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
                let mut touched = Vec::<u16>::new();
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

    fn compute_heavy_group_dense_masks(
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

    fn compute_sliding_word_group_dense_masks(&self, word_group_len: usize) -> DenseBufMaskRows {
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

    fn compute_all_tokens_buf_mask(&self) -> Box<[u32]> {
        let buf_words = self.body_mask_len();
        let mut combined = vec![0u32; buf_words];
        for group in &self.word_group_sparse_masks {
            for &(word_idx, mask) in group {
                combined[word_idx as usize] |= mask;
            }
        }
        combined.into_boxed_slice()
    }

    fn compute_word_group_prefix_buf_masks(&self) -> DenseBufMaskRows {
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

    fn compute_sparse_entry_prefix(groups: &[InternalTokenBufMasks]) -> Vec<usize> {
        let mut prefix = Vec::with_capacity(groups.len() + 1);
        let mut total = 0usize;
        prefix.push(0);
        for group in groups {
            total += group.len();
            prefix.push(total);
        }
        prefix
    }

    fn direct_sparse_weight_buf_cache_enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var("GLRMASK_DIRECT_SPARSE_WEIGHT_BUF_CACHE")
                .map(|value| {
                    let value = value.trim();
                    !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
                })
                .unwrap_or(true)
        })
    }

    fn clear_sparse_buf_scratch(scratch: &mut [u32], touched: &mut Vec<u16>) {
        for word in touched.drain(..) {
            scratch[word as usize] = 0;
        }
    }

    fn build_sparse_buf_mask_from_internal_tokens(
        &self,
        internal_tokens: &RangeSetBlaze<u32>,
        scratch: &mut [u32],
        touched: &mut Vec<u16>,
    ) -> Box<[(u16, u32)]> {
        debug_assert!(touched.is_empty());
        for internal_token in internal_tokens.iter() {
            self.for_each_internal_token_buf_mask_entry(
                internal_token as usize,
                |word, mask| {
                let slot = &mut scratch[word as usize];
                if *slot == 0 {
                    touched.push(word);
                }
                *slot |= mask;
                },
            );
        }
        touched.sort_unstable();
        let sparse = touched
            .iter()
            .map(|&word| (word, scratch[word as usize]))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self::clear_sparse_buf_scratch(scratch, touched);
        sparse
    }

    fn token_set_cardinality_at_most(tokens: &RangeSetBlaze<u32>, limit: u64) -> bool {
        tokens.len() <= limit
    }

    fn direct_sparse_work_prefix(
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

    fn direct_sparse_expanded_work(tokens: &RangeSetBlaze<u32>, work_prefix: &[u64]) -> u64 {
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

    #[inline(always)]
    pub(crate) fn or_weight_token_set_to_buf_if_contained(
        &self,
        dense: &[u64],
        token_set: &Arc<RangeSetBlaze<u32>>,
        buf: &mut [u32],
    ) -> bool {
        let key = Arc::as_ptr(token_set) as usize;
        if self.direct_sparse_weight_token_sets.contains(&key) {
            return self
                .or_dense_token_set_to_buf_sparse(dense, token_set, 2048, buf)
                .unwrap_or(false);
        }
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

    #[inline(always)]
    pub(crate) fn or_dense_token_set_to_buf_sparse(
        &self,
        dense: &[u64],
        token_set: &Arc<RangeSetBlaze<u32>>,
        max_tokens: u64,
        buf: &mut [u32],
    ) -> Option<bool> {
        self.or_runtime_token_set_to_buf_sparse(
            dense,
            RuntimeTokenSetRef::Materialized(token_set),
            max_tokens,
            buf,
        )
    }

    #[inline(always)]
    pub(crate) fn or_runtime_token_set_to_buf_sparse(
        &self,
        dense: &[u64],
        token_set: RuntimeTokenSetRef<'_>,
        max_tokens: u64,
        buf: &mut [u32],
    ) -> Option<bool> {
        if dense.is_empty() || token_set.is_empty() {
            return Some(false);
        }
        let mut total = 0u64;
        match token_set {
            RuntimeTokenSetRef::Materialized(tokens) => {
                // Preserve the ordinary fast path's early exit: an oversized
                // materialized set must not scan all of its ranges first.
                for range in tokens.ranges() {
                    total = total.saturating_add(*range.end() as u64 - *range.start() as u64 + 1);
                    if total > max_tokens {
                        return None;
                    }
                }
            }
            packed => {
                let ranges = match packed {
                    RuntimeTokenSetRef::PackedDwa(tokens) => tokens.range_count() as u64,
                    RuntimeTokenSetRef::PackedPool(tokens) => tokens.range_count() as u64,
                    RuntimeTokenSetRef::Materialized(_) => unreachable!(),
                };
                // Every nonempty range contributes at least one token. Bound
                // this decoding pass without duplicating the projection loop.
                if ranges > max_tokens {
                    return None;
                }
                packed.for_each_range(|start, end| {
                    total = total.saturating_add(end as u64 - start as u64 + 1);
                });
            }
        }
        if total > max_tokens {
            return None;
        }
        let n_internal = self.internal_token_count();
        let mut any = false;
        let mut stats_entries = 0u64;
        token_set.for_each_range(|start, end| {
            let start = start as usize;
            let end = (end as usize).min(n_internal.saturating_sub(1));
            if n_internal == 0 || start > end {
                return;
            }
            for token in start..=end {
                if dense
                    .get(token / 64)
                    .is_some_and(|word| word & (1u64 << (token % 64)) != 0)
                {
                    self.or_internal_token_to_buf_fast::<false>(token, buf, &mut stats_entries);
                    any = true;
                }
            }
        });
        Some(any)
    }


    #[inline(always)]
    pub(crate) fn has_weight_token_set_buf_if_contained(
        &self,
        dense: &[u64],
        token_set: &Arc<RangeSetBlaze<u32>>,
    ) -> bool {
        let key = Arc::as_ptr(token_set) as usize;
        if self.direct_sparse_weight_token_sets.contains(&key) {
            for range in token_set.ranges() {
                let start = *range.start() as usize;
                let end = *range.end() as usize;
                for internal_token in start..=end {
                    let word = internal_token / 64;
                    let bit = internal_token % 64;
                    if dense
                        .get(word)
                        .is_none_or(|dense_word| (dense_word & (1u64 << bit)) == 0)
                    {
                        return false;
                    }
                }
            }
            return true;
        }
        if !self.weight_token_buf_masks.contains_key(&key)
            && !self.weight_token_sparse_buf_masks.contains_key(&key)
        {
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

        true
    }

    fn weight_token_set_inventory_with_packed(
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

    fn weight_token_set_inventory(&self) -> WeightTokenSetInventory {
        self.weight_token_set_inventory_with_packed(None)
    }

    /// Classify final-weight token sets for the direct runtime-intersection
    /// path. The runtime itself performs an exact intersection with the active
    /// dense state, so no final output buffer is needed for sets under its
    /// fixed work cap. Bound both the internal-token scan and the worst-case
    /// expanded output-mask work: token equivalence can make a small internal
    /// set extremely expensive to replay into the original-token mask.
    fn compute_direct_sparse_weight_token_buf_masks(
        &self,
        final_token_sets: &[(usize, Arc<RangeSetBlaze<u32>>) ],
    ) -> DirectSparseWeightBufCaches {
        if final_token_sets.is_empty() {
            return DirectSparseWeightBufCaches::default();
        }
        let profile = std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some();
        let total_started = profile.then(std::time::Instant::now);
        let buf_words = self.body_mask_len();
        let direct_sparse = buf_words != 0
            && buf_words <= u16::MAX as usize
            && Self::direct_sparse_weight_buf_cache_enabled()
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
        DirectSparseWeightBufCaches {
            eligible: batch.eligible.into_iter().collect(),
            fallback: batch.fallback,
        }
    }

    /// Build the small residual set of final-weight output buffers. All direct
    /// sparse sets have already been classified, so this path visits only the
    /// token sets that genuinely require a materialized dense or sparse output.
    fn compute_weight_token_buf_mask_caches_with_prebuilt_sparse(
        &self,
        mut prebuilt: DirectSparseWeightBufCaches,
    ) -> (
        DenseWeightBufMaskCache,
        SparseWeightBufMaskCache,
        DirectSparseWeightTokenSetCache,
    ) {
        #[derive(Default)]
        struct CacheBatch {
            dense: Vec<(usize, Box<[u32]>)>,
            sparse: Vec<(usize, Box<[(u16, u32)]>)>,
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

    fn dense_buf_to_sparse_entries(buf: &[u32]) -> Box<[(u16, u32)]> {
        buf.iter()
            .enumerate()
            .filter_map(|(idx, &word)| {
                if word == 0 {
                    None
                } else {
                    Some((idx as u16, word))
                }
            })
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }

    /// Build dense buf masks for internal tokens with many sparse entries.
    /// A token with >THRESHOLD entries benefits from a sequential 16KB scan
    /// instead of thousands of indexed read-modify-writes.
    fn compute_heavy_token_dense_masks(&self) -> Vec<Option<Box<[u32]>>> {
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
    fn compute_flat_buf_masks(
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
    fn compute_total_internal_buf_cost(
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

    fn compute_internal_token_buf_op_costs(
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

    fn compute_word_group_buf_op_costs(costs: &[usize]) -> Vec<usize> {
        costs
            .chunks(64)
            .map(|chunk| chunk.iter().copied().sum())
            .collect()
    }

    fn compute_dense_token_bytes(&self) -> Vec<Option<Box<[u8]>>> {
        let Some(max_token_id) = self.max_original_token_id() else {
            return Vec::new();
        };

        let mut dense = vec![None; max_token_id as usize + 1];
        for (token_id, bytes) in self.token_bytes_iter() {
            dense[token_id as usize] = Some(bytes.to_vec().into_boxed_slice());
        }
        dense
    }

    pub(crate) fn compute_fast_template_dfas(&self) -> FastTemplateDfasByTerminal {
        self.template_dfas_by_terminal
            .iter()
            .map(|template| {
                template
                    .as_deref()
                    .map(FastCommitTemplateDfas::from_template)
                    .map(Arc::new)
            })
            .collect()
    }

    fn compute_fast_transitions(&self) -> FastDwaTransitions {
        if self.packed_parser_dwa.is_some() {
            return FastDwaTransitions::default();
        }
        let build = |state: &crate::automata::weighted_u32::dwa::DWAState| {
            if state.transitions.is_packed() {
                return FastDwaTransitionRow::from_packed(state.transitions.clone());
            }
            let entries = state
                .transitions
                .entries()
                .map(|(label, target, weight)| (label, (target, weight.clone())));
            FastDwaTransitionRow::from_entries(entries)
        };
        if !self.parser_dwa.has_shared_transition_rows()
            || std::env::var_os("GLRMASK_EXPERIMENTAL_DISABLE_SHARED_FAST_DWA_ROWS").is_some()
        {
            let rows = if rayon::current_num_threads() == 1 {
                self.parser_dwa.states().iter().map(build).collect()
            } else {
                self.parser_dwa.states().par_iter().map(build).collect()
            };
            return FastDwaTransitions::direct(rows);
        }

        let mut row_by_ptr = FxHashMap::<usize, u32>::default();
        let mut representative_states = Vec::<usize>::new();
        let mut state_rows = Vec::<u32>::with_capacity(self.parser_dwa.states().len());
        for (state_index, state) in self.parser_dwa.states().iter().enumerate() {
            let key = state.transitions.ptr_key();
            let row = if let Some(&row) = row_by_ptr.get(&key) {
                row
            } else {
                let row = representative_states.len() as u32;
                row_by_ptr.insert(key, row);
                representative_states.push(state_index);
                row
            };
            state_rows.push(row);
        }
        let rows = if rayon::current_num_threads() == 1 {
            representative_states
                .iter()
                .map(|&state| build(&self.parser_dwa.states()[state]))
                .collect()
        } else {
            representative_states
                .par_iter()
                .map(|&state| build(&self.parser_dwa.states()[state]))
                .collect()
        };
        FastDwaTransitions::shared(rows, state_rows)
    }

    fn indexed_dag_dense_mask_for_tokens(
        &self,
        tokens: &Arc<RangeSetBlaze<u32>>,
    ) -> IndexedDagDenseMask {
        let token_key = Arc::as_ptr(tokens) as usize;
        let words = if let Some(dense) = self.weight_token_dense_masks.get(&token_key) {
            Arc::clone(dense)
        } else {
            let mut dense = vec![0u64; self.internal_token_dense_words];
            let max_token_exclusive = dense.len().saturating_mul(64);
            for range in tokens.ranges() {
                let lo = *range.start() as usize;
                if lo >= max_token_exclusive {
                    continue;
                }
                let hi = (*range.end() as usize).min(max_token_exclusive - 1);
                let word_lo = lo / 64;
                let word_hi = hi / 64;
                for word_index in word_lo..=word_hi {
                    let lo_bit = if word_index == word_lo { lo % 64 } else { 0 };
                    let hi_bit = if word_index == word_hi { hi % 64 } else { 63 };
                    let high_mask = if hi_bit == 63 {
                        !0u64
                    } else {
                        (1u64 << (hi_bit + 1)) - 1
                    };
                    let low_mask = if lo_bit == 0 {
                        0
                    } else {
                        (1u64 << lo_bit) - 1
                    };
                    dense[word_index] |= high_mask & !low_mask;
                }
            }
            dense.into()
        };
        let Some(start) = words.iter().position(|word| *word != 0) else {
            return IndexedDagDenseMask::Empty;
        };
        let end = words
            .iter()
            .rposition(|word| *word != 0)
            .expect("nonzero start implies nonzero end")
            + 1;
        IndexedDagDenseMask::Dense { words, start, end }
    }

    fn compute_indexed_dag_dense_tables(
        &self,
    ) -> (
        IndexedDagDenseTransitions,
        Vec<IndexedDagDenseTransitionMasks>,
    ) {
        if self.packed_parser_dwa.is_some() {
            return (Vec::new(), Vec::new());
        }
        // Indexed-DAG masking is opt-in. Avoid duplicating the parser DWA into
        // dense runtime tables for every ordinary constraint. Unit tests keep
        // the tables available for forced exactness checks.
        if !cfg!(test) && !crate::runtime::mask::indexed_dag_mask_enabled() {
            return (Vec::new(), Vec::new());
        }
        // Narrow transition sets are intentionally absent from the general
        // dense-weight cache. Materialize every distinct token-set pointer at
        // most once here, then share the resulting Arc and span across all DWA
        // transitions and final weights that use it.
        let mut mask_by_token_set = FxHashMap::<usize, IndexedDagDenseMask>::default();
        for state in self.parser_dwa.states() {
            for (_, weight) in state.transitions.values() {
                if weight.is_full() {
                    continue;
                }
                for (_, tokens) in weight.raw_iter() {
                    let key = Arc::as_ptr(tokens) as usize;
                    if !mask_by_token_set.contains_key(&key) {
                        let mask = self.indexed_dag_dense_mask_for_tokens(tokens);
                        mask_by_token_set.insert(key, mask);
                    }
                }
            }
            if let Some(weight) = state.final_weight.as_ref()
                && !weight.is_full()
            {
                for (_, tokens) in weight.raw_iter() {
                    let key = Arc::as_ptr(tokens) as usize;
                    if !mask_by_token_set.contains_key(&key) {
                        let mask = self.indexed_dag_dense_mask_for_tokens(tokens);
                        mask_by_token_set.insert(key, mask);
                    }
                }
            }
        }
        let build = |state: &crate::automata::weighted_u32::dwa::DWAState| {
            IndexedDagDenseTransitionRow::from_entries(state.transitions.iter().map(
                |(&label, (target, weight))| {
                    let masks = if weight.is_full() {
                        IndexedDagDenseTransitionMasks::Full
                    } else {
                        IndexedDagDenseTransitionMasks::from_entries(
                            weight.raw_iter().map(|(tsid, tokens)| {
                                (
                                    tsid,
                                    mask_by_token_set[&(Arc::as_ptr(tokens) as usize)].clone(),
                                )
                            }),
                        )
                    };
                    (
                        label,
                        IndexedDagDenseTransition {
                            target: *target,
                            masks,
                        },
                    )
                },
            ))
        };
        let transitions = if rayon::current_num_threads() == 1 {
            self.parser_dwa.states().iter().map(build).collect()
        } else {
            self.parser_dwa.states().par_iter().map(build).collect()
        };
        let finals = self
            .parser_dwa
            .states()
            .iter()
            .map(|state| match state.final_weight.as_ref() {
                None => IndexedDagDenseTransitionMasks::from_entries(std::iter::empty()),
                Some(weight) if weight.is_full() => IndexedDagDenseTransitionMasks::Full,
                Some(weight) => IndexedDagDenseTransitionMasks::from_entries(
                    weight.raw_iter().map(|(tsid, tokens)| {
                        (
                            tsid,
                            mask_by_token_set[&(Arc::as_ptr(tokens) as usize)].clone(),
                        )
                    }),
                ),
            })
            .collect();
        (transitions, finals)
    }

    // For narrow token sets, RangeSetBlaze word-span intersection is less
    // work than scanning the full dense mask. Keep dense bitmaps only for wide
    // transition sets and for residual final sets that need contained-mask IO.
    const DENSE_WEIGHT_PRECOMPUTE_MIN_WORD_SPANS: usize = 16;
    // Packed token sets are decoded directly from the artifact, so eagerly
    // densifying every set that clears the compiler-side threshold would trade
    // a large amount of load work and memory for little runtime benefit. A
    // packed intersection scans one callback per covered word span, while its
    // dense equivalent scans `internal_token_dense_words` words. Require at
    // least four dense scans worth of packed span work before paying the load
    // cost. This is representation-level work accounting, independent of any
    // parser state or benchmark corpus.
    const PACKED_DWA_DENSE_MIN_SCAN_MULTIPLIER: usize = 4;

    fn token_set_dense_word_spans_at_least(
        tokens: &RangeSetBlaze<u32>,
        dense_word_count: usize,
        threshold: usize,
    ) -> bool {
        if dense_word_count == 0 || threshold == 0 {
            return threshold == 0;
        }
        let max_token = dense_word_count.saturating_mul(64).saturating_sub(1);
        let mut count = 0usize;
        for range in tokens.ranges() {
            let start = *range.start() as usize;
            if start > max_token {
                continue;
            }
            let end = (*range.end() as usize).min(max_token);
            count = count.saturating_add(end / 64 - start / 64 + 1);
            if count >= threshold {
                return true;
            }
        }
        false
    }

    fn compute_dense_token_masks(&self) -> (usize, DenseWeightMaskCache) {
        let inventory = self.weight_token_set_inventory();
        self.compute_dense_token_masks_excluding_direct_final(
            &DirectSparseWeightTokenSetCache::default(),
            inventory,
        )
    }

    fn compute_dense_token_masks_excluding_direct_final(
        &self,
        direct_final_sets: &DirectSparseWeightTokenSetCache,
        inventory: WeightTokenSetInventory,
    ) -> (usize, DenseWeightMaskCache) {
        let profile = std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some();
        let total_started = profile.then(std::time::Instant::now);
        let internal_token_dense_words = self.internal_token_count().div_ceil(64);
        if internal_token_dense_words == 0 {
            return (0, DenseWeightMaskCache::default());
        }
        if inventory.final_sets.is_empty() && inventory.transition_sets.is_empty() {
            return (internal_token_dense_words, DenseWeightMaskCache::default());
        }

        let classify_started = profile.then(std::time::Instant::now);
        let WeightTokenSetInventory {
            final_sets,
            transition_sets: mut unique_sets,
            transition_word_spans,
        } = inventory;
        let mut residual_final_sets: DirectSparseWeightTokenSetCache = Default::default();
        for (key, token_set) in final_sets {
            if !direct_final_sets.contains(&key) {
                // These sets may need the contained-output cache, which
                // requires their dense form regardless of range width.
                residual_final_sets.insert(key);
                unique_sets.entry(key).or_insert(token_set);
            }
        }

        unique_sets.retain(|key, token_set| {
            residual_final_sets.contains(key)
                || transition_word_spans
                    .as_ref()
                    .and_then(|spans| spans.get(key))
                    .map_or_else(
                        || {
                            Self::token_set_dense_word_spans_at_least(
                                token_set,
                                internal_token_dense_words,
                                Self::DENSE_WEIGHT_PRECOMPUTE_MIN_WORD_SPANS,
                            )
                        },
                        |&spans| {
                            spans as usize >= Self::DENSE_WEIGHT_PRECOMPUTE_MIN_WORD_SPANS
                        },
                    )
        });
        let classify_ms = classify_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let build_started = profile.then(std::time::Instant::now);
        let dense_set_count = unique_sets.len();
        let build = |(key, token_set): (usize, Arc<RangeSetBlaze<u32>>)| {
            (
                key,
                Self::dense_words_from_internal_set_with_words(
                    token_set.as_ref(),
                    internal_token_dense_words,
                ),
            )
        };
        let dense_masks: DenseWeightMaskCache = if rayon::current_num_threads() == 1 {
            unique_sets.into_iter().map(build).collect()
        } else {
            unique_sets.into_par_iter().map(build).collect()
        };
        if let Some(total_started) = total_started {
            eprintln!(
                "[glrmask/profile][dense_weight_masks] sets={} words_per_set={} classify_ms={:.3} build_ms={:.3} total_ms={:.3}",
                dense_set_count,
                internal_token_dense_words,
                classify_ms,
                build_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
                total_started.elapsed().as_secs_f64() * 1000.0,
            );
        }

        (internal_token_dense_words, dense_masks)
    }

    fn compute_packed_dwa_dense_token_masks(&self) -> PackedDwaDenseWeightMaskCache {
        let Some(dwa) = self.packed_parser_dwa.as_ref() else {
            return PackedDwaDenseWeightMaskCache::default();
        };
        if self.internal_token_dense_words == 0 {
            return PackedDwaDenseWeightMaskCache::default();
        }

        let profile = std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some();
        let started = profile.then(std::time::Instant::now);
        let min_word_spans = Self::DENSE_WEIGHT_PRECOMPUTE_MIN_WORD_SPANS.max(
            self.internal_token_dense_words
                .saturating_mul(Self::PACKED_DWA_DENSE_MIN_SCAN_MULTIPLIER),
        );
        let transition_ids = dwa.transition_token_set_ids();
        let transition_count = transition_ids.len();
        let build = |id: u32| {
            let token_set = dwa.token_set(id)?;
            if (token_set.word_spans() as usize) < min_word_spans {
                return None;
            }
            Some((
                id,
                self.dense_words_from_runtime_token_set(RuntimeTokenSetRef::PackedDwa(token_set)),
            ))
        };
        let rows: Vec<(u32, DenseWords)> = if rayon::current_num_threads() == 1 {
            transition_ids.into_iter().filter_map(build).collect()
        } else {
            transition_ids.into_par_iter().filter_map(build).collect()
        };
        let result = PackedDwaDenseWeightMaskCache::from_rows(
            dwa.token_set_count(),
            self.internal_token_dense_words,
            rows,
        )
        .expect("compiler-built packed DWA dense-mask cache must be internally consistent");
        if let Some(started) = started {
            eprintln!(
                "[glrmask/profile][packed_dwa_dense_weight_masks] token_sets={} transition_sets={} cached_sets={} min_word_spans={} words_per_set={} bytes={} total_ms={:.3}",
                dwa.token_set_count(),
                transition_count,
                result.len(),
                min_word_spans,
                self.internal_token_dense_words,
                result.len().saturating_mul(self.internal_token_dense_words).saturating_mul(8),
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
        result
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

    pub(crate) fn build_dense_token_bytes(&mut self) {
        self.token_bytes_dense = self.compute_dense_token_bytes();
    }

    fn rebuild_state_internal_tsid_relation(&mut self) {
        let state_count = self.tokenizer.num_states() as usize;
        let mut relation = (0..state_count)
            .map(|_| SmallVec::<[u32; 2]>::new())
            .collect::<Vec<_>>();
        for (internal_tsid, states) in self.internal_tsid_groups().iter().enumerate() {
            for &state in states {
                if let Some(tsids) = relation.get_mut(state as usize) {
                    tsids.push(internal_tsid as u32);
                }
            }
        }
        for (state, tsids) in relation.iter_mut().enumerate() {
            if let Some(&primary) = self.state_to_internal_tsid.get(state)
                && !tsids.contains(&primary)
            {
                tsids.push(primary);
            }
            tsids.sort_unstable();
            tsids.dedup();
        }

        self.state_internal_tsid_offsets.clear();
        self.state_internal_tsid_offsets.reserve(state_count + 1);
        self.state_internal_tsids.clear();
        self.state_internal_tsid_offsets.push(0);
        for tsids in relation {
            self.state_internal_tsids.extend_from_slice(&tsids);
            self.state_internal_tsid_offsets
                .push(self.state_internal_tsids.len() as u32);
        }
    }

    fn rebuild_runtime_product_state_lookup(&mut self) {
        self.runtime_product_state_by_source_subset.clear();
        let Some(source_offset) = self.runtime_source_state_offset else {
            return;
        };
        let product_states = source_offset as usize;
        if self.runtime_product_source_offsets.len() != product_states + 1
            || self.runtime_product_exact_source_states.len() != product_states
        {
            self.runtime_source_state_offset = None;
            self.runtime_product_source_offsets.clear();
            self.runtime_product_source_states.clear();
            self.runtime_product_exact_source_states.clear();
            return;
        }

        self.runtime_product_state_by_source_subset
            .reserve(product_states);
        for product_state in 0..product_states {
            let start = self.runtime_product_source_offsets[product_state] as usize;
            let end = self.runtime_product_source_offsets[product_state + 1] as usize;
            let Some(states) = self.runtime_product_source_states.get(start..end) else {
                self.runtime_product_state_by_source_subset.clear();
                self.runtime_source_state_offset = None;
                return;
            };
            self.runtime_product_state_by_source_subset
                .insert(states.into(), product_state as u32);
        }
    }

    /// Build fast transition lookup tables from the DWA's BTreeMap transitions.
    pub(crate) fn build_fast_transitions(&mut self) {
        self.dwa_fast_transitions = self.compute_fast_transitions();
        let (indexed_dag_dense_transitions, indexed_dag_dense_finals) =
            self.compute_indexed_dag_dense_tables();
        self.indexed_dag_dense_transitions = indexed_dag_dense_transitions;
        self.indexed_dag_dense_finals = indexed_dag_dense_finals;
    }

    pub(crate) fn build_dense_token_masks(&mut self) {
        let (internal_token_dense_words, dense_masks) = self.compute_dense_token_masks();
        self.internal_token_dense_words = internal_token_dense_words;
        self.weight_token_dense_masks = dense_masks;
    }

    /// Precompute dense bitmaps for the seed phase: one bitmap per (state, terminal)
    /// pair, plus the universe bitmap. This lets seed_weight_dense use bitwise ANDNOT
    /// instead of RangeSetBlaze subtraction.
    pub(crate) fn build_seed_dense_masks(&mut self) {
        self.seed_terminal_dense_fallback
            .lock()
            .expect("seed exclusion cache poisoned")
            .clear();
        let dw = self.internal_token_dense_words;
        if dw == 0 {
            self.seed_terminal_dense.clear();
            self.seed_universe_dense = empty_dense_words();
            return;
        }

        let universe = self.internal_token_universe();
        self.seed_universe_dense = self.dense_words_from_internal_set(&universe);

        self.seed_terminal_dense = self.build_seed_terminal_dense_masks();
    }

    fn collect_weight_token_sets<'a>(
        weight: &'a Weight,
        unique_sets: &mut FxHashMap<usize, &'a RangeSetBlaze<u32>>,
    ) {
        if weight.is_full() || weight.is_empty() {
            return;
        }
        // `unique_sets` already deduplicates globally. Avoid allocating and
        // linearly deduplicating a temporary vector for every individual
        // weight before inserting the same pointers here.
        for (_tsid_range, token_set) in weight.raw_range_values() {
            let token_set = token_set.as_ref();
            let key = token_set as *const RangeSetBlaze<u32> as usize;
            unique_sets.entry(key).or_insert(token_set);
        }
    }

    fn dense_words_from_internal_set_with_words(
        internal_tokens: &RangeSetBlaze<u32>,
        dense_word_count: usize,
    ) -> DenseWords {
        let mut words = vec![0u64; dense_word_count];
        let Some(max_token) = dense_word_count.checked_mul(64).and_then(|count| count.checked_sub(1)) else {
            return Arc::from(words.into_boxed_slice());
        };

        for token_range in internal_tokens.ranges() {
            let start = *token_range.start() as usize;
            if start > max_token {
                continue;
            }
            let end = (*token_range.end() as usize).min(max_token);
            let first_word = start / 64;
            let last_word = end / 64;
            let first_bit = start % 64;
            let last_bit = end % 64;

            if first_word == last_word {
                let high_mask = if last_bit == 63 {
                    u64::MAX
                } else {
                    (1u64 << (last_bit + 1)) - 1
                };
                words[first_word] |= (u64::MAX << first_bit) & high_mask;
                continue;
            }

            words[first_word] |= u64::MAX << first_bit;
            if first_word + 1 < last_word {
                words[first_word + 1..last_word].fill(u64::MAX);
            }
            let last_mask = if last_bit == 63 {
                u64::MAX
            } else {
                (1u64 << (last_bit + 1)) - 1
            };
            words[last_word] |= last_mask;
        }
        Arc::from(words.into_boxed_slice())
    }

    fn dense_words_from_internal_set(&self, internal_tokens: &RangeSetBlaze<u32>) -> DenseWords {
        Self::dense_words_from_internal_set_with_words(internal_tokens, self.internal_token_dense_words)
    }

    fn fill_dense_words_from_runtime_token_set(
        words: &mut [u64],
        internal_tokens: RuntimeTokenSetRef<'_>,
    ) {
        let Some(max_token) = words
            .len()
            .checked_mul(64)
            .and_then(|count| count.checked_sub(1))
        else {
            return;
        };
        internal_tokens.for_each_range(|start, end| {
            let start = start as usize;
            if start > max_token {
                return;
            }
            let end = (end as usize).min(max_token);
            let first_word = start / 64;
            let last_word = end / 64;
            let first_bit = start % 64;
            let last_bit = end % 64;
            if first_word == last_word {
                let high_mask = if last_bit == 63 {
                    u64::MAX
                } else {
                    (1u64 << (last_bit + 1)) - 1
                };
                words[first_word] |= (u64::MAX << first_bit) & high_mask;
                return;
            }
            words[first_word] |= u64::MAX << first_bit;
            if first_word + 1 < last_word {
                words[first_word + 1..last_word].fill(u64::MAX);
            }
            let last_mask = if last_bit == 63 {
                u64::MAX
            } else {
                (1u64 << (last_bit + 1)) - 1
            };
            words[last_word] |= last_mask;
        });
    }

    fn dense_words_from_runtime_token_set(
        &self,
        internal_tokens: RuntimeTokenSetRef<'_>,
    ) -> DenseWords {
        let mut words = vec![0u64; self.internal_token_dense_words];
        Self::fill_dense_words_from_runtime_token_set(&mut words, internal_tokens);
        Arc::from(words.into_boxed_slice())
    }

    #[inline]
    pub(crate) fn runtime_token_set_dense_mask(
        &self,
        token_set: RuntimeTokenSetRef<'_>,
    ) -> Option<&[u64]> {
        if let Some(key) = token_set.materialized_key() {
            return self.weight_token_dense_masks.get(&key).map(AsRef::as_ref);
        }
        token_set
            .packed_id()
            .and_then(|id| self.packed_dwa_token_dense_masks.get(id))
    }

    /// Create a fresh state for one generated sequence.
    pub fn start(&self) -> ConstraintState<'_> {
        crate::runtime::initialize_hot_path_config();
        if self.tokenizer_has_epsilon_transitions && !self.tokenizer.has_packed_runtime_metadata() {
            drop(self.tokenizer.all_singleton_epsilon_closures());
        }
        let state = self.initial_state_map();
        let mut state = ConstraintState {
            terminated: false,
            constraint: self,
            state,
            buffers: Default::default(),
            generation: 0,
            mask_cache: Mutex::new(None),
            mask_scratch: Arc::new(Mutex::new(crate::runtime::state::MaskScratch::for_constraint(self))),
        };
        state.prefill_mask_cache();
        state.reserve_linear_stack_hot_path();
        state
    }

    pub(crate) fn start_dynamic(&self) -> ConstraintState<'_> {
        crate::runtime::initialize_hot_path_config();
        let mut state = ConstraintState {
            terminated: false,
            constraint: self,
            state: self.initial_state_map(),
            buffers: crate::runtime::state::CommitBuffers::for_constraint(self),
            generation: 0,
            mask_cache: Mutex::new(None),
            mask_scratch: Arc::new(Mutex::new(crate::runtime::state::MaskScratch::for_constraint(self))),
        };
        state.reserve_linear_stack_hot_path();
        state
    }

    /// Return the number of `u32` words required for a packed token mask.
    pub fn mask_len(&self) -> usize {
        self.max_original_token_id()
            .map(|token_id| (token_id as usize / 32) + 1)
            .unwrap_or(0)
    }

    pub(crate) fn body_mask_len(&self) -> usize {
        self.body_max_original_token_id()
            .map(|token_id| (token_id as usize / 32) + 1)
            .unwrap_or(0)
    }

    /// Apply generation termination without changing the compiled body.
    pub(crate) fn with_end_tokens(mut self, ids: &[u32]) -> crate::Result<Self> {
        let mut ids = ids.to_vec();
        ids.sort_unstable();
        if ids.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(crate::Error::Compilation("duplicate end-token ID".to_owned()));
        }
        self.end_tokens = Arc::from(ids);
        Ok(self)
    }


    #[inline]
    pub(crate) fn direct_regular_wide_frontier_index_for_gss(
        &self,
        gss: &ParserGSS,
    ) -> Option<usize> {
        if self.has_template_parser() { return None; }

        // The cache below can only contain indices into this table. Ordinary
        // parser runtimes may have no direct-regular wide-frontier summaries;
        // avoid materializing deferred dynamic-vocab state merely to query a
        // cache that cannot contain a valid entry.
        if self.direct_regular_wide_frontier_acceptance.is_empty() {
            return None;
        }
        let lower_id = gss.single_interface_lower_id()?;
        if let Some(index) = self
            .direct_regular_wide_frontier_acceptance
            .iter()
            .position(|summary| {
                summary.empty_acc_frontier.single_interface_lower_id() == Some(lower_id)
            })
        {
            return Some(index);
        }
        if let Some(index) = self
            .initialized_dynamic_mask_vocab_for_runtime()
            .and_then(|vocab| vocab.cached_direct_regular_wide_frontier_index(lower_id))
        {
            return Some(index);
        }
        if gss.max_depth() != 1 {
            return None;
        }
        let mut top_values = gss.peek_values();
        top_values.sort_unstable();
        let index = self
            .direct_regular_wide_frontier_acceptance
            .iter()
            .position(|summary| {
                summary.state_count == top_values.len()
                    && summary.frontier_states.as_ref() == top_values.as_slice()
            })?;
        // This cache is only an optimization.  Do not materialize the entire
        // dynamic-mask vocabulary merely to memoize a wide-frontier lookup;
        // doing so can otherwise put tens of milliseconds onto the first
        // commit of an ordinary statically compiled constraint.
        if let Some(vocab) = self.initialized_dynamic_mask_vocab_for_runtime() {
            vocab.cache_direct_regular_wide_frontier_index(lower_id, index);
        }
        Some(index)
    }

    #[inline]
    pub(crate) fn direct_regular_wide_acceptance_for_parser_state(
        &self,
        parser_state: u32,
    ) -> Option<&DirectRegularParserStateAcceptance> {
        self.direct_regular_parser_state_acceptance
            .iter()
            .find(|summary| summary.parser_state == parser_state)
    }

    #[inline]
    pub(crate) fn direct_regular_wide_frontier_for_gss(
        &self,
        gss: &ParserGSS,
    ) -> Option<&DirectRegularWideFrontierAcceptance> {
        if self.has_template_parser() { return None; }

        let index = self.direct_regular_wide_frontier_index_for_gss(gss)?;
        self.direct_regular_wide_frontier_acceptance.get(index)
    }

    pub(crate) fn for_each_direct_regular_l1_acceptance(
        &self,
        parser_state: u32,
        mut visit: impl FnMut(RuntimeWeightRef<'_>),
    ) -> bool {
        if self.runtime_direct_regular_l1_is_empty() {
            return false;
        }
        // A sparse regular parser keeps the same depth-one state alphabet
        // after conversion. Its template-domain support replaces the LR row;
        // retain the already-compiled L1 weights without touching an absent
        // table (including during initial-mask preparation after loading).
        if let Some(row) = self.parser_advance_row(parser_state) {
            let mut found = false;
            for terminal in row.iter_ones() {
                if let Some(weight) =
                    self.runtime_direct_regular_l1_complete(terminal as TerminalID)
                {
                    found = true;
                    visit(weight);
                }
            }
            return found;
        }
        if !self.uses_sparse_direct_regular_runtime() {
            return false;
        }
        let Some(automaton) = self.direct_regular_automaton.as_ref() else {
            return false;
        };
        let mut stack = Vec::new();
        if parser_state == 0 {
            stack.extend(automaton.start_states.iter().copied());
        } else if let Some(raw) = parser_state.checked_sub(1) {
            stack.push(raw);
        }
        let mut seen = vec![false; automaton.states.len()];
        let mut terminals = crate::ds::bitset::BitSet::new(self.parser_terminal_count() as usize);
        while let Some(raw) = stack.pop() {
            let Some(state) = automaton.states.get(raw as usize) else {
                return false;
            };
            if std::mem::replace(&mut seen[raw as usize], true) {
                continue;
            }
            stack.extend(state.epsilons.iter().copied());
            for &terminal in state.transitions.keys() {
                terminals.set(terminal as usize);
            }
        }
        let mut found = false;
        for terminal in terminals.iter_ones() {
            if let Some(weight) =
                self.runtime_direct_regular_l1_complete(terminal as TerminalID)
            {
                found = true;
                visit(weight);
            }
        }
        found
    }

    pub(crate) fn sparse_direct_regular_gss_is_complete(&self, gss: &ParserGSS) -> Option<bool> {
        if !self.uses_sparse_direct_regular_runtime() || gss.max_depth() != 1 {
            return None;
        }
        let automaton = self.direct_regular_automaton.as_ref()?;
        let mut stack = Vec::new();
        gss.for_each_top_value(|state| {
            if state == 0 {
                stack.extend(automaton.start_states.iter().copied());
            } else if let Some(raw) = state.checked_sub(1) {
                stack.push(raw);
            }
        });
        let mut seen = vec![false; automaton.states.len()];
        while let Some(raw) = stack.pop() {
            let state = automaton.states.get(raw as usize)?;
            if std::mem::replace(&mut seen[raw as usize], true) {
                continue;
            }
            if state.is_accepting {
                return Some(true);
            }
            stack.extend(state.epsilons.iter().copied());
        }
        Some(false)
    }

    fn direct_regular_dynamic_frontier(
        &self,
        gss: &ParserGSS,
    ) -> Option<DirectRegularDynamicFrontierCacheEntry> {
        if !self.uses_sparse_direct_regular_runtime() || gss.max_depth() != 1 {
            return None;
        }
        let automaton = self.direct_regular_automaton.as_ref()?;
        let cache_key = gss.single_interface_lower_id();
        if let Some(key) = cache_key
            && let Some(cached) = self
                .initialized_dynamic_mask_vocab_for_runtime()
                .and_then(|vocab| vocab.cached_direct_regular_frontier(key))
        {
            return Some(cached);
        }

        let mut roots = Vec::new();
        gss.for_each_top_value(|state| {
            if state == 0 {
                roots.extend(automaton.start_states.iter().copied());
            } else if let Some(raw) = state.checked_sub(1) {
                roots.push(raw);
            }
        });
        let mut seen = vec![false; automaton.states.len()];
        let mut stack = roots;
        let mut targets_by_terminal = BTreeMap::<TerminalID, Vec<u32>>::new();
        while let Some(raw) = stack.pop() {
            let state = automaton.states.get(raw as usize)?;
            if std::mem::replace(&mut seen[raw as usize], true) {
                continue;
            }
            stack.extend(state.epsilons.iter().copied());
            for (&terminal, targets) in &state.transitions {
                if (terminal as usize) >= self.parser_terminal_count() as usize {
                    continue;
                }
                targets_by_terminal
                    .entry(terminal)
                    .or_default()
                    .extend(targets.iter().map(|target| target + 1));
            }
        }

        let mut actionable_terminals =
            crate::ds::bitset::BitSet::new(self.parser_terminal_count() as usize);
        let advance_by_terminal = targets_by_terminal
            .into_iter()
            .filter_map(|(terminal, mut targets)| {
                targets.sort_unstable();
                targets.dedup();
                if targets.is_empty() {
                    return None;
                }
                actionable_terminals.set(terminal as usize);
                Some((terminal, Arc::<[u32]>::from(targets)))
            })
            .collect::<Vec<_>>();
        let entry = DirectRegularDynamicFrontierCacheEntry {
            source: gss.clone(),
            actionable_terminals,
            advance_by_terminal: Arc::from(advance_by_terminal),
        };
        Some(cache_key.map_or(entry.clone(), |key| {
            self.initialized_dynamic_mask_vocab_for_runtime()
                .map_or(entry.clone(), |vocab| {
                    vocab.cache_direct_regular_frontier(key, entry)
                })
        }))
    }

    pub(crate) fn direct_regular_may_advance_on(
        &self,
        gss: &ParserGSS,
        terminal: TerminalID,
    ) -> Option<bool> {
        if !self.uses_sparse_direct_regular_runtime() || gss.max_depth() != 1 {
            return None;
        }
        let automaton = self.direct_regular_automaton.as_ref()?;
        let support = self.dynamic_mask_vocab.direct_regular_terminal_support();
        if support.is_initialized() {
            let mut found = false;
            gss.for_each_top_value(|state| {
                if found {
                    return;
                }
                if state == 0 {
                    found = automaton
                        .start_states
                        .iter()
                        .any(|&raw| support.contains(raw, terminal));
                } else if let Some(raw) = state.checked_sub(1) {
                    found = support.contains(raw, terminal);
                }
            });
            return Some(found);
        }

        let mut stack = Vec::new();
        gss.for_each_top_value(|state| {
            if state == 0 {
                stack.extend(automaton.start_states.iter().copied());
            } else if let Some(raw) = state.checked_sub(1) {
                stack.push(raw);
            }
        });
        let mut seen = vec![false; automaton.states.len()];
        while let Some(raw) = stack.pop() {
            let state = automaton.states.get(raw as usize)?;
            if std::mem::replace(&mut seen[raw as usize], true) {
                continue;
            }
            if state.transitions.contains_key(&terminal) {
                return Some(true);
            }
            stack.extend(state.epsilons.iter().copied());
        }
        Some(false)
    }

    pub(crate) fn direct_regular_may_advance_on_any(
        &self,
        gss: &ParserGSS,
        terminals: &crate::ds::bitset::BitSet,
    ) -> Option<bool> {
        if !self.uses_sparse_direct_regular_runtime() || gss.max_depth() != 1 {
            return None;
        }
        let automaton = self.direct_regular_automaton.as_ref()?;
        let support = self.dynamic_mask_vocab.direct_regular_terminal_support();
        let sparse_terminals = (terminals.count_ones() <= 8).then(|| {
            terminals
                .iter()
                .map(|terminal| terminal as TerminalID)
                .collect::<SmallVec<[TerminalID; 8]>>()
        });
        if support.is_initialized() {
            let mut found = false;
            gss.for_each_top_value(|state| {
                if found {
                    return;
                }
                let state_supports = |raw| {
                    sparse_terminals.as_ref().map_or_else(
                        || support.intersects(raw, terminals.words()),
                        |candidates| {
                            candidates
                                .iter()
                                .any(|&terminal| support.contains(raw, terminal))
                        },
                    )
                };
                if state == 0 {
                    found = automaton.start_states.iter().copied().any(state_supports);
                } else if let Some(raw) = state.checked_sub(1) {
                    found = state_supports(raw);
                }
            });
            return Some(found);
        }

        let sparse_terminals = sparse_terminals.map(|terminals| {
            terminals
                .into_iter()
                .map(|terminal| terminal as u32)
                .collect::<SmallVec<[u32; 8]>>()
        });
        let mut stack = Vec::new();
        gss.for_each_top_value(|state| {
            if state == 0 {
                stack.extend(automaton.start_states.iter().copied());
            } else if let Some(raw) = state.checked_sub(1) {
                stack.push(raw);
            }
        });
        let mut seen = vec![false; automaton.states.len()];
        while let Some(raw) = stack.pop() {
            let state = automaton.states.get(raw as usize)?;
            if std::mem::replace(&mut seen[raw as usize], true) {
                continue;
            }
            let found = sparse_terminals.as_ref().map_or_else(
                || {
                    state
                        .transitions
                        .keys()
                        .any(|terminal| terminals.contains(*terminal as usize))
                },
                |candidates| {
                    candidates
                        .iter()
                        .any(|terminal| state.transitions.contains_key(terminal))
                },
            );
            if found {
                return Some(true);
            }
            stack.extend(state.epsilons.iter().copied());
        }
        Some(false)
    }

    pub(crate) fn direct_regular_admissible_terminals(
        &self,
        gss: &ParserGSS,
    ) -> Option<crate::ds::bitset::BitSet> {
        if !self.uses_sparse_direct_regular_runtime() || gss.max_depth() != 1 {
            return None;
        }
        let automaton = self.direct_regular_automaton.as_ref()?;
        if let Some(summary) = self.direct_regular_dynamic_hot_frontier_for_gss(gss) {
            return Some(summary.actionable_terminals.clone());
        }
        if !self.direct_regular_wide_frontier_acceptance.is_empty()
            && let Some(summary) = self.direct_regular_wide_frontier_for_gss(gss)
        {
            return Some(summary.actionable_terminals.clone());
        }
        let support = self.dynamic_mask_vocab.direct_regular_terminal_support();
        if support.is_initialized() {
            let mut terminals =
                crate::ds::bitset::BitSet::new(self.parser_terminal_count() as usize);
            gss.for_each_top_value(|state| {
                let mut add_state = |raw| {
                    if !support.for_each_small_state_terminal(raw, |terminal| {
                        terminals.set(terminal as usize);
                    }) {
                        support.or_state_into(raw, terminals.words_mut());
                    }
                };
                if state == 0 {
                    for &raw in &automaton.start_states {
                        add_state(raw);
                    }
                } else if let Some(raw) = state.checked_sub(1) {
                    add_state(raw);
                }
            });
            return Some(terminals);
        }
        self.direct_regular_dynamic_frontier(gss)
            .map(|frontier| frontier.actionable_terminals)
    }

    pub(crate) fn direct_regular_cached_advance(
        &self,
        gss: &ParserGSS,
        terminal: TerminalID,
    ) -> Option<ParserGSS> {
        if self.has_template_parser() { return None; }

        if let Some(summary) = self.direct_regular_dynamic_hot_frontier_for_gss(gss) {
            let acc = gss.uniform_accumulator()?;
            let Ok(index) = summary
                .advance_by_terminal
                .binary_search_by_key(&terminal, |(candidate, _)| *candidate)
            else {
                return Some(ParserGSS::empty());
            };
            let targets = &summary.advance_by_terminal[index].1;
            if Arc::ptr_eq(targets, &summary.frontier_states) {
                return summary.empty_acc_frontier.with_uniform_accumulator(acc);
            }
            if let Some(target_summary) = self
                .direct_regular_dynamic_hot_frontiers
                .iter()
                .find(|candidate| {
                    Arc::ptr_eq(&candidate.frontier_states, targets)
                        || candidate.frontier_states.as_ref() == targets.as_ref()
                })
            {
                return target_summary.empty_acc_frontier.with_uniform_accumulator(acc);
            }
            if let [target] = targets.as_ref() {
                return Some(ParserGSS::from_single_stack(vec![*target], acc));
            }
            return Some(ParserGSS::from_sorted_unique_single_value_stacks(
                targets,
                acc,
            ));
        }
        if let Some(summary) = self.direct_regular_wide_frontier_for_gss(gss) {
            let acc = gss.uniform_accumulator()?;
            let index = summary
                .advance_by_terminal
                .binary_search_by_key(&terminal, |(candidate, _)| *candidate)
                .ok()?;
            let targets = &summary.advance_by_terminal.get(index)?.1;
            if Arc::ptr_eq(targets, &summary.frontier_states) {
                return summary.empty_acc_frontier.with_uniform_accumulator(acc);
            }
            if let [target] = targets.as_ref() {
                return Some(ParserGSS::from_single_stack(vec![*target], acc));
            }
            return Some(ParserGSS::from_sorted_unique_single_value_stacks(
                targets,
                acc,
            ));
        }
        if self.uses_sparse_direct_regular_runtime() {
            if gss.max_depth() != 1 {
                return None;
            }
            let acc = gss.uniform_accumulator()?;
            let automaton = self.direct_regular_automaton.as_ref()?;
            let support = self.dynamic_mask_vocab.direct_regular_terminal_support();
            if support.is_initialized() {
                let mut stack = Vec::<u32>::new();
                gss.for_each_top_value(|state| {
                    if state == 0 {
                        stack.extend(
                            automaton
                                .start_states
                                .iter()
                                .copied()
                                .filter(|&raw| support.contains(raw, terminal)),
                        );
                    } else if let Some(raw) = state.checked_sub(1)
                        && support.contains(raw, terminal)
                    {
                        stack.push(raw);
                    }
                });
                let mut seen = crate::ds::bitset::BitSet::new(automaton.states.len());
                let mut target_bits =
                    crate::ds::bitset::BitSet::new(automaton.states.len() + 1);
                while let Some(raw) = stack.pop() {
                    let raw_index = raw as usize;
                    if seen.contains(raw_index) {
                        continue;
                    }
                    seen.set(raw_index);
                    let state = automaton.states.get(raw_index)?;
                    if let Some(state_targets) = state.transitions.get(&terminal) {
                        for &target in state_targets {
                            target_bits.set(target as usize + 1);
                        }
                    }
                    stack.extend(
                        state
                            .epsilons
                            .iter()
                            .copied()
                            .filter(|&child| support.contains(child, terminal)),
                    );
                }
                let target_count = target_bits.count_ones();
                if target_count == 0 {
                    return Some(ParserGSS::empty());
                }
                if target_count == 1 {
                    let target = target_bits
                        .iter_ones()
                        .next()
                        .expect("one target bit must be present") as u32;
                    return Some(ParserGSS::from_single_stack(vec![target], acc));
                }
                let targets = target_bits
                    .iter_ones()
                    .map(|target| target as u32)
                    .collect::<Vec<_>>();
                let source_is_target = target_count == gss.top_value_count() && {
                    let mut same = true;
                    gss.for_each_top_value(|state| {
                        same &= target_bits.contains(state as usize);
                    });
                    same
                };
                if source_is_target {
                    return Some(gss.clone());
                }
                if let Some(summary) = self
                    .direct_regular_wide_frontier_acceptance
                    .iter()
                    .find(|summary| summary.frontier_states.as_ref() == targets.as_slice())
                    && let Some(canonical) = summary
                        .empty_acc_frontier
                        .with_uniform_accumulator(acc.clone())
                {
                    return Some(canonical);
                }
                return Some(ParserGSS::from_sorted_unique_single_value_stacks(
                    &targets, acc,
                ));
            }

            let frontier = self.direct_regular_dynamic_frontier(gss)?;
            let Ok(index) = frontier
                .advance_by_terminal
                .binary_search_by_key(&terminal, |(candidate, _)| *candidate)
            else {
                return Some(ParserGSS::empty());
            };
            let targets = &frontier.advance_by_terminal[index].1;
            if let [target] = targets.as_ref() {
                return Some(ParserGSS::from_single_stack(vec![*target], acc));
            }
            return Some(ParserGSS::from_sorted_unique_single_value_stacks(
                targets,
                acc,
            ));
        }
        if gss.max_depth() != 1 {
            return None;
        }
        let acc = gss.uniform_accumulator()?;
        let state = gss.single_exclusive_top_value()?;
        let origin = match self.table.action(state, terminal)? {
            Action::ReplaceShifts(targets) => targets.as_ptr() as usize,
            Action::StackShifts(shifts)
                if shifts
                    .iter()
                    .all(|shift| shift.pop == 1 && shift.pushes.len() == 1) =>
            {
                shifts.as_ptr() as usize
            }
            _ => return None,
        };
        self.direct_regular_wide_frontier_acceptance
            .iter()
            .find(|summary| summary.action_origins.contains(&origin))?
            .empty_acc_frontier
            .with_uniform_accumulator(acc)
    }

    pub(crate) fn num_parser_states(&self) -> u32 {
        self.parser_symbol_count()
    }

    pub(crate) fn num_tokenizer_states(&self) -> usize {
        self.tokenizer.num_states() as usize
    }

    pub(crate) fn compute_forced_minimized_tokenizer_state_count(&self) -> usize {
        self.tokenizer.compute_forced_minimized_state_count()
    }

    pub(crate) fn parser_dwa(&self) -> &DWA {
        &self.parser_dwa
    }

    pub(crate) fn possible_matches_for_state_internal(
        &self,
        tokenizer_state: u32,
    ) -> Option<BTreeMap<TerminalID, RangeSetBlaze<u32>>> {
        // Return possible_matches in the final shared constraint-internal vocab
        // space. These ids match parser-DWA weight token ids after reconciliation.
        let mut result = BTreeMap::new();
        for terminal in self.runtime_possible_match_terminals() {
            let Some(weight) = self.runtime_possible_match_weight(terminal) else {
                continue;
            };
            let mut tokens = RangeSetBlaze::new();
            for &internal_tsid in self.internal_tsids_for_state(tokenizer_state) {
                if let Some(token_set) = weight.token_set_for_tsid(internal_tsid) {
                    tokens |= token_set.to_range_set();
                }
            }
            if !tokens.is_empty() {
                result.insert(terminal, tokens);
            }
        }
        if result.is_empty() {
            None
        } else {
            Some(result)
        }
    }

    fn build_internal_token_buf_mask(originals: &[u32]) -> InternalTokenBufMasks {
        let mut result = Vec::<(u16, u32)>::new();
        let mut current_word = None::<u16>;
        let mut current_mask = 0u32;
        for &original in originals {
            let word = (original / 32) as u16;
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
        let mut word_map = BTreeMap::<u16, u32>::new();
        for &original in originals {
            let word = (original / 32) as u16;
            let bit = original % 32;
            *word_map.entry(word).or_default() |= 1u32 << bit;
        }
        word_map.into_iter().collect()
    }

    pub(crate) fn max_original_token_id(&self) -> Option<u32> {
        self.body_max_original_token_id().into_iter()
            .chain(self.end_tokens.last().copied()).max()
    }

    pub(crate) fn body_max_original_token_id(&self) -> Option<u32> {
        self.token_bytes
            .keys()
            .next_back()
            .copied()
            .or_else(|| {
                self.packed_token_bytes
                    .as_ref()
                    .and_then(|packed| packed.max_token_id())
            })
            .into_iter()
            .chain(
                self.special_token_terminals
                    .iter()
                    .filter(|special| {
                        !self.is_late_grammar_placeholder_terminal(special.terminal_id)
                    })
                    .map(|special| special.token_id),
            )
            .max()
    }

    /// Remove compiler-only late-grammar sentinel token IDs from the runtime
    /// model-token coordinate while retaining their terminal metadata for the
    /// linker. Returns whether the token coordinate changed.
    ///
    /// External-grammar placeholders are compiled initially as exact special
    /// tokens so the ordinary compiler can build the parent grammar. Once a
    /// terminal is recorded in `late_grammar_slots`, that backing token ID is
    /// no longer a public/model token. Leaving it in the original-token map
    /// makes persisted sparse mask fragments wider than `mask_len()`, which
    /// correctly excludes unresolved linker sentinels.
    pub(crate) fn sanitize_late_grammar_placeholder_token_domain(&mut self) -> bool {
        if self.late_grammar_slots.is_empty() {
            return false;
        }

        let placeholder_terminals = self
            .late_grammar_slots
            .iter()
            .map(|slot| slot.terminal_id)
            .collect::<BTreeSet<_>>();
        self.sanitize_placeholder_terminal_token_domain(&placeholder_terminals)
    }

    /// Remove non-vocabulary compiler sentinel token IDs associated with an
    /// explicit set of placeholder terminals.
    pub(crate) fn sanitize_placeholder_terminal_token_domain(
        &mut self,
        placeholder_terminals: &BTreeSet<TerminalID>,
    ) -> bool {
        if placeholder_terminals.is_empty() || !self.has_original_token_map() {
            return false;
        }

        let mut placeholder_tokens = self
            .special_token_terminals
            .iter()
            .filter(|special| placeholder_terminals.contains(&special.terminal_id))
            .filter(|special| self.token_bytes_for_id(special.token_id).is_none())
            .filter(|special| {
                !self.special_token_terminals.iter().any(|other| {
                    other.token_id == special.token_id
                        && !placeholder_terminals.contains(&other.terminal_id)
                })
            })
            .map(|special| special.token_id)
            .collect::<Vec<_>>();
        placeholder_tokens.sort_unstable();
        placeholder_tokens.dedup();
        if placeholder_tokens.is_empty()
            || !placeholder_tokens.iter().any(|&token| {
                self.original_token_internal_at(token)
                    .is_some_and(|internal| internal != u32::MAX)
            })
        {
            return false;
        }

        let group_count = self.internal_token_count();
        let mut original_to_internal = self.original_token_map().to_vec();
        for token in placeholder_tokens {
            if let Some(internal) = original_to_internal.get_mut(token as usize) {
                *internal = u32::MAX;
            }
        }
        let mut internal_to_tokens = (0..group_count)
            .map(|_| Vec::<u32>::new())
            .collect::<Vec<_>>();
        for (original, &internal) in original_to_internal.iter().enumerate() {
            if internal == u32::MAX {
                continue;
            }
            if let Some(group) = internal_to_tokens.get_mut(internal as usize) {
                group.push(original as u32);
            }
        }

        self.original_token_to_internal = original_to_internal;
        self.packed_original_token_to_internal = None;
        self.deferred_original_token_to_internal = OnceLock::new();
        self.internal_token_to_tokens = internal_to_tokens;
        self.deferred_internal_token_to_tokens = OnceLock::new();

        // Portable per-token fragments and every aggregate derived from them
        // were built before the slot became linker-only. Force one rebuild in
        // the reduced public token domain; subsequent save/load can reuse the
        // clean caches normally.
        self.internal_token_buf_masks.clear();
        self.internal_token_buf_flat = Box::new([]);
        self.backed_internal_token_buf_flat = None;
        self.internal_token_buf_offsets = Box::new([]);
        self.word_group_buf_masks.clear();
        self.pair_word_group_buf_masks = Default::default();
        self.quad_word_group_buf_masks = Default::default();
        self.super_word_group_buf_masks = Default::default();
        self.mega_word_group_buf_masks = Default::default();
        self.giga_word_group_buf_masks = Default::default();
        self.word_group_sparse_masks.clear();
        self.word_group_prefix_buf_masks = Default::default();
        self.word_group_sparse_prefix_entries.clear();
        self.quad_group_sparse_masks.clear();
        self.quad_group_dense_masks.clear();
        self.byte_group_sparse_masks.clear();
        self.byte_group_dense_masks.clear();
        self.word_group_sparse_total_entries = 0;
        self.word_group_sparse_max_entries = 0;
        self.all_tokens_buf_mask = Box::new([]);
        self.heavy_token_dense_masks.clear();
        self.heavy_token_indices.clear();
        self.internal_token_buf_op_costs.clear();
        self.word_group_buf_op_costs.clear();
        self.total_internal_buf_cost = 0;
        self.heavy_total_cost = 0;
        self.light_avg_cost_x256 = 0;
        self.parser_runtime_caches_prebuilt = false;
        self.serialized_artifact_cache = None;
        true
    }

    pub(crate) fn is_late_grammar_placeholder_terminal(&self, terminal_id: u32) -> bool {
        self.late_grammar_slots
            .iter()
            .any(|slot| slot.terminal_id == terminal_id)
    }

    pub(crate) fn has_special_token_id(&self, token_id: u32) -> bool {
        self.special_token_terminals
            .iter()
            .any(|special| {
                special.token_id == token_id
                    && !self.is_late_grammar_placeholder_terminal(special.terminal_id)
            })
    }

    fn build_seed_terminal_dense_masks(&self) -> SeedTerminalDenseMasks {
        let mut result = SeedTerminalDenseMasks::default();
        let internal_tsid_to_states = self.internal_tsid_groups();
        for terminal_id in self.runtime_possible_match_terminals() {
            let Some(weight) = self.runtime_possible_match_weight(terminal_id) else {
                continue;
            };
            weight.for_each_entry(|start, end, token_set| {
                let dense = self.dense_words_from_runtime_token_set(token_set);
                for internal_tsid in start..=end {
                    if let Some(states) = internal_tsid_to_states.get(internal_tsid as usize) {
                        for &tokenizer_state in states {
                            let entry = result
                                .entry((tokenizer_state, terminal_id))
                                .or_insert_with(empty_dense_words);
                            let mut merged = entry.to_vec();
                            if merged.len() < dense.len() {
                                merged.resize(dense.len(), 0);
                            }
                            for (index, &word) in dense.iter().enumerate() {
                                merged[index] |= word;
                            }
                            *entry = merged.into();
                        }
                    } else {
                        result.insert((internal_tsid, terminal_id), dense.clone());
                    }
                }
            });
        }
        result
    }

    fn or_internal_token_masks_to_buf(&self, internal_token: usize, buf: &mut [u32]) {
        self.for_each_internal_token_buf_mask_entry(internal_token, |word_idx, mask| {
            buf[word_idx as usize] |= mask;
        });
    }

    fn sparse_word_group_entries_in(&self, start: usize, len: usize) -> usize {
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
    fn prefer_dense_buf_scan(buf_words: usize, sparse_entries: usize) -> bool {
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
    fn or_internal_token_to_buf_fast<const PROFILE: bool>(
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

    fn or_original_token_to_buf(&self, token_id: u32, buf: &mut [u32]) {
        let word = token_id as usize / 32;
        let bit = token_id as usize % 32;
        if let Some(slot) = buf.get_mut(word) {
            *slot |= 1u32 << bit;
        }
    }

}

impl<'a> ConstraintState<'a> {
    /// Fill a mask directly from the lexer and parser stack, without using the
    /// parser DWA.
    pub(crate) fn fill_mask_dynamic(&self, buf: &mut [u32]) {
        crate::compiler::boundary_transfer::strict_static_trap_dynamic_for_state(
            "ConstraintState::fill_mask_dynamic",
            self.constraint.uses_dynamic_runtime(),
        );
        if self.constraint.uses_compact_segmented_parser_runtime() {
            match super::dynamic_mask::try_fill_recursive_mask_shared(self, buf) {
                Ok(true) => return,
                Ok(false) => panic!("recursive composition declined its shared mask provider"),
                Err(error) => {
                    panic!("shared recursive dynamic mask generation failed: {error}");
                }
            }
        }
        super::dynamic_mask::fill_mask_dynamic(self, buf);
    }

    pub(crate) fn fill_mask_dynamic_bounded(
        &self,
        buf: &mut [u32],
        timeout_ms: u64,
    ) -> Result<(), String> {
        super::dynamic_mask::fill_mask_dynamic_bounded(self, buf, timeout_ms)
    }

}
#[cfg(test)]
mod dense_internal_token_mask_tests {
    use super::*;

    #[test]
    fn early_observation_filter_matches_historical_rows_across_future_widths() {
        let vocab = Vocab::new((0u32..128).map(|id| (id, vec![id as u8])).collect::<Vec<_>>());
        let mut saw_mixed = false;
        let mut saw_over_limit = false;
        for width in [1usize, 2, 3, 8, 9, 12] {
            let mut grammar = String::from("start start;\n");
            for terminal in 0..width {
                let suffix = char::from(b'a' + terminal as u8);
                grammar.push_str(&format!("t T{terminal} ::= /[\\x00-\\x7F]+{suffix}/;\n"));
            }
            grammar.push_str("nt start ::= ");
            grammar.push_str(&(0..width).map(|t| format!("T{t} T{t}"))
                .collect::<Vec<_>>().join(" | "));
            grammar.push_str(";\n");
            let built = crate::DynamicConstraint::from_glrm_grammar(&grammar, &vocab)
                .expect("finite physical selector fixture compiles");
            let constraint = &built.inner;
            assert!(!constraint.tokenizer.has_any_virtual_runtime());
            for state in 0..constraint.tokenizer.num_states() {
                let count = constraint.tokenizer.possible_future_terminals_iter(state).count();
                saw_mixed |= (2..=8).contains(&count);
                saw_over_limit |= count > 8;
            }
            assert_eq!(
                constraint.build_dynamic_terminal_observation_classes_filtered::<true>(),
                constraint.build_dynamic_terminal_observation_classes_filtered::<false>(),
                "future width {width}",
            );
        }
        assert!(saw_mixed, "fixtures must exercise eligible mixed signatures");
        assert!(saw_over_limit, "fixtures must exercise the nine-entry rejection bound");
    }

    #[test]
    fn bind_vocab_exact_rebinds_equal_vocab_and_rejects_mismatch() {
        let vocab_a = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);
        let vocab_b = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);
        let vocab_bad = Vocab::new(vec![(0, b"a".to_vec()), (1, b"c".to_vec())]);
        let mut constraint = Constraint::from_glrm_grammar(
            "start start;\nt A ::= \"a\";\nnt start ::= A;\n",
            &vocab_a,
        )
        .unwrap();

        assert!(!Arc::ptr_eq(&constraint.token_bytes, &vocab_b.entries_arc()));
        constraint.bind_vocab_exact(&vocab_b).unwrap();
        assert!(Arc::ptr_eq(&constraint.token_bytes, &vocab_b.entries_arc()));
        assert!(constraint.late_bind_vocab.get().is_some());
        assert!(Arc::ptr_eq(
            &constraint
                .late_bind_vocab
                .get()
                .expect("exact vocab bind should seed late-bind vocab")
                .entries_arc(),
            &vocab_b.entries_arc(),
        ));

        let bound = Arc::clone(&constraint.token_bytes);
        assert!(constraint.bind_vocab_exact(&vocab_bad).is_err());
        assert!(Arc::ptr_eq(&constraint.token_bytes, &bound));
    }

    #[test]
    fn loaded_constraint_binds_prepared_equal_vocab_and_rejects_mismatch() {
        let vocab_a = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);
        let vocab_b = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);
        let vocab_bad = Vocab::new(vec![(0, b"a".to_vec()), (1, b"c".to_vec())]);
        let constraint = Constraint::from_glrm_grammar(
            "start start;\nt A ::= \"a\";\nnt start ::= A;\n",
            &vocab_a,
        )
        .unwrap();
        let saved = constraint.save();

        // Prepare exactly the pure Vocab artifact used by the packed-wire fast
        // path, then bind a current-format loaded constraint. This exercises
        // the path used by serving/CFA where vocabulary preparation is shared
        // across many independently loaded constraints.
        let _ = crate::compiler::compile::vocab_packed_token_bytes(&vocab_b);
        let mut loaded = Constraint::load(saved.clone()).unwrap();
        assert!(loaded.packed_token_bytes.is_some());
        loaded.bind_vocab_exact(&vocab_b).unwrap();
        assert!(Arc::ptr_eq(&loaded.token_bytes, &vocab_b.entries_arc()));

        let _ = crate::compiler::compile::vocab_packed_token_bytes(&vocab_bad);
        let mut mismatched = Constraint::load(saved).unwrap();
        assert!(mismatched.packed_token_bytes.is_some());
        assert!(mismatched.bind_vocab_exact(&vocab_bad).is_err());
    }
    use crate::Vocab;

    #[test]
    fn empty_wide_frontier_lookup_does_not_materialize_deferred_dynamic_vocab() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"ab".to_vec()),
        ]);
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start start;
                t A ::= "a";
                t B ::= "b";
                nt start ::= A B;
            "#,
            &vocab,
        )
        .unwrap();
        let loaded = Constraint::load(&constraint.save()).unwrap();
        assert!(loaded.direct_regular_wide_frontier_acceptance.is_empty());
        assert!(
            loaded.lazy_dynamic_mask_vocab.get().is_none(),
            "load should preserve deferred dynamic-vocab materialization",
        );

        let initial = loaded.initial_state_map();
        for gss in initial.values() {
            assert!(loaded.direct_regular_wide_frontier_for_gss(gss).is_none());
        }

        assert!(
            loaded.lazy_dynamic_mask_vocab.get().is_none(),
            "empty wide-frontier lookup must not trigger deferred dynamic-vocab materialization",
        );
    }

    #[test]
    fn nonempty_wide_frontier_lookup_does_not_materialize_vocab_for_cache_only() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"ab".to_vec()),
        ]);
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start start;
                t A ::= "a";
                t B ::= "b";
                nt start ::= A B;
            "#,
            &vocab,
        )
        .unwrap();
        let mut loaded = Constraint::load(&constraint.save()).unwrap();
        assert!(loaded.lazy_dynamic_mask_vocab.get().is_none());

        let initial = loaded.initial_state_map();
        let gss = initial.values().next().unwrap().clone();
        let mut frontier_states = gss.peek_values();
        frontier_states.sort_unstable();
        loaded.direct_regular_wide_frontier_acceptance = vec![
            crate::runtime::artifact::DirectRegularWideFrontierAcceptance {
                action_origins: Vec::new(),
                state_count: frontier_states.len(),
                actionable_terminals: crate::ds::bitset::BitSet::new(
                    loaded.parser_terminal_count() as usize,
                ),
                frontier_states: Arc::<[u32]>::from(frontier_states.as_slice()),
                // Keep this deliberately unrelated to `gss` so the cheap
                // lower-id lookup misses and the fallback frontier-state scan
                // is exercised.
                empty_acc_frontier: ParserGSS::empty(),
                acceptance_parts: Arc::from([]),
                dense_by_tsid: Arc::new(crate::runtime::artifact::DenseAcceptanceRows::default()),
                advance_by_terminal: Arc::from([]),
            },
        ];

        assert!(loaded.direct_regular_wide_frontier_for_gss(&gss).is_none(),
            "native relations decline LR frontier memoization without materializing vocabulary");
        assert!(
            loaded.lazy_dynamic_mask_vocab.get().is_none(),
            "wide-frontier memoization must not materialize the dynamic vocab",
        );
    }

    #[test]
    fn initial_commit_prime_token_ids_accepts_exact_limit() {
        let mask = [u32::MAX >> (32 - INITIAL_COMMIT_PRIME_MAX_TOKENS)];
        assert_eq!(
            initial_commit_prime_token_ids(&mask),
            Some((0..INITIAL_COMMIT_PRIME_MAX_TOKENS as u32).collect()),
        );
    }

    #[test]
    fn initial_commit_prime_token_ids_rejects_above_limit() {
        let mask = [u32::MAX >> (32 - (INITIAL_COMMIT_PRIME_MAX_TOKENS + 1))];
        assert_eq!(initial_commit_prime_token_ids(&mask), None);
    }

    #[test]
    fn template_initial_priming_is_bounded_and_selects_only_allowed_tokens() {
        assert!(template_initial_prime_token_ids(&[]).is_empty());
        assert!(template_initial_prime_token_ids(&[0, 0]).is_empty());
        assert_eq!(template_initial_prime_token_ids(&[0, 0b1010, 0]), vec![33, 35]);
        let wide = [0, u32::MAX, u32::MAX];
        assert_eq!(
            template_initial_prime_token_ids(&wide),
            (32..32 + INITIAL_COMMIT_PRIME_MAX_TOKENS as u32).collect::<Vec<_>>()
        );
        assert!(initial_commit_prime_token_ids(&wide).is_none());
    }

    #[test]
    fn dense_internal_token_masks_match_reference_expansion() {
        let internal_tokens = RangeSetBlaze::from_iter([
            0u32..=0,
            3..=7,
            62..=65,
            127..=130,
            190..=192,
            300..=302,
        ]);
        let actual = Constraint::dense_words_from_internal_set_with_words(&internal_tokens, 5);
        let mut expected = vec![0u64; 5];
        for token in internal_tokens.iter() {
            let word = token as usize / 64;
            let bit = token as usize % 64;
            if let Some(slot) = expected.get_mut(word) {
                *slot |= 1u64 << bit;
            }
        }
        assert_eq!(actual.as_ref(), expected.as_slice());
    }

    #[test]
    fn dense_internal_token_masks_ignore_out_of_bounds_ranges() {
        let internal_tokens = RangeSetBlaze::from_iter([63u32..=65, 190..=400]);
        let actual = Constraint::dense_words_from_internal_set_with_words(&internal_tokens, 3);
        assert_eq!(actual.as_ref(), &[1u64 << 63, 0b11, 1u64 << 62 | 1u64 << 63]);
    }

    #[test]
    fn direct_sparse_expanded_work_counts_alias_expansion_and_heavy_tokens() {
        let masks = vec![
            vec![(0, 1)],
            vec![(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)],
            vec![(0, 1), (1, 1)],
            vec![(0, 1), (1, 1), (2, 1)],
        ];
        let work_prefix = Constraint::direct_sparse_work_prefix(&masks, 16);
        let selected = RangeSetBlaze::from_iter([0u32..=1, 3..=3]);

        // Each selected internal token costs one membership scan. Token 1 is
        // heavy because 5 entries exceed 16 / 4, so runtime uses a 16-word
        // dense OR for it instead of five sparse writes.
        assert_eq!(
            Constraint::direct_sparse_expanded_work(&selected, &work_prefix),
            (1 + 1) + (1 + 16) + (1 + 3)
        );
    }

    #[test]
    fn owned_load_preserves_heavy_token_mask_classification_for_backed_buf_masks() {
        let vocab = Vocab::new(
            (0..200u32)
                .map(|token| (token, b"a".to_vec()))
                .collect(),
        );
        let constraint = Constraint::from_glrm_grammar(
            "start start;\nt A ::= \"a\";\nnt start ::= A;\n",
            &vocab,
        )
        .unwrap();
        assert!(
            !constraint.heavy_token_indices.is_empty(),
            "duplicate-token expansion should produce at least one heavy internal token",
        );

        let loaded = Constraint::load(constraint.save()).unwrap();
        let backed = loaded
            .backed_internal_token_buf_flat
            .as_ref()
            .expect("owned current load should retain IBM2 entries in artifact backing");
        assert!(
            backed.slice(0, backed.len()).is_some(),
            "fresh current-format IBM2 entries should be naturally aligned for native access",
        );
        assert_eq!(loaded.heavy_token_indices, constraint.heavy_token_indices);
        assert_eq!(
            loaded.internal_token_buf_op_costs,
            constraint.internal_token_buf_op_costs,
        );
        assert_eq!(
            loaded.word_group_buf_op_costs,
            constraint.word_group_buf_op_costs,
        );
        assert_eq!(loaded.total_internal_buf_cost, constraint.total_internal_buf_cost);
    }

    #[test]
    fn direct_sparse_expanded_work_clamps_out_of_bounds_ranges() {
        let masks = vec![vec![(0, 1)], vec![(1, 1), (2, 1)]];
        let work_prefix = Constraint::direct_sparse_work_prefix(&masks, 16);
        let selected = RangeSetBlaze::from_iter([1u32..=100]);

        assert_eq!(
            Constraint::direct_sparse_expanded_work(&selected, &work_prefix),
            1 + 2
        );
    }

    #[test]
    fn heavy_group_dense_masks_match_sparse_replay_and_threshold() {
        let groups = vec![
            vec![(0, 0b0011), (2, 0b0100), (5, 0b1000)],
            vec![(1, 0b0101), (6, 0b1010)],
        ];
        let dense = Constraint::compute_heavy_group_dense_masks(&groups, 8);

        assert!(dense[0].is_some(), "3 sparse entries should beat the 8 / 4 threshold");
        assert!(dense[1].is_none(), "2 sparse entries should remain sparse at the threshold");

        let mut sparse_or = vec![0x10u32; 8];
        or_sparse_buf_entries(&mut sparse_or, &groups[0]);
        let mut adaptive_or = vec![0x10u32; 8];
        assert_eq!(
            or_group_buf_mask(&mut adaptive_or, &groups[0], dense[0].as_deref()),
            8,
        );
        assert_eq!(adaptive_or, sparse_or);

        let mut sparse_andnot = vec![u32::MAX; 8];
        andnot_sparse_buf_entries(&mut sparse_andnot, &groups[0]);
        let mut adaptive_andnot = vec![u32::MAX; 8];
        assert_eq!(
            andnot_group_buf_mask(
                &mut adaptive_andnot,
                &groups[0],
                dense[0].as_deref(),
            ),
            8,
        );
        assert_eq!(adaptive_andnot, sparse_andnot);

        let mut light = vec![0u32; 8];
        assert_eq!(
            or_group_buf_mask(&mut light, &groups[1], dense[1].as_deref()),
            groups[1].len(),
        );
    }

    #[test]
    fn dense_or_matches_set_union_exhaustively_without_final_mapping() {
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"b".to_vec()),
                (2, b"ab".to_vec()),
                (3, b"ba".to_vec()),
            ]);
        let mut constraint = Constraint::from_glrm_grammar(
            r#"
                start start;
                t A ::= "a" | "ab";
                t B ::= "b" | "ab";
                nt item ::= A | B;
                nt start ::= item item? item?;
            "#,
            &vocab,
        )
        .unwrap();
        constraint.final_mask_mapping = Default::default();
        let n_internal = constraint.internal_token_to_tokens.len();
        assert!(n_internal <= 16, "small exhaustive test requires a tiny internal vocab");

        for selected in 0u64..(1u64 << n_internal) {
            let mut dense = vec![0u64; n_internal.div_ceil(64)];
            let mut selected_image = 0u32;
            for (internal, originals) in constraint.internal_token_to_tokens.iter().enumerate() {
                if selected & (1u64 << internal) == 0 {
                    continue;
                }
                dense[internal / 64] |= 1u64 << (internal % 64);
                for &original in originals {
                    selected_image |= 1u32 << original;
                }
            }

            for initial in 0u32..=0x0f {
                let expected = initial | selected_image;
                let buf_zeroed = initial == 0;

                let mut profiled = vec![0u32; constraint.body_mask_len()];
                profiled[0] = initial;
                constraint.or_internal_dense_to_buf(&dense, &mut profiled, buf_zeroed);
                assert_eq!(
                    profiled[0],
                    expected,
                    "profiled OR mismatch: selected={selected:#b} initial={initial:#06b} internal_to_original={:?}",
                    constraint.internal_token_to_tokens,
                );

                let mut fast = vec![0u32; constraint.body_mask_len()];
                fast[0] = initial;
                constraint.or_internal_dense_to_buf_fast(&dense, &mut fast, buf_zeroed);
                assert_eq!(
                    fast[0],
                    expected,
                    "fast OR mismatch: selected={selected:#b} initial={initial:#06b} internal_to_original={:?}",
                    constraint.internal_token_to_tokens,
                );

                let mut scratch_fast = vec![0u32; constraint.body_mask_len()];
                scratch_fast[0] = initial;
                let mut dirty_complement_scratch = Vec::new();
                constraint.or_internal_dense_to_buf_fast_with_scratch(
                    &dense,
                    &mut scratch_fast,
                    buf_zeroed,
                    &mut dirty_complement_scratch,
                );
                assert_eq!(
                    scratch_fast[0],
                    expected,
                    "scratch fast OR mismatch: selected={selected:#b} initial={initial:#06b} internal_to_original={:?}",
                    constraint.internal_token_to_tokens,
                );
            }
        }
    }

    #[test]
    fn dirty_dense_conversion_uses_scratch_complement_when_alias_replay_is_expensive() {
        let mut entries = Vec::new();
        for alias in 0u32..64 {
            for byte_index in 0u32..64 {
                entries.push((alias * 64 + byte_index, vec![(byte_index + 32) as u8]));
            }
        }
        let vocab = Vocab::new(entries);
        let mut grammar = String::from("start start;\n");
        for byte_index in 0u32..64 {
            let literal = serde_json::to_string(&((byte_index + 32) as u8 as char).to_string())
                .unwrap();
            grammar.push_str(&format!("t T{byte_index} ::= {literal};\n"));
        }
        grammar.push_str("nt start ::= ");
        for byte_index in 0u32..64 {
            if byte_index != 0 {
                grammar.push_str(" | ");
            }
            grammar.push_str(&format!("T{byte_index} T{byte_index}"));
        }
        grammar.push_str(";\n");
        let mut constraint = Constraint::from_glrm_grammar(
            &grammar,
            &vocab,
        )
        .unwrap();
        constraint.final_mask_mapping = Default::default();

        let n_internal = constraint.internal_token_to_tokens.len();
        assert_eq!(n_internal, 64, "expected duplicate bytes to form 64 internal tokens");

        let mut dense = vec![0u64; n_internal.div_ceil(64)];
        for internal in 0..48 {
            dense[internal / 64] |= 1u64 << (internal % 64);
        }

        let dirty_token = constraint.internal_token_to_tokens[63][0];
        let dirty_word = dirty_token as usize / 32;
        let dirty_bit = dirty_token as usize % 32;

        let mut expected = vec![0u32; constraint.body_mask_len()];
        expected[dirty_word] |= 1u32 << dirty_bit;
        constraint.or_internal_dense_to_buf_fast(&dense, &mut expected, false);

        let mut actual = vec![0u32; constraint.body_mask_len()];
        actual[dirty_word] |= 1u32 << dirty_bit;
        let mut scratch = Vec::new();
        constraint.or_internal_dense_to_buf_fast_with_scratch(
            &dense,
            &mut actual,
            false,
            &mut scratch,
        );

        assert_eq!(actual, expected);
        assert_eq!(
            scratch.len(),
            constraint.body_mask_len(),
            "expected the replay-cost model to select dirty scratch complement conversion",
        );
    }
}
