//! Direct dynamic mask generation.
//!
//! This implementation intentionally does not consult the parser DWA. It walks
//! the vocabulary byte trie while advancing the lexer and GLR parser directly.

use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};
use std::hash::{Hash, Hasher};

use rustc_hash::{FxHashMap, FxHashSet, FxHasher};
use smallvec::SmallVec;

use crate::automata::lexer::Lexer;
use crate::automata::lexer::tokenizer::{
    TerminalResidualDirectCoordinate, Tokenizer, TokenizerExecResult, TokenizerMatch,
    TokenizerStateSet, VirtualResidualDirectCoordinate,
};
use crate::compiler::glr::accumulator::TerminalsDisallowed;
use crate::compiler::glr::parser::{advance_stacks, ParserGSS};
use crate::ds::bitset::BitSet;
use crate::ds::leveled_gss::LeveledGSS;
use crate::ds::u8set::U8Set;
use crate::grammar::flat::TerminalID;

use super::artifact::{
    Constraint, DynamicDenseSubset16, DynamicLazyUnionCache, DynamicLazyUnionMetadata, DynamicMaskLexerStateKey,
    DynamicMaskStateKey, DynamicMaskTrie, DynamicMaskTrieFullWalkOp, DynamicMaskVocab,
    FastTokenizerTransitions, dynamic_mask_llg_master_is_whitespace,
    dynamic_mask_llg_master_safe_chars,
};
use super::state::ConstraintState;

mod boundary_cache;
mod full_walk_dense;
mod recursive_provider;

type ParserStacks = LeveledGSS<u32, ()>;

/// Exact admission for one retained parser frontier during one mask invocation.
/// The borrowed ConstraintState keeps the owner alive for this entire lifetime;
/// the address is only an extra identity check, never a global cache key.
struct RootAdmissionMemo {
    owner: usize,
    gss: ParserStacks,
    admitted: OnceLock<BitSet>,
}

type SharedRootAdmission = Option<Arc<RootAdmissionMemo>>;

fn root_admission_memo_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("GLRMASK_DISABLE_ROOT_ADMISSION_MEMO").is_none()
    })
}

fn raw_terminal_radius_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("GLRMASK_DISABLE_RAW_TERMINAL_RADIUS").is_none()
    })
}

fn exact_parser_admission_for_stacks(constraint: &Constraint, stacks: &ParserStacks) -> BitSet {
    let parser_gss = with_empty_accumulators(stacks);
    constraint.direct_regular_admissible_terminals(&parser_gss).unwrap_or_else(|| {
        let candidates = BitSet::all(constraint.parser_terminal_count() as usize);
        super::commit::exact_admitted_terminals_for_candidates(
            constraint, &parser_gss, &candidates,
        )
    })
}

#[inline]
fn parser_admission_with_root_memo(
    constraint: &Constraint,
    stacks: &ParserStacks,
    shared: &SharedRootAdmission,
) -> BitSet {
    let Some(memo) = shared.as_ref().filter(|memo| {
        memo.owner == constraint as *const Constraint as usize && memo.gss.ptr_eq(stacks)
    }) else {
        return exact_parser_admission_for_stacks(constraint, stacks);
    };
    let admitted = memo.admitted.get_or_init(|| {
        exact_parser_admission_for_stacks(constraint, stacks)
    });
    static VERIFY: OnceLock<bool> = OnceLock::new();
    if *VERIFY.get_or_init(|| {
        std::env::var_os("GLRMASK_ASSERT_ROOT_ADMISSION_MEMO").is_some()
    }) {
        assert_eq!(*admitted, exact_parser_admission_for_stacks(constraint, stacks));
    }
    admitted.clone()
}

fn root_admission_memo(
    constraint: &Constraint,
    stacks: &ParserStacks,
    roots: &DynamicBranches,
) -> SharedRootAdmission {
    if !root_admission_memo_enabled() {
        return None;
    }
    roots.iter().find(|branch| branch.gss.ptr_eq(stacks))
        .and_then(|branch| branch.shared_root_admission.clone())
        .or_else(|| Some(Arc::new(RootAdmissionMemo {
            owner: constraint as *const Constraint as usize,
            gss: stacks.clone(),
            admitted: OnceLock::new(),
        })))
}


/// The current LR action row is a necessary condition for admitting a terminal
/// when no zero-width control transitions can change the top first. It is only
/// a candidate superset: stack-dependent reductions/guards still require the
/// ordinary parser. A fully unconditional row additionally proves exactness.
fn parser_row_projection_candidates(
    constraint: &Constraint,
    stacks: &ParserStacks,
) -> Option<(BitSet, bool)> {
    if constraint.uses_sparse_direct_regular_runtime()
        || constraint.uses_compact_segmented_parser_runtime()
        || constraint.parser_has_controls()
    {
        return None;
    }
    let top = stacks.single_top_value()?;
    let row = constraint.parser_advance_row(top)?;
    let terminal_count = constraint.parser_terminal_count() as usize;
    // The advance row may also contain EOF. Never put it in a lexer filter.
    let mut candidates = BitSet::new(terminal_count);
    for terminal in row.iter_ones().take_while(|&terminal| terminal < terminal_count) {
        candidates.set(terminal);
    }
    let exact = constraint.parser_unconditional_row(top).is_some_and(|row| {
        candidates.iter_ones().all(|terminal| row.contains(terminal))
    });
    Some((candidates, exact))
}

#[cfg(test)]
thread_local! {
    static TEST_FULL_WALK_USES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static TEST_CONFIG_FULL_WALK_USES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static TEST_RECURSIVE_PRODUCT_CACHE_CAPACITY: std::cell::Cell<usize> = const { std::cell::Cell::new(256) };
    static TEST_RECURSIVE_PERSISTENT_MASK_CACHE: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}

// Experimental one-byte coordinate for the overwhelmingly common scalar,
// non-finalizing lexer states encountered by a full vocabulary walk.  Keep the
// generic config executor as the exact fallback for finalizers / epsilon /
// virtual states.
const FULL_WALK_HOT_TAG: u32 = 0x7fff_ff00;
const FULL_WALK_HOT_MASK: u32 = 0xff;
const FULL_WALK_HOT_CAPACITY: usize = 253;
const FULL_WALK_HOT_SLOW: u8 = 253;
const FULL_WALK_HOT_DEAD: u8 = 254;
const FULL_WALK_HOT_UNKNOWN: u8 = 255;

#[inline]
fn env_flag(name: &str, default: bool) -> bool {
    std::env::var(name)
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            !matches!(normalized.as_str(), "" | "0" | "false" | "no" | "off")
        })
        .unwrap_or(default)
}

/// Bound parser-conditioned scalar liveness checks during exact config walks.
///
/// Four consecutive non-pruning checks was neutral in the first 1k corpus and
/// substantially reduced later JSON-schema tails.  Keep the historical env
/// knob as an override: `0`/`false` disables the bounded checks, `1`/`true`
/// retains the old experimental 32-check interpretation, and an integer sets
/// an explicit budget.
#[inline]
fn config_scalar_conditioned_budget() -> Option<u32> {
    static BUDGET: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    *BUDGET.get_or_init(|| {
        match std::env::var("GLRMASK_EXPERIMENT_CONFIG_SCALAR_CONDITIONED_BUDGET") {
            Ok(value) => {
                let trimmed = value.trim();
                if trimmed.is_empty()
                    || trimmed == "0"
                    || trimmed.eq_ignore_ascii_case("false")
                {
                    None
                } else if trimmed == "1" || trimmed.eq_ignore_ascii_case("true") {
                    Some(32)
                } else {
                    Some(trimmed.parse::<u32>().unwrap_or(32))
                }
            }
            Err(_) => Some(4),
        }
    })
}

trait FullWalkTransitionTable {
    type Cell: Copy;

    fn cell(&mut self, state: u32, byte: u8) -> Self::Cell;

    fn cell_is_dead(cell: Self::Cell) -> bool;

    fn cell_has_finalizer(cell: Self::Cell) -> bool;

    fn cell_target(cell: Self::Cell) -> u32;

    #[inline(always)]
    fn transition(&mut self, state: u32, byte: u8) -> u32 {
        let cell = self.cell(state, byte);
        if Self::cell_is_dead(cell) {
            u32::MAX
        } else {
            Self::cell_target(cell)
        }
    }

    fn root_state(&mut self, state: u32) -> Result<u32, String>;

    /// Some lazy transition adapters retain an error until their caller can
    /// return a Result. A mask must be validated before cache publication.
    #[inline]
    fn validate_mask_result(&mut self) -> Result<(), String> { Ok(()) }

    /// Optional exact/overapproximating input projection supplied by the
    /// transition provider. The byte walker and endpoint checks are unchanged.
    /// Implementations must preserve all admitted continuations of `parser`.
    #[inline]
    fn root_state_for_parser(&mut self, state: u32, _parser: &ParserStacks) -> Result<u32, String> {
        self.root_state(state)
    }

    #[inline(always)]
    fn walk_initial_state(
        &mut self,
        constraint: &Constraint,
        vocab: &DynamicMaskVocab,
    ) -> Result<u32, String> {
        self.root_state(vocab.mask_runtime_state(constraint.tokenizer.initial_state()))
    }

    #[inline(always)]
    fn parser_projection_enabled(&self) -> bool { false }

    #[inline(always)]
    fn parser_initial_state(&mut self, initial: u32, _admitted: &BitSet) -> u32 {
        initial
    }

    #[inline(always)]
    fn small_future_terminals(&self, _state: u32) -> Option<SmallVec<[TerminalID; 4]>> {
        None
    }

    fn finalizer_code(&self, state: u32) -> u32;

    fn single_finalizer_continues(&mut self, state: u32) -> bool;

    fn matched_terminals(&self, state: u32) -> SmallVec<[TerminalID; 4]>;

    fn future_contains(&mut self, state: u32, terminal: TerminalID) -> bool;

    fn future_intersects(&mut self, state: u32, terminals: &BitSet) -> bool;

    /// Merge several lexer coordinates that carry the same parser object. This
    /// is only an optimization: returning `None` keeps the branches separate.
    fn merge_states(&mut self, states: &[u32]) -> Option<u32>;

    /// Dense deterministic coordinates use a vector cache for token-boundary
    /// checks. Lazy NFA-subset coordinates use the sparse cache instead.
    fn dense_state_count(&self) -> Option<usize>;

    fn token_boundary_allowed(
        &mut self,
        parser_cache: &mut FullWalkParserCache,
        constraint: &Constraint,
        initial_lexer_state: u32,
        lexer_state: u32,
        parser_node: u32,
    ) -> bool;

    /// Return the exact raw tokenizer state when this transition coordinate is
    /// a singleton raw state. The direct first-match lane uses this only as a
    /// safety check before bypassing the generic subset/config representation.
    fn exact_raw_state(&self, state: u32) -> Option<u32>;

    /// Exact set of bytes with at least one lexical successor from this runtime
    /// coordinate. Implementations may decline when the coordinate is not
    /// cheaply enumerable. Used only as a root-trie scheduler; token admission
    /// remains with the ordinary exact walker.
    #[inline(always)]
    fn exact_next_bytes(&mut self, _state: u32) -> Option<U8Set> { None }

    #[inline(always)]
    fn terminal_is_ignore(&self, constraint: &Constraint, terminal: TerminalID) -> bool {
        Some(terminal) == constraint.ignore_terminal
    }

    /// Recursive composition can route one completed terminal into a different
    /// active lexer leaf (or several leaves after nullable CALL/RETURN closure).
    /// Ordinary tokenizers have one global reset state and keep the established
    /// parser-cache fast path instead.
    #[inline(always)]
    fn uses_scoped_reset_routing(&self) -> bool { false }

    /// The ordinary full walk keeps parser-conditioned prefix pruning opt-in
    /// because its extra check can outweigh the saved work on broad grammars.
    /// Recursive composition previously performed this exact liveness check on
    /// every radix prefix, so enable it by default for that provider only.
    #[inline(always)]
    fn parser_conditioned_dead_skip_default(&self) -> bool { false }

    /// Output-only pruning: skip a subtree only when all its original aliases
    /// are already present in the caller's mask. Never publish partial results
    /// in a cache keyed only by the vocabulary and parser state.
    #[inline(always)]
    fn pre_admitted_original_mask(&self) -> Option<&[u32]> { None }

    #[inline(always)]
    fn product_transition_cache_capacity(&self) -> usize { 0 }

    #[inline(always)]
    fn cache_two_branch_products(&self) -> bool { false }

    /// A scoped/config provider can retain singleton parser/lexer products in
    /// the same exact cache rather than repeatedly rebuilding its scalar lane.
    /// Ordinary providers keep their specialised scalar/hot traversal instead.
    #[inline(always)]
    fn cache_single_branch_products(&self) -> bool { false }

    /// Complete provider walks can defer output until the exact admitted and
    /// rejected populations are known. This reuses the ordinary adaptive
    /// emitter instead of touching every rejected token in a sparse mask.
    #[inline(always)]
    fn prefer_adaptive_output(&self) -> bool { false }

    fn scoped_reset_branches(
        &mut self,
        _parser_cache: &mut FullWalkParserCache,
        _constraint: &Constraint,
        _parser_node: u32,
        _terminal: TerminalID,
    ) -> SmallVec<[(u32, u32); 4]> {
        SmallVec::new()
    }

    #[inline(always)]
    fn hot_state_id(&self, _state: u32) -> Option<u8> { None }

    #[inline(always)]
    fn hot_state_from_id(&self, _id: u8) -> u32 { u32::MAX }

    #[inline(always)]
    fn hot_step(&mut self, _state: u8, _byte: u8) -> u8 { FULL_WALK_HOT_SLOW }

    #[inline(always)]
    fn hot_scalar_lane_enabled(&self) -> bool { false }
    /// If `state` is a singleton virtual residual owned by `terminal`, return
    /// its direct residual coordinate when that residual can still extend.
    /// This is used only by the maximal-munch guard: it avoids materializing a
    /// succession of exact virtual runtime states merely to ask whether the
    /// previously matched terminal can keep consuming bytes.
    #[inline(always)]
    fn direct_prune_coordinates(
        &self,
        _state: u32,
        _terminal: TerminalID,
    ) -> Option<SmallVec<[VirtualResidualDirectCoordinate; 4]>> {
        None
    }

    /// Advance a direct maximal-munch memory by one byte. Implementations only
    /// receive coordinates previously returned by `direct_prune_coordinates`.
    #[inline(always)]
    fn direct_prune_step(
        &self,
        _coordinate: VirtualResidualDirectCoordinate,
        _byte: u8,
    ) -> DirectPruneStep {
        DirectPruneStep::Dead
    }

    #[inline(always)]
    fn terminal_residual_prune_step(
        &self,
        _coordinate: TerminalResidualDirectCoordinate,
        _byte: u8,
    ) -> TerminalResidualPruneStep {
        TerminalResidualPruneStep::Dead
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DirectPruneStep {
    Dead,
    Matched,
    Live(VirtualResidualDirectCoordinate),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum FirstMatchDirectCoordinate {
    Virtual(VirtualResidualDirectCoordinate),
    Physical(TerminalResidualDirectCoordinate),
}

impl FirstMatchDirectCoordinate {
    #[inline(always)]
    fn step(self, tokenizer: &Tokenizer, byte: u8) -> Option<Self> {
        match self {
            Self::Virtual(c) => tokenizer.virtual_residual_direct_coordinate_step(c, byte).map(Self::Virtual),
            Self::Physical(c) => tokenizer.terminal_residual_direct_coordinate_step(c, byte).map(Self::Physical),
        }
    }
    #[inline(always)]
    fn accepting(self, tokenizer: &Tokenizer) -> Option<bool> {
        match self {
            Self::Virtual(c) => tokenizer.virtual_residual_direct_coordinate_accepting(c),
            Self::Physical(c) => tokenizer.terminal_residual_direct_coordinate_accepting(c),
        }
    }
    #[inline(always)]
    fn has_future(self, tokenizer: &Tokenizer) -> Option<bool> {
        match self {
            Self::Virtual(c) => tokenizer.virtual_residual_direct_coordinate_has_future(c),
            Self::Physical(c) => tokenizer.terminal_residual_direct_coordinate_has_future(c),
        }
    }
    #[inline(always)]
    fn raw_state(self, tokenizer: &Tokenizer) -> Option<u32> {
        match self {
            Self::Virtual(c) => tokenizer.virtual_residual_state_for_direct_coordinate(c),
            Self::Physical(c) => Some(tokenizer.terminal_residual_direct_coordinate_raw_state(c)),
        }
    }
    #[inline(always)]
    fn physical(self) -> Option<TerminalResidualDirectCoordinate> {
        match self { Self::Physical(c) => Some(c), Self::Virtual(_) => None }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TerminalResidualPruneStep {
    Dead,
    Matched,
    Live(TerminalResidualDirectCoordinate),
}

#[derive(Clone, Copy)]
struct FullWalkFlat16<'a> {
    transitions: &'a [u16],
    finalizer_code: &'a [u32],
    single_finalizer_continues: &'a [u8],
    tokenizer: &'a Tokenizer,
    vocab: &'a DynamicMaskVocab,
}

impl FullWalkTransitionTable for FullWalkFlat16<'_> {
    type Cell = u16;

    #[inline(always)]
    fn cell(&mut self, state: u32, byte: u8) -> u16 {
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

    #[inline(always)]
    fn root_state(&mut self, state: u32) -> Result<u32, String> { Ok(state) }

    #[inline(always)]
    fn finalizer_code(&self, state: u32) -> u32 {
        unsafe { *self.finalizer_code.get_unchecked(state as usize) }
    }

    #[inline(always)]
    fn single_finalizer_continues(&mut self, state: u32) -> bool {
        unsafe { *self.single_finalizer_continues.get_unchecked(state as usize) != 0 }
    }

    #[inline]
    fn matched_terminals(&self, state: u32) -> SmallVec<[TerminalID; 4]> {
        self.tokenizer.matched_terminals_slice(state).iter().copied().collect()
    }

    #[inline(always)]
    fn future_contains(&mut self, state: u32, terminal: TerminalID) -> bool {
        self.tokenizer.possible_future_terminals(state).contains(terminal as usize)
    }

    #[inline(always)]
    fn future_intersects(&mut self, state: u32, terminals: &BitSet) -> bool {
        !terminals.is_disjoint(self.tokenizer.possible_future_terminals(state))
    }

    fn merge_states(&mut self, states: &[u32]) -> Option<u32> {
        self.vocab.mask_projection_state_for_projection_states(states)
    }

    #[inline(always)]
    fn dense_state_count(&self) -> Option<usize> { Some(self.tokenizer.num_states() as usize) }

    #[inline(always)]
    fn token_boundary_allowed(
        &mut self,
        parser_cache: &mut FullWalkParserCache,
        constraint: &Constraint,
        initial_lexer_state: u32,
        lexer_state: u32,
        parser_node: u32,
    ) -> bool {
        parser_cache.token_boundary_allowed_dense(
            constraint,
            self.tokenizer,
            initial_lexer_state,
            lexer_state,
            parser_node,
        )
    }

    #[inline(always)]
    fn exact_raw_state(&self, state: u32) -> Option<u32> { Some(state) }

}

#[derive(Clone, Copy)]
struct FullWalkFlat32<'a> {
    transitions: &'a [u32],
    finalizer_code: &'a [u32],
    single_finalizer_continues: &'a [u8],
    tokenizer: &'a Tokenizer,
    vocab: &'a DynamicMaskVocab,
}

impl FullWalkTransitionTable for FullWalkFlat32<'_> {
    type Cell = u32;

    #[inline(always)]
    fn cell(&mut self, state: u32, byte: u8) -> u32 {
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

    #[inline(always)]
    fn root_state(&mut self, state: u32) -> Result<u32, String> { Ok(state) }

    #[inline(always)]
    fn finalizer_code(&self, state: u32) -> u32 {
        unsafe { *self.finalizer_code.get_unchecked(state as usize) }
    }

    #[inline(always)]
    fn single_finalizer_continues(&mut self, state: u32) -> bool {
        unsafe { *self.single_finalizer_continues.get_unchecked(state as usize) != 0 }
    }

    #[inline]
    fn matched_terminals(&self, state: u32) -> SmallVec<[TerminalID; 4]> {
        self.tokenizer.matched_terminals_slice(state).iter().copied().collect()
    }

    #[inline(always)]
    fn future_contains(&mut self, state: u32, terminal: TerminalID) -> bool {
        self.tokenizer.possible_future_terminals(state).contains(terminal as usize)
    }

    #[inline(always)]
    fn future_intersects(&mut self, state: u32, terminals: &BitSet) -> bool {
        !terminals.is_disjoint(self.tokenizer.possible_future_terminals(state))
    }

    fn merge_states(&mut self, states: &[u32]) -> Option<u32> {
        self.vocab.mask_projection_state_for_projection_states(states)
    }

    #[inline(always)]
    fn dense_state_count(&self) -> Option<usize> { Some(self.tokenizer.num_states() as usize) }

    #[inline(always)]
    fn token_boundary_allowed(
        &mut self,
        parser_cache: &mut FullWalkParserCache,
        constraint: &Constraint,
        initial_lexer_state: u32,
        lexer_state: u32,
        parser_node: u32,
    ) -> bool {
        parser_cache.token_boundary_allowed_dense(
            constraint,
            self.tokenizer,
            initial_lexer_state,
            lexer_state,
            parser_node,
        )
    }

    #[inline(always)]
    fn exact_raw_state(&self, state: u32) -> Option<u32> { Some(state) }

}

#[derive(Clone, Copy)]
struct RecursiveFullWalkCell {
    target: u32,
    has_finalizer: bool,
}

/// Exact deterministic lexer view for the recursive/provider runtime.
///
/// The ordinary full-walk engine consumes one integer lexer coordinate.  A
/// provider-native composition already has exactly such a coordinate: the
/// disjoint union of intact leaf tokenizer states. Keep the provider mapping
/// here, outside the vocabulary traversal, so the walk itself need not know
/// whether a state belongs to a monolithic tokenizer or a composed leaf.
///
/// This scalar shared-engine lane accepts finite, epsilon-free leaves.
/// Virtual/epsilon leaves use `recursive_provider`'s config/subset adapter over
/// the same vocabulary traversal. The independent recursive walker is test-only;
/// unsupported provider initialization reports an error rather than falling back.
#[derive(Clone, Copy)]
struct RecursiveFullWalkLeaf<'a> {
    constraint: &'a Constraint,
    lexer_offset: u32,
    lexer_count: u32,
    terminal_offset: u32,
    terminal_count: u32,
    reset: u32,
}

struct RecursiveFullWalkTransitions<'a> {
    constraint: &'a Constraint,
    total_states: usize,
    // Resolve the immutable composition tree once per mask, not several times
    // per byte/finalizer/guard. Offsets are the authoritative layout's exact
    // coordinates, so this view requires no DFA or vocabulary preprocessing.
    leaves: SmallVec<[RecursiveFullWalkLeaf<'a>; 4]>,
    token_boundary_cache: FxHashMap<(u32, u32), bool>,
    parser_advance_cache: FxHashMap<(u32, TerminalID), SmallVec<[(u32, u32); 4]>>,
    parser_canonicalizer: RecursiveParserCanonicalizer,
    parser_gss_cache: Vec<Option<ParserGSS>>,
    cache_parser_gss: bool,
    pre_admitted_original_mask: Option<&'a [u32]>,
}



#[derive(Default)]
struct RecursiveParserCanonicalizer {
    semantic: Option<Box<RecursiveParserSemanticCache>>,
}

struct RecursiveParserSemanticCache {
    keys: crate::ds::leveled_gss::GssSemanticKeyInterner<u32, ()>,
    nodes: FxHashMap<u32, u32>,
}

impl RecursiveParserCanonicalizer {
    fn intern(&mut self, stacks: ParserStacks, cache: &mut FullWalkParserCache) -> u32 {
        static THRESHOLD: OnceLock<usize> = OnceLock::new();
        let threshold = *THRESHOLD.get_or_init(|| {
            std::env::var("GLRMASK_PROFILE_PARSER_INTERN_THRESHOLD").ok()
                .and_then(|v| v.parse().ok()).unwrap_or(32)
        });
        self.intern_with_threshold(stacks, cache, threshold)
    }

    fn intern_with_threshold(
        &mut self,
        stacks: ParserStacks,
        cache: &mut FullWalkParserCache,
        threshold: usize,
    ) -> u32 {
        if self.semantic.is_none() {
            if cache.nodes.len() < threshold {
                return cache.intern_stacks(stacks);
            }
            // Canonicalize exact finite stack languages, not pointer identity
            // or just stack tops. Accumulators are unit here; exclusion guards
            // remain correlated in FullWalkBranch and are never merged away.
            let mut semantic = RecursiveParserSemanticCache {
                keys: crate::ds::leveled_gss::GssSemanticKeyInterner::with_budget(
                    16_384, 8_192, 16_384,
                ),
                nodes: FxHashMap::default(),
            };
            for (index, node) in cache.nodes.iter().enumerate() {
                let key = semantic.keys.key(&node.gss);
                if semantic.keys.is_exhausted() { break; }
                semantic.nodes.entry(key).or_insert(index as u32);
            }
            self.semantic = Some(Box::new(semantic));
        }
        let semantic = self.semantic.as_mut().expect("initialized above");
        if semantic.keys.is_exhausted() { return cache.intern_stacks(stacks); }
        let key = semantic.keys.key(&stacks);
        // Budget exhaustion must never let the interner's sentinel represent
        // a real parser language. The established exact path remains valid.
        if semantic.keys.is_exhausted() { return cache.intern_stacks(stacks); }
        if let Some(&node) = semantic.nodes.get(&key) { return node; }
        let node = cache.push_stacks(stacks);
        semantic.nodes.insert(key, node);
        node
    }
}

impl<'a> RecursiveFullWalkTransitions<'a> {
    fn parser_gss(&mut self, node: u32, cache: &FullWalkParserCache) -> ParserGSS {
        if !self.cache_parser_gss { return with_empty_accumulators(&cache.nodes[node as usize].gss); }
        self.parser_gss_cache.resize_with(cache.nodes.len(), || None);
        self.parser_gss_cache[node as usize].get_or_insert_with(|| {
            with_empty_accumulators(&cache.nodes[node as usize].gss)
        }).clone()
    }

    fn new(constraint: &'a Constraint) -> Option<Self> {
        let provider = Self::new_for_parser_routing(constraint)?;
        provider.leaves.iter().all(|leaf| {
            !leaf.constraint.tokenizer_has_epsilon_transitions
                && !leaf.constraint.tokenizer.has_any_virtual_runtime()
        }).then_some(provider)
    }

    fn new_for_parser_routing(constraint: &'a Constraint) -> Option<Self> {
        let layout = constraint.recursive_parser_layout().ok().flatten()?;
        if layout.leaves.is_empty() {
            return None;
        }
        let mut leaves = SmallVec::new();
        for leaf_index in 0..layout.leaves.len() {
            let leaf = constraint.recursive_leaf_constraint(leaf_index)?;
            leaves.push(RecursiveFullWalkLeaf {
                constraint: leaf,
                lexer_offset: layout.leaf_tokenizer_state_offsets[leaf_index],
                lexer_count: leaf.tokenizer.num_states(),
                terminal_offset: layout.outer_terminal_count
                    .checked_add(layout.leaf_terminal_offsets[leaf_index])?,
                terminal_count: leaf.parser_terminal_count(),
                reset: constraint.recursive_tokenizer_reset_state(leaf_index)?,
            });
        }
        Some(Self {
            constraint,
            total_states: layout.total_tokenizer_states as usize,
            leaves,
            token_boundary_cache: FxHashMap::default(),
            parser_advance_cache: FxHashMap::default(),
            parser_canonicalizer: RecursiveParserCanonicalizer::default(),
            parser_gss_cache: Vec::new(),
            cache_parser_gss: env_flag("GLRMASK_PROFILE_PARSER_ACC_CACHE", true),
            pre_admitted_original_mask: None,

        })
    }

    #[inline(always)]
    fn leaf_local(&self, state: u32) -> Option<(usize, u32, &'a Constraint)> {
        if state as usize >= self.total_states { return None; }
        let index = self.leaves.partition_point(|leaf| leaf.lexer_offset <= state)
            .checked_sub(1)?;
        let leaf = &self.leaves[index];
        let local = state - leaf.lexer_offset;
        (local < leaf.lexer_count).then_some((index, local, leaf.constraint))
    }

    #[inline]
    fn mapped_matches(&self, state: u32) -> SmallVec<[TerminalID; 4]> {
        let Some((leaf_index, local_state, leaf)) = self.leaf_local(state) else {
            return SmallVec::new();
        };
        let base = self.leaves[leaf_index].terminal_offset;
        leaf.tokenizer
            .matched_terminals_slice(local_state)
            .iter()
            .map(|&terminal| base + terminal)
            .collect()
    }
}

impl FullWalkTransitionTable for RecursiveFullWalkTransitions<'_> {
    type Cell = RecursiveFullWalkCell;

    #[inline(always)]
    fn cell(&mut self, state: u32, byte: u8) -> Self::Cell {
        let Some((leaf_index, local_state, leaf)) = self.leaf_local(state) else {
            return RecursiveFullWalkCell {
                target: u32::MAX,
                has_finalizer: false,
            };
        };
        let local_target =
            leaf.tokenizer_fast_transitions
                .transition(&leaf.tokenizer, local_state, byte);
        if local_target == u32::MAX {
            return RecursiveFullWalkCell {
                target: u32::MAX,
                has_finalizer: false,
            };
        }
        if local_target >= self.leaves[leaf_index].lexer_count {
            return RecursiveFullWalkCell {
                target: u32::MAX,
                has_finalizer: false,
            };
        }
        let target = self.leaves[leaf_index].lexer_offset + local_target;
        RecursiveFullWalkCell {
            target,
            has_finalizer: leaf
                .tokenizer
                .matched_terminals_slice(local_target)
                .first()
                .is_some(),
        }
    }

    #[inline(always)]
    fn cell_is_dead(cell: Self::Cell) -> bool { cell.target == u32::MAX }

    #[inline(always)]
    fn cell_has_finalizer(cell: Self::Cell) -> bool { cell.has_finalizer }

    #[inline(always)]
    fn cell_target(cell: Self::Cell) -> u32 { cell.target }

    #[inline(always)]
    fn root_state(&mut self, state: u32) -> Result<u32, String> {
        self.leaf_local(state)
            .map(|_| state)
            .ok_or_else(|| format!("recursive full-walk root state {state} is not scoped"))
    }

    #[inline(always)]
    fn walk_initial_state(
        &mut self,
        constraint: &Constraint,
        _vocab: &DynamicMaskVocab,
    ) -> Result<u32, String> {
        debug_assert!(std::ptr::eq(constraint, self.constraint));
        Ok(self.leaves[0].reset)
    }

    #[inline]
    fn finalizer_code(&self, state: u32) -> u32 {
        const NONE: u32 = u32::MAX;
        const MULTI: u32 = u32::MAX - 1;
        let matches = self.mapped_matches(state);
        match matches.as_slice() {
            [] => NONE,
            [terminal] => *terminal,
            _ => MULTI,
        }
    }

    #[inline]
    fn single_finalizer_continues(&mut self, state: u32) -> bool {
        let Some((_, local_state, leaf)) = self.leaf_local(state) else {
            return false;
        };
        let [local_terminal] = leaf.tokenizer.matched_terminals_slice(local_state) else {
            return false;
        };
        leaf.tokenizer.possible_future_terminals(local_state).contains(*local_terminal as usize)
    }

    #[inline]
    fn matched_terminals(&self, state: u32) -> SmallVec<[TerminalID; 4]> {
        self.mapped_matches(state)
    }

    #[inline(always)]
    fn future_contains(&mut self, state: u32, terminal: TerminalID) -> bool {
        let Some((index, local_state, leaf)) = self.leaf_local(state) else { return false; };
        let Some(local_terminal) = terminal.checked_sub(self.leaves[index].terminal_offset)
        else { return false; };
        local_terminal < self.leaves[index].terminal_count
            && leaf.tokenizer.possible_future_terminals(local_state).contains(local_terminal as usize)
    }

    #[inline(always)]
    fn future_intersects(&mut self, state: u32, terminals: &BitSet) -> bool {
        let Some((index, local_state, leaf)) = self.leaf_local(state) else { return false; };
        let offset = self.leaves[index].terminal_offset as usize;
        leaf.tokenizer.possible_future_terminals(local_state).iter_ones()
            .any(|local| terminals.contains(offset + local))
    }

    #[inline]
    fn merge_states(&mut self, states: &[u32]) -> Option<u32> {
        let first = *states.first()?;
        states.iter().all(|&state| state == first).then_some(first)
    }

    #[inline(always)]
    fn dense_state_count(&self) -> Option<usize> { Some(self.total_states) }

    fn token_boundary_allowed(
        &mut self,
        parser_cache: &mut FullWalkParserCache,
        constraint: &Constraint,
        _initial_lexer_state: u32,
        lexer_state: u32,
        parser_node: u32,
    ) -> bool {
        debug_assert!(std::ptr::eq(constraint, self.constraint));
        let Some((index, _, _)) = self.leaf_local(lexer_state) else { return false; };
        if self.leaves[index].reset == lexer_state {
            return true;
        }
        let key = (lexer_state, parser_node);
        if let Some(&cached) = self.token_boundary_cache.get(&key) {
            return cached;
        }
        let Some(future) = constraint.recursive_tokenizer_future_scoped_terminals(lexer_state)
        else {
            self.token_boundary_cache.insert(key, false);
            return false;
        };
        let ignored = future
            .iter_ones()
            .any(|terminal| constraint.recursive_terminal_is_ignore(terminal as u32));
        let allowed = ignored || {
            let parser_gss = self.parser_gss(parser_node, parser_cache);
            constraint
                .compact_segmented_parser_may_advance_on_any(&parser_gss, &future)
                .unwrap_or(false)
        };
        self.token_boundary_cache.insert(key, allowed);
        allowed
    }

    #[inline(always)]
    fn exact_raw_state(&self, state: u32) -> Option<u32> { Some(state) }

    #[inline(always)]
    fn terminal_is_ignore(&self, constraint: &Constraint, terminal: TerminalID) -> bool {
        debug_assert!(std::ptr::eq(constraint, self.constraint));
        let Some(index) = self.leaves.partition_point(|leaf| leaf.terminal_offset <= terminal)
            .checked_sub(1) else { return false; };
        let leaf = &self.leaves[index];
        leaf.constraint.ignore_terminal == Some(terminal - leaf.terminal_offset)
    }

    fn pre_admitted_original_mask(&self) -> Option<&[u32]> { self.pre_admitted_original_mask }

    #[inline(always)]
    fn uses_scoped_reset_routing(&self) -> bool { true }

    #[inline(always)]
    fn parser_conditioned_dead_skip_default(&self) -> bool { true }

    #[inline(always)]
    fn product_transition_cache_capacity(&self) -> usize {
        #[cfg(test)]
        { return TEST_RECURSIVE_PRODUCT_CACHE_CAPACITY.with(|v| v.get()); }
        #[cfg(not(test))]
        { 256 }
    }

    #[inline(always)]
    fn prefer_adaptive_output(&self) -> bool { true }

    fn scoped_reset_branches(
        &mut self,
        parser_cache: &mut FullWalkParserCache,
        constraint: &Constraint,
        parser_node: u32,
        terminal: TerminalID,
    ) -> SmallVec<[(u32, u32); 4]> {
        debug_assert!(std::ptr::eq(constraint, self.constraint));
        let key = (parser_node, terminal);
        if let Some(cached) = self.parser_advance_cache.get(&key) {
            return cached.clone();
        }

        let parser_gss = self.parser_gss(parser_node, parser_cache);
        let advanced = if constraint.recursive_terminal_is_ignore(terminal) {
            // Exact recursive commit closes zero-width CALL/RETURN controls
            // before deciding which leaf owns the tokenizer reset for an
            // ignored terminal. Mirror queue_parser_ignored_reset_state here.
            constraint
                .close_compact_segmented_parser(&parser_gss)
                .unwrap_or(parser_gss)
        } else {
            constraint
                .advance_compact_segmented_parser(&parser_gss, terminal)
                .unwrap_or_else(ParserGSS::empty)
        };
        let mut result = SmallVec::<[(u32, u32); 4]>::new();
        if !advanced.is_empty()
            && let Some(partitions) =
                constraint.partition_recursive_parser_gss_by_active_leaf(&advanced)
        {
            for (leaf_index, partition) in partitions {
                let Some(leaf) = self.leaves.get(leaf_index) else {
                    continue;
                };
                let reset = leaf.reset;
                let stacks = partition.apply(|_| ());
                let node = self.parser_canonicalizer.intern(stacks, parser_cache);
                if !result.contains(&(reset, node)) {
                    result.push((reset, node));
                }
            }
        }
        self.parser_advance_cache.insert(key, result.clone());
        result
    }
}

#[derive(Clone, Copy)]
struct FullWalkConfigCell {
    target: u32,
    has_finalizer: bool,
}

#[derive(Clone)]
struct FullWalkHotScalarCache {
    raw_to_hot: FxHashMap<u32, u8>,
    hot_to_raw: Vec<u32>,
    rows: Vec<[u8; 256]>,
    misses: usize,
    slow: usize,
}

impl FullWalkHotScalarCache {
    fn new() -> Self {
        Self {
            raw_to_hot: FxHashMap::default(),
            hot_to_raw: Vec::new(),
            rows: Vec::new(),
            misses: 0,
            slow: 0,
        }
    }

    #[inline(always)]
    fn tagged(id: u8) -> u32 { FULL_WALK_HOT_TAG | u32::from(id) }

    #[inline(always)]
    fn id(state: u32) -> Option<u8> {
        ((state & 0xffff_ff00) == FULL_WALK_HOT_TAG)
            .then_some((state & FULL_WALK_HOT_MASK) as u8)
    }

    #[inline(always)]
    fn raw(&self, state: u32) -> Option<u32> {
        self.hot_to_raw.get(Self::id(state)? as usize).copied()
    }

    #[inline]
    fn intern(&mut self, raw: u32) -> Option<u8> {
        if let Some(&id) = self.raw_to_hot.get(&raw) {
            return Some(id);
        }
        if self.hot_to_raw.len() >= FULL_WALK_HOT_CAPACITY {
            return None;
        }
        let id = self.hot_to_raw.len() as u8;
        self.raw_to_hot.insert(raw, id);
        self.hot_to_raw.push(raw);
        self.rows.push([FULL_WALK_HOT_UNKNOWN; 256]);
        Some(id)
    }
}

mod raw_high_memo;

// Dense lookups remain unchanged for small IDs. High IDs must not allocate
// arrays proportional to a sparse coordinate merely to cache a few cells.
const RAW_DENSE_MEMO_LIMIT: u32 = 1024;

/// Strict-walk transition backend for the lexer representation selected by the
/// ordinary runtime. In a deterministic tokenizer the config id is simply the
/// raw state id. In an epsilon-NFA tokenizer it is a `DynamicNfaScanCache`
/// handle for an epsilon-closed subset. Subsets are interned lazily only when
/// the complete vocabulary walk actually reaches them.
struct FullWalkConfigTransitions<'a, 'b> {
    cache: &'a mut DynamicNfaScanCache<'b>,
    error: Option<String>,
    raw_cell_rows: Vec<Option<Box<[u64; 256]>>>,
    bounded_high_raw: bool,
    high_raw_cells: raw_high_memo::TaggedMemo<256>,
    high_raw_targets: raw_high_memo::TaggedMemo<64>,
    hot_scalar: FullWalkHotScalarCache,
    hot_enabled: bool,
    hot_persist_key: Option<(usize, u64)>,
    raw_target_cells: Vec<u64>,
    virtual_dense_row_by_raw_state: Vec<u32>,
    virtual_dense_row_ids: FxHashMap<(u32, u32), u32>,
    virtual_dense_rows: Vec<Box<[u64; 256]>>,
    virtual_dense_cell_cache_enabled: bool,
    profile: bool,
    cell_calls: usize,
    raw_cell_hits: usize,
    raw_cell_misses: usize,
    physical_raw_cell_misses: usize,
    virtual_raw_cell_misses: usize,
    physical_raw_transition_ns: u64,
    virtual_raw_transition_ns: u64,
    raw_config_for_start_ns: u64,
    raw_has_finalizer_ns: u64,
    future_contains_calls: usize,
    future_contains_cache: FxHashMap<(u32, TerminalID), bool>,
    future_contains_by_terminal: FxHashMap<TerminalID, usize>,
    future_contains_by_state: FxHashMap<(u32, TerminalID), usize>,
    future_contains_ns: u64,
    future_intersects_calls: usize,
    future_intersects_ns: u64,
    single_finalizer_continues_calls: usize,
    single_finalizer_continues_ns: u64,
    max_raw_state_seen: u32,
    config_cell_calls: usize,
    step_calls_start: usize,
    step_cache_hits_start: usize,
    step_cache_misses_start: usize,
    physical_states_scanned_start: usize,
    intern_calls_start: usize,
    intern_hits_start: usize,
    intern_new_start: usize,
    profile_max_token_len: usize,
    profile_virtual_exact_sources: FxHashSet<u32>,
    profile_virtual_dense_sources: FxHashSet<(u32, u32)>,
    profile_virtual_exact_pairs: FxHashSet<(u32, u8)>,
    profile_virtual_dense_pairs: FxHashSet<(u32, u32, u8)>,
}

thread_local! {
    // Diagnostic persistence store for the hot-scalar oracle. Move the cache
    // in/out rather than cloning it so a same-generation warm fill measures
    // traversal rather than persistence bookkeeping. This is intentionally
    // thread-scoped and is not yet a production cache-lifecycle design.
    static FULL_WALK_HOT_PERSIST_CACHE: std::cell::RefCell<
        FxHashMap<(usize, u64), FullWalkHotScalarCache>
    > = std::cell::RefCell::new(FxHashMap::default());
}

#[inline]
fn full_walk_hot_persist_take(key: (usize, u64)) -> Option<FullWalkHotScalarCache> {
    FULL_WALK_HOT_PERSIST_CACHE.with(|cache| cache.borrow_mut().remove(&key))
}

#[inline]
fn full_walk_hot_persist_put(key: (usize, u64), hot: FullWalkHotScalarCache) {
    FULL_WALK_HOT_PERSIST_CACHE.with(|cache| {
        cache.borrow_mut().insert(key, hot);
    });
}

impl<'a, 'b> FullWalkConfigTransitions<'a, 'b> {
    fn new(cache: &'a mut DynamicNfaScanCache<'b>, max_token_len: usize, generation: u64) -> Self {
    let profile = cache.profile_transition_work;
    let (
        step_calls_start,
        step_cache_hits_start,
        step_cache_misses_start,
        physical_states_scanned_start,
        intern_calls_start,
        intern_hits_start,
        intern_new_start,
    ) = (
        cache.profile_step_calls,
        cache.profile_step_cache_hits,
        cache.profile_step_cache_misses,
        cache.profile_physical_states_scanned,
        cache.profile_intern_calls,
        cache.profile_intern_hits,
        cache.profile_intern_new,
    );
    let hot_enabled = env_flag("GLRMASK_EXPERIMENT_HOT_SCALAR_CACHE", true);
    let hot_persist_key = (hot_enabled
        && env_flag("GLRMASK_EXPERIMENT_HOT_SCALAR_PERSIST", false))
        .then(|| {
            (
                cache.tokenizer() as *const Tokenizer as usize,
                generation,
            )
        });
    let hot_scalar = if hot_enabled {
        hot_persist_key
            .and_then(full_walk_hot_persist_take)
            .unwrap_or_else(FullWalkHotScalarCache::new)
    } else {
        FullWalkHotScalarCache::new()
    };
    Self {
        cache: cache,
        error: None,
        raw_cell_rows: Vec::new(),
        bounded_high_raw: {
            static ENABLED: OnceLock<bool> = OnceLock::new();
            *ENABLED.get_or_init(||env_flag("GLRMASK_BOUNDED_HIGH_RAW_MEMO", true))
        },
        high_raw_cells: raw_high_memo::TaggedMemo::new(),
        high_raw_targets: raw_high_memo::TaggedMemo::new(),
        hot_scalar,
        hot_enabled,
        hot_persist_key,
        raw_target_cells: Vec::new(),
        virtual_dense_row_by_raw_state: Vec::new(),
        virtual_dense_row_ids: FxHashMap::default(),
        virtual_dense_rows: Vec::new(),
        virtual_dense_cell_cache_enabled: dynamic_virtual_dense_cell_cache_experiment_enabled(),
        profile,
        cell_calls: 0,
        raw_cell_hits: 0,
        raw_cell_misses: 0,
        physical_raw_cell_misses: 0,
        virtual_raw_cell_misses: 0,
        physical_raw_transition_ns: 0,
        virtual_raw_transition_ns: 0,
        raw_config_for_start_ns: 0,
        raw_has_finalizer_ns: 0,
        future_contains_calls: 0,
        future_contains_cache: FxHashMap::default(),
        future_contains_by_terminal: FxHashMap::default(),
        future_contains_by_state: FxHashMap::default(),
        future_contains_ns: 0,
        future_intersects_calls: 0,
        future_intersects_ns: 0,
        single_finalizer_continues_calls: 0,
        single_finalizer_continues_ns: 0,
        max_raw_state_seen: 0,
        config_cell_calls: 0,
        step_calls_start,
        step_cache_hits_start,
        step_cache_misses_start,
        physical_states_scanned_start,
        intern_calls_start,
        intern_hits_start,
        intern_new_start,
        profile_max_token_len: max_token_len,
        profile_virtual_exact_sources: FxHashSet::default(),
        profile_virtual_dense_sources: FxHashSet::default(),
        profile_virtual_exact_pairs: FxHashSet::default(),
        profile_virtual_dense_pairs: FxHashSet::default(),
    }    }
}

impl FullWalkConfigTransitions<'_, '_> {
    fn virtual_dense_row_for_raw_state(&mut self, raw_state: u32) -> Option<usize> {
        const UNKNOWN: u32 = u32::MAX;
        const NONE: u32 = u32::MAX - 1;
        let raw_index = raw_state as usize;
        if self.virtual_dense_row_by_raw_state.len() <= raw_index {
            self.virtual_dense_row_by_raw_state
                .resize(raw_index + 1, UNKNOWN);
        }
        let cached = self.virtual_dense_row_by_raw_state[raw_index];
        if cached != UNKNOWN {
            return (cached != NONE).then_some(cached as usize);
        }
        let Some(coordinate) = self
            .cache
            .tokenizer()
            .virtual_residual_direct_coordinate(raw_state)
        else {
            self.virtual_dense_row_by_raw_state[raw_index] = NONE;
            return None;
        };
        let Some(key) = self
            .cache
            .tokenizer()
            .virtual_residual_direct_coordinate_finite_mask_dense_key(
                coordinate,
                self.profile_max_token_len,
            )
        else {
            self.virtual_dense_row_by_raw_state[raw_index] = NONE;
            return None;
        };
        let row = if let Some(&row) = self.virtual_dense_row_ids.get(&key) {
            row
        } else {
            let row = u32::try_from(self.virtual_dense_rows.len()).ok()?;
            self.virtual_dense_row_ids.insert(key, row);
            self.virtual_dense_rows
                .push(Box::new([u64::MAX; 256]));
            row
        };
        self.virtual_dense_row_by_raw_state[raw_index] = row;
        Some(row as usize)
    }

    fn finish(mut self) -> Result<(), String> {
        if self.hot_enabled && std::env::var_os("GLRMASK_PROFILE_HOT_SCALAR_CACHE").is_some() {
            let hot = &self.hot_scalar;
            eprintln!(
                "[glrmask/profile][hot_scalar_cache] states={} bytes={} misses={} slow={}",
                hot.hot_to_raw.len(),
                hot.rows.len() * 256,
                hot.misses,
                hot.slow,
            );
        }
        if self.profile {
            eprintln!(
                "[glrmask/profile][config_transition_work] cell_calls={} raw_cell_hits={} raw_cell_misses={} config_cell_calls={} step_calls={} step_cache_hits={} step_cache_misses={} physical_states_scanned={} intern_calls={} intern_hits={} intern_new={} configs_total={}",
                self.cell_calls,
                self.raw_cell_hits,
                self.raw_cell_misses,
                self.config_cell_calls,
                self.cache.profile_step_calls.saturating_sub(self.step_calls_start),
                self.cache.profile_step_cache_hits.saturating_sub(self.step_cache_hits_start),
                self.cache.profile_step_cache_misses.saturating_sub(self.step_cache_misses_start),
                self.cache.profile_physical_states_scanned.saturating_sub(self.physical_states_scanned_start),
                self.cache.profile_intern_calls.saturating_sub(self.intern_calls_start),
                self.cache.profile_intern_hits.saturating_sub(self.intern_hits_start),
                self.cache.profile_intern_new.saturating_sub(self.intern_new_start),
                self.cache.configs.len(),
            );
            eprintln!(
                "[glrmask/profile][config_raw_work] physical_raw_misses={} virtual_raw_misses={} physical_transition_ms={:.3} virtual_transition_ms={:.3} config_for_start_ms={:.3} has_finalizer_ms={:.3} max_raw_state={} raw_rows_len={} raw_rows_materialized={}",
                self.physical_raw_cell_misses,
                self.virtual_raw_cell_misses,
                self.physical_raw_transition_ns as f64 / 1e6,
                self.virtual_raw_transition_ns as f64 / 1e6,
                self.raw_config_for_start_ns as f64 / 1e6,
                self.raw_has_finalizer_ns as f64 / 1e6,
                self.max_raw_state_seen,
                self.raw_cell_rows.len(),
                self.raw_cell_rows.iter().filter(|row| row.is_some()).count(),
            );
            eprintln!(
                "[glrmask/profile][config_member_transition_work] physical_calls={} physical_ms={:.3} virtual_residual_calls={} virtual_residual_ms={:.3}",
                self.cache.profile_member_physical_transitions,
                self.cache.profile_member_physical_transition_ns as f64 / 1e6,
                self.cache.profile_member_virtual_residual_transitions,
                self.cache.profile_member_virtual_residual_transition_ns as f64 / 1e6,
            );
            if !self.cache.profile_member_virtual_residual_by_terminal.is_empty() {
                let mut rows = self
                    .cache
                    .profile_member_virtual_residual_by_terminal
                    .iter()
                    .map(|(&terminal, &(calls, ns))| (terminal, calls, ns))
                    .collect::<Vec<_>>();
                rows.sort_unstable_by_key(|&(terminal, _, _)| terminal);
                eprintln!(
                    "[glrmask/profile][config_virtual_transition_by_terminal] {:?}",
                    rows
                );
            }
            eprintln!(
                "[glrmask/profile][config_query_work] future_contains_calls={} future_contains_ms={:.3} future_intersects_calls={} future_intersects_ms={:.3} single_finalizer_continues_calls={} single_finalizer_continues_ms={:.3}",
                self.future_contains_calls,
                self.future_contains_ns as f64 / 1e6,
                self.future_intersects_calls,
                self.future_intersects_ns as f64 / 1e6,
                self.single_finalizer_continues_calls,
                self.single_finalizer_continues_ns as f64 / 1e6,
            );
            if !self.future_contains_by_terminal.is_empty() {
                let mut rows = self
                    .future_contains_by_terminal
                    .iter()
                    .map(|(&terminal, &calls)| (terminal, calls))
                    .collect::<Vec<_>>();
                rows.sort_unstable_by_key(|&(terminal, _)| terminal);
                eprintln!("[glrmask/profile][future_contains_by_terminal] {:?}", rows);
            }
            if !self.future_contains_by_state.is_empty() {
                let mut rows = self
                    .future_contains_by_state
                    .iter()
                    .map(|(&(state, terminal), &calls)| (calls, state, terminal))
                    .collect::<Vec<_>>();
                rows.sort_unstable_by(|a, b| b.cmp(a));
                rows.truncate(12);
                eprintln!(
                    "[glrmask/profile][future_contains_states] distinct={} top={:?}",
                    self.future_contains_by_state.len(),
                    rows
                );
            }
            eprintln!(
                "[glrmask/profile][config_virtual_projection_work] exact_sources={} dense_sources={} exact_pairs={} dense_pairs={}",
                self.profile_virtual_exact_sources.len(),
                self.profile_virtual_dense_sources.len(),
                self.profile_virtual_exact_pairs.len(),
                self.profile_virtual_dense_pairs.len(),
            );
        }
        if let Some(key) = self.hot_persist_key
            && self.hot_enabled
        {
            let hot = std::mem::replace(&mut self.hot_scalar, FullWalkHotScalarCache::new());
            full_walk_hot_persist_put(key, hot);
        }
        self.error.map_or(Ok(()), Err)
    }

    fn config_future_contains_exact(&mut self, state: u32, terminal: TerminalID) -> bool {
        match self.cache.config_future_contains_exact(state, terminal) {
            Ok(value) => value,
            Err(error) => {
                if self.error.is_none() { self.error = Some(error); }
                false
            }
        }
    }

    fn config_future_intersects_exact(&mut self, state: u32, terminals: &BitSet) -> bool {
        match self.cache.config_future_intersects_exact(state, terminals) {
            Ok(value) => value,
            Err(error) => {
                if self.error.is_none() { self.error = Some(error); }
                false
            }
        }
    }

    #[cold]
    #[inline(never)]
    fn hot_cell_slow(&mut self, hot_id: u8, byte: u8) -> FullWalkConfigCell {
        let Some(raw_source) = self.hot_scalar.hot_to_raw.get(hot_id as usize).copied()
        else {
            self.error = Some("invalid full-walk hot scalar state".to_owned());
            return FullWalkConfigCell { target: u32::MAX, has_finalizer: false };
        };
        let raw_target = self.cache.transition(raw_source, byte);
        if raw_target == u32::MAX {
            self.hot_scalar.rows[hot_id as usize][byte as usize] = FULL_WALK_HOT_DEAD;
            return FullWalkConfigCell { target: u32::MAX, has_finalizer: false };
        }

        let tokenizer = self.cache.tokenizer();
        let simple = !tokenizer.state_has_epsilon_transitions(raw_target)
            && !tokenizer.state_is_virtual_runtime(raw_target)
            && tokenizer.matched_terminal_bitset(raw_target).is_empty();
        if simple {
            let target_hot = self.hot_scalar.intern(raw_target);
            if let Some(target_hot) = target_hot {
                self.hot_scalar.rows[hot_id as usize][byte as usize] = target_hot;
                return FullWalkConfigCell {
                    target: FullWalkHotScalarCache::tagged(target_hot),
                    has_finalizer: false,
                };
            }
        }

        self.hot_scalar.rows[hot_id as usize][byte as usize] = FULL_WALK_HOT_SLOW;
        self.hot_scalar.slow += 1;
        match self.cache.config_for_raw_start(raw_target) {
            Ok(target) => FullWalkConfigCell {
                target,
                has_finalizer: self.cache.config_has_finalizer(target),
            },
            Err(error) => {
                self.error = Some(error);
                FullWalkConfigCell { target: u32::MAX, has_finalizer: false }
            }
        }
    }

    #[inline(always)]
    fn hot_raw(&self, state: u32) -> Option<u32> {
        self.hot_enabled.then(|| self.hot_scalar.raw(state)).flatten()
    }

    #[inline(always)]
    fn generic_config_for_state(&self, state: u32) -> u32 {
        self.hot_raw(state)
            .map(DynamicNfaScanCache::raw_config)
            .unwrap_or(state)
    }
}

impl FullWalkTransitionTable for FullWalkConfigTransitions<'_, '_> {
    type Cell = FullWalkConfigCell;

    fn validate_mask_result(&mut self) -> Result<(), String> {
        self.error.as_ref().map_or(Ok(()), |error| Err(error.clone()))
    }


    #[inline(always)]
    fn cell(&mut self, state: u32, byte: u8) -> Self::Cell {
        const UNKNOWN: u64 = u64::MAX;
        if let Some(hot_id) = FullWalkHotScalarCache::id(state) {
            let cached = unsafe {
                *self
                    .hot_scalar
                    .rows
                    .get_unchecked(hot_id as usize)
                    .get_unchecked(byte as usize)
            };
            if cached < FULL_WALK_HOT_SLOW {
                return FullWalkConfigCell {
                    target: FullWalkHotScalarCache::tagged(cached),
                    has_finalizer: false,
                };
            }
            if cached == FULL_WALK_HOT_DEAD {
                return FullWalkConfigCell { target: u32::MAX, has_finalizer: false };
            }
            if cached == FULL_WALK_HOT_SLOW {
                return self.hot_cell_slow(hot_id, byte);
            }
            self.hot_scalar.misses += 1;
            return self.hot_cell_slow(hot_id, byte);
        }
        if self.profile {
            self.cell_calls += 1;
        }
        let raw_state = self.cache.raw_state_for_config(state);
        let mut virtual_dense_row = None;
        if let Some(raw_state) = raw_state {
            if self.profile {
                self.max_raw_state_seen = self.max_raw_state_seen.max(raw_state);
            }
            if self.bounded_high_raw && raw_state >= RAW_DENSE_MEMO_LIMIT {
                let key=(u64::from(raw_state)<<8)|u64::from(byte);
                if let Some(packed)=self.high_raw_cells.get(key) {
                    if self.profile { self.raw_cell_hits += 1; }
                    return FullWalkConfigCell { target:packed as u32, has_finalizer:(packed>>32)&1 != 0 };
                }
            }
            let raw_index = raw_state as usize;
            if raw_index < self.raw_cell_rows.len()
                && let Some(row) = unsafe { self.raw_cell_rows.get_unchecked(raw_index) }
            {
                let packed = unsafe { *row.get_unchecked(byte as usize) };
                if packed != UNKNOWN {
                    if self.profile {
                        self.raw_cell_hits += 1;
                    }
                    return FullWalkConfigCell {
                        target: packed as u32,
                        has_finalizer: (packed >> 32) & 1 != 0,
                    };
                }
            }
            if self.virtual_dense_cell_cache_enabled
                && raw_state >= self.cache.tokenizer().num_states()
                && let Some(row_id) = self.virtual_dense_row_for_raw_state(raw_state)
            {
                let packed = self.virtual_dense_rows[row_id][byte as usize];
                if packed != UNKNOWN {
                    return FullWalkConfigCell {
                        target: packed as u32,
                        has_finalizer: (packed >> 32) & 1 != 0,
                    };
                }
                virtual_dense_row = Some(row_id);
            }
            if self.profile {
                self.raw_cell_misses += 1;
                if raw_state < self.cache.tokenizer().num_states() {
                    self.physical_raw_cell_misses += 1;
                } else {
                    self.virtual_raw_cell_misses += 1;
                    self.profile_virtual_exact_sources.insert(raw_state);
                    self.profile_virtual_exact_pairs.insert((raw_state, byte));
                    if let Some(coordinate) = self
                        .cache
                        .tokenizer()
                        .virtual_residual_direct_coordinate(raw_state)
                        && let Some((runtime, projected)) = self
                            .cache
                            .tokenizer()
                            .virtual_residual_direct_coordinate_finite_mask_dense_key(
                                coordinate,
                                self.profile_max_token_len,
                            )
                    {
                        self.profile_virtual_dense_sources.insert((runtime, projected));
                        self.profile_virtual_dense_pairs.insert((runtime, projected, byte));
                    }
                }
            }
        } else if self.profile {
            self.config_cell_calls += 1;
        }
        if self.error.is_some() {
            return FullWalkConfigCell {
                target: u32::MAX,
                has_finalizer: false,
            };
        }
        let cell = if let Some(raw_source) = raw_state {
            let transition_started = self.profile.then(std::time::Instant::now);
            let raw_target = self.cache.transition(raw_source, byte);
            if let Some(started) = transition_started {
                let elapsed = started.elapsed().as_nanos() as u64;
                if raw_source < self.cache.tokenizer().num_states() {
                    self.physical_raw_transition_ns =
                        self.physical_raw_transition_ns.saturating_add(elapsed);
                } else {
                    self.virtual_raw_transition_ns =
                        self.virtual_raw_transition_ns.saturating_add(elapsed);
                }
            }
            if raw_target == u32::MAX {
                FullWalkConfigCell {
                    target: u32::MAX,
                    has_finalizer: false,
                }
            } else {
                let raw_target_index = raw_target as usize;
                let cached_target = if self.bounded_high_raw && raw_target >= RAW_DENSE_MEMO_LIMIT {
                    self.high_raw_targets.get(u64::from(raw_target))
                } else if raw_target_index < self.raw_target_cells.len() {
                    let packed = self.raw_target_cells[raw_target_index];
                    (packed != UNKNOWN).then_some(packed)
                } else {
                    None
                };
                if let Some(packed) = cached_target {
                    FullWalkConfigCell {
                        target: packed as u32,
                        has_finalizer: (packed >> 32) & 1 != 0,
                    }
                } else {
                    let config_started = self.profile.then(std::time::Instant::now);
                    match self.cache.config_for_raw_start(raw_target) {
                        Ok(target) => {
                            if let Some(started) = config_started {
                                self.raw_config_for_start_ns = self
                                    .raw_config_for_start_ns
                                    .saturating_add(started.elapsed().as_nanos() as u64);
                            }
                            let finalizer_started = self.profile.then(std::time::Instant::now);
                            let has_finalizer = self.cache.config_has_finalizer(target);
                            if let Some(started) = finalizer_started {
                                self.raw_has_finalizer_ns = self
                                    .raw_has_finalizer_ns
                                    .saturating_add(started.elapsed().as_nanos() as u64);
                            }
                            let packed=u64::from(target) | ((has_finalizer as u64) << 32);
                            if self.bounded_high_raw && raw_target >= RAW_DENSE_MEMO_LIMIT {
                                self.high_raw_targets.insert(u64::from(raw_target), packed);
                            } else {
                                if self.raw_target_cells.len() <= raw_target_index {
                                    self.raw_target_cells.resize(raw_target_index + 1, UNKNOWN);
                                }
                                self.raw_target_cells[raw_target_index] = packed;
                            }
                            FullWalkConfigCell {
                                target,
                                has_finalizer,
                            }
                        },
                        Err(error) => {
                            self.error = Some(error);
                            FullWalkConfigCell {
                                target: u32::MAX,
                                has_finalizer: false,
                            }
                        }
                    }
                }
            }
        } else {
            match self.cache.step_config(state, byte) {
                Ok(Some(target)) => FullWalkConfigCell {
                    target,
                    has_finalizer: self.cache.config_has_finalizer(target),
                },
                Ok(None) => FullWalkConfigCell {
                    target: u32::MAX,
                    has_finalizer: false,
                },
                Err(error) => {
                    self.error = Some(error);
                    FullWalkConfigCell {
                        target: u32::MAX,
                        has_finalizer: false,
                    }
                }
            }
        };
        let mut returned_cell = cell;
        if !cell.has_finalizer
            && cell.target != u32::MAX
            && let Some(raw_target) = self.cache.raw_state_for_config(cell.target)
            && self.hot_enabled
            && !self.cache.tokenizer().state_has_epsilon_transitions(raw_target)
            && !self.cache.tokenizer().state_is_virtual_runtime(raw_target)
            && self.cache.tokenizer().matched_terminal_bitset(raw_target).is_empty()
            && let Some(id) = self.hot_scalar.intern(raw_target)
        {
            returned_cell = FullWalkConfigCell {
                target: FullWalkHotScalarCache::tagged(id),
                has_finalizer: false,
            };
        }
        if self.error.is_none()
            && let Some(raw_state) = raw_state
        {
            let packed=u64::from(returned_cell.target) | ((returned_cell.has_finalizer as u64)<<32);
            if self.bounded_high_raw && raw_state >= RAW_DENSE_MEMO_LIMIT {
                self.high_raw_cells.insert((u64::from(raw_state)<<8)|u64::from(byte),packed);
            } else {
                let raw_state = raw_state as usize;
                if self.raw_cell_rows.len() <= raw_state {
                    self.raw_cell_rows.resize_with(raw_state + 1, || None);
                }
                let row = self.raw_cell_rows[raw_state]
                    .get_or_insert_with(|| Box::new([UNKNOWN; 256]));
                let slot = unsafe { row.get_unchecked_mut(byte as usize) };
                if *slot == UNKNOWN { *slot = packed; }
            }
        }
        if self.error.is_none()
            && let Some(row_id) = virtual_dense_row
        {
            let slot = &mut self.virtual_dense_rows[row_id][byte as usize];
            if *slot == UNKNOWN {
                *slot = u64::from(returned_cell.target)
                    | ((returned_cell.has_finalizer as u64) << 32);
            }
        }
        returned_cell
    }

    #[inline(always)]
    fn cell_is_dead(cell: Self::Cell) -> bool { cell.target == u32::MAX }

    #[inline(always)]
    fn cell_has_finalizer(cell: Self::Cell) -> bool { cell.has_finalizer }

    #[inline(always)]
    fn cell_target(cell: Self::Cell) -> u32 { cell.target }

    fn root_state(&mut self, state: u32) -> Result<u32, String> {
        self.cache.config_for_raw_start(state)
    }

    #[inline(always)]
    fn parser_projection_enabled(&self) -> bool {
        self.cache.parser_projection_enabled
            && !self.cache.deterministic
            && self.cache.use_constraint_fast_transitions
    }

    fn parser_initial_state(&mut self, initial: u32, admitted: &BitSet) -> u32 {
        let source = self.cache.tokenizer().initial_state();
        match self.cache.config_for_parser_admitted(source, admitted) {
            Ok(Some(config)) => match self.cache.fresh_reset_config(config) {
                Ok(reset) => reset,
                Err(error) => {
                    if self.error.is_none() { self.error = Some(error); }
                    initial
                }
            },
            Ok(None) => initial,
            Err(error) => {
                if self.error.is_none() { self.error = Some(error); }
                initial
            }
        }
    }

    fn small_future_terminals(&self, state: u32) -> Option<SmallVec<[TerminalID; 4]>> {
        if !self.parser_projection_enabled() {
            return None;
        }
        let config = self.generic_config_for_state(state);
        let futures = if let Some(raw) = self.cache.raw_state_for_config(config) {
            self.cache.tokenizer().possible_future_terminals(raw)
        } else {
            &self.cache.config_futures[self.cache.config_index(config)?]
        };
        (futures.count_ones() <= 16).then(|| {
            futures.iter_ones().map(|terminal| terminal as TerminalID).collect()
        })
    }

    fn finalizer_code(&self, state: u32) -> u32 {
        self.cache.config_finalizer_code(self.generic_config_for_state(state))
    }

    fn single_finalizer_continues(&mut self, state: u32) -> bool {
        let started = self.profile.then(std::time::Instant::now);
        if self.profile {
            self.single_finalizer_continues_calls += 1;
        }
        let state = self.generic_config_for_state(state);
        let code = self.cache.config_finalizer_code(state);
        let result = code != u32::MAX
            && code != u32::MAX - 1
            && self.config_future_contains_exact(state, code);
        if let Some(started) = started {
            self.single_finalizer_continues_ns = self
                .single_finalizer_continues_ns
                .saturating_add(started.elapsed().as_nanos() as u64);
        }
        result
    }

    fn matched_terminals(&self, state: u32) -> SmallVec<[TerminalID; 4]> {
        self.cache.config_matched_terminals(self.generic_config_for_state(state))
    }

    fn future_contains(&mut self, state: u32, terminal: TerminalID) -> bool {
        // Hot scalar IDs and ordinary config IDs can name the same residual.
        // Normalize before BOTH lookup and insertion, not just insertion.
        let state = self.generic_config_for_state(state);
        if let Some(&cached) = self.future_contains_cache.get(&(state, terminal)) {
            return cached;
        }
        let started = self.profile.then(std::time::Instant::now);
        if self.profile {
            self.future_contains_calls += 1;
            *self.future_contains_by_terminal.entry(terminal).or_default() += 1;
            *self.future_contains_by_state.entry((state, terminal)).or_default() += 1;
        }
        let result = self.config_future_contains_exact(state, terminal);
        self.future_contains_cache.insert((state, terminal), result);
        if let Some(started) = started {
            self.future_contains_ns = self
                .future_contains_ns
                .saturating_add(started.elapsed().as_nanos() as u64);
        }
        result
    }

    fn future_intersects(&mut self, state: u32, terminals: &BitSet) -> bool {
        let started = self.profile.then(std::time::Instant::now);
        if self.profile {
            self.future_intersects_calls += 1;
        }
        let state = self.generic_config_for_state(state);
        let result = self.config_future_intersects_exact(state, terminals);
        if let Some(started) = started {
            self.future_intersects_ns = self
                .future_intersects_ns
                .saturating_add(started.elapsed().as_nanos() as u64);
        }
        result
    }

    fn merge_states(&mut self, states: &[u32]) -> Option<u32> {
        let normalized = states
            .iter()
            .copied()
            .map(|state| self.generic_config_for_state(state))
            .collect::<SmallVec<[u32; 4]>>();
        match self.cache.union_configs(&normalized) {
            Ok(state) => state,
            Err(error) => {
                if self.error.is_none() { self.error = Some(error); }
                None
            }
        }
    }

    #[inline(always)]
    fn dense_state_count(&self) -> Option<usize> { None }

    #[inline(always)]
    fn token_boundary_allowed(
        &mut self,
        parser_cache: &mut FullWalkParserCache,
        constraint: &Constraint,
        initial_lexer_state: u32,
        lexer_state: u32,
        parser_node: u32,
    ) -> bool {
        if lexer_state == initial_lexer_state
            || parser_cache.nodes[parser_node as usize].reset_lexer == Some(lexer_state)
        {
            return true;
        }
        if let Some(hot_id) = FullWalkHotScalarCache::id(lexer_state) {
            return parser_cache.token_boundary_allowed_hot(
                constraint,
                self,
                lexer_state,
                hot_id,
                parser_node,
            );
        }
        parser_cache.token_boundary_allowed_sparse(constraint, self, lexer_state, parser_node)
    }

    #[inline(always)]
    fn exact_raw_state(&self, state: u32) -> Option<u32> {
        self.hot_raw(state).or_else(|| self.cache.raw_state_for_config(state))
    }

    #[inline(always)]
    fn exact_next_bytes(&mut self, state: u32) -> Option<U8Set> {
        let state = self.generic_config_for_state(state);
        Some(self.cache.config_next_bytes(state))
    }

    #[inline(always)]
    fn hot_state_id(&self, state: u32) -> Option<u8> {
        FullWalkHotScalarCache::id(state)
    }

    #[inline(always)]
    fn hot_state_from_id(&self, id: u8) -> u32 { FullWalkHotScalarCache::tagged(id) }

    #[inline(always)]
    fn hot_step(&mut self, state: u8, byte: u8) -> u8 {
        let cached = unsafe {
            *self
                .hot_scalar
                .rows
                .get_unchecked(state as usize)
                .get_unchecked(byte as usize)
        };
        if cached != FULL_WALK_HOT_UNKNOWN {
            return cached;
        }
        self.hot_scalar.misses += 1;
        let cell = self.hot_cell_slow(state, byte);
        if cell.target == u32::MAX {
            FULL_WALK_HOT_DEAD
        } else if let Some(id) = FullWalkHotScalarCache::id(cell.target) {
            id
        } else {
            FULL_WALK_HOT_SLOW
        }
    }

    #[inline(always)]
    fn hot_scalar_lane_enabled(&self) -> bool { self.hot_enabled }
    fn direct_prune_coordinates(
        &self,
        state: u32,
        terminal: TerminalID,
    ) -> Option<SmallVec<[VirtualResidualDirectCoordinate; 4]>> {
        // A guard can reach a hot scalar alias after its original virtual
        // residual dies. As in every other metadata query, translate that
        // alias before treating the ID as a raw-state/subset coordinate.
        let state = self.generic_config_for_state(state);
        let tokenizer = self.cache.tokenizer();
        let mut direct = SmallVec::<[VirtualResidualDirectCoordinate; 4]>::new();
        for index in 0..self.cache.config_len(state) {
            let raw = self.cache.config_state(state, index);
            if tokenizer.virtual_residual_terminal_for_state(raw) == Some(terminal) {
                let coordinate = tokenizer.virtual_residual_direct_coordinate(raw)?;
                if !tokenizer
                    .virtual_residual_direct_coordinate_has_future(coordinate)?
                {
                    continue;
                }
                if !direct.contains(&coordinate) {
                    direct.push(coordinate);
                }
                continue;
            }
            // A physical/other-runtime member that may still contribute the
            // same terminal means a single direct residual coordinate is not
            // a complete representation of this maximal-munch memory.
            if tokenizer
                .possible_future_terminals(raw)
                .contains(terminal as usize)
            {
                return None;
            }
        }
        Some(direct)
    }

    #[inline(always)]
    fn direct_prune_step(
        &self,
        coordinate: VirtualResidualDirectCoordinate,
        byte: u8,
    ) -> DirectPruneStep {
        let tokenizer = self.cache.tokenizer();
        let Some(target) = tokenizer.virtual_residual_direct_coordinate_step(coordinate, byte)
        else {
            return DirectPruneStep::Dead;
        };
        if tokenizer
            .virtual_residual_direct_coordinate_accepting(target)
            .unwrap_or(false)
        {
            return DirectPruneStep::Matched;
        }
        if tokenizer
            .virtual_residual_direct_coordinate_has_future(target)
            .unwrap_or(false)
        {
            DirectPruneStep::Live(target)
        } else {
            DirectPruneStep::Dead
        }
    }

    fn terminal_residual_prune_step(
        &self,
        coordinate: TerminalResidualDirectCoordinate,
        byte: u8,
    ) -> TerminalResidualPruneStep {
        let tokenizer = self.cache.tokenizer();
        let Some(target) = tokenizer.terminal_residual_direct_coordinate_step(coordinate, byte) else {
            return TerminalResidualPruneStep::Dead;
        };
        if tokenizer.terminal_residual_direct_coordinate_accepting(target).unwrap_or(false) {
            TerminalResidualPruneStep::Matched
        } else if tokenizer.terminal_residual_direct_coordinate_has_future(target).unwrap_or(false) {
            TerminalResidualPruneStep::Live(target)
        } else {
            TerminalResidualPruneStep::Dead
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum FullWalkPruneGuard {
    Passed,
    Pending(SmallVec<[FullWalkPruneMemory; 2]>),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum FullWalkPruneMemory {
    Exact(u32, TerminalID),
    Direct(VirtualResidualDirectCoordinate, TerminalID),
    TerminalResidual(TerminalResidualDirectCoordinate, TerminalID),
}

impl FullWalkPruneGuard {
    fn from_initial<T: FullWalkTransitionTable>(
        guard: &InitialPruneGuard,
        transitions: &mut T,
    ) -> Result<Self, String> {
        match guard {
            InitialPruneGuard::Passed => Ok(Self::Passed),
            InitialPruneGuard::Pending { memories } => {
                let mut projected = SmallVec::<[FullWalkPruneMemory; 2]>::new();
                for &(state, _, terminal) in memories.iter() {
                    // `InitialPruneGuard::new` already stores lexer states in
                    // the mask-runtime tokenizer coordinate. Do not project
                    // them a second time here: virtual projection state IDs
                    // are not exact source-runtime state IDs.
                    let state = transitions.root_state(state)?;
                    if let Some(coordinates) = transitions.direct_prune_coordinates(state, terminal)
                    {
                        for coordinate in coordinates {
                            let memory = FullWalkPruneMemory::Direct(coordinate, terminal);
                            if !projected.contains(&memory) {
                                projected.push(memory);
                            }
                        }
                    } else {
                        projected.push(FullWalkPruneMemory::Exact(state, terminal));
                    }
                }
                Ok(if projected.is_empty() {
                    Self::Passed
                } else {
                    Self::Pending(projected)
                })
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
        transitions: &mut T,
        byte: u8,
    ) -> Option<Self> {
        let Self::Pending(memories) = self else {
            return Some(Self::Passed);
        };
        let mut next = SmallVec::<[FullWalkPruneMemory; 2]>::new();
        for &memory in memories {
            match memory {
                FullWalkPruneMemory::Direct(coordinate, terminal) => {
                    match transitions.direct_prune_step(coordinate, byte) {
                        DirectPruneStep::Dead => {}
                        DirectPruneStep::Matched => return None,
                        DirectPruneStep::Live(target) => {
                            let memory = FullWalkPruneMemory::Direct(target, terminal);
                            if !next.contains(&memory) {
                                next.push(memory);
                            }
                        }
                    }
                }
                FullWalkPruneMemory::TerminalResidual(coordinate, terminal) => {
                    match transitions.terminal_residual_prune_step(coordinate, byte) {
                        TerminalResidualPruneStep::Dead => {}
                        TerminalResidualPruneStep::Matched => return None,
                        TerminalResidualPruneStep::Live(target) => {
                            let memory = FullWalkPruneMemory::TerminalResidual(target, terminal);
                            if !next.contains(&memory) { next.push(memory); }
                        }
                    }
                }
                FullWalkPruneMemory::Exact(lexer_state, terminal) => {
                    let target = transitions.transition(lexer_state, byte);
                    if target == u32::MAX {
                        continue;
                    }
                    if transitions.matched_terminals(target).contains(&terminal) {
                        return None;
                    }
                    if let Some(coordinates) = transitions.direct_prune_coordinates(target, terminal)
                    {
                        for coordinate in coordinates {
                            let memory = FullWalkPruneMemory::Direct(coordinate, terminal);
                            if !next.contains(&memory) {
                                next.push(memory);
                            }
                        }
                    } else if transitions.future_contains(target, terminal) {
                        let memory = FullWalkPruneMemory::Exact(target, terminal);
                        if !next.contains(&memory) {
                            next.push(memory);
                        }
                    }
                }
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
        transitions: &mut T,
        lexer_state: u32,
        terminal: TerminalID,
    ) -> Self {
        if let Some(coordinates) = transitions.direct_prune_coordinates(lexer_state, terminal) {
            if coordinates.is_empty() {
                return self.clone();
            }
            let mut memories = match self {
                Self::Passed => SmallVec::new(),
                Self::Pending(memories) => memories.clone(),
            };
            for coordinate in coordinates {
                let memory = FullWalkPruneMemory::Direct(coordinate, terminal);
                if !memories.contains(&memory) {
                    memories.push(memory);
                }
            }
            return Self::Pending(memories);
        }
        if !transitions.future_contains(lexer_state, terminal) {
            return self.clone();
        }
        let mut memories = match self {
            Self::Passed => SmallVec::new(),
            Self::Pending(memories) => memories.clone(),
        };
        let memory = FullWalkPruneMemory::Exact(lexer_state, terminal);
        if !memories.contains(&memory) {
            memories.push(memory);
        }
        Self::Pending(memories)
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct FullWalkBranch {
    lexer_state: u32,
    parser_node: u32,
    prune_guard: FullWalkPruneGuard,
}

type FullWalkBranches = SmallVec<[FullWalkBranch; 4]>;

#[derive(Clone, PartialEq, Eq, Hash)]
enum FullWalkManyState {
    Branches(FullWalkBranches),
    ThreeSameParser {
        lexers: (u32, u32, u32),
        parser_node: u32,
    },
}

struct FullWalkManyTransitionCache {
    ids: FxHashMap<FullWalkManyState, u32>,
    states: Vec<FullWalkManyState>,
    // Product-state caches are deliberately bounded (currently <= 1024 for
    // recursive providers), so u32 targets waste half of this hot table and
    // double its cache footprint. Keep the public/internal state IDs as u32,
    // but encode cached row targets compactly. Capacity is clamped to the
    // representable exact range; hitting that cache bound only falls back to
    // the existing owned-state path and never drops a branch.
    rows: Vec<Box<[u16; 256]>>,
    boundary_allowed: Vec<Option<bool>>,
    // Exact byte classifications for each retained product: (self-loop,
    // non-self-loop). Unknown bytes can be evaluated through step_handle.
    loop_alphabets: Vec<([u64; 4], [u64; 4])>,
    has_self_loop: Vec<bool>,
    stays_many: Vec<bool>,
    capacity: usize,
    cache_two: bool,
    cache_single: bool,
    profile: bool,
    calls: usize,
    hits: usize,
    misses: usize,
    exhausted: usize,
    max_branches: usize,
}

/// Materialize only owned DFS depths actually saved by the shared walker.
/// Cached products retain their small IDs separately and never need a slot.
#[inline]
fn full_walk_owned_many_slot<T>(
    storage: &mut SmallVec<[Option<T>; 8]>,
    depth: usize,
    stack_len: usize,
) -> &mut Option<T> {
    assert!(depth < stack_len, "owned DFS depth exceeds the trie bound");
    let needed = depth + 1;
    if storage.len() < needed {
        if storage.capacity() < needed {
            // Grow geometrically without exceeding the certified heap bound.
            let target = storage.capacity().saturating_mul(2).max(needed).min(stack_len);
            storage.reserve_exact(target - storage.len());
        }
        storage.resize_with(needed, || None);
    }
    &mut storage[depth]
}

impl FullWalkManyTransitionCache {
    const UNKNOWN: u32 = u32::MAX;
    const DEAD: u32 = u32::MAX - 1;
    const ROW_UNKNOWN: u16 = u16::MAX;
    const ROW_DEAD: u16 = u16::MAX - 1;
    const ROW_MAX_ID: usize = (u16::MAX - 2) as usize;

    fn new(capacity: usize, cache_two: bool, cache_single: bool) -> Self {
        Self {
            ids: FxHashMap::default(),
            states: Vec::new(),
            rows: Vec::new(),
            boundary_allowed: Vec::new(),
            loop_alphabets: Vec::new(),
            has_self_loop: Vec::new(),
            stays_many: Vec::new(),
            capacity: capacity.min(Self::ROW_MAX_ID + 1),
            cache_two,
            cache_single,
            profile: std::env::var_os("GLRMASK_PROFILE_DYNAMIC_CONFIG_TRANSITIONS").is_some(),
            calls: 0,
            hits: 0,
            misses: 0,
            exhausted: 0,
            max_branches: 0,
        }
    }

    fn intern(&mut self, state: &FullWalkManyState) -> Option<u32> {
        if let Some(&id) = self.ids.get(state) {
            return Some(id);
        }
        if self.states.len() >= self.capacity {
            return None;
        }
        let id = self.states.len() as u32;
        let owned = state.clone();
        self.states.push(owned.clone());
        self.rows.push(Box::new([Self::ROW_UNKNOWN; 256]));
        self.boundary_allowed.push(None);
        self.loop_alphabets.push(([0; 4], [0; 4]));
        self.has_self_loop.push(false);
        self.stays_many.push(match state {
            FullWalkManyState::Branches(branches) => {
                // These shapes cannot use the scalar/two or same-parser merge
                // lanes. Reaching the cached shape again need not inspect or
                // copy its guards/frontier merely to rediscover that fact.
                (self.cache_single && branches.len() == 1)
                    || (self.cache_two && branches.len() == 2)
                    || branches.iter().any(|b| !b.prune_guard.is_passed())
                    || (branches.len() > 2 && branches.iter().skip(1)
                        .any(|b| b.parser_node != branches[0].parser_node))
            }
            FullWalkManyState::ThreeSameParser { .. } => false,
        });
        self.ids.insert(owned, id);
        Some(id)
    }

    #[inline]
    fn store_transition(&mut self, source: u32, byte: u8, target: u16) {
        debug_assert_ne!(target, Self::ROW_UNKNOWN);
        self.rows[source as usize][byte as usize] = target;
        let (loops, nonloops) = &mut self.loop_alphabets[source as usize];
        let mask = 1u64 << (byte & 63);
        if u32::from(target) == source {
            self.has_self_loop[source as usize] = true;
            loops[byte as usize >> 6] |= mask;
        } else {
            nonloops[byte as usize >> 6] |= mask;
        }
    }

    #[inline]
    fn cached_loop_alphabet(&self, source: u32, alphabet: [u64; 4]) -> Option<bool> {
        let (loops, nonloops) = self.loop_alphabets.get(source as usize)?;
        if alphabet.iter().zip(nonloops).any(|(a, b)| a & b != 0) {
            Some(false)
        } else if alphabet.iter().zip(loops).all(|(a, b)| a & !b == 0) {
            Some(true)
        } else {
            None
        }
    }

    /// Prove a stable product using only transitions the normal walk has
    /// already evaluated. Completing rows speculatively can be expensive on
    /// large parser products, so an unknown byte simply declines this proof.
    /// If every suffix byte is a self-loop and the boundary admits, induction
    /// proves every descendant token admissible without visiting its edges.
    fn admits_stable_alphabet<T: FullWalkTransitionTable>(
        &mut self, id: u32, alphabet: [u64; 4], initial_lexer_state: u32,
        transitions: &mut T, parser_cache: &mut FullWalkParserCache,
        constraint: &Constraint,
    ) -> bool {
        self.cached_loop_alphabet(id, alphabet) == Some(true)
            && self.token_boundary_allowed(
                id, transitions, parser_cache, constraint, initial_lexer_state,
            )
    }

    #[inline(always)]
    fn cached_many_target(&mut self, source: u32, byte: u8) -> Option<u32> {
        let row = self.rows.get(source as usize)?;
        let encoded = row[byte as usize];
        if encoded >= Self::ROW_DEAD {
            return None;
        }
        let target = u32::from(encoded);
        if !self.stays_many.get(target as usize).copied().unwrap_or(false) { return None; }
        if self.profile { self.calls += 1; self.hits += 1; }
        Some(target)
    }

    /// Advance a bounded, interned parser/lexer product by ID. Cached hits
    /// carry only this ID through the ordinary vocabulary DFS. They must not
    /// hash/clone the complete frontier again at every byte and trie branch.
    /// A full cache retains the existing exact owned-state path, never an
    /// approximate result or a dropped parser alternative.
    fn step_handle<T: FullWalkTransitionTable>(
        &mut self,
        current_id: u32,
        current: &FullWalkManyState,
        byte: u8,
        initial_lexer_state: u32,
        transitions: &mut T,
        parser_cache: &mut FullWalkParserCache,
        constraint: &Constraint,
        uncached_output: &mut FullWalkManyState,
        condition_single: bool,
    ) -> u32 {
        if self.profile {
            self.calls += 1;
            let current = if current_id != Self::UNKNOWN {
                &self.states[current_id as usize]
            } else { current };
            self.max_branches = self.max_branches.max(match current {
                FullWalkManyState::Branches(branches) => branches.len(),
                FullWalkManyState::ThreeSameParser { .. } => 3,
            });
        }
        let source = if current_id != Self::UNKNOWN {
            Some(current_id)
        } else {
            self.intern(current)
        };
        let Some(source) = source else {
            if self.profile { self.exhausted += 1; }
            *uncached_output = full_walk_step_many_state(
                current,
                byte,
                initial_lexer_state,
                transitions,
                parser_cache,
                constraint,
            );
            return Self::UNKNOWN;
        };
        let cached = self.rows[source as usize][byte as usize];
        if cached == Self::ROW_DEAD {
            if self.profile { self.hits += 1; }
            return Self::DEAD;
        }
        if cached != Self::ROW_UNKNOWN {
            if self.profile { self.hits += 1; }
            return u32::from(cached);
        }
        if self.profile { self.misses += 1; }
        let next = full_walk_step_many_state(
            &self.states[source as usize],
            byte,
            initial_lexer_state,
            transitions,
            parser_cache,
            constraint,
        );
        if matches!(&next, FullWalkManyState::Branches(branches) if branches.is_empty()) {
            self.store_transition(source, byte, Self::ROW_DEAD);
            return Self::DEAD;
        }
        // Preserve the existing scalar parser-conditioned dead-prefix proof
        // when keeping a singleton as a product ID. It depends only on this
        // exact parser/lexer state, so both the transition and admission may
        // be memoized for the remainder of this one mask invocation.
        let single_admitted = if self.cache_single && condition_single {
            match &next {
                FullWalkManyState::Branches(branches) => match branches.as_slice() {
                    [branch] if branch.prune_guard.is_passed() => Some(
                        transitions.token_boundary_allowed(
                            parser_cache, constraint, initial_lexer_state,
                            branch.lexer_state, branch.parser_node,
                        ),
                    ),
                    _ => None,
                },
                _ => None,
            }
        } else { None };
        if single_admitted == Some(false) {
            self.store_transition(source, byte, Self::ROW_DEAD);
            return Self::DEAD;
        }
        if let Some(target) = self.intern(&next) {
            debug_assert!(target as usize <= Self::ROW_MAX_ID);
            self.store_transition(source, byte, target as u16);
            if let Some(admitted) = single_admitted {
                self.boundary_allowed[target as usize] = Some(admitted);
            }
            return target;
        }
        *uncached_output = next;
        Self::UNKNOWN
    }

    fn token_boundary_allowed<T: FullWalkTransitionTable>(
        &mut self,
        id: u32,
        transitions: &mut T,
        parser_cache: &mut FullWalkParserCache,
        constraint: &Constraint,
        initial_lexer_state: u32,
    ) -> bool {
        if let Some(result) = self.boundary_allowed[id as usize] { return result; }
        let result = match &self.states[id as usize] {
            FullWalkManyState::Branches(branches) => branches.iter().any(|branch| {
                transitions.token_boundary_allowed(parser_cache, constraint,
                    initial_lexer_state, branch.lexer_state, branch.parser_node)
            }),
            FullWalkManyState::ThreeSameParser { lexers, parser_node } => {
                [lexers.0, lexers.1, lexers.2].into_iter().any(|lexer| {
                    transitions.token_boundary_allowed(parser_cache, constraint,
                        initial_lexer_state, lexer, *parser_node)
                })
            }
        };
        self.boundary_allowed[id as usize] = Some(result);
        result
    }
}

impl Drop for FullWalkManyTransitionCache {
    fn drop(&mut self) {
        if self.profile && self.capacity != 0 {
            eprintln!("[glrmask/profile][many_product_cache] states={} calls={} hits={} misses={} exhausted={} max_branches={}",
                self.states.len(), self.calls, self.hits, self.misses, self.exhausted, self.max_branches);
        }
    }
}

#[cfg(test)]
#[test]
fn singleton_product_cache_keeps_exact_keys_and_capacity() {
    let one = |lexer_state, parser_node| FullWalkManyState::Branches(smallvec::smallvec![
        FullWalkBranch { lexer_state, parser_node, prune_guard: FullWalkPruneGuard::Passed },
    ]);
    let mut cache = FullWalkManyTransitionCache::new(2, true, true);
    let a = cache.intern(&one(7, 3)).unwrap();
    let b = cache.intern(&one(8, 3)).unwrap();
    assert_ne!(a, b);
    assert_eq!(cache.intern(&one(7, 3)), Some(a));
    assert!(cache.stays_many[a as usize]);
    assert!(cache.intern(&one(7, 4)).is_none(), "different parser must not alias a full cache");
    cache.store_transition(a, b'x', b as u16);
    assert_eq!(cache.cached_many_target(a, b'x'), Some(b));
    let alphabet = |bytes: &[u8]| {
        let mut alphabet = [0u64; 4];
        for &byte in bytes { alphabet[byte as usize >> 6] |= 1u64 << (byte & 63); }
        alphabet
    };
    cache.store_transition(a, b'a', a as u16);
    cache.store_transition(a, 0xc2, a as u16);
    assert_eq!(cache.cached_loop_alphabet(a, alphabet(&[b'a', 0xc2])), Some(true));
    assert_eq!(cache.cached_loop_alphabet(a, alphabet(b"ax")), Some(false));
    assert_eq!(cache.cached_loop_alphabet(a, alphabet(b"az")), None);
    assert_eq!(cache.cached_loop_alphabet(b, alphabet(b"a")), None);
    let mut ordinary = FullWalkManyTransitionCache::new(2, true, false);
    let id = ordinary.intern(&one(7, 3)).unwrap();
    assert!(!ordinary.stays_many[id as usize], "ordinary scalar lane stays unchanged");
    assert!(FullWalkManyTransitionCache::new(0, true, true).intern(&one(7, 3)).is_none());
}

struct FullWalkParserNode {
    gss: ParserStacks,
    admitted: Option<BitSet>,
    shared_root_admission: SharedRootAdmission,
    reset_lexer: Option<u32>,
    token_boundary_allowed: Vec<u8>,
    children: SmallVec<[(TerminalID, u32); 16]>,
    last_child_terminal: TerminalID,
    last_child_target: u32,
}

struct FullWalkParserCache {
    nodes: Vec<FullWalkParserNode>,
    dense_lexer_state_count: Option<usize>,
    sparse_token_boundary_allowed: Vec<FxHashMap<u32, u8>>,
    hot_token_boundary_allowed: Vec<[u8; FULL_WALK_HOT_CAPACITY]>,
    profile: bool,
    profile_boundary_calls: usize,
    profile_boundary_hits: usize,
    profile_boundary_misses: usize,
    profile_admitted_builds: usize,
}

impl FullWalkParserCache {
    const DEAD: u32 = u32::MAX;

    fn from_roots(
        root_branches: &DynamicBranches,
        dense_lexer_state_count: Option<usize>,
    ) -> (Self, SmallVec<[u32; 4]>) {
        let mut nodes = Vec::<FullWalkParserNode>::new();
        let mut sparse_token_boundary_allowed = Vec::<FxHashMap<u32, u8>>::new();
        let mut hot_token_boundary_allowed = Vec::<[u8; FULL_WALK_HOT_CAPACITY]>::new();
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
                reset_lexer: None,
                token_boundary_allowed: dense_lexer_state_count
                    .map_or_else(Vec::new, |count| vec![0; count]),
                children: SmallVec::new(),
                last_child_terminal: TerminalID::MAX,
                last_child_target: Self::DEAD,
            });
            sparse_token_boundary_allowed.push(FxHashMap::default());
            hot_token_boundary_allowed.push([0; FULL_WALK_HOT_CAPACITY]);
            root_nodes.push(id);
        }
        (
            Self {
                nodes,
                dense_lexer_state_count,
                sparse_token_boundary_allowed,
                hot_token_boundary_allowed,
                profile: std::env::var_os("GLRMASK_PROFILE_DYNAMIC_CONFIG_TRANSITIONS").is_some(),
                profile_boundary_calls: 0,
                profile_boundary_hits: 0,
                profile_boundary_misses: 0,
                profile_admitted_builds: 0,
            },
            root_nodes,
        )
    }

    fn intern_stacks(&mut self, stacks: ParserStacks) -> u32 {
        if let Some((index, _)) = self
            .nodes
            .iter()
            .enumerate()
            .find(|(_, node)| node.gss.ptr_eq(&stacks))
        {
            return index as u32;
        }
        self.push_stacks(stacks)
    }

    #[inline]
    fn push_stacks(&mut self, stacks: ParserStacks) -> u32 {
        let id = self.nodes.len() as u32;
        self.nodes.push(FullWalkParserNode {
            gss: stacks,
            admitted: None,
            shared_root_admission: None,
            reset_lexer: None,
            token_boundary_allowed: self
                .dense_lexer_state_count
                .map_or_else(Vec::new, |count| vec![0; count]),
            children: SmallVec::new(),
            last_child_terminal: TerminalID::MAX,
            last_child_target: Self::DEAD,
        });
        self.sparse_token_boundary_allowed.push(FxHashMap::default());
        self.hot_token_boundary_allowed.push([0; FULL_WALK_HOT_CAPACITY]);
        id
    }

    #[inline(always)]
    fn advance(
        &mut self,
        constraint: &Constraint,
        node: u32,
        terminal: TerminalID,
    ) -> Option<u32> {
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
            && !constraint.parser_has_controls()
            && self.nodes[node_index]
                .gss
                .single_top_value()
                .is_some_and(|top| !constraint.parser_advance_row_allows(top, terminal))
        {
            self.nodes[node_index].children.push((terminal, Self::DEAD));
            self.nodes[node_index].last_child_terminal = terminal;
            self.nodes[node_index].last_child_target = Self::DEAD;
            return None;
        }
        let next = parser_child(constraint, &self.nodes[node_index].gss, terminal);
        let target = if let Some(gss) = next {
            // Ordinary parser transitions already have their own exact
            // (parent node, terminal) cache. Preserve the original O(1) append;
            // only recursive split/reset routing needs pointer interning.
            self.push_stacks(gss)
        } else {
            Self::DEAD
        };
        self.nodes[node_index].children.push((terminal, target));
        self.nodes[node_index].last_child_terminal = terminal;
        self.nodes[node_index].last_child_target = target;
        (target != Self::DEAD).then_some(target)
    }

    /// Every terminal reset starts the language admitted by the *new* parser
    /// frontier, not the unconstrained union of every lexical terminal.
    #[inline]
    fn initial_lexer(
        &mut self,
        constraint: &Constraint,
        transitions: &mut impl FullWalkTransitionTable,
        initial: u32,
        node: u32,
    ) -> u32 {
        if !transitions.parser_projection_enabled() {
            return initial;
        }
        if let Some(cached) = self.nodes[node as usize].reset_lexer {
            return cached;
        }
        let mut admitted = parser_row_projection_candidates(
            constraint, &self.nodes[node as usize].gss,
        ).map_or_else(|| self.admitted(constraint, node).clone(), |(candidates, _)| candidates);
        if let Some(ignore) = constraint.ignore_terminal {
            admitted.set(ignore as usize);
        }
        let reset = if admitted.count_ones() <= 16 {
            transitions.parser_initial_state(initial, &admitted)
        } else {
            initial
        };
        self.nodes[node as usize].reset_lexer = Some(reset);
        reset
    }

    fn admitted(&mut self, constraint: &Constraint, node: u32) -> &BitSet {
        let index = node as usize;
        if self.nodes[index].admitted.is_none() {
            if self.profile {
                self.profile_admitted_builds += 1;
            }
            let admitted = parser_admission_with_root_memo(
                constraint, &self.nodes[index].gss, &self.nodes[index].shared_root_admission,
            );
            self.nodes[index].admitted = Some(admitted);
        }
        self.nodes[index].admitted.as_ref().unwrap()
    }

    /// For a small lexical future set, query only those actual terminals.
    /// Reuse the same exact parser-child cache needed when one later finalizes;
    /// do not convert/simulate the whole parser stack for every grammar terminal.
    fn narrow_boundary_allowed<T: FullWalkTransitionTable>(
        &mut self,
        constraint: &Constraint,
        transitions: &mut T,
        lexer: u32,
        parser: u32,
    ) -> Option<bool> {
        let candidates = transitions.small_future_terminals(lexer)?;
        Some(candidates.into_iter().any(|terminal| {
            transitions.future_contains(lexer, terminal)
                && (Some(terminal) == constraint.ignore_terminal
                    || self.advance(constraint, parser, terminal).is_some())
        }))
    }

    #[inline(always)]
    fn physical_token_boundary_allowed_dense(
        &mut self,
        constraint: &Constraint,
        tokenizer: &Tokenizer,
        parser_node: u32,
        lexer_state: u32,
    ) -> bool {
        if self.profile {
            self.profile_boundary_calls += 1;
        }
        let node = parser_node as usize;
        let lexer = lexer_state as usize;
        let cached = unsafe {
            *self
                .nodes
                .get_unchecked(node)
                .token_boundary_allowed
                .get_unchecked(lexer)
        };
        if cached != 0 {
            if self.profile {
                self.profile_boundary_hits += 1;
            }
            return cached == 2;
        }
        if self.profile {
            self.profile_boundary_misses += 1;
        }
        let future = tokenizer.possible_future_terminals(lexer_state);
        let allowed = constraint
            .ignore_terminal
            .is_some_and(|terminal| future.contains(terminal as usize))
            || !self.admitted(constraint, parser_node).is_disjoint(future);
        unsafe {
            *self
                .nodes
                .get_unchecked_mut(node)
                .token_boundary_allowed
                .get_unchecked_mut(lexer) = if allowed { 2 } else { 1 };
        }
        allowed
    }

    #[inline(always)]
    fn token_boundary_allowed_dense(
        &mut self,
        constraint: &Constraint,
        tokenizer: &Tokenizer,
        initial_lexer_state: u32,
        lexer_state: u32,
        parser_node: u32,
    ) -> bool {
        lexer_state == initial_lexer_state
            || self.physical_token_boundary_allowed_dense(
                constraint,
                tokenizer,
                parser_node,
                lexer_state,
            )
    }

    #[inline(always)]
    fn token_boundary_allowed_hot<T: FullWalkTransitionTable>(
        &mut self,
        constraint: &Constraint,
        transitions: &mut T,
        lexer_state: u32,
        hot_id: u8,
        parser_node: u32,
    ) -> bool {
        if self.profile {
            self.profile_boundary_calls += 1;
        }
        let node = parser_node as usize;
        let cached = unsafe {
            *self
                .hot_token_boundary_allowed
                .get_unchecked(node)
                .get_unchecked(hot_id as usize)
        };
        if cached != 0 {
            if self.profile {
                self.profile_boundary_hits += 1;
            }
            return cached == 2;
        }
        if self.profile {
            self.profile_boundary_misses += 1;
        }
        let allowed = self.narrow_boundary_allowed(constraint, transitions, lexer_state, parser_node)
            .unwrap_or_else(|| constraint.ignore_terminal.is_some_and(|terminal| {
            transitions.future_contains(lexer_state, terminal)
        }) || transitions.future_intersects(
            lexer_state,
            self.admitted(constraint, parser_node),
        ));
        unsafe {
            *self
                .hot_token_boundary_allowed
                .get_unchecked_mut(node)
                .get_unchecked_mut(hot_id as usize) = if allowed { 2 } else { 1 };
        }
        allowed
    }

    #[inline(always)]
    fn token_boundary_allowed_sparse<T: FullWalkTransitionTable>(
        &mut self,
        constraint: &Constraint,
        transitions: &mut T,
        lexer_state: u32,
        parser_node: u32,
    ) -> bool {
        if self.profile {
            self.profile_boundary_calls += 1;
        }
        let node = parser_node as usize;
        let cached = unsafe { self.sparse_token_boundary_allowed.get_unchecked(node) }
            .get(&lexer_state)
            .copied()
            .unwrap_or(0);
        if cached != 0 {
            if self.profile {
                self.profile_boundary_hits += 1;
            }
            return cached == 2;
        }
        if self.profile {
            self.profile_boundary_misses += 1;
        }
        let allowed = self.narrow_boundary_allowed(constraint, transitions, lexer_state, parser_node)
            .unwrap_or_else(|| constraint.ignore_terminal.is_some_and(|terminal| {
            transitions.future_contains(lexer_state, terminal)
        }) || transitions.future_intersects(
            lexer_state,
            self.admitted(constraint, parser_node),
        ));
        unsafe { self.sparse_token_boundary_allowed.get_unchecked_mut(node) }
            .insert(lexer_state, if allowed { 2 } else { 1 });
        allowed
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

#[inline]
fn full_walk_terminal_reset_branches<T: FullWalkTransitionTable>(
    terminal: TerminalID,
    parser_node: u32,
    initial_lexer_state: u32,
    transitions: &mut T,
    parser_cache: &mut FullWalkParserCache,
    constraint: &Constraint,
) -> SmallVec<[(u32, u32); 4]> {
    if transitions.uses_scoped_reset_routing() {
        return transitions.scoped_reset_branches(
            parser_cache,
            constraint,
            parser_node,
            terminal,
        );
    }
    let mut out = SmallVec::new();
    if let Some(next_parser) = parser_cache.advance(constraint, parser_node, terminal) {
        // Ordinary grammar lanes retain main's exact parser-conditioned reset.
        // Scoped providers return above with their own per-component routes.
        let reset = parser_cache.initial_lexer(
            constraint, transitions, initial_lexer_state, next_parser,
        );
        out.push((reset, next_parser));
    }
    out
}

#[inline(always)]
fn full_walk_continuation_viable<T: FullWalkTransitionTable>(
    lexer_state: u32,
    parser_node: u32,
    initial_lexer_state: u32,
    transitions: &mut T,
    parser_cache: &mut FullWalkParserCache,
    constraint: &Constraint,
) -> bool {
    !transitions.parser_conditioned_dead_skip_default()
        || transitions.token_boundary_allowed(
            parser_cache,
            constraint,
            initial_lexer_state,
            lexer_state,
            parser_node,
        )
}

fn full_walk_classify_finalizer_branches(
    mut branches: FullWalkBranches,
) -> FullWalkScalarFinalizerOutcome {
    match branches.len() {
        0 => unreachable!("finalizer classification requires a continuation branch"),
        1 => FullWalkScalarFinalizerOutcome::Scalar(
            branches.pop().expect("one full-walk branch disappeared"),
        ),
        2 => {
            let second = branches.pop().expect("second full-walk branch disappeared");
            let first = branches.pop().expect("first full-walk branch disappeared");
            FullWalkScalarFinalizerOutcome::Two(first, second)
        }
        _ => FullWalkScalarFinalizerOutcome::Many(branches),
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn full_walk_step_two<T: FullWalkTransitionTable>(
    branches: ((u32, u32), (u32, u32)),
    byte: u8,
    initial_lexer_state: u32,
    transitions: &mut T,
    parser_cache: &mut FullWalkParserCache,
    constraint: &Constraint,
) -> FullWalkTwoStepOutcome {
    let first_cell = transitions.cell(branches.0.0, byte);
    let second_cell = transitions.cell(branches.1.0, byte);

    let first_continuation_live = !T::cell_is_dead(first_cell)
        && (T::cell_has_finalizer(first_cell)
            || full_walk_continuation_viable(
                T::cell_target(first_cell),
                branches.0.1,
                initial_lexer_state,
                transitions,
                parser_cache,
                constraint,
            ));
    let second_continuation_live = !T::cell_is_dead(second_cell)
        && (T::cell_has_finalizer(second_cell)
            || full_walk_continuation_viable(
                T::cell_target(second_cell),
                branches.1.1,
                initial_lexer_state,
                transitions,
                parser_cache,
                constraint,
            ));

    // Dominant correlated-parser case: both deterministic lexer branches stay
    // alive without finalizing, while their exact parser identities differ.
    // Keep the correlation tuple intact and skip the general option/collapse
    // classification below.
    if branches.0.1 != branches.1.1
        && first_continuation_live
        && second_continuation_live
        && !T::cell_has_finalizer(first_cell)
        && !T::cell_has_finalizer(second_cell)
    {
        return FullWalkTwoStepOutcome::Two(
            (T::cell_target(first_cell), branches.0.1),
            (T::cell_target(second_cell), branches.1.1),
        );
    }

    // Dominant two-branch case: neither branch finalizes. Keep the whole byte
    // step as two table loads plus a tiny collapse; only the rare finalizer
    // case below constructs general branch values.
    if !T::cell_has_finalizer(first_cell) && !T::cell_has_finalizer(second_cell) {
        let first = first_continuation_live
            .then_some((T::cell_target(first_cell), branches.0.1));
        let second = second_continuation_live
            .then_some((T::cell_target(second_cell), branches.1.1));
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
    transitions: &mut T,
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
            if full_walk_continuation_viable(
                target,
                parser_node,
                initial_lexer_state,
                transitions,
                parser_cache,
                constraint,
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
fn full_walk_scalar_finalizer(
    target: u32,
    parser_node: u32,
    initial_lexer_state: u32,
    transitions: &mut impl FullWalkTransitionTable,
    parser_cache: &mut FullWalkParserCache,
    constraint: &Constraint,
) -> FullWalkScalarFinalizerOutcome {
    const MULTI: u32 = u32::MAX - 1;
    let code = transitions.finalizer_code(target);
    if code != MULTI {
        let mut next = FullWalkBranches::new();
        let guard = if transitions.terminal_is_ignore(constraint, code) {
            FullWalkPruneGuard::Passed
        } else if transitions.single_finalizer_continues(target) {
            FullWalkPruneGuard::Passed.remember_terminal_match(transitions, target, code)
        } else {
            FullWalkPruneGuard::Passed
        };
        for (reset_lexer, next_parser) in full_walk_terminal_reset_branches(
            code,
            parser_node,
            initial_lexer_state,
            transitions,
            parser_cache,
            constraint,
        ) {
            full_walk_push_unique(
                &mut next,
                FullWalkBranch {
                    lexer_state: reset_lexer,
                    parser_node: next_parser,
                    prune_guard: guard.clone(),
                },
            );
        }
        if full_walk_continuation_viable(
            target,
            parser_node,
            initial_lexer_state,
            transitions,
            parser_cache,
            constraint,
        ) {
            full_walk_push_unique(&mut next, FullWalkBranch {
                lexer_state: target,
                parser_node,
                prune_guard: FullWalkPruneGuard::Passed,
            });
        }
        if next.is_empty() {
            // The terminal finalized but neither parser advancement nor lexical
            // continuation can produce a valid future. Represent this as the
            // exact dead result for the caller.
            return FullWalkScalarFinalizerOutcome::Many(next);
        }
        return full_walk_classify_finalizer_branches(next);
    }

    let mut next = FullWalkBranches::new();
    for terminal in transitions.matched_terminals(target) {
        let guard = if transitions.terminal_is_ignore(constraint, terminal) {
            FullWalkPruneGuard::Passed
        } else {
            FullWalkPruneGuard::Passed.remember_terminal_match(
                transitions, target, terminal,
            )
        };
        for (reset_lexer, next_parser) in full_walk_terminal_reset_branches(
            terminal,
            parser_node,
            initial_lexer_state,
            transitions,
            parser_cache,
            constraint,
        ) {
            full_walk_push_unique(
                &mut next,
                FullWalkBranch {
                    lexer_state: reset_lexer,
                    parser_node: next_parser,
                    prune_guard: guard.clone(),
                },
            );
        }
    }
    if full_walk_continuation_viable(
        target,
        parser_node,
        initial_lexer_state,
        transitions,
        parser_cache,
        constraint,
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
    if next.is_empty() {
        return FullWalkScalarFinalizerOutcome::Many(next);
    }
    full_walk_classify_finalizer_branches(next)
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn full_walk_scalar_finalizer_hot_single(
    target: u32,
    parser_node: u32,
    initial_lexer_state: u32,
    transitions: &mut impl FullWalkTransitionTable,
    parser_cache: &mut FullWalkParserCache,
    constraint: &Constraint,
) -> FullWalkScalarFinalizerOutcome {
    if transitions.uses_scoped_reset_routing() {
        return full_walk_scalar_finalizer(
            target,
            parser_node,
            initial_lexer_state,
            transitions,
            parser_cache,
            constraint,
        );
    }
    const MULTI: u32 = u32::MAX - 1;
    let code = transitions.finalizer_code(target);
    if code == MULTI {
        return full_walk_scalar_finalizer(
            target,
            parser_node,
            initial_lexer_state,
            transitions,
            parser_cache,
            constraint,
        );
    }
    if let Some(next_parser) = parser_cache.advance(constraint, parser_node, code) {
        let reset = FullWalkBranch {
            lexer_state: parser_cache.initial_lexer(constraint, transitions, initial_lexer_state, next_parser),
            parser_node: next_parser,
            prune_guard: if Some(code) == constraint.ignore_terminal {
                FullWalkPruneGuard::Passed
            } else if transitions.single_finalizer_continues(target) {
                FullWalkPruneGuard::Passed.remember_terminal_match(transitions, target, code)
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
fn full_walk_try_apply_plain_single_finalizer(
    target: u32,
    parser_node: u32,
    initial_lexer_state: u32,
    transitions: &mut impl FullWalkTransitionTable,
    parser_cache: &mut FullWalkParserCache,
    constraint: &Constraint,
    two_distinct_marker: u32,
    scalar_lexer: &mut u32,
    scalar_parser: &mut u32,
    current_two: &mut ((u32, u32), (u32, u32)),
) -> bool {
    if transitions.uses_scoped_reset_routing() {
        return false;
    }
    const MULTI: u32 = u32::MAX - 1;
    let code = transitions.finalizer_code(target);
    if code == MULTI
        || (!transitions.terminal_is_ignore(constraint, code)
            && transitions.single_finalizer_continues(target))
    {
        return false;
    }

    let Some(next_parser) = parser_cache.advance(constraint, parser_node, code) else {
        *scalar_lexer = target;
        *scalar_parser = parser_node;
        return true;
    };

    let reset_lexer = parser_cache.initial_lexer(constraint, transitions, initial_lexer_state, next_parser);
    if next_parser != parser_node {
        *scalar_lexer = two_distinct_marker;
        *current_two = (
            (reset_lexer, next_parser),
            (target, parser_node),
        );
        return true;
    }

    // If both exact coordinates are identical there is only one branch. When
    // the parser is the same but lexer coordinates differ, leave the rare case
    // to the existing exact union logic below.
    if reset_lexer == target {
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
    transitions: &mut T,
    parser_cache: &mut FullWalkParserCache,
    constraint: &Constraint,
) -> FullWalkBranches {
    const NONE: u32 = u32::MAX;
    const MULTI: u32 = u32::MAX - 1;
    let mut next = FullWalkBranches::new();
    for branch in branches {
        let Some(advanced_guard) = branch
            .prune_guard
            .advance(transitions, byte)
        else {
            continue;
        };
        let target = transitions.transition(branch.lexer_state, byte);
        if target == u32::MAX {
            continue;
        }
        let code = transitions.finalizer_code(target);
        let continuation_viable = full_walk_continuation_viable(
            target,
            branch.parser_node,
            initial_lexer_state,
            transitions,
            parser_cache,
            constraint,
        );
        if code == MULTI {
            for terminal in transitions.matched_terminals(target) {
                let matched_guard = if transitions.terminal_is_ignore(constraint, terminal) {
                    advanced_guard.clone()
                } else {
                    advanced_guard.remember_terminal_match(transitions, target, terminal)
                };
                for (reset_lexer, parser_node) in full_walk_terminal_reset_branches(
                    terminal,
                    branch.parser_node,
                    initial_lexer_state,
                    transitions,
                    parser_cache,
                    constraint,
                ) {
                    full_walk_push_unique(
                        &mut next,
                        FullWalkBranch {
                            lexer_state: reset_lexer,
                            parser_node,
                            prune_guard: matched_guard.clone(),
                        },
                    );
                }
            }
        } else if code != NONE {
            let matched_guard = if transitions.terminal_is_ignore(constraint, code) {
                advanced_guard.clone()
            } else {
                advanced_guard.remember_terminal_match(transitions, target, code)
            };
            for (reset_lexer, parser_node) in full_walk_terminal_reset_branches(
                code,
                branch.parser_node,
                initial_lexer_state,
                transitions,
                parser_cache,
                constraint,
            ) {
                full_walk_push_unique(
                    &mut next,
                    FullWalkBranch {
                    lexer_state: reset_lexer,
                    parser_node,
                        prune_guard: matched_guard.clone(),
                    },
                );
            }
        }
        if continuation_viable {
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
fn full_walk_projection_union_two<T: FullWalkTransitionTable>(
    transitions: &mut T,
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
    let result = transitions.merge_states(&[key.0, key.1]);
    cache.insert(key, result);
    result
}

#[inline]
fn full_walk_projection_union_three<T: FullWalkTransitionTable>(
    transitions: &mut T,
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
    let result = transitions.merge_states(&states);
    cache.insert(key, result);
    result
}

#[inline]
fn full_walk_merge_two_same_parser<T: FullWalkTransitionTable>(
    transitions: &mut T,
    pair_union_cache: &mut FxHashMap<(u32, u32), Option<u32>>,
    first: (u32, u32),
    second: (u32, u32),
) -> Option<(u32, u32)> {
    if first.1 != second.1 {
        return None;
    }
    full_walk_projection_union_two(transitions, pair_union_cache, first.0, second.0)
        .map(|lexer_state| (lexer_state, first.1))
}

#[inline]
fn full_walk_merge_three_same_parser<T: FullWalkTransitionTable>(
    transitions: &mut T,
    triple_union_cache: &mut FxHashMap<(u32, u32, u32), Option<u32>>,
    lexers: (u32, u32, u32),
    parser_node: u32,
) -> Option<(u32, u32)> {
    full_walk_projection_union_three(
        transitions,
        triple_union_cache,
        lexers.0,
        lexers.1,
        lexers.2,
    )
    .map(|lexer_state| (lexer_state, parser_node))
}

#[inline]
fn full_walk_merge_branches_same_parser<T: FullWalkTransitionTable>(
    transitions: &mut T,
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
        [first, second] => full_walk_projection_union_two(
            transitions,
            pair_union_cache,
            first.lexer_state,
            second.lexer_state,
        ),
        [first, second, third] => full_walk_projection_union_three(
            transitions,
            triple_union_cache,
            first.lexer_state,
            second.lexer_state,
            third.lexer_state,
        ),
        _ => {
            let lexers = branches
                .iter()
                .map(|branch| branch.lexer_state)
                .collect::<SmallVec<[u32; 4]>>();
            transitions.merge_states(&lexers)
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

#[inline]
fn full_walk_many_state_is_empty(state: &FullWalkManyState) -> bool {
    matches!(state, FullWalkManyState::Branches(branches) if branches.is_empty())
}

fn full_walk_many_state_push_branch(
    state: &mut Option<FullWalkManyState>,
    branch: FullWalkBranch,
) {
    match state.take() {
        None => {
            let mut branches = FullWalkBranches::new();
            full_walk_push_unique(&mut branches, branch);
            *state = Some(FullWalkManyState::Branches(branches));
        }
        Some(FullWalkManyState::Branches(mut branches)) => {
            full_walk_push_unique(&mut branches, branch);
            *state = Some(full_walk_many_state_from_branches(branches));
        }
        Some(FullWalkManyState::ThreeSameParser {
            lexers,
            parser_node,
        }) => {
            let mut branches = FullWalkBranches::new();
            for lexer_state in [lexers.0, lexers.1, lexers.2] {
                full_walk_push_unique(
                    &mut branches,
                    FullWalkBranch {
                        lexer_state,
                        parser_node,
                        prune_guard: FullWalkPruneGuard::Passed,
                    },
                );
            }
            full_walk_push_unique(&mut branches, branch);
            *state = Some(full_walk_many_state_from_branches(branches));
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn full_walk_step_many_state<T: FullWalkTransitionTable>(
    state: &FullWalkManyState,
    byte: u8,
    initial_lexer_state: u32,
    transitions: &mut T,
    parser_cache: &mut FullWalkParserCache,
    constraint: &Constraint,
) -> FullWalkManyState {
    match state {
        FullWalkManyState::Branches(branches) => full_walk_many_state_from_branches(
            full_walk_step_many(
                branches,
                byte,
                initial_lexer_state,
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
                if next.0 != next.1 && next.0 != next.2 && next.1 != next.2 {
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
                transitions,
                parser_cache,
                constraint,
            ))
        }
    }
}



/// The exact accelerators are default-on. This diagnostic retains the
/// reference executor without rebuilding or changing the compiled language.
#[inline]
fn full_walk_acceleration_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| !std::env::var("GLRMASK_DISABLE_FULL_WALK_ACCELERATION")
        .ok().is_some_and(|value| matches!(value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on")))
}

/// Scheduling choice only: both executors retain the complete
/// lexer/parser/guard correlation. The shared walk advances pending guards
/// through each candidate byte; the factored walk subtracts the same rejected
/// words afterwards. Keep the established single-root and large-frontier paths.
fn joint_initial_guard_walk_enabled(
    roots: &DynamicBranches,
    initial_tokenizer_state: u32,
) -> bool {
    // Scope fixtures deliberately exercise the factored-root path even when
    // a full-suite environment enables joint execution globally. Keep this
    // override thread-local; parallel tests must not mutate process settings.
    #[cfg(test)]
    if TEST_ROOT_OUTPUT_SCOPE_FORCE.with(|enabled| enabled.get()) { return false; }
    (2..=8).contains(&roots.len())
        && full_walk_acceleration_enabled()
        && joint_initial_guard_shape_profitable(
            roots.len(),
            roots.iter().any(|root| root.parser_filtered_transparent),
            roots.iter().any(|root| {
                !root.initial_prune_guard.is_passed()
                    && root.exact_tokenizer_state == Some(initial_tokenizer_state)
            }),
            roots.iter().any(|root| {
                root.initial_prune_guard.is_passed()
                    && root.exact_tokenizer_state.is_some_and(|state| {
                        state != initial_tokenizer_state
                    })
            }),
        )
}

#[inline]
fn joint_initial_guard_shape_profitable(
    root_count: usize,
    has_transparent_root: bool,
    has_guarded_initial_root: bool,
    has_exact_continuation_root: bool,
) -> bool {
    // A root narrowed by exact parser admission has a dedicated transparent
    // execution path. Factoring its token-start exclusion preserves that
    // shortcut; merging it into a guarded generic frontier can turn a cheap
    // filtered walk into a full-vocabulary traversal. This is a scheduling
    // choice only: both routes preserve each root's correlated exclusions.
    // A guarded initial root paired with a continuation root is similarly
    // asymmetric: the joint executor keeps the pair correlated through the
    // complete vocabulary and disables the continuation's master-trie proof.
    // Factoring retains that proof and subtracts the initial root's immutable
    // blocked-token set afterwards.
    // Exact continuation roots can use the same master-trie proof even when
    // the guarded alternative is itself a continuation. Keep their singleton
    // walks separate so the pending guard cannot disable that proof.
    (2..=8).contains(&root_count)
        && !has_transparent_root
        && !has_guarded_initial_root
        && !has_exact_continuation_root
}

#[cfg(test)]
mod joint_root_scheduling_tests {
    use super::joint_initial_guard_shape_profitable;

    #[test]
    fn factors_specialized_or_guarded_initial_root_shapes() {
        for count in 0..=16 {
            assert_eq!(
                joint_initial_guard_shape_profitable(count, false, false, false),
                (2..=8).contains(&count),
            );
            assert!(!joint_initial_guard_shape_profitable(count, true, false, false));
            assert!(!joint_initial_guard_shape_profitable(count, false, true, false));
            assert!(!joint_initial_guard_shape_profitable(count, false, false, true));
        }
    }
}

/// Compute the complete dynamic mask, preserving correlated parser, lexer and
/// maximal-munch state. Bounded mask-local memoization only reuses exact states;
/// subtree shortcuts require either a complete-state identity proof or an
/// explicit output responsibility that the enclosing union will enforce.
/// Every declined proof or exhausted cache falls back to ordinary traversal.
/// Dense and lazy lexer providers share the same authoritative byte executor.
fn try_full_walk_mask(
    state: &ConstraintState<'_>,
    vocab: &DynamicMaskVocab,
    trie: &DynamicMaskTrie,
    root_branches: &DynamicBranches,
    lexer_scan_cache: &mut DynamicNfaScanCache<'_>,
    buf: &mut [u32],
) -> Result<bool, String> {
    try_full_walk_mask_in_output_scope(
        state, vocab, trie, root_branches, lexer_scan_cache, buf, None,
    )
}

#[cfg(test)]
std::thread_local! {
    static TEST_ROOT_OUTPUT_SCOPE_FORCE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn root_output_scope_enabled() -> bool {
    #[cfg(test)]
    if TEST_ROOT_OUTPUT_SCOPE_FORCE.with(|enabled| enabled.get()) { return true; }
    full_walk_acceleration_enabled()
}

fn try_full_walk_mask_in_output_scope(
    state: &ConstraintState<'_>,
    vocab: &DynamicMaskVocab,
    trie: &DynamicMaskTrie,
    root_branches: &DynamicBranches,
    lexer_scan_cache: &mut DynamicNfaScanCache<'_>,
    buf: &mut [u32],
    ignored_output: Option<&[u32]>,
) -> Result<bool, String> {
    debug_assert!(ignored_output.is_none_or(|mask| mask.len() == buf.len()));
    if root_branches.is_empty() {
        buf.fill(0);
        update_special_token_mask(state, buf);
        state.clear_late_grammar_placeholder_mask(buf);
        #[cfg(test)]
        TEST_FULL_WALK_USES.with(|count| count.set(count.get() + 1));
        return Ok(true);
    }

    // Token-start maximal-munch guards filter complete candidate model tokens;
    // they are independent of which exact lexer executor represents a root.
    // Factor them before choosing Flat16/Flat32/scalar-dispatch/config walking
    // so one guarded alternative cannot disable slicing for every other root.
    // Each root language is evaluated with the guard Passed, its immutable
    // blocked-token set is removed, and the alternative root languages are
    // unioned. This is the same exact algebra previously used only by the
    // scalar-dispatch path, lifted to the common executor boundary.
    if root_branches
        .iter()
        .any(|branch| !branch.initial_prune_guard.is_passed())
        && !joint_initial_guard_walk_enabled(
            root_branches,
            state.constraint.tokenizer.initial_state(),
        )
    {
        let guarded_root_diagnostic = std::env::var("GLRMASK_DIAG_GUARDED_ROOT_GENERATION")
            .ok().and_then(|v| v.parse::<u64>().ok()) == Some(state.generation);
        if guarded_root_diagnostic {
            for (index, branch) in root_branches.iter().enumerate() {
                let memories = match &branch.initial_prune_guard {
                    InitialPruneGuard::Passed => &[][..],
                    InitialPruneGuard::Pending { memories } => memories.as_ref(),
                };
                eprintln!("[glrmask/profile][guarded_root] generation={} index={} config={} exact={:?} gss={} filtered={} transparent={} memories={:?}",
                    state.generation, index, branch.tokenizer_config,
                    branch.exact_tokenizer_state, branch.gss.ptr_key(),
                    branch.parser_filtered_root, branch.parser_filtered_transparent, memories);
            }
        }
        let mut merged = vec![0u32; buf.len()];
        let mut scratch = vec![0u32; buf.len()];
        let scope_enabled = root_output_scope_enabled();
        let mut scope_mask = if scope_enabled { vec![0u32; buf.len()] } else { Vec::new() };
        for (root_index, branch) in root_branches.iter().enumerate() {
            scratch.fill(0);
            let blocked = branch
                .initial_prune_guard
                .blocked_output_mask(state.constraint, buf.len())?;
            let mut one = DynamicBranches::new();
            let mut unguarded = branch.clone();
            unguarded.initial_prune_guard = InitialPruneGuard::Passed;
            one.push(unguarded);
            let scoped_output = if scope_enabled && !branch.parser_filtered_transparent {
                let mut ignored_bits = 0usize;
                for (index, word) in scope_mask.iter_mut().enumerate() {
                    *word = merged[index]
                        | blocked.as_ref().and_then(|mask| mask.get(index)).copied().unwrap_or(0)
                        | ignored_output.and_then(|mask| mask.get(index)).copied().unwrap_or(0);
                    ignored_bits += word.count_ones() as usize;
                }
                // Skip the index setup when little output is redundant. Tests
                // exercise the same exact algorithm on tiny vocabularies.
                let minimum = if cfg!(test) { 0 } else { buf.len().saturating_mul(8) };
                (ignored_bits >= minimum).then_some(scope_mask.as_slice())
            } else {
                ignored_output
            };
            // Only this union owns a partial-output contract. Bits already in
            // `merged` cannot alter its union; blocked bits are removed below.
            // All other callers keep the ordinary complete-mask interface.
            if !try_full_walk_mask_in_output_scope(
                state,
                vocab,
                trie,
                &one,
                lexer_scan_cache,
                &mut scratch,
                scoped_output,
            )? {
                return Ok(false);
            }
            if let Some(blocked) = blocked {
                for (word, &blocked) in scratch.iter_mut().zip(blocked.iter()) {
                    *word &= !blocked;
                }
            }
            if guarded_root_diagnostic {
                let allowed: u32 = scratch.iter().map(|word| word.count_ones()).sum();
                let previous: u32 = merged.iter().map(|word| word.count_ones()).sum();
                let new: u32 = scratch.iter().zip(merged.iter())
                    .map(|(&incoming, &old)| (incoming & !old).count_ones()).sum();
                eprintln!("[glrmask/profile][guarded_root_output] generation={} index={} allowed={} previous={} new={}",
                    state.generation, root_index, allowed, previous, new);
            }
            for (dst, &word) in merged.iter_mut().zip(&scratch) {
                *dst |= word;
            }
        }
        buf.copy_from_slice(&merged);
        return Ok(true);
    }

    if std::env::var_os("GLRMASK_DISABLE_SCALAR_DISPATCH_FULL_WALK").is_none()
        && !lexer_scan_cache.deterministic
        && lexer_scan_cache.tokenizer().has_scalar_deterministic_dispatch()
        && trie.full_walk_max_parent_depth() < 255
    {
        let (transitions16, transitions32, finalizer_code, single_finalizer_continues) =
            match vocab.mask_projection_fast_transitions() {
                Some(FastTokenizerTransitions::Flat16 {
                    transitions,
                    finalizer_code,
                    single_finalizer_continues,
                }) => (
                    Some(transitions.as_ref()),
                    None,
                    Some(finalizer_code.as_ref()),
                    Some(single_finalizer_continues.as_ref()),
                ),
                Some(FastTokenizerTransitions::Flat32 {
                    transitions,
                    finalizer_code,
                    single_finalizer_continues,
                }) => (
                    None,
                    Some(transitions.as_ref()),
                    Some(finalizer_code.as_ref()),
                    Some(single_finalizer_continues.as_ref()),
                ),
                _ => (None, None, None, None),
            };
        let used = full_walk_dense::try_scalar_dispatch(
            state,
            vocab,
            trie,
            root_branches,
            lexer_scan_cache,
            buf,
            transitions16,
            transitions32,
            finalizer_code,
            single_finalizer_continues,
            ignored_output,
        )?;
        if used {
            #[cfg(test)]
            TEST_FULL_WALK_USES.with(|count| count.set(count.get() + 1));
            return Ok(true);
        }
    }
    let tokenizer = lexer_scan_cache.tokenizer();
    match vocab.mask_projection_fast_transitions() {
        Some(FastTokenizerTransitions::Flat16 {
            transitions,
            finalizer_code,
            single_finalizer_continues,
        }) if lexer_scan_cache.deterministic => {
            if trie.full_walk_max_parent_depth() < 255 {
                let result = if root_branches.len() == 1 {
                    full_walk_dense::try_flat16::<true>(
                        state,
                        vocab,
                        trie,
                        root_branches,
                        lexer_scan_cache,
                        buf,
                        transitions.as_ref(),
                        finalizer_code.as_ref(),
                        single_finalizer_continues.as_ref(),
                    )
                } else {
                    full_walk_dense::try_flat16::<false>(
                        state,
                        vocab,
                        trie,
                        root_branches,
                        lexer_scan_cache,
                        buf,
                        transitions.as_ref(),
                        finalizer_code.as_ref(),
                        single_finalizer_continues.as_ref(),
                    )
                };
                #[cfg(test)]
                if matches!(result, Ok(true)) {
                    TEST_FULL_WALK_USES.with(|count| count.set(count.get() + 1));
                }
                return result;
            }
            let mut table = FullWalkFlat16 {
                transitions: transitions.as_ref(),
                finalizer_code: finalizer_code.as_ref(),
                single_finalizer_continues: single_finalizer_continues.as_ref(),
                tokenizer,
                vocab,
            };
            if root_branches.len() == 1 {
                try_full_walk_mask_with_table_dispatch::<_, true>(
                    state,
                    vocab,
                    trie,
                    root_branches,
                    buf,
                    &mut table,
                )
            } else {
                try_full_walk_mask_with_table_dispatch::<_, false>(
                    state,
                    vocab,
                    trie,
                    root_branches,
                    buf,
                    &mut table,
                )
            }
        }
        Some(FastTokenizerTransitions::Flat32 {
            transitions,
            finalizer_code,
            single_finalizer_continues,
        }) if lexer_scan_cache.deterministic => {
            if trie.full_walk_max_parent_depth() < 255 {
                let result = if root_branches.len() == 1 {
                    full_walk_dense::try_flat32::<true>(
                        state,
                        vocab,
                        trie,
                        root_branches,
                        lexer_scan_cache,
                        buf,
                        transitions.as_ref(),
                        finalizer_code.as_ref(),
                        single_finalizer_continues.as_ref(),
                    )
                } else {
                    full_walk_dense::try_flat32::<false>(
                        state,
                        vocab,
                        trie,
                        root_branches,
                        lexer_scan_cache,
                        buf,
                        transitions.as_ref(),
                        finalizer_code.as_ref(),
                        single_finalizer_continues.as_ref(),
                    )
                };
                #[cfg(test)]
                if matches!(result, Ok(true)) {
                    TEST_FULL_WALK_USES.with(|count| count.set(count.get() + 1));
                }
                return result;
            }
            let mut table = FullWalkFlat32 {
                transitions: transitions.as_ref(),
                finalizer_code: finalizer_code.as_ref(),
                single_finalizer_continues: single_finalizer_continues.as_ref(),
                tokenizer,
                vocab,
            };
            if root_branches.len() == 1 {
                try_full_walk_mask_with_table_dispatch::<_, true>(
                    state,
                    vocab,
                    trie,
                    root_branches,
                    buf,
                    &mut table,
                )
            } else {
                try_full_walk_mask_with_table_dispatch::<_, false>(
                    state,
                    vocab,
                    trie,
                    root_branches,
                    buf,
                    &mut table,
                )
            }
        }
        _ => {
            let mut table = FullWalkConfigTransitions::new(
                lexer_scan_cache, vocab.max_token_byte_len(), state.generation,
            );
            let result = if root_branches.len() == 1 {
                try_full_walk_mask_with_table_dispatch::<_, true>(
                    state,
                    vocab,
                    trie,
                    root_branches,
                    buf,
                    &mut table,
                )
            } else {
                try_full_walk_mask_with_table_dispatch::<_, false>(
                    state,
                    vocab,
                    trie,
                    root_branches,
                    buf,
                    &mut table,
                )
            };
            match result {
                Ok(used) => {
                    table.finish()?;
                    #[cfg(test)]
                    if used {
                        TEST_CONFIG_FULL_WALK_USES.with(|count| count.set(count.get() + 1));
                    }
                    Ok(used)
                }
                Err(error) => Err(error),
            }
        }
    }
}


/// Prove that the LEFT operand of a retained top-level exclusion remains a
/// live, non-finalizing prefix for every word in `slice+`, starting from the
/// exact left residual recorded for `source`. The exclusion certificate also
/// proves the RIGHT operand finite. Together these facts imply every such
/// prefix is live in `left \ right`: after any slice prefix, append enough
/// further slice atoms to exceed the finite RIGHT bound, then complete LEFT.
///
/// This proof is deliberately independent of the combined lexer product: its
/// state space is only `(slice_state, left_state)` and therefore stays tiny for
/// the JSON safe-content monitor.
fn exclusion_left_slice_prefix_live(
    tokenizer: &Tokenizer,
    source: u32,
    terminal: TerminalID,
    slice: &crate::compiler::stages::id_map_and_terminal_dwa::classify::VocabPartitionDfa,
    work_limit: usize,
) -> Option<bool> {
    let certificate = tokenizer.terminal_exclusion_continuation(source, terminal)?;
    // A retained certificate is constructed only for a finite RIGHT operand.
    // At the current state RIGHT may already be dead; either case is fine.
    let _finite_right_bound = certificate.right_max_remaining;
    let left = tokenizer.terminal_exclusion_left_dfa(terminal)?;
    let left_live = |state: u32| {
        left.finalizers(state).is_empty()
            && left.possible_future_group_ids(state).contains(0)
    };
    if !left_live(certificate.left_state) {
        return Some(false);
    }

    let mut seen = FxHashSet::<(u32, u32)>::default();
    let mut queue = std::collections::VecDeque::from([(
        slice.start_state(),
        certificate.left_state,
    )]);
    let mut work = 0usize;
    while let Some((slice_state, left_state)) = queue.pop_front() {
        if !seen.insert((slice_state, left_state)) {
            continue;
        }
        for byte_value in 0u16..=255 {
            let byte = byte_value as u8;
            let slice_target = slice.step(slice_state, byte);
            if !slice.can_reach_accepting(slice_target) {
                continue;
            }
            work += 1;
            if work > work_limit {
                return None;
            }
            let Some(left_target) = left.step(left_state, byte) else {
                return Some(false);
            };
            if !left_live(left_target) {
                return Some(false);
            }
            if !seen.contains(&(slice_target, left_target)) {
                queue.push_back((slice_target, left_target));
            }
        }
    }
    Some(true)
}

fn direct_slice_prefix_contained_config<T: FullWalkTransitionTable>(
    transitions: &mut T,
    start: u32,
    terminal: TerminalID,
    slice: &crate::compiler::stages::id_map_and_terminal_dwa::classify::VocabPartitionDfa,
    work_limit: usize,
) -> Option<bool> {
    let mut seen = rustc_hash::FxHashSet::<(u32, u32)>::default();
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
            work += 1;
            if work > work_limit {
                return None;
            }
            let cell = transitions.cell(lexer_state, byte);
            if T::cell_is_dead(cell) {
                return Some(false);
            }
            let target = T::cell_target(cell);
            let terminal_live = transitions.future_contains(target, terminal)
                || transitions.matched_terminals(target).contains(&terminal);
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


const HOT_EDGE_LEXER_TWO_DISTINCT: u32 = u32::MAX - 3;
const HOT_EDGE_LEXER_TWO: u32 = u32::MAX - 2;
const HOT_EDGE_LEXER_DEAD: u32 = u32::MAX;

enum FullWalkHotLaneOutcome {
    Dead,
    /// Scalar target reached without crossing a terminal finalizer. Safe for
    /// the same parser-conditioned liveness test as the generic walker.
    ScalarPlain(u32, u32),
    Scalar(u32, u32),
    Two((u32, u32), (u32, u32)),
    Decline,
}

#[allow(clippy::too_many_arguments)]
#[cold]
#[inline(never)]
fn full_walk_hot_lane_scalar_escape<T: FullWalkTransitionTable>(
    source_lexer: u32,
    byte: u8,
    parser_node: u32,
    initial_lexer_state: u32,
    transitions: &mut T,
    parser_cache: &mut FullWalkParserCache,
    constraint: &Constraint,
    pair_union_cache: &mut FxHashMap<(u32, u32), Option<u32>>,
) -> FullWalkHotLaneOutcome {
    let cell = transitions.cell(source_lexer, byte);
    if T::cell_is_dead(cell) {
        return FullWalkHotLaneOutcome::Dead;
    }
    let target = T::cell_target(cell);
    if !T::cell_has_finalizer(cell) {
        return FullWalkHotLaneOutcome::ScalarPlain(target, parser_node);
    }

    let mut lexer = target;
    let mut parser = parser_node;
    let mut two = ((0u32, 0u32), (0u32, 0u32));
    if full_walk_try_apply_plain_single_finalizer(
        target,
        parser_node,
        initial_lexer_state,
        transitions,
        parser_cache,
        constraint,
        HOT_EDGE_LEXER_TWO_DISTINCT,
        &mut lexer,
        &mut parser,
        &mut two,
    ) {
        return if lexer == HOT_EDGE_LEXER_TWO_DISTINCT {
            FullWalkHotLaneOutcome::Two(two.0, two.1)
        } else {
            FullWalkHotLaneOutcome::Scalar(lexer, parser)
        };
    }

    match full_walk_scalar_finalizer_hot_single(
        target,
        parser_node,
        initial_lexer_state,
        transitions,
        parser_cache,
        constraint,
    ) {
        FullWalkScalarFinalizerOutcome::Scalar(branch) if branch.prune_guard.is_passed() => {
            FullWalkHotLaneOutcome::Scalar(branch.lexer_state, branch.parser_node)
        }
        FullWalkScalarFinalizerOutcome::Two(first, second)
            if first.prune_guard.is_passed() && second.prune_guard.is_passed() =>
        {
            if let Some((lexer_state, parser_node)) = full_walk_merge_two_same_parser(
                transitions,
                pair_union_cache,
                (first.lexer_state, first.parser_node),
                (second.lexer_state, second.parser_node),
            ) {
                FullWalkHotLaneOutcome::Scalar(lexer_state, parser_node)
            } else {
                FullWalkHotLaneOutcome::Two(
                    (first.lexer_state, first.parser_node),
                    (second.lexer_state, second.parser_node),
                )
            }
        }
        FullWalkScalarFinalizerOutcome::Many(mut branches)
            if branches.len() == 2
                && branches.iter().all(|branch| branch.prune_guard.is_passed()) =>
        {
            let second = branches.pop().expect("second hot-lane branch disappeared");
            let first = branches.pop().expect("first hot-lane branch disappeared");
            if let Some((lexer_state, parser_node)) = full_walk_merge_two_same_parser(
                transitions,
                pair_union_cache,
                (first.lexer_state, first.parser_node),
                (second.lexer_state, second.parser_node),
            ) {
                FullWalkHotLaneOutcome::Scalar(lexer_state, parser_node)
            } else {
                FullWalkHotLaneOutcome::Two(
                    (first.lexer_state, first.parser_node),
                    (second.lexer_state, second.parser_node),
                )
            }
        }
        _ => FullWalkHotLaneOutcome::Decline,
    }
}

#[allow(clippy::too_many_arguments)]
#[cold]
#[inline(never)]
fn full_walk_hot_lane_two_escape<T: FullWalkTransitionTable>(
    branches: ((u32, u32), (u32, u32)),
    byte: u8,
    initial_lexer_state: u32,
    transitions: &mut T,
    parser_cache: &mut FullWalkParserCache,
    constraint: &Constraint,
    pair_union_cache: &mut FxHashMap<(u32, u32), Option<u32>>,
) -> FullWalkHotLaneOutcome {
    match full_walk_step_two(
        branches,
        byte,
        initial_lexer_state,
        transitions,
        parser_cache,
        constraint,
    ) {
        FullWalkTwoStepOutcome::Dead => FullWalkHotLaneOutcome::Dead,
        FullWalkTwoStepOutcome::One((lexer_state, parser_node)) => {
            FullWalkHotLaneOutcome::Scalar(lexer_state, parser_node)
        }
        FullWalkTwoStepOutcome::Two(first, second) => {
            if let Some((lexer_state, parser_node)) = full_walk_merge_two_same_parser(
                transitions,
                pair_union_cache,
                first,
                second,
            ) {
                FullWalkHotLaneOutcome::Scalar(lexer_state, parser_node)
            } else {
                FullWalkHotLaneOutcome::Two(first, second)
            }
        }
        FullWalkTwoStepOutcome::Many(_) => FullWalkHotLaneOutcome::Decline,
    }
}

/// Exact narrow lane for the overwhelmingly scalar config full-walk case.
/// It keeps the existing compact sequential FullWalkOp stream, but compiles a
/// much smaller scalar/two-only interpreter. Unsupported branch shapes decline
/// and the caller recomputes through the general walker.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn try_full_walk_hot_scalar_edges<T: FullWalkTransitionTable>(
    state: &ConstraintState<'_>,
    vocab: &DynamicMaskVocab,
    trie: &DynamicMaskTrie,
    root_lexer: u32,
    root_parser: u32,
    initial_lexer_state: u32,
    parser_cache: &mut FullWalkParserCache,
    transitions: &mut T,
    buf: &mut [u32],
) -> Result<Option<bool>, String> {
    if !trie.full_walk_all_consume() {
        return Ok(None);
    }
    let stack_len = usize::from(trie.full_walk_max_parent_depth()).saturating_add(2);
    let mut inline_lexer = [HOT_EDGE_LEXER_DEAD; 256];
    let mut heap_lexer = Vec::<u32>::new();
    let stack_lexer: &mut [u32] = if stack_len <= inline_lexer.len() {
        &mut inline_lexer[..stack_len]
    } else {
        heap_lexer.resize(stack_len, HOT_EDGE_LEXER_DEAD);
        heap_lexer.as_mut_slice()
    };
    let mut inline_parser = [0u32; 256];
    let mut heap_parser = Vec::<u32>::new();
    let stack_parser: &mut [u32] = if stack_len <= inline_parser.len() {
        &mut inline_parser[..stack_len]
    } else {
        heap_parser.resize(stack_len, 0);
        heap_parser.as_mut_slice()
    };
    let mut inline_two = [((0u32, 0u32), (0u32, 0u32)); 256];
    let mut heap_two = Vec::<((u32, u32), (u32, u32))>::new();
    let stack_two: &mut [((u32, u32), (u32, u32))] = if stack_len <= inline_two.len() {
        &mut inline_two[..stack_len]
    } else {
        heap_two.resize(stack_len, ((0, 0), (0, 0)));
        heap_two.as_mut_slice()
    };
    stack_lexer[0] = root_lexer;
    stack_parser[0] = root_parser;

    let mut pair_union_cache = FxHashMap::<(u32, u32), Option<u32>>::default();
    let walk_ops = trie.full_walk_ops();
    let token_markers = vocab.full_walk_token_markers_for(trie);
    let mut token_marker_index = 0usize;
    let mut remaining_ops = walk_ops.iter();
    let mut lexer: u32;
    let mut parser = root_parser;
    let mut two = ((0u32, 0u32), (0u32, 0u32));

    // Endpoint checks are deterministic for one mask and DFS order has very
    // strong run locality in `(parser_node, lexer_state)`. Keep the two most
    // recent exact scalar pairs locally so the common case avoids the nested
    // parser-cache lookup entirely. Two-branch endpoints deliberately bypass
    // this memo and retain the generic exact checks below.
    let mut last_boundary_key = u64::MAX;
    let mut last_boundary_allowed = false;
    let mut second_boundary_key = u64::MAX;
    let mut second_boundary_allowed = false;
    let mut condition_consecutive_non_pruning = 0u32;
    let condition_budget = config_scalar_conditioned_budget();
    let condition_dead_skip = std::env::var_os(
        "GLRMASK_EXPERIMENT_CONFIG_SCALAR_CONDITIONED_DEAD_SKIP",
    )
    .is_some() || condition_budget.is_some();
    let condition_walk_enabled = std::env::var_os(
        "GLRMASK_EXPERIMENT_CONFIG_SCALAR_CONDITIONED",
    )
    .is_some() || condition_dead_skip;
    let condition_initial_only = std::env::var_os(
        "GLRMASK_EXPERIMENT_CONFIG_SCALAR_CONDITIONED_INITIAL_ONLY",
    )
    .is_some();
    let condition_walk_enabled = condition_walk_enabled
        && (!condition_initial_only || state.generation == 0);

    // `full_walk_all_consume()` proves there are no synthetic empty-edge
    // records. Each outer iteration therefore begins at a radix-edge START,
    // and the inner loop can consume consecutive byte ops until END without
    // retesting START/CONSUME on every byte.
    'edge_walk: while let Some(&first_op) = remaining_ops.next() {
        let parent_depth = first_op.parent_depth() as usize;
        lexer = unsafe { *stack_lexer.get_unchecked(parent_depth) };
        if lexer < HOT_EDGE_LEXER_TWO_DISTINCT {
            parser = unsafe { *stack_parser.get_unchecked(parent_depth) };
        } else if lexer == HOT_EDGE_LEXER_TWO_DISTINCT || lexer == HOT_EDGE_LEXER_TWO {
            two = unsafe { *stack_two.get_unchecked(parent_depth) };
        } else {
            return Ok(None);
        }
        let mut op = first_op;
        loop {
        {
            let byte = op.byte();
            if lexer == HOT_EDGE_LEXER_DEAD {
                full_walk_skip_dead_subtree_generic(
                    vocab,
                    trie,
                    walk_ops,
                    &mut remaining_ops,
                    &mut token_marker_index,
                    buf,
                );
                continue 'edge_walk;
            }

            let mut plain_scalar_step = false;
            let outcome = if lexer < HOT_EDGE_LEXER_TWO_DISTINCT {
                if let Some(hot_id) = transitions.hot_state_id(lexer) {
                    let next = transitions.hot_step(hot_id, byte);
                    if next < FULL_WALK_HOT_SLOW {
                        lexer = transitions.hot_state_from_id(next);
                        plain_scalar_step = true;
                        None
                    } else if next == FULL_WALK_HOT_DEAD {
                        Some(FullWalkHotLaneOutcome::Dead)
                    } else {
                        Some(full_walk_hot_lane_scalar_escape(
                            lexer,
                            byte,
                            parser,
                            initial_lexer_state,
                            transitions,
                            parser_cache,
                            state.constraint,
                            &mut pair_union_cache,
                        ))
                    }
                } else {
                    Some(full_walk_hot_lane_scalar_escape(
                        lexer,
                        byte,
                        parser,
                        initial_lexer_state,
                        transitions,
                        parser_cache,
                        state.constraint,
                        &mut pair_union_cache,
                    ))
                }
            } else if lexer == HOT_EDGE_LEXER_TWO_DISTINCT || lexer == HOT_EDGE_LEXER_TWO {
                Some(full_walk_hot_lane_two_escape(
                    two,
                    byte,
                    initial_lexer_state,
                    transitions,
                    parser_cache,
                    state.constraint,
                    &mut pair_union_cache,
                ))
            } else {
                return Ok(None);
            };

            if let Some(outcome) = outcome {
                match outcome {
                    FullWalkHotLaneOutcome::Dead => {
                        full_walk_skip_dead_subtree_generic(
                            vocab,
                            trie,
                            walk_ops,
                            &mut remaining_ops,
                            &mut token_marker_index,
                            buf,
                        );
                        continue 'edge_walk;
                    }
                    FullWalkHotLaneOutcome::ScalarPlain(next_lexer, next_parser) => {
                        lexer = next_lexer;
                        parser = next_parser;
                        plain_scalar_step = true;
                    }
                    FullWalkHotLaneOutcome::Scalar(next_lexer, next_parser) => {
                        lexer = next_lexer;
                        parser = next_parser;
                    }
                    FullWalkHotLaneOutcome::Two(first, second) => {
                        lexer = if first.1 != second.1 {
                            HOT_EDGE_LEXER_TWO_DISTINCT
                        } else {
                            HOT_EDGE_LEXER_TWO
                        };
                        two = (first, second);
                    }
                    FullWalkHotLaneOutcome::Decline => return Ok(None),
                }
            }

            if plain_scalar_step && condition_walk_enabled {
                let condition_active = match condition_budget {
                    Some(max) => condition_consecutive_non_pruning < max,
                    None => true,
                };
                if condition_active {
                    if transitions.token_boundary_allowed(
                        parser_cache,
                        state.constraint,
                        initial_lexer_state,
                        lexer,
                        parser,
                    ) {
                        condition_consecutive_non_pruning =
                            condition_consecutive_non_pruning.saturating_add(1);
                    } else {
                        lexer = HOT_EDGE_LEXER_DEAD;
                        if condition_dead_skip {
                            let pruned = full_walk_skip_dead_subtree_generic(
                                vocab,
                                trie,
                                walk_ops,
                                &mut remaining_ops,
                                &mut token_marker_index,
                                buf,
                            );
                            if pruned > 0 {
                                condition_consecutive_non_pruning = 0;
                            } else {
                                condition_consecutive_non_pruning =
                                    condition_consecutive_non_pruning.saturating_add(1);
                            }
                            continue 'edge_walk;
                        }
                    }
                }
            }
        }

        if op.ends_edge() {
            if op.child_is_token() {
                let token_marker = unsafe { *token_markers.get_unchecked(token_marker_index) };
                token_marker_index += 1;
                let allowed = if lexer == HOT_EDGE_LEXER_DEAD {
                    false
                } else if lexer < HOT_EDGE_LEXER_TWO_DISTINCT {
                    let boundary_key = (u64::from(parser) << 32) | u64::from(lexer);
                    if boundary_key == last_boundary_key {
                        last_boundary_allowed
                    } else if boundary_key == second_boundary_key {
                        std::mem::swap(&mut last_boundary_key, &mut second_boundary_key);
                        std::mem::swap(&mut last_boundary_allowed, &mut second_boundary_allowed);
                        last_boundary_allowed
                    } else {
                        let value = transitions.token_boundary_allowed(
                            parser_cache,
                            state.constraint,
                            initial_lexer_state,
                            lexer,
                            parser,
                        );
                        second_boundary_key = last_boundary_key;
                        second_boundary_allowed = last_boundary_allowed;
                        last_boundary_key = boundary_key;
                        last_boundary_allowed = value;
                        value
                    }
                } else if lexer == HOT_EDGE_LEXER_TWO_DISTINCT || lexer == HOT_EDGE_LEXER_TWO {
                    transitions.token_boundary_allowed(
                        parser_cache,
                        state.constraint,
                        initial_lexer_state,
                        two.0.0,
                        two.0.1,
                    ) || transitions.token_boundary_allowed(
                        parser_cache,
                        state.constraint,
                        initial_lexer_state,
                        two.1.0,
                        two.1.1,
                    )
                } else {
                    return Ok(None);
                };
                if !allowed {
                    clear_dynamic_token_marker(vocab, token_marker, buf);
                }
            }
            unsafe {
                *stack_lexer.get_unchecked_mut(parent_depth + 1) = lexer;
                if lexer < HOT_EDGE_LEXER_TWO_DISTINCT {
                    *stack_parser.get_unchecked_mut(parent_depth + 1) = parser;
                } else if lexer == HOT_EDGE_LEXER_TWO_DISTINCT || lexer == HOT_EDGE_LEXER_TWO {
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
        op = next_op;
        }
    }

    Ok(Some(true))
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn try_full_walk_mask_with_table_dispatch<
    T: FullWalkTransitionTable,
    const HOT_SINGLE_ROOT: bool,
>(
    state: &ConstraintState<'_>,
    vocab: &DynamicMaskVocab,
    trie: &DynamicMaskTrie,
    root_branches: &DynamicBranches,
    buf: &mut [u32],
    transitions: &mut T,
) -> Result<bool, String> {
    if direct_residual_defer_finalizer_experiment_enabled() {
        try_full_walk_mask_with_table::<T, HOT_SINGLE_ROOT, true>(
            state,
            vocab,
            trie,
            root_branches,
            buf,
            transitions,
        )
    } else {
        try_full_walk_mask_with_table::<T, HOT_SINGLE_ROOT, false>(
            state,
            vocab,
            trie,
            root_branches,
            buf,
            transitions,
        )
    }
}

fn try_full_walk_mask_with_table<
    T: FullWalkTransitionTable,
    const HOT_SINGLE_ROOT: bool,
    const DEFER_FIRST_MATCH_FINALIZER: bool,
>(
    state: &ConstraintState<'_>,
    vocab: &DynamicMaskVocab,
    trie: &DynamicMaskTrie,
    root_branches: &DynamicBranches,
    buf: &mut [u32],
    transitions: &mut T,
) -> Result<bool, String> {

    let initial_lexer_state = transitions.walk_initial_state(state.constraint, vocab)?;

    let all_words = vocab.all_original_token_words();
    let copy_len = buf.len().min(all_words.len());
    buf[..copy_len].copy_from_slice(&all_words[..copy_len]);
    if copy_len < buf.len() { buf[copy_len..].fill(0); }

    let (mut parser_cache, root_parser_nodes) = FullWalkParserCache::from_roots(
        root_branches,
        transitions.dense_state_count(),
    );
    let first_match_direct_root = if HOT_SINGLE_ROOT
        && root_branches.len() == 1
        && root_branches[0].initial_prune_guard.is_passed()
        && vocab.mask_runtime_tokenizer().is_none()
        && let Some(source) = root_branches[0].exact_tokenizer_state
        && vocab.mask_runtime_state(source) == source
        && transitions.exact_raw_state(root_branches[0].tokenizer_config) == Some(source)
        && !state.constraint.tokenizer.state_has_epsilon_transitions(source)
        && state.constraint.tokenizer.matched_terminals_slice(source).is_empty()
        && let Some(terminal) = state
            .constraint
            .tokenizer
            .virtual_residual_terminal_for_state(source)
        && Some(terminal) != state.constraint.ignore_terminal
        && state
            .constraint
            .tokenizer
            .possible_future_terminals_iter(source)
            .eq(std::iter::once(terminal))
        && parser_cache
            .admitted(state.constraint, root_parser_nodes[0])
            .contains(terminal as usize)
    {
        state
            .constraint
            .tokenizer
            .virtual_residual_direct_coordinate(source)
            .map(|coordinate| {
                (
                    FirstMatchDirectCoordinate::Virtual(coordinate),
                    root_parser_nodes[0],
                    terminal,
                )
            })
    } else {
        None
    };
    // Ordinary product states can share several lexer terminals even when the
    // parser admits exactly one of them. The retained product-trace sidecar
    // maps such a physical source to the exact standalone-terminal residual.
    // Use that coordinate directly until the first terminal match; all parser
    // interaction before then is provably irrelevant.
    let first_match_direct_root = first_match_direct_root.or_else(|| {
        if std::env::var_os("GLRMASK_EXPERIMENT_PHYSICAL_FIRST_MATCH_DIRECT").is_none()
            || !HOT_SINGLE_ROOT
            || root_branches.len() != 1
            || !root_branches[0].initial_prune_guard.is_passed()
            || vocab.mask_runtime_tokenizer().is_some()
            || state.constraint.ignore_terminal.is_some()
        {
            return None;
        }
        let source = root_branches[0].exact_tokenizer_state?;
        if vocab.mask_runtime_state(source) != source
            || transitions.exact_raw_state(root_branches[0].tokenizer_config) != Some(source)
            || state.constraint.tokenizer.state_has_epsilon_transitions(source)
        {
            return None;
        }
        let parser_node = root_parser_nodes[0];
        let admitted = parser_cache.admitted(state.constraint, parser_node);
        let mut terminals = admitted
            .iter_ones()
            .filter_map(|terminal| TerminalID::try_from(terminal).ok());
        let terminal = terminals.next()?;
        if terminals.next().is_some() {
            return None;
        }
        let coordinate = state
            .constraint
            .tokenizer
            .terminal_residual_direct_coordinate(source, terminal)?;
        Some((
            FirstMatchDirectCoordinate::Physical(coordinate),
            parser_node,
            terminal,
        ))
    });
    // The config walker is the exact fallback for virtual residual coordinates
    // that were not materialized into a finite mask projection. Use the same
    // language-defined master slicer as the dense walker when one exact source
    // root is available. Failure to prove a language simply declines slicing.
    let mut master_decision = None::<(u16, bool)>;
    if std::env::var_os("GLRMASK_DISABLE_CONFIG_MASTER").is_none()
        && HOT_SINGLE_ROOT
        && root_branches.len() == 1
        && root_branches[0].initial_prune_guard.is_passed()
        && vocab.llg_master_trie().is_some()
        && let Some(source) = root_branches[0].exact_tokenizer_state
        && let Some(safe_plus) = vocab.llg_slice_by_cache_id(0)
        && let Some(whitespace) = vocab.llg_slice_by_cache_id(3)
    {
        let parser_node = root_parser_nodes[0];
        let lexer_state = root_branches[0].tokenizer_config;
        let admitted = parser_cache.admitted(state.constraint, parser_node).clone();

        // The config walker is not limited to virtual residuals: ordinary
        // epsilon-NFA configurations can also denote a broad regular language
        // that is exactly transparent to a slice.  Consider every parser-
        // admitted terminal here, then use byte support + current-config
        // liveness as cheap necessary conditions before the exact proof.  The
        // symbolic residual proof remains an optional fast proof for virtual
        // sources; `direct_slice_prefix_contained_config` is the exact fallback
        // for ordinary epsilon configurations.
        let proof_terminals = admitted
            .iter_ones()
            .filter_map(|terminal| TerminalID::try_from(terminal).ok())
            .collect::<SmallVec<[TerminalID; 4]>>();
        let mut safe_candidates = SmallVec::<[TerminalID; 4]>::new();
        let mut whitespace_candidates = SmallVec::<[TerminalID; 4]>::new();
        for terminal in proof_terminals {
            let Some(support) = state.constraint.tokenizer.terminal_byte_support(terminal) else {
                if std::env::var_os("GLRMASK_PROFILE_DYNAMIC_PROOF_PHASES").is_some() {
                    eprintln!(
                        "[glrmask/profile][config_master_candidate] source={} config={} terminal={} support=none",
                        source, lexer_state, terminal,
                    );
                }
                continue;
            };
            let safe_support = safe_plus.slice_token_bytes().is_subset(&support);
            let whitespace_support = whitespace.slice_token_bytes().is_subset(&support);
            if std::env::var_os("GLRMASK_PROFILE_DYNAMIC_PROOF_PHASES").is_some() {
                eprintln!(
                    "[glrmask/profile][config_master_candidate] source={} config={} terminal={} support_count={} safe_support={} whitespace_support={}",
                    source,
                    lexer_state,
                    terminal,
                    support.len(),
                    safe_support,
                    whitespace_support,
                );
            }
            if safe_support {
                safe_candidates.push(terminal);
            }
            if whitespace_support {
                whitespace_candidates.push(terminal);
            }
        }

        if !safe_candidates.is_empty() {
            let prove = |transitions: &mut T,
                         slice: &crate::runtime::artifact::DynamicMaskSliceTrie,
                         candidates: &[TerminalID]| -> bool {
                let cache_id = slice.cache_id() | 0x0001_0000;
                candidates.iter().copied().any(|terminal| {
                    let prepared_slot = if slice.cache_id() == 0 {
                        Some(0usize)
                    } else if slice.cache_id() == 3 {
                        Some(1usize)
                    } else {
                        None
                    };
                    if let Some(prepared) = prepared_slot.and_then(|slot| {
                        vocab.prepared_master_proof_result(source, slot, terminal)
                    }) {
                        return prepared;
                    }
                    if !transitions.future_contains(lexer_state, terminal)
                        && !transitions.matched_terminals(lexer_state).contains(&terminal)
                    {
                        return false;
                    }
                    if std::env::var_os("GLRMASK_DISABLE_DEMAND_TERMINAL_PROOF").is_none()
                        && !vocab.is_grammar_quotiented()
                    {
                        // This path already consumes projected bounded-radius
                        // proofs below, but must first prepare the relevant
                        // terminal. Positive containment is the existing exact
                        // quotient certificate. False/unknown never admits bits:
                        // false proceeds to bounded radius, None to old fallback.
                        vocab.prepare_runtime_projected_terminal_quotient(
                            &state.constraint.tokenizer, terminal,
                        );
                        if let Some(result) = vocab.projected_terminal_slice_contained(
                            terminal, source, slice.cache_id(), slice.dfa(),
                        ) {
                            if std::env::var_os("GLRMASK_DIAG_DEMAND_TERMINAL_PROOF").is_some() {
                                eprintln!("[demand_terminal_proof] generation={} source={} terminal={} slice={} contained={}",
                                    state.generation, source, terminal, slice.cache_id(), result);
                            }
                            return result;
                        }
                    }
                    if let Some(cached) =
                        vocab.cached_direct_slice_contained(terminal, lexer_state, cache_id)
                    {
                        return cached;
                    }
                    let symbolic = full_walk_dense::virtual_residual_slice_prefix_contained(
                        &state.constraint.tokenizer,
                        source,
                        terminal,
                        slice.dfa(),
                        1_536,
                    );
                    // A raw virtual-residual source is epsilon-free and owned
                    // by exactly one terminal.  In that singleton case the
                    // symbolic result is the complete current config language:
                    // an exact `Some(false)` cannot be rescued by unioning a
                    // second lexer branch, so do not launch the much larger
                    // direct product proof after an exact disproof.  Epsilon /
                    // union sources retain the direct fallback below.
                    let symbolic_complete_for_source = state
                        .constraint
                        .tokenizer
                        .virtual_residual_terminal_for_state(source)
                        == Some(terminal)
                        && !state
                            .constraint
                            .tokenizer
                            .state_has_epsilon_transitions(source);
                    let direct = if symbolic == Some(true)
                        || (symbolic == Some(false) && symbolic_complete_for_source)
                    {
                        None
                    } else {
                        direct_slice_prefix_contained_config(
                            transitions,
                            lexer_state,
                            terminal,
                            slice.dfa(),
                            32 * 1024,
                        )
                    };
                    let result = symbolic == Some(true) || direct == Some(true);
                    if std::env::var_os("GLRMASK_PROFILE_DYNAMIC_PROOF_PHASES").is_some() {
                        eprintln!(
                            "[glrmask/profile][config_master_proof] source={} config={} terminal={} slice={} symbolic={:?} direct={:?} result={}",
                            source,
                            lexer_state,
                            terminal,
                            slice.cache_id(),
                            symbolic,
                            direct,
                            result,
                        );
                    }
                    vocab.cache_direct_slice_contained(
                        terminal,
                        lexer_state,
                        cache_id,
                        result,
                    );
                    result
                })
            };

            let safe_plus_proved = prove(transitions, safe_plus, &safe_candidates);
            let safe_radius = if safe_plus_proved {
                u16::MAX
            } else {
                let max_vocab_safe_chars = u32::from(vocab.llg_master_max_safe_chars());
                safe_candidates
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
                        let symbolic_started = std::env::var_os(
                            "GLRMASK_PROFILE_DYNAMIC_PROOF_PHASES",
                        )
                        .is_some()
                        .then(std::time::Instant::now);
                        let symbolic = full_walk_dense::virtual_residual_safe_repeat_radius(
                            &state.constraint.tokenizer,
                            source,
                            terminal,
                            safe_plus.dfa(),
                            max_vocab_safe_chars,
                            16 * 1024,
                        );
                        if std::env::var_os("GLRMASK_PROFILE_DYNAMIC_PROOF_PHASES").is_some() {
                            eprintln!(
                                "[glrmask/profile][config_master_radius] source={} config={} terminal={} projected={:?} symbolic={:?} max={} symbolic_ms={:.3}",
                                source,
                                lexer_state,
                                terminal,
                                projected,
                                symbolic,
                                max_vocab_safe_chars,
                                symbolic_started
                                    .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1e3),
                            );
                        }
                        let known_radius = match (projected, symbolic) {
                            (Some(left), Some(right)) => Some(left.max(right)),
                            (left, right) => left.or(right),
                        };
                        known_radius.or_else(|| {
                            // Scoped/composed/projected coordinates must never be
                            // interpreted as physical coordinates of the owner.
                            if !raw_terminal_radius_enabled()
                                || vocab.mask_runtime_tokenizer().is_some()
                                || transitions.exact_raw_state(lexer_state) != Some(source)
                            {
                                return None;
                            }
                            vocab.raw_terminal_slice_repeat_radius(
                                &state.constraint.tokenizer, terminal, source,
                                safe_plus.cache_id(), safe_plus.dfa(),
                                max_vocab_safe_chars.min(16), 65_536,
                            )
                        })
                    })
                    .filter_map(|radius| u16::try_from(radius).ok())
                    .max()
                    .unwrap_or(0)
            };
            let whitespace_proved = !whitespace_candidates.is_empty()
                && prove(transitions, whitespace, &whitespace_candidates);
            if safe_radius != 0 || whitespace_proved {
                let master = vocab.llg_master_trie().expect("master trie checked above").trie();
                let residual_ops = vocab.llg_master_residual_ops(safe_radius, whitespace_proved);
                let ordinary_ops = trie.full_walk_ops().len();
                // A master certificate is semantically exact but not automatically
                // profitable: selecting the partitioned master trie also gives up
                // the config walker's specialized hot-edge lane.  Route through
                // the master trie only when its exact residual strict-walk volume
                // is sufficiently smaller than the ordinary trie.  Keep the
                // threshold runtime-configurable while we stabilize it across the
                // canonical population; 500 means residual <= half ordinary.
                let max_permille = std::env::var("GLRMASK_EXPERIMENT_CONFIG_MASTER_MAX_RESIDUAL_PERMILLE")
                    .ok()
                    .and_then(|value| value.parse::<usize>().ok())
                    .unwrap_or(1000);
                let profitable = residual_ops.is_some_and(|residual| {
                    residual.saturating_mul(1000)
                        <= ordinary_ops.saturating_mul(max_permille)
                });
                if std::env::var_os("GLRMASK_PROFILE_MASTER_PROFITABILITY").is_some() {
                    eprintln!(
                        "[glrmask/profile][config_master_profitability] generation={} source={} config={} safe_radius={} whitespace={} ordinary_ops={} master_ops={} residual_ops={:?} max_permille={} profitable={}",
                        state.generation,
                        source,
                        lexer_state,
                        safe_radius,
                        whitespace_proved,
                        ordinary_ops,
                        master.full_walk_ops().len(),
                        residual_ops,
                        max_permille,
                        profitable,
                    );
                }
                if profitable {
                    master_decision = Some((safe_radius, whitespace_proved));
                }
            }
        }
    }
    // Finite-exclusion master cover for native multi-root unions.  A same-parser
    // root set can contain a broad `Exclude(open, finite_literals)` source which
    // the legacy single-root master cannot inspect.  The compile-time sidecar
    // gives an exact source -> LEFT residual coordinate and proves RIGHT finite.
    // If LEFT remains live and non-finalizing for the entire safe-content
    // language, every pure-safe vocabulary token is a valid prefix through that
    // root: any still-live finite exclusion can be pumped beyond its maximum
    // remaining length before LEFT is completed.  The existing master trie then
    // seeds exactly those proved tokens; quotes, escapes and boundaries stay on
    // the ordinary exact residual walk.
    if std::env::var_os("GLRMASK_DISABLE_FINITE_EXCLUSION_MASTER").is_none()
        && master_decision.is_none()
        && root_branches.len() > 1
        && root_branches
            .iter()
            .all(|branch| branch.initial_prune_guard.is_passed())
        && let Some(&parser_node) = root_parser_nodes.first()
        && root_parser_nodes.iter().all(|&node| node == parser_node)
        && let Some(safe_plus) = vocab.llg_slice_by_cache_id(0)
        && vocab.llg_master_trie().is_some()
    {
        let admitted = parser_cache.admitted(state.constraint, parser_node).clone();
        let mut proved = None::<(u32, TerminalID)>;
        'sources: for branch in root_branches {
            let Some(source) = branch.exact_tokenizer_state else {
                continue;
            };
            for terminal_index in admitted.iter_ones() {
                let Ok(terminal) = TerminalID::try_from(terminal_index) else {
                    continue;
                };
                if Some(terminal) == state.constraint.ignore_terminal
                    || state
                        .constraint
                        .tokenizer
                        .terminal_exclusion_continuation(source, terminal)
                        .is_none()
                {
                    continue;
                }
                if exclusion_left_slice_prefix_live(
                    &state.constraint.tokenizer,
                    source,
                    terminal,
                    safe_plus.dfa(),
                    8 * 1024,
                ) == Some(true)
                {
                    proved = Some((source, terminal));
                    break 'sources;
                }
            }
        }
        if let Some((source, terminal)) = proved {
            let safe_radius = vocab.llg_master_max_safe_chars();
            let ordinary_ops = trie.full_walk_ops().len();
            let max_permille = std::env::var(
                "GLRMASK_EXPERIMENT_CONFIG_MASTER_MAX_RESIDUAL_PERMILLE",
            )
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(1000);
            let residual_ops = vocab.llg_master_residual_ops(safe_radius, false);
            let profitable = residual_ops.is_some_and(|residual| {
                residual.saturating_mul(1000)
                    <= ordinary_ops.saturating_mul(max_permille)
            });
            if std::env::var_os("GLRMASK_PROFILE_DYNAMIC_PROOF_PHASES").is_some() {
                eprintln!(
                    "[glrmask/profile][finite_exclusion_master] generation={} source={} terminal={} safe_radius={} ordinary_ops={} residual_ops={:?} max_permille={} profitable={}",
                    state.generation,
                    source,
                    terminal,
                    safe_radius,
                    ordinary_ops,
                    residual_ops,
                    max_permille,
                    profitable,
                );
            }
            if profitable {
                master_decision = Some((safe_radius, false));
            }
        }
    }
    const OPTIONAL_SPACE_NONSPACE_SLICE: u32 = 0x40;
    // Default-on, with a diagnostic opt-out for equal-binary comparisons.
    static WORD_SLICE_ENABLED: OnceLock<bool> = OnceLock::new();
    let direct_residual_slice = if *WORD_SLICE_ENABLED.get_or_init(|| {
        std::env::var_os("GLRMASK_DISABLE_DIRECT_RESIDUAL_WORD_SLICE").is_none()
    }) {
        first_match_direct_root.and_then(|(coordinate, _, _)| {
            let FirstMatchDirectCoordinate::Virtual(coordinate) = coordinate else {
                return None;
            };
            let slice = vocab.llg_slice_by_cache_id(OPTIONAL_SPACE_NONSPACE_SLICE)?;
            // A proved word slice is not automatically the cheapest exact
            // route. For a large safe radius, the existing master certificate
            // can leave substantially less work than the word-only trie.
            // Compare immutable residual op counts before paying for another
            // proof; declining this optional route leaves the original
            // master/ordinary execution and its accepted language unchanged.
            static WORD_ROUTE_PERMILLE: OnceLock<Option<usize>> = OnceLock::new();
            if let Some(max_permille) = *WORD_ROUTE_PERMILLE.get_or_init(|| {
                std::env::var("GLRMASK_EXPERIMENT_DIRECT_RESIDUAL_WORD_ROUTE_PERMILLE")
                    .ok()
                    .and_then(|value| value.trim().parse::<usize>().ok())
                    .or(Some(500))
            }) {
                let word_ops = slice.trie().full_walk_ops().len();
                let existing_ops = master_decision
                    .and_then(|(radius, whitespace)| {
                        vocab.llg_master_residual_ops(radius, whitespace)
                    })
                    .unwrap_or_else(|| trie.full_walk_ops().len());
                let prefer_word = word_ops.saturating_mul(1000)
                    < existing_ops.saturating_mul(max_permille);
                if std::env::var_os("GLRMASK_PROFILE_WORD_ROUTE").is_some()
                    && dynamic_mask_profile_enabled(state.generation)
                {
                    eprintln!(
                        "[glrmask/profile][word_route] generation={} master={:?} word_ops={} existing_ops={} max_permille={} attempt={}",
                        state.generation, master_decision, word_ops, existing_ops,
                        max_permille, prefer_word,
                    );
                }
                if !prefer_word {
                    return None;
                }
            }
            let work_limit = std::env::var(
                "GLRMASK_EXPERIMENT_DIRECT_RESIDUAL_ASCII_WORD_SLICE_WORK_LIMIT",
            )
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(8 * 1024);
            let result = state
                .constraint
                .tokenizer
                .virtual_residual_direct_coordinate_parser_transparent_byte_dfa(
                    coordinate,
                    slice.dfa().start_state(),
                    slice.dfa().class_count(),
                    slice.dfa().byte_to_class_map(),
                    slice.dfa().transition_table(),
                    slice.dfa().can_reach_accepting_map(),
                    slice.dfa().has_finite_language(),
                    work_limit,
                );
            (result == Some(true)).then_some(slice)
        })
    } else {
        None
    };
    let trie = if let Some(slice) = direct_residual_slice {
        slice.trie()
    } else {
        master_decision
            .and_then(|_| vocab.llg_master_trie())
            .map_or(trie, |slice| slice.trie())
    };
    let provider_adaptive_output = transitions.prefer_adaptive_output();
    let deferred_output = (provider_adaptive_output || vocab.is_grammar_quotiented())
        && master_decision.is_none()
        && direct_residual_slice.is_none()
        && (provider_adaptive_output || (state.constraint.ignore_terminal.is_none()
        && root_branches
            .iter()
            .all(|branch| branch.initial_prune_guard.is_passed())));
    let legacy_hot_conditions_clear = std::env::var_os(
        "GLRMASK_EXPERIMENT_CONFIG_SCALAR_CONDITIONED",
    )
    .is_none()
        && std::env::var_os("GLRMASK_EXPERIMENT_CONFIG_SCALAR_CONDITIONED_DEAD_SKIP").is_none()
        && config_scalar_conditioned_budget().is_none();
    let hot_edge_lane_eligible = HOT_SINGLE_ROOT
        && root_branches.len() == 1
        && root_branches[0].initial_prune_guard.is_passed()
        && first_match_direct_root.is_none()
        && direct_residual_slice.is_none()
        && master_decision.is_none()
        && !deferred_output
        && transitions.hot_scalar_lane_enabled()
        && std::env::var_os("GLRMASK_PROFILE_DYNAMIC_CONFIG_TRANSITIONS").is_none()
        && (root_branches[0].parser_filtered_root || legacy_hot_conditions_clear);
    if dynamic_mask_profile_enabled(state.generation) {
        eprintln!(
            "[glrmask/profile][hot_scalar_gate] generation={} eligible={} all_consume={} root_config={} exact_raw={:?} hot_id={:?}",
            state.generation,
            hot_edge_lane_eligible,
            trie.full_walk_all_consume(),
            root_branches.first().map_or(u32::MAX, |root| root.tokenizer_config),
            root_branches.first().and_then(|root| transitions.exact_raw_state(root.tokenizer_config)),
            root_branches.first().and_then(|root| transitions.hot_state_id(root.tokenizer_config)),
        );
    }
    if hot_edge_lane_eligible {
        match try_full_walk_hot_scalar_edges(
            state,
            vocab,
            trie,
            root_branches[0].tokenizer_config,
            root_parser_nodes[0],
            initial_lexer_state,
            &mut parser_cache,
            transitions,
            buf,
        )? {
            Some(used) => return Ok(used),
            None => {
                let all_words = vocab.all_original_token_words();
                let copy_len = buf.len().min(all_words.len());
                buf[..copy_len].copy_from_slice(&all_words[..copy_len]);
                if copy_len < buf.len() {
                    buf[copy_len..].fill(0);
                }
            }
        }
    }
    let mut deferred_allowed_markers = Vec::<u64>::new();
    let mut deferred_rejected_markers = Vec::<u64>::new();
    let mut deferred_dead_subtrees = Vec::<u32>::new();
    let mut deferred_admitted_subtrees = Vec::<u32>::new();
    let mut deferred_positive_work = 0usize;
    let mut deferred_negative_work = 0usize;
    // Scalar is overwhelmingly dominant. Encode dead/multi directly in the
    // lexer-state coordinate so the common DFS path needs no separate kind
    // load/store. Full-walk lexer states are bounded far below these u32
    // sentinels by the dense-transition memory budget.
    const FULL_WALK_LEXER_TWO_DISTINCT: u32 = u32::MAX - 3;
    const FULL_WALK_LEXER_TWO: u32 = u32::MAX - 2;
    const FULL_WALK_LEXER_MULTI: u32 = u32::MAX - 1;
    const FULL_WALK_LEXER_DEAD: u32 = u32::MAX;
    let stack_len = usize::from(trie.full_walk_max_parent_depth()).saturating_add(2);
    // Keep the overwhelmingly common <=256-depth vocabulary on the original
    // fixed-array storage.  `SmallVec` here looks attractive, but every
    // `get_unchecked` still has to resolve inline-vs-spilled storage and that
    // showed up measurably across a complete 128k-token walk.  Deep tries pay
    // one allocation up front and then use the same plain-slice inner loop.
    let mut inline_lexer = [FULL_WALK_LEXER_DEAD; 256];
    let mut heap_lexer = Vec::<u32>::new();
    let stack_lexer: &mut [u32] = if stack_len <= inline_lexer.len() {
        &mut inline_lexer[..stack_len]
    } else {
        heap_lexer.resize(stack_len, FULL_WALK_LEXER_DEAD);
        heap_lexer.as_mut_slice()
    };
    let mut inline_parser_transparent = [false; 256];
    let mut heap_parser_transparent = Vec::<bool>::new();
    let stack_parser_transparent: &mut [bool] = if stack_len <= inline_parser_transparent.len() {
        &mut inline_parser_transparent[..stack_len]
    } else {
        heap_parser_transparent.resize(stack_len, false);
        heap_parser_transparent.as_mut_slice()
    };
    let mut inline_first_match_direct: [Option<FirstMatchDirectCoordinate>; 256] = [None; 256];
    let mut heap_first_match_direct = Vec::<Option<FirstMatchDirectCoordinate>>::new();
    let stack_first_match_direct: &mut [Option<FirstMatchDirectCoordinate>] =
        if stack_len <= inline_first_match_direct.len() {
            &mut inline_first_match_direct[..stack_len]
        } else {
            heap_first_match_direct.resize(stack_len, None);
            heap_first_match_direct.as_mut_slice()
        };
    let defer_first_match_finalizer = DEFER_FIRST_MATCH_FINALIZER
        || first_match_direct_root
            .as_ref()
            .is_some_and(|(coordinate, _, _)| coordinate.physical().is_some());
    // This lane is experimental and disabled in production. Do not eagerly
    // construct hundreds of `Option<FullWalkManyState>` stack slots on every
    // mask when it is off; the old inline storage was a measurable hot-path
    // tax even though none of the slots could ever be read.
    let mut first_match_accepting_storage = if defer_first_match_finalizer {
        vec![false; stack_len]
    } else {
        Vec::new()
    };
    let stack_first_match_accepting = first_match_accepting_storage.as_mut_slice();
    let mut first_match_side_storage = if defer_first_match_finalizer {
        std::iter::repeat_with(|| None).take(stack_len).collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let stack_first_match_side = first_match_side_storage.as_mut_slice();
    let mut inline_parser = [0u32; 256];
    let mut heap_parser = Vec::<u32>::new();
    let stack_parser: &mut [u32] = if stack_len <= inline_parser.len() {
        &mut inline_parser[..stack_len]
    } else {
        heap_parser.resize(stack_len, 0);
        heap_parser.as_mut_slice()
    };
    let mut inline_two = [((0u32, 0u32), (0u32, 0u32)); 256];
    let mut heap_two = Vec::<((u32, u32), (u32, u32))>::new();
    let stack_two: &mut [((u32, u32), (u32, u32))] = if stack_len <= inline_two.len() {
        &mut inline_two[..stack_len]
    } else {
        heap_two.resize(stack_len, ((0, 0), (0, 0)));
        heap_two.as_mut_slice()
    };
    // Most products use compact cache IDs. Reserve only a few owned slots
    // inline, growing lazily to depths actually saved (not the whole trie).
    let mut stack_many = SmallVec::<[Option<FullWalkManyState>; 8]>::new();
   let mut pair_union_cache = FxHashMap::<(u32, u32), Option<u32>>::default();
    let mut triple_union_cache = FxHashMap::<(u32, u32, u32), Option<u32>>::default();
    let product_transition_cache_capacity = transitions.product_transition_cache_capacity();
    let cache_two_branch_products = transitions.cache_two_branch_products();
    let cache_single_branch_products = transitions.cache_single_branch_products();
    // Ordinary providers have no product cache. Do not construct its maps,
    // row/alphabet vectors, profile state, or large stack frame for them. The
    // scoped cache remains per-mask and exact; indirection changes storage,
    // not a transition, admission proof, or the capacity-exhaustion fallback.
    let mut many_transition_cache = (product_transition_cache_capacity != 0).then(|| {
        Box::new(FullWalkManyTransitionCache::new(
            product_transition_cache_capacity, cache_two_branch_products,
            cache_single_branch_products,
        ))
    });
    let mut stack_many_ids = if product_transition_cache_capacity != 0 {
        vec![FullWalkManyTransitionCache::UNKNOWN; stack_len]
    } else { Vec::new() };
    if root_branches.len() == 1 && root_branches[0].initial_prune_guard.is_passed() {
        stack_lexer[0] = root_branches[0].tokenizer_config;
        stack_parser[0] = root_parser_nodes[0];
        stack_parser_transparent[0] = root_branches[0].parser_filtered_transparent;
        if let Some((coordinate, _, _)) = first_match_direct_root {
            stack_first_match_direct[0] = Some(coordinate);
            if defer_first_match_finalizer {
                stack_first_match_accepting[0] = false;
            }
        }
    } else if root_branches.len() == 2
        && root_branches.iter().all(|root| root.initial_prune_guard.is_passed())
    {
        let first = (root_branches[0].tokenizer_config, root_parser_nodes[0]);
        let second = (root_branches[1].tokenizer_config, root_parser_nodes[1]);
        if let Some((lexer_state, parser_node)) =
            full_walk_merge_two_same_parser(transitions, &mut pair_union_cache, first, second)
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
                    prune_guard: FullWalkPruneGuard::from_initial(
                        &root.initial_prune_guard,
                        transitions,
                    )?,
                },
            );
        }
        if let Some((lexer_state, parser_node)) = full_walk_merge_branches_same_parser(
            transitions,
            &mut pair_union_cache,
            &mut triple_union_cache,
            &roots,
        )
        {
            stack_lexer[0] = lexer_state;
            stack_parser[0] = parser_node;
        } else {
            *full_walk_owned_many_slot(&mut stack_many, 0, stack_len) =
                Some(full_walk_many_state_from_branches(roots));
        }
    }

    let walk_ops = trie.full_walk_ops();
    let token_markers = vocab.full_walk_token_markers_for(trie);
    let mut token_marker_index = 0usize;
    let mut scalar_lexer = FULL_WALK_LEXER_DEAD;
    let mut scalar_parser = 0u32;
    let mut first_match_direct = None::<FirstMatchDirectCoordinate>;
    let mut parser_transparent = false;
    let mut first_match_direct_accepting = false;
    let mut first_match_side = None::<FullWalkManyState>;
    let first_match_terminal = first_match_direct_root.map(|(_, _, terminal)| terminal);
    let mut first_match_direct_future_cache =
        FxHashMap::<FirstMatchDirectCoordinate, bool>::default();
    let mut current_two = ((0u32, 0u32), (0u32, 0u32));
    let mut current_many = FullWalkManyState::Branches(FullWalkBranches::new());
    let mut current_many_id = FullWalkManyTransitionCache::UNKNOWN;
    let mut partition_root_slot = 0usize;

    // Exact small root-byte scheduler for the generic full-walk backend.
    // The filtered epsilon-NFA root and certified terminal-continuation roots
    // often expose only one or two relevant first bytes even though the shared
    // tokenizer contains hundreds of terminals. Walk only those disjoint root
    // byte subtrees and rebuild the positive mask from zero.
    static GENERIC_SINGLE_BYTE_ROOT_WALK_ENABLED: std::sync::OnceLock<bool> =
        std::sync::OnceLock::new();
    let sparse_root_walk_enabled = *GENERIC_SINGLE_BYTE_ROOT_WALK_ENABLED.get_or_init(|| {
        std::env::var_os("GLRMASK_DISABLE_SINGLE_BYTE_ROOT_WALK").is_none()
    });
    let sparse_root_bytes = if sparse_root_walk_enabled
        && !deferred_output
        && master_decision.is_none()
        && direct_residual_slice.is_none()
        && first_match_direct_root.is_none()
        && root_branches.len() == 1
        && root_branches[0].initial_prune_guard.is_passed()
        && vocab.mask_runtime_tokenizer().is_none()
        && stack_lexer[0] < FULL_WALK_LEXER_TWO_DISTINCT
        && trie.node(0).token_id.is_none()
        && trie.has_full_walk_root_byte_index()
    {
        let root = &root_branches[0];
        let mut bytes = None::<U8Set>;
        if let (Some(source), Some(terminal)) = (
            root.exact_tokenizer_state,
            root.residual_continuation_terminal,
        ) {
            if transitions.exact_raw_state(stack_lexer[0]) == Some(source)
                && let Some(coordinate) = state
                    .constraint
                    .tokenizer
                    .terminal_residual_direct_coordinate(source, terminal)
            {
                let mut residual = U8Set::empty();
                for raw in 0u16..=255 {
                    let byte = raw as u8;
                    if state
                        .constraint
                        .tokenizer
                        .terminal_residual_direct_coordinate_step(coordinate, byte)
                        .is_some()
                    {
                        residual.insert(byte);
                    }
                }
                if !residual.is_empty() {
                    bytes = Some(residual);
                }
            }
        }
        if bytes.is_none() && root.parser_filtered_root {
            bytes = transitions.exact_next_bytes(stack_lexer[0]);
        }
        if bytes.is_none()
            && let Some(source) = root.exact_tokenizer_state
            && transitions.exact_raw_state(stack_lexer[0]) == Some(source)
            && source < state.constraint.tokenizer.num_states()
        {
            let mut raw_bytes = U8Set::empty();
            for (byte, _) in state.constraint.tokenizer.transitions_from(source) {
                raw_bytes.insert(byte);
            }
            bytes = Some(raw_bytes);
        }
        bytes.filter(|bytes| !bytes.is_empty() && bytes.len() <= 8)
    } else {
        None
    };
    let mut sparse_root_ranges = SmallVec::<[(u32, u32, usize); 8]>::new();
    if let Some(bytes) = sparse_root_bytes {
        for raw in 0u16..=255 {
            let byte = raw as u8;
            if !bytes.contains(byte) {
                continue;
            }
            if let Some(range) = trie.full_walk_root_byte_range(byte) {
                sparse_root_ranges.push(range);
            }
        }
        let scheduled_ops = sparse_root_ranges
            .iter()
            .map(|(start, end, _)| end.saturating_sub(*start) as usize)
            .sum::<usize>();
        if sparse_root_ranges.is_empty()
            || scheduled_ops.saturating_mul(2) > walk_ops.len()
        {
            sparse_root_ranges.clear();
        }
    }
    let sparse_positive_rebuild = !sparse_root_ranges.is_empty();
    let mut sparse_root_range_index = 0usize;
    let mut sparse_root_range_end = 0u32;
    if sparse_positive_rebuild {
        buf.fill(0);
    }
    let mut remaining_ops = if let Some(&(start, end, marker_start)) = sparse_root_ranges.first() {
        sparse_root_range_end = end;
        token_marker_index = marker_start;
        walk_ops[start as usize..].iter()
    } else {
        walk_ops.iter()
    };
    let profile_generic_work =
        std::env::var_os("GLRMASK_PROFILE_DYNAMIC_CONFIG_TRANSITIONS").is_some();
    let mut profile_byte_ops = 0usize;
    let mut profile_scalar_byte_ops = 0usize;
    let mut profile_two_byte_ops = 0usize;
    let mut profile_multi_byte_ops = 0usize;
    let mut profile_dead_byte_ops = 0usize;
    let mut profile_token_endpoints = 0usize;
    let mut profile_allowed_token_endpoints = 0usize;
    let mut profile_blocked_token_endpoints = 0usize;
    let mut profile_blocked_dead_endpoints = 0usize;
    let mut profile_blocked_original_tokens = 0usize;
    let mut profile_dead_subtree_original_tokens = 0usize;
    let mut profile_admitted_subtree_original_tokens = 0usize;
    let mut profile_first_match_direct_byte_ops = 0usize;
    let mut profile_first_match_direct_finalizers = 0usize;
    let mut profile_first_match_direct_endpoints = 0usize;
    let mut condition_consecutive_non_pruning = 0u32;
    static CONDITION_CONFIG_SCALAR: std::sync::OnceLock<bool> =
        std::sync::OnceLock::new();
    static CONDITION_CONFIG_SCALAR_DEAD_SKIP: std::sync::OnceLock<bool> =
        std::sync::OnceLock::new();
    static CONDITION_CONFIG_SCALAR_INITIAL_ONLY: std::sync::OnceLock<bool> =
        std::sync::OnceLock::new();
    let condition_budget = config_scalar_conditioned_budget();
    let provider_conditioned_dead_skip = transitions.parser_conditioned_dead_skip_default();
    let condition_dead_skip = provider_conditioned_dead_skip
        || *CONDITION_CONFIG_SCALAR_DEAD_SKIP.get_or_init(|| {
        std::env::var_os("GLRMASK_EXPERIMENT_CONFIG_SCALAR_CONDITIONED_DEAD_SKIP")
            .is_some()
    }) || condition_budget.is_some();
    let condition_config_scalar = provider_conditioned_dead_skip
        || *CONDITION_CONFIG_SCALAR.get_or_init(|| {
        std::env::var_os("GLRMASK_EXPERIMENT_CONFIG_SCALAR_CONDITIONED").is_some()
    }) || condition_dead_skip;
    let condition_initial_only = *CONDITION_CONFIG_SCALAR_INITIAL_ONLY.get_or_init(|| {
        std::env::var_os("GLRMASK_EXPERIMENT_CONFIG_SCALAR_CONDITIONED_INITIAL_ONLY").is_some()
    });
    static SKIP_PREFIX_PROBES: OnceLock<bool> = OnceLock::new();
    let skip_prefix_probes = *SKIP_PREFIX_PROBES.get_or_init(|| {
        std::env::var_os("GLRMASK_PROFILE_SKIP_PREFIX_PROBES").is_some()
    });
    let condition_walk_enabled = condition_config_scalar
        && (!condition_initial_only || state.generation == 0) && !skip_prefix_probes;
    let condition_dead_skip = condition_dead_skip && !skip_prefix_probes;
    loop {
        if sparse_positive_rebuild {
            let current_op = walk_ops.len() - remaining_ops.as_slice().len();
            if current_op >= sparse_root_range_end as usize {
                sparse_root_range_index += 1;
                let Some(&(start, end, marker_start)) =
                    sparse_root_ranges.get(sparse_root_range_index)
                else {
                    break;
                };
                sparse_root_range_end = end;
                token_marker_index = marker_start;
                remaining_ops = walk_ops[start as usize..].iter();
            }
        }
        let Some(&op) = remaining_ops.next() else {
            break;
        };
        let parent_depth = op.parent_depth() as usize;
        if op.starts_edge() {
            if let Some(already) = transitions.pre_admitted_original_mask() {
                let op_index = walk_ops.len() - remaining_ops.as_slice().len() - 1;
                let (child, _) = trie.full_walk_dead_subtree(op_index);
                if vocab.subtree_original_tokens_for(trie, child).iter().all(|&id| {
                    already.get(id as usize / 32).is_some_and(|&w| w & (1 << (id % 32)) != 0)
                }) {
                    full_walk_skip_admitted_subtree_generic(trie, walk_ops,
                        &mut remaining_ops, &mut token_marker_index);
                    continue;
                }
            }
            scalar_lexer = unsafe { *stack_lexer.get_unchecked(parent_depth) };
            first_match_direct = unsafe { *stack_first_match_direct.get_unchecked(parent_depth) };
            parser_transparent = unsafe { *stack_parser_transparent.get_unchecked(parent_depth) };
            if first_match_direct.is_some() && defer_first_match_finalizer {
                first_match_direct_accepting =
                    unsafe { *stack_first_match_accepting.get_unchecked(parent_depth) };
                first_match_side = unsafe {
                    stack_first_match_side
                        .get_unchecked(parent_depth)
                        .as_ref()
                        .cloned()
                };
            } else {
                first_match_direct_accepting = false;
                first_match_side = None;
            }
            if scalar_lexer < FULL_WALK_LEXER_TWO_DISTINCT {
                scalar_parser = unsafe { *stack_parser.get_unchecked(parent_depth) };
            } else if scalar_lexer == FULL_WALK_LEXER_TWO_DISTINCT || scalar_lexer == FULL_WALK_LEXER_TWO {
                current_two = unsafe { *stack_two.get_unchecked(parent_depth) };
            } else if scalar_lexer == FULL_WALK_LEXER_MULTI {
                if product_transition_cache_capacity != 0 {
                    current_many_id = stack_many_ids[parent_depth];
                }
                if product_transition_cache_capacity == 0
                    || current_many_id == FullWalkManyTransitionCache::UNKNOWN
                {
                    // An uncached MULTI parent is saved either at the root
                    // or after its incoming edge, before any descendant read.
                    debug_assert!(parent_depth < stack_many.len());
                    current_many.clone_from(unsafe {
                        stack_many
                            .get_unchecked(parent_depth)
                            .as_ref()
                            .unwrap_unchecked()
                    });
                }
            }

            // A subtree whose parent configuration is already lexically dead
            // cannot revive. Skip the entire structural edge/subtree immediately
            // instead of replaying every byte with DEAD carried through.
            if scalar_lexer == FULL_WALK_LEXER_DEAD && first_match_direct.is_none() {
                let skipped = if deferred_output {
                    let (child, count) = full_walk_skip_dead_subtree_generic_deferred(
                        vocab, trie, walk_ops, &mut remaining_ops, &mut token_marker_index,
                    );
                    deferred_dead_subtrees.push(child);
                    deferred_negative_work += count;
                    count
                } else if sparse_positive_rebuild {
                    let (_, count) = full_walk_skip_dead_subtree_generic_deferred(
                        vocab, trie, walk_ops, &mut remaining_ops, &mut token_marker_index,
                    );
                    count
                } else {
                    full_walk_skip_dead_subtree_generic(
                        vocab, trie, walk_ops, &mut remaining_ops, &mut token_marker_index, buf,
                    )
                };
                if profile_generic_work {
                    profile_dead_subtree_original_tokens += skipped;
                }
                continue;
            }

            if parent_depth == 0 && !op.consumes_byte() {
                let root_slot = partition_root_slot;
                partition_root_slot += 1;
                if let Some((safe_radius, whitespace)) = master_decision
                    && let Some(class) = trie.root_layout_class(root_slot)
                    && {
                        let safe_chars = dynamic_mask_llg_master_safe_chars(class);
                        (safe_chars != 0 && safe_chars <= safe_radius)
                            || (whitespace && dynamic_mask_llg_master_is_whitespace(class))
                    }
                {
                    full_walk_skip_admitted_subtree_generic(
                        trie,
                        walk_ops,
                        &mut remaining_ops,
                        &mut token_marker_index,
                    );
                    continue;
                }
            }
        }

        if cache_single_branch_products && product_transition_cache_capacity != 0
            && scalar_lexer < FULL_WALK_LEXER_TWO_DISTINCT && first_match_direct.is_none()
        {
            let single = FullWalkManyState::Branches(smallvec::smallvec![
                FullWalkBranch { lexer_state: scalar_lexer, parser_node: scalar_parser,
                    prune_guard: FullWalkPruneGuard::Passed },
            ]);
            if let Some(id) = many_transition_cache.as_mut().unwrap().intern(&single) {
                current_many_id = id;
                scalar_lexer = FULL_WALK_LEXER_MULTI;
            }
        }

        if cache_two_branch_products && product_transition_cache_capacity != 0
            && (scalar_lexer == FULL_WALK_LEXER_TWO || scalar_lexer == FULL_WALK_LEXER_TWO_DISTINCT)
        {
            let pair = FullWalkManyState::Branches(smallvec::smallvec![
                FullWalkBranch { lexer_state: current_two.0.0, parser_node: current_two.0.1,
                    prune_guard: FullWalkPruneGuard::Passed },
                FullWalkBranch { lexer_state: current_two.1.0, parser_node: current_two.1.1,
                    prune_guard: FullWalkPruneGuard::Passed },
            ]);
            if let Some(id) = many_transition_cache.as_mut().unwrap().intern(&pair) {
                current_many_id = id;
                scalar_lexer = FULL_WALK_LEXER_MULTI;
            }
        }

        if op.consumes_byte() {
            if product_transition_cache_capacity != 0
                && scalar_lexer != FULL_WALK_LEXER_MULTI
            {
                current_many_id = FullWalkManyTransitionCache::UNKNOWN;
            }
            if profile_generic_work {
                profile_byte_ops += 1;
                if scalar_lexer == FULL_WALK_LEXER_DEAD {
                    profile_dead_byte_ops += 1;
                } else if scalar_lexer < FULL_WALK_LEXER_TWO_DISTINCT {
                    profile_scalar_byte_ops += 1;
                } else if scalar_lexer == FULL_WALK_LEXER_TWO_DISTINCT
                    || scalar_lexer == FULL_WALK_LEXER_TWO
                {
                    profile_two_byte_ops += 1;
                } else if scalar_lexer == FULL_WALK_LEXER_MULTI {
                    profile_multi_byte_ops += 1;
                }
            }
            let byte = op.byte();
            let mut first_match_consumed = false;
            if let Some(source_coordinate) = first_match_direct {
                if profile_generic_work {
                    profile_first_match_direct_byte_ops += 1;
                }
                if defer_first_match_finalizer {
                    if let Some(side) = first_match_side.as_ref() {
                        let advanced = full_walk_step_many_state(
                            side,
                            byte,
                            initial_lexer_state,
                            transitions,
                            &mut parser_cache,
                            state.constraint,
                        );
                        first_match_side =
                            (!full_walk_many_state_is_empty(&advanced)).then_some(advanced);
                    }

                    let target_coordinate =
                        source_coordinate.step(&state.constraint.tokenizer, byte);
                    if let Some(target_coordinate) = target_coordinate {
                        let accepting = target_coordinate
                            .accepting(&state.constraint.tokenizer)
                            .unwrap_or(false);
                        first_match_direct = Some(target_coordinate);
                        first_match_direct_accepting = accepting;
                        first_match_consumed = true;
                        if accepting {
                            if profile_generic_work {
                                profile_first_match_direct_finalizers += 1;
                            }
                            let terminal = first_match_terminal
                                .expect("direct first-match terminal disappeared");
                            if let Some(next_parser) =
                                parser_cache.advance(state.constraint, scalar_parser, terminal)
                            {
                                let future = target_coordinate
                                    .has_future(&state.constraint.tokenizer)
                                    .unwrap_or(false);
                                let prune_guard = if future {
                                    let memory = match target_coordinate {
                                        FirstMatchDirectCoordinate::Virtual(coordinate) => {
                                            FullWalkPruneMemory::Direct(coordinate, terminal)
                                        }
                                        FirstMatchDirectCoordinate::Physical(coordinate) => {
                                            FullWalkPruneMemory::TerminalResidual(coordinate, terminal)
                                        }
                                    };
                                    FullWalkPruneGuard::Pending(smallvec::smallvec![memory])
                                } else {
                                    FullWalkPruneGuard::Passed
                                };
                                full_walk_many_state_push_branch(
                                    &mut first_match_side,
                                    FullWalkBranch {
                                        lexer_state: parser_cache.initial_lexer(state.constraint, transitions, initial_lexer_state, next_parser),
                                        parser_node: next_parser,
                                        prune_guard,
                                    },
                                );
                            }
                        }
                    } else {
                        first_match_direct = None;
                        first_match_direct_accepting = false;
                        first_match_consumed = true;
                        if let Some(side) = first_match_side.take() {
                            scalar_lexer = FULL_WALK_LEXER_MULTI;
                            current_many = side;
                        } else {
                            scalar_lexer = FULL_WALK_LEXER_DEAD;
                        }
                    }
                } else {
                let target_coordinate =
                    source_coordinate.step(&state.constraint.tokenizer, byte);
                let Some(target_coordinate) = target_coordinate else {
                    first_match_direct = None;
                    scalar_lexer = FULL_WALK_LEXER_DEAD;
                    let cleared = if deferred_output {
                        let (child, count) = full_walk_skip_dead_subtree_generic_deferred(
                            vocab,
                            trie,
                            walk_ops,
                            &mut remaining_ops,
                            &mut token_marker_index,
                        );
                        deferred_dead_subtrees.push(child);
                        deferred_negative_work += count;
                        count
                    } else if sparse_positive_rebuild {
                        let (_, count) = full_walk_skip_dead_subtree_generic_deferred(
                            vocab,
                            trie,
                            walk_ops,
                            &mut remaining_ops,
                            &mut token_marker_index,
                        );
                        count
                    } else {
                        full_walk_skip_dead_subtree_generic(
                            vocab,
                            trie,
                            walk_ops,
                            &mut remaining_ops,
                            &mut token_marker_index,
                            buf,
                        )
                    };
                    if profile_generic_work {
                        profile_dead_subtree_original_tokens += cleared;
                    }
                    continue;
                };
                let accepting = target_coordinate
                    .accepting(&state.constraint.tokenizer)
                    .unwrap_or(false);
                if !accepting {
                    first_match_direct = Some(target_coordinate);
                    first_match_consumed = true;
                } else {
                    // Materialize exactly one source state only at the first
                    // terminal boundary, then replay this finalizing byte in
                    // the ordinary config walker. No intermediate coordinate
                    // on the parser-transparent prefix needs a virtual state
                    // id at all.
                    if profile_generic_work {
                        profile_first_match_direct_finalizers += 1;
                    }
                    let raw_source = source_coordinate
                        .raw_state(&state.constraint.tokenizer)
                        .ok_or_else(|| {
                            "direct first-match coordinate could not materialize its source state"
                                .to_owned()
                        })?;
                    first_match_direct = None;
                    scalar_lexer = transitions.root_state(raw_source)?;
                }
                }
            }
            if first_match_consumed {
                // Parser state is intentionally unchanged until the first
                // terminal finalization.
            } else if scalar_lexer == FULL_WALK_LEXER_DEAD {
            } else if scalar_lexer < FULL_WALK_LEXER_TWO_DISTINCT {
                let cell = if let Some(hot_id) = transitions.hot_state_id(scalar_lexer) {
                    let next = transitions.hot_step(hot_id, byte);
                    if next < FULL_WALK_HOT_SLOW {
                        scalar_lexer = transitions.hot_state_from_id(next);
                        None
                    } else {
                        Some(transitions.cell(scalar_lexer, byte))
                    }
                } else {
                    Some(transitions.cell(scalar_lexer, byte))
                };
                if let Some(cell) = cell {
                if T::cell_is_dead(cell) {
                    scalar_lexer = FULL_WALK_LEXER_DEAD;
                    // A lexically dead scalar configuration cannot revive below
                    // this vocabulary prefix. Mirror the dense walk and jump to
                    // the subtree end instead of iterating dead descendants.
                    let cleared = if deferred_output {
                        let (child, count) = full_walk_skip_dead_subtree_generic_deferred(
                            vocab,
                            trie,
                            walk_ops,
                            &mut remaining_ops,
                            &mut token_marker_index,
                        );
                        deferred_dead_subtrees.push(child);
                        deferred_negative_work += count;
                        count
                    } else if sparse_positive_rebuild {
                        let (_, count) = full_walk_skip_dead_subtree_generic_deferred(
                            vocab,
                            trie,
                            walk_ops,
                            &mut remaining_ops,
                            &mut token_marker_index,
                        );
                        count
                    } else {
                        full_walk_skip_dead_subtree_generic(
                            vocab,
                            trie,
                            walk_ops,
                            &mut remaining_ops,
                            &mut token_marker_index,
                            buf,
                        )
                    };
                    if profile_generic_work {
                        profile_dead_subtree_original_tokens += cleared;
                    }
                    continue;
                } else {
                    let target = T::cell_target(cell);
                    if !T::cell_has_finalizer(cell) {
                        let condition_active = condition_walk_enabled
                            && !parser_transparent
                            && match condition_budget {
                                Some(max) => condition_consecutive_non_pruning < max,
                                None => true,
                            };
                        if !condition_active
                            || transitions.token_boundary_allowed(
                                &mut parser_cache,
                                state.constraint,
                                initial_lexer_state,
                                target,
                                scalar_parser,
                            )
                        {
                            scalar_lexer = target;
                            if condition_active {
                                condition_consecutive_non_pruning =
                                    condition_consecutive_non_pruning.saturating_add(1);
                            }
                        } else {
                            scalar_lexer = FULL_WALK_LEXER_DEAD;
                            if condition_dead_skip {
                                let pruned = if deferred_output {
                                    let (child, count) = full_walk_skip_dead_subtree_generic_deferred(
                                        vocab,
                                        trie,
                                        walk_ops,
                                        &mut remaining_ops,
                                        &mut token_marker_index,
                                    );
                                    deferred_dead_subtrees.push(child);
                                    deferred_negative_work += count;
                                    count
                                } else if sparse_positive_rebuild {
                                    let (_, count) = full_walk_skip_dead_subtree_generic_deferred(
                                        vocab,
                                        trie,
                                        walk_ops,
                                        &mut remaining_ops,
                                        &mut token_marker_index,
                                    );
                                    count
                                } else {
                                    full_walk_skip_dead_subtree_generic(
                                        vocab,
                                        trie,
                                        walk_ops,
                                        &mut remaining_ops,
                                        &mut token_marker_index,
                                        buf,
                                    )
                                };
                                if pruned > 0 {
                                    condition_consecutive_non_pruning = 0;
                                } else {
                                    condition_consecutive_non_pruning =
                                        condition_consecutive_non_pruning.saturating_add(1);
                                }
                                continue;
                            }
                        }
                    } else {
                        parser_transparent = false;
                        let direct_applied = HOT_SINGLE_ROOT
                            && full_walk_try_apply_plain_single_finalizer(
                                target,
                                scalar_parser,
                                initial_lexer_state,
                                transitions,
                                &mut parser_cache,
                                state.constraint,
                                FULL_WALK_LEXER_TWO_DISTINCT,
                                &mut scalar_lexer,
                                &mut scalar_parser,
                                &mut current_two,
                            );
                        if !direct_applied {
                            let outcome = if HOT_SINGLE_ROOT {
                                full_walk_scalar_finalizer_hot_single(
                                    target,
                                    scalar_parser,
                                    initial_lexer_state,
                                    transitions,
                                    &mut parser_cache,
                                    state.constraint,
                                )
                            } else {
                                full_walk_scalar_finalizer(
                                    target,
                                    scalar_parser,
                                    initial_lexer_state,
                                    transitions,
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
                                            transitions,
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
                                    scalar_lexer = FULL_WALK_LEXER_MULTI;
                                    current_many = full_walk_many_state_from_branches(next);
                                }
                            }
                            FullWalkScalarFinalizerOutcome::Many(next) => {
                                if let [first, second] = next.as_slice() {
                                    if first.prune_guard.is_passed() && second.prune_guard.is_passed() {
                                        if let Some((lexer_state, parser_node)) =
                                            full_walk_merge_two_same_parser(
                                                transitions,
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
                                        scalar_lexer = FULL_WALK_LEXER_MULTI;
                                        current_many = full_walk_many_state_from_branches(next);
                                    }
                                } else if let Some((lexer_state, parser_node)) =
                                    full_walk_merge_branches_same_parser(
                                        transitions,
                                        &mut pair_union_cache,
                                        &mut triple_union_cache,
                                        &next,
                                    )
                                {
                                    scalar_lexer = lexer_state;
                                    scalar_parser = parser_node;
                                } else {
                                    scalar_lexer = FULL_WALK_LEXER_MULTI;
                                    current_many = full_walk_many_state_from_branches(next);
                                }
                            }
                            }
                        }
                    }
                }
                }
            } else if scalar_lexer == FULL_WALK_LEXER_TWO_DISTINCT {
                // The two branches carry distinct parser contexts, so lexical
                // survival alone is not enough: parser-conditioned future
                // liveness can kill one branch before either lexer cell dies or
                // finalizes. Always use the exact two-branch step here. The old
                // lexical-only shortcut kept parser-dead branches alive across
                // almost the whole vocabulary on common JSON states.
                match full_walk_step_two::<T>(
                    current_two,
                    byte,
                    initial_lexer_state,
                    transitions,
                    &mut parser_cache,
                    state.constraint,
                ) {
                        FullWalkTwoStepOutcome::Dead => {
                            scalar_lexer = FULL_WALK_LEXER_DEAD;
                            let cleared = if deferred_output {
                                let (child, count) = full_walk_skip_dead_subtree_generic_deferred(
                                    vocab, trie, walk_ops, &mut remaining_ops,
                                    &mut token_marker_index,
                                );
                                deferred_dead_subtrees.push(child);
                                deferred_negative_work += count;
                                count
                            } else {
                                full_walk_skip_dead_subtree_generic(
                                    vocab, trie, walk_ops, &mut remaining_ops,
                                    &mut token_marker_index, buf,
                                )
                            };
                            if profile_generic_work {
                                profile_dead_subtree_original_tokens += cleared;
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
                                    transitions,
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
                                            transitions,
                                            &mut pair_union_cache,
                                            &mut triple_union_cache,
                                            &next,
                                        )
                            {
                                scalar_lexer = lexer_state;
                                scalar_parser = parser_node;
                            } else {
                                scalar_lexer = FULL_WALK_LEXER_MULTI;
                                current_many = full_walk_many_state_from_branches(next);
                            }
                        }
                }
            } else if scalar_lexer == FULL_WALK_LEXER_TWO {
                match full_walk_step_two::<T>(
                        current_two,
                        byte,
                        initial_lexer_state,
                        transitions,
                        &mut parser_cache,
                        state.constraint,
                    ) {
                        FullWalkTwoStepOutcome::Dead => {
                            scalar_lexer = FULL_WALK_LEXER_DEAD;
                            let cleared = if deferred_output {
                                let (child, count) = full_walk_skip_dead_subtree_generic_deferred(
                                    vocab, trie, walk_ops, &mut remaining_ops,
                                    &mut token_marker_index,
                                );
                                deferred_dead_subtrees.push(child);
                                deferred_negative_work += count;
                                count
                            } else {
                                full_walk_skip_dead_subtree_generic(
                                    vocab, trie, walk_ops, &mut remaining_ops,
                                    &mut token_marker_index, buf,
                                )
                            };
                            if profile_generic_work {
                                profile_dead_subtree_original_tokens += cleared;
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
                                    transitions,
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
                                            transitions,
                                            &mut pair_union_cache,
                                            &mut triple_union_cache,
                                            &next,
                                        )
                            {
                                scalar_lexer = lexer_state;
                                scalar_parser = parser_node;
                            } else {
                                scalar_lexer = FULL_WALK_LEXER_MULTI;
                                current_many = full_walk_many_state_from_branches(next);
                            }
                        }
                    }
            } else if scalar_lexer == FULL_WALK_LEXER_MULTI {
                if let Some(next_id) = (product_transition_cache_capacity != 0)
                    .then(|| many_transition_cache.as_mut().unwrap().cached_many_target(current_many_id, byte))
                    .flatten()
                {
                    current_many_id = next_id;
                } else {
                let mut owned_next = if product_transition_cache_capacity == 0 {
                    full_walk_step_many_state(
                        &current_many,
                        byte,
                        initial_lexer_state,
                        transitions,
                        &mut parser_cache,
                        state.constraint,
                    )
                } else { FullWalkManyState::Branches(FullWalkBranches::new()) };
                let next_id = if product_transition_cache_capacity != 0 {
                    many_transition_cache.as_mut().unwrap().step_handle(
                        current_many_id,
                        &current_many,
                        byte,
                        initial_lexer_state,
                        transitions,
                        &mut parser_cache,
                        state.constraint,
                        &mut owned_next,
                        condition_walk_enabled,
                    )
                } else { FullWalkManyTransitionCache::UNKNOWN };
                let next = if next_id < FullWalkManyTransitionCache::DEAD {
                    &many_transition_cache.as_ref().unwrap().states[next_id as usize]
                } else { &owned_next };
                match next {
                    FullWalkManyState::Branches(next) => {
                        match next.as_slice() {
                            [] => {
                                scalar_lexer = FULL_WALK_LEXER_DEAD;
                                let cleared = if deferred_output {
                                    let (child, count) =
                                        full_walk_skip_dead_subtree_generic_deferred(
                                            vocab, trie, walk_ops, &mut remaining_ops,
                                            &mut token_marker_index,
                                        );
                                    deferred_dead_subtrees.push(child);
                                    deferred_negative_work += count;
                                    count
                                } else {
                                    full_walk_skip_dead_subtree_generic(
                                        vocab, trie, walk_ops, &mut remaining_ops,
                                        &mut token_marker_index, buf,
                                    )
                                };
                                if profile_generic_work {
                                    profile_dead_subtree_original_tokens += cleared;
                                }
                                continue;
                            }
                            [branch] if branch.prune_guard.is_passed() => {
                                if cache_single_branch_products && next_id < FullWalkManyTransitionCache::DEAD {
                                    scalar_lexer = FULL_WALK_LEXER_MULTI;
                                    current_many_id = next_id;
                                } else {
                                    scalar_lexer = branch.lexer_state;
                                    scalar_parser = branch.parser_node;
                                }
                            }
                            [first, second]
                                if first.prune_guard.is_passed() && second.prune_guard.is_passed() =>
                            {
                                if cache_two_branch_products && next_id < FullWalkManyTransitionCache::DEAD {
                                    scalar_lexer = FULL_WALK_LEXER_MULTI;
                                    current_many_id = next_id;
                                } else if let Some((lexer_state, parser_node)) =
                                    full_walk_merge_two_same_parser(
                                        transitions,
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
                                if let Some((lexer_state, parser_node)) =
                                    full_walk_merge_branches_same_parser(
                                        transitions,
                                        &mut pair_union_cache,
                                        &mut triple_union_cache,
                                        &next,
                                    )
                                {
                                    scalar_lexer = lexer_state;
                                    scalar_parser = parser_node;
                                } else {
                                    scalar_lexer = FULL_WALK_LEXER_MULTI;
                                }
                            }
                        }
                    }
                    FullWalkManyState::ThreeSameParser { lexers, parser_node } => {
                        if let Some((lexer_state, parser_node)) =
                            full_walk_merge_three_same_parser(
                                transitions,
                                &mut triple_union_cache,
                                *lexers,
                                *parser_node,
                            )
                        {
                            scalar_lexer = lexer_state;
                            scalar_parser = parser_node;
                        } else {
                            scalar_lexer = FULL_WALK_LEXER_MULTI;
                        }
                    }
                }
                if scalar_lexer == FULL_WALK_LEXER_MULTI {
                    current_many_id = next_id;
                    if next_id == FullWalkManyTransitionCache::UNKNOWN {
                        current_many = owned_next;
                    }
                }
                }
            }
        }

        if op.ends_edge() {
            if cache_single_branch_products && scalar_lexer == FULL_WALK_LEXER_MULTI
                && current_many_id < FullWalkManyTransitionCache::DEAD
                // Products with no known self-loop cannot prove a nonempty
                // suffix alphabet. Avoid loading trie/edge/byte-set metadata
                // at every endpoint for those changing products.
                && many_transition_cache.as_ref().unwrap().has_self_loop[current_many_id as usize]
                && first_match_direct.is_none()
            {
                let op_index = walk_ops.len() - remaining_ops.as_slice().len() - 1;
                let (child, _) = trie.full_walk_dead_subtree(op_index);
                // At END the whole compressed edge is consumed, so the cached
                // subtree alphabet covers exactly the remaining suffix bytes.
                if trie.node(child).child_len != 0
                    && many_transition_cache.as_mut().unwrap().admits_stable_alphabet(
                        current_many_id, trie.subtree_bytes(child), initial_lexer_state,
                        transitions, &mut parser_cache, state.constraint,
                    )
                {
                    let original_count = vocab.subtree_original_tokens_for(trie, child).len();
                    if deferred_output {
                        deferred_positive_work += original_count;
                        deferred_admitted_subtrees.push(child);
                    }
                    if profile_generic_work {
                        profile_admitted_subtree_original_tokens += original_count;
                    }
                    full_walk_skip_admitted_subtree_generic(
                        trie, walk_ops, &mut remaining_ops, &mut token_marker_index,
                    );
                    continue;
                }
            }
            if op.child_is_token() {
                if profile_generic_work {
                    profile_token_endpoints += 1;
                }
                let token_marker = unsafe { *token_markers.get_unchecked(token_marker_index) };
                token_marker_index += 1;
                let allowed = if let Some(coordinate) = first_match_direct {
                    if profile_generic_work {
                        profile_first_match_direct_endpoints += 1;
                    }
                    debug_assert!(first_match_terminal.is_some());
                    let direct_allowed = if let Some(&cached) = first_match_direct_future_cache.get(&coordinate) {
                        cached
                    } else {
                        let future = coordinate
                            .has_future(&state.constraint.tokenizer)
                            .unwrap_or(false);
                        first_match_direct_future_cache.insert(coordinate, future);
                        future
                    };
                    let side_allowed = defer_first_match_finalizer
                        && first_match_side.as_ref().is_some_and(|side| match side {
                            FullWalkManyState::Branches(branches) => branches.iter().any(|branch| {
                                transitions.token_boundary_allowed(
                                    &mut parser_cache,
                                    state.constraint,
                                    initial_lexer_state,
                                    branch.lexer_state,
                                    branch.parser_node,
                                )
                            }),
                            FullWalkManyState::ThreeSameParser {
                                lexers,
                                parser_node,
                            } => transitions.token_boundary_allowed(
                                &mut parser_cache,
                                state.constraint,
                                initial_lexer_state,
                                lexers.0,
                                *parser_node,
                            ) || transitions.token_boundary_allowed(
                                &mut parser_cache,
                                state.constraint,
                                initial_lexer_state,
                                lexers.1,
                                *parser_node,
                            ) || transitions.token_boundary_allowed(
                                &mut parser_cache,
                                state.constraint,
                                initial_lexer_state,
                                lexers.2,
                                *parser_node,
                            ),
                        });
                    direct_allowed || side_allowed
                } else if scalar_lexer == FULL_WALK_LEXER_DEAD {
                    false
                } else if scalar_lexer < FULL_WALK_LEXER_TWO_DISTINCT {
                    parser_transparent || transitions.token_boundary_allowed(
                            &mut parser_cache,
                            state.constraint,
                            initial_lexer_state,
                            scalar_lexer,
                            scalar_parser,
                        )
                } else if scalar_lexer == FULL_WALK_LEXER_TWO_DISTINCT || scalar_lexer == FULL_WALK_LEXER_TWO {
                    transitions.token_boundary_allowed(
                        &mut parser_cache,
                        state.constraint,
                        initial_lexer_state,
                        current_two.0.0,
                        current_two.0.1,
                    ) || transitions.token_boundary_allowed(
                        &mut parser_cache,
                        state.constraint,
                        initial_lexer_state,
                        current_two.1.0,
                        current_two.1.1,
                    )
                } else if scalar_lexer == FULL_WALK_LEXER_MULTI {
                    if product_transition_cache_capacity != 0
                        && current_many_id != FullWalkManyTransitionCache::UNKNOWN
                    {
                        many_transition_cache.as_mut().unwrap().token_boundary_allowed(
                            current_many_id, transitions, &mut parser_cache,
                            state.constraint, initial_lexer_state,
                        )
                    } else { match &current_many {
                        FullWalkManyState::Branches(branches) => branches.iter().any(|branch| {
                            transitions.token_boundary_allowed(
                                &mut parser_cache,
                                state.constraint,
                                initial_lexer_state,
                                branch.lexer_state,
                                branch.parser_node,
                            )
                        }),
                        FullWalkManyState::ThreeSameParser {
                            lexers,
                            parser_node,
                        } => transitions.token_boundary_allowed(
                            &mut parser_cache,
                            state.constraint,
                            initial_lexer_state,
                            lexers.0,
                            *parser_node,
                        ) || transitions.token_boundary_allowed(
                            &mut parser_cache,
                            state.constraint,
                            initial_lexer_state,
                            lexers.1,
                            *parser_node,
                        ) || transitions.token_boundary_allowed(
                            &mut parser_cache,
                            state.constraint,
                            initial_lexer_state,
                            lexers.2,
                            *parser_node,
                        ),
                    } }
                } else {
                    false
                };
                if allowed {
                    if profile_generic_work {
                        profile_allowed_token_endpoints += 1;
                    }
                    if deferred_output {
                        deferred_positive_work += dynamic_token_marker_original_count(vocab, token_marker);
                        deferred_allowed_markers.push(token_marker);
                    } else if sparse_positive_rebuild {
                        mark_dynamic_token_marker(vocab, token_marker, buf);
                    }
                } else {
                    if profile_generic_work {
                        profile_blocked_token_endpoints += 1;
                        profile_blocked_dead_endpoints +=
                            usize::from(scalar_lexer == FULL_WALK_LEXER_DEAD);
                        if token_marker & DYNAMIC_TOKEN_MARKER_FALLBACK == 0 {
                            profile_blocked_original_tokens += (token_marker as u32).count_ones() as usize;
                        } else {
                            let canonical_token =
                                ((token_marker & !DYNAMIC_TOKEN_MARKER_FALLBACK) - 1) as u32;
                            profile_blocked_original_tokens += vocab
                                .token_word_masks(canonical_token)
                                .iter()
                                .map(|&(_, bits)| bits.count_ones() as usize)
                                .sum::<usize>();
                        }
                    }
                    if deferred_output {
                        deferred_negative_work += dynamic_token_marker_original_count(vocab, token_marker);
                        deferred_rejected_markers.push(token_marker);
                    } else if !sparse_positive_rebuild {
                        clear_dynamic_token_marker(vocab, token_marker, buf);
                    }
                }
            }

            unsafe {
                *stack_first_match_direct.get_unchecked_mut(parent_depth + 1) =
                    first_match_direct;
                *stack_lexer.get_unchecked_mut(parent_depth + 1) = scalar_lexer;
                if first_match_direct.is_some() {
                    if defer_first_match_finalizer {
                        *stack_first_match_accepting.get_unchecked_mut(parent_depth + 1) =
                            first_match_direct_accepting;
                        let side_slot =
                            stack_first_match_side.get_unchecked_mut(parent_depth + 1);
                        if let Some(side) = first_match_side.as_ref() {
                            if let Some(existing) = side_slot.as_mut() {
                                existing.clone_from(side);
                            } else {
                                *side_slot = Some(side.clone());
                            }
                        } else {
                            *side_slot = None;
                        }
                    }
                    *stack_parser.get_unchecked_mut(parent_depth + 1) = scalar_parser;
                } else if scalar_lexer == FULL_WALK_LEXER_DEAD {
                    if defer_first_match_finalizer {
                        *stack_first_match_accepting.get_unchecked_mut(parent_depth + 1) = false;
                        *stack_first_match_side.get_unchecked_mut(parent_depth + 1) = None;
                    }
                } else if scalar_lexer < FULL_WALK_LEXER_TWO_DISTINCT {
                    if defer_first_match_finalizer {
                        *stack_first_match_accepting.get_unchecked_mut(parent_depth + 1) = false;
                        *stack_first_match_side.get_unchecked_mut(parent_depth + 1) = None;
                    }
                    *stack_parser.get_unchecked_mut(parent_depth + 1) = scalar_parser;
                } else if scalar_lexer == FULL_WALK_LEXER_TWO_DISTINCT || scalar_lexer == FULL_WALK_LEXER_TWO {
                    if defer_first_match_finalizer {
                        *stack_first_match_accepting.get_unchecked_mut(parent_depth + 1) = false;
                        *stack_first_match_side.get_unchecked_mut(parent_depth + 1) = None;
                    }
                    *stack_two.get_unchecked_mut(parent_depth + 1) = current_two;
                } else if scalar_lexer == FULL_WALK_LEXER_MULTI {
                    if defer_first_match_finalizer {
                        *stack_first_match_accepting.get_unchecked_mut(parent_depth + 1) = false;
                        *stack_first_match_side.get_unchecked_mut(parent_depth + 1) = None;
                    }
                    if product_transition_cache_capacity != 0 {
                        *stack_many_ids.get_unchecked_mut(parent_depth + 1) = current_many_id;
                    }
                    if product_transition_cache_capacity == 0
                        || current_many_id == FullWalkManyTransitionCache::UNKNOWN
                    {
                        let slot = full_walk_owned_many_slot(
                            &mut stack_many, parent_depth + 1, stack_len,
                        );
                        if let Some(existing) = slot.as_mut() {
                            existing.clone_from(&current_many);
                        } else {
                            *slot = Some(current_many.clone());
                        }
                    }
                }
            }
        }
    }
    if deferred_output {
        if deferred_positive_work <= deferred_negative_work {
            buf.fill(0);
            for marker in deferred_allowed_markers {
                mark_dynamic_token_marker(vocab, marker, buf);
            }
            for child in deferred_admitted_subtrees {
                for &canonical in trie.subtree_tokens(child) {
                    for &(word, bits) in vocab.token_word_masks(canonical) {
                        if let Some(slot) = buf.get_mut(word as usize) { *slot |= bits; }
                    }
                }
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
        // Empty-byte aliases live at the root, outside the edge-marker stream.
        // The eager output starts with these set; reproduce that exact root
        // contribution when the positive adaptive emitter starts from zero.
        if let Some(canonical) = trie.node(0).token_id {
            for &(word, bits) in vocab.token_word_masks(canonical) {
                if let Some(slot) = buf.get_mut(word as usize) { *slot |= bits; }
            }
        }
        if profile_generic_work {
            eprintln!(
                "[glrmask/profile][generic_adaptive_output] generation={} positive_work={} negative_work={}",
                state.generation,
                deferred_positive_work,
                deferred_negative_work,
            );
        }
    }
    if profile_generic_work {
        eprintln!(
            "[glrmask/profile][generic_walk_work] byte_ops={} scalar_byte_ops={} two_byte_ops={} multi_byte_ops={} dead_byte_ops={} token_endpoints={} allowed_endpoints={} blocked_endpoints={} blocked_dead_endpoints={} blocked_original_tokens={} dead_subtree_original_tokens={} admitted_subtree_original_tokens={} boundary_calls={} boundary_hits={} boundary_misses={} admitted_builds={} parser_nodes={} first_match_direct_bytes={} first_match_direct_finalizers={} first_match_direct_endpoints={}",
            profile_byte_ops,
            profile_scalar_byte_ops,
            profile_two_byte_ops,
            profile_multi_byte_ops,
            profile_dead_byte_ops,
            profile_token_endpoints,
            profile_allowed_token_endpoints,
            profile_blocked_token_endpoints,
            profile_blocked_dead_endpoints,
            profile_blocked_original_tokens,
            profile_dead_subtree_original_tokens,
            profile_admitted_subtree_original_tokens,
            parser_cache.profile_boundary_calls,
            parser_cache.profile_boundary_hits,
            parser_cache.profile_boundary_misses,
            parser_cache.profile_admitted_builds,
            parser_cache.nodes.len(),
            profile_first_match_direct_byte_ops,
            profile_first_match_direct_finalizers,
            profile_first_match_direct_endpoints,
        );
    }
    // Ordinary vocabulary bytes and exact special-token-ID paths are a union.
    // The strict walk above computes the byte-language contribution for every
    // model token; the existing special-token routine then ORs in token-ID-only
    // paths. This also handles a token ID that is valid through both routes.
    update_special_token_mask(state, buf);
    state.clear_late_grammar_placeholder_mask(buf);
    #[cfg(test)]
    TEST_FULL_WALK_USES.with(|count| count.set(count.get() + 1));
    Ok(true)
}

#[inline(always)]
fn full_walk_skip_admitted_subtree_generic<'a>(
    trie: &DynamicMaskTrie,
    walk_ops: &'a [DynamicMaskTrieFullWalkOp],
    remaining_ops: &mut std::slice::Iter<'a, DynamicMaskTrieFullWalkOp>,
    token_marker_index: &mut usize,
) -> usize {
    let op_index = walk_ops.len() - remaining_ops.as_slice().len() - 1;
    let (child, subtree_end_op) = trie.full_walk_dead_subtree(op_index);
    let token_count = trie.subtree_tokens(child).len();
    let root_token_offset = usize::from(trie.node(0).token_id.is_some());
    let token_end = trie
        .subtree_token_index_range(child)
        .end
        .saturating_sub(root_token_offset);
    debug_assert!(*token_marker_index <= token_end);
    *token_marker_index = token_end;
    *remaining_ops = walk_ops[subtree_end_op as usize..].iter();
    token_count
}

#[inline(always)]
fn full_walk_skip_dead_subtree_generic<'a>(
    vocab: &DynamicMaskVocab,
    trie: &DynamicMaskTrie,
    walk_ops: &'a [DynamicMaskTrieFullWalkOp],
    remaining_ops: &mut std::slice::Iter<'a, DynamicMaskTrieFullWalkOp>,
    token_marker_index: &mut usize,
    buf: &mut [u32],
) -> usize {
    let op_index = walk_ops.len() - remaining_ops.as_slice().len() - 1;
    let (child, subtree_end_op) = trie.full_walk_dead_subtree(op_index);
    let tokens = vocab.subtree_original_tokens_for(trie, child);
    for &token_id in tokens {
        clear_mask_bit_known_in_range(buf, token_id);
    }
    let root_token_offset = usize::from(trie.node(0).token_id.is_some());
    let token_end = trie
        .subtree_token_index_range(child)
        .end
        .saturating_sub(root_token_offset);
    debug_assert!(*token_marker_index <= token_end);
    *token_marker_index = token_end;
    *remaining_ops = walk_ops[subtree_end_op as usize..].iter();
    tokens.len()
}

#[inline(always)]
fn full_walk_skip_dead_subtree_generic_deferred<'a>(
    vocab: &DynamicMaskVocab,
    trie: &DynamicMaskTrie,
    walk_ops: &'a [DynamicMaskTrieFullWalkOp],
    remaining_ops: &mut std::slice::Iter<'a, DynamicMaskTrieFullWalkOp>,
    token_marker_index: &mut usize,
) -> (u32, usize) {
    let op_index = walk_ops.len() - remaining_ops.as_slice().len() - 1;
    let (child, subtree_end_op) = trie.full_walk_dead_subtree(op_index);
    let token_count = vocab.subtree_original_tokens_for(trie, child).len();
    let root_token_offset = usize::from(trie.node(0).token_id.is_some());
    let token_end = trie
        .subtree_token_index_range(child)
        .end
        .saturating_sub(root_token_offset);
    debug_assert!(*token_marker_index <= token_end);
    *token_marker_index = token_end;
    *remaining_ops = walk_ops[subtree_end_op as usize..].iter();
    (child, token_count)
}

#[derive(Clone)]
struct DynamicBranch {
    tokenizer_config: u32,
    /// Exact source-tokenizer state when this branch came from one real root.
    /// Synthetic projected/subset roots leave this unavailable and decline
    /// regex-partition containment.
    exact_tokenizer_state: Option<u32>,
    /// True only when a root was restricted to exactly parser-admitted
    /// terminals (not merely a row-presence superset). Such a root may use the
    /// existing parser-conditioned scalar hot lane.
    parser_filtered_root: bool,
    /// Stronger proof: every matched/future terminal carried by the filtered
    /// runtime config is parser-admitted. Until the first terminal finalizes,
    /// parser-boundary checks are therefore redundant.
    parser_filtered_transparent: bool,
    /// A matched terminal whose finalize-now parser alternative is represented
    /// by an exact initial-lexer sibling root in this same committed state.
    /// When present, this branch is the lexer-continuation alternative and may
    /// schedule vocabulary root bytes from the retained terminal residual.
    residual_continuation_terminal: Option<TerminalID>,
    gss: ParserStacks,
    shared_root_admission: SharedRootAdmission,
    initial_prune_guard: InitialPruneGuard,
}

type DynamicBranches = SmallVec<[DynamicBranch; 4]>;

#[derive(Clone, PartialEq, Eq, Hash)]
enum InitialPruneGuard {
    Passed,
    Pending {
        /// `(mask-runtime state, exact source state, terminal)`. Traversal uses
        /// the first coordinate; immutable possible-match lookup uses the
        /// second so it can hit the source-tokenizer precomputed index.
        memories: Arc<[(u32, u32, TerminalID)]>,
    },
}

#[inline]
fn set_mask_bit(buf: &mut [u32], token_id: u32) {
    let word = token_id as usize / 32;
    let bit = token_id % 32;
    if let Some(slot) = buf.get_mut(word) {
        *slot |= 1u32 << bit;
    }
}

#[inline]
fn set_mask_bit_known_in_range(buf: &mut [u32], token_id: u32) {
    let word = token_id as usize / 32;
    let bit = token_id % 32;
    debug_assert!(word < buf.len());
    // Dynamic vocabulary ids come from the same Vocab used to size the mask.
    // Avoid a bounds branch for every accepted token in large subtree marks.
    unsafe {
        *buf.get_unchecked_mut(word) |= 1u32 << bit;
    }
}

#[inline(always)]
fn clear_mask_bit_known_in_range(buf: &mut [u32], token_id: u32) {
    let word = token_id as usize / 32;
    let bit = token_id % 32;
    debug_assert!(word < buf.len());
    unsafe { *buf.get_unchecked_mut(word) &= !(1u32 << bit); }
}


const DYNAMIC_TOKEN_MARKER_FALLBACK: u64 = 1u64 << 63;

#[inline(always)]
fn dynamic_token_marker_original_count(vocab: &DynamicMaskVocab, marker: u64) -> usize {
    if marker & DYNAMIC_TOKEN_MARKER_FALLBACK == 0 {
        (marker as u32).count_ones() as usize
    } else {
        let canonical_token = ((marker & !DYNAMIC_TOKEN_MARKER_FALLBACK) - 1) as u32;
        vocab
            .token_word_masks(canonical_token)
            .iter()
            .map(|&(_, bits)| bits.count_ones() as usize)
            .sum()
    }
}

#[inline(always)]
fn mark_dynamic_token_marker(vocab: &DynamicMaskVocab, marker: u64, buf: &mut [u32]) {
    debug_assert_ne!(marker, 0);
    if marker & DYNAMIC_TOKEN_MARKER_FALLBACK == 0 {
        let word = (marker >> 32) as usize;
        let bits = marker as u32;
        debug_assert_ne!(bits, 0);
        debug_assert!(word < buf.len());
        unsafe {
            *buf.get_unchecked_mut(word) |= bits;
        }
        return;
    }

    let canonical_token = ((marker & !DYNAMIC_TOKEN_MARKER_FALLBACK) - 1) as u32;
    for &(word, bits) in vocab.token_word_masks(canonical_token) {
        debug_assert!((word as usize) < buf.len());
        unsafe { *buf.get_unchecked_mut(word as usize) |= bits; }
    }
}

#[inline(always)]
fn clear_dynamic_token_marker(vocab: &DynamicMaskVocab, marker: u64, buf: &mut [u32]) {
    debug_assert_ne!(marker, 0);
    if marker & DYNAMIC_TOKEN_MARKER_FALLBACK == 0 {
        let word = (marker >> 32) as usize;
        let bits = marker as u32;
        debug_assert_ne!(bits, 0);
        debug_assert!(word < buf.len());
        unsafe { *buf.get_unchecked_mut(word) &= !bits; }
        return;
    }
    let canonical_token = ((marker & !DYNAMIC_TOKEN_MARKER_FALLBACK) - 1) as u32;
    for &(word, bits) in vocab.token_word_masks(canonical_token) {
        debug_assert!((word as usize) < buf.len());
        unsafe { *buf.get_unchecked_mut(word as usize) &= !bits; }
    }
}


const DYNAMIC_NFA_CONFIG_UNKNOWN: u32 = u32::MAX;
const DYNAMIC_NFA_CONFIG_DEAD: u32 = u32::MAX - 1;
// In a tokenizer that contains epsilon edges somewhere, most runtime states can
// still be ordinary scalar DFA states with no outgoing epsilon transition. Do
// not force those states through the generic NFA-config interner. The high bit
// tags a raw tokenizer state; interned multi-state configs remain dense low
// integers. Tokenizer state counts are already u32-sized and in practice many
// orders of magnitude below this boundary; fail closed if that ever changes.
const DYNAMIC_NFA_RAW_CONFIG_TAG: u32 = 1 << 31;
// The strict full walker uses the four top u32 values as branch-kind/dead
// sentinels. Tagged raw NFA configs share that high half of the u32 namespace,
// so keep those four values permanently outside the raw-config domain. This is
// also stricter than the old cache contract, which already could not represent
// raw states whose tagged IDs became UNKNOWN or DEAD.
const DYNAMIC_NFA_RAW_STATE_LIMIT: u32 = DYNAMIC_NFA_RAW_CONFIG_TAG - 4;

#[derive(Clone)]
struct DynamicNfaScanCache<'a> {
    constraint: &'a Constraint,
    tokenizer: &'a Tokenizer,
    use_constraint_fast_transitions: bool,
    deterministic: bool,
    subset_union_requested: bool,
    deadline: Option<Instant>,
    max_collection_items: Option<usize>,
    config_ids: FxHashMap<Vec<u32>, u32>,
    configs: Vec<Box<[u32]>>,
    // A restricted config denotes only paths ending at a terminal in its
    // admitted set. The set is part of the hash-cons key: equal physical
    // states with different admitted languages MUST NOT alias.
    parser_projection_enabled: bool,
    admitted_sets: Vec<BitSet>,
    admitted_set_ids: FxHashMap<Vec<u64>, u32>,
    admitted_config_ids: FxHashMap<(u32, Vec<u32>), u32>,
    config_admitted: Vec<Option<u32>>,
    fresh_reset_ids: FxHashMap<u32, u32>,
    config_is_fresh_reset: Vec<bool>,
    transitions: Vec<Option<Box<[u32; 256]>>>,
    residual_configs: Vec<u32>,
    // Canonical union metadata for interned multi-state configs. These make a
    // cached lazy-subset transition genuinely O(1): the strict vocabulary walk
    // must not rescan every member merely to discover finalizer/future bits.
    config_matched: Vec<BitSet>,
    config_futures: Vec<BitSet>,
    raw_start_config: FxHashMap<u32, u32>,
    profile_transition_work: bool,
    profile_step_calls: usize,
    profile_step_cache_hits: usize,
    profile_step_cache_misses: usize,
    profile_physical_states_scanned: usize,
    profile_intern_calls: usize,
    profile_intern_hits: usize,
    profile_intern_new: usize,
    profile_member_physical_transition_ns: u64,
    profile_member_virtual_residual_transition_ns: u64,
    profile_member_physical_transitions: usize,
    profile_member_virtual_residual_transitions: usize,
    profile_member_virtual_residual_by_terminal: FxHashMap<TerminalID, (usize, u64)>,
}

struct DynamicConfigExecResult {
    end_config: Option<u32>,
    matches: Vec<TokenizerMatch>,
}

fn dynamic_use_cached_topology() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("GLRMASK_DYNAMIC_USE_CACHED_TOPOLOGY")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true")))
}

impl<'a> DynamicNfaScanCache<'a> {
    #[inline]
    fn raw_config(state: u32) -> u32 {
        assert!(
            state < DYNAMIC_NFA_RAW_STATE_LIMIT,
            "dynamic lexer state exceeds representable raw-config coordinate"
        );
        DYNAMIC_NFA_RAW_CONFIG_TAG | state
    }

    #[inline]
    fn raw_config_state(config: u32) -> Option<u32> {
        (config & DYNAMIC_NFA_RAW_CONFIG_TAG != 0
            && config != DYNAMIC_NFA_CONFIG_UNKNOWN
            && config != DYNAMIC_NFA_CONFIG_DEAD)
            .then_some(config & !DYNAMIC_NFA_RAW_CONFIG_TAG)
    }

    fn new(constraint: &'a Constraint, deadline: Option<Instant>) -> Self {
        Self::new_with_tokenizer(constraint, &constraint.tokenizer, deadline, true)
    }

    fn new_for_mask(
        constraint: &'a Constraint,
        vocab: &'a DynamicMaskVocab,
        deadline: Option<Instant>,
    ) -> Self {
        let tokenizer = vocab
            .mask_runtime_tokenizer()
            .unwrap_or(&constraint.tokenizer);
        let use_constraint_fast_transitions = std::ptr::eq(tokenizer, constraint.tokenizer.as_ref());
        Self::new_with_tokenizer(
            constraint,
            tokenizer,
            deadline,
            use_constraint_fast_transitions,
        )
    }

    fn new_with_tokenizer(
        constraint: &'a Constraint,
        tokenizer: &'a Tokenizer,
        deadline: Option<Instant>,
        use_constraint_fast_transitions: bool,
    ) -> Self {
        Self {
            constraint,
            tokenizer,
            use_constraint_fast_transitions,
            deterministic: !if use_constraint_fast_transitions && dynamic_use_cached_topology() {
                // This is the exact same tokenizer as the immutable
                // constraint's fast-transition view. Compile/load already
                // computed its topology; rescanning every row for each
                // short-lived scan cache is unnecessary.
                constraint.tokenizer_has_epsilon_transitions
            } else {
                // A projection tokenizer can differ from the constraint's
                // source. Never use the source flag for that coordinate.
                tokenizer.has_epsilon_transitions()
            },
            subset_union_requested: false,
            deadline,
            max_collection_items: deadline.map(|_| 5_000_000),
            config_ids: FxHashMap::default(),
            configs: Vec::new(),
            parser_projection_enabled: std::env::var_os("GLRMASK_DISABLE_PARSER_LEXER_PROJECTION").is_none(),
            admitted_sets: Vec::new(),
            admitted_set_ids: FxHashMap::default(),
            admitted_config_ids: FxHashMap::default(),
            config_admitted: Vec::new(),
            fresh_reset_ids: FxHashMap::default(),
            config_is_fresh_reset: Vec::new(),
            transitions: Vec::new(),
            residual_configs: Vec::new(),
            config_matched: Vec::new(),
            config_futures: Vec::new(),
            raw_start_config: FxHashMap::default(),
            profile_transition_work: std::env::var_os("GLRMASK_PROFILE_DYNAMIC_CONFIG_TRANSITIONS").is_some(),
            profile_step_calls: 0,
            profile_step_cache_hits: 0,
            profile_step_cache_misses: 0,
            profile_physical_states_scanned: 0,
            profile_intern_calls: 0,
            profile_intern_hits: 0,
            profile_intern_new: 0,
            profile_member_physical_transition_ns: 0,
            profile_member_virtual_residual_transition_ns: 0,
            profile_member_physical_transitions: 0,
            profile_member_virtual_residual_transitions: 0,
            profile_member_virtual_residual_by_terminal: FxHashMap::default(),
        }
    }

    #[inline]
    fn tokenizer(&self) -> &Tokenizer {
        self.tokenizer
    }

    #[inline]
    fn raw_state_for_config(&self, config: u32) -> Option<u32> {
        Self::raw_config_state(config)
    }

    #[inline]
    fn config_index(&self, config: u32) -> Option<usize> {
        Self::raw_config_state(config).is_none().then_some(config as usize)
    }

    #[inline]
    fn encode_raw_state(&self, state: u32) -> u32 {
        Self::raw_config(state)
    }

    #[inline]
    fn transition(&self, state: u32, byte: u8) -> u32 {
        if self.use_constraint_fast_transitions {
            self.constraint.tokenizer_fast_transitions.transition(
                &self.constraint.tokenizer,
                state,
                byte,
            )
        } else {
            self.tokenizer.get_transition(state, byte)
        }
    }

    fn check_growth(&self, current: usize, additional: usize) -> Result<(), String> {
        if self.deadline.is_some_and(|deadline| Instant::now() >= deadline) {
            return Err("glrmask_dynamic mask generation timed out".to_owned());
        }
        if self.max_collection_items.is_some_and(|limit| {
            current
                .checked_add(additional)
                .is_none_or(|next| next > limit)
        }) {
            return Err("glrmask_dynamic mask generation exceeded its work ceiling".to_owned());
        }
        Ok(())
    }

    fn intern_config(&mut self, mut states: Vec<u32>) -> Result<u32, String> {
        if self.profile_transition_work {
            self.profile_intern_calls += 1;
        }
        self.check_growth(0, states.len())?;
        states.sort_unstable();
        states.dedup();
        if let [state] = states.as_slice()
            && !self.tokenizer.state_has_epsilon_transitions(*state)
        {
            return Ok(self.encode_raw_state(*state));
        }
        if let Some(&id) = self.config_ids.get(states.as_slice()) {
            if self.profile_transition_work {
                self.profile_intern_hits += 1;
            }
            return Ok(id);
        }
        self.check_growth(self.configs.len(), 1)?;
        if self.configs.len() >= DYNAMIC_NFA_RAW_CONFIG_TAG as usize {
            return Err(
                "dynamic lexer configuration-id namespace exhausted below raw-state tag"
                    .to_owned(),
            );
        }
        let local_id = u32::try_from(self.configs.len())
            .map_err(|_| "dynamic lexer configuration-id overflow".to_owned())?;
        let id = local_id;
        let mut matched = BitSet::new(self.tokenizer.num_terminals() as usize);
        let mut futures = BitSet::new(self.tokenizer.num_terminals() as usize);
        for &state in &states {
            matched.union_with(self.tokenizer.matched_terminal_bitset(state));
            futures.union_with(self.tokenizer.possible_future_terminals(state));
        }
        self.config_ids.insert(states.clone(), id);
        self.configs.push(states.into_boxed_slice());
        self.config_admitted.push(None);
        self.config_is_fresh_reset.push(false);
        self.transitions.push(None);
        self.residual_configs.push(DYNAMIC_NFA_CONFIG_UNKNOWN);
        self.config_matched.push(matched);
        self.config_futures.push(futures);
        if self.profile_transition_work {
            self.profile_intern_new += 1;
        }
        Ok(id)
    }

    fn config_for_raw_start(&mut self, state: u32) -> Result<u32, String> {
        if self.deterministic {
            return Ok(state);
        }
        if !self.tokenizer.state_has_epsilon_transitions(state) {
            return Ok(self.encode_raw_state(state));
        }
        if let Some(&cached) = self.raw_start_config.get(&state) {
            return Ok(cached);
        }
        let closure = self.tokenizer.singleton_epsilon_closure(state).into_vec();
        let config = self.intern_config(closure)?;
        self.raw_start_config.insert(state, config);
        Ok(config)
    }

    fn config_for_mask_root(&mut self, state: u32) -> Result<u32, String> {
        self.config_for_raw_start(state)
    }

    /// Restrict a lexer root to the parser-admitted terminal language. The
    /// projection persists through byte transitions, until a terminal finalizes
    /// and its new parser frontier supplies a new root. The disabled path keeps
    /// the historical fresh-root-only physical-state filter for A/B validation.
    fn config_for_parser_admitted(
        &mut self,
        state: u32,
        admitted: &BitSet,
    ) -> Result<Option<u32>, String> {
        if self.parser_projection_enabled && !self.deterministic {
            let filter = if let Some(&id) = self.admitted_set_ids.get(admitted.words()) {
                id
            } else {
                self.check_growth(self.admitted_sets.len(), 1)?;
                let id = self.admitted_sets.len() as u32;
                self.admitted_sets.push(admitted.clone());
                self.admitted_set_ids.insert(admitted.words().to_vec(), id);
                id
            };
            let states = self.tokenizer.singleton_epsilon_closure(state).into_vec();
            let config = self.intern_admitted_config(states, filter)?;
            return Ok((config != DYNAMIC_NFA_CONFIG_DEAD).then_some(config));
        }
        if self.deterministic || !self.tokenizer.state_has_epsilon_transitions(state) {
            return Ok(None);
        }
        let closure = self.tokenizer.singleton_epsilon_closure(state);
        let mut filtered = Vec::<u32>::with_capacity(closure.len());
        for member in closure {
            let matched = self.tokenizer.matched_terminal_bitset(member);
            if !matched.is_disjoint(admitted) {
                filtered.push(member);
                continue;
            }
            if self.tokenizer.possible_future_terminals(member).is_disjoint(admitted) {
                continue;
            }
            // Epsilon-only dispatch states contribute no byte transition after
            // their closure has been expanded; retaining them would only widen
            // config metadata back to unrelated terminals.
            if self.tokenizer.transitions_from(member).next().is_some() {
                filtered.push(member);
            }
        }
        if filtered.is_empty() {
            return Ok(None);
        }
        self.intern_config(filtered).map(Some)
    }

    fn config_terminals_subset_of(&self, config: u32, terminals: &BitSet) -> bool {
        if !self.deterministic && self.raw_state_for_config(config).is_none() {
            let index = self.config_index(config).expect("known config");
            return self.config_matched[index].is_subset(terminals)
                && self.config_futures[index].is_subset(terminals);
        }
        for index in 0..self.config_len(config) {
            let state = self.config_state(config, index);
            if !self.tokenizer.matched_terminal_bitset(state).is_subset(terminals)
                || !self.tokenizer.possible_future_terminals(state).is_subset(terminals)
            {
                return false;
            }
        }
        true
    }

    /// Expand one runtime configuration into its exact epsilon-closed physical
    /// tokenizer states without taking a byte step. This is used only by the
    /// scalar-dispatch bridge; general NFA execution continues to use
    /// the interned configuration namespace directly.
    fn physical_states_for_config(
        &self,
        config: u32,
        out: &mut SmallVec<[u32; 8]>,
    ) -> Result<(), String> {
        out.clear();
        if self.deterministic {
            out.push(config);
            return Ok(());
        }
        if let Some(state) = self.raw_state_for_config(config) {
            out.push(state);
            return Ok(());
        }
        let index = self
            .config_index(config)
            .ok_or_else(|| "unknown dynamic lexer config".to_owned())?;
        let states = self
            .configs
            .get(index)
            .ok_or_else(|| "dynamic lexer config index out of range".to_owned())?;
        out.extend_from_slice(states);
        Ok(())
    }


    /// Lazy right-trimming of a labeled lexer to one admitted terminal set A.
    ///
    /// For a state q, retain precisely the output colors in A. A state with
    /// neither a current A-finalizer nor a future A-finalizer is dead. Deleting
    /// it cannot remove a prefix of a word in an admitted terminal language.
    /// The same A follows every byte transition; parser advancement creates a
    /// separate reset branch with its own A. Maximal-munch memories remain
    /// attached to their original language, independently of that reset.
    ///
    /// This is part of the lexer transition, not a second per-byte oracle.
    /// Both projected configs and their transitions are hash-consed/cached.
    fn intern_admitted_config(
        &mut self,
        mut states: Vec<u32>,
        filter: u32,
    ) -> Result<u32, String> {
        self.check_growth(0, states.len())?;
        let admitted = &self.admitted_sets[filter as usize];
        states.retain(|&member| {
            let matched = self.tokenizer.matched_terminal_bitset(member);
            if !matched.is_disjoint(admitted) {
                return true;
            }
            if self.tokenizer.possible_future_terminals(member).is_disjoint(admitted) {
                return false;
            }
            // The input is epsilon-closed. A pure epsilon dispatcher need not
            // be carried into the next byte transition as well as its closure.
            !self.tokenizer.state_has_epsilon_transitions(member)
                || self.tokenizer.transitions_from(member).next().is_some()
        });
        if states.is_empty() {
            return Ok(DYNAMIC_NFA_CONFIG_DEAD);
        }
        states.sort_unstable();
        states.dedup();
        let key = (filter, states);
        if let Some(&cached) = self.admitted_config_ids.get(&key) {
            return Ok(cached);
        }
        let mut matched = BitSet::new(self.tokenizer.num_terminals() as usize);
        let mut futures = BitSet::new(self.tokenizer.num_terminals() as usize);
        for &member in &key.1 {
            matched.union_with(self.tokenizer.matched_terminal_bitset(member));
            futures.union_with(self.tokenizer.possible_future_terminals(member));
        }
        if matched.is_subset(admitted) && futures.is_subset(admitted) {
            // All reachable outputs are already admitted. Return to the
            // existing raw/hot lane without retaining a redundant product.
            let states = key.1.clone();
            let result = self.intern_config(states)?;
            self.admitted_config_ids.insert(key, result);
            return Ok(result);
        }
        matched.intersect_with(admitted);
        futures.intersect_with(admitted);
        self.check_growth(self.configs.len(), 1)?;
        if self.configs.len() >= DYNAMIC_NFA_RAW_CONFIG_TAG as usize {
            return Err("dynamic lexer projected configuration-id namespace exhausted".to_owned());
        }
        let result = self.configs.len() as u32;
        self.configs.push(key.1.clone().into_boxed_slice());
        self.config_admitted.push(Some(filter));
        self.config_is_fresh_reset.push(false);
        self.transitions.push(None);
        self.residual_configs.push(DYNAMIC_NFA_CONFIG_UNKNOWN);
        self.config_matched.push(matched);
        self.config_futures.push(futures);
        self.admitted_config_ids.insert(key, result);
        Ok(result)
    }

    /// A parser reset accepts the token ending *at this boundary*, even when
    /// no next terminal is admissible. Keep that epsilon acceptance out of the
    /// ordinary projected-config interner: a consumed-byte lexer loop returning
    /// to the same physical state must not acquire boundary acceptance. Every
    /// byte transition from this wrapper lands in a non-fresh configuration.
    fn fresh_reset_config(&mut self, config: u32) -> Result<u32, String> {
        if let Some(&reset) = self.fresh_reset_ids.get(&config) {
            return Ok(reset);
        }
        let (states, matched, futures, admitted) = if let Some(raw) = self.raw_state_for_config(config) {
            (vec![raw].into_boxed_slice(),
             self.tokenizer.matched_terminal_bitset(raw).clone(),
             self.tokenizer.possible_future_terminals(raw).clone(), None)
        } else {
            let index = self.config_index(config).ok_or_else(|| "unknown projected reset config".to_owned())?;
            (self.configs[index].clone(), self.config_matched[index].clone(),
             self.config_futures[index].clone(), self.config_admitted[index])
        };
        self.check_growth(self.configs.len(), 1)?;
        if self.configs.len() >= DYNAMIC_NFA_RAW_CONFIG_TAG as usize {
            return Err("dynamic lexer reset configuration-id namespace exhausted".to_owned());
        }
        let reset = self.configs.len() as u32;
        self.configs.push(states);
        self.config_admitted.push(admitted);
        self.config_is_fresh_reset.push(true);
        self.transitions.push(None);
        self.residual_configs.push(DYNAMIC_NFA_CONFIG_UNKNOWN);
        self.config_matched.push(matched);
        self.config_futures.push(futures);
        self.fresh_reset_ids.insert(config, reset);
        Ok(reset)
    }

    /// Exact lexer coordinate for a parser reset that accepts the current token
    /// boundary but has no positive-byte continuation. This uses the ordinary
    /// config/fresh-reset representation; it is not a second lexer engine.
    fn fresh_empty_reset_config(&mut self) -> Result<u32, String> {
        if self.deterministic {
            return Err("empty fresh reset is only valid in the NFA-config coordinate".to_owned());
        }
        let empty=self.intern_config(Vec::new())?;
        self.fresh_reset_config(empty)
    }

    fn step_config(&mut self, config: u32, byte: u8) -> Result<Option<u32>, String> {
        if self.profile_transition_work {
            self.profile_step_calls += 1;
        }
        if self.deterministic {
            let target = self.transition(config, byte);
            return Ok((target != u32::MAX).then_some(target));
        }
        if let Some(state) = self.raw_state_for_config(config) {
            let target = self.transition(state, byte);
            return if target == u32::MAX {
                Ok(None)
            } else {
                self.config_for_raw_start(target).map(Some)
            };
        }
        let config_index = self.config_index(config).ok_or_else(|| "unknown dynamic lexer config".to_owned())?;
        if let Some(row) = self.transitions[config_index].as_ref() {
            let cached = row[byte as usize];
            if cached != DYNAMIC_NFA_CONFIG_UNKNOWN {
                if self.profile_transition_work {
                    self.profile_step_cache_hits += 1;
                }
                return Ok((cached != DYNAMIC_NFA_CONFIG_DEAD).then_some(cached));
            }
        }
        if self.profile_transition_work {
            self.profile_step_cache_misses += 1;
            self.profile_physical_states_scanned += self.configs[config_index].len();
        }

        let closed_targets = {
            let mut targets = Vec::<u32>::new();
            let config_len = self.configs[config_index].len();
            for state_index in 0..config_len {
                let state = self.configs[config_index][state_index];
                let virtual_residual_terminal = self
                    .profile_transition_work
                    .then(|| self.tokenizer.virtual_residual_terminal_for_state(state))
                    .flatten();
                let transition_started = self.profile_transition_work.then(std::time::Instant::now);
                let target = self.transition(state, byte);
                if let Some(started) = transition_started {
                    let elapsed = started.elapsed().as_nanos() as u64;
                    if let Some(terminal) = virtual_residual_terminal {
                        self.profile_member_virtual_residual_transitions += 1;
                        self.profile_member_virtual_residual_transition_ns = self
                            .profile_member_virtual_residual_transition_ns
                            .saturating_add(elapsed);
                        let entry = self
                            .profile_member_virtual_residual_by_terminal
                            .entry(terminal)
                            .or_insert((0, 0));
                        entry.0 += 1;
                        entry.1 = entry.1.saturating_add(elapsed);
                    } else {
                        self.profile_member_physical_transitions += 1;
                        self.profile_member_physical_transition_ns = self
                            .profile_member_physical_transition_ns
                            .saturating_add(elapsed);
                    }
                }
                if target != u32::MAX {
                    if self.config_admitted[config_index].is_some() {
                        // Do not first intern the unrestricted closure only
                        // to discard it while constructing the projection.
                        let closure = self.tokenizer.singleton_epsilon_closure(target);
                        self.check_growth(targets.len(), closure.len())?;
                        targets.extend(closure);
                        continue;
                    }
                    let target_config = self.config_for_raw_start(target)?;
                    if let Some(target_state) = self.raw_state_for_config(target_config) {
                        self.check_growth(targets.len(), 1)?;
                        targets.push(target_state);
                    } else {
                        let target_index = self.config_index(target_config).ok_or_else(|| "unknown dynamic target config".to_owned())?;
                        let target_states = &self.configs[target_index];
                        self.check_growth(targets.len(), target_states.len())?;
                        targets.extend_from_slice(target_states);
                    }
                }
            }
            targets
        };
        let target = if closed_targets.is_empty() {
            DYNAMIC_NFA_CONFIG_DEAD
        } else if let Some(filter) = self.config_admitted[config_index] {
            self.intern_admitted_config(closed_targets, filter)?
        } else {
            self.intern_config(closed_targets)?
        };
        let row = self.transitions[config_index]
            .get_or_insert_with(|| Box::new([DYNAMIC_NFA_CONFIG_UNKNOWN; 256]));
        row[byte as usize] = target;
        Ok((target != DYNAMIC_NFA_CONFIG_DEAD).then_some(target))
    }

    fn residual_config(&mut self, config: u32) -> Result<Option<u32>, String> {
        if self.deterministic {
            return Ok(self
                .tokenizer
                .exact_dynamic_state_has_future(config)?
                .then_some(config));
        }
        if let Some(state) = self.raw_state_for_config(config) {
            return Ok(self
                .tokenizer
                .exact_dynamic_state_has_future(state)?
                .then_some(config));
        }
        let config_index = self.config_index(config).ok_or_else(|| "unknown dynamic residual config".to_owned())?;
        if self.config_admitted[config_index].is_some() {
            return Ok((!self.config_futures[config_index].is_empty()).then_some(config));
        }
        let cached = self.residual_configs[config_index];
        if cached != DYNAMIC_NFA_CONFIG_UNKNOWN {
            return Ok((cached != DYNAMIC_NFA_CONFIG_DEAD).then_some(cached));
        }

        let mut residual_states = Vec::new();
        for &state in self.configs[config_index].iter() {
            if self.tokenizer.exact_dynamic_state_has_future(state)? {
                residual_states.push(state);
            }
        }
        let residual = if residual_states.is_empty() {
            DYNAMIC_NFA_CONFIG_DEAD
        } else if residual_states.len() == self.configs[config_index].len() {
            config
        } else {
            self.intern_config(residual_states)?
        };
        self.residual_configs[config_index] = residual;
        Ok((residual != DYNAMIC_NFA_CONFIG_DEAD).then_some(residual))
    }

    fn execute_from_config_all_widths(
        &mut self,
        input: &[u8],
        start_config: u32,
    ) -> Result<DynamicConfigExecResult, String> {
        let mut config = start_config;
        let mut matches = Vec::new();
        for (index, &byte) in input.iter().enumerate() {
            let Some(next_config) = self.step_config(config, byte)? else {
                return Ok(DynamicConfigExecResult {
                    end_config: None,
                    matches,
                });
            };
            config = next_config;
            let width = index + 1;
            for state_index in 0..self.config_len(config) {
                let state = self.config_state(config, state_index);
                for id in self.tokenizer.matched_terminals_iter(state) {
                    self.check_growth(matches.len(), 1)?;
                    matches.push(TokenizerMatch {
                        id,
                        width,
                        end_state: state,
                    });
                }
            }
        }
        Ok(DynamicConfigExecResult {
            end_config: self.residual_config(config)?,
            matches,
        })
    }

    fn execute_from_state_all_widths(
        &mut self,
        input: &[u8],
        start: u32,
    ) -> Result<TokenizerExecResult, String> {
        let start_config = self.config_for_raw_start(start)?;
        let execution = self.execute_from_config_all_widths(input, start_config)?;
        let mut end_state = TokenizerStateSet::new();
        if let Some(end_config) = execution.end_config {
            for state_index in 0..self.config_len(end_config) {
                end_state.push(self.config_state(end_config, state_index));
            }
        }
        Ok(TokenizerExecResult {
            end_state,
            matches: execution.matches,
        })
    }

    #[inline]
    fn config_len(&self, config: u32) -> usize {
        if self.deterministic {
            1
        } else if self.raw_state_for_config(config).is_some() {
            1
        } else {
            self.configs[self.config_index(config).expect("known config")].len()
        }
    }

    #[inline]
    fn config_state(&self, config: u32, index: usize) -> u32 {
        if self.deterministic {
            debug_assert_eq!(index, 0);
            config
        } else if let Some(state) = self.raw_state_for_config(config) {
            debug_assert_eq!(index, 0);
            state
        } else {
            self.configs[self.config_index(config).expect("known config")][index]
        }
    }

    #[inline]
    fn config_has_finalizer(&self, config: u32) -> bool {
        if !self.deterministic && self.raw_state_for_config(config).is_none() {
            return !self.config_matched[self.config_index(config).expect("known config")].is_empty();
        }
        !self
            .tokenizer
            .matched_terminal_bitset(self.config_state(config, 0))
            .is_empty()
    }

    fn config_matched_terminals(&self, config: u32) -> SmallVec<[TerminalID; 4]> {
        if !self.deterministic && self.raw_state_for_config(config).is_none() {
            return self.config_matched[self.config_index(config).expect("known config")]
                .iter_ones()
                .map(|terminal| terminal as TerminalID)
                .collect();
        }
        self.tokenizer
            .matched_terminals_iter(self.config_state(config, 0))
            .collect()
    }

    #[inline]
    fn config_finalizer_code(&self, config: u32) -> u32 {
        const NONE: u32 = u32::MAX;
        const MULTI: u32 = u32::MAX - 1;
        let terminals = self.config_matched_terminals(config);
        match terminals.as_slice() {
            [] => NONE,
            [terminal] => *terminal,
            _ => MULTI,
        }
    }

    #[inline]
    fn config_future_contains(&self, config: u32, terminal: TerminalID) -> bool {
        if !self.deterministic && self.raw_state_for_config(config).is_none() {
            return self.config_futures[self.config_index(config).expect("known config")].contains(terminal as usize);
        }
        self.tokenizer
            .possible_future_terminals(self.config_state(config, 0))
            .contains(terminal as usize)
    }

    fn config_future_contains_exact(
        &self,
        config: u32,
        terminal: TerminalID,
    ) -> Result<bool, String> {
        if !self.deterministic && self.raw_state_for_config(config).is_none()
            && self.config_admitted[self.config_index(config).expect("known config")].is_some()
            && !self.config_futures[self.config_index(config).expect("known config")]
                .contains(terminal as usize)
        {
            return Ok(false);
        }
        for index in 0..self.config_len(config) {
            let state = self.config_state(config, index);
            if let Some(owner_terminal) = self.tokenizer.virtual_residual_terminal_for_state(state) {
                if owner_terminal != terminal {
                    continue;
                }
                if self.tokenizer.exact_dynamic_state_has_future(state)? {
                    return Ok(true);
                }
                continue;
            }
            if self
                .tokenizer
                .possible_future_terminals(state)
                .contains(terminal as usize)
                && self.tokenizer.exact_dynamic_state_has_future(state)?
            {
                return Ok(true);
            }
        }
        Ok(false)
    }

    #[inline]
    fn config_future_intersects(&self, config: u32, terminals: &BitSet) -> bool {
        if !self.deterministic && self.raw_state_for_config(config).is_none() {
            return !terminals.is_disjoint(&self.config_futures[self.config_index(config).expect("known config")]);
        }
        !terminals.is_disjoint(
            self.tokenizer
                .possible_future_terminals(self.config_state(config, 0)),
        )
    }

    fn config_future_intersects_exact(
        &self,
        config: u32,
        terminals: &BitSet,
    ) -> Result<bool, String> {
        if !self.deterministic && self.raw_state_for_config(config).is_none()
            && self.config_admitted[self.config_index(config).expect("known config")].is_some()
            && terminals.is_disjoint(&self.config_futures[self.config_index(config).expect("known config")])
        {
            return Ok(false);
        }
        for index in 0..self.config_len(config) {
            let state = self.config_state(config, index);
            if let Some(owner_terminal) = self.tokenizer.virtual_residual_terminal_for_state(state) {
                if !terminals.contains(owner_terminal as usize) {
                    continue;
                }
                if self.tokenizer.exact_dynamic_state_has_future(state)? {
                    return Ok(true);
                }
                continue;
            }
            if !terminals.is_disjoint(self.tokenizer.possible_future_terminals(state))
                && self.tokenizer.exact_dynamic_state_has_future(state)?
            {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Exact union of already-interned runtime configurations. This is the NFA
    /// analogue of merging deterministic mask states that carry the same parser
    /// object. Deterministic raw coordinates have no subset namespace to merge
    /// into, so only identical states collapse in that mode.
    fn union_configs(&mut self, configs: &[u32]) -> Result<Option<u32>, String> {
        let Some((&first, rest)) = configs.split_first() else {
            return Ok(None);
        };
        if self.deterministic {
            return Ok(rest.iter().all(|&state| state == first).then_some(first));
        }
        if rest.iter().all(|&config| config == first) {
            return Ok(Some(first));
        }
        if configs.iter().any(|&config| {
            self.config_index(config).is_some_and(|index| self.config_is_fresh_reset[index])
        }) {
            // A union of physical lexer members alone cannot represent the
            // fresh branch's epsilon acceptance. Preserve the branches instead.
            return Ok(None);
        }
        let filter_for = |config| {
            self.config_index(config).and_then(|index| self.config_admitted[index])
        };
        let filter = filter_for(first);
        if rest.iter().any(|&config| filter_for(config) != filter) {
            // Erasing a branch's terminal restriction would invalidate the
            // projected-state invariant. Keep differing languages separate.
            return Ok(None);
        }
        let mut states = Vec::<u32>::new();
        for &config in configs {
            if let Some(state) = self.raw_state_for_config(config) {
                self.check_growth(states.len(), 1)?;
                states.push(state);
            } else {
                let index = self.config_index(config).ok_or_else(|| "dynamic config union referenced an unknown config".to_owned())?;
                let members = self
                    .configs
                    .get(index)
                    .ok_or_else(|| "dynamic NFA config union referenced an unknown config".to_owned())?;
                self.check_growth(states.len(), members.len())?;
                states.extend_from_slice(members);
            }
        }
        if states.is_empty() {
            return Ok(None);
        }
        if let Some(filter) = filter {
            let result = self.intern_admitted_config(states, filter)?;
            Ok((result != DYNAMIC_NFA_CONFIG_DEAD).then_some(result))
        } else {
            self.intern_config(states).map(Some)
        }
    }

    fn config_next_bytes(&self, config: u32) -> U8Set {
        let mut bytes = U8Set::empty();
        for state_index in 0..self.config_len(config) {
            let state = self.config_state(config, state_index);
            for (byte, _) in self.tokenizer.transitions_from(state) {
                bytes.insert(byte);
            }
        }
        bytes
    }

}

fn for_each_token_matching_terminal_from_state(
    constraint: &Constraint,
    start_state: u32,
    terminal: TerminalID,
    mut visit_token: impl FnMut(u32),
) -> Result<(), String> {
    if constraint.visit_possible_match_original_tokens(start_state, terminal, &mut visit_token) {
        return Ok(());
    }

    let vocab = constraint.dynamic_mask_vocab_for_runtime();
    let trie = vocab.trie.as_ref();
    let mut scan_cache = DynamicNfaScanCache::new(constraint, None);
    let start_config = scan_cache.config_for_raw_start(start_state)?;
    let mut work = vec![(0u32, start_config)];

    while let Some((node, config)) = work.pop() {
        for edge in trie.children(node) {
            let mut config = config;
            let mut alive = true;
            let mut matched = false;
            for &byte in trie.edge_bytes(edge) {
                let Some(next_config) = scan_cache.step_config(config, byte)? else {
                    alive = false;
                    break;
                };
                config = next_config;
                matched = (0..scan_cache.config_len(config)).any(|state_index| {
                    let state = scan_cache.config_state(config, state_index);
                    constraint
                        .tokenizer
                        .matched_terminals_iter(state)
                        .any(|matched_terminal| matched_terminal == terminal)
                });
                if matched {
                    break;
                }
            }

            if matched {
                for &canonical_token in trie.subtree_tokens(edge.child) {
                    if let Some(token_ids) = vocab.token_ids(canonical_token) {
                        for &token_id in token_ids {
                            visit_token(token_id);
                        }
                    }
                }
            } else if alive {
                work.push((edge.child, config));
            }
        }
    }

    Ok(())
}

fn bounded_scalar_distance_to_terminal(
    tokenizer: &Tokenizer,
    start: u32,
    terminal: TerminalID,
    max_depth: u16,
) -> Option<u16> {
    if tokenizer.state_has_epsilon_transitions(start) || tokenizer.state_is_virtual_runtime(start) {
        return None;
    }
    let mut seen = FxHashSet::<u32>::default();
    let mut queue = std::collections::VecDeque::<(u32, u16)>::new();
    seen.insert(start);
    queue.push_back((start, 0));
    while let Some((state, depth)) = queue.pop_front() {
        if depth >= max_depth {
            continue;
        }
        let next_depth = depth + 1;
        for (_, target) in tokenizer.transitions_from(state) {
            if tokenizer.state_has_epsilon_transitions(target)
                || tokenizer.state_is_virtual_runtime(target)
            {
                return None;
            }
            if tokenizer
                .matched_terminal_bitset(target)
                .contains(terminal as usize)
            {
                return Some(next_depth);
            }
            if next_depth < max_depth
                && tokenizer
                    .possible_future_terminals(target)
                    .contains(terminal as usize)
                && seen.insert(target)
            {
                queue.push_back((target, next_depth));
            }
        }
    }
    Some(u16::MAX)
}

/// Vocabulary-relative terminal-match existence over the finite mask
/// projection's exact Flat16 rows. This is the preferred pending-guard probe:
/// it stays in the same coordinate as `InitialPruneGuard` and uses one dense
/// table lookup per vocabulary byte. `None` is a structural decline.
fn flat16_mask_vocab_terminal_matches(
    constraint: &Constraint,
    mask_state: u32,
    terminal: TerminalID,
    mut blocked_words: Option<&mut [u32]>,
) -> Option<bool> {
    let vocab = constraint.dynamic_mask_vocab_for_runtime();
    // `mask_tokenizer_fast_transitions` is also populated for the ordinary
    // source tokenizer when no separate mask projection exists. Keep the
    // guard probe in that exact active coordinate instead of declining merely
    // because `mask_runtime_tokenizer()` has no projection object to return.
    let tokenizer = vocab
        .mask_runtime_tokenizer()
        .unwrap_or(&constraint.tokenizer);
    let dense = match vocab.mask_projection_fast_transitions() {
        Some(FastTokenizerTransitions::Flat16 {
            transitions,
            finalizer_code,
            ..
        }) => Some((transitions.as_ref(), finalizer_code.as_ref())),
        _ => None,
    };
    if mask_state >= tokenizer.num_states()
        || tokenizer.state_has_epsilon_transitions(mask_state)
        || tokenizer.state_is_virtual_runtime(mask_state)
    {
        return None;
    }
    if !tokenizer
        .possible_future_terminals(mask_state)
        .contains(terminal as usize)
    {
        return Some(false);
    }

    // This probe may inspect hundreds of thousands of vocabulary bytes while
    // touching only a small dense lexer state space. Hoist the terminal and
    // structural predicates out of the hot byte loop instead of repeatedly
    // traversing tokenizer bitsets/metadata for every trie byte.
    const STATE_MATCHES: u8 = 1 << 0;
    const STATE_LIVE: u8 = 1 << 1;
    const STATE_ORDINARY: u8 = 1 << 2;
    // Cold guard masks usually visit only a tiny fraction of a large tokenizer.
    // Derive terminal flags lazily instead of scanning every tokenizer state
    // before the vocabulary walk. Values 0..=7 are real flags; u8::MAX is
    // the uncomputed sentinel.
    let mut state_flags = vec![u8::MAX; tokenizer.num_states() as usize];
    let mut state_flags_for = |state: u32| -> Option<u8> {
        let slot = state_flags.get_mut(state as usize)?;
        if *slot != u8::MAX {
            return Some(*slot);
        }
        let mut flags = 0u8;
        if tokenizer
            .matched_terminal_bitset(state)
            .contains(terminal as usize)
        {
            flags |= STATE_MATCHES;
        }
        if tokenizer
            .possible_future_terminals(state)
            .contains(terminal as usize)
        {
            flags |= STATE_LIVE;
        }
        if !tokenizer.state_has_epsilon_transitions(state)
            && !tokenizer.state_is_virtual_runtime(state)
        {
            flags |= STATE_ORDINARY;
        }
        *slot = flags;
        Some(flags)
    };

    // The common case has the exact Flat16 projection already materialized.
    // Reuse the same pre-flattened strict vocabulary walk as dynamic masking:
    // carrying one lexer state per DFS depth is enough for this single-terminal
    // query.  Once a byte reaches the remembered terminal, every token in the
    // owning radix child subtree is blocked, so jump directly past that subtree.
    // Likewise lexical death/non-liveness can skip the subtree without marking.
    // The older radix walker below remains the structural fallback.
    let walk_ops = vocab.trie.full_walk_ops();
    if !walk_ops.is_empty() {
        let max_depth = vocab.trie.full_walk_max_parent_depth() as usize;
        let mut stack_state = vec![u32::MAX; max_depth.saturating_add(2)];
        let mut local_rows = if dense.is_none() {
            (0..tokenizer.num_states())
                .map(|_| None::<Box<[u32; 256]>>)
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        stack_state[0] = mask_state;
        let mut current_state = mask_state;
        let mut remaining_ops = walk_ops.iter();
        let mut any_match = false;
        // O2 endpoints can each stand for many original model tokens. Record
        // matched quotient endpoints first, then materialize the dense blocked
        // mask in whichever polarity touches fewer canonical classes. This
        // avoids expanding a near-universal blocked set token ID by token ID.
        let adaptive_quotient_words = blocked_words.is_some() && vocab.is_grammar_quotiented();
        let mut matched_canonical = adaptive_quotient_words
            .then(|| vec![false; vocab.canonical_token_count()]);
        let mut matched_canonical_count = 0usize;
        while let Some(&op) = remaining_ops.next() {
            let parent_depth = op.parent_depth() as usize;
            if op.starts_edge() {
                current_state = *stack_state.get(parent_depth)?;
            }
            if op.consumes_byte() {
                let target = if let Some((transitions, _)) = dense {
                    let cell = unsafe {
                        *transitions.get_unchecked(
                            (current_state as usize).wrapping_mul(256) + op.byte() as usize,
                        )
                    };
                    (cell != u16::MAX).then_some(u32::from(cell & 0x7fff))
                } else {
                    let slot = local_rows.get_mut(current_state as usize)?;
                    if slot.is_none() {
                        let mut row = Box::new([u32::MAX; 256]);
                        for (byte, target) in tokenizer.transitions_from(current_state) {
                            row[byte as usize] = target;
                        }
                        *slot = Some(row);
                    }
                    let target = slot.as_ref()?[op.byte() as usize];
                    (target != u32::MAX).then_some(target)
                };
                let Some(target) = target else {
                    let op_index = walk_ops.len() - remaining_ops.as_slice().len() - 1;
                    let (_, subtree_end_op) = vocab.trie.full_walk_dead_subtree(op_index);
                    remaining_ops = walk_ops[subtree_end_op as usize..].iter();
                    continue;
                };
                current_state = target;
                let flags = state_flags_for(current_state)?;
                if flags & STATE_MATCHES != 0 {
                    let op_index = walk_ops.len() - remaining_ops.as_slice().len() - 1;
                    let (child, subtree_end_op) = vocab.trie.full_walk_dead_subtree(op_index);
                    let canonicals = vocab.trie.subtree_tokens(child);
                    if !canonicals.is_empty() {
                        any_match = true;
                        if let Some(flags) = matched_canonical.as_mut() {
                            for &canonical in canonicals {
                                let slot = flags.get_mut(canonical as usize)?;
                                if !*slot {
                                    *slot = true;
                                    matched_canonical_count += 1;
                                }
                            }
                        } else if let Some(words) = blocked_words.as_deref_mut() {
                            for &canonical in canonicals {
                                for &(word, mask) in vocab.token_word_masks(canonical) {
                                    if let Some(slot) = words.get_mut(word as usize) {
                                        *slot |= mask;
                                    }
                                }
                            }
                        } else {
                            return Some(true);
                        }
                    }
                    remaining_ops = walk_ops[subtree_end_op as usize..].iter();
                    continue;
                }
                if flags & STATE_LIVE == 0 {
                    let op_index = walk_ops.len() - remaining_ops.as_slice().len() - 1;
                    let (_, subtree_end_op) = vocab.trie.full_walk_dead_subtree(op_index);
                    remaining_ops = walk_ops[subtree_end_op as usize..].iter();
                    continue;
                }
                if flags & STATE_ORDINARY == 0 {
                    return None;
                }
            }
            if op.ends_edge() {
                *stack_state.get_mut(parent_depth + 1)? = current_state;
            }
        }
        if let (Some(words), Some(flags)) = (blocked_words.as_deref_mut(), matched_canonical) {
            let canonical_count = flags.len();
            if matched_canonical_count <= canonical_count.saturating_sub(matched_canonical_count) {
                for (canonical, matched) in flags.into_iter().enumerate() {
                    if !matched {
                        continue;
                    }
                    for &(word, mask) in vocab.token_word_masks(canonical as u32) {
                        if let Some(slot) = words.get_mut(word as usize) {
                            *slot |= mask;
                        }
                    }
                }
            } else {
                let all_words = vocab.all_original_token_words();
                let copy_len = words.len().min(all_words.len());
                words[..copy_len].copy_from_slice(&all_words[..copy_len]);
                if copy_len < words.len() {
                    words[copy_len..].fill(0);
                }
                for (canonical, matched) in flags.into_iter().enumerate() {
                    if matched {
                        continue;
                    }
                    for &(word, mask) in vocab.token_word_masks(canonical as u32) {
                        if let Some(slot) = words.get_mut(word as usize) {
                            *slot &= !mask;
                        }
                    }
                }
            }
        }
        return Some(any_match);
    }

    // Compute exact positive distance-to-match only for lexer states reached by
    // this vocabulary walk. A whole-token projection can have tens of thousands
    // of unrelated states; constructing a reverse graph for all of them costs
    // milliseconds even when this guard matches after one or two bytes. The
    // vocabulary's true maximum token length is a semantic bound: distances
    // beyond it are equivalent to infinity for this filter.
    let max_token_len = u16::try_from(vocab.max_token_byte_len()).ok()?;
    let mut distance = vec![None::<u16>; tokenizer.num_states() as usize];
    let mut distance_for = |state: u32| -> Option<u16> {
        let slot = distance.get_mut(state as usize)?;
        if let Some(cached) = *slot {
            return Some(cached);
        }
        let value = bounded_scalar_distance_to_terminal(tokenizer, state, terminal, max_token_len)?;
        *slot = Some(value);
        Some(value)
    };
    let start_distance = distance_for(mask_state)?;

    let trie = vocab.trie.as_ref();
    let mut work = vec![(0u32, mask_state)];
    let mut any_match = false;
    while let Some((node, state)) = work.pop() {
        let remaining = trie.node(node).subtree_max_byte_len;
        let state_distance = distance_for(state)?;
        if state_distance == u16::MAX || u32::from(state_distance) > remaining {
            continue;
        }
        for edge in trie.children(node) {
            let edge_remaining = edge
                .byte_len
                .saturating_add(trie.node(edge.child).subtree_max_byte_len);
            if state_distance == u16::MAX || u32::from(state_distance) > edge_remaining {
                continue;
            }
            let mut state = state;
            let mut alive = true;
            let mut matched = false;
            let edge_bytes = trie.edge_bytes(edge);
            for (byte_index, &byte) in edge_bytes.iter().enumerate() {
                state = if let Some((transitions, _)) = dense {
                    let cell = unsafe {
                        *transitions
                            .get_unchecked((state as usize).wrapping_mul(256) + byte as usize)
                    };
                    if cell == u16::MAX {
                        alive = false;
                        break;
                    }
                    u32::from(cell & 0x7fff)
                } else if let Some(target) = Lexer::step(tokenizer, state, byte) {
                    target
                } else {
                    alive = false;
                    break;
                };
                let flags = state_flags_for(state)?;
                if flags & STATE_MATCHES != 0 {
                    matched = true;
                    break;
                }
                if flags & STATE_LIVE == 0 {
                    alive = false;
                    break;
                }
                if flags & STATE_ORDINARY == 0 {
                    return None;
                }
                let remaining_edge = edge_bytes.len() - byte_index - 1;
                let remaining = remaining_edge as u32
                    + trie.node(edge.child).subtree_max_byte_len;
                let state_distance = distance_for(state)?;
                if state_distance == u16::MAX || u32::from(state_distance) > remaining {
                    alive = false;
                    break;
                }
            }
            if matched {
                let originals = vocab.subtree_original_tokens_for(trie, edge.child);
                if !originals.is_empty() {
                    any_match = true;
                    if let Some(words) = blocked_words.as_deref_mut() {
                        for &token_id in originals {
                            let word = token_id as usize / 32;
                            let bit = token_id % 32;
                            if let Some(slot) = words.get_mut(word) {
                                *slot |= 1u32 << bit;
                            }
                        }
                    } else {
                        return Some(true);
                    }
                }
                continue;
            }
            if alive {
                work.push((edge.child, state));
            }
        }
    }
    Some(any_match)
}

pub(crate) fn or_blocked_internal_tokens_for_exclusions(
    constraint: &Constraint,
    exclusions: &TerminalsDisallowed,
    dense: &mut [u64],
) -> Result<(), String> {
    for (&lexer_state, terminals) in exclusions.iter() {
        for &terminal in terminals.iter() {
            for_each_token_matching_terminal_from_state(
                constraint,
                lexer_state,
                terminal,
                |token_id| {
                    if let Some(internal_token) =
                        constraint.final_internal_token_for_original(token_id)
                    {
                        let word = internal_token as usize / 64;
                        let bit = internal_token % 64;
                        if let Some(slot) = dense.get_mut(word) {
                            *slot |= 1u64 << bit;
                        }
                    }
                },
            )?;
        }
    }
    Ok(())
}

fn update_special_token_mask(state: &ConstraintState<'_>, buf: &mut [u32]) {
    let mut previous_token_id = None;
    for special in &state.constraint.special_token_terminals {
        if state
            .constraint
            .is_late_grammar_placeholder_terminal(special.terminal_id)
        {
            continue;
        }
        if previous_token_id == Some(special.token_id) {
            continue;
        }
        previous_token_id = Some(special.token_id);
        if super::commit::advance_special_token_paths(
            state.constraint,
            &state.state,
            special.token_id,
        )
        .is_some_and(|gss| !gss.is_empty())
        {
            set_mask_bit(buf, special.token_id);
        }
    }
}

/// Dynamic masking keeps terminal restrictions outside the parser GSS. The
/// parser table routines still use `ParserGSS`, so give their stack operations
/// an otherwise-unused empty accumulator.
fn with_empty_accumulators(stacks: &ParserStacks) -> ParserGSS {
    stacks.apply(|_| TerminalsDisallowed::new())
}

impl InitialPruneGuard {
    /// Build the token-start pruning state for one correlated tokenizer/GSS
    /// branch. Every remembered `(lexer state, terminal)` pair is an independent
    /// condition: a later match of that same terminal invalidates the
    /// provisional boundary that created this parser path.
    fn new(
        vocab: &DynamicMaskVocab,
        terminals_disallowed: &TerminalsDisallowed,
    ) -> Self {
        let mut memories = Vec::new();
        for (&lexer_state, terminals) in terminals_disallowed.iter() {
            let mask_state = vocab.mask_runtime_state(lexer_state);
            for &terminal in terminals.iter() {
                memories.push((mask_state, lexer_state, terminal));
            }
        }
        if memories.is_empty() {
            return Self::Passed;
        }
        memories.sort_unstable();
        memories.dedup();
        Self::Pending { memories: memories.into() }
    }

    fn new_recursive(terminals_disallowed: &TerminalsDisallowed) -> Self {
        let mut memories = Vec::new();
        for (&lexer_state, terminals) in terminals_disallowed.iter() {
            for &terminal in terminals.iter() {
                memories.push((lexer_state, lexer_state, terminal));
            }
        }
        if memories.is_empty() {
            return Self::Passed;
        }
        memories.sort_unstable();
        memories.dedup();
        Self::Pending { memories: memories.into() }
    }

    #[inline]
    fn is_passed(&self) -> bool {
        matches!(self, Self::Passed)
    }

    fn blocked_output_mask(
        &self,
        constraint: &Constraint,
        mask_words: usize,
    ) -> Result<Option<Arc<Vec<u32>>>, String> {
        let Self::Pending { memories } = self else {
            return Ok(None);
        };
        let vocab = constraint.dynamic_mask_vocab_for_runtime();
        if std::env::var_os("GLRMASK_PROFILE_DYNAMIC_MASK").is_some() {
            for &(mask_state, source_state, terminal) in memories.iter() {
                if let Some(projection) = vocab.self_loop_projection(mask_state) {
                    eprintln!(
                        "[glrmask/profile][pending_guard_self_loop] mask_state={} source_state={} terminal={} futures={:?} frontier_nonempty={}",
                        mask_state,
                        source_state,
                        terminal,
                        projection.future_terminals,
                        projection.pre_match_frontier_words.iter().any(|&word| word != 0),
                    );
                } else {
                    eprintln!(
                        "[glrmask/profile][pending_guard_self_loop] mask_state={} source_state={} terminal={} projection=none",
                        mask_state,
                        source_state,
                        terminal,
                    );
                }
            }
        }
        let mut source_memories = memories
            .iter()
            .map(|&(_, source_state, terminal)| (source_state, terminal))
            .collect::<Vec<_>>();
        source_memories.sort_unstable();
        source_memories.dedup();
        if let Some(cached) = vocab.cached_pending_guard_blocked_mask(&source_memories) {
            return Ok(Some(cached));
        }
        let mut blocked = vec![0u32; mask_words];
        let mut fallback_memories = Vec::with_capacity(source_memories.len());
        for &(mask_state, source_state, terminal) in memories.iter() {
            if flat16_mask_vocab_terminal_matches(
                constraint,
                mask_state,
                terminal,
                Some(&mut blocked),
            )
            .is_some()
            {
                continue;
            }
            if !fallback_memories.contains(&(source_state, terminal)) {
                fallback_memories.push((source_state, terminal));
            }
        }
        for &(source_state, terminal) in &fallback_memories {
            if constraint
                .possible_match_original_tokens_definitely_empty(source_state, terminal)
                == Some(true)
            {
                continue;
            }
            for_each_token_matching_terminal_from_state(
                constraint,
                source_state,
                terminal,
                |token_id| {
                    let word = token_id as usize / 32;
                    let bit = token_id % 32;
                    if let Some(slot) = blocked.get_mut(word) {
                        *slot |= 1u32 << bit;
                    }
                },
            )?;
        }
        if std::env::var_os("GLRMASK_PROFILE_DYNAMIC_MASK").is_some() {
            eprintln!(
                "[glrmask/profile][pending_guard_blocked_mask] memories={:?} blocked_tokens={}",
                source_memories,
                blocked.iter().map(|word| word.count_ones() as usize).sum::<usize>(),
            );
        }
        let cached = vocab.cache_pending_guard_blocked_mask(&source_memories, blocked);
        Ok(Some(cached))
    }
}

fn parser_child(
    constraint: &Constraint,
    stacks: &ParserStacks,
    terminal: TerminalID,
) -> Option<ParserStacks> {
    // Ignore terminals reset the lexer but deliberately leave the parser alone.
    if Some(terminal) == constraint.ignore_terminal {
        return Some(stacks.clone());
    }
    // A proved pure push commutes with mapping every annotation to the empty
    // value. Apply that same generic GSS primitive directly, rather than
    // allocating two annotation-mapping memos/interfaces around the advance.
    // All nontrivial effects retain the complete annotated parser evaluator.
    if let Some(parser) = &constraint.template_parser
        && !constraint.parser_skip_terminals().contains(&terminal)
        && let Some(Some(template)) = constraint.template_dfas_by_terminal.get(terminal as usize)
        && let Some(Some(fast)) = constraint.fast_template_dfas_by_terminal.get(terminal as usize)
        && let Some(advanced) = super::commit::simple_read_shift::apply(template, fast, stacks)
    {
        parser.record_advance();
        #[cfg(test)]
        super::commit::simple_read_shift::MASK_SUCCESSES.with(|count| count.set(count.get() + 1));
        return Some(advanced);
    }
    let parser_gss = with_empty_accumulators(stacks);
    // The actual structural advance is already the definitive admissibility
    // test. Running exact admission first would duplicate reduction simulation
    // for every terminal branch explored by the dynamic traversal.
    let advanced = if constraint.has_template_parser() {
        super::commit::advance_parser_stacks_if_possible(constraint, &parser_gss, terminal)
            .unwrap_or_else(ParserGSS::empty)
    } else if let Some(advanced) =
        constraint.advance_compact_segmented_parser(&parser_gss, terminal)
    {
        advanced
    } else {
        constraint
            .direct_regular_cached_advance(&parser_gss, terminal)
            .or_else(|| super::commit::advance_stacks_template_dfa(constraint, &parser_gss, terminal))
            .unwrap_or_else(|| advance_stacks(&constraint.table, &parser_gss, terminal))
    }
    .apply(|_| ());
    (!advanced.is_empty()).then_some(advanced)
}

struct DynamicDeadlinePoll {
    deadline: Option<Instant>,
    remaining: u16,
}

impl DynamicDeadlinePoll {
    fn new(deadline: Option<Instant>) -> Self {
        Self {
            deadline,
            remaining: 0,
        }
    }

    #[inline]
    fn check(&mut self) -> Result<(), String> {
        let Some(deadline) = self.deadline else {
            return Ok(());
        };
        if self.remaining != 0 {
            self.remaining -= 1;
            return Ok(());
        }
        self.remaining = 1_023;
        if Instant::now() >= deadline {
            Err("glrmask_dynamic mask generation timed out".to_owned())
        } else {
            Ok(())
        }
    }
}

const DYNAMIC_MASK_CACHE_MAX_STACKS: usize = 4_096;
const DYNAMIC_MASK_CACHE_MAX_DEPTH: u32 = 256;

#[derive(Clone, Copy)]
struct DynamicMaskProfileSelector {
    all: bool,
    generation: Option<u64>,
}

#[inline]
pub(crate) fn dynamic_mask_profile_enabled(generation: u64) -> bool {
    static SELECTOR: OnceLock<DynamicMaskProfileSelector> = OnceLock::new();
    let selector = SELECTOR.get_or_init(|| DynamicMaskProfileSelector {
        all: std::env::var_os("GLRMASK_PROFILE_DYNAMIC_MASK").is_some(),
        generation: std::env::var("GLRMASK_PROFILE_DYNAMIC_MASK_GENERATION")
            .ok()
            .and_then(|value| value.parse::<u64>().ok()),
    });
    selector.all || selector.generation == Some(generation)
}

#[inline]
pub(crate) fn dynamic_mask_prewalk_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var_os("GLRMASK_PROFILE_DYNAMIC_PREWALK_PHASES").is_some())
}

#[inline]
pub(crate) fn dynamic_mask_proof_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var_os("GLRMASK_PROFILE_DYNAMIC_PROOF_PHASES").is_some())
}

#[inline]
pub(crate) fn dynamic_mask_flat16_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var_os("GLRMASK_PROFILE_DYNAMIC_FLAT16_PHASES").is_some())
}

#[inline]
fn dynamic_virtual_dense_cell_cache_experiment_enabled() -> bool {
    std::env::var_os("GLRMASK_EXPERIMENT_DYNAMIC_VIRTUAL_DENSE_CELL_CACHE").is_some()
}

#[inline]
fn direct_residual_defer_finalizer_experiment_enabled() -> bool {
    std::env::var_os("GLRMASK_EXPERIMENT_DIRECT_RESIDUAL_DEFER_FINALIZER").is_some()
}

pub(crate) fn dynamic_mask_cache_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("GLRMASK_DISABLE_DYNAMIC_MASK_CACHE").is_none())
}

#[inline(always)]
fn recursive_persistent_mask_cache_enabled() -> bool {
    #[cfg(test)]
    {
        return TEST_RECURSIVE_PERSISTENT_MASK_CACHE.with(|enabled| enabled.get())
            && dynamic_mask_cache_enabled();
    }
    #[cfg(not(test))]
    {
        dynamic_mask_cache_enabled()
    }
}

#[derive(Clone)]
struct TransientPath {
    arena_start: u32,
    arena_end: u32,
    acc: TerminalsDisallowed,
}

#[derive(Clone)]
struct TransientEntry {
    lexer_key: DynamicMaskLexerStateKey,
    path_start: u32,
    path_end: u32,
}

struct DynamicMaskLookupScratch {
    stack_arena: SmallVec<[u32; 128]>,
    paths: SmallVec<[TransientPath; 8]>,
    entries: SmallVec<[TransientEntry; 4]>,
}

fn cmp_terminals_disallowed(
    left: &TerminalsDisallowed,
    right: &TerminalsDisallowed,
) -> std::cmp::Ordering {
    if left.is_empty() && right.is_empty() {
        return std::cmp::Ordering::Equal;
    }
    let mut it_left = left.iter();
    let mut it_right = right.iter();
    loop {
        match (it_left.next(), it_right.next()) {
            (None, None) => return std::cmp::Ordering::Equal,
            (None, Some(_)) => return std::cmp::Ordering::Less,
            (Some(_), None) => return std::cmp::Ordering::Greater,
            (Some((state_l, terms_l)), Some((state_r, terms_r))) => {
                match state_l.cmp(state_r) {
                    std::cmp::Ordering::Equal => {}
                    non_eq => return non_eq,
                }
                let mut tit_l = terms_l.iter();
                let mut tit_r = terms_r.iter();
                loop {
                    match (tit_l.next(), tit_r.next()) {
                        (None, None) => break,
                        (None, Some(_)) => return std::cmp::Ordering::Less,
                        (Some(_), None) => return std::cmp::Ordering::Greater,
                        (Some(tl), Some(tr)) => match tl.cmp(tr) {
                            std::cmp::Ordering::Equal => {}
                            non_eq => return non_eq,
                        },
                    }
                }
            }
        }
    }
}

fn cmp_transient_entries(
    a: &TransientEntry,
    b: &TransientEntry,
    paths: &[TransientPath],
    arena: &[u32],
) -> std::cmp::Ordering {
    match a.lexer_key.cmp(&b.lexer_key) {
        std::cmp::Ordering::Equal => {}
        non_eq => return non_eq,
    }
    let paths_a = &paths[a.path_start as usize..a.path_end as usize];
    let paths_b = &paths[b.path_start as usize..b.path_end as usize];
    let mut it_a = paths_a.iter();
    let mut it_b = paths_b.iter();
    loop {
        match (it_a.next(), it_b.next()) {
            (None, None) => return std::cmp::Ordering::Equal,
            (None, Some(_)) => return std::cmp::Ordering::Less,
            (Some(_), None) => return std::cmp::Ordering::Greater,
            (Some(pa), Some(pb)) => {
                let slice_a = &arena[pa.arena_start as usize..pa.arena_end as usize];
                let slice_b = &arena[pb.arena_start as usize..pb.arena_end as usize];
                match slice_a.cmp(slice_b) {
                    std::cmp::Ordering::Equal => {}
                    non_eq => return non_eq,
                }
                match cmp_terminals_disallowed(&pa.acc, &pb.acc) {
                    std::cmp::Ordering::Equal => {}
                    non_eq => return non_eq,
                }
            }
        }
    }
}

impl DynamicMaskLookupScratch {
    fn compute_hash(&self) -> u64 {
        let mut hasher = FxHasher::default();
        self.entries.len().hash(&mut hasher);
        for entry in &self.entries {
            entry.lexer_key.hash(&mut hasher);
            let paths = &self.paths[entry.path_start as usize..entry.path_end as usize];
            paths.len().hash(&mut hasher);
            for path in paths {
                let slice = &self.stack_arena[path.arena_start as usize..path.arena_end as usize];
                // Traversal already supplies top-first values. Cache keys are
                // in-memory only: preserve this exact order in owned keys too,
                // avoiding a temporary reversed allocation for every deep path.
                slice.hash(&mut hasher);
                path.acc.len().hash(&mut hasher);
                for (state, terminals) in path.acc.iter() {
                    state.hash(&mut hasher);
                    if terminals.len() <= 16 {
                        let mut t_slice = SmallVec::<[TerminalID; 16]>::new();
                        t_slice.extend(terminals.iter().copied());
                        t_slice.as_slice().hash(&mut hasher);
                    } else {
                        let t_vec = terminals.iter().copied().collect::<Vec<_>>();
                        t_vec.as_slice().hash(&mut hasher);
                    }
                }
            }
        }
        hasher.finish()
    }

    fn matches_state(&self, stored: &DynamicMaskStateKey) -> bool {
        if self.entries.len() != stored.len() {
            return false;
        }
        for (entry, stored_entry) in self.entries.iter().zip(stored.iter()) {
            if entry.lexer_key != stored_entry.0 {
                return false;
            }
            let paths = &self.paths[entry.path_start as usize..entry.path_end as usize];
            if paths.len() != stored_entry.1.len() {
                return false;
            }
            for (path, stored_path) in paths.iter().zip(stored_entry.1.iter()) {
                let slice = &self.stack_arena[path.arena_start as usize..path.arena_end as usize];
                if slice.len() != stored_path.0.len() {
                    return false;
                }
                if slice != stored_path.0.as_slice() {
                    return false;
                }
                if path.acc.len() != stored_path.1.len() {
                    return false;
                }
                let mut it_acc = path.acc.iter();
                for (st_stored, terms_stored) in &stored_path.1 {
                    let Some((st_acc, terms_acc)) = it_acc.next() else {
                        return false;
                    };
                    if *st_acc != *st_stored {
                        return false;
                    }
                    if terms_acc.len() != terms_stored.len() {
                        return false;
                    }
                    if !terms_acc.iter().eq(terms_stored.iter()) {
                        return false;
                    }
                }
                if it_acc.next().is_some() {
                    return false;
                }
            }
        }
        true
    }

    fn to_owned_state_key(&self) -> DynamicMaskStateKey {
        let mut key = Vec::with_capacity(self.entries.len());
        for entry in &self.entries {
            let paths = &self.paths[entry.path_start as usize..entry.path_end as usize];
            let mut owned_paths = Vec::with_capacity(paths.len());
            for path in paths {
                let slice = &self.stack_arena[path.arena_start as usize..path.arena_end as usize];
                let stack = slice.to_vec();
                let mut exclusion_entries = Vec::with_capacity(path.acc.len());
                for (excluded_state, terminals) in path.acc.iter() {
                    let term_vec = terminals.iter().copied().collect::<Vec<_>>();
                    exclusion_entries.push((*excluded_state, term_vec));
                }
                owned_paths.push((stack, exclusion_entries));
            }
            key.push((entry.lexer_key, owned_paths));
        }
        key
    }
}

fn dynamic_mask_lookup_query(
    state: &ConstraintState<'_>,
) -> Option<(u64, DynamicMaskLookupScratch)> {
    dynamic_mask_lookup_query_for_vocab(state, state.constraint.dynamic_mask_vocab_for_runtime())
}

fn dynamic_mask_lookup_query_for_vocab(
    state: &ConstraintState<'_>,
    vocab: &DynamicMaskVocab,
) -> Option<(u64, DynamicMaskLookupScratch)> {
    let mut remaining = DYNAMIC_MASK_CACHE_MAX_STACKS;
    let mut scratch = DynamicMaskLookupScratch {
        stack_arena: SmallVec::new(),
        paths: SmallVec::new(),
        entries: SmallVec::new(),
    };
    let recursive = state.constraint.uses_compact_segmented_parser_runtime();
    let virtual_dense_cache_enabled = !recursive && env_flag(
        "GLRMASK_EXPERIMENT_DYNAMIC_VIRTUAL_DENSE_CACHE_KEY",
        true,
    );
    let max_token_byte_len = virtual_dense_cache_enabled.then(|| vocab.max_token_byte_len());
    let observation_cache_enabled = !recursive
        && std::env::var_os("GLRMASK_DISABLE_DYNAMIC_TERMINAL_OBSERVATION_CACHE").is_none()
            && vocab.has_terminal_observation_classes()
            && !state.constraint.tokenizer.has_any_virtual_runtime()
            // Static/dynamic composition can defer parser terminals and repair
            // component switches. Those observations are not represented by
            // the ordinary singleton parser-admission proof below.
            && state.constraint.static_dynamic_overlay.is_none();
    for (&tokenizer_state, gss) in &state.state {
        if gss.max_depth() > DYNAMIC_MASK_CACHE_MAX_DEPTH {
            return None;
        }
        let path_start = scratch.paths.len() as u32;
        let mut path_count = 0usize;
        let mut exclusions_empty = true;
        let completed = gss.for_each_stack_top_first_bounded(remaining, |top_first, acc| {
            path_count += 1;
            if !acc.is_empty() {
                exclusions_empty = false;
            }
            let arena_start = scratch.stack_arena.len() as u32;
            scratch.stack_arena.extend_from_slice(top_first);
            let arena_end = scratch.stack_arena.len() as u32;
            scratch.paths.push(TransientPath {
                arena_start,
                arena_end,
                acc: acc.clone(),
            });
        });
        if !completed {
            return None;
        }
        remaining = remaining.checked_sub(path_count)?;
        let path_end = scratch.paths.len() as u32;
        if path_end - path_start > 1 {
            scratch.paths[path_start as usize..path_end as usize].sort_unstable_by(|a, b| {
                let slice_a = &scratch.stack_arena[a.arena_start as usize..a.arena_end as usize];
                let slice_b = &scratch.stack_arena[b.arena_start as usize..b.arena_end as usize];
                slice_a.cmp(slice_b)
                    .then_with(|| cmp_terminals_disallowed(&a.acc, &b.acc))
            });
        }

        // Exact parser-relative lexer quotient. When this parser frontier admits
        // exactly one terminal, every lexer event before the first successful
        // parser advance is observable only through that terminal's
        // `(matched, possible-future)` pair. Equal precomputed exact quotient
        // classes therefore have the same next-token mask; after a finalization
        // both executions enter the same parser child and common lexer reset.
        // Recursive IDs already include the active-leaf namespace. Do not
        // interpret them using the outer tokenizer or its ordinary quotients.
        // Stack paths and correlated exclusions use the same exact key builder.
        let lexer_key = if recursive {
            DynamicMaskLexerStateKey::RecursiveExact(tokenizer_state)
        } else if virtual_dense_cache_enabled
            && let Some(coordinate) = state
                .constraint
                .tokenizer
                .virtual_residual_direct_coordinate(tokenizer_state)
            && let Some((runtime, projected_state)) = state
                .constraint
                .tokenizer
                .virtual_residual_direct_coordinate_mask_result_dense_key(
                    coordinate,
                    max_token_byte_len.unwrap_or(0),
                )
        {
            DynamicMaskLexerStateKey::VirtualDenseProjection {
                runtime,
                state: projected_state,
                initial: tokenizer_state == state.constraint.tokenizer.initial_state(),
            }
        } else if observation_cache_enabled
            && exclusions_empty
            && state.constraint.ignore_terminal.is_none_or(|ignore| {
                let ignore = ignore as usize;
                !state
                    .constraint
                    .tokenizer
                    .matched_terminal_bitset(tokenizer_state)
                    .contains(ignore)
                    && !state
                        .constraint
                        .tokenizer
                        .possible_future_terminals(tokenizer_state)
                        .contains(ignore)
            })
        {
            let admitted = state
                .constraint
                .direct_regular_admissible_terminals(gss)
                .unwrap_or_else(|| {
                    let candidates = BitSet::all(state.constraint.parser_terminal_count() as usize);
                    super::commit::exact_admitted_terminals_for_candidates(state.constraint, gss, &candidates)
                });
            let mut terminals = admitted.iter_ones();
            terminals
                .next()
                .filter(|_| terminals.next().is_none())
                .and_then(|terminal| {
                    let terminal = terminal as TerminalID;
                    vocab
                        .terminal_observation_class(terminal, tokenizer_state)
                        .map(|class| DynamicMaskLexerStateKey::TerminalObservation {
                            terminal,
                            class,
                            initial: tokenizer_state == state.constraint.tokenizer.initial_state(),
                        })
                })
                .unwrap_or_else(|| {
                    if vocab.mask_projection_tokenizer().is_some() {
                        DynamicMaskLexerStateKey::MaskProjection {
                            state: vocab.mask_runtime_state(tokenizer_state),
                            initial: tokenizer_state == state.constraint.tokenizer.initial_state(),
                        }
                    } else {
                        DynamicMaskLexerStateKey::Exact(tokenizer_state)
                    }
                })
        } else if vocab.mask_projection_tokenizer().is_some() {
            DynamicMaskLexerStateKey::MaskProjection {
                state: vocab.mask_runtime_state(tokenizer_state),
                initial: tokenizer_state == state.constraint.tokenizer.initial_state(),
            }
        } else {
            DynamicMaskLexerStateKey::Exact(tokenizer_state)
        };
        scratch.entries.push(TransientEntry {
            lexer_key,
            path_start,
            path_end,
        });
    }
    // Raw parser-state entries are sorted by tokenizer id. Observation classes
    // deliberately identify different raw ids, so canonicalize ordering after
    // replacing that coordinate and collapse redundant equivalent branches.
    if scratch.entries.len() > 1 {
        scratch.entries.sort_unstable_by(|a, b| {
            cmp_transient_entries(a, b, &scratch.paths, &scratch.stack_arena)
        });
        scratch.entries.dedup_by(|b, a| {
            cmp_transient_entries(a, b, &scratch.paths, &scratch.stack_arena)
                == std::cmp::Ordering::Equal
        });
    }
    let hash = scratch.compute_hash();
    Some((hash, scratch))
}

fn dynamic_mask_state_key(state: &ConstraintState<'_>) -> Option<DynamicMaskStateKey> {
    dynamic_mask_lookup_query(state).map(|(_, scratch)| scratch.to_owned_state_key())
}

pub(crate) fn dynamic_mask_state_has_cached_result(state: &ConstraintState<'_>) -> bool {
    if !dynamic_mask_cache_enabled() {
        return false;
    }
    let Some((hash, query)) = dynamic_mask_lookup_query(state) else {
        return false;
    };
    state
        .constraint
        .dynamic_mask_vocab_for_runtime()
        .has_cached_mask_with_predicate(hash, |candidate| query.matches_state(candidate))
}

/// Execute a recursive/provider-native composition through the same strict
/// vocabulary traversal used by ordinary dynamic masking. Only the lexer/parser
/// coordinate adapters differ. Finite epsilon-free leaves use direct mapped
/// transitions; epsilon/virtual leaves delegate to the ordinary config lexer
/// executor through a scoped adapter over this same vocabulary traversal.
pub(crate) fn try_fill_recursive_mask_shared(
    state: &ConstraintState<'_>,
    buf: &mut [u32],
) -> Result<bool, String> {
    if !state.constraint.uses_compact_segmented_parser_runtime() {
        return Ok(false);
    }
    let required = state.constraint.mask_len();
    assert!(buf.len() >= required, "mask buffer is smaller than constraint mask");
    let (mask, tail) = buf.split_at_mut(required);
    tail.fill(0);
    let vocab = state.constraint.dynamic_mask_vocab_for_runtime();
    try_fill_recursive_mask_with_vocab(state, mask, vocab)
}

/// Shared exact mask entry point. Vocabulary selection is independent of the
/// transition provider and the hot full-walk implementation. The persistent
/// cache belongs to this immutable vocabulary/binding, never to a global ID set.
fn try_fill_recursive_mask_with_vocab(
    state: &ConstraintState<'_>, mask: &mut [u32], vocab: &DynamicMaskVocab,
) -> Result<bool, String> {
    with_recursive_vocab_mask_cache(state, mask, vocab, |mask| {
        let filled = if let Some(mut transitions) = RecursiveFullWalkTransitions::new(state.constraint) {
            fill_recursive_mask_using_vocab(state, mask, &mut transitions, vocab)?
        } else {
            recursive_provider::fill_with_vocab(state, mask, vocab)?
        };
        Ok(filled)
    })
}

fn with_recursive_vocab_mask_cache(
    state: &ConstraintState<'_>,
    mask: &mut [u32],
    vocab: &DynamicMaskVocab,
    evaluate: impl FnOnce(&mut [u32]) -> Result<bool, String>,
) -> Result<bool, String> {
    let lookup_query = recursive_persistent_mask_cache_enabled()
        .then(|| dynamic_mask_lookup_query_for_vocab(state, vocab))
        .flatten();
    if let Some((hash, ref query)) = lookup_query
        && vocab.copy_cached_mask_with_predicate(
            hash,
            |candidate| query.matches_state(candidate),
            mask,
        )
    {
        return Ok(true);
    }

    let started = lookup_query
        .as_ref()
        .map(|_| std::time::Instant::now());
    let filled = evaluate(mask)?;
    if filled
        && let Some((hash, ref query)) = lookup_query
    {
        vocab.cache_mask(
            query.to_owned_state_key(),
            hash,
            mask,
            started.is_some_and(|started| started.elapsed().as_micros() < 20),
        );
    }
    Ok(filled)
}

fn fill_recursive_mask_using<T: FullWalkTransitionTable>(
    state: &ConstraintState<'_>,
    buf: &mut [u32],
    transitions: &mut T,
) -> Result<bool, String> {
    let vocab = state.constraint.dynamic_mask_vocab_for_runtime();
    fill_recursive_mask_using_vocab(state, buf, transitions, vocab)
}

fn prepare_recursive_mask_roots<T: FullWalkTransitionTable>(
    state: &ConstraintState<'_>, transitions: &mut T,
) -> Result<DynamicBranches, String> {
    let mut roots = DynamicBranches::new();
    for (&lexer_state, gss) in state.state.iter() {
        for (stacks, terminals_disallowed) in gss.partition_by_accumulator() {
            roots.push(DynamicBranch {
                // Scoped provider coordinates already own their parser/lexer
                // coupling. Ordinary flattened-root optimizations must not
                // infer a separate projection from the coordinator shell.
                parser_filtered_root: false,
                parser_filtered_transparent: false,
                residual_continuation_terminal: None,
                tokenizer_config: transitions.root_state_for_parser(lexer_state, &stacks)?,
                // Scoped recursive states are not source states of the outer
                // tokenizer. Keeping this None also disables ordinary-only
                // residual/slice proofs until provider-native proofs exist.
                exact_tokenizer_state: None,
                gss: stacks,
                shared_root_admission: None,
                initial_prune_guard: InitialPruneGuard::new_recursive(
                    &terminals_disallowed,
                ),
            });
        }
    }
    Ok(roots)
}

fn fill_recursive_mask_using_vocab<T: FullWalkTransitionTable>(
    state: &ConstraintState<'_>, buf: &mut [u32], transitions: &mut T,
    vocab: &DynamicMaskVocab,
) -> Result<bool, String> {
    let roots = prepare_recursive_mask_roots(state, transitions)?;
    fill_recursive_mask_using_roots(state, buf, transitions, vocab, &roots)
}

fn fill_recursive_mask_using_roots<T: FullWalkTransitionTable>(
    state: &ConstraintState<'_>, buf: &mut [u32], transitions: &mut T,
    vocab: &DynamicMaskVocab, roots: &DynamicBranches,
) -> Result<bool, String> {
    let required = state.constraint.mask_len();
    assert!(buf.len() >= required, "mask buffer is smaller than constraint mask");
    let (buf, tail) = buf.split_at_mut(required);
    tail.fill(0);
    let trie = vocab.trie.as_ref();
    if roots.is_empty() {
        buf.fill(0);
        update_special_token_mask(state, buf);
        state.clear_late_grammar_placeholder_mask(buf);
        return Ok(true);
    }
    let filled = if roots.len() == 1 {
        try_full_walk_mask_with_table_dispatch::<_, true>(
            state,
            vocab,
            trie,
            roots,
            buf,
            transitions,
        )
    } else {
        try_full_walk_mask_with_table_dispatch::<_, false>(
            state,
            vocab,
            trie,
            roots,
            buf,
            transitions,
        )
    }?;
    if filled && let Some(canonical) = trie.node(0).token_id {
        // A radix-root entry consumes no bytes. Current composition semantics
        // admit such an ID only through its exact special-token terminal, not
        // as a zero-progress byte alternative. Both adaptive output polarities
        // and the eager baseline can contain root aliases, so remove their byte
        // contribution before restoring independently admissible special IDs.
        // Keep this domain adjustment outside the shared hot walk and outside
        // the ordinary non-composition path.
        for &(word, bits) in vocab.token_word_masks(canonical) {
            if let Some(slot) = buf.get_mut(word as usize) { *slot &= !bits; }
        }
        update_special_token_mask(state, buf);
        state.clear_late_grammar_placeholder_mask(buf);
    }
    Ok(filled)
}

/// Necessary byte-domain metadata; an absent summary means the candidate
/// set includes special/zero-byte semantics and cannot use this byte-only proof.
#[derive(Clone, Debug)]
struct PureMaskByteDomain {
    heads: [u64; 4],
    one_byte_heads: [u64; 4],
    second_bytes: Option<Box<[[u64; 4]]>>,
    max_token_len: usize,
}

impl PureMaskByteDomain {
    fn for_ids(constraint: &Constraint, ids: &[u32]) -> Option<Self> {
        static DEPTH: OnceLock<usize> = OnceLock::new();
        let depth = *DEPTH.get_or_init(|| std::env::var("GLRMASK_PREPARED_VOCAB_PREFLIGHT_PREFIX_BYTES")
            .ok().and_then(|value| value.parse::<usize>().ok()).unwrap_or(2).clamp(1, 2));
        Self::for_ids_with_depth(constraint, ids, depth)
    }

    fn for_ids_with_depth(constraint: &Constraint, ids: &[u32], depth: usize) -> Option<Self> {
        let mut result = Self { heads: [0; 4], one_byte_heads: [0; 4],
            second_bytes: (depth >= 2).then(|| vec![[0u64; 4]; 256].into_boxed_slice()),
            max_token_len: 0 };
        for &id in ids {
            if constraint.has_special_token_id(id) { return None; }
            let bytes = constraint.token_bytes_for_id(id)?;
            let &first = bytes.first()?;
            result.heads[first as usize / 64] |= 1u64 << (first % 64);
            result.max_token_len = result.max_token_len.max(bytes.len());
            if let Some(&second) = bytes.get(1) {
                if let Some(table) = result.second_bytes.as_mut() {
                    table[first as usize][second as usize / 64] |= 1u64 << (second % 64);
                }
            } else {
                result.one_byte_heads[first as usize / 64] |= 1u64 << (first % 64);
            }
        }
        Some(result)
    }
}

/// Prepare the SAME provider roots as ordinary masking and prove an empty
/// byte language before asking for an expensive vocabulary. A live first byte
/// retains the provider caches and uses the existing shared byte walker.
fn fill_pure_byte_mask_using_factory<T, V, F>(
    state: &ConstraintState<'_>, buf: &mut [u32], transitions: &mut T,
    domain: PureMaskByteDomain, factory: F,
) -> Result<bool, String>
where
    T: FullWalkTransitionTable,
    V: std::ops::Deref<Target = DynamicMaskVocab>,
    F: FnOnce() -> Result<V, String>,
{
    let roots = prepare_recursive_mask_roots(state, transitions)?;
    let mut live_head = false;
    'roots: for root in &roots {
        // Ignoring the initial prune guard is a safe overapproximation. Its
        // exact original value remains attached when the full walk runs.
        for (word_index, &word) in domain.heads.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                let bit = bits.trailing_zeros();
                bits &= bits - 1;
                let byte = (word_index * 64 + bit as usize) as u8;
                let target = transitions.transition(root.tokenizer_config, byte);
                if target == u32::MAX { continue; }
                // Never interpret parser effects here. A one-byte candidate
                // or ANY finalizer after the first byte declines the optional
                // second-byte proof and uses the ordinary shared walker.
                let Some(second_bytes) = domain.second_bytes.as_ref() else {
                    live_head = true; break 'roots;
                };
                if domain.one_byte_heads[word_index] & (1u64 << bit) != 0
                    || transitions.finalizer_code(target) != u32::MAX
                { live_head = true; break 'roots; }
                for (second_word_index, &second_word) in second_bytes[byte as usize].iter().enumerate() {
                    let mut seconds = second_word;
                    while seconds != 0 {
                        let next_bit = seconds.trailing_zeros(); seconds &= seconds - 1;
                        let next_byte = (second_word_index * 64 + next_bit as usize) as u8;
                        if transitions.transition(target, next_byte) != u32::MAX {
                            live_head = true; break 'roots;
                        }
                    }
                }
            }
        }
    }
    transitions.validate_mask_result()?;
    if !live_head {
        // Domain contains exclusively nonempty ordinary byte tokens. Any
        // provider error must still be checked by the caller's finish().
        buf.fill(0);
        return Ok(true);
    }
    let vocab = factory()?;
    if vocab.max_token_byte_len() > domain.max_token_len {
        return Err("lazy vocabulary exceeds its certified byte horizon".to_owned());
    }
    if transitions.pre_admitted_original_mask().is_some() {
        let result = fill_recursive_mask_using_roots(state, buf, transitions, &vocab, &roots);
        transitions.validate_mask_result()?;
        return result;
    }
    with_recursive_vocab_mask_cache(state, buf, &vocab, |buf| {
        let result = fill_recursive_mask_using_roots(state, buf, transitions, &vocab, &roots);
        // The memo wrapper publishes only after this callback succeeds.
        transitions.validate_mask_result()?;
        result
    })
}

fn fill_pure_byte_mask_with_factory<V, F>(
    state: &ConstraintState<'_>, buf: &mut [u32],
    domain: PureMaskByteDomain, factory: F,
) -> Result<bool, String>
where
    V: std::ops::Deref<Target = DynamicMaskVocab>,
    F: FnOnce() -> Result<V, String>,
{
    if let Some(mut transitions) = RecursiveFullWalkTransitions::new(state.constraint) {
        fill_pure_byte_mask_using_factory(state, buf, &mut transitions, domain, factory)
    } else {
        recursive_provider::fill_with_vocab_factory(state, buf, domain, factory)
    }
}

fn fill_recursive_mask_additive_vocab(
    state: &ConstraintState<'_>, buf: &mut [u32], vocab: &DynamicMaskVocab, already: &[u32],
) -> Result<bool, String> {
    if let Some(mut transitions) = RecursiveFullWalkTransitions::new(state.constraint) {
        transitions.pre_admitted_original_mask = Some(already);
        fill_recursive_mask_using_vocab(state, buf, &mut transitions, vocab)
    } else {
        recursive_provider::fill_with_vocab_additive(state, buf, vocab, already)
    }
}

fn fill_recursive_mask_additive_factory<V, F>(
    state: &ConstraintState<'_>, buf: &mut [u32], domain: PureMaskByteDomain,
    already: &[u32], factory: F,
) -> Result<bool, String>
where V: std::ops::Deref<Target = DynamicMaskVocab>, F: FnOnce() -> Result<V, String> {
    if let Some(mut transitions) = RecursiveFullWalkTransitions::new(state.constraint) {
        transitions.pre_admitted_original_mask = Some(already);
        fill_pure_byte_mask_using_factory(state, buf, &mut transitions, domain, factory)
    } else {
        recursive_provider::fill_with_vocab_factory_additive(state, buf, domain, already, factory)
    }
}

fn prepared_vocabulary_view_capacity() -> usize {
    static CAPACITY: OnceLock<usize> = OnceLock::new();
    *CAPACITY.get_or_init(|| std::env::var("GLRMASK_PREPARED_VOCAB_VIEW_CAPACITY")
        .ok().and_then(|value| value.parse::<usize>().ok()).unwrap_or(64).clamp(1, 256))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PreparedVocabularyPolicy { Eager, LazyOnly, DeadHeadPreflight }

/// A caller-selected vocabulary for the ordinary shared dynamic walk. The
/// semantic byte/GLR executor is unchanged. This object is runtime-derived and
/// tied to the immutable constraint whose token IDs and parser states it uses.
#[derive(Debug)]
pub(crate) struct PreparedMaskVocabulary {
    vocab: OnceLock<Result<Arc<DynamicMaskVocab>, String>>,
    policy: PreparedVocabularyPolicy,
    original_ids: Vec<u32>,
    max_original_byte_len: usize,
    domain_words: Vec<(usize, u32)>,
    // Views omit only tokens already admitted by the caller. They change data,
    // not semantics or the walker. Bounded, binding-local, no global ID cache.
    views: Mutex<FxHashMap<Vec<u32>, Arc<DynamicMaskVocab>>>,
}

impl PreparedMaskVocabulary {
    fn make_vocab(constraint: &Constraint, ids: &[u32]) -> Result<Arc<DynamicMaskVocab>, String> {
        static PACKED: OnceLock<bool> = OnceLock::new();
        if *PACKED.get_or_init(|| env_flag("GLRMASK_BOUNDARY_PACKED_VOCAB", true)) {
            return Self::make_vocab_packed(constraint, ids);
        }
        static BORROW_BYTES: OnceLock<bool> = OnceLock::new();
        if *BORROW_BYTES.get_or_init(|| std::env::var("GLRMASK_PREPARED_VOCAB_BORROW_BYTES")
            .is_ok_and(|value| matches!(value.as_str(), "1" | "true")))
        {
            Self::make_vocab_borrowed(constraint, ids)
        } else {
            Self::make_vocab_owned_reference(constraint, ids)
        }
    }

    fn make_vocab_packed(constraint: &Constraint, ids: &[u32]) -> Result<Arc<DynamicMaskVocab>, String> {
        let mut entries = Vec::<(u32, &[u8])>::with_capacity(ids.len());
        for &id in ids {
            if let Some(bytes) = constraint.token_bytes_for_id(id) { entries.push((id, bytes)); }
            else if !constraint.has_special_token_id(id) {
                return Err(format!("candidate token {id} has no byte or special-token semantics"));
            }
        }
        entries.sort_unstable_by(|a,b| a.1.cmp(b.1).then_with(|| a.0.cmp(&b.0)));
        let mut refs = Vec::<(u16, usize, &[u8])>::with_capacity(entries.len());
        let mut offsets = Vec::<u32>::with_capacity(entries.len() + 1);
        let mut originals = Vec::<u32>::with_capacity(entries.len());
        for (id, bytes) in entries {
            if refs.last().is_none_or(|last| last.2 != bytes) {
                offsets.push(u32::try_from(originals.len()).map_err(|_| "too many vocabulary aliases")?);
                refs.push((0, refs.len(), bytes));
            }
            originals.push(id);
        }
        offsets.push(u32::try_from(originals.len()).map_err(|_| "too many vocabulary aliases")?);
        let trie = Arc::new(DynamicMaskTrie::from_partitioned_token_refs(&refs));
        Ok(Arc::new(DynamicMaskVocab::from_materialized_flat_ordered(
            trie, Arc::new(offsets), Arc::new(originals),
        ).with_mask_cache_entry_cap(64)))
    }

    /// Canonicalize immutable byte references. The resulting trie owns its
    /// edge bytes, and aliases own their original IDs; no borrowed reference
    /// escapes preparation. Selection and runtime traversal stay unchanged.
    fn make_vocab_borrowed(constraint: &Constraint, ids: &[u32]) -> Result<Arc<DynamicMaskVocab>, String> {
        let mut entries = Vec::<(u32, &[u8])>::with_capacity(ids.len());
        for &token in ids {
            if let Some(bytes) = constraint.token_bytes_for_id(token) {
                entries.push((token, bytes));
            } else if !constraint.has_special_token_id(token) {
                return Err(format!("candidate token {token} has no byte or special-token semantics"));
            }
        }
        entries.sort_unstable_by(|a,b| a.1.cmp(b.1).then_with(|| a.0.cmp(&b.0)));
        let mut bytes = Vec::<&[u8]>::with_capacity(entries.len());
        let mut aliases = Vec::<Vec<u32>>::with_capacity(entries.len());
        for (id, token) in entries {
            if bytes.last().is_some_and(|last| *last == token) {
                aliases.last_mut().expect("alias row exists").push(id);
            } else { bytes.push(token); aliases.push(vec![id]); }
        }
        let refs = bytes.iter().enumerate().map(|(i,b)| (0u16, i, *b)).collect::<Vec<_>>();
        let trie = Arc::new(DynamicMaskTrie::from_partitioned_token_refs(&refs));
        Ok(Arc::new(DynamicMaskVocab::from_materialized_ordered(trie, Arc::new(aliases))
            .with_mask_cache_entry_cap(64)))
    }

    fn make_vocab_owned_reference(constraint: &Constraint, ids: &[u32]) -> Result<Arc<DynamicMaskVocab>, String> {
        let mut entries = Vec::<(u32, Vec<u8>)>::with_capacity(ids.len());
        for &token in ids {
            if let Some(bytes) = constraint.token_bytes_for_id(token) {
                // A token can have both byte and special-ID paths; preserve
                // their union exactly as the ordinary shared engine does.
                entries.push((token, bytes.to_vec()));
            } else if !constraint.has_special_token_id(token) {
                return Err(format!("candidate token {token} has no byte or special-token semantics"));
            }
        }
        entries.sort_unstable_by(|a,b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        let mut bytes = Vec::<Vec<u8>>::new();
        let mut aliases = Vec::<Vec<u32>>::new();
        for (id, token) in entries {
            if bytes.last().is_some_and(|last| *last == token) {
                aliases.last_mut().expect("alias row exists").push(id);
            } else { bytes.push(token); aliases.push(vec![id]); }
        }
        let refs = bytes.iter().enumerate().map(|(i,b)| (0u16, i, b.as_slice())).collect::<Vec<_>>();
        let trie = Arc::new(DynamicMaskTrie::from_partitioned_token_refs(&refs));
        Ok(Arc::new(DynamicMaskVocab::from_materialized_ordered(trie, Arc::new(aliases))
            .with_mask_cache_entry_cap(64)))
    }

    pub(crate) fn new(constraint: &Constraint, candidates: &[u32]) -> Result<Self, String> {
        static POLICY: OnceLock<PreparedVocabularyPolicy> = OnceLock::new();
        let policy = *POLICY.get_or_init(|| {
            if env_flag("GLRMASK_PREPARED_VOCAB_DEAD_HEAD_PREFLIGHT", true) {
                PreparedVocabularyPolicy::DeadHeadPreflight
            } else if env_flag("GLRMASK_PREPARED_VOCAB_DEFER_ONLY", false) {
                PreparedVocabularyPolicy::LazyOnly
            } else { PreparedVocabularyPolicy::Eager }
        });
        Self::new_with_policy(constraint, candidates, policy)
    }

    fn new_with_policy(
        constraint: &Constraint, candidates: &[u32], policy: PreparedVocabularyPolicy,
    ) -> Result<Self, String> {
        let mut ids = candidates.to_vec(); ids.sort_unstable(); ids.dedup();
        let mut domain_words = Vec::<(usize, u32)>::new();
        let mut max_original_byte_len = 0;
        for &token in &ids {
            max_original_byte_len = max_original_byte_len.max(
                constraint.token_bytes_for_id(token).map_or(0, <[u8]>::len));
            let word = token as usize / 32;
            if word >= constraint.mask_len() {
                return Err(format!("candidate token {token} exceeds the model vocabulary"));
            }
            // Invalid IDs remain errors even if no byte head can be consumed.
            if policy != PreparedVocabularyPolicy::Eager
                && constraint.token_bytes_for_id(token).is_none()
                && !constraint.has_special_token_id(token)
            { return Err(format!("candidate token {token} has no byte or special-token semantics")); }
            if let Some((last_word, bits)) = domain_words.last_mut() && *last_word == word
            { *bits |= 1 << (token % 32); }
            else { domain_words.push((word, 1 << (token % 32))); }
        }
        let vocab = OnceLock::new();
        if policy == PreparedVocabularyPolicy::Eager {
            let ready = Self::make_vocab(constraint, &ids)?;
            let _ = vocab.set(Ok(ready));
        }
        Ok(Self { vocab, policy, original_ids: ids, max_original_byte_len, domain_words,
            views: Mutex::new(FxHashMap::default()) })
    }

    fn cached_vocab(&self, key: &[u32]) -> Result<Option<Arc<DynamicMaskVocab>>, String> {
        if key.len() == self.original_ids.len() {
            self.vocab.get().cloned().transpose()
        } else {
            Ok(self.views.lock().unwrap_or_else(|e| e.into_inner()).get(key).cloned())
        }
    }

    fn resolve_vocab(&self, constraint: &Constraint, key: &[u32]) -> Result<Arc<DynamicMaskVocab>, String> {
        if key.len() == self.original_ids.len() {
            return self.vocab.get_or_init(|| Self::make_vocab(constraint, &self.original_ids)).clone();
        }
        if let Some(cached) = self.cached_vocab(key)? { return Ok(cached); }
        let built = Self::make_vocab(constraint, key)?;
        let mut views = self.views.lock().unwrap_or_else(|e| e.into_inner());
        if views.len() >= prepared_vocabulary_view_capacity() { views.clear(); }
        Ok(Arc::clone(views.entry(key.to_vec()).or_insert(built)))
    }

    fn or_mask_with_pre_admission(&self, state: &ConstraintState<'_>, buf: &mut [u32]) -> Result<bool, String> {
        if !state.constraint.uses_compact_segmented_parser_runtime() { return Ok(false); }
        let pending = self.original_ids.iter().copied().filter(|&id| {
            !buf.get(id as usize / 32).is_some_and(|&w| w & (1 << (id % 32)) != 0)
        }).collect::<Vec<_>>();
        if pending.is_empty() { return Ok(true); }
        let mut scratch = vec![0u32; state.constraint.mask_len()];
        let filled = if let Some(vocab) = self.cached_vocab(&self.original_ids)? {
            fill_recursive_mask_additive_vocab(state, &mut scratch, &vocab, buf)?
        } else if self.policy == PreparedVocabularyPolicy::DeadHeadPreflight
            && let Some(mut domain) = PureMaskByteDomain::for_ids(state.constraint, &pending)
        {
            // Only pending heads need the preflight, but the shared canonical
            // vocabulary may contain longer already-admitted byte strings.
            domain.max_token_len = self.max_original_byte_len;
            fill_recursive_mask_additive_factory(state, &mut scratch, domain, buf, || {
                self.resolve_vocab(state.constraint, &self.original_ids)
            })?
        } else {
            let vocab = self.resolve_vocab(state.constraint, &self.original_ids)?;
            fill_recursive_mask_additive_vocab(state, &mut scratch, &vocab, buf)?
        };
        if !filled { return Ok(false); }
        for &(word, bits) in &self.domain_words {
            if let Some(dst) = buf.get_mut(word) { *dst |= scratch[word] & bits; }
        }
        Ok(true)
    }

    pub(crate) fn or_mask(&self, state: &ConstraintState<'_>, buf: &mut [u32]) -> Result<bool, String> {
        static PRUNED: OnceLock<bool> = OnceLock::new();
        if *PRUNED.get_or_init(|| env_flag("GLRMASK_BOUNDARY_PRUNED_FIXED_VOCABULARY", true)) {
            self.or_mask_with_pre_admission(state, buf)
        } else {
            self.or_mask_filtered(state, buf)
        }
    }

    fn or_mask_filtered(&self, state: &ConstraintState<'_>, buf: &mut [u32]) -> Result<bool, String> {
        if !state.constraint.uses_compact_segmented_parser_runtime() { return Ok(false); }
        let key = self.original_ids.iter().copied().filter(|&id| {
            !buf.get(id as usize / 32).is_some_and(|w| w & (1 << (id % 32)) != 0)
        }).collect::<Vec<_>>();
        if key.is_empty() { return Ok(true); }
        let mut mask = vec![0u32; state.constraint.mask_len()];
        let profile = std::env::var_os("GLRMASK_PROFILE_DYNAMIC_BOUNDARY_CANDIDATE_TRIE").is_some();
        // Includes view selection and actual construction, not only warm walk.
        let started = profile.then(Instant::now);
        let nodes = std::cell::Cell::new(0usize);
        let filled = if let Some(vocab) = self.cached_vocab(&key)? {
            nodes.set(vocab.trie.node_count());
            try_fill_recursive_mask_with_vocab(state, &mut mask, &vocab)?
        } else if self.policy == PreparedVocabularyPolicy::DeadHeadPreflight
            && let Some(domain) = PureMaskByteDomain::for_ids(state.constraint, &key)
        {
            fill_pure_byte_mask_with_factory(state, &mut mask, domain, || {
                let vocab = self.resolve_vocab(state.constraint, &key)?;
                nodes.set(vocab.trie.node_count());
                Ok(vocab)
            })?
        } else {
            let vocab = self.resolve_vocab(state.constraint, &key)?;
            nodes.set(vocab.trie.node_count());
            try_fill_recursive_mask_with_vocab(state, &mut mask, &vocab)?
        };
        if !filled { return Ok(false); }
        for &(word, bits) in &self.domain_words {
            if let Some(dst) = buf.get_mut(word) { *dst |= mask[word] & bits; }
        }
        if let Some(started) = started {
            eprintln!("[glrmask/profile][prepared_mask_vocabulary] tokens={} nodes={} total_ns={}",
                self.domain_words.iter().map(|(_,bits)| bits.count_ones()).sum::<u32>(),
                nodes.get(), started.elapsed().as_nanos());
        }
        Ok(true)
    }

}


pub(crate) fn fill_mask_dynamic(state: &ConstraintState<'_>, buf: &mut [u32]) {
    crate::compiler::boundary_transfer::strict_static_trap_dynamic_for_state(
        "fill_mask_dynamic",
        state.constraint.uses_dynamic_runtime(),
    );
    assert!(
        !state.constraint.uses_compact_segmented_parser_runtime(),
        "unified dynamic walker cannot consume recursive provider coordinates",
    );
    fill_mask_dynamic_impl(state, buf, None, false, None)
        .expect("unbounded dynamic mask generation cannot time out");
}

pub(crate) fn fill_mask_dynamic_bounded(
    state: &ConstraintState<'_>,
    buf: &mut [u32],
    timeout_ms: u64,
) -> Result<(), String> {
    crate::compiler::boundary_transfer::strict_static_trap_dynamic_for_state(
        "fill_mask_dynamic_bounded",
        state.constraint.uses_dynamic_runtime(),
    );
    if state.constraint.uses_compact_segmented_parser_runtime() {
        return Err(
            "bounded unified dynamic walker cannot consume recursive provider coordinates"
                .to_owned(),
        );
    }
    fill_mask_dynamic_impl(
        state,
        buf,
        Some(Instant::now() + Duration::from_millis(timeout_ms)),
        false,
        None,
    )
}


/// OR the complete exact dynamic language into an existing static/component
/// baseline. The baseline does not prune traversal: the strict walker still
/// visits the complete vocabulary and combines the exact result afterward.
pub(crate) fn or_mask_dynamic_additions(state: &ConstraintState<'_>, buf: &mut [u32]) {
    crate::compiler::boundary_transfer::strict_static_trap_dynamic_for_state(
        "or_mask_dynamic_additions",
        state.constraint.uses_dynamic_runtime(),
    );
    assert!(
        !state.constraint.uses_compact_segmented_parser_runtime(),
        "unified dynamic additions cannot consume recursive provider coordinates",
    );
    fill_mask_dynamic_impl(state, buf, None, true, None)
        .expect("unbounded additive dynamic mask generation cannot time out");
}

/// OR only requested candidate bits from the complete exact dynamic result
/// into `buf`. `candidate_mask` filters the result after traversal; it never
/// restricts which vocabulary prefixes/endpoints the strict walker visits.
pub(crate) fn or_mask_dynamic_candidate_additions(
    state: &ConstraintState<'_>,
    buf: &mut [u32],
    candidate_mask: &[u32],
) {
    crate::compiler::boundary_transfer::strict_static_trap_dynamic_for_state(
        "or_mask_dynamic_candidate_additions",
        state.constraint.uses_dynamic_runtime(),
    );
    assert!(
        !state.constraint.uses_compact_segmented_parser_runtime(),
        "unified dynamic candidate additions cannot consume recursive provider coordinates",
    );
    fill_mask_dynamic_impl(state, buf, None, true, Some(candidate_mask))
        .expect("unbounded candidate dynamic mask generation cannot time out");
}

#[inline]
fn fill_mask_dynamic_impl(
    state: &ConstraintState<'_>,
    buf: &mut [u32],
    deadline: Option<Instant>,
    additive_static_baseline: bool,
    candidate_mask: Option<&[u32]>,
) -> Result<(), String> {
    let required = state.constraint.body_mask_len();
    assert!(buf.len() >= required, "mask buffer is smaller than constraint mask");
    let (buf, tail) = buf.split_at_mut(required);
    tail.fill(0);
    let mut deadline_poll = DynamicDeadlinePoll::new(deadline);
    let vocab = state.constraint.dynamic_mask_vocab_for_runtime();
    let profile = dynamic_mask_profile_enabled(state.generation);
    let profile_prewalk = dynamic_mask_prewalk_profile_enabled();
    let prewalk_profile_start = profile_prewalk.then(std::time::Instant::now);
    let total_started_at = profile.then(std::time::Instant::now);
    static CACHE_KEY_ONLY: OnceLock<bool> = OnceLock::new();
    let cache_key_only = *CACHE_KEY_ONLY
        .get_or_init(|| std::env::var_os("GLRMASK_EXPERIMENT_DYNAMIC_MASK_CACHE_KEY_ONLY").is_some());
    static CACHE_LOOKUP_ONLY: OnceLock<bool> = OnceLock::new();
    let cache_lookup_only = *CACHE_LOOKUP_ONLY
        .get_or_init(|| std::env::var_os("GLRMASK_EXPERIMENT_DYNAMIC_MASK_CACHE_LOOKUP_ONLY").is_some());
    static CACHE_MIN_STORE_US: OnceLock<u64> = OnceLock::new();
    let cache_min_store_us = *CACHE_MIN_STORE_US.get_or_init(|| {
        std::env::var("GLRMASK_EXPERIMENT_DYNAMIC_MASK_CACHE_MIN_STORE_US")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            // Materializing a Llama-sized cache payload is itself several
            // microseconds. Cheap first-use masks are better left on probation:
            // a second occurrence upgrades the exact state to a real cache
            // entry, while one-off narrow masks avoid paying more to cache the
            // result than they saved by computing it. Set the env var to 0 to
            // restore eager first-use storage.
            .unwrap_or(20)
    });

    let cache_enabled = !additive_static_baseline && dynamic_mask_cache_enabled();
    let key_started_at = profile.then(std::time::Instant::now);
    let lookup_query = cache_enabled.then(|| dynamic_mask_lookup_query(state)).flatten();
    let cache_hash = lookup_query.as_ref().map(|(hash, _)| *hash);
    let key_ms = key_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

    if !cache_key_only
        && let Some((hash, ref query)) = lookup_query
        && vocab.copy_cached_mask_with_predicate(hash, |candidate| query.matches_state(candidate), buf)
    {
        if let Some(total_started_at) = total_started_at {
            eprintln!(
                "[glrmask/profile][dynamic_mask] generation={} cache_hit=true key_ms={:.3} total_ms={:.3}",
                state.generation,
                key_ms,
                total_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        return Ok(());
    }
    let cache_miss_started = (cache_min_store_us != 0
        && lookup_query.is_some()
        && !cache_key_only
        && !cache_lookup_only)
        .then(std::time::Instant::now);

    let exact_initial_tsid = state.constraint.tokenizer.initial_state();
    let initial_tsid = vocab.mask_runtime_state(exact_initial_tsid);
    let mut root_branches = DynamicBranches::new();
    let mut lexer_scan_cache = DynamicNfaScanCache::new_for_mask(state.constraint, vocab, deadline);
    let trie = vocab.trie.as_ref();
    if profile {
        eprintln!(
            "[glrmask/profile][dynamic_mask_config] tokenizer_states={} epsilon={} exact_tokenizer_states={} projected={}",
            lexer_scan_cache.tokenizer().num_states(),
            lexer_scan_cache.tokenizer().has_epsilon_transitions(),
            state.constraint.tokenizer.num_states(),
            !std::ptr::eq(lexer_scan_cache.tokenizer(), state.constraint.tokenizer.as_ref()),
        );
        if let Ok(value) = std::env::var("GLRMASK_PROFILE_DYNAMIC_RELEVANT_TERMINALS") {
            let terminals = value
                .split(',')
                .filter_map(|value| value.trim().parse::<TerminalID>().ok())
                .map(|terminal| {
                    (
                        terminal,
                        state
                            .constraint
                            .terminal_display_name(terminal)
                            .unwrap_or("<unnamed>")
                            .to_string(),
                    )
                })
                .collect::<Vec<_>>();
            eprintln!(
                "[glrmask/profile][dynamic_relevant_terminal_names] {:?}",
                terminals
            );
        }
    }

    let mut seed_entries = SmallVec::<[(u32, u32, &ParserGSS); 16]>::new();
    // The deterministic mask projection contains only subsets reachable from
    // its own lexical start states. Parser filtering can later expose an exact
    // union of projection states that was never reachable lexically on its own.
    // In that case use the existing lazy config interner as an outer subset
    // coordinate for this mask call. Its non-deterministic mode is also exact
    // for an epsilon-free tokenizer: singleton states are tagged raw configs,
    // arbitrary unions are flattened/canonicalized, and successor rows are
    // memoized lazily per (subset, byte).
    let mut saw_missing_subset = false;
    if vocab.has_mask_subset_provenance() {
        let mut groups = SmallVec::<[(usize, &ParserGSS, SmallVec<[u32; 8]>); 8]>::new();
        for (&tokenizer_state, gss) in &state.state {
            let key = gss.ptr_key();
            if let Some((_, _, states)) = groups.iter_mut().find(|(candidate, _, _)| *candidate == key) {
                states.push(tokenizer_state);
            } else {
                groups.push((key, gss, SmallVec::from_slice(&[tokenizer_state])));
            }
        }
        for (_, gss, mut states) in groups {
            states.sort_unstable();
            states.dedup();
            if states.len() > 1 {
                if let Some(projected) = vocab.mask_runtime_state_for_source_states(
                    &state.constraint.tokenizer,
                    &states,
                ) {
                    seed_entries.push((states[0], projected, gss));
                    continue;
                }
                // Same parser object, but the parser-filtered lexer subset was
                // not materialized by compile-time determinization. The Flat16
                // walker can represent this exact frontier as one runtime
                // subset state.
                saw_missing_subset = true;
            }
            for raw_state in states {
                seed_entries.push((raw_state, vocab.mask_runtime_state(raw_state), gss));
            }
        }
    } else {
        for (&tokenizer_state, gss) in &state.state {
            seed_entries.push((
                tokenizer_state,
                vocab.mask_runtime_state(tokenizer_state),
                gss,
            ));
        }
    }
    let seed_profile_ms = prewalk_profile_start
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

    if saw_missing_subset && lexer_scan_cache.deterministic {
        // The deterministic projection contains only subsets reachable from
        // its lexical start states. Tell the Flat16 walker that parser
        // filtering exposed an exact same-parser union missing from that
        // projection. It uses the runtime subset representation for this call.
        if matches!(vocab.mask_projection_fast_transitions(), Some(FastTokenizerTransitions::Flat16 { .. })) {
            lexer_scan_cache.subset_union_requested = true;
        }
        if profile {
            eprintln!(
                "[glrmask/profile][dynamic_mask_config] subset_union_mode=true"
            );
        }
    }

    // Preserve exact parser identity across lexer-only fanout. `state.state`
    // can contain several tokenizer states carrying the same ParserGSS object.
    // Partitioning that object independently for every lexer state would create
    // distinct transformed ParserStacks objects and erase that provenance.
    // Partition once, then clone the resulting ParserStacks for each lexer
    // seed; those clones retain pointer identity by construction.
    let mut seed_groups =
        SmallVec::<[(usize, &ParserGSS, SmallVec<[(u32, u32); 8]>); 8]>::new();
    for (tokenizer_state, projected_tokenizer_state, gss) in seed_entries {
        let key = gss.ptr_key();
        if let Some((_, _, seeds)) = seed_groups
            .iter_mut()
            .find(|(candidate, _, _)| *candidate == key)
        {
            seeds.push((tokenizer_state, projected_tokenizer_state));
        } else {
            seed_groups.push((
                key,
                gss,
                SmallVec::from_slice(&[(tokenizer_state, projected_tokenizer_state)]),
            ));
        }
    }
    let mut partitioned_seed_entries =
        SmallVec::<[(u32, u32, ParserStacks, TerminalsDisallowed); 16]>::new();
    let partition_profile_start = profile_prewalk.then(std::time::Instant::now);
    for (_, gss, seeds) in seed_groups {
        deadline_poll.check()?;
        for (stacks, terminals_disallowed) in gss.partition_by_accumulator() {
            for &(tokenizer_state, projected_tokenizer_state) in &seeds {
                partitioned_seed_entries.push((
                    tokenizer_state,
                    projected_tokenizer_state,
                    stacks.clone(),
                    terminals_disallowed.clone(),
                ));
            }
        }
    }
    let partition_profile_ms = partition_profile_start
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

    let root_profile_start = profile_prewalk.then(std::time::Instant::now);
    for (tokenizer_state, projected_tokenizer_state, stacks, terminals_disallowed) in
        partitioned_seed_entries
    {
        deadline_poll.check()?;
        let initial_prune_guard = InitialPruneGuard::new(vocab, &terminals_disallowed);
            if profile {
                // Keep diagnostic parser-signature queries out of the live
                // full-walk parser cache. They must not prime or otherwise
                // alter the mask walk whose interaction transcript is measured.
                let admitted_for = |stacks: &ParserStacks| {
                    let parser_gss = with_empty_accumulators(stacks);
                    state
                        .constraint
                        .direct_regular_admissible_terminals(&parser_gss)
                        .unwrap_or_else(|| {
                            let candidates =
                                BitSet::all(state.constraint.parser_terminal_count() as usize);
                            super::commit::exact_admitted_terminals_for_candidates(
                                state.constraint,
                                &parser_gss,
                                &candidates,
                            )
                        })
                };
                let bitset_fingerprint = |bits: &BitSet| {
                    bits.words().iter().fold(0xcbf29ce484222325u64, |hash, &word| {
                        (hash ^ word).wrapping_mul(0x100000001b3)
                    })
                };
                let root_admissible = admitted_for(&stacks);
                let root_admissible_fingerprint = bitset_fingerprint(&root_admissible);
                let root_admissible_count = root_admissible.count_ones();
                if root_admissible_count <= 16 {
                    let names = root_admissible
                        .iter_ones()
                        .map(|terminal| {
                            let terminal = terminal as TerminalID;
                            (
                                terminal,
                                state.constraint.terminal_display_name(terminal).unwrap_or("<unnamed>"),
                            )
                        })
                        .collect::<Vec<_>>();
                    eprintln!(
                        "[glrmask/profile][dynamic_seed_admitted] generation={} tokenizer_state={} terminals={:?}",
                        state.generation, tokenizer_state, names
                    );
                }
                let diagnostic_terminals = std::env::var(
                    "GLRMASK_PROFILE_DYNAMIC_RELEVANT_TERMINALS",
                )
                .ok()
                .map(|value| {
                    value
                        .split(',')
                        .filter_map(|value| value.trim().parse::<TerminalID>().ok())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
                let relevant_signature = |admitted: &BitSet| {
                    diagnostic_terminals
                        .iter()
                        .copied()
                        .enumerate()
                        .fold(0u64, |mask, (index, terminal)| {
                            if index < 64 && admitted.contains(terminal as usize) {
                                mask | (1u64 << index)
                            } else {
                                mask
                            }
                        })
                };
                let root_relevant_signature = relevant_signature(&root_admissible);
                // These two optional signatures describe LR action bodies.
                // Admission, stack and post-advance diagnostics above/below
                // are backend-neutral; a template-only constraint has no LR
                // body to print and must never reconstruct or dereference one.
                let diagnostic_lr_available = !state.constraint.has_template_parser()
                    && !state.constraint.uses_compact_segmented_parser_runtime();
                let top_action_fingerprint = diagnostic_lr_available
                .then(|| stacks.single_top_value())
                .flatten()
                .map(|top| {
                    let mut hasher = rustc_hash::FxHasher::default();
                    top.hash(&mut hasher);
                    for terminal in 0..state.constraint.parser_terminal_count() {
                        terminal.hash(&mut hasher);
                        state
                            .constraint
                            .table
                            .action(top, terminal)
                            .hash(&mut hasher);
                    }
                    hasher.finish()
                });
                let mut futures = state
                    .constraint
                    .tokenizer
                    .possible_future_terminals_iter(tokenizer_state);
                let sole_future = futures.next().filter(|_| futures.next().is_none());
                let sole_future_action = diagnostic_lr_available
                .then_some(sole_future)
                .flatten()
                .and_then(|terminal| {
                    stacks.single_top_value().map(|top| {
                        (
                            terminal,
                            top,
                            state.constraint.table.action(top, terminal).cloned(),
                        )
                    })
                });
                let post_first = sole_future.and_then(|terminal| {
                    parser_child(state.constraint, &stacks, terminal).map(|child| {
                        let admitted = admitted_for(&child);
                        (
                            terminal,
                            bitset_fingerprint(&admitted),
                            admitted.count_ones(),
                            relevant_signature(&admitted),
                            child.single_top_value(),
                            (admitted.count_ones() <= 16).then(|| {
                                admitted
                                    .iter_ones()
                                    .map(|terminal| terminal as TerminalID)
                                    .collect::<Vec<_>>()
                            }),
                        )
                    })
                });
                eprintln!(
                    "[glrmask/profile][dynamic_seed_parser_signature] generation={} tokenizer_state={} root_admissible={:016x} root_admissible_count={} relevant_signature={:016x} top_action={:?} sole_future_action={:?} post_first={:?}",
                    state.generation,
                    tokenizer_state,
                    root_admissible_fingerprint,
                    root_admissible_count,
                    root_relevant_signature,
                    top_action_fingerprint,
                    sole_future_action,
                    post_first,
                );
                if let Some((stack, _)) = stacks.try_single_stack_bounded(128) {
                    eprintln!(
                        "[glrmask/profile][dynamic_seed_stack] tokenizer_state={} depth={} bottom_first={:?}",
                        tokenizer_state,
                        stack.len(),
                        stack,
                    );
                }
                let matched_named = state
                    .constraint
                    .tokenizer
                    .matched_terminals_iter(tokenizer_state)
                    .map(|terminal| (terminal, state.constraint.terminal_display_name(terminal).unwrap_or("<unnamed>")))
                    .collect::<Vec<_>>();
                let future_named = state
                    .constraint
                    .tokenizer
                    .possible_future_terminals_iter(tokenizer_state)
                    .take(16)
                    .map(|terminal| (terminal, state.constraint.terminal_display_name(terminal).unwrap_or("<unnamed>")))
                    .collect::<Vec<_>>();
                eprintln!(
                    "[glrmask/profile][dynamic_seed_lexical_sets] generation={} tokenizer_state={} matched={:?} futures_first16={:?}",
                    state.generation, tokenizer_state, matched_named, future_named
                );
                if let Some(expr) = state.constraint.tokenizer.terminal_expr(79) {
                    eprintln!("[glrmask/profile][terminal79_expr] {:?}", expr);
                }
                eprintln!(
                    "[glrmask/profile][dynamic_seed] generation={} tokenizer_state={} initial={} stack_paths={} exclusions={} transitions={} matched={} futures={}",
                    state.generation,
                    tokenizer_state,
                    tokenizer_state == exact_initial_tsid,
                    stacks.path_count_at_most(1_000_000),
                    terminals_disallowed
                        .iter()
                        .map(|(_, terminals)| terminals.len())
                        .sum::<usize>(),
                    lexer_scan_cache
                        .tokenizer()
                        .transitions_from(projected_tokenizer_state)
                        .count(),
                    state
                        .constraint
                        .tokenizer
                        .matched_terminals_iter(tokenizer_state)
                        .count(),
                    state
                        .constraint
                        .tokenizer
                        .possible_future_terminals_iter(tokenizer_state)
                        .count(),
                );
            }
            let shared_root_admission = root_admission_memo(
                state.constraint, &stacks, &root_branches,
            );
            // Project fresh starts and resets, but preserve an existing
            // continuation root's exact raw coordinate. That coordinate enables
            // the retained-terminal residual scheduler and established hot paths;
            // wrapping it in a new product would trade those wins away.
            let (tokenizer_config, parser_filtered_root, parser_filtered_transparent) = if tokenizer_state == exact_initial_tsid
                && projected_tokenizer_state == initial_tsid
                // The initial maximal-munch guard is an independent token-set
                // subtraction below. Narrow the positive lexer language first
                // and leave the guard unchanged: (L intersect A) minus blocked.
                // In particular, the finalized sibling of an extendable number
                // carries a Pending guard; excluding it forfeits the main win.
                && (initial_prune_guard.is_passed()
                    || (lexer_scan_cache.parser_projection_enabled && !lexer_scan_cache.deterministic))
                && vocab.mask_runtime_tokenizer().is_none()
            {
                let row_candidates = lexer_scan_cache.parser_projection_enabled
                    .then(|| parser_row_projection_candidates(state.constraint, &stacks))
                    .flatten();
                let (mut admitted, exact_admission) = row_candidates.unwrap_or_else(|| {
                    let admitted = parser_admission_with_root_memo(
                        state.constraint, &stacks, &shared_root_admission,
                    );
                    (admitted, true)
                });
                if let Some(ignore) = state.constraint.ignore_terminal {
                    admitted.set(ignore as usize);
                }
                if admitted.count_ones() <= 16 {
                    if let Some(filtered) = lexer_scan_cache
                        .config_for_parser_admitted(projected_tokenizer_state, &admitted)?
                    {
                        let transparent = exact_admission
                            && lexer_scan_cache.config_terminals_subset_of(filtered, &admitted);
                        (filtered, exact_admission, transparent)
                    } else {
                        (
                            lexer_scan_cache.config_for_mask_root(projected_tokenizer_state)?,
                            false,
                            false,
                        )
                    }
                } else {
                    (
                        lexer_scan_cache.config_for_mask_root(projected_tokenizer_state)?,
                        false,
                        false,
                    )
                }
            } else {
                (
                    lexer_scan_cache.config_for_mask_root(projected_tokenizer_state)?,
                    false,
                    false,
                )
            };
            root_branches.push(DynamicBranch {
                tokenizer_config,
                exact_tokenizer_state: Some(tokenizer_state),
                parser_filtered_root,
                parser_filtered_transparent,
                residual_continuation_terminal: None,
                gss: stacks,
                shared_root_admission,
                initial_prune_guard,
            });
    }
    // A committed model token can leave both (a) a lexer continuation state
    // for a terminal that matched at the token boundary and (b) the parser-
    // advanced initial lexer state for finalizing that same terminal now.
    // Certify that relationship structurally before allowing residual-byte
    // scheduling on the continuation branch. This avoids assuming that an
    // arbitrary initial-state sibling covers the same parser alternative.
    if root_branches.len() <= 16 {
        let initial = exact_initial_tsid;
        let snapshot = root_branches
            .iter()
            .map(|branch| {
                (
                    branch.exact_tokenizer_state,
                    branch.gss.clone(),
                )
            })
            .collect::<SmallVec<[(Option<u32>, ParserStacks); 16]>>();
        for branch in &mut root_branches {
            let Some(source) = branch.exact_tokenizer_state else {
                continue;
            };
            if source == initial || !branch.initial_prune_guard.is_passed() {
                continue;
            }
            let mut certified = None::<TerminalID>;
            let mut ambiguous = false;
            for terminal in state.constraint.tokenizer.matched_terminals_iter(source) {
                if Some(terminal) == state.constraint.ignore_terminal {
                    continue;
                }
                let Some(coordinate) = state
                    .constraint
                    .tokenizer
                    .terminal_residual_direct_coordinate(source, terminal)
                else {
                    continue;
                };
                if !state
                    .constraint
                    .tokenizer
                    .terminal_residual_direct_coordinate_has_future(coordinate)
                    .unwrap_or(false)
                {
                    continue;
                }
                let Some(advanced) = parser_child(state.constraint, &branch.gss, terminal) else {
                    continue;
                };
                let sibling_exists = snapshot
                    .iter()
                    .any(|(candidate_state, candidate_gss)| {
                        *candidate_state == Some(initial) && *candidate_gss == advanced
                    });
                if !sibling_exists {
                    continue;
                }
                if certified.replace(terminal).is_some() {
                    ambiguous = true;
                    break;
                }
            }
            if !ambiguous {
                branch.residual_continuation_terminal = certified;
            }
        }
    }

    let root_profile_ms = root_profile_start
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

    if profile_prewalk {
        eprintln!(
            "[glrmask/profile][dynamic_prewalk_phases] seed_ms={:.3} partition_ms={:.3} root_ms={:.3} total_ms={:.3} roots={}",
            seed_profile_ms,
            partition_profile_ms,
            root_profile_ms,
            prewalk_profile_start
                .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
            root_branches.len(),
        );
    }

    let prewalk_ms = total_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let full_walk_started_at = profile.then(std::time::Instant::now);
    // Additive composition used to carry `repair_used` through a second trie
    // recognizer so it could enumerate only B-minus-A. The strict walker is
    // already an exact complete-vocabulary evaluator of the composed parser
    // and lexer, so compute that language directly and combine it with the
    // caller's static baseline. Candidate-only callers likewise filter the
    // complete exact result after traversal rather than pruning vocabulary
    // subtrees during traversal.
    let mut full_walk_scratch = Vec::<u32>::new();
    let full_walk_output = if additive_static_baseline || candidate_mask.is_some() {
        full_walk_scratch.resize(required, 0);
        full_walk_scratch.as_mut_slice()
    } else {
        &mut *buf
    };
    let full_walk_used = try_full_walk_mask(
        state,
        vocab,
        trie,
        &root_branches,
        &mut lexer_scan_cache,
        full_walk_output,
    )?;
    if full_walk_used {
        if !full_walk_scratch.is_empty() {
            for (word_index, dst) in buf.iter_mut().enumerate() {
                let exact = full_walk_scratch[word_index];
                let allowed = candidate_mask
                    .and_then(|candidate| candidate.get(word_index))
                    .copied()
                    .unwrap_or(u32::MAX);
                *dst |= exact & allowed;
            }
        }
        if profile {
            let full_walk_layout = match vocab.mask_projection_fast_transitions() {
                Some(FastTokenizerTransitions::Flat16 { .. }) => "flat16",
                Some(FastTokenizerTransitions::Flat32 { .. }) => "flat32",
                _ => "unknown",
            };
            eprintln!(
                "[glrmask/profile][dynamic_mask] generation={} full_walk=true layout={} tokenizer_states={} prewalk_ms={:.3} full_walk_ms={:.3} total_ms={:.3}",
                state.generation,
                full_walk_layout,
                lexer_scan_cache.tokenizer().num_states(),
                prewalk_ms,
                full_walk_started_at
                    .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
                total_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
            );
        }
        let probation_if_absent = cache_miss_started.is_some_and(|started| {
            started.elapsed().as_micros() < u128::from(cache_min_store_us)
        });
        if !cache_key_only && !cache_lookup_only
            && let Some((hash, ref query)) = lookup_query
        {
            let cache_key = query.to_owned_state_key();
            vocab.cache_mask(
                cache_key,
                hash,
                buf,
                probation_if_absent,
            );
        }
        return Ok(());
    }

    Err("strict full vocabulary walk unexpectedly declined".to_owned())
}

#[cfg(test)]
mod tests {
    /// Local canonical-fixture qualification; never silently succeeds without
    /// the explicit fixture paths. Production builds do not include this test.
    #[test]
    #[ignore = "requires the pinned canonical vocabulary and residual witness corpus"]
    fn mask_result_key_canonical_witnesses_match_independent_full_masks() {
        let corpus_path = std::env::var("GLRMASK_TEST_WITNESS_CORPUS").unwrap();
        let vocab_path = std::env::var("GLRMASK_TEST_WITNESS_VOCAB_HEX").unwrap();
        let compressed = std::fs::read(corpus_path).unwrap();
        let decoded = zstd::stream::decode_all(compressed.as_slice()).unwrap();
        let corpus: serde_json::Value = serde_json::from_slice(&decoded).unwrap();
        let hex_tokens: Vec<String> = serde_json::from_slice(&std::fs::read(vocab_path).unwrap()).unwrap();
        let max_token_len = hex_tokens.iter().map(|token| token.len() / 2).max().unwrap();
        assert_eq!(max_token_len, 128, "canonical vocabulary fixture identity");
        let vocab = Vocab::new(hex_tokens.into_iter().enumerate().map(|(id, hex)| {
            assert_eq!(hex.len() % 2, 0);
            let bytes = (0..hex.len()).step_by(2)
                .map(|offset| u8::from_str_radix(&hex[offset..offset + 2], 16).unwrap()).collect();
            (id as u32, bytes)
        }).collect());
        let mut checked = 0;
        for (suffix, example_index, generations) in [("o21073", 3u64, [64usize, 65]), ("o55337", 0, [118, 120])] {
            let problem = corpus["problems"].as_array().unwrap().iter()
                .find(|p| p["problem_id"].as_str().unwrap().ends_with(suffix)).unwrap();
            let example = problem["tests"].as_array().unwrap().iter()
                .find(|t| t["index"].as_u64() == Some(example_index)).unwrap();
            let tokens: Vec<u32> = example["token_ids"].as_array().unwrap().iter()
                .map(|t| u32::try_from(t.as_u64().unwrap()).unwrap()).collect();
            let compiled = crate::DynamicConstraint::compile(
                Grammar::json_schema(&problem["schema"].to_string()), &vocab).unwrap();
            assert!(compiled.inner.has_template_parser());
            let mut public = compiled.start();
            let mut exact = compiled.inner.start_dynamic();
            let mut keys = Vec::new();
            let mut strong = Vec::new();
            for generation in 0..=generations[1] {
                let mut mask = vec![0; compiled.mask_len()];
                public.fill_mask(&mut mask);
                assert_eq!(exact.mask(), mask, "public alternative union at {suffix}/{generation}");
                if generations.contains(&generation) {
                    let mut independent = vec![0; mask.len()];
                    exact.fill_recursive_mask_by_exact_full_walk(&mut independent);
                    assert_eq!(mask, independent, "independent full mask at {suffix}/{generation}");
                    let (hash, query) = dynamic_mask_lookup_query(&exact).expect("result key");
                    let key = query.to_owned_state_key();
                    assert!(key.iter().any(|entry| matches!(entry.0, DynamicMaskLexerStateKey::VirtualDenseProjection { .. })));
                    keys.push((hash, key));
                    let coordinates: Vec<_> = exact.state.keys().map(|&state| {
                        let coordinate = compiled.inner.tokenizer.virtual_residual_direct_coordinate(state).unwrap();
                        compiled.inner.tokenizer.virtual_residual_direct_coordinate_finite_mask_dense_key(coordinate, max_token_len).unwrap()
                    }).collect();
                    strong.push(coordinates);
                    checked += 1;
                }
                if generation < generations[1] {
                    let token = tokens[generation];
                    assert!(token_allowed(&mask, token), "fixture prefix is live {suffix}/{generation}");
                    // This API explicitly rejects multiple alternatives. The
                    // check prevents treating one inner constraint as a union.
                    public.commit_token_profiled(token).expect("single alternative");
                    exact.commit_token(token).unwrap();
                }
            }
            assert_eq!(keys[0], keys[1], "proven closed-subgraph key must recover {suffix}");
            assert_ne!(strong[0], strong[1], "compiler/transition headroom must remain stronger {suffix}");
            println!("mask_result_witness {suffix} exact_full_masks=2 result_keys_equal=true strong_keys_equal=false");
        }
        assert_eq!(checked, 4);
    }

    #[test]
    fn packed_boundary_vocab_preserves_aliases_bytes_and_complete_masks() {
        let entries = vec![(0,vec![]),(1,b"P".to_vec()),(2,b"a".to_vec()),
            (3,b"b".to_vec()),(4,b"c".to_vec()),(5,b"Q".to_vec()),
            (8,b"abQ".to_vec()),(17,b"acQ".to_vec()),(31,b"bQ".to_vec()),
            (65,b"cQ".to_vec()),(99,b"abQ".to_vec()),(4001,vec![0,255]),
            (9001,b"bad".to_vec())];
        let vocab = Vocab::new(entries.clone());
        for nullable in [false,true] {
            let parent = Constraint::compile(Grammar::glrm(
                r#"glrm 1; start root; extern grammar leaf; nt root = "P" leaf "Q";"#), &vocab).unwrap();
            let source = if nullable { r#"glrm 1; start leaf; nt leaf = ("ab" | "ac")?;"# }
                else { r#"glrm 1; start leaf; nt leaf = "ab" | "ac";"# };
            let child = crate::ConstraintSpec::builder(Grammar::glrm(source), &vocab).unwrap()
                .build().unwrap().compile_dynamic().unwrap();
            let bound = parent.bind_grammar_dynamic_boundary("leaf", child).unwrap();
            let loaded = Constraint::load(bound.save()).unwrap();
            for c in [&bound, &loaded] {
                for ids in [Vec::new(),vec![8],vec![8,99],vec![99,8,99,8],entries.iter().rev().map(|(id,_)|*id).collect()] {
                    let a = PreparedMaskVocabulary::make_vocab_owned_reference(c,&ids).unwrap();
                    let b = PreparedMaskVocabulary::make_vocab_packed(c,&ids).unwrap();
                    assert_eq!(bincode::serialize(a.trie.as_ref()).unwrap(),bincode::serialize(b.trie.as_ref()).unwrap());
                    assert_eq!(a.all_original_token_words(),b.all_original_token_words());
                    for id in 0..=a.canonical_token_count() as u32 {
                        assert_eq!(a.token_ids(id),b.token_ids(id));
                        // token_ids permits unknown IDs; word-mask access requires a valid canonical ID.
                        if id < a.canonical_token_count() as u32 {
                            assert_eq!(a.token_word_masks(id),b.token_word_masks(id));
                        }
                    }
                    assert_eq!(a.full_walk_token_markers(),b.full_walk_token_markers());
                    for prefix in [b"".as_slice(),b"P",b"Pa",b"Pab",b"Pac",b"PabQ"] {
                        let mut state=c.start();state.commit_bytes(prefix).unwrap();
                        let mut x=vec![0;c.mask_len()];let mut y=x.clone();
                        assert!(try_fill_recursive_mask_with_vocab(&state,&mut x,&a).unwrap());
                        assert!(try_fill_recursive_mask_with_vocab(&state,&mut y,&b).unwrap());
                        assert_eq!(x,y,"nullable={nullable}, ids={ids:?}, prefix={prefix:?}");
                    }
                }
                assert!(PreparedMaskVocabulary::make_vocab_packed(c,&[999999]).is_err());
            }
        }
    }

    #[test]
    fn prepared_boundary_vocabularies_keep_simultaneous_siblings_and_unknown_fallback() {
        use crate::__private::ConstraintExt;
        // Both children remain viable after Xa. Each boundary owns a different
        // vocabulary and frontier, while their lower caller stack is shared.
        let entries = vec![
            (0, Vec::new()), (1, b"X".to_vec()), (2, b"a".to_vec()),
            (3, b"b".to_vec()), (4, b"c".to_vec()), (5, b"!".to_vec()),
            (6, b"?".to_vec()), (7, b"ab!".to_vec()), (8, b"ac?".to_vec()),
            (9, b"Xaab!".to_vec()), (10, b"Xaac?".to_vec()),
            (19, b"a".to_vec()), (31, b"b!".to_vec()), (40, b"c?".to_vec()),
            (65, b"aab!".to_vec()), (99, b"aac?".to_vec()), (130, b"bad".to_vec()),
        ];
        let ids = entries.iter().map(|(id, _)| *id).collect::<Vec<_>>();
        let vocab = Vocab::new(entries);
        let parent = Constraint::compile(Grammar::glrm(
            r#"glrm 1; start root; extern grammar left; extern grammar right;
                nt root = "X" (left "!" | right "?");"#,
        ), &vocab).unwrap();
        let left = Constraint::compile(Grammar::glrm(
            r#"glrm 1; start left; t A = "a"+; nt left = A "b";"#,
        ), &vocab).unwrap();
        let right = Constraint::compile(Grammar::glrm(
            r#"glrm 1; start right; t A = "a"+; nt right = A "c";"#,
        ), &vocab).unwrap();
        let bound = parent.compose_compiled_subgrammars_dynamic(
            &[("left", &left), ("right", &right)], &vocab,
        ).unwrap();
        let saved = bound.save();
        let mut saw_simultaneous_siblings = false;
        for mode in 0..4 {
            let mut constraint = Constraint::load(saved.clone()).unwrap();
            // Native metadata, all-known conservative domains, one unknown
            // child, and all unknown. Unknown must widen, never mean empty.
            if mode != 0 {
                constraint.serialized_artifact_cache = None;
                let overlay = constraint.static_dynamic_overlay.as_mut().unwrap();
                assert_eq!(overlay.segmented_parser_components.len(), 3);
                for (owner, component) in overlay.segmented_parser_components.iter_mut().enumerate() {
                    let shard = component.boundary.as_mut().unwrap();
                    shard.candidate_tokens = if mode == 3 || (mode == 2 && owner == 2) {
                        None
                    } else {
                        Some(Arc::from(ids.clone()))
                    };
                    shard.mask_vocabulary = Default::default();
                }
                overlay.segmented_boundary_shards = overlay.segmented_parser_components.iter()
                    .filter_map(|component| component.boundary.clone()).collect();
            }
            for prefix in ["", "X", "Xa", "Xaa", "Xaab", "Xaac", "Xaab!", "Xaac?"] {
                let mut state = constraint.start();
                state.commit_bytes(prefix.as_bytes()).unwrap();
                let mut owners = BTreeSet::new();
                for gss in state.state.values() {
                    for top in gss.peek_values() {
                        if let Some((owner, _)) = constraint.compact_segmented_parser_component(top) {
                            if owner != 0 { owners.insert(owner); }
                        }
                    }
                }
                saw_simultaneous_siblings |= owners.len() >= 2;
                let mut expected = vec![0; constraint.mask_len()];
                state.fill_recursive_mask_by_exact_full_walk(&mut expected);
                for _ in 0..3 {
                    assert_eq!(state.mask(), expected, "mode={mode}, prefix={prefix:?}, owners={owners:?}");
                }
                // Check each admitted model-token route, not just the token
                // consumed by one preferred trace. Aliases remain independent.
                for &id in &ids {
                    if token_allowed(&expected, id) {
                        let mut branch = state.clone();
                        branch.commit_token(id).unwrap();
                    }
                }
            }
        }
        assert!(saw_simultaneous_siblings, "the fixture must exercise two active child owners");
    }


    #[test]
    fn prepared_mask_vocabulary_partial_baselines_and_evicted_views_are_exact() {
        let mut entries = vec![(0, Vec::new()), (1,b"X".to_vec()), (2,b"!".to_vec())];
        for n in 1..=12 {
            entries.push((10+n, format!("{}!", "a".repeat(n as usize)).into_bytes()));
        }
        entries.push((70,b"a!".to_vec()));
        entries.push((95,b"bad".to_vec()));
        let ids = entries.iter().map(|(id,_)| *id).collect::<Vec<_>>();
        let vocab = Vocab::new(entries);
        let parent = Constraint::compile(Grammar::glrm(
            r#"glrm 1; start root; extern grammar child; nt root = "X" child "!";"#,
        ), &vocab).unwrap();
        let child = Constraint::compile(Grammar::glrm(
            r#"glrm 1; start child; t A = "a"+; nt child = A;"#,
        ), &vocab).unwrap();
        let bound = parent.bind_grammar_dynamic_boundary("child",child).unwrap();
        let prepared = PreparedMaskVocabulary::new(&bound,&ids).unwrap();
        let filtered = PreparedMaskVocabulary::new(&bound,&ids).unwrap();
        let mut state = bound.start(); state.commit_bytes(b"X").unwrap();
        let mut full = vec![0;bound.mask_len()];
        state.fill_recursive_mask_by_exact_full_walk(&mut full);
        let mut seed=0xb3ea1935u32;
        // More distinct views than the bounded cache can hold. Preserve all
        // caller bits outside our domain, even padding words and invalid IDs.
        for round in 0..96 {
            let mut actual = (0..full.len()+2).map(|_| {
                seed^=seed<<13; seed^=seed>>17; seed^=seed<<5; seed
            }).collect::<Vec<_>>();
            let mut expected=actual.clone();
            for &id in &ids { expected[id as usize/32] |= full[id as usize/32] & (1<<(id%32)); }
            let mut reference_view = actual.clone();
            prepared.or_mask_with_pre_admission(&state,&mut actual).unwrap();
            filtered.or_mask_filtered(&state,&mut reference_view).unwrap();
            assert_eq!(actual,expected,"canonical partial baseline round {round}");
            assert_eq!(reference_view,expected,"filtered partial baseline round {round}");
        }
        assert!(!filtered.views.lock().unwrap().is_empty(), "reference must exercise filtered vocabulary storage");
        assert!(prepared.views.lock().unwrap().is_empty(), "canonical path must not build partial vocabularies");
        assert!(PreparedMaskVocabulary::new(&bound,&[7]).is_err(),"missing in-range ID is not an empty token");
        assert!(PreparedMaskVocabulary::new(&bound,&[u32::MAX]).is_err());
    }


    #[test]
    fn pre_admitted_subtree_pruning_matches_views_and_does_not_poison_full_cache() {
        let entries = vec![(0,b"P".to_vec()),(2,b"a".to_vec()),(5,b"b".to_vec()),(7,b"c".to_vec()),
            (9,b"Q".to_vec()),(11,b"abQ".to_vec()),(13,b"acQ".to_vec()),(17,b"bQ".to_vec()),
            (21,b"cQ".to_vec()),(24,b"x".to_vec()),(31,vec![]),(41,vec![0,255]),(97,b"abQ".to_vec())];
        let ids = entries.iter().map(|(i,_)| *i).collect::<Vec<_>>();
        let vocab = Vocab::new(entries);
        for nullable in [false,true] {
            let parent = Constraint::compile(Grammar::glrm(
                r#"glrm 1; start root; extern grammar leaf; nt root = "P" leaf "Q";"#), &vocab).unwrap();
            let source = if nullable { r#"glrm 1; start x; nt x = ("ab" | "ac")?;"# }
                else { r#"glrm 1; start x; nt x = "ab" | "ac";"# };
            let child = crate::ConstraintSpec::builder(Grammar::glrm(source), &vocab).unwrap()
                .build().unwrap().compile_dynamic().unwrap();
            let bound = parent.bind_grammar_dynamic_boundary("leaf", child).unwrap();
            for prefix in [b"".as_slice(),b"P",b"Pa",b"Pab",b"Pac",b"PabQ"] {
                let mut state = bound.start(); state.commit_bytes(prefix).unwrap();
                let reference = PreparedMaskVocabulary::new_with_policy(&bound, &ids, PreparedVocabularyPolicy::DeadHeadPreflight).unwrap();
                let reusable = PreparedMaskVocabulary::new_with_policy(&bound, &ids, PreparedVocabularyPolicy::DeadHeadPreflight).unwrap();
                for seed in 0..7u32 {
                    let initial = (0..bound.mask_len()).map(|w| match seed {
                        0=>0,1=>u32::MAX,2=>0x55555555,3=>0xaaaaaaaa,
                        _=>(w as u32).wrapping_mul(1664525).wrapping_add(seed*101390422),
                    }).collect::<Vec<_>>();
                    let mut a = initial.clone(); let mut b = initial;
                    assert!(reference.or_mask_filtered(&state,&mut a).unwrap());
                    assert!(reusable.or_mask_with_pre_admission(&state,&mut b).unwrap());
                    assert_eq!(a,b,"nullable={nullable} prefix={prefix:?} seed={seed}");
                    let mut full_a=vec![0;bound.mask_len()];let mut full_b=full_a.clone();
                    assert!(reference.or_mask_filtered(&state,&mut full_a).unwrap());
                    assert!(reusable.or_mask_filtered(&state,&mut full_b).unwrap());
                    assert_eq!(full_a,full_b,"partial result must not poison full result cache");
                }
                assert!(reusable.views.lock().unwrap().is_empty(),"one canonical trie must suffice");
            }
        }
    }

    use super::*;
    use crate::{DynamicConstraint, Constraint as Constraint, Grammar, Vocab};
    use std::collections::BTreeSet;

    fn token_allowed(mask: &[u32], token_id: u32) -> bool {
        let word = token_id as usize / 32;
        let bit = token_id % 32;
        mask.get(word).is_some_and(|word| word & (1u32 << bit) != 0)
    }

    fn direct_mask(state: &ConstraintState<'_>) -> Vec<u32> {
        let mut mask = vec![0u32; state.constraint.body_mask_len()];
        state.fill_mask_dynamic(&mut mask);
        mask
    }

    #[test]
    fn root_output_scope_preserves_overlapping_terminals_and_cross_word_aliases() {
        struct RestoreScope(bool);
        impl Drop for RestoreScope {
            fn drop(&mut self) { TEST_ROOT_OUTPUT_SCOPE_FORCE.with(|flag| flag.set(self.0)); }
        }
        let _restore = RestoreScope(TEST_ROOT_OUTPUT_SCOPE_FORCE.with(|flag| flag.replace(true)));
        full_walk_dense::TEST_OUTPUT_SCOPE_SKIPPED.with(|count| count.set(0));
        let mut words = Vec::<Vec<u8>>::new();
        let mut layer = vec![Vec::new()];
        for _ in 0..3 {
            let mut next = Vec::new();
            for prefix in layer {
                for &byte in b"ab -" {
                    let mut word = prefix.clone();
                    word.push(byte);
                    next.push(word);
                }
            }
            words.extend(next.iter().cloned());
            layer = next;
        }
        let mut entries = words.iter().enumerate()
            .map(|(id, bytes)| (id as u32, bytes.clone())).collect::<Vec<_>>();
        entries.push((140, b"a".to_vec()));
        entries.push((201, b"a".to_vec()));
        let vocab = Vocab::new(entries);
        for separated in [false, true] {
            let groups = if separated {
                "lexer group a ::= A; lexer group b ::= B; lexer group ws ::= WS;"
            } else { "" };
            for pair in [
                "t A ::= 'a'+; t B ::= 'a'+ 'b';",
                "t A ::= 'a' 'b'; t B ::= 'a';",
                "t A ::= 'a'+; t B ::= 'a'+ 'b'?;",
            ] {
                let grammar = format!(
                    "start start; ignore WS; t WS ::= (' ' | '-')+; {groups} {pair} nt item ::= A | B; nt start ::= item item? item?;"
                );
                let static_constraint = Constraint::from_glrm_grammar(&grammar, &vocab).unwrap();
                let dynamic = DynamicConstraint::from_glrm_grammar(&grammar, &vocab).unwrap();
                for prefix in std::iter::once(&[][..]).chain(words.iter().map(Vec::as_slice)) {
                    let mut reference = static_constraint.start();
                    let mut candidate = dynamic.inner.start();
                    let reference_result = reference.commit_bytes(prefix);
                    let candidate_result = candidate.commit_bytes(prefix);
                    assert_eq!(reference_result.is_ok(), candidate_result.is_ok(),
                        "commit status differs for {prefix:?}: {grammar}");
                    let reference_mask = reference.mask();
                    let candidate_mask = direct_mask(&candidate);
                    assert_eq!(candidate_mask, reference_mask, "scope mismatch for {prefix:?}: {grammar}");
                    assert_eq!(candidate.is_accepting(), reference.is_accepting());
                    assert_eq!(token_allowed(&candidate_mask, 140), token_allowed(&candidate_mask, 201));
                }
            }
        }
        let skipped = full_walk_dense::TEST_OUTPUT_SCOPE_SKIPPED.with(|count| count.get());
        assert!(skipped > 0, "fixture must execute output-scope subtree skips, not merely decline them");
    }

    /// Check the independent oracle, natural probation/store path, and a
    /// genuine cached hit. A hit must not run the vocabulary walker again.
    fn assert_recursive_persistent_roundtrip(state: &ConstraintState<'_>) {
        let mut expected = vec![0; state.constraint.mask_len()];
        state.fill_recursive_mask_by_exact_full_walk(&mut expected);
        let mut output = vec![u32::MAX; expected.len() + 3];
        for _ in 0..2 {
            assert!(try_fill_recursive_mask_shared(state, &mut output).unwrap());
            assert_eq!(&output[..expected.len()], expected.as_slice());
            assert!(output[expected.len()..].iter().all(|&word| word == 0));
        }
        assert!(dynamic_mask_state_has_cached_result(state));
        let walks = TEST_FULL_WALK_USES.with(|count| count.get());
        output.fill(u32::MAX);
        assert!(try_fill_recursive_mask_shared(state, &mut output).unwrap());
        assert_eq!(TEST_FULL_WALK_USES.with(|count| count.get()), walks,
            "cached recursive mask must not walk the vocabulary");
        assert_eq!(&output[..expected.len()], expected.as_slice());
        assert!(output[expected.len()..].iter().all(|&word| word == 0));
    }


    #[test]
    fn packed_boundary_vocab_preserves_special_byte_union_and_sparse_aliases() {
        let entries=vec![(0,vec![]),(1,b"X".to_vec()),(2,b"a".to_vec()),
            (3,b"!".to_vec()),(7,vec![]),(19,b"a".to_vec()),(40,b"a!".to_vec()),
            (50,vec![0,255]),(71,b"Xa!".to_vec()),(83,b"aaa".to_vec())];
        let ids=entries.iter().map(|(id,_)|*id).chain([9001]).collect::<Vec<_>>();
        let vocab=Vocab::new(entries);
        let child=crate::ConstraintSpec::builder(Grammar::glrm(
            r#"glrm 1; start child; extern token MARK; nt child = MARK "a" | "a";"#),&vocab)
            .unwrap().bind_token("MARK",[7,9001]).unwrap().build().unwrap().compile().unwrap();
        let parent=Constraint::compile(Grammar::glrm(
            r#"glrm 1; start root; extern grammar child; nt root = "X" child "!";"#),&vocab).unwrap();
        let bound=parent.bind_grammar_dynamic_boundary("child",child).unwrap();
        let loaded=Constraint::load(bound.save()).unwrap();
        for constraint in [&bound,&loaded] {
            for domain in [ids.clone(),vec![],vec![19,2,2,40],vec![0,7,50],vec![9001]] {
                let owned=PreparedMaskVocabulary::make_vocab_owned_reference(constraint,&domain).unwrap();
                let borrowed=PreparedMaskVocabulary::make_vocab_packed(constraint,&domain).unwrap();
                assert_eq!(bincode::serialize(owned.trie.as_ref()).unwrap(),
                    bincode::serialize(borrowed.trie.as_ref()).unwrap(),"domain={domain:?}");
                for &canonical in owned.trie.all_subtree_tokens() {
                    assert_eq!(owned.token_ids(canonical),borrowed.token_ids(canonical));
                }
                if let Some(canonical)=owned.trie.node(0).token_id {
                    assert_eq!(owned.token_ids(canonical),borrowed.token_ids(canonical));
                }
                for prefix in ["","X","Xa","Xa!"] {
                    let mut state=constraint.start();state.commit_bytes(prefix.as_bytes()).unwrap();
                    let mut a=vec![0;constraint.mask_len()];let mut b=a.clone();
                    assert_eq!(try_fill_recursive_mask_with_vocab(&state,&mut a,&owned).unwrap(),
                        try_fill_recursive_mask_with_vocab(&state,&mut b,&borrowed).unwrap());
                    assert_eq!(a,b,"domain={domain:?} prefix={prefix:?}");
                }
            }
            assert!(PreparedMaskVocabulary::make_vocab_packed(constraint,&[8]).is_err());
            assert!(PreparedMaskVocabulary::make_vocab_owned_reference(constraint,&[8]).is_err());
        }
    }

    #[test]
    fn prepared_mask_vocabulary_borrowed_bytes_matches_owned_structure_and_aliases() {
        let entries=vec![(0,vec![]),(1,b"X".to_vec()),(2,b"a".to_vec()),
            (3,b"!".to_vec()),(7,vec![]),(19,b"a".to_vec()),(40,b"a!".to_vec()),
            (50,vec![0,255]),(71,b"Xa!".to_vec()),(83,b"aaa".to_vec())];
        let ids=entries.iter().map(|(id,_)|*id).chain([9001]).collect::<Vec<_>>();
        let vocab=Vocab::new(entries);
        let child=crate::ConstraintSpec::builder(Grammar::glrm(
            r#"glrm 1; start child; extern token MARK; nt child = MARK "a" | "a";"#),&vocab)
            .unwrap().bind_token("MARK",[7,9001]).unwrap().build().unwrap().compile().unwrap();
        let parent=Constraint::compile(Grammar::glrm(
            r#"glrm 1; start root; extern grammar child; nt root = "X" child "!";"#),&vocab).unwrap();
        let bound=parent.bind_grammar_dynamic_boundary("child",child).unwrap();
        let loaded=Constraint::load(bound.save()).unwrap();
        for constraint in [&bound,&loaded] {
            for domain in [ids.clone(),vec![],vec![19,2,2,40],vec![0,7,50],vec![9001]] {
                let owned=PreparedMaskVocabulary::make_vocab_owned_reference(constraint,&domain).unwrap();
                let borrowed=PreparedMaskVocabulary::make_vocab_borrowed(constraint,&domain).unwrap();
                assert_eq!(bincode::serialize(owned.trie.as_ref()).unwrap(),
                    bincode::serialize(borrowed.trie.as_ref()).unwrap(),"domain={domain:?}");
                for &canonical in owned.trie.all_subtree_tokens() {
                    assert_eq!(owned.token_ids(canonical),borrowed.token_ids(canonical));
                }
                if let Some(canonical)=owned.trie.node(0).token_id {
                    assert_eq!(owned.token_ids(canonical),borrowed.token_ids(canonical));
                }
                for prefix in ["","X","Xa","Xa!"] {
                    let mut state=constraint.start();state.commit_bytes(prefix.as_bytes()).unwrap();
                    let mut a=vec![0;constraint.mask_len()];let mut b=a.clone();
                    assert_eq!(try_fill_recursive_mask_with_vocab(&state,&mut a,&owned).unwrap(),
                        try_fill_recursive_mask_with_vocab(&state,&mut b,&borrowed).unwrap());
                    assert_eq!(a,b,"domain={domain:?} prefix={prefix:?}");
                }
            }
            assert!(PreparedMaskVocabulary::make_vocab_borrowed(constraint,&[8]).is_err());
            assert!(PreparedMaskVocabulary::make_vocab_owned_reference(constraint,&[8]).is_err());
        }
    }

    #[test]
    fn prepared_mask_vocabulary_two_byte_proof_never_crosses_a_finalizer() {
        let vocab=Vocab::new(vec![(0,b"P".to_vec()),(1,b"x".to_vec()),
            (2,b"xy!".to_vec()),(3,b"x!".to_vec()),(4,b"xz!".to_vec()),
            (5,b"!".to_vec()),(6,b"z".to_vec())]);
        for leaf in ["x", "xz"] {
            let source=format!("glrm 1; start child; nt child = {leaf:?};");
            let child=Constraint::compile(Grammar::glrm(&source),&vocab).unwrap();
            let parent=Constraint::compile(Grammar::glrm(
                r#"glrm 1; start root; extern grammar child; nt root = "P" child "!";"#),&vocab).unwrap();
            let bound=parent.bind_grammar_dynamic_boundary("child",child).unwrap();
            let loaded=Constraint::load(bound.save()).unwrap();
            for constraint in [&bound,&loaded] {
                let mut state=constraint.start();state.commit_bytes(b"P").unwrap();
                for id in [1,2,3,4] {
                    for depth in [1,2] {
                        let domain=PureMaskByteDomain::for_ids_with_depth(constraint,&[id],depth).unwrap();
                        let called=std::cell::Cell::new(false);
                        let mut mask=vec![0;constraint.mask_len()];
                        assert!(fill_pure_byte_mask_with_factory(&state,&mut mask,domain,||{
                            called.set(true);PreparedMaskVocabulary::make_vocab(constraint,&[id])
                        }).unwrap());
                        let expected=if leaf=="x" {id==1 || id==3} else {id==1 || id==4};
                        assert_eq!(token_allowed(&mask,id),expected,"leaf={leaf} id={id} depth={depth}");
                        if id==1 || leaf=="x" || depth==1 {
                            assert!(called.get(),"one-byte candidates and first-byte finalizers must resolve");
                        } else if id==2 || id==3 {
                            assert!(!called.get(),"two non-finalizing raw transitions prove the domain empty");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn prepared_mask_vocabulary_dead_head_propagates_deferred_errors_before_success() {
        struct Fails { calls: usize, fail_at: usize, pending: bool }
        impl FullWalkTransitionTable for Fails {
            type Cell = u32;
            fn cell(&mut self, _:u32, _:u8)->u32 {
                self.calls+=1;
                if self.calls>=self.fail_at { self.pending=true; u32::MAX } else { 0 }
            }
            fn cell_is_dead(c:u32)->bool { c==u32::MAX }
            fn cell_has_finalizer(_:u32)->bool { false }
            fn cell_target(c:u32)->u32 { c }
            fn root_state(&mut self,_:u32)->Result<u32,String> { Ok(0) }
            fn validate_mask_result(&mut self)->Result<(),String> {
                if self.pending {Err("injected transition failure".into())}else{Ok(())}
            }
            fn finalizer_code(&self,_:u32)->u32 {u32::MAX}
            fn single_finalizer_continues(&mut self,_:u32)->bool {false}
            fn matched_terminals(&self,_:u32)->SmallVec<[TerminalID;4]> {SmallVec::new()}
            fn future_contains(&mut self,_:u32,_:TerminalID)->bool {true}
            fn future_intersects(&mut self,_:u32,_:&BitSet)->bool {true}
            fn merge_states(&mut self,_:&[u32])->Option<u32> {None}
            fn dense_state_count(&self)->Option<usize> {Some(1)}
            fn token_boundary_allowed(&mut self,_:&mut FullWalkParserCache,_:&crate::runtime::artifact::Constraint,
                _:u32,_:u32,_:u32)->bool {false}
            fn exact_raw_state(&self,_:u32)->Option<u32> {None}
        }
        let vocab=Vocab::new(vec![(0,b"a".to_vec())]);
        let constraint=Constraint::compile(Grammar::glrm(r#"glrm 1; start root; nt root = "a";"#),&vocab).unwrap();
        let state=constraint.start();
        let domain=PureMaskByteDomain::for_ids(&constraint,&[0]).unwrap();
        for fail_at in [1,2] {
            let mut transitions=Fails{calls:0,fail_at,pending:false};
            let called=std::cell::Cell::new(false);
            let mut mask=vec![0;constraint.mask_len()];
            let result=fill_pure_byte_mask_using_factory(&state,&mut mask,&mut transitions,domain.clone(),||{
                called.set(true); PreparedMaskVocabulary::make_vocab(&constraint,&[0])
            });
            assert_eq!(result.unwrap_err(),"injected transition failure");
            assert_eq!(called.get(),fail_at==2);
            assert!(transitions.pending);
        }
    }

    #[test]
    fn prepared_mask_vocabulary_dead_head_skips_factory_then_builds_on_live_prefix() {
        use super::PreparedVocabularyPolicy;
        let vocab = Vocab::new(vec![
            (0, b"P".to_vec()), (1, b"x".to_vec()), (2, b"a!".to_vec()),
            (3, b"!".to_vec()), (5, b"a".to_vec()), (7, b"a!".to_vec()),
            (9, b"z".to_vec()),
        ]);
        let child = Constraint::compile(Grammar::glrm(
            r#"glrm 1; start child; nt child = "xa";"#), &vocab).unwrap();
        let parent = Constraint::compile(Grammar::glrm(
            r#"glrm 1; start root; extern grammar child; nt root = "P" child "!";"#), &vocab).unwrap();
        let bound = parent.bind_grammar_dynamic_boundary("child", child).unwrap();
        let loaded = Constraint::load(bound.save()).unwrap();
        for constraint in [&bound, &loaded] {
            let domain = [2, 7];
            let prepared = PreparedMaskVocabulary::new_with_policy(
                constraint, &domain, PreparedVocabularyPolicy::DeadHeadPreflight).unwrap();
            assert!(prepared.vocab.get().is_none());
            let mut state = constraint.start(); state.commit_bytes(b"P").unwrap();
            let mut mask = vec![0; constraint.mask_len()];
            assert!(prepared.or_mask(&state, &mut mask).unwrap());
            assert!(mask.iter().all(|&word| word == 0));
            assert!(prepared.vocab.get().is_none(), "all dead heads must not build the base");
            assert!(prepared.views.lock().unwrap().is_empty(), "nor a filtered trie");
            let proof = PureMaskByteDomain::for_ids(constraint, &domain).unwrap();
            let called = std::cell::Cell::new(false);
            assert!(fill_pure_byte_mask_with_factory(&state, &mut mask, proof, || {
                called.set(true); PreparedMaskVocabulary::make_vocab(constraint, &domain)
            }).unwrap());
            assert!(!called.get(), "supplier callback must not run for a dead domain");
            state.commit_bytes(b"x").unwrap();
            assert!(prepared.or_mask(&state, &mut mask).unwrap());
            assert!(token_allowed(&mask, 2) && token_allowed(&mask, 7));
            assert!(prepared.vocab.get().is_some(), "first live prefix must resolve the same base");
            let ready = prepared.vocab.get().unwrap().as_ref().unwrap().clone();
            mask.fill(0); assert!(prepared.or_mask(&state, &mut mask).unwrap());
            assert!(Arc::ptr_eq(&ready, prepared.vocab.get().unwrap().as_ref().unwrap()));
            for policy in [PreparedVocabularyPolicy::Eager, PreparedVocabularyPolicy::LazyOnly,
                           PreparedVocabularyPolicy::DeadHeadPreflight] {
                assert!(PreparedMaskVocabulary::new_with_policy(constraint, &[6], policy).is_err(),
                    "unknown IDs must fail even with deferred construction");
            }
        }
    }

    #[test]
    fn prepared_mask_vocabulary_dead_head_preserves_special_aliases_empty_ids_and_baselines() {
        use super::PreparedVocabularyPolicy;
        let entries = vec![
            (0, b"P".to_vec()), (1, b"x".to_vec()), (2, b"a!".to_vec()),
            (3, b"!".to_vec()), (5, b"a".to_vec()), (7, b"a!".to_vec()),
            (8, vec![]), (9, b"z".to_vec()), (20, vec![0, 255]),
        ];
        let vocab = Vocab::new(entries);
        let child = crate::ConstraintSpec::builder(Grammar::glrm(
            r#"glrm 1; start child; extern token SPECIAL; nt child = "xa" | SPECIAL;"#), &vocab)
            .unwrap().bind_token("SPECIAL", [2, 9001]).unwrap()
            .build().unwrap().compile().unwrap();
        let parent = Constraint::compile(Grammar::glrm(
            r#"glrm 1; start root; extern grammar child; nt root = "P" child "!";"#), &vocab).unwrap();
        let bound = parent.bind_grammar_dynamic_boundary("child", child).unwrap();
        let loaded = Constraint::load(bound.save()).unwrap();
        for constraint in [&bound, &loaded] {
            assert!(PureMaskByteDomain::for_ids(constraint, &[2]).is_none());
            assert!(PureMaskByteDomain::for_ids(constraint, &[9001]).is_none());
            assert!(PureMaskByteDomain::for_ids(constraint, &[8]).is_none());
            assert!(PureMaskByteDomain::for_ids(constraint, &[7]).is_some());
            for domain in [vec![2, 7, 8, 9001], vec![7], vec![2, 9001], vec![], vec![1, 3, 5, 7, 9, 20]] {
                let reference = PreparedMaskVocabulary::new_with_policy(
                    constraint, &domain, PreparedVocabularyPolicy::Eager).unwrap();
                let actual = PreparedMaskVocabulary::new_with_policy(
                    constraint, &domain, PreparedVocabularyPolicy::DeadHeadPreflight).unwrap();
                for prefix in [b"".as_slice(), b"P", b"Px", b"Pxa", b"Pxa!"] {
                    let mut state = constraint.start(); state.commit_bytes(prefix).unwrap();
                    for seed in 0..16u32 {
                        let mut a = vec![0; constraint.mask_len()];
                        for &id in &[0u32, 1, 2, 3, 5, 7, 8, 9, 20, 9001] {
                            if id.wrapping_mul(1664525).wrapping_add(seed) % 5 <= 1 {
                                a[id as usize / 32] |= 1 << (id % 32);
                            }
                        }
                        let mut b = a.clone();
                        assert!(reference.or_mask(&state, &mut a).unwrap());
                        assert!(actual.or_mask(&state, &mut b).unwrap());
                        assert_eq!(a, b, "domain={domain:?} prefix={prefix:?} seed={seed}");
                    }
                }
            }
            let mut state = constraint.start(); state.commit_bytes(b"P").unwrap();
            let prepared = PreparedMaskVocabulary::new_with_policy(constraint, &[2, 7, 9001],
                PreparedVocabularyPolicy::DeadHeadPreflight).unwrap();
            let mut mask = vec![0; constraint.mask_len()];
            assert!(prepared.or_mask(&state, &mut mask).unwrap());
            assert!(token_allowed(&mask, 2) && token_allowed(&mask, 9001));
            assert!(!token_allowed(&mask, 7), "ordinary byte alias has no special route");
            assert!(prepared.vocab.get().is_some(), "mixed special domains bypass byte-only proof");
        }
    }

    #[test]
    fn prepared_mask_vocabulary_preserves_domain_aliases_and_reload() {
        let vocab = Vocab::new(vec![
            (0, vec![]), (1, b"X".to_vec()), (7, b"a".to_vec()),
            (19, b"a".to_vec()), (22, b"!".to_vec()), (31, b"a!".to_vec()),
            (65, b"Xa!".to_vec()), (99, b"b".to_vec()), (130, b"aa".to_vec()),
        ]);
        let parent = Constraint::compile(Grammar::glrm(
            r#"glrm 1; start root; extern grammar child; nt root = "X" child "!";"#,
        ), &vocab).unwrap();
        let child = Constraint::compile(Grammar::glrm(
            r#"glrm 1; start child; t A = "a"+; nt child = A;"#,
        ), &vocab).unwrap();
        let bound = parent.bind_grammar_dynamic_boundary("child", child).unwrap();
        let loaded = Constraint::load(bound.save()).unwrap();
        for constraint in [&bound, &loaded] {
            for domain in [vec![], vec![7], vec![19,7,7], vec![0,65,31,22],
                vec![130,99,65,31,22,19,7,1,0]] {
                let prepared = PreparedMaskVocabulary::new(constraint, &domain).unwrap();
                for prefix in ["", "X", "Xa", "Xaa", "Xa!"] {
                    let mut state = constraint.start();
                    state.commit_bytes(prefix.as_bytes()).unwrap();
                    let mut full = vec![0; constraint.mask_len()];
                    state.fill_recursive_mask_by_exact_full_walk(&mut full);
                    for baseline in [vec![0; full.len()], full.clone()] {
                        let mut expected = baseline.clone();
                        for &id in &domain {
                            expected[id as usize / 32] |= full[id as usize / 32] & (1 << (id % 32));
                        }
                        for _ in 0..3 {
                            let mut actual = baseline.clone();
                            assert!(prepared.or_mask(&state, &mut actual).unwrap());
                            assert_eq!(actual, expected, "domain={domain:?} prefix={prefix:?}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn prepared_mask_vocabulary_isolated_across_bindings_and_bytes() {
        for word in ["a", "b"] {
            let vocab = Vocab::new(vec![(0,b"X".to_vec()),(1,word.as_bytes().to_vec()),
                (2,b"!".to_vec()),(3,b"a!".to_vec()),(4,b"b!".to_vec())]);
            let parent = Constraint::compile(Grammar::glrm(
                r#"glrm 1; start root; extern grammar child; nt root = "X" child "!";"#,
            ), &vocab).unwrap();
            let child = Constraint::compile(Grammar::glrm(
                &format!("glrm 1; start child; nt child = {word:?};")), &vocab).unwrap();
            let bound = parent.bind_grammar_dynamic_boundary("child", child).unwrap();
            let prepared = PreparedMaskVocabulary::new(&bound, &[1,3,4]).unwrap();
            let mut state = bound.start(); state.commit_bytes(b"X").unwrap();
            for _ in 0..3 {
                let mut actual = vec![0; bound.mask_len()];
                prepared.or_mask(&state, &mut actual).unwrap();
                assert!(token_allowed(&actual,1));
                assert_eq!(token_allowed(&actual,3), word=="a");
                assert_eq!(token_allowed(&actual,4), word=="b");
                assert!(!token_allowed(&actual,0));
            }
        }
    }

    #[test]
    fn prepared_mask_vocabulary_preserves_special_and_byte_union() {
        let vocab = Vocab::new(vec![(0,Vec::new()),(1,b"X".to_vec()),(2,b"a".to_vec()),
            (3,b"!".to_vec()),(7,Vec::new()),(31,Vec::new()),(40,b"a!".to_vec())]);
        let child = crate::ConstraintSpec::builder(Grammar::glrm(
            r#"glrm 1; start child; extern token MARK; nt child = MARK "a" | "a";"#,
        ), &vocab).unwrap().bind_token("MARK",[7]).unwrap().build().unwrap().compile().unwrap();
        let parent = Constraint::compile(Grammar::glrm(
            r#"glrm 1; start root; extern grammar child; nt root = "X" child "!";"#,
        ), &vocab).unwrap();
        let bound = parent.bind_grammar_dynamic_boundary("child",child).unwrap();
        for domain in [vec![0,7,31],vec![0,31,40],vec![7],vec![2,7,40]] {
            let prepared=PreparedMaskVocabulary::new(&bound,&domain).unwrap();
            for prefix in ["", "X", "Xa", "Xa!"] {
                let mut state=bound.start();state.commit_bytes(prefix.as_bytes()).unwrap();
                let mut full=vec![0;bound.mask_len()];state.fill_recursive_mask_by_exact_full_walk(&mut full);
                let mut expected=vec![0;full.len()];for &id in &domain {
                    expected[id as usize/32]|=full[id as usize/32]&(1<<(id%32));
                }
                for _ in 0..3 {
                    let mut actual=vec![0;full.len()];prepared.or_mask(&state,&mut actual).unwrap();
                    assert_eq!(actual,expected,"special domain={domain:?} prefix={prefix:?}");
                }
            }
        }
    }

    #[test]
    fn recursive_full_walk_transition_provider_matches_exact_leaf_execution() {
        let vocab = Vocab::new(vec![
            (0, b"X".to_vec()),
            (1, b"[".to_vec()),
            (2, b"a".to_vec()),
            (3, b"b".to_vec()),
            (4, b"c".to_vec()),
            (5, b" ".to_vec()),
            (6, b"]".to_vec()),
            (7, b"!".to_vec()),
            (8, b"abc".to_vec()),
            (9, b" a".to_vec()),
            (10, b"a]!".to_vec()),
        ]);
        let leaf = Constraint::compile(
            Grammar::glrm(
                r#"glrm 1; start leaf; ignore WS; t WS = " "+; t WORD = /[a-c]+/; nt leaf = WORD;"#,
            ),
            &vocab,
        )
        .unwrap();
        let middle = Constraint::compile(
            Grammar::glrm(
                r#"glrm 1; start middle; extern grammar leaf; nt middle = "[" leaf "]";"#,
            ),
            &vocab,
        )
        .unwrap()
        .bind_grammar_dynamic_boundary("leaf", leaf)
        .unwrap();
        let bound = Constraint::compile(
            Grammar::glrm(
                r#"glrm 1; start outer; extern grammar middle; nt outer = "X" middle "!";"#,
            ),
            &vocab,
        )
        .unwrap()
        .bind_grammar_dynamic_boundary("middle", middle)
        .unwrap();
        assert!(bound.uses_compact_segmented_parser_runtime());

        let mut provider = RecursiveFullWalkTransitions::new(&bound)
            .expect("finite recursive fixture must support shared lexer provider");
        let layout = bound.recursive_parser_layout().unwrap().unwrap();
        let mut scratch =
            crate::runtime::commit::tokenizer_scan::ReusableTokenizerExecScratch::default();
        for leaf_index in 0..layout.leaves.len() {
            let leaf = bound.recursive_leaf_constraint(leaf_index).unwrap();
            for local_state in 0..leaf.tokenizer.num_states() {
                let scoped = bound
                    .recursive_tokenizer_scoped_state(leaf_index, local_state)
                    .unwrap();
                assert_eq!(provider.root_state(scoped).unwrap(), scoped);
                for byte in u8::MIN..=u8::MAX {
                    let cell = provider.cell(scoped, byte);
                    assert!(
                        crate::runtime::commit::tokenizer_scan::execute_recursive_tokenizer_reusable(
                            &bound,
                            &[byte],
                            scoped,
                            &mut scratch,
                        ),
                        "bounded exact tokenizer execution unexpectedly declined",
                    );
                    if scratch.states.is_empty() {
                        assert!(
                            RecursiveFullWalkTransitions::cell_is_dead(cell),
                            "leaf={leaf_index} local={local_state} byte={byte}",
                        );
                        continue;
                    }
                    assert_eq!(
                        scratch.states.as_slice(),
                        &[RecursiveFullWalkTransitions::cell_target(cell)],
                        "leaf={leaf_index} local={local_state} byte={byte}",
                    );
                    assert_eq!(
                        RecursiveFullWalkTransitions::cell_has_finalizer(cell),
                        !scratch.matches.is_empty(),
                        "leaf={leaf_index} local={local_state} byte={byte}",
                    );
                    let target = RecursiveFullWalkTransitions::cell_target(cell);
                    let mut actual = provider.matched_terminals(target);
                    actual.sort_unstable();
                    actual.dedup();
                    let mut expected = scratch
                        .matches
                        .iter()
                        .map(|matched| matched.id)
                        .collect::<SmallVec<[u32; 4]>>();
                    expected.sort_unstable();
                    expected.dedup();
                    assert_eq!(
                        actual, expected,
                        "finalizers differ leaf={leaf_index} local={local_state} byte={byte}",
                    );
                }
            }
        }
    }

    #[test]
    fn recursive_shared_full_walk_matches_existing_exact_walker() {
        // Exercise the byte-walk at every capacity, rather than satisfying
        // later iterations from a persistent result produced by the first.
        struct RestoreCacheCapacity(usize, bool);
        impl Drop for RestoreCacheCapacity {
            fn drop(&mut self) {
                TEST_RECURSIVE_PRODUCT_CACHE_CAPACITY.with(|v| v.set(self.0));
                TEST_RECURSIVE_PERSISTENT_MASK_CACHE.with(|v| v.set(self.1));
            }
        }
        let _restore = RestoreCacheCapacity(
            TEST_RECURSIVE_PRODUCT_CACHE_CAPACITY.with(|v| v.get()),
            TEST_RECURSIVE_PERSISTENT_MASK_CACHE.with(|v| v.replace(false)),
        );
        let mut words = vec![Vec::<u8>::new()];
        let mut layer = vec![Vec::<u8>::new()];
        for _ in 0..3 {
            let mut next = Vec::new();
            for prefix in layer {
                for &byte in b"abcX[]! ()" {
                    let mut word = prefix.clone();
                    word.push(byte);
                    next.push(word);
                }
            }
            words.extend(next.iter().cloned());
            layer = next;
        }
        let vocab = Vocab::new(
            words
                .into_iter()
                .enumerate()
                .map(|(id, bytes)| (id as u32, bytes))
                .collect(),
        );
        let middle_source =
            r#"glrm 1; start middle; extern grammar leaf; nt middle = "[" leaf "]";"#;
        let outer_source =
            r#"glrm 1; start outer; extern grammar middle; nt outer = "X" middle "!";"#;

        for (leaf_source, prefixes) in [
            (
                r#"glrm 1; start leaf; t WORD = /[a-c]{1,4}/; nt leaf = WORD;"#,
                vec![
                    b"".as_slice(),
                    b"X",
                    b"X[",
                    b"X[a",
                    b"X[abc",
                    b"X[abc]",
                    b"X[abc]!",
                ],
            ),
            (
                r#"glrm 1; start leaf; ignore WS; t WS = " "+; t WORD = /[a-c]{1,4}/; nt leaf = WORD;"#,
                vec![
                    b"".as_slice(),
                    b"X",
                    b"X[",
                    b"X[ ",
                    b"X[ a",
                    b"X[ a ]",
                    b"X[ a ]!",
                ],
            ),
            (
                r#"glrm 1; start leaf; nt leaf = ("a")?;"#,
                vec![
                    b"".as_slice(),
                    b"X",
                    b"X[",
                    b"X[a",
                    b"X[a]",
                    b"X[a]!",
                    b"X[]",
                    b"X[]!",
                ],
            ),
            (
                r#"glrm 1; start leaf; t SHORT = "a"; t LONG = /ab?/; nt leaf = LONG | SHORT "b";"#,
                vec![b"".as_slice(), b"X[", b"X[a", b"X[ab", b"X[ab]", b"X[ab]!"],
            ),
            (
                r#"glrm 1; start leaf; nt leaf = "(" leaf ")" | "a";"#,
                vec![b"".as_slice(), b"X[", b"X[(", b"X[((a", b"X[((a)", b"X[((a))]", b"X[((a))]!"],
            ),
        ] {
            let leaf = Constraint::compile(Grammar::glrm(leaf_source), &vocab).unwrap();
            let middle = Constraint::compile(Grammar::glrm(middle_source), &vocab)
                .unwrap()
                .bind_grammar_dynamic_boundary("leaf", leaf)
                .unwrap();
            let bound = Constraint::compile(Grammar::glrm(outer_source), &vocab)
                .unwrap()
                .bind_grammar_dynamic_boundary("middle", middle)
                .unwrap();
            assert!(bound.uses_compact_segmented_parser_runtime());
            for constraint in [
                bound.clone(),
                Constraint::load(bound.save()).expect("reload recursive fixture"),
            ] {
                for prefix in &prefixes {
                    let mut state = constraint.start_dynamic();
                    state.commit_bytes(prefix).unwrap();
                    let mut reference = vec![0u32; constraint.mask_len()];
                    state.fill_recursive_mask_by_exact_full_walk(&mut reference);
                    let mut shared = vec![0u32; constraint.mask_len()];
                    for capacity in [0, 1, 2, 256, 2048] {
                        TEST_RECURSIVE_PRODUCT_CACHE_CAPACITY.with(|v| v.set(capacity));
                        assert!(
                            try_fill_recursive_mask_shared(&state, &mut shared).unwrap(),
                            "finite recursive fixture unexpectedly declined"
                        );
                        assert_eq!(
                            shared,
                            reference,
                            "shared/full recursive mask mismatch capacity={capacity} leaf={leaf_source} prefix={:?}",
                            String::from_utf8_lossy(prefix),
                        );
                        // Force the ordinary config executor even when the
                        // finite direct adapter is eligible. This checks
                        // scoped namespace/guard/reset transport independently
                        // of the fast finite representation.
                        assert!(recursive_provider::fill(&state, &mut shared).unwrap());
                        assert_eq!(shared, reference,
                            "config recursive mismatch capacity={capacity} leaf={leaf_source} prefix={prefix:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn recursive_parser_semantic_interning_preserves_exact_languages_and_exhaustion() {
        let (mut cache, _) = FullWalkParserCache::from_roots(&DynamicBranches::new(), None);
        let mut canonicalizer = RecursiveParserCanonicalizer::default();
        let left = ParserStacks::from_single_stack(vec![0, 1, 2], ());
        let left_again = ParserStacks::from_single_stack(vec![0, 1, 2], ());
        assert!(!left.ptr_eq(&left_again));
        let a = canonicalizer.intern_with_threshold(left.clone(), &mut cache, 0);
        let b = canonicalizer.intern_with_threshold(left_again.clone(), &mut cache, 0);
        assert_eq!(a, b, "different allocations of one language must share a node");
        let right = ParserStacks::from_single_stack(vec![0, 1, 3], ());
        let c = canonicalizer.intern_with_threshold(right.clone(), &mut cache, 0);
        assert_ne!(a, c, "equal tops/prefixes are not an equality proof");
        let union_a = left.merge(&right);
        let union_b = right.merge(&left_again);
        let u = canonicalizer.intern_with_threshold(union_a, &mut cache, 0);
        let v = canonicalizer.intern_with_threshold(union_b, &mut cache, 0);
        assert_eq!(u, v, "union construction order must not change the language key");
        assert_ne!(u, a);
        canonicalizer.semantic = Some(Box::new(RecursiveParserSemanticCache {
            keys: crate::ds::leveled_gss::GssSemanticKeyInterner::with_budget(1, 1, 1),
            nodes: FxHashMap::default(),
        }));
        let x = canonicalizer.intern_with_threshold(
            ParserStacks::from_single_stack(vec![0, 4, 5], ()), &mut cache, 0,
        );
        let y = canonicalizer.intern_with_threshold(
            ParserStacks::from_single_stack(vec![0, 4, 6], ()), &mut cache, 0,
        );
        assert!(canonicalizer.semantic.as_ref().unwrap().keys.is_exhausted());
        assert_ne!(x, y, "exhaustion must not collapse distinct languages into sentinel zero");
    }

    #[test]
    fn recursive_persistent_mask_cache_uses_exact_scoped_state_key() {
        struct Restore(bool);
        impl Drop for Restore {
            fn drop(&mut self) {
                TEST_RECURSIVE_PERSISTENT_MASK_CACHE.with(|enabled| enabled.set(self.0));
            }
        }
        let _restore =
            Restore(TEST_RECURSIVE_PERSISTENT_MASK_CACHE.with(|enabled| enabled.replace(true)));
        let vocab = Vocab::new(vec![
            (0, b"X".to_vec()),
            (1, b"[".to_vec()),
            (2, b"a".to_vec()),
            (3, b"b".to_vec()),
            (4, b"]".to_vec()),
            (5, b"!".to_vec()),
            (6, b"ab".to_vec()),
            (7, b"a]!".to_vec()),
        ]);
        let leaf = Constraint::compile(
            Grammar::glrm(r#"glrm 1; start leaf; t WORD = /[ab]{1,4}/; nt leaf = WORD;"#),
            &vocab,
        )
        .unwrap();
        let middle = Constraint::compile(
            Grammar::glrm(
                r#"glrm 1; start middle; extern grammar leaf; nt middle = "[" leaf "]";"#,
            ),
            &vocab,
        )
        .unwrap()
        .bind_grammar_dynamic_boundary("leaf", leaf)
        .unwrap();
        let bound = Constraint::compile(
            Grammar::glrm(
                r#"glrm 1; start outer; extern grammar middle; nt outer = "X" middle "!";"#,
            ),
            &vocab,
        )
        .unwrap()
        .bind_grammar_dynamic_boundary("middle", middle)
        .unwrap();
        let mut state = bound.start_dynamic();
        state.commit_bytes(b"X[a").unwrap();
        let (hash, query) = dynamic_mask_lookup_query(&state).expect("recursive key");
        let key = query.to_owned_state_key();
        assert!(matches!(
            key.first().map(|entry| entry.0),
            Some(DynamicMaskLexerStateKey::RecursiveExact(_))
        ));
        let mut ordinary_coordinate = key.clone();
        if let Some(entry) = ordinary_coordinate.first_mut()
            && let DynamicMaskLexerStateKey::RecursiveExact(id) = entry.0
        {
            entry.0 = DynamicMaskLexerStateKey::Exact(id);
        }
        assert_eq!(crate::runtime::artifact::dynamic_mask_state_key_hash(&key), hash);
        assert!(!query.matches_state(&ordinary_coordinate),
            "ordinary and scoped recursive coordinates must be distinguished even on hash collision");

        assert_recursive_persistent_roundtrip(&state);
        let loaded = Constraint::load(bound.save()).unwrap();
        let mut loaded_state = loaded.start_dynamic();
        loaded_state.commit_bytes(b"X[a").unwrap();
        assert_recursive_persistent_roundtrip(&loaded_state);

        // A colliding hash is only a bucket hint; exact equality remains
        // mandatory before a cached payload can be returned.
        let dyn_vocab = bound.dynamic_mask_vocab_for_runtime();
        let mut actual = vec![0; bound.mask_len()];
        assert!(!dyn_vocab.copy_cached_mask_with_predicate(
            hash, |candidate| candidate == &ordinary_coordinate, &mut actual));

        // Query-only synthetic states test key guards without asking the
        // runtime to execute artificial parser states or exclusions.
        let mut excluded = state.clone();
        for (&lexer, gss) in state.state.iter() {
            excluded.state.insert(lexer, gss.apply(|_| TerminalsDisallowed::new()
                .try_with_insert_inline(lexer, 0).unwrap()));
        }
        let (_, excluded_query) = dynamic_mask_lookup_query(&excluded).unwrap();
        assert!(!excluded_query.matches_state(&key), "exclusions are part of the exact key");

        let (&lexer, _) = state.state.iter().next().unwrap();
        let mut too_deep = state.clone();
        too_deep.state.clear();
        too_deep.state.insert(lexer, ParserGSS::from_single_stack(
            vec![0; DYNAMIC_MASK_CACHE_MAX_DEPTH as usize + 1], TerminalsDisallowed::new()));
        assert!(dynamic_mask_lookup_query(&too_deep).is_none(),
            "a depth-limited key must decline rather than cache a truncated stack");

        state.commit_bytes(b"b").unwrap();
        let (next_hash, next_query) =
            dynamic_mask_lookup_query(&state).expect("next recursive key");
        assert!(
            next_hash != hash || !next_query.matches_state(&query.to_owned_state_key()),
            "a changed exact recursive state must not alias the cached predecessor",
        );
    }

    #[test]
    fn recursive_persistent_cache_isolated_between_bindings() {
        let vocab = Vocab::new(vec![
            (0, b"X".to_vec()), (1, b"a".to_vec()), (2, b"b".to_vec()),
            (3, b"!".to_vec()), (4, b"a!".to_vec()), (5, b"b!".to_vec()),
        ]);
        let parent = Constraint::compile(Grammar::glrm(
            r#"glrm 1; start root; extern grammar child; nt root = "X" child "!";"#), &vocab).unwrap();
        let a = Constraint::compile(Grammar::glrm(
            r#"glrm 1; start leaf; nt leaf = "a";"#), &vocab).unwrap();
        let b = Constraint::compile(Grammar::glrm(
            r#"glrm 1; start leaf; nt leaf = "b";"#), &vocab).unwrap();
        let bound_a = parent.bind_grammar_dynamic_boundary("child", a).unwrap();
        let bound_b = parent.bind_grammar_dynamic_boundary("child", b).unwrap();
        let mut state_a = bound_a.start_dynamic();
        let mut state_b = bound_b.start_dynamic();
        state_a.commit_bytes(b"X").unwrap();
        state_b.commit_bytes(b"X").unwrap();
        for _ in 0..3 {
            assert_recursive_persistent_roundtrip(&state_a);
            assert_recursive_persistent_roundtrip(&state_b);
        }
        let mut mask_a = vec![0; bound_a.mask_len()];
        let mut mask_b = vec![0; bound_b.mask_len()];
        try_fill_recursive_mask_shared(&state_a, &mut mask_a).unwrap();
        try_fill_recursive_mask_shared(&state_b, &mut mask_b).unwrap();
        assert_ne!(mask_a, mask_b);
        assert!(token_allowed(&mask_a, 1) && !token_allowed(&mask_a, 2));
        assert!(token_allowed(&mask_b, 2) && !token_allowed(&mask_b, 1));
    }

    #[test]
    fn recursive_shared_config_covers_virtual_and_epsilon_leaves() {
        let words: Vec<Vec<u8>> = vec![
            b"".to_vec(), b"X".to_vec(), b"!".to_vec(), b"\"".to_vec(),
            b"x:".to_vec(), b"a".to_vec(), b"b".to_vec(), b"ab".to_vec(),
            b"aa".to_vec(), b"X\"x:".to_vec(), b"a\"!".to_vec(),
            b"\"!".to_vec(), b"Xab!".to_vec(), b" ".to_vec(),
            b"\\u00".to_vec(), b"\\\"".to_vec(), vec![0xc2], vec![0xc2, 0xa0],
        ];
        let vocab = Vocab::new(words.into_iter().enumerate().map(|(i,b)| (i as u32,b)).collect());
        let virtual_child = Constraint::from_json_schema(
            r#"{"type":"string","format":"uri","minLength":1,"maxLength":5000}"#, &vocab,
        ).unwrap();
        assert!(virtual_child.tokenizer.has_virtual_residual_runtime());
        let epsilon_child = Constraint::compile(Grammar::glrm(r#"
            start payload;
            lexer group a ::= A;
            lexer group b ::= B;
            t A ::= "a"+;
            t B ::= "ab"+;
            nt payload ::= A | B;
        "#), &vocab).unwrap();
        let parent = Constraint::compile(Grammar::glrm(
            r#"glrm 1; start root; extern grammar payload; nt root = "X" payload "!";"#,
        ), &vocab).unwrap();
        for (child, prefixes) in [
            (virtual_child, vec!["", "X", "X\"", "X\"x:", "X\"x:a", "X\"x:a\"", "X\"x:a\"!"]),
            (epsilon_child, vec!["", "X", "Xa", "Xab", "Xaa", "Xab!"]),
        ] {
            let composed = parent.bind_grammar_dynamic_boundary("payload", child).unwrap();
            let loaded = Constraint::load(composed.save()).unwrap();
            for constraint in [&composed, &loaded] {
                for prefix in &prefixes {
                    let mut state = constraint.start();
                    state.commit_bytes(prefix.as_bytes()).unwrap();
                    let mut expected = vec![0;constraint.mask_len()];
                    state.fill_recursive_mask_by_exact_full_walk(&mut expected);
                    let mut actual = vec![0;expected.len()];
                    struct CapacityGuard(usize);
                    impl Drop for CapacityGuard {
                        fn drop(&mut self) {
                            TEST_RECURSIVE_PRODUCT_CACHE_CAPACITY.with(|v| v.set(self.0));
                        }
                    }
                    let _capacity = CapacityGuard(TEST_RECURSIVE_PRODUCT_CACHE_CAPACITY.with(|v| v.get()));
                    for capacity in [0, 1, 2, 16, 256, 2048] {
                        TEST_RECURSIVE_PRODUCT_CACHE_CAPACITY.with(|v| v.set(capacity));
                        assert!(recursive_provider::fill(&state,&mut actual).unwrap());
                        assert_eq!(actual, expected, "general recursive prefix {prefix:?} capacity={capacity}");
                    }
                    assert!(try_fill_recursive_mask_shared(&state,&mut actual).unwrap());
                    assert_eq!(actual, expected, "production shared prefix {prefix:?}");
                    assert_recursive_persistent_roundtrip(&state);
                }
            }
        }
    }

    #[test]
    fn dynamic_mask_cache_key_uses_exact_mask_projection_coordinate() {
        let vocab = Vocab::new(vec![
            (0, b"\"".to_vec()),
            (1, b"a".to_vec()),
            (2, b"aa".to_vec()),
            (3, b"b".to_vec()),
        ]);
        let dynamic = DynamicConstraint::from_json_schema(
            r#"{"type":"string","maxLength":1000000000}"#,
            &vocab,
        )
        .unwrap();
        let mut state = dynamic.inner.start();
        state.commit_token(0).unwrap();
        state.commit_token(1).unwrap();
        let exact_before = state.state.clone();
        let key_before = dynamic_mask_state_key(&state).expect("cache key before commit");
        let mask_before = direct_mask(&state);

        state.commit_token(1).unwrap();
        assert_ne!(exact_before, state.state, "test must change the exact lexer coordinate");
        let key_after = dynamic_mask_state_key(&state).expect("cache key after commit");
        assert_eq!(
            key_before, key_after,
            "one-token-equivalent projected lexer states must share the persistent dynamic-mask cache key",
        );
        assert_eq!(mask_before, direct_mask(&state));
    }

    #[test]
    fn dynamic_mask_lookup_top_first_order_preserves_exact_path_keys() {
        // A literal old-orientation key is the independent reference. Include
        // empty, inline, heap-sized and maximum-depth stacks; key identity must
        // retain values at every depth and every correlated exclusion.
        let mut reference: DynamicMaskStateKey = Vec::new();
        for entry in 0..7_u32 {
            let lexer = match entry % 3 {
                0 => DynamicMaskLexerStateKey::Exact(entry),
                1 => DynamicMaskLexerStateKey::RecursiveExact(entry),
                _ => DynamicMaskLexerStateKey::MaskProjection { state: entry, initial: false },
            };
            let mut paths = Vec::new();
            for (case, depth) in [0, 1, 2, 31, 32, 33, 65, 128, 256].into_iter().enumerate() {
                let stack = (0..depth).map(|i| (i * 17 + case * 3) as u32).collect();
                let mut exclusions = Vec::new();
                if case % 3 != 0 {
                    exclusions.push((entry, (0..case * 5).map(|i| i as u32).collect()));
                    exclusions.push((entry + 100, vec![0, 255, 1024]));
                }
                paths.push((stack, exclusions));
            }
            paths.sort();
            reference.push((lexer, paths));
        }
        reference.sort();

        let mut scratch = DynamicMaskLookupScratch {
            stack_arena: SmallVec::new(), paths: SmallVec::new(), entries: SmallVec::new(),
        };
        // Reverse entry/path insertion order, and add a duplicate lexer group.
        for (lexer, paths) in reference.iter().rev().chain(reference.first()) {
            let path_start = scratch.paths.len() as u32;
            for (bottom_first, exclusions) in paths.iter().rev() {
                let arena_start = scratch.stack_arena.len() as u32;
                scratch.stack_arena.extend(bottom_first.iter().rev().copied());
                let mut acc = TerminalsDisallowed::new();
                for (lexer, terminals) in exclusions {
                    for terminal in terminals { acc = acc.with_insert(*lexer, *terminal); }
                }
                scratch.paths.push(TransientPath {
                    arena_start, arena_end: scratch.stack_arena.len() as u32, acc,
                });
            }
            let path_end = scratch.paths.len() as u32;
            scratch.paths[path_start as usize..path_end as usize].sort_unstable_by(|a, b| {
                scratch.stack_arena[a.arena_start as usize..a.arena_end as usize]
                    .cmp(&scratch.stack_arena[b.arena_start as usize..b.arena_end as usize])
                    .then_with(|| cmp_terminals_disallowed(&a.acc, &b.acc))
            });
            scratch.entries.push(TransientEntry { lexer_key: *lexer, path_start, path_end });
        }
        scratch.entries.sort_unstable_by(|a, b|
            cmp_transient_entries(a, b, &scratch.paths, &scratch.stack_arena));
        scratch.entries.dedup_by(|a, b|
            cmp_transient_entries(a, b, &scratch.paths, &scratch.stack_arena).is_eq());
        let owned = scratch.to_owned_state_key();
        assert!(scratch.matches_state(&owned));
        assert_eq!(scratch.compute_hash(), crate::runtime::artifact::dynamic_mask_state_key_hash(&owned));
        let mut old_orientation = owned.clone();
        for (_, paths) in &mut old_orientation {
            for (stack, _) in paths.iter_mut() { stack.reverse(); }
            paths.sort();
        }
        old_orientation.sort();
        assert_eq!(old_orientation, reference, "orientation cannot change exact key equivalence");

        let mut changed = owned.clone();
        let deep = changed[0].1.iter_mut().find(|p| p.0.len() == 256).unwrap();
        deep.0[128] ^= 1;
        assert!(!scratch.matches_state(&changed), "all stack values participate in equality");
        changed = owned.clone();
        let excluded = changed[0].1.iter_mut().find(|p| !p.1.is_empty()).unwrap();
        excluded.1[0].1.push(8192);
        assert!(!scratch.matches_state(&changed), "all exclusions participate in equality");
        assert!(!scratch.matches_state(&reference), "opposite orientations must not alias");
    }

    #[test]
    fn dynamic_mask_lookup_query_exact_match_with_state_key() {
        let vocab = Vocab::new(vec![
            (0, b"\"".to_vec()),
            (1, b"a".to_vec()),
            (2, b"aa".to_vec()),
            (3, b"b".to_vec()),
        ]);
        let dynamic = DynamicConstraint::from_json_schema(
            r#"{"type":"string","maxLength":1000000000}"#,
            &vocab,
        )
        .unwrap();
        let mut state = dynamic.inner.start();
        state.commit_token(0).unwrap();
        for step in [1, 2, 3, 1, 2] {
            let _ = state.commit_token(step % 4);
            let (query_hash, query) = dynamic_mask_lookup_query(&state).expect("query");
            let owned_key = dynamic_mask_state_key(&state).expect("state key");
            let owned_hash = crate::runtime::artifact::dynamic_mask_state_key_hash(&owned_key);
            assert_eq!(query_hash, owned_hash, "query hash must match owned state key hash");
            assert!(query.matches_state(&owned_key), "query must match owned state key");
            assert_eq!(query.to_owned_state_key(), owned_key, "to_owned_state_key must match");

            // Negative match test: modified owned key should not match
            let mut perturbed_key = owned_key.clone();
            if let Some(entry) = perturbed_key.first_mut() {
                entry.0 = match entry.0 {
                    DynamicMaskLexerStateKey::Exact(id) => DynamicMaskLexerStateKey::Exact(id + 9999),
                    DynamicMaskLexerStateKey::RecursiveExact(id) => {
                        DynamicMaskLexerStateKey::RecursiveExact(id + 9999)
                    }
                    DynamicMaskLexerStateKey::MaskProjection { state, initial } => {
                        DynamicMaskLexerStateKey::MaskProjection { state: state + 9999, initial }
                    }
                    DynamicMaskLexerStateKey::TerminalObservation { class, terminal, initial } => {
                        DynamicMaskLexerStateKey::TerminalObservation { class: class + 9999, terminal, initial }
                    }
                    DynamicMaskLexerStateKey::VirtualDenseProjection { runtime, state, initial } => {
                        DynamicMaskLexerStateKey::VirtualDenseProjection {
                            runtime,
                            state: state + 9999,
                            initial,
                        }
                    }
                };
                assert!(!query.matches_state(&perturbed_key), "perturbed lexer key must not match");
            }
        }
    }

    #[test]
    fn dynamic_mask_lookup_query_branching_and_exclusions() {
        let vocab = Vocab::new(vec![
            (0, b"{".to_vec()),
            (1, b"}".to_vec()),
            (2, b"\"a\"".to_vec()),
            (3, b":".to_vec()),
            (4, b"1".to_vec()),
            (5, b",".to_vec()),
            (6, b"\"b\"".to_vec()),
            (7, b"true".to_vec()),
        ]);
        let schema = r#"{"anyOf":[{"type":"object","properties":{"a":{"type":"integer"}}},{"type":"object","properties":{"b":{"type":"boolean"}}}]}"#;
        let dynamic = DynamicConstraint::from_json_schema(schema, &vocab).unwrap();
        let mut state = dynamic.inner.start();
        state.commit_token(0).unwrap(); // "{"
        let (query_hash, query) = dynamic_mask_lookup_query(&state).expect("query");
        let owned_key = dynamic_mask_state_key(&state).expect("state key");
        let owned_hash = crate::runtime::artifact::dynamic_mask_state_key_hash(&owned_key);
        assert_eq!(query_hash, owned_hash, "query hash must match owned state key hash");
        assert!(query.matches_state(&owned_key));
        assert_eq!(query.to_owned_state_key(), owned_key);

        // Perturb path to verify exact matching on multi-path / complex states
        let mut perturbed = owned_key.clone();
        if let Some((_, paths)) = perturbed.first_mut() {
            if let Some(first_path) = paths.first_mut() {
                first_path.0.push(9999);
            }
        }
        assert!(!query.matches_state(&perturbed));
    }

    #[test]
    fn dynamic_mask_lookup_query_microbench() {
        let vocab = Vocab::new(vec![
            (0, b"\"".to_vec()),
            (1, b"hello".to_vec()),
            (2, b"world".to_vec()),
            (3, b"\"".to_vec()),
        ]);
        let dynamic = DynamicConstraint::from_json_schema(
            r#"{"type":"string"}"#,
            &vocab,
        ).unwrap();
        let mut state = dynamic.inner.start();
        state.commit_token(0).unwrap();
        state.commit_token(1).unwrap();

        let iters = 20_000;

        // 1. Transient query construction
        let start = std::time::Instant::now();
        let mut sink_hash = 0u64;
        for _ in 0..iters {
            if let Some((hash, _)) = dynamic_mask_lookup_query(&state) {
                sink_hash ^= hash;
            }
        }
        let transient_ns = start.elapsed().as_nanos() as f64 / iters as f64;

        // 2. Materialized owned state key
        let start = std::time::Instant::now();
        let mut sink_len = 0usize;
        for _ in 0..iters {
            if let Some(key) = dynamic_mask_state_key(&state) {
                sink_len += key.len();
            }
        }
        let owned_ns = start.elapsed().as_nanos() as f64 / iters as f64;

        // 3. Cache lookup hit with transient predicate
        let (hash, query) = dynamic_mask_lookup_query(&state).unwrap();
        let owned_key = query.to_owned_state_key();
        let dyn_vocab = state.constraint.dynamic_mask_vocab_for_runtime();
        let mut mask_buf = vec![0u32; state.constraint.body_mask_len()];
        dyn_vocab.cache_mask(owned_key, hash, &mask_buf, false);

        let start = std::time::Instant::now();
        let mut hits = 0;
        for _ in 0..iters {
            if dyn_vocab.copy_cached_mask_with_predicate(hash, |cand| query.matches_state(cand), &mut mask_buf) {
                hits += 1;
            }
        }
        let cache_hit_ns = start.elapsed().as_nanos() as f64 / iters as f64;

        eprintln!(
            "\n[MICROBENCH] transient_query_ns={:.1}ns ({:.3}us) | owned_key_ns={:.1}ns ({:.3}us) | speedup={:.2}x | cache_hit_ns={:.1}ns ({:.3}us) | hits={}/{} (sink={}/{})\n",
            transient_ns,
            transient_ns / 1000.0,
            owned_ns,
            owned_ns / 1000.0,
            owned_ns / transient_ns,
            cache_hit_ns,
            cache_hit_ns / 1000.0,
            hits,
            iters,
            sink_hash,
            sink_len,
        );
        assert_eq!(hits, iters);
    }

    fn assert_dynamic_parity(state: &ConstraintState<'_>) {
        assert_eq!(state.mask(), direct_mask(state));
    }

    fn assert_dynamic_parity_on_reachable_states(
        constraint: &Constraint,
        max_depth: usize,
        context: &str,
    ) {
        let mut frontier = vec![(constraint.start(), Vec::<u32>::new())];
        let mut seen = BTreeSet::new();

        for depth in 0..=max_depth {
            let mut next = Vec::new();
            for (state, path) in frontier {
                if let Some(key) = dynamic_mask_state_key(&state)
                    && !seen.insert(key)
                {
                    continue;
                }

                let static_mask = state.mask();
                let dynamic_mask = direct_mask(&state);
                assert_eq!(
                    static_mask, dynamic_mask,
                    "dynamic/static mask mismatch: {context} depth={depth} path={path:?}"
                );
                if depth == max_depth {
                    continue;
                }

                for (token_id, bytes) in constraint.token_bytes_iter() {
                    let expected = token_allowed(&static_mask, token_id);
                    let mut advanced = state.clone();
                    let accepted = advanced.commit_bytes(bytes).is_ok();
                    assert_eq!(
                        accepted, expected,
                        "static mask/commit mismatch during dynamic sweep: {context} depth={depth} path={path:?} token={token_id}"
                    );
                    if accepted {
                        let mut next_path = path.clone();
                        next_path.push(token_id);
                        next.push((advanced, next_path));
                    }
                }
            }
            frontier = next;
        }
    }

    #[test]
    fn root_admission_memo_is_exact_and_owner_frontier_scoped() {
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);
        let constraint = Constraint::compile(Grammar::glrm(
            r#"glrm 1; start s; t A = "a"; t B = "b"; nt s = A B;"#,
        ), &vocab).unwrap();
        let other = Constraint::compile(Grammar::glrm(
            r#"glrm 1; start s; t A = "a"; t B = "b"; nt s = B A;"#,
        ), &vocab).unwrap();
        let stacks = ParserStacks::from_single_stack(vec![0], ());
        let shared = Some(Arc::new(RootAdmissionMemo {
            owner: &constraint as *const Constraint as usize,
            gss: stacks.clone(), admitted: OnceLock::new(),
        }));
        assert!(shared.as_ref().unwrap().admitted.get().is_none());
        let expected = exact_parser_admission_for_stacks(&constraint, &stacks);
        assert_eq!(parser_admission_with_root_memo(&constraint, &stacks, &shared), expected);
        assert_eq!(parser_admission_with_root_memo(&constraint, &stacks.clone(), &shared), expected);
        assert_eq!(shared.as_ref().unwrap().admitted.get(), Some(&expected));

        let terminal = constraint.terminal_display_names.iter()
            .position(|name| name == "A").unwrap() as TerminalID;
        let child = parser_child(&constraint, &stacks, terminal).unwrap();
        assert!(!child.ptr_eq(&stacks));
        assert_eq!(parser_admission_with_root_memo(&constraint, &child, &shared),
                   exact_parser_admission_for_stacks(&constraint, &child));
        assert_eq!(parser_admission_with_root_memo(&other, &stacks, &shared),
                   exact_parser_admission_for_stacks(&other, &stacks));
        assert_eq!(shared.as_ref().unwrap().admitted.get(), Some(&expected),
                   "foreign/frontier fallback must not overwrite retained root proof");
    }

    #[test]
    fn full_walk_future_cache_normalizes_hot_and_generic_coordinates() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()), (1, b"bc".to_vec()), (2, b"xyz".to_vec()),
        ]);
        let constraint = Constraint::compile(Grammar::glrm(
            r#"glrm 1; start s; t WORD = "abc"; t OTHER = "xyz"; nt s = WORD | OTHER;"#,
        ), &vocab).unwrap();
        let word = constraint.terminal_display_names.iter().position(|name| name == "WORD").unwrap() as u32;
        let other = constraint.terminal_display_names.iter().position(|name| name == "OTHER").unwrap() as u32;
        let mut scan = DynamicNfaScanCache::new(&constraint, None);
        let raw = scan.transition(constraint.tokenizer.initial_state(), b'a');
        assert_ne!(raw, u32::MAX);
        // Hot scalar IDs are used by the ordinary lazy-config executor, not
        // deterministic identity-config execution. Match that representation
        // explicitly, as the missing-subset ordinary runtime path does.
        scan.deterministic = false;
        let mut table = FullWalkConfigTransitions::new(&mut scan, 3, 0);
        table.hot_enabled = true;
        table.profile = true;
        let id = table.hot_scalar.intern(raw).unwrap();
        let hot = FullWalkHotScalarCache::tagged(id);
        let generic = table.generic_config_for_state(hot);
        assert_ne!(hot, generic, "fixture must exercise two coordinate representations");
        assert!(table.future_contains(hot, word));
        assert_eq!(table.future_contains_calls, 1);
        assert!(table.future_contains(hot, word));
        assert!(table.future_contains(generic, word));
        assert_eq!(table.future_contains_calls, 1, "both aliases must reuse the same positive result");
        assert!(!table.future_contains(hot, other));
        assert!(!table.future_contains(hot, other));
        assert!(!table.future_contains(generic, other));
        assert_eq!(table.future_contains_calls, 2, "negative results must be memoized too");
        assert_eq!(table.future_contains_cache.len(), 2);
        // Guard transitions use the same hot/config aliases. A physical state
        // with a live WORD continuation cannot be summarized as only virtual
        // residuals; an absent OTHER continuation is an empty exact set.
        assert_eq!(table.direct_prune_coordinates(hot, word), None);
        assert_eq!(table.direct_prune_coordinates(generic, word), None);
        assert_eq!(table.direct_prune_coordinates(hot, other), Some(SmallVec::new()));
        assert_eq!(table.direct_prune_coordinates(generic, other), Some(SmallVec::new()));
    }

    #[test]
    fn deterministic_dynamic_scan_does_not_index_identity_configs() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"ab".to_vec()),
        ]);
        let grammar = r#"
start start;
t A ::= 'a'+;
t B ::= 'b';
nt start ::= A B | A;
"#;
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();
        let mut cache = DynamicNfaScanCache::new(&constraint, None);
        assert!(cache.raw_start_config.is_empty());

        let initial = constraint.tokenizer.initial_state();
        let config = cache.config_for_raw_start(initial).unwrap();
        assert_eq!(config, initial);
        assert!(cache.raw_start_config.is_empty());
        let _ = cache.step_config(config, b'a').unwrap();
        assert!(cache.raw_start_config.is_empty());
    }

    #[test]
    fn strict_full_walk_executes_native_epsilon_nfa_configs() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"aa".to_vec()),
            (3, b"ab".to_vec()),
            (4, b"ba".to_vec()),
            (5, b"bb".to_vec()),
        ]);
        // The dynamic constraint below substitutes a deliberately retained
        // epsilon-NFA tokenizer with the same two terminal IDs.
        let parser_source = Constraint::from_glrm_grammar(
            r#"
start start;
t A ::= 'a';
t B ::= 'b';
nt start ::= A | B | A A | A B | B A | B B;
"#,
            &vocab,
        )
        .unwrap();
        let tokenizer =
            crate::automata::lexer::tokenizer::arbitrary_epsilon_l1_test_tokenizer();
        assert!(tokenizer.has_epsilon_transitions());
        // Temporary compiler analysis is allowed; from_parts derives native
        // relations and drops this table before constructing the Constraint.
        let grammar = parser_source.template_parser.as_ref().unwrap().link_grammar.as_ref().unwrap();
        let mut table = crate::compiler::glr::table::GLRTable::build_with_default_construction(
            &grammar.analyze(parser_source.terminal_display_names.clone()),
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );
        table.set_embedded_start_nullable(grammar.root_nullable);
        table.set_embedded_end_token_ids(&grammar.embedded_end_token_ids);
        let mut dynamic = DynamicConstraint::from_parts(
            table,
            parser_source.terminal_display_names.clone(),
            tokenizer,
            None,
            None,
            Vec::new(),
            &vocab,
        );
        assert!(dynamic.inner.tokenizer.has_epsilon_transitions());
        // Production may prepare a dense deterministic execution coordinate
        // when it fits. Disable that representation here so this regression
        // exercises the native epsilon-NFA backend itself.
        dynamic
            .inner
            .dynamic_mask_vocab
            .disable_prepared_mask_execution_for_test();
        assert!(
            dynamic.inner.dynamic_mask_vocab.mask_projection_tokenizer().is_none(),
            "test setup should retain the source epsilon-NFA coordinate",
        );
        assert!(
            dynamic.inner.dynamic_mask_vocab.mask_projection_fast_transitions().is_none(),
            "epsilon-NFA strict walk must not depend on a dense DFA table",
        );

        TEST_FULL_WALK_USES.with(|count| count.set(0));
        TEST_CONFIG_FULL_WALK_USES.with(|count| count.set(0));
        let state = dynamic.inner.start();
        let mask = state.mask();
        TEST_FULL_WALK_USES.with(|count| assert!(count.get() > 0));
        TEST_CONFIG_FULL_WALK_USES.with(|count| assert!(count.get() > 0));

        for (token_id, _) in dynamic.inner.token_bytes_iter() {
            let expected = token_allowed(&mask, token_id);
            let mut advanced = state.clone();
            let accepted = advanced.commit_token(token_id).is_ok();
            assert_eq!(accepted, expected, "token={token_id}");
        }
    }

    #[test]
    fn additive_and_candidate_modes_use_complete_strict_full_walk() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"ab".to_vec()),
            (3, b"ba".to_vec()),
            (4, b"aa".to_vec()),
        ]);
        let dynamic = DynamicConstraint::from_glrm_grammar(
            r#"
start start;
t A ::= 'a'+;
t B ::= 'b';
nt start ::= A B | B A;
"#,
            &vocab,
        )
        .unwrap();
        let state = dynamic.inner.start();
        let mut exact = vec![0u32; dynamic.inner.body_mask_len()];
        fill_mask_dynamic(&state, &mut exact);

        let mut additive = vec![0u32; exact.len()];
        set_mask_bit(&mut additive, 4);
        let mut expected_additive = additive.clone();
        for (dst, &word) in expected_additive.iter_mut().zip(&exact) {
            *dst |= word;
        }
        TEST_FULL_WALK_USES.with(|count| count.set(0));
        or_mask_dynamic_additions(&state, &mut additive);
        TEST_FULL_WALK_USES.with(|count| assert!(count.get() > 0));
        assert_eq!(additive, expected_additive);

        let mut candidates = vec![0u32; exact.len()];
        set_mask_bit(&mut candidates, 0);
        set_mask_bit(&mut candidates, 3);
        let mut filtered = vec![0u32; exact.len()];
        set_mask_bit(&mut filtered, 4);
        let mut expected_filtered = filtered.clone();
        for ((dst, &word), &candidate) in expected_filtered
            .iter_mut()
            .zip(&exact)
            .zip(&candidates)
        {
            *dst |= word & candidate;
        }
        TEST_FULL_WALK_USES.with(|count| count.set(0));
        or_mask_dynamic_candidate_additions(&state, &mut filtered, &candidates);
        TEST_FULL_WALK_USES.with(|count| assert!(count.get() > 0));
        assert_eq!(filtered, expected_filtered);
    }

    #[test]
    fn strict_full_walk_spills_for_vocab_depth_beyond_255_edges() {
        let mut entries = Vec::new();
        for depth in 1..=300u32 {
            let mut token = vec![b'a'; depth as usize];
            token.push(b'b');
            entries.push((depth - 1, token));
        }
        entries.push((300, vec![b'a'; 301]));
        let vocab = Vocab::new(entries);
        let dynamic = DynamicConstraint::from_glrm_grammar(
            r#"
start start;
t A ::= /a+b?/;
nt start ::= A;
"#,
            &vocab,
        )
        .unwrap();
        assert!(
            dynamic
                .inner
                .dynamic_mask_vocab_for_runtime()
                .trie
                .full_walk_max_parent_depth()
                >= 255,
            "test vocabulary did not create the intended deep radix tree",
        );

        TEST_FULL_WALK_USES.with(|count| count.set(0));
        let state = dynamic.start();
        let mask = state.mask();
        TEST_FULL_WALK_USES.with(|count| assert!(count.get() > 0));
        for (token_id, _) in dynamic.inner.token_bytes_iter() {
            let expected = token_allowed(&mask, token_id);
            let mut advanced = state.clone();
            assert_eq!(advanced.commit_token(token_id).is_ok(), expected, "token={token_id}");
        }
    }

    #[test]
    fn dynamic_mask_matches_normal_for_repeat_and_cross_terminal_tokens() {
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"aa".to_vec()),
                (2, b"b".to_vec()),
                (3, b"ab".to_vec()),
                (4, b"aab".to_vec()),
                (5, b"aaa".to_vec()),
            ]);
        let grammar = r#"
start start;
t A ::= 'a'+;
t B ::= 'b';
nt start ::= A B | A;
"#;
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();

        let mut state = constraint.start();
        assert_dynamic_parity(&state);
        assert!(token_allowed(&state.mask(), 3));

        state.commit_token(1).unwrap();
        assert_dynamic_parity(&state);
        assert!(token_allowed(&state.mask(), 2));

        state.commit_token(2).unwrap();
        assert!(state.is_accepting());
        assert_dynamic_parity(&state);
    }

    #[test]
    fn dynamic_mask_trie_is_rebuilt_after_load() {
        let vocab = Vocab::new(
            vec![(0, b"a".to_vec()), (1, b"b".to_vec()), (2, b"ab".to_vec())]);
        let grammar = r#"
start start;
t A ::= 'a';
t B ::= 'b';
nt start ::= A B;
"#;
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();
        let loaded = Constraint::load(&constraint.save()).unwrap();
        assert_dynamic_parity(&loaded.start());
    }

    #[test]
    fn dynamic_mask_keeps_duplicate_byte_token_aliases() {
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"b".to_vec()),
                (7, b"a".to_vec()),
                (12, b"ab".to_vec()),
            ]);
        let grammar = r#"
start start;
t A ::= 'a';
t B ::= 'b';
nt start ::= A B;
"#;
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();

        let mut state = constraint.start();
        assert_dynamic_parity(&state);
        let mask = direct_mask(&state);
        assert!(token_allowed(&mask, 0));
        assert!(token_allowed(&mask, 7));
        assert!(token_allowed(&mask, 12));

        state.commit_token(7).unwrap();
        assert_dynamic_parity(&state);
        assert!(token_allowed(&direct_mask(&state), 1));
    }

    #[test]
    fn dynamic_mask_matches_normal_across_an_ignore_terminal() {
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"aa".to_vec()),
                (2, b"b".to_vec()),
                (3, b" b".to_vec()),
                (4, b"  b".to_vec()),
            ]);
        let grammar = r#"
start start;
ignore WS;
t WS ::= ' '+;
t A ::= 'a'+;
t B ::= 'b';
nt start ::= A B;
"#;
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();

        let mut state = constraint.start();
        assert_dynamic_parity(&state);
        state.commit_token(1).unwrap();
        assert_dynamic_parity(&state);
        assert!(token_allowed(&state.mask(), 3));

        state.commit_token(3).unwrap();
        assert!(state.is_accepting());
        assert_dynamic_parity(&state);
    }

    #[test]
    fn dynamic_mask_preserves_repeated_terminal_after_ignore_reset_inside_token() {
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"b".to_vec()),
                (2, b"c".to_vec()),
                (3, b"aa".to_vec()),
                (4, b"bb".to_vec()),
                (5, b"cc".to_vec()),
                (6, b"ab".to_vec()),
                (7, b"ac".to_vec()),
                (8, b"ba".to_vec()),
                (9, b"bc".to_vec()),
                (10, b"abc".to_vec()),
                (11, b"aab".to_vec()),
                (12, b"abb".to_vec()),
                (13, b"acc".to_vec()),
                (14, b" ".to_vec()),
                (15, b"  ".to_vec()),
                (16, b" a".to_vec()),
                (17, b"a ".to_vec()),
                (18, b" a ".to_vec()),
                (19, b"ab c".to_vec()),
            ]);
        let grammar = r#"
start start;
ignore WS;
lexer group ws ::= WS;
lexer group a ::= A;
lexer group b ::= B;
lexer group c ::= C;
t WS ::= " "+;
t A ::= "a"+;
t B ::= "b";
t C ::= "c";
nt item ::= A | B | C;
nt start ::= item item? item?;
"#;
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();
        let mut state = constraint.start();
        state.commit_token(0).unwrap();
        state.commit_token(16).unwrap();

        assert_dynamic_parity(&state);
        assert!(token_allowed(&direct_mask(&state), 0));
        assert!(token_allowed(&direct_mask(&state), 3));
        assert!(token_allowed(&direct_mask(&state), 17));
    }

    #[test]
    fn masks_preserve_overlap_continuation_after_ignore_reset() {
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"b".to_vec()),
                (2, b"ab".to_vec()),
                (3, b" ".to_vec()),
                (4, b" a".to_vec()),
            ]);
        let grammar = r#"
start start;
ignore WS;
t WS ::= " "+;
t A ::= "ab";
t B ::= "a" | "ab";
nt item ::= A | B;
nt start ::= item item? item?;
"#;
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();
        let mut state = constraint.start();
        state.commit_token(0).unwrap();
        state.commit_token(4).unwrap();

        let static_mask = state.mask();
        let dynamic_mask = direct_mask(&state);

        let mut probe = state.clone();
        assert!(probe.commit_bytes(b"b").is_ok());
        assert!(
            token_allowed(&static_mask, 1),
            "static mask must admit b because a-ab = B WS A"
        );
        assert!(
            token_allowed(&dynamic_mask, 1),
            "dynamic mask must admit b because a-ab = B WS A"
        );
    }

    #[test]
    fn dynamic_mask_generated_small_language_sweep() {
        const WORDS: [&str; 4] = ["a", "b", "ab", "ba"];
        let vocab = Vocab::new(
            [
                "a", "b", "ab", "ba", " ", " a", "a ", " b", "b ", " a ", " b ",
            ]
            .into_iter()
            .enumerate()
            .map(|(id, word)| (id as u32, word.as_bytes().to_vec()))
            .collect());
        let languages = (1u32..1u32 << WORDS.len())
            .filter(|mask| mask.count_ones() <= 2)
            .collect::<Vec<_>>();
        let rule = |name: &str, mask: u32| {
            let rhs = WORDS
                .iter()
                .enumerate()
                .filter_map(|(index, word)| {
                    (mask & (1 << index) != 0).then(|| format!("\"{word}\""))
                })
                .collect::<Vec<_>>()
                .join(" | ");
            format!("t {name} ::= {rhs};\n")
        };

        for grouped in [false, true] {
            for ignored in [false, true] {
                let grouping = if grouped {
                    if ignored {
                        "lexer group ws ::= WS;\nlexer group a ::= A;\nlexer group b ::= B;\n"
                    } else {
                        "lexer group a ::= A;\nlexer group b ::= B;\n"
                    }
                } else {
                    ""
                };
                let ignore = if ignored {
                    "ignore WS;\nt WS ::= \" \"+;\n"
                } else {
                    ""
                };

                for &a in &languages {
                    for &b in &languages {
                        if grouped && a == b {
                            continue;
                        }
                        for start_rule in [
                            "nt item ::= A | B;\nnt start ::= item item? item?;",
                            "nt start ::= A A | B B;",
                            "nt start ::= A B | B A;",
                        ] {
                            let grammar = format!(
                                "start start;\n{ignore}{grouping}{}{}{start_rule}\n",
                                rule("A", a),
                                rule("B", b),
                            );
                            let constraint =
                                Constraint::from_glrm_grammar(&grammar, &vocab).unwrap();
                            let context = format!(
                                "finite grouped={grouped} ignored={ignored} A={a:#06b} B={b:#06b}\ngrammar:\n{grammar}"
                            );
                            assert_dynamic_parity_on_reachable_states(&constraint, 3, &context);
                        }
                    }
                }

                let grammar = format!(
                    "start start;\n{ignore}{grouping}t A ::= \"a\"+;\nt B ::= \"b\"+;\nnt item ::= A | B;\nnt start ::= item item? item?;\n"
                );
                let constraint = Constraint::from_glrm_grammar(&grammar, &vocab).unwrap();
                let context = format!(
                    "repeat grouped={grouped} ignored={ignored}\ngrammar:\n{grammar}"
                );
                assert_dynamic_parity_on_reachable_states(&constraint, 4, &context);

                let grammar = format!(
                    "start start;\n{ignore}{grouping}t A ::= \"a\"+ \"b\";\nt B ::= \"a\"+;\nnt item ::= A | B;\nnt start ::= item item? item?;\n"
                );
                let constraint = Constraint::from_glrm_grammar(&grammar, &vocab).unwrap();
                let context = format!(
                    "delayed-overlap grouped={grouped} ignored={ignored}\ngrammar:\n{grammar}"
                );
                assert_dynamic_parity_on_reachable_states(&constraint, 4, &context);
            }
        }
    }

    #[test]
    fn dynamic_mask_matches_normal_at_every_reachable_small_state() {
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"aa".to_vec()),
                (2, b"b".to_vec()),
                (3, b"bb".to_vec()),
                (4, b"c".to_vec()),
                (5, b"ab".to_vec()),
                (6, b"ba".to_vec()),
                (7, b"a c".to_vec()),
                (8, b"b c".to_vec()),
                (9, b" aa".to_vec()),
                (10, b" bb".to_vec()),
            ]);
        let grammar = r#"
start start;
ignore WS;
t WS ::= ' '+;
t A ::= 'a'+;
t B ::= 'b'+;
t C ::= 'c';
nt start ::= A B C | B A C | A C | B C;
"#;
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();

        fn visit(state: ConstraintState<'_>, depth: usize) {
            assert_dynamic_parity(&state);
            if depth == 3 {
                return;
            }
            let mask = state.mask();
            for token_id in 0..11u32 {
                if !token_allowed(&mask, token_id) {
                    continue;
                }
                let mut next = state.clone();
                next.commit_token(token_id).unwrap();
                visit(next, depth + 1);
            }
        }

        visit(constraint.start(), 0);
    }

    #[test]
    fn dynamic_mask_matches_normal_when_one_repeated_terminal_crosses_tokens() {
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"aa".to_vec()),
                (2, b"aaa".to_vec()),
                (3, b"aaaa".to_vec()),
            ]);
        let grammar = r#"
start start;
t A ::= 'a'+;
nt start ::= A A;
"#;
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();

        fn visit(state: ConstraintState<'_>, depth: usize) {
            assert_dynamic_parity(&state);
            if depth == 3 {
                return;
            }
            let mask = state.mask();
            for token_id in 0..4u32 {
                if !token_allowed(&mask, token_id) {
                    continue;
                }
                let mut next = state.clone();
                next.commit_token(token_id).unwrap();
                visit(next, depth + 1);
            }
        }

        visit(constraint.start(), 0);
    }

    #[test]
    fn dynamic_mask_matches_normal_for_a_partial_json_string() {
        let vocab = Vocab::new(
            vec![
                (0, b"\"".to_vec()),
                (1, b"a".to_vec()),
                (2, b"b".to_vec()),
                (3, b"\\\"".to_vec()),
                (4, b"\"a".to_vec()),
                (5, b"a\"".to_vec()),
            ]);
        let constraint =
            Constraint::from_json_schema(r#"{"type":"string"}"#, &vocab).unwrap();

        let mut state = constraint.start();
        assert_dynamic_parity(&state);
        state.commit_token(0).unwrap();
        assert_dynamic_parity(&state);
        state.commit_token(1).unwrap();
        assert_dynamic_parity(&state);
        state.commit_token(0).unwrap();
        assert!(state.is_accepting());
        assert_dynamic_parity(&state);
    }

    #[test]
    fn dynamic_mask_matches_certified_long_terminal_run() {
        let vocab = Vocab::new(
            vec![
                (0, b"++++++++a".to_vec()),
                (1, b"++++".to_vec()),
                (2, b"a".to_vec()),
            ]);
        let grammar = r#"
start start;
t U ::= '+';
nt start ::= U* 'a';
"#;
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();
        let mut state = constraint.start();

        assert_dynamic_parity(&state);
        assert!(token_allowed(&state.mask(), 0));
        assert!(token_allowed(&state.mask(), 1));

        state.commit_token(1).unwrap();
        assert_dynamic_parity(&state);
        assert!(token_allowed(&state.mask(), 2));
        state.commit_token(2).unwrap();
    }

    #[test]
    fn dynamic_mask_handles_monolithic_json_number() {
        let vocab = Vocab::new(
            vec![
                (0, b"-".to_vec()),
                (1, b"0".to_vec()),
                (2, b"1".to_vec()),
                (3, b"2".to_vec()),
                (4, b"3".to_vec()),
                (5, b".".to_vec()),
                (6, b"e".to_vec()),
                (7, b"+".to_vec()),
            ]);
        let constraint =
            Constraint::from_json_schema(r#"{"type":"number"}"#, &vocab).unwrap();

        let mut state = constraint.start();
        assert_dynamic_parity(&state);
        for bytes in [b"1".as_slice(), b".".as_slice(), b"2".as_slice(), b"e".as_slice(), b"-".as_slice(), b"3".as_slice()] {
            state.commit_bytes(bytes).unwrap();
            assert_dynamic_parity(&state);
        }
        assert!(state.is_accepting());
    }

    #[test]
    fn dynamic_mask_keeps_other_gss_paths_when_one_path_excludes_a_terminal() {
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"b".to_vec()),
                (2, b"c".to_vec()),
                (3, b"ab".to_vec()),
            ]);
        let grammar = r#"
start start;
t A ::= 'a' | 'ab';
t B ::= 'a';
t C ::= 'c';
t D ::= 'b';
nt start ::= A C | B D;
"#;
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();
        let mut state = constraint.start();
        state.commit_token(0).unwrap();

        let paths = state
            .state
            .values()
            .flat_map(|gss| gss.to_stacks(4_096).expect("stack enumeration exceeded explicit limit"))
            .collect::<Vec<_>>();
        assert!(paths.iter().any(|(_, exclusions)| exclusions.is_empty()));
        assert!(paths.iter().any(|(_, exclusions)| !exclusions.is_empty()));

        assert_dynamic_parity(&state);
        assert!(token_allowed(&direct_mask(&state), 1));
        state.commit_token(1).unwrap();
        assert!(state.is_accepting());
        assert_dynamic_parity(&state);
    }


    #[test]
    fn dynamic_full_walk_accepts_long_compressed_vocab_edge() {
        let vocab = Vocab::new(vec![
            (0, vec![b'a'; 300]),
            (1, b"a".to_vec()),
            (2, b"b".to_vec()),
        ]);
        let grammar = r#"
start start;
t A ::= 'a'+;
nt start ::= A;
"#;
        let dynamic = DynamicConstraint::from_glrm_grammar(grammar, &vocab).unwrap();
        let mask_vocab = dynamic.inner.dynamic_mask_vocab_for_runtime();
        assert!(mask_vocab.trie.subtree_max_total_byte_len(0) >= 300);
        assert!(
            mask_vocab.trie.full_walk_max_parent_depth() < 255,
            "a long compressed edge must not consume one DFS stack slot per byte",
        );
        assert!(mask_vocab.mask_projection_fast_transitions().is_some());

        let mask = dynamic.start().mask();
        assert!(token_allowed(&mask, 0));
        assert!(token_allowed(&mask, 1));
        assert!(!token_allowed(&mask, 2));
    }

    #[test]
    fn dynamic_virtual_unit_repeat_uses_static_mask_projection_end_to_end() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"aa".to_vec()),
            (2, b"aaa".to_vec()),
            (3, b"aaaa".to_vec()),
            (4, b"aaaaa".to_vec()),
            (5, b"aaaaaa".to_vec()),
            (6, b"b".to_vec()),
        ]);

        let billion_grammar = r#"
start start;
t A ::= /a{0,1000000000}/;
nt start ::= A;
"#;
        let billion = DynamicConstraint::from_glrm_grammar(billion_grammar, &vocab).unwrap();
        assert_eq!(
            billion.inner.tokenizer.num_states(),
            1,
            "the exact billion-bound lexer must remain an arithmetic runtime state",
        );
        assert!(
            billion.inner.dynamic_mask_vocab.mask_projection_tokenizer().is_some(),
            "virtual mask projection is immutable build-derived runtime data and must be prepared before the first mask",
        );
        let mask_tokenizer = billion
            .inner
            .dynamic_mask_vocab
            .mask_projection_tokenizer()
            .expect("build finalization must prepare the finite virtual mask projection");
        let billion_mask = billion.inner.start().mask();
        assert!(
            billion.inner.lazy_dynamic_mask_vocab.get().is_none(),
            "first mask must not materialize hidden projection state",
        );
        assert_eq!(
            mask_tokenizer.num_states(),
            vocab.max_token_byte_len() as u32 + 3,
            "mask lexer size must depend on vocabulary horizon, not the repeat bound",
        );
        for token in 0..=5 {
            assert!(token_allowed(&billion_mask, token), "a-only token {token} was rejected");
        }
        assert!(!token_allowed(&billion_mask, 6));

        // For a small bound, compare the virtual dynamic implementation with
        // the ordinary materialized static implementation through the exact
        // upper-bound transition. This checks both re-projection after commit
        // and rejection of a vocabulary token which crosses the bound.
        let boundary_grammar = r#"
start start;
t A ::= /a{0,5}/;
nt start ::= A;
"#;
        let dynamic = DynamicConstraint::from_glrm_grammar(boundary_grammar, &vocab).unwrap();
        let ordinary = Constraint::from_glrm_grammar(boundary_grammar, &vocab).unwrap();
        let mut dynamic_state = dynamic.inner.start();
        let mut ordinary_state = ordinary.start();

        assert_eq!(dynamic_state.mask(), ordinary_state.mask());
        assert!(!token_allowed(&dynamic_state.mask(), 5));
        dynamic_state.commit_token(2).unwrap(); // consume three a's
        ordinary_state.commit_token(2).unwrap();
        assert_eq!(dynamic_state.mask(), ordinary_state.mask());
        assert!(token_allowed(&dynamic_state.mask(), 0));
        assert!(token_allowed(&dynamic_state.mask(), 1));
        assert!(!token_allowed(&dynamic_state.mask(), 2));

        dynamic_state.commit_token(1).unwrap(); // reach the exact upper bound
        ordinary_state.commit_token(1).unwrap();
        assert_eq!(dynamic_state.mask(), ordinary_state.mask());
        assert!(dynamic_state.is_accepting());
        assert!(!token_allowed(&dynamic_state.mask(), 0));
        assert!(dynamic_state.clone().commit_token(0).is_err());
    }

    #[test]
    fn dynamic_hybrid_virtual_repeat_coexists_with_static_terminals() {
        let vocab = Vocab::new(vec![
            (0, b"b".to_vec()),
            (1, b"a".to_vec()),
            (2, b"aa".to_vec()),
            (3, b"aaa".to_vec()),
            (4, b"baaa".to_vec()),
            (5, b"baaaaa".to_vec()),
            (6, b"baaaaaa".to_vec()),
            (7, b"x".to_vec()),
        ]);
        let billion_grammar = r#"
start start;
t A ::= /a{0,1000000000}/;
t B ::= 'b';
nt start ::= B A;
"#;
        let hybrid = DynamicConstraint::from_glrm_grammar(billion_grammar, &vocab).unwrap();
        assert!(
            hybrid
                .inner
                .tokenizer
                .virtual_zero_min_unit_repeat_mask_tokenizer(vocab.max_token_byte_len())
                .is_some(),
            "the pathological terminal should be the arithmetic component",
        );
        assert!(
            hybrid.inner.tokenizer.num_states() < 64,
            "physical hybrid lexer unexpectedly scales with the billion bound",
        );
        let start_mask = hybrid.start().mask();
        assert!(token_allowed(&start_mask, 4));
        assert!(token_allowed(&start_mask, 5));
        assert!(token_allowed(&start_mask, 6));
        assert!(!token_allowed(&start_mask, 7));

        let small_grammar = r#"
start start;
t A ::= /a{0,5}/;
t B ::= 'b';
nt start ::= B A;
"#;
        // The hybrid threshold intentionally leaves this small grammar on the
        // ordinary path; it is therefore an independent materialized oracle.
        let dynamic_small = DynamicConstraint::from_glrm_grammar(small_grammar, &vocab).unwrap();
        let ordinary_small = Constraint::from_glrm_grammar(small_grammar, &vocab).unwrap();
        let mut dynamic_state = dynamic_small.start();
        let mut ordinary_state = ordinary_small.start();
        assert_eq!(dynamic_state.mask(), ordinary_state.mask());
        assert!(token_allowed(&dynamic_state.mask(), 5));
        assert!(!token_allowed(&dynamic_state.mask(), 6));

        dynamic_state.commit_token(0).unwrap();
        ordinary_state.commit_token(0).unwrap();
        assert_eq!(dynamic_state.mask(), ordinary_state.mask());
        dynamic_state.commit_token(3).unwrap();
        ordinary_state.commit_token(3).unwrap();
        assert_eq!(dynamic_state.mask(), ordinary_state.mask());
    }

    #[test]
    fn static_and_dynamic_direct_regular_compilation_preserve_backend_contract() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"ab".to_vec()),
            (2, b"b".to_vec()),
            (3, b"x".to_vec()),
            (4, b"bx".to_vec()),
        ]);
        let mut grammar = String::from(
            "start start;
t A ::= 'a' | 'ab';
t X ::= 'x';
nt start ::= A r0;
",
        );
        for index in 0..39 {
            grammar.push_str(&format!("nt r{index} ::= X r{};
", index + 1));
        }
        grammar.push_str("nt r39 ::= X;
");

        let constraint = Constraint::from_glrm_grammar(&grammar, &vocab).unwrap();
        assert!(!constraint.uses_dynamic_runtime());
        assert!(constraint.possible_matches_complete);
        assert!(!constraint.possible_matches.is_empty());

        let dynamic = DynamicConstraint::from_glrm_grammar(&grammar, &vocab).unwrap();
        assert!(dynamic.inner.uses_dynamic_runtime());
        assert!(!dynamic.inner.possible_matches_complete);

        fn delayed_query(constraint: &Constraint) -> (u32, TerminalID) {
            let execution = constraint.tokenizer.execute_from_state_all_widths(
                b"a",
                constraint.tokenizer.initial_state(),
            );
            execution
                .matches
                .iter()
                .find_map(|matched| {
                    constraint
                        .tokenizer
                        .possible_future_terminals(matched.end_state)
                        .contains(matched.id as usize)
                        .then_some((matched.end_state, matched.id))
                })
                .expect("A=a|ab must create one delayed-terminal query state")
        }

        let (compiled_state, compiled_terminal) = delayed_query(&constraint);
        let (fallback_state, fallback_terminal) = delayed_query(&dynamic.inner);
        assert_eq!(compiled_terminal, fallback_terminal);

        let mut direct_table_tokens = Vec::new();
        assert!(constraint.visit_possible_match_original_tokens(
            compiled_state,
            compiled_terminal,
            |token| direct_table_tokens.push(token),
        ));
        direct_table_tokens.sort_unstable();
        direct_table_tokens.dedup();

        let mut helper_table_tokens = Vec::new();
        for_each_token_matching_terminal_from_state(
            &constraint,
            compiled_state,
            compiled_terminal,
            |token| helper_table_tokens.push(token),
        )
        .unwrap();
        helper_table_tokens.sort_unstable();
        helper_table_tokens.dedup();

        let mut fallback_tokens = Vec::new();
        for_each_token_matching_terminal_from_state(
            &dynamic.inner,
            fallback_state,
            fallback_terminal,
            |token| fallback_tokens.push(token),
        )
        .unwrap();
        fallback_tokens.sort_unstable();
        fallback_tokens.dedup();

        assert_eq!(helper_table_tokens, direct_table_tokens);
        assert_eq!(direct_table_tokens, fallback_tokens);
        assert_eq!(direct_table_tokens, vec![2, 4]);
    }

    #[test]
    fn derived_single_use_terminal_possible_matches_match_dynamic_fallback() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"aa".to_vec()),
            (2, b"aaa".to_vec()),
            (3, b"ab".to_vec()),
            (4, b"b".to_vec()),
            (5, b"ba".to_vec()),
        ]);
        let grammar = r#"
start start;
t A ::= 'a'+;
nt start ::= A;
"#;
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();
        assert!(!constraint.uses_dynamic_runtime());
        assert!(constraint.possible_matches_complete);
        assert_eq!(constraint.possible_matches.len(), 1);

        // Committing one accepting prefix leaves the same terminal live, so
        // the next mask exercises the delayed-terminal exclusion table that is
        // derived from the sole global-L1 transition.
        let mut after_a = constraint.start();
        after_a.commit_token(0).unwrap();
        assert!(after_a.is_accepting());
        assert!(after_a.state.values().any(|gss| {
            !gss.all_accs_satisfy(|excluded: &TerminalsDisallowed| excluded.is_empty())
        }));
        assert_dynamic_parity(&after_a);

        assert_dynamic_parity_on_reachable_states(
            &constraint,
            3,
            "derived single-use terminal possible matches",
        );
        assert!(
            constraint
                .seed_terminal_dense_fallback
                .lock()
                .expect("fallback cache poisoned")
                .is_empty(),
            "complete derived possible matches must not invoke runtime fallback",
        );
    }

    #[test]
    fn dynamic_mask_handles_overlapping_live_terminal_paths() {
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"ab".to_vec()),
                (2, b"b".to_vec()),
                (3, b"bc".to_vec()),
                (4, b"c".to_vec()),
            ]);
        let grammar = r#"
start start;
t A ::= 'a' | 'ab';
t B ::= 'b' | 'bc';
nt start ::= A B;
"#;
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();

        let mut state = constraint.start();
        assert_dynamic_parity(&state);
        state.commit_token(1).unwrap();
        assert_dynamic_parity(&state);
        assert!(token_allowed(&state.mask(), 3));
        state.commit_token(3).unwrap();
        assert!(state.is_accepting());
        assert_dynamic_parity(&state);
    }

    #[test]
    fn dynamic_mask_handles_a_live_cross_terminal_prefix() {
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"ab".to_vec()),
                (2, b"abc".to_vec()),
                (3, b"bc".to_vec()),
                (4, b"c".to_vec()),
            ]);
        let grammar = r#"
start start;
t A ::= 'a' | 'abc';
nt start ::= A;
"#;
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();

        let mut state = constraint.start();
        assert_dynamic_parity(&state);
        state.commit_token(0).unwrap();
        assert_dynamic_parity(&state);
        assert!(token_allowed(&state.mask(), 3));
        assert!(!token_allowed(&state.mask(), 4));
        state.commit_token(3).unwrap();
        assert!(state.is_accepting());
        assert_dynamic_parity(&state);
    }


    #[test]
    fn recursive_subtree_proof_preserves_sparse_aliases_and_output_polarity() {
        struct CapacityGuard(usize);
        impl Drop for CapacityGuard {
            fn drop(&mut self) {
                TEST_RECURSIVE_PRODUCT_CACHE_CAPACITY.with(|v| v.set(self.0));
            }
        }
        let _capacity = CapacityGuard(TEST_RECURSIVE_PRODUCT_CACHE_CAPACITY.with(|v| v.get()));
        // Nested a-prefix terminals make the radix walk learn an a self-loop
        // before reaching the remaining a-only subtrees. Duplicate byte strings
        // have distinct, sparse original IDs. More rejected original tokens
        // switches the same exact result to adaptive positive-side emission.
        for distractors in [0usize, 192] {
            let mut entries = vec![
                (0, b"".to_vec()), (2, b"X".to_vec()), (4, b"!".to_vec()),
                (6, b"a!".to_vec()), (8, b"Xa!".to_vec()), (10, b"z".to_vec()),
            ];
            for len in 1..=64usize {
                let id = 16 + (len as u32 - 1) * 4;
                entries.push((id, vec![b'a'; len]));
                entries.push((id + 1, vec![b'a'; len]));
            }
            for index in 0..distractors {
                entries.push((1024 + index as u32 * 2, format!("z{index:03}").into_bytes()));
            }
            let vocab = Vocab::new(entries);
            let parent = Constraint::compile(Grammar::glrm(
                r#"glrm 1; start root; extern grammar payload; nt root = "X" payload "!";"#,
            ), &vocab).unwrap();
            let child = Constraint::compile(Grammar::glrm(
                r#"start payload; t A ::= "a"+; nt payload ::= A;"#,
            ), &vocab).unwrap();
            let composed = parent.bind_grammar_dynamic_boundary("payload", child).unwrap();
            let loaded = Constraint::load(composed.save()).unwrap();
            for constraint in [&composed, &loaded] {
                for prefix in ["X", "Xa", "Xaa"] {
                    let mut state = constraint.start();
                    state.commit_bytes(prefix.as_bytes()).unwrap();
                    let mut expected = vec![0; constraint.mask_len()];
                    state.fill_recursive_mask_by_exact_full_walk(&mut expected);
                    for capacity in [0, 1, 2, 16, 2048] {
                        TEST_RECURSIVE_PRODUCT_CACHE_CAPACITY.with(|v| v.set(capacity));
                        let mut actual = vec![u32::MAX; expected.len()];
                        assert!(recursive_provider::fill(&state, &mut actual).unwrap());
                        assert_eq!(actual, expected,
                            "sparse aliases: prefix={prefix:?}, capacity={capacity}, distractors={distractors}");
                        for len in 1..=64usize {
                            let id = 16 + (len as u32 - 1) * 4;
                            assert!(token_allowed(&actual, id));
                            assert!(token_allowed(&actual, id + 1));
                            assert!(!token_allowed(&actual, id + 2), "unused sparse ID must stay clear");
                        }
                        assert!(!token_allowed(&actual, 10));
                        for index in 0..distractors {
                            assert!(!token_allowed(&actual, 1024 + index as u32 * 2));
                        }
                    }
                }
            }
        }
    }


    #[test]
    fn recursive_zero_byte_domain_preserves_only_live_exact_special_ids() {
        let vocab = Vocab::new(vec![
            (0, Vec::new()), (1, b"X".to_vec()), (2, b"a".to_vec()),
            (3, b"!".to_vec()), (7, Vec::new()), (31, Vec::new()),
            (40, b"a!".to_vec()),
        ]);
        let child = crate::ConstraintSpec::builder(Grammar::glrm(
            r#"glrm 1; start child; extern token MARK; nt child = MARK "a" | "a";"#,
        ), &vocab).unwrap().bind_token("MARK", [7]).unwrap()
            .build().unwrap().compile().unwrap();
        let parent = Constraint::compile(Grammar::glrm(
            r#"glrm 1; start root; extern grammar child; nt root = "X" child "!";"#,
        ), &vocab).unwrap();
        let bound = parent.bind_grammar_dynamic_boundary("child", child).unwrap();
        let loaded = Constraint::load(bound.save()).unwrap();
        for constraint in [&bound, &loaded] {
            assert_eq!(constraint.empty_byte_token_ids.as_ref(), &[0, 7, 31]);
            // Domain sanitization must not build a full model-vocabulary trie
            // just to find zero-byte spellings, including after artifact load.
            let was_lazy = constraint.lazy_dynamic_mask_vocab.get().is_some();
            let blank = constraint.start();
            let mut invalid = vec![0; constraint.mask_len()];
            for id in [0u32, 7, 31] { set_mask_bit(&mut invalid, id); }
            blank.clear_late_grammar_placeholder_mask(&mut invalid);
            assert!(invalid.iter().all(|&word| word == 0));
            assert_eq!(constraint.lazy_dynamic_mask_vocab.get().is_some(), was_lazy);
            for prefix in ["", "X", "Xa", "Xa!"] {
                let mut state = constraint.start();
                state.commit_bytes(prefix.as_bytes()).unwrap();
                let mut oracle = vec![0; constraint.mask_len()];
                state.fill_recursive_mask_by_exact_full_walk(&mut oracle);
                let mut actual = vec![u32::MAX; oracle.len()];
                assert!(recursive_provider::fill(&state, &mut actual).unwrap());
                assert_eq!(actual, oracle, "config provider at {prefix:?}");
                assert!(!token_allowed(&actual, 0), "unbound empty alias");
                assert!(!token_allowed(&actual, 31), "second unbound empty alias");
                assert_eq!(token_allowed(&actual, 7), prefix == "X",
                    "empty exact ID requires a live MARK path at {prefix:?}");
                for _ in 0..2 {
                    assert!(try_fill_recursive_mask_shared(&state, &mut actual).unwrap());
                    assert_eq!(actual, oracle, "shared/cache domain at {prefix:?}");
                }
                assert_eq!(state.mask(), oracle);
            }
            let mut state = constraint.start();
            for token in [1, 7, 2, 3] {
                assert!(token_allowed(&state.mask(), token));
                state.commit_token(token).unwrap();
            }
            assert!(state.is_accepting());
        }
    }

}


#[cfg(test)]
mod epsilon_only_reset_config_tests {
    use super::*;
    use crate::{Constraint,Grammar,Vocab};

    fn fixture()->Constraint {
        let vocab=Vocab::new(vec![(0,b"a".to_vec()),(1,b"b".to_vec()),(2,b"ab".to_vec())]);
        Constraint::compile(Grammar::glrm(
            r#"glrm 1; start s; t A = "a"; t B = "b"; nt s = A B | A;"#,
        ),&vocab).unwrap()
    }

    #[test]
    fn fresh_empty_reset_preserves_only_boundary_acceptance() {
        let c=fixture();
        let mut scan=DynamicNfaScanCache::new(&c,None);
        // Exercise the config representation even if this tiny fixture's
        // physical tokenizer is deterministic.
        scan.deterministic=false;
        let ordinary_empty=scan.intern_config(Vec::new()).unwrap();
        let fresh=scan.fresh_empty_reset_config().unwrap();
        assert_ne!(fresh,ordinary_empty);
        let fi=scan.config_index(fresh).unwrap();
        let oi=scan.config_index(ordinary_empty).unwrap();
        assert!(scan.config_is_fresh_reset[fi]);
        assert!(!scan.config_is_fresh_reset[oi]);
        assert_eq!(scan.config_len(fresh),0);
        assert!(!scan.config_has_finalizer(fresh));
        assert!(scan.config_matched[fi].is_empty());
        assert!(scan.config_futures[fi].is_empty());
        assert_eq!(scan.residual_config(fresh).unwrap(),None);
        for byte in 0u8..=u8::MAX {
            assert_eq!(scan.step_config(fresh,byte).unwrap(),None,"byte={byte}");
        }
    }

    #[test]
    fn fresh_empty_reset_is_memoized_and_ordinary_empty_stays_nonfresh() {
        let c=fixture();
        let mut scan=DynamicNfaScanCache::new(&c,None);
        scan.deterministic=false;
        let a=scan.fresh_empty_reset_config().unwrap();
        let b=scan.fresh_empty_reset_config().unwrap();
        assert_eq!(a,b);
        let empty=scan.intern_config(Vec::new()).unwrap();
        assert_ne!(a,empty);
        assert!(!scan.config_is_fresh_reset[scan.config_index(empty).unwrap()]);
        assert!(scan.config_is_fresh_reset[scan.config_index(a).unwrap()]);
    }
}


#[cfg(test)]
mod saved_depth_storage_tests {
    use super::{full_walk_owned_many_slot, SmallVec};

    #[test]
    fn untouched_cached_walk_needs_no_slots_and_descending_visits_keep_high_slots() {
        let mut slots=SmallVec::<[Option<u32>;8]>::new();
        assert!(slots.is_empty());assert!(!slots.spilled());
        *full_walk_owned_many_slot(&mut slots,0,11)=Some(7);
        assert_eq!(slots.len(),1);assert!(!slots.spilled());
        *full_walk_owned_many_slot(&mut slots,9,11)=Some(31);
        assert_eq!(slots.len(),10);assert!(slots.capacity()<=11);
        *full_walk_owned_many_slot(&mut slots,2,11)=Some(9);
        assert_eq!(slots.len(),10);assert_eq!(slots[0],Some(7));assert_eq!(slots[9],Some(31));
        *full_walk_owned_many_slot(&mut slots,9,11)=Some(42);
        assert_eq!(slots[9],Some(42));assert!(slots[1].is_none());
    }

    #[test]
    fn deep_saved_path_preserves_every_slot_and_checks_certified_bound() {
        let mut slots=SmallVec::<[Option<usize>;8]>::new();
        for depth in 0..513 {
            *full_walk_owned_many_slot(&mut slots,depth,513)=Some(depth*3);
            assert_eq!(slots.len(),depth+1);
            if slots.spilled(){assert!(slots.capacity()<=513);}
        }
        for depth in (0..513).rev(){assert_eq!(slots[depth],Some(depth*3));}
        let mut invalid=SmallVec::<[Option<u32>;8]>::new();
        let result=std::panic::catch_unwind(std::panic::AssertUnwindSafe(||{
            let _=full_walk_owned_many_slot(&mut invalid,3,3);
        }));
        assert!(result.is_err());assert!(invalid.is_empty());
    }

    #[test]
    fn overwritten_and_saved_owned_values_are_dropped_exactly_once() {
        use std::sync::{Arc,atomic::{AtomicUsize,Ordering}};
        struct Tracked(Arc<AtomicUsize>);
        impl Drop for Tracked {fn drop(&mut self){self.0.fetch_add(1,Ordering::SeqCst);}}
        let drops=Arc::new(AtomicUsize::new(0));
        {
            let mut slots=SmallVec::<[Option<Tracked>;8]>::new();
            *full_walk_owned_many_slot(&mut slots,0,17)=Some(Tracked(drops.clone()));
            *full_walk_owned_many_slot(&mut slots,16,17)=Some(Tracked(drops.clone()));
            assert_eq!(drops.load(Ordering::SeqCst),0);
            *full_walk_owned_many_slot(&mut slots,16,17)=Some(Tracked(drops.clone()));
            assert_eq!(drops.load(Ordering::SeqCst),1);
        }
        assert_eq!(drops.load(Ordering::SeqCst),3);
    }
}
