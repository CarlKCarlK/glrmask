use crate::automata::lexer::Lexer;
pub(crate) mod profile;
pub(crate) mod template_advance;
pub(crate) mod push_suffixes;
mod flat_template;
pub(crate) mod push_dag;
pub(crate) mod phase_dag;
pub(crate) mod single_cursor;
pub(crate) use template_advance::advance_stacks_template_dfa;
pub(crate) mod tokenizer_scan;

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

use crate::automata::lexer::tokenizer::{Tokenizer, TokenizerExecResult, TokenizerMatch, TokenizerStateSet};
use crate::compiler::glr::accumulator::TerminalsDisallowed;
use crate::compiler::glr::parser::{
    ParserGSS,
    apply_guarded_stack_shifts_fast,
    advance_control_closed_stacks,
    advance_control_closed_stacks_owned,
    advance_stacks,
    advance_stacks_profiled,
    advance_stacks_owned,
    advance_stacks_disjoint_top_terminals_bounded,
    normalize_lookahead_invariant_reductions,
    AdvanceProfile,
    stack_may_advance_on,
    stack_may_advance_on_control_closed,
    stack_may_advance_on_any,
    stack_may_advance_on_any_control_closed,
    stack_may_advance_disjoint_top_terminals_bounded,
    stack_admissible_terminals,
};
use crate::compiler::glr::table::{Action, AdmissionPolicy, GLRTable};
use crate::compiler::glr::table::row::ActionRow;
use crate::runtime::artifact::DynamicMaskVocab;
use crate::runtime::constraint::Constraint;
use crate::runtime::state::{
    CommitBuffers, ConstraintState, INLINE_PARSER_STATE_CAPACITY, LINEAR_STACK_RESERVE,
    ParserAdmissionCacheEntry, ParserRelativeMaskEqCacheEntry, ParserStateMap,
};
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use self::profile::{
    apply_advance_profile,
    fast_action_advance_profile,
    CommitProfile,
    PerAdvanceEntry,
};
use self::template_advance::advance_stacks_template_dfa_owned;
pub(crate) use self::template_advance::TemplateAdvanceRuntime;
use self::tokenizer_scan::{
    execute_recursive_tokenizer_reusable, execute_tokenizer_from_state_small,
    execute_tokenizer_reusable, execute_tokenizer_reusable_from_states, InitialCommitScan,
};

type ParserStatesByTokenizer = FxHashMap<u32, ParserGSS>;

type SmallParserStates =
    SmallVec<[(u32, ParserGSS); INLINE_PARSER_STATE_CAPACITY]>;

const SMALL_LANGUAGE_QUEUE_CAPACITY: usize = 32;
const LANGUAGE_QUEUE_MAX_INPUT_STACK_DEPTH: u32 = 512;
const LANGUAGE_QUEUE_MIN_TOP_VALUES: usize = 3;
const LANGUAGE_QUEUE_MIN_PATHS: usize = 32;
const LANGUAGE_QUEUE_MIN_NODES: usize = 48;

#[cold]
#[inline(never)]
fn try_terminal_observation_quotient_commit_reuse(
    buffers: &mut CommitBuffers,
    generation: u64,
    vocab: &DynamicMaskVocab,
    tokenizer: &Tokenizer,
    left_source: u32,
    right_source: u32,
    admitted: &crate::ds::bitset::BitSet,
    cache_left: u32,
    cache_right: u32,
    visited_nodes: usize,
    byte_steps: usize,
    started: Option<std::time::Instant>,
) -> bool {
    let Some(quotient_relevant) = vocab.terminal_observation_equivalent_for_live_admitted(
        left_source,
        right_source,
        admitted,
        tokenizer.matched_terminal_bitset(left_source),
        tokenizer.possible_future_terminals(left_source),
        tokenizer.matched_terminal_bitset(right_source),
        tokenizer.possible_future_terminals(right_source),
    ) else {
        return false;
    };

    if let Some(started) = started {
        eprintln!(
            "[glrmask/profile][parser_relative_commit_reuse] generation={} result=terminal_observation_quotient left={} right={} admitted={} relevant={} nodes={} bytes={} elapsed_us={:.1}",
            generation,
            left_source,
            right_source,
            admitted.count_ones(),
            quotient_relevant,
            visited_nodes,
            byte_steps,
            started.elapsed().as_secs_f64() * 1e6,
        );
    }

    const CACHE_CAPACITY: usize = 8;
    if buffers.parser_relative_mask_eq_cache.len() == CACHE_CAPACITY {
        buffers.parser_relative_mask_eq_cache.remove(0);
    }
    buffers
        .parser_relative_mask_eq_cache
        .push(ParserRelativeMaskEqCacheEntry {
            left: cache_left,
            right: cache_right,
            admitted: admitted.clone(),
        });
    true
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SmallLanguageParserState {
    tokenizer_state: u32,
    language: u32,
    accumulator: TerminalsDisallowed,
}

type SmallLanguageParserStates = SmallVec<[SmallLanguageParserState; 16]>;

#[derive(Debug)]
pub(crate) struct SmallCommitQueueScratch {
    processing: [SmallParserStates; 17],
    pending: SmallParserStates,
    language_processing: [SmallLanguageParserStates; 17],
    language_pending: SmallLanguageParserStates,
    prune_union_starts: SmallVec<[u32; 8]>,
}

impl Default for SmallCommitQueueScratch {
    fn default() -> Self {
        Self {
            processing: std::array::from_fn(|_| SmallVec::new()),
            pending: SmallVec::new(),
            language_processing: std::array::from_fn(|_| SmallVec::new()),
            language_pending: SmallVec::new(),
            prune_union_starts: SmallVec::new(),
        }
    }
}

impl SmallCommitQueueScratch {
    pub(crate) fn clear(&mut self) {
        for bucket in &mut self.processing {
            bucket.clear();
        }
        self.pending.clear();
        for bucket in &mut self.language_processing {
            bucket.clear();
        }
        self.language_pending.clear();
        self.prune_union_starts.clear();
    }
}

#[cfg(test)]
#[test]
fn mask_only_flat_frontier_keeps_empty_semantics_without_branch_allocations() {
    let lean = FlatFrontierScratch::for_mask_only_shadow();
    let commit = FlatFrontierScratch::with_preallocated_gss(0);
    assert_eq!(lean.len, commit.len);
    assert_eq!(lean.continuation_cache_len, commit.continuation_cache_len);
    assert!(lean.gss_pool.is_empty() && commit.gss_pool.is_empty());
    assert!(lean.retired_gss.is_empty() && commit.retired_gss.is_empty());
    for (left, right) in lean.branches.iter().zip(&commit.branches) {
        assert!(left.stack.is_empty() && right.stack.is_empty());
        assert_eq!(left.stack.capacity(), 0);
        assert!(right.stack.capacity() >= crate::runtime::state::LINEAR_STACK_RESERVE);
        assert_eq!(left.offset, right.offset);
        assert_eq!(left.tokenizer_state, right.tokenizer_state);
        assert_eq!(left.processed, right.processed);
        assert_eq!(left.initial_pruned, right.initial_pruned);
        assert_eq!(left.acc, right.acc);
    }
}

const FLAT_FRONTIER_MAX_BRANCHES: usize = 128;
const FLAT_CONTINUATION_CACHE_CAPACITY: usize = 128;
const FLAT_ACTION_MAX_BRANCHES: usize = 16;
const FLAT_ACTION_MAX_STEPS: usize = 256;
const FLAT_FRONTIER_PREALLOCATED_GSS: usize = FLAT_FRONTIER_GSS_POOL_CAPACITY;

type FlatInlineStack = SmallVec<[u32; LINEAR_STACK_RESERVE]>;

#[derive(Debug, Default)]
struct FlatActionScratch {
    pending: SmallVec<[FlatInlineStack; FLAT_ACTION_MAX_BRANCHES]>,
    complete: SmallVec<[FlatInlineStack; FLAT_ACTION_MAX_BRANCHES]>,
}

impl FlatActionScratch {
    fn clear(&mut self) {
        self.pending.clear();
        self.complete.clear();
    }

    fn push_pending(&mut self, stack: FlatInlineStack) -> bool {
        if self.pending.iter().any(|existing| existing == &stack) {
            return true;
        }
        if self.pending.len() == self.pending.capacity() {
            return false;
        }
        self.pending.push(stack);
        true
    }

    fn push_complete(&mut self, stack: FlatInlineStack) -> bool {
        if self.complete.iter().any(|existing| existing == &stack) {
            return true;
        }
        if self.complete.len() == self.complete.capacity() {
            return false;
        }
        self.complete.push(stack);
        true
    }
}
// Keep a bounded reserve large enough for repeated handoffs between persistent
// GSS states and the allocation-free flat frontier.  One object reserves a
// 64-entry linear stack, so 256 spares cost about 65â€“75 KiB per active
// constraint state while avoiding allocator cliffs on multi-thousand-token
// JSON examples. The bound remains fixed; unsupported larger frontiers still
// fall back to the general persistent-GSS path.
const FLAT_FRONTIER_GSS_POOL_CAPACITY: usize = 256;
const FLAT_FRONTIER_RETIRED_GSS_CAPACITY: usize = 256;

#[derive(Debug)]
struct FlatBranchScratch {
    offset: usize,
    tokenizer_state: u32,
    stack: Vec<u32>,
    acc: TerminalsDisallowed,
    processed: bool,
    initial_pruned: bool,
}

#[derive(Clone, Copy, Debug, Default)]
struct FlatContinuationDecision {
    offset: usize,
    tokenizer_state: u32,
    end_state: u32,
    viable: bool,
}

impl Default for FlatBranchScratch {
    fn default() -> Self {
        Self::with_stack_capacity(crate::runtime::state::LINEAR_STACK_RESERVE)
    }
}

impl FlatBranchScratch {
    fn with_stack_capacity(stack_capacity: usize) -> Self {
        Self {
            offset: 0,
            tokenizer_state: 0,
            stack: Vec::with_capacity(stack_capacity),
            acc: TerminalsDisallowed::new(),
            processed: false,
            initial_pruned: false,
        }
    }
}

#[derive(Debug)]
pub(crate) struct FlatFrontierScratch {
    branches: [FlatBranchScratch; FLAT_FRONTIER_MAX_BRANCHES],
    len: usize,
    action: FlatActionScratch,
    continuation_cache: [FlatContinuationDecision; FLAT_CONTINUATION_CACHE_CAPACITY],
    continuation_cache_len: usize,
    // Double-buffered single-path GSS objects. Runtime commits write a small
    // output frontier into these preallocated objects and then swap them with
    // the active state. This permits branch creation/collapse without allocator
    // activity. Old active entries are recycled here after the atomic swap.
    gss_pool: SmallVec<[ParserGSS; FLAT_FRONTIER_GSS_POOL_CAPACITY]>,
    // Non-segment single-path GSSs displaced by a successful flat commit are
    // retained until the state is dropped. This prevents their destructors from
    // running in the token hot path without allowing unbounded retention.
    retired_gss: SmallVec<[ParserGSS; FLAT_FRONTIER_RETIRED_GSS_CAPACITY]>,
}

impl Default for FlatFrontierScratch {
    fn default() -> Self {
        Self::with_preallocated_gss(FLAT_FRONTIER_PREALLOCATED_GSS)
    }
}

impl FlatFrontierScratch {
    /// Read-only mask shadows never execute the flat commit frontier. Avoid
    /// reserving 128 unused branch stacks as well as the already-omitted GSS
    /// pool. Logical empty state is identical; ordinary commits keep their
    /// original branch-stack reservations unchanged.
    pub(crate) fn for_mask_only_shadow() -> Self {
        Self::with_reservation_policy(0, 0)
    }

    pub(crate) fn with_preallocated_gss(preallocated_gss: usize) -> Self {
        Self::with_reservation_policy(preallocated_gss, crate::runtime::state::LINEAR_STACK_RESERVE)
    }

    fn with_reservation_policy(preallocated_gss: usize, stack_capacity: usize) -> Self {
        let mut gss_pool = SmallVec::new();
        for _ in 0..preallocated_gss.min(FLAT_FRONTIER_GSS_POOL_CAPACITY) {
            let mut gss = ParserGSS::from_single_stack(
                vec![0],
                TerminalsDisallowed::new(),
            );
            let reserved = gss.reserve_single_segment_capacity(
                crate::runtime::state::LINEAR_STACK_RESERVE,
            );
            debug_assert!(reserved, "fresh flat-frontier GSS must be reservable");
            gss_pool.push(gss);
        }
        Self {
            branches: std::array::from_fn(|_| FlatBranchScratch::with_stack_capacity(stack_capacity)),
            len: 0,
            action: FlatActionScratch::default(),
            continuation_cache: [FlatContinuationDecision::default();
                FLAT_CONTINUATION_CACHE_CAPACITY],
            continuation_cache_len: 0,
            gss_pool,
            retired_gss: SmallVec::new(),
        }
    }

    pub(crate) fn clear(&mut self) {
        self.len = 0;
        self.action.clear();
        self.continuation_cache_len = 0;
    }

    fn continuation_decision(
        &self,
        offset: usize,
        tokenizer_state: u32,
        end_state: u32,
    ) -> Option<bool> {
        self.continuation_cache[..self.continuation_cache_len]
            .iter()
            .find(|decision| {
                decision.offset == offset
                    && decision.tokenizer_state == tokenizer_state
                    && decision.end_state == end_state
            })
            .map(|decision| decision.viable)
    }

    fn cache_continuation_decision(
        &mut self,
        offset: usize,
        tokenizer_state: u32,
        end_state: u32,
        viable: bool,
    ) {
        if self.continuation_cache_len == self.continuation_cache.len() {
            return;
        }
        self.continuation_cache[self.continuation_cache_len] = FlatContinuationDecision {
            offset,
            tokenizer_state,
            end_state,
            viable,
        };
        self.continuation_cache_len += 1;
    }

    fn enqueue(
        &mut self,
        offset: usize,
        tokenizer_state: u32,
        stack: &[u32],
        acc: TerminalsDisallowed,
    ) -> bool {
        if !acc.is_inline() {
            return false;
        }
        for branch in &mut self.branches[..self.len] {
            if branch.offset == offset
                && branch.tokenizer_state == tokenizer_state
                && branch.stack.as_slice() == stack
            {
                let Some(merged) = branch.acc.try_merge_inline(&acc) else {
                    return false;
                };
                branch.acc = merged;
                return true;
            }
        }
        if self.len == self.branches.len() {
            return false;
        }
        let branch = &mut self.branches[self.len];
        if stack.len() > branch.stack.capacity() {
            return false;
        }
        branch.offset = offset;
        branch.tokenizer_state = tokenizer_state;
        branch.stack.clear();
        branch.stack.extend_from_slice(stack);
        branch.acc = acc;
        branch.processed = false;
        branch.initial_pruned = false;
        self.len += 1;
        true
    }

    fn can_recycle_old_state(&self, state: &ParserStateMap, selected_count: usize) -> bool {
        let mut recyclable = 0usize;
        let mut retired = 0usize;
        for (_, gss) in &state.entries {
            if gss.can_replace_single_path_state_in_place(&[0]) {
                recyclable += 1;
            } else {
                retired += 1;
            }
        }
        self.gss_pool.len().saturating_sub(selected_count) + recyclable
            <= FLAT_FRONTIER_GSS_POOL_CAPACITY
            && self.retired_gss.len() + retired <= FLAT_FRONTIER_RETIRED_GSS_CAPACITY
    }

    fn reclaim_retired_gss(&mut self) {
        if self.retired_gss.is_empty() || self.gss_pool.len() == FLAT_FRONTIER_GSS_POOL_CAPACITY {
            return;
        }
        let mut still_retired =
            SmallVec::<[ParserGSS; FLAT_FRONTIER_RETIRED_GSS_CAPACITY]>::new();
        for gss in self.retired_gss.drain(..) {
            if self.gss_pool.len() < FLAT_FRONTIER_GSS_POOL_CAPACITY
                && gss.can_replace_single_path_state_in_place(&[0])
            {
                self.gss_pool.push(gss);
            } else {
                still_retired.push(gss);
            }
        }
        self.retired_gss = still_retired;
    }

    fn recycle_old_entries(&mut self, old_entries: SmallVec<[(u32, ParserGSS); INLINE_PARSER_STATE_CAPACITY]>) {
        self.reclaim_retired_gss();
        for (_, gss) in old_entries {
            if gss.can_replace_single_path_state_in_place(&[0]) {
                debug_assert!(self.gss_pool.len() < FLAT_FRONTIER_GSS_POOL_CAPACITY);
                self.gss_pool.push(gss);
            } else {
                debug_assert!(self.retired_gss.len() < FLAT_FRONTIER_RETIRED_GSS_CAPACITY);
                self.retired_gss.push(gss);
            }
        }
    }

    fn replace_state_with_shared_uniform_stack_keys(
        &mut self,
        state: &mut ParserStateMap,
        keys: &[u32],
        stack: &[u32],
        acc: &TerminalsDisallowed,
    ) -> bool {
        if keys.is_empty()
            || keys.windows(2).any(|pair| pair[0] >= pair[1])
            || !self.can_recycle_old_state(state, 1)
        {
            return false;
        }
        self.reclaim_retired_gss();
        let Some(pool_index) = self
            .gss_pool
            .iter()
            .rposition(|gss| gss.can_replace_single_path_state_in_place(stack))
        else {
            return false;
        };
        let mut shared = self.gss_pool.swap_remove(pool_index);
        if !shared.try_replace_single_path_state_in_place(stack, acc.clone()) {
            self.gss_pool.push(shared);
            return false;
        }

        let mut new_entries = SmallVec::<[(u32, ParserGSS); INLINE_PARSER_STATE_CAPACITY]>::new();
        new_entries.reserve(keys.len());
        for &key in &keys[..keys.len() - 1] {
            new_entries.push((key, shared.clone()));
        }
        new_entries.push((*keys.last().unwrap(), shared));
        let old_entries = std::mem::replace(&mut state.entries, new_entries);
        self.recycle_old_entries(old_entries);
        true
    }

    fn replace_state_with_uniform_stack_keys(
        &mut self,
        state: &mut ParserStateMap,
        keys: &[u32],
        stack: &[u32],
        acc: &TerminalsDisallowed,
    ) -> bool {
        if keys.is_empty()
            || keys.len() > INLINE_PARSER_STATE_CAPACITY
            || keys.windows(2).any(|pair| pair[0] >= pair[1])
            || self.gss_pool.len() < keys.len()
            || !self.can_recycle_old_state(state, keys.len())
        {
            return false;
        }

        let mut selected = SmallVec::<[ParserGSS; INLINE_PARSER_STATE_CAPACITY]>::new();
        for _ in keys {
            let Some(pool_index) = self
                .gss_pool
                .iter()
                .rposition(|gss| gss.can_replace_single_path_state_in_place(stack))
            else {
                for gss in selected.drain(..) {
                    self.gss_pool.push(gss);
                }
                return false;
            };
            selected.push(self.gss_pool.swap_remove(pool_index));
        }

        let mut new_entries = SmallVec::<[(u32, ParserGSS); INLINE_PARSER_STATE_CAPACITY]>::new();
        for (mut gss, &key) in selected.into_iter().zip(keys) {
            let replaced = gss.try_replace_single_path_state_in_place(stack, acc.clone());
            debug_assert!(replaced, "uniform flat-frontier spare eligibility changed");
            if !replaced {
                unreachable!("uniform spare was validated before mutation");
            }
            new_entries.push((key, gss));
        }

        let old_entries = std::mem::replace(&mut state.entries, new_entries);
        self.recycle_old_entries(old_entries);
        true
    }
}

const SMALL_NORMALIZED_MATCH_LINEAR_SCAN_MAX: usize = 16;

#[derive(Clone, Copy)]
struct NormalizedMatch {
    terminal_id: u32,
    width: usize,
    ignored: bool,
}

const SINGLE_CONCRETE_STACK_EFFECT_MAX_DEPTH: usize = 256;
static TEMPLATE_ADVANCE_ENABLED: OnceLock<bool> = OnceLock::new();
static VALIDATE_TEMPLATE_ADVANCE_ENABLED: OnceLock<bool> = OnceLock::new();

fn template_advance_enabled() -> bool {
    *TEMPLATE_ADVANCE_ENABLED
        .get_or_init(|| std::env::var_os("GLRMASK_ENABLE_TEMPLATE_DFA_ADVANCE").is_some())
}

#[inline]
fn template_advance_selected(constraint: &Constraint) -> bool {
    constraint.has_template_parser() || template_advance_enabled()
}

fn validate_template_advance_enabled() -> bool {
    *VALIDATE_TEMPLATE_ADVANCE_ENABLED
        .get_or_init(|| std::env::var_os("GLRMASK_VALIDATE_TEMPLATE_DFA_ADVANCE").is_some())
}

fn advance_parser_stacks(
    constraint: &Constraint,
    stack: &ParserGSS,
    terminal: u32,
) -> ParserGSS {
    if let Some(parser) = &constraint.template_parser {
        parser.record_advance();
        return advance_stacks_template_dfa(constraint, stack, terminal)
            .expect("complete template program must contain this terminal");
    }

    if let Some(advanced) = constraint.advance_compact_segmented_parser(stack, terminal) {
        return advanced;
    }
    if let Some(cached) = constraint.direct_regular_cached_advance(stack, terminal) {
        return cached;
    }
    if template_advance_selected(constraint)
        && let Some(template_advanced) = advance_stacks_template_dfa(constraint, stack, terminal)
    {
        if validate_template_advance_enabled() {
            let table_advanced = advance_stacks(&constraint.table, stack, terminal);
            assert!(
                template_advanced.semantically_eq(&table_advanced, 4_096).expect("template validation exceeded explicit stack limit"),
                "template-DFA advance mismatch for terminal {terminal}; template={:?} table={:?}",
                template_advanced.to_stacks(4_096).expect("stack enumeration exceeded explicit limit"),
                table_advanced.to_stacks(4_096).expect("stack enumeration exceeded explicit limit"),
            );
        }
        return template_advanced;
    }

    if !constraint.parser_has_controls() {
        advance_stacks(&constraint.table, stack, terminal)
    } else {
        advance_control_closed_stacks(&constraint.table, stack, terminal)
    }
}

fn advance_parser_stacks_owned(
    constraint: &Constraint,
    stack: ParserGSS,
    terminal: u32,
) -> ParserGSS {
    if let Some(parser) = &constraint.template_parser {
        parser.record_advance();
        return advance_stacks_template_dfa_owned(constraint, stack, terminal)
            .expect("complete template program must contain this terminal");
    }

    if let Some(advanced) = constraint.advance_compact_segmented_parser(&stack, terminal) {
        return advanced;
    }
    if let Some(cached) = constraint.direct_regular_cached_advance(&stack, terminal) {
        return cached;
    }
    if template_advance_selected(constraint)
        && let Some(template_advanced) =
            advance_stacks_template_dfa_owned(constraint, stack.clone(), terminal)
    {
        if validate_template_advance_enabled() {
            let table_advanced = advance_stacks_owned(&constraint.table, stack, terminal);
            assert!(
                template_advanced.semantically_eq(&table_advanced, 4_096).expect("template validation exceeded explicit stack limit"),
                "template-DFA advance mismatch for terminal {terminal}; template={:?} table={:?}",
                template_advanced.to_stacks(4_096).expect("stack enumeration exceeded explicit limit"),
                table_advanced.to_stacks(4_096).expect("stack enumeration exceeded explicit limit"),
            );
        }
        return template_advanced;
    }

    if !constraint.parser_has_controls() {
        advance_stacks_owned(&constraint.table, stack, terminal)
    } else {
        advance_control_closed_stacks_owned(&constraint.table, stack, terminal)
    }
}

fn advance_parser_stacks_profiled(
    constraint: &Constraint,
    stack: &ParserGSS,
    terminal: u32,
) -> (ParserGSS, AdvanceProfile) {
    if let Some(parser) = &constraint.template_parser {
        parser.record_advance();
        let advanced = advance_stacks_template_dfa(constraint, stack, terminal)
            .expect("complete template program must contain this terminal");
        return (advanced, AdvanceProfile::default());
    }

    let template_start = std::time::Instant::now();
    if let Some(advanced) = constraint.advance_compact_segmented_parser(stack, terminal) {
        let elapsed = template_start.elapsed().as_nanos() as u64;
        return (
            advanced,
            AdvanceProfile {
                total_ns: elapsed,
                fast_path_ns: elapsed,
                top_states: stack.top_value_count() as u32,
                gss_depth: stack.max_depth(),
                ..AdvanceProfile::default()
            },
        );
    }
    if let Some(cached) = constraint.direct_regular_cached_advance(stack, terminal) {
        let elapsed = template_start.elapsed().as_nanos() as u64;
        return (
            cached,
            AdvanceProfile {
                total_ns: elapsed,
                fast_path_ns: elapsed,
                top_states: stack.top_value_count() as u32,
                gss_depth: stack.max_depth(),
                ..AdvanceProfile::default()
            },
        );
    }
    if template_advance_selected(constraint)
        && let Some(template_advanced) = advance_stacks_template_dfa(constraint, stack, terminal)
    {
        let template_elapsed = template_start.elapsed().as_nanos() as u64;
        if validate_template_advance_enabled() {
            let (table_advanced, table_profile) =
                advance_stacks_profiled(&constraint.table, stack, terminal);
            assert!(
                template_advanced.semantically_eq(&table_advanced, 4_096).expect("template validation exceeded explicit stack limit"),
                "template-DFA advance mismatch for terminal {terminal}; template={:?} table={:?}",
                template_advanced.to_stacks(4_096).expect("stack enumeration exceeded explicit limit"),
                table_advanced.to_stacks(4_096).expect("stack enumeration exceeded explicit limit"),
            );
            return (template_advanced, table_profile);
        }
        return (
            template_advanced,
            AdvanceProfile {
                total_ns: template_elapsed,
                fast_path_ns: template_elapsed,
                top_states: stack.peek_values().len() as u32,
                gss_depth: stack.max_depth(),
                vstack_len: stack
                    .try_virtual_stack()
                    .map_or(0, |vstack| vstack.len() as u32),
                ..AdvanceProfile::default()
            },
        );
    }

    advance_stacks_profiled(&constraint.table, stack, terminal)
}

/// Advance once when admission requires exact simulation. Row-presence tables
/// retain their cheap precheck; exact-simulation tables must not execute the
/// same reduction closure once for admission and again for the actual advance.
#[inline]
pub(crate) fn parser_may_advance_on(constraint: &Constraint, stack: &ParserGSS, terminal: u32) -> bool {
    if let Some(parser) = &constraint.template_parser {
        return parser.admits(stack, terminal);
    }

    if let Some(result) = constraint.compact_segmented_parser_may_advance_on(stack, terminal) {
        return result;
    }
    constraint
        .direct_regular_admissible_terminals(stack)
        .map_or_else(
            || {
                if !constraint.parser_has_controls() {
                    stack_may_advance_on(&constraint.table, stack, terminal)
                } else {
                    stack_may_advance_on_control_closed(&constraint.table, stack, terminal)
                }
            },
            |terminals| terminals.contains(terminal as usize),
        )
}

#[inline]
fn bitset_prefix_intersects(
    left: &crate::ds::bitset::BitSet,
    right: &crate::ds::bitset::BitSet,
) -> bool {
    left.words()
        .iter()
        .zip(right.words())
        .any(|(left, right)| (*left & *right) != 0)
}

#[inline]
fn bitset_union_intersection_prefix(
    dst: &mut crate::ds::bitset::BitSet,
    left: &crate::ds::bitset::BitSet,
    right: &crate::ds::bitset::BitSet,
) {
    for ((dst, left), right) in dst
        .words_mut()
        .iter_mut()
        .zip(left.words())
        .zip(right.words())
    {
        *dst |= *left & *right;
    }
}

/// Return the sole candidate in `actions` that is requested by `terminals`
/// and not already covered by the unconditional-advance row.  `Err(())`
/// means there are at least two candidates, so the single-terminal bounded
/// shortcut is not applicable.
#[inline]
fn single_conditional_candidate(
    actions: &ActionRow,
    unconditional: &crate::ds::bitset::BitSet,
    terminals: &crate::ds::bitset::BitSet,
) -> Result<Option<u32>, ()> {
    let mut selected = None;
    for (terminal, _action) in actions.iter() {
        let bit = terminal as usize;
        if bit >= terminals.len()
            || !terminals.contains(bit)
            || unconditional.contains(bit)
        {
            continue;
        }
        if selected.is_some() {
            return Err(());
        }
        selected = Some(terminal);
    }
    Ok(selected)
}

fn exact_simulation_prefilter_may_advance_on_any(
    constraint: &Constraint,
    stack: &ParserGSS,
    terminals: &crate::ds::bitset::BitSet,
) -> Option<bool> {
    if constraint.has_template_parser() { return None; }

    if constraint.parser_admission_policy() != AdmissionPolicy::ExactSimulation
        || constraint.parser_has_controls()
        || constraint.table.unconditional_advance.len() != constraint.parser_symbol_count() as usize
    {
        return None;
    }

    let tops = stack.peek_values();
    let mut any_relevant = false;
    for &state in &tops {
        let advance = constraint.table.advance.get(state as usize)?;
        let unconditional = constraint.parser_unconditional_row(state)?;
        if bitset_prefix_intersects(unconditional, terminals) {
            return Some(true);
        }
        any_relevant |= bitset_prefix_intersects(advance, terminals);
    }
    if !any_relevant {
        return Some(false);
    }

    // The unconditional portion is already known empty.  Try the bounded
    // concrete-stack exact path when each current top has at most one remaining
    // candidate terminal; otherwise fall through to the general exact closure.
    let mut terminal_by_top = SmallVec::<[(u32, u32); 8]>::new();
    let mut bounded_applicable = true;
    'tops: for &top in &tops {
        let actions = constraint.table.action.get(top as usize)?;
        let unconditional = constraint.parser_unconditional_row(top)?;
        let selected = match single_conditional_candidate(actions, unconditional, terminals) {
            Ok(selected) => selected,
            Err(()) => {
                bounded_applicable = false;
                break 'tops;
            }
        };
        if let Some(terminal) = selected {
            terminal_by_top.push((top, terminal));
        }
    }
    if bounded_applicable
        && !terminal_by_top.is_empty()
        && let Some(result) = stack_may_advance_disjoint_top_terminals_bounded(
            &constraint.table,
            stack,
            &terminal_by_top,
        )
    {
        return Some(result);
    }
    Some(stack_may_advance_on_any(&constraint.table, stack, terminals))
}

#[inline]
fn parser_may_advance_on_any(
    constraint: &Constraint,
    stack: &ParserGSS,
    terminals: &crate::ds::bitset::BitSet,
) -> bool {
    if let Some(parser) = &constraint.template_parser {
        return parser.admits_any(stack, terminals);
    }

    if let Some(result) = constraint.compact_segmented_parser_may_advance_on_any(stack, terminals) {
        return result;
    }
    if let Some(result) = exact_simulation_prefilter_may_advance_on_any(
        constraint,
        stack,
        terminals,
    ) {
        return result;
    }
    constraint
        .direct_regular_admissible_terminals(stack)
        .map_or_else(
            || {
                if !constraint.parser_has_controls() {
                    stack_may_advance_on_any(&constraint.table, stack, terminals)
                } else {
                    stack_may_advance_on_any_control_closed(&constraint.table, stack, terminals)
                }
            },
            |admitted| {
                admitted
                    .words()
                    .iter()
                    .zip(terminals.words())
                    .any(|(left, right)| (*left & *right) != 0)
            },
        )
}

pub(crate) fn advance_parser_stacks_if_possible(
    constraint: &Constraint,
    stack: &ParserGSS,
    terminal: u32,
) -> Option<ParserGSS> {
    if constraint.parser_admission_policy() == AdmissionPolicy::RowPresenceExact
        && !parser_may_advance_on(constraint, stack, terminal)
    {
        return None;
    }
    let advanced = advance_parser_stacks(constraint, stack, terminal);
    (!advanced.is_empty()).then_some(advanced)
}

/// Advance against the authoritative composed GLR table without consulting
/// parser-DWA/direct-regular admission caches.
///
/// Segmented boundary B deliberately represents language that is absent from
/// retained component/static A. After loading a hybrid, the constraint's
/// parser DWA may therefore be exactly A; using the ordinary admission
/// prefilter here would circularly reject B-only cross-component terminals.
pub(crate) fn advance_parser_stacks_table_exact(
    constraint: &Constraint,
    stack: &ParserGSS,
    terminal: u32,
) -> Option<ParserGSS> {
    if constraint.has_template_parser() {
        return advance_parser_stacks_if_possible(constraint, stack, terminal);
    }

    if let Some(advanced) = constraint.advance_compact_segmented_parser(stack, terminal) {
        return (!advanced.is_empty()).then_some(advanced);
    }
    let advanced = if !constraint.parser_has_controls() {
        advance_stacks(&constraint.table, stack, terminal)
    } else {
        advance_control_closed_stacks(&constraint.table, stack, terminal)
    };
    (!advanced.is_empty()).then_some(advanced)
}

struct ProfiledAdvanceAttempt {
    advanced: ParserGSS,
    profile: AdvanceProfile,
    may_ns: u64,
    core_ns: u64,
}

fn advance_parser_stacks_profiled_if_possible(
    constraint: &Constraint,
    stack: &ParserGSS,
    terminal: u32,
) -> ProfiledAdvanceAttempt {
    use std::time::Instant;

    let mut may_ns = 0;
    if constraint.parser_admission_policy() == AdmissionPolicy::RowPresenceExact {
        let may_started_at = Instant::now();
        let admitted = parser_may_advance_on(constraint, stack, terminal);
        may_ns = may_started_at.elapsed().as_nanos() as u64;
        if !admitted {
            return ProfiledAdvanceAttempt {
                advanced: ParserGSS::empty(),
                profile: AdvanceProfile::default(),
                may_ns,
                core_ns: 0,
            };
        }
    }

    let core_started_at = Instant::now();
    let (advanced, profile) = advance_parser_stacks_profiled(constraint, stack, terminal);
    let core_ns = core_started_at.elapsed().as_nanos() as u64;
    ProfiledAdvanceAttempt {
        advanced,
        profile,
        may_ns,
        core_ns,
    }
}

/// Cache for `advance_stacks` results, keyed by (GSS pointer, terminal).
/// Stores the key GSS alongside the result to keep its Arc alive and prevent
/// address reuse (ABA problem) within a single `commit_bytes_impl` call.
type AdvanceResultCache = FxHashMap<(usize, u32), (ParserGSS, ParserGSS)>;

fn state_has_nonempty_accumulators(state: &ParserStateMap) -> bool {
    state
        .values()
        .any(|gss| !gss.all_accs_satisfy(|td: &TerminalsDisallowed| td.is_empty()))
}


fn parser_stacks_only(gss: &ParserGSS) -> Vec<Vec<u32>> {
    gss.to_stacks(4_096).expect("stack enumeration exceeded explicit limit").into_iter().map(|(stack, _)| stack).collect()
}


fn token_bytes_for_id(constraint: &Constraint, token_id: u32) -> Option<&[u8]> {
    constraint
        .token_bytes_dense
        .get(token_id as usize)
        .and_then(|bytes| bytes.as_deref())
        .or_else(|| constraint.token_bytes_for_id(token_id))
}

const COMMIT_ASSERT_MASK_EQUIVALENCE: u8 = 1 << 0;
const COMMIT_ASSERT_FAST_PATH_EQUIVALENCE: u8 = 1 << 1;

fn commit_assertion_flags() -> u8 {
    static FLAGS: OnceLock<u8> = OnceLock::new();
    *FLAGS.get_or_init(|| {
        let mut flags = 0;
        if cfg!(debug_assertions)
            || std::env::var("GLRMASK_ASSERT_COMMIT_TOKEN_MASK_EQUIVALENCE")
                .map(|value| {
                    let normalized = value.trim().to_ascii_lowercase();
                    matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
                })
                .unwrap_or(false)
        {
            flags |= COMMIT_ASSERT_MASK_EQUIVALENCE;
        }
        if std::env::var("GLRMASK_ASSERT_COMMIT_FAST_PATH_EQUIVALENCE")
            .map(|value| {
                let normalized = value.trim().to_ascii_lowercase();
                matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
            })
            .unwrap_or(false)
        {
            flags |= COMMIT_ASSERT_FAST_PATH_EQUIVALENCE;
        }
        flags
    })
}

fn canonical_commit_state_for_equivalence_assert(
    state: &ParserStateMap,
) -> Vec<(u32, Vec<(Vec<u32>, Vec<(u32, Vec<u32>)>)>)> {
    let mut grouped = BTreeMap::<u32, Vec<(Vec<u32>, Vec<(u32, Vec<u32>)>)>>::new();
    for (&tokenizer_state, gss) in state.iter() {
        let out = grouped.entry(tokenizer_state).or_default();
        out.extend(
            gss.to_stacks(100_000)
                .expect("stack enumeration exceeded explicit limit")
                .into_iter()
                .map(|(stack, terminals_disallowed)| {
                    let disallowed = terminals_disallowed
                        .iter()
                        .map(|(&state, terminals)| {
                            (state, terminals.iter().copied().collect::<Vec<_>>())
                        })
                        .collect::<Vec<_>>();
                    (stack, disallowed)
                }),
        );
    }
    for stacks in grouped.values_mut() {
        stacks.sort();
        stacks.dedup();
    }
    grouped.into_iter().collect()
}

fn profile_allow_fast_paths() -> bool {
    std::env::var("GLRMASK_PROFILE_ALLOW_FAST_PATHS")
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

pub(crate) fn initialize_runtime_config() {
    let _ = template_advance_enabled();
    let _ = validate_template_advance_enabled();
    let _ = commit_assertion_flags();
}

fn token_in_mask(mask: &[u32], token_id: u32) -> bool {
    let word_idx = token_id as usize / 32;
    let bit_idx = token_id as usize % 32;
    word_idx < mask.len() && ((mask[word_idx] >> bit_idx) & 1) != 0
}

fn snapshot_mask_membership(
    state: &ConstraintState<'_>,
    token_id: u32,
    assertion_flags: u8,
) -> Option<bool> {
    if assertion_flags & COMMIT_ASSERT_MASK_EQUIVALENCE == 0 {
        return None;
    }
    let mut mask = vec![0u32; state.constraint.mask_len()];
    state.fill_mask(&mut mask);
    Some(token_in_mask(&mask, token_id))
}

fn format_token_bytes(token_bytes: &[u8]) -> String {
    let mut escaped = String::new();
    for byte in token_bytes {
        for ch in std::ascii::escape_default(*byte) {
            escaped.push(ch as char);
        }
    }
    format!("b\"{}\"", escaped)
}

fn format_optional_token_bytes(token_bytes: Option<&[u8]>) -> String {
    token_bytes
        .map(format_token_bytes)
        .unwrap_or_else(|| "<no vocabulary bytes>".to_owned())
}

#[inline]
fn assert_commit_oracles(
    constraint: &Constraint,
    token_id: u32,
    token_bytes: Option<&[u8]>,
    was_in_mask: Option<bool>,
    fast_path_reference: Option<ParserStateMap>,
    actual_state: &ParserStateMap,
    commit_succeeded: bool,
) {
    if let Some(was_in_mask) = was_in_mask {
        assert!(
            commit_succeeded == was_in_mask,
            "commit/mask mismatch for token_id {} bytes {}: token_in_mask={} commit_succeeded={}",
            token_id,
            format_optional_token_bytes(token_bytes),
            was_in_mask,
            commit_succeeded,
        );
    }
    if let Some(reference_state) = fast_path_reference {
        assert_commit_fast_path_equivalence(
            constraint,
            reference_state,
            token_id,
            actual_state,
            commit_succeeded,
        );
    }
}

pub(super) fn advance_special_token_paths(
    constraint: &Constraint,
    state: &ParserStateMap,
    token_id: u32,
) -> Option<ParserGSS> {
    let mut merged = None::<ParserGSS>;

    for (&initial_state, initial_gss) in state.iter() {
        if !runtime_tokenizer_is_reset_state(constraint, initial_state) {
            continue;
        }
        for special in constraint
            .special_token_terminals
            .iter()
            .filter(|special| special.token_id == token_id)
        {
            let pruned = prune_single_initial_state_for_terminal(
                initial_gss.clone(),
                initial_state,
                special.terminal_id,
                None,
            );
            if pruned.is_empty() {
                continue;
            }
            let Some(advanced) =
                advance_parser_stacks_if_possible(constraint, &pruned, special.terminal_id)
            else {
                continue;
            };
            merged = Some(match merged.take() {
                Some(existing) => existing.merge(&advanced),
                None => advanced,
            });
        }
    }

    merged
}

#[derive(Default)]
struct SpecialTokenAdvanceProfile {
    paths: Option<ParserGSS>,
    prune_ns: u64,
    may_check_ns: u64,
    advance_ns: u64,
    summary_ns: u64,
    advances: Vec<AdvanceProfile>,
}

fn advance_special_token_paths_profiled(
    constraint: &Constraint,
    state: &ParserStateMap,
    token_id: u32,
    mut per_advance: Option<&mut Vec<PerAdvanceEntry>>,
) -> SpecialTokenAdvanceProfile {
    use std::time::Instant;

    let mut result = SpecialTokenAdvanceProfile::default();

    for (&initial_state, initial_gss) in state.iter() {
        if !runtime_tokenizer_is_reset_state(constraint, initial_state) {
            continue;
        }
        for special in constraint
            .special_token_terminals
            .iter()
            .filter(|special| special.token_id == token_id)
        {
            let prune_started_at = Instant::now();
            let pruned = prune_single_initial_state_for_terminal(
                initial_gss.clone(),
                initial_state,
                special.terminal_id,
                None,
            );
            result.prune_ns += prune_started_at.elapsed().as_nanos() as u64;
            if pruned.is_empty() {
                continue;
            }

            let attempt =
                advance_parser_stacks_profiled_if_possible(constraint, &pruned, special.terminal_id);
            result.may_check_ns += attempt.may_ns;
            result.advance_ns += attempt.core_ns;
            if attempt.advanced.is_empty() {
                continue;
            }
            let advanced = attempt.advanced;
            let advance_profile = attempt.profile;

            if let Some(entries) = per_advance.as_deref_mut() {
                result.summary_ns += record_per_advance_entry(
                    entries,
                    initial_state,
                    special.terminal_id,
                    &pruned,
                    &advanced,
                    0,
                    0,
                    0,
                    &[],
                    advance_profile.clone(),
                );
            }
            result.advances.push(advance_profile);
            result.paths = Some(match result.paths.take() {
                Some(existing) => existing.merge(&advanced),
                None => advanced,
            });
        }
    }

    result
}

fn apply_special_token_advance_profile(
    profile: &mut CommitProfile,
    special: &SpecialTokenAdvanceProfile,
) {
    profile.prune_ns += special.prune_ns;
    profile.advance_may_check_ns += special.may_check_ns;
    profile.may_advance_ns += special.may_check_ns;
    profile.advance_core_ns += special.advance_ns;
    profile.advance_ns += special.advance_ns;
    profile.adv_summary_ns += special.summary_ns;
    profile.n_advances += special.advances.len() as u64;
    for advance in &special.advances {
        apply_advance_profile(profile, advance);
    }
}

fn merge_special_token_paths(
    constraint: &Constraint,
    state: &mut ParserStateMap,
    special_paths: Option<ParserGSS>,
) -> Result<(), String> {
    let Some(gss) = special_paths.filter(|gss| !gss.is_empty()) else {
        return Ok(());
    };
    if !constraint.uses_compact_segmented_parser_runtime() {
        state.merge_insert(constraint.runtime_commit_initial_state(), gss);
        return Ok(());
    }
    let partitions = constraint
        .partition_recursive_parser_gss_by_active_leaf(&gss)
        .ok_or_else(|| "recursive special-token reset routing failed".to_owned())?;
    for (leaf_index, partition) in partitions {
        let reset_state = constraint
            .recursive_tokenizer_reset_state(leaf_index)
            .ok_or_else(|| {
                format!("recursive special-token leaf {leaf_index} has no tokenizer reset")
            })?;
        state.merge_insert(reset_state, partition);
    }
    Ok(())
}

fn finish_token_commit(state: &ParserStateMap) -> Result<(), String> {
    if state.is_empty() {
        Err("commit rejected: no valid parser states remain".to_owned())
    } else {
        Ok(())
    }
}

/// Restore the exact source-tokenizer coordinate before commit.
///
/// Invariant: all duplicate parser alternatives under one product key came
/// from the same source-state subset. The historical flat frontier decides
/// continuation viability from the union of alternatives sharing a lexer key,
/// then carries each alternative independently. Reconstruct exactly that
/// relation: compute one viable source subset from the merged group, and copy
/// every original alternative across that common subset.
fn expand_runtime_product_states(constraint: &Constraint, state: &mut ParserStateMap) {
    if constraint.uses_compact_segmented_parser_runtime() {
        // Recursive state keys are already exact disjoint-union leaf tokenizer
        // states, not product states from the transitional outer tokenizer.
        return;
    }
    let Some(source_offset) = constraint.runtime_source_state_offset() else {
        return;
    };
    if !state.keys().any(|&state| state < source_offset) {
        return;
    }

    let debug = std::env::var_os("GLRMASK_DEBUG_RUNTIME_PRODUCT").is_some();
    let old = std::mem::take(state).entries;
    let mut index = 0usize;
    while index < old.len() {
        let tokenizer_state = old[index].0;
        let group_end = old[index..]
            .partition_point(|(candidate, _)| *candidate == tokenizer_state)
            + index;
        let group = &old[index..group_end];

        if tokenizer_state >= source_offset {
            for (_, gss) in group.iter().cloned() {
                state.insert_flat_alternative(tokenizer_state, gss);
            }
            index = group_end;
            continue;
        }
        // The reset state is still one historical scanner lane whose epsilon
        // closure happens to be represented deterministically. Other product
        // states denote a *set of current runtime lanes*. Even when that set is
        // also one raw state's epsilon closure, replacing the lanes by that raw
        // state changes per-start-state longest-match behavior.
        if tokenizer_state == constraint.tokenizer.initial_state()
            && let Some(source_state) =
                constraint.runtime_product_exact_source_state(tokenizer_state)
        {
            for (_, gss) in group.iter().cloned() {
                if debug {
                    eprintln!(
                        "[glrmask/debug][runtime_product_expand] product={} exact_source={} gss={}",
                        tokenizer_state,
                        source_state,
                        gss.ptr_key(),
                    );
                }
                state.insert_flat_alternative(source_offset + source_state, gss);
            }
            index = group_end;
            continue;
        }

        debug_assert!(group
            .iter()
            .all(|(_, gss)| gss.all_accs_satisfy(|acc| acc.is_empty())));
        let Some(source_states) =
            constraint.runtime_product_source_states(tokenizer_state)
        else {
            for (_, gss) in group.iter().cloned() {
                state.insert_flat_alternative(tokenizer_state, gss);
            }
            index = group_end;
            continue;
        };
        if debug {
            eprintln!(
                "[glrmask/debug][runtime_product_expand] product={} sources={:?} alternatives={}",
                tokenizer_state,
                source_states,
                group.len(),
            );
        }
        for (_, gss) in group.iter().cloned() {
            for &source_state in source_states {
                state.insert_flat_alternative(source_offset + source_state, gss.clone());
            }
        }
        index = group_end;
    }
}

/// Re-coalesce source states only when they carry the identical multiset of
/// parser alternatives.
///
/// For source subset `S` and alternatives `G`, the exact relation is then the
/// Cartesian product `S Ã— G`, which duplicate entries under one product key
/// represent without loss. Grouping independently by GSS is insufficient:
/// distinct source groups can transition to the same product state while
/// carrying different alternative sets, erasing provenance on the next step.
fn coalesce_uniform_runtime_source_states(
    constraint: &Constraint,
    state: &mut ParserStateMap,
) {
    if constraint.uses_compact_segmented_parser_runtime() {
        return;
    }
    let Some(source_offset) = constraint.runtime_source_state_offset() else {
        return;
    };
    if !state.keys().any(|&state| state >= source_offset) {
        return;
    }

    let debug = std::env::var_os("GLRMASK_DEBUG_RUNTIME_PRODUCT").is_some();
    let old = std::mem::take(state).entries;

    // Product states are a boundary-only representation. The whole source
    // frontier may therefore collapse exactly when every source key carries
    // the same multiset of existing parser alternatives: the relation is
    // `S Ã— G`. Keep the representative GSS objects themselves so their
    // allocation-free in-place capacities and flat decomposition are retained.
    if old.is_empty() || old.iter().any(|(key, _)| *key < source_offset) {
        state.entries = old;
        return;
    }

    let first_key = old[0].0;
    let first_end = old.partition_point(|(key, _)| *key == first_key);
    let representative = &old[..first_end];
    let mut source_states = vec![first_key - source_offset];
    let mut eligible = representative
        .iter()
        .all(|(_, gss)| gss.all_accs_satisfy(|acc| acc.is_empty()));
    let mut index = first_end;
    while eligible && index < old.len() {
        let key = old[index].0;
        let end = old[index..].partition_point(|(candidate, _)| *candidate == key) + index;
        let alternatives = &old[index..end];
        if alternatives.len() != representative.len()
            || alternatives
                .iter()
                .any(|(_, gss)| !gss.all_accs_satisfy(|acc| acc.is_empty()))
        {
            eligible = false;
            break;
        }
        let mut used = [false; INLINE_PARSER_STATE_CAPACITY];
        for (_, expected) in representative {
            let Some(match_index) = alternatives
                .iter()
                .enumerate()
                .position(|(candidate_index, (_, candidate))| {
                    !used[candidate_index]
                        && (expected.ptr_eq(candidate) || expected == candidate)
                })
            else {
                eligible = false;
                break;
            };
            used[match_index] = true;
        }
        if eligible {
            source_states.push(key - source_offset);
        }
        index = end;
    }

    if eligible
        && let Some(product_state) =
            constraint.runtime_product_state_for_source_subset(&source_states)
    {
        if debug {
            eprintln!(
                "[glrmask/debug][runtime_product_coalesce] sources={:?} product={} alternatives={}",
                source_states,
                product_state,
                first_end,
            );
        }
        for (_, gss) in old.into_iter().take(first_end) {
            state.insert_flat_alternative(product_state, gss);
        }
    } else {
        if debug {
            eprintln!(
                "[glrmask/debug][runtime_product_keep_source] entries={} sources={:?}",
                old.len(),
                source_states,
            );
        }
        state.entries = old;
    }
}

fn commit_token_impl(
    constraint: &Constraint,
    state: &mut ParserStateMap,
    buffers: &mut CommitBuffers,
    token_id: u32,
) -> Result<(), String> {
    let bytes = token_bytes_for_id(constraint, token_id);
    let has_special = constraint.has_special_token_id(token_id);
    if bytes.is_none() && !has_special {
        return Err(format!(
            "commit_token: token_id {token_id} not in vocabulary or special-token terminals"
        ));
    }

    expand_runtime_product_states(constraint, state);
    let special_paths = has_special
        .then(|| advance_special_token_paths(constraint, state, token_id))
        .flatten();
    if let Some(bytes) = bytes {
        if commit_bytes_impl(constraint, state, bytes, buffers).is_err() {
            state.clear();
            buffers.reset_all();
        }
    } else {
        state.clear();
    }
    merge_special_token_paths(constraint, state, special_paths)?;
    maybe_normalize_lookahead_invariant_reductions(constraint, state);
    coalesce_uniform_runtime_source_states(constraint, state);
    finish_token_commit(state)
}

/// Exact one-token admission probe over an existing runtime state.
///
/// This is intentionally the ordinary authoritative commit implementation,
/// not a second recognizer. Recursive boundary-B masking uses it to validate
/// its small composition-owned candidate domain without translating the live
/// leaf-scoped tokenizer/parser state back into the transitional outer lexer
/// coordinate. `buffers` is reusable across probes; state-dependent caches are
/// reset between candidates while their allocated capacity is retained.
pub(crate) fn token_admissible_from_state_exact(
    constraint: &Constraint,
    source: &ParserStateMap,
    buffers: &mut CommitBuffers,
    token_id: u32,
) -> bool {
    buffers.reset_all();
    let mut probe = source.clone();
    commit_token_impl(constraint, &mut probe, buffers, token_id).is_ok()
}

/// Advance an exact runtime state through one byte prefix without committing a
/// model-token boundary. Recursive composition masking retains the returned
/// frontier at vocabulary-trie nodes and reuses it for all child edges.
pub(crate) fn advance_bytes_from_state_exact(
    constraint: &Constraint,
    source: &ParserStateMap,
    buffers: &mut CommitBuffers,
    bytes: &[u8],
) -> Option<ParserStateMap> {
    if bytes.is_empty() {
        return Some(source.clone());
    }
    buffers.reset_all();
    let mut next = source.clone();
    commit_bytes_impl(constraint, &mut next, bytes, buffers).ok()?;
    (!next.is_empty()).then_some(next)
}

/// Exact batched admission for ordinary byte-backed model tokens.
///
/// Candidates are sorted by byte string and evaluated as a radix tree.  Each
/// shared byte prefix is committed once through the ordinary authoritative
/// commit engine, then the resulting `ParserStateMap` is cheaply cloned for
/// its child branches.  A dead prefix rejects the complete subtree below it.
///
/// This helper deliberately handles byte semantics only.  Callers must keep
/// tokens with special-token semantics on `token_admissible_from_state_exact`,
/// because those tokens additionally fork parser paths that are not represented
/// by their byte spelling.
pub(crate) fn admissible_byte_token_candidates_from_state_exact<'a>(
    constraint: &Constraint,
    source: &ParserStateMap,
    buffers: &mut CommitBuffers,
    candidates: &mut Vec<(u32, &'a [u8])>,
    admitted: &mut Vec<u32>,
) {
    if candidates.is_empty() || source.is_empty() {
        return;
    }
    candidates.sort_unstable_by(|(left_id, left), (right_id, right)| {
        left.cmp(right).then_with(|| left_id.cmp(right_id))
    });

    fn walk<'a>(
        constraint: &Constraint,
        source: &ParserStateMap,
        buffers: &mut CommitBuffers,
        entries: &[(u32, &'a [u8])],
        consumed: usize,
        admitted: &mut Vec<u32>,
    ) {
        let Some((_, first)) = entries.first() else {
            return;
        };
        let last = entries.last().expect("nonempty radix candidate group").1;
        let mut common_end = consumed;
        let common_limit = first.len().min(last.len());
        while common_end < common_limit && first[common_end] == last[common_end] {
            common_end += 1;
        }

        let Some(frontier) = advance_bytes_from_state_exact(
            constraint,
            source,
            buffers,
            &first[consumed..common_end],
        ) else {
            return;
        };

        // Every candidate ending at this prefix has exactly the byte language
        // already consumed into `frontier`.  Ordinary token commit rejects iff
        // that frontier is empty; token-end normalization is non-destructive in
        // the recursive runtime, and special-token alternatives are excluded by
        // the caller.
        let mut index = 0usize;
        while index < entries.len() && entries[index].1.len() == common_end {
            admitted.push(entries[index].0);
            index += 1;
        }

        while index < entries.len() {
            debug_assert!(entries[index].1.len() > common_end);
            let branch_byte = entries[index].1[common_end];
            let mut end = index + 1;
            while end < entries.len()
                && entries[end].1.len() > common_end
                && entries[end].1[common_end] == branch_byte
            {
                end += 1;
            }
            walk(
                constraint,
                &frontier,
                buffers,
                &entries[index..end],
                common_end,
                admitted,
            );
            index = end;
        }
    }

    walk(constraint, source, buffers, candidates, 0, admitted);
}

#[cold]
pub(crate) fn prime_initial_commits(
    constraint: &Constraint,
    initial_state: &ParserStateMap,
    buffers: &mut CommitBuffers,
    token_ids: &[u32],
) {
    for &token_id in token_ids {
        let mut state = initial_state.clone();
        let _ = commit_token_impl(constraint, &mut state, buffers, token_id);
    }
}

#[cold]
fn commit_token_no_fast_path_reference(
    constraint: &Constraint,
    state: &mut ParserStateMap,
    token_id: u32,
) -> Result<(), String> {
    let bytes = token_bytes_for_id(constraint, token_id);
    let has_special = constraint.has_special_token_id(token_id);
    if bytes.is_none() && !has_special {
        return Err(format!(
            "commit_token: token_id {token_id} not in vocabulary or special-token terminals"
        ));
    }

    expand_runtime_product_states(constraint, state);
    let special_paths = has_special
        .then(|| advance_special_token_paths(constraint, state, token_id))
        .flatten();
    if let Some(bytes) = bytes {
        let mut buffers = CommitBuffers::default();
        if commit_bytes_impl_profiled(
            constraint,
            state,
            bytes,
            &mut buffers,
            None,
            false,
        )
        .is_err()
        {
            state.clear();
        }
    } else {
        state.clear();
    }
    merge_special_token_paths(constraint, state, special_paths)?;
    coalesce_uniform_runtime_source_states(constraint, state);
    finish_token_commit(state)
}

#[cold]
fn assert_commit_fast_path_equivalence(
    constraint: &Constraint,
    mut reference_state: ParserStateMap,
    token_id: u32,
    actual_state: &ParserStateMap,
    actual_succeeded: bool,
) {
    let reference_result =
        commit_token_no_fast_path_reference(constraint, &mut reference_state, token_id);
    assert_eq!(
        actual_succeeded,
        reference_result.is_ok(),
        "commit fast-path result mismatch for token_id {token_id}: actual_succeeded={} reference={:?}",
        actual_succeeded,
        reference_result,
    );
    let actual_canonical = canonical_commit_state_for_equivalence_assert(actual_state);
    let reference_canonical = canonical_commit_state_for_equivalence_assert(&reference_state);
    if actual_canonical != reference_canonical {
        // Bounded flat frontiers deliberately preserve lexer/parser
        // correlation that the map-only reference normalizes away. The fast
        // state can therefore be a strict internal refinement of the reference
        // without changing the accepted continuation language. Keep exact
        // state equality as the strongest oracle, but fall back to the public
        // semantic observations when the reference has merged correlations.
        let mut actual_semantic = constraint.start();
        actual_semantic.state = actual_state.clone();
        let mut reference_semantic = constraint.start();
        reference_semantic.state = reference_state;
        assert_eq!(
            actual_semantic.is_accepting(),
            reference_semantic.is_accepting(),
            "commit fast-path completion mismatch for token_id {token_id} bytes {}\nactual={actual_canonical:?}\nreference={reference_canonical:?}",
            format_optional_token_bytes(token_bytes_for_id(constraint, token_id)),
        );
        assert_eq!(
            actual_semantic.mask(),
            reference_semantic.mask(),
            "commit fast-path successor-mask mismatch for token_id {token_id} bytes {}\nactual={actual_canonical:?}\nreference={reference_canonical:?}",
            format_optional_token_bytes(token_bytes_for_id(constraint, token_id)),
        );
    }
}

#[inline]
fn runtime_tokenizer_is_reset_state(constraint: &Constraint, state: u32) -> bool {
    if constraint.uses_compact_segmented_parser_runtime() {
        constraint.recursive_tokenizer_is_reset_state(state)
    } else {
        state == constraint.runtime_commit_initial_state()
    }
}

#[inline]
fn runtime_tokenizer_future_terminals(
    constraint: &Constraint,
    state: u32,
) -> Option<Cow<'_, crate::ds::bitset::BitSet>> {
    if constraint.uses_compact_segmented_parser_runtime() {
        constraint.recursive_tokenizer_future_scoped_terminals(state)
    } else {
        Some(Cow::Borrowed(
            constraint.tokenizer.possible_future_terminals(state),
        ))
    }
}

#[inline]
fn runtime_terminal_count(constraint: &Constraint) -> usize {
    constraint
        .uses_compact_segmented_parser_runtime()
        .then(|| constraint.recursive_runtime_terminal_count())
        .flatten()
        .unwrap_or(constraint.parser_terminal_count() as usize)
}

/// An unfinished ignore lexeme is a valid byte prefix even though completing
/// it will not shift a parser terminal. Lexer-continuation tests must retain
/// that route independently of the parser's ordinary terminal admission.
#[inline]
fn runtime_future_contains_ignore(
    constraint: &Constraint,
    future: &crate::ds::bitset::BitSet,
) -> bool {
    if constraint.uses_compact_segmented_parser_runtime() {
        future.iter_ones().any(|terminal| {
            constraint.recursive_terminal_is_ignore(terminal as u32)
        })
    } else {
        constraint.ignore_terminal.is_some_and(|terminal| future.contains(terminal as usize))
    }
}

#[inline]
fn end_state_may_advance(constraint: &Constraint, gss: &ParserGSS, end_state: u32) -> bool {
    if runtime_tokenizer_is_reset_state(constraint, end_state) {
        return true;
    }
    let future = runtime_tokenizer_future_terminals(constraint, end_state)
        .expect("runtime tokenizer end state must belong to its active coordinate");
    parser_may_advance_on_any(constraint, gss, future.as_ref())
        || runtime_future_contains_ignore(constraint, future.as_ref())
}

/// For a tokenizer execution that produced several end states against the same
/// parser GSS, compute exact parser admission once over the union of their
/// possible-future terminal sets.  Individual end-state viability is then an
/// intersection with this admitted set.
///
/// This is exactly equivalent to repeated `end_state_may_advance` because
///
///   exists t in F_i: can_advance(G, t)
///
/// iff `F_i` intersects the exact admitted set over `union_i F_i`.
///
/// Single non-initial end states deliberately keep the old boolean path: the
/// exact-set computation cannot beat its early-exit simulation in that case.
fn batched_end_state_admitted_terminals(
    constraint: &Constraint,
    gss: &ParserGSS,
    end_states: &[u32],
) -> Option<crate::ds::bitset::BitSet> {
    // Template admission is an existential domain query: its exact boolean
    // path can stop at a suffix-independent witness. Computing the full set
    // here would classify unrelated conditional terminals before intersecting
    // each lexer future, defeating that shortcut.
    if constraint.has_template_parser() { return None; }
    let mut candidates = crate::ds::bitset::BitSet::new(runtime_terminal_count(constraint));
    let mut non_initial = 0usize;
    for &end_state in end_states {
        if runtime_tokenizer_is_reset_state(constraint, end_state) {
            continue;
        }
        non_initial += 1;
        let future = runtime_tokenizer_future_terminals(constraint, end_state)?;
        candidates.union_with_prefix(future.as_ref());
    }
    if non_initial <= 1 {
        return None;
    }
    if let Some(admitted) = constraint.compact_segmented_parser_admitted_terminals(gss, &candidates) {
        return Some(admitted);
    }
    if let Some(direct) = constraint.direct_regular_admissible_terminals(gss) {
        let mut admitted = candidates;
        admitted.intersect_with(&direct);
        Some(admitted)
    } else {
        Some(exact_simulation_prefiltered_admitted_terminals(
            constraint,
            gss,
            &candidates,
        ))
    }
}


const PARSER_ADMISSION_CACHE_CAPACITY: usize = 8;
const PARSER_ADMISSION_BOOLEAN_CACHE_CAPACITY: usize = 8;

fn exact_simulation_prefiltered_admitted_terminals(
    constraint: &Constraint,
    gss: &ParserGSS,
    candidates: &crate::ds::bitset::BitSet,
) -> crate::ds::bitset::BitSet {
    if let Some(parser) = &constraint.template_parser {
        return parser.admitted(gss, candidates);
    }

    if constraint.parser_admission_policy() != AdmissionPolicy::ExactSimulation
        || constraint.parser_has_controls()
        || constraint.table.unconditional_advance.len() != constraint.parser_symbol_count() as usize
    {
        return stack_admissible_terminals(&constraint.table, gss, candidates);
    }

    let mut guaranteed = crate::ds::bitset::BitSet::new(candidates.len());
    let mut unresolved = crate::ds::bitset::BitSet::new(candidates.len());
    for state in gss.peek_values() {
        let Some(advance) = constraint.table.advance.get(state as usize) else {
            return stack_admissible_terminals(&constraint.table, gss, candidates);
        };
        let Some(unconditional) = constraint.parser_unconditional_row(state) else {
            return stack_admissible_terminals(&constraint.table, gss, candidates);
        };
        bitset_union_intersection_prefix(&mut guaranteed, unconditional, candidates);
        bitset_union_intersection_prefix(&mut unresolved, advance, candidates);
    }
    for (unresolved_word, guaranteed_word) in unresolved
        .words_mut()
        .iter_mut()
        .zip(guaranteed.words())
    {
        *unresolved_word &= !*guaranteed_word;
    }
    if unresolved.is_empty() {
        return guaranteed;
    }
    let simulated = stack_admissible_terminals(&constraint.table, gss, &unresolved);
    guaranteed.union_with(&simulated);
    guaranteed
}

#[inline]
pub(crate) fn exact_admitted_terminals_for_candidates(
    constraint: &Constraint,
    gss: &ParserGSS,
    candidates: &crate::ds::bitset::BitSet,
) -> crate::ds::bitset::BitSet {
    if let Some(parser) = &constraint.template_parser {
        return parser.admitted(gss, candidates);
    }

    if let Some(admitted) = constraint.compact_segmented_parser_admitted_terminals(gss, candidates) {
        return admitted;
    }
    if let Some(direct) = constraint.direct_regular_admissible_terminals(gss) {
        let mut admitted = candidates.clone();
        admitted.intersect_with(&direct);
        admitted
    } else {
        exact_simulation_prefiltered_admitted_terminals(constraint, gss, candidates)
    }
}

fn admission_cache_entry_index(
    cache: &mut SmallVec<[ParserAdmissionCacheEntry; 8]>,
    gss: &ParserGSS,
    terminal_count: usize,
) -> usize {
    if let Some(index) = cache
        .iter()
        .position(|entry| entry.gss.ptr_eq(gss))
    {
        return index;
    }
    if cache.len() >= PARSER_ADMISSION_CACHE_CAPACITY {
        cache.remove(0);
    }
    cache.push(ParserAdmissionCacheEntry {
        gss: gss.clone(),
        tested: crate::ds::bitset::BitSet::new(terminal_count),
        admitted: crate::ds::bitset::BitSet::new(terminal_count),
        boolean_queries: SmallVec::new(),
    });
    cache.len() - 1
}

fn try_local_row_presence_admission_words(
    constraint: &Constraint,
    gss: &ParserGSS,
    end_states: &[u32],
) -> Option<[u64; 32]> {
    const WORDS: usize = 32;
    if constraint.uses_compact_segmented_parser_runtime()
        || constraint.parser_admission_policy() != AdmissionPolicy::ExactSimulation
        || constraint.parser_has_controls()
        || constraint.tokenizer.num_terminals() as usize > WORDS * 64
        || (!constraint.has_template_parser()
            && (constraint.table.advance.len() != constraint.parser_symbol_count() as usize
                || constraint.table.unconditional_advance.len()
                    != constraint.parser_symbol_count() as usize))
    {
        return None;
    }
    let tops = gss.peek_values();
    if tops.is_empty() {
        return None;
    }

    // Only terminals reachable from the tokenizer continuation matter. Both
    // backends provide a sound upper bound P and a suffix-independent lower
    // bound U. If (P - U) intersects none of these candidates, U is exact for
    // this query. Template rows are derived from the input-domain automata;
    // this shortcut neither reads an LR action nor materializes output stacks.
    let initial = constraint.runtime_commit_initial_state();
    let mut candidates = [0u64; WORDS];
    for &end_state in end_states {
        if end_state == initial {
            continue;
        }
        for (index, &word) in constraint
            .tokenizer
            .possible_future_terminals(end_state)
            .words()
            .iter()
            .enumerate()
        {
            if index >= WORDS {
                return None;
            }
            candidates[index] |= word;
        }
    }

    let mut admitted = [0u64; WORDS];
    for state in tops {
        let advance = constraint.parser_advance_row(state)?;
        let unconditional = constraint.parser_unconditional_row(state)?;
        for index in 0..advance.words().len().min(WORDS) {
            let candidate_word = candidates[index];
            if candidate_word == 0 {
                continue;
            }
            let advance_word = advance.words()[index];
            let unconditional_word = unconditional.words().get(index).copied().unwrap_or(0);
            if (advance_word & !unconditional_word & candidate_word) != 0 {
                return None;
            }
            admitted[index] |= unconditional_word & candidate_word;
        }
    }
    Some(admitted)
}

#[inline]
fn end_state_may_advance_from_row_words(
    constraint: &Constraint,
    end_state: u32,
    admitted_words: &[u64; 32],
) -> bool {
    if runtime_tokenizer_is_reset_state(constraint, end_state) {
        return true;
    }
    let future = runtime_tokenizer_future_terminals(constraint, end_state)
        .expect("runtime tokenizer end state must belong to its active coordinate");
    future
        .words()
        .iter()
        .zip(admitted_words.iter())
        .any(|(&future, admitted)| (future & *admitted) != 0)
        || runtime_future_contains_ignore(constraint, future.as_ref())
}

/// Cached exact-set version of `batched_end_state_admitted_terminals`.
/// Returns the cache entry containing exact admission facts for every terminal
/// occurring in these end-state future sets. Single-end-state callers use the
/// cheaper boolean cache instead.
fn cached_batched_end_state_admission(
    constraint: &Constraint,
    gss: &ParserGSS,
    end_states: &[u32],
    cache: &mut SmallVec<[ParserAdmissionCacheEntry; 8]>,
) -> Option<usize> {
    if constraint.has_template_parser() { return None; }
    let mut candidates = crate::ds::bitset::BitSet::new(runtime_terminal_count(constraint));
    let mut non_initial = 0usize;
    for &end_state in end_states {
        if runtime_tokenizer_is_reset_state(constraint, end_state) {
            continue;
        }
        non_initial += 1;
        let future = runtime_tokenizer_future_terminals(constraint, end_state)?;
        candidates.union_with_prefix(future.as_ref());
    }
    if non_initial <= 1 {
        return None;
    }
    let index = admission_cache_entry_index(cache, gss, candidates.len());
    let delta = candidates.difference(&cache[index].tested);
    if !delta.is_empty() {
        let newly_admitted = exact_admitted_terminals_for_candidates(constraint, gss, &delta);
        let entry = &mut cache[index];
        entry.tested.union_with(&delta);
        entry.admitted.union_with(&newly_admitted);
    }
    Some(index)
}

/// Exact cached boolean admission for one tokenizer end state. If a prior
/// batched query has already classified all future terminals, answer directly
/// from the pointwise cache. Otherwise cache the old exact existential result
/// for this complete future set.
fn cached_single_end_state_may_advance(
    constraint: &Constraint,
    gss: &ParserGSS,
    end_state: u32,
    cache: &mut SmallVec<[ParserAdmissionCacheEntry; 8]>,
) -> bool {
    if runtime_tokenizer_is_reset_state(constraint, end_state) {
        return true;
    }
    let Some(future) = runtime_tokenizer_future_terminals(constraint, end_state) else {
        return false;
    };
    let future = future.as_ref();
    let index = admission_cache_entry_index(cache, gss, future.len());
    {
        let entry = &cache[index];
        if future.is_subset_of_extended(&entry.tested) {
            return !future.is_disjoint(&entry.admitted)
                || runtime_future_contains_ignore(constraint, future);
        }
        if let Some((_, result)) = entry
            .boolean_queries
            .iter()
            .find(|(query, _)| query == future)
        {
            return *result;
        }
    }
    let result = parser_may_advance_on_any(constraint, gss, future)
        || runtime_future_contains_ignore(constraint, future);
    let entry = &mut cache[index];
    if entry.boolean_queries.len() >= PARSER_ADMISSION_BOOLEAN_CACHE_CAPACITY {
        entry.boolean_queries.remove(0);
    }
    entry.boolean_queries.push((future.clone(), result));
    result
}

#[inline]
fn end_state_may_advance_from_cache_entry(
    constraint: &Constraint,
    end_state: u32,
    entry: &ParserAdmissionCacheEntry,
) -> bool {
    if runtime_tokenizer_is_reset_state(constraint, end_state) {
        return true;
    }
    let future = runtime_tokenizer_future_terminals(constraint, end_state)
        .expect("runtime tokenizer end state must belong to its active coordinate");
    !entry.admitted.is_disjoint_prefix(future.as_ref())
        || runtime_future_contains_ignore(constraint, future.as_ref())
}


#[inline]
fn end_state_may_advance_with_batch(
    constraint: &Constraint,
    gss: &ParserGSS,
    end_state: u32,
    admitted: Option<&crate::ds::bitset::BitSet>,
) -> bool {
    if runtime_tokenizer_is_reset_state(constraint, end_state) {
        return true;
    }
    match admitted {
        Some(admitted) => {
            let future = runtime_tokenizer_future_terminals(constraint, end_state)
                .expect("runtime tokenizer end state must belong to its active coordinate");
            !admitted.is_disjoint_prefix(future.as_ref())
                || runtime_future_contains_ignore(constraint, future.as_ref())
        }
        None => end_state_may_advance(constraint, gss, end_state),
    }
}

#[inline]
fn wide_frontier_end_state_may_advance(
    constraint: &Constraint,
    summary: &crate::runtime::artifact::DirectRegularWideFrontierAcceptance,
    end_state: u32,
) -> bool {
    if end_state == constraint.tokenizer.initial_state() {
        return true;
    }
    summary
        .actionable_terminals
        .words()
        .iter()
        .zip(constraint.tokenizer.possible_future_terminals(end_state).words())
        .any(|(actionable, future)| (*actionable & *future) != 0)
        || runtime_future_contains_ignore(constraint, constraint.tokenizer.possible_future_terminals(end_state))
}

enum ActionableTerminals {
    DirectDynamic(crate::ds::bitset::BitSet),
    SingleState(u32),
    WideFrontier(usize),
    ManyStates(SmallVec<[u32; 8]>),
}

impl ActionableTerminals {
    fn from_gss(constraint: &Constraint, gss: &ParserGSS) -> Option<Self> {
        if constraint.uses_compact_segmented_parser_runtime() {
            return None;
        }
        if let Some(terminals) = constraint.direct_regular_admissible_terminals(gss) {
            return Some(Self::DirectDynamic(terminals));
        }
        if let Some(state_id) = gss.single_top_value() {
            return Some(Self::SingleState(state_id));
        }
        if let Some(index) = constraint.direct_regular_wide_frontier_index_for_gss(gss) {
            return Some(Self::WideFrontier(index));
        }

        let states = gss.peek_values();
        if states.is_empty() {
            None
        } else {
            Some(Self::ManyStates(states))
        }
    }

    fn bitset<'a>(&'a self, constraint: &'a Constraint) -> Option<&'a crate::ds::bitset::BitSet> {
        match self {
            Self::DirectDynamic(terminals) => Some(terminals),
            Self::SingleState(state_id) => constraint.parser_advance_row(*state_id),
            Self::WideFrontier(index) => constraint
                .direct_regular_wide_frontier_acceptance
                .get(*index)
                .map(|summary| &summary.actionable_terminals),
            Self::ManyStates(_) => None,
        }
    }

    fn contains(&self, constraint: &Constraint, terminal: u32) -> bool {
        match self {
            Self::DirectDynamic(terminals) => terminals.contains(terminal as usize),
            Self::SingleState(state_id) => constraint.parser_advance_row_allows(*state_id, terminal),
            Self::WideFrontier(index) => constraint
                .direct_regular_wide_frontier_acceptance
                .get(*index)
                .is_some_and(|summary| summary.actionable_terminals.contains(terminal as usize)),
            Self::ManyStates(states) => states
                .iter()
                .any(|state_id| constraint.parser_advance_row_allows(*state_id, terminal)),
        }
    }
}

impl InitialCommitScan {
    fn collect(
        constraint: &Constraint,
        state: &ParserStateMap,
        bytes: &[u8],
    ) -> Self {
        let mut exec_results = FxHashMap::default();

        for &tokenizer_state in state.keys() {
            let exec_result = execute_tokenizer_from_state_small(constraint, bytes, tokenizer_state);
            exec_results.insert(tokenizer_state, exec_result);
        }

        Self { exec_results }
    }

    fn take_exec_result(&mut self, tokenizer_state: u32) -> Option<TokenizerExecResult> {
        self.exec_results.remove(&tokenizer_state)
    }
}

fn is_ignored_terminal(ignore_terminal: Option<u32>, terminal: u32) -> bool {
    Some(terminal) == ignore_terminal
}

#[inline]
fn runtime_is_ignored_terminal(
    constraint: &Constraint,
    ignore_terminal: Option<u32>,
    terminal: u32,
) -> bool {
    if constraint.uses_compact_segmented_parser_runtime() {
        constraint.recursive_terminal_is_ignore(terminal)
    } else {
        is_ignored_terminal(ignore_terminal, terminal)
    }
}

fn is_actionable_terminal(
    actionable_terminals: Option<&ActionableTerminals>,
    constraint: &Constraint,
    terminal: u32,
) -> bool {
    !actionable_terminals
        .is_some_and(|actionable| !actionable.contains(constraint, terminal))
}


fn for_each_relevant_matched_terminal(
    constraint: &Constraint,
    tokenizer_state: u32,
    actionable_terminals: Option<&ActionableTerminals>,
    mut visit: impl FnMut(u32, bool),
) {
    let matched = constraint.tokenizer.matched_terminal_bitset(tokenizer_state);
    if let Some(actionable) = actionable_terminals.and_then(|value| value.bitset(constraint)) {
        let ignored = constraint.ignore_terminal;
        for (word_index, (&matched_word, &actionable_word)) in
            matched.words().iter().zip(actionable.words()).enumerate()
        {
            let mut intersection = matched_word & actionable_word;
            while intersection != 0 {
                let bit = intersection.trailing_zeros() as usize;
                let terminal = (word_index * 64 + bit) as u32;
                if Some(terminal) != ignored {
                    visit(terminal, false);
                }
                intersection &= intersection - 1;
            }
        }
        if let Some(ignored) = ignored
            && matched.contains(ignored as usize)
        {
            visit(ignored, true);
        }
        return;
    }

    for terminal in constraint.tokenizer.matched_terminals_iter(tokenizer_state) {
        let ignored = is_ignored_terminal(constraint.ignore_terminal, terminal);
        if ignored || is_actionable_terminal(actionable_terminals, constraint, terminal) {
            visit(terminal, ignored);
        }
    }
}

fn scan_wide_frontier_lexer_only(
    constraint: &Constraint,
    bytes: &[u8],
    start_state: u32,
    summary: &crate::runtime::artifact::DirectRegularWideFrontierAcceptance,
) -> Option<u32> {
    if constraint.tokenizer_has_epsilon_transitions {
        return None;
    }
    let mut state = start_state;
    for &byte in bytes {
        state = constraint.tokenizer_fast_transitions.transition(
            &constraint.tokenizer,
            state,
            byte,
        );
        if state == u32::MAX {
            return None;
        }
        let matched = constraint.tokenizer.matched_terminal_bitset(state);
        if matched
            .words()
            .iter()
            .zip(summary.actionable_terminals.words())
            .any(|(left, right)| (*left & *right) != 0)
            || constraint
                .ignore_terminal
                .is_some_and(|terminal| matched.contains(terminal as usize))
        {
            return None;
        }
    }
    Some(state)
}

fn advance_uniform_disallowed_interest_only(
    constraint: &Constraint,
    terminals_disallowed: &TerminalsDisallowed,
    bytes: &[u8],
) -> Option<TerminalsDisallowed> {
    if terminals_disallowed.is_empty() {
        return Some(TerminalsDisallowed::new());
    }
    if constraint.tokenizer_has_epsilon_transitions {
        return None;
    }

    let mut remapped = BTreeMap::<u32, BTreeSet<u32>>::new();
    for (&continuation_state, disallowed) in terminals_disallowed.iter() {
        let mut state = continuation_state;
        let mut alive = true;
        for &byte in bytes {
            state = constraint.tokenizer_fast_transitions.transition(
                &constraint.tokenizer,
                state,
                byte,
            );
            if state == u32::MAX {
                alive = false;
                break;
            }
            let matched = constraint.tokenizer.matched_terminal_bitset(state);
            if disallowed
                .iter()
                .any(|terminal| matched.contains(*terminal as usize))
            {
                return None;
            }
        }
        if !alive {
            continue;
        }
        let future = constraint.tokenizer.possible_future_terminals(state);
        for &terminal in disallowed {
            if future.contains(terminal as usize) {
                remapped.entry(state).or_default().insert(terminal);
            }
        }
    }
    Some(TerminalsDisallowed::from_map(remapped))
}

fn collect_unique_actionable_reusable_matches(
    constraint: &Constraint,
    actionable_terminals: Option<&ActionableTerminals>,
    ignore_terminal: Option<u32>,
    matches: &[TokenizerMatch],
) -> SmallVec<[NormalizedMatch; 16]> {
    // `execute_tokenizer_reusable*` canonicalizes to one record per terminal at
    // its longest width, so no second duplicate-removal pass is needed here.
    let mut normalized = SmallVec::<[NormalizedMatch; 16]>::new();
    for matched in matches {
        let ignored = runtime_is_ignored_terminal(constraint, ignore_terminal, matched.id);
        if !ignored && !is_actionable_terminal(actionable_terminals, constraint, matched.id) {
            continue;
        }
        normalized.push(NormalizedMatch {
            terminal_id: matched.id,
            width: matched.width,
            ignored,
        });
    }
    normalized
}

fn collect_unique_actionable_matches(
    constraint: &Constraint,
    actionable_terminals: Option<&ActionableTerminals>,
    ignore_terminal: Option<u32>,
    matches: &[TokenizerMatch],
    reusable_seen_matches: Option<&mut FxHashSet<(usize, u32)>>,
) -> SmallVec<[NormalizedMatch; 16]> {
    let mut normalized = SmallVec::<[NormalizedMatch; 16]>::new();

    if matches.len() <= SMALL_NORMALIZED_MATCH_LINEAR_SCAN_MAX {
        'matches: for matched in matches {
            let ignored = runtime_is_ignored_terminal(constraint, ignore_terminal, matched.id);
            if !ignored && !is_actionable_terminal(actionable_terminals, constraint, matched.id) {
                continue;
            }
            for existing in &normalized {
                if existing.width == matched.width && existing.terminal_id == matched.id {
                    continue 'matches;
                }
            }
            normalized.push(NormalizedMatch {
                terminal_id: matched.id,
                width: matched.width,
                ignored,
            });
        }
        return normalized;
    }

    if let Some(seen_matches) = reusable_seen_matches {
        seen_matches.clear();
        for matched in matches {
            let ignored = runtime_is_ignored_terminal(constraint, ignore_terminal, matched.id);
            if !ignored && !is_actionable_terminal(actionable_terminals, constraint, matched.id) {
                continue;
            }
            if !seen_matches.insert((matched.width, matched.id)) {
                continue;
            }
            normalized.push(NormalizedMatch {
                terminal_id: matched.id,
                width: matched.width,
                ignored,
            });
        }
        return normalized;
    }

    let mut seen_matches = FxHashSet::default();
    for matched in matches {
        let ignored = runtime_is_ignored_terminal(constraint, ignore_terminal, matched.id);
        if !ignored && !is_actionable_terminal(actionable_terminals, constraint, matched.id) {
            continue;
        }
        if !seen_matches.insert((matched.width, matched.id)) {
            continue;
        }
        normalized.push(NormalizedMatch {
            terminal_id: matched.id,
            width: matched.width,
            ignored,
        });
    }
    normalized
}

fn prune_single_initial_state_for_exec(
    constraint: &Constraint,
    gss: ParserGSS,
    tokenizer_state: u32,
    exec_result: &TokenizerExecResult,
    bytes: &[u8],
) -> ParserGSS {
    prune_single_initial_state_for_parts(
        constraint,
        gss,
        tokenizer_state,
        &exec_result.end_state,
        &exec_result.matches,
        bytes,
    )
}

fn advance_terminals_disallowed_over_bytes(
    constraint: &Constraint,
    terminals_disallowed: &TerminalsDisallowed,
    bytes: &[u8],
    reusable_execution: Option<(u32, &[u32], &[TokenizerMatch])>,
) -> Option<TerminalsDisallowed> {
    if terminals_disallowed.is_empty() {
        return Some(TerminalsDisallowed::new());
    }

    let mut remapped = BTreeMap::new();
    for (&continuation_tokenizer_state, disallowed) in terminals_disallowed.iter() {
        let owned_execution;
        let (end_states, matches) = match reusable_execution {
            Some((state, end_states, matches)) if state == continuation_tokenizer_state => {
                (end_states, matches)
            }
            _ => {
                owned_execution = execute_tokenizer_from_state_small(
                    constraint,
                    bytes,
                    continuation_tokenizer_state,
                );
                (&owned_execution.end_state[..], &owned_execution.matches[..])
            }
        };
        if matches.iter()
            .any(|matched| disallowed.contains(&matched.id))
        {
            return None;
        }
        for &end_state in end_states {
            let future = runtime_tokenizer_future_terminals(constraint, end_state)?;
            for &terminal in disallowed.iter() {
                if future.contains(terminal as usize) {
                    remapped
                        .entry(end_state)
                        .or_insert_with(BTreeSet::new)
                        .insert(terminal);
                }
            }
        }
    }
    Some(TerminalsDisallowed::from_map(remapped))
}

fn single_disallowed_pair(acc: &TerminalsDisallowed) -> Option<(u32, u32)> {
    if acc.len() != 1 {
        return None;
    }
    let mut states = acc.iter();
    let (state, terminals) = states.next()?;
    if states.next().is_some() || terminals.len() != 1 {
        return None;
    }
    Some((*state, *terminals.iter().next()?))
}

fn try_prune_single_initial_state_batched_accumulators(
    constraint: &Constraint,
    gss: &ParserGSS,
    bytes: &[u8],
    scratch: &mut tokenizer_scan::ReusableTokenizerExecScratch,
    cached_starts: &mut SmallVec<[u32; 8]>,
) -> Option<ParserGSS> {
    let mut accumulators = SmallVec::<[TerminalsDisallowed; 8]>::new();
    let mut overflow = false;
    gss.for_each_acc(|acc| {
        if overflow || acc.is_empty() || accumulators.contains(acc) {
            return;
        }
        if accumulators.len() == accumulators.capacity() {
            overflow = true;
            return;
        }
        accumulators.push(acc.clone());
    });
    if overflow || accumulators.len() < 2 {
        return None;
    }

    let mut pairs = SmallVec::<[(u32, u32); 8]>::new();
    for acc in &accumulators {
        pairs.push(single_disallowed_pair(acc)?);
    }
    // Each logical exclusion must belong to a disjoint lexer lane.  Then one
    // union execution preserves lane-local longest-match semantics because no
    // terminal ID can be observed from two starts.
    for i in 0..pairs.len() {
        let left = constraint.tokenizer.possible_future_terminals(pairs[i].0);
        for &(right_state, _) in &pairs[..i] {
            if !left.is_disjoint(
                constraint.tokenizer.possible_future_terminals(right_state),
            ) {
                return None;
            }
        }
    }
    let starts = pairs
        .iter()
        .map(|&(state, _)| state)
        .collect::<SmallVec<[u32; 8]>>();
    if !execute_tokenizer_reusable_from_states(constraint, bytes, &starts, scratch) {
        return None;
    }
    cached_starts.clear();
    cached_starts.extend(starts.iter().copied());

    let mut remapped = SmallVec::<[(TerminalsDisallowed, Option<TerminalsDisallowed>); 8]>::new();
    for (acc, &(_, terminal)) in accumulators.iter().zip(&pairs) {
        if scratch.matches.iter().any(|matched| matched.id == terminal) {
            remapped.push((acc.clone(), None));
            continue;
        }
        let mut next = TerminalsDisallowed::new();
        for &end_state in &scratch.states {
            if constraint
                .tokenizer
                .possible_future_terminals(end_state)
                .contains(terminal as usize)
            {
                next = next.with_insert(end_state, terminal);
            }
        }
        remapped.push((acc.clone(), Some(next)));
    }

    Some(gss.apply_and_prune_no_promote(|acc| {
        if acc.is_empty() {
            return Some(TerminalsDisallowed::new());
        }
        remapped
            .iter()
            .find(|(source, _)| source == acc)
            .and_then(|(_, result)| result.clone())
    }))
}

fn prune_single_initial_state_for_parts(
    constraint: &Constraint,
    gss: ParserGSS,
    tokenizer_state: u32,
    end_states: &[u32],
    matches: &[TokenizerMatch],
    bytes: &[u8],
) -> ParserGSS {
    gss.apply_and_prune_no_promote(|terminals_disallowed: &TerminalsDisallowed| {
        advance_terminals_disallowed_over_bytes(
            constraint,
            terminals_disallowed,
            bytes,
            Some((tokenizer_state, end_states, matches)),
        )
    })
}

fn prune_single_initial_state_for_terminal(
    gss: ParserGSS,
    tokenizer_state: u32,
    terminal: u32,
    end_state: Option<u32>,
) -> ParserGSS {
    if end_state.is_none()
        && gss.all_accs_satisfy(|td: &TerminalsDisallowed| {
            td.get(&tokenizer_state)
                .is_none_or(|disallowed| !disallowed.contains(&terminal))
        })
    {
        return gss.apply(|_: &TerminalsDisallowed| TerminalsDisallowed::new());
    }

    gss.apply_and_prune_no_promote(|terminals_disallowed: &TerminalsDisallowed| {
        if terminals_disallowed.is_empty() {
            return Some(TerminalsDisallowed::new());
        }
        if let Some(disallowed) = terminals_disallowed.get(&tokenizer_state) {
            if disallowed.contains(&terminal) {
                return None;
            }
        }

        let mut remapped = BTreeMap::new();
        if let Some(end_state) = end_state {
            if let Some(disallowed) = terminals_disallowed.get(&tokenizer_state) {
                remapped
                    .entry(end_state)
                    .or_insert_with(BTreeSet::new)
                    .extend(disallowed.iter().copied());
            }
        }
        Some(TerminalsDisallowed::from_map(remapped))
    })
}

fn merge_parser_state(
    states: &mut ParserStatesByTokenizer,
    tokenizer_state: u32,
    gss: ParserGSS,
) {
    states
        .entry(tokenizer_state)
        .and_modify(|existing| *existing = existing.merge(&gss))
        .or_insert(gss);
}

fn queue_parser_state(
    processing_queue: &mut [ParserStatesByTokenizer],
    pending_state: &mut ParserStatesByTokenizer,
    new_offset: usize,
    total_len: usize,
    tokenizer_state: u32,
    gss: ParserGSS,
) {
    if new_offset == total_len {
        merge_parser_state(pending_state, tokenizer_state, gss);
    } else {
        merge_parser_state(&mut processing_queue[new_offset], tokenizer_state, gss);
    }
}

fn queue_parser_reset_state(
    constraint: &Constraint,
    processing_queue: &mut [ParserStatesByTokenizer],
    pending_state: &mut ParserStatesByTokenizer,
    new_offset: usize,
    total_len: usize,
    gss: ParserGSS,
) -> bool {
    if !constraint.uses_compact_segmented_parser_runtime() {
        queue_parser_state(
            processing_queue,
            pending_state,
            new_offset,
            total_len,
            constraint.runtime_commit_initial_state(),
            gss,
        );
        return true;
    }
    let Some(partitions) = constraint.partition_recursive_parser_gss_by_active_leaf(&gss) else {
        return false;
    };
    for (leaf_index, partition) in partitions {
        let Some(reset_state) = constraint.recursive_tokenizer_reset_state(leaf_index) else {
            return false;
        };
        queue_parser_state(
            processing_queue,
            pending_state,
            new_offset,
            total_len,
            reset_state,
            partition,
        );
    }
    true
}

fn queue_parser_ignored_reset_state(
    constraint: &Constraint,
    processing_queue: &mut [ParserStatesByTokenizer],
    pending_state: &mut ParserStatesByTokenizer,
    new_offset: usize,
    total_len: usize,
    gss: ParserGSS,
) -> bool {
    let gss = if constraint.uses_compact_segmented_parser_runtime() {
        constraint
            .close_compact_segmented_parser(&gss)
            .unwrap_or(gss)
    } else {
        gss
    };
    queue_parser_reset_state(
        constraint,
        processing_queue,
        pending_state,
        new_offset,
        total_len,
        gss,
    )
}

fn finalize_pending_state(
    pending_state: &mut ParserStatesByTokenizer,
) -> ParserStateMap {
    match pending_state.len() {
        0 => ParserStateMap::default(),
        1 => {
            let (tokenizer_state, parser_state) = pending_state.drain().next().unwrap();
            let fused = parser_state.fuse(Some(1));
            if fused.is_empty() {
                ParserStateMap::default()
            } else {
                ParserStateMap::singleton(tokenizer_state, fused)
            }
        }
        _ => {
            let mut new_state: ParserStateMap = pending_state.drain().collect();
            for parser_state in new_state.values_mut() {
                *parser_state = parser_state.fuse(Some(1));
            }
            new_state.retain(|_, parser_state| !parser_state.is_empty());
            new_state
        }
    }
}

fn apply_future_terminal_disallow(
    constraint: &Constraint,
    exec_result: &TokenizerExecResult,
    terminal: u32,
    gss: ParserGSS,
) -> ParserGSS {
    apply_future_terminal_disallow_for_states(
        constraint,
        &exec_result.end_state,
        terminal,
        gss,
    )
}

fn apply_future_terminal_disallow_for_states(
    constraint: &Constraint,
    end_states: &[u32],
    terminal: u32,
    gss: ParserGSS,
) -> ParserGSS {
    if gss.is_empty() || end_states.is_empty() {
        return gss;
    }
    let relevant: SmallVec<[u32; INLINE_PARSER_STATE_CAPACITY]> = end_states
        .iter()
        .copied()
        .filter(|&end_state| {
            runtime_tokenizer_future_terminals(constraint, end_state)
                .is_some_and(|future| future.contains(terminal as usize))
        })
        .collect();
    if relevant.is_empty() {
        return gss;
    }

    gss.apply(|terminals_disallowed: &TerminalsDisallowed| {
        let mut updated = terminals_disallowed.clone();
        for &end_state in &relevant {
            updated = updated.with_insert(end_state, terminal);
        }
        updated
    })
}

#[inline]
fn try_apply_single_top_action_in_place(gss: &mut ParserGSS, action: &Action) -> bool {
    match action {
        Action::Skip => true,
        Action::Shift(target, replace) => {
            let pushes = [*target];
            gss.try_apply_single_segment_stack_effect_in_place(usize::from(*replace), &pushes)
        }
        Action::ReplaceShifts(targets) if targets.len() == 1 => {
            gss.try_apply_single_segment_stack_effect_in_place(1, targets)
        }
        Action::StackShifts(shifts) => {
            let [shift] = shifts.as_slice() else {
                return false;
            };
            gss.try_apply_single_segment_stack_effect_in_place(
                shift.pop as usize,
                &shift.pushes,
            )
        }
        _ => false,
    }
}

#[inline]
fn apply_single_top_action_fast(
    constraint: &Constraint,
    gss: &ParserGSS,
    state: u32,
    terminal: u32,
    action: &Action,
) -> Option<ParserGSS> {
    if let Some(cached) = constraint.direct_regular_cached_advance(gss, terminal) {
        return Some(cached);
    }
    let table = &constraint.table;
    match action {
        Action::Skip => Some(gss.clone()),
        Action::Shift(target, is_replace) => {
            if let Some(mut stack) = gss.try_virtual_stack() {
                if *is_replace && stack.pop(1) != 0 {
                    return Some(gss.popn(1).push(*target));
                }
                stack.push(*target);
                return Some(stack.into_gss());
            } else {
                Some(if *is_replace {
                    gss.popn(1).push(*target)
                } else {
                    gss.push(*target)
                })
            }
        }
        Action::ReplaceShifts(targets) => {
            let stack = gss.try_virtual_stack()?;
            stack.into_gss_after_popping_and_pushing_unique_single_branches(
                1,
                targets.iter(),
            )
        }
        Action::StackShifts(shifts) => {
            if let [shift] = shifts.as_slice() {
                let mut branch = gss.try_virtual_stack()?;
                if branch.pop(shift.pop as usize) != 0 {
                    return None;
                }
                for &target in &shift.pushes {
                    branch.push(target);
                }
                return Some(branch.into_gss());
            }
            if let Some(stack) = gss.try_virtual_stack()
                && let Some(first) = shifts.first()
                && !first.pushes.is_empty()
                && shifts
                    .iter()
                    .all(|shift| shift.pop == first.pop && !shift.pushes.is_empty())
                && let Some(shifted) = stack.into_gss_after_popping_and_pushing_branches(
                    first.pop as usize,
                    shifts.iter().map(|shift| shift.pushes.as_slice()),
                )
            {
                return Some(shifted);
            }
            if let Some(shifted) = gss.apply_stack_effects_to_single_concrete_path(
                shifts
                    .iter()
                    .map(|shift| (shift.pop as usize, shift.pushes.as_slice())),
                SINGLE_CONCRETE_STACK_EFFECT_MAX_DEPTH,
            ) {
                return Some(shifted);
            }
            if let Some(first) = shifts.first()
                && shifts
                    .iter()
                    .all(|shift| shift.pop == first.pop && shift.pushes.len() == 1)
                && let Some(shifted) = gss.apply_shared_pop_push_single_branches(
                    first.pop as usize,
                    shifts.iter().map(|shift| &shift.pushes[0]),
                )
            {
                return Some(shifted);
            }
            if let Some(first) = shifts.first()
                && !first.pushes.is_empty()
                && shifts
                    .iter()
                    .all(|shift| shift.pop == first.pop && !shift.pushes.is_empty())
                && let Some(shifted) = gss.apply_shared_pop_push_branches(
                    first.pop as usize,
                    shifts.iter().map(|shift| shift.pushes.as_slice()),
                )
            {
                return Some(shifted);
            }

            let stack = gss.try_virtual_stack()?;
            let mut shifted = ParserGSS::empty();
            for shift in shifts {
                let mut branch = stack.clone();
                if branch.pop(shift.pop as usize) != 0 {
                    return None;
                }
                for &target in &shift.pushes {
                    branch.push(target);
                }
                let branch = branch.into_gss();
                shifted = if shifted.is_empty() {
                    branch
                } else {
                    shifted.merge(&branch)
                };
            }
            Some(shifted)
        }
        Action::GuardedStackShifts(shifts) => {
            apply_guarded_stack_shifts_fast(gss, shifts, table.guarded_shift_index(state, terminal))
        }
        Action::Reduce(..) => apply_single_path_reduce_chain_fast(table, gss, terminal),
        _ => None,
    }
}

fn try_apply_action_to_carried_virtual_stack(
    stack: &mut crate::ds::leveled_gss::VirtualStack<u32, TerminalsDisallowed>,
    action: &Action,
) -> bool {
    match action {
        Action::Skip => true,
        Action::Shift(target, is_replace) => {
            if *is_replace {
                stack.replace_top(*target)
            } else {
                stack.push(*target);
                true
            }
        }
        Action::StackShifts(shifts) => {
            let [shift] = shifts.as_slice() else {
                return false;
            };
            let mut candidate = stack.clone();
            if candidate.pop(shift.pop as usize) != 0 {
                return false;
            }
            for &target in &shift.pushes {
                candidate.push(target);
            }
            if candidate.top().is_none() {
                // The next parser/top-row decision requires a concrete top
                // state. If the visible prefix has been exhausted, continuing
                // to carry this virtual stack would make the stale materialized
                // GSS an invalid proxy for the current parser frontier.
                return false;
            }
            *stack = candidate;
            true
        }
        _ => false,
    }
}

/// Preserve a uniform reduction result's shared bottom-of-stack prefix.
/// Rebuilding the entire concrete-stack trie repeats work for each retained
/// value, although every output has the same accumulator and usually differs
/// only in its final one or two states. Mixed accumulators retain the general
/// builder: exclusions must never be detached from their particular paths.
fn materialize_reduce_chain_outputs(
    mut outputs: Vec<(Vec<u32>, TerminalsDisallowed)>,
) -> ParserGSS {
    if outputs.len() == 1 {
        let (stack, accumulator) = outputs.pop().unwrap();
        return ParserGSS::from_single_stack(stack, accumulator);
    }
    let Some((first, accumulator)) = outputs.first() else {
        return ParserGSS::empty();
    };
    if outputs.iter().any(|(_, other)| other != accumulator) {
        return ParserGSS::from_stacks(&outputs);
    }
    let mut shared = first.len();
    for (stack, _) in outputs.iter().skip(1) {
        shared = first[..shared].iter().zip(stack).take_while(|(a, b)| a == b).count();
    }
    if shared == 0 {
        return ParserGSS::from_stacks(&outputs);
    }

    let base = ParserGSS::from_single_stack(first[..shared].to_vec(), accumulator.clone());
    let accepts_prefix = outputs.iter().any(|(stack, _)| stack.len() == shared);
    if outputs.iter().all(|(stack, _)| stack.len() == shared) {
        return base;
    }
    let branches = base.apply_shared_pop_push_branches(
        0,
        outputs.iter().filter(|(stack, _)| stack.len() > shared)
            .map(|(stack, _)| &stack[shared..]),
    ).expect("a nonempty single-stack prefix admits every nonempty suffix");
    if accepts_prefix { base.merge(&branches) } else { branches }
}

fn apply_single_path_reduce_chain_fast(
    table: &GLRTable,
    gss: &ParserGSS,
    terminal: u32,
) -> Option<ParserGSS> {
    let (mut stack, acc) =
        gss.try_single_stack_bounded(SINGLE_CONCRETE_STACK_EFFECT_MAX_DEPTH)?;

    loop {
        let state = *stack.last()?;
        match table.action(state, terminal)? {
            Action::Skip => {
                return Some(ParserGSS::from_single_stack(stack, acc));
            }
            Action::Reduce(nt, len) => {
                let rhs_len = *len as usize;
                if rhs_len >= stack.len() {
                    return None;
                }
                stack.truncate(stack.len() - rhs_len);
                let goto_from = *stack.last()?;
                let (target, is_replace) = table.goto_target(goto_from, *nt)?;
                if is_replace {
                    *stack.last_mut()? = target;
                } else {
                    stack.push(target);
                }
            }
            Action::Shift(target, is_replace) => {
                if *is_replace {
                    *stack.last_mut()? = *target;
                } else {
                    stack.push(*target);
                }
                return Some(ParserGSS::from_single_stack(stack, acc));
            }
            Action::StackShifts(shifts) => {
                return ParserGSS::from_single_stack(stack, acc)
                    .apply_stack_effects_to_single_concrete_path(
                        shifts
                            .iter()
                            .map(|shift| (shift.pop as usize, shift.pushes.as_slice())),
                        SINGLE_CONCRETE_STACK_EFFECT_MAX_DEPTH,
                    );
            }
            Action::Split {
                shift,
                reduces,
                accept: false,
            } => {
                let mut out: Vec<(Vec<u32>, TerminalsDisallowed)> = Vec::new();

                if let Some((target, is_replace)) = shift {
                    let mut branch = stack.clone();
                    if *is_replace {
                        *branch.last_mut()? = *target;
                    } else {
                        branch.push(*target);
                    }
                    out.push((branch, acc.clone()));
                }

                for &(nt, len) in reduces {
                    let mut branch = stack.clone();
                    let rhs_len = len as usize;
                    if rhs_len >= branch.len() {
                        return None;
                    }
                    branch.truncate(branch.len() - rhs_len);
                    let goto_from = *branch.last()?;
                    let (target, is_replace) = table.goto_target(goto_from, nt)?;
                    if is_replace {
                        *branch.last_mut()? = target;
                    } else {
                        branch.push(target);
                    }

                    let follow_state = *branch.last()?;
                    match table.action(follow_state, terminal)? {
                        Action::Skip => {
                            out.push((branch, acc.clone()));
                        }
                        Action::Shift(target, is_replace) => {
                            if *is_replace {
                                *branch.last_mut()? = *target;
                            } else {
                                branch.push(*target);
                            }
                            out.push((branch, acc.clone()));
                        }
                        Action::StackShifts(shifts) => {
                            let shifted = ParserGSS::from_single_stack(branch, acc.clone())
                                .apply_stack_effects_to_single_concrete_path(
                                    shifts
                                        .iter()
                                        .map(|shift| (shift.pop as usize, shift.pushes.as_slice())),
                                    SINGLE_CONCRETE_STACK_EFFECT_MAX_DEPTH,
                                )?;
                            let shifted_stacks = shifted
                                .to_stacks(shifts.len())
                                .expect("stack-shift result exceeded its effect count");
                            out.extend(shifted_stacks);
                        }
                        _ => return None,
                    }
                }

                return (!out.is_empty()).then(|| materialize_reduce_chain_outputs(out));
            }
            _ => return None,
        }
    }
}

fn advance_terminal_match(
    constraint: &Constraint,
    gss_at_offset: &ParserGSS,
    terminal: u32,
    exec_result: &TokenizerExecResult,
    advance_result_cache: &mut AdvanceResultCache,
    terminal_result_cache: &mut FxHashMap<u32, ParserGSS>,
) -> Option<ParserGSS> {
    if let Some(cached) = terminal_result_cache.get(&terminal) {
        return (!cached.is_empty()).then(|| cached.clone());
    }

    let advance_cache_key = (gss_at_offset.ptr_key(), terminal);
    let advanced = if let Some((_, cached)) = advance_result_cache.get(&advance_cache_key) {
        cached.clone()
    } else {
        let advanced = advance_parser_stacks(constraint, gss_at_offset, terminal);
        advance_result_cache.insert(advance_cache_key, (gss_at_offset.clone(), advanced.clone()));
        advanced
    };

    let advanced = apply_future_terminal_disallow(constraint, exec_result, terminal, advanced);
    terminal_result_cache.insert(terminal, advanced.clone());
    (!advanced.is_empty()).then_some(advanced)
}

/// Fast path for the common case: exactly 1 tokenizer state, the tokenizer
/// produces exactly 1 non-ignored terminal match that consumes all bytes,
/// and no pending end-state needs to be queued. This avoids:
/// - FxHashMap allocations (InitialCommitScan, seen_matches, caches)
/// - Processing queue allocation
/// - Prune iteration (when terminals_disallowed is empty)
///
/// Returns `Some(Ok(()))` on success, `Some(Err(...))` on rejection,
/// or `None` to fall through to the general path.
///
/// `exec_result` is the pre-computed tokenizer output for the single state.
fn commit_bytes_fast_path(
    constraint: &Constraint,
    state: &mut ParserStateMap,
    bytes: &[u8],
    tokenizer_state: u32,
    exec_result: &TokenizerExecResult,
) -> Option<Result<(), String>> {
    let gss = state.values().next().unwrap();
    let ignore_terminal = constraint.ignore_terminal;
    let has_linker_controls = constraint.parser_has_controls()
        || constraint.uses_compact_segmented_parser_runtime();

    // Find exactly 1 non-ignored, actionable terminal match consuming all bytes
    let mut sole_terminal: Option<u32> = None;
    for matched in &exec_result.matches {
        if matched.width != bytes.len() {
            return None;
        }
        if is_ignored_terminal(ignore_terminal, matched.id) {
            return None;
        }
        if !parser_may_advance_on(constraint, gss, matched.id) {
            continue;
        }
        if sole_terminal.is_some() {
            return None;
        }
        sole_terminal = Some(matched.id);
    }
    let terminal = sole_terminal?;

    let no_end_state = exec_result.end_state.is_empty();
    let accs_empty = gss.all_accs_satisfy(|td: &TerminalsDisallowed| td.is_empty());
    // The stale-exclusion bug that originally required the epsilon guard only
    // exists when exclusions must be transported from one NFA configuration
    // to another.  With empty accumulators this routine is already fully
    // state-set aware: it advances the sole full-width terminal and preserves
    // every viable lexer continuation independently.
    if constraint.tokenizer_has_epsilon_transitions && !accs_empty {
        return None;
    }
    let all_accs_empty = no_end_state && accs_empty;

    // Ultra-fast path: single Interface, empty accs, no end_state, pure shift.
    // Inlines the entire advance + prune + fuse to avoid all function call overhead.
    if !has_linker_controls && all_accs_empty && !template_advance_selected(constraint) {
        let top_state = gss.single_exclusive_top_value();
        if let Some(top_state) = top_state {
            if let Some(action) = constraint.table.action(top_state, terminal) {
                if let Some(gss) = state.values_mut().next()
                    && try_apply_single_top_action_in_place(gss, action)
                {
                    return Some(Ok(()));
                }
                let gss = state.values().next().unwrap();
                if let Some(shifted) =
                    apply_single_top_action_fast(constraint, gss, top_state, terminal, action)
                {
                    state.clear();
                    state.insert(constraint.runtime_commit_initial_state(), shifted);
                    return Some(Ok(()));
                }
            }
        }
    }

    // Take ownership of the GSS for the standard fast path.
    // This allows advance_stacks_owned to avoid cloning the inner Arc.
    let (_, gss_owned) = state.pop_first().unwrap();

    // Standard fast path: skip prune when accumulators are empty.
    let pruned_gss = if accs_empty {
        gss_owned
    } else {
        let pruned = prune_single_initial_state_for_exec(
            constraint,
            gss_owned,
            tokenizer_state,
            exec_result,
            bytes,
        );

        if pruned.is_empty() {
            return Some(Err(
                "commit rejected: no valid parser states remain".to_string(),
            ));
        }
        pruned
    };

    let end_states_to_keep: TokenizerStateSet = exec_result
        .end_state
        .iter()
        .copied()
        .filter(|&end_state| end_state_may_advance(constraint, &pruned_gss, end_state))
        .collect();

    // The terminal and tokenizer end-state continuations are independent.
    // Preserve either branch if it produces viable parser state.
    let advanced = if !has_linker_controls
        && !template_advance_selected(constraint)
        && let Some(top_state) = pruned_gss.single_exclusive_top_value()
        && let Some(action) = constraint.table.action(top_state, terminal)
        && let Some(advanced) = apply_single_top_action_fast(
            constraint,
            &pruned_gss,
            top_state,
            terminal,
            action,
        )
    {
        advanced
    } else {
        advance_parser_stacks_owned(constraint, pruned_gss.clone(), terminal)
    };
    let mut produced_state = false;
    if !advanced.is_empty() {
        let advanced =
            apply_future_terminal_disallow(constraint, &exec_result, terminal, advanced);
        if !advanced.is_empty() {
            let fused = advanced.fuse(Some(1));
            if !fused.is_empty() {
                state.insert(constraint.runtime_commit_initial_state(), fused);
                produced_state = true;
            }
        }
    }

    if !end_states_to_keep.is_empty() {
        let fused = pruned_gss.fuse(Some(1));
        if !fused.is_empty() {
            for &end_state in &end_states_to_keep {
                state.merge_insert(end_state, fused.clone());
            }
            produced_state = true;
        }
    }

    if !produced_state {
        return Some(Err(
            "commit rejected: no valid parser states remain".to_string(),
        ));
    }
    Some(Ok(()))
}

fn commit_bytes_full_width_fast_path(
    constraint: &Constraint,
    state: &mut ParserStateMap,
    bytes: &[u8],
) -> Option<Result<(), String>> {
    let has_linker_controls = constraint.parser_has_controls()
        || constraint.uses_compact_segmented_parser_runtime();
    if constraint.tokenizer_has_epsilon_transitions
        && state_has_nonempty_accumulators(state)
    {
        return None;
    }
    if state.len() > 2 {
        return None;
    }
    if state.len() > 1 && bytes.len() > 4 && state_has_nonempty_accumulators(state) {
        return None;
    }

    let mut output = ParserStatesByTokenizer::default();
    for (&tokenizer_state, gss) in state.iter() {
        let exec_result = execute_tokenizer_from_state_small(constraint, bytes, tokenizer_state);
        let actionable_terminals = (!has_linker_controls)
            .then(|| ActionableTerminals::from_gss(constraint, gss))
            .flatten();
        let mut terminal = None;

        for matched in &exec_result.matches {
            if matched.width != bytes.len()
                || is_ignored_terminal(constraint.ignore_terminal, matched.id)
            {
                return None;
            }
            if !is_actionable_terminal(actionable_terminals.as_ref(), constraint, matched.id) {
                continue;
            }
            if terminal.is_some_and(|existing| existing != matched.id) {
                return None;
            }
            terminal = Some(matched.id);
        }

        let pruned_gss = if gss.all_accs_satisfy(|td: &TerminalsDisallowed| td.is_empty()) {
            gss.clone()
        } else {
            let pruned = prune_single_initial_state_for_exec(
                constraint,
                gss.clone(),
                tokenizer_state,
                &exec_result,
                bytes,
            );
            if pruned.is_empty() {
                continue;
            }
            pruned
        };

        if let Some(terminal) = terminal {
            let advanced = if !has_linker_controls
                && !template_advance_selected(constraint)
                && let Some(top_state) = pruned_gss.single_exclusive_top_value()
                && let Some(action) = constraint.table.action(top_state, terminal)
                && let Some(advanced) =
                    apply_single_top_action_fast(
                        constraint,
                        &pruned_gss,
                        top_state,
                        terminal,
                        action,
                    )
            {
                advanced
            } else {
                advance_parser_stacks_if_possible(constraint, &pruned_gss, terminal)?
            };
            if advanced.is_empty() {
                continue;
            }
            let advanced = apply_future_terminal_disallow(
                constraint,
                &exec_result,
                terminal,
                advanced,
            );
            if !advanced.is_empty() {
                merge_parser_state(
                    &mut output,
                    constraint.runtime_commit_initial_state(),
                    advanced,
                );
            }
        }

        let admitted_end_state_terminals = batched_end_state_admitted_terminals(
            constraint,
            &pruned_gss,
            &exec_result.end_state,
        );
        for &end_state in &exec_result.end_state {
            if end_state_may_advance_with_batch(
                constraint,
                &pruned_gss,
                end_state,
                admitted_end_state_terminals.as_ref(),
            ) {
                merge_parser_state(&mut output, end_state, pruned_gss.clone());
            }
        }
    }

    let new_state = finalize_pending_state(&mut output);
    if new_state.is_empty() {
        return Some(Err(
            "commit rejected: no valid parser states remain".to_string(),
        ));
    }
    *state = new_state;
    Some(Ok(()))
}

fn merge_small_parser_state(
    states: &mut SmallVec<[(u32, ParserGSS); INLINE_PARSER_STATE_CAPACITY]>,
    tokenizer_state: u32,
    gss: ParserGSS,
) {
    for (existing_state, existing_gss) in states.iter_mut() {
        if *existing_state == tokenizer_state {
            *existing_gss = existing_gss.merge(&gss);
            return;
        }
    }
    states.push((tokenizer_state, gss));
}

fn merge_small_language_parser_state(
    runtime: &mut TemplateAdvanceRuntime,
    states: &mut SmallLanguageParserStates,
    tokenizer_state: u32,
    language: u32,
    accumulator: TerminalsDisallowed,
) -> bool {
    if language == 0 {
        return true;
    }
    for existing in states.iter_mut() {
        if existing.tokenizer_state == tokenizer_state && existing.accumulator == accumulator {
            existing.language = runtime.union_languages(existing.language, language);
            return !runtime.is_exhausted();
        }
    }
    if states.len() == SMALL_LANGUAGE_QUEUE_CAPACITY {
        return false;
    }
    states.push(SmallLanguageParserState {
        tokenizer_state,
        language,
        accumulator,
    });
    true
}

fn actionable_terminals_from_language(
    runtime: &TemplateAdvanceRuntime,
    language: u32,
) -> Option<ActionableTerminals> {
    let states = runtime.language_top_states(language);
    match states.as_slice() {
        [] => None,
        [state] => Some(ActionableTerminals::SingleState(*state)),
        _ => Some(ActionableTerminals::ManyStates(states)),
    }
}

fn prune_uniform_accumulator_for_parts(
    constraint: &Constraint,
    actionable_terminals: Option<&ActionableTerminals>,
    accumulator: &TerminalsDisallowed,
    tokenizer_state: u32,
    end_states: &[u32],
    matches: &[TokenizerMatch],
) -> Option<TerminalsDisallowed> {
    if accumulator.is_empty() {
        return Some(TerminalsDisallowed::new());
    }

    let accepted_terminals = matches
        .iter()
        .filter(|matched| !is_ignored_terminal(constraint.ignore_terminal, matched.id))
        .filter(|matched| {
            is_actionable_terminal(actionable_terminals, constraint, matched.id)
        })
        .map(|matched| matched.id)
        .collect::<SmallVec<[u32; INLINE_PARSER_STATE_CAPACITY]>>();

    if let Some(disallowed) = accumulator.get(&tokenizer_state)
        && !accepted_terminals.is_empty()
        && accepted_terminals
            .iter()
            .all(|terminal| disallowed.contains(terminal))
    {
        return None;
    }

    if let Some(remapped) =
        accumulator.try_remap_single_state_inline(tokenizer_state, end_states)
    {
        return Some(remapped);
    }

    let terminals = accumulator
        .get(&tokenizer_state)
        .map(|values| values.iter().copied().collect::<Vec<_>>())
        .unwrap_or_default();
    if terminals.is_empty() || end_states.is_empty() {
        return Some(TerminalsDisallowed::new());
    }
    let mut remapped = BTreeMap::<u32, BTreeSet<u32>>::new();
    for &end_state in end_states {
        remapped
            .entry(end_state)
            .or_default()
            .extend(terminals.iter().copied());
    }
    Some(TerminalsDisallowed::from_map(remapped))
}

fn apply_future_terminal_disallow_to_accumulator(
    constraint: &Constraint,
    end_states: &[u32],
    terminal: u32,
    mut accumulator: TerminalsDisallowed,
) -> TerminalsDisallowed {
    for &end_state in end_states {
        if constraint
            .tokenizer
            .possible_future_terminals(end_state)
            .contains(terminal as usize)
        {
            accumulator = accumulator.with_insert(end_state, terminal);
        }
    }
    accumulator
}

fn language_end_state_may_advance(
    constraint: &Constraint,
    runtime: &mut TemplateAdvanceRuntime,
    language: u32,
    end_state: u32,
) -> Option<bool> {
    if end_state == constraint.runtime_commit_initial_state() {
        return Some(true);
    }
    if runtime_future_contains_ignore(constraint, constraint.tokenizer.possible_future_terminals(end_state)) {
        return Some(true);
    }
    for terminal in constraint
        .tokenizer
        .possible_future_terminals(end_state)
        .iter_ones()
    {
        let terminal = u32::try_from(terminal).ok()?;
        let advanced = runtime.advance_language(constraint, terminal, language)?;
        if advanced != 0 {
            return Some(true);
        }
    }
    Some(false)
}

fn language_queue_top_value_count_at_most(
    state: &ParserStateMap,
    limit: usize,
) -> usize {
    let mut count = 0usize;
    for (_, gss) in state.iter() {
        count = count.saturating_add(gss.top_value_count()).min(limit);
        if count == limit {
            break;
        }
    }
    count
}

fn language_queue_path_count_at_most(state: &ParserStateMap, limit: usize) -> usize {
    let mut count = 0usize;
    for (_, gss) in state.iter() {
        count = count.saturating_add(gss.path_count_at_most(limit)).min(limit);
        if count == limit {
            break;
        }
    }
    count
}

fn language_queue_node_count_at_most(state: &ParserStateMap, limit: usize) -> usize {
    let mut count = 0usize;
    for (_, gss) in state.iter() {
        count = count.saturating_add(gss.node_count_at_most(limit)).min(limit);
        if count == limit {
            break;
        }
    }
    count
}

fn language_small_queue_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let flag = |name: &str| {
            std::env::var(name).ok().map(|value| {
                let normalized = value.trim().to_ascii_lowercase();
                !matches!(normalized.as_str(), "" | "0" | "false" | "no" | "off")
            })
        };
        if flag("GLRMASK_DISABLE_LANGUAGE_SMALL_QUEUE") == Some(true) {
            return false;
        }
        flag("GLRMASK_ENABLE_LANGUAGE_SMALL_QUEUE").unwrap_or(true)
    })
}

fn language_queue_input_is_bounded(state: &ParserStateMap) -> bool {
    state
        .iter()
        .all(|(_, gss)| gss.max_depth() <= LANGUAGE_QUEUE_MAX_INPUT_STACK_DEPTH)
}

/// Return whether one model token contains multiple parser-actionable,
/// non-ignored terminal completion boundaries at different byte offsets.
///
/// This is the structural case where the ordinary byte queue must materialize
/// and carry parser states at more than one offset inside the same token. The
/// language queue evaluates all such offset alternatives before reconstructing
/// a GSS once at token completion. Multiple terminals ending at the same byte
/// offset do not qualify: the ordinary queue already merges those without an
/// offset frontier split.
fn has_multiple_actionable_terminal_boundaries(
    constraint: &Constraint,
    state: &ParserStateMap,
    bytes: &[u8],
    tokenizer_scratch: &mut tokenizer_scan::ReusableTokenizerExecScratch,
) -> bool {
    for (&tokenizer_state, gss) in state.iter() {
        if !execute_tokenizer_reusable(
            constraint,
            bytes,
            tokenizer_state,
            tokenizer_scratch,
        ) {
            return false;
        }
        let actionable = ActionableTerminals::from_gss(constraint, gss);
        let matches = collect_unique_actionable_matches(
            constraint,
            actionable.as_ref(),
            constraint.ignore_terminal,
            &tokenizer_scratch.matches,
            None,
        );
        let mut widths = SmallVec::<[usize; 4]>::new();
        for matched in matches {
            if matched.ignored || widths.contains(&matched.width) {
                continue;
            }
            widths.push(matched.width);
            if widths.len() >= 2 {
                return true;
            }
        }
    }
    false
}

#[derive(Default)]
struct LanguageCommitSimulationProfile {
    canonicalize_ns: u64,
    evaluate_ns: u64,
    continuation_reconstruct_ns: u64,
    continuation_check_ns: u64,
}

fn simulate_language_commit(
    constraint: &Constraint,
    state: &ParserStateMap,
    bytes: &[u8],
    tokenizer_scratch: &mut tokenizer_scan::ReusableTokenizerExecScratch,
    queue_scratch: &mut SmallCommitQueueScratch,
    template_runtime: &mut TemplateAdvanceRuntime,
    mut profile: Option<&mut LanguageCommitSimulationProfile>,
) -> Option<Result<SmallLanguageParserStates, String>> {
    if bytes.is_empty() || bytes.len() > 8 || state.is_empty() || state.len() > 8 {
        return None;
    }

    queue_scratch.clear();
    for (&tokenizer_state, gss) in state.iter() {
        let started = profile.is_some().then(std::time::Instant::now);
        let components = template_runtime.language_components_from_gss(gss)?;
        if let (Some(profile), Some(started)) = (profile.as_deref_mut(), started) {
            profile.canonicalize_ns += started.elapsed().as_nanos() as u64;
        }
        if template_runtime.is_exhausted() {
            return None;
        }
        for (language, accumulator) in components {
            if !merge_small_language_parser_state(
                template_runtime,
                &mut queue_scratch.language_processing[0],
                tokenizer_state,
                language,
                accumulator,
            ) {
                return None;
            }
        }
    }

    let initial_tokenizer_state = constraint.runtime_commit_initial_state();
    let mut offset = 0usize;
    while offset <= bytes.len() {
        if queue_scratch.language_processing[offset].is_empty() {
            offset += 1;
            continue;
        }

        let states_to_process =
            std::mem::take(&mut queue_scratch.language_processing[offset]);
        for mut entry in states_to_process {
            if !execute_tokenizer_reusable(
                constraint,
                &bytes[offset..],
                entry.tokenizer_state,
                tokenizer_scratch,
            ) {
                return None;
            }

            let actionable_terminals =
                actionable_terminals_from_language(template_runtime, entry.language);
            if offset == 0 && !entry.accumulator.is_empty() {
                entry.accumulator = prune_uniform_accumulator_for_parts(
                    constraint,
                    actionable_terminals.as_ref(),
                    &entry.accumulator,
                    entry.tokenizer_state,
                    &tokenizer_scratch.states,
                    &tokenizer_scratch.matches,
                )?;
            }

            let normalized_matches = collect_unique_actionable_matches(
                constraint,
                actionable_terminals.as_ref(),
                constraint.ignore_terminal,
                &tokenizer_scratch.matches,
                None,
            );
            let mut emitted = SmallVec::<[(usize, u32, TerminalsDisallowed); 4]>::new();

            for matched in normalized_matches {
                let new_offset = offset + matched.width;
                if new_offset > bytes.len() {
                    return None;
                }

                let (advanced_language, advanced_accumulator) = if matched.ignored {
                    (entry.language, entry.accumulator.clone())
                } else {
                    let started = profile.is_some().then(std::time::Instant::now);
                    let advanced = template_runtime.advance_language(
                        constraint,
                        matched.terminal_id,
                        entry.language,
                    )?;
                    if let (Some(profile), Some(started)) = (profile.as_deref_mut(), started) {
                        profile.evaluate_ns += started.elapsed().as_nanos() as u64;
                    }
                    if advanced == 0 {
                        continue;
                    }
                    (
                        advanced,
                        apply_future_terminal_disallow_to_accumulator(
                            constraint,
                            &tokenizer_scratch.states,
                            matched.terminal_id,
                            entry.accumulator.clone(),
                        ),
                    )
                };

                if emitted.iter().any(|(emitted_offset, language, accumulator)| {
                    *emitted_offset == new_offset
                        && *language == advanced_language
                        && accumulator == &advanced_accumulator
                }) {
                    continue;
                }
                emitted.push((
                    new_offset,
                    advanced_language,
                    advanced_accumulator.clone(),
                ));

                let destination = if new_offset == bytes.len() {
                    &mut queue_scratch.language_pending
                } else {
                    &mut queue_scratch.language_processing[new_offset]
                };
                if !merge_small_language_parser_state(
                    template_runtime,
                    destination,
                    initial_tokenizer_state,
                    advanced_language,
                    advanced_accumulator,
                ) {
                    return None;
                }
            }

            if !tokenizer_scratch.states.is_empty() {
                let mut fallback_gss = None;
                for &end_state in &tokenizer_scratch.states {
                    let check_started = profile.is_some().then(std::time::Instant::now);
                    let language_viable = language_end_state_may_advance(
                        constraint,
                        template_runtime,
                        entry.language,
                        end_state,
                    );
                    if template_runtime.is_exhausted() {
                        return None;
                    }
                    let viable = match language_viable {
                        Some(viable) => viable,
                        None => {
                            let reconstruct_started =
                                profile.is_some().then(std::time::Instant::now);
                            let gss = fallback_gss.get_or_insert_with(|| {
                                template_runtime.gss_from_language(
                                    entry.language,
                                    entry.accumulator.clone(),
                                )
                            });
                            if let (Some(profile), Some(started)) =
                                (profile.as_deref_mut(), reconstruct_started)
                            {
                                profile.continuation_reconstruct_ns +=
                                    started.elapsed().as_nanos() as u64;
                            }
                            end_state_may_advance(constraint, gss, end_state)
                        }
                    };
                    if let (Some(profile), Some(started)) =
                        (profile.as_deref_mut(), check_started)
                    {
                        profile.continuation_check_ns +=
                            started.elapsed().as_nanos() as u64;
                    }
                    if viable
                        && !merge_small_language_parser_state(
                            template_runtime,
                            &mut queue_scratch.language_pending,
                            end_state,
                            entry.language,
                            entry.accumulator.clone(),
                        )
                    {
                        return None;
                    }
                }
            }
        }
        offset += 1;
    }

    if template_runtime.is_exhausted() {
        return None;
    }
    if queue_scratch.language_pending.is_empty() {
        return Some(Err(
            "commit rejected: no valid parser states remain".to_string(),
        ));
    }
    Some(Ok(std::mem::take(
        &mut queue_scratch.language_pending,
    )))
}

fn commit_bytes_language_small_queue_fast_path(
    constraint: &Constraint,
    state: &mut ParserStateMap,
    bytes: &[u8],
    tokenizer_scratch: &mut tokenizer_scan::ReusableTokenizerExecScratch,
    queue_scratch: &mut SmallCommitQueueScratch,
    template_runtime: &mut TemplateAdvanceRuntime,
    profitability_prechecked: bool,
) -> Option<Result<(), String>> {
    if !language_small_queue_enabled() || !(2..=8).contains(&bytes.len()) || state.len() > 2 {
        return None;
    }

    let profile_enabled = std::env::var_os("GLRMASK_PROFILE_LANGUAGE_SMALL_QUEUE").is_some();
    let top_values =
        language_queue_top_value_count_at_most(state, LANGUAGE_QUEUE_MIN_TOP_VALUES);
    if top_values < LANGUAGE_QUEUE_MIN_TOP_VALUES {
        if profile_enabled {
            eprintln!(
                "[glrmask/profile][language_small_queue_decline] reason=narrow_top_frontier bytes={} top_values={} min_top_values={}",
                format_token_bytes(bytes),
                top_values,
                LANGUAGE_QUEUE_MIN_TOP_VALUES,
            );
        }
        return None;
    }

    if !profitability_prechecked
        && !has_multiple_actionable_terminal_boundaries(
            constraint,
            state,
            bytes,
            tokenizer_scratch,
        )
    {
        return None;
    }

    let parser_paths = language_queue_path_count_at_most(state, LANGUAGE_QUEUE_MIN_PATHS);
    if parser_paths < LANGUAGE_QUEUE_MIN_PATHS {
        if profile_enabled {
            eprintln!(
                "[glrmask/profile][language_small_queue_decline] reason=insufficient_stack_ambiguity bytes={} parser_paths={} min_paths={}",
                format_token_bytes(bytes),
                parser_paths,
                LANGUAGE_QUEUE_MIN_PATHS,
            );
        }
        return None;
    }
    let parser_nodes = language_queue_node_count_at_most(state, LANGUAGE_QUEUE_MIN_NODES);
    if parser_nodes < LANGUAGE_QUEUE_MIN_NODES {
        if profile_enabled {
            eprintln!(
                "[glrmask/profile][language_small_queue_decline] reason=insufficient_compact_gss_work bytes={} parser_paths_at_least={} parser_nodes={} min_nodes={}",
                format_token_bytes(bytes),
                LANGUAGE_QUEUE_MIN_PATHS,
                parser_nodes,
                LANGUAGE_QUEUE_MIN_NODES,
            );
        }
        return None;
    }
    if profile_enabled {
        eprintln!(
            "[glrmask/profile][language_small_queue_dispatch] selected bytes={} state_entries={} parser_paths_at_least={} parser_nodes_at_least={}",
            format_token_bytes(bytes),
            state.len(),
            LANGUAGE_QUEUE_MIN_PATHS,
            LANGUAGE_QUEUE_MIN_NODES,
        );
    }
    if !language_queue_input_is_bounded(state) {
        if profile_enabled {
            eprintln!(
                "[glrmask/profile][language_small_queue_decline] reason=input_depth bytes={}",
                format_token_bytes(bytes),
            );
        }
        return None;
    }

    let total_started = profile_enabled.then(std::time::Instant::now);
    let mut profile = LanguageCommitSimulationProfile::default();
    template_runtime.begin_commit();
    let simulation = simulate_language_commit(
        constraint,
        state,
        bytes,
        tokenizer_scratch,
        queue_scratch,
        template_runtime,
        profile_enabled.then_some(&mut profile),
    );
    let Some(simulation) = simulation else {
        if profile_enabled {
            let reason = if template_runtime.is_exhausted() {
                "work_budget"
            } else {
                "simulation_bound_or_missing_template"
            };
            eprintln!(
                "[glrmask/profile][language_small_queue_decline] reason={} bytes={}",
                reason,
                format_token_bytes(bytes),
            );
        }
        return None;
    };
    let pending = match simulation {
        Ok(pending) => pending,
        // The language queue is an accelerator, not the authority for
        // rejection. Fall through to the established commit path so an
        // optimization bug cannot introduce a false negative.
        Err(_) => return None,
    };

    if template_runtime.is_exhausted() {
        if profile_enabled {
            eprintln!(
                "[glrmask/profile][language_small_queue_decline] reason=work_budget_after_simulation bytes={}",
                format_token_bytes(bytes),
            );
        }
        return None;
    }
    let mut final_reconstruct_ns = 0u64;
    let mut new_state = ParserStateMap::default();
    for entry in pending {
        let started = profile_enabled.then(std::time::Instant::now);
        let gss = template_runtime.gss_from_language(entry.language, entry.accumulator);
        if let Some(started) = started {
            final_reconstruct_ns += started.elapsed().as_nanos() as u64;
        }
        new_state.merge_insert(entry.tokenizer_state, gss);
    }
    for parser_state in new_state.values_mut() {
        *parser_state = parser_state.fuse(Some(1));
    }
    new_state.retain(|_, parser_state| !parser_state.is_empty());
    if new_state.is_empty() {
        return None;
    }
    if profile_enabled {
        let (
            template_calls,
            template_memo_hits,
            template_memo_entries,
            template_products_started,
            semantic_nodes,
            semantic_lower_keys,
            semantic_upper_keys,
            semantic_union_entries,
        ) = template_runtime.work_summary();
        eprintln!(
            "[glrmask/profile][language_small_queue] bytes={} total_ns={} canonicalize_ns={} evaluate_ns={} continuation_reconstruct_ns={} continuation_check_ns={} final_reconstruct_ns={} template_calls={} template_memo_hits={} template_memo_entries={} template_products_started={} semantic_nodes={} semantic_lower_keys={} semantic_upper_keys={} semantic_union_entries={} final_summaries={:?}",
            format_token_bytes(bytes),
            total_started.expect("language queue profile start exists").elapsed().as_nanos(),
            profile.canonicalize_ns,
            profile.evaluate_ns,
            profile.continuation_reconstruct_ns,
            profile.continuation_check_ns,
            final_reconstruct_ns,
            template_calls,
            template_memo_hits,
            template_memo_entries,
            template_products_started,
            semantic_nodes,
            semantic_lower_keys,
            semantic_upper_keys,
            semantic_union_entries,
            new_state.values().map(ParserGSS::summary).collect::<Vec<_>>(),
        );
    }
    *state = new_state;
    Some(Ok(()))
}

fn try_advance_unique_actionable_top_fast(
    constraint: &Constraint,
    gss: &ParserGSS,
    terminal: u32,
) -> Option<ParserGSS> {
    if constraint.parser_has_controls() || template_advance_selected(constraint) {
        return None;
    }
    // If every nonempty path has this top, isolate(Some(top)) is precisely
    // the original immutable GSS. Avoid enumerating/copying the top set and
    // retaining an extra Arc. The caller must not retry a declined action:
    // repeating it on this same GSS cannot make the shortcut more applicable.
    if let Some(top) = gss.single_exclusive_top_value() {
        let action = constraint.table.action(top, terminal)?;
        return apply_single_top_action_fast(constraint, gss, top, terminal, action);
    }
    let mut selected = None;
    for top in gss.peek_values() {
        let Some(action) = constraint.table.action(top, terminal) else {
            continue;
        };
        if selected.is_some() {
            return None;
        }
        selected = Some((top, action));
    }
    let (top, action) = selected?;
    let isolated = gss.isolate(Some(top));
    (!isolated.is_empty())
        .then(|| apply_single_top_action_fast(constraint, &isolated, top, terminal, action))
        .flatten()
}

fn try_batch_same_width_disjoint_alias_actions(
    constraint: &Constraint,
    gss: &ParserGSS,
    matches: &[NormalizedMatch],
    width: usize,
    continuation_states: &[u32],
) -> Option<ParserGSS> {
    if constraint.parser_has_controls() || template_advance_selected(constraint) {
        return None;
    }
    let group = matches
        .iter()
        .filter(|matched| matched.width == width && !matched.ignored)
        .collect::<SmallVec<[&NormalizedMatch; 16]>>();
    if group.len() < 2 {
        return None;
    }
    for matched in &group {
        if continuation_states.iter().any(|&state| {
            constraint
                .tokenizer
                .possible_future_terminals(state)
                .contains(matched.terminal_id as usize)
        }) {
            return None;
        }
    }

    let mut terminal_by_top = SmallVec::<[(u32, u32); 8]>::new();
    for top in gss.peek_values() {
        let mut selected = None;
        for matched in &group {
            if constraint.table.action(top, matched.terminal_id).is_none() {
                continue;
            }
            if selected.is_some() {
                return None;
            }
            selected = Some(matched.terminal_id);
        }
        if let Some(terminal) = selected {
            terminal_by_top.push((top, terminal));
        }
    }
    if terminal_by_top.len() < 2 {
        return None;
    }
    advance_stacks_disjoint_top_terminals_bounded(
        &constraint.table,
        gss,
        &terminal_by_top,
    )
}

fn try_batch_same_width_pure_matches(
    constraint: &Constraint,
    gss: &ParserGSS,
    matches: &[NormalizedMatch],
    width: usize,
    continuation_states: &[u32],
) -> Option<ParserGSS> {
    if constraint.has_template_parser() { return None; }

    if constraint.parser_has_controls() {
        return None;
    }
    let group = matches
        .iter()
        .filter(|matched| matched.width == width && !matched.ignored)
        .collect::<SmallVec<[&NormalizedMatch; 16]>>();
    if group.len() < 2 {
        return None;
    }

    // The historical per-terminal path adds delayed longest-match exclusions
    // only when that same logical terminal remains possible after the model
    // token. Batch only when that transform is identity for every member.
    for matched in &group {
        if continuation_states.iter().any(|&state| {
            constraint
                .tokenizer
                .possible_future_terminals(state)
                .contains(matched.terminal_id as usize)
        }) {
            return None;
        }
    }

    let tops = gss.peek_values();
    let mut shifts = SmallVec::<[(u32, u32, bool); 32]>::new();
    for &top in &tops {
        for matched in &group {
            let Some(action) = constraint.table.action(top, matched.terminal_id) else {
                continue;
            };
            match action {
                Action::Shift(target, replace) => {
                    let edge = (top, *target, *replace);
                    if !shifts.contains(&edge) {
                        shifts.push(edge);
                    }
                }
                Action::ReplaceShifts(targets) => {
                    for &target in targets.iter() {
                        let edge = (top, target, true);
                        if !shifts.contains(&edge) {
                            shifts.push(edge);
                        }
                    }
                }
                Action::StackShifts(stack_shifts) => {
                    for shift in stack_shifts {
                        if shift.pushes.len() != 1 || shift.pop > 1 {
                            return None;
                        }
                        let edge = (top, shift.pushes[0], shift.pop == 1);
                        if !shifts.contains(&edge) {
                            shifts.push(edge);
                        }
                    }
                }
                _ => return None,
            }
        }
    }
    if shifts.is_empty() {
        return None;
    }

    let advanced = if tops.len() == 1
        && shifts.iter().all(|(top, _, _)| *top == tops[0])
        && shifts
            .first()
            .is_some_and(|(_, _, replace)| shifts.iter().all(|(_, _, other)| other == replace))
        && shifts
            .iter()
            .enumerate()
            .all(|(index, (_, target, _))| {
                !shifts[..index]
                    .iter()
                    .any(|(_, prior_target, _)| prior_target == target)
            })
        && let Some(stack) = gss.try_virtual_stack()
    {
        let replace = shifts[0].2;
        stack
            .into_gss_after_popping_and_pushing_unique_single_branches(
                usize::from(replace),
                shifts.iter().map(|(_, target, _)| target),
            )
            .unwrap_or_else(|| gss.apply_top_pure_shifts(shifts.clone()))
    } else {
        gss.apply_top_pure_shifts(shifts)
    };
    (!advanced.is_empty()).then_some(advanced)
}

fn commit_bytes_small_queue_fast_path(
    constraint: &Constraint,
    state: &mut ParserStateMap,
    bytes: &[u8],
    tokenizer_scratch: &mut tokenizer_scan::ReusableTokenizerExecScratch,
    queue_scratch: &mut SmallCommitQueueScratch,
    admission_cache: &mut SmallVec<[ParserAdmissionCacheEntry; 8]>,
    prune_tokenizer_scratch: &mut tokenizer_scan::ReusableTokenizerExecScratch,
) -> Option<Result<(), String>> {
    let has_linker_controls = constraint.parser_has_controls()
        || constraint.uses_compact_segmented_parser_runtime();
    if bytes.len() > 16 || state.len() > 8 {
        return None;
    }
    queue_scratch.clear();
    for (&tokenizer_state, gss) in state.iter() {
        merge_small_parser_state(
            &mut queue_scratch.processing[0],
            tokenizer_state,
            gss.clone(),
        );
    }

    let initial_tokenizer_state = constraint.runtime_commit_initial_state();
    let mut offset = 0usize;
    while offset <= bytes.len() {
        if queue_scratch.processing[offset].is_empty() {
            offset += 1;
            continue;
        }

        let states_to_process = std::mem::take(&mut queue_scratch.processing[offset]);
        let mut groups = SmallVec::<[(ParserGSS, SmallVec<[u32; 8]>); 8]>::new();
        for (tokenizer_state, gss) in states_to_process {
            let can_group = gss.all_accs_satisfy(|acc: &TerminalsDisallowed| acc.is_empty());
            let mut grouped = false;
            if can_group {
                'groups: for (existing_gss, tokenizer_states) in &mut groups {
                    if !existing_gss.ptr_eq(&gss) {
                        continue;
                    }
                    let future = constraint.tokenizer.possible_future_terminals(tokenizer_state);
                    for &other_state in tokenizer_states.iter() {
                        if !future.is_disjoint(
                            constraint.tokenizer.possible_future_terminals(other_state),
                        ) {
                            continue 'groups;
                        }
                    }
                    tokenizer_states.push(tokenizer_state);
                    grouped = true;
                    break;
                }
            }
            if !grouped {
                groups.push((gss, smallvec::smallvec![tokenizer_state]));
            }
        }

        for (mut gss_at_offset, tokenizer_states) in groups {
            let tokenizer_state = tokenizer_states[0];
            let reused_prune_scan = offset == 0
                && !queue_scratch.prune_union_starts.is_empty()
                && queue_scratch.prune_union_starts.as_slice() == tokenizer_states.as_slice();
            if reused_prune_scan {
                tokenizer_scratch.states.clear();
                tokenizer_scratch
                    .states
                    .extend(prune_tokenizer_scratch.states.iter().copied());
                tokenizer_scratch.matches.clear();
                tokenizer_scratch
                    .matches
                    .extend(prune_tokenizer_scratch.matches.iter().cloned());
            } else if !execute_tokenizer_reusable_from_states(
                constraint,
                &bytes[offset..],
                &tokenizer_states,
                tokenizer_scratch,
            ) {
                return None;
            }

            if offset == 0
                && !gss_at_offset.all_accs_satisfy(|td: &TerminalsDisallowed| td.is_empty())
            {
                gss_at_offset = try_prune_single_initial_state_batched_accumulators(
                    constraint,
                    &gss_at_offset,
                    bytes,
                    prune_tokenizer_scratch,
                    &mut queue_scratch.prune_union_starts,
                )
                .unwrap_or_else(|| {
                    prune_single_initial_state_for_parts(
                        constraint,
                        gss_at_offset,
                        tokenizer_state,
                        &tokenizer_scratch.states,
                        &tokenizer_scratch.matches,
                        bytes,
                    )
                });
                if gss_at_offset.is_empty() {
                    continue;
                }
            }

            let (actionable_terminals, normalized_matches) =
                if tokenizer_scratch.matches.is_empty() {
                    (None, SmallVec::<[NormalizedMatch; 16]>::new())
                } else {
                    let actionable_terminals = (!has_linker_controls)
                        .then(|| ActionableTerminals::from_gss(constraint, &gss_at_offset))
                        .flatten();
                    let normalized_matches = collect_unique_actionable_reusable_matches(
                        constraint,
                        actionable_terminals.as_ref(),
                        constraint.ignore_terminal,
                        &tokenizer_scratch.matches,
                    );
                    (actionable_terminals, normalized_matches)
                };
            let mut emitted_terminal_outputs = SmallVec::<[(usize, ParserGSS); 4]>::new();
            let mut batched_widths = SmallVec::<[usize; 4]>::new();
            for matched in &normalized_matches {
                if matched.ignored || batched_widths.contains(&matched.width) {
                    continue;
                }
                let batched = try_batch_same_width_pure_matches(
                    constraint,
                    &gss_at_offset,
                    &normalized_matches,
                    matched.width,
                    &tokenizer_scratch.states,
                )
                .or_else(|| {
                    try_batch_same_width_disjoint_alias_actions(
                        constraint,
                        &gss_at_offset,
                        &normalized_matches,
                        matched.width,
                        &tokenizer_scratch.states,
                    )
                });
                if let Some(advanced) = batched {
                    let new_offset = offset + matched.width;
                    if new_offset > bytes.len() {
                        return None;
                    }
                    if emitted_terminal_outputs.iter().any(|(emitted_offset, emitted_gss)| {
                        *emitted_offset == new_offset && emitted_gss == &advanced
                    }) {
                        batched_widths.push(matched.width);
                        continue;
                    }
                    emitted_terminal_outputs.push((new_offset, advanced.clone()));
                    if new_offset == bytes.len() {
                        merge_small_parser_state(
                            &mut queue_scratch.pending,
                            initial_tokenizer_state,
                            advanced,
                        );
                    } else {
                        merge_small_parser_state(
                            &mut queue_scratch.processing[new_offset],
                            initial_tokenizer_state,
                            advanced,
                        );
                    }
                    batched_widths.push(matched.width);
                }
            }

            for matched in normalized_matches {
                let new_offset = offset + matched.width;
                if !matched.ignored && batched_widths.contains(&matched.width) {
                    continue;
                }
                if new_offset > bytes.len() {
                    return None;
                }

                if matched.ignored {
                    if new_offset == bytes.len() {
                        merge_small_parser_state(
                            &mut queue_scratch.pending,
                            initial_tokenizer_state,
                            gss_at_offset.clone(),
                        );
                    } else {
                        merge_small_parser_state(
                            &mut queue_scratch.processing[new_offset],
                            initial_tokenizer_state,
                            gss_at_offset.clone(),
                        );
                    }
                    continue;
                }

                let advanced = if !has_linker_controls
                    && !template_advance_selected(constraint)
                    && let Some(advanced) = try_advance_unique_actionable_top_fast(
                        constraint,
                        &gss_at_offset,
                        matched.terminal_id,
                    )
                {
                    advanced
                } else {
                    let Some(advanced) = advance_parser_stacks_if_possible(
                        constraint,
                        &gss_at_offset,
                        matched.terminal_id,
                    ) else {
                        continue;
                    };
                    advanced
                };
                let advanced = apply_future_terminal_disallow_for_states(
                    constraint,
                    &tokenizer_scratch.states,
                    matched.terminal_id,
                    advanced,
                );
                if advanced.is_empty() {
                    continue;
                }
                if emitted_terminal_outputs.iter().any(|(emitted_offset, emitted_gss)| {
                    *emitted_offset == new_offset && emitted_gss == &advanced
                }) {
                    continue;
                }
                emitted_terminal_outputs.push((new_offset, advanced.clone()));
                if new_offset == bytes.len() {
                    merge_small_parser_state(
                        &mut queue_scratch.pending,
                        initial_tokenizer_state,
                        advanced,
                    );
                } else {
                    merge_small_parser_state(
                        &mut queue_scratch.processing[new_offset],
                        initial_tokenizer_state,
                        advanced,
                    );
                }
            }

            let local_row_admission = try_local_row_presence_admission_words(
                constraint,
                &gss_at_offset,
                &tokenizer_scratch.states,
            );
            let admission_cache_index = if local_row_admission.is_none() {
                cached_batched_end_state_admission(
                    constraint,
                    &gss_at_offset,
                    &tokenizer_scratch.states,
                    admission_cache,
                )
            } else {
                None
            };
            for &end_state in &tokenizer_scratch.states {
                let may_advance = if let Some(words) = local_row_admission.as_ref() {
                    end_state_may_advance_from_row_words(constraint, end_state, words)
                } else if let Some(index) = admission_cache_index {
                    end_state_may_advance_from_cache_entry(
                        constraint,
                        end_state,
                        &admission_cache[index],
                    )
                } else {
                    cached_single_end_state_may_advance(
                        constraint,
                        &gss_at_offset,
                        end_state,
                        admission_cache,
                    )
                };
                if may_advance {
                    merge_small_parser_state(
                        &mut queue_scratch.pending,
                        end_state,
                        gss_at_offset.clone(),
                    );
                }
            }
        }
        offset += 1;
    }

    let mut new_state = ParserStateMap::default();
    let mut fused_by_source = SmallVec::<[(ParserGSS, ParserGSS); 8]>::new();
    for (tokenizer_state, parser_state) in queue_scratch.pending.drain(..) {
        let mut fused = fused_by_source
            .iter()
            .find(|(source, _)| source.ptr_eq(&parser_state))
            .map(|(_, fused)| fused.clone())
            .unwrap_or_else(|| {
                let source = parser_state.clone();
                let fused = parser_state.fuse(Some(1));
                fused_by_source.push((source, fused.clone()));
                fused
            });
        if let Some((_, canonical)) = fused_by_source
            .iter()
            .find(|(_, candidate)| *candidate == fused)
        {
            fused = canonical.clone();
        }
        if !fused.is_empty() {
            if let Some(_uniform) = fused.uniform_accumulator() {
                new_state.insert_flat_alternative(tokenizer_state, fused);
            } else {
                let mut accumulators = SmallVec::<[TerminalsDisallowed; 4]>::new();
                let mut overflow = false;
                fused.for_each_acc(|acc| {
                    if overflow || accumulators.contains(acc) {
                        return;
                    }
                    if accumulators.len() == accumulators.capacity() {
                        overflow = true;
                        return;
                    }
                    accumulators.push(acc.clone());
                });
                if overflow || accumulators.len() <= 1 {
                    new_state.insert_flat_alternative(tokenizer_state, fused);
                } else {
                    for acc in accumulators {
                        let part = fused.apply_and_prune_no_promote(|candidate| {
                            (candidate == &acc).then_some(candidate.clone())
                        });
                        if !part.is_empty() {
                            new_state.insert_flat_alternative(tokenizer_state, part);
                        }
                    }
                }
            }
        }
    }
    if new_state.is_empty() {
        return Some(Err("commit rejected: no valid parser states remain".to_string()));
    }
    *state = new_state;
    Some(Ok(()))
}

enum LinearFastPathResult {
    Complete(Result<ParserGSS, String>),
    Continue { gss: ParserGSS, offset: usize },
    Restart,
}

struct DirectLinearStep {
    width: usize,
    terminal: u32,
    ignored: bool,
    end_state: Option<u32>,
}

fn choose_direct_linear_step(
    constraint: &Constraint,
    gss: &ParserGSS,
    bytes: &[u8],
    start_state: u32,
    carried_top_state: Option<u32>,
) -> Option<DirectLinearStep> {
    let ignore_terminal = constraint.ignore_terminal;
    let mut tokenizer_state = start_state;
    let mut chosen: Option<(usize, u32, bool)> = None;
    let mut consumed_all = true;
    let mut actionable_terminals = carried_top_state.map(ActionableTerminals::SingleState);

    for (index, &byte) in bytes.iter().enumerate() {
        let next_state = constraint.tokenizer_fast_transitions.transition(
            &constraint.tokenizer,
            tokenizer_state,
            byte,
        );
        if next_state == u32::MAX {
            consumed_all = false;
            break;
        };
        tokenizer_state = next_state;
        let width = index + 1;
        let mut chosen_at_width = false;

        if actionable_terminals.is_none() {
            actionable_terminals = ActionableTerminals::from_gss(constraint, gss);
        }
        let mut conflict = false;
        for_each_relevant_matched_terminal(
            constraint,
            tokenizer_state,
            actionable_terminals.as_ref(),
            |terminal, ignored| {
                let candidate = (width, terminal, ignored);
                chosen_at_width = true;
                if let Some((_, existing_terminal, _)) = chosen {
                    if existing_terminal == terminal {
                        chosen = Some(candidate);
                    } else {
                        conflict = true;
                    }
                } else {
                    chosen = Some(candidate);
                }
            },
        );
        if conflict {
            return None;
        }

        if chosen_at_width && chosen.is_some_and(|(_, _, ignored)| ignored) {
            return Some(DirectLinearStep {
                width,
                terminal: chosen.unwrap().1,
                ignored: true,
                end_state: None,
            });
        }

        if chosen_at_width
            && chosen.is_some_and(|(_, _, ignored)| !ignored)
            && index + 1 < bytes.len()
        {
            let next_byte = bytes[index + 1];
            let next_state = constraint.tokenizer_fast_transitions.transition(
                &constraint.tokenizer,
                tokenizer_state,
                next_byte,
            );
            if next_state == u32::MAX {
                let (_, terminal, _) = chosen.unwrap();
                return Some(DirectLinearStep {
                    width,
                    terminal,
                    ignored: false,
                    end_state: None,
                });
            }
        }
    }

    let (width, terminal, ignored) = chosen?;
    let end_state = consumed_all.then_some(tokenizer_state);

    Some(DirectLinearStep {
        width,
        terminal,
        ignored,
        end_state,
    })
}

fn commit_bytes_direct_linear_fast_path(
    constraint: &Constraint,
    start_gss: ParserGSS,
    bytes: &[u8],
    start_tokenizer_state: u32,
    mut profile: Option<&mut CommitProfile>,
) -> Option<LinearFastPathResult> {
    if constraint.has_template_parser() { return None; }

    let mut gss = start_gss;
    let mut carried_stack = gss.try_virtual_stack();
    let mut offset = 0usize;
    let mut tokenizer_state = start_tokenizer_state;

    while offset < bytes.len() {
        let choose_start = profile.as_ref().map(|_| std::time::Instant::now());
        let carried_top_state = carried_stack.as_ref().and_then(|stack| stack.top().copied());
        let Some(step) = choose_direct_linear_step(
            constraint,
            &gss,
            &bytes[offset..],
            tokenizer_state,
            carried_top_state,
        ) else {
            if let Some(stack) = carried_stack.take() {
                let materialize_start = profile.as_ref().map(|_| std::time::Instant::now());
                gss = stack.into_gss();
                if let (Some(profile), Some(start)) = (profile.as_deref_mut(), materialize_start) {
                    profile.linear_fast_path_materialize_ns += start.elapsed().as_nanos() as u64;
                }
            }
            if offset > 0 && profile.is_none() {
                return Some(LinearFastPathResult::Continue { gss, offset });
            }
            return None;
        };
        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), choose_start) {
            profile.linear_fast_path_match_scan_ns += start.elapsed().as_nanos() as u64;
            profile.linear_fast_path_steps += 1;
        }

        let keep_carried = if let Some(end_state) = step.end_state
            && let Some(stack) = carried_stack.as_ref()
        {
            let carried_gate_start = profile.as_ref().map(|_| std::time::Instant::now());
            let keep_carried = stack.top().copied().is_some_and(|top_state| {
                end_state != constraint.runtime_commit_initial_state()
                    && !constraint.parser_advance_row_intersects(
                        top_state,
                        constraint.tokenizer.possible_future_terminals(end_state),
                    )
                    && !constraint
                        .tokenizer
                        .possible_future_terminals(end_state)
                        .contains(step.terminal as usize)
            });
            if let (Some(profile), Some(start)) = (profile.as_deref_mut(), carried_gate_start) {
                let elapsed = start.elapsed().as_nanos() as u64;
                profile.linear_fast_path_carried_gate_ns += elapsed;
                profile.linear_fast_path_end_state_check_ns += elapsed;
            }
            keep_carried
        } else {
            false
        };
        if step.end_state.is_some() {
            if !keep_carried {
                if let Some(stack) = carried_stack.take() {
                    let materialize_start = profile.as_ref().map(|_| std::time::Instant::now());
                    gss = stack.into_gss();
                    if let (Some(profile), Some(start)) = (profile.as_deref_mut(), materialize_start) {
                        profile.linear_fast_path_materialize_ns += start.elapsed().as_nanos() as u64;
                    }
                }
            }
        }

        if let Some(end_state) = step.end_state {
            let carried_gate_start = profile.as_ref().map(|_| std::time::Instant::now());
            let should_restart = end_state_may_advance(constraint, &gss, end_state);
            if let (Some(profile), Some(start)) = (profile.as_deref_mut(), carried_gate_start) {
                let elapsed = start.elapsed().as_nanos() as u64;
                profile.linear_fast_path_carried_gate_ns += elapsed;
                profile.linear_fast_path_end_state_check_ns += elapsed;
            }
            if should_restart {
                if let Some(stack) = carried_stack.take() {
                    let materialize_start = profile.as_ref().map(|_| std::time::Instant::now());
                    gss = stack.into_gss();
                    if let (Some(profile), Some(start)) = (profile.as_deref_mut(), materialize_start) {
                        profile.linear_fast_path_materialize_ns += start.elapsed().as_nanos() as u64;
                    }
                }
                if offset > 0 && profile.is_none() {
                    return Some(LinearFastPathResult::Continue { gss, offset });
                }
                return None;
            }
        }

        if !step.ignored {
            if offset == 0 {
                if !gss.all_accs_satisfy(|td: &TerminalsDisallowed| td.is_empty()) {
                    // Delayed exclusions are keyed by continuation tokenizer states,
                    // not necessarily by the current tokenizer state. The
                    // general/flat paths execute every continuation state over
                    // the whole model token; this single-step shortcut lacks
                    // that information, so decline rather than applying the
                    // old current-state-only pruning rule.
                    return None;
                }
            }

            let mut shifted_carried_stack = false;
            let mut carried_apply_elapsed_ns = 0u64;
            let action_lookup_start = profile.as_ref().map(|_| std::time::Instant::now());
            let carried_action = if let Some(stack) = carried_stack.as_ref()
                && let Some(top_state) = stack.top().copied()
                && step.end_state.is_none_or(|end_state| {
                    end_state != constraint.runtime_commit_initial_state()
                        && !constraint.parser_advance_row_intersects(
                            top_state,
                            constraint.tokenizer.possible_future_terminals(end_state),
                        )
                        && !constraint
                            .tokenizer
                            .possible_future_terminals(end_state)
                            .contains(step.terminal as usize)
                })
            {
                constraint.table.action(top_state, step.terminal)
            } else {
                None
            };
            if let (Some(profile), Some(start)) = (profile.as_deref_mut(), action_lookup_start) {
                profile.linear_fast_path_action_lookup_ns += start.elapsed().as_nanos() as u64;
            }
            if let Some(action) = carried_action {
                let apply_action_start = profile.as_ref().map(|_| std::time::Instant::now());
                if !template_advance_selected(constraint)
                    && let Some(stack) = carried_stack.as_mut()
                {
                    shifted_carried_stack =
                        try_apply_action_to_carried_virtual_stack(stack, action);
                }
                if let (Some(profile), Some(start)) = (profile.as_deref_mut(), apply_action_start) {
                    carried_apply_elapsed_ns = start.elapsed().as_nanos() as u64;
                    profile.linear_fast_path_apply_action_wall_ns += carried_apply_elapsed_ns;
                }
            }
            if shifted_carried_stack {
                if let Some(profile) = profile.as_deref_mut() {
                    let bookkeeping_start = std::time::Instant::now();
                    let advance_profile = fast_action_advance_profile(
                        &gss,
                        carried_action.unwrap(),
                        carried_apply_elapsed_ns,
                    );
                    profile.advance_core_ns += advance_profile.total_ns;
                    profile.advance_ns += carried_apply_elapsed_ns;
                    profile.linear_fast_path_advance_ns += carried_apply_elapsed_ns;
                    profile.n_advances += 1;
                    apply_advance_profile(profile, &advance_profile);
                    profile.linear_fast_path_profile_bookkeeping_ns +=
                        bookkeeping_start.elapsed().as_nanos() as u64;
                }

                offset += step.width;
                tokenizer_state = constraint.runtime_commit_initial_state();
                continue;
            }

            if let Some(stack) = carried_stack.take() {
                let materialize_start = profile.as_ref().map(|_| std::time::Instant::now());
                gss = stack.into_gss();
                if let (Some(profile), Some(start)) = (profile.as_deref_mut(), materialize_start) {
                    profile.linear_fast_path_materialize_ns += start.elapsed().as_nanos() as u64;
                }
            }
            let advance_start = profile.as_ref().map(|_| std::time::Instant::now());
            let advanced = if !template_advance_selected(constraint)
                && let Some(top_state) = gss.single_exclusive_top_value()
                && let Some(action) = constraint.table.action(top_state, step.terminal)
                && let Some(advanced) =
                    apply_single_top_action_fast(
                        constraint,
                        &gss,
                        top_state,
                        step.terminal,
                        action,
                    )
            {
                advanced
            } else {
                if let Some(profile) = profile.as_deref_mut() {
                    let (advanced, advance_profile) =
                        advance_parser_stacks_profiled(constraint, &gss, step.terminal);
                    if advanced.is_empty() {
                        return None;
                    }
                    let bookkeeping_start = std::time::Instant::now();
                    profile.advance_core_ns += advance_profile.total_ns;
                    apply_advance_profile(profile, &advance_profile);
                    profile.linear_fast_path_profile_bookkeeping_ns +=
                        bookkeeping_start.elapsed().as_nanos() as u64;
                    advanced
                } else {
                    let advanced = advance_parser_stacks(constraint, &gss, step.terminal);
                    if advanced.is_empty() {
                        return None;
                    }
                    advanced
                }
            };
            if let (Some(profile), Some(start)) = (profile.as_deref_mut(), advance_start) {
                let elapsed = start.elapsed().as_nanos() as u64;
                profile.linear_fast_path_apply_action_wall_ns += elapsed;
                profile.advance_ns += elapsed;
                profile.linear_fast_path_advance_ns += elapsed;
                profile.n_advances += 1;
            }
            if advanced.is_empty() {
                return Some(LinearFastPathResult::Complete(Err(
                    "commit rejected: no valid parser states remain".to_string(),
                )));
            }
            let exec_result = TokenizerExecResult {
                end_state: step.end_state.into_iter().collect(),
                matches: Vec::new(),
            };
            let future_start = profile.as_ref().map(|_| std::time::Instant::now());
            gss = apply_future_terminal_disallow(
                constraint,
                &exec_result,
                step.terminal,
                advanced,
            );
            if let (Some(profile), Some(start)) = (profile.as_deref_mut(), future_start) {
                let elapsed = start.elapsed().as_nanos() as u64;
                profile.advance_future_disallow_ns += elapsed;
                profile.linear_fast_path_future_disallow_ns += elapsed;
                profile.linear_fast_path_advance_ns += elapsed;
            }
            if gss.is_empty() {
                return Some(LinearFastPathResult::Complete(Err(
                    "commit rejected: no valid parser states remain".to_string(),
                )));
            }
        }

        offset += step.width;
        tokenizer_state = constraint.runtime_commit_initial_state();
    }

    if let Some(stack) = carried_stack.take() {
        let materialize_start = profile.as_ref().map(|_| std::time::Instant::now());
        gss = stack.into_gss();
        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), materialize_start) {
            profile.linear_fast_path_materialize_ns += start.elapsed().as_nanos() as u64;
        }
    }
    let fuse_start = profile.as_ref().map(|_| std::time::Instant::now());
    let fused = if constraint.direct_regular_wide_frontier_for_gss(&gss).is_some() {
        gss
    } else {
        gss.fuse(Some(1))
    };
    if let (Some(profile), Some(start)) = (profile.as_deref_mut(), fuse_start) {
        let elapsed = start.elapsed().as_nanos() as u64;
        profile.linear_fast_path_fuse_ns += elapsed;
        profile.fuse_ns += elapsed;
    }
    if fused.is_empty() {
        return Some(LinearFastPathResult::Complete(Err(
            "commit rejected: no valid parser states remain".to_string(),
        )));
    }
    Some(LinearFastPathResult::Complete(Ok(fused)))
}

fn record_per_advance_entry(
    advances: &mut Vec<PerAdvanceEntry>,
    tokenizer_state: u32,
    terminal_id: u32,
    before_gss: &ParserGSS,
    after_gss: &ParserGSS,
    match_start: usize,
    match_end: usize,
    token_bound: usize,
    match_bytes: &[u8],
    profile: AdvanceProfile,
) -> u64 {
    use std::time::Instant;

    let summary_start = Instant::now();
    let gss_stacks_before = parser_stacks_only(before_gss);
    let gss_stacks_after = parser_stacks_only(after_gss);
    let gss_summary_before = before_gss.summary();
    let gss_summary_after = after_gss.summary();
    let match_bytes = match_bytes.to_vec();
    let summary_ns = summary_start.elapsed().as_nanos() as u64;
    advances.push(PerAdvanceEntry {
        terminal_id,
        tokenizer_state,
        gss_stacks_before,
        gss_stacks_after,
        gss_summary_before,
        gss_summary_after,
        match_start,
        match_end,
        token_bound,
        match_bytes,
        profile,
        summary_ns,
    });
    summary_ns
}

fn commit_bytes_fast_path_profiled(
    constraint: &Constraint,
    state: &mut ParserStateMap,
    bytes: &[u8],
    tokenizer_state: u32,
    exec_result: &TokenizerExecResult,
    advances: Option<&mut Vec<PerAdvanceEntry>>,
    profile: &mut CommitProfile,
) -> Option<Result<(), String>> {
    use std::time::Instant;

    let total_start = Instant::now();
    let gss = state.values().next().unwrap();
    let ignore_terminal = constraint.ignore_terminal;
    if constraint.tokenizer_has_epsilon_transitions
        && !gss.all_accs_satisfy(|td: &TerminalsDisallowed| td.is_empty())
    {
        profile.failed_fast_path_probe_ns += total_start.elapsed().as_nanos() as u64;
        return None;
    }

    let scan_start = Instant::now();
    let mut sole_terminal: Option<u32> = None;
    for matched in &exec_result.matches {
        if matched.width != bytes.len() {
            profile.failed_fast_path_probe_ns += total_start.elapsed().as_nanos() as u64;
            return None;
        }
        if is_ignored_terminal(ignore_terminal, matched.id) {
            profile.failed_fast_path_probe_ns += total_start.elapsed().as_nanos() as u64;
            return None;
        }
        if !parser_may_advance_on(constraint, gss, matched.id) {
            continue;
        }
        if sole_terminal.is_some() {
            profile.failed_fast_path_probe_ns += total_start.elapsed().as_nanos() as u64;
            return None;
        }
        sole_terminal = Some(matched.id);
    }
    profile.fast_path_match_scan_ns = scan_start.elapsed().as_nanos() as u64;
    let Some(terminal) = sole_terminal else {
        profile.failed_fast_path_probe_ns += total_start.elapsed().as_nanos() as u64;
        return None;
    };

    let no_end_state = exec_result.end_state.is_empty();
    let all_accs_empty = no_end_state
        && gss.all_accs_satisfy(|td: &TerminalsDisallowed| td.is_empty());

    if all_accs_empty && !template_advance_selected(constraint) {
        if let Some(top_state) = gss.single_exclusive_top_value() {
            if let Some(Action::Shift(target, is_replace)) = constraint.table.action(top_state, terminal) {
                let advance_start = Instant::now();
                let shifted = if *is_replace {
                    gss.popn(1).push(*target)
                } else {
                    gss.push(*target)
                };
                profile.fast_path_advance_ns = advance_start.elapsed().as_nanos() as u64;
                profile.advance_core_ns = profile.fast_path_advance_ns;
                profile.advance_ns = profile.fast_path_advance_ns;
                profile.n_advances = 1;

                if let Some(advances) = advances {
                    profile.adv_summary_ns += record_per_advance_entry(
                        advances,
                        tokenizer_state,
                        terminal,
                        gss,
                        &shifted,
                        0,
                        bytes.len(),
                        bytes.len(),
                        bytes,
                        AdvanceProfile {
                            pure_shift: true,
                            fast_path_ns: profile.fast_path_advance_ns,
                            stack_shift_apply_ns: profile.fast_path_advance_ns,
                            total_ns: profile.fast_path_advance_ns,
                            top_states: gss.peek_values().len() as u32,
                            gss_depth: gss.max_depth(),
                            vstack_len: gss.try_virtual_stack().map_or(0, |vstack| vstack.len() as u32),
                            ..AdvanceProfile::default()
                        },
                    );
                }

                let update_start = Instant::now();
                state.clear();
                state.insert(constraint.runtime_commit_initial_state(), shifted);
                profile.fast_path_state_update_ns = update_start.elapsed().as_nanos() as u64;
                profile.fast_path_total_ns = total_start.elapsed().as_nanos() as u64;
                profile.total_ns = profile.fast_path_total_ns;
                profile.fast_path_tokenizer_exec_ns = profile.exec_ns;
                return Some(Ok(()));
            }
            if !template_advance_selected(constraint)
                && let Some(Action::StackShifts(shifts)) = constraint.table.action(top_state, terminal)
            {
                let advance_start = Instant::now();
                let (shifted, advance_profile) =
                    advance_parser_stacks_profiled(constraint, gss, terminal);
                profile.fast_path_advance_ns = advance_start.elapsed().as_nanos() as u64;
                profile.advance_core_ns = profile.fast_path_advance_ns;
                profile.advance_ns = profile.fast_path_advance_ns;
                profile.n_advances = 1;
                apply_advance_profile(profile, &advance_profile);
                if let Some(advances) = advances {
                    profile.adv_summary_ns += record_per_advance_entry(
                        advances,
                        tokenizer_state,
                        terminal,
                        gss,
                        &shifted,
                        0,
                        bytes.len(),
                        bytes.len(),
                        bytes,
                        advance_profile,
                    );
                }

                let update_start = Instant::now();
                state.clear();
                state.insert(constraint.runtime_commit_initial_state(), shifted);
                profile.fast_path_state_update_ns = update_start.elapsed().as_nanos() as u64;
                profile.fast_path_total_ns = total_start.elapsed().as_nanos() as u64;
                profile.total_ns = profile.fast_path_total_ns;
                profile.fast_path_tokenizer_exec_ns = profile.exec_ns;
                return Some(Ok(()));
            }
        }
    }

    let (_, gss_owned) = state.pop_first().unwrap();

    let prune_start = Instant::now();
    let pruned_gss = if all_accs_empty {
        gss_owned
    } else {
        let pruned = prune_single_initial_state_for_exec(
            constraint,
            gss_owned,
            tokenizer_state,
            exec_result,
            bytes,
        );

        if pruned.is_empty() {
            return Some(Err("commit rejected: no valid parser states remain".to_string()));
        }
        pruned
    };
    profile.fast_path_prune_ns = prune_start.elapsed().as_nanos() as u64;
    profile.prune_ns = profile.fast_path_prune_ns;

    let end_state_check_start = Instant::now();
    let end_states_to_keep: TokenizerStateSet = exec_result
        .end_state
        .iter()
        .copied()
        .filter(|&end_state| end_state_may_advance(constraint, &pruned_gss, end_state))
        .collect();
    profile.fast_path_end_state_check_ns = end_state_check_start.elapsed().as_nanos() as u64;

    let advance_start = Instant::now();
    let advanced = advance_parser_stacks_owned(constraint, pruned_gss.clone(), terminal);
    profile.fast_path_advance_ns = advance_start.elapsed().as_nanos() as u64;
    profile.advance_core_ns = profile.fast_path_advance_ns;
    profile.n_advances = 1;

    if let Some(advances) = advances {
        let (after_for_entry, advance_profile) =
            advance_parser_stacks_profiled(constraint, &pruned_gss, terminal);
        profile.adv_summary_ns += record_per_advance_entry(
            advances,
            tokenizer_state,
            terminal,
            &pruned_gss,
            &after_for_entry,
            0,
            bytes.len(),
            bytes.len(),
            bytes,
            advance_profile.clone(),
        );
        apply_advance_profile(profile, &advance_profile);
    }

    let mut produced_state = false;
    if !advanced.is_empty() {
        let future_start = Instant::now();
        let advanced = apply_future_terminal_disallow(constraint, exec_result, terminal, advanced);
        profile.fast_path_future_disallow_ns = future_start.elapsed().as_nanos() as u64;
        profile.advance_future_disallow_ns = profile.fast_path_future_disallow_ns;
        profile.advance_ns = profile.fast_path_advance_ns + profile.fast_path_future_disallow_ns;

        if !advanced.is_empty() {
            let fuse_start = Instant::now();
            let fused = advanced.fuse(Some(1));
            profile.fast_path_fuse_ns = fuse_start.elapsed().as_nanos() as u64;
            profile.fuse_ns = profile.fast_path_fuse_ns;

            if !fused.is_empty() {
                let update_start = Instant::now();
                state.insert(constraint.runtime_commit_initial_state(), fused);
                profile.fast_path_state_update_ns += update_start.elapsed().as_nanos() as u64;
                produced_state = true;
            }
        }
    } else {
        profile.advance_ns = profile.fast_path_advance_ns;
    }

    if !end_states_to_keep.is_empty() {
        let fuse_start = Instant::now();
        let fused = pruned_gss.fuse(Some(1));
        let fuse_elapsed = fuse_start.elapsed().as_nanos() as u64;
        profile.fast_path_fuse_ns += fuse_elapsed;
        profile.fuse_ns += fuse_elapsed;
        if !fused.is_empty() {
            let update_start = Instant::now();
            for &end_state in &end_states_to_keep {
                state.merge_insert(end_state, fused.clone());
            }
            profile.fast_path_state_update_ns += update_start.elapsed().as_nanos() as u64;
            produced_state = true;
        }
    }

    if !produced_state {
        return Some(Err("commit rejected: no valid parser states remain".to_string()));
    }
    profile.fast_path_total_ns = total_start.elapsed().as_nanos() as u64;
    profile.total_ns = profile.fast_path_total_ns;
    profile.fast_path_tokenizer_exec_ns = profile.exec_ns;
    Some(Ok(()))
}

fn commit_bytes_impl_profiled(
    constraint: &Constraint,
    state: &mut ParserStateMap,
    bytes: &[u8],
    bufs: &mut CommitBuffers,
    advances: Option<&mut Vec<PerAdvanceEntry>>,
    allow_fast_paths: bool,
) -> Result<CommitProfile, String> {
    expand_runtime_product_states(constraint, state);
    let result = commit_bytes_impl_profiled_inner(
        constraint,
        state,
        bytes,
        bufs,
        advances,
        allow_fast_paths,
    );
    if result.is_ok() {
        maybe_normalize_lookahead_invariant_reductions(constraint, state);
        coalesce_uniform_runtime_source_states(constraint, state);
    }
    result
}

fn commit_bytes_impl_profiled_inner(
    constraint: &Constraint,
    state: &mut ParserStateMap,
    bytes: &[u8],
    bufs: &mut CommitBuffers,
    mut advances: Option<&mut Vec<PerAdvanceEntry>>,
    allow_fast_paths: bool,
) -> Result<CommitProfile, String> {
    // Profiling and the authoritative queue operate on the persistent map/GSS
    // representation, not the bounded duplicate-key flat frontier.
    state.normalize_duplicate_keys();
    use std::time::Instant;

    let total_start = Instant::now();
    let mut profile = CommitProfile {
        n_tokenizer_states: state.len() as u64,
        ..CommitProfile::default()
    };

    if bytes.is_empty() {
        profile.total_ns = total_start.elapsed().as_nanos() as u64;
        return Ok(profile);
    }

    let ignore_terminal = constraint.ignore_terminal;

    if allow_fast_paths
        && constraint.tokenizer_has_epsilon_transitions
        && state.len() == 1
    {
        let (&tokenizer_state, _) = state.iter().next().unwrap();
        let exec_start = Instant::now();
        let exec_result = execute_tokenizer_from_state_small(constraint, bytes, tokenizer_state);
        let exec_elapsed = exec_start.elapsed().as_nanos() as u64;
        profile.initial_exec_ns = exec_elapsed;
        profile.exec_ns = exec_elapsed;
        profile.fast_path_tokenizer_exec_ns = exec_elapsed;
        if let Some(result) = commit_bytes_fast_path_profiled(
            constraint,
            state,
            bytes,
            tokenizer_state,
            &exec_result,
            advances.as_deref_mut(),
            &mut profile,
        ) {
            return result.map(|()| profile);
        }
    }

    if allow_fast_paths && !constraint.tokenizer_has_epsilon_transitions && state.len() == 1 {
        let (&tokenizer_state, parser_gss) = state.iter().next().unwrap();
        if parser_gss.single_exclusive_top_value().is_some() {
            let direct_start = Instant::now();
            match commit_bytes_direct_linear_fast_path(
                constraint,
                parser_gss.clone(),
                bytes,
                tokenizer_state,
                Some(&mut profile),
            ) {
                Some(LinearFastPathResult::Complete(result)) => {
                    profile.linear_fast_path_total_ns = direct_start.elapsed().as_nanos() as u64;
                    let result = result.map(|final_gss| {
                        let update_start = Instant::now();
                        state.clear();
                        state.insert(constraint.runtime_commit_initial_state(), final_gss);
                        profile.linear_fast_path_state_update_ns +=
                            update_start.elapsed().as_nanos() as u64;
                        profile.total_ns = total_start.elapsed().as_nanos() as u64;
                        profile
                    });
                    return result;
                }
                Some(LinearFastPathResult::Continue { .. }) => {
                    unreachable!("direct linear fast path never returns Continue")
                }
                Some(LinearFastPathResult::Restart) | None => {
                    profile.failed_fast_path_probe_ns += direct_start.elapsed().as_nanos() as u64;
                }
            }
        }

        let exec_start = Instant::now();
        let exec_result = execute_tokenizer_from_state_small(constraint, bytes, tokenizer_state);
        let initial_exec_elapsed = exec_start.elapsed().as_nanos() as u64;
        profile.initial_exec_ns = initial_exec_elapsed;
        profile.exec_ns = initial_exec_elapsed;
        profile.fast_path_tokenizer_exec_ns = initial_exec_elapsed;

        if allow_fast_paths {
            if let Some(result) = commit_bytes_fast_path_profiled(
                constraint,
                state,
                bytes,
                tokenizer_state,
                &exec_result,
                advances.as_deref_mut(),
                &mut profile,
            ) {
                let result = result.map(|()| profile);
                return result;
            }

            let linear_eligibility_start = Instant::now();
            let linear_fast_path_eligible = !exec_result.end_state.iter().copied().any(|end_state| {
                    state
                        .values()
                        .next()
                        .is_some_and(|gss| end_state_may_advance(constraint, gss, end_state))
                });
            profile.linear_fast_path_eligibility_ns +=
                linear_eligibility_start.elapsed().as_nanos() as u64;
            if linear_fast_path_eligible {
                let linear_setup_start = Instant::now();
                let current_gss = state.values().next().unwrap();
                let start_gss = if current_gss.all_accs_satisfy(|td: &TerminalsDisallowed| td.is_empty()) {
                    current_gss.clone()
                } else {
                    prune_single_initial_state_for_exec(
                        constraint,
                        current_gss.clone(),
                        tokenizer_state,
                        &exec_result,
                        bytes,
                    )
                };
                if start_gss.is_empty() {
                    return Err("commit rejected: no valid parser states remain".to_string());
                }
                let mut linear_profile = profile.clone();
                let mut linear_advances = Vec::new();
                let linear_advances_sink = if advances.is_some() {
                    Some(&mut linear_advances)
                } else {
                    None
                };
                linear_profile.linear_fast_path_setup_ns +=
                    linear_setup_start.elapsed().as_nanos() as u64;
                match commit_bytes_linear_fast_path_profiled(
                    constraint,
                    start_gss,
                    bytes,
                    exec_result.clone(),
                    linear_advances_sink,
                    &mut linear_profile,
                ) {
                    LinearFastPathResult::Complete(result) => {
                        let result = result.map(|final_gss| {
                            if let Some(advances) = advances.as_deref_mut() {
                                advances.extend(linear_advances);
                            }
                            let update_start = Instant::now();
                            state.clear();
                            state.insert(constraint.runtime_commit_initial_state(), final_gss);
                            linear_profile.linear_fast_path_state_update_ns +=
                                update_start.elapsed().as_nanos() as u64;
                            linear_profile.total_ns = total_start.elapsed().as_nanos() as u64;
                            linear_profile
                        });
                        return result;
                    }
                    LinearFastPathResult::Continue { gss, offset } => {
                        profile = linear_profile;
                        if let Some(advances) = advances.as_deref_mut() {
                            advances.extend(linear_advances);
                        }
                        let update_start = Instant::now();
                        state.clear();
                        state.insert(constraint.runtime_commit_initial_state(), gss);
                        profile.linear_fast_path_state_update_ns +=
                            update_start.elapsed().as_nanos() as u64;

                        let queue_start = Instant::now();
                        let needed_queue_len = bytes.len() + 1;
                        let mut pending_state = ParserStatesByTokenizer::default();
                        let mut processing_queue: Vec<ParserStatesByTokenizer> =
                            (0..needed_queue_len).map(|_| ParserStatesByTokenizer::default()).collect();
                        processing_queue[offset] = std::mem::take(state).into_iter().collect();

                        let mut queue_offset = offset;
                        while queue_offset < needed_queue_len {
                            if processing_queue[queue_offset].is_empty() {
                                queue_offset += 1;
                                continue;
                            }

                            let states_to_process = std::mem::take(&mut processing_queue[queue_offset]);
                            for (tokenizer_state, gss_at_offset) in states_to_process {
                                profile.n_queue_entries += 1;

                                let actionable_start = Instant::now();
                                let actionable_terminals =
                                    ActionableTerminals::from_gss(constraint, &gss_at_offset);
                                profile.actionable_ns += actionable_start.elapsed().as_nanos() as u64;

                                let exec_start = Instant::now();
                                let exec_result = execute_tokenizer_from_state_small(
                                    constraint,
                                    &bytes[queue_offset..],
                                    tokenizer_state,
                                );
                                let queue_exec_elapsed = exec_start.elapsed().as_nanos() as u64;
                                profile.queue_exec_ns += queue_exec_elapsed;
                                profile.exec_ns += queue_exec_elapsed;

                                let match_start = Instant::now();
                                let normalized_matches = collect_unique_actionable_matches(
                                    constraint,
                                    actionable_terminals.as_ref(),
                                    ignore_terminal,
                                    &exec_result.matches,
                                    None,
                                );
                                profile.queue_match_ns += match_start.elapsed().as_nanos() as u64;

                                for matched in normalized_matches {
                                    let new_offset = queue_offset + matched.width;

                                    if matched.ignored {
                                        let enqueue_start = Instant::now();
                                        queue_parser_state(
                                            &mut processing_queue,
                                            &mut pending_state,
                                            new_offset,
                                            bytes.len(),
                                            constraint.runtime_commit_initial_state(),
                                            gss_at_offset.clone(),
                                        );
                                        profile.queue_enqueue_ns +=
                                            enqueue_start.elapsed().as_nanos() as u64;
                                        continue;
                                    }

                                    let attempt = advance_parser_stacks_profiled_if_possible(
                                        constraint,
                                        &gss_at_offset,
                                        matched.terminal_id,
                                    );
                                    let may_elapsed = attempt.may_ns;
                                    let advance_core_elapsed = attempt.core_ns;
                                    profile.advance_may_check_ns += may_elapsed;
                                    profile.advance_core_ns += advance_core_elapsed;
                                    if attempt.advanced.is_empty() {
                                        continue;
                                    }
                                    let advanced_before_disallow = attempt.advanced;
                                    let advance_profile = attempt.profile;
                                    apply_advance_profile(&mut profile, &advance_profile);

                                    if let Some(advances) = advances.as_deref_mut() {
                                        profile.adv_summary_ns += record_per_advance_entry(
                                            advances,
                                            tokenizer_state,
                                            matched.terminal_id,
                                            &gss_at_offset,
                                            &advanced_before_disallow,
                                            queue_offset,
                                            new_offset,
                                            bytes.len(),
                                            &bytes[queue_offset..new_offset],
                                            advance_profile.clone(),
                                        );
                                    }

                                    let future_start = Instant::now();
                                    let advanced = apply_future_terminal_disallow(
                                        constraint,
                                        &exec_result,
                                        matched.terminal_id,
                                        advanced_before_disallow,
                                    );
                                    let future_elapsed = future_start.elapsed().as_nanos() as u64;
                                    profile.advance_future_disallow_ns += future_elapsed;
                                    profile.advance_ns +=
                                        may_elapsed + advance_core_elapsed + future_elapsed;
                                    profile.n_advances += 1;

                                    if advanced.is_empty() {
                                        continue;
                                    }

                                    let enqueue_start = Instant::now();
                                    queue_parser_state(
                                        &mut processing_queue,
                                        &mut pending_state,
                                        new_offset,
                                        bytes.len(),
                                        constraint.runtime_commit_initial_state(),
                                        advanced,
                                    );
                                    profile.queue_enqueue_ns += enqueue_start.elapsed().as_nanos() as u64;
                                }

                                let may_start = Instant::now();
                                let admitted_end_state_terminals = batched_end_state_admitted_terminals(
                                    constraint,
                                    &gss_at_offset,
                                    &exec_result.end_state,
                                );
                                let batch_elapsed = may_start.elapsed().as_nanos() as u64;
                                profile.may_advance_ns += batch_elapsed;
                                for &end_state in &exec_result.end_state {
                                    let may_start = Instant::now();
                                    let may_advance = end_state_may_advance_with_batch(
                                        constraint,
                                        &gss_at_offset,
                                        end_state,
                                        admitted_end_state_terminals.as_ref(),
                                    );
                                    if admitted_end_state_terminals.is_none() {
                                        profile.may_advance_ns += may_start.elapsed().as_nanos() as u64;
                                    }
                                    if !may_advance {
                                        continue;
                                    }

                                    let enqueue_start = Instant::now();
                                    queue_parser_state(
                                        &mut processing_queue,
                                        &mut pending_state,
                                        bytes.len(),
                                        bytes.len(),
                                        end_state,
                                        gss_at_offset.clone(),
                                    );
                                    profile.queue_enqueue_ns += enqueue_start.elapsed().as_nanos() as u64;
                                }
                            }
                            queue_offset += 1;

                        }

                        profile.queue_ns = queue_start.elapsed().as_nanos() as u64;
                        let queue_accounted_ns = profile
                            .actionable_ns
                            .saturating_add(profile.queue_exec_ns)
                            .saturating_add(profile.queue_match_ns)
                            .saturating_add(profile.advance_ns)
                            .saturating_add(profile.may_advance_ns)
                            .saturating_add(profile.queue_enqueue_ns);
                        profile.queue_bookkeeping_ns =
                            profile.queue_ns.saturating_sub(queue_accounted_ns);

                        let fuse_start = Instant::now();
                        let new_state = finalize_pending_state(&mut pending_state);
                        profile.fuse_ns += fuse_start.elapsed().as_nanos() as u64;

                        *state = new_state;
                        if state.is_empty() {
                            return Err("commit rejected: no valid parser states remain".to_string());
                        }

                        profile.total_ns = total_start.elapsed().as_nanos() as u64;
                        return Ok(profile);
                    }
                    LinearFastPathResult::Restart => {
                        profile = linear_profile;
                    }
                }
            }
        }
    }

    let scan_start = Instant::now();
    let mut initial_scan = InitialCommitScan::collect(constraint, state, bytes);
    profile.scan_ns = scan_start.elapsed().as_nanos() as u64;

    let queue_start = Instant::now();
    let mut pending_state = ParserStatesByTokenizer::default();
    let mut processing_queue: Vec<ParserStatesByTokenizer> =
        (0..=bytes.len()).map(|_| ParserStatesByTokenizer::default()).collect();
    processing_queue[0] = std::mem::take(state).into_iter().collect();

    let mut offset = 0usize;
    while offset < processing_queue.len() {
        if processing_queue[offset].is_empty() {
            offset += 1;
            continue;
        }

        let states_to_process = std::mem::take(&mut processing_queue[offset]);
        for (tokenizer_state, mut gss_at_offset) in states_to_process {
            profile.n_queue_entries += 1;

            let exec_start = Instant::now();
            let exec_result = if offset == 0 {
                initial_scan.take_exec_result(tokenizer_state).unwrap_or_else(|| {
                    execute_tokenizer_from_state_small(constraint, &bytes[offset..], tokenizer_state)
                })
            } else {
                execute_tokenizer_from_state_small(constraint, &bytes[offset..], tokenizer_state)
            };
            let queue_exec_elapsed = exec_start.elapsed().as_nanos() as u64;
            profile.queue_exec_ns += queue_exec_elapsed;
            profile.exec_ns += queue_exec_elapsed;

            if offset == 0
                && !gss_at_offset
                    .all_accs_satisfy(|td: &TerminalsDisallowed| td.is_empty())
            {
                let prune_start = Instant::now();
                gss_at_offset = prune_single_initial_state_for_exec(
                    constraint,
                    gss_at_offset,
                    tokenizer_state,
                    &exec_result,
                    bytes,
                );
                profile.prune_ns += prune_start.elapsed().as_nanos() as u64;
                if gss_at_offset.is_empty() {
                    continue;
                }
            }

            let actionable_start = Instant::now();
            let actionable_terminals = ActionableTerminals::from_gss(constraint, &gss_at_offset);
            profile.actionable_ns += actionable_start.elapsed().as_nanos() as u64;

            let match_start = Instant::now();
            let normalized_matches = collect_unique_actionable_matches(
                constraint,
                actionable_terminals.as_ref(),
                ignore_terminal,
                &exec_result.matches,
                None,
            );
            profile.queue_match_ns += match_start.elapsed().as_nanos() as u64;

            for matched in normalized_matches {
                let new_offset = offset + matched.width;

                if matched.ignored {
                    let enqueue_start = Instant::now();
                    if !queue_parser_ignored_reset_state(
                        constraint,
                        &mut processing_queue,
                        &mut pending_state,
                        new_offset,
                        bytes.len(),
                        gss_at_offset.clone(),
                    ) {
                        return Err("recursive tokenizer reset routing failed".to_owned());
                    }
                    profile.queue_enqueue_ns += enqueue_start.elapsed().as_nanos() as u64;
                    continue;
                }

                let attempt = advance_parser_stacks_profiled_if_possible(
                    constraint,
                    &gss_at_offset,
                    matched.terminal_id,
                );
                let may_elapsed = attempt.may_ns;
                let advance_core_elapsed = attempt.core_ns;
                profile.advance_may_check_ns += may_elapsed;
                profile.advance_core_ns += advance_core_elapsed;
                if attempt.advanced.is_empty() {
                    continue;
                }
                let advanced_before_disallow = attempt.advanced;
                let advance_profile = attempt.profile;
                apply_advance_profile(&mut profile, &advance_profile);

                if let Some(advances) = advances.as_deref_mut() {
                    profile.adv_summary_ns += record_per_advance_entry(
                        advances,
                        tokenizer_state,
                        matched.terminal_id,
                        &gss_at_offset,
                        &advanced_before_disallow,
                        offset,
                        new_offset,
                        bytes.len(),
                        &bytes[offset..new_offset],
                        advance_profile.clone(),
                    );
                }

                let future_start = Instant::now();
                let advanced = apply_future_terminal_disallow(
                    constraint,
                    &exec_result,
                    matched.terminal_id,
                    advanced_before_disallow,
                );
                let future_elapsed = future_start.elapsed().as_nanos() as u64;
                profile.advance_future_disallow_ns += future_elapsed;
                profile.advance_ns += may_elapsed + advance_core_elapsed + future_elapsed;
                profile.n_advances += 1;

                if advanced.is_empty() {
                    continue;
                }

                let enqueue_start = Instant::now();
                if !queue_parser_reset_state(
                    constraint,
                    &mut processing_queue,
                    &mut pending_state,
                    new_offset,
                    bytes.len(),
                    advanced,
                ) {
                    return Err("recursive tokenizer reset routing failed".to_owned());
                }
                profile.queue_enqueue_ns += enqueue_start.elapsed().as_nanos() as u64;
            }

            let may_start = Instant::now();
            let admitted_end_state_terminals = batched_end_state_admitted_terminals(
                constraint,
                &gss_at_offset,
                &exec_result.end_state,
            );
            profile.may_advance_ns += may_start.elapsed().as_nanos() as u64;
            for &end_state in &exec_result.end_state {
                let may_start = Instant::now();
                let may_advance = end_state_may_advance_with_batch(
                    constraint,
                    &gss_at_offset,
                    end_state,
                    admitted_end_state_terminals.as_ref(),
                );
                if admitted_end_state_terminals.is_none() {
                    profile.may_advance_ns += may_start.elapsed().as_nanos() as u64;
                }
                if !may_advance {
                    continue;
                }

                let enqueue_start = Instant::now();
                queue_parser_state(
                    &mut processing_queue,
                    &mut pending_state,
                    bytes.len(),
                    bytes.len(),
                    end_state,
                    gss_at_offset.clone(),
                );
                profile.queue_enqueue_ns += enqueue_start.elapsed().as_nanos() as u64;
            }
        }
        offset += 1;
    }
    profile.queue_ns = queue_start.elapsed().as_nanos() as u64;
    let queue_accounted_ns = profile
        .actionable_ns
        .saturating_add(profile.queue_exec_ns)
        .saturating_add(profile.queue_match_ns)
        .saturating_add(profile.advance_ns)
        .saturating_add(profile.may_advance_ns)
        .saturating_add(profile.queue_enqueue_ns);
    profile.queue_bookkeeping_ns = profile.queue_ns.saturating_sub(queue_accounted_ns);

    let fuse_start = Instant::now();

    let new_state = finalize_pending_state(&mut pending_state);
    profile.fuse_ns = fuse_start.elapsed().as_nanos() as u64;

    *state = new_state;
    if state.is_empty() {
        return Err("commit rejected: no valid parser states remain".to_string());
    }

    profile.total_ns = total_start.elapsed().as_nanos() as u64;
    Ok(profile)
}

fn final_stacks(state: &ParserStateMap) -> Vec<(u32, Vec<Vec<u32>>)> {
    let mut grouped = BTreeMap::<u32, Vec<Vec<u32>>>::new();
    for (&tokenizer_state, gss) in state.iter() {
        grouped
            .entry(tokenizer_state)
            .or_default()
            .extend(parser_stacks_only(gss));
    }
    for stacks in grouped.values_mut() {
        stacks.sort();
        stacks.dedup();
    }
    grouped.into_iter().collect()
}

fn clear_state_on_commit_error<T>(
    state: &mut ParserStateMap,
    result: Result<T, String>,
) -> Result<T, String> {
    if result.is_err() {
        state.clear();
    }
    result
}

fn commit_bytes_linear_fast_path(
    constraint: &Constraint,
    start_gss: ParserGSS,
    bytes: &[u8],
    first_exec_result: TokenizerExecResult,
) -> LinearFastPathResult {
    let ignore_terminal = constraint.ignore_terminal;
    let mut gss = start_gss;
    let mut carried_stack = gss.try_virtual_stack();
    let mut offset = 0usize;
    let mut exec_result = first_exec_result;

    loop {
        let actionable_terminals = if let Some(stack) = carried_stack.as_ref() {
            stack.top().copied().map(ActionableTerminals::SingleState)
        } else {
            ActionableTerminals::from_gss(constraint, &gss)
        };
        let mut chosen: Option<(usize, u32, bool)> = None;

        for matched in &exec_result.matches {
            let ignored = is_ignored_terminal(ignore_terminal, matched.id);
            if !ignored
                && !is_actionable_terminal(
                    actionable_terminals.as_ref(),
                    constraint,
                    matched.id,
                )
            {
                continue;
            }

            let candidate = (matched.width, matched.id, ignored);
            if let Some(existing) = chosen {
                if existing != candidate {
                    return if offset > 0 {
                        LinearFastPathResult::Continue { gss, offset }
                    } else {
                        LinearFastPathResult::Restart
                    };
                }
            } else {
                chosen = Some(candidate);
            }
        }

        let Some((width, terminal, ignored)) = chosen else {
            return if offset > 0 {
                LinearFastPathResult::Continue { gss, offset }
            } else {
                LinearFastPathResult::Restart
            };
        };

        if exec_result.end_state.len() > 1 {
            return if offset > 0 {
                LinearFastPathResult::Continue { gss, offset }
            } else {
                LinearFastPathResult::Restart
            };
        }
        if let Some(end_state) = exec_result.end_state.first().copied()
            && let Some(stack) = carried_stack.as_ref()
        {
            let keep_carried = stack.top().copied().is_some_and(|top_state| {
                end_state != constraint.runtime_commit_initial_state()
                    && !constraint.parser_advance_row_intersects(
                        top_state,
                        constraint.tokenizer.possible_future_terminals(end_state),
                    )
                    && !constraint
                        .tokenizer
                        .possible_future_terminals(end_state)
                        .contains(terminal as usize)
            });
            if !keep_carried {
                gss = carried_stack.take().unwrap().into_gss();
            }
        }

        if let Some(end_state) = exec_result.end_state.first().copied() {
            if end_state_may_advance(constraint, &gss, end_state) {
                return if offset > 0 {
                    LinearFastPathResult::Continue { gss, offset }
                } else {
                    LinearFastPathResult::Restart
                };
            }
        }

        if !ignored {
            let mut shifted_carried_stack = false;
            if !template_advance_selected(constraint)
                && let Some(stack) = carried_stack.as_mut()
                && let Some(top_state) = stack.top().copied()
                && let Some(Action::Shift(target, is_replace)) = constraint.table.action(top_state, terminal)
                && exec_result.end_state.iter().copied().all(|end_state| {
                    end_state != constraint.runtime_commit_initial_state()
                        && !constraint.parser_advance_row_intersects(
                            top_state,
                            constraint.tokenizer.possible_future_terminals(end_state),
                        )
                        && !constraint
                            .tokenizer
                            .possible_future_terminals(end_state)
                            .contains(terminal as usize)
                })
            {
                if *is_replace {
                    if stack.replace_top(*target) {
                        shifted_carried_stack = true;
                    }
                } else {
                    stack.push(*target);
                    shifted_carried_stack = true;
                }
            }

            if shifted_carried_stack {
                offset += width;
                if offset == bytes.len() {
                    gss = carried_stack.take().unwrap().into_gss();
                    let fused = if constraint.direct_regular_wide_frontier_for_gss(&gss).is_some() {
                        gss
                    } else {
                        gss.fuse(Some(1))
                    };
                    if fused.is_empty() {
                        return LinearFastPathResult::Complete(Err(
                            "commit rejected: no valid parser states remain".to_string(),
                        ));
                    }
                    return LinearFastPathResult::Complete(Ok(fused));
                }

                exec_result = execute_tokenizer_from_state_small(
                    constraint,
                    &bytes[offset..],
                    constraint.runtime_commit_initial_state(),
                );
                continue;
            }

            if let Some(stack) = carried_stack.take() {
                gss = stack.into_gss();
            }

            let fast_advanced = if !template_advance_selected(constraint)
                && let Some(top_state) = gss.single_exclusive_top_value()
                && let Some(action) = constraint.table.action(top_state, terminal)
            {
                apply_single_top_action_fast(
                    constraint,
                    &gss,
                    top_state,
                    terminal,
                    action,
                )
            } else {
                None
            };

            let advanced = if let Some(advanced) = fast_advanced {
                advanced
            } else {
                let advanced = advance_parser_stacks(constraint, &gss, terminal);
                if advanced.is_empty() {
                    return if offset > 0 {
                        LinearFastPathResult::Continue { gss, offset }
                    } else {
                        LinearFastPathResult::Restart
                    };
                }
                advanced
            };
            if advanced.is_empty() {
                return LinearFastPathResult::Complete(Err(
                    "commit rejected: no valid parser states remain".to_string(),
                ));
            }
            gss = apply_future_terminal_disallow(constraint, &exec_result, terminal, advanced);
            if gss.is_empty() {
                return LinearFastPathResult::Complete(Err(
                    "commit rejected: no valid parser states remain".to_string(),
                ));
            }
        }

        offset += width;
        if offset == bytes.len() {
            if let Some(stack) = carried_stack.take() {
                gss = stack.into_gss();
            }
            let fused = if constraint.direct_regular_wide_frontier_for_gss(&gss).is_some() {
                gss
            } else {
                gss.fuse(Some(1))
            };
            if fused.is_empty() {
                return LinearFastPathResult::Complete(Err(
                    "commit rejected: no valid parser states remain".to_string(),
                ));
            }
            return LinearFastPathResult::Complete(Ok(fused));
        }

        exec_result = execute_tokenizer_from_state_small(
            constraint,
            &bytes[offset..],
            constraint.runtime_commit_initial_state(),
        );
    }
}

fn commit_bytes_linear_fast_path_profiled(
    constraint: &Constraint,
    start_gss: ParserGSS,
    bytes: &[u8],
    first_exec_result: TokenizerExecResult,
    mut advances: Option<&mut Vec<PerAdvanceEntry>>,
    profile: &mut CommitProfile,
) -> LinearFastPathResult {
    use std::time::Instant;

    let total_start = Instant::now();
    let ignore_terminal = constraint.ignore_terminal;
    let mut gss = start_gss;
    let mut offset = 0usize;
    let mut exec_result = first_exec_result;
    profile.linear_fast_path_exec_ns = profile.initial_exec_ns;

    loop {
        profile.linear_fast_path_steps += 1;

        let scan_start = Instant::now();
        let actionable_terminals = ActionableTerminals::from_gss(constraint, &gss);
        let mut chosen: Option<(usize, u32, bool)> = None;
        for matched in &exec_result.matches {
            let ignored = is_ignored_terminal(ignore_terminal, matched.id);
            if !ignored
                && !is_actionable_terminal(
                    actionable_terminals.as_ref(),
                    constraint,
                    matched.id,
                )
            {
                continue;
            }

            let candidate = (matched.width, matched.id, ignored);
            if let Some(existing) = chosen {
                if existing != candidate {
                    let result = if offset > 0 {
                        LinearFastPathResult::Continue { gss, offset }
                    } else {
                        LinearFastPathResult::Restart
                    };
                    profile.linear_fast_path_total_ns = total_start.elapsed().as_nanos() as u64;
                    return result;
                }
            } else {
                chosen = Some(candidate);
            }
        }
        profile.linear_fast_path_match_scan_ns += scan_start.elapsed().as_nanos() as u64;

        let Some((width, terminal, ignored)) = chosen else {
            let result = if offset > 0 {
                LinearFastPathResult::Continue { gss, offset }
            } else {
                LinearFastPathResult::Restart
            };
            profile.linear_fast_path_total_ns = total_start.elapsed().as_nanos() as u64;
            return result;
        };

        let end_state_start = Instant::now();
        if exec_result.end_state.len() > 1 {
            profile.linear_fast_path_end_state_check_ns +=
                end_state_start.elapsed().as_nanos() as u64;
            profile.linear_fast_path_total_ns = total_start.elapsed().as_nanos() as u64;
            return if offset > 0 {
                LinearFastPathResult::Continue { gss, offset }
            } else {
                LinearFastPathResult::Restart
            };
        }
        if let Some(end_state) = exec_result.end_state.first().copied() {
            if end_state_may_advance(constraint, &gss, end_state) {
                profile.linear_fast_path_end_state_check_ns +=
                    end_state_start.elapsed().as_nanos() as u64;
                let result = if offset > 0 {
                    LinearFastPathResult::Continue { gss, offset }
                } else {
                    LinearFastPathResult::Restart
                };
                profile.linear_fast_path_total_ns = total_start.elapsed().as_nanos() as u64;
                return result;
            }
        }
        profile.linear_fast_path_end_state_check_ns +=
            end_state_start.elapsed().as_nanos() as u64;

        if !ignored {
            let fast_start = Instant::now();
            let fast_advanced = if !template_advance_selected(constraint)
                && let Some(top_state) = gss.single_exclusive_top_value()
                && let Some(action) = constraint.table.action(top_state, terminal)
                && let Some(advanced) =
                    apply_single_top_action_fast(
                        constraint,
                        &gss,
                        top_state,
                        terminal,
                        action,
                    )
            {
                let elapsed = fast_start.elapsed().as_nanos() as u64;
                Some((advanced, fast_action_advance_profile(&gss, action, elapsed)))
            } else {
                None
            };

            let (advanced, advance_profile, advance_elapsed) =
                if let Some((advanced, advance_profile)) = fast_advanced {
                    let advance_elapsed = advance_profile.total_ns;
                    (advanced, advance_profile, advance_elapsed)
                } else {
                    let advance_start = Instant::now();
                    let (advanced, advance_profile) =
                        advance_parser_stacks_profiled(constraint, &gss, terminal);
                    let advance_elapsed = advance_start.elapsed().as_nanos() as u64;
                    if advanced.is_empty() {
                        profile.advance_core_ns += advance_elapsed;
                        let result = if offset > 0 {
                            LinearFastPathResult::Continue { gss, offset }
                        } else {
                            LinearFastPathResult::Restart
                        };
                        profile.linear_fast_path_total_ns = total_start.elapsed().as_nanos() as u64;
                        return result;
                    }
                    (advanced, advance_profile, advance_elapsed)
                };
            profile.advance_core_ns += advance_profile.total_ns;
            profile.linear_fast_path_advance_ns += advance_profile.total_ns;
            apply_advance_profile(profile, &advance_profile);

            if advanced.is_empty() {
                profile.advance_ns += advance_elapsed;
                profile.linear_fast_path_total_ns = total_start.elapsed().as_nanos() as u64;
                return LinearFastPathResult::Complete(Err(
                    "commit rejected: no valid parser states remain".to_string(),
                ));
            }

            if let Some(advances) = advances.as_deref_mut() {
                let summary_ns = record_per_advance_entry(
                    advances,
                    constraint.runtime_commit_initial_state(),
                    terminal,
                    &gss,
                    &advanced,
                    offset,
                    offset + width,
                    bytes.len(),
                    &bytes[offset..offset + width],
                    advance_profile.clone(),
                );
                profile.adv_summary_ns += summary_ns;
            }

            let future_start = Instant::now();
            gss = apply_future_terminal_disallow(constraint, &exec_result, terminal, advanced);
            let future_elapsed = future_start.elapsed().as_nanos() as u64;
            profile.advance_future_disallow_ns += future_elapsed;
            profile.linear_fast_path_future_disallow_ns += future_elapsed;
            profile.linear_fast_path_advance_ns += future_elapsed;
            profile.advance_ns += advance_elapsed + future_elapsed;
            profile.n_advances += 1;
            if gss.is_empty() {
                profile.linear_fast_path_total_ns = total_start.elapsed().as_nanos() as u64;
                return LinearFastPathResult::Complete(Err(
                    "commit rejected: no valid parser states remain".to_string(),
                ));
            }
        }

        offset += width;
        if offset == bytes.len() {
            let fuse_start = Instant::now();
            let fused = gss.fuse(Some(1));
            profile.linear_fast_path_fuse_ns = fuse_start.elapsed().as_nanos() as u64;
            profile.fuse_ns = profile.linear_fast_path_fuse_ns;
            profile.linear_fast_path_total_ns = total_start.elapsed().as_nanos() as u64;
            if fused.is_empty() {
                return LinearFastPathResult::Complete(Err(
                    "commit rejected: no valid parser states remain".to_string(),
                ));
            }
            return LinearFastPathResult::Complete(Ok(fused));
        }

        let exec_start = Instant::now();
        exec_result = execute_tokenizer_from_state_small(
            constraint,
            &bytes[offset..],
            constraint.runtime_commit_initial_state(),
        );
        let exec_elapsed = exec_start.elapsed().as_nanos() as u64;
        profile.linear_fast_path_exec_ns += exec_elapsed;
        profile.exec_ns += exec_elapsed;
    }
}

fn apply_flat_reduce(
    constraint: &Constraint,
    stack: &mut FlatInlineStack,
    nonterminal: u32,
    len: u32,
) -> Option<bool> {
    let pop = len as usize;
    if pop >= stack.len() {
        return Some(false);
    }
    stack.truncate(stack.len() - pop);
    let goto_from = *stack.last()?;
    let (target, replace) = constraint.table.goto_target(goto_from, nonterminal)?;
    if replace {
        *stack.last_mut()? = target;
    } else {
        if stack.len() == LINEAR_STACK_RESERVE {
            return None;
        }
        stack.push(target);
    }
    Some(true)
}

fn apply_flat_shift(
    stack: &mut FlatInlineStack,
    target: u32,
    replace: bool,
) -> Option<bool> {
    if replace {
        *stack.last_mut()? = target;
    } else {
        if stack.len() == LINEAR_STACK_RESERVE {
            return None;
        }
        stack.push(target);
    }
    Some(true)
}

fn apply_flat_stack_effect(
    stack: &mut FlatInlineStack,
    pop: u32,
    pushes: &[u32],
) -> Option<bool> {
    let pop = pop as usize;
    // Stack effects are atomic: popping the final visible state is valid when
    // the same effect pushes a replacement. Only a true underflow is invalid.
    if pop > stack.len() {
        return Some(false);
    }
    let new_len = stack.len() - pop + pushes.len();
    if new_len > LINEAR_STACK_RESERVE {
        return None;
    }
    stack.truncate(stack.len() - pop);
    stack.extend_from_slice(pushes);
    Some(true)
}

fn flat_guard_matches(stack: &[u32], guard: &crate::compiler::glr::table::StackShiftGuard) -> bool {
    let pop = guard.pop as usize;
    pop < stack.len()
        && guard
            .states
            .iter()
            .any(|state| *state == stack[stack.len() - 1 - pop])
}

/// Apply one terminal to one concrete LR stack, retaining a bounded set of
/// exact concrete outcomes. This is the small-ambiguity counterpart to the
/// persistent GSS interpreter: no heap allocation is permitted, and any
/// capacity overflow or accept-state edge declines to the authoritative path.
fn apply_terminal_to_flat_stacks(
    constraint: &Constraint,
    terminal: u32,
    source: &[u32],
    scratch: &mut FlatActionScratch,
) -> Option<bool> {
    if let Some(parser) = constraint.template_parser.as_ref() {
        let fast = constraint.fast_template_dfas_by_terminal.get(terminal as usize)?.as_ref()?;
        let result = flat_template::apply(fast, source, scratch);
        if result.is_some() { parser.record_advance(); }
        return result;
    }

    if source.is_empty() || source.len() > LINEAR_STACK_RESERVE {
        return None;
    }
    scratch.clear();
    let mut initial = FlatInlineStack::new();
    initial.extend_from_slice(source);
    if !scratch.push_pending(initial) {
        return None;
    }

    let mut steps = 0usize;
    while let Some(mut stack) = scratch.pending.pop() {
        steps += 1;
        if steps > FLAT_ACTION_MAX_STEPS {
            return None;
        }
        let top = *stack.last()?;
        match constraint.table.action(top, terminal)? {
            Action::Skip => {
                if !scratch.push_complete(stack) {
                    return None;
                }
            }
            Action::Shift(target, replace) => {
                if apply_flat_shift(&mut stack, *target, *replace)?
                    && !scratch.push_complete(stack)
                {
                    return None;
                }
            }
            Action::ReplaceShifts(targets) => {
                for &target in targets.iter() {
                    let mut candidate = stack.clone();
                    if apply_flat_stack_effect(&mut candidate, 1, &[target])?
                        && !scratch.push_complete(candidate)
                    {
                        return None;
                    }
                }
            }
            Action::StackShifts(shifts) => {
                for shift in shifts {
                    let mut candidate = stack.clone();
                    if apply_flat_stack_effect(&mut candidate, shift.pop, &shift.pushes)?
                        && !scratch.push_complete(candidate)
                    {
                        return None;
                    }
                }
            }
            Action::GuardedStackShifts(shifts) => {
                for shift in shifts {
                    if !shift.guards.iter().all(|guard| flat_guard_matches(&stack, guard)) {
                        continue;
                    }
                    let mut candidate = stack.clone();
                    if apply_flat_stack_effect(&mut candidate, shift.pop, &shift.pushes)?
                        && !scratch.push_complete(candidate)
                    {
                        return None;
                    }
                }
            }
            Action::Reduce(nonterminal, len) => {
                if apply_flat_reduce(constraint, &mut stack, *nonterminal, *len)?
                    && !scratch.push_pending(stack)
                {
                    return None;
                }
            }
            Action::Split {
                shift,
                reduces,
                accept,
            } => {
                if *accept {
                    return None;
                }
                if let Some((target, replace)) = shift {
                    let mut candidate = stack.clone();
                    if apply_flat_shift(&mut candidate, *target, *replace)?
                        && !scratch.push_complete(candidate)
                    {
                        return None;
                    }
                }
                for &(nonterminal, len) in reduces {
                    let mut candidate = stack.clone();
                    if apply_flat_reduce(constraint, &mut candidate, nonterminal, len)?
                        && !scratch.push_pending(candidate)
                    {
                        return None;
                    }
                }
            }
            Action::Accept => return None,
        }
    }
    Some(!scratch.complete.is_empty())
}

fn flat_stack_may_advance_on_any(
    constraint: &Constraint,
    stack: &[u32],
    terminals: &crate::ds::bitset::BitSet,
    scratch: &mut FlatActionScratch,
) -> Option<bool> {
    if let Some(parser) = constraint.template_parser.as_ref() {
        if runtime_future_contains_ignore(constraint, terminals) {
            return Some(true);
        }
        return Some(parser.admits_flat_any(stack, terminals));
    }
    let top = *stack.last()?;
    if runtime_future_contains_ignore(constraint, terminals) {
        return Some(true);
    }
    if constraint.parser_admission_policy() == AdmissionPolicy::RowPresenceExact {
        return Some(constraint.parser_advance_row_intersects(top, terminals));
    }

    let mut unknown = false;
    for terminal in terminals.iter() {
        match apply_terminal_to_flat_stacks(constraint, terminal as u32, stack, scratch) {
            Some(true) => return Some(true),
            Some(false) => {}
            None => unknown = true,
        }
    }
    (!unknown).then_some(false)
}

fn flat_frontier_group_may_advance_on_any(
    constraint: &Constraint,
    branches: &[FlatBranchScratch],
    offset: usize,
    tokenizer_state: u32,
    terminals: &crate::ds::bitset::BitSet,
    scratch: &mut FlatActionScratch,
) -> Option<bool> {
    let mut saw_branch = false;
    let mut unknown = false;
    for branch in branches {
        if branch.offset != offset || branch.tokenizer_state != tokenizer_state {
            continue;
        }
        saw_branch = true;
        match flat_stack_may_advance_on_any(
            constraint,
            &branch.stack,
            terminals,
            scratch,
        ) {
            Some(true) => return Some(true),
            Some(false) => {}
            None => unknown = true,
        }
    }

    if !saw_branch {
        Some(false)
    } else if unknown {
        None
    } else {
        Some(false)
    }
}

fn prune_flat_branch_acc(
    constraint: &Constraint,
    acc: &TerminalsDisallowed,
    tokenizer_state: u32,
    end_states: &[u32],
    matches: &[TokenizerMatch],
    bytes: &[u8],
) -> Option<Option<TerminalsDisallowed>> {
    if !acc.is_inline() {
        return None;
    }
    match advance_terminals_disallowed_over_bytes(
        constraint,
        acc,
        bytes,
        Some((tokenizer_state, end_states, matches)),
    ) {
        None => Some(None),
        Some(remapped) if remapped.is_inline() => Some(Some(remapped)),
        Some(_) => None,
    }
}

fn apply_flat_future_disallow(
    constraint: &Constraint,
    acc: &TerminalsDisallowed,
    end_states: &[u32],
    terminal: u32,
) -> Option<TerminalsDisallowed> {
    let mut updated = acc.clone();
    for &end_state in end_states {
        if constraint
            .tokenizer
            .possible_future_terminals(end_state)
            .contains(terminal as usize)
        {
            updated = updated.try_with_insert_inline(end_state, terminal)?;
        }
    }
    Some(updated)
}

/// Advance a bounded multi-state lexer frontier without rebuilding parser GSSs.
///
/// This covers tokens that only move the tokenizer while every parser stack and
/// accumulator remains unchanged.  The general queue used to materialize fresh
/// GSS objects for this case, even when a wide frontier merely carried shared
/// parser stacks under new tokenizer-state keys.  Reusing Arc-backed GSS values
/// is exact and allocation-free; duplicate tokenizer keys remain explicit until
/// a later map-only operation requests normalization.
fn try_commit_multi_state_lexer_only(
    constraint: &Constraint,
    state: &mut ParserStateMap,
    bytes: &[u8],
    tokenizer_scratch: &mut tokenizer_scan::ReusableTokenizerExecScratch,
    frontier: &mut FlatFrontierScratch,
    stack_scratch: &mut Vec<u32>,
    admission_cache: &mut SmallVec<[ParserAdmissionCacheEntry; 8]>,
) -> Option<Result<(), String>> {
    if state.len() <= 1
        || state.len() > INLINE_PARSER_STATE_CAPACITY
        || bytes.is_empty()
        || state_has_nonempty_accumulators(state)
    {
        return None;
    }

    let mut output = SmallVec::<[(u32, usize); INLINE_PARSER_STATE_CAPACITY]>::new();
    // The answer depends on BOTH the parser language and the matches produced
    // by this lexer lane. Two lexer states can share one immutable GSS while
    // producing different terminal matches for the same bytes.
    let mut actionable_cache =
        SmallVec::<[(usize, u32, bool); INLINE_PARSER_STATE_CAPACITY]>::new();
    let mut input_index = 0usize;
    while input_index < state.entries.len() {
        let tokenizer_state = state.entries[input_index].0;
        let group_end = state.entries[input_index..]
            .partition_point(|(candidate, _)| *candidate == tokenizer_state)
            + input_index;

        if !execute_tokenizer_reusable(
            constraint,
            bytes,
            tokenizer_state,
            tokenizer_scratch,
        ) {
            return None;
        }

        if tokenizer_scratch
            .matches
            .iter()
            .any(|matched| is_ignored_terminal(constraint.ignore_terminal, matched.id))
        {
            return None;
        }

        for (relative_index, (_, gss)) in state.entries[input_index..group_end].iter().enumerate() {
            let source_index = input_index + relative_index;
            let gss_key = gss.ptr_key();
            if !tokenizer_scratch.matches.is_empty() {
                let has_actionable_match = actionable_cache
                    .iter()
                    .find_map(|&(cached_gss, cached_lexer, cached_result)| {
                        (cached_gss == gss_key && cached_lexer == tokenizer_state)
                            .then_some(cached_result)
                    })
                    .unwrap_or_else(|| {
                        let actionable = ActionableTerminals::from_gss(constraint, gss);
                        let result = tokenizer_scratch.matches.iter().any(|matched| {
                            is_actionable_terminal(actionable.as_ref(), constraint, matched.id)
                        });
                        if actionable_cache.len() < actionable_cache.capacity() {
                            actionable_cache.push((gss_key, tokenizer_state, result));
                        }
                        result
                    });
                if has_actionable_match {
                    return None;
                }
            }

            let admission_cache_index = cached_batched_end_state_admission(
                constraint,
                gss,
                &tokenizer_scratch.states,
                admission_cache,
            );
            for &end_state in &tokenizer_scratch.states {
                let may_advance = if let Some(index) = admission_cache_index {
                    end_state_may_advance_from_cache_entry(
                        constraint,
                        end_state,
                        &admission_cache[index],
                    )
                } else {
                    cached_single_end_state_may_advance(
                        constraint,
                        gss,
                        end_state,
                        admission_cache,
                    )
                };
                if !may_advance {
                    continue;
                }
                if output.len() == output.capacity() {
                    return None;
                }
                output.push((end_state, source_index));
            }
        }

        input_index = group_end;
    }

    if output.is_empty() {
        return Some(Err(
            "commit rejected: no valid parser states remain".to_string(),
        ));
    }
    output.sort_unstable_by_key(|(tokenizer_state, _)| *tokenizer_state);
    let mut remaining_uses = [0u8; INLINE_PARSER_STATE_CAPACITY];
    for &(_, source_index) in &output {
        remaining_uses[source_index] = remaining_uses[source_index].saturating_add(1);
    }
    frontier.reclaim_retired_gss();
    let mut remaining_for_assignment = remaining_uses;
    let mut used_pool = [false; FLAT_FRONTIER_GSS_POOL_CAPACITY];
    let mut pool_assignment =
        SmallVec::<[usize; INLINE_PARSER_STATE_CAPACITY]>::new();
    for &(_, source_index) in &output {
        if remaining_for_assignment[source_index] <= 1 {
            pool_assignment.push(usize::MAX);
            remaining_for_assignment[source_index] = 0;
            continue;
        }
        let source_gss = &state.entries[source_index].1;
        if !source_gss.copy_single_path_stack_into(stack_scratch) {
            return None;
        }
        let Some(_acc) = source_gss.single_path_acc() else {
            return None;
        };
        let Some(pool_index) = frontier
            .gss_pool
            .iter()
            .enumerate()
            .position(|(pool_index, gss)| {
                !used_pool[pool_index]
                    && gss.can_replace_single_path_state_in_place(stack_scratch)
            })
        else {
            return None;
        };
        used_pool[pool_index] = true;
        pool_assignment.push(pool_index);
        remaining_for_assignment[source_index] -= 1;
    }

    let old_entries = std::mem::take(&mut state.entries);
    let mut old_gss = SmallVec::<[Option<ParserGSS>; INLINE_PARSER_STATE_CAPACITY]>::new();
    old_gss.extend(old_entries.into_iter().map(|(_, gss)| Some(gss)));
    let mut available_pool = SmallVec::<
        [Option<ParserGSS>; FLAT_FRONTIER_GSS_POOL_CAPACITY],
    >::new();
    available_pool.extend(frontier.gss_pool.drain(..).map(Some));
    let mut new_entries = SmallVec::<[(u32, ParserGSS); INLINE_PARSER_STATE_CAPACITY]>::new();
    for ((tokenizer_state, source_index), pool_index) in
        output.into_iter().zip(pool_assignment)
    {
        debug_assert!(remaining_uses[source_index] > 0);
        remaining_uses[source_index] -= 1;
        let gss = if pool_index == usize::MAX {
            old_gss[source_index]
                .take()
                .expect("lexer-only source GSS must be available on final use")
        } else {
            let source_gss = old_gss[source_index]
                .as_ref()
                .expect("lexer-only source GSS must remain available before final use");
            stack_scratch.clear();
            let copied = source_gss.copy_single_path_stack_into(stack_scratch);
            debug_assert!(copied);
            let acc = source_gss
                .single_path_acc()
                .expect("prevalidated lexer-only duplicate source must remain single-path");
            let mut gss = available_pool[pool_index]
                .take()
                .expect("prevalidated lexer-only pool assignment must remain available");
            let replaced = gss.try_replace_single_path_state_in_place(stack_scratch, acc);
            debug_assert!(replaced);
            gss
        };
        new_entries.push((tokenizer_state, gss));
    }
    frontier
        .gss_pool
        .extend(available_pool.into_iter().flatten());
    let mut unused_entries = SmallVec::<[(u32, ParserGSS); INLINE_PARSER_STATE_CAPACITY]>::new();
    for gss in old_gss {
        if let Some(gss) = gss {
            unused_entries.push((0, gss));
        }
    }
    frontier.recycle_old_entries(unused_entries);
    state.entries = new_entries;
    Some(Ok(()))
}

fn try_commit_flat_frontier_in_place(
    constraint: &Constraint,
    state: &mut ParserStateMap,
    bytes: &[u8],
    original: &mut Vec<u32>,
    work: &mut Vec<u32>,
    tokenizer_scratch: &mut tokenizer_scan::ReusableTokenizerExecScratch,
    frontier: &mut FlatFrontierScratch,
) -> Option<Result<(), String>> {
    frontier.reclaim_retired_gss();
    macro_rules! flat_decline {
        ($reason:literal) => {{
            if std::env::var_os("GLRMASK_DEBUG_COMMIT_PATH").is_some() {
                eprintln!("[glrmask/debug][commit_path] flat_frontier_decline reason={}", $reason);
            }
            return None;
        }};
    }
    if state.is_empty() || state.len() > FLAT_FRONTIER_MAX_BRANCHES || bytes.is_empty() {
        flat_decline!("input-shape");
    }
    frontier.clear();
    for (&tokenizer_state, gss) in state.iter() {
        if !gss.copy_single_path_stack_into(original) {
            flat_decline!("input-not-single-path");
        }
        let Some(acc) = gss.single_path_acc() else {
            flat_decline!("missing-single-acc");
        };
        if !frontier.enqueue(0, tokenizer_state, original, acc) {
            flat_decline!("initial-enqueue");
        }
    }

    let initial_tokenizer_state = constraint.runtime_commit_initial_state();
    for offset in 0..bytes.len() {
        let mut index = 0usize;
        while index < frontier.len {
            if frontier.branches[index].processed
                || frontier.branches[index].offset != offset
            {
                index += 1;
                continue;
            }

            let tokenizer_state = frontier.branches[index].tokenizer_state;
            if frontier.branches[index].stack.len() > original.capacity() {
                flat_decline!("copy-capacity");
            }
            original.clear();
            original.extend_from_slice(&frontier.branches[index].stack);
            let mut acc = frontier.branches[index].acc.clone();
            frontier.branches[index].processed = true;

            if !execute_tokenizer_reusable(
                constraint,
                &bytes[offset..],
                tokenizer_state,
                tokenizer_scratch,
            ) || tokenizer_scratch.matches.len() > SMALL_NORMALIZED_MATCH_LINEAR_SCAN_MAX
            {
                flat_decline!("tokenizer-scratch-or-matches");
            }
            let top_state = *original.last()?;
            if offset == 0 {
                // Delayed exclusions are correlated with parser alternatives,
                // but tokenizer execution is shared by every alternative under
                // one lexer key. Prune the whole group once and persist the
                // exact remapped accumulators before deciding continuation
                // viability from the group. The former per-branch local copy
                // left stale nonempty accumulators in sibling branches and
                // forced a general-path fallback even when pruning discharged
                // every exclusion.
                if !frontier.branches[index].initial_pruned {
                    let group_len = frontier.len;
                    for prune_index in 0..group_len {
                        if frontier.branches[prune_index].offset != offset
                            || frontier.branches[prune_index].tokenizer_state != tokenizer_state
                            || frontier.branches[prune_index].initial_pruned
                        {
                            continue;
                        }
                        let result = prune_flat_branch_acc(
                            constraint,
                            &frontier.branches[prune_index].acc,
                            tokenizer_state,
                            &tokenizer_scratch.states,
                            &tokenizer_scratch.matches,
                            bytes,
                        );
                        match result {
                            None => flat_decline!("prune-acc-promotion"),
                            Some(Some(pruned)) => {
                                frontier.branches[prune_index].acc = pruned;
                                frontier.branches[prune_index].initial_pruned = true;
                            }
                            Some(None) => {
                                // Keep array indices stable while removing this
                                // alternative from all later offset/key groups.
                                frontier.branches[prune_index].offset = usize::MAX;
                                frontier.branches[prune_index].processed = true;
                                frontier.branches[prune_index].initial_pruned = true;
                            }
                        }
                    }
                }
                if frontier.branches[index].offset != offset {
                    index += 1;
                    continue;
                }
                acc = frontier.branches[index].acc.clone();
            }

            let actionable = ActionableTerminals::SingleState(top_state);
            let normalized_matches = collect_unique_actionable_matches(
                constraint,
                Some(&actionable),
                constraint.ignore_terminal,
                &tokenizer_scratch.matches,
                None,
            );
            for matched in normalized_matches {
                let new_offset = offset + matched.width;
                if matched.width == 0 || new_offset > bytes.len() {
                    flat_decline!("invalid-match-width");
                }
                if matched.ignored {
                    if !frontier.enqueue(
                        new_offset,
                        initial_tokenizer_state,
                        original,
                        acc.clone(),
                    ) {
                        flat_decline!("ignored-enqueue");
                    }
                    continue;
                }

                let parser_action_result = apply_terminal_to_flat_stacks(
                    constraint,
                    matched.terminal_id,
                    original,
                    &mut frontier.action,
                );
                match parser_action_result {
                    Some(true) => {}
                    Some(false) => continue,
                    None => flat_decline!("parser-action-capacity"),
                }
                let Some(advanced_acc) = apply_flat_future_disallow(
                    constraint,
                    &acc,
                    &tokenizer_scratch.states,
                    matched.terminal_id,
                ) else {
                    return None;
                };
                for action_index in 0..frontier.action.complete.len() {
                    work.clear();
                    work.extend_from_slice(&frontier.action.complete[action_index]);
                    if !frontier.enqueue(
                        new_offset,
                        initial_tokenizer_state,
                        work,
                        advanced_acc.clone(),
                    ) {
                        flat_decline!("advanced-enqueue");
                    }
                }
            }

            for &end_state in &tokenizer_scratch.states {
                let viable = if end_state == initial_tokenizer_state {
                    true
                } else {
                    let branches = &frontier.branches[..frontier.len];
                    let mut group = branches.iter().filter(|branch| {
                        branch.offset == offset && branch.tokenizer_state == tokenizer_state
                    });
                    let Some(first) = group.next() else {
                        flat_decline!("missing-continuation-group");
                    };
                    let correlated = group.next().is_some();
                    let possible = constraint
                        .tokenizer
                        .possible_future_terminals(end_state);
                    if !correlated {
                        let Some(viable) = flat_stack_may_advance_on_any(
                            constraint,
                            &first.stack,
                            possible,
                            &mut frontier.action,
                        ) else {
                            flat_decline!("continuation-admission-ambiguous");
                        };
                        viable
                    } else if let Some(viable) = frontier.continuation_decision(
                        offset,
                        tokenizer_state,
                        end_state,
                    ) {
                        viable
                    } else {
                        let group_is_safe = offset > 0
                            || branches
                                .iter()
                                .filter(|branch| {
                                    branch.offset == offset
                                        && branch.tokenizer_state == tokenizer_state
                                })
                                .all(|branch| branch.acc.is_empty());
                        if !group_is_safe {
                            flat_decline!("initial-continuation-group-needs-pruning");
                        }
                        let Some(viable) = flat_frontier_group_may_advance_on_any(
                            constraint,
                            branches,
                            offset,
                            tokenizer_state,
                            possible,
                            &mut frontier.action,
                        ) else {
                            flat_decline!("continuation-admission-ambiguous");
                        };
                        frontier.cache_continuation_decision(
                            offset,
                            tokenizer_state,
                            end_state,
                            viable,
                        );
                        viable
                    }
                };
                if viable
                    && !frontier.enqueue(bytes.len(), end_state, original, acc.clone())
                {
                    flat_decline!("continuation-enqueue");
                }
            }
            index += 1;
        }
    }

    let mut outputs = SmallVec::<[usize; INLINE_PARSER_STATE_CAPACITY]>::new();
    for index in 0..frontier.len {
        if frontier.branches[index].offset == bytes.len() {
            // The inline capacity is a performance tier, not a semantic limit.
            // SmallVec spills once for 65-128 outputs while retaining the exact
            // flat algorithm instead of restarting through the general queue.
            outputs.push(index);
        }
    }
    if outputs.is_empty() {
        return Some(Err(
            "commit rejected: no valid parser states remain".to_string(),
        ));
    }
    outputs.sort_unstable_by(|&left, &right| {
        frontier.branches[left]
            .tokenizer_state
            .cmp(&frontier.branches[right].tokenizer_state)
            .then_with(|| frontier.branches[left].stack.cmp(&frontier.branches[right].stack))
    });

    // Use both preallocated spares and uniquely-owned active GSS objects for
    // the output double buffer. The old implementation required the spare pool
    // alone to cover every output, even when all active entries were exactly
    // recyclable. A seven-way lexer frontier could therefore decline with two
    // spares plus seven recyclable active entries and fall into the allocating
    // general path despite having nine usable objects for seven outputs.
    let recyclable_old = state
        .entries
        .iter()
        .filter(|(_, gss)| gss.can_replace_single_path_state_in_place(&[0]))
        .count();
    let retired_old = state.len() - recyclable_old;
    let available_count = frontier.gss_pool.len() + recyclable_old;
    if available_count < outputs.len()
        || available_count - outputs.len() > FLAT_FRONTIER_GSS_POOL_CAPACITY
        || frontier.retired_gss.len() + retired_old > FLAT_FRONTIER_RETIRED_GSS_CAPACITY
    {
        flat_decline!("gss-pool-capacity");
    }

    // Verify a concrete one-to-one assignment before moving anything out of
    // the active state. Match the longest output stacks first so a larger spare
    // cannot be consumed by a short stack while a smaller spare remains.
    let assignments = {
        let mut candidates = SmallVec::<[&ParserGSS; 32]>::new();
        candidates.extend(frontier.gss_pool.iter());
        candidates.extend(
            state
                .entries
                .iter()
                .map(|(_, gss)| gss)
                .filter(|gss| gss.can_replace_single_path_state_in_place(&[0])),
        );
        debug_assert_eq!(candidates.len(), available_count);

        let mut output_order = SmallVec::<[usize; INLINE_PARSER_STATE_CAPACITY]>::new();
        output_order.extend(0..outputs.len());
        output_order.sort_unstable_by_key(|&output_index| {
            std::cmp::Reverse(frontier.branches[outputs[output_index]].stack.len())
        });

        let mut used =
            [false; FLAT_FRONTIER_GSS_POOL_CAPACITY + FLAT_FRONTIER_MAX_BRANCHES];
        let mut assignments = SmallVec::<[usize; INLINE_PARSER_STATE_CAPACITY]>::new();
        assignments.resize(outputs.len(), usize::MAX);
        for output_index in output_order {
            let stack = &frontier.branches[outputs[output_index]].stack;
            let Some(candidate_index) = candidates
                .iter()
                .enumerate()
                .position(|(candidate_index, gss)| {
                    !used[candidate_index]
                        && gss.can_replace_single_path_state_in_place(stack)
                })
            else {
                flat_decline!("gss-pool-shared-or-shape");
            };
            used[candidate_index] = true;
            assignments[output_index] = candidate_index;
        }
        assignments
    };

    // Candidate order is stable: existing spares first, followed by recyclable
    // active entries. Non-recyclable active GSSs remain retained so dropping
    // shared persistent nodes never lands in the token hot path.
    let mut available = SmallVec::<[Option<ParserGSS>; 32]>::new();
    available.extend(frontier.gss_pool.drain(..).map(Some));
    let old_entries = std::mem::take(&mut state.entries);
    for (_, gss) in old_entries {
        if gss.can_replace_single_path_state_in_place(&[0]) {
            available.push(Some(gss));
        } else {
            frontier.retired_gss.push(gss);
        }
    }
    debug_assert_eq!(available.len(), available_count);

    let mut new_entries = SmallVec::<[(u32, ParserGSS); INLINE_PARSER_STATE_CAPACITY]>::new();
    for (output_index, &output) in outputs.iter().enumerate() {
        let mut gss = available[assignments[output_index]]
            .take()
            .expect("prevalidated flat-frontier GSS assignment must remain available");
        let branch = &frontier.branches[output];
        let replaced = gss.try_replace_single_path_state_in_place(
            &branch.stack,
            branch.acc.clone(),
        );
        debug_assert!(replaced, "flat-frontier spare eligibility changed");
        if !replaced {
            unreachable!("flat-frontier spare was validated before mutation");
        }
        new_entries.push((branch.tokenizer_state, gss));
    }
    for gss in available.into_iter().flatten() {
        frontier.gss_pool.push(gss);
    }
    debug_assert!(frontier.gss_pool.len() <= FLAT_FRONTIER_GSS_POOL_CAPACITY);
    state.entries = new_entries;
    Some(Ok(()))
}

fn try_commit_direct_linear_in_place(
    constraint: &Constraint,
    state: &mut ParserStateMap,
    bytes: &[u8],
    original: &mut Vec<u32>,
    work: &mut Vec<u32>,
    tokenizer_scratch: &mut tokenizer_scan::ReusableTokenizerExecScratch,
    frontier: &mut FlatFrontierScratch,
) -> Option<Result<(), String>> {
    if constraint.has_template_parser() { return None; }

    let debug_path = std::env::var_os("GLRMASK_DEBUG_COMMIT_PATH").is_some();
    let (&start_tokenizer_state, gss) = state.iter().next()?;
    let acc = gss.single_path_acc()?;
    if !acc.is_empty()
        || !gss.copy_single_path_stack_into(original)
        || original.len() > work.capacity()
    {
        return None;
    }
    work.clear();
    work.extend_from_slice(original);

    let initial_tokenizer_state = constraint.runtime_commit_initial_state();
    let mut tokenizer_state = start_tokenizer_state;
    let mut offset = 0usize;
    let final_keys: SmallVec<[u32; INLINE_PARSER_STATE_CAPACITY]> = loop {
        if offset == bytes.len() {
            break smallvec::smallvec![initial_tokenizer_state];
        }
        if !execute_tokenizer_reusable(
            constraint,
            &bytes[offset..],
            tokenizer_state,
            tokenizer_scratch,
        ) {
            return None;
        }

        let top = *work.last()?;
        let actionable = ActionableTerminals::SingleState(top);
        let normalized_matches = collect_unique_actionable_reusable_matches(
            constraint,
            Some(&actionable),
            constraint.ignore_terminal,
            &tokenizer_scratch.matches,
        );
        if debug_path {
            eprintln!(
                "[glrmask/debug][direct_linear_step] offset={} tokenizer_state={} stack={:?} raw_matches={:?} normalized={:?} end_states={:?}",
                offset,
                tokenizer_state,
                work,
                tokenizer_scratch.matches,
                normalized_matches
                    .iter()
                    .map(|m| (m.terminal_id, m.width, m.ignored))
                    .collect::<Vec<_>>(),
                tokenizer_scratch.states,
            );
        }
        if normalized_matches.len() > 1 {
            return None;
        }

        let mut viable_end_states = SmallVec::<[u32; INLINE_PARSER_STATE_CAPACITY]>::new();
        if tokenizer_scratch.states.len() > SMALL_NORMALIZED_MATCH_LINEAR_SCAN_MAX {
            // Wide lexer-only continuation after a parser action: reconstruct
            // the one current parser stack once, compute exact admission over
            // the union of all end-state futures, then classify each lexer
            // state by a cheap bitset intersection. This is the same exact
            // batching used by the authoritative queue path, but avoids one
            // parser query per raw lexer state.
            let probe_gss = ParserGSS::from_single_stack(work.to_vec(), acc.clone());
            let admitted = batched_end_state_admitted_terminals(
                constraint,
                &probe_gss,
                &tokenizer_scratch.states,
            );
            for &end_state in &tokenizer_scratch.states {
                if end_state_may_advance_with_batch(
                    constraint,
                    &probe_gss,
                    end_state,
                    admitted.as_ref(),
                ) {
                    viable_end_states.push(end_state);
                }
            }
        } else {
            for &end_state in &tokenizer_scratch.states {
                let viable = if end_state == initial_tokenizer_state {
                    true
                } else {
                    flat_stack_may_advance_on_any(
                        constraint,
                        work,
                        constraint.tokenizer.possible_future_terminals(end_state),
                        &mut frontier.action,
                    )?
                };
                if viable {
                    viable_end_states.push(end_state);
                }
            }
        }

        let Some(matched) = normalized_matches.first() else {
            if viable_end_states.is_empty() {
                return Some(Err(
                    "commit rejected: no valid parser states remain".to_string(),
                ));
            }
            break viable_end_states;
        };
        if !viable_end_states.is_empty() {
            return None;
        }

        let new_offset = offset + matched.width;
        if new_offset > bytes.len() || matched.width == 0 {
            return None;
        }
        if !matched.ignored {
            if tokenizer_scratch.states.iter().any(|&end_state| {
                constraint
                    .tokenizer
                    .possible_future_terminals(end_state)
                    .contains(matched.terminal_id as usize)
            }) {
                // The general path must attach a future-terminal exclusion.
                return None;
            }
            let applied = apply_terminal_to_flat_stacks(
                constraint,
                matched.terminal_id,
                work,
                &mut frontier.action,
            );
            if debug_path {
                eprintln!(
                    "[glrmask/debug][direct_linear_apply] terminal={} top_before={:?} table_action={:?} applied={:?} complete={:?}",
                    matched.terminal_id,
                    work.last().copied(),
                    work.last().and_then(|&top| constraint.table.action(top, matched.terminal_id)),
                    applied,
                    frontier.action.complete,
                );
            }
            match applied {
                Some(true) if frontier.action.complete.len() == 1 => {
                    work.clear();
                    work.extend_from_slice(&frontier.action.complete[0]);
                }
                Some(true) => return None,
                Some(false) => {
                    return Some(Err(
                        "commit rejected: no valid parser states remain".to_string(),
                    ));
                }
                None => return None,
            }
        }
        offset = new_offset;
        tokenizer_state = initial_tokenizer_state;
    };

    frontier
        .replace_state_with_shared_uniform_stack_keys(state, &final_keys, work, &acc)
        .then_some(Ok(()))
}


#[inline]
fn maybe_normalize_lookahead_invariant_reductions(
    constraint: &Constraint,
    state: &mut ParserStateMap,
) {
    if constraint.has_template_parser() { return; }

    if constraint.static_dynamic_overlay.is_none()
        || constraint.uses_compact_segmented_parser_runtime()
        || std::env::var_os("GLRMASK_EXPERIMENT_EAGER_INVARIANT_REDUCTIONS").is_none()
    {
        return;
    }
    for gss in state.values_mut() {
        *gss = normalize_lookahead_invariant_reductions(&constraint.table, gss);
    }
}

fn commit_bytes_impl(
    constraint: &Constraint,
    state: &mut ParserStateMap,
    bytes: &[u8],
    bufs: &mut CommitBuffers,
) -> Result<(), String> {
    expand_runtime_product_states(constraint, state);
    commit_bytes_impl_inner(constraint, state, bytes, bufs)?;

    // General symbolic residuals expose a conservative infallible future bit
    // during the inner commit so ordinary parser/lexer machinery can remain
    // unchanged. Before accepting the token, resolve every non-reset lexer key
    // through the bounded exact residual-liveness query. A hard Boolean case
    // that exceeds the work ceiling is an error, never a false live/dead bit.
    if constraint.tokenizer.has_virtual_residual_runtime() {
        let initial = constraint.runtime_commit_initial_state();
        let mut dead = Vec::<u32>::new();
        for &tokenizer_state in state.keys() {
            if tokenizer_state != initial
                && !constraint
                    .tokenizer
                    .exact_dynamic_state_has_future(tokenizer_state)?
            {
                dead.push(tokenizer_state);
            }
        }
        if !dead.is_empty() {
            state.retain(|tokenizer_state, _| dead.binary_search(tokenizer_state).is_err());
        }
        if state.is_empty() {
            return Err("commit rejected: no valid parser states remain".to_owned());
        }
    }

    coalesce_uniform_runtime_source_states(constraint, state);
    Ok(())
}

/// Exact single-lane lexer-only commit.
///
/// The parser GSS is left byte-for-byte unchanged. The shortcut applies only
/// when bounded tokenizer execution succeeds, there are no delayed exclusions,
/// and every terminal completed anywhere in `bytes` is both non-ignore and
/// parser-inadmissible. Therefore no parser action (including recursive
/// CALL/RETURN closure) can be skipped. Only viable tokenizer continuation
/// keys are substituted for the original key; every failure to prove those
/// conditions declines without mutating `state`.
fn try_commit_single_state_lexer_only(
    constraint: &Constraint,
    state: &mut ParserStateMap,
    bytes: &[u8],
    ignore_terminal: Option<u32>,
    tokenizer_scratch: &mut tokenizer_scan::ReusableTokenizerExecScratch,
    flat_frontier: &mut FlatFrontierScratch,
    stack_scratch: &mut Vec<u32>,
) -> bool {
    let [(start_tokenizer_state, gss)] = state.entries.as_slice() else {
        return false;
    };
    let start_tokenizer_state = *start_tokenizer_state;
    let gss = gss.clone();
    if !gss.all_accs_satisfy(|td: &TerminalsDisallowed| td.is_empty()) {
        return false;
    }
    let recursive_runtime = constraint.uses_compact_segmented_parser_runtime();
    let tokenizer_executed = if recursive_runtime {
        execute_recursive_tokenizer_reusable(
            constraint,
            bytes,
            start_tokenizer_state,
            tokenizer_scratch,
        )
    } else {
        execute_tokenizer_reusable(constraint, bytes, start_tokenizer_state, tokenizer_scratch)
    };
    if !tokenizer_executed {
        return false;
    }

    // Direct-regular wide-frontier summaries are monolithic-coordinate
    // accelerators. Recursive admission is already exact through the provider,
    // so deliberately do not consult those summaries here.
    let wide_frontier = (!recursive_runtime)
        .then(|| constraint.direct_regular_wide_frontier_for_gss(&gss))
        .flatten();
    if tokenizer_scratch.matches.iter().any(|matched| {
        runtime_is_ignored_terminal(constraint, ignore_terminal, matched.id)
            || wide_frontier.map_or_else(
                || parser_may_advance_on(constraint, &gss, matched.id),
                |summary| summary.actionable_terminals.contains(matched.id as usize),
            )
    }) {
        return false;
    }

    tokenizer_scratch.states.retain(|end_state| {
        wide_frontier.map_or_else(
            || end_state_may_advance(constraint, &gss, *end_state),
            |summary| wide_frontier_end_state_may_advance(constraint, summary, *end_state),
        )
    });

    // The parser frontier is unchanged. Re-key its existing Arc directly
    // instead of decomposing, rebuilding, and fusing the represented stack
    // language.
    if state.replace_single_keys(&tokenizer_scratch.states) {
        return true;
    }
    if let Some(acc) = gss.single_path_acc()
        && gss.copy_single_path_stack_into(stack_scratch)
        && flat_frontier.replace_state_with_uniform_stack_keys(
            state,
            &tokenizer_scratch.states,
            stack_scratch,
            &acc,
        )
    {
        return true;
    }
    false
}

fn commit_bytes_impl_inner(
    constraint: &Constraint,
    state: &mut ParserStateMap,
    bytes: &[u8],
    bufs: &mut CommitBuffers,
) -> Result<(), String> {
    if bytes.is_empty() {
        return Ok(());
    }

    let debug_path = std::env::var_os("GLRMASK_DEBUG_COMMIT_PATH").is_some();
    if debug_path {
        eprintln!(
            "[glrmask/debug][commit_input] entries={} keys_paths={:?}",
            state.len(),
            state
                .iter()
                .map(|(&key, gss)| (key, gss.path_count_at_most(17)))
                .collect::<Vec<_>>(),
        );
    }
    let ignore_terminal = constraint.ignore_terminal;
    // Linker control terminals are zero-width parser transitions.  The
    // allocation-free commit fast paths assume that every parser transition is
    // driven by a completed lexer terminal and therefore cannot yet preserve
    // control closure between lexemes.  Keep explicit-control tables on the
    // authoritative queue path until each fast path has its own equivalence
    // proof/implementation.
    let has_linker_controls = constraint.parser_has_controls()
        || constraint.uses_compact_segmented_parser_runtime();
    let direct_dynamic = constraint.uses_dynamic_runtime()
        && constraint.direct_regular_automaton.is_some();
    if state.len() <= 1 && !bufs.admission_cache.is_empty() {
        bufs.admission_cache.clear();
    }

    // A wide direct-regular frontier already carries its exact actionable
    // terminal support. Scan only that support instead of materializing every
    // tokenizer finalizer, which can be proportional to the remaining source
    // file in project-scale diff grammars.
    if !has_linker_controls
        && state.len() == 1
        && !constraint.tokenizer_has_epsilon_transitions
    {
        let (&start_tokenizer_state, gss) = state.iter().next().unwrap();
        if let Some(summary) = constraint.direct_regular_wide_frontier_for_gss(gss)
            && let Some(accumulator) = gss.uniform_accumulator()
            && let Some(end_state) = scan_wide_frontier_lexer_only(
                constraint,
                bytes,
                start_tokenizer_state,
                summary,
            )
        {
            let Some(updated_accumulator) = advance_uniform_disallowed_interest_only(
                constraint,
                &accumulator,
                bytes,
            ) else {
                state.clear();
                return Err("commit rejected: no valid parser states remain".to_string());
            };
            if !wide_frontier_end_state_may_advance(constraint, summary, end_state) {
                state.clear();
                return Err("commit rejected: no valid parser states remain".to_string());
            }
            if let Some(updated) = gss.with_uniform_accumulator(updated_accumulator) {
                state.clear();
                state.insert(end_state, updated);
                return Ok(());
            }
        }
    }

    if !has_linker_controls
        && !direct_dynamic
        && state.len() == 1
        && let Some(result) = try_commit_direct_linear_in_place(
            constraint,
            state,
            bytes,
            &mut bufs.linear_stack_original,
            &mut bufs.linear_stack_work,
            &mut bufs.reusable_tokenizer_exec,
            &mut bufs.flat_frontier,
        )
    {
        if debug_path {
            eprintln!("[glrmask/debug][commit_path] direct_linear states={}", state.len());
        }
        return result;
    }

    if !has_linker_controls && !direct_dynamic {
        let lexer_only_result = try_commit_multi_state_lexer_only(
            constraint,
            state,
            bytes,
            &mut bufs.reusable_tokenizer_exec,
            &mut bufs.flat_frontier,
            &mut bufs.linear_stack_original,
            &mut bufs.admission_cache,
        );
        if let Some(result) = lexer_only_result {
            if debug_path {
                eprintln!("[glrmask/debug][commit_path] multi_state_lexer_only states={}", state.len());
            }
            return result;
        }
    }
    // Exact-simulation tables make continuation admission the dominant cost
    // for tiny composed frontiers.  The small queue batches that exact set query,
    // so try it before the flat accelerator (whose bounded proof may decline and
    // duplicate the work) when its ordinary profitability bounds already hold.
    if !has_linker_controls
        && !direct_dynamic
        && constraint.parser_admission_policy() == AdmissionPolicy::ExactSimulation
        && bytes.len() <= 16
        && state.len() <= 8
    {
        let early_small_queue = commit_bytes_small_queue_fast_path(
            constraint,
            state,
            bytes,
            &mut bufs.reusable_tokenizer_exec,
            &mut bufs.small_queue,
            &mut bufs.admission_cache,
            &mut bufs.prune_tokenizer_exec,
        );
        if let Some(result) = early_small_queue {
            if debug_path {
                eprintln!("[glrmask/debug][commit_path] early_small_queue states={}", state.len());
            }
            return result;
        }
    }

    if !has_linker_controls
        && !direct_dynamic
        && state.len() <= FLAT_FRONTIER_MAX_BRANCHES
        && let Some(result) = try_commit_flat_frontier_in_place(
            constraint,
            state,
            bytes,
            &mut bufs.linear_stack_original,
            &mut bufs.linear_stack_work,
            &mut bufs.reusable_tokenizer_exec,
            &mut bufs.flat_frontier,
        )
    {
        if debug_path {
            eprintln!("[glrmask/debug][commit_path] flat_frontier states={}", state.len());
        }
        return result;
    }

    // All remaining paths use map/GSS semantics. Materialize any bounded flat
    // alternatives only after the allocation-free path has declined.
    state.normalize_duplicate_keys();

    // Exact no-match self-loops are the simplest lexer-only case: neither the
    // tokenizer key nor the parser GSS changes.
    if !has_linker_controls
        && !constraint.tokenizer_has_epsilon_transitions
        && state.len() == 1
    {
        let (&start_tokenizer_state, _) = state.iter().next().unwrap();
        let mut tokenizer_state = start_tokenizer_state;
        let mut no_matches = true;
        for &byte in bytes {
            tokenizer_state = constraint.tokenizer_fast_transitions.transition(
                &constraint.tokenizer,
                tokenizer_state,
                byte,
            );
            if tokenizer_state == u32::MAX
                || constraint
                    .tokenizer
                    .matched_terminals_iter(tokenizer_state)
                    .next()
                    .is_some()
            {
                no_matches = false;
                break;
            }
        }
        if no_matches && tokenizer_state == start_tokenizer_state {
            return Ok(());
        }
    }

    // A large fraction of model tokens only advance the lexer. This proof is
    // valid in both monolithic and recursive leaf-scoped coordinates.
    if try_commit_single_state_lexer_only(
        constraint,
        state,
        bytes,
        ignore_terminal,
        &mut bufs.reusable_tokenizer_exec,
        &mut bufs.flat_frontier,
        &mut bufs.linear_stack_original,
    ) {
        if debug_path && constraint.uses_compact_segmented_parser_runtime() {
            eprintln!(
                "[glrmask/debug][commit_path] recursive_lexer_only states={}",
                state.len(),
            );
        }
        return Ok(());
    }

    // Common deterministic case: consume one complete model token and mutate
    // the uniquely-owned linear parser stack directly. All eligibility checks
    // happen before mutation, so declining this path leaves the old fallback
    // semantics untouched. A plain shift is exact regardless of whether the
    // equivalent template-DFA advance is enabled.
    if !has_linker_controls && state.len() == 1 {
        let initial_tokenizer_state = constraint.runtime_commit_initial_state();
        let (&tokenizer_state, gss) = state.iter().next().unwrap();
        if tokenizer_state == initial_tokenizer_state
            && gss.all_accs_satisfy(|td: &TerminalsDisallowed| td.is_empty())
            && let Some(top_state) = gss.single_exclusive_top_value()
            && let Some(step) = choose_direct_linear_step(
                constraint,
                gss,
                bytes,
                tokenizer_state,
                Some(top_state),
            )
            && step.width == bytes.len()
            && !step.ignored
        {
            let continuation_is_inert = step.end_state.is_none_or(|end_state| {
                let future = constraint.tokenizer.possible_future_terminals(end_state);
                !end_state_may_advance(constraint, gss, end_state)
                    && !future.contains(step.terminal as usize)
            });
            if continuation_is_inert
                && !constraint.has_template_parser()
                && let Some(Action::Shift(target, replace)) =
                    constraint.table.action(top_state, step.terminal)
                && let Some(gss) = state.values_mut().next()
            {
                let pushes = [*target];
                if gss.try_apply_single_segment_stack_effect_in_place(
                    usize::from(*replace),
                    &pushes,
                ) {
                    return Ok(());
                }
            }
        }
    }

    if !has_linker_controls && state.len() == 1 {
        let (&tokenizer_state, _) = state.iter().next().unwrap();
        let exec_result = execute_tokenizer_from_state_small(constraint, bytes, tokenizer_state);
        if let Some(result) = commit_bytes_fast_path(
            constraint,
            state,
            bytes,
            tokenizer_state,
            &exec_result,
        ) {
            return result;
        }
    }

    // Single tokenizer state: execute tokenizer ONCE, try fast path, reuse result
    if !has_linker_controls && state.len() == 1 {
        let (&tokenizer_state, parser_gss) = state.iter().next().unwrap();
        if parser_gss.single_exclusive_top_value().is_some() {
            if let Some(result) = commit_bytes_direct_linear_fast_path(
                constraint,
                parser_gss.clone(),
                bytes,
                tokenizer_state,
                None,
            ) {
                match result {
                    LinearFastPathResult::Complete(result) => match result {
                        Ok(final_gss) => {
                            state.clear();
                            state.insert(constraint.runtime_commit_initial_state(), final_gss);
                            return Ok(());
                        }
                        Err(err) => return Err(err),
                    },
                    LinearFastPathResult::Continue { gss, offset } => {
                        state.clear();
                        state.insert(constraint.runtime_commit_initial_state(), gss);
                        return commit_bytes_impl(constraint, state, &bytes[offset..], bufs);
                    }
                    LinearFastPathResult::Restart => {}
                }
            }
        }
    }

    if !has_linker_controls {
        let language_small_queue_result = commit_bytes_language_small_queue_fast_path(
            constraint,
            state,
            bytes,
            &mut bufs.reusable_tokenizer_exec,
            &mut bufs.small_queue,
            &mut bufs.template_advance_runtime,
            false,
        );
        if let Some(result) = language_small_queue_result {
            return result;
        }
    }

    let small_queue_result = if constraint.uses_compact_segmented_parser_runtime() {
        None
    } else {
        commit_bytes_small_queue_fast_path(
            constraint,
            state,
            bytes,
            &mut bufs.reusable_tokenizer_exec,
            &mut bufs.small_queue,
            &mut bufs.admission_cache,
            &mut bufs.prune_tokenizer_exec,
        )
    };
    if let Some(result) = small_queue_result {
        if debug_path {
            eprintln!("[glrmask/debug][commit_path] small_queue states={}", state.len());
        }
        return result;
    }

    if !constraint.uses_compact_segmented_parser_runtime()
        && let Some(result) = commit_bytes_full_width_fast_path(constraint, state, bytes)
    {
        if debug_path {
            eprintln!("[glrmask/debug][commit_path] full_width states={}", state.len());
        }
        return result;
    }

    if !has_linker_controls
        && !constraint.tokenizer_has_epsilon_transitions
        && state.len() == 1
    {
        let (&tokenizer_state, _) = state.iter().next().unwrap();
        let exec_result = execute_tokenizer_from_state_small(constraint, bytes, tokenizer_state);

        // Try fast path with pre-computed exec_result
        if let Some(result) = commit_bytes_fast_path(
            constraint, state, bytes, tokenizer_state, &exec_result,
        ) {
            return result;
        }

        if !exec_result
            .end_state
            .iter()
            .copied()
            .any(|end_state| {
                state
                    .values()
                    .next()
                    .is_some_and(|gss| end_state_may_advance(constraint, gss, end_state))
            })
        {
            let current_gss = state.values().next().unwrap();
            let start_gss = if current_gss.all_accs_satisfy(|td: &TerminalsDisallowed| td.is_empty()) {
                current_gss.clone()
            } else {
                prune_single_initial_state_for_exec(
                    constraint,
                    current_gss.clone(),
                    tokenizer_state,
                    &exec_result,
                    bytes,
                )
            };
            if start_gss.is_empty() {
                return Err("commit rejected: no valid parser states remain".to_string());
            }
            match commit_bytes_linear_fast_path(
                constraint,
                start_gss,
                bytes,
                exec_result.clone(),
            ) {
                LinearFastPathResult::Complete(result) => {
                    match result {
                        Ok(final_gss) => {
                            state.clear();
                            state.insert(constraint.runtime_commit_initial_state(), final_gss);
                            return Ok(());
                        }
                        Err(err) => return Err(err),
                    }
                }
                LinearFastPathResult::Continue { gss, offset } => {
                    bufs.clear_all();
                    state.clear();
                    state.insert(constraint.runtime_commit_initial_state(), gss);

                    if bytes.len() - offset == 1 {
                        return commit_bytes_impl(constraint, state, &bytes[offset..], bufs);
                    }

                    let needed_queue_len = bytes.len() + 1;
                    let mut processing_queue = std::mem::take(&mut bufs.processing_queue);
                    if processing_queue.len() < needed_queue_len {
                        processing_queue.resize_with(needed_queue_len, ParserStatesByTokenizer::default);
                    }
                    for bucket in processing_queue.iter_mut().take(needed_queue_len) {
                        bucket.clear();
                    }
                    processing_queue[offset] = std::mem::take(state).into_iter().collect();

                    let mut offset = offset;
                    while offset < needed_queue_len {
                        if processing_queue[offset].is_empty() {
                            offset += 1;
                            continue;
                        }

                        let states_to_process = std::mem::take(&mut processing_queue[offset]);
                        for (tokenizer_state, gss_at_offset) in states_to_process {
                            let actionable_terminals = ActionableTerminals::from_gss(constraint, &gss_at_offset);
                            let exec_result = execute_tokenizer_from_state_small(
                                constraint,
                                &bytes[offset..],
                                tokenizer_state,
                            );

                            bufs.terminal_result_cache.clear();

                            let normalized_matches = collect_unique_actionable_matches(
                                constraint,
                                actionable_terminals.as_ref(),
                                ignore_terminal,
                                &exec_result.matches,
                                Some(&mut bufs.seen_matches),
                            );

                            for matched in normalized_matches {
                                let new_offset = offset + matched.width;

                                if matched.ignored {
                                    queue_parser_state(
                                        &mut processing_queue,
                                        &mut bufs.pending_state,
                                        new_offset,
                                        bytes.len(),
                                        constraint.runtime_commit_initial_state(),
                                        gss_at_offset.clone(),
                                    );
                                    continue;
                                }

                                let Some(gss) = advance_terminal_match(
                                    constraint,
                                    &gss_at_offset,
                                    matched.terminal_id,
                                    &exec_result,
                                    &mut bufs.advance_result_cache,
                                    &mut bufs.terminal_result_cache,
                                ) else {
                                    continue;
                                };

                                queue_parser_state(
                                    &mut processing_queue,
                                    &mut bufs.pending_state,
                                    new_offset,
                                    bytes.len(),
                                    constraint.runtime_commit_initial_state(),
                                    gss,
                                );
                            }

                            let admission_cache_index = cached_batched_end_state_admission(
                                constraint,
                                &gss_at_offset,
                                &exec_result.end_state,
                                &mut bufs.admission_cache,
                            );
                            for &end_state in &exec_result.end_state {
                                let may_advance = if let Some(index) = admission_cache_index {
                                    end_state_may_advance_from_cache_entry(
                                        constraint,
                                        end_state,
                                        &bufs.admission_cache[index],
                                    )
                                } else {
                                    cached_single_end_state_may_advance(
                                        constraint,
                                        &gss_at_offset,
                                        end_state,
                                        &mut bufs.admission_cache,
                                    )
                                };
                                if !may_advance {
                                    continue;
                                }

                                queue_parser_state(
                                    &mut processing_queue,
                                    &mut bufs.pending_state,
                                    bytes.len(),
                                    bytes.len(),
                                    end_state,
                                    gss_at_offset.clone(),
                                );
                            }
                        }
                    }

                    let new_state = finalize_pending_state(&mut bufs.pending_state);

                    *state = new_state;
                    bufs.processing_queue = processing_queue;
                    if state.is_empty() {
                        return Err("commit rejected: no valid parser states remain".to_string());
                    }
                    return Ok(());
                }
                LinearFastPathResult::Restart => {}
            }
        }

        // Fast path failed â€” build scan data from already-computed exec_result
        bufs.clear_all();
        bufs.exec_results.insert(tokenizer_state, exec_result);
    } else {
        bufs.clear_all();

        for &tokenizer_state in state.keys() {
            let exec_result = execute_tokenizer_from_state_small(constraint, bytes, tokenizer_state);
            bufs.exec_results.insert(tokenizer_state, exec_result);
        }
    }

    let needed_queue_len = bytes.len() + 1;
    let mut processing_queue = std::mem::take(&mut bufs.processing_queue);
    if processing_queue.len() < needed_queue_len {
        processing_queue.resize_with(needed_queue_len, ParserStatesByTokenizer::default);
    }
    for bucket in processing_queue.iter_mut().take(needed_queue_len) {
        bucket.clear();
    }
    processing_queue[0] = std::mem::take(state).into_iter().collect();

    let mut offset = 0usize;
    while offset < needed_queue_len {
        if processing_queue[offset].is_empty() {
            offset += 1;
            continue;
        }

        let states_to_process = std::mem::take(&mut processing_queue[offset]);
        for (tokenizer_state, mut gss_at_offset) in states_to_process {
            let exec_result = if offset == 0 {
                bufs.exec_results.remove(&tokenizer_state).unwrap_or_else(|| {
                    execute_tokenizer_from_state_small(constraint, &bytes[offset..], tokenizer_state)
                })
            } else {
                execute_tokenizer_from_state_small(constraint, &bytes[offset..], tokenizer_state)
            };

            if offset == 0
                && !gss_at_offset
                    .all_accs_satisfy(|td: &TerminalsDisallowed| td.is_empty())
            {
                gss_at_offset = prune_single_initial_state_for_exec(
                    constraint,
                    gss_at_offset,
                    tokenizer_state,
                    &exec_result,
                    bytes,
                );
                if gss_at_offset.is_empty() {
                    continue;
                }
            }

            let actionable_terminals = ActionableTerminals::from_gss(constraint, &gss_at_offset);

            bufs.terminal_result_cache.clear();

            let normalized_matches = collect_unique_actionable_matches(
                constraint,
                actionable_terminals.as_ref(),
                ignore_terminal,
                &exec_result.matches,
                Some(&mut bufs.seen_matches),
            );

            for matched in normalized_matches {
                let new_offset = offset + matched.width;

                if matched.ignored {
                    if !queue_parser_ignored_reset_state(
                        constraint,
                        &mut processing_queue,
                        &mut bufs.pending_state,
                        new_offset,
                        bytes.len(),
                        gss_at_offset.clone(),
                    ) {
                        return Err("recursive tokenizer reset routing failed".to_owned());
                    }
                    continue;
                }

                let Some(gss) = advance_terminal_match(
                    constraint,
                    &gss_at_offset,
                    matched.terminal_id,
                    &exec_result,
                    &mut bufs.advance_result_cache,
                    &mut bufs.terminal_result_cache,
                ) else {
                    continue;
                };

                if !queue_parser_reset_state(
                    constraint,
                    &mut processing_queue,
                    &mut bufs.pending_state,
                    new_offset,
                    bytes.len(),
                    gss,
                ) {
                    return Err("recursive tokenizer reset routing failed".to_owned());
                }
            }

            let admission_cache_index = cached_batched_end_state_admission(
                constraint,
                &gss_at_offset,
                &exec_result.end_state,
                &mut bufs.admission_cache,
            );
            for &end_state in &exec_result.end_state {
                let may_advance = if let Some(index) = admission_cache_index {
                    end_state_may_advance_from_cache_entry(
                        constraint,
                        end_state,
                        &bufs.admission_cache[index],
                    )
                } else {
                    cached_single_end_state_may_advance(
                        constraint,
                        &gss_at_offset,
                        end_state,
                        &mut bufs.admission_cache,
                    )
                };
                if !may_advance {
                    continue;
                }

                queue_parser_state(
                    &mut processing_queue,
                    &mut bufs.pending_state,
                    bytes.len(),
                    bytes.len(),
                    end_state,
                    gss_at_offset.clone(),
                );
            }
        }

        offset += 1;
    }

    let new_state = finalize_pending_state(&mut bufs.pending_state);

    *state = new_state;
    bufs.processing_queue = processing_queue;
    if state.is_empty() {
        return Err("commit rejected: no valid parser states remain".to_string());
    }

    Ok(())
}

impl<'a> ConstraintState<'a> {
    pub(crate) fn knows_token_id(&self, token_id: u32) -> bool {
        token_bytes_for_id(self.constraint, token_id).is_some()
            || self.constraint.has_special_token_id(token_id)
    }

    /// Snapshot the exact runtime state only when a reusable mask is currently
    /// cached. `ParserGSS` is Arc-backed, so cloning this small map keeps the
    /// old GSS identities alive without copying parser stacks.
    #[inline]
    fn snapshot_current_mask_state(&self) -> Option<ParserStateMap> {
        let cache = self.mask_cache.lock().unwrap();
        cache
            .as_ref()
            .is_some_and(|cache_data| cache_data.generation == self.generation)
            .then(|| self.state.clone())
    }

    /// Advance the commit generation while retaining a cached mask when commit
    /// left the complete semantic runtime state unchanged. `ParserStateMap`
    /// equality compares tokenizer keys and exact immutable GSS structure;
    /// `LeveledGSS` itself checks Arc identity first, so the usual unchanged
    /// object case remains a pointer comparison. `fill_mask` is a pure
    /// function of this state and the immutable constraint.
    #[inline]
    fn finish_commit_generation(
        &mut self,
        previous_state: Option<ParserStateMap>,
        commit_succeeded: bool,
    ) {
        let previous_generation = self.generation;
        self.generation += 1;
        if !commit_succeeded {
            return;
        }
        let Some(previous_state) = previous_state else {
            return;
        };
        let mask_state_unchanged = previous_state == self.state
            || (self.constraint.uses_dynamic_runtime()
                // Ordinary tokenizer quotients cannot interpret scoped
                // recursive composition IDs. Only exact state equality above
                // is currently proved for retaining a composed result.
                && !self.constraint.uses_compact_segmented_parser_runtime()
                && (self.dynamic_mask_projection_state_eq(&previous_state, &self.state)
                    || (std::env::var_os(
                        "GLRMASK_DISABLE_DYNAMIC_MASK_PARSER_RELATIVE_COMMIT_REUSE",
                    )
                    .is_none()
                        && self.dynamic_parser_relative_vocab_mask_eq(&previous_state))));
        if !mask_state_unchanged {
            return;
        }

        let mut cache = self.mask_cache.lock().unwrap();
        if let Some(cache_data) = cache.as_mut()
            && cache_data.generation == previous_generation
        {
            cache_data.generation = self.generation;
        }
    }

    /// Exact equality in the lexer coordinate actually consumed by dynamic
    /// mask generation. A commit may advance the source/runtime lexer through
    /// states that are distinct only outside the model-vocabulary horizon; if
    /// every correlated parser GSS is unchanged and those source states map to
    /// the same mask-execution state, the next-token mask is identical and the
    /// already-cached mask can survive the generation bump.
    ///
    /// This is deliberately stricter than general semantic cache lookup: it
    /// does not merge parser alternatives or rely on observation classes. It
    /// only replaces the exact tokenizer-state coordinate with the exact
    /// finite mask coordinate selected by the constraint.
    fn dynamic_mask_projection_state_eq(
        &self,
        previous: &ParserStateMap,
        current: &ParserStateMap,
    ) -> bool {
        if previous.len() != current.len() {
            return false;
        }
        let vocab = self.constraint.dynamic_mask_vocab_for_runtime();
        let exact_initial = self.constraint.tokenizer.initial_state();

        // The common case is one correlated lexer/parser branch. Keep that
        // path allocation-free and constant-time.
        if let ([(previous_state, previous_gss)], [(current_state, current_gss)]) =
            (previous.entries.as_slice(), current.entries.as_slice())
        {
            return previous_gss == current_gss
                && (*previous_state == exact_initial) == (*current_state == exact_initial)
                && vocab.mask_runtime_state(*previous_state)
                    == vocab.mask_runtime_state(*current_state);
        }

        // Flat GLR frontiers can contain duplicate exact tokenizer keys. Match
        // correlated branches as a multiset in mask-coordinate space rather
        // than assuming source-state ordering survives quotienting.
        let mut matched = vec![false; current.len()];
        'previous: for (previous_state, previous_gss) in &previous.entries {
            let projected = vocab.mask_runtime_state(*previous_state);
            let initial = *previous_state == exact_initial;
            for (index, (current_state, current_gss)) in current.entries.iter().enumerate() {
                if !matched[index]
                    && initial == (*current_state == exact_initial)
                    && projected == vocab.mask_runtime_state(*current_state)
                    && previous_gss == current_gss
                {
                    matched[index] = true;
                    continue 'previous;
                }
            }
            return false;
        }
        true
    }

    /// Prove equality of the next-token mask for the common singleton parser
    /// frontier even when the exact lexer residual changed.
    ///
    /// This proof is deliberately vocabulary- and parser-relative. Two lexer
    /// states may differ because one still carries futures for terminals the
    /// unchanged parser frontier cannot consume. Those differences do not
    /// affect the next model-token mask. We walk the model vocabulary trie in
    /// lockstep and compare only matched/future observations for terminals
    /// admitted by this parser GSS. Once the lexer states converge, or both
    /// no-finalization branches become parser-dead, the complete suffix
    /// subtree is equal and no further work is needed.
    ///
    /// Equal admitted finalizers also need no recursive parser simulation:
    /// both sides advance the same GSS on the same terminal and reset to the
    /// same lexer initial state, so every post-finalization suffix branch is
    /// identical by construction.
    fn dynamic_parser_relative_vocab_mask_eq(
        &mut self,
        previous: &ParserStateMap,
    ) -> bool {
        let current = &self.state;
        if self.constraint.static_dynamic_overlay.is_some()
            || self.constraint.uses_compact_segmented_parser_runtime()
            || self.constraint.tokenizer.has_any_virtual_runtime()
        {
            return false;
        }
        let (left_source, left_gss, right_source, right_gss) =
            if let ([(left_source, left_gss)], [(right_source, right_gss)]) =
                (previous.entries.as_slice(), current.entries.as_slice())
            {
                (left_source, left_gss, right_source, right_gss)
            } else {
                // Conservative multi-branch extension: first pair off every
                // branch whose exact correlated lexer/GSS state is unchanged.
                // Only if exactly one branch remains on each side do we apply
                // the parser-relative lexer proof below.  This covers the
                // common GLR shape where one guarded/reset branch is stable
                // and one ordinary lexer residual advances, without needing a
                // general assignment search over unrelated alternatives.
                if previous.len() != current.len() {
                    return false;
                }
                let mut matched = vec![false; current.len()];
                let mut unmatched_previous = None::<(&u32, &ParserGSS)>;
                for (previous_state, previous_gss) in &previous.entries {
                    let exact = current
                        .entries
                        .iter()
                        .enumerate()
                        .find(|(index, (current_state, current_gss))| {
                            !matched[*index]
                                && previous_state == current_state
                                && previous_gss == current_gss
                        })
                        .map(|(index, _)| index);
                    if let Some(index) = exact {
                        matched[index] = true;
                        continue;
                    }
                    if unmatched_previous
                        .replace((previous_state, previous_gss))
                        .is_some()
                    {
                        return false;
                    }
                }
                let Some((left_source, left_gss)) = unmatched_previous else {
                    return false;
                };
                let mut unmatched_current = current
                    .entries
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| !matched[*index])
                    .map(|(_, pair)| pair);
                let Some((right_source, right_gss)) = unmatched_current.next() else {
                    return false;
                };
                if unmatched_current.next().is_some() {
                    return false;
                }
                (left_source, left_gss, right_source, right_gss)
            };
        if left_gss != right_gss
            || !left_gss.all_accs_satisfy(|acc: &TerminalsDisallowed| acc.is_empty())
        {
            return false;
        }

        let tokenizer = &self.constraint.tokenizer;
        let initial = tokenizer.initial_state();
        if (*left_source == initial) != (*right_source == initial)
            || tokenizer.state_has_epsilon_transitions(*left_source)
            || tokenizer.state_has_epsilon_transitions(*right_source)
        {
            return false;
        }

        let (cache_left, cache_right) = if left_source <= right_source {
            (*left_source, *right_source)
        } else {
            (*right_source, *left_source)
        };
        let candidates = crate::ds::bitset::BitSet::all(
            self.constraint.parser_terminal_count() as usize,
        );
        let mut admitted = exact_admitted_terminals_for_candidates(
            self.constraint,
            left_gss,
            &candidates,
        );
        // IGNORE is parser-transparent but still lexer-observable: a live
        // ignore future permits a token boundary, and an ignore match resets
        // the lexer while leaving the current GSS unchanged.  Treat it as an
        // always-observed terminal in this equivalence proof. Equal ignore
        // events therefore converge exactly just like equal parser-admitted
        // finalizers, except without a parser advance.
        if let Some(ignore) = self.constraint.ignore_terminal {
            admitted.set(ignore as usize);
        }
        if admitted.count_ones() == 0 {
            return true;
        }

        if self
            .buffers
            .parser_relative_mask_eq_cache
            .iter()
            .any(|entry| {
                entry.left == cache_left
                    && entry.right == cache_right
                    && entry.admitted == admitted
            })
        {
            if std::env::var_os("GLRMASK_PROFILE_DYNAMIC_MASK_COMMIT_REUSE").is_some() {
                eprintln!(
                    "[glrmask/profile][parser_relative_commit_reuse] generation={} result=cache_hit left={} right={} admitted={}",
                    self.generation,
                    left_source,
                    right_source,
                    admitted.count_ones(),
                );
            }
            return true;
        }
        // A new relation certificate can cost tens of microseconds. If the
        // destination state already has a materialized global dynamic-mask
        // cache entry, proving equality with the previous state cannot beat
        // simply letting the next fill copy that existing result. Keep this
        // lookup below all cheap structural/relation-cache gates so ordinary
        // commits never pay dynamic-mask key construction merely to decline.
        if super::dynamic_mask::dynamic_mask_state_has_cached_result(self) {
            return false;
        }

        #[inline(always)]
        fn equal_under_mask(
            left: &crate::ds::bitset::BitSet,
            right: &crate::ds::bitset::BitSet,
            mask: &crate::ds::bitset::BitSet,
        ) -> bool {
            left.words()
                .iter()
                .zip(right.words())
                .zip(mask.words())
                .all(|((&left, &right), &mask)| ((left ^ right) & mask) == 0)
        }

        #[inline(always)]
        fn any_under_mask(
            value: &crate::ds::bitset::BitSet,
            mask: &crate::ds::bitset::BitSet,
        ) -> bool {
            value
                .words()
                .iter()
                .zip(mask.words())
                .any(|(&value, &mask)| value & mask != 0)
        }

        let compare = |left: u32, right: u32| -> Option<(bool, bool)> {
            const DEAD: u32 = u32::MAX;
            if left == DEAD || right == DEAD {
                let live = |state: u32| {
                    state != DEAD
                        && (state == initial
                            || any_under_mask(
                                tokenizer.matched_terminal_bitset(state),
                                &admitted,
                            )
                            || any_under_mask(
                                tokenizer.possible_future_terminals(state),
                                &admitted,
                            ))
                };
                let left_live = live(left);
                let right_live = live(right);
                return Some((left_live == right_live, left_live && right_live));
            }
            if tokenizer.state_has_epsilon_transitions(left)
                || tokenizer.state_has_epsilon_transitions(right)
                || (left == initial) != (right == initial)
            {
                return None;
            }
            let matched_equal = equal_under_mask(
                tokenizer.matched_terminal_bitset(left),
                tokenizer.matched_terminal_bitset(right),
                &admitted,
            );
            let futures_equal = equal_under_mask(
                tokenizer.possible_future_terminals(left),
                tokenizer.possible_future_terminals(right),
                &admitted,
            );
            if !matched_equal || !futures_equal {
                return Some((false, true));
            }
            let live = left == initial
                || any_under_mask(tokenizer.matched_terminal_bitset(left), &admitted)
                || any_under_mask(tokenizer.possible_future_terminals(left), &admitted);
            Some((true, live))
        };

        let Some((root_equal, root_live)) = compare(*left_source, *right_source) else {
            return false;
        };
        if !root_equal {
            return false;
        }
        if !root_live || left_source == right_source {
            return true;
        }

        let vocab = self.constraint.dynamic_mask_vocab_for_runtime();
        let trie = vocab.trie.as_ref();
        let max_byte_steps = std::env::var(
            "GLRMASK_DYNAMIC_MASK_PARSER_RELATIVE_COMMIT_REUSE_MAX_STEPS",
        )
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        // Speculative commit-time proof work must stay well below the cost of
        // an ordinary full vocabulary mask. Successful broad-interior
        // equivalences converge rapidly because most first-byte branches either
        // die under the parser observation or enter the same lexer state. A
        // fixed transition-work budget keeps adversarial/non-equivalent pairs
        // from degenerating into a second full vocabulary traversal.
        .unwrap_or(2_048);
        let started = std::env::var_os("GLRMASK_PROFILE_DYNAMIC_MASK_COMMIT_REUSE")
            .is_some()
            .then(std::time::Instant::now);
        let mut stack = vec![(0u32, *left_source, *right_source)];
        let mut visited_nodes = 0usize;
        let mut byte_steps = 0usize;
        let mut converged_subtrees = 0usize;
        let mut dead_subtrees = 0usize;
        // Reuse the existing byte-budget branch as the one-shot checkpoint for
        // the stronger terminal-observation quotient. Grammars without a
        // prepared quotient retain the historical 2,048-step budget exactly.
        // With a prepared quotient, pause once at 256 steps; if the quotient
        // cannot certify equality, restore the historical budget and continue.
        const TERMINAL_OBSERVATION_QUOTIENT_CHECKPOINT: usize = 256;
        let mut byte_step_budget = if max_byte_steps > TERMINAL_OBSERVATION_QUOTIENT_CHECKPOINT
            && vocab.shares_terminal_observation_class(*left_source, *right_source)
        {
            TERMINAL_OBSERVATION_QUOTIENT_CHECKPOINT
        } else {
            max_byte_steps
        };

        while let Some((node, left_state, right_state)) = stack.pop() {
            visited_nodes += 1;
            for edge in trie.children(node) {
                let mut left = left_state;
                let mut right = right_state;
                let mut subtree_done = false;
                for &byte in trie.edge_bytes(edge) {
                    if left == right {
                        converged_subtrees += 1;
                        subtree_done = true;
                        break;
                    }
                    byte_steps += 1;
                    if byte_steps > byte_step_budget
                        || tokenizer.state_has_epsilon_transitions(left)
                        || tokenizer.state_has_epsilon_transitions(right)
                    {
                        if byte_steps > byte_step_budget && byte_step_budget < max_byte_steps {
                            byte_step_budget = max_byte_steps;
                            if try_terminal_observation_quotient_commit_reuse(
                                &mut self.buffers,
                                self.generation,
                                vocab,
                                tokenizer,
                                *left_source,
                                *right_source,
                                &admitted,
                                cache_left,
                                cache_right,
                                visited_nodes,
                                byte_steps,
                                started,
                            ) {
                                return true;
                            }
                        }
                        if byte_steps > byte_step_budget
                            || tokenizer.state_has_epsilon_transitions(left)
                            || tokenizer.state_has_epsilon_transitions(right)
                        {
                            if let Some(started) = started {
                                eprintln!(
                                    "[glrmask/profile][parser_relative_commit_reuse] generation={} result=decline reason={} nodes={} bytes={} converged={} dead={} elapsed_us={:.1}",
                                    self.generation,
                                    if byte_steps > byte_step_budget { "budget" } else { "epsilon" },
                                    visited_nodes,
                                    byte_steps,
                                    converged_subtrees,
                                    dead_subtrees,
                                    started.elapsed().as_secs_f64() * 1e6,
                                );
                            }
                            return false;
                        }
                    }
                    let left_next = tokenizer.get_transition(left, byte);
                    let right_next = tokenizer.get_transition(right, byte);
                    let Some((equal, live)) = compare(left_next, right_next) else {
                        if let Some(started) = started {
                            eprintln!(
                                "[glrmask/profile][parser_relative_commit_reuse] generation={} result=decline reason=epsilon_target nodes={} bytes={} converged={} dead={} elapsed_us={:.1}",
                                self.generation,
                                visited_nodes,
                                byte_steps,
                                converged_subtrees,
                                dead_subtrees,
                                started.elapsed().as_secs_f64() * 1e6,
                            );
                        }
                        return false;
                    };
                    if !equal {
                        if let Some(started) = started {
                            eprintln!(
                                "[glrmask/profile][parser_relative_commit_reuse] generation={} result=different nodes={} bytes={} converged={} dead={} elapsed_us={:.1}",
                                self.generation,
                                visited_nodes,
                                byte_steps,
                                converged_subtrees,
                                dead_subtrees,
                                started.elapsed().as_secs_f64() * 1e6,
                            );
                        }
                        return false;
                    }
                    if !live {
                        dead_subtrees += 1;
                        subtree_done = true;
                        break;
                    }
                    left = left_next;
                    right = right_next;
                }
                if subtree_done {
                    continue;
                }
                if left == right {
                    converged_subtrees += 1;
                    continue;
                }
                stack.push((edge.child, left, right));
            }
        }

        if let Some(started) = started {
            eprintln!(
                "[glrmask/profile][parser_relative_commit_reuse] generation={} result=equal left={} right={} admitted={} nodes={} bytes={} converged={} dead={} elapsed_us={:.1}",
                self.generation,
                left_source,
                right_source,
                admitted.count_ones(),
                visited_nodes,
                byte_steps,
                converged_subtrees,
                dead_subtrees,
                started.elapsed().as_secs_f64() * 1e6,
            );
        }
        const CACHE_CAPACITY: usize = 8;
        if self.buffers.parser_relative_mask_eq_cache.len() == CACHE_CAPACITY {
            self.buffers.parser_relative_mask_eq_cache.remove(0);
        }
        self.buffers
            .parser_relative_mask_eq_cache
            .push(ParserRelativeMaskEqCacheEntry {
                left: cache_left,
                right: cache_right,
                admitted,
            });
        true
    }

    /// Commit a sampled token, advancing the constraint state.
    ///
    /// `token_id` must either exist in the vocabulary the constraint was built
    /// with or be declared by a special-token terminal in the grammar.
    /// Committing a token that is grammatically invalid (not in the current
    /// mask) drives the constraint into a fail state â€” this is normal and
    /// observable via an all-zero mask.
    ///
    /// # Errors
    ///
    /// Returns an error if `token_id` is neither present in the vocabulary nor
    /// declared by a special-token terminal.
    pub fn commit_token(
        &mut self,
        token_id: u32,
    ) -> crate::Result<()> {
        self.commit_token_raw(token_id).map_err(crate::Error::State)
    }

    /// Return Some when root termination policy handles this commitment.
    fn commit_end_token(&mut self, token_id: u32) -> Option<Result<(), String>> {
        if self.terminated {
            return Some(Err("sequence has already terminated".to_owned()));
        }
        if self.constraint.end_tokens.binary_search(&token_id).is_err() {
            return None;
        }
        if self.is_accepting() {
            self.terminated = true;
        } else {
            // Like a known but grammatically invalid token, early EOS rejects
            // the sequence instead of being consumed through its byte spelling.
            self.state = Default::default();
        }
        self.generation = self.generation.wrapping_add(1);
        *self.mask_cache.lock().unwrap() = None;
        Some(Ok(()))
    }

    pub(crate) fn commit_token_raw(
        &mut self,
        token_id: u32,
    ) -> Result<(), String> {
        if let Some(result) = self.commit_end_token(token_id) { return result; }
        let constraint = self.constraint;
        let bytes = token_bytes_for_id(constraint, token_id);
        if !self.knows_token_id(token_id) {
            return Err(format!(
                "commit_token: token_id {token_id} not in vocabulary or special-token terminals"
            ));
        }
        let assertion_flags = commit_assertion_flags();
        let was_in_mask = snapshot_mask_membership(self, token_id, assertion_flags);
        let equivalence_reference = (assertion_flags & COMMIT_ASSERT_FAST_PATH_EQUIVALENCE != 0)
            .then(|| self.state.clone());
        let mask_state_before_commit = self.snapshot_current_mask_state();
        let result = commit_token_impl(constraint, &mut self.state, &mut self.buffers, token_id);
        self.finish_commit_generation(mask_state_before_commit, result.is_ok());
        assert_commit_oracles(
            constraint,
            token_id,
            bytes,
            was_in_mask,
            equivalence_reference,
            &self.state,
            result.is_ok(),
        );
        result
    }

    pub(crate) fn commit_token_dynamic(&mut self, token_id: u32) -> Result<(), String> {
        if let Some(result) = self.commit_end_token(token_id) { return result; }
        let constraint = self.constraint;
        let bytes = token_bytes_for_id(constraint, token_id);
        if !self.knows_token_id(token_id) {
            return Err(format!(
                "commit_token: token_id {token_id} not in vocabulary or special-token terminals"
            ));
        }
        let assertion_flags = commit_assertion_flags();
        let was_in_mask = if assertion_flags & COMMIT_ASSERT_MASK_EQUIVALENCE != 0 {
            let mut mask = vec![0u32; constraint.mask_len()];
            self.fill_mask_dynamic(&mut mask);
            Some(token_in_mask(&mask, token_id))
        } else {
            None
        };
        let mask_state_before_commit = self.snapshot_current_mask_state();
        let result = commit_token_impl(constraint, &mut self.state, &mut self.buffers, token_id);
        self.finish_commit_generation(mask_state_before_commit, result.is_ok());
        assert_commit_oracles(
            constraint,
            token_id,
            bytes,
            was_in_mask,
            None,
            &self.state,
            result.is_ok(),
        );
        result
    }

    pub(crate) fn commit_token_timed_ns(&mut self, token_id: u32) -> Result<u64, String> {
        if self.terminated || !self.constraint.end_tokens.is_empty() {
            let policy_started = std::time::Instant::now();
            if let Some(result) = self.commit_end_token(token_id) {
                return result.map(|()| policy_started.elapsed().as_nanos() as u64);
            }
        }
        use std::time::Instant;

        let constraint = self.constraint;
        let bytes = token_bytes_for_id(constraint, token_id);
        let assertion_flags = commit_assertion_flags();
        let was_in_mask = snapshot_mask_membership(self, token_id, assertion_flags);
        let equivalence_reference = (assertion_flags & COMMIT_ASSERT_FAST_PATH_EQUIVALENCE != 0)
            .then(|| self.state.clone());
        let mask_state_before_commit = self.snapshot_current_mask_state();
        let start = Instant::now();
        let result = commit_token_impl(constraint, &mut self.state, &mut self.buffers, token_id);
        self.finish_commit_generation(mask_state_before_commit, result.is_ok());
        // Cache-preservation proofs are part of commit's runtime cost. Keep
        // them inside the timed interval so TBM/commit reporting cannot make a
        // mask optimization look free merely by shifting work after the timer.
        let total_ns = start.elapsed().as_nanos() as u64;
        assert_commit_oracles(
            constraint,
            token_id,
            bytes,
            was_in_mask,
            equivalence_reference,
            &self.state,
            result.is_ok(),
        );
        result.map(|()| total_ns)
    }

    pub(crate) fn commit_token_profiled(&mut self, token_id: u32) -> Result<CommitProfile, String> {
        if let Some(result) = self.commit_end_token(token_id) {
            return result.map(|()| CommitProfile::default());
        }
        let constraint = self.constraint;
        let bytes = token_bytes_for_id(constraint, token_id);
        let has_special = constraint.has_special_token_id(token_id);
        if bytes.is_none() && !has_special {
            return Err(format!(
                "commit_token: token_id {token_id} not in vocabulary or special-token terminals"
            ));
        }
        let assertion_flags = commit_assertion_flags();
        let was_in_mask = snapshot_mask_membership(self, token_id, assertion_flags);
        let equivalence_reference = (assertion_flags & COMMIT_ASSERT_FAST_PATH_EQUIVALENCE != 0)
            .then(|| self.state.clone());
        let mask_state_before_commit = self.snapshot_current_mask_state();
        let total_started_at = std::time::Instant::now();
        expand_runtime_product_states(constraint, &mut self.state);
        let special = if has_special {
            advance_special_token_paths_profiled(constraint, &self.state, token_id, None)
        } else {
            SpecialTokenAdvanceProfile::default()
        };
        let mut profile = if let Some(bytes) = bytes {
            match commit_bytes_impl_profiled(
                constraint,
                &mut self.state,
                bytes,
                &mut self.buffers,
                None,
                true,
            ) {
                Ok(profile) => profile,
                Err(_) => {
                    self.state.clear();
                    self.buffers.reset_all();
                    CommitProfile::default()
                }
            }
        } else {
            self.state.clear();
            CommitProfile::default()
        };
        apply_special_token_advance_profile(&mut profile, &special);
        let special_merge_result =
            merge_special_token_paths(constraint, &mut self.state, special.paths);
        coalesce_uniform_runtime_source_states(constraint, &mut self.state);
        profile.total_ns = total_started_at.elapsed().as_nanos() as u64;
        let result = special_merge_result
            .and_then(|()| finish_token_commit(&self.state))
            .map(|()| profile);
        self.finish_commit_generation(mask_state_before_commit, result.is_ok());
        assert_commit_oracles(
            constraint,
            token_id,
            bytes,
            was_in_mask,
            equivalence_reference,
            &self.state,
            result.is_ok(),
        );
        result
    }

    pub(crate) fn commit_token_per_advance(
        &mut self,
        token_id: u32,
    ) -> Result<(Vec<PerAdvanceEntry>, Vec<(u32, Vec<Vec<u32>>)>, CommitProfile), String> {
        if let Some(result) = self.commit_end_token(token_id) {
            return result.map(|()| (Vec::new(), final_stacks(&self.state), CommitProfile::default()));
        }
        let constraint = self.constraint;
        let bytes = token_bytes_for_id(constraint, token_id);
        let has_special = constraint.has_special_token_id(token_id);
        if bytes.is_none() && !has_special {
            return Err(format!(
                "commit_token: token_id {token_id} not in vocabulary or special-token terminals"
            ));
        }
        let assertion_flags = commit_assertion_flags();
        let was_in_mask = snapshot_mask_membership(self, token_id, assertion_flags);
        let equivalence_reference = (assertion_flags & COMMIT_ASSERT_FAST_PATH_EQUIVALENCE != 0)
            .then(|| self.state.clone());
        let mask_state_before_commit = self.snapshot_current_mask_state();
        let total_started_at = std::time::Instant::now();
        let mut advances = Vec::new();
        expand_runtime_product_states(constraint, &mut self.state);
        let special = if has_special {
            advance_special_token_paths_profiled(
                constraint,
                &self.state,
                token_id,
                Some(&mut advances),
            )
        } else {
            SpecialTokenAdvanceProfile::default()
        };
        let mut profile = if let Some(bytes) = bytes {
            match commit_bytes_impl_profiled(
                constraint,
                &mut self.state,
                bytes,
                &mut self.buffers,
                Some(&mut advances),
                profile_allow_fast_paths(),
            ) {
                Ok(profile) => profile,
                Err(_) => {
                    self.state.clear();
                    self.buffers.clear_all();
                    advances.clear();
                    CommitProfile::default()
                }
            }
        } else {
            self.state.clear();
            CommitProfile::default()
        };
        apply_special_token_advance_profile(&mut profile, &special);
        let special_merge_result =
            merge_special_token_paths(constraint, &mut self.state, special.paths);
        coalesce_uniform_runtime_source_states(constraint, &mut self.state);
        profile.total_ns = total_started_at.elapsed().as_nanos() as u64;
        let result = special_merge_result
            .and_then(|()| finish_token_commit(&self.state))
            .map(|()| (advances, final_stacks(&self.state), profile));
        self.finish_commit_generation(mask_state_before_commit, result.is_ok());
        assert_commit_oracles(
            constraint,
            token_id,
            bytes,
            was_in_mask,
            equivalence_reference,
            &self.state,
            result.is_ok(),
        );
        result
    }

    /// Advance the state by raw bytes.
    pub fn commit_bytes(&mut self, bytes: &[u8]) -> crate::Result<()> {
        self.commit_bytes_raw(bytes).map_err(crate::Error::State)
    }

    pub(crate) fn commit_bytes_raw(&mut self, bytes: &[u8]) -> Result<(), String> {
        if self.terminated {
            return Err("sequence has already terminated".to_owned());
        }
        let mask_state_before_commit = self.snapshot_current_mask_state();
        let result = commit_bytes_impl(self.constraint, &mut self.state, bytes, &mut self.buffers);
        let result = clear_state_on_commit_error(&mut self.state, result);
        self.finish_commit_generation(mask_state_before_commit, result.is_ok());
        result
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Constraint as Constraint, DynamicConstraint, Grammar, Vocab};
    use std::collections::BTreeSet;

    type CanonicalCommitState =
        Vec<(u32, Vec<(Vec<u32>, Vec<(u32, Vec<u32>)>)>)>;

    #[test]
    fn unique_actionable_top_matches_isolation_reference_including_empty_paths() {
        fn reference(constraint: &Constraint, gss: &ParserGSS, terminal: u32) -> Option<ParserGSS> {
            if constraint.parser_has_controls() || template_advance_selected(constraint) {
                return None;
            }
            let mut selected = None;
            for top in gss.peek_values() {
                let Some(action) = constraint.table.action(top, terminal) else { continue; };
                if selected.is_some() { return None; }
                selected = Some((top, action));
            }
            let (top, action) = selected?;
            let isolated = gss.isolate(Some(top));
            (!isolated.is_empty())
                .then(|| apply_single_top_action_fast(constraint, &isolated, top, terminal, action))
                .flatten()
        }
        let vocab = Vocab::new(["a", "b", "ab", "ba", ",", " ", "(", ")"]
            .into_iter().enumerate().map(|(i, token)| (i as u32, token.as_bytes().to_vec())).collect());
        let grammars = [
            r#"start start; t A ::= "a" | "ab"; t B ::= "a" | "ba";
                nt start ::= A B? | B A?;"#,
            r#"start start; ignore WS; t WS ::= " "+; t A ::= "a"+; t B ::= "a"+ "b"?;
                nt item ::= A | B; nt start ::= item ("," item)*;"#,
            r#"start start; t A ::= "a"; nt start ::= A? | "(" start ")";"#,
        ];
        let empty = ParserGSS::from_single_stack(Vec::new(), TerminalsDisallowed::new());
        let mut exclusive = 0usize;
        let mut with_empty = 0usize;
        let mut comparisons = 0usize;
        for source in grammars {
            let built = Constraint::compile(Grammar::glrm(source), &vocab).unwrap();
            let loaded = Constraint::load(built.save()).unwrap();
            for constraint in [&built, &loaded] {
                let mut frontier = vec![constraint.start()];
                let mut seen = BTreeSet::new();
                for depth in 0..=3 {
                    let mut next = Vec::new();
                    for state in frontier {
                        if !seen.insert(canonical_commit_state(&state.state)) { continue; }
                        for gss in state.state.values() {
                            for input in [gss.clone(), gss.merge(&empty)] {
                                if let Some(top) = input.single_exclusive_top_value() {
                                    exclusive += 1;
                                    assert!(input.ptr_eq(&input.isolate(Some(top))));
                                } else {
                                    with_empty += 1;
                                }
                                for terminal in 0..constraint.parser_terminal_count() {
                                    let expected = reference(constraint, &input, terminal);
                                    let actual = try_advance_unique_actionable_top_fast(constraint, &input, terminal);
                                    assert_eq!(actual.as_ref().map(canonical_gss), expected.as_ref().map(canonical_gss),
                                        "terminal={terminal} depth={depth} source={source}");
                                    comparisons += 1;
                                }
                            }
                        }
                        if depth < 3 {
                            let mask = state.mask();
                            for (token, _) in constraint.token_bytes_iter() {
                                if !token_in_mask(&mask, token) { continue; }
                                let mut advanced = state.clone();
                                advanced.commit_token(token).unwrap();
                                next.push(advanced);
                            }
                        }
                    }
                    frontier = next;
                }
            }
        }
        assert!(exclusive > 0 && with_empty > 0 && comparisons > 20);
    }

    #[test]
    fn reduce_chain_output_factoring_preserves_stacks_and_exclusions() {
        let plain = TerminalsDisallowed::new();
        let guarded = plain.clone().with_insert(7, 3);
        for depth in [1usize, 2, 14, 64, 256] {
            let prefix: Vec<u32> = (0..depth as u32).collect();
            let extend = |suffix: &[u32]| {
                let mut stack = prefix.clone();
                stack.extend_from_slice(suffix);
                stack
            };
            let cases = vec![
                vec![],
                vec![(prefix.clone(), plain.clone())],
                vec![(prefix.clone(), plain.clone()), (prefix.clone(), plain.clone())],
                vec![(extend(&[500]), plain.clone()), (extend(&[501]), plain.clone())],
                vec![(extend(&[500, 502]), plain.clone()), (extend(&[501, 502]), plain.clone())],
                vec![(prefix.clone(), plain.clone()), (extend(&[501]), plain.clone())],
                vec![(vec![], plain.clone()), (prefix.clone(), plain.clone())],
                vec![(prefix.clone(), plain.clone()), (prefix.clone(), guarded.clone())],
                vec![(extend(&[500]), guarded.clone()), (extend(&[501]), plain.clone())],
            ];
            for outputs in cases {
                let reference = ParserGSS::from_stacks(&outputs);
                let actual = materialize_reduce_chain_outputs(outputs);
                assert!(actual.semantically_eq(&reference, 32).unwrap(), "depth={depth}");
            }
        }
    }

    #[test]
    fn lexer_only_actionable_cache_keeps_distinct_lexer_lanes() {
        let vocab = Vocab::new(vec![
            (0, b"ab".to_vec()), (1, b"cd".to_vec()),
            (2, b"abcd".to_vec()), (3, b"abx".to_vec()),
            (4, b"zabc".to_vec()),
        ]);
        let compiled = Constraint::compile(Grammar::glrm(r#"
            start document;
            lexer group first ::= A, C;
            lexer group second ::= B;
            t A ::= "abx";
            t C ::= "abc";
            t B ::= "abcd";
            nt document ::= A | B | "z" C;
        "#), &vocab).unwrap();
        for constraint in [&compiled, &Constraint::load(compiled.save()).unwrap()] {
            let mut state = constraint.start();
            state.commit_bytes(b"ab").unwrap();
            let mut reference = state.state.clone();
            assert!(commit_token_no_fast_path_reference(constraint, &mut reference, 1).is_ok());
            let mut actual = state.state.clone();
            let mut scratch = CommitBuffers::default();
            assert!(commit_token_impl(constraint, &mut actual, &mut scratch, 1).is_ok(),
                "lexer lanes sharing parser stacks must not share actionable-match answers");
            assert_commit_fast_path_equivalence(constraint, state.state.clone(), 1, &actual, true);
        }
    }

    #[test]
    fn unfinished_ignore_prefixes_survive_all_commit_paths() {
        let fragments: &[&[u8]] = &[
            b"a", b"+", b"b", b"//", b"/////", b"line", b"\n",
            b"/*", b"/***", b"body", b"*/", b" /*unfinished", b"//line\n+b",
        ];
        let vocab = Vocab::new(fragments.iter().enumerate()
            .map(|(id, bytes)| (id as u32, bytes.to_vec())).collect());
        for partitions in ["", "lexer group words ::= WORD; lexer group trivia ::= WS;"] {
            let source = format!(r#"
                start document;
                {partitions}
                ignore WS;
                t WS ::= " "+ | "//" [^\n]* "\n" | "/*" ([^*] | "*" [^/])* "*/";
                t WORD ::= /[a-z]+/;
                nt document ::= WORD ("+" WORD)*;
            "#);
            let built = Constraint::compile(Grammar::glrm(&source), &vocab).unwrap();
            let loaded = Constraint::load(built.save()).unwrap();
            for constraint in [&built, &loaded] {
                for ids in [
                    vec![0,4,5,6,1,2],
                    vec![0,7,9,10,1,2],
                    vec![0,8,9,10,1,2],
                    vec![0,11,10,1,2],
                    vec![0,12],
                ] {
                    let mut actual = constraint.start();
                    let mut bytewise = constraint.start();
                    for token in ids {
                        let before = actual.state.clone();
                        assert!(token_in_mask(&actual.mask(), token),
                            "ignore prefix absent from mask partitions={partitions} token={token}");
                        let mut reference = before.clone();
                        assert!(commit_token_no_fast_path_reference(constraint, &mut reference, token).is_ok(),
                            "general queue must retain unfinished ignore token={token}");
                        actual.commit_token(token).unwrap();
                        assert_commit_fast_path_equivalence(constraint, before, token, &actual.state, true);
                        for byte in fragments[token as usize] { bytewise.commit_bytes(&[*byte]).unwrap(); }
                        assert_eq!(actual.mask(), bytewise.mask(), "token-vs-byte prefix token={token}");
                    }
                    assert!(actual.is_accepting());
                    assert!(bytewise.is_accepting());
                }
            }
        }
    }

    #[test]
    fn recursive_batched_admission_matches_scalar_provider_reference() {
        use crate::ds::bitset::BitSet;
        let vocab = Vocab::new(b"[ab]! ".iter().enumerate()
            .map(|(id,byte)|(id as u32,vec![*byte])).collect());
        for source in [
            r#"start child; t WORD ::= /[ab]+/; nt child ::= WORD;"#,
            r#"start child; ignore WS; t WS ::= " "+; t WORD ::= /[ab]+/; nt child ::= WORD?;"#,
            r#"start child; t A ::= "a"; t AB ::= "ab"; nt child ::= (A | AB)*;"#,
        ] {
            let child = Constraint::compile(Grammar::glrm(source), &vocab).unwrap();
            let middle = Constraint::compile(Grammar::glrm(
                r#"start middle; extern grammar child; nt middle ::= "[" child "]";"#,
            ), &vocab).unwrap().bind_grammar_dynamic_boundary("child", child).unwrap();
            let outer = Constraint::compile(Grammar::glrm(
                r#"start outer; extern grammar middle; nt outer ::= middle "!";"#,
            ), &vocab).unwrap().bind_grammar_dynamic_boundary("middle", middle).unwrap();
            let loaded = Constraint::load(outer.save()).unwrap();
            for constraint in [&outer, &loaded] {
                for prefix in ["", "[", "[a", "[ab", "[ab]", "[ab]!", "[ ", "[]"] {
                    let mut state = constraint.start();
                    if state.commit_bytes(prefix.as_bytes()).is_err() {continue;}
                    let runtime = state.constraint;
                    let count = runtime_terminal_count(runtime);
                    for (_, gss) in state.state.iter() {
                        for stride in [1,2,3] {
                            let mut candidates=BitSet::new(count);
                            for terminal in (0..count).step_by(stride) {candidates.set(terminal);}
                            let actual=runtime.compact_segmented_parser_admitted_terminals(gss,&candidates).unwrap();
                            let mut expected=BitSet::new(count);
                            for terminal in candidates.iter_ones() {
                                if runtime.compact_segmented_parser_may_advance_on(gss,terminal as u32).unwrap() {
                                    expected.set(terminal);
                                }
                            }
                            assert_eq!(actual,expected,"prefix={prefix} stride={stride} source={source}");
                            assert_eq!(runtime.compact_segmented_parser_may_advance_on_any(gss,&candidates),Some(!expected.is_empty()));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn unchanged_runtime_state_preserves_fill_mask_cache() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"aa".to_vec()),
            (2, b"b".to_vec()),
        ]);
        let constraint = Constraint::compile(
            Grammar::glrm(
                r#"glrm 1;
                start start;
                t A = /a+/;
                t B = "b";
                nt start = A B;
                "#,
            ),
            &vocab,
        )
        .unwrap();

        let mut state = constraint.start();
        state.commit_token(0).unwrap();

        // Populate the ordinary per-sequence mask cache while inside A. A
        // further `a` is an exact lexer self-loop: no parser terminal is
        // consumed and the complete runtime state remains identical.
        let cached_mask = state.mask();
        let before_generation = state.generation;
        let before_state = state.state.clone();
        {
            let cache = state.mask_cache.lock().unwrap();
            assert_eq!(cache.as_ref().unwrap().generation, before_generation);
        }

        state.commit_token(0).unwrap();
        assert_eq!(state.generation, before_generation + 1);
        assert_eq!(before_state, state.state);
        {
            let cache = state.mask_cache.lock().unwrap();
            assert_eq!(cache.as_ref().unwrap().generation, state.generation);
            assert_eq!(cache.as_ref().unwrap().mask, cached_mask);
        }
        assert_eq!(state.mask(), cached_mask);

        // Completing A and advancing the parser must invalidate the cached
        // generation. The following mask is therefore recomputed for B.
        state.commit_token(2).unwrap();
        {
            let cache = state.mask_cache.lock().unwrap();
            assert_ne!(cache.as_ref().unwrap().generation, state.generation);
        }
    }

    #[test]
    fn unchanged_dynamic_mask_projection_preserves_fill_mask_cache() {
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

        // Enter the quoted lazy bounded-string residual, consume one interior
        // byte, then cache its mask. Far from the upper bound, another interior
        // byte changes the exact symbolic residual coordinate while the parser
        // has not seen a terminal boundary and the finite vocabulary-horizon
        // mask coordinate remains unchanged.
        state.commit_token(0).unwrap(); // opening quote
        state.commit_token(1).unwrap(); // first interior 'a'
        let cached_mask = state.mask();
        let before_generation = state.generation;
        let before_state = state.state.clone();

        state.commit_token(1).unwrap(); // second interior 'a'
        assert_ne!(before_state, state.state, "test must exercise a changed exact lexer state");
        assert!(
            state.dynamic_mask_projection_state_eq(&before_state, &state.state),
            "changed exact state should remain equal in the mask-only coordinate",
        );
        {
            let cache = state.mask_cache.lock().unwrap();
            assert_eq!(cache.as_ref().unwrap().generation, before_generation + 1);
            assert_eq!(cache.as_ref().unwrap().mask, cached_mask);
        }
        assert_eq!(state.mask(), cached_mask);
    }

    #[test]
    fn recursive_radix_candidate_admission_matches_pointwise_exact_commit() {
        let vocab = Vocab::new(vec![
            (0, b"X".to_vec()),
            (1, b"X[".to_vec()),
            (2, b"X[a".to_vec()),
            (3, b"X[a]".to_vec()),
            (4, b"X[a]!".to_vec()),
            (5, b"X[a]!".to_vec()),
            (6, b"[".to_vec()),
            (7, b"[a".to_vec()),
            (8, b"[a]".to_vec()),
            (9, b"[a]!".to_vec()),
            (10, b"a".to_vec()),
            (11, b"a]".to_vec()),
            (12, b"a]!".to_vec()),
            (13, b"]".to_vec()),
            (14, b"]!".to_vec()),
            (15, b"!".to_vec()),
        ]);
        let child = Constraint::compile(
            Grammar::glrm("glrm 1; start child; nt child = \"[\" \"a\" \"]\";"),
            &vocab,
        )
        .unwrap();
        let parent = Constraint::compile(
            Grammar::glrm(
                "glrm 1; start document; extern grammar child; nt document = \"X\" child \"!\";",
            ),
            &vocab,
        )
        .unwrap();
        let bound = parent
            .bind_grammar_dynamic_boundary("child", child)
            .unwrap();
        assert!(bound.uses_compact_segmented_parser_runtime());

        for path in [
            &[][..],
            &[0][..],
            &[0, 6][..],
            &[0, 6, 10][..],
            &[0, 6, 10, 13][..],
        ] {
            let mut state = bound.start();
            for &token in path {
                state.commit_token(token).unwrap();
            }

            let mut batch_candidates = bound.token_bytes_iter().collect::<Vec<_>>();
            let mut batch_buffers = CommitBuffers::default();
            let mut batched = Vec::new();
            admissible_byte_token_candidates_from_state_exact(
                &bound,
                &state.state,
                &mut batch_buffers,
                &mut batch_candidates,
                &mut batched,
            );
            batched.sort_unstable();

            let mut pointwise_buffers = CommitBuffers::default();
            let mut pointwise = bound
                .token_bytes_iter()
                .filter_map(|(token_id, _)| {
                    token_admissible_from_state_exact(
                        &bound,
                        &state.state,
                        &mut pointwise_buffers,
                        token_id,
                    )
                    .then_some(token_id)
                })
                .collect::<Vec<_>>();
            pointwise.sort_unstable();
            assert_eq!(batched, pointwise, "candidate mismatch after path {path:?}");
        }
    }

    #[test]
    fn recursive_scoped_tokenizer_exec_uses_only_the_active_leaf() {
        let vocab = Vocab::new(vec![
            (0, b"X".to_vec()),
            (1, b"a".to_vec()),
            (2, b"!".to_vec()),
            (3, b"Xa!".to_vec()),
        ]);
        let child = Constraint::compile(
            Grammar::glrm("glrm 1; start child; t A = \"a\"; nt child = A;"),
            &vocab,
        )
        .unwrap();
        let parent = Constraint::compile(
            Grammar::glrm(
                "glrm 1; start document; extern grammar child; t X = \"X\"; t BANG = \"!\"; nt document = X child BANG;",
            ),
            &vocab,
        )
        .unwrap();
        let local_x = parent
            .terminal_display_names
            .iter()
            .position(|name| name == "X")
            .unwrap() as u32;
        let local_a = child
            .terminal_display_names
            .iter()
            .position(|name| name == "A")
            .unwrap() as u32;
        let bound = parent.bind_grammar("child", child).unwrap();
        let loaded = Constraint::load(bound.save()).unwrap();
        for constraint in [&bound, &loaded] {
            let layout = constraint.recursive_parser_layout().unwrap().unwrap();
            assert_eq!(layout.leaves.len(), 2);
            let scoped_x = constraint.recursive_terminal_scoped_id(0, local_x).unwrap();
            let scoped_a = constraint.recursive_terminal_scoped_id(1, local_a).unwrap();
            assert!(scoped_x >= constraint.parser_terminal_count());
            assert!(scoped_a >= constraint.parser_terminal_count());

            let root_reset = constraint.recursive_tokenizer_reset_state(0).unwrap();
            let root_x = tokenizer_scan::execute_recursive_tokenizer_from_state_small(
                constraint,
                b"X",
                root_reset,
            )
            .unwrap();
            assert!(root_x.matches.iter().any(|matched| matched.id == scoped_x));
            assert!(root_x.end_state.iter().all(|&state| {
                constraint.recursive_tokenizer_leaf_state(state).unwrap().0 == 0
            }));
            let root_a = tokenizer_scan::execute_recursive_tokenizer_from_state_small(
                constraint,
                b"a",
                root_reset,
            )
            .unwrap();
            assert!(
                root_a.matches.iter().all(|matched| matched.id != scoped_a),
                "child terminal leaked from inactive leaf: {:?}",
                root_a.matches,
            );

            let child_reset = constraint.recursive_tokenizer_reset_state(1).unwrap();
            let child_a = tokenizer_scan::execute_recursive_tokenizer_from_state_small(
                constraint,
                b"a",
                child_reset,
            )
            .unwrap();
            assert!(child_a.matches.iter().any(|matched| matched.id == scoped_a));
            assert!(child_a.end_state.iter().all(|&state| {
                constraint.recursive_tokenizer_leaf_state(state).unwrap().0 == 1
            }));
        }
    }

    fn canonical_commit_state(
        state: &ParserStateMap,
    ) -> CanonicalCommitState {
        canonical_commit_state_for_equivalence_assert(state)
    }

    #[test]
    fn exact_prefilter_rejects_multiple_conditional_candidates_independent_of_row_order() {
        let mut terminals = crate::ds::bitset::BitSet::new(8);
        terminals.set(1);
        terminals.set(2);
        let unconditional = crate::ds::bitset::BitSet::new(8);

        let row_a = ActionRow::from_iter([
            (1, Action::Reduce(0, 1)),
            (2, Action::Reduce(0, 1)),
        ]);
        let row_b = ActionRow::from_iter([
            (2, Action::Reduce(0, 1)),
            (1, Action::Reduce(0, 1)),
        ]);

        assert_eq!(
            single_conditional_candidate(&row_a, &unconditional, &terminals),
            Err(()),
        );
        assert_eq!(
            single_conditional_candidate(&row_b, &unconditional, &terminals),
            Err(()),
        );

        let mut one_terminal = crate::ds::bitset::BitSet::new(8);
        one_terminal.set(2);
        assert_eq!(
            single_conditional_candidate(&row_a, &unconditional, &one_terminal),
            Ok(Some(2)),
        );
    }

    #[test]
    fn batched_and_cached_end_state_admission_match_pointwise_simulation() {
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start start;
                t A ::= "a";
                t AB ::= "ab";
                t B ::= "b";
                t C ::= "c";
                nt item ::= A | AB | B;
                nt start ::= item C;
            "#,
            &Vocab::new(vec![
                (0, b"a".to_vec()),
                (1, b"ab".to_vec()),
                (2, b"b".to_vec()),
                (3, b"c".to_vec()),
                (4, b"ac".to_vec()),
                (5, b"abc".to_vec()),
            ]),
        )
        .unwrap();
        let state = constraint.start();
        let gss = state.state.values().next().unwrap();
        let initial = constraint.runtime_commit_initial_state();
        let non_initial = (0..constraint.tokenizer.num_states())
            .find(|&end_state| {
                end_state != initial
                    && !constraint
                        .tokenizer
                        .possible_future_terminals(end_state)
                        .is_empty()
            })
            .expect("test tokenizer should have a live non-initial state");
        // Repeating one exact continuation is enough to exercise batching and
        // cache reuse; production callers usually provide many distinct states,
        // but the theorem does not depend on their distinctness.
        let end_states = vec![initial, non_initial, non_initial];

        let admitted = batched_end_state_admitted_terminals(
            &constraint,
            gss,
            &end_states,
        )
        .expect("multiple non-initial tokenizer states should batch");
        for &end_state in &end_states {
            assert_eq!(
                end_state_may_advance_with_batch(
                    &constraint,
                    gss,
                    end_state,
                    Some(&admitted),
                ),
                end_state_may_advance(&constraint, gss, end_state),
                "batched admission differs for tokenizer state {end_state}",
            );
        }

        let mut cache = SmallVec::<[ParserAdmissionCacheEntry; 8]>::new();
        let index = cached_batched_end_state_admission(
            &constraint,
            gss,
            &end_states,
            &mut cache,
        )
        .expect("multiple non-initial tokenizer states should populate cache");
        for &end_state in &end_states {
            assert_eq!(
                end_state_may_advance_from_cache_entry(
                    &constraint,
                    end_state,
                    &cache[index],
                ),
                end_state_may_advance(&constraint, gss, end_state),
                "cached admission differs for tokenizer state {end_state}",
            );
        }

        // Repeating the identical query must be a pure cache hit and retain the
        // exact pointwise facts.
        let tested = cache[index].tested.clone();
        let admitted_before = cache[index].admitted.clone();
        let repeat = cached_batched_end_state_admission(
            &constraint,
            gss,
            &end_states,
            &mut cache,
        )
        .unwrap();
        assert_eq!(repeat, index);
        assert_eq!(cache[index].tested, tested);
        assert_eq!(cache[index].admitted, admitted_before);
    }

    #[test]
    fn admission_cache_reset_drops_persistent_gss_references() {
        let gss = ParserGSS::from_single_stack(
            vec![0_u32, 1],
            TerminalsDisallowed::new(),
        );
        let mut buffers = CommitBuffers::default();
        buffers.admission_cache.push(ParserAdmissionCacheEntry {
            gss,
            tested: crate::ds::bitset::BitSet::new(2),
            admitted: crate::ds::bitset::BitSet::new(2),
            boolean_queries: SmallVec::new(),
        });
        buffers.clear_all();
        assert_eq!(
            buffers.admission_cache.len(),
            1,
            "ordinary scratch reuse should preserve admission facts",
        );
        buffers.reset_all();
        assert!(buffers.admission_cache.is_empty());
    }


    #[test]
    #[ignore]
    fn debug_selected10_completed_child_lookahead_actions() {
        use crate::compiler::glr::analysis::EOF;
        use crate::compiler::glr::parser::stack_admissible_terminals;
        use crate::compiler::glr::table::Action;
        use crate::ds::bitset::BitSet;
        use std::collections::BTreeMap;

        let path = std::env::var("GLRMASK_DEBUG_ARTIFACT").expect("GLRMASK_DEBUG_ARTIFACT");
        let bytes = std::fs::read(path).unwrap();
        let constraint = Constraint::load(&bytes).unwrap();
        let mut state = constraint.start();
        state.commit_bytes(b"const x = tools.tool_0({})").unwrap();
        eprintln!("STACKS {:?}", state.debug_parser_stacks());
        for (_, gss) in state.state.iter() {
            let all = BitSet::all(constraint.parser_terminal_count() as usize + 1);
            let admitted = stack_admissible_terminals(&constraint.table, gss, &all);
            let mut factor = gss.clone();
            let mut blockers = Vec::<u32>::new();
            for depth in 0..16 {
                let top = factor.single_exclusive_top_value();
                let admitted_now =
                    stack_admissible_terminals(&constraint.table, &factor, &all);
                eprintln!(
                    "FACTOR_DEPTH {depth} top={top:?} stack={:?} admitted={:?} blockers={:?}",
                    factor.to_stacks(64),
                    admitted_now.iter_ones().collect::<Vec<_>>(),
                    blockers,
                );
                let Some((next, blocked)) =
                    crate::compiler::glr::parser::lookahead_reduction_factor(
                        &constraint.table,
                        &factor,
                    )
                else {
                    break;
                };
                for terminal in blocked {
                    if !blockers.contains(&terminal) {
                        blockers.push(terminal);
                    }
                }
                factor = next;
            }
            for top in gss.peek_values() {
                let mut kinds = BTreeMap::<String, Vec<u32>>::new();
                for bit in admitted.iter_ones() {
                    let terminal = if bit == constraint.parser_terminal_count() as usize {
                        EOF
                    } else {
                        bit as u32
                    };
                    let desc = match constraint.table.action(top, terminal) {
                        Some(Action::Reduce(nt, len)) => format!("reduce:{nt}:{len}"),
                        Some(Action::Shift(target, replace)) => format!("shift:{target}:{replace}"),
                        Some(Action::StackShifts(shifts)) => format!("stackshifts:{}", shifts.len()),
                        Some(Action::GuardedStackShifts(shifts)) => format!("guarded:{}", shifts.len()),
                        Some(Action::ReplaceShifts(targets)) => format!("replace:{}", targets.len()),
                        Some(Action::Split { shift, reduces, accept }) => format!("split:{shift:?}:{reduces:?}:{accept}"),
                        Some(Action::Accept) => "accept".into(),
                        Some(Action::Skip) => "skip".into(),
                        None => "none".into(),
                    };
                    kinds.entry(desc).or_default().push(terminal);
                }
                eprintln!("TOP {top} ADMITTED {} KINDS {}", admitted.count_ones(), kinds.len());
                for (kind, terminals) in kinds {
                    eprintln!("  {kind} count={} sample={:?}", terminals.len(), &terminals[..terminals.len().min(24)]);
                }
            }
        }
    }


    #[test]
    #[ignore]
    fn debug_selected10_core_fast_weight_transport() {
        use crate::compiler::glr::labels::DEFAULT_LABEL;
        use crate::ds::weight::Weight;

        fn accepted_weight(constraint: &Constraint, labels: &[u32]) -> Weight {
            let mut state = constraint.parser_dwa.start_state();
            let mut weight = Weight::all();
            for &label in labels {
                let row = &constraint.parser_dwa.states()[state as usize];
                let Some((target, edge_weight)) = row
                    .transitions
                    .get(&(label as i32))
                    .or_else(|| row.transitions.get(&DEFAULT_LABEL))
                else {
                    return Weight::empty();
                };
                weight = weight.intersection(edge_weight);
                state = *target;
            }
            let Some(final_weight) = constraint.parser_dwa.states()[state as usize]
                .final_weight
                .as_ref()
            else {
                return Weight::empty();
            };
            weight.intersection(final_weight)
        }

        fn dump(name: &str, constraint: &Constraint, tokenizer_state: u32, token: u32) {
            let internal_token = constraint
                .original_token_to_internal
                .get(token as usize)
                .copied()
                .unwrap_or(u32::MAX);
            let tsids = constraint.internal_tsids_for_state(tokenizer_state).to_vec();
            eprintln!("WEIGHT_TRANSPORT {name} tokenizer_state={tokenizer_state} tsids={tsids:?} token={token} internal_token={internal_token}");
            for labels in [&[22_u32][..], &[22_u32, 0][..]] {
                let weight = accepted_weight(constraint, labels);
                let memberships = tsids
                    .iter()
                    .map(|&tsid| (tsid, internal_token != u32::MAX && weight.tokens_for_tsid(tsid).contains(internal_token), weight.tokens_for_tsid(tsid).len()))
                    .collect::<Vec<_>>();
                eprintln!("WEIGHT_TRANSPORT {name} labels={labels:?} empty={} memberships={memberships:?}", weight.is_empty());
            }
            for key in [22_i32, DEFAULT_LABEL] {
                if let Some(weight) = constraint.parser_top_accept.get(&key) {
                    let memberships = tsids.iter().map(|&tsid| (tsid, weight.tokens_for_tsid(tsid).contains(internal_token), weight.tokens_for_tsid(tsid).len())).collect::<Vec<_>>();
                    eprintln!("WEIGHT_TRANSPORT {name} top_accept key={key} memberships={memberships:?}");
                }
                if let Some(parts) = constraint.parser_top_accept_parts.get(&key) {
                    for (part, weight) in parts.iter().enumerate() {
                        let memberships = tsids.iter().map(|&tsid| (tsid, weight.tokens_for_tsid(tsid).contains(internal_token), weight.tokens_for_tsid(tsid).len())).collect::<Vec<_>>();
                        eprintln!("WEIGHT_TRANSPORT {name} top_part key={key} part={part} memberships={memberships:?}");
                    }
                }
            }
            let mut l1 = Vec::new();
            constraint.for_each_direct_regular_l1_acceptance(22, |weight| {
                l1.push(tsids.iter().map(|&tsid| { let tokens = weight.token_set_for_tsid(tsid).map(|tokens| tokens.to_range_set()).unwrap_or_default(); (tsid, tokens.contains(internal_token), tokens.len()) }).collect::<Vec<_>>());
            });
            eprintln!("WEIGHT_TRANSPORT {name} l1={l1:?}");
        }

        let core_path = std::env::var("GLRMASK_DEBUG_CORE_ARTIFACT").unwrap();
        let fast_path = std::env::var("GLRMASK_DEBUG_ARTIFACT").unwrap();
        let core = Constraint::load(&std::fs::read(core_path).unwrap()).unwrap();
        let fast = Constraint::load(&std::fs::read(fast_path).unwrap()).unwrap();
        dump("core", &core, 38, 443);
        dump("fast", &fast, 39, 443);
    }

    #[test]
    #[ignore]
    fn debug_selected10_boundary_shallow_acceptance() {
        use crate::compiler::glr::labels::DEFAULT_LABEL;
        use crate::ds::weight::Weight;

        fn accepted_weight(constraint: &Constraint, labels: &[u32]) -> Weight {
            let mut state = constraint.parser_dwa.start_state();
            let mut weight = Weight::all();
            for &label in labels {
                let row = &constraint.parser_dwa.states()[state as usize];
                let Some((target, edge_weight)) = row
                    .transitions
                    .get(&(label as i32))
                    .or_else(|| row.transitions.get(&DEFAULT_LABEL))
                else { return Weight::empty(); };
                weight = weight.intersection(edge_weight);
                state = *target;
            }
            let Some(final_weight) = constraint.parser_dwa.states()[state as usize].final_weight.as_ref()
            else { return Weight::empty(); };
            weight.intersection(final_weight)
        }

        fn contains(constraint: &Constraint, tokenizer_state: u32, token: u32, weight: &Weight) -> bool {
            let internal = constraint.original_token_to_internal[token as usize];
            constraint.internal_tsids_for_state(tokenizer_state)
                .iter()
                .any(|&tsid| weight.tokens_for_tsid(tsid).contains(internal))
        }

        fn dump_case(constraint: &Constraint, name: &str, tokenizer_state: u32, stack: &[u32], tokens: &[u32]) {
            let reversed = stack.iter().rev().copied().collect::<Vec<_>>();
            eprintln!("SHALLOW {name} tokenizer_state={tokenizer_state} stack={stack:?} tsids={:?}", constraint.internal_tsids_for_state(tokenizer_state));
            let start_final = constraint.parser_dwa.states()[constraint.parser_dwa.start_state() as usize]
                .final_weight.clone().unwrap_or_else(Weight::empty);
            for &token in tokens {
                let mut first = contains(constraint, tokenizer_state, token, &start_final).then_some(0usize);
                for depth in 1..=reversed.len() {
                    let w = accepted_weight(constraint, &reversed[..depth]);
                    if contains(constraint, tokenizer_state, token, &w) {
                        first = Some(depth);
                        break;
                    }
                }
                eprintln!("SHALLOW {name} token={token} first_depth={first:?}");
            }
        }

        let full_path = std::env::var("GLRMASK_DEBUG_FULL_ARTIFACT").unwrap();
        let full = Constraint::load(&std::fs::read(full_path).unwrap()).unwrap();
        dump_case(&full, "tools_reset_a", 0, &[0,22,198,18], &[739,2446,7255,21966]);
        dump_case(&full, "tools_reset_b", 0, &[0,22,198,167], &[739,2446,7255,21966]);
        dump_case(&full, "tools_residual", 444, &[0,22,198], &[739,2446,7255,21966]);
        dump_case(&full, "tool0", 0, &[0,22,198,193,862], &[49209,69906]);
        dump_case(&full, "inside", 0, &[0,22,198,193,873,903], &[3033,3602,4649,9000,14419,16638,17041,29448,31893,32988,35183,36199,39942,44160,53511,71741,79237,82274,95445]);
        dump_case(&full, "await_tools", 0, &[0,22,198,109,193], &[3324,32809]);
    }

    #[test]
    #[ignore]
    fn debug_selected10_iterated_mask_factor() {
        use std::time::Instant;

        let fast_path = std::env::var("GLRMASK_DEBUG_ARTIFACT").unwrap();
        let full_path = std::env::var("GLRMASK_DEBUG_FULL_ARTIFACT").unwrap();
        let fast = Constraint::load(&std::fs::read(fast_path).unwrap()).unwrap();
        let full = Constraint::load(&std::fs::read(full_path).unwrap()).unwrap();
        unsafe {
            std::env::set_var("GLRMASK_EXPERIMENT_MASK_LOOKAHEAD_FACTOR", "1");
            std::env::set_var("GLRMASK_EXPERIMENT_MASK_LOOKAHEAD_FACTOR_MAX_DEPTH", "2");
            std::env::set_var("GLRMASK_EXPERIMENT_SCOPED_IGNORE_EXACT_OVERLAY", "1");
            std::env::remove_var("GLRMASK_EXPERIMENT_STATIC_DYNAMIC_OVERLAY");
        }
        let prefixes: &[&[u8]] = &[
            b"",
            b"const x",
            b"const x =",
            b"const x = tools",
            b"const x = tools.tool_0",
            b"const x = tools.tool_0({",
            b"const x = tools.tool_0({})",
            b"const x = tools.tool_0({});",
            b"const r = await tools.",
        ];
        for &prefix in prefixes {
            let mut fs = full.start();
            let mut hs = fast.start();
            if !prefix.is_empty() {
                fs.commit_bytes(prefix).unwrap();
                hs.commit_bytes(prefix).unwrap();
            }
            let fm = fs.mask();
            let started = Instant::now();
            let hm = hs.mask();
            let elapsed = started.elapsed().as_micros();
            let mut extra = 0usize;
            let mut missing = 0usize;
            for (&left, &right) in hm.iter().zip(&fm) {
                extra += (left & !right).count_ones() as usize;
                missing += (right & !left).count_ones() as usize;
            }
            eprintln!(
                "IGNORE_PM prefix={:?} us={} extra={} missing={}",
                String::from_utf8_lossy(prefix), elapsed, extra, missing
            );
        }
        std::mem::forget(fast);
        std::mem::forget(full);
    }

    #[test]
    fn language_queue_structural_gate_rejects_tiny_and_accepts_wide_complex_gss() {
        let mut tiny = ParserStateMap::default();
        tiny.insert(
            0,
            ParserGSS::from_single_stack(
                vec![0_u32, 1, 2],
                TerminalsDisallowed::new(),
            ),
        );
        assert!(
            language_queue_top_value_count_at_most(&tiny, LANGUAGE_QUEUE_MIN_TOP_VALUES)
                < LANGUAGE_QUEUE_MIN_TOP_VALUES
        );

        let stacks = (0_u32..64)
            .map(|index| {
                (
                    vec![0, 1_000 + index, 2_000 + index, 10 + index % 4],
                    TerminalsDisallowed::new(),
                )
            })
            .collect::<Vec<_>>();
        let mut complex = ParserStateMap::default();
        complex.insert(0, ParserGSS::from_stacks(&stacks));
        assert_eq!(
            language_queue_top_value_count_at_most(
                &complex,
                LANGUAGE_QUEUE_MIN_TOP_VALUES,
            ),
            LANGUAGE_QUEUE_MIN_TOP_VALUES,
        );
        assert_eq!(
            language_queue_path_count_at_most(&complex, LANGUAGE_QUEUE_MIN_PATHS),
            LANGUAGE_QUEUE_MIN_PATHS,
        );
        assert_eq!(
            language_queue_node_count_at_most(&complex, LANGUAGE_QUEUE_MIN_NODES),
            LANGUAGE_QUEUE_MIN_NODES,
        );
    }

    #[test]
    fn actionable_terminal_boundary_trigger_requires_distinct_offsets() {
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start start;
                t A ::= "a";
                t AB ::= "ab";
                nt start ::= A | AB;
            "#,
            &Vocab::new(vec![
                (0, b"a".to_vec()),
                (1, b"ab".to_vec()),
                (2, b"b".to_vec()),
            ]),
        )
        .expect("boundary-trigger grammar should compile");
        let state = constraint.start();
        let mut scratch = tokenizer_scan::ReusableTokenizerExecScratch::default();

        assert!(
            !has_multiple_actionable_terminal_boundaries(
                &constraint,
                &state.state,
                b"a",
                &mut scratch,
            ),
            "one actionable completion boundary must not select the language queue",
        );
        assert!(
            has_multiple_actionable_terminal_boundaries(
                &constraint,
                &state.state,
                b"ab",
                &mut scratch,
            ),
            "actionable terminal completions at byte offsets one and two must select",
        );
    }

    #[test]
    fn actionable_terminal_boundary_trigger_ignores_same_offset_ambiguity() {
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start start;
                t A ::= "a";
                t ALSO_A ::= "a";
                nt start ::= A | ALSO_A;
            "#,
            &Vocab::new(vec![(0, b"a".to_vec())]),
        )
        .expect("same-offset trigger grammar should compile");
        let state = constraint.start();
        let mut scratch = tokenizer_scan::ReusableTokenizerExecScratch::default();

        assert!(
            !has_multiple_actionable_terminal_boundaries(
                &constraint,
                &state.state,
                b"a",
                &mut scratch,
            ),
            "multiple actionable terminals ending at one offset must remain on the ordinary queue",
        );
    }

    #[test]
    fn flat_stack_effect_can_replace_the_only_state() {
        let mut stack = FlatInlineStack::new();
        stack.push(36);

        assert_eq!(apply_flat_stack_effect(&mut stack, 1, &[37]), Some(true));
        assert_eq!(stack.as_slice(), &[37]);
    }

    #[test]
    fn compile_time_initial_commit_priming_preserves_runtime_semantics() {
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start start;
                t A ::= "a";
                t B ::= "b";
                nt start ::= A B;
            "#,
            &Vocab::new(vec![
                (0, b"a".to_vec()),
                (1, b"ab".to_vec()),
                (2, b"b".to_vec()),
                (3, b"x".to_vec()),
            ]),
        )
        .unwrap();

        let mut state = constraint.start();
        assert_eq!(
            state
                .mask()
                .iter()
                .enumerate()
                .flat_map(|(word, &bits)| {
                    (0..32).filter_map(move |bit| {
                        ((bits >> bit) & 1 == 1).then_some((word * 32 + bit) as u32)
                    })
                })
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([0, 1]),
        );
        state.commit_token(0).unwrap();
        state.commit_token(2).unwrap();
        assert!(state.is_accepting());

        let loaded = Constraint::load(&constraint.save()).unwrap();
        let mut loaded_state = loaded.start();
        loaded_state.commit_token(1).unwrap();
        assert!(loaded_state.is_accepting());
    }

    #[test]
    fn duplicate_key_flat_state_reports_completion_without_normalization() {
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start start;
                t A ::= "a";
                t B ::= "a" | "ab";
                nt item ::= A | B;
                nt start ::= item;
            "#,
            &Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]),
        )
        .unwrap();

        let mut state = constraint.start();
        state.commit_token(0).unwrap();
        assert!(
            state.state.has_duplicate_keys(),
            "the bounded runtime should retain the completed alternatives separately"
        );
        assert!(state.is_accepting());

        let mut normalized = state.clone();
        normalized.state.normalize_duplicate_keys();
        assert!(normalized.is_accepting());
        assert_eq!(
            canonical_commit_state(&state.state),
            canonical_commit_state(&normalized.state),
        );
    }

    #[test]
    fn special_token_advance_matches_normalized_duplicate_key_state() {
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start start;
                t A ::= "a";
                t B ::= "a" | "ab";
                nt item ::= A | B;
                nt start ::= item @token(100);
            "#,
            &Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]),
        )
        .unwrap();

        let mut flat = constraint.start();
        flat.commit_token(0).unwrap();
        assert!(flat.state.has_duplicate_keys());

        let mut normalized = flat.clone();
        normalized.state.normalize_duplicate_keys();
        assert_eq!(flat.mask(), normalized.mask());

        flat.commit_token(100).unwrap();
        normalized.commit_token(100).unwrap();
        assert!(flat.is_accepting());
        assert!(normalized.is_accepting());
        assert_eq!(
            canonical_commit_state(&flat.state),
            canonical_commit_state(&normalized.state),
        );
    }

    #[test]
    fn carried_virtual_stack_stops_when_stack_effect_exposes_branched_floor() {
        let acc = TerminalsDisallowed::new();
        let left = ParserGSS::from_single_stack(vec![0, 1, 10], acc.clone());
        let right = ParserGSS::from_single_stack(vec![0, 2, 10], acc);
        let merged = left.merge(&right);

        let mut exhausted = merged
            .try_virtual_stack()
            .expect("merged common top should form a virtual stack");
        assert!(!try_apply_action_to_carried_virtual_stack(
            &mut exhausted,
            &Action::StackShifts(vec![crate::compiler::glr::table::StackShift {
                pop: 1,
                pushes: Vec::new(),
            }]),
        ));
        assert_eq!(exhausted.top(), Some(&10));

        let mut restored_common_top = merged
            .try_virtual_stack()
            .expect("merged common top should form a virtual stack");
        assert!(try_apply_action_to_carried_virtual_stack(
            &mut restored_common_top,
            &Action::StackShifts(vec![crate::compiler::glr::table::StackShift {
                pop: 1,
                pushes: vec![40],
            }]),
        ));
        assert_eq!(restored_common_top.top(), Some(&40));
        let mut stacks = restored_common_top.into_gss().to_stacks(4_096).expect("stack enumeration exceeded explicit limit");
        stacks.sort_by(|left, right| left.0.cmp(&right.0));
        assert_eq!(
            stacks.into_iter().map(|(stack, _)| stack).collect::<Vec<_>>(),
            vec![vec![0, 1, 40], vec![0, 2, 40]],
        );

        let mut emptied = ParserGSS::from_single_stack(
            vec![10],
            TerminalsDisallowed::new(),
        )
        .try_virtual_stack()
        .expect("single stack should form a virtual stack");
        assert!(!try_apply_action_to_carried_virtual_stack(
            &mut emptied,
            &Action::StackShifts(vec![crate::compiler::glr::table::StackShift {
                pop: 1,
                pushes: Vec::new(),
            }]),
        ));
        assert_eq!(emptied.top(), Some(&10));
    }

    fn canonical_gss(gss: &ParserGSS) -> Vec<(Vec<u32>, Vec<(u32, Vec<u32>)>)> {
        canonical_commit_state(&ParserStateMap::singleton(0, gss.clone()))
            .pop()
            .unwrap()
            .1
    }

    #[test]
    fn fused_exact_admission_matches_legacy_two_pass_reference() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"ab".to_vec()),
        ]);
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start start;
                t A ::= "a";
                t AB ::= "ab";
                t B ::= "b";
                nt item ::= A | AB | A B;
                nt start ::= item item?;
            "#,
            &vocab,
        )
        .unwrap();
        assert_eq!(
            constraint.parser_admission_policy(),
            AdmissionPolicy::ExactSimulation,
        );

        let mut states = vec![constraint.start()];
        let mut after_a = constraint.start();
        after_a.commit_token(0).unwrap();
        states.push(after_a);

        for state in states {
            for gss in state.state.values() {
                for terminal in 0..constraint.parser_terminal_count() {
                    let legacy = if stack_may_advance_on(&constraint.table, gss, terminal) {
                        let advanced = advance_parser_stacks(&constraint, gss, terminal);
                        (!advanced.is_empty()).then_some(advanced)
                    } else {
                        None
                    };
                    let fused = advance_parser_stacks_if_possible(&constraint, gss, terminal);
                    assert_eq!(
                        fused.as_ref().map(canonical_gss),
                        legacy.as_ref().map(canonical_gss),
                        "terminal={terminal} source={:#?}",
                        canonical_gss(gss),
                    );

                    let profiled =
                        advance_parser_stacks_profiled_if_possible(&constraint, gss, terminal);
                    assert_eq!(
                        (!profiled.advanced.is_empty())
                            .then(|| canonical_gss(&profiled.advanced)),
                        legacy.as_ref().map(canonical_gss),
                        "profiled terminal={terminal} source={:#?}",
                        canonical_gss(gss),
                    );
                    assert_eq!(profiled.may_ns, 0);
                }
            }
        }
    }

    fn top_local_prune_reference(
        constraint: &Constraint,
        gss: &ParserGSS,
        bytes: &[u8],
    ) -> ParserGSS {
        let prune_partition = |partition: ParserGSS| {
            partition.apply_and_prune_no_promote(
                |terminals_disallowed: &TerminalsDisallowed| {
                    if terminals_disallowed.is_empty() {
                        return Some(TerminalsDisallowed::new());
                    }

                    let mut remapped = BTreeMap::new();
                    for (&continuation_tokenizer_state, disallowed) in
                        terminals_disallowed.iter()
                    {
                        let execution = execute_tokenizer_from_state_small(
                            constraint,
                            bytes,
                            continuation_tokenizer_state,
                        );
                        if execution
                            .matches
                            .iter()
                            .any(|matched| disallowed.contains(&matched.id))
                        {
                            return None;
                        }
                        for end_state in execution.end_state {
                            let future = constraint
                                .tokenizer
                                .possible_future_terminals(end_state);
                            for &terminal in disallowed.iter() {
                                if future.contains(terminal as usize) {
                                    remapped
                                        .entry(end_state)
                                        .or_insert_with(BTreeSet::new)
                                        .insert(terminal);
                                }
                            }
                        }
                    }
                    Some(TerminalsDisallowed::from_map(remapped))
                },
            )
        };

        let mut partitions = Vec::new();
        let root = gss.isolate(None);
        if !root.is_empty() {
            let root = prune_partition(root);
            if !root.is_empty() {
                partitions.push(root);
            }
        }
        for parser_state in gss.peek_values() {
            let partition = prune_partition(gss.isolate(Some(parser_state)));
            if !partition.is_empty() {
                partitions.push(partition);
            }
        }

        let mut iter = partitions.into_iter();
        let Some(mut merged) = iter.next() else {
            return ParserGSS::empty();
        };
        for partition in iter {
            merged = merged.merge(&partition);
        }
        merged
    }

    #[test]
    fn initial_prune_advances_each_continuation_tokenizer_state() {
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"b".to_vec()),
                (2, b"ab".to_vec()),
            ]);
        let grammar = r#"
start start;
t A ::= "a";
t B ::= "a" | "ab";
nt item ::= A | B;
nt start ::= item item? item?;
"#;
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();
        let mut state = constraint.start();
        state.commit_token(0).unwrap();

        let tokenizer_state = constraint.runtime_commit_initial_state();
        // This test exercises the GSS pruning primitive itself, so cross the
        // bounded flat-state boundary explicitly before inspecting one map value.
        state.state.normalize_duplicate_keys();
        let gss = state
            .state
            .get(&tokenizer_state)
            .expect("ambiguous a must retain a reset lexer branch");
        assert!(
            !gss.all_accs_satisfy(|td: &TerminalsDisallowed| td.is_empty()),
            "MRE requires a stale residual exclusion on the reset branch"
        );

        let exec_result =
            execute_tokenizer_from_state_small(&constraint, b"b", tokenizer_state);
        assert!(exec_result.matches.is_empty());
        assert!(exec_result.end_state.is_empty());

        let pruned = prune_single_initial_state_for_exec(
            &constraint,
            gss.clone(),
            tokenizer_state,
            &exec_result,
            b"b",
        );
        assert!(
            !pruned.is_empty()
                && pruned.all_accs_satisfy(|td: &TerminalsDisallowed| td.is_empty()),
            "the valid A=a branch must survive while the provisional B=a branch is invalidated by B=ab"
        );
    }

    #[test]
    fn delayed_exclusion_survives_across_model_token_boundaries() {
        let vocab = Vocab::new(vec![
            (0, b"@".to_vec()),
            (1, b" double".to_vec()),
            (2, b"Quote".to_vec()),
            (3, b"x".to_vec()),
            (4, b"=".to_vec()),
            (5, b"%".to_vec()),
            (6, b"0".to_vec()),
        ]);
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start start;
                ignore WS;
                nt start ::= declaration expression;
                nt declaration ::= "@" ID;
                nt expression ::= ID OP "0";
                t WS ::= [ \t\r\n]+;
                t OP ::= "=" | "%=";
                t ID ::= [A-Za-z_$] [A-Za-z0-9_$]*;
            "#,
            &vocab,
        )
        .unwrap();

        let mut state = constraint.start();
        state.commit_token(0).unwrap();
        state.commit_token(1).unwrap();
        state.commit_token(2).unwrap();

        let mask = state.mask();
        let allowed = |token: u32| {
            ((mask[token as usize / 32] >> (token % 32)) & 1) != 0
        };
        assert!(allowed(3), "continuing the current ID remains viable");
        assert!(!allowed(4), "the consumed ID cannot also satisfy the next ID");
        assert!(!allowed(5), "a prefix of %= cannot follow until a new ID is consumed");
        let mut probe = state.clone();
        assert!(probe.commit_token(4).is_err());
        let mut probe = state.clone();
        assert!(probe.commit_token(5).is_err());
    }

    #[test]
    fn recursive_delayed_exclusion_stays_in_leaf_tokenizer_terminal_coordinate() {
        let vocab = Vocab::new(vec![
            (0, b"@".to_vec()),
            (1, b" double".to_vec()),
            (2, b"Quote".to_vec()),
            (3, b"x".to_vec()),
            (4, b"=".to_vec()),
            (5, b"%".to_vec()),
            (6, b"0".to_vec()),
        ]);
        let body = Constraint::from_glrm_grammar(
            r#"
                start start;
                ignore WS;
                nt start ::= declaration expression;
                nt declaration ::= "@" ID;
                nt expression ::= ID OP "0";
                t WS ::= [ \t\r\n]+;
                t OP ::= "=" | "%=";
                t ID ::= [A-Za-z_$] [A-Za-z0-9_$]*;
            "#,
            &vocab,
        )
        .unwrap();
        let parent = Constraint::compile(
            Grammar::glrm("glrm 1; start document; extern grammar body; nt document = body;"),
            &vocab,
        )
        .unwrap();
        let bound = parent
            .bind_grammar_dynamic_boundary("body", body.clone())
            .unwrap();
        assert!(bound.uses_compact_segmented_parser_runtime());
        let loaded = Constraint::load(bound.save()).unwrap();

        let mut expected = body.start();
        for token in [0, 1, 2] {
            expected.commit_token(token).unwrap();
        }
        let expected_mask = expected.mask();
        for constraint in [&bound, &loaded] {
            let mut state = constraint.start();
            for token in [0, 1, 2] {
                state.commit_token(token).unwrap();
            }
            assert_eq!(state.mask(), expected_mask);
            assert!(state.state.iter().any(|(_, gss)| {
                !gss.all_accs_satisfy(|td: &TerminalsDisallowed| td.is_empty())
            }));
            let mut probe = state.clone();
            assert!(probe.commit_token(4).is_err());
            let mut probe = state.clone();
            assert!(probe.commit_token(5).is_err());
        }
    }

    #[test]
    fn initial_prune_matches_top_local_reference_on_generated_small_languages() {
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

        for &a in &languages {
            for &b in &languages {
                for start_rule in [
                    "nt item ::= A | B;\nnt start ::= item item? item?;",
                    "nt start ::= A A | B B;",
                    "nt start ::= A B | B A;",
                ] {
                    let grammar = format!(
                        "start start;\nignore WS;\nt WS ::= \" \"+;\n{}{}{start_rule}\n",
                        rule("A", a),
                        rule("B", b),
                    );
                    let constraint = Constraint::from_glrm_grammar(&grammar, &vocab).unwrap();
                    let mut frontier = vec![(constraint.start(), Vec::<u32>::new())];
                    let mut seen = BTreeSet::new();

                    for depth in 0..=4 {
                        let mut next = Vec::new();
                        for (state, path) in frontier {
                            let state_key = canonical_commit_state(&state.state);
                            if !seen.insert(state_key) {
                                continue;
                            }
                            for (token_id, bytes) in constraint.token_bytes_iter() {
                                for (&tokenizer_state, gss) in &state.state {
                                    if gss
                                        .all_accs_satisfy(|td: &TerminalsDisallowed| td.is_empty())
                                    {
                                        continue;
                                    }
                                    let exec_result = execute_tokenizer_from_state_small(
                                        &constraint,
                                        bytes,
                                        tokenizer_state,
                                    );
                                    let actual = prune_single_initial_state_for_exec(
                                        &constraint,
                                        gss.clone(),
                                        tokenizer_state,
                                        &exec_result,
                                        bytes,
                                    );
                                    let expected = top_local_prune_reference(
                                        &constraint,
                                        gss,
                                        bytes,
                                    );
                                    assert_eq!(
                                        canonical_gss(&actual),
                                        canonical_gss(&expected),
                                        "initial prune crossed parser-top correlations: A={a:#06b} B={b:#06b} depth={depth} path={path:?} token={token_id} bytes={bytes:?} tokenizer_state={tokenizer_state}\ngrammar:\n{grammar}\nsource={:#?}\nactual={:#?}\nexpected={:#?}",
                                        canonical_gss(gss),
                                        canonical_gss(&actual),
                                        canonical_gss(&expected),
                                    );
                                }

                                if depth < 4 {
                                    let mut advanced = state.clone();
                                    if advanced.commit_bytes(bytes).is_ok() {
                                        let mut next_path = path.clone();
                                        next_path.push(token_id);
                                        next.push((advanced, next_path));
                                    }
                                }
                            }
                        }
                        frontier = next;
                    }
                }
            }
        }
    }

    #[test]
    fn rejected_public_commits_enter_fail_state() {
        let vocab = Vocab::new(
            vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start start;
                t A ::= "a";
                nt start ::= A;
            "#,
            &vocab,
        )
        .unwrap();

        let assert_failed = |state: &ConstraintState<'_>| {
            assert!(state.state.is_empty());
            assert!(state.mask().iter().all(|&word| word == 0));
        };

        let mut state = constraint.start();
        assert!(state.commit_token(1).is_err());
        assert_failed(&state);

        let mut state = constraint.start();
        assert!(state.commit_token_timed_ns(1).is_err());
        assert_failed(&state);

        let mut state = constraint.start();
        assert!(state.commit_token_profiled(1).is_err());
        assert_failed(&state);

        let mut state = constraint.start();
        assert!(state.commit_token_per_advance(1).is_err());
        assert_failed(&state);

        let mut state = constraint.start();
        assert!(state.commit_bytes(b"b").is_err());
        assert_failed(&state);
    }

    fn assert_fast_and_general_queue_match<'a>(
        constraint: &'a Constraint,
        fast_state: &ConstraintState<'a>,
        token_id: u32,
        bytes: &[u8],
        context: &str,
    ) -> Option<ConstraintState<'a>> {
        let mut fast = fast_state.clone();
        let mut profiled = fast_state.clone();
        let mut general = fast_state.clone();

        let fast_result = commit_bytes_impl(
            constraint,
            &mut fast.state,
            bytes,
            &mut fast.buffers,
        );
        let profiled_result = commit_bytes_impl_profiled(
            constraint,
            &mut profiled.state,
            bytes,
            &mut profiled.buffers,
            None,
            true,
        );
        let general_result = commit_bytes_impl_profiled(
            constraint,
            &mut general.state,
            bytes,
            &mut general.buffers,
            None,
            false,
        );

        assert_eq!(
            fast_result.is_ok(),
            general_result.is_ok(),
            "commit result mismatch: {context} token_id={token_id} bytes={bytes:?}\nfast={:?}\ngeneral={:?}",
            fast.state,
            general.state,
        );
        assert_eq!(
            profiled_result.is_ok(),
            general_result.is_ok(),
            "profiled commit result mismatch: {context} token_id={token_id} bytes={bytes:?}\nprofiled={:?}\ngeneral={:?}",
            profiled.state,
            general.state,
        );
        if fast_result.is_err() {
            return None;
        }
        assert_eq!(
            canonical_commit_state(&fast.state),
            canonical_commit_state(&general.state),
            "successful commit state mismatch: {context} token_id={token_id} bytes={bytes:?}\nfast_stacks={:#?}\ngeneral_stacks={:#?}",
            fast.state
                .iter()
                .map(|(&ts, gss)| (ts, gss.to_stacks(4_096).expect("stack enumeration exceeded explicit limit")))
                .collect::<Vec<_>>(),
            general
                .state
                .iter()
                .map(|(&ts, gss)| (ts, gss.to_stacks(4_096).expect("stack enumeration exceeded explicit limit")))
                .collect::<Vec<_>>(),
        );
        assert_eq!(
            canonical_commit_state(&profiled.state),
            canonical_commit_state(&general.state),
            "successful profiled commit state mismatch: {context} token_id={token_id} bytes={bytes:?}\nprofiled_stacks={:#?}\ngeneral_stacks={:#?}",
            profiled
                .state
                .iter()
                .map(|(&ts, gss)| (ts, gss.to_stacks(4_096).expect("stack enumeration exceeded explicit limit")))
                .collect::<Vec<_>>(),
            general
                .state
                .iter()
                .map(|(&ts, gss)| (ts, gss.to_stacks(4_096).expect("stack enumeration exceeded explicit limit")))
                .collect::<Vec<_>>(),
        );

        Some(fast)
    }

    #[test]
    fn flat_frontier_accepts_input_past_the_inline_state_capacity() {
        let vocab = Vocab::new(vec![(0, b"a".to_vec())]);
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start start;
                t A ::= "a";
                nt start ::= A;
            "#,
            &vocab,
        )
        .expect("single-terminal grammar should compile");
        let start = constraint.start();
        let (tokenizer_state, parser_gss) = start.state.entries[0].clone();
        let mut state = ParserStateMap::default();
        for _ in 0..=INLINE_PARSER_STATE_CAPACITY {
            state.insert_flat_alternative(tokenizer_state, parser_gss.clone());
        }
        assert_eq!(state.len(), INLINE_PARSER_STATE_CAPACITY + 1);

        let mut original = Vec::with_capacity(LINEAR_STACK_RESERVE);
        let mut work = Vec::with_capacity(LINEAR_STACK_RESERVE);
        let mut tokenizer_scratch = tokenizer_scan::ReusableTokenizerExecScratch::default();
        let mut frontier = FlatFrontierScratch::default();
        let result = try_commit_flat_frontier_in_place(
            &constraint,
            &mut state,
            b"a",
            &mut original,
            &mut work,
            &mut tokenizer_scratch,
            &mut frontier,
        );
        assert!(matches!(result, Some(Ok(()))));
        assert!(!state.is_empty());
    }

    #[test]
    fn flat_frontier_preserves_all_stacks_in_a_live_lexer_continuation_group() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"ab".to_vec()),
            (3, b"ba".to_vec()),
        ]);
        let grammar = r#"
            start start;
            t A ::= "a";
            t B ::= "b" | "ab";
            nt item ::= A | B;
            nt start ::= item item? item?;
        "#;
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();
        let start = constraint.start();
        let after_ab = assert_fast_and_general_queue_match(
            &constraint,
            &start,
            2,
            b"ab",
            "grouped lexer continuation regression after ab",
        )
        .unwrap();
        let after_ba = assert_fast_and_general_queue_match(
            &constraint,
            &after_ab,
            3,
            b"ba",
            "grouped lexer continuation regression after ab, ba",
        )
        .unwrap();

        let stacks = canonical_commit_state(&after_ba.state);
        let tokenizer_one = stacks
            .iter()
            .find(|(tokenizer_state, _)| *tokenizer_state == 1)
            .expect("lexer continuation state must remain live");
        assert_eq!(
            tokenizer_one.1,
            vec![
                (vec![0, 3, 2], Vec::new()),
                (vec![0, 3, 5, 2], Vec::new()),
            ],
        );
    }

    #[test]
    fn monolithic_commit_fast_paths_match_general_queue_on_small_language_space() {
        const WORDS: [&str; 4] = ["a", "b", "ab", "ba"];
        let vocab = Vocab::new(
            WORDS
                .iter()
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

        for &a in &languages {
            for &b in &languages {
                let grammar = format!(
                    "start start;\n{}{}nt item ::= A | B;\nnt start ::= item item? item?;\n",
                    rule("A", a),
                    rule("B", b),
                );
                let constraint = Constraint::from_glrm_grammar(&grammar, &vocab).unwrap();
                let mut frontier = vec![(constraint.start(), Vec::<u32>::new())];

                for depth in 0..3 {
                    let mut next = Vec::new();
                    for (state, path) in frontier {
                        let mask = state.mask();
                        for (&token_id, bytes) in vocab.entries_map().iter() {
                            let context = format!(
                                "A_mask={a:#06b} B_mask={b:#06b} depth={depth} path={path:?}\ngrammar:\n{grammar}"
                            );
                            let next_state = assert_fast_and_general_queue_match(
                                &constraint,
                                &state,
                                token_id,
                                bytes,
                                &context,
                            );
                            let token_in_mask = mask
                                .get(token_id as usize / 32)
                                .is_some_and(|word| {
                                    word & (1u32 << (token_id % 32)) != 0
                                });
                            assert_eq!(
                                token_in_mask,
                                next_state.is_some(),
                                "mask/commit mismatch: {context} token_id={token_id} bytes={bytes:?}"
                            );
                            if let Some(next_state) = next_state {
                                let mut next_path = path.clone();
                                next_path.push(token_id);
                                next.push((next_state, next_path));
                            }
                        }
                    }
                    frontier = next;
                }
            }
        }
    }

    #[test]
    fn residual_bc_fast_path_matches_general_queue() {
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"b".to_vec()),
                (2, b"c".to_vec()),
                (3, b"ab".to_vec()),
                (4, b"ba".to_vec()),
                (5, b"bc".to_vec()),
                (6, b"abc".to_vec()),
            ]);
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start start;
                t A ::= "a" | "ab";
                t B ::= "bc";
                nt item ::= A | B;
                nt start ::= item item? item?;
            "#,
            &vocab,
        )
        .unwrap();

        let mut fast = constraint.start();
        let mut slow = constraint.start();
        let fast_result = commit_bytes_impl(
            &constraint,
            &mut fast.state,
            vocab.entries_map().get(&0).unwrap(),
            &mut fast.buffers,
        );
        fast.generation += 1;
        assert!(fast_result.is_ok());
        let slow_result = commit_bytes_impl_profiled(
            &constraint,
            &mut slow.state,
            vocab.entries_map().get(&0).unwrap(),
            &mut slow.buffers,
            None,
            false,
        );
        slow.generation += 1;
        assert!(slow_result.is_ok());
        assert_eq!(fast.state, slow.state, "state mismatch after token a");

        let mut next_fast = fast.clone();
        let mut next_slow = slow.clone();
        let fast_result = commit_bytes_impl(
            &constraint,
            &mut next_fast.state,
            vocab.entries_map().get(&5).unwrap(),
            &mut next_fast.buffers,
        );
        let slow_result = commit_bytes_impl_profiled(
            &constraint,
            &mut next_slow.state,
            vocab.entries_map().get(&5).unwrap(),
            &mut next_slow.buffers,
            None,
            false,
        );
        assert_eq!(
            fast_result.is_ok(),
            slow_result.is_ok(),
            "fast={:?}\nslow={:?}",
            next_fast.state,
            next_slow.state,
        );
        if fast_result.is_ok() {
            assert_eq!(next_fast.state, next_slow.state);
        }
    }

    #[test]
    fn epsilon_commit_fast_paths_match_no_fast_path_reference() {
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"b".to_vec()),
                (2, b"c".to_vec()),
                (3, b"aa".to_vec()),
                (4, b"ab".to_vec()),
                (5, b" ".to_vec()),
                (6, b" a".to_vec()),
                (7, b"a ".to_vec()),
                (8, b" a ".to_vec()),
                (9, b"abc".to_vec()),
                (10, b"aab".to_vec()),
            ]);
        let grammar = crate::grammar::glrm::from_glrm(
            r#"
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
            "#,
        )
        .unwrap();
        let grammar = crate::grammar::ast::lower(&grammar).unwrap();
        let constraint = crate::compiler::pipeline::compile_owned_with_lexer_adaptive(
            grammar,
            &vocab,
            false,
        );
        assert!(constraint.tokenizer_has_epsilon_transitions);

        let mut frontier = vec![(
            constraint.start(),
            constraint.start(),
            constraint.start(),
            Vec::<u32>::new(),
        )];
        for depth in 0..=4 {
            let mut next = Vec::new();
            for (fast, profiled, general, path) in frontier {
                assert_eq!(
                    fast.mask(),
                    general.mask(),
                    "epsilon mask mismatch after path {path:?}\nfast={:#?}\ngeneral={:#?}",
                    canonical_commit_state(&fast.state),
                    canonical_commit_state(&general.state),
                );
                assert_eq!(
                    profiled.mask(),
                    general.mask(),
                    "epsilon profiled-mask mismatch after path {path:?}\nprofiled={:#?}\ngeneral={:#?}",
                    canonical_commit_state(&profiled.state),
                    canonical_commit_state(&general.state),
                );
                assert_eq!(
                    fast.is_accepting(),
                    general.is_accepting(),
                    "epsilon completion mismatch after path {path:?}",
                );
                assert_eq!(
                    profiled.is_accepting(),
                    general.is_accepting(),
                    "epsilon profiled completion mismatch after path {path:?}",
                );
                if depth == 4 {
                    continue;
                }

                for (&token_id, bytes) in vocab.entries_map().iter() {
                    let mut next_fast = fast.clone();
                    let mut next_profiled = profiled.clone();
                    let mut next_general = general.clone();
                    let fast_result = commit_bytes_impl(
                        &constraint,
                        &mut next_fast.state,
                        bytes,
                        &mut next_fast.buffers,
                    );
                    let profiled_result = commit_bytes_impl_profiled(
                        &constraint,
                        &mut next_profiled.state,
                        bytes,
                        &mut next_profiled.buffers,
                        None,
                        true,
                    );
                    let general_result = commit_bytes_impl_profiled(
                        &constraint,
                        &mut next_general.state,
                        bytes,
                        &mut next_general.buffers,
                        None,
                        false,
                    );
                    assert_eq!(
                        fast_result.is_ok(),
                        general_result.is_ok(),
                        "epsilon commit result mismatch after path {path:?} token_id={token_id} bytes={bytes:?}\nfast={:#?}\ngeneral={:#?}",
                        canonical_commit_state(&next_fast.state),
                        canonical_commit_state(&next_general.state),
                    );
                    assert_eq!(
                        profiled_result.is_ok(),
                        general_result.is_ok(),
                        "epsilon profiled commit result mismatch after path {path:?} token_id={token_id} bytes={bytes:?}\nprofiled={:#?}\ngeneral={:#?}",
                        canonical_commit_state(&next_profiled.state),
                        canonical_commit_state(&next_general.state),
                    );
                    if fast_result.is_ok() {
                        let mut next_path = path.clone();
                        next_path.push(token_id);
                        next.push((next_fast, next_profiled, next_general, next_path));
                    }
                }
            }
            frontier = next;
        }
    }

    #[test]
    fn epsilon_full_width_terminal_with_empty_accumulators_uses_fast_path() {
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);
        let grammar = crate::grammar::glrm::from_glrm(
            r#"
                start start;
                lexer group left ::= A;
                lexer group right ::= B;
                t A ::= "a";
                t B ::= "b";
                nt start ::= A | B;
            "#,
        )
        .unwrap();
        let grammar = crate::grammar::ast::lower(&grammar).unwrap();
        let constraint = crate::compiler::pipeline::compile_owned_with_lexer_adaptive(
            grammar,
            &vocab,
            false,
        );

        let mut state = constraint.start();
        let profile = state.commit_token_profiled(0).unwrap();
        assert!(profile.fast_path_total_ns > 0, "profile={profile:?}");
        assert_eq!(profile.n_queue_entries, 0, "profile={profile:?}");
        assert_eq!(profile.n_advances, 1, "profile={profile:?}");
    }
}
