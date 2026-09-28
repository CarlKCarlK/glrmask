//! Exact parser admission and bounded caches for tokenizer continuations.

use crate::automata::lexer::Lexer;
use crate::compiler::glr::parser::ParserGSS;
use crate::compiler::glr::parser::stack_admissible_terminals;
use crate::compiler::glr::parser::stack_may_advance_disjoint_top_terminals_bounded;
use crate::compiler::glr::parser::stack_may_advance_on;
use crate::compiler::glr::parser::stack_may_advance_on_any;
use crate::compiler::glr::parser::stack_may_advance_on_any_control_closed;
use crate::compiler::glr::parser::stack_may_advance_on_control_closed;
use crate::compiler::glr::table::AdmissionPolicy;
use crate::compiler::glr::table::row::ActionRow;
use crate::runtime::constraint::Constraint;
use crate::runtime::state::ParserAdmissionCacheEntry;
use smallvec::SmallVec;
use std::borrow::Cow;

/// Advance once when admission requires exact simulation. Row-presence tables
/// retain their cheap precheck; exact-simulation tables must not execute the
/// same reduction closure once for admission and again for the actual advance.
#[inline]
pub(super) fn parser_may_advance_on(
    constraint: &Constraint,
    stack: &ParserGSS,
    terminal: u32,
) -> bool {
    if let Some(result) = constraint.compact_segmented_parser_may_advance_on(stack, terminal) {
        return result;
    }
    constraint.direct_regular_admissible_terminals(stack).map_or_else(
        || {
            if constraint.table.control_terminals.is_empty() {
                stack_may_advance_on(&constraint.table, stack, terminal)
            } else {
                stack_may_advance_on_control_closed(&constraint.table, stack, terminal)
            }
        },
        |terminals| terminals.contains(terminal as usize),
    )
}

#[inline]
pub(super) fn bitset_prefix_intersects(
    left: &crate::ds::bitset::BitSet,
    right: &crate::ds::bitset::BitSet,
) -> bool {
    left.words().iter().zip(right.words()).any(|(left, right)| (*left & *right) != 0)
}

#[inline]
pub(super) fn bitset_union_intersection_prefix(
    dst: &mut crate::ds::bitset::BitSet,
    left: &crate::ds::bitset::BitSet,
    right: &crate::ds::bitset::BitSet,
) {
    for ((dst, left), right) in dst.words_mut().iter_mut().zip(left.words()).zip(right.words()) {
        *dst |= *left & *right;
    }
}

/// Return the sole candidate in `actions` that is requested by `terminals`
/// and not already covered by the unconditional-advance row.  `Err(())`
/// means there are at least two candidates, so the single-terminal bounded
/// shortcut is not applicable.
#[inline]
pub(super) fn single_conditional_candidate(
    actions: &ActionRow,
    unconditional: &crate::ds::bitset::BitSet,
    terminals: &crate::ds::bitset::BitSet,
) -> Result<Option<u32>, ()> {
    let mut selected = None;
    for (terminal, _action) in actions.iter() {
        let bit = terminal as usize;
        if bit >= terminals.len() || !terminals.contains(bit) || unconditional.contains(bit) {
            continue;
        }
        if selected.is_some() {
            return Err(());
        }
        selected = Some(terminal);
    }
    Ok(selected)
}

pub(super) fn exact_simulation_prefilter_may_advance_on_any(
    constraint: &Constraint,
    stack: &ParserGSS,
    terminals: &crate::ds::bitset::BitSet,
) -> Option<bool> {
    if constraint.table.admission_policy != AdmissionPolicy::ExactSimulation
        || !constraint.table.control_terminals.is_empty()
        || constraint.table.unconditional_advance.len() != constraint.table.num_states as usize
    {
        return None;
    }

    let tops = stack.peek_values();
    let mut any_relevant = false;
    for &state in &tops {
        let advance = constraint.table.advance.get(state as usize)?;
        let unconditional = constraint.table.unconditional_advance_row(state)?;
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
        let unconditional = constraint.table.unconditional_advance_row(top)?;
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
pub(super) fn parser_may_advance_on_any(
    constraint: &Constraint,
    stack: &ParserGSS,
    terminals: &crate::ds::bitset::BitSet,
) -> bool {
    if let Some(result) = constraint.compact_segmented_parser_may_advance_on_any(stack, terminals) {
        return result;
    }
    if let Some(result) =
        exact_simulation_prefilter_may_advance_on_any(constraint, stack, terminals)
    {
        return result;
    }
    constraint.direct_regular_admissible_terminals(stack).map_or_else(
        || {
            if constraint.table.control_terminals.is_empty() {
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

#[inline]
pub(super) fn runtime_tokenizer_is_reset_state(constraint: &Constraint, state: u32) -> bool {
    if constraint.uses_compact_segmented_parser_runtime() {
        constraint.recursive_tokenizer_is_reset_state(state)
    } else {
        state == constraint.runtime_commit_initial_state()
    }
}

#[inline]
pub(super) fn runtime_tokenizer_future_terminals(
    constraint: &Constraint,
    state: u32,
) -> Option<Cow<'_, crate::ds::bitset::BitSet>> {
    if constraint.uses_compact_segmented_parser_runtime() {
        constraint.recursive_tokenizer_future_scoped_terminals(state)
    } else {
        Some(Cow::Borrowed(constraint.tokenizer.possible_future_terminals(state)))
    }
}

#[inline]
pub(super) fn runtime_terminal_count(constraint: &Constraint) -> usize {
    constraint
        .uses_compact_segmented_parser_runtime()
        .then(|| constraint.recursive_runtime_terminal_count())
        .flatten()
        .unwrap_or(constraint.table.num_terminals as usize)
}

/// An unfinished ignore lexeme is a valid byte prefix even though completing
/// it will not shift a parser terminal. Lexer-continuation tests must retain
/// that route independently of the parser's ordinary terminal admission.
#[inline]
pub(super) fn runtime_future_contains_ignore(
    constraint: &Constraint,
    future: &crate::ds::bitset::BitSet,
) -> bool {
    if constraint.uses_compact_segmented_parser_runtime() {
        future.iter_ones().any(|terminal| constraint.recursive_terminal_is_ignore(terminal as u32))
    } else {
        constraint.ignore_terminal.is_some_and(|terminal| future.contains(terminal as usize))
    }
}

#[inline]
pub(super) fn end_state_may_advance(
    constraint: &Constraint,
    gss: &ParserGSS,
    end_state: u32,
) -> bool {
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
pub(super) fn batched_end_state_admitted_terminals(
    constraint: &Constraint,
    gss: &ParserGSS,
    end_states: &[u32],
) -> Option<crate::ds::bitset::BitSet> {
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
    if let Some(admitted) = constraint.compact_segmented_parser_admitted_terminals(gss, &candidates)
    {
        return Some(admitted);
    }
    if let Some(direct) = constraint.direct_regular_admissible_terminals(gss) {
        let mut admitted = candidates;
        admitted.intersect_with(&direct);
        Some(admitted)
    } else {
        Some(exact_simulation_prefiltered_admitted_terminals(constraint, gss, &candidates))
    }
}

pub(super) const PARSER_ADMISSION_CACHE_CAPACITY: usize = 8;

pub(super) const PARSER_ADMISSION_BOOLEAN_CACHE_CAPACITY: usize = 8;

pub(super) fn exact_simulation_prefiltered_admitted_terminals(
    constraint: &Constraint,
    gss: &ParserGSS,
    candidates: &crate::ds::bitset::BitSet,
) -> crate::ds::bitset::BitSet {
    if constraint.table.admission_policy != AdmissionPolicy::ExactSimulation
        || !constraint.table.control_terminals.is_empty()
        || constraint.table.unconditional_advance.len() != constraint.table.num_states as usize
    {
        return stack_admissible_terminals(&constraint.table, gss, candidates);
    }

    let mut guaranteed = crate::ds::bitset::BitSet::new(candidates.len());
    let mut unresolved = crate::ds::bitset::BitSet::new(candidates.len());
    for state in gss.peek_values() {
        let Some(advance) = constraint.table.advance.get(state as usize) else {
            return stack_admissible_terminals(&constraint.table, gss, candidates);
        };
        let Some(unconditional) = constraint.table.unconditional_advance_row(state) else {
            return stack_admissible_terminals(&constraint.table, gss, candidates);
        };
        bitset_union_intersection_prefix(&mut guaranteed, unconditional, candidates);
        bitset_union_intersection_prefix(&mut unresolved, advance, candidates);
    }
    for (unresolved_word, guaranteed_word) in
        unresolved.words_mut().iter_mut().zip(guaranteed.words())
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
    if let Some(admitted) = constraint.compact_segmented_parser_admitted_terminals(gss, candidates)
    {
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

pub(super) fn admission_cache_entry_index(
    cache: &mut SmallVec<[ParserAdmissionCacheEntry; 8]>,
    gss: &ParserGSS,
    terminal_count: usize,
) -> usize {
    if let Some(index) = cache.iter().position(|entry| entry.gss.ptr_eq(gss)) {
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

pub(super) fn try_local_row_presence_admission_words(
    constraint: &Constraint,
    gss: &ParserGSS,
    end_states: &[u32],
) -> Option<[u64; 32]> {
    const WORDS: usize = 32;
    if constraint.uses_compact_segmented_parser_runtime()
        || constraint.table.admission_policy != AdmissionPolicy::ExactSimulation
        || !constraint.table.control_terminals.is_empty()
        || constraint.tokenizer.num_terminals() as usize > WORDS * 64
        || constraint.table.advance.len() != constraint.table.num_states as usize
        || constraint.table.unconditional_advance.len() != constraint.table.num_states as usize
    {
        return None;
    }
    let tops = gss.peek_values();
    if tops.is_empty() {
        return None;
    }

    // Only terminals reachable from the tokenizer continuation matter for this
    // admission query. A table may contain stack-dependent actions elsewhere
    // without invalidating row-presence admission for this exact future set.
    let initial = constraint.runtime_commit_initial_state();
    let mut candidates = [0u64; WORDS];
    for &end_state in end_states {
        if end_state == initial {
            continue;
        }
        for (index, &word) in
            constraint.tokenizer.possible_future_terminals(end_state).words().iter().enumerate()
        {
            if index >= WORDS {
                return None;
            }
            candidates[index] |= word;
        }
    }

    let mut admitted = [0u64; WORDS];
    for state in tops {
        let advance = constraint.table.advance.get(state as usize)?;
        let unconditional = constraint.table.unconditional_advance_row(state)?;
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
pub(super) fn end_state_may_advance_from_row_words(
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
pub(super) fn cached_batched_end_state_admission(
    constraint: &Constraint,
    gss: &ParserGSS,
    end_states: &[u32],
    cache: &mut SmallVec<[ParserAdmissionCacheEntry; 8]>,
) -> Option<usize> {
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
pub(super) fn cached_single_end_state_may_advance(
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
        if let Some((_, result)) = entry.boolean_queries.iter().find(|(query, _)| query == future) {
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
pub(super) fn end_state_may_advance_from_cache_entry(
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
pub(super) fn end_state_may_advance_with_batch(
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
pub(super) fn wide_frontier_end_state_may_advance(
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
        || runtime_future_contains_ignore(
            constraint,
            constraint.tokenizer.possible_future_terminals(end_state),
        )
}

pub(super) enum ActionableTerminals {
    DirectDynamic(crate::ds::bitset::BitSet),
    SingleState(u32),
    WideFrontier(usize),
    ManyStates(SmallVec<[u32; 8]>),
}

impl ActionableTerminals {
    pub(super) fn from_gss(constraint: &Constraint, gss: &ParserGSS) -> Option<Self> {
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
        if states.is_empty() { None } else { Some(Self::ManyStates(states)) }
    }

    pub(super) fn bitset<'a>(
        &'a self,
        constraint: &'a Constraint,
    ) -> Option<&'a crate::ds::bitset::BitSet> {
        match self {
            Self::DirectDynamic(terminals) => Some(terminals),
            Self::SingleState(state_id) => constraint.table.advance_row(*state_id),
            Self::WideFrontier(index) => constraint
                .direct_regular_wide_frontier_acceptance
                .get(*index)
                .map(|summary| &summary.actionable_terminals),
            Self::ManyStates(_) => None,
        }
    }

    pub(super) fn contains(&self, constraint: &Constraint, terminal: u32) -> bool {
        match self {
            Self::DirectDynamic(terminals) => terminals.contains(terminal as usize),
            Self::SingleState(state_id) => constraint.table.advance_row_allows(*state_id, terminal),
            Self::WideFrontier(index) => constraint
                .direct_regular_wide_frontier_acceptance
                .get(*index)
                .is_some_and(|summary| summary.actionable_terminals.contains(terminal as usize)),
            Self::ManyStates(states) => states
                .iter()
                .any(|state_id| constraint.table.advance_row_allows(*state_id, terminal)),
        }
    }
}
