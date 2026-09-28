//! Actionable lexer matches and delayed terminal exclusions.

use super::admission::ActionableTerminals;
use super::admission::runtime_tokenizer_future_terminals;
use super::tokenizer_scan;
use super::tokenizer_scan::InitialCommitScan;
use super::tokenizer_scan::execute_tokenizer_from_state_small;
use super::tokenizer_scan::execute_tokenizer_reusable_from_states;
use crate::automata::lexer::Lexer;
use crate::automata::lexer::tokenizer::TokenizerExecResult;
use crate::automata::lexer::tokenizer::TokenizerMatch;
use crate::compiler::glr::accumulator::TerminalsDisallowed;
use crate::compiler::glr::parser::ParserGSS;
use crate::runtime::constraint::Constraint;
use crate::runtime::state::INLINE_PARSER_STATE_CAPACITY;
use crate::runtime::state::ParserStateMap;
use rustc_hash::FxHashMap;
use rustc_hash::FxHashSet;
use smallvec::SmallVec;
use std::collections::BTreeMap;
use std::collections::BTreeSet;

pub(super) const SMALL_NORMALIZED_MATCH_LINEAR_SCAN_MAX: usize = 16;

#[derive(Clone, Copy)]
pub(super) struct NormalizedMatch {
    pub(super) terminal_id: u32,
    pub(super) width: usize,
    pub(super) ignored: bool,
}

pub(super) fn state_has_nonempty_accumulators(state: &ParserStateMap) -> bool {
    state.values().any(|gss| !gss.all_accs_satisfy(|td: &TerminalsDisallowed| td.is_empty()))
}

impl InitialCommitScan {
    pub(super) fn collect(constraint: &Constraint, state: &ParserStateMap, bytes: &[u8]) -> Self {
        let mut exec_results = FxHashMap::default();

        for &tokenizer_state in state.keys() {
            let exec_result =
                execute_tokenizer_from_state_small(constraint, bytes, tokenizer_state);
            exec_results.insert(tokenizer_state, exec_result);
        }

        Self { exec_results }
    }

    pub(super) fn take_exec_result(&mut self, tokenizer_state: u32) -> Option<TokenizerExecResult> {
        self.exec_results.remove(&tokenizer_state)
    }
}

pub(super) fn is_ignored_terminal(ignore_terminal: Option<u32>, terminal: u32) -> bool {
    Some(terminal) == ignore_terminal
}

#[inline]
pub(super) fn runtime_is_ignored_terminal(
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

pub(super) fn is_actionable_terminal(
    actionable_terminals: Option<&ActionableTerminals>,
    constraint: &Constraint,
    terminal: u32,
) -> bool {
    !actionable_terminals.is_some_and(|actionable| !actionable.contains(constraint, terminal))
}

pub(super) fn for_each_relevant_matched_terminal(
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

pub(super) fn advance_uniform_disallowed_interest_only(
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
            if disallowed.iter().any(|terminal| matched.contains(*terminal as usize)) {
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

pub(super) fn collect_unique_actionable_reusable_matches(
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
        normalized.push(NormalizedMatch { terminal_id: matched.id, width: matched.width, ignored });
    }
    normalized
}

pub(super) fn collect_unique_actionable_matches(
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
        normalized.push(NormalizedMatch { terminal_id: matched.id, width: matched.width, ignored });
    }
    normalized
}

pub(super) fn prune_single_initial_state_for_exec(
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

pub(super) fn advance_terminals_disallowed_over_bytes(
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
        if matches.iter().any(|matched| disallowed.contains(&matched.id)) {
            return None;
        }
        for &end_state in end_states {
            let future = runtime_tokenizer_future_terminals(constraint, end_state)?;
            for &terminal in disallowed.iter() {
                if future.contains(terminal as usize) {
                    remapped.entry(end_state).or_insert_with(BTreeSet::new).insert(terminal);
                }
            }
        }
    }
    Some(TerminalsDisallowed::from_map(remapped))
}

pub(super) fn single_disallowed_pair(acc: &TerminalsDisallowed) -> Option<(u32, u32)> {
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

pub(super) fn try_prune_single_initial_state_batched_accumulators(
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
            if !left.is_disjoint(constraint.tokenizer.possible_future_terminals(right_state)) {
                return None;
            }
        }
    }
    let starts = pairs.iter().map(|&(state, _)| state).collect::<SmallVec<[u32; 8]>>();
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
            if constraint.tokenizer.possible_future_terminals(end_state).contains(terminal as usize)
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
        remapped.iter().find(|(source, _)| source == acc).and_then(|(_, result)| result.clone())
    }))
}

pub(super) fn prune_single_initial_state_for_parts(
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

pub(super) fn prune_single_initial_state_for_terminal(
    gss: ParserGSS,
    tokenizer_state: u32,
    terminal: u32,
    end_state: Option<u32>,
) -> ParserGSS {
    if end_state.is_none()
        && gss.all_accs_satisfy(|td: &TerminalsDisallowed| {
            td.get(&tokenizer_state).is_none_or(|disallowed| !disallowed.contains(&terminal))
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

pub(super) fn apply_future_terminal_disallow(
    constraint: &Constraint,
    exec_result: &TokenizerExecResult,
    terminal: u32,
    gss: ParserGSS,
) -> ParserGSS {
    apply_future_terminal_disallow_for_states(constraint, &exec_result.end_state, terminal, gss)
}

pub(super) fn apply_future_terminal_disallow_for_states(
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
