//! Exact lexer-only shortcuts that preserve the parser frontier.

use super::admission::ActionableTerminals;
use super::admission::cached_batched_end_state_admission;
use super::admission::cached_single_end_state_may_advance;
use super::admission::end_state_may_advance;
use super::admission::end_state_may_advance_from_cache_entry;
use super::admission::parser_may_advance_on;
use super::admission::wide_frontier_end_state_may_advance;
use super::flat_frontier::FLAT_FRONTIER_GSS_POOL_CAPACITY;
use super::flat_frontier::FlatFrontierScratch;
use super::lexical::is_actionable_terminal;
use super::lexical::is_ignored_terminal;
use super::lexical::runtime_is_ignored_terminal;
use super::lexical::state_has_nonempty_accumulators;
use super::tokenizer_scan;
use super::tokenizer_scan::execute_recursive_tokenizer_reusable;
use super::tokenizer_scan::execute_tokenizer_reusable;
use crate::automata::lexer::Lexer;
use crate::compiler::glr::accumulator::TerminalsDisallowed;
use crate::compiler::glr::parser::ParserGSS;
use crate::runtime::constraint::Constraint;
use crate::runtime::state::INLINE_PARSER_STATE_CAPACITY;
use crate::runtime::state::ParserAdmissionCacheEntry;
use crate::runtime::state::ParserStateMap;
use smallvec::SmallVec;

pub(super) fn scan_wide_frontier_lexer_only(
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
        state =
            constraint.tokenizer_fast_transitions.transition(&constraint.tokenizer, state, byte);
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

/// Advance a bounded multi-state lexer frontier without rebuilding parser GSSs.
///
/// This covers tokens that only move the tokenizer while every parser stack and
/// accumulator remains unchanged.  The general queue used to materialize fresh
/// GSS objects for this case, even when a wide frontier merely carried shared
/// parser stacks under new tokenizer-state keys.  Reusing Arc-backed GSS values
/// is exact and allocation-free; duplicate tokenizer keys remain explicit until
/// a later map-only operation requests normalization.
pub(super) fn try_commit_multi_state_lexer_only(
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

        if !execute_tokenizer_reusable(constraint, bytes, tokenizer_state, tokenizer_scratch) {
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
                    cached_single_end_state_may_advance(constraint, gss, end_state, admission_cache)
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
        return Some(Err("commit rejected: no valid parser states remain".to_string()));
    }
    output.sort_unstable_by_key(|(tokenizer_state, _)| *tokenizer_state);
    let mut remaining_uses = [0u8; INLINE_PARSER_STATE_CAPACITY];
    for &(_, source_index) in &output {
        remaining_uses[source_index] = remaining_uses[source_index].saturating_add(1);
    }
    frontier.reclaim_retired_gss();
    let mut remaining_for_assignment = remaining_uses;
    let mut used_pool = [false; FLAT_FRONTIER_GSS_POOL_CAPACITY];
    let mut pool_assignment = SmallVec::<[usize; INLINE_PARSER_STATE_CAPACITY]>::new();
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
        let Some(pool_index) =
            frontier.gss_pool.iter().enumerate().position(|(pool_index, gss)| {
                !used_pool[pool_index] && gss.can_replace_single_path_state_in_place(stack_scratch)
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
    let mut available_pool =
        SmallVec::<[Option<ParserGSS>; FLAT_FRONTIER_GSS_POOL_CAPACITY]>::new();
    available_pool.extend(frontier.gss_pool.drain(..).map(Some));
    let mut new_entries = SmallVec::<[(u32, ParserGSS); INLINE_PARSER_STATE_CAPACITY]>::new();
    for ((tokenizer_state, source_index), pool_index) in output.into_iter().zip(pool_assignment) {
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
    frontier.gss_pool.extend(available_pool.into_iter().flatten());
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

/// Exact single-lane lexer-only commit.
///
/// The parser GSS is left byte-for-byte unchanged. The shortcut applies only
/// when bounded tokenizer execution succeeds, there are no delayed exclusions,
/// and every terminal completed anywhere in `bytes` is both non-ignore and
/// parser-inadmissible. Therefore no parser action (including recursive
/// CALL/RETURN closure) can be skipped. Only viable tokenizer continuation
/// keys are substituted for the original key; every failure to prove those
/// conditions declines without mutating `state`.
pub(super) fn try_commit_single_state_lexer_only(
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
