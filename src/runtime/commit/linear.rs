//! Single-match and linear-stack commit kernels, including profiled variants.

use super::admission::ActionableTerminals;
use super::admission::batched_end_state_admitted_terminals;
use super::admission::end_state_may_advance;
use super::admission::end_state_may_advance_with_batch;
use super::admission::parser_may_advance_on;
use super::advance::advance_parser_stacks;
use super::advance::advance_parser_stacks_if_possible;
use super::advance::advance_parser_stacks_owned;
use super::advance::advance_parser_stacks_profiled;
use super::advance::apply_single_top_action_fast;
use super::advance::template_advance_enabled;
use super::advance::try_apply_action_to_carried_virtual_stack;
use super::advance::try_apply_single_top_action_in_place;
use super::flat_frontier::FlatFrontierScratch;
use super::flat_frontier::apply_terminal_to_flat_stacks;
use super::flat_frontier::flat_stack_may_advance_on_any;
use super::frontier::ParserStatesByTokenizer;
use super::frontier::finalize_pending_state;
use super::frontier::merge_parser_state;
use super::lexical::SMALL_NORMALIZED_MATCH_LINEAR_SCAN_MAX;
use super::lexical::apply_future_terminal_disallow;
use super::lexical::collect_unique_actionable_reusable_matches;
use super::lexical::for_each_relevant_matched_terminal;
use super::lexical::is_actionable_terminal;
use super::lexical::is_ignored_terminal;
use super::lexical::prune_single_initial_state_for_exec;
use super::lexical::state_has_nonempty_accumulators;
use super::profile::CommitProfile;
use super::profile::PerAdvanceEntry;
use super::profile::apply_advance_profile;
use super::profile::fast_action_advance_profile;
use super::profiled::record_per_advance_entry;
use super::tokenizer_scan;
use super::tokenizer_scan::execute_tokenizer_from_state_small;
use super::tokenizer_scan::execute_tokenizer_reusable;
use crate::automata::lexer::Lexer;
use crate::automata::lexer::tokenizer::TokenizerExecResult;
use crate::automata::lexer::tokenizer::TokenizerStateSet;
use crate::compiler::glr::accumulator::TerminalsDisallowed;
use crate::compiler::glr::parser::AdvanceProfile;
use crate::compiler::glr::parser::ParserGSS;
use crate::compiler::glr::table::Action;
use crate::runtime::constraint::Constraint;
use crate::runtime::state::INLINE_PARSER_STATE_CAPACITY;
use crate::runtime::state::ParserStateMap;
use smallvec::SmallVec;

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
pub(super) fn commit_bytes_fast_path(
    constraint: &Constraint,
    state: &mut ParserStateMap,
    bytes: &[u8],
    tokenizer_state: u32,
    exec_result: &TokenizerExecResult,
) -> Option<Result<(), String>> {
    let gss = state.values().next().unwrap();
    let ignore_terminal = constraint.ignore_terminal;
    let has_linker_controls = !constraint.table.control_terminals.is_empty()
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
    if !has_linker_controls && all_accs_empty && !template_advance_enabled() {
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
            return Some(Err("commit rejected: no valid parser states remain".to_string()));
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
        && !template_advance_enabled()
        && let Some(top_state) = pruned_gss.single_exclusive_top_value()
        && let Some(action) = constraint.table.action(top_state, terminal)
        && let Some(advanced) =
            apply_single_top_action_fast(constraint, &pruned_gss, top_state, terminal, action)
    {
        advanced
    } else {
        advance_parser_stacks_owned(constraint, pruned_gss.clone(), terminal)
    };
    let mut produced_state = false;
    if !advanced.is_empty() {
        let advanced = apply_future_terminal_disallow(constraint, &exec_result, terminal, advanced);
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
        return Some(Err("commit rejected: no valid parser states remain".to_string()));
    }
    Some(Ok(()))
}

pub(super) fn commit_bytes_full_width_fast_path(
    constraint: &Constraint,
    state: &mut ParserStateMap,
    bytes: &[u8],
) -> Option<Result<(), String>> {
    let has_linker_controls = !constraint.table.control_terminals.is_empty()
        || constraint.uses_compact_segmented_parser_runtime();
    if constraint.tokenizer_has_epsilon_transitions && state_has_nonempty_accumulators(state) {
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
                && !template_advance_enabled()
                && let Some(top_state) = pruned_gss.single_exclusive_top_value()
                && let Some(action) = constraint.table.action(top_state, terminal)
                && let Some(advanced) = apply_single_top_action_fast(
                    constraint,
                    &pruned_gss,
                    top_state,
                    terminal,
                    action,
                ) {
                advanced
            } else {
                advance_parser_stacks_if_possible(constraint, &pruned_gss, terminal)?
            };
            if advanced.is_empty() {
                continue;
            }
            let advanced =
                apply_future_terminal_disallow(constraint, &exec_result, terminal, advanced);
            if !advanced.is_empty() {
                merge_parser_state(
                    &mut output,
                    constraint.runtime_commit_initial_state(),
                    advanced,
                );
            }
        }

        let admitted_end_state_terminals =
            batched_end_state_admitted_terminals(constraint, &pruned_gss, &exec_result.end_state);
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
        return Some(Err("commit rejected: no valid parser states remain".to_string()));
    }
    *state = new_state;
    Some(Ok(()))
}

pub(super) enum LinearFastPathResult {
    Complete(Result<ParserGSS, String>),
    Continue { gss: ParserGSS, offset: usize },
    Restart,
}

pub(super) struct DirectLinearStep {
    pub(super) width: usize,
    pub(super) terminal: u32,
    pub(super) ignored: bool,
    pub(super) end_state: Option<u32>,
}

pub(super) fn choose_direct_linear_step(
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
                return Some(DirectLinearStep { width, terminal, ignored: false, end_state: None });
            }
        }
    }

    let (width, terminal, ignored) = chosen?;
    let end_state = consumed_all.then_some(tokenizer_state);

    Some(DirectLinearStep { width, terminal, ignored, end_state })
}

pub(super) fn commit_bytes_direct_linear_fast_path(
    constraint: &Constraint,
    start_gss: ParserGSS,
    bytes: &[u8],
    start_tokenizer_state: u32,
    mut profile: Option<&mut CommitProfile>,
) -> Option<LinearFastPathResult> {
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
                    && !constraint.table.advance_row_intersects(
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
                    if let (Some(profile), Some(start)) =
                        (profile.as_deref_mut(), materialize_start)
                    {
                        profile.linear_fast_path_materialize_ns +=
                            start.elapsed().as_nanos() as u64;
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
                    if let (Some(profile), Some(start)) =
                        (profile.as_deref_mut(), materialize_start)
                    {
                        profile.linear_fast_path_materialize_ns +=
                            start.elapsed().as_nanos() as u64;
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
                        && !constraint.table.advance_row_intersects(
                            top_state,
                            constraint.tokenizer.possible_future_terminals(end_state),
                        )
                        && !constraint
                            .tokenizer
                            .possible_future_terminals(end_state)
                            .contains(step.terminal as usize)
                }) {
                constraint.table.action(top_state, step.terminal)
            } else {
                None
            };
            if let (Some(profile), Some(start)) = (profile.as_deref_mut(), action_lookup_start) {
                profile.linear_fast_path_action_lookup_ns += start.elapsed().as_nanos() as u64;
            }
            if let Some(action) = carried_action {
                let apply_action_start = profile.as_ref().map(|_| std::time::Instant::now());
                if !template_advance_enabled()
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
            let advanced = if !template_advance_enabled()
                && let Some(top_state) = gss.single_exclusive_top_value()
                && let Some(action) = constraint.table.action(top_state, step.terminal)
                && let Some(advanced) =
                    apply_single_top_action_fast(constraint, &gss, top_state, step.terminal, action)
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
            gss = apply_future_terminal_disallow(constraint, &exec_result, step.terminal, advanced);
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

pub(super) fn commit_bytes_fast_path_profiled(
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
    let all_accs_empty =
        no_end_state && gss.all_accs_satisfy(|td: &TerminalsDisallowed| td.is_empty());

    if all_accs_empty && !template_advance_enabled() {
        if let Some(top_state) = gss.single_exclusive_top_value() {
            if let Some(Action::Shift(target, is_replace)) =
                constraint.table.action(top_state, terminal)
            {
                let advance_start = Instant::now();
                let shifted =
                    if *is_replace { gss.popn(1).push(*target) } else { gss.push(*target) };
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
                            vstack_len: gss
                                .try_virtual_stack()
                                .map_or(0, |vstack| vstack.len() as u32),
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
            if !template_advance_enabled()
                && let Some(Action::StackShifts(shifts)) =
                    constraint.table.action(top_state, terminal)
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

pub(super) fn commit_bytes_linear_fast_path(
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
                && !is_actionable_terminal(actionable_terminals.as_ref(), constraint, matched.id)
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
                    && !constraint.table.advance_row_intersects(
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
            if !template_advance_enabled()
                && let Some(stack) = carried_stack.as_mut()
                && let Some(top_state) = stack.top().copied()
                && let Some(Action::Shift(target, is_replace)) =
                    constraint.table.action(top_state, terminal)
                && exec_result.end_state.iter().copied().all(|end_state| {
                    end_state != constraint.runtime_commit_initial_state()
                        && !constraint.table.advance_row_intersects(
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

            let fast_advanced = if !template_advance_enabled()
                && let Some(top_state) = gss.single_exclusive_top_value()
                && let Some(action) = constraint.table.action(top_state, terminal)
            {
                apply_single_top_action_fast(constraint, &gss, top_state, terminal, action)
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

pub(super) fn commit_bytes_linear_fast_path_profiled(
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
                && !is_actionable_terminal(actionable_terminals.as_ref(), constraint, matched.id)
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
        profile.linear_fast_path_end_state_check_ns += end_state_start.elapsed().as_nanos() as u64;

        if !ignored {
            let fast_start = Instant::now();
            let fast_advanced = if !template_advance_enabled()
                && let Some(top_state) = gss.single_exclusive_top_value()
                && let Some(action) = constraint.table.action(top_state, terminal)
                && let Some(advanced) =
                    apply_single_top_action_fast(constraint, &gss, top_state, terminal, action)
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

pub(super) fn try_commit_direct_linear_in_place(
    constraint: &Constraint,
    state: &mut ParserStateMap,
    bytes: &[u8],
    original: &mut Vec<u32>,
    work: &mut Vec<u32>,
    tokenizer_scratch: &mut tokenizer_scan::ReusableTokenizerExecScratch,
    frontier: &mut FlatFrontierScratch,
) -> Option<Result<(), String>> {
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
                return Some(Err("commit rejected: no valid parser states remain".to_string()));
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
                    return Some(Err("commit rejected: no valid parser states remain".to_string()));
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
