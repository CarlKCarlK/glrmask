//! Instrumented queue execution and per-advance observation.

use super::admission::ActionableTerminals;
use super::admission::batched_end_state_admitted_terminals;
use super::admission::end_state_may_advance;
use super::admission::end_state_may_advance_with_batch;
use super::advance::advance_parser_stacks_profiled_if_possible;
use super::frontier::ParserStatesByTokenizer;
use super::frontier::coalesce_uniform_runtime_source_states;
use super::frontier::expand_runtime_product_states;
use super::frontier::finalize_pending_state;
use super::frontier::maybe_normalize_lookahead_invariant_reductions;
use super::frontier::queue_parser_ignored_reset_state;
use super::frontier::queue_parser_reset_state;
use super::frontier::queue_parser_state;
use super::lexical::apply_future_terminal_disallow;
use super::lexical::collect_unique_actionable_matches;
use super::lexical::prune_single_initial_state_for_exec;
use super::linear::LinearFastPathResult;
use super::linear::commit_bytes_direct_linear_fast_path;
use super::linear::commit_bytes_fast_path_profiled;
use super::linear::commit_bytes_linear_fast_path_profiled;
use super::profile::CommitProfile;
use super::profile::PerAdvanceEntry;
use super::profile::apply_advance_profile;
use super::tokenizer_scan::InitialCommitScan;
use super::tokenizer_scan::execute_tokenizer_from_state_small;
use crate::compiler::glr::accumulator::TerminalsDisallowed;
use crate::compiler::glr::parser::AdvanceProfile;
use crate::compiler::glr::parser::ParserGSS;
use crate::runtime::constraint::Constraint;
use crate::runtime::state::CommitBuffers;
use crate::runtime::state::ParserStateMap;
use std::collections::BTreeMap;

pub(super) fn parser_stacks_only(gss: &ParserGSS) -> Vec<Vec<u32>> {
    gss.to_stacks(4_096)
        .expect("stack enumeration exceeded explicit limit")
        .into_iter()
        .map(|(stack, _)| stack)
        .collect()
}

pub(super) fn record_per_advance_entry(
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

pub(super) fn commit_bytes_impl_profiled(
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

pub(super) fn commit_bytes_impl_profiled_inner(
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
    let mut profile =
        CommitProfile { n_tokenizer_states: state.len() as u64, ..CommitProfile::default() };

    if bytes.is_empty() {
        profile.total_ns = total_start.elapsed().as_nanos() as u64;
        return Ok(profile);
    }

    let ignore_terminal = constraint.ignore_terminal;
    // Flat fast paths cannot preserve the zero-width CALL/RETURN closure of
    // linked components. Use the same admission rule as ordinary commitment.
    let allow_fast_paths = allow_fast_paths
        && constraint.table.control_terminals.is_empty()
        && !constraint.uses_compact_segmented_parser_runtime();

    if allow_fast_paths && constraint.tokenizer_has_epsilon_transitions && state.len() == 1 {
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
            let linear_fast_path_eligible =
                !exec_result.end_state.iter().copied().any(|end_state| {
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
                let start_gss =
                    if current_gss.all_accs_satisfy(|td: &TerminalsDisallowed| td.is_empty()) {
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
                let linear_advances_sink =
                    if advances.is_some() { Some(&mut linear_advances) } else { None };
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
                        let mut processing_queue: Vec<ParserStatesByTokenizer> = (0
                            ..needed_queue_len)
                            .map(|_| ParserStatesByTokenizer::default())
                            .collect();
                        processing_queue[offset] = std::mem::take(state).into_iter().collect();

                        let mut queue_offset = offset;
                        while queue_offset < needed_queue_len {
                            if processing_queue[queue_offset].is_empty() {
                                queue_offset += 1;
                                continue;
                            }

                            let states_to_process =
                                std::mem::take(&mut processing_queue[queue_offset]);
                            for (tokenizer_state, gss_at_offset) in states_to_process {
                                profile.n_queue_entries += 1;

                                let actionable_start = Instant::now();
                                let actionable_terminals =
                                    ActionableTerminals::from_gss(constraint, &gss_at_offset);
                                profile.actionable_ns +=
                                    actionable_start.elapsed().as_nanos() as u64;

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
                                    profile.queue_enqueue_ns +=
                                        enqueue_start.elapsed().as_nanos() as u64;
                                }

                                let may_start = Instant::now();
                                let admitted_end_state_terminals =
                                    batched_end_state_admitted_terminals(
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
                                        profile.may_advance_ns +=
                                            may_start.elapsed().as_nanos() as u64;
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
                                    profile.queue_enqueue_ns +=
                                        enqueue_start.elapsed().as_nanos() as u64;
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
                            return Err(
                                "commit rejected: no valid parser states remain".to_string()
                            );
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
                    execute_tokenizer_from_state_small(
                        constraint,
                        &bytes[offset..],
                        tokenizer_state,
                    )
                })
            } else {
                execute_tokenizer_from_state_small(constraint, &bytes[offset..], tokenizer_state)
            };
            let queue_exec_elapsed = exec_start.elapsed().as_nanos() as u64;
            profile.queue_exec_ns += queue_exec_elapsed;
            profile.exec_ns += queue_exec_elapsed;

            if offset == 0
                && !gss_at_offset.all_accs_satisfy(|td: &TerminalsDisallowed| td.is_empty())
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

pub(super) fn final_stacks(state: &ParserStateMap) -> Vec<(u32, Vec<Vec<u32>>)> {
    let mut grouped = BTreeMap::<u32, Vec<Vec<u32>>>::new();
    for (&tokenizer_state, gss) in state.iter() {
        grouped.entry(tokenizer_state).or_default().extend(parser_stacks_only(gss));
    }
    for stacks in grouped.values_mut() {
        stacks.sort();
        stacks.dedup();
    }
    grouped.into_iter().collect()
}

pub(super) fn clear_state_on_commit_error<T>(
    state: &mut ParserStateMap,
    result: Result<T, String>,
) -> Result<T, String> {
    if result.is_err() {
        state.clear();
    }
    result
}
