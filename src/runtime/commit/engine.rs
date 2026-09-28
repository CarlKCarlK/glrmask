//! Ordinary byte/token dispatch, authoritative queue, and exact probes.

use super::admission::ActionableTerminals;
use super::admission::cached_batched_end_state_admission;
use super::admission::cached_single_end_state_may_advance;
use super::admission::end_state_may_advance;
use super::admission::end_state_may_advance_from_cache_entry;
use super::admission::wide_frontier_end_state_may_advance;
use super::advance::advance_terminal_match;
use super::controls::advance_special_token_paths;
use super::controls::merge_special_token_paths;
use super::flat_frontier::FLAT_FRONTIER_MAX_BRANCHES;
use super::flat_frontier::try_commit_flat_frontier_in_place;
use super::frontier::ParserStatesByTokenizer;
use super::frontier::coalesce_uniform_runtime_source_states;
use super::frontier::expand_runtime_product_states;
use super::frontier::finalize_pending_state;
use super::frontier::maybe_normalize_lookahead_invariant_reductions;
use super::frontier::queue_parser_ignored_reset_state;
use super::frontier::queue_parser_reset_state;
use super::frontier::queue_parser_state;
use super::lexer_only::scan_wide_frontier_lexer_only;
use super::lexer_only::try_commit_multi_state_lexer_only;
use super::lexer_only::try_commit_single_state_lexer_only;
use super::lexical::advance_uniform_disallowed_interest_only;
use super::lexical::collect_unique_actionable_matches;
use super::lexical::prune_single_initial_state_for_exec;
use super::linear::LinearFastPathResult;
use super::linear::choose_direct_linear_step;
use super::linear::commit_bytes_direct_linear_fast_path;
use super::linear::commit_bytes_fast_path;
use super::linear::commit_bytes_full_width_fast_path;
use super::linear::commit_bytes_linear_fast_path;
use super::linear::try_commit_direct_linear_in_place;
use super::small_queue::commit_bytes_language_small_queue_fast_path;
use super::small_queue::commit_bytes_small_queue_fast_path;
use super::token_bytes_for_id;
use super::tokenizer_scan::execute_tokenizer_from_state_small;
use crate::automata::lexer::Lexer;
use crate::compiler::glr::accumulator::TerminalsDisallowed;
use crate::compiler::glr::parser::ParserGSS;
use crate::compiler::glr::table::Action;
use crate::compiler::glr::table::AdmissionPolicy;
use crate::runtime::constraint::Constraint;
use crate::runtime::state::CommitBuffers;
use crate::runtime::state::ParserStateMap;
use rustc_hash::FxHashMap;

/// Cache for `advance_stacks` results, keyed by (GSS pointer, terminal).
/// Stores the key GSS alongside the result to keep its Arc alive and prevent
/// address reuse (ABA problem) within a single `commit_bytes_impl` call.
pub(super) type AdvanceResultCache = FxHashMap<(usize, u32), (ParserGSS, ParserGSS)>;

pub(super) fn finish_token_commit(state: &ParserStateMap) -> Result<(), String> {
    if state.is_empty() {
        Err("commit rejected: no valid parser states remain".to_owned())
    } else {
        Ok(())
    }
}

pub(super) fn commit_token_impl(
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
    let special_paths =
        has_special.then(|| advance_special_token_paths(constraint, state, token_id)).flatten();
    if let Some(bytes) = bytes.filter(|piece| !piece.is_empty()) {
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
            walk(constraint, &frontier, buffers, &entries[index..end], common_end, admitted);
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

pub(super) fn commit_bytes_impl(
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
                && !constraint.tokenizer.exact_dynamic_state_has_future(tokenizer_state)?
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

pub(super) fn commit_bytes_impl_inner(
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
            state.iter().map(|(&key, gss)| (key, gss.path_count_at_most(17))).collect::<Vec<_>>(),
        );
    }
    let ignore_terminal = constraint.ignore_terminal;
    // Linker control terminals are zero-width parser transitions.  The
    // allocation-free commit fast paths assume that every parser transition is
    // driven by a completed lexer terminal and therefore cannot yet preserve
    // control closure between lexemes.  Keep explicit-control tables on the
    // authoritative queue path until each fast path has its own equivalence
    // proof/implementation.
    let has_linker_controls = !constraint.table.control_terminals.is_empty()
        || constraint.uses_compact_segmented_parser_runtime();
    let direct_dynamic =
        constraint.uses_dynamic_runtime() && constraint.direct_regular_automaton.is_some();
    if state.len() <= 1 && !bufs.admission_cache.is_empty() {
        bufs.admission_cache.clear();
    }

    // A wide direct-regular frontier already carries its exact actionable
    // terminal support. Scan only that support instead of materializing every
    // tokenizer finalizer, which can be proportional to the remaining source
    // file in project-scale diff grammars.
    if !has_linker_controls && state.len() == 1 && !constraint.tokenizer_has_epsilon_transitions {
        let (&start_tokenizer_state, gss) = state.iter().next().unwrap();
        if let Some(summary) = constraint.direct_regular_wide_frontier_for_gss(gss)
            && let Some(accumulator) = gss.uniform_accumulator()
            && let Some(end_state) =
                scan_wide_frontier_lexer_only(constraint, bytes, start_tokenizer_state, summary)
        {
            let Some(updated_accumulator) =
                advance_uniform_disallowed_interest_only(constraint, &accumulator, bytes)
            else {
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
                eprintln!(
                    "[glrmask/debug][commit_path] multi_state_lexer_only states={}",
                    state.len()
                );
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
        && constraint.table.admission_policy == AdmissionPolicy::ExactSimulation
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
    if !has_linker_controls && !constraint.tokenizer_has_epsilon_transitions && state.len() == 1 {
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
                || constraint.tokenizer.matched_terminals_iter(tokenizer_state).next().is_some()
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
            eprintln!("[glrmask/debug][commit_path] recursive_lexer_only states={}", state.len(),);
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
            && let Some(step) =
                choose_direct_linear_step(constraint, gss, bytes, tokenizer_state, Some(top_state))
            && step.width == bytes.len()
            && !step.ignored
        {
            let continuation_is_inert = step.end_state.is_none_or(|end_state| {
                let future = constraint.tokenizer.possible_future_terminals(end_state);
                !end_state_may_advance(constraint, gss, end_state)
                    && !future.contains(step.terminal as usize)
            });
            if continuation_is_inert
                && let Some(Action::Shift(target, replace)) =
                    constraint.table.action(top_state, step.terminal)
                && let Some(gss) = state.values_mut().next()
            {
                let pushes = [*target];
                if gss
                    .try_apply_single_segment_stack_effect_in_place(usize::from(*replace), &pushes)
                {
                    return Ok(());
                }
            }
        }
    }

    if !has_linker_controls && state.len() == 1 {
        let (&tokenizer_state, _) = state.iter().next().unwrap();
        let exec_result = execute_tokenizer_from_state_small(constraint, bytes, tokenizer_state);
        if let Some(result) =
            commit_bytes_fast_path(constraint, state, bytes, tokenizer_state, &exec_result)
        {
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

    if !has_linker_controls && !constraint.tokenizer_has_epsilon_transitions && state.len() == 1 {
        let (&tokenizer_state, _) = state.iter().next().unwrap();
        let exec_result = execute_tokenizer_from_state_small(constraint, bytes, tokenizer_state);

        // Try fast path with pre-computed exec_result
        if let Some(result) =
            commit_bytes_fast_path(constraint, state, bytes, tokenizer_state, &exec_result)
        {
            return result;
        }

        if !exec_result.end_state.iter().copied().any(|end_state| {
            state
                .values()
                .next()
                .is_some_and(|gss| end_state_may_advance(constraint, gss, end_state))
        }) {
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
            match commit_bytes_linear_fast_path(constraint, start_gss, bytes, exec_result.clone()) {
                LinearFastPathResult::Complete(result) => match result {
                    Ok(final_gss) => {
                        state.clear();
                        state.insert(constraint.runtime_commit_initial_state(), final_gss);
                        return Ok(());
                    }
                    Err(err) => return Err(err),
                },
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
                        processing_queue
                            .resize_with(needed_queue_len, ParserStatesByTokenizer::default);
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
                            let actionable_terminals =
                                ActionableTerminals::from_gss(constraint, &gss_at_offset);
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
            let exec_result =
                execute_tokenizer_from_state_small(constraint, bytes, tokenizer_state);
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
                    execute_tokenizer_from_state_small(
                        constraint,
                        &bytes[offset..],
                        tokenizer_state,
                    )
                })
            } else {
                execute_tokenizer_from_state_small(constraint, &bytes[offset..], tokenizer_state)
            };

            if offset == 0
                && !gss_at_offset.all_accs_satisfy(|td: &TerminalsDisallowed| td.is_empty())
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
