//! Runtime-coordinate normalization and shared parser-frontier operations.

use crate::automata::lexer::Lexer;
use crate::compiler::glr::parser::ParserGSS;
use crate::compiler::glr::parser::normalize_lookahead_invariant_reductions;
use crate::runtime::constraint::Constraint;
use crate::runtime::state::INLINE_PARSER_STATE_CAPACITY;
use crate::runtime::state::ParserStateMap;
use rustc_hash::FxHashMap;

pub(super) type ParserStatesByTokenizer = FxHashMap<u32, ParserGSS>;

/// Restore the exact source-tokenizer coordinate before commit.
///
/// Invariant: all duplicate parser alternatives under one product key came
/// from the same source-state subset. The historical flat frontier decides
/// continuation viability from the union of alternatives sharing a lexer key,
/// then carries each alternative independently. Reconstruct exactly that
/// relation: compute one viable source subset from the merged group, and copy
/// every original alternative across that common subset.
pub(super) fn expand_runtime_product_states(constraint: &Constraint, state: &mut ParserStateMap) {
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
        let group_end =
            old[index..].partition_point(|(candidate, _)| *candidate == tokenizer_state) + index;
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

        debug_assert!(group.iter().all(|(_, gss)| gss.all_accs_satisfy(|acc| acc.is_empty())));
        let Some(source_states) = constraint.runtime_product_source_states(tokenizer_state) else {
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
pub(super) fn coalesce_uniform_runtime_source_states(
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
    let mut eligible =
        representative.iter().all(|(_, gss)| gss.all_accs_satisfy(|acc| acc.is_empty()));
    let mut index = first_end;
    while eligible && index < old.len() {
        let key = old[index].0;
        let end = old[index..].partition_point(|(candidate, _)| *candidate == key) + index;
        let alternatives = &old[index..end];
        if alternatives.len() != representative.len()
            || alternatives.iter().any(|(_, gss)| !gss.all_accs_satisfy(|acc| acc.is_empty()))
        {
            eligible = false;
            break;
        }
        let mut used = [false; INLINE_PARSER_STATE_CAPACITY];
        for (_, expected) in representative {
            let Some(match_index) =
                alternatives.iter().enumerate().position(|(candidate_index, (_, candidate))| {
                    !used[candidate_index] && (expected.ptr_eq(candidate) || expected == candidate)
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
                source_states, product_state, first_end,
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

pub(super) fn merge_parser_state(
    states: &mut ParserStatesByTokenizer,
    tokenizer_state: u32,
    gss: ParserGSS,
) {
    states
        .entry(tokenizer_state)
        .and_modify(|existing| *existing = existing.merge(&gss))
        .or_insert(gss);
}

pub(super) fn queue_parser_state(
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

pub(super) fn queue_parser_reset_state(
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

pub(super) fn queue_parser_ignored_reset_state(
    constraint: &Constraint,
    processing_queue: &mut [ParserStatesByTokenizer],
    pending_state: &mut ParserStatesByTokenizer,
    new_offset: usize,
    total_len: usize,
    gss: ParserGSS,
) -> bool {
    let gss = if constraint.uses_compact_segmented_parser_runtime() {
        constraint.close_compact_segmented_parser(&gss).unwrap_or(gss)
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

pub(super) fn finalize_pending_state(
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

#[inline]
pub(super) fn maybe_normalize_lookahead_invariant_reductions(
    constraint: &Constraint,
    state: &mut ParserStateMap,
) {
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
