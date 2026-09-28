//! Exact-token alternatives and zero-width control-path advancement.

use super::admission::runtime_tokenizer_is_reset_state;
use super::advance::advance_parser_stacks_if_possible;
use super::advance::advance_parser_stacks_profiled_if_possible;
use super::lexical::prune_single_initial_state_for_terminal;
use super::profile::CommitProfile;
use super::profile::PerAdvanceEntry;
use super::profile::apply_advance_profile;
use super::profiled::record_per_advance_entry;
use crate::compiler::glr::parser::AdvanceProfile;
use crate::compiler::glr::parser::ParserGSS;
use crate::runtime::constraint::Constraint;
use crate::runtime::state::ParserStateMap;

pub(in crate::runtime) fn advance_special_token_paths(
    constraint: &Constraint,
    state: &ParserStateMap,
    token_id: u32,
) -> Option<ParserGSS> {
    let mut merged = None::<ParserGSS>;

    for (&initial_state, initial_gss) in state.iter() {
        if !runtime_tokenizer_is_reset_state(constraint, initial_state) {
            continue;
        }
        for special in
            constraint.special_token_terminals.iter().filter(|special| special.token_id == token_id)
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
pub(super) struct SpecialTokenAdvanceProfile {
    pub(super) paths: Option<ParserGSS>,
    pub(super) prune_ns: u64,
    pub(super) may_check_ns: u64,
    pub(super) advance_ns: u64,
    pub(super) summary_ns: u64,
    pub(super) advances: Vec<AdvanceProfile>,
}

pub(super) fn advance_special_token_paths_profiled(
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
        for special in
            constraint.special_token_terminals.iter().filter(|special| special.token_id == token_id)
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

            let attempt = advance_parser_stacks_profiled_if_possible(
                constraint,
                &pruned,
                special.terminal_id,
            );
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

pub(super) fn apply_special_token_advance_profile(
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

pub(super) fn merge_special_token_paths(
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
        let reset_state =
            constraint.recursive_tokenizer_reset_state(leaf_index).ok_or_else(|| {
                format!("recursive special-token leaf {leaf_index} has no tokenizer reset")
            })?;
        state.merge_insert(reset_state, partition);
    }
    Ok(())
}
