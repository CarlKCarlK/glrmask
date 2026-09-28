//! Token-commit lifecycle and internal facade.
//!
//! Root termination, cache generation, and diagnostic policies stay explicit.
//! Execution kernels live with their eligibility rules and scratch structures.

mod admission;
mod advance;
mod assertions;
mod controls;
mod engine;
mod flat_frontier;
mod frontier;
mod lexer_only;
mod lexical;
mod linear;
mod profiled;
mod small_queue;

use self::profile::CommitProfile;
use self::profile::PerAdvanceEntry;
pub(crate) use self::template_advance::TemplateAdvanceRuntime;
use crate::compiler::glr::parser::ParserGSS;
use crate::runtime::constraint::Constraint;
use crate::runtime::state::ConstraintState;
pub(crate) use admission::exact_admitted_terminals_for_candidates;
#[allow(unused_imports)] // Retain the internal probe path for optional tooling.
pub(crate) use advance::advance_parser_stacks_if_possible;
pub(crate) use advance::advance_parser_stacks_table_exact;
use advance::template_advance_enabled;
use advance::validate_template_advance_enabled;
use assertions::COMMIT_ASSERT_FAST_PATH_EQUIVALENCE;
use assertions::COMMIT_ASSERT_MASK_EQUIVALENCE;
use assertions::assert_commit_oracles;
use assertions::commit_assertion_flags;
use assertions::profile_allow_fast_paths;
use assertions::snapshot_mask_membership;
use assertions::token_in_mask;
use controls::SpecialTokenAdvanceProfile;
pub(super) use controls::advance_special_token_paths;
use controls::advance_special_token_paths_profiled;
use controls::apply_special_token_advance_profile;
use controls::merge_special_token_paths;
#[allow(unused_imports)] // Retain the internal probe path for optional tooling.
pub(crate) use engine::admissible_byte_token_candidates_from_state_exact;
#[allow(unused_imports)] // Retain the internal probe path for optional tooling.
pub(crate) use engine::advance_bytes_from_state_exact;
use engine::commit_bytes_impl;
use engine::commit_token_impl;
use engine::finish_token_commit;
pub(crate) use engine::prime_initial_commits;
#[allow(unused_imports)] // Retain the internal probe path for optional tooling.
pub(crate) use engine::token_admissible_from_state_exact;
pub(crate) use flat_frontier::FlatFrontierScratch;
use frontier::coalesce_uniform_runtime_source_states;
use frontier::expand_runtime_product_states;
use profiled::clear_state_on_commit_error;
use profiled::commit_bytes_impl_profiled;
use profiled::final_stacks;
pub(crate) use small_queue::SmallCommitQueueScratch;
pub(crate) use template_advance::advance_stacks_template_dfa;

#[cfg(test)]
use crate::automata::lexer::Lexer;
#[cfg(test)]
use self::tokenizer_scan::execute_tokenizer_from_state_small;
#[cfg(test)]
use crate::compiler::glr::accumulator::TerminalsDisallowed;
#[cfg(test)]
use crate::compiler::glr::parser::stack_may_advance_on;
#[cfg(test)]
use crate::compiler::glr::table::Action;
#[cfg(test)]
use crate::compiler::glr::table::AdmissionPolicy;
#[cfg(test)]
use crate::compiler::glr::table::row::ActionRow;
#[cfg(test)]
use crate::runtime::state::CommitBuffers;
#[cfg(test)]
use crate::runtime::state::INLINE_PARSER_STATE_CAPACITY;
#[cfg(test)]
use crate::runtime::state::LINEAR_STACK_RESERVE;
#[cfg(test)]
use crate::runtime::state::ParserAdmissionCacheEntry;
#[cfg(test)]
use crate::runtime::state::ParserStateMap;
#[cfg(test)]
use admission::batched_end_state_admitted_terminals;
#[cfg(test)]
use admission::cached_batched_end_state_admission;
#[cfg(test)]
use admission::end_state_may_advance;
#[cfg(test)]
use admission::end_state_may_advance_from_cache_entry;
#[cfg(test)]
use admission::end_state_may_advance_with_batch;
#[cfg(test)]
use admission::runtime_terminal_count;
#[cfg(test)]
use admission::single_conditional_candidate;
#[cfg(test)]
use advance::advance_parser_stacks;
#[cfg(test)]
use advance::advance_parser_stacks_profiled_if_possible;
#[cfg(test)]
use advance::try_apply_action_to_carried_virtual_stack;
#[cfg(test)]
use assertions::assert_commit_fast_path_equivalence;
#[cfg(test)]
use assertions::canonical_commit_state_for_equivalence_assert;
#[cfg(test)]
use assertions::commit_token_no_fast_path_reference;
#[cfg(test)]
use flat_frontier::FlatInlineStack;
#[cfg(test)]
use flat_frontier::apply_flat_stack_effect;
#[cfg(test)]
use flat_frontier::try_commit_flat_frontier_in_place;
#[cfg(test)]
use lexical::prune_single_initial_state_for_exec;
#[cfg(test)]
use small_queue::LANGUAGE_QUEUE_MIN_NODES;
#[cfg(test)]
use small_queue::LANGUAGE_QUEUE_MIN_PATHS;
#[cfg(test)]
use small_queue::LANGUAGE_QUEUE_MIN_TOP_VALUES;
#[cfg(test)]
use small_queue::has_multiple_actionable_terminal_boundaries;
#[cfg(test)]
use small_queue::language_queue_node_count_at_most;
#[cfg(test)]
use small_queue::language_queue_path_count_at_most;
#[cfg(test)]
use small_queue::language_queue_top_value_count_at_most;
#[cfg(test)]
use smallvec::SmallVec;
#[cfg(test)]
use std::collections::BTreeMap;

pub(crate) mod profile;

mod mask_reuse;

mod template_advance;

pub(crate) mod tokenizer_scan;

fn token_bytes_for_id(constraint: &Constraint, token_id: u32) -> Option<&[u8]> {
    constraint
        .token_bytes_dense
        .get(token_id as usize)
        .and_then(|bytes| bytes.as_deref())
        .or_else(|| constraint.token_bytes_for_id(token_id))
}

pub(crate) fn initialize_runtime_config() {
    let _ = template_advance_enabled();
    let _ = validate_template_advance_enabled();
    let _ = commit_assertion_flags();
}

impl<'a> ConstraintState<'a> {
    pub(crate) fn knows_token_id(&self, token_id: u32) -> bool {
        token_bytes_for_id(self.constraint, token_id).is_some()
            || self.constraint.has_special_token_id(token_id)
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
    pub fn commit_token(&mut self, token_id: u32) -> crate::Result<()> {
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

    pub(crate) fn commit_token_raw(&mut self, token_id: u32) -> Result<(), String> {
        if let Some(result) = self.commit_end_token(token_id) {
            return result;
        }
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
        if let Some(result) = self.commit_end_token(token_id) {
            return result;
        }
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
        let mut profile = if let Some(bytes) = bytes.filter(|piece| !piece.is_empty()) {
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
        let result = special_merge_result.and_then(|()| finish_token_commit(&self.state));
        let reuse_started = std::time::Instant::now();
        self.finish_commit_generation(mask_state_before_commit, result.is_ok());
        profile.mask_cache_reuse_ns = reuse_started.elapsed().as_nanos() as u64;
        profile.total_ns = total_started_at.elapsed().as_nanos() as u64;
        assert_commit_oracles(
            constraint,
            token_id,
            bytes,
            was_in_mask,
            equivalence_reference,
            &self.state,
            result.is_ok(),
        );
        result.map(|()| profile)
    }

    pub(crate) fn commit_token_per_advance(
        &mut self,
        token_id: u32,
    ) -> Result<(Vec<PerAdvanceEntry>, Vec<(u32, Vec<Vec<u32>>)>, CommitProfile), String> {
        if let Some(result) = self.commit_end_token(token_id) {
            return result
                .map(|()| (Vec::new(), final_stacks(&self.state), CommitProfile::default()));
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
        let mut profile = if let Some(bytes) = bytes.filter(|piece| !piece.is_empty()) {
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
        let result = special_merge_result.and_then(|()| finish_token_commit(&self.state));
        let reuse_started = std::time::Instant::now();
        self.finish_commit_generation(mask_state_before_commit, result.is_ok());
        profile.mask_cache_reuse_ns = reuse_started.elapsed().as_nanos() as u64;
        profile.total_ns = total_started_at.elapsed().as_nanos() as u64;
        assert_commit_oracles(
            constraint,
            token_id,
            bytes,
            was_in_mask,
            equivalence_reference,
            &self.state,
            result.is_ok(),
        );
        result.map(|()| (advances, final_stacks(&self.state), profile))
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
mod tests;
