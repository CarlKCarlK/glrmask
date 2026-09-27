//! Public constraint lifecycle and the private runtime implementation boundary.

use crate::automata::lexer::Lexer;
mod cache;
mod compiler_views;
mod dynamic_vocab;
mod mask_cache;
mod mask_replay;
mod observations;
mod parser;
mod parser_cache;
mod recursive_parser;
mod regular;
mod vocabulary;
mod weights;

pub use super::artifact::Constraint;
pub(crate) use super::mask_mapping::{DeltaReplayProfileStats, DenseToBufProfileStats};
pub(crate) use weights::{RuntimeTokenSetRef, RuntimeWeightRef};
pub(crate) use mask_cache::{InternalTokenMaskPrebuild, TokenMaskCachePrebuild};
#[cfg(test)]
use crate::compiler::glr::parser::ParserGSS;
#[cfg(test)]
use self::cache::INITIAL_COMMIT_PRIME_MAX_TOKENS;
#[cfg(test)]
use self::cache::initial_commit_prime_token_ids;
#[cfg(test)]
use self::mask_replay::andnot_group_buf_mask;
#[cfg(test)]
use self::mask_replay::andnot_sparse_buf_entries;
#[cfg(test)]
use self::mask_replay::or_group_buf_mask;
#[cfg(test)]
use self::mask_replay::or_sparse_buf_entries;
use crate::automata::weighted::dwa::DWA;
use crate::compiler::glr::table::TableAmbiguity;
use crate::grammar::flat::TerminalID;
use crate::runtime::state::ConstraintState;
use range_set_blaze::RangeSetBlaze;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::Mutex;

impl<'a> ConstraintState<'a> {
    /// Fill a mask directly from the lexer and parser stack, without using the
    /// parser DWA.
    pub(crate) fn fill_mask_dynamic(&self, buf: &mut [u32]) {
        crate::compiler::composition::boundary::transfer::strict_static_trap_dynamic_for_state(
            "ConstraintState::fill_mask_dynamic",
            self.constraint.uses_dynamic_runtime(),
        );
        if self.constraint.uses_compact_segmented_parser_runtime() {
            match super::dynamic_mask::try_fill_recursive_mask_shared(self, buf) {
                Ok(true) => return,
                Ok(false) => panic!("recursive composition declined its shared mask provider"),
                Err(error) => {
                    panic!("shared recursive dynamic mask generation failed: {error}");
                }
            }
        }
        super::dynamic_mask::fill_mask_dynamic(self, buf);
    }

    pub(crate) fn fill_mask_dynamic_bounded(
        &self,
        buf: &mut [u32],
        timeout_ms: u64,
    ) -> Result<(), String> {
        super::dynamic_mask::fill_mask_dynamic_bounded(self, buf, timeout_ms)
    }

}

impl Constraint {


    #[inline]
    pub(crate) fn uses_dynamic_runtime(&self) -> bool {
        matches!(
            self.runtime_backend,
            super::artifact::ConstraintRuntimeBackend::Dynamic
        )
    }

    pub(crate) fn table_ambiguous_actions(&self) -> Vec<TableAmbiguity> {
        self.table.ambiguous_actions()
    }

    pub(crate) fn table_has_ambiguity(&self) -> bool {
        self.table.has_ambiguity()
    }

    pub(crate) fn terminal_display_names(&self) -> &[String] {
        &self.terminal_display_names
    }

    pub(crate) fn terminal_display_name(&self, terminal_id: TerminalID) -> Option<&str> {
        self.terminal_display_names
            .get(terminal_id as usize)
            .map(String::as_str)
    }

    /// Create a fresh state for one generated sequence.
    pub fn start(&self) -> ConstraintState<'_> {
        crate::runtime::initialize_hot_path_config();
        if self.tokenizer_has_epsilon_transitions && !self.tokenizer.has_packed_runtime_metadata() {
            drop(self.tokenizer.all_singleton_epsilon_closures());
        }
        let state = self.initial_state_map();
        let mut state = ConstraintState {
            terminated: false,
            constraint: self,
            state,
            buffers: Default::default(),
            generation: 0,
            mask_cache: Mutex::new(None),
            mask_scratch: Arc::new(Mutex::new(crate::runtime::state::MaskScratch::for_constraint(self))),
        };
        state.prefill_mask_cache();
        state.reserve_linear_stack_hot_path();
        state
    }

    pub(crate) fn start_dynamic(&self) -> ConstraintState<'_> {
        crate::runtime::initialize_hot_path_config();
        let mut state = ConstraintState {
            terminated: false,
            constraint: self,
            state: self.initial_state_map(),
            buffers: crate::runtime::state::CommitBuffers::for_constraint(self),
            generation: 0,
            mask_cache: Mutex::new(None),
            mask_scratch: Arc::new(Mutex::new(crate::runtime::state::MaskScratch::for_constraint(self))),
        };
        state.reserve_linear_stack_hot_path();
        state
    }

    /// Return the number of `u32` words required for a packed token mask.
    pub fn mask_len(&self) -> usize {
        self.max_original_token_id()
            .map(|token_id| (token_id as usize / 32) + 1)
            .unwrap_or(0)
    }

    pub(crate) fn body_mask_len(&self) -> usize {
        self.body_max_original_token_id()
            .map(|token_id| (token_id as usize / 32) + 1)
            .unwrap_or(0)
    }

    /// Apply generation termination without changing the compiled body.
    pub(crate) fn with_end_tokens(mut self, ids: &[u32]) -> crate::Result<Self> {
        let mut ids = ids.to_vec();
        ids.sort_unstable();
        if ids.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(crate::Error::Compilation("duplicate end-token ID".to_owned()));
        }
        self.end_tokens = Arc::from(ids);
        Ok(self)
    }

    pub(crate) fn num_parser_states(&self) -> u32 {
        self.table.num_states
    }

    pub(crate) fn num_tokenizer_states(&self) -> usize {
        self.tokenizer.num_states() as usize
    }

    pub(crate) fn compute_forced_minimized_tokenizer_state_count(&self) -> usize {
        self.tokenizer.compute_forced_minimized_state_count()
    }

    pub(crate) fn parser_dwa(&self) -> &DWA {
        &self.parser_dwa
    }

    pub(crate) fn possible_matches_for_state_internal(
        &self,
        tokenizer_state: u32,
    ) -> Option<BTreeMap<TerminalID, RangeSetBlaze<u32>>> {
        // Return possible_matches in the final shared constraint-internal vocab
        // space. These ids match parser-DWA weight token ids after reconciliation.
        let mut result = BTreeMap::new();
        for terminal in self.runtime_possible_match_terminals() {
            let Some(weight) = self.runtime_possible_match_weight(terminal) else {
                continue;
            };
            let mut tokens = RangeSetBlaze::new();
            for &internal_tsid in self.internal_tsids_for_state(tokenizer_state) {
                if let Some(token_set) = weight.token_set_for_tsid(internal_tsid) {
                    tokens |= token_set.to_range_set();
                }
            }
            if !tokens.is_empty() {
                result.insert(terminal, tokens);
            }
        }
        if result.is_empty() {
            None
        } else {
            Some(result)
        }
    }
}


#[cfg(test)]
mod dense_internal_token_mask_tests;