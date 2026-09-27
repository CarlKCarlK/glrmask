//! Exact mask-only lexer coordinates and terminal observation proofs.

use super::{DynamicMaskVocab, FastTokenizerTransitions};
use crate::automata::lexer::Lexer;
use crate::automata::lexer::runtime_repeat_product::VirtualBinaryRepeatIntersectionMaskProjection;
use crate::automata::lexer::runtime_unit_repeat::VirtualZeroMinUnitRepeatMaskProjection;
use crate::automata::lexer::tokenizer::TerminalProjectedQuotient;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::lexer::tokenizer::VirtualResidualMaskProjection;
use crate::compiler::stages::id_map_and_terminal_dwa::classify::VocabPartitionDfa;
use crate::ds::bitset::BitSet;
use crate::ds::u8set::U8Set;
use crate::grammar::flat::TerminalID;
use rustc_hash::FxHashMap;
use rustc_hash::FxHashSet;
use smallvec::SmallVec;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;
#[derive(Debug, Clone, Default)]
pub(crate) struct DynamicBoundedObservationSets {
    pub(super) pool: Arc<[U8Set]>,
    pub(super) horizon16: Arc<[u32]>,
    pub(super) horizon64: Arc<[u32]>,
}

impl DynamicBoundedObservationSets {
    pub(crate) fn from_raw(horizon16: Box<[U8Set]>, horizon64: Box<[U8Set]>) -> Self {
        debug_assert_eq!(horizon16.len(), horizon64.len());
        let mut ids = FxHashMap::<U8Set, u32>::default();
        let mut pool = Vec::<U8Set>::new();
        let mut intern = |set: U8Set| -> u32 {
            if let Some(&id) = ids.get(&set) {
                return id;
            }
            let id = pool.len() as u32;
            pool.push(set);
            ids.insert(set, id);
            id
        };
        let horizon16 = horizon16
            .iter()
            .copied()
            .map(&mut intern)
            .collect::<Vec<_>>();
        let horizon64 = horizon64
            .iter()
            .copied()
            .map(&mut intern)
            .collect::<Vec<_>>();
        Self {
            pool: Arc::from(pool),
            horizon16: Arc::from(horizon16),
            horizon64: Arc::from(horizon64),
        }
    }

    #[inline]
    pub(crate) fn safe_bytes(&self, state: u32, required_horizon: u32) -> Option<U8Set> {
        let ids = if required_horizon <= 16 {
            self.horizon16.as_ref()
        } else if required_horizon <= 64 {
            self.horizon64.as_ref()
        } else {
            return None;
        };
        let id = *ids.get(state as usize)? as usize;
        self.pool.get(id).copied()
    }

    #[inline]
    pub(crate) fn state_count(&self) -> usize {
        self.horizon16.len()
    }

    #[inline]
    pub(crate) fn unique_set_count(&self) -> usize {
        self.pool.len()
    }
}

impl DynamicMaskVocab {
    pub(super) fn build_full_walk_fast_transitions(
        tokenizer: &Tokenizer,
    ) -> Option<FastTokenizerTransitions> {
        // Scalar-dispatch projections deliberately execute through the lazy
        // physical-row/subset cache. A complete Flat16 slab is pure derived
        // acceleration and is too expensive to materialize before the first
        // mask; the lazy executor builds only rows actually reached by vocab
        // traversal. Ordinary deterministic tokenizers still use the dense
        // table below.
        if tokenizer.has_any_virtual_runtime() || tokenizer.has_epsilon_transitions() {
            return None;
        }
        FastTokenizerTransitions::full_walk_dense_for(
            tokenizer,
            Self::FULL_WALK_DENSE_TRANSITION_BYTES,
        )
    }

    /// Prepare the dense transition cells used exclusively by the strict full
    /// vocabulary walker. The mask projection, when present, is the exact
    /// coordinate masking advances; otherwise use the source runtime tokenizer.
    /// These tables are derived runtime data and are deliberately not serialized.
    pub(crate) fn prepare_full_walk_fast_transitions(&mut self, source: &Tokenizer) {
        // Every operation that changes the active mask tokenizer either builds
        // this derived table for the new coordinate or explicitly clears it.
        // Dynamic finalization calls this preparation step again after choosing
        // the final coordinate, so rebuilding an already-present table here is
        // duplicate work (and is material for large finite residual projections).
        if self.mask_tokenizer_fast_transitions.is_some() {
            return;
        }
        let tokenizer = self
            .mask_determinized_tokenizer
            .as_deref()
            .or(self.mask_tokenizer.as_deref())
            .unwrap_or(source);
        self.mask_tokenizer_fast_transitions = Self::build_full_walk_fast_transitions(tokenizer);
    }

    /// Prepare a deterministic execution representation for the exact finite
    /// lexer coordinate used by mask generation. The source constraint lexer
    /// remains unchanged; this is analogous to choosing Flat16 versus Flat32
    /// for the strict mask walk. If the derived representation does not fit,
    /// the strict walker executes the epsilon-NFA coordinate directly.
    pub(crate) fn prepare_mask_execution(
        &mut self,
        source_tokenizer: &Tokenizer,
        horizon: usize,
    ) -> bool {
        // Bound eager work by the same memory budget that decides whether the
        // result can use Flat32. This avoids introducing an independent
        // state-count determinization policy on top of lexer compilation.
        const CELL_BYTES: usize = std::mem::size_of::<u32>();
        const ALPHABET: usize = 256;
        let state_limit = Self::FULL_WALK_DENSE_TRANSITION_BYTES / (CELL_BYTES * ALPHABET);
        let transition_limit = state_limit.saturating_mul(ALPHABET);
        let source = self.mask_tokenizer.as_deref().unwrap_or(source_tokenizer);
        if source.has_any_virtual_runtime() {
            return false;
        }
        if !source.has_epsilon_transitions() {
            self.mask_determinized_tokenizer = None;
            self.mask_projection_to_determinized = Arc::from(Vec::<u32>::new());
            self.mask_tokenizer_fast_transitions = Self::build_full_walk_fast_transitions(source);
            return self.mask_tokenizer_fast_transitions.is_some();
        }
        let Some((built, source_to_determinized)) = source
            .try_reusing_horizon_determinization_all_starts(horizon, state_limit, transition_limit)
            .or_else(|| {
                source.try_horizon_determinization_all_starts(
                    horizon,
                    state_limit,
                    transition_limit,
                )
            })
        else {
            return false;
        };
        let source_subsets = built.source_subsets;
        let deterministic = built.tokenizer;
        let Some(fast) = Self::build_full_walk_fast_transitions(&deterministic) else {
            return false;
        };
        if self.mask_tokenizer.is_none() {
            // Directly-derived source execution coordinate. Represent it as
            // the ordinary source->mask quotient so all existing runtime
            // metadata sees the same coordinate, and retain exact subset
            // provenance for root/branch unions.
            self.set_mask_tokenizer_quotient(deterministic, source_to_determinized);
            self.set_mask_tokenizer_source_subsets(source_subsets);
            return true;
        }
        self.mask_determinized_tokenizer = Some(Arc::new(deterministic));
        self.mask_projection_to_determinized = Arc::from(source_to_determinized);
        self.mask_tokenizer_fast_transitions = Some(fast);
        // The derivative determinizer subsets are expressed in the finite
        // projection coordinate. Retain them so multiple exact lexer states
        // carrying one parser object can be coalesced into an already-built
        // execution subset at the beginning of a mask walk.
        self.set_mask_tokenizer_source_subsets(source_subsets);
        true
    }

    #[inline]
    pub(super) fn active_projected_terminal_quotients(
        &self,
    ) -> &[(TerminalID, Arc<TerminalProjectedQuotient>)] {
        if self.projected_terminal_quotients_prepared {
            self.projected_terminal_quotients.as_ref()
        } else {
            self.runtime_projected_terminal_quotients
                .get()
                .map(Arc::as_ref)
                .unwrap_or(&[])
        }
    }

    pub(crate) fn set_mask_tokenizer_quotient(
        &mut self,
        tokenizer: Tokenizer,
        full_to_mask_state: Vec<u32>,
    ) {
        debug_assert!(!full_to_mask_state.is_empty());
        debug_assert!(
            full_to_mask_state
                .iter()
                .all(|&state| state < tokenizer.num_states())
        );
        self.mask_tokenizer_fast_transitions = Self::build_full_walk_fast_transitions(&tokenizer);
        self.mask_tokenizer = Some(Arc::new(tokenizer));
        self.mask_determinized_tokenizer = None;
        self.mask_projection_to_determinized = Arc::from(Vec::<u32>::new());
        self.full_to_mask_state = Arc::from(full_to_mask_state);
        self.mask_state_source_subsets = Arc::from(Vec::<Arc<[u32]>>::new());
        self.mask_source_subset_to_state = Arc::new(FxHashMap::default());
        self.virtual_unit_repeat_projection = None;
        self.virtual_repeat_intersection_projections.clear();
        self.virtual_residual_projections.clear();
    }

    pub(crate) fn set_mask_tokenizer_source_subsets(&mut self, source_subsets: Vec<Box<[u32]>>) {
        let Some(tokenizer) = self.mask_runtime_tokenizer() else {
            self.mask_state_source_subsets = Arc::from(Vec::<Arc<[u32]>>::new());
            self.mask_source_subset_to_state = Arc::new(FxHashMap::default());
            return;
        };
        if source_subsets.len() != tokenizer.num_states() as usize {
            self.mask_state_source_subsets = Arc::from(Vec::<Arc<[u32]>>::new());
            self.mask_source_subset_to_state = Arc::new(FxHashMap::default());
            return;
        }
        let by_state = source_subsets
            .into_iter()
            .map(Arc::<[u32]>::from)
            .collect::<Vec<_>>();
        let mut by_subset = FxHashMap::default();
        by_subset.reserve(by_state.len());
        for (state, subset) in by_state.iter().enumerate() {
            by_subset.insert(Arc::clone(subset), state as u32);
        }
        self.mask_state_source_subsets = Arc::from(by_state);
        self.mask_source_subset_to_state = Arc::new(by_subset);
    }

    #[inline]
    pub(crate) fn has_mask_tokenizer_source_subsets(&self) -> bool {
        !self.mask_source_subset_to_state.is_empty()
            // A derivative of an intermediate mask projection records subsets
            // in projection-state coordinates, not exact source-tokenizer
            // coordinates. It remains useful for unioning execution states in
            // the strict walker, but must not be queried with ConstraintState
            // source lexer states.
            && !(self.mask_tokenizer.is_some() && self.mask_determinized_tokenizer.is_some())
    }

    #[inline]
    pub(crate) fn has_mask_subset_provenance(&self) -> bool {
        !self.mask_source_subset_to_state.is_empty()
    }

    /// Return an already-materialized mask execution state for the union of
    /// exact constraint lexer states. For a direct determinization the retained
    /// subsets are in exact-source coordinates. For a second-stage derivative
    /// of a finite virtual projection, first map each exact state into that
    /// projection and close it there; the inverse subset table is keyed in that
    /// intermediate coordinate.
    pub(crate) fn mask_runtime_state_for_source_states(
        &self,
        source_tokenizer: &Tokenizer,
        source_states: &[u32],
    ) -> Option<u32> {
        if source_states.is_empty() || self.mask_source_subset_to_state.is_empty() {
            return None;
        }
        if self.mask_tokenizer.is_some() && self.mask_determinized_tokenizer.is_some() {
            let projection = self.mask_tokenizer.as_deref()?;
            let mut subset = SmallVec::<[u32; 32]>::new();
            for &state in source_states {
                let projected = self.mask_projection_state(state);
                subset.extend_from_slice(&projection.singleton_epsilon_closure(projected));
            }
            subset.sort_unstable();
            subset.dedup();
            return self
                .mask_source_subset_to_state
                .get(subset.as_slice())
                .copied();
        }
        self.mask_projection_state_for_source_states(source_tokenizer, source_states)
    }

    pub(crate) fn mask_projection_state_for_source_states(
        &self,
        source_tokenizer: &Tokenizer,
        source_states: &[u32],
    ) -> Option<u32> {
        if source_states.is_empty() || self.mask_source_subset_to_state.is_empty() {
            return None;
        }
        let mut subset = SmallVec::<[u32; 16]>::new();
        for &state in source_states {
            let closure = source_tokenizer.singleton_epsilon_closure(state);
            subset.extend_from_slice(&closure);
        }
        subset.sort_unstable();
        subset.dedup();
        self.mask_source_subset_to_state
            .get(subset.as_slice())
            .copied()
    }

    /// Return an already-materialized deterministic mask state whose exact
    /// source-state subset is the union of `projection_states`.
    ///
    /// This never creates a DFA state at runtime. It is only an inverse lookup
    /// into source-subset provenance retained from compile-time determinization.
    pub(crate) fn mask_projection_state_for_projection_states(
        &self,
        projection_states: &[u32],
    ) -> Option<u32> {
        if projection_states.is_empty()
            || self.mask_state_source_subsets.is_empty()
            || self.mask_source_subset_to_state.is_empty()
        {
            return None;
        }
        let mut subset = SmallVec::<[u32; 32]>::new();
        for &state in projection_states {
            let source = self.mask_state_source_subsets.get(state as usize)?;
            subset.extend_from_slice(source);
        }
        subset.sort_unstable();
        subset.dedup();
        self.mask_source_subset_to_state
            .get(subset.as_slice())
            .copied()
    }

    pub(crate) fn set_virtual_unit_repeat_mask_projection(
        &mut self,
        tokenizer: Tokenizer,
        projection: VirtualZeroMinUnitRepeatMaskProjection,
    ) {
        debug_assert_eq!(tokenizer.num_states(), projection.mask_state_count(),);
        self.mask_tokenizer_fast_transitions = Self::build_full_walk_fast_transitions(&tokenizer);
        self.mask_tokenizer = Some(Arc::new(tokenizer));
        self.mask_determinized_tokenizer = None;
        self.mask_projection_to_determinized = Arc::from(Vec::<u32>::new());
        self.full_to_mask_state = Arc::from(Vec::<u32>::new());
        self.mask_state_source_subsets = Arc::from(Vec::<Arc<[u32]>>::new());
        self.mask_source_subset_to_state = Arc::new(FxHashMap::default());
        self.virtual_unit_repeat_projection = Some(projection);
        self.virtual_repeat_intersection_projections.clear();
        self.virtual_residual_projections.clear();
    }

    pub(crate) fn set_virtual_repeat_intersection_mask_projection(
        &mut self,
        tokenizer: Tokenizer,
        projection: VirtualBinaryRepeatIntersectionMaskProjection,
    ) {
        self.set_virtual_repeat_intersections_mask_projection(tokenizer, vec![projection]);
    }

    pub(crate) fn set_virtual_repeat_intersections_mask_projection(
        &mut self,
        tokenizer: Tokenizer,
        projections: Vec<VirtualBinaryRepeatIntersectionMaskProjection>,
    ) {
        debug_assert!(!projections.is_empty());
        self.mask_tokenizer_fast_transitions = Self::build_full_walk_fast_transitions(&tokenizer);
        self.mask_tokenizer = Some(Arc::new(tokenizer));
        self.mask_determinized_tokenizer = None;
        self.mask_projection_to_determinized = Arc::from(Vec::<u32>::new());
        self.full_to_mask_state = Arc::from(Vec::<u32>::new());
        self.mask_state_source_subsets = Arc::from(Vec::<Arc<[u32]>>::new());
        self.mask_source_subset_to_state = Arc::new(FxHashMap::default());
        self.virtual_unit_repeat_projection = None;
        self.virtual_repeat_intersection_projections = projections;
        self.virtual_residual_projections.clear();
    }

    pub(crate) fn set_virtual_residuals_mask_projection(
        &mut self,
        tokenizer: Tokenizer,
        projections: Vec<VirtualResidualMaskProjection>,
    ) {
        debug_assert!(!projections.is_empty());
        self.mask_tokenizer_fast_transitions = Self::build_full_walk_fast_transitions(&tokenizer);
        self.mask_tokenizer = Some(Arc::new(tokenizer));
        self.mask_determinized_tokenizer = None;
        self.mask_projection_to_determinized = Arc::from(Vec::<u32>::new());
        self.full_to_mask_state = Arc::from(Vec::<u32>::new());
        self.mask_state_source_subsets = Arc::from(Vec::<Arc<[u32]>>::new());
        self.mask_source_subset_to_state = Arc::new(FxHashMap::default());
        self.virtual_unit_repeat_projection = None;
        self.virtual_repeat_intersection_projections.clear();
        self.virtual_residual_projections = projections;
    }

    pub(crate) fn virtual_residual_mask_projection_parts(
        &self,
    ) -> Option<(&Tokenizer, &[VirtualResidualMaskProjection])> {
        if self.virtual_residual_projections.is_empty() {
            return None;
        }
        Some((
            self.mask_tokenizer.as_deref()?,
            self.virtual_residual_projections.as_slice(),
        ))
    }

    /// Preserve lexer-derived dynamic-mask metadata when a deferred vocabulary
    /// placeholder is replaced by its fully materialized runtime trie.
    ///
    /// These quotients/projections are constraint/lexer derived while the trie
    /// is vocabulary derived, so deferred dynamic compilation may construct
    /// them at different times. Sharing the immutable Arc-backed metadata
    /// avoids cloning it during that handoff.
    pub(crate) fn inherit_dynamic_lexer_metadata_from(&mut self, source: &Self) {
        self.mask_tokenizer = source.mask_tokenizer.clone();
        self.mask_determinized_tokenizer = source.mask_determinized_tokenizer.clone();
        self.mask_projection_to_determinized = Arc::clone(&source.mask_projection_to_determinized);
        self.mask_tokenizer_fast_transitions = source.mask_tokenizer_fast_transitions.clone();
        self.full_to_mask_state = Arc::clone(&source.full_to_mask_state);
        self.mask_state_source_subsets = Arc::clone(&source.mask_state_source_subsets);
        self.mask_source_subset_to_state = Arc::clone(&source.mask_source_subset_to_state);
        self.terminal_observation_classes = Arc::clone(&source.terminal_observation_classes);
        self.projected_terminal_quotients = Arc::clone(&source.projected_terminal_quotients);
        self.runtime_projected_terminal_quotients =
            Arc::clone(&source.runtime_projected_terminal_quotients);
        self.runtime_projected_terminal_quotient_cache =
            Arc::clone(&source.runtime_projected_terminal_quotient_cache);
        self.projected_terminal_quotients_prepared = source.projected_terminal_quotients_prepared;
        self.prepared_master_prover_row_ids = Arc::clone(&source.prepared_master_prover_row_ids);
        self.prepared_master_prover_offsets = Arc::clone(&source.prepared_master_prover_offsets);
        self.prepared_master_prover_terminals =
            Arc::clone(&source.prepared_master_prover_terminals);
        self.prepared_master_coverage_row_ids =
            Arc::clone(&source.prepared_master_coverage_row_ids);
        self.prepared_master_coverage_offsets =
            Arc::clone(&source.prepared_master_coverage_offsets);
        self.prepared_master_coverage_terminals =
            Arc::clone(&source.prepared_master_coverage_terminals);
        self.prepared_safe_plus_complete_terminals =
            Arc::clone(&source.prepared_safe_plus_complete_terminals);
        self.prepared_safe_radius_row_ids = Arc::clone(&source.prepared_safe_radius_row_ids);
        self.prepared_safe_radius_offsets = Arc::clone(&source.prepared_safe_radius_offsets);
        self.prepared_safe_radius_entries = Arc::clone(&source.prepared_safe_radius_entries);
        self.virtual_unit_repeat_projection = source.virtual_unit_repeat_projection;
        self.virtual_repeat_intersection_projections =
            source.virtual_repeat_intersection_projections.clone();
        self.virtual_residual_projections = source.virtual_residual_projections.clone();
    }

    pub(crate) fn mask_tokenizer_quotient_for_transfer(&self) -> Option<(Tokenizer, Vec<u32>)> {
        if self.virtual_unit_repeat_projection.is_some()
            || !self.virtual_repeat_intersection_projections.is_empty()
            || !self.virtual_residual_projections.is_empty()
        {
            // This compact structural projection is rebuilt from the exact
            // virtual tokenizer and bound vocabulary after load. The legacy
            // transfer tuple can only express a dense full-state vector.
            return None;
        }
        self.mask_tokenizer.as_ref().map(|tokenizer| {
            (
                (**tokenizer).clone(),
                self.full_to_mask_state.as_ref().to_vec(),
            )
        })
    }

    #[inline]
    pub(crate) fn mask_projection_tokenizer(&self) -> Option<&Tokenizer> {
        self.mask_tokenizer.as_deref()
    }

    /// Exact tokenizer coordinate used by dynamic mask generation. A finite
    /// serialized projection may have a second-stage deterministic derivative
    /// for the hot walk while static artifact tables remain keyed by the base
    /// projection above.
    #[inline]
    pub(crate) fn mask_runtime_tokenizer(&self) -> Option<&Tokenizer> {
        self.mask_determinized_tokenizer
            .as_deref()
            .or(self.mask_tokenizer.as_deref())
    }

    #[inline]
    pub(crate) fn has_dense_mask_tokenizer_projection(&self) -> bool {
        self.mask_tokenizer.is_some() && !self.full_to_mask_state.is_empty()
    }

    #[inline]
    pub(crate) fn mask_projection_fast_transitions(&self) -> Option<&FastTokenizerTransitions> {
        self.mask_tokenizer_fast_transitions.as_ref()
    }

    pub(crate) fn clear_mask_projection_fast_transitions(&mut self) {
        self.mask_tokenizer_fast_transitions = None;
    }

    #[cfg(test)]
    pub(crate) fn disable_prepared_mask_execution_for_test(&mut self) {
        self.mask_tokenizer = None;
        self.mask_determinized_tokenizer = None;
        self.mask_projection_to_determinized = Arc::from(Vec::<u32>::new());
        self.mask_tokenizer_fast_transitions = None;
        self.full_to_mask_state = Arc::from(Vec::<u32>::new());
        self.mask_state_source_subsets = Arc::from(Vec::<Arc<[u32]>>::new());
        self.mask_source_subset_to_state = Arc::new(FxHashMap::default());
    }

    #[inline]
    pub(crate) fn mask_projection_state(&self, full_state: u32) -> u32 {
        if !self.virtual_residual_projections.is_empty() {
            for projection in &self.virtual_residual_projections {
                if let Some(projected) = projection.project(full_state) {
                    return projected;
                }
            }
            // Ordinary physical states retain their IDs in the residual mask
            // tokenizer; only exact virtual states require an owning projection.
            if self
                .virtual_residual_projections
                .iter()
                .all(|projection| full_state < projection.physical_state_count())
            {
                return full_state;
            }
            panic!(
                "exact residual tokenizer state {full_state} has no owning finite-mask projection"
            );
        }
        if !self.virtual_repeat_intersection_projections.is_empty() {
            for projection in &self.virtual_repeat_intersection_projections {
                if let Some(projected) = projection.project(full_state) {
                    return projected;
                }
            }
            panic!(
                "exact virtual tokenizer state {full_state} has no owning finite-mask projection"
            );
        }
        if let Some(projection) = self.virtual_unit_repeat_projection {
            return projection.project(full_state).unwrap_or_else(|| {
                panic!(
                    "exact arithmetic tokenizer state {full_state} has no owning finite-mask projection"
                )
            });
        }
        self.full_to_mask_state
            .get(full_state as usize)
            .copied()
            .unwrap_or(full_state)
    }

    #[inline]
    pub(crate) fn mask_runtime_state(&self, full_state: u32) -> u32 {
        let projected = self.mask_projection_state(full_state);
        self.mask_projection_to_determinized
            .get(projected as usize)
            .copied()
            .unwrap_or(projected)
    }

    pub(crate) fn mask_projection_state_multiplicities(&self) -> Option<Vec<usize>> {
        let tokenizer = self.mask_tokenizer.as_ref()?;
        if !self.virtual_residual_projections.is_empty()
            || !self.virtual_repeat_intersection_projections.is_empty()
        {
            // The exact virtual state domain is populated lazily, so no finite
            // global full-state multiplicity table exists. Optimizations that
            // require such a table must simply decline.
            return None;
        }
        if let Some(projection) = self.virtual_unit_repeat_projection {
            let counts = projection.multiplicities();
            debug_assert_eq!(counts.len(), tokenizer.num_states() as usize);
            return Some(counts);
        }
        let mut counts = vec![0usize; tokenizer.num_states() as usize];
        for &state in self.full_to_mask_state.iter() {
            if let Some(count) = counts.get_mut(state as usize) {
                *count += 1;
            }
        }
        Some(counts)
    }

    /// Exact full-tokenizer preimage for quotient states that have exactly one
    /// runtime source.  Non-unique and unreachable quotient states are
    /// represented by `u32::MAX`.
    pub(crate) fn mask_projection_unique_full_states(&self) -> Option<Vec<u32>> {
        let tokenizer = self.mask_tokenizer.as_ref()?;
        if !self.virtual_repeat_intersection_projections.is_empty() {
            return None;
        }
        if let Some(projection) = self.virtual_unit_repeat_projection {
            let unique = projection.unique_full_states();
            debug_assert_eq!(unique.len(), tokenizer.num_states() as usize);
            return Some(unique);
        }
        let mut unique = vec![u32::MAX; tokenizer.num_states() as usize];
        let mut duplicate = vec![false; tokenizer.num_states() as usize];
        for (full_state, &mask_state) in self.full_to_mask_state.iter().enumerate() {
            let index = mask_state as usize;
            if index >= unique.len() {
                continue;
            }
            if unique[index] == u32::MAX && !duplicate[index] {
                unique[index] = full_state as u32;
            } else {
                duplicate[index] = true;
                unique[index] = u32::MAX;
            }
        }
        Some(unique)
    }

    pub(crate) fn set_bounded_observation_sets(&mut self, sets: DynamicBoundedObservationSets) {
        self.bounded_observation_sets = Arc::new(sets);
    }

    #[inline]
    pub(crate) fn bounded_observation_safe_bytes(
        &self,
        source: u32,
        required_horizon: u32,
    ) -> Option<U8Set> {
        self.bounded_observation_sets
            .safe_bytes(source, required_horizon)
    }

    #[inline]
    pub(crate) fn bounded_observation_set_counts(&self) -> (usize, usize) {
        (
            self.bounded_observation_sets.state_count(),
            self.bounded_observation_sets.unique_set_count(),
        )
    }

    pub(crate) fn set_terminal_observation_classes(
        &mut self,
        mut classes: Vec<(TerminalID, Arc<[u32]>)>,
    ) {
        classes.sort_unstable_by_key(|(terminal, _)| *terminal);
        classes.dedup_by_key(|(terminal, _)| *terminal);
        self.terminal_observation_classes = Arc::from(classes);
    }

    #[inline]
    pub(crate) fn terminal_observation_class(
        &self,
        terminal: TerminalID,
        state: u32,
    ) -> Option<u32> {
        let index = self
            .terminal_observation_classes
            .binary_search_by_key(&terminal, |(candidate, _)| *candidate)
            .ok()?;
        self.terminal_observation_classes[index]
            .1
            .get(state as usize)
            .copied()
            .filter(|&class| class != 0)
    }

    #[inline]
    pub(crate) fn has_terminal_observation_classes(&self) -> bool {
        !self.terminal_observation_classes.is_empty()
    }

    /// Prove equality of every parser-admitted terminal observation that is
    /// live at either lexer source, using only the selectively prepared exact
    /// per-terminal quotient rows. Returns the number of relevant live
    /// terminals on success; missing quotient coverage is a conservative
    /// decline.
    ///
    /// Iterate the tiny prepared-class sidecar (normally one or two terminals)
    /// rather than all parser-admitted terminals. The live-count pass is word
    /// based, so the common <=64-terminal grammar costs a handful of integer
    /// operations instead of repeated bitset probes and binary searches.
    #[inline]
    pub(crate) fn terminal_observation_equivalent_for_live_admitted(
        &self,
        left_state: u32,
        right_state: u32,
        admitted: &BitSet,
        left_matched: &BitSet,
        left_future: &BitSet,
        right_matched: &BitSet,
        right_future: &BitSet,
    ) -> Option<usize> {
        if self.terminal_observation_classes.is_empty() {
            return None;
        }

        let mut live_count = 0usize;
        for (word_index, &admitted_word) in admitted.words().iter().enumerate() {
            let word = |set: &BitSet| set.words().get(word_index).copied().unwrap_or(0);
            let live = admitted_word
                & (word(left_matched)
                    | word(left_future)
                    | word(right_matched)
                    | word(right_future));
            live_count += live.count_ones() as usize;
        }
        if live_count == 0 {
            return None;
        }

        let mut covered = 0usize;
        for (terminal, classes) in self.terminal_observation_classes.iter() {
            let terminal = *terminal as usize;
            if !admitted.contains(terminal)
                || !(left_matched.contains(terminal)
                    || left_future.contains(terminal)
                    || right_matched.contains(terminal)
                    || right_future.contains(terminal))
            {
                continue;
            }
            covered += 1;
            let left = classes.get(left_state as usize).copied().unwrap_or(0);
            let right = classes.get(right_state as usize).copied().unwrap_or(0);
            if left == 0 || left != right {
                return None;
            }
        }
        (covered == live_count).then_some(live_count)
    }

    pub(crate) fn terminal_observation_classes_cloned(&self) -> Vec<(TerminalID, Arc<[u32]>)> {
        self.terminal_observation_classes
            .iter()
            .map(|(terminal, classes)| (*terminal, Arc::clone(classes)))
            .collect()
    }

    pub(crate) fn terminal_observation_classes_for_artifact(&self) -> Vec<(TerminalID, Vec<u32>)> {
        self.terminal_observation_classes
            .iter()
            .map(|(terminal, classes)| (*terminal, classes.as_ref().to_vec()))
            .collect()
    }

    pub(crate) fn set_projected_terminal_quotients(
        &mut self,
        mut quotients: Vec<(TerminalID, TerminalProjectedQuotient)>,
    ) {
        quotients.sort_unstable_by_key(|(terminal, _)| *terminal);
        quotients.dedup_by_key(|(terminal, _)| *terminal);
        self.projected_terminal_quotients = Arc::from(
            quotients
                .into_iter()
                .map(|(terminal, quotient)| (terminal, Arc::new(quotient)))
                .collect::<Vec<_>>(),
        );
        self.runtime_projected_terminal_quotients = Arc::new(OnceLock::new());
        self.runtime_projected_terminal_quotient_cache = Arc::new(Mutex::new(FxHashMap::default()));
        self.projected_terminal_quotients_prepared = true;
        self.projected_terminal_text_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
        self.projected_terminal_partition_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
        self.projected_terminal_radius_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
    }

    pub(crate) fn projected_terminal_quotients_for_artifact(
        &self,
    ) -> Vec<(TerminalID, TerminalProjectedQuotient)> {
        self.projected_terminal_quotients
            .iter()
            .map(|(terminal, quotient)| (*terminal, quotient.as_ref().clone()))
            .collect()
    }

    #[inline]
    pub(crate) fn projected_terminal_quotient(
        &self,
        terminal: TerminalID,
        source: u32,
    ) -> Option<Arc<TerminalProjectedQuotient>> {
        let quotients = self.active_projected_terminal_quotients();
        if let Ok(index) = quotients.binary_search_by_key(&terminal, |(candidate, _)| *candidate) {
            let quotient = Arc::clone(&quotients[index].1);
            return quotient.contains_source(source).then_some(quotient);
        }
        if self.projected_terminal_quotients_prepared
            || self.runtime_projected_terminal_quotients.get().is_some()
        {
            return None;
        }
        let quotient = self
            .runtime_projected_terminal_quotient_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&terminal)
            .cloned()
            .flatten()?;
        quotient.contains_source(source).then_some(quotient)
    }

    /// Prepare exactly one runtime containment quotient. This is the hot-path
    /// counterpart to `prepare_runtime_projected_terminal_quotients`: callers
    /// that already know the parser/lexer-admitted terminal should not pay to
    /// materialize every terminal whose byte support happens to cover the
    /// generic safe-string alphabet.
    pub(crate) fn prepare_runtime_projected_terminal_quotient(
        &self,
        source: &Tokenizer,
        terminal: TerminalID,
    ) {
        if self.projected_terminal_quotients_prepared
            || self.runtime_projected_terminal_quotients.get().is_some()
        {
            return;
        }
        {
            let cache = self
                .runtime_projected_terminal_quotient_cache
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if cache.contains_key(&terminal) {
                return;
            }
        }

        let state_cap = std::env::var("GLRMASK_EXPERIMENT_DEMAND_PROOF_STATE_CAP")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(2048);
        let built = if std::env::var_os("GLRMASK_DISABLE_DEMAND_PROOF_CAP").is_none() {
            let edge_cap = std::env::var("GLRMASK_EXPERIMENT_DEMAND_PROOF_EDGE_CAP")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or_else(|| state_cap.saturating_mul(128));
            let quotient = source.build_terminal_projected_quotient_for_containment_bounded(
                terminal, state_cap, edge_cap,
            );
            if std::env::var_os("GLRMASK_DIAG_DEMAND_PROOF_CAP").is_some() {
                eprintln!(
                    "[demand_proof_cap] terminal={} state_cap={} edge_cap={} built={}",
                    terminal,
                    state_cap,
                    edge_cap,
                    quotient.is_some()
                );
            }
            quotient.map(Arc::new)
        } else {
            source
                .build_terminal_projected_quotients_for_containment_candidates(&[terminal])
                .into_iter()
                .find_map(|(candidate, quotient)| {
                    (candidate == terminal).then(|| Arc::new(quotient))
                })
        };
        self.runtime_projected_terminal_quotient_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .entry(terminal)
            .or_insert(built);
    }

    pub(crate) fn prepare_runtime_projected_terminal_quotients(
        &self,
        source: &Tokenizer,
        safe_slice_bytes: &U8Set,
        component_state_cap: Option<usize>,
    ) {
        if self.projected_terminal_quotients_prepared
            || self.runtime_projected_terminal_quotients.get().is_some()
        {
            return;
        }
        let _ = self.runtime_projected_terminal_quotients.get_or_init(|| {
            let candidates = (0..source.num_terminals())
                .filter(|&terminal| {
                    source
                        .terminal_byte_support(terminal)
                        .is_some_and(|support| safe_slice_bytes.is_subset(&support))
                })
                .collect::<Vec<_>>();
            let mut quotients = match component_state_cap {
                Some(cap) => source
                    .build_terminal_projected_quotients_for_containment_candidates_bounded(
                        &candidates,
                        cap,
                        cap.saturating_mul(256),
                    ),
                None => source
                    .build_terminal_projected_quotients_for_containment_candidates(&candidates),
            };
            quotients.sort_unstable_by_key(|(terminal, _)| *terminal);
            quotients.dedup_by_key(|(terminal, _)| *terminal);
            Arc::from(
                quotients
                    .into_iter()
                    .map(|(terminal, quotient)| (terminal, Arc::new(quotient)))
                    .collect::<Vec<_>>(),
            )
        });
    }

    #[inline]
    pub(crate) fn has_projected_terminal_quotients(&self) -> bool {
        if self.projected_terminal_quotients_prepared {
            !self.projected_terminal_quotients.is_empty()
        } else {
            self.runtime_projected_terminal_quotients
                .get()
                .is_some_and(|quotients| !quotients.is_empty())
                || self
                    .runtime_projected_terminal_quotient_cache
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .values()
                    .any(Option::is_some)
        }
    }

    #[inline]
    pub(crate) fn projected_terminal_quotients_prepared(&self) -> bool {
        self.projected_terminal_quotients_prepared
    }

    pub(crate) fn projected_terminal_text_liveness(
        &self,
        terminal: TerminalID,
        source: u32,
        alphabet: U8Set,
        include_utf8_scalars: bool,
    ) -> Option<bool> {
        let quotient = self.projected_terminal_quotient(terminal, source)?;
        let key = (terminal, source, alphabet, include_utf8_scalars);
        if let Some(&cached) = self
            .projected_terminal_text_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&key)
        {
            return Some(cached);
        }
        let certified = quotient
            .text_liveness_closed_bounded(source, alphabet, 1_024, 200_000, include_utf8_scalars)
            .unwrap_or(false);
        self.projected_terminal_text_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(key, certified);
        Some(certified)
    }
    pub(crate) fn projected_terminal_slice_contained(
        &self,
        terminal: TerminalID,
        source: u32,
        slice_cache_id: u32,
        slice_dfa: &VocabPartitionDfa,
    ) -> Option<bool> {
        let key = (terminal, source, 0x8000_0000u32 | slice_cache_id);
        if let Some(&cached) = self
            .projected_terminal_partition_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&key)
        {
            return Some(cached);
        }
        let quotient = self.projected_terminal_quotient(terminal, source)?;
        let certified = Self::terminal_partition_product_is_transparent(
            quotient.as_ref(),
            slice_dfa,
            source,
            200_000,
        )
        .unwrap_or(false);
        self.projected_terminal_partition_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(key, certified);
        Some(certified)
    }

    /// Prove a bounded safe-atom prefix radius directly in the physical lexer.
    /// Unlike the projected proof this explores only the requested horizon,
    /// without first building/minimizing the terminal's entire residual DFA.
    /// The slice must mark each completed atom by entering an accepting state,
    /// as the safe+ UTF-8 slice does. The shortest failing accepted word gives
    /// a universal safe lower bound for all strictly shorter atom layers.
    /// Every traversed prefix must retain the selected terminal. Epsilon or
    /// nonphysical coordinates and exhausted work budgets fail closed.
    pub(crate) fn raw_terminal_slice_repeat_radius(
        &self,
        tokenizer: &Tokenizer,
        terminal: TerminalID,
        source: u32,
        slice_cache_id: u32,
        slice: &VocabPartitionDfa,
        max_repetitions: u32,
        work_limit: usize,
    ) -> Option<u32> {
        if max_repetitions == 0
            || source >= tokenizer.num_states()
            || tokenizer.state_is_virtual_runtime(source)
            || tokenizer.state_has_epsilon_transitions(source)
            || slice
                .accepting_map()
                .get(slice.start_state() as usize)
                .copied()?
        {
            return None;
        }
        // A dedicated cache cannot alias projected/symbolic certificate keys.
        // Slice IDs refer to immutable languages within this vocabulary instance.
        // Include budget so an earlier cheap failed attempt cannot mask a later
        // larger-budget certificate. Cached zero is always conservative.
        let key = (
            terminal,
            source,
            slice_cache_id,
            max_repetitions,
            work_limit,
        );
        if let Some(&cached) = self
            .raw_terminal_radius_cache
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get(&key)
        {
            return Some(cached);
        }
        static PROFILE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let diagnostic = *PROFILE
            .get_or_init(|| std::env::var_os("GLRMASK_PROFILE_RAW_TERMINAL_RADIUS").is_some());
        static BATCH: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        static VERIFY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let batch_rows = *BATCH
            .get_or_init(|| std::env::var_os("GLRMASK_DISABLE_RAW_RADIUS_BATCH_ROWS").is_none());
        let verify_rows = *VERIFY
            .get_or_init(|| std::env::var_os("GLRMASK_ASSERT_RAW_RADIUS_BATCH_ROWS").is_some());
        let started = diagnostic.then(std::time::Instant::now);
        let mut work = 0usize;
        let mut explored = 0usize;
        let result = (|| {
            let slice_count = slice.accepting_map().len();
            let mut reverse = vec![Vec::<(u32, u8)>::new(); slice_count];
            for state in 0..slice_count as u32 {
                let mut seen = FxHashSet::default();
                for byte in 0u16..=255 {
                    let target = slice.step(state, byte as u8);
                    if (target as usize) < slice_count && seen.insert(target) {
                        reverse[target as usize]
                            .push((state, u8::from(slice.accepting_map()[target as usize])));
                    }
                }
            }
            let mut min_to_accept = vec![u32::MAX; slice_count];
            let mut distances = VecDeque::new();
            for (state, &accepting) in slice.accepting_map().iter().enumerate() {
                if accepting {
                    min_to_accept[state] = 0;
                    distances.push_back(state as u32);
                }
            }
            while let Some(target) = distances.pop_front() {
                for &(state, cost) in &reverse[target as usize] {
                    let next = min_to_accept[target as usize].saturating_add(u32::from(cost));
                    if next < min_to_accept[state as usize] {
                        min_to_accept[state as usize] = next;
                        if cost == 0 {
                            distances.push_front(state);
                        } else {
                            distances.push_back(state);
                        }
                    }
                }
            }
            let mut best = FxHashMap::<(u32, u32), u32>::default();
            let mut queue = VecDeque::from([(slice.start_state(), source, 0u32)]);
            best.insert((slice.start_state(), source), 0);
            let mut first_bad = max_repetitions.saturating_add(1);
            while let Some((slice_state, lexer_state, completed)) = queue.pop_front() {
                if best.get(&(slice_state, lexer_state)).copied() != Some(completed)
                    || completed >= first_bad
                    || completed > max_repetitions
                {
                    continue;
                }
                if lexer_state >= tokenizer.num_states()
                    || tokenizer.state_is_virtual_runtime(lexer_state)
                    || tokenizer.state_has_epsilon_transitions(lexer_state)
                {
                    return None;
                }
                explored += 1;
                // The guarded physical, epsilon-free row is exactly step().
                // Materialize once per product node; no row survives the proof.
                let mut row = [None; 256];
                if batch_rows {
                    for (byte, target) in tokenizer.transitions_from(lexer_state) {
                        row[byte as usize] = Some(target);
                    }
                    if verify_rows {
                        for byte in 0u16..=255 {
                            assert_eq!(row[byte as usize], tokenizer.step(lexer_state, byte as u8));
                        }
                    }
                }
                let mut previous_edge = None;
                for byte in 0u16..=255 {
                    let target_slice = slice.step(slice_state, byte as u8);
                    if target_slice as usize >= slice_count
                        || !slice.can_reach_accepting(target_slice)
                    {
                        continue;
                    }
                    let next_completed = completed
                        .saturating_add(u32::from(slice.accepting_map()[target_slice as usize]));
                    let rest = min_to_accept[target_slice as usize];
                    if rest == u32::MAX {
                        continue;
                    }
                    let shortest_word = next_completed.saturating_add(rest);
                    if shortest_word > max_repetitions || shortest_word >= first_bad {
                        continue;
                    }
                    work += 1;
                    if work > work_limit {
                        return None;
                    }
                    let target = if batch_rows {
                        row[byte as usize]
                    } else {
                        tokenizer.step(lexer_state, byte as u8)
                    };
                    // Equal product targets have identical atom costs, liveness
                    // and successors. The original eligible-byte budget has
                    // ALREADY been charged above, including duplicate edges.
                    if batch_rows {
                        let edge = (target_slice, target);
                        if previous_edge == Some(edge) {
                            continue;
                        }
                        previous_edge = Some(edge);
                    }
                    if target.is_some_and(|t| tokenizer.state_is_virtual_runtime(t)) {
                        return None;
                    }
                    let live = target.is_some_and(|t| {
                        t < tokenizer.num_states()
                            && (tokenizer
                                .possible_future_terminals(t)
                                .contains(terminal as usize)
                                || tokenizer.matched_terminals_slice(t).contains(&terminal))
                    });
                    if !live {
                        first_bad = first_bad.min(shortest_word);
                        continue;
                    }
                    let target = target.expect("live target exists");
                    if tokenizer.state_has_epsilon_transitions(target) {
                        return None;
                    }
                    if next_completed >= first_bad {
                        continue;
                    }
                    let state_key = (target_slice, target);
                    if next_completed < best.get(&state_key).copied().unwrap_or(u32::MAX) {
                        best.insert(state_key, next_completed);
                        if slice.accepting_map()[target_slice as usize] {
                            queue.push_back((target_slice, target, next_completed));
                        } else {
                            queue.push_front((target_slice, target, next_completed));
                        }
                    }
                }
            }
            Some(first_bad.saturating_sub(1).min(max_repetitions))
        })();
        // Zero is always a sound lower bound. Remember failures so an expensive
        // source cannot repeatedly spend the same bounded proof budget.
        let radius = result.unwrap_or(0);
        self.raw_terminal_radius_cache
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(key, radius);
        if diagnostic {
            eprintln!(
                "[raw_terminal_radius] source={} terminal={} radius={} complete={} work={} states={} max={} ms={:.3}",
                source,
                terminal,
                radius,
                result.is_some(),
                work,
                explored,
                max_repetitions,
                started.map_or(0.0, |s| s.elapsed().as_secs_f64() * 1000.0)
            );
        }
        Some(radius)
    }

    /// Return the largest completed-atom count `r <= max_repetitions` such
    /// that every word in the regular `slice+` language with at most `r`
    /// completed atoms remains a live prefix of this exact projected terminal.
    ///
    /// The slice DFA must mark completion of one atom by entering an accepting
    /// state (the safe+ UTF-8 DFA has exactly this property). This is the finite
    /// quotient analogue of the symbolic residual repeat-radius proof: find the
    /// shortest slice word that reaches a dead/missing quotient transition, and
    /// admit every strictly shorter completed-atom layer.
    pub(crate) fn projected_terminal_slice_repeat_radius(
        &self,
        terminal: TerminalID,
        source: u32,
        slice_cache_id: u32,
        slice: &VocabPartitionDfa,
        max_repetitions: u32,
        work_limit: usize,
    ) -> Option<u32> {
        if max_repetitions == 0
            || slice
                .accepting_map()
                .get(slice.start_state() as usize)
                .copied()?
        {
            return None;
        }
        let key = (terminal, source, slice_cache_id, max_repetitions);
        if let Some(&cached) = self
            .projected_terminal_radius_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&key)
        {
            return Some(cached);
        }
        let quotient = self.projected_terminal_quotient(terminal, source)?;
        let quotient_start = quotient.projected_state_for_source(source)?;
        if !quotient.projected_state_is_accepting(quotient_start)
            && !quotient.projected_state_has_future(quotient_start)
        {
            return Some(0);
        }

        let slice_state_count = slice.accepting_map().len();
        if slice_state_count == 0 {
            return None;
        }

        // Exact representatives for the common refinement of the slice and
        // quotient byte partitions.
        let mut representatives = Vec::<u8>::new();
        let mut seen_classes = FxHashSet::<(u8, u8)>::default();
        for byte in 0u16..=255 {
            let byte = byte as u8;
            let pair = (slice.byte_class(byte), quotient.projected_byte_class(byte));
            if seen_classes.insert(pair) {
                representatives.push(byte);
            }
        }

        // Minimum additional completed atoms needed to reach slice acceptance
        // from every slice state. Entering an accepting state completes one
        // atom; UTF-8 continuation states therefore have zero-cost edges.
        let mut reverse = vec![Vec::<(u32, u8)>::new(); slice_state_count];
        for source_state in 0..slice_state_count as u32 {
            let mut seen_targets = FxHashSet::<u32>::default();
            for &byte in &representatives {
                let target = slice.step(source_state, byte);
                if target as usize >= slice_state_count || !seen_targets.insert(target) {
                    continue;
                }
                reverse[target as usize].push((
                    source_state,
                    u8::from(slice.accepting_map()[target as usize]),
                ));
            }
        }
        let mut min_to_accept = vec![u32::MAX; slice_state_count];
        let mut distance_queue = VecDeque::<u32>::new();
        for (state, &accepting) in slice.accepting_map().iter().enumerate() {
            if accepting {
                min_to_accept[state] = 0;
                distance_queue.push_back(state as u32);
            }
        }
        while let Some(target) = distance_queue.pop_front() {
            let target_distance = min_to_accept[target as usize];
            for &(source_state, cost) in &reverse[target as usize] {
                let candidate = target_distance.saturating_add(u32::from(cost));
                if candidate < min_to_accept[source_state as usize] {
                    min_to_accept[source_state as usize] = candidate;
                    if cost == 0 {
                        distance_queue.push_front(source_state);
                    } else {
                        distance_queue.push_back(source_state);
                    }
                }
            }
        }

        let mut best = FxHashMap::<(u32, u32), u32>::default();
        let mut queue = VecDeque::<(u32, u32, u32)>::new();
        let slice_start = slice.start_state();
        best.insert((slice_start, quotient_start), 0);
        queue.push_back((slice_start, quotient_start, 0));
        let mut work = 0usize;
        let mut first_counterexample = max_repetitions.saturating_add(1);

        while let Some((slice_state, quotient_state, completed)) = queue.pop_front() {
            if best.get(&(slice_state, quotient_state)).copied() != Some(completed)
                || completed >= first_counterexample
                || completed > max_repetitions
            {
                continue;
            }
            for &byte in &representatives {
                let slice_target = slice.step(slice_state, byte);
                if slice_target as usize >= slice_state_count
                    || !slice.can_reach_accepting(slice_target)
                {
                    continue;
                }
                let completed_target = completed
                    .saturating_add(u32::from(slice.accepting_map()[slice_target as usize]));
                let completion_cost = min_to_accept[slice_target as usize];
                if completion_cost == u32::MAX {
                    continue;
                }
                let shortest_complete_word = completed_target.saturating_add(completion_cost);
                if shortest_complete_word > max_repetitions {
                    continue;
                }
                work = work.saturating_add(1);
                if work > work_limit {
                    return None;
                }
                let quotient_class = quotient.projected_byte_class(byte);
                let quotient_target = quotient.projected_step_class(quotient_state, quotient_class);
                let target_live = quotient_target.is_some_and(|target| {
                    quotient.projected_state_is_accepting(target)
                        || quotient.projected_state_has_future(target)
                });
                if !target_live {
                    first_counterexample = first_counterexample.min(shortest_complete_word);
                    continue;
                }
                let quotient_target = quotient_target.expect("live quotient target must exist");
                if completed_target >= first_counterexample || completed_target > max_repetitions {
                    continue;
                }
                let state_key = (slice_target, quotient_target);
                if completed_target < best.get(&state_key).copied().unwrap_or(u32::MAX) {
                    best.insert(state_key, completed_target);
                    if slice.accepting_map()[slice_target as usize] {
                        queue.push_back((slice_target, quotient_target, completed_target));
                    } else {
                        queue.push_front((slice_target, quotient_target, completed_target));
                    }
                }
            }
        }

        let radius = first_counterexample.saturating_sub(1).min(max_repetitions);
        self.projected_terminal_radius_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(key, radius);
        Some(radius)
    }

    pub(super) fn terminal_partition_product_is_transparent(
        quotient: &TerminalProjectedQuotient,
        partition: &VocabPartitionDfa,
        source: u32,
        work_limit: usize,
    ) -> Option<bool> {
        let quotient_start = quotient.projected_state_for_source(source)?;
        if !quotient.projected_state_is_accepting(quotient_start)
            && !quotient.projected_state_has_future(quotient_start)
        {
            return Some(false);
        }
        let partition_start = partition.start_state();
        if !partition.can_reach_accepting(partition_start) {
            return Some(true);
        }

        // Refine the two exact global byte partitions. A byte representative is
        // interchangeable only when both automata put it in the same class.
        let mut representatives = Vec::<(u8, u8, u8)>::new();
        for byte in 0u16..=255 {
            let byte = byte as u8;
            let partition_class = partition.byte_class(byte);
            let quotient_class = quotient.projected_byte_class(byte);
            if !representatives
                .iter()
                .any(|&(p, q, _)| p == partition_class && q == quotient_class)
            {
                representatives.push((partition_class, quotient_class, byte));
            }
        }

        let mut seen = FxHashSet::<(u32, u32)>::default();
        let mut queue = VecDeque::from([(partition_start, quotient_start)]);
        let mut work = 0usize;
        while let Some((partition_state, quotient_state)) = queue.pop_front() {
            if !seen.insert((partition_state, quotient_state)) {
                continue;
            }
            for &(_, quotient_class, byte) in &representatives {
                work = work.saturating_add(1);
                if work > work_limit {
                    return None;
                }
                let partition_target = partition.step(partition_state, byte);
                if !partition.can_reach_accepting(partition_target) {
                    continue;
                }
                let Some(quotient_target) =
                    quotient.projected_step_class(quotient_state, quotient_class)
                else {
                    return Some(false);
                };
                if !quotient.projected_state_is_accepting(quotient_target)
                    && !quotient.projected_state_has_future(quotient_target)
                {
                    return Some(false);
                }
                if !seen.contains(&(partition_target, quotient_target)) {
                    queue.push_back((partition_target, quotient_target));
                }
            }
        }
        Some(true)
    }
}
