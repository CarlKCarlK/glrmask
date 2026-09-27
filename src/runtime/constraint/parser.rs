//! Representation-independent reads of materialized and packed parser machinery.

use crate::compiler::glr::labels::DEFAULT_LABEL;
use crate::compiler::glr::labels::encode_positive_label;
use crate::ds::weight::Weight;
use crate::grammar::flat::TerminalID;
use crate::runtime::artifact::Constraint;
use crate::runtime::artifact::FastDwaTransitionRow;
use crate::runtime::artifact::IndexedDagDenseTransition;
use crate::runtime::artifact::IndexedDagDenseTransitionRow;
use smallvec::SmallVec;
use super::RuntimeWeightRef;

impl Constraint {


    #[inline]
    pub(crate) fn parser_state_domain_label(&self, parser_state: u32) -> Option<i32> {
        self.parser_state_domain_labels
            .get(parser_state as usize)
            .copied()
            .filter(|&label| label != i32::MAX)
    }

    #[inline]
    pub(crate) fn fast_parser_dwa_transition<'a>(
        &self,
        row: &'a FastDwaTransitionRow,
        parser_state: u32,
    ) -> Option<(u32, &'a Weight)> {
        let positive = encode_positive_label(parser_state);
        row.get(&positive)
            .or_else(|| self.parser_state_domain_label(parser_state).and_then(|label| row.get(&label)))
            .or_else(|| row.get(&DEFAULT_LABEL))
    }

    #[inline]
    pub(crate) fn runtime_parser_dwa_state_count(&self) -> usize {
        self.packed_parser_dwa
            .as_ref()
            .map_or_else(|| self.parser_dwa.states().len(), |dwa| dwa.state_count())
    }

    #[inline]
    pub(crate) fn runtime_parser_dwa_start_state(&self) -> u32 {
        self.packed_parser_dwa
            .as_ref()
            .map_or_else(|| self.parser_dwa.start_state(), |dwa| dwa.start_state())
    }

    #[inline]
    pub(crate) fn runtime_parser_dwa_final_weight(
        &self,
        dwa_state: u32,
    ) -> Option<RuntimeWeightRef<'_>> {
        if dwa_state == self.runtime_parser_dwa_start_state()
            && let Some(weight) = self.parser_start_final_override.as_ref()
        {
            return (!weight.is_empty()).then_some(RuntimeWeightRef::Materialized(weight));
        }
        if let Some(dwa) = &self.packed_parser_dwa {
            return dwa
                .final_weight(dwa_state)
                .map(RuntimeWeightRef::PackedDwa);
        }
        self.parser_dwa
            .states()
            .get(dwa_state as usize)?
            .final_weight
            .as_ref()
            .map(RuntimeWeightRef::Materialized)
    }

    #[inline]
    pub(crate) fn runtime_parser_dwa_transition(
        &self,
        dwa_state: u32,
        parser_state: u32,
    ) -> Option<(u32, RuntimeWeightRef<'_>)> {
        let positive = encode_positive_label(parser_state);
        if let Some(dwa) = &self.packed_parser_dwa {
            return dwa
                .transition(dwa_state, positive)
                .or_else(|| {
                    self.parser_state_domain_label(parser_state)
                        .and_then(|label| dwa.transition(dwa_state, label))
                })
                .or_else(|| dwa.transition(dwa_state, DEFAULT_LABEL))
                .map(|(target, weight)| (target, RuntimeWeightRef::PackedDwa(weight)));
        }
        let row = self.dwa_fast_transitions.get(dwa_state as usize)?;
        self.fast_parser_dwa_transition(row, parser_state)
            .map(|(target, weight)| (target, RuntimeWeightRef::Materialized(weight)))
    }

    #[inline]
    pub(crate) fn runtime_parser_dwa_row_is_empty(&self, dwa_state: u32) -> bool {
        if let Some(dwa) = &self.packed_parser_dwa {
            dwa.row_is_empty(dwa_state)
        } else {
            self.dwa_fast_transitions
                .get(dwa_state as usize)
                .is_none_or(FastDwaTransitionRow::is_empty)
        }
    }

    #[inline]
    fn runtime_pooled_weight(&self, id: u32) -> Option<RuntimeWeightRef<'_>> {
        let packed = self.packed_non_dwa_weights.as_ref()?;
        packed.pool.weight(id).map(RuntimeWeightRef::PackedPool)
    }

    #[inline]
    pub(crate) fn runtime_parser_top_accept(
        &self,
        label: i32,
    ) -> Option<RuntimeWeightRef<'_>> {
        if let Some(packed) = &self.packed_non_dwa_weights {
            let id = packed
                .parser_top_accept
                .get(&label)
                .or_else(|| packed.parser_top_accept.get(&DEFAULT_LABEL))?;
            return self.runtime_pooled_weight(*id);
        }
        self.parser_top_accept
            .get(&label)
            .or_else(|| self.parser_top_accept.get(&DEFAULT_LABEL))
            .map(RuntimeWeightRef::Materialized)
    }

    pub(crate) fn runtime_parser_top_accept_parts(
        &self,
        label: i32,
    ) -> SmallVec<[RuntimeWeightRef<'_>; 4]> {
        if let Some(packed) = &self.packed_non_dwa_weights {
            let Some(ids) = packed
                .parser_top_accept_parts
                .get(&label)
                .or_else(|| packed.parser_top_accept_parts.get(&DEFAULT_LABEL))
            else {
                return SmallVec::new();
            };
            return ids
                .iter()
                .filter_map(|&id| self.runtime_pooled_weight(id))
                .collect();
        }
        self.parser_top_accept_parts
            .get(&label)
            .or_else(|| self.parser_top_accept_parts.get(&DEFAULT_LABEL))
            .into_iter()
            .flatten()
            .map(RuntimeWeightRef::Materialized)
            .collect()
    }

    #[inline]
    pub(crate) fn runtime_direct_regular_l1_complete(
        &self,
        terminal: TerminalID,
    ) -> Option<RuntimeWeightRef<'_>> {
        if let Some(packed) = &self.packed_non_dwa_weights {
            return packed
                .direct_regular_l1_complete_by_terminal
                .get(&terminal)
                .and_then(|&id| self.runtime_pooled_weight(id));
        }
        self.direct_regular_l1_complete_by_terminal
            .get(&terminal)
            .map(RuntimeWeightRef::Materialized)
    }

    #[inline]
    pub(crate) fn runtime_possible_match_weight(
        &self,
        terminal: TerminalID,
    ) -> Option<RuntimeWeightRef<'_>> {
        if let Some(packed) = &self.packed_non_dwa_weights {
            return packed
                .possible_matches
                .get(&terminal)
                .and_then(|&id| self.runtime_pooled_weight(id));
        }
        self.possible_matches
            .get(&terminal)
            .map(RuntimeWeightRef::Materialized)
    }

    pub(crate) fn runtime_possible_match_terminals(
        &self,
    ) -> Box<dyn Iterator<Item = TerminalID> + '_> {
        if let Some(packed) = &self.packed_non_dwa_weights {
            Box::new(packed.possible_matches.keys().copied())
        } else {
            Box::new(self.possible_matches.keys().copied())
        }
    }

    #[inline]
    pub(crate) fn runtime_direct_regular_l1_is_empty(&self) -> bool {
        self.packed_non_dwa_weights.as_ref().map_or_else(
            || self.direct_regular_l1_complete_by_terminal.is_empty(),
            |packed| packed.direct_regular_l1_complete_by_terminal.is_empty(),
        )
    }

    pub(crate) fn runtime_direct_regular_l1_terminals(
        &self,
    ) -> Box<dyn Iterator<Item = TerminalID> + '_> {
        if let Some(packed) = &self.packed_non_dwa_weights {
            Box::new(
                packed
                    .direct_regular_l1_complete_by_terminal
                    .keys()
                    .copied(),
            )
        } else {
            Box::new(self.direct_regular_l1_complete_by_terminal.keys().copied())
        }
    }

    #[inline]
    pub(crate) fn indexed_parser_dwa_transition<'a>(
        &self,
        row: &'a IndexedDagDenseTransitionRow,
        parser_state: u32,
    ) -> Option<&'a IndexedDagDenseTransition> {
        let positive = encode_positive_label(parser_state);
        row.get(&positive)
            .or_else(|| self.parser_state_domain_label(parser_state).and_then(|label| row.get(&label)))
            .or_else(|| row.get(&DEFAULT_LABEL))
    }
}
