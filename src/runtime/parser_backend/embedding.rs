//! Finite embedding transfers derived while a built-in parser is compiled.
//!
//! Completion is an input predicate, not a stack-return operation. Preserve
//! the actual finite return relation before discarding the built-in LR table;
//! a later linker must never reconstruct that table or guess a return depth.
use std::{collections::{BTreeMap, BTreeSet}, sync::Arc};
use super::{CommitTemplateDfas, Constraint};
use crate::compiler::glr::table::GLRTable;
use glrmask_parser_dwa::__private::templates::characterize::{
    characterize_finish_transfer, FinishEndpointPolicy,
};
use glrmask_parser_dwa::__private::templates::compile_dfa::{Templates,
    specialize_template_dfa_defaults_for_commit_split_input, try_split_commit_template_dfas};

#[derive(Debug, Clone)]
pub(crate) struct TemplateEmbedding {
    pub(crate) nullable: bool,
    pub(crate) return_pop: u32,
    pub(crate) entries: BTreeSet<u32>,
    pub(crate) finish: Arc<CommitTemplateDfas>,
    pub(crate) finish_view: super::scoped_program::ScopedProgram,
}

impl TemplateEmbedding {
    pub(crate) fn new(nullable: bool, return_pop: u32, entries: BTreeSet<u32>,
        finish: Arc<CommitTemplateDfas>, symbols: u32) -> Result<Self, String> {
        let finish_view = super::scoped_program::ScopedProgram::prepare(Arc::clone(&finish), symbols)
            .map_err(|error| error.to_string())?;
        Ok(Self { nullable, return_pop, entries, finish, finish_view })
    }

    /// Only for the built-in depth-one regular frontend: its generated EOF
    /// program has exactly the POP-one return semantics, not just a predicate.
    /// Retain source nullability even when standalone preparation removed it.
    pub(crate) fn from_sparse_regular(
        completion: &Arc<CommitTemplateDfas>,
        symbols: u32,
        source_nullable: bool,
        slots: impl IntoIterator<Item = u32>,
    ) -> Result<Self, String> {
        let nullable = source_nullable || super::compile_domain(completion)
            .map_err(|error| error.to_string())?.matches_top_first([0]);
        let finish = if nullable {
            Arc::new(super::link_program::compile(&[
                super::link_program::action_nfa(completion)?,
                super::link_program::nullable_return(0),
            ])?)
        } else { Arc::clone(completion) };
        Self::new(nullable, 1, slots.into_iter().collect(), finish, symbols)
    }

    pub(crate) fn from_table(table: &GLRTable, nullable: bool, return_pop: u32,
        slots: impl IntoIterator<Item = u32>) -> Result<Self, String> {
        let transfer = characterize_finish_transfer(table, &FinishEndpointPolicy {
            return_pop, nullable_child_start: nullable.then_some(0),
        })?;
        let raw = Templates::from_characterizations(&BTreeMap::from([(0, transfer.characterization)]))
            .by_terminal.remove(&0).ok_or("missing finite embedding return relation")?;
        let raw = specialize_template_dfa_defaults_for_commit_split_input(&raw);
        let finish = try_split_commit_template_dfas(&raw)
            .ok_or("embedding return is not a finite POP/READ/PUSH relation")?;
        super::compile_domain(&finish).map_err(|error| error.to_string())?;
        let mut entries = BTreeSet::new();
        for terminal in slots {
            // Preserve only slots whose complete terminal relation can be
            // turned into CALL by appending the child's initial stack symbol.
            crate::compiler::boundary_transfer::validate_slot_entry_shape(table, terminal)?;
            entries.insert(terminal);
        }
        Self::new(nullable, return_pop, entries, Arc::new(finish), table.num_states)
    }

    pub(crate) fn from_constraint(constraint: &Constraint) -> Result<Self, String> {
        Self::from_table(&constraint.table, constraint.composition_start_nullable()?,
            constraint.composition_child_return_pop()?,
            constraint.late_grammar_slots.iter().map(|slot| slot.terminal_id))
    }
}
