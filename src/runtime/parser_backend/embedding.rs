//! Finite embedding transfers derived while a built-in parser is compiled.
//!
//! Completion is an input predicate, not a stack-return operation. Preserve
//! the actual finite return relation before discarding the built-in LR table;
//! a later linker must never reconstruct that table or guess a return depth.

use std::{collections::BTreeSet, sync::Arc};

use super::{CommitTemplateDfas, Constraint};
use crate::compiler::glr::table::GLRTable;
use glrmask_parser_dwa::__private::templates::characterize::{
    FinishEndpointPolicy,
};
use glrmask_parser_dwa::__private::templates::native::{
    NativeTableIndex, ProgramCompiler,
};

#[derive(Debug, Clone)]
pub(crate) struct TemplateEmbedding {
    pub(crate) nullable: bool,
    pub(crate) return_pop: u32,
    pub(crate) entries: BTreeSet<u32>,
    pub(crate) finish: Arc<CommitTemplateDfas>,
    pub(crate) finish_view: super::scoped_program::ScopedProgram,
}

impl TemplateEmbedding {
    pub(crate) fn new(
        nullable: bool,
        return_pop: u32,
        entries: BTreeSet<u32>,
        finish: Arc<CommitTemplateDfas>,
        symbols: u32,
    ) -> Result<Self, String> {
        let finish_view = super::scoped_program::ScopedProgram::prepare(
            Arc::clone(&finish), symbols,
        ).map_err(|error| error.to_string())?;
        Ok(Self { nullable, return_pop, entries, finish, finish_view })
    }

    /// Only for the built-in depth-one regular frontend: its generated EOF
    /// program has exactly the POP-one return semantics, not just a predicate.
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
        } else {
            Arc::clone(completion)
        };
        Self::new(nullable, 1, slots.into_iter().collect(), finish, symbols)
    }

    /// Native compiler path: reuse the table index and the entry certificate
    /// issued by that same immutable borrow. No second slot scan or discarded
    /// completion/domain preparation is performed.
    pub(crate) fn from_index(
        index: &NativeTableIndex<'_>,
        compiler: &ProgramCompiler,
        nullable: bool,
        return_pop: u32,
    ) -> Result<Self, String> {
        let transfer = index.finish(FinishEndpointPolicy {
            return_pop,
            nullable_child_start: nullable.then_some(0),
        })?;
        let finish = compiler.compile(&transfer.characterization)?;
        Self::new(
            nullable,
            return_pop,
            index.valid_slot_entry_terminals().clone(),
            Arc::new(finish),
            index.table().num_states,
        )
    }

    /// Existing selected-slot compiler API. Arbitrary requested slots retain
    /// their mandatory shape checks; only the index-issued all-valid inventory
    /// can use from_index without repeating those checks.
    pub(crate) fn from_table(
        table: &GLRTable,
        nullable: bool,
        return_pop: u32,
        slots: impl IntoIterator<Item = u32>,
    ) -> Result<Self, String> {
        let selected = vec![false; table.num_terminals as usize];
        let index = NativeTableIndex::new(table, &selected)?;
        let transfer = index.finish(FinishEndpointPolicy {
            return_pop,
            nullable_child_start: nullable.then_some(0),
        })?;
        let finish = ProgramCompiler::new().compile(&transfer.characterization)?;
        let entries = slots.into_iter().collect::<BTreeSet<_>>();
        crate::compiler::boundary_transfer::validate_slot_entry_shapes(table, &entries)?;
        Self::new(
            nullable, return_pop, entries, Arc::new(finish), table.num_states,
        )
    }

    pub(crate) fn from_constraint(constraint: &Constraint) -> Result<Self, String> {
        Self::from_table(
            &constraint.table,
            constraint.composition_start_nullable()?,
            constraint.composition_child_return_pop()?,
            constraint.late_grammar_slots.iter().map(|slot| slot.terminal_id),
        )
    }
}
