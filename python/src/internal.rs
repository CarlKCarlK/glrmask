//! Registration and thin forwarding functions for the unstable tooling surface.

use crate::allocator::{collect_allocator, mimalloc_purge_decommits, mimalloc_purge_delay};
use crate::compiler::{
    auto_json_schema_shape_counts, clear_stale_weights, clear_weight_caches,
    clear_weight_op_caches, compile_ebnf_serialized, compile_ebnf_serialized_profiled,
    compile_glrm_serialized, compile_glrm_serialized_profiled, compile_grammar_def_json,
    compile_json_schema_serialized, compile_json_schema_serialized_profiled,
    compile_lark_serialized, compile_lark_serialized_profiled, compiler_cache_stats,
    compose_compiled_subgrammars, dump_json_schema_grammar_glrm, prepare_vocab_for_compile,
    prepare_vocab_for_dynamic_compile,
};
use crate::conversion::WritableMask;
use crate::{partition, PyConstraint, PyConstraintState};
use pyo3::prelude::*;
use pyo3::types::PyDict;

// ---------------------------------------------------------------------------
// UnlinkedConstraint
// ---------------------------------------------------------------------------

#[pyfunction]
pub(super) fn num_parser_states(constraint: PyRef<'_, PyConstraint>) -> u32 {
    constraint.num_parser_states()
}

#[pyfunction]
pub(super) fn terminal_display_names(constraint: PyRef<'_, PyConstraint>) -> Vec<String> {
    constraint.terminal_display_names()
}

#[pyfunction]
pub(super) fn terminal_display_name(
    constraint: PyRef<'_, PyConstraint>,
    terminal_id: u32,
) -> Option<String> {
    constraint.terminal_display_name(terminal_id)
}

#[pyfunction]
pub(super) fn fill_mask_timed_ns(
    state: PyRef<'_, PyConstraintState>,
    bitmask: WritableMask<'_>,
) -> PyResult<u64> {
    state.fill_mask_timed_ns(bitmask)
}

#[cfg(feature = "allocation-tracking")]
#[pyfunction]
pub(super) fn fill_mask_timed_allocation_stats(
    state: PyRef<'_, PyConstraintState>,
    bitmask: WritableMask<'_>,
) -> PyResult<Vec<u64>> {
    state.fill_mask_timed_allocation_stats(bitmask)
}

#[pyfunction]
pub(super) fn fill_mask_profiled<'py>(
    py: Python<'py>,
    state: PyRef<'py, PyConstraintState>,
    bitmask: WritableMask<'_>,
) -> PyResult<Bound<'py, PyDict>> {
    state.fill_mask_profiled(py, bitmask)
}

#[pyfunction]
pub(super) fn commit_token_timed_ns(
    mut state: PyRefMut<'_, PyConstraintState>,
    token_id: u32,
) -> PyResult<u64> {
    state.commit_token_timed_ns(token_id)
}

#[cfg(feature = "allocation-tracking")]
#[pyfunction]
pub(super) fn commit_token_timed_allocation_stats(
    mut state: PyRefMut<'_, PyConstraintState>,
    token_id: u32,
) -> PyResult<Vec<u64>> {
    state.commit_token_timed_allocation_stats(token_id)
}

#[pyfunction]
pub(super) fn commit_token_profiled<'py>(
    py: Python<'py>,
    mut state: PyRefMut<'py, PyConstraintState>,
    token_id: u32,
) -> PyResult<Bound<'py, PyDict>> {
    state.commit_token_profiled(py, token_id)
}

#[pyfunction]
pub(super) fn parser_root_count(state: PyRef<'_, PyConstraintState>) -> usize {
    state.parser_root_count()
}

#[pyfunction]
pub(super) fn parser_path_count(state: PyRef<'_, PyConstraintState>, limit: usize) -> usize {
    state.parser_path_count(limit)
}

#[pyfunction]
pub(super) fn debug_parser_stacks(
    state: PyRef<'_, PyConstraintState>,
) -> Vec<(u32, Vec<(Vec<u32>, Vec<(u32, Vec<u32>)>)>)> {
    state.debug_parser_stacks()
}

#[pyfunction]
pub(super) fn commit_token_per_advance<'py>(
    py: Python<'py>,
    mut state: PyRefMut<'py, PyConstraintState>,
    token_id: u32,
) -> PyResult<Bound<'py, PyDict>> {
    state.commit_token_per_advance(py, token_id)
}

pub(super) fn add_internal_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let internal = PyModule::new(m.py(), "_internal")?;
    internal.add_class::<partition::PyVocabPartition>()?;
    internal.setattr(
        "__doc__",
        "Unstable internal API for CFA and repository tooling. No compatibility guarantees.",
    )?;
    internal.add_function(wrap_pyfunction!(clear_stale_weights, &internal)?)?;
    internal.add_function(wrap_pyfunction!(clear_weight_op_caches, &internal)?)?;
    internal.add_function(wrap_pyfunction!(clear_weight_caches, &internal)?)?;
    internal.add_function(wrap_pyfunction!(compiler_cache_stats, &internal)?)?;
    internal.add_function(wrap_pyfunction!(prepare_vocab_for_compile, &internal)?)?;
    internal.add_function(wrap_pyfunction!(
        prepare_vocab_for_dynamic_compile,
        &internal
    )?)?;
    internal.add_function(wrap_pyfunction!(compile_grammar_def_json, &internal)?)?;
    internal.add_function(wrap_pyfunction!(dump_json_schema_grammar_glrm, &internal)?)?;
    internal.add_function(wrap_pyfunction!(compose_compiled_subgrammars, &internal)?)?;
    internal.add_function(wrap_pyfunction!(auto_json_schema_shape_counts, &internal)?)?;
    internal.add_function(wrap_pyfunction!(
        compile_ebnf_serialized_profiled,
        &internal
    )?)?;
    internal.add_function(wrap_pyfunction!(
        compile_lark_serialized_profiled,
        &internal
    )?)?;
    internal.add_function(wrap_pyfunction!(
        compile_json_schema_serialized_profiled,
        &internal
    )?)?;
    internal.add_function(wrap_pyfunction!(
        compile_glrm_serialized_profiled,
        &internal
    )?)?;
    internal.add_function(wrap_pyfunction!(compile_ebnf_serialized, &internal)?)?;
    internal.add_function(wrap_pyfunction!(compile_lark_serialized, &internal)?)?;
    internal.add_function(wrap_pyfunction!(compile_json_schema_serialized, &internal)?)?;
    internal.add_function(wrap_pyfunction!(compile_glrm_serialized, &internal)?)?;
    internal.add_function(wrap_pyfunction!(num_parser_states, &internal)?)?;
    internal.add_function(wrap_pyfunction!(terminal_display_names, &internal)?)?;
    internal.add_function(wrap_pyfunction!(terminal_display_name, &internal)?)?;
    internal.add_function(wrap_pyfunction!(fill_mask_timed_ns, &internal)?)?;
    #[cfg(feature = "allocation-tracking")]
    internal.add_function(wrap_pyfunction!(
        fill_mask_timed_allocation_stats,
        &internal
    )?)?;
    internal.add_function(wrap_pyfunction!(fill_mask_profiled, &internal)?)?;
    internal.add_function(wrap_pyfunction!(commit_token_timed_ns, &internal)?)?;
    #[cfg(feature = "allocation-tracking")]
    internal.add_function(wrap_pyfunction!(
        commit_token_timed_allocation_stats,
        &internal
    )?)?;
    internal.add_function(wrap_pyfunction!(commit_token_profiled, &internal)?)?;
    internal.add_function(wrap_pyfunction!(parser_root_count, &internal)?)?;
    internal.add_function(wrap_pyfunction!(parser_path_count, &internal)?)?;
    internal.add_function(wrap_pyfunction!(debug_parser_stacks, &internal)?)?;
    internal.add_function(wrap_pyfunction!(commit_token_per_advance, &internal)?)?;
    internal.add_function(wrap_pyfunction!(mimalloc_purge_delay, &internal)?)?;
    internal.add_function(wrap_pyfunction!(mimalloc_purge_decommits, &internal)?)?;
    internal.add_function(wrap_pyfunction!(collect_allocator, &internal)?)?;
    m.add_submodule(&internal)?;
    Ok(())
}
