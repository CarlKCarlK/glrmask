#![recursion_limit = "512"]
//! PyO3 Python bindings for glrmask.
//!
//! The public facade follows Grammar -> UnlinkedConstraint -> Constraint -> State.
//! NumPy masks use model-token coordinates; unstable analysis and measurement
//! hooks live under `glrmask._internal`, not the public API.
//!
//! # Lifetime handling
//!
//! `glrmask::ConstraintState<'a>` borrows `&'a Constraint`. PyO3 pyclass structs
//! must be `'static`, so we cannot store a `ConstraintState<'_>` directly.
//!
//! Solution: pair the `ConstraintState<'a>` with its `Arc<Constraint>` owner inside
//! a [`self_cell::self_cell!`] struct (`OwnedState`). `self_cell` generates the
//! necessary unsafe bookkeeping internally (owner outlives dependent, stable
//! address via heap allocation) and exposes a safe public API for the owner /
//! dependent relationship. The only handwritten `unsafe` in this file is the
//! NumPy `i32` to `u32` bitmask view cast used by `fill_mask`.

mod api;
mod partition;

#[cfg(feature = "allocation-tracking")]
mod allocation_tracking;

#[cfg(feature = "allocation-tracking")]
#[global_allocator]
static GLOBAL: allocation_tracking::TrackingAllocator =
    allocation_tracking::TrackingAllocator(mimalloc::MiMalloc);

#[cfg(not(feature = "allocation-tracking"))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

// `libmimalloc-sys` intentionally does not expose constants for these advanced
// options. The values are pinned by the mimalloc v3 `mi_option_e` ordering.
const MIMALLOC_PURGE_DECOMMITS_OPTION: libmimalloc_sys::mi_option_t = 5;
const MIMALLOC_PURGE_DELAY_OPTION: libmimalloc_sys::mi_option_t = 15;

fn configure_mimalloc_runtime_default() {
    // Keep delayed automatic purging, but reset unused pages with
    // MADV_FREE/MEM_RESET rather than synchronously decommitting them. Reset
    // pages remain reclaimable by the OS without charging decommit work to an
    // arbitrary runtime allocation. An explicit mimalloc environment setting
    // remains authoritative.
    if std::env::var_os("MIMALLOC_PURGE_DECOMMITS").is_none()
        && std::env::var_os("MIMALLOC_RESET_DECOMMITS").is_none()
    {
        unsafe {
            libmimalloc_sys::mi_option_set_enabled(MIMALLOC_PURGE_DECOMMITS_OPTION, false);
        }
    }
}

use numpy::{PyArray1, PyArrayMethods, PyReadwriteArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict};
use self_cell::self_cell;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;
use glrmask::__private::{
    ConstraintExt as _, ConstraintStateExt as _, DynamicConstraintExt as _, VocabExt as _,
};

// ---------------------------------------------------------------------------
// OwnedState — `self_cell`-generated safe owner/dependent pair.
// ---------------------------------------------------------------------------

type ConstraintState<'a> = glrmask::ConstraintState<'a>;
type DynamicConstraintState<'a> = glrmask::DynamicConstraintState<'a>;

self_cell!(
    struct OwnedState {
        owner: Arc<glrmask::Constraint>,
        #[not_covariant]
        dependent: ConstraintState,
    }
);

impl OwnedState {
    fn from_arc(arc: Arc<glrmask::Constraint>) -> Self {
        OwnedState::new(arc, |arc_ref| arc_ref.start())
    }
}

self_cell!(
    struct OwnedDynamicState {
        owner: Arc<glrmask::DynamicConstraint>,
        #[not_covariant]
        dependent: DynamicConstraintState,
    }
);

impl OwnedDynamicState {
    fn from_arc(arc: Arc<glrmask::DynamicConstraint>) -> Self {
        OwnedDynamicState::new(arc, |arc_ref| arc_ref.start())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn dict_to_vocab(token_to_id: &Bound<'_, PyDict>) -> PyResult<glrmask::Vocab> {
    let mut entries = Vec::with_capacity(token_to_id.len());
    for (key, value) in token_to_id.iter() {
        let token_bytes = key
            .downcast::<PyBytes>()
            .map_err(|_| PyValueError::new_err("vocab keys must be Python bytes"))?
            .as_bytes()
            .to_vec();
        let token_id: u32 = value.extract()?;
        entries.push((token_id, token_bytes));
    }
    Ok(glrmask::Vocab::new(entries))
}

fn id_to_bytes_dict_to_vocab(id_to_bytes: &Bound<'_, PyDict>) -> PyResult<glrmask::Vocab> {
    let mut entries = Vec::with_capacity(id_to_bytes.len());
    for (key, value) in id_to_bytes.iter() {
        let token_id: u32 = key.extract()?;
        let token_bytes = value
            .downcast::<PyBytes>()
            .map_err(|_| PyValueError::new_err("vocab values must be Python bytes"))?
            .as_bytes()
            .to_vec();
        entries.push((token_id, token_bytes));
    }
    Ok(glrmask::Vocab::new(entries))
}

fn external_terminal_bindings_from_dict(
    bindings: Option<&Bound<'_, PyDict>>,
) -> PyResult<Vec<(String, Vec<u32>)>> {
    let Some(bindings) = bindings else {
        return Ok(Vec::new());
    };
    let mut result = Vec::with_capacity(bindings.len());
    for (name, value) in bindings.iter() {
        let name = name.extract::<String>().map_err(|_| {
            PyValueError::new_err("external terminal binding names must be strings")
        })?;
        let token_ids = if let Ok(token_id) = value.extract::<u32>() {
            vec![token_id]
        } else {
            value.extract::<Vec<u32>>().map_err(|_| {
                PyValueError::new_err(format!(
                    "binding {name:?} must be a non-negative token ID or an iterable of token IDs"
                ))
            })?
        };
        result.push((name, token_ids));
    }
    Ok(result)
}

fn llama_cpp_to_vocab(llm: &Bound<'_, PyAny>) -> PyResult<(glrmask::Vocab, Vec<u32>)> {
    let py = llm.py();
    let llama_cpp = py.import("llama_cpp")?;
    let ctypes = py.import("ctypes")?;
    let llama_vocab = llama_cpp
        .getattr("llama_model_get_vocab")?
        .call1((llm.getattr("model")?,))?;
    let n_vocab: u32 = llm.call_method0("n_vocab")?.extract()?;
    let excluded_attrs: u32 = llama_cpp
        .getattr("LLAMA_TOKEN_ATTR_CONTROL")?
        .extract::<u32>()?
        | llama_cpp
            .getattr("LLAMA_TOKEN_ATTR_UNUSED")?
            .extract::<u32>()?;
    let is_eog = llama_cpp.getattr("llama_vocab_is_eog")?;
    let get_attr = llama_cpp.getattr("llama_vocab_get_attr")?;
    let token_to_piece = llama_cpp.getattr("llama_token_to_piece")?;
    let create_string_buffer = ctypes.getattr("create_string_buffer")?;

    let mut entries = Vec::with_capacity(n_vocab as usize);
    let mut end_token_ids = Vec::new();
    let mut exact_only_token_ids = Vec::new();
    for token_id in 0..n_vocab {
        if is_eog.call1((&llama_vocab, token_id))?.is_truthy()? {
            end_token_ids.push(token_id);
            exact_only_token_ids.push(token_id);
            continue;
        }

        let attrs: u32 = get_attr.call1((&llama_vocab, token_id))?.extract()?;
        if attrs & excluded_attrs != 0 {
            exact_only_token_ids.push(token_id);
            continue;
        }

        let required: isize = token_to_piece
            .call1((&llama_vocab, token_id, py.None(), 0, 0, false))?
            .extract()?;
        let capacity = if required < 0 {
            required.checked_neg().ok_or_else(|| {
                PyValueError::new_err(format!(
                    "llama_token_to_piece returned an invalid size for token {token_id}"
                ))
            })?
        } else {
            required
        };
        if capacity == 0 {
            exact_only_token_ids.push(token_id);
            continue;
        }

        let buffer = create_string_buffer.call1((capacity,))?;
        let length: isize = token_to_piece
            .call1((&llama_vocab, token_id, &buffer, capacity, 0, false))?
            .extract()?;
        if length < 0 || length > capacity {
            return Err(PyValueError::new_err(format!(
                "llama_token_to_piece returned invalid length {length} for token {token_id}"
            )));
        }
        if length == 0 {
            exact_only_token_ids.push(token_id);
            continue;
        }

        let raw = buffer.getattr("raw")?.downcast_into::<PyBytes>()?;
        let length = length as usize;
        let raw = raw.as_bytes();
        if length > raw.len() {
            return Err(PyValueError::new_err(format!(
                "llama.cpp wrote {length} bytes for token {token_id} into a {}-byte buffer",
                raw.len()
            )));
        }
        entries.push((token_id, raw[..length].to_vec()));
    }

    Ok((
        glrmask::Vocab::new_with_exact_token_ids(entries, exact_only_token_ids),
        end_token_ids,
    ))
}

fn constraint_result<T, E: std::fmt::Display>(result: Result<T, E>) -> PyResult<T> {
    result.map_err(|e| PyValueError::new_err(format!("{e}")))
}

fn words_to_bool_array<'py>(
    py: Python<'py>,
    words: &[u32],
    token_count: usize,
) -> Bound<'py, PyArray1<bool>> {
    let n = token_count;
    let n_full_words = n / 32;
    let remainder = n % 32;
    let mut bools = vec![false; n];
    for (wi, &word) in words[..n_full_words.min(words.len())].iter().enumerate() {
        let base = wi * 32;
        let mut w = word;
        for bit in &mut bools[base..base + 32] {
            *bit = w & 1 != 0;
            w >>= 1;
        }
    }
    if remainder > 0 && n_full_words < words.len() {
        let base = n_full_words * 32;
        let mut w = words[n_full_words];
        for bit in &mut bools[base..] {
            *bit = w & 1 != 0;
            w >>= 1;
        }
    }
    PyArray1::from_vec(py, bools)
}

fn resolved_mask_size(max_token: u32, requested: Option<usize>) -> PyResult<usize> {
    let minimum = max_token as usize + 1;
    let size = requested.unwrap_or(minimum);
    if size < minimum {
        return Err(PyValueError::new_err(format!(
            "mask size {size} is smaller than the constraint token range {minimum}"
        )));
    }
    Ok(size)
}

fn bitmask_u32_view<'a, 'py>(
    bitmask: &'a mut PyReadwriteArray1<'py, i32>,
) -> PyResult<&'a mut [u32]> {
    let slice = bitmask
        .as_slice_mut()
        .map_err(|e| PyValueError::new_err(format!("Array must be contiguous: {e:?}")))?;
    // Safety: i32 and u32 have identical size, alignment, and bit representation.
    Ok(unsafe {
        std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut u32, slice.len())
    })
}

fn set_gss_summary_fields(
    dict: &Bound<'_, PyDict>,
    prefix: &str,
    path_count: usize,
    summary: &glrmask::__private::GssProfileSummary,
) -> PyResult<()> {
    dict.set_item(format!("{prefix}_path_count"), path_count)?;
    dict.set_item(format!("{prefix}_top_values_count"), summary.top_values_count)?;
    dict.set_item(format!("{prefix}_upper_branch_nodes"), summary.upperbranch_nodes)?;
    dict.set_item(format!("{prefix}_upper_interface_nodes"), summary.interface_nodes)?;
    dict.set_item(format!("{prefix}_lower_nodes"), summary.lower_nodes)?;
    dict.set_item(
        format!("{prefix}_lower_general_nodes"),
        summary.lower_general_nodes,
    )?;
    dict.set_item(
        format!("{prefix}_lower_segment_nodes"),
        summary.lower_segment_nodes,
    )?;
    dict.set_item(
        format!("{prefix}_total_unique_nodes"),
        summary.total_unique_nodes,
    )?;
    dict.set_item(format!("{prefix}_total_edges"), summary.total_edges)?;
    dict.set_item(
        format!("{prefix}_accumulator_instances"),
        summary.accumulator_instances,
    )?;
    dict.set_item(format!("{prefix}_max_depth"), summary.max_depth)?;
    Ok(())
}

fn mask_profile_to_dict<'py>(
    py: Python<'py>,
    profile: glrmask::__private::MaskProfile,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("total_ns", profile.total_ns)?;
    dict.set_item("cache_hit", profile.cache_hit)?;
    dict.set_item("single_path_direct", profile.single_path_direct)?;
    dict.set_item("seed_decompose_ns", profile.seed_decompose_ns)?;
    dict.set_item("queue_pop_ns", profile.queue_pop_ns)?;
    dict.set_item("loop_decompose_ns", profile.loop_decompose_ns)?;
    dict.set_item("loop_decompose_callback_ns", profile.loop_decompose_callback_ns)?;
    dict.set_item("transition_lookup_ns", profile.transition_lookup_ns)?;
    dict.set_item("transition_apply_ns", profile.transition_apply_ns)?;
    dict.set_item(
        "transition_apply_intersect_ns",
        profile.transition_apply_intersect_ns,
    )?;
    dict.set_item("transition_apply_gss_ns", profile.transition_apply_gss_ns)?;
    dict.set_item("token_accumulation_ns", profile.token_accumulation_ns)?;
    dict.set_item("enqueue_merge_ns", profile.enqueue_merge_ns)?;
    dict.set_item("queue_lookup_ns", profile.queue_lookup_ns)?;
    dict.set_item("queue_merge_ns", profile.queue_merge_ns)?;
    dict.set_item("queue_insert_ns", profile.queue_insert_ns)?;
    dict.set_item("queue_fuse_ns", profile.queue_fuse_ns)?;
    dict.set_item("finalize_ns", profile.finalize_ns)?;
    dict.set_item("finalize_zero_ns", profile.finalize_zero_ns)?;
    dict.set_item("finalize_dense_to_buf_ns", profile.finalize_dense_to_buf_ns)?;
    dict.set_item("finalize_cache_ns", profile.finalize_cache_ns)?;
    dict.set_item("delta_prev_available", profile.delta_prev_available)?;
    dict.set_item("delta_added_bits", profile.delta_added_bits)?;
    dict.set_item("delta_removed_bits", profile.delta_removed_bits)?;
    dict.set_item("delta_unchanged_words", profile.delta_unchanged_words)?;
    dict.set_item("delta_unchanged_bits", profile.delta_unchanged_bits)?;
    dict.set_item("delta_added_cost", profile.delta_added_cost)?;
    dict.set_item("delta_removed_cost", profile.delta_removed_cost)?;
    dict.set_item("delta_copy_cost_words", profile.delta_copy_cost_words)?;
    dict.set_item(
        "delta_scratch_estimated_cost",
        profile.delta_scratch_estimated_cost,
    )?;
    dict.set_item("delta_estimated_cost", profile.delta_estimated_cost)?;
    dict.set_item("delta_estimated_savings", profile.delta_estimated_savings)?;
    dict.set_item("delta_used_seed", profile.delta_used_seed)?;
    dict.set_item(
        "delta_added_word_group_hits",
        profile.delta_added_word_group_hits,
    )?;
    dict.set_item(
        "delta_added_word_group_entries",
        profile.delta_added_word_group_entries,
    )?;
    dict.set_item(
        "delta_removed_word_group_hits",
        profile.delta_removed_word_group_hits,
    )?;
    dict.set_item(
        "delta_removed_word_group_entries",
        profile.delta_removed_word_group_entries,
    )?;
    dict.set_item(
        "delta_added_byte_group_hits",
        profile.delta_added_byte_group_hits,
    )?;
    dict.set_item(
        "delta_added_byte_group_entries",
        profile.delta_added_byte_group_entries,
    )?;
    dict.set_item(
        "delta_removed_byte_group_hits",
        profile.delta_removed_byte_group_hits,
    )?;
    dict.set_item(
        "delta_removed_byte_group_entries",
        profile.delta_removed_byte_group_entries,
    )?;
    dict.set_item(
        "delta_added_token_iterations",
        profile.delta_added_token_iterations,
    )?;
    dict.set_item("delta_added_token_entries", profile.delta_added_token_entries)?;
    dict.set_item(
        "delta_removed_token_iterations",
        profile.delta_removed_token_iterations,
    )?;
    dict.set_item(
        "delta_removed_token_entries",
        profile.delta_removed_token_entries,
    )?;
    dict.set_item(
        "finalize_equal_dense_copy_seed",
        profile.finalize_equal_dense_copy_seed,
    )?;
    dict.set_item("finalize_delta_replay", profile.finalize_delta_replay)?;
    dict.set_item("finalize_scratch_rebuild", profile.finalize_scratch_rebuild)?;
    dict.set_item("dense_words_visited", profile.dense_words_visited)?;
    dict.set_item(
        "dense_complement_path_used",
        profile.dense_complement_path_used,
    )?;
    dict.set_item(
        "dense_normal_full_word_hits",
        profile.dense_normal_full_word_hits,
    )?;
    dict.set_item(
        "dense_normal_group_complement_hits",
        profile.dense_normal_group_complement_hits,
    )?;
    dict.set_item(
        "dense_complement_full_word_hits",
        profile.dense_complement_full_word_hits,
    )?;
    dict.set_item(
        "dense_complement_full_byte_groups",
        profile.dense_complement_full_byte_groups,
    )?;
    dict.set_item(
        "dense_complement_full_nibble_groups",
        profile.dense_complement_full_nibble_groups,
    )?;
    dict.set_item(
        "dense_complement_remaining_bits",
        profile.dense_complement_remaining_bits,
    )?;
    dict.set_item(
        "dense_normal_token_iterations",
        profile.dense_normal_token_iterations,
    )?;
    dict.set_item(
        "dense_complement_token_iterations",
        profile.dense_complement_token_iterations,
    )?;
    dict.set_item(
        "dense_normal_sparse_entries",
        profile.dense_normal_sparse_entries,
    )?;
    dict.set_item(
        "dense_normal_group_complement_sparse_entries",
        profile.dense_normal_group_complement_sparse_entries,
    )?;
    dict.set_item(
        "dense_complement_sparse_entries",
        profile.dense_complement_sparse_entries,
    )?;
    dict.set_item(
        "dense_complement_heavy_dense_clears",
        profile.dense_complement_heavy_dense_clears,
    )?;
    dict.set_item(
        "dense_complement_max_sparse_span",
        profile.dense_complement_max_sparse_span,
    )?;
    dict.set_item("dense_group_or_sparse_entries", profile.dense_group_or_sparse_entries)?;
    dict.set_item(
        "dense_group_andnot_sparse_entries",
        profile.dense_group_andnot_sparse_entries,
    )?;
    dict.set_item("enqueue_calls", profile.enqueue_calls)?;
    dict.set_item("merge_hits", profile.merge_hits)?;
    dict.set_item(
        "insert_without_merge_count",
        profile.insert_without_merge_count,
    )?;
    dict.set_item("fuse_calls", profile.fuse_calls)?;
    dict.set_item("fuse_changed_depth", profile.fuse_changed_depth)?;
    dict.set_item("stale_schedule_skips", profile.stale_schedule_skips)?;
    dict.set_item("popped_items", profile.popped_items)?;
    dict.set_item("seed_decompose_callbacks", profile.seed_decompose_callbacks)?;
    dict.set_item("loop_decompose_callbacks", profile.loop_decompose_callbacks)?;
    dict.set_item(
        "parser_dwa_transitions_enqueued",
        profile.parser_dwa_transitions_enqueued,
    )?;
    dict.set_item("other_ns", profile.other_ns)?;
    Ok(dict)
}
fn string_result<T, E: std::fmt::Display>(result: Result<T, E>) -> PyResult<T> {
    result.map_err(|error| PyValueError::new_err(error.to_string()))
}

fn advance_trace_to_dict<'py>(
    py: Python<'py>,
    trace: &glrmask::__private::AdvanceTrace,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);

    let det_steps = pyo3::types::PyList::empty(py);
    for step in &trace.det_steps {
        det_steps.append(advance_trace_step_to_dict(py, step)?)?;
    }
    dict.set_item("det_steps", det_steps)?;

    let nondet_waves = pyo3::types::PyList::empty(py);
    for wave in &trace.nondet_waves {
        let wave_dict = PyDict::new(py);
        wave_dict.set_item("wave_index", wave.wave_index)?;
        wave_dict.set_item("frontier_states", wave.frontier_states.clone())?;
        let branches = pyo3::types::PyList::empty(py);
        for branch in &wave.branches {
            branches.append(advance_trace_step_to_dict(py, branch)?)?;
        }
        wave_dict.set_item("branches", branches)?;
        nondet_waves.append(wave_dict)?;
    }
    dict.set_item("nondet_waves", nondet_waves)?;

    Ok(dict)
}

fn advance_trace_step_to_dict<'py>(
    py: Python<'py>,
    step: &glrmask::__private::AdvanceTraceStep,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("source_state", step.source_state)?;
    dict.set_item("action_kind", step.action_kind.as_str())?;
    if let Some(target) = step.shift_target {
        dict.set_item("shift_target", target)?;
    }
    if let Some(replace) = step.shift_replace {
        dict.set_item("shift_replace", replace)?;
    }
    let reduces = pyo3::types::PyList::empty(py);
    for reduce in &step.reduces {
        let reduce_dict = PyDict::new(py);
        reduce_dict.set_item("lhs_nt", reduce.lhs_nt)?;
        if let Some(lhs_name) = &reduce.lhs_name {
            reduce_dict.set_item("lhs_name", lhs_name.as_str())?;
        }
        reduce_dict.set_item("pop_len", reduce.pop_len)?;
        reduce_dict.set_item("goto_sources", reduce.goto_sources.clone())?;
        let goto_targets = pyo3::types::PyList::empty(py);
        for goto in &reduce.goto_targets {
            let goto_dict = PyDict::new(py);
            goto_dict.set_item("source_state", goto.source_state)?;
            goto_dict.set_item("target_state", goto.target_state)?;
            goto_dict.set_item("replace", goto.replace)?;
            goto_targets.append(goto_dict)?;
        }
        reduce_dict.set_item("goto_targets", goto_targets)?;
        reduces.append(reduce_dict)?;
    }
    dict.set_item("reduces", reduces)?;
    Ok(dict)
}

// ---------------------------------------------------------------------------
// PyVocab
// ---------------------------------------------------------------------------

#[pyclass(name = "Vocab")]
#[derive(Clone)]
pub struct PyVocab {
    inner: glrmask::Vocab,
    llama_cpp_end_token_ids: Vec<u32>,
}

#[pymethods]
impl PyVocab {
    /// Resolve one exact token, preserving this complete vocabulary identity.
    fn token(&self, id: u32) -> PyResult<api::PyExactToken> {
        self.inner.token(id).map(|inner| api::PyExactToken { inner })
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// Resolve a nonempty set of distinct exact token IDs.
    fn tokens(&self, ids: Vec<u32>) -> PyResult<api::PyExactTokens> {
        self.inner.tokens(ids).map(|inner| api::PyExactTokens { inner })
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn __len__(&self) -> usize { self.inner.len() }

    #[staticmethod]
    fn from_dict(token_to_id: &Bound<'_, PyDict>) -> PyResult<Self> {
        let vocab = dict_to_vocab(token_to_id)?;
        Ok(Self {
            inner: vocab,
            llama_cpp_end_token_ids: Vec::new(),
        })
    }

    #[staticmethod]
    fn from_id_to_bytes(id_to_bytes: &Bound<'_, PyDict>) -> PyResult<Self> {
        let vocab = id_to_bytes_dict_to_vocab(id_to_bytes)?;
        Ok(Self {
            inner: vocab,
            llama_cpp_end_token_ids: Vec::new(),
        })
    }

    /// Build the byte vocabulary used by a llama-cpp-python `Llama` model.
    ///
    /// EOG, control, unused, and empty-piece tokens are omitted from the byte
    /// vocabulary. EOG IDs remain available through `llama_cpp_end_token_ids`
    /// for the decoder's stopping policy.
    #[staticmethod]
    fn from_llama_cpp(llm: &Bound<'_, PyAny>) -> PyResult<Self> {
        let (vocab, llama_cpp_end_token_ids) = llama_cpp_to_vocab(llm)?;
        Ok(Self {
            inner: vocab,
            llama_cpp_end_token_ids,
        })
    }

    #[getter]
    fn llama_cpp_end_token_ids(&self) -> Vec<u32> {
        self.llama_cpp_end_token_ids.clone()
    }

}









// ---------------------------------------------------------------------------
// PyConstraint
// ---------------------------------------------------------------------------

/// Compiled grammar constraint. Immutable, thread-safe.
#[pyclass(name = "Constraint")]
#[derive(Clone)]
pub struct PyConstraint {
    inner: Arc<glrmask::Constraint>,
    max_token: u32,
}

impl PyConstraint {
    fn from_constraint_result<E: std::fmt::Display>(
        constraint: Result<glrmask::Constraint, E>,
        _vocab: &PyVocab,
    ) -> PyResult<Self> {
        let constraint = constraint_result(constraint)?;
        let max_token = constraint.max_original_token_id().unwrap_or(0);
        Ok(Self {
            inner: Arc::new(constraint),
            max_token,
        })
    }


    /// Return the number of GLR parser states.
    fn num_parser_states(&self) -> u32 {
        self.inner.num_parser_states()
    }

    /// Return display names for grammar terminals by terminal id.
    fn terminal_display_names(&self) -> Vec<String> {
        self.inner.terminal_display_names().to_vec()
    }

    /// Return the display name for a grammar terminal id, if present.
    fn terminal_display_name(&self, terminal_id: u32) -> Option<String> {
        self.inner
            .terminal_display_name(terminal_id)
            .map(str::to_string)
    }
}

#[pymethods]
impl PyConstraint {
    /// Serialize the compiled body and final termination policy as bytes.
    fn save<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let bytes = py.allow_threads(|| self.inner.save());
        PyBytes::new(py, &bytes)
    }

    #[staticmethod]
    #[pyo3(signature = (data, vocab=None))]
    fn load(py: Python<'_>, data: &[u8], vocab: Option<&PyVocab>) -> PyResult<Self> {
        let data = data.to_vec();
        let vocab = vocab.map(|vocab| vocab.inner.clone());
        let loaded = py.allow_threads(move || match vocab {
            Some(vocab) => glrmask::Constraint::load_with_vocab(data, &vocab),
            None => glrmask::Constraint::load(data),
        });
        let inner = constraint_result(loaded)?;
        let max_token = inner.max_original_token_id().unwrap_or(0);
        Ok(Self { inner: Arc::new(inner), max_token })
    }

    fn start(&self) -> PyConstraintState {
        PyConstraintState {
            inner: OwnedState::from_arc(self.inner.clone()),
            max_token: self.max_token,
        }
    }

    fn mask_len(&self) -> usize {
        self.inner.mask_len()
    }

}

// ---------------------------------------------------------------------------
// PyDynamicConstraint
// ---------------------------------------------------------------------------

#[pyclass(name = "DynamicConstraint")]
#[derive(Clone)]
pub struct PyDynamicConstraint {
    inner: Arc<glrmask::DynamicConstraint>,
    max_token: u32,
}

impl PyDynamicConstraint {
    fn from_constraint_result<E: std::fmt::Display>(
        constraint: Result<glrmask::DynamicConstraint, E>,
        _vocab: &PyVocab,
    ) -> PyResult<Self> {
        let constraint = constraint_result(constraint)?;
        let max_token = constraint.max_original_token_id().unwrap_or(0);
        Ok(Self {
            inner: Arc::new(constraint),
            max_token,
        })
    }
}

#[pymethods]
impl PyDynamicConstraint {
    #[staticmethod]
    #[pyo3(signature = (schema, vocab, vocab_partition=false))]
    fn from_json_schema(schema: &str, vocab: &PyVocab, vocab_partition: bool) -> PyResult<Self> {
        let grammar = glrmask::Grammar::json_schema(schema);
        let constraint = if vocab_partition {
            glrmask::DynamicConstraint::compile_with_vocab_partition(grammar, &vocab.inner)
        } else {
            glrmask::DynamicConstraint::compile(grammar, &vocab.inner)
        };
        Self::from_constraint_result(constraint, vocab)
    }

    #[staticmethod]
    #[pyo3(signature = (lark_source, vocab, vocab_partition=false))]
    fn from_lark(lark_source: &str, vocab: &PyVocab, vocab_partition: bool) -> PyResult<Self> {
        let grammar = glrmask::Grammar::lark(lark_source);
        let constraint = if vocab_partition {
            glrmask::DynamicConstraint::compile_with_vocab_partition(grammar, &vocab.inner)
        } else {
            glrmask::DynamicConstraint::compile(grammar, &vocab.inner)
        };
        Self::from_constraint_result(constraint, vocab)
    }

    #[staticmethod]
    #[pyo3(signature = (glrm_source, vocab, subgrammars=None, bindings=None, vocab_partition=false))]
    fn from_glrm_grammar(
        py: Python<'_>,
        glrm_source: &str,
        vocab: &PyVocab,
        subgrammars: Option<BTreeMap<String, Py<PyConstraint>>>,
        bindings: Option<&Bound<'_, PyDict>>,
        vocab_partition: bool,
    ) -> PyResult<Self> {
        if vocab_partition {
            if subgrammars.as_ref().is_some_and(|children| !children.is_empty())
                || bindings.is_some()
            {
                return Err(PyValueError::new_err(
                    "vocab_partition=True is not yet supported together with GLRM subgrammars or token bindings",
                ));
            }
            return Self::from_constraint_result(
                glrmask::DynamicConstraint::compile_with_vocab_partition(
                    glrmask::Grammar::glrm(glrm_source),
                    &vocab.inner,
                ),
                vocab,
            );
        }
        let mut builder = glrmask::ConstraintSpec::builder(
            glrmask::Grammar::glrm(glrm_source),
            &vocab.inner,
        )
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
        for (name, token_ids) in external_terminal_bindings_from_dict(bindings)? {
            builder = builder
                .bind_token(name, token_ids)
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
        }
        if let Some(subgrammars) = subgrammars {
            for (name, child) in subgrammars {
                let child = child.borrow(py);
                builder = builder
                    .bind_grammar(name, Arc::clone(&child.inner))
                    .map_err(|error| PyValueError::new_err(error.to_string()))?;
            }
        }
        let spec = builder
            .build()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Self::from_constraint_result(
            spec.compile_dynamic(),
            vocab,
        )
    }

    #[staticmethod]
    #[pyo3(signature = (ebnf_source, vocab, vocab_partition=false))]
    fn from_ebnf(ebnf_source: &str, vocab: &PyVocab, vocab_partition: bool) -> PyResult<Self> {
        let grammar = glrmask::Grammar::ebnf(ebnf_source);
        let constraint = if vocab_partition {
            glrmask::DynamicConstraint::compile_with_vocab_partition(grammar, &vocab.inner)
        } else {
            glrmask::DynamicConstraint::compile(grammar, &vocab.inner)
        };
        Self::from_constraint_result(constraint, vocab)
    }

    #[staticmethod]
    fn load(data: &[u8], vocab: &PyVocab) -> PyResult<Self> {
        Self::from_constraint_result(
            glrmask::DynamicConstraint::load_with_vocab(data, &vocab.inner),
            vocab,
        )
    }

    fn save(&self) -> Vec<u8> {
        // DynamicConstraint.load() already requires the vocabulary, so Python
        // persistence need not duplicate vocabulary bytes in the artifact.
        self.inner.save_with_external_vocab()
    }

    fn mask_len(&self) -> usize {
        self.inner.mask_len()
    }

    fn start(&self) -> PyDynamicConstraintState {
        PyDynamicConstraintState {
            inner: OwnedDynamicState::from_arc(self.inner.clone()),
            max_token: self.max_token,
        }
    }
}

#[pyclass(name = "DynamicConstraintState")]
pub struct PyDynamicConstraintState {
    inner: OwnedDynamicState,
    max_token: u32,
}

#[pymethods]
impl PyDynamicConstraintState {
    fn commit_bytes(&mut self, data: &[u8]) -> PyResult<()> {
        self.inner
            .with_dependent_mut(|_owner, state| string_result(state.commit_bytes(data)))
    }

    fn commit_token(&mut self, token_id: u32) -> PyResult<()> {
        self.inner
            .with_dependent_mut(|_owner, state| string_result(state.commit_token(token_id)))
    }

    /// Diagnostic commit profile for the common single-alternative dynamic
    /// constraint. This mirrors ConstraintState.commit_token_profiled so CFA's
    /// step profiler can inspect O1/O2 commit tails directly.
    fn commit_token_profiled<'py>(
        &mut self,
        py: Python<'py>,
        token_id: u32,
    ) -> PyResult<Bound<'py, PyDict>> {
        let profile = self.inner.with_dependent_mut(|_owner, state| {
            state
                .commit_token_profiled(token_id)
                .map_err(PyValueError::new_err)
        })?;
        let dict = PyDict::new(py);
        dict.set_item("total_ns", profile.total_ns)?;
        dict.set_item("scan_ns", profile.scan_ns)?;
        dict.set_item("prune_ns", profile.prune_ns)?;
        dict.set_item("queue_ns", profile.queue_ns)?;
        dict.set_item("fuse_ns", profile.fuse_ns)?;
        dict.set_item("initial_exec_ns", profile.initial_exec_ns)?;
        dict.set_item("exec_ns", profile.exec_ns)?;
        dict.set_item("queue_exec_ns", profile.queue_exec_ns)?;
        dict.set_item("queue_match_ns", profile.queue_match_ns)?;
        dict.set_item("queue_enqueue_ns", profile.queue_enqueue_ns)?;
        dict.set_item("queue_bookkeeping_ns", profile.queue_bookkeeping_ns)?;
        dict.set_item("advance_ns", profile.advance_ns)?;
        dict.set_item("advance_may_check_ns", profile.advance_may_check_ns)?;
        dict.set_item("advance_core_ns", profile.advance_core_ns)?;
        dict.set_item("advance_future_disallow_ns", profile.advance_future_disallow_ns)?;
        dict.set_item("actionable_ns", profile.actionable_ns)?;
        dict.set_item("may_advance_ns", profile.may_advance_ns)?;
        dict.set_item("n_tokenizer_states", profile.n_tokenizer_states)?;
        dict.set_item("n_queue_entries", profile.n_queue_entries)?;
        dict.set_item("n_advances", profile.n_advances)?;
        dict.set_item("fast_path_total_ns", profile.fast_path_total_ns)?;
        dict.set_item("fast_path_tokenizer_exec_ns", profile.fast_path_tokenizer_exec_ns)?;
        dict.set_item("fast_path_match_scan_ns", profile.fast_path_match_scan_ns)?;
        dict.set_item("fast_path_end_state_check_ns", profile.fast_path_end_state_check_ns)?;
        dict.set_item("fast_path_prune_ns", profile.fast_path_prune_ns)?;
        dict.set_item("fast_path_advance_ns", profile.fast_path_advance_ns)?;
        dict.set_item("fast_path_future_disallow_ns", profile.fast_path_future_disallow_ns)?;
        dict.set_item("fast_path_fuse_ns", profile.fast_path_fuse_ns)?;
        dict.set_item("fast_path_state_update_ns", profile.fast_path_state_update_ns)?;
        dict.set_item("failed_fast_path_probe_ns", profile.failed_fast_path_probe_ns)?;
        dict.set_item("linear_fast_path_total_ns", profile.linear_fast_path_total_ns)?;
        dict.set_item("linear_fast_path_exec_ns", profile.linear_fast_path_exec_ns)?;
        dict.set_item("linear_fast_path_match_scan_ns", profile.linear_fast_path_match_scan_ns)?;
        dict.set_item("linear_fast_path_end_state_check_ns", profile.linear_fast_path_end_state_check_ns)?;
        dict.set_item("linear_fast_path_advance_ns", profile.linear_fast_path_advance_ns)?;
        dict.set_item("linear_fast_path_fuse_ns", profile.linear_fast_path_fuse_ns)?;
        Ok(dict)
    }

    fn fill_mask(&self, mut bitmask: PyReadwriteArray1<i32>) -> PyResult<()> {
        let buf = bitmask_u32_view(&mut bitmask)?;
        self.inner
            .with_dependent(|_owner, state| state.fill_mask(buf));
        Ok(())
    }

    #[doc(hidden)]
    fn fill_mask_timed_ns(&self, mut bitmask: PyReadwriteArray1<i32>) -> PyResult<u64> {
        let buf = bitmask_u32_view(&mut bitmask)?;
        Ok(self.inner.with_dependent(|_owner, state| {
            let start = Instant::now();
            state.fill_mask(buf);
            start.elapsed().as_nanos() as u64
        }))
    }


    fn forced(&self) -> Vec<u32> {
        self.inner.with_dependent(|_owner, state| state.forced())
    }

    fn is_accepting(&self) -> bool {
        self.inner
            .with_dependent(|_owner, state| state.is_accepting())
    }

    fn is_rejected(&self) -> bool {
        self.inner
            .with_dependent(|_owner, state| state.is_rejected())
    }

    #[pyo3(signature = (size=None))]
    fn mask<'py>(
        &self,
        py: Python<'py>,
        size: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let size = resolved_mask_size(self.max_token, size)?;
        let mut words = vec![0u32; size.div_ceil(32)];
        self.inner
            .with_dependent(|_owner, state| state.fill_mask(&mut words));
        Ok(words_to_bool_array(py, &words, size))
    }
}

// ---------------------------------------------------------------------------
// PyConstraintState
// ---------------------------------------------------------------------------

/// Mutable per-sequence parse state.
#[pyclass(name = "ConstraintState")]
pub struct PyConstraintState {
    inner: OwnedState,
    max_token: u32,
}

#[pymethods]
impl PyConstraintState {
    /// Whether an allowed final end token has completed this sequence.
    fn is_terminated(&self) -> bool {
        self.inner.with_dependent(|_owner, state| state.is_terminated())
    }

    #[pyo3(signature = (size=None))]
    fn mask<'py>(
        &self,
        py: Python<'py>,
        size: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let size = resolved_mask_size(self.max_token, size)?;
        let mut words = vec![0u32; size.div_ceil(32)];
        self.inner
            .with_dependent(|_owner, state| state.fill_mask(&mut words));
        Ok(words_to_bool_array(py, &words, size))
    }

    fn fill_mask(&self, mut bitmask: PyReadwriteArray1<i32>) -> PyResult<()> {
        let slice = bitmask.as_slice_mut().map_err(|e| {
            PyValueError::new_err(format!("Array must be contiguous: {e:?}"))
        })?;
        // Safety: i32 and u32 have identical size, alignment, and bit representation.
        // fill_mask writes valid u32 bitmask values where the high bit is meaningful.
        let buf: &mut [u32] = unsafe {
            std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut u32, slice.len())
        };
        let required = self.inner.borrow_owner().mask_len();
        if buf.len() < required {
            return Err(PyValueError::new_err(format!("mask needs at least {required} packed words")));
        }
        self.inner.with_dependent(|_owner, state| state.fill_mask(buf));
        Ok(())
    }

    fn commit_token(&mut self, token_id: u32) -> PyResult<()> {
        self.inner
            .with_dependent_mut(|_owner, state| string_result(state.commit_token(token_id)))
    }

    fn commit_bytes(&mut self, data: &[u8]) -> PyResult<()> {
        self.inner
            .with_dependent_mut(|_owner, state| string_result(state.commit_bytes(data)))
    }

    fn is_rejected(&self) -> bool {
        self.inner.with_dependent(|_owner, state| state.is_rejected())
    }

    fn forced(&self) -> Vec<u32> {
        self.inner.with_dependent(|_owner, state| state.forced())
    }

    fn is_accepting(&self) -> bool {
        self.inner.with_dependent(|_owner, state| state.is_accepting())
    }

    #[cfg(feature = "allocation-tracking")]
    #[pyo3(name = "fill_mask_timed_allocation_stats")]
    fn py_fill_mask_timed_allocation_stats(
        &self,
        bitmask: PyReadwriteArray1<i32>,
    ) -> PyResult<Vec<u64>> {
        self.fill_mask_timed_allocation_stats(bitmask)
    }

    #[cfg(feature = "allocation-tracking")]
    #[pyo3(name = "commit_token_timed_allocation_stats")]
    fn py_commit_token_timed_allocation_stats(&mut self, token_id: u32) -> PyResult<Vec<u64>> {
        self.commit_token_timed_allocation_stats(token_id)
    }
}

#[cfg(feature = "allocation-tracking")]
fn allocation_stats_tuple(
    elapsed_ns: u64,
    stats: allocation_tracking::AllocationStats,
) -> Vec<u64> {
    vec![
        elapsed_ns,
        stats.alloc_calls,
        stats.alloc_zeroed_calls,
        stats.realloc_calls,
        stats.dealloc_calls,
        stats.allocated_bytes,
        stats.reallocated_bytes,
        stats.deallocated_bytes,
        stats.alloc_ns,
        stats.max_alloc_ns,
        stats.realloc_ns,
        stats.max_realloc_ns,
        stats.dealloc_ns,
        stats.max_dealloc_ns,
    ]
}

impl PyConstraintState {
    fn fill_mask_timed_ns(&self, mut bitmask: PyReadwriteArray1<i32>) -> PyResult<u64> {
        let slice = bitmask.as_slice_mut().map_err(|e| {
            PyValueError::new_err(format!("Array must be contiguous: {e:?}"))
        })?;
        let buf: &mut [u32] = unsafe {
            std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut u32, slice.len())
        };
        Ok(self.inner.with_dependent(|_owner, state| state.fill_mask_timed_ns(buf)))
    }

    #[cfg(feature = "allocation-tracking")]
    fn fill_mask_timed_allocation_stats(
        &self,
        mut bitmask: PyReadwriteArray1<i32>,
    ) -> PyResult<Vec<u64>> {
        let slice = bitmask.as_slice_mut().map_err(|e| {
            PyValueError::new_err(format!("Array must be contiguous: {e:?}"))
        })?;
        let buf: &mut [u32] = unsafe {
            std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut u32, slice.len())
        };
        let (elapsed_ns, stats) = allocation_tracking::measure(|| {
            self.inner
                .with_dependent(|_owner, state| state.fill_mask_timed_ns(buf))
        });
        Ok(allocation_stats_tuple(elapsed_ns, stats))
    }

    fn fill_mask_profiled<'py>(
        &self,
        py: Python<'py>,
        mut bitmask: PyReadwriteArray1<i32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let slice = bitmask.as_slice_mut().map_err(|e| {
            PyValueError::new_err(format!("Array must be contiguous: {e:?}"))
        })?;
        let buf: &mut [u32] = unsafe {
            std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut u32, slice.len())
        };
        let profile = self
            .inner
            .with_dependent(|_owner, state| state.fill_mask_profiled(buf));
        mask_profile_to_dict(py, profile)
    }

    fn commit_token_timed_ns(&mut self, token_id: u32) -> PyResult<u64> {
        self.inner.with_dependent_mut(|_owner, state| {
            state
                .commit_token_timed_ns(token_id)
                .map_err(PyValueError::new_err)
        })
    }

    #[cfg(feature = "allocation-tracking")]
    fn commit_token_timed_allocation_stats(
        &mut self,
        token_id: u32,
    ) -> PyResult<Vec<u64>> {
        let (result, stats) = allocation_tracking::measure(|| {
            self.inner.with_dependent_mut(|_owner, state| {
                state.commit_token_timed_ns(token_id)
            })
        });
        let elapsed_ns = result.map_err(PyValueError::new_err)?;
        Ok(allocation_stats_tuple(elapsed_ns, stats))
    }

    /// Like commit_token but returns profiling stats as a dict.
    fn commit_token_profiled<'py>(&mut self, py: Python<'py>, token_id: u32) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let profile = self.inner.with_dependent_mut(|_owner, state| {
            state.commit_token_profiled(token_id).map_err(|e| PyValueError::new_err(e))
        })?;
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("total_ns", profile.total_ns)?;
        dict.set_item("scan_ns", profile.scan_ns)?;
        dict.set_item("prune_ns", profile.prune_ns)?;
        dict.set_item("queue_ns", profile.queue_ns)?;
        dict.set_item("fuse_ns", profile.fuse_ns)?;
        dict.set_item("initial_exec_ns", profile.initial_exec_ns)?;
        dict.set_item("exec_ns", profile.exec_ns)?;
        dict.set_item("queue_exec_ns", profile.queue_exec_ns)?;
        dict.set_item("queue_match_ns", profile.queue_match_ns)?;
        dict.set_item("queue_enqueue_ns", profile.queue_enqueue_ns)?;
        dict.set_item("queue_bookkeeping_ns", profile.queue_bookkeeping_ns)?;
        dict.set_item("advance_ns", profile.advance_ns)?;
        dict.set_item("advance_may_check_ns", profile.advance_may_check_ns)?;
        dict.set_item("advance_core_ns", profile.advance_core_ns)?;
        dict.set_item("advance_future_disallow_ns", profile.advance_future_disallow_ns)?;
        dict.set_item("actionable_ns", profile.actionable_ns)?;
        dict.set_item("may_advance_ns", profile.may_advance_ns)?;
        dict.set_item("n_tokenizer_states", profile.n_tokenizer_states)?;
        dict.set_item("n_queue_entries", profile.n_queue_entries)?;
        dict.set_item("n_advances", profile.n_advances)?;
        dict.set_item("adv_n_reduces_above_floor", profile.adv_n_reduces_above_floor)?;
        dict.set_item("adv_n_floor_crossings", profile.adv_n_floor_crossings)?;
        dict.set_item("adv_n_nondet_waves", profile.adv_n_nondet_waves)?;
        dict.set_item("adv_n_nondet_branches", profile.adv_n_nondet_branches)?;
        dict.set_item("adv_clone_ns", profile.adv_clone_ns)?;
        dict.set_item("adv_fast_path_ns", profile.adv_fast_path_ns)?;
        dict.set_item("adv_stack_shift_apply_ns", profile.adv_stack_shift_apply_ns)?;
        dict.set_item("adv_det_ns", profile.adv_det_ns)?;
        dict.set_item("adv_det_floor_cross_ns", profile.adv_det_floor_cross_ns)?;
        dict.set_item("adv_nondet_ns", profile.adv_nondet_ns)?;
        dict.set_item("adv_vstack_len", profile.adv_vstack_len)?;
        dict.set_item("adv_gss_depth", profile.adv_gss_depth)?;
        dict.set_item("adv_det_exit_reason", profile.adv_det_exit_reason)?;
        dict.set_item("adv_det_exit_state", profile.adv_det_exit_state)?;
        dict.set_item("adv_n_det_action_lookups", profile.adv_n_det_action_lookups)?;
        dict.set_item("adv_n_det_goto_lookups", profile.adv_n_det_goto_lookups)?;
        dict.set_item("adv_n_det_popn_ops", profile.adv_n_det_popn_ops)?;
        dict.set_item("adv_n_nondet_reduce_ops", profile.adv_n_nondet_reduce_ops)?;
        dict.set_item("adv_n_nondet_merges", profile.adv_n_nondet_merges)?;
        dict.set_item("adv_n_nondet_isolates", profile.adv_n_nondet_isolates)?;
        dict.set_item("adv_nondet_det_ns", profile.adv_nondet_det_ns)?;
        dict.set_item(
            "adv_nondet_det_floor_cross_ns",
            profile.adv_nondet_det_floor_cross_ns,
        )?;
        dict.set_item("adv_summary_ns", profile.adv_summary_ns)?;
        dict.set_item("fast_path_total_ns", profile.fast_path_total_ns)?;
        dict.set_item("fast_path_tokenizer_exec_ns", profile.fast_path_tokenizer_exec_ns)?;
        dict.set_item("fast_path_match_scan_ns", profile.fast_path_match_scan_ns)?;
        dict.set_item("fast_path_end_state_check_ns", profile.fast_path_end_state_check_ns)?;
        dict.set_item("fast_path_prune_ns", profile.fast_path_prune_ns)?;
        dict.set_item("fast_path_advance_ns", profile.fast_path_advance_ns)?;
        dict.set_item("fast_path_future_disallow_ns", profile.fast_path_future_disallow_ns)?;
        dict.set_item("fast_path_fuse_ns", profile.fast_path_fuse_ns)?;
        dict.set_item("fast_path_state_update_ns", profile.fast_path_state_update_ns)?;
        dict.set_item("failed_fast_path_probe_ns", profile.failed_fast_path_probe_ns)?;
        dict.set_item("linear_fast_path_total_ns", profile.linear_fast_path_total_ns)?;
        dict.set_item("linear_fast_path_exec_ns", profile.linear_fast_path_exec_ns)?;
        dict.set_item("linear_fast_path_match_scan_ns", profile.linear_fast_path_match_scan_ns)?;
        dict.set_item("linear_fast_path_end_state_check_ns", profile.linear_fast_path_end_state_check_ns)?;
        dict.set_item("linear_fast_path_advance_ns", profile.linear_fast_path_advance_ns)?;
        dict.set_item("linear_fast_path_action_lookup_ns", profile.linear_fast_path_action_lookup_ns)?;
        dict.set_item("linear_fast_path_carried_gate_ns", profile.linear_fast_path_carried_gate_ns)?;
        dict.set_item("linear_fast_path_materialize_ns", profile.linear_fast_path_materialize_ns)?;
        dict.set_item("linear_fast_path_apply_action_wall_ns", profile.linear_fast_path_apply_action_wall_ns)?;
        dict.set_item("linear_fast_path_profile_bookkeeping_ns", profile.linear_fast_path_profile_bookkeeping_ns)?;
        dict.set_item("linear_fast_path_future_disallow_ns", profile.linear_fast_path_future_disallow_ns)?;
        dict.set_item("linear_fast_path_fuse_ns", profile.linear_fast_path_fuse_ns)?;
        dict.set_item("linear_fast_path_eligibility_ns", profile.linear_fast_path_eligibility_ns)?;
        dict.set_item("linear_fast_path_setup_ns", profile.linear_fast_path_setup_ns)?;
        dict.set_item("linear_fast_path_state_update_ns", profile.linear_fast_path_state_update_ns)?;
        dict.set_item("linear_fast_path_steps", profile.linear_fast_path_steps)?;
        Ok(dict)
    }

    /// Return total parser GSS root count across all tokenizer states.
    fn parser_root_count(&self) -> usize {
        self.inner.with_dependent(|_owner, state| state.parser_root_count())
    }

    /// Return parser path count (capped at limit).
    fn parser_path_count(&self, limit: usize) -> usize {
        self.inner.with_dependent(|_owner, state| state.parser_path_count(limit))
    }

    /// Return all flattened parser stacks for debugging.
    fn debug_parser_stacks(&self) -> Vec<(u32, Vec<(Vec<u32>, Vec<(u32, Vec<u32>)>)>)> {
        self.inner.with_dependent(|_owner, state| state.debug_parser_stacks())
    }

    /// Per-advance profiling: returns a list of per-advance entries and final GSS stacks.
    fn commit_token_per_advance<'py>(&mut self, py: Python<'py>, token_id: u32) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let (advances, final_stacks, commit_profile) = self.inner.with_dependent_mut(|_owner, state| {
            state.commit_token_per_advance(token_id).map_err(|e| PyValueError::new_err(e))
        })?;

        let result = pyo3::types::PyDict::new(py);

        // Convert advances to list of dicts
        let advance_list = pyo3::types::PyList::empty(py);
        for entry in advances {
            let d = pyo3::types::PyDict::new(py);
            let gss_stacks_before_len = entry.gss_stacks_before.len();
            let gss_stacks_after_len = entry.gss_stacks_after.len();
            d.set_item("terminal_id", entry.terminal_id)?;
            d.set_item("tokenizer_state", entry.tokenizer_state)?;
            d.set_item("gss_stacks_before", entry.gss_stacks_before)?;
            d.set_item("gss_stacks_after", entry.gss_stacks_after)?;
            set_gss_summary_fields(&d, "gss_before", gss_stacks_before_len, &entry.gss_summary_before)?;
            set_gss_summary_fields(&d, "gss", gss_stacks_after_len, &entry.gss_summary_after)?;
            d.set_item("match_start", entry.match_start)?;
            d.set_item("match_end", entry.match_end)?;
            d.set_item("token_bound", entry.token_bound)?;
            d.set_item("match_bytes", entry.match_bytes)?;

            // Profile fields
            let p = &entry.profile;
            d.set_item("pure_shift", p.pure_shift)?;
            d.set_item("deterministic_entered", p.deterministic_entered)?;
            d.set_item("deterministic_finished", p.deterministic_finished)?;
            d.set_item("nondeterministic_entered", p.nondeterministic_entered)?;
            d.set_item("vstack_len", p.vstack_len)?;
            d.set_item("n_reduces_above_floor", p.n_reduces_above_floor)?;
            d.set_item("n_floor_crossings", p.n_floor_crossings)?;
            d.set_item("n_nondet_waves", p.n_nondet_waves)?;
            d.set_item("n_nondet_branches", p.n_nondet_branches)?;
            d.set_item("top_states", p.top_states)?;
            d.set_item("gss_depth", p.gss_depth)?;
            d.set_item("total_ns", p.total_ns)?;
            d.set_item("clone_ns", p.clone_ns)?;
            d.set_item("fast_path_ns", p.fast_path_ns)?;
            d.set_item("stack_shift_apply_ns", p.stack_shift_apply_ns)?;
            d.set_item("det_ns", p.det_ns)?;
            d.set_item("det_floor_cross_ns", p.det_floor_cross_ns)?;
            d.set_item("nondet_ns", p.nondet_ns)?;
            d.set_item("nondet_det_ns", p.nondet_det_ns)?;
            d.set_item("nondet_det_floor_cross_ns", p.nondet_det_floor_cross_ns)?;
            d.set_item("det_exit_reason", p.det_exit_reason)?;
            d.set_item("det_exit_state", p.det_exit_state)?;
            d.set_item("n_det_action_lookups", p.n_det_action_lookups)?;
            d.set_item("n_det_goto_lookups", p.n_det_goto_lookups)?;
            d.set_item("n_det_popn_ops", p.n_det_popn_ops)?;
            d.set_item("n_nondet_reduce_ops", p.n_nondet_reduce_ops)?;
            d.set_item("n_nondet_merges", p.n_nondet_merges)?;
            d.set_item("n_nondet_isolates", p.n_nondet_isolates)?;
            if let Some(trace) = &p.trace {
                d.set_item("trace", advance_trace_to_dict(py, trace)?)?;
            }
            d.set_item("summary_ns", entry.summary_ns)?;
            advance_list.append(d)?;
        }
        result.set_item("advances", advance_list)?;
        result.set_item("final_stacks", final_stacks)?;
        let commit_dict = pyo3::types::PyDict::new(py);
        commit_dict.set_item("total_ns", commit_profile.total_ns)?;
        commit_dict.set_item("scan_ns", commit_profile.scan_ns)?;
        commit_dict.set_item("prune_ns", commit_profile.prune_ns)?;
        commit_dict.set_item("queue_ns", commit_profile.queue_ns)?;
        commit_dict.set_item("fuse_ns", commit_profile.fuse_ns)?;
        commit_dict.set_item("initial_exec_ns", commit_profile.initial_exec_ns)?;
        commit_dict.set_item("exec_ns", commit_profile.exec_ns)?;
        commit_dict.set_item("queue_exec_ns", commit_profile.queue_exec_ns)?;
        commit_dict.set_item("queue_match_ns", commit_profile.queue_match_ns)?;
        commit_dict.set_item("queue_enqueue_ns", commit_profile.queue_enqueue_ns)?;
        commit_dict.set_item("queue_bookkeeping_ns", commit_profile.queue_bookkeeping_ns)?;
        commit_dict.set_item("advance_ns", commit_profile.advance_ns)?;
        commit_dict.set_item("advance_may_check_ns", commit_profile.advance_may_check_ns)?;
        commit_dict.set_item("advance_core_ns", commit_profile.advance_core_ns)?;
        commit_dict.set_item("advance_future_disallow_ns", commit_profile.advance_future_disallow_ns)?;
        commit_dict.set_item("actionable_ns", commit_profile.actionable_ns)?;
        commit_dict.set_item("may_advance_ns", commit_profile.may_advance_ns)?;
        commit_dict.set_item("n_tokenizer_states", commit_profile.n_tokenizer_states)?;
        commit_dict.set_item("n_queue_entries", commit_profile.n_queue_entries)?;
        commit_dict.set_item("n_advances", commit_profile.n_advances)?;
        commit_dict.set_item("adv_n_reduces_above_floor", commit_profile.adv_n_reduces_above_floor)?;
        commit_dict.set_item("adv_n_floor_crossings", commit_profile.adv_n_floor_crossings)?;
        commit_dict.set_item("adv_n_nondet_waves", commit_profile.adv_n_nondet_waves)?;
        commit_dict.set_item("adv_n_nondet_branches", commit_profile.adv_n_nondet_branches)?;
        commit_dict.set_item("adv_clone_ns", commit_profile.adv_clone_ns)?;
        commit_dict.set_item("adv_fast_path_ns", commit_profile.adv_fast_path_ns)?;
        commit_dict.set_item("adv_stack_shift_apply_ns", commit_profile.adv_stack_shift_apply_ns)?;
        commit_dict.set_item("adv_det_ns", commit_profile.adv_det_ns)?;
        commit_dict.set_item(
            "adv_det_floor_cross_ns",
            commit_profile.adv_det_floor_cross_ns,
        )?;
        commit_dict.set_item("adv_nondet_ns", commit_profile.adv_nondet_ns)?;
        commit_dict.set_item("adv_vstack_len", commit_profile.adv_vstack_len)?;
        commit_dict.set_item("adv_gss_depth", commit_profile.adv_gss_depth)?;
        commit_dict.set_item("adv_det_exit_reason", commit_profile.adv_det_exit_reason)?;
        commit_dict.set_item("adv_det_exit_state", commit_profile.adv_det_exit_state)?;
        commit_dict.set_item("adv_n_det_action_lookups", commit_profile.adv_n_det_action_lookups)?;
        commit_dict.set_item("adv_n_det_goto_lookups", commit_profile.adv_n_det_goto_lookups)?;
        commit_dict.set_item("adv_n_det_popn_ops", commit_profile.adv_n_det_popn_ops)?;
        commit_dict.set_item("adv_n_nondet_reduce_ops", commit_profile.adv_n_nondet_reduce_ops)?;
        commit_dict.set_item("adv_n_nondet_merges", commit_profile.adv_n_nondet_merges)?;
        commit_dict.set_item("adv_n_nondet_isolates", commit_profile.adv_n_nondet_isolates)?;
        commit_dict.set_item("adv_nondet_det_ns", commit_profile.adv_nondet_det_ns)?;
        commit_dict.set_item(
            "adv_nondet_det_floor_cross_ns",
            commit_profile.adv_nondet_det_floor_cross_ns,
        )?;
        commit_dict.set_item("adv_summary_ns", commit_profile.adv_summary_ns)?;
        commit_dict.set_item("fast_path_total_ns", commit_profile.fast_path_total_ns)?;
        commit_dict.set_item("fast_path_tokenizer_exec_ns", commit_profile.fast_path_tokenizer_exec_ns)?;
        commit_dict.set_item("fast_path_match_scan_ns", commit_profile.fast_path_match_scan_ns)?;
        commit_dict.set_item("fast_path_end_state_check_ns", commit_profile.fast_path_end_state_check_ns)?;
        commit_dict.set_item("fast_path_prune_ns", commit_profile.fast_path_prune_ns)?;
        commit_dict.set_item("fast_path_advance_ns", commit_profile.fast_path_advance_ns)?;
        commit_dict.set_item("fast_path_future_disallow_ns", commit_profile.fast_path_future_disallow_ns)?;
        commit_dict.set_item("fast_path_fuse_ns", commit_profile.fast_path_fuse_ns)?;
        commit_dict.set_item("fast_path_state_update_ns", commit_profile.fast_path_state_update_ns)?;
        commit_dict.set_item("linear_fast_path_total_ns", commit_profile.linear_fast_path_total_ns)?;
        commit_dict.set_item("linear_fast_path_exec_ns", commit_profile.linear_fast_path_exec_ns)?;
        commit_dict.set_item("linear_fast_path_match_scan_ns", commit_profile.linear_fast_path_match_scan_ns)?;
        commit_dict.set_item("linear_fast_path_end_state_check_ns", commit_profile.linear_fast_path_end_state_check_ns)?;
        commit_dict.set_item("linear_fast_path_advance_ns", commit_profile.linear_fast_path_advance_ns)?;
        commit_dict.set_item("linear_fast_path_action_lookup_ns", commit_profile.linear_fast_path_action_lookup_ns)?;
        commit_dict.set_item("linear_fast_path_carried_gate_ns", commit_profile.linear_fast_path_carried_gate_ns)?;
        commit_dict.set_item("linear_fast_path_materialize_ns", commit_profile.linear_fast_path_materialize_ns)?;
        commit_dict.set_item("linear_fast_path_apply_action_wall_ns", commit_profile.linear_fast_path_apply_action_wall_ns)?;
        commit_dict.set_item("linear_fast_path_profile_bookkeeping_ns", commit_profile.linear_fast_path_profile_bookkeeping_ns)?;
        commit_dict.set_item("linear_fast_path_future_disallow_ns", commit_profile.linear_fast_path_future_disallow_ns)?;
        commit_dict.set_item("linear_fast_path_fuse_ns", commit_profile.linear_fast_path_fuse_ns)?;
        commit_dict.set_item("linear_fast_path_steps", commit_profile.linear_fast_path_steps)?;
        result.set_item("commit_profile", commit_dict)?;

        Ok(result)
    }

}


#[pyfunction]
fn compose_compiled_subgrammars(
    py: Python<'_>,
    parent: PyRef<'_, PyConstraint>,
    subgrammars: BTreeMap<String, Py<PyConstraint>>,
    vocab: &PyVocab,
) -> PyResult<PyConstraint> {
    let owned_children = subgrammars
        .into_iter()
        .map(|(name, child)| {
            let child = child.borrow(py);
            (name, Arc::clone(&child.inner))
        })
        .collect::<Vec<_>>();
    let shared_children = owned_children
        .iter()
        .map(|(name, child)| (name.as_str(), Arc::clone(child)))
        .collect::<Vec<_>>();
    let mut parent = parent.inner.as_ref().clone();
    parent
        .bind_vocab_exact(&vocab.inner)
        .map_err(PyValueError::new_err)?;
    PyConstraint::from_constraint_result(
        parent.compose_compiled_subgrammars_shared(&shared_children, &vocab.inner),
        vocab,
    )
}

fn auto_ref_passthrough_key(key: &str) -> bool {
    matches!(
        key,
        "$anchor"
            | "$comment"
            | "$defs"
            | "$dynamicAnchor"
            | "$id"
            | "$ref"
            | "$schema"
            | "default"
            | "definitions"
            | "deprecated"
            | "description"
            | "examples"
            | "id"
            | "readOnly"
            | "title"
            | "writeOnly"
    )
}

fn auto_schema_type_contains(value: Option<&serde_json::Value>, expected: &str) -> bool {
    match value {
        Some(serde_json::Value::String(value)) => value == expected,
        Some(serde_json::Value::Array(values)) => values
            .iter()
            .any(|value| value.as_str().is_some_and(|value| value == expected)),
        _ => false,
    }
}

fn auto_nonnegative_integer(value: Option<&serde_json::Value>) -> Option<u64> {
    let value = value?;
    if let Some(value) = value.as_u64() {
        return Some(value.min(1_000_000_000));
    }
    if let Some(value) = value.as_i64() {
        return (value >= 0).then_some((value as u64).min(1_000_000_000));
    }
    let value = value.as_f64()?;
    (value.is_finite() && value >= 0.0)
        .then_some((value.trunc() as u64).min(1_000_000_000))
}

/// Return the exact source-shape counts used by Python AutoConstraint policy.
///
/// The walk mirrors CFA's two semantics-preserving JSON-Schema normalizations
/// without allocating normalized copies: ignored legacy `$ref` siblings are
/// omitted, and an empty `properties` object next to non-empty
/// `patternProperties` is skipped.  Keeping this native avoids making the auto
/// selector's build tail proportional to a Python object-tree traversal.
#[pyfunction]
fn auto_json_schema_shape_counts(schema: &str) -> PyResult<Vec<u64>> {
    let root: serde_json::Value =
        serde_json::from_str(schema).map_err(|err| PyValueError::new_err(err.to_string()))?;
    let legacy_ref_siblings = root
        .as_object()
        .and_then(|root| root.get("$schema"))
        .and_then(serde_json::Value::as_str)
        .is_some_and(|schema_uri| {
            let schema_uri = schema_uri.to_ascii_lowercase();
            ["draft-03", "draft-04", "draft-06", "draft-07"]
                .iter()
                .any(|marker| schema_uri.contains(marker))
        });

    // nodes, dicts, lists, leaves, properties, property_name_chars,
    // string_types, number_types, array_types, pattern_chars, max_length
    let mut counts = [0u64; 11];
    let mut stack = vec![&root];
    while let Some(value) = stack.pop() {
        counts[0] += 1;
        match value {
            serde_json::Value::Object(map) => {
                counts[1] += 1;
                let legacy_ref = legacy_ref_siblings
                    && map
                        .get("$ref")
                        .is_some_and(|value| matches!(value, serde_json::Value::String(_)));
                let visible = |key: &str| !legacy_ref || auto_ref_passthrough_key(key);

                let properties = visible("properties")
                    .then(|| map.get("properties"))
                    .flatten()
                    .and_then(serde_json::Value::as_object);
                if let Some(properties) = properties {
                    counts[4] += properties.len() as u64;
                    counts[5] += properties
                        .keys()
                        .map(|name| name.chars().count() as u64)
                        .sum::<u64>();
                }

                let raw_type = visible("type").then(|| map.get("type")).flatten();
                counts[6] += u64::from(auto_schema_type_contains(raw_type, "string"));
                counts[7] += u64::from(auto_schema_type_contains(raw_type, "number"));
                counts[8] += u64::from(auto_schema_type_contains(raw_type, "array"));

                if visible("pattern") {
                    if let Some(pattern) = map.get("pattern").and_then(serde_json::Value::as_str) {
                        counts[9] += pattern.chars().count() as u64;
                    }
                }
                if visible("maxLength") {
                    if let Some(max_length) = auto_nonnegative_integer(map.get("maxLength")) {
                        counts[10] = counts[10].max(max_length);
                    }
                }

                let pattern_properties_nonempty = visible("patternProperties")
                    && map
                        .get("patternProperties")
                        .and_then(serde_json::Value::as_object)
                        .is_some_and(|map| !map.is_empty());
                let skip_empty_properties = properties.is_some_and(|map| map.is_empty())
                    && pattern_properties_nonempty;
                for (key, child) in map.iter().rev() {
                    if !visible(key) || (key == "properties" && skip_empty_properties) {
                        continue;
                    }
                    stack.push(child);
                }
            }
            serde_json::Value::Array(values) => {
                counts[2] += 1;
                stack.extend(values.iter().rev());
            }
            _ => counts[3] += 1,
        }
    }
    Ok(counts.to_vec())
}

#[pyfunction]
#[pyo3(signature = (ebnf_source, vocab, end_token_ids=None, vocab_partition=false))]
fn compile_ebnf_serialized_profiled(
    ebnf_source: &str,
    vocab: &PyVocab,
    end_token_ids: Option<Vec<u32>>,
    vocab_partition: bool,
) -> PyResult<(Vec<u8>, u64, u64)> {
    constraint_result(glrmask::DynamicConstraint::compile_ebnf_serialized_profiled_with_end_tokens(
        ebnf_source,
        &vocab.inner,
        end_token_ids.as_deref().unwrap_or(&[]),
        vocab_partition,
    ))
}

#[pyfunction]
#[pyo3(signature = (lark_source, vocab, end_token_ids=None, vocab_partition=false))]
fn compile_lark_serialized_profiled(
    lark_source: &str,
    vocab: &PyVocab,
    end_token_ids: Option<Vec<u32>>,
    vocab_partition: bool,
) -> PyResult<(Vec<u8>, u64, u64)> {
    constraint_result(glrmask::DynamicConstraint::compile_lark_serialized_profiled_with_end_tokens(
        lark_source,
        &vocab.inner,
        end_token_ids.as_deref().unwrap_or(&[]),
        vocab_partition,
    ))
}

#[pyfunction]
#[pyo3(signature = (schema, vocab, end_token_ids=None, vocab_partition=false))]
fn compile_json_schema_serialized_profiled(
    schema: &str,
    vocab: &PyVocab,
    end_token_ids: Option<Vec<u32>>,
    vocab_partition: bool,
) -> PyResult<(Vec<u8>, u64, u64)> {
    constraint_result(glrmask::DynamicConstraint::compile_json_schema_serialized_profiled_with_end_tokens(
        schema,
        &vocab.inner,
        end_token_ids.as_deref().unwrap_or(&[]),
        vocab_partition,
    ))
}

#[pyfunction]
#[pyo3(signature = (glrm_source, vocab, end_token_ids=None, vocab_partition=false))]
fn compile_glrm_serialized_profiled(
    glrm_source: &str,
    vocab: &PyVocab,
    end_token_ids: Option<Vec<u32>>,
    vocab_partition: bool,
) -> PyResult<(Vec<u8>, u64, u64)> {
    constraint_result(glrmask::DynamicConstraint::compile_glrm_serialized_profiled_with_end_tokens(
        glrm_source,
        &vocab.inner,
        end_token_ids.as_deref().unwrap_or(&[]),
        vocab_partition,
    ))
}

#[pyfunction]
#[pyo3(signature = (ebnf_source, vocab, end_token_ids=None))]
fn compile_ebnf_serialized(
    ebnf_source: &str,
    vocab: &PyVocab,
    end_token_ids: Option<Vec<u32>>,
) -> PyResult<Vec<u8>> {
    constraint_result(glrmask::DynamicConstraint::compile_ebnf_serialized_with_end_tokens(
        ebnf_source,
        &vocab.inner,
        end_token_ids.as_deref().unwrap_or(&[]),
    ))
}

#[pyfunction]
#[pyo3(signature = (lark_source, vocab, end_token_ids=None))]
fn compile_lark_serialized(
    lark_source: &str,
    vocab: &PyVocab,
    end_token_ids: Option<Vec<u32>>,
) -> PyResult<Vec<u8>> {
    constraint_result(glrmask::DynamicConstraint::compile_lark_serialized_with_end_tokens(
        lark_source,
        &vocab.inner,
        end_token_ids.as_deref().unwrap_or(&[]),
    ))
}

#[pyfunction]
#[pyo3(signature = (schema, vocab, end_token_ids=None))]
fn compile_json_schema_serialized(
    schema: &str,
    vocab: &PyVocab,
    end_token_ids: Option<Vec<u32>>,
) -> PyResult<Vec<u8>> {
    constraint_result(glrmask::DynamicConstraint::compile_json_schema_serialized_with_end_tokens(
        schema,
        &vocab.inner,
        end_token_ids.as_deref().unwrap_or(&[]),
    ))
}

#[pyfunction]
#[pyo3(signature = (glrm_source, vocab, end_token_ids=None))]
fn compile_glrm_serialized(
    glrm_source: &str,
    vocab: &PyVocab,
    end_token_ids: Option<Vec<u32>>,
) -> PyResult<Vec<u8>> {
    constraint_result(glrmask::DynamicConstraint::compile_glrm_serialized_with_end_tokens(
        glrm_source,
        &vocab.inner,
        end_token_ids.as_deref().unwrap_or(&[]),
    ))
}

// ---------------------------------------------------------------------------
// UnlinkedConstraint
// ---------------------------------------------------------------------------


#[pyfunction]
fn num_parser_states(constraint: PyRef<'_, PyConstraint>) -> u32 {
    constraint.num_parser_states()
}

#[pyfunction]
fn terminal_display_names(constraint: PyRef<'_, PyConstraint>) -> Vec<String> {
    constraint.terminal_display_names()
}

#[pyfunction]
fn terminal_display_name(
    constraint: PyRef<'_, PyConstraint>,
    terminal_id: u32,
) -> Option<String> {
    constraint.terminal_display_name(terminal_id)
}

#[pyfunction]
fn fill_mask_timed_ns(
    state: PyRef<'_, PyConstraintState>,
    bitmask: PyReadwriteArray1<i32>,
) -> PyResult<u64> {
    state.fill_mask_timed_ns(bitmask)
}

#[cfg(feature = "allocation-tracking")]
#[pyfunction]
fn fill_mask_timed_allocation_stats(
    state: PyRef<'_, PyConstraintState>,
    bitmask: PyReadwriteArray1<i32>,
) -> PyResult<Vec<u64>> {
    state.fill_mask_timed_allocation_stats(bitmask)
}

#[pyfunction]
fn fill_mask_profiled<'py>(
    py: Python<'py>,
    state: PyRef<'py, PyConstraintState>,
    bitmask: PyReadwriteArray1<i32>,
) -> PyResult<Bound<'py, PyDict>> {
    state.fill_mask_profiled(py, bitmask)
}

#[pyfunction]
fn commit_token_timed_ns(
    mut state: PyRefMut<'_, PyConstraintState>,
    token_id: u32,
) -> PyResult<u64> {
    state.commit_token_timed_ns(token_id)
}

#[cfg(feature = "allocation-tracking")]
#[pyfunction]
fn commit_token_timed_allocation_stats(
    mut state: PyRefMut<'_, PyConstraintState>,
    token_id: u32,
) -> PyResult<Vec<u64>> {
    state.commit_token_timed_allocation_stats(token_id)
}

#[pyfunction]
fn commit_token_profiled<'py>(
    py: Python<'py>,
    mut state: PyRefMut<'py, PyConstraintState>,
    token_id: u32,
) -> PyResult<Bound<'py, PyDict>> {
    state.commit_token_profiled(py, token_id)
}

#[pyfunction]
fn parser_root_count(state: PyRef<'_, PyConstraintState>) -> usize {
    state.parser_root_count()
}

#[pyfunction]
fn parser_path_count(state: PyRef<'_, PyConstraintState>, limit: usize) -> usize {
    state.parser_path_count(limit)
}

#[pyfunction]
fn debug_parser_stacks(
    state: PyRef<'_, PyConstraintState>,
) -> Vec<(u32, Vec<(Vec<u32>, Vec<(u32, Vec<u32>)>)>)> {
    state.debug_parser_stacks()
}

#[pyfunction]
fn commit_token_per_advance<'py>(
    py: Python<'py>,
    mut state: PyRefMut<'py, PyConstraintState>,
    token_id: u32,
) -> PyResult<Bound<'py, PyDict>> {
    state.commit_token_per_advance(py, token_id)
}

fn add_internal_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
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
    internal.add_function(wrap_pyfunction!(prepare_vocab_for_dynamic_compile, &internal)?)?;
    internal.add_function(wrap_pyfunction!(compile_grammar_def_json, &internal)?)?;
    internal.add_function(wrap_pyfunction!(dump_json_schema_grammar_glrm, &internal)?)?;
    internal.add_function(wrap_pyfunction!(compose_compiled_subgrammars, &internal)?)?;
    internal.add_function(wrap_pyfunction!(auto_json_schema_shape_counts, &internal)?)?;
    internal.add_function(wrap_pyfunction!(compile_ebnf_serialized_profiled, &internal)?)?;
    internal.add_function(wrap_pyfunction!(compile_lark_serialized_profiled, &internal)?)?;
    internal.add_function(wrap_pyfunction!(compile_json_schema_serialized_profiled, &internal)?)?;
    internal.add_function(wrap_pyfunction!(compile_glrm_serialized_profiled, &internal)?)?;
    internal.add_function(wrap_pyfunction!(compile_ebnf_serialized, &internal)?)?;
    internal.add_function(wrap_pyfunction!(compile_lark_serialized, &internal)?)?;
    internal.add_function(wrap_pyfunction!(compile_json_schema_serialized, &internal)?)?;
    internal.add_function(wrap_pyfunction!(compile_glrm_serialized, &internal)?)?;
    internal.add_function(wrap_pyfunction!(num_parser_states, &internal)?)?;
    internal.add_function(wrap_pyfunction!(terminal_display_names, &internal)?)?;
    internal.add_function(wrap_pyfunction!(terminal_display_name, &internal)?)?;
    internal.add_function(wrap_pyfunction!(fill_mask_timed_ns, &internal)?)?;
    #[cfg(feature = "allocation-tracking")]
    internal.add_function(wrap_pyfunction!(fill_mask_timed_allocation_stats, &internal)?)?;
    internal.add_function(wrap_pyfunction!(fill_mask_profiled, &internal)?)?;
    internal.add_function(wrap_pyfunction!(commit_token_timed_ns, &internal)?)?;
    #[cfg(feature = "allocation-tracking")]
    internal.add_function(wrap_pyfunction!(commit_token_timed_allocation_stats, &internal)?)?;
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

#[pymodule]
fn _glrmask(m: &Bound<'_, PyModule>) -> PyResult<()> {
    configure_mimalloc_runtime_default();
    // rust-numpy lazily publishes and caches its mutable-borrow checking API on
    // the first PyReadwriteArray extraction. Paying that process-global setup
    // inside the first fill_mask call creates an artificial runtime-latency
    // spike. Initialize it while the extension module itself is loading; this
    // touches only a zero-length NumPy array and does not build or execute a
    // constraint.
    drop(PyArray1::<i32>::zeros(m.py(), 0, false).readwrite());
    glrmask::Constraint::warm_ti_pool();
    api::register(m)?;
    m.add_class::<PyVocab>()?;
    m.add_class::<PyConstraint>()?;
    m.add_class::<PyConstraintState>()?;
    m.add_class::<PyDynamicConstraint>()?;
    m.add_class::<PyDynamicConstraintState>()?;
    add_internal_module(m)?;
    m.setattr(
        "__all__",
        [
            "Grammar",
            "UnlinkedConstraint",
            "ExactToken",
            "ExactTokens",
            "Optimization",
            "Vocab",
            "Constraint",
            "ConstraintState",
            "_internal",
        ],
    )?;
    Ok(())
}

#[pyfunction]
fn mimalloc_purge_delay() -> i64 {
    unsafe { libmimalloc_sys::mi_option_get(MIMALLOC_PURGE_DELAY_OPTION) as i64 }
}

#[pyfunction]
fn mimalloc_purge_decommits() -> bool {
    unsafe { libmimalloc_sys::mi_option_is_enabled(MIMALLOC_PURGE_DECOMMITS_OPTION) }
}

#[pyfunction]
#[pyo3(signature = (force=true))]
fn collect_allocator(force: bool) {
    unsafe {
        let purge_delay = libmimalloc_sys::mi_option_get(MIMALLOC_PURGE_DELAY_OPTION);
        if purge_delay < 0 {
            // An explicit no-purge policy should not make explicit collection
            // a no-op. Temporarily permit this caller-selected collection and
            // restore the configured policy before returning.
            libmimalloc_sys::mi_option_set(MIMALLOC_PURGE_DELAY_OPTION, 0);
        }
        libmimalloc_sys::mi_collect(force);
        if purge_delay < 0 {
            libmimalloc_sys::mi_option_set(MIMALLOC_PURGE_DELAY_OPTION, purge_delay);
        }
    }
}


#[pyfunction]
fn clear_stale_weights() {
    glrmask::Constraint::clear_stale_weights();
}

#[pyfunction]
fn clear_weight_op_caches() {
    glrmask::Constraint::clear_weight_op_caches();
}

#[pyfunction]
fn clear_weight_caches() {
    glrmask::Constraint::clear_weight_op_caches();
    glrmask::Constraint::clear_stale_weights();
}

#[pyfunction]
fn compiler_cache_stats(vocab: Option<&PyVocab>) -> std::collections::BTreeMap<&'static str, u64> {
    let stats = glrmask::__private::compiler_cache_stats(vocab.map(|vocab| &vocab.inner));
    std::collections::BTreeMap::from([
        ("token_set_entries", stats.token_set_entries as u64),
        ("live_token_set_entries", stats.live_token_set_entries as u64),
        ("weight_buckets", stats.weight_buckets as u64),
        ("weight_entries", stats.weight_entries as u64),
        ("live_weight_entries", stats.live_weight_entries as u64),
        ("current_thread_weight_ops", stats.current_thread_weight_ops as u64),
        ("current_thread_token_set_ops", stats.current_thread_token_set_ops as u64),
        ("current_thread_public_intersections", stats.current_thread_public_intersections as u64),
        ("current_thread_weight_hashes", stats.current_thread_weight_hashes as u64),
        ("weight_op_generation", stats.weight_op_generation),
        ("weight_hash_generation", stats.weight_hash_generation),
        ("vocab_artifacts", stats.vocab_artifacts as u64),
    ])
}

#[pyfunction]
fn prepare_vocab_for_compile(vocab: &PyVocab) {
    vocab.inner.prepare_for_compile();
}

#[pyfunction]
fn prepare_vocab_for_dynamic_compile(vocab: &PyVocab) {
    vocab.inner.prepare_for_dynamic_compile();
}

#[pyfunction]
fn compile_grammar_def_json(grammar_def_json: &str, vocab: &PyVocab) -> PyResult<PyConstraint> {
    let constraint = glrmask::Constraint::compile_grammar_def_json(grammar_def_json, &vocab.inner)
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;
    let max_token = constraint.max_original_token_id().unwrap_or(0);
    Ok(PyConstraint {
        inner: std::sync::Arc::new(constraint),
        max_token,
    })
}

#[pyfunction]
fn dump_json_schema_grammar_glrm(schema_json: &str) -> PyResult<String> {
    glrmask::Constraint::dump_json_schema_grammar_glrm(schema_json)
        .map_err(|e| PyValueError::new_err(format!("{e}")))
}
