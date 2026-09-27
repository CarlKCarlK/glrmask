//! Owned native constraints and per-sequence states.
//!
//! Each self_cell keeps its Arc owner alive for the complete dependent lifetime.
//! Public and diagnostic methods use the same native state and packed coordinates.

use crate::conversion::WritableMask;

use crate::conversion::{
    bitmask_u32_view, constraint_result, external_terminal_bindings_from_dict, resolved_mask_size,
    string_result, words_to_bool_array,
};
use crate::profiling::{
    advance_trace_to_dict, commit_profile_to_dict, mask_profile_to_dict, set_gss_summary_fields,
};
use crate::PyVocab;
#[cfg(feature = "allocation-tracking")]
use crate::{allocation_tracking, profiling::allocation_stats_tuple};
use glrmask::__private::{ConstraintExt as _, ConstraintStateExt as _, DynamicConstraintExt as _};
use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use self_cell::self_cell;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

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
    pub(super) fn from_arc(arc: Arc<glrmask::Constraint>) -> Self {
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
    pub(super) fn from_arc(arc: Arc<glrmask::DynamicConstraint>) -> Self {
        OwnedDynamicState::new(arc, |arc_ref| arc_ref.start())
    }
}

// ---------------------------------------------------------------------------
// PyConstraint
// ---------------------------------------------------------------------------

/// Compiled grammar constraint. Immutable, thread-safe.
#[pyclass(name = "Constraint")]
#[derive(Clone)]
pub struct PyConstraint {
    pub(super) inner: Arc<glrmask::Constraint>,
    pub(super) max_token: u32,
}

impl PyConstraint {
    pub(super) fn from_constraint_result<E: std::fmt::Display>(
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
    pub(super) fn num_parser_states(&self) -> u32 {
        self.inner.num_parser_states()
    }

    /// Return display names for grammar terminals by terminal id.
    pub(super) fn terminal_display_names(&self) -> Vec<String> {
        self.inner.terminal_display_names().to_vec()
    }

    /// Return the display name for a grammar terminal id, if present.
    pub(super) fn terminal_display_name(&self, terminal_id: u32) -> Option<String> {
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
        Ok(Self {
            inner: Arc::new(inner),
            max_token,
        })
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
    pub(super) inner: Arc<glrmask::DynamicConstraint>,
    pub(super) max_token: u32,
}

impl PyDynamicConstraint {
    pub(super) fn from_constraint_result<E: std::fmt::Display>(
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
            if subgrammars
                .as_ref()
                .is_some_and(|children| !children.is_empty())
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
        let mut builder =
            glrmask::ConstraintSpec::builder(glrmask::Grammar::glrm(glrm_source), &vocab.inner)
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
        Self::from_constraint_result(spec.compile_dynamic(), vocab)
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
    pub(super) max_token: u32,
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
        commit_profile_to_dict(py, profile)
    }

    fn fill_mask(&self, mut bitmask: WritableMask<'_>) -> PyResult<()> {
        let buf = bitmask_u32_view(&mut bitmask, self.inner.borrow_owner().mask_len())?;
        self.inner
            .with_dependent(|_owner, state| state.fill_mask(buf));
        Ok(())
    }

    #[doc(hidden)]
    fn fill_mask_timed_ns(&self, mut bitmask: WritableMask<'_>) -> PyResult<u64> {
        let buf = bitmask_u32_view(&mut bitmask, self.inner.borrow_owner().mask_len())?;
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
    pub(super) max_token: u32,
}

#[pymethods]
impl PyConstraintState {
    /// Whether an allowed final end token has completed this sequence.
    fn is_terminated(&self) -> bool {
        self.inner
            .with_dependent(|_owner, state| state.is_terminated())
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

    fn fill_mask(&self, mut bitmask: WritableMask<'_>) -> PyResult<()> {
        let buf = bitmask_u32_view(&mut bitmask, self.inner.borrow_owner().mask_len())?;
        self.inner
            .with_dependent(|_owner, state| state.fill_mask(buf));
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
        self.inner
            .with_dependent(|_owner, state| state.is_rejected())
    }

    fn forced(&self) -> Vec<u32> {
        self.inner.with_dependent(|_owner, state| state.forced())
    }

    fn is_accepting(&self) -> bool {
        self.inner
            .with_dependent(|_owner, state| state.is_accepting())
    }

    #[cfg(feature = "allocation-tracking")]
    #[pyo3(name = "fill_mask_timed_allocation_stats")]
    fn py_fill_mask_timed_allocation_stats(&self, bitmask: WritableMask<'_>) -> PyResult<Vec<u64>> {
        self.fill_mask_timed_allocation_stats(bitmask)
    }

    #[cfg(feature = "allocation-tracking")]
    #[pyo3(name = "commit_token_timed_allocation_stats")]
    fn py_commit_token_timed_allocation_stats(&mut self, token_id: u32) -> PyResult<Vec<u64>> {
        self.commit_token_timed_allocation_stats(token_id)
    }
}

impl PyConstraintState {
    pub(super) fn fill_mask_timed_ns(&self, mut bitmask: WritableMask<'_>) -> PyResult<u64> {
        let buf = bitmask_u32_view(&mut bitmask, self.inner.borrow_owner().mask_len())?;
        Ok(self
            .inner
            .with_dependent(|_owner, state| state.fill_mask_timed_ns(buf)))
    }

    #[cfg(feature = "allocation-tracking")]
    pub(super) fn fill_mask_timed_allocation_stats(
        &self,
        mut bitmask: WritableMask<'_>,
    ) -> PyResult<Vec<u64>> {
        let buf = bitmask_u32_view(&mut bitmask, self.inner.borrow_owner().mask_len())?;
        let (elapsed_ns, stats) = allocation_tracking::measure(|| {
            self.inner
                .with_dependent(|_owner, state| state.fill_mask_timed_ns(buf))
        });
        Ok(allocation_stats_tuple(elapsed_ns, stats))
    }

    pub(super) fn fill_mask_profiled<'py>(
        &self,
        py: Python<'py>,
        mut bitmask: WritableMask<'_>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let buf = bitmask_u32_view(&mut bitmask, self.inner.borrow_owner().mask_len())?;
        let profile = self
            .inner
            .with_dependent(|_owner, state| state.fill_mask_profiled(buf));
        mask_profile_to_dict(py, profile)
    }

    pub(super) fn commit_token_timed_ns(&mut self, token_id: u32) -> PyResult<u64> {
        self.inner.with_dependent_mut(|_owner, state| {
            state
                .commit_token_timed_ns(token_id)
                .map_err(PyValueError::new_err)
        })
    }

    #[cfg(feature = "allocation-tracking")]
    pub(super) fn commit_token_timed_allocation_stats(
        &mut self,
        token_id: u32,
    ) -> PyResult<Vec<u64>> {
        let (result, stats) = allocation_tracking::measure(|| {
            self.inner
                .with_dependent_mut(|_owner, state| state.commit_token_timed_ns(token_id))
        });
        let elapsed_ns = result.map_err(PyValueError::new_err)?;
        Ok(allocation_stats_tuple(elapsed_ns, stats))
    }

    /// Like commit_token but returns profiling stats as a dict.
    pub(super) fn commit_token_profiled<'py>(
        &mut self,
        py: Python<'py>,
        token_id: u32,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let profile = self.inner.with_dependent_mut(|_owner, state| {
            state
                .commit_token_profiled(token_id)
                .map_err(|e| PyValueError::new_err(e))
        })?;
        commit_profile_to_dict(py, profile)
    }

    /// Return total parser GSS root count across all tokenizer states.
    pub(super) fn parser_root_count(&self) -> usize {
        self.inner
            .with_dependent(|_owner, state| state.parser_root_count())
    }

    /// Return parser path count (capped at limit).
    pub(super) fn parser_path_count(&self, limit: usize) -> usize {
        self.inner
            .with_dependent(|_owner, state| state.parser_path_count(limit))
    }

    /// Return all flattened parser stacks for debugging.
    pub(super) fn debug_parser_stacks(&self) -> Vec<(u32, Vec<(Vec<u32>, Vec<(u32, Vec<u32>)>)>)> {
        self.inner
            .with_dependent(|_owner, state| state.debug_parser_stacks())
    }

    /// Per-advance profiling: returns a list of per-advance entries and final GSS stacks.
    pub(super) fn commit_token_per_advance<'py>(
        &mut self,
        py: Python<'py>,
        token_id: u32,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let (advances, final_stacks, commit_profile) =
            self.inner.with_dependent_mut(|_owner, state| {
                state
                    .commit_token_per_advance(token_id)
                    .map_err(|e| PyValueError::new_err(e))
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
            set_gss_summary_fields(
                &d,
                "gss_before",
                gss_stacks_before_len,
                &entry.gss_summary_before,
            )?;
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
        let commit_dict = commit_profile_to_dict(py, commit_profile)?;
        result.set_item("commit_profile", commit_dict)?;

        Ok(result)
    }
}
