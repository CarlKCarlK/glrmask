#![recursion_limit = "512"]
//! Python module registration.
//!
//! api owns source/binding lifecycle; runtime owns compiled handles and sequences;
//! vocab owns token adapters. conversion is the single NumPy bitmask boundary.
//! Profiling, compiler helpers, and vocabulary partitions remain private tooling.

mod allocator;
mod api;
mod compiler;
mod conversion;
mod internal;
mod partition;
mod profiling;
mod runtime;
mod vocab;

#[cfg(feature = "allocation-tracking")]
mod allocation_tracking;

use allocator::configure_mimalloc_runtime_default;
use conversion::bitmask_u32_view;
use glrmask::__private::ConstraintExt as _;
use internal::add_internal_module;
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use runtime::{PyConstraint, PyConstraintState, PyDynamicConstraint, PyDynamicConstraintState};
use vocab::PyVocab;

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
