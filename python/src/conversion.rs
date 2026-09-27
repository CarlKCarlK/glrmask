//! Checked Python conversions shared by public and private binding entry points.

use numpy::{PyArray1, PyArrayMethods, PyReadwriteArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

pub(super) fn external_terminal_bindings_from_dict(
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

pub(super) fn constraint_result<T, E: std::fmt::Display>(result: Result<T, E>) -> PyResult<T> {
    result.map_err(|e| PyValueError::new_err(format!("{e}")))
}

pub(super) fn words_to_bool_array<'py>(
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

pub(super) fn resolved_mask_size(max_token: u32, requested: Option<usize>) -> PyResult<usize> {
    let minimum = max_token as usize + 1;
    let size = requested.unwrap_or(minimum);
    if size < minimum {
        return Err(PyValueError::new_err(format!(
            "mask size {size} is smaller than the constraint token range {minimum}"
        )));
    }
    Ok(size)
}

pub(super) fn bitmask_u32_view<'a, 'py>(
    bitmask: &'a mut WritableMask<'py>,
    required: usize,
) -> PyResult<&'a mut [u32]> {
    let slice = bitmask
        .0
        .as_slice_mut()
        .map_err(|e| PyValueError::new_err(format!("Array must be contiguous: {e:?}")))?;
    if slice.len() < required {
        return Err(PyValueError::new_err(format!(
            "mask needs at least {required} packed words"
        )));
    }
    // Safety: i32 and u32 have identical size, alignment, and bit representation.
    Ok(unsafe { std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut u32, slice.len()) })
}

pub(super) fn string_result<T, E: std::fmt::Display>(result: Result<T, E>) -> PyResult<T> {
    result.map_err(|error| PyValueError::new_err(error.to_string()))
}

/// A fallible NumPy write guard retained for the complete native call.
///
/// rust-numpy's default argument extractor calls `readwrite()`, which panics on
/// a read-only or already-borrowed array. Borrow failure is a Python input error.
pub(super) struct WritableMask<'py>(PyReadwriteArray1<'py, i32>);

impl<'py> FromPyObject<'py> for WritableMask<'py> {
    fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        let array = obj.downcast::<PyArray1<i32>>()?;
        // Contiguity does not guarantee the alignment required by a Rust slice.
        if (array.data() as usize) % std::mem::align_of::<i32>() != 0 {
            return Err(PyValueError::new_err("mask array must be aligned for int32"));
        }
        array.try_readwrite().map(Self).map_err(|error| {
            PyValueError::new_err(format!(
                "mask array must be writable and exclusively borrowed: {error}"
            ))
        })
    }
}
