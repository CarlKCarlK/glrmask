//! Vocabulary constructors and the llama.cpp byte/token adapter.

use crate::api;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

pub(super) fn dict_to_vocab(token_to_id: &Bound<'_, PyDict>) -> PyResult<glrmask::Vocab> {
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

pub(super) fn id_to_bytes_dict_to_vocab(
    id_to_bytes: &Bound<'_, PyDict>,
) -> PyResult<glrmask::Vocab> {
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

pub(super) fn llama_cpp_to_vocab(llm: &Bound<'_, PyAny>) -> PyResult<(glrmask::Vocab, Vec<u32>)> {
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

// ---------------------------------------------------------------------------
// PyVocab
// ---------------------------------------------------------------------------

#[pyclass(name = "Vocab")]
#[derive(Clone)]
pub struct PyVocab {
    pub(super) inner: glrmask::Vocab,
    pub(super) llama_cpp_end_token_ids: Vec<u32>,
}

#[pymethods]
impl PyVocab {
    /// Resolve one exact token, preserving this complete vocabulary identity.
    fn token(&self, id: u32) -> PyResult<api::PyExactToken> {
        self.inner
            .token(id)
            .map(|inner| api::PyExactToken { inner })
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// Resolve a nonempty set of distinct exact token IDs.
    fn tokens(&self, ids: Vec<u32>) -> PyResult<api::PyExactTokens> {
        self.inner
            .tokens(ids)
            .map(|inner| api::PyExactTokens { inner })
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

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
