//! Experimental vocabulary equivalence, deliberately outside the public facade.
//!
//! The immutable analysis object maps grammar-equivalent model token IDs into
//! a private dense class coordinate and expands class masks back to model IDs.

use numpy::PyReadwriteArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use super::{PyVocab, bitmask_u32_view};

fn parse_vocab_partition_strategy(value: Option<&str>) -> PyResult<glrmask::VocabPartitionStrategy> {
    match value.unwrap_or("automatic").trim().to_ascii_lowercase().as_str() {
        "automatic" | "auto" => Ok(glrmask::VocabPartitionStrategy::Automatic),
        "compact" => Ok(glrmask::VocabPartitionStrategy::Compact),
        "dedicated" => Ok(glrmask::VocabPartitionStrategy::Dedicated),
        value => Err(PyValueError::new_err(format!(
            "unknown vocab partition strategy {value:?}; expected 'automatic', 'compact', or 'dedicated'"
        ))),
    }
}

#[pyclass(frozen, name = "VocabPartition", module = "glrmask._internal")]
#[derive(Clone)]
pub(super) struct PyVocabPartition {
    inner: glrmask::VocabPartition,
}

impl PyVocabPartition {
    fn from_result(result: glrmask::Result<glrmask::VocabPartition>) -> PyResult<Self> {
        result
            .map(|inner| Self { inner })
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }
}

#[pymethods]
impl PyVocabPartition {
    #[staticmethod]
    #[pyo3(signature = (schema, vocab, strategy=None))]
    fn from_json_schema(schema: &str, vocab: &PyVocab, strategy: Option<&str>) -> PyResult<Self> {
        let strategy = parse_vocab_partition_strategy(strategy)?;
        Self::from_result(glrmask::VocabPartition::compile_with_strategy(
            glrmask::Grammar::json_schema(schema),
            &vocab.inner,
            strategy,
        ))
    }

    #[staticmethod]
    #[pyo3(signature = (source, vocab, strategy=None))]
    fn from_ebnf(source: &str, vocab: &PyVocab, strategy: Option<&str>) -> PyResult<Self> {
        let strategy = parse_vocab_partition_strategy(strategy)?;
        Self::from_result(glrmask::VocabPartition::compile_with_strategy(
            glrmask::Grammar::ebnf(source),
            &vocab.inner,
            strategy,
        ))
    }

    #[staticmethod]
    #[pyo3(signature = (source, vocab, strategy=None))]
    fn from_lark(source: &str, vocab: &PyVocab, strategy: Option<&str>) -> PyResult<Self> {
        let strategy = parse_vocab_partition_strategy(strategy)?;
        Self::from_result(glrmask::VocabPartition::compile_with_strategy(
            glrmask::Grammar::lark(source),
            &vocab.inner,
            strategy,
        ))
    }

    #[staticmethod]
    #[pyo3(signature = (source, vocab, strategy=None))]
    fn from_glrm(source: &str, vocab: &PyVocab, strategy: Option<&str>) -> PyResult<Self> {
        let strategy = parse_vocab_partition_strategy(strategy)?;
        Self::from_result(glrmask::VocabPartition::compile_with_strategy(
            glrmask::Grammar::glrm(source),
            &vocab.inner,
            strategy,
        ))
    }

    #[getter]
    fn num_classes(&self) -> usize { self.inner.num_classes() }

    #[getter]
    fn internal_mask_len(&self) -> usize { self.inner.internal_mask_len() }

    #[getter]
    fn original_mask_len(&self) -> usize { self.inner.original_mask_len() }

    fn class_of(&self, token_id: u32) -> Option<u32> { self.inner.class_of(token_id) }

    fn representative(&self, class_id: u32) -> Option<u32> {
        self.inner.representative(class_id)
    }

    fn classes(&self) -> Vec<Vec<u32>> { self.inner.classes().to_vec() }

    fn original_to_class(&self) -> Vec<u32> { self.inner.original_to_class().to_vec() }

    fn expand_mask(&self, internal_mask: Vec<u64>) -> Vec<u32> {
        self.inner.expand_mask(&internal_mask)
    }

    fn fill_expanded_mask(
        &self,
        internal_mask: Vec<u64>,
        mut bitmask: PyReadwriteArray1<i32>,
    ) -> PyResult<()> {
        let buf = bitmask_u32_view(&mut bitmask)?;
        self.inner.fill_expanded_mask(&internal_mask, buf);
        Ok(())
    }
}
