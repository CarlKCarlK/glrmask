//! Data-only parser programs. Python objects are converted to owned Rust data
//! before validation/compilation; no Python callback is retained at runtime.
use super::{PyConstraint, PyVocab};
use super::final_api::{api_error, constraint, PyOptimization};
use glrmask::template_parser::{
    LexerDefinition, ParserDefinition, ParserProgram, TemplateBuildOptions, TerminalPattern,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyModule, PyString};

/// Immutable, validated acyclic parser program described by JSON-compatible
/// POP/READ/PUSH phase graphs. It can be compiled with multiple vocabularies.
#[pyclass(name = "ParserProgram", module = "glrmask", frozen)]
#[derive(Clone)]
pub(super) struct PyParserProgram {
    inner: ParserProgram,
    terminal_count: usize,
}

#[pymethods]
impl PyParserProgram {
    /// Construct from a ParserDefinition JSON string or a JSON-compatible
    /// mapping. Stack labels are {"Symbol": id} or "Default". Graph cycles,
    /// invalid links and resource-limit violations raise ValueError.
    #[new]
    fn new(py: Python<'_>, definition: &Bound<'_, PyAny>) -> PyResult<Self> {
        let source = if let Ok(text) = definition.downcast::<PyString>() {
            text.clone()
        } else {
            py.import("json")?.getattr("dumps")?.call1((definition,))?.downcast_into::<PyString>()?
        };
        let text = source.to_str()?;
        // Check before allocating the owned Rust input and decoding it.
        // Python's json.dumps may already have allocated the serialized text;
        // this is not a whole-process memory bound. The Rust validator also
        // bounds graph states, edges and derived metadata independently.
        if text.len() > 64 * 1024 * 1024 {
            return Err(PyValueError::new_err("parser definition exceeds the 64 MiB JSON input limit"));
        }
        let source = text.to_owned();
        py.allow_threads(move || {
            let definition: ParserDefinition = serde_json::from_str(&source).map_err(api_error)?;
            let terminal_count = definition.terminals.len();
            let inner = ParserProgram::new(definition).map_err(api_error)?;
            Ok(Self { inner, terminal_count })
        })
    }

    #[getter]
    fn terminal_count(&self) -> usize { self.terminal_count }

    /// Compile using the ordinary shared mask/commit engines. Each terminal
    /// pattern is bytes for a literal, or str for a regex; ordering matches the
    /// parser definition. FAST_RUNTIME selects static masks; FAST_BUILD, AUTO
    /// and the default select the dynamic mask engine. All choices remain
    /// physically LR-table-free. Ignored terminals must have identity actions.
    #[pyo3(signature = (vocab, terminal_patterns, *, ignore_terminal=None, end_tokens=None, optimization=None))]
    fn compile(
        &self,
        py: Python<'_>,
        vocab: &PyVocab,
        terminal_patterns: &Bound<'_, PyAny>,
        ignore_terminal: Option<u32>,
        end_tokens: Option<Vec<u32>>,
        optimization: Option<PyRef<'_, PyOptimization>>,
    ) -> PyResult<PyConstraint> {
        let mut terminals = Vec::with_capacity(self.terminal_count);
        for value in terminal_patterns.try_iter()? {
            let value = value?;
            if terminals.len() == self.terminal_count {
                return Err(PyValueError::new_err("too many terminal patterns for the parser definition"));
            }
            if let Ok(bytes) = value.downcast::<PyBytes>() {
                terminals.push(TerminalPattern::literal(bytes.as_bytes().to_vec()));
            } else if let Ok(regex) = value.downcast::<PyString>() {
                terminals.push(TerminalPattern::regex(regex.to_str()?.to_owned()));
            } else {
                return Err(PyTypeError::new_err("a terminal pattern must be bytes (literal) or str (regex)"));
            }
        }
        let mut lexer = LexerDefinition::new(terminals);
        if let Some(terminal) = ignore_terminal { lexer = lexer.ignoring(terminal); }
        let mut options = TemplateBuildOptions::default().end_tokens(end_tokens.unwrap_or_default());
        if let Some(value) = optimization {
            options = options.optimization(super::final_api::optimization(*value));
        }
        let vocab = vocab.inner.clone();
        py.allow_threads(|| self.inner.compile_with(&lexer, &vocab, options))
            .map(constraint).map_err(api_error)
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyParserProgram>()?;
    Ok(())
}
