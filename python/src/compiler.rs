//! Private compilation adapters, shape diagnostics, and compiler-cache controls.

use crate::conversion::constraint_result;
use crate::{PyConstraint, PyVocab};
use glrmask::__private::{ConstraintExt as _, DynamicConstraintExt as _, VocabExt as _};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::BTreeMap;
use std::sync::Arc;

#[pyfunction]
pub(super) fn compose_compiled_subgrammars(
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

pub(super) fn auto_ref_passthrough_key(key: &str) -> bool {
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

pub(super) fn auto_schema_type_contains(value: Option<&serde_json::Value>, expected: &str) -> bool {
    match value {
        Some(serde_json::Value::String(value)) => value == expected,
        Some(serde_json::Value::Array(values)) => values
            .iter()
            .any(|value| value.as_str().is_some_and(|value| value == expected)),
        _ => false,
    }
}

pub(super) fn auto_nonnegative_integer(value: Option<&serde_json::Value>) -> Option<u64> {
    let value = value?;
    if let Some(value) = value.as_u64() {
        return Some(value.min(1_000_000_000));
    }
    if let Some(value) = value.as_i64() {
        return (value >= 0).then_some((value as u64).min(1_000_000_000));
    }
    let value = value.as_f64()?;
    (value.is_finite() && value >= 0.0).then_some((value.trunc() as u64).min(1_000_000_000))
}

/// Return the exact source-shape counts used by Python AutoConstraint policy.
///
/// The walk mirrors CFA's two semantics-preserving JSON-Schema normalizations
/// without allocating normalized copies: ignored legacy `$ref` siblings are
/// omitted, and an empty `properties` object next to non-empty
/// `patternProperties` is skipped.  Keeping this native avoids making the auto
/// selector's build tail proportional to a Python object-tree traversal.
#[pyfunction]
pub(super) fn auto_json_schema_shape_counts(schema: &str) -> PyResult<Vec<u64>> {
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
                let skip_empty_properties =
                    properties.is_some_and(|map| map.is_empty()) && pattern_properties_nonempty;
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
pub(super) fn compile_ebnf_serialized_profiled(
    ebnf_source: &str,
    vocab: &PyVocab,
    end_token_ids: Option<Vec<u32>>,
    vocab_partition: bool,
) -> PyResult<(Vec<u8>, u64, u64)> {
    constraint_result(
        glrmask::DynamicConstraint::compile_ebnf_serialized_profiled_with_end_tokens(
            ebnf_source,
            &vocab.inner,
            end_token_ids.as_deref().unwrap_or(&[]),
            vocab_partition,
        ),
    )
}

#[pyfunction]
#[pyo3(signature = (lark_source, vocab, end_token_ids=None, vocab_partition=false))]
pub(super) fn compile_lark_serialized_profiled(
    lark_source: &str,
    vocab: &PyVocab,
    end_token_ids: Option<Vec<u32>>,
    vocab_partition: bool,
) -> PyResult<(Vec<u8>, u64, u64)> {
    constraint_result(
        glrmask::DynamicConstraint::compile_lark_serialized_profiled_with_end_tokens(
            lark_source,
            &vocab.inner,
            end_token_ids.as_deref().unwrap_or(&[]),
            vocab_partition,
        ),
    )
}

#[pyfunction]
#[pyo3(signature = (schema, vocab, end_token_ids=None, vocab_partition=false))]
pub(super) fn compile_json_schema_serialized_profiled(
    schema: &str,
    vocab: &PyVocab,
    end_token_ids: Option<Vec<u32>>,
    vocab_partition: bool,
) -> PyResult<(Vec<u8>, u64, u64)> {
    constraint_result(
        glrmask::DynamicConstraint::compile_json_schema_serialized_profiled_with_end_tokens(
            schema,
            &vocab.inner,
            end_token_ids.as_deref().unwrap_or(&[]),
            vocab_partition,
        ),
    )
}

#[pyfunction]
#[pyo3(signature = (glrm_source, vocab, end_token_ids=None, vocab_partition=false))]
pub(super) fn compile_glrm_serialized_profiled(
    glrm_source: &str,
    vocab: &PyVocab,
    end_token_ids: Option<Vec<u32>>,
    vocab_partition: bool,
) -> PyResult<(Vec<u8>, u64, u64)> {
    constraint_result(
        glrmask::DynamicConstraint::compile_glrm_serialized_profiled_with_end_tokens(
            glrm_source,
            &vocab.inner,
            end_token_ids.as_deref().unwrap_or(&[]),
            vocab_partition,
        ),
    )
}

#[pyfunction]
#[pyo3(signature = (ebnf_source, vocab, end_token_ids=None))]
pub(super) fn compile_ebnf_serialized(
    ebnf_source: &str,
    vocab: &PyVocab,
    end_token_ids: Option<Vec<u32>>,
) -> PyResult<Vec<u8>> {
    constraint_result(
        glrmask::DynamicConstraint::compile_ebnf_serialized_with_end_tokens(
            ebnf_source,
            &vocab.inner,
            end_token_ids.as_deref().unwrap_or(&[]),
        ),
    )
}

#[pyfunction]
#[pyo3(signature = (lark_source, vocab, end_token_ids=None))]
pub(super) fn compile_lark_serialized(
    lark_source: &str,
    vocab: &PyVocab,
    end_token_ids: Option<Vec<u32>>,
) -> PyResult<Vec<u8>> {
    constraint_result(
        glrmask::DynamicConstraint::compile_lark_serialized_with_end_tokens(
            lark_source,
            &vocab.inner,
            end_token_ids.as_deref().unwrap_or(&[]),
        ),
    )
}

#[pyfunction]
#[pyo3(signature = (schema, vocab, end_token_ids=None))]
pub(super) fn compile_json_schema_serialized(
    schema: &str,
    vocab: &PyVocab,
    end_token_ids: Option<Vec<u32>>,
) -> PyResult<Vec<u8>> {
    constraint_result(
        glrmask::DynamicConstraint::compile_json_schema_serialized_with_end_tokens(
            schema,
            &vocab.inner,
            end_token_ids.as_deref().unwrap_or(&[]),
        ),
    )
}

#[pyfunction]
#[pyo3(signature = (glrm_source, vocab, end_token_ids=None))]
pub(super) fn compile_glrm_serialized(
    glrm_source: &str,
    vocab: &PyVocab,
    end_token_ids: Option<Vec<u32>>,
) -> PyResult<Vec<u8>> {
    constraint_result(
        glrmask::DynamicConstraint::compile_glrm_serialized_with_end_tokens(
            glrm_source,
            &vocab.inner,
            end_token_ids.as_deref().unwrap_or(&[]),
        ),
    )
}

#[pyfunction]
pub(super) fn clear_stale_weights() {
    glrmask::Constraint::clear_stale_weights();
}

#[pyfunction]
pub(super) fn clear_weight_op_caches() {
    glrmask::Constraint::clear_weight_op_caches();
}

#[pyfunction]
pub(super) fn clear_weight_caches() {
    glrmask::Constraint::clear_weight_op_caches();
    glrmask::Constraint::clear_stale_weights();
}

#[pyfunction]
pub(super) fn compiler_cache_stats(
    vocab: Option<&PyVocab>,
) -> std::collections::BTreeMap<&'static str, u64> {
    let stats = glrmask::__private::compiler_cache_stats(vocab.map(|vocab| &vocab.inner));
    std::collections::BTreeMap::from([
        ("token_set_entries", stats.token_set_entries as u64),
        (
            "live_token_set_entries",
            stats.live_token_set_entries as u64,
        ),
        ("weight_buckets", stats.weight_buckets as u64),
        ("weight_entries", stats.weight_entries as u64),
        ("live_weight_entries", stats.live_weight_entries as u64),
        (
            "current_thread_weight_ops",
            stats.current_thread_weight_ops as u64,
        ),
        (
            "current_thread_token_set_ops",
            stats.current_thread_token_set_ops as u64,
        ),
        (
            "current_thread_public_intersections",
            stats.current_thread_public_intersections as u64,
        ),
        (
            "current_thread_weight_hashes",
            stats.current_thread_weight_hashes as u64,
        ),
        ("weight_op_generation", stats.weight_op_generation),
        ("weight_hash_generation", stats.weight_hash_generation),
        ("vocab_artifacts", stats.vocab_artifacts as u64),
    ])
}

#[pyfunction]
pub(super) fn prepare_vocab_for_compile(vocab: &PyVocab) {
    vocab.inner.prepare_for_compile();
}

#[pyfunction]
pub(super) fn prepare_vocab_for_dynamic_compile(vocab: &PyVocab) {
    vocab.inner.prepare_for_dynamic_compile();
}

#[pyfunction]
pub(super) fn compile_grammar_def_json(
    grammar_def_json: &str,
    vocab: &PyVocab,
) -> PyResult<PyConstraint> {
    let constraint = glrmask::Constraint::compile_grammar_def_json(grammar_def_json, &vocab.inner)
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;
    let max_token = constraint.max_original_token_id().unwrap_or(0);
    Ok(PyConstraint {
        inner: std::sync::Arc::new(constraint),
        max_token,
    })
}

#[pyfunction]
pub(super) fn dump_json_schema_grammar_glrm(schema_json: &str) -> PyResult<String> {
    glrmask::Constraint::dump_json_schema_grammar_glrm(schema_json)
        .map_err(|e| PyValueError::new_err(format!("{e}")))
}
