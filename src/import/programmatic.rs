//! Unsupported legacy JSON-value adapters and the retained test-only prototype.

#[cfg(test)]
use super::bindings::external_placeholder_token_id_avoiding;
#[cfg(test)]
use super::json_schema;
#[cfg(test)]
use super::lowering::prepare_json_schema_named;
#[cfg(test)]
use super::lowering::with_large_import_stack;
#[cfg(test)]
use super::static_compile::compile_from_named_grammar;
#[cfg(test)]
use crate::compiler::glr::table::GlrTableConstruction;
use crate::runtime::Constraint;
#[cfg(test)]
use std::collections::BTreeSet;
impl Constraint {
    /// Compile a JSON Schema whose nested property/array value positions may
    /// also be satisfied by an already-compiled dynamic-value subgrammar.
    ///
    /// The schema root itself remains schema-controlled. For an object schema,
    /// for example, `customer` cannot replace the whole arguments object, but
    /// `{"customer_id": customer.id}` may use the dynamic child for the
    /// `customer_id` value. Literal values continue through the ordinary schema
    /// branch and therefore retain enum/range/pattern/object-shape validation.
    #[deprecated(note = "Programmatic JSON schema values are unsupported and not implemented")]
    #[allow(deprecated)]
    pub(crate) fn from_json_schema_with_dynamic_value(
        schema: &str,
        dynamic_value: &Constraint,
        vocab: &crate::Vocab,
    ) -> crate::Result<Self> {
        Self::from_json_schema_with_dynamic_value_and_end_tokens(schema, dynamic_value, vocab, &[])
    }

    /// Compile a JSON Schema with a nested dynamic-value subgrammar and model
    /// end-token IDs.
    #[deprecated(note = "Programmatic JSON schema values are unsupported and not implemented")]
    #[allow(deprecated)]
    pub(crate) fn from_json_schema_with_dynamic_value_and_end_tokens(
        schema: &str,
        dynamic_value: &Constraint,
        vocab: &crate::Vocab,
        end_token_ids: &[u32],
    ) -> crate::Result<Self> {
        // This private legacy entry point remains intentionally unsupported.
        return Err(crate::GlrMaskError::Compilation(
            "programmatic JSON schema values are unsupported (not implemented)".into(),
        ));
    }

    /// Compile JSON Schema for programmatic JavaScript object/array values.
    /// Opaque runtime values are accepted at nested value positions, while
    /// conditional expressions keep both result branches recursively constrained
    /// by the same schema.
    #[deprecated(note = "Programmatic JSON schema values are unsupported and not implemented")]
    pub(crate) fn from_json_schema_with_programmatic_values(
        schema: &str,
        dynamic_value: &Constraint,
        condition: &Constraint,
        vocab: &crate::Vocab,
    ) -> crate::Result<Self> {
        // This private legacy entry point remains intentionally unsupported.
        return Err(crate::GlrMaskError::Compilation(
            "programmatic JSON schema values are unsupported (not implemented)".into(),
        ));
    }

    /// Test-only variant of [`Self::from_json_schema_with_programmatic_values`]
    /// that composes the schema parent with the value/condition children using
    /// an explicit segmented boundary backend (or the flattened `None` path).
    /// This isolates whether the flattened schema composition or the shared
    /// prepared-leaf/walk stage drops a reserved-literal prefix token.
    #[cfg(test)]
    #[allow(deprecated)]
    pub(crate) fn from_json_schema_with_programmatic_values_backend(
        schema: &str,
        dynamic_value: &Constraint,
        condition: &Constraint,
        vocab: &crate::Vocab,
        backend: Option<crate::compiler::composition::SegmentedBoundaryBackend>,
    ) -> crate::Result<Self> {
        with_large_import_stack(schema.len(), || {
            let child_reserved = dynamic_value
                .special_token_terminals
                .iter()
                .chain(condition.special_token_terminals.iter())
                .map(|special| special.token_id)
                .collect::<BTreeSet<_>>();
            let value_token_id =
                external_placeholder_token_id_avoiding(vocab, child_reserved.iter().copied())?;
            let condition_token_id = external_placeholder_token_id_avoiding(
                vocab,
                child_reserved.iter().copied().chain(std::iter::once(value_token_id)),
            )?;
            let schema_value: serde_json::Value =
                serde_json::from_str(schema).map_err(|error| {
                    crate::GlrMaskError::GrammarParse(format!("invalid JSON: {error}"))
                })?;
            let mut named = json_schema::schema_to_named_grammar_with_programmatic_value_tokens(
                &schema_value,
                value_token_id,
                condition_token_id,
            )?;
            prepare_json_schema_named(&mut named)?;
            let parent = compile_from_named_grammar(
                named,
                vocab,
                "json_schema_programmatic_value",
                GlrTableConstruction::LegacyRowBisim,
                &[],
            )?;
            let terminal_for = |constraint: &Constraint, token_id: u32| -> crate::Result<u32> {
                constraint
                    .special_token_terminals
                    .iter()
                    .find(|special| special.token_id == token_id)
                    .map(|special| special.terminal_id)
                    .ok_or_else(|| {
                        crate::GlrMaskError::Compilation(format!(
                            "programmatic JSON Schema lost linker token {token_id}"
                        ))
                    })
            };
            let value_terminal = terminal_for(&parent, value_token_id)?;
            let compose = |parent: Constraint,
                           placeholder: u32,
                           child: &Constraint|
             -> crate::Result<Constraint> {
                let input = [crate::compiler::composition::CompiledSubgrammarInput {
                    placeholder_terminal: placeholder,
                    additional_placeholder_terminals: &[],
                    constraint: child,
                }];
                let composition = match backend {
                    None => crate::compiler::composition::compose_constraints_owned_parent(
                        parent, &input, vocab,
                    ),
                    Some(backend) => {
                        crate::compiler::composition::compose_constraints_owned_parent_segmented(
                            parent, &input, vocab, backend,
                        )
                    }
                };
                composition
                    .map(|composition| composition.constraint)
                    .map_err(crate::GlrMaskError::Compilation)
            };
            let with_value = compose(parent, value_terminal, dynamic_value)?;
            let condition_terminal = terminal_for(&with_value, condition_token_id)?;
            compose(with_value, condition_terminal, condition)
        })
    }
}
