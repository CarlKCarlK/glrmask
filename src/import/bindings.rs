//! GLRM external-token allocation and compiled-child binding adapters.

use super::dynamic_compile::compile_dynamic_from_named;
use super::lowering::with_large_import_stack;
use super::static_compile::compile_from_named_grammar;
use crate::compiler::glr::table::GlrTableConstruction;
use crate::runtime::Constraint;
use crate::runtime::dynamic::DynamicConstraint;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
pub(super) fn first_external_placeholder_token_id(vocab: &crate::Vocab) -> crate::Result<u32> {
    if vocab.is_empty() {
        return Ok(0);
    }
    if let Some(next) = vocab.max_token_id().checked_add(1) {
        return Ok(next);
    }

    // Only the pathological `u32::MAX` case needs a vocabulary scan. Normal
    // dense model vocabularies stay on the constant-time max+1 path.
    let mut expected = 0u32;
    for (token_id, _) in vocab.iter() {
        if token_id != expected {
            return Ok(expected);
        }
        expected = expected.checked_add(1).ok_or_else(|| {
            crate::GlrMaskError::Compilation(
                "every u32 token ID is occupied; no external-subgrammar placeholder is available"
                    .to_string(),
            )
        })?;
    }
    Ok(expected)
}

pub(crate) fn external_placeholder_token_id_avoiding(
    vocab: &crate::Vocab,
    reserved: impl IntoIterator<Item = u32>,
) -> crate::Result<u32> {
    let reserved = reserved.into_iter().collect::<BTreeSet<_>>();
    let mut candidate = first_external_placeholder_token_id(vocab)?;
    loop {
        if !reserved.contains(&candidate) && !vocab.contains_exact_token_id(candidate) {
            return Ok(candidate);
        }
        candidate = candidate.checked_add(1).ok_or_else(|| {
            crate::GlrMaskError::Compilation(
                "every u32 token ID is occupied; no external-subgrammar placeholder is available"
                    .to_string(),
            )
        })?;
    }
}

impl Constraint {
    /// Compile a GLRM parent shell while retaining unresolved `extern grammar`
    /// declarations as named hidden linker terminals. Those terminals use
    /// non-vocabulary token IDs, so an unresolved call site is unreachable at
    /// runtime until a later compiled-constraint binding replaces it.
    pub(crate) fn from_glrm_grammar_with_unbound_subgrammars_bindings_and_end_tokens(
        glrm: &str,
        vocab: &crate::Vocab,
        terminal_bindings: &[(&str, &[u32])],
        end_token_ids: &[u32],
    ) -> crate::Result<Self> {
        with_large_import_stack(glrm.len(), || {
            let first_placeholder_token_id = first_external_placeholder_token_id(vocab)?;
            let parsed = crate::grammar::glrm::from_glrm_with_bindings_and_external_subgrammars(
                glrm,
                first_placeholder_token_id,
                end_token_ids.iter().copied(),
                terminal_bindings,
            )?;
            let mut parent = compile_from_named_grammar(
                parsed.grammar,
                vocab,
                "glrm",
                GlrTableConstruction::ExperimentalCoreMerged,
                end_token_ids,
            )?;
            let mut slots = BTreeMap::new();
            for placeholder in parsed.placeholders {
                let mut matching = parent
                    .special_token_terminals
                    .iter()
                    .filter(|special| special.token_id == placeholder.token_id)
                    .map(|special| special.terminal_id);
                let terminal_id = matching.next().ok_or_else(|| {
                    crate::GlrMaskError::Compilation(format!(
                        "compiled GLRM external subgrammar {:?} lost its hidden linker terminal",
                        placeholder.binding_name,
                    ))
                })?;
                if matching.next().is_some() {
                    return Err(crate::GlrMaskError::Compilation(format!(
                        "compiled GLRM external subgrammar {:?} has multiple hidden linker terminals",
                        placeholder.binding_name,
                    )));
                }
                if slots.insert(placeholder.binding_name.clone(), terminal_id).is_some() {
                    return Err(crate::GlrMaskError::Compilation(format!(
                        "compiled GLRM external subgrammar {:?} was emitted more than once",
                        placeholder.binding_name,
                    )));
                }
            }
            parent.unbound_grammar_placeholders = slots;
            Ok(parent)
        })
    }

    /// Compile GLRM containing typed `extern grammar name;` declarations and bind
    /// each declaration to an already-compiled child constraint.
    ///
    /// Binding names are the source names of top-level externals. Externals
    /// nested inside inline subgrammars use qualified names such as
    /// `outer::leaf`. Hidden non-vocabulary linker-control IDs are allocated
    /// automatically; callers never need to manufacture `@token(...)`
    /// sentinels. Missing bindings are retained as reusable late-binding slots;
    /// duplicate and unknown supplied bindings are rejected before linking.
    pub(crate) fn from_glrm_grammar_with_subgrammars(
        glrm: &str,
        children: &[(&str, &Constraint)],
        vocab: &crate::Vocab,
    ) -> crate::Result<Self> {
        Self::from_glrm_grammar_with_subgrammars_and_end_tokens(glrm, children, vocab, &[])
    }

    /// Compile GLRM with typed external subgrammars and model end-token IDs.
    pub(crate) fn from_glrm_grammar_with_subgrammars_and_end_tokens(
        glrm: &str,
        children: &[(&str, &Constraint)],
        vocab: &crate::Vocab,
        end_token_ids: &[u32],
    ) -> crate::Result<Self> {
        Self::from_glrm_grammar_with_subgrammars_bindings_and_end_tokens(
            glrm,
            children,
            vocab,
            &[],
            end_token_ids,
        )
    }

    pub(crate) fn from_glrm_grammar_with_subgrammars_bindings_and_end_tokens(
        glrm: &str,
        children: &[(&str, &Constraint)],
        vocab: &crate::Vocab,
        terminal_bindings: &[(&str, &[u32])],
        end_token_ids: &[u32],
    ) -> crate::Result<Self> {
        with_large_import_stack(glrm.len(), || {
            let first_placeholder_token_id = first_external_placeholder_token_id(vocab)?;
            let parsed = crate::grammar::glrm::from_glrm_with_bindings_and_external_subgrammars(
                glrm,
                first_placeholder_token_id,
                end_token_ids.iter().copied().chain(children.iter().flat_map(|(_, child)| {
                    child.special_token_terminals.iter().map(|special| special.token_id)
                })),
                terminal_bindings,
            )?;

            let mut children_by_name = BTreeMap::<&str, &Constraint>::new();
            for &(binding_name, child) in children {
                if children_by_name.insert(binding_name, child).is_some() {
                    return Err(crate::GlrMaskError::Compilation(format!(
                        "external subgrammar binding {binding_name:?} was supplied more than once",
                    )));
                }
            }

            let mut external_bindings = Vec::with_capacity(parsed.placeholders.len());
            let mut unresolved_placeholders = Vec::new();
            for placeholder in &parsed.placeholders {
                if let Some(child) = children_by_name.remove(placeholder.binding_name.as_str()) {
                    external_bindings.push((
                        placeholder.token_id,
                        placeholder.binding_name.as_str(),
                        child,
                    ));
                } else {
                    unresolved_placeholders
                        .push((placeholder.token_id, placeholder.binding_name.clone()));
                }
            }
            if let Some((&unknown, _)) = children_by_name.first_key_value() {
                return Err(crate::GlrMaskError::Compilation(format!(
                    "compiled child was supplied for unknown external subgrammar {unknown:?}",
                )));
            }

            let mut parent = compile_from_named_grammar(
                parsed.grammar,
                vocab,
                "glrm",
                GlrTableConstruction::ExperimentalCoreMerged,
                end_token_ids,
            )?;
            for (placeholder_token_id, binding_name) in unresolved_placeholders {
                let mut matching_terminals = parent
                    .special_token_terminals
                    .iter()
                    .filter(|special| special.token_id == placeholder_token_id)
                    .map(|special| special.terminal_id);
                let terminal_id = matching_terminals.next().ok_or_else(|| {
                    crate::GlrMaskError::Compilation(format!(
                        "compiled GLRM external subgrammar {binding_name:?} lost its hidden linker terminal",
                    ))
                })?;
                if matching_terminals.next().is_some() {
                    return Err(crate::GlrMaskError::Compilation(format!(
                        "compiled GLRM external subgrammar {binding_name:?} has multiple hidden linker terminals",
                    )));
                }
                parent
                    .late_grammar_slots
                    .push(crate::runtime::LateGrammarSlot { name: binding_name, terminal_id });
            }
            if parent.sanitize_late_grammar_placeholder_token_domain() {
                parent.rebuild_runtime_caches();
            }
            if external_bindings.is_empty() {
                return Ok(parent);
            }

            let mut composition_inputs = Vec::with_capacity(external_bindings.len());
            for &(placeholder_token_id, binding_name, child) in &external_bindings {
                let mut matching_terminals = parent
                    .special_token_terminals
                    .iter()
                    .filter(|special| special.token_id == placeholder_token_id)
                    .map(|special| special.terminal_id);
                let placeholder_terminal = matching_terminals.next().ok_or_else(|| {
                    crate::GlrMaskError::Compilation(format!(
                        "compiled GLRM external subgrammar {binding_name:?} lost its hidden linker terminal",
                    ))
                })?;
                if matching_terminals.next().is_some() {
                    return Err(crate::GlrMaskError::Compilation(format!(
                        "compiled GLRM external subgrammar {binding_name:?} has multiple hidden linker terminals",
                    )));
                }
                composition_inputs.push(crate::compiler::composition::CompiledSubgrammarInput {
                    placeholder_terminal,
                    additional_placeholder_terminals: &[],
                    constraint: child,
                });
            }
            let parent_late_slots = parent.late_grammar_slots.clone();
            let mut composition =
                crate::compiler::composition::compose_constraints_owned_parent_segmented(
                    parent,
                    &composition_inputs,
                    vocab,
                    crate::compiler::composition::SegmentedBoundaryBackend::StaticParserDwa,
                )
                .map_err(crate::GlrMaskError::Compilation)?;
            // Parent terminals keep offset zero. Child slots are rebased into
            // the unified table and qualified to avoid collisions.
            let mut retained_slots = parent_late_slots;
            for (child_index, (_, binding_name, child)) in external_bindings.iter().enumerate() {
                let offset = composition.terminal_offsets[child_index + 1];
                retained_slots.extend(child.late_grammar_slots.iter().map(|slot| {
                    crate::runtime::LateGrammarSlot {
                        name: format!("{binding_name}::{}", slot.name),
                        terminal_id: offset + slot.terminal_id,
                    }
                }));
            }
            composition.constraint.late_grammar_slots = retained_slots;
            if composition.constraint.sanitize_late_grammar_placeholder_token_domain() {
                composition.constraint.rebuild_runtime_caches();
            }
            Ok(composition.constraint)
        })
    }
}

impl DynamicConstraint {
    pub(crate) fn from_glrm_grammar_with_subgrammars_and_bindings(
        glrm: &str,
        children: &[(&str, &Constraint)],
        vocab: &crate::Vocab,
        terminal_bindings: &[(&str, &[u32])],
    ) -> crate::Result<Self> {
        with_large_import_stack(glrm.len(), || {
            let first_placeholder_token_id = first_external_placeholder_token_id(vocab)?;
            let parsed = crate::grammar::glrm::from_glrm_with_bindings_and_external_subgrammars(
                glrm,
                first_placeholder_token_id,
                children.iter().flat_map(|(_, child)| {
                    child.special_token_terminals.iter().map(|special| special.token_id)
                }),
                terminal_bindings,
            )?;

            let mut children_by_name = BTreeMap::<&str, &Constraint>::new();
            for &(binding_name, child) in children {
                if children_by_name.insert(binding_name, child).is_some() {
                    return Err(crate::GlrMaskError::Compilation(format!(
                        "external subgrammar binding {binding_name:?} was supplied more than once",
                    )));
                }
            }

            let mut external_bindings = Vec::with_capacity(parsed.placeholders.len());
            let mut unresolved_placeholders = Vec::new();
            for placeholder in &parsed.placeholders {
                if let Some(child) = children_by_name.remove(placeholder.binding_name.as_str()) {
                    external_bindings.push((
                        placeholder.token_id,
                        placeholder.binding_name.as_str(),
                        child,
                    ));
                } else {
                    unresolved_placeholders
                        .push((placeholder.token_id, placeholder.binding_name.clone()));
                }
            }
            if let Some((&unknown, _)) = children_by_name.first_key_value() {
                return Err(crate::GlrMaskError::Compilation(format!(
                    "compiled child was supplied for unknown external subgrammar {unknown:?}",
                )));
            }

            let mut dynamic_parent = compile_dynamic_from_named(
                parsed.grammar,
                vocab,
                GlrTableConstruction::ExperimentalCoreMerged,
                &[],
            )?;
            dynamic_parent.attach_late_grammar_placeholders(&unresolved_placeholders)?;
            if external_bindings.is_empty() {
                return Ok(dynamic_parent);
            }
            let parents = dynamic_parent.clone_constraints();
            let mut composed = Vec::with_capacity(parents.len());
            for parent in parents {
                let mut composition_inputs = Vec::new();
                for &(placeholder_token_id, binding_name, child) in &external_bindings {
                    let mut matching_terminals = parent
                        .special_token_terminals
                        .iter()
                        .filter(|special| special.token_id == placeholder_token_id)
                        .map(|special| special.terminal_id);
                    let Some(placeholder_terminal) = matching_terminals.next() else {
                        continue;
                    };
                    if matching_terminals.next().is_some() {
                        return Err(crate::GlrMaskError::Compilation(format!(
                            "compiled GLRM external subgrammar {binding_name:?} has multiple hidden linker terminals",
                        )));
                    }
                    composition_inputs.push(
                        crate::compiler::composition::CompiledSubgrammarInput {
                            placeholder_terminal,
                            additional_placeholder_terminals: &[],
                            constraint: child,
                        },
                    );
                }
                if composition_inputs.is_empty() {
                    composed.push(parent);
                } else {
                    composed.push(
                        crate::compiler::composition::compose_constraints_owned_parent_segmented(
                            parent,
                            &composition_inputs,
                            vocab,
                            crate::compiler::composition::SegmentedBoundaryBackend::Dynamic,
                        )
                        .map_err(crate::GlrMaskError::Compilation)?
                        .constraint,
                    );
                }
            }
            Ok(DynamicConstraint::from_constraints(composed))
        })
    }
}
