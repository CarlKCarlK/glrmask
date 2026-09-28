//! Static source compilation and private constructor adapters.

use super::ast;
use super::lowering::NamedGrammarParser;
use super::lowering::NamedGrammarTransform;
use super::lowering::append_end_token_choice;
use super::lowering::emit_import_phase_end;
use super::lowering::emit_import_phase_start;
use super::lowering::lower_factored_named_grammar;
use super::lowering::parse_ebnf_to_named;
use super::lowering::parse_glrm_to_named;
use super::lowering::parse_glrm_with_external_terminal_bindings;
use super::lowering::parse_json_schema_to_named;
use super::lowering::parse_lark_to_named;
use super::lowering::prepare_json_schema_named;
use super::lowering::with_large_import_stack;
use crate::compiler::compile::compile_owned_profiled_with_table_construction;
use crate::compiler::compile::compile_owned_with_table_construction;
use crate::compiler::compile::compile_owned_with_table_construction_and_protected_shift_terminal_names;
use crate::compiler::compile::compile_profile_enabled;
use crate::compiler::compile::compile_top_profile_enabled;
use crate::compiler::compile::emit_compile_profile_summary;
use crate::compiler::glr::table::GlrTableConstruction;
use crate::grammar::factoring::factor_named_grammar;
use crate::runtime::Constraint;
pub(super) fn compile_from_source(
    source: &str,
    vocab: &crate::Vocab,
    source_kind: &str,
    default_table_construction: GlrTableConstruction,
    parse: NamedGrammarParser,
    transform: Option<NamedGrammarTransform>,
    end_token_ids: &[u32],
) -> crate::Result<Constraint> {
    let compile_from_source_started_at = emit_import_phase_start("compile_from_source");
    if compile_profile_enabled() || compile_top_profile_enabled() {
        let parse_started_at = std::time::Instant::now();
        let grammar = lower_factored_named_grammar(source, parse, transform, end_token_ids)?;
        let import_ms = parse_started_at.elapsed().as_secs_f64() * 1000.0;
        let (mut constraint, profile) = crate::error::catch_internal_invariant(|| {
            compile_owned_profiled_with_table_construction(
                grammar,
                vocab,
                default_table_construction,
            )
        })?;
        constraint.table.set_embedded_end_token_ids(end_token_ids);
        emit_compile_profile_summary(Some(source_kind), Some(import_ms), &profile);
        emit_import_phase_end("compile_from_source", compile_from_source_started_at);
        return Ok(constraint);
    }

    let grammar = lower_factored_named_grammar(source, parse, transform, end_token_ids)?;
    let mut constraint = crate::error::catch_internal_invariant(|| {
        compile_owned_with_table_construction(grammar, vocab, default_table_construction)
    })?;
    constraint.table.set_embedded_end_token_ids(end_token_ids);
    emit_import_phase_end("compile_from_source", compile_from_source_started_at);
    Ok(constraint)
}

pub(crate) fn compile_from_named_grammar(
    named: ast::NamedGrammar,
    vocab: &crate::Vocab,
    source_kind: &str,
    default_table_construction: GlrTableConstruction,
    end_token_ids: &[u32],
) -> crate::Result<Constraint> {
    let import_started_at = std::time::Instant::now();
    let mut factored = factor_named_grammar(named);
    append_end_token_choice(&mut factored, end_token_ids);
    let grammar = ast::lower(&factored)?;
    let import_ms = import_started_at.elapsed().as_secs_f64() * 1000.0;

    if compile_profile_enabled() || compile_top_profile_enabled() {
        let (mut constraint, profile) = crate::error::catch_internal_invariant(|| {
            compile_owned_profiled_with_table_construction(
                grammar,
                vocab,
                default_table_construction,
            )
        })?;
        constraint.table.set_embedded_end_token_ids(end_token_ids);
        emit_compile_profile_summary(Some(source_kind), Some(import_ms), &profile);
        return Ok(constraint);
    }

    let mut constraint = crate::error::catch_internal_invariant(|| {
        compile_owned_with_table_construction(grammar, vocab, default_table_construction)
    })?;
    constraint.table.set_embedded_end_token_ids(end_token_ids);
    Ok(constraint)
}

pub(crate) fn compile_glrm_with_protected_shift_terminals(
    glrm: &str,
    protected_terminal_names: &[&str],
    vocab: &crate::Vocab,
) -> crate::Result<Constraint> {
    with_large_import_stack(glrm.len(), || {
        let named = parse_glrm_to_named(glrm)?;
        let factored = factor_named_grammar(named);
        let grammar = ast::lower(&factored)?;
        let protected =
            protected_terminal_names.iter().map(|name| (*name).to_string()).collect::<Vec<_>>();
        crate::error::catch_internal_invariant(|| {
            compile_owned_with_table_construction_and_protected_shift_terminal_names(
                grammar,
                vocab,
                GlrTableConstruction::ExperimentalCoreMerged,
                protected,
            )
        })
    })
}

impl Constraint {
    /// Compile an EBNF grammar for `vocab`.
    pub(crate) fn from_ebnf(ebnf: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        Self::from_ebnf_with_end_tokens(ebnf, vocab, &[])
    }

    /// Compile an EBNF grammar and declare model end-token IDs.
    pub(crate) fn from_ebnf_with_end_tokens(
        ebnf: &str,
        vocab: &crate::Vocab,
        end_token_ids: &[u32],
    ) -> crate::Result<Self> {
        with_large_import_stack(ebnf.len(), || {
            compile_from_source(
                ebnf,
                vocab,
                "ebnf",
                GlrTableConstruction::ExperimentalCoreMerged,
                parse_ebnf_to_named,
                None,
                end_token_ids,
            )
        })
    }

    /// Compile a Lark grammar for `vocab`.
    pub(crate) fn from_lark(lark: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        Self::from_lark_with_end_tokens(lark, vocab, &[])
    }

    /// Compile a Lark grammar and declare model end-token IDs.
    pub(crate) fn from_lark_with_end_tokens(
        lark: &str,
        vocab: &crate::Vocab,
        end_token_ids: &[u32],
    ) -> crate::Result<Self> {
        with_large_import_stack(lark.len(), || {
            compile_from_source(
                lark,
                vocab,
                "lark",
                GlrTableConstruction::ExperimentalCoreMerged,
                parse_lark_to_named,
                None,
                end_token_ids,
            )
        })
    }

    /// Compile a JSON Schema for `vocab`.
    pub(crate) fn from_json_schema(schema: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        Self::from_json_schema_with_end_tokens(schema, vocab, &[])
    }

    /// Compile a JSON Schema and declare model end-token IDs.
    pub(crate) fn from_json_schema_with_end_tokens(
        schema: &str,
        vocab: &crate::Vocab,
        end_token_ids: &[u32],
    ) -> crate::Result<Self> {
        with_large_import_stack(schema.len(), || {
            crate::compiler::stages::id_map_and_terminal_dwa::l2p::with_ti_pool(|| {
                compile_from_source(
                    schema,
                    vocab,
                    "json_schema",
                    GlrTableConstruction::LegacyRowBisim,
                    parse_json_schema_to_named,
                    Some(prepare_json_schema_named),
                    end_token_ids,
                )
            })
        })
    }

    /// Compile a grammar in GLRMask's native GLRM format.
    pub(crate) fn from_glrm_grammar(glrm: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        Self::from_glrm_grammar_with_end_tokens(glrm, vocab, &[])
    }

    /// Compile a GLRM grammar and declare model end-token IDs.
    pub(crate) fn from_glrm_grammar_with_end_tokens(
        glrm: &str,
        vocab: &crate::Vocab,
        end_token_ids: &[u32],
    ) -> crate::Result<Self> {
        Self::from_glrm_grammar_with_bindings_and_end_tokens(glrm, vocab, &[], end_token_ids)
    }

    pub(crate) fn from_glrm_grammar_with_bindings_and_end_tokens(
        glrm: &str,
        vocab: &crate::Vocab,
        bindings: &[(&str, &[u32])],
        end_token_ids: &[u32],
    ) -> crate::Result<Self> {
        with_large_import_stack(glrm.len(), || {
            let named = parse_glrm_with_external_terminal_bindings(glrm, bindings)?;
            compile_from_named_grammar(
                named,
                vocab,
                "glrm",
                GlrTableConstruction::ExperimentalCoreMerged,
                end_token_ids,
            )
        })
    }
}
