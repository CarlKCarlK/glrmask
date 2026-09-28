//! Dynamic alternative preparation and private runtime constructors.

use super::ast;
use super::lowering::NamedGrammarParser;
use super::lowering::NamedGrammarTransform;
use super::lowering::append_end_token_choice;
use super::lowering::parse_ebnf_to_named;
use super::lowering::parse_glrm_to_named;
use super::lowering::parse_glrm_with_external_terminal_bindings;
use super::lowering::parse_json_schema_to_named_dynamic;
use super::lowering::parse_json_schema_to_named_dynamic_vocab_partition;
use super::lowering::parse_lark_to_named;
use super::lowering::prepare_json_schema_named;
use super::lowering::with_large_import_stack;
use crate::compiler::glr::table::GlrTableConstruction;
use crate::compiler::pipeline::compile_dynamic_owned_with_table_construction;
use crate::compiler::pipeline::compile_dynamic_owned_with_vocab_partition_with_table_construction;
use crate::grammar::factoring::factor_named_grammar;
use crate::runtime::dynamic::DynamicConstraint;
pub(super) fn dynamic_named_alternatives(
    source: &str,
    parse: NamedGrammarParser,
    transform: Option<NamedGrammarTransform>,
    end_token_ids: &[u32],
) -> crate::Result<Vec<ast::NamedGrammar>> {
    let profile_top = std::env::var_os("GLRMASK_PROFILE_DYNAMIC_TOP").is_some();
    let parse_started = profile_top.then(std::time::Instant::now);
    let named = parse(source)?;
    let parse_ms = parse_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let alternatives_started = profile_top.then(std::time::Instant::now);
    let alternatives = dynamic_named_alternatives_from_named(named, transform, end_token_ids)?;
    if let Some(started) = alternatives_started {
        eprintln!(
            "[glrmask/profile][dynamic_import_top] parse_named_ms={:.3} alternatives_ms={:.3}",
            parse_ms,
            started.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Ok(alternatives)
}

pub(super) fn dynamic_named_alternatives_from_named(
    named: ast::NamedGrammar,
    transform: Option<NamedGrammarTransform>,
    end_token_ids: &[u32],
) -> crate::Result<Vec<ast::NamedGrammar>> {
    let profile_top = std::env::var_os("GLRMASK_PROFILE_DYNAMIC_TOP").is_some();
    let factor_started = profile_top.then(std::time::Instant::now);
    let mut factored = factor_named_grammar(named);
    let factor_ms = factor_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let transform_started = profile_top.then(std::time::Instant::now);
    if let Some(transform) = transform {
        transform(&mut factored)?;
    }
    let transform_ms =
        transform_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

    let extract_started = profile_top.then(std::time::Instant::now);

    let start_index =
        factored.rules.iter().position(|rule| !rule.is_terminal && rule.name == factored.start);
    let options = start_index.and_then(|index| match &factored.rules[index].expr {
        ast::GrammarExpr::Choice(options) if options.len() > 1 => Some(options.clone()),
        _ => None,
    });
    let has_embedded_region = factored.rules.iter().any(|rule| {
        !rule.is_terminal
            && matches!(
                &rule.expr,
                ast::GrammarExpr::ExprNFA(expr_nfa)
                    if expr_nfa.prefer_direct_nfa_emission
                        && !expr_nfa.complete_parser_language
            )
    });

    if !has_embedded_region || options.is_none() {
        append_end_token_choice(&mut factored, end_token_ids);
        if let Some(started) = extract_started {
            eprintln!(
                "[glrmask/profile][dynamic_alternatives_top] factor_ms={:.3} transform_ms={:.3} extract_ms={:.3} alternatives=1 embedded_region=false",
                factor_ms,
                transform_ms,
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
        return Ok(vec![factored]);
    }

    let start_index = start_index.expect("start choice index was resolved");
    let mut alternatives = Vec::new();
    for option in options.expect("start choice options were resolved") {
        let mut alternative = factored.clone();
        alternative.rules[start_index].expr = option.clone();
        if let ast::GrammarExpr::Ref(region_name) = &option
            && let Some(region_rule) = alternative
                .rules
                .iter_mut()
                .find(|rule| !rule.is_terminal && rule.name == *region_name)
            && let ast::GrammarExpr::ExprNFA(expr_nfa) = &mut region_rule.expr
            && expr_nfa.prefer_direct_nfa_emission
        {
            expr_nfa.complete_parser_language = true;
            alternative.start = region_name.clone();
        }
        crate::grammar::right_linear::retain_reachable_rules(&mut alternative);
        append_end_token_choice(&mut alternative, end_token_ids);
        alternatives.push(alternative);
    }
    if let Some(started) = extract_started {
        eprintln!(
            "[glrmask/profile][dynamic_alternatives_top] factor_ms={:.3} transform_ms={:.3} extract_ms={:.3} alternatives={} embedded_region=true",
            factor_ms,
            transform_ms,
            started.elapsed().as_secs_f64() * 1000.0,
            alternatives.len(),
        );
    }
    Ok(alternatives)
}

pub(super) fn compile_dynamic_from_named(
    named: ast::NamedGrammar,
    vocab: &crate::Vocab,
    default_table_construction: GlrTableConstruction,
    end_token_ids: &[u32],
) -> crate::Result<DynamicConstraint> {
    let alternatives = dynamic_named_alternatives_from_named(named, None, end_token_ids)?;
    let mut compiled = Vec::with_capacity(alternatives.len());
    for alternative in alternatives {
        let grammar = ast::lower(&alternative)?;
        compiled.push(compile_dynamic_owned_with_table_construction(
            grammar,
            vocab,
            default_table_construction,
        )?);
    }
    Ok(DynamicConstraint::from_alternatives(compiled))
}

pub(super) fn compile_dynamic_from_source(
    source: &str,
    vocab: &crate::Vocab,
    default_table_construction: GlrTableConstruction,
    parse: NamedGrammarParser,
    transform: Option<NamedGrammarTransform>,
    end_token_ids: &[u32],
) -> crate::Result<DynamicConstraint> {
    let profile_top = std::env::var_os("GLRMASK_PROFILE_DYNAMIC_TOP").is_some();
    let total_started = profile_top.then(std::time::Instant::now);
    let import_started = profile_top.then(std::time::Instant::now);
    let alternatives = dynamic_named_alternatives(source, parse, transform, end_token_ids)?;
    let import_ms = import_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let mut compiled = Vec::with_capacity(alternatives.len());
    let mut lower_ms = 0.0;
    let mut compile_ms = 0.0;
    for alternative in alternatives {
        let lower_started = profile_top.then(std::time::Instant::now);
        let grammar = ast::lower(&alternative)?;
        lower_ms += lower_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let compile_started = profile_top.then(std::time::Instant::now);
        compiled.push(compile_dynamic_owned_with_table_construction(
            grammar,
            vocab,
            default_table_construction,
        )?);
        compile_ms +=
            compile_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    }
    let wrap_started = profile_top.then(std::time::Instant::now);
    let result = DynamicConstraint::from_alternatives(compiled);
    let wrap_ms = wrap_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    if let Some(total_started) = total_started {
        eprintln!(
            "[glrmask/profile][dynamic_top] import_ms={:.3} ast_lower_ms={:.3} compile_ms={:.3} wrap_ms={:.3} total_ms={:.3}",
            import_ms,
            lower_ms,
            compile_ms,
            wrap_ms,
            total_started.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Ok(result)
}

pub(super) fn compile_dynamic_from_source_with_vocab_partition(
    source: &str,
    vocab: &crate::Vocab,
    default_table_construction: GlrTableConstruction,
    parse: NamedGrammarParser,
    transform: Option<NamedGrammarTransform>,
    end_token_ids: &[u32],
) -> crate::Result<DynamicConstraint> {
    let alternatives = dynamic_named_alternatives(source, parse, transform, end_token_ids)?;
    let mut compiled = Vec::with_capacity(alternatives.len());
    for alternative in alternatives {
        let grammar = ast::lower(&alternative)?;
        compiled.push(compile_dynamic_owned_with_vocab_partition_with_table_construction(
            grammar,
            vocab,
            default_table_construction,
        )?);
    }
    Ok(DynamicConstraint::from_alternatives(compiled))
}

impl DynamicConstraint {
    /// Compile an EBNF grammar with reduced compilation latency.
    pub(crate) fn from_ebnf(ebnf: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        Self::from_ebnf_with_end_tokens(ebnf, vocab, &[])
    }

    /// Compile an EBNF grammar using the grammar-specific vocabulary quotient
    /// for local dynamic mask generation.
    pub(crate) fn from_ebnf_with_vocab_partition(
        ebnf: &str,
        vocab: &crate::Vocab,
    ) -> crate::Result<Self> {
        with_large_import_stack(ebnf.len(), || {
            compile_dynamic_from_source_with_vocab_partition(
                ebnf,
                vocab,
                GlrTableConstruction::ExperimentalCoreMerged,
                parse_ebnf_to_named,
                None,
                &[],
            )
        })
    }

    /// Compile an EBNF grammar with reduced latency and model end-token IDs.
    pub(crate) fn from_ebnf_with_end_tokens(
        ebnf: &str,
        vocab: &crate::Vocab,
        end_token_ids: &[u32],
    ) -> crate::Result<Self> {
        with_large_import_stack(ebnf.len(), || {
            compile_dynamic_from_source(
                ebnf,
                vocab,
                GlrTableConstruction::ExperimentalCoreMerged,
                parse_ebnf_to_named,
                None,
                end_token_ids,
            )
        })
    }

    /// Compile a Lark grammar with reduced compilation latency.
    pub(crate) fn from_lark(lark: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        Self::from_lark_with_end_tokens(lark, vocab, &[])
    }

    /// Compile a Lark grammar using the grammar-specific vocabulary quotient
    /// for local dynamic mask generation.
    pub(crate) fn from_lark_with_vocab_partition(
        lark: &str,
        vocab: &crate::Vocab,
    ) -> crate::Result<Self> {
        with_large_import_stack(lark.len(), || {
            compile_dynamic_from_source_with_vocab_partition(
                lark,
                vocab,
                GlrTableConstruction::ExperimentalCoreMerged,
                parse_lark_to_named,
                None,
                &[],
            )
        })
    }

    /// Compile a Lark grammar with reduced latency and model end-token IDs.
    pub(crate) fn from_lark_with_end_tokens(
        lark: &str,
        vocab: &crate::Vocab,
        end_token_ids: &[u32],
    ) -> crate::Result<Self> {
        with_large_import_stack(lark.len(), || {
            compile_dynamic_from_source(
                lark,
                vocab,
                GlrTableConstruction::ExperimentalCoreMerged,
                parse_lark_to_named,
                None,
                end_token_ids,
            )
        })
    }

    /// Compile a JSON Schema with reduced compilation latency.
    pub(crate) fn from_json_schema(schema: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        Self::from_json_schema_with_end_tokens(schema, vocab, &[])
    }

    /// Compile JSON Schema using the grammar-specific vocabulary quotient for
    /// local dynamic mask generation.
    pub(crate) fn from_json_schema_with_vocab_partition(
        schema: &str,
        vocab: &crate::Vocab,
    ) -> crate::Result<Self> {
        with_large_import_stack(schema.len(), || {
            compile_dynamic_from_source_with_vocab_partition(
                schema,
                vocab,
                GlrTableConstruction::Lalr,
                parse_json_schema_to_named_dynamic_vocab_partition,
                Some(prepare_json_schema_named),
                &[],
            )
        })
    }

    /// Compile a JSON Schema with reduced latency and model end-token IDs.
    pub(crate) fn from_json_schema_with_end_tokens(
        schema: &str,
        vocab: &crate::Vocab,
        end_token_ids: &[u32],
    ) -> crate::Result<Self> {
        with_large_import_stack(schema.len(), || {
            compile_dynamic_from_source(
                schema,
                vocab,
                GlrTableConstruction::Lalr,
                parse_json_schema_to_named_dynamic,
                Some(prepare_json_schema_named),
                end_token_ids,
            )
        })
    }

    /// Compile a GLRM grammar with reduced compilation latency.
    pub(crate) fn from_glrm_grammar(glrm: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        Self::from_glrm_grammar_with_end_tokens(glrm, vocab, &[])
    }

    /// Compile GLRM using the grammar-specific vocabulary quotient for local
    /// dynamic mask generation.
    pub(crate) fn from_glrm_grammar_with_vocab_partition(
        glrm: &str,
        vocab: &crate::Vocab,
    ) -> crate::Result<Self> {
        with_large_import_stack(glrm.len(), || {
            compile_dynamic_from_source_with_vocab_partition(
                glrm,
                vocab,
                GlrTableConstruction::ExperimentalCoreMerged,
                parse_glrm_to_named,
                None,
                &[],
            )
        })
    }

    /// Compile a GLRM grammar with reduced latency and model end-token IDs.
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
            compile_dynamic_from_named(
                named,
                vocab,
                GlrTableConstruction::ExperimentalCoreMerged,
                end_token_ids,
            )
        })
    }
}
