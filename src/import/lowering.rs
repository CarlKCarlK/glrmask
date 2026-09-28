//! Source parsing, factoring, schema preparation and vocabulary-analysis lowering.

use super::ast;
use super::ebnf;
use super::json_schema;
use super::lark;
use crate::compiler::compile::compile_profile_enabled;
use crate::grammar::factoring::factor_named_grammar;
use crate::grammar::flat::GrammarDef;
use std::collections::BTreeSet;
pub(super) fn parse_ebnf_to_named(source: &str) -> crate::Result<ast::NamedGrammar> {
    Ok(ebnf::parse_ebnf_to_named(source)?)
}

pub(super) fn parse_lark_to_named(source: &str) -> crate::Result<ast::NamedGrammar> {
    Ok(lark::parse_lark_to_named(source)?)
}

pub(super) fn parse_glrm_to_named(source: &str) -> crate::Result<ast::NamedGrammar> {
    Ok(crate::grammar::glrm::from_glrm(source)?)
}

pub(super) fn parse_glrm_with_external_terminal_bindings(
    source: &str,
    bindings: &[(&str, &[u32])],
) -> crate::Result<ast::NamedGrammar> {
    Ok(crate::grammar::glrm::from_glrm_with_external_terminals(source, bindings)?)
}

pub(super) fn prepare_json_schema_named(grammar: &mut ast::NamedGrammar) -> crate::Result<()> {
    Ok(json_schema::prepare_named_grammar(grammar)?)
}

pub(super) type NamedGrammarParser = fn(&str) -> crate::Result<ast::NamedGrammar>;

pub(super) type NamedGrammarTransform = fn(&mut ast::NamedGrammar) -> crate::Result<()>;

pub(super) const LARGE_IMPORT_SOURCE_BYTES: usize = 64 * 1024;

#[cfg(windows)]
pub(super) const WINDOWS_LARGE_IMPORT_STACK_BYTES: usize = 64 * 1024 * 1024;

pub(super) fn with_large_import_stack<T, F>(source_len: usize, compile: F) -> T
where
    T: Send,
    F: FnOnce() -> T + Send,
{
    #[cfg(windows)]
    {
        if source_len >= LARGE_IMPORT_SOURCE_BYTES {
            return std::thread::scope(|scope| {
                let thread = std::thread::Builder::new()
                    .name("glrmask-grammar-compile".to_owned())
                    .stack_size(WINDOWS_LARGE_IMPORT_STACK_BYTES)
                    .spawn_scoped(scope, compile)
                    .expect("failed to spawn large-stack grammar compiler thread");
                match thread.join() {
                    Ok(result) => result,
                    Err(panic) => std::panic::resume_unwind(panic),
                }
            });
        }
    }

    #[cfg(not(windows))]
    let _ = source_len;

    compile()
}

pub(crate) fn choice_or_single(mut options: Vec<ast::GrammarExpr>) -> ast::GrammarExpr {
    if options.len() == 1 { options.pop().unwrap() } else { ast::GrammarExpr::Choice(options) }
}

pub(crate) fn sequence_or_single(mut items: Vec<ast::GrammarExpr>) -> ast::GrammarExpr {
    match items.len() {
        0 => ast::GrammarExpr::Sequence(Vec::new()),
        1 => items.pop().unwrap(),
        _ => ast::GrammarExpr::Sequence(items),
    }
}

pub(super) fn append_end_token_choice(grammar: &mut ast::NamedGrammar, end_token_ids: &[u32]) {
    let end_token_ids = end_token_ids.iter().copied().collect::<BTreeSet<_>>();
    if end_token_ids.is_empty() {
        return;
    }

    let original_start = grammar.start.clone();
    let base = "__glrmask_start_with_end_token";
    let mut generated_start = base.to_owned();
    let mut suffix = 2usize;
    while grammar.rules.iter().any(|rule| rule.name == generated_start) {
        generated_start = format!("{base}_{suffix}");
        suffix += 1;
    }

    let end =
        choice_or_single(end_token_ids.into_iter().map(ast::GrammarExpr::SpecialToken).collect());
    grammar.rules.push(ast::NamedRule {
        name: generated_start.clone(),
        expr: sequence_or_single(vec![ast::GrammarExpr::Ref(original_start), end]),
        is_terminal: false,
        is_internal: false,
    });
    grammar.start = generated_start;
}

pub(super) fn emit_import_phase_start(name: &'static str) -> Option<std::time::Instant> {
    if !compile_profile_enabled() {
        return None;
    }

    eprintln!("[glrmask/profile][import-phase-start] name={}", name);
    Some(std::time::Instant::now())
}

pub(super) fn emit_import_phase_end(name: &'static str, started_at: Option<std::time::Instant>) {
    if let Some(started_at) = started_at {
        eprintln!(
            "[glrmask/profile][import-phase-end] name={} elapsed_ms={:.3}",
            name,
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
}

pub(super) fn lower_factored_named_grammar(
    source: &str,
    parse_named: NamedGrammarParser,
    transform: Option<NamedGrammarTransform>,
    end_token_ids: &[u32],
) -> crate::Result<GrammarDef> {
    let lower_started_at = emit_import_phase_start("lower_factored_named_grammar");
    let parse_named_started_at = emit_import_phase_start("parse_named");
    let named = parse_named(source)?;
    emit_import_phase_end("parse_named", parse_named_started_at);

    let factor_started_at = emit_import_phase_start("factor_named_grammar");
    let mut factored = factor_named_grammar(named);
    emit_import_phase_end("factor_named_grammar", factor_started_at);

    if let Some(transform) = transform {
        let transform_started_at = emit_import_phase_start("transform_named_grammar");
        transform(&mut factored)?;
        emit_import_phase_end("transform_named_grammar", transform_started_at);
    }
    append_end_token_choice(&mut factored, end_token_ids);

    let ast_lower_started_at = emit_import_phase_start("ast_lower");
    let grammar = ast::lower(&factored);
    emit_import_phase_end("ast_lower", ast_lower_started_at);
    emit_import_phase_end("lower_factored_named_grammar", lower_started_at);
    Ok(grammar?)
}

pub(super) fn lower_json_schema_for_vocab_partition(source: &str) -> crate::Result<GrammarDef> {
    let lower_started_at = emit_import_phase_start("lower_factored_named_grammar");
    let parse_named_started_at = emit_import_phase_start("parse_named");
    let named = parse_json_schema_to_named_dynamic_vocab_partition(source)?;
    emit_import_phase_end("parse_named", parse_named_started_at);

    let factor_started_at = emit_import_phase_start("factor_named_grammar");
    let mut factored = factor_named_grammar(named);
    emit_import_phase_end("factor_named_grammar", factor_started_at);

    let transform_started_at = emit_import_phase_start("transform_named_grammar");
    let resolved_terminal_exprs = json_schema::prepare_named_grammar_for_lowering(&mut factored)?;
    emit_import_phase_end("transform_named_grammar", transform_started_at);

    let ast_lower_started_at = emit_import_phase_start("ast_lower");
    let grammar = ast::lower_with_resolved_terminal_exprs(&factored, resolved_terminal_exprs);
    emit_import_phase_end("ast_lower", ast_lower_started_at);
    emit_import_phase_end("lower_factored_named_grammar", lower_started_at);
    Ok(grammar?)
}

pub(crate) fn lower_source_for_vocab_partition(
    source_kind: &str,
    source: &str,
) -> crate::Result<GrammarDef> {
    match source_kind {
        "ebnf" => lower_factored_named_grammar(source, parse_ebnf_to_named, None, &[]),
        "lark" => lower_factored_named_grammar(source, parse_lark_to_named, None, &[]),
        "json_schema" => lower_json_schema_for_vocab_partition(source),
        "glrm" => {
            let named = parse_glrm_with_external_terminal_bindings(source, &[])?;
            let factored = factor_named_grammar(named);
            Ok(ast::lower(&factored)?)
        }
        other => {
            Err(crate::Error::GrammarParse(format!("unsupported grammar source kind {other:?}")))
        }
    }
}

/// Profiling-only entry point: runs the JSON-schema import pipeline
/// (parse → factor → AST lower) without the downstream compile. Hidden from the
/// public API; used by `examples/profile_glr.rs` to isolate import timings.
#[doc(hidden)]
pub fn __profile_json_schema_import(schema_json: &str) -> crate::Result<()> {
    let grammar = lower_factored_named_grammar(
        schema_json,
        parse_json_schema_to_named,
        Some(prepare_json_schema_named),
        &[],
    )?;
    std::hint::black_box(&grammar);
    Ok(())
}

pub(super) fn parse_json_schema_to_named(schema_json: &str) -> crate::Result<ast::NamedGrammar> {
    let profile_top = std::env::var_os("GLRMASK_PROFILE_DYNAMIC_TOP").is_some();
    let parse_top_started = profile_top.then(std::time::Instant::now);
    let json_parse_started_at = emit_import_phase_start("serde_json_from_str");
    let schema: serde_json::Value = serde_json::from_str(schema_json)
        .map_err(|e| crate::GlrMaskError::GrammarParse(format!("invalid JSON: {e}")))?;
    emit_import_phase_end("serde_json_from_str", json_parse_started_at);
    let parse_top_ms =
        parse_top_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

    let lower_top_started = profile_top.then(std::time::Instant::now);
    let schema_to_named_started_at = emit_import_phase_start("schema_to_named_grammar");
    let named = json_schema::schema_to_named_grammar(&schema);
    emit_import_phase_end("schema_to_named_grammar", schema_to_named_started_at);
    if let Some(started) = lower_top_started {
        eprintln!(
            "[glrmask/profile][json_parse_top] serde_json_ms={:.3} schema_to_named_ms={:.3}",
            parse_top_ms,
            started.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Ok(named?)
}

pub(super) fn parse_json_schema_to_named_dynamic(
    schema_json: &str,
) -> crate::Result<ast::NamedGrammar> {
    let json_parse_started_at = emit_import_phase_start("serde_json_from_str");
    let schema: serde_json::Value = serde_json::from_str(schema_json)
        .map_err(|e| crate::GlrMaskError::GrammarParse(format!("invalid JSON: {e}")))?;
    emit_import_phase_end("serde_json_from_str", json_parse_started_at);

    let schema_to_named_started_at = emit_import_phase_start("schema_to_named_grammar");
    let named =
        glrmask_json_schema::__private::schema_to_named_grammar_for_runtime_dynamic(&schema, false);
    emit_import_phase_end("schema_to_named_grammar", schema_to_named_started_at);
    Ok(named?)
}

pub(super) fn parse_json_schema_to_named_dynamic_vocab_partition(
    schema_json: &str,
) -> crate::Result<ast::NamedGrammar> {
    let json_parse_started_at = emit_import_phase_start("serde_json_from_str");
    let schema: serde_json::Value = serde_json::from_str(schema_json)
        .map_err(|e| crate::GlrMaskError::GrammarParse(format!("invalid JSON: {e}")))?;
    emit_import_phase_end("serde_json_from_str", json_parse_started_at);

    let schema_to_named_started_at = emit_import_phase_start("schema_to_named_grammar");
    let named =
        glrmask_json_schema::__private::schema_to_named_grammar_for_runtime_dynamic(&schema, true);
    emit_import_phase_end("schema_to_named_grammar", schema_to_named_started_at);
    Ok(named?)
}
