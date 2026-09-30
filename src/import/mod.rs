pub use crate::grammar::ast as ast;
pub(crate) use glrmask_grammar::__private::import::ebnf;
pub mod json_schema;
pub(crate) use glrmask_grammar::__private::import::lark;

use std::collections::{BTreeMap, BTreeSet};

use crate::automata::lexer::Lexer;
use crate::compiler::compile::{
    compile_owned_profiled_with_table_construction,
    compile_owned_with_table_construction,
    compile_owned_with_table_construction_and_protected_shift_terminal_names,
    compile_profile_enabled,
    compile_top_profile_enabled,
    emit_compile_profile_summary,
};
use crate::compiler::pipeline::{
    compile_dynamic_owned_unfinalized_with_table_construction,
    compile_dynamic_owned_with_table_construction,
    compile_dynamic_owned_with_vocab_partition_unfinalized_with_table_construction,
    compile_dynamic_owned_with_vocab_partition_with_table_construction,
};
use crate::grammar::factoring::factor_named_grammar;
use crate::grammar::flat::GrammarDef;
use crate::compiler::glr::table::GlrTableConstruction;
use crate::runtime::Constraint;
use crate::dynamic_constraint::DynamicConstraint;

pub(crate) fn parse_ebnf_to_named(source: &str) -> crate::Result<ast::NamedGrammar> {
    Ok(ebnf::parse_ebnf_to_named(source)?)
}

pub(crate) fn parse_lark_to_named(source: &str) -> crate::Result<ast::NamedGrammar> {
    Ok(lark::parse_lark_to_named(source)?)
}

pub(crate) fn parse_glrm_to_named(source: &str) -> crate::Result<ast::NamedGrammar> {
    Ok(crate::grammar::glrm::from_glrm(source)?)
}

fn parse_glrm_with_external_terminal_bindings(
    source: &str,
    bindings: &[(&str, &[u32])],
) -> crate::Result<ast::NamedGrammar> {
    Ok(crate::grammar::glrm::from_glrm_with_external_terminals(
        source,
        bindings,
    )?)
}

fn prepare_json_schema_named(grammar: &mut ast::NamedGrammar) -> crate::Result<()> {
    Ok(json_schema::prepare_named_grammar(grammar)?)
}

type GrammarParser = fn(&str) -> crate::Result<GrammarDef>;
type NamedGrammarParser = fn(&str) -> crate::Result<ast::NamedGrammar>;
type NamedGrammarTransform = fn(&mut ast::NamedGrammar) -> crate::Result<()>;

const LARGE_IMPORT_SOURCE_BYTES: usize = 64 * 1024;
#[cfg(windows)]
const WINDOWS_LARGE_IMPORT_STACK_BYTES: usize = 64 * 1024 * 1024;

pub(crate) fn with_large_import_stack<T, F>(source_len: usize, compile: F) -> T
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
    if options.len() == 1 {
        options.pop().unwrap()
    } else {
        ast::GrammarExpr::Choice(options)
    }
}

pub(crate) fn sequence_or_single(mut items: Vec<ast::GrammarExpr>) -> ast::GrammarExpr {
    match items.len() {
        0 => ast::GrammarExpr::Sequence(Vec::new()),
        1 => items.pop().unwrap(),
        _ => ast::GrammarExpr::Sequence(items),
    }
}

fn append_end_token_choice(grammar: &mut ast::NamedGrammar, end_token_ids: &[u32]) {
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

    let end = choice_or_single(
        end_token_ids
            .into_iter()
            .map(ast::GrammarExpr::SpecialToken)
            .collect(),
    );
    grammar.rules.push(ast::NamedRule {
        name: generated_start.clone(),
        expr: sequence_or_single(vec![ast::GrammarExpr::Ref(original_start), end]),
        is_terminal: false,
        is_internal: false,
    });
    grammar.start = generated_start;
}

fn emit_import_phase_start(name: &'static str) -> Option<std::time::Instant> {
    if !compile_profile_enabled() {
        return None;
    }

    eprintln!("[glrmask/profile][import-phase-start] name={}", name);
    Some(std::time::Instant::now())
}

fn emit_import_phase_end(name: &'static str, started_at: Option<std::time::Instant>) {
    if let Some(started_at) = started_at {
        eprintln!(
            "[glrmask/profile][import-phase-end] name={} elapsed_ms={:.3}",
            name,
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
}

fn lower_factored_named_grammar(
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

fn lower_json_schema_for_vocab_partition(source: &str) -> crate::Result<GrammarDef> {
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
        other => Err(crate::Error::GrammarParse(format!(
            "unsupported grammar source kind {other:?}"
        ))),
    }
}

fn compile_from_source(
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
        let protected = protected_terminal_names
            .iter()
            .map(|name| (*name).to_string())
            .collect::<Vec<_>>();
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

fn first_external_placeholder_token_id(vocab: &crate::Vocab) -> crate::Result<u32> {
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
        if !reserved.contains(&candidate)
            && !vocab.contains_exact_token_id(candidate)
        {
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

fn dynamic_named_alternatives(
    source: &str,
    parse: NamedGrammarParser,
    transform: Option<NamedGrammarTransform>,
    end_token_ids: &[u32],
) -> crate::Result<Vec<ast::NamedGrammar>> {
    let profile_top = std::env::var_os("GLRMASK_PROFILE_DYNAMIC_TOP").is_some();
    let parse_started = profile_top.then(std::time::Instant::now);
    let named = parse(source)?;
    let parse_ms = parse_started
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
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

fn dynamic_named_alternatives_from_named(
    named: ast::NamedGrammar,
    transform: Option<NamedGrammarTransform>,
    end_token_ids: &[u32],
) -> crate::Result<Vec<ast::NamedGrammar>> {
    let profile_top = std::env::var_os("GLRMASK_PROFILE_DYNAMIC_TOP").is_some();
    let factor_started = profile_top.then(std::time::Instant::now);
    let mut factored = factor_named_grammar(named);
    let factor_ms = factor_started
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let transform_started = profile_top.then(std::time::Instant::now);
    if let Some(transform) = transform {
        transform(&mut factored)?;
    }
    let transform_ms = transform_started
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

    let extract_started = profile_top.then(std::time::Instant::now);

    let start_index = factored
        .rules
        .iter()
        .position(|rule| !rule.is_terminal && rule.name == factored.start);
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
            && let Some(region_rule) = alternative.rules.iter_mut().find(|rule| {
                !rule.is_terminal && rule.name == *region_name
            })
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

fn compile_bounded_template_from_source(
    source: &str, vocab: &crate::Vocab, table_construction: GlrTableConstruction,
    parse: fn(&str) -> crate::Result<ast::NamedGrammar>,
    transform: Option<NamedGrammarTransform>,
) -> crate::Result<DynamicConstraint> {
    with_large_import_stack(source.len(), || {
        let alternatives = dynamic_named_alternatives(source, parse, transform, &[])?;
        let mut compiled = Vec::with_capacity(alternatives.len());
        for alternative in alternatives {
            let grammar = ast::lower(&alternative)?;
            let prepared = crate::compiler::grammar::transforms::prepare_grammar_transforms_only(grammar);
            compiled.push(compile_dynamic_owned_with_vocab_partition_with_table_construction(
                prepared, vocab, table_construction,
            )?);
        }
        let mut constraint = DynamicConstraint::from_alternatives(compiled);
        for component in constraint.constraints_mut() { component.install_template_parser()?; }
        Ok(constraint)
    })
}

fn compile_dynamic_from_named(
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

fn compile_dynamic_from_source(
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
    let import_ms = import_started
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let mut compiled = Vec::with_capacity(alternatives.len());
    let mut lower_ms = 0.0;
    let mut compile_ms = 0.0;
    for alternative in alternatives {
        let lower_started = profile_top.then(std::time::Instant::now);
        let grammar = ast::lower(&alternative)?;
        lower_ms += lower_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let compile_started = profile_top.then(std::time::Instant::now);
        compiled.push(compile_dynamic_owned_with_table_construction(
            grammar,
            vocab,
            default_table_construction,
        )?);
        compile_ms += compile_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    }
    let wrap_started = profile_top.then(std::time::Instant::now);
    let result = DynamicConstraint::from_alternatives(compiled);
    let wrap_ms = wrap_started
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
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

fn compile_dynamic_from_source_with_vocab_partition(
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
        compiled.push(
            compile_dynamic_owned_with_vocab_partition_with_table_construction(
                grammar,
                vocab,
                default_table_construction,
            )?,
        );
    }
    Ok(DynamicConstraint::from_alternatives(compiled))
}

fn compile_dynamic_serialized_from_source_profiled(
    source: &str,
    vocab: &crate::Vocab,
    default_table_construction: GlrTableConstruction,
    parse: NamedGrammarParser,
    transform: Option<NamedGrammarTransform>,
    end_token_ids: &[u32],
    vocab_partition: bool,
) -> crate::Result<(Vec<u8>, u64, u64)> {
    let wall_started = std::time::Instant::now();
    let profile = compile_profile_enabled() || compile_top_profile_enabled();
    let total_started = profile.then(std::time::Instant::now);
    let import_started = profile.then(std::time::Instant::now);
    let alternatives = dynamic_named_alternatives(source, parse, transform, end_token_ids)?;
    let import_ms = import_started
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let compile_started = profile.then(std::time::Instant::now);
    let mut compiled = Vec::with_capacity(alternatives.len());
    let mut lower_ms = 0.0;
    for alternative in alternatives {
        let lower_started = profile.then(std::time::Instant::now);
        let grammar = ast::lower(&alternative)?;
        lower_ms += lower_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let mut constraint = if vocab_partition {
            compile_dynamic_owned_with_vocab_partition_unfinalized_with_table_construction(
                grammar,
                vocab,
                default_table_construction,
            )?
        } else {
            compile_dynamic_owned_unfinalized_with_table_construction(
                grammar,
                vocab,
                default_table_construction,
            )?
        };
        let direct_residual_master_provers_enabled = std::env::var(
            "GLRMASK_DYNAMIC_DIRECT_RESIDUAL_MASTER_PROVERS",
        )
        .ok()
        .map(|value| {
            !matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "" | "0" | "false" | "no" | "off"
            )
        })
        .unwrap_or(true)
            || std::env::var("GLRMASK_EXPERIMENT_DIRECT_RESIDUAL_MASTER_PROVERS")
                .ok()
                .is_some_and(|value| {
                    !matches!(
                        value.trim().to_ascii_lowercase().as_str(),
                        "" | "0" | "false" | "no" | "off"
                    )
                });
        let mut terminal_observation_prepared_in_parallel = false;
        if direct_residual_master_provers_enabled
            && constraint
                .inner
                .tokenizer
                .terminal_residual_coordinates()
                .is_some()
        {
            use crate::compiler::stages::id_map_and_terminal_dwa::classify::VocabPartitionDfa;
            let safe_plus = VocabPartitionDfa::compile_utf8_regex(
                "llg-safe+",
                r#"[^"\\\x00-\x1F\x7F]+"#,
            )
            .expect("safe-string slice regex must compile");
            let whitespace = VocabPartitionDfa::compile_utf8_regex(
                "llg-whitespace",
                r"[\x20\x0A\x0D\x09]+",
            )
            .expect("whitespace slice regex must compile");
            let safe_slice_bytes =
                crate::compiler::constraint_possible_matches::llg_safe_slice_token_bytes_for_vocab(vocab);
            let max_safe_chars =
                crate::compiler::constraint_possible_matches::llg_master_max_safe_chars_for_vocab(vocab);
            let prepare_observation_classes = !constraint
                .inner
                .dynamic_mask_vocab
                .has_terminal_observation_classes()
                && std::env::var("GLRMASK_DYNAMIC_TERMINAL_OBSERVATION_CLASSES")
                    .ok()
                    .is_none_or(|value| {
                        !matches!(value.trim(), "0" | "false" | "no" | "off")
                    });
            let source_state_count = constraint.inner.tokenizer.num_states() as usize;
            let mut dynamic_mask_vocab = std::mem::take(&mut constraint.inner.dynamic_mask_vocab);
            let ((result, proof_elapsed_ms), observation_classes) = rayon::join(
                || {
                    let started = std::time::Instant::now();
                    let result = dynamic_mask_vocab.prepare_master_provers_from_residual_coordinates(
                        &constraint.inner.tokenizer,
                        source_state_count,
                        &safe_plus,
                        &whitespace,
                        safe_slice_bytes,
                        max_safe_chars,
                    );
                    (result, started.elapsed().as_secs_f64() * 1e3)
                },
                || {
                    prepare_observation_classes
                        .then(|| constraint.inner.build_dynamic_terminal_observation_classes())
                },
            );
            if let Some(classes) = observation_classes {
                dynamic_mask_vocab.set_terminal_observation_classes(classes);
                terminal_observation_prepared_in_parallel = true;
            }
            constraint.inner.dynamic_mask_vocab = dynamic_mask_vocab;
            if profile
                || std::env::var_os("GLRMASK_PROFILE_DIRECT_RESIDUAL_MASTER_PROVERS").is_some()
            {
                if let Some((entries, product_pairs, edges)) = result {
                    eprintln!(
                        "[glrmask/profile][direct_residual_master_provers] entries={} product_pairs={} edges={} max_safe_chars={} elapsed_ms={:.3}",
                        entries,
                        product_pairs,
                        edges,
                        max_safe_chars,
                        proof_elapsed_ms,
                    );
                } else {
                    eprintln!(
                        "[glrmask/profile][direct_residual_master_provers] unavailable elapsed_ms={:.3}",
                        proof_elapsed_ms,
                    );
                }
            }
        }
        if !terminal_observation_prepared_in_parallel {
            constraint.inner.prepare_dynamic_terminal_observation_classes_for_artifact();
        }
        constraint.inner.prepare_dynamic_virtual_residual_mask_projections_for_artifact();
        compiled.push(constraint);
    }
    let compile_ms = compile_started
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let compile_ns = wall_started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
    let serialize_wall_started = std::time::Instant::now();
    let serialize_started = profile.then(std::time::Instant::now);
    let bytes = DynamicConstraint::from_alternatives(compiled).into_saved();
    let serialize_ns = serialize_wall_started
        .elapsed()
        .as_nanos()
        .min(u128::from(u64::MAX)) as u64;
    let serialize_ms = serialize_started
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    if let Some(total_started) = total_started {
        eprintln!(
            "[glrmask/profile][dynamic_serialized_source] vocab_partition={} import_ms={:.3} lower_ms={:.3} compile_with_lower_ms={:.3} serialize_ms={:.3} bytes={} total_ms={:.3}",
            vocab_partition,
            import_ms,
            lower_ms,
            compile_ms,
            serialize_ms,
            bytes.len(),
            total_started.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Ok((bytes, compile_ns, serialize_ns))
}

fn compile_dynamic_serialized_from_source(
    source: &str,
    vocab: &crate::Vocab,
    default_table_construction: GlrTableConstruction,
    parse: NamedGrammarParser,
    transform: Option<NamedGrammarTransform>,
    end_token_ids: &[u32],
) -> crate::Result<Vec<u8>> {
    compile_dynamic_serialized_from_source_profiled(
        source,
        vocab,
        default_table_construction,
        parse,
        transform,
        end_token_ids,
        false,
    )
    .map(|(bytes, _, _)| bytes)
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

pub(crate) fn parse_json_schema_to_named(schema_json: &str) -> crate::Result<ast::NamedGrammar> {
    let profile_top = std::env::var_os("GLRMASK_PROFILE_DYNAMIC_TOP").is_some();
    let parse_top_started = profile_top.then(std::time::Instant::now);
    let json_parse_started_at = emit_import_phase_start("serde_json_from_str");
    let schema: serde_json::Value = serde_json::from_str(schema_json)
        .map_err(|e| crate::GlrMaskError::GrammarParse(format!("invalid JSON: {e}")))?;
    emit_import_phase_end("serde_json_from_str", json_parse_started_at);
    let parse_top_ms = parse_top_started
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

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

fn parse_json_schema_to_named_dynamic(schema_json: &str) -> crate::Result<ast::NamedGrammar> {
    let json_parse_started_at = emit_import_phase_start("serde_json_from_str");
    let schema: serde_json::Value = serde_json::from_str(schema_json)
        .map_err(|e| crate::GlrMaskError::GrammarParse(format!("invalid JSON: {e}")))?;
    emit_import_phase_end("serde_json_from_str", json_parse_started_at);

    let schema_to_named_started_at = emit_import_phase_start("schema_to_named_grammar");
    let named = glrmask_json_schema::__private::schema_to_named_grammar_for_runtime_dynamic(&schema, false);
    emit_import_phase_end("schema_to_named_grammar", schema_to_named_started_at);
    Ok(named?)
}

fn parse_json_schema_to_named_dynamic_vocab_partition(
    schema_json: &str,
) -> crate::Result<ast::NamedGrammar> {
    let json_parse_started_at = emit_import_phase_start("serde_json_from_str");
    let schema: serde_json::Value = serde_json::from_str(schema_json)
        .map_err(|e| crate::GlrMaskError::GrammarParse(format!("invalid JSON: {e}")))?;
    emit_import_phase_end("serde_json_from_str", json_parse_started_at);

    let schema_to_named_started_at = emit_import_phase_start("schema_to_named_grammar");
    let named = glrmask_json_schema::__private::schema_to_named_grammar_for_runtime_dynamic(&schema, true);
    emit_import_phase_end("schema_to_named_grammar", schema_to_named_started_at);
    Ok(named?)
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
        Self::from_json_schema_with_dynamic_value_and_end_tokens(
            schema,
            dynamic_value,
            vocab,
            &[],
        )
    }

    /// Compile a JSON Schema with a nested dynamic-value subgrammar and model
    /// end-token IDs.
    #[deprecated(note = "Programmatic JSON schema values are unsupported and not implemented")]
    #[allow(unreachable_code, deprecated)]
    pub(crate) fn from_json_schema_with_dynamic_value_and_end_tokens(
        schema: &str,
        dynamic_value: &Constraint,
        vocab: &crate::Vocab,
        end_token_ids: &[u32],
    ) -> crate::Result<Self> {
        // Retained for reference; this entry point is intentionally unsupported.
        return Err(crate::GlrMaskError::Compilation(
            "programmatic JSON schema values are unsupported (not implemented)".into(),
        ));
        with_large_import_stack(schema.len(), || {
            let placeholder_token_id = external_placeholder_token_id_avoiding(
                vocab,
                end_token_ids.iter().copied().chain(
                    dynamic_value
                        .special_token_terminals
                        .iter()
                        .map(|special| special.token_id),
                ),
            )?;
            let schema_value: serde_json::Value = serde_json::from_str(schema).map_err(|error| {
                crate::GlrMaskError::GrammarParse(format!("invalid JSON: {error}"))
            })?;
            let mut named =
                json_schema::schema_to_named_grammar_with_dynamic_value_token(
                    &schema_value,
                    placeholder_token_id,
                )?;
            prepare_json_schema_named(&mut named)?;
            let parent = compile_from_named_grammar(
                named,
                vocab,
                "json_schema_dynamic_value",
                GlrTableConstruction::LegacyRowBisim,
                end_token_ids,
            )?;
            let placeholder_terminal = parent
                .special_token_terminals
                .iter()
                .find(|special| special.token_id == placeholder_token_id)
                .map(|special| special.terminal_id)
                .ok_or_else(|| {
                    crate::GlrMaskError::Compilation(
                        "dynamic-value JSON Schema lost its linker terminal".to_string(),
                    )
                })?;
            crate::compiler::constraint_compose::compose_constraints_owned_parent(
                parent,
                &[crate::compiler::constraint_compose::CompiledSubgrammarInput {
                    placeholder_terminal,
                    additional_placeholder_terminals: &[],
                    constraint: dynamic_value,
                }],
                vocab,
            )
            .map(|composition| composition.constraint)
            .map_err(crate::GlrMaskError::Compilation)
        })
    }

    /// Compile JSON Schema for programmatic JavaScript object/array values.
    /// Opaque runtime values are accepted at nested value positions, while
    /// conditional expressions keep both result branches recursively constrained
    /// by the same schema.
    #[deprecated(note = "Programmatic JSON schema values are unsupported and not implemented")]
    #[allow(unreachable_code)]
    pub(crate) fn from_json_schema_with_programmatic_values(
        schema: &str,
        dynamic_value: &Constraint,
        condition: &Constraint,
        vocab: &crate::Vocab,
    ) -> crate::Result<Self> {
        // Retained for reference; this entry point is intentionally unsupported.
        return Err(crate::GlrMaskError::Compilation(
            "programmatic JSON schema values are unsupported (not implemented)".into(),
        ));
        with_large_import_stack(schema.len(), || {
            let child_reserved = dynamic_value
                .special_token_terminals
                .iter()
                .chain(condition.special_token_terminals.iter())
                .map(|special| special.token_id)
                .collect::<BTreeSet<_>>();
            let value_token_id = external_placeholder_token_id_avoiding(
                vocab,
                child_reserved.iter().copied(),
            )?;
            let condition_token_id = external_placeholder_token_id_avoiding(
                vocab,
                child_reserved.iter().copied().chain(std::iter::once(value_token_id)),
            )?;
            let schema_value: serde_json::Value = serde_json::from_str(schema).map_err(|error| {
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
            let terminal_for = |token_id: u32| -> crate::Result<u32> {
                parent
                    .special_token_terminals
                    .iter()
                    .find(|special| special.token_id == token_id)
                    .map(|special| special.terminal_id)
                    .ok_or_else(|| crate::GlrMaskError::Compilation(format!(
                        "programmatic JSON Schema lost linker token {token_id}"
                    )))
            };
            let value_terminal = terminal_for(value_token_id)?;
            // Link one child at a time. The generic multi-child linker attempts
            // an expensive structural-sharing quotient when there are multiple
            // children; these two distinct JS children have no sibling regions
            // worth sharing, and sequential exact composition preserves the
            // same language while avoiding that unnecessary pass.
            let with_value = crate::compiler::constraint_compose::compose_constraints_owned_parent_segmented(
                parent,
                &[crate::compiler::constraint_compose::CompiledSubgrammarInput {
                    placeholder_terminal: value_terminal,
                    additional_placeholder_terminals: &[],
                    constraint: dynamic_value,
                }],
                vocab,
                crate::compiler::constraint_compose::SegmentedBoundaryBackend::StaticParserDwa,
            )
            .map(|composition| composition.constraint)
            .map_err(crate::GlrMaskError::Compilation)?;
            let condition_terminal = with_value
                .special_token_terminals
                .iter()
                .find(|special| special.token_id == condition_token_id)
                .map(|special| special.terminal_id)
                .ok_or_else(|| crate::GlrMaskError::Compilation(
                    "programmatic JSON Schema lost its condition linker terminal after value composition".to_string(),
                ))?;
            crate::compiler::constraint_compose::compose_constraints_owned_parent_segmented(
                with_value,
                &[crate::compiler::constraint_compose::CompiledSubgrammarInput {
                    placeholder_terminal: condition_terminal,
                    additional_placeholder_terminals: &[],
                    constraint: condition,
                }],
                vocab,
                crate::compiler::constraint_compose::SegmentedBoundaryBackend::StaticParserDwa,
            )
            .map(|composition| composition.constraint)
            .map_err(crate::GlrMaskError::Compilation)
        })
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
        backend: Option<crate::compiler::constraint_compose::SegmentedBoundaryBackend>,
    ) -> crate::Result<Self> {
        with_large_import_stack(schema.len(), || {
            let child_reserved = dynamic_value
                .special_token_terminals
                .iter()
                .chain(condition.special_token_terminals.iter())
                .map(|special| special.token_id)
                .collect::<BTreeSet<_>>();
            let value_token_id = external_placeholder_token_id_avoiding(
                vocab,
                child_reserved.iter().copied(),
            )?;
            let condition_token_id = external_placeholder_token_id_avoiding(
                vocab,
                child_reserved.iter().copied().chain(std::iter::once(value_token_id)),
            )?;
            let schema_value: serde_json::Value = serde_json::from_str(schema).map_err(|error| {
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
                let input = [crate::compiler::constraint_compose::CompiledSubgrammarInput {
                    placeholder_terminal: placeholder,
                    additional_placeholder_terminals: &[],
                    constraint: child,
                }];
                let composition = match backend {
                    None => crate::compiler::constraint_compose::compose_constraints_owned_parent(
                        parent, &input, vocab,
                    ),
                    Some(backend) => {
                        crate::compiler::constraint_compose::compose_constraints_owned_parent_segmented(
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
                end_token_ids.iter().copied().chain(
                    children
                        .iter()
                        .flat_map(|(_, child)| {
                            child
                                .special_token_terminals
                                .iter()
                                .map(|special| special.token_id)
                        }),
                ),
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
                    unresolved_placeholders.push((
                        placeholder.token_id,
                        placeholder.binding_name.clone(),
                    ));
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
                parent.late_grammar_slots.push(crate::runtime::LateGrammarSlot {
                    name: binding_name,
                    terminal_id,
                });
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
                composition_inputs.push(
                    crate::compiler::constraint_compose::CompiledSubgrammarInput {
                        placeholder_terminal,
                        additional_placeholder_terminals: &[],
                        constraint: child,
                    },
                );
            }
            let parent_late_slots = parent.late_grammar_slots.clone();
            let mut composition = crate::compiler::constraint_compose::compose_constraints_owned_parent_segmented(
                parent,
                &composition_inputs,
                vocab,
                crate::compiler::constraint_compose::SegmentedBoundaryBackend::StaticParserDwa,
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
            if composition
                .constraint
                .sanitize_late_grammar_placeholder_token_domain()
            {
                composition.constraint.rebuild_runtime_caches();
            }
            Ok(composition.constraint)
        })
    }
}

impl DynamicConstraint {
    #[doc(hidden)]
    pub(crate) fn compile_ebnf_serialized_profiled_with_end_tokens(
        ebnf: &str,
        vocab: &crate::Vocab,
        end_token_ids: &[u32],
        vocab_partition: bool,
    ) -> crate::Result<(Vec<u8>, u64, u64)> {
        with_large_import_stack(ebnf.len(), || {
            compile_dynamic_serialized_from_source_profiled(
                ebnf,
                vocab,
                GlrTableConstruction::ExperimentalCoreMerged,
                parse_ebnf_to_named,
                None,
                end_token_ids,
                vocab_partition,
            )
        })
    }

    #[doc(hidden)]
    pub(crate) fn compile_lark_serialized_profiled_with_end_tokens(
        lark: &str,
        vocab: &crate::Vocab,
        end_token_ids: &[u32],
        vocab_partition: bool,
    ) -> crate::Result<(Vec<u8>, u64, u64)> {
        with_large_import_stack(lark.len(), || {
            compile_dynamic_serialized_from_source_profiled(
                lark,
                vocab,
                GlrTableConstruction::ExperimentalCoreMerged,
                parse_lark_to_named,
                None,
                end_token_ids,
                vocab_partition,
            )
        })
    }

    #[doc(hidden)]
    pub(crate) fn compile_json_schema_serialized_profiled_with_end_tokens(
        schema: &str,
        vocab: &crate::Vocab,
        end_token_ids: &[u32],
        vocab_partition: bool,
    ) -> crate::Result<(Vec<u8>, u64, u64)> {
        with_large_import_stack(schema.len(), || {
            compile_dynamic_serialized_from_source_profiled(
                schema,
                vocab,
                GlrTableConstruction::Lalr,
                parse_json_schema_to_named_dynamic,
                Some(prepare_json_schema_named),
                end_token_ids,
                vocab_partition,
            )
        })
    }

    #[doc(hidden)]
    pub(crate) fn compile_glrm_serialized_profiled_with_end_tokens(
        glrm: &str,
        vocab: &crate::Vocab,
        end_token_ids: &[u32],
        vocab_partition: bool,
    ) -> crate::Result<(Vec<u8>, u64, u64)> {
        with_large_import_stack(glrm.len(), || {
            compile_dynamic_serialized_from_source_profiled(
                glrm,
                vocab,
                GlrTableConstruction::ExperimentalCoreMerged,
                parse_glrm_to_named,
                None,
                end_token_ids,
                vocab_partition,
            )
        })
    }

    /// Compile EBNF directly to a serialized dynamic artifact without building
    /// runtime-only caches in the producing process.
    #[doc(hidden)]
    pub(crate) fn compile_ebnf_serialized_with_end_tokens(
        ebnf: &str,
        vocab: &crate::Vocab,
        end_token_ids: &[u32],
    ) -> crate::Result<Vec<u8>> {
        with_large_import_stack(ebnf.len(), || {
            compile_dynamic_serialized_from_source(
                ebnf,
                vocab,
                GlrTableConstruction::ExperimentalCoreMerged,
                parse_ebnf_to_named,
                None,
                end_token_ids,
            )
        })
    }

    /// Compile Lark directly to a serialized dynamic artifact.
    #[doc(hidden)]
    pub(crate) fn compile_lark_serialized_with_end_tokens(
        lark: &str,
        vocab: &crate::Vocab,
        end_token_ids: &[u32],
    ) -> crate::Result<Vec<u8>> {
        with_large_import_stack(lark.len(), || {
            compile_dynamic_serialized_from_source(
                lark,
                vocab,
                GlrTableConstruction::ExperimentalCoreMerged,
                parse_lark_to_named,
                None,
                end_token_ids,
            )
        })
    }

    /// Compile JSON Schema directly to a serialized dynamic artifact.
    #[doc(hidden)]
    pub(crate) fn compile_json_schema_serialized_with_end_tokens(
        schema: &str,
        vocab: &crate::Vocab,
        end_token_ids: &[u32],
    ) -> crate::Result<Vec<u8>> {
        with_large_import_stack(schema.len(), || {
            compile_dynamic_serialized_from_source(
                schema,
                vocab,
                GlrTableConstruction::Lalr,
                parse_json_schema_to_named_dynamic,
                Some(prepare_json_schema_named),
                end_token_ids,
            )
        })
    }

    /// Compile GLRM directly to a serialized dynamic artifact.
    #[doc(hidden)]
    pub(crate) fn compile_glrm_serialized_with_end_tokens(
        glrm: &str,
        vocab: &crate::Vocab,
        end_token_ids: &[u32],
    ) -> crate::Result<Vec<u8>> {
        with_large_import_stack(glrm.len(), || {
            compile_dynamic_serialized_from_source(
                glrm,
                vocab,
                GlrTableConstruction::ExperimentalCoreMerged,
                parse_glrm_to_named,
                None,
                end_token_ids,
            )
        })
    }

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

    /// Compile the same O2 runtime with the bounded grammar normal form
    /// required by an acyclic stack-action backend. This does not build the
    /// static token-mask DWA, and it never approximates a recursive action
    /// relation by a finite-depth unrolling.
    pub(crate) fn from_glrm_with_bounded_template_parser(source: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        compile_bounded_template_from_source(source, vocab, GlrTableConstruction::ExperimentalCoreMerged, parse_glrm_to_named, None)
    }

    pub(crate) fn from_ebnf_with_bounded_template_parser(source: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        compile_bounded_template_from_source(source, vocab, GlrTableConstruction::ExperimentalCoreMerged, parse_ebnf_to_named, None)
    }

    pub(crate) fn from_lark_with_bounded_template_parser(source: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        compile_bounded_template_from_source(source, vocab, GlrTableConstruction::ExperimentalCoreMerged, parse_lark_to_named, None)
    }

    pub(crate) fn from_json_schema_with_bounded_template_parser(source: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        compile_bounded_template_from_source(source, vocab, GlrTableConstruction::Lalr,
            parse_json_schema_to_named_dynamic_vocab_partition, Some(prepare_json_schema_named))
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
                    child
                        .special_token_terminals
                        .iter()
                        .map(|special| special.token_id)
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
                    unresolved_placeholders.push((
                        placeholder.token_id,
                        placeholder.binding_name.clone(),
                    ));
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
                        crate::compiler::constraint_compose::CompiledSubgrammarInput {
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
                        crate::compiler::constraint_compose::compose_constraints_owned_parent_segmented(
                            parent,
                            &composition_inputs,
                            vocab,
                            crate::compiler::constraint_compose::SegmentedBoundaryBackend::Dynamic,
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


#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::glr::table::{AdmissionPolicy, GlrTableConstruction};
    use crate::Vocab;

    #[cfg(windows)]
    #[test]
    fn large_imports_use_dedicated_windows_stack() {
        let thread_name = with_large_import_stack(LARGE_IMPORT_SOURCE_BYTES, || {
            std::thread::current().name().map(str::to_owned)
        });
        assert_eq!(thread_name.as_deref(), Some("glrmask-grammar-compile"));
    }

    fn vocab(entries: &[&str]) -> Vocab {
        Vocab::new(
            entries
                .iter()
                .enumerate()
                .map(|(id, text)| (id as u32, text.as_bytes().to_vec()))
                .collect())
    }

    fn accepts_bytes(constraint: &Constraint, bytes: &[u8]) -> bool {
        let mut state = constraint.start();
        state.commit_bytes(bytes).is_ok() && state.is_accepting()
    }

    #[test]
    #[allow(deprecated)]
    fn programmatic_json_constructors_are_unsupported() {
        let vocab = vocab(&["x"]);
        let dynamic = Constraint::from_glrm_grammar(
            "start dynamic; t IDENT ::= /[A-Za-z_$][A-Za-z0-9_$]*/; nt dynamic ::= IDENT;",
            &vocab,
        )
        .unwrap();
        let condition = dynamic.clone();

        let error = Constraint::from_json_schema_with_programmatic_values(
            "this is not json",
            &dynamic,
            &condition,
            &vocab,
        )
        .err()
        .expect("programmatic JSON schema values must be unsupported");
        match error {
            crate::GlrMaskError::Compilation(message) => assert!(
                message.contains("programmatic JSON schema values are unsupported"),
                "unexpected message: {message}",
            ),
            other => panic!("expected a Compilation error, got {other:?}"),
        }

        let error = Constraint::from_json_schema_with_dynamic_value(
            "this is not json",
            &dynamic,
            &vocab,
        )
        .err()
        .expect("dynamic-value JSON schema must be unsupported");
        match error {
            crate::GlrMaskError::Compilation(message) => assert!(
                message.contains("programmatic JSON schema values are unsupported"),
                "unexpected message: {message}",
            ),
            other => panic!("expected a Compilation error, got {other:?}"),
        }

        let error = Constraint::from_json_schema_with_dynamic_value_and_end_tokens(
            "this is not json",
            &dynamic,
            &vocab,
            &[],
        )
        .err()
        .expect("dynamic-value JSON schema with end tokens must be unsupported");
        match error {
            crate::GlrMaskError::Compilation(message) => assert!(
                message.contains("programmatic JSON schema values are unsupported"),
                "unexpected message: {message}",
            ),
            other => panic!("expected a Compilation error, got {other:?}"),
        }
    }

    #[test]
    fn vocab_partition_json_seeded_ast_lower_matches_ordinary_lower() {
        let schemas = [
            r#"{"type":["null","boolean"]}"#,
            r#"{
                "type":"object",
                "properties":{
                    "code":{
                        "type":"string",
                        "pattern":"^[A-Z0-9]+$",
                        "minLength":2,
                        "maxLength":64
                    }
                },
                "required":["code"],
                "additionalProperties":false
            }"#,
        ];

        for schema in schemas {
            let named = parse_json_schema_to_named_dynamic(schema).unwrap();
            let mut ordinary = factor_named_grammar(named.clone());
            prepare_json_schema_named(&mut ordinary).unwrap();
            let ordinary = ast::lower(&ordinary).unwrap();

            let mut seeded = factor_named_grammar(named);
            let resolved = json_schema::prepare_named_grammar_for_lowering(&mut seeded).unwrap();
            let seeded = ast::lower_with_resolved_terminal_exprs(&seeded, resolved).unwrap();

            assert_eq!(
                bincode::serialize(&ordinary).unwrap(),
                bincode::serialize(&seeded).unwrap(),
                "seeded lowering changed GrammarDef for schema {schema}",
            );
        }
    }

    #[test]
    #[ignore = "Programmatic JSON schema API is intentionally unsupported"]
    #[allow(deprecated)]
    fn json_schema_dynamic_value_is_nested_but_not_root_escape() {
        let vocab = vocab(&["x", "{", "}", "\"name\"", ": ", "123"]);
        let dynamic = Constraint::from_glrm_grammar(
            "start dynamic; t IDENT ::= /[A-Za-z_$][A-Za-z0-9_$]*/; nt dynamic ::= IDENT;",
            &vocab,
        )
        .unwrap();
        let schema = r#"{
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": false
        }"#;
        let constraint =
            Constraint::from_json_schema_with_dynamic_value(schema, &dynamic, &vocab).unwrap();

        assert!(accepts_bytes(&constraint, br#"{"name": "literal"}"#));
        assert!(accepts_bytes(&constraint, br#"{"name": x}"#));
        assert!(!accepts_bytes(&constraint, b"x"));
        assert!(!accepts_bytes(&constraint, br#"{"name": 123}"#));
        assert!(!accepts_bytes(&constraint, br#"{"wrong": x}"#));
    }

    #[test]
    #[ignore = "Programmatic JSON schema API is intentionally unsupported"]
    #[allow(deprecated)]
    fn json_schema_dynamic_value_applies_to_array_items() {
        let vocab = vocab(&["x", "[", "]", ", ", "1"]);
        let dynamic = Constraint::from_glrm_grammar(
            "start dynamic; t IDENT ::= /[A-Za-z_$][A-Za-z0-9_$]*/; nt dynamic ::= IDENT;",
            &vocab,
        )
        .unwrap();
        let schema = r#"{"type":"array","items":{"type":"integer"}}"#;
        let constraint =
            Constraint::from_json_schema_with_dynamic_value(schema, &dynamic, &vocab).unwrap();

        assert!(accepts_bytes(&constraint, b"[1, x]"));
        assert!(!accepts_bytes(&constraint, b"x"));
        assert!(!accepts_bytes(&constraint, br#"["wrong"]"#));
    }

    #[test]
    #[ignore = "Programmatic JSON schema API is intentionally unsupported"]
    #[allow(deprecated)]
    fn json_schema_dynamic_value_allows_runtime_enum_but_rejects_bad_literal() {
        let vocab = vocab(&[
            "{", "}", "\"status\"", ": ", "\"open\"", "\"closed\"", "\"bogus\"",
            "result", ".status",
        ]);
        let dynamic = Constraint::from_glrm_grammar(
            "start expr; nt expr ::= 'result' '.status';",
            &vocab,
        )
        .unwrap();
        let schema = r#"{
          "type":"object",
          "properties":{"status":{"enum":["open","closed"]}},
          "required":["status"],
          "additionalProperties":false
        }"#;
        let constraint = Constraint::from_json_schema_with_dynamic_value(schema, &dynamic, &vocab).unwrap();
        assert!(accepts_bytes(&constraint, br#"{"status": "open"}"#));
        assert!(!accepts_bytes(&constraint, br#"{"status": "bogus"}"#));
        assert!(accepts_bytes(&constraint, br#"{"status": result.status}"#));
    }

    #[test]
    #[ignore = "Programmatic JSON schema API is intentionally unsupported"]
    #[allow(deprecated)]
    fn json_schema_dynamic_value_allows_runtime_const_but_rejects_bad_literal() {
        let vocab = vocab(&[
            "{", "}", "\"kind\"", ": ", "\"fixed\"", "\"wrong\"", "result", ".kind",
        ]);
        let dynamic = Constraint::from_glrm_grammar(
            "start expr; nt expr ::= 'result' '.kind';",
            &vocab,
        )
        .unwrap();
        let schema = r#"{
          "type":"object",
          "properties":{"kind":{"const":"fixed"}},
          "required":["kind"],
          "additionalProperties":false
        }"#;
        let constraint = Constraint::from_json_schema_with_dynamic_value(schema, &dynamic, &vocab).unwrap();
        assert!(accepts_bytes(&constraint, br#"{"kind": "fixed"}"#));
        assert!(!accepts_bytes(&constraint, br#"{"kind": "wrong"}"#));
        assert!(accepts_bytes(&constraint, br#"{"kind": result.kind}"#));
    }

    #[test]
    fn json_schema_import_uses_legacy_row_bisim_table_by_default() {
        let constraint = Constraint::from_json_schema(
            r#"{"type":"string"}"#,
            &vocab(&["\"", "a", "\"a\""]),
        )
        .unwrap();

        assert_eq!(constraint.table.construction, GlrTableConstruction::LegacyRowBisim);
        assert_eq!(constraint.table.admission_policy, AdmissionPolicy::RowPresenceExact);
    }

    fn token_allowed(mask: &[u32], token_id: u32) -> bool {
        mask.get(token_id as usize / 32)
            .is_some_and(|word| word & (1u32 << (token_id % 32)) != 0)
    }

    #[test]
    fn json_schema_end_tokens_are_exact_parser_terminals() {
        let vocab = vocab(&["\"", "a", "\"a\""]);
        let constraint = Constraint::from_json_schema_with_end_tokens(
            r#"{"const":"a"}"#,
            &vocab,
            &[101, 100, 101],
        )
        .unwrap();
        assert_eq!(constraint.mask_len(), 4);

        let mut state = constraint.start();
        assert!(!token_allowed(&state.mask(), 100));
        assert!(!token_allowed(&state.mask(), 101));
        state.commit_token(2).unwrap();
        assert!(!state.is_accepting());
        let mask = state.mask();
        assert!(token_allowed(&mask, 100));
        assert!(token_allowed(&mask, 101));
        assert_eq!(state.forced(), Vec::<u32>::new());
        state.commit_token(100).unwrap();
        assert!(state.is_accepting());
    }

    #[test]
    fn json_schema_single_end_token_is_forced() {
        let vocab = vocab(&["\"", "a", "\"a\""]);
        let constraint = Constraint::from_json_schema_with_end_tokens(
            r#"{"const":"a"}"#,
            &vocab,
            &[100],
        )
        .unwrap();

        let mut state = constraint.start();
        state.commit_token(2).unwrap();
        assert_eq!(state.forced(), vec![100]);
        state.commit_token(100).unwrap();
        assert!(state.is_accepting());
    }

    #[test]
    fn json_schema_end_token_can_also_have_byte_semantics() {
        let vocab = Vocab::new(vec![(100, b"\"a\"".to_vec())]);
        let constraint = Constraint::from_json_schema_with_end_tokens(
            r#"{"const":"a"}"#,
            &vocab,
            &[100],
        )
        .unwrap();

        let mut state = constraint.start();
        assert_eq!(state.forced(), vec![100, 100]);
        state.commit_token(100).unwrap();
        assert!(!state.is_accepting());
        assert_eq!(state.forced(), vec![100]);
        state.commit_token(100).unwrap();
        assert!(state.is_accepting());
    }

    #[test]
    fn caller_sized_masks_zero_unknown_trailing_tokens() {
        let vocab = vocab(&["\"a\""]);
        let constraint = Constraint::from_json_schema_with_end_tokens(
            r#"{"const":"a"}"#,
            &vocab,
            &[100],
        )
        .unwrap();
        let state = constraint.start();
        let mut oversized = vec![u32::MAX; constraint.mask_len() + 3];
        state.fill_mask(&mut oversized);
        assert!(oversized[constraint.mask_len()..].iter().all(|&word| word == 0));
        oversized.fill(u32::MAX);
        state.fill_mask(&mut oversized);
        assert!(oversized[constraint.mask_len()..].iter().all(|&word| word == 0));

        let dynamic = DynamicConstraint::from_json_schema_with_end_tokens(
            r#"{"const":"a"}"#,
            &vocab,
            &[100],
        )
        .unwrap();
        let mut dynamic_mask = vec![u32::MAX; dynamic.mask_len() + 3];
        let dynamic_state = dynamic.start();
        dynamic_state.fill_mask(&mut dynamic_mask);
        assert!(dynamic_mask[dynamic.mask_len()..].iter().all(|&word| word == 0));
        dynamic_mask.fill(u32::MAX);
        dynamic_state.fill_mask(&mut dynamic_mask);
        assert!(dynamic_mask[dynamic.mask_len()..].iter().all(|&word| word == 0));
    }

    #[test]
    fn glrm_import_uses_core_merged_table_by_default() {
        let constraint = Constraint::from_glrm_grammar(
            "start start;\nt A ::= 'a' ;\nnt start ::= A ;\n",
            &vocab(&["a"]),
        )
        .unwrap();

        assert_eq!(
            constraint.table.construction,
            GlrTableConstruction::ExperimentalCoreMerged
        );
        assert_eq!(constraint.table.admission_policy, AdmissionPolicy::ExactSimulation);
    }

    #[test]
    fn glrm_import_merges_partially_mapped_terminal_families() {
        let mut entries = (0u32..=255)
            .map(|byte| (byte, vec![byte as u8]))
            .collect::<Vec<_>>();
        entries.extend([
            (256, b"{\"value\": ".to_vec()),
            (257, b"left".to_vec()),
            (258, b"right".to_vec()),
            (259, b"}".to_vec()),
        ]);
        let vocab = Vocab::new(entries);

        Constraint::from_glrm_grammar(
            r#"
                start document;
                t LEFT ::= @token(1000000);
                t RIGHT ::= @token(1000001);
                nt document ::= "{" "\"value\": " (LEFT | RIGHT) "}";
            "#,
            &vocab,
        )
        .expect("partial terminal-family maps should merge without indexing the unmapped sentinel");
    }

    #[test]
    fn glrm_uniform_subgrammar_ignore_uses_global_terminal_dwa_path() {
        let vocab = vocab(&[" ", "<", ">", "a", "b"]);
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start document;
                ignore OUTER_WS;
                t OUTER_WS ::= " "+;

                g inner ::= {
                    start value;
                    ignore INNER_WS;
                    t INNER_WS ::= " "+;
                    nt value ::= "a" "b";
                };

                nt document ::= "<" inner ">";
            "#,
            &vocab,
        )
        .unwrap();

        assert!(
            constraint.ignore_terminal.is_some(),
            "uniform scoped ignore should remain a global ignore terminal",
        );
        let mut state = constraint.start();
        state.commit_bytes(b"  < a   b >  ").unwrap();
        assert!(state.is_accepting());
    }

    #[test]
    fn glrm_nested_uniform_subgrammar_ignore_uses_global_terminal_dwa_path() {
        let vocab = vocab(&[" ", "<", ">", "[", "]", "x"]);
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start document;
                ignore ROOT_WS;
                t ROOT_WS ::= " "+;

                g middle ::= {
                    start wrapped;
                    ignore MIDDLE_WS;
                    t MIDDLE_WS ::= " "+;

                    g leaf ::= {
                        start value;
                        ignore LEAF_WS;
                        t LEAF_WS ::= " "+;
                        nt value ::= "x";
                    };

                    nt wrapped ::= "[" leaf "]";
                };

                nt document ::= "<" middle ">";
            "#,
            &vocab,
        )
        .unwrap();

        assert!(constraint.ignore_terminal.is_some());
        let mut state = constraint.start();
        state.commit_bytes(b" < [ x ] > ").unwrap();
        assert!(state.is_accepting());
    }

    #[test]
    fn glrm_mixed_subgrammar_ignore_keeps_scoped_grammar_lowering() {
        let vocab = vocab(&[" ", "\t", "<", ">", "a", "b"]);
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start document;
                ignore OUTER_WS;
                t OUTER_WS ::= " "+;

                g inner ::= {
                    start value;
                    ignore INNER_WS;
                    t INNER_WS ::= "\t"+;
                    nt value ::= "a" "b";
                };

                nt document ::= "<" inner ">";
            "#,
            &vocab,
        )
        .unwrap();

        assert!(
            constraint.ignore_terminal.is_none(),
            "different scoped ignores must retain explicit scope-local lowering",
        );
        let mut state = constraint.start();
        state.commit_bytes(b" <\ta\t\tb\t> ").unwrap();
        assert!(state.is_accepting());
    }

    #[test]
    fn glrm_child_without_ignore_keeps_scoped_grammar_lowering() {
        let vocab = vocab(&[" ", "<", ">", "a", "b"]);
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start document;
                ignore OUTER_WS;
                t OUTER_WS ::= " "+;

                g inner ::= {
                    start value;
                    nt value ::= "a" "b";
                };

                nt document ::= "<" inner ">";
            "#,
            &vocab,
        )
        .unwrap();

        assert!(constraint.ignore_terminal.is_none());
        let mut state = constraint.start();
        state.commit_bytes(b" <ab> ").unwrap();
        assert!(state.is_accepting());
    }

    #[test]
    fn glrm_external_subgrammar_api_matches_inline_grammar() {
        let vocab = vocab(&["X", "a", "b", "!", "Xa", "ab!", "Xab!"]);
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child ::= "a" "b";
            "#,
            &vocab,
        )
        .unwrap();
        let composed = Constraint::from_glrm_grammar_with_subgrammars(
            r#"
                start document;
                extern grammar payload;
                nt document ::= "X" payload "!";
            "#,
            &[("payload", &child)],
            &vocab,
        )
        .unwrap();
        let inline = Constraint::from_glrm_grammar(
            r#"
                start document;
                g payload ::= {
                    start child;
                    nt child ::= "a" "b";
                };
                nt document ::= "X" payload "!";
            "#,
            &vocab,
        )
        .unwrap();

        for sequence in [vec![6], vec![0, 5], vec![4, 2, 3]] {
            let mut actual = composed.start();
            let mut expected = inline.start();
            for token_id in sequence {
                assert_eq!(actual.mask(), expected.mask());
                actual.commit_token(token_id).unwrap();
                expected.commit_token(token_id).unwrap();
            }
            assert_eq!(actual.is_accepting(), expected.is_accepting());
            assert!(actual.is_accepting());
        }
    }

    #[test]
    fn glrm_external_subgrammars_support_adjacent_reuse() {
        let vocab = vocab(&["X", "a", "!", "Xa", "aa!"]);
        let child = Constraint::from_glrm_grammar(
            "start child; nt child ::= \"a\";",
            &vocab,
        )
        .unwrap();
        let composed = Constraint::from_glrm_grammar_with_subgrammars(
            r#"
                start document;
                extern grammar left;
                extern grammar right;
                nt document ::= "X" left right "!";
            "#,
            &[("left", &child), ("right", &child)],
            &vocab,
        )
        .unwrap();
        let inline = Constraint::from_glrm_grammar(
            r#"
                start document;
                nt child ::= "a";
                nt document ::= "X" child child "!";
            "#,
            &vocab,
        )
        .unwrap();

        let mut actual = composed.start();
        let mut expected = inline.start();
        for token_id in [3, 1, 2] {
            assert_eq!(actual.mask(), expected.mask());
            actual.commit_token(token_id).unwrap();
            expected.commit_token(token_id).unwrap();
        }
        assert!(actual.is_accepting());
        assert!(expected.is_accepting());
    }

    #[test]
    fn glrm_external_subgrammars_support_qualified_nested_bindings() {
        let vocab = vocab(&["<", ">", "[", "]", "a", "<[a]>"]);
        let leaf = Constraint::from_glrm_grammar(
            "start leaf; nt leaf ::= \"a\";",
            &vocab,
        )
        .unwrap();
        let composed = Constraint::from_glrm_grammar_with_subgrammars(
            r#"
                start document;
                g wrapper ::= {
                    start value;
                    extern grammar leaf;
                    nt value ::= "[" leaf "]";
                };
                nt document ::= "<" wrapper ">";
            "#,
            &[("wrapper::leaf", &leaf)],
            &vocab,
        )
        .unwrap();

        let mut state = composed.start();
        state.commit_token(5).unwrap();
        assert!(state.is_accepting());
    }

    #[test]
    fn glrm_external_subgrammar_api_preserves_scoped_ignores() {
        let vocab = vocab(&[" ", "\t", "<", ">", "a", "b"]);
        let child = Constraint::from_glrm_grammar(
            r#"
                start value;
                ignore CHILD_WS;
                t CHILD_WS ::= "\t"+;
                nt value ::= "a" "b";
            "#,
            &vocab,
        )
        .unwrap();
        let composed = Constraint::from_glrm_grammar_with_subgrammars(
            r#"
                start document;
                ignore PARENT_WS;
                t PARENT_WS ::= " "+;
                extern grammar child;
                nt document ::= "<" child ">";
            "#,
            &[("child", &child)],
            &vocab,
        )
        .unwrap();
        let inline = Constraint::from_glrm_grammar(
            r#"
                start document;
                ignore PARENT_WS;
                t PARENT_WS ::= " "+;
                g child ::= {
                    start value;
                    ignore CHILD_WS;
                    t CHILD_WS ::= "\t"+;
                    nt value ::= "a" "b";
                };
                nt document ::= "<" child ">";
            "#,
            &vocab,
        )
        .unwrap();

        let bytes = b" <\ta\t\tb\t> ";
        let mut actual = composed.start();
        let mut expected = inline.start();
        actual.commit_bytes(bytes).unwrap();
        expected.commit_bytes(bytes).unwrap();
        assert!(actual.is_accepting());
        assert!(expected.is_accepting());
    }

    #[test]
    fn glrm_external_subgrammar_api_retains_missing_slot_and_rejects_invalid_bindings() {
        let vocab = vocab(&["a"]);
        let child = Constraint::from_glrm_grammar(
            "start child; nt child ::= \"a\";",
            &vocab,
        )
        .unwrap();
        let source = "start document; extern grammar child; nt document ::= child;";

        let unresolved = Constraint::from_glrm_grammar_with_subgrammars(source, &[], &vocab)
            .expect("an unresolved extern grammar is a valid late-binding slot");
        assert_eq!(unresolved.late_grammar_slots.len(), 1);
        assert_eq!(unresolved.late_grammar_slots[0].name, "child");
        let late_bound = unresolved
            .bind_grammar("child", child.clone())
            .expect("the retained external slot must remain bindable");
        let mut state = late_bound.start();
        state.commit_token(0).unwrap();
        assert!(state.is_accepting());

        let duplicate = Constraint::from_glrm_grammar_with_subgrammars(
            source,
            &[("child", &child), ("child", &child)],
            &vocab,
        )
        .unwrap_err()
        .to_string();
        assert!(duplicate.contains("more than once"), "{duplicate}");

        let unknown = Constraint::from_glrm_grammar_with_subgrammars(
            source,
            &[("child", &child), ("other", &child)],
            &vocab,
        )
        .unwrap_err()
        .to_string();
        assert!(unknown.contains("unknown external"), "{unknown}");
    }

    #[test]
    fn glrm_external_subgrammar_allocator_avoids_child_special_tokens() {
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"!".to_vec())]);
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                t MARK ::= @token(2);
                nt child ::= "a" MARK;
            "#,
            &vocab,
        )
        .unwrap();
        let composed = Constraint::from_glrm_grammar_with_subgrammars(
            r#"
                start document;
                extern grammar child;
                nt document ::= child "!";
            "#,
            &[("child", &child)],
            &vocab,
        )
        .unwrap();

        let mut state = composed.start();
        state.commit_token(0).unwrap();
        state.commit_token(2).unwrap();
        state.commit_token(1).unwrap();
        assert!(state.is_accepting());
    }

    #[test]
    fn external_placeholder_allocator_finds_hole_below_u32_max() {
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (u32::MAX, b"z".to_vec())]);
        assert_eq!(first_external_placeholder_token_id(&vocab).unwrap(), 1);
    }

    #[test]
    fn typed_glrm_api_without_externals_is_the_normal_compile_path() {
        let vocab = vocab(&["a"]);
        let source = "start document; nt document ::= \"a\";";
        let typed = Constraint::from_glrm_grammar_with_subgrammars(source, &[], &vocab).unwrap();
        let normal = Constraint::from_glrm_grammar(source, &vocab).unwrap();
        assert_eq!(typed.start().mask(), normal.start().mask());
    }

    #[test]
    fn glrm_external_subgrammar_placeholders_avoid_end_tokens() {
        let vocab = Vocab::new(vec![(0, b"a".to_vec())]);
        let child = Constraint::from_glrm_grammar(
            "start child; nt child ::= \"a\";",
            &vocab,
        )
        .unwrap();
        let composed = Constraint::from_glrm_grammar_with_subgrammars_and_end_tokens(
            "start document; extern grammar child; nt document ::= child;",
            &[("child", &child)],
            &vocab,
            &[1],
        )
        .unwrap();
        assert!(
            composed.table.control_terminals.is_empty(),
            "a child followed directly by an end token must compile linker controls away",
        );
        let loaded = Constraint::load(&composed.save()).unwrap();

        for constraint in [&composed, &loaded] {
            let mut state = constraint.start();
            state.commit_token(0).unwrap();
            for _ in 0..2 {
                let mask = state.mask();
                assert_ne!(mask[0] & (1 << 1), 0, "end token missing from cached mask");
            }
            state.commit_token(1).unwrap();
            assert!(state.is_accepting());
        }
    }

    #[test]
    fn ebnf_import_uses_core_merged_table_by_default() {
        let constraint = Constraint::from_ebnf("start ::= 'a'", &vocab(&["a"])).unwrap();

        assert_eq!(
            constraint.table.construction,
            GlrTableConstruction::ExperimentalCoreMerged
        );
        assert_eq!(constraint.table.admission_policy, AdmissionPolicy::ExactSimulation);
    }
}
