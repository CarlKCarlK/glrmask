//! Focused crate-internal witnesses for the ordinary retained
//! `ParserBackend::LrTable` backend. These assert the *construction* shape that
//! the public enum alone cannot prove: a retained LR table with no template
//! parser, no template DFAs and no direct-regular runtime bypass.

use super::ParserTableStorage;
use crate::public_api::{BuildOptions, Optimization, ParserBackend};
use crate::Grammar;

fn vocab_from(words: &[&str]) -> crate::Vocab {
    crate::Vocab::new(
        words
            .iter()
            .enumerate()
            .map(|(index, word)| (index as u32, word.as_bytes().to_vec()))
            .collect(),
    )
}

fn lr(vocab: &crate::Vocab, source: &str) -> crate::Constraint {
    crate::DynamicConstraint::from_ebnf(source, vocab)
        .expect("ordinary Dynamic compile").into_constraint()
}

fn native(vocab: &crate::Vocab, source: &str) -> crate::Constraint {
    Grammar::from_ebnf(source)
        .compile_with(
            vocab,
            BuildOptions::default().optimization(Optimization::FastBuild),
        )
        .expect("native template compile")
}

fn assert_no_template_artifacts(constraint: &crate::Constraint) {
    assert!(!constraint.has_template_parser(), "LR must not install a template parser");
    assert!(constraint.template_parser.is_none());
    assert!(constraint.template_dfas_by_terminal.is_empty(), "no template DFA inventory may be built");
    assert!(constraint.fast_template_dfas_by_terminal.is_empty());
    assert!(constraint.direct_regular_automaton.is_none(), "no direct-regular runtime bypass");
    assert_eq!(constraint.parser_backend(), ParserBackend::LrTable);
}

#[test]
fn explicit_lr_constructs_a_retained_table_and_no_template() {
    let vocab = vocab_from(&["a", "b", "ab", "c"]);
    let constraint = lr(&vocab, r#"start ::= "a" start? "b""#);
    assert_no_template_artifacts(&constraint);
    let table = constraint
        .table
        .as_lr()
        .expect("explicit LR must retain an executable table");
    assert!(table.num_states > 0);
    assert!(!table.action.is_empty(), "nontrivial grammar must retain real LR rows");
}

#[test]
fn explicit_lr_witness_holds_for_more_than_sixty_four_rules() {
    // Make the grammar eligible for the default early-overlap schedule, and
    // prove lowering did not collapse the >64-rule witness to an automaton.
    let mut source = String::from("start ::=");
    let mut words = Vec::new();
    for index in 0..70 {
        source.push_str(&format!(" r{index}"));
    }
    source.push('\n');
    for index in 0..70 {
        let word = format!("w{index}");
        words.push(word.clone());
        source.push_str(&format!("r{index} ::= \"{word}\"\n"));
    }
    let word_refs = words.iter().map(String::as_str).collect::<Vec<_>>();
    let vocab = vocab_from(&word_refs);
    let lowered = crate::grammar::ast::lower(&crate::import::parse_ebnf_to_named(&source).unwrap()).unwrap();
    assert!(lowered.rules.len() > 64);
    assert!(lowered.direct_regular_automaton.is_none());
    let constraint = lr(&vocab, &source);
    assert_no_template_artifacts(&constraint);
    assert!(constraint.table.as_lr().is_some());
}

#[test]
fn explicit_lr_direct_dynamic_frontend_uses_a_real_table() {
    // A right-linear/JSON frontend can retain its parser language as a direct
    // automaton. Explicit LR must still build a genuine LR table and must not
    // keep the direct-regular runtime bypass coordinate.
    let vocab = crate::Vocab::new(vec![
        (0, b"\"a\"".to_vec()),
        (1, b"\"ab\"".to_vec()),
    ]);
    let constraint = crate::DynamicConstraint::from_json_schema(r#"{"enum":["a","ab"]}"#, &vocab)
        .expect("ordinary Dynamic JSON compile").into_constraint();
    let table = constraint
        .table
        .as_lr()
        .expect("explicit LR must retain an executable table for the direct frontend");
    assert!(!table.action.is_empty(), "direct frontend must retain real LR action rows");
    assert_no_template_artifacts(&constraint);
}

#[test]
fn o2_is_unchanged_table_free() {
    let vocab = vocab_from(&["a", "b", "ab", "c"]);
    let constraint = native(&vocab, r#"start ::= "a" start? "b""#);
    assert!(constraint.has_template_parser());
    assert!(!constraint.table.is_present(), "native template must not retain a table");
    assert_eq!(constraint.parser_backend(), ParserBackend::TemplateDfa);
}

#[test]
fn absent_storage_still_panics_loudly() {
    let storage = ParserTableStorage::absent();
    assert!(!storage.is_present());
    assert!(storage.as_lr().is_none());
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| storage.into_lr())).is_err());
}
