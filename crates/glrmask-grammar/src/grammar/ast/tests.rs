use super::{
    collect_terminal_rule_refs, grammar_expr_is_nullable, lower, GrammarExpr, Lowerer,
    NamedGrammar, NamedRule, Quantifier,
};
use crate::grammar::expr_nfa::ExprNfaBuilder;
use rustc_hash::{FxHashMap, FxHashSet};

fn nonterminal(name: &str, expr: GrammarExpr) -> NamedRule {
    NamedRule {
        name: name.to_string(),
        expr,
        is_terminal: false,
        is_internal: false,
    }
}

fn terminal(name: &str, expr: GrammarExpr) -> NamedRule {
    NamedRule {
        name: name.to_string(),
        expr,
        is_terminal: true,
        is_internal: false,
    }
}

fn literal(text: &str) -> GrammarExpr {
    GrammarExpr::Literal(text.as_bytes().to_vec())
}

fn subtract(lhs: &str, exclude: GrammarExpr) -> GrammarExpr {
    GrammarExpr::Exclude {
        expr: Box::new(GrammarExpr::Ref(lhs.to_string())),
        exclude: Box::new(exclude),
    }
}

#[test]
fn expr_nfa_nullability_uses_epsilon_and_nullable_label_reachability() {
    let mut builder = ExprNfaBuilder::new();
    let nullable_state = builder.add_state();
    let accept = builder.add_state();
    builder.add_epsilon(builder.start_state(), nullable_state);
    // Include a cycle to verify the reachability walk terminates.
    builder.add_epsilon(nullable_state, builder.start_state());
    builder.add_transition(
        nullable_state,
        GrammarExpr::Ref("EMPTY".to_string()),
        accept,
    );
    builder.add_transition(
        builder.start_state(),
        literal("x"),
        accept,
    );
    builder.set_accepting(accept);
    let expr = GrammarExpr::ExprNFA(Box::new(builder.build()));

    let mut nullable = FxHashMap::default();
    assert!(!grammar_expr_is_nullable(&expr, &nullable));
    nullable.insert("EMPTY".to_string(), true);
    assert!(grammar_expr_is_nullable(&expr, &nullable));
}

#[test]
fn terminal_reference_collection_walks_nested_expression_forms() {
    let expr = GrammarExpr::Exclude {
        expr: Box::new(GrammarExpr::Choice(vec![
            GrammarExpr::Ref("first".to_string()),
            GrammarExpr::Grouped(Box::new(GrammarExpr::Ref("second".to_string()))),
        ])),
        exclude: Box::new(GrammarExpr::SeparatedSequence {
            items: vec![(GrammarExpr::Ref("third".to_string()), None)],
            separator: Box::new(GrammarExpr::Ref("separator".to_string())),
            allow_empty: false,
        }),
    };
    let mut refs = FxHashSet::default();
    collect_terminal_rule_refs(&expr, &mut refs);

    assert_eq!(refs.len(), 4);
    assert!(refs.contains("first"));
    assert!(refs.contains("second"));
    assert!(refs.contains("third"));
    assert!(refs.contains("separator"));
}

#[test]
fn exact_subtraction_matches_nonterminal_alias_body() {
    let grammar = NamedGrammar {
        rules: vec![
            terminal("JSON_STRING_BODY", literal("body\"")),
            nonterminal(
                "json_string",
                GrammarExpr::Sequence(vec![
                    literal("\""),
                    GrammarExpr::Ref("JSON_STRING_BODY".to_string()),
                ]),
            ),
            nonterminal(
                "json_value",
                GrammarExpr::Choice(vec![
                    GrammarExpr::Ref("json_string".to_string()),
                    literal("0"),
                ]),
            ),
            nonterminal(
                "start",
                subtract("json_value", GrammarExpr::Ref("json_string".to_string())),
            ),
        ],
        start: "start".to_string(),
        ignore: None,
        lexer_partitions: Default::default(),
        lexer_literal_partitions: Default::default(),
        default_lexer_partition: None,
    };

    lower(&grammar).unwrap();
}

#[test]
fn exact_subtraction_canonicalization_is_cycle_safe() {
    let grammar = NamedGrammar {
        rules: vec![
            nonterminal(
                "loop",
                GrammarExpr::Sequence(vec![
                    GrammarExpr::Ref("loop".to_string()),
                    literal("y"),
                ]),
            ),
            nonterminal(
                "A",
                GrammarExpr::Choice(vec![
                    GrammarExpr::Ref("loop".to_string()),
                    literal("x"),
                ]),
            ),
            nonterminal("start", subtract("A", literal("z"))),
        ],
        start: "start".to_string(),
        ignore: None,
        lexer_partitions: Default::default(),
        lexer_literal_partitions: Default::default(),
        default_lexer_partition: None,
    };

    lower(&grammar).unwrap();
}

fn repeated_compound_label_expr_nfa(canonical: bool) -> GrammarExpr {
    let mut builder = ExprNfaBuilder::new();
    let middle = builder.add_state();
    let accept = builder.add_state();
    let label = GrammarExpr::Sequence(vec![
        literal("a"),
        GrammarExpr::Ref("item".to_string()),
    ]);
    builder.add_transition(builder.start_state(), label.clone(), middle);
    builder.add_transition(middle, label, accept);
    builder.set_accepting(accept);
    let expr_nfa = builder.build();
    GrammarExpr::ExprNFA(Box::new(if canonical {
        expr_nfa.into_determinized_and_minimized()
    } else {
        expr_nfa
    }))
}

fn assert_repeated_compound_label_is_shared(canonical: bool) {
    let grammar = NamedGrammar {
        rules: vec![
            nonterminal("item", literal("b")),
            nonterminal("start", repeated_compound_label_expr_nfa(canonical)),
        ],
        start: "start".to_string(),
        ignore: None,
        lexer_partitions: Default::default(),
        lexer_literal_partitions: Default::default(),
        default_lexer_partition: None,
    };
    let gdef = lower(&grammar).unwrap();
    let a_terminal = gdef
        .terminals
        .iter()
        .position(|terminal| terminal.name() == "a")
        .expect("literal a terminal") as u32;
    let wrapper_rules = gdef
        .rules
        .iter()
        .filter(|rule| {
            matches!(
                rule.rhs.first(),
                Some(crate::grammar::flat::Symbol::Terminal(tid)) if *tid == a_terminal
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        wrapper_rules.len(),
        1,
        "one immutable wrapper should serve every edge carrying the same compound label"
    );
}

#[test]
fn expr_nfa_dfa_edges_reuse_compound_labels() {
    assert_repeated_compound_label_is_shared(false);
}

#[test]
fn canonical_expr_nfa_dfa_edges_reuse_compound_labels() {
    assert_repeated_compound_label_is_shared(true);
}

#[test]
fn literal_terminal_cache_preserves_lossy_name_updates() {
    let mut lowerer = Lowerer::new();
    let first_bytes = [0xff];
    let second_bytes = [0xfe];
    let lossy_name = String::from_utf8_lossy(&first_bytes).into_owned();
    assert_eq!(lossy_name, String::from_utf8_lossy(&second_bytes));

    let first_id = lowerer.literal_terminal_id(&first_bytes);
    let second_id = lowerer.literal_terminal_id(&second_bytes);
    assert_ne!(first_id, second_id);
    assert_eq!(lowerer.terminal_ids_by_name.get(&lossy_name), Some(&second_id));

    let repeated_first_id = lowerer.literal_terminal_id(&first_bytes);
    assert_eq!(repeated_first_id, first_id);
    assert_eq!(lowerer.terminal_ids_by_name.get(&lossy_name), Some(&first_id));
    assert_eq!(lowerer.terminals.len(), 2);
}

#[test]
fn lower_deduplicates_identical_rules() {
    let grammar = NamedGrammar {
        rules: vec![nonterminal(
            "start",
            GrammarExpr::Choice(vec![literal("a"), literal("a")]),
        )],
        start: "start".to_string(),
        ignore: None,
        lexer_partitions: Default::default(),
        lexer_literal_partitions: Default::default(),
        default_lexer_partition: None,
    };

    let gdef = lower(&grammar).unwrap();
    let start_rules = gdef
        .rules
        .iter()
        .filter(|rule| rule.lhs == gdef.start)
        .count();
    assert_eq!(start_rules, 1, "duplicate alternatives should not create duplicate rules");
}

#[test]
fn nonnullable_sequence_with_nonnullable_part_reduces_rules() {
    let grammar = NamedGrammar {
        rules: vec![
            nonterminal(
                "body",
                GrammarExpr::Choice(vec![
                    literal("abc"),
                    GrammarExpr::Epsilon,
                ]),
            ),
            nonterminal(
                "item",
                GrammarExpr::Quantified(Box::new(GrammarExpr::Sequence(vec![
                    literal("{"),
                    GrammarExpr::Ref("body".to_string()),
                    literal("}"),
                ])), Quantifier::OnePlus),
            ),
            nonterminal("start", GrammarExpr::Ref("item".to_string())),
        ],
        start: "start".to_string(),
        ignore: None,
        lexer_partitions: Default::default(),
        lexer_literal_partitions: Default::default(),
        default_lexer_partition: None,
    };

    let gdef = lower(&grammar).unwrap();
    let brace_rules_count = gdef
        .rules
        .iter()
        .filter(|rule| {
            matches!(
                rule.rhs.first(),
                Some(crate::grammar::flat::Symbol::Terminal(tid))
                    if gdef.terminal_display_name(*tid) == "{"
            )
        })
        .count();
    assert_eq!(
        brace_rules_count,
        1,
        "nonnullable sequence should not synthesize duplicate brace-start alternatives"
    );
}
