use super::*;
use crate::grammar::ast::lower;

#[test]
fn direct_unicode_is_preserved_in_glrm_regexes_and_classes() {
    let tokens = Lexer::new("t RX ::= /é — 😀/; t CLASS ::= [é—😀]/utf8;")
        .tokenize()
        .unwrap();
    assert!(tokens.contains(&Tok::RegexLit("é — 😀".to_string())));
    assert!(tokens.contains(&Tok::CharClass("é—😀".to_string(), true)));

    let grammar = from_glrm("start start; t RX ::= / —/; nt start ::= RX;").unwrap();
    let rx = grammar.rules.iter().find(|rule| rule.name == "RX").unwrap();
    assert_eq!(rx.expr, GrammarExpr::RawRegex(" —".to_string()));

    let grammar = from_glrm("start start; t CLASS ::= [—]/utf8; nt start ::= CLASS;").unwrap();
    let class = grammar.rules.iter().find(|rule| rule.name == "CLASS").unwrap();
    assert_eq!(
        class.expr,
        GrammarExpr::CharClass {
            def: "—".to_string(),
            negate: false,
            utf8: true,
        }
    );
}

use crate::grammar::flat::Symbol;

fn single_path_terminal_names(
    lowered: &crate::grammar::flat::GrammarDef,
    symbol: &Symbol,
) -> Vec<String> {
    match symbol {
        Symbol::Terminal(id) => vec![lowered.terminal_display_name(*id)],
        Symbol::Nonterminal(id) => {
            let rules = lowered
                .rules
                .iter()
                .filter(|rule| rule.lhs == *id)
                .collect::<Vec<_>>();
            assert_eq!(rules.len(), 1, "expected a single-path helper nonterminal");
            rules[0]
                .rhs
                .iter()
                .flat_map(|child| single_path_terminal_names(lowered, child))
                .collect()
        }
    }
}

#[test]
fn parses_named_expr_nfa_definition() {
    let grammar = from_glrm(
        r#"
start obj;

fa obj ::= {
start 0;
accept 4;

0 -- "\"name\": " --> 1;
1 -- "," "\"email\": " --> 2;
1 -- "," "\"description\": " --> 3;
2 -- "," "\"thumbnail\": " --> 3;
2 --> 4;
3 --> 4;
};
"#,
    )
    .unwrap();

    assert_eq!(grammar.rules.len(), 1);
    assert!(matches!(grammar.rules[0].expr, GrammarExpr::ExprNFA(_)));
    lower(&grammar).unwrap();
}

#[test]
fn dumps_expr_nfa_as_own_definition() {
    let grammar = from_glrm(
        r#"
start obj;
fa obj ::= {
start 0;
accept 1;
0 -- "a" --> 1;
};
"#,
    )
    .unwrap();
    let dumped = to_glrm(&grammar);
    assert!(dumped.contains("fa obj ::= {"), "{dumped}");
    assert!(dumped.contains("  start 0;"), "{dumped}");
    assert!(dumped.contains("  accept 1;"), "{dumped}");
    assert!(dumped.contains("  0 -- \"a\" --> 1;"), "{dumped}");
    assert!(!dumped.contains("ExprNFA("), "{dumped}");
}

#[test]
fn special_llm_token_atom_roundtrips() {
    let grammar = from_glrm(
        r#"
                start start;
                t END ::= @token(128009);
                nt start ::= "a" END @token(42);
            "#,
    )
    .unwrap();
    let dumped = to_glrm(&grammar);
    assert!(dumped.contains("@token(128009)"), "{dumped}");
    assert!(dumped.contains("@token(42)"), "{dumped}");
    assert_eq!(from_glrm(&dumped).unwrap().rules, grammar.rules);
}

#[test]
fn expr_nfa_transition_symbols_accept_full_expressions() {
    let grammar = from_glrm(
        r#"
start obj;
fa obj ::= {
start 0;
accept 1;
0 -- [a-z] - "x" --> 1;
};
"#,
    )
    .unwrap();
    let GrammarExpr::ExprNFA(expr_nfa) = &grammar.rules[0].expr else {
        panic!("expected ExprNFA rule");
    };
    assert!(matches!(
        expr_nfa.symbols.first(),
        Some(GrammarExpr::Exclude { .. })
    ));
}

#[test]
fn expr_nfa_transition_symbols_reject_raw_regex_literals() {
    let err = from_glrm(
        r#"
start obj;
fa obj ::= {
start 0;
accept 1;
0 -- /[a-z]+/ --> 1;
};
"#,
    )
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("raw regex literals are only allowed in terminal (`t`) rules"),
        "{err}"
    );
}

#[test]
fn exclude_rhs_sequence_requires_parentheses() {
    let err = from_glrm(
        r#"
start z;
nt A ::= a b | c d | e f;
nt z ::= x (A - c d);
"#,
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("RHS sequence subtraction must be parenthesized"), "{err}");
}

#[test]
fn grouped_exclude_rhs_preserves_parenthesized_ref() {
    let grammar = from_glrm(
        r#"
start z;
nt A ::= a b | c d | e f;
nt B ::= c d | e f;
nt z ::= x (A - B) | x (A - (B));
"#,
    )
    .unwrap();
    let GrammarExpr::Choice(options) = &grammar.rules[2].expr else {
        panic!("expected choice");
    };
    assert!(matches!(
        options[0],
        GrammarExpr::Sequence(_)
    ));
    let GrammarExpr::Sequence(second_parts) = &options[1] else {
        panic!("expected sequence");
    };
    let GrammarExpr::Exclude { exclude, .. } = &second_parts[1] else {
        panic!("expected exclude expr");
    };
    assert!(matches!(exclude.as_ref(), GrammarExpr::Grouped(_)));
}

#[test]
fn lowering_subtracts_exact_nonterminal_alternatives() {
    let grammar = from_glrm(
        r#"
start z;
nt A ::= "a" "b" | "c" "d" | "e" "f";
nt B ::= "c" "d" | "e" "f";
nt z ::= "x" (A - B);
"#,
    )
    .unwrap();

    let lowered = lower(&grammar).unwrap();
    let z_rule = lowered
        .rules
        .iter()
        .find(|rule| rule.lhs == lowered.start)
        .expect("start rule should exist");
    assert_eq!(z_rule.rhs.len(), 2);

    let Symbol::Nonterminal(filtered_nt) = z_rule.rhs[1] else {
        panic!("expected filtered nonterminal");
    };
    let filtered_rules = lowered
        .rules
        .iter()
        .filter(|rule| rule.lhs == filtered_nt)
        .collect::<Vec<_>>();
    assert_eq!(filtered_rules.len(), 1);
    assert_eq!(filtered_rules[0].rhs.len(), 2);

    let filtered_terminals = filtered_rules[0]
        .rhs
        .iter()
        .flat_map(|symbol| single_path_terminal_names(&lowered, symbol))
        .collect::<Vec<_>>();
    assert_eq!(filtered_terminals, vec!["a".to_string(), "b".to_string()]);
}

#[test]
fn lowering_accepts_parenthesized_ref_exact_subtraction() {
    let grammar = from_glrm(
        r#"
start z;
nt A ::= "a" "b" | "c" "d" | "e" "f";
nt B ::= "c" "d" | "e" "f";
nt z ::= "x" (A - (B));
"#,
    )
    .unwrap();

    lower(&grammar).unwrap();
}


#[test]
fn sepseq_rejects_stacked_item_quantifiers() {
    let err = from_glrm(
        r#"
start s;
nt s ::= "," ~+ ( item+? );
nt item ::= "a";
"#,
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("only one postfix quantifier may bind"), "{err}");
}

#[test]
fn sepseq_allows_grouped_inner_quantifier_then_outer_item_quantifier() {
    let grammar = from_glrm(
        r#"
start s;
nt s ::= "," ~+ ( (item+)? );
nt item ::= "a";
"#,
    )
    .unwrap();

    let GrammarExpr::SeparatedSequence { items, .. } = &grammar.rules[0].expr else {
        panic!("expected separated sequence");
    };
    assert_eq!(items.len(), 1);
    assert!(matches!(items[0].1, Some(Quantifier::Optional)));
    assert!(matches!(items[0].0, GrammarExpr::Grouped(_)));
}

#[test]
fn rejects_nested_expr_nfa_at_lowering() {
    let nfa_rule = from_glrm(
        r#"
start inner;
fa inner ::= {
start 0;
accept 1;
0 -- "a" --> 1;
};
"#,
    )
    .unwrap()
    .rules
    .into_iter()
    .next()
    .unwrap();

    let grammar = NamedGrammar {
        rules: vec![NamedRule {
            name: "start".to_string(),
            expr: GrammarExpr::Sequence(vec![nfa_rule.expr, GrammarExpr::Literal(b"b".to_vec())]),
            is_terminal: false,
            is_internal: false,
        }],
        start: "start".to_string(),
        ignore: None,
        lexer_partitions: Default::default(),
        lexer_literal_partitions: Default::default(),
        default_lexer_partition: None,
    };

    let err = lower(&grammar).unwrap_err().to_string();
    assert!(err.contains("complete expression of a named rule"), "{err}");
}

#[test]
fn lexer_groups_accept_anonymous_literals_and_a_catch_all() {
    let grammar = from_glrm(
        r#"
start s;
lexer group punctuation ::= "{";
lexer group literals ::= @literals;
lexer group patterns ::= *;
t WORD ::= /[a-z]+/;
nt s ::= "{" WORD "}";
"#,
    )
    .unwrap();
    assert_eq!(
        grammar
            .lexer_literal_partitions
            .get(b"{".as_slice())
            .map(String::as_str),
        Some("punctuation"),
    );
    assert_eq!(
        grammar
            .lexer_literal_partitions
            .get(b"}".as_slice())
            .map(String::as_str),
        Some("literals"),
    );
    assert_eq!(grammar.default_lexer_partition.as_deref(), Some("patterns"));

    let dumped = to_glrm(&grammar);
    assert!(
        dumped.contains("lexer group punctuation ::= \"{\";"),
        "{dumped}",
    );
    assert!(
        dumped.contains("lexer group literals ::= @literals;"),
        "{dumped}",
    );
    assert!(dumped.contains("lexer group patterns ::= *;"), "{dumped}");
    let reparsed = from_glrm(&dumped).unwrap();
    assert_eq!(
        reparsed.lexer_literal_partitions,
        grammar.lexer_literal_partitions,
    );
    assert_eq!(
        reparsed.default_lexer_partition,
        grammar.default_lexer_partition,
    );

    let lowered = lower(&grammar).unwrap();
    assert_eq!(lowered.lexer_partitions.len(), lowered.terminals.len());
    let punctuation = lowered
        .lexer_partitions
        .values()
        .filter(|partition| partition.as_str() == "punctuation")
        .count();
    let patterns = lowered
        .lexer_partitions
        .values()
        .filter(|partition| partition.as_str() == "patterns")
        .count();
    assert_eq!(punctuation, 1);
    assert_eq!(patterns, 1);
    assert_eq!(
        lowered
            .lexer_partitions
            .values()
            .filter(|partition| partition.as_str() == "literals")
            .count(),
        1,
    );
}

#[test]
fn lexer_group_assignments_follow_deduplicated_terminal_aliases() {
    let grammar = from_glrm(
        r#"
start s;
lexer group same ::= A, B;
t A ::= "a";
t B ::= "a";
nt s ::= A | B;
"#,
    )
    .unwrap();
    let lowered = lower(&grammar).unwrap();
    assert_eq!(lowered.terminals.len(), 1);
    assert_eq!(lowered.lexer_partitions.len(), 1);
    assert_eq!(
        lowered.lexer_partitions.values().next().map(String::as_str),
        Some("same"),
    );
}

#[test]
fn conflicting_groups_for_deduplicated_terminal_aliases_are_rejected() {
    let grammar = from_glrm(
        r#"
start s;
lexer group left ::= A;
lexer group right ::= B;
t A ::= "a";
t B ::= "a";
nt s ::= A | B;
"#,
    )
    .unwrap();
    let error = lower(&grammar).unwrap_err().to_string();
    assert!(error.contains("both lexer groups"), "{error}");
}

#[test]
fn named_grammar_isolate_terminal_is_lowered_to_a_partition() {
    let mut grammar = from_glrm(
        r#"
start s;
t A ::= "a";
t B ::= "b";
nt s ::= A B;
"#,
    )
    .unwrap();
    grammar.isolate_terminal("B");
    let lowered = lower(&grammar).unwrap();
    assert_eq!(lowered.lexer_partitions.len(), 1);
    let b_id = lowered
        .terminal_names
        .iter()
        .find_map(|(&id, name)| (name == "B").then_some(id))
        .unwrap();
    assert_eq!(
        lowered.lexer_partitions.get(&b_id).map(String::as_str),
        Some("__isolated_B"),
    );
}

#[test]
fn named_grammar_isolate_terminal_avoids_existing_partition_name() {
    let mut grammar = from_glrm(
        r#"
start s;
t A ::= "a";
t B ::= "b";
nt s ::= A B;
"#,
    )
    .unwrap();
    grammar.set_lexer_partition("__isolated_B", ["A"]);
    grammar.isolate_terminal("B");
    assert_eq!(
        grammar.lexer_partitions.get("B").map(String::as_str),
        Some("__isolated_B_2"),
    );
}

#[test]
fn lexer_group_rejects_unknown_terminal_at_lowering() {
    let grammar = from_glrm(
        r#"
start s;
lexer group bad ::= MISSING;
t A ::= "a";
nt s ::= A;
"#,
    )
    .unwrap();
    let error = lower(&grammar).unwrap_err().to_string();
    assert!(error.contains("unknown or non-emitting terminal 'MISSING'"), "{error}");
}

#[test]
fn flattened_subgrammar_dump_reparses_to_the_same_flat_grammar() {
    let grammar = from_glrm(
        r#"
start document;
ignore WS;
t WS ::= " "+;

g inner ::= {
    start value;
    ignore NL;
    t NL ::= "\n"+;
    nt value ::= "a" "b";
};

nt document ::= "<" inner ">";
"#,
    )
    .unwrap();
    let dumped = to_glrm(&grammar);
    let reparsed = from_glrm(&dumped).unwrap();
    assert_eq!(
        serde_json::to_value(lower(&grammar).unwrap()).unwrap(),
        serde_json::to_value(lower(&reparsed).unwrap()).unwrap(),
        "dumped flattened grammar:\n{dumped}",
    );
}

#[test]
fn subgrammar_lexer_catch_all_is_scope_local_after_flattening() {
    let grammar = from_glrm(
        r#"
start document;
lexer group outer ::= *;
t OUTER ::= "x";

g inner ::= {
    start value;
    lexer group inner ::= *;
    t INNER ::= "a";
    nt value ::= INNER [bc];
};

nt document ::= OUTER inner;
"#,
    )
    .unwrap();

    assert!(grammar.default_lexer_partition.is_none());
    assert_eq!(grammar.lexer_partitions.get("OUTER").map(String::as_str), Some("outer"));
    for (terminal, partition) in &grammar.lexer_partitions {
        if terminal.starts_with("__glrm_subgrammar_") {
            assert_ne!(partition, "outer", "{terminal} inherited the outer catch-all");
        }
    }
    assert!(
        grammar
            .lexer_partitions
            .iter()
            .any(|(terminal, partition)| terminal.starts_with("__glrm_subgrammar_")
                && partition.contains("_lexer_inner")),
        "expected a private child lexer partition: {:?}",
        grammar.lexer_partitions,
    );
}

#[test]
fn identical_terminals_in_independent_catch_all_scopes_coalesce_partition_groups() {
    let grammar = from_glrm(
        r#"
start document;
lexer group outer ::= *;
t A ::= "a";

g inner ::= {
    start value;
    lexer group inner ::= *;
    t A ::= "a";
    nt value ::= A;
};

nt document ::= A inner;
"#,
    )
    .unwrap();

    let lowered = lower(&grammar).unwrap();
    assert_eq!(lowered.terminals.len(), 1, "identical terminal languages should still deduplicate");
    assert_eq!(lowered.lexer_partitions.len(), 1);
}

#[test]
fn plain_glrm_parse_rejects_unbound_external_subgrammars() {
    let error = from_glrm(
        r#"
start document;
extern grammar payload;
nt document ::= "<" payload ">";
"#,
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("explicit bindings"), "{error}");
}

#[test]
fn external_subgrammar_parse_returns_typed_manifest() {
    let parsed = from_glrm_with_external_subgrammars(
        r#"
start document;
extern grammar payload;
g wrapper ::= {
    start value;
    extern grammar leaf;
    nt value ::= "[" leaf "]";
};
t END ::= @token(52);
nt document ::= "<" payload wrapper ">" END?;
"#,
        52,
        [50, 51],
    )
    .unwrap();

    assert_eq!(
        parsed
            .placeholders
            .iter()
            .map(|placeholder| placeholder.binding_name.as_str())
            .collect::<Vec<_>>(),
        vec!["payload", "wrapper::leaf"],
    );
    assert_eq!(
        parsed
            .placeholders
            .iter()
            .map(|placeholder| placeholder.token_id)
            .collect::<Vec<_>>(),
        vec![53, 54],
    );
    for placeholder in &parsed.placeholders {
        let rules = parsed
            .grammar
            .rules
            .iter()
            .filter(|rule| rule.expr == GrammarExpr::SpecialToken(placeholder.token_id))
            .collect::<Vec<_>>();
        assert_eq!(rules.len(), 1, "external placeholder terminal must be emitted exactly once");
        assert!(rules[0].is_terminal);
    }
    lower(&parsed.grammar).expect("external parent shell must lower normally");
}

#[test]
fn external_subgrammar_placeholder_allocation_does_not_wrap_u32_max() {
    let error = from_glrm_with_external_subgrammars(
        r#"
start document;
extern grammar left;
extern grammar right;
nt document ::= left right;
"#,
        u32::MAX,
        [],
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("another external subgrammar"), "{error}");
}

#[test]
fn glrm_v1_requires_versioned_equals_syntax_and_legacy_stays_legacy() {
    let legacy = from_glrm("start start; nt start ::= 'a';").unwrap();
    assert_eq!(legacy.start, "start");
    assert!(from_glrm("start start; nt start = 'a';").is_err());

    let v1 = from_glrm("glrm 1; start start; nt start = 'a';").unwrap();
    assert_eq!(v1.start, "start");
    let error = from_glrm("glrm 1; start start; nt start ::= 'a';")
        .unwrap_err()
        .to_string();
    assert!(error.contains("require '='"), "{error}");
}

#[test]
fn glrm_v1_quote_styles_have_the_same_literal_semantics() {
    for (single, double) in [
        ("'true'", "\"true\""),
        ("'\"'", "\"\\\"\""),
        ("'\\\''", "\"'\""),
    ] {
        let single = from_glrm(&format!(
            "glrm 1; start start; nt start = {single};"
        ))
        .unwrap();
        let double = from_glrm(&format!(
            "glrm 1; start start; nt start = {double};"
        ))
        .unwrap();
        assert_eq!(single.rules, double.rules);
    }
}

#[test]
fn glrm_v1_formatter_minimizes_literal_quote_escaping() {
    let grammar = from_glrm(
        r#"
glrm 1;
start start;
nt start = '"' | "'" | "plain";
"#,
    )
    .unwrap();
    let dumped = to_glrm_v1(&grammar).unwrap();
    assert!(dumped.contains("'\"'"), "{dumped}");
    assert!(dumped.contains("\"'\""), "{dumped}");
    assert!(dumped.contains("\"plain\""), "{dumped}");
    assert!(!dumped.contains("\\\""), "{dumped}");
    assert_eq!(from_glrm(&dumped).unwrap().rules, grammar.rules);
}

#[test]
fn glrm_v1_rejects_nonregular_and_malformed_regex_syntax() {
    for pattern in [r"(a)\1", r"(?=a)a", r"a(?<=b)", r"\bword", r"a{3,2}", r"(abc", r"[abc"] {
        let source = format!("glrm 1; start start; t RX = /{pattern}/; nt start = RX;");
        assert!(from_glrm(&source).is_err(), "pattern unexpectedly accepted: {pattern}");
    }
    assert!(from_glrm(r#"glrm 1; start start; t RX = /(?:ab|a)+/; nt start = RX;"#).is_ok());
}

#[test]
fn glrm_v1_rejects_descending_repetition_bounds() {
    assert!(from_glrm(r#"glrm 1; start start; nt start = "a"{3,2};"#).is_err());
    assert!(from_glrm(r#"glrm 1; start start; nt item = "a"; nt start = "," ~ ( item{3,2} );"#).is_err());
    assert!(from_glrm(r#"glrm 1; start start; nt start = "a"{2,3};"#).is_ok());
}

#[test]
fn legacy_implicit_epsilon_formats_as_explicit_v1_eps() {
    for source in [
        "start start; nt start ::= ;",
        "start start; nt start ::= 'a' |;",
    ] {
        let grammar = from_glrm(source).unwrap();
        let dumped = to_glrm_v1(&grammar).unwrap();
        assert!(dumped.contains("eps"), "{dumped}");
        from_glrm(&dumped).unwrap();
    }
}

#[test]
fn glrm_v1_rejects_numeric_token_syntax_and_implicit_epsilon() {
    let error = from_glrm("glrm 1; start start; nt start = @token(7);")
        .unwrap_err()
        .to_string();
    assert!(error.contains("does not expose numeric @token"), "{error}");

    assert!(from_glrm("start start; nt start ::= 'a' |;").is_ok());
    assert!(from_glrm("glrm 1; start start; nt start = 'a' |;").is_err());
    assert!(from_glrm("glrm 1; start start; nt start = ();").is_err());
    assert!(from_glrm("glrm 1; start start; nt start = 'a' | eps;").is_ok());
}

#[test]
fn glrm_v1_external_terminals_bind_by_name_and_support_multiple_ids() {
    let ids = [100, 101];
    let grammar = from_glrm_with_external_terminals(
        r#"
glrm 1;
start start;
extern token END_TURN;
nt start = "a" END_TURN;
"#,
        &[("END_TURN", &ids)],
    )
    .unwrap();
    lower(&grammar).unwrap();
    assert!(grammar.rules.iter().any(|rule| {
        rule.name == "END_TURN"
            && matches!(
                &rule.expr,
                GrammarExpr::Choice(items)
                    if items == &vec![GrammarExpr::SpecialToken(100), GrammarExpr::SpecialToken(101)]
            )
    }));
}

#[test]
fn glrm_v1_external_terminal_binding_errors_are_explicit() {
    let source = "glrm 1; start start; extern token X; nt start = X;";
    let missing = from_glrm(source).unwrap_err().to_string();
    assert!(missing.contains("no exact-token binding"), "{missing}");
    let empty = from_glrm_with_external_terminals(source, &[("X", &[])])
        .unwrap_err()
        .to_string();
    assert!(empty.contains("at least one token ID"), "{empty}");
    let unknown_ids = [1];
    let unknown = from_glrm_with_external_terminals(source, &[("Y", &unknown_ids)])
        .unwrap_err()
        .to_string();
    assert!(unknown.contains("unknown external terminal"), "{unknown}");
    let duplicate_ids = [1, 1];
    let duplicate = from_glrm_with_external_terminals(source, &[("X", &duplicate_ids)])
        .unwrap_err()
        .to_string();
    assert!(duplicate.contains("duplicate token ID"), "{duplicate}");
}

#[test]
fn glrm_v1_external_terminal_is_not_visible_inside_terminal_bodies() {
    let ids = [100];
    let error = from_glrm_with_external_terminals(
        "glrm 1; start start; extern token X; t BAD = X; nt start = BAD;",
        &[("X", &ids)],
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("cannot be referenced from terminal body"), "{error}");
}

#[test]
fn glrm_v1_fa_edges_accept_multiline_compound_expressions() {
    let grammar = from_glrm(
        r#"
glrm 1;
start declaration;
t IDENTIFIER = /[A-Za-z_][A-Za-z0-9_]*/;
nt expression = IDENTIFIER;
nt declaration = fa {
    start 0;
    accept 3;
    0 -> 1: "const" | "let" | "var";
    1 -> 2: IDENTIFIER;
    2 -> 3:
        ("=" expression)?
        ";";
};
"#,
    )
    .unwrap();
    let declaration = grammar
        .rules
        .iter()
        .find(|rule| rule.name == "declaration")
        .unwrap();
    assert!(matches!(declaration.expr, GrammarExpr::ExprNFA(_)));
    lower(&grammar).unwrap();
}

#[test]
fn glrm_v1_fa_body_supports_named_states_and_terminal_fas() {
    let grammar = from_glrm(
        r#"
glrm 1;
start start;
t WORD = fa {
    start begin;
    accept done;
    begin -> middle: /a+/;
    middle -> done: "b";
};
nt start = fa {
    start root;
    accept end;
    root -> end: WORD;
};
"#,
    )
    .unwrap();
    let word = grammar.rules.iter().find(|rule| rule.name == "WORD").unwrap();
    assert!(word.is_terminal);
    assert!(matches!(word.expr, GrammarExpr::ExprNFA(_)));
    let start = grammar.rules.iter().find(|rule| rule.name == "start").unwrap();
    assert!(matches!(start.expr, GrammarExpr::ExprNFA(_)));
    lower(&grammar).unwrap();
}

#[test]
fn glrm_v1_source_formatter_preserves_external_terminal_names_and_bindings() {
    let source = r#"
glrm 1;
start message;
pragma glrmask {
    lexer group words = WORD;
}
extern token END_TURN;
g inner = {
    start value;
    extern token BEGIN_ASSISTANT;
    nt value = BEGIN_ASSISTANT "x";
};
t WORD = /[a-z]+/;
nt message = inner WORD END_TURN;
"#;
    let begin_ids = [41];
    let end_ids = [42, 43];
    let bindings = [
        ("inner::BEGIN_ASSISTANT", begin_ids.as_slice()),
        ("END_TURN", end_ids.as_slice()),
    ];
    let before = from_glrm_with_external_terminals(source, &bindings).unwrap();
    let formatted = format_glrm_v1(source).unwrap();
    assert!(formatted.starts_with("glrm 1;\n"), "{formatted}");
    assert!(formatted.contains("extern token END_TURN;"), "{formatted}");
    assert!(formatted.contains("extern token BEGIN_ASSISTANT;"), "{formatted}");
    assert!(formatted.contains("g inner = {"), "{formatted}");
    assert!(formatted.contains("pragma glrmask {"), "{formatted}");
    assert!(!formatted.contains("@token"), "{formatted}");
    assert!(!formatted.contains("41"), "{formatted}");
    assert!(!formatted.contains("42"), "{formatted}");
    assert!(!formatted.contains("43"), "{formatted}");

    let after = from_glrm_with_external_terminals(&formatted, &bindings).unwrap();
    assert_eq!(after.rules, before.rules);
    assert_eq!(after.start, before.start);
    assert_eq!(after.ignore, before.ignore);
    assert_eq!(after.lexer_partitions, before.lexer_partitions);
    assert_eq!(
        after.lexer_literal_partitions,
        before.lexer_literal_partitions
    );
    assert_eq!(after.default_lexer_partition, before.default_lexer_partition);
}

#[test]
fn glrm_v1_source_formatter_preserves_external_subgrammar_declarations() {
    let source = r#"
glrm 1;
start document;
extern grammar payload;
nt document = "<" payload ">";
"#;
    let formatted = format_glrm_v1(source).unwrap();
    assert!(formatted.contains("extern grammar payload;"), "{formatted}");
    let before = from_glrm_with_external_subgrammars(source, 100, []).unwrap();
    let after = from_glrm_with_external_subgrammars(&formatted, 100, []).unwrap();
    assert_eq!(after.grammar.rules, before.grammar.rules);
    assert_eq!(after.grammar.start, before.grammar.start);
    assert_eq!(after.placeholders, before.placeholders);
}

#[test]
fn glrm_v1_source_formatter_requires_version_header() {
    let error = format_glrm_v1("start start; nt start ::= 'x';")
        .unwrap_err()
        .to_string();
    assert!(error.contains("glrm 1;"), "{error}");
}

#[test]
fn glrm_v1_dump_roundtrips_versioned_fa_and_pragma_syntax() {
    let grammar = from_glrm(
        r#"
glrm 1;
start start;
pragma glrmask {
    lexer group words = WORD;
}
t WORD = fa {
    start begin;
    accept done;
    begin -> middle: "a";
    middle -> done: "b";
};
nt start = WORD | "x";
"#,
    )
    .unwrap();
    let dumped = to_glrm_v1(&grammar).unwrap();
    assert!(dumped.starts_with("glrm 1;\n"), "{dumped}");
    assert!(dumped.contains("lexer group words = WORD;"), "{dumped}");
    assert!(dumped.contains("t WORD = fa {"), "{dumped}");
    assert!(dumped.contains("begin -> middle: \"a\";"), "{dumped}");
    let reparsed = from_glrm(&dumped).unwrap();
    assert_eq!(reparsed.rules, grammar.rules);
    assert_eq!(reparsed.start, grammar.start);
    assert_eq!(reparsed.ignore, grammar.ignore);
    assert_eq!(reparsed.lexer_partitions, grammar.lexer_partitions);
    assert_eq!(
        reparsed.lexer_literal_partitions,
        grammar.lexer_literal_partitions
    );
}

#[test]
fn glrm_v1_dump_does_not_leak_numeric_special_token_ids() {
    let legacy = from_glrm("start start; nt start ::= @token(7);").unwrap();
    let error = to_glrm_v1(&legacy).unwrap_err().to_string();
    assert!(error.contains("cannot dump bound exact-token IDs"), "{error}");
}

#[test]
fn legacy_terminal_fa_from_feature_branch_remains_supported() {
    let grammar = from_glrm(
        r#"
start start;
t WORD ::= fa {
    start begin;
    accept done;
    begin -- /a+/ --> middle;
    middle -- "b" --> done;
};
nt start ::= WORD;
"#,
    )
    .unwrap();
    assert!(matches!(
        grammar.rules.iter().find(|rule| rule.name == "WORD").unwrap().expr,
        GrammarExpr::ExprNFA(_)
    ));
    lower(&grammar).unwrap();
    let dumped = to_glrm(&grammar);
    assert!(dumped.contains("t WORD ::= fa {"), "{dumped}");
    from_glrm(&dumped).unwrap();
}

#[test]
fn glrm_v1_namespaces_lexer_groups_under_glrmask_pragma() {
    let grammar = from_glrm(
        r#"
glrm 1;
start start;
pragma glrmask {
    lexer group words = WORD;
}
t WORD = /[a-z]+/;
nt start = WORD;
"#,
    )
    .unwrap();
    assert_eq!(grammar.lexer_partitions.get("WORD").map(String::as_str), Some("words"));
    let dumped = to_glrm_v1(&grammar).unwrap();
    assert!(dumped.starts_with("glrm 1;\n"), "{dumped}");
    assert!(dumped.contains("pragma glrmask {"), "{dumped}");
    assert!(dumped.contains("t WORD = /[a-z]+/;"), "{dumped}");
    from_glrm(&dumped).unwrap();
}
