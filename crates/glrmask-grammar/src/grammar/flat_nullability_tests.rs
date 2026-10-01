use super::*;

#[test]
fn terminal_byte_nullability_does_not_treat_exact_tokens_as_epsilon() {
    for (terminal, expected) in [
        (Terminal::Literal { id: 0, bytes: vec![] }, true),
        (Terminal::Literal { id: 0, bytes: b"a".to_vec() }, false),
        (Terminal::Pattern { id: 0, pattern: "a?".into(), utf8: true }, true),
        (Terminal::Pattern { id: 0, pattern: "a+".into(), utf8: true }, false),
        (Terminal::Expr { id: 0, expr: Expr::Epsilon }, true),
        (Terminal::Expr { id: 0, expr: Expr::Exclude {
            expr: Box::new(Expr::Epsilon), exclude: Box::new(Expr::Epsilon),
        } }, false),
        (Terminal::SpecialToken { id: 0, token_id: 0 }, false),
    ] {
        assert_eq!(terminal.is_nullable(), expected, "{terminal:?}");
    }
}

#[test]
fn source_nullable_regexes_propagate_through_recursive_sparse_rule_ids() {
    let mut grammar = GrammarDef {
        start: 999,
        rules: vec![
            Rule { lhs: 999, rhs: vec![Symbol::Nonterminal(7), Symbol::Terminal(200)] },
            Rule { lhs: 7, rhs: vec![Symbol::Nonterminal(8)] },
            Rule { lhs: 8, rhs: vec![Symbol::Nonterminal(7)] },
            Rule { lhs: 8, rhs: vec![Symbol::Terminal(100)] },
        ],
        terminals: vec![
            Terminal::Literal { id: 200, bytes: vec![] },
            Terminal::Pattern { id: 100, pattern: "a*".into(), utf8: true },
            Terminal::Literal { id: 0, bytes: b"unused".to_vec() },
        ], ..Default::default()
    };
    assert!(grammar.start_is_nullable());
    grammar.terminals[0] = Terminal::SpecialToken { id: 200, token_id: 0 };
    assert!(!grammar.start_is_nullable());
    grammar.terminals[0] = Terminal::Literal { id: 200, bytes: b"b".to_vec() };
    assert!(!grammar.start_is_nullable());
    grammar.rules[0].rhs.pop();
    assert!(grammar.start_is_nullable());
    grammar.rules.pop(); // A recursive cycle alone does not derive epsilon.
    assert!(!grammar.start_is_nullable());
}

#[test]
fn regular_source_automaton_follows_nullable_terminal_edges_without_expansion() {
    let mut grammar = GrammarDef {
        terminals: vec![
            Terminal::Expr { id: 40, expr: Expr::Epsilon },
            Terminal::Pattern { id: 70, pattern: "(ab|c){0,5000}".into(), utf8: true },
        ],
        direct_regular_automaton: Some(DirectRegularAutomaton { start_states: vec![0], states: vec![
            DirectRegularState { transitions: BTreeMap::from([(40, vec![1])]), ..Default::default() },
            DirectRegularState { transitions: BTreeMap::from([(70, vec![2])]), epsilons: vec![0], ..Default::default() },
            DirectRegularState { is_accepting: true, ..Default::default() },
        ] }), ..Default::default()
    };
    assert!(grammar.start_is_nullable());
    grammar.terminals[1] = Terminal::SpecialToken { id: 70, token_id: 7 };
    assert!(!grammar.start_is_nullable());
    grammar.direct_regular_automaton.as_mut().unwrap().states[1].epsilons.push(2);
    assert!(grammar.start_is_nullable());
}

#[test]
fn parsed_source_and_explicit_optional_terminal_have_equal_body_nullability() {
    use crate::grammar::{ast, glrm};
    for (expression, expected) in [
        ("/a?/", true), ("/a*/ & /a*/", true), ("/a+/ & /a*/", false),
        ("/(ab|c){0,5000}/ & /(a|bc){0,4000}/", true),
    ] {
        let source = format!("start root; t A ::= {expression}; nt root ::= A;");
        let named = glrm::from_glrm(&source).unwrap();
        let flat = ast::lower(&named).unwrap();
        assert_eq!(flat.start_is_nullable(), expected, "{source}");
    }
}
