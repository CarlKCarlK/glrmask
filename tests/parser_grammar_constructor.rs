//! An independent finite-language compiler consumes only public parser data.
//! It knows nothing about GLR tables, lexer expressions, or masking internals.
use std::{cell::Cell, collections::{BTreeMap, BTreeSet}, sync::Arc};
use glrmask::{Constraint, Grammar, Optimization, ParserBackend, Vocab};
use glrmask::template_parser::{CfgRecursion, FlatParserGrammar, ParserDefinition, ParserExpr,
    ParserGrammar, ParserSymbol, PreparedParserGrammar, StackDfa, StackLabel,
    StackState, StackTemplate, StackTransition, TemplateBuildOptions};
use glrmask_grammar::{GrammarExpr as E, NamedGrammar, NamedRule, Quantifier as Q};

type Words = BTreeSet<Vec<u32>>;

fn finite_words(grammar: &FlatParserGrammar, id: u32, active: &mut BTreeSet<u32>,
    cache: &mut BTreeMap<u32, Words>) -> glrmask::Result<Words> {
    if let Some(words) = cache.get(&id) { return Ok(words.clone()); }
    if !active.insert(id) { return Err(glrmask::Error::Compilation("test compiler requires a finite acyclic CFG".into())); }
    let mut words = Words::new();
    for rule in grammar.rules.iter().filter(|rule| rule.lhs == id) {
        let mut prefix = Words::from([vec![]]);
        for symbol in &rule.rhs {
            let suffix = match symbol {
                ParserSymbol::Terminal(t) => Words::from([vec![*t]]),
                ParserSymbol::Nonterminal(n) => finite_words(grammar, *n, active, cache)?,
            };
            let mut joined = Words::new();
            for a in &prefix { for b in &suffix {
                if a.len() + b.len() > 128 || joined.len() > 8192 {
                    return Err(glrmask::Error::Compilation("test finite-language budget exceeded".into()));
                }
                let mut word = a.clone(); word.extend(b); joined.insert(word);
            }}
            prefix = joined;
        }
        words.extend(prefix);
    }
    active.remove(&id); cache.insert(id, words.clone()); Ok(words)
}

fn finite_compiler(grammar: &ParserGrammar) -> glrmask::Result<ParserDefinition> {
    let cfg = grammar.lower_to_cfg()?;
    let words = finite_words(&cfg, cfg.start, &mut BTreeSet::new(), &mut BTreeMap::new())?;
    let mut states = vec![BTreeMap::<u32, u32>::new()];
    let mut accepts = BTreeSet::new();
    for word in words {
        let mut state = 0usize;
        for terminal in word {
            let next = if let Some(&next) = states[state].get(&terminal) { next } else {
                let next = states.len() as u32; states.push(BTreeMap::new());
                states[state].insert(terminal, next); next
            };
            state = next as usize;
        }
        accepts.insert(state as u32);
    }
    let mut terminals = Vec::new();
    for terminal in 0..grammar.terminal_count() {
        if Some(terminal) == grammar.ignore_terminal() { terminals.push(StackTemplate::identity()); continue; }
        let mut template = StackTemplate::reject();
        template.pop_to_push.push(None);
        for (source, row) in states.iter().enumerate() {
            let Some(&target) = row.get(&terminal) else { continue; };
            let popped = template.pop.states.len() as u32;
            template.pop.states.push(StackState::default());
            template.pop.states[0].transitions.push(StackTransition { label: StackLabel::Symbol(source as u32), target: popped });
            let push = template.push.states.len() as u32;
            template.push.states.push(StackState { accepting: false, transitions: vec![StackTransition {
                label: StackLabel::Symbol(target), target: push + 1 }] });
            template.push.states.push(StackState { accepting: true, transitions: vec![] });
            template.pop_to_push.push(Some(push));
        }
        terminals.push(template);
    }
    Ok(ParserDefinition { stack_symbol_count: states.len() as u32, terminals,
        completion: StackTemplate::read_top_and_push(accepts, []) })
}

fn named(rules: Vec<(&str, bool, E)>) -> NamedGrammar {
    NamedGrammar { rules: rules.into_iter().map(|(name, is_terminal, expr)| NamedRule {
        name: name.into(), is_terminal, is_internal: false, expr }).collect(),
        start: "root".into(), ignore: None, lexer_partitions: BTreeMap::new(),
        lexer_literal_partitions: BTreeMap::new(), default_lexer_partition: None }
}
fn literal(bytes: &[u8]) -> E { E::Literal(bytes.to_vec()) }
fn reference(name: &str) -> E { E::Ref(name.into()) }
fn vocabulary() -> (Vocab, Vec<Vec<u8>>) {
    let mut words = (0..128).map(|byte| vec![byte]).collect::<Vec<_>>();
    words.extend([b"ab".as_slice(), b"aab", b"abc", b"a,b", b"b,c", b"\"ab\"", b"\"a\"", b" a", b"a ", b" x"].into_iter().map(Vec::from));
    let mut tokens = words.iter().enumerate().map(|(id, bytes)| (id as u32, bytes.clone())).collect::<Vec<_>>();
    tokens.push((512, vec![])); (Vocab::new(tokens), words)
}
fn bit(mask: &[u32], token: u32) -> bool { mask.get(token as usize / 32).is_some_and(|word| word & (1 << (token % 32)) != 0) }
fn check_words(c: &Constraint, vocab: &[Vec<u8>], language: &[&[u8]]) {
    assert_eq!(c.parser_backend(), ParserBackend::TemplateDfa);
    let mut prefixes = BTreeSet::new();
    for word in language { for length in 0..=word.len() { prefixes.insert(word[..length].to_vec()); } }
    for prefix in prefixes {
        let mut state = c.start(); state.commit_bytes(&prefix).unwrap();
        assert_eq!(state.is_accepting(), language.contains(&prefix.as_slice()), "completion at {prefix:?}");
        let mask = state.mask();
        for (token, bytes) in vocab.iter().enumerate() {
            let mut extension = prefix.clone(); extension.extend(bytes);
            assert_eq!(bit(&mask, token as u32), language.iter().any(|word| word.starts_with(&extension)),
                "prefix={prefix:?} token={bytes:?}");
            if bit(&mask, token as u32) {
                let mut replay = c.start(); replay.commit_bytes(&prefix).unwrap(); replay.commit_token(token as u32).unwrap();
                assert_eq!(replay.is_accepting(), language.contains(&extension.as_slice()));
            }
        }
        assert_eq!(bit(&mask, 512), state.is_accepting());
    }
}

#[test]
fn source_graph_is_resolved_not_flattened_and_callback_is_compile_time_only() {
    let source = named(vec![("root", false, E::Sequence(vec![reference("pair"), E::Quantified(Box::new(reference("pair")), Q::Optional)])),
        ("pair", false, E::Sequence(vec![reference("A"), literal(b"b")])), ("A", true, literal(b"a"))]);
    let prepared = PreparedParserGrammar::from_named(&source).unwrap();
    let grammar = prepared.grammar(); assert_eq!(grammar.rules().len(), 2);
    assert!(matches!(&grammar.rules()[0].expr, ParserExpr::Sequence(parts)
        if matches!(parts.as_slice(), [ParserExpr::Nonterminal(1), ParserExpr::Quantified(_, Q::Optional)])));
    let calls = Cell::new(0);
    let compiler = |g: &ParserGrammar| { calls.set(calls.get() + 1); finite_compiler(g) };
    let parser = prepared.compile_parser(&compiler).unwrap();
    let (v, bytes) = vocabulary();
    for optimization in [Optimization::FastBuild, Optimization::FastRuntime, Optimization::Auto] {
        let c = parser.compile_with(&v, TemplateBuildOptions::default().optimization(optimization).end_tokens([512])).unwrap();
        check_words(&c, &bytes, &[b"ab", b"abab"]);
        let saved = c.save(); let reloaded = Constraint::load(&saved).unwrap();
        check_words(&reloaded, &bytes, &[b"ab", b"abab"]); assert_eq!(reloaded.save(), saved);
        let external = c.save_with_external_vocab().unwrap();
        assert!(Constraint::load(&external).is_err());
        check_words(&Constraint::load_with_vocab(external, &v).unwrap(), &bytes, &[b"ab", b"abab"]);
    }
    assert_eq!(calls.get(), 1);
    fn send_sync<T: Send + Sync>() {} send_sync::<PreparedParserGrammar>();
    send_sync::<glrmask::template_parser::GrammarParserProgram>();
}

#[test]
fn optional_cfg_views_and_analysis_are_lazy_shared_and_shape_explicit() {
    let prepared = Grammar::from_ebnf("root ::= item*\nitem ::= \"a\" \"b\"?").prepare_parser_grammar().unwrap();
    let g = prepared.grammar(); let left = g.lower_to_cfg().unwrap(); let right = g.lower_to_cfg_with(CfgRecursion::Right).unwrap();
    assert!(Arc::ptr_eq(&left, &g.clone().lower_to_cfg().unwrap()));
    assert!(Arc::ptr_eq(&right, &g.lower_to_cfg_with(CfgRecursion::Right).unwrap()));
    assert_ne!(left.rules, right.rules);
    assert!(left.rules.iter().any(|r| matches!(r.rhs.first(), Some(ParserSymbol::Nonterminal(id)) if *id == r.lhs)));
    assert!(right.rules.iter().any(|r| matches!(r.rhs.last(), Some(ParserSymbol::Nonterminal(id)) if *id == r.lhs)));
    let analysis = g.analysis().unwrap(); assert!(Arc::ptr_eq(&analysis, &g.clone().analysis().unwrap()));
    assert!(analysis.nullable[g.start() as usize]); assert!(analysis.follows_eof[g.start() as usize]);
    let item = g.rules().iter().position(|rule| rule.name == "item").unwrap();
    assert!(!analysis.nullable[item]); assert_eq!(analysis.first_terminals[item].len(), 1);
    assert_eq!(analysis.follow_terminals[item], analysis.first_terminals[item]); assert!(analysis.follows_eof[item]);
}

#[test]
fn nullable_lexical_terminals_preserve_epsilon_without_a_nullable_lexer() {
    let source = named(vec![("root", false, E::Sequence(vec![reference("A"), literal(b"b")])),
        ("A", true, E::Quantified(Box::new(literal(b"a")), Q::Optional))]);
    let prepared = PreparedParserGrammar::from_named(&source).unwrap();
    assert!(matches!(&prepared.grammar().rules()[0].expr, ParserExpr::Sequence(parts)
        if matches!(&parts[0], ParserExpr::Choice(choices) if choices.contains(&ParserExpr::Epsilon))));
    let parser = prepared.compile_parser(&finite_compiler).unwrap(); let (v, bytes) = vocabulary();
    for optimization in [Optimization::FastRuntime, Optimization::FastBuild] {
        let c = parser.compile_with(&v, TemplateBuildOptions::default().optimization(optimization).end_tokens([512])).unwrap();
        check_words(&c, &bytes, &[b"b", b"ab"]);
    }
}

#[test]
fn zero_terminal_epsilon_and_empty_languages_are_distinct() {
    let (v, _) = vocabulary();
    for (expression, accepts) in [(E::Epsilon, true), (E::Choice(vec![]), false)] {
        let p = PreparedParserGrammar::from_named(&named(vec![("root", false, expression)])).unwrap();
        assert_eq!(p.grammar().terminal_count(), 0);
        let parser = p.compile_parser(&finite_compiler).unwrap();
        for optimization in [Optimization::FastRuntime, Optimization::FastBuild] {
            let c = parser.compile_with(&v, TemplateBuildOptions::default().optimization(optimization).end_tokens([512])).unwrap();
            for c in [&c, &Constraint::load(c.save()).unwrap()] {
                let state = c.start(); assert_eq!(state.is_accepting(), accepts);
                let mask = state.mask(); assert_eq!(bit(&mask, 512), accepts);
                assert!((0..138).all(|token| !bit(&mask, token)));
            }
        }
    }
}

#[test]
fn separated_sequence_keeps_item_quantifiers_and_required_group_semantics() {
    let source = named(vec![("root", false, E::SeparatedSequence {
        items: vec![(literal(b"a"), Some(Q::Optional)), (literal(b"b"), None), (literal(b"c"), Some(Q::Optional))],
        separator: Box::new(literal(b",")), allow_empty: true })]);
    let prepared = PreparedParserGrammar::from_named(&source).unwrap();
    assert!(matches!(&prepared.grammar().rules()[0].expr, ParserExpr::SeparatedSequence { items, allow_empty: true, .. } if items.len() == 3));
    let left = prepared.grammar().lower_to_cfg().unwrap();
    let right = prepared.grammar().lower_to_cfg_with(CfgRecursion::Right).unwrap();
    assert_eq!(finite_words(&left, left.start, &mut BTreeSet::new(), &mut BTreeMap::new()).unwrap(),
        finite_words(&right, right.start, &mut BTreeSet::new(), &mut BTreeMap::new()).unwrap());
    let parser = prepared.compile_parser(&finite_compiler).unwrap(); let (v, bytes) = vocabulary();
    for optimization in [Optimization::FastRuntime, Optimization::FastBuild] {
        let c = parser.compile_with(&v, TemplateBuildOptions::default().optimization(optimization).end_tokens([512])).unwrap();
        check_words(&c, &bytes, &[b"b", b"a,b", b"b,c", b"a,b,c"]);
    }
}

#[test]
fn all_source_frontends_reach_the_public_constructor() {
    let (v, _) = vocabulary();
    let sources = [(Grammar::from_glrm(r#"start root; nt root ::= "a" "b"?;"#), b"ab".as_slice()),
        (Grammar::from_ebnf(r#"start ::= "a" "b"?"#), b"ab"),
        (Grammar::from_lark(r#"start: "a" "b"?"#), b"ab"),
        (Grammar::from_json_schema(r#"{"enum":["a","ab"]}"#), b"\"ab\"")];
    for (source, word) in sources {
        for optimization in [Optimization::FastBuild, Optimization::FastRuntime] {
            let c = source.compile_with_parser(&v, &finite_compiler, TemplateBuildOptions::default().optimization(optimization)).unwrap();
            let mut state = c.start(); state.commit_bytes(word).unwrap(); assert!(state.is_accepting());
        }
    }
}

#[test]
fn exact_token_binding_is_not_its_vocabulary_byte_spelling() {
    let prepared = PreparedParserGrammar::from_named(&named(vec![("root", false,
        E::Sequence(vec![E::SpecialToken(70), literal(b"b")]))])).unwrap();
    let parser = prepared.compile_parser(&finite_compiler).unwrap();
    let v = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec()), (70, b"a".to_vec())]);
    for optimization in [Optimization::FastRuntime, Optimization::FastBuild] {
        let c = parser.compile_with(&v, TemplateBuildOptions::default().optimization(optimization)).unwrap();
        for c in [&c, &Constraint::load(c.save()).unwrap()] {
            let mut state = c.start(); let mask = state.mask(); assert!(bit(&mask, 70)); assert!(!bit(&mask, 0));
            state.commit_token(70).unwrap(); assert!(!state.is_accepting()); assert!(bit(&state.mask(), 1));
            state.commit_token(1).unwrap(); assert!(state.is_accepting());
            let mut wrong = c.start(); assert!(wrong.commit_token(0).is_err());
        }
    }
}

#[test]
fn malformed_grammar_and_compiler_results_fail_before_generation() {
    for source in [named(vec![("root", false, reference("missing"))]),
        named(vec![("root", false, E::Epsilon), ("root", false, E::Epsilon)]),
        named(vec![("root", false, E::Quantified(Box::new(literal(b"a")), Q::Range(3, Some(1))))])] {
        assert!(PreparedParserGrammar::from_named(&source).is_err());
    }
    let prepared = Grammar::from_ebnf(r#"start ::= "a""#).prepare_parser_grammar().unwrap();
    let wrong_count = |_: &ParserGrammar| Ok(ParserDefinition { stack_symbol_count: 1, terminals: vec![], completion: StackTemplate::identity() });
    assert!(prepared.compile_parser(&wrong_count).is_err());
    let fail = |_: &ParserGrammar| Err(glrmask::Error::Compilation("compiler's own reason".into()));
    assert!(prepared.compile_parser(&fail).unwrap_err().to_string().contains("compiler's own reason"));
    let malformed = |g: &ParserGrammar| {
        let mut definition = finite_compiler(g)?; definition.terminals[0].pop = StackDfa { start: 99, states: vec![] }; Ok(definition)
    };
    assert!(prepared.compile_parser(&malformed).is_err());
}

#[test]
fn expression_automaton_preserves_epsilon_edges_shared_symbols_and_empty_language() {
    use glrmask_finite_automata::unweighted_u32::nfa::{NFA, NFAState};
    use glrmask_grammar::__private::grammar::expr_nfa::ExprNFA;
    let graph = ExprNFA::new(NFA { start_states: vec![0], states: vec![
        NFAState { is_accepting: false, transitions: BTreeMap::new(), epsilons: vec![1] },
        NFAState { is_accepting: false, transitions: BTreeMap::from([(0, vec![2]), (1, vec![2])]), epsilons: vec![] },
        NFAState { is_accepting: true, transitions: BTreeMap::new(), epsilons: vec![] },
    ]}, vec![reference("item"), literal(b"b")]);
    let source = named(vec![("root", false, E::ExprNFA(Box::new(graph.clone()))), ("item", false, literal(b"a"))]);
    let p = PreparedParserGrammar::from_named(&source).unwrap();
    let ParserExpr::Automaton(resolved) = &p.grammar().rules()[0].expr else { panic!("automaton was flattened") };
    assert_eq!(resolved.states.len(), 3); assert_eq!(resolved.states[0].epsilons, vec![1]);
    assert_eq!(resolved.states[1].transitions, vec![(0, 2), (1, 2)]);
    assert_eq!(resolved.symbols[0], ParserExpr::Nonterminal(1));
    let (v, bytes) = vocabulary(); let parser = p.compile_parser(&finite_compiler).unwrap();
    for optimization in [Optimization::FastBuild, Optimization::FastRuntime] {
        let c = parser.compile_with(&v, TemplateBuildOptions::default().optimization(optimization).end_tokens([512])).unwrap();
        check_words(&c, &bytes, &[b"a", b"b"]);
        check_words(&Constraint::load(c.save()).unwrap(), &bytes, &[b"a", b"b"]);
    }
    let mut invalid = graph.clone(); invalid.nfa.states[0].epsilons = vec![99];
    assert!(PreparedParserGrammar::from_named(&named(vec![("root", false, E::ExprNFA(Box::new(invalid)))])).is_err());
    let empty = ExprNFA::new(NFA { start_states: vec![], states: vec![] }, vec![]);
    let p = PreparedParserGrammar::from_named(&named(vec![("root", false, E::ExprNFA(Box::new(empty)))])).unwrap();
    let cfg = p.grammar().lower_to_cfg().unwrap();
    assert!(finite_words(&cfg, cfg.start, &mut BTreeSet::new(), &mut BTreeMap::new()).unwrap().is_empty());
}

#[test]
fn ignores_internal_lexical_helpers_and_partition_conflicts_remain_host_owned() {
    let mut source = named(vec![("root", false, reference("A")), ("A", true, reference("LETTER")),
        ("LETTER", true, literal(b"a")), ("WS", true, E::Quantified(Box::new(literal(b" ")), Q::OnePlus))]);
    source.rules[2].is_internal = true; source.ignore = Some("WS".into());
    let p = PreparedParserGrammar::from_named(&source).unwrap();
    assert_eq!(p.grammar().rules().len(), 1); assert_eq!(p.grammar().terminal_count(), 2);
    let ignore = p.grammar().ignore_terminal().unwrap();
    let malformed = |g: &ParserGrammar| { let mut result = finite_compiler(g)?;
        result.terminals[ignore as usize] = StackTemplate::reject(); Ok(result) };
    assert!(p.compile_parser(&malformed).is_err());
    let parser = p.compile_parser(&finite_compiler).unwrap(); let (v, _) = vocabulary();
    for optimization in [Optimization::FastBuild, Optimization::FastRuntime] {
        let c = parser.compile_with(&v, TemplateBuildOptions::default().optimization(optimization)).unwrap();
        for word in [b"a".as_slice(), b" a", b"a ", b"  a  "] {
            let mut state = c.start(); state.commit_bytes(word).unwrap(); assert!(state.is_accepting(), "{word:?}");
        }
        let mut state = c.start(); state.commit_bytes(b" ").unwrap(); assert!(!state.is_accepting());
        assert!(bit(&state.mask(), b'a' as u32)); assert!(!bit(&state.mask(), b'b' as u32));
    }
    let mut private_reference = source.clone(); private_reference.rules[0].expr = reference("LETTER");
    assert!(PreparedParserGrammar::from_named(&private_reference).is_err());
    let mut collision = named(vec![("root", false, E::Choice(vec![reference("A"), reference("B")])),
        ("A", true, literal(b"a")), ("B", true, literal(b"a"))]);
    collision.lexer_partitions = BTreeMap::from([("A".into(), "one".into()), ("B".into(), "two".into())]);
    assert!(PreparedParserGrammar::from_named(&collision).is_err());
    source.lexer_partitions.insert("missing".into(), "partition".into());
    assert!(PreparedParserGrammar::from_named(&source).is_err());
}

#[test]
fn structural_subtraction_is_not_general_context_free_language_difference() {
    let expression = E::Exclude { expr: Box::new(reference("choice")), exclude: Box::new(reference("left")) };
    let source = named(vec![("root", false, expression), ("choice", false, E::Choice(vec![reference("left"), reference("right")])),
        ("left", false, literal(b"a")), ("right", false, literal(b"b"))]);
    let p = PreparedParserGrammar::from_named(&source).unwrap(); let (v, bytes) = vocabulary();
    let parser = p.compile_parser(&finite_compiler).unwrap();
    for optimization in [Optimization::FastBuild, Optimization::FastRuntime] {
        let c = parser.compile_with(&v, TemplateBuildOptions::default().optimization(optimization).end_tokens([512])).unwrap();
        check_words(&c, &bytes, &[b"b"]);
    }
    let mut empty = source.clone(); empty.rules[0].expr = E::Exclude {
        expr: Box::new(reference("left")), exclude: Box::new(reference("left")) };
    let p = PreparedParserGrammar::from_named(&empty).unwrap();
    assert!(!p.compile_parser(&finite_compiler).unwrap().compile(&v).unwrap().start().is_accepting());
    let mut unsupported = source;
    unsupported.rules[0].expr = E::Intersect { expr: Box::new(reference("choice")), intersect: Box::new(reference("right")) };
    assert!(PreparedParserGrammar::from_named(&unsupported).is_err());
}

#[test]
#[cfg(feature = "internal-api")]
fn required_nullable_separated_items_match_the_existing_grammar_semantics() {
    use glrmask::__private::ConstraintExt;
    use glrmask_grammar::__private::grammar::ast::lower;
    // The legacy grammar-def bridge predates the public empty-token policy.
    // Compare the complete ordinary-token coordinate without an empty alias;
    // end-token behavior is independently covered by the constructor tests.
    let (_, bytes) = vocabulary();
    let v = Vocab::new(bytes.into_iter().enumerate().map(|(id, bytes)| (id as u32, bytes)).collect());
    for allow_empty in [false, true] {
        for optional in [None, Some(Q::Optional)] {
            let source = named(vec![("root", false, E::SeparatedSequence {
                items: vec![(E::Choice(vec![E::Epsilon, literal(b"a")]), optional), (literal(b"b"), None)],
                separator: Box::new(literal(b",")), allow_empty })]);
            let flat = lower(&source).unwrap();
            let reference = Constraint::compile_grammar_def_json(&serde_json::to_string(&flat).unwrap(), &v).unwrap();
            let parser = PreparedParserGrammar::from_named(&source).unwrap().compile_parser(&finite_compiler).unwrap();
            for optimization in [Optimization::FastRuntime, Optimization::FastBuild] {
                let c = parser.compile_with(&v, TemplateBuildOptions::default().optimization(optimization)).unwrap();
                let mut words = vec![Vec::new()];
                for _ in 0..4 { let previous = words.clone(); for word in previous { for byte in b"ab," {
                    let mut next = word.clone(); next.push(*byte); words.push(next);
                }}}
                words.sort(); words.dedup();
                for word in words {
                    let mut a = reference.start(); let mut b = c.start();
                    let accepted_a = a.commit_bytes(&word).is_ok(); let accepted_b = b.commit_bytes(&word).is_ok();
                    assert_eq!(accepted_a, accepted_b, "prefix {word:?}");
                    if accepted_a { assert_eq!(a.is_accepting(), b.is_accepting(), "word {word:?}");
                        assert_eq!(a.mask(), b.mask(), "word={word:?} allow_empty={allow_empty}"); }
                }
            }
        }
    }
}
