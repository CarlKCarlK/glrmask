use super::*;
use crate::{BuildOptions, Grammar, Optimization, ParserBackend, Vocab};
use crate::grammar::flat::{DirectRegularState, GrammarDef, Rule, Symbol, Terminal};

fn fixture(nullable: bool, vocab: &Vocab) -> Constraint {
    let mut rules = vec![
        Rule { lhs: 0, rhs: vec![Symbol::Terminal(0)] },
        Rule { lhs: 0, rhs: vec![Symbol::Terminal(1)] },
        Rule { lhs: 0, rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1)] },
    ];
    if nullable { rules.push(Rule { lhs: 0, rhs: vec![] }); }
    let grammar = GrammarDef {
        rules, start: 0,
        terminals: vec![Terminal::Literal { id: 0, bytes: b"a".to_vec() },
                        Terminal::Literal { id: 1, bytes: b"b".to_vec() }],
        direct_regular_automaton: Some(DirectRegularAutomaton {
            start_states: vec![0],
            states: vec![
                DirectRegularState { is_accepting: nullable,
                    transitions: BTreeMap::from([(0, vec![1]), (1, vec![2])]), epsilons: vec![] },
                DirectRegularState { is_accepting: true,
                    transitions: BTreeMap::from([(1, vec![2])]), epsilons: vec![] },
                DirectRegularState { is_accepting: true, ..Default::default() },
            ],
        }), ..Default::default()
    };
    let child = crate::compile_grammar_def_json(&serde_json::to_string(&grammar).unwrap(), vocab).unwrap();
    assert!(child.uses_sparse_direct_regular_runtime(), "fixture must exercise the sparse frontend");
    child
}

fn check(c: &Constraint, vocab: &[(u32, Vec<u8>)], words: &[Vec<u8>]) {
    let prefixes = words.iter().flat_map(|word|
        (0..=word.len()).map(|end| word[..end].to_vec())).collect::<BTreeSet<_>>();
    assert_eq!(c.parser_backend(), ParserBackend::TemplateDfa);
    assert!(!c.table.is_present());
    for prefix in prefixes {
        let mut state = c.start(); state.commit_bytes(&prefix).unwrap();
        assert_eq!(state.is_accepting(), words.contains(&prefix));
        let mask = state.mask();
        for (id, bytes) in vocab {
            let mut extended = prefix.clone(); extended.extend(bytes);
            let expected = words.iter().any(|word| word.starts_with(&extended));
            let actual = mask[*id as usize / 32] & (1 << (*id % 32)) != 0;
            assert_eq!(actual, expected, "prefix={prefix:?} token={bytes:?}");
            if actual {
                let mut branch = state.clone(); branch.commit_token(*id).unwrap();
                assert_eq!(branch.is_accepting(), words.contains(&extended));
            }
        }
    }
}

#[test]
fn sparse_builtin_frames_link_and_return_without_table_reconstruction() {
    if crate::isolate_environment_test(false) { return; }
    let _guard = crate::TEST_ENV_LOCK.lock().unwrap();
    unsafe { std::env::set_var("GLRMASK_STRICT_STATIC_TRAP_DYNAMIC", "1"); }
    let mut tokens = (0..128u32).map(|id| (id, vec![id as u8])).collect::<Vec<_>>();
    tokens.extend(["xay", "xby", "xaby", "xy", "xa", "ay", "ab", "by", "pxayq"]
        .iter().enumerate().map(|(id, s)| (256 + id as u32, s.as_bytes().to_vec())));
    let vocab = Vocab::new(tokens.clone());
    let options = |mode| BuildOptions::default().optimization(mode).parser_backend(ParserBackend::TemplateDfa);
    for nullable in [false, true] {
        let mut child = fixture(nullable, &vocab);
        child.install_template_parser().unwrap();
        let embedding = child.template_parser.as_ref().unwrap().embedding.as_ref().unwrap();
        assert_eq!(embedding.return_pop, 1); assert_eq!(embedding.nullable, nullable);
        let child = Constraint::load(child.save()).unwrap();
        let host = Grammar::from_glrm(r#"glrm 1; start root; extern grammar C; nt root = "x" C "y";"#)
            .compile_unlinked(&vocab).unwrap().bind("C", &child).unwrap();
        let mut words = vec![b"xay".to_vec(), b"xby".to_vec(), b"xaby".to_vec()];
        if nullable { words.push(b"xy".to_vec()); }
        for mode in [Optimization::FastBuild, Optimization::FastRuntime] {
            let linked = host.link_with(options(mode)).unwrap();
            let loaded = Constraint::load(linked.save()).unwrap();
            let external = Constraint::load_with_vocab(linked.save_with_external_vocab().unwrap(), &vocab).unwrap();
            for c in [&linked, &loaded, &external] { check(c, &tokens, &words); }
            let outer = Grammar::from_glrm(r#"glrm 1; start root; extern grammar C; nt root = "p" C "q";"#)
                .compile_unlinked(&vocab).unwrap().bind("C", &loaded).unwrap().link_with(options(mode)).unwrap();
            let outer_words = words.iter().map(|word| {
                let mut out = vec![b'p']; out.extend(word); out.push(b'q'); out
            }).collect::<Vec<_>>();
            check(&outer, &tokens, &outer_words);
        }
    }
    unsafe { std::env::remove_var("GLRMASK_STRICT_STATIC_TRAP_DYNAMIC"); }
}

#[test]
fn malformed_sparse_coordinates_do_not_drop_transitions_or_overflow() {
    for graph in [
        DirectRegularAutomaton { start_states: vec![1], states: vec![Default::default()] },
        DirectRegularAutomaton { start_states: vec![0], states: vec![DirectRegularState {
            transitions: BTreeMap::from([(1, vec![0])]), ..Default::default() }] },
        DirectRegularAutomaton { start_states: vec![0], states: vec![DirectRegularState {
            transitions: BTreeMap::from([(0, vec![u32::MAX])]), ..Default::default() }] },
        DirectRegularAutomaton { start_states: vec![0], states: vec![DirectRegularState {
            epsilons: vec![1], ..Default::default() }] },
    ] { assert!(sparse_regular_templates(&graph, 1).is_err()); }
}
