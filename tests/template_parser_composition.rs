use std::collections::BTreeSet;
use glrmask::{BuildOptions, Constraint, Grammar, Optimization, ParserBackend, Vocab};

const HOST: &str = r#"glrm 1; start root; extern grammar child; nt root = "x" child "y";"#;

#[test]
fn nullable_lexical_source_is_preserved_by_every_compiled_child_backend() {
    let (vocab, tokens) = vocabulary();
    let sources = [
        Grammar::from_glrm("start root; t A ::= /a?/; nt root ::= A;"),
        Grammar::from_glrm("start root; t A ::= /a?/ & /a*/; nt root ::= A;"),
        // EBNF has no slash-delimited regex syntax; its explicit optional
        // literal is the independent parser-nullability control case.
        Grammar::from_ebnf(r#"start ::= "a"?"#),
        Grammar::from_lark("start: A\nA: /a?/"),
    ];
    let words: &[&[u8]] = &[b"xy", b"xay"];
    for (source_index, source) in sources.iter().enumerate() {
        for backend in [ParserBackend::TemplateDfa] {
            for child_mode in [Optimization::FastBuild, Optimization::FastRuntime] {
                let child = source.compile_with(&vocab, BuildOptions::default()
                    .optimization(child_mode).parser_backend(backend)).unwrap();
                let saved_child = Constraint::load(child.save()).unwrap();
                for child in [&child, &saved_child] {
                    let bound = Grammar::from_glrm(HOST).compile_unlinked(&vocab).unwrap()
                        .bind("child", child).unwrap();
                    for link_mode in [Optimization::FastBuild, Optimization::FastRuntime] {
                        let linked = bound.link_with(BuildOptions::default()
                            .optimization(link_mode).parser_backend(backend)).unwrap();
                        let loaded = Constraint::load(linked.save()).unwrap();
                        let external = if backend == ParserBackend::TemplateDfa {
                            Some(Constraint::load_with_vocab(
                                linked.save_with_external_vocab().unwrap(), &vocab).unwrap())
                        } else {
                            // External-vocabulary Constraint envelopes are a
                            // template-only API. Keep the LR rejection contract
                            // and compare its fresh/self-contained forms.
                            assert!(linked.save_with_external_vocab().is_err());
                            None
                        };
                        for c in [&linked, &loaded].into_iter().chain(external.as_ref()) {
                            for prefix in [b"".as_slice(), b"x", b"xy", b"xa", b"xay"] {
                                let mut state = c.start(); state.commit_bytes(prefix).unwrap();
                                assert_eq!(state.is_accepting(), words.contains(&prefix));
                                let mask = state.mask();
                                for (id, bytes) in &tokens {
                                    let mut word = prefix.to_vec(); word.extend(bytes);
                                    let expected = words.iter().any(|candidate| candidate.starts_with(&word));
                                    let actual = mask.get(*id as usize / 32)
                                        .is_some_and(|bits| bits & (1 << (id % 32)) != 0);
                                    assert_eq!(actual, expected, "source={source_index} backend={backend:?} \
                                        child={child_mode:?} link={link_mode:?} prefix={prefix:?} token={bytes:?}");
                                    if actual {
                                        let mut branch = state.clone(); branch.commit_token(*id).unwrap();
                                        assert_eq!(branch.is_accepting(), words.contains(&word.as_slice()));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn sparse_regular_precompiled_children_keep_their_one_symbol_return_frame() {
    use glrmask::__private::{ConstraintExt, into_template_parser, parser_backend_report};
    use glrmask_grammar::__private::grammar::flat::{DirectRegularAutomaton, DirectRegularState,
        GrammarDef, Rule, Symbol, Terminal};
    use std::collections::BTreeMap;
    // The outer p C q wrapper must have actual crossing model tokens; without
    // them it correctly needs no B shard at that level. Keep both entering and
    // leaving crossings, and a whole token crossing both interfaces.
    let (_, mut tokens) = vocabulary();
    tokens.extend([(4000, b"px".to_vec()), (4001, b"yq".to_vec()),
        (4002, b"pxayq".to_vec()), (4003, b"pxyq".to_vec())]);
    let vocab = Vocab::new(tokens.clone());
    for nullable in [false, true] {
        let mut rules = vec![Rule { lhs: 0, rhs: vec![Symbol::Terminal(0)] },
            Rule { lhs: 0, rhs: vec![Symbol::Terminal(1)] },
            Rule { lhs: 0, rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1)] }];
        if nullable { rules.push(Rule { lhs: 0, rhs: vec![] }); }
        let grammar = GrammarDef { rules, start: 0,
            terminals: vec![Terminal::Literal { id: 0, bytes: b"a".to_vec() },
                Terminal::Literal { id: 1, bytes: b"b".to_vec() }],
            direct_regular_automaton: Some(DirectRegularAutomaton { start_states: vec![0], states: vec![
                DirectRegularState { is_accepting: nullable,
                    transitions: BTreeMap::from([(0, vec![1]), (1, vec![2])]), epsilons: vec![] },
                DirectRegularState { is_accepting: true,
                    transitions: BTreeMap::from([(1, vec![2])]), epsilons: vec![] },
                DirectRegularState { is_accepting: true, ..Default::default() },
            ] }), ..Default::default() };
        let source = Constraint::compile_grammar_def_json(&serde_json::to_string(&grammar).unwrap(), &vocab).unwrap();
        assert_eq!(parser_backend_report(&source)["sparse_regular"], true,
            "this test must exercise the sparse regular frontend, not an ordinary LR row");
        let words: &[&[u8]] = if nullable { &[b"xy", b"xay", b"xby", b"xaby"] }
            else { &[b"xay", b"xby", b"xaby"] };
        // A raw sparse child must select the native template linker as well;
        // the LR splicer requires augmented rules which this frontend omits.
        let legacy = Grammar::from_glrm(HOST).compile_unlinked(&vocab).unwrap().bind("child", &source).unwrap();
        for mode in [Optimization::FastBuild, Optimization::FastRuntime] {
            let linked = legacy.link_with(BuildOptions::default().optimization(mode)
                .parser_backend(ParserBackend::TemplateDfa)).unwrap();
            assert_language(&linked, &tokens, words);
            assert_language(&Constraint::load(linked.save()).unwrap(), &tokens, words);
        }
        let child = into_template_parser(source).unwrap();
        assert_eq!(parser_backend_report(&child)["finite_embedding"], true);
        let child = Constraint::load(child.save()).unwrap();
        let host = Grammar::from_glrm(HOST).compile_unlinked(&vocab).unwrap().bind("child", &child).unwrap();
        for mode in [Optimization::FastBuild, Optimization::FastRuntime] {
            let c = host.link_with(BuildOptions::default().optimization(mode)
                .parser_backend(ParserBackend::TemplateDfa)).unwrap();
            let loaded = Constraint::load(c.save()).unwrap();
            let external = Constraint::load_with_vocab(c.save_with_external_vocab().unwrap(), &vocab).unwrap();
            for c in [&c, &loaded, &external] { assert_language(c, &tokens, words); }
            let outer = Grammar::from_glrm(r#"glrm 1; start root; extern grammar C; nt root = "p" C "q";"#)
                .compile_unlinked(&vocab).unwrap().bind("C", &loaded).unwrap()
                .link_with(BuildOptions::default().optimization(mode).parser_backend(ParserBackend::TemplateDfa)).unwrap();
            let nested_words = words.iter().map(|word| {
                let mut out = vec![b'p']; out.extend_from_slice(word); out.push(b'q'); out
            }).collect::<Vec<_>>();
            let nested_words = nested_words.iter().map(Vec::as_slice).collect::<Vec<_>>();
            assert_language(&outer, &tokens, &nested_words);
            assert_language(&Constraint::load(outer.save()).unwrap(), &tokens, &nested_words);
            if mode == Optimization::FastRuntime { assert_static_boundaries(&outer); }
        }
    }
}

#[test]
fn static_composition_with_no_crossing_tokens_needs_no_boundary_shards() {
    const CHILD: &str = "GLRMASK_TEMPLATE_STATIC_EMPTY_BOUNDARY_TEST";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .arg("--exact").arg("static_composition_with_no_crossing_tokens_needs_no_boundary_shards")
            .arg("--nocapture").arg("--test-threads=1")
            .env(CHILD, "1").env("GLRMASK_STRICT_STATIC_TRAP_DYNAMIC", "1")
            .output().unwrap();
        assert!(output.status.success(), "{}\n{}", String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr));
        return;
    }
    let tokens = b"xaypq".iter().enumerate().map(|(id, byte)| (id as u32, vec![*byte]))
        .collect::<Vec<_>>();
    let vocab = Vocab::new(tokens.clone());
    let options = || BuildOptions::default().optimization(Optimization::FastRuntime)
        .parser_backend(ParserBackend::TemplateDfa);
    let leaf = Grammar::from_ebnf(r#"start ::= "a""#).compile_with(&vocab, options()).unwrap();
    let middle = Grammar::from_glrm(HOST).compile_unlinked(&vocab).unwrap()
        .bind("child", &leaf).unwrap().link_with(options()).unwrap();
    let outer = Grammar::from_glrm(r#"glrm 1; start root; extern grammar C; nt root = "p" C "q";"#)
        .compile_unlinked(&vocab).unwrap().bind("C", &middle).unwrap().link_with(options()).unwrap();
    fn check_empty_shards(report: &serde_json::Value) {
        assert_eq!(report["lr_table_present"], false);
        if let Some(children) = report["component_parsers"].as_array() {
            assert_eq!(report["static_boundary_shards"], 0, "{report}");
            assert_eq!(report["dynamic_boundary_shards"], 0, "{report}");
            assert_eq!(report["packed_lr_compiler_table_present"], false);
            for child in children { check_empty_shards(child); }
        }
    }
    for c in [&outer, &Constraint::load(outer.save()).unwrap(),
        &Constraint::load_with_vocab(outer.save_with_external_vocab().unwrap(), &vocab).unwrap()] {
        check_empty_shards(&glrmask::__private::parser_backend_report(c));
        assert_language(c, &tokens, &[b"pxayq"]);
    }
}

fn assert_static_boundaries(constraint: &Constraint) {
    fn check(report: &serde_json::Value) {
        assert_eq!(report["lr_table_present"], false);
        if let Some(children) = report["component_parsers"].as_array() {
            assert_eq!(report["dynamic_boundary_shards"], 0, "{report}");
            assert_eq!(report["packed_lr_compiler_table_present"], false);
            for child in children { check(child); }
        }
    }
    let report = glrmask::__private::parser_backend_report(constraint);
    assert!(report["static_boundary_shards"].as_u64().unwrap_or(0) > 0, "{report}");
    check(&report);
}

#[test]
fn strict_static_crossings_preserve_delayed_lexemes_sparse_ids_and_aliases() {
    const CHILD: &str = "GLRMASK_TEMPLATE_STATIC_EXCLUSIONS_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .arg("--exact").arg("strict_static_crossings_preserve_delayed_lexemes_sparse_ids_and_aliases")
            .arg("--nocapture").arg("--test-threads=1")
            .env(CHILD, "1").env("GLRMASK_STRICT_STATIC_TRAP_DYNAMIC", "1")
            .output().unwrap();
        assert!(output.status.success(), "{}\n{}", String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr));
        return;
    }
    let (_, mut tokens) = vocabulary();
    tokens.extend([(1000, b"xaby".to_vec()), (1001, b"xaby".to_vec()), (1033, b"xabby".to_vec()),
        (2048, b"abby".to_vec()), (2051, b"aay".to_vec()), (2052, b"xaaay".to_vec()),
        (2060, b"aq".to_vec()), (3000, b"ayq".to_vec())]);
    let vocab = Vocab::new(tokens.clone());
    let cases: &[(&str, &[&[u8]])] = &[
        (r#"start root; t A ::= "a" | "ab"; nt root ::= A "b"?;"#,
            &[b"xay", b"xaby", b"xabby"]),
        (r#"start root; t A ::= "a" | "aa"; nt root ::= A "a"?;"#,
            &[b"xay", b"xaay", b"xaaay"]),
    ];
    let options = || BuildOptions::default().optimization(Optimization::FastRuntime).parser_backend(ParserBackend::TemplateDfa);
    for (source, words) in cases {
        let child = Grammar::from_glrm(source).compile_with(&vocab, options()).unwrap();
        let bound = Grammar::from_glrm(HOST).compile_unlinked(&vocab).unwrap().bind("child", &child).unwrap();
        let candidate = bound.link_with(options()).unwrap();
        let loaded = Constraint::load(candidate.save()).unwrap();
        let external = Constraint::load_with_vocab(candidate.save_with_external_vocab().unwrap(), &vocab).unwrap();
        for candidate in [&candidate, &loaded, &external] {
            assert_static_boundaries(candidate);
            assert_language(candidate, &tokens, words);
            assert!(!candidate.start().mask().get(999 / 32).is_some_and(|word| word & (1 << (999 % 32)) != 0));
        }
    }
}

#[test]
fn strict_static_table_free_nested_composition_uses_no_dynamic_boundary() {
    const CHILD: &str = "GLRMASK_TEMPLATE_STATIC_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .arg("--exact").arg("strict_static_table_free_nested_composition_uses_no_dynamic_boundary")
            .arg("--nocapture").arg("--test-threads=1")
            .env(CHILD, "1").env("GLRMASK_STRICT_STATIC_TRAP_DYNAMIC", "1")
            .output().unwrap();
        assert!(output.status.success(), "{}\n{}", String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr));
        return;
    }
    let (vocab, tokens) = vocabulary();
    let build = |optimization| BuildOptions::default().optimization(optimization).parser_backend(ParserBackend::TemplateDfa);
    let leaf = Grammar::from_ebnf(r#"start ::= "a" | "b""#).compile_with(&vocab, build(Optimization::FastRuntime)).unwrap();
    // The nested component starts with a dynamic boundary. The outer static
    // construction must materialize that boundary too, not hide a fallback.
    let middle = Grammar::from_glrm(r#"glrm 1; start mid; extern grammar leaf; nt mid = "p" leaf "q";"#)
        .compile_unlinked(&vocab).unwrap().bind("leaf", &leaf).unwrap().link_with(build(Optimization::FastBuild)).unwrap();
    let middle = Constraint::load(middle.save()).unwrap();
    let outer = Grammar::from_glrm(r#"glrm 1; start root; extern grammar middle; nt root = "x" middle middle "y";"#)
        .compile_unlinked(&vocab).unwrap().bind("middle", &middle).unwrap().link_with(build(Optimization::FastRuntime)).unwrap();
    let words: &[&[u8]] = &[b"xpaqpaqy", b"xpaqpbqy", b"xpbqpaqy", b"xpbqpbqy"];
    let loaded = Constraint::load(outer.save()).unwrap();
    let external = Constraint::load_with_vocab(outer.save_with_external_vocab().unwrap(), &vocab).unwrap();
    for candidate in [&outer, &loaded, &external] {
        assert_static_boundaries(candidate);
        assert_language(candidate, &tokens, words);
    }
    // The user's reusable dynamic child is immutable and remains reusable.
    let original = glrmask::__private::parser_backend_report(&middle);
    assert!(original["dynamic_boundary_shards"].as_u64().unwrap_or(0) > 0);
}

#[test]
fn strict_static_nullable_controls_have_no_unfolding_depth_limit() {
    const CHILD: &str = "GLRMASK_TEMPLATE_NULLABLE_STATIC_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .arg("--exact").arg("strict_static_nullable_controls_have_no_unfolding_depth_limit")
            .arg("--nocapture").arg("--test-threads=1")
            .env(CHILD, "1").env("GLRMASK_STRICT_STATIC_TRAP_DYNAMIC", "1")
            .output().unwrap();
        assert!(output.status.success(), "{}\n{}", String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr));
        return;
    }
    let (vocab, tokens) = vocabulary();
    let options = || BuildOptions::default().optimization(Optimization::FastRuntime).parser_backend(ParserBackend::TemplateDfa);
    let child = Grammar::from_ebnf(r#"start ::= "a"?"#).compile_with(&vocab, options()).unwrap();
    // A single gap can require 128 CALL/RETURN events despite link depth 1.
    let parent = Grammar::from_glrm(r#"glrm 1; start root; extern grammar child; nt root = "x" child{64} "y";"#)
        .compile_unlinked(&vocab).unwrap();
    let c = parent.bind("child", &child).unwrap().link_with(options()).unwrap();
    let words = (0..=64).map(|count| format!("x{}y", "a".repeat(count)).into_bytes()).collect::<Vec<_>>();
    let words = words.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let loaded = Constraint::load(c.save()).unwrap();
    let external = Constraint::load_with_vocab(c.save_with_external_vocab().unwrap(), &vocab).unwrap();
    for c in [&c, &loaded, &external] { assert_static_boundaries(c); assert_language(c, &tokens, &words); }

    // Nullable repetition has no finite control-word enumeration bound.
    // Its language is x a* y; check against a direct byte-language predicate.
    let parent = Grammar::from_glrm(r#"glrm 1; start root; extern grammar child; nt root = "x" child* "y";"#)
        .compile_unlinked(&vocab).unwrap();
    let c = parent.bind("child", &child).unwrap().link_with(options()).unwrap();
    let loaded = Constraint::load(c.save()).unwrap();
    fn viable(word: &[u8]) -> bool {
        if word.is_empty() { return true; }
        if word[0] != b'x' { return false; }
        let tail = word[1..].strip_suffix(b"y").unwrap_or(&word[1..]);
        tail.iter().all(|&byte| byte == b'a')
    }
    for c in [&c, &loaded] {
        assert_static_boundaries(c);
        for count in [0usize, 1, 2, 7, 31, 65, 128] {
            let prefix = format!("x{}", "a".repeat(count)).into_bytes();
            let mut state = c.start(); state.commit_bytes(&prefix).unwrap(); assert!(!state.is_accepting());
            let mask = state.mask();
            for (id, bytes) in &tokens {
                let mut candidate = prefix.clone(); candidate.extend(bytes);
                let actual = mask.get(*id as usize / 32).is_some_and(|word| word & (1 << (*id % 32)) != 0);
                assert_eq!(actual, viable(&candidate), "prefix_len={} token={bytes:?}", prefix.len());
            }
            state.commit_bytes(b"y").unwrap(); assert!(state.is_accepting());
        }
    }
}

#[test]
fn nullable_nested_table_free_children_keep_every_repetition_count() {
    let (vocab, tokens) = vocabulary();
    let options = || BuildOptions::default().optimization(Optimization::FastBuild).parser_backend(ParserBackend::TemplateDfa);
    let leaf = Grammar::from_ebnf(r#"start ::= "a"?"#).compile_with(&vocab, options()).unwrap();
    let middle = Grammar::from_glrm(r#"glrm 1; start mid; extern grammar leaf; nt mid = leaf leaf;"#)
        .compile_unlinked(&vocab).unwrap().bind("leaf", &leaf).unwrap().link_with(options()).unwrap();
    for middle in [&middle, &Constraint::load(middle.save()).unwrap()] {
        let outer = Grammar::from_glrm(r#"glrm 1; start root; extern grammar middle; nt root = "x" middle middle "y";"#)
            .compile_unlinked(&vocab).unwrap().bind("middle", middle).unwrap().link_with(options()).unwrap();
        let words: &[&[u8]] = &[b"xy", b"xay", b"xaay", b"xaaay", b"xaaaay"];
        assert_language(&outer, &tokens, words);
        assert_language(&Constraint::load(outer.save()).unwrap(), &tokens, words);
    }
}

#[test]
fn scoped_ignores_match_the_independent_lr_composition() {
    let (vocab, _) = vocabulary();
    let parent = Grammar::from_glrm(r#"glrm 1; start root; ignore WS; t WS = "~"+; extern grammar child; nt root = "x" child "y";"#)
        .compile_unlinked(&vocab).unwrap();
    let grammar = Grammar::from_glrm(r#"start root; ignore WS; t WS ::= "_"+; nt root ::= "a" "b"?;"#);
    let lr_child = grammar.compile(&vocab).unwrap();
    let lr = parent.bind("child", &lr_child).unwrap().link_with(BuildOptions::default().optimization(Optimization::FastBuild)).unwrap();
    let child = grammar.compile_with(&vocab, BuildOptions::default().parser_backend(ParserBackend::TemplateDfa)).unwrap();
    let candidate = parent.bind("child", &child).unwrap().link_with(BuildOptions::default()
        .optimization(Optimization::FastBuild).parser_backend(ParserBackend::TemplateDfa)).unwrap();
    let loaded = Constraint::load(candidate.save()).unwrap();
    let external = Constraint::load_with_vocab(candidate.save_with_external_vocab().unwrap(), &vocab).unwrap();
    let mut prefixes = vec![vec![]]; let mut layer = vec![vec![]];
    for _ in 0..4 {
        let mut next = Vec::new();
        for word in layer { for byte in b"xayb~_" { let mut extended = word.clone(); extended.push(*byte); next.push(extended); } }
        prefixes.extend(next.iter().cloned()); layer = next;
    }
    prefixes.extend([b"~x_a_by~".to_vec(), b"x_a~by".to_vec(), b"x~a_by".to_vec(), b"x__a__b__y".to_vec()]);
    for prefix in prefixes {
        let mut reference = lr.start(); let valid = reference.commit_bytes(&prefix).is_ok();
        for constraint in [&candidate, &loaded, &external] {
            let mut state = constraint.start();
            assert_eq!(state.commit_bytes(&prefix).is_ok(), valid, "ignore prefix {prefix:?}");
            if valid {
                assert_eq!(state.is_accepting(), reference.is_accepting(), "ignore completion {prefix:?}");
                assert_eq!(state.mask(), reference.mask(), "ignore mask {prefix:?}");
            }
        }
    }
}

#[test]
fn table_free_links_preserve_exact_empty_tokens_and_root_end_policy() {
    let (_, mut tokens) = vocabulary();
    tokens.extend([(600, b"special".to_vec()), (601, vec![]), (602, b"special".to_vec()), (700, vec![])]);
    let vocab = Vocab::new(tokens);
    let child = Grammar::from_glrm(r#"start root; t SPECIAL ::= @token(600); t EMPTY ::= @token(601); nt root ::= SPECIAL "a" | EMPTY "b";"#)
        .compile_with(&vocab, BuildOptions::default().parser_backend(ParserBackend::TemplateDfa)).unwrap();
    let parent = Grammar::from_glrm(HOST).compile_unlinked(&vocab).unwrap();
    let result = parent.bind("child", &child).unwrap().link_with(BuildOptions::default()
        .optimization(Optimization::FastBuild).parser_backend(ParserBackend::TemplateDfa).end_tokens([700])).unwrap();
    let loaded = Constraint::load(result.save()).unwrap();
    for c in [&result, &loaded] {
        for (special, letter) in [(600, b'a'), (601, b'b')] {
            let mut state = c.start(); state.commit_bytes(b"x").unwrap();
            let allowed = |mask: &[u32], token: u32| mask[token as usize / 32] & (1 << (token % 32)) != 0;
            let mask = state.mask(); assert!(allowed(&mask, special)); assert!(!allowed(&mask, 602)); assert!(!allowed(&mask, 700));
            state.commit_token(special).unwrap(); state.commit_bytes(&[letter, b'y']).unwrap(); assert!(state.is_accepting());
            assert!(allowed(&state.mask(), 700)); state.commit_token(700).unwrap(); assert!(state.is_terminated());
        }
    }
}

#[test]
fn already_table_free_children_link_without_reconstructing_tables() {
    let (vocab, tokens) = vocabulary();
    for child_mode in [Optimization::FastRuntime, Optimization::FastBuild] {
        for nullable in [false, true] {
            let source = if nullable { r#"start ::= "a"?"# } else { r#"start ::= "a" | "b" | "a" "b""# };
            let child = Grammar::from_ebnf(source).compile_with(&vocab, BuildOptions::default()
                .optimization(child_mode).parser_backend(ParserBackend::TemplateDfa)).unwrap();
            let loaded = Constraint::load(child.save()).unwrap();
            for child in [&child, &loaded] {
                assert_eq!(glrmask::__private::parser_backend_report(child)["finite_embedding"], true);
                let bound = Grammar::from_glrm(HOST).compile_unlinked(&vocab).unwrap().bind("child", child).unwrap();
                for mode in [Optimization::FastBuild, Optimization::Auto, Optimization::FastRuntime] {
                    let result = bound.link_with(BuildOptions::default().optimization(mode)
                        .parser_backend(ParserBackend::TemplateDfa)).unwrap();
                    let words: &[&[u8]] = if nullable { &[b"xy", b"xay"] } else { &[b"xay", b"xby", b"xaby"] };
                    assert_language(&result, &tokens, words);
                    assert_language(&Constraint::load(result.save()).unwrap(), &tokens, words);
                }
            }
        }
    }
}

#[test]
fn table_free_composition_can_itself_be_reused_as_a_child() {
    let (vocab, tokens) = vocabulary();
    let options = || BuildOptions::default().optimization(Optimization::FastBuild).parser_backend(ParserBackend::TemplateDfa);
    let leaf = Grammar::from_ebnf(r#"start ::= "a" | "b""#).compile_with(&vocab, options()).unwrap();
    let middle = Grammar::from_glrm(r#"glrm 1; start mid; extern grammar leaf; nt mid = "p" leaf "q";"#)
        .compile_unlinked(&vocab).unwrap().bind("leaf", &leaf).unwrap().link_with(options()).unwrap();
    for middle in [&middle, &Constraint::load(middle.save()).unwrap()] {
        let outer = Grammar::from_glrm(r#"glrm 1; start root; extern grammar middle; nt root = "x" middle middle "y";"#)
            .compile_unlinked(&vocab).unwrap().bind("middle", middle).unwrap().link_with(options()).unwrap();
        let words: &[&[u8]] = &[b"xpaqpaqy", b"xpaqpbqy", b"xpbqpaqy", b"xpbqpbqy"];
        assert_language(&outer, &tokens, words);
        assert_language(&Constraint::load(outer.save()).unwrap(), &tokens, words);
    }
}

fn vocabulary() -> (Vocab, Vec<(u32, Vec<u8>)>) {
    let mut tokens = (0..128).map(|id| (id, vec![id as u8])).collect::<Vec<_>>();
    for (index, token) in ["xay", "xby", "xaby", "xa", "ay", "ab", "by", "xy", "yx", " ",
        "xp", "paq", "aqy", "xpaqy", "xpaqpaqy", "xpaqpbqy", "qy", "pq" ].iter().enumerate() {
        tokens.push((256 + index as u32, token.as_bytes().to_vec()));
    }
    (Vocab::new(tokens.clone()), tokens)
}

#[test]
fn nested_repeated_child_retains_scope_across_token_boundaries_and_reload() {
    let (vocab, tokens) = vocabulary();
    let leaf = Grammar::from_ebnf(r#"start ::= "a" | "b""#).compile(&vocab).unwrap();
    let middle = Grammar::from_glrm(r#"glrm 1; start mid; extern grammar leaf; nt mid = "p" leaf "q";"#)
        .compile_unlinked(&vocab).unwrap().bind("leaf", &leaf).unwrap().link().unwrap();
    let outer = Grammar::from_glrm(r#"glrm 1; start root; extern grammar middle; nt root = "x" middle middle "y";"#)
        .compile_unlinked(&vocab).unwrap().bind("middle", &middle).unwrap();
    for optimization in [Optimization::FastRuntime, Optimization::FastBuild] {
        let candidate = outer.link_with(BuildOptions::default().optimization(optimization)
            .parser_backend(ParserBackend::TemplateDfa)).unwrap();
        assert_language(&candidate, &tokens, &[b"xpaqpaqy", b"xpaqpbqy", b"xpbqpaqy", b"xpbqpbqy"]);
        let loaded = Constraint::load(candidate.save()).unwrap();
        assert_language(&loaded, &tokens, &[b"xpaqpaqy", b"xpaqpbqy", b"xpbqpaqy", b"xpbqpbqy"]);
    }
}

fn assert_language(constraint: &Constraint, tokens: &[(u32, Vec<u8>)], words: &[&[u8]]) {
    assert_eq!(constraint.parser_backend(), ParserBackend::TemplateDfa);
    let report = glrmask::__private::parser_backend_report(constraint);
    assert_eq!(report["lr_table_present"], false);
    let mut prefixes = BTreeSet::new();
    for word in words {
        for length in 0..=word.len() { prefixes.insert(word[..length].to_vec()); }
    }
    for prefix in prefixes {
        let mut state = constraint.start();
        state.commit_bytes(&prefix).unwrap();
        assert_eq!(state.is_accepting(), words.contains(&prefix.as_slice()), "completion {prefix:?}");
        let mask = state.mask();
        for (id, bytes) in tokens {
            let mut candidate = prefix.clone(); candidate.extend(bytes);
            let expected = words.iter().any(|word| word.starts_with(&candidate));
            let actual = mask.get(*id as usize / 32).is_some_and(|word| word & (1 << (*id % 32)) != 0);
            assert_eq!(actual, expected, "prefix={prefix:?}, token={bytes:?}, id={id}, report={report}");
            if actual {
                let mut replay = constraint.start();
                replay.commit_bytes(&prefix).unwrap();
                replay.commit_token(*id).unwrap_or_else(|error| panic!("prefix={prefix:?} token={bytes:?} id={id}: {error}"));
                assert_eq!(replay.is_accepting(), words.contains(&candidate.as_slice()), "commit {candidate:?}");
            }
        }
    }
}

#[test]
fn compiled_child_composition_uses_templates_for_advance_and_masks() {
    let (vocab, tokens) = vocabulary();
    let child = Grammar::from_ebnf(r#"start ::= "a" | "b" | "a" "b""#).compile(&vocab).unwrap();
    let bound = Grammar::from_glrm(HOST).compile_unlinked(&vocab).unwrap().bind("child", &child).unwrap();
    for optimization in [Optimization::FastRuntime, Optimization::FastBuild, Optimization::Auto] {
        let candidate = bound.link_with(BuildOptions::default().optimization(optimization)
            .parser_backend(ParserBackend::TemplateDfa)).unwrap();
        assert_language(&candidate, &tokens, &[b"xay", b"xby", b"xaby"]);
        let saved = candidate.save();
        let loaded = Constraint::load(&saved).unwrap();
        assert_language(&loaded, &tokens, &[b"xay", b"xby", b"xaby"]);
        assert_eq!(saved, loaded.save());
        assert_eq!(child.parser_backend(), ParserBackend::TemplateDfa, "link must not mutate a shared child");
    }
}

#[test]
fn source_bound_composition_uses_template_backend() {
    let (vocab, tokens) = vocabulary();
    let child = Grammar::from_ebnf(r#"start ::= "a" | "b" | "a" "b""#);
    let source = Grammar::from_glrm(HOST).bind("child", &child).unwrap();
    for optimization in [Optimization::FastRuntime, Optimization::FastBuild, Optimization::Auto] {
        let candidate = source.compile_with(&vocab, BuildOptions::default().optimization(optimization)
            .parser_backend(ParserBackend::TemplateDfa)).unwrap();
        assert_language(&candidate, &tokens, &[b"xay", b"xby", b"xaby"]);
        let loaded = Constraint::load(candidate.save()).unwrap();
        assert_language(&loaded, &tokens, &[b"xay", b"xby", b"xaby"]);
    }
}

#[test]
fn nullable_child_controls_preserve_empty_body_and_parent_suffix() {
    let (vocab, tokens) = vocabulary();
    let child = Grammar::from_ebnf(r#"start ::= "a"?"#).compile(&vocab).unwrap();
    let bound = Grammar::from_glrm(HOST).compile_unlinked(&vocab).unwrap().bind("child", &child).unwrap();
    for optimization in [Optimization::FastRuntime, Optimization::FastBuild] {
        let candidate = bound.link_with(BuildOptions::default().optimization(optimization)
            .parser_backend(ParserBackend::TemplateDfa)).unwrap();
        assert_language(&candidate, &tokens, &[b"xy", b"xay"]);
        let loaded = Constraint::load(candidate.save()).unwrap();
        assert_language(&loaded, &tokens, &[b"xy", b"xay"]);
    }
}
