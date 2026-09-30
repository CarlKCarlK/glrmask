use std::collections::BTreeSet;
use glrmask::{BuildOptions, Constraint, Grammar, Optimization, ParserBackend, Vocab};

const HOST: &str = r#"glrm 1; start root; extern grammar child; nt root = "x" child "y";"#;

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
                for mode in [Optimization::FastBuild, Optimization::Auto] {
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
                replay.commit_token(*id).unwrap();
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
        assert_eq!(child.parser_backend(), ParserBackend::LrTable, "link must not mutate a shared child");
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
