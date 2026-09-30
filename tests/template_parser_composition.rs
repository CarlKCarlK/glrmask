use std::collections::BTreeSet;
use glrmask::{BuildOptions, Constraint, Grammar, Optimization, ParserBackend, Vocab};

const HOST: &str = r#"glrm 1; start root; extern grammar child; nt root = "x" child "y";"#;

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
