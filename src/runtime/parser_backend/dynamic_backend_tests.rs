//! Public behavior coverage for the ordinary retained
//! `ParserBackend::LrTable` runtime, compared against the default native
//! `TemplateDfa` backend. Construction-shape witnesses (retained table, no
//! template artifacts) live in the crate-internal `explicit_lr_tests` module.

use crate::{BuildOptions, Constraint, Grammar, Optimization, ParserBackend, Vocab};

/// Duplicate byte spellings, sparse IDs, an empty-byte token and a token used
/// only as an explicit end token exercise the vocabulary surfaces shared by
/// both backends.
fn vocab() -> Vocab {
    Vocab::new(vec![
        (0, b"a".to_vec()),
        (1, b"b".to_vec()),
        (2, b"(".to_vec()),
        (3, b")".to_vec()),
        (4, b"a".to_vec()),
        (7, b"ab".to_vec()),
        (9, Vec::new()),
        (20, b"Z".to_vec()),
    ])
}

const END_TOKEN: u32 = 20;

fn compile(vocab: &Vocab, source: &str, backend: ParserBackend) -> Constraint {
    if backend == ParserBackend::LrTable {
        crate::DynamicConstraint::from_ebnf(source, vocab)
            .expect("ordinary Dynamic compile").into_constraint()
            .with_end_tokens(&[END_TOKEN]).expect("end tokens")
    } else {
        Grammar::from_ebnf(source).compile_with(vocab,
            BuildOptions::default().optimization(Optimization::FastBuild)
                .end_tokens([END_TOKEN])).expect("O2 template compile")
    }
}

fn allowed(mask: &[u32], id: u32) -> bool {
    mask.get(id as usize / 32)
        .is_some_and(|word| word & (1 << (id % 32)) != 0)
}

/// Directly compare two already-built constraints (for example in-memory vs
/// reloaded) across a full positive trace, including final acceptance and the
/// explicit end token, without recompiling.
fn assert_pair_trace(
    left: &Constraint,
    right: &Constraint,
    trace: &[u32],
    final_accepting: bool,
) {
    assert_eq!(left.mask_len(), right.mask_len());
    let mut a = left.start();
    let mut b = right.start();
    let mut left_mask = vec![0u32; left.mask_len()];
    let mut right_mask = vec![0u32; right.mask_len()];
    left_mask.fill(0);
    right_mask.fill(0);
    a.fill_mask(&mut left_mask);
    b.fill_mask(&mut right_mask);
    assert_eq!(left_mask, right_mask, "initial mask parity");
    for &token in trace {
        assert!(allowed(&left_mask, token), "positive token {token} must be admitted");
        a.commit_token(token).unwrap_or_else(|err| panic!("left rejected {token}: {err:?}"));
        b.commit_token(token).unwrap_or_else(|err| panic!("right rejected {token}: {err:?}"));
        left_mask.fill(0);
        right_mask.fill(0);
        a.fill_mask(&mut left_mask);
        b.fill_mask(&mut right_mask);
        assert_eq!(left_mask, right_mask, "mask parity after {token}");
    }
    assert_eq!(a.is_accepting(), final_accepting, "left final acceptance");
    assert_eq!(b.is_accepting(), final_accepting, "right final acceptance");
    if final_accepting {
        assert!(allowed(&left_mask, END_TOKEN), "end token must be admitted at acceptance");
        a.commit_token(END_TOKEN).expect("left end token commit");
        b.commit_token(END_TOKEN).expect("right end token commit");
    }
}

/// Positive trace: every commit must be accepted, per-step masks and acceptance
/// must match the native backend, and the final acceptance must equal the
/// caller's exact expectation.
fn assert_positive_trace(source: &str, trace: &[u32], final_accepting: bool) {
    let vocab = vocab();
    let lr = compile(&vocab, source, ParserBackend::LrTable);
    let native = compile(&vocab, source, ParserBackend::TemplateDfa);
    assert_eq!(lr.parser_backend(), ParserBackend::LrTable);
    assert_eq!(native.parser_backend(), ParserBackend::TemplateDfa);

    let mut left = lr.start();
    let mut right = native.start();
    let mut left_mask = vec![0u32; lr.mask_len()];
    let mut right_mask = vec![0u32; native.mask_len()];
    left.fill_mask(&mut left_mask);
    right.fill_mask(&mut right_mask);
    assert_eq!(left_mask, right_mask, "initial masks differ for {source:?}");
    assert_eq!(left.is_accepting(), right.is_accepting(), "initial acceptance differs");
    for &token in trace {
        left_mask.fill(0);
        right_mask.fill(0);
        left.fill_mask(&mut left_mask);
        right.fill_mask(&mut right_mask);
        assert_eq!(left_mask, right_mask, "mask parity before {token} for {source:?}");
        assert!(
            allowed(&left_mask, token),
            "explicit LR mask must allow positive trace token {token} for {source:?}",
        );
        left.commit_token(token)
            .unwrap_or_else(|err| panic!("explicit LR must accept token {token} for {source:?}: {err:?}"));
        right.commit_token(token)
            .unwrap_or_else(|err| panic!("native must accept token {token} for {source:?}: {err:?}"));
    }
    left_mask.fill(0);
    right_mask.fill(0);
    left.fill_mask(&mut left_mask);
    right.fill_mask(&mut right_mask);
    assert_eq!(left_mask, right_mask, "final masks differ for {source:?}");
    assert_eq!(
        left.is_accepting(),
        final_accepting,
        "explicit LR final acceptance mismatch for {source:?}",
    );
    assert_eq!(
        right.is_accepting(),
        final_accepting,
        "native final acceptance mismatch for {source:?}",
    );
}

/// Parity-only trace for inputs whose membership is intentionally not asserted
/// (for example zero-byte tokens); still requires per-step agreement.
fn assert_parity_trace(source: &str, trace: &[u32]) {
    let vocab = vocab();
    let lr = compile(&vocab, source, ParserBackend::LrTable);
    let native = compile(&vocab, source, ParserBackend::TemplateDfa);
    let mut left = lr.start();
    let mut right = native.start();
    let mut left_mask = vec![0u32; lr.mask_len()];
    let mut right_mask = vec![0u32; native.mask_len()];
    for &token in trace {
        left_mask.fill(0);
        right_mask.fill(0);
        left.fill_mask(&mut left_mask);
        right.fill_mask(&mut right_mask);
        assert_eq!(left_mask, right_mask, "mask parity before {token}");
        let left_ok = left.commit_token(token).is_ok();
        let right_ok = right.commit_token(token).is_ok();
        assert_eq!(left_ok, right_ok, "commit parity for token {token}");
        assert_eq!(left.is_accepting(), right.is_accepting(), "acceptance parity for {token}");
    }
}

#[test]
fn acyclic_positive_trace_and_end_token() {
    let source = r#"start ::= "a" "b""#;
    assert_positive_trace(source, &[0, 1], true);

    // The explicit end token is only *admitted* once the parser is accepting.
    let vocab = vocab();
    let lr = compile(&vocab, source, ParserBackend::LrTable);
    let mut state = lr.start();
    assert!(!state.is_accepting());
    assert!(!allowed(&state.mask(), END_TOKEN), "end token must not be allowed before acceptance");
    state.commit_token(0).unwrap();
    state.commit_token(1).unwrap();
    assert!(state.is_accepting());
    assert!(allowed(&state.mask(), END_TOKEN), "end token must be allowed at acceptance");
    state.commit_token(END_TOKEN).expect("end token commit at accepting state");
    // Root termination policy: the mask becomes empty and further commits fail.
    assert_eq!(state.mask(), vec![0u32; lr.mask_len()]);
    assert!(state.commit_token(0).is_err(), "commit after termination must fail");

    // Committing an end token before acceptance never reaches acceptance.
    let mut early = lr.start();
    let _ = early.commit_token(END_TOKEN);
    assert!(!early.is_accepting(), "early end token commit must not accept");
}

#[test]
fn recursive_productive_balanced_traces() {
    // Productive, non-nullable balanced parentheses with an optional tail.
    let source = r#"start ::= "(" start ")" start? | "a""#;
    assert_positive_trace(source, &[2, 0, 3], true); // (a)
    assert_positive_trace(source, &[2, 0, 3, 0], true); // (a)a
    assert_positive_trace(source, &[2, 2, 0, 3, 3], true); // ((a))
    assert_positive_trace(source, &[2, 0, 3, 2, 0, 3], true); // (a)(a)
    assert_positive_trace(source, &[2, 0], false); // non-accepting prefix
    assert_positive_trace(source, &[2], false);
}

#[test]
fn nullable_empty_start_and_eof_completion() {
    let source = r#"start ::= "a"?"#;
    assert_positive_trace(source, &[], true); // empty sentence accepted
    assert_positive_trace(source, &[0], true); // "a"
    // Zero-byte token is a lexer-level concern; require exact parity only.
    assert_parity_trace(source, &[9]);
}

#[test]
fn nullable_recursive_balanced_traces() {
    // `start` is both nullable and genuinely recursive.
    let source = r#"start ::= ("(" start ")")?"#;
    assert_positive_trace(source, &[], true); // empty
    assert_positive_trace(source, &[2, 3], true); // ()
    assert_positive_trace(source, &[2, 2, 3, 3], true); // (())
    assert_positive_trace(source, &[2], false);
}

#[test]
fn known_invalid_commit_drives_fail_state_like_native() {
    // Documented contract (commit/mod.rs): committing a token that exists in
    // the vocabulary but is grammatically invalid drives the constraint into a
    // fail state, observable as an all-zero mask. It is not transactional.
    let vocab = vocab();
    let source = r#"start ::= "a" "b""#;
    let lr = compile(&vocab, source, ParserBackend::LrTable);
    let native = compile(&vocab, source, ParserBackend::TemplateDfa);
    let mut left = lr.start();
    let mut right = native.start();
    // Token 1 ("b") is known but not a valid first token.
    let left_result = left.commit_token(1).is_ok();
    let right_result = right.commit_token(1).is_ok();
    assert_eq!(left_result, right_result, "commit result must match native");
    assert_eq!(
        left.mask(),
        vec![0u32; lr.mask_len()],
        "explicit LR fail-state mask must be all zero",
    );
    assert_eq!(
        right.mask(),
        vec![0u32; native.mask_len()],
        "native fail-state mask must be all zero",
    );
    assert_eq!(left.is_accepting(), right.is_accepting());
    assert!(!left.is_accepting());
}

#[test]
fn unknown_token_id_is_rejected_without_mutating_state() {
    // An ID neither present in the vocabulary nor declared by a special-token
    // terminal must error through the existing knows-token check and leave the
    // state usable. This is separate from the known-invalid fail-state policy.
    let vocab = vocab();
    let source = r#"start ::= "a" "b""#;
    let lr = compile(&vocab, source, ParserBackend::LrTable);
    let native = compile(&vocab, source, ParserBackend::TemplateDfa);
    const UNKNOWN: u32 = 4095;
    for constraint in [&lr, &native] {
        let mut state = constraint.start();
        let before = state.mask();
        assert!(
            state.commit_token(UNKNOWN).is_err(),
            "unknown token ID must be rejected",
        );
        assert_eq!(state.mask(), before, "unknown token must not mutate state");
        // The state remains usable and identical to a fresh state.
    }
    let mut left = lr.start();
    let mut right = native.start();
    assert_eq!(left.commit_token(UNKNOWN).is_err(), right.commit_token(UNKNOWN).is_err());
    assert_eq!(left.commit_token(0).is_ok(), right.commit_token(0).is_ok());
    assert_eq!(left.mask(), right.mask());
}

#[test]
fn self_contained_roundtrip_preserves_backend_and_behavior() {
    let vocab = vocab();
    let source = r#"start ::= "(" start ")" start? | "a""#;
    for backend in [ParserBackend::LrTable, ParserBackend::TemplateDfa] {
        let constraint = compile(&vocab, source, backend);
        let loaded = Constraint::load(constraint.save()).expect("self-contained roundtrip");
        assert_eq!(loaded.parser_backend(), backend, "backend identity changed");
        assert_pair_trace(&constraint, &loaded, &[2, 0, 3], true);
        // A second, longer balanced sentence.
        assert_pair_trace(&constraint, &loaded, &[2, 2, 0, 3, 3, 0], true);
    }
}

#[test]
fn external_vocab_roundtrip_preserves_backend_and_behavior() {
    let vocab = vocab();
    let source = r#"start ::= "(" start ")" start? | "a""#;
    for backend in [ParserBackend::LrTable, ParserBackend::TemplateDfa] {
        let constraint = compile(&vocab, source, backend);
        let bytes = constraint.save_with_external_vocab().expect("external-vocab save");
        let loaded = Constraint::load_with_vocab(bytes.clone(), &vocab)
            .expect("external-vocab roundtrip");
        assert_eq!(loaded.parser_backend(), backend, "backend identity changed");
        assert_pair_trace(&constraint, &loaded, &[2, 0, 3], true);
        assert_pair_trace(&constraint, &loaded, &[2, 2, 0, 3, 3, 0], true);
        // The external envelope requires its exact vocabulary.
        assert!(Constraint::load(bytes).is_err(), "missing external vocabulary must be rejected");
    }
}

#[test]
fn external_lr_rejects_wrong_vocabulary_with_same_ids() {
    let vocab = vocab();
    let constraint = compile(&vocab, r#"start ::= "a" "b""#, ParserBackend::LrTable);
    let bytes = constraint.save_with_external_vocab().unwrap();
    // Same IDs and byte lengths, different byte content.
    let wrong = Vocab::new(vec![
        (0, b"x".to_vec()),
        (1, b"y".to_vec()),
        (2, b"(".to_vec()),
        (3, b")".to_vec()),
        (4, b"x".to_vec()),
        (7, b"xy".to_vec()),
        (9, Vec::new()),
        (20, b"Q".to_vec()),
    ]);
    assert!(
        Constraint::load_with_vocab(bytes, &wrong).is_err(),
        "external LR artifact must validate the exact vocabulary content digest",
    );
}


#[test]
fn development_toggle_is_compile_only_and_o2_static_are_isolated() {
    use std::process::Command;
    let vocab = vocab();
    let source = r#"start ::= ("(" start ")")?"#;
    const CHILD: &str = "GLRMASK_DYNAMIC_BACKEND_TEST_CHILD";
    if let Ok(directory) = std::env::var(CHILD) {
        let directory = std::path::Path::new(&directory);
        let expected = if std::env::var("GLRMASK_DYNAMIC_TEMPLATE_DFA").unwrap() == "1" {
            ParserBackend::TemplateDfa
        } else { ParserBackend::LrTable };
        let dynamic = crate::DynamicConstraint::from_ebnf(source, &vocab).unwrap();
        assert_eq!(dynamic.clone().into_constraint().parser_backend(), expected);
        let o2 = crate::DynamicConstraint::from_ebnf_with_vocab_partition(source, &vocab)
            .unwrap().into_constraint();
        assert_eq!(o2.parser_backend(), ParserBackend::TemplateDfa);
        assert!(!o2.table.is_present());
        let fast_build = compile(&vocab, source, ParserBackend::TemplateDfa);
        assert_eq!(fast_build.parser_backend(), ParserBackend::TemplateDfa);
        let static_constraint = Grammar::from_ebnf(source).compile_with(&vocab,
            BuildOptions::default().optimization(Optimization::FastRuntime)).unwrap();
        assert_eq!(static_constraint.parser_backend(), ParserBackend::TemplateDfa);
        let loaded = Constraint::load(std::fs::read(directory.join("lr.constraint")).unwrap()).unwrap();
        assert_eq!(loaded.parser_backend(), ParserBackend::LrTable);
        assert!(loaded.start().is_accepting());
        let loaded_dynamic = crate::DynamicConstraint::load(
            &std::fs::read(directory.join("lr.dynamic")).unwrap()).unwrap();
        assert_eq!(loaded_dynamic.into_constraint().parser_backend(), ParserBackend::LrTable);
        if expected == ParserBackend::TemplateDfa {
            std::fs::write(directory.join("native.constraint"), dynamic.clone().into_constraint().save()).unwrap();
            std::fs::write(directory.join("native.dynamic"), dynamic.save()).unwrap();
        }
        return;
    }
    let directory = std::env::temp_dir().join(format!("glrmask-dynamic-backend-test-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let dynamic = crate::DynamicConstraint::from_ebnf(source, &vocab).unwrap();
    assert_eq!(dynamic.clone().into_constraint().parser_backend(), ParserBackend::LrTable);
    std::fs::write(directory.join("lr.constraint"), dynamic.clone().into_constraint().save()).unwrap();
    std::fs::write(directory.join("lr.dynamic"), dynamic.save()).unwrap();
    for value in ["0", "1"] {
        let status = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "runtime::parser_backend::dynamic_backend_tests::development_toggle_is_compile_only_and_o2_static_are_isolated", "--nocapture"])
            .env(CHILD, &directory).env("GLRMASK_DYNAMIC_TEMPLATE_DFA", value)
            // Legacy diagnostic advance switches must never convert LR at runtime.
            .env("GLRMASK_ENABLE_TEMPLATE_DFA_ADVANCE", "1")
            .status().unwrap();
        assert!(status.success(), "backend isolation child {value} failed");
    }
    let native = Constraint::load(std::fs::read(directory.join("native.constraint")).unwrap()).unwrap();
    assert_eq!(native.parser_backend(), ParserBackend::TemplateDfa);
    let native_dynamic = crate::DynamicConstraint::load(
        &std::fs::read(directory.join("native.dynamic")).unwrap()).unwrap();
    assert_eq!(native_dynamic.into_constraint().parser_backend(), ParserBackend::TemplateDfa);
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn nullable_ignore_and_loaded_prefixes_preserve_completion() {
    let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b" ".to_vec()),
        (2, b"(".to_vec()), (3, b")".to_vec()), (20, b"Z".to_vec())]);
    let source = "start: [\"(\" start \")\"]\n%ignore / +/";
    let lr = crate::DynamicConstraint::from_lark(source, &vocab).unwrap()
        .into_constraint().with_end_tokens(&[END_TOKEN]).unwrap();
    let native = Grammar::from_lark(source).compile_with(&vocab,
        BuildOptions::default().optimization(Optimization::FastBuild).end_tokens([END_TOKEN])).unwrap();
    for constraint in [&lr, &native] {
        for loaded in [Constraint::load(constraint.save()).unwrap(),
            Constraint::load_with_vocab(constraint.save_with_external_vocab().unwrap(), &vocab).unwrap()] {
            for (trace, accepting) in [(&[][..], true), (&[1][..], true), (&[2][..], false),
                (&[1, 2, 1][..], false), (&[2, 1, 3, 1][..], true)] {
                assert_pair_trace(constraint, &loaded, trace, accepting);
            }
        }
    }
    for trace in [&[][..], &[1][..], &[2][..], &[1, 2, 1][..], &[2, 1, 3, 1][..]] {
        let mut a = lr.start(); let mut b = native.start();
        for &id in trace { a.commit_token(id).unwrap(); b.commit_token(id).unwrap(); }
        assert_eq!(a.mask(), b.mask(), "ignore trace {trace:?}");
        assert_eq!(a.is_accepting(), b.is_accepting(), "ignore trace {trace:?}");
    }
}

#[test]
fn empty_special_id_is_exact_and_ordinary_empty_alias_is_rejected() {
    let vocab = Vocab::new(vec![(0, b"a".to_vec()), (9, Vec::new()), (11, Vec::new())]);
    let source = "glrm 1; start start; extern token MARK; nt start = MARK;";
    let lr = crate::DynamicConstraint::from_glrm_grammar_with_bindings_and_end_tokens(
        source, &vocab, &[("MARK", &[9])], &[]).unwrap().into_constraint();
    let native = Grammar::from_glrm(source).bind("MARK", vocab.token(9).unwrap()).unwrap()
        .compile_with(&vocab, BuildOptions::default().optimization(Optimization::FastBuild)).unwrap();
    for constraint in [&lr, &native] {
        let mut state = constraint.start();
        assert!(allowed(&state.mask(), 9));
        assert!(!allowed(&state.mask(), 11));
        state.commit_token(9).unwrap();
        assert!(state.is_accepting());
        let mut rejected = constraint.start();
        assert!(rejected.commit_token(11).is_err());
        assert!(rejected.is_rejected());
    }
}

#[test]
fn focused_dynamic_empty_token_and_external_identity_regression() {
    let vocab = Vocab::new(vec![(0, b"(".to_vec()), (2, b")".to_vec()),
        (7, b"()".to_vec()), (15, b"(".to_vec()), (20, b"x".to_vec()),
        (25, Vec::new()), (26, Vec::new())]);
    let source = r#"start ::= "(" start ")" start | """#;
    for dynamic in [crate::DynamicConstraint::from_ebnf(source, &vocab).unwrap(),
        crate::DynamicConstraint::from_ebnf_with_vocab_partition(source, &vocab).unwrap()] {
        let backend = dynamic.clone().into_constraint().parser_backend();
        let transfer = dynamic.save_with_external_vocab();
        let version = u16::from_le_bytes(transfer[8..10].try_into().unwrap());
        assert_eq!(version, if backend == ParserBackend::LrTable { 15 } else { 14 });
        let self_loaded = crate::DynamicConstraint::load(&dynamic.save()).unwrap();
        let external_loaded = crate::DynamicConstraint::load_with_vocab(&transfer, &vocab).unwrap();
        assert_eq!(external_loaded.save_with_external_vocab(), transfer);
        for compiled in [&dynamic, &self_loaded, &external_loaded] {
            for prefix in [b"".as_slice(), b"(", b"()", b"()("] {
                let mut state = compiled.start();
                state.commit_bytes(prefix).unwrap();
                for _ in 0..2 { // Repeat the query to exercise the mask cache.
                    assert!(!allowed(&state.mask(), 25), "{backend:?} {prefix:?}");
                    assert!(!allowed(&state.mask(), 26));
                }
                assert!(state.commit_token(25).is_err());
                assert!(state.is_rejected());
            }
        }
        for changed in [
            Vocab::new(vec![(0,b"[".to_vec()),(2,b")".to_vec()),(7,b"()".to_vec()),
                (15,b"(".to_vec()),(20,b"x".to_vec()),(25,Vec::new()),(26,Vec::new())]),
            Vocab::new(vec![(0,b"(".to_vec()),(2,b")".to_vec()),(7,b"()".to_vec()),
                (15,b"(".to_vec()),(20,b"y".to_vec()),(25,Vec::new()),(26,Vec::new())]),
            Vocab::new(vec![(0,b"(".to_vec()),(2,b")".to_vec()),(7,b"()".to_vec()),
                (16,b"(".to_vec()),(20,b"x".to_vec()),(25,Vec::new()),(26,Vec::new())]),
        ] {
            assert!(crate::DynamicConstraint::load_with_vocab(&transfer, &changed).is_err());
            assert!(crate::DynamicConstraint::load_with_vocab(&dynamic.save(), &changed).is_err());
        }
        if backend == ParserBackend::LrTable {
            let mut corrupt = transfer.clone();
            corrupt[26] ^= 1; // First byte of the mandatory envelope digest.
            assert!(crate::DynamicConstraint::load_with_vocab(&corrupt, &vocab).is_err());
            for version in 1u16..=13 {
                let mut old = transfer.clone();
                old[8..10].copy_from_slice(&version.to_le_bytes());
                let error = crate::DynamicConstraint::load_with_vocab(&old, &vocab).unwrap_err();
                assert!(error.to_string().contains("lacks mandatory exact vocabulary identity"));
            }
        }
    }
}

#[test]
fn focused_public_empty_token_policy_preserves_empty_eos_across_reload() {
    let vocab = Vocab::new(vec![(0,b"(".to_vec()),(2,b")".to_vec()),
        (25,Vec::new()),(26,Vec::new())]);
    let source = r#"start ::= "(" start ")" start | """#;
    let lr = crate::DynamicConstraint::from_ebnf(source, &vocab).unwrap()
        .into_constraint().with_end_tokens(&[26]).unwrap();
    let mut constraints = vec![lr];
    for mode in [Optimization::Auto, Optimization::FastBuild, Optimization::FastRuntime] {
        constraints.push(Grammar::from_ebnf(source).compile_with(&vocab,
            BuildOptions::default().optimization(mode).end_tokens([26])).unwrap());
    }
    for compiled in constraints {
        for constraint in [&compiled,
            &Constraint::load(compiled.save()).unwrap(),
            &Constraint::load_with_vocab(compiled.save_with_external_vocab().unwrap(), &vocab).unwrap()] {
            for (prefix, accepting) in [(b"".as_slice(),true),(b"(",false),(b"()",true)] {
                let mut state = constraint.start();state.commit_bytes(prefix).unwrap();
                for _ in 0..2 {
                    assert!(!allowed(&state.mask(),25));
                    assert_eq!(allowed(&state.mask(),26),accepting);
                }
                if accepting {
                    state.commit_token(26).unwrap();assert!(state.is_terminated());
                    assert!(state.mask().iter().all(|&word| word==0));
                } else {
                    // Established root EOS API returns Ok while rejecting an
                    // early end token; the mask and resulting state are exact.
                    state.commit_token(26).unwrap();
                    assert!(state.is_rejected());assert!(!state.is_terminated());
                }
            }
        }
    }
}

#[test]
fn focused_empty_special_token_is_live_after_prefix_across_reload() {
    let vocab = Vocab::new(vec![(0,b"a".to_vec()),(25,Vec::new()),(26,Vec::new())]);
    let source = "glrm 1; start root; extern token MARK; nt root = \"a\" MARK;";
    let lr = crate::DynamicConstraint::from_glrm_grammar_with_bindings_and_end_tokens(
        source,&vocab,&[("MARK",&[25])],&[]).unwrap().into_constraint();
    let mut constraints = vec![lr];
    for mode in [Optimization::Auto,Optimization::FastBuild,Optimization::FastRuntime] {
        constraints.push(Grammar::from_glrm(source).bind("MARK",vocab.token(25).unwrap()).unwrap()
            .compile_with(&vocab,BuildOptions::default().optimization(mode)).unwrap());
    }
    for compiled in constraints {
        for constraint in [&compiled,&Constraint::load(compiled.save()).unwrap(),
            &Constraint::load_with_vocab(compiled.save_with_external_vocab().unwrap(),&vocab).unwrap()] {
            let mut state=constraint.start();assert!(!allowed(&state.mask(),25));
            state.commit_token(0).unwrap();assert!(allowed(&state.mask(),25));
            assert!(!allowed(&state.mask(),26));state.commit_token(25).unwrap();
            assert!(state.is_accepting());
        }
    }
}
