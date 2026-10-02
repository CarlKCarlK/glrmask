use glrmask::{
    BuildOptions, Constraint, ConstraintSpec, DynamicConstraint, Grammar, Optimization, UnlinkedConstraint,
    Vocab,
};

fn allowed(mask: &[u32], token_id: u32) -> bool {
    let word = token_id as usize / 32;
    let bit = token_id % 32;
    mask.get(word)
        .is_some_and(|value| value & (1u32 << bit) != 0)
}

fn assert_static_xy_matches(reference: &Constraint, candidate: &Constraint) {
    let mut reference_state = reference.start();
    let mut candidate_state = candidate.start();
    assert_eq!(candidate_state.mask(), reference_state.mask());
    assert!(allowed(&candidate_state.mask(), 2));
    assert!(!allowed(&candidate_state.mask(), 3));
    candidate_state.commit_token(2).unwrap();
    reference_state.commit_token(2).unwrap();
    assert_eq!(candidate_state.is_accepting(), reference_state.is_accepting());

    let mut reference_state = reference.start();
    let mut candidate_state = candidate.start();
    candidate_state.commit_token(0).unwrap();
    reference_state.commit_token(0).unwrap();
    assert_eq!(candidate_state.mask(), reference_state.mask());
    candidate_state.commit_token(1).unwrap();
    reference_state.commit_token(1).unwrap();
    assert_eq!(candidate_state.is_accepting(), reference_state.is_accepting());
}

fn assert_dynamic_xy_matches(reference: &Constraint, candidate: &DynamicConstraint) {
    let mut reference_state = reference.start();
    let mut candidate_state = candidate.start();
    assert_eq!(candidate_state.mask(), reference_state.mask());
    assert!(allowed(&candidate_state.mask(), 2));
    assert!(!allowed(&candidate_state.mask(), 3));
    candidate_state.commit_token(2).unwrap();
    reference_state.commit_token(2).unwrap();
    assert_eq!(candidate_state.is_accepting(), reference_state.is_accepting());

    let mut reference_state = reference.start();
    let mut candidate_state = candidate.start();
    candidate_state.commit_token(0).unwrap();
    reference_state.commit_token(0).unwrap();
    assert_eq!(candidate_state.mask(), reference_state.mask());
    candidate_state.commit_token(1).unwrap();
    reference_state.commit_token(1).unwrap();
    assert_eq!(candidate_state.is_accepting(), reference_state.is_accepting());
}

fn assert_static_xyz_matches(reference: &Constraint, candidate: &Constraint) {
    for tokens in [&[5][..], &[2, 3][..], &[0, 4][..], &[0, 1, 3][..]] {
        let mut reference_state = reference.start();
        let mut candidate_state = candidate.start();
        for &token in tokens {
            assert_eq!(candidate_state.mask(), reference_state.mask());
            candidate_state.commit_token(token).unwrap();
            reference_state.commit_token(token).unwrap();
        }
        assert_eq!(candidate_state.is_accepting(), reference_state.is_accepting());
    }
}

const EXACT_GRAMMAR: &str = r#"
glrm 1;
start start;
extern token SPECIAL;
nt start = "a" SPECIAL "b";
"#;

#[test]
fn named_external_token_is_exact_not_a_byte_language() {
    let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec()), (7, b"SPECIAL".to_vec())]);
    let spec = ConstraintSpec::builder(Grammar::glrm(EXACT_GRAMMAR), &vocab)
        .unwrap()
        .bind_token("SPECIAL", [7])
        .unwrap()
        .build()
        .unwrap();
    let constraint = spec.compile().unwrap();

    let mut state = constraint.start();
    assert!(allowed(&state.mask(), 0));
    state.commit_token(0).unwrap();
    assert!(allowed(&state.mask(), 7));
    assert!(!allowed(&state.mask(), 1));

    let mut by_bytes = state.clone();
    assert!(by_bytes.commit_bytes(b"SPECIAL").is_err());

    state.commit_token(7).unwrap();
    assert!(allowed(&state.mask(), 1));
    state.commit_token(1).unwrap();
    assert!(state.is_accepting());
}

#[test]
fn byte_less_decoder_ids_are_valid_and_multiple_ids_are_interchangeable() {
    let vocab = Vocab::new(Vec::new());
    let source = "glrm 1; start start; extern token SPECIAL; nt start = SPECIAL;";
    let spec = ConstraintSpec::builder(Grammar::glrm(source), &vocab)
        .unwrap()
        .bind_token("SPECIAL", [100, 101])
        .unwrap()
        .build()
        .unwrap();
    let static_constraint = spec.compile().unwrap();
    let dynamic_constraint = spec.compile_dynamic().unwrap();

    for token in [100, 101] {
        let mut static_state = static_constraint.start();
        let mut dynamic_state = dynamic_constraint.start();
        assert!(allowed(&static_state.mask(), 100));
        assert!(allowed(&static_state.mask(), 101));
        assert_eq!(static_state.mask(), dynamic_state.mask());
        static_state.commit_token(token).unwrap();
        dynamic_state.commit_token(token).unwrap();
        assert!(static_state.is_accepting());
        assert!(dynamic_state.is_accepting());
    }
}

#[test]
fn external_token_validation_is_early_and_complete() {
    let vocab = Vocab::new(vec![(7, Vec::new())]);
    let source = r#"
glrm 1;
start start;
extern token CONTROL;
extern grammar payload;
nt start = CONTROL payload;
"#;

    assert!(ConstraintSpec::builder(Grammar::glrm(source), &vocab)
        .unwrap()
        .bind_token("CONTROL", [])
        .is_err());
    assert!(ConstraintSpec::builder(Grammar::glrm(source), &vocab)
        .unwrap()
        .bind_token("payload", [7])
        .is_err());
    // `extern grammar` slots are deliberately allowed to remain unresolved;
    // only exact-token externs must be complete at `build()`.
    let open_spec = ConstraintSpec::builder(Grammar::glrm(source), &vocab)
        .unwrap()
        .bind_token("CONTROL", [7])
        .unwrap()
        .build()
        .unwrap();
    let static_open = open_spec.clone().compile().unwrap();
    let dynamic_open = open_spec.compile_dynamic().unwrap();
    let mut static_state = static_open.start();
    let mut dynamic_state = dynamic_open.start();
    assert_eq!(static_state.mask(), dynamic_state.mask());
    assert!(allowed(&static_state.mask(), 7));
    static_state.commit_token(7).unwrap();
    dynamic_state.commit_token(7).unwrap();
    assert!(!static_state.is_accepting());
    assert!(!dynamic_state.is_accepting());
}

#[test]
fn commit_token_keeps_exact_and_decoded_byte_interpretations() {
    let vocab = Vocab::new(vec![(0, b"x".to_vec()), (1, b"z".to_vec()), (7, b"a".to_vec())]);
    let source = r#"
glrm 1;
start start;
extern token SPECIAL;
nt start = "a" SPECIAL | "x" "a" "z";
"#;
    let constraint = ConstraintSpec::builder(Grammar::glrm(source), &vocab)
        .unwrap()
        .bind_token("SPECIAL", [7])
        .unwrap()
        .build()
        .unwrap()
        .compile()
        .unwrap();

    let mut byte_path = constraint.start();
    byte_path.commit_token(0).unwrap();
    byte_path.commit_token(7).unwrap();
    byte_path.commit_token(1).unwrap();
    assert!(byte_path.is_accepting());

    let mut mixed_path = constraint.start();
    mixed_path.commit_token(7).unwrap();
    mixed_path.commit_token(7).unwrap();
    assert!(mixed_path.is_accepting());
}

#[test]
fn exact_tokens_survive_static_and_dynamic_serialization() {
    let vocab = Vocab::new(vec![(100, Vec::new())]);
    let source = "glrm 1; start start; extern token SPECIAL; nt start = SPECIAL;";
    let spec = ConstraintSpec::builder(Grammar::glrm(source), &vocab)
        .unwrap()
        .bind_token("SPECIAL", [100])
        .unwrap()
        .build()
        .unwrap();

    let compiled = spec.compile().unwrap();
    let loaded = Constraint::load(&compiled.save()).unwrap();
    let mut state = loaded.start();
    state.commit_token(100).unwrap();
    assert!(state.is_accepting());

    let compiled = spec.compile_dynamic().unwrap();
    let loaded = DynamicConstraint::load(&compiled.save()).unwrap();
    let mut state = loaded.start();
    state.commit_token(100).unwrap();
    assert!(state.is_accepting());
}

#[test]
fn compiled_parent_late_binding_matches_monolithic_across_backend_matrix() {
    let vocab = Vocab::new(vec![
        (0, b"x".to_vec()),
        (1, b"y".to_vec()),
        (2, b"xy".to_vec()),
        // Relevant prefix followed by a suffix outside the language.
        (3, b"xyz".to_vec()),
    ]);
    let parent_source =
        "glrm 1; start start; extern grammar child; nt start = \"x\" child;";
    let child_grammar = Grammar::ebnf(r#"start ::= "y""#);
    let reference = Constraint::compile(Grammar::ebnf(r#"start ::= "x" "y""#), &vocab)
        .unwrap();
    let static_child = Constraint::compile(child_grammar.clone(), &vocab).unwrap();
    let dynamic_child = DynamicConstraint::compile(child_grammar, &vocab).unwrap();
    let static_parent = Constraint::compile(Grammar::glrm(parent_source), &vocab).unwrap();
    let dynamic_parent =
        DynamicConstraint::compile(Grammar::glrm(parent_source), &vocab).unwrap();

    assert_static_xy_matches(
        &reference,
        &static_parent.bind_grammar("child", &static_child).unwrap(),
    );
    assert_static_xy_matches(
        &reference,
        &static_parent.bind_grammar("child", &dynamic_child).unwrap(),
    );
    assert_static_xy_matches(
        &reference,
        &static_parent
            .bind_grammar_dynamic_boundary("child", &static_child)
            .unwrap(),
    );
    assert_static_xy_matches(
        &reference,
        &static_parent
            .bind_grammar_dynamic_boundary("child", &dynamic_child)
            .unwrap(),
    );

    assert_dynamic_xy_matches(
        &reference,
        &dynamic_parent.bind_grammar("child", &static_child).unwrap(),
    );
    assert_dynamic_xy_matches(
        &reference,
        &dynamic_parent.bind_grammar("child", &dynamic_child).unwrap(),
    );
    assert_dynamic_xy_matches(
        &reference,
        &dynamic_parent
            .bind_grammar_dynamic_boundary("child", &static_child)
            .unwrap(),
    );
    let dynamic_bound_static_boundary = dynamic_parent.bind_grammar("child", &dynamic_child).unwrap();
    assert_dynamic_xy_matches(&reference, &dynamic_bound_static_boundary);
    let dynamic_bound_dynamic_boundary = dynamic_parent
        .bind_grammar_dynamic_boundary("child", &dynamic_child)
        .unwrap();
    assert_dynamic_xy_matches(&reference, &dynamic_bound_dynamic_boundary);

    let loaded_dynamic_bound_static =
        DynamicConstraint::load(&dynamic_bound_static_boundary.save()).unwrap();
    assert_dynamic_xy_matches(&reference, &loaded_dynamic_bound_static);
    let loaded_dynamic_bound_dynamic =
        DynamicConstraint::load(&dynamic_bound_dynamic_boundary.save()).unwrap();
    assert_dynamic_xy_matches(&reference, &loaded_dynamic_bound_dynamic);

    let loaded_static_parent = UnlinkedConstraint::load(
        Grammar::from_glrm(parent_source)
            .compile_unlinked(&vocab)
            .unwrap()
            .save(),
    )
    .unwrap();
    let loaded_static_bound = loaded_static_parent
        .bind("child", &static_child)
        .unwrap()
        .link()
        .unwrap();
    assert_static_xy_matches(&reference, &loaded_static_bound);

    let error = DynamicConstraint::load(&dynamic_parent.save()).unwrap_err();
    assert!(
        error.to_string().contains("unresolved slots")
            && error.to_string().contains("UnlinkedConstraint"),
        "an unresolved parent must not load as a runnable constraint: {error}",
    );
    let loaded_dynamic_parent = UnlinkedConstraint::load(
        Grammar::from_glrm(parent_source)
            .compile_unlinked(&vocab)
            .unwrap()
            .save(),
    )
    .unwrap();
    let loaded_dynamic_bound = loaded_dynamic_parent
        .bind("child", &static_child)
        .unwrap()
        .link_with(BuildOptions::default().optimization(Optimization::FastBuild))
        .unwrap();
    assert_static_xy_matches(&reference, &loaded_dynamic_bound);

    assert!(static_parent.bind_grammar("missing", &static_child).is_err());
    assert!(dynamic_parent
        .bind_grammar_dynamic_boundary("missing", &dynamic_child)
        .is_err());
}

#[test]
fn unresolved_late_parent_roundtrip_excludes_private_linker_token_from_masks() {
    let vocab = Vocab::new(vec![
        (0, b"x".to_vec()),
        (1, b"y".to_vec()),
        (2, b"xy".to_vec()),
    ]);
    let parent = Grammar::from_glrm(
        "glrm 1; start start; extern grammar child; nt start = \"x\" child;",
    )
    .compile_unlinked(&vocab)
    .unwrap();

    // An unresolved grammar slot belongs to UnlinkedConstraint, never Constraint. Its
    // private linker coordinate survives UnlinkedConstraint persistence without becoming
    // part of the public model-token mask.
    let loaded = UnlinkedConstraint::load(parent.save()).unwrap();
    let child = Grammar::from_ebnf(r#"start ::= "y""#)
        .compile(&vocab)
        .unwrap();
    let reference = Constraint::compile(Grammar::ebnf(r#"start ::= "x" "y""#), &vocab).unwrap();
    let bound = loaded
        .bind("child", &child)
        .unwrap()
        .link()
        .unwrap();
    assert_eq!(bound.mask_len(), 1);
    assert_eq!(bound.start().mask().len(), 1);
    assert_static_xy_matches(&reference, &bound);
}

#[test]
fn late_binding_multiple_adjacent_slots_handles_internal_multi_boundary_tokens() {
    let vocab = Vocab::new(vec![
        (0, b"x".to_vec()),
        (1, b"y".to_vec()),
        (2, b"xy".to_vec()),
        (3, b"z".to_vec()),
        (4, b"yz".to_vec()),
        (5, b"xyz".to_vec()),
        (6, b"xyzz".to_vec()),
    ]);
    let parent = Grammar::glrm(
        "glrm 1; start start; extern grammar left; extern grammar right; \
         nt start = \"x\" left right;",
    );
    let reference =
        Constraint::compile(Grammar::ebnf(r#"start ::= "x" "y" "z""#), &vocab).unwrap();
    let static_left = Constraint::compile(Grammar::ebnf(r#"start ::= "y""#), &vocab).unwrap();
    let dynamic_right =
        DynamicConstraint::compile(Grammar::ebnf(r#"start ::= "z""#), &vocab).unwrap();

    let open = Constraint::compile(parent, &vocab).unwrap();
    let partial = open.bind_grammar("left", &static_left).unwrap();
    let complete = partial
        .bind_grammar_dynamic_boundary("right", &dynamic_right)
        .unwrap();
    assert_static_xyz_matches(&reference, &complete);
    let state = complete.start();
    assert!(!allowed(&state.mask(), 6));

    let complete = partial
        .bind_grammar_dynamic_boundary("right", &dynamic_right)
        .unwrap();
    assert_static_xyz_matches(&reference, &complete);
    let state = complete.start();
    assert!(!allowed(&state.mask(), 6));
}

#[test]
fn dynamic_boundary_cross_prefilter_preserves_scoped_ignore_paths() {
    let vocab = Vocab::new(vec![
        (0, b"X".to_vec()),
        (1, b" ".to_vec()),
        (2, b"\t".to_vec()),
        (3, b"a".to_vec()),
        (4, b"!".to_vec()),
        (5, b"X a !".to_vec()),
        (6, b" \ta".to_vec()),
        (7, b"a\t ".to_vec()),
        (8, b"\t\t".to_vec()),
        (9, b"  ".to_vec()),
        (10, b"a ".to_vec()),
    ]);
    let parent = Constraint::compile(
        Grammar::glrm(
            r#"
                glrm 1;
                start document;
                ignore PARENT_WS;
                t PARENT_WS = " "+;
                extern grammar child;
                nt document = "X" child "!";
            "#,
        ),
        &vocab,
    )
    .unwrap();
    let child = DynamicConstraint::compile(
        Grammar::glrm(
            r#"
                glrm 1;
                start child;
                nt child = "a";
            "#,
        ),
        &vocab,
    )
    .unwrap();
    let reference = Constraint::compile(
        Grammar::glrm(
            r#"
                glrm 1;
                start document;
                ignore PARENT_WS;
                t PARENT_WS = " "+;
                g child = {
                    start child;
                    nt child = "a";
                };
                nt document = "X" child "!";
            "#,
        ),
        &vocab,
    )
    .unwrap();
    let bound = parent
        .bind_grammar_dynamic_boundary("child", &child)
        .unwrap();
    let loaded = Constraint::load(&bound.save()).unwrap();

    for tokens in [
        &[5][..],
        &[0, 3, 4][..],
        &[0, 9, 3, 4][..],
        &[0, 10, 4][..],
    ] {
        let mut expected = reference.start();
        let mut actual = bound.start();
        let mut restored = loaded.start();
        for &token in tokens {
            assert_eq!(actual.mask(), expected.mask(), "bound mask before {tokens:?}");
            assert_eq!(restored.mask(), expected.mask(), "loaded mask before {tokens:?}");
            actual.commit_token(token).unwrap();
            restored.commit_token(token).unwrap();
            expected.commit_token(token).unwrap();
        }
        assert_eq!(actual.is_accepting(), expected.is_accepting(), "bound {tokens:?}");
        assert_eq!(restored.is_accepting(), expected.is_accepting(), "loaded {tokens:?}");
    }
}

#[test]
fn compiled_children_are_authoritative_and_compose_both_directions() {
    let vocab = Vocab::new(vec![(0, Vec::new()), (55, Vec::new()), (56, Vec::new())]);
    let child_source = "glrm 1; start start; extern token VALUE; nt start = VALUE;";
    let child_spec = ConstraintSpec::builder(Grammar::glrm(child_source), &vocab)
        .unwrap()
        .bind_token("VALUE", [55, 56])
        .unwrap()
        .build()
        .unwrap();
    let static_child = child_spec.compile().unwrap();
    let dynamic_child = child_spec.compile_dynamic().unwrap();
    let loaded_dynamic_child = DynamicConstraint::load(&dynamic_child.save()).unwrap();
    let parent_source = r#"
glrm 1;
start document;
extern token MARK;
extern grammar payload;
nt document = MARK payload;
"#;

    for (static_parent, dynamic_parent) in [
        {
            let spec = ConstraintSpec::builder(Grammar::glrm(parent_source), &vocab)
                .unwrap()
                .bind_token("MARK", [0])
                .unwrap()
                .bind_grammar("payload", &static_child)
                .unwrap()
                .build()
                .unwrap();
            (
                spec.compile().unwrap(),
                spec.compile_dynamic().unwrap(),
            )
        },
        {
            let spec = ConstraintSpec::builder(Grammar::glrm(parent_source), &vocab)
                .unwrap()
                .bind_token("MARK", [0])
                .unwrap()
                .bind_grammar("payload", &dynamic_child)
                .unwrap()
                .build()
                .unwrap();
            (
                spec.compile().unwrap(),
                spec.compile_dynamic().unwrap(),
            )
        },
        {
            let spec = ConstraintSpec::builder(Grammar::glrm(parent_source), &vocab)
                .unwrap()
                .bind_token("MARK", [0])
                .unwrap()
                .bind_grammar("payload", &loaded_dynamic_child)
                .unwrap()
                .build()
                .unwrap();
            (
                spec.compile().unwrap(),
                spec.compile_dynamic().unwrap(),
            )
        },
    ] {
        for token in [55, 56] {
            let mut static_state = static_parent.start();
            let mut dynamic_state = dynamic_parent.start();
            static_state.commit_token(0).unwrap();
            dynamic_state.commit_token(0).unwrap();
            static_state.commit_token(token).unwrap();
            dynamic_state.commit_token(token).unwrap();
            assert!(static_state.is_accepting());
            assert!(dynamic_state.is_accepting());
        }
    }
}

#[test]
fn loaded_direct_regular_dynamic_child_remains_composable() {
    let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"aa".to_vec())]);
    let dynamic_child = DynamicConstraint::compile(
        Grammar::lark("start: /a+/"),
        &vocab,
    )
    .unwrap();
    let loaded_child = DynamicConstraint::load(&dynamic_child.save()).unwrap();
    let parent_source = "glrm 1; start start; extern grammar payload; nt start = payload;";
    let spec = ConstraintSpec::builder(Grammar::glrm(parent_source), &vocab)
        .unwrap()
        .bind_grammar("payload", &loaded_child)
        .unwrap()
        .build()
        .unwrap();
    let static_parent = spec.compile().unwrap();
    let dynamic_parent = spec.compile_dynamic().unwrap();
    for token in [0, 1] {
        let mut static_state = static_parent.start();
        let mut dynamic_state = dynamic_parent.start();
        static_state.commit_token(token).unwrap();
        dynamic_state.commit_token(token).unwrap();
        assert!(static_state.is_accepting());
        assert!(dynamic_state.is_accepting());
    }
}

#[test]
fn grammar_can_bind_source_subgrammar_before_target_selection() {
    let vocab = Vocab::new(vec![(0, b"x".to_vec()), (1, b"y".to_vec())]);
    let parent = Grammar::glrm(
        "glrm 1; start start; extern grammar payload; nt start = payload;",
    )
    .bind_grammar("payload", Grammar::ebnf(r#"start ::= "x" | "y""#))
    .unwrap();

    let static_constraint =
        Constraint::compile(parent.clone(), &vocab).unwrap();
    let dynamic_constraint =
        DynamicConstraint::compile(parent, &vocab).unwrap();

    for token in [0, 1] {
        let mut static_state = static_constraint.start();
        let mut dynamic_state = dynamic_constraint.start();
        static_state.commit_token(token).unwrap();
        dynamic_state.commit_token(token).unwrap();
        assert!(static_state.is_accepting());
        assert!(dynamic_state.is_accepting());
    }
}

#[test]
fn grammar_source_bindings_can_nest_and_mix_with_constraintspec_token_bindings() {
    let vocab = Vocab::new(vec![(7, Vec::new()), (8, b"z".to_vec())]);
    let leaf = Grammar::ebnf(r#"start ::= "z""#);
    let child = Grammar::glrm(
        "glrm 1; start start; extern grammar leaf; nt start = leaf;",
    )
    .bind_grammar("leaf", leaf)
    .unwrap();
    let parent = Grammar::glrm(
        "glrm 1; start start; extern token MARK; extern grammar child; nt start = MARK child;",
    )
    .bind_grammar("child", child)
    .unwrap();

    let spec = ConstraintSpec::builder(parent, &vocab)
        .unwrap()
        .bind_token("MARK", [7])
        .unwrap()
        .build()
        .unwrap();
    let constraint = spec.compile_dynamic().unwrap();
    let mut state = constraint.start();
    state.commit_token(7).unwrap();
    state.commit_token(8).unwrap();
    assert!(state.is_accepting());
}

#[test]
fn grammar_source_binding_accepts_shorter_lived_child_source() {
    let child_source = String::from(r#"start ::= "x""#);
    let _grammar = Grammar::glrm(
        "glrm 1; start start; extern grammar payload; nt start = payload;",
    )
    .bind_grammar("payload", Grammar::ebnf(&child_source))
    .unwrap();
}

#[test]
fn grammar_source_bindings_validate_parent_declarations() {
    let wrong_kind = Grammar::glrm(
        "glrm 1; start start; extern token X; nt start = X;",
    )
    .bind_grammar("X", Grammar::ebnf(r#"start ::= "x""#))
    .unwrap_err()
    .to_string();
    assert!(wrong_kind.contains("kind token, not grammar"), "{wrong_kind}");

    let unknown = Grammar::glrm("glrm 1; start start; nt start = \"x\";")
        .bind_grammar("missing", Grammar::ebnf(r#"start ::= "x""#))
        .unwrap_err()
        .to_string();
    assert!(unknown.contains("no external grammar"), "{unknown}");
}

#[test]
fn bind_grammar_accepts_source_and_spec_and_does_not_inherit_parent_bindings() {
    let vocab = Vocab::new(vec![(0, b"x".to_vec())]);
    let parent = "glrm 1; start start; extern grammar child; nt start = child;";
    let source_child = Grammar::ebnf(r#"start ::= "x""#);
    let source_bound = ConstraintSpec::builder(Grammar::glrm(parent), &vocab)
        .unwrap()
        .bind_grammar("child", source_child.clone())
        .unwrap()
        .build()
        .unwrap()
        .compile()
        .unwrap();
    let mut state = source_bound.start();
    state.commit_token(0).unwrap();
    assert!(state.is_accepting());

    let child_spec = ConstraintSpec::builder(source_child, &vocab)
        .unwrap()
        .build()
        .unwrap();
    assert!(ConstraintSpec::builder(Grammar::glrm(parent), &vocab)
        .unwrap()
        .bind_grammar("child", child_spec)
        .unwrap()
        .build()
        .is_ok());

    let unresolved_grammar_child = Grammar::from_glrm(
        "glrm 1; start start; extern grammar nested; nt start = nested;",
    );
    let open = Grammar::from_glrm(parent)
        .bind("child", unresolved_grammar_child)
        .unwrap()
        .compile_unlinked(&vocab)
        .unwrap()
        ;
    // Independent open components may reuse the same private sentinel token
    // number. Terminal IDs, not those hidden token IDs, are the linker
    // coordinate; qualified nested slots must survive serialization and bind.
    let open = UnlinkedConstraint::load(open.save()).unwrap();
    let leaf = Grammar::from_ebnf(r#"start ::= "x""#)
        .compile(&vocab)
        .unwrap();
    let fully_bound = open
        .bind("child.nested", &leaf)
        .unwrap()
        .link()
        .unwrap();
    let mut state = fully_bound.start();
    state.commit_token(0).unwrap();
    assert!(state.is_accepting());

    let dynamic_fully_bound = open
        .bind("child.nested", &leaf)
        .unwrap()
        .link_with(BuildOptions::default().optimization(Optimization::FastBuild))
        .unwrap();
    let mut state = dynamic_fully_bound.start();
    state.commit_token(0).unwrap();
    assert!(state.is_accepting());

    let unresolved_token_child = Grammar::glrm(
        "glrm 1; start start; extern token TOKEN; nt start = TOKEN;",
    );
    let spec = ConstraintSpec::builder(Grammar::glrm(parent), &vocab)
        .unwrap()
        .bind_grammar("child", unresolved_token_child)
        .unwrap()
        .build()
        .unwrap();
    assert!(spec.compile().is_err());
}

#[test]
fn incompatible_compiled_child_target_is_rejected_at_bind_time() {
    let child_vocab = Vocab::new(vec![(0, b"a".to_vec())]);
    let parent_vocab = Vocab::new(vec![(0, b"b".to_vec())]);
    let child = Constraint::compile(
        Grammar::ebnf(r#"start ::= "a""#),
        &child_vocab,
    )
    .unwrap();
    let parent = "glrm 1; start start; extern grammar child; nt start = child;";
    assert!(ConstraintSpec::builder(Grammar::glrm(parent), &parent_vocab)
        .unwrap()
        .bind_grammar("child", &child)
        .is_err());
}

#[test]
fn v1_terminal_fa_compiles_and_masks_tokens() {
    let vocab = Vocab::new(vec![(0, b"ab".to_vec()), (1, b"a".to_vec()), (2, b"b".to_vec())]);
    let source = r#"
glrm 1;
start start;
t WORD = fa {
    start begin;
    accept done;
    begin -> middle: "a";
    middle -> done: "b";
};
nt start = WORD;
"#;
    let constraint = Constraint::compile(
        Grammar::glrm(source),
        &vocab,
    )
    .unwrap();
    let mut state = constraint.start();
    assert!(allowed(&state.mask(), 0));
    assert!(allowed(&state.mask(), 1));
    state.commit_token(0).unwrap();
    assert!(state.is_accepting());
}

#[test]
fn nested_external_tokens_use_qualified_names() {
    let vocab = Vocab::new(vec![(123, Vec::new())]);
    let source = r#"
glrm 1;
start document;
g inner = {
    start value;
    extern token END;
    nt value = END;
};
nt document = inner;
"#;
    let constraint = ConstraintSpec::builder(Grammar::glrm(source), &vocab)
        .unwrap()
        .bind_token("inner::END", [123])
        .unwrap()
        .build()
        .unwrap()
        .compile()
        .unwrap();
    let mut state = constraint.start();
    state.commit_token(123).unwrap();
    assert!(state.is_accepting());
}

#[test]
fn external_token_can_label_nonterminal_fa_edges() {
    let vocab = Vocab::new(vec![(77, Vec::new())]);
    let source = r#"
glrm 1;
start start;
extern token END;
nt start = fa {
    start begin;
    accept done;
    begin -> done: END;
};
"#;
    let spec = ConstraintSpec::builder(Grammar::glrm(source), &vocab)
        .unwrap()
        .bind_token("END", [77])
        .unwrap()
        .build()
        .unwrap();
    assert!(allowed(&spec.compile().unwrap().start().mask(), 77));
    assert!(allowed(&spec.compile_dynamic().unwrap().start().mask(), 77));
}

#[test]
fn lexer_pragma_does_not_change_token_language() {
    let vocab = Vocab::new(vec![(0, b"abc".to_vec()), (1, b"a".to_vec()), (2, b"bc".to_vec())]);
    let plain = "glrm 1; start start; t WORD = /[a-z]+/; nt start = WORD;";
    let hinted = r#"
glrm 1;
start start;
pragma glrmask { lexer group words = WORD; }
t WORD = /[a-z]+/;
nt start = WORD;
"#;
    let plain = Constraint::compile(Grammar::glrm(plain), &vocab).unwrap();
    let hinted = Constraint::compile(Grammar::glrm(hinted), &vocab).unwrap();
    let mut left = plain.start();
    let mut right = hinted.start();
    assert_eq!(left.mask(), right.mask());
    left.commit_token(1).unwrap();
    right.commit_token(1).unwrap();
    assert_eq!(left.mask(), right.mask());
}
