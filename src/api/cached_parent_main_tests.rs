    use super::*;

    fn vocab() -> Vocab {
        Vocab::new(vec![
            (0, b"<".to_vec()),
            (1, b">".to_vec()),
            (2, b"[".to_vec()),
            (3, b"]".to_vec()),
            (4, b"a".to_vec()),
            (5, b"b".to_vec()),
            (6, b"x".to_vec()),
            (7, b"<a>".to_vec()),
            (8, b"<b>".to_vec()),
            (9, b"<[a]>".to_vec()),
        ])
    }

    fn accepts(constraint: &RuntimeConstraint, bytes: &[u8]) -> bool {
        let mut state = constraint.start();
        state.commit_bytes(bytes).is_ok() && state.is_accepting()
    }

    #[test]
    fn compiled_parent_can_bind_external_grammar_after_compile_and_load() {
        let vocab = vocab();
        let parent = RuntimeConstraint::compile(
            Grammar::glrm(
                r#"
                    glrm 1;
                    extern grammar payload;
                    start document;
                    nt document = "x" | "<" payload ">";
                "#,
            ),
            &vocab,
        )
        .unwrap();
        assert_eq!(
            parent
                .late_grammar_slots
                .iter()
                .map(|slot| slot.name.as_str())
                .collect::<Vec<_>>(),
            vec!["payload"],
        );
        assert!(accepts(&parent, b"x"));
        assert!(!accepts(&parent, b"<a>"));

        let child_a = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start value; nt value = \"a\";"),
            &vocab,
        )
        .unwrap();
        let child_b = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start value; nt value = \"b\";"),
            &vocab,
        )
        .unwrap();

        let with_a = parent.bind_grammar("payload", &child_a).unwrap();
        let with_b = parent.bind_grammar("payload", &child_b).unwrap();
        assert!(accepts(&with_a, b"x"));
        assert!(accepts(&with_a, b"<a>"));
        assert!(!accepts(&with_a, b"<b>"));
        assert!(accepts(&with_b, b"<b>"));
        assert!(!accepts(&with_b, b"<a>"));
        assert!(parent
            .late_grammar_slots
            .iter()
            .any(|slot| slot.name == "payload"));

        let saved = parent.save();
        let loaded = RuntimeConstraint::load_body_artifact(&saved).unwrap();
        let loaded_with_a = loaded.bind_grammar("payload", &child_a).unwrap();
        assert!(accepts(&loaded_with_a, b"<a>"));
        assert!(!accepts(&loaded_with_a, b"<b>"));
    }

    #[test]
    fn compiled_parent_can_fill_multiple_slots_incrementally() {
        let vocab = vocab();
        let parent = RuntimeConstraint::compile(
            Grammar::glrm(
                r#"
                    glrm 1;
                    extern grammar left;
                    extern grammar right;
                    start document;
                    nt document = "<" left right ">";
                "#,
            ),
            &vocab,
        )
        .unwrap();
        let a = RuntimeConstraint::compile(
            Grammar::glrm(r#"glrm 1; start value; nt value = "a";"#),
            &vocab,
        )
        .unwrap();
        let b = RuntimeConstraint::compile(
            Grammar::glrm(r#"glrm 1; start value; nt value = "b";"#),
            &vocab,
        )
        .unwrap();
        let half = parent.bind_grammar_dynamic_boundary("left", &a).unwrap();
        assert!(half
            .late_grammar_slots
            .iter()
            .any(|slot| slot.name == "right"));
        let full = half.bind_grammar_dynamic_boundary("right", &b).unwrap();
        assert!(full.late_grammar_slots.is_empty());
        assert!(accepts(&full, b"<ab>"));

        let unresolved_child = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; extern grammar leaf; start value; nt value = leaf;",
            ),
            &vocab,
        )
        .unwrap();
        // Main used to reject this because independently compiled unresolved
        // children could collide in the private placeholder-token namespace.
        // The successor architecture transports those slots by qualified name
        // and sanitizes the private linker token from the public token domain.
        let open_left = parent
            .bind_grammar_dynamic_boundary("left", &unresolved_child)
            .unwrap();
        assert!(open_left
            .late_grammar_slots
            .iter()
            .any(|slot| slot.name == "left.leaf"));
        let left = open_left.bind_grammar_dynamic_boundary("left.leaf", &a).unwrap();
        let nested_full = left.bind_grammar_dynamic_boundary("right", &b).unwrap();
        assert!(nested_full.late_grammar_slots.is_empty());
        assert!(accepts(&nested_full, b"<ab>"));
    }

    #[test]
    fn dynamic_compile_preserves_unbound_external_grammar() {
        let vocab = vocab();
        let open = DynamicConstraint::compile(
            Grammar::glrm(
                "glrm 1; extern grammar payload; start document; nt document = payload;",
            ),
            &vocab,
        )
        .unwrap();
        assert!(open
            .clone_constraints()
            .iter()
            .all(|alternative| alternative
                .late_grammar_slots
                .iter()
                .any(|slot| slot.name == "payload")));
        let child = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start value; nt value = \"a\";"),
            &vocab,
        )
        .unwrap();
        let bound = open.bind_grammar("payload", &child).unwrap();
        let mut state = bound.start();
        state.commit_bytes(b"a").unwrap();
        assert!(state.is_accepting());
    }

    #[test]
    fn final_optimization_dispatch_selects_dynamic_build_and_static_runtime_paths() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"ab".to_vec()),
        ]);
        let grammar = Grammar::ebnf(r#"start ::= "a" "b""#);
        let fast_build = grammar
            .compile_with(
                &vocab,
                BuildOptions::default().optimization(Optimization::FastBuild),
            )
            .unwrap();
        let fast_runtime = grammar
            .compile_with(
                &vocab,
                BuildOptions::default().optimization(Optimization::FastRuntime),
            )
            .unwrap();
        assert!(fast_build.uses_dynamic_runtime());
        assert!(!fast_runtime.uses_dynamic_runtime());

        let mut build_state = fast_build.start();
        let mut runtime_state = fast_runtime.start();
        assert_eq!(build_state.mask(), runtime_state.mask());
        build_state.commit_token(2).unwrap();
        runtime_state.commit_token(2).unwrap();
        assert_eq!(build_state.is_accepting(), runtime_state.is_accepting());

        let loaded = RuntimeConstraint::load(fast_build.save()).unwrap();
        assert!(loaded.uses_dynamic_runtime());
        assert_eq!(loaded.start().mask(), fast_runtime.start().mask());
    }
