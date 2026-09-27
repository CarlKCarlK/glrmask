    use super::*;

    fn assert_static_boundary(constraint: &RuntimeConstraint) {
        let overlay = constraint.static_dynamic_overlay.as_ref().unwrap();
        assert!(!overlay.segmented_boundary_shards.is_empty());
        assert!(overlay.segmented_boundary_shards.iter().all(|shard| matches!(
            shard.backend,
            crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_)
        )));
    }

    fn assert_dynamic_boundary(constraint: &RuntimeConstraint) {
        let overlay = constraint.static_dynamic_overlay.as_ref().unwrap();
        assert!(!overlay.segmented_boundary_shards.is_empty());
        assert!(overlay.segmented_boundary_shards.iter().all(|shard| matches!(
            shard.backend,
            crate::runtime::SegmentedBoundaryShardBackend::DynamicDirect
        )));
    }

    #[test]
    fn unlinked_bind_defers_composition_until_link_and_link_optimization_selects_boundary() {
        let vocab = Vocab::new(vec![
            (0, b"x".to_vec()),
            (1, b"y".to_vec()),
            (2, b"xy".to_vec()),
        ]);
        let host = Grammar::glrm(
            r#"glrm 1; start start; extern grammar child; nt start = "x" child;"#,
        )
        .compile_unlinked(&vocab)
        .unwrap();
        let child = Grammar::ebnf(r#"start ::= "y""#)
            .compile(&vocab)
            .unwrap();

        let bound = host.bind("child", &child).unwrap();
        assert!(bound.inner.late_grammar_slots.iter().any(|slot| slot.name == "child"));
        assert!(bound.inner.static_dynamic_overlay.is_none(), "bind must not compile a boundary eagerly");

        let fast_build = bound.link_with(
            BuildOptions::default().optimization(Optimization::FastBuild),
        ).unwrap();
        let fast_runtime = bound.link_with(
            BuildOptions::default().optimization(Optimization::FastRuntime),
        ).unwrap();
        assert_dynamic_boundary(&fast_build);
        assert_static_boundary(&fast_runtime);
        assert_eq!(fast_build.start().mask(), fast_runtime.start().mask());

        let loaded = UnlinkedConstraint::load(bound.save()).unwrap();
        assert!(loaded.inner.late_grammar_slots.iter().any(|slot| slot.name == "child"));
        assert_dynamic_boundary(&loaded.link_with(
            BuildOptions::default().optimization(Optimization::FastBuild),
        ).unwrap());
        assert_static_boundary(&loaded.link_with(
            BuildOptions::default().optimization(Optimization::FastRuntime),
        ).unwrap());
    }
