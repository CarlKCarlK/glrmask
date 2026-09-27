    use super::*;
    use crate::automata::lexer::tokenizer::Lexer;

    fn token_allowed(mask: &[u32], token_id: u32) -> bool {
        let word = token_id as usize / 32;
        let bit = token_id % 32;
        mask.get(word)
            .is_some_and(|bits| bits & (1u32 << bit) != 0)
    }

    #[test]
    fn vocab_partition_covers_sparse_vocab_and_merges_identical_tokens() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (3, b"a".to_vec()),
            (7, b"b".to_vec()),
            (11, b"ab".to_vec()),
            (67, b"a".to_vec()),
        ]);
        let partition = VocabPartition::compile(Grammar::ebnf(r#"start ::= "a"+"#), &vocab)
            .unwrap();

        assert_eq!(partition.class_of(1), None);
        for token in [0, 3, 7, 11, 67] {
            assert!(partition.class_of(token).is_some(), "token {token} is unmapped");
        }
        assert_eq!(partition.class_of(0), partition.class_of(3));
        assert_eq!(partition.class_of(0), partition.class_of(67));
        let mut covered = partition.classes().iter().flatten().copied().collect::<Vec<_>>();
        covered.sort_unstable();
        assert_eq!(covered, vec![0, 3, 7, 11, 67]);
        for class in 0..partition.num_classes() as u32 {
            let representative = partition.representative(class).unwrap();
            assert_eq!(partition.class_of(representative), Some(class));
        }

        assert_eq!(partition.internal_mask_len(), partition.num_classes().div_ceil(64));
        assert_eq!(partition.original_mask_len(), 3);

        let class = partition.class_of(0).unwrap() as usize;
        let mut internal = vec![0u64; partition.internal_mask_len()];
        internal[class / 64] |= 1u64 << (class % 64);
        let expanded = partition.expand_mask(&internal);
        assert_ne!(expanded[0] & (1u32 << 0), 0);
        assert_ne!(expanded[0] & (1u32 << 3), 0);
        assert_eq!(expanded[0] & (1u32 << 1), 0);
        assert_ne!(expanded[2] & (1u32 << 3), 0);

        let mut reused = vec![u32::MAX; partition.original_mask_len() + 2];
        partition.fill_expanded_mask(&internal, &mut reused);
        assert_eq!(&reused[..partition.original_mask_len()], expanded.as_slice());
        assert_eq!(&reused[partition.original_mask_len()..], &[0, 0]);
    }

    #[test]
    fn vocab_partition_expands_multiple_classes_and_ignores_high_internal_bits() {
        let partition = VocabPartition {
            original_to_class: vec![0, u32::MAX, 1, 0, u32::MAX, 2, 2, u32::MAX, 1],
            classes: vec![vec![0, 3], vec![2, 8], vec![5, 6]],
            representatives: vec![0, 2, 5],
            class_output_masks: vec![
                vec![(0, (1u32 << 0) | (1u32 << 3))],
                vec![(0, (1u32 << 2) | (1u32 << 8))],
                vec![(0, (1u32 << 5) | (1u32 << 6))],
            ],
        };

        let expanded = partition.expand_mask(&[(1u64 << 0) | (1u64 << 2) | (1u64 << 63)]);
        assert_eq!(expanded, vec![(1u32 << 0) | (1u32 << 3) | (1u32 << 5) | (1u32 << 6)]);
    }

    #[test]
    fn vocab_partition_supports_json_schema_frontend() {
        let vocab = Vocab::new(vec![
            (0, b"null".to_vec()),
            (1, b"true".to_vec()),
            (2, b"false".to_vec()),
            (3, b"0".to_vec()),
            (4, b"x".to_vec()),
        ]);
        let partition = VocabPartition::compile(
            Grammar::json_schema(r#"{"type":["null","boolean"]}"#),
            &vocab,
        )
        .unwrap();
        assert!(partition.num_classes() > 0);
        assert!(partition.num_classes() <= vocab.len());
    }

    #[test]
    fn explicit_vocab_partition_strategies_are_conservative() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"ab".to_vec()),
            (3, b"ba".to_vec()),
            (4, b"aa".to_vec()),
            (5, b"bb".to_vec()),
        ]);
        let source = r#"start ::= ("a" | "b")+"#;
        let compact = VocabPartition::compile_with_strategy(
            Grammar::ebnf(source),
            &vocab,
            VocabPartitionStrategy::Compact,
        )
        .unwrap();
        let dedicated = VocabPartition::compile_with_strategy(
            Grammar::ebnf(source),
            &vocab,
            VocabPartitionStrategy::Dedicated,
        )
        .unwrap();

        for class in dedicated.classes() {
            let mut compact_class = None;
            for &token_id in class {
                let current = compact.class_of(token_id).unwrap();
                assert!(compact_class.is_none_or(|expected| expected == current));
                compact_class = Some(current);
            }
        }
    }
    #[test]
    fn vocab_partition_rejects_bound_subgrammar_instead_of_ignoring_it() {
        let vocab = Vocab::new(vec![(0, b"a".to_vec())]);
        let grammar = Grammar::glrm(
            "glrm 1; extern grammar child; start root; nt root = child;",
        )
        .bind_grammar("child", Grammar::ebnf(r#"start ::= "a""#))
        .unwrap();
        let error = VocabPartition::compile(grammar, &vocab).unwrap_err();
        assert!(error.to_string().contains("bound grammar values"));
    }

    #[test]
    fn partition_optimized_dynamic_mask_matches_ordinary_dynamic() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"aa".to_vec()),
            (3, b"bb".to_vec()),
            (4, b"ab".to_vec()),
            (5, b"ba".to_vec()),
            (6, b"x".to_vec()),
        ]);
        let grammar_source = "start ::= [ab]+".to_owned();
        let grammar = Grammar::ebnf(&grammar_source);
        let partition = VocabPartition::compile(grammar.clone(), &vocab).unwrap();
        assert_eq!(partition.class_of(0), partition.class_of(1));
        assert!(partition.num_classes() < vocab.len());

        let ordinary = DynamicConstraint::compile(grammar.clone(), &vocab).unwrap();
        let optimized =
            DynamicConstraint::compile_with_vocab_partition(grammar, &vocab).unwrap();

        assert_eq!(optimized.inner.token_bytes_count(), vocab.len());
        assert!(
            optimized
                .inner
                .dynamic_mask_vocab_for_runtime()
                .is_grammar_quotiented(),
            "O2 must not silently fall back to ordinary dynamic for tiny grammars",
        );
        assert!(
            optimized
                .inner
                .dynamic_mask_vocab_for_runtime()
                .canonical_token_count()
                < ordinary
                    .inner
                    .dynamic_mask_vocab_for_runtime()
                    .canonical_token_count(),
            "optimized dynamic runtime did not reduce the mask vocabulary",
        );

        let ordinary_start = ordinary.start();
        let optimized_start = optimized.start();
        assert_eq!(ordinary_start.mask(), optimized_start.mask());

        for token in [0u32, 1, 2, 3, 4, 5] {
            let mut ordinary_state = ordinary.start();
            let mut optimized_state = optimized.start();
            ordinary_state.commit_token(token).unwrap();
            optimized_state.commit_token(token).unwrap();
            assert_eq!(ordinary_state.is_accepting(), optimized_state.is_accepting());
            assert_eq!(ordinary_state.mask(), optimized_state.mask(), "token {token}");
        }
    }

    #[test]
    fn singleton_literal_partition_matches_dynamic_masks_exactly() {
        let vocab = Vocab::new(vec![
            (0, b"x".to_vec()),
            (1, b"{".to_vec()),
            (2, b"}".to_vec()),
            (3, b"{}".to_vec()),
            (4, b"{}x".to_vec()),
            (5, b"{".to_vec()),
            (6, b"x{}".to_vec()),
        ]);
        let grammar_source = r#"start ::= "{" "}""#.to_owned();
        let grammar = Grammar::ebnf(&grammar_source);
        let partition = VocabPartition::compile(grammar.clone(), &vocab).unwrap();

        assert_eq!(partition.num_classes(), 4);
        assert_eq!(partition.class_of(0), partition.class_of(4));
        assert_eq!(partition.class_of(0), partition.class_of(6));
        assert_eq!(partition.class_of(1), partition.class_of(5));
        assert_ne!(partition.class_of(1), partition.class_of(2));
        assert_ne!(partition.class_of(1), partition.class_of(3));
        assert_ne!(partition.class_of(2), partition.class_of(3));

        let ordinary = DynamicConstraint::compile(grammar.clone(), &vocab).unwrap();
        let optimized =
            DynamicConstraint::compile_with_vocab_partition(grammar, &vocab).unwrap();

        let ordinary_start = ordinary.start();
        let optimized_start = optimized.start();
        assert_eq!(ordinary_start.mask(), optimized_start.mask());
        for token in [0u32, 1, 2, 3, 4, 5, 6] {
            assert_eq!(
                token_allowed(&ordinary_start.mask(), token),
                token_allowed(&optimized_start.mask(), token),
                "start mask differs for token {token}",
            );
        }

        for open in [1u32, 5] {
            let mut ordinary_state = ordinary.start();
            let mut optimized_state = optimized.start();
            ordinary_state.commit_token(open).unwrap();
            optimized_state.commit_token(open).unwrap();
            assert_eq!(ordinary_state.is_accepting(), optimized_state.is_accepting());
            assert_eq!(ordinary_state.mask(), optimized_state.mask());
        }

        let mut ordinary_done = ordinary.start();
        let mut optimized_done = optimized.start();
        ordinary_done.commit_token(3).unwrap();
        optimized_done.commit_token(3).unwrap();
        assert!(ordinary_done.is_accepting());
        assert!(optimized_done.is_accepting());
        assert_eq!(ordinary_done.mask(), optimized_done.mask());

        // The production singleton path uses the flat alias representation and
        // deliberately defers its first transfer-artifact serialization. Verify
        // that both survive the real external-vocab save/load boundary.
        let saved = optimized.save_with_external_vocab();
        let loaded = DynamicConstraint::load_with_vocab(&saved, &vocab).unwrap();
        assert_eq!(loaded.start().mask(), optimized.start().mask());
        let mut loaded_done = loaded.start();
        loaded_done.commit_token(3).unwrap();
        assert!(loaded_done.is_accepting());
    }

    #[test]
    fn singleton_literal_partition_matches_all_reachable_cross_terminal_prefixes() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (2, b"b".to_vec()),
            (5, b"c".to_vec()),
            (9, b"d".to_vec()),
            (12, b"ab".to_vec()),
            (18, b"bc".to_vec()),
            (24, b"cd".to_vec()),
            (31, b"abc".to_vec()),
            (37, b"bcd".to_vec()),
            (44, b"abcd".to_vec()),
            (51, b"x".to_vec()),
            (58, b"bd".to_vec()),
            (63, b"abcx".to_vec()),
            (71, b"bc".to_vec()),
        ]);
        let token_ids = vocab.iter().map(|(token_id, _)| token_id).collect::<Vec<_>>();
        let grammar_source = r#"start ::= "ab" "cd""#.to_owned();
        let grammar = Grammar::ebnf(&grammar_source);

        // Compile ordinary first so the optimized compile also exercises the
        // cached byte-order lookup used by prepared production vocabularies.
        let ordinary = DynamicConstraint::compile(grammar.clone(), &vocab).unwrap();
        let optimized =
            DynamicConstraint::compile_with_vocab_partition(grammar, &vocab).unwrap();

        let mut pending = vec![Vec::<u32>::new()];
        let mut seen = std::collections::BTreeSet::<Vec<u32>>::new();
        seen.insert(Vec::new());
        while let Some(prefix) = pending.pop() {
            let mut ordinary_state = ordinary.start();
            let mut optimized_state = optimized.start();
            for &token in &prefix {
                ordinary_state.commit_token(token).unwrap();
                optimized_state.commit_token(token).unwrap();
            }

            assert_eq!(
                ordinary_state.is_accepting(),
                optimized_state.is_accepting(),
                "acceptance differs after token prefix {prefix:?}",
            );
            let ordinary_mask = ordinary_state.mask();
            let optimized_mask = optimized_state.mask();
            assert_eq!(
                ordinary_mask, optimized_mask,
                "mask differs after token prefix {prefix:?}",
            );

            for &token in &token_ids {
                let ordinary_allowed = token_allowed(&ordinary_mask, token);
                let optimized_allowed = token_allowed(&optimized_mask, token);
                assert_eq!(
                    ordinary_allowed, optimized_allowed,
                    "token {token} differs after token prefix {prefix:?}",
                );
                if ordinary_allowed {
                    let mut next = prefix.clone();
                    next.push(token);
                    // Four one-byte tokens can consume the complete language;
                    // no valid path can require a deeper token prefix.
                    if next.len() <= 4 && seen.insert(next.clone()) {
                        pending.push(next);
                    }
                }
            }
        }
    }

    #[test]
    fn partition_optimized_dynamic_save_load_preserves_quotient_mask() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"aa".to_vec()),
            (3, b"bb".to_vec()),
            (4, b"x".to_vec()),
        ]);
        let grammar_source = format!(
            "start ::= [ab]+{}",
            (0..17)
                .map(|index| format!(r#" | "unused_{index}""#))
                .collect::<String>()
        );
        let optimized =
            DynamicConstraint::compile_with_vocab_partition(Grammar::ebnf(&grammar_source), &vocab)
                .unwrap();
        let original_canonical = optimized
            .inner
            .dynamic_mask_vocab_for_runtime()
            .canonical_token_count();
        assert!(original_canonical < vocab.len());
        let bytes = optimized.save_with_external_vocab();
        let loaded = DynamicConstraint::load_with_vocab(&bytes, &vocab).unwrap();
        assert_eq!(loaded.start().mask(), optimized.start().mask());
        assert_eq!(
            loaded
                .inner
                .dynamic_mask_vocab_for_runtime()
                .canonical_token_count(),
            original_canonical,
            "save/load must retain the representative vocabulary rather than rebuilding the full vocabulary",
        );
    }

    #[test]
    fn partition_optimized_bounded_string_schema_does_not_fall_back_to_ordinary_dynamic() {
        let vocab = Vocab::new(vec![
            (0, b"\"abcdefgh\"".to_vec()),
            (1, b"\"ijklmnop\"".to_vec()),
            (2, b"true".to_vec()),
            (3, b"false".to_vec()),
            (4, b"{".to_vec()),
            (5, b"}".to_vec()),
            (6, b":".to_vec()),
            (7, b",".to_vec()),
            (8, b" ".to_vec()),
        ]);
        let mut properties = serde_json::Map::new();
        properties.insert(
            "bounded".to_owned(),
            serde_json::json!({"type":"string", "minLength":8, "maxLength":8}),
        );
        for index in 0..20 {
            properties.insert(format!("flag_{index}"), serde_json::json!({"type":"boolean"}));
        }
        let schema = serde_json::json!({
            "type":"object",
            "properties": properties,
            "additionalProperties": false
        })
        .to_string();

        let optimized = DynamicConstraint::compile_with_vocab_partition(
            Grammar::json_schema(&schema),
            &vocab,
        )
        .unwrap();
        assert!(
            optimized
                .inner
                .dynamic_mask_vocab_for_runtime()
                .is_grammar_quotiented(),
            "O2 must retain its grammar quotient for bounded-string residual schemas",
        );

        let bytes = optimized.save_with_external_vocab();
        let loaded = DynamicConstraint::load_with_vocab(&bytes, &vocab).unwrap();
        assert!(
            loaded
                .inner
                .dynamic_mask_vocab_for_runtime()
                .is_grammar_quotiented(),
            "external-vocab transfer must preserve the O2 grammar quotient",
        );
        assert_eq!(loaded.start().mask(), optimized.start().mask());
    }

    #[test]
    fn partition_optimized_plain_bounded_string_builds_a_real_quotient() {
        let vocab = Vocab::new(vec![
            (0, b"\"".to_vec()),
            (1, b"a".to_vec()),
            (2, b"b".to_vec()),
            (3, b"aa".to_vec()),
            (4, b"bb".to_vec()),
            (5, b"ab".to_vec()),
            (6, b"ba".to_vec()),
            (7, b"\"a".to_vec()),
            (8, b"\"b".to_vec()),
            (9, b"x".to_vec()),
        ]);
        let schema = r#"{"type":"string","maxLength":100}"#;

        let ordinary =
            DynamicConstraint::compile(Grammar::json_schema(schema), &vocab).unwrap();
        let optimized = DynamicConstraint::compile_with_vocab_partition(
            Grammar::json_schema(schema),
            &vocab,
        )
        .unwrap();

        let optimized_vocab = optimized.inner.dynamic_mask_vocab_for_runtime();
        assert!(optimized_vocab.is_grammar_quotiented());
        assert!(
            optimized_vocab.canonical_token_count() < vocab.len(),
            "bounded-string O2 must not degrade to the identity vocabulary quotient",
        );
        assert_eq!(ordinary.start().mask(), optimized.start().mask());

        let mut ordinary_state = ordinary.start();
        let mut optimized_state = optimized.start();
        ordinary_state.commit_token(0).unwrap();
        optimized_state.commit_token(0).unwrap();
        assert_eq!(ordinary_state.mask(), optimized_state.mask());
    }

    #[test]
    fn partition_optimized_multi_alternative_roundtrips_without_sharing_one_quotient() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"aa".to_vec()),
            (3, b"bb".to_vec()),
            (4, b"ab".to_vec()),
            (5, b"ba".to_vec()),
            (6, b"x".to_vec()),
        ]);
        let first = DynamicConstraint::compile_with_vocab_partition(
            Grammar::ebnf(r#"start ::= [ab]+"#),
            &vocab,
        )
        .unwrap();
        let second = DynamicConstraint::compile_with_vocab_partition(
            Grammar::ebnf(r#"start ::= "a"+"#),
            &vocab,
        )
        .unwrap();
        let optimized = DynamicConstraint::from_alternatives(vec![first, second]);
        assert_eq!(optimized.clone_constraints().len(), 2);
        let expected_mask = optimized.start().mask();

        // The self-contained wire has one shared vocab accelerator, so this
        // case intentionally drops the per-alternative quotients and rebuilds
        // the full vocab lazily after load rather than applying one quotient to
        // every alternative.
        let loaded = DynamicConstraint::load(&optimized.save()).unwrap();
        assert_eq!(loaded.start().mask(), expected_mask);

        // The external-vocab transfer has per-alternative metadata and can keep
        // each quotient independently.
        let transferred = DynamicConstraint::load_with_vocab(
            &optimized.save_with_external_vocab(),
            &vocab,
        )
        .unwrap();
        assert_eq!(transferred.start().mask(), expected_mask);
    }

    #[test]
    fn partition_optimized_component_keeps_boundary_token_distinctions() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"ax".to_vec()),
            (3, b"ay".to_vec()),
            (4, b"bx".to_vec()),
            (5, b"by".to_vec()),
            (6, b"x".to_vec()),
            (7, b"y".to_vec()),
        ]);
        let child_grammar = Grammar::ebnf(r#"start ::= [ab]"#);
        let child_partition = VocabPartition::compile(child_grammar.clone(), &vocab).unwrap();
        assert_eq!(child_partition.class_of(2), child_partition.class_of(3));
        assert_eq!(child_partition.class_of(4), child_partition.class_of(5));

        let ordinary_child = DynamicConstraint::compile(child_grammar.clone(), &vocab).unwrap();
        let optimized_child =
            DynamicConstraint::compile_with_vocab_partition(child_grammar, &vocab).unwrap();
        assert_eq!(optimized_child.inner.token_bytes_count(), vocab.len());

        let parent = Grammar::glrm(
            "glrm 1; start start; extern grammar child; nt start = child \"x\";",
        );
        let ordinary = ConstraintSpec::builder(parent.clone(), &vocab)
            .unwrap()
            .bind_grammar("child", &ordinary_child)
            .unwrap()
            .build()
            .unwrap()
            .compile_dynamic()
            .unwrap();
        let optimized = ConstraintSpec::builder(parent, &vocab)
            .unwrap()
            .bind_grammar("child", &optimized_child)
            .unwrap()
            .build()
            .unwrap()
            .compile_dynamic()
            .unwrap();

        let ordinary_mask = ordinary.start().mask();
        let optimized_mask = optimized.start().mask();
        assert_eq!(ordinary_mask, optimized_mask);
        assert!(token_allowed(&optimized_mask, 2), "ax must cross child -> parent boundary");
        assert!(!token_allowed(&optimized_mask, 3), "ay must remain rejected by the parent boundary");
        assert!(token_allowed(&optimized_mask, 4), "bx must cross child -> parent boundary");
        assert!(!token_allowed(&optimized_mask, 5), "by must remain rejected by the parent boundary");
    }

    #[test]
    fn first_save_priming_threshold_tracks_large_composition_cache() {
        let vocab = Vocab::new(vec![(0, b"a".to_vec())]);
        let mut constraint = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1;\nstart start;\nt A = \"a\";\nnt start = A;\n"),
            &vocab,
        )
        .unwrap();
        assert!(!should_prime_first_save_artifact(&constraint));

        let mut large = crate::automata::unweighted_u32::dfa::DFA::new();
        large
            .states
            .resize_with(FIRST_SAVE_TEMPLATE_STATE_PRIME_THRESHOLD, Default::default);
        constraint.composition_parser_templates_by_terminal = vec![Some(large)];
        assert!(should_prime_first_save_artifact(&constraint));
    }

    fn component_backend_flags(constraint: &RuntimeConstraint) -> Vec<bool> {
        constraint
            .static_dynamic_overlay
            .as_ref()
            .expect("bound constraint must use the explicit segmented runtime")
            .segmented_parser_components
            .iter()
            .map(|component| component.constraint.uses_dynamic_runtime())
            .collect()
    }

    fn leaf_backend_flags(constraint: &RuntimeConstraint) -> Vec<bool> {
        match constraint.static_dynamic_overlay.as_ref() {
            Some(overlay) if !overlay.segmented_parser_components.is_empty() => overlay
                .segmented_parser_components
                .iter()
                .flat_map(|component| leaf_backend_flags(&component.constraint))
                .collect(),
            _ => vec![constraint.uses_dynamic_runtime()],
        }
    }

    fn assert_recursive_compiler_views_detached(constraint: &RuntimeConstraint) {
        if !constraint.uses_compact_segmented_parser_runtime() {
            return;
        }
        assert_eq!(
            constraint.table.num_states, 0,
            "recursive coordinator retained a flattened LR state machine",
        );
        assert!(constraint.table.action.is_empty() && constraint.table.goto.is_empty());
        let overlay = constraint.static_dynamic_overlay.as_ref().unwrap();
        assert!(
            overlay.recursive_compiler_table.get().is_some(),
            "recursive coordinator lost its lazy compiler table",
        );
        for component in &overlay.segmented_parser_components {
            assert_recursive_compiler_views_detached(&component.constraint);
        }
    }

    fn poison_materialized_outer_table(constraint: &mut RuntimeConstraint) {
        constraint.recursive_parser_layout().unwrap().unwrap();
        constraint.table.action.clear();
        constraint.table.goto.clear();
        constraint.table.advance.clear();
        constraint.table.unconditional_advance.clear();
        constraint.table.rules.clear();
        constraint.table.forwarded_shifts.clear();
        constraint.table.control_terminals.clear();
        constraint.table.skip_terminals.clear();
        constraint.table.guarded_shift_index.clear();
        constraint.table.direct_regular_wide_frontiers.clear();
        constraint.table.num_states = 0;
        constraint.table.num_terminals = 0;
        constraint.table.num_rules = 0;
    }

    fn assert_static_boundary(constraint: &RuntimeConstraint) {
        let overlay = constraint
            .static_dynamic_overlay
            .as_ref()
            .expect("bound constraint must use the explicit segmented runtime");
        assert!(overlay.segmented_boundary_parser.is_none());
        assert!(overlay.segmented_boundary_terminal_trie.is_none());
        assert!(
            !overlay.segmented_boundary_shards.is_empty(),
            "static boundary policy must publish component-owned shards",
        );
        assert!(overlay.segmented_boundary_shards.iter().all(|shard| matches!(
            shard.backend,
            crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_)
        )));
    }

    fn assert_dynamic_boundary(constraint: &RuntimeConstraint) {
        let overlay = constraint
            .static_dynamic_overlay
            .as_ref()
            .expect("bound constraint must use the explicit segmented runtime");
        assert!(overlay.segmented_boundary_parser.is_none());
        assert!(overlay.segmented_boundary_terminal_trie.is_none());
        assert!(overlay.segmented_boundary_shards.iter().all(|shard| matches!(
            shard.backend,
            crate::runtime::SegmentedBoundaryShardBackend::DynamicDirect
        )));
    }

    #[test]
    fn segmented_composition_preserves_supplied_component_backends() {
        let vocab = Vocab::new(vec![
            (0, b"x".to_vec()),
            (1, b"y".to_vec()),
            // Forces B to carry a strictly internal parent -> child crossing.
            (2, b"xy".to_vec()),
        ]);
        let parent = Grammar::glrm(
            "glrm 1; start start; extern grammar child; nt start = \"x\" child;",
        );
        let static_child = RuntimeConstraint::compile(Grammar::ebnf(r#"start ::= "y""#), &vocab)
            .unwrap();
        let dynamic_child =
            DynamicConstraint::compile(Grammar::ebnf(r#"start ::= "y""#), &vocab).unwrap();

        let static_with_dynamic = ConstraintSpec::builder(parent.clone(), &vocab)
            .unwrap()
            .bind_grammar("child", &dynamic_child)
            .unwrap()
            .build()
            .unwrap()
            .compile()
            .unwrap();
        assert_eq!(component_backend_flags(&static_with_dynamic), vec![false, true]);
        assert!(
            static_with_dynamic.serialized_artifact_cache.is_none(),
            "compiling a hybrid must not eagerly staticify its dynamic leaf for serialization",
        );
        assert_static_boundary(&static_with_dynamic);

        let dynamic_with_static = ConstraintSpec::builder(parent.clone(), &vocab)
            .unwrap()
            .bind_grammar("child", &static_child)
            .unwrap()
            .build()
            .unwrap()
            .compile_dynamic()
            .unwrap();
        for alternative in dynamic_with_static.clone_constraints() {
            assert_eq!(component_backend_flags(&alternative), vec![true, false]);
            let overlay = alternative.static_dynamic_overlay.as_ref().unwrap();
            assert!(overlay.segmented_boundary_parser.is_none());
            assert!(overlay.segmented_boundary_terminal_trie.is_none());
            assert!(overlay.segmented_boundary_shards.iter().all(|shard| matches!(
                shard.backend,
                crate::runtime::SegmentedBoundaryShardBackend::DynamicDirect
            )));
        }

        let dynamic_with_dynamic = ConstraintSpec::builder(parent, &vocab)
            .unwrap()
            .bind_grammar("child", &dynamic_child)
            .unwrap()
            .build()
            .unwrap()
            .compile_dynamic()
            .unwrap();
        for alternative in dynamic_with_dynamic.clone_constraints() {
            assert_eq!(component_backend_flags(&alternative), vec![true, true]);
        }
    }

    #[test]
    fn compiled_parent_late_binding_preserves_leaf_backends_and_boundary_choice() {
        let vocab = Vocab::new(vec![
            (0, b"x".to_vec()),
            (1, b"y".to_vec()),
            (2, b"xy".to_vec()),
            (3, b"z".to_vec()),
            (4, b"yz".to_vec()),
            (5, b"xyz".to_vec()),
        ]);
        let one_slot = Grammar::glrm(
            "glrm 1; start start; extern grammar child; nt start = \"x\" child;",
        );
        let static_parent = RuntimeConstraint::compile(one_slot.clone(), &vocab).unwrap();
        let dynamic_parent = DynamicConstraint::compile(one_slot, &vocab).unwrap();
        let static_child = RuntimeConstraint::compile(Grammar::ebnf(r#"start ::= "y""#), &vocab)
            .unwrap();
        let dynamic_child =
            DynamicConstraint::compile(Grammar::ebnf(r#"start ::= "y""#), &vocab).unwrap();

        let bound = static_parent.bind_grammar("child", &static_child).unwrap();
        assert_eq!(leaf_backend_flags(&bound), vec![false, false]);
        assert_static_boundary(&bound);
        assert!(
            bound.serialized_artifact_cache.is_none(),
            "late binding must not pay first-save serialization eagerly",
        );

        let bound = static_parent
            .bind_grammar_dynamic_boundary("child", &static_child)
            .unwrap();
        assert_eq!(leaf_backend_flags(&bound), vec![false, false]);
        assert_dynamic_boundary(&bound);

        let bound = static_parent.bind_grammar("child", &dynamic_child).unwrap();
        assert_eq!(leaf_backend_flags(&bound), vec![false, true]);
        assert_static_boundary(&bound);
        assert!(
            bound.serialized_artifact_cache.is_none(),
            "late binding a dynamic child must not trigger serialization-time staticification",
        );

        let bound = static_parent
            .bind_grammar_dynamic_boundary("child", &dynamic_child)
            .unwrap();
        assert_eq!(leaf_backend_flags(&bound), vec![false, true]);
        assert_dynamic_boundary(&bound);

        let bound = dynamic_parent.bind_grammar("child", &static_child).unwrap();
        for alternative in bound.clone_constraints() {
            assert_eq!(leaf_backend_flags(&alternative), vec![true, false]);
            assert_static_boundary(&alternative);
        }

        let bound = dynamic_parent
            .bind_grammar_dynamic_boundary("child", &static_child)
            .unwrap();
        for alternative in bound.clone_constraints() {
            assert_eq!(leaf_backend_flags(&alternative), vec![true, false]);
            assert_dynamic_boundary(&alternative);
        }

        let bound = dynamic_parent.bind_grammar("child", &dynamic_child).unwrap();
        for alternative in bound.clone_constraints() {
            assert_eq!(leaf_backend_flags(&alternative), vec![true, true]);
            assert_static_boundary(&alternative);
        }

        let bound = dynamic_parent
            .bind_grammar_dynamic_boundary("child", &dynamic_child)
            .unwrap();
        for alternative in bound.clone_constraints() {
            assert_eq!(leaf_backend_flags(&alternative), vec![true, true]);
            assert_dynamic_boundary(&alternative);
        }

        let two_slots = Grammar::glrm(
            "glrm 1; start start; extern grammar left; extern grammar right; \
             nt start = \"x\" left right;",
        );
        let open = RuntimeConstraint::compile(two_slots, &vocab).unwrap();
        let partially_bound = open.bind_grammar("left", &static_child).unwrap();
        assert!(
            partially_bound.late_bind_vocab.get().is_some(),
            "partially bound constraints should carry the shared vocabulary cache into the next bind",
        );
        assert!(partially_bound
            .late_grammar_slots
            .iter()
            .any(|slot| slot.name == "right"));
        let fully_bound = partially_bound
            .bind_grammar(
                "right",
                DynamicConstraint::compile(Grammar::ebnf(r#"start ::= "z""#), &vocab).unwrap(),
            )
            .unwrap();
        assert!(fully_bound.late_grammar_slots.is_empty());
        assert_eq!(leaf_backend_flags(&fully_bound), vec![false, false, true]);
        assert_static_boundary(&fully_bound);

        let fully_bound = partially_bound
            .bind_grammar_dynamic_boundary(
                "right",
                DynamicConstraint::compile(Grammar::ebnf(r#"start ::= "z""#), &vocab).unwrap(),
            )
            .unwrap();
        assert!(fully_bound.late_grammar_slots.is_empty());
        assert_eq!(leaf_backend_flags(&fully_bound), vec![false, false, true]);
        assert_dynamic_boundary(&fully_bound);

        let open = DynamicConstraint::compile(
            Grammar::glrm(
                "glrm 1; start start; extern grammar left; extern grammar right; \
                 nt start = \"x\" left right;",
            ),
            &vocab,
        )
        .unwrap();
        let partially_bound = open.bind_grammar("left", &static_child).unwrap();
        let dynamic_right =
            DynamicConstraint::compile(Grammar::ebnf(r#"start ::= "z""#), &vocab).unwrap();
        let fully_bound = partially_bound.bind_grammar("right", &dynamic_right).unwrap();
        for alternative in fully_bound.clone_constraints() {
            assert_eq!(leaf_backend_flags(&alternative), vec![true, false, true]);
            assert_static_boundary(&alternative);
        }
        let fully_bound = partially_bound
            .bind_grammar_dynamic_boundary("right", &dynamic_right)
            .unwrap();
        for alternative in fully_bound.clone_constraints() {
            assert_eq!(leaf_backend_flags(&alternative), vec![true, false, true]);
            assert_dynamic_boundary(&alternative);
        }
    }

    #[test]
    fn all_static_late_bound_segmented_roundtrip_preserves_fused_boundary_tokens() {
        let vocab = Vocab::new(vec![
            (0, b"x".to_vec()),
            (1, b"y".to_vec()),
            (2, b"xy".to_vec()),
        ]);
        let parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start start; extern grammar child; nt start = \"x\" child;",
            ),
            &vocab,
        )
        .unwrap();
        let child = RuntimeConstraint::compile(Grammar::ebnf(r#"start ::= "y""#), &vocab)
            .unwrap();

        for dynamic_boundary in [false, true] {
            let bound = if dynamic_boundary {
                parent
                    .bind_grammar_dynamic_boundary("child", child.clone())
                    .unwrap()
            } else {
                parent.bind_grammar("child", child.clone()).unwrap()
            };
            let live_mask = bound.start().mask();
            assert_ne!(
                live_mask[0] & (1 << 2),
                0,
                "fused parent/child token must be live before serialization",
            );

            let loaded = RuntimeConstraint::load(bound.save()).unwrap();
            let overlay = loaded
                .static_dynamic_overlay
                .as_ref()
                .expect("round-tripped segmented runtime must retain A/B metadata");
            assert!(
                !overlay.segmented_static_baseline,
                "new segmented saves must not synthesize a flattened static parser baseline",
            );
            assert!(
                overlay.segmented_parser_components.len() == 2,
                "new segmented saves must retain the parent and child A components exactly",
            );
            assert!(overlay
                .segmented_parser_components
                .iter()
                .all(|component| component.root_disallowed_terminal.is_none()));
            assert_eq!(
                loaded.start().mask(),
                live_mask,
                "all-static segmented A+B changed across save/load (dynamic_boundary={dynamic_boundary})",
            );

            let mut state = loaded.start();
            state.commit_token(2).unwrap();
            assert!(state.is_accepting());

            let mut live_state = bound.start();
            let mut loaded_state = loaded.start();
            for token in [0, 1] {
                assert_eq!(
                    loaded_state.mask(),
                    live_state.mask(),
                    "all-static segmented state changed after round-trip before token {token} (dynamic_boundary={dynamic_boundary})",
                );
                live_state.commit_token(token).unwrap();
                loaded_state.commit_token(token).unwrap();
            }
            assert_eq!(loaded_state.is_accepting(), live_state.is_accepting());
            assert!(loaded_state.is_accepting());
        }
    }

    #[test]
    fn composed_dynamic_constraint_transfer_roundtrip_preserves_recursive_runtime() {
        let vocab = Vocab::new(vec![
            (0, b"x".to_vec()),
            (1, b"y".to_vec()),
            (2, b"xy".to_vec()),
            (3, b"xyz".to_vec()),
        ]);
        let reference = RuntimeConstraint::compile(
            Grammar::ebnf(r#"start ::= "x" "y""#),
            &vocab,
        )
        .unwrap();
        let parent = DynamicConstraint::compile(
            Grammar::glrm(
                "glrm 1; start start; extern grammar child; nt start = \"x\" child;",
            ),
            &vocab,
        )
        .unwrap();
        let child = DynamicConstraint::compile(Grammar::ebnf(r#"start ::= "y""#), &vocab)
            .unwrap();

        for dynamic_boundary in [false, true] {
            let bound = if dynamic_boundary {
                parent
                    .bind_grammar_dynamic_boundary("child", &child)
                    .unwrap()
            } else {
                parent.bind_grammar("child", &child).unwrap()
            };
            assert_eq!(bound.start().mask(), reference.start().mask());

            let transfer = bound.clone().into_saved();
            let loaded = DynamicConstraint::load_with_vocab(&transfer, &vocab).unwrap();
            assert_eq!(
                loaded.start().mask(),
                bound.start().mask(),
                "composed dynamic transfer changed initial mask (dynamic_boundary={dynamic_boundary})",
            );
            let mut state = loaded.start();
            state.commit_token(2).unwrap();
            assert!(state.is_accepting());
        }
    }

    #[test]
    fn dynamic_boundary_token_triggers_survive_component_roundtrip() {
        let vocab = Vocab::new(vec![
            (0, b"x".to_vec()),
            (1, b"y".to_vec()),
            (2, b"xy".to_vec()),
            (3, b"z".to_vec()),
        ]);
        let reference = RuntimeConstraint::compile(
            Grammar::ebnf(r#"start ::= "x" "y""#),
            &vocab,
        )
        .unwrap();
        let mut parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start start; extern grammar child; nt start = \"x\" child;",
            ),
            &vocab,
        )
        .unwrap();
        let mut child = RuntimeConstraint::compile(
            Grammar::ebnf(r#"start ::= "y""#),
            &vocab,
        )
        .unwrap();

        parent.build_boundary_token_trigger().unwrap();
        child.build_boundary_token_trigger().unwrap();
        assert!(parent
            .boundary_trigger
            .token_summary()
            .expect("parent token trigger must be built")
            .contains(&2));

        let loaded_parent = RuntimeConstraint::load_body_artifact(parent.save()).unwrap();
        let loaded_child = RuntimeConstraint::load(child.save()).unwrap();
        let bound = loaded_parent
            .bind_grammar_dynamic_boundary("child", loaded_child)
            .unwrap();
        assert_eq!(bound.start().mask(), reference.start().mask());

        let overlay = bound
            .static_dynamic_overlay
            .as_ref()
            .expect("dynamic composition must retain component runtime metadata");
        let parent_tokens = overlay.segmented_parser_components[0]
            .constraint
            .boundary_trigger
            .token_summary()
            .expect("parent Tokens trigger must survive save/load");
        let child_tokens = overlay.segmented_parser_components[1]
            .constraint
            .boundary_trigger
            .token_summary()
            .expect("child Tokens trigger must survive save/load");
        assert!(parent_tokens.contains(&2));
        assert!(child_tokens.iter().all(|&token| token < 4));
    }

    #[test]
    fn exact_boundary_triggers_roundtrip_and_gate_entry_and_finish() {
        let vocab = Vocab::new(vec![
            (0, b"x".to_vec()),
            (1, b"y".to_vec()),
            (2, b"z".to_vec()),
            (3, b"xy".to_vec()),
            (4, b"yz".to_vec()),
            (5, b"xyz".to_vec()),
        ]);
        let reference = RuntimeConstraint::compile(
            Grammar::ebnf(r#"start ::= "x" "y" "z""#),
            &vocab,
        )
        .unwrap();
        let mut parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start start; extern grammar child; nt start = \"x\" child \"z\";",
            ),
            &vocab,
        )
        .unwrap();
        let mut child = RuntimeConstraint::compile(
            Grammar::ebnf(r#"start ::= "y""#),
            &vocab,
        )
        .unwrap();

        parent.build_exact_boundary_trigger().unwrap();
        child.build_exact_boundary_trigger().unwrap();
        assert!(matches!(
            parent.boundary_trigger,
            crate::runtime::BoundaryTrigger::Exact(_)
        ));
        assert!(matches!(
            child.boundary_trigger,
            crate::runtime::BoundaryTrigger::Exact(_)
        ));

        let loaded_parent = RuntimeConstraint::load_body_artifact(parent.save()).unwrap();
        let loaded_child = RuntimeConstraint::load(child.save()).unwrap();
        let bound = loaded_parent
            .bind_grammar_dynamic_boundary("child", loaded_child)
            .unwrap();
        let overlay = bound
            .static_dynamic_overlay
            .as_ref()
            .expect("dynamic composition must retain component runtime metadata");
        assert!(overlay.segmented_parser_components.iter().all(|component| {
            matches!(
                component.constraint.boundary_trigger,
                crate::runtime::BoundaryTrigger::Exact(_)
            )
        }));

        for tokens in [&[5][..], &[0, 4][..], &[3, 2][..], &[0, 1, 2][..]] {
            let mut actual = bound.start();
            let mut expected = reference.start();
            for &token in tokens {
                assert_eq!(
                    actual.mask(),
                    expected.mask(),
                    "Exact-trigger dynamic boundary mismatch before {tokens:?} token {token}",
                );
                actual.commit_token(token).unwrap();
                expected.commit_token(token).unwrap();
            }
            assert_eq!(actual.is_accepting(), expected.is_accepting(), "{tokens:?}");
            assert!(actual.is_accepting(), "{tokens:?}");
        }
    }

    #[test]
    fn exact_trigger_supports_control_bearing_composed_component() {
        let vocab = Vocab::new(vec![
            (0, b"<".to_vec()),
            (1, b"a".to_vec()),
            (2, b">".to_vec()),
            (3, b"!".to_vec()),
            (4, b"<a>!".to_vec()),
            (5, b"a>!".to_vec()),
            (6, b">!".to_vec()),
        ]);

        let inner_parent = RuntimeConstraint::compile(
            Grammar::glrm(
                r#"glrm 1; start inner; extern grammar leaf; nt inner = "<" leaf ">";"#,
            ),
            &vocab,
        )
        .unwrap();
        let leaf = RuntimeConstraint::compile(Grammar::ebnf(r#"start ::= "a""#), &vocab)
            .unwrap();
        let mut inner = inner_parent
            .bind_grammar_dynamic_boundary("leaf", leaf)
            .unwrap();
        assert_recursive_compiler_views_detached(&inner);
        let mut compiler_view = inner.clone();
        compiler_view
            .prepare_recursive_compiler_table_for_composition()
            .unwrap();
        assert!(
            !compiler_view.table.control_terminals.is_empty(),
            "fixture must exercise Exact construction over compiler-materialized linker controls",
        );
        inner.build_exact_boundary_trigger().unwrap();
        assert_recursive_compiler_views_detached(&inner);
        assert!(matches!(
            inner.boundary_trigger,
            crate::runtime::BoundaryTrigger::Exact(_)
        ));

        let outer = RuntimeConstraint::compile(
            Grammar::glrm(
                r#"glrm 1; start document; extern grammar inner; nt document = inner "!";"#,
            ),
            &vocab,
        )
        .unwrap();
        let composed = outer
            .bind_grammar_dynamic_boundary("inner", RuntimeConstraint::load(inner.save()).unwrap())
            .unwrap();
        let reference = RuntimeConstraint::compile(
            Grammar::ebnf(r#"start ::= "<" "a" ">" "!""#),
            &vocab,
        )
        .unwrap();

        let overlay = composed.static_dynamic_overlay.as_ref().unwrap();
        assert!(matches!(
            overlay.segmented_parser_components[1]
                .constraint
                .boundary_trigger,
            crate::runtime::BoundaryTrigger::Exact(_)
        ));

        for tokens in [&[4][..], &[0, 5][..], &[0, 1, 6][..], &[0, 1, 2, 3][..]] {
            let mut actual = composed.start();
            let mut expected = reference.start();
            for &token in tokens {
                assert_eq!(
                    actual.mask(),
                    expected.mask(),
                    "control-bearing Exact trigger mismatch before {tokens:?} token {token}",
                );
                actual.commit_token(token).unwrap();
                expected.commit_token(token).unwrap();
            }
            assert_eq!(actual.is_accepting(), expected.is_accepting(), "{tokens:?}");
            assert!(actual.is_accepting(), "{tokens:?}");
        }
    }

    #[test]
    fn loaded_constraint_trigger_upgrade_resaves_updated_link_metadata() {
        let vocab = Vocab::new(vec![
            (0, b"x".to_vec()),
            (1, b"y".to_vec()),
            (2, b"xy".to_vec()),
        ]);
        let parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start start; extern grammar child; nt start = \"x\" child;",
            ),
            &vocab,
        )
        .unwrap();
        let child = RuntimeConstraint::compile(Grammar::ebnf(r#"start ::= "y""#), &vocab)
            .unwrap();

        let mut loaded = RuntimeConstraint::load_body_artifact(parent.save()).unwrap();
        assert!(loaded.deferred_composition_metadata_blob.is_some());
        loaded.build_exact_boundary_trigger().unwrap();
        assert!(matches!(
            loaded.boundary_trigger,
            crate::runtime::BoundaryTrigger::Exact(_)
        ));
        assert!(
            loaded.deferred_composition_metadata_blob.is_some(),
            "trigger upgrade should not force materialization of the heavy parser-cache section",
        );

        let reloaded = RuntimeConstraint::load_body_artifact(loaded.save()).unwrap();
        let bound = reloaded
            .bind_grammar_dynamic_boundary("child", child)
            .unwrap();
        let overlay = bound.static_dynamic_overlay.as_ref().unwrap();
        assert!(matches!(
            overlay.segmented_parser_components[0]
                .constraint
                .boundary_trigger,
            crate::runtime::BoundaryTrigger::Exact(_)
        ));
        assert_ne!(bound.start().mask()[0] & (1 << 2), 0);
    }

    #[test]
    fn exact_triggers_survive_outer_runtime_lexer_product_multi_source_states() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"c".to_vec()),
            (3, b"!".to_vec()),
            (4, b"?".to_vec()),
            (5, b"c!".to_vec()),
            (6, b"c?".to_vec()),
        ]);
        let child_spec = ConstraintSpec::builder(
            Grammar::ebnf(r#"start ::= "a" "b" "c""#),
            &vocab,
        )
        .unwrap()
        .boundary_trigger_detail(crate::BoundaryTriggerDetail::Exact)
        .build()
        .unwrap();
        // Keep one child component so its local-LR -> composed-LR relation is
        // functional, while the parent also has an equivalent local lexical
        // lane. The outer lexer product can then coalesce parent + child lanes
        // after `a`/`b` without relying on duplicate child call sites.
        let parent = Grammar::glrm(
            "glrm 1; start document; extern grammar child; \
             nt document = child \"!\" | \"a\" \"b\" \"c\" \"?\";",
        );
        let composed = ConstraintSpec::builder(parent, &vocab)
            .unwrap()
            .bind_grammar("child", child_spec)
            .unwrap()
            .build()
            .unwrap()
            .compile_dynamic()
            .unwrap();
        let alternatives = composed.clone_constraints();
        assert_eq!(alternatives.len(), 1);
        let composed = &alternatives[0];
        let reference = RuntimeConstraint::compile(
            Grammar::ebnf(r#"start ::= "a" "b" "c" "!" | "a" "b" "c" "?""#),
            &vocab,
        )
        .unwrap();

        let explicitly_disabled = std::env::var("GLRMASK_COMPOSE_RUNTIME_LEXER_PRODUCT")
            .ok()
            .is_some_and(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "" | "0" | "false" | "no" | "off"
                )
            });

        let mut actual = composed.start();
        let mut expected = reference.start();
        for token in [0, 1] {
            assert_eq!(actual.mask(), expected.mask());
            actual.commit_token(token).unwrap();
            expected.commit_token(token).unwrap();
        }

        if !explicitly_disabled
            && let Some(source_offset) = composed.runtime_source_state_offset()
        {
            let product_state = *actual.state.keys().next().unwrap();
            assert!(product_state < source_offset);
            assert!(
                composed
                    .runtime_product_source_states(product_state)
                    .is_some_and(|sources| sources.len() >= 2),
                "a selected outer product state must preserve every represented lexer lane",
            );
        }

        let mask = actual.mask();
        let reference_mask = expected.mask();
        assert_eq!(mask, reference_mask);
        assert_ne!(mask[0] & (1 << 5), 0, "c! must cross child -> parent internally");
        assert_ne!(mask[0] & (1 << 6), 0, "c? must remain valid on the parent-local branch");
    }

    #[test]
    fn exact_trigger_detail_is_independent_of_dynamic_component_compilation() {
        let vocab = Vocab::new(vec![
            (0, b"y".to_vec()),
            (1, b"z".to_vec()),
            (2, b"yz".to_vec()),
        ]);
        let spec = ConstraintSpec::builder(Grammar::ebnf(r#"start ::= "y""#), &vocab)
            .unwrap()
            .boundary_trigger_detail(crate::BoundaryTriggerDetail::Exact)
            .build()
            .unwrap();

        let static_constraint = spec.compile().unwrap();
        assert!(matches!(
            static_constraint.boundary_trigger,
            crate::runtime::BoundaryTrigger::Exact(_)
        ));

        let dynamic_constraint = spec.compile_dynamic().unwrap();
        assert!(dynamic_constraint.clone_constraints().iter().all(|constraint| {
            matches!(constraint.boundary_trigger, crate::runtime::BoundaryTrigger::Exact(_))
        }));

        let loaded_dynamic = DynamicConstraint::load(&dynamic_constraint.save()).unwrap();
        assert!(loaded_dynamic.clone_constraints().iter().all(|constraint| {
            matches!(constraint.boundary_trigger, crate::runtime::BoundaryTrigger::Exact(_))
        }));

        let transfer = dynamic_constraint.clone().into_saved();
        let loaded_transfer = DynamicConstraint::load_with_vocab(&transfer, &vocab).unwrap();
        assert!(loaded_transfer.clone_constraints().iter().all(|constraint| {
            matches!(constraint.boundary_trigger, crate::runtime::BoundaryTrigger::Exact(_))
        }));
    }

    #[test]
    fn boundary_token_trigger_follows_lexeme_resets_inside_model_token() {
        let vocab = Vocab::new(vec![
            (0, b"x".to_vec()),
            (1, b"q".to_vec()),
            (2, b"y".to_vec()),
            (3, b"xqy".to_vec()),
        ]);
        let mut parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start start; extern grammar child; nt start = \"x\" \"q\" child;",
            ),
            &vocab,
        )
        .unwrap();
        parent.build_boundary_token_trigger().unwrap();
        assert!(
            parent
                .boundary_trigger
                .token_summary()
                .expect("Tokens trigger must be built")
                .contains(&3),
            "a token that reaches the child only after multiple local lexemes must remain boundary-relevant",
        );

        let child = RuntimeConstraint::compile(Grammar::ebnf(r#"start ::= "y""#), &vocab)
            .unwrap();
        let bound = parent
            .bind_grammar_dynamic_boundary("child", child)
            .unwrap();
        assert_ne!(bound.start().mask()[0] & (1 << 3), 0);
    }

    #[test]
    fn nested_dynamic_boundaries_preserve_token_crossing_leaf_middle_and_outer() {
        let vocab = Vocab::new(vec![
            (0, b"X".to_vec()),
            (1, b"[".to_vec()),
            (2, b"a".to_vec()),
            (3, b"]".to_vec()),
            (4, b"!".to_vec()),
            (5, b"a]!".to_vec()),
            (6, b"[a]!".to_vec()),
            (7, b"X[a]!".to_vec()),
        ]);
        let leaf = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start leaf; nt leaf = \"a\";"),
            &vocab,
        )
        .unwrap();
        let middle_parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start middle; extern grammar leaf; nt middle = \"[\" leaf \"]\";",
            ),
            &vocab,
        )
        .unwrap();
        let middle = middle_parent
            .bind_grammar_dynamic_boundary("leaf", leaf)
            .unwrap();
        let outer_parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start document; extern grammar middle; nt document = \"X\" middle \"!\";",
            ),
            &vocab,
        )
        .unwrap();
        let bound = outer_parent
            .bind_grammar_dynamic_boundary("middle", middle)
            .unwrap();
        let monolithic = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start document; nt document = \"X\" \"[\" \"a\" \"]\" \"!\";"),
            &vocab,
        )
        .unwrap();

        let loaded = RuntimeConstraint::load(&bound.save()).unwrap();
        assert_recursive_compiler_views_detached(&bound);
        assert_recursive_compiler_views_detached(&loaded);
        let mut no_trigger = bound.clone();
        {
            let overlay = no_trigger.static_dynamic_overlay.as_mut().unwrap();
            for component in &mut overlay.segmented_parser_components {
                let Some(shard) = component.boundary.as_mut() else {
                    continue;
                };
                if !matches!(
                    shard.backend,
                    crate::runtime::SegmentedBoundaryShardBackend::DynamicDirect
                ) {
                    continue;
                }
                shard.candidate_tokens = None;
                std::sync::Arc::make_mut(&mut component.constraint).boundary_trigger =
                    crate::runtime::BoundaryTrigger::None;
            }
        }
        let mut no_outer_tokenizer = loaded.clone();
        {
            let root = no_outer_tokenizer
                .static_dynamic_overlay
                .as_ref()
                .unwrap()
                .segmented_parser_components[0]
                .constraint
                .clone();
            no_outer_tokenizer.tokenizer = root.tokenizer.clone();
            no_outer_tokenizer.tokenizer_fast_transitions =
                root.tokenizer_fast_transitions.clone();
            no_outer_tokenizer.tokenizer_has_epsilon_transitions =
                root.tokenizer_has_epsilon_transitions;
        }
        let mut no_outer_table = loaded.clone();
        poison_materialized_outer_table(&mut no_outer_table);
        for constraint in [
            &bound,
            &loaded,
            &no_trigger,
            &no_outer_tokenizer,
            &no_outer_table,
        ] {
            let mut pending = vec![Vec::<u32>::new()];
            while let Some(path) = pending.pop() {
                let mut actual = constraint.start();
                let mut expected = monolithic.start();
                for &token in &path {
                    actual.commit_token(token).unwrap();
                    expected.commit_token(token).unwrap();
                }
                let actual_mask = actual.mask();
                let expected_mask = expected.mask();
                assert_eq!(actual_mask, expected_mask, "mask mismatch at {path:?}");
                assert_eq!(
                    actual.is_accepting(),
                    expected.is_accepting(),
                    "acceptance mismatch at {path:?}",
                );
                if path.len() == 5 {
                    continue;
                }
                for token in 0..8u32 {
                    if actual_mask
                        .get(token as usize / 32)
                        .is_none_or(|word| *word & (1u32 << (token % 32)) == 0)
                    {
                        continue;
                    }
                    let mut next = path.clone();
                    next.push(token);
                    pending.push(next);
                }
            }
        }

        for tokens in [
            &[7][..],
            &[0, 6][..],
            &[0, 1, 5][..],
            &[0, 1, 2, 3, 4][..],
        ] {
            let mut actual = bound.start();
            let mut expected = monolithic.start();
            for &token in tokens {
                assert_eq!(actual.mask(), expected.mask(), "before {tokens:?} token {token}");
                actual.commit_token(token).unwrap();
                expected.commit_token(token).unwrap();
            }
            assert_eq!(actual.is_accepting(), expected.is_accepting(), "{tokens:?}");
            assert!(actual.is_accepting(), "{tokens:?}");
        }
    }

    #[test]
    fn recursive_live_parser_expands_static_nested_component_with_native_boundary() {
        let vocab = Vocab::new(vec![
            (0, b"X".to_vec()),
            (1, b"[".to_vec()),
            (2, b"a".to_vec()),
            (3, b"]".to_vec()),
            (4, b"!".to_vec()),
            (5, b"a]!".to_vec()),
            (6, b"[a]!".to_vec()),
            (7, b"X[a]!".to_vec()),
        ]);
        let leaf = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start leaf; nt leaf = \"a\";"),
            &vocab,
        )
        .unwrap();
        let middle_parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start middle; extern grammar leaf; nt middle = \"[\" leaf \"]\";",
            ),
            &vocab,
        )
        .unwrap();
        // Static B is transported exactly onto the recursive parser coordinate,
        // so the middle wrapper can expose its intact parent/child leaves.
        let middle = middle_parent.bind_grammar("leaf", leaf).unwrap();
        assert!(middle.uses_compact_segmented_parser_runtime());
        let outer_parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start document; extern grammar middle; nt document = \"X\" middle \"!\";",
            ),
            &vocab,
        )
        .unwrap();
        let bound = outer_parent
            .bind_grammar_dynamic_boundary("middle", middle)
            .unwrap();
        let layout = bound.recursive_parser_layout().unwrap().unwrap();
        assert_eq!(layout.leaves.len(), 3);
        let monolithic = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start document; nt document = \"X\" \"[\" \"a\" \"]\" \"!\";"),
            &vocab,
        )
        .unwrap();

        for constraint in [&bound, &RuntimeConstraint::load(&bound.save()).unwrap()] {
            for tokens in [
                &[7][..],
                &[0, 6][..],
                &[0, 1, 5][..],
                &[0, 1, 2, 3, 4][..],
            ] {
                let mut actual = constraint.start();
                let mut expected = monolithic.start();
                for &token in tokens {
                    assert_eq!(actual.mask(), expected.mask(), "before {tokens:?} token {token}");
                    actual.commit_token(token).unwrap();
                    expected.commit_token(token).unwrap();
                }
                assert_eq!(actual.is_accepting(), expected.is_accepting(), "{tokens:?}");
                assert!(actual.is_accepting(), "{tokens:?}");
            }
        }
    }

    #[test]
    fn recursive_special_token_routes_through_active_child_leaf_and_returns_to_parent() {
        let vocab = Vocab::new(vec![(0, b"X".to_vec()), (1, b"!".to_vec())]);
        let child = RuntimeConstraint::compile(
            Grammar::glrm("start child; nt child ::= @token(100);"),
            &vocab,
        )
        .unwrap();
        let parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start document; extern grammar child; nt document = \"X\" child \"!\";",
            ),
            &vocab,
        )
        .unwrap();
        let bound = parent.bind_grammar("child", child).unwrap();
        assert!(bound.uses_compact_segmented_parser_runtime());

        let loaded = RuntimeConstraint::load(&bound.save()).unwrap();
        let mut no_outer_table = loaded.clone();
        poison_materialized_outer_table(&mut no_outer_table);
        for constraint in [&bound, &loaded, &no_outer_table] {
            let mut state = constraint.start();
            state.commit_token(0).unwrap();
            let mask = state.mask();
            assert_ne!(mask[100 / 32] & (1u32 << (100 % 32)), 0);

            state.commit_token(100).unwrap();
            let mask = state.mask();
            assert_ne!(mask[1 / 32] & (1u32 << (1 % 32)), 0);

            state.commit_token(1).unwrap();
            assert!(state.is_accepting());
        }
    }

    #[test]
    fn recursive_static_virtual_residual_child_handles_fused_boundaries_and_roundtrip() {
        let vocab = Vocab::new(vec![
            (0, b"X".to_vec()),
            (1, b"!".to_vec()),
            (2, b"\"".to_vec()),
            (3, b"x:".to_vec()),
            (4, b"a".to_vec()),
            (5, b"X\"".to_vec()),
            (6, b"\"!".to_vec()),
            (7, b"xa".to_vec()),
            (8, b":".to_vec()),
            (9, b"X\"x:".to_vec()),
            (10, b"a\"!".to_vec()),
            (11, b"aa".to_vec()),
            (12, b"a\"".to_vec()),
        ]);
        let child = RuntimeConstraint::from_json_schema(
            r#"{"type":"string","format":"uri","minLength":1,"maxLength":5000}"#,
            &vocab,
        )
        .unwrap();
        assert!(child.tokenizer.has_virtual_residual_runtime());

        let parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start document; extern grammar payload; nt document = \"X\" payload \"!\";",
            ),
            &vocab,
        )
        .unwrap();
        // A virtual-residual child cannot take a static shard: requesting
        // static must decline loudly, never silently succeed as dynamic.
        assert!(
            parent.bind_grammar("payload", &child).is_err(),
            "static bind of a virtual-residual child must decline loudly",
        );
        let bound = parent
            .bind_grammar_dynamic_boundary("payload", child)
            .unwrap();
        let loaded = RuntimeConstraint::load(&bound.save()).unwrap();

        let token_allowed = |mask: &[u32], token: u32| {
            mask[token as usize / 32] & (1u32 << (token % 32)) != 0
        };
        for constraint in [&bound, &loaded] {
            let mut state = constraint.start();
            assert!(token_allowed(&state.mask(), 9));
            state.commit_token(9).unwrap();
            assert!(token_allowed(&state.mask(), 10));
            state.commit_token(10).unwrap();
            assert!(state.is_accepting());

            let mut exact = constraint.start();
            let mut exact_bytes = b"X\"x:".to_vec();
            exact_bytes.extend(std::iter::repeat_n(b'a', 4998));
            exact_bytes.extend_from_slice(b"\"!");
            exact.commit_bytes(&exact_bytes).unwrap();
            assert!(exact.is_accepting());

            let mut over = constraint.start();
            let mut over_bytes = b"X\"x:".to_vec();
            over_bytes.extend(std::iter::repeat_n(b'a', 4999));
            over_bytes.extend_from_slice(b"\"!");
            assert!(over.commit_bytes(&over_bytes).is_err());
        }
    }

    #[test]
    fn nested_boundaries_use_recursive_parser_coordinate_live_and_loaded() {
        let vocab = Vocab::new(vec![
            (0, b"X".to_vec()),
            (1, b"[".to_vec()),
            (2, b"a".to_vec()),
            (3, b"]".to_vec()),
            (4, b"!".to_vec()),
            (5, b"a]!".to_vec()),
            (6, b"[a]!".to_vec()),
            (7, b"X[a]!".to_vec()),
        ]);
        let leaf = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start leaf; nt leaf = \"a\";"),
            &vocab,
        )
        .unwrap();
        let middle_parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start middle; extern grammar leaf; nt middle = \"[\" leaf \"]\";",
            ),
            &vocab,
        )
        .unwrap();
        let middle = middle_parent.bind_grammar("leaf", leaf).unwrap();
        assert!(middle.uses_compact_segmented_parser_runtime());

        let outer_parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start document; extern grammar middle; nt document = \"X\" middle \"!\";",
            ),
            &vocab,
        )
        .unwrap();
        let bound = outer_parent
            .bind_grammar_dynamic_boundary("middle", middle)
            .unwrap();
        assert!(bound.uses_compact_segmented_parser_runtime());
        assert!(
            bound
                .static_dynamic_overlay
                .as_ref()
                .unwrap()
                .segmented_component_union_root_dispatch
                .is_empty(),
            "recursive runtime must not retain materialized-state component dispatch",
        );
        assert!(
            bound
                .static_dynamic_overlay
                .as_ref()
                .unwrap()
                .segmented_parser_state_offsets
                .is_empty(),
            "recursive runtime must derive state intervals from its component tree",
        );
        let bound_overlay = bound.static_dynamic_overlay.as_ref().unwrap();
        assert!(bound_overlay.segmented_boundary_shards.iter().all(|shard| {
            shard.start_parser_states.len() == 0
        }));
        assert!(bound_overlay.segmented_parser_components.iter().all(|component| {
            component
                .boundary
                .as_ref()
                .is_none_or(|shard| shard.start_parser_states.len() == 0)
        }));
        let layout = bound.recursive_parser_layout().unwrap().unwrap();
        assert_eq!(layout.leaves.len(), 3);

        let loaded = RuntimeConstraint::load(&bound.save()).unwrap();
        assert!(loaded.uses_compact_segmented_parser_runtime());
        assert!(
            loaded
                .static_dynamic_overlay
                .as_ref()
                .unwrap()
                .segmented_component_union_root_dispatch
                .is_empty(),
        );
        assert!(
            loaded
                .static_dynamic_overlay
                .as_ref()
                .unwrap()
                .segmented_parser_state_offsets
                .is_empty(),
        );
        let loaded_overlay = loaded.static_dynamic_overlay.as_ref().unwrap();
        assert!(loaded_overlay.segmented_boundary_shards.iter().all(|shard| {
            shard.start_parser_states.len() == 0
        }));
        assert!(loaded_overlay.segmented_parser_components.iter().all(|component| {
            component
                .boundary
                .as_ref()
                .is_none_or(|shard| shard.start_parser_states.len() == 0)
        }));
        assert_eq!(loaded.recursive_parser_layout().unwrap().unwrap().leaves.len(), 3);

        let monolithic = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start document; nt document = \"X\" \"[\" \"a\" \"]\" \"!\";"),
            &vocab,
        )
        .unwrap();
        let mut no_outer_table = loaded.clone();
        poison_materialized_outer_table(&mut no_outer_table);
        fn compare_reachable_prefix_tree(
            actual_constraint: &RuntimeConstraint,
            expected_constraint: &RuntimeConstraint,
            max_depth: usize,
        ) {
            let mut pending = vec![Vec::<u32>::new()];
            while let Some(path) = pending.pop() {
                let mut actual = actual_constraint.start();
                let mut expected = expected_constraint.start();
                for &token in &path {
                    actual.commit_token(token).unwrap();
                    expected.commit_token(token).unwrap();
                }
                let actual_mask = actual.mask();
                let expected_mask = expected.mask();
                assert_eq!(actual_mask, expected_mask, "mask mismatch at {path:?}");
                assert_eq!(
                    actual.is_accepting(),
                    expected.is_accepting(),
                    "acceptance mismatch at {path:?}",
                );
                if path.len() == max_depth {
                    continue;
                }
                for token in 0..8u32 {
                    let word = token as usize / 32;
                    let bit = token % 32;
                    if actual_mask
                        .get(word)
                        .is_none_or(|word| *word & (1u32 << bit) == 0)
                    {
                        continue;
                    }
                    let mut next = path.clone();
                    next.push(token);
                    pending.push(next);
                }
            }
        }
        for constraint in [&bound, &loaded, &no_outer_table] {
            compare_reachable_prefix_tree(constraint, &monolithic, 5);
            for tokens in [
                &[7][..],
                &[0, 6][..],
                &[0, 1, 5][..],
                &[0, 1, 2, 3, 4][..],
            ] {
                let mut actual = constraint.start();
                let mut expected = monolithic.start();
                for &token in tokens {
                    assert_eq!(actual.mask(), expected.mask(), "before {tokens:?} token {token}");
                    actual.commit_token(token).unwrap();
                    expected.commit_token(token).unwrap();
                }
                assert_eq!(actual.is_accepting(), expected.is_accepting(), "{tokens:?}");
                assert!(actual.is_accepting(), "{tokens:?}");
            }
        }
    }

    #[test]
    fn nested_static_nullable_wrapper_returns_in_recursive_live_runtime() {
        let vocab = Vocab::new(vec![
            (0, b"X".to_vec()),
            (1, b"a".to_vec()),
            (2, b"!".to_vec()),
            (3, b"X!".to_vec()),
            (4, b"Xa!".to_vec()),
            (5, b"a!".to_vec()),
        ]);
        let leaf = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start leaf; nt leaf = \"a\";"),
            &vocab,
        )
        .unwrap();
        let middle_parent = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start middle; extern grammar leaf; nt middle = leaf?;"),
            &vocab,
        )
        .unwrap();
        assert!(middle_parent.table.embedded_start_nullable());
        let middle = middle_parent.bind_grammar("leaf", leaf).unwrap();
        assert!(middle.table.embedded_start_nullable());
        assert!(middle.uses_compact_segmented_parser_runtime());

        let outer_parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start document; extern grammar middle; nt document = \"X\" middle \"!\";",
            ),
            &vocab,
        )
        .unwrap();
        let bound = outer_parent
            .bind_grammar_dynamic_boundary("middle", middle)
            .unwrap();
        assert!(bound.uses_compact_segmented_parser_runtime());
        let loaded = RuntimeConstraint::load(&bound.save()).unwrap();
        assert!(loaded.uses_compact_segmented_parser_runtime());
        let monolithic = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start document; nt document = \"X\" \"a\"? \"!\";"),
            &vocab,
        )
        .unwrap();
        let mut no_outer_table = loaded.clone();
        poison_materialized_outer_table(&mut no_outer_table);

        for constraint in [&bound, &loaded, &no_outer_table] {
            for tokens in [&[3][..], &[4][..], &[0, 2][..], &[0, 5][..], &[0, 1, 2][..]] {
                let mut actual = constraint.start();
                let mut expected = monolithic.start();
                for &token in tokens {
                    assert_eq!(actual.mask(), expected.mask(), "before {tokens:?} token {token}");
                    actual.commit_token(token).unwrap();
                    expected.commit_token(token).unwrap();
                }
                assert_eq!(actual.is_accepting(), expected.is_accepting(), "{tokens:?}");
                assert!(actual.is_accepting(), "{tokens:?}");
            }
        }
    }

    #[test]
    fn recursive_parser_layout_flattens_state_coordinate_but_preserves_outer_wrapper_owner() {
        let vocab = Vocab::new(vec![
            (0, b"X".to_vec()),
            (1, b"[".to_vec()),
            (2, b"a".to_vec()),
            (3, b"]".to_vec()),
            (4, b"!".to_vec()),
        ]);
        let leaf = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start leaf; nt leaf = \"a\";"),
            &vocab,
        )
        .unwrap();
        let leaf_states = leaf.table.num_states;
        let leaf_tokenizer_states = leaf.tokenizer.num_states();
        let leaf_tokenizer_reset = leaf.runtime_commit_initial_state();
        let middle_parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start middle; extern grammar leaf; nt middle = \"[\" leaf \"]\";",
            ),
            &vocab,
        )
        .unwrap();
        let middle_parent_states = middle_parent.table.num_states;
        let middle_parent_tokenizer_states = middle_parent.tokenizer.num_states();
        let middle_parent_tokenizer_reset = middle_parent.runtime_commit_initial_state();
        let middle = middle_parent
            .bind_grammar_dynamic_boundary("leaf", leaf)
            .unwrap();
        let outer_parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start document; extern grammar middle; nt document = \"X\" middle \"!\";",
            ),
            &vocab,
        )
        .unwrap();
        let outer_parent_states = outer_parent.table.num_states;
        let outer_parent_tokenizer_states = outer_parent.tokenizer.num_states();
        let outer_parent_tokenizer_reset = outer_parent.runtime_commit_initial_state();
        let bound = outer_parent
            .bind_grammar_dynamic_boundary("middle", middle)
            .unwrap();

        for constraint in [&bound, &RuntimeConstraint::load(&bound.save()).unwrap()] {
            let layout = constraint
                .recursive_parser_layout()
                .unwrap()
                .expect("nested composition must expose a recursive parser layout");
            assert_eq!(layout.component_offsets, vec![0, outer_parent_states]);
            assert_eq!(layout.leaves.len(), 3);
            assert_eq!(
                layout
                    .leaves
                    .iter()
                    .map(|leaf| leaf.component_path.clone())
                    .collect::<Vec<_>>(),
                vec![vec![0], vec![1, 0], vec![1, 1]],
            );
            assert_eq!(
                layout
                    .leaves
                    .iter()
                    .map(|leaf| (leaf.state_offset, leaf.state_count, leaf.top_component))
                    .collect::<Vec<_>>(),
                vec![
                    (0, outer_parent_states, 0),
                    (outer_parent_states, middle_parent_states, 1),
                    (
                        outer_parent_states + middle_parent_states,
                        leaf_states,
                        1,
                    ),
                ],
            );
            assert_eq!(
                layout.total_states,
                outer_parent_states + middle_parent_states + leaf_states,
            );
            assert_eq!(
                layout.leaf_tokenizer_state_offsets,
                vec![
                    0,
                    outer_parent_tokenizer_states,
                    outer_parent_tokenizer_states + middle_parent_tokenizer_states,
                ],
            );
            assert_eq!(
                layout.total_tokenizer_states,
                outer_parent_tokenizer_states
                    + middle_parent_tokenizer_states
                    + leaf_tokenizer_states,
            );
            for (global_terminal, targets) in layout.terminal_targets.iter().enumerate() {
                for &(leaf_index, local_terminal) in targets {
                    let scoped = constraint
                        .recursive_terminal_scoped_id(leaf_index as usize, local_terminal)
                        .unwrap();
                    assert_eq!(
                        constraint.recursive_terminal_leaf_local(scoped),
                        Some((leaf_index as usize, local_terminal)),
                        "terminal routing lost global={global_terminal} leaf={leaf_index} local={local_terminal}",
                    );
                }
            }
            let tokenizer_counts = [
                outer_parent_tokenizer_states,
                middle_parent_tokenizer_states,
                leaf_tokenizer_states,
            ];
            let tokenizer_resets = [
                outer_parent_tokenizer_reset,
                middle_parent_tokenizer_reset,
                leaf_tokenizer_reset,
            ];
            for leaf_index in 0..layout.leaves.len() {
                for local_state in 0..tokenizer_counts[leaf_index] {
                    let scoped = constraint
                        .recursive_tokenizer_scoped_state(leaf_index, local_state)
                        .unwrap();
                    assert_eq!(
                        constraint.recursive_tokenizer_leaf_state(scoped),
                        Some((leaf_index, local_state)),
                    );
                }
                assert_eq!(
                    constraint.recursive_tokenizer_reset_state(leaf_index),
                    constraint.recursive_tokenizer_scoped_state(
                        leaf_index,
                        tokenizer_resets[leaf_index],
                    ),
                );
            }
            assert!(layout.leaves[1].state_count >= 2);
            let root_top = layout.leaves[0].state_offset;
            let middle_top_a = layout.leaves[1].state_offset;
            let middle_top_b = middle_top_a + 1;
            let leaf_top = layout.leaves[2].state_offset;
            let acc0 = crate::compiler::glr::accumulator::TerminalsDisallowed::new()
                .with_insert(10, 20);
            let acc1 = crate::compiler::glr::accumulator::TerminalsDisallowed::new()
                .with_insert(11, 21);
            let acc2 = crate::compiler::glr::accumulator::TerminalsDisallowed::new()
                .with_insert(12, 22);
            let acc3 = crate::compiler::glr::accumulator::TerminalsDisallowed::new()
                .with_insert(13, 23);
            let mixed = crate::compiler::glr::parser::ParserGSS::from_stacks(&[
                (vec![root_top], acc0.clone()),
                (vec![root_top, middle_top_a], acc1.clone()),
                (vec![root_top, middle_top_b], acc2.clone()),
                (vec![root_top, leaf_top], acc3.clone()),
            ]);
            let partitions = constraint
                .partition_recursive_parser_gss_by_active_leaf(&mixed)
                .expect("recursive GSS must partition by active tokenizer leaf");
            assert_eq!(
                partitions.iter().map(|(leaf, _)| *leaf).collect::<Vec<_>>(),
                vec![0, 1, 2],
            );
            let mut middle_stacks = partitions[1].1.to_stacks(16).unwrap();
            middle_stacks.sort_by_key(|(stack, _)| *stack.last().unwrap());
            assert_eq!(
                middle_stacks,
                vec![
                    (vec![root_top, middle_top_a], acc1),
                    (vec![root_top, middle_top_b], acc2),
                ],
            );
            assert_eq!(
                partitions[0].1.to_stacks(16).unwrap(),
                vec![(vec![root_top], acc0)],
            );
            assert_eq!(
                partitions[2].1.to_stacks(16).unwrap(),
                vec![(vec![root_top, leaf_top], acc3)],
            );
            assert_eq!(constraint.recursive_parser_state_span().unwrap(), layout.total_states);
            assert_eq!(layout.links.len(), 2);
            assert!(layout
                .links
                .iter()
                .any(|link| link.parent_component == 1 && link.child_component == 2));
            assert!(layout
                .links
                .iter()
                .any(|link| link.parent_component == 0 && link.child_component == 1));
        }
    }

    #[test]
    fn recursive_parser_reference_executes_nested_calls_without_materialized_child_table() {
        let vocab = Vocab::new(vec![
            (0, b"X".to_vec()),
            (1, b"[".to_vec()),
            (2, b"a".to_vec()),
            (3, b"]".to_vec()),
            (4, b"!".to_vec()),
        ]);
        let terminal = |constraint: &RuntimeConstraint, name: &str| {
            constraint
                .terminal_display_names
                .iter()
                .position(|candidate| candidate == name)
                .unwrap() as u32
        };
        let leaf = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start leaf; t A = \"a\"; nt leaf = A;"),
            &vocab,
        )
        .unwrap();
        let middle_parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start middle; extern grammar leaf; t L = \"[\"; t R = \"]\"; nt middle = L leaf R;",
            ),
            &vocab,
        )
        .unwrap();
        let middle = middle_parent
            .bind_grammar_dynamic_boundary("leaf", &leaf)
            .unwrap();
        let outer_parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start document; extern grammar middle; t X = \"X\"; t BANG = \"!\"; nt document = X middle BANG;",
            ),
            &vocab,
        )
        .unwrap();
        let bound = outer_parent
            .bind_grammar_dynamic_boundary("middle", &middle)
            .unwrap();

        let outer_overlay = bound.static_dynamic_overlay.as_ref().unwrap();
        let middle_outer_offset = outer_overlay.segmented_parser_components[1].terminal_offset;
        let middle_overlay = middle.static_dynamic_overlay.as_ref().unwrap();
        let middle_parent_offset = middle_overlay.segmented_parser_components[0].terminal_offset;
        let leaf_offset = middle_overlay.segmented_parser_components[1].terminal_offset;
        let terminals = [
            outer_overlay.segmented_parser_components[0].terminal_offset
                + terminal(&outer_parent, "X"),
            middle_outer_offset + middle_parent_offset + terminal(&middle_parent, "L"),
            middle_outer_offset + leaf_offset + terminal(&leaf, "A"),
            middle_outer_offset + middle_parent_offset + terminal(&middle_parent, "R"),
            outer_overlay.segmented_parser_components[0].terminal_offset
                + terminal(&outer_parent, "BANG"),
        ];

        for constraint in [&bound, &RuntimeConstraint::load(&bound.save()).unwrap()] {
            let start = crate::compiler::glr::parser::ParserGSS::from_single_stack(
                vec![0],
                crate::compiler::glr::accumulator::TerminalsDisallowed::new(),
            );
            let mut parser = constraint
                .close_recursive_segmented_parser_reference(&start)
                .unwrap()
                .unwrap();
            for &terminal in &terminals {
                parser = constraint
                    .advance_recursive_segmented_parser_reference(&parser, terminal)
                    .unwrap()
                    .unwrap();
                assert!(
                    !parser.is_empty(),
                    "recursive parser rejected terminal {terminal}",
                );
            }
            assert_eq!(
                constraint
                    .recursive_segmented_parser_is_finished_reference(&parser)
                    .unwrap(),
                Some(true),
            );

            let mut invalid = constraint
                .close_recursive_segmented_parser_reference(&start)
                .unwrap()
                .unwrap();
            for &terminal in &[terminals[0], terminals[1], terminals[3]] {
                invalid = constraint
                    .advance_recursive_segmented_parser_reference(&invalid, terminal)
                    .unwrap()
                    .unwrap();
            }
            assert!(invalid.is_empty(), "recursive parser accepted a missing leaf token");
        }
    }

    #[test]
    fn recursive_parser_reference_resolves_new_link_through_precomposed_parent() {
        let vocab = Vocab::new(vec![
            (0, b"<".to_vec()),
            (1, b"a".to_vec()),
            (2, b"b".to_vec()),
            (3, b">".to_vec()),
        ]);
        let terminal = |constraint: &RuntimeConstraint, name: &str| {
            constraint
                .terminal_display_names
                .iter()
                .position(|candidate| candidate == name)
                .unwrap() as u32
        };
        let parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start document; extern grammar left; extern grammar right; t LT = \"<\"; t GT = \">\"; nt document = LT left right GT;",
            ),
            &vocab,
        )
        .unwrap();
        let left = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start left; t A = \"a\"; nt left = A;"),
            &vocab,
        )
        .unwrap();
        let right = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start right; t B = \"b\"; nt right = B;"),
            &vocab,
        )
        .unwrap();
        let half = parent
            .bind_grammar_dynamic_boundary("left", &left)
            .unwrap();
        let full = half
            .bind_grammar_dynamic_boundary("right", &right)
            .unwrap();

        let full_overlay = full.static_dynamic_overlay.as_ref().unwrap();
        let half_offset = full_overlay.segmented_parser_components[0].terminal_offset;
        let right_offset = full_overlay.segmented_parser_components[1].terminal_offset;
        let half_overlay = half.static_dynamic_overlay.as_ref().unwrap();
        let original_parent_offset = half_overlay.segmented_parser_components[0].terminal_offset;
        let left_offset = half_overlay.segmented_parser_components[1].terminal_offset;
        let terminals = [
            half_offset + original_parent_offset + terminal(&parent, "LT"),
            half_offset + left_offset + terminal(&left, "A"),
            right_offset + terminal(&right, "B"),
            half_offset + original_parent_offset + terminal(&parent, "GT"),
        ];

        for constraint in [&full, &RuntimeConstraint::load(&full.save()).unwrap()] {
            let layout = constraint.recursive_parser_layout().unwrap().unwrap();
            assert_eq!(
                layout
                    .leaves
                    .iter()
                    .map(|leaf| leaf.component_path.clone())
                    .collect::<Vec<_>>(),
                vec![vec![0, 0], vec![0, 1], vec![1]],
            );
            assert_eq!(layout.links.len(), 2);
            assert!(layout
                .links
                .iter()
                .any(|link| link.parent_component == 0 && link.child_component == 1));
            assert!(layout
                .links
                .iter()
                .any(|link| link.parent_component == 0 && link.child_component == 2));

            let start = crate::compiler::glr::parser::ParserGSS::from_single_stack(
                vec![0],
                crate::compiler::glr::accumulator::TerminalsDisallowed::new(),
            );
            let mut parser = constraint
                .close_recursive_segmented_parser_reference(&start)
                .unwrap()
                .unwrap();
            for &terminal in &terminals {
                parser = constraint
                    .advance_recursive_segmented_parser_reference(&parser, terminal)
                    .unwrap()
                    .unwrap();
                assert!(!parser.is_empty(), "recursive parser rejected terminal {terminal}");
            }
            assert_eq!(
                constraint
                    .recursive_segmented_parser_is_finished_reference(&parser)
                    .unwrap(),
                Some(true),
            );
        }
    }

    #[test]
    fn loaded_recursive_composition_rebinds_from_lazy_compiler_views() {
        let vocab = Vocab::new(vec![
            (0, b"<".to_vec()),
            (1, b"a".to_vec()),
            (2, b"b".to_vec()),
            (3, b">".to_vec()),
            (4, b"<ab>".to_vec()),
        ]);
        let parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start document; extern grammar left; extern grammar right; nt document = \"<\" left right \">\";",
            ),
            &vocab,
        )
        .unwrap();
        let left = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start left; nt left = \"a\";"),
            &vocab,
        )
        .unwrap();
        let right = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start right; nt right = \"b\";"),
            &vocab,
        )
        .unwrap();

        let half = parent
            .bind_grammar_dynamic_boundary("left", &left)
            .unwrap();
        let half_overlay = half.static_dynamic_overlay.as_ref().unwrap();
        let half_layout = half.recursive_parser_layout().unwrap().unwrap();
        assert_eq!(
            half.tokenizer.num_states(),
            half_overlay.segmented_parser_components[0]
                .constraint
                .tokenizer
                .num_states(),
            "recursive coordinator must retain only the root-leaf tokenizer",
        );
        assert!(
            half.tokenizer.num_states() < half_layout.total_tokenizer_states,
            "test fixture must actually contain more than one tokenizer leaf",
        );
        assert_eq!(
            half.table.num_states, 0,
            "recursive coordinator must not retain a flattened LR state machine",
        );
        assert!(half.table.action.is_empty() && half.table.goto.is_empty());
        assert!(half_overlay
            .segmented_parser_components
            .iter()
            .all(|component| component.global_to_local_parser_state.is_empty()));

        let loaded_half = RuntimeConstraint::load_body_artifact(half.save()).unwrap();
        let loaded_half_overlay = loaded_half.static_dynamic_overlay.as_ref().unwrap();
        let loaded_half_layout = loaded_half.recursive_parser_layout().unwrap().unwrap();
        assert_eq!(
            loaded_half.tokenizer.num_states(),
            loaded_half_overlay.segmented_parser_components[0]
                .constraint
                .tokenizer
                .num_states(),
            "loaded recursive coordinator must not reconstruct the outer union tokenizer eagerly",
        );
        assert!(loaded_half.tokenizer.num_states() < loaded_half_layout.total_tokenizer_states);
        assert_eq!(
            loaded_half.table.num_states, 0,
            "loaded recursive coordinator must keep the flattened parser table lazy",
        );
        assert!(loaded_half.table.action.is_empty() && loaded_half.table.goto.is_empty());
        assert!(loaded_half_overlay
            .segmented_parser_components
            .iter()
            .all(|component| component.global_to_local_parser_state.is_empty()));

        let fresh_full = half
            .bind_grammar_dynamic_boundary("right", &right)
            .unwrap();
        let loaded_full = loaded_half
            .bind_grammar_dynamic_boundary("right", &right)
            .unwrap();
        for constraint in [&fresh_full, &loaded_full] {
            assert_recursive_compiler_views_detached(constraint);
        }
        let fresh_start = crate::compiler::glr::parser::ParserGSS::from_single_stack(
            vec![0],
            crate::compiler::glr::accumulator::TerminalsDisallowed::new(),
        );
        let loaded_start = fresh_start.clone();
        let fresh_closed = fresh_full
            .close_compact_segmented_parser(&fresh_start)
            .unwrap();
        let loaded_closed = loaded_full
            .close_compact_segmented_parser(&loaded_start)
            .unwrap();
        assert!(fresh_closed.semantically_eq(&loaded_closed, 4096).unwrap());
        for terminal in 0..fresh_full.table.num_terminals {
            let fresh_advanced = fresh_full
                .advance_compact_segmented_parser(&fresh_closed, terminal)
                .unwrap();
            let loaded_advanced = loaded_full
                .advance_compact_segmented_parser(&loaded_closed, terminal)
                .unwrap();
            assert!(
                fresh_advanced
                    .semantically_eq(&loaded_advanced, 4096)
                    .unwrap(),
                "recursive parser advance differs for terminal {terminal}",
            );
        }
        for constraint in [&fresh_full, &loaded_full] {
            let overlay = constraint.static_dynamic_overlay.as_ref().unwrap();
            assert!(overlay
                .segmented_parser_components
                .iter()
                .all(|component| component.global_to_local_parser_state.is_empty()));
            let mut split = constraint.start();
            for token in [0, 1, 2, 3] {
                assert_ne!(split.mask()[0] & (1 << token), 0);
                split.commit_token(token).unwrap();
            }
            assert!(split.is_accepting());

            let mut fused = constraint.start();
            assert_ne!(fused.mask()[0] & (1 << 4), 0);
            fused.commit_token(4).unwrap();
            assert!(fused.is_accepting());
        }
    }

    #[test]
    fn recursive_parser_reference_preserves_nested_nullable_wrapper_return() {
        let vocab = Vocab::new(vec![
            (0, b"X".to_vec()),
            (1, b"a".to_vec()),
            (2, b"!".to_vec()),
        ]);
        let terminal = |constraint: &RuntimeConstraint, name: &str| {
            constraint
                .terminal_display_names
                .iter()
                .position(|candidate| candidate == name)
                .unwrap() as u32
        };
        let leaf = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start leaf; t A = \"a\"; nt leaf = A?;"),
            &vocab,
        )
        .unwrap();
        let middle_parent = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start middle; extern grammar leaf; nt middle = leaf;"),
            &vocab,
        )
        .unwrap();
        assert!(!middle_parent.table.embedded_start_nullable());
        let middle = middle_parent
            .bind_grammar_dynamic_boundary("leaf", &leaf)
            .unwrap();
        assert!(middle.table.embedded_start_nullable());
        let outer_parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start document; extern grammar middle; t X = \"X\"; t BANG = \"!\"; nt document = X middle BANG;",
            ),
            &vocab,
        )
        .unwrap();
        let bound = outer_parent
            .bind_grammar_dynamic_boundary("middle", &middle)
            .unwrap();

        let outer_overlay = bound.static_dynamic_overlay.as_ref().unwrap();
        let middle_outer_offset = outer_overlay.segmented_parser_components[1].terminal_offset;
        let middle_overlay = middle.static_dynamic_overlay.as_ref().unwrap();
        let leaf_offset = middle_overlay.segmented_parser_components[1].terminal_offset;
        let x = outer_overlay.segmented_parser_components[0].terminal_offset
            + terminal(&outer_parent, "X");
        let a = middle_outer_offset + leaf_offset + terminal(&leaf, "A");
        let bang = outer_overlay.segmented_parser_components[0].terminal_offset
            + terminal(&outer_parent, "BANG");

        for constraint in [&bound, &RuntimeConstraint::load(&bound.save()).unwrap()] {
            let layout = constraint.recursive_parser_layout().unwrap().unwrap();
            let outer_link = layout
                .links
                .iter()
                .find(|link| link.parent_component == 0 && link.child_component == 1)
                .expect("outer link must target the middle wrapper root leaf");
            assert!(outer_link.child_start_nullable);

            for terminals in [&[x, bang][..], &[x, a, bang][..]] {
                let start = crate::compiler::glr::parser::ParserGSS::from_single_stack(
                    vec![0],
                    crate::compiler::glr::accumulator::TerminalsDisallowed::new(),
                );
                let mut parser = constraint
                    .close_recursive_segmented_parser_reference(&start)
                    .unwrap()
                    .unwrap();
                for &terminal in terminals {
                    parser = constraint
                        .advance_recursive_segmented_parser_reference(&parser, terminal)
                        .unwrap()
                        .unwrap();
                    assert!(!parser.is_empty(), "recursive nullable parser rejected {terminals:?}");
                }
                assert_eq!(
                    constraint
                        .recursive_segmented_parser_is_finished_reference(&parser)
                        .unwrap(),
                    Some(true),
                    "{terminals:?}",
                );
            }
        }
    }

    #[test]
    fn static_boundary_shards_are_authoritative_across_multiple_internal_crossings() {
        let vocab = Vocab::new(vec![
            (0, b"x".to_vec()),
            (1, b"y".to_vec()),
            (2, b"z".to_vec()),
            (3, b"xy".to_vec()),
            (4, b"yz".to_vec()),
            (5, b"xyz".to_vec()),
        ]);
        let parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start start; extern grammar left; extern grammar right; \
                 nt start = \"x\" left right;",
            ),
            &vocab,
        )
        .unwrap();
        let left = RuntimeConstraint::compile(Grammar::ebnf(r#"start ::= "y""#), &vocab)
            .unwrap();
        let right = RuntimeConstraint::compile(Grammar::ebnf(r#"start ::= "z""#), &vocab)
            .unwrap();
        let bound = parent
            .bind_grammar("left", left)
            .unwrap()
            .bind_grammar("right", right)
            .unwrap();

        let overlay = bound
            .static_dynamic_overlay
            .as_ref()
            .expect("static composition must retain segmented A+B metadata");
        assert!(
            !overlay.segmented_boundary_shards.is_empty(),
            "static composition must publish component-scoped B shards",
        );
        assert!(
            overlay.segmented_boundary_parser.is_none(),
            "partitioned static B must not retain a redundant global boundary parser",
        );
        assert!(overlay.segmented_boundary_terminal_trie.is_none());

        let loaded = RuntimeConstraint::load(bound.save()).unwrap();
        let loaded_overlay = loaded.static_dynamic_overlay.as_ref().unwrap();
        assert!(loaded_overlay.segmented_boundary_parser.is_none());
        assert!(loaded_overlay.segmented_boundary_terminal_trie.is_none());
        assert!(!loaded_overlay.segmented_boundary_shards.is_empty());
        let reference = RuntimeConstraint::compile(
            Grammar::ebnf(r#"start ::= "x" "y" "z""#),
            &vocab,
        )
        .unwrap();

        for tokens in [&[5][..], &[0, 4][..], &[0, 1, 2][..]] {
            let mut sharded = bound.start();
            let mut restored = loaded.start();
            let mut expected = reference.start();
            for &token in tokens {
                assert_eq!(
                    sharded.mask(),
                    expected.mask(),
                    "partitioned static B mismatch before {tokens:?} token {token}",
                );
                assert_eq!(
                    restored.mask(),
                    expected.mask(),
                    "restored static shards differ before {tokens:?} token {token}",
                );
                sharded.commit_token(token).unwrap();
                restored.commit_token(token).unwrap();
                expected.commit_token(token).unwrap();
            }
            assert_eq!(sharded.is_accepting(), expected.is_accepting(), "{tokens:?}");
            assert_eq!(restored.is_accepting(), expected.is_accepting(), "{tokens:?}");
            assert!(sharded.is_accepting(), "{tokens:?}");
        }
    }

    #[test]
    fn authoritative_ab_keeps_component_parser_dwas_unchanged_across_scoped_ignores() {
        let vocab = Vocab::new(vec![
            (0, b"X".to_vec()),
            (1, b" ".to_vec()),
            (2, b"\t".to_vec()),
            (3, b"a".to_vec()),
            (4, b"!".to_vec()),
            (5, b"X \ta!".to_vec()),
            (6, b"X\t a!".to_vec()),
            (7, b"Xa\t !".to_vec()),
            (8, b"Xa \t!".to_vec()),
        ]);
        let parent = RuntimeConstraint::compile(
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
        let child = RuntimeConstraint::compile(
            Grammar::glrm(
                r#"
                    glrm 1;
                    start child;
                    ignore CHILD_WS;
                    t CHILD_WS = "\t"+;
                    nt child = "a";
                "#,
            ),
            &vocab,
        )
        .unwrap();
        assert!(
            !parent.composition_reset_tokens_by_terminal.is_empty(),
            "late-bind parent compilation should precompute reset-token composition metadata",
        );
        let mut cached_parent = RuntimeConstraint::load_body_artifact(parent.save()).unwrap();
        cached_parent
            .materialize_composition_metadata_for_compilation()
            .unwrap();
        assert_eq!(
            cached_parent.composition_reset_tokens_by_terminal,
            parent.composition_reset_tokens_by_terminal,
            "late-bind reset-token cache must survive save/load",
        );
        let monolithic = RuntimeConstraint::compile(
            Grammar::glrm(
                r#"
                    glrm 1;
                    start document;
                    ignore PARENT_WS;
                    t PARENT_WS = " "+;
                    g child = {
                        start child;
                        ignore CHILD_WS;
                        t CHILD_WS = "\t"+;
                        nt child = "a";
                    };
                    nt document = "X" child "!";
                "#,
            ),
            &vocab,
        )
        .unwrap();

        let parent_dwa = parent.parser_dwa.clone();
        let child_dwa = child.parser_dwa.clone();
        let bound = parent
            .bind_grammar_dynamic_boundary("child", child.clone())
            .unwrap();
        let overlay = bound
            .static_dynamic_overlay
            .as_ref()
            .expect("authoritative A+B metadata");
        assert!(overlay.segmented_mask_authoritative);
        assert!(!overlay.segmented_static_baseline);
        assert_eq!(overlay.segmented_parser_components.len(), 2);
        assert_eq!(overlay.segmented_parser_components[0].constraint.parser_dwa, parent_dwa);
        assert_eq!(overlay.segmented_parser_components[1].constraint.parser_dwa, child_dwa);
        assert!(overlay
            .segmented_parser_components
            .iter()
            .all(|component| component.root_disallowed_terminal.is_none()));

        let loaded = RuntimeConstraint::load(bound.save()).unwrap();
        assert!(bound.uses_compact_segmented_parser_runtime());
        assert!(loaded.uses_compact_segmented_parser_runtime());
        let loaded_overlay = loaded
            .static_dynamic_overlay
            .as_ref()
            .expect("round-tripped authoritative A+B metadata");
        assert_eq!(loaded_overlay.segmented_parser_components.len(), 2);
        let mut loaded_parent = loaded_overlay.segmented_parser_components[0]
            .constraint
            .as_ref()
            .clone();
        let mut loaded_child = loaded_overlay.segmented_parser_components[1]
            .constraint
            .as_ref()
            .clone();
        loaded_parent.materialize_parser_dwa_for_compilation().unwrap();
        loaded_child.materialize_parser_dwa_for_compilation().unwrap();
        assert_eq!(loaded_parent.parser_dwa, parent_dwa);
        assert_eq!(loaded_child.parser_dwa, child_dwa);
        for (kind, constraint) in [("source", &bound), ("loaded", &loaded)] {
            let mut actual = constraint.start();
            let mut expected = monolithic.start();
            assert_eq!(actual.mask(), expected.mask(), "{kind} initial mask");
            for token in [1, 0, 2, 3, 1, 4] {
                actual.commit_token(token).unwrap();
                expected.commit_token(token).unwrap();
                assert_eq!(actual.mask(), expected.mask(), "after token {token}");
            }
            assert!(actual.is_accepting());
            assert!(expected.is_accepting());

            for token in [5, 6, 7, 8] {
                let mut actual = constraint.start();
                let mut expected = monolithic.start();
                assert_eq!(
                    actual.commit_token(token).is_ok(),
                    expected.commit_token(token).is_ok(),
                    "fused token {token}",
                );
                assert_eq!(actual.is_accepting(), expected.is_accepting());
            }
        }
    }

    #[test]
    fn authoritative_ab_handles_multi_terminal_parent_token_after_zero_width_return() {
        let vocab = Vocab::new(vec![
            (0, b"X".to_vec()),
            (1, b"a".to_vec()),
            (2, b"!".to_vec()),
            (3, b"?".to_vec()),
            (4, b"!?".to_vec()),
        ]);
        let parent = RuntimeConstraint::compile(
            Grammar::glrm(
                r#"
                    glrm 1;
                    start document;
                    extern grammar child;
                    nt document = "X" child "!" "?";
                "#,
            ),
            &vocab,
        )
        .unwrap();
        let child = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start child; nt child = \"a\";"),
            &vocab,
        )
        .unwrap();
        let monolithic = RuntimeConstraint::compile(
            Grammar::glrm(
                r#"
                    glrm 1;
                    start document;
                    g child = { start child; nt child = "a"; };
                    nt document = "X" child "!" "?";
                "#,
            ),
            &vocab,
        )
        .unwrap();

        let bound = parent
            .bind_grammar_dynamic_boundary("child", child)
            .unwrap();
        for constraint in [&bound, &RuntimeConstraint::load(bound.save()).unwrap()] {
            let mut actual = constraint.start();
            let mut expected = monolithic.start();
            for token in [0, 1] {
                assert_eq!(actual.mask(), expected.mask(), "before token {token}");
                actual.commit_token(token).unwrap();
                expected.commit_token(token).unwrap();
            }
            let actual_mask = actual.mask();
            let expected_mask = expected.mask();
            assert_eq!(actual_mask, expected_mask, "after child completion");
            assert_ne!(actual_mask[0] & (1 << 4), 0, "fused parent token !? must be allowed");
            actual.commit_token(4).unwrap();
            expected.commit_token(4).unwrap();
            assert!(actual.is_accepting());
            assert!(expected.is_accepting());
        }
    }

    #[test]
    fn dynamic_late_bind_vocab_cache_is_shared_reused_and_not_serialized() {
        let vocab = Vocab::new(vec![
            (0, b"x".to_vec()), (1, b"y".to_vec()), (2, b"xy".to_vec()),
        ]);
        let parent = DynamicConstraint::compile(
            Grammar::glrm("glrm 1; start start; extern grammar child; nt start = \"x\" child;"),
            &vocab,
        ).unwrap();
        let child = DynamicConstraint::compile(Grammar::ebnf(r#"start ::= "y""#), &vocab).unwrap();
        let retained = parent.inner.late_bind_vocab.get().expect("retain supplied vocabulary");
        assert!(Arc::ptr_eq(&retained.entries_arc(), &vocab.entries_arc()));

        let loaded = DynamicConstraint::load(&parent.save()).unwrap();
        assert!(loaded.inner.late_bind_vocab.get().is_none());
        let first = loaded.bind_grammar_dynamic_boundary("child", &child).unwrap();
        let first_backing = loaded.inner.late_bind_vocab.get()
            .expect("prime retained loaded parent, not a throwaway clone").entries_arc();
        let second = loaded.bind_grammar_dynamic_boundary("child", &child).unwrap();
        assert!(Arc::ptr_eq(
            &first_backing, &loaded.inner.late_bind_vocab.get().unwrap().entries_arc(),
        ));
        assert_eq!(first.start().mask(), second.start().mask());
        let reloaded = DynamicConstraint::load(&loaded.save()).unwrap();
        assert!(reloaded.inner.late_bind_vocab.get().is_none(), "cache stays off wire");
        let rebound = reloaded.bind_grammar_dynamic_boundary("child", &child).unwrap();
        assert_eq!(first.start().mask(), rebound.start().mask());
    }

    #[test]
    fn late_bind_vocab_cache_is_reused_and_not_serialized() {
        let vocab = Vocab::new(vec![
            (0, b"x".to_vec()),
            (1, b"y".to_vec()),
            (2, b"xy".to_vec()),
        ]);
        let parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start start; extern grammar child; nt start = \"x\" child;",
            ),
            &vocab,
        )
        .unwrap();
        let child =
            DynamicConstraint::compile(Grammar::ebnf(r#"start ::= "y""#), &vocab).unwrap();

        assert!(
            parent.late_bind_vocab.get().is_some(),
            "fresh compilation should retain the supplied vocabulary for later binds",
        );

        let loaded_parent = RuntimeConstraint::load_body_artifact(parent.save()).unwrap();
        assert!(
            loaded_parent.late_bind_vocab.get().is_none(),
            "late-bind vocabulary memoization is runtime-only and must not enter the wire format",
        );
        let first = loaded_parent
            .bind_grammar_dynamic_boundary("child", &child)
            .unwrap();
        assert!(loaded_parent.late_bind_vocab.get().is_some());
        let second = loaded_parent
            .bind_grammar_dynamic_boundary("child", &child)
            .unwrap();
        assert_eq!(first.start().mask(), second.start().mask());

        let loaded = RuntimeConstraint::load_body_artifact(loaded_parent.save()).unwrap();
        assert!(
            loaded.late_bind_vocab.get().is_none(),
            "late-bind vocabulary memoization is runtime-only and must not enter the wire format",
        );
        let rebound = loaded
            .bind_grammar_dynamic_boundary("child", &child)
            .unwrap();
        assert_eq!(first.start().mask(), rebound.start().mask());
    }
