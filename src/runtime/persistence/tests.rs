    use super::*;
    use crate::Vocab;
    use crate::automata::unweighted_u32::dfa::DFA as UnweightedDfa;
    use crate::runtime::CommitTemplateDfas;
    use std::sync::Arc;

    #[test]
    fn packed_reencode_preserves_non_dwa_ids_after_cache_invalidation() {
        let vocab = Vocab::new(vec![
            (0, b"x".to_vec()), (1, b"a".to_vec()), (2, b"b".to_vec()),
            (3, b"y".to_vec()), (4, b"xay".to_vec()), (5, b"ab".to_vec()),
            (7, b"a".to_vec()), (8, Vec::new()),
        ]);
        let original = Constraint::from_ebnf(
            r#"start ::= @token(9) "b" | "a" "y""#,
            &vocab,
        ).unwrap();
        let mut loaded = Constraint::load(original.save()).unwrap();
        let packed = loaded.packed_non_dwa_weights.as_ref().unwrap();
        let distinct_ids = packed.parser_top_accept.values()
            .copied().collect::<std::collections::BTreeSet<_>>();
        assert!(distinct_ids.len() >= 2, "regression needs distinct packed references");

        // Linking can invalidate a cached artifact while retaining its packed
        // runtime pools. Re-serialization must preserve the sidecar IDs, not
        // intern the empty structural Weight placeholders by pointer.
        loaded.serialized_artifact_cache = None;
        let again = Constraint::load(loaded.save()).unwrap();
        assert_eq!(loaded.packed_non_dwa_weights.as_ref().unwrap().parser_top_accept,
                   again.packed_non_dwa_weights.as_ref().unwrap().parser_top_accept);
        for path in [&[][..], &[1], &[1, 3], &[9], &[9, 2]] {
            let mut expected = original.start();
            let mut actual = again.start();
            for &id in path {
                assert_eq!(expected.mask(), actual.mask());
                expected.commit_token(id).unwrap();
                actual.commit_token(id).unwrap();
            }
            assert_eq!(expected.mask(), actual.mask(), "after {path:?}");
            assert_eq!(expected.is_accepting(), actual.is_accepting());
        }
    }

    fn tiny_constraint() -> Constraint {
        Constraint::from_glrm_grammar(
            r#"
                start start;
                t A ::= "a";
                t B ::= "b";
                nt start ::= A B;
            "#,
            &Vocab::new(vec![
                (0, b"a".to_vec()),
                (1, b"b".to_vec()),
                (2, b"ab".to_vec()),
            ]),
        )
        .unwrap()
    }

    fn sample_boundary_fingerprint() -> crate::runtime::BoundaryCandidateFingerprint {
        crate::runtime::BoundaryCandidateFingerprint {
            algorithm_version: 77,
            component_semantics: [7; 32],
            public_interface: [8; 32],
            vocabulary: [9; 32],
        }
    }

    fn ignored_constraint() -> Constraint {
        Constraint::from_glrm_grammar(
            r#"
                start start;
                ignore WS;
                t WS ::= " "+;
                nt start ::= "a";
            "#,
            &Vocab::new(vec![(0, b"a".to_vec()), (1, b" ".to_vec())]),
        )
        .unwrap()
    }

    #[test]
    fn static_bounded_uri_virtual_projection_roundtrips_with_packed_vocab() {
        let vocab = Vocab::new(vec![
            (0, b"\"".to_vec()),
            (1, b"x:".to_vec()),
            (2, b"a".to_vec()),
            (3, b"aa".to_vec()),
        ]);
        let schema = r#"{"type":"string","format":"uri","minLength":1,"maxLength":5000}"#;
        let constraint = Constraint::from_json_schema(schema, &vocab)
            .expect("large bounded URI should compile through the exact Static residual lane");
        assert!(constraint.tokenizer.has_virtual_residual_runtime());
        assert!(constraint.dynamic_mask_vocab.mask_projection_tokenizer().is_some());

        let saved = constraint.save();
        assert!(saved.windows(4).any(|window| window == STATIC_RESIDUAL_MASK_MAGIC));
        assert!(saved.windows(4).any(|window| window == b"TKS3"));
        assert!(saved.windows(4).any(|window| window == b"BCO2"));
        let loaded = Constraint::load(saved.clone())
            .expect("Static residual artifact should round-trip");
        assert!(loaded.tokenizer.has_virtual_residual_runtime());
        assert!(loaded.dynamic_mask_vocab.mask_projection_tokenizer().is_some());
        assert!(loaded.token_bytes.is_empty());
        assert!(loaded.packed_token_bytes.is_some());

        let compare_mask = |prefix: &[u8]| {
            let mut expected = constraint.start();
            let mut actual = loaded.start();
            expected.commit_bytes(prefix).unwrap();
            actual.commit_bytes(prefix).unwrap();
            assert_eq!(actual.mask(), expected.mask(), "prefix length {}", prefix.len());
        };
        compare_mask(b"\"x:");

        let mut at_max = b"\"x:".to_vec();
        at_max.extend(std::iter::repeat_n(b'a', 4_998));
        compare_mask(&at_max);

        let mut expected = constraint.start();
        let mut actual = loaded.start();
        expected.commit_bytes(&at_max).unwrap();
        actual.commit_bytes(&at_max).unwrap();
        expected.commit_bytes(b"\"").unwrap();
        actual.commit_bytes(b"\"").unwrap();
        assert!(expected.is_accepting());
        assert!(actual.is_accepting());

        let mut over_max = b"\"x:".to_vec();
        over_max.extend(std::iter::repeat_n(b'a', 4_999));
        assert!(constraint.start().commit_bytes(&over_max).is_err());
        assert!(loaded.start().commit_bytes(&over_max).is_err());

        let mut corrupted = saved;
        let marker = corrupted
            .windows(4)
            .position(|window| window == STATIC_RESIDUAL_MASK_MAGIC)
            .expect("SRM3 marker must be present");
        corrupted[marker] ^= 0x01;
        assert!(
            Constraint::load(corrupted).is_err(),
            "corrupted SRM3 marker must fail closed",
        );
    }

    #[test]
    fn fresh_packed_non_dwa_weight_runtime_matches_materialized_and_roundtrips() {
        let baseline = tiny_constraint();
        let mut packed = tiny_constraint();
        let (weights, _, _) = constraint_serialized_weight_pool_with_ids(&packed);
        assert!(!weights.is_empty());
        assert!(packed.packed_non_dwa_weights.is_none());
        assert!(compact_non_dwa_weight_runtime_if_at_least(&mut packed, 0));
        assert!(packed.packed_non_dwa_weights.is_some());

        let mut expected = baseline.start();
        let mut actual = packed.start();
        assert_eq!(actual.mask(), expected.mask());
        expected.commit_token(0).unwrap();
        actual.commit_token(0).unwrap();
        assert_eq!(actual.mask(), expected.mask());

        let loaded = Constraint::load(packed.save()).unwrap();
        let mut loaded_state = loaded.start();
        assert_eq!(loaded_state.mask(), baseline.start().mask());
        loaded_state.commit_token(0).unwrap();
        assert_eq!(loaded_state.mask(), expected.mask());
    }

    #[test]
    fn compact_seed_terminal_dense_deduplicates_and_roundtrips() {
        let mut original = crate::runtime::artifact::SeedTerminalDenseMasks::default();
        original.insert((3, 7), Arc::<[u64]>::from([1, 2, 3, 4]));
        original.insert((9, 11), Arc::<[u64]>::from([1, 2, 3, 4]));
        original.insert((12, 13), Arc::<[u64]>::from([8, 9]));

        let compact = SeedTerminalDenseCompact::from_map(&original);
        assert_eq!(compact.masks.len(), 2);
        assert_eq!(compact.entries.len(), 3);
        let decoded = compact.into_map().unwrap();
        assert_eq!(decoded, original);
        assert!(Arc::ptr_eq(
            decoded.get(&(3, 7)).unwrap(),
            decoded.get(&(9, 11)).unwrap(),
        ));
    }

    #[test]
    fn constraint_envelope_roundtrips_and_rejects_previous_formats() {
        let constraint = tiny_constraint();
        let saved = constraint.save();
        assert!(saved.starts_with(&CONSTRAINT_MAGIC));
        assert!(bincode::deserialize::<DeserializedConstraint>(&saved).is_err());
        let loaded = Constraint::load(&saved).unwrap();
        assert_eq!(loaded.start().mask(), constraint.start().mask());

        let raw = bincode::serialize(&SerializedConstraint(&constraint)).unwrap();
        assert!(Constraint::load(&raw)
            .unwrap_err()
            .to_string()
            .contains("header"));

        for version in 0..=CONSTRAINT_VERSION + 1 {
            if version == CONSTRAINT_VERSION { continue; }
            let mut previous = saved.clone();
            previous[8..10].copy_from_slice(&version.to_le_bytes());
            assert!(Constraint::load(&previous).unwrap_err().to_string().contains("unsupported"),
                "unexpected acceptance of version {version}");
        }
    }

    #[test]
    fn load_moves_owned_vec_and_accepts_borrowed_bytes() {
        let constraint = tiny_constraint();

        let owned = constraint.save();
        let owned_ptr = owned.as_ptr();
        let loaded_owned = Constraint::load(owned).unwrap();
        let owned_backing = loaded_owned
            .serialized_artifact_cache
            .as_ref()
            .expect("current owned load should retain artifact backing");
        assert_eq!(owned_backing.as_ptr(), owned_ptr);

        let borrowed = constraint.save();
        let borrowed_ptr = borrowed.as_ptr();
        let loaded_borrowed = Constraint::load(borrowed.as_slice()).unwrap();
        let borrowed_backing = loaded_borrowed
            .serialized_artifact_cache
            .as_ref()
            .expect("current borrowed load should create retained backing");
        assert_ne!(borrowed_backing.as_ptr(), borrowed_ptr);
        assert_eq!(loaded_borrowed.start().mask(), constraint.start().mask());

        // `&Vec<u8>` is a common caller shape and should remain ergonomic.
        let loaded_vec_ref = Constraint::load(&borrowed).unwrap();
        assert_eq!(loaded_vec_ref.start().mask(), constraint.start().mask());
    }

    #[test]
    fn small_public_compile_defers_first_save_artifact() {
        let vocab = crate::Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);
        let constraint = Constraint::compile(
            crate::Grammar::glrm("glrm 1;\nstart start;\nt A = \"a\";\nnt start = A;\n"),
            &vocab,
        )
        .unwrap();
        assert!(
            constraint.serialized_artifact_cache.is_none(),
            "ordinary small compiles should not prepay a complete save artifact",
        );
        let saved = constraint.save();
        let loaded = Constraint::load(saved).unwrap();
        assert_eq!(loaded.start().mask(), constraint.start().mask());
    }

    #[test]
    fn owned_load_keeps_token_mask_prefix_matrix_in_artifact_backing() {
        let constraint = tiny_constraint();
        let saved = constraint.save();
        let loaded = Constraint::load(saved).unwrap();

        let backing = loaded
            .serialized_artifact_cache
            .as_ref()
            .expect("owned current load should retain artifact backing");
        let prefix = loaded
            .word_group_prefix_buf_masks
            .as_contiguous()
            .expect("current token-mask prefix matrix should be contiguous");
        assert!(!prefix.is_empty());
        let backing_start = backing.as_ptr() as usize;
        let backing_end = backing_start + backing.len();
        let prefix_start = prefix.as_ptr() as usize;
        let prefix_end = prefix_start + std::mem::size_of_val(prefix);
        assert!(
            prefix_start >= backing_start && prefix_end <= backing_end,
            "loaded token-mask prefix matrix should borrow the retained artifact allocation"
        );

        let mut expected = constraint.start();
        let mut actual = loaded.start();
        assert_eq!(actual.mask(), expected.mask());
        expected.commit_token(0).unwrap();
        actual.commit_token(0).unwrap();
        assert_eq!(actual.mask(), expected.mask());
    }

    #[test]
    fn current_save_handles_packed_dwa_without_fast_wire_length() {
        let mut constraint = tiny_constraint();
        let packed = crate::automata::weighted::dwa::PackedRuntimeDwa::from_dwa(
            &constraint.parser_dwa,
        )
        .unwrap();
        assert!(
            packed.fast_wire_len().is_none(),
            "tiny fallback fixture should require actual wire emission for sizing",
        );
        let wire = packed.fast_wire_bytes();
        assert!(wire.starts_with(b"DWF"));

        // This is the fallback branch the old DWF5-specific regression was
        // really protecting: when no exact precomputed wire length exists, the
        // outer serializer must use the actual emitted wire rather than a stale
        // estimate.
        constraint.packed_parser_dwa = Some(Arc::new(packed));
        let saved = constraint.save();
        let loaded = Constraint::load(&saved).unwrap();
        assert_eq!(loaded.start().mask(), constraint.start().mask());
    }

    #[test]
    fn current_constraint_artifact_rejects_invalid_parser_state_domain_map() {
        let mut constraint = ignored_constraint();
        constraint.parser_state_domain_labels = vec![0];
        let error = Constraint::load(&constraint.save()).unwrap_err().to_string();
        assert!(error.contains("parser-state domain map") || error.contains("domain label"));
    }

    #[test]
    fn current_constraint_artifact_preserves_parser_state_domain_labels() {
        let mut constraint = ignored_constraint();
        constraint.parser_state_domain_labels =
            vec![i32::MAX; constraint.table.num_states as usize];
        if let Some(first) = constraint.parser_state_domain_labels.first_mut() {
            *first = constraint.table.num_states as i32;
        }
        let loaded = Constraint::load(&constraint.save()).unwrap();
        assert_eq!(
            loaded.parser_state_domain_labels,
            constraint.parser_state_domain_labels,
        );
    }

    #[test]
    fn current_constraint_artifact_preserves_static_dynamic_overlay() {
        let mut constraint = tiny_constraint();
        constraint.static_dynamic_overlay = Some(crate::runtime::artifact::StaticDynamicOverlayMetadata {
            terminal_offsets: vec![0, 3, 7],
            tokenizer_state_offsets: vec![1, 11, 29],
            repair_terminals: vec![false, true, false, true],
            non_parent_only_parser_states: vec![true, false, true],
            ..Default::default()
        });

        let loaded = Constraint::load(&constraint.save()).unwrap();
        let overlay = loaded
            .static_dynamic_overlay
            .as_ref()
            .expect("current artifact should preserve composition overlay metadata");
        assert_eq!(overlay.terminal_offsets, vec![0, 3, 7]);
        assert_eq!(overlay.tokenizer_state_offsets, vec![1, 11, 29]);
        assert_eq!(overlay.repair_terminals, vec![false, true, false, true]);
        assert_eq!(
            overlay.non_parent_only_parser_states,
            vec![true, false, true],
        );
        assert!(overlay.segmented_parser_components.is_empty());
        assert!(overlay.segmented_component_union_root_dispatch.is_empty());
        assert!(overlay.segmented_boundary_parser.is_none());
        assert!(overlay.segmented_boundary_terminal_trie.is_none());
        assert_eq!(loaded.start().mask(), constraint.start().mask());
    }

    #[test]
    fn current_constraint_artifact_preserves_boundary_token_trigger() {
        let mut constraint = tiny_constraint();
        constraint.boundary_trigger = crate::runtime::BoundaryTrigger::Tokens(Arc::from(
            vec![1u32, 3u32].into_boxed_slice(),
        ));
        let mut loaded = Constraint::load(&constraint.save()).unwrap();
        assert!(matches!(
            loaded.boundary_trigger,
            crate::runtime::BoundaryTrigger::None
        ));
        loaded
            .materialize_composition_link_metadata_for_compilation()
            .unwrap();
        assert_eq!(loaded.boundary_trigger.token_summary(), Some(&[1u32, 3u32][..]));
    }

    #[test]
    fn current_constraint_artifact_preserves_boundary_candidate_summary() {
        let constraint = tiny_constraint();
        constraint
            .boundary_candidate_summary
            .set(crate::runtime::BoundaryCandidateSummary::Known {
                fingerprint: crate::runtime::BoundaryCandidateFingerprint {
                    algorithm_version: 7,
                    component_semantics: [1; 32],
                    public_interface: [2; 32],
                    vocabulary: [3; 32],
                },
                tokens: crate::runtime::OriginalTokenSet::Sparse(Arc::from(
                    vec![0u32, 2u32].into_boxed_slice(),
                )),
                precision: crate::runtime::SummaryPrecision::RegularUpperBound,
            })
            .unwrap();
        let mut loaded = Constraint::load(&constraint.save()).unwrap();
        assert!(loaded.boundary_candidate_summary.get().is_none());
        loaded
            .materialize_composition_link_metadata_for_compilation()
            .unwrap();
        let summary = loaded
            .boundary_candidate_summary
            .get()
            .expect("summary must materialize from CMS5");
        match summary {
            crate::runtime::BoundaryCandidateSummary::Known {
                fingerprint,
                tokens,
                precision,
            } => {
                assert_eq!(fingerprint.algorithm_version, 7);
                assert_eq!(fingerprint.component_semantics, [1; 32]);
                assert_eq!(fingerprint.public_interface, [2; 32]);
                assert_eq!(fingerprint.vocabulary, [3; 32]);
                assert_eq!(tokens.canonical_ids(loaded.token_bytes_iter()), vec![0, 2]);
                assert_eq!(*precision, crate::runtime::SummaryPrecision::RegularUpperBound);
            }
            other => panic!("unexpected materialized summary: {other:?}"),
        }
    }

    #[test]
    fn current_constraint_artifact_preserves_empty_all_and_unknown_boundary_summaries() {
        let mut dense = crate::ds::bitset::BitSet::new(3);
        dense.set(0);
        dense.set(2);
        for summary in [
            crate::runtime::BoundaryCandidateSummary::Known {
                fingerprint: sample_boundary_fingerprint(),
                tokens: crate::runtime::OriginalTokenSet::Empty,
                precision: crate::runtime::SummaryPrecision::RegularUpperBound,
            },
            crate::runtime::BoundaryCandidateSummary::Known {
                fingerprint: sample_boundary_fingerprint(),
                tokens: crate::runtime::OriginalTokenSet::Dense(Arc::new(dense)),
                precision: crate::runtime::SummaryPrecision::RegularUpperBound,
            },
            crate::runtime::BoundaryCandidateSummary::Known {
                fingerprint: sample_boundary_fingerprint(),
                tokens: crate::runtime::OriginalTokenSet::AllByteTokensAtLeastTwo,
                precision: crate::runtime::SummaryPrecision::BudgetWidenedUpperBound,
            },
            crate::runtime::BoundaryCandidateSummary::Unknown {
                reason: crate::runtime::SummaryUnavailable::Deferred,
            },
        ] {
            let constraint = tiny_constraint();
            constraint.boundary_candidate_summary.set(summary.clone()).unwrap();
            let mut loaded = Constraint::load(&constraint.save()).unwrap();
            loaded
                .materialize_composition_link_metadata_for_compilation()
                .unwrap();
            let restored = loaded
                .boundary_candidate_summary
                .get()
                .expect("CMS5 summary must materialize");
            match (&summary, restored) {
                (
                    crate::runtime::BoundaryCandidateSummary::Known { tokens: expected, .. },
                    crate::runtime::BoundaryCandidateSummary::Known { tokens: actual, .. },
                ) => assert_eq!(
                    expected.canonical_ids(constraint.token_bytes_iter()),
                    actual.canonical_ids(loaded.token_bytes_iter()),
                ),
                (
                    crate::runtime::BoundaryCandidateSummary::Unknown { reason: expected },
                    crate::runtime::BoundaryCandidateSummary::Unknown { reason: actual },
                ) => assert_eq!(expected, actual),
                pair => panic!("summary variant changed across CMS5 roundtrip: {pair:?}"),
            }
        }
    }

    #[test]
    fn current_boundary_summary_preserves_duplicate_byte_token_ids() {
        let vocab = Vocab::new(vec![
            (10, b"ab".to_vec()),
            (11, b"ab".to_vec()),
            (12, b"a".to_vec()),
        ]);
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start doc;
                nt doc ::= "a";
            "#,
            &vocab,
        )
        .unwrap();
        constraint
            .boundary_candidate_summary
            .set(crate::runtime::BoundaryCandidateSummary::Known {
                fingerprint: sample_boundary_fingerprint(),
                tokens: crate::runtime::OriginalTokenSet::Sparse(Arc::from(
                    vec![10u32, 11u32].into_boxed_slice(),
                )),
                precision: crate::runtime::SummaryPrecision::RegularUpperBound,
            })
            .unwrap();
        let mut loaded = Constraint::load(&constraint.save()).unwrap();
        loaded
            .materialize_composition_link_metadata_for_compilation()
            .unwrap();
        let crate::runtime::BoundaryCandidateSummary::Known { tokens, .. } = loaded
            .boundary_candidate_summary
            .get()
            .expect("duplicate-ID summary must materialize")
        else {
            panic!("duplicate-ID summary changed variant");
        };
        assert_eq!(tokens.canonical_ids(loaded.token_bytes_iter()), vec![10, 11]);
    }

    #[test]
    fn current_constraint_artifact_preserves_composition_reset_tokens() {
        let mut constraint = tiny_constraint();
        constraint.ensure_composition_reset_tokens_by_terminal();
        assert_eq!(
            constraint.composition_reset_tokens_by_terminal.len(),
            constraint.table.num_terminals as usize,
        );
        assert!(constraint
            .composition_reset_tokens_by_terminal
            .iter()
            .any(|row| !row.is_empty()));
        let expected = constraint.composition_reset_tokens_by_terminal.clone();
        let mut loaded = Constraint::load(&constraint.save()).unwrap();
        assert!(loaded.composition_reset_tokens_by_terminal.is_empty());
        assert!(loaded.deferred_composition_metadata_blob.is_some());
        assert_eq!(loaded.start().mask(), constraint.start().mask());
        loaded
            .materialize_composition_metadata_for_compilation()
            .unwrap();
        assert_eq!(loaded.composition_reset_tokens_by_terminal, expected);
        assert_eq!(loaded.start().mask(), constraint.start().mask());
    }

    #[test]
    fn retained_template_read_is_exact_and_does_not_materialize_constraint() {
        let original = tiny_constraint();
        let expected = original.composition_parser_templates_by_terminal.clone();
        assert!(!expected.is_empty());
        assert!(matches!(original.retained_parser_templates_for_compilation().unwrap(), Cow::Borrowed(_)));
        let bytes = original.save();
        let loaded = Constraint::load(&bytes).unwrap();
        assert!(loaded.composition_parser_templates_by_terminal.is_empty());
        let templates = loaded.retained_parser_templates_for_compilation().unwrap();
        assert!(matches!(templates, Cow::Owned(_)));
        assert_eq!(templates.as_ref(), expected.as_slice());
        assert!(loaded.composition_parser_templates_by_terminal.is_empty());
        assert!(loaded.composition_parser_characterizations_by_terminal.is_empty());
        assert!(loaded.deferred_composition_metadata_blob.is_some());
        assert_eq!(loaded.save(), bytes);
        assert_eq!(loaded.start().mask(), original.start().mask());
    }

    #[test]
    fn current_constraint_artifact_preserves_composition_parser_templates() {
        let constraint = tiny_constraint();
        assert_eq!(
            constraint.composition_parser_templates_by_terminal.len(),
            constraint.table.num_terminals as usize,
        );
        assert!(constraint
            .composition_parser_templates_by_terminal
            .iter()
            .any(Option::is_some));
        let mut loaded = Constraint::load(&constraint.save()).unwrap();
        assert!(loaded.composition_parser_templates_by_terminal.is_empty());
        assert!(loaded.deferred_composition_metadata_blob.is_some());
        assert_eq!(loaded.start().mask(), constraint.start().mask());
        loaded
            .materialize_composition_metadata_for_compilation()
            .unwrap();
        assert_eq!(
            loaded.composition_parser_templates_by_terminal,
            constraint.composition_parser_templates_by_terminal,
        );
        assert_eq!(loaded.start().mask(), constraint.start().mask());
    }

    #[test]
    fn current_constraint_artifact_preserves_composition_parser_characterizations() {
        let constraint = tiny_constraint();
        assert_eq!(
            constraint
                .composition_parser_characterizations_by_terminal
                .len(),
            constraint.table.num_terminals as usize,
        );
        assert!(constraint
            .composition_parser_characterizations_by_terminal
            .iter()
            .any(Option::is_some));
        let mut loaded = Constraint::load(&constraint.save()).unwrap();
        assert!(loaded
            .composition_parser_characterizations_by_terminal
            .is_empty());
        assert!(loaded.deferred_composition_metadata_blob.is_some());
        assert_eq!(loaded.start().mask(), constraint.start().mask());
        loaded
            .materialize_composition_metadata_for_compilation()
            .unwrap();
        assert_eq!(
            loaded.composition_parser_characterizations_by_terminal,
            constraint.composition_parser_characterizations_by_terminal,
        );
        assert_eq!(loaded.start().mask(), constraint.start().mask());
    }

    #[test]
    fn current_constraint_artifact_preserves_composition_grammar_summary() {
        let constraint = tiny_constraint();
        let expected = constraint
            .composition_grammar_summary
            .clone()
            .expect("fresh static constraint should retain composition grammar summary");
        let mut loaded = Constraint::load(&constraint.save()).unwrap();
        assert!(loaded.composition_grammar_summary.is_none());
        assert!(loaded.deferred_composition_metadata_blob.is_some());
        assert_eq!(loaded.start().mask(), constraint.start().mask());
        loaded
            .materialize_composition_metadata_for_compilation()
            .unwrap();
        assert_eq!(loaded.composition_grammar_summary, Some(expected));
        assert_eq!(loaded.start().mask(), constraint.start().mask());
    }

    #[test]
    fn split_composition_metadata_allows_link_only_materialization() {
        let mut constraint = tiny_constraint();
        constraint.ensure_composition_reset_tokens_by_terminal();
        let expected_resets = constraint.composition_reset_tokens_by_terminal.clone();
        let expected_summary = constraint.composition_grammar_summary.clone();
        let expected_templates = constraint.composition_parser_templates_by_terminal.clone();
        assert!(expected_summary.is_some());
        assert!(expected_templates.iter().any(Option::is_some));

        let mut loaded = Constraint::load(&constraint.save()).unwrap();
        let deferred = loaded
            .deferred_composition_metadata_blob
            .as_ref()
            .expect("current artifact should defer composition metadata");
        assert!(deferred.as_slice().starts_with(&COMPOSITION_METADATA_SPLIT_MAGIC));
        assert!(!loaded.composition_link_metadata_materialized);
        assert!(loaded.composition_parser_templates_by_terminal.is_empty());

        loaded
            .materialize_composition_link_metadata_for_compilation()
            .unwrap();
        assert_eq!(loaded.composition_reset_tokens_by_terminal, expected_resets);
        assert_eq!(loaded.composition_grammar_summary, expected_summary);
        assert!(loaded.composition_link_metadata_materialized);
        assert!(
            loaded.composition_parser_templates_by_terminal.is_empty(),
            "link-only materialization must not instantiate static parser caches",
        );
        assert!(
            loaded.deferred_composition_metadata_blob.is_some(),
            "full compiler metadata must remain available for a later static composition",
        );

        loaded
            .materialize_composition_metadata_for_compilation()
            .unwrap();
        assert_eq!(
            loaded.composition_parser_templates_by_terminal,
            expected_templates,
        );
        assert!(loaded.deferred_composition_metadata_blob.is_none());
    }

    #[test]
    fn current_constraint_artifact_preserves_global_ignore_descriptor() {
        let constraint = ignored_constraint();
        let loaded = Constraint::load(&constraint.save()).unwrap();
        assert!(constraint.ignore_expr.is_some());
        assert_eq!(loaded.ignore_expr, constraint.ignore_expr);
        assert!(
            loaded.tokenizer.terminal_exprs().is_none(),
            "current load should defer terminal expression reconstruction",
        );
        assert_eq!(
            loaded.retained_terminal_exprs(),
            constraint.tokenizer.terminal_exprs(),
            "current artifacts should lazily retain terminal proof expressions",
        );
        if constraint.can_defer_internal_tsid_inverse() {
            assert!(
                loaded.internal_tsid_to_states.is_empty(),
                "current scalar-state artifacts should defer the redundant TSID inverse",
            );
            assert_eq!(
                loaded.internal_tsid_groups(),
                constraint.internal_tsid_to_states.as_slice(),
                "deferred TSID inverse must reconstruct exactly",
            );
        }
        assert_eq!(loaded.start().mask(), constraint.start().mask());
    }

    #[test]
    fn constraint_envelope_rejects_version_and_length_mismatches() {
        let constraint = tiny_constraint();
        let mut wrong_version = constraint.save();
        wrong_version[8..10].copy_from_slice(&(CONSTRAINT_VERSION + 1).to_le_bytes());
        assert!(Constraint::load(&wrong_version)
            .unwrap_err()
            .to_string()
            .contains("version"));

        let mut wrong_length = constraint.save();
        wrong_length[10..18].copy_from_slice(&0u64.to_le_bytes());
        assert!(Constraint::load(&wrong_length)
            .unwrap_err()
            .to_string()
            .contains("payload length"));
    }

    #[test]
    fn constraint_roundtrip_preserves_commit_template_dfas() {
        let mut constraint = tiny_constraint();
        let mut pop = UnweightedDfa::new();
        let accepted = pop.add_state();
        pop.add_transition(pop.start_state, 7, accepted);
        pop.set_accepting(accepted, true);
        let template = CommitTemplateDfas {
            pop,
            read: UnweightedDfa::default(),
            push: UnweightedDfa::default(),
            pop_to_read: vec![None; 2],
            pop_to_push: vec![None; 2],
            read_to_push: Vec::new(),
        };
        constraint.template_dfas_by_terminal = vec![None, Some(Arc::new(template.clone()))];

        let loaded = Constraint::load(&constraint.save()).expect("template artifact should load");
        let loaded_template = loaded.template_dfas_by_terminal[1]
            .as_deref()
            .expect("serialized template should survive load");
        let loaded_fast_template = loaded.fast_template_dfas_by_terminal[1]
            .as_deref()
            .expect("runtime template transition cache should be rebuilt after load");
        assert_eq!(loaded_template.pop, template.pop);
        assert_eq!(loaded_template.read, template.read);
        assert_eq!(loaded_template.push, template.push);
        assert_eq!(loaded_template.pop_to_read, template.pop_to_read);
        assert_eq!(loaded_template.pop_to_push, template.pop_to_push);
        assert_eq!(loaded_template.read_to_push, template.read_to_push);
        assert_eq!(loaded_fast_template.pop.start_state, template.pop.start_state);
        assert_eq!(
            loaded_fast_template.pop.states[accepted as usize].is_accepting,
            template.pop.states[accepted as usize].is_accepting
        );
        assert_eq!(
            loaded_fast_template.pop.states[template.pop.start_state as usize]
                .transitions
                .get(7),
            Some(accepted)
        );
    }



#[test]
fn loaded_partitioned_bounded_strings_keep_first_key_prefix_live() {
    use crate::automata::lexer::Lexer;
    let vocab = Vocab::new(
        [b"{\"".as_slice(), b"{", b"\"", b"a", b"b", b"\": \"", b"\", \"", b"\"}", b"x", b" ", b"xx"]
            .into_iter().enumerate().map(|(id, bytes)| (id as u32, bytes.to_vec())).collect(),
    );
    let schema = r#"{
            "type":"object",
            "properties":{
                "a":{"type":"string","minLength":1,"maxLength":200,
                     "pattern":"^(?:\\S+\\s+){0,19}\\S+$"},
                "b":{"type":"string","minLength":1,"maxLength":100}
            },
            "required":["a","b"],"additionalProperties":false
        }"#;
    let original = Constraint::compile(crate::Grammar::json_schema(schema), &vocab).unwrap();
    let loaded = Constraint::load(original.save()).unwrap();
    assert!(original.tokenizer.has_compressed_transition_segments());
    assert!(loaded.tokenizer.has_compressed_transition_segments());
    assert_eq!(original.tokenizer.num_states(), loaded.tokenizer.num_states());
    for state in 0..original.tokenizer.num_states().min(20) {
        assert_eq!(original.tokenizer.step(state, b'"'), loaded.tokenizer.step(state, b'"'),
            "opening-quote transition differs at tokenizer state {state}");
    }
    assert_eq!(original.start().mask(), loaded.start().mask());
    let mut expected = original.start();
    let mut actual = loaded.start();
    expected.commit_token(0).expect("compiled key prefix must commit");
    actual.commit_token(0).expect("loaded key prefix admitted by the mask must commit");
    assert_eq!(expected.mask(), actual.mask());
    // Complete {"a": "x", "b": "x"}, checking every next-token mask.
    for token in [3, 5, 8, 6, 4, 5, 8, 7] {
        expected.commit_token(token).unwrap();
        actual.commit_token(token).unwrap();
        assert_eq!(expected.mask(), actual.mask());
    }
    assert!(expected.is_accepting());
    assert!(actual.is_accepting());
}
