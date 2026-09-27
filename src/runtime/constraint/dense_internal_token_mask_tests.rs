    use super::*;

    #[test]
    fn early_observation_filter_matches_historical_rows_across_future_widths() {
        let vocab = Vocab::new((0u32..128).map(|id| (id, vec![id as u8])).collect::<Vec<_>>());
        let mut saw_mixed = false;
        let mut saw_over_limit = false;
        for width in [1usize, 2, 3, 8, 9, 12] {
            let mut grammar = String::from("start start;\n");
            for terminal in 0..width {
                let suffix = char::from(b'a' + terminal as u8);
                grammar.push_str(&format!("t T{terminal} ::= /[\\x00-\\x7F]+{suffix}/;\n"));
            }
            grammar.push_str("nt start ::= ");
            grammar.push_str(&(0..width).map(|t| format!("T{t} T{t}"))
                .collect::<Vec<_>>().join(" | "));
            grammar.push_str(";\n");
            let built = crate::DynamicConstraint::from_glrm_grammar(&grammar, &vocab)
                .expect("finite physical selector fixture compiles");
            let constraint = &built.inner;
            assert!(!constraint.tokenizer.has_any_virtual_runtime());
            for state in 0..constraint.tokenizer.num_states() {
                let count = constraint.tokenizer.possible_future_terminals_iter(state).count();
                saw_mixed |= (2..=8).contains(&count);
                saw_over_limit |= count > 8;
            }
            assert_eq!(
                constraint.build_dynamic_terminal_observation_classes_filtered::<true>(),
                constraint.build_dynamic_terminal_observation_classes_filtered::<false>(),
                "future width {width}",
            );
        }
        assert!(saw_mixed, "fixtures must exercise eligible mixed signatures");
        assert!(saw_over_limit, "fixtures must exercise the nine-entry rejection bound");
    }

    #[test]
    fn bind_vocab_exact_rebinds_equal_vocab_and_rejects_mismatch() {
        let vocab_a = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);
        let vocab_b = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);
        let vocab_bad = Vocab::new(vec![(0, b"a".to_vec()), (1, b"c".to_vec())]);
        let mut constraint = Constraint::from_glrm_grammar(
            "start start;\nt A ::= \"a\";\nnt start ::= A;\n",
            &vocab_a,
        )
        .unwrap();

        assert!(!Arc::ptr_eq(&constraint.token_bytes, &vocab_b.entries_arc()));
        constraint.bind_vocab_exact(&vocab_b).unwrap();
        assert!(Arc::ptr_eq(&constraint.token_bytes, &vocab_b.entries_arc()));
        assert!(constraint.late_bind_vocab.get().is_some());
        assert!(Arc::ptr_eq(
            &constraint
                .late_bind_vocab
                .get()
                .expect("exact vocab bind should seed late-bind vocab")
                .entries_arc(),
            &vocab_b.entries_arc(),
        ));

        let bound = Arc::clone(&constraint.token_bytes);
        assert!(constraint.bind_vocab_exact(&vocab_bad).is_err());
        assert!(Arc::ptr_eq(&constraint.token_bytes, &bound));
    }

    #[test]
    fn loaded_constraint_binds_prepared_equal_vocab_and_rejects_mismatch() {
        let vocab_a = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);
        let vocab_b = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);
        let vocab_bad = Vocab::new(vec![(0, b"a".to_vec()), (1, b"c".to_vec())]);
        let constraint = Constraint::from_glrm_grammar(
            "start start;\nt A ::= \"a\";\nnt start ::= A;\n",
            &vocab_a,
        )
        .unwrap();
        let saved = constraint.save();

        // Prepare exactly the pure Vocab artifact used by the packed-wire fast
        // path, then bind a current-format loaded constraint. This exercises
        // the path used by serving/CFA where vocabulary preparation is shared
        // across many independently loaded constraints.
        let _ = crate::compiler::compile::vocab_packed_token_bytes(&vocab_b);
        let mut loaded = Constraint::load(saved.clone()).unwrap();
        assert!(loaded.packed_token_bytes.is_some());
        loaded.bind_vocab_exact(&vocab_b).unwrap();
        assert!(Arc::ptr_eq(&loaded.token_bytes, &vocab_b.entries_arc()));

        let _ = crate::compiler::compile::vocab_packed_token_bytes(&vocab_bad);
        let mut mismatched = Constraint::load(saved).unwrap();
        assert!(mismatched.packed_token_bytes.is_some());
        assert!(mismatched.bind_vocab_exact(&vocab_bad).is_err());
    }
    use crate::Vocab;

    #[test]
    fn empty_wide_frontier_lookup_does_not_materialize_deferred_dynamic_vocab() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"ab".to_vec()),
        ]);
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start start;
                t A ::= "a";
                t B ::= "b";
                nt start ::= A B;
            "#,
            &vocab,
        )
        .unwrap();
        let loaded = Constraint::load(&constraint.save()).unwrap();
        assert!(loaded.direct_regular_wide_frontier_acceptance.is_empty());
        assert!(
            loaded.lazy_dynamic_mask_vocab.get().is_none(),
            "load should preserve deferred dynamic-vocab materialization",
        );

        let initial = loaded.initial_state_map();
        for gss in initial.values() {
            assert!(loaded.direct_regular_wide_frontier_for_gss(gss).is_none());
        }

        assert!(
            loaded.lazy_dynamic_mask_vocab.get().is_none(),
            "empty wide-frontier lookup must not trigger deferred dynamic-vocab materialization",
        );
    }

    #[test]
    fn nonempty_wide_frontier_lookup_does_not_materialize_vocab_for_cache_only() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"ab".to_vec()),
        ]);
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start start;
                t A ::= "a";
                t B ::= "b";
                nt start ::= A B;
            "#,
            &vocab,
        )
        .unwrap();
        let mut loaded = Constraint::load(&constraint.save()).unwrap();
        assert!(loaded.lazy_dynamic_mask_vocab.get().is_none());

        let initial = loaded.initial_state_map();
        let gss = initial.values().next().unwrap().clone();
        let mut frontier_states = gss.peek_values();
        frontier_states.sort_unstable();
        loaded.direct_regular_wide_frontier_acceptance = vec![
            crate::runtime::artifact::DirectRegularWideFrontierAcceptance {
                action_origins: Vec::new(),
                state_count: frontier_states.len(),
                actionable_terminals: crate::ds::bitset::BitSet::new(
                    loaded.table.num_terminals as usize,
                ),
                frontier_states: Arc::<[u32]>::from(frontier_states.as_slice()),
                // Keep this deliberately unrelated to `gss` so the cheap
                // lower-id lookup misses and the fallback frontier-state scan
                // is exercised.
                empty_acc_frontier: ParserGSS::empty(),
                acceptance_parts: Arc::from([]),
                dense_by_tsid: Arc::new(crate::runtime::artifact::DenseAcceptanceRows::default()),
                advance_by_terminal: Arc::from([]),
            },
        ];

        assert!(loaded.direct_regular_wide_frontier_for_gss(&gss).is_some());
        assert!(
            loaded.lazy_dynamic_mask_vocab.get().is_none(),
            "wide-frontier memoization must not materialize the dynamic vocab",
        );
    }

    #[test]
    fn initial_commit_prime_token_ids_accepts_exact_limit() {
        let mask = [u32::MAX >> (32 - INITIAL_COMMIT_PRIME_MAX_TOKENS)];
        assert_eq!(
            initial_commit_prime_token_ids(&mask),
            Some((0..INITIAL_COMMIT_PRIME_MAX_TOKENS as u32).collect()),
        );
    }

    #[test]
    fn initial_commit_prime_token_ids_rejects_above_limit() {
        let mask = [u32::MAX >> (32 - (INITIAL_COMMIT_PRIME_MAX_TOKENS + 1))];
        assert_eq!(initial_commit_prime_token_ids(&mask), None);
    }

    #[test]
    fn dense_internal_token_masks_match_reference_expansion() {
        let internal_tokens = RangeSetBlaze::from_iter([
            0u32..=0,
            3..=7,
            62..=65,
            127..=130,
            190..=192,
            300..=302,
        ]);
        let actual = Constraint::dense_words_from_internal_set_with_words(&internal_tokens, 5);
        let mut expected = vec![0u64; 5];
        for token in internal_tokens.iter() {
            let word = token as usize / 64;
            let bit = token as usize % 64;
            if let Some(slot) = expected.get_mut(word) {
                *slot |= 1u64 << bit;
            }
        }
        assert_eq!(actual.as_ref(), expected.as_slice());
    }

    #[test]
    fn dense_internal_token_masks_ignore_out_of_bounds_ranges() {
        let internal_tokens = RangeSetBlaze::from_iter([63u32..=65, 190..=400]);
        let actual = Constraint::dense_words_from_internal_set_with_words(&internal_tokens, 3);
        assert_eq!(actual.as_ref(), &[1u64 << 63, 0b11, 1u64 << 62 | 1u64 << 63]);
    }

    #[test]
    fn direct_sparse_expanded_work_counts_alias_expansion_and_heavy_tokens() {
        let masks = vec![
            vec![(0, 1)],
            vec![(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)],
            vec![(0, 1), (1, 1)],
            vec![(0, 1), (1, 1), (2, 1)],
        ];
        let work_prefix = Constraint::direct_sparse_work_prefix(&masks, 16);
        let selected = RangeSetBlaze::from_iter([0u32..=1, 3..=3]);

        // Each selected internal token costs one membership scan. Token 1 is
        // heavy because 5 entries exceed 16 / 4, so runtime uses a 16-word
        // dense OR for it instead of five sparse writes.
        assert_eq!(
            Constraint::direct_sparse_expanded_work(&selected, &work_prefix),
            (1 + 1) + (1 + 16) + (1 + 3)
        );
    }

    #[test]
    fn owned_load_preserves_heavy_token_mask_classification_for_backed_buf_masks() {
        let vocab = Vocab::new(
            (0..200u32)
                .map(|token| (token, b"a".to_vec()))
                .collect(),
        );
        let constraint = Constraint::from_glrm_grammar(
            "start start;\nt A ::= \"a\";\nnt start ::= A;\n",
            &vocab,
        )
        .unwrap();
        assert!(
            !constraint.heavy_token_indices.is_empty(),
            "duplicate-token expansion should produce at least one heavy internal token",
        );

        let loaded = Constraint::load(constraint.save()).unwrap();
        let backed = loaded
            .backed_internal_token_buf_flat
            .as_ref()
            .expect("owned current load should retain IBM3 entries in artifact backing");
        assert!(
            backed.slice(0, backed.len()).is_some(),
            "fresh current-format IBM3 entries should be naturally aligned for native access",
        );
        assert_eq!(loaded.heavy_token_indices, constraint.heavy_token_indices);
        assert_eq!(
            loaded.internal_token_buf_op_costs,
            constraint.internal_token_buf_op_costs,
        );
        assert_eq!(
            loaded.word_group_buf_op_costs,
            constraint.word_group_buf_op_costs,
        );
        assert_eq!(loaded.total_internal_buf_cost, constraint.total_internal_buf_cost);
    }

    #[test]
    fn direct_sparse_expanded_work_clamps_out_of_bounds_ranges() {
        let masks = vec![vec![(0, 1)], vec![(1, 1), (2, 1)]];
        let work_prefix = Constraint::direct_sparse_work_prefix(&masks, 16);
        let selected = RangeSetBlaze::from_iter([1u32..=100]);

        assert_eq!(
            Constraint::direct_sparse_expanded_work(&selected, &work_prefix),
            1 + 2
        );
    }

    #[test]
    fn heavy_group_dense_masks_match_sparse_replay_and_threshold() {
        let groups = vec![
            vec![(0, 0b0011), (2, 0b0100), (5, 0b1000)],
            vec![(1, 0b0101), (6, 0b1010)],
        ];
        let dense = Constraint::compute_heavy_group_dense_masks(&groups, 8);

        assert!(dense[0].is_some(), "3 sparse entries should beat the 8 / 4 threshold");
        assert!(dense[1].is_none(), "2 sparse entries should remain sparse at the threshold");

        let mut sparse_or = vec![0x10u32; 8];
        or_sparse_buf_entries(&mut sparse_or, &groups[0]);
        let mut adaptive_or = vec![0x10u32; 8];
        assert_eq!(
            or_group_buf_mask(&mut adaptive_or, &groups[0], dense[0].as_deref()),
            8,
        );
        assert_eq!(adaptive_or, sparse_or);

        let mut sparse_andnot = vec![u32::MAX; 8];
        andnot_sparse_buf_entries(&mut sparse_andnot, &groups[0]);
        let mut adaptive_andnot = vec![u32::MAX; 8];
        assert_eq!(
            andnot_group_buf_mask(
                &mut adaptive_andnot,
                &groups[0],
                dense[0].as_deref(),
            ),
            8,
        );
        assert_eq!(adaptive_andnot, sparse_andnot);

        let mut light = vec![0u32; 8];
        assert_eq!(
            or_group_buf_mask(&mut light, &groups[1], dense[1].as_deref()),
            groups[1].len(),
        );
    }

    #[test]
    fn dense_or_matches_set_union_exhaustively_without_final_mapping() {
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"b".to_vec()),
                (2, b"ab".to_vec()),
                (3, b"ba".to_vec()),
            ]);
        let mut constraint = Constraint::from_glrm_grammar(
            r#"
                start start;
                t A ::= "a" | "ab";
                t B ::= "b" | "ab";
                nt item ::= A | B;
                nt start ::= item item? item?;
            "#,
            &vocab,
        )
        .unwrap();
        constraint.final_mask_mapping = Default::default();
        let n_internal = constraint.internal_token_to_tokens.len();
        assert!(n_internal <= 16, "small exhaustive test requires a tiny internal vocab");

        for selected in 0u64..(1u64 << n_internal) {
            let mut dense = vec![0u64; n_internal.div_ceil(64)];
            let mut selected_image = 0u32;
            for (internal, originals) in constraint.internal_token_to_tokens.iter().enumerate() {
                if selected & (1u64 << internal) == 0 {
                    continue;
                }
                dense[internal / 64] |= 1u64 << (internal % 64);
                for &original in originals {
                    selected_image |= 1u32 << original;
                }
            }

            for initial in 0u32..=0x0f {
                let expected = initial | selected_image;
                let buf_zeroed = initial == 0;

                let mut profiled = vec![0u32; constraint.body_mask_len()];
                profiled[0] = initial;
                constraint.or_internal_dense_to_buf(&dense, &mut profiled, buf_zeroed);
                assert_eq!(
                    profiled[0],
                    expected,
                    "profiled OR mismatch: selected={selected:#b} initial={initial:#06b} internal_to_original={:?}",
                    constraint.internal_token_to_tokens,
                );

                let mut fast = vec![0u32; constraint.body_mask_len()];
                fast[0] = initial;
                constraint.or_internal_dense_to_buf_fast(&dense, &mut fast, buf_zeroed);
                assert_eq!(
                    fast[0],
                    expected,
                    "fast OR mismatch: selected={selected:#b} initial={initial:#06b} internal_to_original={:?}",
                    constraint.internal_token_to_tokens,
                );

                let mut scratch_fast = vec![0u32; constraint.body_mask_len()];
                scratch_fast[0] = initial;
                let mut dirty_complement_scratch = Vec::new();
                constraint.or_internal_dense_to_buf_fast_with_scratch(
                    &dense,
                    &mut scratch_fast,
                    buf_zeroed,
                    &mut dirty_complement_scratch,
                );
                assert_eq!(
                    scratch_fast[0],
                    expected,
                    "scratch fast OR mismatch: selected={selected:#b} initial={initial:#06b} internal_to_original={:?}",
                    constraint.internal_token_to_tokens,
                );
            }
        }
    }

    #[test]
    fn dirty_dense_conversion_uses_scratch_complement_when_alias_replay_is_expensive() {
        let mut entries = Vec::new();
        for alias in 0u32..64 {
            for byte_index in 0u32..64 {
                entries.push((alias * 64 + byte_index, vec![(byte_index + 32) as u8]));
            }
        }
        let vocab = Vocab::new(entries);
        let mut grammar = String::from("start start;\n");
        for byte_index in 0u32..64 {
            let literal = serde_json::to_string(&((byte_index + 32) as u8 as char).to_string())
                .unwrap();
            grammar.push_str(&format!("t T{byte_index} ::= {literal};\n"));
        }
        grammar.push_str("nt start ::= ");
        for byte_index in 0u32..64 {
            if byte_index != 0 {
                grammar.push_str(" | ");
            }
            grammar.push_str(&format!("T{byte_index} T{byte_index}"));
        }
        grammar.push_str(";\n");
        let mut constraint = Constraint::from_glrm_grammar(
            &grammar,
            &vocab,
        )
        .unwrap();
        constraint.final_mask_mapping = Default::default();

        let n_internal = constraint.internal_token_to_tokens.len();
        assert_eq!(n_internal, 64, "expected duplicate bytes to form 64 internal tokens");

        let mut dense = vec![0u64; n_internal.div_ceil(64)];
        for internal in 0..48 {
            dense[internal / 64] |= 1u64 << (internal % 64);
        }

        let dirty_token = constraint.internal_token_to_tokens[63][0];
        let dirty_word = dirty_token as usize / 32;
        let dirty_bit = dirty_token as usize % 32;

        let mut expected = vec![0u32; constraint.body_mask_len()];
        expected[dirty_word] |= 1u32 << dirty_bit;
        constraint.or_internal_dense_to_buf_fast(&dense, &mut expected, false);

        let mut actual = vec![0u32; constraint.body_mask_len()];
        actual[dirty_word] |= 1u32 << dirty_bit;
        let mut scratch = Vec::new();
        constraint.or_internal_dense_to_buf_fast_with_scratch(
            &dense,
            &mut actual,
            false,
            &mut scratch,
        );

        assert_eq!(actual, expected);
        assert_eq!(
            scratch.len(),
            constraint.body_mask_len(),
            "expected the replay-cost model to select dirty scratch complement conversion",
        );
    }
