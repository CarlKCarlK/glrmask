use super::*;
use crate::automata::lexer::tokenizer::arbitrary_epsilon_l1_test_tokenizer;

#[test]
fn large_p2_projection_prefers_exact_profile_reuse() {
    assert!(prefer_exact_profiles_over_projected_l1("p2", 82_270, 3_780));
    assert!(!prefer_exact_profiles_over_projected_l1("p2", 49_999, 3_780));
    assert!(!prefer_exact_profiles_over_projected_l1("p2", 82_270, 1_023));
    assert!(!prefer_exact_profiles_over_projected_l1("p1", 82_270, 3_780));
}

#[test]
fn locally_derived_subset_order_matches_standalone_order() {
    let parent = Vocab::new(vec![
        (9, b"z".to_vec()),
        (2, b" alpha".to_vec()),
        (7, b"".to_vec()),
        (4, b"apple".to_vec()),
        (1, b" beta".to_vec()),
        (12, vec![0xff, b'x']),
    ]);
    let subset_entries = vec![
        (12, vec![0xff, b'x']),
        (7, b"".to_vec()),
        (2, b" alpha".to_vec()),
        (4, b"apple".to_vec()),
    ];
    let subset = Vocab::new(subset_entries.clone());
    let parent_order = l1_identity_vocab_order(&parent);
    let derived = derive_l1_identity_vocab_order_from_parent(&parent_order, &subset);
    let standalone = l1_identity_vocab_order(&Vocab::new(subset_entries));

    assert_eq!(derived.token_ids_sorted, standalone.token_ids_sorted);
    assert_eq!(derived.original_to_internal, standalone.original_to_internal);
    assert_eq!(
        derived
            .token_entries_sorted
            .iter()
            .map(|(id, bytes)| (*id, bytes.as_ref()))
            .collect::<Vec<_>>(),
        standalone
            .token_entries_sorted
            .iter()
            .map(|(id, bytes)| (*id, bytes.as_ref()))
            .collect::<Vec<_>>(),
    );
    assert_eq!(
        derived.token_buckets.empty_token_indices,
        standalone.token_buckets.empty_token_indices,
    );
    assert_eq!(
        derived.token_buckets.token_indices_by_first_byte,
        standalone.token_buckets.token_indices_by_first_byte,
    );
    assert_eq!(
        derived.token_buckets.suffix_lcps_by_first_byte,
        standalone.token_buckets.suffix_lcps_by_first_byte,
    );
    assert_eq!(
        derived.token_buckets.suffix_subtree_bytes,
        standalone.token_buckets.suffix_subtree_bytes,
    );
    assert_eq!(
        derived.token_buckets.suffix_first_bytes_by_bucket,
        standalone.token_buckets.suffix_first_bytes_by_bucket,
    );
    assert_eq!(
        derived.token_buckets.has_empty_suffix_by_bucket,
        standalone.token_buckets.has_empty_suffix_by_bucket,
    );
}

#[test]
fn small_vocab_powerset_probe_selects_supported_structural_domains() {
    assert!(l1_generic_nfa_small_vocab_powerset_probe_enabled(
        60_000, 128, 240,
    ));
    assert!(l1_generic_nfa_small_vocab_powerset_probe_enabled(
        97_046, 4, 240,
    ));
    assert!(l1_generic_nfa_small_vocab_powerset_probe_enabled(
        41_451, 630, 46,
    ));
    assert!(l1_generic_nfa_small_vocab_powerset_probe_enabled(
        97_046, 630, 189,
    ));
    assert!(!l1_generic_nfa_small_vocab_powerset_probe_enabled(
        44_597, 630, 193,
    ));
    assert!(!l1_generic_nfa_small_vocab_powerset_probe_enabled(
        97_046, 630, 225,
    ));
    assert!(!l1_generic_nfa_small_vocab_powerset_probe_enabled(
        59_999, 128, 240,
    ));
    assert!(!l1_generic_nfa_small_vocab_powerset_probe_enabled(
        97_046, 129, 240,
    ));
    assert!(!l1_generic_nfa_small_vocab_powerset_probe_enabled(
        39_999, 630, 46,
    ));
    assert!(!l1_generic_nfa_small_vocab_powerset_probe_enabled(
        60_001, 630, 46,
    ));
}

#[test]
fn large_vocab_prefers_bounded_relevant_powerset_probe_before_token_bounded_fallback() {
    let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
    let raw_states = (0..tokenizer.num_states() as usize).collect::<Vec<_>>();
    let token = Arc::<[u8]>::from(b"a".as_slice());
    let token_entries = (0..=L1_GENERIC_NFA_TOKEN_BOUNDED_MAX_VOCAB as u32)
        .map(|token_id| (token_id, Arc::clone(&token)))
        .collect::<Vec<_>>();
    let active = [true, true];

    assert!(!l1_generic_nfa_token_bounded_view_enabled(
        raw_states.len(),
        token_entries.len(),
    ));
    let flat_trans = build_flat_transition_table(&tokenizer);
    let (_, _, analysis_view) = build_l1_generic_nfa_analysis_view(
        &tokenizer,
        &raw_states,
        &token_entries,
        &active,
        &flat_trans,
        None,
        None,
        super::super::l2p::equivalence_analysis::state_equivalence::nfa::TokenBoundedAnalysisWorkBudget {
            max_configurations: usize::MAX,
            max_trie_visits: 0,
        },
    );
    assert_eq!(analysis_view, "relevant_powerset_probe");
}

#[test]
fn token_bounded_large_work_budget_keeps_small_completion_headroom() {
    let default = l1_generic_nfa_token_bounded_large_work_budget_with_override(None);
    assert_eq!(default.max_configurations, 64_000);
    assert_eq!(default.max_trie_visits, 1_050_000);

    let overridden =
        l1_generic_nfa_token_bounded_large_work_budget_with_override(Some(1_234_567));
    assert_eq!(overridden.max_configurations, 64_000);
    assert_eq!(overridden.max_trie_visits, 1_234_567);
}

#[test]
fn token_bounded_view_respects_construction_budget() {
    // The depth-1 workload that motivated the bounded topology remains in
    // the fast regime, while larger raw-state/vocab products fall back to
    // the exact relevant-powerset proof instead of eagerly expanding a
    // prohibitively large token topology.
    assert!(l1_generic_nfa_token_bounded_view_enabled(18_943, 15_264));
    assert!(!l1_generic_nfa_token_bounded_view_enabled(26_965, 15_264));
    assert!(!l1_generic_nfa_token_bounded_view_enabled(26_965, 82_270));
    assert!(!l1_generic_nfa_token_bounded_view_enabled(3_343, 82_270));
    assert!(!l1_generic_nfa_token_bounded_view_enabled(3_343, 21_310));
}

fn build_scalar_generic_nfa_terminal_dwa(
    tokenizer: &Tokenizer,
    vocab_order: &L1IdentityVocabOrder,
    id_map: &mut InternalIdMap,
    num_terminals: u32,
    active_terminals: &[bool],
) -> DWA {
    let mut deferred_by_terminal = (0..num_terminals)
        .map(|_| Vec::<(u32, Arc<RangeSetBlaze<u32>>)>::new())
        .collect::<Vec<_>>();

    for (internal_tsid, raw_state) in id_map
        .tokenizer_states
        .iter_representative_ids()
        .enumerate()
    {
        let mut token_ids_by_terminal = FxHashMap::<u32, Vec<u32>>::default();
        for (internal_token_id, (_, bytes)) in
            vocab_order.token_entries_sorted.iter().enumerate()
        {
            let end_states = tokenizer.execute_from_state_end_only(bytes, raw_state);
            let mut active_signature = Vec::<u32>::new();
            for &state in &end_states {
                active_signature.extend(collect_active_terminal_signature(
                    tokenizer,
                    state,
                    active_terminals,
                ));
            }
            active_signature.sort_unstable();
            active_signature.dedup();
            for terminal in active_signature {
                token_ids_by_terminal
                    .entry(terminal)
                    .or_default()
                    .push(internal_token_id as u32);
            }
        }

        for (terminal, token_ids) in token_ids_by_terminal {
            let token_set = shared_rangeset(token_ids.into_iter().collect());
            if !token_set.is_empty() {
                deferred_by_terminal[terminal as usize]
                    .push((internal_tsid as u32, token_set));
            }
        }
    }

    merge_deferred_equivalent_tsids(id_map, &mut deferred_by_terminal);
    let mut dwa = DWA::new(id_map.num_tsids(), id_map.max_internal_token_id());
    let end_state = dwa.add_state();
    dwa.set_final_weight(end_state, Weight::all());
    for (terminal, entries) in deferred_by_terminal.into_iter().enumerate() {
        let weight = Weight::from_per_tsid_shared(entries);
        if !weight.is_empty() {
            dwa.add_transition(dwa.start_state(), terminal as i32, end_state, weight);
        }
    }
    dwa
}

#[test]
fn generic_epsilon_l1_powerset_trie_matches_scalar_reference() {
    let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
    let vocab = Vocab::new(
        vec![
            (0, b"".to_vec()),
            (1, b"a".to_vec()),
            (2, b"aa".to_vec()),
            (3, b"aaa".to_vec()),
            (4, b"ab".to_vec()),
            (5, b"b".to_vec()),
            (6, b"ba".to_vec()),
            (7, b"bb".to_vec()),
            (8, b"x".to_vec()),
        ]);
    let active = [true, true];
    let (optimized_id_map, order, _, _, _) =
        build_l1_generic_nfa_fallback_id_map(&tokenizer, &vocab, None);
    let mut optimized_id_map = optimized_id_map;
    let mut scalar_id_map = optimized_id_map.clone();
    let (optimized, _) = build_l1_generic_nfa_terminal_dwa(
        &tokenizer,
        order.as_ref(),
        &mut optimized_id_map,
        2,
        &active,
    )
    .expect("optimized generic epsilon L1 DWA");
    let scalar = build_scalar_generic_nfa_terminal_dwa(
        &tokenizer,
        order.as_ref(),
        &mut scalar_id_map,
        2,
        &active,
    );

    for raw_state in 0..tokenizer.num_states() {
        let optimized_tsid =
            optimized_id_map.tokenizer_states.original_to_internal[raw_state as usize];
        let scalar_tsid = scalar_id_map.tokenizer_states.original_to_internal[raw_state as usize];
        for (&token_id, bytes) in vocab.entries_map().iter() {
            let optimized_token =
                optimized_id_map.internal_token_for_original(token_id).expect("optimized token");
            let scalar_token = scalar_id_map.internal_token_for_original(token_id).expect("scalar token");
            for terminal in 0..2u32 {
                let optimized_accepts = optimized
                    .eval_word(&[terminal as i32])
                    .tokens_for_tsid(optimized_tsid)
                    .contains(optimized_token);
                let scalar_accepts = scalar
                    .eval_word(&[terminal as i32])
                    .tokens_for_tsid(scalar_tsid)
                    .contains(scalar_token);
                assert_eq!(
                    optimized_accepts, scalar_accepts,
                    "raw_state={raw_state} token={token_id} bytes={bytes:?} terminal={terminal}",
                );
            }
        }
    }
}

#[test]
fn generic_epsilon_l1_shared_superset_topology_matches_subset_topology() {
    let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
    let full_vocab = Vocab::new(
        vec![
            (0, b"".to_vec()),
            (1, b"a".to_vec()),
            (2, b"aa".to_vec()),
            (3, b"ab".to_vec()),
            (4, b"b".to_vec()),
            (5, b"ba".to_vec()),
            (6, b"bb".to_vec()),
            (7, b"x".to_vec()),
        ]);
    let subset_vocab = Vocab::new(
        vec![
            (0, b"".to_vec()),
            (1, b"a".to_vec()),
            (2, b"aa".to_vec()),
            (3, b"ab".to_vec()),
            (4, b"b".to_vec()),
            (5, b"ba".to_vec()),
            (6, b"bb".to_vec()),
        ]);
    let active = [true, true];
    let raw_states = (0..tokenizer.num_states() as usize).collect::<Vec<_>>();
    let full_tokens = full_vocab
        .entries_map()
        .values()
        .map(|bytes| bytes.as_slice())
        .collect::<Vec<_>>();
    let topology = crate::compiler::stages::id_map_and_terminal_dwa::l2p::equivalence_analysis::state_equivalence::nfa::build_token_bounded_analysis_topology(
        &tokenizer,
        &raw_states,
        &full_tokens,
    );
    let flat_trans = build_flat_transition_table(&tokenizer);

    let (mut shared_map, shared_order, _, _, shared_reuse) =
        build_l1_generic_nfa_exact_id_map(
            &tokenizer,
            &subset_vocab,
            &active,
            &flat_trans,
            None,
            Some(&topology),
            None,
            None,
        );
    let (mut standalone_map, standalone_order, _, _, standalone_reuse) =
        build_l1_generic_nfa_exact_id_map(
            &tokenizer,
            &subset_vocab,
            &active,
            &flat_trans,
            None,
            None,
            None,
            None,
        );

    let shared_classes = &shared_map.tokenizer_states.original_to_internal;
    let standalone_classes = &standalone_map.tokenizer_states.original_to_internal;
    for left in 0..shared_classes.len() {
        for right in 0..shared_classes.len() {
            assert_eq!(
                shared_classes[left] == shared_classes[right],
                standalone_classes[left] == standalone_classes[right],
                "shared superset topology changed subset L1 partition for {left} <> {right}",
            );
        }
    }

    let flat_trans = build_flat_transition_table(&tokenizer);
    let (shared_dwa, _) = build_l1_terminal_dwa(
        &tokenizer,
        shared_order.as_ref(),
        &mut shared_map,
        2,
        &active,
        flat_trans.as_ref(),
        shared_reuse.as_ref(),
    )
    .expect("shared-topology exact L1 DWA");
    let (standalone_dwa, _) = build_l1_terminal_dwa(
        &tokenizer,
        standalone_order.as_ref(),
        &mut standalone_map,
        2,
        &active,
        flat_trans.as_ref(),
        standalone_reuse.as_ref(),
    )
    .expect("standalone-topology exact L1 DWA");
    for raw_state in 0..tokenizer.num_states() as usize {
        let shared_tsid = shared_map.tokenizer_states.original_to_internal[raw_state];
        let standalone_tsid = standalone_map.tokenizer_states.original_to_internal[raw_state];
        for (&token_id, bytes) in subset_vocab.entries_map().iter() {
            let shared_token = shared_map.internal_token_for_original(token_id).expect("shared token");
            let standalone_token =
                standalone_map.internal_token_for_original(token_id).expect("standalone token");
            for terminal in 0..2u32 {
                assert_eq!(
                    shared_dwa
                        .eval_word(&[terminal as i32])
                        .tokens_for_tsid(shared_tsid)
                        .contains(shared_token),
                    standalone_dwa
                        .eval_word(&[terminal as i32])
                        .tokens_for_tsid(standalone_tsid)
                        .contains(standalone_token),
                    "raw_state={raw_state} token={token_id} bytes={bytes:?} terminal={terminal}",
                );
            }
        }
    }
}

#[test]
fn generic_epsilon_l1_exact_composes_a_proved_seed_partition() {
    let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
    let vocab = Vocab::new(vec![
        (0, b"".to_vec()),
        (1, b"a".to_vec()),
        (2, b"aa".to_vec()),
        (3, b"ab".to_vec()),
        (4, b"b".to_vec()),
        (5, b"ba".to_vec()),
        (6, b"bb".to_vec()),
    ]);
    let active = [true, true];
    let flat_trans = build_flat_transition_table(&tokenizer);

    let (mut baseline_map, baseline_order, _, _, baseline_reuse) =
        build_l1_generic_nfa_exact_id_map(
            &tokenizer,
            &vocab,
            &active,
            &flat_trans,
            None,
            None,
            None,
            None,
        );
    let seed_map = baseline_map.tokenizer_states.clone();
    let (mut projected_map, projected_order, _, projected_profile, projected_reuse) =
        build_l1_generic_nfa_exact_id_map(
            &tokenizer,
            &vocab,
            &active,
            &flat_trans,
            Some(&seed_map),
            None,
            None,
            None,
        );
    assert_eq!(
        projected_profile.initial_states_considered,
        seed_map.num_internal_ids() as usize,
    );

    let baseline_classes = &baseline_map.tokenizer_states.original_to_internal;
    let projected_classes = &projected_map.tokenizer_states.original_to_internal;
    for left in 0..baseline_classes.len() {
        for right in 0..baseline_classes.len() {
            assert_eq!(
                baseline_classes[left] == baseline_classes[right],
                projected_classes[left] == projected_classes[right],
                "seeded exact L1 composition changed the partition for {left} <> {right}",
            );
        }
    }

    let (baseline_dwa, _) = build_l1_terminal_dwa(
        &tokenizer,
        baseline_order.as_ref(),
        &mut baseline_map,
        2,
        &active,
        flat_trans.as_ref(),
        baseline_reuse.as_ref(),
    )
    .expect("baseline exact generic epsilon L1 DWA");
    let (projected_dwa, _) = build_l1_terminal_dwa(
        &tokenizer,
        projected_order.as_ref(),
        &mut projected_map,
        2,
        &active,
        flat_trans.as_ref(),
        projected_reuse.as_ref(),
    )
    .expect("projected exact generic epsilon L1 DWA");
    for raw_state in 0..tokenizer.num_states() as usize {
        let baseline_tsid = baseline_map.tokenizer_states.original_to_internal[raw_state];
        let projected_tsid = projected_map.tokenizer_states.original_to_internal[raw_state];
        for (&token_id, bytes) in vocab.entries_map().iter() {
            let baseline_token = baseline_map
                .internal_token_for_original(token_id)
                .expect("baseline token");
            let projected_token = projected_map
                .internal_token_for_original(token_id)
                .expect("projected token");
            for terminal in 0..2u32 {
                assert_eq!(
                    baseline_dwa
                        .eval_word(&[terminal as i32])
                        .tokens_for_tsid(baseline_tsid)
                        .contains(baseline_token),
                    projected_dwa
                        .eval_word(&[terminal as i32])
                        .tokens_for_tsid(projected_tsid)
                        .contains(projected_token),
                    "raw_state={raw_state} token={token_id} bytes={bytes:?} terminal={terminal}",
                );
            }
        }
    }
}

#[test]
fn generic_epsilon_l1_weights_match_exact_active_state_set_signatures() {
    let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
    let vocab = Vocab::new(
        vec![
            (0, b"".to_vec()),
            (1, b"a".to_vec()),
            (2, b"b".to_vec()),
            (3, b"aa".to_vec()),
        ]);
    let active = [true, true];

    let (mut fallback_id_map, fallback_order, _, _, _) =
        build_l1_generic_nfa_fallback_id_map(&tokenizer, &vocab, None);
    let (fallback_dwa, _) = build_l1_generic_nfa_terminal_dwa(
        &tokenizer,
        fallback_order.as_ref(),
        &mut fallback_id_map,
        2,
        &active,
    )
    .expect("generic epsilon L1 fallback fixture must produce a terminal DWA");

    let flat_trans = build_flat_transition_table(&tokenizer);
    let (mut exact_id_map, exact_order, _, _, exact_reuse) =
        build_l1_generic_nfa_exact_id_map(
            &tokenizer,
            &vocab,
            &active,
            &flat_trans,
            None,
            None,
            None,
            None,
        );
    let (exact_dwa, _) = build_l1_terminal_dwa(
        &tokenizer,
        exact_order.as_ref(),
        &mut exact_id_map,
        2,
        &active,
        flat_trans.as_ref(),
        exact_reuse.as_ref(),
    )
    .expect("generic epsilon L1 exact fixture must produce a terminal DWA");

    for raw_state in 0..tokenizer.num_states() {
        let fallback_tsid =
            fallback_id_map.tokenizer_states.original_to_internal[raw_state as usize];
        let exact_tsid = exact_id_map.tokenizer_states.original_to_internal[raw_state as usize];
        assert_ne!(fallback_tsid, u32::MAX, "fallback raw_state={raw_state}");
        assert_ne!(exact_tsid, u32::MAX, "exact raw_state={raw_state}");
        for (&token_id, bytes) in vocab.entries_map().iter() {
            let fallback_token =
                fallback_id_map.internal_token_for_original(token_id).expect("fallback token");
            let exact_token = exact_id_map.internal_token_for_original(token_id).expect("exact token");
            assert_ne!(fallback_token, u32::MAX, "fallback token={token_id}");
            assert_ne!(exact_token, u32::MAX, "exact token={token_id}");
            let end_states = tokenizer.execute_from_state_end_only(bytes, raw_state);
            for terminal in 0..2u32 {
                let expected = end_states.iter().any(|&state| {
                    collect_active_terminal_signature(&tokenizer, state, &active)
                        .contains(&terminal)
                });
                let fallback_actual = fallback_dwa
                    .eval_word(&[terminal as i32])
                    .tokens_for_tsid(fallback_tsid)
                    .contains(fallback_token);
                let exact_actual = exact_dwa
                    .eval_word(&[terminal as i32])
                    .tokens_for_tsid(exact_tsid)
                    .contains(exact_token);
                assert_eq!(
                    fallback_actual, expected,
                    "fallback raw_state={raw_state} token={token_id} bytes={bytes:?} terminal={terminal}",
                );
                assert_eq!(
                    exact_actual, expected,
                    "exact raw_state={raw_state} token={token_id} bytes={bytes:?} terminal={terminal}",
                );
            }
        }
    }
}
