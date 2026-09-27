use std::sync::Arc;

use super::*;
use crate::automata::lexer::ast::Expr;
use crate::automata::lexer::compile::{
    build_regex, build_regex_partitioned_with_adaptive,
};

#[test]
fn remaining_horizon_policy_selects_large_vocabulary_state_work() {
    assert!(l1_remaining_horizon_quotients_enabled(48_220, 82_266));
    assert!(l1_remaining_horizon_quotients_enabled(5_709, 82_266));
    assert!(!l1_remaining_horizon_quotients_enabled(48_220, 49_999));
    assert!(!l1_remaining_horizon_quotients_enabled(2_000, 50_000));
    assert!(!l1_remaining_horizon_quotients_enabled(100_001, 82_266));
}

#[test]
fn remaining_horizon_probe_rejects_small_raw_frontiers() {
    assert!(!l1_remaining_horizon_target_probe_supports_quotients(9_949));
    assert!(!l1_remaining_horizon_target_probe_supports_quotients(19_999));
    assert!(l1_remaining_horizon_target_probe_supports_quotients(45_355));
}

#[test]
fn filtered_exact_profiles_use_projected_self_loops() {
    let expressions = vec![
        Expr::U8Seq(b"a".to_vec()),
        Expr::Repeat {
            expr: Box::new(Expr::U8Seq(b"b".to_vec())),
            min: 0,
            max: None,
        },
    ];
    let tokenizer = build_regex(&expressions).into_tokenizer(
        expressions.len() as u32,
        Some(Arc::from(expressions.into_boxed_slice())),
    );
    let flat_trans: Arc<[u32]> = Arc::from(build_flat_transition_table(&tokenizer));
    let active = [true, false];
    let filtered = TokenizerView::new_filtered_from_flat_trans(
        &flat_trans,
        &tokenizer,
        &active,
    );
    assert!(
        (0..tokenizer.num_states() as usize).any(|state| {
            tokenizer.all_self_loop_bytes()[state].contains(b'b')
                && filtered.dfa().trans(state, b'b' as usize) == u32::MAX
        }),
        "fixture must project an inactive raw self-loop to dead",
    );

    let vocab = Vocab::new(vec![
        (0, b"a".to_vec()),
        (1, b"b".to_vec()),
        (2, b"bb".to_vec()),
        (3, b"bbb".to_vec()),
    ]);
    let order = l1_identity_vocab_order(&vocab);
    let states = (0..tokenizer.num_states() as usize).collect::<Vec<_>>();
    let (actual, _) = find_l1_exact_state_equivalence_by_token_signatures(
        &tokenizer,
        order.as_ref(),
        &states,
        &active,
        &flat_trans,
        None,
    );
    let (state_to_terminal_signature, terminal_signatures) =
        build_l1_flat_state_to_terminal_signatures(filtered.dfa());
    let (expected, _) = find_l1_exact_state_equivalence_by_flat_signatures(
        order.as_ref(),
        &states,
        state_to_terminal_signature,
        terminal_signatures,
        &filtered,
        None,
        true,
        0.0,
        None,
    );
    assert_eq!(actual, expected);
}

#[test]
fn packed_suffix_profiles_match_batched_profiles() {
    let expressions = vec![
        Expr::U8Seq(b"a".to_vec()),
        Expr::U8Seq(b"ab".to_vec()),
        Expr::Choice(vec![
            Expr::U8Seq(b"ac".to_vec()),
            Expr::U8Seq(b"ba".to_vec()),
        ]),
        Expr::U8Seq(b"cab".to_vec()),
    ];
    let tokenizer = build_regex(&expressions).into_tokenizer(
        expressions.len() as u32,
        Some(Arc::from(expressions.into_boxed_slice())),
    );
    let sorted_entries: Vec<(u32, Arc<[u8]>)> = vec![
        (0, Arc::from(&b""[..])),
        (1, Arc::from(&b"a"[..])),
        (2, Arc::from(&b"ab"[..])),
        (3, Arc::from(&b"ab"[..])),
        (4, Arc::from(&b"abc"[..])),
        (5, Arc::from(&b"abd"[..])),
        (6, Arc::from(&b"ac"[..])),
        (7, Arc::from(&b"b"[..])),
        (8, Arc::from(&b"ba"[..])),
        (9, Arc::from(&b"bb"[..])),
        (10, Arc::from(&b"c"[..])),
        (11, Arc::from(&b"cab"[..])),
    ];
    let buckets = build_l1_sorted_token_buckets(&sorted_entries);
    let active_terminals = vec![true, false, true, true];
    let (state_to_terminal_signature, _) =
        build_l1_state_to_terminal_signatures(&tokenizer, &active_terminals);
    let flat_trans: Arc<[u32]> = Arc::from(build_flat_transition_table(&tokenizer));
    let self_loop_bytes_by_state = tokenizer.all_self_loop_bytes();
    let targets: Vec<u32> = (0..tokenizer.num_states()).collect();
    let mut relevant_bytes = [false; 256];
    let max_token_len = sorted_entries
        .iter()
        .map(|(_, bytes)| {
            for &byte in bytes.iter() {
                relevant_bytes[byte as usize] = true;
            }
            bytes.len()
        })
        .max()
        .unwrap_or(0);
    let tokenizer_view = TokenizerView::new_filtered(&tokenizer, &active_terminals);
    let byte_to_class = compute_byte_classes(tokenizer_view.dfa());
    let horizon_maps = super::super::l2p::equivalence_analysis::state::max_length::find_canonical_state_maps_by_depth_from_labels(
        &tokenizer_view,
        max_token_len,
        &state_to_terminal_signature,
        Some(&relevant_bytes),
        Some(&byte_to_class),
    );
    let reference_horizon_maps = super::super::l2p::equivalence_analysis::state::max_length::find_canonical_state_maps_by_depth_from_labels_reference(
        &tokenizer_view,
        max_token_len,
        &state_to_terminal_signature,
        Some(&relevant_bytes),
        Some(&byte_to_class),
    );
    assert_eq!(horizon_maps, reference_horizon_maps);

    for first_byte in 0..256usize {
        let token_ids = &buckets.token_indices_by_first_byte[first_byte];
        if token_ids.is_empty() {
            continue;
        }
        let mut expected = l1_bucket_suffix_signature_profiles_batched(
            first_byte as u8,
            &targets,
            &sorted_entries,
            token_ids,
            &buckets.suffix_lcps_by_first_byte[first_byte],
            &buckets.suffix_subtree_bytes[first_byte],
            &buckets.suffix_first_bytes_by_bucket[first_byte],
              buckets.has_empty_suffix_by_bucket[first_byte],
              &state_to_terminal_signature,
              &flat_trans,
              None,
              tokenizer.num_states() as usize,
              None,
        );
        let suffix_horizon = token_ids
            .iter()
            .map(|&token_id| sorted_entries[token_id].1.len().saturating_sub(1))
            .max()
            .unwrap_or(0);
        let mut actual = l1_bucket_suffix_signature_profiles_packed(
            first_byte as u8,
            &targets,
            &sorted_entries,
            token_ids,
            &buckets.suffix_lcps_by_first_byte[first_byte],
            &buckets.suffix_subtree_bytes[first_byte],
            &buckets.suffix_first_bytes_by_bucket[first_byte],
            buckets.has_empty_suffix_by_bucket[first_byte],
            &state_to_terminal_signature,
            self_loop_bytes_by_state.as_ref(),
            &flat_trans,
            None,
            tokenizer.num_states() as usize,
            None,
            None,
            suffix_horizon,
            None,
        );
        let mut quotient_actual = l1_bucket_suffix_signature_profiles_packed(
            first_byte as u8,
            &targets,
            &sorted_entries,
            token_ids,
            &buckets.suffix_lcps_by_first_byte[first_byte],
            &buckets.suffix_subtree_bytes[first_byte],
            &buckets.suffix_first_bytes_by_bucket[first_byte],
            buckets.has_empty_suffix_by_bucket[first_byte],
            &state_to_terminal_signature,
            self_loop_bytes_by_state.as_ref(),
            &flat_trans,
            None,
            tokenizer.num_states() as usize,
            None,
            Some(&horizon_maps),
            suffix_horizon,
            None,
        );
        expected.sort_unstable_by_key(|(key, _)| *key);
        actual.sort_unstable_by_key(|(key, _)| *key);
        quotient_actual.sort_unstable_by_key(|(key, _)| *key);
        let actual: Vec<((u8, u32), Vec<(u32, u32, u32)>)> = actual
            .into_iter()
            .map(|(key, profile)| (key, profile.as_ref().to_vec()))
            .collect();
        let quotient_actual: Vec<((u8, u32), Vec<(u32, u32, u32)>)> = quotient_actual
            .into_iter()
            .map(|(key, profile)| (key, profile.as_ref().to_vec()))
            .collect();
        assert_eq!(actual, expected, "raw packed first byte {first_byte}");
        assert_eq!(
            quotient_actual, expected,
            "quotiented packed first byte {first_byte}",
        );
    }
}

#[test]
fn suffix_trie_bottom_up_ranges_cover_prefix_and_siblings() {
    let entries: Vec<(u32, Arc<[u8]>)> = vec![
        (0, Arc::from(b"a".as_slice())),
        (1, Arc::from(b"ab".as_slice())),
        (2, Arc::from(b"ac".as_slice())),
        (3, Arc::from(b"b".as_slice())),
    ];
    let trie = L1PackedSuffixTrie::build(&entries, &[0, 1, 2], &[0, 0, 0]);
    assert_eq!((trie.nodes[0].subtree_start, trie.nodes[0].subtree_end), (0, 2));
    let first_child = trie.nodes[0].first_child as usize;
    let second_child = trie.nodes[first_child].next_sibling as usize;
    assert_eq!((trie.nodes[first_child].subtree_start, trie.nodes[first_child].subtree_end), (1, 1));
    assert_eq!((trie.nodes[second_child].subtree_start, trie.nodes[second_child].subtree_end), (2, 2));
}

#[test]
fn suffix_trie_preserves_duplicate_byte_alias_ranges() {
    let entries: Vec<(u32, Arc<[u8]>)> = vec![
        (91, Arc::from(b"a".as_slice())),
        (7, Arc::from(b"a".as_slice())),
        (4, Arc::from(b"ab".as_slice())),
    ];
    let trie = L1PackedSuffixTrie::build(&entries, &[0, 1, 2], &[0, 0, 0]);
    let root = trie.nodes[0];
    assert_eq!((root.terminal_token, root.terminal_token_end), (0, 1));
    assert_eq!((root.subtree_start, root.subtree_end), (0, 2));
}

#[test]
fn sparse_end_rep_groups_match_terminal_membership() {
    let groups = vec![vec![0usize, 2usize], vec![1usize], vec![3usize]];
    let terminal_to_end_reps = vec![vec![0u32, 2], vec![1], vec![0, 2], vec![3]];
    assert_eq!(
        build_end_rep_groups(&groups, &terminal_to_end_reps, 4),
        vec![vec![0], vec![1], vec![0], vec![2]],
    );
}

#[test]
fn frozen_walk_profiles_preserve_signature_ranges() {
    let empty: Arc<[(u32, u32, u32)]> = Arc::from([]);
    let profile_one: Arc<[(u32, u32, u32)]> = Arc::from([
        (1, 2, 3),
        (2, 4, 4),
        (1, 7, 8),
        (0, 9, 10),
    ]);
    let reuse = L1ExactProfileReuse {
        target_to_profile_id: [((b'a', 17), 1u32)].into_iter().collect(),
        walk_profiles_by_id: vec![
            freeze_l1_walk_profile(&empty),
            freeze_l1_walk_profile(&profile_one),
        ],
        profile_representatives_by_internal: Arc::from([]),
        representative_profile_ids: FxHashMap::default(),
        direct_terminal_signatures: Arc::from([]),
        direct_state_to_terminal_signature: Arc::from([]),
    };
    let cache = reuse.materialize_walk_cache();
    let profile = cache.get(&(b'a', 17)).expect("profile present");
    let grouped: Vec<(u32, Vec<(u32, u32)>)> = profile
        .iter()
        .map(|(signature, ranges)| (*signature, ranges.as_ref().to_vec()))
        .collect();
    assert_eq!(grouped, vec![(0, vec![(2, 3), (7, 8)]), (1, vec![(4, 4)])]);
}

#[test]
fn deterministic_dispatch_exact_profile_reuse_matches_scalar_and_fallback() {
    let expressions = vec![
        Expr::U8Seq(b"a".to_vec()),
        Expr::U8Seq(b"ab".to_vec()),
        Expr::Repeat {
            expr: Box::new(Expr::U8Seq(b"b".to_vec())),
            min: 1,
            max: None,
        },
    ];
    let tokenizer = build_regex_partitioned_with_adaptive(
        &expressions,
        &[0, 1, 2],
        false,
    )
    .into_tokenizer(
        expressions.len() as u32,
        Some(Arc::from(expressions.clone().into_boxed_slice())),
    );
    assert!(tokenizer.has_deterministic_dispatch());

    let vocab = Vocab::new(
        vec![
            (0, b"".to_vec()),
            (1, b"a".to_vec()),
            (2, b"ab".to_vec()),
            (3, b"b".to_vec()),
            (4, b"bb".to_vec()),
            (5, b"x".to_vec()),
        ]);
    let active_terminals = vec![true; expressions.len()];
    let flat_trans: Arc<[u32]> = build_flat_transition_table(&tokenizer).into();
    let (id_map, order, _, _, exact_profile_reuse) = build_l1_id_map(
        "test",
        &tokenizer,
        &vocab,
        &active_terminals,
        &flat_trans,
        None,
        None,
    );
    let exact_profile_reuse =
        exact_profile_reuse.expect("structured dispatch must retain exact L1 profiles");

    let mut optimized_id_map = id_map.clone();
    let mut fallback_id_map = id_map;
    let (optimized, _) = build_l1_terminal_dwa(
        &tokenizer,
        order.as_ref(),
        &mut optimized_id_map,
        expressions.len() as u32,
        &active_terminals,
        flat_trans.as_ref(),
        Some(&exact_profile_reuse),
    )
    .expect("optimized L1 DWA");
    let (fallback, _) = build_l1_terminal_dwa(
        &tokenizer,
        order.as_ref(),
        &mut fallback_id_map,
        expressions.len() as u32,
        &active_terminals,
        flat_trans.as_ref(),
        None,
    )
    .expect("fallback L1 DWA");

    for raw_state in 0..tokenizer.num_states() {
        let optimized_tsid =
            optimized_id_map.tokenizer_states.original_to_internal[raw_state as usize];
        let fallback_tsid =
            fallback_id_map.tokenizer_states.original_to_internal[raw_state as usize];
        assert_ne!(optimized_tsid, u32::MAX, "raw_state={raw_state}");
        assert_ne!(fallback_tsid, u32::MAX, "raw_state={raw_state}");

        for (&token_id, bytes) in vocab.entries_map().iter() {
            let optimized_token =
                optimized_id_map.internal_token_for_original(token_id).expect("optimized token");
            let fallback_token =
                fallback_id_map.internal_token_for_original(token_id).expect("fallback token");
            let end_states = tokenizer.execute_from_state_end_only(bytes, raw_state);

            for terminal in 0..expressions.len() as u32 {
                let expected = end_states.iter().any(|&state| {
                    collect_active_terminal_signature(
                        &tokenizer,
                        state,
                        &active_terminals,
                    )
                    .contains(&terminal)
                });
                let optimized_actual = optimized
                    .eval_word(&[terminal as i32])
                    .tokens_for_tsid(optimized_tsid)
                    .contains(optimized_token);
                let fallback_actual = fallback
                    .eval_word(&[terminal as i32])
                    .tokens_for_tsid(fallback_tsid)
                    .contains(fallback_token);
                assert_eq!(
                    fallback_actual, expected,
                    "fallback raw_state={raw_state} token={token_id} bytes={bytes:?} terminal={terminal}"
                );
                assert_eq!(
                    optimized_actual, expected,
                    "optimized raw_state={raw_state} token={token_id} bytes={bytes:?} terminal={terminal}"
                );
            }
        }
    }
}

#[test]
fn deterministic_dispatch_reuse_survives_initial_profile_class_isolation() {
    let expressions = vec![
        Expr::U8Seq(b"z".to_vec()),
        Expr::U8Seq(b"q".to_vec()),
    ];
    let tokenizer = build_regex_partitioned_with_adaptive(
        &expressions,
        &[0, 1],
        false,
    )
    .into_tokenizer(
        expressions.len() as u32,
        Some(Arc::from(expressions.clone().into_boxed_slice())),
    );
    assert!(tokenizer.has_deterministic_dispatch());

    let vocab = Vocab::new(
        vec![(0, b"".to_vec()), (1, b"a".to_vec())]);
    let active_terminals = vec![true, false];
    let flat_trans: Arc<[u32]> = build_flat_transition_table(&tokenizer).into();
    let (id_map, order, _, _, exact_profile_reuse) = build_l1_id_map(
        "test",
        &tokenizer,
        &vocab,
        &active_terminals,
        &flat_trans,
        None,
        None,
    );
    let exact_profile_reuse =
        exact_profile_reuse.expect("structured dispatch must retain exact L1 profiles");
    assert_eq!(
        id_map.num_tsids() as usize,
        exact_profile_reuse.profile_representatives_by_internal.len() + 1,
        "the synthetic initial state must be split out of a pre-isolation exact profile class",
    );
    let initial_tsid =
        id_map.tokenizer_states.original_to_internal[tokenizer.initial_state_id() as usize];
    assert_eq!(
        initial_tsid as usize,
        exact_profile_reuse.profile_representatives_by_internal.len(),
        "isolate_original must append the synthetic initial TSID without renaming existing exact classes",
    );

    let mut optimized_id_map = id_map.clone();
    let mut fallback_id_map = id_map;
    let (optimized, _) = build_l1_terminal_dwa(
        &tokenizer,
        order.as_ref(),
        &mut optimized_id_map,
        expressions.len() as u32,
        &active_terminals,
        flat_trans.as_ref(),
        Some(&exact_profile_reuse),
    )
    .expect("optimized L1 DWA");
    let (fallback, _) = build_l1_terminal_dwa(
        &tokenizer,
        order.as_ref(),
        &mut fallback_id_map,
        expressions.len() as u32,
        &active_terminals,
        flat_trans.as_ref(),
        None,
    )
    .expect("fallback L1 DWA");

    for raw_state in 0..tokenizer.num_states() {
        let optimized_tsid =
            optimized_id_map.tokenizer_states.original_to_internal[raw_state as usize];
        let fallback_tsid =
            fallback_id_map.tokenizer_states.original_to_internal[raw_state as usize];
        for (&token_id, bytes) in vocab.entries_map().iter() {
            let optimized_token =
                optimized_id_map.internal_token_for_original(token_id).expect("optimized token");
            let fallback_token =
                fallback_id_map.internal_token_for_original(token_id).expect("fallback token");
            let end_states = tokenizer.execute_from_state_end_only(bytes, raw_state);
            let expected = end_states.iter().any(|&state| {
                collect_active_terminal_signature(
                    &tokenizer,
                    state,
                    &active_terminals,
                )
                .contains(&0)
            });
            let optimized_actual = optimized
                .eval_word(&[0])
                .tokens_for_tsid(optimized_tsid)
                .contains(optimized_token);
            let fallback_actual = fallback
                .eval_word(&[0])
                .tokens_for_tsid(fallback_tsid)
                .contains(fallback_token);
            assert_eq!(
                optimized_actual, expected,
                "optimized raw_state={raw_state} token={token_id} bytes={bytes:?}",
            );
            assert_eq!(
                fallback_actual, expected,
                "fallback raw_state={raw_state} token={token_id} bytes={bytes:?}",
            );
        }
    }
}

#[test]
fn large_group_assembly_uses_parallel_crossover_only_with_workers() {
    assert!(auto_use_sequential_l1_group_assembly(
        PARALLEL_L1_GROUP_ASSEMBLY_MIN_VISITS - 1,
        8,
    ));
    assert!(!auto_use_sequential_l1_group_assembly(
        PARALLEL_L1_GROUP_ASSEMBLY_MIN_VISITS,
        8,
    ));
    assert!(auto_use_sequential_l1_group_assembly(
        PARALLEL_L1_GROUP_ASSEMBLY_MIN_VISITS,
        1,
    ));
}

#[test]
fn exact_state_hash_partition_matches_direct_token_profiles() {
    let expressions = vec![
        Expr::U8Seq(b"a".to_vec()),
        Expr::U8Seq(b"ab".to_vec()),
        Expr::Choice(vec![Expr::U8Seq(b"ac".to_vec()), Expr::U8Seq(b"ba".to_vec())]),
        Expr::U8Seq(b"cab".to_vec()),
    ];
    let tokenizer = build_regex(&expressions).into_tokenizer(
        expressions.len() as u32,
        Some(Arc::from(expressions.into_boxed_slice())),
    );
    let vocab = Vocab::new(
        vec![
            (0, b"".to_vec()),
            (1, b"a".to_vec()),
            (2, b"ab".to_vec()),
            (3, b"abc".to_vec()),
            (4, b"abd".to_vec()),
            (5, b"ac".to_vec()),
            (6, b"b".to_vec()),
            (7, b"ba".to_vec()),
            (8, b"bb".to_vec()),
            (9, b"c".to_vec()),
            (10, b"cab".to_vec()),
        ]);
    let active_terminals = vec![true, false, true, true];
    let order = l1_identity_vocab_order(&vocab);
    let flat_trans: Arc<[u32]> = Arc::from(build_flat_transition_table(&tokenizer));
    let states: Vec<usize> = (0..tokenizer.num_states() as usize).collect();
    let (mapping, _) =
        find_l1_exact_state_equivalence_by_token_signatures_with_first_target_cache(
            &tokenizer,
            &order,
            &states,
            &active_terminals,
            &flat_trans,
            None,
            Some(true),
        );
    let (uncached_mapping, _) =
        find_l1_exact_state_equivalence_by_token_signatures_with_first_target_cache(
            &tokenizer,
            &order,
            &states,
            &active_terminals,
            &flat_trans,
            None,
            Some(false),
        );
    assert_eq!(uncached_mapping, mapping);
    let num_states = tokenizer.num_states() as usize;
    let mut transitions_by_byte = vec![u32::MAX; num_states * 256];
    for state in 0..num_states {
        for byte in 0..256usize {
            transitions_by_byte[byte * num_states + state] = flat_trans[state * 256 + byte];
        }
    }
    let (transposed_mapping, _) = find_l1_exact_state_equivalence_by_token_signatures(
        &tokenizer,
        &order,
        &states,
        &active_terminals,
        &flat_trans,
        Some(&transitions_by_byte),
    );
    assert_eq!(transposed_mapping, mapping);
    let (state_to_signature, _) =
        build_l1_state_to_terminal_signatures(&tokenizer, &active_terminals);
    let profiles: Vec<Vec<(u32, u32, u32)>> = states
        .iter()
        .map(|&state| {
            l1_token_signature_profile_for_state(
                state as u32,
                order.token_entries_sorted.as_ref(),
                &order.token_buckets,
                &state_to_signature,
                &flat_trans,
            )
        })
        .collect();

    for left in 0..states.len() {
        for right in 0..states.len() {
            assert_eq!(
                mapping[left] == mapping[right],
                profiles[left] == profiles[right],
                "state pair ({left}, {right})"
            );
        }
    }
}
