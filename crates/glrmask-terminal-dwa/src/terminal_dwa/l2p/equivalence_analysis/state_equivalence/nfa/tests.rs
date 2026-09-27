use super::*;

#[test]
fn state_map_compacts_sparse_worklist_class_labels() {
    let map = build_state_map(
        &[vec![0], vec![1], vec![2]],
        &[0, 1, 2],
        &[0, 2, 2],
        3,
    );

    assert_eq!(map.original_to_internal, vec![0, 1, 1]);
    assert_eq!(map.internal_to_originals, vec![vec![0], vec![1, 2]]);
    assert_eq!(map.representative_original_ids, vec![0, 1]);
}

#[test]
fn prebuilt_superset_token_trie_matches_subset_analysis() {
    let tokenizer =
        crate::automata::lexer::tokenizer::arbitrary_epsilon_l1_test_tokenizer();
    let raw_states = (0..tokenizer.num_states() as usize).collect::<Vec<_>>();
    let flat_trans = crate::compiler::stages::id_map_and_terminal_dwa::l1::build_flat_transition_table(&tokenizer);
    let parent_tokens: Vec<&[u8]> = vec![b"", b"a", b"aa", b"ab", b"b", b"ba", b"x"];
    let subset_tokens: Vec<&[u8]> = vec![b"", b"a", b"aa", b"ab", b"b", b"ba"];
    let active = [true, true];
    let parent_trie = build_token_bounded_analysis_trie_sorted(&parent_tokens);
    let direct = build_token_bounded_analysis_view_projected_sorted_with_raw_transitions(
        &tokenizer,
        &flat_trans,
        &raw_states,
        &subset_tokens,
        &active,
    );
    let reused =
        build_token_bounded_analysis_view_projected_sorted_with_raw_transitions_and_trie(
            &tokenizer,
            &flat_trans,
            &raw_states,
            &subset_tokens,
            &active,
            Some(&parent_trie),
        );
    for &raw_state in &raw_states {
        let direct_start = direct.view_state_for_raw_start(raw_state);
        let reused_start = reused.view_state_for_raw_start(raw_state);
        for &token in &subset_tokens {
            assert_eq!(
                view_trace(&direct.tokenizer_view, direct_start, token),
                view_trace(&reused.tokenizer_view, reused_start, token),
                "raw_state={raw_state} token={token:?}",
            );
        }
    }
}

#[test]
fn prebuilt_superset_token_trie_matches_bounded_reset_suffix_analysis() {
    let tokenizer =
        crate::automata::lexer::tokenizer::arbitrary_epsilon_l1_test_tokenizer();
    let raw_states = (0..tokenizer.num_states() as usize).collect::<Vec<_>>();
    let parent_tokens: Vec<&[u8]> =
        vec![b"a", b"aa", b"ab", b"aba", b"b", b"ba", b"x"];
    let subset_tokens: Vec<&[u8]> = vec![b"aa", b"aba", b"ba"];
    let parent_trie = build_token_bounded_analysis_trie_sorted(&parent_tokens);
    let direct = build_bounded_analysis_view(
        &tokenizer,
        &raw_states,
        &subset_tokens,
        Some(&[true, true]),
    );
    let reused = build_bounded_analysis_view_with_trie(
        &tokenizer,
        &raw_states,
        &subset_tokens,
        Some(&[true, true]),
        Some(&parent_trie),
    );
    let observed = subset_tokens
        .iter()
        .flat_map(|token| (0..token.len()).map(move |offset| &token[offset..]))
        .collect::<Vec<_>>();
    for &raw_state in &raw_states {
        let direct_start = direct.view_state_for_raw_start(raw_state);
        let reused_start = reused.view_state_for_raw_start(raw_state);
        for &bytes in &observed {
            assert_eq!(
                view_trace(&direct.tokenizer_view, direct_start, bytes),
                view_trace(&reused.tokenizer_view, reused_start, bytes),
                "raw_state={raw_state} bytes={bytes:?}",
            );
        }
    }
}

#[test]
fn sorted_byte_trie_matches_generic_builder() {
    let sequences: Vec<&[u8]> = vec![
        b"",
        b"a",
        b"a",
        b"aa",
        b"ab",
        b"aba",
        b"b",
        b"ba",
        b"z",
        b"\xff",
    ];
    assert_eq!(
        build_byte_trie(sequences.iter().copied()),
        build_byte_trie_sorted(&sequences),
    );
}
use crate::automata::lexer::tokenizer::arbitrary_epsilon_l1_test_tokenizer;

#[test]
fn cached_closed_config_step_matches_scalar_step_all() {
    let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
    let singleton_closures = tokenizer.all_singleton_epsilon_closures();
    let mut raw_transitions = vec![u32::MAX; tokenizer.num_states() as usize * 256];
    for state in 0..tokenizer.num_states() {
        for (byte, target) in tokenizer.transitions_from(state) {
            raw_transitions[state as usize * 256 + byte as usize] = target;
        }
    }
    let active_group_masks: [Option<&[bool]>; 4] = [
        None,
        Some(&[true, false]),
        Some(&[false, true]),
        Some(&[true, true]),
    ];

    for active_groups in active_group_masks {
        let active_language = raw_active_language_states(&tokenizer, active_groups);
        let mut target_marks = vec![0u32; tokenizer.num_states() as usize];
        let mut target_generation = 0u32;
        for raw_state in 0..tokenizer.num_states() as usize {
            let config = singleton_closures[raw_state]
                .iter()
                .copied()
                .filter(|&state| {
                    active_language
                        .as_deref()
                        .is_none_or(|active| active[state as usize])
                })
                .collect::<Vec<_>>();
            for byte in 0u8..=u8::MAX {
                let scalar = tokenizer
                    .step_all(&config, byte)
                    .iter()
                    .copied()
                    .filter(|&state| {
                        active_language
                            .as_deref()
                            .is_none_or(|active| active[state as usize])
                    })
                    .collect::<Vec<_>>();
                let cached = step_epsilon_closed_config_cached(
                    &tokenizer,
                    None,
                    &config,
                    byte,
                    singleton_closures.as_ref(),
                    active_language.as_deref(),
                    &mut target_marks,
                    &mut target_generation,
                );
                assert_eq!(cached, scalar, "raw_state={raw_state} byte={byte}");
                let cached_from_raw = step_epsilon_closed_config_cached(
                    &tokenizer,
                    Some(&raw_transitions),
                    &config,
                    byte,
                    singleton_closures.as_ref(),
                    active_language.as_deref(),
                    &mut target_marks,
                    &mut target_generation,
                );
                assert_eq!(
                    cached_from_raw, scalar,
                    "raw transition table raw_state={raw_state} byte={byte}",
                );
            }
        }
    }
}

#[test]
fn projected_sorted_raw_transition_view_matches_tokenizer_step_view() {
    let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
    let raw_start_states = (0..tokenizer.num_states() as usize).collect::<Vec<_>>();
    let active_groups = [true, false];
    let tokens: [&[u8]; 7] = [b"a", b"aa", b"ab", b"aba", b"b", b"ba", b"xyz"];
    let mut raw_transitions = vec![u32::MAX; tokenizer.num_states() as usize * 256];
    for state in 0..tokenizer.num_states() {
        for (byte, target) in tokenizer.transitions_from(state) {
            raw_transitions[state as usize * 256 + byte as usize] = target;
        }
    }

    let scalar = build_token_bounded_analysis_view_projected_sorted(
        &tokenizer,
        &raw_start_states,
        &tokens,
        &active_groups,
    );
    let dense = build_token_bounded_analysis_view_projected_sorted_with_raw_transitions(
        &tokenizer,
        &raw_transitions,
        &raw_start_states,
        &tokens,
        &active_groups,
    );

    assert_eq!(
        dense.tokenizer_view.dfa().states.len(),
        scalar.tokenizer_view.dfa().states.len(),
    );
    for &raw_state in &raw_start_states {
        let scalar_start = scalar.view_state_for_raw_start(raw_state);
        let dense_start = dense.view_state_for_raw_start(raw_state);
        for &token in &tokens {
            assert_eq!(
                view_trace(&dense.tokenizer_view, dense_start, token),
                view_trace(&scalar.tokenizer_view, scalar_start, token),
                "raw_state={raw_state} token={token:?}",
            );
        }
    }
}

fn view_trace(
    view: &TokenizerView,
    start_state: usize,
    token: &[u8],
) -> Vec<(Vec<usize>, Vec<usize>, bool)> {
    let dfa = view.dfa();
    let mut state = start_state;
    let mut trace = vec![(
        dfa.states[state].finalizers.clone(),
        dfa.states[state].possible_future_group_ids.clone(),
        false,
    )];
    for &byte in token {
        let target = dfa.trans(state, byte as usize);
        if target == u32::MAX {
            trace.push((Vec::new(), Vec::new(), true));
            break;
        }
        state = target as usize;
        trace.push((
            dfa.states[state].finalizers.clone(),
            dfa.states[state].possible_future_group_ids.clone(),
            false,
        ));
    }
    trace
}

#[test]
fn trie_frontier_expansion_matches_pair_hash_dfs() {
    let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
    let raw_start_states = (0..tokenizer.num_states() as usize).collect::<Vec<_>>();
    let active_groups = [true, false];
    let token_sets: [&[&[u8]]; 2] = [
        &[b"a", b"aa", b"ab", b"aba"],
        &[b"a", b"ab", b"b", b"ba", b"x", b"xyz"],
    ];

    for tokens in token_sets {
        let (frontier, frontier_work) = build_bounded_analysis_topology_impl_with_expansion(
            &tokenizer,
            None,
            &raw_start_states,
            tokens,
            false,
            true,
            false,
            true,
            Some(&active_groups),
            None,
            None,
            true,
        )
        .expect("frontier topology build");
        let (dfs, dfs_work) = build_bounded_analysis_topology_impl_with_expansion(
            &tokenizer,
            None,
            &raw_start_states,
            tokens,
            false,
            true,
            false,
            true,
            Some(&active_groups),
            None,
            None,
            false,
        )
        .expect("DFS topology build");
        assert_eq!(frontier_work, dfs_work);

        let frontier = frontier.materialize_already_projected(&tokenizer, &active_groups);
        let dfs = dfs.materialize_already_projected(&tokenizer, &active_groups);
        for &raw_state in &raw_start_states {
            let frontier_start = frontier.view_state_for_raw_start(raw_state);
            let dfs_start = dfs.view_state_for_raw_start(raw_state);
            for &token in tokens {
                assert_eq!(
                    view_trace(&frontier.tokenizer_view, frontier_start, token),
                    view_trace(&dfs.tokenizer_view, dfs_start, token),
                    "raw_state={raw_state} token={token:?}",
                );
            }
        }
    }

    // A sparse-root edge can reach a raw state that was not itself one of
    // the preseeded start states. The sparse path must fall back to exact
    // configuration construction rather than treating the missing
    // raw_start_to_view entry as DEAD.
    let subset_start_states = vec![2usize, 4usize];
    let subset_tokens: &[&[u8]] = &[b"a", b"b"];
    let subset_active_groups = [true, true];
    let (frontier, frontier_work) = build_bounded_analysis_topology_impl_with_expansion(
        &tokenizer,
        None,
        &subset_start_states,
        subset_tokens,
        false,
        true,
        false,
        true,
        Some(&subset_active_groups),
        None,
        None,
        true,
    )
    .expect("frontier topology build with unpreseeded targets");
    let (dfs, dfs_work) = build_bounded_analysis_topology_impl_with_expansion(
        &tokenizer,
        None,
        &subset_start_states,
        subset_tokens,
        false,
        true,
        false,
        true,
        Some(&subset_active_groups),
        None,
        None,
        false,
    )
    .expect("DFS topology build with unpreseeded targets");
    assert_eq!(frontier_work, dfs_work);
    let frontier = frontier.materialize_already_projected(&tokenizer, &subset_active_groups);
    let dfs = dfs.materialize_already_projected(&tokenizer, &subset_active_groups);
    for &raw_state in &subset_start_states {
        let frontier_start = frontier.view_state_for_raw_start(raw_state);
        let dfs_start = dfs.view_state_for_raw_start(raw_state);
        for &token in subset_tokens {
            assert_eq!(
                view_trace(&frontier.tokenizer_view, frontier_start, token),
                view_trace(&dfs.tokenizer_view, dfs_start, token),
                "unpreseeded target raw_state={raw_state} token={token:?}",
            );
        }
    }

    // Also cover the generic unsorted-token path, combined starts, and the
    // reset-suffix trie used by non-L1 callers.
    let unsorted_tokens: &[&[u8]] = &[b"xyz", b"a", b"ba", b"ab", b"x"];
    let (frontier, frontier_work) = build_bounded_analysis_topology_impl_with_expansion(
        &tokenizer,
        None,
        &raw_start_states,
        unsorted_tokens,
        true,
        false,
        true,
        false,
        None,
        None,
        None,
        true,
    )
    .expect("frontier topology build with reset suffixes");
    let (dfs, dfs_work) = build_bounded_analysis_topology_impl_with_expansion(
        &tokenizer,
        None,
        &raw_start_states,
        unsorted_tokens,
        true,
        false,
        true,
        false,
        None,
        None,
        None,
        false,
    )
    .expect("DFS topology build with reset suffixes");
    assert_eq!(frontier_work, dfs_work);
    let frontier = frontier.materialize(&tokenizer, None);
    let dfs = dfs.materialize(&tokenizer, None);
    for &raw_state in &raw_start_states {
        let frontier_start = frontier.view_state_for_raw_start(raw_state);
        let dfs_start = dfs.view_state_for_raw_start(raw_state);
        for &token in unsorted_tokens {
            assert_eq!(
                view_trace(&frontier.tokenizer_view, frontier_start, token),
                view_trace(&dfs.tokenizer_view, dfs_start, token),
                "combined raw_state={raw_state} token={token:?}",
            );
        }
    }
}

#[test]
fn active_filtered_powerset_drops_inactive_members_before_refinement() {
    use crate::automata::lexer::ast::Expr;
    use crate::automata::lexer::compile::build_regex_partitioned_with_adaptive;

    let expressions = vec![
        Expr::U8Seq(b"ab".to_vec()),
        Expr::U8Seq(b"xq".to_vec()),
        Expr::U8Seq(b"yrr".to_vec()),
    ];
    let tokenizer = build_regex_partitioned_with_adaptive(
        &expressions,
        &[0, 1, 2],
        false,
    )
    .into_tokenizer(
        expressions.len() as u32,
        Some(Arc::from(expressions.into_boxed_slice())),
    );
    let view = build_relevant_powerset_view(
        &tokenizer,
        &[true; 256],
        Some(&[true, false, false]),
        None,
    );
    let start_config = &view.configurations[view.start_state];
    assert!(!start_config.is_empty());
    assert!(start_config.iter().all(|&state| {
        tokenizer.matched_terminals_iter(state).any(|terminal| terminal == 0)
            || tokenizer
                .possible_future_terminals_iter(state)
                .any(|terminal| terminal == 0)
    }));
    let edge_start = view.edge_offsets[view.start_state] as usize;
    let edge_end = view.edge_offsets[view.start_state + 1] as usize;
    let start_edges = &view.edges[edge_start..edge_end];
    assert!(!start_edges.iter().any(|&(byte, _)| byte == b'x' || byte == b'y'));
}

#[test]
fn token_bounded_topology_materialization_drops_inactive_members() {
    use crate::automata::lexer::ast::Expr;
    use crate::automata::lexer::compile::build_regex_partitioned_with_adaptive;

    let expressions = vec![
        Expr::U8Seq(b"ab".to_vec()),
        Expr::U8Seq(b"xq".to_vec()),
        Expr::U8Seq(b"yrr".to_vec()),
    ];
    let tokenizer = build_regex_partitioned_with_adaptive(
        &expressions,
        &[0, 1, 2],
        false,
    )
    .into_tokenizer(
        expressions.len() as u32,
        Some(Arc::from(expressions.into_boxed_slice())),
    );
    let raw_start_states = (0..tokenizer.num_states() as usize).collect::<Vec<_>>();
    let tokens = [
        b"a".as_slice(),
        b"ab".as_slice(),
        b"x".as_slice(),
        b"xq".as_slice(),
        b"y".as_slice(),
        b"yrr".as_slice(),
    ];
    let topology =
        build_token_bounded_analysis_topology(&tokenizer, &raw_start_states, &tokens);
    let projected = topology.materialize(&tokenizer, Some(&[true, false, false]));
    let dfa = projected.tokenizer_view.dfa();

    assert_ne!(dfa.trans(dfa.start_state, b'a' as usize), u32::MAX);
    assert_eq!(dfa.trans(dfa.start_state, b'x' as usize), u32::MAX);
    assert_eq!(dfa.trans(dfa.start_state, b'y' as usize), u32::MAX);
}

#[test]
fn already_projected_token_topology_materialization_matches_reprojection() {
    let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
    let raw_start_states = (0..tokenizer.num_states() as usize).collect::<Vec<_>>();
    let tokens = [
        b"a".as_slice(),
        b"aa".as_slice(),
        b"ab".as_slice(),
        b"ba".as_slice(),
        b"bb".as_slice(),
    ];
    let active_groups = [true, false];
    let topology = build_bounded_analysis_topology_impl(
        &tokenizer,
        None,
        &raw_start_states,
        &tokens,
        false,
        true,
        false,
        false,
        Some(&active_groups),
        None,
        None,
    )
    .expect("projected topology build")
    .0;
    let direct = topology.materialize_already_projected(&tokenizer, &active_groups);
    let reprojected = topology.materialize(&tokenizer, Some(&active_groups));

    for &raw_state in &raw_start_states {
        let direct_start = direct.view_state_for_raw_start(raw_state);
        let reprojected_start = reprojected.view_state_for_raw_start(raw_state);
        for token in tokens {
            assert_eq!(
                view_trace(&direct.tokenizer_view, direct_start, token),
                view_trace(&reprojected.tokenizer_view, reprojected_start, token),
            );
        }
    }
}

#[test]
fn token_bounded_projected_build_aborts_at_actual_work_budget() {
    let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
    let raw_start_states = (0..tokenizer.num_states() as usize).collect::<Vec<_>>();
    let tokens = [
        b"a".as_slice(),
        b"aa".as_slice(),
        b"ab".as_slice(),
        b"ba".as_slice(),
        b"bb".as_slice(),
    ];
    let active_groups = [true, true];
    let tiny_budget = TokenBoundedAnalysisWorkBudget {
        max_configurations: usize::MAX,
        max_trie_visits: 1,
    };

    let aborted = try_build_token_bounded_analysis_view_projected(
        &tokenizer,
        &raw_start_states,
        &tokens,
        &active_groups,
        tiny_budget,
    );
    let work = match aborted {
        Ok(_) => panic!("tiny actual-work budget unexpectedly completed"),
        Err(work) => work,
    };
    assert_eq!(work.trie_visits, tiny_budget.max_trie_visits + 1);

    let generous_budget = TokenBoundedAnalysisWorkBudget {
        max_configurations: 1_000,
        max_trie_visits: 10_000,
    };
    let (budgeted, _) = try_build_token_bounded_analysis_view_projected(
        &tokenizer,
        &raw_start_states,
        &tokens,
        &active_groups,
        generous_budget,
    )
    .unwrap_or_else(|work| panic!("generous budget aborted unexpectedly: {work:?}"));
    let reference = build_token_bounded_analysis_view_projected(
        &tokenizer,
        &raw_start_states,
        &tokens,
        &active_groups,
    );

    for &raw_state in &raw_start_states {
        let budgeted_start = budgeted.view_state_for_raw_start(raw_state);
        let reference_start = reference.view_state_for_raw_start(raw_state);
        for token in tokens {
            assert_eq!(
                view_trace(&budgeted.tokenizer_view, budgeted_start, token),
                view_trace(&reference.tokenizer_view, reference_start, token),
            );
        }
    }
}

#[test]
fn bounded_nfa_common_first_factorization_preserves_observed_token_trajectories() {
    let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
    let raw_start_states = (0..tokenizer.num_states() as usize).collect::<Vec<_>>();
    let tokens = [b"aa".as_slice(), b"ab".as_slice(), b"aab".as_slice()];
    let factored = build_bounded_analysis_view_impl(
        &tokenizer,
        &raw_start_states,
        &tokens,
        None,
        false,
        true,
        true,
    );
    let reference = build_bounded_analysis_view_impl(
        &tokenizer,
        &raw_start_states,
        &tokens,
        None,
        false,
        false,
        true,
    );

    for &raw_state in &raw_start_states {
        let factored_start = factored.view_state_for_raw_start(raw_state);
        let reference_start = reference.view_state_for_raw_start(raw_state);
        for token in tokens {
            assert_eq!(
                view_trace(&factored.tokenizer_view, factored_start, token),
                view_trace(&reference.tokenizer_view, reference_start, token),
            );
        }
    }

    for token in tokens {
        for offset in 0..token.len() {
            assert_eq!(
                view_trace(
                    &factored.tokenizer_view,
                    factored.tokenizer_view.dfa().start_state,
                    &token[offset..],
                ),
                view_trace(
                    &reference.tokenizer_view,
                    reference.tokenizer_view.dfa().start_state,
                    &token[offset..],
                ),
            );
        }
    }
}

#[test]
fn relevant_powerset_view_preserves_bounded_token_trajectories() {
    let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
    let raw_start_states = (0..tokenizer.num_states() as usize).collect::<Vec<_>>();
    let tokens = [
        b"".as_slice(),
        b"a".as_slice(),
        b"b".as_slice(),
        b"aa".as_slice(),
        b"ab".as_slice(),
        b"ba".as_slice(),
        b"aab".as_slice(),
    ];
    let active_groups = [true, true];
    let bounded = build_bounded_analysis_view(
        &tokenizer,
        &raw_start_states,
        &tokens,
        Some(&active_groups),
    );
    let mut relevant_bytes = [false; 256];
    for token in tokens {
        for &byte in token {
            relevant_bytes[byte as usize] = true;
        }
    }
    let powerset = build_relevant_powerset_view(
        &tokenizer,
        &relevant_bytes,
        Some(&active_groups),
        None,
    );
    let raw_start_to_powerset = Arc::clone(&powerset.raw_start_to_view);
    let powerset = powerset.into_tokenizer_view();

    for &raw_state in &raw_start_states {
        let bounded_start = bounded.view_state_for_raw_start(raw_state);
        let powerset_start = raw_start_to_powerset[raw_state] as usize;
        for token in tokens {
            assert_eq!(
                view_trace(&bounded.tokenizer_view, bounded_start, token),
                view_trace(&powerset, powerset_start, token),
                "raw_state={raw_state} token={token:?}",
            );
        }
    }
}

#[test]
fn raw_relevant_powerset_projected_targets_match_scalar_exactly() {
    fn assert_same(left: &RelevantPowersetView, right: &RelevantPowersetView) {
        assert_eq!(left.start_state, right.start_state);
        assert_eq!(left.bytes, right.bytes);
        assert_eq!(left.edge_offsets, right.edge_offsets);
        assert_eq!(left.edges, right.edges);
        assert_eq!(left.raw_start_to_view, right.raw_start_to_view);
        assert_eq!(left.configurations, right.configurations);
        assert_eq!(left.states.len(), right.states.len());
        for (left, right) in left.states.iter().zip(&right.states) {
            assert_eq!(left.finalizers, right.finalizers);
            assert_eq!(
                left.possible_future_group_ids,
                right.possible_future_group_ids,
            );
        }
    }

    let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
    let mut relevant_bytes = [false; 256];
    relevant_bytes[b'a' as usize] = true;
    relevant_bytes[b'b' as usize] = true;

    for active_groups in [None, Some([true, true]), Some([true, false]), Some([false, true])] {
        let active_groups = active_groups.as_ref().map(<[_; 2]>::as_slice);
        let scalar = try_build_relevant_powerset_view(
            &tokenizer,
            &relevant_bytes,
            active_groups,
            None,
            None,
            RawPowersetTargetMode::Scalar,
        )
        .expect("unbounded scalar powerset construction");
        let direct = try_build_relevant_powerset_view(
            &tokenizer,
            &relevant_bytes,
            active_groups,
            None,
            None,
            RawPowersetTargetMode::Direct,
        )
        .expect("unbounded direct powerset construction");
        assert_same(&scalar, &direct);
    }
}

#[test]
fn powerset_restricted_bounded_view_matches_direct_bounded_view() {
    let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
    let raw_start_states = (0..tokenizer.num_states() as usize).collect::<Vec<_>>();
    let tokens = [
        b"".as_slice(),
        b"a".as_slice(),
        b"b".as_slice(),
        b"aa".as_slice(),
        b"ab".as_slice(),
        b"ba".as_slice(),
        b"aab".as_slice(),
    ];
    let active_groups = [true, true];
    let direct = build_bounded_analysis_view(
        &tokenizer,
        &raw_start_states,
        &tokens,
        Some(&active_groups),
    );
    let mut relevant_bytes = [false; 256];
    for token in tokens {
        for &byte in token {
            relevant_bytes[byte as usize] = true;
        }
    }
    let powerset = build_relevant_powerset_view(
        &tokenizer,
        &relevant_bytes,
        Some(&active_groups),
        None,
    );
    let restricted = build_bounded_analysis_view_from_relevant_powerset(
        &powerset,
        &raw_start_states,
        &tokens,
    );

    for &raw_state in &raw_start_states {
        let direct_start = direct.view_state_for_raw_start(raw_state);
        let restricted_start = restricted.view_state_for_raw_start(raw_state);
        for token in tokens {
            assert_eq!(
                view_trace(&restricted.tokenizer_view, restricted_start, token),
                view_trace(&direct.tokenizer_view, direct_start, token),
                "raw_state={raw_state} token={token:?}",
            );
        }
    }
    for token in tokens {
        for offset in 0..token.len() {
            assert_eq!(
                view_trace(
                    &restricted.tokenizer_view,
                    restricted.tokenizer_view.dfa().start_state,
                    &token[offset..],
                ),
                view_trace(
                    &direct.tokenizer_view,
                    direct.tokenizer_view.dfa().start_state,
                    &token[offset..],
                ),
                "reset suffix={:?}",
                &token[offset..],
            );
        }
    }
}

#[test]
fn powerset_restricted_bounded_view_matches_direct_across_projections() {
    fn binary_inputs(max_len: usize) -> Vec<Vec<u8>> {
        let mut inputs = vec![Vec::new()];
        for len in 1..=max_len {
            let start = inputs.len();
            for bits in 0..(1usize << len) {
                inputs.push(
                    (0..len)
                        .rev()
                        .map(|shift| if bits & (1 << shift) == 0 { b'a' } else { b'b' })
                        .collect(),
                );
            }
            debug_assert_eq!(inputs.len() - start, 1usize << len);
        }
        inputs
    }

    let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
    let raw_start_states = (0..tokenizer.num_states() as usize).collect::<Vec<_>>();
    let exhaustive = binary_inputs(4);
    let sparse_cases = vec![
        vec![Vec::new(), b"ab".to_vec(), b"baa".to_vec(), b"bbbb".to_vec()],
        vec![b"a".to_vec(), b"bb".to_vec(), b"aaba".to_vec()],
    ];
    let mut token_cases = vec![exhaustive];
    token_cases.extend(sparse_cases);

    for active_groups in [[true, true], [true, false], [false, true]] {
        for owned_tokens in &token_cases {
            let tokens = owned_tokens.iter().map(Vec::as_slice).collect::<Vec<_>>();
            let direct = build_bounded_analysis_view(
                &tokenizer,
                &raw_start_states,
                &tokens,
                Some(&active_groups),
            );
            let mut relevant_bytes = [false; 256];
            for token in &tokens {
                for &byte in *token {
                    relevant_bytes[byte as usize] = true;
                }
            }
            let powerset = build_relevant_powerset_view(
                &tokenizer,
                &relevant_bytes,
                Some(&active_groups),
                None,
            );
            let restricted = build_bounded_analysis_view_from_relevant_powerset(
                &powerset,
                &raw_start_states,
                &tokens,
            );

            for &raw_state in &raw_start_states {
                let direct_start = direct.view_state_for_raw_start(raw_state);
                let restricted_start = restricted.view_state_for_raw_start(raw_state);
                for token in &tokens {
                    assert_eq!(
                        view_trace(&restricted.tokenizer_view, restricted_start, token),
                        view_trace(&direct.tokenizer_view, direct_start, token),
                        "active_groups={active_groups:?} raw_state={raw_state} token={token:?}",
                    );
                }
            }
            for token in &tokens {
                for offset in 0..token.len() {
                    assert_eq!(
                        view_trace(
                            &restricted.tokenizer_view,
                            restricted.tokenizer_view.dfa().start_state,
                            &token[offset..],
                        ),
                        view_trace(
                            &direct.tokenizer_view,
                            direct.tokenizer_view.dfa().start_state,
                            &token[offset..],
                        ),
                        "active_groups={active_groups:?} reset suffix={:?}",
                        &token[offset..],
                    );
                }
            }
        }
    }
}

#[test]
fn stable_restricted_observation_quotient_preserves_relevant_powerset_traces() {
    let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
    let active_groups = [true, true];
    let mut relevant_bytes = [false; 256];
    relevant_bytes[b'a' as usize] = true;
    relevant_bytes[b'b' as usize] = true;
    let stable_map = compute_state_map(
        &tokenizer,
        &relevant_bytes,
        Some(&active_groups),
        None,
        RefinementDepth::Stable,
    );
    let raw_powerset = build_relevant_powerset_view(
        &tokenizer,
        &relevant_bytes,
        Some(&active_groups),
        None,
    );
    let quotient_powerset = build_relevant_powerset_view(
        &tokenizer,
        &relevant_bytes,
        Some(&active_groups),
        Some(&stable_map),
    );
    let raw_start_states = (0..tokenizer.num_states() as usize).collect::<Vec<_>>();
    let owned_tokens = [
        Vec::new(),
        b"a".to_vec(),
        b"b".to_vec(),
        b"ab".to_vec(),
        b"ba".to_vec(),
        b"aab".to_vec(),
        b"bba".to_vec(),
        b"abba".to_vec(),
    ];
    let tokens = owned_tokens.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let raw_bounded = build_bounded_analysis_view_from_relevant_powerset(
        &raw_powerset,
        &raw_start_states,
        &tokens,
    );
    let quotient_bounded = build_bounded_analysis_view_from_relevant_powerset(
        &quotient_powerset,
        &raw_start_states,
        &tokens,
    );

    for &raw_state in &raw_start_states {
        let raw_start = raw_bounded.view_state_for_raw_start(raw_state);
        let quotient_start = quotient_bounded.view_state_for_raw_start(raw_state);
        for token in &tokens {
            assert_eq!(
                view_trace(&raw_bounded.tokenizer_view, raw_start, token),
                view_trace(&quotient_bounded.tokenizer_view, quotient_start, token),
                "raw_state={raw_state} token={token:?}"
            );
        }
    }
    for token in &tokens {
        for offset in 0..token.len() {
            assert_eq!(
                view_trace(
                    &raw_bounded.tokenizer_view,
                    raw_bounded.tokenizer_view.dfa().start_state,
                    &token[offset..],
                ),
                view_trace(
                    &quotient_bounded.tokenizer_view,
                    quotient_bounded.tokenizer_view.dfa().start_state,
                    &token[offset..],
                ),
                "reset suffix={:?}",
                &token[offset..],
            );
        }
    }
}

#[test]
fn relevant_powerset_handles_fully_filtered_empty_configurations() {
    let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
    let relevant = [true; 256];
    let inactive_groups = [false, false];
    let view = build_relevant_powerset_view(
        &tokenizer,
        &relevant,
        Some(&inactive_groups),
        None,
    );

    assert!(view.configurations.iter().all(|config| config.is_empty()));
    assert!(view.edges.is_empty());
    assert!(view.raw_start_to_view.iter().all(|&state| state == view.start_state as u32));
}

#[test]
fn relevant_powerset_budget_aborts_without_changing_successful_construction() {
    let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
    let relevant = [true; 256];
    let reference = build_relevant_powerset_view(&tokenizer, &relevant, None, None);

    let generous = build_relevant_powerset_view_budgeted(
        &tokenizer,
        &relevant,
        None,
        None,
        RelevantPowersetWorkBudget {
            max_configurations: reference.configurations.len(),
            max_edges: reference.edges.len(),
        },
    )
    .expect("budget equal to realized exact work must succeed");
    assert_eq!(generous.configurations.as_ref(), reference.configurations.as_ref());
    assert_eq!(generous.edge_offsets, reference.edge_offsets);
    assert_eq!(generous.edges, reference.edges);

    let too_small = build_relevant_powerset_view_budgeted(
        &tokenizer,
        &relevant,
        None,
        None,
        RelevantPowersetWorkBudget {
            max_configurations: reference.configurations.len(),
            max_edges: reference.edges.len().saturating_sub(1),
        },
    );
    let Err(work) = too_small else {
        panic!("undersized edge budget must abort powerset construction");
    };
    assert!(work.edges > reference.edges.len().saturating_sub(1));
}

#[test]
fn set_valued_refinement_distinguishes_epsilon_successor_class_sets() {
    let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
    let mut relevant = [false; 256];
    relevant[b'a' as usize] = true;
    relevant[b'b' as usize] = true;
    let map = compute_state_map(
        &tokenizer,
        &relevant,
        None,
        None,
        RefinementDepth::Stable,
    );
    assert_ne!(map.original_to_internal[2], map.original_to_internal[4]);
    assert_ne!(map.original_to_internal[1], map.original_to_internal[2]);
}

#[test]
fn identity_input_map_does_not_block_nfa_equivalence_merges() {
    let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
    let relevant = [true; 256];
    let identity = super::super::identity_state_map(tokenizer.num_states() as usize);
    let direct = compute_state_map(
        &tokenizer,
        &relevant,
        None,
        None,
        RefinementDepth::Stable,
    );
    let from_identity = compute_state_map(
        &tokenizer,
        &relevant,
        None,
        Some(&identity),
        RefinementDepth::Stable,
    );

    assert!(direct.num_internal_ids() < tokenizer.num_states());
    assert!(same_partition(
        &direct.original_to_internal,
        &from_identity.original_to_internal,
    ));
}

#[test]
fn direct_nfa_worklist_matches_synchronous_refinement() {
    let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
    let mut relevant_ab = [false; 256];
    relevant_ab[b'a' as usize] = true;
    relevant_ab[b'b' as usize] = true;
    let relevant_all = [true; 256];
    let active_left = [true, false];
    let active_right = [false, true];
    let active_both = [true, true];

    for relevant in [&relevant_ab, &relevant_all] {
        for active_groups in [
            None,
            Some(active_left.as_slice()),
            Some(active_right.as_slice()),
            Some(active_both.as_slice()),
        ] {
            let synchronous = compute_state_map(
                &tokenizer,
                relevant,
                active_groups,
                None,
                RefinementDepth::Bounded(tokenizer.num_states() as usize),
            );
            let worklist = compute_state_map(
                &tokenizer,
                relevant,
                active_groups,
                None,
                RefinementDepth::Stable,
            );
            assert!(same_partition(
                &synchronous.original_to_internal,
                &worklist.original_to_internal,
            ));

            let seed = compute_state_map(
                &tokenizer,
                relevant,
                active_groups,
                None,
                RefinementDepth::Bounded(1),
            );
            let seeded_synchronous = compute_state_map(
                &tokenizer,
                relevant,
                active_groups,
                Some(&seed),
                RefinementDepth::Bounded(tokenizer.num_states() as usize),
            );
            let seeded_worklist = compute_state_map(
                &tokenizer,
                relevant,
                active_groups,
                Some(&seed),
                RefinementDepth::Stable,
            );
            assert!(same_partition(
                &seeded_synchronous.original_to_internal,
                &seeded_worklist.original_to_internal,
            ));
        }
    }
}

#[test]
fn prebuilt_sparse_powerset_refinement_matches_fresh_nfa_refinement() {
    let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
    let mut relevant = [false; 256];
    relevant[b'a' as usize] = true;
    relevant[b'b' as usize] = true;
    let direct = compute_state_map(
        &tokenizer,
        &relevant,
        None,
        None,
        RefinementDepth::Stable,
    );
    let view = build_relevant_powerset_view(&tokenizer, &relevant, None, None);
    let mut output_ids = FxHashMap::<Vec<u64>, u32>::default();
    let output_class_by_config = view
        .configurations
        .iter()
        .map(|config| {
            let observation = observation_words(&tokenizer, config, None);
            let next = output_ids.len() as u32;
            *output_ids.entry(observation).or_insert(next)
        })
        .collect::<Vec<_>>();
    let reused = compute_state_map_from_prebuilt_sparse_powerset(
        &tokenizer,
        None,
        RefinementDepth::Stable,
        &view.raw_start_to_view,
        &view.configurations,
        &output_class_by_config,
        None,
        &view.edge_offsets,
        &view.edges,
    );
    let synchronous = compute_state_map_from_prebuilt_sparse_powerset(
        &tokenizer,
        None,
        RefinementDepth::Bounded(tokenizer.num_states() as usize),
        &view.raw_start_to_view,
        &view.configurations,
        &output_class_by_config,
        None,
        &view.edge_offsets,
        &view.edges,
    );

    assert!(same_partition(
        &direct.original_to_internal,
        &reused.original_to_internal,
    ));
    assert!(same_partition(
        &synchronous.original_to_internal,
        &reused.original_to_internal,
    ));
}

#[test]
fn prebuilt_sparse_worklist_matches_projected_synchronous_refinement() {
    let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
    let mut relevant = [false; 256];
    relevant[b'a' as usize] = true;
    relevant[b'b' as usize] = true;
    let active_groups = [true, false];
    let view = build_relevant_powerset_view(
        &tokenizer,
        &relevant,
        Some(&active_groups),
        None,
    );
    let output_class_by_config = powerset_output_class_ids(&view);
    let active_language = raw_active_language_states(&tokenizer, Some(&active_groups))
        .expect("active group projection");

    let seed = compute_state_map_from_prebuilt_sparse_powerset(
        &tokenizer,
        None,
        RefinementDepth::Bounded(1),
        &view.raw_start_to_view,
        &view.configurations,
        &output_class_by_config,
        Some(&active_language),
        &view.edge_offsets,
        &view.edges,
    );
    let worklist = compute_state_map_from_prebuilt_sparse_powerset(
        &tokenizer,
        Some(&seed),
        RefinementDepth::Stable,
        &view.raw_start_to_view,
        &view.configurations,
        &output_class_by_config,
        Some(&active_language),
        &view.edge_offsets,
        &view.edges,
    );
    let synchronous = compute_state_map_from_prebuilt_sparse_powerset(
        &tokenizer,
        Some(&seed),
        RefinementDepth::Bounded(tokenizer.num_states() as usize),
        &view.raw_start_to_view,
        &view.configurations,
        &output_class_by_config,
        Some(&active_language),
        &view.edge_offsets,
        &view.edges,
    );

    assert!(same_partition(
        &synchronous.original_to_internal,
        &worklist.original_to_internal,
    ));
}

#[test]
fn prebuilt_sparse_worklist_matches_synchronous_on_random_topologies() {
    fn next_u32(state: &mut u64) -> u32 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (*state >> 32) as u32
    }

    let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
    let num_states = tokenizer.num_states() as usize;
    for seed in 0..128u64 {
        let mut rng = seed ^ 0x9e37_79b9_7f4a_7c15;
        let extra_configs = num_states * 3 + 1;
        let mut configurations = (0..num_states)
            .map(|state| vec![state as u32].into_boxed_slice())
            .collect::<Vec<_>>();
        for _ in 0..extra_configs {
            let mut config = Vec::<u32>::new();
            for raw in 0..num_states {
                if next_u32(&mut rng).is_multiple_of(3) {
                    config.push(raw as u32);
                }
            }
            if config.is_empty() && next_u32(&mut rng) & 1 != 0 {
                config.push((next_u32(&mut rng) as usize % num_states) as u32);
            }
            configurations.push(config.into_boxed_slice());
        }

        let raw_start_to_view = (0..num_states as u32).collect::<Vec<_>>();
        let output_class_by_config = (0..configurations.len())
            .map(|_| next_u32(&mut rng) % 5)
            .collect::<Vec<_>>();
        let mut edge_offsets = Vec::<u32>::with_capacity(configurations.len() + 1);
        let mut edges = Vec::<(u8, u32)>::new();
        edge_offsets.push(0);
        for _ in 0..configurations.len() {
            for byte in 0..5u8 {
                if next_u32(&mut rng) & 1 != 0 {
                    edges.push((
                        byte,
                        next_u32(&mut rng) % configurations.len() as u32,
                    ));
                }
            }
            edge_offsets.push(edges.len() as u32);
        }
        let mut active_language = (0..num_states)
            .map(|_| next_u32(&mut rng) & 1 != 0)
            .collect::<Vec<_>>();
        if !active_language.iter().any(|&active| active) {
            active_language[next_u32(&mut rng) as usize % num_states] = true;
        }

        for active_language in [None, Some(active_language.as_slice())] {
            let synchronous = compute_state_map_from_prebuilt_sparse_powerset(
                &tokenizer,
                None,
                RefinementDepth::Bounded(num_states),
                &raw_start_to_view,
                &configurations,
                &output_class_by_config,
                active_language,
                &edge_offsets,
                &edges,
            );
            let worklist = compute_state_map_from_prebuilt_sparse_powerset(
                &tokenizer,
                None,
                RefinementDepth::Stable,
                &raw_start_to_view,
                &configurations,
                &output_class_by_config,
                active_language,
                &edge_offsets,
                &edges,
            );
            assert!(
                same_partition(
                    &synchronous.original_to_internal,
                    &worklist.original_to_internal,
                ),
                "unseeded mismatch at seed={seed} active_projection={}",
                active_language.is_some(),
            );

            let initial = compute_state_map_from_prebuilt_sparse_powerset(
                &tokenizer,
                None,
                RefinementDepth::Bounded(1),
                &raw_start_to_view,
                &configurations,
                &output_class_by_config,
                active_language,
                &edge_offsets,
                &edges,
            );
            let seeded_synchronous = compute_state_map_from_prebuilt_sparse_powerset(
                &tokenizer,
                Some(&initial),
                RefinementDepth::Bounded(num_states),
                &raw_start_to_view,
                &configurations,
                &output_class_by_config,
                active_language,
                &edge_offsets,
                &edges,
            );
            let seeded_worklist = compute_state_map_from_prebuilt_sparse_powerset(
                &tokenizer,
                Some(&initial),
                RefinementDepth::Stable,
                &raw_start_to_view,
                &configurations,
                &output_class_by_config,
                active_language,
                &edge_offsets,
                &edges,
            );
            assert!(
                same_partition(
                    &seeded_synchronous.original_to_internal,
                    &seeded_worklist.original_to_internal,
                ),
                "seeded mismatch at seed={seed} active_projection={}",
                active_language.is_some(),
            );
        }
    }
}

#[test]
fn bounded_analysis_view_materializes_reset_suffix_transitions_from_start_state() {
    let tokenizer =
        crate::automata::lexer::tokenizer::arbitrary_epsilon_l1_test_tokenizer();
    let raw_states = (0..tokenizer.num_states() as usize).collect::<Vec<_>>();
    let tokens: Vec<&[u8]> = vec![b"xa", b"xaa", b"xb"];
    let bounded = build_bounded_analysis_view_with_trie(
        &tokenizer,
        &raw_states,
        &tokens,
        None,
        None,
    );
    let dfa = bounded.tokenizer_view.dfa();
    let start_state = dfa.start_state;
    let trans_a = dfa.trans(start_state, b'a' as usize);
    assert_ne!(
        trans_a,
        u32::MAX,
        "bounded view must materialize suffix transitions from start_state for byte 'a'",
    );
}
