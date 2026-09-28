use super::*;
use crate::automata::lexer::ast::{bytes, plus};
use crate::automata::lexer::compile::build_regex;

fn tokenizer_from_exprs(exprs: Vec<Expr>) -> Tokenizer {
    let num_terminals = exprs.len() as u32;
    build_regex(&exprs).into_tokenizer(
        num_terminals,
        Some(Arc::from(exprs.into_boxed_slice())),
    )
}


#[test]
fn global_scalar_proof_matches_local_scan_on_generated_graphs() {
    // Includes loops, absent bytes, shared targets and reset-unreachable
    // states. Every root is compared, not just the tokenizer's start.
    for states in [1usize, 2, 7, 31] {
        for seed in 0..12u64 {
            let mut bits = seed + 1;
            let mut dfa = DFA::new(states);
            dfa.ensure_group_capacity(2);
            for source in 0..states as u32 {
                for byte in [0u8, b'a', b'b', b'"', 128, 255] {
                    bits = bits.wrapping_mul(6364136223846793005).wrapping_add(1);
                    if bits & 3 != 0 {
                        dfa.add_transition(source, byte, ((bits >> 32) % states as u64) as u32);
                    }
                }
            }
            let tokenizer = Tokenizer::from_parts(dfa, 2, None);
            assert!(!tokenizer.has_epsilon_transitions());
            assert!(!tokenizer.has_any_virtual_runtime());
            for root in 0..states as u32 {
                assert!(tokenizer.scalar_physical_component_root_impl::<true>(root));
                assert_eq!(tokenizer.scalar_physical_component_root_impl::<true>(root),
                    tokenizer.scalar_physical_component_root_impl::<false>(root));
            }
            for root in [states as u32, u32::MAX] {
                assert!(!tokenizer.scalar_physical_component_root_impl::<true>(root));
                assert!(!tokenizer.scalar_physical_component_root_impl::<false>(root));
            }
        }
    }
}

#[test]
fn global_scalar_proof_preserves_mixed_component_refusals() {
    let mut dfa = DFA::new(5);
    dfa.ensure_group_capacity(1);
    dfa.add_transition(0, b'a', 1);
    dfa.add_transition(1, b'b', 2);
    dfa.add_epsilon_transition(2, 3);
    // Root 4 remains scalar even though unrelated roots are not.
    dfa.add_transition(4, b'z', 4);
    let tokenizer = Tokenizer::from_parts(dfa, 1, None);
    assert!(tokenizer.has_epsilon_transitions());
    for root in 0..5 {
        assert_eq!(tokenizer.scalar_physical_component_root_impl::<true>(root),
            tokenizer.scalar_physical_component_root_impl::<false>(root));
    }
    assert!(!tokenizer.scalar_physical_component_root_impl::<true>(0));
    assert!(tokenizer.scalar_physical_component_root_impl::<true>(4));

    let mut tokenizer = tokenizer_from_exprs(vec![
        Expr::U8Seq(b"b".to_vec()), Expr::U8Class(U8Set::empty()),
    ]);
    tokenizer.isolate_start_state_and_drain_nullable_terminals();
    tokenizer.install_virtual_zero_min_unit_repeat_component(
        U8Set::single(b'a'), 1_000_000_000, 1,
    ).unwrap();
    assert!(tokenizer.has_any_virtual_runtime());
    for root in 0..tokenizer.num_states() {
        assert_eq!(tokenizer.scalar_physical_component_root_impl::<true>(root),
            tokenizer.scalar_physical_component_root_impl::<false>(root));
    }
}

#[test]
fn bounded_eager_candidates_preserve_exact_rows_and_single_worker_execution() {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    pool.install(|| {
        let t = Tokenizer::from_parts(one_byte_component(b'a'), 1, None);
        let original = t.build_terminal_projected_quotients_for_containment_candidates(&[0]);
        let bounded = t.build_terminal_projected_quotients_for_containment_candidates_bounded(&[0, 0, 7], 2, 1);
        assert_eq!(original.len(), 1);
        assert_eq!(bounded.len(), 1);
        assert_eq!(bincode::serialize(&original).unwrap(), bincode::serialize(&bounded).unwrap());
        assert!(t.build_terminal_projected_quotients_for_containment_candidates_bounded(&[0], 1, 1).is_empty());
        assert!(t.build_terminal_projected_quotients_for_containment_candidates_bounded(&[0], 2, 0).is_empty());
    });
}

#[test]
fn coldcap_component_limits_are_exact_and_cycles_are_deduplicated() {
    let t = Tokenizer::from_parts(one_byte_component(b'a'), 1, None);
    let full = t.scalar_physical_component_states(0).unwrap();
    assert_eq!(full, vec![0, 1]);
    assert!(t.scalar_physical_component_states_bounded(0, 0, 10).is_none());
    assert!(t.scalar_physical_component_states_bounded(0, 1, 10).is_none());
    assert!(t.scalar_physical_component_states_bounded(0, 2, 0).is_none());
    assert_eq!(t.scalar_physical_component_states_bounded(0, 2, 1), Some(full));
    let looped = tokenizer_from_exprs(vec![plus(bytes(&[b'a', b'b', b'c']))]);
    let root = looped.start_state();
    let states = looped.scalar_physical_component_states(root).unwrap();
    let edges = states.iter().map(|&s| looped.transitions_from(s).count()).sum();
    assert_eq!(looped.scalar_physical_component_states_bounded(root, states.len(), edges), Some(states));
}

#[test]
fn coldcap_accepted_quotient_is_identical_to_unbounded_constructor() {
    let t = Tokenizer::from_parts(one_byte_component(b'a'), 1, None);
    let old = t.build_terminal_projected_quotients_for_containment_candidates(&[0]);
    let new = t.build_terminal_projected_quotient_for_containment_bounded(0, 2, 1).unwrap();
    assert_eq!(old.len(), 1);
    assert_eq!(new.full_states, old[0].1.full_states);
    assert_eq!(new.projected_states, old[0].1.projected_states);
    assert_eq!(new.byte_to_class, old[0].1.byte_to_class);
    assert_eq!(new.class_representatives, old[0].1.class_representatives);
    assert_eq!(new.class_targets, old[0].1.class_targets);
    assert!(t.build_terminal_projected_quotient_for_containment_bounded(0, 1, 1).is_none());
    assert!(t.build_terminal_projected_quotient_for_containment_bounded(7, 2, 1).is_none());
}

#[test]
fn coldcap_rejects_epsilon_component_without_partial_certificate() {
    let mut dfa = one_byte_component(b'a');
    dfa.add_epsilon_transition(0, 1);
    let t = Tokenizer::from_parts(dfa, 1, None);
    assert!(t.scalar_physical_component_states_bounded(0, 100, 1000).is_none());
}

fn one_byte_component(byte: u8) -> DFA {
    let mut dfa = DFA::new(2);
    dfa.ensure_group_capacity(1);
    dfa.set_group_u8set(0, U8Set::single(byte));
    dfa.add_transition(0, byte, 1);
    let mut finalizers = BitSet::new(1);
    finalizers.set(0);
    dfa.overwrite_state_metadata(1, finalizers, BitSet::new(1));
    dfa.recompute_possible_futures();
    dfa
}

#[test]
fn direct_mask_component_isolated_start_future_update_matches_full_recompute() {
    let mut host = DFA::new(2);
    host.ensure_group_capacity(2);
    host.set_group_u8set(0, U8Set::single(b'a'));
    host.set_group_u8set(1, U8Set::single(b'b'));
    host.add_epsilon_transition(0, 1);
    host.add_transition(1, b'a', 1);
    let mut finalizers = BitSet::new(2);
    finalizers.set(0);
    host.overwrite_state_metadata(1, finalizers, BitSet::new(2));
    host.recompute_possible_futures();
    let mut tokenizer = Tokenizer::from_parts(host, 2, None);

    tokenizer
        .install_direct_mask_components(vec![(one_byte_component(b'b'), 0, 1)])
        .expect("isolated-start direct component must install");

    let mut reference = tokenizer.dfa.clone();
    reference.recompute_possible_futures();
    assert_eq!(tokenizer.dfa.num_states(), reference.num_states());
    for state in 0..tokenizer.dfa.num_states() as u32 {
        assert_eq!(
            tokenizer.dfa.possible_future_group_ids(state),
            reference.possible_future_group_ids(state),
            "future mismatch at state {state}",
        );
    }
}

#[test]
fn direct_mask_component_nonisolated_start_falls_back_to_full_recompute() {
    let mut host = DFA::new(2);
    host.ensure_group_capacity(2);
    host.set_group_u8set(0, U8Set::single(b'a'));
    host.set_group_u8set(1, U8Set::single(b'b'));
    host.add_transition(1, b'a', 0);
    host.recompute_possible_futures();
    let mut tokenizer = Tokenizer::from_parts(host, 2, None);

    tokenizer
        .install_direct_mask_components(vec![(one_byte_component(b'b'), 0, 1)])
        .expect("nonisolated-start direct component must install via fallback");

    let mut reference = tokenizer.dfa.clone();
    reference.recompute_possible_futures();
    for state in 0..tokenizer.dfa.num_states() as u32 {
        assert_eq!(
            tokenizer.dfa.possible_future_group_ids(state),
            reference.possible_future_group_ids(state),
            "future mismatch at state {state}",
        );
    }
}

#[test]
fn batched_terminal_residual_replacement_preserves_appended_state_order() {
    let base_dfas = (0..3)
        .map(|_| Arc::new(DFA::new(2)))
        .collect::<Vec<_>>();
    let coordinates = TerminalResidualCoordinates::from_rows_and_dfas(
        vec![
            vec![(0, 0), (1, 0), (2, 0)],
            vec![(0, 1), (1, 1)],
            vec![(2, 1)],
        ],
        base_dfas,
    );
    let replacement_one = Arc::new(DFA::new(3));
    let replacement_two = Arc::new(DFA::new(2));
    let replaced = coordinates
        .replace_terminals_with_appended_dfas(&[
            (1, Arc::clone(&replacement_one)),
            (2, Arc::clone(&replacement_two)),
        ])
        .expect("valid batched replacement");

    let expected = [
        vec![(0, 0)],
        vec![(0, 1)],
        vec![],
        vec![(1, 0)],
        vec![(1, 1)],
        vec![(1, 2)],
        vec![(2, 0)],
        vec![(2, 1)],
    ];
    assert_eq!(replaced.len(), expected.len());
    for (state, expected_row) in expected.iter().enumerate() {
        assert_eq!(replaced.row(state as u32), Some(expected_row.as_slice()));
    }
    assert_eq!(replaced.terminal_dfa(1).unwrap().num_states(), 3);
    assert_eq!(replaced.terminal_dfa(2).unwrap().num_states(), 2);
}

#[test]
fn batched_terminal_residual_replacement_matches_sequential_duplicate_semantics() {
    let coordinates = TerminalResidualCoordinates::from_rows_and_dfas(
        vec![vec![(0, 0), (1, 7)]],
        vec![Arc::new(DFA::new(1)), Arc::new(DFA::new(8))],
    );
    let replaced = coordinates
        .replace_terminals_with_appended_dfas(&[
            (1, Arc::new(DFA::new(2))),
            (1, Arc::new(DFA::new(3))),
        ])
        .expect("duplicate replacements remain well defined");

    let expected = [
        vec![(0, 0)],
        vec![],
        vec![],
        vec![(1, 0)],
        vec![(1, 1)],
        vec![(1, 2)],
    ];
    assert_eq!(replaced.len(), expected.len());
    for (state, expected_row) in expected.iter().enumerate() {
        assert_eq!(replaced.row(state as u32), Some(expected_row.as_slice()));
    }
    assert_eq!(replaced.terminal_dfa(1).unwrap().num_states(), 3);
}

#[test]
fn grouped_terminal_residual_replacement_shares_physical_rows() {
    let base = Arc::new(DFA::new(1));
    let coordinates = TerminalResidualCoordinates::from_rows_and_dfas(
        vec![vec![(0, 0), (1, 0), (2, 0)]],
        vec![Arc::clone(&base), Arc::clone(&base), base],
    );
    let shared = Arc::new(DFA::new(2));
    let replaced = coordinates
        .replace_terminal_alias_groups_with_appended_dfas(&[(
            vec![1, 2],
            Arc::clone(&shared),
        )])
        .expect("valid grouped alias replacement");

    assert_eq!(replaced.len(), 3);
    assert_eq!(replaced.row(0), Some(&[(0, 0)][..]));
    assert_eq!(replaced.row(1), Some(&[(1, 0), (2, 0)][..]));
    assert_eq!(replaced.row(2), Some(&[(1, 1), (2, 1)][..]));
    assert!(Arc::ptr_eq(&replaced.terminal_dfas[1], &shared));
    assert!(Arc::ptr_eq(&replaced.terminal_dfas[2], &shared));
}

fn direct_mask_future_test_tokenizer(start_has_incoming: bool) -> Tokenizer {
    let mut dfa = DFA::new(2);
    dfa.ensure_group_capacity(2);
    let mut a = U8Set::empty();
    a.insert(b'a');
    dfa.set_group_u8set(0, a);
    let mut x = U8Set::empty();
    x.insert(b'x');
    dfa.set_group_u8set(1, x);
    dfa.add_transition(0, b'a', 1);
    if start_has_incoming {
        dfa.add_transition(1, b'b', 0);
    }
    let mut finalizers = BitSet::new(2);
    finalizers.set(0);
    dfa.overwrite_state_metadata(1, finalizers, BitSet::new(2));
    dfa.recompute_possible_futures();
    Tokenizer::from_parts(dfa, 2, None)
}

fn direct_mask_future_test_component() -> (DFA, u32, TerminalID) {
    let mut dfa = DFA::new(2);
    dfa.ensure_group_capacity(1);
    let mut x = U8Set::empty();
    x.insert(b'x');
    dfa.set_group_u8set(0, x);
    dfa.add_transition(0, b'x', 1);
    let mut finalizers = BitSet::new(1);
    finalizers.set(0);
    dfa.overwrite_state_metadata(1, finalizers, BitSet::new(1));
    dfa.recompute_possible_futures();
    (dfa, 0, 1)
}

fn assert_same_future_metadata(left: &Tokenizer, right: &Tokenizer) {
    assert_eq!(left.dfa.num_states(), right.dfa.num_states());
    for state in 0..left.dfa.num_states() as u32 {
        assert_eq!(left.dfa.finalizers(state), right.dfa.finalizers(state));
        assert_eq!(
            left.dfa.possible_future_group_ids(state),
            right.dfa.possible_future_group_ids(state),
            "future metadata differs at state {state}",
        );
    }
}

#[test]
fn direct_mask_incremental_start_futures_match_full_recompute() {
    let base = direct_mask_future_test_tokenizer(false);
    let mut incremental = base.clone();
    let mut full = base;
    incremental
        .install_direct_mask_components_impl(vec![direct_mask_future_test_component()], false)
        .unwrap();
    full.install_direct_mask_components_impl(vec![direct_mask_future_test_component()], true)
        .unwrap();
    assert_same_future_metadata(&incremental, &full);
    assert!(incremental.dfa.possible_future_group_ids(0).contains(1));
}

#[test]
fn direct_mask_incremental_futures_fall_back_when_start_has_incoming_edge() {
    let base = direct_mask_future_test_tokenizer(true);
    let mut guarded = base.clone();
    let mut full = base;
    guarded
        .install_direct_mask_components_impl(vec![direct_mask_future_test_component()], false)
        .unwrap();
    full.install_direct_mask_components_impl(vec![direct_mask_future_test_component()], true)
        .unwrap();
    assert_same_future_metadata(&guarded, &full);
    assert!(
        guarded.dfa.possible_future_group_ids(1).contains(1),
        "predecessor of start must see the newly installed terminal future",
    );
}

#[test]
fn direct_mask_exact_aliases_share_physical_component_without_changing_terminal_outcomes() {
    let mut base_dfa = DFA::new(1);
    base_dfa.ensure_group_capacity(3);
    let mut base = Tokenizer::from_parts(base_dfa, 3, None);
    let dummy = Arc::new(DFA::new(1));
    base.terminal_residual_coordinates = Some(Arc::new(
        TerminalResidualCoordinates::from_rows_and_dfas(
            vec![vec![(0, 0), (1, 0), (2, 0)]],
            vec![Arc::clone(&dummy), Arc::clone(&dummy), dummy],
        ),
    ));

    let component = Arc::new(direct_mask_future_test_component().0);
    let mut separate = base.clone();
    separate
        .install_direct_mask_components_shared(vec![
            (Arc::clone(&component), 0, 1),
            (Arc::clone(&component), 0, 2),
        ])
        .unwrap();
    let mut grouped = base;
    grouped
        .install_direct_mask_component_alias_groups(vec![(
            Arc::clone(&component),
            0,
            vec![1, 2],
        )])
        .unwrap();

    assert_eq!(
        separate.dfa.num_states(),
        grouped.dfa.num_states() + component.num_states(),
    );
    let separate_closure = separate.dfa.epsilon_closure(&[0]);
    let grouped_closure = grouped.dfa.epsilon_closure(&[0]);
    let separate_after_x = separate.dfa.step_all(&separate_closure, b'x');
    let grouped_after_x = grouped.dfa.step_all(&grouped_closure, b'x');
    let mut separate_finalizers = BitSet::new(3);
    for state in separate_after_x {
        separate_finalizers.union_with(separate.dfa.finalizers(state));
    }
    let mut grouped_finalizers = BitSet::new(3);
    for state in grouped_after_x {
        grouped_finalizers.union_with(grouped.dfa.finalizers(state));
    }
    assert_eq!(separate_finalizers, grouped_finalizers);
    assert!(grouped_finalizers.contains(1));
    assert!(grouped_finalizers.contains(2));
    assert!(grouped.dfa.possible_future_group_ids(0).contains(1));
    assert!(grouped.dfa.possible_future_group_ids(0).contains(2));

    let coordinates = grouped
        .terminal_residual_coordinates
        .as_ref()
        .expect("grouped install retains residual coordinates");
    assert_eq!(coordinates.len(), grouped.dfa.num_states());
    assert_eq!(coordinates.row(1), Some(&[(1, 0), (2, 0)][..]));
    assert_eq!(coordinates.row(2), Some(&[(1, 1), (2, 1)][..]));

    let (coordinate_partition, _, _) = grouped
        .terminal_residual_coordinate_observation_partition(1)
        .expect("retained residual coordinates should yield an observation partition");
    let (exact_partition, _, _) = grouped
        .exact_terminal_observation_partition(1, 1_024, 100_000)
        .expect("small deterministic fixture should admit the exact partition");
    let mut exact_by_coordinate = FxHashMap::<u32, u32>::default();
    for (&coordinate_class, &exact_class) in
        coordinate_partition.iter().zip(exact_partition.iter())
    {
        assert_eq!(coordinate_class == 0, exact_class == 0);
        if coordinate_class != 0 {
            if let Some(previous) = exact_by_coordinate.insert(coordinate_class, exact_class) {
                assert_eq!(previous, exact_class);
            }
        }
    }
}

#[test]
fn terminal_expr_projected_quotient_proves_exact_text_closure() {
    let tokenizer = tokenizer_from_exprs(vec![plus(bytes(b"a")), plus(bytes(b"b"))]);
    let source = tokenizer
        .terminal_scalar_dispatch_root(0)
        .expect("deterministic test tokenizer has a scalar terminal root");
    let quotient = tokenizer
        .terminal_expr_projected_quotient_from_root(0)
        .expect("retained terminal expression should admit an exact projected quotient");

    assert!(quotient.contains_source(source));
    assert!(quotient.byte_class_count() < 256);
    quotient
        .validate_exact_projection(&tokenizer, 0)
        .expect("freshly built projected quotient must validate against its tokenizer");

    let mut malformed = quotient.clone();
    malformed.byte_to_class[0] ^= 1;
    assert!(
        malformed.validate_exact_projection(&tokenizer, 0).is_err(),
        "wire validation must reject corrupted byte-class metadata"
    );

    let mut only_a = U8Set::empty();
    only_a.insert(b'a');
    assert_eq!(
        quotient.text_liveness_closed_bounded(source, only_a, 128, 4_096, false),
        Some(true),
    );

    let mut a_or_b = only_a;
    a_or_b.insert(b'b');
    assert_eq!(
        quotient.text_liveness_closed_bounded(source, a_or_b, 128, 4_096, false),
        Some(false),
        "a byte outside the projected terminal language must fail closed",
    );
}

fn serialized_roundtrip(tokenizer: &Tokenizer) -> Tokenizer {
    let bytes = bincode::serialize(tokenizer).unwrap();
    let loaded: Tokenizer = bincode::deserialize(&bytes).unwrap();
    assert!(loaded.exprs.is_none(), "Expr sidecars are intentionally not serialized");
    loaded
}

#[test]
fn segment_wire_roundtrips_residual_and_compressed_transitions() {
    let mut dfa = DFA::new(4);
    dfa.ensure_group_capacity(1);
    let mut group = U8Set::empty();
    for byte in [b'a', b'b', b'x'] {
        group.insert(byte);
    }
    dfa.set_group_u8set(0, group);
    dfa.add_transition(0, b'x', 1);
    // Covered rows deliberately remain present in the compile-time DFA;
    // TKS2 must omit them and recover their behavior from the segment.
    dfa.add_transition(1, b'a', 2);
    dfa.add_transition(2, b'b', 3);
    dfa.add_epsilon_transition(0, 1);
    for state in 0..4u32 {
        let mut futures = BitSet::new(1);
        futures.set(0);
        let mut finalizers = BitSet::new(1);
        if state == 3 {
            finalizers.set(0);
        }
        dfa.overwrite_state_metadata(state, finalizers, futures);
    }
    let mut byte_to_class = vec![u8::MAX; 256];
    byte_to_class[b'a' as usize] = 0;
    byte_to_class[b'b' as usize] = 1;
    let segment = CompressedTransitionSegment {
        state_offset: 1,
        state_count: 3,
        byte_to_class: Arc::from(byte_to_class.into_boxed_slice()),
        class_members: Arc::from(
            vec![vec![b'a'].into_boxed_slice(), vec![b'b'].into_boxed_slice()]
                .into_boxed_slice(),
        ),
        row_offsets: Arc::from(vec![0u32, 1, 2, 2].into_boxed_slice()),
        entries: CompressedTransitionEntries::from_parts(vec![0, 1], vec![1, 2]),
        expanded_transition_count: 2,
    };
    let original = Tokenizer::from_parts_with_compressed_transitions(
        dfa,
        1,
        None,
        vec![segment],
    );
    let wire = artifact_serde::to_segment_bytes(&original);
    assert!(wire.starts_with(b"TKS2"));
    let loaded = artifact_serde::from_fast_bytes(&wire).unwrap();
    assert_eq!(loaded.compressed_transition_segments.len(), 1);
    assert_eq!(loaded.transition_count(), original.transition_count());
    enumerate_bytes(b"abx", 3, |input| {
        for state in 0..original.num_states() {
            assert_eq!(
                normalized_exec(&loaded, input, state),
                normalized_exec(&original, input, state),
                "TKS2 mismatch from state {state} on {input:?}",
            );
        }
    });

    let huge_wire = artifact_serde::build_huge_bytes(&original).expect("small TKS3 fixture");
    assert!(huge_wire.starts_with(b"TKS3"));
    let huge_loaded = artifact_serde::from_fast_bytes(&huge_wire).unwrap();
    assert!(huge_loaded.has_packed_runtime_metadata());
    assert_eq!(huge_loaded.num_states(), original.num_states());
    assert_eq!(huge_loaded.transition_count(), original.transition_count());
    enumerate_bytes(b"abx", 3, |input| {
        for state in 0..original.num_states() {
            assert_eq!(
                normalized_exec(&huge_loaded, input, state),
                normalized_exec(&original, input, state),
                "TKS3 mismatch from state {state} on {input:?}",
            );
        }
    });

    // TKS2 is also the fallback for packed sources whose compressed
    // components are separated by ordinary states. It must not rely on
    // the loaded tokenizer retaining either owned rows or owned metadata.
    let fallback_wire = artifact_serde::to_segment_bytes(&huge_loaded);
    let fallback = artifact_serde::from_fast_bytes(&fallback_wire).unwrap();
    assert_eq!(fallback.num_states(), original.num_states());
    assert_eq!(fallback.compressed_transition_segments.len(), 1);
    enumerate_bytes(b"abx", 3, |input| {
        for state in 0..original.num_states() {
            assert_eq!(normalized_exec(&fallback, input, state),
                       normalized_exec(&original, input, state),
                       "packed-to-TKS2 mismatch state={state} input={input:?}");
        }
    });

}

#[test]
fn segment_fallback_preserves_packed_overflow_deltas_and_ordinary_tail() {
    const COMPRESSED_STATES: u32 = 70_000;
    let mut dfa = DFA::new(COMPRESSED_STATES as usize + 1);
    dfa.ensure_group_capacity(1);
    let mut live = BitSet::new(1);
    live.set(0);
    for state in 0..=COMPRESSED_STATES {
        dfa.overwrite_state_metadata(state,
            if state == COMPRESSED_STATES { live.clone() } else { BitSet::new(1) },
            live.clone());
    }
    dfa.add_epsilon_transition(0, 1);
    let mut byte_to_class = vec![u8::MAX; 256];
    for byte in b'a'..=b'z' { byte_to_class[byte as usize] = 0; }
    let targets = (0..COMPRESSED_STATES).map(|state| {
        if state == 0 { COMPRESSED_STATES - 1 }
        else if state == COMPRESSED_STATES - 1 { 0 }
        else { state }
    }).collect();
    let original = Tokenizer::from_parts_with_compressed_transitions(dfa, 1, None,
        vec![CompressedTransitionSegment {
            state_offset: 1, state_count: COMPRESSED_STATES,
            byte_to_class: Arc::from(byte_to_class.into_boxed_slice()),
            class_members: Arc::from([Vec::from_iter(b'a'..=b'z').into_boxed_slice()]),
            row_offsets: Arc::from(Vec::from_iter(0..=COMPRESSED_STATES).into_boxed_slice()),
            entries: CompressedTransitionEntries::from_parts(vec![0; COMPRESSED_STATES as usize], targets),
            expanded_transition_count: COMPRESSED_STATES as usize * 26,
        }]);
    let wire = Arc::new(artifact_serde::build_huge_bytes(&original).unwrap());
    let mut packed = artifact_serde::from_fast_bytes_backed(&wire, Arc::clone(&wire), 0).unwrap();
    assert!(!packed.packed_compressed_transition_segments[0].overflow_indices.is_empty());
    packed.materialize_runtime_metadata_for_structural_mutation();
    let tail = packed.dfa.add_state();
    packed.dfa.overwrite_state_metadata(tail, live.clone(), live);
    packed.dfa.add_transition(tail, b'!', 0);
    assert!(artifact_serde::build_huge_bytes(&packed).is_none(),
        "an ordinary tail must exercise the non-suffix fallback");
    let fallback_wire = artifact_serde::to_segment_bytes(&packed);
    assert!(fallback_wire.len() < 40 * COMPRESSED_STATES as usize,
        "class rows must not expand into 26 byte edges each");
    let loaded = artifact_serde::from_fast_bytes(&fallback_wire).unwrap();
    assert_eq!(loaded.num_states(), packed.num_states());
    assert!(loaded.has_compressed_transition_segments());
    for state in [0, 1, 2, COMPRESSED_STATES, tail] {
        assert_eq!(loaded.possible_future_terminals(state), packed.possible_future_terminals(state));
        assert_eq!(loaded.matched_terminal_bitset(state), packed.matched_terminal_bitset(state));
        for byte in 0..=255 {
            assert_eq!(loaded.step(state, byte), packed.step(state, byte),
                "packed overflow/tail mismatch state={state} byte={byte}");
        }
    }
}

#[test]
fn huge_wire_roundtrips_compressed_transitions_with_wide_terminal_metadata() {
    const TERMINALS: usize = 300;
    const WIDE_TERMINAL: usize = 299;

    let mut dfa = DFA::new(4);
    dfa.ensure_group_capacity(TERMINALS);
    let mut group = U8Set::empty();
    group.insert(b'a');
    group.insert(b'b');
    dfa.set_group_u8set(WIDE_TERMINAL as u32, group);
    dfa.add_epsilon_transition(0, 1);

    for state in 0..4u32 {
        let mut futures = BitSet::new(TERMINALS);
        if state != 3 {
            futures.set(WIDE_TERMINAL);
        }
        let mut finalizers = BitSet::new(TERMINALS);
        if state == 3 {
            finalizers.set(WIDE_TERMINAL);
        }
        dfa.overwrite_state_metadata(state, finalizers, futures);
    }

    let mut byte_to_class = vec![u8::MAX; 256];
    byte_to_class[b'a' as usize] = 0;
    byte_to_class[b'b' as usize] = 1;
    let segment = CompressedTransitionSegment {
        state_offset: 1,
        state_count: 3,
        byte_to_class: Arc::from(byte_to_class.into_boxed_slice()),
        class_members: Arc::from(
            vec![vec![b'a'].into_boxed_slice(), vec![b'b'].into_boxed_slice()]
                .into_boxed_slice(),
        ),
        row_offsets: Arc::from(vec![0u32, 1, 2, 2].into_boxed_slice()),
        entries: CompressedTransitionEntries::from_parts(vec![0, 1], vec![1, 2]),
        expanded_transition_count: 2,
    };
    let original = Tokenizer::from_parts_with_compressed_transitions(
        dfa,
        TERMINALS as u32,
        None,
        vec![segment],
    );

    let wire = artifact_serde::build_huge_bytes(&original).expect("wide TKS3 fixture");
    assert!(wire.starts_with(b"TKS3"));
    let backing = Arc::new(wire);
    let loaded = artifact_serde::from_fast_bytes_backed(
        backing.as_slice(),
        Arc::clone(&backing),
        0,
    )
    .expect("wide backed TKS3 roundtrip");

    assert_eq!(loaded.num_terminals(), TERMINALS as u32);
    assert_eq!(loaded.num_states(), original.num_states());
    assert!(loaded.has_packed_runtime_metadata());
    assert!(loaded.has_compressed_transition_segments());
    assert!(loaded.matched_terminal_bitset(3).contains(WIDE_TERMINAL));
    assert!(loaded.possible_future_terminals(1).contains(WIDE_TERMINAL));
    enumerate_bytes(b"abx", 3, |input| {
        for state in 0..original.num_states() {
            assert_eq!(
                normalized_exec(&loaded, input, state),
                normalized_exec(&original, input, state),
                "wide TKS3 mismatch from state {state} on {input:?}",
            );
        }
    });

    let reencoded = artifact_serde::build_huge_bytes(&loaded)
        .expect("loaded wide TKS3 tokenizer must remain compactly serializable");
    assert!(reencoded.starts_with(b"TKS3"));
    let reloaded = artifact_serde::from_fast_bytes(&reencoded)
        .expect("re-encoded wide TKS3 roundtrip");
    assert_eq!(reloaded.num_terminals(), TERMINALS as u32);
    assert!(reloaded.matched_terminal_bitset(3).contains(WIDE_TERMINAL));
    assert!(reloaded.possible_future_terminals(1).contains(WIDE_TERMINAL));
}

#[test]
fn fast_wire_with_packed_metadata_preserves_tkf2_transition_prefix() {
    let mut dfa = DFA::new(4);
    dfa.ensure_group_capacity(2);
    dfa.add_transition(0, b'a', 1);
    dfa.add_transition(0, b'b', 2);
    dfa.add_transition(1, b'b', 3);
    dfa.recompute_possible_futures();
    let tokenizer = Tokenizer::from_parts(dfa, 2, None);

    let tkf3 = artifact_serde::to_fast_bytes_with_packed_metadata(&tokenizer);
    let mut legacy_prefix = artifact_serde::to_fast_bytes(&tokenizer);
    let terminal_count = u32::from_le_bytes(legacy_prefix[4..8].try_into().unwrap()) as usize;
    let state_count = u32::from_le_bytes(legacy_prefix[8..12].try_into().unwrap()) as usize;
    let transition_count =
        u32::from_le_bytes(legacy_prefix[12..16].try_into().unwrap()) as usize;
    let state_id_width = legacy_prefix[28] as usize;
    let transition_end = 32
        + terminal_count * 32
        + (state_count + 1) * 4
        + transition_count
        + transition_count * state_id_width;
    legacy_prefix.truncate(transition_end);
    legacy_prefix[..4].copy_from_slice(b"TKF3");

    assert_eq!(&tkf3[..transition_end], legacy_prefix.as_slice());
}

#[test]
fn fast_wire_with_packed_metadata_roundtrips_execution() {
    let mut dfa = DFA::new(4);
    dfa.ensure_group_capacity(2);
    dfa.add_transition(0, b'a', 1);
    dfa.add_transition(0, b'b', 2);
    dfa.add_transition(1, b'b', 3);
    let mut final_a = BitSet::new(2);
    final_a.set(0);
    dfa.overwrite_state_metadata(1, final_a, BitSet::new(2));
    let mut final_ab = BitSet::new(2);
    final_ab.set(1);
    dfa.overwrite_state_metadata(3, final_ab, BitSet::new(2));
    dfa.recompute_possible_futures();

    let original = Tokenizer::from_parts(dfa, 2, None);
    let wire = artifact_serde::to_fast_bytes_with_packed_metadata(&original);
    assert!(wire.starts_with(b"TKF3"));
    let loaded = artifact_serde::from_fast_bytes(&wire).expect("TKF3 tokenizer roundtrip");
    assert!(loaded.has_packed_runtime_transitions());
    assert!(loaded.has_packed_runtime_metadata());
    assert_eq!(loaded.num_states(), original.num_states());
    assert_eq!(loaded.transition_count(), original.transition_count());

    for input in [
        b"".as_slice(),
        b"a".as_slice(),
        b"b".as_slice(),
        b"ab".as_slice(),
        b"abb".as_slice(),
    ] {
        for state in 0..original.num_states() {
            assert_eq!(
                normalized_exec(&loaded, input, state),
                normalized_exec(&original, input, state),
                "TKF3 mismatch from state {state} on {input:?}",
            );
        }
    }

    let mut unknown_flags = wire.clone();
    unknown_flags[artifact_serde::FAST_WIRE_FLAGS_OFFSET] |= 0x80;
    assert!(
        artifact_serde::from_fast_bytes(&unknown_flags).is_err(),
        "unknown TKF3 flags must fail closed",
    );

    let mut corrupted = wire;
    corrupted[8..12].copy_from_slice(&0u32.to_le_bytes());
    assert!(
        artifact_serde::from_fast_bytes(&corrupted).is_err(),
        "invalid TKF3 state count must fail closed",
    );
}

#[test]
fn fast_wire_packed_transitions_preserve_multi_state_execution() {
    let mut dfa = DFA::new(3);
    dfa.ensure_group_capacity(2);
    dfa.add_transition(0, b'a', 1);
    dfa.add_transition(0, b'b', 2);
    let mut final_a = BitSet::new(2);
    final_a.set(0);
    dfa.overwrite_state_metadata(1, final_a, BitSet::new(2));
    let mut final_b = BitSet::new(2);
    final_b.set(1);
    dfa.overwrite_state_metadata(2, final_b, BitSet::new(2));
    dfa.recompute_possible_futures();

    let original = Tokenizer::from_parts(dfa, 2, None);
    let wire = artifact_serde::to_fast_bytes(&original);
    let loaded = artifact_serde::from_fast_bytes(&wire).expect("fast tokenizer roundtrip");
    assert!(loaded.has_packed_runtime_transitions());

    for input in [b"a".as_slice(), b"b".as_slice(), b"ab".as_slice()] {
        assert_eq!(
            normalized_exec(&loaded, input, loaded.initial_state()),
            normalized_exec(&original, input, original.initial_state()),
            "fast-wire execution mismatch on {input:?}",
        );
    }
}

#[test]
fn disjoint_union_preserves_loaded_packed_transition_rows_without_materializing() {
    let mut dfa = DFA::new(3);
    dfa.ensure_group_capacity(2);
    dfa.add_transition(0, b'a', 1);
    dfa.add_transition(0, b'b', 2);
    let mut final_a = BitSet::new(2);
    final_a.set(0);
    dfa.overwrite_state_metadata(1, final_a, BitSet::new(2));
    let mut final_b = BitSet::new(2);
    final_b.set(1);
    dfa.overwrite_state_metadata(2, final_b, BitSet::new(2));
    dfa.recompute_possible_futures();

    let original = Tokenizer::from_parts(dfa, 2, None);
    let wire = artifact_serde::to_fast_bytes(&original);
    let loaded = artifact_serde::from_fast_bytes(&wire).expect("fast tokenizer roundtrip");
    assert!(loaded.packed_runtime_transitions.is_some());
    assert!(loaded.dfa.states()[0].transitions.is_empty());

    let (merged, offsets) = Tokenizer::disjoint_union_with_terminal_offsets(&[(&loaded, 0)]);
    let offset = offsets[0];
    assert!(merged.packed_runtime_transitions.is_none());
    assert_eq!(merged.packed_runtime_transition_segments.len(), 1);
    assert_eq!(merged.step(offset + loaded.initial_state(), b'a'), Some(offset + 1));
    assert_eq!(merged.step(offset + loaded.initial_state(), b'b'), Some(offset + 2));
    for input in [b"a".as_slice(), b"b".as_slice(), b"ab".as_slice()] {
        let (source_ends, source_matches) =
            normalized_exec(&loaded, input, loaded.initial_state());
        let expected = (
            source_ends
                .into_iter()
                .map(|state| offset + state)
                .collect::<Vec<_>>(),
            source_matches
                .into_iter()
                .map(|(terminal, width, state)| (terminal, width, offset + state))
                .collect::<Vec<_>>(),
        );
        assert_eq!(
            normalized_exec(&merged, input, offset + loaded.initial_state()),
            expected,
            "rebased packed execution mismatch on {input:?}",
        );
    }
}

#[test]
fn fast_wire_roundtrip_preserves_nested_packed_transition_segments() {
    let mut dfa = DFA::new(3);
    dfa.ensure_group_capacity(2);
    dfa.add_transition(0, b'a', 1);
    dfa.add_transition(0, b'b', 2);
    let mut final_a = BitSet::new(2);
    final_a.set(0);
    dfa.overwrite_state_metadata(1, final_a, BitSet::new(2));
    let mut final_b = BitSet::new(2);
    final_b.set(1);
    dfa.overwrite_state_metadata(2, final_b, BitSet::new(2));
    dfa.recompute_possible_futures();

    let original = Tokenizer::from_parts(dfa, 2, None);
    let first_wire = artifact_serde::to_fast_bytes(&original);
    let loaded = artifact_serde::from_fast_bytes(&first_wire).expect("first fast tokenizer roundtrip");
    let (nested, offsets) = Tokenizer::disjoint_union_with_terminal_offsets(&[(&loaded, 0)]);
    let offset = offsets[0];
    assert_eq!(nested.packed_runtime_transition_segments.len(), 1);

    let layout = artifact_serde::fast_layout_for_write(&nested)
        .expect("ordinary packed segments should still support TKF2");
    let mut second_wire = vec![0u8; layout.len()];
    artifact_serde::write_fast_bytes_with_layout(&nested, layout, &mut second_wire)
        .expect("nested packed tokenizer should serialize");
    let roundtripped = artifact_serde::from_fast_bytes(&second_wire)
        .expect("nested fast tokenizer roundtrip");

    for input in [b"a".as_slice(), b"b".as_slice(), b"ab".as_slice()] {
        let (source_ends, source_matches) = normalized_exec(&loaded, input, loaded.initial_state());
        let expected = (
            source_ends.into_iter().map(|state| offset + state).collect::<Vec<_>>(),
            source_matches
                .into_iter()
                .map(|(terminal, width, state)| (terminal, width, offset + state))
                .collect::<Vec<_>>(),
        );
        assert_eq!(
            normalized_exec(&roundtripped, input, offset + loaded.initial_state()),
            expected,
            "nested packed fast-wire mismatch on {input:?}",
        );
    }
}

#[test]
fn serialized_terminal_language_equivalence_is_exact_without_exprs() {
    let left = serialized_roundtrip(&tokenizer_from_exprs(vec![bytes(b"ab")]));
    let equal = serialized_roundtrip(&tokenizer_from_exprs(vec![bytes(b"ab")]));
    let different_same_support =
        serialized_roundtrip(&tokenizer_from_exprs(vec![bytes(b"ba")]));

    assert_eq!(
        left.terminal_language_equivalent_bounded(0, &equal, 0, 64, 10_000),
        Some(true),
    );
    assert_eq!(
        left.terminal_language_equivalent_bounded(
            0,
            &different_same_support,
            0,
            64,
            10_000,
        ),
        Some(false),
    );
}

#[test]
fn serialized_terminal_language_equivalence_budget_fails_closed() {
    let left = serialized_roundtrip(&tokenizer_from_exprs(vec![bytes(b"ab")]));
    let right = serialized_roundtrip(&tokenizer_from_exprs(vec![bytes(b"ab")]));
    assert_eq!(
        left.terminal_language_equivalent_bounded(0, &right, 0, 1, 10_000),
        None,
        "budget exhaustion must never be interpreted as equivalence",
    );
}

#[test]
fn exact_terminal_observation_partition_matches_pairwise_bisimulation() {
    let tokenizers = vec![
        arbitrary_epsilon_l1_test_tokenizer(),
        tokenizer_from_exprs(vec![plus(bytes(b"a")), bytes(b"ab"), bytes(b"ba")]),
    ];

    for tokenizer in tokenizers {
        for terminal in 0..tokenizer.num_terminals() {
            let (classes, _, _) = tokenizer
                .exact_terminal_observation_partition(terminal, 10_000, 1_000_000)
                .expect("small test tokenizer must quotient within budget");
            let mut observed = BitSet::new(tokenizer.num_terminals() as usize);
            observed.set(terminal as usize);
            for left in 0..tokenizer.num_states() {
                for right in 0..tokenizer.num_states() {
                    let exact = tokenizer
                        .state_sets_observation_equivalent_exact(
                            &[left],
                            &[right],
                            &observed,
                            10_000,
                        )
                        .expect("small pairwise proof must stay within budget");
                    assert_eq!(
                        classes[left as usize] == classes[right as usize],
                        exact,
                        "terminal {terminal}, raw states {left} and {right}",
                    );
                }
            }
        }
    }
}

#[test]
fn scalar_exact_terminal_partition_never_aliases_uncovered_live_residuals() {
    let tokenizer = dispatch_prefix_tokenizer(true);
    assert!(tokenizer.has_scalar_deterministic_dispatch());
    let terminal = 0;
    let (classes, _, _) = tokenizer
        .exact_terminal_observation_partition(terminal, 10_000, 1_000_000)
        .expect("small scalar tokenizer must quotient within budget");
    let mut observed = BitSet::new(tokenizer.num_terminals() as usize);
    observed.set(terminal as usize);

    // State 4 is deliberately reset-unreachable but terminal-live.  It
    // must not be class zero merely because the fast scalar proof starts
    // from the certified dispatch component.
    assert!(tokenizer.state_live_for_terminal(4, terminal));
    assert_ne!(classes[4], 0);

    // The fast path may conservatively leave uncovered live residuals in
    // singleton classes, but every actual nonzero alias must still be an
    // exact arbitrary-continuation equivalence proof.
    for left in 0..tokenizer.num_states() {
        for right in 0..tokenizer.num_states() {
            if classes[left as usize] == 0
                || classes[left as usize] != classes[right as usize]
            {
                continue;
            }
            assert_eq!(
                tokenizer
                    .state_sets_observation_equivalent_exact(
                        &[left],
                        &[right],
                        &observed,
                        10_000,
                    )
                    .expect("small pairwise proof must stay within budget"),
                true,
                "scalar alias for raw states {left} and {right} must be exact",
            );
        }
    }
}

fn normalized_exec(
    tokenizer: &Tokenizer,
    input: &[u8],
    start: u32,
) -> (Vec<u32>, Vec<(u32, usize, u32)>) {
    let result = tokenizer.execute_from_state_all_widths(input, start);
    let mut end_states = result.end_state.into_vec();
    end_states.sort_unstable();
    let mut matches = result
        .matches
        .into_iter()
        .map(|matched| (matched.id, matched.width, matched.end_state))
        .collect::<Vec<_>>();
    matches.sort_unstable();
    (end_states, matches)
}

fn enumerate_bytes(
    alphabet: &[u8],
    max_len: usize,
    mut visit: impl FnMut(&[u8]),
) {
    fn rec(
        alphabet: &[u8],
        remaining: usize,
        word: &mut Vec<u8>,
        visit: &mut impl FnMut(&[u8]),
    ) {
        visit(word);
        if remaining == 0 {
            return;
        }
        for &byte in alphabet {
            word.push(byte);
            rec(alphabet, remaining - 1, word, visit);
            word.pop();
        }
    }
    rec(alphabet, max_len, &mut Vec::new(), &mut visit);
}

fn semantic_exec(tokenizer: &Tokenizer, input: &[u8]) -> (bool, Vec<(u32, usize)>) {
    let result = tokenizer.execute_from_state_all_widths(input, tokenizer.initial_state());
    let mut matches = result
        .matches
        .into_iter()
        .map(|matched| (matched.id, matched.width))
        .collect::<Vec<_>>();
    matches.sort_unstable();
    matches.dedup();
    (!result.end_state.is_empty(), matches)
}

#[test]
fn hybrid_virtual_unit_repeat_proxy_root_steps_exactly() {
    let mut tokenizer = tokenizer_from_exprs(vec![
        Expr::U8Seq(b"b".to_vec()),
        Expr::U8Class(U8Set::empty()),
    ]);
    tokenizer.isolate_start_state_and_drain_nullable_terminals();
    tokenizer
        .install_virtual_zero_min_unit_repeat_component(
            U8Set::single(b'a'),
            1_000_000_000,
            1,
        )
        .unwrap();
    let root = tokenizer
        .epsilon_closure_states(&[tokenizer.initial_state()])
        .into_iter()
        .find(|&state| {
            state != tokenizer.initial_state()
                && tokenizer.possible_future_terminals(state).contains(1)
        })
        .expect("virtual repeat proxy root");
    let target = tokenizer.get_transition(root, b'a');
    assert_ne!(target, u32::MAX);
    assert_eq!(tokenizer.matched_terminals_iter(target).collect::<Vec<_>>(), vec![1]);
}

#[test]
fn exact_terminal_observation_partition_declines_virtual_state_space() {
    let mut tokenizer = tokenizer_from_exprs(vec![
        Expr::U8Seq(b"b".to_vec()),
        Expr::U8Class(U8Set::empty()),
    ]);
    tokenizer.isolate_start_state_and_drain_nullable_terminals();
    tokenizer
        .install_virtual_zero_min_unit_repeat_component(
            U8Set::single(b'a'),
            1_000_000_000,
            1,
        )
        .unwrap();
    assert!(tokenizer.has_any_virtual_runtime());
    assert!(
        tokenizer
            .exact_terminal_observation_partition(1, 10_000, 1_000_000)
            .is_none(),
        "a raw-state observation quotient cannot certify lazily allocated virtual states",
    );
}

#[test]
fn lazy_binary_repeat_intersection_matches_materialized_small_oracle() {
    use crate::automata::lexer::compile::compile_terminal_expr_dfa;
    use crate::automata::lexer::runtime_repeat_product::{
        VirtualBinaryRepeatIntersectionDescriptor, VirtualBoundedRepeatSpec,
    };

    let left_body = Expr::Choice(vec![
        Expr::U8Seq(b"ab".to_vec()),
        Expr::U8Seq(b"c".to_vec()),
    ]);
    let right_body = Expr::Choice(vec![
        Expr::U8Seq(b"a".to_vec()),
        Expr::U8Seq(b"bc".to_vec()),
    ]);
    let left_max = 4usize;
    let right_max = 3usize;
    let expression = Expr::Intersect {
        expr: Box::new(Expr::Repeat {
            expr: Box::new(left_body.clone()),
            min: 0,
            max: Some(left_max),
        }),
        intersect: Box::new(Expr::Repeat {
            expr: Box::new(right_body.clone()),
            min: 0,
            max: Some(right_max),
        }),
    };

    let mut ordinary = tokenizer_from_exprs(vec![expression]);
    ordinary.isolate_start_state_and_drain_nullable_terminals();

    let left_dfa = Arc::new(compile_terminal_expr_dfa(&left_body));
    let right_dfa = Arc::new(compile_terminal_expr_dfa(&right_body));
    let mut virtualized = tokenizer_from_exprs(vec![Expr::U8Class(U8Set::empty())]);
    virtualized.isolate_start_state_and_drain_nullable_terminals();
    virtualized
        .install_virtual_binary_repeat_intersection_component(
            VirtualBinaryRepeatIntersectionDescriptor {
                byte_support: U8Set::from_bytes(b"abc"),
                left: VirtualBoundedRepeatSpec {
                    base_dfa: left_dfa,
                    min: 0,
                    max: left_max as u32,
                },
                right: VirtualBoundedRepeatSpec {
                    base_dfa: right_dfa,
                    min: 0,
                    max: right_max as u32,
                },
            },
            0,
        )
        .unwrap();

    enumerate_bytes(b"abcx", 8, |input| {
        let (_, ordinary_matches) = semantic_exec(&ordinary, input);
        let (_, virtual_matches) = semantic_exec(&virtualized, input);
        assert_eq!(
            virtual_matches, ordinary_matches,
            "lazy repeat product changed exact matches on {input:?}",
        );
    });
    assert!(virtualized.virtual_binary_repeat_intersection_interned_state_count() < 128);
}

#[test]
fn repeat_product_state_owner_index_dispatches_multiple_components_directly() {
    use crate::automata::lexer::compile::compile_terminal_expr_dfa;
    use crate::automata::lexer::runtime_repeat_product::{
        VirtualBinaryRepeatIntersectionDescriptor, VirtualBoundedRepeatSpec,
    };

    let descriptor = |byte| {
        let base = Arc::new(compile_terminal_expr_dfa(&Expr::U8Seq(vec![byte])));
        VirtualBinaryRepeatIntersectionDescriptor {
            byte_support: U8Set::single(byte),
            left: VirtualBoundedRepeatSpec {
                base_dfa: Arc::clone(&base),
                min: 0,
                max: 32,
            },
            right: VirtualBoundedRepeatSpec {
                base_dfa: base,
                min: 0,
                max: 32,
            },
        }
    };

    let mut tokenizer = Tokenizer::from_parts(DFA::new(1), 2, None);
    tokenizer
        .install_virtual_binary_repeat_intersection_components(vec![
            (descriptor(b'a'), 0),
            (descriptor(b'b'), 1),
        ])
        .expect("two repeat-product components must install");

    let a_state = tokenizer.step(1, b'a').expect("first product must advance");
    let b_state = tokenizer.step(2, b'b').expect("second product must advance");
    assert_ne!(a_state, b_state);
    assert_eq!(
        tokenizer
            .virtual_repeat_runtime_for_state(a_state)
            .map(VirtualBinaryRepeatIntersectionRuntime::terminal),
        Some(0),
    );
    assert_eq!(
        tokenizer
            .virtual_repeat_runtime_for_state(b_state)
            .map(VirtualBinaryRepeatIntersectionRuntime::terminal),
        Some(1),
    );

    let aa_state = tokenizer
        .step(a_state, b'a')
        .expect("interleaved first product must remain owned");
    let bb_state = tokenizer
        .step(b_state, b'b')
        .expect("interleaved second product must remain owned");
    assert_eq!(
        tokenizer
            .virtual_repeat_runtime_for_state(aa_state)
            .map(VirtualBinaryRepeatIntersectionRuntime::terminal),
        Some(0),
    );
    assert_eq!(
        tokenizer
            .virtual_repeat_runtime_for_state(bb_state)
            .map(VirtualBinaryRepeatIntersectionRuntime::terminal),
        Some(1),
    );
}

#[test]
fn lazy_binary_repeat_body_product_budget_fails_before_proxy_install() {
    use crate::automata::lexer::runtime_repeat_product::{
        VirtualBinaryRepeatIntersectionDescriptor, VirtualBoundedRepeatSpec,
    };

    // 257^2 exceeds the exact body-product analysis budget. The installer
    // must reject this before appending any proxy root to the physical DFA.
    let descriptor = VirtualBinaryRepeatIntersectionDescriptor {
        byte_support: U8Set::from_bytes(b"a"),
        left: VirtualBoundedRepeatSpec {
            base_dfa: Arc::new(DFA::new(257)),
            min: 0,
            max: 10_000,
        },
        right: VirtualBoundedRepeatSpec {
            base_dfa: Arc::new(DFA::new(257)),
            min: 0,
            max: 10_000,
        },
    };
    let mut tokenizer = tokenizer_from_exprs(vec![Expr::U8Class(U8Set::empty())]);
    tokenizer.isolate_start_state_and_drain_nullable_terminals();
    let before_states = tokenizer.num_states();
    assert!(
        tokenizer
            .install_virtual_binary_repeat_intersection_component(descriptor, 0)
            .is_none(),
    );
    assert_eq!(tokenizer.num_states(), before_states);
    assert!(!tokenizer.has_virtual_binary_repeat_intersection());
}

#[test]
fn nonzero_min_repeat_product_rejects_unsynchronized_manual_descriptor() {
    use crate::automata::lexer::compile::compile_terminal_expr_dfa;
    use crate::automata::lexer::runtime_repeat_product::{
        VirtualBinaryRepeatIntersectionDescriptor, VirtualBoundedRepeatSpec,
    };

    let left = Arc::new(compile_terminal_expr_dfa(&Expr::U8Seq(b"a".to_vec())));
    let right = Arc::new(compile_terminal_expr_dfa(&Expr::U8Seq(b"aa".to_vec())));
    let descriptor = VirtualBinaryRepeatIntersectionDescriptor {
        byte_support: U8Set::single(b'a'),
        left: VirtualBoundedRepeatSpec {
            base_dfa: left,
            min: 4,
            max: 4,
        },
        right: VirtualBoundedRepeatSpec {
            base_dfa: right,
            min: 1,
            max: 1,
        },
    };
    let mut tokenizer = tokenizer_from_exprs(vec![Expr::U8Class(U8Set::empty())]);
    tokenizer.isolate_start_state_and_drain_nullable_terminals();
    let before_states = tokenizer.num_states();
    assert!(
        tokenizer
            .install_virtual_binary_repeat_intersection_component(descriptor, 0)
            .is_none()
    );
    assert_eq!(tokenizer.num_states(), before_states);
    assert!(!tokenizer.has_virtual_binary_repeat_intersection());
}

#[test]
fn nonzero_min_repeat_product_rejects_empty_body_language() {
    use crate::automata::lexer::compile::compile_terminal_expr_dfa;
    use crate::automata::lexer::runtime_repeat_product::{
        VirtualBinaryRepeatIntersectionDescriptor, VirtualBoundedRepeatSpec,
    };

    let empty_body = Expr::Seq(vec![
        Expr::U8Class(U8Set::empty()),
        Expr::U8Seq(b"a".to_vec()),
    ]);
    let base = Arc::new(compile_terminal_expr_dfa(&empty_body));
    assert!(!base.possible_future_group_ids(0).contains(0));
    let spec = VirtualBoundedRepeatSpec {
        base_dfa: Arc::clone(&base),
        min: 1,
        max: 10_000,
    };
    let descriptor = VirtualBinaryRepeatIntersectionDescriptor {
        byte_support: U8Set::single(b'a'),
        left: spec.clone(),
        right: spec,
    };
    let mut tokenizer = tokenizer_from_exprs(vec![Expr::U8Class(U8Set::empty())]);
    tokenizer.isolate_start_state_and_drain_nullable_terminals();
    let before_states = tokenizer.num_states();
    assert!(
        tokenizer
            .install_virtual_binary_repeat_intersection_component(descriptor, 0)
            .is_none()
    );
    assert_eq!(tokenizer.num_states(), before_states);
    assert!(!tokenizer.has_virtual_binary_repeat_intersection());
}

#[test]
fn lazy_binary_repeat_mask_projection_is_exact_within_vocab_horizon() {
    use crate::automata::lexer::compile::compile_terminal_expr_dfa;
    use crate::automata::lexer::runtime_repeat_product::{
        VirtualBinaryRepeatIntersectionDescriptor, VirtualBoundedRepeatSpec,
    };

    const HORIZON: usize = 3;
    let left_body = Expr::Choice(vec![
        Expr::U8Seq(b"ab".to_vec()),
        Expr::U8Seq(b"c".to_vec()),
    ]);
    let right_body = Expr::Choice(vec![
        Expr::U8Seq(b"a".to_vec()),
        Expr::U8Seq(b"bc".to_vec()),
    ]);
    let descriptor = VirtualBinaryRepeatIntersectionDescriptor {
        byte_support: U8Set::from_bytes(b"abc"),
        left: VirtualBoundedRepeatSpec {
            base_dfa: Arc::new(compile_terminal_expr_dfa(&left_body)),
            min: 0,
            max: 20,
        },
        right: VirtualBoundedRepeatSpec {
            base_dfa: Arc::new(compile_terminal_expr_dfa(&right_body)),
            min: 0,
            max: 17,
        },
    };
    let mut exact = tokenizer_from_exprs(vec![Expr::U8Class(U8Set::empty())]);
    exact.isolate_start_state_and_drain_nullable_terminals();
    exact
        .install_virtual_binary_repeat_intersection_component(descriptor, 0)
        .unwrap();
    let (mask, projection) = exact
        .virtual_binary_repeat_intersection_mask_tokenizer(HORIZON)
        .unwrap();

    let exact_root = exact
        .epsilon_closure_states(&[exact.initial_state()])
        .into_iter()
        .find(|&state| state != exact.initial_state())
        .unwrap();

    // Reach a representative collection of exact interior residuals. The
    // prefixes themselves may be longer than HORIZON; only each suffix
    // comparison is horizon-bounded.
    enumerate_bytes(b"abc", 8, |prefix| {
        let mut exact_state = exact_root;
        for &byte in prefix {
            let next = exact.get_transition(exact_state, byte);
            if next == u32::MAX {
                return;
            }
            exact_state = next;
        }
        let mask_state = projection
            .project(exact_state)
            .expect("every exact lazy residual must have a static mask state");
        assert_eq!(
            exact.matched_terminal_bitset(exact_state),
            mask.matched_terminal_bitset(mask_state),
            "source finalizers differ after prefix {prefix:?}",
        );

        enumerate_bytes(b"abcx", HORIZON, |suffix| {
            let mut full = exact_state;
            let mut projected = mask_state;
            for (offset, &byte) in suffix.iter().enumerate() {
                let full_next = exact.get_transition(full, byte);
                let projected_next = mask.get_transition(projected, byte);
                assert_eq!(
                    full_next == u32::MAX,
                    projected_next == u32::MAX,
                    "liveness differs after prefix {prefix:?}, suffix {suffix:?} at {offset}",
                );
                if full_next == u32::MAX {
                    break;
                }
                assert_eq!(
                    exact.matched_terminal_bitset(full_next),
                    mask.matched_terminal_bitset(projected_next),
                    "finalizers differ after prefix {prefix:?}, suffix {suffix:?} at {offset}",
                );
                assert_eq!(
                    exact.possible_future_terminals(full_next),
                    mask.possible_future_terminals(projected_next),
                    "futures differ after prefix {prefix:?}, suffix {suffix:?} at {offset}",
                );
                full = full_next;
                projected = projected_next;
            }
        });
    });
}

#[test]
fn synchronized_nonzero_min_repeat_projection_matches_materialized_oracle() {
    use crate::automata::lexer::compile::compile_terminal_expr_dfa;
    use crate::automata::lexer::runtime_repeat_product::{
        VirtualBinaryRepeatIntersectionDescriptor, VirtualBoundedRepeatSpec,
    };

    const HORIZON: usize = 3;
    let body = Expr::Choice(vec![
        Expr::U8Seq(b"ab".to_vec()),
        Expr::U8Seq(b"c".to_vec()),
    ]);
    let expression = Expr::Repeat {
        expr: Box::new(body.clone()),
        min: 6,
        max: Some(9),
    };
    let mut ordinary = tokenizer_from_exprs(vec![expression]);
    ordinary.isolate_start_state_and_drain_nullable_terminals();

    let base = Arc::new(compile_terminal_expr_dfa(&body));
    let spec = VirtualBoundedRepeatSpec {
        base_dfa: base,
        min: 6,
        max: 9,
    };
    let descriptor = VirtualBinaryRepeatIntersectionDescriptor {
        byte_support: U8Set::from_bytes(b"abc"),
        left: spec.clone(),
        right: spec,
    };
    let mut exact = tokenizer_from_exprs(vec![Expr::U8Class(U8Set::empty())]);
    exact.isolate_start_state_and_drain_nullable_terminals();
    exact
        .install_virtual_binary_repeat_intersection_component(descriptor, 0)
        .unwrap();
    let (mask, projection) = exact
        .virtual_binary_repeat_intersection_mask_tokenizer(HORIZON)
        .unwrap();

    enumerate_bytes(b"abcx", 10, |input| {
        let (_, ordinary_matches) = semantic_exec(&ordinary, input);
        let (_, exact_matches) = semantic_exec(&exact, input);
        assert_eq!(
            exact_matches, ordinary_matches,
            "nonzero-min lazy repeat changed exact matches on {input:?}",
        );
    });

    let exact_root = exact
        .epsilon_closure_states(&[exact.initial_state()])
        .into_iter()
        .find(|&state| state != exact.initial_state())
        .unwrap();
    enumerate_bytes(b"abc", 8, |prefix| {
        let mut exact_state = exact_root;
        for &byte in prefix {
            let next = exact.get_transition(exact_state, byte);
            if next == u32::MAX {
                return;
            }
            exact_state = next;
        }
        let mask_state = projection
            .project(exact_state)
            .expect("every synchronized exact residual must have a finite projection");
        assert_eq!(
            exact.matched_terminal_bitset(exact_state),
            mask.matched_terminal_bitset(mask_state),
            "source finalizers differ after prefix {prefix:?}",
        );
        assert_eq!(
            exact.possible_future_terminals(exact_state),
            mask.possible_future_terminals(mask_state),
            "source futures differ after prefix {prefix:?}",
        );

        enumerate_bytes(b"abcx", HORIZON, |suffix| {
            let mut full = exact_state;
            let mut projected = mask_state;
            for (offset, &byte) in suffix.iter().enumerate() {
                let full_next = exact.get_transition(full, byte);
                let projected_next = mask.get_transition(projected, byte);
                assert_eq!(
                    full_next == u32::MAX,
                    projected_next == u32::MAX,
                    "liveness differs after prefix {prefix:?}, suffix {suffix:?} at {offset}",
                );
                if full_next == u32::MAX {
                    break;
                }
                assert_eq!(
                    exact.matched_terminal_bitset(full_next),
                    mask.matched_terminal_bitset(projected_next),
                    "finalizers differ after prefix {prefix:?}, suffix {suffix:?} at {offset}",
                );
                assert_eq!(
                    exact.possible_future_terminals(full_next),
                    mask.possible_future_terminals(projected_next),
                    "futures differ after prefix {prefix:?}, suffix {suffix:?} at {offset}",
                );
                full = full_next;
                projected = projected_next;
            }
        });
    });
}

#[test]
fn lazy_binary_repeat_mask_projection_declines_quadratic_horizon_before_allocation() {
    use crate::automata::lexer::compile::compile_terminal_expr_dfa;
    use crate::automata::lexer::runtime_repeat_product::{
        VirtualBinaryRepeatIntersectionDescriptor, VirtualBoundedRepeatSpec,
    };

    let body = Expr::U8Seq(b"a".to_vec());
    let base = Arc::new(compile_terminal_expr_dfa(&body));
    let descriptor = VirtualBinaryRepeatIntersectionDescriptor {
        byte_support: U8Set::from_bytes(b"a"),
        left: VirtualBoundedRepeatSpec {
            base_dfa: Arc::clone(&base),
            min: 0,
            max: 1_000_000_000,
        },
        right: VirtualBoundedRepeatSpec {
            base_dfa: base,
            min: 0,
            max: 900_000_000,
        },
    };
    let mut exact = tokenizer_from_exprs(vec![Expr::U8Class(U8Set::empty())]);
    exact.isolate_start_state_and_drain_nullable_terminals();
    exact
        .install_virtual_binary_repeat_intersection_component(descriptor, 0)
        .unwrap();

    assert!(
        exact
            .virtual_binary_repeat_intersection_mask_tokenizer(600)
            .is_none(),
        "a pathological finite-token horizon must decline the optional quotient instead of eagerly seeding its quadratic boundary grid",
    );
    assert!(
        exact.virtual_binary_repeat_intersection_interned_state_count() <= 1,
        "declining the quotient must not materialize exact residual states",
    );
}

#[test]
fn virtual_unit_repeat_mask_projection_is_exact_for_every_bounded_word() {
    use crate::automata::lexer::compile::build_virtual_zero_min_unit_repeat_tokenizer;

    const FULL_MAX: usize = 20;
    const HORIZON: usize = 3;
    let expression = Expr::Repeat {
        expr: Box::new(Expr::U8Class(U8Set::single(b'a'))),
        min: 0,
        max: Some(FULL_MAX),
    };
    let full = build_virtual_zero_min_unit_repeat_tokenizer(&[expression]).unwrap();
    let (mask, projection) = full
        .virtual_zero_min_unit_repeat_mask_tokenizer(HORIZON)
        .unwrap();

    assert_eq!(full.num_states(), 1, "the exact billion-style counter is not materialized");
    assert_eq!(mask.num_states(), HORIZON as u32 + 3);

    for full_state in 0..=FULL_MAX as u32 {
        let mask_state = projection.project(full_state).unwrap();
        assert_eq!(
            full.matched_terminal_bitset(full_state),
            mask.matched_terminal_bitset(mask_state),
            "source finalizers differ at exact state {full_state}",
        );
        assert_eq!(
            full.possible_future_terminals(full_state),
            mask.possible_future_terminals(mask_state),
            "source futures differ at exact state {full_state}",
        );

        enumerate_bytes(b"ab", HORIZON, |input| {
            let mut full_cursor = full_state;
            let mut mask_cursor = mask_state;
            for (offset, &byte) in input.iter().enumerate() {
                let full_next = full.get_transition(full_cursor, byte);
                let mask_next = mask.get_transition(mask_cursor, byte);
                assert_eq!(
                    full_next == u32::MAX,
                    mask_next == u32::MAX,
                    "liveness differs from exact state {full_state} on {input:?} at {offset}",
                );
                if full_next == u32::MAX {
                    break;
                }
                assert_eq!(
                    full.matched_terminal_bitset(full_next),
                    mask.matched_terminal_bitset(mask_next),
                    "finalizers differ from exact state {full_state} on {input:?} at {offset}",
                );
                assert_eq!(
                    full.possible_future_terminals(full_next),
                    mask.possible_future_terminals(mask_next),
                    "futures differ from exact state {full_state} on {input:?} at {offset}",
                );
                full_cursor = full_next;
                mask_cursor = mask_next;
            }
        });
    }

    let billion = Expr::Repeat {
        expr: Box::new(Expr::U8Class(U8Set::single(b'a'))),
        min: 0,
        max: Some(1_000_000_000),
    };
    let billion = build_virtual_zero_min_unit_repeat_tokenizer(&[billion]).unwrap();
    let (billion_mask, _) = billion
        .virtual_zero_min_unit_repeat_mask_tokenizer(HORIZON)
        .unwrap();
    assert_eq!(billion.num_states(), 1);
    assert_eq!(billion_mask.num_states(), HORIZON as u32 + 3);
}

#[test]
fn virtual_nonzero_min_unit_repeat_projection_matches_exact_counter() {
    use crate::automata::lexer::compile::build_virtual_unit_repeat_tokenizer;

    const FULL_MIN: usize = 8;
    const FULL_MAX: usize = 20;
    const HORIZON: usize = 3;
    let expression = Expr::Repeat {
        expr: Box::new(Expr::U8Class(U8Set::single(b'a'))),
        min: FULL_MIN,
        max: Some(FULL_MAX),
    };
    let full = build_virtual_unit_repeat_tokenizer(&[expression]).unwrap();
    let (mask, projection) = full.virtual_unit_repeat_mask_tokenizer(HORIZON).unwrap();

    assert_eq!(full.num_states(), 1, "the exact counter must stay arithmetic");
    assert!(projection.deep_lower_state().is_some());
    assert!(projection.interior_state().is_some());

    for full_state in 0..=FULL_MAX as u32 {
        let mask_state = projection.project(full_state).unwrap();
        assert_eq!(
            full.matched_terminal_bitset(full_state),
            mask.matched_terminal_bitset(mask_state),
            "source finalizers differ at exact state {full_state}",
        );
        assert_eq!(
            full.possible_future_terminals(full_state),
            mask.possible_future_terminals(mask_state),
            "source futures differ at exact state {full_state}",
        );

        enumerate_bytes(b"ab", HORIZON, |input| {
            let mut full_cursor = full_state;
            let mut mask_cursor = mask_state;
            for (offset, &byte) in input.iter().enumerate() {
                let full_next = full.get_transition(full_cursor, byte);
                let mask_next = mask.get_transition(mask_cursor, byte);
                assert_eq!(
                    full_next == u32::MAX,
                    mask_next == u32::MAX,
                    "liveness differs from exact state {full_state} on {input:?} at {offset}",
                );
                if full_next == u32::MAX {
                    break;
                }
                assert_eq!(
                    full.matched_terminal_bitset(full_next),
                    mask.matched_terminal_bitset(mask_next),
                    "finalizers differ from exact state {full_state} on {input:?} at {offset}",
                );
                assert_eq!(
                    full.possible_future_terminals(full_next),
                    mask.possible_future_terminals(mask_next),
                    "futures differ from exact state {full_state} on {input:?} at {offset}",
                );
                full_cursor = full_next;
                mask_cursor = mask_next;
            }
        });
    }

    let counts = projection.multiplicities();
    assert_eq!(counts.iter().sum::<usize>(), FULL_MAX + 1);
    let unique = projection.unique_full_states();
    for (mask_state, &multiplicity) in counts.iter().enumerate() {
        assert_eq!(
            unique[mask_state] != u32::MAX,
            multiplicity == 1,
            "unique-full-state metadata disagrees at mask state {mask_state}",
        );
    }
}

#[test]
fn materialized_certified_bounded_code_terminal_does_not_require_virtual_owner_metadata() {
    let expression = Expr::Intersect {
        expr: Box::new(Expr::Seq(vec![
            bytes(b"\""),
            Expr::Repeat {
                expr: Box::new(bytes(b"a")),
                min: 0,
                max: None,
            },
            bytes(b"\""),
        ])),
        intersect: Box::new(Expr::Seq(vec![
            bytes(b"\""),
            Expr::Repeat {
                expr: Box::new(bytes(b"a")),
                min: 0,
                max: Some(128),
            },
            bytes(b"\""),
        ])),
    };
    assert!(
        crate::automata::lexer::compile::expression_supports_bounded_code_residual_runtime(
            &expression,
        ),
        "test expression must be eligible for the optional bounded-code residual representation",
    );
    let original = tokenizer_from_exprs(vec![expression.clone()]);
    assert!(!original.has_any_virtual_runtime());
    assert!(
        (0..original.num_states())
            .any(|state| original.state_finalizers(state).contains(0)),
        "the compatibility case must contain the terminal physically",
    );

    let mut loaded = serialized_roundtrip(&original);
    loaded
        .restore_terminal_exprs_with_virtual_runtime_metadata(
            Some(vec![expression]),
            &[],
            false,
        )
        .expect("an older fully-materialized certifiable terminal must not require a residual sidecar");
    assert!(!loaded.has_any_virtual_runtime());
}

#[test]
fn exact_unit_runtime_restore_rejects_corrupt_serialized_futures() {
    use crate::automata::lexer::compile::build_virtual_zero_min_unit_repeat_tokenizer;

    let expression = Expr::Repeat {
        expr: Box::new(Expr::U8Seq(b"a".to_vec())),
        min: 0,
        max: Some(10_000),
    };
    let original = build_virtual_zero_min_unit_repeat_tokenizer(&[expression.clone()])
        .expect("standalone unit repeat should use the arithmetic runtime");
    let metadata = original.virtual_runtime_metadata();
    assert_eq!(metadata.len(), 1);
    assert_eq!(metadata[0].kind, VirtualTokenizerRuntimeKind::UnitRepeat);

    let mut loaded = serialized_roundtrip(&original);
    // Simulate a decodable but inconsistent artifact. Exact restoration
    // must validate this physical row before attaching the sidecar, rather
    // than reading the reconstructed runtime's own future set.
    loaded.dfa.overwrite_state_metadata(
        loaded.start_state(),
        BitSet::new(1),
        BitSet::new(1),
    );
    let error = loaded
        .restore_terminal_exprs_with_virtual_runtime_metadata(
            Some(vec![expression]),
            &metadata,
            false,
        )
        .unwrap_err();
    assert!(
        error.contains("inconsistent future metadata"),
        "unexpected restoration error: {error}",
    );
    assert!(
        loaded.virtual_unit_repeat.is_none(),
        "failed exact restoration must not leave a reconstructed sidecar installed",
    );
}

#[test]
fn residual_runtime_restore_migrates_legacy_exact_dead_root_futures() {
    let expression = Expr::Intersect {
        expr: Box::new(Expr::Seq(vec![
            Expr::Repeat {
                expr: Box::new(bytes(b"a")),
                min: 0,
                max: Some(10_000),
            },
            bytes(b"b"),
        ])),
        intersect: Box::new(Expr::Repeat {
            expr: Box::new(bytes(b"a")),
            min: 0,
            max: None,
        }),
    };
    let mut original = Tokenizer::from_parts(DFA::new(1), 1, None);
    original
        .install_virtual_residual_components(vec![(expression.clone(), 0)])
        .expect("dead Boolean residual must install conservatively");
    let metadata = original.virtual_runtime_metadata();
    assert_eq!(metadata.len(), 1);
    assert_eq!(metadata[0].kind, VirtualTokenizerRuntimeKind::ResidualExpr);
    let root = metadata[0].root_state;
    assert!(original.state_futures(root).contains(0));
    assert_eq!(original.exact_dynamic_state_has_future(root).unwrap(), false);

    let mut loaded = serialized_roundtrip(&original);
    assert!(!loaded.has_packed_runtime_metadata());
    // Pre-conservative artifacts stored the exact-dead root as future=false
    // and consequently omitted that terminal from the reset-state future
    // set as well.
    let root_finalizers = loaded.state_finalizers(root).clone();
    loaded
        .dfa
        .overwrite_state_metadata(root, root_finalizers, BitSet::new(1));
    let start = loaded.start_state();
    let start_finalizers = loaded.state_finalizers(start).clone();
    loaded
        .dfa
        .overwrite_state_metadata(start, start_finalizers, BitSet::new(1));

    loaded
        .restore_terminal_exprs_with_virtual_runtime_metadata(
            Some(vec![expression]),
            &metadata,
            true,
        )
        .expect("exact-dead legacy residual metadata must migrate");
    assert!(loaded.has_virtual_residual_runtime());
    assert!(loaded.state_futures(root).contains(0));
    assert!(loaded.state_futures(start).contains(0));
    assert_eq!(loaded.exact_dynamic_state_has_future(root).unwrap(), false);
}

#[test]
fn residual_runtime_restore_rejects_exact_dead_root_without_legacy_provenance() {
    let expression = Expr::Intersect {
        expr: Box::new(Expr::Seq(vec![
            Expr::Repeat {
                expr: Box::new(bytes(b"a")),
                min: 0,
                max: Some(10_000),
            },
            bytes(b"b"),
        ])),
        intersect: Box::new(Expr::Repeat {
            expr: Box::new(bytes(b"a")),
            min: 0,
            max: None,
        }),
    };
    let mut original = Tokenizer::from_parts(DFA::new(1), 1, None);
    original
        .install_virtual_residual_components(vec![(expression.clone(), 0)])
        .expect("dead Boolean residual must install conservatively");
    let metadata = original.virtual_runtime_metadata();
    let root = metadata[0].root_state;

    let mut loaded = serialized_roundtrip(&original);
    let root_finalizers = loaded.state_finalizers(root).clone();
    loaded
        .dfa
        .overwrite_state_metadata(root, root_finalizers, BitSet::new(1));
    let start = loaded.start_state();
    let start_finalizers = loaded.state_finalizers(start).clone();
    loaded
        .dfa
        .overwrite_state_metadata(start, start_finalizers, BitSet::new(1));

    let error = loaded
        .restore_terminal_exprs_with_virtual_runtime_metadata(
            Some(vec![expression]),
            &metadata,
            false,
        )
        .unwrap_err();
    assert!(
        error.contains("inconsistent future metadata"),
        "current-version exact-dead corruption must not be mistaken for legacy metadata: {error}",
    );
    assert!(loaded.virtual_residuals.is_empty());
}

#[test]
fn residual_runtime_restore_rejects_missing_future_for_live_root() {
    let expression = Expr::Seq(vec![
        Expr::Repeat {
            expr: Box::new(Expr::Choice(vec![bytes(b"a"), bytes(b"aa")])),
            min: 0,
            max: Some(10_000),
        },
        bytes(b"z"),
    ]);
    let mut original = Tokenizer::from_parts(DFA::new(1), 1, None);
    original
        .install_virtual_residual_components(vec![(expression.clone(), 0)])
        .expect("live residual must install");
    let metadata = original.virtual_runtime_metadata();
    let root = metadata[0].root_state;
    assert_eq!(original.exact_dynamic_state_has_future(root).unwrap(), true);

    let mut loaded = serialized_roundtrip(&original);
    let root_finalizers = loaded.state_finalizers(root).clone();
    loaded
        .dfa
        .overwrite_state_metadata(root, root_finalizers, BitSet::new(1));
    let start = loaded.start_state();
    let start_finalizers = loaded.state_finalizers(start).clone();
    loaded
        .dfa
        .overwrite_state_metadata(start, start_finalizers, BitSet::new(1));

    let error = loaded
        .restore_terminal_exprs_with_virtual_runtime_metadata(
            Some(vec![expression]),
            &metadata,
            false,
        )
        .unwrap_err();
    assert!(
        error.contains("inconsistent future metadata"),
        "unexpected restoration error: {error}",
    );
    assert!(loaded.virtual_residuals.is_empty());
}

#[test]
fn grouped_summary_execution_matches_independent_scans() {
    let tokenizer = tokenizer_from_exprs(vec![
        bytes(b"ab"),
        plus(bytes(b"a")),
        bytes(b"ba"),
    ]);
    let starts = (0..tokenizer.num_states()).collect::<Vec<_>>();

    enumerate_bytes(b"abx", 4, |input| {
        let groups = tokenizer.execute_summary_groups_from_states(input, &starts);
        let mut grouped_by_start = std::collections::BTreeMap::new();
        for (end_states, matches, support) in groups {
            for start in support {
                assert!(
                    grouped_by_start
                        .insert(start, (end_states.clone(), matches.clone()))
                        .is_none(),
                    "start state {start} appeared in more than one summary group for {input:?}",
                );
            }
        }
        assert_eq!(grouped_by_start.len(), starts.len());
        for &start in &starts {
            assert_eq!(
                grouped_by_start.get(&start),
                Some(&tokenizer.execute_summary_from_state(input, start)),
                "grouped residual scan differs for start state {start}, input {input:?}",
            );
        }
    });
}

#[test]
fn disjoint_union_preserves_every_source_residual_and_unions_resets() {
    let left = tokenizer_from_exprs(vec![bytes(b"ab"), plus(bytes(b"x"))]);
    let right = tokenizer_from_exprs(vec![bytes(b"bc"), plus(bytes(b"y"))]);
    let right_terminal_offset = left.num_terminals();
    let (merged, state_offsets) = Tokenizer::disjoint_union_with_terminal_offsets(&[
        (&left, 0),
        (&right, right_terminal_offset),
    ]);
    assert_eq!(state_offsets.len(), 2);

    enumerate_bytes(b"abcxy", 3, |input| {
        for (source, terminal_offset, state_offset) in [
            (&left, 0u32, state_offsets[0]),
            (&right, right_terminal_offset, state_offsets[1]),
        ] {
            for source_state in 0..source.num_states() {
                let (source_end, source_matches) =
                    normalized_exec(source, input, source_state);
                let expected_end = source_end
                    .into_iter()
                    .map(|state| state_offset + state)
                    .collect::<Vec<_>>();
                let expected_matches = source_matches
                    .into_iter()
                    .map(|(terminal, width, state)| {
                        (terminal_offset + terminal, width, state_offset + state)
                    })
                    .collect::<Vec<_>>();
                assert_eq!(
                    normalized_exec(&merged, input, state_offset + source_state),
                    (expected_end, expected_matches),
                    "residual mismatch from source state {source_state} on {input:?}",
                );
            }
        }

        let mut expected_end = Vec::new();
        let mut expected_matches = Vec::new();
        for (source, terminal_offset, state_offset) in [
            (&left, 0u32, state_offsets[0]),
            (&right, right_terminal_offset, state_offsets[1]),
        ] {
            let (end, matches) = normalized_exec(source, input, source.start_state());
            expected_end.extend(end.into_iter().map(|state| state_offset + state));
            expected_matches.extend(matches.into_iter().map(|(terminal, width, state)| {
                (terminal_offset + terminal, width, state_offset + state)
            }));
        }
        if input.is_empty() {
            // The fresh epsilon dispatcher is a real physical state but
            // contributes no finalizer of its own. It remains in the empty
            // execution closure and disappears after the first byte.
            expected_end.push(merged.start_state());
        }
        expected_end.sort_unstable();
        expected_end.dedup();
        expected_matches.sort_unstable();
        expected_matches.dedup();
        assert_eq!(
            normalized_exec(&merged, input, merged.start_state()),
            (expected_end, expected_matches),
            "reset-union mismatch on {input:?}",
        );
    });
}

#[test]
fn full_determinization_accepts_disjoint_union_local_metadata_widths() {
    let left = tokenizer_from_exprs(vec![bytes(b"ab"), bytes(b"ax")]);
    let right = tokenizer_from_exprs(vec![bytes(b"ab")]);
    let (merged, _) = Tokenizer::disjoint_union_with_terminal_offsets(&[
        (&left, 0),
        (&right, left.num_terminals()),
    ]);
    assert!(merged.has_epsilon_transitions());

    let built = merged
        .try_full_determinization(128, 4_096)
        .expect("small disjoint union must determinize across local metadata widths");
    assert!(!built.tokenizer.has_epsilon_transitions());

    enumerate_bytes(b"abx", 3, |input| {
        let source = merged.execute_from_state(input, merged.initial_state());
        let product = built
            .tokenizer
            .execute_from_state(input, built.tokenizer.initial_state());
        let normalize = |matches: Vec<TokenizerMatch>| {
            let mut matches = matches
                .into_iter()
                .map(|matched| (matched.id, matched.width))
                .collect::<Vec<_>>();
            matches.sort_unstable();
            matches.dedup();
            matches
        };
        assert_eq!(normalize(product.matches), normalize(source.matches));
    });
}

#[test]
fn full_determinization_refuses_virtual_residual_state_space() {
    let mut tokenizer = Tokenizer::from_parts(DFA::new(1), 1, None);
    let expression = Expr::Seq(vec![
        Expr::Repeat {
            expr: Box::new(Expr::Choice(vec![bytes(b"a"), bytes(b"aa")])),
            min: 0,
            max: Some(1_000_000_000),
        },
        bytes(b"z"),
    ]);
    tokenizer
        .install_virtual_residual_components(vec![(expression, 0)])
        .expect("general residual component must install");
    assert!(tokenizer.has_epsilon_transitions());
    assert!(tokenizer.has_virtual_residual_runtime());
    assert!(
        tokenizer.try_full_determinization(128, 4_096).is_none(),
        "physical subset construction must not discard virtual residual states",
    );
}

#[test]
fn virtual_residual_mask_projection_matches_one_token_observations() {
    const HORIZON: usize = 4;
    const MAX: usize = 40;
    let body = Expr::U8Class(U8Set::from_bytes(b"ab"));
    let envelope = Expr::Seq(vec![
        bytes(b"\""),
        Expr::Repeat {
            expr: Box::new(body.clone()),
            min: 1,
            max: Some(MAX),
        },
        bytes(b"\""),
    ]);
    let pattern = Expr::Seq(vec![
        bytes(b"\"a"),
        Expr::Repeat {
            expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"ab"))),
            min: 0,
            max: None,
        },
        bytes(b"\""),
    ]);
    let expression = Expr::Intersect {
        expr: Box::new(envelope),
        intersect: Box::new(pattern),
    };
    let mut exact = Tokenizer::from_parts(DFA::new(1), 1, None);
    exact
        .install_virtual_residual_components_preserving_oracle_coordinates(vec![(expression, 0)])
        .expect("bounded-code residual runtime must install");
    assert_eq!(exact.virtual_residual_bounded_code_liveness_oracle_count(), 1);
    assert_eq!(
        exact.virtual_residual_mask_projection_dense_state_work(HORIZON),
        Some(72),
    );
    let (mask, projections) = exact
        .virtual_residuals_mask_tokenizer(HORIZON)
        .expect("bounded-code residual must admit a finite mask projection");
    assert_eq!(projections.len(), 1);
    assert!(mask.num_states() < 2_000, "small oracle should stay compact enough for the test");
    let projection = &projections[0];
    let root = exact
        .epsilon_closure_states(&[exact.start_state()])
        .into_iter()
        .find(|&state| state != exact.start_state())
        .unwrap();

    // Every exact state reachable in this finite oracle must have a
    // projection, not merely the hand-picked count samples below. This
    // protects the symbolic source-root construction against dropping a
    // valid deep token-boundary coordinate.
    let mut seen = BTreeSet::from([root]);
    let mut queue = VecDeque::from([root]);
    while let Some(state) = queue.pop_front() {
        assert!(
            projection.project(state).is_some(),
            "reachable exact residual state {state} has no finite projection",
        );
        for byte in 0u16..=255 {
            let next = exact.get_transition(state, byte as u8);
            if next != u32::MAX && seen.insert(next) {
                queue.push_back(next);
            }
        }
    }

    let mut starting_states = vec![root];
    for count in [1usize, 2, 5, 12, 30, 35, 38, 39] {
        let mut state = root;
        state = exact.get_transition(state, b'\"');
        assert_ne!(state, u32::MAX);
        for index in 0..count {
            state = exact.get_transition(state, if index == 0 { b'a' } else { b'b' });
            assert_ne!(state, u32::MAX, "count={count} index={index}");
        }
        starting_states.push(state);
    }

    for full_start in starting_states {
        let mask_start = projection
            .project(full_start)
            .expect("every certified exact residual must project");
        assert_eq!(
            exact.matched_terminal_bitset(full_start),
            mask.matched_terminal_bitset(mask_start),
            "source finalizers differ at exact state {full_start}",
        );
        assert_eq!(
            exact.possible_future_terminals(full_start),
            mask.possible_future_terminals(mask_start),
            "source futures differ at exact state {full_start}",
        );

        enumerate_bytes(b"ab\"x", HORIZON, |suffix| {
            let mut full = full_start;
            let mut finite = mask_start;
            for (offset, &byte) in suffix.iter().enumerate() {
                let full_next = exact.get_transition(full, byte);
                let finite_next = mask.get_transition(finite, byte);
                assert_eq!(
                    full_next == u32::MAX,
                    finite_next == u32::MAX,
                    "liveness differs from exact state {full_start} on {suffix:?} at {offset}",
                );
                if full_next == u32::MAX {
                    break;
                }
                assert_eq!(
                    exact.matched_terminal_bitset(full_next),
                    mask.matched_terminal_bitset(finite_next),
                    "finalizers differ from exact state {full_start} on {suffix:?} at {offset}",
                );
                assert_eq!(
                    exact.possible_future_terminals(full_next),
                    mask.possible_future_terminals(finite_next),
                    "futures differ from exact state {full_start} on {suffix:?} at {offset}",
                );
                // Runtime commit reprojects at token boundaries. The finite
                // byte-walk endpoint need not be the same raw mask state,
                // but it must carry the same observation as the independently
                // projected exact endpoint.
                let reprojected = projection.project(full_next).unwrap();
                assert_eq!(
                    mask.matched_terminal_bitset(finite_next),
                    mask.matched_terminal_bitset(reprojected),
                );
                assert_eq!(
                    mask.possible_future_terminals(finite_next),
                    mask.possible_future_terminals(reprojected),
                );
                full = full_next;
                finite = finite_next;
            }
        });
    }
}

#[test]
fn prepared_virtual_residual_install_matches_direct_dynamic_install() {
    let body = Expr::U8Class(U8Set::from_bytes(b"ab"));
    let envelope = Expr::Seq(vec![
        bytes(b"\""),
        Expr::Repeat {
            expr: Box::new(body.clone()),
            min: 0,
            max: Some(4),
        },
        bytes(b"\""),
    ]);
    let pattern = Expr::Seq(vec![
        bytes(b"\"a"),
        Expr::Repeat {
            expr: Box::new(body),
            min: 0,
            max: None,
        },
        bytes(b"\""),
    ]);
    let expression = Expr::Intersect {
        expr: Box::new(envelope),
        intersect: Box::new(pattern),
    };

    let mut direct = Tokenizer::from_parts(DFA::new(1), 1, None);
    direct
        .install_virtual_residual_components(vec![(expression.clone(), 0)])
        .expect("direct dynamic residual must install");

    let prepared = Tokenizer::prepare_virtual_residual_components(vec![(expression, 0)])
        .expect("bounded-code residual must prepare");
    let mut overlapped = Tokenizer::from_parts(DFA::new(1), 1, None);
    overlapped
        .install_prepared_virtual_residual_components(prepared)
        .expect("prepared dynamic residual must install");

    assert_eq!(direct.virtual_runtime_metadata(), overlapped.virtual_runtime_metadata());
    assert_eq!(
        direct.virtual_residual_bounded_code_liveness_oracle_count(),
        overlapped.virtual_residual_bounded_code_liveness_oracle_count(),
    );
    enumerate_bytes(b"ab\"", 4, |input| {
        assert_eq!(
            normalized_exec(&direct, input, direct.start_state()),
            normalized_exec(&overlapped, input, overlapped.start_state()),
            "prepared residual runtime differs on {input:?}",
        );
    });
}

#[test]
fn virtual_residual_mask_projection_preserves_fast_loaded_physical_futures() {
    const HORIZON: usize = 4;
    let ordinary = bytes(b"<");
    let residual = Expr::Intersect {
        expr: Box::new(Expr::Seq(vec![
            bytes(b"\""),
            Expr::Repeat {
                expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"ab"))),
                min: 0,
                max: None,
            },
            bytes(b"\""),
        ])),
        intersect: Box::new(Expr::Seq(vec![
            bytes(b"\""),
            Expr::Repeat {
                expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"ab"))),
                min: 1,
                max: Some(128),
            },
            bytes(b"\""),
        ])),
    };
    let expressions = vec![ordinary.clone(), residual.clone()];

    // Build terminal 0 physically and reserve terminal 1 for the exact
    // residual runtime, matching the hybrid dynamic-tokenizer layout.
    let mut exact = tokenizer_from_exprs(vec![
        ordinary,
        Expr::U8Class(U8Set::empty()),
    ]);
    exact
        .restore_terminal_exprs_without_virtual_runtime(Some(expressions.clone()))
        .unwrap();
    exact
        .install_virtual_residual_components(vec![(residual, 1)])
        .expect("bounded-code residual runtime must install");
    let metadata = exact.virtual_runtime_metadata();
    assert_eq!(metadata.len(), 1);
    assert!(exact.possible_future_terminals(exact.start_state()).contains(0));

    // Current fast loads keep ordinary byte transitions and observation
    // metadata packed. Projection construction must structurally replace
    // the residual proxy without losing the packed physical language.
    let wire = artifact_serde::to_fast_bytes(&exact);
    let mut loaded = artifact_serde::from_fast_bytes(&wire).expect("fast tokenizer roundtrip");
    assert!(loaded.has_packed_runtime_transitions());
    assert!(loaded.has_packed_runtime_metadata());
    loaded
        .restore_terminal_exprs_with_virtual_runtime_metadata(
            Some(expressions),
            &metadata,
            false,
        )
        .expect("residual runtime must restore from explicit metadata");
    assert!(loaded.possible_future_terminals(loaded.start_state()).contains(0));

    let (mask, projections) = loaded
        .virtual_residuals_mask_tokenizer(HORIZON)
        .expect("fast-loaded residual tokenizer must admit a finite mask projection");
    assert_eq!(projections.len(), 1);
    assert!(
        mask.possible_future_terminals(mask.start_state()).contains(0),
        "projection must preserve ordinary physical futures stored in packed transition rows",
    );
    assert_eq!(
        normalized_exec(&mask, b"<", mask.start_state()),
        normalized_exec(&loaded, b"<", loaded.start_state()),
        "projection changed the ordinary physical terminal language",
    );
}

#[test]
fn virtual_residual_mask_projection_accepts_full_bound_stencil() {
    const HORIZON: usize = 4;
    const MAX: usize = 3;
    let body = Expr::U8Class(U8Set::from_bytes(b"ab"));
    let envelope = Expr::Seq(vec![
        bytes(b"\""),
        Expr::Repeat {
            expr: Box::new(body),
            min: 0,
            max: Some(MAX),
        },
        bytes(b"\""),
    ]);
    let pattern = Expr::Seq(vec![
        bytes(b"\""),
        Expr::Repeat {
            expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"ab"))),
            min: 0,
            max: None,
        },
        bytes(b"\""),
    ]);
    let expression = Expr::Intersect {
        expr: Box::new(envelope),
        intersect: Box::new(pattern),
    };
    let mut exact = Tokenizer::from_parts(DFA::new(1), 1, None);
    exact
        .install_virtual_residual_components_preserving_oracle_coordinates(vec![(expression, 0)])
        .expect("bounded-code residual runtime must install");
    // The merged bounded-code oracle drops the redundant unbounded copy of
    // this same envelope from its pattern coordinate. That leaves a
    // one-state universal pattern coordinate, reducing the dense stencil
    // work from the old 30 states to 10 without changing the exact
    // one-token projection checked exhaustively below.
    assert_eq!(
        exact.virtual_residual_mask_projection_dense_state_work(HORIZON),
        Some(10),
    );
    let (mask, projections) = exact
        .virtual_residuals_mask_tokenizer(HORIZON)
        .expect("a cheap full-bound finite projection must remain available");
    let projection = &projections[0];
    let root = exact
        .epsilon_closure_states(&[exact.start_state()])
        .into_iter()
        .find(|&state| state != exact.start_state())
        .unwrap();

    let mask_root = projection.project(root).unwrap();
    enumerate_bytes(b"ab\"x", HORIZON, |suffix| {
        let mut full = root;
        let mut finite = mask_root;
        for &byte in suffix {
            let full_next = exact.get_transition(full, byte);
            let finite_next = mask.get_transition(finite, byte);
            assert_eq!(full_next == u32::MAX, finite_next == u32::MAX);
            if full_next == u32::MAX {
                break;
            }
            assert_eq!(
                exact.matched_terminal_bitset(full_next),
                mask.matched_terminal_bitset(finite_next),
            );
            assert_eq!(
                exact.possible_future_terminals(full_next),
                mask.possible_future_terminals(finite_next),
            );
            full = full_next;
            finite = finite_next;
        }
    });
}

#[test]
fn residual_state_owner_index_dispatches_multiple_components_directly() {
    let repeated_suffix = |byte, suffix| {
        Expr::Seq(vec![
            Expr::Repeat {
                expr: Box::new(Expr::Choice(vec![
                    bytes(&[byte]),
                    bytes(&[byte, byte]),
                ])),
                min: 0,
                max: Some(1_000_000_000),
            },
            bytes(&[suffix]),
        ])
    };
    let mut tokenizer = Tokenizer::from_parts(DFA::new(1), 2, None);
    tokenizer
        .install_virtual_residual_components(vec![
            (repeated_suffix(b'a', b'z'), 0),
            (repeated_suffix(b'b', b'y'), 1),
        ])
        .expect("two general residual components must install");

    let a_state = tokenizer.step(1, b'a').expect("first residual must advance");
    let b_state = tokenizer.step(2, b'b').expect("second residual must advance");
    assert_ne!(a_state, b_state);
    assert_eq!(
        tokenizer
            .virtual_residual_runtime_for_state(a_state)
            .map(VirtualResidualRuntime::terminal),
        Some(0),
    );
    assert_eq!(
        tokenizer
            .virtual_residual_runtime_for_state(b_state)
            .map(VirtualResidualRuntime::terminal),
        Some(1),
    );

    let aa_state = tokenizer
        .step(a_state, b'a')
        .expect("interleaved first residual must remain owned");
    let bb_state = tokenizer
        .step(b_state, b'b')
        .expect("interleaved second residual must remain owned");
    assert_eq!(
        tokenizer
            .virtual_residual_runtime_for_state(aa_state)
            .map(VirtualResidualRuntime::terminal),
        Some(0),
    );
    assert_eq!(
        tokenizer
            .virtual_residual_runtime_for_state(bb_state)
            .map(VirtualResidualRuntime::terminal),
        Some(1),
    );
}

use crate::automata::lexer::dfa::DFA;

#[test]
fn compressed_transition_entries_preserve_pair_sequence_wire_format() {
    let legacy = vec![(0u8, 7u32), (3, 11), (49, 782_231)];
    let entries = CompressedTransitionEntries::from_parts(
        legacy.iter().map(|&(class, _)| class).collect(),
        legacy.iter().map(|&(_, target)| target).collect(),
    );
    assert_eq!(
        bincode::serialize(&entries).unwrap(),
        bincode::serialize(&legacy).unwrap(),
    );
    let decoded: CompressedTransitionEntries =
        bincode::deserialize(&bincode::serialize(&legacy).unwrap()).unwrap();
    assert_eq!(decoded.iter_range(0, decoded.len()).collect::<Vec<_>>(), legacy);
}

fn dispatch_prefix_tokenizer(with_appended_residual: bool) -> Tokenizer {
    let mut dfa = DFA::new(if with_appended_residual { 5 } else { 4 });
    dfa.ensure_group_capacity(1);
    dfa.add_epsilon_transition(0, 1);
    dfa.add_epsilon_transition(0, 3);
    dfa.add_transition(1, b'a', 2);
    dfa.add_transition(2, b'a', 2);
    dfa.add_transition(3, b'x', 3);
    let mut accepting = BitSet::new(1);
    accepting.set(0);
    dfa.overwrite_state_metadata(2, accepting.clone(), BitSet::new(1));
    if with_appended_residual {
        // This state is deliberately not reset-reachable. It models the
        // externally-entered residuals appended by structural synthesis.
        dfa.add_transition(4, b'a', 2);
        dfa.add_transition(4, b'b', 4);
        dfa.overwrite_state_metadata(4, accepting, BitSet::new(1));
    }
    dfa.recompute_possible_futures();
    Tokenizer::from_parts(dfa, 1, None)
}

#[test]
fn deterministic_dispatch_execution_enters_roots_without_retaining_dispatcher() {
    let tokenizer = dispatch_prefix_tokenizer(false);
    assert_eq!(tokenizer.deterministic_dispatch_roots(), Some(&[1, 3][..]));
    assert!(tokenizer.has_scalar_deterministic_dispatch());

    let empty = tokenizer.execute_from_state_end_only(b"", tokenizer.initial_state_id());
    assert_eq!(empty.as_slice(), &[0, 1, 3]);

    let a = tokenizer.execute_from_state(b"a", tokenizer.initial_state_id());
    assert!(a.matches.iter().any(|matched| matched.id == 0 && matched.width == 1));
    assert_eq!(a.end_state.as_slice(), &[2]);

    let x = tokenizer.execute_from_state_end_only(b"x", tokenizer.initial_state_id());
    assert_eq!(x.as_slice(), &[3]);
}

#[test]
fn huge_wire_preserves_preproven_scalar_dispatch_for_compressed_suffix() {
    let mut tokenizer = dispatch_prefix_tokenizer(false);
    let mut byte_to_class = vec![u8::MAX; 256];
    byte_to_class[b'a' as usize] = 0;
    byte_to_class[b'x' as usize] = 1;
    tokenizer.compressed_transition_segments = Arc::from([CompressedTransitionSegment {
        state_offset: 1,
        state_count: 3,
        byte_to_class: Arc::from(byte_to_class.into_boxed_slice()),
        class_members: Arc::from(
            vec![vec![b'a'].into_boxed_slice(), vec![b'x'].into_boxed_slice()]
                .into_boxed_slice(),
        ),
        row_offsets: Arc::from([0u32, 1, 2, 3]),
        entries: CompressedTransitionEntries::from_parts(
            vec![0, 0, 1],
            vec![1, 1, 2],
        ),
        expanded_transition_count: 3,
    }]);
    for state in &mut tokenizer.dfa.states_mut()[1..] {
        state.transitions.clear();
    }
    tokenizer.invalidate_derived_caches();
    assert!(tokenizer.has_scalar_deterministic_dispatch());
    assert_eq!(tokenizer.scalar_deterministic_dispatch_cache.get(), Some(&true));

    let wire = artifact_serde::build_huge_bytes(&tokenizer).expect("scalar TKS3 fixture");
    let loaded = artifact_serde::from_fast_bytes(&wire).expect("scalar TKS3 roundtrip");
    assert_eq!(
        loaded.scalar_deterministic_dispatch_cache.get(),
        Some(&true),
        "TKS3 must restore the worker-side scalar-dispatch proof without rescanning",
    );
    assert!(loaded.has_scalar_deterministic_dispatch());
    enumerate_bytes(b"ax", 3, |input| {
        assert_eq!(
            normalized_exec(&loaded, input, loaded.initial_state_id()),
            normalized_exec(&tokenizer, input, tokenizer.initial_state_id()),
            "scalar TKS3 execution mismatch on {input:?}",
        );
    });

    let reencoded = artifact_serde::build_huge_bytes(&loaded)
        .expect("loaded scalar TKS3 tokenizer must remain compactly serializable");
    let reloaded = artifact_serde::from_fast_bytes(&reencoded)
        .expect("re-encoded scalar TKS3 roundtrip");
    assert_eq!(
        reloaded.scalar_deterministic_dispatch_cache.get(),
        Some(&true),
        "TKS3 re-encoding must preserve the preproven scalar-dispatch bit",
    );
}

#[test]
fn fast_roundtrip_preserves_scalar_deterministic_dispatch() {
    let tokenizer = dispatch_prefix_tokenizer(false);
    assert!(tokenizer.has_scalar_deterministic_dispatch());
    assert_eq!(tokenizer.terminal_dispatch_root_candidate(0), Some(1));

    let wire = artifact_serde::to_fast_bytes_with_packed_metadata(&tokenizer);
    let loaded = artifact_serde::from_fast_bytes(&wire).expect("fast tokenizer roundtrip");

    assert_eq!(loaded.deterministic_dispatch_roots(), Some(&[1, 3][..]));
    assert_eq!(
        loaded.scalar_deterministic_dispatch_cache.get(),
        Some(&true),
        "TKF3 must restore the worker-side scalar-dispatch proof without rescanning",
    );
    assert!(
        loaded.has_scalar_deterministic_dispatch(),
        "packed runtime storage must not change the reset-dispatch topology proof",
    );
    assert_eq!(
        loaded.terminal_dispatch_root_candidate(0),
        Some(1),
        "packed runtime storage must preserve terminal-specific dispatch-root discovery",
    );
}

#[test]
fn matched_terminal_iterator_does_not_materialize_all_state_cache() {
    let tokenizer = dispatch_prefix_tokenizer(false);
    assert!(tokenizer.matched_terminals_cache.get().is_none());

    assert_eq!(tokenizer.matched_terminals_iter(2).collect::<Vec<_>>(), vec![0]);
    assert!(
        tokenizer.matched_terminals_cache.get().is_none(),
        "per-state iteration must not build the all-state matched-terminal cache",
    );

    assert_eq!(tokenizer.matched_terminals_slice(2), &[0]);
    assert!(tokenizer.matched_terminals_cache.get().is_some());
}

#[test]
fn whole_tokenizer_caches_are_reused_and_invalidated_after_mutation() {
    let source = dispatch_prefix_tokenizer(true);
    let rebuilt = dispatch_prefix_tokenizer(false);
    let mut local = rebuilt.clone();

    let loops_before = local.all_self_loop_bytes();
    let loops_before_again = local.all_self_loop_bytes();
    assert!(Arc::ptr_eq(&loops_before, &loops_before_again));
    assert!(loops_before[2].contains(b'a'));
    assert!(loops_before[3].contains(b'x'));
    let closures_before = local.all_singleton_epsilon_closures();
    let transition_count_before = local.transition_count();
    assert!(local.scalar_deterministic_dispatch_cache.get().is_none());
    let scalar_dispatch_before = local.has_scalar_deterministic_dispatch();
    assert_eq!(
        local.scalar_deterministic_dispatch_cache.get(),
        Some(&scalar_dispatch_before),
    );

    let rebuilt_to_local = (0..rebuilt.num_states()).collect::<Vec<_>>();
    local
        .augment_from_verified_component_prefixes(
            &source,
            &rebuilt,
            &rebuilt_to_local,
        )
        .expect("verified append-only component relation");

    assert!(local.scalar_deterministic_dispatch_cache.get().is_none());
    let scalar_dispatch_after = local.has_scalar_deterministic_dispatch();
    assert_eq!(
        local.scalar_deterministic_dispatch_cache.get(),
        Some(&scalar_dispatch_after),
    );
    let loops_after = local.all_self_loop_bytes();
    assert!(!Arc::ptr_eq(&loops_before, &loops_after));
    assert!(loops_after[4].contains(b'b'));
    assert!(!Arc::ptr_eq(
        &closures_before,
        &local.all_singleton_epsilon_closures(),
    ));
    assert!(local.transition_count() > transition_count_before);
}

#[test]
fn precomputed_bounded_observation_sets_respect_16_and_64_horizons() {
    const BAD: u32 = 21;
    let mut dfa = DFA::new((BAD + 1) as usize);
    dfa.ensure_group_capacity(2);

    let stable_finalizers = BitSet::new(2);
    let mut stable_futures = BitSet::new(2);
    stable_futures.set(0);
    let mut bad_futures = BitSet::new(2);
    bad_futures.set(1);

    let string_bytes = (0x20u8..=0x7e)
        .filter(|&byte| !matches!(byte, b'"' | b'\\'))
        .collect::<Vec<_>>();
    for state in 0..BAD {
        dfa.overwrite_state_metadata(
            state,
            stable_finalizers.clone(),
            stable_futures.clone(),
        );
        for &byte in &string_bytes {
            if byte == b'0' {
                dfa.add_transition(state, byte, state);
            } else {
                dfa.add_transition(state, byte, state + 1);
            }
        }
    }
    dfa.overwrite_state_metadata(BAD, BitSet::new(2), bad_futures);
    dfa.add_transition(BAD, b'0', BAD);

    let mut tokenizer = Tokenizer::from_parts(dfa, 2, None);
    let advancing_bytes = string_bytes
        .iter()
        .copied()
        .filter(|&byte| byte != b'0')
        .collect::<Vec<_>>();
    // Deliberately split the advancing family across two byte classes.
    // The precompute must recover the larger semantic family by grouping
    // classes with the same destination rather than selecting one class.
    let advancing_a = advancing_bytes
        .iter()
        .copied()
        .filter(|byte| byte & 1 == 0)
        .collect::<Vec<_>>();
    let advancing_b = advancing_bytes
        .iter()
        .copied()
        .filter(|byte| byte & 1 != 0)
        .collect::<Vec<_>>();
    let mut byte_to_class = vec![3u8; 256];
    for &byte in &advancing_a {
        byte_to_class[byte as usize] = 0;
    }
    for &byte in &advancing_b {
        byte_to_class[byte as usize] = 1;
    }
    byte_to_class[b'0' as usize] = 2;
    let class_members = vec![
        advancing_a.into_boxed_slice(),
        advancing_b.into_boxed_slice(),
        vec![b'0'].into_boxed_slice(),
        (0u16..=255)
            .map(|byte| byte as u8)
            .filter(|byte| !advancing_bytes.contains(byte) && *byte != b'0')
            .collect::<Vec<_>>()
            .into_boxed_slice(),
    ];
    let mut row_offsets = Vec::<u32>::with_capacity((BAD + 2) as usize);
    let mut classes = Vec::<u8>::new();
    let mut targets = Vec::<u32>::new();
    row_offsets.push(0);
    for state in 0..BAD {
        classes.extend([0, 1, 2]);
        targets.extend([state + 1, state + 1, state]);
        row_offsets.push(classes.len() as u32);
    }
    classes.push(2);
    targets.push(BAD);
    row_offsets.push(classes.len() as u32);
    tokenizer.compressed_transition_segments = Arc::from([CompressedTransitionSegment {
        state_offset: 0,
        state_count: BAD + 1,
        byte_to_class: Arc::from(byte_to_class.into_boxed_slice()),
        class_members: Arc::from(class_members.into_boxed_slice()),
        row_offsets: Arc::from(row_offsets.into_boxed_slice()),
        entries: CompressedTransitionEntries::from_parts(classes, targets),
        expanded_transition_count: BAD as usize * 93 + 1,
    }]);
    let (safe16, safe64) = tokenizer.precompute_bounded_observation_safe_byte_sets();

    // The 92 advancing bytes all co-target the same state and are stable
    // for 16 steps from state 0, despite being split across two tokenizer
    // classes. They can reach the observation-changing state before 64
    // steps, so H64 falls back to the infinitely-safe literal self-loop.
    assert!(safe16[0].contains(b'a'));
    assert!(safe16[0].contains(b'Z'));
    assert!(safe16[0].contains(b'_'));
    assert!(!safe16[0].contains(b'0'));
    assert_eq!(safe16[0].len(), 92);
    assert!(!safe64[0].contains(b'a'));
    assert!(safe64[0].contains(b'0'));

    // Cross-check the advertised sets against the generic exact reference
    // proof using the complete terminal domain as the observation mask.
    let mut all_terminals = BitSet::new(2);
    all_terminals.set(0);
    all_terminals.set(1);
    assert_eq!(
        tokenizer.bounded_observation_safe_horizon_from_state(
            0,
            safe16[0],
            &all_terminals,
            16,
        ),
        16,
    );
    assert_eq!(
        tokenizer.bounded_observation_safe_horizon_from_state(
            0,
            safe64[0],
            &all_terminals,
            64,
        ),
        64,
    );
}

#[test]
fn failed_or_noop_mutations_preserve_derived_caches() {
    let mut tokenizer = dispatch_prefix_tokenizer(false);
    let loops = tokenizer.all_self_loop_bytes();
    let closures = tokenizer.all_singleton_epsilon_closures();

    assert!(tokenizer
        .isolate_start_state_and_drain_nullable_terminals()
        .is_empty());
    assert!(Arc::ptr_eq(&loops, &tokenizer.all_self_loop_bytes()));
    assert!(Arc::ptr_eq(
        &closures,
        &tokenizer.all_singleton_epsilon_closures(),
    ));

    let incompatible = Tokenizer::from_parts(DFA::new(1), 2, None);
    assert!(tokenizer
        .augment_from_verified_component_prefixes(&incompatible, &tokenizer.clone(), &[])
        .is_none());
    assert!(Arc::ptr_eq(&loops, &tokenizer.all_self_loop_bytes()));
}

#[test]
fn compressed_bounded_observation_certificate_matches_generic_reference() {
    let mut dfa = DFA::new(4);
    dfa.ensure_group_capacity(1);
    let mut live = BitSet::new(1);
    live.set(0);
    for state in 0..3u32 {
        dfa.overwrite_state_metadata(state, BitSet::new(1), live.clone());
    }
    dfa.overwrite_state_metadata(3, live.clone(), live.clone());

    let mut tokenizer = Tokenizer::from_parts(dfa, 1, None);
    let byte_to_class = vec![0u8; 256];
    let class_members = vec![(0u16..=255)
        .map(|byte| byte as u8)
        .collect::<Vec<_>>()
        .into_boxed_slice()];
    tokenizer.compressed_transition_segments = Arc::from([CompressedTransitionSegment {
        state_offset: 0,
        state_count: 4,
        byte_to_class: Arc::from(byte_to_class.into_boxed_slice()),
        class_members: Arc::from(class_members.into_boxed_slice()),
        row_offsets: Arc::from([0u32, 1, 2, 3, 4]),
        entries: CompressedTransitionEntries::from_parts(
            vec![0, 0, 0, 0],
            vec![1, 2, 3, 3],
        ),
        expanded_transition_count: 4 * 256,
    }]);

    let mut bytes = U8Set::empty();
    bytes.insert(b'a');
    bytes.insert(b'b');
    let mut active = BitSet::new(1);
    active.set(0);

    for max_horizon in 1..=8 {
        let optimized = tokenizer.bounded_observation_safe_horizon_from_state(
            0,
            bytes,
            &active,
            max_horizon,
        );
        let reference = <Tokenizer as Lexer>::bounded_observation_safe_horizon_from_state(
            &tokenizer,
            0,
            bytes,
            &active,
            max_horizon,
        );
        assert_eq!(optimized, reference, "max_horizon={max_horizon}");
    }
    assert_eq!(
        tokenizer.bounded_observation_safe_horizon_from_state(0, bytes, &active, 8),
        2,
    );

    let (source_horizon, witnesses) = tokenizer
        .bounded_observation_safe_horizon_with_witnesses(0, bytes, &active, 8);
    assert_eq!(source_horizon, 2);
    for (state, depth) in witnesses {
        if depth > source_horizon {
            continue;
        }
        let inherited = source_horizon - depth;
        let reference = <Tokenizer as Lexer>::bounded_observation_safe_horizon_from_state(
            &tokenizer,
            state,
            bytes,
            &active,
            inherited,
        );
        assert_eq!(
            reference, inherited,
            "witness state={state} depth={depth} did not inherit its conservative bound",
        );
    }
}

#[test]
fn deterministic_reset_dispatch_does_not_certify_a_later_epsilon_frontier() {
    let mut dfa = DFA::new(6);
    dfa.ensure_group_capacity(1);
    dfa.add_epsilon_transition(0, 1);
    dfa.add_epsilon_transition(0, 4);
    dfa.add_transition(1, b'a', 2);
    dfa.add_epsilon_transition(2, 3);
    dfa.add_transition(3, b'b', 3);
    dfa.add_transition(4, b'x', 5);
    let mut accepting = BitSet::new(1);
    accepting.set(0);
    dfa.overwrite_state_metadata(3, accepting.clone(), BitSet::new(1));
    dfa.overwrite_state_metadata(5, accepting, BitSet::new(1));
    dfa.recompute_possible_futures();

    let tokenizer = Tokenizer::from_parts(dfa, 1, None);
    assert_eq!(tokenizer.deterministic_dispatch_roots(), Some(&[1, 4][..]));
    assert!(tokenizer.has_deterministic_dispatch());
    assert!(!tokenizer.has_scalar_deterministic_dispatch());

    let result = tokenizer.execute_from_state_end_only(b"ab", tokenizer.initial_state_id());
    assert_eq!(result.as_slice(), &[3]);
}

#[test]
fn structural_prefix_augmentation_rejects_a_non_homomorphic_target_prefix() {
    let source = dispatch_prefix_tokenizer(true);
    let rebuilt = dispatch_prefix_tokenizer(false);
    let mut local = rebuilt.clone();
    local
        .dfa
        .set_transitions_from_sorted_entries(2, vec![(b'a', 3)]);
    let rebuilt_to_local = (0..rebuilt.num_states()).collect::<Vec<_>>();

    assert!(local
        .augment_from_verified_component_prefixes(
            &source,
            &rebuilt,
            &rebuilt_to_local,
        )
        .is_none());
}

#[test]
fn structural_prefix_augmentation_clones_only_appended_residuals() {
    let source = dispatch_prefix_tokenizer(true);
    let rebuilt = dispatch_prefix_tokenizer(false);
    let mut local = rebuilt.clone();
    let rebuilt_to_local = (0..rebuilt.num_states()).collect::<Vec<_>>();

    let source_to_local = local
        .augment_from_verified_component_prefixes(
            &source,
            &rebuilt,
            &rebuilt_to_local,
        )
        .expect("verified append-only component relation");

    assert_eq!(local.num_states(), source.num_states());
    assert_eq!(source_to_local, vec![0, 1, 2, 3, 4]);
    for input in [b"".as_slice(), b"a", b"b", b"ba", b"bba"] {
        let source_result = source.execute_from_state_all_widths(input, 4);
        let local_result = local.execute_from_state_all_widths(input, source_to_local[4]);
        assert_eq!(source_result.matches, local_result.matches, "input={input:?}");
        assert_eq!(source_result.end_state, local_result.end_state, "input={input:?}");
    }
}

#[test]
fn matched_terminal_cache_is_invalidated_by_alias_canonicalization() {
    let mut dfa = DFA::new(1);
    dfa.ensure_group_capacity(2);
    let mut alias = BitSet::new(2);
    alias.set(1);
    dfa.overwrite_state_metadata(0, alias, BitSet::new(2));
    dfa.recompute_possible_futures();

    let mut tokenizer = Tokenizer::from_parts(dfa, 2, None);
    assert_eq!(tokenizer.matched_terminals_slice(0), &[1]);
    tokenizer.canonicalize_terminal_aliases(0, &[1]);
    assert_eq!(tokenizer.matched_terminals_slice(0), &[0]);
}

#[test]
fn owned_parent_union_invalidates_preinitialized_runtime_caches() {
    fn one_byte(byte: u8) -> Tokenizer {
        let mut dfa = DFA::new(2);
        dfa.ensure_group_capacity(1);
        dfa.add_transition(0, byte, 1);
        let mut accepting = BitSet::new(1);
        accepting.set(0);
        dfa.overwrite_state_metadata(1, accepting, BitSet::new(1));
        dfa.recompute_possible_futures();
        Tokenizer::from_parts(dfa, 1, None)
    }

    let left = one_byte(b'a');
    let right = one_byte(b'b');
    assert_eq!(left.matched_terminals_slice(1), &[0]);
    assert!(left.initial_byte_frontiers()[b'a' as usize].contains(&1));

    let (merged, offsets) =
        Tokenizer::disjoint_union_with_owned_parent(left, 0, &[(&right, 1)]);
    let right_accept = offsets[1] + 1;
    assert_eq!(merged.matched_terminals_slice(right_accept), &[1]);
    assert!(merged.initial_byte_frontiers()[b'b' as usize].contains(&right_accept));
}

#[test]
fn owned_parent_union_preserves_fast_loaded_parent_root_and_new_epsilon_child() {
    fn one_byte(byte: u8) -> Tokenizer {
        let mut dfa = DFA::new(2);
        dfa.ensure_group_capacity(1);
        dfa.add_transition(0, byte, 1);
        let mut accepting = BitSet::new(1);
        accepting.set(0);
        dfa.overwrite_state_metadata(1, accepting, BitSet::new(1));
        dfa.recompute_possible_futures();
        Tokenizer::from_parts(dfa, 1, None)
    }

    let parent = one_byte(b'<');
    let left = one_byte(b'a');
    let (half, _) = Tokenizer::disjoint_union_with_owned_parent(parent, 0, &[(&left, 1)]);
    let wire = artifact_serde::to_fast_bytes(&half);
    let loaded = artifact_serde::from_fast_bytes(&wire).expect("fast tokenizer roundtrip");
    assert!(loaded.packed_runtime_transitions.is_some());
    assert!(
        loaded.packed_runtime_metadata.is_some(),
        "current fast loads keep observation metadata packed until structural mutation",
    );
    assert_eq!(
        loaded.dfa.num_states(),
        1,
        "loaded structural DFA should remain a stub",
    );

    let right = one_byte(b'b');
    let (nested, offsets) =
        Tokenizer::disjoint_union_with_owned_parent(loaded, 0, &[(&right, 2)]);
    assert_eq!(offsets[0], 0);
    assert_eq!(nested.deterministic_reset_states().as_slice(), &[0]);
    assert!(
        nested.packed_runtime_transitions.is_some(),
        "structural mutation must not expand the loaded parent's packed byte rows",
    );

    for (input, terminal) in [(b"<".as_slice(), 0), (b"a", 1), (b"b", 2)] {
        let result = nested.execute_from_state(input, nested.initial_state_id());
        assert!(
            result
                .matches
                .iter()
                .any(|matched| matched.id == terminal && matched.width == 1),
            "missing terminal {terminal} for {input:?}: {result:?}",
        );
    }
}

#[test]
fn execution_handles_epsilon_edges_before_and_after_a_byte() {
    let mut dfa = DFA::new(6);
    dfa.ensure_group_capacity(2);
    dfa.add_epsilon_transition(0, 1);
    dfa.add_epsilon_transition(1, 2);
    dfa.add_epsilon_transition(2, 1);
    dfa.add_transition(1, b'a', 3);
    dfa.add_transition(2, b'a', 4);
    dfa.add_epsilon_transition(3, 5);

    let mut terminal_zero = BitSet::new(2);
    terminal_zero.set(0);
    dfa.overwrite_state_metadata(5, terminal_zero, BitSet::new(2));
    let mut terminal_one = BitSet::new(2);
    terminal_one.set(1);
    dfa.overwrite_state_metadata(4, terminal_one, BitSet::new(2));
    dfa.recompute_possible_futures();

    let tokenizer = Tokenizer::from_parts(dfa, 2, None);
    let execution = tokenizer.execute_from_state_all_widths(b"a", 0);
    let mut matches = execution
        .matches
        .iter()
        .map(|matched| (matched.id, matched.width))
        .collect::<Vec<_>>();
    matches.sort_unstable();
    assert_eq!(matches, vec![(0, 1), (1, 1)]);
    assert!(execution.end_state.is_empty());
    let longest = tokenizer.execute_from_state(b"a", 0);
    assert_eq!(longest.end_state.as_slice(), &[3, 4, 5]);
    assert_eq!(tokenizer.matched_terminals(3), BTreeSet::from([0]));

    let interests = BitSet::all(2);
    let (matched, continuation) =
        tokenizer.scan_terminal_matches_from_state(b"a", 0, &interests);
    assert!(matched.contains(0));
    assert!(matched.contains(1));
    assert!(continuation.is_empty());
}

#[test]
fn draining_nullable_initial_closure_preserves_later_root_matches() {
    let mut dfa = DFA::new(2);
    dfa.ensure_group_capacity(1);
    dfa.add_epsilon_transition(0, 1);
    dfa.add_transition(1, b'a', 1);
    let mut accepting = BitSet::new(1);
    accepting.set(0);
    dfa.overwrite_state_metadata(1, accepting, BitSet::new(1));
    dfa.recompute_possible_futures();

    let mut tokenizer = Tokenizer::from_parts(dfa, 1, None);
    assert_eq!(tokenizer.matched_terminals(0), BTreeSet::from([0]));
    assert_eq!(
        tokenizer.isolate_start_state_and_drain_nullable_terminals(),
        BTreeSet::from([0]),
    );
    assert!(tokenizer.matched_terminals(0).is_empty());

    let one = tokenizer.execute_from_state(b"a", tokenizer.initial_state());
    assert!(one.matches.iter().any(|matched| matched.id == 0 && matched.width == 1));
    let two = tokenizer.execute_from_state(b"aa", tokenizer.initial_state());
    assert!(two.matches.iter().any(|matched| matched.id == 0 && matched.width == 2));
}

#[test]
fn longest_match_preserves_every_accepting_end_state_for_one_terminal() {
    let mut dfa = DFA::new(5);
    dfa.ensure_group_capacity(1);
    dfa.add_epsilon_transition(0, 1);
    dfa.add_epsilon_transition(0, 2);
    dfa.add_transition(1, b'a', 3);
    dfa.add_transition(2, b'a', 4);
    let mut accepting = BitSet::new(1);
    accepting.set(0);
    dfa.overwrite_state_metadata(3, accepting.clone(), BitSet::new(1));
    dfa.overwrite_state_metadata(4, accepting, BitSet::new(1));
    dfa.recompute_possible_futures();

    let tokenizer = Tokenizer::from_parts(dfa, 1, None);
    let mut end_states = tokenizer
        .execute_from_state(b"a", 0)
        .matches
        .into_iter()
        .filter(|matched| matched.id == 0 && matched.width == 1)
        .map(|matched| matched.end_state)
        .collect::<Vec<_>>();
    end_states.sort_unstable();
    assert_eq!(end_states, vec![3, 4]);
}

#[test]
fn full_determinization_is_exact_subset_construction() {
    let source = arbitrary_epsilon_l1_test_tokenizer();
    let built = source
        .try_full_determinization(128, 4_096)
        .expect("small epsilon tokenizer must fully determinize");
    let deterministic = &built.tokenizer;

    assert!(!deterministic.has_epsilon_transitions());
    assert_eq!(
        built.source_subsets.len(),
        deterministic.num_states() as usize,
    );

    for state in 0..deterministic.num_states() {
        let subset = &built.source_subsets[state as usize];
        let mut finalizers = BitSet::new(source.num_terminals() as usize);
        let mut futures = BitSet::new(source.num_terminals() as usize);
        for &source_state in subset.iter() {
            finalizers.union_with(source.dfa.finalizers(source_state));
            futures.union_with(source.dfa.possible_future_group_ids(source_state));
        }
        assert_eq!(deterministic.dfa.finalizers(state), &finalizers);
        assert_eq!(
            deterministic.dfa.possible_future_group_ids(state),
            &futures,
        );

        for byte in 0u16..=255 {
            let mut expected = SmallVec::<[u32; 8]>::new();
            for &source_state in subset.iter() {
                if let Some(target) = source.step(source_state, byte as u8) {
                    expected.push(target);
                }
            }
            expected.sort_unstable();
            expected.dedup();
            let mut expected = source.dfa.epsilon_closure(&expected);
            expected.sort_unstable();
            expected.dedup();

            match deterministic.step(state, byte as u8) {
                Some(target) => assert_eq!(
                    built.source_subsets[target as usize].as_ref(),
                    expected.as_slice(),
                    "state={state} byte={byte}",
                ),
                None => assert!(expected.is_empty(), "state={state} byte={byte}"),
            }
        }
    }

    for input in [
        b"".as_slice(),
        b"a",
        b"b",
        b"aa",
        b"ab",
        b"ba",
        b"aaa",
    ] {
        let source_result = source.execute_from_state(input, source.initial_state_id());
        let deterministic_result = deterministic
            .execute_from_state(input, deterministic.initial_state_id());
        let mut source_matches = source_result
            .matches
            .iter()
            .map(|matched| (matched.id, matched.width))
            .collect::<Vec<_>>();
        source_matches.sort_unstable();
        source_matches.dedup();
        let mut deterministic_matches = deterministic_result
            .matches
            .iter()
            .map(|matched| (matched.id, matched.width))
            .collect::<Vec<_>>();
        deterministic_matches.sort_unstable();
        deterministic_matches.dedup();
        assert_eq!(source_matches, deterministic_matches, "input={input:?}");

        let represented_end_states = deterministic_result
            .end_state
            .iter()
            .flat_map(|&state| built.source_subsets[state as usize].iter().copied())
            .collect::<BTreeSet<_>>();
        assert_eq!(
            represented_end_states,
            source_result.end_state.iter().copied().collect(),
            "input={input:?}",
        );
    }
}

#[test]
fn full_determinization_all_starts_maps_every_raw_runtime_state_exactly() {
    let source = arbitrary_epsilon_l1_test_tokenizer();
    let (built, raw_to_determinized) = source
        .try_full_determinization_all_starts(256, 16_384)
        .expect("small epsilon tokenizer must determinize from all raw starts");
    let deterministic = &built.tokenizer;
    let closures = source.all_singleton_epsilon_closures();

    assert_eq!(raw_to_determinized.len(), source.num_states() as usize);
    assert!(raw_to_determinized
        .iter()
        .all(|&state| state < deterministic.num_states()));

    for raw_state in 0..source.num_states() {
        let deterministic_state = raw_to_determinized[raw_state as usize];
        assert_eq!(
            built.source_subsets[deterministic_state as usize].as_ref(),
            closures[raw_state as usize].as_ref(),
            "raw_state={raw_state}",
        );

        for byte in 0u16..=255 {
            let mut targets = SmallVec::<[u32; 8]>::new();
            for &source_state in closures[raw_state as usize].iter() {
                if let Some(target) = source.step(source_state, byte as u8) {
                    targets.extend_from_slice(&closures[target as usize]);
                }
            }
            targets.sort_unstable();
            targets.dedup();

            match deterministic.step(deterministic_state, byte as u8) {
                Some(target) => assert_eq!(
                    built.source_subsets[target as usize].as_ref(),
                    targets.as_slice(),
                    "raw_state={raw_state} byte={byte}",
                ),
                None => assert!(
                    targets.is_empty(),
                    "raw_state={raw_state} byte={byte}",
                ),
            }
        }
    }
}

#[test]
fn all_starts_state_budget_rejects_acyclic_raw_state_lower_bound() {
    let mut dfa = DFA::new(2);
    dfa.ensure_group_capacity(1);
    dfa.add_epsilon_transition(0, 1);
    let source = Tokenizer::from_parts(dfa, 1, None);

    assert!(source
        .try_full_determinization_all_starts(1, 256)
        .is_none());
}

#[test]
fn all_starts_state_budget_does_not_reject_epsilon_cycle_collapse() {
    let mut dfa = DFA::new(2);
    dfa.ensure_group_capacity(1);
    dfa.add_epsilon_transition(0, 1);
    dfa.add_epsilon_transition(1, 0);
    let source = Tokenizer::from_parts(dfa, 1, None);

    let (built, raw_to_determinized) = source
        .try_full_determinization_all_starts(1, 256)
        .expect("one epsilon SCC should fit in one determinized state");
    assert_eq!(built.tokenizer.num_states(), 1);
    assert_eq!(raw_to_determinized, vec![0, 0]);
}
