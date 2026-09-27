use std::sync::Arc;

use super::*;
use crate::automata::weighted::dwa::DwaTransitionMap;
use crate::automata::lexer::ast::Expr;
use crate::automata::lexer::compile::{
    build_regex_monolithic as build_regex, build_regex_partitioned_with_adaptive,
};

fn tokenizer(expressions: Vec<Expr>) -> Tokenizer {
    let terminal_count = expressions.len() as u32;
    build_regex(&expressions).into_tokenizer(
        terminal_count,
        Some(Arc::from(expressions.into_boxed_slice())),
    )
}

#[test]
fn permitted_candidate_groups_still_require_and_retain_exact_witnesses() {
    let tokenizer = tokenizer(vec![
        Expr::U8Seq(b"a".to_vec()),
        Expr::U8Seq(b"a".to_vec()),
        Expr::U8Seq(b"a".to_vec()),
    ]);
    let mut relevant = [false; 256];
    relevant[b'a' as usize] = true;
    let context = TiDiscoveryContext::new(&tokenizer, &relevant);
    let active = [true, true, true];
    let permitted = vec![vec![0, 1]];
    let round = discover_one_round_with_transport_witnesses_in_context_permitted(
        &tokenizer,
        &active,
        &context,
        None,
        Some(&permitted),
    );
    assert_eq!(round.partition.get(&0), Some(&BTreeSet::from([0, 1])));
    assert_eq!(round.partition.get(&2), Some(&BTreeSet::from([2])));
    assert_eq!(round.partition.len(), 2);
    assert!(
        round.maps.contains_key(&(0, 1)),
        "the permitted merge must retain its exact transport witness",
    );
    assert!(
        round.maps.keys().all(|&(left, right)| left != 2 && right != 2),
        "a terminal outside the permitted family must remain singleton",
    );
}

#[test]
fn partition_invariants_hide_only_merged_active_members() {
    let active = [true, false, true, true, false];
    let partition = BTreeMap::from([
        (0, BTreeSet::from([0, 2])),
        (3, BTreeSet::from([3])),
    ]);
    assert_partition_invariants(&partition, &active);
    assert_eq!(active_terminals_for_partition(&partition, active.len()), [true, false, false, true, false]);
    assert_eq!(
        visible_output_raw_labels(&partition, &active),
        [true, false, false, true, false]
    );
}

#[test]
fn sparse_coalesced_disallowed_follows_matches_direct_member_pair_predicate() {
    fn direct(
        partition: &BTreeMap<TerminalID, BTreeSet<TerminalID>>,
        original: &BTreeMap<u32, BitSet>,
        num_terminals: usize,
    ) -> BTreeMap<u32, BitSet> {
        let mut result = BTreeMap::new();
        for (&predecessor_representative, predecessors) in partition {
            let mut bits = BitSet::new(num_terminals);
            for (&successor_representative, successors) in partition {
                if predecessors.iter().all(|&predecessor| {
                    successors.iter().all(|&successor| {
                        original
                            .get(&(predecessor as u32))
                            .is_some_and(|bits| bits.contains(successor as usize))
                    })
                }) {
                    bits.set(successor_representative as usize);
                }
            }
            result.insert(predecessor_representative as u32, bits);
        }
        result
    }

    let partition = BTreeMap::from([
        (0, BTreeSet::from([0, 1])),
        (2, BTreeSet::from([2, 3])),
        (4, BTreeSet::from([4])),
        (5, BTreeSet::from([5])),
    ]);
    let mut original = BTreeMap::new();
    let row = |successors: &[usize]| {
        let mut bits = BitSet::new(6);
        for &successor in successors {
            bits.set(successor);
        }
        bits
    };
    original.insert(0, row(&[2, 3, 4]));
    original.insert(1, row(&[2, 3]));
    original.insert(2, row(&[0, 1]));
    original.insert(3, row(&[0]));

    assert_eq!(
        coalesced_disallowed_follows(&partition, &original, 6),
        direct(&partition, &original, 6),
    );
}

#[test]
fn folding_memberships_requires_only_current_classes_and_round() {
    let active = [true, true, true, true];
    let initial = singleton_partition(&active);
    let first_round = BTreeMap::from([
        (0, BTreeSet::from([0, 1])),
        (2, BTreeSet::from([2])),
        (3, BTreeSet::from([3])),
    ]);
    let after_first = fold_one_round_partition(&initial, &first_round);
    let second_round = BTreeMap::from([
        (0, BTreeSet::from([0, 2])),
        (3, BTreeSet::from([3])),
    ]);
    let final_classes = fold_one_round_partition(&after_first, &second_round);

    assert_eq!(
        final_classes,
        BTreeMap::from([
            (0, BTreeSet::from([0, 1, 2])),
            (3, BTreeSet::from([3])),
        ]),
    );
    assert_partition_invariants(&final_classes, &active);
}

#[test]
fn transport_coordinate_quotient_matches_target_only_mode_signature() {
    let ordinary = ManyToOneIdMap::from_original_to_internal_with_representatives(
        vec![0, 0, 1, 1, 2, 2, 3],
        4,
        vec![0, 2, 4, 6],
    );
    let mut modes = vec![
        TerminalNwaTransportMode::ordinary(7),
        TerminalNwaTransportMode::member(
            TransportScannerStateMap::Explicit(vec![1, 0, 3, 2, 5, 4, 6].into()),
            0,
            1,
        ),
        TerminalNwaTransportMode::member(
            TransportScannerStateMap::Explicit(vec![2, 3, 0, 1, 6, 6, 4].into()),
            2,
            3,
        ),
    ];

    let expected_signatures = (0..ordinary.original_to_internal.len())
        .map(|source| {
            modes
                .iter()
                .map(|mode| {
                    ordinary.original_to_internal
                        [mode.scanner_state_for_original.scanner_state(source as u32) as usize]
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let quotient = transport_coordinate_quotient(&ordinary, &modes);
    for left in 0..expected_signatures.len() {
        for right in 0..expected_signatures.len() {
            assert_eq!(
                quotient.original_to_internal[left] == quotient.original_to_internal[right],
                expected_signatures[left] == expected_signatures[right],
                "signature quotient disagreed for states {left} and {right}",
            );
        }
    }

    canonicalize_transport_mode_states(&mut modes, &ordinary);
    let canonical_quotient = transport_coordinate_quotient(&ordinary, &modes);
    for left in 0..expected_signatures.len() {
        for right in 0..expected_signatures.len() {
            assert_eq!(
                canonical_quotient.original_to_internal[left]
                    == canonical_quotient.original_to_internal[right],
                expected_signatures[left] == expected_signatures[right],
                "canonical target-only transport changed the quotient for states {left} and {right}",
            );
        }
    }
}

#[test]
fn transport_coordinate_quotient_observes_composed_mode_deviations() {
    let ordinary = ManyToOneIdMap::from_original_to_internal_with_representatives(
        vec![0, 0, 1, 1],
        2,
        vec![0, 2],
    );
    let class_for_original: Arc<[u32]> = vec![0, 1, 2, 3].into();
    let representative_for_class: Arc<[u32]> = vec![0, 1, 2, 3].into();
    let inner = Arc::new(TransportScannerStateMap::Quotient {
        state_count: 4,
        class_for_original: Arc::clone(&class_for_original),
        representative_for_class: Arc::clone(&representative_for_class),
        source_class_for_target_deviations: vec![(0, 1), (1, 0)].into_boxed_slice(),
    });
    let outer = Arc::new(TransportScannerStateMap::Quotient {
        state_count: 4,
        class_for_original,
        representative_for_class,
        source_class_for_target_deviations: vec![(0, 2), (2, 0)].into_boxed_slice(),
    });
    let composed = TransportScannerStateMap::compose(outer, inner);
    let modes = vec![
        TerminalNwaTransportMode::ordinary(4),
        TerminalNwaTransportMode::member(composed.as_ref().clone(), 0, 1),
    ];
    let expected_signatures = (0..ordinary.original_to_internal.len())
        .map(|source| {
            modes
                .iter()
                .map(|mode| {
                    ordinary.original_to_internal
                        [mode.scanner_state_for_original.scanner_state(source as u32) as usize]
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    assert_ne!(expected_signatures[0], expected_signatures[1]);
    let quotient = transport_coordinate_quotient(&ordinary, &modes);
    for left in 0..expected_signatures.len() {
        for right in 0..expected_signatures.len() {
            assert_eq!(
                quotient.original_to_internal[left] == quotient.original_to_internal[right],
                expected_signatures[left] == expected_signatures[right],
                "composed transport signature quotient disagreed for states {left} and {right}",
            );
        }
    }
}

#[test]
fn post_dwa_member_expansion_reads_representative_weight_at_transport_target() {
    let state_map = ManyToOneIdMap::from_original_to_internal_with_representatives(
        vec![0, 1],
        2,
        vec![0, 1],
    );
    let member_map = TransportScannerStateMap::Explicit(vec![1, 0].into());
    let modes = vec![
        TerminalNwaTransportMode::ordinary(2),
        TerminalNwaTransportMode::member(member_map, 0, 1),
    ];
    let start_weight = Weight::from_uniform(
        0..=1,
        range_set_blaze::RangeSetBlaze::from_iter([10..=10, 20..=20]),
    );
    let suffix_weight = Weight::from_per_tsid_token_sets([
        (0, range_set_blaze::RangeSetBlaze::from_iter([10..=10])),
        (1, range_set_blaze::RangeSetBlaze::from_iter([20..=20])),
    ]);
    let core = DWA::from_parts(
        vec![
            DWAState {
                transitions: DwaTransitionMap::Owned(BTreeMap::from([(0, (1, start_weight))])),
                final_weight: None,
            },
            DWAState {
                transitions: DwaTransitionMap::Owned(BTreeMap::from([(2, (2, suffix_weight))])),
                final_weight: None,
            },
            DWAState {
                transitions: DwaTransitionMap::Owned(BTreeMap::new()),
                final_weight: Some(Weight::all()),
            },
        ],
        0,
    );

    let expanded = expand_representative_dwa_after_minimization(
        &core,
        &state_map,
        &state_map,
        &modes,
    );
    let member_word = expanded.eval_word(&[1, 2]);
    let ordinary_word = expanded.eval_word(&[0, 2]);
    assert!(member_word.tokens_for_tsid(0).contains(20));
    assert!(member_word.tokens_for_tsid(1).contains(10));
    assert!(
        !member_word.tokens_for_tsid(0).contains(10),
        "the member's suffix must use its transported representative coordinate",
    );
    assert!(ordinary_word.tokens_for_tsid(0).contains(10));
    assert!(!ordinary_word.tokens_for_tsid(0).contains(20));
}

#[test]
fn forward_domain_normalization_removes_unreachable_weight_coordinates() {
    let reachable = Weight::from_uniform(
        0..=0,
        range_set_blaze::RangeSetBlaze::from_iter([7..=7]),
    );
    let source = DWAState {
        transitions: DwaTransitionMap::Owned(BTreeMap::from([(10, (1, reachable.clone()))])),
        final_weight: None,
    };
    let middle = DWAState {
        transitions: DwaTransitionMap::Owned(BTreeMap::from([(11, (2, Weight::all()))])),
        final_weight: Some(Weight::all()),
    };
    let final_state = DWAState {
        transitions: DwaTransitionMap::Owned(BTreeMap::new()),
        final_weight: Some(Weight::all()),
    };
    let before = DWA::from_parts(vec![source, middle, final_state], 0);
    let after = restrict_weights_to_forward_domains(&before);

    assert_eq!(after.eval_word(&[10]), before.eval_word(&[10]));
    assert_eq!(after.eval_word(&[10, 11]), before.eval_word(&[10, 11]));
    assert_eq!(after.states()[1].final_weight.as_ref(), Some(&reachable));
    assert_eq!(after.states()[1].transitions.get(&11).unwrap().1, reachable);
}

#[test]
fn forward_domain_normalization_converges_on_cycles() {
    let reachable = Weight::from_uniform(
        0..=0,
        range_set_blaze::RangeSetBlaze::from_iter([7..=7]),
    );
    let before = DWA::from_parts(
        vec![
            DWAState {
                transitions: DwaTransitionMap::Owned(BTreeMap::from([(10, (1, reachable.clone()))])),
                final_weight: None,
            },
            DWAState {
                transitions: DwaTransitionMap::Owned(BTreeMap::from([(11, (1, Weight::all()))])),
                final_weight: Some(Weight::all()),
            },
        ],
        0,
    );
    let after = restrict_weights_to_forward_domains(&before);

    assert_eq!(after.eval_word(&[10]), before.eval_word(&[10]));
    assert_eq!(after.eval_word(&[10, 11, 11]), before.eval_word(&[10, 11, 11]));
    assert_eq!(after.states()[1].final_weight.as_ref(), Some(&reachable));
    assert_eq!(after.states()[1].transitions.get(&11).unwrap().1, reachable);
}

#[test]
fn iterative_discovery_stops_at_the_first_stable_round() {
    // Distinct literals whose alphabetic interiors are unobserved in this
    // punctuation-only L2P byte partition. The first exact round merges
    // them; the next single-representative round is the fixed point.
    let tokenizer = tokenizer(vec![
        Expr::U8Seq(b"CREATE\"".to_vec()),
        Expr::U8Seq(b"CrossFit\"".to_vec()),
        Expr::U8Seq(b"DELETE\"".to_vec()),
        Expr::U8Seq(b"Drums\"".to_vec()),
    ]);
    let mut active = vec![true; 4];
    let mut classes = singleton_partition(&active);
    let mut rounds = 0usize;
    let mut punctuation_only = [false; 256];
    punctuation_only[b'"' as usize] = true;
    loop {
        let round = discover_one_round(&tokenizer, &active, &punctuation_only, None);
        let next_active = active_terminals_for_partition(&round, active.len());
        classes = fold_one_round_partition(&classes, &round);
        rounds += 1;
        if next_active == active {
            break;
        }
        active = next_active;
    }

    assert_eq!(rounds, 2);
    assert_eq!(active, vec![true, false, false, false]);
    assert_eq!(classes, BTreeMap::from([(0, BTreeSet::from([0, 1, 2, 3]))]));
}

#[test]
fn rooted_map_rejects_a_reset_moving_rotation() {
    let tokenizer = tokenizer(vec![
        Expr::Seq(vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"aaaa".to_vec())),
                min: 0,
                max: None,
            },
        ]),
        Expr::Seq(vec![
            Expr::U8Seq(b"aaa".to_vec()),
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"aaaa".to_vec())),
                min: 0,
                max: None,
            },
        ]),
    ]);
    let mut dfa = InterchangeabilityDfa::new(&tokenizer, &[true, true], &[true; 256]);
    assert!(dfa.interchange_map(0, 1).is_none());
}

#[test]
fn identical_literals_have_a_rooted_interchange_map() {
    let tokenizer = tokenizer(vec![
        Expr::U8Seq(b"same".to_vec()),
        Expr::U8Seq(b"same".to_vec()),
    ]);
    let mut dfa = InterchangeabilityDfa::new(&tokenizer, &[true, true], &[true; 256]);
    let map = dfa.interchange_map(0, 1).expect("identical literals must transport");
    let root = tokenizer.initial_state_id() as usize;
    assert_eq!(map.scanner_state_map.scanner_state(root as u32), tokenizer.initial_state_id());
    let representatives = map.materialized_scanner_states();
    assert_eq!(map.scanner_state_map.scanner_state(root as u32), representatives[root]);
    let partition = discover_one_round(&tokenizer, &[true, true], &[true; 256], None);
    assert_eq!(partition, BTreeMap::from([(0, BTreeSet::from([0, 1]))]));
}

#[test]
fn partitioned_epsilon_nfa_ti_matches_monolithic_and_keeps_raw_transport() {
    let expressions = vec![
        Expr::U8Seq(b"same".to_vec()),
        Expr::U8Seq(b"same".to_vec()),
        Expr::U8Seq(b"other".to_vec()),
    ];
    let monolithic = tokenizer(expressions.clone());
    let terminal_count = expressions.len() as u32;
    let partitioned =
        build_regex_partitioned_with_adaptive(&expressions, &[0, 1, 2], false)
            .into_tokenizer(
                terminal_count,
                Some(Arc::from(expressions.into_boxed_slice())),
            );
    assert!(partitioned.has_epsilon_transitions());

    let active = [true, true, true];
    let relevant_bytes = [true; 256];
    let monolithic_round = discover_one_round_with_transport_witnesses(
        &monolithic,
        &active,
        &relevant_bytes,
        None,
    );
    let partitioned_round = discover_one_round_with_transport_witnesses(
        &partitioned,
        &active,
        &relevant_bytes,
        None,
    );

    assert_eq!(partitioned_round.partition, monolithic_round.partition);
    assert_eq!(
        partitioned_round.partition,
        BTreeMap::from([
            (0, BTreeSet::from([0, 1])),
            (2, BTreeSet::from([2])),
        ]),
    );
    assert!(
        partitioned_round.maps.contains_key(&(0, 1)),
        "the accepted NFA TI merge must retain its build-local transport witness",
    );
    for map in partitioned_round.maps.values() {
        assert_eq!(map.len(), partitioned.num_states() as usize);
        for raw_state in 0..partitioned.num_states() {
            assert!(
                map.scanner_state(raw_state) < partitioned.num_states(),
                "TI transport must stay in the raw TSID coordinate space",
            );
        }
    }
}

#[test]
fn quotient_projection_replaces_unrepresented_dead_class_sentinel() {
    let topology = RestrictedTopology {
        bytes: Vec::new(),
        edge_offsets: vec![0, 0, 0, 0],
        edges: Vec::new(),
        reverse_predecessors: Arc::from([Vec::new(), Vec::new()]),
        observed_destinations: Arc::from([false, false, false]),
        raw_state_count: 1,
        raw_representative_by_state: Some(Arc::from([0, u32::MAX])),
        transport_representative_by_state: Some(Arc::from([0, 0])),
        state_for_raw: Some(Arc::from([0])),
        nfa_output_rows: None,
        nfa_configurations: None,
        nfa_configurations_use_raw_states: false,
        real_state_count: 2,
        initial_state: 0,
        max_outdegree: 0,
    };

    let (_, _, representatives) = topology.scanner_projection_from_internal_quotient(
        Arc::from([0, 1]),
        Arc::from([0, 1]),
    );
    assert_eq!(representatives.as_ref(), &[0, 0]);
}

#[test]
fn global_scanner_state_quotient_preserves_exact_ti_round() {
    let tokenizer = tokenizer(vec![
        Expr::U8Seq(b"same".to_vec()),
        Expr::U8Seq(b"same".to_vec()),
    ]);
    let state_count = tokenizer.num_states() as usize;
    let identity = ManyToOneIdMap::from_original_to_internal_with_representatives(
        (0..state_count as u32).collect(),
        state_count as u32,
        (0..state_count as u32).collect(),
    );
    let global_state_quotient =
        GlobalScannerStateQuotient::from_total_raw_state_map(identity, state_count);

    let raw = discover_one_round_with_transport_witnesses(
        &tokenizer,
        &[true, true],
        &[true; 256],
        None,
    );
    let quotient = discover_one_round_with_transport_witnesses_with_global_state_quotient(
        &tokenizer,
        &[true, true],
        &[true; 256],
        None,
        &global_state_quotient,
    );

    assert_eq!(quotient.partition, raw.partition);
    assert!(quotient.maps.values().all(|map| map.len() == state_count));
}

#[test]
fn partitioned_epsilon_ti_consumes_nonidentity_global_state_coordinate() {
    let tokenizer =
        crate::automata::lexer::tokenizer::arbitrary_epsilon_l1_test_tokenizer();

    let vocab = crate::Vocab::new(
        vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"aa".to_vec()),
            (3, b"ba".to_vec()),
        ]);
    let (global_state_quotient, _) = crate::compiler::stages::id_map_and_terminal_dwa::l2p::
        equivalence_analysis::state_equivalence::global_token_position::
        compute_global_token_position_state_quotient(&tokenizer, &vocab);
    assert!(
        (global_state_quotient
            .as_many_to_one()
            .num_internal_ids() as usize)
            < tokenizer.num_states() as usize,
        "the fixture must exercise a non-identity C coordinate",
    );

    let relevant_bytes = [true; 256];
    let raw_topology = RestrictedTopology::new(&tokenizer, &relevant_bytes);
    let c_topology = RestrictedTopology::new_with_global_state_quotient(
        &tokenizer,
        &relevant_bytes,
        &global_state_quotient,
    );
    assert!(
        c_topology.real_state_count < raw_topology.real_state_count,
        "epsilon TI must build its powerset in the supplied C coordinate",
    );

    let active = [true, true];
    let raw = discover_one_round_with_transport_witnesses(
        &tokenizer,
        &active,
        &relevant_bytes,
        None,
    );
    let c = discover_one_round_with_transport_witnesses_with_global_state_quotient(
        &tokenizer,
        &active,
        &relevant_bytes,
        None,
        &global_state_quotient,
    );
    assert_eq!(c.partition, raw.partition);
    assert!(
        c.maps
            .values()
            .all(|map| map.len() == tokenizer.num_states() as usize),
        "C-coordinate TI transport must remain in raw scanner coordinates",
    );
}

#[test]
fn alpha_interiors_are_ignored_when_only_punctuation_is_enabled() {
    let tokenizer = tokenizer(vec![
        Expr::U8Seq(b"CREATE\"".to_vec()),
        Expr::U8Seq(b"CrossFit\"".to_vec()),
        Expr::U8Seq(b"DELETE\"".to_vec()),
        Expr::U8Seq(b"Drums\"".to_vec()),
    ]);
    let mut punctuation_only = [false; 256];
    punctuation_only[b'"' as usize] = true;
    let partition = discover_one_round(
        &tokenizer,
        &[true, true, true, true],
        &punctuation_only,
        None,
    );
    assert_eq!(partition, BTreeMap::from([(0, BTreeSet::from([0, 1, 2, 3]))]));
}

#[test]
fn unobserved_byte_nonidentity_map_matches_hash_reference() {
    let tokenizer = tokenizer(vec![
        Expr::U8Seq(b"a".to_vec()),
        Expr::U8Seq(b"b".to_vec()),
    ]);
    let mut relevant_bytes = [false; 256];
    relevant_bytes[b'c' as usize] = true;
    let mut dfa = InterchangeabilityDfa::new(&tokenizer, &[true, true], &relevant_bytes);
    let optimized = dfa
        .interchange_map(0, 1)
        .expect("unobserved terminals must transport");
    let reference = dfa
        .reference_interchange_map(0, 1)
        .expect("hash reference must transport the same pair");
    assert_eq!(optimized, reference);
    assert!(
        optimized
            .materialized_scanner_states()
            .iter()
            .enumerate()
            .any(|(state, &target)| target != state as u32),
        "the MRE requires a nonidentity raw scanner map",
    );
}

#[test]
fn byte_restriction_does_not_recompute_frozen_future_finalizers() {
    let tokenizer = tokenizer(vec![
        Expr::U8Seq(b"a".to_vec()),
        Expr::U8Seq(b"az".to_vec()),
    ]);
    let after_a = tokenizer.get_transition(tokenizer.initial_state_id(), b'a') as usize;
    let mut only_a = [false; 256];
    only_a[b'a' as usize] = true;
    let restricted = InterchangeabilityDfa::new(&tokenizer, &[true, true], &only_a);
    assert_eq!(restricted.topology.bytes, vec![b'a']);
    assert_eq!(restricted.destination_for_slot(after_a, 0), restricted.dead_state());
    let output_id = restricted.output_pair_by_state[after_a] as usize;
    assert!(restricted.output_pairs[output_id]
        .future_finalizers
        .contains(1));
}

#[test]
fn unobserved_outputs_do_not_split_structural_prefilter() {
    let tokenizer = tokenizer(vec![
        Expr::U8Seq(b"a".to_vec()),
        Expr::U8Seq(b"bb".to_vec()),
    ]);
    let active = vec![true, true];
    let candidates = vec![0, 1];
    let mut only_x = [false; 256];
    only_x[b'x' as usize] = true;

    // No raw terminal-output state is an enabled-byte destination. The
    // reference consequently observes only the synthetic dead output.
    let topology = RestrictedTopology::new(&tokenizer, &only_x);
    let (root_groups, _) = rooted_candidate_groups(&tokenizer, &candidates, &topology);
    let (structural_signatures, _) =
        structural_candidate_signatures(
            &tokenizer,
            &active,
            &candidates,
            &topology,
            STRUCTURAL_REFINEMENT_ROUNDS,
        );
    let filtered_groups =
        refine_candidate_groups_by_structure(root_groups, &candidates, &structural_signatures);
    assert!(group_contains_pair(&filtered_groups, 0, 1));

    let mut dfa = InterchangeabilityDfa::new(&tokenizer, &active, &only_x);
    assert!(dfa.interchange_map(0, 1).is_some());
    let partition = discover_one_round(&tokenizer, &active, &only_x, None);
    assert_eq!(partition, BTreeMap::from([(0, BTreeSet::from([0, 1]))]));
}

#[test]
fn inactive_outputs_are_not_observed() {
    let tokenizer = tokenizer(vec![
        Expr::U8Seq(b"a".to_vec()),
        Expr::U8Seq(b"b".to_vec()),
        Expr::U8Seq(b"a".to_vec()),
    ]);
    let mut dfa = InterchangeabilityDfa::new(&tokenizer, &[true, false, true], &[true; 256]);
    assert!(dfa.interchange_map(0, 2).is_some());
}

fn group_contains_pair(groups: &[Vec<TerminalID>], left: TerminalID, right: TerminalID) -> bool {
    groups.iter().any(|group| {
        group.contains(&left) && group.contains(&right)
    })
}

#[test]
fn exact_prefilters_never_reject_a_reference_interchange_pair() {
    let tokenizer = tokenizer(vec![
        Expr::U8Seq(b"same".to_vec()),
        Expr::U8Seq(b"same".to_vec()),
        Expr::U8Seq(b"different".to_vec()),
        Expr::U8Seq(b"differs".to_vec()),
    ]);
    let active = vec![true; 4];
    let candidates = (0..4).collect::<Vec<TerminalID>>();
    let relevant_bytes = [true; 256];
    let topology = RestrictedTopology::new(&tokenizer, &relevant_bytes);
    let (root_groups, _) = rooted_candidate_groups(&tokenizer, &candidates, &topology);
    let (structural_signatures, _) = structural_candidate_signatures(
        &tokenizer,
        &active,
        &candidates,
        &topology,
        STRUCTURAL_REFINEMENT_ROUNDS,
    );
    let filtered_groups = refine_candidate_groups_by_structure(
        root_groups.clone(),
        &candidates,
        &structural_signatures,
    );
    let mut dfa = InterchangeabilityDfa::new(&tokenizer, &active, &relevant_bytes);
    for (index, &left) in candidates.iter().enumerate() {
        for &right in &candidates[index + 1..] {
            if let Some(left_to_right) = dfa.interchange_map(left, right) {
                let right_to_left = dfa
                    .interchange_map(right, left)
                    .expect("the same transposition must produce the same map");
                assert!(
                    group_contains_pair(&root_groups, left, right),
                    "root prefilter rejected exact pair {left} <-> {right}",
                );
                assert!(
                    group_contains_pair(&filtered_groups, left, right),
                    "structural prefilter rejected exact pair {left} <-> {right}",
                );
                assert!(
                    dfa.canonical_round_one_still_possible(left, right),
                    "first-round prefilter rejected exact pair {left} <-> {right}",
                );
                assert_eq!(
                    left_to_right.materialized_scanner_states(),
                    right_to_left.materialized_scanner_states(),
                    "the reversed pair call must be operationally identical",
                );
            }
        }
    }
}

#[test]
fn canonical_sparse_quotient_matches_hash_reference_for_all_pairs() {
    let tokenizer = tokenizer(vec![Expr::U8Seq(b"same".to_vec()), Expr::U8Seq(b"same".to_vec()), Expr::U8Seq(b"sample".to_vec()), Expr::U8Seq(b"simple".to_vec()), Expr::U8Seq(b"a".to_vec()), Expr::U8Seq(b"ab".to_vec()), Expr::U8Seq(b"b".to_vec()), Expr::U8Seq(b"ba".to_vec())]);
    let active = vec![true; 8];
    for left in 0..active.len() as TerminalID {
        for right in left + 1..active.len() as TerminalID {
            let mut canonical = InterchangeabilityDfa::new(&tokenizer, &active, &[true; 256]);
            let mut reference = InterchangeabilityDfa::new(&tokenizer, &active, &[true; 256]);
            assert_eq!(canonical.interchange_map(left, right), reference.reference_interchange_map(left, right), "canonical refinement disagreed with hash reference for {left} <-> {right}");
        }
    }
}

#[test]
fn canonical_paired_hopcroft_matches_hash_reference_for_mixed_long_shapes() {
    let fixtures = vec![
        vec![
            Expr::U8Seq(b"same".to_vec()),
            Expr::U8Seq(b"same".to_vec()),
            Expr::U8Seq(b"sample".to_vec()),
            Expr::U8Seq(b"simple".to_vec()),
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"ab".to_vec()),
            Expr::U8Seq(b"b".to_vec()),
            Expr::U8Seq(b"ba".to_vec()),
        ],
        vec![
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                min: 1,
                max: Some(255),
            },
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                min: 1,
                max: Some(251),
            },
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                min: 0,
                max: Some(255),
            },
            Expr::U8Seq(b"b".to_vec()),
        ],
        vec![
            Expr::Seq(vec![
                Expr::U8Seq(b"a".to_vec()),
                Expr::Repeat {
                    expr: Box::new(Expr::U8Seq(b"aaaa".to_vec())),
                    min: 0,
                    max: None,
                },
            ]),
            Expr::Seq(vec![
                Expr::U8Seq(b"aaa".to_vec()),
                Expr::Repeat {
                    expr: Box::new(Expr::U8Seq(b"aaaa".to_vec())),
                    min: 0,
                    max: None,
                },
            ]),
            Expr::U8Seq(b"aaaaa".to_vec()),
            Expr::U8Seq(b"c".to_vec()),
        ],
    ];

    for expressions in fixtures {
        let tokenizer = tokenizer(expressions);
        let active = vec![true; tokenizer.num_terminals() as usize];
        for left in 0..active.len() as TerminalID {
            for right in left + 1..active.len() as TerminalID {
                let mut paired =
                    InterchangeabilityDfa::new(&tokenizer, &active, &[true; 256]);
                let paired_result =
                    paired.canonical_paired_hopcroft_interchange_map(left, right);
                let mut reference =
                    InterchangeabilityDfa::new(&tokenizer, &active, &[true; 256]);
                let reference_result = reference.reference_interchange_map(left, right);
                assert_eq!(
                    paired_result, reference_result,
                    "paired Hopcroft disagreed with hash reference for {left}<>{right}",
                );
            }
        }
    }
}

#[test]
fn support_quotient_identity_transport_matches_moore_history_exactly() {
    let fixtures = vec![
        vec![
            Expr::U8Seq(b"same".to_vec()),
            Expr::U8Seq(b"same".to_vec()),
            Expr::U8Seq(b"sample".to_vec()),
            Expr::U8Seq(b"simple".to_vec()),
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"ab".to_vec()),
            Expr::U8Seq(b"b".to_vec()),
            Expr::U8Seq(b"ba".to_vec()),
        ],
        vec![
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                min: 1,
                max: Some(255),
            },
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                min: 1,
                max: Some(251),
            },
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                min: 0,
                max: Some(255),
            },
            Expr::U8Seq(b"b".to_vec()),
        ],
    ];

    for expressions in fixtures {
        let tokenizer = tokenizer(expressions);
        let active = vec![true; tokenizer.num_terminals() as usize];
        let mut support =
            InterchangeabilityDfa::new(&tokenizer, &active, &[true; 256]);
        support.ensure_support_quotient();
        let support_map = support.identity_map_from_unseeded_support_quotient();

        let mut moore = InterchangeabilityDfa::new(&tokenizer, &active, &[true; 256]);
        let moore_map = moore.canonical_identity_map_via_moore_history();
        assert_eq!(
            support_map.materialized_scanner_states(),
            moore_map.materialized_scanner_states(),
            "stable support quotient and canonical Moore history must induce the same identity transport",
        );
    }

    // Exercise the epsilon/NFA discovery topology as well.  The support
    // quotient is over internal powerset states there, and the common
    // scanner-map projection must still reproduce the exact raw-state
    // identity transport chosen by the Moore history.
    let expressions = vec![
        Expr::U8Seq(b"same".to_vec()),
        Expr::U8Seq(b"same".to_vec()),
        Expr::U8Seq(b"other".to_vec()),
    ];
    let terminal_count = expressions.len() as u32;
    let tokenizer = build_regex_partitioned_with_adaptive(&expressions, &[0, 1, 2], false)
        .into_tokenizer(
            terminal_count,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
    assert!(tokenizer.has_epsilon_transitions());
    let active = vec![true; terminal_count as usize];
    let mut support = InterchangeabilityDfa::new(&tokenizer, &active, &[true; 256]);
    support.ensure_support_quotient();
    let support_map = support.identity_map_from_unseeded_support_quotient();
    let mut moore = InterchangeabilityDfa::new(&tokenizer, &active, &[true; 256]);
    let moore_map = moore.canonical_identity_map_via_moore_history();
    assert_eq!(
        support_map.materialized_scanner_states(),
        moore_map.materialized_scanner_states(),
        "epsilon/NFA support quotient must induce the same raw scanner identity transport",
    );
}

#[test]
fn seeded_support_quotient_identity_transport_falls_back_to_exact_moore() {
    let tokenizer = tokenizer(vec![
        Expr::U8Seq(b"same".to_vec()),
        Expr::U8Seq(b"same".to_vec()),
        Expr::U8Seq(b"sample".to_vec()),
        Expr::U8Seq(b"simple".to_vec()),
        Expr::U8Seq(b"a".to_vec()),
        Expr::U8Seq(b"ab".to_vec()),
        Expr::U8Seq(b"b".to_vec()),
        Expr::U8Seq(b"ba".to_vec()),
    ]);
    let topology = Arc::new(RestrictedTopology::new(&tokenizer, &[true; 256]));
    let raw = Arc::new(TiRawDiscoveryData::new(&tokenizer, &topology, None));

    let all_active = vec![true; 8];
    let mut seed_dfa = InterchangeabilityDfa::from_raw_discovery_data(
        &all_active,
        Arc::clone(&topology),
        Arc::clone(&raw),
        None,
        None,
    );
    seed_dfa.ensure_support_quotient();
    let seed = SupportPartitionSeed {
        active_terminals: all_active.into(),
        quotient: Arc::clone(
            seed_dfa
                .support_quotient
                .as_ref()
                .expect("support quotient initialized"),
        ),
    };

    let current_active = [true, true, false, false, true, false, true, false];
    let mut seeded = InterchangeabilityDfa::from_raw_discovery_data(
        &current_active,
        Arc::clone(&topology),
        Arc::clone(&raw),
        Some(seed),
        None,
    );
    // Even with the support shortcut enabled by default, a seeded round must
    // retain the canonical Moore route because the projected support quotient
    // can be a strict refinement of the fresh current-round fixed point.
    let seeded_map = seeded.canonical_identity_map();

    let mut unseeded = InterchangeabilityDfa::from_raw_discovery_data(
        &current_active,
        topology,
        raw,
        None,
        None,
    );
    let moore_map = unseeded.canonical_identity_map_via_moore_history();
    assert_eq!(
        seeded_map.materialized_scanner_states(),
        moore_map.materialized_scanner_states(),
        "seeded TI identity transport must fall back to the exact current-round Moore fixed point",
    );
}

#[test]
fn canonical_identity_rounds_from_fine_support_seed_match_raw_rounds_exactly() {
    let tokenizer = tokenizer(vec![
        Expr::U8Seq(b"same".to_vec()),
        Expr::U8Seq(b"same".to_vec()),
        Expr::U8Seq(b"sample".to_vec()),
        Expr::U8Seq(b"simple".to_vec()),
        Expr::U8Seq(b"a".to_vec()),
        Expr::U8Seq(b"ab".to_vec()),
        Expr::U8Seq(b"b".to_vec()),
        Expr::U8Seq(b"ba".to_vec()),
    ]);
    let topology = Arc::new(RestrictedTopology::new(&tokenizer, &[true; 256]));
    let raw = Arc::new(TiRawDiscoveryData::new(&tokenizer, &topology, None));
    let all_active = vec![true; 8];
    let mut seed_dfa = InterchangeabilityDfa::from_raw_discovery_data(
        &all_active,
        Arc::clone(&topology),
        Arc::clone(&raw),
        None,
        None,
    );
    seed_dfa.ensure_support_quotient();
    let seed = SupportPartitionSeed {
        active_terminals: all_active.into(),
        quotient: Arc::clone(
            seed_dfa
                .support_quotient
                .as_ref()
                .expect("support quotient initialized"),
        ),
    };
    let current_active = [true, true, false, false, true, false, true, false];
    let dfa = InterchangeabilityDfa::from_raw_discovery_data(
        &current_active,
        topology,
        raw,
        Some(seed.clone()),
        None,
    );

    let mut previous = vec![0u32; dfa.state_count()];
    for round in 1..=dfa.state_count() * 2 {
        let raw_round = dfa.canonical_identity_round(&previous);
        let seeded = dfa.canonical_identity_round_from_support_seed(&previous, &seed);
        assert_eq!(seeded.classes, raw_round.classes, "round {round}");
        assert_eq!(
            seeded.representative_by_class,
            raw_round.representative_by_class,
            "round {round}",
        );
        assert_eq!(
            seeded.classes_by_signature_hash,
            raw_round.classes_by_signature_hash,
            "round {round}",
        );
        let stable = same_equality_partition_u32(&previous, &raw_round.classes);
        previous = raw_round.classes;
        if stable {
            return;
        }
    }
    panic!("identity refinement did not stabilize");
}

#[test]
fn equality_partition_stability_ignores_changing_digest_values() {
    let a = CharacterizationHash([1; blake3::OUT_LEN]);
    let b = CharacterizationHash([2; blake3::OUT_LEN]);
    let x = CharacterizationHash([9; blake3::OUT_LEN]);
    let y = CharacterizationHash([10; blake3::OUT_LEN]);
    assert!(same_equality_partition_pair(&[a, a, b], &[a, a, b], &[x, x, y], &[x, x, y]));
    assert!(!same_equality_partition_pair(&[a, a, b], &[a, a, b], &[x, y, y], &[x, y, y]));
}

#[test]
fn combined_integer_partition_stability_rejects_cross_side_split() {
    assert!(same_equality_partition_pair_u32(
        &[0, 0, 1],
        &[0, 0, 1],
        &[4, 4, 5],
        &[4, 4, 5],
    ));
    // Each side is individually stable, but the shared old class `0`
    // refines differently across sides. This cannot certify a transport.
    assert!(!same_equality_partition_pair_u32(
        &[0, 0],
        &[0, 0],
        &[4, 4],
        &[4, 5],
    ));
}
