use super::{
    ClassProfile, PointwiseBehaviorInterner, PointwiseBehaviorMap, PointwiseBehaviorMapLayout,
    PointwiseMergeGroup, PointwiseProfile, PointwiseRegionBuildCache, PointwiseRegionInterner,
    TokenBehaviorRange, batch_build_weight, build_exact_group_summary, build_pointwise_profile,
    build_token_behavior_region, final_weights_compatible_on_domain, find_difference,
    memberwise_group_compatible, minimize_acyclic, minimize_acyclic_owned_path_conditioned,
    overlay_compatible_token_behavior_ranges, pointwise_groups_to_builders,
    pointwise_groups_to_builders_reference, push_weights, reverse_hashcons_acyclic_owned,
    sorted_weights_compatible_on_domain, sorted_weights_compatible_on_domain_intersection,
    try_minimize_small_pairwise_direct, weight_is_disjoint_from_domain_intersection,
    weights_equal_on_domain, weights_equal_on_domain_intersection,
};
use crate::ds::weight::Weight;
use crate::weighted_u32::dwa::{DWA, DWAState};
use range_set_blaze::RangeSetBlaze;
use std::sync::Arc;

fn token_set(ranges: &[(u32, u32)]) -> RangeSetBlaze<u32> {
    ranges.iter().copied().map(|(start, end)| start..=end).collect()
}

fn weight(entries: &[(u32, &[(u32, u32)])]) -> Weight {
    Weight::from_per_tsid_token_sets(
        entries
            .iter()
            .copied()
            .map(|(tsid, ranges)| (tsid, token_set(ranges))),
    )
}

#[test]
fn reverse_hashcons_merges_only_identical_weighted_suffixes() {
    let mut states = vec![DWAState::default(); 4];
    states[0]
        .transitions
        .insert(1, (1, Weight::all()));
    states[0]
        .transitions
        .insert(2, (2, Weight::all()));
    let shared_edge = weight(&[(0, &[(1, 7)])]);
    states[1]
        .transitions
        .insert(3, (3, shared_edge.clone()));
    states[2]
        .transitions
        .insert(3, (3, shared_edge));
    states[3].final_weight = Some(weight(&[(0, &[(2, 6)])]));
    let original = DWA::from_parts(states, 0);

    let hashconsed = reverse_hashcons_acyclic_owned(original.clone());
    assert_eq!(hashconsed.num_states(), 3);
    assert_eq!(
        find_difference(&original, &hashconsed).expect("acyclic equivalence must be decidable"),
        None,
    );
}

#[test]
fn small_direct_allows_large_leaf_bucket() {
    let leaf_count = 65usize;
    let mut states = Vec::with_capacity(leaf_count + 1);
    let mut start = DWAState::default();
    for leaf in 0..leaf_count {
        start.transitions.insert(
            leaf as i32,
            ((leaf + 1) as u32, Weight::all()),
        );
    }
    states.push(start);
    for _ in 0..leaf_count {
        let mut leaf = DWAState::default();
        leaf.final_weight = Some(Weight::all());
        states.push(leaf);
    }
    let dwa = DWA::from_parts(states, 0);

    let minimized = try_minimize_small_pairwise_direct(&dwa)
        .expect("height-0 leaf count must not trip the pairwise bucket limit");
    assert_eq!(
        find_difference(&dwa, &minimized).expect("equivalence check must succeed"),
        None,
    );
    assert_eq!(minimized.states().len(), 2);
    assert_eq!(minimized.states()[minimized.start_state() as usize].transitions.len(), leaf_count);
}

#[test]
fn small_direct_accepts_over_128_states_when_nonleaf_buckets_stay_bounded() {
    // Mirror the shape that motivated widening the total-state guard:
    // many height-0 leaves, a modest height-1 bucket, and one root.  The
    // pairwise work is controlled by the 48-state non-leaf bucket even
    // though the complete automaton has well over 128 states.
    let middle_count = 48usize;
    let leaf_count = 103usize;
    let root = 0usize;
    let first_middle = 1usize;
    let first_leaf = first_middle + middle_count;
    let total_states = first_leaf + leaf_count;
    assert!(total_states > 128 && total_states <= 192);

    let mut states = vec![DWAState::default(); total_states];
    for middle in 0..middle_count {
        states[root].transitions.insert(
            middle as i32,
            ((first_middle + middle) as u32, Weight::all()),
        );
        states[first_middle + middle].transitions.insert(
            10_000,
            ((first_leaf + middle % leaf_count) as u32, Weight::all()),
        );
    }
    for leaf in 0..leaf_count {
        states[first_leaf + leaf].final_weight = Some(Weight::all());
    }

    let dwa = DWA::from_parts(states, root as u32);
    let minimized = try_minimize_small_pairwise_direct(&dwa)
        .expect("bounded non-leaf buckets should admit the >128-state direct path");
    assert_eq!(
        find_difference(&dwa, &minimized).expect("equivalence check must succeed"),
        None,
    );
    assert!(minimized.num_states() < dwa.num_states());
}

#[test]
fn reconstruction_one_sweep_matches_reduction_tree() {
    let pending: Vec<Weight> = (0..257u32)
        .map(|index| {
            Weight::from_per_tsid_token_sets(std::iter::once((
                index % 17,
                RangeSetBlaze::from_iter(std::iter::once((index % 31)..=(index % 31 + 2))),
            )))
        })
        .collect();

    let direct = batch_build_weight(pending.clone());
    let mut tree = pending;
    while tree.len() > 64 {
        tree = tree
            .chunks(64)
            .map(|chunk| Weight::union_all(chunk.iter()))
            .collect();
    }
    let reduction_tree = Weight::union_all(tree.iter());
    assert_eq!(direct, reduction_tree);
}

fn assert_disjoint_matches_overlap(weight: &Weight, left: &Weight, right: &Weight) {
    let overlap = left.intersection(right);
    assert_eq!(
        weight_is_disjoint_from_domain_intersection(weight, left, right),
        weight.is_disjoint(&overlap),
    );
}

fn assert_equal_matches_overlap(a: &Weight, b: &Weight, left: &Weight, right: &Weight) {
    let overlap = left.intersection(right);
    assert_eq!(
        weights_equal_on_domain_intersection(a, b, left, right),
        weights_equal_on_domain(a, b, &overlap),
    );
}

#[test]
fn monotone_pointwise_profile_matches_boundary_materialization() {
    let domain = Weight::from_per_tsid_token_sets(
        [0u32, 1, 2, 4, 5, 6]
            .into_iter()
            .map(|tsid| (tsid, token_set(&[(0, 3)]))),
    );
    let primary = domain.clone();
    let secondary = Weight::from_per_tsid_token_sets(
        [1u32, 2, 4, 5]
            .into_iter()
            .map(|tsid| (tsid, token_set(&[(2, 3)]))),
    );
    let final_weight = Weight::from_per_tsid_token_sets(
        [0u32, 2, 5]
            .into_iter()
            .map(|tsid| (tsid, token_set(&[(0, 1)]))),
    );
    let profile = ClassProfile {
        targets: vec![(7, 1), (8, 2)],
        weights: vec![(7, primary), (8, secondary)],
        final_weight: Some(final_weight),
    };

    let mut behaviors = PointwiseBehaviorInterner::default();
    let mut regions = PointwiseRegionInterner::default();
    let old = build_pointwise_profile(
        &domain,
        &profile,
        &mut behaviors,
        &mut regions,
        &mut PointwiseRegionBuildCache::default(),
        false,
    )
    .unwrap();
    let monotone = build_pointwise_profile(
        &domain,
        &profile,
        &mut behaviors,
        &mut regions,
        &mut PointwiseRegionBuildCache::default(),
        true,
    )
    .unwrap();

    assert_eq!(old.by_tsid.len(), monotone.by_tsid.len());
    for ((old_tsid, old_region), (new_tsid, new_region)) in
        old.by_tsid.iter().zip(&monotone.by_tsid)
    {
        assert_eq!(old_tsid, new_tsid);
        assert_eq!(old_region.as_ref(), new_region.as_ref(), "tsid={old_tsid}");
    }
}

#[test]
fn pointwise_region_build_cache_includes_transition_target() {
    let domain_tokens = token_set(&[(0, 3)]);
    let transition_tokens = token_set(&[(0, 3)]);
    let mut behaviors = PointwiseBehaviorInterner::default();
    let mut regions = PointwiseRegionInterner::default();
    let mut cache = PointwiseRegionBuildCache::default();
    let first_transitions = [(7, 1, &transition_tokens)];
    let second_transitions = [(7, 2, &transition_tokens)];

    let first = build_token_behavior_region(
        &domain_tokens,
        None,
        &first_transitions,
        &mut behaviors,
        &mut regions,
        &mut cache,
    )
    .unwrap();
    let first_again = build_token_behavior_region(
        &domain_tokens,
        None,
        &first_transitions,
        &mut behaviors,
        &mut regions,
        &mut cache,
    )
    .unwrap();
    let second = build_token_behavior_region(
        &domain_tokens,
        None,
        &second_transitions,
        &mut behaviors,
        &mut regions,
        &mut cache,
    )
    .unwrap();

    assert!(Arc::ptr_eq(&first, &first_again));
    assert_ne!(first.as_ref(), second.as_ref());
    assert_eq!(cache.entries.len(), 2);
    assert_eq!(cache.hits, 1);
    assert_eq!(cache.misses, 2);
}

#[test]
fn direct_mapped_region_overlay_cache_matches_exact_overlay() {
    let mut regions = PointwiseRegionInterner::with_direct_overlay_slots(8);
    let left = regions.intern(vec![
        TokenBehaviorRange {
            start: 0,
            end: 3,
            behavior: 1,
        },
        TokenBehaviorRange {
            start: 8,
            end: 9,
            behavior: 2,
        },
    ]);
    let right = regions.intern(vec![
        TokenBehaviorRange {
            start: 2,
            end: 5,
            behavior: 1,
        },
        TokenBehaviorRange {
            start: 10,
            end: 12,
            behavior: 3,
        },
    ]);
    let expected = overlay_compatible_token_behavior_ranges(left.as_ref(), right.as_ref());

    let first = regions.overlay_compatible(&left, &right);
    let second = regions.overlay_compatible(&right, &left);

    assert_eq!(first.as_ref(), &expected);
    assert!(Arc::ptr_eq(&first, &second));
    assert_eq!(regions.direct_overlay_stats(), (8, 1, 1, 0));
}

#[test]
fn dense_pointwise_behavior_map_matches_sparse() {
    let left = Arc::new(vec![TokenBehaviorRange {
        start: 0,
        end: 3,
        behavior: 1,
    }]);
    let right = Arc::new(vec![TokenBehaviorRange {
        start: 4,
        end: 7,
        behavior: 2,
    }]);
    let first = PointwiseProfile {
        by_tsid: vec![(1, Arc::clone(&left)), (3, Arc::clone(&right))],
    };
    let second = PointwiseProfile {
        by_tsid: vec![(1, Arc::clone(&right)), (2, Arc::clone(&left))],
    };

    let mut sparse = PointwiseBehaviorMap::new(PointwiseBehaviorMapLayout::Sparse);
    let mut dense = PointwiseBehaviorMap::new(PointwiseBehaviorMapLayout::Dense { slots: 4 });
    let mut sparse_regions = PointwiseRegionInterner::default();
    let mut dense_regions = PointwiseRegionInterner::default();
    for profile in [&first, &second] {
        sparse.merge_profile(profile, &mut sparse_regions);
        dense.merge_profile(profile, &mut dense_regions);
    }

    for tsid in 0..4 {
        assert_eq!(
            sparse.get(tsid).map(AsRef::as_ref),
            dense.get(tsid).map(AsRef::as_ref),
            "tsid={tsid}",
        );
    }
    assert_eq!(sparse.region_entry_count(), dense.region_entry_count());
}

#[test]
fn path_conditioned_minimizer_preserves_weighted_language() {
    // The two middle states represent the same structural residual under
    // disjoint cumulative path domains. This is the shape emitted by
    // determinizing an already backward-pushed token-conditional NWA.
    let left_domain = Weight::from_uniform(1..=1, token_set(&[(10, 19)]));
    let right_domain = Weight::from_uniform(1..=1, token_set(&[(20, 29)]));
    let all_domain = left_domain.union(&right_domain);

    let mut start = DWAState::default();
    start.transitions.insert(1, (1, left_domain.clone()));
    start.transitions.insert(2, (2, right_domain.clone()));

    let mut left = DWAState::default();
    left.transitions.insert(3, (3, left_domain.clone()));
    let mut right = DWAState::default();
    right.transitions.insert(3, (3, right_domain.clone()));

    let mut leaf = DWAState::default();
    leaf.final_weight = Some(all_domain);

    let original = DWA::from_parts(vec![start, left, right, leaf], 0);
    let minimized = minimize_acyclic_owned_path_conditioned(original.clone());
    assert_eq!(
        find_difference(&original, &minimized).unwrap(),
        None,
        "path-conditioned minimization changed the weighted language",
    );
    assert!(minimized.num_states() < original.num_states());
}

#[test]
fn push_weights_maintains_needed_containment_invariant() {
    let mut source = DWAState::default();
    source.transitions.insert(
        42,
        (1, Weight::from_uniform(1..=1, token_set(&[(10, 20)]))),
    );
    let mut target = DWAState::default();
    target.final_weight = Some(Weight::from_uniform(1..=1, token_set(&[(12, 14)])));

    let mut dwa = DWA::from_parts(vec![source, target], 0);
    let (_, topo, needed) = push_weights(&mut dwa);
    assert!(topo.is_some());

    let transition_weight = &dwa.states()[0].transitions.get(&42).unwrap().1;
    let target_final_weight = dwa.states()[1].final_weight.as_ref().unwrap();

    assert!(target_final_weight.is_subset(&needed[1]));
    assert!(transition_weight.is_subset(&needed[0]));
    assert!(transition_weight.is_subset(&needed[1]));
    assert_eq!(
        transition_weight,
        &Weight::from_uniform(1..=1, token_set(&[(12, 14)]))
    );
}

#[test]
fn transition_compat_accepts_matching_label_equal_on_overlap_but_different_outside() {
    let overlap = weight(&[(1, &[(10, 20)])]);
    let class_weights = vec![(7, weight(&[(1, &[(10, 20)]), (2, &[(30, 30)])]))];
    let group_weights = vec![(7, weight(&[(1, &[(10, 20)]), (3, &[(40, 40)])]))];

    assert!(sorted_weights_compatible_on_domain_intersection(
        &class_weights,
        &group_weights,
        &overlap,
        &overlap,
    ));
}

#[test]
fn transition_compat_rejects_class_only_label_active_on_overlap() {
    let overlap = weight(&[(1, &[(10, 20)])]);
    let class_weights = vec![(7, weight(&[(1, &[(10, 20)])]))];
    let group_weights = Vec::new();

    assert!(!sorted_weights_compatible_on_domain_intersection(
        &class_weights,
        &group_weights,
        &overlap,
        &overlap,
    ));
}

#[test]
fn transition_compat_rejects_group_only_label_active_on_overlap() {
    let overlap = weight(&[(1, &[(10, 20)])]);
    let class_weights = Vec::new();
    let group_weights = vec![(7, weight(&[(1, &[(10, 20)])]))];

    assert!(!sorted_weights_compatible_on_domain_intersection(
        &class_weights,
        &group_weights,
        &overlap,
        &overlap,
    ));
}

#[test]
fn transition_compat_rejects_same_target_shape_with_extra_active_label() {
    let overlap = weight(&[(1, &[(10, 20)])]);
    let class_weights = vec![(7, weight(&[(1, &[(10, 20)])]))];
    let group_weights = vec![
        (7, weight(&[(1, &[(10, 20)])])),
        (9, weight(&[(1, &[(10, 20)])])),
    ];

    assert!(!sorted_weights_compatible_on_domain_intersection(
        &class_weights,
        &group_weights,
        &overlap,
        &overlap,
    ));
}

#[test]
fn transition_compat_accepts_class_and_group_weights_disjoint_from_overlap() {
    let overlap = weight(&[(1, &[(10, 20)])]);
    let class_weights = vec![(7, weight(&[(2, &[(10, 20)])]))];
    let group_weights = vec![(9, weight(&[(3, &[(10, 20)])]))];

    assert!(sorted_weights_compatible_on_domain_intersection(
        &class_weights,
        &group_weights,
        &overlap,
        &overlap,
    ));
}

#[test]
fn minimize_acyclic_merges_overlapping_partial_transition_states_exactly() {
    let branch_weights = [
        Weight::from_uniform(0..=0, token_set(&[(1, 2)])),
        Weight::from_uniform(0..=0, token_set(&[(2, 3)])),
        Weight::from_uniform(0..=0, token_set(&[(3, 4)])),
    ];
    let mut start = DWAState::default();
    for (idx, label) in [10, 11, 12].into_iter().enumerate() {
        start.transitions.insert(label, ((idx + 1) as u32, Weight::all()));
    }

    let mut states = vec![start];
    for weight in &branch_weights {
        let mut branch = DWAState::default();
        branch.transitions.insert(20, (4, weight.clone()));
        states.push(branch);
    }
    let mut leaf = DWAState::default();
    leaf.final_weight = Some(Weight::all());
    states.push(leaf);
    let dwa = DWA::from_parts(states, 0);

    let words = [[10, 20], [11, 20], [12, 20], [10, 21], [11, 21], [12, 21]];
    let expected = words.map(|word| dwa.eval_word(&word));
    let minimized = minimize_acyclic(&dwa);

    assert_eq!(minimized.num_states(), 3);
    for (word, expected) in words.into_iter().zip(expected) {
        assert_eq!(minimized.eval_word(&word), expected, "word={word:?}");
    }
}

#[test]
fn materialized_profile_comparison_matches_intersection_profile_comparison() {
    let class_weights = vec![
        (7, weight(&[(0, &[(1, 3)]), (1, &[(4, 5)])])),
        (9, weight(&[(0, &[(8, 9)])])),
    ];
    let group_weights = vec![
        (7, weight(&[(0, &[(2, 4)]), (1, &[(4, 5)])])),
        (8, weight(&[(0, &[(10, 11)])])),
    ];
    let class_domain = weight(&[(0, &[(1, 3)]), (1, &[(4, 5)])]);
    let group_domain = weight(&[(0, &[(2, 4)]), (1, &[(4, 6)])]);
    let overlap = class_domain.intersection(&group_domain);

    assert_eq!(
        sorted_weights_compatible_on_domain_intersection(
            &class_weights,
            &group_weights,
            &class_domain,
            &group_domain,
        ),
        sorted_weights_compatible_on_domain(&class_weights, &group_weights, &overlap),
    );
}

#[test]
fn exact_group_summary_matches_memberwise_compatibility() {
    let needed = vec![
        weight(&[(0, &[(0, 9)])]),
        weight(&[(1, &[(0, 9)])]),
        weight(&[(1, &[(0, 9)])]),
    ];
    let members = vec![
        ClassProfile {
            targets: Vec::new(),
            weights: vec![(7, weight(&[(0, &[(2, 4)])]))],
            final_weight: None,
        },
        ClassProfile {
            targets: Vec::new(),
            weights: vec![(7, weight(&[(1, &[(3, 5)])]))],
            final_weight: None,
        },
    ];
    let summary = build_exact_group_summary(&[0, 1], &needed, &members);

    let compatible = ClassProfile {
        targets: Vec::new(),
        weights: vec![(7, weight(&[(1, &[(3, 5)])]))],
        final_weight: None,
    };
    let incompatible = ClassProfile {
        targets: Vec::new(),
        weights: vec![(7, weight(&[(1, &[(6, 8)])]))],
        final_weight: None,
    };

    for candidate in [&compatible, &incompatible] {
        let overlap = needed[2].intersection(&summary.needed_union);
        let via_summary = final_weights_compatible_on_domain(
            candidate.final_weight.as_ref(),
            summary.merged_final_weight.as_ref(),
            &overlap,
        ) && sorted_weights_compatible_on_domain(
            &candidate.weights,
            &summary.transition_weights,
            &overlap,
        );
        let via_members = memberwise_group_compatible(
            &needed[2],
            candidate,
            &[0, 1],
            &needed,
            &members,
        );
        assert_eq!(via_summary, via_members);
    }
    assert!(memberwise_group_compatible(
        &needed[2],
        &compatible,
        &[0, 1],
        &needed,
        &members,
    ));
    assert!(!memberwise_group_compatible(
        &needed[2],
        &incompatible,
        &[0, 1],
        &needed,
        &members,
    ));
}

#[test]
fn minimize_acyclic_helpers_match_materialized_overlap_for_empty_tsid_intersection() {
    let left = Weight::from_uniform(0..=1, token_set(&[(1, 3)]));
    let right = Weight::from_uniform(4..=5, token_set(&[(1, 3)]));
    let weight_a = weight(&[(0, &[(1, 2)]), (4, &[(2, 4)])]);
    let weight_b = weight(&[(1, &[(1, 1)]), (5, &[(3, 5)])]);

    assert_disjoint_matches_overlap(&weight_a, &left, &right);
    assert_disjoint_matches_overlap(&weight_b, &left, &right);
    assert_equal_matches_overlap(&weight_a, &weight_b, &left, &right);
}

#[test]
fn minimize_acyclic_helpers_match_materialized_overlap_for_disjoint_token_domains() {
    let left = Weight::from_uniform(0..=2, token_set(&[(1, 2)]));
    let right = Weight::from_uniform(0..=2, token_set(&[(5, 6)]));
    let weight_a = weight(&[(0, &[(1, 2)]), (1, &[(5, 6)])]);
    let weight_b = weight(&[(0, &[(2, 3)]), (2, &[(6, 7)])]);

    assert_disjoint_matches_overlap(&weight_a, &left, &right);
    assert_disjoint_matches_overlap(&weight_b, &left, &right);
    assert_equal_matches_overlap(&weight_a, &weight_b, &left, &right);
}

#[test]
fn minimize_acyclic_helpers_match_materialized_overlap_when_weight_range_is_missing() {
    let left = Weight::from_uniform(1..=3, token_set(&[(1, 4)]));
    let right = Weight::from_uniform(2..=4, token_set(&[(2, 5)]));
    let weight_a = weight(&[(2, &[(2, 4)])]);
    let weight_b = weight(&[(2, &[(2, 4)]), (3, &[(2, 4)])]);

    assert_disjoint_matches_overlap(&weight_a, &left, &right);
    assert_disjoint_matches_overlap(&weight_b, &left, &right);
    assert_equal_matches_overlap(&weight_a, &weight_b, &left, &right);
}

#[test]
fn minimize_acyclic_helpers_match_materialized_overlap_for_equal_weights_by_value() {
    let left = weight(&[(0, &[(1, 3)]), (1, &[(2, 4)])]);
    let right = weight(&[(0, &[(2, 5)]), (1, &[(1, 4)])]);
    let weight_a = weight(&[(0, &[(2, 3)]), (1, &[(2, 4)])]);
    let weight_b = weight(&[(0, &[(2, 3)]), (1, &[(2, 4)])]);

    assert_disjoint_matches_overlap(&weight_a, &left, &right);
    assert_disjoint_matches_overlap(&weight_b, &left, &right);
    assert_equal_matches_overlap(&weight_a, &weight_b, &left, &right);
}

#[test]
fn minimize_acyclic_helpers_match_materialized_overlap_when_difference_is_outside_overlap() {
    let left = Weight::from_uniform(0..=1, token_set(&[(1, 2)]));
    let right = Weight::from_uniform(0..=1, token_set(&[(1, 2)]));
    let weight_a = weight(&[(0, &[(1, 2), (5, 5)]), (1, &[(1, 2)])]);
    let weight_b = weight(&[(0, &[(1, 2)]), (1, &[(1, 2)])]);

    assert_disjoint_matches_overlap(&weight_a, &left, &right);
    assert_disjoint_matches_overlap(&weight_b, &left, &right);
    assert_equal_matches_overlap(&weight_a, &weight_b, &left, &right);
}

#[test]
fn minimize_acyclic_helpers_match_materialized_overlap_when_difference_is_inside_overlap() {
    let left = Weight::from_uniform(0..=1, token_set(&[(1, 3)]));
    let right = Weight::from_uniform(0..=1, token_set(&[(2, 4)]));
    let weight_a = weight(&[(0, &[(1, 2)]), (1, &[(2, 3)])]);
    let weight_b = weight(&[(0, &[(2, 3)]), (1, &[(2, 4)])]);

    assert_disjoint_matches_overlap(&weight_a, &left, &right);
    assert_disjoint_matches_overlap(&weight_b, &left, &right);
    assert_equal_matches_overlap(&weight_a, &weight_b, &left, &right);
}

/// Stable vs DescendingDomain must preserve the ORIGINAL input language:
/// find_difference(input, each) and find_difference(stable, descending)
/// are all None. Seeded DAGs: 6-12 states, sorted labels, overlapping
/// domains, divergent continuations, nontrivial finals, missing edges.
fn seeded_dag_input(seed: u64) -> DWA {
    let mut rng = seed.wrapping_add(1);
    let mut next = move || {
        rng ^= rng >> 12;
        rng ^= rng << 25;
        rng ^= rng >> 27;
        rng = rng.wrapping_mul(0x2545_f491_4f6c_dd1d);
        rng
    };
    let n_states = 6 + (next() % 7) as usize; // 6..=12
    let mut states = vec![DWAState::default(); n_states];
    let outer_domains: [&[(u32, &[(u32, u32)])]; 3] = [
        &[(0, &[(1, 8)]), (1, &[(3, 10)])],
        &[(0, &[(5, 12)]), (2, &[(1, 4)])],
        &[(1, &[(1, 6)]), (2, &[(7, 15)])],
    ];
    for i in 0..n_states {
        let mut label = 1i32;
        while label <= 4 {
            if next() % 4 != 0 && i + 1 < n_states {
                let span = (next() % (n_states - i - 1) as u64) as usize;
                let target = i + 1 + span;
                let dom = outer_domains[(next() % 3) as usize];
                let w = if next() % 2 == 0 {
                    weight(dom)
                } else {
                    weight(&[(0, &[(4, 9)]), (1, &[(8, 13)])])
                };
                states[i].transitions.insert(label, (target as u32, w));
            }
            label += 1;
        }
        if next() % 3 == 0 {
            states[i].final_weight =
                Some(weight(&[(0, &[(2, 5)]), (2, &[(9, 11)])]));
        }
        if next() % 5 == 0 {
            states[i].final_weight =
                Some(weight(&[(1, &[(1, 3)]), (3, &[(20, 25)])]));
        }
    }
    // Accept spine: label-5 chain with uniform weight plus matching final,
    // so every input has nonempty language by construction.
    let spine = weight(&[(0, &[(1, 8)])]);
    for i in 0..n_states - 1 {
        states[i].transitions.insert(5, (i as u32 + 1, spine.clone()));
    }
    states[n_states - 1].final_weight = Some(spine);
    let input = DWA::from_parts(states, 0);
    assert!(
        !input.eval_word(&vec![5; n_states - 1]).is_empty(),
        "seed {seed}: spine must accept",
    );
    input
}

fn assert_orders_preserve_language(seed: u64, input: &DWA) {
    use super::{minimize_acyclic_owned_with_pointwise_class_order, PointwiseClassOrder};
    let stable = minimize_acyclic_owned_with_pointwise_class_order(
        input.clone(),
        PointwiseClassOrder::Stable,
    );
    let descending = minimize_acyclic_owned_with_pointwise_class_order(
        input.clone(),
        PointwiseClassOrder::DescendingDomain,
    );
    for (name, minimized) in [("stable", &stable), ("descending", &descending)] {
        match find_difference(input, minimized) {
            Ok(None) => {}
            Ok(Some(word)) => {
                panic!("seed {seed} {name}: differs from input at word {word:?}")
            }
            Err(e) => panic!("seed {seed} {name}: checker error: {e:?}"),
        }
    }
    match find_difference(&stable, &descending) {
        Ok(None) => {}
        Ok(Some(word)) => {
            panic!("seed {seed}: stable vs descending differ at word {word:?}")
        }
        Err(e) => panic!("seed {seed}: order-pair checker error: {e:?}"),
    }
}

#[test]
fn minimize_pointwise_orders_preserve_language_seeded_family() {
    for seed in 0..64u64 {
        assert_orders_preserve_language(seed, &seeded_dag_input(seed));
    }
}

#[test]
fn pointwise_region_decode_reuse_matches_reference() {
    // Behavior ids are per-interner indices, so id 0 in `interner` and
    // id 0 in `other_interner` below denote DIFFERENT behaviors. The
    // decode cache is keyed by region Arc pointer and lives for one
    // `pointwise_groups_to_builders` call only, so the second call must
    // decode the same region Arc under its own interner and produce a
    // different weight. If the cache ever retained entries across calls,
    // the second call would wrongly reuse the first call's token sets.
    let mut interner = PointwiseBehaviorInterner::default();
    let final_only = interner.intern(true, Vec::new());
    let labeled = interner.intern(false, vec![(7, 3)]);
    let labeled_final = interner.intern(true, vec![(7, 3)]);
    let multi_a = interner.intern(false, vec![(7, 3), (9, 5)]);
    let mut other_interner = PointwiseBehaviorInterner::default();
    // Deliberately intern in a different order so shared ids collide:
    // id 0 = label 17 -> target 23 (not final), id 1 = final-only.
    let other_labeled = other_interner.intern(false, vec![(17, 23)]);
    let other_final_only = other_interner.intern(true, Vec::new());
    assert_eq!(final_only, 0);
    assert_eq!(other_labeled, 0);
    assert_eq!(other_final_only, 1);

    let region_final = Arc::new(vec![TokenBehaviorRange {
        start: 0,
        end: 4,
        behavior: final_only,
    }]);
    let region_labeled = Arc::new(vec![
        TokenBehaviorRange {
            start: 0,
            end: 2,
            behavior: labeled_final,
        },
        TokenBehaviorRange {
            start: 3,
            end: 9,
            behavior: labeled,
        },
    ]);
    let region_multi = Arc::new(vec![
        TokenBehaviorRange {
            start: 0,
            end: 3,
            behavior: multi_a,
        },
        TokenBehaviorRange {
            start: 100,
            end: 109,
            behavior: multi_a,
        },
    ]);
    let region_empty = Arc::new(Vec::new());
    // Structurally equal but distinct Arc: same decode, different pointer.
    let region_labeled_clone = Arc::new((*region_labeled).clone());
    assert!(!Arc::ptr_eq(&region_labeled, &region_labeled_clone));
    assert_eq!(&*region_labeled, &*region_labeled_clone);
    // Max-valid-TSID region. `u32::MAX` itself is the reserved
    // `WEIGHT_ALL_SENTINEL` / "no coordinate" marker (see
    // `shared_tokens_for_sorted_tsids`: `u32::MAX` always yields the empty
    // token set), so `u32::MAX - 1` is the largest TSID this path may
    // carry. The single-token region below pins that boundary exactly.
    let region_max = Arc::new(vec![TokenBehaviorRange {
        start: u32::MAX - 1,
        end: u32::MAX - 1,
        behavior: labeled_final,
    }]);

    // Direct Sparse construction (not via merge_profile) so the
    // structurally-equal distinct Arcs keep distinct pointers by
    // construction; the asserts below prove the distinction survived.
    let mut sparse_entries: rustc_hash::FxHashMap<u32, Arc<Vec<TokenBehaviorRange>>> =
        rustc_hash::FxHashMap::default();
    sparse_entries.insert(1, Arc::clone(&region_final));
    sparse_entries.insert(2, Arc::clone(&region_labeled));
    sparse_entries.insert(3, Arc::clone(&region_labeled));
    sparse_entries.insert(5, Arc::clone(&region_multi));
    sparse_entries.insert(6, Arc::clone(&region_empty));
    sparse_entries.insert(9, Arc::clone(&region_labeled_clone));
    sparse_entries.insert(11, Arc::clone(&region_max));
    assert!(Arc::ptr_eq(
        sparse_entries.get(&2).expect("tsid 2 present"),
        &region_labeled
    ));
    assert!(Arc::ptr_eq(
        sparse_entries.get(&3).expect("tsid 3 present"),
        &region_labeled
    ));
    assert!(!Arc::ptr_eq(
        sparse_entries.get(&2).expect("tsid 2 present"),
        sparse_entries.get(&9).expect("tsid 9 present")
    ));
    assert_eq!(
        sparse_entries.get(&2).expect("tsid 2 present").as_ref(),
        sparse_entries.get(&9).expect("tsid 9 present").as_ref()
    );
    let sparse_group = PointwiseMergeGroup {
        targets_by_label: [(7, 3), (9, 5)].into_iter().collect(),
        behavior_by_tsid: PointwiseBehaviorMap::Sparse(sparse_entries),
        member_classes: vec![0],
    };
    let mut dense_slots: Vec<Option<Arc<Vec<TokenBehaviorRange>>>> = vec![None; 12];
    dense_slots[0] = Some(Arc::clone(&region_multi));
    dense_slots[2] = Some(Arc::clone(&region_labeled));
    dense_slots[4] = Some(Arc::clone(&region_labeled_clone));
    dense_slots[7] = Some(Arc::clone(&region_final));
    assert!(!Arc::ptr_eq(
        dense_slots[2].as_ref().expect("slot 2 present"),
        dense_slots[4].as_ref().expect("slot 4 present")
    ));
    let dense_group = PointwiseMergeGroup {
        targets_by_label: [(7, 3), (9, 5)].into_iter().collect(),
        behavior_by_tsid: PointwiseBehaviorMap::Dense(dense_slots),
        member_classes: vec![1],
    };
    // Cross-group Arc sharing: same region_labeled Arc in both groups.
    let groups = vec![sparse_group, dense_group];

    let expected = pointwise_groups_to_builders_reference(&groups, &interner);
    let actual = pointwise_groups_to_builders(&groups, &interner);
    assert_eq!(expected.len(), actual.len());
    for (index, (want, got)) in expected.iter().zip(actual.iter()).enumerate() {
        assert_eq!(
            want.final_weights_pending, got.final_weights_pending,
            "group {index} final weights differ",
        );
        assert_eq!(
            want.transitions_pending.len(),
            got.transitions_pending.len(),
            "group {index} transition label count differs",
        );
        for (label, (want_target, want_weights)) in &want.transitions_pending {
            let (got_target, got_weights) = got
                .transitions_pending
                .get(label)
                .unwrap_or_else(|| panic!("group {index} missing label {label}"));
            assert_eq!(
                want_target, got_target,
                "group {index} label {label} target differs"
            );
            assert_eq!(
                want_weights, got_weights,
                "group {index} label {label} weights differ"
            );
        }
    }
    // Empty-region TSID 6 contributes no final and no transition rows.
    assert!(
        expected[0]
            .transitions_pending
            .values()
            .all(|(_, weights)| weights.iter().all(|w| !w.is_empty()))
    );

    // Cross-context regression: the SAME region Arc decodes under two
    // different interners where behavior id 0 means different things
    // (final-only above vs label 17 -> target 23 here). Each call must
    // decode with its own interner; a cache retained across calls would
    // leak the first call's final-only token set into the second call.
    let region_collision = Arc::new(vec![TokenBehaviorRange {
        start: 0,
        end: 5,
        behavior: 0,
    }]);
    let mut first_entries: rustc_hash::FxHashMap<u32, Arc<Vec<TokenBehaviorRange>>> =
        rustc_hash::FxHashMap::default();
    first_entries.insert(10, Arc::clone(&region_collision));
    let first_groups = vec![PointwiseMergeGroup {
        targets_by_label: rustc_hash::FxHashMap::default(),
        behavior_by_tsid: PointwiseBehaviorMap::Sparse(first_entries),
        member_classes: vec![0],
    }];
    let first_expected = pointwise_groups_to_builders_reference(&first_groups, &interner);
    let first_actual = pointwise_groups_to_builders(&first_groups, &interner);
    assert_eq!(
        first_expected[0].final_weights_pending,
        first_actual[0].final_weights_pending
    );
    assert!(first_actual[0].transitions_pending.is_empty());
    assert!(!first_actual[0].final_weights_pending.is_empty());

    let mut second_entries: rustc_hash::FxHashMap<u32, Arc<Vec<TokenBehaviorRange>>> =
        rustc_hash::FxHashMap::default();
    second_entries.insert(10, Arc::clone(&region_collision));
    let second_groups = vec![PointwiseMergeGroup {
        targets_by_label: [(17, 23)].into_iter().collect(),
        behavior_by_tsid: PointwiseBehaviorMap::Sparse(second_entries),
        member_classes: vec![0],
    }];
    let second_expected =
        pointwise_groups_to_builders_reference(&second_groups, &other_interner);
    let second_actual = pointwise_groups_to_builders(&second_groups, &other_interner);
    assert_eq!(
        second_expected[0].final_weights_pending,
        second_actual[0].final_weights_pending
    );
    assert_eq!(
        second_expected[0].transitions_pending,
        second_actual[0].transitions_pending
    );
    // Contexts genuinely differ: final-only vs label-17 transition.
    assert!(second_actual[0].final_weights_pending.is_empty());
    assert!(second_actual[0].transitions_pending.contains_key(&17));
    assert!(!first_actual[0].transitions_pending.contains_key(&17));
    assert_ne!(
        first_actual[0].final_weights_pending,
        second_actual[0].final_weights_pending
    );
}

#[test]
fn minimize_pointwise_orders_preserve_language_overlapping_partial_states() {
    // Two states share label 1 with overlapping (not equal) weights and
    // divergent continuations; label 2 missing from one; distinct finals.
    let mut states = vec![DWAState::default(); 5];
    states[0].transitions.insert(1, (1, weight(&[(0, &[(1, 8)])])));
    states[0].transitions.insert(2, (2, weight(&[(0, &[(1, 8)])])));
    states[1].transitions.insert(1, (3, weight(&[(0, &[(5, 12)])])));
    states[1].transitions.insert(3, (4, weight(&[(1, &[(1, 6)])])));
    states[2].transitions.insert(1, (3, weight(&[(0, &[(1, 4)])])));
    states[2].transitions.insert(4, (4, weight(&[(2, &[(7, 15)])])));
    states[3].final_weight = Some(weight(&[(0, &[(2, 5)]), (2, &[(9, 11)])]));
    states[4].final_weight = Some(weight(&[(1, &[(1, 3)]), (3, &[(20, 25)])]));
    assert_orders_preserve_language(u64::MAX, &DWA::from_parts(states, 0));
}
