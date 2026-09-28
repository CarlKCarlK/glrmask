use std::sync::Arc;

use super::{
    CompactMap, CompactOrdMap, GssSemanticKeyInterner, IndexedLeveledGss,
    IndexedLeveledGssNode, LeveledGSS, Lower, Merge, new_interface,
};

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct TestAcc(u32);

impl Merge for TestAcc {
    fn merge(&self, other: &Self) -> Self {
        Self(self.0.max(other.0))
    }
}

#[test]
fn filter_map_values_remaps_entire_stack_and_prunes_unmapped_paths() {
    let source = LeveledGSS::from_stacks(&[
        (vec![1_u32, 2, 3], TestAcc(1)),
        (vec![1, 4, 3], TestAcc(2)),
        (vec![7, 8], TestAcc(3)),
    ]);
    let mapped = source.filter_map_values(|value| match value {
        1 => Some(11_u32),
        2 => Some(12),
        3 => Some(13),
        4 => None,
        7 => Some(17),
        8 => Some(18),
        _ => None,
    });
    let mut actual = mapped.to_stacks(16).expect("small mapped GSS should enumerate");
    actual.sort();
    let mut expected = vec![
        (vec![11_u32, 12, 13], TestAcc(1)),
        (vec![17, 18], TestAcc(3)),
    ];
    expected.sort();
    assert_eq!(actual, expected);
}

#[test]
fn filter_map_values_merges_paths_that_map_to_same_stack() {
    let source = LeveledGSS::from_stacks(&[
        (vec![1_u32, 3], TestAcc(1)),
        (vec![2_u32, 3], TestAcc(9)),
    ]);
    let mapped = source.filter_map_values(|value| match value {
        1 | 2 => Some(10_u32),
        3 => Some(30),
        _ => None,
    });
    assert_eq!(
        mapped.to_stacks(4),
        Some(vec![(vec![10_u32, 30], TestAcc(9))]),
    );
}

#[test]
fn sorted_unique_single_value_frontier_reuses_lower_topology() {
    let values = [3_u32, 5, 7, 11];
    let frontier = LeveledGSS::from_sorted_unique_single_value_stacks(
        &values,
        TestAcc(1),
    );
    let lower_id = frontier
        .single_interface_lower_id()
        .expect("frontier should have one interface");

    let mut actual = frontier
        .to_stacks(16)
        .expect("small frontier should enumerate");
    actual.sort();
    let mut expected = values
        .iter()
        .map(|&value| (vec![value], TestAcc(1)))
        .collect::<Vec<_>>();
    expected.sort();
    assert_eq!(actual, expected);

    let updated = frontier
        .with_uniform_accumulator(TestAcc(9))
        .expect("interface accumulator should be replaceable");
    assert_eq!(updated.single_interface_lower_id(), Some(lower_id));
    let mut updated_actual = updated
        .to_stacks(16)
        .expect("small frontier should enumerate");
    updated_actual.sort();
    let mut updated_expected = values
        .iter()
        .map(|&value| (vec![value], TestAcc(9)))
        .collect::<Vec<_>>();
    updated_expected.sort();
    assert_eq!(updated_actual, updated_expected);
}

#[test]
fn unique_single_branch_constructor_matches_checked_constructor() {
    let base = LeveledGSS::from_single_stack(vec![1_u32, 2, 3], TestAcc(4));
    let targets = [10_u32, 20, 30, 40];
    let fast = base
        .try_virtual_stack()
        .expect("base should be a virtual stack")
        .into_gss_after_popping_and_pushing_unique_single_branches(
            1,
            targets.iter(),
        )
        .expect("unique branch construction should succeed");
    let checked = base
        .try_virtual_stack()
        .expect("base should be a virtual stack")
        .into_gss_after_popping_and_pushing_single_branches(1, targets.iter())
        .expect("checked branch construction should succeed");
    assert!(
        fast.semantically_eq(&checked, 64)
            .expect("small frontier should compare exactly")
    );
}

#[test]
fn indexed_dag_exactly_preserves_stack_language_and_accumulators() {
    fn enumerate_lower(
        dag: &IndexedLeveledGss<u32, TestAcc>,
        node: u32,
        top_first: &mut Vec<u32>,
        accumulator: &TestAcc,
        output: &mut Vec<(Vec<u32>, TestAcc)>,
    ) {
        match &dag.nodes[node as usize] {
            IndexedLeveledGssNode::LowerGeneral {
                empty, children, ..
            } => {
                if *empty {
                    let mut stack = top_first.clone();
                    stack.reverse();
                    output.push((stack, accumulator.clone()));
                }
                for (value, child) in children {
                    top_first.push(*value);
                    enumerate_lower(dag, *child, top_first, accumulator, output);
                    top_first.pop();
                }
            }
            IndexedLeveledGssNode::LowerSegment { values, next, .. } => {
                let old_len = top_first.len();
                top_first.extend(values.iter().rev().copied());
                enumerate_lower(dag, *next, top_first, accumulator, output);
                top_first.truncate(old_len);
            }
            IndexedLeveledGssNode::UpperBranch { .. }
            | IndexedLeveledGssNode::Interface { .. } => {
                panic!("lower traversal reached an upper node")
            }
        }
    }

    fn enumerate_upper(
        dag: &IndexedLeveledGss<u32, TestAcc>,
        node: u32,
        top_first: &mut Vec<u32>,
        output: &mut Vec<(Vec<u32>, TestAcc)>,
    ) {
        match &dag.nodes[node as usize] {
            IndexedLeveledGssNode::UpperBranch { empty, children } => {
                if let Some(accumulator) = empty {
                    let mut stack = top_first.clone();
                    stack.reverse();
                    output.push((stack, accumulator.clone()));
                }
                for (value, child) in children {
                    top_first.push(*value);
                    enumerate_upper(dag, *child, top_first, output);
                    top_first.pop();
                }
            }
            IndexedLeveledGssNode::Interface { accumulator, lower } => {
                enumerate_lower(dag, *lower, top_first, accumulator, output);
            }
            IndexedLeveledGssNode::LowerGeneral { .. }
            | IndexedLeveledGssNode::LowerSegment { .. } => {
                panic!("upper traversal reached a lower node")
            }
        }
    }

    let gss = LeveledGSS::from_stacks(&[
        (vec![0_u32, 1, 2, 7], TestAcc(1)),
        (vec![0, 1, 3, 7], TestAcc(2)),
        (vec![0, 4, 5, 8, 9], TestAcc(1)),
        (vec![0, 4, 6, 8, 9], TestAcc(3)),
        (vec![10], TestAcc(2)),
    ]);
    let dag = gss.indexed_dag();
    let mut actual = Vec::new();
    enumerate_upper(&dag, dag.root, &mut Vec::new(), &mut actual);
    actual.sort();
    let mut expected = gss
        .to_stacks(4_096)
        .expect("reference stack language exceeded explicit limit");
    expected.sort();
    assert_eq!(actual, expected);

    let merged = gss.merge(&LeveledGSS::from_stacks(&[
        (vec![0, 1, 2, 7], TestAcc(3)),
        (vec![11, 12], TestAcc(4)),
    ]));
    let (many, roots) = LeveledGSS::indexed_dag_many(&[gss.clone(), merged.clone()]);
    for (root, source) in roots.into_iter().zip([gss, merged]) {
        let mut actual = Vec::new();
        enumerate_upper(&many, root, &mut Vec::new(), &mut actual);
        actual.sort();
        let mut expected = source
            .to_stacks(4_096)
            .expect("reference stack language exceeded explicit limit");
        expected.sort();
        assert_eq!(actual, expected);
    }
}

#[test]

fn semantic_equality_compares_stack_accumulator_sets() {
    let left = LeveledGSS::merge_many([
        LeveledGSS::from_single_stack(vec![0_u32, 1, 7], TestAcc(1)),
        LeveledGSS::from_single_stack(vec![0_u32, 1, 9], TestAcc(2)),
    ]);
    let same = LeveledGSS::from_stacks(&[
        (vec![0_u32, 1, 9], TestAcc(2)),
        (vec![0_u32, 1, 7], TestAcc(1)),
    ]);
    let different_acc = LeveledGSS::from_stacks(&[
        (vec![0_u32, 1, 9], TestAcc(3)),
        (vec![0_u32, 1, 7], TestAcc(1)),
    ]);

    assert!(left.semantically_eq(&same, 4_096).expect("semantic comparison exceeded explicit stack limit"));
    assert!(same.semantically_eq(&left, 4_096).expect("semantic comparison exceeded explicit stack limit"));
    assert!(!left.semantically_eq(&different_acc, 4_096).expect("semantic comparison exceeded explicit stack limit"));
}

#[test]
fn to_stacks_returns_none_instead_of_truncating() {
    let gss = LeveledGSS::from_stacks(&[
        (vec![0_u32, 1], TestAcc(1)),
        (vec![0_u32, 2], TestAcc(2)),
        (vec![0_u32, 3], TestAcc(3)),
    ]);

    assert!(gss.to_stacks(2).is_none());
    assert_eq!(gss.to_stacks(3).unwrap().len(), 3);
}

#[test]
fn to_stacks_limit_stops_compressed_path_explosion() {
    let mut lower = Arc::new(Lower::General {
        children: CompactMap::new(),
        empty: true,
        max_depth: 0,
    });
    for depth in 0_u32..8 {
        let mut children = CompactMap::new();
        let child = CompactOrdMap::unit(depth, lower.clone());
        children.insert(0_u32, child.clone());
        children.insert(1_u32, child);
        lower = Arc::new(Lower::General {
            children,
            empty: false,
            max_depth: depth + 1,
        });
    }
    let gss = LeveledGSS {
        inner: new_interface(lower, TestAcc(0)),
    };

    // The DAG has only 41 Lower nodes but represents 2^40 concrete paths.
    // The explicit limit must stop after discovering the third path.
    assert!(gss.to_stacks(2).is_none());
}

#[test]
fn semantic_key_ignores_segment_general_layout_and_accumulators() {
    let segment = LeveledGSS::from_single_stack(vec![0_u32, 1], TestAcc(1));

    let floor = LeveledGSS::from_single_stack(Vec::<u32>::new(), TestAcc(9));
    let one_segment = LeveledGSS::from_single_stack(vec![0_u32], TestAcc(9));
    let one_general = one_segment.absorb_push_same_acc(0, &floor);
    let segment_over_general = one_general.push(1);
    let two_generals = segment_over_general.absorb_push_same_acc(1, &one_general);

    let segment_stacks = segment
        .to_stacks(1)
        .expect("single Segment stack must fit the explicit limit");
    let general_stacks = two_generals
        .to_stacks(1)
        .expect("single General-chain stack must fit the explicit limit");
    assert_eq!(segment_stacks[0].0, general_stacks[0].0);

    let mut interner = GssSemanticKeyInterner::new();
    assert_eq!(interner.key(&segment), interner.key(&two_generals));

    let different = LeveledGSS::from_single_stack(vec![0_u32, 2], TestAcc(1));
    assert_ne!(interner.key(&segment), interner.key(&different));
}

#[test]
fn semantic_key_deep_iterative_fallback_roundtrips() {
    let stack = (0_u32..96).collect::<Vec<_>>();
    let original = LeveledGSS::from_single_stack(stack.clone(), TestAcc(1));
    let mut interner = GssSemanticKeyInterner::new();

    let key = interner.key(&original);
    assert_ne!(key, 0);
    let rebuilt = interner.gss_from_key(key, TestAcc(9));
    let rebuilt_stacks = rebuilt
        .to_stacks(1)
        .expect("deep single language should remain one explicit stack");
    assert_eq!(rebuilt_stacks.len(), 1);
    assert_eq!(rebuilt_stacks[0].0, stack);
}

#[test]
fn semantic_key_exactly_matches_small_stack_languages() {
    let universe = [
        Vec::<u32>::new(),
        vec![0],
        vec![1],
        vec![0, 0],
        vec![0, 1],
        vec![1, 0],
    ];
    let mut interner = GssSemanticKeyInterner::new();
    let mut language_by_key = std::collections::HashMap::<u32, u32>::new();

    for language_bits in 0_u32..(1 << universe.len()) {
        let stacks = universe
            .iter()
            .enumerate()
            .filter(|(index, _)| language_bits & (1 << index) != 0)
            .map(|(index, stack)| (stack.clone(), TestAcc(index as u32)))
            .collect::<Vec<_>>();
        let canonical = LeveledGSS::from_stacks(&stacks);
        let reversed_merge = LeveledGSS::merge_many(
            stacks
                .iter()
                .rev()
                .map(|(stack, acc)| LeveledGSS::from_single_stack(stack.clone(), acc.clone())),
        );
        let different_accumulators = LeveledGSS::merge_many(
            stacks.iter().map(|(stack, _)| {
                LeveledGSS::from_single_stack(stack.clone(), TestAcc(100))
            }),
        );

        let key = interner.key(&canonical);
        assert_eq!(key, interner.key(&reversed_merge));
        assert_eq!(key, interner.key(&different_accumulators));
        assert_eq!(language_by_key.insert(key, language_bits), None);
    }
}

#[test]
fn semantic_key_stays_compressed_for_exponential_path_dag() {
    let mut lower = Arc::new(Lower::General {
        children: CompactMap::new(),
        empty: true,
        max_depth: 0,
    });
    for depth in 0_u32..40 {
        let mut children = CompactMap::new();
        let child = CompactOrdMap::unit(depth, lower.clone());
        children.insert(0_u32, child.clone());
        children.insert(1_u32, child);
        lower = Arc::new(Lower::General {
            children,
            empty: false,
            max_depth: depth + 1,
        });
    }
    let gss = LeveledGSS {
        inner: new_interface(lower, TestAcc(0)),
    };

    let mut interner = GssSemanticKeyInterner::new();
    assert_ne!(interner.key(&gss), 0);
    // Empty language + accepting floor + one canonical node per GSS level.
    assert_eq!(interner.node_count(), 42);
}

#[test]
fn node_count_at_most_counts_compact_nodes_without_expanding_paths() {
    let gss = LeveledGSS::from_stacks(&[
        (vec![0_u32, 1, 3], TestAcc(1)),
        (vec![0, 2, 3], TestAcc(1)),
    ]);
    let exact = gss.summary().total_unique_nodes;
    assert_eq!(gss.node_count_at_most(usize::MAX), exact);
    for limit in 0..=exact + 1 {
        assert_eq!(gss.node_count_at_most(limit), exact.min(limit));
    }
}

#[test]
fn bounded_accumulator_partition_declines_without_partial_output() {
    let gss = LeveledGSS::from_stacks(&[
        (vec![0_u32, 1], TestAcc(1)),
        (vec![0, 2], TestAcc(2)),
        (vec![0, 3], TestAcc(3)),
    ]);
    assert!(gss.partition_by_accumulator_at_most(2, 64).is_none());
    let partitions = gss
        .partition_by_accumulator_at_most(3, 64)
        .expect("three accumulator components fit the exact budget");
    assert_eq!(partitions.len(), 3);
    for (paths, accumulator) in partitions {
        let stacks = paths
            .to_stacks(4)
            .expect("each bounded partition has one explicit stack");
        assert_eq!(stacks.len(), 1);
        assert_eq!(stacks[0].1, ());
        assert_eq!(stacks[0].0.last().copied(), Some(accumulator.0));
    }
}

#[test]
fn semantic_key_budget_declines_without_partial_language() {
    let gss = LeveledGSS::from_single_stack(
        vec![0_u32, 1, 2, 3, 4],
        TestAcc(1),
    );
    let mut interner = GssSemanticKeyInterner::with_budget(3, 64, 64);
    assert_eq!(interner.key(&gss), 0);
    assert!(interner.is_exhausted());
    assert!(interner.node_count() <= 3);
}

#[test]
fn bounded_single_stack_accepts_deterministic_general_chain() {
    let floor = Arc::new(Lower::General {
        children: CompactMap::new(),
        empty: true,
        max_depth: 0,
    });
    let lower_zero = Arc::new(Lower::General {
        children: CompactMap::unit(0_u32, CompactOrdMap::unit(0, floor)),
        empty: false,
        max_depth: 1,
    });
    let lower_one = Arc::new(Lower::General {
        children: CompactMap::unit(1_u32, CompactOrdMap::unit(1, lower_zero)),
        empty: false,
        max_depth: 2,
    });
    let gss = LeveledGSS {
        inner: new_interface(lower_one, TestAcc(7)),
    };

    assert_eq!(
        gss.try_single_stack_bounded(2),
        Some((vec![0, 1], TestAcc(7)))
    );
    assert!(gss.try_single_stack_bounded(1).is_none());
}

#[test]
fn bounded_single_stack_rejects_very_deep_general_chain_in_constant_time() {
    const DEPTH: u32 = 100_000;
    let mut lower = Arc::new(Lower::General {
        children: CompactMap::new(),
        empty: true,
        max_depth: 0,
    });
    for value in 0..DEPTH {
        let child_depth = lower.max_depth();
        lower = Arc::new(Lower::General {
            children: CompactMap::unit(
                value,
                CompactOrdMap::unit(child_depth, lower),
            ),
            empty: false,
            max_depth: child_depth + 1,
        });
    }
    let gss = LeveledGSS {
        inner: new_interface(lower, TestAcc(0)),
    };

    assert_eq!(gss.max_depth(), DEPTH);
    assert!(gss.try_single_stack_bounded(256).is_none());

    let mut interner = GssSemanticKeyInterner::new();
    assert_ne!(interner.key(&gss), 0);
    assert_eq!(interner.node_count(), DEPTH as usize + 2);

    // Avoid recursively dropping deliberately pathological 100k-node chains.
    std::mem::forget(gss);
    std::mem::forget(interner);
}

#[test]
fn bounded_top_first_stack_traversal_matches_to_stacks() {
    let gss = LeveledGSS::from_stacks(&[
        (vec![0_u32, 1, 7], TestAcc(1)),
        (vec![0_u32, 2, 8, 9], TestAcc(2)),
        (vec![0_u32, 2, 8, 10], TestAcc(2)),
    ]);
    let expected = gss.to_stacks(4_096).expect("stack enumeration exceeded explicit limit");
    let mut actual = Vec::new();
    assert!(gss.for_each_stack_top_first_bounded(3, |top_first, acc| {
        let mut bottom_first = top_first.to_vec();
        bottom_first.reverse();
        actual.push((bottom_first, acc.clone()));
    }));
    assert_eq!(actual.len(), expected.len());
    for entry in &expected {
        assert!(actual.contains(entry));
    }
    for entry in &actual {
        assert!(expected.contains(entry));
    }

    let mut visited = 0usize;
    assert!(!gss.for_each_stack_top_first_bounded(2, |_, _| {
        visited += 1;
    }));
    assert_eq!(visited, 2);
}

#[test]
fn bounded_stack_length_traversal_matches_concrete_stacks() {
    let gss = LeveledGSS::from_stacks(&[
        (vec![0_u32, 1, 7], TestAcc(1)),
        (vec![0_u32, 2, 8, 9], TestAcc(2)),
        (vec![0_u32, 2, 8, 10], TestAcc(2)),
    ]);
    let mut expected = gss
        .to_stacks(4_096)
        .expect("stack enumeration exceeded explicit limit")
        .into_iter()
        .map(|(stack, acc)| (stack.len(), acc))
        .collect::<Vec<_>>();
    let mut actual = Vec::new();
    assert!(gss.for_each_stack_len_bounded(3, |len, acc| {
        actual.push((len, acc.clone()));
    }));
    expected.sort_unstable_by_key(|(len, acc)| (*len, acc.0));
    actual.sort_unstable_by_key(|(len, acc)| (*len, acc.0));
    assert_eq!(actual, expected);

    let mut visited = 0usize;
    assert!(!gss.for_each_stack_len_bounded(2, |_, _| {
        visited += 1;
    }));
    assert_eq!(visited, 2);
}

#[test]
fn isolate_top_value_preserves_branch_accumulator_correlation() {
    let gss = LeveledGSS::from_stacks(&[
        (vec![0_u32, 10, 20], TestAcc(1)),
        (vec![0_u32, 10, 21], TestAcc(2)),
    ]);

    assert_eq!(
        gss.isolate(Some(20)).to_stacks(4_096).expect("stack enumeration exceeded explicit limit"),
        vec![(vec![0_u32, 10, 20], TestAcc(1))],
    );
    assert_eq!(
        gss.isolate(Some(21)).to_stacks(4_096).expect("stack enumeration exceeded explicit limit"),
        vec![(vec![0_u32, 10, 21], TestAcc(2))],
    );
}

#[test]
fn branch_pure_shift_preserves_selected_accumulator_correlation() {
    let gss = LeveledGSS::from_stacks(&[
        (vec![0_u32, 10, 20], TestAcc(1)),
        (vec![0_u32, 10, 21], TestAcc(2)),
    ]);

    assert_eq!(
        gss.apply_top_pure_shifts([(20_u32, 40_u32, false)])
            .to_stacks(4_096).expect("stack enumeration exceeded explicit limit"),
        vec![(vec![0_u32, 10, 20, 40], TestAcc(1))],
    );
}

#[test]
fn merging_distinct_accumulator_branches_preserves_path_correlation() {
    let left = LeveledGSS::from_single_stack(vec![0_u32, 10, 20, 40], TestAcc(1));
    let right = LeveledGSS::from_single_stack(vec![0_u32, 46], TestAcc(2));
    let merged = left.merge(&right);
    let stacks = merged.to_stacks(4_096).expect("stack enumeration exceeded explicit limit");

    assert_eq!(stacks.len(), 2, "stacks={stacks:#?}");
    assert!(stacks.contains(&(vec![0_u32, 10, 20, 40], TestAcc(1))));
    assert!(stacks.contains(&(vec![0_u32, 46], TestAcc(2))));
}

#[test]
fn absorb_push_preserves_different_interface_accumulator_correlation() {
    let shifted = LeveledGSS::from_single_stack(vec![0_u32, 46], TestAcc(2));
    let base = LeveledGSS::from_single_stack(vec![0_u32, 10, 20], TestAcc(1));
    let absorbed = shifted.absorb_push_same_acc(40, &base);
    let stacks = absorbed.to_stacks(4_096).expect("stack enumeration exceeded explicit limit");

    assert_eq!(stacks.len(), 2, "stacks={stacks:#?}");
    assert!(stacks.contains(&(vec![0_u32, 10, 20, 40], TestAcc(1))));
    assert!(stacks.contains(&(vec![0_u32, 46], TestAcc(2))));
}

#[test]
fn absorb_push_preserves_push_when_receiver_is_branch() {
    let shifted = LeveledGSS::from_stacks(&[
        (vec![0_u32, 46], TestAcc(2)),
        (vec![0_u32, 47], TestAcc(3)),
    ]);
    let base = LeveledGSS::from_single_stack(vec![0_u32, 10, 20], TestAcc(1));
    let absorbed = shifted.absorb_push_same_acc(40, &base);
    let stacks = absorbed.to_stacks(4_096).expect("stack enumeration exceeded explicit limit");

    assert_eq!(stacks.len(), 3, "stacks={stacks:#?}");
    assert!(stacks.contains(&(vec![0_u32, 10, 20, 40], TestAcc(1))));
    assert!(stacks.contains(&(vec![0_u32, 46], TestAcc(2))));
    assert!(stacks.contains(&(vec![0_u32, 47], TestAcc(3))));

}

#[test]
fn apply_shared_pop_push_branches_matches_virtual_stack_branch_builder() {
    let gss = LeveledGSS::from_single_stack(vec![10_u32, 20, 30, 40], TestAcc(1));
    let pushes = [vec![50_u32, 60], vec![70_u32, 80], vec![90_u32, 60]];

    let expected = gss
        .try_virtual_stack()
        .unwrap()
        .into_gss_after_popping_and_pushing_branches(2, pushes.iter().map(|push| push.as_slice()))
        .unwrap();
    let actual = gss
        .apply_shared_pop_push_branches(2, pushes.iter().map(|push| push.as_slice()))
        .unwrap();

    assert_eq!(actual, expected);
    assert_eq!(actual.to_stacks(4_096).expect("stack enumeration exceeded explicit limit"), expected.to_stacks(4_096).expect("stack enumeration exceeded explicit limit"));
}

#[test]
fn apply_shared_pop_push_single_branches_deduplicates_targets() {
    let gss = LeveledGSS::from_single_stack(vec![10_u32, 20, 30, 40], TestAcc(1));
    let targets = [60_u32, 70, 60];

    let expected = LeveledGSS::from_stacks(&[
        (vec![10_u32, 20, 60], TestAcc(1)),
        (vec![10_u32, 20, 70], TestAcc(1)),
    ]);
    let actual = gss
        .apply_shared_pop_push_single_branches(2, targets.iter())
        .unwrap();

    let actual_stacks = actual.to_stacks(4_096).expect("stack enumeration exceeded explicit limit");
    let expected_stacks = expected.to_stacks(4_096).expect("stack enumeration exceeded explicit limit");
    assert_eq!(actual_stacks.len(), expected_stacks.len());
    for expected_stack in expected_stacks {
        assert!(actual_stacks.contains(&expected_stack));
    }
}

#[test]
fn shared_suffix_single_branches_share_one_lower_segment() {
    // Abstract MRE for the GSS shape seen in CFA o13029 before committing
    // token ` [],`: eleven stacks differ only in the top value and share a
    // four-state suffix. This is intentionally independent of LR parsing,
    // tokenization, commit, and the JSON Schema importer.
    //
    // Expected shape:
    //   Interface -> General(top values 100..110) -> one shared Segment [0,1,12,30]
    // The empty floor is also represented as a Lower::General.
    let acc = TestAcc(7);
    let stacks: Vec<_> = (100_u32..111)
        .map(|top| (vec![0_u32, 1, 12, 30, top], acc.clone()))
        .collect();
    let branched = LeveledGSS::from_stacks(&stacks);

    let summary = branched.summary();
    let flattened = branched.to_stacks(4_096).expect("stack enumeration exceeded explicit limit");

    assert_eq!(flattened.len(), 11, "flattened={flattened:#?}");
    assert_eq!(summary.top_values_count, 11, "summary={summary:#?} flattened={flattened:#?}");
    assert_eq!(summary.interface_nodes, 1, "summary={summary:#?} flattened={flattened:#?}");
    assert_eq!(summary.lower_general_nodes, 2, "summary={summary:#?} flattened={flattened:#?}");
    assert_eq!(summary.lower_segment_nodes, 1, "summary={summary:#?} flattened={flattened:#?}");
    assert_eq!(summary.max_depth, 5, "summary={summary:#?} flattened={flattened:#?}");
}

#[test]
fn shared_suffix_compaction_is_order_insensitive() {
    fn assert_compact(gss: LeveledGSS<u32, TestAcc>) {
        let summary = gss.summary();
        assert_eq!(summary.top_values_count, 11, "summary={summary:#?}");
        assert_eq!(summary.lower_general_nodes, 2, "summary={summary:#?}");
        assert_eq!(summary.lower_segment_nodes, 1, "summary={summary:#?}");
        assert_eq!(summary.max_depth, 5, "summary={summary:#?}");
    }

    let acc = TestAcc(7);
    let orders: [Vec<u32>; 3] = [
        (100_u32..111).collect(),
        (100_u32..111).rev().collect(),
        vec![104, 100, 110, 101, 108, 103, 106, 102, 109, 105, 107],
    ];

    for order in orders {
        let stacks: Vec<_> = order
            .iter()
            .map(|top| (vec![0_u32, 1, 12, 30, *top], acc.clone()))
            .collect();
        assert_compact(LeveledGSS::from_stacks(&stacks));

        let merged = LeveledGSS::merge_many(
            stacks
                .iter()
                .map(|(stack, acc)| LeveledGSS::from_single_stack(stack.clone(), acc.clone())),
        );
        assert_compact(merged);
    }
}

#[test]
fn selective_top_pure_shift_extracts_one_shared_prefix_path() {
    let acc = TestAcc(7);
    let gss = LeveledGSS::from_stacks(&[
        (vec![0_u32, 1, 17, 47, 74, 131], acc.clone()),
        (vec![0_u32, 1, 17, 47, 74, 132], acc.clone()),
        (vec![0_u32, 1, 17, 47, 74, 133], acc.clone()),
    ]);

    let shifted = gss
        .try_apply_selective_top_pure_shifts([(131_u32, 96_u32, false)])
        .unwrap();

    assert_eq!(
        shifted.to_stacks(4_096).expect("stack enumeration exceeded explicit limit"),
        vec![(vec![0_u32, 1, 17, 47, 74, 131, 96], acc)]
    );
}

#[test]
fn generic_top_pure_shift_matches_selective_shared_prefix_shape() {
    let acc = TestAcc(7);
    let gss = LeveledGSS::from_stacks(&[
        (vec![0_u32, 1, 17, 47, 74, 131], acc.clone()),
        (vec![0_u32, 1, 17, 47, 74, 132], acc.clone()),
        (vec![0_u32, 1, 17, 47, 74, 133], acc.clone()),
    ]);

    let shifted = gss.apply_top_pure_shifts([(131_u32, 96_u32, false)]);

    assert_eq!(
        shifted.to_stacks(4_096).expect("stack enumeration exceeded explicit limit"),
        vec![(vec![0_u32, 1, 17, 47, 74, 131, 96], acc)]
    );
}

#[test]
#[ignore]
fn bench_generic_top_pure_shift_shared_prefix_shape() {
    let acc = TestAcc(7);
    let gss = LeveledGSS::from_stacks(&[
        (vec![0_u32, 1, 17, 47, 74, 131], acc.clone()),
        (vec![0_u32, 1, 17, 47, 74, 132], acc.clone()),
        (vec![0_u32, 1, 17, 47, 74, 133], acc),
    ]);

    let iterations = 100_000u32;
    let start = std::time::Instant::now();
    let mut shifted = None;
    for _ in 0..iterations {
        shifted = Some(std::hint::black_box(&gss).apply_top_pure_shifts(std::hint::black_box([
            (131_u32, 96_u32, false),
        ])));
    }
    let elapsed = start.elapsed();
    let avg_ns = elapsed.as_nanos() / u128::from(iterations);
    let shifted = shifted.unwrap();

    println!(
        "generic_top_pure_shift_shared_prefix_shape: avg={}ns iterations={}",
        avg_ns, iterations
    );
    assert_eq!(
        shifted.to_stacks(4_096).expect("stack enumeration exceeded explicit limit"),
        vec![(vec![0_u32, 1, 17, 47, 74, 131, 96], TestAcc(7))]
    );
}
#[test]
fn partition_by_accumulator_preserves_path_correlation() {
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    struct Acc(u8);

    impl Merge for Acc {
        fn merge(&self, other: &Self) -> Self {
            Acc(self.0.max(other.0))
        }
    }

    let gss = LeveledGSS::from_stacks(&[
        (vec![1, 2], Acc(1)),
        (vec![1, 3], Acc(2)),
        (vec![4], Acc(1)),
    ]);
    let mut partitions = gss.partition_by_accumulator();
    partitions.sort_by_key(|(_, accumulator)| accumulator.0);

    let mut first = partitions.remove(0).0.to_stacks(4_096).expect("stack enumeration exceeded explicit limit");
    let mut second = partitions.remove(0).0.to_stacks(4_096).expect("stack enumeration exceeded explicit limit");
    first.sort();
    second.sort();
    assert_eq!(first, vec![(vec![1, 2], ()), (vec![4], ())]);
    assert_eq!(second, vec![(vec![1, 3], ())]);
}

#[test]
fn partial_segment_pop_fast_path_matches_literal_stacks() {
    for depth in [2, 3, 7, 16, 33, 65, 129] {
        let values = (0..depth as u32).collect::<Vec<_>>();
        let source = LeveledGSS::from_single_stack(values.clone(), TestAcc(7));
        for popped in 1..depth {
            let fast = source.popn_single_interface_path(popped as isize)
                .expect("a pop inside a deterministic segment needs no graph traversal");
            let expected = vec![(values[..depth - popped].to_vec(), TestAcc(7))];
            assert_eq!(fast.to_stacks(2), Some(expected.clone()));
            assert_eq!(source.popn(popped as isize).to_stacks(2), Some(expected));
        }
        assert_eq!(source.to_stacks(2), Some(vec![(values, TestAcc(7))]),
            "popping a shared graph must not mutate its source");
    }
}

#[test]
fn partial_segment_pop_preserves_a_shared_branching_tail() {
    let stacks = vec![
        (vec![10_u32, 1], TestAcc(7)),
        (vec![20, 2], TestAcc(7)),
        (vec![30, 3], TestAcc(7)),
    ];
    let mut source = LeveledGSS::from_stacks(&stacks);
    for value in 100..120 { source = source.push(value); }
    for popped in 1..20 {
        let fast = source.popn_single_interface_path(popped)
            .expect("only the common top segment is popped, not its branching tail");
        let mut actual = fast.to_stacks(8).unwrap();
        let mut expected = stacks.iter().map(|(bottom, acc)| {
            let mut values = bottom.clone();
            values.extend(100..120 - popped as u32);
            (values, acc.clone())
        }).collect::<Vec<_>>();
        actual.sort(); expected.sort();
        assert_eq!(actual, expected);
    }
}
