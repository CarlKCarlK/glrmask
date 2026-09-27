//! Exactness tests for the cap-free GSS x boundary-DWA product evaluator.
//!
//! Each test builds an adversarial ambiguous GSS with >128 distinct paths
//! over shared tails (which makes the old
//! `for_each_stack_top_first_bounded(128, _)` evaluator decline) and
//! compares the memoized DAG evaluator against a literal per-path
//! reference walk written independently in test code.

use super::{BoundaryMask64DagEvaluator, BoundaryWeightDagEvaluator};
use crate::compiler::glr::accumulator::TerminalsDisallowed;
use crate::compiler::glr::labels::{DEFAULT_LABEL, encode_positive_label};
use crate::compiler::glr::parser::ParserGSS;
use crate::compiler::stages::parser_dwa::SmallBoundaryDwa;

/// 2^8 = 256 stacks sharing a common bottom tail. Level `i` offers
/// values `{2*i, 2*i+1}`; the shared tail is `[900, 901, 902]`.
/// Bottom-to-top order in each stack.
fn adversarial_stacks() -> Vec<(Vec<u32>, TerminalsDisallowed)> {
    let mut stacks = Vec::new();
    for bits in 0..256u32 {
        let mut stack = vec![900, 901, 902];
        for level in 0..8u32 {
            let pick = if (bits >> level) & 1 == 0 {
                2 * level
            } else {
                2 * level + 1
            };
            stack.push(pick);
        }
        stacks.push((stack, TerminalsDisallowed::new()));
    }
    stacks
}

fn adversarial_gss() -> ParserGSS {
    ParserGSS::from_stacks(&adversarial_stacks())
}

/// Compact DWA exercising positive edges, DEFAULT fallback, prefix
/// finals, and a dead end. Single TSID, 6 tokens.
///
/// - state 0: final=[0,2]; +0 -> 1 (mask [0,1]); DEFAULT -> 2 (all).
/// - state 1: final=[1]; +2 -> 2 (mask [1,3]); no DEFAULT.
/// - state 2: final=[3]; no transitions.
fn adversarial_compact_dwa() -> SmallBoundaryDwa {
    let mut weights = vec![[0u64; 16]; 8];
    weights[1] = [0b00111111, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    weights[2] = [0b00000101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    weights[3] = [0b00000011, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    weights[4] = [0b00001010, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    weights[5] = [0b00000010, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    weights[6] = [0b00001000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    SmallBoundaryDwa {
        states: vec![
            crate::compiler::stages::parser_dwa::SmallBoundaryDwaState {
                transitions: vec![
                    (encode_positive_label(0), 1, 3),
                    (DEFAULT_LABEL, 2, 1),
                ],
                final_weight: 2,
            },
            crate::compiler::stages::parser_dwa::SmallBoundaryDwaState {
                transitions: vec![(encode_positive_label(2), 2, 4)],
                final_weight: 5,
            },
            crate::compiler::stages::parser_dwa::SmallBoundaryDwaState {
                transitions: vec![],
                final_weight: 6,
            },
        ],
        weights,
        tsid_count: 1,
        token_count: 6,
    }
}

/// Literal per-path reference walk for the compact representation,
/// mirroring `accepted_mask_for_stack` but written independently here.
fn reference_mask_for_stack(
    dwa: &SmallBoundaryDwa,
    tsid: u32,
    top_first: &[u32],
) -> u64 {
    if tsid >= dwa.tsid_count as u32 {
        return 0;
    }
    let mut state_id = dwa.start_state();
    let mut path_mask = dwa.all_token_mask();
    let mut accepted = 0u64;
    let accumulate = |state_id: u32, path_mask: u64, accepted: &mut u64| {
        let Some(state) = dwa.states.get(state_id as usize) else {
            return;
        };
        if state.final_weight != 0 {
            *accepted |= path_mask & dwa.weight_mask(state.final_weight, tsid);
        }
    };
    accumulate(state_id, path_mask, &mut accepted);
    for &parser_state in top_first {
        let label = encode_positive_label(parser_state);
        let Some(state) = dwa.states.get(state_id as usize) else {
            break;
        };
        let edge = state
            .transitions
            .iter()
            .find(|(edge_label, _, _)| *edge_label == label)
            .or_else(|| {
                state
                    .transitions
                    .iter()
                    .find(|(edge_label, _, _)| *edge_label == DEFAULT_LABEL)
            });
        let Some(&(_, target, weight)) = edge else {
            break;
        };
        path_mask &= dwa.weight_mask(weight, tsid);
        if path_mask == 0 {
            break;
        }
        state_id = target;
        accumulate(state_id, path_mask, &mut accepted);
    }
    accepted
}

fn reference_union_all_paths(
    dwa: &SmallBoundaryDwa,
    tsid: u32,
    gss: &ParserGSS,
    top_live: &dyn Fn(Option<u32>) -> bool,
) -> (u64, usize) {
    let mut union = 0u64;
    let mut count = 0usize;
    let complete = gss.for_each_stack_top_first_bounded(100_000, |top_first, _| {
        let live = match top_first.first().copied() {
            Some(top) => top_live(Some(top)),
            None => top_live(None),
        };
        if !live {
            return;
        }
        count += 1;
        union |= reference_mask_for_stack(dwa, tsid, top_first);
    });
    assert!(complete, "reference enumeration must complete");
    (union, count)
}

#[test]
fn compact_dag_evaluator_matches_literal_paths_past_128() {
    let gss = adversarial_gss();
    assert!(!gss.is_single_path());
    assert_eq!(gss.path_count_at_most(129), 129);
    // The old bounded evaluator declines on this GSS.
    assert!(!gss.for_each_stack_top_first_bounded(128, |_, _| {}));

    let dwa = adversarial_compact_dwa();
    let dag = gss.indexed_dag();
    let top_live = |_: Option<u32>| true;
    let (expected, path_count) = reference_union_all_paths(&dwa, 0, &gss, &top_live);
    assert_eq!(path_count, 256);

    let mut evaluator = BoundaryMask64DagEvaluator::new(&dwa, 0, &dag);
    let groups = evaluator.eval_root(&top_live);
    let mut actual = 0u64;
    for (group, mask) in &groups {
        // Single accumulator for every path: one correlated group.
        assert!(
            evaluator.group_accumulator(*group).is_some(),
            "every group resolves to an accumulator"
        );
        actual |= *mask;
    }
    assert_eq!(actual, expected);
}

#[test]
fn compact_dag_evaluator_respects_top_filter_and_default() {
    let gss = adversarial_gss();
    let dwa = adversarial_compact_dwa();
    let dag = gss.indexed_dag();
    // Tops are the level-7 values 14/15 (every stack ends with one of
    // them). Filter out top 14: half the paths use the +14 positive edge
    // (none here — 14 hits DEFAULT) and half keep top 15.
    let top_live = |top: Option<u32>| top.is_none_or(|value| value != 14);
    let (expected, path_count) = reference_union_all_paths(&dwa, 0, &gss, &top_live);
    assert!(path_count < 256 && path_count > 0);

    let mut evaluator = BoundaryMask64DagEvaluator::new(&dwa, 0, &dag);
    let groups = evaluator.eval_root(&top_live);
    let actual = groups.values().fold(0u64, |acc, mask| acc | *mask);
    assert_eq!(actual, expected);
}

#[test]
fn compact_dag_evaluator_keeps_per_path_acc_groups() {
    // Two accumulator classes over the same 256 stack shapes: the
    // evaluator must keep their acceptances in separate groups so the
    // caller can intersect each with its own eligibility set.
    let mut stacks = adversarial_stacks();
    for (index, (_, acc)) in stacks.iter_mut().enumerate() {
        if index % 2 == 0 {
            *acc = acc.clone().with_insert(7, 9);
        }
    }
    let gss = ParserGSS::from_stacks(&stacks);
    assert!(!gss.is_single_path());
    let dwa = adversarial_compact_dwa();
    let dag = gss.indexed_dag();
    let top_live = |_: Option<u32>| true;

    let mut evaluator = BoundaryMask64DagEvaluator::new(&dwa, 0, &dag);
    let groups = evaluator.eval_root(&top_live);
    assert!(
        groups.len() >= 2,
        "distinct accumulators stay in distinct groups, got {}",
        groups.len()
    );
    // Union over groups still equals the literal per-path union (the u64
    // acceptance itself is accumulator-independent; correlation is
    // enforced by the caller's per-group eligibility intersection).
    let (expected, _) = reference_union_all_paths(&dwa, 0, &gss, &top_live);
    let actual = groups.values().fold(0u64, |acc, mask| acc | *mask);
    assert_eq!(actual, expected);
}

#[test]
fn weight_dag_evaluator_matches_literal_paths_past_128() {
    use std::collections::BTreeSet;
    let gss = adversarial_gss();
    let compact = adversarial_compact_dwa();
    let dwa = compact.to_generic_dwa();
    let dag = gss.indexed_dag();
    let top_live = |_: Option<u32>| true;

    let mut evaluator = BoundaryWeightDagEvaluator::new(&dwa, &dag);
    let groups = evaluator.eval_root(&top_live);
    let mut actual = BTreeSet::new();
    for (group, weight) in &groups {
        assert!(
            evaluator.group_accumulator(*group).is_some(),
            "every group resolves to an accumulator"
        );
        let Some(tokens) = weight.token_set_for_tsid_ref(0) else {
            continue;
        };
        actual.extend(tokens.iter());
    }

    // Literal reference: per-path Weight walk over the generic DWA.
    let mut ops = crate::ds::weight::ScopedWeightOpCache::default();
    let mut expected = BTreeSet::new();
    let complete = gss.for_each_stack_top_first_bounded(100_000, |top_first, _| {
        let mut state_id = dwa.start_state();
        let mut path_weight = crate::ds::weight::Weight::all();
        let mut accepted = crate::ds::weight::Weight::empty();
        let accumulate = |state_id: u32,
                              path_weight: &crate::ds::weight::Weight,
                              accepted: &mut crate::ds::weight::Weight,
                              ops: &mut crate::ds::weight::ScopedWeightOpCache| {
            if let Some(final_weight) = dwa
                .states()
                .get(state_id as usize)
                .and_then(|state| state.final_weight.as_ref())
            {
                let contribution = ops.intersection(path_weight, final_weight);
                if !contribution.is_empty() {
                    *accepted = ops.union(accepted, &contribution);
                }
            }
        };
        accumulate(state_id, &path_weight, &mut accepted, &mut ops);
        for &parser_state in top_first {
            let label = encode_positive_label(parser_state);
            let Some(state) = dwa.states().get(state_id as usize) else {
                break;
            };
            let Some((target, edge_weight)) = state
                .transitions
                .get(&label)
                .or_else(|| state.transitions.get(&DEFAULT_LABEL))
            else {
                break;
            };
            path_weight = ops.intersection(&path_weight, edge_weight);
            if path_weight.is_empty() {
                break;
            }
            state_id = *target;
            accumulate(state_id, &path_weight, &mut accepted, &mut ops);
        }
        if let Some(tokens) = accepted.token_set_for_tsid_ref(0) {
            expected.extend(tokens.iter());
        }
    });
    assert!(complete, "reference enumeration must complete");
    assert_eq!(actual, expected);
}

#[test]
fn weight_dag_evaluator_empty_stack_filtering() {
    // A lone empty stack contributes exactly the empty-prefix final; the
    // top filter admitting/rejecting the empty stack toggles the group.
    let stacks = vec![(Vec::new(), TerminalsDisallowed::new())];
    let gss = ParserGSS::from_stacks(&stacks);
    let compact = adversarial_compact_dwa();
    let dwa = compact.to_generic_dwa();
    let dag = gss.indexed_dag();

    let accept_all = |_: Option<u32>| true;
    let mut evaluator = BoundaryWeightDagEvaluator::new(&dwa, &dag);
    let with_empty = evaluator.eval_root(&accept_all);
    assert_eq!(with_empty.len(), 1);
    let group_weight = with_empty.values().next().expect("one group");
    let tokens: Vec<u32> = group_weight
        .token_set_for_tsid_ref(0)
        .map(|set| set.iter().collect())
        .unwrap_or_default();
    // Empty-prefix final of state 0 is weights[2] = {0, 2}.
    assert_eq!(tokens, vec![0, 2]);

    let reject_empty = |top: Option<u32>| top.is_some();
    let mut evaluator = BoundaryWeightDagEvaluator::new(&dwa, &dag);
    let without_empty = evaluator.eval_root(&reject_empty);
    assert!(
        without_empty.is_empty(),
        "rejecting the empty stack drops its group"
    );
}

#[test]
fn shared_queue_boundary_providers_match_dag_oracles_with_correlated_groups() {
    let mut stacks = adversarial_stacks();
    for (index, (_, acc)) in stacks.iter_mut().enumerate() {
        if index % 2 == 0 { *acc = acc.clone().with_insert(7,9); }
    }
    stacks.push((Vec::new(), TerminalsDisallowed::new()));
    let compact = adversarial_compact_dwa();
    let generic = compact.to_generic_dwa();
    for gss in [ParserGSS::from_stacks(&stacks), ParserGSS::empty()] {
        for filter in 0..3 {
            let top_live = |top: Option<u32>| match filter {
                0 => true,
                1 => top.is_some_and(|value| value != 14),
                _ => false,
            };
            for (language, accumulator) in gss.partition_by_accumulator() {
                let original = language.apply(|_| accumulator.clone());
                let dag = original.indexed_dag();
                let mut oracle = BoundaryMask64DagEvaluator::new(&compact,0,&dag);
                let expected = oracle.eval_root(&top_live).values().fold(0,|a,b|a|b);
                let compact_actual = super::boundary_mask64_via_shared_queue(&compact,0,&language,&top_live)
                    .map_or(0,|a|a.0.iter().fold(0,|bits,(_,w)|bits|w[0]));
                let generic_actual = super::boundary_weight_mask_via_shared_queue(
                    &generic,&[0],compact.token_count as usize,&language,&top_live,
                ).map_or(0,|a|a.0.iter().fold(0,|bits,(_,w)|bits|w[0]));
                assert_eq!(compact_actual,expected,"compact group={accumulator:?} filter={filter}");
                assert_eq!(generic_actual,expected,"generic group={accumulator:?} filter={filter}");
            }
        }
    }
}

#[test]
fn shared_boundary_queue_keeps_tokenizer_coordinates_separate() {
    let mut compact = adversarial_compact_dwa();
    compact.tsid_count = 2;
    for (index, weight) in compact.weights.iter_mut().enumerate() {
        // Distinct per-coordinate languages, including a transition which
        // is dead only in coordinate 1. Merging coordinates would admit
        // tokens that the independent per-coordinate oracle rejects.
        weight[1] = if index == 3 {
            0
        } else {
            ((weight[0] << 1) | (weight[0] >> 5)) & 0b11_1111
        };
    }
    let generic = compact.to_generic_dwa();
    let mut stacks = adversarial_stacks();
    stacks.push((Vec::new(), TerminalsDisallowed::new()));
    for (index, (_, accumulator)) in stacks.iter_mut().enumerate() {
        if index % 3 == 0 {
            *accumulator = accumulator.clone().with_insert(7, 9);
        }
    }
    let gss = ParserGSS::from_stacks(&stacks);
    for filter in 0..3 {
        let top_live = |top: Option<u32>| match filter {
            0 => true,
            1 => top.is_some_and(|value| value != 14),
            _ => false,
        };
        for (language, accumulator) in gss.partition_by_accumulator() {
            let original = language.apply(|_| accumulator.clone());
            let dag = original.indexed_dag();
            for coordinates in [&[0, 1][..], &[1][..]] {
                let actual = super::boundary_weight_mask_via_shared_queue(
                    &generic,
                    coordinates,
                    compact.token_count as usize,
                    &language,
                    &top_live,
                );
                for tsid in 0..2 {
                    let expected = if coordinates.contains(&tsid) {
                        let mut oracle = BoundaryMask64DagEvaluator::new(
                            &compact, tsid, &dag,
                        );
                        oracle.eval_root(&top_live).values().fold(0, |a, b| a | b)
                    } else {
                        0
                    };
                    let observed = actual.as_ref().map_or(0, |allowed| {
                        allowed.0.iter()
                            .filter(|(coordinate, _)| *coordinate == tsid)
                            .fold(0, |bits, (_, words)| bits | words[0])
                    });
                    assert_eq!(observed, expected,
                        "coordinate={tsid} selected={coordinates:?} group={accumulator:?} filter={filter}");
                }
            }
        }
    }
}
