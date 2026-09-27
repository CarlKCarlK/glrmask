use super::*;
use crate::compiler::stages::id_map_and_terminal_dwa::l2p::equivalence_analysis::compat::{
    FlatDfa, FlatDfaState,
};
use crate::compiler::stages::id_map_and_terminal_dwa::l2p::equivalence_analysis::shared::representative_tokens_for_vocab_classes;
use std::sync::Arc;

#[test]
fn vocab_first_selection_uses_large_state_or_large_pair_work() {
    assert!(prefer_vocab_first_equivalence(512, 256));
    assert!(prefer_vocab_first_equivalence(56_610, 40));
    assert!(prefer_vocab_first_equivalence(47_171, 61));
    assert!(prefer_vocab_first_equivalence(29_801, 31));

    assert!(!prefer_vocab_first_equivalence(4_657, 182));
    assert!(!prefer_vocab_first_equivalence(10_142, 40));
    assert!(!prefer_vocab_first_equivalence(511, usize::MAX));
}

#[test]
fn adaptive_nfa_view_skips_powerset_probe_below_bounded_work_threshold() {
    assert!(!should_probe_l2p_nfa_powerset(
        L2pNfaAnalysisViewPolicy::Adaptive,
        499_999,
        500_000,
    ));
    assert!(should_probe_l2p_nfa_powerset(
        L2pNfaAnalysisViewPolicy::Adaptive,
        500_000,
        500_000,
    ));
}

#[test]
fn adaptive_nfa_view_uses_only_small_probed_powersets() {
    assert!(should_use_l2p_nfa_powerset(
        L2pNfaAnalysisViewPolicy::Adaptive,
        true,
        8_192,
        8_192,
    ));
    assert!(!should_use_l2p_nfa_powerset(
        L2pNfaAnalysisViewPolicy::Adaptive,
        true,
        8_193,
        8_192,
    ));
    assert!(!should_use_l2p_nfa_powerset(
        L2pNfaAnalysisViewPolicy::Adaptive,
        false,
        0,
        8_192,
    ));
}

#[test]
fn forced_nfa_view_policies_override_adaptive_gates() {
    assert!(!should_probe_l2p_nfa_powerset(
        L2pNfaAnalysisViewPolicy::Bounded,
        usize::MAX,
        500_000,
    ));
    assert!(!should_use_l2p_nfa_powerset(
        L2pNfaAnalysisViewPolicy::Bounded,
        true,
        1,
        8_192,
    ));
    assert!(should_probe_l2p_nfa_powerset(
        L2pNfaAnalysisViewPolicy::Powerset,
        0,
        500_000,
    ));
    assert!(should_use_l2p_nfa_powerset(
        L2pNfaAnalysisViewPolicy::Powerset,
        true,
        usize::MAX,
        8_192,
    ));
}

fn partition_from_representatives<T: Ord + Copy>(
    values: &[T],
    representatives: &[T],
) -> BTreeSet<BTreeSet<T>> {
    let mut by_representative = BTreeMap::<T, BTreeSet<T>>::new();
    for (&value, &representative) in values.iter().zip(representatives) {
        by_representative.entry(representative).or_default().insert(value);
    }
    by_representative.into_values().collect()
}

fn synthetic_view() -> TokenizerView {
    let state_count = 5usize;
    let mut transitions = vec![u32::MAX; state_count * 256];
    let set = |transitions: &mut [u32], state: usize, byte: u8, target: u32| {
        transitions[state * 256 + byte as usize] = target;
    };
    set(&mut transitions, 0, b'a', 1);
    set(&mut transitions, 0, b'b', 2);
    set(&mut transitions, 1, b'a', 1);
    set(&mut transitions, 1, b'b', 3);
    set(&mut transitions, 2, b'a', 3);
    set(&mut transitions, 2, b'b', 2);
    set(&mut transitions, 3, b'a', 3);
    set(&mut transitions, 3, b'b', 3);
    // State 4 is behaviorally identical to state 1.
    set(&mut transitions, 4, b'a', 4);
    set(&mut transitions, 4, b'b', 3);

    TokenizerView {
        flat_dfa: FlatDfa {
            start_state: 0,
            transitions: Arc::from(transitions),
            states: vec![
                FlatDfaState {
                    finalizers: vec![],
                    possible_future_group_ids: vec![0, 1],
                },
                FlatDfaState {
                    finalizers: vec![0],
                    possible_future_group_ids: vec![0, 1],
                },
                FlatDfaState {
                    finalizers: vec![1],
                    possible_future_group_ids: vec![0, 1],
                },
                FlatDfaState {
                    finalizers: vec![0, 1],
                    possible_future_group_ids: vec![0, 1],
                },
                FlatDfaState {
                    finalizers: vec![0],
                    possible_future_group_ids: vec![0, 1],
                },
            ],
        },
    }
}

#[test]
fn state_then_vocab_equivalence_matches_vocab_then_state() {
    let view = synthetic_view();
    let tokens: Vec<&[u8]> = vec![
        b"a", b"b", b"aa", b"ab", b"ba", b"bb", b"x", b"y",
    ];
    let states: Vec<usize> = (0..view.dfa().states.len()).collect();
    let byte_to_class = super::super::compat::compute_byte_classes(view.dfa());
    let disallowed = BTreeMap::<u32, BitSet>::new();
    let normalized = normalize_disallowed_follows(2, &disallowed);

    let old_vocab = vocab_equivalence_analysis::find_vocab_equivalence_classes_with_group_filter(
        &view,
        &tokens,
        &states,
        &disallowed,
        Some(&byte_to_class),
        None,
        None,
        None,
    );
    let old_token_reps = representative_tokens_for_vocab_classes(&old_vocab, &tokens);
    let old_state_reps =
        state_equivalence_analysis::find_state_equivalence_classes_with_disallowed(
            &view,
            &old_token_reps,
            &states,
            &normalized,
        );

    let reversed_state_reps =
        state_equivalence_analysis::find_state_equivalence_classes_with_disallowed(
            &view,
            &tokens,
            &states,
            &normalized,
        );
    let final_state_reps: Vec<usize> = {
        let mut reps = reversed_state_reps.clone();
        reps.sort_unstable();
        reps.dedup();
        reps
    };
    let reversed_vocab = vocab_equivalence_analysis::find_vocab_equivalence_classes_with_group_filter(
        &view,
        &tokens,
        &final_state_reps,
        &disallowed,
        Some(&byte_to_class),
        None,
        None,
        None,
    );

    assert_eq!(
        partition_from_representatives(&states, &old_state_reps),
        partition_from_representatives(&states, &reversed_state_reps),
    );
    assert_eq!(old_vocab, reversed_vocab);
}

#[test]
fn first_byte_vocab_factorization_matches_full_exact_partition() {
    let view = synthetic_view();
    let tokens: Vec<&[u8]> = vec![
        b"a", b"b", b"ax", b"ay", b"bx", b"by", b"x", b"y",
    ];
    let states: Vec<usize> = (0..view.dfa().states.len()).collect();
    let byte_to_class = super::super::compat::compute_byte_classes(view.dfa());
    let mut disallowed = BTreeMap::<u32, BitSet>::new();
    let mut blocked = BitSet::new(2);
    blocked.set(1);
    disallowed.insert(0, blocked);

    let full = vocab_equivalence_analysis::find_vocab_equivalence_classes_with_group_filter(
        &view,
        &tokens,
        &states,
        &disallowed,
        Some(&byte_to_class),
        None,
        None,
        None,
    );
    let (factored, profile) = first_byte_factored_vocab_classes(
        &view,
        &tokens,
        &states,
        &disallowed,
        &byte_to_class,
    )
    .expect("test vocabulary should admit a nontrivial first-byte prepartition");

    assert!(profile.preliminary_classes < tokens.len());
    assert_eq!(factored, full);
}

#[test]
fn one_byte_common_prefix_factorization_matches_full_token_state_equivalence() {
    let view = synthetic_view();
    let tokens: Vec<&[u8]> = vec![b"aa", b"ab", b"aaa", b"abb"];
    let suffix_tokens: Vec<&[u8]> = tokens.iter().map(|token| &token[1..]).collect();
    let states: Vec<usize> = (0..view.dfa().states.len()).collect();
    let normalized = normalize_disallowed_follows(2, &BTreeMap::new());

    let full = state_equivalence_analysis::find_state_equivalence_classes_with_disallowed(
        &view,
        &tokens,
        &states,
        &normalized,
    );

    let mut prefix_targets = BTreeSet::<usize>::new();
    let targets = states
        .iter()
        .map(|&source| view.dfa().trans(source, b'a' as usize))
        .collect::<Vec<_>>();
    for &target in &targets {
        if target != u32::MAX {
            prefix_targets.insert(target as usize);
        }
    }
    let prefix_targets = prefix_targets.into_iter().collect::<Vec<_>>();
    let target_representatives = state_equivalence_analysis::
        find_state_equivalence_classes_with_disallowed_and_shared_base_with_initial_finalizers(
            &view,
            &suffix_tokens,
            &prefix_targets,
            &normalized,
            None,
        );
    let behavior_for_target = prefix_targets
        .iter()
        .copied()
        .zip(target_representatives)
        .collect::<BTreeMap<_, _>>();
    let behaviors = targets
        .iter()
        .map(|&target| {
            (target != u32::MAX).then(|| behavior_for_target[&(target as usize)])
        })
        .collect::<Vec<_>>();
    let mut representative_for_behavior = BTreeMap::<Option<usize>, usize>::new();
    for (&source, &behavior) in states.iter().zip(&behaviors) {
        representative_for_behavior.entry(behavior).or_insert(source);
    }
    let factored = behaviors
        .into_iter()
        .map(|behavior| representative_for_behavior[&behavior])
        .collect::<Vec<_>>();

    assert_eq!(
        partition_from_representatives(&states, &full),
        partition_from_representatives(&states, &factored),
    );
}

#[test]
fn directional_quotient_composition_retains_preclass_representative() {
    let directional = ManyToOneIdMap {
        original_to_internal: vec![0, 1, 0, 1],
        internal_to_originals: vec![vec![0, 2], vec![1, 3]],
        representative_original_ids: vec![2, 3],
    };

    let composed = compose_raw_quotient_state_map_preserving_directional_representatives(
        &directional,
        &[1, 1],
    );

    assert_eq!(composed.original_to_internal, vec![0, 0, 0, 0]);
    assert_eq!(composed.internal_to_originals, vec![vec![0, 1, 2, 3]]);
    assert_eq!(composed.representative_original_ids, vec![3]);
}

#[test]
fn bounded_vocab_pass_keeps_exact_analysis_view_representatives() {
    let preclasses = ManyToOneIdMap {
        original_to_internal: vec![0, 1, 0, 1],
        internal_to_originals: vec![vec![0, 2], vec![1, 3]],
        representative_original_ids: vec![2, 3],
    };
    let composed = compose_raw_quotient_state_map(&preclasses, &[1, 1]);
    let sparse_raw_start_to_view = [u32::MAX, u32::MAX, 10, 20];

    assert_eq!(composed.representative_original_ids, vec![0]);
    assert_eq!(
        sparse_raw_start_to_view[composed.representative_original_ids[0] as usize],
        u32::MAX,
    );
    assert_eq!(exact_analysis_view_representatives(&[20, 20]), vec![20]);
}

#[test]
fn lexical_dedup_reordering_preserves_representatives_and_remaps_inputs() {
    let tokens: Vec<&[u8]> = vec![b"z", b"ab", b"aa", b"ac"];
    let mut byte_to_class: [u8; 256] = std::array::from_fn(|byte| byte as u8);
    byte_to_class[b'c' as usize] = b'b';

    let ordinary = deduplicate_tokens_by_byte_class(&tokens, &byte_to_class, None);
    assert_eq!(ordinary.representative_original_indices, vec![0, 1, 2]);
    assert_eq!(ordinary.original_to_repr, vec![0, 1, 2, 1]);

    // Raw lexical order is aa, ab, ac, z. `ac` is projected-equivalent to
    // `ab`, so only the already-chosen representative for that class moves.
    let lexical = deduplicate_tokens_by_byte_class(
        &tokens,
        &byte_to_class,
        Some(&[2, 1, 3, 0]),
    );
    assert_eq!(lexical.representative_token_bytes, vec![b"aa".as_slice(), b"ab".as_slice(), b"z".as_slice()]);
    assert_eq!(lexical.representative_original_indices, vec![2, 1, 0]);
    assert_eq!(lexical.original_to_repr, vec![2, 1, 0, 1]);
    for left in 0..tokens.len() {
        for right in 0..tokens.len() {
            assert_eq!(
                ordinary.original_to_repr[left] == ordinary.original_to_repr[right],
                lexical.original_to_repr[left] == lexical.original_to_repr[right],
                "dedup relation changed for ({left}, {right})",
            );
        }
    }
}
#[test]
fn selects_direct_refinement_when_byte_bounded_prepass_cannot_amortize() {
    // Small vocabulary with many relevant bytes: direct token walks are
    // cheaper than a full k-bounded byte refinement per lexer state.
    assert!(direct_refinement_work_is_no_larger(100, 10, 41));
    // Larger vocabulary over a smaller byte alphabet amortizes the exact
    // prepass and should retain it.
    assert!(!direct_refinement_work_is_no_larger(900, 14, 19));
}
