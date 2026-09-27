use super::*;
use crate::compiler::stages::id_map_and_terminal_dwa::l2p::equivalence_analysis::compat::{
    FlatDfa, FlatDfaState, TokenizerView,
};
use std::sync::Arc;

#[test]
fn sequential_trie_work_limit_relaxes_only_for_wide_pools() {
    assert_eq!(
        vocab_sequential_trie_work_max_for_threads(1),
        VOCAB_SEQUENTIAL_TRIE_WORK_MAX_DEFAULT
    );
    assert_eq!(
        vocab_sequential_trie_work_max_for_threads(31),
        VOCAB_SEQUENTIAL_TRIE_WORK_MAX_DEFAULT
    );
    assert_eq!(
        vocab_sequential_trie_work_max_for_threads(32),
        VOCAB_SEQUENTIAL_TRIE_WORK_MAX_LARGE_POOL
    );
    assert_eq!(
        vocab_sequential_trie_work_max_for_threads(96),
        VOCAB_SEQUENTIAL_TRIE_WORK_MAX_LARGE_POOL
    );
}

#[test]
fn first_transition_factor_bucket_policy_tracks_pool_and_state_domain() {
    assert!(!first_transition_factor_parallel_buckets_default(1, 1));
    assert!(first_transition_factor_parallel_buckets_default(4, 50_000));
    assert!(first_transition_factor_parallel_buckets_default(8, 50_000));
    assert!(first_transition_factor_parallel_buckets_default(12, 4_989));
    assert!(!first_transition_factor_parallel_buckets_default(12, 12_000));
}

#[test]
fn byte_classes_from_unpruned_view_are_not_valid_for_bounded_view() {
    fn view(include_bang: bool) -> TokenizerView {
        let mut transitions = vec![u32::MAX; 2 * 256];
        transitions[b'a' as usize] = 1;
        if include_bang {
            transitions[b'!' as usize] = 1;
        }
        TokenizerView {
            flat_dfa: FlatDfa {
                start_state: 0,
                transitions: Arc::from(transitions),
                states: vec![
                    FlatDfaState {
                        finalizers: vec![],
                        possible_future_group_ids: vec![0],
                    },
                    FlatDfaState {
                        finalizers: vec![0],
                        possible_future_group_ids: vec![],
                    },
                ],
            },
        }
    }

    let full = view(true);
    let bounded = view(false);
    let full_classes = compute_byte_classes(full.dfa());
    let bounded_classes = compute_byte_classes(bounded.dfa());

    assert_eq!(
        full_classes[b'!' as usize],
        full_classes[b'a' as usize],
        "the bytes are genuinely equivalent in the full view",
    );
    assert_ne!(
        bounded.dfa().trans(0, b'!' as usize),
        bounded.dfa().trans(0, b'a' as usize),
        "relevant-byte pruning splits that class in the bounded view",
    );
    assert_ne!(
        bounded_classes[b'!' as usize],
        bounded_classes[b'a' as usize],
        "classes recomputed from the bounded view must reflect that split",
    );
    let tokens = [b"a".as_slice(), b"aa".as_slice()];
    let classes = find_vocab_equivalence_classes_with_group_filter(
        &bounded,
        &tokens,
        &[0],
        &BTreeMap::new(),
        Some(&bounded_classes),
        None,
        None,
        None,
    );
    assert_eq!(classes, BTreeSet::from([vec![0], vec![1]]));
}

fn sample_dfa() -> FlatDfa {
    let mut transitions = vec![u32::MAX; 3 * 256];
    transitions[b'a' as usize] = 1;
    transitions[b'b' as usize] = 2;
    transitions[256 + b'a' as usize] = 1;
    transitions[256 + b'b' as usize] = 2;
    transitions[2 * 256 + b'a' as usize] = 2;
    transitions[2 * 256 + b'b' as usize] = 1;
    FlatDfa {
        states: vec![
            FlatDfaState {
                finalizers: vec![],
                possible_future_group_ids: vec![0],
            },
            FlatDfaState {
                finalizers: vec![0],
                possible_future_group_ids: vec![0],
            },
            FlatDfaState {
                finalizers: vec![1],
                possible_future_group_ids: vec![0, 1],
            },
        ],
        start_state: 0,
        transitions: Arc::from(transitions),
    }
}


fn first_transition_factor_dfa() -> FlatDfa {
    let state_count = 8usize;
    let mut transitions = vec![u32::MAX; state_count * 256];
    let mut set = |state: usize, byte: u8, target: usize| {
        transitions[state * 256 + byte as usize] = target as u32;
    };

    // 'a' and 'A' are one semantic byte class: their transition columns
    // are exactly equal. States 0 and 1 also share their first successor.
    for state in 0..state_count {
        let target = match state {
            0 | 1 | 2 | 3 => Some(2),
            4 => Some(4),
            5 => Some(5),
            _ => None,
        };
        if let Some(target) = target {
            set(state, b'a', target);
            set(state, b'A', target);
        }
    }
    for state in 0..=5 {
        let target = match state {
            0 | 1 | 2 | 3 => 3,
            4 => 4,
            5 => 5,
            _ => unreachable!(),
        };
        set(state, b'b', target);
    }

    // 'p' and 'q' use different transition columns, but their destinations
    // 4 and 5 have identical observations and continuation behavior. The
    // final authority pass must therefore be able to merge across leading
    // semantic classes.
    for state in 0..=5 {
        let p_target = if state == 5 { 5 } else { 4 };
        let q_target = if state == 4 { 4 } else { 5 };
        set(state, b'p', p_target);
        set(state, b'q', q_target);
    }
    set(2, b'x', 6);
    set(3, b'x', 6);
    set(2, b'y', 4);
    set(3, b'y', 5);
    set(0, b'x', 7);
    set(0, b'y', 7);

    FlatDfa {
        states: vec![
            FlatDfaState {
                finalizers: vec![],
                possible_future_group_ids: vec![0, 1],
            },
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
                finalizers: vec![0],
                possible_future_group_ids: vec![0],
            },
            FlatDfaState {
                finalizers: vec![0],
                possible_future_group_ids: vec![0],
            },
            FlatDfaState {
                finalizers: vec![1],
                possible_future_group_ids: vec![],
            },
            FlatDfaState {
                finalizers: vec![],
                possible_future_group_ids: vec![],
            },
        ],
        start_state: 0,
        transitions: Arc::from(transitions),
    }
}


#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ExactSuffixObservation {
    completion: Vec<usize>,
    edges: Vec<(usize, Option<Box<ExactSuffixObservation>>)>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ExactStateObservation {
    completion: Vec<usize>,
    edges: Vec<(usize, Option<Box<ExactSuffixObservation>>)>,
}

fn exact_completion(
    dfa: &Dfa,
    state: Option<usize>,
    disallowed: Option<&BitSet>,
) -> Vec<usize> {
    let Some(state) = state else {
        return Vec::new();
    };
    dfa.possible_future_groups[state]
        .iter()
        .copied()
        .filter(|&gid| !disallowed.is_some_and(|blocked| blocked.contains(gid)))
        .collect()
}

fn exact_run_from_state(
    dfa: &Dfa,
    token: &[u8],
    initial_state: usize,
    include_initial_finalizers: bool,
) -> (Option<usize>, Vec<Option<usize>>) {
    let mut latest = vec![None; dfa.num_groups];
    let mut current = initial_state;
    let mut done = dfa.is_dead_end[current];
    if include_initial_finalizers {
        for &gid in &dfa.finalizers[current] {
            if gid < latest.len() {
                latest[gid] = Some(0);
            }
        }
    }
    for (offset, &byte) in token.iter().enumerate() {
        if done {
            break;
        }
        let next = dfa.transition(current, byte);
        if next == NONE {
            done = true;
            break;
        }
        current = next as usize;
        for &gid in &dfa.finalizers[current] {
            if gid < latest.len() {
                latest[gid] = Some(offset + 1);
            }
        }
        if dfa.is_dead_end[current] {
            done = true;
        }
    }
    ((!done).then_some(current), latest)
}

fn exact_intersect_disallowed(
    slots: &mut BTreeMap<usize, BitSet>,
    position: usize,
    incoming: &BitSet,
) {
    slots
        .entry(position)
        .and_modify(|existing| *existing = existing.intersection(incoming))
        .or_insert_with(|| incoming.clone());
}

fn exact_state_observation(
    dfa: &Dfa,
    token: &[u8],
    initial_state: usize,
) -> ExactStateObservation {
    let (end_state, latest) = exact_run_from_state(dfa, token, initial_state, false);
    let root_matches = latest
        .into_iter()
        .enumerate()
        .filter_map(|(gid, position)| {
            position
                .filter(|&position| position > 0)
                .map(|position| (gid, position))
        })
        .collect::<Vec<_>>();
    let mut root_gids = BTreeMap::<usize, Vec<usize>>::new();
    for &(gid, position) in &root_matches {
        root_gids.entry(position).or_default().push(gid);
    }

    let mut suffix_runs = BTreeMap::<usize, (Option<usize>, Vec<(usize, usize)>)>::new();
    let mut pending = root_gids
        .keys()
        .copied()
        .filter(|&position| position < token.len())
        .collect::<BTreeSet<_>>();
    while let Some(position) = pending.pop_first() {
        if suffix_runs.contains_key(&position) {
            continue;
        }
        let (suffix_end, suffix_latest) = exact_run_from_state(
            dfa,
            &token[position..],
            dfa.start_state,
            true,
        );
        let mut edges = suffix_latest
            .into_iter()
            .enumerate()
            .filter_map(|(gid, relative)| {
                relative
                    .filter(|&relative| relative > 0)
                    .map(|relative| (gid, position + relative))
            })
            .collect::<Vec<_>>();
        edges.sort_unstable();
        for &(_, target) in &edges {
            if target < token.len() && !suffix_runs.contains_key(&target) {
                pending.insert(target);
            }
        }
        suffix_runs.insert(position, (suffix_end, edges));
    }

    let mut disallowed_at = BTreeMap::<usize, BitSet>::new();
    for (&position, gids) in &root_gids {
        let mut rows = gids.iter().map(|&gid| dfa.disallowed_for(gid));
        if let Some(first) = rows.next() {
            let mut combined = first.clone();
            for row in rows {
                combined = combined.intersection(row);
            }
            disallowed_at.insert(position, combined);
        }
    }
    for (&position, (_, edges)) in &suffix_runs {
        let blocked = disallowed_at.get(&position).cloned();
        for &(gid, target) in edges {
            if blocked.as_ref().is_some_and(|blocked| blocked.contains(gid)) {
                continue;
            }
            if target < token.len() {
                exact_intersect_disallowed(
                    &mut disallowed_at,
                    target,
                    dfa.disallowed_for(gid),
                );
            }
        }
    }

    let mut built = BTreeMap::<usize, ExactSuffixObservation>::new();
    for (&position, (suffix_end, edges)) in suffix_runs.iter().rev() {
        let blocked = disallowed_at.get(&position);
        let exact_edges = edges
            .iter()
            .filter(|(gid, _)| !blocked.is_some_and(|blocked| blocked.contains(*gid)))
            .map(|&(gid, target)| {
                let child = (target < token.len())
                    .then(|| Box::new(built[&target].clone()));
                (gid, child)
            })
            .collect();
        built.insert(
            position,
            ExactSuffixObservation {
                completion: exact_completion(dfa, *suffix_end, blocked),
                edges: exact_edges,
            },
        );
    }

    let edges = root_matches
        .into_iter()
        .map(|(gid, position)| {
            let child = (position < token.len())
                .then(|| Box::new(built[&position].clone()));
            (gid, child)
        })
        .collect();
    ExactStateObservation {
        completion: exact_completion(dfa, end_state, None),
        edges,
    }
}

fn exact_vocab_partition(
    dfa: &Dfa,
    tokens: &[impl AsRef<[u8]>],
    states: &[usize],
) -> VocabEquivalenceResult {
    let mut classes = BTreeMap::<Vec<ExactStateObservation>, Vec<usize>>::new();
    for (token_idx, token) in tokens.iter().enumerate() {
        let key = states
            .iter()
            .map(|&state| exact_state_observation(dfa, token.as_ref(), state))
            .collect::<Vec<_>>();
        classes.entry(key).or_default().push(token_idx);
    }
    classes.into_values().collect()
}

#[test]
fn first_transition_factor_matches_ordinary_exact_partition() {
    let view = TokenizerView {
        flat_dfa: first_transition_factor_dfa(),
    };
    let tokens: Vec<&[u8]> = vec![
        b"a", b"A", b"aa", b"Aa", b"ax", b"Ax", b"ay", b"Ay", b"b", b"bx",
        b"by", b"p", b"q", b"pp", b"qq", b"c", b"d", b"ax", b"x", b"x",
    ];
    let states = (0..view.dfa().states.len()).collect::<Vec<_>>();
    let byte_classes = compute_byte_classes(view.dfa());
    let mut blocked = BitSet::new(2);
    blocked.set(1);
    let disallowed = BTreeMap::from([(0u32, blocked)]);

    let (ordinary, _) = find_vocab_equivalence_classes_with_group_filter_profiled_impl(
        &view,
        &tokens,
        &states,
        &disallowed,
        Some(&byte_classes),
        None,
        None,
        None,
        FirstTransitionFactorMode::Disabled,
        None,
        false,
    );
    let (factored, _) = find_vocab_equivalence_classes_with_group_filter_profiled_impl(
        &view,
        &tokens,
        &states,
        &disallowed,
        Some(&byte_classes),
        None,
        None,
        None,
        FirstTransitionFactorMode::Force,
        None,
        false,
    );
    let analysis_dfa =
        build_dfa_with_group_filter(&view, &disallowed, Some(&byte_classes), None, None);
    let direct = exact_vocab_partition(&analysis_dfa, &tokens, &states);
    let mut refine_states = |preliminary_tokens: &[usize]| {
        let mut class_positions = BTreeMap::<Vec<ExactStateObservation>, usize>::new();
        for (position, &state) in states.iter().enumerate() {
            let key = preliminary_tokens
                .iter()
                .map(|&token| exact_state_observation(&analysis_dfa, tokens[token], state))
                .collect::<Vec<_>>();
            class_positions.entry(key).or_insert(position);
        }
        class_positions.into_values().collect::<Vec<_>>()
    };
    let (staged, _) = find_vocab_equivalence_classes_with_group_filter_profiled_impl(
        &view,
        &tokens,
        &states,
        &disallowed,
        Some(&byte_classes),
        None,
        None,
        None,
        FirstTransitionFactorMode::Force,
        Some(&mut refine_states),
        false,
    );

    assert_eq!(ordinary, direct, "ordinary hash partition must match direct observations");
    assert_eq!(factored, direct, "factored partition must match direct observations");
    assert_eq!(staged, direct, "staged factor/state refinement must remain exact");
    assert!(
        factored
            .iter()
            .any(|class| class.contains(&4) && class.contains(&17)),
        "duplicate byte strings must merge: {factored:?}",
    );
    assert!(
        factored.iter().any(|class| class.contains(&11) && class.contains(&12)),
        "the final authority pass must merge equivalent tokens from different leading classes",
    );
    assert!(
        factored.iter().any(|class| class.contains(&18) && class.contains(&19)),
        "a first transition into a finalizing dead-end state must be preserved",
    );
}


#[test]
fn first_transition_factor_randomized_direct_differential() {
    fn next(random: &mut u64) -> u64 {
        *random = random
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *random
    }

    const ALPHABET: [u8; 7] = [b'a', b'A', b'b', b'p', b'q', b'x', b'y'];
    for seed in 0..48u64 {
        let mut random = seed.wrapping_add(0x9e3779b97f4a7c15);
        let state_count = 3 + (next(&mut random) as usize % 6);
        let group_count = 1 + (next(&mut random) as usize % 4);
        let mut transitions = vec![u32::MAX; state_count * 256];
        for state in 0..state_count {
            for &byte in &[b'a', b'b', b'p', b'q', b'x', b'y'] {
                let draw = next(&mut random);
                if draw % 5 != 0 {
                    transitions[state * 256 + byte as usize] =
                        (draw as usize % state_count) as u32;
                }
            }
            // Guarantee one nontrivial semantic byte class in every case.
            transitions[state * 256 + b'A' as usize] =
                transitions[state * 256 + b'a' as usize];
        }
        let states = (0..state_count)
            .map(|_| {
                let mut finalizers = (0..group_count)
                    .filter(|_| next(&mut random) % 4 == 0)
                    .collect::<Vec<_>>();
                let mut future = (0..group_count)
                    .filter(|_| next(&mut random) % 3 != 0)
                    .collect::<Vec<_>>();
                finalizers.sort_unstable();
                future.sort_unstable();
                FlatDfaState {
                    finalizers,
                    possible_future_group_ids: future,
                }
            })
            .collect::<Vec<_>>();
        let view = TokenizerView {
            flat_dfa: FlatDfa {
                states,
                start_state: next(&mut random) as usize % state_count,
                transitions: Arc::from(transitions),
            },
        };
        let mut tokens = vec![
            b"a".to_vec(),
            b"A".to_vec(),
            b"aa".to_vec(),
            b"Aa".to_vec(),
            b"ax".to_vec(),
            b"Ax".to_vec(),
            b"p".to_vec(),
            b"q".to_vec(),
            b"ax".to_vec(),
        ];
        for _ in 0..24 {
            let len = 1 + next(&mut random) as usize % 4;
            let token = (0..len)
                .map(|_| ALPHABET[next(&mut random) as usize % ALPHABET.len()])
                .collect::<Vec<_>>();
            tokens.push(token);
        }
        let initial_states = (0..state_count).collect::<Vec<_>>();
        let mut disallowed = BTreeMap::<u32, BitSet>::new();
        for gid in 0..group_count {
            let mut row = BitSet::new(group_count);
            for blocked in 0..group_count {
                if next(&mut random) % 4 == 0 {
                    row.set(blocked);
                }
            }
            if !row.is_zero() {
                disallowed.insert(gid as u32, row);
            }
        }
        let byte_classes = compute_byte_classes(view.dfa());
        let (ordinary, _) = find_vocab_equivalence_classes_with_group_filter_profiled_impl(
            &view,
            &tokens,
            &initial_states,
            &disallowed,
            Some(&byte_classes),
            None,
            None,
            None,
            FirstTransitionFactorMode::Disabled,
            None,
            false,
        );
        let (factored, _) = find_vocab_equivalence_classes_with_group_filter_profiled_impl(
            &view,
            &tokens,
            &initial_states,
            &disallowed,
            Some(&byte_classes),
            None,
            None,
            None,
            FirstTransitionFactorMode::Force,
            None,
            false,
        );
        let analysis_dfa = build_dfa_with_group_filter(
            &view,
            &disallowed,
            Some(&byte_classes),
            None,
            None,
        );

        assert_eq!(factored, ordinary, "factored/reference mismatch at seed {seed}");

        // Directly certify the quotient invariant without relying on hash
        // equality: within one semantic first-byte class, sources with the
        // same effective first outcome must have identical structural
        // observations for the complete token.
        for token in &tokens {
            let first_byte = token[0];
            let mut observations = BTreeMap::<u32, ExactStateObservation>::new();
            for &source in &initial_states {
                let effective_target = if analysis_dfa.is_dead_end[source] {
                    NONE
                } else {
                    analysis_dfa.transition(source, first_byte)
                };
                let observation =
                    exact_state_observation(&analysis_dfa, token, source);
                if let Some(previous) = observations.insert(effective_target, observation.clone()) {
                    assert_eq!(
                        observation, previous,
                        "first-successor invariant mismatch at seed {seed} token={token:?} source={source}",
                    );
                }
            }
        }

        // Bytes in one semantic class must be interchangeable at the first
        // position when the suffix is held fixed.
        assert_eq!(byte_classes[b'a' as usize], byte_classes[b'A' as usize]);
        for suffix in [b"".as_slice(), b"x", b"ay", b"pq"] {
            let mut lower = vec![b'a'];
            lower.extend_from_slice(suffix);
            let mut upper = vec![b'A'];
            upper.extend_from_slice(suffix);
            for &source in &initial_states {
                assert_eq!(
                    exact_state_observation(&analysis_dfa, &lower, source),
                    exact_state_observation(&analysis_dfa, &upper, source),
                    "semantic leading-byte mismatch at seed {seed} source={source} suffix={suffix:?}",
                );
            }
        }
    }
}

#[test]
fn first_transition_factor_falls_back_for_empty_tokens() {
    let view = TokenizerView {
        flat_dfa: first_transition_factor_dfa(),
    };
    let tokens: Vec<&[u8]> = vec![b"", b"a", b"A", b"ax", b"ax"];
    let states = (0..view.dfa().states.len()).collect::<Vec<_>>();
    let byte_classes = compute_byte_classes(view.dfa());
    let disallowed = BTreeMap::new();
    let (ordinary, _) = find_vocab_equivalence_classes_with_group_filter_profiled_impl(
        &view,
        &tokens,
        &states,
        &disallowed,
        Some(&byte_classes),
        None,
        None,
        None,
        FirstTransitionFactorMode::Disabled,
        None,
        false,
    );
    let (forced, _) = find_vocab_equivalence_classes_with_group_filter_profiled_impl(
        &view,
        &tokens,
        &states,
        &disallowed,
        Some(&byte_classes),
        None,
        None,
        None,
        FirstTransitionFactorMode::Force,
        None,
        false,
    );
    assert_eq!(forced, ordinary);
}

fn padded_sample_dfa() -> FlatDfa {
    let mut dfa = sample_dfa();
    let target_state_count = 600;
    dfa.states.resize_with(target_state_count, || FlatDfaState {
        finalizers: Vec::new(),
        possible_future_group_ids: Vec::new(),
    });
    let mut transitions = dfa.transitions.to_vec();
    transitions.resize(target_state_count * 256, u32::MAX);
    dfa.transitions = Arc::from(transitions);
    dfa
}

#[test]
fn finite_token_probe_view_preserves_exact_witness_signatures() {
    let view = TokenizerView {
        flat_dfa: padded_sample_dfa(),
    };
    let tokens: Vec<&[u8]> = vec![b"a", b"b", b"aa", b"ba"];
    let initial_states = vec![0usize, 1, 2];
    let disallowed = BTreeMap::<u32, BitSet>::new();
    let full = build_dfa_with_group_filter(&view, &disallowed, None, None, None);
    let (probe_view, probe_initial) =
        finite_token_probe_view(&view, &initial_states, &tokens);
    let probe =
        build_dfa_with_group_filter(&probe_view, &disallowed, None, None, None);

    assert_eq!(
        token_signatures_for_states(&probe, &tokens, &probe_initial),
        token_signatures_for_states(&full, &tokens, &initial_states),
    );
}

#[test]
fn singleton_identity_probe_requires_pairwise_distinct_signatures() {
    let view = TokenizerView { flat_dfa: sample_dfa() };
    let disallowed = BTreeMap::<u32, BitSet>::new();
    let dfa = build_dfa_with_group_filter(&view, &disallowed, None, None, None);
    let states = vec![0usize, 1, 2];
    let distinct: Vec<&[u8]> = vec![b"a", b"b"];
    let collided: Vec<&[u8]> = vec![b"a", b"a"];

    assert!(token_signatures_are_pairwise_distinct(
        &dfa, &distinct, &states,
    ));
    assert!(!token_signatures_are_pairwise_distinct(
        &dfa, &collided, &states,
    ));
}

#[test]
fn trie_live_frontier_signatures_match_independent_token_scans() {
    let view = TokenizerView { flat_dfa: sample_dfa() };
    let disallowed = BTreeMap::<u32, BitSet>::new();
    let dfa = build_dfa_with_group_filter(&view, &disallowed, None, None, None);
    let states = vec![0usize, 1, 2];
    let tokens = vec![
        b"".to_vec(),
        b"a".to_vec(),
        b"aa".to_vec(),
        b"ab".to_vec(),
        b"aba".to_vec(),
        b"abb".to_vec(),
        b"b".to_vec(),
        b"ba".to_vec(),
        b"bb".to_vec(),
        b"c".to_vec(),
        b"ca".to_vec(),
    ];
    let expected = token_signatures_for_states(&dfa, &tokens, &states);
    let mut sorted_indices = (0..tokens.len()).collect::<Vec<_>>();
    sorted_indices.sort_unstable_by(|&left, &right| tokens[left].cmp(&tokens[right]));
    let state_group_size = vocab_state_group_size(states.len(), dfa.num_groups);
    let mut scratch = Scratch::new(states.len(), dfa.num_groups);
    let mut trie = TrieWalkState::new();
    let (pairs, _) = trie_walk_chunk_signatures(
        &dfa,
        &tokens,
        &sorted_indices,
        &states,
        state_group_size,
        &mut scratch,
        &mut trie,
        false,
    );
    let mut actual = vec![0u64; tokens.len()];
    for (token_idx, signature) in pairs {
        actual[token_idx] = signature;
    }

    assert_eq!(actual, expected);
}

#[test]
fn trie_noop_self_loop_skip_matches_independent_token_scans() {
    let mut flat_dfa = sample_dfa();
    let mut transitions = flat_dfa.transitions.to_vec();
    transitions[b'a' as usize] = 0;
    flat_dfa.transitions = Arc::from(transitions);
    let view = TokenizerView { flat_dfa };
    let disallowed = BTreeMap::<u32, BitSet>::new();
    let dfa = build_dfa_with_group_filter(&view, &disallowed, None, None, None);
    assert!(dfa.finalizers[0].is_empty());
    assert!(dfa.self_loop_bytes[0].contains(b'a'));

    let states = vec![0usize, 1, 2];
    let tokens = vec![
        b"a".to_vec(),
        b"aa".to_vec(),
        b"aaa".to_vec(),
        b"aab".to_vec(),
        b"ab".to_vec(),
        b"b".to_vec(),
    ];
    let expected = token_signatures_for_states(&dfa, &tokens, &states);
    let sorted_indices = token_indices_in_lexical_order(&tokens);
    let state_group_size = vocab_state_group_size(states.len(), dfa.num_groups);
    let mut scratch = Scratch::new(states.len(), dfa.num_groups);
    let mut trie = TrieWalkState::new();
    let (pairs, _) = trie_walk_chunk_signatures(
        &dfa,
        &tokens,
        &sorted_indices,
        &states,
        state_group_size,
        &mut scratch,
        &mut trie,
        false,
    );
    let mut actual = vec![0u64; tokens.len()];
    for (token_idx, signature) in pairs {
        actual[token_idx] = signature;
    }
    assert_eq!(actual, expected);
}

#[test]
fn sparse_live_clean_signature_matches_dense_fold() {
    let view = TokenizerView { flat_dfa: sample_dfa() };
    let disallowed = BTreeMap::<u32, BitSet>::new();
    let dfa = build_dfa_with_group_filter(&view, &disallowed, None, None, None);
    let state_count = 6usize;
    let mut scratch = Scratch::new(state_count, dfa.num_groups);
    ensure_completion_weights(&mut scratch, state_count);
    let all_none_signature = all_none_completion_signature(&dfa, state_count);

    for live_mask in 0usize..(1usize << state_count) {
        let mut live_indices = Vec::new();
        for index in 0..state_count {
            if live_mask & (1usize << index) == 0 {
                scratch.current_states[index] = STATE_NONE;
            } else {
                scratch.current_states[index] = (index + live_mask) % dfa.num_states;
                live_indices.push(index);
            }
        }
        assert_eq!(
            finish_token_signature_sparse_live_clean(
                &dfa,
                &live_indices,
                &scratch,
                all_none_signature,
            ),
            finish_token_signature_clean(&dfa, state_count, &scratch),
            "live_mask={live_mask:#b}",
        );
    }
}

#[test]
fn sparse_dirty_signature_matches_dense_fold_with_live_and_dead_dirty_states() {
    let view = TokenizerView { flat_dfa: sample_dfa() };
    let disallowed = BTreeMap::<u32, BitSet>::new();
    let dfa = build_dfa_with_group_filter(&view, &disallowed, None, None, None);
    let state_count = 6usize;
    let mut scratch = Scratch::new(state_count, dfa.num_groups);
    ensure_completion_weights(&mut scratch, state_count);
    let all_none_signature = all_none_completion_signature(&dfa, state_count);

    for (states, live_indices) in [
        // Sparse side of the crossover.
        (
            [0, STATE_NONE, 2 % dfa.num_states, STATE_NONE, STATE_NONE, 0],
            vec![0usize, 2, 5],
        ),
        // Dense side of the crossover.
        (
            [0, STATE_NONE, 2 % dfa.num_states, STATE_NONE, 1 % dfa.num_states, 0],
            vec![0usize, 2, 4, 5],
        ),
    ] {
        scratch.current_states[..state_count].copy_from_slice(&states);
        scratch.dirty_group_masks.fill(0);
        scratch.dirty_state_flags.fill(0);
        scratch.dirty_state_bits.fill(0);
        scratch.match_positions.fill(NONE);

        // Exercise both a live dirty state and a dirty state whose DFA path
        // has already died. Missing DAG nodes intentionally hash as zero in
        // both implementations, isolating only the polynomial baseline.
        for &(state_index, gid, position) in
            &[(1usize, 0usize, 1u32), (2usize, 0usize, 2u32)]
        {
            scratch.dirty_state_flags[state_index] = 1;
            set_dirty_state_bit(&mut scratch.dirty_state_bits, state_index);
            let flat_dirty = state_index * scratch.dirty_words + gid / 64;
            scratch.dirty_group_masks[flat_dirty] |= 1u64 << (gid % 64);
            scratch.match_positions[state_index * dfa.num_groups + gid] = position;
        }

        assert_eq!(
            finish_token_signature_sparse_dirty(
                &dfa,
                state_count,
                &live_indices,
                &scratch,
                all_none_signature,
            ),
            finish_token_signature_no_cleanup(&dfa, state_count, &scratch),
        );
    }
}

#[test]
fn default_vocab_batch_size_is_state_and_memory_bounded() {
    assert_eq!(default_vocab_batch_size(0, 10, 10_000), 0);
    assert_eq!(default_vocab_batch_size(123, 0, 10_000), 123);
    assert_eq!(default_vocab_batch_size(10_000, 0, 10_000), 4_000);
    assert_eq!(default_vocab_batch_size(10_000, 1, 1_000), 4_000);
    assert_eq!(default_vocab_batch_size(10_000, 262, 1_000), 4_000);
    assert_eq!(default_vocab_batch_size(10_000, 263, 1_000), 3_986);
    assert_eq!(default_vocab_batch_size(10_000, usize::MAX, 1_000), 1);
    assert_eq!(default_vocab_batch_size(1_999, 100, 20_000), 1_999);
    assert_eq!(default_vocab_batch_size(5_241, 259, 15_155), 759);
}

#[test]
fn moderate_first_transition_factor_extension_requires_single_batch_authority() {
    assert!(first_transition_factor_moderate_extension_enabled(
        2_879, 1_662, 1,
    ));
    assert!(!first_transition_factor_moderate_extension_enabled(
        1_023, 1_662, 1,
    ));
    assert!(!first_transition_factor_moderate_extension_enabled(
        8_193, 1_662, 1,
    ));
    assert!(!first_transition_factor_moderate_extension_enabled(
        2_879, 4_001, 1,
    ));
}

#[test]
fn precomputed_lexical_order_matches_sorting_each_active_subset() {
    let tokens = vec![
        b"ba".to_vec(),
        b"a".to_vec(),
        b"".to_vec(),
        b"aa".to_vec(),
        b"a".to_vec(),
        b"b".to_vec(),
    ];
    let lexical_order = token_indices_in_lexical_order(&tokens);
    for mask in 0usize..(1usize << tokens.len()) {
        let active_tokens = (0..tokens.len())
            .map(|token_idx| mask & (1usize << token_idx) != 0)
            .collect::<Vec<_>>();
        let actual = active_indices_in_lexical_order(&lexical_order, &active_tokens);
        let mut expected = (0..tokens.len())
            .filter(|&token_idx| active_tokens[token_idx])
            .collect::<Vec<_>>();
        expected.sort_unstable_by(|&left, &right| {
            tokens[left]
                .cmp(&tokens[right])
                .then_with(|| left.cmp(&right))
        });
        assert_eq!(actual, expected, "mask={mask:#b}");
    }
}

#[test]
fn tiny_vocab_input_compaction_matches_uncompacted_analysis() {
    let view = TokenizerView {
        flat_dfa: padded_sample_dfa(),
    };
    let tokens: Vec<&[u8]> = vec![b"a", b"b", b"aa", b"ba"];
    let initial_states = vec![0usize, 1, 2];
    let disallowed = BTreeMap::<u32, BitSet>::new();

    let compacted = find_vocab_equivalence_classes_with_group_filter(
        &view,
        &tokens,
        &initial_states,
        &disallowed,
        None,
        None,
        None,
        None,
    );
    let uncached_base = SharedVocabDfaCache::default();
    let uncompacted = find_vocab_equivalence_classes_with_group_filter(
        &view,
        &tokens,
        &initial_states,
        &disallowed,
        None,
        None,
        Some(&uncached_base),
        None,
    );

    assert_eq!(compacted, uncompacted);
}

#[test]
fn invalid_sentinel_initial_state_is_a_structured_invariant_error() {
    let view = TokenizerView {
        flat_dfa: sample_dfa(),
    };
    let tokens: Vec<&[u8]> = vec![b"a", b"b", b"aa"];
    let disallowed = BTreeMap::<u32, BitSet>::new();
    let error = crate::error::catch_internal_invariant(|| {
        find_vocab_equivalence_classes_with_group_filter(
            &view,
            &tokens,
            &[u32::MAX as usize],
            &disallowed,
            None,
            None,
            None,
            None,
        )
    })
    .expect_err("an unmapped raw-start sentinel must fail compilation");

    assert!(matches!(error, crate::error::Error::InternalInvariant(_)));
    assert!(error.to_string().contains("sentinel=true"));
}

#[test]
fn dead_transition_sentinel_remains_a_valid_dfa_edge_encoding() {
    let view = TokenizerView {
        flat_dfa: sample_dfa(),
    };
    assert!(view.dfa().transitions.iter().any(|&target| target == u32::MAX));

    let tokens: Vec<&[u8]> = vec![b"a", b"b", b"aa"];
    let disallowed = BTreeMap::<u32, BitSet>::new();
    let result = crate::error::catch_internal_invariant(|| {
        find_vocab_equivalence_classes_with_group_filter(
            &view,
            &tokens,
            &[0],
            &disallowed,
            None,
            None,
            None,
            None,
        )
    });

    assert!(result.is_ok(), "dead transitions are not invalid state coordinates");
}

#[test]
fn shared_analysis_dfa_cache_matches_uncached_vocab_equivalence() {
    let view = TokenizerView { flat_dfa: sample_dfa() };
    let tokens: Vec<&[u8]> = vec![b"a", b"b", b"aa", b"ba"];
    let initial_states = vec![0usize, 1, 2];
    let disallowed = BTreeMap::<u32, BitSet>::new();

    let uncached = find_vocab_equivalence_classes_with_group_filter(
        &view,
        &tokens,
        &initial_states,
        &disallowed,
        None,
        None,
        None,
        None,
    );
    let cache = SharedVocabAnalysisDfaCache::default();
    let cached_first = find_vocab_equivalence_classes_with_group_filter(
        &view,
        &tokens,
        &initial_states,
        &disallowed,
        None,
        None,
        None,
        Some(&cache),
    );
    let cached_hit = find_vocab_equivalence_classes_with_group_filter(
        &view,
        &tokens,
        &initial_states,
        &disallowed,
        None,
        None,
        None,
        Some(&cache),
    );

    assert_eq!(cached_first, uncached);
    assert_eq!(cached_hit, uncached);
    assert_eq!(cache.entries.lock().unwrap().len(), 1);
}

#[test]
fn shared_analysis_dfa_cache_keys_filtered_views_and_normalized_follows() {
    let view = TokenizerView { flat_dfa: sample_dfa() };
    let tokens: Vec<&[u8]> = vec![b"a", b"b", b"aa", b"ba"];
    let initial_states = vec![0usize, 1, 2];
    let cache = SharedVocabAnalysisDfaCache::default();

    let active_zero = [true, false];
    let active_all = [true, true];
    let absent_follows = BTreeMap::<u32, BitSet>::new();
    let explicit_empty_follows = BTreeMap::from([(0u32, BitSet::new(2))]);
    let mut blocked_row = BitSet::new(2);
    blocked_row.set(1);
    let blocked_follows = BTreeMap::from([(0u32, blocked_row)]);

    let cached_zero = find_vocab_equivalence_classes_with_group_filter(
        &view,
        &tokens,
        &initial_states,
        &absent_follows,
        None,
        Some(&active_zero),
        None,
        Some(&cache),
    );
    let cached_zero_explicit_empty = find_vocab_equivalence_classes_with_group_filter(
        &view,
        &tokens,
        &initial_states,
        &explicit_empty_follows,
        None,
        Some(&active_zero),
        None,
        Some(&cache),
    );
    assert_eq!(cached_zero_explicit_empty, cached_zero);
    assert_eq!(
        cache.entries.lock().unwrap().len(),
        1,
        "absent and explicit empty follow rows must canonicalize to one key",
    );

    let uncached_all = find_vocab_equivalence_classes_with_group_filter(
        &view,
        &tokens,
        &initial_states,
        &absent_follows,
        None,
        Some(&active_all),
        None,
        None,
    );
    let cached_all = find_vocab_equivalence_classes_with_group_filter(
        &view,
        &tokens,
        &initial_states,
        &absent_follows,
        None,
        Some(&active_all),
        None,
        Some(&cache),
    );
    assert_eq!(cached_all, uncached_all);
    assert_eq!(cache.entries.lock().unwrap().len(), 2);

    let uncached_blocked = find_vocab_equivalence_classes_with_group_filter(
        &view,
        &tokens,
        &initial_states,
        &blocked_follows,
        None,
        Some(&active_all),
        None,
        None,
    );
    let cached_blocked = find_vocab_equivalence_classes_with_group_filter(
        &view,
        &tokens,
        &initial_states,
        &blocked_follows,
        None,
        Some(&active_all),
        None,
        Some(&cache),
    );
    assert_eq!(cached_blocked, uncached_blocked);
    assert_eq!(cache.entries.lock().unwrap().len(), 3);
}

#[test]
fn trie_target_aggregate_matches_dirty_mask_scan() {
    let mut scratch = Scratch::new(2, 3);
    scratch.trie_target_bits_enabled = true;
    reset_trie_target_aggregate(&mut scratch, 4);

    let set_match = |scratch: &mut Scratch, state: usize, gid: usize, position: u32| {
        let index = state * scratch.num_groups + gid;
        let previous = scratch.match_positions[index];
        if previous == NONE {
            mark_dirty_group(scratch, state, gid);
        }
        update_trie_target_aggregate(scratch, gid, previous, position);
        scratch.match_positions[index] = position;
    };

    set_match(&mut scratch, 0, 0, 1);
    set_match(&mut scratch, 1, 0, 2);
    set_match(&mut scratch, 0, 1, 3);
    set_match(&mut scratch, 1, 2, 4);
    // Replacing a state-local latest match must remove only its old pair.
    set_match(&mut scratch, 0, 0, 4);

    collect_targets(&mut scratch, 3, 4, 0, 2);
    let mut expected_targets = scratch.targets.clone();
    expected_targets.sort_unstable();
    let expected_target_gids = scratch.target_gids.clone();
    let expected_single_target_pos = scratch.single_target_pos;
    let expected_single_target_gids = scratch.single_target_gids.clone();

    collect_trie_targets(&mut scratch, 4, false);
    assert_eq!(scratch.targets, expected_targets);
    assert_eq!(scratch.target_gids, expected_target_gids);
    if expected_targets.len() == 1 {
        assert_eq!(scratch.single_target_pos, expected_single_target_pos);
        assert_eq!(scratch.single_target_gids, expected_single_target_gids);
    } else {
        // The legacy scan leaves these stale after materializing the map;
        // multi-target consumers use only `targets` and `target_gids`.
        assert_eq!(scratch.single_target_pos, usize::MAX);
        assert!(scratch.single_target_gids.is_empty());
    }
}

#[test]
fn dense_trie_suffix_dag_matches_generic_and_precomputed_root() {
    let view = TokenizerView { flat_dfa: sample_dfa() };
    let disallowed = BTreeMap::<u32, BitSet>::new();
    let dfa = build_dfa_with_group_filter(&view, &disallowed, None, None, None);
    let token = b"abba";

    let mut generic = Scratch::new(1, dfa.num_groups);
    generic.targets.extend([1usize, 2]);
    generic.target_gids.insert(1, SmallVec::from_slice(&[0]));
    generic.target_gids.insert(2, SmallVec::from_slice(&[1]));
    hash_suffixes(&dfa, token, &mut generic, false);

    let mut dense = Scratch::new(1, dfa.num_groups);
    dense.trie_target_bits_enabled = true;
    reset_trie_target_aggregate(&mut dense, token.len());
    update_trie_target_aggregate(&mut dense, 0, NONE, 1);
    update_trie_target_aggregate(&mut dense, 1, NONE, 2);
    dense.targets.extend([1usize, 2]);
    hash_trie_suffixes_dense(&dfa, token, &mut dense, None);

    let mut dense_reuse = Scratch::new(1, dfa.num_groups);
    dense_reuse.trie_target_bits_enabled = true;
    reset_trie_target_aggregate(&mut dense_reuse, token.len());
    update_trie_target_aggregate(&mut dense_reuse, 0, NONE, 1);
    update_trie_target_aggregate(&mut dense_reuse, 1, NONE, 2);
    dense_reuse.targets.extend([1usize, 2]);
    hash_trie_suffixes_dense_impl(&dfa, token, &mut dense_reuse, None, true);

    for pos in 1..token.len() {
        let generic_hash = generic
            .dag_nodes
            .get(pos)
            .and_then(|node| node.as_ref())
            .map(|node| node.hash);
        let dense_hash = dense
            .dag_nodes
            .get(pos)
            .and_then(|node| node.as_ref())
            .map(|node| node.hash);
        assert_eq!(dense_hash, generic_hash, "suffix position {pos}");
        let dense_reuse_hash = dense_reuse
            .dag_nodes
            .get(pos)
            .and_then(|node| node.as_ref())
            .map(|node| node.hash);
        assert_eq!(
            dense_reuse_hash, generic_hash,
            "reused-slot suffix position {pos}"
        );
    }

    let root_pos = 1usize;
    let root_disallowed = dfa.disallowed_for(0).clone();
    let (root_end_state, root_edges) = run_suffix(
        &dfa,
        &token[root_pos..],
        root_pos,
        &mut dense.suffix_match_positions,
        &mut dense.suffix_dirty_groups,
    );
    assert!(
        root_edges.iter().any(|&(_, target)| target < token.len()),
        "test must exercise the single-target fallback"
    );

    let mut reused = Scratch::new(1, dfa.num_groups);
    reused.trie_target_bits_enabled = true;
    reset_trie_target_aggregate(&mut reused, token.len());
    update_trie_target_aggregate(&mut reused, 0, NONE, root_pos as u32);
    reused.targets.push(root_pos);
    hash_trie_suffixes_dense(
        &dfa,
        token,
        &mut reused,
        Some(PrecomputedDenseSuffixRoot {
            pos: root_pos,
            end_state: root_end_state.unwrap_or(STATE_NONE),
            edges: root_edges,
            disallowed: root_disallowed,
        }),
    );

    let mut reused_slots = Scratch::new(1, dfa.num_groups);
    reused_slots.trie_target_bits_enabled = true;
    reset_trie_target_aggregate(&mut reused_slots, token.len());
    update_trie_target_aggregate(&mut reused_slots, 0, NONE, root_pos as u32);
    reused_slots.targets.push(root_pos);
    let (root_end_state2, root_edges2) = run_suffix(
        &dfa,
        &token[root_pos..],
        root_pos,
        &mut reused_slots.suffix_match_positions,
        &mut reused_slots.suffix_dirty_groups,
    );
    hash_trie_suffixes_dense_impl(
        &dfa,
        token,
        &mut reused_slots,
        Some(PrecomputedDenseSuffixRoot {
            pos: root_pos,
            end_state: root_end_state2.unwrap_or(STATE_NONE),
            edges: root_edges2,
            disallowed: dfa.disallowed_for(0).clone(),
        }),
        true,
    );

    let mut fresh = Scratch::new(1, dfa.num_groups);
    fresh.trie_target_bits_enabled = true;
    reset_trie_target_aggregate(&mut fresh, token.len());
    update_trie_target_aggregate(&mut fresh, 0, NONE, root_pos as u32);
    fresh.targets.push(root_pos);
    hash_trie_suffixes_dense(&dfa, token, &mut fresh, None);

    for pos in 1..token.len() {
        let reused_hash = reused
            .dag_nodes
            .get(pos)
            .and_then(|node| node.as_ref())
            .map(|node| node.hash);
        let fresh_hash = fresh
            .dag_nodes
            .get(pos)
            .and_then(|node| node.as_ref())
            .map(|node| node.hash);
        assert_eq!(reused_hash, fresh_hash, "reused suffix position {pos}");
        let reused_slots_hash = reused_slots
            .dag_nodes
            .get(pos)
            .and_then(|node| node.as_ref())
            .map(|node| node.hash);
        assert_eq!(
            reused_slots_hash, fresh_hash,
            "reused-slot precomputed suffix position {pos}"
        );
    }
}

#[test]
fn reusable_single_target_disallowed_matches_fresh_root_and_fallback() {
    let view = TokenizerView { flat_dfa: sample_dfa() };
    let disallowed = BTreeMap::<u32, BitSet>::new();
    let dfa = build_dfa_with_group_filter(&view, &disallowed, None, None, None);

    // A one-byte suffix reaches a terminal only at the token end, so this
    // exercises the direct single-target hash without entering the DAG.
    let direct_token = b"ab";
    let mut direct_fresh = Scratch::new(1, dfa.num_groups);
    direct_fresh.single_target_pos = 1;
    direct_fresh.single_target_gids.push(0);
    let fresh_nodes =
        hash_single_target_suffix_dense_impl(&dfa, direct_token, &mut direct_fresh, false);

    let mut direct_reused = Scratch::new(1, dfa.num_groups);
    direct_reused.single_target_pos = 1;
    direct_reused.single_target_gids.push(0);
    let reused_nodes =
        hash_single_target_suffix_dense_impl(&dfa, direct_token, &mut direct_reused, true);
    assert_eq!(reused_nodes, fresh_nodes);
    assert_eq!(direct_reused.single_target_hash_pos, direct_fresh.single_target_hash_pos);
    assert_eq!(direct_reused.single_target_hash, direct_fresh.single_target_hash);

    // This longer suffix discovers an interior terminal and therefore
    // exercises the precomputed-root fallback into the dense suffix DAG.
    let fallback_token = b"abba";
    let mut fallback_fresh = Scratch::new(1, dfa.num_groups);
    fallback_fresh.single_target_pos = 1;
    fallback_fresh.single_target_gids.push(0);
    let fresh_nodes = hash_single_target_suffix_dense_impl(
        &dfa,
        fallback_token,
        &mut fallback_fresh,
        false,
    );

    let mut fallback_reused = Scratch::new(1, dfa.num_groups);
    fallback_reused.single_target_pos = 1;
    fallback_reused.single_target_gids.push(0);
    let reused_nodes = hash_single_target_suffix_dense_impl(
        &dfa,
        fallback_token,
        &mut fallback_reused,
        true,
    );
    assert_eq!(reused_nodes, fresh_nodes);
    for pos in 1..fallback_token.len() {
        let fresh_hash = fallback_fresh
            .dag_nodes
            .get(pos)
            .and_then(|node| node.as_ref())
            .map(|node| node.hash);
        let reused_hash = fallback_reused
            .dag_nodes
            .get(pos)
            .and_then(|node| node.as_ref())
            .map(|node| node.hash);
        assert_eq!(reused_hash, fresh_hash, "fallback suffix position {pos}");
    }
}

#[test]
fn shared_base_row_major_layout_matches_flat_dfa() {
    let dfa = sample_dfa();
    let base = SharedVocabDfaBase::build_from_dfa(&dfa);
    let byte_to_class = base.byte_to_class_ref();
    let row_major = base.transitions_by_state_class();

    for state in 0..dfa.states.len() {
        for byte in 0..=255usize {
            let class = byte_to_class[byte] as usize;
            assert_eq!(
                row_major[state * base.num_classes + class],
                dfa.trans(state, byte),
                "state={state}, byte={byte}"
            );
        }
    }
    assert!(base.is_compatible_with_dfa(&dfa));

    let independently_allocated = FlatDfa {
        states: dfa.states.clone(),
        start_state: dfa.start_state,
        transitions: Arc::from(dfa.transitions.to_vec()),
    };
    assert!(base.is_compatible_with_dfa(&independently_allocated));
}

#[test]
fn relevant_shared_base_preserves_every_relevant_transition_column() {
    let dfa = sample_dfa();
    let full_classes = compute_byte_classes(&dfa);
    let mut relevant = [false; 256];
    relevant[b'a' as usize] = true;
    relevant[b'b' as usize] = true;
    relevant[b'!' as usize] = true;

    let base = SharedVocabDfaBase::build_from_dfa_relevant(&dfa, &relevant);
    let byte_to_class = base.byte_to_class_ref();
    let row_major = base.transitions_by_state_class();

    for state in 0..dfa.states.len() {
        for byte in 0..=255usize {
            if !relevant[byte] {
                continue;
            }
            let class = byte_to_class[byte] as usize;
            assert_eq!(
                row_major[state * base.num_classes + class],
                dfa.trans(state, byte),
                "state={state}, byte={byte}"
            );
        }
    }

    for left in 0..=255usize {
        if !relevant[left] {
            assert_eq!(byte_to_class[left], 0);
            continue;
        }
        for right in left + 1..=255usize {
            if !relevant[right] {
                continue;
            }
            assert_eq!(
                full_classes[left] == full_classes[right],
                byte_to_class[left] == byte_to_class[right],
                "relevant byte equivalence must match the full exact columns"
            );
            if byte_to_class[left] == byte_to_class[right] {
                for state in 0..dfa.states.len() {
                    assert_eq!(
                        dfa.trans(state, left),
                        dfa.trans(state, right),
                        "relevant bytes sharing a class must have identical columns"
                    );
                }
            }
        }
    }
}

#[test]
fn relevant_shared_base_all_bytes_matches_full_exact_base() {
    let dfa = sample_dfa();
    let relevant = [true; 256];
    let restricted = SharedVocabDfaBase::build_from_dfa_relevant(&dfa, &relevant);
    let full = SharedVocabDfaBase::build_from_dfa(&dfa);

    assert_eq!(restricted.byte_to_class_ref(), full.byte_to_class_ref());
    assert_eq!(restricted.num_classes, full.num_classes);
    assert_eq!(
        restricted.transitions_by_state_class(),
        full.transitions_by_state_class()
    );
    assert_eq!(restricted.self_loop_bytes, full.self_loop_bytes);
}

#[test]
fn state_signature_blocks_recombine_to_monolithic_polynomial() {
    let observations = [3u64, 11, u64::MAX - 7, 19, 23, 29, 31];
    let fold = |values: &[u64]| {
        values.iter().fold(HASH_SEED3, |signature, &value| {
            signature.wrapping_mul(HASH_SEED1).wrapping_add(value)
        })
    };

    let mut combined = HASH_SEED3;
    for block in [&observations[..2], &observations[2..5], &observations[5..]] {
        combined = append_state_signature_block(
            combined,
            fold(block),
            state_signature_block_factor(block.len()),
        );
    }
    assert_eq!(combined, fold(&observations));
}
