//! Literal small-bitmask oracle, independent of the weighted algebra and DWA
//! normalizer. Every generated input graph is finite and acyclic.
use super::*;
use range_set_blaze::RangeSetBlaze;

const ALL: u32 = (1 << 24) - 1;
fn weight(bits: u32) -> Weight {
    Weight::from_per_tsid_token_sets((0..3u32).filter_map(|row| {
        let tokens = (0..8u32)
            .filter(|&t| bits & (1 << (row * 8 + t)) != 0)
            .collect::<RangeSetBlaze<u32>>();
        (!tokens.is_empty()).then_some((row, tokens))
    }))
}
fn bits(w: &Weight) -> u32 {
    let mut result = 0;
    for row in 0..3u32 {
        for token in 0..8u32 {
            if !w.intersection(&weight(1 << (row * 8 + token))).is_empty() {
                result |= 1 << (row * 8 + token);
            }
        }
    }
    result
}
#[derive(Clone, Default)]
struct State {
    final_bits: u32,
    edges: BTreeMap<i32, Vec<(usize, u32)>>,
    eps: Vec<(usize, u32)>,
}
fn build(states: &[State]) -> NWA {
    let mut nwa = NWA::new(3, 7);
    for _ in states {
        nwa.add_state();
    }
    nwa.set_start_states(vec![0]);
    for (id, state) in states.iter().enumerate() {
        if state.final_bits != 0 {
            nwa.set_final_weight(id as u32, weight(state.final_bits));
        }
        for (&label, targets) in &state.edges {
            for &(target, w) in targets {
                nwa.add_transition(id as u32, label, target as u32, weight(w));
            }
        }
        for &(target, w) in &state.eps {
            nwa.add_epsilon(id as u32, target as u32, weight(w));
        }
    }
    nwa
}
fn epsilon(states: &[State], active: &mut [u32]) {
    for (id, state) in states.iter().enumerate() {
        for &(target, w) in &state.eps {
            active[target] |= active[id] & w;
        }
    }
}
fn literal(states: &[State], word: &[i32]) -> u32 {
    let mut active = vec![0; states.len()];
    active[0] = ALL;
    epsilon(states, &mut active);
    let mut accepted = states
        .iter()
        .zip(&active)
        .fold(0, |a, (s, w)| a | s.final_bits & w);
    for label in word {
        let mut next = vec![0; states.len()];
        for (id, s) in states.iter().enumerate() {
            for &(target, w) in s.edges.get(label).into_iter().flatten() {
                next[target] |= active[id] & w;
            }
        }
        epsilon(states, &mut next);
        accepted |= states
            .iter()
            .zip(&next)
            .fold(0, |a, (s, w)| a | s.final_bits & w);
        active = next;
    }
    accepted
}
fn compiled(dwa: &DWA, word: &[i32]) -> u32 {
    let mut state = dwa.start_state();
    let mut accumulated = ALL;
    let mut accepted = dwa.states()[state as usize]
        .final_weight
        .as_ref()
        .map_or(0, bits);
    for label in word {
        let Some((target, w)) = dwa.states()[state as usize]
            .transitions
            .get(label)
            .or_else(|| dwa.states()[state as usize].transitions.get(&DEFAULT_LABEL))
        else {
            break;
        };
        accumulated &= bits(w);
        state = *target;
        accepted |= accumulated
            & dwa.states()[state as usize]
                .final_weight
                .as_ref()
                .map_or(0, bits);
    }
    accepted
}

#[test]
fn consuming_full_alphabet_does_not_accept_an_empty_stack() {
    let mut states = vec![State::default(); 2];
    states[1].final_bits = ALL;
    for label in 0..3 {
        states[0].edges.insert(label, vec![(1, ALL)]);
    }
    let nwa = build(&states);
    let exact = normalize_weighted_stack_predicate_for_symbol_count(3, &nwa);
    assert_eq!(compiled(&exact, &[]), 0);
    for label in 0..3 {
        assert_eq!(compiled(&exact, &[label]), ALL);
    }
    // The old entry point deliberately retains its existing LR policy.
    let old = normalize_weighted_parser_stack_nwa_for_parser_state_count(3, &nwa);
    assert_eq!(compiled(&old, &[]), ALL);
}

#[test]
fn incomplete_alphabet_cannot_be_widened_by_a_default_row() {
    let mut states = vec![State::default(); 2];
    states[1].final_bits = ALL;
    for label in 0..2 {
        states[0].edges.insert(label, vec![(1, ALL)]);
    }
    let exact = normalize_weighted_stack_predicate_for_symbol_count(3, &build(&states));
    for word in [vec![], vec![0], vec![1], vec![2], vec![2, 1]] {
        assert_eq!(
            compiled(&exact, &word),
            literal(&states, &word),
            "word={word:?}"
        );
    }
}

#[test]
fn general_stack_normalization_matches_literal_weighted_prefix_oracle() {
    fn next(s: &mut u64) -> u64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *s
    }
    let mut seed = 608894;
    for case in 0..64 {
        let mut states = vec![State::default(); 8];
        for i in 0..8 {
            states[i].final_bits = next(&mut seed) as u32 & ALL;
            for target in i + 1..8 {
                for label in 0..3 {
                    if next(&mut seed) % 3 == 0 {
                        states[i]
                            .edges
                            .entry(label)
                            .or_default()
                            .push((target, next(&mut seed) as u32 & ALL));
                    }
                }
                if next(&mut seed) % 4 == 0 {
                    states[i].eps.push((target, next(&mut seed) as u32 & ALL));
                }
            }
        }
        let exact = normalize_weighted_stack_predicate_for_symbol_count(3, &build(&states));
        let mut words = vec![Vec::new()];
        while let Some(word) = words.pop() {
            assert_eq!(
                compiled(&exact, &word),
                literal(&states, &word),
                "case={case} word={word:?}"
            );
            if word.len() < 4 {
                for label in 0..3 {
                    let mut w = word.clone();
                    w.push(label);
                    words.push(w);
                }
            }
        }
    }
}
