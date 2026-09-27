use super::*;
use crate::automata::lexer::ast::Expr;
use crate::automata::lexer::tokenizer::arbitrary_epsilon_l1_test_tokenizer;
use crate::compiler::pipeline::build_tokenizer_from_exprs_partitioned_with_adaptive;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::BTreeSet;

#[test]
fn possible_match_configs_report_table_completeness() {
    assert!(ConstraintPossibleMatchesConfig::EAGER.is_complete());
    assert!(!ConstraintPossibleMatchesConfig::DEFER_TO_DYNAMIC_MASK.is_complete());
}

fn directly_matched_terminals(
    tokenizer: &Tokenizer,
    start_state: u32,
    bytes: &[u8],
) -> BTreeSet<u32> {
    let mut states = tokenizer.execute_from_state_end_only(&[], start_state);
    let mut terminals = BTreeSet::new();
    if bytes.is_empty() {
        for &state in &states {
            terminals.extend(tokenizer.matched_terminals_iter(state));
        }
    }
    for &byte in bytes {
        states = tokenizer.step_all(&states, byte);
        for &state in &states {
            terminals.extend(tokenizer.matched_terminals_iter(state));
        }
        if states.is_empty() {
            break;
        }
    }
    terminals
}

fn assert_demanded_pm_matches_direct(
    tokenizer: &Tokenizer,
    entries: &[(u32, Vec<u8>)],
    context: &str,
) {
    let vocab = Vocab::new(entries.to_vec());
    let computation = compute_constraint_possible_matches_for_vocab(
        tokenizer,
        &vocab,
        ConstraintPossibleMatchesConfig::EAGER,
    );
    assert!(computation.complete);
    let mapped = &computation.mapped_possible_matches;
    let demand = delayed_terminal_demand(tokenizer);

    for terminal in 0..tokenizer.num_terminals() {
        if !demand.terminals.contains(terminal as usize) {
            assert!(
                !mapped.artifact().contains_key(&terminal),
                "non-demand terminal {terminal} must not force a PM row",
            );
        }
    }

    for state in 0..tokenizer.num_states() {
        let internal_state =
            mapped.id_map().tokenizer_states.original_to_internal[state as usize];
        if !demand.raw_query_state[state as usize] {
            assert_eq!(internal_state, u32::MAX, "non-query state={state}");
            continue;
        }
        assert_ne!(internal_state, u32::MAX, "query state={state}");
        for (token_id, bytes) in entries {
            let internal_token =
                mapped.id_map().vocab_tokens.original_to_internal[*token_id as usize];
            let expected = directly_matched_terminals(tokenizer, state, bytes);
            for terminal in demand.terminals.iter().map(|terminal| terminal as u32) {
                let actual = internal_token != u32::MAX
                    && mapped.artifact().get(&terminal).is_some_and(|weight| {
                        weight
                            .tokens_for_tsid(internal_state)
                            .contains(internal_token)
                    });
                assert_eq!(
                    actual,
                    expected.contains(&terminal),
                    "{context} state={state} token={token_id} bytes={bytes:?} terminal={terminal}",
                );
            }
        }
    }
}

#[test]
fn demanded_possible_matches_match_direct_state_set_execution() {
    let expressions = vec![
        Expr::U8Seq(b"a".to_vec()),
        Expr::U8Seq(b"ab".to_vec()),
        Expr::U8Seq(b"b".to_vec()),
        Expr::Repeat {
            expr: Box::new(Expr::U8Seq(b" ".to_vec())),
            min: 1,
            max: None,
        },
    ];
    let tokenizer = build_tokenizer_from_exprs_partitioned_with_adaptive(
        &expressions,
        None,
        &[0, 1, 2, 2],
        false,
    );
    assert!(tokenizer.has_deterministic_dispatch());
    assert!(pm_vocab_equiv_supported(&tokenizer));

    let entries = vec![
        (0, b"a".to_vec()),
        (1, b"ab".to_vec()),
        (2, b"b".to_vec()),
        (3, b" a".to_vec()),
        (4, b"a ".to_vec()),
        (5, b"x".to_vec()),
        (6, b"ab".to_vec()),
    ];
    let demand = delayed_terminal_demand(&tokenizer);
    assert_eq!(demand.terminals.iter().collect::<Vec<_>>(), vec![3]);
    assert_demanded_pm_matches_direct(&tokenizer, &entries, "structured dispatch");
}

#[test]
fn demanded_possible_matches_match_direct_execution_on_random_small_lexers() {
    let alphabet = [b'a', b'b', b' '];
    let mut entries = vec![(0u32, Vec::new())];
    fn add_words(
        entries: &mut Vec<(u32, Vec<u8>)>,
        alphabet: &[u8],
        prefix: &mut Vec<u8>,
        remaining: usize,
    ) {
        if remaining == 0 {
            return;
        }
        for &byte in alphabet {
            prefix.push(byte);
            entries.push((entries.len() as u32, prefix.clone()));
            add_words(entries, alphabet, prefix, remaining - 1);
            prefix.pop();
        }
    }
    add_words(&mut entries, &alphabet, &mut Vec::new(), 4);

    let mut rng = StdRng::seed_from_u64(0x504d_4445_4d41_4e44);
    for case in 0..32 {
        let terminal_count = rng.gen_range(2..=6);
        let mut expressions = Vec::with_capacity(terminal_count);
        let mut partitions = Vec::with_capacity(terminal_count);
        for _ in 0..terminal_count {
            let byte = alphabet[rng.gen_range(0..alphabet.len())];
            let other = alphabet[rng.gen_range(0..alphabet.len())];
            let expression = match rng.gen_range(0..5) {
                0 => Expr::U8Seq(vec![byte]),
                1 => Expr::Repeat {
                    expr: Box::new(Expr::U8Seq(vec![byte])),
                    min: 1,
                    max: None,
                },
                2 => Expr::Repeat {
                    expr: Box::new(Expr::U8Seq(vec![byte])),
                    min: 1,
                    max: Some(3),
                },
                3 => Expr::Seq(vec![
                    Expr::U8Seq(vec![byte]),
                    Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(vec![other])),
                        min: 0,
                        max: None,
                    },
                ]),
                _ => Expr::Choice(vec![
                    Expr::U8Seq(vec![byte]),
                    Expr::U8Seq(vec![byte, other]),
                ]),
            };
            expressions.push(expression);
            partitions.push(rng.gen_range(0..3));
        }
        let tokenizer = build_tokenizer_from_exprs_partitioned_with_adaptive(
            &expressions,
            None,
            &partitions,
            false,
        );
        assert_demanded_pm_matches_direct(
            &tokenizer,
            &entries,
            &format!("case={case} expressions={expressions:?} partitions={partitions:?}"),
        );
    }
}

#[test]
fn no_delayed_terminals_produces_complete_empty_possible_matches() {
    let tokenizer = build_tokenizer_from_exprs_partitioned_with_adaptive(
        &[Expr::U8Seq(b"a".to_vec()), Expr::U8Seq(b"b".to_vec())],
        None,
        &[0, 1],
        false,
    );
    let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);
    let demand = delayed_terminal_demand(&tokenizer);
    assert!(demand.terminals.is_zero());

    let computation = compute_constraint_possible_matches_for_vocab(
        &tokenizer,
        &vocab,
        ConstraintPossibleMatchesConfig::EAGER,
    );
    assert!(computation.complete);
    assert!(computation.mapped_possible_matches.artifact().is_empty());
    assert!(
        !computation.runtime_dynamic_vocab.vocab.is_initialized(),
        "complete empty PM must not eagerly build the dynamic-mask vocabulary",
    );
    assert!(computation
        .mapped_possible_matches
        .id_map()
        .tokenizer_states
        .original_to_internal
        .iter()
        .all(|&state| state == u32::MAX));
    assert!(computation
        .mapped_possible_matches
        .id_map()
        .vocab_tokens
        .original_to_internal
        .iter()
        .all(|&token| token == u32::MAX));
}

#[test]
fn batched_future_masks_follow_bytes_and_boundary_edges() {
    // 0 -a-> 1 -b-> 2, with 3 projecting to 1 at a token boundary.
    // Only state 2 directly matches demanded bit 0b10; every predecessor
    // that can reach it must inherit that bit, while disconnected state 4
    // must remain empty.
    let mut transitions = vec![vec![u32::MAX; 5]; 256];
    transitions[b'a' as usize][0] = 1;
    transitions[b'b' as usize][1] = 2;
    let boundary = [u32::MAX, u32::MAX, u32::MAX, 1, u32::MAX];
    let masks = batched_future_matched_masks(
        &[0, 0, 0b10, 0, 0],
        &transitions,
        Some(&boundary),
    );
    assert_eq!(masks, vec![0b10, 0b10, 0b10, 0b10, 0]);
}

#[test]
fn epsilon_nfa_possible_match_collector_defaults_by_state_scale() {
    assert!(nfa_powerset_collect_default(914, 1_707));
    assert!(nfa_powerset_collect_default(8_108, 1_000));
    assert!(nfa_powerset_collect_default(10_355, 1_000));
    assert!(nfa_powerset_collect_default(
        PM_NFA_POWERSET_DEFAULT_MAX_STATES,
        usize::MAX,
    ));
    assert!(!nfa_powerset_collect_default(18_943, 1_707));
    assert!(nfa_powerset_collect_default(26_965, 192));
    assert!(!nfa_powerset_collect_default(
        PM_NFA_POWERSET_NARROW_MAX_STATES + 1,
        192,
    ));
    assert!(!nfa_powerset_collect_default(26_965, 1_707));
}

#[test]
fn epsilon_powerset_interval_collector_matches_sparse_nfa_rows() {
    let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
    assert!(tokenizer.has_epsilon_transitions());
    assert!(!tokenizer.has_deterministic_dispatch());

    let vocab = Vocab::new(
        vec![
            (0, b"".to_vec()),
            (1, b"a".to_vec()),
            (2, b"aa".to_vec()),
            (3, b"ab".to_vec()),
            (4, b"b".to_vec()),
            (5, b"ba".to_vec()),
            (6, b"x".to_vec()),
        ]);
    let artifacts = get_ordered_vocab_trie_artifacts_for_vocab(&vocab).0;
    let raw_states = (0..tokenizer.num_states()).collect::<Vec<_>>();
    let sparse = collect_sparse_root_possible_matches(
        &tokenizer,
        &artifacts.trie.root,
        &raw_states,
        None,
    );

    let mut relevant_bytes = [false; 256];
    for bytes in &artifacts.ordered_vocab.ordered_token_bytes {
        for &byte in bytes {
            relevant_bytes[byte as usize] = true;
        }
    }
    let demand = DelayedTerminalDemand {
        terminals: BitSet::all(tokenizer.num_terminals() as usize),
        raw_state_relevant: vec![true; tokenizer.num_states() as usize],
        raw_query_state: vec![true; tokenizer.num_states() as usize],
        accepting_future_states: tokenizer.num_states() as usize,
    };
    let powerset =
        build_possible_match_powerset_view(&tokenizer, &relevant_bytes, None, &demand);
    let mut view_entries = powerset.raw_start_to_view.clone();
    view_entries.retain(|&state| state != u32::MAX);
    view_entries.sort_unstable();
    view_entries.dedup();
    let (powerset_rows, _) =
        collector::collect_possible_matches_interval_trie_class_build_precomputed(
            &artifacts.trie.root,
            &view_entries,
            Some(&powerset.boundary_state),
            powerset.num_states,
            tokenizer.num_terminals() as usize,
            &powerset.matched_terminals,
            &powerset.is_end,
            &powerset.byte_transitions,
            &powerset.self_loop_bytes,
        );
    let sparse_expanded = expand_interval_class_maps(&sparse.class_maps);
    let powerset_expanded = expand_interval_class_maps(&powerset_rows.class_maps);

    for raw_state in raw_states {
        let sparse_class = sparse.state_classes[raw_state as usize];
        assert_ne!(sparse_class, u32::MAX, "raw_state={raw_state}");
        let view_state = powerset.raw_start_to_view[raw_state as usize] as usize;
        let powerset_class = powerset_rows.state_classes[view_state];
        assert_ne!(powerset_class, u32::MAX, "raw_state={raw_state}");
        assert_eq!(
            sparse_expanded[sparse_class as usize].as_ref(),
            powerset_expanded[powerset_class as usize].as_ref(),
            "raw_state={raw_state} view_state={view_state}",
        );
    }
}

#[test]
fn epsilon_pm_vocab_equivalence_distinguishes_terminals_above_127() {
    let mut expressions = Vec::new();
    let mut partitions = Vec::new();
    for terminal in 0..130u32 {
        expressions.push(Expr::U8Seq(vec![terminal as u8]));
        partitions.push(terminal % 3);
    }
    let tokenizer = build_tokenizer_from_exprs_partitioned_with_adaptive(
        &expressions,
        None,
        &partitions,
        false,
    );
    assert!(tokenizer.has_epsilon_transitions());

    let vocab = Vocab::new(vec![(0, vec![128]), (1, vec![129])]);
    let full_artifacts = get_ordered_vocab_trie_artifacts_for_vocab(&vocab).0;
    let classes = compute_pm_vocab_equivalence_map(
        &tokenizer,
        full_artifacts.ordered_vocab.as_ref(),
        full_artifacts.trie.as_ref(),
    );

    assert_ne!(
        classes.original_to_internal[0],
        classes.original_to_internal[1],
        "PM vocab equivalence must include terminal IDs above the old u128 ceiling",
    );
}
