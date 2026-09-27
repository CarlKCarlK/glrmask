use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Arc, Mutex};

use super::{
    build_active_suffix_start_by_byte, build_reverse_transitions_by_byte,
    execute_prepared_dense_suffix_trie, token_has_active_terminal_suffix,
    vocab_adjacent_pair_index, vocab_suffix_trie,
};
use super::{

    classify_terminal_path_lengths, classify_vocab_char_type, classify_with_partition_set,
    compile_vocab_partition_set, custom_vocab_partition_rules,
    fast_eval_char_type_regular_partition, reference_char_type_regular_partition,

    exact_terminal_path_two_plus, exact_terminal_path_two_plus_candidate_dfa,
    exact_terminal_path_two_plus_finite_literals,
    parse_exact_l2p_boundary_filter_mode,
    suffix_has_allowed_l2p_follow_from_reset, ExactL2pBoundaryFilterMode,
    SharedClassifyBytesets,
    TokenL2pRouteHint, state_future_intersects_words,
    token_has_active_l2p_boundary_words, token_has_exact_active_l2p_boundary,
    token_l2p_route_hint, tokens_have_exact_active_l2p_boundary,
};

#[test]
fn regular_char_type_partition_matches_legacy_decision_tree() {
    let check = |bytes: &[u8]| {
        assert_eq!(
            fast_eval_char_type_regular_partition(bytes),
            reference_char_type_regular_partition(bytes),
            "bytes={bytes:?} utf8={:?}",
            std::str::from_utf8(bytes).ok(),
        );
    };

    check(b"");
    for byte in 0u8..=u8::MAX {
        check(&[byte]);
    }
    for first in 0u8..=u8::MAX {
        for second in 0u8..=u8::MAX {
            check(&[first, second]);
        }
    }

    for sample in [
        "hello", " hello", "123", " 123", "_field", "\"_field", " true",
        "[falsehood", "---", ":::::::::", "你好", " 你好", "١٢٣", "a你9",
        "🙂", "🙂🙂", "éclair", " éclair", "a-b", " -", " +",
    ] {
        check(sample.as_bytes());
    }

    // Deterministic mixed valid/invalid byte strings exercise longer
    // paths without introducing a test-only random dependency.
    let mut state = 0x9e37_79b9u32;
    for len in 0..=48usize {
        for _ in 0..256 {
            let mut bytes = Vec::with_capacity(len);
            for _ in 0..len {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                bytes.push(state as u8);
            }
            check(&bytes);
        }
    }
}

#[test]
fn regular_char_type_overflow_partition_matches_legacy_routing() {
    use crate::terminal_dwa::regular_partition::{
        char_type_final_partition, char_type_regular_partition, CharTypeRegularPartitionKey,
    };

    let thresholds = [
        (Some(16), Some(20), Some(8), Some(32)),
        (None, None, Some(8), None),
        (Some(3), Some(3), Some(3), Some(3)),
    ];
    let mut state = 0x243f_6a88u32;
    for (p0, p1, p2, p4) in thresholds {
        let key = CharTypeRegularPartitionKey {
            p0_overflow_threshold: p0,
            p1_overflow_threshold: p1,
            p2_overflow_threshold: p2,
            p4_overflow_threshold: p4,
            structural_boundary_enabled: super::structural_boundary_lexical_partition_enabled(),
        };
        let spec = char_type_regular_partition(key);
        let check = |bytes: &[u8]| {
            let mut expected = fast_eval_char_type_regular_partition(bytes) as usize;
            if expected == 0 && p0.is_some_and(|threshold| bytes.len() > threshold) {
                expected = 12;
            } else if expected == 1 && p1.is_some_and(|threshold| bytes.len() > threshold) {
                expected = 10;
            } else if expected == 2 && p2.is_some_and(|threshold| bytes.len() > threshold) {
                expected = 9;
            } else if expected == 4 && p4.is_some_and(|threshold| bytes.len() > threshold) {
                expected = 11;
            }
            assert_eq!(
                char_type_final_partition(spec.partition_index(bytes), bytes.len(), key),
                expected,
                "bytes={bytes:?}",
            );
        };
        for sample in [
            b"".as_slice(), b"+", b" -", b"hello", b" hello", b"123456789",
            b":::::::::", b"abcdefghijklmnopq", "你好世界你好世界".as_bytes(),
        ] {
            check(sample);
        }
        for len in 0..=64usize {
            for _ in 0..128 {
                let mut bytes = Vec::with_capacity(len);
                for _ in 0..len {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    bytes.push(state as u8);
                }
                check(&bytes);
            }
        }
    }
}
use crate::automata::lexer::ast::Expr;
use crate::automata::lexer::compile::{
    build_regex,
    build_regex_partitioned_with_adaptive,
};
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::lexer::Lexer;
use crate::compiler::stages::id_map_and_terminal_dwa::l1::build_flat_transition_table;
use crate::compiler::stages::id_map_and_terminal_dwa::types::TerminalPathLength;
use crate::ds::bitset::BitSet;
use crate::ds::u8set::U8Set;
use crate::Vocab;

fn reference_bytesets(tokenizer: &Tokenizer, num_terminals: u32) -> SharedClassifyBytesets {
    let nt = num_terminals as usize;
    let mut reachable_bytes = vec![U8Set::empty(); nt];
    let mut last_bytes = vec![U8Set::empty(); nt];

    for state in 0..tokenizer.num_states() {
        for (byte, target) in tokenizer.transitions_from(state) {
            for terminal in tokenizer.matched_terminal_bitset(target).iter() {
                if terminal < nt {
                    reachable_bytes[terminal].insert(byte);
                    last_bytes[terminal].insert(byte);
                }
            }
            for terminal in tokenizer.possible_future_terminals(target).iter() {
                if terminal < nt {
                    reachable_bytes[terminal].insert(byte);
                }
            }
        }
    }

    let mut first_bytes = vec![U8Set::empty(); nt];
    for reset_state in tokenizer.deterministic_reset_states() {
        for (byte, target) in tokenizer.transitions_from(reset_state) {
            for terminal in tokenizer.matched_terminal_bitset(target).iter() {
                if terminal < nt {
                    first_bytes[terminal].insert(byte);
                }
            }
            for terminal in tokenizer.possible_future_terminals(target).iter() {
                if terminal < nt {
                    first_bytes[terminal].insert(byte);
                }
            }
        }
    }

    SharedClassifyBytesets {
        reachable_bytes,
        first_bytes,
        last_bytes,
        transitions_by_byte: Vec::new(),
        sparse_transitions_by_byte: Vec::new(),
        reverse_transitions_by_byte: Vec::new(),
        matched_terminals_by_state: Arc::from(Vec::<Box<[u32]>>::new()),
        future_terminals_by_state: Arc::from(Vec::<Box<[u32]>>::new()),
        matched_states_by_terminal: Arc::from(Vec::<Vec<u32>>::new()),
        future_states_by_terminal: Arc::from(Vec::<Vec<u32>>::new()),
        has_matched_terminal_by_state: Vec::new(),
        future_by_state_words: Vec::new(),
        representative_future_terminal_by_state: Vec::new(),
        words_per_terminal_set: 0,
        active_route_setup_cache: Mutex::new(HashMap::new()),
    }
}

#[test]
fn prepared_suffix_trie_matches_scalar_dense_suffix_execution() {
    let vocab = Vocab::new(vec![
        (0, b"zbc".to_vec()),
        (1, b"xbcq".to_vec()),
        (2, b"cccc".to_vec()),
        (3, b"ab".to_vec()),
        (4, b"zz".to_vec()),
    ]);
    let trie = vocab_suffix_trie(&vocab);
    let expressions = Arc::<[Expr]>::from(
        vec![
            Expr::U8Seq(b"b".to_vec()),
            Expr::U8Seq(b"bc".to_vec()),
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"c".to_vec())),
                min: 1,
                max: Some(3),
            },
        ]
        .into_boxed_slice(),
    );
    let tokenizer = build_regex(expressions.as_ref())
        .into_tokenizer(expressions.len() as u32, Some(Arc::clone(&expressions)));
    let flat = build_flat_transition_table(&tokenizer);
    let mut finalizers = vec![0u64; tokenizer.num_states() as usize];
    let mut futures = vec![0u64; tokenizer.num_states() as usize];
    for state in 0..tokenizer.num_states() as usize {
        finalizers[state] = tokenizer.matched_terminal_bitset(state as u32).words()[0];
        futures[state] = tokenizer.possible_future_terminals(state as u32).words()[0];
    }
    let reset = tokenizer.initial_state_id();
    let (states, matched) = execute_prepared_dense_suffix_trie(
        &trie,
        reset,
        &flat,
        &finalizers,
        &futures,
    );
    let split_index = vocab_adjacent_pair_index(&vocab);

    for (entry_index, bytes) in vocab.entries_map().values().enumerate() {
        let split_base = split_index.entry_split_offsets[entry_index];
        for split_after in 0..bytes.len().saturating_sub(1) {
            let mut state = reset;
            let mut scalar_matched = 0u64;
            let mut consumed_suffix = true;
            for &byte in &bytes[split_after + 1..] {
                if futures[state as usize] == 0 {
                    consumed_suffix = false;
                    break;
                }
                state = flat[state as usize * 256 + byte as usize];
                if state == u32::MAX {
                    consumed_suffix = false;
                    break;
                }
                scalar_matched |= finalizers[state as usize];
            }
            if consumed_suffix {
                scalar_matched |= futures[state as usize];
            }

            let split = split_base + split_after;
            let node = trie.split_nodes[split] as usize;
            let mut trie_matched = matched[node];
            let trie_state = states[node];
            if trie_state != u32::MAX {
                trie_matched |= futures[trie_state as usize];
            }
            assert_eq!(
                trie_matched, scalar_matched,
                "suffix mismatch for token {:?} after byte {}",
                bytes, split_after,
            );
        }
    }
}

#[test]
fn parse_exact_l2p_boundary_filter_mode_accepts_auto_and_forced_values() {
    assert!(matches!(parse_exact_l2p_boundary_filter_mode(""), ExactL2pBoundaryFilterMode::Auto));
    assert!(matches!(parse_exact_l2p_boundary_filter_mode("auto"), ExactL2pBoundaryFilterMode::Auto));
    assert!(matches!(parse_exact_l2p_boundary_filter_mode("1"), ExactL2pBoundaryFilterMode::Force(true)));
    assert!(matches!(parse_exact_l2p_boundary_filter_mode("true"), ExactL2pBoundaryFilterMode::Force(true)));
    assert!(matches!(parse_exact_l2p_boundary_filter_mode("on"), ExactL2pBoundaryFilterMode::Force(true)));
    assert!(matches!(parse_exact_l2p_boundary_filter_mode("0"), ExactL2pBoundaryFilterMode::Force(false)));
    assert!(matches!(parse_exact_l2p_boundary_filter_mode("false"), ExactL2pBoundaryFilterMode::Force(false)));
    assert!(matches!(parse_exact_l2p_boundary_filter_mode("off"), ExactL2pBoundaryFilterMode::Force(false)));
}

#[test]
fn deterministic_dispatch_suffix_uses_all_reset_roots() {
    let expressions = vec![
        Expr::U8Seq(b"a".to_vec()),
        Expr::U8Seq(b"b".to_vec()),
    ];
    let tokenizer = build_regex_partitioned_with_adaptive(&expressions, &[0, 1], false)
        .into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
    assert!(tokenizer.has_deterministic_dispatch());

    let mut allowed = BitSet::new(2);
    allowed.set(1);
    assert!(suffix_has_allowed_l2p_follow_from_reset(
        &tokenizer,
        b"b",
        &allowed,
    ));
    assert!(!suffix_has_allowed_l2p_follow_from_reset(
        &tokenizer,
        b"a",
        &allowed,
    ));
}

#[test]
fn epsilon_nfa_exact_boundary_batch_matches_scalar_state_set_execution() {
    let expressions = vec![
        Expr::U8Seq(b"ab".to_vec()),
        Expr::U8Seq(b"cd".to_vec()),
    ];
    let tokenizer = build_regex_partitioned_with_adaptive(&expressions, &[0, 1], false)
        .into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
    assert!(tokenizer.has_epsilon_transitions());

    let bytesets = SharedClassifyBytesets::build(&tokenizer, tokenizer.num_terminals());
    let flat_trans = build_flat_transition_table(&tokenizer);
    let mut active = BitSet::new(tokenizer.num_terminals() as usize);
    for terminal in 0..tokenizer.num_terminals() as usize {
        active.set(terminal);
    }
    let disallowed = BTreeMap::new();
    let active_start_states = (0..tokenizer.num_states()).collect::<Vec<_>>();
    let tokens = [
        b"abcd".as_slice(),
        b"abce".as_slice(),
        b"abab".as_slice(),
        b"cdab".as_slice(),
        b"zzzz".as_slice(),
    ];

    let batch = tokens_have_exact_active_l2p_boundary(
        &tokenizer,
        &bytesets,
        &flat_trans,
        &bytesets.transitions_by_byte,
        &tokens,
        &active,
        &disallowed,
        &active_start_states,
        None,
    );
    let scalar = tokens
        .iter()
        .map(|bytes| {
            token_has_exact_active_l2p_boundary(
                &tokenizer,
                bytes,
                &active,
                &disallowed,
                &active_start_states,
            )
        })
        .collect::<Vec<_>>();

    assert_eq!(batch, scalar);
    assert!(batch.iter().any(|&boundary| boundary));
    assert!(batch.iter().any(|&boundary| !boundary));
}

#[test]
fn terminal_path_classification_requires_a_real_within_token_boundary() {
    let expressions = vec![
        Expr::U8Seq(b"key ".to_vec()),
        Expr::U8Seq(b"c".to_vec()),
        Expr::U8Seq(b"a".to_vec()),
        Expr::U8Seq(b"b".to_vec()),
    ];
    let tokenizer = build_regex(&expressions).into_tokenizer(
        expressions.len() as u32,
        Some(Arc::from(expressions.into_boxed_slice())),
    );
    // Space and `c` both occur somewhere in the vocabulary, so the old
    // global last-byte/first-byte overlap heuristic classified `key ` and
    // `c` as L2P. No single token actually places a `c`-starting suffix
    // after a completion of `key `. The `a|b` split in `ab` is real.
    let vocab = Vocab::new(
        vec![
            (0, b" xyz ".to_vec()),
            (1, b"c".to_vec()),
            (2, b"ab".to_vec()),
        ]);

    let lengths = classify_terminal_path_lengths(
        "test",
        &tokenizer,
        &vocab,
        &BTreeMap::new(),
        tokenizer.num_terminals(),
        None,
    );

    assert_eq!(
        lengths,
        vec![
            TerminalPathLength::One,
            TerminalPathLength::One,
            TerminalPathLength::TwoPlus,
            TerminalPathLength::TwoPlus,
        ],
    );
}

#[test]
fn terminal_path_classification_propagates_later_boundaries_inside_a_token() {
    let expressions = vec![
        Expr::Repeat {
            expr: Box::new(Expr::U8Seq(b"a".to_vec())),
            min: 1,
            max: None,
        },
        Expr::Repeat {
            expr: Box::new(Expr::U8Seq(b" ".to_vec())),
            min: 1,
            max: None,
        },
        Expr::U8Seq(b"c".to_vec()),
    ];
    let tokenizer = build_regex(&expressions).into_tokenizer(
        expressions.len() as u32,
        Some(Arc::from(expressions.into_boxed_slice())),
    );
    let vocab = Vocab::new(vec![(0, b"a c".to_vec())]);
    let mut after_a = BitSet::all(3);
    after_a.clear(1);
    let mut after_space = BitSet::all(3);
    after_space.clear(2);
    let disallowed = BTreeMap::from([(0u32, after_a), (1u32, after_space)]);

    let lengths = classify_terminal_path_lengths(
        "test",
        &tokenizer,
        &vocab,
        &disallowed,
        tokenizer.num_terminals(),
        None,
    );

    assert_eq!(
        lengths,
        vec![
            TerminalPathLength::TwoPlus,
            TerminalPathLength::TwoPlus,
            TerminalPathLength::TwoPlus,
        ],
        "C participates in the later WS -> C boundary of A -> WS -> C",
    );
}

#[test]
fn candidate_dfa_terminal_paths_match_generic_exact_epsilon_reference() {
    let expressions = vec![
        Expr::U8Seq(b"ab".to_vec()),
        Expr::U8Seq(b"c".to_vec()),
        Expr::Repeat {
            expr: Box::new(Expr::U8Seq(b"a".to_vec())),
            min: 1,
            max: Some(3),
        },
        Expr::Choice(vec![Expr::U8Seq(b"xy".to_vec()), Expr::U8Seq(b"z".to_vec())]),
    ];
    let num_terminals = expressions.len();
    let tokenizer = build_regex_partitioned_with_adaptive(
        &expressions,
        &[0, 1, 2, 3],
        false,
    )
    .into_tokenizer(
        expressions.len() as u32,
        Some(Arc::from(expressions.into_boxed_slice())),
    );
    assert!(tokenizer.has_epsilon_transitions());
    let vocab = Vocab::new(
        vec![
            (0, b"abc".to_vec()),
            (1, b"aa".to_vec()),
            (2, b"az".to_vec()),
            (3, b"xyc".to_vec()),
            (4, b"q".to_vec()),
        ]);
    let mut blocked_after_ab = BitSet::new(num_terminals);
    blocked_after_ab.set(1);
    let disallowed = BTreeMap::from([(0u32, blocked_after_ab)]);
    let active = BitSet::all(num_terminals);
    let bytesets = SharedClassifyBytesets::build(&tokenizer, tokenizer.num_terminals());

    let optimized = exact_terminal_path_two_plus_candidate_dfa(
        &tokenizer,
        &vocab,
        &disallowed,
        &active,
        &bytesets,
        None,
    );
    let reference = exact_terminal_path_two_plus(
        &tokenizer,
        &vocab,
        &disallowed,
        &bytesets,
        &active,
    );

    assert_eq!(optimized.two_plus, reference.two_plus);
}

#[test]
fn finite_literal_terminal_paths_match_generic_exact_reference() {
    let expressions = vec![
        Expr::U8Seq(b"ab".to_vec()),
        Expr::U8Seq(b"b".to_vec()),
        Expr::Choice(vec![Expr::U8Seq(b"cd".to_vec()), Expr::U8Seq(b"cde".to_vec())]),
        Expr::Seq(vec![Expr::U8Seq(b"x".to_vec()), Expr::U8Seq(b"y".to_vec())]),
    ];
    let num_terminals = expressions.len();
    let tokenizer = build_regex_partitioned_with_adaptive(
        &expressions,
        &[0, 1, 2, 3],
        false,
    )
    .into_tokenizer(
        expressions.len() as u32,
        Some(Arc::from(expressions.into_boxed_slice())),
    );
    let vocab = Vocab::new(
        vec![
            (0, b"abb".to_vec()),
            (1, b"bc".to_vec()),
            (2, b"abcd".to_vec()),
            (3, b"cdey".to_vec()),
            (4, b"xyab".to_vec()),
            (5, b"zz".to_vec()),
        ]);
    let mut blocked = BitSet::new(num_terminals);
    blocked.set(1);
    let disallowed = BTreeMap::from([(0u32, blocked)]);
    let active = BitSet::all(num_terminals);
    let bytesets = SharedClassifyBytesets::build(&tokenizer, tokenizer.num_terminals());

    let specialized = exact_terminal_path_two_plus_finite_literals(
        &tokenizer,
        &vocab,
        &disallowed,
        &active,
    )
    .expect("finite literal specialization");
    let reference = exact_terminal_path_two_plus(
        &tokenizer,
        &vocab,
        &disallowed,
        &bytesets,
        &active,
    );

    assert_eq!(specialized.two_plus, reference.two_plus);
}

#[test]
fn finite_literal_classification_propagates_later_token_boundaries() {
    let expressions = vec![Expr::U8Seq(b"+".to_vec()), Expr::U8Seq(b"a".to_vec())];
    let tokenizer = build_regex_partitioned_with_adaptive(&expressions, &[0, 1], false)
        .into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
    let vocab = Vocab::new(vec![(0, b"++++++++a".to_vec())]);
    let active = BitSet::all(2);
    let disallowed = BTreeMap::new();
    let bytesets = SharedClassifyBytesets::build(&tokenizer, tokenizer.num_terminals());

    let specialized = exact_terminal_path_two_plus_finite_literals(
        &tokenizer,
        &vocab,
        &disallowed,
        &active,
    )
    .expect("finite literal specialization");
    let candidate = exact_terminal_path_two_plus_candidate_dfa(
        &tokenizer,
        &vocab,
        &disallowed,
        &active,
        &bytesets,
        None,
    );

    assert_eq!(specialized.two_plus, active);
    assert_eq!(candidate.two_plus, active);
    assert!(specialized.witnesses[1].as_ref().is_some_and(|witness| {
        witness.terminal_1 == 0
            && witness.terminal_2 == 1
            && witness.split_position == 8
    }));
}

#[test]
fn candidate_dfa_direct_prefix_states_match_reference_across_mask_words() {
    let expressions = (0..70)
        .map(|terminal| Expr::U8Seq(format!("t{terminal:02}").into_bytes()))
        .collect::<Vec<_>>();
    let partitions = (0..expressions.len() as u32).collect::<Vec<_>>();
    let tokenizer = build_regex_partitioned_with_adaptive(
        &expressions,
        &partitions,
        false,
    )
    .into_tokenizer(
        expressions.len() as u32,
        Some(Arc::from(expressions.into_boxed_slice())),
    );
    let vocab = Vocab::new(
        (0..70)
            .map(|terminal| {
                let next = (terminal + 1) % 70;
                (
                    terminal as u32,
                    format!("t{terminal:02}t{next:02}").into_bytes(),
                )
            })
            .collect());
    let active = BitSet::all(70);
    let disallowed = BTreeMap::new();
    let bytesets = SharedClassifyBytesets::build(&tokenizer, tokenizer.num_terminals());

    let optimized = exact_terminal_path_two_plus_candidate_dfa(
        &tokenizer,
        &vocab,
        &disallowed,
        &active,
        &bytesets,
        None,
    );
    let reference = exact_terminal_path_two_plus(
        &tokenizer,
        &vocab,
        &disallowed,
        &bytesets,
        &active,
    );

    assert_eq!(optimized.two_plus, reference.two_plus);
}

#[test]
fn ordered_custom_regex_partitions_have_exact_effective_languages() {
    let rules = custom_vocab_partition_rules(
        r#"[
                {"name":"short","regex":"[a-z]{1,2}"},
                {"name":"medium","regex":"[a-z]{1,4}"}
            ]"#,
    )
    .unwrap();
    let set = compile_vocab_partition_set(rules, true).unwrap();
    assert_eq!(set.languages.len(), 3);
    assert_eq!(set.languages[0].name, "short");
    assert_eq!(set.languages[1].name, "medium");
    assert_eq!(set.languages[2].name, "rest");
    assert_eq!(classify_with_partition_set(&set, b"a"), Some(0));
    assert_eq!(classify_with_partition_set(&set, b"ab"), Some(0));
    assert_eq!(classify_with_partition_set(&set, b"abc"), Some(1));
    assert_eq!(classify_with_partition_set(&set, b"abcd"), Some(1));
    assert_eq!(classify_with_partition_set(&set, b"abcde"), Some(2));
    assert_eq!(classify_with_partition_set(&set, b"\""), Some(2));

    let mut medium = set.languages[1].compile_effective_regex().unwrap();
    assert!(!medium.is_match_bytes(b"a"));
    assert!(!medium.is_match_bytes(b"ab"));
    assert!(medium.is_match_bytes(b"abc"));
    assert!(medium.is_match_bytes(b"abcd"));
    assert!(!medium.is_match_bytes(b"abcde"));
}

#[test]
fn newline_custom_regex_partition_syntax_is_supported() {
    let rules = custom_vocab_partition_rules("[0-9]+\n[A-Za-z_]+\n").unwrap();
    let set = compile_vocab_partition_set(rules, true).unwrap();
    assert_eq!(classify_with_partition_set(&set, b"123"), Some(0));
    assert_eq!(classify_with_partition_set(&set, b"abc"), Some(1));
    assert_eq!(classify_with_partition_set(&set, b"!"), Some(2));
}

#[test]
fn underscore_is_alphabetic_for_vocab_partitioning() {
    // `_` is treated like an ASCII alphabetic byte only for partition
    // routing. It must keep identifier-style tokens out of punctuation
    // partitions, including the quoted structural-boundary route.
    for bytes in [b"_".as_slice(), b" _", b"snake_case", b"123_456", b"__"] {
        assert_eq!(classify_vocab_char_type(bytes), 2, "bytes={bytes:?}");
    }
    assert_eq!(classify_vocab_char_type(b"123"), 3);
    assert_eq!(classify_vocab_char_type(b"_!"), 1);
    assert_eq!(classify_vocab_char_type(b"\"_field"), 8);
}

#[test]
fn structural_boundary_lexical_tokens_split_literal_and_quoted_identifier_routes() {
    for bytes in [
        b" true".as_slice(),
        b" nullptr".as_slice(),
        b"[n".as_slice(),
        b" -".as_slice(),
    ] {
        assert_eq!(classify_vocab_char_type(bytes), 7, "bytes={bytes:?}");
    }
    for bytes in [
        b"t".as_slice(),
        b"true".as_slice(),
        b"falsehood".as_slice(),
        b"nullable".as_slice(),
    ] {
        assert_eq!(classify_vocab_char_type(bytes), 2, "bytes={bytes:?}");
    }
    for bytes in [
        b"\"name".as_slice(),
        b"\"_field".as_slice(),
        b"\"This".as_slice(),
    ] {
        assert_eq!(classify_vocab_char_type(bytes), 8, "bytes={bytes:?}");
    }
}

#[test]
fn combined_l2p_route_hint_matches_two_pass_route_scan() {
    let mut active_reachable = U8Set::empty();
    active_reachable.insert(b'a');
    active_reachable.insert(b'z');
    let mut active_reachable_by_byte = [0u8; 256];
    for byte in active_reachable.iter() {
        active_reachable_by_byte[byte as usize] = 1;
    }
    let mut pairs = [0u64; 1024];
    let pair_index = ((b'x' as usize) << 8) | b'y' as usize;
    pairs[pair_index >> 6] |= 1u64 << (pair_index & 63);

    for bytes in [b"".as_slice(), b"a", b"q", b"xy", b"xya", b"qz", b"ax"] {
        let expected = if token_has_active_l2p_boundary_words(bytes, &pairs) {
            TokenL2pRouteHint::Adjacent
        } else if bytes.iter().any(|&byte| active_reachable.contains(byte)) {
            TokenL2pRouteHint::Single
        } else {
            TokenL2pRouteHint::Irrelevant
        };
        assert_eq!(
            token_l2p_route_hint(bytes, &pairs, &active_reachable_by_byte),
            expected,
            "bytes={bytes:?}"
        );
    }
}

#[test]
fn reverse_byte_transition_index_preserves_frontier_targets() {
    let by_byte = vec![
        vec![(0, 2), (2, 2), (3, 5), (4, 2), (5, 5)],
        vec![(1, 4), (2, 4)],
    ];
    let reverse = build_reverse_transitions_by_byte(&by_byte, 6);

    assert_eq!(reverse[0].targets, vec![2, 5]);
    assert_eq!(reverse[0].source_offsets, vec![0, 3, 5]);
    assert_eq!(reverse[0].sources, vec![0, 2, 4, 3, 5]);

    let frontier = BTreeSet::from([2u32, 3, 5]);
    let direct_targets = by_byte[0]
        .iter()
        .filter_map(|&(source, target)| frontier.contains(&source).then_some(target))
        .collect::<BTreeSet<_>>();
    let reverse_targets = reverse[0]
        .targets
        .iter()
        .enumerate()
        .filter_map(|(index, &target)| {
            let start = reverse[0].source_offsets[index] as usize;
            let end = reverse[0].source_offsets[index + 1] as usize;
            reverse[0].sources[start..end]
                .iter()
                .any(|source| frontier.contains(source))
                .then_some(target)
        })
        .collect::<BTreeSet<_>>();

    assert_eq!(reverse_targets, direct_targets);
}

#[test]
fn byte_bucket_bytesets_match_reference_transition_scan() {
    let expressions = vec![
        Expr::U8Seq(b"a".to_vec()),
        Expr::U8Seq(b"ab".to_vec()),
        Expr::Choice(vec![Expr::U8Seq(b"ac".to_vec()), Expr::U8Seq(b"b".to_vec())]),
        Expr::Seq(vec![
            Expr::U8Class(U8Set::from_bytes(b"xy")),
            Expr::Shared(Arc::new(Expr::U8Seq(b"z".to_vec()))),
        ]),
    ];
    let tokenizer = build_regex(&expressions).into_tokenizer(
        expressions.len() as u32,
        Some(Arc::from(expressions.into_boxed_slice())),
    );

    let actual = SharedClassifyBytesets::build(&tokenizer, tokenizer.num_terminals());
    let expected = reference_bytesets(&tokenizer, tokenizer.num_terminals());
    assert_eq!(actual.reachable_bytes, expected.reachable_bytes);
    assert_eq!(actual.first_bytes, expected.first_bytes);
    assert_eq!(actual.last_bytes, expected.last_bytes);
    for state in 0..tokenizer.num_states() {
        assert_eq!(
            actual.has_matched_terminal_by_state[state as usize] != 0,
            tokenizer
                .matched_terminals_iter(state)
                .any(|terminal| terminal < tokenizer.num_terminals()),
            "state={state}"
        );
    }

    let mut active_sets = vec![BitSet::new(tokenizer.num_terminals() as usize)];
    let mut all_active = BitSet::new(tokenizer.num_terminals() as usize);
    for terminal in 0..tokenizer.num_terminals() as usize {
        all_active.set(terminal);
        let mut single = BitSet::new(tokenizer.num_terminals() as usize);
        single.set(terminal);
        active_sets.push(single);
    }
    active_sets.push(all_active);
    let flat_trans = build_flat_transition_table(&tokenizer);
    let unrestricted_suffix_start = [1u8; 256];
    for active in &active_sets {
        let active_suffix_start =
            build_active_suffix_start_by_byte(&tokenizer, &actual, active.words());
        for bytes in [
            b"".as_slice(),
            b"a",
            b"ab",
            b"ac",
            b"xy",
            b"xyz",
            b"zz",
            b"bxyz",
        ] {
            assert_eq!(
                token_has_active_terminal_suffix(
                    &tokenizer,
                    &actual,
                    &flat_trans,
                    bytes,
                    active.words(),
                    None,
                    &active_suffix_start,
                ),
                token_has_active_terminal_suffix(
                    &tokenizer,
                    &actual,
                    &flat_trans,
                    bytes,
                    active.words(),
                    None,
                    &unrestricted_suffix_start,
                ),
                "active={active:?} bytes={bytes:?}"
            );
        }
        for state in 0..tokenizer.num_states() {
            let representative = actual.representative_future_terminal_by_state[state as usize];
            let future = tokenizer.possible_future_terminals(state);
            assert_eq!(representative == u32::MAX, future.is_empty());
            if representative != u32::MAX {
                assert!(future.contains(representative as usize));
            }
            let full = state_future_intersects_words(&actual, state, active.words());
            let fast = representative != u32::MAX && active.contains(representative as usize)
                || (representative == u32::MAX || !active.contains(representative as usize))
                    && full;
            assert_eq!(fast, full, "state={state}");
        }
    }
}
