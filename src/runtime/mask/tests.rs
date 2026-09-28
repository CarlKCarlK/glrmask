use super::{
    exact_component_trigger_accepted_weight, single_path_direct_plan_reuse_dominates,
    single_path_direct_stack_work,
    DenseMaskAcc,
    DenseTokenMaskCache,
    DenseTokenSetIntersectionSmallCache,
    MASK_SINGLE_PATH_DIRECT_INLINE_PATH_CAPACITY,
    MASK_SINGLE_PATH_DIRECT_MAX_PATHS_PER_GSS,
    MASK_SINGLE_PATH_DIRECT_MAX_TOTAL_STACK_VALUES,
};
use crate::automata::lexer::Lexer;
use crate::compiler::glr::accumulator::TerminalsDisallowed;
use crate::compiler::glr::parser::ParserGSS;
use crate::{Constraint as Constraint, Grammar, Vocab};
use range_set_blaze::RangeSetBlaze;
use rustc_hash::FxHashMap;
use std::sync::Arc;

fn precomputed_for(
    token_set: &Arc<RangeSetBlaze<u32>>,
    mask: Arc<[u64]>,
) -> DenseTokenMaskCache {
    let mut precomputed: FxHashMap<usize, Arc<[u64]>> = FxHashMap::default();
    precomputed.insert(Arc::as_ptr(token_set) as usize, mask);
    precomputed
}

fn mask_contains(mask: &[u32], token: u32) -> bool {
    mask.get(token as usize / 32)
        .is_some_and(|word| word & (1u32 << (token % 32)) != 0)
}

#[test]
fn component_mask_word_copy_matches_exact_membership_dense_sparse_and_specials() {
    // Dense prefixes ending on and between word boundaries; sparse holes;
    // byte aliases; out-of-vocabulary specials; and longer output buffers.
    let vocabularies = [
        (0..2).collect::<Vec<u32>>(),
        (0..32).collect::<Vec<u32>>(),
        (0..35).collect::<Vec<u32>>(),
        vec![0, 3, 31, 32, 65, 96],
    ];
    for ids in vocabularies {
        let vocab = Vocab::new(ids.iter().map(|&id| (id, b"a".to_vec())).collect());
        let original = crate::ConstraintSpec::builder(
            Grammar::glrm("glrm 1; start document; extern token SPECIAL; nt document = \"a\" | SPECIAL;"),
            &vocab,
        ).unwrap().bind_token("SPECIAL", [70, 130]).unwrap().build().unwrap().compile().unwrap();
        let loaded = Constraint::load(original.save()).unwrap();
        for constraint in [&original, &loaded] {
            let state = constraint.start();
            for seed in [0u32, 0x5555_5555, 0xffff_ffff] {
                let mut actual = vec![seed; constraint.mask_len() + 2];
                let mut expected = actual.clone();
                let source: Vec<_> = (0..actual.len() + 1)
                    .map(|i| if seed == 0 { u32::MAX } else { seed.rotate_left(i as u32) })
                    .collect();
                for (word, (&bits, output)) in source.iter().zip(&mut expected).enumerate() {
                    for bit in 0..32 {
                        if bits & (1u32 << bit) != 0 && state.knows_token_id(word as u32 * 32 + bit) {
                            *output |= 1u32 << bit;
                        }
                    }
                }
                state.or_segmented_component_mask(&mut actual, &source);
                assert_eq!(actual, expected, "ids={ids:?}, seed={seed}");
            }
        }
    }
}

#[test]
fn segmented_static_result_cache_tracks_exact_scoped_state() {
    let vocab = Vocab::new(vec![
        (0, b"[".to_vec()), (1, b"a".to_vec()), (2, b"]".to_vec()),
        (3, b"!".to_vec()), (4, b"[aa]!".to_vec()), (5, b" ".to_vec()),
    ]);
    let child = Constraint::compile(Grammar::glrm(
        r#"start child; ignore WS; t WS ::= " "+; t WORD ::= /[a-z]+/; nt child ::= WORD;"#
    ), &vocab).unwrap();
    let parent = Constraint::compile(Grammar::glrm(
        r#"start document; extern grammar child; nt document ::= "[" child "]!";"#
    ), &vocab).unwrap();
    let bound = parent.bind_grammar("child", child).unwrap();
    assert!(bound.uses_compact_segmented_parser_runtime());
    let loaded = Constraint::load(bound.save()).unwrap();
    for constraint in [&bound, &loaded] {
        let mut state = constraint.start();
        let mut unchanged_cache_reuses = 0;
        for token in [0,1,1,1,5,2,3] {
            let mask = state.mask();
            assert!(mask_contains(&mask, token));
            let mut cached = vec![0; mask.len()];
            assert!(state.try_fill_mask_from_cache(&mut cached));
            assert_eq!(cached, mask);
            let before = state.state.clone();
            state.commit_token(token).unwrap();
            if before == state.state {
                assert!(state.try_fill_mask_from_cache(&mut cached));
                unchanged_cache_reuses += 1;
            } else {
                assert!(!state.try_fill_mask_from_cache(&mut cached));
            }
            assert_eq!(state.mask(), state.clone().mask(), "cached/fresh token={token}");
        }
        assert!(unchanged_cache_reuses > 0, "fixture must exercise a repeated-state cache hit");
        assert!(state.is_accepting());
        assert!(state.commit_token(0).is_err());
        assert!(state.mask().iter().all(|&word| word == 0), "failed commit invalidates cache");
        assert_eq!(state.mask(), state.clone().mask());
    }
}


fn exact_start_trigger_contains(constraint: &Constraint, token: u32) -> bool {
    let crate::runtime::BoundaryTrigger::Exact(dwa) = &constraint.boundary_trigger else {
        panic!("constraint does not carry an Exact boundary trigger");
    };
    let state = constraint.start();
    let mut accepted = false;
    for (&tokenizer_state, gss) in state.state.iter() {
        let complete = gss.for_each_stack_top_first_bounded(128, |top_first, _| {
            let weight =
                exact_component_trigger_accepted_weight(constraint, dwa, top_first);
            if weight.tokens_for_tsid(tokenizer_state).contains(token) {
                accepted = true;
            }
        });
        assert!(complete, "tiny trigger test GSS traversal must complete");
    }
    accepted
}

#[test]
fn exact_finish_trigger_requires_a_proper_internal_offset() {
    let vocab = Vocab::new(vec![
        (0, b"x".to_vec()),
        (1, b"xy".to_vec()),
        (2, b"xx".to_vec()),
        (3, b"yx".to_vec()),
    ]);
    let mut constraint = Constraint::from_ebnf(r#"start ::= "x""#, &vocab).unwrap();
    constraint.build_exact_boundary_trigger().unwrap();

    assert!(exact_start_trigger_contains(&constraint, 1));
    assert!(exact_start_trigger_contains(&constraint, 2));
    assert!(
        !exact_start_trigger_contains(&constraint, 0),
        "finish exactly at model-token end is not an internal trigger",
    );
    assert!(!exact_start_trigger_contains(&constraint, 3));
}

#[test]
fn exact_entry_trigger_requires_placeholder_readiness_after_prefix() {
    let vocab = Vocab::new(vec![
        (0, b"x".to_vec()),
        (1, b"xy".to_vec()),
        (2, b"y".to_vec()),
    ]);
    let mut parent = Constraint::from_glrm_grammar_with_unbound_subgrammars_bindings_and_end_tokens(
        "glrm 1; start start; extern grammar child; nt start = \"x\" child;",
        &vocab,
        &[],
        &[],
    )
    .unwrap();
    parent.build_exact_boundary_trigger().unwrap();

    assert!(exact_start_trigger_contains(&parent, 1));
    assert!(
        !exact_start_trigger_contains(&parent, 0),
        "placeholder reached only at model-token end is handled by next-call control closure",
    );
    assert!(!exact_start_trigger_contains(&parent, 2));
}

#[test]
fn recursive_exact_full_walk_ignores_outer_table_and_tokenizer() {
    let vocab = Vocab::new(vec![
        (0, b"X".to_vec()),
        (1, b"a".to_vec()),
        (2, b"!".to_vec()),
        (3, b"Xa!".to_vec()),
        (4, b"a!".to_vec()),
    ]);
    let child = Constraint::compile(
        Grammar::glrm("glrm 1; start child; nt child = \"a\";"),
        &vocab,
    )
    .unwrap();
    let parent = Constraint::compile(
        Grammar::glrm(
            "glrm 1; start document; extern grammar child; nt document = \"X\" child \"!\";",
        ),
        &vocab,
    )
    .unwrap();
    let bound = parent
        .bind_grammar_dynamic_boundary("child", child)
        .unwrap();
    let monolithic = Constraint::compile(
        Grammar::glrm("glrm 1; start document; nt document = \"X\" \"a\" \"!\";"),
        &vocab,
    )
    .unwrap();

    let mut poisoned = Constraint::load(bound.save()).unwrap();
    poisoned.recursive_parser_layout().unwrap().unwrap();
    let root = poisoned
        .static_dynamic_overlay
        .as_ref()
        .unwrap()
        .segmented_parser_components[0]
        .constraint
        .clone();
    poisoned.tokenizer = root.tokenizer.clone();
    poisoned.tokenizer_fast_transitions = root.tokenizer_fast_transitions.clone();
    poisoned.tokenizer_has_epsilon_transitions = root.tokenizer_has_epsilon_transitions;
    poisoned.table.action.clear();
    poisoned.table.goto.clear();
    poisoned.table.advance.clear();
    poisoned.table.unconditional_advance.clear();
    poisoned.table.rules.clear();
    poisoned.table.forwarded_shifts.clear();
    poisoned.table.control_terminals.clear();
    poisoned.table.skip_terminals.clear();
    poisoned.table.guarded_shift_index.clear();
    poisoned.table.direct_regular_wide_frontiers.clear();
    poisoned.table.num_states = 0;
    poisoned.table.num_terminals = 0;
    poisoned.table.num_rules = 0;

    let mut actual = poisoned.start();
    let mut expected = monolithic.start();
    for prefix_token in [None, Some(0), Some(1)] {
        if let Some(token) = prefix_token {
            actual.commit_token(token).unwrap();
            expected.commit_token(token).unwrap();
        }
        let mut fallback = vec![0u32; poisoned.body_mask_len()];
        actual.fill_recursive_mask_by_exact_full_walk(&mut fallback);
        let expected_mask = expected.mask();
        assert_eq!(fallback, expected_mask);

        let mut dynamic_reference = vec![0u32; poisoned.body_mask_len()];
        actual.fill_mask_dynamic(&mut dynamic_reference);
        assert_eq!(dynamic_reference, expected_mask);

        let mut profiled = vec![0u32; poisoned.body_mask_len()];
        actual.fill_mask_profiled(&mut profiled);
        assert_eq!(profiled, expected_mask);
    }
}

/// The former recursive walker deliberately scanned dead descendants.
/// Keep that independent control flow as a regression oracle for DFS jumps.
fn recursive_unpruned_reference(state: &super::ConstraintState<'_>) -> Vec<u32> {
    let vocab = state.constraint.dynamic_mask_vocab_for_runtime();
    let trie = vocab.trie.as_ref();
    let mut buffers = super::CommitBuffers::default();
    let mut output = vec![0u32; state.constraint.mask_len()];
    let mut stack = vec![None; usize::from(trie.full_walk_max_parent_depth()) + 2];
    stack[0] = Some(state.state.clone());
    let mark = |node, output: &mut [u32]| {
        if let Some(canonical) = trie.node(node).token_id {
            for &id in vocab.token_ids(canonical).unwrap() {
                if !state.constraint.has_special_token_id(id) {
                    super::set_original_mask_bit(output, id);
                }
            }
        }
    };
    // The root's zero-byte aliases are not model-token byte transitions.
    // Exact special-token paths for those IDs are evaluated below.
    for edge in trie.walk_edges() {
        let depth = edge.parent_depth as usize;
        let next = stack[depth].as_ref().and_then(|parent| {
            crate::runtime::commit::advance_bytes_from_state_exact(
                state.constraint, parent, &mut buffers, trie.walk_edge_bytes(edge),
            )
        });
        if next.is_some() {
            mark(edge.child, &mut output);
        }
        stack[depth + 1] = next;
    }
    for special in &state.constraint.special_token_terminals {
        if !state.constraint.is_late_grammar_placeholder_terminal(special.terminal_id)
            && recursive_token_domain_reference(state, &mut buffers, special.token_id)
        {
            super::set_original_mask_bit(&mut output, special.token_id);
        }
    }
    output
}

fn recursive_token_domain_reference(
    state: &super::ConstraintState<'_>,
    buffers: &mut super::CommitBuffers,
    token_id: u32,
) -> bool {
    if state.constraint.token_bytes_for_id(token_id).is_some_and(|b| b.is_empty()) {
        // commit_bytes(empty) is intentionally a no-op; it is not a proof
        // of model-token admission. Match the explicit current public API
        // contract without suppressing any nonempty or special-ID probes.
        state.constraint.has_special_token_id(token_id)
            && crate::runtime::commit::advance_special_token_paths(
                state.constraint, &state.state, token_id,
            ).is_some_and(|gss| !gss.is_empty())
    } else {
        crate::runtime::commit::token_admissible_from_state_exact(
            state.constraint, &state.state, buffers, token_id,
        )
    }
}

#[test]
fn recursive_dead_subtree_jumps_match_unpruned_and_pointwise_oracles() {
    // Large dead families share ancestors with live siblings. Duplicate
    // byte spellings, an empty token, and a special-token spelling within
    // a dead byte family exercise independent endpoint/semantic routing.
    let mut entries = vec![
        (0, b"X".to_vec()), (1, b"[".to_vec()), (2, b"a".to_vec()),
        (3, b"]".to_vec()), (4, b"!".to_vec()), (5, b"a]!".to_vec()),
        (6, b"[a]!".to_vec()), (7, b"X[a]!".to_vec()),
        (8, b"X[]!".to_vec()), (9, Vec::new()), (10, b"a".to_vec()),
        (11, b"qdead-special".to_vec()), (12, b"qdead-special".to_vec()),
        (13, b"]!".to_vec()), (14, b"[]!".to_vec()),
    ];
    for i in 0..512u32 {
        entries.push((32 + i, format!("qdead-{i:04}-suffix").into_bytes()));
        entries.push((600 + i, format!("X[ab-{i:04}-suffix").into_bytes()));
    }
    let ids: Vec<u32> = entries.iter().map(|(id, _)| *id).chain([9001]).collect();
    let vocab = Vocab::new(entries);
    for nullable in [false, true] {
        let leaf_source = if nullable {
            "glrm 1; start leaf; extern token SPECIAL; nt leaf = \"a\"? | SPECIAL;"
        } else {
            "glrm 1; start leaf; extern token SPECIAL; nt leaf = \"a\" | SPECIAL;"
        };
        let leaf = crate::ConstraintSpec::builder(Grammar::glrm(leaf_source), &vocab)
            .unwrap().bind_token("SPECIAL", [11, 9001]).unwrap()
            .build().unwrap().compile().unwrap();
        let middle_parent = Constraint::compile(Grammar::glrm(
            "glrm 1; start middle; extern grammar leaf; nt middle = \"[\" leaf \"]\";",
        ), &vocab).unwrap();
        let middle = middle_parent.bind_grammar_dynamic_boundary("leaf", leaf).unwrap();
        let outer_parent = Constraint::compile(Grammar::glrm(
            "glrm 1; start outer; extern grammar middle; nt outer = \"X\" middle \"!\";",
        ), &vocab).unwrap();
        let bound = outer_parent.bind_grammar_dynamic_boundary("middle", middle).unwrap();
        let loaded = Constraint::load(bound.save()).unwrap();
        for constraint in [&bound, &loaded] {
            assert!(constraint.uses_compact_segmented_parser_runtime());
            let check = |state: &super::ConstraintState<'_>| {
                let mut actual = vec![0u32; constraint.mask_len()];
                state.fill_recursive_mask_by_exact_full_walk(&mut actual);
                assert_eq!(actual, recursive_unpruned_reference(state), "nullable={nullable}");
                for budget in [0, 1, 2] {
                    let mut bounded = vec![0u32; constraint.mask_len()];
                    state.or_recursive_dynamic_full_walk_exact_with_budget(&mut bounded, budget);
                    assert_eq!(bounded, actual, "cache budget={budget}, nullable={nullable}");
                }
                let mut buffers = super::CommitBuffers::default();
                for &id in &ids {
                    let exact = recursive_token_domain_reference(state, &mut buffers, id);
                    let allowed = actual[id as usize / 32] & (1 << (id % 32)) != 0;
                    assert_eq!(allowed, exact, "token={id}, nullable={nullable}");
                }
                assert_eq!(state.mask(), actual);
            };
            for prefix in [b"".as_slice(), b"X", b"X[", b"X[a", b"X[a]", b"X[a]!"] {
                let mut state = constraint.start();
                state.commit_bytes(prefix).unwrap();
                check(&state);
            }
            for special in [11, 9001] {
                let mut state = constraint.start();
                state.commit_bytes(b"X[").unwrap();
                let mask = state.mask();
                assert_ne!(mask[special / 32] & (1 << (special % 32)), 0);
                assert_eq!(mask[0] & (1 << 12), 0, "byte alias has no special route");
                state.commit_token(special as u32).unwrap();
                check(&state);
                state.commit_bytes(b"]!").unwrap();
                check(&state);
            }
            if nullable {
                let mut state = constraint.start();
                state.commit_bytes(b"X[]!").unwrap();
                check(&state);
            }
        }
    }
}

#[test]
fn recursive_transition_cache_matches_radix_on_text_and_ambiguity() {
    let mut words = vec![Vec::<u8>::new()];
    let mut layer = vec![Vec::<u8>::new()];
    for _ in 0..4 {
        let mut next = Vec::new();
        for prefix in layer {
            for &byte in b"abcX[]! " {
                let mut word = prefix.clone();
                word.push(byte);
                next.push(word);
            }
        }
        words.extend(next.iter().cloned());
        layer = next;
    }
    let vocab = Vocab::new(words.into_iter().enumerate()
        .map(|(id, bytes)| (id as u32, bytes)).collect::<Vec<_>>());
    for leaf_source in [
        "glrm 1; start leaf; t TEXT = /[a-c ]{1,8}/; nt leaf = TEXT;",
        "glrm 1; start leaf; t WORD = /[a-c]{1,4}/; nt leaf = WORD | WORD WORD;",
        "glrm 1; start leaf; ignore WS; t WS = \" \"+; t WORD = /[a-c]{1,4}/; nt leaf = WORD;",
    ] {
        let leaf = Constraint::compile(Grammar::glrm(leaf_source), &vocab).unwrap();
        let middle = Constraint::compile(Grammar::glrm(
            "glrm 1; start middle; extern grammar leaf; nt middle = \"[\" leaf \"]\";",
        ), &vocab).unwrap().bind_grammar_dynamic_boundary("leaf", leaf).unwrap();
        let bound = Constraint::compile(Grammar::glrm(
            "glrm 1; start outer; extern grammar middle; nt outer = \"X\" middle \"!\";",
        ), &vocab).unwrap().bind_grammar_dynamic_boundary("middle", middle).unwrap();
        let alphabet = super::recursive_mask_byte_representatives(&bound);
        assert_eq!(alphabet[b'a' as usize], alphabet[b'b' as usize]);
        assert_eq!(alphabet[b'b' as usize], alphabet[b'c' as usize]);
        assert_ne!(alphabet[b'a' as usize], alphabet[b']' as usize]);
        for prefix in [b"".as_slice(), b"X", b"X[", b"X[a", b"X[abc", b"X[abc]", b"X[abc]!"] {
            let mut state = bound.start();
            state.commit_bytes(prefix).unwrap();
            let expected = recursive_unpruned_reference(&state);
            for budget in [0, 2, 256] {
                let mut actual = vec![0u32; bound.mask_len()];
                state.or_recursive_dynamic_full_walk_exact_with_budget(&mut actual, budget);
                assert_eq!(actual, expected, "prefix={prefix:?} budget={budget} leaf={leaf_source}");
            }
        }
    }
}

#[test]
fn precomputed_dense_intersection_reuses_arc_when_unchanged() {
    let dense: Arc<[u64]> = Arc::from([0b1011_u64, 0b0101]);
    let token_set = Arc::new(RangeSetBlaze::from_iter([0_u32..=127]));
    let precomputed = precomputed_for(&token_set, Arc::from([!0_u64, !0_u64]));

    let mut cache = FxHashMap::default();
    let intersected = DenseMaskAcc::intersect_dense_with_token_set_cached(
        0,
        &dense,
        &token_set,
        &precomputed,
        &mut cache,
    )
    .unwrap();

    assert!(Arc::ptr_eq(&intersected, &dense));
}

#[test]
fn precomputed_dense_intersection_allocates_when_pruned() {
    let dense: Arc<[u64]> = Arc::from([0b1011_u64, 0b0101]);
    let token_set = Arc::new(RangeSetBlaze::from_iter([0_u32..=127]));
    let precomputed = precomputed_for(&token_set, Arc::from([0b0011_u64, 0b0000]));

    let mut cache = FxHashMap::default();
    let intersected = DenseMaskAcc::intersect_dense_with_token_set_cached(
        0,
        &dense,
        &token_set,
        &precomputed,
        &mut cache,
    )
    .unwrap();

    assert!(!Arc::ptr_eq(&intersected, &dense));
    assert_eq!(&*intersected, &[0b0011_u64, 0b0000]);
}

#[test]
fn small_intersection_cache_reuses_exact_result() {
    let dense: Arc<[u64]> = Arc::from([0b1011_u64, 0b0101]);
    let token_set = Arc::new(RangeSetBlaze::from_iter([0_u32..=127]));
    let precomputed = precomputed_for(&token_set, Arc::from([0b0011_u64, 0b0100]));
    let mut cache = DenseTokenSetIntersectionSmallCache::new();

    let first = DenseMaskAcc::intersect_dense_with_token_set_small_cached(
        &dense,
        &token_set,
        &precomputed,
        &mut cache,
    )
    .unwrap();
    let second = DenseMaskAcc::intersect_dense_with_token_set_small_cached(
        &dense,
        &token_set,
        &precomputed,
        &mut cache,
    )
    .unwrap();

    assert_eq!(&*first, &[0b0011_u64, 0b0100]);
    assert!(Arc::ptr_eq(&first, &second));
    assert_eq!(cache.len(), 1);
    assert!(Arc::ptr_eq(&cache[0].0, &dense));
}

#[test]
fn empty_possible_matches_uses_exact_seed_exclusion_scan() {
    let mut constraint = Constraint::from_glrm_grammar(
        r#"
                start start;
                t A ::= "a";
                t B ::= "b";
                nt start ::= A | B;
            "#,
        &Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"b".to_vec()),
                (2, b"ab".to_vec()),
            ]),
    )
    .expect("test constraint should compile");
    let terminal_a = constraint
        .terminal_display_names
        .iter()
        .position(|name| name == "A")
        .expect("A terminal should have a display name") as u32;
    constraint.possible_matches.clear();
    constraint.possible_matches_complete = false;

    let tokenizer_state = constraint.tokenizer.initial_state();
    let disallowed = TerminalsDisallowed::new().with_insert(tokenizer_state, terminal_a);
    let mut state = constraint.start_dynamic();
    state.state = crate::runtime::state::ParserStateMap::singleton(
        tokenizer_state,
        ParserGSS::from_stacks(&[(vec![0u32], disallowed)]),
    );

    let mut expected = vec![0u32; constraint.body_mask_len()];
    state.fill_mask_dynamic(&mut expected);
    let mut actual = vec![0u32; constraint.body_mask_len()];
    state.fill_mask(&mut actual);
    assert_eq!(actual, expected);

    let loaded = Constraint::load(&constraint.save()).expect("empty-PM constraint should roundtrip");
    assert!(loaded.possible_matches.is_empty());
    let mut loaded_state = loaded.start_dynamic();
    let loaded_tokenizer_state = loaded.tokenizer.initial_state();
    let loaded_disallowed =
        TerminalsDisallowed::new().with_insert(loaded_tokenizer_state, terminal_a);
    loaded_state.state = crate::runtime::state::ParserStateMap::singleton(
        loaded_tokenizer_state,
        ParserGSS::from_stacks(&[(vec![0u32], loaded_disallowed)]),
    );
    let mut loaded_expected = vec![0u32; loaded.body_mask_len()];
    loaded_state.fill_mask_dynamic(&mut loaded_expected);
    let mut loaded_actual = vec![0u32; loaded.body_mask_len()];
    loaded_state.fill_mask(&mut loaded_actual);
    assert_eq!(loaded_actual, loaded_expected);
}

#[test]
fn literal_choice_terminal_mask_keeps_multibyte_prefix_tokens() {
    let vocab = Vocab::new(vec![
        (0, b"\"".to_vec()),
        (1, b"S".to_vec()),
        (2, b"Se".to_vec()),
        (3, b"Service".to_vec()),
        (4, b"I".to_vec()),
        (5, b"In".to_vec()),
        (6, b"Independent".to_vec()),
        (7, b" provider assertion\"".to_vec()),
        (8, b" validation of assertion\"".to_vec()),
    ]);
    let constraint = Constraint::from_glrm_grammar(
        r#"
                start start;
                fa assurance_body ::= {
                    start 0;
                    accept 1;
                    0 -- "\"" ("Service provider assertion\"" | "Independent validation of assertion\"") --> 1;
                };
                nt start ::= assurance_body;
            "#,
        &vocab,
    )
    .expect("literal-choice terminal should compile");

    let mut state = constraint.start();
    state.commit_token(0).expect("opening quote should commit");

    let mut static_mask = vec![0u32; constraint.body_mask_len()];
    state.fill_mask(&mut static_mask);
    let mut dynamic_mask = vec![0u32; constraint.body_mask_len()];
    state.fill_mask_dynamic(&mut dynamic_mask);

    assert_eq!(static_mask, dynamic_mask);
    for token in [1, 2, 3, 4, 5, 6] {
        assert!(
            mask_contains(&static_mask, token),
            "literal prefix token {token} should be accepted"
        );
    }
}

#[test]
fn direct_mask_spills_past_the_inline_path_capacity() {
    let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);
    let constraint = Constraint::from_glrm_grammar(
        r#"
                start start;
                t A ::= "a";
                nt start ::= A;
            "#,
        &vocab,
    )
    .expect("single-terminal grammar should compile");
    let mut state = constraint.start();
    let (tokenizer_state, parser_gss) = state.state.entries[0].clone();
    state.state.entries.clear();
    for _ in 0..=MASK_SINGLE_PATH_DIRECT_INLINE_PATH_CAPACITY {
        state
            .state
            .insert_flat_alternative(tokenizer_state, parser_gss.clone());
    }
    assert_eq!(
        state.state.len(),
        MASK_SINGLE_PATH_DIRECT_INLINE_PATH_CAPACITY + 1,
    );

    let mut direct = vec![0u32; constraint.body_mask_len()];
    assert!(state.try_fill_mask_single_path_direct(&mut direct));
    let mut dynamic = vec![0u32; constraint.body_mask_len()];
    state.fill_mask_dynamic(&mut dynamic);
    assert_eq!(direct, dynamic);
    assert!(mask_contains(&direct, 0));
    assert!(!mask_contains(&direct, 1));
}

#[test]
fn direct_mask_declines_wide_shared_gss_without_partial_output() {
    let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);
    let constraint = Constraint::from_glrm_grammar(
        r#"start start; t A ::= "a"; nt start ::= A;"#,
        &vocab,
    )
    .expect("routing-test grammar should compile");
    let mut state = constraint.start();
    let tokenizer_state = state.state.entries[0].0;
    // Deliberately synthetic parser labels: admission must decline BEFORE
    // looking up any of them or evaluating an incomplete subset of paths.
    // Both the one-pass and the many-lexer two-pass admission paths matter.
    let wide = ParserGSS::from_stacks(
        &(0..=MASK_SINGLE_PATH_DIRECT_MAX_PATHS_PER_GSS)
            .map(|path| (vec![0, 10_000 + path as u32, 7], TerminalsDisallowed::default()))
            .collect::<Vec<_>>(),
    );
    for lexer_branches in [1, MASK_SINGLE_PATH_DIRECT_INLINE_PATH_CAPACITY] {
        state.state.entries.clear();
        for _ in 0..lexer_branches {
            state.state.insert_flat_alternative(tokenizer_state, wide.clone());
        }
        let mut mask = vec![0x5a5a_5a5a; constraint.body_mask_len()];
        let before = mask.clone();
        assert!(!state.try_fill_mask_single_path_direct(&mut mask));
        assert_eq!(mask, before, "declining admission must not publish a partial mask");
    }
}

#[test]
fn stack_plan_admission_depends_on_reuse_not_the_old_path_boundary() {
    for path_count in [31, 32, 33, 64] {
        assert!(!single_path_direct_plan_reuse_dominates(
            path_count,
            path_count * 8,
            0,
        ));
    }
    assert!(!single_path_direct_plan_reuse_dominates(2, 16, 8));
    assert!(!single_path_direct_plan_reuse_dominates(3, 24, 16));
    assert!(!single_path_direct_plan_reuse_dominates(9, 72, 56));
    assert!(single_path_direct_plan_reuse_dominates(9, 144, 112));
}

#[test]
fn single_path_direct_stack_work_budget_accepts_shallow_ambiguity() {
    assert_eq!(single_path_direct_stack_work([8; 10]), Some(80));
    assert_eq!(
        single_path_direct_stack_work([MASK_SINGLE_PATH_DIRECT_MAX_TOTAL_STACK_VALUES]),
        Some(MASK_SINGLE_PATH_DIRECT_MAX_TOTAL_STACK_VALUES),
    );
    assert_eq!(
        single_path_direct_stack_work([
            MASK_SINGLE_PATH_DIRECT_MAX_TOTAL_STACK_VALUES,
            1,
        ]),
        None,
    );
}

#[test]
fn indexed_dag_mask_matches_dynamic_on_all_small_reachable_states() {
    use std::collections::BTreeSet;
    fn allowed(mask: &[u32], token: u32) -> bool {
        mask.get(token as usize / 32)
            .is_some_and(|word| word & (1u32 << (token % 32)) != 0)
    }

    let vocab = Vocab::new(
        ["a", "b", "ab", "ba", "aa", "bb"]
            .into_iter()
            .enumerate()
            .map(|(id, bytes)| (id as u32, bytes.as_bytes().to_vec()))
            .collect(),
    );
    let grammars = [
        r#"
                start start;
                t A ::= "a" | "ab";
                t B ::= "a" | "ba";
                nt item ::= A | B;
                nt start ::= item item? item?;
            "#,
        r#"
                start start;
                t A ::= "a"+;
                t B ::= "a"+ "b"?;
                nt start ::= A A | B B | A B | B A;
            "#,
    ];

    let mut ambiguous_states = 0usize;
    for grammar in grammars {
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab)
            .expect("small indexed-DAG parity grammar should compile");
        let mut frontier = vec![(constraint.start(), Vec::<u32>::new())];
        let mut seen = BTreeSet::new();
        for depth in 0..=3 {
            let mut next = Vec::new();
            for (state, path) in frontier {
                let key = state.debug_parser_stacks();
                if !seen.insert(format!("{key:?}")) {
                    continue;
                }
                let mut expected = vec![0u32; constraint.body_mask_len()];
                state.fill_mask_dynamic(&mut expected);
                if state.has_parser_ambiguity() {
                    ambiguous_states += 1;
                    let mut actual = vec![0u32; constraint.body_mask_len()];
                    assert!(state.fill_mask_indexed_dag(&mut actual, true));
                    assert_eq!(
                        actual, expected,
                        "indexed/dynamic mask mismatch depth={depth} path={path:?} grammar={grammar}"
                    );
                }
                if depth == 3 {
                    continue;
                }
                for (token, bytes) in constraint.token_bytes_iter() {
                    if !allowed(&expected, token) {
                        continue;
                    }
                    let mut advanced = state.clone();
                    advanced
                        .commit_bytes(bytes)
                        .expect("dynamically admitted token must commit");
                    let mut next_path = path.clone();
                    next_path.push(token);
                    next.push((advanced, next_path));
                }
            }
            frontier = next;
        }
    }
    assert!(ambiguous_states > 0, "test must exercise indexed ambiguous states");
}

#[test]
fn indexed_dag_cache_stays_exact_across_commits_and_restore() {
    let vocab = Vocab::new(
        ["a", "b", "ab", "ba", "aa", "bb"]
            .into_iter()
            .enumerate()
            .map(|(id, bytes)| (id as u32, bytes.as_bytes().to_vec()))
            .collect(),
    );
    let constraint = Constraint::from_glrm_grammar(
        r#"
                start start;
                t A ::= "a"+;
                t B ::= "a"+ "b"?;
                nt item ::= A | B;
                nt start ::= item item? item? item?;
            "#,
        &vocab,
    )
    .expect("persistent indexed-DAG parity grammar should compile");
    let mut state = constraint.start();
    let sequence: [&[u8]; 4] = [b"a", b"a", b"a", b"b"];
    let mut checkpoint = None;

    for (index, bytes) in sequence.into_iter().enumerate() {
        let mut expected = vec![0u32; constraint.body_mask_len()];
        state.fill_mask_dynamic(&mut expected);
        if state.has_parser_ambiguity() {
            let mut first = vec![0u32; constraint.body_mask_len()];
            let mut second = vec![0u32; constraint.body_mask_len()];
            assert!(state.fill_mask_indexed_dag(&mut first, true));
            assert!(state.fill_mask_indexed_dag(&mut second, true));
            assert_eq!(first, expected);
            assert_eq!(second, expected, "same-state cache hit changed the mask");
        }
        state
            .commit_bytes(bytes)
            .expect("test sequence should remain valid");
        if index == 1 {
            checkpoint = Some(state.clone());
        }
    }

    state = checkpoint.expect("checkpoint should be captured");
    let mut expected = vec![0u32; constraint.body_mask_len()];
    state.fill_mask_dynamic(&mut expected);
    if state.has_parser_ambiguity() {
        let mut actual = vec![0u32; constraint.body_mask_len()];
        assert!(state.fill_mask_indexed_dag(&mut actual, true));
        assert_eq!(actual, expected, "restored indexed mask diverged");
    }
}
