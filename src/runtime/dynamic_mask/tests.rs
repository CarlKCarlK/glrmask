use super::*;
use crate::{DynamicConstraint, Constraint as Constraint, Grammar, Vocab};
use std::collections::BTreeSet;

fn token_allowed(mask: &[u32], token_id: u32) -> bool {
    let word = token_id as usize / 32;
    let bit = token_id % 32;
    mask.get(word).is_some_and(|word| word & (1u32 << bit) != 0)
}

fn direct_mask(state: &ConstraintState<'_>) -> Vec<u32> {
    let mut mask = vec![0u32; state.constraint.body_mask_len()];
    state.fill_mask_dynamic(&mut mask);
    mask
}

/// Check the independent oracle, natural probation/store path, and a
/// genuine cached hit. A hit must not run the vocabulary walker again.
fn assert_recursive_persistent_roundtrip(state: &ConstraintState<'_>) {
    let mut expected = vec![0; state.constraint.mask_len()];
    state.fill_recursive_mask_by_exact_full_walk(&mut expected);
    let mut output = vec![u32::MAX; expected.len() + 3];
    for _ in 0..2 {
        assert!(try_fill_recursive_mask_shared(state, &mut output).unwrap());
        assert_eq!(&output[..expected.len()], expected.as_slice());
        assert!(output[expected.len()..].iter().all(|&word| word == 0));
    }
    assert!(dynamic_mask_state_has_cached_result(state));
    let walks = TEST_FULL_WALK_USES.with(|count| count.get());
    output.fill(u32::MAX);
    assert!(try_fill_recursive_mask_shared(state, &mut output).unwrap());
    assert_eq!(TEST_FULL_WALK_USES.with(|count| count.get()), walks,
        "cached recursive mask must not walk the vocabulary");
    assert_eq!(&output[..expected.len()], expected.as_slice());
    assert!(output[expected.len()..].iter().all(|&word| word == 0));
}

#[test]
fn recursive_full_walk_transition_provider_matches_exact_leaf_execution() {
    let vocab = Vocab::new(vec![
        (0, b"X".to_vec()),
        (1, b"[".to_vec()),
        (2, b"a".to_vec()),
        (3, b"b".to_vec()),
        (4, b"c".to_vec()),
        (5, b" ".to_vec()),
        (6, b"]".to_vec()),
        (7, b"!".to_vec()),
        (8, b"abc".to_vec()),
        (9, b" a".to_vec()),
        (10, b"a]!".to_vec()),
    ]);
    let leaf = Constraint::compile(
        Grammar::glrm(
            r#"glrm 1; start leaf; ignore WS; t WS = " "+; t WORD = /[a-c]+/; nt leaf = WORD;"#,
        ),
        &vocab,
    )
    .unwrap();
    let middle = Constraint::compile(
        Grammar::glrm(
            r#"glrm 1; start middle; extern grammar leaf; nt middle = "[" leaf "]";"#,
        ),
        &vocab,
    )
    .unwrap()
    .bind_grammar_dynamic_boundary("leaf", leaf)
    .unwrap();
    let bound = Constraint::compile(
        Grammar::glrm(
            r#"glrm 1; start outer; extern grammar middle; nt outer = "X" middle "!";"#,
        ),
        &vocab,
    )
    .unwrap()
    .bind_grammar_dynamic_boundary("middle", middle)
    .unwrap();
    assert!(bound.uses_compact_segmented_parser_runtime());

    let mut provider = RecursiveFullWalkTransitions::new(&bound)
        .expect("finite recursive fixture must support shared lexer provider");
    let layout = bound.recursive_parser_layout().unwrap().unwrap();
    let mut scratch =
        crate::runtime::commit::tokenizer_scan::ReusableTokenizerExecScratch::default();
    for leaf_index in 0..layout.leaves.len() {
        let leaf = bound.recursive_leaf_constraint(leaf_index).unwrap();
        for local_state in 0..leaf.tokenizer.num_states() {
            let scoped = bound
                .recursive_tokenizer_scoped_state(leaf_index, local_state)
                .unwrap();
            assert_eq!(provider.root_state(scoped).unwrap(), scoped);
            for byte in u8::MIN..=u8::MAX {
                let cell = provider.cell(scoped, byte);
                assert!(
                    crate::runtime::commit::tokenizer_scan::execute_recursive_tokenizer_reusable(
                        &bound,
                        &[byte],
                        scoped,
                        &mut scratch,
                    ),
                    "bounded exact tokenizer execution unexpectedly declined",
                );
                if scratch.states.is_empty() {
                    assert!(
                        RecursiveFullWalkTransitions::cell_is_dead(cell),
                        "leaf={leaf_index} local={local_state} byte={byte}",
                    );
                    continue;
                }
                assert_eq!(
                    scratch.states.as_slice(),
                    &[RecursiveFullWalkTransitions::cell_target(cell)],
                    "leaf={leaf_index} local={local_state} byte={byte}",
                );
                assert_eq!(
                    RecursiveFullWalkTransitions::cell_has_finalizer(cell),
                    !scratch.matches.is_empty(),
                    "leaf={leaf_index} local={local_state} byte={byte}",
                );
                let target = RecursiveFullWalkTransitions::cell_target(cell);
                let mut actual = provider.matched_terminals(target);
                actual.sort_unstable();
                actual.dedup();
                let mut expected = scratch
                    .matches
                    .iter()
                    .map(|matched| matched.id)
                    .collect::<SmallVec<[u32; 4]>>();
                expected.sort_unstable();
                expected.dedup();
                assert_eq!(
                    actual, expected,
                    "finalizers differ leaf={leaf_index} local={local_state} byte={byte}",
                );
            }
        }
    }
}

#[test]
fn recursive_shared_full_walk_matches_existing_exact_walker() {
    // Exercise the byte-walk at every capacity, rather than satisfying
    // later iterations from a persistent result produced by the first.
    struct RestoreCacheCapacity(usize, bool);
    impl Drop for RestoreCacheCapacity {
        fn drop(&mut self) {
            TEST_RECURSIVE_PRODUCT_CACHE_CAPACITY.with(|v| v.set(self.0));
            TEST_RECURSIVE_PERSISTENT_MASK_CACHE.with(|v| v.set(self.1));
        }
    }
    let _restore = RestoreCacheCapacity(
        TEST_RECURSIVE_PRODUCT_CACHE_CAPACITY.with(|v| v.get()),
        TEST_RECURSIVE_PERSISTENT_MASK_CACHE.with(|v| v.replace(false)),
    );
    let mut words = vec![Vec::<u8>::new()];
    let mut layer = vec![Vec::<u8>::new()];
    for _ in 0..3 {
        let mut next = Vec::new();
        for prefix in layer {
            for &byte in b"abcX[]! ()" {
                let mut word = prefix.clone();
                word.push(byte);
                next.push(word);
            }
        }
        words.extend(next.iter().cloned());
        layer = next;
    }
    let vocab = Vocab::new(
        words
            .into_iter()
            .enumerate()
            .map(|(id, bytes)| (id as u32, bytes))
            .collect(),
    );
    let middle_source =
        r#"glrm 1; start middle; extern grammar leaf; nt middle = "[" leaf "]";"#;
    let outer_source =
        r#"glrm 1; start outer; extern grammar middle; nt outer = "X" middle "!";"#;

    for (leaf_source, prefixes) in [
        (
            r#"glrm 1; start leaf; t WORD = /[a-c]{1,4}/; nt leaf = WORD;"#,
            vec![
                b"".as_slice(),
                b"X",
                b"X[",
                b"X[a",
                b"X[abc",
                b"X[abc]",
                b"X[abc]!",
            ],
        ),
        (
            r#"glrm 1; start leaf; ignore WS; t WS = " "+; t WORD = /[a-c]{1,4}/; nt leaf = WORD;"#,
            vec![
                b"".as_slice(),
                b"X",
                b"X[",
                b"X[ ",
                b"X[ a",
                b"X[ a ]",
                b"X[ a ]!",
            ],
        ),
        (
            r#"glrm 1; start leaf; nt leaf = ("a")?;"#,
            vec![
                b"".as_slice(),
                b"X",
                b"X[",
                b"X[a",
                b"X[a]",
                b"X[a]!",
                b"X[]",
                b"X[]!",
            ],
        ),
        (
            r#"glrm 1; start leaf; t SHORT = "a"; t LONG = /ab?/; nt leaf = LONG | SHORT "b";"#,
            vec![b"".as_slice(), b"X[", b"X[a", b"X[ab", b"X[ab]", b"X[ab]!"],
        ),
        (
            r#"glrm 1; start leaf; nt leaf = "(" leaf ")" | "a";"#,
            vec![b"".as_slice(), b"X[", b"X[(", b"X[((a", b"X[((a)", b"X[((a))]", b"X[((a))]!"],
        ),
    ] {
        let leaf = Constraint::compile(Grammar::glrm(leaf_source), &vocab).unwrap();
        let middle = Constraint::compile(Grammar::glrm(middle_source), &vocab)
            .unwrap()
            .bind_grammar_dynamic_boundary("leaf", leaf)
            .unwrap();
        let bound = Constraint::compile(Grammar::glrm(outer_source), &vocab)
            .unwrap()
            .bind_grammar_dynamic_boundary("middle", middle)
            .unwrap();
        assert!(bound.uses_compact_segmented_parser_runtime());
        for constraint in [
            bound.clone(),
            Constraint::load(bound.save()).expect("reload recursive fixture"),
        ] {
            for prefix in &prefixes {
                let mut state = constraint.start_dynamic();
                state.commit_bytes(prefix).unwrap();
                let mut reference = vec![0u32; constraint.mask_len()];
                state.fill_recursive_mask_by_exact_full_walk(&mut reference);
                let mut shared = vec![0u32; constraint.mask_len()];
                for capacity in [0, 1, 2, 256, 2048] {
                    TEST_RECURSIVE_PRODUCT_CACHE_CAPACITY.with(|v| v.set(capacity));
                    assert!(
                        try_fill_recursive_mask_shared(&state, &mut shared).unwrap(),
                        "finite recursive fixture unexpectedly declined"
                    );
                    assert_eq!(
                        shared,
                        reference,
                        "shared/full recursive mask mismatch capacity={capacity} leaf={leaf_source} prefix={:?}",
                        String::from_utf8_lossy(prefix),
                    );
                    // Force the ordinary config executor even when the
                    // finite direct adapter is eligible. This checks
                    // scoped namespace/guard/reset transport independently
                    // of the fast finite representation.
                    assert!(recursive_provider::fill(&state, &mut shared).unwrap());
                    assert_eq!(shared, reference,
                        "config recursive mismatch capacity={capacity} leaf={leaf_source} prefix={prefix:?}");
                }
            }
        }
    }
}

#[test]
fn recursive_parser_semantic_interning_preserves_exact_languages_and_exhaustion() {
    let (mut cache, _) = FullWalkParserCache::from_roots(&DynamicBranches::new(), None);
    let mut canonicalizer = RecursiveParserCanonicalizer::default();
    let left = ParserStacks::from_single_stack(vec![0, 1, 2], ());
    let left_again = ParserStacks::from_single_stack(vec![0, 1, 2], ());
    assert!(!left.ptr_eq(&left_again));
    let a = canonicalizer.intern_with_threshold(left.clone(), &mut cache, 0);
    let b = canonicalizer.intern_with_threshold(left_again.clone(), &mut cache, 0);
    assert_eq!(a, b, "different allocations of one language must share a node");
    let right = ParserStacks::from_single_stack(vec![0, 1, 3], ());
    let c = canonicalizer.intern_with_threshold(right.clone(), &mut cache, 0);
    assert_ne!(a, c, "equal tops/prefixes are not an equality proof");
    let union_a = left.merge(&right);
    let union_b = right.merge(&left_again);
    let u = canonicalizer.intern_with_threshold(union_a, &mut cache, 0);
    let v = canonicalizer.intern_with_threshold(union_b, &mut cache, 0);
    assert_eq!(u, v, "union construction order must not change the language key");
    assert_ne!(u, a);
    canonicalizer.semantic = Some(Box::new(RecursiveParserSemanticCache {
        keys: crate::ds::leveled_gss::GssSemanticKeyInterner::with_budget(1, 1, 1),
        nodes: FxHashMap::default(),
    }));
    let x = canonicalizer.intern_with_threshold(
        ParserStacks::from_single_stack(vec![0, 4, 5], ()), &mut cache, 0,
    );
    let y = canonicalizer.intern_with_threshold(
        ParserStacks::from_single_stack(vec![0, 4, 6], ()), &mut cache, 0,
    );
    assert!(canonicalizer.semantic.as_ref().unwrap().keys.is_exhausted());
    assert_ne!(x, y, "exhaustion must not collapse distinct languages into sentinel zero");
}

#[test]
fn recursive_persistent_mask_cache_uses_exact_scoped_state_key() {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            TEST_RECURSIVE_PERSISTENT_MASK_CACHE.with(|enabled| enabled.set(self.0));
        }
    }
    let _restore =
        Restore(TEST_RECURSIVE_PERSISTENT_MASK_CACHE.with(|enabled| enabled.replace(true)));
    let vocab = Vocab::new(vec![
        (0, b"X".to_vec()),
        (1, b"[".to_vec()),
        (2, b"a".to_vec()),
        (3, b"b".to_vec()),
        (4, b"]".to_vec()),
        (5, b"!".to_vec()),
        (6, b"ab".to_vec()),
        (7, b"a]!".to_vec()),
    ]);
    let leaf = Constraint::compile(
        Grammar::glrm(r#"glrm 1; start leaf; t WORD = /[ab]{1,4}/; nt leaf = WORD;"#),
        &vocab,
    )
    .unwrap();
    let middle = Constraint::compile(
        Grammar::glrm(
            r#"glrm 1; start middle; extern grammar leaf; nt middle = "[" leaf "]";"#,
        ),
        &vocab,
    )
    .unwrap()
    .bind_grammar_dynamic_boundary("leaf", leaf)
    .unwrap();
    let bound = Constraint::compile(
        Grammar::glrm(
            r#"glrm 1; start outer; extern grammar middle; nt outer = "X" middle "!";"#,
        ),
        &vocab,
    )
    .unwrap()
    .bind_grammar_dynamic_boundary("middle", middle)
    .unwrap();
    let mut state = bound.start_dynamic();
    state.commit_bytes(b"X[a").unwrap();
    let (hash, query) = dynamic_mask_lookup_query(&state).expect("recursive key");
    let key = query.to_owned_state_key();
    assert!(matches!(
        key.first().map(|entry| entry.0),
        Some(DynamicMaskLexerStateKey::RecursiveExact(_))
    ));
    let mut ordinary_coordinate = key.clone();
    if let Some(entry) = ordinary_coordinate.first_mut()
        && let DynamicMaskLexerStateKey::RecursiveExact(id) = entry.0
    {
        entry.0 = DynamicMaskLexerStateKey::Exact(id);
    }
    assert_eq!(crate::runtime::artifact::dynamic_mask_state_key_hash(&key), hash);
    assert!(!query.matches_state(&ordinary_coordinate),
        "ordinary and scoped recursive coordinates must be distinguished even on hash collision");

    assert_recursive_persistent_roundtrip(&state);
    let loaded = Constraint::load(bound.save()).unwrap();
    let mut loaded_state = loaded.start_dynamic();
    loaded_state.commit_bytes(b"X[a").unwrap();
    assert_recursive_persistent_roundtrip(&loaded_state);

    // A colliding hash is only a bucket hint; exact equality remains
    // mandatory before a cached payload can be returned.
    let dyn_vocab = bound.dynamic_mask_vocab_for_runtime();
    let mut actual = vec![0; bound.mask_len()];
    assert!(!dyn_vocab.copy_cached_mask_with_predicate(
        hash, |candidate| candidate == &ordinary_coordinate, &mut actual));

    // Query-only synthetic states test key guards without asking the
    // runtime to execute artificial parser states or exclusions.
    let mut excluded = state.clone();
    for (&lexer, gss) in state.state.iter() {
        excluded.state.insert(lexer, gss.apply(|_| TerminalsDisallowed::new()
            .try_with_insert_inline(lexer, 0).unwrap()));
    }
    let (_, excluded_query) = dynamic_mask_lookup_query(&excluded).unwrap();
    assert!(!excluded_query.matches_state(&key), "exclusions are part of the exact key");

    let (&lexer, _) = state.state.iter().next().unwrap();
    let mut too_deep = state.clone();
    too_deep.state.clear();
    too_deep.state.insert(lexer, ParserGSS::from_single_stack(
        vec![0; DYNAMIC_MASK_CACHE_MAX_DEPTH as usize + 1], TerminalsDisallowed::new()));
    assert!(dynamic_mask_lookup_query(&too_deep).is_none(),
        "a depth-limited key must decline rather than cache a truncated stack");

    state.commit_bytes(b"b").unwrap();
    let (next_hash, next_query) =
        dynamic_mask_lookup_query(&state).expect("next recursive key");
    assert!(
        next_hash != hash || !next_query.matches_state(&query.to_owned_state_key()),
        "a changed exact recursive state must not alias the cached predecessor",
    );
}

#[test]
fn recursive_persistent_cache_isolated_between_bindings() {
    let vocab = Vocab::new(vec![
        (0, b"X".to_vec()), (1, b"a".to_vec()), (2, b"b".to_vec()),
        (3, b"!".to_vec()), (4, b"a!".to_vec()), (5, b"b!".to_vec()),
    ]);
    let parent = Constraint::compile(Grammar::glrm(
        r#"glrm 1; start root; extern grammar child; nt root = "X" child "!";"#), &vocab).unwrap();
    let a = Constraint::compile(Grammar::glrm(
        r#"glrm 1; start leaf; nt leaf = "a";"#), &vocab).unwrap();
    let b = Constraint::compile(Grammar::glrm(
        r#"glrm 1; start leaf; nt leaf = "b";"#), &vocab).unwrap();
    let bound_a = parent.bind_grammar_dynamic_boundary("child", a).unwrap();
    let bound_b = parent.bind_grammar_dynamic_boundary("child", b).unwrap();
    let mut state_a = bound_a.start_dynamic();
    let mut state_b = bound_b.start_dynamic();
    state_a.commit_bytes(b"X").unwrap();
    state_b.commit_bytes(b"X").unwrap();
    for _ in 0..3 {
        assert_recursive_persistent_roundtrip(&state_a);
        assert_recursive_persistent_roundtrip(&state_b);
    }
    let mut mask_a = vec![0; bound_a.mask_len()];
    let mut mask_b = vec![0; bound_b.mask_len()];
    try_fill_recursive_mask_shared(&state_a, &mut mask_a).unwrap();
    try_fill_recursive_mask_shared(&state_b, &mut mask_b).unwrap();
    assert_ne!(mask_a, mask_b);
    assert!(token_allowed(&mask_a, 1) && !token_allowed(&mask_a, 2));
    assert!(token_allowed(&mask_b, 2) && !token_allowed(&mask_b, 1));
}

#[test]
fn recursive_shared_config_covers_virtual_and_epsilon_leaves() {
    let words: Vec<Vec<u8>> = vec![
        b"".to_vec(), b"X".to_vec(), b"!".to_vec(), b"\"".to_vec(),
        b"x:".to_vec(), b"a".to_vec(), b"b".to_vec(), b"ab".to_vec(),
        b"aa".to_vec(), b"X\"x:".to_vec(), b"a\"!".to_vec(),
        b"\"!".to_vec(), b"Xab!".to_vec(), b" ".to_vec(),
        b"\\u00".to_vec(), b"\\\"".to_vec(), vec![0xc2], vec![0xc2, 0xa0],
    ];
    let vocab = Vocab::new(words.into_iter().enumerate().map(|(i,b)| (i as u32,b)).collect());
    let virtual_child = Constraint::from_json_schema(
        r#"{"type":"string","format":"uri","minLength":1,"maxLength":5000}"#, &vocab,
    ).unwrap();
    assert!(virtual_child.tokenizer.has_virtual_residual_runtime());
    let epsilon_child = Constraint::compile(Grammar::glrm(r#"
            start payload;
            lexer group a ::= A;
            lexer group b ::= B;
            t A ::= "a"+;
            t B ::= "ab"+;
            nt payload ::= A | B;
        "#), &vocab).unwrap();
    let parent = Constraint::compile(Grammar::glrm(
        r#"glrm 1; start root; extern grammar payload; nt root = "X" payload "!";"#,
    ), &vocab).unwrap();
    for (child, prefixes) in [
        (virtual_child, vec!["", "X", "X\"", "X\"x:", "X\"x:a", "X\"x:a\"", "X\"x:a\"!"]),
        (epsilon_child, vec!["", "X", "Xa", "Xab", "Xaa", "Xab!"]),
    ] {
        let composed = parent.bind_grammar_dynamic_boundary("payload", child).unwrap();
        let loaded = Constraint::load(composed.save()).unwrap();
        for constraint in [&composed, &loaded] {
            for prefix in &prefixes {
                let mut state = constraint.start();
                state.commit_bytes(prefix.as_bytes()).unwrap();
                let mut expected = vec![0;constraint.mask_len()];
                state.fill_recursive_mask_by_exact_full_walk(&mut expected);
                let mut actual = vec![0;expected.len()];
                struct CapacityGuard(usize);
                impl Drop for CapacityGuard {
                    fn drop(&mut self) {
                        TEST_RECURSIVE_PRODUCT_CACHE_CAPACITY.with(|v| v.set(self.0));
                    }
                }
                let _capacity = CapacityGuard(TEST_RECURSIVE_PRODUCT_CACHE_CAPACITY.with(|v| v.get()));
                for capacity in [0, 1, 2, 16, 256, 2048] {
                    TEST_RECURSIVE_PRODUCT_CACHE_CAPACITY.with(|v| v.set(capacity));
                    assert!(recursive_provider::fill(&state,&mut actual).unwrap());
                    assert_eq!(actual, expected, "general recursive prefix {prefix:?} capacity={capacity}");
                }
                assert!(try_fill_recursive_mask_shared(&state,&mut actual).unwrap());
                assert_eq!(actual, expected, "production shared prefix {prefix:?}");
                assert_recursive_persistent_roundtrip(&state);
            }
        }
    }
}

#[test]
fn dynamic_mask_cache_key_uses_exact_mask_projection_coordinate() {
    let vocab = Vocab::new(vec![
        (0, b"\"".to_vec()),
        (1, b"a".to_vec()),
        (2, b"aa".to_vec()),
        (3, b"b".to_vec()),
    ]);
    let dynamic = DynamicConstraint::from_json_schema(
        r#"{"type":"string","maxLength":1000000000}"#,
        &vocab,
    )
    .unwrap();
    let mut state = dynamic.inner.start();
    state.commit_token(0).unwrap();
    state.commit_token(1).unwrap();
    let exact_before = state.state.clone();
    let key_before = dynamic_mask_state_key(&state).expect("cache key before commit");
    let mask_before = direct_mask(&state);

    state.commit_token(1).unwrap();
    assert_ne!(exact_before, state.state, "test must change the exact lexer coordinate");
    let key_after = dynamic_mask_state_key(&state).expect("cache key after commit");
    assert_eq!(
        key_before, key_after,
        "one-token-equivalent projected lexer states must share the persistent dynamic-mask cache key",
    );
    assert_eq!(mask_before, direct_mask(&state));
}

#[test]
fn dynamic_mask_lookup_query_exact_match_with_state_key() {
    let vocab = Vocab::new(vec![
        (0, b"\"".to_vec()),
        (1, b"a".to_vec()),
        (2, b"aa".to_vec()),
        (3, b"b".to_vec()),
    ]);
    let dynamic = DynamicConstraint::from_json_schema(
        r#"{"type":"string","maxLength":1000000000}"#,
        &vocab,
    )
    .unwrap();
    let mut state = dynamic.inner.start();
    state.commit_token(0).unwrap();
    for step in [1, 2, 3, 1, 2] {
        let _ = state.commit_token(step % 4);
        let (query_hash, query) = dynamic_mask_lookup_query(&state).expect("query");
        let owned_key = dynamic_mask_state_key(&state).expect("state key");
        let owned_hash = crate::runtime::artifact::dynamic_mask_state_key_hash(&owned_key);
        assert_eq!(query_hash, owned_hash, "query hash must match owned state key hash");
        assert!(query.matches_state(&owned_key), "query must match owned state key");
        assert_eq!(query.to_owned_state_key(), owned_key, "to_owned_state_key must match");

        // Negative match test: modified owned key should not match
        let mut perturbed_key = owned_key.clone();
        if let Some(entry) = perturbed_key.first_mut() {
            entry.0 = match entry.0 {
                DynamicMaskLexerStateKey::Exact(id) => DynamicMaskLexerStateKey::Exact(id + 9999),
                DynamicMaskLexerStateKey::RecursiveExact(id) => {
                    DynamicMaskLexerStateKey::RecursiveExact(id + 9999)
                }
                DynamicMaskLexerStateKey::MaskProjection { state, initial } => {
                    DynamicMaskLexerStateKey::MaskProjection { state: state + 9999, initial }
                }
                DynamicMaskLexerStateKey::TerminalObservation { class, terminal, initial } => {
                    DynamicMaskLexerStateKey::TerminalObservation { class: class + 9999, terminal, initial }
                }
                DynamicMaskLexerStateKey::VirtualDenseProjection { runtime, state, initial } => {
                    DynamicMaskLexerStateKey::VirtualDenseProjection {
                        runtime,
                        state: state + 9999,
                        initial,
                    }
                }
            };
            assert!(!query.matches_state(&perturbed_key), "perturbed lexer key must not match");
        }
    }
}

#[test]
fn dynamic_mask_lookup_query_branching_and_exclusions() {
    let vocab = Vocab::new(vec![
        (0, b"{".to_vec()),
        (1, b"}".to_vec()),
        (2, b"\"a\"".to_vec()),
        (3, b":".to_vec()),
        (4, b"1".to_vec()),
        (5, b",".to_vec()),
        (6, b"\"b\"".to_vec()),
        (7, b"true".to_vec()),
    ]);
    let schema = r#"{"anyOf":[{"type":"object","properties":{"a":{"type":"integer"}}},{"type":"object","properties":{"b":{"type":"boolean"}}}]}"#;
    let dynamic = DynamicConstraint::from_json_schema(schema, &vocab).unwrap();
    let mut state = dynamic.inner.start();
    state.commit_token(0).unwrap(); // "{"
    let (query_hash, query) = dynamic_mask_lookup_query(&state).expect("query");
    let owned_key = dynamic_mask_state_key(&state).expect("state key");
    let owned_hash = crate::runtime::artifact::dynamic_mask_state_key_hash(&owned_key);
    assert_eq!(query_hash, owned_hash, "query hash must match owned state key hash");
    assert!(query.matches_state(&owned_key));
    assert_eq!(query.to_owned_state_key(), owned_key);

    // Perturb path to verify exact matching on multi-path / complex states
    let mut perturbed = owned_key.clone();
    if let Some((_, paths)) = perturbed.first_mut() {
        if let Some(first_path) = paths.first_mut() {
            first_path.0.push(9999);
        }
    }
    assert!(!query.matches_state(&perturbed));
}

#[test]
fn dynamic_mask_lookup_query_microbench() {
    let vocab = Vocab::new(vec![
        (0, b"\"".to_vec()),
        (1, b"hello".to_vec()),
        (2, b"world".to_vec()),
        (3, b"\"".to_vec()),
    ]);
    let dynamic = DynamicConstraint::from_json_schema(
        r#"{"type":"string"}"#,
        &vocab,
    ).unwrap();
    let mut state = dynamic.inner.start();
    state.commit_token(0).unwrap();
    state.commit_token(1).unwrap();

    let iters = 20_000;

    // 1. Transient query construction
    let start = std::time::Instant::now();
    let mut sink_hash = 0u64;
    for _ in 0..iters {
        if let Some((hash, _)) = dynamic_mask_lookup_query(&state) {
            sink_hash ^= hash;
        }
    }
    let transient_ns = start.elapsed().as_nanos() as f64 / iters as f64;

    // 2. Materialized owned state key
    let start = std::time::Instant::now();
    let mut sink_len = 0usize;
    for _ in 0..iters {
        if let Some(key) = dynamic_mask_state_key(&state) {
            sink_len += key.len();
        }
    }
    let owned_ns = start.elapsed().as_nanos() as f64 / iters as f64;

    // 3. Cache lookup hit with transient predicate
    let (hash, query) = dynamic_mask_lookup_query(&state).unwrap();
    let owned_key = query.to_owned_state_key();
    let dyn_vocab = state.constraint.dynamic_mask_vocab_for_runtime();
    let mut mask_buf = vec![0u32; state.constraint.body_mask_len()];
    dyn_vocab.cache_mask(owned_key, hash, &mask_buf, false);

    let start = std::time::Instant::now();
    let mut hits = 0;
    for _ in 0..iters {
        if dyn_vocab.copy_cached_mask_with_predicate(hash, |cand| query.matches_state(cand), &mut mask_buf) {
            hits += 1;
        }
    }
    let cache_hit_ns = start.elapsed().as_nanos() as f64 / iters as f64;

    eprintln!(
        "\n[MICROBENCH] transient_query_ns={:.1}ns ({:.3}us) | owned_key_ns={:.1}ns ({:.3}us) | speedup={:.2}x | cache_hit_ns={:.1}ns ({:.3}us) | hits={}/{} (sink={}/{})\n",
        transient_ns,
        transient_ns / 1000.0,
        owned_ns,
        owned_ns / 1000.0,
        owned_ns / transient_ns,
        cache_hit_ns,
        cache_hit_ns / 1000.0,
        hits,
        iters,
        sink_hash,
        sink_len,
    );
    assert_eq!(hits, iters);
}

fn assert_dynamic_parity(state: &ConstraintState<'_>) {
    assert_eq!(state.mask(), direct_mask(state));
}

fn assert_dynamic_parity_on_reachable_states(
    constraint: &Constraint,
    max_depth: usize,
    context: &str,
) {
    let mut frontier = vec![(constraint.start(), Vec::<u32>::new())];
    let mut seen = BTreeSet::new();

    for depth in 0..=max_depth {
        let mut next = Vec::new();
        for (state, path) in frontier {
            if let Some(key) = dynamic_mask_state_key(&state)
                && !seen.insert(key)
            {
                continue;
            }

            let static_mask = state.mask();
            let dynamic_mask = direct_mask(&state);
            assert_eq!(
                static_mask, dynamic_mask,
                "dynamic/static mask mismatch: {context} depth={depth} path={path:?}"
            );
            if depth == max_depth {
                continue;
            }

            for (token_id, bytes) in constraint.token_bytes_iter() {
                let expected = token_allowed(&static_mask, token_id);
                let mut advanced = state.clone();
                let accepted = advanced.commit_bytes(bytes).is_ok();
                assert_eq!(
                    accepted, expected,
                    "static mask/commit mismatch during dynamic sweep: {context} depth={depth} path={path:?} token={token_id}"
                );
                if accepted {
                    let mut next_path = path.clone();
                    next_path.push(token_id);
                    next.push((advanced, next_path));
                }
            }
        }
        frontier = next;
    }
}

#[test]
fn root_admission_memo_is_exact_and_owner_frontier_scoped() {
    let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);
    let constraint = Constraint::compile(Grammar::glrm(
        r#"glrm 1; start s; t A = "a"; t B = "b"; nt s = A B;"#,
    ), &vocab).unwrap();
    let other = Constraint::compile(Grammar::glrm(
        r#"glrm 1; start s; t A = "a"; t B = "b"; nt s = B A;"#,
    ), &vocab).unwrap();
    let stacks = ParserStacks::from_single_stack(vec![0], ());
    let shared = Some(Arc::new(RootAdmissionMemo {
        owner: &constraint as *const Constraint as usize,
        gss: stacks.clone(), admitted: OnceLock::new(),
    }));
    assert!(shared.as_ref().unwrap().admitted.get().is_none());
    let expected = exact_parser_admission_for_stacks(&constraint, &stacks);
    assert_eq!(parser_admission_with_root_memo(&constraint, &stacks, &shared), expected);
    assert_eq!(parser_admission_with_root_memo(&constraint, &stacks.clone(), &shared), expected);
    assert_eq!(shared.as_ref().unwrap().admitted.get(), Some(&expected));

    let terminal = constraint.terminal_display_names.iter()
        .position(|name| name == "A").unwrap() as TerminalID;
    let child = parser_child(&constraint, &stacks, terminal).unwrap();
    assert!(!child.ptr_eq(&stacks));
    assert_eq!(parser_admission_with_root_memo(&constraint, &child, &shared),
               exact_parser_admission_for_stacks(&constraint, &child));
    assert_eq!(parser_admission_with_root_memo(&other, &stacks, &shared),
               exact_parser_admission_for_stacks(&other, &stacks));
    assert_eq!(shared.as_ref().unwrap().admitted.get(), Some(&expected),
               "foreign/frontier fallback must not overwrite retained root proof");
}

#[test]
fn full_walk_future_cache_normalizes_hot_and_generic_coordinates() {
    let vocab = Vocab::new(vec![
        (0, b"a".to_vec()), (1, b"bc".to_vec()), (2, b"xyz".to_vec()),
    ]);
    let constraint = Constraint::compile(Grammar::glrm(
        r#"glrm 1; start s; t WORD = "abc"; t OTHER = "xyz"; nt s = WORD | OTHER;"#,
    ), &vocab).unwrap();
    let word = constraint.terminal_display_names.iter().position(|name| name == "WORD").unwrap() as u32;
    let other = constraint.terminal_display_names.iter().position(|name| name == "OTHER").unwrap() as u32;
    let mut scan = DynamicNfaScanCache::new(&constraint, None);
    let raw = scan.transition(constraint.tokenizer.initial_state(), b'a');
    assert_ne!(raw, u32::MAX);
    // Hot scalar IDs are used by the ordinary lazy-config executor, not
    // deterministic identity-config execution. Match that representation
    // explicitly, as the missing-subset ordinary runtime path does.
    scan.deterministic = false;
    let mut table = FullWalkConfigTransitions::new(&mut scan, 3, 0);
    table.hot_enabled = true;
    table.profile = true;
    let id = table.hot_scalar.intern(raw).unwrap();
    let hot = FullWalkHotScalarCache::tagged(id);
    let generic = table.generic_config_for_state(hot);
    assert_ne!(hot, generic, "fixture must exercise two coordinate representations");
    assert!(table.future_contains(hot, word));
    assert_eq!(table.future_contains_calls, 1);
    assert!(table.future_contains(hot, word));
    assert!(table.future_contains(generic, word));
    assert_eq!(table.future_contains_calls, 1, "both aliases must reuse the same positive result");
    assert!(!table.future_contains(hot, other));
    assert!(!table.future_contains(hot, other));
    assert!(!table.future_contains(generic, other));
    assert_eq!(table.future_contains_calls, 2, "negative results must be memoized too");
    assert_eq!(table.future_contains_cache.len(), 2);
}

#[test]
fn deterministic_dynamic_scan_does_not_index_identity_configs() {
    let vocab = Vocab::new(vec![
        (0, b"a".to_vec()),
        (1, b"b".to_vec()),
        (2, b"ab".to_vec()),
    ]);
    let grammar = r#"
start start;
t A ::= 'a'+;
t B ::= 'b';
nt start ::= A B | A;
"#;
    let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();
    let mut cache = DynamicNfaScanCache::new(&constraint, None);
    assert!(cache.raw_start_config.is_empty());

    let initial = constraint.tokenizer.initial_state();
    let config = cache.config_for_raw_start(initial).unwrap();
    assert_eq!(config, initial);
    assert!(cache.raw_start_config.is_empty());
    let _ = cache.step_config(config, b'a').unwrap();
    assert!(cache.raw_start_config.is_empty());
}

#[test]
fn strict_full_walk_executes_native_epsilon_nfa_configs() {
    let vocab = Vocab::new(vec![
        (0, b"a".to_vec()),
        (1, b"b".to_vec()),
        (2, b"aa".to_vec()),
        (3, b"ab".to_vec()),
        (4, b"ba".to_vec()),
        (5, b"bb".to_vec()),
    ]);
    // Only the parser table is used from this ordinary constraint. The
    // dynamic constraint below substitutes a deliberately retained
    // epsilon-NFA tokenizer with the same two terminal IDs.
    let parser_source = Constraint::from_glrm_grammar(
        r#"
start start;
t A ::= 'a';
t B ::= 'b';
nt start ::= A | B | A A | A B | B A | B B;
"#,
        &vocab,
    )
    .unwrap();
    let tokenizer =
        crate::automata::lexer::tokenizer::arbitrary_epsilon_l1_test_tokenizer();
    assert!(tokenizer.has_epsilon_transitions());
    let mut dynamic = DynamicConstraint::from_parts(
        parser_source.table.clone(),
        parser_source.terminal_display_names.clone(),
        tokenizer,
        None,
        None,
        Vec::new(),
        &vocab,
    );
    assert!(dynamic.inner.tokenizer.has_epsilon_transitions());
    // Production may prepare a dense deterministic execution coordinate
    // when it fits. Disable that representation here so this regression
    // exercises the native epsilon-NFA backend itself.
    dynamic
        .inner
        .dynamic_mask_vocab
        .disable_prepared_mask_execution_for_test();
    assert!(
        dynamic.inner.dynamic_mask_vocab.mask_projection_tokenizer().is_none(),
        "test setup should retain the source epsilon-NFA coordinate",
    );
    assert!(
        dynamic.inner.dynamic_mask_vocab.mask_projection_fast_transitions().is_none(),
        "epsilon-NFA strict walk must not depend on a dense DFA table",
    );

    TEST_FULL_WALK_USES.with(|count| count.set(0));
    TEST_CONFIG_FULL_WALK_USES.with(|count| count.set(0));
    let state = dynamic.inner.start();
    let mask = state.mask();
    TEST_FULL_WALK_USES.with(|count| assert!(count.get() > 0));
    TEST_CONFIG_FULL_WALK_USES.with(|count| assert!(count.get() > 0));

    for (token_id, _) in dynamic.inner.token_bytes_iter() {
        let expected = token_allowed(&mask, token_id);
        let mut advanced = state.clone();
        let accepted = advanced.commit_token(token_id).is_ok();
        assert_eq!(accepted, expected, "token={token_id}");
    }
}

#[test]
fn additive_and_candidate_modes_use_complete_strict_full_walk() {
    let vocab = Vocab::new(vec![
        (0, b"a".to_vec()),
        (1, b"b".to_vec()),
        (2, b"ab".to_vec()),
        (3, b"ba".to_vec()),
        (4, b"aa".to_vec()),
    ]);
    let dynamic = DynamicConstraint::from_glrm_grammar(
        r#"
start start;
t A ::= 'a'+;
t B ::= 'b';
nt start ::= A B | B A;
"#,
        &vocab,
    )
    .unwrap();
    let state = dynamic.inner.start();
    let mut exact = vec![0u32; dynamic.inner.body_mask_len()];
    fill_mask_dynamic(&state, &mut exact);

    let mut additive = vec![0u32; exact.len()];
    set_mask_bit(&mut additive, 4);
    let mut expected_additive = additive.clone();
    for (dst, &word) in expected_additive.iter_mut().zip(&exact) {
        *dst |= word;
    }
    TEST_FULL_WALK_USES.with(|count| count.set(0));
    or_mask_dynamic_additions(&state, &mut additive);
    TEST_FULL_WALK_USES.with(|count| assert!(count.get() > 0));
    assert_eq!(additive, expected_additive);

    let mut candidates = vec![0u32; exact.len()];
    set_mask_bit(&mut candidates, 0);
    set_mask_bit(&mut candidates, 3);
    let mut filtered = vec![0u32; exact.len()];
    set_mask_bit(&mut filtered, 4);
    let mut expected_filtered = filtered.clone();
    for ((dst, &word), &candidate) in expected_filtered
        .iter_mut()
        .zip(&exact)
        .zip(&candidates)
    {
        *dst |= word & candidate;
    }
    TEST_FULL_WALK_USES.with(|count| count.set(0));
    or_mask_dynamic_candidate_additions(&state, &mut filtered, &candidates);
    TEST_FULL_WALK_USES.with(|count| assert!(count.get() > 0));
    assert_eq!(filtered, expected_filtered);
}

#[test]
fn strict_full_walk_spills_for_vocab_depth_beyond_255_edges() {
    let mut entries = Vec::new();
    for depth in 1..=300u32 {
        let mut token = vec![b'a'; depth as usize];
        token.push(b'b');
        entries.push((depth - 1, token));
    }
    entries.push((300, vec![b'a'; 301]));
    let vocab = Vocab::new(entries);
    let dynamic = DynamicConstraint::from_glrm_grammar(
        r#"
start start;
t A ::= /a+b?/;
nt start ::= A;
"#,
        &vocab,
    )
    .unwrap();
    assert!(
        dynamic
            .inner
            .dynamic_mask_vocab_for_runtime()
            .trie
            .full_walk_max_parent_depth()
            >= 255,
        "test vocabulary did not create the intended deep radix tree",
    );

    TEST_FULL_WALK_USES.with(|count| count.set(0));
    let state = dynamic.start();
    let mask = state.mask();
    TEST_FULL_WALK_USES.with(|count| assert!(count.get() > 0));
    for (token_id, _) in dynamic.inner.token_bytes_iter() {
        let expected = token_allowed(&mask, token_id);
        let mut advanced = state.clone();
        assert_eq!(advanced.commit_token(token_id).is_ok(), expected, "token={token_id}");
    }
}

#[test]
fn dynamic_mask_matches_normal_for_repeat_and_cross_terminal_tokens() {
    let vocab = Vocab::new(
        vec![
            (0, b"a".to_vec()),
            (1, b"aa".to_vec()),
            (2, b"b".to_vec()),
            (3, b"ab".to_vec()),
            (4, b"aab".to_vec()),
            (5, b"aaa".to_vec()),
        ]);
    let grammar = r#"
start start;
t A ::= 'a'+;
t B ::= 'b';
nt start ::= A B | A;
"#;
    let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();

    let mut state = constraint.start();
    assert_dynamic_parity(&state);
    assert!(token_allowed(&state.mask(), 3));

    state.commit_token(1).unwrap();
    assert_dynamic_parity(&state);
    assert!(token_allowed(&state.mask(), 2));

    state.commit_token(2).unwrap();
    assert!(state.is_accepting());
    assert_dynamic_parity(&state);
}

#[test]
fn dynamic_mask_trie_is_rebuilt_after_load() {
    let vocab = Vocab::new(
        vec![(0, b"a".to_vec()), (1, b"b".to_vec()), (2, b"ab".to_vec())]);
    let grammar = r#"
start start;
t A ::= 'a';
t B ::= 'b';
nt start ::= A B;
"#;
    let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();
    let loaded = Constraint::load(&constraint.save()).unwrap();
    assert_dynamic_parity(&loaded.start());
}

#[test]
fn dynamic_mask_keeps_duplicate_byte_token_aliases() {
    let vocab = Vocab::new(
        vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (7, b"a".to_vec()),
            (12, b"ab".to_vec()),
        ]);
    let grammar = r#"
start start;
t A ::= 'a';
t B ::= 'b';
nt start ::= A B;
"#;
    let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();

    let mut state = constraint.start();
    assert_dynamic_parity(&state);
    let mask = direct_mask(&state);
    assert!(token_allowed(&mask, 0));
    assert!(token_allowed(&mask, 7));
    assert!(token_allowed(&mask, 12));

    state.commit_token(7).unwrap();
    assert_dynamic_parity(&state);
    assert!(token_allowed(&direct_mask(&state), 1));
}

#[test]
fn dynamic_mask_matches_normal_across_an_ignore_terminal() {
    let vocab = Vocab::new(
        vec![
            (0, b"a".to_vec()),
            (1, b"aa".to_vec()),
            (2, b"b".to_vec()),
            (3, b" b".to_vec()),
            (4, b"  b".to_vec()),
        ]);
    let grammar = r#"
start start;
ignore WS;
t WS ::= ' '+;
t A ::= 'a'+;
t B ::= 'b';
nt start ::= A B;
"#;
    let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();

    let mut state = constraint.start();
    assert_dynamic_parity(&state);
    state.commit_token(1).unwrap();
    assert_dynamic_parity(&state);
    assert!(token_allowed(&state.mask(), 3));

    state.commit_token(3).unwrap();
    assert!(state.is_accepting());
    assert_dynamic_parity(&state);
}

#[test]
fn dynamic_mask_preserves_repeated_terminal_after_ignore_reset_inside_token() {
    let vocab = Vocab::new(
        vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"c".to_vec()),
            (3, b"aa".to_vec()),
            (4, b"bb".to_vec()),
            (5, b"cc".to_vec()),
            (6, b"ab".to_vec()),
            (7, b"ac".to_vec()),
            (8, b"ba".to_vec()),
            (9, b"bc".to_vec()),
            (10, b"abc".to_vec()),
            (11, b"aab".to_vec()),
            (12, b"abb".to_vec()),
            (13, b"acc".to_vec()),
            (14, b" ".to_vec()),
            (15, b"  ".to_vec()),
            (16, b" a".to_vec()),
            (17, b"a ".to_vec()),
            (18, b" a ".to_vec()),
            (19, b"ab c".to_vec()),
        ]);
    let grammar = r#"
start start;
ignore WS;
lexer group ws ::= WS;
lexer group a ::= A;
lexer group b ::= B;
lexer group c ::= C;
t WS ::= " "+;
t A ::= "a"+;
t B ::= "b";
t C ::= "c";
nt item ::= A | B | C;
nt start ::= item item? item?;
"#;
    let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();
    let mut state = constraint.start();
    state.commit_token(0).unwrap();
    state.commit_token(16).unwrap();

    assert_dynamic_parity(&state);
    assert!(token_allowed(&direct_mask(&state), 0));
    assert!(token_allowed(&direct_mask(&state), 3));
    assert!(token_allowed(&direct_mask(&state), 17));
}

#[test]
fn masks_preserve_overlap_continuation_after_ignore_reset() {
    let vocab = Vocab::new(
        vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"ab".to_vec()),
            (3, b" ".to_vec()),
            (4, b" a".to_vec()),
        ]);
    let grammar = r#"
start start;
ignore WS;
t WS ::= " "+;
t A ::= "ab";
t B ::= "a" | "ab";
nt item ::= A | B;
nt start ::= item item? item?;
"#;
    let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();
    let mut state = constraint.start();
    state.commit_token(0).unwrap();
    state.commit_token(4).unwrap();

    let static_mask = state.mask();
    let dynamic_mask = direct_mask(&state);

    let mut probe = state.clone();
    assert!(probe.commit_bytes(b"b").is_ok());
    assert!(
        token_allowed(&static_mask, 1),
        "static mask must admit b because a-ab = B WS A"
    );
    assert!(
        token_allowed(&dynamic_mask, 1),
        "dynamic mask must admit b because a-ab = B WS A"
    );
}

#[test]
fn dynamic_mask_generated_small_language_sweep() {
    const WORDS: [&str; 4] = ["a", "b", "ab", "ba"];
    let vocab = Vocab::new(
        [
            "a", "b", "ab", "ba", " ", " a", "a ", " b", "b ", " a ", " b ",
        ]
        .into_iter()
        .enumerate()
        .map(|(id, word)| (id as u32, word.as_bytes().to_vec()))
        .collect());
    let languages = (1u32..1u32 << WORDS.len())
        .filter(|mask| mask.count_ones() <= 2)
        .collect::<Vec<_>>();
    let rule = |name: &str, mask: u32| {
        let rhs = WORDS
            .iter()
            .enumerate()
            .filter_map(|(index, word)| {
                (mask & (1 << index) != 0).then(|| format!("\"{word}\""))
            })
            .collect::<Vec<_>>()
            .join(" | ");
        format!("t {name} ::= {rhs};\n")
    };

    for grouped in [false, true] {
        for ignored in [false, true] {
            let grouping = if grouped {
                if ignored {
                    "lexer group ws ::= WS;\nlexer group a ::= A;\nlexer group b ::= B;\n"
                } else {
                    "lexer group a ::= A;\nlexer group b ::= B;\n"
                }
            } else {
                ""
            };
            let ignore = if ignored {
                "ignore WS;\nt WS ::= \" \"+;\n"
            } else {
                ""
            };

            for &a in &languages {
                for &b in &languages {
                    if grouped && a == b {
                        continue;
                    }
                    for start_rule in [
                        "nt item ::= A | B;\nnt start ::= item item? item?;",
                        "nt start ::= A A | B B;",
                        "nt start ::= A B | B A;",
                    ] {
                        let grammar = format!(
                            "start start;\n{ignore}{grouping}{}{}{start_rule}\n",
                            rule("A", a),
                            rule("B", b),
                        );
                        let constraint =
                            Constraint::from_glrm_grammar(&grammar, &vocab).unwrap();
                        let context = format!(
                            "finite grouped={grouped} ignored={ignored} A={a:#06b} B={b:#06b}\ngrammar:\n{grammar}"
                        );
                        assert_dynamic_parity_on_reachable_states(&constraint, 3, &context);
                    }
                }
            }

            let grammar = format!(
                "start start;\n{ignore}{grouping}t A ::= \"a\"+;\nt B ::= \"b\"+;\nnt item ::= A | B;\nnt start ::= item item? item?;\n"
            );
            let constraint = Constraint::from_glrm_grammar(&grammar, &vocab).unwrap();
            let context = format!(
                "repeat grouped={grouped} ignored={ignored}\ngrammar:\n{grammar}"
            );
            assert_dynamic_parity_on_reachable_states(&constraint, 4, &context);

            let grammar = format!(
                "start start;\n{ignore}{grouping}t A ::= \"a\"+ \"b\";\nt B ::= \"a\"+;\nnt item ::= A | B;\nnt start ::= item item? item?;\n"
            );
            let constraint = Constraint::from_glrm_grammar(&grammar, &vocab).unwrap();
            let context = format!(
                "delayed-overlap grouped={grouped} ignored={ignored}\ngrammar:\n{grammar}"
            );
            assert_dynamic_parity_on_reachable_states(&constraint, 4, &context);
        }
    }
}

#[test]
fn dynamic_mask_matches_normal_at_every_reachable_small_state() {
    let vocab = Vocab::new(
        vec![
            (0, b"a".to_vec()),
            (1, b"aa".to_vec()),
            (2, b"b".to_vec()),
            (3, b"bb".to_vec()),
            (4, b"c".to_vec()),
            (5, b"ab".to_vec()),
            (6, b"ba".to_vec()),
            (7, b"a c".to_vec()),
            (8, b"b c".to_vec()),
            (9, b" aa".to_vec()),
            (10, b" bb".to_vec()),
        ]);
    let grammar = r#"
start start;
ignore WS;
t WS ::= ' '+;
t A ::= 'a'+;
t B ::= 'b'+;
t C ::= 'c';
nt start ::= A B C | B A C | A C | B C;
"#;
    let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();

    fn visit(state: ConstraintState<'_>, depth: usize) {
        assert_dynamic_parity(&state);
        if depth == 3 {
            return;
        }
        let mask = state.mask();
        for token_id in 0..11u32 {
            if !token_allowed(&mask, token_id) {
                continue;
            }
            let mut next = state.clone();
            next.commit_token(token_id).unwrap();
            visit(next, depth + 1);
        }
    }

    visit(constraint.start(), 0);
}

#[test]
fn dynamic_mask_matches_normal_when_one_repeated_terminal_crosses_tokens() {
    let vocab = Vocab::new(
        vec![
            (0, b"a".to_vec()),
            (1, b"aa".to_vec()),
            (2, b"aaa".to_vec()),
            (3, b"aaaa".to_vec()),
        ]);
    let grammar = r#"
start start;
t A ::= 'a'+;
nt start ::= A A;
"#;
    let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();

    fn visit(state: ConstraintState<'_>, depth: usize) {
        assert_dynamic_parity(&state);
        if depth == 3 {
            return;
        }
        let mask = state.mask();
        for token_id in 0..4u32 {
            if !token_allowed(&mask, token_id) {
                continue;
            }
            let mut next = state.clone();
            next.commit_token(token_id).unwrap();
            visit(next, depth + 1);
        }
    }

    visit(constraint.start(), 0);
}

#[test]
fn dynamic_mask_matches_normal_for_a_partial_json_string() {
    let vocab = Vocab::new(
        vec![
            (0, b"\"".to_vec()),
            (1, b"a".to_vec()),
            (2, b"b".to_vec()),
            (3, b"\\\"".to_vec()),
            (4, b"\"a".to_vec()),
            (5, b"a\"".to_vec()),
        ]);
    let constraint =
        Constraint::from_json_schema(r#"{"type":"string"}"#, &vocab).unwrap();

    let mut state = constraint.start();
    assert_dynamic_parity(&state);
    state.commit_token(0).unwrap();
    assert_dynamic_parity(&state);
    state.commit_token(1).unwrap();
    assert_dynamic_parity(&state);
    state.commit_token(0).unwrap();
    assert!(state.is_accepting());
    assert_dynamic_parity(&state);
}

#[test]
fn dynamic_mask_matches_certified_long_terminal_run() {
    let vocab = Vocab::new(
        vec![
            (0, b"++++++++a".to_vec()),
            (1, b"++++".to_vec()),
            (2, b"a".to_vec()),
        ]);
    let grammar = r#"
start start;
t U ::= '+';
nt start ::= U* 'a';
"#;
    let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();
    let mut state = constraint.start();

    assert_dynamic_parity(&state);
    assert!(token_allowed(&state.mask(), 0));
    assert!(token_allowed(&state.mask(), 1));

    state.commit_token(1).unwrap();
    assert_dynamic_parity(&state);
    assert!(token_allowed(&state.mask(), 2));
    state.commit_token(2).unwrap();
}

#[test]
fn dynamic_mask_handles_monolithic_json_number() {
    let vocab = Vocab::new(
        vec![
            (0, b"-".to_vec()),
            (1, b"0".to_vec()),
            (2, b"1".to_vec()),
            (3, b"2".to_vec()),
            (4, b"3".to_vec()),
            (5, b".".to_vec()),
            (6, b"e".to_vec()),
            (7, b"+".to_vec()),
        ]);
    let constraint =
        Constraint::from_json_schema(r#"{"type":"number"}"#, &vocab).unwrap();

    let mut state = constraint.start();
    assert_dynamic_parity(&state);
    for bytes in [b"1".as_slice(), b".".as_slice(), b"2".as_slice(), b"e".as_slice(), b"-".as_slice(), b"3".as_slice()] {
        state.commit_bytes(bytes).unwrap();
        assert_dynamic_parity(&state);
    }
    assert!(state.is_accepting());
}

#[test]
fn dynamic_mask_keeps_other_gss_paths_when_one_path_excludes_a_terminal() {
    let vocab = Vocab::new(
        vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"c".to_vec()),
            (3, b"ab".to_vec()),
        ]);
    let grammar = r#"
start start;
t A ::= 'a' | 'ab';
t B ::= 'a';
t C ::= 'c';
t D ::= 'b';
nt start ::= A C | B D;
"#;
    let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();
    let mut state = constraint.start();
    state.commit_token(0).unwrap();

    let paths = state
        .state
        .values()
        .flat_map(|gss| gss.to_stacks(4_096).expect("stack enumeration exceeded explicit limit"))
        .collect::<Vec<_>>();
    assert!(paths.iter().any(|(_, exclusions)| exclusions.is_empty()));
    assert!(paths.iter().any(|(_, exclusions)| !exclusions.is_empty()));

    assert_dynamic_parity(&state);
    assert!(token_allowed(&direct_mask(&state), 1));
    state.commit_token(1).unwrap();
    assert!(state.is_accepting());
    assert_dynamic_parity(&state);
}


#[test]
fn dynamic_full_walk_accepts_long_compressed_vocab_edge() {
    let vocab = Vocab::new(vec![
        (0, vec![b'a'; 300]),
        (1, b"a".to_vec()),
        (2, b"b".to_vec()),
    ]);
    let grammar = r#"
start start;
t A ::= 'a'+;
nt start ::= A;
"#;
    let dynamic = DynamicConstraint::from_glrm_grammar(grammar, &vocab).unwrap();
    let mask_vocab = dynamic.inner.dynamic_mask_vocab_for_runtime();
    assert!(mask_vocab.trie.subtree_max_total_byte_len(0) >= 300);
    assert!(
        mask_vocab.trie.full_walk_max_parent_depth() < 255,
        "a long compressed edge must not consume one DFS stack slot per byte",
    );
    assert!(mask_vocab.mask_projection_fast_transitions().is_some());

    let mask = dynamic.start().mask();
    assert!(token_allowed(&mask, 0));
    assert!(token_allowed(&mask, 1));
    assert!(!token_allowed(&mask, 2));
}

#[test]
fn dynamic_virtual_unit_repeat_uses_static_mask_projection_end_to_end() {
    let vocab = Vocab::new(vec![
        (0, b"a".to_vec()),
        (1, b"aa".to_vec()),
        (2, b"aaa".to_vec()),
        (3, b"aaaa".to_vec()),
        (4, b"aaaaa".to_vec()),
        (5, b"aaaaaa".to_vec()),
        (6, b"b".to_vec()),
    ]);

    let billion_grammar = r#"
start start;
t A ::= /a{0,1000000000}/;
nt start ::= A;
"#;
    let billion = DynamicConstraint::from_glrm_grammar(billion_grammar, &vocab).unwrap();
    assert_eq!(
        billion.inner.tokenizer.num_states(),
        1,
        "the exact billion-bound lexer must remain an arithmetic runtime state",
    );
    assert!(
        billion.inner.dynamic_mask_vocab.mask_projection_tokenizer().is_some(),
        "virtual mask projection is immutable build-derived runtime data and must be prepared before the first mask",
    );
    let mask_tokenizer = billion
        .inner
        .dynamic_mask_vocab
        .mask_projection_tokenizer()
        .expect("build finalization must prepare the finite virtual mask projection");
    let billion_mask = billion.inner.start().mask();
    assert!(
        billion.inner.lazy_dynamic_mask_vocab.get().is_none(),
        "first mask must not materialize hidden projection state",
    );
    assert_eq!(
        mask_tokenizer.num_states(),
        vocab.max_token_byte_len() as u32 + 3,
        "mask lexer size must depend on vocabulary horizon, not the repeat bound",
    );
    for token in 0..=5 {
        assert!(token_allowed(&billion_mask, token), "a-only token {token} was rejected");
    }
    assert!(!token_allowed(&billion_mask, 6));

    // For a small bound, compare the virtual dynamic implementation with
    // the ordinary materialized static implementation through the exact
    // upper-bound transition. This checks both re-projection after commit
    // and rejection of a vocabulary token which crosses the bound.
    let boundary_grammar = r#"
start start;
t A ::= /a{0,5}/;
nt start ::= A;
"#;
    let dynamic = DynamicConstraint::from_glrm_grammar(boundary_grammar, &vocab).unwrap();
    let ordinary = Constraint::from_glrm_grammar(boundary_grammar, &vocab).unwrap();
    let mut dynamic_state = dynamic.inner.start();
    let mut ordinary_state = ordinary.start();

    assert_eq!(dynamic_state.mask(), ordinary_state.mask());
    assert!(!token_allowed(&dynamic_state.mask(), 5));
    dynamic_state.commit_token(2).unwrap(); // consume three a's
    ordinary_state.commit_token(2).unwrap();
    assert_eq!(dynamic_state.mask(), ordinary_state.mask());
    assert!(token_allowed(&dynamic_state.mask(), 0));
    assert!(token_allowed(&dynamic_state.mask(), 1));
    assert!(!token_allowed(&dynamic_state.mask(), 2));

    dynamic_state.commit_token(1).unwrap(); // reach the exact upper bound
    ordinary_state.commit_token(1).unwrap();
    assert_eq!(dynamic_state.mask(), ordinary_state.mask());
    assert!(dynamic_state.is_accepting());
    assert!(!token_allowed(&dynamic_state.mask(), 0));
    assert!(dynamic_state.clone().commit_token(0).is_err());
}

#[test]
fn dynamic_hybrid_virtual_repeat_coexists_with_static_terminals() {
    let vocab = Vocab::new(vec![
        (0, b"b".to_vec()),
        (1, b"a".to_vec()),
        (2, b"aa".to_vec()),
        (3, b"aaa".to_vec()),
        (4, b"baaa".to_vec()),
        (5, b"baaaaa".to_vec()),
        (6, b"baaaaaa".to_vec()),
        (7, b"x".to_vec()),
    ]);
    let billion_grammar = r#"
start start;
t A ::= /a{0,1000000000}/;
t B ::= 'b';
nt start ::= B A;
"#;
    let hybrid = DynamicConstraint::from_glrm_grammar(billion_grammar, &vocab).unwrap();
    assert!(
        hybrid
            .inner
            .tokenizer
            .virtual_zero_min_unit_repeat_mask_tokenizer(vocab.max_token_byte_len())
            .is_some(),
        "the pathological terminal should be the arithmetic component",
    );
    assert!(
        hybrid.inner.tokenizer.num_states() < 64,
        "physical hybrid lexer unexpectedly scales with the billion bound",
    );
    let start_mask = hybrid.start().mask();
    assert!(token_allowed(&start_mask, 4));
    assert!(token_allowed(&start_mask, 5));
    assert!(token_allowed(&start_mask, 6));
    assert!(!token_allowed(&start_mask, 7));

    let small_grammar = r#"
start start;
t A ::= /a{0,5}/;
t B ::= 'b';
nt start ::= B A;
"#;
    // The hybrid threshold intentionally leaves this small grammar on the
    // ordinary path; it is therefore an independent materialized oracle.
    let dynamic_small = DynamicConstraint::from_glrm_grammar(small_grammar, &vocab).unwrap();
    let ordinary_small = Constraint::from_glrm_grammar(small_grammar, &vocab).unwrap();
    let mut dynamic_state = dynamic_small.start();
    let mut ordinary_state = ordinary_small.start();
    assert_eq!(dynamic_state.mask(), ordinary_state.mask());
    assert!(token_allowed(&dynamic_state.mask(), 5));
    assert!(!token_allowed(&dynamic_state.mask(), 6));

    dynamic_state.commit_token(0).unwrap();
    ordinary_state.commit_token(0).unwrap();
    assert_eq!(dynamic_state.mask(), ordinary_state.mask());
    dynamic_state.commit_token(3).unwrap();
    ordinary_state.commit_token(3).unwrap();
    assert_eq!(dynamic_state.mask(), ordinary_state.mask());
}

#[test]
fn static_and_dynamic_direct_regular_compilation_preserve_backend_contract() {
    let vocab = Vocab::new(vec![
        (0, b"a".to_vec()),
        (1, b"ab".to_vec()),
        (2, b"b".to_vec()),
        (3, b"x".to_vec()),
        (4, b"bx".to_vec()),
    ]);
    let mut grammar = String::from(
        "start start;
t A ::= 'a' | 'ab';
t X ::= 'x';
nt start ::= A r0;
",
    );
    for index in 0..39 {
        grammar.push_str(&format!("nt r{index} ::= X r{};
", index + 1));
    }
    grammar.push_str("nt r39 ::= X;
");

    let constraint = Constraint::from_glrm_grammar(&grammar, &vocab).unwrap();
    assert!(!constraint.uses_dynamic_runtime());
    assert!(constraint.possible_matches_complete);
    assert!(!constraint.possible_matches.is_empty());

    let dynamic = DynamicConstraint::from_glrm_grammar(&grammar, &vocab).unwrap();
    assert!(dynamic.inner.uses_dynamic_runtime());
    assert!(!dynamic.inner.possible_matches_complete);

    fn delayed_query(constraint: &Constraint) -> (u32, TerminalID) {
        let execution = constraint.tokenizer.execute_from_state_all_widths(
            b"a",
            constraint.tokenizer.initial_state(),
        );
        execution
            .matches
            .iter()
            .find_map(|matched| {
                constraint
                    .tokenizer
                    .possible_future_terminals(matched.end_state)
                    .contains(matched.id as usize)
                    .then_some((matched.end_state, matched.id))
            })
            .expect("A=a|ab must create one delayed-terminal query state")
    }

    let (compiled_state, compiled_terminal) = delayed_query(&constraint);
    let (fallback_state, fallback_terminal) = delayed_query(&dynamic.inner);
    assert_eq!(compiled_terminal, fallback_terminal);

    let mut direct_table_tokens = Vec::new();
    assert!(constraint.visit_possible_match_original_tokens(
        compiled_state,
        compiled_terminal,
        |token| direct_table_tokens.push(token),
    ));
    direct_table_tokens.sort_unstable();
    direct_table_tokens.dedup();

    let mut helper_table_tokens = Vec::new();
    for_each_token_matching_terminal_from_state(
        &constraint,
        compiled_state,
        compiled_terminal,
        |token| helper_table_tokens.push(token),
    )
    .unwrap();
    helper_table_tokens.sort_unstable();
    helper_table_tokens.dedup();

    let mut fallback_tokens = Vec::new();
    for_each_token_matching_terminal_from_state(
        &dynamic.inner,
        fallback_state,
        fallback_terminal,
        |token| fallback_tokens.push(token),
    )
    .unwrap();
    fallback_tokens.sort_unstable();
    fallback_tokens.dedup();

    assert_eq!(helper_table_tokens, direct_table_tokens);
    assert_eq!(direct_table_tokens, fallback_tokens);
    assert_eq!(direct_table_tokens, vec![2, 4]);
}

#[test]
fn derived_single_use_terminal_possible_matches_match_dynamic_fallback() {
    let vocab = Vocab::new(vec![
        (0, b"a".to_vec()),
        (1, b"aa".to_vec()),
        (2, b"aaa".to_vec()),
        (3, b"ab".to_vec()),
        (4, b"b".to_vec()),
        (5, b"ba".to_vec()),
    ]);
    let grammar = r#"
start start;
t A ::= 'a'+;
nt start ::= A;
"#;
    let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();
    assert!(!constraint.uses_dynamic_runtime());
    assert!(constraint.possible_matches_complete);
    assert_eq!(constraint.possible_matches.len(), 1);

    // Committing one accepting prefix leaves the same terminal live, so
    // the next mask exercises the delayed-terminal exclusion table that is
    // derived from the sole global-L1 transition.
    let mut after_a = constraint.start();
    after_a.commit_token(0).unwrap();
    assert!(after_a.is_accepting());
    assert!(after_a.state.values().any(|gss| {
        !gss.all_accs_satisfy(|excluded: &TerminalsDisallowed| excluded.is_empty())
    }));
    assert_dynamic_parity(&after_a);

    assert_dynamic_parity_on_reachable_states(
        &constraint,
        3,
        "derived single-use terminal possible matches",
    );
    assert!(
        constraint
            .seed_terminal_dense_fallback
            .lock()
            .expect("fallback cache poisoned")
            .is_empty(),
        "complete derived possible matches must not invoke runtime fallback",
    );
}

#[test]
fn dynamic_mask_handles_overlapping_live_terminal_paths() {
    let vocab = Vocab::new(
        vec![
            (0, b"a".to_vec()),
            (1, b"ab".to_vec()),
            (2, b"b".to_vec()),
            (3, b"bc".to_vec()),
            (4, b"c".to_vec()),
        ]);
    let grammar = r#"
start start;
t A ::= 'a' | 'ab';
t B ::= 'b' | 'bc';
nt start ::= A B;
"#;
    let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();

    let mut state = constraint.start();
    assert_dynamic_parity(&state);
    state.commit_token(1).unwrap();
    assert_dynamic_parity(&state);
    assert!(token_allowed(&state.mask(), 3));
    state.commit_token(3).unwrap();
    assert!(state.is_accepting());
    assert_dynamic_parity(&state);
}

#[test]
fn dynamic_mask_handles_a_live_cross_terminal_prefix() {
    let vocab = Vocab::new(
        vec![
            (0, b"a".to_vec()),
            (1, b"ab".to_vec()),
            (2, b"abc".to_vec()),
            (3, b"bc".to_vec()),
            (4, b"c".to_vec()),
        ]);
    let grammar = r#"
start start;
t A ::= 'a' | 'abc';
nt start ::= A;
"#;
    let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();

    let mut state = constraint.start();
    assert_dynamic_parity(&state);
    state.commit_token(0).unwrap();
    assert_dynamic_parity(&state);
    assert!(token_allowed(&state.mask(), 3));
    assert!(!token_allowed(&state.mask(), 4));
    state.commit_token(3).unwrap();
    assert!(state.is_accepting());
    assert_dynamic_parity(&state);
}


#[test]
fn recursive_subtree_proof_preserves_sparse_aliases_and_output_polarity() {
    struct CapacityGuard(usize);
    impl Drop for CapacityGuard {
        fn drop(&mut self) {
            TEST_RECURSIVE_PRODUCT_CACHE_CAPACITY.with(|v| v.set(self.0));
        }
    }
    let _capacity = CapacityGuard(TEST_RECURSIVE_PRODUCT_CACHE_CAPACITY.with(|v| v.get()));
    // Nested a-prefix terminals make the radix walk learn an a self-loop
    // before reaching the remaining a-only subtrees. Duplicate byte strings
    // have distinct, sparse original IDs. More rejected original tokens
    // switches the same exact result to adaptive positive-side emission.
    for distractors in [0usize, 192] {
        let mut entries = vec![
            (0, b"".to_vec()), (2, b"X".to_vec()), (4, b"!".to_vec()),
            (6, b"a!".to_vec()), (8, b"Xa!".to_vec()), (10, b"z".to_vec()),
        ];
        for len in 1..=64usize {
            let id = 16 + (len as u32 - 1) * 4;
            entries.push((id, vec![b'a'; len]));
            entries.push((id + 1, vec![b'a'; len]));
        }
        for index in 0..distractors {
            entries.push((1024 + index as u32 * 2, format!("z{index:03}").into_bytes()));
        }
        let vocab = Vocab::new(entries);
        let parent = Constraint::compile(Grammar::glrm(
            r#"glrm 1; start root; extern grammar payload; nt root = "X" payload "!";"#,
        ), &vocab).unwrap();
        let child = Constraint::compile(Grammar::glrm(
            r#"start payload; t A ::= "a"+; nt payload ::= A;"#,
        ), &vocab).unwrap();
        let composed = parent.bind_grammar_dynamic_boundary("payload", child).unwrap();
        let loaded = Constraint::load(composed.save()).unwrap();
        for constraint in [&composed, &loaded] {
            for prefix in ["X", "Xa", "Xaa"] {
                let mut state = constraint.start();
                state.commit_bytes(prefix.as_bytes()).unwrap();
                let mut expected = vec![0; constraint.mask_len()];
                state.fill_recursive_mask_by_exact_full_walk(&mut expected);
                for capacity in [0, 1, 2, 16, 2048] {
                    TEST_RECURSIVE_PRODUCT_CACHE_CAPACITY.with(|v| v.set(capacity));
                    let mut actual = vec![u32::MAX; expected.len()];
                    assert!(recursive_provider::fill(&state, &mut actual).unwrap());
                    assert_eq!(actual, expected,
                        "sparse aliases: prefix={prefix:?}, capacity={capacity}, distractors={distractors}");
                    for len in 1..=64usize {
                        let id = 16 + (len as u32 - 1) * 4;
                        assert!(token_allowed(&actual, id));
                        assert!(token_allowed(&actual, id + 1));
                        assert!(!token_allowed(&actual, id + 2), "unused sparse ID must stay clear");
                    }
                    assert!(!token_allowed(&actual, 10));
                    for index in 0..distractors {
                        assert!(!token_allowed(&actual, 1024 + index as u32 * 2));
                    }
                }
            }
        }
    }
}


#[test]
fn recursive_zero_byte_domain_preserves_only_live_exact_special_ids() {
    let vocab = Vocab::new(vec![
        (0, Vec::new()), (1, b"X".to_vec()), (2, b"a".to_vec()),
        (3, b"!".to_vec()), (7, Vec::new()), (31, Vec::new()),
        (40, b"a!".to_vec()),
    ]);
    let child = crate::ConstraintSpec::builder(Grammar::glrm(
        r#"glrm 1; start child; extern token MARK; nt child = MARK "a" | "a";"#,
    ), &vocab).unwrap().bind_token("MARK", [7]).unwrap()
        .build().unwrap().compile().unwrap();
    let parent = Constraint::compile(Grammar::glrm(
        r#"glrm 1; start root; extern grammar child; nt root = "X" child "!";"#,
    ), &vocab).unwrap();
    let bound = parent.bind_grammar_dynamic_boundary("child", child).unwrap();
    let loaded = Constraint::load(bound.save()).unwrap();
    for constraint in [&bound, &loaded] {
        for prefix in ["", "X", "Xa", "Xa!"] {
            let mut state = constraint.start();
            state.commit_bytes(prefix.as_bytes()).unwrap();
            let mut oracle = vec![0; constraint.mask_len()];
            state.fill_recursive_mask_by_exact_full_walk(&mut oracle);
            let mut actual = vec![u32::MAX; oracle.len()];
            assert!(recursive_provider::fill(&state, &mut actual).unwrap());
            assert_eq!(actual, oracle, "config provider at {prefix:?}");
            assert!(!token_allowed(&actual, 0), "unbound empty alias");
            assert!(!token_allowed(&actual, 31), "second unbound empty alias");
            assert_eq!(token_allowed(&actual, 7), prefix == "X",
                "empty exact ID requires a live MARK path at {prefix:?}");
            for _ in 0..2 {
                assert!(try_fill_recursive_mask_shared(&state, &mut actual).unwrap());
                assert_eq!(actual, oracle, "shared/cache domain at {prefix:?}");
            }
            assert_eq!(state.mask(), oracle);
        }
        let mut state = constraint.start();
        for token in [1, 7, 2, 3] {
            assert!(token_allowed(&state.mask(), token));
            state.commit_token(token).unwrap();
        }
        assert!(state.is_accepting());
    }
}
