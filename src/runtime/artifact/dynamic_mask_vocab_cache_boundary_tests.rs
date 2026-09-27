use super::*;

#[test]
fn full_walk_flat32_covers_tokenizers_beyond_flat16_state_space() {
    const TARGET: u32 = 32_768;
    let tokenizer = crate::automata::lexer::tokenizer::arbitrary_flat32_test_tokenizer();

    assert!(FastTokenizerTransitions::flat16_for(&tokenizer).is_none());
    let flat32 = FastTokenizerTransitions::flat32_for(&tokenizer)
        .expect("32769-state tokenizer must fit the Flat32 full-walk coordinate");
    assert_eq!(flat32.len(), TARGET as usize + 1);
    assert_eq!(flat32.transition(&tokenizer, 0, b'a'), TARGET);
    assert_eq!(flat32.transition(&tokenizer, 0, b'b'), u32::MAX);
    let FastTokenizerTransitions::Flat32 { transitions, .. } = flat32 else {
        panic!("wide tokenizer unexpectedly used a non-Flat32 representation");
    };
    let encoded = transitions[b'a' as usize];
    assert_ne!(encoded & 0x8000_0000, 0, "finalizer bit was not encoded");
    assert_eq!(encoded & 0x7fff_ffff, TARGET);
}

#[test]
fn packed_dwa_dense_mask_cache_rejects_malformed_flat_layouts() {
    assert!(PackedDwaDenseWeightMaskCache::from_flat(4, 2, vec![1], vec![7]).is_err());
    assert!(
        PackedDwaDenseWeightMaskCache::from_flat(4, 2, vec![1, 1], vec![1, 2, 3, 4])
            .is_err()
    );
    assert!(
        PackedDwaDenseWeightMaskCache::from_flat(2, 1, vec![2], vec![7]).is_err()
    );
}

#[test]
fn vocab_only_artifact_rejects_missing_trie_root() {
    let vocab = DynamicMaskVocab::from_materialized_ordered(
        Arc::new(DynamicMaskTrie::new()),
        Arc::new(Vec::new()),
    );
    let mut artifact = vocab
        .to_vocab_artifact()
        .expect("initialized vocabulary should serialize");
    assert!(artifact.mask_tokenizer.is_none());
    assert!(artifact.full_to_mask_state.is_empty());
    artifact.trie.nodes.clear();
    let error = DynamicMaskVocab::from_artifact(artifact).unwrap_err();
    assert!(error.contains("no trie root"));
}

#[test]
fn dense_mask_projection_union_lookup_matches_exact_source_subset_union() {
    let source =
        crate::automata::lexer::tokenizer::arbitrary_epsilon_l1_test_tokenizer();
    let (built, full_to_mask_state) = source
        .try_full_determinization_all_starts(256, 16_384)
        .expect("small epsilon tokenizer should determinize from every raw start");
    let source_subsets = built.source_subsets.clone();
    let state_count = built.tokenizer.num_states();

    let mut vocab = DynamicMaskVocab::from_materialized_ordered(
        Arc::new(DynamicMaskTrie::new()),
        Arc::new(Vec::new()),
    );
    vocab.set_mask_tokenizer_quotient(built.tokenizer, full_to_mask_state);
    vocab.set_mask_tokenizer_source_subsets(source_subsets.clone());

    for left in 0..state_count {
        for right in 0..state_count {
            let mut union = source_subsets[left as usize].to_vec();
            union.extend_from_slice(&source_subsets[right as usize]);
            union.sort_unstable();
            union.dedup();
            let expected = source_subsets
                .iter()
                .position(|subset| subset.as_ref() == union.as_slice())
                .map(|state| state as u32);
            assert_eq!(
                vocab.mask_projection_state_for_projection_states(&[left, right]),
                expected,
                "projection-state union mismatch for ({left}, {right})",
            );
        }
    }
}

#[test]
fn full_artifact_round_trips_dense_mask_tokenizer_quotient() {
    let source =
        crate::automata::lexer::tokenizer::arbitrary_epsilon_l1_test_tokenizer();
    let (built, full_to_mask_state) = source
        .try_full_determinization_all_starts(256, 16_384)
        .expect("small epsilon tokenizer should determinize from every raw start");
    let expected_states = built.tokenizer.num_states();

    let mut vocab = DynamicMaskVocab::from_materialized_ordered(
        Arc::new(DynamicMaskTrie::new()),
        Arc::new(Vec::new()),
    );
    vocab.set_mask_tokenizer_quotient(built.tokenizer, full_to_mask_state.clone());

    let artifact = vocab.to_artifact().expect("full runtime artifact should serialize");
    assert_eq!(artifact.full_to_mask_state, full_to_mask_state);
    assert_eq!(
        artifact.mask_tokenizer.as_ref().map(Tokenizer::num_states),
        Some(expected_states),
    );

    let loaded = DynamicMaskVocab::from_artifact(artifact).unwrap();
    assert_eq!(loaded.full_to_mask_state.as_ref(), full_to_mask_state.as_slice());
    assert_eq!(
        loaded.mask_projection_tokenizer().map(Tokenizer::num_states),
        Some(expected_states),
    );
    assert!(matches!(
        loaded.mask_projection_fast_transitions(),
        Some(FastTokenizerTransitions::Flat16 { .. }),
    ));
}

#[test]
fn vocab_artifact_restores_root_layout_metadata_from_token_bytes() {
    let token_bytes = BTreeMap::from([
        (0u32, b"abc".to_vec()),
        (1u32, "é".as_bytes().to_vec()),
    ]);
    let mut entries = token_bytes
        .iter()
        .enumerate()
        .map(|(canonical, (_, bytes))| {
            (
                dynamic_mask_vocab_layout_class(classify_vocab_char_type(bytes), bytes),
                canonical,
                bytes.as_slice(),
            )
        })
        .collect::<Vec<_>>();
    entries.sort_unstable_by(|left, right| {
        left.0
            .cmp(&right.0)
            .then_with(|| left.2.cmp(right.2))
            .then_with(|| left.1.cmp(&right.1))
    });
    let trie = DynamicMaskTrie::from_partitioned_token_refs(&entries);
    let expected_classes = trie.root_layout_classes.clone();
    let expected_utf8 = trie.root_layout_all_valid_utf8.clone();
    let vocab = DynamicMaskVocab::from_materialized_ordered(
        Arc::new(trie),
        Arc::new(vec![vec![0], vec![1]]),
    );
    let artifact = vocab.to_vocab_artifact().unwrap();
    let loaded = DynamicMaskVocab::from_artifact(artifact).unwrap();
    assert_eq!(loaded.trie.root_layout_classes, expected_classes);
    assert_eq!(loaded.trie.root_layout_all_valid_utf8, expected_utf8);
}

#[test]
fn full_walk_all_consume_exactly_rejects_empty_edges() {
    fn one_edge(bytes: &[u8]) -> DynamicMaskTrie {
        let mut trie = DynamicMaskTrie::new();
        trie.nodes.push(DynamicMaskTrieNode {
            token_id: Some(0),
            ..DynamicMaskTrieNode::default()
        });
        let (byte_start, byte_len) = trie.push_edge_bytes(bytes);
        trie.edges.push(DynamicMaskTrieEdge {
            byte_start,
            byte_len,
            child: 1,
        });
        trie.nodes[0].first_child = 0;
        trie.nodes[0].child_len = 1;
        trie.finalize_subtree_metadata();
        trie
    }

    let nonempty = one_edge(b"abc");
    assert_eq!(nonempty.full_walk_ops().len(), 3);
    assert!(nonempty.full_walk_all_consume());

    let empty = one_edge(b"");
    assert_eq!(empty.full_walk_ops().len(), 1);
    assert!(!empty.full_walk_all_consume());
}

#[test]
fn vocab_runtime_exactly_checks_original_token_bytes() {
    let mut trie = DynamicMaskTrie::new();
    trie.nodes.push(DynamicMaskTrieNode {
        token_id: Some(0),
        ..DynamicMaskTrieNode::default()
    });
    let (byte_start, byte_len) = trie.push_edge_bytes(b"a");
    trie.edges.push(DynamicMaskTrieEdge {
        byte_start,
        byte_len,
        child: 1,
    });
    trie.nodes[0].first_child = 0;
    trie.nodes[0].child_len = 1;
    trie.finalize_subtree_metadata();
    let vocab = DynamicMaskVocab::from_materialized_ordered(
        Arc::new(trie),
        Arc::new(vec![vec![7]]),
    );
    assert!(vocab.matches_token_bytes_exact(&BTreeMap::from([(7, b"a".to_vec())])));
    assert!(!vocab.matches_token_bytes_exact(&BTreeMap::from([(7, b"b".to_vec())])));
}

#[test]
fn lazy_union_cache_try_lock_is_nonblocking_when_in_use() {
    let vocab = DynamicMaskVocab::from_materialized_ordered(
        Arc::new(DynamicMaskTrie::new()),
        Arc::new(Vec::new()),
    );
    let _owner = vocab.lock_lazy_union_cache();
    assert!(vocab.try_lock_lazy_union_cache().is_none());
}

#[test]
fn fresh_runtime_instance_shares_only_vocab_derived_data() {
    let template = DynamicMaskVocab::from_materialized_ordered(
        Arc::new(DynamicMaskTrie::new()),
        Arc::new(Vec::new()),
    );
    let fresh = template.fresh_runtime_instance();

    assert!(Arc::ptr_eq(&template.trie, &fresh.trie));
    match (&template.token_aliases, &fresh.token_aliases) {
        (DynamicMaskAliasStore::Ordered(left), DynamicMaskAliasStore::Ordered(right)) => {
            assert!(Arc::ptr_eq(left, right));
        }
        _ => panic!("materialized ordered vocabulary changed alias representation"),
    }
    assert!(Arc::ptr_eq(
        &template.canonical_original_token_offsets,
        &fresh.canonical_original_token_offsets,
    ));
    assert!(Arc::ptr_eq(
        &template.canonical_original_tokens,
        &fresh.canonical_original_tokens,
    ));
    assert!(Arc::ptr_eq(
        &template.node_token_markers,
        &fresh.node_token_markers,
    ));
    assert!(Arc::ptr_eq(
        &template.subtree_original_token_offsets,
        &fresh.subtree_original_token_offsets,
    ));
    assert!(Arc::ptr_eq(
        &template.subtree_original_tokens,
        &fresh.subtree_original_tokens,
    ));

    assert!(!Arc::ptr_eq(&template.mask_cache, &fresh.mask_cache));
    assert!(!Arc::ptr_eq(
        &template.direct_regular_frontier_cache,
        &fresh.direct_regular_frontier_cache,
    ));
    assert!(!Arc::ptr_eq(
        &template.direct_regular_wide_frontier_index_cache,
        &fresh.direct_regular_wide_frontier_index_cache,
    ));
    assert!(!Arc::ptr_eq(
        &template.direct_regular_terminal_support,
        &fresh.direct_regular_terminal_support,
    ));
    assert!(!Arc::ptr_eq(
        &template.self_loop_projections,
        &fresh.self_loop_projections,
    ));
}
