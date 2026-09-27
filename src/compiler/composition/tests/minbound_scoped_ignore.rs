// Test-only minimal-reproduction harness for scoped-ignore boundary
// over-admission. `include!()`'d inside `constraint_compose::tests`.
//
// Regression coverage for two previously observed failure mechanisms:
//   1. a parent boundary shard over-admits an invalid scoped-ignore boundary
//      token (the `distinct` witness);
//   2. the parent component mask over-admits after a partial parent ignore (the
//      `scoped_visible` witness).
//
// Every mask is decoded automatically from its `u32` words; no human bit
// arithmetic is trusted anywhere.

/// Automatic mask decoding: the set of set bits, low bit first.
fn mb_scoped_mask_bits(mask: &[u32]) -> Vec<u32> {
    let mut bits = Vec::new();
    for (word_index, &word) in mask.iter().enumerate() {
        let mut word = word;
        while word != 0 {
            let bit = word.trailing_zeros() as usize;
            bits.push((word_index * 32 + bit) as u32);
            word &= word - 1;
        }
    }
    bits
}

/// One tiny scoped-ignore case. `parent_explicit` and `parent_inline` carry the
/// *same* grammar rules and vocabulary; only the `SUB` binding differs
/// (`t SUB ::= @token(999);` for the explicit segmented composition,
/// `extern grammar SUB;` for the AST inline reference).
struct MbScopedIgnoreVariant {
    name: &'static str,
    parent_explicit: &'static str,
    parent_inline: &'static str,
    child: &'static str,
    vocab: Vec<(u32, Vec<u8>)>,
    /// Tokens independently expected admissible from the start state, derived
    /// from the literal grammar (not from any backend).
    expected_valid: Vec<u32>,
    /// Tokens independently expected inadmissible from the start state.
    expected_invalid: Vec<u32>,
}

fn mb_scoped_ignore_variants() -> Vec<MbScopedIgnoreVariant> {
    let child_full = "start child;\nignore CHILD_WS;\nt CHILD_WS ::= \"\\t\"+;\nnt child ::= \"a\";\n";
    vec![
        MbScopedIgnoreVariant {
            name: "full",
            parent_explicit: "start document;\nignore PARENT_WS;\nt PARENT_WS ::= \" \"+;\nt SUB ::= @token(999);\nnt document ::= \"X\" SUB \"!\";\n",
            parent_inline: "start document;\nignore PARENT_WS;\nt PARENT_WS ::= \" \"+;\nextern grammar SUB;\nnt document ::= \"X\" SUB \"!\";\n",
            child: child_full,
            vocab: vec![(0, b"Xa!".to_vec()), (1, b"X\t a!".to_vec())],
            expected_valid: vec![0],
            expected_invalid: vec![1],
        },
        MbScopedIgnoreVariant {
            name: "no_bang",
            parent_explicit: "start document;\nignore PARENT_WS;\nt PARENT_WS ::= \" \"+;\nt SUB ::= @token(999);\nnt document ::= \"X\" SUB;\n",
            parent_inline: "start document;\nignore PARENT_WS;\nt PARENT_WS ::= \" \"+;\nextern grammar SUB;\nnt document ::= \"X\" SUB;\n",
            child: child_full,
            vocab: vec![(0, b"Xa".to_vec()), (1, b"X\t a".to_vec())],
            expected_valid: vec![0],
            expected_invalid: vec![1],
        },
        MbScopedIgnoreVariant {
            name: "single_ws",
            parent_explicit: "start document;\nignore PARENT_WS;\nt PARENT_WS ::= \" \";\nt SUB ::= @token(999);\nnt document ::= \"X\" SUB \"!\";\n",
            parent_inline: "start document;\nignore PARENT_WS;\nt PARENT_WS ::= \" \";\nextern grammar SUB;\nnt document ::= \"X\" SUB \"!\";\n",
            child: "start child;\nignore CHILD_WS;\nt CHILD_WS ::= \"\\t\";\nnt child ::= \"a\";\n",
            vocab: vec![(0, b"Xa!".to_vec()), (1, b"X\t a!".to_vec())],
            expected_valid: vec![0],
            expected_invalid: vec![1],
        },
        MbScopedIgnoreVariant {
            name: "no_bang_single_ws",
            parent_explicit: "start document;\nignore PARENT_WS;\nt PARENT_WS ::= \" \";\nt SUB ::= @token(999);\nnt document ::= \"X\" SUB;\n",
            parent_inline: "start document;\nignore PARENT_WS;\nt PARENT_WS ::= \" \";\nextern grammar SUB;\nnt document ::= \"X\" SUB;\n",
            child: "start child;\nignore CHILD_WS;\nt CHILD_WS ::= \"\\t\";\nnt child ::= \"a\";\n",
            vocab: vec![(0, b"Xa".to_vec()), (1, b"X\t a".to_vec())],
            expected_valid: vec![0],
            expected_invalid: vec![1],
        },
    ]
}

/// Compile one variant through all three backends and return
/// `(static_bits, dynamic_bits, inline_bits)` from the start state.
fn mb_run_scoped_ignore_variant(
    variant: &MbScopedIgnoreVariant,
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let vocab = Vocab::new(variant.vocab.clone());
    let parent = Constraint::from_glrm_grammar(variant.parent_explicit, &vocab)
        .unwrap_or_else(|error| panic!("{}: parent compile: {error}", variant.name));
    let child = Constraint::from_glrm_grammar(variant.child, &vocab)
        .unwrap_or_else(|error| panic!("{}: child compile: {error}", variant.name));
    let composed_static = parent
        .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
        .unwrap_or_else(|error| panic!("{}: static compose: {error}", variant.name));
    let composed_dynamic = parent
        .compose_linked_children_for_test_dynamic(&[("SUB", &child)], &vocab)
        .unwrap_or_else(|error| panic!("{}: dynamic compose: {error}", variant.name));
    let inline =
        compile_inline_source_oracle(variant.parent_inline, &[("SUB", variant.child)], &vocab);

    let static_bits = mb_scoped_mask_bits(&composed_static.start().mask());
    let dynamic_bits = mb_scoped_mask_bits(&composed_dynamic.start().mask());
    let inline_bits = mb_scoped_mask_bits(&inline.start().mask());

    eprintln!("=== [min-repro] variant {} ===", variant.name);
    eprintln!("[min-repro] parent_explicit:\n{}", variant.parent_explicit);
    eprintln!("[min-repro] child:\n{}", variant.child);
    eprintln!(
        "[min-repro] vocab={:?}",
        variant
            .vocab
            .iter()
            .map(|(id, bytes)| (*id, String::from_utf8_lossy(bytes).into_owned()))
            .collect::<Vec<_>>()
    );
    eprintln!(
        "[min-repro] start_bits static={static_bits:?} dynamic={dynamic_bits:?} inline={inline_bits:?}"
    );
    eprintln!(
        "[min-repro] static_only={:?} inline_only={:?}",
        static_bits
            .iter()
            .copied()
            .filter(|bit| !inline_bits.contains(bit))
            .collect::<Vec<_>>(),
        inline_bits
            .iter()
            .copied()
            .filter(|bit| !static_bits.contains(bit))
            .collect::<Vec<_>>(),
    );

    for (token, bytes) in &variant.vocab {
        let token = *token;
        let spelling = String::from_utf8_lossy(bytes).into_owned();
        let mut static_state = composed_static.start();
        let static_ok = static_state.commit_token(token).is_ok();
        let mut dynamic_state = composed_dynamic.start();
        let dynamic_ok = dynamic_state.commit_token(token).is_ok();
        let mut inline_state = inline.start();
        let inline_ok = inline_state.commit_token(token).is_ok();
        eprintln!(
            "[min-repro] token {token} {spelling:?} start_allows static={} dynamic={} inline={} | commit static={static_ok} dynamic={dynamic_ok} inline={inline_ok} | accept static={} dynamic={} inline={}",
            static_bits.contains(&token),
            dynamic_bits.contains(&token),
            inline_bits.contains(&token),
            static_ok && static_state.is_accepting(),
            dynamic_ok && dynamic_state.is_accepting(),
            inline_ok && inline_state.is_accepting(),
        );
    }

    (static_bits, dynamic_bits, inline_bits)
}

/// The full `distinct` witness grammar: `document ::= "X" SUB "!"`, parent
/// ignore `" "+`, child `"a"` with ignore `"\t"+`.
fn mb_distinct_grammar() -> (&'static str, &'static str, &'static str) {
    (
        "start document;\nignore PARENT_WS;\nt PARENT_WS ::= \" \"+;\nt SUB ::= @token(999);\nnt document ::= \"X\" SUB \"!\";\n",
        "start document;\nignore PARENT_WS;\nt PARENT_WS ::= \" \"+;\nextern grammar SUB;\nnt document ::= \"X\" SUB \"!\";\n",
        "start child;\nignore CHILD_WS;\nt CHILD_WS ::= \"\\t\"+;\nnt child ::= \"a\";\n",
    )
}

fn mb_distinct_original_vocab() -> Vec<(u32, Vec<u8>)> {
    vec![
        (0, b"X \ta!".to_vec()),
        (1, b"X\ta!".to_vec()),
        (2, b"X".to_vec()),
        (3, b" ".to_vec()),
        (4, b"\t".to_vec()),
        (5, b"a".to_vec()),
        (6, b"!".to_vec()),
        (7, b" \ta".to_vec()),
        (8, b"a!".to_vec()),
        (9, b"X\t a!".to_vec()),
        (10, b"Xa\t !".to_vec()),
        (11, b"Xa \t!".to_vec()),
        (12, b"\tX a!".to_vec()),
    ]
}

/// Compile the `distinct` grammar with the given vocabulary and return the
/// divergent start-mask bits `(static_only, inline_only)`, or `None` when the
/// segmented-static and inline backends agree.
fn mb_distinct_divergence(vocab_entries: &[(u32, Vec<u8>)]) -> Option<(Vec<u32>, Vec<u32>)> {
    let (parent_explicit, parent_inline, child_src) = mb_distinct_grammar();
    let vocab = Vocab::new(vocab_entries.to_vec());
    let parent = Constraint::from_glrm_grammar(parent_explicit, &vocab).ok()?;
    let child = Constraint::from_glrm_grammar(child_src, &vocab).ok()?;
    let composed_static = parent
        .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
        .ok()?;
    let inline = compile_inline_source_oracle(parent_inline, &[("SUB", child_src)], &vocab);
    let static_bits = mb_scoped_mask_bits(&composed_static.start().mask());
    let inline_bits = mb_scoped_mask_bits(&inline.start().mask());
    let static_only = static_bits
        .iter()
        .copied()
        .filter(|bit| !inline_bits.contains(bit))
        .collect::<Vec<_>>();
    let inline_only = inline_bits
        .iter()
        .copied()
        .filter(|bit| !static_bits.contains(bit))
        .collect::<Vec<_>>();
    (!static_only.is_empty() || !inline_only.is_empty()).then_some((static_only, inline_only))
}

/// Greedy single-token deletion over the `distinct` vocabulary, keeping any
/// deletion that still diverges. Reports the measured (not proven-minimal)
/// vocabulary plus the surviving divergent bits.
fn mb_minimize_distinct_vocab() -> (Vec<(u32, Vec<u8>)>, Vec<u32>, Vec<u32>) {
    let mut vocab = mb_distinct_original_vocab();
    let Some(mut divergence) = mb_distinct_divergence(&vocab) else {
        return (vocab, Vec::new(), Vec::new());
    };
    loop {
        let mut removed_any = false;
        let mut index = 0;
        while index < vocab.len() {
            let mut candidate = vocab.clone();
            candidate.remove(index);
            if let Some(candidate_divergence) = mb_distinct_divergence(&candidate) {
                vocab = candidate;
                divergence = candidate_divergence;
                removed_any = true;
            } else {
                index += 1;
            }
        }
        if !removed_any {
            break;
        }
    }
    (vocab, divergence.0, divergence.1)
}

fn mb_spelling(vocab: &[(u32, Vec<u8>)], token: u32) -> String {
    vocab
        .iter()
        .find(|(id, _)| *id == token)
        .map(|(_, bytes)| String::from_utf8_lossy(bytes).into_owned())
        .unwrap_or_else(|| format!("<token {token} not in vocab>"))
}

/// Mapping trace: original -> composed terminal ids, original -> composed
/// tokenizer-state ids, and the disjoint -> compact parser-state layout.
fn mb_scoped_ignore_mapping_trace(
    label: &str,
    parent: &Constraint,
    child: &Constraint,
    composed_static: &Constraint,
) {
    eprintln!("[map-trace {label}] parent_terminals={}", parent.table.num_terminals);
    eprintln!("[map-trace {label}] child_terminals={}", child.table.num_terminals);
    for (index, name) in parent.terminal_display_names.iter().enumerate() {
        eprintln!("[map-trace {label}] parent local terminal {index} -> composed {index} name={name:?}");
    }
    eprintln!(
        "[map-trace {label}] tokenizer_states parent={} child={} composed={}",
        parent.tokenizer.num_states(),
        child.tokenizer.num_states(),
        composed_static.tokenizer.num_states(),
    );
    let Some(overlay) = composed_static.static_dynamic_overlay.as_ref() else {
        eprintln!("[map-trace {label}] no static_dynamic_overlay");
        return;
    };
    eprintln!("[map-trace {label}] terminal_offsets={:?}", overlay.terminal_offsets);
    eprintln!(
        "[map-trace {label}] tokenizer_state_offsets={:?}",
        overlay.tokenizer_state_offsets
    );
    if let Some(&child_offset) = overlay.terminal_offsets.get(1) {
        for (index, name) in child.terminal_display_names.iter().enumerate() {
            eprintln!(
                "[map-trace {label}] child local terminal {index} -> composed {} name={name:?}",
                child_offset + index as u32,
            );
        }
    }
    eprintln!(
        "[map-trace {label}] root_state0_component={:?}",
        composed_static.compact_segmented_parser_component(0),
    );
    match composed_static.recursive_parser_layout() {
        Ok(Some(layout)) => eprintln!(
            "[map-trace {label}] parser_layout component_offsets={:?} leaf_state_offsets={:?} leaves_top_components={:?} total_states={}",
            layout.component_offsets,
            layout.leaf_state_offsets,
            layout
                .leaves
                .iter()
                .map(|leaf| leaf.top_component)
                .collect::<Vec<_>>(),
            layout.total_states,
        ),
        Ok(None) => eprintln!("[map-trace {label}] no recursive parser layout"),
        Err(error) => eprintln!("[map-trace {label}] recursive parser layout error: {error}"),
    }
    for (component_index, component) in overlay.segmented_parser_components.iter().enumerate() {
        let (backend, candidates) = match component.boundary.as_ref() {
            Some(shard) => {
                let backend = match &shard.backend {
                    crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_) => {
                        "StaticParser"
                    }
                    crate::runtime::SegmentedBoundaryShardBackend::DynamicTerminalTrie(_) => {
                        "DynamicTerminalTrie"
                    }
                    crate::runtime::SegmentedBoundaryShardBackend::DynamicDirect => {
                        "DynamicDirect"
                    }
                };
                (
                    backend,
                    shard
                        .candidate_tokens
                        .as_ref()
                        .map(|tokens| tokens.iter().copied().collect::<Vec<_>>())
                        .unwrap_or_default(),
                )
            }
            None => ("<no boundary shard>", Vec::new()),
        };
        eprintln!(
            "[map-trace {label}] component {component_index} terminal_offset={} boundary_backend={backend} candidate_tokens={candidates:?}",
            component.terminal_offset,
        );
    }
}

/// Compile the `distinct` grammar for one vocabulary and assert that the
/// segmented-static backend agrees with the inline reference (it does not, until
/// the parent-side over-admission is fixed). Reports the start masks first.
fn mb_assert_distinct_vocab_agrees(label: &str, vocab_entries: &[(u32, Vec<u8>)]) {
    let (parent_explicit, parent_inline, child_src) = mb_distinct_grammar();
    let vocab = Vocab::new(vocab_entries.to_vec());
    let parent = Constraint::from_glrm_grammar(parent_explicit, &vocab).unwrap();
    let child = Constraint::from_glrm_grammar(child_src, &vocab).unwrap();
    let composed_static = parent
        .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
        .unwrap();
    let composed_dynamic = parent
        .compose_linked_children_for_test_dynamic(&[("SUB", &child)], &vocab)
        .unwrap();
    let inline = compile_inline_source_oracle(parent_inline, &[("SUB", child_src)], &vocab);
    let static_bits = mb_scoped_mask_bits(&composed_static.start().mask());
    let dynamic_bits = mb_scoped_mask_bits(&composed_dynamic.start().mask());
    let inline_bits = mb_scoped_mask_bits(&inline.start().mask());
    eprintln!(
        "[min-repro {label}] vocab={:?}",
        vocab_entries
            .iter()
            .map(|(id, bytes)| (*id, String::from_utf8_lossy(bytes).into_owned()))
            .collect::<Vec<_>>()
    );
    eprintln!(
        "[min-repro {label}] start static={static_bits:?} dynamic={dynamic_bits:?} inline={inline_bits:?} static_only={:?}",
        static_bits
            .iter()
            .copied()
            .filter(|bit| !inline_bits.contains(bit))
            .collect::<Vec<_>>(),
    );
    assert_eq!(dynamic_bits, inline_bits, "{label}: dynamic vs inline start mask");
    assert_eq!(static_bits, inline_bits, "{label}: static vs inline start mask");
}

/// Dense-id remap of the measured minimal witness: the same two tokens under ids
/// 0/1 (not the original 10/11) must reproduce, so the old ids do not obscure
/// the mapping.
#[test]
fn minimal_scoped_ignore_dense_id_remap_reproduces() {
    mb_assert_distinct_vocab_agrees(
        "dense_ids_0_1",
        &[(0, b"Xa\t !".to_vec()), (1, b"Xa \t!".to_vec())],
    );
}

/// Single-invalid-token vocabulary: only the invalid boundary token is present.
#[test]
fn minimal_scoped_ignore_single_invalid_token_reproduces() {
    mb_assert_distinct_vocab_agrees("single_invalid", &[(0, b"Xa \t!".to_vec())]);
}

/// Construction-only regression for the boundary L2P equivalence partition.
///
/// Parent invariant: two model tokens may share a boundary internal token id
/// only if their weighted terminal languages are identical for every relevant
/// starting tokenizer state. For the dense minimal witness the merged composed
/// tokenizer must emit two *different* ordered terminal-label sequences
/// (`[X,a,CHILD_WS,PARENT_WS,!]` vs `[X,a,PARENT_WS,CHILD_WS,!]`), so the
/// boundary id map must keep the two originals in separate classes.
///
/// This asserts on the composed constraint's own internal-token map, which is
/// the same producer the boundary shard consumes, so it is independent of the
/// parser product and of any shard DWA.
#[test]
fn minimal_scoped_ignore_boundary_token_classes_keep_scoped_orderings_separate() {
    let vocab_entries: Vec<(u32, Vec<u8>)> = vec![
        (0, b"Xa\t !".to_vec()),
        (1, b"Xa \t!".to_vec()),
    ];
    let (parent_explicit, parent_inline, child_src) = mb_distinct_grammar();
    let vocab = Vocab::new(vocab_entries.clone());
    let parent = Constraint::from_glrm_grammar(parent_explicit, &vocab).unwrap();
    let child = Constraint::from_glrm_grammar(child_src, &vocab).unwrap();
    let composed_static = parent
        .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
        .unwrap();
    let _ = parent_inline;

    let overlay = composed_static
        .static_dynamic_overlay
        .as_ref()
        .expect("static_dynamic_overlay must be present");
    let boundary_shard = overlay.segmented_parser_components[0]
        .boundary
        .as_ref()
        .expect("component 0 must have a boundary shard");
    let (boundary_valid_internal, boundary_invalid_internal, groups) = match &boundary_shard.backend {
        crate::runtime::SegmentedBoundaryShardBackend::StaticParser(parser) => {
            let find_internal = |token: u32| {
                parser
                    .internal_token_to_originals
                    .iter()
                    .position(|group| group.contains(&token))
            };
            (find_internal(0), find_internal(1), parser.internal_token_to_originals.clone())
        }
        other => panic!("expected StaticParser backend, got {other:?}"),
    };
    eprintln!(
        "[l2p-token-classes][boundary_shard] valid=0 internal={boundary_valid_internal:?} invalid=1 internal={boundary_invalid_internal:?} groups={groups:?}",
    );

    assert!(
        boundary_valid_internal.is_some() && boundary_invalid_internal.is_some(),
        "both boundary tokens must be present in the boundary shard internal-token map",
    );
    assert_ne!(
        boundary_valid_internal, boundary_invalid_internal,
        "boundary equivalence merged the valid and invalid scoped-ignore orderings in boundary shard",
    );
}

/// Swapped-id vocabulary: invalid token at id 0, valid token at id 1.
/// Asserts that id ordering does not affect separate boundary classes and start mask agreement.
#[test]
fn minimal_scoped_ignore_swapped_ids_reproduces() {
    let vocab_entries: Vec<(u32, Vec<u8>)> = vec![
        (0, b"Xa \t!".to_vec()),
        (1, b"Xa\t !".to_vec()),
    ];
    mb_assert_distinct_vocab_agrees("swapped_ids_0_1", &vocab_entries);

    let (parent_explicit, _parent_inline, child_src) = mb_distinct_grammar();
    let vocab = Vocab::new(vocab_entries);
    let parent = Constraint::from_glrm_grammar(parent_explicit, &vocab).unwrap();
    let child = Constraint::from_glrm_grammar(child_src, &vocab).unwrap();
    let composed_static = parent
        .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
        .unwrap();

    let overlay = composed_static
        .static_dynamic_overlay
        .as_ref()
        .expect("static_dynamic_overlay must be present");
    let boundary_shard = overlay.segmented_parser_components[0]
        .boundary
        .as_ref()
        .expect("component 0 must have a boundary shard");
    let (boundary_invalid_internal, boundary_valid_internal) = match &boundary_shard.backend {
        crate::runtime::SegmentedBoundaryShardBackend::StaticParser(parser) => {
            let find_internal = |token: u32| {
                parser
                    .internal_token_to_originals
                    .iter()
                    .position(|group| group.contains(&token))
            };
            (find_internal(0), find_internal(1))
        }
        other => panic!("expected StaticParser backend, got {other:?}"),
    };
    assert!(
        boundary_valid_internal.is_some() && boundary_invalid_internal.is_some(),
        "both boundary tokens must be present in the boundary shard internal-token map",
    );
    assert_ne!(
        boundary_valid_internal, boundary_invalid_internal,
        "boundary equivalence must keep swapped valid and invalid scoped-ignore orderings separate",
    );
}

/// Single-valid-token vocabulary: only the valid boundary token is present.
#[test]
fn minimal_scoped_ignore_single_valid_token_reproduces() {
    mb_assert_distinct_vocab_agrees("single_valid", &[(0, b"Xa\t !".to_vec())]);
}

/// Reload parity: a composed constraint with scoped ignores survives serialization
/// and roundtrips with identical start mask.
#[test]
fn minimal_scoped_ignore_survives_reload() {
    let vocab_entries: Vec<(u32, Vec<u8>)> = vec![
        (0, b"Xa\t !".to_vec()),
        (1, b"Xa \t!".to_vec()),
    ];
    let (parent_explicit, _parent_inline, child_src) = mb_distinct_grammar();
    let vocab = Vocab::new(vocab_entries);
    let parent = Constraint::from_glrm_grammar(parent_explicit, &vocab).unwrap();
    let child = Constraint::from_glrm_grammar(child_src, &vocab).unwrap();
    let composed_static = parent
        .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
        .unwrap();

    let bytes = composed_static.save();
    let reloaded = Constraint::load(&bytes).expect("reload composed static constraint");
    assert_eq!(
        mb_scoped_mask_bits(&reloaded.start().mask()),
        mb_scoped_mask_bits(&composed_static.start().mask()),
        "reloaded start mask must match original static mask",
    );
}

/// Multi-terminal composition without ignores: child starts with 'a', parent has prefix 'X',
/// testing that bounded view suffix trajectories across component roots correctly partition
/// distinct tokens without relying on scoped ignores.
#[test]
fn multi_terminal_epsilon_union_without_ignores() {
    let vocab = Vocab::new(vec![
        (0, b"Xa".to_vec()),
        (1, b"Xb".to_vec()),
    ]);
    let parent_src = "start document;\nt SUB ::= @token(999);\nnt document ::= \"X\" SUB;\n";
    let child_src = "start child;\nnt child ::= \"a\";\n";
    let parent = Constraint::from_glrm_grammar(parent_src, &vocab).unwrap();
    let child = Constraint::from_glrm_grammar(child_src, &vocab).unwrap();
    let composed_static = parent
        .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
        .unwrap();
    let composed_dynamic = parent
        .compose_linked_children_for_test_dynamic(&[("SUB", &child)], &vocab)
        .unwrap();
    let static_bits = mb_scoped_mask_bits(&composed_static.start().mask());
    let dynamic_bits = mb_scoped_mask_bits(&composed_dynamic.start().mask());
    assert_eq!(
        static_bits, dynamic_bits,
        "static vs dynamic start mask for multi-terminal composition without ignores",
    );
    assert_eq!(static_bits, vec![0], "only 'Xa' should be valid at start");
}

/// Minimal `distinct` witness: the parent boundary shard admits an invalid
/// boundary token while the inline reference and the dynamic backend reject it.
/// The tiny hand-written variants may not reproduce; a greedy deletion search
/// over the full vocabulary then measures the smallest reproducing subset.
#[test]
fn minimal_scoped_ignore_parent_shard_overadmits_invalid_boundary_token() {
    let mut reproduced = Vec::new();
    for variant in &mb_scoped_ignore_variants() {
        let (static_bits, dynamic_bits, inline_bits) = mb_run_scoped_ignore_variant(variant);
        for token in &variant.expected_valid {
            assert!(
                inline_bits.contains(token),
                "{}: inline rejects independently-valid token {token}",
                variant.name,
            );
        }
        for token in &variant.expected_invalid {
            assert!(
                !inline_bits.contains(token),
                "{}: inline admits independently-invalid token {token}",
                variant.name,
            );
        }
        assert_eq!(
            dynamic_bits, inline_bits,
            "{}: dynamic vs inline start mask",
            variant.name,
        );
        if static_bits != inline_bits {
            reproduced.push(variant.name);
        }
    }
    eprintln!("[min-repro] distinct tiny variants reproducing divergence: {reproduced:?}");

    let (min_vocab, static_only, inline_only) = mb_minimize_distinct_vocab();
    eprintln!(
        "[min-repro] minimized vocab={:?}",
        min_vocab
            .iter()
            .map(|(id, bytes)| (*id, String::from_utf8_lossy(bytes).into_owned()))
            .collect::<Vec<_>>()
    );
    eprintln!(
        "[min-repro] minimized static_only={:?} inline_only={:?}",
        static_only
            .iter()
            .map(|bit| (*bit, mb_spelling(&min_vocab, *bit)))
            .collect::<Vec<_>>(),
        inline_only
            .iter()
            .map(|bit| (*bit, mb_spelling(&min_vocab, *bit)))
            .collect::<Vec<_>>(),
    );

    // Mapping trace for the full witness (which reproduces).
    let (parent_explicit, parent_inline, child_src) = mb_distinct_grammar();
    let full_vocab = Vocab::new(mb_distinct_original_vocab());
    let full_parent = Constraint::from_glrm_grammar(parent_explicit, &full_vocab).unwrap();
    let full_child = Constraint::from_glrm_grammar(child_src, &full_vocab).unwrap();
    let full_static = full_parent
        .compose_linked_children_for_test(&[("SUB", &full_child)], &full_vocab)
        .unwrap();
    let full_inline =
        compile_inline_source_oracle(parent_inline, &[("SUB", child_src)], &full_vocab);
    let full_static_bits = mb_scoped_mask_bits(&full_static.start().mask());
    let full_inline_bits = mb_scoped_mask_bits(&full_inline.start().mask());
    eprintln!(
        "[min-repro] full static_bits={full_static_bits:?} inline_bits={full_inline_bits:?}"
    );
    mb_scoped_ignore_mapping_trace("full", &full_parent, &full_child, &full_static);

    assert!(
        static_only.is_empty() && inline_only.is_empty(),
        "parent boundary shard over-admits; minimized static-only bits {:?} (spellings {:?})",
        static_only,
        static_only
            .iter()
            .map(|bit| mb_spelling(&min_vocab, *bit))
            .collect::<Vec<_>>(),
    );
}

/// Byte-level control: the same grammar with a one-byte-per-token vocabulary.
/// The invalid boundary order `X TAB SPACE a !` must be rejected by the inline
/// and dynamic backends independently of how the text is tokenized.
#[test]
fn minimal_scoped_ignore_byte_level_control_sequence() {
    let vocab = Vocab::new(vec![
        (0, b"X".to_vec()),
        (1, b"a".to_vec()),
        (2, b"!".to_vec()),
        (3, b" ".to_vec()),
        (4, b"\t".to_vec()),
    ]);
    let parent_explicit = "start document;\nignore PARENT_WS;\nt PARENT_WS ::= \" \"+;\nt SUB ::= @token(999);\nnt document ::= \"X\" SUB \"!\";\n";
    let parent_inline = "start document;\nignore PARENT_WS;\nt PARENT_WS ::= \" \"+;\nextern grammar SUB;\nnt document ::= \"X\" SUB \"!\";\n";
    let child_src = "start child;\nignore CHILD_WS;\nt CHILD_WS ::= \"\\t\"+;\nnt child ::= \"a\";\n";
    let parent = Constraint::from_glrm_grammar(parent_explicit, &vocab).unwrap();
    let child = Constraint::from_glrm_grammar(child_src, &vocab).unwrap();
    let composed_static = parent
        .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
        .unwrap();
    let composed_dynamic = parent
        .compose_linked_children_for_test_dynamic(&[("SUB", &child)], &vocab)
        .unwrap();
    let inline = compile_inline_source_oracle(parent_inline, &[("SUB", child_src)], &vocab);

    let valid = [0u32, 4, 1, 2];
    let invalid = [0u32, 4, 3, 1, 2];
    let mut static_invalid_accepted = false;
    for (label, sequence, expected) in [
        ("valid_X_TAB_a_BANG", valid.as_slice(), true),
        ("invalid_X_TAB_SPACE_a_BANG", invalid.as_slice(), false),
    ] {
        for (name, constraint) in [
            ("static", &composed_static),
            ("dynamic", &composed_dynamic),
            ("inline", &inline),
        ] {
            let mut state = constraint.start();
            let mut committed = true;
            for &token in sequence {
                if state.commit_token(token).is_err() {
                    committed = false;
                    break;
                }
            }
            let accepted = committed && state.is_accepting();
            eprintln!("[min-repro-byte] {label} {name} accepted={accepted}");
            if name != "static" {
                assert_eq!(accepted, expected, "{label} {name} accepted mismatch");
            } else if accepted != expected {
                static_invalid_accepted = true;
            }
        }
    }
    assert!(
        !static_invalid_accepted,
        "static backend disagrees with inline on the byte-level scoped-ignore sequence",
    );
}

/// Minimal `scoped_visible` witness: after a partial parent ignore (`Xa`), the
/// parent component mask admits `b!`, which would need the child to have been
/// entered first. Reported separately so it is not assumed to share the
/// `distinct` cause.
#[test]
fn minimal_scoped_visible_parent_component_overadmits_after_partial_ignore() {
    let vocab = Vocab::new(vec![
        (0, b"Xa".to_vec()),
        (1, b"bt".to_vec()),
        (2, b"b!".to_vec()),
        (3, b"X".to_vec()),
        (4, b"a".to_vec()),
        (5, b"b".to_vec()),
        (6, b"t".to_vec()),
        (7, b"!".to_vec()),
    ]);
    let parent_explicit = "start document;\nignore PARENT_WS;\nt PARENT_WS ::= \"ab\";\nt X ::= \"X\";\nt BANG ::= \"!\";\nt SUB ::= @token(999);\nnt document ::= X SUB BANG;\n";
    let parent_inline = "start document;\nignore PARENT_WS;\nt PARENT_WS ::= \"ab\";\nt X ::= \"X\";\nt BANG ::= \"!\";\nextern grammar SUB;\nnt document ::= X SUB BANG;\n";
    let child_src = "start child;\nt CHILD_T ::= \"t\";\nnt child ::= CHILD_T;\n";
    let parent = Constraint::from_glrm_grammar(parent_explicit, &vocab).unwrap();
    let child = Constraint::from_glrm_grammar(child_src, &vocab).unwrap();
    let composed_static = parent
        .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
        .unwrap();
    let composed_dynamic = parent
        .compose_linked_children_for_test_dynamic(&[("SUB", &child)], &vocab)
        .unwrap();
    let inline = compile_inline_source_oracle(parent_inline, &[("SUB", child_src)], &vocab);

    mb_scoped_ignore_mapping_trace("visible", &parent, &child, &composed_static);

    let mut static_state = composed_static.start();
    let mut dynamic_state = composed_dynamic.start();
    let mut inline_state = inline.start();
    let static_start = mb_scoped_mask_bits(&static_state.mask());
    let dynamic_start = mb_scoped_mask_bits(&dynamic_state.mask());
    let inline_start = mb_scoped_mask_bits(&inline_state.mask());
    eprintln!(
        "[min-repro-visible] start static={static_start:?} dynamic={dynamic_start:?} inline={inline_start:?}"
    );

    static_state.commit_token(0).unwrap();
    dynamic_state.commit_token(0).unwrap();
    inline_state.commit_token(0).unwrap();
    let static_after = mb_scoped_mask_bits(&static_state.mask());
    let dynamic_after = mb_scoped_mask_bits(&dynamic_state.mask());
    let inline_after = mb_scoped_mask_bits(&inline_state.mask());
    eprintln!(
        "[min-repro-visible] after Xa static={static_after:?} dynamic={dynamic_after:?} inline={inline_after:?}"
    );
    eprintln!(
        "[min-repro-visible] after Xa static_only={:?} inline_only={:?}",
        static_after
            .iter()
            .copied()
            .filter(|bit| !inline_after.contains(bit))
            .collect::<Vec<_>>(),
        inline_after
            .iter()
            .copied()
            .filter(|bit| !static_after.contains(bit))
            .collect::<Vec<_>>(),
    );

    assert_eq!(
        dynamic_after, inline_after,
        "dynamic vs inline mask after partial parent ignore",
    );
    assert_eq!(
        static_after, inline_after,
        "static parent component mask over-admits after partial parent ignore",
    );
}

/// Vocabulary regression: mixed first byte retaining valid/invalid pair plus separate prefix token.
/// Also checks genuine equivalent distinct bytes (e.g. Xa! vs Xaa!) where child terminal `a+`
/// retains the expected equivalence class behavior across backends.
#[test]
fn test_scoped_ignore_vocab_mixed_first_byte_and_distinct_equivalent_bytes() {
    let vocab = Vocab::new(vec![
        (0, b"Xa!".to_vec()),
        (1, b"X\t a!".to_vec()),
        (2, b"X".to_vec()),
        (3, b"Xaa!".to_vec()),
        (4, b"X\t aa!".to_vec()),
        (5, b"a".to_vec()),
        (6, b"!".to_vec()),
    ]);
    let parent_explicit = "start document;\nignore PARENT_WS;\nt PARENT_WS ::= \" \"+;\nt SUB ::= @token(999);\nnt document ::= \"X\" SUB \"!\";\n";
    let parent_inline = "start document;\nignore PARENT_WS;\nt PARENT_WS ::= \" \"+;\nextern grammar SUB;\nnt document ::= \"X\" SUB \"!\";\n";
    let child_src = "start child;\nignore CHILD_WS;\nt CHILD_WS ::= \"\\t\"+;\nnt child ::= \"a\"+;\n";

    let parent = Constraint::from_glrm_grammar(parent_explicit, &vocab).unwrap();
    let child = Constraint::from_glrm_grammar(child_src, &vocab).unwrap();
    let composed_static = parent
        .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
        .unwrap();
    let composed_dynamic = parent
        .compose_linked_children_for_test_dynamic(&[("SUB", &child)], &vocab)
        .unwrap();
    let inline = compile_inline_source_oracle(parent_inline, &[("SUB", child_src)], &vocab);

    let mut static_state = composed_static.start();
    let mut dynamic_state = composed_dynamic.start();
    let mut inline_state = inline.start();

    let static_start = mb_scoped_mask_bits(&static_state.mask());
    let dynamic_start = mb_scoped_mask_bits(&dynamic_state.mask());
    let inline_start = mb_scoped_mask_bits(&inline_state.mask());

    // Both Xa! (token 0) and Xaa! (token 3) must be admitted; invalid tokens (1 and 4) must not
    assert!(static_start.contains(&0));
    assert!(static_start.contains(&3));
    assert!(!static_start.contains(&1));
    assert!(!static_start.contains(&4));

    assert_eq!(static_start, inline_start, "static start mask matches inline oracle");
    assert_eq!(dynamic_start, inline_start, "dynamic start mask matches inline oracle");

    // Commit separate prefix token 'X' (token 2)
    static_state.commit_token(2).unwrap();
    dynamic_state.commit_token(2).unwrap();
    inline_state.commit_token(2).unwrap();

    let static_after_x = mb_scoped_mask_bits(&static_state.mask());
    let dynamic_after_x = mb_scoped_mask_bits(&dynamic_state.mask());
    let inline_after_x = mb_scoped_mask_bits(&inline_state.mask());

    assert_eq!(static_after_x, inline_after_x, "static mask after prefix X matches inline");
    assert_eq!(dynamic_after_x, inline_after_x, "dynamic mask after prefix X matches inline");
}

/// Three-level nested scopes test: outer -> middle -> inner.
/// Documents whether 3-level composition succeeds or hits the known table entry collision.
#[test]
fn test_three_level_nested_scopes() {
    let vocab = Vocab::new(vec![
        (0, b"<".to_vec()),
        (1, b">".to_vec()),
        (2, b"[".to_vec()),
        (3, b"]".to_vec()),
        (4, b"a".to_vec()),
        (5, b" ".to_vec()),
    ]);
    let inner_src = "start inner;\nnt inner ::= \"a\";\n";
    let middle_src = "start middle;\nt INNER_SLOT ::= @token(1001);\nnt middle ::= \"[\" INNER_SLOT \"]\";\n";
    let outer_src = "start outer;\nt MID_SLOT ::= @token(1002);\nnt outer ::= \"<\" MID_SLOT \">\";\n";

    let inner = Constraint::from_glrm_grammar(inner_src, &vocab).unwrap();
    let middle = Constraint::from_glrm_grammar(middle_src, &vocab).unwrap();
    let outer = Constraint::from_glrm_grammar(outer_src, &vocab).unwrap();

    let middle_composed = middle.compose_linked_children_for_test(&[("INNER_SLOT", &inner)], &vocab);
    assert!(middle_composed.is_ok(), "level 1->2 composition succeeds");
    let middle_composed = middle_composed.unwrap();

    let outer_composed = outer.compose_linked_children_for_test(&[("MID_SLOT", &middle_composed)], &vocab);
    // Dynamic composition should succeed:
    let outer_dynamic = outer.compose_linked_children_for_test_dynamic(&[("MID_SLOT", &middle_composed)], &vocab);
    assert!(outer_dynamic.is_ok(), "three-level dynamic composition succeeds");

    match outer_composed {
        Ok(composed) => {
            let mut state = composed.start();
            assert!(state.commit_token(0).is_ok());
        }
        Err(err) => {
            eprintln!("[three-level-nested] static composition blocked by entry collision: {err}");
            let err_msg = err.to_string();
            assert!(
                err_msg.contains("subgrammar entry action collision")
                    || err_msg.contains("entry action collision"),
                "unexpected error in 3-level nesting: {err_msg}",
            );
        }
    }
}
