use super::*;

fn vocab() -> Vocab {
    Vocab::new(vec![
        (0, b"!".to_vec()),
        (1, b"x".to_vec()),
        (2, b";".to_vec()),
        (3, b"!x;".to_vec()),
        (7, b"!!".to_vec()),
        (11, b"x;".to_vec()),
        (17, b"!".to_vec()),
        (31, b"?".to_vec()),
        (40, b"x!".to_vec()),
    ])
}

fn source() -> &'static str {
    "start p; nt p ::= unary ';'; nt unary ::= 'x' | '!' unary;"
}

fn compile_dynamic(vocab: &Vocab) -> DynamicConstraint {
    // These fixtures exercise template ready views and the template envelope.
    // Select that representation explicitly so restoring ordinary Dynamic LR
    // does not change the subject or require a process-global environment switch.
    let native = crate::Grammar::glrm(source()).compile_with(
        vocab, crate::BuildOptions::default().optimization(crate::Optimization::FastBuild),
    ).unwrap();
    DynamicConstraint::from_constraints(vec![native])
}

fn old_envelope(dynamic: &DynamicConstraint, external: bool) -> Vec<u8> {
    let mut payload = Vec::new();
    let count = dynamic.alternatives.len() + 1;
    payload.extend_from_slice(&(count as u32).to_le_bytes());
    for constraint in std::iter::once(&dynamic.inner).chain(&dynamic.alternatives) {
        let body = if external {
            constraint.save_template_with_external_vocab()
        } else {
            constraint.save()
        };
        payload.extend_from_slice(&(body.len() as u64).to_le_bytes());
        payload.extend_from_slice(&body);
    }

    let mut bytes = Vec::new();
    bytes.extend_from_slice(if external {
        &DYNAMIC_TRANSFER_MAGIC
    } else {
        &DYNAMIC_CONSTRAINT_MAGIC
    });
    bytes.extend_from_slice(
        &(if external {
            TEMPLATE_DYNAMIC_TRANSFER_VERSION
        } else {
            TEMPLATE_DYNAMIC_CONSTRAINT_VERSION
        })
        .to_le_bytes(),
    );
    bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&payload);
    bytes
}

fn byte_oracle(bytes: &[u8]) -> (bool, bool) {
    let mut position = 0usize;
    while bytes.get(position) == Some(&b'!') {
        position += 1;
    }
    if position == bytes.len() {
        return (true, false);
    }
    if bytes.get(position) != Some(&b'x') {
        return (false, false);
    }
    position += 1;
    if position == bytes.len() {
        return (true, false);
    }
    if bytes.get(position) != Some(&b';') {
        return (false, false);
    }
    position += 1;
    (position == bytes.len(), position == bytes.len())
}

fn expected_mask(vocab: &Vocab, prefix: &[u8], words: usize) -> Vec<u32> {
    let mut mask = vec![0u32; words];
    for (id, bytes) in vocab.iter() {
        let mut extension = prefix.to_vec();
        extension.extend_from_slice(bytes);
        if byte_oracle(&extension).0 {
            mask[id as usize / 32] |= 1 << (id % 32);
        }
    }
    mask
}

#[test]
fn ready_adapter_preserves_template_and_runtime_arc_identity() {
    let vocab = vocab();
    let compiled = compile_dynamic(&vocab);

    let named = crate::grammar::glrm::from_glrm(source()).unwrap();
    let lowered = crate::grammar::ast::lower(
        &crate::grammar::factoring::factor_named_grammar(named),
    )
    .unwrap();
    let grammar =
        crate::compiler::grammar::transforms::prepare_dynamic_glr_transforms_only(lowered);
    let analyzed = crate::compiler::glr::analysis::AnalyzedGrammar::from_grammar_def(&grammar);
    let table = GLRTable::build_with_default_construction(
        &analyzed,
        crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
    );

    let prepared =
        crate::runtime::parser_backend::PreparedTemplateParser::from_compiler_parts(
            &table,
            None,
            &[],
            grammar.ignore_terminal,
            true,
            false,
        )
        .unwrap();

    let templates = prepared.templates.clone();
    let runtime = prepared.runtime.clone();
    let completion = Arc::clone(&prepared.parser.completion_template);
    drop(table);
    drop(analyzed);

    let result = DynamicConstraint::from_template_runtime_parts_unfinalized_with_views(
        compiled.inner.tokenizer.as_ref().clone(),
        compiled.inner.terminal_display_names.clone(),
        compiled.inner.ignore_terminal,
        prepared.templates,
        Arc::new(prepared.parser),
        &vocab,
        compiled.inner.dynamic_mask_vocab.clone(),
        Some(prepared.runtime),
    );

    assert!(!result.table.is_present());
    assert!(Arc::ptr_eq(
        &completion,
        &result.template_parser.as_ref().unwrap().completion_template,
    ));
    for (before, after) in templates.iter().zip(&result.template_dfas_by_terminal) {
        match (before, after) {
            (Some(before), Some(after)) => assert!(Arc::ptr_eq(before, after)),
            (None, None) => {}
            _ => panic!("template presence changed during installation"),
        }
    }
    for (before, after) in runtime.iter().zip(&result.fast_template_dfas_by_terminal) {
        match (before, after) {
            (Some(before), Some(after)) => assert!(Arc::ptr_eq(before, after)),
            (None, None) => {}
            _ => panic!("runtime presence changed during installation"),
        }
    }
}

#[test]
fn dynamic_envelope_is_byte_identical_to_previous_assembly() {
    let vocab = vocab();
    let one = compile_dynamic(&vocab);
    let many = DynamicConstraint::from_alternatives(vec![one.clone(), one.clone()]);
    for dynamic in [&one, &many] {
        for external in [false, true] {
            assert_eq!(
                dynamic.save_template_alternatives(external),
                old_envelope(dynamic, external),
            );
        }
    }
}

#[test]
fn uniquely_owned_cached_transfer_is_moved_by_into_saved() {
    let vocab = vocab();
    let mut dynamic = compile_dynamic(&vocab);
    dynamic.cache_external_vocab_artifact_for_save();

    let cached = dynamic.external_vocab_artifact_cache.as_ref().unwrap();
    assert_eq!(Arc::strong_count(cached), 1);
    let pointer = cached.as_ref().as_ptr();
    let expected = cached.as_ref().clone();

    let saved = dynamic.into_saved();
    assert_eq!(saved.as_ptr(), pointer);
    assert_eq!(saved, expected);
    let restored = DynamicConstraint::load_with_vocab(&saved, &vocab).unwrap();
    assert!(restored.inner.has_template_parser());
    assert!(!restored.inner.table.is_present());
}

#[test]
fn shared_cached_transfer_is_copied_without_invalidating_the_other_owner() {
    let vocab = vocab();
    let mut dynamic = compile_dynamic(&vocab);
    dynamic.cache_external_vocab_artifact_for_save();
    let other = dynamic.clone();
    let expected = other.save_with_external_vocab();
    let saved = dynamic.into_saved();
    assert_eq!(saved, expected);
    assert_eq!(other.save_with_external_vocab(), expected);
}

#[test]
fn full_masks_commits_rejections_and_completion_survive_all_dynamic_formats() {
    let vocab = vocab();
    let direct = compile_dynamic(&vocab);
    let self_contained = DynamicConstraint::load(&direct.save()).unwrap();
    let external =
        DynamicConstraint::load_with_vocab(&direct.save_with_external_vocab(), &vocab)
            .unwrap();

    for constraint in [&direct, &self_contained, &external] {
        assert!(constraint.inner.has_template_parser());
        assert!(!constraint.inner.table.is_present());

        for depth in [0usize, 1, 17, 257, 1024] {
            let mut prefix = vec![b'!'; depth];
            for suffix in [b"".as_slice(), b"x", b"x;"] {
                prefix.truncate(depth);
                prefix.extend_from_slice(suffix);
                let mut state = constraint.start();
                state.commit_bytes(&prefix).unwrap();

                let expected = expected_mask(&vocab, &prefix, constraint.mask_len());
                let mut actual = vec![0u32; constraint.mask_len()];
                state.fill_mask(&mut actual);
                assert_eq!(actual, expected, "prefix length {}", prefix.len());
                assert_eq!(state.is_accepting(), byte_oracle(&prefix).1);

                for (id, bytes) in vocab.iter() {
                    let mut extension = prefix.clone();
                    extension.extend_from_slice(bytes);
                    let (admitted, accepting) = byte_oracle(&extension);
                    let mut branch = state.clone();
                    let result = branch.commit_token(id);
                    assert_eq!(result.is_ok(), admitted, "token {id}");
                    if admitted {
                        assert_eq!(branch.is_accepting(), accepting);
                        let mut after = vec![0u32; constraint.mask_len()];
                        branch.fill_mask(&mut after);
                        assert_eq!(
                            after,
                            expected_mask(&vocab, &extension, constraint.mask_len()),
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn current_dynamic_envelope_rejects_truncation_and_trailing_data() {
    let vocab = vocab();
    let dynamic = compile_dynamic(&vocab);
    let bytes = dynamic.save_with_external_vocab();

    for length in [0usize, 1, 17, bytes.len() - 1] {
        assert!(DynamicConstraint::load_with_vocab(&bytes[..length], &vocab).is_err());
    }

    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(DynamicConstraint::load_with_vocab(&trailing, &vocab).is_err());
}

#[test]
fn canonical_and_static_public_routes_keep_complete_mask_words() {
    let vocab = vocab();
    for optimization in [
        crate::Optimization::FastBuild,
        crate::Optimization::FastRuntime,
        crate::Optimization::Auto,
    ] {
        let direct = crate::Grammar::glrm(source())
            .compile_with(
                &vocab,
                crate::BuildOptions::default().optimization(optimization),
            )
            .unwrap();
        let loaded = crate::Constraint::load(direct.save()).unwrap();
        let external = crate::Constraint::load_with_vocab(
            direct.save_with_external_vocab().unwrap(),
            &vocab,
        )
        .unwrap();

        for constraint in [&direct, &loaded, &external] {
            assert_eq!(
                constraint.parser_backend(),
                crate::ParserBackend::TemplateDfa,
            );
            for prefix in [b"".as_slice(), b"!!", b"!!x", b"!!x;"] {
                let mut state = constraint.start();
                state.commit_bytes(prefix).unwrap();
                assert_eq!(
                    state.mask(),
                    expected_mask(&vocab, prefix, constraint.mask_len()),
                );
                assert_eq!(state.is_accepting(), byte_oracle(prefix).1);
            }
        }
    }
}
