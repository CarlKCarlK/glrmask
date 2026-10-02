#![cfg(feature = "internal-api")]
//! Derived caches must preserve exact masks, cloned artifacts and serialized
//! bytes for both parser backends. Cache misses do not authorize a token.
use glrmask::__private::{assert_packed_final_mask_cache, assert_tokenizer_transition_cache};
use glrmask::{BuildOptions, Constraint, Grammar, Optimization, ParserBackend, Vocab};

fn vocabulary() -> Vocab {
    let mut words = (32u8..=126).map(|b| vec![b]).collect::<Vec<_>>();
    words.extend([
        b"\n".to_vec(),
        b"true".to_vec(),
        b"false".to_vec(),
        b"null".to_vec(),
    ]);
    for i in 0..128 {
        words.push(format!("\"field{i}\"").into_bytes());
    }
    for i in 0..128 {
        words.push(format!("\"value{i}\"").into_bytes());
    }
    words.extend([
        b"{}".to_vec(),
        b"[]".to_vec(),
        b"\":\"".to_vec(),
        b"\",\"".to_vec(),
    ]);
    Vocab::new(
        words
            .into_iter()
            .enumerate()
            .map(|(i, w)| (i as u32, w))
            .collect(),
    )
}

fn check_prefixes(reference: &Constraint, candidate: &Constraint) {
    assert_eq!(reference.mask_len(), candidate.mask_len());
    let cases = [
        "",
        "{",
        "{\"field0\"",
        "{\"field0\":",
        "{\"field0\":\"value7\"}",
        "[]",
        "[\"value11\",true,null]",
        "\"value7\"",
        "\"valu",
        "true",
        " false ",
        "{\"unknown\":123}",
        "[[]]",
        "{}",
        "[123,\"x\"]",
        "\\",
        "[}",
    ];
    for prefix in cases {
        let mut a = reference.start();
        let mut b = candidate.start();
        let mut am = vec![0; reference.mask_len()];
        let mut bm = am.clone();
        for byte in prefix.bytes().map(Some).chain([None]) {
            a.fill_mask(&mut am);
            b.fill_mask(&mut bm);
            assert_eq!(am, bm, "mask before {byte:?} in {prefix:?}");
            assert_eq!(a.is_accepting(), b.is_accepting());
            assert_eq!(a.is_rejected(), b.is_rejected());
            let Some(byte) = byte else {
                break;
            };
            assert_eq!(
                a.commit_bytes(&[byte]).is_ok(),
                b.commit_bytes(&[byte]).is_ok()
            );
            if a.is_rejected() {
                break;
            }
        }
    }
}

#[test]
fn packed_final_cache_preserves_wire_masks_and_cloned_loads() {
    let vocab = vocabulary();
    let enums = (0..128).map(|i| format!("value{i}")).collect::<Vec<_>>();
    let schemas = [
        serde_json::json!({"type":"object","additionalProperties":false,
        "properties":{"field0":{"enum":enums},"field1":{"type":"integer"},
            "field2":{"type":"array","items":{"type":"boolean"}}}}),
        serde_json::json!({"type":"array","items":{"anyOf":[{"type":"string"},
            {"type":"integer"},{"type":"null"},{"type":"boolean"}]}}),
        serde_json::json!({"type":"string","pattern":"^[a-z0-9]+$","minLength":1,"maxLength":16}),
    ];
    let mut total_checks = 0;
    let mut transition_checks = 0;
    for schema in schemas {
        let source = schema.to_string();
        for optimization in [Optimization::FastRuntime, Optimization::FastBuild] {
            for explicit_backend in [false, true] {
                let backend = ParserBackend::TemplateDfa;
                let options = BuildOptions::default().optimization(optimization);
                let options = if explicit_backend { options.parser_backend(backend) } else { options };
                let fresh = Grammar::from_json_schema(&source)
                    .compile_with(&vocab, options)
                    .unwrap();
                let bytes = fresh.save();
                let loaded = Constraint::load(&bytes).unwrap();
                let cloned = loaded.clone();
                assert_eq!(cloned.parser_backend(), backend);
                transition_checks += assert_tokenizer_transition_cache(&fresh);
                transition_checks += assert_tokenizer_transition_cache(&loaded);
                transition_checks += assert_tokenizer_transition_cache(&cloned);
                total_checks += assert_packed_final_mask_cache(&loaded, 32);
                total_checks += assert_packed_final_mask_cache(&cloned, 32);
                check_prefixes(&fresh, &loaded);
                check_prefixes(&fresh, &cloned);
                assert_eq!(
                    loaded.save(),
                    bytes,
                    "derived caches must not alter wire bytes"
                );
                assert_eq!(cloned.save(), bytes);
                if backend == ParserBackend::TemplateDfa {
                    let external = fresh.save_with_external_vocab().unwrap();
                    let externally_loaded = Constraint::load_with_vocab(&external, &vocab).unwrap();
                    transition_checks += assert_tokenizer_transition_cache(&externally_loaded);
                    total_checks += assert_packed_final_mask_cache(&externally_loaded, 32);
                    check_prefixes(&fresh, &externally_loaded);
                    assert_eq!(
                        externally_loaded.save_with_external_vocab().unwrap(),
                        external
                    );
                }
            }
        }
    }
    assert!(
        total_checks > 100,
        "test fixture must exercise nonempty packed indices and masks"
    );
    assert!(
        transition_checks > 256,
        "fixture must exercise cached byte transitions"
    );
}

#[test]
fn loaded_exact_transition_cache_crosses_the_old_8192_state_cutoff() {
    // A long nonperiodic literal produces distinct physical lexer positions.
    // It exercises the load-time acceleration above the former 8192-row cut,
    // without requiring a large model vocabulary or persisting giant masks.
    let literal = (0usize..8_300)
        .map(|i| {
            if (i.wrapping_mul(0x9e37) ^ (i >> 3)).count_ones() & 1 == 0 {
                'a'
            } else {
                'b'
            }
        })
        .collect::<String>();
    let source = format!("start root; nt root ::= {literal:?};");
    let vocab = Vocab::new(vec![
        (0, b"a".to_vec()),
        (1, b"b".to_vec()),
        (2, b"x".to_vec()),
    ]);
    for explicit_backend in [false, true] {
        let backend = ParserBackend::TemplateDfa;
        let options = BuildOptions::default().optimization(Optimization::FastBuild);
        let options = if explicit_backend { options.parser_backend(backend) } else { options };
        let fresh = Grammar::from_glrm(&source)
            .compile_with(&vocab, options)
            .unwrap();
        let bytes = fresh.save();
        let loaded = Constraint::load(&bytes).unwrap();
        let fresh_checks = assert_tokenizer_transition_cache(&fresh);
        let loaded_checks = assert_tokenizer_transition_cache(&loaded);
        assert!(
            fresh_checks > 8_192 * 256,
            "fixture must exceed the old cutoff"
        );
        assert_eq!(
            loaded_checks, fresh_checks,
            "loaded tokenizer must retain exact acceleration"
        );
        assert_eq!(loaded.parser_backend(), backend);
        let mut left = fresh.start();
        let mut right = loaded.start();
        left.commit_bytes(literal.as_bytes()).unwrap();
        right.commit_bytes(literal.as_bytes()).unwrap();
        assert_eq!(left.is_accepting(), right.is_accepting());
        assert!(left.is_accepting());
        let mut a = vec![0; fresh.mask_len()];
        let mut b = vec![0; loaded.mask_len()];
        left.fill_mask(&mut a);
        right.fill_mask(&mut b);
        assert_eq!(a, b);
        assert_eq!(loaded.save(), bytes);
    }
}
