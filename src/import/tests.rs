use super::*;
use crate::Vocab;
use crate::compiler::glr::table::{AdmissionPolicy, GlrTableConstruction};

#[cfg(windows)]
#[test]
fn large_imports_use_dedicated_windows_stack() {
    let thread_name = with_large_import_stack(LARGE_IMPORT_SOURCE_BYTES, || {
        std::thread::current().name().map(str::to_owned)
    });
    assert_eq!(thread_name.as_deref(), Some("glrmask-grammar-compile"));
}

fn vocab(entries: &[&str]) -> Vocab {
    Vocab::new(
        entries
            .iter()
            .enumerate()
            .map(|(id, text)| (id as u32, text.as_bytes().to_vec()))
            .collect(),
    )
}

fn accepts_bytes(constraint: &Constraint, bytes: &[u8]) -> bool {
    let mut state = constraint.start();
    state.commit_bytes(bytes).is_ok() && state.is_accepting()
}

#[test]
#[allow(deprecated)]
fn programmatic_json_constructors_are_unsupported() {
    let vocab = vocab(&["x"]);
    let dynamic = Constraint::from_glrm_grammar(
        "start dynamic; t IDENT ::= /[A-Za-z_$][A-Za-z0-9_$]*/; nt dynamic ::= IDENT;",
        &vocab,
    )
    .unwrap();
    let condition = dynamic.clone();

    let error = Constraint::from_json_schema_with_programmatic_values(
        "this is not json",
        &dynamic,
        &condition,
        &vocab,
    )
    .err()
    .expect("programmatic JSON schema values must be unsupported");
    match error {
        crate::GlrMaskError::Compilation(message) => assert!(
            message.contains("programmatic JSON schema values are unsupported"),
            "unexpected message: {message}",
        ),
        other => panic!("expected a Compilation error, got {other:?}"),
    }

    let error =
        Constraint::from_json_schema_with_dynamic_value("this is not json", &dynamic, &vocab)
            .err()
            .expect("dynamic-value JSON schema must be unsupported");
    match error {
        crate::GlrMaskError::Compilation(message) => assert!(
            message.contains("programmatic JSON schema values are unsupported"),
            "unexpected message: {message}",
        ),
        other => panic!("expected a Compilation error, got {other:?}"),
    }

    let error = Constraint::from_json_schema_with_dynamic_value_and_end_tokens(
        "this is not json",
        &dynamic,
        &vocab,
        &[],
    )
    .err()
    .expect("dynamic-value JSON schema with end tokens must be unsupported");
    match error {
        crate::GlrMaskError::Compilation(message) => assert!(
            message.contains("programmatic JSON schema values are unsupported"),
            "unexpected message: {message}",
        ),
        other => panic!("expected a Compilation error, got {other:?}"),
    }
}

#[test]
fn vocab_partition_json_seeded_ast_lower_matches_ordinary_lower() {
    let schemas = [
        r#"{"type":["null","boolean"]}"#,
        r#"{
                "type":"object",
                "properties":{
                    "code":{
                        "type":"string",
                        "pattern":"^[A-Z0-9]+$",
                        "minLength":2,
                        "maxLength":64
                    }
                },
                "required":["code"],
                "additionalProperties":false
            }"#,
    ];

    for schema in schemas {
        let named = parse_json_schema_to_named_dynamic(schema).unwrap();
        let mut ordinary = factor_named_grammar(named.clone());
        prepare_json_schema_named(&mut ordinary).unwrap();
        let ordinary = ast::lower(&ordinary).unwrap();

        let mut seeded = factor_named_grammar(named);
        let resolved = json_schema::prepare_named_grammar_for_lowering(&mut seeded).unwrap();
        let seeded = ast::lower_with_resolved_terminal_exprs(&seeded, resolved).unwrap();

        assert_eq!(
            bincode::serialize(&ordinary).unwrap(),
            bincode::serialize(&seeded).unwrap(),
            "seeded lowering changed GrammarDef for schema {schema}",
        );
    }
}

#[test]
#[ignore = "Programmatic JSON schema API is intentionally unsupported"]
#[allow(deprecated)]
fn json_schema_dynamic_value_is_nested_but_not_root_escape() {
    let vocab = vocab(&["x", "{", "}", "\"name\"", ": ", "123"]);
    let dynamic = Constraint::from_glrm_grammar(
        "start dynamic; t IDENT ::= /[A-Za-z_$][A-Za-z0-9_$]*/; nt dynamic ::= IDENT;",
        &vocab,
    )
    .unwrap();
    let schema = r#"{
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": false
        }"#;
    let constraint =
        Constraint::from_json_schema_with_dynamic_value(schema, &dynamic, &vocab).unwrap();

    assert!(accepts_bytes(&constraint, br#"{"name": "literal"}"#));
    assert!(accepts_bytes(&constraint, br#"{"name": x}"#));
    assert!(!accepts_bytes(&constraint, b"x"));
    assert!(!accepts_bytes(&constraint, br#"{"name": 123}"#));
    assert!(!accepts_bytes(&constraint, br#"{"wrong": x}"#));
}

#[test]
#[ignore = "Programmatic JSON schema API is intentionally unsupported"]
#[allow(deprecated)]
fn json_schema_dynamic_value_applies_to_array_items() {
    let vocab = vocab(&["x", "[", "]", ", ", "1"]);
    let dynamic = Constraint::from_glrm_grammar(
        "start dynamic; t IDENT ::= /[A-Za-z_$][A-Za-z0-9_$]*/; nt dynamic ::= IDENT;",
        &vocab,
    )
    .unwrap();
    let schema = r#"{"type":"array","items":{"type":"integer"}}"#;
    let constraint =
        Constraint::from_json_schema_with_dynamic_value(schema, &dynamic, &vocab).unwrap();

    assert!(accepts_bytes(&constraint, b"[1, x]"));
    assert!(!accepts_bytes(&constraint, b"x"));
    assert!(!accepts_bytes(&constraint, br#"["wrong"]"#));
}

#[test]
#[ignore = "Programmatic JSON schema API is intentionally unsupported"]
#[allow(deprecated)]
fn json_schema_dynamic_value_allows_runtime_enum_but_rejects_bad_literal() {
    let vocab = vocab(&[
        "{",
        "}",
        "\"status\"",
        ": ",
        "\"open\"",
        "\"closed\"",
        "\"bogus\"",
        "result",
        ".status",
    ]);
    let dynamic =
        Constraint::from_glrm_grammar("start expr; nt expr ::= 'result' '.status';", &vocab)
            .unwrap();
    let schema = r#"{
          "type":"object",
          "properties":{"status":{"enum":["open","closed"]}},
          "required":["status"],
          "additionalProperties":false
        }"#;
    let constraint =
        Constraint::from_json_schema_with_dynamic_value(schema, &dynamic, &vocab).unwrap();
    assert!(accepts_bytes(&constraint, br#"{"status": "open"}"#));
    assert!(!accepts_bytes(&constraint, br#"{"status": "bogus"}"#));
    assert!(accepts_bytes(&constraint, br#"{"status": result.status}"#));
}

#[test]
#[ignore = "Programmatic JSON schema API is intentionally unsupported"]
#[allow(deprecated)]
fn json_schema_dynamic_value_allows_runtime_const_but_rejects_bad_literal() {
    let vocab = vocab(&["{", "}", "\"kind\"", ": ", "\"fixed\"", "\"wrong\"", "result", ".kind"]);
    let dynamic =
        Constraint::from_glrm_grammar("start expr; nt expr ::= 'result' '.kind';", &vocab).unwrap();
    let schema = r#"{
          "type":"object",
          "properties":{"kind":{"const":"fixed"}},
          "required":["kind"],
          "additionalProperties":false
        }"#;
    let constraint =
        Constraint::from_json_schema_with_dynamic_value(schema, &dynamic, &vocab).unwrap();
    assert!(accepts_bytes(&constraint, br#"{"kind": "fixed"}"#));
    assert!(!accepts_bytes(&constraint, br#"{"kind": "wrong"}"#));
    assert!(accepts_bytes(&constraint, br#"{"kind": result.kind}"#));
}

#[test]
fn json_schema_import_uses_legacy_row_bisim_table_by_default() {
    let constraint =
        Constraint::from_json_schema(r#"{"type":"string"}"#, &vocab(&["\"", "a", "\"a\""]))
            .unwrap();

    assert_eq!(constraint.table.construction, GlrTableConstruction::LegacyRowBisim);
    assert_eq!(constraint.table.admission_policy, AdmissionPolicy::RowPresenceExact);
}

fn token_allowed(mask: &[u32], token_id: u32) -> bool {
    mask.get(token_id as usize / 32).is_some_and(|word| word & (1u32 << (token_id % 32)) != 0)
}

#[test]
fn json_schema_end_tokens_are_exact_parser_terminals() {
    let vocab = vocab(&["\"", "a", "\"a\""]);
    let constraint =
        Constraint::from_json_schema_with_end_tokens(r#"{"const":"a"}"#, &vocab, &[101, 100, 101])
            .unwrap();
    assert_eq!(constraint.mask_len(), 4);

    let mut state = constraint.start();
    assert!(!token_allowed(&state.mask(), 100));
    assert!(!token_allowed(&state.mask(), 101));
    state.commit_token(2).unwrap();
    assert!(!state.is_accepting());
    let mask = state.mask();
    assert!(token_allowed(&mask, 100));
    assert!(token_allowed(&mask, 101));
    assert_eq!(state.forced(), Vec::<u32>::new());
    state.commit_token(100).unwrap();
    assert!(state.is_accepting());
}

#[test]
fn json_schema_single_end_token_is_forced() {
    let vocab = vocab(&["\"", "a", "\"a\""]);
    let constraint =
        Constraint::from_json_schema_with_end_tokens(r#"{"const":"a"}"#, &vocab, &[100]).unwrap();

    let mut state = constraint.start();
    state.commit_token(2).unwrap();
    assert_eq!(state.forced(), vec![100]);
    state.commit_token(100).unwrap();
    assert!(state.is_accepting());
}

#[test]
fn json_schema_end_token_can_also_have_byte_semantics() {
    let vocab = Vocab::new(vec![(100, b"\"a\"".to_vec())]);
    let constraint =
        Constraint::from_json_schema_with_end_tokens(r#"{"const":"a"}"#, &vocab, &[100]).unwrap();

    let mut state = constraint.start();
    assert_eq!(state.forced(), vec![100, 100]);
    state.commit_token(100).unwrap();
    assert!(!state.is_accepting());
    assert_eq!(state.forced(), vec![100]);
    state.commit_token(100).unwrap();
    assert!(state.is_accepting());
}

#[test]
fn caller_sized_masks_zero_unknown_trailing_tokens() {
    let vocab = vocab(&["\"a\""]);
    let constraint =
        Constraint::from_json_schema_with_end_tokens(r#"{"const":"a"}"#, &vocab, &[100]).unwrap();
    let state = constraint.start();
    let mut oversized = vec![u32::MAX; constraint.mask_len() + 3];
    state.fill_mask(&mut oversized);
    assert!(oversized[constraint.mask_len()..].iter().all(|&word| word == 0));
    oversized.fill(u32::MAX);
    state.fill_mask(&mut oversized);
    assert!(oversized[constraint.mask_len()..].iter().all(|&word| word == 0));

    let dynamic =
        DynamicConstraint::from_json_schema_with_end_tokens(r#"{"const":"a"}"#, &vocab, &[100])
            .unwrap();
    let mut dynamic_mask = vec![u32::MAX; dynamic.mask_len() + 3];
    let dynamic_state = dynamic.start();
    dynamic_state.fill_mask(&mut dynamic_mask);
    assert!(dynamic_mask[dynamic.mask_len()..].iter().all(|&word| word == 0));
    dynamic_mask.fill(u32::MAX);
    dynamic_state.fill_mask(&mut dynamic_mask);
    assert!(dynamic_mask[dynamic.mask_len()..].iter().all(|&word| word == 0));
}

#[test]
fn glrm_import_uses_core_merged_table_by_default() {
    let constraint = Constraint::from_glrm_grammar(
        "start start;\nt A ::= 'a' ;\nnt start ::= A ;\n",
        &vocab(&["a"]),
    )
    .unwrap();

    assert_eq!(constraint.table.construction, GlrTableConstruction::ExperimentalCoreMerged);
    assert_eq!(constraint.table.admission_policy, AdmissionPolicy::ExactSimulation);
}

#[test]
fn glrm_import_merges_partially_mapped_terminal_families() {
    let mut entries = (0u32..=255).map(|byte| (byte, vec![byte as u8])).collect::<Vec<_>>();
    entries.extend([
        (256, b"{\"value\": ".to_vec()),
        (257, b"left".to_vec()),
        (258, b"right".to_vec()),
        (259, b"}".to_vec()),
    ]);
    let vocab = Vocab::new(entries);

    Constraint::from_glrm_grammar(
        r#"
                start document;
                t LEFT ::= @token(1000000);
                t RIGHT ::= @token(1000001);
                nt document ::= "{" "\"value\": " (LEFT | RIGHT) "}";
            "#,
        &vocab,
    )
    .expect("partial terminal-family maps should merge without indexing the unmapped sentinel");
}

#[test]
fn glrm_uniform_subgrammar_ignore_uses_global_terminal_dwa_path() {
    let vocab = vocab(&[" ", "<", ">", "a", "b"]);
    let constraint = Constraint::from_glrm_grammar(
        r#"
                start document;
                ignore OUTER_WS;
                t OUTER_WS ::= " "+;

                g inner ::= {
                    start value;
                    ignore INNER_WS;
                    t INNER_WS ::= " "+;
                    nt value ::= "a" "b";
                };

                nt document ::= "<" inner ">";
            "#,
        &vocab,
    )
    .unwrap();

    assert!(
        constraint.ignore_terminal.is_some(),
        "uniform scoped ignore should remain a global ignore terminal",
    );
    let mut state = constraint.start();
    state.commit_bytes(b"  < a   b >  ").unwrap();
    assert!(state.is_accepting());
}

#[test]
fn glrm_nested_uniform_subgrammar_ignore_uses_global_terminal_dwa_path() {
    let vocab = vocab(&[" ", "<", ">", "[", "]", "x"]);
    let constraint = Constraint::from_glrm_grammar(
        r#"
                start document;
                ignore ROOT_WS;
                t ROOT_WS ::= " "+;

                g middle ::= {
                    start wrapped;
                    ignore MIDDLE_WS;
                    t MIDDLE_WS ::= " "+;

                    g leaf ::= {
                        start value;
                        ignore LEAF_WS;
                        t LEAF_WS ::= " "+;
                        nt value ::= "x";
                    };

                    nt wrapped ::= "[" leaf "]";
                };

                nt document ::= "<" middle ">";
            "#,
        &vocab,
    )
    .unwrap();

    assert!(constraint.ignore_terminal.is_some());
    let mut state = constraint.start();
    state.commit_bytes(b" < [ x ] > ").unwrap();
    assert!(state.is_accepting());
}

#[test]
fn glrm_mixed_subgrammar_ignore_keeps_scoped_grammar_lowering() {
    let vocab = vocab(&[" ", "\t", "<", ">", "a", "b"]);
    let constraint = Constraint::from_glrm_grammar(
        r#"
                start document;
                ignore OUTER_WS;
                t OUTER_WS ::= " "+;

                g inner ::= {
                    start value;
                    ignore INNER_WS;
                    t INNER_WS ::= "\t"+;
                    nt value ::= "a" "b";
                };

                nt document ::= "<" inner ">";
            "#,
        &vocab,
    )
    .unwrap();

    assert!(
        constraint.ignore_terminal.is_none(),
        "different scoped ignores must retain explicit scope-local lowering",
    );
    let mut state = constraint.start();
    state.commit_bytes(b" <\ta\t\tb\t> ").unwrap();
    assert!(state.is_accepting());
}

#[test]
fn glrm_child_without_ignore_keeps_scoped_grammar_lowering() {
    let vocab = vocab(&[" ", "<", ">", "a", "b"]);
    let constraint = Constraint::from_glrm_grammar(
        r#"
                start document;
                ignore OUTER_WS;
                t OUTER_WS ::= " "+;

                g inner ::= {
                    start value;
                    nt value ::= "a" "b";
                };

                nt document ::= "<" inner ">";
            "#,
        &vocab,
    )
    .unwrap();

    assert!(constraint.ignore_terminal.is_none());
    let mut state = constraint.start();
    state.commit_bytes(b" <ab> ").unwrap();
    assert!(state.is_accepting());
}

#[test]
fn glrm_external_subgrammar_api_matches_inline_grammar() {
    let vocab = vocab(&["X", "a", "b", "!", "Xa", "ab!", "Xab!"]);
    let child = Constraint::from_glrm_grammar(
        r#"
                start child;
                nt child ::= "a" "b";
            "#,
        &vocab,
    )
    .unwrap();
    let composed = Constraint::from_glrm_grammar_with_subgrammars(
        r#"
                start document;
                extern grammar payload;
                nt document ::= "X" payload "!";
            "#,
        &[("payload", &child)],
        &vocab,
    )
    .unwrap();
    let inline = Constraint::from_glrm_grammar(
        r#"
                start document;
                g payload ::= {
                    start child;
                    nt child ::= "a" "b";
                };
                nt document ::= "X" payload "!";
            "#,
        &vocab,
    )
    .unwrap();

    for sequence in [vec![6], vec![0, 5], vec![4, 2, 3]] {
        let mut actual = composed.start();
        let mut expected = inline.start();
        for token_id in sequence {
            assert_eq!(actual.mask(), expected.mask());
            actual.commit_token(token_id).unwrap();
            expected.commit_token(token_id).unwrap();
        }
        assert_eq!(actual.is_accepting(), expected.is_accepting());
        assert!(actual.is_accepting());
    }
}

#[test]
fn glrm_external_subgrammars_support_adjacent_reuse() {
    let vocab = vocab(&["X", "a", "!", "Xa", "aa!"]);
    let child = Constraint::from_glrm_grammar("start child; nt child ::= \"a\";", &vocab).unwrap();
    let composed = Constraint::from_glrm_grammar_with_subgrammars(
        r#"
                start document;
                extern grammar left;
                extern grammar right;
                nt document ::= "X" left right "!";
            "#,
        &[("left", &child), ("right", &child)],
        &vocab,
    )
    .unwrap();
    let inline = Constraint::from_glrm_grammar(
        r#"
                start document;
                nt child ::= "a";
                nt document ::= "X" child child "!";
            "#,
        &vocab,
    )
    .unwrap();

    let mut actual = composed.start();
    let mut expected = inline.start();
    for token_id in [3, 1, 2] {
        assert_eq!(actual.mask(), expected.mask());
        actual.commit_token(token_id).unwrap();
        expected.commit_token(token_id).unwrap();
    }
    assert!(actual.is_accepting());
    assert!(expected.is_accepting());
}

#[test]
fn glrm_external_subgrammars_support_qualified_nested_bindings() {
    let vocab = vocab(&["<", ">", "[", "]", "a", "<[a]>"]);
    let leaf = Constraint::from_glrm_grammar("start leaf; nt leaf ::= \"a\";", &vocab).unwrap();
    let composed = Constraint::from_glrm_grammar_with_subgrammars(
        r#"
                start document;
                g wrapper ::= {
                    start value;
                    extern grammar leaf;
                    nt value ::= "[" leaf "]";
                };
                nt document ::= "<" wrapper ">";
            "#,
        &[("wrapper::leaf", &leaf)],
        &vocab,
    )
    .unwrap();

    let mut state = composed.start();
    state.commit_token(5).unwrap();
    assert!(state.is_accepting());
}

#[test]
fn glrm_external_subgrammar_api_preserves_scoped_ignores() {
    let vocab = vocab(&[" ", "\t", "<", ">", "a", "b"]);
    let child = Constraint::from_glrm_grammar(
        r#"
                start value;
                ignore CHILD_WS;
                t CHILD_WS ::= "\t"+;
                nt value ::= "a" "b";
            "#,
        &vocab,
    )
    .unwrap();
    let composed = Constraint::from_glrm_grammar_with_subgrammars(
        r#"
                start document;
                ignore PARENT_WS;
                t PARENT_WS ::= " "+;
                extern grammar child;
                nt document ::= "<" child ">";
            "#,
        &[("child", &child)],
        &vocab,
    )
    .unwrap();
    let inline = Constraint::from_glrm_grammar(
        r#"
                start document;
                ignore PARENT_WS;
                t PARENT_WS ::= " "+;
                g child ::= {
                    start value;
                    ignore CHILD_WS;
                    t CHILD_WS ::= "\t"+;
                    nt value ::= "a" "b";
                };
                nt document ::= "<" child ">";
            "#,
        &vocab,
    )
    .unwrap();

    let bytes = b" <\ta\t\tb\t> ";
    let mut actual = composed.start();
    let mut expected = inline.start();
    actual.commit_bytes(bytes).unwrap();
    expected.commit_bytes(bytes).unwrap();
    assert!(actual.is_accepting());
    assert!(expected.is_accepting());
}

#[test]
fn glrm_external_subgrammar_api_retains_missing_slot_and_rejects_invalid_bindings() {
    let vocab = vocab(&["a"]);
    let child = Constraint::from_glrm_grammar("start child; nt child ::= \"a\";", &vocab).unwrap();
    let source = "start document; extern grammar child; nt document ::= child;";

    let unresolved = Constraint::from_glrm_grammar_with_subgrammars(source, &[], &vocab)
        .expect("an unresolved extern grammar is a valid late-binding slot");
    assert_eq!(unresolved.late_grammar_slots.len(), 1);
    assert_eq!(unresolved.late_grammar_slots[0].name, "child");
    let late_bound = unresolved
        .bind_grammar("child", child.clone())
        .expect("the retained external slot must remain bindable");
    let mut state = late_bound.start();
    state.commit_token(0).unwrap();
    assert!(state.is_accepting());

    let duplicate = Constraint::from_glrm_grammar_with_subgrammars(
        source,
        &[("child", &child), ("child", &child)],
        &vocab,
    )
    .unwrap_err()
    .to_string();
    assert!(duplicate.contains("more than once"), "{duplicate}");

    let unknown = Constraint::from_glrm_grammar_with_subgrammars(
        source,
        &[("child", &child), ("other", &child)],
        &vocab,
    )
    .unwrap_err()
    .to_string();
    assert!(unknown.contains("unknown external"), "{unknown}");
}

#[test]
fn glrm_external_subgrammar_allocator_avoids_child_special_tokens() {
    let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"!".to_vec())]);
    let child = Constraint::from_glrm_grammar(
        r#"
                start child;
                t MARK ::= @token(2);
                nt child ::= "a" MARK;
            "#,
        &vocab,
    )
    .unwrap();
    let composed = Constraint::from_glrm_grammar_with_subgrammars(
        r#"
                start document;
                extern grammar child;
                nt document ::= child "!";
            "#,
        &[("child", &child)],
        &vocab,
    )
    .unwrap();

    let mut state = composed.start();
    state.commit_token(0).unwrap();
    state.commit_token(2).unwrap();
    state.commit_token(1).unwrap();
    assert!(state.is_accepting());
}

#[test]
fn external_placeholder_allocator_finds_hole_below_u32_max() {
    let vocab = Vocab::new(vec![(0, b"a".to_vec()), (u32::MAX, b"z".to_vec())]);
    assert_eq!(first_external_placeholder_token_id(&vocab).unwrap(), 1);
}

#[test]
fn typed_glrm_api_without_externals_is_the_normal_compile_path() {
    let vocab = vocab(&["a"]);
    let source = "start document; nt document ::= \"a\";";
    let typed = Constraint::from_glrm_grammar_with_subgrammars(source, &[], &vocab).unwrap();
    let normal = Constraint::from_glrm_grammar(source, &vocab).unwrap();
    assert_eq!(typed.start().mask(), normal.start().mask());
}

#[test]
fn glrm_external_subgrammar_placeholders_avoid_end_tokens() {
    let vocab = Vocab::new(vec![(0, b"a".to_vec())]);
    let child = Constraint::from_glrm_grammar("start child; nt child ::= \"a\";", &vocab).unwrap();
    let composed = Constraint::from_glrm_grammar_with_subgrammars_and_end_tokens(
        "start document; extern grammar child; nt document ::= child;",
        &[("child", &child)],
        &vocab,
        &[1],
    )
    .unwrap();
    assert!(
        composed.table.control_terminals.is_empty(),
        "a child followed directly by an end token must compile linker controls away",
    );
    let loaded = Constraint::load(&composed.save()).unwrap();

    for constraint in [&composed, &loaded] {
        let mut state = constraint.start();
        state.commit_token(0).unwrap();
        for _ in 0..2 {
            let mask = state.mask();
            assert_ne!(mask[0] & (1 << 1), 0, "end token missing from cached mask");
        }
        state.commit_token(1).unwrap();
        assert!(state.is_accepting());
    }
}

#[test]
fn ebnf_import_uses_core_merged_table_by_default() {
    let constraint = Constraint::from_ebnf("start ::= 'a'", &vocab(&["a"])).unwrap();

    assert_eq!(constraint.table.construction, GlrTableConstruction::ExperimentalCoreMerged);
    assert_eq!(constraint.table.admission_policy, AdmissionPolicy::ExactSimulation);
}
