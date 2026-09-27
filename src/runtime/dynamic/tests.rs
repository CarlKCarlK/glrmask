    use super::*;

    /// Pins `TEST_COMPAT_MODE` to JsonSchema for full-JSON tests, restoring
    /// the prior mode on drop. Mirrors `EnvVarGuard`'s compat handling in the
    /// json_schema tests; the mode cell is the live control (env only seeds
    /// fresh threads), so no env change is needed.
    struct CompatModeGuard {
        original: crate::import::json_schema::string::JsonStringCompatMode,
    }

    impl CompatModeGuard {
        fn json_schema() -> Self {
            Self::set_mode(crate::import::json_schema::string::JsonStringCompatMode::JsonSchema)
        }

        fn native() -> Self {
            Self::set_mode(
                crate::import::json_schema::string::JsonStringCompatMode::LlGuidanceNative,
            )
        }

        fn set_mode(mode: crate::import::json_schema::string::JsonStringCompatMode) -> Self {
            use crate::import::json_schema::string::TEST_COMPAT_MODE;
            let original = TEST_COMPAT_MODE.with(|cell| cell.get());
            TEST_COMPAT_MODE.with(|cell| cell.set(mode));
            Self { original }
        }
    }

    impl Drop for CompatModeGuard {
        fn drop(&mut self) {
            use crate::import::json_schema::string::TEST_COMPAT_MODE;
            TEST_COMPAT_MODE.with(|cell| cell.set(self.original));
        }
    }

    mod projected_quotient_fixture {
        use crate as glrmask;

        include!("../../../tests/fixtures/snowplow_hostname.rsinc");

        pub(super) fn schema_and_vocab() -> (&'static str, Vocab) {
            (SNOWPLOW_SCHEMA, snowplow_vocab())
        }

        pub(super) fn replay_ids() -> &'static [u32] {
            SNOWPLOW_REPLAY_IDS
        }
    }

    fn token_allowed(mask: &[u32], token_id: u32) -> bool {
        let word = token_id as usize / 32;
        let bit = token_id % 32;
        mask.get(word)
            .is_some_and(|bits| bits & (1u32 << bit) != 0)
    }

    fn compressed_lark_grammar(source: &str) -> crate::grammar::flat::GrammarDef {
        let mut named = crate::import::lark::parse_lark_to_named_uncompressed(source).unwrap();
        assert!(crate::grammar::right_linear::compress_large_right_linear_grammar(
            &mut named
        ));
        let factored = crate::grammar::factoring::factor_named_grammar(named);
        crate::grammar::ast::lower(&factored).unwrap()
    }

    fn compile_compressed_static(source: &str, vocab: &Vocab) -> crate::Constraint {
        crate::compiler::pipeline::compile_owned_with_table_construction(
            compressed_lark_grammar(source),
            vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
    }

    fn compile_compressed_dynamic(source: &str, vocab: &Vocab) -> DynamicConstraint {
        crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            compressed_lark_grammar(source),
            vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap()
    }

    fn vocab() -> Vocab {
        Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"ab".to_vec()),
            (3, b"aa".to_vec()),
            (4, b" ".to_vec()),
        ])
    }

    #[test]
    fn dynamic_constraint_matches_constraint_masks_and_commits() {
        let vocab = vocab();
        let grammar = r#"
            start start;
            t A ::= 'a'+;
            t B ::= 'b';
            nt start ::= A B;
        "#;
        let normal = crate::Constraint::from_glrm_grammar(grammar, &vocab).unwrap();
        let dynamic = DynamicConstraint::from_glrm_grammar(grammar, &vocab).unwrap();
        let mut normal_state = normal.start();
        let mut dynamic_state = dynamic.start();

        assert_eq!(normal_state.mask(), dynamic_state.mask());
        normal_state.commit_token(3).unwrap();
        dynamic_state.commit_token(3).unwrap();
        assert_eq!(normal_state.mask(), dynamic_state.mask());
        normal_state.commit_token(1).unwrap();
        dynamic_state.commit_token(1).unwrap();
        assert_eq!(normal_state.is_accepting(), dynamic_state.is_accepting());
        assert_eq!(normal_state.mask(), dynamic_state.mask());
    }

    #[test]
    fn dynamic_json_schema_ordinary_bounded_array_item_stays_a_lazy_terminal() {
        let vocab = Vocab::new(vec![
            (0, b"[".to_vec()),
            (1, b"]".to_vec()),
            (2, b",".to_vec()),
            (3, b"\"".to_vec()),
            (4, b"a".to_vec()),
            (5, b"b".to_vec()),
            (6, b"aa".to_vec()),
            (7, b"\"a\"".to_vec()),
            (8, b"\"b\"".to_vec()),
            (9, b" ".to_vec()),
        ]);
        let schema = r#"{
            "type": "array",
            "items": {
                "type": "string",
                "minLength": 1,
                "maxLength": 1024
            }
        }"#;
        let dynamic = DynamicConstraint::from_json_schema(schema, &vocab).unwrap();
        assert!(dynamic.inner.tokenizer.has_virtual_residual_runtime());
        assert_eq!(
            dynamic
                .inner
                .tokenizer
                .virtual_residual_bounded_code_liveness_oracle_count(),
            1,
        );
        assert!(
            dynamic.inner.tokenizer.num_states() < 1_000,
            "the 1024-character item bound must not be materialized into the physical tokenizer",
        );

        let accepts = |bytes: &[u8]| {
            let mut state = dynamic.start();
            state.commit_bytes(bytes).is_ok() && state.is_accepting()
        };
        assert!(accepts(br#"["a", "bb"]"#));
        assert!(!accepts(br#"[""]"#));

        let mut at_limit = Vec::with_capacity(1028);
        at_limit.extend_from_slice(b"[\"");
        at_limit.extend(std::iter::repeat_n(b'a', 1024));
        at_limit.extend_from_slice(b"\"]");
        assert!(accepts(&at_limit));

        let mut too_long = Vec::with_capacity(1029);
        too_long.extend_from_slice(b"[\"");
        too_long.extend(std::iter::repeat_n(b'a', 1025));
        too_long.extend_from_slice(b"\"]");
        assert!(!accepts(&too_long));

        let loaded = DynamicConstraint::load(&dynamic.save()).unwrap();
        assert!(loaded.inner.tokenizer.has_virtual_residual_runtime());
        assert_eq!(loaded.start().mask(), dynamic.start().mask());

        let mut dynamic_state = dynamic.start();
        let mut loaded_state = loaded.start();
        for (step, token) in [0u32, 7, 2, 9, 8, 1]
        .into_iter()
        .enumerate()
        {
            assert_eq!(dynamic_state.is_accepting(), loaded_state.is_accepting());
            let dynamic_mask = dynamic_state.mask();
            let loaded_mask = loaded_state.mask();
            assert_eq!(dynamic_mask, loaded_mask, "save/load mask mismatch at step {step}");
            assert!(
                token_allowed(&dynamic_mask, token),
                "fixture token {token} must be admitted at step {step}",
            );
            dynamic_state
                .commit_token(token)
                .unwrap_or_else(|error| panic!("dynamic commit failed at step {step}: {error}"));
            loaded_state
                .commit_token(token)
                .unwrap_or_else(|error| panic!("loaded commit failed at step {step}: {error}"));
        }
        assert_eq!(dynamic_state.is_accepting(), loaded_state.is_accepting());
        assert_eq!(dynamic_state.mask(), loaded_state.mask());
    }

    #[test]
    fn dynamic_json_schema_ordinary_bounded_string_bound_is_symbolic() {
        let vocab = Vocab::new(vec![
            (0, b"\"".to_vec()),
            (1, b"a".to_vec()),
            (2, b"aa".to_vec()),
            (3, b"b".to_vec()),
        ]);
        let mut physical_state_counts = Vec::new();
        for max_length in [255usize, 1024, 1_000_000_000_000] {
            let schema = format!(r#"{{"type":"string","maxLength":{max_length}}}"#);
            let dynamic = DynamicConstraint::from_json_schema(&schema, &vocab).unwrap();
            assert!(dynamic.inner.tokenizer.has_virtual_residual_runtime());
            assert_eq!(
                dynamic
                    .inner
                    .tokenizer
                    .virtual_residual_bounded_code_liveness_oracle_count(),
                1,
            );
            physical_state_counts.push(dynamic.inner.tokenizer.num_states());
        }
        assert!(
            physical_state_counts.windows(2).all(|pair| pair[0] == pair[1]),
            "physical tokenizer size must not depend on the numeric bounded-string maximum: {physical_state_counts:?}",
        );
    }

    #[test]
    fn dynamic_json_schema_bounded_pattern_rejects_escaped_spellings_in_native_mode() {
        // Native-default counterpart to the JsonSchema-mode escaped-spelling
        // tests: the LlGuidanceNative lexical language deliberately does not
        // materialize `\uXXXX` spellings (production default for automata
        // size); raw spellings keep working. Same schema as the bounded
        // liveness-oracle test; representative single pin, not a clone set.
        let _env_lock = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _compat = CompatModeGuard::native();
        let vocab = Vocab::new(vec![
            (0, b"\"".to_vec()),
            (1, b"a".to_vec()),
            (2, b"aa".to_vec()),
            (3, b"b".to_vec()),
            (4, b"\\".to_vec()),
            (5, b"u".to_vec()),
            (6, b"0".to_vec()),
            (7, b"6".to_vec()),
            (8, b"1".to_vec()),
        ]);
        let schema = r#"{
            "type": "string",
            "pattern": "^(?:a|bb)+$",
            "minLength": 2,
            "maxLength": 5000
        }"#;
        let dynamic = DynamicConstraint::from_json_schema(schema, &vocab).unwrap();

        let accepts = |bytes: &[u8]| {
            let mut state = dynamic.start();
            state.commit_bytes(bytes).is_ok() && state.is_accepting()
        };
        assert!(accepts(br#""aa""#));
        assert!(!accepts(br#""\u0061\u0061""#));
        assert!(!accepts(br#""a\u0061""#));
    }

    #[test]
    fn dynamic_json_schema_bounded_pattern_uses_exact_code_liveness_oracle() {
        // Pin JsonSchema mode: full-JSON expectations predate the native default (assertions unchanged).
        let _env_lock = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _compat = CompatModeGuard::json_schema();
        let vocab = Vocab::new(vec![
            (0, b"\"".to_vec()),
            (1, b"a".to_vec()),
            (2, b"aa".to_vec()),
            (3, b"b".to_vec()),
            (4, b"\\".to_vec()),
            (5, b"u".to_vec()),
            (6, b"0".to_vec()),
            (7, b"6".to_vec()),
            (8, b"1".to_vec()),
        ]);
        let schema = r#"{
            "type": "string",
            "pattern": "^(?:a|bb)+$",
            "minLength": 2,
            "maxLength": 5000
        }"#;
        let dynamic = DynamicConstraint::from_json_schema(schema, &vocab).unwrap();
        assert!(dynamic.inner.tokenizer.has_virtual_residual_runtime());
        assert!(
            !dynamic
                .inner
                .dynamic_mask_vocab_for_runtime()
                .has_terminal_observation_classes(),
            "physical terminal-observation quotients must stay disabled for virtual residual runtimes",
        );
        assert!(
            dynamic
                .inner
                .tokenizer
                .virtual_residual_bounded_code_liveness_oracle_count()
                > 0,
            "the importer-generated pattern/length intersection must carry the certified prefix-code liveness oracle",
        );

        let accepts = |bytes: &[u8]| {
            let mut state = dynamic.start();
            state.commit_bytes(bytes).is_ok() && state.is_accepting()
        };
        assert!(!accepts(br#""a""#));
        assert!(accepts(br#""aa""#));
        assert!(
            accepts(br#""\u0061\u0061""#),
            "two escaped spellings of decoded 'a' must count as two JSON characters",
        );
        assert!(!accepts(br#""ab""#));

        let mut at_limit = Vec::with_capacity(5002);
        at_limit.push(b'"');
        at_limit.extend(std::iter::repeat_n(b'a', 5000));
        at_limit.push(b'"');
        assert!(accepts(&at_limit));

        let mut too_long = Vec::with_capacity(5003);
        too_long.push(b'"');
        too_long.extend(std::iter::repeat_n(b'a', 5001));
        too_long.push(b'"');
        assert!(!accepts(&too_long));

        let loaded = DynamicConstraint::load(&dynamic.save()).unwrap();
        assert!(
            !loaded
                .inner
                .dynamic_mask_vocab_for_runtime()
                .has_terminal_observation_classes(),
            "save/load must not attach a physical observation quotient to a virtual residual runtime",
        );
        assert!(
            loaded
                .inner
                .tokenizer
                .virtual_residual_bounded_code_liveness_oracle_count()
                > 0,
            "save/load must reconstruct the certified liveness oracle from the retained terminal expression",
        );
        assert_eq!(loaded.start().mask(), dynamic.start().mask());
    }

    #[test]
    fn dynamic_json_schema_bounded_format_uses_exact_code_liveness_oracle() {
        let vocab = Vocab::new(vec![
            (0, b"\"".to_vec()),
            (1, b"a".to_vec()),
            (2, b"b".to_vec()),
            (3, b".".to_vec()),
            (4, b"-".to_vec()),
        ]);
        let schema = r#"{
            "type": "string",
            "format": "hostname",
            "minLength": 3,
            "maxLength": 5000
        }"#;
        let dynamic = DynamicConstraint::from_json_schema(schema, &vocab).unwrap();
        assert!(dynamic.inner.tokenizer.has_virtual_residual_runtime());
        assert!(
            dynamic
                .inner
                .tokenizer
                .virtual_residual_bounded_code_liveness_oracle_count()
                > 0,
            "format/length intersections must carry the same certified JSON decoded-length liveness oracle as pattern/length intersections",
        );

        let accepts = |bytes: &[u8]| {
            let mut state = dynamic.start();
            state.commit_bytes(bytes).is_ok() && state.is_accepting()
        };
        assert!(!accepts(br#""aa""#));
        assert!(accepts(br#""aaa""#));
        assert!(accepts(br#""a.b""#));
        assert!(!accepts(br#""a..b""#));

        let loaded = DynamicConstraint::load(&dynamic.save()).unwrap();
        assert!(
            loaded
                .inner
                .tokenizer
                .virtual_residual_bounded_code_liveness_oracle_count()
                > 0,
            "save/load must reconstruct the bounded format liveness oracle",
        );
        assert_eq!(loaded.start().mask(), dynamic.start().mask());
    }

    #[test]
    fn dynamic_json_schema_pattern_format_and_length_share_exact_code_liveness_oracle() {
        let vocab = Vocab::new(vec![
            (0, b"\"".to_vec()),
            (1, b"a".to_vec()),
            (2, b"aa".to_vec()),
            (3, b"b".to_vec()),
            (4, b".".to_vec()),
        ]);
        let schema = r#"{
            "type": "string",
            "pattern": "^(?:a|bb)+$",
            "format": "hostname",
            "minLength": 2,
            "maxLength": 128
        }"#;
        let dynamic = DynamicConstraint::from_json_schema(schema, &vocab).unwrap();
        assert!(
            dynamic
                .inner
                .tokenizer
                .virtual_residual_bounded_code_liveness_oracle_count()
                > 0,
            "nested pattern/format/length intersections below the generic giant-repeat threshold must still use the certified residual representation",
        );

        let accepts = |bytes: &[u8]| {
            let mut state = dynamic.start();
            state.commit_bytes(bytes).is_ok() && state.is_accepting()
        };
        assert!(!accepts(br#""a""#));
        assert!(accepts(br#""aa""#));
        assert!(accepts(br#""bb""#));
        assert!(!accepts(br#""a.b""#));

        let loaded = DynamicConstraint::load(&dynamic.save()).unwrap();
        assert!(
            loaded
                .inner
                .tokenizer
                .virtual_residual_bounded_code_liveness_oracle_count()
                > 0,
        );
        assert_eq!(loaded.start().mask(), dynamic.start().mask());

        let encode_current = |payload: DynamicArtifact| {
            let payload = bincode::serialize(&payload).unwrap();
            let mut bytes = Vec::with_capacity(DYNAMIC_CONSTRAINT_HEADER_LEN + payload.len());
            bytes.extend_from_slice(&DYNAMIC_CONSTRAINT_MAGIC);
            bytes.extend_from_slice(&DYNAMIC_CONSTRAINT_VERSION.to_le_bytes());
            bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
            bytes.extend_from_slice(&payload);
            bytes
        };

        let mut missing_owner = DynamicArtifact {
            alternatives: vec![DynamicConstraint::alternative_for_constraint(&dynamic.inner)],
            dynamic_mask_vocab: None,
        };
        assert_eq!(missing_owner.alternatives[0].virtual_runtimes.len(), 1);
        missing_owner.alternatives[0].virtual_runtimes.clear();
        let error = DynamicConstraint::load(&encode_current(missing_owner)).unwrap_err();
        assert!(
            error.to_string().contains("terminal ownership mismatch"),
            "dropping a below-threshold residual owner from its physical proxy artifact must fail closed: {error}",
        );

        let mut forged_owner = DynamicArtifact {
            alternatives: vec![DynamicConstraint::alternative_for_constraint(&dynamic.inner)],
            dynamic_mask_vocab: None,
        };
        let terminal = forged_owner.alternatives[0].virtual_runtimes[0].terminal as usize;
        forged_owner.alternatives[0]
            .core
            .terminal_exprs
            .as_mut()
            .expect("current dynamic artifact retains terminal expressions")[terminal] =
            Expr::U8Seq(b"a".to_vec());
        let error = DynamicConstraint::load(&encode_current(forged_owner)).unwrap_err();
        assert!(
            error.to_string().contains("certified bounded-code residual"),
            "a below-threshold residual owner cannot be forged for an uncertified expression: {error}",
        );
    }

    #[test]
    fn dynamic_json_schema_cross_branch_bounded_string_constraints_share_exact_code_liveness_oracle() {
        let vocab = Vocab::new(vec![
            (0, b"\"".to_vec()),
            (1, b"a".to_vec()),
            (2, b"aa".to_vec()),
            (3, b"b".to_vec()),
            (4, b"bb".to_vec()),
            (5, b".".to_vec()),
            (6, b"-".to_vec()),
        ]);
        let schemas = [
            r#"{
                "allOf": [
                    {"type":"string","format":"hostname","minLength":2,"maxLength":5000},
                    {"type":"string","pattern":"^(?:a|bb)+$","minLength":3,"maxLength":5000},
                    {"type":"string","pattern":"^(?:a|bbb)+$","maxLength":5000}
                ]
            }"#,
            r#"{
                "allOf": [
                    {"type":"string","pattern":"^(?:a|bb)+$","minLength":2,"maxLength":6000},
                    {"type":"string","pattern":"^(?:a|bbb)+$","minLength":3,"maxLength":5000},
                    {"type":"string","format":"hostname","maxLength":5500}
                ]
            }"#,
        ];

        for schema in schemas {
            let dynamic = DynamicConstraint::from_json_schema(schema, &vocab).unwrap();
            assert!(
                dynamic.inner.tokenizer.has_virtual_residual_runtime(),
                "expected virtual residual runtime for cross-branch schema: {schema}",
            );
            assert!(
                dynamic
                    .inner
                    .tokenizer
                    .virtual_residual_bounded_code_liveness_oracle_count()
                    > 0,
                "cross-branch bounded string constraints must flatten to one common JSON decoded-length envelope plus finite constraint operands: {schema}",
            );
            assert!(
                !dynamic
                    .inner
                    .dynamic_mask_vocab_for_runtime()
                    .has_terminal_observation_classes(),
                "virtual residual constraints must not attach a physical observation quotient",
            );

            let loaded = DynamicConstraint::load(&dynamic.save()).unwrap();
            assert!(
                loaded
                    .inner
                    .tokenizer
                    .virtual_residual_bounded_code_liveness_oracle_count()
                    > 0,
                "save/load must reconstruct the cross-branch bounded-code oracle: {schema}",
            );
            assert_eq!(loaded.start().mask(), dynamic.start().mask());
        }
    }

    #[test]
    fn dynamic_json_schema_allof_bounded_patterns_share_exact_code_liveness_oracle() {
        // Pin JsonSchema mode: full-JSON expectations predate the native default (assertions unchanged).
        let _env_lock = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _compat = CompatModeGuard::json_schema();
        let vocab = Vocab::new(vec![
            (0, b"\"".to_vec()),
            (1, b"a".to_vec()),
            (2, b"aa".to_vec()),
            (3, b"b".to_vec()),
            (4, b"c".to_vec()),
            (5, b"\\".to_vec()),
            (6, b"u".to_vec()),
            (7, b"0".to_vec()),
            (8, b"6".to_vec()),
            (9, b"1".to_vec()),
        ]);
        let schema = r#"{
            "allOf": [
                {
                    "type": "string",
                    "pattern": "^(?:a|bb)+$",
                    "minLength": 2,
                    "maxLength": 5000
                },
                {
                    "type": "string",
                    "pattern": "^(?:a|cc)+$",
                    "minLength": 3,
                    "maxLength": 4000
                }
            ]
        }"#;
        let dynamic = DynamicConstraint::from_json_schema(schema, &vocab).unwrap();
        assert!(
            dynamic
                .inner
                .tokenizer
                .virtual_residual_bounded_code_liveness_oracle_count()
                > 0,
            "allOf branches with the same JSON length envelope language should share one exact bounded-code oracle",
        );

        let accepts = |bytes: &[u8]| {
            let mut state = dynamic.start();
            state.commit_bytes(bytes).is_ok() && state.is_accepting()
        };
        assert!(!accepts(br#""aa""#));
        assert!(accepts(br#""aaa""#));
        assert!(accepts(br#""\u0061\u0061\u0061""#));
        assert!(!accepts(br#""bbb""#));
        assert!(!accepts(br#""ccc""#));

        let loaded = DynamicConstraint::load(&dynamic.save()).unwrap();
        assert!(
            loaded
                .inner
                .tokenizer
                .virtual_residual_bounded_code_liveness_oracle_count()
                > 0,
            "save/load must reconstruct the coalesced allOf oracle",
        );
        assert_eq!(loaded.start().mask(), dynamic.start().mask());
    }

    #[test]
    fn dynamic_json_schema_bounded_unicode_pattern_keeps_raw_and_escaped_spellings_exact() {
        // Pin JsonSchema mode: full-JSON expectations predate the native default (assertions unchanged).
        let _env_lock = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _compat = CompatModeGuard::json_schema();
        let vocab = Vocab::new(vec![
            (0, b"\"".to_vec()),
            (1, "é".as_bytes().to_vec()),
            (2, br"\u00e9".to_vec()),
            (3, br"\u00E9".to_vec()),
            (4, b"x".to_vec()),
        ]);
        let schema = r#"{
            "type": "string",
            "pattern": "^(?:é|xx)+$",
            "minLength": 2,
            "maxLength": 5000
        }"#;
        let dynamic = DynamicConstraint::from_json_schema(schema, &vocab).unwrap();
        assert!(
            dynamic
                .inner
                .tokenizer
                .virtual_residual_bounded_code_liveness_oracle_count()
                > 0,
        );

        let accepts = |bytes: &[u8]| {
            let mut state = dynamic.start();
            state.commit_bytes(bytes).is_ok() && state.is_accepting()
        };
        assert!(!accepts("\"é\"".as_bytes()));
        assert!(accepts("\"éé\"".as_bytes()));
        assert!(accepts(br#""\u00e9\u00E9""#));
        assert!(accepts("\"é\\u00e9\"".as_bytes()));
        assert!(!accepts("\"éx\"".as_bytes()));
    }

    #[test]
    fn static_constraint_artifact_preserves_dynamic_runtime_sidecars() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"aa".to_vec()),
            (2, b"x".to_vec()),
        ]);
        let dynamic = DynamicConstraint::from_glrm_grammar(
            r#"
                start start;
                t A ::= /a{0,10000}/;
                nt start ::= A;
            "#,
            &vocab,
        )
        .unwrap();
        let original = dynamic.clone().into_constraint();
        assert!(!original.tokenizer.virtual_runtime_metadata().is_empty());
        let loaded = Constraint::load(original.save()).unwrap();
        assert_eq!(
            loaded.tokenizer.virtual_runtime_metadata(),
            original.tokenizer.virtual_runtime_metadata(),
        );
        assert_eq!(loaded.start().mask(), original.start().mask());
    }

    #[test]
    fn dynamic_constraint_save_load_round_trip() {
        let vocab = vocab();
        let constraint = DynamicConstraint::from_glrm_grammar(
            r#"
                start start;
                ignore WS;
                t WS ::= " "+;
                nt start ::= "a"+ "b";
            "#,
            &vocab,
        )
        .unwrap();
        let loaded = DynamicConstraint::load(&constraint.save()).unwrap();
        assert!(constraint.inner.ignore_expr.is_some());
        assert_eq!(loaded.inner.ignore_expr, constraint.inner.ignore_expr);
        assert_eq!(
            loaded.inner.tokenizer.terminal_exprs(),
            constraint.inner.tokenizer.terminal_exprs(),
        );
        assert_eq!(constraint.mask_len(), loaded.mask_len());
        assert_eq!(constraint.start().mask(), loaded.start().mask());
    }

    #[test]
    fn dynamic_v20_persists_vocab_and_validates_shared_vocab() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"ab".to_vec()),
            (2, b"b".to_vec()),
        ]);
        let constraint = DynamicConstraint::from_glrm_grammar(
            r#"
                start start;
                t A ::= /a+/;
                nt start ::= A;
            "#,
            &vocab,
        )
        .unwrap();

        let current = constraint.save();
        assert_eq!(
            u16::from_le_bytes([current[8], current[9]]),
            DYNAMIC_CONSTRAINT_VERSION,
        );
        let payload: DynamicArtifact =
            bincode::deserialize(&current[DYNAMIC_CONSTRAINT_HEADER_LEN..]).unwrap();
        assert!(payload.dynamic_mask_vocab.is_some());
        let current_loaded = DynamicConstraint::load(&current).unwrap();
        assert_eq!(current_loaded.start().mask(), constraint.start().mask());

        let mut mismatched_shared_vocab = payload.clone();
        let mut second = mismatched_shared_vocab.alternatives[0].clone();
        second.core.token_bytes =
            Arc::new(BTreeMap::from([(0, b"z".to_vec())]));
        mismatched_shared_vocab.alternatives.push(second);
        let mismatched_shared_vocab = bincode::serialize(&mismatched_shared_vocab).unwrap();
        let mut malformed =
            Vec::with_capacity(DYNAMIC_CONSTRAINT_HEADER_LEN + mismatched_shared_vocab.len());
        malformed.extend_from_slice(&DYNAMIC_CONSTRAINT_MAGIC);
        malformed.extend_from_slice(&DYNAMIC_CONSTRAINT_VERSION.to_le_bytes());
        malformed.extend_from_slice(&(mismatched_shared_vocab.len() as u64).to_le_bytes());
        malformed.extend_from_slice(&mismatched_shared_vocab);
        let error = DynamicConstraint::load(&malformed).unwrap_err();
        assert!(error
            .to_string()
            .contains("shares a vocabulary index across alternatives with different token bytes"));

    }

    #[test]
    fn dynamic_virtual_unit_repeat_save_load_round_trip() {
        let vocab = Vocab::new(vec![
            (0, b"b".to_vec()),
            (1, b"a".to_vec()),
            (2, b"aa".to_vec()),
            (3, b"aaa".to_vec()),
            (4, b"baaa".to_vec()),
            (5, b"baaaaa".to_vec()),
            (6, b"x".to_vec()),
        ]);
        let grammar = r#"
            start start;
            t A ::= /a{0,1000000000}/;
            t B ::= 'b';
            nt start ::= B A;
        "#;
        let constraint = DynamicConstraint::from_glrm_grammar(grammar, &vocab).unwrap();
        assert!(
            constraint
                .inner
                .tokenizer
                .virtual_zero_min_unit_repeat_mask_tokenizer(vocab.max_token_byte_len())
                .is_some(),
        );

        let bytes = constraint.save();
        let loaded = DynamicConstraint::load(&bytes).unwrap();
        assert!(
            loaded
                .inner
                .tokenizer
                .virtual_zero_min_unit_repeat_mask_tokenizer(vocab.max_token_byte_len())
                .is_some(),
            "load must reconstruct the exact arithmetic lexer sidecar",
        );
        assert_eq!(constraint.start().mask(), loaded.start().mask());

        let mut original_state = constraint.start();
        let mut loaded_state = loaded.start();
        original_state.commit_token(4).unwrap();
        loaded_state.commit_token(4).unwrap();
        assert_eq!(original_state.is_accepting(), loaded_state.is_accepting());
        assert_eq!(original_state.mask(), loaded_state.mask());
    }

    #[test]
    fn dynamic_virtual_runtime_metadata_is_exact_and_fail_closed() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"aa".to_vec()),
            (3, b"bb".to_vec()),
            (4, b"x".to_vec()),
        ]);
        let constraint = DynamicConstraint::from_glrm_grammar(
            r#"
                start start;
                t A ::= /a{0,10000}/;
                t B ::= /b{0,9000}/;
                nt start ::= A | B;
            "#,
            &vocab,
        )
        .unwrap();
        let metadata = constraint.inner.tokenizer.virtual_runtime_metadata();
        assert_eq!(metadata.len(), 2);

        let saved = constraint.save();
        assert_eq!(
            u16::from_le_bytes([saved[8], saved[9]]),
            DYNAMIC_CONSTRAINT_VERSION,
        );
        let loaded = DynamicConstraint::load(&saved).unwrap();
        assert_eq!(loaded.inner.tokenizer.virtual_runtime_metadata().len(), 2);
        assert_eq!(constraint.start().mask(), loaded.start().mask());

        let transfer = constraint.clone().into_saved();
        assert_eq!(
            u16::from_le_bytes([transfer[8], transfer[9]]),
            DYNAMIC_TRANSFER_VERSION,
        );
        let transferred = DynamicConstraint::load_with_vocab(&transfer, &vocab).unwrap();
        assert_eq!(transferred.inner.tokenizer.virtual_runtime_metadata().len(), 2);
        assert_eq!(constraint.start().mask(), transferred.start().mask());

        fn encode_payload(alternative: DynamicAlternative) -> Vec<u8> {
            let payload = DynamicArtifact {
                alternatives: vec![alternative],
                dynamic_mask_vocab: None,
            };
            let payload = bincode::serialize(&payload).unwrap();
            let mut bytes = Vec::with_capacity(DYNAMIC_CONSTRAINT_HEADER_LEN + payload.len());
            bytes.extend_from_slice(&DYNAMIC_CONSTRAINT_MAGIC);
            bytes.extend_from_slice(&DYNAMIC_CONSTRAINT_VERSION.to_le_bytes());
            bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
            bytes.extend_from_slice(&payload);
            bytes
        }

        let mut missing = DynamicConstraint::alternative_for_constraint(&constraint.inner);
        missing.virtual_runtimes.pop();
        let error = DynamicConstraint::load(&encode_payload(missing)).unwrap_err();
        assert!(
            error.to_string().contains("terminal ownership mismatch"),
            "unexpected missing-runtime error: {error}",
        );

        let mut duplicate_root = DynamicConstraint::alternative_for_constraint(&constraint.inner);
        let root = duplicate_root.virtual_runtimes[0].root_state;
        duplicate_root.virtual_runtimes[1].root_state = root;
        let error = DynamicConstraint::load(&encode_payload(duplicate_root)).unwrap_err();
        assert!(
            error.to_string().contains("invalid terminal/root ownership"),
            "unexpected duplicate-root error: {error}",
        );

        let mut mismatched_support = DynamicConstraint::alternative_for_constraint(&constraint.inner);
        let terminal = mismatched_support.virtual_runtimes[0].terminal as usize;
        if let Some(exprs) = mismatched_support.core.terminal_exprs.as_mut() {
            exprs[terminal] = Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"z".to_vec())),
                min: 0,
                max: Some(10_000),
            };
        }
        let error = DynamicConstraint::load(&encode_payload(mismatched_support)).unwrap_err();
        assert!(
            error.to_string().contains("byte support"),
            "unexpected byte-support mismatch error: {error}",
        );
    }

    #[test]
    fn dynamic_current_formats_roundtrip_and_reject_all_other_versions() {
        let vocab = Vocab::new(vec![
            (0, b"\"".to_vec()),
            (1, b"a".to_vec()),
            (2, b"aa".to_vec()),
            (3, b"b".to_vec()),
            (4, b"bb".to_vec()),
        ]);
        let schema = r#"{
            "type": "string",
            "pattern": "^(?:a|bb)+$",
            "minLength": 2,
            "maxLength": 5000
        }"#;
        let constraint = DynamicConstraint::from_json_schema(schema, &vocab).unwrap();
        assert!(constraint.inner.tokenizer.has_any_virtual_runtime());

        // Test current V13 save and load.
        let v13_bytes = constraint.save_with_external_vocab();
        assert_eq!(u16::from_le_bytes([v13_bytes[8], v13_bytes[9]]), DYNAMIC_TRANSFER_VERSION);
        assert_eq!(DYNAMIC_TRANSFER_VERSION, 13);

        let v13_loaded = DynamicConstraint::load_with_vocab(&v13_bytes, &vocab).unwrap();
        // Loaded constraint should carry the mask tokenizer projection directly from the wire
        assert!(v13_loaded.inner.dynamic_mask_vocab.mask_projection_tokenizer().is_some());
        assert_eq!(v13_loaded.start().mask(), constraint.start().mask());

        for version in 0..=DYNAMIC_TRANSFER_VERSION + 1 {
            if version == DYNAMIC_TRANSFER_VERSION { continue; }
            let mut old = v13_bytes.clone();
            old[8..10].copy_from_slice(&version.to_le_bytes());
            assert!(DynamicConstraint::load_with_vocab(&old, &vocab)
                .unwrap_err().to_string().contains("unsupported"), "version {version}");
        }
        let current = constraint.save();
        for version in 0..=DYNAMIC_CONSTRAINT_VERSION + 1 {
            if version == DYNAMIC_CONSTRAINT_VERSION { continue; }
            let mut old = current.clone();
            old[8..10].copy_from_slice(&version.to_le_bytes());
            assert!(DynamicConstraint::load(&old).unwrap_err().to_string().contains("unsupported"),
                "version {version}");
        }
    }

    #[test]
    fn current_dynamic_transfer_defers_terminal_expression_trees() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"ab".to_vec()),
        ]);
        let constraint = DynamicConstraint::from_glrm_grammar(
            r#"
                start start;
                t A ::= /a+/;
                t B ::= /b+/;
                nt start ::= A B;
            "#,
            &vocab,
        )
        .unwrap();
        let expected = constraint
            .inner
            .tokenizer
            .terminal_exprs()
            .expect("fresh constraint retains terminal expressions")
            .to_vec();
        assert!(constraint.inner.tokenizer.virtual_runtime_metadata().is_empty());

        let transfer = constraint.into_saved();
        let loaded = DynamicConstraint::load_with_vocab(&transfer, &vocab).unwrap();
        assert!(
            loaded.inner.tokenizer.terminal_exprs().is_none(),
            "ordinary transfer load must not eagerly materialize source expression trees",
        );
        assert!(matches!(
            loaded.inner.deferred_terminal_exprs_blob,
            Some(crate::runtime::DeferredTerminalExprBytes::CompressedBacked { .. })
        ));
        assert_eq!(
            loaded
                .inner
                .retained_terminal_exprs()
                .expect("composition must recover deferred source expressions"),
            expected,
        );
        assert_eq!(
            loaded
                .inner
                .retained_terminal_exprs()
                .expect("decoded expressions should stay cached"),
            expected,
        );
    }

    #[test]
    fn dynamic_v20_terminal_observation_certificate_is_shape_validated() {
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);
        let constraint = DynamicConstraint::from_glrm_grammar(
            r#"
                start start;
                t A ::= /a+/;
                t B ::= /b+/;
                nt start ::= A | B;
            "#,
            &vocab,
        )
        .unwrap();
        let states = constraint.inner.tokenizer.num_states() as usize;
        let terminals = constraint.inner.tokenizer.num_terminals();

        let encode = |terminal_observation_classes: Vec<(TerminalID, Vec<u32>)>| {
            let mut alternative = DynamicConstraint::alternative_for_constraint(&constraint.inner);
            alternative.terminal_observation_classes = terminal_observation_classes;
            let payload = DynamicArtifact {
                alternatives: vec![alternative],
                dynamic_mask_vocab: None,
            };
            let payload = bincode::serialize(&payload).unwrap();
            let mut bytes = Vec::with_capacity(DYNAMIC_CONSTRAINT_HEADER_LEN + payload.len());
            bytes.extend_from_slice(&DYNAMIC_CONSTRAINT_MAGIC);
            bytes.extend_from_slice(&DYNAMIC_CONSTRAINT_VERSION.to_le_bytes());
            bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
            bytes.extend_from_slice(&payload);
            bytes
        };

        let bad_terminal = DynamicConstraint::load(&encode(vec![(
            terminals,
            vec![1; states],
        )]))
        .unwrap_err();
        assert!(bad_terminal
            .to_string()
            .contains("terminal-observation certificate references terminal"));

        let duplicate = DynamicConstraint::load(&encode(vec![
            (0, vec![1; states]),
            (0, vec![1; states]),
        ]))
        .unwrap_err();
        assert!(duplicate
            .to_string()
            .contains("terminal-observation certificate repeats terminal"));

        let bad_len = DynamicConstraint::load(&encode(vec![(
            0,
            vec![1; states.saturating_sub(1)],
        )]))
        .unwrap_err();
        assert!(bad_len
            .to_string()
            .contains("terminal-observation certificate for terminal"));
    }

    #[test]
    fn dynamic_standalone_virtual_unit_v20_and_transfer_v9_round_trip() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"aa".to_vec()),
            (2, b"aaa".to_vec()),
            (3, b"x".to_vec()),
        ]);
        let constraint = DynamicConstraint::from_glrm_grammar(
            r#"
                start start;
                t A ::= /a{0,10000}/;
                nt start ::= A;
            "#,
            &vocab,
        )
        .unwrap();
        let metadata = constraint.inner.tokenizer.virtual_runtime_metadata();
        assert_eq!(metadata.len(), 1);
        assert_eq!(metadata[0].kind, crate::automata::lexer::tokenizer::VirtualTokenizerRuntimeKind::UnitRepeat);
        assert_eq!(metadata[0].root_state, 0);

        let saved = constraint.save();
        assert_eq!(u16::from_le_bytes([saved[8], saved[9]]), DYNAMIC_CONSTRAINT_VERSION);
        let loaded = DynamicConstraint::load(&saved).unwrap();
        assert_eq!(loaded.inner.tokenizer.virtual_runtime_metadata(), metadata);
        assert_eq!(loaded.start().mask(), constraint.start().mask());

        let transfer = constraint.clone().into_saved();
        assert_eq!(u16::from_le_bytes([transfer[8], transfer[9]]), DYNAMIC_TRANSFER_VERSION);
        let transferred = DynamicConstraint::load_with_vocab(&transfer, &vocab).unwrap();
        assert_eq!(transferred.inner.tokenizer.virtual_runtime_metadata(), metadata);
        assert_eq!(transferred.start().mask(), constraint.start().mask());

        for token in [0u32, 1, 2] {
            let mut original = constraint.start();
            let mut loaded_state = loaded.start();
            let mut transferred_state = transferred.start();
            original.commit_token(token).unwrap();
            loaded_state.commit_token(token).unwrap();
            transferred_state.commit_token(token).unwrap();
            assert_eq!(loaded_state.is_accepting(), original.is_accepting());
            assert_eq!(transferred_state.is_accepting(), original.is_accepting());
            assert_eq!(loaded_state.mask(), original.mask());
            assert_eq!(transferred_state.mask(), original.mask());
        }

    }

    #[test]
    fn dynamic_v20_combines_composition_and_residual_runtime_metadata() {
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"aa".to_vec())]);
        let mut constraint = DynamicConstraint::from_glrm_grammar(
            r#"
                start start;
                t A ::= /a{0,10000}/;
                nt start ::= A;
            "#,
            &vocab,
        )
        .unwrap();
        constraint.inner.late_grammar_slots = vec![crate::runtime::LateGrammarSlot {
            name: "child".to_owned(),
            terminal_id: 0,
        }];
        constraint.inner.boundary_trigger = crate::runtime::BoundaryTrigger::Tokens(
            Arc::from(vec![1u32].into_boxed_slice()),
        );
        let state_count = constraint.inner.tokenizer.num_states() as u32;
        constraint.inner.dynamic_mask_vocab.set_terminal_observation_classes(vec![(
            0,
            Arc::from((1..=state_count).collect::<Vec<_>>().into_boxed_slice()),
        )]);
        assert!(constraint.inner.dynamic_mask_vocab.to_vocab_artifact().is_some());
        let metadata = constraint.inner.tokenizer.virtual_runtime_metadata();
        assert!(!metadata.is_empty());

        // This test mutates private persistence metadata after public
        // compilation has already frozen the canonical transfer artifact.
        // Refresh that cache explicitly so the transfer half of the test
        // continues to cover the wire representation produced from the
        // mutated fixture. Production constraints are immutable after build.
        constraint.external_vocab_artifact_cache = None;
        constraint.cache_external_vocab_artifact_for_save();

        let saved = constraint.save();
        assert_eq!(
            u16::from_le_bytes([saved[8], saved[9]]),
            DYNAMIC_CONSTRAINT_VERSION,
        );
        let loaded = DynamicConstraint::load(&saved).unwrap();
        assert_eq!(loaded.inner.late_grammar_slots, constraint.inner.late_grammar_slots);
        assert_eq!(loaded.inner.tokenizer.virtual_runtime_metadata(), metadata);
        assert!(loaded.inner.dynamic_mask_vocab.has_terminal_observation_classes());
        assert!(loaded.inner.dynamic_mask_vocab.to_vocab_artifact().is_some());
        assert_eq!(
            loaded.inner.boundary_trigger.token_summary().map(|tokens| tokens.to_vec()),
            Some(vec![1]),
        );
        assert_eq!(loaded.start().mask(), constraint.start().mask());

        let transfer = constraint.clone().into_saved();
        assert_eq!(
            u16::from_le_bytes([transfer[8], transfer[9]]),
            DYNAMIC_TRANSFER_VERSION,
        );
        let transferred = DynamicConstraint::load_with_vocab(&transfer, &vocab).unwrap();
        assert_eq!(transferred.inner.tokenizer.virtual_runtime_metadata(), metadata);
        assert!(transferred.inner.dynamic_mask_vocab.has_terminal_observation_classes());
        assert_eq!(
            transferred
                .inner
                .boundary_trigger
                .token_summary()
                .map(|tokens| tokens.to_vec()),
            Some(vec![1]),
        );
        assert_eq!(transferred.start().mask(), constraint.start().mask());

    }

    fn variable_width_repeat_intersection_grammar(
        left_max: usize,
        right_max: usize,
    ) -> crate::grammar::flat::GrammarDef {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let left_body = Expr::Choice(vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"bb".to_vec()),
        ]);
        let right_body = Expr::Choice(vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"b".to_vec()),
        ]);
        GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Intersect {
                    expr: Box::new(Expr::Repeat {
                        expr: Box::new(left_body),
                        min: 0,
                        max: Some(left_max),
                    }),
                    intersect: Box::new(Expr::Repeat {
                        expr: Box::new(right_body),
                        min: 0,
                        max: Some(right_max),
                    }),
                },
            }],
            ..GrammarDef::default()
        }
    }

    fn variable_width_repeat_intersection_vocab() -> Vocab {
        Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"bb".to_vec()),
            (2, b"abb".to_vec()),
            (3, b"aabb".to_vec()),
            (4, b"bbbb".to_vec()),
            (5, b"ab".to_vec()),
            (6, b"b".to_vec()),
            (7, b"x".to_vec()),
        ])
    }

    #[test]
    fn dynamic_billion_by_billion_repeat_intersection_is_lazy_end_to_end() {
        let vocab = variable_width_repeat_intersection_vocab();
        let grammar = variable_width_repeat_intersection_grammar(
            1_000_000_000,
            900_000_000,
        );
        let constraint = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar,
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            variable_width_repeat_intersection_grammar(32, 32),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );

        assert!(constraint.inner.tokenizer.has_virtual_binary_repeat_intersection());
        assert!(
            constraint.inner.tokenizer.num_states() < 32,
            "physical tokenizer must not scale with either billion-sized bound",
        );
        assert!(
            constraint
                .inner
                .tokenizer
                .virtual_binary_repeat_intersection_interned_state_count()
                <= 4,
            "build-time exact residual discovery must stay constant and independent of N*M",
        );
        assert!(
            constraint
                .inner
                .dynamic_mask_vocab
                .mask_projection_tokenizer()
                .is_some(),
            "finite mask projection is an immutable runtime accelerator and must be prepared during build finalization",
        );

        let mask_tokenizer = constraint
            .inner
            .dynamic_mask_vocab
            .mask_projection_tokenizer()
            .expect("build finalization must prepare the finite exact product");
        let start_mask = constraint.start().mask();
        assert!(
            constraint.inner.lazy_dynamic_mask_vocab.get().is_none(),
            "first mask must not hide projection construction in a lazy runtime cache",
        );
        assert!(
            mask_tokenizer.num_states() < 2_000,
            "mask tokenizer must scale with vocab horizon/body DFAs, not N*M",
        );

        assert_eq!(
            start_mask,
            oracle.start().mask(),
            "far from either upper bound, the billion-scale lazy lexer must have the same finite-vocabulary observations as a materialized 32x32 oracle",
        );
        assert!(!token_allowed(&start_mask, 7), "x cannot begin either repeat language");

        let mut state = constraint.start();
        let mut oracle_state = oracle.start();
        state.commit_token(2).unwrap(); // "abb" = "a" + "bb"
        oracle_state.commit_token(2).unwrap();
        assert!(state.is_accepting());
        let after = state.mask();
        assert_eq!(after, oracle_state.mask());
        assert!(
            constraint
                .inner
                .tokenizer
                .virtual_binary_repeat_intersection_interned_state_count()
                < 32,
            "a short commit must discover only a short exact residual path",
        );
    }

    #[test]
    fn dynamic_repeat_intersection_matches_materialized_boundary_oracle() {
        let vocab = variable_width_repeat_intersection_vocab();
        let grammar = variable_width_repeat_intersection_grammar(4, 5);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar.clone(),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let ordinary = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar,
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );
        assert!(!dynamic.inner.tokenizer.has_virtual_binary_repeat_intersection());

        let mut dynamic_state = dynamic.start();
        let mut ordinary_state = ordinary.start();
        assert_eq!(dynamic_state.mask(), ordinary_state.mask());
        dynamic_state.commit_token(2).unwrap();
        ordinary_state.commit_token(2).unwrap();
        assert_eq!(dynamic_state.mask(), ordinary_state.mask());
        dynamic_state.commit_token(1).unwrap();
        ordinary_state.commit_token(1).unwrap();
        assert_eq!(dynamic_state.is_accepting(), ordinary_state.is_accepting());
        assert_eq!(dynamic_state.mask(), ordinary_state.mask());
    }

    #[test]
    fn dynamic_virtual_repeat_intersection_save_load_round_trip() {
        let vocab = variable_width_repeat_intersection_vocab();
        let constraint = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            variable_width_repeat_intersection_grammar(1_000_000_000, 900_000_000),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        assert!(constraint.inner.tokenizer.has_virtual_binary_repeat_intersection());

        let bytes = constraint.save();
        let loaded = DynamicConstraint::load(&bytes).unwrap();
        assert!(
            loaded.inner.tokenizer.has_virtual_binary_repeat_intersection(),
            "load must reconstruct the lazy exact product sidecar",
        );
        assert!(
            loaded
                .inner
                .tokenizer
                .virtual_binary_repeat_intersection_interned_state_count()
                <= 4,
            "load-time residual reconstruction must stay constant and independent of N*M",
        );
        assert_eq!(constraint.start().mask(), loaded.start().mask());

        let mut original_state = constraint.start();
        let mut loaded_state = loaded.start();
        original_state.commit_token(2).unwrap();
        loaded_state.commit_token(2).unwrap();
        assert_eq!(original_state.is_accepting(), loaded_state.is_accepting());
        assert_eq!(original_state.mask(), loaded_state.mask());
    }

    #[test]
    fn dynamic_virtual_repeat_intersection_rejects_dead_common_prefix() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar = GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Intersect {
                    expr: Box::new(Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"ab".to_vec())),
                        min: 0,
                        max: Some(1_000_000_000),
                    }),
                    intersect: Box::new(Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"ac".to_vec())),
                        min: 0,
                        max: Some(900_000_000),
                    }),
                },
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"ab".to_vec()),
            (2, b"ac".to_vec()),
            (3, b"x".to_vec()),
        ]);
        let constraint = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar,
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle_grammar = GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Intersect {
                    expr: Box::new(Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"ab".to_vec())),
                        min: 0,
                        max: Some(8),
                    }),
                    intersect: Box::new(Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"ac".to_vec())),
                        min: 0,
                        max: Some(8),
                    }),
                },
            }],
            ..GrammarDef::default()
        };
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            oracle_grammar,
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );
        assert!(constraint.inner.tokenizer.has_virtual_binary_repeat_intersection());
        let mask = constraint.start().mask();
        assert_eq!(mask, oracle.start().mask());
        assert!(
            !token_allowed(&mask, 0),
            "the common first byte 'a' is dead: left requires b next while right requires c",
        );
        assert!(mask.iter().all(|&word| word == 0));
    }

    #[test]
    fn dynamic_billion_bound_variable_width_repeat_is_lazy_end_to_end() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |max| GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Repeat {
                    expr: Box::new(Expr::Choice(vec![
                        Expr::U8Seq(b"ab".to_vec()),
                        Expr::U8Seq(b"ac".to_vec()),
                    ])),
                    min: 0,
                    max: Some(max),
                },
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"ab".to_vec()),
            (2, b"ac".to_vec()),
            (3, b"abab".to_vec()),
            (4, b"acab".to_vec()),
            (5, b"x".to_vec()),
        ]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(1_000_000_000),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(16),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );

        assert!(dynamic.inner.tokenizer.has_virtual_binary_repeat_intersection());
        assert!(dynamic.inner.tokenizer.num_states() < 32);
        assert_eq!(dynamic.start().mask(), oracle.start().mask());

        let mut dynamic_state = dynamic.start();
        let mut oracle_state = oracle.start();
        for token in [3, 4, 1, 2] {
            dynamic_state.commit_token(token).unwrap();
            oracle_state.commit_token(token).unwrap();
            assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
            assert_eq!(dynamic_state.mask(), oracle_state.mask());
        }

        let saved = dynamic.save();
        let loaded = DynamicConstraint::load(&saved).unwrap();
        assert!(loaded.inner.tokenizer.has_virtual_binary_repeat_intersection());
        assert_eq!(loaded.start().mask(), dynamic.start().mask());

        let mut original_state = dynamic.start();
        let mut loaded_state = loaded.start();
        original_state.commit_token(0).unwrap();
        loaded_state.commit_token(0).unwrap();
        assert_eq!(loaded_state.is_accepting(), original_state.is_accepting());
        assert_eq!(loaded_state.mask(), original_state.mask());
    }

    #[test]
    fn dynamic_billion_nonzero_min_variable_width_repeat_is_lazy_end_to_end() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |min, max| GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Repeat {
                    expr: Box::new(Expr::Choice(vec![
                        Expr::U8Seq(b"ab".to_vec()),
                        Expr::U8Seq(b"c".to_vec()),
                    ])),
                    min,
                    max: Some(max),
                },
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"ab".to_vec()),
            (2, b"c".to_vec()),
            (3, b"abc".to_vec()),
            (4, b"cab".to_vec()),
            (5, b"x".to_vec()),
        ]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(999_999_997, 1_000_000_000),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        // A small ordinary oracle with the same local lower/upper-bound
        // geometry is enough for the first few model-token walks. Both starts
        // are farther than one vocabulary horizon from either boundary.
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(10, 13),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );

        assert!(dynamic.inner.tokenizer.has_virtual_binary_repeat_intersection());
        assert!(dynamic.inner.tokenizer.num_states() < 32);
        assert_eq!(dynamic.start().mask(), oracle.start().mask());

        let mut dynamic_state = dynamic.start();
        let mut oracle_state = oracle.start();
        for token in [1u32, 2, 3] {
            dynamic_state.commit_token(token).unwrap();
            oracle_state.commit_token(token).unwrap();
            assert!(!dynamic_state.is_accepting());
            assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
            assert_eq!(dynamic_state.mask(), oracle_state.mask());
        }

        let saved = dynamic.save();
        let loaded = DynamicConstraint::load(&saved).unwrap();
        assert!(loaded.inner.tokenizer.has_virtual_binary_repeat_intersection());
        assert!(
            loaded
                .inner
                .tokenizer
                .virtual_binary_repeat_intersections_mask_tokenizer(vocab.max_token_byte_len())
                .is_some()
        );
        assert_eq!(loaded.start().mask(), dynamic.start().mask());

        let mut original_state = dynamic.start();
        let mut loaded_state = loaded.start();
        for token in [1u32, 2, 3] {
            original_state.commit_token(token).unwrap();
            loaded_state.commit_token(token).unwrap();
            assert_eq!(loaded_state.is_accepting(), original_state.is_accepting());
            assert_eq!(loaded_state.mask(), original_state.mask());
        }
    }

    #[test]
    fn dynamic_same_body_nonzero_repeat_intersection_factors_end_to_end() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |left_min, left_max, right_min, right_max| {
            let body = Expr::Choice(vec![
                Expr::U8Seq(b"ab".to_vec()),
                Expr::U8Seq(b"c".to_vec()),
            ]);
            GrammarDef {
                start: 0,
                rules: vec![Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Terminal(0)],
                }],
                terminals: vec![Terminal::Expr {
                    id: 0,
                    expr: Expr::Intersect {
                        expr: Box::new(Expr::Repeat {
                            expr: Box::new(body.clone()),
                            min: left_min,
                            max: Some(left_max),
                        }),
                        intersect: Box::new(Expr::Repeat {
                            expr: Box::new(body),
                            min: right_min,
                            max: Some(right_max),
                        }),
                    },
                }],
                ..GrammarDef::default()
            }
        };
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"ab".to_vec()),
            (2, b"c".to_vec()),
            (3, b"abc".to_vec()),
            (4, b"cab".to_vec()),
            (5, b"x".to_vec()),
        ]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(900_000_000, 1_000_000_000, 999_999_997, 999_999_999),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(5, 13, 10, 12),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );

        assert!(dynamic.inner.tokenizer.has_virtual_binary_repeat_intersection());
        assert!(dynamic.inner.tokenizer.num_states() < 32);
        assert_eq!(dynamic.start().mask(), oracle.start().mask());

        let mut dynamic_state = dynamic.start();
        let mut oracle_state = oracle.start();
        for token in [1u32, 2, 3] {
            dynamic_state.commit_token(token).unwrap();
            oracle_state.commit_token(token).unwrap();
            assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
            assert_eq!(dynamic_state.mask(), oracle_state.mask());
        }

        let saved = dynamic.save();
        let loaded = DynamicConstraint::load(&saved).unwrap();
        assert!(loaded.inner.tokenizer.has_virtual_binary_repeat_intersection());
        assert_eq!(loaded.start().mask(), dynamic.start().mask());
        let mut original_state = dynamic.start();
        let mut loaded_state = loaded.start();
        for token in [1u32, 2, 3] {
            original_state.commit_token(token).unwrap();
            loaded_state.commit_token(token).unwrap();
            assert_eq!(loaded_state.is_accepting(), original_state.is_accepting());
            assert_eq!(loaded_state.mask(), original_state.mask());
        }
    }

    #[test]
    fn dynamic_billion_nonzero_min_unit_repeat_uses_arithmetic_runtime() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |min, max| GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Repeat {
                    expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                    min,
                    max: Some(max),
                },
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"aa".to_vec()),
            (2, b"aaa".to_vec()),
            (3, b"x".to_vec()),
        ]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(500_000_000, 1_000_000_000),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        assert!(!dynamic.inner.tokenizer.has_virtual_binary_repeat_intersection());
        assert!(
            dynamic
                .inner
                .tokenizer
                .virtual_unit_repeat_mask_tokenizer(vocab.max_token_byte_len())
                .is_some(),
            "one-byte nonzero-min repeats should use the O(1) arithmetic sidecar",
        );

        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(10, 20),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );
        assert_eq!(dynamic.start().mask(), oracle.start().mask());
        let mut dynamic_state = dynamic.start();
        let mut oracle_state = oracle.start();
        for token in [2u32, 1, 0] {
            dynamic_state.commit_token(token).unwrap();
            oracle_state.commit_token(token).unwrap();
            assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
            assert_eq!(dynamic_state.mask(), oracle_state.mask());
        }

        let saved = dynamic.save();
        let loaded = DynamicConstraint::load(&saved).unwrap();
        assert!(!loaded.inner.tokenizer.has_virtual_binary_repeat_intersection());
        assert!(
            loaded
                .inner
                .tokenizer
                .virtual_unit_repeat_mask_tokenizer(vocab.max_token_byte_len())
                .is_some(),
        );
        let mut original_state = dynamic.start();
        let mut loaded_state = loaded.start();
        for token in [2u32, 1, 0] {
            original_state.commit_token(token).unwrap();
            loaded_state.commit_token(token).unwrap();
            assert_eq!(loaded_state.is_accepting(), original_state.is_accepting());
            assert_eq!(loaded_state.mask(), original_state.mask());
        }
    }

    #[test]
    fn dynamic_aligned_nonzero_unit_intersection_uses_arithmetic_runtime() {
        use crate::automata::regex::Expr;
        use crate::ds::u8set::U8Set;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |left_min, left_max, right_min, right_max| GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Intersect {
                    expr: Box::new(Expr::Repeat {
                        expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"ab"))),
                        min: left_min,
                        max: Some(left_max),
                    }),
                    intersect: Box::new(Expr::Repeat {
                        expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"bc"))),
                        min: right_min,
                        max: Some(right_max),
                    }),
                },
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![
            (0, b"b".to_vec()),
            (1, b"bb".to_vec()),
            (2, b"bbb".to_vec()),
            (3, b"a".to_vec()),
            (4, b"c".to_vec()),
        ]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(500_000_000, 1_000_000_000, 600_000_000, 900_000_000),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(5, 20, 10, 15),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );

        assert!(!dynamic.inner.tokenizer.has_virtual_binary_repeat_intersection());
        assert!(
            dynamic
                .inner
                .tokenizer
                .virtual_unit_repeat_mask_tokenizer(vocab.max_token_byte_len())
                .is_some(),
            "aligned one-byte intersection should factor to the arithmetic repeat runtime",
        );
        assert_eq!(dynamic.start().mask(), oracle.start().mask());

        let mut dynamic_state = dynamic.start();
        let mut oracle_state = oracle.start();
        for token in [2u32, 1, 0] {
            dynamic_state.commit_token(token).unwrap();
            oracle_state.commit_token(token).unwrap();
            assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
            assert_eq!(dynamic_state.mask(), oracle_state.mask());
        }

        let saved = dynamic.save();
        let loaded = DynamicConstraint::load(&saved).unwrap();
        assert!(!loaded.inner.tokenizer.has_virtual_binary_repeat_intersection());
        assert!(
            loaded
                .inner
                .tokenizer
                .virtual_unit_repeat_mask_tokenizer(vocab.max_token_byte_len())
                .is_some(),
        );
        assert_eq!(loaded.start().mask(), dynamic.start().mask());
    }

    #[test]
    fn dynamic_giant_aligned_unit_empty_intersection_normalizes_before_validation() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |min| GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Intersect {
                    expr: Box::new(Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                        min,
                        max: Some(1_000_000_000),
                    }),
                    intersect: Box::new(Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"b".to_vec())),
                        min,
                        max: Some(1_000_000_000),
                    }),
                },
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);

        for min in [0usize, 1] {
            let dynamic =
                crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
                    grammar_for(min),
                    &vocab,
                    crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
                )
                .unwrap();
            assert!(!dynamic.inner.tokenizer.has_virtual_binary_repeat_intersection());
            assert!(
                dynamic
                    .inner
                    .tokenizer
                    .virtual_unit_repeat_mask_tokenizer(vocab.max_token_byte_len())
                    .is_none(),
                "empty/epsilon normalization should remove the giant repeat entirely",
            );
            let saved = dynamic.save();
            let loaded = DynamicConstraint::load(&saved).unwrap();
            assert_eq!(loaded.start().mask(), dynamic.start().mask());
        }
    }

    #[test]
    fn dynamic_distinct_nonzero_min_giant_intersection_uses_general_residual_runtime() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |max| GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Intersect {
                    expr: Box::new(Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                        min: 3,
                        max: Some(max),
                    }),
                    intersect: Box::new(Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"aa".to_vec())),
                        min: 2,
                        max: Some(max),
                    }),
                },
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"aa".to_vec())]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(10_000),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(8),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );
        assert!(dynamic.inner.tokenizer.has_virtual_residual_runtime());
        assert_eq!(dynamic.start().mask(), oracle.start().mask());
        for sequence in [
            vec![0u32],
            vec![1u32],
            vec![0u32, 0, 0],
            vec![1u32, 1],
            vec![0u32, 1, 0],
        ] {
            let mut dynamic_state = dynamic.start();
            let mut oracle_state = oracle.start();
            for token in sequence {
                dynamic_state.commit_token(token).unwrap();
                oracle_state.commit_token(token).unwrap();
                assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
                assert_eq!(dynamic_state.mask(), oracle_state.mask());
            }
        }
        let loaded = DynamicConstraint::load(&dynamic.save()).unwrap();
        assert!(loaded.inner.tokenizer.has_virtual_residual_runtime());
        assert_eq!(loaded.start().mask(), dynamic.start().mask());
    }

    #[test]
    fn dynamic_multiple_large_plain_repeats_use_exact_virtual_runtime() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |max| GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1)],
            }],
            terminals: vec![
                Terminal::Expr {
                    id: 0,
                    expr: Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                        min: 0,
                        max: Some(max),
                    },
                },
                Terminal::Expr {
                    id: 1,
                    expr: Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"b".to_vec())),
                        min: 0,
                        max: Some(max),
                    },
                },
            ],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(10_000),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(8),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );
        assert!(
            dynamic.inner.tokenizer.has_virtual_binary_repeat_intersection(),
            "both large terminals must stay on exact lazy virtual runtimes",
        );
        assert_eq!(dynamic.start().mask(), oracle.start().mask());
        for sequence in [vec![0u32], vec![1u32], vec![0u32, 1], vec![0u32, 0, 1]] {
            let mut dynamic_state = dynamic.start();
            let mut oracle_state = oracle.start();
            for token in sequence {
                dynamic_state.commit_token(token).unwrap();
                oracle_state.commit_token(token).unwrap();
                assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
                assert_eq!(dynamic_state.mask(), oracle_state.mask());
            }
        }
    }

    #[test]
    fn dynamic_non_prefix_free_large_repeat_uses_general_residual_runtime() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |max| GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Repeat {
                    // `a | aa` is not prefix-free, so one scalar repeat
                    // coordinate is not a complete exact residual.
                    expr: Box::new(Expr::Choice(vec![
                        Expr::U8Seq(b"a".to_vec()),
                        Expr::U8Seq(b"aa".to_vec()),
                    ])),
                    min: 0,
                    max: Some(max),
                },
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"aa".to_vec())]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(10_000),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(8),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );
        assert!(dynamic.inner.tokenizer.has_virtual_residual_runtime());
        assert_eq!(dynamic.start().mask(), oracle.start().mask());
        for sequence in [vec![0u32], vec![1u32], vec![0u32, 1], vec![1u32, 1, 0]] {
            let mut dynamic_state = dynamic.start();
            let mut oracle_state = oracle.start();
            for token in sequence {
                dynamic_state.commit_token(token).unwrap();
                oracle_state.commit_token(token).unwrap();
                assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
                assert_eq!(dynamic_state.mask(), oracle_state.mask());
            }
        }
    }

    #[test]
    fn dynamic_general_residual_prunes_semantically_dead_token_prefix() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |max| {
            let giant_branch = |suffix: &[u8]| {
                Expr::Seq(vec![
                    Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"x".to_vec())),
                        min: 0,
                        max: Some(max),
                    },
                    Expr::U8Seq(suffix.to_vec()),
                ])
            };
            GrammarDef {
                start: 0,
                rules: vec![Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Terminal(0)],
                }],
                terminals: vec![Terminal::Expr {
                    id: 0,
                    expr: Expr::Intersect {
                        expr: Box::new(Expr::Choice(vec![giant_branch(b"ab"), Expr::U8Seq(b"c".to_vec())])),
                        intersect: Box::new(Expr::Choice(vec![
                            giant_branch(b"ac"),
                            Expr::U8Seq(b"c".to_vec()),
                        ])),
                    },
                }],
                ..GrammarDef::default()
            }
        };
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"c".to_vec()),
            (2, b"x".to_vec()),
            (3, b"b".to_vec()),
        ]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(10_000),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(8),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );
        assert!(dynamic.inner.tokenizer.has_virtual_residual_runtime());
        assert_eq!(dynamic.start().mask(), oracle.start().mask());

        let start_mask = dynamic.start().mask();
        let allowed = |token: u32| {
            let word = token as usize / 32;
            let bit = token % 32;
            start_mask[word] & (1u32 << bit) != 0
        };
        assert!(!allowed(0), "a leads to the dead residual b ∩ c and must be pruned");
        assert!(allowed(1), "c is the one common accepted word");
        assert!(!allowed(2), "entering the giant x* branches can never reach a common suffix");

        let mut rejected = dynamic.start();
        assert!(rejected.commit_token(0).is_err());
        let mut accepted = dynamic.start();
        accepted.commit_token(1).unwrap();
        assert!(accepted.is_accepting());

        let loaded = DynamicConstraint::load(&dynamic.save()).unwrap();
        assert!(loaded.inner.tokenizer.has_virtual_residual_runtime());
        assert_eq!(loaded.start().mask(), start_mask);
    }

    #[test]
    fn dynamic_nested_large_repeat_suffix_compiles_lazily_and_exactly() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |max| GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Seq(vec![
                    Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                        min: 0,
                        max: Some(max),
                    },
                    Expr::U8Seq(b"b".to_vec()),
                ]),
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(10_000),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(8),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );
        assert_eq!(dynamic.start().mask(), oracle.start().mask());
        for sequence in [vec![1u32], vec![0u32, 1], vec![0u32, 0, 1]] {
            let mut dynamic_state = dynamic.start();
            let mut oracle_state = oracle.start();
            for token in sequence {
                dynamic_state.commit_token(token).unwrap();
                oracle_state.commit_token(token).unwrap();
                assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
                assert_eq!(dynamic_state.mask(), oracle_state.mask());
            }
        }
    }

    #[test]
    fn dynamic_nested_large_repeat_in_lazy_intersection_remains_supported() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |max| GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Intersect {
                    expr: Box::new(Expr::Seq(vec![
                        Expr::U8Seq(b"[".to_vec()),
                        Expr::Repeat {
                            expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                            min: 0,
                            max: Some(max),
                        },
                        Expr::U8Seq(b"]".to_vec()),
                    ])),
                    intersect: Box::new(Expr::Seq(vec![
                        Expr::U8Seq(b"[".to_vec()),
                        Expr::Repeat {
                            expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                            min: 0,
                            max: Some(7),
                        },
                        Expr::U8Seq(b"]".to_vec()),
                    ])),
                },
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![
            (0, b"[".to_vec()),
            (1, b"a".to_vec()),
            (2, b"aa".to_vec()),
            (3, b"]".to_vec()),
            (4, b"[a]".to_vec()),
            (5, b"[aa]".to_vec()),
            (6, b"x".to_vec()),
        ]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(1_000_000_000),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(8),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );

        assert!(
            dynamic.inner.tokenizer.num_states() < 128,
            "nested giant repeat must stay on the existing lazy intersection path",
        );
        assert!(
            !dynamic
                .inner
                .tokenizer
                .has_virtual_binary_repeat_intersection(),
            "this regression must exercise the nested repeat-with-suffix lane, not the top-level virtual repeat product",
        );
        assert_eq!(dynamic.start().mask(), oracle.start().mask());

        let mut dynamic_state = dynamic.start();
        let mut oracle_state = oracle.start();
        for token in [0, 2, 3] {
            dynamic_state.commit_token(token).unwrap();
            oracle_state.commit_token(token).unwrap();
            assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
            assert_eq!(dynamic_state.mask(), oracle_state.mask());
        }
    }

    #[test]
    fn dynamic_general_residual_repeat_supports_u32_max_bound() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |max| GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Seq(vec![
                    Expr::Repeat {
                        expr: Box::new(Expr::Choice(vec![
                            Expr::U8Seq(b"a".to_vec()),
                            Expr::U8Seq(b"aa".to_vec()),
                        ])),
                        min: 0,
                        max: Some(max),
                    },
                    Expr::U8Seq(b"z".to_vec()),
                ]),
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"aa".to_vec()),
            (2, b"z".to_vec()),
            (3, b"aaz".to_vec()),
        ]);

        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(u32::MAX as usize),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(8),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );
        assert!(dynamic.inner.tokenizer.has_virtual_residual_runtime());
        assert_eq!(dynamic.start().mask(), oracle.start().mask());
        for sequence in [vec![2u32], vec![3u32], vec![0u32, 2], vec![1u32, 2]] {
            let mut dynamic_state = dynamic.start();
            let mut oracle_state = oracle.start();
            for token in sequence {
                dynamic_state.commit_token(token).unwrap();
                oracle_state.commit_token(token).unwrap();
                assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
                assert_eq!(dynamic_state.mask(), oracle_state.mask());
            }
        }
    }

    #[test]
    fn dynamic_hybrid_unit_repeat_near_state_id_limit_uses_repeat_product() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar = GrammarDef {
            start: 0,
            rules: vec![
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Terminal(0)],
                },
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Terminal(1)],
                },
            ],
            terminals: vec![
                Terminal::Expr {
                    id: 0,
                    expr: Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                        min: 0,
                        max: Some(((1u64 << 31) - 1) as usize),
                    },
                },
                Terminal::Expr {
                    id: 1,
                    expr: Expr::U8Seq(b"b".to_vec()),
                },
            ],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"aa".to_vec()),
            (2, b"b".to_vec()),
            (3, b"x".to_vec()),
        ]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar,
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        assert!(
            dynamic.inner.tokenizer.has_virtual_binary_repeat_intersection(),
            "hybrid physical states leave too little high-bit state-ID space for the arithmetic unit lane",
        );
        assert!(dynamic.inner.tokenizer.num_states() < 64);

        let saved = dynamic.save();
        let loaded = DynamicConstraint::load(&saved).unwrap();
        assert!(loaded.inner.tokenizer.has_virtual_binary_repeat_intersection());
        assert_eq!(loaded.start().mask(), dynamic.start().mask());
    }

    #[test]
    fn dynamic_top_level_large_repeat_inside_finite_intersection_stays_exact() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |max| GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Intersect {
                    expr: Box::new(Expr::Repeat {
                        expr: Box::new(Expr::Choice(vec![
                            Expr::U8Seq(b"ab".to_vec()),
                            Expr::U8Seq(b"c".to_vec()),
                        ])),
                        min: 3,
                        max: Some(max),
                    }),
                    // `abcabc` has the unique body factorization
                    // `ab · c · ab · c`, so it is accepted at count four.
                    // The finite right coordinate bounds generic product
                    // discovery independently of the giant repeat maximum.
                    intersect: Box::new(Expr::U8Seq(b"abcabc".to_vec())),
                },
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![
            (0, b"abcabc".to_vec()),
            (1, b"ab".to_vec()),
            (2, b"c".to_vec()),
            (3, b"abc".to_vec()),
            (4, b"x".to_vec()),
        ]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(1_000_000_000),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(8),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );

        assert!(
            !dynamic.inner.tokenizer.has_virtual_binary_repeat_intersection(),
            "this case should use the generic product's virtual repeat coordinate, not a runtime sidecar",
        );
        assert!(dynamic.inner.tokenizer.num_states() < 64);
        assert_eq!(dynamic.start().mask(), oracle.start().mask());

        for sequence in [vec![0u32], vec![1u32, 2, 1, 2], vec![3u32, 3]] {
            let mut dynamic_state = dynamic.start();
            let mut oracle_state = oracle.start();
            for token in sequence {
                dynamic_state.commit_token(token).unwrap();
                oracle_state.commit_token(token).unwrap();
                assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
                assert_eq!(dynamic_state.mask(), oracle_state.mask());
            }
        }

        let saved = dynamic.save();
        let loaded = DynamicConstraint::load(&saved).unwrap();
        assert_eq!(loaded.start().mask(), dynamic.start().mask());
        let mut loaded_state = loaded.start();
        loaded_state.commit_token(0).unwrap();
        assert!(loaded_state.is_accepting());
    }

    #[test]
    fn dynamic_top_level_large_repeat_inside_cyclic_intersection_uses_residual_runtime() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |max| GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Intersect {
                    expr: Box::new(Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                        min: 0,
                        max: Some(max),
                    }),
                    intersect: Box::new(Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                        min: 0,
                        max: None,
                    }),
                },
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"aa".to_vec())]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(10_000),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(8),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );
        assert!(dynamic.inner.tokenizer.has_virtual_residual_runtime());
        assert_eq!(dynamic.start().mask(), oracle.start().mask());
        for sequence in [vec![0u32], vec![1u32], vec![1u32, 1], vec![0u32, 1, 0]] {
            let mut dynamic_state = dynamic.start();
            let mut oracle_state = oracle.start();
            for token in sequence {
                dynamic_state.commit_token(token).unwrap();
                oracle_state.commit_token(token).unwrap();
                assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
                assert_eq!(dynamic_state.mask(), oracle_state.mask());
            }
        }
    }

    #[test]
    fn dynamic_nested_giant_with_budgeted_other_repeat_uses_general_residual_runtime() {
        use crate::automata::regex::Expr;
        use crate::ds::u8set::U8Set;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |left_max| GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Intersect {
                    expr: Box::new(Expr::Seq(vec![
                        Expr::Repeat {
                            expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                            min: 0,
                            max: Some(left_max),
                        },
                        Expr::U8Seq(b"b".to_vec()),
                    ])),
                    // This root repeat is above the generic product's direct
                    // threshold and would normally stay virtual. Its exact DFA
                    // is nevertheless tiny enough for the lazy intersection's
                    // explicitly budgeted ordinary side.
                    intersect: Box::new(Expr::Repeat {
                        expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"ab"))),
                        min: 0,
                        max: Some(100),
                    }),
                },
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"aa".to_vec()),
            (2, b"b".to_vec()),
            (3, b"ab".to_vec()),
            (4, b"x".to_vec()),
        ]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(1_000_000_000),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(128),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );

        assert!(
            dynamic.inner.tokenizer.num_states() < 512,
            "the physical tokenizer must stay small independently of the giant max",
        );
        assert!(!dynamic.inner.tokenizer.has_virtual_binary_repeat_intersection());
        assert!(
            dynamic.inner.tokenizer.has_virtual_residual_runtime(),
            "nested giant intersections outside the arithmetic fast paths must use the general residual runtime",
        );
        assert_eq!(dynamic.start().mask(), oracle.start().mask());

        for sequence in [vec![2u32], vec![3u32], vec![1u32, 2], vec![0u32, 0, 2]] {
            let mut dynamic_state = dynamic.start();
            let mut oracle_state = oracle.start();
            for token in sequence {
                dynamic_state.commit_token(token).unwrap();
                oracle_state.commit_token(token).unwrap();
                assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
                assert_eq!(dynamic_state.mask(), oracle_state.mask());
            }
        }

        let mut dynamic_boundary = dynamic.start();
        let mut oracle_boundary = oracle.start();
        for _ in 0..49 {
            dynamic_boundary.commit_token(1).unwrap();
            oracle_boundary.commit_token(1).unwrap();
        }
        assert_eq!(dynamic_boundary.is_accepting(), oracle_boundary.is_accepting());
        assert_eq!(dynamic_boundary.mask(), oracle_boundary.mask());
        assert!(!dynamic_boundary.is_accepting());

        // At 98 leading 'a' bytes, one more 'a' is still live but another
        // two-byte "aa" token would overshoot the right-hand 100-byte cap once
        // the required trailing 'b' is included.
        let mut dynamic_overrun = dynamic_boundary.clone();
        let mut oracle_overrun = oracle_boundary.clone();
        assert!(dynamic_overrun.commit_token(1).is_err());
        assert!(oracle_overrun.commit_token(1).is_err());

        dynamic_boundary.commit_token(0).unwrap();
        oracle_boundary.commit_token(0).unwrap();
        assert_eq!(dynamic_boundary.is_accepting(), oracle_boundary.is_accepting());
        assert_eq!(dynamic_boundary.mask(), oracle_boundary.mask());
        assert!(!dynamic_boundary.is_accepting());
        dynamic_boundary.commit_token(2).unwrap();
        oracle_boundary.commit_token(2).unwrap();
        assert!(dynamic_boundary.is_accepting());
        assert_eq!(dynamic_boundary.is_accepting(), oracle_boundary.is_accepting());
        assert_eq!(dynamic_boundary.mask(), oracle_boundary.mask());

        let saved = dynamic.save();
        let loaded = DynamicConstraint::load(&saved).unwrap();
        assert_eq!(loaded.start().mask(), dynamic.start().mask());
        assert!(
            loaded.inner.tokenizer.has_virtual_residual_runtime(),
            "save/load must reconstruct the general residual runtime",
        );
        let mut loaded_state = loaded.start();
        let mut dynamic_state = dynamic.start();
        let mut oracle_state = oracle.start();
        loaded_state.commit_token(3).unwrap();
        dynamic_state.commit_token(3).unwrap();
        oracle_state.commit_token(3).unwrap();
        assert!(loaded_state.is_accepting());
        assert_eq!(loaded_state.is_accepting(), dynamic_state.is_accepting());
        assert_eq!(loaded_state.is_accepting(), oracle_state.is_accepting());
        assert_eq!(loaded_state.mask(), dynamic_state.mask());
        assert_eq!(loaded_state.mask(), oracle_state.mask());
    }

    #[test]
    fn dynamic_nested_giant_with_large_finite_other_uses_general_residual_runtime() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let mut long_body = vec![b'a'; 31];
        long_body.push(b'b');
        let grammar_for = |left_max, right_max| GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Intersect {
                    expr: Box::new(Expr::Seq(vec![
                        Expr::Repeat {
                            expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                            min: 0,
                            max: Some(left_max),
                        },
                        Expr::U8Seq(b"b".to_vec()),
                    ])),
                    intersect: Box::new(Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(long_body.clone())),
                        min: 0,
                        max: Some(right_max),
                    }),
                },
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, long_body.clone()),
        ]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(10_000, 4_095),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(64, 4),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );
        assert!(dynamic.inner.tokenizer.has_virtual_residual_runtime());
        assert_eq!(dynamic.start().mask(), oracle.start().mask());
        let mut dynamic_state = dynamic.start();
        let mut oracle_state = oracle.start();
        dynamic_state.commit_token(2).unwrap();
        oracle_state.commit_token(2).unwrap();
        assert!(dynamic_state.is_accepting());
        assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
        assert_eq!(dynamic_state.mask(), oracle_state.mask());

        let loaded = DynamicConstraint::load(&dynamic.save()).unwrap();
        assert!(loaded.inner.tokenizer.has_virtual_residual_runtime());
        assert_eq!(loaded.start().mask(), dynamic.start().mask());
    }

    #[test]
    fn dynamic_empty_giant_cyclic_intersection_becomes_dead_residual_proxy() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |max| GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Intersect {
                    expr: Box::new(Expr::Seq(vec![
                        Expr::Repeat {
                            expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                            min: 0,
                            max: Some(max),
                        },
                        Expr::U8Seq(b"b".to_vec()),
                    ])),
                    intersect: Box::new(Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                        min: 0,
                        max: None,
                    }),
                },
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(10_000),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(8),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );
        assert!(dynamic.inner.tokenizer.has_virtual_residual_runtime());
        assert_eq!(dynamic.start().mask(), oracle.start().mask());
        assert!(dynamic.start().mask().iter().all(|&word| word == 0));

        let loaded = DynamicConstraint::load(&dynamic.save()).unwrap();
        assert!(loaded.inner.tokenizer.has_virtual_residual_runtime());
        assert_eq!(loaded.start().mask(), dynamic.start().mask());
    }

    #[test]
    fn dynamic_two_nested_large_repeat_components_with_disjoint_delimiters_factor_to_empty() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let repeated_suffix = |suffix| {
            Expr::Seq(vec![
                Expr::Repeat {
                    expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                    min: 0,
                    max: Some(10_000),
                },
                Expr::U8Seq(vec![suffix]),
            ])
        };
        let grammar = GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Intersect {
                    expr: Box::new(repeated_suffix(b'b')),
                    intersect: Box::new(repeated_suffix(b'c')),
                },
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"c".to_vec()),
        ]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar,
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let start_mask = dynamic.start().mask();
        assert!(
            start_mask.iter().all(|&word| word == 0),
            "different uniquely-delimited literal suffixes make the intersection empty",
        );

        let loaded = DynamicConstraint::load(&dynamic.save()).unwrap();
        assert_eq!(loaded.start().mask(), start_mask);
    }

    #[test]
    fn dynamic_positive_min_giant_suffix_with_disjoint_counts_factors_to_empty() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let component = |min, max| {
            Expr::Seq(vec![
                Expr::Repeat {
                    expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                    min,
                    max: Some(max),
                },
                Expr::U8Seq(b"b".to_vec()),
            ])
        };
        let grammar = GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Intersect {
                    expr: Box::new(component(5, 10_000)),
                    intersect: Box::new(component(1, 3)),
                },
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar,
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        assert!(dynamic.start().mask().iter().all(|&word| word == 0));
    }

    #[test]
    fn dynamic_multiple_giant_terminals_share_exact_virtual_runtime() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |a_max, b_max| GrammarDef {
            start: 0,
            rules: vec![
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Terminal(0)],
                },
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Terminal(1)],
                },
            ],
            terminals: vec![
                Terminal::Expr {
                    id: 0,
                    expr: Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                        min: 0,
                        max: Some(a_max),
                    },
                },
                Terminal::Expr {
                    id: 1,
                    expr: Expr::Repeat {
                        expr: Box::new(Expr::Choice(vec![
                            Expr::U8Seq(b"a".to_vec()),
                            Expr::U8Seq(b"b".to_vec()),
                        ])),
                        min: 0,
                        max: Some(b_max),
                    },
                },
            ],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"aa".to_vec()),
            (3, b"bb".to_vec()),
            (4, b"ab".to_vec()),
            (5, b"x".to_vec()),
        ]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(1_000_000_000, 900_000_000),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(8, 7),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );
        assert!(
            dynamic.inner.tokenizer.has_virtual_binary_repeat_intersection(),
            "multiple giant terminals must stay on shared lazy exact runtimes",
        );
        let exact_after_a = dynamic.inner.tokenizer.run(b"a");
        assert_eq!(
            exact_after_a.len(),
            2,
            "overlapping virtual terminals must retain distinct exact states",
        );
        let matched_after_a = exact_after_a
            .iter()
            .flat_map(|&state| dynamic.inner.tokenizer.matched_terminals(state))
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(matched_after_a, [0u32, 1u32].into_iter().collect());
        assert_eq!(dynamic.start().mask(), oracle.start().mask());

        for token in [0u32, 1, 2, 3, 4] {
            let mut dynamic_state = dynamic.start();
            let mut oracle_state = oracle.start();
            dynamic_state.commit_token(token).unwrap();
            oracle_state.commit_token(token).unwrap();
            assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
            assert_eq!(dynamic_state.mask(), oracle_state.mask());
        }

        let loaded = DynamicConstraint::load(&dynamic.save()).unwrap();
        assert!(loaded.inner.tokenizer.has_virtual_binary_repeat_intersection());
        assert_eq!(dynamic.start().mask(), loaded.start().mask());
        for token in [0u32, 1] {
            let mut original_state = dynamic.start();
            let mut loaded_state = loaded.start();
            original_state.commit_token(token).unwrap();
            loaded_state.commit_token(token).unwrap();
            assert_eq!(original_state.is_accepting(), loaded_state.is_accepting());
            assert_eq!(original_state.mask(), loaded_state.mask());
        }

        let transfer = dynamic.clone().into_saved();
        let transferred = DynamicConstraint::load_with_vocab(&transfer, &vocab).unwrap();
        assert!(
            transferred
                .inner
                .tokenizer
                .has_virtual_binary_repeat_intersection(),
            "v6 transfer load must reconstruct all shared lazy repeat runtimes",
        );
        assert_eq!(dynamic.start().mask(), transferred.start().mask());
        for token in [0u32, 1] {
            let mut original_state = dynamic.start();
            let mut transferred_state = transferred.start();
            original_state.commit_token(token).unwrap();
            transferred_state.commit_token(token).unwrap();
            assert_eq!(
                original_state.is_accepting(),
                transferred_state.is_accepting()
            );
            assert_eq!(original_state.mask(), transferred_state.mask());
        }
    }

    #[test]
    fn dynamic_mixed_specialized_and_general_giants_share_residual_runtime_family() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |max| GrammarDef {
            start: 0,
            rules: vec![
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Terminal(0)],
                },
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Terminal(1)],
                },
            ],
            terminals: vec![
                Terminal::Expr {
                    id: 0,
                    expr: Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                        min: 0,
                        max: Some(max),
                    },
                },
                Terminal::Expr {
                    id: 1,
                    expr: Expr::Seq(vec![
                        Expr::Repeat {
                            // Prefix ambiguity keeps this out of the simple
                            // bounded-repeat descriptor fast path.
                            expr: Box::new(Expr::Choice(vec![
                                Expr::U8Seq(b"x".to_vec()),
                                Expr::U8Seq(b"xx".to_vec()),
                            ])),
                            min: 0,
                            max: Some(max),
                        },
                        Expr::U8Seq(b"z".to_vec()),
                    ]),
                },
            ],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"x".to_vec()),
            (2, b"xx".to_vec()),
            (3, b"z".to_vec()),
            (4, b"xz".to_vec()),
        ]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(10_000),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(8),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );
        assert!(dynamic.inner.tokenizer.has_virtual_residual_runtime());
        assert!(!dynamic.inner.tokenizer.has_virtual_binary_repeat_intersection());
        assert_eq!(dynamic.start().mask(), oracle.start().mask());
        for sequence in [vec![0u32], vec![3u32], vec![4u32], vec![2u32, 4]] {
            let mut dynamic_state = dynamic.start();
            let mut oracle_state = oracle.start();
            for token in sequence {
                dynamic_state.commit_token(token).unwrap();
                oracle_state.commit_token(token).unwrap();
                assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
                assert_eq!(dynamic_state.mask(), oracle_state.mask());
            }
        }

        let loaded = DynamicConstraint::load(&dynamic.save()).unwrap();
        assert!(loaded.inner.tokenizer.has_virtual_residual_runtime());
        assert_eq!(loaded.start().mask(), dynamic.start().mask());
    }

    #[test]
    fn dynamic_multiple_variable_width_giant_terminals_match_small_oracle() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |left_max, right_max| GrammarDef {
            start: 0,
            rules: vec![
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Terminal(0)],
                },
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Terminal(1)],
                },
            ],
            terminals: vec![
                Terminal::Expr {
                    id: 0,
                    expr: Expr::Repeat {
                        expr: Box::new(Expr::Choice(vec![
                            Expr::U8Seq(b"a".to_vec()),
                            Expr::U8Seq(b"bb".to_vec()),
                        ])),
                        min: 0,
                        max: Some(left_max),
                    },
                },
                Terminal::Expr {
                    id: 1,
                    expr: Expr::Repeat {
                        expr: Box::new(Expr::Choice(vec![
                            Expr::U8Seq(b"a".to_vec()),
                            Expr::U8Seq(b"cc".to_vec()),
                        ])),
                        min: 0,
                        max: Some(right_max),
                    },
                },
            ],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"bb".to_vec()),
            (2, b"cc".to_vec()),
            (3, b"abb".to_vec()),
            (4, b"acc".to_vec()),
            (5, b"x".to_vec()),
        ]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(1_000_000_000, 900_000_000),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(8, 7),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );

        assert!(dynamic.inner.tokenizer.has_virtual_binary_repeat_intersection());
        let exact_after_a = dynamic.inner.tokenizer.run(b"a");
        assert_eq!(exact_after_a.len(), 2);
        assert_eq!(dynamic.start().mask(), oracle.start().mask());
        for token in 0u32..=4 {
            let mut dynamic_state = dynamic.start();
            let mut oracle_state = oracle.start();
            dynamic_state.commit_token(token).unwrap();
            oracle_state.commit_token(token).unwrap();
            assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
            assert_eq!(dynamic_state.mask(), oracle_state.mask());
        }

        let loaded = DynamicConstraint::load(&dynamic.save()).unwrap();
        assert_eq!(dynamic.start().mask(), loaded.start().mask());
        let mut original_state = dynamic.start();
        let mut loaded_state = loaded.start();
        original_state.commit_token(3).unwrap();
        loaded_state.commit_token(3).unwrap();
        assert_eq!(original_state.is_accepting(), loaded_state.is_accepting());
        assert_eq!(original_state.mask(), loaded_state.mask());
    }

    #[test]
    fn dynamic_nested_giant_repeat_body_uses_general_residual_runtime() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |inner_max, outer_max| GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Repeat {
                    expr: Box::new(Expr::Seq(vec![
                        Expr::Repeat {
                            expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                            min: 0,
                            max: Some(inner_max),
                        },
                        Expr::U8Seq(b"b".to_vec()),
                    ])),
                    min: 0,
                    max: Some(outer_max),
                },
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"ab".to_vec()),
            (3, b"bb".to_vec()),
            (4, b"aab".to_vec()),
        ]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(10_000, 10_000),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(4, 4),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );
        assert!(dynamic.inner.tokenizer.has_virtual_residual_runtime());
        assert_eq!(dynamic.start().mask(), oracle.start().mask());
        for sequence in [vec![1u32], vec![2u32], vec![3u32], vec![4u32], vec![2u32, 1]] {
            let mut dynamic_state = dynamic.start();
            let mut oracle_state = oracle.start();
            for token in sequence {
                dynamic_state.commit_token(token).unwrap();
                oracle_state.commit_token(token).unwrap();
                assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
                assert_eq!(dynamic_state.mask(), oracle_state.mask());
            }
        }

        let loaded = DynamicConstraint::load(&dynamic.save()).unwrap();
        assert!(loaded.inner.tokenizer.has_virtual_residual_runtime());
        assert_eq!(loaded.start().mask(), dynamic.start().mask());
    }

    #[test]
    fn dynamic_transfer_load_uses_prepared_vocab_and_matches_original() {
        let vocab = vocab();
        crate::compiler::constraint_possible_matches::prepare_vocab_for_dynamic_mask(&vocab);
        let original = DynamicConstraint::from_ebnf("start ::= 'a'+ 'b'", &vocab).unwrap();
        let original_mask = original.start().mask();
        let transfer = original.into_saved();
        assert!(transfer.starts_with(&DYNAMIC_TRANSFER_MAGIC));

        let loaded = DynamicConstraint::load_with_vocab(&transfer, &vocab).unwrap();
        assert_eq!(original_mask, loaded.start().mask());
    }

    #[test]
    fn dynamic_v20_round_trips_minimal_nonempty_projected_quotients() {
        // Serialization-unit intent: a minimal deterministic nonempty
        // projected-quotient sidecar round-trips through save/load/transfer
        // and preserves mask semantics exactly. The quotients come from the
        // existing per-terminal containment builder (same exactness proofs as
        // the shared-component selector) on a two-literal fixture — no
        // reset-dispatcher sharing, no optimizer-selection assumptions, no
        // JSON lowering, no compat-mode sensitivity.
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"c".to_vec()),
            (3, b"d".to_vec()),
            (4, b"e".to_vec()),
            (5, b"f".to_vec()),
            (6, b"g".to_vec()),
        ]);
        let mut constraint = DynamicConstraint::from_glrm_grammar(
            "start start; t A ::= \"abcdef\"; t B ::= \"abcdeg\"; nt start ::= A | B;",
            &vocab,
        )
        .unwrap();
        let twin = constraint.clone();
        assert!(
            !twin
                .inner
                .dynamic_mask_vocab
                .projected_terminal_quotients_prepared(),
            "reference twin must be quotient-free (unprepared)"
        );
        assert!(
            !twin
                .inner
                .dynamic_mask_vocab
                .has_projected_terminal_quotients(),
            "reference twin must be quotient-free (empty)"
        );
        let quotients = constraint
            .inner
            .tokenizer
            .build_terminal_projected_quotients_for_containment();
        assert_eq!(
            quotients.len(),
            2,
            "both terminals of the minimal fixture must yield exact quotients"
        );
        constraint
            .inner
            .dynamic_mask_vocab
            .set_projected_terminal_quotients(quotients);
        assert!(
            constraint
                .inner
                .dynamic_mask_vocab
                .projected_terminal_quotients_prepared()
        );
        assert!(
            constraint
                .inner
                .dynamic_mask_vocab
                .has_projected_terminal_quotients()
        );
        let expected = constraint
            .inner
            .dynamic_mask_vocab
            .projected_terminal_quotients_for_artifact();
        assert!(!expected.is_empty());

        // See the Snowplow sidecar test below: this is a test-only mutation of
        // otherwise immutable post-build persistence metadata.
        constraint.external_vocab_artifact_cache = None;
        constraint.cache_external_vocab_artifact_for_save();

        let saved = constraint.save();
        assert_eq!(
            u16::from_le_bytes([saved[8], saved[9]]),
            DYNAMIC_CONSTRAINT_VERSION,
        );
        let loaded = DynamicConstraint::load(&saved).unwrap();
        assert!(
            loaded
                .inner
                .dynamic_mask_vocab
                .projected_terminal_quotients_prepared()
        );
        assert!(
            loaded
                .inner
                .dynamic_mask_vocab
                .has_projected_terminal_quotients()
        );
        assert_eq!(
            bincode::serialize(
                &loaded
                    .inner
                    .dynamic_mask_vocab
                    .projected_terminal_quotients_for_artifact()
            )
            .unwrap(),
            bincode::serialize(&expected).unwrap(),
        );

        let transfer = constraint.clone().into_saved();
        assert_eq!(
            u16::from_le_bytes([transfer[8], transfer[9]]),
            DYNAMIC_TRANSFER_VERSION,
        );
        let transferred = DynamicConstraint::load_with_vocab(&transfer, &vocab).unwrap();
        assert!(
            transferred
                .inner
                .dynamic_mask_vocab
                .projected_terminal_quotients_prepared()
        );
        assert!(
            transferred
                .inner
                .dynamic_mask_vocab
                .has_projected_terminal_quotients()
        );
        assert_eq!(
            bincode::serialize(
                &transferred
                    .inner
                    .dynamic_mask_vocab
                    .projected_terminal_quotients_for_artifact()
            )
            .unwrap(),
            bincode::serialize(&expected).unwrap(),
        );

        // Semantic exactness: prepared quotients must not change the accepted
        // language or any mask. Differential of the quotient-free twin against
        // the quotient-bearing original plus both reloaded artifacts over
        // accept/reject paths.
        let paths: Vec<Vec<u32>> = vec![
            vec![],
            vec![0],
            vec![0, 1],
            vec![0, 1, 2, 3, 4, 5],
            vec![0, 1, 2, 3, 4, 6],
            vec![1],
            vec![0, 0],
        ];
        for path in &paths {
            let mut states = [
                twin.start(),
                constraint.start(),
                loaded.start(),
                transferred.start(),
            ];
            let mut failed = false;
            for &token in path {
                let masks: Vec<_> = states.iter().map(|state| state.mask()).collect();
                for (index, mask) in masks.iter().enumerate().skip(1) {
                    assert_eq!(
                        *mask, masks[0],
                        "mask agreement at {path:?} before token {token} (state {index})",
                    );
                }
                let results: Vec<bool> = states
                    .iter_mut()
                    .map(|state| state.commit_token(token).is_ok())
                    .collect();
                for (index, result) in results.iter().enumerate().skip(1) {
                    assert_eq!(
                        *result, results[0],
                        "commit agreement at {path:?} token {token} (state {index})",
                    );
                }
                if !results[0] {
                    // All four agree on rejection; the API's post-failure
                    // guarantee is `is_rejected`, so assert that (not masks or
                    // acceptance) and stop this path.
                    for (index, state) in states.iter().enumerate() {
                        assert!(
                            state.is_rejected(),
                            "rejected state {index} at {path:?} token {token} must report is_rejected",
                        );
                    }
                    failed = true;
                    break;
                }
            }
            if failed {
                continue;
            }
            // Every commit succeeded: compare the post-path masks (initial and
            // after-every-successful-commit coverage) and acceptance.
            let masks: Vec<_> = states.iter().map(|state| state.mask()).collect();
            for (index, mask) in masks.iter().enumerate().skip(1) {
                assert_eq!(
                    *mask, masks[0],
                    "final mask agreement at {path:?} (state {index})",
                );
            }
            let accepting: Vec<bool> =
                states.iter().map(|state| state.is_accepting()).collect();
            for (index, value) in accepting.iter().enumerate().skip(1) {
                assert_eq!(
                    *value, accepting[0],
                    "acceptance agreement at {path:?} (state {index})",
                );
            }
        }
        // Explicit acceptance/rejection pins on every artifact (not only
        // cross-artifact equality), so the differential above is not vacuous.
        for (label, artifact) in [
            ("twin", &twin),
            ("original", &constraint),
            ("loaded", &loaded),
            ("transferred", &transferred),
        ] {
            for path in [vec![0u32, 1, 2, 3, 4, 5], vec![0, 1, 2, 3, 4, 6]] {
                let mut state = artifact.start();
                for &token in &path {
                    state.commit_token(token).unwrap();
                }
                assert!(state.is_accepting(), "{label} must accept {path:?}");
            }
            let mut state = artifact.start();
            assert!(
                !state.is_accepting(),
                "{label} must not accept the empty prefix"
            );
            assert!(
                state.commit_token(1).is_err(),
                "{label} must reject a leading b"
            );
            let mut state = artifact.start();
            state.commit_token(0).unwrap();
            assert!(
                !state.is_accepting(),
                "{label} must not accept the proper prefix [a]"
            );
            assert!(
                state.commit_token(0).is_err(),
                "{label} must reject the aa divergence"
            );
        }
    }

    #[test]
    fn dynamic_v20_round_trips_snowplow_optimizer_selected_quotients() {
        let _env_lock = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let (schema, vocab) = projected_quotient_fixture::schema_and_vocab();
        // Integration tolerance: whatever the shared-component optimizer
        // selects for Snowplow (currently nothing — the lexer isolates this
        // fixture's large terminals), the sidecar round-trips and masks agree
        // end to end. Nonempty wire-format coverage lives in the minimal
        // unit test above; this test must NOT pin the optimizer's selection.
        // Do not restore production sharing just to make Snowplow nonempty.
        let schema_value: serde_json::Value = serde_json::from_str(schema).unwrap();
        let named = crate::import::json_schema::schema_to_named_grammar(&schema_value).unwrap();
        let mut factored = crate::grammar::factoring::factor_named_grammar(named);
        crate::import::json_schema::prepare_named_grammar(&mut factored).unwrap();
        let grammar = crate::grammar::ast::lower(&factored).unwrap();
        let mut constraint = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar,
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::LegacyRowBisim,
        )
        .unwrap();
        let quotients = constraint
            .inner
            .tokenizer
            .build_shared_component_terminal_projected_quotients(256);
        constraint
            .inner
            .dynamic_mask_vocab
            .set_projected_terminal_quotients(quotients);
        let expected = constraint
            .inner
            .dynamic_mask_vocab
            .projected_terminal_quotients_for_artifact();

        // The private mutation above occurs after public finalization froze the
        // canonical transfer artifact. Refresh it explicitly for wire-format
        // coverage; ordinary callers cannot mutate a compiled constraint.
        constraint.external_vocab_artifact_cache = None;
        constraint.cache_external_vocab_artifact_for_save();

        let saved = constraint.save();
        assert_eq!(
            u16::from_le_bytes([saved[8], saved[9]]),
            DYNAMIC_CONSTRAINT_VERSION,
        );
        let loaded = DynamicConstraint::load(&saved).unwrap();
        assert!(loaded.inner.dynamic_mask_vocab.projected_terminal_quotients_prepared());
        assert_eq!(
            bincode::serialize(
                &loaded
                    .inner
                    .dynamic_mask_vocab
                    .projected_terminal_quotients_for_artifact()
            )
            .unwrap(),
            bincode::serialize(&expected).unwrap(),
        );
        assert_eq!(loaded.start().mask(), constraint.start().mask());

        let transfer = constraint.clone().into_saved();
        assert_eq!(
            u16::from_le_bytes([transfer[8], transfer[9]]),
            DYNAMIC_TRANSFER_VERSION,
        );
        let transferred = DynamicConstraint::load_with_vocab(&transfer, &vocab).unwrap();
        assert!(
            transferred
                .inner
                .dynamic_mask_vocab
                .projected_terminal_quotients_prepared()
        );
        assert_eq!(
            bincode::serialize(
                &transferred
                    .inner
                    .dynamic_mask_vocab
                    .projected_terminal_quotients_for_artifact()
            )
            .unwrap(),
            bincode::serialize(&expected).unwrap(),
        );
        assert_eq!(transferred.start().mask(), constraint.start().mask());

        let mut original_state = constraint.start();
        let mut loaded_state = loaded.start();
        let mut transferred_state = transferred.start();
        for &token in projected_quotient_fixture::replay_ids() {
            let expected = original_state.mask();
            assert_eq!(loaded_state.mask(), expected, "self-contained load before token {token}");
            assert_eq!(
                transferred_state.mask(),
                expected,
                "transfer load before token {token}"
            );
            original_state.commit_token(token).unwrap();
            loaded_state.commit_token(token).unwrap();
            transferred_state.commit_token(token).unwrap();
        }
        assert_eq!(loaded_state.mask(), original_state.mask());
        assert_eq!(transferred_state.mask(), original_state.mask());
    }

    #[test]
    fn dynamic_v20_round_trips_prepared_empty_projected_terminal_quotients() {
        let vocab = vocab();
        let mut constraint = DynamicConstraint::from_ebnf("start ::= 'a'+ 'b'", &vocab).unwrap();
        constraint
            .inner
            .dynamic_mask_vocab
            .set_projected_terminal_quotients(Vec::new());
        assert!(constraint.inner.dynamic_mask_vocab.projected_terminal_quotients_prepared());
        assert!(!constraint.inner.dynamic_mask_vocab.has_projected_terminal_quotients());

        // See the nonempty sidecar test above: this is a test-only mutation of
        // otherwise immutable post-build persistence metadata.
        constraint.external_vocab_artifact_cache = None;
        constraint.cache_external_vocab_artifact_for_save();

        let loaded = DynamicConstraint::load(&constraint.save()).unwrap();
        assert!(loaded.inner.dynamic_mask_vocab.projected_terminal_quotients_prepared());
        assert!(!loaded.inner.dynamic_mask_vocab.has_projected_terminal_quotients());
        assert_eq!(loaded.start().mask(), constraint.start().mask());

        let transfer = constraint.clone().into_saved();
        let transferred = DynamicConstraint::load_with_vocab(&transfer, &vocab).unwrap();
        assert!(
            transferred
                .inner
                .dynamic_mask_vocab
                .projected_terminal_quotients_prepared()
        );
        assert!(!transferred.inner.dynamic_mask_vocab.has_projected_terminal_quotients());
        assert_eq!(transferred.start().mask(), constraint.start().mask());
    }

    #[test]
    fn current_dynamic_transfer_preserves_unprepared_projected_terminal_quotients() {
        let vocab = vocab();
        let constraint = DynamicConstraint::from_glrm_grammar(
            r#"
start start;
t A ::= /a{0,1000000000}/;
nt start ::= A;
"#,
            &vocab,
        )
        .unwrap();
        assert!(constraint.inner.tokenizer.has_any_virtual_runtime());
        assert!(
            !constraint
                .inner
                .dynamic_mask_vocab
                .projected_terminal_quotients_prepared(),
            "fresh dynamic compilation should defer projected-terminal proof artifacts",
        );

        let transfer = constraint.into_saved();
        let transferred = DynamicConstraint::load_with_vocab(&transfer, &vocab).unwrap();
        assert!(
            !transferred
                .inner
                .dynamic_mask_vocab
                .projected_terminal_quotients_prepared(),
            "transfer load must preserve unprepared-empty rather than converting it to prepared-empty",
        );
    }

    #[test]
    fn self_contained_dynamic_load_with_vocab_shares_exact_vocab() {
        let vocab = vocab();
        let original = DynamicConstraint::from_ebnf("start ::= 'a'+ 'b'", &vocab).unwrap();
        let loaded = DynamicConstraint::load_with_vocab(&original.save(), &vocab).unwrap();

        assert!(std::sync::Arc::ptr_eq(
            &loaded.inner.token_bytes,
            &vocab.entries_arc(),
        ));
        assert!(std::sync::Arc::ptr_eq(
            &loaded
                .inner
                .late_bind_vocab
                .get()
                .expect("dynamic load_with_vocab should seed late-bind vocab")
                .entries_arc(),
            &vocab.entries_arc(),
        ));
    }

    #[test]
    fn precompiled_dynamic_artifacts_require_rebuild() {
        let vocab = vocab();
        let constraint = DynamicConstraint::from_ebnf("start ::= 'a'+ 'b'", &vocab).unwrap();
        let mut bytes = constraint.save();
        bytes[8..10].copy_from_slice(&8u16.to_le_bytes());
        let error = DynamicConstraint::load(&bytes).unwrap_err().to_string();
        assert!(error.contains("unsupported dynamic constraint artifact version"));
    }

    #[test]
    fn direct_regular_dynamic_constraint_uses_dynamic_runtime() {
        let vocab = vocab();
        let mut grammar = String::from("start: r0\n");
        for index in 0..63 {
            grammar.push_str(&format!("r{index}: \"a\" r{}\n", index + 1));
        }
        grammar.push_str("r63: \"b\"\n");

        let normal = compile_compressed_static(&grammar, &vocab);
        let dynamic = compile_compressed_dynamic(&grammar, &vocab);
        assert_eq!(dynamic.inner.table.num_rules, 0);
        assert!(dynamic.inner.uses_dynamic_runtime());
        assert_eq!(normal.start().mask(), dynamic.start().mask());
    }

    #[test]
    fn compressed_static_unions_mixed_l1_l2p_token_lengths() {
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"aa".to_vec())]);
        let mut grammar = String::from("start: r0\n");
        for index in 0..63 {
            grammar.push_str(&format!("r{index}: \"a\" r{}\n", index + 1));
        }
        grammar.push_str("r63: \"a\"\n");

        let static_constraint = compile_compressed_static(&grammar, &vocab);
        let dynamic_constraint = compile_compressed_dynamic(&grammar, &vocab);
        assert!(!static_constraint.uses_dynamic_runtime());
        assert_eq!(static_constraint.table.num_rules, 0);
        assert!(
            !static_constraint.parser_top_accept_parts.is_empty(),
            "regression must exercise the direct parser-acceptance summaries",
        );

        let static_mask = static_constraint.start().mask();
        assert_ne!(static_mask[0] & (1 << 0), 0, "single-byte token must be allowed");
        assert_ne!(static_mask[0] & (1 << 1), 0, "two-byte token must be allowed");
        assert_eq!(static_mask, dynamic_constraint.start().mask());
    }

    #[test]
    fn compressed_right_linear_plus_loop_commits_terminal_at_token_boundary() {
        let mut grammar = String::from("start: H s0\nplus_line: PLUS_LINE\n");
        let n = 15;
        for i in 0..n {
            grammar.push_str(&format!(
                "s{i}: line{i}{}\n",
                if i + 1 < n {
                    format!(" | s{}", i + 1)
                } else {
                    String::new()
                }
            ));
        }
        for i in 0..n {
            grammar.push_str(&format!(
                "line{i}: plus_line* SRC_{i} plus_line* {}\n",
                if i + 1 < n {
                    format!("line{}?", i + 1)
                } else {
                    String::new()
                }
            ));
        }
        grammar.push_str("H: \"h\\n\"\nPLUS_LINE: /\\+[^\\n\\r]*\\n/\n");
        for i in 0..n {
            grammar.push_str(&format!("SRC_{i}: \" a{i}\\n\"\n"));
        }
        let vocab = Vocab::new(vec![
            (0, b"h\n".to_vec()),
            (1, b"+x".to_vec()),
            (2, b"\n".to_vec()),
            (3, b" a0\n".to_vec()),
        ]);
        let constraint = compile_compressed_static(&grammar, &vocab);
        assert!(!constraint.uses_dynamic_runtime());
        let mut state = constraint.start();

        state.commit_token(0).unwrap();
        state.commit_token(1).unwrap();
        assert_ne!(state.mask()[0] & (1 << 2), 0);
        state.commit_token(2).unwrap();
        assert_ne!(state.mask()[0] & (1 << 3), 0);
        state.commit_token(3).unwrap();
        assert!(state.is_accepting());
    }

    #[test]
    fn direct_regular_static_constraint_roundtrips_and_matches_dynamic() {
        let vocab = vocab();
        let mut grammar = String::from("start: r0\n");
        for index in 0..63 {
            grammar.push_str(&format!("r{index}: \"a\" r{}\n", index + 1));
        }
        grammar.push_str("r63: \"b\"\n");

        let constraint = crate::Constraint::from_lark(&grammar, &vocab).unwrap();
        assert!(!constraint.uses_dynamic_runtime());
        assert!(constraint.possible_matches_complete);
        let dynamic = compile_compressed_dynamic(&grammar, &vocab);
        assert!(!dynamic.inner.possible_matches_complete);

        let mut static_state = constraint.start();
        let mut dynamic_state = dynamic.start();
        assert_eq!(static_state.mask(), dynamic_state.mask());
        assert_eq!(static_state.forced(), dynamic_state.forced());

        for token in [3, 3] {
            static_state.commit_token(token).unwrap();
            dynamic_state.commit_token(token).unwrap();
            assert_eq!(static_state.mask(), dynamic_state.mask());
            assert_eq!(static_state.is_accepting(), dynamic_state.is_accepting());
        }

        let before_third = static_state.mask();
        let checkpoint = static_state.clone();
        static_state.commit_token(0).unwrap();
        dynamic_state.commit_token(0).unwrap();
        let after_third = static_state.mask();
        assert_eq!(after_third, dynamic_state.mask());
        static_state = checkpoint;
        assert_eq!(static_state.mask(), before_third);
        static_state.commit_token(0).unwrap();
        assert_eq!(static_state.mask(), after_third);

        let loaded = crate::Constraint::load(&constraint.save()).unwrap();
        assert!(!loaded.uses_dynamic_runtime());
        let mut loaded_state = loaded.start();
        let mut original_state = constraint.start();
        assert_eq!(loaded_state.mask(), original_state.mask());
        for token in [3, 3, 0] {
            loaded_state.commit_token(token).unwrap();
            original_state.commit_token(token).unwrap();
            assert_eq!(loaded_state.mask(), original_state.mask());
            assert_eq!(loaded_state.is_accepting(), original_state.is_accepting());
        }
    }

    #[test]
    fn non_regular_constraint_keeps_static_backend() {
        let vocab = vocab();
        let constraint = crate::Constraint::from_ebnf(
            "start ::= 'a' start 'b' | ''",
            &vocab,
        )
        .unwrap();
        assert!(!constraint.uses_dynamic_runtime());
    }

    #[test]
    fn dynamic_forced_uses_dynamic_masks() {
        let vocab = Vocab::new(vec![(0, b"a".to_vec())]);
        let constraint = DynamicConstraint::from_ebnf("start ::= 'a'", &vocab).unwrap();
        assert_eq!(constraint.start().forced(), vec![0]);
    }
