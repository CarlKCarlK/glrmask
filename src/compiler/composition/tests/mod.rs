use super::structural_sharing::{
    composition_terminal_classes, contextually_share_composed_states,
    quotient_composed_table_structurally, structural_nonterminal_classes,
};
use super::parser_union::{
    compose_component_parser_dwas_and_possible_matches,
    determinize_epsilon_free_component_union, parser_nwa_preserve_defaults,
    supports_overlap_local_union, union_boundary_parser_dwa,
};

    use super::*;
    use crate::compiler::glr::table::{
        GlrTableConstruction, SubgrammarTableInput, compose_subgrammar_tables,
    };
    use crate::grammar::flat::TerminalID;
    include!("minbound_trigram.rs");
    include!("minbound_factor.rs");
    include!("minbound_scoped_ignore.rs");

    fn byte_vocab() -> Vocab {
        Vocab::new(
            (0u32..=255)
                .map(|byte| (byte, vec![byte as u8]))
                .collect(),
        )
    }

    fn terminal(constraint: &Constraint, name: &str) -> TerminalID {
        constraint
            .terminal_display_names
            .iter()
            .position(|candidate| candidate == name)
            .unwrap() as u32
    }

    /// Ordinary (non-composition) compile of a grammar-level inline reference:
    /// structurally rewrite `extern grammar` occurrences to inline subgrammars
    /// and compile through the normal named-grammar path, using the ordinary
    /// GLRM default table construction (`ExperimentalCoreMerged`).
    fn compile_inline_source_oracle(
        parent: &str,
        children: &[(&str, &str)],
        vocab: &Vocab,
    ) -> Constraint {
        let named = crate::grammar::glrm::from_glrm_with_inline_subgrammars(parent, children)
            .expect("grammar-level inline subgrammar rewrite");
        crate::import::compile_from_named_grammar(
            named,
            vocab,
            "inline_source_oracle",
            GlrTableConstruction::ExperimentalCoreMerged,
            &[],
        )
        .expect("ordinary compile of grammar-level inline reference")
    }

    /// Differential check: the grammar-level inline reference (ordinary
    /// compile) must agree with explicit segmented Static and Dynamic built
    /// from the same source/vocabulary, including fused model tokens that cross
    /// the child boundary.
    #[cfg(feature = "internal-api")]
    #[test]
    fn grammar_level_inline_reference_matches_segmented_static_and_dynamic() {
        let vocab = Vocab::new(vec![
            (0, b"X".to_vec()),
            (1, b"a".to_vec()),
            (2, b"b".to_vec()),
            (3, b"!".to_vec()),
            (4, b"Xa".to_vec()),
            (5, b"b!".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "X" SUB "!";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child ::= "a" "b";
            "#,
            &vocab,
        )
        .unwrap();

        let inline_parent = r#"
            start document;
            extern grammar SUB;
            nt document ::= "X" SUB "!";
        "#;
        let inline_children = [("SUB", "start child; nt child ::= \"a\" \"b\";")];
        let named = crate::grammar::glrm::from_glrm_with_inline_subgrammars(
            inline_parent,
            &inline_children,
        )
        .unwrap();
        let inline = crate::__private::compile_named_grammar(named.clone(), &vocab).unwrap();
        let compiled = crate::__private::compile_named_grammar(named, &vocab).unwrap();
        let segmented_static = parent
            .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
            .unwrap();
        let segmented_dynamic = parent
            .compose_linked_children_for_test_dynamic(&[("SUB", &child)], &vocab)
            .unwrap();

        for (name, constraint) in [
            ("inline", &inline),
            ("compile_named_grammar", &compiled),
            ("segmented_static", &segmented_static),
            ("segmented_dynamic", &segmented_dynamic),
        ] {
            let mut state = constraint.start();
            assert!(state.commit_bytes(b"Xa").is_ok(), "{name} rejected fused Xa");
            assert!(state.commit_bytes(b"b").is_ok(), "{name} rejected b");
            assert!(state.commit_bytes(b"!").is_ok(), "{name} rejected !");
            assert!(state.is_accepting(), "{name} did not accept Xab!");
        }
        assert_constraints_mask_equivalent_on_reachable_prefixes_labeled(
            &inline,
            &segmented_static,
            &vocab,
            4,
            "inline-vs-segmented-static",
        );
        assert_constraints_mask_equivalent_on_reachable_prefixes_labeled(
            &segmented_dynamic,
            &segmented_static,
            &vocab,
            4,
            "segmented-dynamic-vs-segmented-static",
        );
    }


    #[test]
    fn dynamic_recursive_unprepared_inputs_use_minimal_shared_linker() {
        let vocab = Vocab::new(vec![
            (0, b"X".to_vec()), (1, b"a".to_vec()), (2, b"b".to_vec()),
            (3, b"!".to_vec()), (4, b"Xa".to_vec()), (5, b"b!".to_vec()),
            (6, b"Xab!".to_vec()), (7, b"XX".to_vec()), (8, b"c".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"start document; t SUB ::= @token(999); nt document ::= "X" SUB "!";"#,
            &vocab,
        ).unwrap();
        let mut child = Constraint::from_glrm_grammar(
            r#"start child; t A ::= "a"; t B ::= "b"; nt child ::= A B;"#, &vocab,
        ).unwrap();
        child.ensure_composition_reset_tokens_by_terminal();
        child.serialized_artifact_cache = None;
        let inner = parent.compose_linked_children_for_test_dynamic(
            &[("SUB", &child)], &vocab,
        ).unwrap();
        let parent_bytes = parent.save();
        for (expected_text, child_bytes) in [("Xab!", child.save()), ("XXab!!", inner.save())] {
            let loaded_child = Arc::new(Constraint::load_with_vocab(&child_bytes, &vocab).unwrap());
            let was_materialized = loaded_child.composition_link_metadata_materialized;
            if expected_text == "Xab!" {
                assert!(loaded_child.deferred_composition_metadata_blob.is_some());
                assert!(!was_materialized, "ordinary loaded child must exercise deferred link preparation");
            }
            let reference = Constraint::from_glrm_grammar(
                &format!("start document; nt document ::= {expected_text:?};"), &vocab,
            ).unwrap();
            for shared in [false, true] {
                let loaded_parent = Constraint::load_with_vocab(&parent_bytes, &vocab).unwrap();
                let inputs = [CompiledSubgrammarInput {
                    placeholder_terminal: terminal(&loaded_parent, "SUB"),
                    additional_placeholder_terminals: &[],
                    constraint: loaded_child.as_ref(),
                }];
                let bound = if shared {
                    compose_constraints_owned_parent_segmented_shared(
                        loaded_parent, &inputs, &[Arc::clone(&loaded_child)], &vocab,
                        SegmentedBoundaryBackend::Dynamic,
                    )
                } else {
                    compose_constraints_owned_parent_segmented(
                        loaded_parent, &inputs, &vocab, SegmentedBoundaryBackend::Dynamic,
                    )
                }.unwrap().constraint;
                assert!(bound.uses_compact_segmented_parser_runtime());
                assert_eq!(bound.table.num_states, 0, "binding must publish only the fast coordinator");
                let overlay = bound.static_dynamic_overlay.as_ref().unwrap();
                assert!(overlay.segmented_parser_components.iter().all(|component| {
                    component.global_to_local_parser_state.is_empty()
                }), "binding must not construct flattened parser projections");
                assert_constraints_mask_equivalent_on_reachable_prefixes_labeled(
                    &bound, &reference, &vocab, 6, "unprepared dynamic link vs monolithic",
                );
                let restored = Constraint::load_with_vocab(bound.save(), &vocab).unwrap();
                assert_constraints_mask_equivalent_on_reachable_prefixes_labeled(
                    &restored, &reference, &vocab, 6, "unprepared dynamic link reload",
                );
                assert_eq!(loaded_child.composition_link_metadata_materialized, was_materialized,
                    "preparing a retained child must not mutate the caller's shared/borrowed input");
                assert_eq!(loaded_child.save(), child_bytes);
            }
        }
    }

    #[test]
    fn dynamic_recursive_shared_fast_matches_borrowed_and_reload() {
        let vocab = Vocab::new(vec![
            (0, b"X".to_vec()),
            (1, b"a".to_vec()),
            (2, b"b".to_vec()),
            (3, b"!".to_vec()),
            (4, b"Xa".to_vec()),
            (5, b"b!".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "X" SUB "!";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child ::= "a" "b";
            "#,
            &vocab,
        )
        .unwrap();
        let shared_child = Arc::new(child.clone());
        let placeholder_terminal = terminal(&parent, "SUB");
        let inputs = [CompiledSubgrammarInput {
            placeholder_terminal,
            additional_placeholder_terminals: &[],
            constraint: shared_child.as_ref(),
        }];
        let fast = compose_constraints_owned_parent_segmented_shared(
            parent.clone(),
            &inputs,
            &[Arc::clone(&shared_child)],
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .unwrap()
        .constraint;
        let borrowed = parent
            .compose_linked_children_for_test_dynamic(&[("SUB", &child)], &vocab)
            .unwrap();

        assert!(fast.uses_compact_segmented_parser_runtime());
        let overlay = fast.static_dynamic_overlay.as_ref().unwrap();
        assert!(
            overlay
                .recursive_tokenizer_internal_tsids
                .get()
                .is_none(),
            "all-DynamicDirect fast coordinator should omit the redundant recursive TSID relation",
        );
        assert!(
            overlay
                .segmented_parser_components
                .iter()
                .all(|component| component.global_to_local_parser_state.is_empty()),
            "fast coordinator should not retain materialized composed-state projections",
        );

        assert_constraints_mask_equivalent_on_reachable_prefixes_labeled(
            &fast,
            &borrowed,
            &vocab,
            4,
            "dynamic-shared-fast-vs-borrowed",
        );

        let bytes = fast.save();
        let reloaded = Constraint::load_with_vocab(&bytes, &vocab).unwrap();
        assert!(reloaded.uses_compact_segmented_parser_runtime());
        assert_constraints_mask_equivalent_on_reachable_prefixes_labeled(
            &fast,
            &reloaded,
            &vocab,
            4,
            "dynamic-shared-fast-vs-reload",
        );

        // A later static-boundary link is allowed to reconstruct compiler-only
        // flattened views on demand from the retained recursive component tree.
        // The fast dynamic build itself must never pay this cost.
        let outer_parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t OUTER ::= @token(998);
                nt document ::= "X" OUTER "!";
            "#,
            &vocab,
        )
        .unwrap();
        let static_outer = outer_parent
            .compose_linked_children_for_test(&[("OUTER", &fast)], &vocab)
            .unwrap();
        let dynamic_outer = Constraint::from_glrm_grammar(
            r#"
                start document;
                t OUTER ::= @token(998);
                nt document ::= "X" OUTER "!";
            "#,
            &vocab,
        )
        .unwrap()
        .compose_linked_children_for_test_dynamic(&[("OUTER", &fast)], &vocab)
        .unwrap();
        assert_constraints_mask_equivalent_on_reachable_prefixes_labeled(
            &static_outer,
            &dynamic_outer,
            &vocab,
            4,
            "dynamic-shared-fast-static-upgrade-vs-dynamic",
        );
    }

    /// Differential check for a nested external child+grandchild through the
    /// grammar-level inline reference versus explicit segmented Static.
    #[cfg(feature = "internal-api")]
    #[test]
    fn grammar_level_inline_reference_nested_matches_segmented_static() {
        let vocab = byte_vocab();
        let inner = Constraint::from_glrm_grammar(
            r#"
                start inner;
                nt inner ::= "b";
            "#,
            &vocab,
        )
        .unwrap();
        let outer_parent = Constraint::from_glrm_grammar(
            r#"
                start outer;
                t INNER ::= @token(998);
                nt outer ::= "a" INNER;
            "#,
            &vocab,
        )
        .unwrap();
        let outer = outer_parent
            .compose_linked_children_for_test(&[("INNER", &inner)], &vocab)
            .unwrap();
        let document_parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t OUTER ::= @token(999);
                nt document ::= "<" OUTER ">";
            "#,
            &vocab,
        )
        .unwrap();
        let segmented = document_parent
            .compose_linked_children_for_test(&[("OUTER", &outer)], &vocab)
            .unwrap();

        let named = crate::grammar::glrm::from_glrm_with_inline_subgrammars(
            r#"
                start document;
                extern grammar OUTER;
                nt document ::= "<" OUTER ">";
            "#,
            &[
                (
                    "OUTER",
                    "start outer; extern grammar INNER; nt outer ::= \"a\" INNER;",
                ),
                ("OUTER::INNER", "start inner; nt inner ::= \"b\";"),
            ],
        )
        .unwrap();
        let inline = crate::__private::compile_named_grammar(named, &vocab).unwrap();

        let mut inline_state = inline.start();
        let mut segmented_state = segmented.start();
        assert!(inline_state.commit_bytes(b"<ab>").is_ok());
        assert!(segmented_state.commit_bytes(b"<ab>").is_ok());
        assert!(inline_state.is_accepting());
        assert!(segmented_state.is_accepting());
        assert_constraints_mask_equivalent_on_reachable_prefixes_labeled(
            &inline,
            &segmented,
            &vocab,
            4,
            "nested-inline-vs-segmented-static",
        );
    }

    trait ComposeLinkedChildrenForTest {
        fn compose_linked_children_for_test(
            &self,
            children: &[(&str, &Constraint)],
            vocab: &Vocab,
        ) -> crate::Result<Constraint>;

        fn compose_linked_children_for_test_owned(
            self,
            children: &[(&str, &Constraint)],
            vocab: &Vocab,
        ) -> crate::Result<Constraint>;

        /// Flattened composition (baseline backend). For tests whose subject
        /// is flattened-table behavior (skip-terminal materialization,
        /// runtime-product selection, control-edge sequencing) or pre-existing
        /// static gaps (scoped ignores) that Phase 2b preserves as-is.
        #[allow(deprecated)]
        fn compose_linked_children_for_test_flattened(
            &self,
            children: &[(&str, &Constraint)],
            vocab: &Vocab,
        ) -> crate::Result<Constraint>;

        /// Exact dynamic segmented composition. Reference route for nested
        /// links (nullable nested links decline loudly on the static route;
        /// this backend accepts those shapes).
        fn compose_linked_children_for_test_dynamic(
            &self,
            children: &[(&str, &Constraint)],
            vocab: &Vocab,
        ) -> crate::Result<Constraint>;
    }

    impl ComposeLinkedChildrenForTest for Constraint {
        fn compose_linked_children_for_test(
            &self,
            children: &[(&str, &Constraint)],
            vocab: &Vocab,
        ) -> crate::Result<Constraint> {
            let mut inputs = Vec::with_capacity(children.len());
            let mut seen = BTreeSet::new();
            for &(name, child) in children {
                let placeholder_terminal = terminal(self, name);
                if !seen.insert(placeholder_terminal) {
                    return Err(crate::GlrMaskError::Compilation(format!(
                        "parent placeholder terminal {name:?} was supplied more than once",
                    )));
                }
                inputs.push(CompiledSubgrammarInput {
                    placeholder_terminal,
                    additional_placeholder_terminals: &[],
                    constraint: child,
                });
            }
            compose_constraints_owned_parent_segmented(
                self.clone(),
                &inputs,
                vocab,
                SegmentedBoundaryBackend::StaticParserDwa,
            )
            .map(|composition| composition.constraint)
            .map_err(crate::GlrMaskError::Compilation)
        }

        fn compose_linked_children_for_test_owned(
            self,
            children: &[(&str, &Constraint)],
            vocab: &Vocab,
        ) -> crate::Result<Constraint> {
            let mut inputs = Vec::with_capacity(children.len());
            let mut seen = BTreeSet::new();
            for &(name, child) in children {
                let placeholder_terminal = terminal(&self, name);
                if !seen.insert(placeholder_terminal) {
                    return Err(crate::GlrMaskError::Compilation(format!(
                        "parent placeholder terminal {name:?} was supplied more than once",
                    )));
                }
                inputs.push(CompiledSubgrammarInput {
                    placeholder_terminal,
                    additional_placeholder_terminals: &[],
                    constraint: child,
                });
            }
            compose_constraints_owned_parent_segmented(
                self,
                &inputs,
                vocab,
                SegmentedBoundaryBackend::StaticParserDwa,
            )
            .map(|composition| composition.constraint)
            .map_err(crate::GlrMaskError::Compilation)
        }

        #[allow(deprecated)]
        fn compose_linked_children_for_test_flattened(
            &self,
            children: &[(&str, &Constraint)],
            vocab: &Vocab,
        ) -> crate::Result<Constraint> {
            let mut inputs = Vec::with_capacity(children.len());
            let mut seen = BTreeSet::new();
            for &(name, child) in children {
                let placeholder_terminal = terminal(self, name);
                if !seen.insert(placeholder_terminal) {
                    return Err(crate::GlrMaskError::Compilation(format!(
                        "parent placeholder terminal {name:?} was supplied more than once",
                    )));
                }
                inputs.push(CompiledSubgrammarInput {
                    placeholder_terminal,
                    additional_placeholder_terminals: &[],
                    constraint: child,
                });
            }
            compose_constraints(self, &inputs, vocab)
                .map(|composition| composition.constraint)
                .map_err(crate::GlrMaskError::Compilation)
        }

        fn compose_linked_children_for_test_dynamic(
            &self,
            children: &[(&str, &Constraint)],
            vocab: &Vocab,
        ) -> crate::Result<Constraint> {
            let mut inputs = Vec::with_capacity(children.len());
            let mut seen = BTreeSet::new();
            for &(name, child) in children {
                let placeholder_terminal = terminal(self, name);
                if !seen.insert(placeholder_terminal) {
                    return Err(crate::GlrMaskError::Compilation(format!(
                        "parent placeholder terminal {name:?} was supplied more than once",
                    )));
                }
                inputs.push(CompiledSubgrammarInput {
                    placeholder_terminal,
                    additional_placeholder_terminals: &[],
                    constraint: child,
                });
            }
            compose_constraints_owned_parent_segmented(
                self.clone(),
                &inputs,
                vocab,
                SegmentedBoundaryBackend::Dynamic,
            )
            .map(|composition| composition.constraint)
            .map_err(crate::GlrMaskError::Compilation)
        }
    }

    #[test]
    fn structural_sharing_quotients_duplicate_child_lr_regions() {
        let vocab = Vocab::new(vec![
            (0, b"<abc>,<abc>".to_vec()),
            (2, b"<".to_vec()),
            (3, b"a".to_vec()),
            (4, b"b".to_vec()),
            (7, b"c".to_vec()),
            (5, b">,<".to_vec()),
            (6, b">".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t LEFT ::= @token(998);
                t RIGHT ::= @token(999);
                nt document ::= "<" LEFT ">,<" RIGHT ">";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt value ::= "a" "b" "c";
                nt child ::= value;
            "#,
            &vocab,
        )
        .unwrap();
        let loaded_child = Constraint::load(&child.save()).unwrap();
        let children = [
            CompiledSubgrammarInput {
                placeholder_terminal: terminal(&parent, "LEFT"),
                additional_placeholder_terminals: &[],
constraint: &child,
            },
            CompiledSubgrammarInput {
                placeholder_terminal: terminal(&parent, "RIGHT"),
                additional_placeholder_terminals: &[],
constraint: &loaded_child,
            },
        ];
        let table_inputs = children
            .iter()
            .map(|child| SubgrammarTableInput {
                placeholder_terminal: child.placeholder_terminal,
                additional_placeholder_terminals: &[],
table: &child.constraint.table,
                ignore_terminal: child.constraint.ignore_terminal,
                start_nullable: child.constraint.table.embedded_start_nullable(),
            })
            .collect::<Vec<_>>();
        let mut composed = compose_subgrammar_tables(&parent.table, None, &table_inputs).unwrap();
        let before = composed.table.num_states;
        let terminal_analysis = composition_terminal_classes(&parent, &children, &composed);
        assert!(
            loaded_child.tokenizer.terminal_exprs().is_none()
                && loaded_child.retained_terminal_exprs().is_some(),
            "current serialized artifacts should retain terminal proof expressions lazily",
        );
        assert!(
            terminal_analysis
                .classes
                .iter()
                .enumerate()
                .any(|(terminal, &class)| terminal as u32 != class),
            "the separately loaded child should still form terminal aliases",
        );
        let nonterminal_classes = structural_nonterminal_classes(
            &composed.table,
            &terminal_analysis.classes,
            &composed.boundary_nonterminals,
        );
        let (candidate_groups, contextual_saved) =
            contextually_share_composed_states(
                &mut composed,
                &parent,
                &children,
                &terminal_analysis.classes,
                &nonterminal_classes,
            );
        let _ = quotient_composed_table_structurally(
            &mut composed,
            &terminal_analysis,
            &nonterminal_classes,
        )
        .unwrap();

        assert!(candidate_groups > 0);
        assert!(
            contextual_saved > 0 && composed.table.num_states < before,
            "duplicate independently compiled children should share at least one LR state",
        );

        let shared_local_states = composed.state_relations[1]
            .iter()
            .zip(&composed.state_relations[2])
            .filter(|(left, right)| left == right)
            .count();
        assert!(
            shared_local_states > 0,
            "at least one corresponding child-local LR state should map to the same quotient state",
        );

        // Exercise the real artifact-reuse path as well: component parser DWAs
        // are transported through the many-to-one LR-state relation rather
        // than rebuilt from the quotient table.
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                nt value ::= "a" "b" "c";
                nt child ::= value;
                nt document ::= "<" child ">,<" child ">";
            "#,
            &vocab,
        )
        .unwrap();
        let runtime_composed = parent
            .compose_linked_children_for_test(&[("LEFT", &child), ("RIGHT", &loaded_child)], &vocab)
            .unwrap();
        assert_constraints_equivalent_on_reachable_prefixes(
            &runtime_composed,
            &monolithic,
            &vocab,
            8,
        );
        let mut actual = runtime_composed.start();
        actual.commit_token(0).unwrap();
        assert!(actual.is_accepting());

    }

    #[test]
    fn contextual_sharing_rejects_ambiguity_with_no_stack_provenance() {
        let vocab = byte_vocab();
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t P ::= "p";
                t Q ::= "p";
                t BANG ::= "!";
                t QUESTION ::= "?";
                t LEFT ::= @token(998);
                t RIGHT ::= @token(999);
                nt left_prefix ::= P;
                nt right_prefix ::= Q;
                nt left_branch ::= left_prefix LEFT;
                nt right_branch ::= right_prefix RIGHT;
                nt document ::= left_branch BANG | right_branch QUESTION;
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                t CA ::= "a";
                t CB ::= "b";
                t CC ::= "c";
                nt child ::= CA CB CC;
            "#,
            &vocab,
        )
        .unwrap();
        let children = [
            CompiledSubgrammarInput {
                placeholder_terminal: terminal(&parent, "LEFT"),
                additional_placeholder_terminals: &[],
constraint: &child,
            },
            CompiledSubgrammarInput {
                placeholder_terminal: terminal(&parent, "RIGHT"),
                additional_placeholder_terminals: &[],
constraint: &child,
            },
        ];
        let table_inputs = children
            .iter()
            .map(|child| SubgrammarTableInput {
                placeholder_terminal: child.placeholder_terminal,
                additional_placeholder_terminals: &[],
table: &child.constraint.table,
                ignore_terminal: child.constraint.ignore_terminal,
                start_nullable: child.constraint.table.embedded_start_nullable(),
            })
            .collect::<Vec<_>>();
        let mut shared = compose_subgrammar_tables(&parent.table, None, &table_inputs).unwrap();
        let terminal_analysis = composition_terminal_classes(&parent, &children, &shared);
        let nonterminal_classes = structural_nonterminal_classes(
            &shared.table,
            &terminal_analysis.classes,
            &shared.boundary_nonterminals,
        );
        let (candidate_count, saved) =
            contextually_share_composed_states(
                &mut shared,
                &parent,
                &children,
                &terminal_analysis.classes,
                &nonterminal_classes,
            );
        assert!(candidate_count > 0, "the duplicate child states should be detected structurally");
        assert_eq!(
            saved, 0,
            "when two linked copies have the exact same lower stack context, a table-only quotient must preserve their distinct LR states",
        );
    }

    #[test]
    #[ignore = "historical unsupported compiled-artifact flattened composer"]
    fn runtime_lexer_product_coalesces_equivalent_ambiguous_child_lanes() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"c".to_vec()),
            (3, b"!".to_vec()),
            (4, b"?".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t BANG ::= "!";
                t QUESTION ::= "?";
                t LEFT ::= @token(998);
                t RIGHT ::= @token(999);
                nt document ::= LEFT BANG | RIGHT QUESTION;
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                t WORD ::= "abc";
                nt child ::= WORD;
            "#,
            &vocab,
        )
        .unwrap();
        let composed = parent
            .compose_linked_children_for_test_flattened(&[("LEFT", &child), ("RIGHT", &child)], &vocab)
            .unwrap();

        let mut state = composed.start();
        state.commit_token(0).unwrap();
        let explicitly_disabled = std::env::var("GLRMASK_COMPOSE_RUNTIME_LEXER_PRODUCT")
            .ok()
            .is_some_and(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "" | "0" | "false" | "no" | "off"
                )
            });
        if explicitly_disabled {
            assert!(composed.runtime_source_state_offset().is_none());
            assert_eq!(
                state.state.len(),
                2,
                "without the runtime product, the partial token 'a' should leave the two equivalent child lexer lanes split",
            );
            return;
        }

        let source_offset = composed
            .runtime_source_state_offset()
            .expect("duplicate ambiguous child lexer lanes should select the exact runtime product");
        assert!(source_offset > 0);
        assert_eq!(
            state.state.len(),
            1,
            "the persistent lexer frontier should have one product key after the partial token 'a'",
        );
        let product_state = *state.state.keys().next().unwrap();
        assert!(product_state < source_offset);
        assert!(
            composed
                .runtime_product_source_states(product_state)
                .is_some_and(|sources| sources.len() >= 2),
            "the single product key must represent at least two exact source lexer lanes",
        );

        for suffix in [[1, 2, 3], [1, 2, 4]] {
            let mut cursor = composed.start();
            cursor.commit_token(0).unwrap();
            for token in suffix {
                cursor.commit_token(token).unwrap();
            }
            assert!(cursor.is_accepting());
        }
    }

    #[test]
    #[ignore = "historical unsupported compiled-artifact flattened composer"]
    fn multi_tsid_runtime_lexer_product_remains_recomposable() {
        let vocab = byte_vocab();
        let left = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child ::= "abc" | "z";
            "#,
            &vocab,
        )
        .unwrap();
        let right = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child ::= "abd" | "z";
            "#,
            &vocab,
        )
        .unwrap();
        let middle_parent = Constraint::from_glrm_grammar(
            r#"
                start call;
                t LEFT ::= @token(998);
                t RIGHT ::= @token(999);
                nt call ::= LEFT | RIGHT;
            "#,
            &vocab,
        )
        .unwrap();
        let middle = middle_parent
            .compose_linked_children_for_test_flattened(&[("LEFT", &left), ("RIGHT", &right)], &vocab)
            .unwrap();

        let explicitly_disabled = std::env::var("GLRMASK_COMPOSE_RUNTIME_LEXER_PRODUCT")
            .ok()
            .is_some_and(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "" | "0" | "false" | "no" | "off"
                )
            });
        if !explicitly_disabled {
            let source_offset = middle
                .runtime_source_state_offset()
                .expect("the overlapping abc/abd child lanes should select a runtime product");
            assert!(
                (0..source_offset).any(|state| middle.internal_tsids_for_state(state).len() > 1),
                "the regression requires at least one product state with multiple TSID memberships",
            );
        }

        let outer_parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t CALL ::= @token(1000);
                nt document ::= "<" CALL ">";
            "#,
            &vocab,
        )
        .unwrap();
        let outer = outer_parent
            .compose_linked_children_for_test_flattened(&[("CALL", &middle)], &vocab)
            .unwrap();
        let loaded_middle = Constraint::load(&middle.save()).unwrap();
        let outer_loaded = Constraint::from_glrm_grammar(
            r#"
                start document;
                t CALL ::= @token(1000);
                nt document ::= "<" CALL ">";
            "#,
            &vocab,
        )
        .unwrap()
        .compose_linked_children_for_test_flattened(&[("CALL", &loaded_middle)], &vocab)
        .unwrap();

        for bytes in [b"<abc>".as_slice(), b"<abd>", b"<z>"] {
            for constraint in [&outer, &outer_loaded] {
                let mut state = constraint.start();
                for &byte in bytes {
                    state.commit_token(byte as u32).unwrap();
                }
                assert!(state.is_accepting(), "recomposed runtime-product child rejected {bytes:?}");
            }
        }
    }

    #[test]
    fn stored_parser_domain_label_expands_before_nested_transport() {
        let relation = vec![vec![10], vec![11, 12], vec![20], vec![30]];
        let domain = 4i32;
        let domain_labels = vec![NO_PARSER_DOMAIN_LABEL, domain, domain, NO_PARSER_DOMAIN_LABEL];

        assert_eq!(
            mapped_labels(domain, &relation, &domain_labels).unwrap(),
            vec![
                encode_positive_label(11),
                encode_positive_label(12),
                encode_positive_label(20),
            ],
            "a stored synthetic parser-domain label must expand to every concrete local state in that domain before the outer state relation is applied",
        );
    }

    #[test]
    fn symbolic_child_default_domain_expands_exactly_to_materialized_transport() {
        let relation = vec![vec![10], vec![11], vec![12, 13], vec![20]];
        let mut source = DWAState::default();
        source.transitions.insert(
            encode_positive_label(0),
            (1, Weight::all()),
        );
        // This explicit one-to-many local label must override the synthetic
        // DEFAULT on both of its mapped concrete parser states.
        source.transitions.insert(
            encode_positive_label(2),
            (1, Weight::all()),
        );
        source
            .transitions
            .insert(DEFAULT_LABEL, (2, Weight::all()));

        let mut baseline = NWA::new(1, 0);
        for _ in 0..3 {
            baseline.add_state();
        }
        add_component_parser_state_transitions(
            &mut baseline,
            0,
            &source,
            &relation,
            &[],
            relation.len() as u32,
            None,
        )
        .unwrap();

        let mut domain_states = BitSet::new(32);
        for state in [11usize, 12, 13] {
            domain_states.set(state);
        }
        let domain = ParserDefaultDomain {
            label: 30,
            base_has_states: true,
            nested_labels: BTreeMap::new(),
            states: domain_states,
            predicted_saved_edges: 3,
        };
        let mut compressed = NWA::new(1, 0);
        for _ in 0..3 {
            compressed.add_state();
        }
        add_component_parser_state_transitions(
            &mut compressed,
            0,
            &source,
            &relation,
            &[],
            relation.len() as u32,
            Some(&domain),
        )
        .unwrap();

        // Expand the runtime lookup semantics `concrete -> domain` back onto
        // concrete labels. The resulting transition relation must equal the old
        // fully-materialized transport exactly.
        let domain_targets = compressed.states()[0]
            .transitions
            .get(&domain.label)
            .cloned()
            .expect("compressed row must carry one domain fallback");
        let mut expanded = compressed.states()[0].transitions.clone();
        expanded.remove(&domain.label);
        for parser_state in domain.states.iter_ones() {
            expanded
                .entry(encode_positive_label(parser_state as u32))
                .or_insert_with(|| domain_targets.clone());
        }
        assert_eq!(expanded, baseline.states()[0].transitions);
        assert_eq!(
            expanded[&encode_positive_label(12)][0].0,
            1,
            "explicit local transitions must retain precedence over domain DEFAULT",
        );
        assert_eq!(expanded[&encode_positive_label(11)][0].0, 2);
        assert_eq!(expanded[&encode_positive_label(20)][0].0, 2);
    }

    #[test]
    fn fast_reverse_hashcons_matches_exact_structural_quotient_on_weighted_dags() {
        fn weight(start: u32, end: u32) -> Weight {
            Weight::from_token_set_for_tsid(
                0,
                RangeSetBlaze::from_iter([start..=end]),
            )
        }

        fn generated(seed: u32) -> NWA {
            // Construct duplicate sub-DAGs deliberately.  Every edge points to
            // a larger old state id, so the graph is acyclic; paired states use
            // distinct old targets whose suffixes are nevertheless equivalent.
            let mut nwa = NWA::new(1, 63);
            for _ in 0..14 {
                nwa.add_state();
            }
            let full = Weight::all();
            let narrow = weight(seed % 16, (seed % 16) + 7);
            nwa.set_final_weight(12, narrow.clone());
            nwa.set_final_weight(13, narrow.clone());
            for (left, right, left_target, right_target, label) in [
                (10, 11, 12, 13, 1),
                (8, 9, 10, 11, 2),
                (6, 7, 8, 9, DEFAULT_LABEL),
                (4, 5, 6, 7, 3),
                (2, 3, 4, 5, 4),
                (0, 1, 2, 3, 5),
            ] {
                nwa.add_transition(left, label, left_target, narrow.clone());
                nwa.add_transition(right, label, right_target, narrow.clone());
                if label != DEFAULT_LABEL {
                    nwa.add_epsilon(left, left_target, full.clone());
                    nwa.add_epsilon(right, right_target, full.clone());
                }
                if (left + seed) % 3 == 0 {
                    nwa.set_final_weight(left, full.clone());
                    nwa.set_final_weight(right, full.clone());
                }
            }
            nwa.set_start_states(vec![0, 1]);
            nwa
        }

        for seed in 0..64 {
            let input = generated(seed);
            let reference = reverse_hashcons_positive_acyclic_nwa(input.clone());
            let reverse_topo = full_nwa_topological_order(&input)
                .expect("generated quotient graph must be acyclic")
                .into_iter()
                .rev()
                .collect::<Vec<_>>();
            let fast = reverse_hashcons_positive_acyclic_nwa_fast(input.clone());
            let reused = reverse_hashcons_positive_acyclic_nwa_fast_with_reverse_topo(
                input,
                Some(reverse_topo),
            );
            assert_eq!(
                fast.start_states(),
                reference.start_states(),
                "fast quotient start states differ for seed {seed}",
            );
            assert_eq!(
                fast.states(),
                reference.states(),
                "fast quotient structure differs from exact structural quotient for seed {seed}",
            );
            assert_eq!(
                reused.start_states(),
                reference.start_states(),
                "reused-topology quotient start states differ for seed {seed}",
            );
            assert_eq!(
                reused.states(),
                reference.states(),
                "reused-topology quotient differs from exact structural quotient for seed {seed}",
            );
        }
    }

    #[test]
    fn overlap_local_union_matches_generic_reference_on_generated_acyclic_inputs() {
        fn next_u32(state: &mut u64) -> u32 {
            *state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (*state >> 32) as u32
        }

        fn generated_dwa(random: &mut u64, salt: u32) -> DWA {
            let mut dwa = DWA::new(1, 63);
            for _ in 0..3 {
                dwa.add_state();
            }
            for state in 0..4u32 {
                if next_u32(random) % 3 != 0 {
                    let start = (next_u32(random) + salt) % 48;
                    let end = (start + 3 + next_u32(random) % 12).min(63);
                    dwa.set_final_weight(
                        state,
                        Weight::from_token_set_for_tsid(
                            0,
                            RangeSetBlaze::from_iter([start..=end]),
                        ),
                    );
                }
                if state == 3 {
                    continue;
                }
                let mut labels = BTreeSet::<i32>::new();
                for concrete in 0..6u32 {
                    if next_u32(random) % 3 != 0 {
                        labels.insert(encode_positive_label(concrete));
                    }
                }
                if next_u32(random) % 2 == 0 {
                    labels.insert(encode_negative_label(next_u32(random) % 6));
                }
                if next_u32(random) % 2 == 0 {
                    labels.insert(20 + (next_u32(random) % 3) as i32);
                }
                if next_u32(random) % 3 == 0 {
                    labels.insert(DEFAULT_LABEL);
                }
                for label in labels {
                    let target = state + 1 + next_u32(random) % (3 - state);
                    let start = (next_u32(random) + salt * 3) % 48;
                    let end = (start + 2 + next_u32(random) % 14).min(63);
                    dwa.add_transition(
                        state,
                        label,
                        target,
                        Weight::from_token_set_for_tsid(
                            0,
                            RangeSetBlaze::from_iter([start..=end]),
                        ),
                    );
                }
            }
            dwa
        }

        let mut random = 0xd15c_a11c_5eed_2026u64;
        for arity in 2..=3usize {
            for case in 0..64u32 {
                let dwas = (0..arity)
                    .map(|index| generated_dwa(&mut random, case * 4 + index as u32 + 1))
                    .collect::<Vec<_>>();
                let direct_inputs = dwas
                    .iter()
                    .map(parser_nwa_preserve_defaults)
                    .collect::<Vec<_>>();
                let (direct, _) =
                    determinize_epsilon_free_component_union(direct_inputs, Some(6))
                        .expect("generated inputs are epsilon-free and acyclic");

                let extra_positive_labels = dwas
                    .iter()
                    .flat_map(|dwa| dwa.states())
                    .flat_map(|state| state.transitions.keys().copied())
                    .filter(|&label| label >= 6 && label != DEFAULT_LABEL)
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect::<Vec<_>>();
                let mut union = NWA::new(1, 63);
                let mut starts = Vec::new();
                for dwa in &dwas {
                    let explicit = explicit_parser_nwa(dwa, 6, &extra_positive_labels);
                    let body = union.append_with_body(&explicit);
                    starts.extend(body.start_states);
                }
                union.set_start_states(starts);
                let generic = determinize(&union).expect("generic reference must determinize");
                let direct_explicit_nwa = explicit_parser_nwa(&direct, 6, &extra_positive_labels);
                let direct_explicit = determinize(&direct_explicit_nwa)
                    .expect("expanded direct result must determinize");
                let difference = find_difference(&direct_explicit, &generic)
                    .expect("generated direct/reference equivalence must be decidable");
                assert_eq!(
                    difference, None,
                    "overlap-local union differs from generic reference for arity {arity} case {case}",
                );

                if arity == 3 {
                    let (first_pair, _) = determinize_epsilon_free_component_union(
                        vec![
                            parser_nwa_preserve_defaults(&dwas[0]),
                            parser_nwa_preserve_defaults(&dwas[1]),
                        ],
                        Some(6),
                    )
                    .expect("first generated pair is epsilon-free and acyclic");
                    let (nested_direct, _) = determinize_epsilon_free_component_union(
                        vec![
                            parser_nwa_preserve_defaults(&first_pair),
                            parser_nwa_preserve_defaults(&dwas[2]),
                        ],
                        Some(6),
                    )
                    .expect("nested generated direct union stays epsilon-free and acyclic");
                    let nested_explicit_nwa =
                        explicit_parser_nwa(&nested_direct, 6, &extra_positive_labels);
                    let nested_explicit = determinize(&nested_explicit_nwa)
                        .expect("expanded nested direct result must determinize");
                    let nested_difference = find_difference(&nested_explicit, &generic)
                        .expect("nested direct/reference equivalence must be decidable");
                    assert_eq!(
                        nested_difference, None,
                        "nested overlap-local union differs from generic reference in case {case}",
                    );
                }
            }
        }
    }

    #[test]
    fn overlap_local_union_preserves_symbolic_default_semantics() {
        let mut wildcard = NWA::new(1, 0);
        let wildcard_start = wildcard.add_state();
        let wildcard_final = wildcard.add_state();
        wildcard.set_start_states(vec![wildcard_start]);
        wildcard.add_transition(
            wildcard_start,
            DEFAULT_LABEL,
            wildcard_final,
            Weight::all(),
        );
        wildcard.set_final_weight(wildcard_final, Weight::all());

        let mut explicit = NWA::new(1, 0);
        let explicit_start = explicit.add_state();
        let explicit_final = explicit.add_state();
        explicit.set_start_states(vec![explicit_start]);
        explicit.add_transition(
            explicit_start,
            encode_positive_label(3),
            explicit_final,
            Weight::all(),
        );
        explicit.add_transition(
            explicit_start,
            encode_negative_label(5),
            explicit_final,
            Weight::all(),
        );
        // Synthetic child-domain labels are ordinary nonnegative symbols above
        // the concrete parser-state range. A global DEFAULT from another
        // automaton must still contribute on them.
        explicit.add_transition(
            explicit_start,
            100,
            explicit_final,
            Weight::all(),
        );
        explicit.set_final_weight(explicit_final, Weight::all());

        let (union, _) = determinize_epsilon_free_component_union(
            vec![wildcard, explicit],
            Some(8),
        )
        .expect("epsilon-free acyclic union must use overlap-local path");
        let evaluate_runtime_label = |label: i32| {
            let start = &union.states()[union.start_state() as usize];
            let transition = start
                .transitions
                .get(&label)
                .or_else(|| (label >= 0).then(|| start.transitions.get(&DEFAULT_LABEL)).flatten());
            let Some((target, edge_weight)) = transition else {
                return Weight::empty();
            };
            let Some(final_weight) = union.states()[*target as usize].final_weight.as_ref() else {
                return Weight::empty();
            };
            edge_weight.intersection(final_weight)
        };

        assert!(!evaluate_runtime_label(encode_positive_label(3)).is_empty());
        assert!(!evaluate_runtime_label(encode_positive_label(4)).is_empty());
        assert!(!evaluate_runtime_label(encode_negative_label(5)).is_empty());
        assert!(!evaluate_runtime_label(100).is_empty());
        assert!(evaluate_runtime_label(encode_negative_label(6)).is_empty());
        assert!(union.states()[union.start_state() as usize]
            .transitions
            .contains_key(&DEFAULT_LABEL));
    }

    #[test]
    fn overlap_local_union_declines_epsilon_inputs_for_generic_fallback() {
        let mut automaton = NWA::new(1, 0);
        let start = automaton.add_state();
        let target = automaton.add_state();
        automaton.set_start_states(vec![start]);
        automaton.add_epsilon(start, target, Weight::all());
        automaton.set_final_weight(target, Weight::all());

        assert!(!supports_overlap_local_union(std::slice::from_ref(&automaton)));
        assert!(determinize_epsilon_free_component_union(vec![automaton.clone()], None).is_none());

        let generic = determinize(&automaton).expect("generic determinization handles epsilon input");
        assert!(generic.num_states() > 0);
        assert!(generic.states().iter().any(|state| state.final_weight.is_some()));
    }

    #[test]
    fn compiled_parser_dwas_reconcile_and_union_with_one_to_many_start_mapping() {
        let vocab = byte_vocab();
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "<" SUB ">" SUB "!";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child ::= "a" "b";
            "#,
            &vocab,
        )
        .unwrap();
        let composed_table = compose_subgrammar_tables(
            &parent.table,
            None,
            &[SubgrammarTableInput {
                placeholder_terminal: terminal(&parent, "SUB"),
                additional_placeholder_terminals: &[],
table: &child.table,
                ignore_terminal: child.ignore_terminal,
                start_nullable: child.table.embedded_start_nullable(),
            }],
        )
        .unwrap();
        let (merged_tokenizer, tokenizer_offsets) =
            crate::automata::lexer::tokenizer::Tokenizer::disjoint_union_with_terminal_offsets(&[
                (&parent.tokenizer, composed_table.terminal_offsets[0]),
                (&child.tokenizer, composed_table.terminal_offsets[1]),
            ]);
        let parser_components = [
            ParserDwaComponent {
                constraint: &parent,
                parser_state_relation: &composed_table.state_relations[0],
                tokenizer_state_offset: tokenizer_offsets[0],
                terminal_offset: 0,
                composed_table: None,
            },
            ParserDwaComponent {
                constraint: &child,
                parser_state_relation: &composed_table.state_relations[1],
                tokenizer_state_offset: tokenizer_offsets[1],
                terminal_offset: 0,
                composed_table: None,
            },
        ];
        let default_domains = build_parser_default_domain_plan(
            &parser_components,
            composed_table.table.num_states,
        );
        let merged = compose_component_parser_dwas_and_possible_matches(
            &parser_components,
            &composed_table.terminal_offsets,
            &default_domains.component_domains,
            merged_tokenizer.num_states() as usize,
            &vocab.entries_map().keys().copied().collect::<Vec<_>>(),
            false,
        )
        .unwrap();
        assert!(merged.0.artifact().0.num_states() > 0);
        assert!(merged.0.artifact().0.num_transitions() > 0);
        assert_eq!(
            merged.0.id_map().tokenizer_states.original_to_internal.len(),
            merged_tokenizer.num_states() as usize,
        );
        assert_eq!(composed_table.state_relations[1][0].len(), 2);
    }

    fn assert_accepts(constraint: &Constraint, bytes: &[u8]) {
        let mut state = constraint.start();
        state.commit_bytes(bytes).unwrap();
        assert!(state.is_accepting(), "expected {:?} to finish", bytes);
    }

    fn assert_rejects(constraint: &Constraint, bytes: &[u8]) {
        let mut state = constraint.start();
        let accepted = state.commit_bytes(bytes).is_ok() && state.is_accepting();
        assert!(!accepted, "expected {:?} to reject", bytes);
    }

    fn token_allowed(mask: &[u32], token: u32) -> bool {
        mask.get(token as usize / 32)
            .is_some_and(|word| word & (1u32 << (token % 32)) != 0)
    }

    fn assert_constraints_equivalent_on_reachable_prefixes(
        actual: &Constraint,
        expected: &Constraint,
        vocab: &Vocab,
        max_depth: usize,
    ) {
        assert_constraints_equivalent_on_reachable_prefixes_inner(
            actual,
            expected,
            vocab,
            max_depth,
            true,
            "",
        );
    }

    /// Labeled variant: prefixes every failure message with `label` so a
    /// multi-backend comparison identifies which side (oracle/static/dynamic)
    /// diverged.
    fn assert_constraints_equivalent_on_reachable_prefixes_labeled(
        actual: &Constraint,
        expected: &Constraint,
        vocab: &Vocab,
        max_depth: usize,
        label: &str,
    ) {
        assert_constraints_equivalent_on_reachable_prefixes_inner(
            actual,
            expected,
            vocab,
            max_depth,
            true,
            label,
        );
    }

    fn assert_constraints_mask_equivalent_on_reachable_prefixes(
        actual: &Constraint,
        expected: &Constraint,
        vocab: &Vocab,
        max_depth: usize,
    ) {
        assert_constraints_equivalent_on_reachable_prefixes_inner(
            actual,
            expected,
            vocab,
            max_depth,
            false,
            "",
        );
    }

    /// Labeled variant of the mask-only comparison.
    fn assert_constraints_mask_equivalent_on_reachable_prefixes_labeled(
        actual: &Constraint,
        expected: &Constraint,
        vocab: &Vocab,
        max_depth: usize,
        label: &str,
    ) {
        assert_constraints_equivalent_on_reachable_prefixes_inner(
            actual,
            expected,
            vocab,
            max_depth,
            false,
            label,
        );
    }

    fn assert_constraints_equivalent_on_reachable_prefixes_inner(
        actual: &Constraint,
        expected: &Constraint,
        vocab: &Vocab,
        max_depth: usize,
        compare_completion: bool,
        label: &str,
    ) {
        let tag = if label.is_empty() {
            String::new()
        } else {
            format!("{label}: ")
        };
        let token_ids = vocab.entries_map().keys().copied().collect::<Vec<_>>();
        let mut frontier = vec![Vec::<u32>::new()];
        for depth in 0..=max_depth {
            let mut next = Vec::new();
            for prefix in frontier {
                let mut actual_state = actual.start();
                let mut expected_state = expected.start();
                for &token in &prefix {
                    actual_state.commit_token(token).unwrap_or_else(|error| {
                        panic!("{tag}actual rejected reachable prefix {prefix:?}: {error}")
                    });
                    expected_state.commit_token(token).unwrap_or_else(|error| {
                        panic!("{tag}expected rejected its own reachable prefix {prefix:?}: {error}")
                    });
                }

                let actual_mask = actual_state.mask();
                let expected_mask = expected_state.mask();
                if std::env::var_os("GLRMASK_DEBUG_PREFIX10_STACKS").is_some()
                    && prefix == [10]
                {
                    eprintln!("PREFIX10_ACTUAL {:?}", actual_state.debug_parser_stacks());
                    eprintln!("PREFIX10_EXPECTED {:?}", expected_state.debug_parser_stacks());
                }
                assert_eq!(
                    actual_mask, expected_mask,
                    "{tag}mask mismatch after reachable prefix {prefix:?}",
                );
                if compare_completion {
                    assert_eq!(
                        actual_state.is_accepting(),
                        expected_state.is_accepting(),
                        "{tag}completion mismatch after reachable prefix {prefix:?}",
                    );
                }

                if depth == max_depth {
                    continue;
                }
                for &token in &token_ids {
                    if token_allowed(&expected_mask, token) {
                        let mut extended = prefix.clone();
                        extended.push(token);
                        next.push(extended);
                    }
                }
            }
            frontier = next;
        }
    }

    #[test]
    fn owned_parent_composition_matches_borrowed_and_monolithic() {
        let vocab = Vocab::new(vec![
            (0, b"<ab>".to_vec()),
            (1, b"<a".to_vec()),
            (2, b"b>".to_vec()),
            (3, b"<".to_vec()),
            (4, b"a".to_vec()),
            (5, b"b".to_vec()),
            (6, b">".to_vec()),
        ]);
        let compile_parent = || {
            Constraint::from_glrm_grammar(
                r#"
                    start document;
                    t SUB ::= @token(999);
                    nt document ::= "<" SUB ">";
                "#,
                &vocab,
            )
            .unwrap()
        };
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child ::= "a" "b";
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                nt child ::= "a" "b";
                nt document ::= "<" child ">";
            "#,
            &vocab,
        )
        .unwrap();
        let borrowed_parent = compile_parent();
        let borrowed = borrowed_parent
            .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
            .unwrap();
        let owned = compile_parent()
            .compose_linked_children_for_test_owned(&[("SUB", &child)], &vocab)
            .unwrap();

        assert_constraints_equivalent_on_reachable_prefixes(
            &owned,
            &monolithic,
            &vocab,
            4,
        );
        assert_constraints_equivalent_on_reachable_prefixes(
            &owned,
            &borrowed,
            &vocab,
            4,
        );
        assert_eq!(owned.tokenizer.start_state(), 0);
        assert!(owned.num_tokenizer_states() <= borrowed.num_tokenizer_states());
        for state in 0..owned.tokenizer.num_states() {
            for byte in u8::MIN..=u8::MAX {
                assert_eq!(
                    owned.tokenizer_fast_transitions.transition(
                        &owned.tokenizer,
                        state,
                        byte,
                    ),
                    owned.tokenizer.get_transition(state, byte),
                    "transported fast tokenizer transition differs at state {state}, byte {byte}",
                );
            }
            let closures = owned.tokenizer.all_singleton_epsilon_closures();
            assert_eq!(
                closures.get(state as usize).expect("closure state is in range"),
                owned.tokenizer.singleton_epsilon_closure(state).as_ref(),
                "transported singleton epsilon closure differs at state {state}",
            );
        }
    }

    #[test]
    #[allow(deprecated)]
    fn legacy_owned_parent_composition_is_unsupported() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"c".to_vec()),
            (3, b"d".to_vec()),
            (4, b"ab".to_vec()),
            (5, b"cd".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "a" SUB "d";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child ::= "b" "c";
            "#,
            &vocab,
        )
        .unwrap();
        let placeholder = terminal(&parent, "SUB");
        let inputs = [CompiledSubgrammarInput {
            placeholder_terminal: placeholder,
            additional_placeholder_terminals: &[],
            constraint: &child,
        }];

        let ordinary = compose_constraints_owned_parent(parent.clone(), &inputs, &vocab)
            .err()
            .expect("legacy owned-parent composition must be rejected");
        assert!(
            ordinary.contains("legacy owned-parent composition is unsupported"),
            "unexpected message: {ordinary}",
        );

        let shared_children = [Arc::new(child.clone())];
        let shared = compose_constraints_owned_parent_shared(
            parent.clone(),
            &inputs,
            &shared_children,
            &vocab,
        )
        .err()
        .expect("legacy shared owned-parent composition must be rejected");
        assert!(
            shared.contains("legacy owned-parent composition is unsupported"),
            "unexpected message: {shared}",
        );

        // The separate borrowed flattened path is unsupported as well.
        let borrowed = compose_constraints(&parent, &inputs, &vocab)
            .err()
            .expect("legacy borrowed flattened composition must be rejected");
        assert!(
            borrowed.contains("legacy compiled-artifact flattened composition is unsupported"),
            "unexpected message: {borrowed}",
        );

        // Explicit segmented StaticParserDwa and Dynamic stay supported and agree.
        let static_owned = compose_constraints_owned_parent_segmented(
            parent.clone(),
            &inputs,
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .map(|composition| composition.constraint)
        .expect("explicit StaticParserDwa segmented composition must remain supported");
        let dynamic_owned = compose_constraints_owned_parent_segmented(
            parent.clone(),
            &inputs,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .map(|composition| composition.constraint)
        .expect("explicit Dynamic segmented composition must remain supported");

        let mut static_state = static_owned.start();
        let mut dynamic_state = dynamic_owned.start();
        assert_eq!(static_state.mask(), dynamic_state.mask(), "start masks must agree");
        assert!(static_state.commit_bytes(b"ab").is_ok(), "static must accept the cross-boundary token");
        assert!(dynamic_state.commit_bytes(b"ab").is_ok(), "dynamic must accept the cross-boundary token");
        assert_eq!(
            static_state.mask(),
            dynamic_state.mask(),
            "masks after the cross-boundary token must agree",
        );
        assert!(static_state.commit_bytes(b"cd").is_ok());
        assert!(dynamic_state.commit_bytes(b"cd").is_ok());
        assert!(static_state.is_accepting() && dynamic_state.is_accepting());
    }

    #[test]
    fn prepared_loaded_components_match_fresh_owned_composition() {
        let vocab = Vocab::new(vec![
            (0, b"<ab>".to_vec()),
            (1, b"<a".to_vec()),
            (2, b"b>".to_vec()),
            (3, b"<".to_vec()),
            (4, b"a".to_vec()),
            (5, b"b".to_vec()),
            (6, b">".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "<" SUB ">";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child ::= "a" "b";
            "#,
            &vocab,
        )
        .unwrap();
        let fresh = parent
            .clone()
            .compose_linked_children_for_test_owned(&[("SUB", &child)], &vocab)
            .unwrap();

        let mut loaded_parent = Constraint::load(&parent.save()).unwrap();
        let mut loaded_child = Constraint::load(&child.save()).unwrap();
        loaded_parent.prepare_for_composition_internal(&vocab).unwrap();
        loaded_child.prepare_for_composition_internal(&vocab).unwrap();
        assert!(loaded_parent.packed_parser_dwa.is_some());
        assert!(loaded_child.packed_parser_dwa.is_some());
        assert!(loaded_parent.deferred_composition_metadata_blob.is_none());
        assert!(loaded_child.deferred_composition_metadata_blob.is_none());

        let prepared = loaded_parent
            .compose_linked_children_for_test_owned(&[("SUB", &loaded_child)], &vocab)
            .unwrap();
        assert_constraints_equivalent_on_reachable_prefixes(
            &prepared,
            &fresh,
            &vocab,
            4,
        );
    }

    #[test]
    fn composition_rejects_placeholder_that_matches_a_real_vocab_token() {
        let vocab = Vocab::new(vec![
            (0, b"X".to_vec()),
            (1, b"a".to_vec()),
            (2, b"!".to_vec()),
            // The placeholder token deliberately exists in the model vocab.
            // Composition must still remove its exact-token path.
            (3, b"<placeholder>".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(3);
                nt document ::= "X" SUB "!";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child ::= "a";
            "#,
            &vocab,
        )
        .unwrap();
        let error = parent
            .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
            .expect_err("real vocabulary tokens cannot be used as composition sentinels");
        assert!(error.to_string().contains("non-vocabulary sentinels"));
        assert!(error.to_string().contains("outside the supplied vocabulary"));
    }

    #[test]
    fn repeated_call_sites_match_monolithic_on_all_small_reachable_prefixes() {
        let vocab = Vocab::new(vec![
            (0, b"<a>b!".to_vec()),
            (1, b"<b>a!".to_vec()),
            (2, b"<".to_vec()),
            (3, b"a".to_vec()),
            (4, b"b".to_vec()),
            (5, b">".to_vec()),
            (6, b"!".to_vec()),
            (7, b"a>b".to_vec()),
            (8, b"b>a".to_vec()),
            (9, b">b!".to_vec()),
            (10, b">a!".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "<" SUB ">" SUB "!";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child ::= "a" | "b";
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                nt child ::= "a" | "b";
                nt document ::= "<" child ">" child "!";
            "#,
            &vocab,
        )
        .unwrap();
        let composed = parent
            .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
            .unwrap();

        assert_constraints_equivalent_on_reachable_prefixes(
            &composed,
            &monolithic,
            &vocab,
            5,
        );
    }

    #[test]
    fn recursive_child_matches_monolithic_on_all_small_reachable_prefixes() {
        let vocab = Vocab::new(vec![
            (0, b"Xaa!".to_vec()),
            (1, b"Xa!".to_vec()),
            (2, b"Xa".to_vec()),
            (3, b"aa".to_vec()),
            (4, b"a!".to_vec()),
            (5, b"X".to_vec()),
            (6, b"a".to_vec()),
            (7, b"!".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "X" SUB "!";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child ::= "a" child?;
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                nt child ::= "a" child?;
                nt document ::= "X" child "!";
            "#,
            &vocab,
        )
        .unwrap();
        let composed = parent
            .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
            .unwrap();

        assert_constraints_equivalent_on_reachable_prefixes(
            &composed,
            &monolithic,
            &vocab,
            5,
        );
    }

    #[test]
    fn parent_ignore_terminal_matches_monolithic_across_child_boundaries() {
        let vocab = Vocab::new(vec![
            (0, b"X a!".to_vec()),
            (1, b"Xa!".to_vec()),
            (2, b"X".to_vec()),
            (3, b" ".to_vec()),
            (4, b"a".to_vec()),
            (5, b"!".to_vec()),
            (6, b" a".to_vec()),
            (7, b"a!".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                ignore WS;
                t WS ::= " "+;
                t SUB ::= @token(999);
                nt document ::= "X" SUB "!";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child ::= "a";
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                ignore WS;
                t WS ::= " "+;
                nt document ::= "X" "a" "!";
            "#,
            &vocab,
        )
        .unwrap();
        let composed_static = parent
            .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
            .unwrap();
        let composed_dynamic = parent
            .compose_linked_children_for_test_dynamic(&[("SUB", &child)], &vocab)
            .unwrap();
        let source_oracle = compile_inline_source_oracle(
            r#"
                start document;
                ignore WS;
                t WS ::= " "+;
                extern grammar SUB;
                nt document ::= "X" SUB "!";
            "#,
            &[("SUB", "start child; nt child ::= \"a\";")],
            &vocab,
        );

        assert_constraints_equivalent_on_reachable_prefixes_labeled(
            &source_oracle,
            &monolithic,
            &vocab,
            5,
            "oracle-vs-monolithic",
        );
        assert_constraints_equivalent_on_reachable_prefixes_labeled(
            &composed_static,
            &source_oracle,
            &vocab,
            5,
            "static-vs-oracle",
        );
        assert_constraints_equivalent_on_reachable_prefixes_labeled(
            &composed_dynamic,
            &source_oracle,
            &vocab,
            5,
            "dynamic-vs-oracle",
        );
    }

    #[test]
    fn child_exact_special_token_survives_composition() {
        let vocab = Vocab::new(vec![
            (0, b"X".to_vec()),
            (1, b"a".to_vec()),
            (2, b"!".to_vec()),
            (3, b"<END>".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "X" SUB "!";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                t END ::= @token(3);
                nt child ::= "a" END;
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                t END ::= @token(3);
                nt document ::= "X" "a" END "!";
            "#,
            &vocab,
        )
        .unwrap();
        let composed = parent
            .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
            .unwrap();
        let loaded = Constraint::load(&composed.save())
            .expect("composed exact-special-token constraint must survive serialization");

        assert!(composed
            .special_token_terminals
            .iter()
            .any(|special| special.token_id == 3));
        assert!(loaded
            .special_token_terminals
            .iter()
            .any(|special| special.token_id == 3));
        assert_constraints_equivalent_on_reachable_prefixes(
            &composed,
            &monolithic,
            &vocab,
            4,
        );
        assert_constraints_equivalent_on_reachable_prefixes(
            &loaded,
            &monolithic,
            &vocab,
            4,
        );
    }

    #[test]
    fn scoped_visible_terminals_survive_model_token_boundaries() {
        // Entry orientation:
        //   token 0 = "Xa" leaves parent IGNORE="ab" in progress;
        //   token 1 = "bt" completes that IGNORE and enters the child.
        // Exit orientation:
        //   token 3 = "X" enters the child at reset;
        //   token 4 = "ta" completes child T and starts parent IGNORE;
        //   token 5 = "b!" completes IGNORE in the following model token.
        let vocab = Vocab::new(vec![
            (0, b"Xa".to_vec()),
            (1, b"bt".to_vec()),
            (2, b"!".to_vec()),
            (3, b"X".to_vec()),
            (4, b"ta".to_vec()),
            (5, b"b!".to_vec()),
            (6, b"a".to_vec()),
            (7, b"b".to_vec()),
            (8, b"t".to_vec()),
            (9, b"ab".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                ignore PARENT_WS;
                t PARENT_WS ::= "ab";
                t X ::= "X";
                t BANG ::= "!";
                t SUB ::= @token(999);
                nt document ::= X SUB BANG;
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                t CHILD_T ::= "t";
                nt child ::= CHILD_T;
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                ignore PARENT_WS;
                t PARENT_WS ::= "ab";
                t X ::= "X";
                t BANG ::= "!";
                g child ::= {
                    start child;
                    t CHILD_T ::= "t";
                    nt child ::= CHILD_T;
                };
                nt document ::= X child BANG;
            "#,
            &vocab,
        )
        .unwrap();

        // Scoped trivia is represented as a real terminal in the composed LR
        // table. Its parser semantics is state-dependent identity (`Skip`), not
        // a lexer-side boundary special case.
        let composed_table = compose_subgrammar_tables(
            &parent.table,
            Some(terminal(&parent, "PARENT_WS")),
            &[SubgrammarTableInput {
                placeholder_terminal: terminal(&parent, "SUB"),
                additional_placeholder_terminals: &[],
table: &child.table,
                ignore_terminal: child.ignore_terminal,
                start_nullable: child.table.embedded_start_nullable(),
            }],
        )
        .unwrap();
        let parent_ignore = terminal(&parent, "PARENT_WS");
        assert!(composed_table.table.skip_terminals.contains(&parent_ignore));
        assert!(composed_table.table.action.iter().any(|row| {
            matches!(row.get(&parent_ignore), Some(Action::Skip))
        }));

        let composed_static = parent
            .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
            .unwrap();
        let composed_dynamic = parent
            .compose_linked_children_for_test_dynamic(&[("SUB", &child)], &vocab)
            .unwrap();
        let source_oracle = compile_inline_source_oracle(
            r#"
                start document;
                ignore PARENT_WS;
                t PARENT_WS ::= "ab";
                t X ::= "X";
                t BANG ::= "!";
                extern grammar SUB;
                nt document ::= X SUB BANG;
            "#,
            &[(
                "SUB",
                "start child; t CHILD_T ::= \"t\"; nt child ::= CHILD_T;",
            )],
            &vocab,
        );
        assert_constraints_equivalent_on_reachable_prefixes_labeled(
            &source_oracle,
            &monolithic,
            &vocab,
            4,
            "oracle-vs-monolithic",
        );
        let mut prepared_parent = parent.clone();
        let mut prepared_child = child.clone();
        prepared_parent.ensure_composition_reset_tokens_by_terminal();
        prepared_child.ensure_composition_reset_tokens_by_terminal();
        let mut prepared_parent = Constraint::load(&prepared_parent.save()).unwrap();
        let mut prepared_child = Constraint::load(&prepared_child.save()).unwrap();
        prepared_parent
            .materialize_composition_metadata_for_compilation()
            .unwrap();
        prepared_child
            .materialize_composition_metadata_for_compilation()
            .unwrap();
        assert_eq!(
            prepared_parent.composition_reset_tokens_by_terminal.len(),
            prepared_parent.tokenizer.num_terminals() as usize,
        );
        assert_eq!(
            prepared_child.composition_reset_tokens_by_terminal.len(),
            prepared_child.tokenizer.num_terminals() as usize,
        );
        let cached_composed = prepared_parent
            .compose_linked_children_for_test(&[("SUB", &prepared_child)], &vocab)
            .unwrap();
        for sequence in [[0u32, 1, 2].as_slice(), [3u32, 4, 5].as_slice()] {
            let mut actual = composed_static.start();
            let mut dynamic = composed_dynamic.start();
            let mut cached = cached_composed.start();
            let mut expected = monolithic.start();
            for &token in sequence {
                let actual_mask = actual.mask();
                let dynamic_mask = dynamic.mask();
                let cached_mask = cached.mask();
                let expected_mask = expected.mask();
                assert_eq!(
                    cached_mask, expected_mask,
                    "prepared-cache mask mismatch before token {token} in sequence {sequence:?}",
                );
                assert_eq!(
                    dynamic_mask, expected_mask,
                    "dynamic mask mismatch before token {token} in sequence {sequence:?}",
                );
                if actual_mask != expected_mask {
                    eprintln!(
                        "IGNORE_FUSION_STATE sequence={sequence:?} before_token={token} actual={:?} expected={:?}",
                        actual.debug_parser_stacks(),
                        expected.debug_parser_stacks(),
                    );
                    let differing = vocab
                        .entries_map()
                        .iter()
                        .filter_map(|(&candidate, bytes)| {
                            (token_allowed(&actual_mask, candidate)
                                != token_allowed(&expected_mask, candidate))
                                .then_some((candidate, bytes.clone(), token_allowed(&actual_mask, candidate), token_allowed(&expected_mask, candidate)))
                        })
                        .collect::<Vec<_>>();
                    eprintln!("IGNORE_FUSION_DIFF {differing:?}");
                }
                assert_eq!(
                    actual_mask,
                    expected_mask,
                    "mask mismatch before token {token} in sequence {sequence:?}",
                );
                actual.commit_token(token).unwrap_or_else(|error| {
                    panic!("composed rejected token {token} in {sequence:?}: {error}")
                });
                dynamic.commit_token(token).unwrap_or_else(|error| {
                    panic!("dynamic composed rejected token {token} in {sequence:?}: {error}")
                });
                cached.commit_token(token).unwrap_or_else(|error| {
                    panic!("prepared-cache composed rejected token {token} in {sequence:?}: {error}")
                });
                expected.commit_token(token).unwrap_or_else(|error| {
                    panic!("monolithic rejected token {token} in {sequence:?}: {error}")
                });
            }
            assert_eq!(actual.mask(), expected.mask());
            assert_eq!(dynamic.mask(), expected.mask());
            assert_eq!(cached.mask(), expected.mask());
            assert_eq!(actual.is_accepting(), expected.is_accepting());
            assert_eq!(dynamic.is_accepting(), expected.is_accepting());
            assert_eq!(cached.is_accepting(), expected.is_accepting());
            assert!(actual.is_accepting(), "sequence {sequence:?} should finish");
        }
    }

    #[test]
    fn child_ignore_terminal_is_scoped_across_fused_boundaries() {
        let vocab = Vocab::new(vec![
            (0, b"X a!".to_vec()),
            (1, b"Xa!".to_vec()),
            (2, b"X".to_vec()),
            (3, b" ".to_vec()),
            (4, b"a".to_vec()),
            (5, b"!".to_vec()),
            (6, b" a".to_vec()),
            (7, b"a!".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "X" SUB "!";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                ignore WS;
                t WS ::= " "+;
                nt child ::= "a";
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;

                g child ::= {
                    start child;
                    ignore WS;
                    t WS ::= " "+;
                    nt child ::= "a";
                };

                nt document ::= "X" child "!";
            "#,
            &vocab,
        )
        .unwrap();
        let composed_static = parent
            .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
            .unwrap();
        let composed_dynamic = parent
            .compose_linked_children_for_test_dynamic(&[("SUB", &child)], &vocab)
            .unwrap();
        let source_oracle = compile_inline_source_oracle(
            r#"
                start document;
                extern grammar SUB;
                nt document ::= "X" SUB "!";
            "#,
            &[(
                "SUB",
                "start child; ignore WS; t WS ::= \" \"+; nt child ::= \"a\";",
            )],
            &vocab,
        );

        // Fused-token integration: the scoped child ignore is transparent to
        // within-token follow pruning, so the single fused model token "X a!"
        // (parent X -> child scoped-ignore WS -> child a -> parent !) must be
        // admitted at start by the static shard, matching the oracle and the
        // dynamic backend.
        assert!(
            token_allowed(&composed_static.start().mask(), 0),
            "static shard must admit the fused parent-X + scoped-child-ignore token"
        );
        assert!(token_allowed(&source_oracle.start().mask(), 0));
        assert!(token_allowed(&composed_dynamic.start().mask(), 0));

        // Inside the child scope the child ignore is allowed, but the child
        // cannot complete the document alone (acceptance stays false until the
        // parent's "!").
        let mut static_inside_child = composed_static.start();
        let mut oracle_inside_child = source_oracle.start();
        static_inside_child.commit_token(2).unwrap();
        oracle_inside_child.commit_token(2).unwrap();
        assert!(token_allowed(&static_inside_child.mask(), 3));
        assert!(token_allowed(&oracle_inside_child.mask(), 3));
        assert!(!static_inside_child.is_accepting());
        assert!(!oracle_inside_child.is_accepting());

        assert_constraints_equivalent_on_reachable_prefixes_labeled(
            &source_oracle,
            &monolithic,
            &vocab,
            5,
            "oracle-vs-monolithic",
        );
        assert_constraints_equivalent_on_reachable_prefixes_labeled(
            &composed_static,
            &source_oracle,
            &vocab,
            5,
            "static-vs-oracle",
        );
        assert_constraints_equivalent_on_reachable_prefixes_labeled(
            &composed_dynamic,
            &source_oracle,
            &vocab,
            5,
            "dynamic-vs-oracle",
        );

        // Child trivia is not globally active before the parent has entered
        // the child scope.
        assert!(!token_allowed(&composed_static.start().mask(), 3));
        assert!(!token_allowed(&monolithic.start().mask(), 3));
    }

    #[test]
    fn compact_dynamic_scoped_ignore_parser_preserves_both_entry_scopes() {
        let vocab = byte_vocab();
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                ignore PARENT_WS;
                t PARENT_WS ::= " "+;
                t SUB ::= @token(999);
                nt document ::= "X" SUB "!";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                ignore CHILD_WS;
                t CHILD_WS ::= "\t"+;
                nt child ::= "a";
            "#,
            &vocab,
        )
        .unwrap();
        let slot = terminal(&parent, "SUB");
        let x = terminal(&parent, "X");
        let parent_ws = terminal(&parent, "PARENT_WS");
        let child_ws_local = terminal(&child, "CHILD_WS");
        let composed = compose_constraints_owned_parent_segmented(
            parent,
            &[CompiledSubgrammarInput {
                placeholder_terminal: slot,
                additional_placeholder_terminals: &[],
                constraint: &child,
            }],
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .unwrap()
        .constraint;
        assert!(composed.uses_compact_segmented_parser_runtime());
        let overlay = composed.static_dynamic_overlay.as_ref().unwrap();
        let child_ws = overlay.segmented_parser_components[1].terminal_offset + child_ws_local;
        let state = composed.initial_state_map();
        let start = state.values().next().unwrap();
        let after_x = composed
            .advance_compact_segmented_parser(start, x)
            .expect("compact parser runtime");
        let mut scopes = after_x
            .peek_values()
            .into_iter()
            .map(|state| composed.compact_segmented_parser_component(state).unwrap().0)
            .collect::<Vec<_>>();
        scopes.sort_unstable();
        scopes.dedup();
        assert_eq!(scopes, vec![0, 1], "CALL closure must retain parent and child choices");

        let after_parent_ws = composed
            .advance_compact_segmented_parser(&after_x, parent_ws)
            .expect("compact parser runtime");
        assert!(!after_parent_ws.is_empty(), "parent trivia must remain legal before CALL");
        assert!(after_parent_ws.peek_values().into_iter().any(|state| {
            composed
                .compact_segmented_parser_component(state)
                .is_some_and(|(component, _)| component == 1)
        }));

        let after_child_ws = composed
            .advance_compact_segmented_parser(&after_x, child_ws)
            .expect("compact parser runtime");
        assert!(!after_child_ws.is_empty(), "child trivia must be legal after CALL");
        assert!(after_child_ws.peek_values().into_iter().all(|state| {
            composed
                .compact_segmented_parser_component(state)
                .is_some_and(|(component, _)| component == 1)
        }));
    }

    #[test]
    fn distinct_parent_and_child_ignore_terminals_match_inline_scoped_semantics() {
        let vocab = Vocab::new(vec![
            (0, b"X \ta!".to_vec()),
            (1, b"X\ta!".to_vec()),
            (2, b"X".to_vec()),
            (3, b" ".to_vec()),
            (4, b"\t".to_vec()),
            (5, b"a".to_vec()),
            (6, b"!".to_vec()),
            (7, b" \ta".to_vec()),
            (8, b"a!".to_vec()),
            // Distinguish scope ordering at entry and return. Child trivia
            // followed by parent trivia before child syntax is invalid; child
            // trailing trivia followed by parent trivia is valid, while the
            // reverse ordering is invalid after return has begun.
            (9, b"X\t a!".to_vec()),
            (10, b"Xa\t !".to_vec()),
            (11, b"Xa \t!".to_vec()),
            (12, b"\tX a!".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                ignore PARENT_WS;
                t PARENT_WS ::= " "+;
                t SUB ::= @token(999);
                nt document ::= "X" SUB "!";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                ignore CHILD_WS;
                t CHILD_WS ::= "\t"+;
                nt child ::= "a";
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                ignore PARENT_WS;
                t PARENT_WS ::= " "+;

                g child ::= {
                    start child;
                    ignore CHILD_WS;
                    t CHILD_WS ::= "\t"+;
                    nt child ::= "a";
                };

                nt document ::= "X" child "!";
            "#,
            &vocab,
        )
        .unwrap();
        let composed_static = parent
            .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
            .unwrap();
        let composed_dynamic = parent
            .compose_linked_children_for_test_dynamic(&[("SUB", &child)], &vocab)
            .unwrap();
        let loaded = Constraint::load(&composed_static.save()).unwrap();

        for constraint in [&composed_static, &loaded] {
            assert!(constraint.ignore_terminal.is_none());
        }

        // The inline scoped-ignore lowering materialises nullable skip
        // productions. Its `is_complete()` predicate can report a trivia-only
        // prefix as complete before the visible root has parsed. Compare exact
        // masks/commit language here and assert completion on the complete
        // boundary strings below, rather than preserving that unrelated
        // inline-lowering artefact in the explicit linker. Per-stage mask
        // attribution for this witness lives in `runtime::mask`
        // (`GLRMASK_DEBUG_MASK_STAGES`).
        assert_constraints_mask_equivalent_on_reachable_prefixes_labeled(
            &composed_static,
            &monolithic,
            &vocab,
            5,
            "static-vs-monolithic",
        );
        assert_constraints_mask_equivalent_on_reachable_prefixes_labeled(
            &composed_dynamic,
            &monolithic,
            &vocab,
            5,
            "dynamic-vs-monolithic",
        );
        assert_constraints_equivalent_on_reachable_prefixes_labeled(
            &loaded,
            &composed_static,
            &vocab,
            5,
            "loaded-vs-static",
        );

        for (token, expected) in [(0, true), (1, true), (9, false), (10, true), (11, false), (12, false)] {
            let mut actual = composed_static.start();
            let mut dynamic = composed_dynamic.start();
            let mut loaded_state = loaded.start();
            let mut reference = monolithic.start();
            assert_eq!(actual.commit_token(token).is_ok(), expected, "token {token}");
            assert_eq!(dynamic.commit_token(token).is_ok(), expected, "dynamic token {token}");
            assert_eq!(
                loaded_state.commit_token(token).is_ok(),
                expected,
                "loaded token {token}",
            );
            assert_eq!(reference.commit_token(token).is_ok(), expected, "reference token {token}");
            assert_eq!(actual.is_accepting(), expected, "token {token}");
            assert_eq!(dynamic.is_accepting(), expected, "dynamic token {token}");
            assert_eq!(loaded_state.is_accepting(), expected, "loaded token {token}");
            assert_eq!(reference.is_accepting(), expected, "reference token {token}");
        }
    }

    #[test]
    fn scoped_ignore_oracle_survives_reload_and_nested_recomposition() {
        let vocab = Vocab::new(vec![
            (0, b"<".to_vec()),
            (1, b">".to_vec()),
            (2, b"X".to_vec()),
            (3, b"!".to_vec()),
            (4, b" ".to_vec()),
            (5, b"\t".to_vec()),
            (6, b"a".to_vec()),
            (7, b"\t\t".to_vec()),
            (8, b"\ta".to_vec()),
            (9, b"a\t".to_vec()),
            (10, b"X\t".to_vec()),
            (11, b"a ".to_vec()),
            (12, b"a\t ".to_vec()),
            (13, b"a \t".to_vec()),
            (14, b"<X\t".to_vec()),
            (15, b"!>".to_vec()),
            (16, b"X\ta\t !".to_vec()),
            (17, b"<X\ta\t !>".to_vec()),
            (18, b"X \ta!".to_vec()),
            (19, b"X\t a!".to_vec()),
            (20, b"Xa\t !".to_vec()),
            (21, b"Xa \t!".to_vec()),
            (22, b"X a!".to_vec()),
            (23, b"<X\t a!>".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                ignore PARENT_WS;
                t PARENT_WS ::= " "+;
                t SUB ::= @token(999);
                nt document ::= "X" SUB "!";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                ignore CHILD_WS;
                t CHILD_WS ::= "\t"+;
                nt child ::= "a";
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                ignore PARENT_WS;
                t PARENT_WS ::= " "+;
                g child ::= {
                    start child;
                    ignore CHILD_WS;
                    t CHILD_WS ::= "\t"+;
                    nt child ::= "a";
                };
                nt document ::= "X" child "!";
            "#,
            &vocab,
        )
        .unwrap();
        let composed = parent
            .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
            .unwrap();
        let composed_dynamic = parent
            .compose_linked_children_for_test_dynamic(&[("SUB", &child)], &vocab)
            .unwrap();
        let loaded = Constraint::load(&composed.save()).unwrap();
        let make_states = || {
            let mut states = Vec::with_capacity(4);
            states.push(composed.start());
            states.push(composed_dynamic.start());
            states.push(loaded.start());
            states.push(monolithic.start());
            states
        };

        // Exhaust the small reachable token-prefix graph.  We compare masks
        // rather than the inline lowering's trivia-only completion artifact;
        // successful complete strings are checked explicitly below.
        assert_constraints_mask_equivalent_on_reachable_prefixes_labeled(
            &composed,
            &monolithic,
            &vocab,
            4,
            "static-vs-monolithic",
        );
        assert_constraints_mask_equivalent_on_reachable_prefixes_labeled(
            &composed_dynamic,
            &monolithic,
            &vocab,
            4,
            "dynamic-vs-monolithic",
        );
        assert_constraints_equivalent_on_reachable_prefixes_labeled(
            &loaded,
            &composed,
            &vocab,
            4,
            "loaded-vs-static",
        );

        // Cover each scoped-ignore boundary shape explicitly:
        // - parent ignore before child;
        // - child ignore as the first child token;
        // - repeated / ignore-only child tokens;
        // - ignore+real and real+ignore fused model tokens;
        // - child return followed by parent ignore;
        // - entry/exit fusion in one model token.
        let valid_sequences: &[&[u32]] = &[
            &[2, 4, 5, 6, 3],
            &[2, 5, 6, 3],
            &[2, 7, 6, 3],
            &[2, 8, 3],
            &[2, 9, 3],
            &[2, 11, 3],
            &[2, 12, 3],
            &[10, 6, 3],
            &[16],
            &[18],
            &[20],
            &[22],
        ];
        for &sequence in valid_sequences {
            let mut actual = composed.start();
            let mut dynamic = composed_dynamic.start();
            let mut restored = loaded.start();
            let mut expected = monolithic.start();
            for &token in sequence {
                assert_eq!(actual.mask(), expected.mask(), "mask before {sequence:?} token {token}");
                assert_eq!(dynamic.mask(), expected.mask(), "dynamic mask before {sequence:?} token {token}");
                assert_eq!(restored.mask(), expected.mask(), "loaded mask before {sequence:?} token {token}");
                actual.commit_token(token).unwrap();
                dynamic.commit_token(token).unwrap();
                restored.commit_token(token).unwrap();
                expected.commit_token(token).unwrap();
            }
            assert!(actual.is_accepting(), "composed incomplete for {sequence:?}");
            assert!(dynamic.is_accepting(), "dynamic composed incomplete for {sequence:?}");
            assert!(restored.is_accepting(), "loaded incomplete for {sequence:?}");
            assert!(expected.is_accepting(), "reference incomplete for {sequence:?}");
        }

        // Parent trivia must not become active while the child scope is still
        // parsing, and child trivia must not leak back into the parent after
        // the child has returned.
        for sequence in [&[2u32, 13][..], &[19][..], &[21][..]] {
            let mut actual = composed.start();
            let mut dynamic = composed_dynamic.start();
            let mut restored = loaded.start();
            let mut expected = monolithic.start();
            for &token in &sequence[..sequence.len() - 1] {
                actual.commit_token(token).unwrap();
                dynamic.commit_token(token).unwrap();
                restored.commit_token(token).unwrap();
                expected.commit_token(token).unwrap();
            }
            let token = *sequence.last().unwrap();
            let expected_result = expected.commit_token(token).is_ok();
            assert_eq!(actual.commit_token(token).is_ok(), expected_result);
            assert_eq!(dynamic.commit_token(token).is_ok(), expected_result);
            assert_eq!(restored.commit_token(token).is_ok(), expected_result);
            assert!(!expected_result, "invalid scoped sequence {sequence:?} was accepted");
        }

        // Reuse the serialized, already-composed constraint as a child again.
        // This is the inherited-skip case: CHILD_WS is no longer a top-level
        // ignore of `loaded`, but its scoped phase behavior must survive the
        // next call/return boundary exactly.
        let outer_parent = Constraint::from_glrm_grammar(
            r#"
                start outer;
                t INNER ::= @token(1000);
                nt outer ::= "<" INNER ">";
            "#,
            &vocab,
        )
        .unwrap();
        let outer = outer_parent
            .compose_linked_children_for_test(&[("INNER", &loaded)], &vocab)
            .unwrap();
        let outer_dynamic = outer_parent
            .compose_linked_children_for_test_dynamic(&[("INNER", &loaded)], &vocab)
            .unwrap();
        let outer_monolithic = Constraint::from_glrm_grammar(
            r#"
                start outer;
                g inner ::= {
                    start document;
                    ignore PARENT_WS;
                    t PARENT_WS ::= " "+;
                    g child ::= {
                        start child;
                        ignore CHILD_WS;
                        t CHILD_WS ::= "\t"+;
                        nt child ::= "a";
                    };
                    nt document ::= "X" child "!";
                };
                nt outer ::= "<" inner ">";
            "#,
            &vocab,
        )
        .unwrap();

        assert_constraints_mask_equivalent_on_reachable_prefixes_labeled(
            &outer,
            &outer_monolithic,
            &vocab,
            4,
            "outer-static-vs-outer-monolithic",
        );
        assert_constraints_mask_equivalent_on_reachable_prefixes_labeled(
            &outer_dynamic,
            &outer_monolithic,
            &vocab,
            4,
            "outer-dynamic-vs-outer-monolithic",
        );
        for sequence in [&[0u32, 16, 1][..], &[14, 6, 15][..], &[17][..]] {
            let mut actual = outer.start();
            let mut dynamic = outer_dynamic.start();
            let mut expected = outer_monolithic.start();
            for &token in sequence {
                assert_eq!(actual.mask(), expected.mask(), "outer mask before {sequence:?} token {token}");
                assert_eq!(dynamic.mask(), expected.mask(), "outer dynamic mask before {sequence:?} token {token}");
                actual.commit_token(token).unwrap();
                dynamic.commit_token(token).unwrap();
                expected.commit_token(token).unwrap();
            }
            assert!(actual.is_accepting(), "outer incomplete for {sequence:?}");
            assert!(dynamic.is_accepting(), "outer dynamic incomplete for {sequence:?}");
            assert!(expected.is_accepting(), "outer reference incomplete for {sequence:?}");
        }

        let mut actual = outer.start();
        let mut dynamic = outer_dynamic.start();
        let mut expected = outer_monolithic.start();
        assert_eq!(actual.commit_token(23).is_ok(), expected.commit_token(23).is_ok());
        assert_eq!(dynamic.commit_token(23).is_ok(), expected.commit_token(23).is_ok());
        assert!(!expected.is_accepting());
    }

    /// Replay a token sequence through one route, returning the index of the
    /// first rejected token (`None` if the whole sequence commits).
    fn replay_first_rejection(
        commit: &mut dyn FnMut(u32) -> bool,
        sequence: &[u32],
    ) -> Option<usize> {
        sequence.iter().position(|&token| !commit(token))
    }

    #[test]
    fn integration_scoped_ignore_follow_boundary_masks() {
        // Parent `X SUB !` with a parent-space ignore; child `a` then `b`
        // with a child-tab ignore. Completion-aware mask parity (depth 4)
        // across static/dynamic/loaded/inline; explicit valid completions;
        // rejections with first-rejection-index agreement; save/load parity;
        // real StaticParser shards installed.
        let vocab = Vocab::new(vec![
            (0, b"X".to_vec()),
            (1, b"!".to_vec()),
            (2, b"a".to_vec()),
            (3, b"b".to_vec()),
            (4, b"c".to_vec()),
            (5, b" ".to_vec()),
            (6, b"\t".to_vec()),
            (7, b"X\ta".to_vec()),
            (8, b"X\tab!".to_vec()),
            (9, b"a\tb".to_vec()),
            (10, b"b\t !".to_vec()),
            (11, b"X\tb!".to_vec()),
            (12, b"Xa\tc!".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                ignore PARENT_WS;
                t PARENT_WS ::= " "+;
                t SUB ::= @token(999);
                nt document ::= "X" SUB "!";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                ignore CHILD_WS;
                t CHILD_WS ::= "\t"+;
                nt child ::= "a" "b";
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                ignore PARENT_WS;
                t PARENT_WS ::= " "+;
                g child ::= {
                    start child;
                    ignore CHILD_WS;
                    t CHILD_WS ::= "\t"+;
                    nt child ::= "a" "b";
                };
                nt document ::= "X" child "!";
            "#,
            &vocab,
        )
        .unwrap();
        let composed = parent
            .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
            .unwrap();
        let composed_dynamic = parent
            .compose_linked_children_for_test_dynamic(&[("SUB", &child)], &vocab)
            .unwrap();
        let loaded = Constraint::load(&composed.save()).unwrap();
        let make_states = || {
            let mut states = Vec::with_capacity(4);
            states.push(composed.start());
            states.push(composed_dynamic.start());
            states.push(loaded.start());
            states.push(monolithic.start());
            states
        };
        // Real StaticParser shards installed (not fallback-only).
        let overlay = composed
            .static_dynamic_overlay
            .as_ref()
            .expect("static_dynamic_overlay must be present");
        assert!(
            !overlay.segmented_parser_components.is_empty(),
            "static link must install parser components",
        );
        for (index, component) in overlay.segmented_parser_components.iter().enumerate() {
            match &component.boundary.as_ref().expect("component must have a boundary shard").backend {
                crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_) => {}
                other => panic!("component {index} must be StaticParser, got {other:?}"),
            }
        }
        assert_constraints_equivalent_on_reachable_prefixes_labeled(
            &composed,
            &monolithic,
            &vocab,
            4,
            "integration-ignore static-vs-inline",
        );
        assert_constraints_equivalent_on_reachable_prefixes_labeled(
            &composed_dynamic,
            &monolithic,
            &vocab,
            4,
            "integration-ignore dynamic-vs-inline",
        );
        // Valid strings complete on all four builds (incl. ignore ownership:
        // [0,5,6,2,6,3,6,5,1] spaces the parent parts, tabs the child parts;
        // [0,2,10] fuses child-b + parent-ignore + `!`).
        for sequence in [
            &[0u32, 2, 3, 1][..],
            &[7, 3, 1][..],
            &[0, 9, 1][..],
            &[8][..],
            &[0, 5, 6, 2, 6, 3, 6, 5, 1][..],
            &[0, 2, 10][..],
        ] {
            let mut states = make_states();
            for &token in sequence {
                let masks: Vec<Vec<u32>> =
                    states.iter().map(|state| state.mask()).collect();
                assert_eq!(masks[0], masks[3], "static mask before {sequence:?} token {token}");
                assert_eq!(masks[1], masks[3], "dynamic mask before {sequence:?} token {token}");
                assert_eq!(masks[2], masks[3], "loaded mask before {sequence:?} token {token}");
                for state in states.iter_mut() {
                    state.commit_token(token).unwrap_or_else(|error| {
                        panic!("valid sequence {sequence:?} rejected at {token}: {error}")
                    });
                }
            }
            for (label, state) in
                ["static", "dynamic", "loaded", "inline"].iter().zip(states.iter())
            {
                assert!(state.is_accepting(), "{label} incomplete for {sequence:?}");
            }
        }
        // Invalid: missing `a` ([0,1]), unexpected `c` ([12]), wrong-scope
        // space inside the child ([0,2,5,3,1]), stray fused token ([11]).
        // Every route must reject at least one token; first-rejection index
        // must agree across routes.
        for sequence in [&[0u32, 1][..], &[0, 2, 4, 1][..], &[12][..], &[11][..], &[0, 2, 5, 3, 1][..]] {
            let mut rejects = Vec::new();
            for mut state in make_states() {
                rejects.push(replay_first_rejection(
                    &mut |token| state.commit_token(token).is_ok(),
                    sequence,
                ));
            }
            for (label, rejection) in
                ["static", "dynamic", "loaded", "inline"].iter().zip(rejects.iter())
            {
                assert!(
                    rejection.is_some(),
                    "{label} accepted invalid scoped sequence {sequence:?}",
                );
            }
            assert_eq!(rejects[0], rejects[3], "static/inline rejection index on {sequence:?}");
            assert_eq!(rejects[1], rejects[3], "dynamic/inline rejection index on {sequence:?}");
            assert_eq!(rejects[2], rejects[3], "loaded/inline rejection index on {sequence:?}");
        }
        // Save/load preserves masks + completion.
        assert_constraints_equivalent_on_reachable_prefixes_labeled(
            &loaded,
            &composed,
            &vocab,
            4,
            "integration-ignore loaded-vs-static",
        );
    }

    #[test]
    fn integration_finite_exclusion_boundary_masks() {
        // Parent `X SUB !` (no ignores); child `OPEN ::= [ab]+ - "ab"`.
        // Tests exclusion-operator composition compatibility (static vs
        // dynamic vs inline, save/load). NOT a certificate/master-activation
        // proof: the tiny fixture does not trigger dynamic-virtual
        // optimization (see the dedicated lexer certificate unit test).
        // Prefix `Xab` stays extendible; only completed `Xab!` is forbidden.
        let vocab = Vocab::new(vec![
            (0, b"X".to_vec()),
            (1, b"!".to_vec()),
            (2, b"a".to_vec()),
            (3, b"b".to_vec()),
            (4, b"Xa".to_vec()),
            (5, b"Xb".to_vec()),
            (6, b"Xab".to_vec()),
            (7, b"Xaa".to_vec()),
            (8, b"ab".to_vec()),
            (9, b"aa".to_vec()),
            (10, b"ab!".to_vec()),
            (11, b"aa!".to_vec()),
            (12, b"Xa!".to_vec()),
            (13, b"Xb!".to_vec()),
            (14, b"Xab!".to_vec()),
            (15, b"Xaa!".to_vec()),
            (16, b"Xaba!".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "X" SUB "!";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                t OPEN ::= [ab]+ - "ab";
                nt child ::= OPEN;
            "#,
            &vocab,
        )
        .unwrap();
        // The child lowers `OPEN` to an exclusion operator (AST-level proof
        // the sidecar-eligible shape reaches composition; independent of any
        // tokenizer certificate location).
        let open_terminal = child
            .terminal_display_names()
            .iter()
            .position(|name| name == "OPEN")
            .expect("child must have OPEN terminal") as u32;
        assert!(
            matches!(
                child.retained_terminal_expr(open_terminal),
                Some(Expr::Exclude { .. })
            ),
            "child OPEN terminal must lower to Expr::Exclude",
        );
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                g child ::= {
                    start child;
                    t OPEN ::= [ab]+ - "ab";
                    nt child ::= OPEN;
                };
                nt document ::= "X" child "!";
            "#,
            &vocab,
        )
        .unwrap();
        let composed = parent
            .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
            .unwrap();
        let composed_dynamic = parent
            .compose_linked_children_for_test_dynamic(&[("SUB", &child)], &vocab)
            .unwrap();
        let loaded = Constraint::load(&composed.save()).unwrap();
        // Keep the large ConstraintState objects off this test function's
        // stack. Several four-state inline arrays were enough to exceed the
        // default Rust test-thread stack on macOS before the first statement
        // executed, despite the runtime code itself being fine.
        let make_states = || {
            let mut states = Vec::with_capacity(4);
            states.push(composed.start());
            states.push(composed_dynamic.start());
            states.push(loaded.start());
            states.push(monolithic.start());
            states
        };
        // Real StaticParser shards installed (not fallback-only).
        let overlay = composed
            .static_dynamic_overlay
            .as_ref()
            .expect("static_dynamic_overlay must be present");
        assert!(
            !overlay.segmented_parser_components.is_empty(),
            "static link must install parser components",
        );
        for (index, component) in overlay.segmented_parser_components.iter().enumerate() {
            match &component.boundary.as_ref().expect("component must have a boundary shard").backend {
                crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_) => {}
                other => panic!("component {index} must be StaticParser, got {other:?}"),
            }
        }
        assert_constraints_equivalent_on_reachable_prefixes_labeled(
            &composed,
            &composed_dynamic,
            &vocab,
            4,
            "integration-exclusion static-vs-dynamic",
        );
        assert_constraints_equivalent_on_reachable_prefixes_labeled(
            &composed,
            &monolithic,
            &vocab,
            4,
            "integration-exclusion static-vs-inline",
        );
        // Valid completions, incl. legitimate Xabab!/Xaab! and char-by-char.
        for sequence in [
            &[12u32][..],
            &[13][..],
            &[15][..],
            &[16][..],
            &[0, 2, 1][..],
            &[0, 3, 1][..],
            &[0, 2, 2, 1][..],
            &[0, 2, 3, 2, 1][..],
            &[6, 8, 1][..],
            &[4, 8, 1][..],
            &[4, 2, 1][..],
            &[4, 9, 1][..],
        ] {
            let mut states = make_states();
            for &token in sequence {
                let masks: Vec<Vec<u32>> =
                    states.iter().map(|state| state.mask()).collect();
                assert_eq!(masks[0], masks[1], "static/dynamic mask before {sequence:?} token {token}");
                assert_eq!(masks[0], masks[3], "static/inline mask before {sequence:?} token {token}");
                assert_eq!(masks[2], masks[3], "loaded/inline mask before {sequence:?} token {token}");
                for state in states.iter_mut() {
                    state.commit_token(token).unwrap_or_else(|error| {
                        panic!("valid exclusion sequence {sequence:?} rejected at {token}: {error}")
                    });
                }
            }
            for (label, state) in
                ["static", "dynamic", "loaded", "inline"].iter().zip(states.iter())
            {
                assert!(state.is_accepting(), "{label} incomplete for {sequence:?}");
            }
        }
        // `Xab` remains extendible on every route: commit 6, then 2, then 1,
        // and assert acceptance (Xaba! is valid).
        for mut state in make_states() {
            for &token in &[6u32, 2, 1] {
                state.commit_token(token).unwrap_or_else(|error| {
                    panic!("Xab extension rejected at {token}: {error}")
                });
            }
            assert!(state.is_accepting(), "Xaba! must complete");
        }
        // Rejected: completed `Xab!` only.
        for sequence in [&[14u32][..], &[0, 2, 3, 1][..], &[6, 1][..], &[4, 3, 1][..], &[0, 8, 1][..]] {
            let mut rejects = Vec::new();
            for mut state in make_states() {
                rejects.push(replay_first_rejection(
                    &mut |token| state.commit_token(token).is_ok(),
                    sequence,
                ));
            }
            for (label, rejection) in
                ["static", "dynamic", "loaded", "inline"].iter().zip(rejects.iter())
            {
                assert!(
                    rejection.is_some(),
                    "{label} accepted forbidden exclusion sequence {sequence:?}",
                );
            }
            assert_eq!(rejects[0], rejects[3], "static/inline rejection index on {sequence:?}");
            assert_eq!(rejects[1], rejects[3], "dynamic/inline rejection index on {sequence:?}");
            assert_eq!(rejects[2], rejects[3], "loaded/inline rejection index on {sequence:?}");
        }
        // Save/load preserves masks + completion.
        assert_constraints_equivalent_on_reachable_prefixes_labeled(
            &loaded,
            &composed,
            &vocab,
            4,
            "integration-exclusion loaded-vs-static",
        );
    }

    #[test]
    fn adjacent_precomposed_child_can_begin_with_child_only_token() {
        let vocab = Vocab::new(vec![
            // End the first child at a model-token boundary, then begin the
            // already-composed second child with a token containing no parent
            // or first-child terminal at all.  A mixed-owner-only boundary
            // selector misses token 1; FIRST/FOLLOW boundary beginnings must
            // retain it.
            (0, b"Xa".to_vec()),
            (1, b"b".to_vec()),
            (2, b"!".to_vec()),
            (3, b"\nb".to_vec()),
            (4, b" b".to_vec()),
            (5, b"a".to_vec()),
            (6, b"X".to_vec()),
            (7, b"\t".to_vec()),
            (8, b"\n".to_vec()),
            (9, b" ".to_vec()),
            (10, b"\n\n".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                ignore PARENT_WS;
                t PARENT_WS ::= " "+;
                t LEFT ::= @token(998);
                t RIGHT ::= @token(999);
                nt document ::= "X" LEFT RIGHT "!";
            "#,
            &vocab,
        )
        .unwrap();
        let left = Constraint::from_glrm_grammar(
            r#"
                start left;
                ignore LEFT_WS;
                t LEFT_WS ::= "\t"+;
                nt left ::= "a";
            "#,
            &vocab,
        )
        .unwrap();
        let right = Constraint::from_glrm_grammar(
            r#"
                start right;
                ignore RIGHT_WS;
                t RIGHT_WS ::= "\n"+;
                nt right ::= "b";
            "#,
            &vocab,
        )
        .unwrap();

        // Compose RIGHT first.  When LEFT is linked later, LEFT's parent
        // continuation begins by traversing RIGHT's already-existing control
        // edge.  The direct continuation row therefore exposes a control
        // terminal, not lexical terminal "b".
        let parent_with_right = parent
            .compose_linked_children_for_test(&[("RIGHT", &right)], &vocab)
            .unwrap();
        let composed = parent_with_right
            .compose_linked_children_for_test(&[("LEFT", &left)], &vocab)
            .unwrap();
        let parent_with_right_dynamic = parent
            .compose_linked_children_for_test_dynamic(&[("RIGHT", &right)], &vocab)
            .unwrap();
        let composed_dynamic = parent_with_right_dynamic
            .compose_linked_children_for_test_dynamic(&[("LEFT", &left)], &vocab)
            .unwrap();

        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                ignore PARENT_WS;
                t PARENT_WS ::= " "+;

                g left ::= {
                    start left;
                    ignore LEFT_WS;
                    t LEFT_WS ::= "\t"+;
                    nt left ::= "a";
                };
                g right ::= {
                    start right;
                    ignore RIGHT_WS;
                    t RIGHT_WS ::= "\n"+;
                    nt right ::= "b";
                };

                nt document ::= "X" left right "!";
            "#,
            &vocab,
        )
        .unwrap();

        let mut actual = composed.start();
        let mut reference = monolithic.start();
        actual.commit_token(0).unwrap();
        reference.commit_token(0).unwrap();
        assert!(token_allowed(&actual.mask(), 1));
        assert!(token_allowed(&reference.mask(), 1));
        actual.commit_token(1).unwrap();
        reference.commit_token(1).unwrap();
        actual.commit_token(2).unwrap();
        reference.commit_token(2).unwrap();
        assert!(actual.is_accepting());
        assert!(reference.is_accepting());

        // Leading RIGHT trivia also begins only after return-from-LEFT followed
        // by enter-RIGHT, and must preserve that scope across the token.
        let mut actual = composed.start();
        let mut reference = monolithic.start();
        actual.commit_token(0).unwrap();
        reference.commit_token(0).unwrap();
        assert!(token_allowed(&actual.mask(), 3));
        assert!(token_allowed(&reference.mask(), 3));
        actual.commit_token(3).unwrap();
        reference.commit_token(3).unwrap();
        actual.commit_token(2).unwrap();
        reference.commit_token(2).unwrap();
        assert!(actual.is_accepting());
        assert!(reference.is_accepting());

        // A multi-byte token containing only RIGHT trivia must itself be a
        // boundary-begin path: return from LEFT, enter RIGHT, consume trivia,
        // and persist the RIGHT start state for the next model token. This is
        // not an owner-crossing token and is not covered by one-byte seed
        // relations.
        let mut actual = composed.start();
        let mut reference = monolithic.start();
        actual.commit_token(0).unwrap();
        reference.commit_token(0).unwrap();
        assert!(token_allowed(&actual.mask(), 10));
        assert!(token_allowed(&reference.mask(), 10));
        actual.commit_token(10).unwrap();
        reference.commit_token(10).unwrap();
        actual.commit_token(1).unwrap();
        reference.commit_token(1).unwrap();
        actual.commit_token(2).unwrap();
        reference.commit_token(2).unwrap();
        assert!(actual.is_accepting());
        assert!(reference.is_accepting());

        assert_constraints_mask_equivalent_on_reachable_prefixes_labeled(
            &composed_dynamic,
            &monolithic,
            &vocab,
            4,
            "dynamic-vs-monolithic",
        );
    }

    #[test]
    fn adjacent_calls_to_same_child_accept_child_only_second_token() {
        let vocab = Vocab::new(vec![
            (0, b"Xa".to_vec()),
            (1, b"a".to_vec()),
            (2, b"!".to_vec()),
            (3, b"Xaa!".to_vec()),
            (4, b"X".to_vec()),
            (5, b"a!".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "X" SUB SUB "!";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child ::= "a";
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                nt child ::= "a";
                nt document ::= "X" child child "!";
            "#,
            &vocab,
        )
        .unwrap();
        let composed = parent
            .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
            .unwrap();

        let mut actual = composed.start();
        let mut reference = monolithic.start();
        actual.commit_token(0).unwrap();
        reference.commit_token(0).unwrap();
        assert!(token_allowed(&actual.mask(), 1));
        assert!(token_allowed(&reference.mask(), 1));
        actual.commit_token(1).unwrap();
        reference.commit_token(1).unwrap();
        actual.commit_token(2).unwrap();
        reference.commit_token(2).unwrap();
        assert!(actual.is_accepting());
        assert!(reference.is_accepting());

        assert_constraints_equivalent_on_reachable_prefixes_labeled(
            &composed,
            &monolithic,
            &vocab,
            4,
            "polluter-static-vs-monolithic",
        );
    }

    #[test]
    fn adjacent_children_with_uniform_ignore_compile_controls_and_keep_global_ignore() {
        let vocab = Vocab::new(vec![
            (0, b"X a".to_vec()),
            (1, b" a".to_vec()),
            (2, b" !".to_vec()),
            (3, b" X a a ! ".to_vec()),
            (4, b"X".to_vec()),
            (5, b" ".to_vec()),
            (6, b"a".to_vec()),
            (7, b"!".to_vec()),
            (8, b"<".to_vec()),
            (9, b">".to_vec()),
            (10, b"< X a a ! >".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                ignore WS;
                t WS ::= " "+;
                t SUB ::= @token(999);
                nt document ::= "X" SUB SUB "!";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                ignore WS;
                t WS ::= " "+;
                nt child ::= "a";
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                ignore WS;
                t WS ::= " "+;
                nt child ::= "a";
                nt document ::= "X" child child "!";
            "#,
            &vocab,
        )
        .unwrap();
        let composed = parent
            .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
            .unwrap();
        let composed_dynamic = parent
            .compose_linked_children_for_test_dynamic(&[("SUB", &child)], &vocab)
            .unwrap();
        let loaded = Constraint::load(&composed.save()).unwrap();

        for constraint in [&composed, &loaded] {
            assert!(constraint.ignore_terminal.is_some());
            assert!(
                constraint.table.control_terminals.is_empty(),
                "linker entry/return controls must be compiled out of the runtime table"
            );
            assert!(
                constraint.table.skip_terminals.is_empty(),
                "a globally erasable ignore must not be materialized in parser rows"
            );
            assert!(constraint.table.action.iter().all(|row| {
                row.iter().all(|(_, action)| !matches!(action, Action::Skip))
            }));
        }

        for sequence in [vec![0, 1, 2], vec![3], vec![4, 5, 6, 5, 6, 5, 7, 5]] {
            let mut actual = composed.start();
            let mut dynamic = composed_dynamic.start();
            let mut restored = loaded.start();
            let mut expected = monolithic.start();
            for token in sequence {
                assert_eq!(actual.mask(), expected.mask(), "mask mismatch before token {token}");
                assert_eq!(dynamic.mask(), expected.mask(), "dynamic mask mismatch before token {token}");
                assert_eq!(restored.mask(), expected.mask(), "loaded mask mismatch before token {token}");
                actual.commit_token(token).unwrap();
                dynamic.commit_token(token).unwrap();
                restored.commit_token(token).unwrap();
                expected.commit_token(token).unwrap();
            }
            assert!(actual.is_accepting());
            assert!(dynamic.is_accepting());
            assert!(restored.is_accepting());
            assert!(expected.is_accepting());
        }

        assert_constraints_equivalent_on_reachable_prefixes_labeled(
            &composed,
            &monolithic,
            &vocab,
            4,
            "static-vs-monolithic",
        );
        assert_constraints_equivalent_on_reachable_prefixes_labeled(
            &composed_dynamic,
            &monolithic,
            &vocab,
            4,
            "dynamic-vs-monolithic",
        );
        assert_constraints_equivalent_on_reachable_prefixes_labeled(
            &loaded,
            &composed,
            &vocab,
            4,
            "loaded-vs-static",
        );

        // The compiled child no longer carries runtime controls. Reusing it in
        // an outer composition must retain the same globally erased ignore and
        // compile the new call/return controls away again.
        let outer_parent = Constraint::from_glrm_grammar(
            r#"
                start outer;
                ignore WS;
                t WS ::= " "+;
                t INNER ::= @token(1000);
                nt outer ::= "<" INNER ">";
            "#,
            &vocab,
        )
        .unwrap();
        let outer_monolithic = Constraint::from_glrm_grammar(
            r#"
                start outer;
                ignore WS;
                t WS ::= " "+;
                nt child ::= "a";
                nt outer ::= "<" "X" child child "!" ">";
            "#,
            &vocab,
        )
        .unwrap();
        let outer = outer_parent
            .clone()
            .compose_linked_children_for_test(&[("INNER", &composed)], &vocab)
            .unwrap();
        let outer_from_loaded = outer_parent
            .compose_linked_children_for_test(&[("INNER", &loaded)], &vocab)
            .unwrap();
        for constraint in [&outer, &outer_from_loaded] {
            assert!(constraint.ignore_terminal.is_some());
            assert!(constraint.ignore_expr.is_some());
            assert!(constraint.table.control_terminals.is_empty());
            assert!(constraint.table.skip_terminals.is_empty());
            assert!(constraint.table.action.iter().all(|row| {
                row.iter().all(|(_, action)| !matches!(action, Action::Skip))
            }));
        }
        for sequence in [vec![10], vec![8, 3, 9]] {
            let mut actual = outer.start();
            let mut restored_actual = outer_from_loaded.start();
            let mut expected = outer_monolithic.start();
            for token in sequence {
                assert_eq!(actual.mask(), expected.mask(), "nested mask mismatch before token {token}");
                assert_eq!(
                    restored_actual.mask(),
                    expected.mask(),
                    "loaded-child nested mask mismatch before token {token}"
                );
                actual.commit_token(token).unwrap();
                restored_actual.commit_token(token).unwrap();
                expected.commit_token(token).unwrap();
            }
            assert!(actual.is_accepting());
            assert!(restored_actual.is_accepting());
            assert!(expected.is_accepting());
        }
    }

    #[test]
    fn nonterminal_mediated_adjacent_calls_use_explicit_linker() {
        let vocab = Vocab::new(vec![
            (0, b"Xa".to_vec()),
            (1, b"a".to_vec()),
            (2, b"!".to_vec()),
            (3, b"Xaa!".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                nt second ::= SUB;
                nt document ::= "X" SUB second "!";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child ::= "a";
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                nt child ::= "a";
                nt second ::= child;
                nt document ::= "X" child second "!";
            "#,
            &vocab,
        )
        .unwrap();
        let composed = parent
            .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
            .unwrap();

        assert_constraints_equivalent_on_reachable_prefixes(
            &composed,
            &monolithic,
            &vocab,
            4,
        );
    }

    #[test]
    fn same_compiled_child_can_fill_two_distinct_placeholders() {
        let vocab = Vocab::new(vec![
            (0, b"<a>,<b>".to_vec()),
            (1, b"<b>,<a>".to_vec()),
            (2, b"<".to_vec()),
            (3, b"a".to_vec()),
            (4, b"b".to_vec()),
            (5, b">,<".to_vec()),
            (6, b">".to_vec()),
            (7, b"<a".to_vec()),
            (8, b"<b".to_vec()),
            (9, b"a>".to_vec()),
            (10, b"b>".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t LEFT ::= @token(998);
                t RIGHT ::= @token(999);
                nt document ::= "<" LEFT ">,<" RIGHT ">";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child ::= "a" | "b";
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                nt child ::= "a" | "b";
                nt document ::= "<" child ">,<" child ">";
            "#,
            &vocab,
        )
        .unwrap();
        let composed = parent
            .compose_linked_children_for_test(&[("LEFT", &child), ("RIGHT", &child)], &vocab)
            .unwrap();

        assert_constraints_equivalent_on_reachable_prefixes(
            &composed,
            &monolithic,
            &vocab,
            5,
        );
    }

    #[test]
    fn three_distinct_children_match_monolithic_across_one_fused_token() {
        let vocab = Vocab::new(vec![
            (0, b"[a|b|c]".to_vec()),
            (1, b"[a|".to_vec()),
            (2, b"b|".to_vec()),
            (3, b"c]".to_vec()),
            (4, b"[".to_vec()),
            (5, b"a".to_vec()),
            (6, b"|".to_vec()),
            (7, b"b".to_vec()),
            (8, b"c".to_vec()),
            (9, b"]".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t FIRST ::= @token(997);
                t SECOND ::= @token(998);
                t THIRD ::= @token(999);
                nt document ::= "[" FIRST "|" SECOND "|" THIRD "]";
            "#,
            &vocab,
        )
        .unwrap();
        let first = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child ::= "a";
            "#,
            &vocab,
        )
        .unwrap();
        let second = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child ::= "b";
            "#,
            &vocab,
        )
        .unwrap();
        let third = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child ::= "c";
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                nt first ::= "a";
                nt second ::= "b";
                nt third ::= "c";
                nt document ::= "[" first "|" second "|" third "]";
            "#,
            &vocab,
        )
        .unwrap();
        let composed = compose_constraints_owned_parent_segmented(
            parent.clone(),
            &[
                CompiledSubgrammarInput {
                    placeholder_terminal: terminal(&parent, "FIRST"),
                    additional_placeholder_terminals: &[],
                    constraint: &first,
                },
                CompiledSubgrammarInput {
                    placeholder_terminal: terminal(&parent, "SECOND"),
                    additional_placeholder_terminals: &[],
                    constraint: &second,
                },
                CompiledSubgrammarInput {
                    placeholder_terminal: terminal(&parent, "THIRD"),
                    additional_placeholder_terminals: &[],
                    constraint: &third,
                },
            ],
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .unwrap()
        .constraint;

        let mut composed_bytes = composed.start();
        composed_bytes.commit_bytes(b"[a|b|c]").unwrap();
        assert!(composed_bytes.is_accepting());
        let mut monolithic_bytes = monolithic.start();
        monolithic_bytes.commit_bytes(b"[a|b|c]").unwrap();
        assert!(monolithic_bytes.is_accepting());

        assert_constraints_equivalent_on_reachable_prefixes(
            &composed,
            &monolithic,
            &vocab,
            7,
        );
        let mut fused = composed.start();
        fused.commit_token(0).unwrap();
        assert!(fused.is_accepting());
    }

    #[test]
    fn composition_preserves_out_of_vocab_end_tokens() {
        const PLACEHOLDER_TOKEN: u32 = 999;
        const END_TOKEN: u32 = 1000;
        let vocab = Vocab::new(vec![
            (0, b"Xa!".to_vec()),
            (1, b"X".to_vec()),
            (2, b"a".to_vec()),
            (3, b"!".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar_with_end_tokens(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "X" SUB "!";
            "#,
            &vocab,
            &[END_TOKEN],
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child ::= "a";
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar_with_end_tokens(
            r#"
                start document;
                nt child ::= "a";
                nt document ::= "X" child "!";
            "#,
            &vocab,
            &[END_TOKEN],
        )
        .unwrap();
        let composed = parent
            .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
            .unwrap();
        assert_eq!(composed.table.embedded_end_token_ids(), vec![END_TOKEN]);

        for content in [vec![0], vec![1, 2, 3]] {
            let mut actual = composed.start();
            let mut expected = monolithic.start();
            assert!(!token_allowed(&actual.mask(), END_TOKEN));
            assert!(!token_allowed(&actual.mask(), PLACEHOLDER_TOKEN));
            for token in content {
                assert_eq!(actual.mask(), expected.mask());
                actual.commit_token(token).unwrap();
                expected.commit_token(token).unwrap();
            }
            assert_eq!(actual.mask(), expected.mask());
            assert!(token_allowed(&actual.mask(), END_TOKEN));
            assert!(!token_allowed(&actual.mask(), PLACEHOLDER_TOKEN));
            assert_eq!(actual.forced(), vec![END_TOKEN]);
            actual.commit_token(END_TOKEN).unwrap();
            expected.commit_token(END_TOKEN).unwrap();
            assert!(actual.is_accepting());
            assert!(expected.is_accepting());
        }

        let loaded = Constraint::load(&composed.save()).unwrap();
        assert_eq!(loaded.table.embedded_end_token_ids(), vec![END_TOKEN]);
        let mut loaded_state = loaded.start();
        loaded_state.commit_token(0).unwrap();
        assert_eq!(loaded_state.forced(), vec![END_TOKEN]);
        loaded_state.commit_token(END_TOKEN).unwrap();
        assert!(loaded_state.is_accepting());
    }

    #[test]

    fn composition_compiles_named_special_continuation_into_static_mask() {
        const SPECIAL_TOKEN: u32 = 1000;
        let vocab = Vocab::new(vec![
            (0, b"Xa".to_vec()),
            (1, b"X".to_vec()),
            (2, b"a".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                t DONE ::= @token(1000);
                nt document ::= "X" SUB DONE;
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child ::= "a";
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                t DONE ::= @token(1000);
                nt document ::= "X" "a" DONE;
            "#,
            &vocab,
        )
        .unwrap();
        let composed = parent
            .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
            .unwrap();

        assert!(composed.table.control_terminals.is_empty());
        for content in [vec![0], vec![1, 2]] {
            let mut actual = composed.start();
            let mut expected = monolithic.start();
            for token in content {
                assert_eq!(actual.mask(), expected.mask());
                actual.commit_token(token).unwrap();
                expected.commit_token(token).unwrap();
            }
            assert_eq!(actual.mask(), expected.mask());
            assert_eq!(actual.forced(), vec![SPECIAL_TOKEN]);
            actual.commit_token(SPECIAL_TOKEN).unwrap();
            expected.commit_token(SPECIAL_TOKEN).unwrap();
            assert!(actual.is_accepting());
            assert!(expected.is_accepting());
        }
    }

    #[test]
    fn nullable_child_to_named_special_is_masked() {
        const SPECIAL_TOKEN: u32 = 1000;
        let vocab = Vocab::new(vec![(0, b"a".to_vec())]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                t DONE ::= @token(1000);
                nt document ::= SUB DONE;
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt item ::= "a";
                nt child ::= item?;
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                t DONE ::= @token(1000);
                nt item ::= "a";
                nt document ::= item? DONE;
            "#,
            &vocab,
        )
        .unwrap();
        // Nullable bound children are outside the signed-transfer static
        // linker's supported class (unbounded silent Entry/Return episodes);
        // static links decline them loudly, so this differential runs on the
        // exact dynamic backend.
        let composed = parent
            .compose_linked_children_for_test_dynamic(&[("SUB", &child)], &vocab)
            .unwrap();

        assert!(composed.table.control_terminals.is_empty());
        for sequence in [vec![SPECIAL_TOKEN], vec![0, SPECIAL_TOKEN]] {
            let mut actual = composed.start();
            let mut expected = monolithic.start();
            for token in sequence {
                assert_eq!(actual.mask(), expected.mask());
                actual.commit_token(token).unwrap();
                expected.commit_token(token).unwrap();
            }
            assert!(actual.is_accepting());
            assert!(expected.is_accepting());
        }
    }

    #[test]

    fn composition_rejects_placeholder_token_id_reused_by_live_end_token() {
        const SHARED_TOKEN: u32 = 999;
        let vocab = Vocab::new(vec![
            (0, b"Xa!".to_vec()),
            (1, b"X".to_vec()),
            (2, b"a".to_vec()),
            (3, b"!".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar_with_end_tokens(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "X" SUB "!";
            "#,
            &vocab,
            &[SHARED_TOKEN],
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child ::= "a";
            "#,
            &vocab,
        )
        .unwrap();

        let error = parent
            .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
            .expect_err("a placeholder sentinel ID cannot remain live as an end token");
        assert!(
            error
                .to_string()
                .contains("also configured as a grammar-level end token")
        );
        assert!(error.to_string().contains("unique sentinel token ID"));
    }

    #[test]
    fn child_end_token_survives_composition_and_blocks_nested_sentinel_reuse() {
        const END_TOKEN: u32 = 100;
        let vocab = Vocab::new(vec![
            (0, b"X".to_vec()),
            (1, b"a".to_vec()),
            (2, b"!".to_vec()),
            (3, b"Xa".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "X" SUB "!";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar_with_end_tokens(
            r#"
                start child;
                nt child ::= "a";
            "#,
            &vocab,
            &[END_TOKEN],
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                t END ::= @token(100);
                nt document ::= "X" "a" END "!";
            "#,
            &vocab,
        )
        .unwrap();
        let composed = parent
            .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
            .unwrap();
        assert_eq!(composed.table.embedded_end_token_ids(), vec![END_TOKEN]);
        let loaded = Constraint::load(&composed.save()).unwrap();
        assert_eq!(loaded.table.embedded_end_token_ids(), vec![END_TOKEN]);

        for sequence in [vec![3, END_TOKEN, 2], vec![0, 1, END_TOKEN, 2]] {
            let mut actual = loaded.start();
            let mut expected = monolithic.start();
            for token in sequence {
                assert_eq!(actual.mask(), expected.mask());
                actual.commit_token(token).unwrap();
                expected.commit_token(token).unwrap();
            }
            assert!(actual.is_accepting());
            assert!(expected.is_accepting());
        }

        let outer = Constraint::from_glrm_grammar(
            r#"
                start outer;
                t INNER ::= @token(100);
                nt outer ::= INNER;
            "#,
            &vocab,
        )
        .unwrap();
        let error = outer
            .compose_linked_children_for_test(&[("INNER", &loaded)], &vocab)
            .expect_err("nested placeholder must not reuse the child's end-token ID");
        assert!(
            error
                .to_string()
                .contains("also configured as a grammar-level end token")
        );
    }

    #[test]
    fn substitution_can_make_the_composed_start_nullable_for_later_embedding() {
        let vocab = Vocab::new(vec![
            (0, b"X!".to_vec()),
            (1, b"Xa!".to_vec()),
            (2, b"X".to_vec()),
            (3, b"a".to_vec()),
            (4, b"!".to_vec()),
        ]);
        let nullable_parent = Constraint::from_glrm_grammar(
            r#"
                start middle;
                t CHILD ::= @token(998);
                nt middle ::= CHILD;
            "#,
            &vocab,
        )
        .unwrap();
        assert!(!nullable_parent.table.embedded_start_nullable());
        let nullable_child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt item ::= "a";
                nt child ::= item?;
            "#,
            &vocab,
        )
        .unwrap();
        // Nullable bound children are outside the signed-transfer static
        // linker's supported class; compose the nullable middle dynamically.
        // Table nullability metadata is backend-independent.
        let middle = nullable_parent
            .compose_linked_children_for_test_dynamic(&[("CHILD", &nullable_child)], &vocab)
            .unwrap();
        assert!(middle.table.embedded_start_nullable());
        let middle = Constraint::load(&middle.save()).unwrap();
        assert!(middle.table.embedded_start_nullable());

        let outer_parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t MIDDLE ::= @token(999);
                nt document ::= "X" MIDDLE "!";
            "#,
            &vocab,
        )
        .unwrap();
        let composed = outer_parent
            .compose_linked_children_for_test_dynamic(&[("MIDDLE", &middle)], &vocab)
            .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                nt item ::= "a";
                nt document ::= "X" item? "!";
            "#,
            &vocab,
        )
        .unwrap();

        assert_constraints_equivalent_on_reachable_prefixes(
            &composed,
            &monolithic,
            &vocab,
            4,
        );
    }

    #[test]
    fn composed_constraint_matches_monolithic_when_tokens_do_not_cross_boundaries() {
        let vocab = byte_vocab();
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "<" SUB ">" SUB "!";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child ::= "a" "b";
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                g inner ::= {
                    start child;
                    nt child ::= "a" "b";
                };
                nt document ::= "<" inner ">" inner "!";
            "#,
            &vocab,
        )
        .unwrap();
        let composed = compose_constraints_owned_parent_segmented(
            parent.clone(),
            &[CompiledSubgrammarInput {
                placeholder_terminal: terminal(&parent, "SUB"),
                additional_placeholder_terminals: &[],
                constraint: &child,
            }],
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .unwrap()
        .constraint;

        let valid = b"<ab>ab!";
        let mut expected = monolithic.start();
        let mut actual = composed.start();
        for (offset, &byte) in valid.iter().enumerate() {
            assert_eq!(
                actual.mask(),
                expected.mask(),
                "mask mismatch before offset {offset}, byte {byte:?}",
            );
            actual.commit_bytes(&[byte]).unwrap();
            expected.commit_bytes(&[byte]).unwrap();
        }
        assert_eq!(actual.mask(), expected.mask(), "final mask mismatch");
        assert_eq!(actual.is_accepting(), expected.is_accepting());
        assert!(actual.is_accepting());

        for bytes in [
            b"<ab>ab!".as_slice(),
            b"<a>ab!".as_slice(),
            b"<ab>a!".as_slice(),
            b"<ab>ab".as_slice(),
            b"ab>ab!".as_slice(),
            b"<ab><ab>!".as_slice(),
        ] {
            let mut expected = monolithic.start();
            let expected_accepts = expected.commit_bytes(bytes).is_ok() && expected.is_accepting();
            let mut actual = composed.start();
            let actual_accepts = actual.commit_bytes(bytes).is_ok() && actual.is_accepting();
            assert_eq!(actual_accepts, expected_accepts, "language mismatch for {bytes:?}");
        }
        assert_accepts(&composed, valid);
        assert_rejects(&composed, b"<ab>a!");
    }

    #[test]
    fn composed_constraint_reuses_one_child_across_distinct_placeholders() {
        let vocab = byte_vocab();
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t LEFT ::= @token(998);
                t RIGHT ::= @token(999);
                nt document ::= "<" LEFT ">" RIGHT "!";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child ::= "a" "b";
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                g inner ::= {
                    start child;
                    nt child ::= "a" "b";
                };
                nt document ::= "<" inner ">" inner "!";
            "#,
            &vocab,
        )
        .unwrap();
        let left = terminal(&parent, "LEFT");
        let right = terminal(&parent, "RIGHT");
        let composed = compose_constraints_owned_parent_segmented(
            parent.clone(),
            &[CompiledSubgrammarInput {
                placeholder_terminal: left,
                additional_placeholder_terminals: &[right],
                constraint: &child,
            }],
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .unwrap()
        .constraint;
        let segmented = compose_constraints_owned_parent_segmented(
            parent.clone(),
            &[CompiledSubgrammarInput {
                placeholder_terminal: left,
                additional_placeholder_terminals: &[right],
                constraint: &child,
            }],
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .unwrap()
        .constraint;

        let overlay = segmented
            .static_dynamic_overlay
            .as_ref()
            .expect("segmented composition should retain component wrapper metadata");
        assert_eq!(overlay.segmented_parser_links.len(), 2);
        assert_eq!(
            overlay
                .segmented_parser_links
                .iter()
                .map(|link| link.slot_terminal)
                .collect::<Vec<_>>(),
            vec![left, right],
        );
        for link in &overlay.segmented_parser_links {
            assert_eq!(link.parent_component, 0);
            assert_eq!(link.child_component, 1);
            assert_eq!(link.child_start, 0);
            assert!(!link.child_start_nullable);
            assert!(matches!(link.return_pop, 1 | 2));
        }

        let valid = b"<ab>ab!";
        let mut expected = monolithic.start();
        let mut actual = composed.start();
        for (offset, &byte) in valid.iter().enumerate() {
            assert_eq!(
                actual.mask(),
                expected.mask(),
                "mask mismatch before offset {offset}, byte {byte:?}",
            );
            actual.commit_bytes(&[byte]).unwrap();
            expected.commit_bytes(&[byte]).unwrap();
        }
        assert_eq!(actual.mask(), expected.mask(), "final mask mismatch");
        assert_eq!(actual.is_accepting(), expected.is_accepting());
        assert!(actual.is_accepting());
        assert_accepts(&composed, valid);
        assert_rejects(&composed, b"<ab>a!");

        assert!(
            segmented.uses_compact_segmented_parser_runtime(),
            "source-built dynamic composition should use compact parser coordinates",
        );
        let loaded_segmented = Constraint::load(&segmented.save()).unwrap();
        assert!(
            loaded_segmented.uses_compact_segmented_parser_runtime(),
            "v24 dynamic composition should restore compact parser coordinates",
        );
        for constraint in [&segmented, &loaded_segmented] {
            let mut expected = monolithic.start();
            let mut actual = constraint.start();
            for (offset, &byte) in valid.iter().enumerate() {
                assert_eq!(
                    actual.mask(),
                    expected.mask(),
                    "segmented dynamic mask mismatch before offset {offset}, byte {byte:?}",
                );
                actual.commit_bytes(&[byte]).unwrap();
                expected.commit_bytes(&[byte]).unwrap();
            }
            assert_eq!(actual.mask(), expected.mask(), "segmented dynamic final mask mismatch");
            assert_eq!(actual.is_accepting(), expected.is_accepting());
            assert!(actual.is_accepting());
        }
    }

    #[test]
    fn hybrid_boundary_policy_is_per_component_and_roundtrips() {
        // This fixture deliberately executes DynamicDirect shards. Other
        // tests temporarily arm the process-wide strict-static trap, so its
        // readers must share the writers' lock as well.
        let _env_lock = crate::TEST_ENV_LOCK.lock().unwrap();
        let vocab = Vocab::new(vec![
            (0, b"x".to_vec()),
            (1, b"y".to_vec()),
            (2, b"z".to_vec()),
            (3, b"xy".to_vec()),
            (4, b"yz".to_vec()),
            (5, b"xyz".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "x" SUB "z";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child ::= "y";
            "#,
            &vocab,
        )
        .unwrap();
        let reference = Constraint::from_glrm_grammar(
            r#"
                start document;
                nt document ::= "x" "y" "z";
            "#,
            &vocab,
        )
        .unwrap();
        let slot = terminal(&parent, "SUB");
        let child_input = [CompiledSubgrammarInput {
            placeholder_terminal: slot,
            additional_placeholder_terminals: &[],
            constraint: &child,
        }];

        for static_component in [0usize, 1usize] {
            let mut static_components = BitSet::new(2);
            static_components.set(static_component);
            let hybrid = compose_constraints_owned_parent_segmented_hybrid(
                parent.clone(),
                &child_input,
                &vocab,
                &static_components,
            )
            .unwrap()
            .constraint;

            let assert_policy = |constraint: &Constraint| {
                let overlay = constraint.static_dynamic_overlay.as_ref().unwrap();
                assert!(overlay.segmented_boundary_parser.is_none());
                assert!(overlay.segmented_boundary_terminal_trie.is_none());
                assert_eq!(overlay.segmented_parser_components.len(), 2);
                for (index, component) in overlay.segmented_parser_components.iter().enumerate() {
                    let shard = component
                        .boundary
                        .as_ref()
                        .expect("each component in this fixture has crossing tokens");
                    if index == static_component {
                        assert!(matches!(
                            shard.backend,
                            crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_)
                        ));
                    } else {
                        assert!(matches!(
                            shard.backend,
                            crate::runtime::SegmentedBoundaryShardBackend::DynamicDirect
                        ));
                    }
                }
            };
            assert_policy(&hybrid);
            let loaded = Constraint::load(&hybrid.save()).unwrap();
            assert_policy(&loaded);

            for constraint in [&hybrid, &loaded] {
                for tokens in [&[5][..], &[0, 4][..], &[3, 2][..], &[0, 1, 2][..]] {
                    let mut actual = constraint.start();
                    let mut expected = reference.start();
                    for &token in tokens {
                        assert_eq!(
                            actual.mask(),
                            expected.mask(),
                            "hybrid component {static_component} mismatch before {tokens:?} token {token}",
                        );
                        actual.commit_token(token).unwrap();
                        expected.commit_token(token).unwrap();
                    }
                    assert_eq!(actual.is_accepting(), expected.is_accepting(), "{tokens:?}");
                    assert!(actual.is_accepting(), "{tokens:?}");
                }
            }
        }
    }

    #[test]
    fn segmented_parser_links_preserve_nullable_child_local_coordinates() {
        let vocab = byte_vocab();
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "<" SUB ">";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt item ::= "a";
                nt child ::= item?;
            "#,
            &vocab,
        )
        .unwrap();
        let slot = terminal(&parent, "SUB");
        let composed = compose_constraints_owned_parent_segmented(
            parent,
            &[CompiledSubgrammarInput {
                placeholder_terminal: slot,
                additional_placeholder_terminals: &[],
                constraint: &child,
            }],
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .unwrap()
        .constraint;
        let overlay = composed.static_dynamic_overlay.as_ref().unwrap();
        let [link] = overlay.segmented_parser_links.as_slice() else {
            panic!("expected one local parser link");
        };
        assert_eq!(link.parent_component, 0);
        assert_eq!(link.slot_terminal, slot);
        assert_eq!(link.child_component, 1);
        assert_eq!(link.child_start, 0);
        assert!(link.child_start_nullable);
        let rules = child.retained_table_rules().unwrap();
        assert_eq!(
            link.return_pop,
            crate::compiler::glr::table::subgrammar_child_return_pop(&child.table, rules).unwrap(),
        );
        let layout = composed
            .recursive_parser_layout()
            .unwrap()
            .expect("segmented runtime must derive the recursive parser coordinate");
        assert_eq!(layout.links.len(), 1);
        assert_eq!(layout.links[0].parent_component, 0);
        assert_eq!(layout.links[0].child_component, 1);
        assert_eq!(layout.links[0].child_start, 0);
        assert!(layout.links[0].child_start_nullable);
        assert_eq!(layout.leaf_state_offsets[0], 0);
        assert_eq!(
            layout.leaf_state_offsets[1],
            layout.leaves[0].state_count,
            "child root must be expressed in the contiguous recursive leaf coordinate",
        );
    }

    #[test]
    fn composed_constraint_matches_monolithic_for_fused_entry_and_exit_tokens() {
        let vocab = Vocab::new(vec![
            (0, b"<a".to_vec()),
            (1, b"b>".to_vec()),
            (2, b"ab".to_vec()),
            (3, b"!".to_vec()),
            (4, b"<".to_vec()),
            (5, b">".to_vec()),
            (6, b"a".to_vec()),
            (7, b"b".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "<" SUB ">" "!";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child ::= "a" "b";
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                g inner ::= {
                    start child;
                    nt child ::= "a" "b";
                };
                nt document ::= "<" inner ">" "!";
            "#,
            &vocab,
        )
        .unwrap();
        let composed = compose_constraints_owned_parent_segmented(
            parent.clone(),
            &[CompiledSubgrammarInput {
                placeholder_terminal: terminal(&parent, "SUB"),
                additional_placeholder_terminals: &[],
                constraint: &child,
            }],
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .unwrap()
        .constraint;

        let mut expected = monolithic.start();
        let mut actual = composed.start();
        for token in [0, 1, 3] {
            assert_eq!(actual.mask(), expected.mask(), "mask mismatch before token {token}");
            actual.commit_token(token).unwrap();
            expected.commit_token(token).unwrap();
        }
        assert_eq!(actual.is_accepting(), expected.is_accepting());
        assert!(actual.is_accepting());
    }

    #[test]
    fn composed_constraint_matches_monolithic_for_child_outer_child_token() {
        let vocab = Vocab::new(vec![
            (0, b"<a".to_vec()),
            (1, b"b>,<c".to_vec()),
            (2, b"d>".to_vec()),
            (3, b"!".to_vec()),
            (4, b"<".to_vec()),
            (5, b">,<".to_vec()),
            (6, b">".to_vec()),
            (7, b"a".to_vec()),
            (8, b"b".to_vec()),
            (9, b"c".to_vec()),
            (10, b"d".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t LEFT ::= @token(998);
                t RIGHT ::= @token(999);
                nt document ::= "<" LEFT ">,<" RIGHT ">" "!";
            "#,
            &vocab,
        )
        .unwrap();
        let left = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child ::= "a" "b";
            "#,
            &vocab,
        )
        .unwrap();
        let right = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child ::= "c" "d";
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                g left ::= {
                    start child;
                    nt child ::= "a" "b";
                };
                g right ::= {
                    start child;
                    nt child ::= "c" "d";
                };
                nt document ::= "<" left ">,<" right ">" "!";
            "#,
            &vocab,
        )
        .unwrap();
        let composed = compose_constraints_owned_parent_segmented(
            parent.clone(),
            &[
                CompiledSubgrammarInput {
                    placeholder_terminal: terminal(&parent, "LEFT"),
                    additional_placeholder_terminals: &[],
                    constraint: &left,
                },
                CompiledSubgrammarInput {
                    placeholder_terminal: terminal(&parent, "RIGHT"),
                    additional_placeholder_terminals: &[],
                    constraint: &right,
                },
            ],
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .unwrap()
        .constraint;

        let mut expected_separate = monolithic.start();
        let mut actual_separate = composed.start();
        for token in [4, 7, 8, 5, 9, 10, 6, 3] {
            assert_eq!(
                actual_separate.mask(),
                expected_separate.mask(),
                "separate-token mask mismatch before token {token}",
            );
            actual_separate.commit_token(token).unwrap();
            expected_separate.commit_token(token).unwrap();
        }
        assert!(actual_separate.is_accepting());
        assert!(expected_separate.is_accepting());

        let mut expected = monolithic.start();
        let mut actual = composed.start();
        for token in [0, 1, 2, 3] {
            assert_eq!(actual.mask(), expected.mask(), "mask mismatch before token {token}");
            actual.commit_token(token).unwrap();
            expected.commit_token(token).unwrap();
        }
        assert_eq!(actual.is_accepting(), expected.is_accepting());
        assert!(actual.is_accepting());
    }

    #[test]
    fn same_child_divergent_continuations_match_dynamic_and_monolithic() {
        // One placeholder called from two parent states with divergent
        // continuations (L SUB x | R SUB y). The fused exit tokens ax/ay
        // discriminate the return linkage: after L only ax (not ay) may
        // complete the child, and after R only ay (not ax). A shared
        // child-start row that resolves the return to a single call site
        // under-admits the other site's fused token here.
        let vocab = Vocab::new(vec![
            (0, b"ax".to_vec()),
            (1, b"ay".to_vec()),
            (2, b"L".to_vec()),
            (3, b"R".to_vec()),
            (4, b"a".to_vec()),
            (5, b"x".to_vec()),
            (6, b"y".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "L" SUB "x" | "R" SUB "y";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child ::= "a";
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                nt inner ::= "a";
                nt document ::= "L" inner "x" | "R" inner "y";
            "#,
            &vocab,
        )
        .unwrap();
        let sub = terminal(&parent, "SUB");
        assert!(
            (0..parent.table.num_states)
                .filter(|&state| parent.table.action(state, sub).is_some())
                .count()
                >= 2,
            "parent must shift SUB from two divergent call-site states",
        );
        let inputs = [CompiledSubgrammarInput {
            placeholder_terminal: sub,
            additional_placeholder_terminals: &[],
            constraint: &child,
        }];
        let static_composed = compose_constraints_owned_parent_segmented(
            parent.clone(),
            &inputs,
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .unwrap()
        .constraint;
        let dynamic_composed = compose_constraints_owned_parent_segmented(
            parent.clone(),
            &inputs,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .unwrap()
        .constraint;

        for sequence in [vec![2, 0], vec![3, 1], vec![2, 4, 5], vec![3, 4, 6]] {
            let mut mono = monolithic.start();
            let mut static_state = static_composed.start();
            let mut dynamic_state = dynamic_composed.start();
            for token in sequence.clone() {
                assert_eq!(
                    static_state.mask(),
                    mono.mask(),
                    "static mask mismatch before token {token} in {sequence:?}",
                );
                assert_eq!(
                    dynamic_state.mask(),
                    mono.mask(),
                    "dynamic mask mismatch before token {token} in {sequence:?}",
                );
                static_state.commit_token(token).unwrap();
                dynamic_state.commit_token(token).unwrap();
                mono.commit_token(token).unwrap();
            }
            assert_eq!(static_state.is_accepting(), mono.is_accepting());
            assert!(static_state.is_accepting());
            assert_eq!(dynamic_state.is_accepting(), mono.is_accepting());
        }
        for sequence in [vec![2, 1], vec![3, 0]] {
            for (name, constraint) in [
                ("monolithic", &monolithic),
                ("static", &static_composed),
                ("dynamic", &dynamic_composed),
            ] {
                let mut state = constraint.start();
                let mut committed = true;
                for token in &sequence {
                    if state.commit_token(*token).is_err() {
                        committed = false;
                        break;
                    }
                }
                assert!(
                    !(committed && state.is_accepting()),
                    "{name} wrongly accepts cross-site {sequence:?}",
                );
            }
        }
    }

    #[test]
    fn nullable_child_composition_matches_monolithic_across_empty_and_nonempty_paths() {
        let vocab = Vocab::new(vec![
            (0, b"X!".to_vec()),
            (1, b"Xa!".to_vec()),
            (2, b"X".to_vec()),
            (3, b"a".to_vec()),
            (4, b"!".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "X" SUB "!";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt item ::= "a";
                nt child ::= item?;
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                nt item ::= "a";
                nt document ::= "X" item? "!";
            "#,
            &vocab,
        )
        .unwrap();
        assert!(child.table.embedded_start_nullable());
        let loaded_child = Constraint::load(&child.save()).unwrap();
        assert!(loaded_child.table.embedded_start_nullable());

        // Nullable bound children are outside the signed-transfer static
        // linker's supported class (unbounded silent Entry/Return episodes);
        // static links decline them loudly, so this X! differential runs on
        // the exact dynamic backend.
        for child in [&child, &loaded_child] {
            let composed = parent
                .compose_linked_children_for_test_dynamic(&[("SUB", child)], &vocab)
                .unwrap();
            for sequence in [vec![0], vec![1], vec![2, 4], vec![2, 3, 4]] {
                let mut expected = monolithic.start();
                let mut actual = composed.start();
                for token in sequence {
                    assert_eq!(
                        actual.mask(),
                        expected.mask(),
                        "mask mismatch before token {token}",
                    );
                    actual.commit_token(token).unwrap();
                    expected.commit_token(token).unwrap();
                }
                assert_eq!(actual.is_accepting(), expected.is_accepting());
                assert!(actual.is_accepting());
            }
        }
    }

    #[test]
    fn contextual_structural_sharing_remains_recomposable_when_children_share_whole_start_alternative() {
        let vocab = byte_vocab();

        let expr = Constraint::from_glrm_grammar(
            r#"
                start expr;
                nt expr ::= "x";
            "#,
            &vocab,
        )
        .unwrap();

        let arg_a_parent = Constraint::from_glrm_grammar(
            r#"
                start args;
                t EXPR ::= @token(996);
                nt args ::= "{" "a" ":" EXPR "}" | EXPR;
            "#,
            &vocab,
        )
        .unwrap();
        let arg_a = arg_a_parent
            .compose_linked_children_for_test(&[("EXPR", &expr)], &vocab)
            .unwrap();
        let arg_a_dynamic = arg_a_parent
            .compose_linked_children_for_test_dynamic(&[("EXPR", &expr)], &vocab)
            .unwrap();

        let arg_b_parent = Constraint::from_glrm_grammar(
            r#"
                start args;
                t EXPR ::= @token(996);
                nt args ::= "{" "b" ":" EXPR "}" | EXPR;
            "#,
            &vocab,
        )
        .unwrap();
        let arg_b = arg_b_parent
            .compose_linked_children_for_test(&[("EXPR", &expr)], &vocab)
            .unwrap();
        let arg_b_dynamic = arg_b_parent
            .compose_linked_children_for_test_dynamic(&[("EXPR", &expr)], &vocab)
            .unwrap();

        let dispatch_parent = Constraint::from_glrm_grammar(
            r#"
                start call;
                t ARGA ::= @token(997);
                t ARGB ::= @token(998);
                nt call ::= "t" "." "ta" "(" ARGA ")"
                          | "t" "." "tb" "(" ARGB ")";
            "#,
            &vocab,
        )
        .unwrap();
        let dispatch = dispatch_parent
            .compose_linked_children_for_test(&[("ARGA", &arg_a), ("ARGB", &arg_b)], &vocab)
            .unwrap();
        let dispatch_dynamic = dispatch_parent
            .compose_linked_children_for_test_dynamic(
                &[("ARGA", &arg_a_dynamic), ("ARGB", &arg_b_dynamic)],
                &vocab,
            )
            .unwrap();

        // The two argument children both expose the same nested `expr` as a
        // whole-start alternative. Contextual structural sharing can prove an
        // interior quotient for that overlap. The resulting table must still
        // remain a valid child for a *later* composition level.
        let outer_parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t CALL ::= @token(999);
                nt document ::= "X" CALL "!";
            "#,
            &vocab,
        )
        .unwrap();
        let composed = outer_parent
            .clone()
            .compose_linked_children_for_test(&[("CALL", &dispatch)], &vocab)
            .unwrap();
        let composed_dynamic = outer_parent
            .clone()
            .compose_linked_children_for_test_dynamic(&[("CALL", &dispatch_dynamic)], &vocab)
            .unwrap();
        let loaded_dispatch = Constraint::load(&dispatch.save()).unwrap();
        let composed_from_loaded = outer_parent
            .compose_linked_children_for_test(&[("CALL", &loaded_dispatch)], &vocab)
            .unwrap();

        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                nt expr ::= "x";
                nt arg_a ::= "{" "a" ":" expr "}" | expr;
                nt arg_b ::= "{" "b" ":" expr "}" | expr;
                nt call ::= "t" "." "ta" "(" arg_a ")"
                          | "t" "." "tb" "(" arg_b ")";
                nt document ::= "X" call "!";
            "#,
            &vocab,
        )
        .unwrap();

        for bytes in [b"Xt.ta(x)!".as_slice(), b"Xt.tb(x)!", b"Xt.ta({a:x})!", b"Xt.tb({b:x})!"] {
            let mut expected = monolithic.start();
            let mut actual = composed.start();
            let mut dynamic = composed_dynamic.start();
            let mut restored = composed_from_loaded.start();
            for &byte in bytes {
                assert_eq!(actual.mask(), expected.mask(), "mask mismatch before byte {byte:?}");
                assert_eq!(dynamic.mask(), expected.mask(), "dynamic mask mismatch before byte {byte:?}");
                assert_eq!(restored.mask(), expected.mask(), "loaded mask mismatch before byte {byte:?}");
                actual.commit_token(byte as u32).unwrap();
                dynamic.commit_token(byte as u32).unwrap();
                restored.commit_token(byte as u32).unwrap();
                expected.commit_token(byte as u32).unwrap();
            }
            assert!(actual.is_accepting());
            assert!(dynamic.is_accepting());
            assert!(restored.is_accepting());
            assert!(expected.is_accepting());
        }

        // Structural sharing is a table-size optimization only. With sharing at
        // its default (no env opt-out), both explicit segmented backends must
        // still agree with the fully source-inlined grammar on masks *and*
        // acceptance, which requires the quotient to be skipped whenever an
        // explicit segmented boundary is requested (both build the same
        // global->local state_relations inverse).
        assert_constraints_equivalent_on_reachable_prefixes_labeled(
            &composed,
            &monolithic,
            &vocab,
            4,
            "contextual-static-vs-source-inline",
        );
        assert_constraints_equivalent_on_reachable_prefixes_labeled(
            &composed_dynamic,
            &monolithic,
            &vocab,
            4,
            "contextual-dynamic-vs-source-inline",
        );
    }

    #[test]
    fn nested_composition_matches_flat_monolithic_across_all_boundaries() {
        let vocab = Vocab::new(vec![
            (0, b"X[a]!".to_vec()),
            (1, b"X[".to_vec()),
            (2, b"a]".to_vec()),
            (3, b"!".to_vec()),
            (4, b"X".to_vec()),
            (5, b"[".to_vec()),
            (6, b"a".to_vec()),
            (7, b"]".to_vec()),
        ]);
        let leaf = Constraint::from_glrm_grammar(
            r#"
                start leaf;
                nt leaf ::= "a";
            "#,
            &vocab,
        )
        .unwrap();
        let middle_parent = Constraint::from_glrm_grammar(
            r#"
                start middle;
                t LEAF ::= @token(998);
                nt middle ::= "[" LEAF "]";
            "#,
            &vocab,
        )
        .unwrap();
        let middle = middle_parent
            .compose_linked_children_for_test(&[("LEAF", &leaf)], &vocab)
            .unwrap();
        let outer_parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t MIDDLE ::= @token(999);
                nt document ::= "X" MIDDLE "!";
            "#,
            &vocab,
        )
        .unwrap();
        let composed = outer_parent
            .compose_linked_children_for_test_dynamic(&[("MIDDLE", &middle)], &vocab)
            .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                nt document ::= "X" "[" "a" "]" "!";
            "#,
            &vocab,
        )
        .unwrap();

        for sequence in [vec![0], vec![1, 2, 3], vec![4, 5, 6, 7, 3]] {
            let mut expected = monolithic.start();
            let mut actual = composed.start();
            for token in sequence {
                assert_eq!(actual.mask(), expected.mask(), "mask mismatch before token {token}");
                actual.commit_token(token).unwrap();
                expected.commit_token(token).unwrap();
            }
            assert_eq!(actual.is_accepting(), expected.is_accepting());
            assert!(actual.is_accepting());
        }
    }

    #[test]
    fn nested_composition_preserves_nullable_start_through_save_load() {
        let vocab = Vocab::new(vec![
            (0, b"X!".to_vec()),
            (1, b"Xa!".to_vec()),
            (2, b"X".to_vec()),
            (3, b"a".to_vec()),
            (4, b"!".to_vec()),
        ]);
        let leaf = Constraint::from_glrm_grammar(
            r#"
                start leaf;
                nt leaf ::= "a";
            "#,
            &vocab,
        )
        .unwrap();
        let middle_parent = Constraint::from_glrm_grammar(
            r#"
                start middle;
                t LEAF ::= @token(998);
                nt middle ::= LEAF?;
            "#,
            &vocab,
        )
        .unwrap();
        let middle = middle_parent
            .compose_linked_children_for_test(&[("LEAF", &leaf)], &vocab)
            .unwrap();
        assert!(middle.table.embedded_start_nullable());
        let middle = Constraint::load(&middle.save()).unwrap();
        assert!(middle.table.embedded_start_nullable());

        let outer_parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t MIDDLE ::= @token(999);
                nt document ::= "X" MIDDLE "!";
            "#,
            &vocab,
        )
        .unwrap();
        // Nested link: the middle component is itself a segmented
        // composition whose start is nullable (deferred class), so the outer
        // link uses the exact Dynamic boundary backend. Behavior must still
        // match the monolithic constraint exactly, including through
        // save/load.
        let nested_inputs = [CompiledSubgrammarInput {
            placeholder_terminal: terminal(&outer_parent, "MIDDLE"),
            additional_placeholder_terminals: &[],
            constraint: &middle,
        }];
        // Requesting static shards for a nullable nested link must decline
        // loudly (nullable linked subgrammars are deferred), never silently
        // succeed as dynamic.
        let declined = compose_constraints_owned_parent_segmented(
            outer_parent.clone(),
            &nested_inputs,
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        );
        assert!(
            declined.is_err(),
            "nullable nested static link must decline loudly, got success",
        );
        let composed = compose_constraints_owned_parent_segmented(
            outer_parent.clone(),
            &nested_inputs,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .unwrap()
        .constraint;
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                nt item ::= "a";
                nt document ::= "X" item? "!";
            "#,
            &vocab,
        )
        .unwrap();

        for sequence in [vec![0], vec![1], vec![2, 4], vec![2, 3, 4]] {
            let mut expected = monolithic.start();
            let mut actual = composed.start();
            for token in sequence {
                assert_eq!(actual.mask(), expected.mask(), "mask mismatch before token {token}");
                actual.commit_token(token).unwrap();
                expected.commit_token(token).unwrap();
            }
            assert_eq!(actual.is_accepting(), expected.is_accepting());
            assert!(actual.is_accepting());
        }
    }

    #[test]
    fn nested_static_link_matches_dynamic_through_public_compose() {
        if crate::isolate_environment_test(false) { return; }
        // Production nested static link (acyclic, effectively nonnullable,
        // depth 2) through the PUBLIC compose route: inner Dynamic reference
        // composition, outer StaticParserDwa link, mask-gated corpus
        // differential with the strict-static trap armed on the static side,
        // plus backend pins and a shard ablation so the test cannot pass
        // vacuously.
        struct TrapGuard(bool);
        impl TrapGuard {
            fn set() -> Self {
                unsafe {
                    std::env::set_var("GLRMASK_STRICT_STATIC_TRAP_DYNAMIC", "1");
                }
                Self(true)
            }
            fn clear(&mut self) {
                if self.0 {
                    unsafe {
                        std::env::remove_var("GLRMASK_STRICT_STATIC_TRAP_DYNAMIC");
                    }
                    self.0 = false;
                }
            }
        }
        impl Drop for TrapGuard {
            fn drop(&mut self) {
                self.clear();
            }
        }
        fn admits(mask: &[u32], token: u32) -> bool {
            mask.get(token as usize / 32)
                .is_some_and(|word| word & (1u32 << (token % 32)) != 0)
        }
        fn is_subset(a: &[u32], b: &[u32]) -> bool {
            a.iter().zip(b.iter()).all(|(x, y)| x & !y == 0) && a.len() <= b.len()
        }

        let _env_lock = crate::TEST_ENV_LOCK.lock().unwrap();
        // Token ids: 0 L, 1 R, 2 x, 3 y, 4 m, 5 g, 6 Lm, 7 Rm, 8 mg, 9 gm,
        // 10 gx, 11 gy (same fixture as the lower-level nested pin).
        const L: u32 = 0;
        const R: u32 = 1;
        const X: u32 = 2;
        const Y: u32 = 3;
        const M: u32 = 4;
        const G: u32 = 5;
        const LM: u32 = 6;
        const RM: u32 = 7;
        const MG: u32 = 8;
        const GM: u32 = 9;
        const GX: u32 = 10;
        const GY: u32 = 11;
        let vocab = Vocab::new(vec![
            (0, b"L".to_vec()),
            (1, b"R".to_vec()),
            (2, b"x".to_vec()),
            (3, b"y".to_vec()),
            (4, b"m".to_vec()),
            (5, b"g".to_vec()),
            (6, b"Lm".to_vec()),
            (7, b"Rm".to_vec()),
            (8, b"mg".to_vec()),
            (9, b"gm".to_vec()),
            (10, b"gx".to_vec()),
            (11, b"gy".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "L" SUB SUB "x" | "R" SUB SUB "y";
            "#,
            &vocab,
        )
        .unwrap();
        let mid = Constraint::from_glrm_grammar(
            r#"
                start m;
                t SUB2 ::= @token(998);
                nt m ::= "m" SUB2;
            "#,
            &vocab,
        )
        .unwrap();
        let grandchild = Constraint::from_glrm_grammar(
            r#"
                start g;
                nt g ::= "g";
            "#,
            &vocab,
        )
        .unwrap();
        assert!(
            !mid.table.embedded_start_nullable(),
            "nested fixture mid must stay effectively nonnullable",
        );
        assert!(
            !grandchild.table.embedded_start_nullable(),
            "nested fixture grandchild must stay effectively nonnullable",
        );
        // Inner reference composition through the production dynamic route.
        let mid_inputs = [CompiledSubgrammarInput {
            placeholder_terminal: terminal(&mid, "SUB2"),
            additional_placeholder_terminals: &[],
            constraint: &grandchild,
        }];
        let mid_dyn = compose_constraints_owned_parent_segmented(
            mid.clone(),
            &mid_inputs,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("inner dynamic compose")
        .constraint;
        let outer_inputs = [CompiledSubgrammarInput {
            placeholder_terminal: terminal(&parent, "SUB"),
            additional_placeholder_terminals: &[],
            constraint: &mid_dyn,
        }];
        let outer_dyn = compose_constraints_owned_parent_segmented(
            parent.clone(),
            &outer_inputs,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("outer dynamic compose")
        .constraint;
        // The supported nested class links statically through the public
        // route (no loud decline).
        let outer_static = compose_constraints_owned_parent_segmented(
            parent.clone(),
            &outer_inputs,
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .expect("outer nested static compose")
        .constraint;
        // Backend pins: both top-level components hold StaticParser shards and
        // the nested middle overlay carries no shards of its own.
        {
            let overlay = outer_static
                .static_dynamic_overlay
                .as_ref()
                .expect("static overlay");
            assert_eq!(overlay.segmented_parser_components.len(), 2);
            for (index, component) in overlay.segmented_parser_components.iter().enumerate() {
                let shard = component.boundary.as_ref().unwrap_or_else(|| {
                    panic!("installed top component {index} must carry a static shard")
                });
                assert!(
                    matches!(
                        shard.backend,
                        crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_)
                    ),
                    "installed component {index} must be a StaticParser shard",
                );
            }
            let mid_inner = overlay.segmented_parser_components[1]
                .constraint
                .static_dynamic_overlay
                .as_ref()
                .expect("mid overlay");
            assert!(
                mid_inner
                    .segmented_parser_components
                    .iter()
                    .all(|component| component.boundary.is_none()),
                "mid overlay must carry no shards (outer block shard covers block-start crossings)",
            );
            assert!(
                mid_inner.segmented_boundary_shards.is_empty(),
                "mid overlay shard list must be cleared",
            );
        }
        // Mask-gated corpus from the dynamic reference (no trap): prefixes of
        // the complete L/R strings plus fused-spelling entry points.
        struct Node {
            path: Vec<u32>,
            mask: Vec<u32>,
        }
        let candidates: Vec<Vec<u32>> = vec![
            vec![],
            vec![L],
            vec![R],
            vec![L, M],
            vec![R, M],
            vec![LM],
            vec![RM],
            vec![L, M, G],
            vec![R, M, G],
            vec![LM, G],
            vec![RM, G],
            vec![L, M, G, M],
            vec![R, M, G, M],
            vec![L, M, G, M, G],
            vec![R, M, G, M, G],
            vec![L, M, G, M, G, X],
            vec![R, M, G, M, G, Y],
        ];
        let mut nodes = Vec::new();
        for path in &candidates {
            let mut st = outer_dyn.start();
            let mut ok = true;
            for &token in path {
                if !admits(&st.mask(), token) {
                    ok = false;
                    break;
                }
                if st.commit_token(token).is_err() {
                    ok = false;
                    break;
                }
            }
            if !ok {
                continue;
            }
            let mask = st.mask();
            nodes.push(Node { path: path.clone(), mask });
        }
        assert!(
            nodes.len() >= 10,
            "nested corpus must cover real positions, got {}",
            nodes.len()
        );
        // Trap-armed replay: every recorded path matches identically on the
        // public static composition. Any hidden dynamic fallback panics here.
        let mut trap = TrapGuard::set();
        let mut mismatches = 0usize;
        for node in &nodes {
            let mut st = outer_static.start();
            for &token in &node.path {
                st.commit_token(token).expect("static replay");
            }
            let actual = st.mask();
            if actual != node.mask {
                eprintln!(
                    "[glrmask/test][public_nested_static_mismatch] path={:?} dynamic={:?} static={:?}",
                    node.path, node.mask, actual,
                );
                mismatches += 1;
            }
        }
        trap.clear();
        assert_eq!(mismatches, 0, "nested static must match dynamic on every path");
        // Readable call-site discrimination through two nesting levels.
        for (prefix, fused, sibling, site) in [
            (vec![L, M, G, M], GX, GY, "L"),
            (vec![R, M, G, M], GY, GX, "R"),
        ] {
            let mut st = outer_static.start();
            for &token in &prefix {
                st.commit_token(token).unwrap();
            }
            let mask = st.mask();
            assert!(admits(&mask, fused), "nested admits {fused} after {site}mgm");
            assert!(!admits(&mask, sibling), "nested rejects {sibling} after {site}mgm");
        }
        {
            let mut st = outer_static.start();
            st.commit_token(L).unwrap();
            assert!(admits(&st.mask(), M), "nested admits m after L");
            assert!(admits(&st.mask(), MG), "nested admits mg after L");
            assert!(!admits(&st.mask(), RM), "nested rejects Rm after L");
        }
        // Ablation: clearing the installed top-level shards must lose
        // admissions (the shards are genuinely used, not redundant).
        let mut ablated = outer_static.clone();
        install_published_static_boundary_shards(
            ablated.static_dynamic_overlay.as_mut().expect("overlay"),
            Vec::new(),
        )
        .expect("clear shards");
        let mut strict = 0usize;
        for path in &candidates {
            let mut dyn_st = outer_dyn.start();
            let mut abl_st = ablated.start();
            for &token in path {
                if !admits(&dyn_st.mask(), token) {
                    break;
                }
                dyn_st.commit_token(token).unwrap();
                if !admits(&abl_st.mask(), token) {
                    strict += 1;
                    break;
                }
                abl_st.commit_token(token).unwrap();
                assert!(
                    is_subset(&abl_st.mask(), &dyn_st.mask()),
                    "shard-less mask must stay a subset of the reference",
                );
            }
        }
        assert!(strict >= 1, "ablation must lose at least one admission");
    }

    /// Mask-for-mask differential over byte-prefix scenarios: static and dynamic
    /// compositions must agree on masks, commit outcomes, and acceptance.
    /// Oracle-driven (no hardcoded masks).
    fn assert_composed_masks_equal(
        name: &str,
        static_comp: &Constraint,
        dynamic: &Constraint,
        prefixes: &[Vec<u8>],
    ) {
        for (scenario, prefix) in prefixes.iter().enumerate() {
            let mut st_static = static_comp.start();
            let mut st_dyn = dynamic.start();
            let mut consumed = 0usize;
            loop {
                assert_eq!(
                    st_static.mask(),
                    st_dyn.mask(),
                    "{name}: mask mismatch scenario={scenario} consumed={consumed}",
                );
                if consumed == prefix.len() {
                    break;
                }
                let r_static = st_static.commit_bytes(&prefix[consumed..consumed + 1]);
                let r_dyn = st_dyn.commit_bytes(&prefix[consumed..consumed + 1]);
                assert_eq!(
                    r_static.is_ok(),
                    r_dyn.is_ok(),
                    "{name}: commit divergence scenario={scenario} consumed={consumed}",
                );
                if r_static.is_err() {
                    break;
                }
                consumed += 1;
            }
            assert_eq!(
                st_static.is_accepting(),
                st_dyn.is_accepting(),
                "{name}: acceptance mismatch scenario={scenario}",
            );
        }
    }

    /// Token-level differential over fused/multibyte scenarios: commit whole
    /// vocabulary tokens (not byte slices) on both backends, comparing masks,
    /// commit outcomes, and acceptance. Oracle-driven (no hardcoded masks).
    /// Multibyte tokens that span a component boundary cross CALL/RETURN inside
    /// the boundary shards; byte-prefix differentials alone cannot see them.
    fn assert_composed_token_masks_equal(
        name: &str,
        static_comp: &Constraint,
        dynamic: &Constraint,
        paths: &[Vec<u32>],
    ) {
        for (scenario, path) in paths.iter().enumerate() {
            let mut st_static = static_comp.start();
            let mut st_dyn = dynamic.start();
            let mut step = 0usize;
            loop {
                assert_eq!(
                    st_static.mask(),
                    st_dyn.mask(),
                    "{name}: mask mismatch scenario={scenario} step={step}",
                );
                if step == path.len() {
                    break;
                }
                let r_static = st_static.commit_token(path[step]);
                let r_dyn = st_dyn.commit_token(path[step]);
                assert_eq!(
                    r_static.is_ok(),
                    r_dyn.is_ok(),
                    "{name}: commit divergence scenario={scenario} step={step} token={}",
                    path[step],
                );
                if r_static.is_err() {
                    break;
                }
                step += 1;
            }
            assert_eq!(
                st_static.is_accepting(),
                st_dyn.is_accepting(),
                "{name}: acceptance mismatch scenario={scenario}",
            );
        }
    }

    /// Process-wide strict-static trap: any hidden dynamic fallback on a
    /// claimed-static path panics while armed. Tests arm it only around the
    /// static side (the dynamic oracle is built unarmed) under `TEST_ENV_LOCK`.
    struct StrictStaticTrapGuard(bool);
    impl StrictStaticTrapGuard {
        fn arm() -> Self {
            unsafe {
                std::env::set_var("GLRMASK_STRICT_STATIC_TRAP_DYNAMIC", "1");
            }
            Self(true)
        }
        fn disarm(&mut self) {
            if self.0 {
                unsafe {
                    std::env::remove_var("GLRMASK_STRICT_STATIC_TRAP_DYNAMIC");
                }
                self.0 = false;
            }
        }
    }
    impl Drop for StrictStaticTrapGuard {
        fn drop(&mut self) {
            self.disarm();
        }
    }

    fn admits_token(mask: &[u32], token: u32) -> bool {
        mask.get(token as usize / 32)
            .is_some_and(|word| word & (1u32 << (token % 32)) != 0)
    }

    #[test]
    fn floor_flag_does_not_suppress_explicit_static_boundary_shards() {
        // The legacy `GLRMASK_EXPERIMENT_OWNED_COMPONENTS_ONLY_STATIC` floor
        // flag must not suppress boundary shards on modern explicit Static /
        // Dynamic links: the env read is gated on
        // `explicit_segmented_boundary.is_none()`. Uses the public nested
        // fixture (12-token vocab; parent `"L" SUB SUB "x" | "R" SUB SUB "y"`;
        // mid `"m" SUB2`; grandchild `"g"`): inner Dynamic reference, outer
        // Static + Dynamic. Unflagged baseline first; then flag set (RAII)
        // with flagged static + dynamic; then all four compared mask-for-mask
        // over [LM,G,M,GX] and [RM,G,M,GY] with end acceptance.
        let _env_lock = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        struct FloorFlagGuard {
            previous: Option<std::ffi::OsString>,
        }
        impl FloorFlagGuard {
            fn set(value: Option<&str>) -> Self {
                const KEY: &str = "GLRMASK_EXPERIMENT_OWNED_COMPONENTS_ONLY_STATIC";
                let previous = std::env::var_os(KEY);
                unsafe {
                    match value {
                        Some(v) => std::env::set_var(KEY, v),
                        None => std::env::remove_var(KEY),
                    }
                }
                Self { previous }
            }
        }
        impl Drop for FloorFlagGuard {
            fn drop(&mut self) {
                const KEY: &str = "GLRMASK_EXPERIMENT_OWNED_COMPONENTS_ONLY_STATIC";
                unsafe {
                    match &self.previous {
                        Some(v) => std::env::set_var(KEY, v),
                        None => std::env::remove_var(KEY),
                    }
                }
            }
        }
        fn has_static_shard(constraint: &Constraint) -> bool {
            constraint
                .static_dynamic_overlay
                .as_ref()
                .is_some_and(|overlay| {
                    overlay
                        .segmented_parser_components
                        .iter()
                        .any(|component| {
                            matches!(
                                component.boundary.as_ref().map(|shard| &shard.backend),
                                Some(crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_))
                            )
                        })
                })
        }
        const M: u32 = 4;
        const G: u32 = 5;
        const LM: u32 = 6;
        const RM: u32 = 7;
        const GX: u32 = 10;
        const GY: u32 = 11;
        let vocab = Vocab::new(vec![
            (0, b"L".to_vec()),
            (1, b"R".to_vec()),
            (2, b"x".to_vec()),
            (3, b"y".to_vec()),
            (4, b"m".to_vec()),
            (5, b"g".to_vec()),
            (6, b"Lm".to_vec()),
            (7, b"Rm".to_vec()),
            (8, b"mg".to_vec()),
            (9, b"gm".to_vec()),
            (10, b"gx".to_vec()),
            (11, b"gy".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "L" SUB SUB "x" | "R" SUB SUB "y";
            "#,
            &vocab,
        )
        .unwrap();
        let mid = Constraint::from_glrm_grammar(
            r#"
                start m;
                t SUB2 ::= @token(998);
                nt m ::= "m" SUB2;
            "#,
            &vocab,
        )
        .unwrap();
        let grandchild = Constraint::from_glrm_grammar(
            r#"
                start g;
                nt g ::= "g";
            "#,
            &vocab,
        )
        .unwrap();
        let mid_inputs = [CompiledSubgrammarInput {
            placeholder_terminal: terminal(&mid, "SUB2"),
            additional_placeholder_terminals: &[],
            constraint: &grandchild,
        }];
        let mid_dyn = compose_constraints_owned_parent_segmented(
            mid.clone(),
            &mid_inputs,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("inner dynamic compose")
        .constraint;
        let outer_inputs = [CompiledSubgrammarInput {
            placeholder_terminal: terminal(&parent, "SUB"),
            additional_placeholder_terminals: &[],
            constraint: &mid_dyn,
        }];
        // Unflagged baseline static + dynamic FIRST (flag explicitly absent).
        let _baseline_unset = FloorFlagGuard::set(None);
        let base_static = compose_constraints_owned_parent_segmented(
            parent.clone(),
            &outer_inputs,
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .expect("baseline static compose")
        .constraint;
        let base_dynamic = compose_constraints_owned_parent_segmented(
            parent.clone(),
            &outer_inputs,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("baseline dynamic compose")
        .constraint;
        assert!(
            has_static_shard(&base_static),
            "baseline static link must publish a static shard",
        );
        // Flagged static + dynamic under RAII flag (restores prior value).
        let (flag_static, flag_dynamic) = {
            let _flag = FloorFlagGuard::set(Some("1"));
            (
                compose_constraints_owned_parent_segmented(
                    parent.clone(),
                    &outer_inputs,
                    &vocab,
                    SegmentedBoundaryBackend::StaticParserDwa,
                )
                .expect("flagged static compose")
                .constraint,
                compose_constraints_owned_parent_segmented(
                    parent.clone(),
                    &outer_inputs,
                    &vocab,
                    SegmentedBoundaryBackend::Dynamic,
                )
                .expect("flagged dynamic compose")
                .constraint,
            )
        };
        assert!(
            has_static_shard(&flag_static),
            "floor flag must not suppress static boundary shards on an explicit static link",
        );
        // All four compared mask-for-mask: INITIAL masks first, then the
        // commit loop; fresh states (reset) per sequence.
        for sequence in [&[LM, G, M, GX][..], &[RM, G, M, GY][..]] {
            let mut states = [
                base_static.start(),
                base_dynamic.start(),
                flag_static.start(),
                flag_dynamic.start(),
            ];
            let initial = [
                states[0].mask(),
                states[1].mask(),
                states[2].mask(),
                states[3].mask(),
            ];
            assert_eq!(initial[0], initial[1], "baseline initial static==dynamic");
            assert_eq!(initial[2], initial[3], "flagged initial static==dynamic");
            assert_eq!(initial[0], initial[2], "flagged initial==baseline initial");
            for &token in sequence {
                for st in &mut states {
                    assert!(
                        admits_token(&st.mask(), token),
                        "floor-flag token {token} admitted",
                    );
                    st.commit_token(token).expect("floor-flag commit");
                }
                let masks = [
                    states[0].mask(),
                    states[1].mask(),
                    states[2].mask(),
                    states[3].mask(),
                ];
                assert_eq!(masks[0], masks[1], "baseline static==dynamic");
                assert_eq!(masks[2], masks[3], "flagged static==dynamic");
                assert_eq!(masks[0], masks[2], "flagged==baseline");
            }
            for st in &states {
                assert!(st.is_accepting(), "floor-flag sequence must accept");
            }
        }
    }

    #[test]
    #[ignore = "diagnostic: requires GLRMASK_MINIMIZE_EQ_LEFT/RIGHT prepared static.bin paths (A1/B1); run explicitly"]
    fn prepared_static_minimize_orders_are_weighted_equivalent() {
        // Diagnostic weighted-language equivalence between two prepared
        // static compositions differing ONLY in signed-shard minimize order
        // (Stable vs DescendingDomain). Both must be built by the SAME binary
        // from the SAME inputs (selected10 cache, sequential-3 prepare flow).
        // For each StaticParser shard present in BOTH (matched by
        // start_component, nonempty on both sides, identical component set),
        // `find_difference` (symmetric checker) must return None on the
        // recursive-parser-coordinate DWAs. Coordinate comparability is
        // verified from loaded metadata (`uses_composed_tsid_coordinate`,
        // `tokenizer_state_to_tsid`, `internal_token_to_originals` equal);
        // any mismatch stops the test instead of comparing incomparable
        // weights. State counts are NOT compared (representation may differ).
        let _env_lock = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let left_path = std::env::var("GLRMASK_MINIMIZE_EQ_LEFT")
            .expect("GLRMASK_MINIMIZE_EQ_LEFT must name the Stable prepared static.bin");
        let right_path = std::env::var("GLRMASK_MINIMIZE_EQ_RIGHT")
            .expect("GLRMASK_MINIMIZE_EQ_RIGHT must name the Descending prepared static.bin");
        let phase1 = std::env::var("PHASE1_DIR")
            .expect("PHASE1_DIR must name the selected10 cache root");
        let vocab = super::load_vocab(
            &std::path::Path::new(&phase1).join("vocab_dump.bin").display().to_string(),
        );
        let load = |path: &str| {
            let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
            Constraint::load_with_vocab(bytes, &vocab)
                .unwrap_or_else(|e| panic!("load {path}: {e:?}"))
        };
        let left = load(&left_path);
        let right = load(&right_path);
        let left_overlay = left
            .static_dynamic_overlay
            .as_ref()
            .expect("LEFT prepared static must retain a segmented overlay");
        let right_overlay = right
            .static_dynamic_overlay
            .as_ref()
            .expect("RIGHT prepared static must retain a segmented overlay");
        let collect = |overlay: &crate::runtime::StaticDynamicOverlayMetadata| {
            let mut shards = std::collections::BTreeMap::new();
            for component in &overlay.segmented_parser_components {
                let Some(shard) = component.boundary.as_ref() else {
                    continue;
                };
                let crate::runtime::SegmentedBoundaryShardBackend::StaticParser(parser) =
                    &shard.backend
                else {
                    continue;
                };
                assert!(
                    shards.insert(shard.start_component, parser.clone()).is_none(),
                    "duplicate start_component shard",
                );
            }
            shards
        };
        let left_shards = collect(left_overlay);
        let right_shards = collect(right_overlay);
        assert!(!left_shards.is_empty(), "LEFT must carry >=1 static shard");
        assert!(!right_shards.is_empty(), "RIGHT must carry >=1 static shard");
        let left_set: Vec<u32> = left_shards.keys().copied().collect();
        let right_set: Vec<u32> = right_shards.keys().copied().collect();
        assert_eq!(
            left_set, right_set,
            "static shard component sets must be identical (LEFT {left_set:?} vs RIGHT {right_set:?})"
        );
        let mut checked = 0usize;
        for (component, left_parser) in &left_shards {
            let right_parser = &right_shards[component];
            eprintln!("[minimize-eq] start_component={component}: checking");
            assert_eq!(
                left_parser.uses_composed_tsid_coordinate,
                right_parser.uses_composed_tsid_coordinate,
                "component {component}: composed-TSID coordinate assumption differs",
            );
            assert_eq!(
                left_parser.tokenizer_state_to_tsid, right_parser.tokenizer_state_to_tsid,
                "component {component}: tokenizer_state_to_tsid differs; incomparable weights",
            );
            assert_eq!(
                left_parser.internal_token_to_originals,
                right_parser.internal_token_to_originals,
                "component {component}: internal_token_to_originals differs; incomparable weights",
            );
            let (Some(left_dwa), Some(right_dwa)) = (
                left_parser.recursive_parser_dwa.as_ref(),
                right_parser.recursive_parser_dwa.as_ref(),
            ) else {
                panic!("component {component}: both sides must carry recursive_parser_dwa");
            };
            assert!(left_dwa.is_acyclic(), "component {component}: LEFT DWA must be acyclic");
            assert!(right_dwa.is_acyclic(), "component {component}: RIGHT DWA must be acyclic");
            match crate::automata::weighted_u32::equivalence::find_difference(left_dwa, right_dwa) {
                Ok(None) => {
                    eprintln!(
                        "[minimize-eq] start_component={component}: done equivalent (states {} vs {})",
                        left_dwa.num_states(),
                        right_dwa.num_states(),
                    );
                    checked += 1;
                }
                Ok(Some(word)) => panic!(
                    "component {component}: weighted-language difference at word {word:?}"
                ),
                Err(e) => panic!("component {component}: checker error: {e:?}"),
            }
        }
        assert!(checked > 0, "must check >=1 shard non-vacuously");
        eprintln!("[minimize-eq] all {checked} static shards weighted-equivalent");
    }

    #[test]
    fn already_composed_parent_static_link_extends_and_matches_dynamic() {
        // Chained static binds: reusing an already-composed constraint as the
        // *parent* of a static link expands the parent block to intact leaves
        // exactly like nested children (no dynamic fallback, no silent
        // substitution). First link fills SUB through the dynamic route,
        // leaving SUB2 free; the static second link must succeed with
        // installed static shards and dynamic-identical masks.
        let _env_lock = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let vocab = Vocab::new(vec![
            (0, b"L".to_vec()),
            (1, b"R".to_vec()),
            (2, b"x".to_vec()),
            (3, b"y".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                t SUB2 ::= @token(998);
                nt document ::= "L" SUB "x" | "R" SUB2 "y";
            "#,
            &vocab,
        )
        .unwrap();
        let child_a = Constraint::from_glrm_grammar(
            r#"
                start a;
                nt a ::= "x";
            "#,
            &vocab,
        )
        .unwrap();
        let child_b = Constraint::from_glrm_grammar(
            r#"
                start b;
                nt b ::= "y";
            "#,
            &vocab,
        )
        .unwrap();
        let first_inputs = [CompiledSubgrammarInput {
            placeholder_terminal: terminal(&parent, "SUB"),
            additional_placeholder_terminals: &[],
            constraint: &child_a,
        }];
        let composed_parent = compose_constraints_owned_parent_segmented(
            parent.clone(),
            &first_inputs,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("first dynamic compose")
        .constraint;
        assert!(
            composed_parent.has_recursive_segmented_parser_tree(),
            "first composition must leave a segmented parent",
        );
        // Coverage boundary: this shape carries no terminal aliases, so slot
        // resolution runs the interval path. (`global_terminal_aliases` are
        // ignore-fold metadata produced only by
        // `build_segmented_runtime_metadata` from `merged_ignore_terminals`
        // (which sees only per-component `ignore_terminal`s) — additional
        // placeholder terminals produce links, never aliases. A slot naming a
        // folded terminal therefore declines loudly as ambiguous, by design.)
        for component in composed_parent
            .static_dynamic_overlay
            .as_ref()
            .expect("segmented overlay")
            .segmented_parser_components
            .iter()
        {
            assert!(
                component.global_terminal_aliases.is_empty(),
                "pin fixture shape must stay alias-free",
            );
        }
        let second_inputs = [CompiledSubgrammarInput {
            placeholder_terminal: terminal(&composed_parent, "SUB2"),
            additional_placeholder_terminals: &[],
            constraint: &child_b,
        }];
        let extended = compose_constraints_owned_parent_segmented(
            composed_parent.clone(),
            &second_inputs,
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .expect("static link over a composed parent must extend")
        .constraint;
        let overlay = extended
            .static_dynamic_overlay
            .as_ref()
            .expect("extended composition must retain segmented metadata");
        assert!(
            !overlay.segmented_boundary_shards.is_empty(),
            "extended composition must publish static shards",
        );
        assert!(
            overlay.segmented_boundary_parser.is_none(),
            "extended composition must not retain a redundant global boundary parser",
        );
        let reference = compose_constraints_owned_parent_segmented(
            composed_parent,
            &second_inputs,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("dynamic second link")
        .constraint;
        let prefixes: Vec<Vec<u8>> = vec![
            vec![],
            vec![b'L'],
            vec![b'L', b'x'],
            vec![b'L', b'x', b'x'],
            vec![b'R'],
            vec![b'R', b'y'],
            vec![b'R', b'y', b'y'],
            vec![b'L', b'y'],
            vec![b'R', b'x'],
            vec![b'x'],
        ];
        assert_composed_masks_equal("composed-parent-extend", &extended, &reference, &prefixes);
    }

    #[test]
    fn composed_parent_three_bind_chain_with_nonroot_slot_matches_dynamic() {
        if crate::isolate_environment_test(false) { return; }
        // Same deep-debug-stack headroom as the branching fixture (measured
        // there): 3-level multibyte static prime/mask exceeds the default 2MB.
        std::thread::Builder::new()
            .name("three-bind".into())
            .stack_size(64 * 1024 * 1024)
            .spawn(composed_parent_three_bind_chain_body)
            .expect("spawn three-bind fixture worker")
            .join()
            .expect("three-bind fixture worker panicked");
    }

    /// Deep-stack worker body for the three-bind test (see the wrapper above).
    fn composed_parent_three_bind_chain_body() {
        // Three chained static binds where the second bind's slot is owned by
        // a NON-ROOT leaf of the parent block (A's open SUB_A), and the third
        // bind fills two placeholders (additional terminals) with a composed
        // child: composed-as-parent and composed-as-child in one chain.
        // Multibyte tokens cross CALL (Lx, xy, Ry) and RETURN (yx, yy, yz) over
        // three nesting levels.
        let _env_lock = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        const L: u32 = 0;
        const R: u32 = 1;
        const X: u32 = 2;
        const Y: u32 = 3;
        const Z: u32 = 4;
        const LX: u32 = 5;
        const XY: u32 = 6;
        const YX: u32 = 7;
        const RY: u32 = 8;
        const YY: u32 = 9;
        const YZ: u32 = 10;
        let vocab = Vocab::new(vec![
            (0, b"L".to_vec()),
            (1, b"R".to_vec()),
            (2, b"x".to_vec()),
            (3, b"y".to_vec()),
            (4, b"z".to_vec()),
            (5, b"Lx".to_vec()),
            (6, b"xy".to_vec()),
            (7, b"yx".to_vec()),
            (8, b"Ry".to_vec()),
            (9, b"yy".to_vec()),
            (10, b"yz".to_vec()),
        ]);
        let root = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                t SUB2 ::= @token(998);
                t SUB3 ::= @token(997);
                nt document ::= "L" SUB "x" | "R" SUB2 "y" | "R" SUB3 "z";
            "#,
            &vocab,
        )
        .unwrap();
        let child_a = Constraint::from_glrm_grammar(
            r#"
                start a;
                t SUB_A ::= @token(996);
                nt a ::= "x" SUB_A;
            "#,
            &vocab,
        )
        .unwrap();
        let child_c = Constraint::from_glrm_grammar(
            r#"
                start c;
                nt c ::= "y";
            "#,
            &vocab,
        )
        .unwrap();
        // Composed child for the third bind (two slots, one filler).
        let parent_b = Constraint::from_glrm_grammar(
            r#"
                start bp;
                t SB1 ::= @token(995);
                t SB2 ::= @token(994);
                nt bp ::= SB1 | SB2;
            "#,
            &vocab,
        )
        .unwrap();
        let child_e = Constraint::from_glrm_grammar(
            r#"
                start e;
                nt e ::= "y";
            "#,
            &vocab,
        )
        .unwrap();
        let b_inputs = [CompiledSubgrammarInput {
            placeholder_terminal: terminal(&parent_b, "SB1"),
            additional_placeholder_terminals: &[terminal(&parent_b, "SB2")],
            constraint: &child_e,
        }];
        let child_b = compose_constraints_owned_parent_segmented(
            parent_b,
            &b_inputs,
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .expect("bind0: composed child")
        .constraint;
        // Placeholder/link contract: additional placeholder terminals produce
        // one link per slot sharing the child (`build_segmented_parser_links`),
        // never `global_terminal_aliases` entries (those are ignore-fold
        // metadata from `build_segmented_runtime_metadata` over
        // `merged_ignore_terminals`, which sees only `ignore_terminal`s).
        {
            let overlay = child_b
                .static_dynamic_overlay
                .as_ref()
                .expect("bind0 segmented overlay");
            assert_eq!(
                overlay.segmented_parser_links.len(),
                2,
                "two slots (SB1, SB2) must produce two links",
            );
            for link in &overlay.segmented_parser_links {
                assert_eq!(link.parent_component, 0);
                assert_eq!(link.child_component, 1);
            }
            assert_eq!(
                overlay
                    .segmented_parser_links
                    .iter()
                    .map(|link| link.slot_terminal)
                    .collect::<Vec<_>>(),
                vec![b_inputs[0].placeholder_terminal]
                    .into_iter()
                    .chain(b_inputs[0].additional_placeholder_terminals.iter().copied())
                    .collect::<Vec<_>>(),
                "one link per placeholder slot, in slot order",
            );
            assert!(
                overlay
                    .segmented_parser_components
                    .iter()
                    .all(|component| component.global_terminal_aliases.is_empty()),
                "multi-slot binds must not produce terminal aliases",
            );
        }
        let bind1 = [CompiledSubgrammarInput {
            placeholder_terminal: terminal(&root, "SUB"),
            additional_placeholder_terminals: &[],
            constraint: &child_a,
        }];
        let stage1 = compose_constraints_owned_parent_segmented(
            root.clone(),
            &bind1,
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .expect("bind1")
        .constraint;
        let stage1_dyn = compose_constraints_owned_parent_segmented(
            root.clone(),
            &bind1,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("bind1 dynamic")
        .constraint;
        // SUB_A is owned by A's leaf (non-root): its composed id is A's overlay
        // offset plus its A-local id (the public bind path exposes the same id
        // as a qualified late-grammar slot).
        let sub_a = stage1
            .static_dynamic_overlay
            .as_ref()
            .expect("segmented overlay")
            .segmented_parser_components[1]
            .terminal_offset
            + terminal(&child_a, "SUB_A");
        let bind2 = [CompiledSubgrammarInput {
            placeholder_terminal: sub_a,
            additional_placeholder_terminals: &[],
            constraint: &child_c,
        }];
        // White-box pin: the SUB_A slot lives in A's leaf (non-root), so the
        // outer link must target a nonzero parent leaf with a leaf-local slot.
        let expansion = crate::compiler::composition::boundary::walk::expand_nested_link_leaves(&stage1, &bind2)
            .expect("bind2 expansion");
        assert_eq!(expansion.top_leaf_ranges.len(), 2, "two tops");
        assert_eq!(expansion.top_leaf_ranges[0].len(), 2, "parent block spans two leaves");
        assert_eq!(expansion.links.len(), 2, "one inner plus one outer link");
        let outer = expansion
            .links
            .iter()
            .find(|link| link.child_component == expansion.top_root_leaves[1])
            .expect("outer link to the new child");
        assert_ne!(
            outer.parent_component, 0,
            "non-root-owned slot must not resolve to leaf 0",
        );
        assert!(
            (outer.slot_terminal as usize)
                < expansion.leaves[outer.parent_component as usize]
                    .table
                    .num_terminals as usize,
            "resolved slot must be leaf-local",
        );
        let stage2 = compose_constraints_owned_parent_segmented(
            stage1.clone(),
            &bind2,
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .expect("bind2")
        .constraint;
        let stage2_dyn = compose_constraints_owned_parent_segmented(
            stage1_dyn,
            &bind2,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("bind2 dynamic")
        .constraint;
        let bind3 = [CompiledSubgrammarInput {
            placeholder_terminal: terminal(&stage2, "SUB2"),
            additional_placeholder_terminals: &[terminal(&stage2, "SUB3")],
            constraint: &child_b,
        }];
        let stage3 = compose_constraints_owned_parent_segmented(
            stage2.clone(),
            &bind3,
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .expect("bind3")
        .constraint;
        let stage3_dyn = compose_constraints_owned_parent_segmented(
            stage2_dyn,
            &bind3,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("bind3 dynamic")
        .constraint;
        // bind3 fills two placeholders with one composed child: two links, no
        // aliases (same placeholder/link contract as bind0).
        {
            let overlay = stage3
                .static_dynamic_overlay
                .as_ref()
                .expect("bind3 segmented overlay");
            assert_eq!(
                overlay.segmented_parser_links.len(),
                2,
                "two slots (SUB2, SUB3) must produce two links",
            );
            for link in &overlay.segmented_parser_links {
                assert_eq!(link.parent_component, 0);
                assert_eq!(link.child_component, 1);
            }
            assert!(
                overlay
                    .segmented_parser_components
                    .iter()
                    .all(|component| component.global_terminal_aliases.is_empty()),
                "multi-slot binds must not produce terminal aliases",
            );
        }
        // Static shard authority: both tops hold StaticParser shards and no
        // redundant global boundary parser. The already-composed parent block
        // retains its own static inner repair: under immediate block ownership
        // the outer shard omits block-internal crossings by construction.
        {
            let overlay = stage3
                .static_dynamic_overlay
                .as_ref()
                .expect("static overlay");
            assert_eq!(overlay.segmented_parser_components.len(), 2);
            for (index, component) in overlay.segmented_parser_components.iter().enumerate() {
                let shard = component.boundary.as_ref().unwrap_or_else(|| {
                    panic!("installed top component {index} must carry a static shard")
                });
                assert!(
                    matches!(
                        shard.backend,
                        crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_)
                    ),
                    "installed component {index} must be a StaticParser shard",
                );
            }
            assert!(
                overlay.segmented_boundary_parser.is_none(),
                "extended composition must not retain a redundant global boundary parser",
            );
            let inner = overlay.segmented_parser_components[0]
                .constraint
                .static_dynamic_overlay
                .as_ref()
                .expect("parent-block overlay");
            assert!(
                inner
                    .segmented_parser_components
                    .iter()
                    .all(|component| component.boundary.as_ref().is_none_or(|shard| matches!(
                        shard.backend,
                        crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_)
                    ))),
                "parent-block overlay must retain only static inner shards",
            );
            assert!(
                !inner.segmented_boundary_shards.is_empty()
                    && inner.segmented_boundary_shards.iter().all(|shard| matches!(
                        shard.backend,
                        crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_)
                    )),
                "parent-block overlay static shard list must survive the extension",
            );
        }
        // Byte-prefix differential (single-byte commits).
        let prefixes: Vec<Vec<u8>> = vec![
            vec![],
            vec![b'L'],
            vec![b'L', b'x'],
            vec![b'L', b'x', b'y'],
            vec![b'L', b'x', b'y', b'x'],
            vec![b'R'],
            vec![b'R', b'y'],
            vec![b'R', b'y', b'y'],
            vec![b'R', b'y', b'z'],
            vec![b'L', b'y'],
            vec![b'R', b'x'],
            vec![b'z'],
        ];
        assert_composed_masks_equal("bind3-bytes", &stage3, &stage3_dyn, &prefixes);
        // Token-level differential (whole multibyte commits across CALL/RETURN).
        let paths: Vec<Vec<u32>> = vec![
            vec![],
            vec![L],
            vec![R],
            vec![LX],
            vec![RY],
            vec![L, X],
            vec![L, X, Y],
            vec![L, XY],
            vec![LX, Y],
            vec![LX, YX],
            vec![L, X, Y, X],
            vec![R, Y],
            vec![RY, Y],
            vec![R, YY],
            vec![R, Y, Y],
            vec![RY, Z],
            vec![R, YZ],
            vec![R, Y, Z],
            vec![L, Y],
            vec![R, X],
            vec![L, X, Z],
            vec![Z],
        ];
        assert_composed_token_masks_equal("bind3-tokens", &stage3, &stage3_dyn, &paths);
        // Fused/single equivalence: the same bytes through fused crossings
        // reach the same masks and acceptance as single-byte tokens, proving
        // the multibyte crossings resolve through the static shards.
        for (fused, single) in [
            (vec![LX], vec![L, X]),
            (vec![RY], vec![R, Y]),
            (vec![L, XY], vec![L, X, Y]),
            (vec![LX, YX], vec![L, X, Y, X]),
            (vec![R, YY], vec![R, Y, Y]),
            (vec![R, YZ], vec![R, Y, Z]),
        ] {
            let mut st_fused = stage3.start();
            for &token in &fused {
                st_fused.commit_token(token).unwrap();
            }
            let mut st_single = stage3.start();
            for &token in &single {
                st_single.commit_token(token).unwrap();
            }
            assert_eq!(
                st_fused.mask(),
                st_single.mask(),
                "fused/single mask equivalence for {fused:?} vs {single:?}",
            );
            assert_eq!(
                st_fused.is_accepting(),
                st_single.is_accepting(),
                "fused/single acceptance equivalence for {fused:?} vs {single:?}",
            );
        }
        assert!(admits_token(&stage3.start().mask(), LX), "Lx crossing must be live");
        assert!(admits_token(&stage3.start().mask(), RY), "Ry crossing must be live");
        // Trap-armed replay of the oracle-gated corpus: any hidden dynamic
        // fallback on the static side panics here.
        let mut nodes: Vec<(Vec<u32>, Vec<u32>)> = Vec::new();
        for path in &paths {
            let mut st = stage3_dyn.start();
            let mut ok = true;
            for &token in path {
                if !admits_token(&st.mask(), token) || st.commit_token(token).is_err() {
                    ok = false;
                    break;
                }
            }
            if ok {
                nodes.push((path.clone(), st.mask()));
            }
        }
        assert!(
            nodes.len() >= 10,
            "three-bind corpus must cover real positions, got {}",
            nodes.len()
        );
        let mut trap = StrictStaticTrapGuard::arm();
        let mut mismatches = 0usize;
        for (path, mask) in &nodes {
            let mut st = stage3.start();
            for &token in path {
                st.commit_token(token).expect("static replay");
            }
            if st.mask() != *mask {
                mismatches += 1;
            }
        }
        trap.disarm();
        assert_eq!(mismatches, 0, "three-bind static must match dynamic on every path");
        // Ablation: clearing the installed top-level shards must lose
        // admissions (the shards are genuinely authoritative, not redundant).
        let mut ablated = stage3.clone();
        install_published_static_boundary_shards(
            ablated.static_dynamic_overlay.as_mut().expect("overlay"),
            Vec::new(),
        )
        .expect("clear shards");
        let mut strict = 0usize;
        for path in &paths {
            let mut dyn_st = stage3_dyn.start();
            let mut abl_st = ablated.start();
            for &token in path {
                if !admits_token(&dyn_st.mask(), token) {
                    break;
                }
                dyn_st.commit_token(token).unwrap();
                if !admits_token(&abl_st.mask(), token) {
                    strict += 1;
                    break;
                }
                abl_st.commit_token(token).unwrap();
            }
        }
        assert!(strict >= 1, "ablation must lose at least one admission");
    }

    #[test]
    fn composed_parent_branching_then_nonroot_extension_matches_dynamic() {
        if crate::isolate_environment_test(false) { return; }
        // Static prime/mask evaluation recurses per nesting level with large
        // debug scratch frames (pre-existing runtime characteristic: the
        // existing deep tests need >1MB too, measured); this 3-level multibyte
        // shape needs more than the default 2MB, so the whole body runs on a
        // worker with headroom. The panic hook still prints assertion details;
        // a join failure fails the test.
        std::thread::Builder::new()
            .name("branching-extension".into())
            .stack_size(64 * 1024 * 1024)
            .spawn(composed_parent_branching_then_nonroot_extension_body)
            .expect("spawn branching fixture worker")
            .join()
            .expect("branching fixture worker panicked");
    }

    /// Deep-stack worker body for the branching test (see the wrapper above).
    fn composed_parent_branching_then_nonroot_extension_body() {
        // Branching-then-nonroot-extension: bind1a/bind1b fill sibling slots A
        // and B; bind2 fills a slot owned by A's leaf (non-root) with C. Leaf
        // expansion follows overlay storage ([root, A, B] plus the appended
        // C), while the analysis splice packs link-tree DFS ([root, A, C, B]);
        // the splice must remap explicitly into leaf coordinate instead of
        // assuming the orders coincide. Multibyte tokens cross CALL (Lx, xz,
        // Ry) and RETURN (zx, yy) over two nesting levels.
        let _env_lock = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        const L: u32 = 0;
        const R: u32 = 1;
        const X: u32 = 2;
        const Y: u32 = 3;
        const Z: u32 = 4;
        const LX: u32 = 5;
        const XZ: u32 = 6;
        const ZX: u32 = 7;
        const RY: u32 = 8;
        const YY: u32 = 9;
        let vocab = Vocab::new(vec![
            (0, b"L".to_vec()),
            (1, b"R".to_vec()),
            (2, b"x".to_vec()),
            (3, b"y".to_vec()),
            (4, b"z".to_vec()),
            (5, b"Lx".to_vec()),
            (6, b"xz".to_vec()),
            (7, b"zx".to_vec()),
            (8, b"Ry".to_vec()),
            (9, b"yy".to_vec()),
        ]);
        let root = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB_A ::= @token(999);
                t SUB_B ::= @token(998);
                nt document ::= "L" SUB_A "x" | "R" SUB_B "y";
            "#,
            &vocab,
        )
        .unwrap();
        let child_a = Constraint::from_glrm_grammar(
            r#"
                start a;
                t SUB_INNER ::= @token(996);
                nt a ::= "x" SUB_INNER;
            "#,
            &vocab,
        )
        .unwrap();
        let child_b = Constraint::from_glrm_grammar(
            r#"
                start b;
                nt b ::= "y";
            "#,
            &vocab,
        )
        .unwrap();
        let child_c = Constraint::from_glrm_grammar(
            r#"
                start c;
                nt c ::= "z";
            "#,
            &vocab,
        )
        .unwrap();
        // Two sequential sibling binds (multi-child single links are a known
        // dynamic-oracle gap: the structural quotient breaks the functional
        // LR-state relation there, so the oracle cannot be built that way).
        // bind1a fills SUB_A, bind1b fills SUB_B; the existing root then has
        // siblings A and B with overlay nesting [[root, A], B].
        let bind1a = [CompiledSubgrammarInput {
            placeholder_terminal: terminal(&root, "SUB_A"),
            additional_placeholder_terminals: &[],
            constraint: &child_a,
        }];
        let stage1a = compose_constraints_owned_parent_segmented(
            root.clone(),
            &bind1a,
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .expect("bind1a")
        .constraint;
        let stage1a_dyn = compose_constraints_owned_parent_segmented(
            root.clone(),
            &bind1a,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("bind1a dynamic")
        .constraint;
        let bind1b = [CompiledSubgrammarInput {
            placeholder_terminal: terminal(&stage1a, "SUB_B"),
            additional_placeholder_terminals: &[],
            constraint: &child_b,
        }];
        let stage1 = compose_constraints_owned_parent_segmented(
            stage1a.clone(),
            &bind1b,
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .expect("bind1b")
        .constraint;
        let stage1_dyn = compose_constraints_owned_parent_segmented(
            stage1a_dyn,
            &bind1b,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("bind1b dynamic")
        .constraint;
        {
            let overlay = stage1
                .static_dynamic_overlay
                .as_ref()
                .expect("stage1 segmented overlay");
            assert_eq!(
                overlay.segmented_parser_components.len(),
                2,
                "bind1b stores [[root, A], B]",
            );
            assert!(
                overlay.segmented_parser_components[0]
                    .constraint
                    .has_recursive_segmented_parser_tree(),
                "bind1b component 0 must stay a segmented subtree",
            );
        }
        // SUB_INNER is owned by A's leaf: component 0's offset plus A's offset
        // inside the nested overlay plus its A-local id (the same id the
        // public bind path exposes as a qualified slot).
        let sub_inner = {
            let overlay = stage1
                .static_dynamic_overlay
                .as_ref()
                .expect("segmented overlay");
            let nested = &overlay.segmented_parser_components[0];
            let nested_overlay = nested
                .constraint
                .static_dynamic_overlay
                .as_ref()
                .expect("nested segmented overlay");
            nested.terminal_offset
                + nested_overlay.segmented_parser_components[1].terminal_offset
                + terminal(&child_a, "SUB_INNER")
        };
        let bind2 = [CompiledSubgrammarInput {
            placeholder_terminal: sub_inner,
            additional_placeholder_terminals: &[],
            constraint: &child_c,
        }];
        // White-box pin: leaves are storage order [root, A, B, C] with the
        // parent block spanning three leaves, and the outer link targets A's
        // (non-root) leaf with a leaf-local slot.
        let expansion = crate::compiler::composition::boundary::walk::expand_nested_link_leaves(&stage1, &bind2)
            .expect("bind2 expansion");
        assert_eq!(expansion.leaves.len(), 4, "root, A, B, C");
        assert_eq!(
            expansion.top_leaf_ranges,
            vec![vec![0usize, 1usize, 2usize], vec![3usize]],
            "parent block spans three leaves; C is its own top",
        );
        let outer = expansion
            .links
            .iter()
            .find(|link| link.child_component == expansion.top_root_leaves[1])
            .expect("outer link to the new child");
        assert_eq!(
            outer.parent_component, 1,
            "slot owned by A must resolve to A's leaf",
        );
        assert!(
            (outer.slot_terminal as usize)
                < expansion.leaves[1].table.num_terminals as usize,
            "resolved slot must be leaf-local",
        );
        // The fixture must genuinely interleave: leaves stay storage order
        // [root, A, B, C] while the splice packs DFS [root, A, C, B].
        let dfs_order = crate::compiler::composition::boundary::walk::nested_splice_leaf_order(&expansion)
            .expect("bind2 DFS order");
        assert_eq!(
            dfs_order,
            vec![0usize, 1, 3, 2],
            "splice DFS order must interleave with leaf order",
        );
        let stage2 = compose_constraints_owned_parent_segmented(
            stage1.clone(),
            &bind2,
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .expect("bind2: late non-root extension must link statically")
        .constraint;
        let stage2_dyn = compose_constraints_owned_parent_segmented(
            stage1_dyn,
            &bind2,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("bind2 dynamic")
        .constraint;
        // Static shard authority: both tops hold StaticParser shards and no
        // redundant global boundary parser. The already-composed parent block
        // retains its own static inner repair: under immediate block ownership
        // the new outer shard deliberately omits block-internal crossings, so
        // those retained shards are load-bearing coverage rather than duplicate
        // work.
        {
            let overlay = stage2
                .static_dynamic_overlay
                .as_ref()
                .expect("static overlay");
            assert_eq!(overlay.segmented_parser_components.len(), 2);
            for (index, component) in overlay.segmented_parser_components.iter().enumerate() {
                let shard = component.boundary.as_ref().unwrap_or_else(|| {
                    panic!("installed top component {index} must carry a static shard")
                });
                assert!(
                    matches!(
                        shard.backend,
                        crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_)
                    ),
                    "installed component {index} must be a StaticParser shard",
                );
            }
            assert!(
                overlay.segmented_boundary_parser.is_none(),
                "extended composition must not retain a redundant global boundary parser",
            );
            let inner = overlay.segmented_parser_components[0]
                .constraint
                .static_dynamic_overlay
                .as_ref()
                .expect("parent-block overlay");
            assert!(
                inner
                    .segmented_parser_components
                    .iter()
                    .all(|component| component.boundary.as_ref().is_none_or(|shard| matches!(
                        shard.backend,
                        crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_)
                    ))),
                "parent-block overlay must retain only static inner shards",
            );
            assert!(
                !inner.segmented_boundary_shards.is_empty()
                    && inner.segmented_boundary_shards.iter().all(|shard| matches!(
                        shard.backend,
                        crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_)
                    )),
                "parent-block overlay static shard list must survive the extension",
            );
        }
        // Byte-prefix differential (single-byte commits).
        let prefixes: Vec<Vec<u8>> = vec![
            vec![],
            vec![b'L'],
            vec![b'L', b'x'],
            vec![b'L', b'x', b'z'],
            vec![b'L', b'x', b'z', b'x'],
            vec![b'R'],
            vec![b'R', b'y'],
            vec![b'R', b'y', b'y'],
            vec![b'L', b'y'],
            vec![b'R', b'x'],
            vec![b'L', b'x', b'y'],
            vec![b'z'],
        ];
        assert_composed_masks_equal("branch-extend-bytes", &stage2, &stage2_dyn, &prefixes);
        // Token-level differential (whole multibyte commits across CALL/RETURN).
        let paths: Vec<Vec<u32>> = vec![
            vec![],
            vec![L],
            vec![R],
            vec![LX],
            vec![RY],
            vec![L, X],
            vec![L, X, Z],
            vec![L, XZ],
            vec![LX, Z],
            vec![LX, ZX],
            vec![L, X, Z, X],
            vec![R, Y],
            vec![RY, Y],
            vec![R, YY],
            vec![R, Y, Y],
            vec![L, Y],
            vec![R, X],
            vec![L, X, Y],
            vec![Z],
        ];
        assert_composed_token_masks_equal("branch-extend-tokens", &stage2, &stage2_dyn, &paths);
        // Fused/single equivalence: the same bytes through fused crossings
        // reach the same masks and acceptance as single-byte tokens, proving
        // the multibyte crossings resolve through the static shards.
        for (fused, single) in [
            (vec![LX], vec![L, X]),
            (vec![RY], vec![R, Y]),
            (vec![L, XZ], vec![L, X, Z]),
            (vec![LX, ZX], vec![L, X, Z, X]),
            (vec![R, YY], vec![R, Y, Y]),
        ] {
            let mut st_fused = stage2.start();
            for &token in &fused {
                st_fused.commit_token(token).unwrap();
            }
            let mut st_single = stage2.start();
            for &token in &single {
                st_single.commit_token(token).unwrap();
            }
            assert_eq!(
                st_fused.mask(),
                st_single.mask(),
                "fused/single mask equivalence for {fused:?} vs {single:?}",
            );
            assert_eq!(
                st_fused.is_accepting(),
                st_single.is_accepting(),
                "fused/single acceptance equivalence for {fused:?} vs {single:?}",
            );
        }
        assert!(admits_token(&stage2.start().mask(), LX), "Lx crossing must be live");
        assert!(admits_token(&stage2.start().mask(), RY), "Ry crossing must be live");
        // Trap-armed replay of the oracle-gated corpus: any hidden dynamic
        // fallback on the static side panics here.
        let mut nodes: Vec<(Vec<u32>, Vec<u32>)> = Vec::new();
        for path in &paths {
            let mut st = stage2_dyn.start();
            let mut ok = true;
            for &token in path {
                if !admits_token(&st.mask(), token) || st.commit_token(token).is_err() {
                    ok = false;
                    break;
                }
            }
            if ok {
                nodes.push((path.clone(), st.mask()));
            }
        }
        assert!(
            nodes.len() >= 10,
            "branching corpus must cover real positions, got {}",
            nodes.len()
        );
        let mut trap = StrictStaticTrapGuard::arm();
        let mut mismatches = 0usize;
        for (path, mask) in &nodes {
            let mut st = stage2.start();
            for &token in path {
                st.commit_token(token).expect("static replay");
            }
            if st.mask() != *mask {
                mismatches += 1;
            }
        }
        trap.disarm();
        assert_eq!(mismatches, 0, "branching static must match dynamic on every path");
        // Ablation: clearing the installed top-level shards must lose
        // admissions (the shards are genuinely authoritative, not redundant).
        let mut ablated = stage2.clone();
        install_published_static_boundary_shards(
            ablated.static_dynamic_overlay.as_mut().expect("overlay"),
            Vec::new(),
        )
        .expect("clear shards");
        let mut strict = 0usize;
        for path in &paths {
            let mut dyn_st = stage2_dyn.start();
            let mut abl_st = ablated.start();
            for &token in path {
                if !admits_token(&dyn_st.mask(), token) {
                    break;
                }
                dyn_st.commit_token(token).unwrap();
                if !admits_token(&abl_st.mask(), token) {
                    strict += 1;
                    break;
                }
                abl_st.commit_token(token).unwrap();
            }
        }
        assert!(strict >= 1, "ablation must lose at least one admission");
    }

    #[test]
    fn composed_parent_cross_parent_multislot_static_matches_dynamic() {
        if crate::isolate_environment_test(false) { return; }
        // Same deep-debug-stack headroom as the other 3-level multibyte
        // static prime/mask fixtures: exceeds the default 2MB.
        std::thread::Builder::new()
            .name("cross-parent-multislot".into())
            .stack_size(64 * 1024 * 1024)
            .spawn(composed_parent_cross_parent_multislot_body)
            .expect("spawn cross-parent fixture worker")
            .join()
            .expect("cross-parent fixture worker panicked");
    }

    /// Deep-stack worker body (see above): a cross-parent multi-slot bind is
    /// a VALID public bind case ("incorporated once and shared by every
    /// listed call site") supported by the Dynamic backend, and the static
    /// route supports it too via occurrence mapping. ONE child input fills
    /// slots owned by DIFFERENT leaves (root leaf + A's leaf), so the packed
    /// analysis visits the shared child once per parent path ([0,1,2,2])
    /// while the runtime keeps ONE shared child. Static-vs-dynamic
    /// differentials over multibyte CALL/RETURN crossings from both parents,
    /// fused/single equivalence, a trap-armed replay, and shard ablation
    /// prove the acceptance is genuine. This is NOT the same case as DAG
    /// reuse across inputs (distinct leaves; see the reused-sibling test).
    fn composed_parent_cross_parent_multislot_body() {
        let _env_lock = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        const L: u32 = 0;
        const R: u32 = 1;
        const X: u32 = 2;
        const Y: u32 = 3;
        const LX: u32 = 4;
        const RY: u32 = 5;
        const XY: u32 = 6;
        const YX: u32 = 7;
        const YY: u32 = 8;
        let vocab = Vocab::new(vec![
            (0, b"L".to_vec()),
            (1, b"R".to_vec()),
            (2, b"x".to_vec()),
            (3, b"y".to_vec()),
            (4, b"Lx".to_vec()),
            (5, b"Ry".to_vec()),
            (6, b"xy".to_vec()),
            (7, b"yx".to_vec()),
            (8, b"yy".to_vec()),
        ]);
        let root = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB_A ::= @token(999);
                t SUB_B ::= @token(998);
                nt document ::= "L" SUB_A "x" | "R" SUB_B "y";
            "#,
            &vocab,
        )
        .unwrap();
        let child_a = Constraint::from_glrm_grammar(
            r#"
                start a;
                t SUB_INNER ::= @token(996);
                nt a ::= "x" SUB_INNER;
            "#,
            &vocab,
        )
        .unwrap();
        let child_c = Constraint::from_glrm_grammar(
            r#"
                start c;
                nt c ::= "y";
            "#,
            &vocab,
        )
        .unwrap();
        let bind1a = [CompiledSubgrammarInput {
            placeholder_terminal: terminal(&root, "SUB_A"),
            additional_placeholder_terminals: &[],
            constraint: &child_a,
        }];
        let stage1a = compose_constraints_owned_parent_segmented(
            root.clone(),
            &bind1a,
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .expect("bind1a")
        .constraint;
        let stage1a_dyn = compose_constraints_owned_parent_segmented(
            root.clone(),
            &bind1a,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("bind1a dynamic")
        .constraint;
        let sub_b = terminal(&stage1a, "SUB_B");
        let sub_inner = stage1a
            .static_dynamic_overlay
            .as_ref()
            .expect("overlay")
            .segmented_parser_components[1]
            .terminal_offset
            + terminal(&child_a, "SUB_INNER");
        let bind2 = [CompiledSubgrammarInput {
            placeholder_terminal: sub_b,
            additional_placeholder_terminals: &[sub_inner],
            constraint: &child_c,
        }];
        // White-box pin: the expansion really is the DAG shape (child root
        // with two distinct parents), and the occurrence oracle visits the
        // shared child once per parent path.
        let expansion =
            crate::compiler::composition::boundary::walk::expand_nested_link_leaves(&stage1a, &bind2)
                .expect("expansion");
        assert_eq!(expansion.leaves.len(), 3, "root, A, C");
        let child_root = expansion.top_root_leaves[1] as usize;
        let mut parents: Vec<u32> = expansion
            .links
            .iter()
            .filter(|link| link.child_component as usize == child_root)
            .map(|link| link.parent_component)
            .collect();
        parents.sort_unstable();
        parents.dedup();
        assert_eq!(parents, vec![0, 1], "one child, two distinct parents");
        let dfs_order =
            crate::compiler::composition::boundary::walk::nested_splice_leaf_order(&expansion)
                .expect("occurrence oracle accepts the shared child");
        assert_eq!(
            dfs_order,
            vec![0, 1, 2, 2],
            "shared child occurs once per parent path"
        );
        // The static route supports the valid bind (no deferral).
        let stage2 = compose_constraints_owned_parent_segmented(
            stage1a.clone(),
            &bind2,
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .expect("static cross-parent multi-slot bind is supported")
        .constraint;
        let stage2_dyn = compose_constraints_owned_parent_segmented(
            stage1a_dyn,
            &bind2,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("bind2 dynamic")
        .constraint;
        // Static shard authority on the extended artifact.
        {
            let overlay = stage2
                .static_dynamic_overlay
                .as_ref()
                .expect("static overlay");
            assert_eq!(overlay.segmented_parser_components.len(), 2);
            for (index, component) in overlay.segmented_parser_components.iter().enumerate() {
                let shard = component.boundary.as_ref().unwrap_or_else(|| {
                    panic!("installed top component {index} must carry a static shard")
                });
                assert!(
                    matches!(
                        shard.backend,
                        crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_)
                    ),
                    "installed component {index} must be a StaticParser shard",
                );
            }
            assert!(
                overlay.segmented_boundary_parser.is_none(),
                "extended composition must not retain a redundant global boundary parser",
            );
        }
        // Byte-prefix differential: crossings into/out of the shared child
        // from BOTH parents.
        let prefixes: Vec<Vec<u8>> = vec![
            vec![],
            vec![b'L'],
            vec![b'L', b'x'],
            vec![b'L', b'x', b'y'],
            vec![b'L', b'x', b'y', b'x'],
            vec![b'R'],
            vec![b'R', b'y'],
            vec![b'R', b'y', b'y'],
            vec![b'L', b'y'],
            vec![b'R', b'x'],
            vec![b'y'],
        ];
        assert_composed_masks_equal("xparent-bytes", &stage2, &stage2_dyn, &prefixes);
        // Token-level differential (multibyte CALL/RETURN crossings).
        let paths: Vec<Vec<u32>> = vec![
            vec![],
            vec![L],
            vec![R],
            vec![LX],
            vec![RY],
            vec![L, X],
            vec![L, XY],
            vec![LX, Y],
            vec![L, X, Y],
            vec![L, X, Y, X],
            vec![LX, Y, X],
            vec![L, XY, X],
            vec![L, X, YX],
            vec![R, Y],
            vec![RY, Y],
            vec![R, YY],
            vec![R, Y, Y],
            vec![L, Y],
            vec![R, X],
            vec![Y],
        ];
        assert_composed_token_masks_equal("xparent-tokens", &stage2, &stage2_dyn, &paths);
        // Fused/single equivalence across both parents' CALL/RETURN sites.
        for (fused, single) in [
            (vec![LX, Y, X], vec![L, X, Y, X]),
            (vec![L, XY, X], vec![L, X, Y, X]),
            (vec![L, X, YX], vec![L, X, Y, X]),
            (vec![RY, Y], vec![R, Y, Y]),
            (vec![R, YY], vec![R, Y, Y]),
        ] {
            let mut st_fused = stage2.start();
            for &token in &fused {
                st_fused.commit_token(token).unwrap();
            }
            let mut st_single = stage2.start();
            for &token in &single {
                st_single.commit_token(token).unwrap();
            }
            assert_eq!(
                st_fused.mask(),
                st_single.mask(),
                "fused/single mask equivalence for {fused:?} vs {single:?}",
            );
            assert_eq!(
                st_fused.is_accepting(),
                st_single.is_accepting(),
                "fused/single acceptance equivalence for {fused:?} vs {single:?}",
            );
        }
        // Both parents genuinely reach the shared child: each call site
        // accepts completions on both backends.
        for path in [vec![L, X, Y, X], vec![R, Y, Y]] {
            let mut st = stage2_dyn.start();
            for &token in &path {
                st.commit_token(token).unwrap();
            }
            assert!(st.is_accepting(), "oracle must accept {path:?}");
            let mut st = stage2.start();
            for &token in &path {
                st.commit_token(token).unwrap();
            }
            assert!(st.is_accepting(), "static must accept {path:?}");
        }
        // Trap-armed replay of the oracle-gated corpus.
        let mut nodes: Vec<(Vec<u32>, Vec<u32>)> = Vec::new();
        for path in &paths {
            let mut st = stage2_dyn.start();
            let mut ok = true;
            for &token in path {
                if !admits_token(&st.mask(), token) || st.commit_token(token).is_err() {
                    ok = false;
                    break;
                }
            }
            if ok {
                nodes.push((path.clone(), st.mask()));
            }
        }
        assert!(
            nodes.len() >= 10,
            "cross-parent corpus must cover real positions, got {}",
            nodes.len()
        );
        let mut trap = StrictStaticTrapGuard::arm();
        let mut mismatches = 0usize;
        for (path, mask) in &nodes {
            let mut st = stage2.start();
            for &token in path {
                st.commit_token(token).expect("static replay");
            }
            if st.mask() != *mask {
                mismatches += 1;
            }
        }
        trap.disarm();
        assert_eq!(
            mismatches, 0,
            "cross-parent static must match dynamic on every path"
        );
        // Ablation: clearing the installed top-level shards must lose
        // admissions.
        let mut ablated = stage2.clone();
        install_published_static_boundary_shards(
            ablated.static_dynamic_overlay.as_mut().expect("overlay"),
            Vec::new(),
        )
        .expect("clear shards");
        let mut strict = 0usize;
        for path in &paths {
            let mut dyn_st = stage2_dyn.start();
            let mut abl_st = ablated.start();
            for &token in path {
                if !admits_token(&dyn_st.mask(), token) {
                    break;
                }
                dyn_st.commit_token(token).unwrap();
                if !admits_token(&abl_st.mask(), token) {
                    strict += 1;
                    break;
                }
                abl_st.commit_token(token).unwrap();
            }
        }
        assert!(strict >= 1, "ablation must lose at least one admission");
    }

    #[test]
    fn composed_parent_cross_parent_shared_subtree_matches_dynamic() {
        if crate::isolate_environment_test(false) { return; }
        // Same deep-debug-stack headroom as the sibling 3-level multibyte
        // static prime/mask fixtures: exceeds the default 2MB.
        std::thread::Builder::new()
            .name("cross-parent-subtree".into())
            .stack_size(64 * 1024 * 1024)
            .spawn(composed_parent_cross_parent_shared_subtree_body)
            .expect("spawn cross-parent subtree fixture worker")
            .join()
            .expect("cross-parent subtree fixture worker panicked");
    }

    /// Deep-stack worker body (see above): repeated-subtree authority. The
    /// shared child C has its own bound descendant D (late bind into the
    /// shared leaf AFTER the cross-parent bind), so the packed analysis
    /// repeats the whole C+D subtree once per parent path ([0,1,2,3,2,3])
    /// while the runtime still shares one C and one D. Same fused
    /// multibyte CALL/RETURN/trap/ablation authority as the sibling test,
    /// extended across the C-to-D call site.
    fn composed_parent_cross_parent_shared_subtree_body() {
        let _env_lock = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        const L: u32 = 0;
        const R: u32 = 1;
        const X: u32 = 2;
        const Y: u32 = 3;
        const Z: u32 = 4;
        const LX: u32 = 5;
        const RY: u32 = 6;
        const XY: u32 = 7;
        const YX: u32 = 8;
        const YY: u32 = 9;
        const YZ: u32 = 10;
        const ZX: u32 = 11;
        let vocab = Vocab::new(vec![
            (0, b"L".to_vec()),
            (1, b"R".to_vec()),
            (2, b"x".to_vec()),
            (3, b"y".to_vec()),
            (4, b"z".to_vec()),
            (5, b"Lx".to_vec()),
            (6, b"Ry".to_vec()),
            (7, b"xy".to_vec()),
            (8, b"yx".to_vec()),
            (9, b"yy".to_vec()),
            (10, b"yz".to_vec()),
            (11, b"zx".to_vec()),
        ]);
        let root = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB_A ::= @token(999);
                t SUB_B ::= @token(998);
                nt document ::= "L" SUB_A "x" | "R" SUB_B "y";
            "#,
            &vocab,
        )
        .unwrap();
        let child_a = Constraint::from_glrm_grammar(
            r#"
                start a;
                t SUB_INNER ::= @token(996);
                nt a ::= "x" SUB_INNER;
            "#,
            &vocab,
        )
        .unwrap();
        let child_c = Constraint::from_glrm_grammar(
            r#"
                start c;
                t SUB_D ::= @token(995);
                nt c ::= "y" SUB_D;
            "#,
            &vocab,
        )
        .unwrap();
        let child_d = Constraint::from_glrm_grammar(
            r#"
                start d;
                nt d ::= "z";
            "#,
            &vocab,
        )
        .unwrap();
        let bind1a = [CompiledSubgrammarInput {
            placeholder_terminal: terminal(&root, "SUB_A"),
            additional_placeholder_terminals: &[],
            constraint: &child_a,
        }];
        let stage1a = compose_constraints_owned_parent_segmented(
            root.clone(),
            &bind1a,
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .expect("bind1a")
        .constraint;
        let stage1a_dyn = compose_constraints_owned_parent_segmented(
            root.clone(),
            &bind1a,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("bind1a dynamic")
        .constraint;
        let sub_b = terminal(&stage1a, "SUB_B");
        let sub_inner = stage1a
            .static_dynamic_overlay
            .as_ref()
            .expect("overlay")
            .segmented_parser_components[1]
            .terminal_offset
            + terminal(&child_a, "SUB_INNER");
        let bind2 = [CompiledSubgrammarInput {
            placeholder_terminal: sub_b,
            additional_placeholder_terminals: &[sub_inner],
            constraint: &child_c,
        }];
        // The slotted C binds cross-parent exactly like the atomic sibling.
        let expansion =
            crate::compiler::composition::boundary::walk::expand_nested_link_leaves(&stage1a, &bind2)
                .expect("expansion");
        assert_eq!(expansion.leaves.len(), 3, "root, A, C");
        let stage2 = compose_constraints_owned_parent_segmented(
            stage1a.clone(),
            &bind2,
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .expect("static cross-parent bind of the slotted child")
        .constraint;
        let stage2_dyn = compose_constraints_owned_parent_segmented(
            stage1a_dyn,
            &bind2,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("bind2 dynamic")
        .constraint;
        // Late bind D into the SHARED leaf: C is a top-level component of
        // stage2, so its slot addresses directly off its terminal offset.
        let sub_d = stage2
            .static_dynamic_overlay
            .as_ref()
            .expect("stage2 overlay")
            .segmented_parser_components[1]
            .terminal_offset
            + terminal(&child_c, "SUB_D");
        let bind3 = [CompiledSubgrammarInput {
            placeholder_terminal: sub_d,
            additional_placeholder_terminals: &[],
            constraint: &child_d,
        }];
        // White-box pin: the shared child keeps both parents, D hangs under
        // the shared child alone (shape pin, numbering-independent), and the
        // occurrence oracle repeats the whole shared subtree once per parent
        // path (leaves append in bind order: root, A, C, D).
        let expansion3 =
            crate::compiler::composition::boundary::walk::expand_nested_link_leaves(&stage2, &bind3)
                .expect("expansion3");
        assert_eq!(expansion3.leaves.len(), 4, "root, A, C, D");
        let parents_of = |leaf: usize| -> Vec<u32> {
            let mut parents: Vec<u32> = expansion3
                .links
                .iter()
                .filter(|link| link.child_component as usize == leaf)
                .map(|link| link.parent_component)
                .collect();
            parents.sort_unstable();
            parents.dedup();
            parents
        };
        let mut shared = None;
        for leaf in 0..expansion3.leaves.len() {
            if parents_of(leaf).len() == 2 {
                assert!(shared.is_none(), "exactly one shared child");
                shared = Some(leaf);
            }
        }
        let shared = shared.expect("shared child keeps two parents");
        let mut under_shared = Vec::new();
        for leaf in 0..expansion3.leaves.len() {
            if parents_of(leaf) == vec![shared as u32] {
                under_shared.push(leaf);
            }
        }
        assert_eq!(
            under_shared.len(),
            1,
            "one descendant hangs under the shared child alone"
        );
        let dfs_order3 =
            crate::compiler::composition::boundary::walk::nested_splice_leaf_order(&expansion3)
                .expect("occurrence oracle accepts the shared subtree");
        assert_eq!(
            dfs_order3,
            vec![0, 1, 2, 3, 2, 3],
            "shared subtree occurs once per parent path"
        );
        let stage3 = compose_constraints_owned_parent_segmented(
            stage2.clone(),
            &bind3,
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .expect("static late bind into the shared leaf")
        .constraint;
        let stage3_dyn = compose_constraints_owned_parent_segmented(
            stage2_dyn,
            &bind3,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("bind3 dynamic")
        .constraint;
        // Static shard authority on the twice-extended artifact.
        {
            let overlay = stage3
                .static_dynamic_overlay
                .as_ref()
                .expect("static overlay");
            assert_eq!(overlay.segmented_parser_components.len(), 2);
            for (index, component) in overlay.segmented_parser_components.iter().enumerate() {
                let shard = component.boundary.as_ref().unwrap_or_else(|| {
                    panic!("installed top component {index} must carry a static shard")
                });
                assert!(
                    matches!(
                        shard.backend,
                        crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_)
                    ),
                    "installed component {index} must be a StaticParser shard",
                );
            }
            assert!(
                overlay.segmented_boundary_parser.is_none(),
                "extended composition must not retain a redundant global boundary parser",
            );
        }
        // Byte-prefix differential across the C-to-D call site.
        let prefixes: Vec<Vec<u8>> = vec![
            vec![],
            vec![b'L'],
            vec![b'L', b'x'],
            vec![b'L', b'x', b'y'],
            vec![b'L', b'x', b'y', b'z'],
            vec![b'L', b'x', b'y', b'z', b'x'],
            vec![b'R'],
            vec![b'R', b'y'],
            vec![b'R', b'y', b'z'],
            vec![b'R', b'y', b'z', b'y'],
            vec![b'L', b'y'],
            vec![b'R', b'x'],
            vec![b'z'],
        ];
        assert_composed_masks_equal("xsubtree-bytes", &stage3, &stage3_dyn, &prefixes);
        // Token-level differential (multibyte CALL/RETURN crossings).
        let paths: Vec<Vec<u32>> = vec![
            vec![],
            vec![L],
            vec![R],
            vec![LX],
            vec![RY],
            vec![L, X],
            vec![L, XY],
            vec![LX, Y],
            vec![L, X, Y],
            vec![L, X, Y, Z],
            vec![L, X, Y, Z, X],
            vec![LX, Y, Z, X],
            vec![L, XY, Z, X],
            vec![L, X, YZ, X],
            vec![L, X, Y, ZX],
            vec![R, Y],
            vec![RY, Y],
            vec![R, Y, Z],
            vec![RY, Z],
            vec![R, YZ],
            vec![R, Y, Z, Y],
            vec![RY, Z, Y],
            vec![R, YZ, Y],
            vec![L, Y],
            vec![R, X],
            vec![Z],
        ];
        assert_composed_token_masks_equal("xsubtree-tokens", &stage3, &stage3_dyn, &paths);
        // Fused/single equivalence across every CALL/RETURN site.
        for (fused, single) in [
            (vec![LX, Y, Z, X], vec![L, X, Y, Z, X]),
            (vec![L, XY, Z, X], vec![L, X, Y, Z, X]),
            (vec![L, X, YZ, X], vec![L, X, Y, Z, X]),
            (vec![L, X, Y, ZX], vec![L, X, Y, Z, X]),
            (vec![RY, Z, Y], vec![R, Y, Z, Y]),
            (vec![R, YZ, Y], vec![R, Y, Z, Y]),
        ] {
            let mut st_fused = stage3.start();
            for &token in &fused {
                st_fused.commit_token(token).unwrap();
            }
            let mut st_single = stage3.start();
            for &token in &single {
                st_single.commit_token(token).unwrap();
            }
            assert_eq!(
                st_fused.mask(),
                st_single.mask(),
                "fused/single mask equivalence for {fused:?} vs {single:?}",
            );
            assert_eq!(
                st_fused.is_accepting(),
                st_single.is_accepting(),
                "fused/single acceptance equivalence for {fused:?} vs {single:?}",
            );
        }
        // Both parents genuinely reach the shared subtree on both backends.
        for path in [vec![L, X, Y, Z, X], vec![R, Y, Z, Y]] {
            let mut st = stage3_dyn.start();
            for &token in &path {
                st.commit_token(token).unwrap();
            }
            assert!(st.is_accepting(), "oracle must accept {path:?}");
            let mut st = stage3.start();
            for &token in &path {
                st.commit_token(token).unwrap();
            }
            assert!(st.is_accepting(), "static must accept {path:?}");
        }
        // Trap-armed replay of the oracle-gated corpus.
        let mut nodes: Vec<(Vec<u32>, Vec<u32>)> = Vec::new();
        for path in &paths {
            let mut st = stage3_dyn.start();
            let mut ok = true;
            for &token in path {
                if !admits_token(&st.mask(), token) || st.commit_token(token).is_err() {
                    ok = false;
                    break;
                }
            }
            if ok {
                nodes.push((path.clone(), st.mask()));
            }
        }
        assert!(
            nodes.len() >= 10,
            "shared-subtree corpus must cover real positions, got {}",
            nodes.len()
        );
        let mut trap = StrictStaticTrapGuard::arm();
        let mut mismatches = 0usize;
        for (path, mask) in &nodes {
            let mut st = stage3.start();
            for &token in path {
                st.commit_token(token).expect("static replay");
            }
            if st.mask() != *mask {
                mismatches += 1;
            }
        }
        trap.disarm();
        assert_eq!(
            mismatches, 0,
            "shared-subtree static must match dynamic on every path"
        );
        // Ablation: clearing the installed top-level shards must lose
        // admissions.
        let mut ablated = stage3.clone();
        install_published_static_boundary_shards(
            ablated.static_dynamic_overlay.as_mut().expect("overlay"),
            Vec::new(),
        )
        .expect("clear shards");
        let mut strict = 0usize;
        for path in &paths {
            let mut dyn_st = stage3_dyn.start();
            let mut abl_st = ablated.start();
            for &token in path {
                if !admits_token(&dyn_st.mask(), token) {
                    break;
                }
                dyn_st.commit_token(token).unwrap();
                if !admits_token(&abl_st.mask(), token) {
                    strict += 1;
                    break;
                }
                abl_st.commit_token(token).unwrap();
            }
        }
        assert!(strict >= 1, "ablation must lose at least one admission");
    }

    #[test]
    fn composed_parent_extend_with_whitespace_aliases_matches_dynamic() {
        if crate::isolate_environment_test(false) { return; }
        // Same deep-debug-stack headroom as the branching/three-bind fixtures
        // (measured there): 3-level multibyte static prime/mask exceeds the
        // default 2MB.
        std::thread::Builder::new()
            .name("whitespace-aliases".into())
            .stack_size(64 * 1024 * 1024)
            .spawn(composed_parent_whitespace_aliases_body)
            .expect("spawn whitespace fixture worker")
            .join()
            .expect("whitespace fixture worker panicked");
    }

    /// Deep-stack worker body for the whitespace-alias test (see above).
    fn composed_parent_whitespace_aliases_body() {
        // Late bind over alias-bearing overlays: every grammar declares an
        // identical whitespace ignore, so each composed overlay carries REAL
        // `global_terminal_aliases` (child ignores folded into the canonical
        // parent ignore). The static extension must still succeed with
        // dynamic-identical behavior: slots are non-vocabulary placeholder
        // sentinels while alias keys are vocabulary-matching ignore terminals,
        // so no legitimate slot can name an alias key (placeholder validation
        // rejects vocabulary-matching terminals) and resolution takes the
        // interval path. Success alongside provably non-empty aliases proves
        // the interval path narrowly (any alias hit would decline loudly).
        // Multibyte tokens cross CALL/RETURN with whitespace flowing across
        // component boundaries on both sides of the late bind.
        let _env_lock = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        const L: u32 = 0;
        const R: u32 = 1;
        const X: u32 = 2;
        const Y: u32 = 3;
        const Z: u32 = 4;
        const SP: u32 = 5;
        const LX: u32 = 6;
        const XZ: u32 = 7;
        const ZX: u32 = 8;
        const RY: u32 = 9;
        const YY: u32 = 10;
        let vocab = Vocab::new(vec![
            (0, b"L".to_vec()),
            (1, b"R".to_vec()),
            (2, b"x".to_vec()),
            (3, b"y".to_vec()),
            (4, b"z".to_vec()),
            (5, b" ".to_vec()),
            (6, b"Lx".to_vec()),
            (7, b"xz".to_vec()),
            (8, b"zx".to_vec()),
            (9, b"Ry".to_vec()),
            (10, b"yy".to_vec()),
        ]);
        let root = Constraint::from_glrm_grammar(
            r#"
                start document;
                ignore WS;
                t SUB_A ::= @token(999);
                t SUB_B ::= @token(998);
                t WS ::= " "+;
                nt document ::= "L" SUB_A "x" | "R" SUB_B "y";
            "#,
            &vocab,
        )
        .unwrap();
        let child_a = Constraint::from_glrm_grammar(
            r#"
                start a;
                ignore WS;
                t SUB_INNER ::= @token(996);
                t WS ::= " "+;
                nt a ::= "x" SUB_INNER;
            "#,
            &vocab,
        )
        .unwrap();
        let child_b = Constraint::from_glrm_grammar(
            r#"
                start b;
                ignore WS;
                t WS ::= " "+;
                nt b ::= "y";
            "#,
            &vocab,
        )
        .unwrap();
        let child_c = Constraint::from_glrm_grammar(
            r#"
                start c;
                ignore WS;
                t WS ::= " "+;
                nt c ::= "z";
            "#,
            &vocab,
        )
        .unwrap();
        for (name, constraint) in
            [("root", &root), ("A", &child_a), ("B", &child_b), ("C", &child_c)]
        {
            assert!(
                constraint.ignore_terminal.is_some(),
                "{name} must carry a real ignore terminal",
            );
        }
        let bind1a = [CompiledSubgrammarInput {
            placeholder_terminal: terminal(&root, "SUB_A"),
            additional_placeholder_terminals: &[],
            constraint: &child_a,
        }];
        let stage1a = compose_constraints_owned_parent_segmented(
            root.clone(),
            &bind1a,
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .expect("bind1a")
        .constraint;
        let stage1a_dyn = compose_constraints_owned_parent_segmented(
            root.clone(),
            &bind1a,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("bind1a dynamic")
        .constraint;
        // bind1a's overlay carries a REAL alias: A's whitespace folded into
        // the canonical root whitespace (exact producer math).
        let canonical_a = terminal(&root, "WS");
        {
            let overlay = stage1a
                .static_dynamic_overlay
                .as_ref()
                .expect("stage1a segmented overlay");
            assert_eq!(overlay.segmented_parser_components.len(), 2);
            assert!(
                overlay.segmented_parser_components[0]
                    .global_terminal_aliases
                    .is_empty(),
                "canonical component carries no alias entries",
            );
            assert_eq!(
                overlay.segmented_parser_components[1].global_terminal_aliases,
                vec![(canonical_a, terminal(&child_a, "WS"))],
                "child ignore must fold into the canonical parent ignore",
            );
            // The alias key matches vocabulary (whitespace) while the bound
            // slot is a non-vocabulary sentinel: no overlap by construction.
            assert!(
                stage1a
                    .possible_matches
                    .get(&canonical_a)
                    .is_some_and(|weight| !weight.is_empty()),
                "canonical ignore must match vocabulary",
            );
            assert_ne!(
                bind1a[0].placeholder_terminal, canonical_a,
                "bound slot must not name the alias key",
            );
        }
        let bind1b = [CompiledSubgrammarInput {
            placeholder_terminal: terminal(&stage1a, "SUB_B"),
            additional_placeholder_terminals: &[],
            constraint: &child_b,
        }];
        let stage1 = compose_constraints_owned_parent_segmented(
            stage1a.clone(),
            &bind1b,
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .expect("bind1b")
        .constraint;
        let stage1_dyn = compose_constraints_owned_parent_segmented(
            stage1a_dyn,
            &bind1b,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("bind1b dynamic")
        .constraint;
        // bind1b's overlay likewise folds B's whitespace into the same
        // canonical ignore (component 0 starts at terminal 0).
        {
            let overlay = stage1
                .static_dynamic_overlay
                .as_ref()
                .expect("stage1 segmented overlay");
            assert_eq!(
                overlay.segmented_parser_components[1].global_terminal_aliases,
                vec![(canonical_a, terminal(&child_b, "WS"))],
                "second child ignore must fold into the same canonical ignore",
            );
            assert_ne!(
                bind1b[0].placeholder_terminal, canonical_a,
                "bound slot must not name the alias key",
            );
        }
        let sub_inner = {
            let overlay = stage1.static_dynamic_overlay.as_ref().expect("overlay");
            let nested = &overlay.segmented_parser_components[0];
            let nested_overlay = nested
                .constraint
                .static_dynamic_overlay
                .as_ref()
                .expect("nested overlay");
            nested.terminal_offset
                + nested_overlay.segmented_parser_components[1].terminal_offset
                + terminal(&child_a, "SUB_INNER")
        };
        // The late-bind slot (in stage1's composed coordinate) also cannot
        // name the alias key: non-vocabulary sentinel vs whitespace.
        assert_ne!(
            sub_inner, canonical_a,
            "late-bind slot must not name the alias key",
        );
        let bind2 = [CompiledSubgrammarInput {
            placeholder_terminal: sub_inner,
            additional_placeholder_terminals: &[],
            constraint: &child_c,
        }];
        let stage2 = compose_constraints_owned_parent_segmented(
            stage1.clone(),
            &bind2,
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .expect("bind2 over alias-bearing parent must extend statically")
        .constraint;
        let stage2_dyn = compose_constraints_owned_parent_segmented(
            stage1_dyn,
            &bind2,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("bind2 dynamic")
        .constraint;
        // Static shard authority on the extended artifact.
        {
            let overlay = stage2
                .static_dynamic_overlay
                .as_ref()
                .expect("static overlay");
            assert_eq!(overlay.segmented_parser_components.len(), 2);
            for (index, component) in overlay.segmented_parser_components.iter().enumerate() {
                let shard = component.boundary.as_ref().unwrap_or_else(|| {
                    panic!("installed top component {index} must carry a static shard")
                });
                assert!(
                    matches!(
                        shard.backend,
                        crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_)
                    ),
                    "installed component {index} must be a StaticParser shard",
                );
            }
            assert!(
                overlay.segmented_boundary_parser.is_none(),
                "extended composition must not retain a redundant global boundary parser",
            );
        }
        // Byte-prefix differential with whitespace flowing across boundaries.
        let prefixes: Vec<Vec<u8>> = vec![
            vec![],
            vec![b'L'],
            vec![b'L', b'x'],
            vec![b'L', b'x', b'z'],
            vec![b'L', b'x', b'z', b'x'],
            vec![b'R'],
            vec![b'R', b'y'],
            vec![b'R', b'y', b'y'],
            vec![b' '],
            vec![b'L', b' '],
            vec![b'L', b' ', b'x'],
            vec![b'L', b'x', b' ', b'z', b'x'],
            vec![b'L', b'x', b'z', b' ', b'x'],
            vec![b'R', b' ', b'y', b'y'],
            vec![b'L', b'y'],
            vec![b'z'],
        ];
        assert_composed_masks_equal("ws-alias-bytes", &stage2, &stage2_dyn, &prefixes);
        // Token-level differential (multibyte crossings + whitespace tokens).
        let paths: Vec<Vec<u32>> = vec![
            vec![],
            vec![L],
            vec![R],
            vec![LX],
            vec![RY],
            vec![SP],
            vec![L, X],
            vec![L, SP, X],
            vec![LX, SP],
            vec![L, X, Z],
            vec![L, XZ],
            vec![LX, Z],
            vec![LX, ZX],
            vec![L, X, Z, X],
            vec![LX, SP, ZX],
            vec![L, SP, X, SP, Z, X],
            vec![R, Y],
            vec![RY, Y],
            vec![R, YY],
            vec![R, Y, Y],
            vec![R, SP, Y, Y],
            vec![L, Y],
            vec![R, X],
            vec![Z],
        ];
        assert_composed_token_masks_equal("ws-alias-tokens", &stage2, &stage2_dyn, &paths);
        // Fused/single equivalence (whitespace-free pairs, proven pattern).
        for (fused, single) in [
            (vec![LX], vec![L, X]),
            (vec![RY], vec![R, Y]),
            (vec![L, XZ], vec![L, X, Z]),
            (vec![LX, ZX], vec![L, X, Z, X]),
            (vec![R, YY], vec![R, Y, Y]),
        ] {
            let mut st_fused = stage2.start();
            for &token in &fused {
                st_fused.commit_token(token).unwrap();
            }
            let mut st_single = stage2.start();
            for &token in &single {
                st_single.commit_token(token).unwrap();
            }
            assert_eq!(
                st_fused.mask(),
                st_single.mask(),
                "fused/single mask equivalence for {fused:?} vs {single:?}",
            );
            assert_eq!(
                st_fused.is_accepting(),
                st_single.is_accepting(),
                "fused/single acceptance equivalence for {fused:?} vs {single:?}",
            );
        }
        // Whitespace genuinely flows: the oracle accepts spaced completions.
        for path in [vec![L, SP, X, Z, X], vec![R, SP, Y, Y]] {
            let mut st = stage2_dyn.start();
            for &token in &path {
                st.commit_token(token).unwrap();
            }
            assert!(st.is_accepting(), "oracle must accept spaced {path:?}");
        }
        // Trap-armed replay of the oracle-gated corpus.
        let mut nodes: Vec<(Vec<u32>, Vec<u32>)> = Vec::new();
        for path in &paths {
            let mut st = stage2_dyn.start();
            let mut ok = true;
            for &token in path {
                if !admits_token(&st.mask(), token) || st.commit_token(token).is_err() {
                    ok = false;
                    break;
                }
            }
            if ok {
                nodes.push((path.clone(), st.mask()));
            }
        }
        assert!(
            nodes.len() >= 10,
            "whitespace corpus must cover real positions, got {}",
            nodes.len()
        );
        let mut trap = StrictStaticTrapGuard::arm();
        let mut mismatches = 0usize;
        for (path, mask) in &nodes {
            let mut st = stage2.start();
            for &token in path {
                st.commit_token(token).expect("static replay");
            }
            if st.mask() != *mask {
                mismatches += 1;
            }
        }
        trap.disarm();
        assert_eq!(mismatches, 0, "whitespace static must match dynamic on every path");
        // Ablation: clearing the installed top-level shards must lose
        // admissions.
        let mut ablated = stage2.clone();
        install_published_static_boundary_shards(
            ablated.static_dynamic_overlay.as_mut().expect("overlay"),
            Vec::new(),
        )
        .expect("clear shards");
        let mut strict = 0usize;
        for path in &paths {
            let mut dyn_st = stage2_dyn.start();
            let mut abl_st = ablated.start();
            for &token in path {
                if !admits_token(&dyn_st.mask(), token) {
                    break;
                }
                dyn_st.commit_token(token).unwrap();
                if !admits_token(&abl_st.mask(), token) {
                    strict += 1;
                    break;
                }
                abl_st.commit_token(token).unwrap();
            }
        }
        assert!(strict >= 1, "ablation must lose at least one admission");
    }

    #[test]
    fn composed_parent_static_extend_with_reused_sibling() {
        // DAG reuse: the same child constraint fills slots on both binds. Leaf
        // expansion keys visits by position (never a global seen set), so the
        // shared child expands once per use with distinct leaves.
        let _env_lock = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let vocab = Vocab::new(vec![
            (0, b"L".to_vec()),
            (1, b"R".to_vec()),
            (2, b"x".to_vec()),
            (3, b"y".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                t SUB2 ::= @token(998);
                nt document ::= "L" SUB "x" | "R" SUB2 "y";
            "#,
            &vocab,
        )
        .unwrap();
        let child_a = Constraint::from_glrm_grammar(
            r#"
                start a;
                nt a ::= "x";
            "#,
            &vocab,
        )
        .unwrap();
        let bind1 = [CompiledSubgrammarInput {
            placeholder_terminal: terminal(&parent, "SUB"),
            additional_placeholder_terminals: &[],
            constraint: &child_a,
        }];
        let stage1 = compose_constraints_owned_parent_segmented(
            parent.clone(),
            &bind1,
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .expect("bind1")
        .constraint;
        let bind2 = [CompiledSubgrammarInput {
            placeholder_terminal: terminal(&stage1, "SUB2"),
            additional_placeholder_terminals: &[],
            constraint: &child_a,
        }];
        let expansion = crate::compiler::composition::boundary::walk::expand_nested_link_leaves(&stage1, &bind2)
            .expect("bind2 expansion");
        assert_eq!(expansion.leaves.len(), 3, "root plus two uses of the shared child");
        assert_eq!(
            expansion.top_leaf_ranges,
            vec![vec![0usize, 1usize], vec![2usize]],
            "each use of the shared child must occupy its own leaf (no identity dedup)",
        );
        assert_eq!(
            expansion.leaves[1].table.num_terminals,
            expansion.leaves[2].table.num_terminals,
            "both uses expand the same reassembled child table",
        );
        let stage2 = compose_constraints_owned_parent_segmented(
            stage1.clone(),
            &bind2,
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .expect("bind2")
        .constraint;
        let stage1_dyn = compose_constraints_owned_parent_segmented(
            parent,
            &bind1,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("bind1 dynamic")
        .constraint;
        let stage2_dyn = compose_constraints_owned_parent_segmented(
            stage1_dyn,
            &bind2,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("bind2 dynamic")
        .constraint;
        let prefixes: Vec<Vec<u8>> = vec![
            vec![],
            vec![b'L'],
            vec![b'L', b'x'],
            vec![b'L', b'x', b'x'],
            vec![b'R'],
            vec![b'R', b'x'],
            vec![b'R', b'x', b'y'],
            vec![b'L', b'y'],
            vec![b'R', b'y'],
        ];
        assert_composed_masks_equal("reused-sibling", &stage2, &stage2_dyn, &prefixes);
    }

    #[test]
    fn composed_parent_static_extend_after_serialization_round_trip() {
        if crate::isolate_environment_test(false) { return; }
        // Components and links survive save/load, so a loaded composed parent
        // extends statically with identical results. Multibyte tokens cross
        // CALL (Lx, Ry) and RETURN (xx, yy) on both the pre- and post-serde
        // links.
        let _env_lock = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        const L: u32 = 0;
        const R: u32 = 1;
        const X: u32 = 2;
        const Y: u32 = 3;
        const LX: u32 = 4;
        const XX: u32 = 5;
        const RY: u32 = 6;
        const YY: u32 = 7;
        let vocab = Vocab::new(vec![
            (0, b"L".to_vec()),
            (1, b"R".to_vec()),
            (2, b"x".to_vec()),
            (3, b"y".to_vec()),
            (4, b"Lx".to_vec()),
            (5, b"xx".to_vec()),
            (6, b"Ry".to_vec()),
            (7, b"yy".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                t SUB2 ::= @token(998);
                nt document ::= "L" SUB "x" | "R" SUB2 "y";
            "#,
            &vocab,
        )
        .unwrap();
        let child_a = Constraint::from_glrm_grammar(
            r#"
                start a;
                nt a ::= "x";
            "#,
            &vocab,
        )
        .unwrap();
        let child_b = Constraint::from_glrm_grammar(
            r#"
                start b;
                nt b ::= "y";
            "#,
            &vocab,
        )
        .unwrap();
        let bind1 = [CompiledSubgrammarInput {
            placeholder_terminal: terminal(&parent, "SUB"),
            additional_placeholder_terminals: &[],
            constraint: &child_a,
        }];
        let stage1 = compose_constraints_owned_parent_segmented(
            parent,
            &bind1,
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .expect("bind1")
        .constraint;
        let bytes = stage1.save();
        let loaded = Constraint::load(bytes.as_slice()).expect("load composed parent");
        assert!(
            loaded.has_recursive_segmented_parser_tree(),
            "loaded parent must retain its segmented tree",
        );
        // The loaded overlay must carry bind1's link intact (slot, child, and
        // alias-free components), since the static extend resolves slots
        // through it.
        {
            let overlay = loaded
                .static_dynamic_overlay
                .as_ref()
                .expect("loaded segmented overlay");
            assert_eq!(
                overlay.segmented_parser_components.len(),
                2,
                "loaded parent keeps [root, A]",
            );
            let [link] = overlay.segmented_parser_links.as_slice() else {
                panic!(
                    "loaded parent must retain bind1's single link, got {:?}",
                    overlay.segmented_parser_links.len(),
                );
            };
            assert_eq!(link.parent_component, 0);
            assert_eq!(link.slot_terminal, bind1[0].placeholder_terminal);
            assert_eq!(link.child_component, 1);
            assert!(
                overlay
                    .segmented_parser_components
                    .iter()
                    .all(|component| component.global_terminal_aliases.is_empty()),
                "loaded components must stay alias-free",
            );
        }
        let bind2 = [CompiledSubgrammarInput {
            placeholder_terminal: terminal(&loaded, "SUB2"),
            additional_placeholder_terminals: &[],
            constraint: &child_b,
        }];
        let stage2 = compose_constraints_owned_parent_segmented(
            loaded.clone(),
            &bind2,
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .expect("static extend after load")
        .constraint;
        let stage2_dyn = compose_constraints_owned_parent_segmented(
            loaded,
            &bind2,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("dynamic extend after load")
        .constraint;
        // Static shard authority on the post-serde extend: both tops hold
        // StaticParser shards and no redundant global boundary parser. The
        // loaded parent block retains its own static inner repair.
        {
            let overlay = stage2
                .static_dynamic_overlay
                .as_ref()
                .expect("static overlay");
            assert_eq!(overlay.segmented_parser_components.len(), 2);
            for (index, component) in overlay.segmented_parser_components.iter().enumerate() {
                let shard = component.boundary.as_ref().unwrap_or_else(|| {
                    panic!("installed top component {index} must carry a static shard")
                });
                assert!(
                    matches!(
                        shard.backend,
                        crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_)
                    ),
                    "installed component {index} must be a StaticParser shard",
                );
            }
            assert!(
                overlay.segmented_boundary_parser.is_none(),
                "extended composition must not retain a redundant global boundary parser",
            );
            let inner = overlay.segmented_parser_components[0]
                .constraint
                .static_dynamic_overlay
                .as_ref()
                .expect("parent-block overlay");
            assert!(
                inner
                    .segmented_parser_components
                    .iter()
                    .all(|component| component.boundary.as_ref().is_none_or(|shard| matches!(
                        shard.backend,
                        crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_)
                    ))),
                "parent-block overlay must retain only static inner shards",
            );
            assert!(
                !inner.segmented_boundary_shards.is_empty()
                    && inner.segmented_boundary_shards.iter().all(|shard| matches!(
                        shard.backend,
                        crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_)
                    )),
                "parent-block overlay static shard list must survive the post-serde extension",
            );
        }
        // Byte-prefix differential (single-byte commits).
        let prefixes: Vec<Vec<u8>> = vec![
            vec![],
            vec![b'L'],
            vec![b'L', b'x'],
            vec![b'L', b'x', b'x'],
            vec![b'R'],
            vec![b'R', b'y'],
            vec![b'R', b'y', b'y'],
            vec![b'L', b'y'],
        ];
        assert_composed_masks_equal("post-serde-extend-bytes", &stage2, &stage2_dyn, &prefixes);
        // Token-level differential (whole multibyte commits across CALL/RETURN).
        let paths: Vec<Vec<u32>> = vec![
            vec![],
            vec![L],
            vec![R],
            vec![LX],
            vec![RY],
            vec![L, X],
            vec![L, XX],
            vec![LX, X],
            vec![L, X, X],
            vec![R, Y],
            vec![RY, Y],
            vec![R, YY],
            vec![R, Y, Y],
            vec![L, Y],
            vec![R, X],
            vec![X],
        ];
        assert_composed_token_masks_equal("post-serde-extend-tokens", &stage2, &stage2_dyn, &paths);
        // Fused/single equivalence over the extended (deeper) artifact.
        for (fused, single) in [
            (vec![LX], vec![L, X]),
            (vec![RY], vec![R, Y]),
            (vec![L, XX], vec![L, X, X]),
            (vec![LX, X], vec![L, X, X]),
            (vec![R, YY], vec![R, Y, Y]),
        ] {
            let mut st_fused = stage2.start();
            for &token in &fused {
                st_fused.commit_token(token).unwrap();
            }
            let mut st_single = stage2.start();
            for &token in &single {
                st_single.commit_token(token).unwrap();
            }
            assert_eq!(
                st_fused.mask(),
                st_single.mask(),
                "fused/single mask equivalence for {fused:?} vs {single:?}",
            );
            assert_eq!(
                st_fused.is_accepting(),
                st_single.is_accepting(),
                "fused/single acceptance equivalence for {fused:?} vs {single:?}",
            );
        }
        assert!(admits_token(&stage2.start().mask(), LX), "Lx crossing must be live");
        assert!(admits_token(&stage2.start().mask(), RY), "Ry crossing must be live");
        // Trap-armed replay of the oracle-gated corpus.
        let mut nodes: Vec<(Vec<u32>, Vec<u32>)> = Vec::new();
        for path in &paths {
            let mut st = stage2_dyn.start();
            let mut ok = true;
            for &token in path {
                if !admits_token(&st.mask(), token) || st.commit_token(token).is_err() {
                    ok = false;
                    break;
                }
            }
            if ok {
                nodes.push((path.clone(), st.mask()));
            }
        }
        assert!(
            nodes.len() >= 10,
            "post-serde corpus must cover real positions, got {}",
            nodes.len()
        );
        let mut trap = StrictStaticTrapGuard::arm();
        let mut mismatches = 0usize;
        for (path, mask) in &nodes {
            let mut st = stage2.start();
            for &token in path {
                st.commit_token(token).expect("static replay");
            }
            if st.mask() != *mask {
                mismatches += 1;
            }
        }
        trap.disarm();
        assert_eq!(mismatches, 0, "post-serde static must match dynamic on every path");
        // Ablation: clearing the installed top-level shards must lose
        // admissions.
        let mut ablated = stage2.clone();
        install_published_static_boundary_shards(
            ablated.static_dynamic_overlay.as_mut().expect("overlay"),
            Vec::new(),
        )
        .expect("clear shards");
        let mut strict = 0usize;
        for path in &paths {
            let mut dyn_st = stage2_dyn.start();
            let mut abl_st = ablated.start();
            for &token in path {
                if !admits_token(&dyn_st.mask(), token) {
                    break;
                }
                dyn_st.commit_token(token).unwrap();
                if !admits_token(&abl_st.mask(), token) {
                    strict += 1;
                    break;
                }
                abl_st.commit_token(token).unwrap();
            }
        }
        assert!(strict >= 1, "ablation must lose at least one admission");
        // The extended artifact itself round-trips with identical behavior.
        let reloaded =
            Constraint::load(stage2.save().as_slice()).expect("reload extended composition");
        assert_composed_masks_equal("post-serde-reload-bytes", &reloaded, &stage2_dyn, &prefixes);
        assert_composed_token_masks_equal(
            "post-serde-reload-tokens",
            &reloaded,
            &stage2_dyn,
            &paths,
        );
    }

    #[test]
    fn composed_parent_static_extend_pins_runtime_leaf_layout() {
        // The installing runtime must visit the extended link's leaves in the
        // walk's preorder (parent block first): pin terminal offsets exactly
        // and tokenizer offsets structurally against the leaf expansion.
        let _env_lock = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let vocab = Vocab::new(vec![
            (0, b"L".to_vec()),
            (1, b"R".to_vec()),
            (2, b"x".to_vec()),
            (3, b"y".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                t SUB2 ::= @token(998);
                nt document ::= "L" SUB "x" | "R" SUB2 "y";
            "#,
            &vocab,
        )
        .unwrap();
        let child_a = Constraint::from_glrm_grammar(
            r#"
                start a;
                nt a ::= "x";
            "#,
            &vocab,
        )
        .unwrap();
        let child_b = Constraint::from_glrm_grammar(
            r#"
                start b;
                nt b ::= "y";
            "#,
            &vocab,
        )
        .unwrap();
        let bind1 = [CompiledSubgrammarInput {
            placeholder_terminal: terminal(&parent, "SUB"),
            additional_placeholder_terminals: &[],
            constraint: &child_a,
        }];
        let stage1 = compose_constraints_owned_parent_segmented(
            parent,
            &bind1,
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .expect("bind1")
        .constraint;
        let bind2 = [CompiledSubgrammarInput {
            placeholder_terminal: terminal(&stage1, "SUB2"),
            additional_placeholder_terminals: &[],
            constraint: &child_b,
        }];
        let stage2 = compose_constraints_owned_parent_segmented(
            stage1.clone(),
            &bind2,
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .expect("bind2")
        .constraint;
        let expansion = crate::compiler::composition::boundary::walk::expand_nested_link_leaves(&stage1, &bind2)
            .expect("bind2 expansion");
        let layout = stage2
            .recursive_parser_layout()
            .expect("layout result")
            .expect("recursive layout present");
        assert_eq!(
            layout.leaf_terminal_offsets, expansion.leaf_terminal_offsets,
            "runtime terminal preorder must match the walk leaf order",
        );
        assert_eq!(
            layout.total_leaf_terminals, expansion.num_terminals,
            "runtime terminal total must match the walk domain",
        );
        assert_eq!(
            layout.leaf_tokenizer_state_offsets.len(),
            expansion.leaves.len(),
            "one tokenizer block per leaf",
        );
        assert!(
            layout
                .leaf_tokenizer_state_offsets
                .windows(2)
                .all(|pair| pair[0] < pair[1]),
            "tokenizer blocks must pack back-to-back in leaf order: {:?}",
            layout.leaf_tokenizer_state_offsets,
        );
        assert!(
            layout.total_tokenizer_states
                > *layout.leaf_tokenizer_state_offsets.last().unwrap_or(&0),
            "tokenizer total must cover the packed blocks",
        );
    }

    #[test]
    fn composed_parent_static_extend_with_nullable_child_declines_loudly() {
        // Extending a composed parent must not smuggle in deferred nullable
        // support: a nullable second child declines exactly like the flat case.
        let _env_lock = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let vocab = Vocab::new(vec![
            (0, b"L".to_vec()),
            (1, b"R".to_vec()),
            (2, b"x".to_vec()),
            (3, b"y".to_vec()),
            (4, b"a".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                t SUB2 ::= @token(998);
                nt document ::= "L" SUB "x" | "R" SUB2 "y";
            "#,
            &vocab,
        )
        .unwrap();
        let child_a = Constraint::from_glrm_grammar(
            r#"
                start a;
                nt a ::= "x";
            "#,
            &vocab,
        )
        .unwrap();
        let nullable_b = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt item ::= "a";
                nt child ::= item?;
            "#,
            &vocab,
        )
        .unwrap();
        assert!(
            nullable_b.table.embedded_start_nullable(),
            "pin fixture must stay effectively nullable",
        );
        let bind1 = [CompiledSubgrammarInput {
            placeholder_terminal: terminal(&parent, "SUB"),
            additional_placeholder_terminals: &[],
            constraint: &child_a,
        }];
        let stage1 = compose_constraints_owned_parent_segmented(
            parent,
            &bind1,
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .expect("bind1")
        .constraint;
        let bind2 = [CompiledSubgrammarInput {
            placeholder_terminal: terminal(&stage1, "SUB2"),
            additional_placeholder_terminals: &[],
            constraint: &nullable_b,
        }];
        let error = match compose_constraints_owned_parent_segmented(
            stage1,
            &bind2,
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        ) {
            Err(error) => error,
            Ok(_) => panic!("nullable second bind must decline loudly"),
        };
        assert!(
            error.contains("nullable"),
            "decline must name nullability, got: {error}",
        );
    }

    #[test]
    fn internal_composition_matches_monolithic_for_child_alternatives() {
        let vocab = Vocab::new(vec![
            (0, b"Xa".to_vec()),
            (1, b"b!".to_vec()),
            (2, b"Xc".to_vec()),
            (3, b"d!".to_vec()),
            (4, b"X".to_vec()),
            (5, b"!".to_vec()),
            (6, b"a".to_vec()),
            (7, b"b".to_vec()),
            (8, b"c".to_vec()),
            (9, b"d".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t LEFT ::= @token(998);
                t RIGHT ::= @token(999);
                nt document ::= "X" (LEFT | RIGHT) "!";
            "#,
            &vocab,
        )
        .unwrap();
        let left = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child ::= "a" "b";
            "#,
            &vocab,
        )
        .unwrap();
        let right = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child ::= "c" "d";
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                g left ::= {
                    start child;
                    nt child ::= "a" "b";
                };
                g right ::= {
                    start child;
                    nt child ::= "c" "d";
                };
                nt document ::= "X" (left | right) "!";
            "#,
            &vocab,
        )
        .unwrap();
        let duplicate_error = parent
            .compose_linked_children_for_test(&[("LEFT", &left), ("LEFT", &right)], &vocab)
            .expect_err("duplicate placeholder inputs must be rejected");
        assert!(duplicate_error
            .to_string()
            .contains("was supplied more than once"));
        let composed = parent
            .compose_linked_children_for_test(&[("LEFT", &left), ("RIGHT", &right)], &vocab)
            .unwrap();
        let loaded = Constraint::load(&composed.save())
            .expect("composed constraints must survive serialization");

        for sequence in [[0, 1], [2, 3]] {
            let mut expected = monolithic.start();
            let mut actual = composed.start();
            let mut roundtripped = loaded.start();
            for token in sequence {
                assert_eq!(actual.mask(), expected.mask(), "mask mismatch before token {token}");
                assert_eq!(
                    roundtripped.mask(),
                    expected.mask(),
                    "round-tripped mask mismatch before token {token}",
                );
                actual.commit_token(token).unwrap();
                roundtripped.commit_token(token).unwrap();
                expected.commit_token(token).unwrap();
            }
            assert!(actual.is_accepting());
            assert!(roundtripped.is_accepting());
            assert!(expected.is_accepting());
        }

        let mut crossed = composed.start();
        crossed.commit_token(0).unwrap();
        assert!(crossed.commit_token(3).is_err() || !crossed.is_accepting());
    }

    fn sizeable_json_schema(prefix: &str, choices: usize) -> String {
        let choices = (0..choices)
            .map(|index| format!(r#""{prefix}_choice_{index:04}_long_literal_value""#))
            .collect::<Vec<_>>()
            .join(",");
        format!(
            r#"{{"type":"object","additionalProperties":false,"properties":{{"choice":{{"type":"string","enum":[{choices}]}},"payload":{{"type":"string","minLength":1,"maxLength":32}}}},"required":["choice","payload"]}}"#,
        )
    }

    fn composition_benchmark_vocab() -> Vocab {
        let mut entries = (0u32..=255)
            .map(|byte| (byte, vec![byte as u8]))
            .collect::<Vec<_>>();
        entries.extend([
            (256, b"\": {".to_vec()),
            (257, b"}, \"value2\": {".to_vec()),
            (258, b"}}".to_vec()),
            (259, b"{\"value\": {".to_vec()),
        ]);
        Vocab::new(entries)
    }

    fn run_sizeable_json_schema_composition_benchmark(mode: &str, parent_source: &str) {
        let vocab = composition_benchmark_vocab();
        let choices = std::env::var("GLRMASK_COMPOSE_BENCH_CHOICES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(7500);
        let left_schema = sizeable_json_schema("left", choices);
        let right_schema = sizeable_json_schema("right", choices);

        eprintln!("[compose-bench] compile {mode} parent start");
        let parent = Constraint::from_glrm_grammar(parent_source, &vocab).unwrap();
        eprintln!("[compose-bench] compile {mode} parent done");

        let left_started = Instant::now();
        eprintln!("[compose-bench] compile left start");
        let left = Constraint::from_json_schema(&left_schema, &vocab).unwrap();
        let left_ms = left_started.elapsed().as_secs_f64() * 1000.0;
        eprintln!("[compose-bench] compile left done {left_ms:.3} ms");
        let right_started = Instant::now();
        eprintln!("[compose-bench] compile right start");
        let right = Constraint::from_json_schema(&right_schema, &vocab).unwrap();
        let right_ms = right_started.elapsed().as_secs_f64() * 1000.0;
        eprintln!("[compose-bench] compile right done {right_ms:.3} ms");

        let composition_runs = std::env::var("GLRMASK_COMPOSE_BENCH_RUNS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(3)
            .max(1);
        let mut composition_samples_ms = Vec::with_capacity(composition_runs);
        for run in 0..composition_runs {
            let composition_started = Instant::now();
            eprintln!("[compose-bench] compose {mode} run={} start", run + 1);
            let composed = parent
                .compose_linked_children_for_test(&[("LEFT", &left), ("RIGHT", &right)], &vocab)
                .unwrap();
            let composition_ms = composition_started.elapsed().as_secs_f64() * 1000.0;
            eprintln!(
                "[compose-bench] compose {mode} run={} done {composition_ms:.3} ms",
                run + 1,
            );
            std::hint::black_box(composed);
            composition_samples_ms.push(composition_ms);
        }
        composition_samples_ms.sort_by(f64::total_cmp);
        let composition_ms = composition_samples_ms[composition_samples_ms.len() / 2];
        let composition_min_ms = composition_samples_ms[0];
        let composition_max_ms = *composition_samples_ms.last().unwrap();

        let child_sum_ms = left_ms + right_ms;
        eprintln!(
            "sizeable JSON composition: mode={mode} left_ms={left_ms:.3} right_ms={right_ms:.3} child_sum_ms={child_sum_ms:.3} composition_runs={} composition_median_ms={composition_ms:.3} composition_min_ms={composition_min_ms:.3} composition_max_ms={composition_max_ms:.3}",
            composition_samples_ms.len(),
        );
        assert!(
            composition_ms * 20.0 < child_sum_ms,
            "composition should be at least 20x faster than rebuilding both children: child_sum_ms={child_sum_ms:.3}, composition_ms={composition_ms:.3}",
        );
        if !cfg!(debug_assertions) {
            assert!(
                composition_ms < 20.0,
                "optimized composition should remain below 20 ms: {composition_ms:.3} ms",
            );
        }
    }

    #[test]
    #[ignore]
    fn explicit_scoped_control_runtime_benchmark_probe() {
        let vocab = Vocab::new(vec![
            (0, b"X".to_vec()),
            (1, b" ".to_vec()),
            (2, b"\t".to_vec()),
            (3, b"a".to_vec()),
            (4, b"!".to_vec()),
            (5, b"X \ta\t !".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                ignore PARENT_WS;
                t PARENT_WS ::= " "+;
                t SUB ::= @token(999);
                nt document ::= "X" SUB "!";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                ignore CHILD_WS;
                t CHILD_WS ::= "\t"+;
                nt child ::= "a";
            "#,
            &vocab,
        )
        .unwrap();
        let composed = parent
            .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
            .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                ignore PARENT_WS;
                t PARENT_WS ::= " "+;
                g child ::= {
                    start child;
                    ignore CHILD_WS;
                    t CHILD_WS ::= "\t"+;
                    nt child ::= "a";
                };
                nt document ::= "X" child "!";
            "#,
            &vocab,
        )
        .unwrap();

        let runs = std::env::var("GLRMASK_COMPOSE_RUNTIME_BENCH_RUNS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(10_000)
            .max(1);
        let split = [0u32, 1, 2, 3, 2, 1, 4];
        let measure = |constraint: &Constraint, sequence: &[u32]| {
            let mut samples = Vec::with_capacity(runs);
            for _ in 0..runs {
                let mut state = constraint.start();
                let mut total = 0u64;
                for &token in sequence {
                    total += state.commit_token_timed_ns(token).unwrap();
                }
                assert!(state.is_accepting());
                samples.push(total);
            }
            samples.sort_unstable();
            samples[samples.len() / 2]
        };
        let composed_split = measure(&composed, &split);
        let monolithic_split = measure(&monolithic, &split);
        let composed_fused = measure(&composed, &[5]);
        let monolithic_fused = measure(&monolithic, &[5]);
        let mut profiled = composed.start();
        let fused_profile = profiled.commit_token_profiled(5).unwrap();
        eprintln!(
            "explicit scoped runtime: runs={runs} split_composed_ns={composed_split} split_monolithic_ns={monolithic_split} fused_composed_ns={composed_fused} fused_monolithic_ns={monolithic_fused} fused_profile={fused_profile:?}",
        );
        std::hint::black_box((
            composed_split,
            monolithic_split,
            composed_fused,
            monolithic_fused,
        ));
    }

    #[test]
    #[ignore]
    fn sizeable_json_schema_sequential_composition_benchmark_probe() {
        run_sizeable_json_schema_composition_benchmark(
            "sequential",
            r#"
                start document;
                t LEFT ::= @token(1000000);
                t RIGHT ::= @token(1000001);
                nt document ::= "{" "\"value1\": " LEFT ", \"value2\": " RIGHT "}";
            "#,
        );
    }

    #[test]
    #[ignore]
    fn sizeable_json_schema_alternative_composition_benchmark_probe() {
        run_sizeable_json_schema_composition_benchmark(
            "alternative",
            r#"
                start document;
                t LEFT ::= @token(1000000);
                t RIGHT ::= @token(1000001);
                nt document ::= "{" ("\"left\": " LEFT | "\"right\": " RIGHT) "}";
            "#,
        );
    }

    // MINBOUND/oracle helpers hoisted to module scope so the Phase 1 probe
    // (phase1_restricted_walk_selected10) can reuse them. Bodies unchanged.
    fn build_terminal_dwa_parts(
        name: &str,
        tokenizer: &Tokenizer,
        table: &crate::compiler::glr::table::GLRTable,
        terminal_display_names: &[String],
        ignore_terminal: Option<u32>,
        vocab: &Vocab,
    ) -> MappedArtifact<DWA> {
        let started = Instant::now();
        let augmented_start = table
            .rules
            .first()
            .expect("terminal-DWA oracle table has augmented start")
            .lhs;
        let grammar = AnalyzedGrammar::from_composed_rules(
            table.rules.clone(),
            table.num_terminals,
            terminal_display_names.to_vec(),
            table.nonterminal_display_names.clone(),
            augmented_start,
        );
        let disallowed = crate::compiler::pipeline::compute_disallowed_follows(&grammar);
        let flat: Arc<[u32]> = Arc::from(
            crate::compiler::stages::id_map_and_terminal_dwa::l1::build_flat_transition_table(
                tokenizer,
            ),
        );
        // This experiment wants language, not the production global tokenizer-state
        // quotient. Keep the raw tokenizer-state coordinate exact and singleton so we
        // do not pay the large max-length/global-equivalence preparation just to compare
        // terminal languages. Reconciliation below will still put every DWA into one
        // exact common TSID refinement.
        let raw_ids = (0..tokenizer.num_states()).collect::<Vec<_>>();
        let state_map = ManyToOneIdMap::from_singleton_original_to_internal_with_representatives(
            raw_ids.clone(),
            raw_ids,
        );
        let coloring = TerminalColoring::identity(grammar.num_terminals as usize);
        let (artifact, profile) =
            crate::compiler::stages::id_map_and_terminal_dwa::
                build_restricted_id_map_and_terminal_dwa_with_precomputed_global_max_length(
                    tokenizer,
                    vocab,
                    &coloring,
                    false,
                    ignore_terminal,
                    &grammar,
                    &disallowed,
                    flat,
                    &state_map,
                    None,
                    None,
                );
        let (automaton, id_map) = artifact.into_parts();
        let dwa = match automaton {
            TerminalAutomaton::Dwa(dwa) => dwa,
            TerminalAutomaton::TokenDeterministicNwa(nwa)
            | TerminalAutomaton::EpsilonNwa(nwa) => determinize(&nwa).unwrap(),
        };
        eprintln!(
            "MINBOUND terminal_build name={name} states={} transitions={} tsids={} tokens={} disallowed_pairs={} ms={:.3} profile_total_ms={:.3}",
            dwa.num_states(),
            dwa.num_transitions(),
            id_map.num_tsids(),
            id_map.num_internal_tokens(),
            disallowed.values().map(BitSet::count_ones).sum::<usize>(),
            started.elapsed().as_secs_f64() * 1000.0,
            profile.total_ms(),
        );
        MappedArtifact::new(dwa, id_map)
    }

    fn offset_terminal_labels(
        name: &str,
        mut artifact: MappedArtifact<DWA>,
        terminal_offset: u32,
    ) -> MappedArtifact<DWA> {
        for state in artifact.artifact_mut().states_mut() {
            let old = std::mem::take(&mut state.transitions);
            for (&label, edge) in old.iter() {
                let mapped = if label == DEFAULT_LABEL {
                    DEFAULT_LABEL
                } else {
                    assert!(label >= 0, "{name}: unexpected negative terminal label {label}");
                    label + terminal_offset as i32
                };
                assert!(state.transitions.insert(mapped, edge.clone()).is_none());
            }
        }
        artifact
    }

    fn rebase_tokenizer_state_universe(
        name: &str,
        artifact: MappedArtifact<DWA>,
        global_offset: u32,
        enclosing_reset_states: &[u32],
        monolithic_state_count: usize,
    ) -> MappedArtifact<DWA> {
        let (dwa, mut id_map) = artifact.into_parts();
        let local = id_map.tokenizer_states;
        let local_state_count = local.original_to_internal.len();
        let mut original_to_internal = vec![u32::MAX; monolithic_state_count];
        let mut internal_to_originals = vec![Vec::<u32>::new(); local.internal_to_originals.len()];

        for (local_state, &tsid) in local.original_to_internal.iter().enumerate() {
            if tsid == u32::MAX {
                continue;
            }
            let global_state = global_offset
                .checked_add(local_state as u32)
                .expect("terminal-DWA experiment tokenizer-state offset overflow");
            assert!(
                (global_state as usize) < monolithic_state_count,
                "{name}: rebased state {global_state} lies outside monolithic tokenizer ({monolithic_state_count})",
            );
            assert_eq!(
                original_to_internal[global_state as usize],
                u32::MAX,
                "{name}: duplicate local tokenizer-state embedding at global state {global_state}",
            );
            original_to_internal[global_state as usize] = tsid;
            internal_to_originals[tsid as usize].push(global_state);
        }

        // Tokenizers always start at local state zero. A composed reset epsilon-dispatches
        // into each child start state, so the child's start TSID is observable at every
        // enclosing reset coordinate as well as at its physically rebased local state.
        let start_tsid = local.original_to_internal.first().copied().unwrap_or(u32::MAX);
        assert_ne!(start_tsid, u32::MAX, "{name}: tokenizer start state is unmapped");
        for &reset in enclosing_reset_states {
            assert!((reset as usize) < monolithic_state_count);
            match original_to_internal[reset as usize] {
                u32::MAX => {
                    original_to_internal[reset as usize] = start_tsid;
                    internal_to_originals[start_tsid as usize].push(reset);
                }
                existing if existing == start_tsid => {}
                existing => panic!(
                    "{name}: reset state {reset} is already assigned to local TSID {existing}, cannot also assign start TSID {start_tsid}"
                ),
            }
        }
        for states in &mut internal_to_originals {
            states.sort_unstable();
            states.dedup();
        }
        let representative_original_ids = internal_to_originals
            .iter()
            .map(|states| states.first().copied().unwrap_or(u32::MAX))
            .collect::<Vec<_>>();
        id_map.tokenizer_states = ManyToOneIdMap {
            original_to_internal,
            internal_to_originals,
            representative_original_ids,
        };
        eprintln!(
            "MINBOUND state_rebase name={name} local_states={local_state_count} offset={global_offset} resets={enclosing_reset_states:?} monolithic_states={monolithic_state_count}",
        );
        MappedArtifact::new(dwa, id_map)
    }

    fn union_dwas(dwas: &[DWA], id_map: &InternalIdMap) -> DWA {
        let started = Instant::now();
        let mut nwa = NWA::new(id_map.num_tsids(), id_map.max_internal_token_id());
        let mut starts = Vec::new();
        for dwa in dwas {
            let body = nwa.append_with_body(&dwa.to_nwa());
            starts.extend(body.start_states);
        }
        nwa.set_start_states(starts);
        let raw_states = nwa.num_states();
        let raw_transitions = nwa.num_transitions();
        let dwa = determinize(&nwa).expect("component terminal DWA union determinization");
        eprintln!(
            "MINBOUND component_union inputs={} raw_states={} raw_transitions={} states={} transitions={} ms={:.3}",
            dwas.len(), raw_states, raw_transitions, dwa.num_states(), dwa.num_transitions(),
            started.elapsed().as_secs_f64() * 1000.0,
        );
        dwa
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    struct ResidualKey {
        mono: u32,
        component: u32,
        mono_weight: usize,
        component_weight: usize,
    }

    fn exact_weighted_difference(monolithic: &DWA, components: &DWA) -> DWA {
        let started = Instant::now();
        assert!(monolithic.is_acyclic(), "monolithic terminal DWA must be acyclic");
        assert!(components.is_acyclic(), "component terminal DWA union must be acyclic");
        assert!(
            monolithic.states().iter().all(|s| !s.transitions.contains_key(&DEFAULT_LABEL)),
            "experiment residual builder expects no DEFAULT label in monolithic terminal DWA",
        );
        let mut ops = ScopedWeightOpCache::default();
        let all = Weight::all();
        let empty = Weight::empty();
        let start_key = ResidualKey {
            mono: monolithic.start_state(),
            component: components.start_state(),
            mono_weight: all.ptr_key(),
            component_weight: all.ptr_key(),
        };
        let mut states = vec![DWAState::default()];
        let mut payloads = vec![(
            monolithic.start_state(),
            Some(components.start_state()),
            all.clone(),
            all,
        )];
        let mut ids = FxHashMap::<ResidualKey, u32>::default();
        ids.insert(start_key, 0);
        let mut queue = VecDeque::from([0u32]);
        let mut nonempty_finals = 0usize;
        let mut total_final_outer_ranges = 0usize;
        let mut total_final_token_ranges = 0usize;

        while let Some(out_state) = queue.pop_front() {
            let (mono_state, component_state, mono_prefix, component_prefix) =
                payloads[out_state as usize].clone();
            let mono_row = &monolithic.states()[mono_state as usize];
            if let Some(mono_final) = mono_row.final_weight.as_ref() {
                let mono_accept = ops.intersection(&mono_prefix, mono_final);
                let component_accept = component_state
                    .and_then(|state| components.states()[state as usize].final_weight.as_ref())
                    .map(|final_weight| ops.intersection(&component_prefix, final_weight))
                    .unwrap_or_else(Weight::empty);
                let residual = ops.difference(&mono_accept, &component_accept);
                if !residual.is_empty() {
                    nonempty_finals += 1;
                    total_final_outer_ranges += residual.raw_range_values().count();
                    total_final_token_ranges += residual
                        .raw_range_values()
                        .map(|(_, tokens)| tokens.ranges().count())
                        .sum::<usize>();
                    states[out_state as usize].final_weight = Some(residual);
                }
            }

            for (&label, (mono_target, mono_edge_weight)) in &mono_row.transitions {
                let next_mono_weight = ops.intersection(&mono_prefix, mono_edge_weight);
                if next_mono_weight.is_empty() {
                    continue;
                }
                let (next_component, next_component_weight) = if let Some(component_state) = component_state {
                    let row = &components.states()[component_state as usize];
                    if let Some((target, edge_weight)) = row
                        .transitions
                        .get(&label)
                        .or_else(|| row.transitions.get(&DEFAULT_LABEL))
                    {
                        let support = ops.intersection(&component_prefix, edge_weight);
                        if support.is_empty() {
                            (None, empty.clone())
                        } else {
                            (Some(*target), support)
                        }
                    } else {
                        (None, empty.clone())
                    }
                } else {
                    (None, empty.clone())
                };
                let key = ResidualKey {
                    mono: *mono_target,
                    component: next_component.unwrap_or(u32::MAX),
                    mono_weight: next_mono_weight.ptr_key(),
                    component_weight: next_component_weight.ptr_key(),
                };
                let target = if let Some(&target) = ids.get(&key) {
                    target
                } else {
                    let target = states.len() as u32;
                    ids.insert(key, target);
                    states.push(DWAState::default());
                    payloads.push((
                        *mono_target,
                        next_component,
                        next_mono_weight,
                        next_component_weight,
                    ));
                    queue.push_back(target);
                    target
                };
                states[out_state as usize]
                    .transitions
                    .insert(label, (target, Weight::all()));
            }
        }
        let raw = DWA::from_parts(states, 0);
        let raw_states = raw.num_states();
        let raw_transitions = raw.num_transitions();
        let minimize_started = Instant::now();
        let minimized = crate::automata::weighted_u32::minimize_acyclic::minimize_acyclic_owned(raw);
        let minimize_ms = minimize_started.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "MINBOUND residual raw_states={} raw_transitions={} raw_nonempty_finals={} final_outer_ranges={} final_token_ranges={} minimized_states={} minimized_transitions={} minimize_ms={minimize_ms:.3} total_ms={:.3}",
            raw_states,
            raw_transitions,
            nonempty_finals,
            total_final_outer_ranges,
            total_final_token_ranges,
            minimized.num_states(),
            minimized.num_transitions(),
            started.elapsed().as_secs_f64() * 1000.0,
        );
        minimized
    }

    fn filter_residual_cross_outer_component(
        residual: &DWA,
        terminal_offsets: &[u32],
    ) -> DWA {
        const NONE: u16 = u16::MAX;
        const CROSS: u16 = u16::MAX - 1;
        let started = Instant::now();
        let mut states = vec![DWAState::default()];
        let mut ids = FxHashMap::<(u32, u16), u32>::default();
        let mut payloads = vec![(residual.start_state(), NONE)];
        ids.insert((residual.start_state(), NONE), 0);
        let mut queue = VecDeque::from([0u32]);
        while let Some(out) = queue.pop_front() {
            let (source, seen) = payloads[out as usize];
            let source_state = &residual.states()[source as usize];
            if seen == CROSS {
                states[out as usize].final_weight = source_state.final_weight.clone();
            }
            for (&label, (target, weight)) in &source_state.transitions {
                assert!(
                    label >= 0 && label != DEFAULT_LABEL,
                    "terminal residual has non-terminal label {label}"
                );
                let component = terminal_offsets
                    .partition_point(|&offset| offset <= label as u32)
                    .saturating_sub(1) as u16;
                let next_seen = match seen {
                    NONE => component,
                    CROSS => CROSS,
                    existing if existing == component => existing,
                    _ => CROSS,
                };
                let key = (*target, next_seen);
                let next = if let Some(&id) = ids.get(&key) {
                    id
                } else {
                    let id = states.len() as u32;
                    ids.insert(key, id);
                    states.push(DWAState::default());
                    payloads.push((*target, next_seen));
                    queue.push_back(id);
                    id
                };
                states[out as usize]
                    .transitions
                    .insert(label, (next, weight.clone()));
            }
        }
        let raw = DWA::from_parts(states, 0);
        let raw_states = raw.num_states();
        let raw_transitions = raw.num_transitions();
        let minimized =
            crate::automata::weighted_u32::minimize_acyclic::minimize_acyclic_owned(raw);
        eprintln!(
            "MINBOUND ideal_filter kind=cross_outer_component raw_states={} raw_transitions={} states={} transitions={} ms={:.3}",
            raw_states,
            raw_transitions,
            minimized.num_states(),
            minimized.num_transitions(),
            started.elapsed().as_secs_f64() * 1000.0,
        );
        minimized
    }

    #[test]
    #[ignore]
    fn debug_minimal_terminal_boundary_subtraction_selected10() {
        use std::fs;
        use std::sync::Arc;
        use std::time::Instant;




        fn save_terminal_capture(path: &str, artifact: &MappedArtifact<DWA>, names: &[String]) {
            use std::io::Write;
            fn write_u32(out: &mut Vec<u8>, value: u32) { out.extend_from_slice(&value.to_le_bytes()); }
            fn write_u64(out: &mut Vec<u8>, value: u64) { out.extend_from_slice(&value.to_le_bytes()); }
            fn write_vec(out: &mut Vec<u8>, values: &[u32]) {
                write_u32(out, values.len() as u32);
                for &value in values { write_u32(out, value); }
            }
            fn write_vec_vec(out: &mut Vec<u8>, values: &[Vec<u32>]) {
                write_u32(out, values.len() as u32);
                for value in values { write_vec(out, value); }
            }
            fn write_map(out: &mut Vec<u8>, map: &ManyToOneIdMap) {
                write_vec(out, &map.original_to_internal);
                write_vec_vec(out, &map.internal_to_originals);
                write_vec(out, &map.representative_original_ids);
            }
            let encoded = bincode::serialize(artifact.artifact()).unwrap();
            let mut out = Vec::new();
            out.extend_from_slice(b"GLRMTD1\0");
            write_u32(&mut out, names.len() as u32);
            for name in names {
                write_u32(&mut out, name.len() as u32);
                out.extend_from_slice(name.as_bytes());
            }
            write_map(&mut out, &artifact.id_map().tokenizer_states);
            write_map(&mut out, &artifact.id_map().vocab_tokens);
            write_u64(&mut out, encoded.len() as u64);
            out.extend_from_slice(&encoded);
            let mut file = fs::File::create(path).unwrap();
            file.write_all(&out).unwrap();
            eprintln!("MINBOUND saved_capture path={path} states={} transitions={} bytes={}", artifact.artifact().num_states(), artifact.artifact().num_transitions(), out.len());
        }

        fn load_terminal_capture(path: &str) -> (MappedArtifact<DWA>, Vec<String>) {
            let bytes = fs::read(path).expect("read terminal-DWA capture");
            let mut offset = 0usize;
            assert_eq!(&bytes[..8], b"GLRMTD1\0");
            offset += 8;
            fn read_u32(bytes: &[u8], offset: &mut usize) -> u32 {
                let end = *offset + 4;
                let value = u32::from_le_bytes(bytes[*offset..end].try_into().unwrap());
                *offset = end;
                value
            }
            fn read_u64(bytes: &[u8], offset: &mut usize) -> u64 {
                let end = *offset + 8;
                let value = u64::from_le_bytes(bytes[*offset..end].try_into().unwrap());
                *offset = end;
                value
            }
            fn read_vec(bytes: &[u8], offset: &mut usize) -> Vec<u32> {
                let len = read_u32(bytes, offset) as usize;
                (0..len).map(|_| read_u32(bytes, offset)).collect()
            }
            fn read_vec_vec(bytes: &[u8], offset: &mut usize) -> Vec<Vec<u32>> {
                let len = read_u32(bytes, offset) as usize;
                (0..len).map(|_| read_vec(bytes, offset)).collect()
            }
            fn read_map(bytes: &[u8], offset: &mut usize) -> ManyToOneIdMap {
                ManyToOneIdMap {
                    original_to_internal: read_vec(bytes, offset),
                    internal_to_originals: read_vec_vec(bytes, offset),
                    representative_original_ids: read_vec(bytes, offset),
                }
            }
            let name_count = read_u32(&bytes, &mut offset) as usize;
            let mut names = Vec::with_capacity(name_count);
            for _ in 0..name_count {
                let len = read_u32(&bytes, &mut offset) as usize;
                let end = offset + len;
                names.push(String::from_utf8(bytes[offset..end].to_vec()).unwrap());
                offset = end;
            }
            let tokenizer_states = read_map(&bytes, &mut offset);
            let vocab_tokens = read_map(&bytes, &mut offset);
            let dwa_len = read_u64(&bytes, &mut offset) as usize;
            let end = offset + dwa_len;
            let dwa: DWA = bincode::deserialize(&bytes[offset..end]).expect("deserialize terminal DWA");
            offset = end;
            assert_eq!(offset, bytes.len());
            let id_map = InternalIdMap {
                tokenizer_states,
                vocab_tokens,
                deferred_vocab_singleton_original_ids: None,
            };
            (MappedArtifact::new(dwa, id_map), names)
        }

        fn analyzed(constraint: &Constraint) -> AnalyzedGrammar {
            let augmented_start = constraint
                .table
                .rules
                .first()
                .expect("constraint table has augmented start")
                .lhs;
            AnalyzedGrammar::from_composed_rules(
                constraint.table.rules.clone(),
                constraint.table.num_terminals,
                constraint.terminal_display_names.clone(),
                constraint.table.nonterminal_display_names.clone(),
                augmented_start,
            )
        }


        fn build_terminal_dwa(
            name: &str,
            constraint: &Constraint,
            vocab: &Vocab,
        ) -> MappedArtifact<DWA> {
            build_terminal_dwa_parts(
                name,
                &constraint.tokenizer,
                &constraint.table,
                &constraint.terminal_display_names,
                constraint.ignore_terminal,
                vocab,
            )
        }


        fn exact_terminal_language_classes(tokenizer: &Tokenizer) -> Vec<u32> {
            let mut representative_by_expr = FxHashMap::<crate::automata::regex::Expr, u32>::default();
            let mut classes = Vec::with_capacity(tokenizer.num_terminals() as usize);
            for terminal in 0..tokenizer.num_terminals() {
                let representative = tokenizer
                    .terminal_expr(terminal)
                    .map(|expr| {
                        *representative_by_expr.entry(expr.clone()).or_insert(terminal)
                    })
                    .unwrap_or(terminal);
                classes.push(representative);
            }
            classes
        }

        fn quotient_terminal_language_labels(name: &str, dwa: &DWA, classes: &[u32]) -> DWA {
            let started = Instant::now();
            let mut nwa = dwa.to_nwa();
            for state in nwa.states_mut() {
                let old = std::mem::take(&mut state.transitions);
                for (label, targets) in old {
                    let mapped = if label == DEFAULT_LABEL {
                        DEFAULT_LABEL
                    } else {
                        assert!(label >= 0, "{name}: unexpected negative terminal label {label}");
                        classes.get(label as usize).copied().unwrap_or(label as u32) as i32
                    };
                    state.transitions.entry(mapped).or_default().extend(targets);
                }
            }
            let determinized = determinize(&nwa).expect("terminal-language quotient determinization");
            let minimized = crate::automata::weighted_u32::minimize_acyclic::minimize_acyclic_owned(determinized);
            eprintln!(
                "MINBOUND language_quotient name={name} input_states={} input_transitions={} states={} transitions={} ms={:.3}",
                dwa.num_states(), dwa.num_transitions(), minimized.num_states(), minimized.num_transitions(),
                started.elapsed().as_secs_f64() * 1000.0,
            );
            minimized
        }


        fn remap_terminal_labels(
            name: &str,
            mut artifact: MappedArtifact<DWA>,
            local_names: &[String],
            monolithic_names: &[String],
            prefix: &str,
        ) -> MappedArtifact<DWA> {
            let mut ids_by_name = BTreeMap::<&str, Vec<u32>>::new();
            for (id, display) in monolithic_names.iter().enumerate() {
                ids_by_name.entry(display.as_str()).or_default().push(id as u32);
            }
            let used = artifact
                .artifact()
                .states()
                .iter()
                .flat_map(|state| state.transitions.keys().copied())
                .filter(|&label| label != DEFAULT_LABEL)
                .collect::<BTreeSet<_>>();
            // Display names are not unique: generated JSON terminals often reuse
            // names such as `__terminal_expr_16` for several distinct IDs. Composition
            // preserves terminal order within each embedded grammar, so map the Nth local
            // occurrence of a display name to the Nth prefixed monolithic occurrence.
            let mut local_occurrence = vec![0usize; local_names.len()];
            let mut seen = BTreeMap::<&str, usize>::new();
            for (local, display) in local_names.iter().enumerate() {
                let slot = seen.entry(display.as_str()).or_default();
                local_occurrence[local] = *slot;
                *slot += 1;
            }
            let mut mapping = BTreeMap::<i32, i32>::new();
            for label in used {
                assert!(label >= 0, "terminal DWA unexpectedly contains negative label {label}");
                let local = label as usize;
                let display = local_names
                    .get(local)
                    .unwrap_or_else(|| panic!("{name}: terminal label {label} outside display names"));
                let expected = format!("{prefix}{display}");
                let candidates = ids_by_name
                    .get(expected.as_str())
                    .unwrap_or_else(|| panic!("{name}: monolithic terminal not found for local {label} {display:?}, expected {expected:?}"));
                let occurrence = local_occurrence[local];
                let mapped = *candidates.get(occurrence).unwrap_or_else(|| {
                    panic!("{name}: local occurrence {occurrence} of {display:?} has no corresponding monolithic ID; candidates={candidates:?}")
                });
                mapping.insert(label, mapped as i32);
            }
            let mut default_rows = 0usize;
            for state in artifact.artifact_mut().states_mut() {
                if state.transitions.contains_key(&DEFAULT_LABEL) {
                    default_rows += 1;
                }
                let old = std::mem::take(&mut state.transitions);
                for (&label, edge) in old.iter() {
                    let mapped = if label == DEFAULT_LABEL {
                        DEFAULT_LABEL
                    } else {
                        *mapping.get(&label).unwrap_or_else(|| panic!("{name}: no terminal mapping for explicit label {label}"))
                    };
                    assert!(
                        state.transitions.insert(mapped, edge.clone()).is_none(),
                        "{name}: terminal label remap merged distinct transitions onto {mapped}",
                    );
                }
            }
            eprintln!("MINBOUND label_map name={name} used={} default_rows={} prefix={prefix:?}", mapping.len(), default_rows);
            artifact
        }








        fn acyclic_path_stats(name: &str, dwa: &DWA) {
            fn dfs(state: u32, dwa: &DWA, memo: &mut [Option<(usize, u128)>]) -> (usize, u128) {
                if let Some(value) = memo[state as usize] { return value; }
                let row = &dwa.states()[state as usize];
                let mut max_len = if row.final_weight.as_ref().is_some_and(|w| !w.is_empty()) { 0 } else { 0 };
                let mut paths = if row.final_weight.as_ref().is_some_and(|w| !w.is_empty()) { 1u128 } else { 0 };
                for (_label, (target, _weight)) in &row.transitions {
                    let (tail_len, tail_paths) = dfs(*target, dwa, memo);
                    if tail_paths != 0 { max_len = max_len.max(1 + tail_len); }
                    paths = paths.saturating_add(tail_paths);
                }
                let value = (max_len, paths);
                memo[state as usize] = Some(value);
                value
            }
            let mut memo = vec![None; dwa.num_states() as usize];
            let (max_len, paths) = dfs(dwa.start_state(), dwa, &mut memo);
            eprintln!("MINBOUND path_stats name={name} max_terminal_length={max_len} accepting_label_paths={paths}");
        }

        fn acyclic_parser_path_stats(name: &str, dwa: &DWA) {
            assert!(dwa.is_acyclic(), "{name}: parser DWA unexpectedly cyclic");
            fn dfs(state: u32, dwa: &DWA, memo: &mut [Option<(usize, u128)>]) -> (usize, u128) {
                if let Some(value) = memo[state as usize] { return value; }
                let row = &dwa.states()[state as usize];
                let mut max_len = 0usize;
                let mut paths = if row.final_weight.as_ref().is_some_and(|w| !w.is_empty()) { 1u128 } else { 0 };
                for (_label, (target, _weight)) in &row.transitions {
                    let (tail_len, tail_paths) = dfs(*target, dwa, memo);
                    if tail_paths != 0 { max_len = max_len.max(1 + tail_len); }
                    paths = paths.saturating_add(tail_paths);
                }
                let value = (max_len, paths);
                memo[state as usize] = Some(value);
                value
            }
            let mut memo = vec![None; dwa.num_states() as usize];
            let (max_len, paths) = dfs(dwa.start_state(), dwa, &mut memo);
            let mut positive = 0usize;
            let mut negative = 0usize;
            let mut defaults = 0usize;
            for state in dwa.states() {
                for &label in state.transitions.keys() {
                    if label == DEFAULT_LABEL { defaults += 1; }
                    else if is_negative_label(label) { negative += 1; }
                    else { positive += 1; }
                }
            }
            let finals = dwa.states().iter().filter(|state| state.final_weight.as_ref().is_some_and(|w| !w.is_empty())).count();
            eprintln!(
                "MINBOUND parser_path_stats name={name} max_stack_effect_labels={max_len} accepting_label_paths={paths} final_states={finals} positive_edges={positive} negative_edges={negative} default_edges={defaults}"
            );
        }

        fn terminal_component_id(display: &str) -> u16 {
            const CHILD_PREFIX: &str = "subgrammar0::subgrammar";
            if let Some(rest) = display.strip_prefix(CHILD_PREFIX) {
                let digits = rest.chars().take_while(|ch| ch.is_ascii_digit()).collect::<String>();
                if !digits.is_empty() && rest[digits.len()..].starts_with("::") {
                    return 2 + digits.parse::<u16>().expect("schema component id");
                }
            }
            if display.starts_with("subgrammar0::") { 1 } else { 0 }
        }

        fn filter_residual_min_terminals(residual: &DWA, minimum: u8) -> DWA {
            let started = Instant::now();
            let mut states = vec![DWAState::default()];
            let mut ids = FxHashMap::<(u32, u8), u32>::default();
            let mut payloads = vec![(residual.start_state(), 0u8)];
            ids.insert((residual.start_state(), 0), 0);
            let mut queue = VecDeque::from([0u32]);
            while let Some(out) = queue.pop_front() {
                let (source, depth) = payloads[out as usize];
                let source_state = &residual.states()[source as usize];
                if depth >= minimum {
                    states[out as usize].final_weight = source_state.final_weight.clone();
                }
                for (&label, (target, weight)) in &source_state.transitions {
                    let next_depth = depth.saturating_add(1).min(minimum);
                    let key = (*target, next_depth);
                    let next = if let Some(&id) = ids.get(&key) {
                        id
                    } else {
                        let id = states.len() as u32;
                        ids.insert(key, id);
                        states.push(DWAState::default());
                        payloads.push((*target, next_depth));
                        queue.push_back(id);
                        id
                    };
                    states[out as usize].transitions.insert(label, (next, weight.clone()));
                }
            }
            let raw = DWA::from_parts(states, 0);
            let raw_states = raw.num_states();
            let raw_transitions = raw.num_transitions();
            let minimized = crate::automata::weighted_u32::minimize_acyclic::minimize_acyclic_owned(raw);
            eprintln!(
                "MINBOUND ideal_filter kind=min_terminals minimum={} raw_states={} raw_transitions={} states={} transitions={} ms={:.3}",
                minimum, raw_states, raw_transitions, minimized.num_states(), minimized.num_transitions(),
                started.elapsed().as_secs_f64() * 1000.0,
            );
            minimized
        }

        fn filter_residual_cross_component(residual: &DWA, terminal_names: &[String]) -> DWA {
            const NONE: u16 = u16::MAX;
            const CROSS: u16 = u16::MAX - 1;
            let started = Instant::now();
            let mut states = vec![DWAState::default()];
            let mut ids = FxHashMap::<(u32, u16), u32>::default();
            let mut payloads = vec![(residual.start_state(), NONE)];
            ids.insert((residual.start_state(), NONE), 0);
            let mut queue = VecDeque::from([0u32]);
            while let Some(out) = queue.pop_front() {
                let (source, seen) = payloads[out as usize];
                let source_state = &residual.states()[source as usize];
                if seen == CROSS {
                    states[out as usize].final_weight = source_state.final_weight.clone();
                }
                for (&label, (target, weight)) in &source_state.transitions {
                    assert!(label >= 0 && label != DEFAULT_LABEL, "terminal residual has non-terminal label {label}");
                    let component = terminal_component_id(
                        terminal_names.get(label as usize).expect("terminal name for residual label"),
                    );
                    let next_seen = match seen {
                        NONE => component,
                        CROSS => CROSS,
                        existing if existing == component => existing,
                        _ => CROSS,
                    };
                    let key = (*target, next_seen);
                    let next = if let Some(&id) = ids.get(&key) {
                        id
                    } else {
                        let id = states.len() as u32;
                        ids.insert(key, id);
                        states.push(DWAState::default());
                        payloads.push((*target, next_seen));
                        queue.push_back(id);
                        id
                    };
                    states[out as usize].transitions.insert(label, (next, weight.clone()));
                }
            }
            let raw = DWA::from_parts(states, 0);
            let raw_states = raw.num_states();
            let raw_transitions = raw.num_transitions();
            let minimized = crate::automata::weighted_u32::minimize_acyclic::minimize_acyclic_owned(raw);
            eprintln!(
                "MINBOUND ideal_filter kind=cross_component raw_states={} raw_transitions={} states={} transitions={} ms={:.3}",
                raw_states, raw_transitions, minimized.num_states(), minimized.num_transitions(),
                started.elapsed().as_secs_f64() * 1000.0,
            );
            minimized
        }


        fn residual_diagnostics(
            name: &str,
            residual: &DWA,
            id_map: &InternalIdMap,
            terminal_names: &[String],
        ) {
            fn support_counts(weight: &Weight, id_map: &InternalIdMap) -> (usize, usize, usize, usize) {
                if weight.is_empty() {
                    return (0, 0, 0, 0);
                }
                let mut tsids = BTreeSet::<u32>::new();
                let mut raw_states = BTreeSet::<u32>::new();
                let mut internal_tokens = BTreeSet::<u32>::new();
                let mut original_tokens = BTreeSet::<u32>::new();
                for (tsid_range, tokens) in weight.raw_range_values() {
                    for tsid in tsid_range {
                        tsids.insert(tsid);
                        if let Some(states) = id_map.tokenizer_states.internal_to_originals.get(tsid as usize) {
                            raw_states.extend(states.iter().copied());
                        }
                    }
                    for token_range in tokens.ranges() {
                        for token in token_range {
                            internal_tokens.insert(token);
                            if let Some(originals) = id_map.vocab_tokens.internal_to_originals.get(token as usize) {
                                original_tokens.extend(originals.iter().copied());
                            }
                        }
                    }
                }
                (tsids.len(), raw_states.len(), internal_tokens.len(), original_tokens.len())
            }

            let total_weight = Weight::union_all(
                residual.states().iter().filter_map(|state| state.final_weight.as_ref()),
            );
            let total = support_counts(&total_weight, id_map);
            let start = &residual.states()[residual.start_state() as usize];
            let mut one_step = Vec::<(i32, Weight)>::new();
            for (&label, (target, edge_weight)) in &start.transitions {
                if label == DEFAULT_LABEL {
                    continue;
                }
                if let Some(final_weight) = residual.states()[*target as usize].final_weight.as_ref() {
                    let support = edge_weight.intersection(final_weight);
                    if !support.is_empty() {
                        one_step.push((label, support));
                    }
                }
            }
            let one_weight = Weight::union_all(one_step.iter().map(|(_, weight)| weight));
            let one = support_counts(&one_weight, id_map);
            eprintln!(
                "MINBOUND support name={name} total_tsids={} total_raw_states={} total_internal_tokens={} total_original_tokens={} one_terminal_labels={} one_terminal_tsids={} one_terminal_raw_states={} one_terminal_internal_tokens={} one_terminal_original_tokens={}",
                total.0, total.1, total.2, total.3,
                one_step.len(), one.0, one.1, one.2, one.3,
            );
            let mut one_details = one_step
                .iter()
                .map(|(label, weight)| {
                    let counts = support_counts(weight, id_map);
                    let display = terminal_names
                        .get(*label as usize)
                        .map(String::as_str)
                        .unwrap_or("<unknown>");
                    (*label, display.to_string(), counts)
                })
                .collect::<Vec<_>>();
            one_details.sort_by_key(|(_, _, counts)| std::cmp::Reverse(counts.3));
            for (label, display, counts) in one_details.into_iter().take(30) {
                eprintln!(
                    "MINBOUND one_terminal name={name} terminal={} display={:?} tsids={} raw_states={} internal_tokens={} original_tokens={}",
                    label, display, counts.0, counts.1, counts.2, counts.3,
                );
            }
        }

        fn parser_from_residual(
            name: &str,
            residual: &DWA,
            full: &Constraint,
            common_id_map: &InternalIdMap,
            vocab: &Vocab,
        ) {
            let mut selected = vec![false; full.table.num_terminals as usize];
            for state in residual.states() {
                for &label in state.transitions.keys() {
                    if label >= 0 && (label as usize) < selected.len() {
                        selected[label as usize] = true;
                    }
                }
            }
            let active = selected.iter().filter(|&&value| value).count();
            let grammar = analyzed(full);
            let template_started = Instant::now();
            let (templates, _, _) = build_composition_templates(&full.table, &grammar, &selected);
            let template_ms = template_started.elapsed().as_secs_f64() * 1000.0;
            let parser_started = Instant::now();
            let parser = build_parser_dwa_from_terminal_dwa_with_precomputed_templates(
                &full.table,
                &grammar,
                &TerminalAutomaton::Dwa(residual.clone()),
                &templates,
                vocab,
                common_id_map,
                false,
            );
            let parser_ms = parser_started.elapsed().as_secs_f64() * 1000.0;
            eprintln!(
                "MINBOUND parser name={name} active_terminals={} residual_states={} residual_transitions={} template_ms={template_ms:.3} parser_ms={parser_ms:.3} parser_states={} parser_transitions={}",
                active,
                residual.num_states(),
                residual.num_transitions(),
                parser.num_states(),
                parser.num_transitions(),
            );
        }

        /// Keep exactly those weighted terminal paths whose composed LR stack-effect
        /// relation is nonempty.  This is stronger than the pair/trigram CFG filters:
        /// it asks the same parser-template semantics used to build the boundary parser.
        ///
        /// The residual is acyclic, so compute parser-domain lanes backwards.  A lane
        /// `(terminal_state, root)` denotes suffixes from `terminal_state` whose exact
        /// parser-stack preimage is `root`.  Reifying those lanes as an NWA preserves
        /// path distinctions that would be lost by merely pruning individual edges.
        fn parser_domain_tighten_residual(
            name: &str,
            residual: &DWA,
            table: &crate::compiler::glr::table::GLRTable,
            templates: &Templates,
        ) -> DWA {
            assert!(residual.is_acyclic(), "{name}: grammar tightening expects acyclic residual");
            let started = Instant::now();
            let n = residual.num_states() as usize;

            let mut indegree = vec![0usize; n];
            for state in residual.states() {
                for &(target, _) in state.transitions.values() {
                    indegree[target as usize] += 1;
                }
            }
            let mut queue = VecDeque::new();
            for (state, &degree) in indegree.iter().enumerate() {
                if degree == 0 {
                    queue.push_back(state as u32);
                }
            }
            let mut topo = Vec::with_capacity(n);
            while let Some(source) = queue.pop_front() {
                topo.push(source);
                for &(target, _) in residual.states()[source as usize].transitions.values() {
                    indegree[target as usize] -= 1;
                    if indegree[target as usize] == 0 {
                        queue.push_back(target);
                    }
                }
            }
            assert_eq!(topo.len(), n, "{name}: residual unexpectedly cyclic");

            let mut seen_terminal = vec![false; table.num_terminals as usize];
            let mut non_skip_terminal = vec![false; table.num_terminals as usize];
            let mut skip_states_by_terminal = vec![Vec::<u32>::new(); table.num_terminals as usize];
            for (source, row) in table.action.iter().enumerate() {
                for (terminal, action) in row {
                    let Some(seen) = seen_terminal.get_mut(terminal as usize) else { continue };
                    *seen = true;
                    match action {
                        Action::Skip => skip_states_by_terminal[terminal as usize].push(source as u32),
                        _ => non_skip_terminal[terminal as usize] = true,
                    }
                }
            }
            let pure_skip = (0..table.num_terminals as usize)
                .map(|terminal| seen_terminal[terminal] && !non_skip_terminal[terminal])
                .collect::<Vec<_>>();

            #[derive(Clone)]
            struct Edge {
                source_root: u32,
                label: u32,
                target_state: u32,
                target_root: u32,
                support: Weight,
            }

            let mut arena = SharedBooleanParserDomains::new();
            let mut domains = vec![BTreeMap::<u32, Weight>::new(); n];
            let mut edges_by_state = vec![Vec::<Edge>::new(); n];
            let mut bundles = FxHashMap::<u32, Arc<NWA>>::default();
            let mut preimage_cache = FxHashMap::<(u32, u32), u32>::default();
            let mut weight_ops = ScopedWeightOpCache::default();
            let mut preimage_calls = 0usize;

            for &source in topo.iter().rev() {
                let mut lanes = BTreeMap::<u32, Weight>::new();
                if let Some(final_weight) = residual.states()[source as usize].final_weight.as_ref() {
                    lanes.insert(SharedBooleanParserDomains::UNIVERSAL, final_weight.clone());
                }
                for (&label, (target, edge_weight)) in &residual.states()[source as usize].transitions {
                    assert!(label >= 0 && label != DEFAULT_LABEL, "{name}: non-terminal residual label {label}");
                    let terminal = label as u32;
                    for (&target_root, target_support) in &domains[*target as usize] {
                        let support = weight_ops.intersection(edge_weight, target_support);
                        if support.is_empty() {
                            continue;
                        }
                        let source_root = if let Some(&root) = preimage_cache.get(&(terminal, target_root)) {
                            root
                        } else {
                            let root = if pure_skip.get(terminal as usize).copied().unwrap_or(false) {
                                arena.preimage_identity_skip(
                                    target_root,
                                    &skip_states_by_terminal[terminal as usize],
                                )
                            } else {
                                let bundle = if let Some(bundle) = bundles.get(&terminal) {
                                    Arc::clone(bundle)
                                } else {
                                    let bundle = build_boolean_terminal_bundle_nwa(templates, &[terminal])
                                        .unwrap_or_else(|| NWA::new(0, 0));
                                    let bundle = Arc::new(bundle);
                                    bundles.insert(terminal, Arc::clone(&bundle));
                                    bundle
                                };
                                arena.preimage_bundle(&bundle, target_root).unwrap_or(SharedBooleanParserDomains::EMPTY)
                            };
                            preimage_calls += 1;
                            preimage_cache.insert((terminal, target_root), root);
                            root
                        };
                        if source_root == SharedBooleanParserDomains::EMPTY {
                            continue;
                        }
                        if let Some(existing) = lanes.get_mut(&source_root) {
                            *existing = weight_ops.union(existing, &support);
                        } else {
                            lanes.insert(source_root, support.clone());
                        }
                        edges_by_state[source as usize].push(Edge {
                            source_root,
                            label: terminal,
                            target_state: *target,
                            target_root,
                            support,
                        });
                    }
                }
                domains[source as usize] = lanes;
            }

            let mut lane_id = FxHashMap::<(u32, u32), u32>::default();
            let mut nwa = NWA::new(0, 0);
            for (state, lanes) in domains.iter().enumerate() {
                for &root in lanes.keys() {
                    let id = nwa.add_state();
                    lane_id.insert((state as u32, root), id);
                }
            }
            let starts = domains[residual.start_state() as usize]
                .keys()
                .map(|&root| lane_id[&(residual.start_state(), root)])
                .collect::<Vec<_>>();
            nwa.set_start_states(starts);
            for (state, lanes) in domains.iter().enumerate() {
                if let Some(final_weight) = residual.states()[state].final_weight.as_ref() {
                    if lanes.contains_key(&SharedBooleanParserDomains::UNIVERSAL) {
                        let id = lane_id[&(state as u32, SharedBooleanParserDomains::UNIVERSAL)];
                        nwa.set_final_weight(id, final_weight.clone());
                    }
                }
                for edge in &edges_by_state[state] {
                    let from = lane_id[&(state as u32, edge.source_root)];
                    let to = lane_id[&(edge.target_state, edge.target_root)];
                    nwa.add_transition(from, edge.label as i32, to, edge.support.clone());
                }
            }
            let lane_states = nwa.num_states();
            let lane_transitions = nwa.num_transitions();
            let deterministic = determinize(&nwa).expect("parser-domain-tight residual determinization");
            let premin_states = deterministic.num_states();
            let premin_transitions = deterministic.num_transitions();
            let minimized = crate::automata::weighted_u32::minimize_acyclic::minimize_acyclic_owned(deterministic);
            eprintln!(
                "MINBOUND parser_domain_tight name={name} input_states={} input_transitions={} lane_states={} lane_transitions={} start_lanes={} parser_roots={} preimage_calls={} premin_states={} premin_transitions={} states={} transitions={} ms={:.3}",
                residual.num_states(),
                residual.num_transitions(),
                lane_states,
                lane_transitions,
                domains[residual.start_state() as usize].len(),
                arena.node_count(),
                preimage_calls,
                premin_states,
                premin_transitions,
                minimized.num_states(),
                minimized.num_transitions(),
                started.elapsed().as_secs_f64() * 1000.0,
            );
            minimized
        }


        if std::env::var_os("GLRMASK_MINBOUND_LIVE_OUTER").is_some() {
            let root = std::env::var("GLRMASK_MINBOUND_DIR").expect("GLRMASK_MINBOUND_DIR");
            let vocab_path = std::env::var("GLRMASK_MINBOUND_VOCAB").expect("GLRMASK_MINBOUND_VOCAB");
            let mut vocab = load_vocab(&vocab_path);
            if let Ok(filter_path) = std::env::var("GLRMASK_MINBOUND_VOCAB_FILTER") {
                let allowed = fs::read_to_string(&filter_path)
                    .expect("read minimum-boundary vocab filter")
                    .split_whitespace()
                    .map(|value| value.parse::<u32>().expect("parse filtered token id"))
                    .collect::<BTreeSet<_>>();
                vocab = Vocab::new(
                    vocab
                        .entries_map()
                        .iter()
                        .filter(|(token, _)| allowed.contains(token))
                        .map(|(&token, bytes)| (token, bytes.clone()))
                        .collect(),
                );
                eprintln!("MINBOUND filtered_vocab tokens={}", vocab.entries_map().len());
            }
            let load = |name: &str| -> Constraint {
                Constraint::load(&fs::read(std::path::Path::new(&root).join(name)).unwrap()).unwrap()
            };
            let mut core = load("core.bin");
            let dispatch_name = std::env::var("GLRMASK_MINBOUND_DISPATCH")
                .unwrap_or_else(|_| "dispatch-literal.bin".to_string());
            let mut dispatch = load(&dispatch_name);
            // Measurement-oracle fix (worktree-only): loaded artifacts keep
            // terminal expressions only in the deferred blob, but the terminal-DWA
            // rebuilds below (B and A_i) classify through tokenizer-held
            // expressions. Restore them exactly as production composition does.
            for component in [&mut core, &mut dispatch] {
                if component.tokenizer.terminal_exprs().is_none() {
                    let exprs = component.retained_terminal_exprs().map(|exprs| exprs.to_vec());
                    if let Some(exprs) = exprs {
                        Arc::make_mut(&mut component.tokenizer).restore_terminal_exprs(Some(exprs))
                            .expect("restore component terminal exprs for minbound oracle");
                    }
                }
            }
            eprintln!(
                "MINBOUND outer_expr_restore core_exprs={} dispatch_exprs={}",
                core.tokenizer.terminal_exprs().is_some(),
                dispatch.tokenizer.terminal_exprs().is_some(),
            );
            // Measurement-oracle fix (worktree-only): large loaded artifacts
            // keep only the augmented-start rule inline in table.rules with the
            // rest in a deferred blob. The oracle's follow/disallowed analysis
            // reads table.rules directly, so restore the retained rules first
            // (no-op when the table already carries complete rules).
            for (label, component) in [("core", &mut core), ("dispatch", &mut dispatch)] {
                let inline_rules = component.table.rules.len();
                let retained = component
                    .retained_table_rules()
                    .expect("decode retained table rules for minbound oracle")
                    .len();
                eprintln!(
                    "MINBOUND outer_rules {label} inline_rules={inline_rules} retained_rules={retained} num_rules={}",
                    component.table.num_rules,
                );
                if inline_rules != retained {
                    let rules = component
                        .retained_table_rules()
                        .expect("decode retained table rules for minbound oracle")
                        .to_vec();
                    component.table.rules = rules;
                }
            }
            // Same deferred-metadata class: the template-reuse studies below
            // read per-terminal parser characterizations/templates, which load
            // keeps in the deferred composition-metadata blob. Materialize them
            // exactly as production composition does.
            for component in [&mut core, &mut dispatch] {
                component
                    .materialize_composition_metadata_for_compilation()
                    .expect("materialize composition metadata for minbound oracle");
            }
            let placeholder = terminal(&core, "PROGRAMMATIC_TOOL_SUFFIX");
            let child = CompiledSubgrammarInput {
                placeholder_terminal: placeholder,
                additional_placeholder_terminals: &[],
constraint: &dispatch,
            };
            let children = [child];
            let global_ignores = component_ignores_are_globally_erasable(&core, &children);
            let table_inputs = [SubgrammarTableInput {
                placeholder_terminal: placeholder,
                additional_placeholder_terminals: &[],
table: &dispatch.table,
                ignore_terminal: (!global_ignores).then_some(dispatch.ignore_terminal).flatten(),
                start_nullable: dispatch.table.embedded_start_nullable(),
            }];
            let outer_table_started = Instant::now();
            let mut composed_table = compose_subgrammar_tables(
                &core.table,
                (!global_ignores).then_some(core.ignore_terminal).flatten(),
                &table_inputs,
            )
            .expect("compose outer table for minimum-boundary oracle");
            eliminate_composed_runtime_controls(&mut composed_table)
                .expect("eliminate outer controls for minimum-boundary oracle");
            let outer_table_ms = outer_table_started.elapsed().as_secs_f64() * 1000.0;
            let terminal_names = merged_terminal_display_names(&core, &children);
            let tokenizer_inputs = [
                (core.tokenizer.as_ref(), composed_table.terminal_offsets[0]),
                (dispatch.tokenizer.as_ref(), composed_table.terminal_offsets[1]),
            ];
            let outer_tokenizer_started = Instant::now();
            let (mut merged_tokenizer, tokenizer_offsets) =
                Tokenizer::disjoint_union_with_terminal_offsets(&tokenizer_inputs);
            if merged_tokenizer.terminal_exprs().is_none() {
                if let Some(exprs) = merged_retained_terminal_exprs(
                    &[&core, &dispatch],
                    &composed_table.terminal_offsets,
                    composed_table.table.num_terminals,
                ) {
                    merged_tokenizer
                        .restore_terminal_exprs(Some(exprs))
                        .expect("restore merged terminal exprs for minbound oracle");
                }
            }
            let outer_tokenizer_ms = outer_tokenizer_started.elapsed().as_secs_f64() * 1000.0;
            eprintln!(
                "MINBOUND outer_expr_restore merged_exprs={}",
                merged_tokenizer.terminal_exprs().is_some(),
            );
            let merged_ignores = merged_ignore_terminals(
                &core,
                &children,
                &composed_table.terminal_offsets,
                global_ignores,
            );
            eprintln!(
                "MINBOUND outer_setup global_ignores={} lr_states={} terminals={} tokenizer_states={} offsets={:?} terminal_offsets={:?}",
                global_ignores,
                composed_table.table.num_states,
                composed_table.table.num_terminals,
                merged_tokenizer.num_states(),
                tokenizer_offsets,
                composed_table.terminal_offsets,
            );
            eprintln!(
                "MINBOUND OUTER_BASE_TIMES table_ms={outer_table_ms:.3} tokenizer_ms={outer_tokenizer_ms:.3}"
            );

            let monolithic = build_terminal_dwa_parts(
                "outer_B_composed",
                &merged_tokenizer,
                &composed_table.table,
                &terminal_names,
                merged_ignores.canonical,
                &vocab,
            );
            // Worktree-only oracle diagnostics: path shape of B and each A_i.
            acyclic_path_stats("outer_B_composed", monolithic.artifact());
            // A and B must use the same visible-terminal semantics. When the
            // component ignores differ, the composed lexer does not erase them;
            // therefore the standalone component terminal DWAs used in A must
            // also expose IGNORE as an ordinary terminal for this oracle.
            let core_dwa = offset_terminal_labels(
                "outer_A_core",
                build_terminal_dwa_parts(
                    "outer_A_core",
                    &core.tokenizer,
                    &core.table,
                    &core.terminal_display_names,
                    global_ignores.then_some(core.ignore_terminal).flatten(),
                    &vocab,
                ),
                composed_table.terminal_offsets[0],
            );
            let dispatch_dwa = offset_terminal_labels(
                "outer_A_dispatch",
                build_terminal_dwa_parts(
                    "outer_A_dispatch",
                    &dispatch.tokenizer,
                    &dispatch.table,
                    &dispatch.terminal_display_names,
                    global_ignores.then_some(dispatch.ignore_terminal).flatten(),
                    &vocab,
                ),
                composed_table.terminal_offsets[1],
            );
            acyclic_path_stats("outer_A_core_pre_rebase", core_dwa.artifact());
            acyclic_path_stats("outer_A_dispatch_pre_rebase", dispatch_dwa.artifact());
            let state_count = merged_tokenizer.num_states() as usize;
            let core_dwa = rebase_tokenizer_state_universe(
                "outer_A_core",
                core_dwa,
                tokenizer_offsets[0],
                &[0],
                state_count,
            );
            let dispatch_dwa = rebase_tokenizer_state_universe(
                "outer_A_dispatch",
                dispatch_dwa,
                tokenizer_offsets[1],
                &[0],
                state_count,
            );
            let reconcile_started = Instant::now();
            let reconciled = MappedArtifact::reconcile_vec(vec![monolithic, core_dwa, dispatch_dwa]);
            let (all, common_id_map) = reconciled.into_parts();
            eprintln!(
                "MINBOUND outer_reconcile tsids={} tokens={} ms={:.3}",
                common_id_map.num_tsids(),
                common_id_map.num_internal_tokens(),
                reconcile_started.elapsed().as_secs_f64() * 1000.0,
            );
            let a = union_dwas(&all[1..], &common_id_map);
            let concrete_residual = exact_weighted_difference(&all[0], &a);
            acyclic_path_stats("outer_exact_concrete_B_minus_A", &concrete_residual);
            residual_diagnostics(
                "outer_exact_concrete_B_minus_A",
                &concrete_residual,
                &common_id_map,
                &terminal_names,
            );
            let concrete_depth2 = filter_residual_min_terminals(&concrete_residual, 2);
            let concrete_crossed = filter_residual_cross_outer_component(
                &concrete_residual,
                &composed_table.terminal_offsets,
            );
            let concrete_non_cross =
                exact_weighted_difference(&concrete_residual, &concrete_crossed);
            let concrete_non_cross_nonempty = concrete_non_cross
                .states()
                .iter()
                .filter(|state| {
                    state
                        .final_weight
                        .as_ref()
                        .is_some_and(|weight| !weight.is_empty())
                })
                .count();
            eprintln!(
                "MINBOUND OUTER_CONCRETE_DECOMP depth_ge_2_states={} depth_ge_2_transitions={} crossed_states={} crossed_transitions={} non_cross_states={} non_cross_transitions={} non_cross_nonempty_finals={}",
                concrete_depth2.num_states(),
                concrete_depth2.num_transitions(),
                concrete_crossed.num_states(),
                concrete_crossed.num_transitions(),
                concrete_non_cross.num_states(),
                concrete_non_cross.num_transitions(),
                concrete_non_cross_nonempty,
            );
            eprintln!(
                "MINBOUND OUTER_CONCRETE_RESULT states={} transitions={} labels={}",
                concrete_residual.num_states(),
                concrete_residual.num_transitions(),
                concrete_residual.states().iter().flat_map(|state| state.transitions.keys().copied()).filter(|&label| label >= 0).collect::<BTreeSet<_>>().len(),
            );

            let concrete_grammar = AnalyzedGrammar::from_composed_rules(
                composed_table.table.rules.clone(),
                composed_table.table.num_terminals,
                terminal_names.clone(),
                composed_table.table.nonterminal_display_names.clone(),
                composed_table
                    .table
                    .rules
                    .first()
                    .expect("outer composed table has augmented start")
                    .lhs,
            );

            // Exact bounded grammar-factor oracle.  Unlike the trigram proxy
            // below, this recognizes arbitrary-length terminal factors lazily
            // and is explored only along the finite concrete B-A residual.
            // Parser controls and scoped/global skip terminals are invisible
            // grammar symbols for this necessary-condition filter.
            let mut factor_zero_width = composed_table.table.control_terminals.clone();
            factor_zero_width.extend(composed_table.table.skip_terminals.iter().copied());
            factor_zero_width.extend(merged_ignores.scoped.iter().map(|terminal| terminal as u32));
            let concrete_factor_tight = mb_filter_exact_factor_lazy(
                &concrete_residual,
                &concrete_grammar,
                &factor_zero_width,
            );
            acyclic_path_stats(
                "outer_exact_concrete_B_minus_A_lazy_grammar_factor",
                &concrete_factor_tight,
            );
            residual_diagnostics(
                "outer_exact_concrete_B_minus_A_lazy_grammar_factor",
                &concrete_factor_tight,
                &common_id_map,
                &terminal_names,
            );
            save_terminal_capture(
                std::path::Path::new(&root)
                    .join("tdwa_outer_exact_concrete_B_minus_A_lazy_grammar_factor.cap")
                    .to_str()
                    .unwrap(),
                &MappedArtifact::new(concrete_factor_tight.clone(), common_id_map.clone()),
                &terminal_names,
            );

            let concrete_weight = Weight::union_all(
                concrete_residual
                    .states()
                    .iter()
                    .filter_map(|state| state.final_weight.as_ref()),
            );
            let mut concrete_original_tokens = BTreeSet::<u32>::new();
            for (_, internal_tokens) in concrete_weight.raw_range_values() {
                for range in internal_tokens.ranges() {
                    for internal_token in range {
                        if let Some(originals) = common_id_map
                            .vocab_tokens
                            .internal_to_originals
                            .get(internal_token as usize)
                        {
                            concrete_original_tokens.extend(originals.iter().copied());
                        }
                    }
                }
            }
            eprintln!("MINBOUND OUTER_CONCRETE_TOKENS {:?}", concrete_original_tokens);
            let concrete_capture = MappedArtifact::new(concrete_residual.clone(), common_id_map.clone());
            save_terminal_capture(
                std::path::Path::new(&root)
                    .join("tdwa_outer_exact_concrete_B_minus_A.cap")
                    .to_str()
                    .unwrap(),
                &concrete_capture,
                &terminal_names,
            );

            // Exact grammar/parser-domain tightening on the concrete terminal labels.
            // Unlike the trigram diagnostic below, this uses the same LR stack-effect
            // templates as boundary parser construction and preserves path distinctions.
            let mut concrete_selected = vec![false; composed_table.table.num_terminals as usize];
            for state in concrete_residual.states() {
                for &label in state.transitions.keys() {
                    if label >= 0 && (label as usize) < concrete_selected.len() {
                        concrete_selected[label as usize] = true;
                    }
                }
            }
            let concrete_template_started = Instant::now();
            let (concrete_templates, _, _) = build_composition_templates(
                &composed_table.table,
                &concrete_grammar,
                &concrete_selected,
            );
            let concrete_template_ms = concrete_template_started.elapsed().as_secs_f64() * 1000.0;
            let concrete_grammar_tight = parser_domain_tighten_residual(
                "outer_exact_concrete_B_minus_A",
                &concrete_residual,
                &composed_table.table,
                &concrete_templates,
            );
            acyclic_path_stats(
                "outer_exact_concrete_B_minus_A_parser_domain_tight",
                &concrete_grammar_tight,
            );
            residual_diagnostics(
                "outer_exact_concrete_B_minus_A_parser_domain_tight",
                &concrete_grammar_tight,
                &common_id_map,
                &terminal_names,
            );

            // The grammar-factor and parser-stack-domain filters are distinct
            // necessary conditions.  Measure their exact weighted overlap
            // rather than assuming either subsumes the other.
            let factor_minus_parser =
                exact_weighted_difference(&concrete_factor_tight, &concrete_grammar_tight);
            let parser_minus_factor =
                exact_weighted_difference(&concrete_grammar_tight, &concrete_factor_tight);
            let concrete_ideal =
                exact_weighted_difference(&concrete_factor_tight, &factor_minus_parser);
            let factor_general_minimized = minimize_owned(concrete_factor_tight.clone());
            let concrete_ideal_minimized = minimize_owned(concrete_ideal.clone());
            let factor_accepted_tokens =
                accepted_original_tokens(&factor_general_minimized, &common_id_map);
            let ideal_accepted_tokens =
                accepted_original_tokens(&concrete_ideal_minimized, &common_id_map);
            let nonempty_finals = |dwa: &DWA| {
                dwa.states()
                    .iter()
                    .filter(|state| {
                        state
                            .final_weight
                            .as_ref()
                            .is_some_and(|weight| !weight.is_empty())
                    })
                    .count()
            };
            eprintln!(
                "MINBOUND OUTER_CONCRETE_FILTER_RELATION factor_minus_parser_states={} factor_minus_parser_transitions={} factor_minus_parser_finals={} parser_minus_factor_states={} parser_minus_factor_transitions={} parser_minus_factor_finals={} intersection_states={} intersection_transitions={} factor_general_states={} factor_general_transitions={} factor_accepted_tokens={} ideal_general_states={} ideal_general_transitions={} ideal_accepted_tokens={}",
                factor_minus_parser.num_states(),
                factor_minus_parser.num_transitions(),
                nonempty_finals(&factor_minus_parser),
                parser_minus_factor.num_states(),
                parser_minus_factor.num_transitions(),
                nonempty_finals(&parser_minus_factor),
                concrete_ideal.num_states(),
                concrete_ideal.num_transitions(),
                factor_general_minimized.num_states(),
                factor_general_minimized.num_transitions(),
                factor_accepted_tokens.len(),
                concrete_ideal_minimized.num_states(),
                concrete_ideal_minimized.num_transitions(),
                ideal_accepted_tokens.len(),
            );
            acyclic_path_stats(
                "outer_exact_concrete_B_minus_A_ideal_intersection",
                &concrete_ideal_minimized,
            );
            residual_diagnostics(
                "outer_exact_concrete_B_minus_A_ideal_intersection",
                &concrete_ideal_minimized,
                &common_id_map,
                &terminal_names,
            );
            eprintln!(
                "MINBOUND OUTER_CONCRETE_IDEAL_ACCEPTED_TOKENS {:?}",
                ideal_accepted_tokens,
            );
            if std::env::var_os("GLRMASK_MINBOUND_PROFILE_INTERFACE_PREFILTER").is_some() {
                let base_pairs = visible_boundary_interface_pairs(
                    &concrete_grammar,
                    &composed_table.boundary_nonterminals,
                    &composed_table.control_terminals,
                );
                let extended_pairs = extend_boundary_interfaces_through_stack_neutral_lr_actions(
                    &composed_table.table,
                    &base_pairs,
                );
                let owner = |terminal: u32| {
                    composed_table
                        .terminal_offsets
                        .partition_point(|&offset| offset <= terminal)
                        .saturating_sub(1)
                };
                let cross_base = base_pairs
                    .iter()
                    .copied()
                    .filter(|&(left, right)| owner(left) != owner(right))
                    .collect::<BTreeSet<_>>();
                let cross_extended = extended_pairs
                    .iter()
                    .copied()
                    .filter(|&(left, right)| owner(left) != owner(right))
                    .collect::<BTreeSet<_>>();
                let all_candidates = boundary_interface_adjacent_pair_candidates(
                    &vocab,
                    &[&core, &dispatch],
                    &composed_table.terminal_offsets,
                    &extended_pairs,
                )
                .into_iter()
                .collect::<BTreeSet<_>>();
                let cross_candidates = boundary_interface_adjacent_pair_candidates(
                    &vocab,
                    &[&core, &dispatch],
                    &composed_table.terminal_offsets,
                    &cross_extended,
                )
                .into_iter()
                .collect::<BTreeSet<_>>();
                let missing_all = ideal_accepted_tokens
                    .difference(&all_candidates)
                    .copied()
                    .collect::<Vec<_>>();
                let missing_cross = ideal_accepted_tokens
                    .difference(&cross_candidates)
                    .copied()
                    .collect::<Vec<_>>();
                eprintln!(
                    "MINBOUND INTERFACE_PREFILTER base_pairs={} extended_pairs={} cross_base_pairs={} cross_extended_pairs={} all_candidates={} cross_candidates={} ideal_tokens={} missing_all={} missing_cross={} missing_all_ids={missing_all:?} missing_cross_ids={missing_cross:?}",
                    base_pairs.len(),
                    extended_pairs.len(),
                    cross_base.len(),
                    cross_extended.len(),
                    all_candidates.len(),
                    cross_candidates.len(),
                    ideal_accepted_tokens.len(),
                    missing_all.len(),
                    missing_cross.len(),
                );
            }
            save_terminal_capture(
                std::path::Path::new(&root)
                    .join("tdwa_outer_exact_concrete_B_minus_A_ideal_intersection.cap")
                    .to_str()
                    .unwrap(),
                &MappedArtifact::new(concrete_ideal_minimized.clone(), common_id_map.clone()),
                &terminal_names,
            );
            if std::env::var_os("GLRMASK_MINBOUND_STOP_AFTER_FACTOR_COMPARE").is_some() {
                return;
            }
            let concrete_tight_weight = Weight::union_all(
                concrete_grammar_tight
                    .states()
                    .iter()
                    .filter_map(|state| state.final_weight.as_ref()),
            );
            let mut concrete_tight_original_tokens = BTreeSet::<u32>::new();
            for (_, internal_tokens) in concrete_tight_weight.raw_range_values() {
                for range in internal_tokens.ranges() {
                    for internal_token in range {
                        if let Some(originals) = common_id_map
                            .vocab_tokens
                            .internal_to_originals
                            .get(internal_token as usize)
                        {
                            concrete_tight_original_tokens.extend(originals.iter().copied());
                        }
                    }
                }
            }
            eprintln!(
                "MINBOUND OUTER_CONCRETE_GRAMMAR_TIGHT template_ms={:.3} states={} transitions={} original_tokens={} removed_tokens={} tokens={:?}",
                concrete_template_ms,
                concrete_grammar_tight.num_states(),
                concrete_grammar_tight.num_transitions(),
                concrete_tight_original_tokens.len(),
                concrete_original_tokens.difference(&concrete_tight_original_tokens).count(),
                concrete_tight_original_tokens,
            );
            let concrete_tight_capture = MappedArtifact::new(
                concrete_grammar_tight.clone(),
                common_id_map.clone(),
            );
            save_terminal_capture(
                std::path::Path::new(&root)
                    .join("tdwa_outer_exact_concrete_B_minus_A_parser_domain_tight.cap")
                    .to_str()
                    .unwrap(),
                &concrete_tight_capture,
                &terminal_names,
            );

            let mut tight_selected = vec![false; composed_table.table.num_terminals as usize];
            for state in concrete_grammar_tight.states() {
                for &label in state.transitions.keys() {
                    if label >= 0 && (label as usize) < tight_selected.len() {
                        tight_selected[label as usize] = true;
                    }
                }
            }
            let tight_active_terminals = tight_selected.iter().filter(|&&yes| yes).count();
            let tight_template_started = Instant::now();
            let (tight_templates, _, _) = build_composition_templates(
                &composed_table.table,
                &concrete_grammar,
                &tight_selected,
            );
            let tight_template_ms = tight_template_started.elapsed().as_secs_f64() * 1000.0;
            let tight_delta_started = Instant::now();
            let tight_delta_components = [&core, &dispatch];
            let tight_delta_plan = prepare_concrete_boundary_delta_plan(
                &composed_table,
                &tight_delta_components,
                &tight_selected,
                &tight_templates,
                concrete_grammar.num_terminals,
            );
            let deferred_plan_started = Instant::now();
            let deferred_plan = try_prepare_cached_composition_template_plan(
                &composed_table,
                &tight_delta_components,
                &concrete_grammar,
                &tight_selected,
                &merged_ignores.scoped,
            )
            .expect("ideal selected10 boundary should admit cached-template plan");
            let deferred_plan_ms = deferred_plan_started.elapsed().as_secs_f64() * 1000.0;
            let deferred_reused = deferred_plan.reused_dfas.len();
            let deferred_patched = deferred_plan.characterization_deltas.len();
            let deferred_fresh = deferred_plan.fresh_characterizations.len();
            let deferred_materialize_started = Instant::now();
            let deferred_templates = deferred_plan.materialize(&core);
            let deferred_materialize_ms =
                deferred_materialize_started.elapsed().as_secs_f64() * 1000.0;
            let mut deferred_mismatches = Vec::new();
            for (terminal, &active) in tight_selected.iter().enumerate() {
                if !active {
                    continue;
                }
                let terminal = terminal as u32;
                let candidate = deferred_templates
                    .by_terminal
                    .get(&terminal)
                    .expect("deferred ideal template plan covers active terminal");
                let reference = tight_templates
                    .by_terminal
                    .get(&terminal)
                    .expect("full ideal template reference covers active terminal");
                if !unweighted_dfa_language_is_empty(&unweighted_dfa_difference(candidate, reference))
                    || !unweighted_dfa_language_is_empty(&unweighted_dfa_difference(reference, candidate))
                {
                    deferred_mismatches.push(terminal);
                }
            }
            eprintln!(
                "MINBOUND OUTER_IDEAL_DEFERRED_TEMPLATE_PLAN active={} reused={} patched={} fresh={} prep_ms={deferred_plan_ms:.3} materialize_ms={deferred_materialize_ms:.3} mismatches={} mismatch_ids={:?}",
                tight_active_terminals,
                deferred_reused,
                deferred_patched,
                deferred_fresh,
                deferred_mismatches.len(),
                deferred_mismatches,
            );
            if std::env::var_os("GLRMASK_EXPERIMENT_CACHED_BOUNDARY_TEMPLATES").is_some() {
                let cached_production_started = Instant::now();
                let (cached_templates, cached_commit_templates, cached_reported_ms) =
                    try_build_cached_composition_templates(
                        &composed_table,
                        &tight_delta_components,
                        &concrete_grammar,
                        &tight_selected,
                        &merged_ignores.scoped,
                    )
                    .expect("ideal selected10 boundary should satisfy cached-template proof");
                let mut cached_mismatches = Vec::new();
                for (terminal, &active) in tight_selected.iter().enumerate() {
                    if !active {
                        continue;
                    }
                    let terminal = terminal as u32;
                    let candidate = cached_templates
                        .by_terminal
                        .get(&terminal)
                        .expect("cached production path covers ideal terminal");
                    let reference = tight_templates
                        .by_terminal
                        .get(&terminal)
                        .expect("full reference covers ideal terminal");
                    if !unweighted_dfa_language_is_empty(&unweighted_dfa_difference(candidate, reference))
                        || !unweighted_dfa_language_is_empty(&unweighted_dfa_difference(reference, candidate))
                    {
                        cached_mismatches.push(terminal);
                    }
                }
                eprintln!(
                    "MINBOUND OUTER_IDEAL_PRODUCTION_CACHED_TEMPLATES active={} raw_templates={} commit_templates={} reported_ms={cached_reported_ms:.3} wall_ms={:.3} mismatches={} mismatch_ids={:?}",
                    tight_active_terminals,
                    cached_templates.by_terminal.len(),
                    cached_commit_templates.iter().flatten().count(),
                    cached_production_started.elapsed().as_secs_f64() * 1000.0,
                    cached_mismatches.len(),
                    cached_mismatches,
                );
            }
            eprintln!(
                "MINBOUND OUTER_IDEAL_TEMPLATE_REUSE active={} changed={} unsafe={} unchanged={} changed_ids={:?} compare_ms={:.3}",
                tight_active_terminals,
                tight_delta_plan.by_global_terminal.len(),
                tight_delta_plan.unsafe_terminals.len(),
                tight_active_terminals
                    .saturating_sub(tight_delta_plan.by_global_terminal.len())
                    .saturating_sub(tight_delta_plan.unsafe_terminals.len()),
                tight_delta_plan
                    .by_global_terminal
                    .keys()
                    .copied()
                    .collect::<Vec<_>>(),
                tight_delta_started.elapsed().as_secs_f64() * 1000.0,
            );
            let split_reuse_started = Instant::now();
            let old_transported = rebuild_transported_component_templates(
                &composed_table,
                &tight_delta_components,
                &tight_selected,
            );
            let mut split_present = 0usize;
            let mut split_exact = 0usize;
            let mut split_exact_changed = 0usize;
            let mut split_exact_unchanged = 0usize;
            let mut split_mismatches = Vec::<u32>::new();
            let mut split_missing = Vec::<u32>::new();
            for (terminal, &active) in tight_selected.iter().enumerate() {
                if !active {
                    continue;
                }
                let terminal = terminal as u32;
                let component_index = composed_table
                    .terminal_offsets
                    .partition_point(|&offset| offset <= terminal)
                    .saturating_sub(1);
                let Some(component) = tight_delta_components.get(component_index).copied() else {
                    split_missing.push(terminal);
                    continue;
                };
                let local = terminal - composed_table.terminal_offsets[component_index];
                let Some(split) = component
                    .template_dfas_by_terminal
                    .get(local as usize)
                    .and_then(|entry| entry.as_deref())
                else {
                    split_missing.push(terminal);
                    continue;
                };
                split_present += 1;
                let reconstructed = crate::compiler::stages::templates::compile_dfa::recombine_split_commit_template_dfa(split);
                let Some(transported) = transport_composition_template_dfa(
                    reconstructed,
                    &composed_table.state_relations[component_index],
                ) else {
                    split_mismatches.push(terminal);
                    continue;
                };
                let Some(reference) = old_transported.get(&terminal) else {
                    split_mismatches.push(terminal);
                    continue;
                };
                let reconstructed_minus_reference =
                    unweighted_dfa_difference(&transported, reference);
                let reference_minus_reconstructed =
                    unweighted_dfa_difference(reference, &transported);
                if unweighted_dfa_language_is_empty(&reconstructed_minus_reference)
                    && unweighted_dfa_language_is_empty(&reference_minus_reconstructed)
                {
                    split_exact += 1;
                    if tight_delta_plan.by_global_terminal.contains_key(&terminal) {
                        split_exact_changed += 1;
                    } else {
                        split_exact_unchanged += 1;
                    }
                } else {
                    split_mismatches.push(terminal);
                }
            }
            eprintln!(
                "MINBOUND OUTER_IDEAL_SERIALIZED_TEMPLATE_REUSE active={} split_present={} split_exact={} split_exact_changed={} split_exact_unchanged={} split_missing={} split_mismatches={} mismatch_ids={:?} missing_ids={:?} ms={:.3}",
                tight_active_terminals,
                split_present,
                split_exact,
                split_exact_changed,
                split_exact_unchanged,
                split_missing.len(),
                split_mismatches.len(),
                split_mismatches,
                split_missing,
                split_reuse_started.elapsed().as_secs_f64() * 1000.0,
            );
            let hybrid_started = Instant::now();
            let parent_terminal_end = composed_table.terminal_offsets[1];
            let mut parent_selected = vec![false; tight_selected.len()];
            let mut child_selected = tight_selected.clone();
            for terminal in 0..parent_terminal_end as usize {
                parent_selected[terminal] = tight_selected[terminal];
                child_selected[terminal] = false;
            }
            let ((mut hybrid_dfas, hybrid_transport_ms), (parent_templates, parent_characterize_ms, parent_template_ms)) =
                macro_join(
                    "compose_hybrid_template_transport_and_parent_templates",
                    || {
                        let started = Instant::now();
                        let result = rebuild_transported_component_templates(
                            &composed_table,
                            &tight_delta_components,
                            &child_selected,
                        );
                        (result, started.elapsed().as_secs_f64() * 1000.0)
                    },
                    || {
                        let started = Instant::now();
                        let parent_characterizations = characterize_selected_terminals(
                            &composed_table.table,
                            &concrete_grammar,
                            &parent_selected,
                        );
                        let parent_characterize_ms =
                            started.elapsed().as_secs_f64() * 1000.0;
                        let started = Instant::now();
                        let (parent_templates, _) =
                            Templates::from_characterizations_profiled(&parent_characterizations);
                        let parent_template_ms = started.elapsed().as_secs_f64() * 1000.0;
                        (parent_templates, parent_characterize_ms, parent_template_ms)
                    },
                );
            hybrid_dfas.extend(parent_templates.by_terminal);
            let skeleton_started = Instant::now();
            let hybrid_templates = Templates::from_terminal_dfas(hybrid_dfas);
            let skeleton_ms = skeleton_started.elapsed().as_secs_f64() * 1000.0;
            let validate_started = Instant::now();
            let mut hybrid_mismatches = Vec::<u32>::new();
            for (terminal, &active) in tight_selected.iter().enumerate() {
                if !active {
                    continue;
                }
                let terminal = terminal as u32;
                let Some(reference) = tight_templates.by_terminal.get(&terminal) else {
                    hybrid_mismatches.push(terminal);
                    continue;
                };
                let Some(candidate) = hybrid_templates.by_terminal.get(&terminal) else {
                    hybrid_mismatches.push(terminal);
                    continue;
                };
                if !unweighted_dfa_language_is_empty(&unweighted_dfa_difference(candidate, reference))
                    || !unweighted_dfa_language_is_empty(&unweighted_dfa_difference(reference, candidate))
                {
                    hybrid_mismatches.push(terminal);
                }
            }
            let validate_ms = validate_started.elapsed().as_secs_f64() * 1000.0;
            let child_cache_hits = tight_selected
                .iter()
                .enumerate()
                .skip(parent_terminal_end as usize)
                .filter(|(_, active)| **active)
                .filter(|(terminal, _)| {
                    dispatch
                        .composition_parser_templates_by_terminal
                        .get(*terminal - parent_terminal_end as usize)
                        .is_some_and(Option::is_some)
                })
                .count();
            eprintln!(
                "MINBOUND OUTER_IDEAL_HYBRID_TEMPLATES active={} parent_active={} child_active={} child_cache_hits={} mismatches={} mismatch_ids={:?} states={} transitions={} transport_ms={hybrid_transport_ms:.3} parent_characterize_ms={parent_characterize_ms:.3} parent_template_ms={parent_template_ms:.3} skeleton_ms={skeleton_ms:.3} validate_ms={validate_ms:.3} total_ms={:.3}",
                tight_active_terminals,
                parent_selected.iter().filter(|&&selected| selected).count(),
                tight_active_terminals.saturating_sub(parent_selected.iter().filter(|&&selected| selected).count()),
                child_cache_hits,
                hybrid_mismatches.len(),
                hybrid_mismatches,
                hybrid_templates.by_terminal.values().map(|dfa| dfa.states.len()).sum::<usize>(),
                hybrid_templates.by_terminal.values().flat_map(|dfa| dfa.states.iter()).map(|state| state.transitions.len()).sum::<usize>(),
                hybrid_started.elapsed().as_secs_f64() * 1000.0,
            );
            let direct_delta_started = Instant::now();
            let parent_relation_is_identity = composed_table.state_relations[0]
                .iter()
                .enumerate()
                .all(|(local, targets)| targets.as_slice() == [local as u32]);
            assert!(
                parent_relation_is_identity,
                "selected10 direct parent-delta oracle currently requires identity parent state transport",
            );
            let mut action_seeds =
                vec![Vec::<u32>::new(); concrete_grammar.num_terminals as usize];
            let mut removed_parent_actions = Vec::<(u32, u32)>::new();
            for terminal in 0..parent_terminal_end {
                if !tight_selected[terminal as usize] {
                    continue;
                }
                for state in 0..composed_table.table.num_states {
                    let new_action = composed_table.table.action(state, terminal);
                    let old_action = if state < core.table.num_states {
                        core.table.action(state, terminal)
                    } else {
                        None
                    };
                    let new_forwarded = composed_table
                        .table
                        .forwarded_shifts
                        .contains(&(state, terminal));
                    let old_forwarded = state < core.table.num_states
                        && core.table.forwarded_shifts.contains(&(state, terminal));
                    if new_action != old_action || new_forwarded != old_forwarded {
                        if new_action.is_some() {
                            action_seeds[terminal as usize].push(state);
                        } else if old_action.is_some() {
                            removed_parent_actions.push((state, terminal));
                        }
                    }
                }
            }
            let seed_count = action_seeds.iter().map(Vec::len).sum::<usize>();
            let seed_terminals = action_seeds.iter().filter(|states| !states.is_empty()).count();
            let seed_characterize_started = Instant::now();
            let seeded_characterizations = crate::compiler::stages::templates::characterize::characterize_terminal_action_state_seeds(
                &composed_table.table,
                &concrete_grammar,
                &action_seeds,
            );
            let seed_characterize_ms =
                seed_characterize_started.elapsed().as_secs_f64() * 1000.0;
            let seed_template_started = Instant::now();
            let seeded_templates = Templates::from_characterizations(&seeded_characterizations);
            let seed_template_ms = seed_template_started.elapsed().as_secs_f64() * 1000.0;
            let empty_dfa = UnweightedDfa::new();
            let mut delta_mismatches = Vec::<u32>::new();
            let mut delta_mismatch_witnesses = Vec::<(u32, Option<Vec<i32>>, Option<Vec<i32>>)>::new();
            let mut seeded_delta_states = 0usize;
            let mut exact_delta_states = 0usize;
            for terminal in 0..parent_terminal_end {
                if !tight_selected[terminal as usize] {
                    continue;
                }
                let old = old_transported
                    .get(&terminal)
                    .expect("active parent terminal has transported old template");
                let seeded = seeded_templates
                    .by_terminal
                    .get(&terminal)
                    .unwrap_or(&empty_dfa);
                let seeded_delta = trim_unweighted_dfa_productive(
                    unweighted_dfa_difference(seeded, old),
                );
                let exact_delta = tight_delta_plan
                    .by_global_terminal
                    .get(&terminal)
                    .map(|entry| &entry.delta_template)
                    .unwrap_or(&empty_dfa);
                seeded_delta_states += seeded_delta.states.len();
                exact_delta_states += exact_delta.states.len();
                let seeded_minus_exact = unweighted_dfa_difference(&seeded_delta, exact_delta);
                let exact_minus_seeded = unweighted_dfa_difference(exact_delta, &seeded_delta);
                if !unweighted_dfa_language_is_empty(&seeded_minus_exact)
                    || !unweighted_dfa_language_is_empty(&exact_minus_seeded)
                {
                    delta_mismatches.push(terminal);
                    if delta_mismatch_witnesses.len() < 8 {
                        delta_mismatch_witnesses.push((
                            terminal,
                            unweighted_dfa_shortest_word(&seeded_minus_exact),
                            unweighted_dfa_shortest_word(&exact_minus_seeded),
                        ));
                    }
                }
            }
            eprintln!(
                "MINBOUND OUTER_IDEAL_DIRECT_PARENT_DELTA parent_active={} seed_terminals={} seed_states={} removed_parent_actions={} seeded_delta_states={} exact_delta_states={} mismatches={} mismatch_ids={:?} witnesses={:?} characterize_ms={seed_characterize_ms:.3} template_ms={seed_template_ms:.3} total_ms={:.3}",
                parent_selected.iter().filter(|&&selected| selected).count(),
                seed_terminals,
                seed_count,
                removed_parent_actions.len(),
                seeded_delta_states,
                exact_delta_states,
                delta_mismatches.len(),
                delta_mismatches,
                delta_mismatch_witnesses,
                direct_delta_started.elapsed().as_secs_f64() * 1000.0,
            );
            let symbolic_delta_started = Instant::now();
            let core_augmented_start = core.table.rules[0].lhs;
            let core_analyzed = AnalyzedGrammar::from_composed_rules(
                core.table.rules.clone(),
                core.table.num_terminals,
                core.terminal_display_names().to_vec(),
                core.table.nonterminal_display_names.clone(),
                core_augmented_start,
            );
            let mut core_selected = vec![false; core.table.num_terminals as usize];
            for terminal in 0..parent_terminal_end as usize {
                core_selected[terminal] = tight_selected[terminal];
            }
            let core_characterize_started = Instant::now();
            let core_characterizations = characterize_selected_terminals(
                &core.table,
                &core_analyzed,
                &core_selected,
            );
            let core_characterize_ms =
                core_characterize_started.elapsed().as_secs_f64() * 1000.0;
            let composed_parent_characterizations = characterize_selected_terminals(
                &composed_table.table,
                &concrete_grammar,
                &parent_selected,
            );
            fn sorted_vec_difference<T: Ord + Clone>(new: &[T], old: &[T]) -> Vec<T> {
                let old = old.iter().cloned().collect::<BTreeSet<_>>();
                new.iter()
                    .filter(|value| !old.contains(*value))
                    .cloned()
                    .collect()
            }
            let mut symbolic_deltas = BTreeMap::new();
            let mut symbolic_removed = 0usize;
            let mut symbolic_delta_items = [0usize; 4];
            let mut symbolic_rereduce_examples = Vec::new();
            let mut symbolic_reduce_examples = Vec::new();
            for terminal in 0..parent_terminal_end {
                if !tight_selected[terminal as usize] {
                    continue;
                }
                let old = core_characterizations
                    .get(&terminal)
                    .expect("active core terminal has characterization");
                let new = composed_parent_characterizations
                    .get(&terminal)
                    .expect("active composed parent terminal has characterization");
                symbolic_removed += old
                    .escapes
                    .iter()
                    .filter(|item| !new.escapes.contains(item))
                    .count();
                symbolic_removed += old
                    .reduces
                    .iter()
                    .filter(|item| !new.reduces.contains(item))
                    .count();
                symbolic_removed += old
                    .nt_escapes
                    .iter()
                    .filter(|item| !new.nt_escapes.contains(item))
                    .count();
                symbolic_removed += old
                    .nt_rereduces
                    .iter()
                    .filter(|item| !new.nt_rereduces.contains(item))
                    .count();
                let escapes = sorted_vec_difference(&new.escapes, &old.escapes);
                let reduces = sorted_vec_difference(&new.reduces, &old.reduces);
                let nt_escapes = sorted_vec_difference(&new.nt_escapes, &old.nt_escapes);
                let nt_rereduces = sorted_vec_difference(&new.nt_rereduces, &old.nt_rereduces);
                symbolic_delta_items[0] += escapes.len();
                symbolic_delta_items[1] += reduces.len();
                symbolic_delta_items[2] += nt_escapes.len();
                symbolic_delta_items[3] += nt_rereduces.len();
                if symbolic_rereduce_examples.len() < 12 {
                    symbolic_rereduce_examples.extend(
                        nt_rereduces
                            .iter()
                            .take(12 - symbolic_rereduce_examples.len())
                            .cloned()
                            .map(|item| (terminal, item)),
                    );
                }
                if symbolic_reduce_examples.len() < 12 {
                    symbolic_reduce_examples.extend(
                        reduces
                            .iter()
                            .take(12 - symbolic_reduce_examples.len())
                            .cloned()
                            .map(|item| (terminal, item)),
                    );
                }
                if !escapes.is_empty()
                    || !reduces.is_empty()
                    || !nt_escapes.is_empty()
                    || !nt_rereduces.is_empty()
                {
                    symbolic_deltas.insert(
                        terminal,
                        crate::compiler::stages::templates::characterize::TerminalCharacterization {
                            escapes,
                            reduces,
                            nt_escapes,
                            nt_rereduces,
                            all_nts: new.all_nts.clone(),
                        },
                    );
                }
            }
            let symbolic_template_started = Instant::now();
            let symbolic_delta_templates = Templates::from_characterizations(&symbolic_deltas);
            let symbolic_template_ms =
                symbolic_template_started.elapsed().as_secs_f64() * 1000.0;
            let mut symbolic_mismatches = Vec::<u32>::new();
            for terminal in 0..parent_terminal_end {
                if !tight_selected[terminal as usize] {
                    continue;
                }
                let candidate = symbolic_delta_templates
                    .by_terminal
                    .get(&terminal)
                    .unwrap_or(&empty_dfa);
                let exact = tight_delta_plan
                    .by_global_terminal
                    .get(&terminal)
                    .map(|entry| &entry.delta_template)
                    .unwrap_or(&empty_dfa);
                if !unweighted_dfa_language_is_empty(&unweighted_dfa_difference(candidate, exact))
                    || !unweighted_dfa_language_is_empty(&unweighted_dfa_difference(exact, candidate))
                {
                    symbolic_mismatches.push(terminal);
                }
            }
            eprintln!(
                "MINBOUND OUTER_IDEAL_SYMBOLIC_PARENT_DELTA core_characterize_ms={core_characterize_ms:.3} symbolic_removed={} delta_escapes={} delta_reduces={} delta_nt_escapes={} delta_nt_rereduces={} delta_terminals={} rereduce_examples={:?} reduce_examples={:?} template_ms={symbolic_template_ms:.3} mismatches={} mismatch_ids={:?} total_ms={:.3}",
                symbolic_removed,
                symbolic_delta_items[0],
                symbolic_delta_items[1],
                symbolic_delta_items[2],
                symbolic_delta_items[3],
                symbolic_deltas.len(),
                symbolic_rereduce_examples,
                symbolic_reduce_examples,
                symbolic_mismatches.len(),
                symbolic_mismatches,
                symbolic_delta_started.elapsed().as_secs_f64() * 1000.0,
            );
            let direct_symbolic_started = Instant::now();
            let mut nt_predecessor_seeds =
                vec![Vec::<(u32, u32, u32, bool)>::new(); concrete_grammar.num_terminals as usize];
            for (revealed_state, row) in composed_table.table.goto.iter().enumerate() {
                for &boundary_nonterminal in &composed_table.boundary_nonterminals {
                    let Some(&(top_state, goto_replace)) = row.get(&boundary_nonterminal) else {
                        continue;
                    };
                    for terminal in 0..parent_terminal_end {
                        if !tight_selected[terminal as usize]
                            || composed_table.table.action(top_state, terminal).is_none()
                        {
                            continue;
                        }
                        nt_predecessor_seeds[terminal as usize].push((
                            top_state,
                            revealed_state as u32,
                            boundary_nonterminal,
                            goto_replace,
                        ));
                    }
                }
            }
            for seeds in &mut nt_predecessor_seeds {
                seeds.sort_unstable();
                seeds.dedup();
            }
            let nt_seed_count = nt_predecessor_seeds.iter().map(Vec::len).sum::<usize>();
            let nt_seed_characterizations = crate::compiler::stages::templates::characterize::
                characterize_terminal_nt_predecessor_seeds(
                    &composed_table.table,
                    &concrete_grammar,
                    &nt_predecessor_seeds,
                );
            let mut direct_symbolic = BTreeMap::new();
            for terminal in 0..parent_terminal_end {
                if !tight_selected[terminal as usize] {
                    continue;
                }
                let initial = seeded_characterizations.get(&terminal);
                let nonterminal = nt_seed_characterizations.get(&terminal);
                if initial.is_none() && nonterminal.is_none() {
                    continue;
                }
                let mut escapes = Vec::new();
                let mut reduces = Vec::new();
                let mut nt_escapes = Vec::new();
                let mut nt_rereduces = Vec::new();
                let mut all_nts = BTreeSet::new();
                for characterization in [initial, nonterminal].into_iter().flatten() {
                    escapes.extend(characterization.escapes.iter().cloned());
                    reduces.extend(characterization.reduces.iter().cloned());
                    nt_escapes.extend(characterization.nt_escapes.iter().cloned());
                    nt_rereduces.extend(characterization.nt_rereduces.iter().cloned());
                    all_nts.extend(characterization.all_nts.iter().copied());
                }
                escapes.sort();
                escapes.dedup();
                reduces.sort();
                reduces.dedup();
                nt_escapes.sort();
                nt_escapes.dedup();
                nt_rereduces.sort();
                nt_rereduces.dedup();
                direct_symbolic.insert(
                    terminal,
                    crate::compiler::stages::templates::characterize::TerminalCharacterization {
                        escapes,
                        reduces,
                        nt_escapes,
                        nt_rereduces,
                        all_nts,
                    },
                );
            }
            let mut direct_symbolic_mismatches = Vec::new();
            let mut direct_symbolic_examples = Vec::new();
            for terminal in 0..parent_terminal_end {
                if !tight_selected[terminal as usize] {
                    continue;
                }
                let reference = symbolic_deltas.get(&terminal);
                let candidate = direct_symbolic.get(&terminal);
                if candidate != reference {
                    direct_symbolic_mismatches.push(terminal);
                    if direct_symbolic_examples.len() < 3 {
                        direct_symbolic_examples.push((terminal, candidate.cloned(), reference.cloned()));
                    }
                }
            }
            eprintln!(
                "MINBOUND OUTER_IDEAL_DIRECT_SYMBOLIC_PATCH action_seed_states={} nt_predecessor_seeds={} terminals={} mismatches={} mismatch_ids={:?} examples={:?} total_ms={:.3}",
                seed_count,
                nt_seed_count,
                direct_symbolic.len(),
                direct_symbolic_mismatches.len(),
                direct_symbolic_mismatches,
                direct_symbolic_examples,
                direct_symbolic_started.elapsed().as_secs_f64() * 1000.0,
            );
            let fast_template_path_started = Instant::now();
            let mut reuse_selected = tight_selected.clone();
            for terminal in 0..parent_terminal_end {
                if direct_symbolic.contains_key(&terminal) {
                    reuse_selected[terminal as usize] = false;
                }
            }
            let fast_transport_started = Instant::now();
            let mut fast_dfas = rebuild_transported_component_templates(
                &composed_table,
                &tight_delta_components,
                &reuse_selected,
            );
            let fast_transport_ms =
                fast_transport_started.elapsed().as_secs_f64() * 1000.0;
            let fast_patch_assembly_started = Instant::now();
            let mut fast_patched_characterizations = BTreeMap::new();
            let mut fast_characterization_cache_hits = 0usize;
            for (&terminal, delta) in &direct_symbolic {
                let mut patched = core
                    .composition_parser_characterizations_by_terminal
                    .get(terminal as usize)
                    .and_then(Option::as_ref)
                    .unwrap_or_else(|| {
                        core_characterizations
                            .get(&terminal)
                            .expect("changed parent terminal has characterization source")
                    })
                    .clone();
                if core
                    .composition_parser_characterizations_by_terminal
                    .get(terminal as usize)
                    .is_some_and(Option::is_some)
                {
                    fast_characterization_cache_hits += 1;
                }
                patched.escapes.extend(delta.escapes.iter().cloned());
                patched.reduces.extend(delta.reduces.iter().cloned());
                patched.nt_escapes.extend(delta.nt_escapes.iter().cloned());
                patched.nt_rereduces.extend(delta.nt_rereduces.iter().cloned());
                patched.escapes.sort();
                patched.escapes.dedup();
                patched.reduces.sort();
                patched.reduces.dedup();
                patched.nt_escapes.sort();
                patched.nt_escapes.dedup();
                patched.nt_rereduces.sort();
                patched.nt_rereduces.dedup();
                patched.all_nts.extend(delta.all_nts.iter().copied());
                fast_patched_characterizations.insert(terminal, patched);
            }
            let fast_patch_assembly_ms =
                fast_patch_assembly_started.elapsed().as_secs_f64() * 1000.0;
            let fast_patch_compile_started = Instant::now();
            let fast_patched_templates =
                Templates::from_characterizations(&fast_patched_characterizations);
            let fast_patch_compile_ms =
                fast_patch_compile_started.elapsed().as_secs_f64() * 1000.0;
            fast_dfas.extend(fast_patched_templates.by_terminal);
            let fast_skeleton_started = Instant::now();
            let fast_templates = Templates::from_terminal_dfas(fast_dfas);
            let fast_skeleton_ms = fast_skeleton_started.elapsed().as_secs_f64() * 1000.0;
            let fast_commit_started = Instant::now();
            let split = |(&terminal, dfa): (&u32, &UnweightedDfa)| {
                    let commit_dfa = specialize_template_dfa_defaults_for_commit_split_input(dfa);
                    try_split_commit_template_dfas(&commit_dfa)
                        .map(|split| (terminal, Arc::new(split)))
            };
            let fast_commit_templates = if macro_parallelism_disabled() {
                let mut timings = Vec::with_capacity(fast_templates.by_terminal.len());
                let result = fast_templates
                    .by_terminal
                    .iter()
                    .filter_map(|item| {
                        let started = Instant::now();
                        let result = split(item);
                        timings.push(started.elapsed().as_secs_f64() * 1000.0);
                        result
                    })
                    .collect::<Vec<_>>();
                report_macro_item_timings("compose_fast_commit_template_splits", &timings);
                result
            } else {
                fast_templates
                    .by_terminal
                    .par_iter()
                    .filter_map(split)
                    .collect::<Vec<_>>()
            };
            let fast_commit_ms = fast_commit_started.elapsed().as_secs_f64() * 1000.0;
            let fast_validate_started = Instant::now();
            let mut fast_template_mismatches = Vec::new();
            for (terminal, &active) in tight_selected.iter().enumerate() {
                if !active {
                    continue;
                }
                let terminal = terminal as u32;
                let candidate = fast_templates
                    .by_terminal
                    .get(&terminal)
                    .expect("fast ideal template path covers every active terminal");
                let reference = tight_templates
                    .by_terminal
                    .get(&terminal)
                    .expect("reference ideal template covers every active terminal");
                if !unweighted_dfa_language_is_empty(&unweighted_dfa_difference(candidate, reference))
                    || !unweighted_dfa_language_is_empty(&unweighted_dfa_difference(reference, candidate))
                {
                    fast_template_mismatches.push(terminal);
                }
            }
            let fast_validate_ms = fast_validate_started.elapsed().as_secs_f64() * 1000.0;
            let fast_nwa_started = Instant::now();
            let fast_parser_nwa = crate::compiler::stages::parser_dwa::
                build_parser_nwa_from_terminal_dwa_with_precomputed_templates(
                    &TerminalAutomaton::Dwa(concrete_grammar_tight.clone()),
                    &concrete_grammar,
                    &fast_templates,
                    &composed_table.table,
                )
                .expect("fast ideal templates should induce a parser NWA");
            let fast_nwa_ms = fast_nwa_started.elapsed().as_secs_f64() * 1000.0;
            eprintln!(
                "MINBOUND OUTER_IDEAL_FAST_TEMPLATE_NWA active={} reused={} patched={} characterization_cache_hits={} transport_ms={fast_transport_ms:.3} patch_assembly_ms={fast_patch_assembly_ms:.3} patch_compile_ms={fast_patch_compile_ms:.3} skeleton_ms={fast_skeleton_ms:.3} commit_templates={} commit_ms={fast_commit_ms:.3} validate_ms={fast_validate_ms:.3} mismatches={} nwa_states={} nwa_transitions={} nwa_ms={fast_nwa_ms:.3} counted_total_ms={:.3} total_with_validation_ms={:.3}",
                tight_active_terminals,
                reuse_selected.iter().filter(|&&selected| selected).count(),
                fast_patched_characterizations.len(),
                fast_characterization_cache_hits,
                fast_commit_templates.len(),
                fast_template_mismatches.len(),
                fast_parser_nwa.states().len(),
                fast_parser_nwa.states().iter().map(|state| state.epsilons.len() + state.transitions.values().map(Vec::len).sum::<usize>()).sum::<usize>(),
                fast_transport_ms + fast_patch_assembly_ms + fast_patch_compile_ms + fast_skeleton_ms + fast_commit_ms + fast_nwa_ms,
                fast_template_path_started.elapsed().as_secs_f64() * 1000.0,
            );
            let patched_characterization_started = Instant::now();
            let mut patched_characterizations = BTreeMap::new();
            for (&terminal, delta) in &symbolic_deltas {
                let mut patched = core_characterizations
                    .get(&terminal)
                    .expect("changed parent terminal has cached characterization source")
                    .clone();
                patched.escapes.extend(delta.escapes.iter().cloned());
                patched.reduces.extend(delta.reduces.iter().cloned());
                patched.nt_escapes.extend(delta.nt_escapes.iter().cloned());
                patched.nt_rereduces.extend(delta.nt_rereduces.iter().cloned());
                patched.escapes.sort();
                patched.escapes.dedup();
                patched.reduces.sort();
                patched.reduces.dedup();
                patched.nt_escapes.sort();
                patched.nt_escapes.dedup();
                patched.nt_rereduces.sort();
                patched.nt_rereduces.dedup();
                patched.all_nts.extend(delta.all_nts.iter().copied());
                patched_characterizations.insert(terminal, patched);
            }
            let patched_assembly_ms =
                patched_characterization_started.elapsed().as_secs_f64() * 1000.0;
            let patched_template_started = Instant::now();
            let patched_templates = Templates::from_characterizations(&patched_characterizations);
            let patched_template_ms =
                patched_template_started.elapsed().as_secs_f64() * 1000.0;
            let mut patched_mismatches = Vec::<u32>::new();
            for (&terminal, candidate) in &patched_templates.by_terminal {
                let reference = tight_templates
                    .by_terminal
                    .get(&terminal)
                    .expect("changed parent terminal has full template");
                if !unweighted_dfa_language_is_empty(&unweighted_dfa_difference(candidate, reference))
                    || !unweighted_dfa_language_is_empty(&unweighted_dfa_difference(reference, candidate))
                {
                    patched_mismatches.push(terminal);
                }
            }
            eprintln!(
                "MINBOUND OUTER_IDEAL_PATCHED_PARENT_CHARACTERIZATIONS changed={} assembly_ms={patched_assembly_ms:.3} template_ms={patched_template_ms:.3} mismatches={} mismatch_ids={:?} total_ms={:.3}",
                patched_characterizations.len(),
                patched_mismatches.len(),
                patched_mismatches,
                patched_characterization_started.elapsed().as_secs_f64() * 1000.0,
            );
            let prefix_factor_started = Instant::now();
            let mut prefix_counts = BTreeMap::<Vec<i32>, usize>::new();
            let mut residual_subset_old = 0usize;
            let mut residual_equal_old = 0usize;
            let mut residual_total_states = 0usize;
            let mut residual_examples = Vec::<(
                u32,
                Vec<i32>,
                Option<Vec<i32>>,
                Option<Vec<i32>>,
            )>::new();
            for (&terminal, entry) in &tight_delta_plan.by_global_terminal {
                if terminal >= parent_terminal_end {
                    continue;
                }
                let mut state = entry.delta_template.start_state;
                let mut prefix = Vec::<i32>::new();
                loop {
                    let node = &entry.delta_template.states[state as usize];
                    if node.is_accepting || node.transitions.len() != 1 {
                        break;
                    }
                    let (&label, &target) = node.transitions.iter().next().unwrap();
                    prefix.push(label);
                    state = target;
                    if prefix.len() >= 32 {
                        break;
                    }
                }
                *prefix_counts.entry(prefix.clone()).or_default() += 1;
                let mut residual = entry.delta_template.clone();
                residual.start_state = state;
                residual_total_states += residual.states.len();
                let residual_minus_old =
                    unweighted_dfa_difference(&residual, &entry.old_template);
                let old_minus_residual =
                    unweighted_dfa_difference(&entry.old_template, &residual);
                let subset = unweighted_dfa_language_is_empty(&residual_minus_old);
                let reverse_subset = unweighted_dfa_language_is_empty(&old_minus_residual);
                residual_subset_old += usize::from(subset);
                residual_equal_old += usize::from(subset && reverse_subset);
                if residual_examples.len() < 8 {
                    residual_examples.push((
                        terminal,
                        prefix,
                        unweighted_dfa_shortest_word(&residual_minus_old),
                        unweighted_dfa_shortest_word(&old_minus_residual),
                    ));
                }
            }
            eprintln!(
                "MINBOUND OUTER_IDEAL_PARENT_DELTA_PREFIX_FACTORS changed={} distinct_forced_prefixes={} prefix_counts={:?} residual_subset_old={} residual_equal_old={} residual_total_states={} examples={:?} ms={:.3}",
                tight_delta_plan
                    .by_global_terminal
                    .keys()
                    .filter(|&&terminal| terminal < parent_terminal_end)
                    .count(),
                prefix_counts.len(),
                prefix_counts,
                residual_subset_old,
                residual_equal_old,
                residual_total_states,
                residual_examples,
                prefix_factor_started.elapsed().as_secs_f64() * 1000.0,
            );
            if std::env::var_os("GLRMASK_MINBOUND_TEMPLATE_STORAGE").is_some() {
                let active_raw = bincode::serialize(&old_transported)
                    .expect("serialize active transported parser templates");
                let active_zstd = zstd::stream::encode_all(active_raw.as_slice(), 1)
                    .expect("compress active transported parser templates");
                let all_started = Instant::now();
                let all_selected = vec![true; composed_table.table.num_terminals as usize];
                let all_transported = rebuild_transported_component_templates(
                    &composed_table,
                    &tight_delta_components,
                    &all_selected,
                );
                let all_states = all_transported
                    .values()
                    .map(|dfa| dfa.states.len())
                    .sum::<usize>();
                let all_transitions = all_transported
                    .values()
                    .flat_map(|dfa| dfa.states.iter())
                    .map(|state| state.transitions.len())
                    .sum::<usize>();
                let all_raw = bincode::serialize(&all_transported)
                    .expect("serialize all transported parser templates");
                let all_zstd = zstd::stream::encode_all(all_raw.as_slice(), 1)
                    .expect("compress all transported parser templates");
                eprintln!(
                    "MINBOUND OUTER_TEMPLATE_STORAGE active_templates={} active_raw_bytes={} active_zstd_bytes={} all_templates={} all_states={} all_transitions={} all_raw_bytes={} all_zstd_bytes={} rebuild_all_ms={:.3}",
                    old_transported.len(),
                    active_raw.len(),
                    active_zstd.len(),
                    all_transported.len(),
                    all_states,
                    all_transitions,
                    all_raw.len(),
                    all_zstd.len(),
                    all_started.elapsed().as_secs_f64() * 1000.0,
                );
            }
            let tight_parser_nwa_started = Instant::now();
            let tight_parser_nwa = crate::compiler::stages::parser_dwa::
                build_parser_nwa_from_terminal_dwa_with_precomputed_templates(
                &TerminalAutomaton::Dwa(concrete_grammar_tight.clone()),
                &concrete_grammar,
                &tight_templates,
                &composed_table.table,
            )
            .expect("ideal boundary terminal DWA should induce a parser NWA");
            let tight_parser_nwa_ms =
                tight_parser_nwa_started.elapsed().as_secs_f64() * 1000.0;
            let tight_parser_nwa_transitions = tight_parser_nwa
                .states()
                .iter()
                .map(|state| {
                    state.epsilons.len()
                        + state
                            .transitions
                            .values()
                            .map(Vec::len)
                            .sum::<usize>()
                })
                .sum::<usize>();
            let tight_parser_nwa_negative_edges = tight_parser_nwa
                .states()
                .iter()
                .flat_map(|state| state.transitions.iter())
                .filter(|(label, _)| is_negative_label(**label))
                .map(|(_, targets)| targets.len())
                .sum::<usize>();
            eprintln!(
                "MINBOUND OUTER_IDEAL_BOUNDARY_PARSER_NWA states={} starts={} transitions={} negative_edges={} build_ms={tight_parser_nwa_ms:.3}",
                tight_parser_nwa.states().len(),
                tight_parser_nwa.start_states().len(),
                tight_parser_nwa_transitions,
                tight_parser_nwa_negative_edges,
            );
            if std::env::var_os("GLRMASK_MINBOUND_STOP_AFTER_TEMPLATE_REUSE").is_some() {
                return;
            }
            let tight_parser_started = Instant::now();
            let tight_parser_raw = build_parser_dwa_from_terminal_dwa_with_precomputed_templates(
                &composed_table.table,
                &concrete_grammar,
                &TerminalAutomaton::Dwa(concrete_grammar_tight.clone()),
                &tight_templates,
                &vocab,
                &common_id_map,
                false,
            );
            let tight_parser_build_ms = tight_parser_started.elapsed().as_secs_f64() * 1000.0;
            let tight_parser_raw_states = tight_parser_raw.num_states();
            let tight_parser_raw_transitions = tight_parser_raw.num_transitions();
            acyclic_parser_path_stats("outer_ideal_boundary_parser_raw", &tight_parser_raw);
            let tight_parser_raw_for_union = tight_parser_raw.clone();
            let tight_hashcons_started = Instant::now();
            let tight_parser_hashconsed = reverse_hashcons_owned(tight_parser_raw);
            let tight_hashcons_ms = tight_hashcons_started.elapsed().as_secs_f64() * 1000.0;
            let tight_parser_hashcons_states = tight_parser_hashconsed.num_states();
            let tight_parser_hashcons_transitions = tight_parser_hashconsed.num_transitions();
            let tight_parser_hashconsed_for_union = tight_parser_hashconsed.clone();
            let tight_parser_hashconsed_for_acyclic = tight_parser_hashconsed.clone();
            let tight_acyclic_started = Instant::now();
            let tight_parser_acyclic =
                crate::automata::weighted_u32::minimize_acyclic::minimize_acyclic_owned(
                    tight_parser_hashconsed_for_acyclic,
                );
            let tight_acyclic_ms = tight_acyclic_started.elapsed().as_secs_f64() * 1000.0;
            let tight_minimize_started = Instant::now();
            let tight_parser_minimized = minimize_owned(tight_parser_hashconsed);
            let tight_minimize_ms = tight_minimize_started.elapsed().as_secs_f64() * 1000.0;
            assert_eq!(
                find_difference(&tight_parser_minimized, &tight_parser_acyclic)
                    .expect("compare parser minimizers"),
                None,
                "acyclic and general parser minimizers differ",
            );
            acyclic_parser_path_stats("outer_ideal_boundary_parser_minimized", &tight_parser_minimized);
            let tight_parser_tokens = accepted_original_tokens(&tight_parser_minimized, &common_id_map);
            eprintln!(
                "MINBOUND OUTER_CONCRETE_GRAMMAR_TIGHT_PARSER active_terminals={} template_ms={:.3} build_ms={:.3} raw_states={} raw_transitions={} hashcons_ms={:.3} hashcons_states={} hashcons_transitions={} acyclic_minimize_ms={:.3} acyclic_states={} acyclic_transitions={} minimize_ms={:.3} minimized_states={} minimized_transitions={} original_tokens={} tokens={:?}",
                tight_active_terminals,
                tight_template_ms,
                tight_parser_build_ms,
                tight_parser_raw_states,
                tight_parser_raw_transitions,
                tight_hashcons_ms,
                tight_parser_hashcons_states,
                tight_parser_hashcons_transitions,
                tight_acyclic_ms,
                tight_parser_acyclic.num_states(),
                tight_parser_acyclic.num_transitions(),
                tight_minimize_ms,
                tight_parser_minimized.num_states(),
                tight_parser_minimized.num_transitions(),
                tight_parser_tokens.len(),
                tight_parser_tokens,
            );

            // Time the two parser-union stages separately, assuming the ideal
            // terminal boundary has already been supplied.  First transport and
            // union only the cached component parser DWAs; then union that result
            // with the minimized ideal boundary parser DWA.
            let oracle_specials = merged_special_token_terminals(
                &core,
                &children,
                &composed_table.terminal_offsets,
                &composed_table.table,
                &composed_table.control_terminals,
            );
            let oracle_original_token_ids = merged_original_token_ids(&vocab, &oracle_specials);
            let oracle_parser_components = vec![
                ParserDwaComponent {
                    constraint: &core,
                    parser_state_relation: &composed_table.state_relations[0],
                    tokenizer_state_offset: tokenizer_offsets[0],
                    terminal_offset: composed_table.terminal_offsets[0],
                    composed_table: Some(&composed_table.table),
                },
                ParserDwaComponent {
                    constraint: &dispatch,
                    parser_state_relation: &composed_table.state_relations[1],
                    tokenizer_state_offset: tokenizer_offsets[1],
                    terminal_offset: composed_table.terminal_offsets[1],
                    composed_table: Some(&composed_table.table),
                },
            ];
            let oracle_default_domains = build_parser_default_domain_plan(
                &oracle_parser_components,
                composed_table.table.num_states,
            );
            let component_union_started = Instant::now();
            let (component_artifacts, _component_top_accept) =
                compose_component_parser_dwas_and_possible_matches(
                    &oracle_parser_components,
                    &composed_table.terminal_offsets,
                    &oracle_default_domains.component_domains,
                    merged_tokenizer.num_states() as usize,
                    &oracle_original_token_ids,
                    !global_ignores,
                )
                .expect("compose component parser DWAs for ideal-boundary timing");
            let component_union_ms = component_union_started.elapsed().as_secs_f64() * 1000.0;
            let component_parser_states = component_artifacts.artifact().0.num_states();
            let component_parser_transitions = component_artifacts.artifact().0.num_transitions();

            let run_final_union = |kind: &str, boundary_dwa: DWA| {
                let boundary_states = boundary_dwa.num_states();
                let boundary_transitions = boundary_dwa.num_transitions();
                let started = Instant::now();
                let union = union_boundary_parser_dwa(
                    component_artifacts.clone(),
                    MappedArtifact::new(boundary_dwa, common_id_map.clone()),
                    composed_table.table.num_states,
                )
                .unwrap_or_else(|error| panic!("union {kind} ideal boundary parser: {error}"));
                let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
                eprintln!(
                    "MINBOUND OUTER_IDEAL_FINAL_UNION_VARIANT kind={} boundary_states={} boundary_transitions={} ms={:.3} final_states={} final_transitions={}",
                    kind,
                    boundary_states,
                    boundary_transitions,
                    elapsed_ms,
                    union.artifact().0.num_states(),
                    union.artifact().0.num_transitions(),
                );
                (elapsed_ms, union.artifact().0.num_states(), union.artifact().0.num_transitions())
            };
            if std::env::var_os("GLRMASK_MINBOUND_MINIMIZED_UNION_ONLY").is_none() {
                let _raw_union = run_final_union("raw", tight_parser_raw_for_union);
                let _hashcons_union = run_final_union("hashcons", tight_parser_hashconsed_for_union);
            }
            let (final_union_ms, final_parser_states, final_parser_transitions) =
                run_final_union("minimized", tight_parser_minimized.clone());
            eprintln!(
                "MINBOUND OUTER_IDEAL_UNION_TIMES component_union_ms={component_union_ms:.3} component_states={} component_transitions={} boundary_states={} boundary_transitions={} final_union_ms={final_union_ms:.3} final_states={} final_transitions={}",
                component_parser_states,
                component_parser_transitions,
                tight_parser_minimized.num_states(),
                tight_parser_minimized.num_transitions(),
                final_parser_states,
                final_parser_transitions,
            );

            let classes = exact_terminal_language_classes(&merged_tokenizer);
            let alias_count = classes.iter().enumerate().filter(|&(terminal, &class)| terminal as u32 != class).count();
            let b_quotient = quotient_terminal_language_labels("outer_B_language_quotient", &all[0], &classes);
            let a_quotient = quotient_terminal_language_labels("outer_A_language_quotient", &a, &classes);
            eprintln!("MINBOUND language_classes terminals={} aliases={} classes={}", classes.len(), alias_count, classes.iter().copied().collect::<BTreeSet<_>>().len());
            let reverse = exact_weighted_difference(&a_quotient, &b_quotient);
            let reverse_nonempty = reverse.states().iter().filter(|state| state.final_weight.as_ref().is_some_and(|weight| !weight.is_empty())).count();
            eprintln!(
                "MINBOUND OUTER_SUBSET_CHECK A_minus_B_states={} A_minus_B_transitions={} A_minus_B_nonempty_finals={}",
                reverse.num_states(), reverse.num_transitions(), reverse_nonempty,
            );
            let residual = exact_weighted_difference(&b_quotient, &a_quotient);
            acyclic_path_stats("outer_exact_B_minus_A", &residual);
            residual_diagnostics(
                "outer_exact_B_minus_A",
                &residual,
                &common_id_map,
                &terminal_names,
            );

            // Optional grammar-tightening pass after exact language quotient.
            // Replace each terminal occurrence in the composed CFG by its exact
            // lexer-language representative, so validity is existential over
            // aliases even when their LR actions differ.
            let mut quotient_rules = composed_table.table.rules.clone();
            for rule in &mut quotient_rules {
                for symbol in &mut rule.rhs {
                    if let Symbol::Terminal(terminal) = symbol
                        && let Some(&class) = classes.get(*terminal as usize)
                    {
                        *terminal = class;
                    }
                }
            }
            let quotient_augmented_start = quotient_rules
                .first()
                .expect("quotient grammar has augmented start")
                .lhs;
            let quotient_grammar = AnalyzedGrammar::from_composed_rules(
                quotient_rules,
                composed_table.table.num_terminals,
                terminal_names.clone(),
                composed_table.table.nonterminal_display_names.clone(),
                quotient_augmented_start,
            );
            let quotient_controls = composed_table
                .control_terminals
                .iter()
                .map(|terminal| classes.get(*terminal as usize).copied().unwrap_or(*terminal))
                .collect::<BTreeSet<_>>();
            let candidate_trigrams = mb_candidate_trigrams(&residual);
            let valid_trigrams = mb_valid_candidate_trigrams(
                &quotient_grammar,
                &quotient_controls,
                &candidate_trigrams,
            );
            let grammar_tight = mb_filter_valid_trigrams(&residual, &valid_trigrams);
            acyclic_path_stats("outer_exact_B_minus_A_grammar_tight", &grammar_tight);
            let grammar_tight_labels = grammar_tight
                .states()
                .iter()
                .flat_map(|state| state.transitions.keys().copied())
                .filter(|&label| label >= 0)
                .collect::<BTreeSet<_>>();
            eprintln!(
                "MINBOUND OUTER_GRAMMAR_TIGHT states={} transitions={} labels={} candidate_trigrams={} valid_trigrams={}",
                grammar_tight.num_states(), grammar_tight.num_transitions(), grammar_tight_labels.len(), candidate_trigrams.len(), valid_trigrams.len(),
            );
            let labels = residual
                .states()
                .iter()
                .flat_map(|state| state.transitions.keys().copied())
                .filter(|&label| label >= 0)
                .collect::<BTreeSet<_>>();
            eprintln!(
                "MINBOUND OUTER_EXACT_RESULT A_states={} A_transitions={} B_states={} B_transitions={} residual_states={} residual_transitions={} residual_labels={}",
                a_quotient.num_states(),
                a_quotient.num_transitions(),
                b_quotient.num_states(),
                b_quotient.num_transitions(),
                residual.num_states(),
                residual.num_transitions(),
                labels.len(),
            );
            let augmented_start = composed_table
                .table
                .rules
                .first()
                .expect("outer composed table has augmented start")
                .lhs;
            let grammar = AnalyzedGrammar::from_composed_rules(
                composed_table.table.rules.clone(),
                composed_table.table.num_terminals,
                terminal_names.clone(),
                composed_table.table.nonterminal_display_names.clone(),
                augmented_start,
            );
            let mut selected = vec![false; composed_table.table.num_terminals as usize];
            for state in residual.states() {
                for &label in state.transitions.keys() {
                    if label >= 0 && (label as usize) < selected.len() {
                        selected[label as usize] = true;
                    }
                }
            }
            let active = selected.iter().filter(|&&yes| yes).count();
            let template_started = Instant::now();
            let (templates, _, _) = build_composition_templates(
                &composed_table.table,
                &grammar,
                &selected,
            );
            let template_ms = template_started.elapsed().as_secs_f64() * 1000.0;
            let parser_started = Instant::now();
            let parser = build_parser_dwa_from_terminal_dwa_with_precomputed_templates(
                &composed_table.table,
                &grammar,
                &TerminalAutomaton::Dwa(residual.clone()),
                &templates,
                &vocab,
                &common_id_map,
                false,
            );
            let parser_ms = parser_started.elapsed().as_secs_f64() * 1000.0;
            eprintln!(
                "MINBOUND OUTER_EXACT_PARSER active_terminals={} template_ms={:.3} parser_ms={:.3} parser_states={} parser_transitions={}",
                active,
                template_ms,
                parser_ms,
                parser.num_states(),
                parser.num_transitions(),
            );
            let depth2 = filter_residual_min_terminals(&residual, 2);
            let crossed = filter_residual_cross_component(&residual, &terminal_names);
            eprintln!(
                "MINBOUND OUTER_EXACT_DECOMP depth_ge_2_states={} depth_ge_2_transitions={} crossed_states={} crossed_transitions={}",
                depth2.num_states(),
                depth2.num_transitions(),
                crossed.num_states(),
                crossed.num_transitions(),
            );
            let capture = MappedArtifact::new(residual, common_id_map);
            save_terminal_capture(
                std::path::Path::new(&root)
                    .join("tdwa_outer_exact_B_minus_A.cap")
                    .to_str()
                    .unwrap(),
                &capture,
                &terminal_names,
            );
            return;
        }

        let root = std::env::var("GLRMASK_MINBOUND_DIR").expect("GLRMASK_MINBOUND_DIR");
        let vocab_path = std::env::var("GLRMASK_MINBOUND_VOCAB").expect("GLRMASK_MINBOUND_VOCAB");
        let vocab = load_vocab(&vocab_path);
        let load = |name: &str| -> Constraint {
            Constraint::load(&fs::read(format!("{root}\\{name}")).unwrap()).unwrap()
        };
        // All terminal DWAs are now captured. The subtraction phase itself does
        // not invoke the terminal compiler at all.
        let (monolithic_artifact, full_terminal_names) =
            load_terminal_capture(&format!("{root}\\tdwa_monolithic.cap"));
        acyclic_path_stats("monolithic", monolithic_artifact.artifact());
        let monolithic_state_count = monolithic_artifact.id_map().tokenizer_states.original_to_internal.len();
        eprintln!(
            "MINBOUND capture monolithic terminals={} states={} transitions={} raw_tokenizer_states={}",
            full_terminal_names.len(),
            monolithic_artifact.artifact().num_states(),
            monolithic_artifact.artifact().num_transitions(),
            monolithic_state_count,
        );

        // Read all primitive state counts first. The owned-parent composition layout is
        // cumulative, so this completely determines the exact embedding without loading
        // any Constraint or reconstructing a tokenizer.
        let (core_raw, core_names) = load_terminal_capture(&format!("{root}\\tdwa_core.cap"));
        let core_state_count = core_raw.id_map().tokenizer_states.original_to_internal.len();
        let (dispatch_raw, dispatch_names) = load_terminal_capture(&format!("{root}\\tdwa_dispatch.cap"));
        let dispatch_state_count = dispatch_raw.id_map().tokenizer_states.original_to_internal.len();
        let mut schema_raw = Vec::new();
        let mut schema_state_counts = Vec::new();
        for index in 0..10 {
            let pair = load_terminal_capture(&format!("{root}\\tdwa_schema_{index}.cap"));
            schema_state_counts.push(pair.0.id_map().tokenizer_states.original_to_internal.len());
            schema_raw.push(pair);
        }
        let schema_states_sum = schema_state_counts.iter().sum::<usize>();
        assert_eq!(
            core_state_count + 1 + dispatch_state_count + schema_states_sum,
            monolithic_state_count,
            "primitive tokenizer states plus the dispatcher's fresh reset must exactly cover the monolithic tokenizer",
        );
        // Outer owned-parent composition keeps the core at offset zero. The nested
        // dispatcher composition contributes one fresh dispatcher-reset state followed
        // by its primitive parent tokenizer and then schema tokenizers in order.
        let dispatch_reset = core_state_count as u32;
        let dispatch_parent_offset = dispatch_reset + 1;
        eprintln!(
            "MINBOUND tokenizer_layout core={} dispatch_reset={} dispatch_parent={} schemas={:?} monolithic={}",
            core_state_count, dispatch_reset, dispatch_state_count, schema_state_counts, monolithic_state_count,
        );

        let mut artifacts = Vec::<MappedArtifact<DWA>>::new();
        artifacts.push(monolithic_artifact);

        let core = remap_terminal_labels("core", core_raw, &core_names, &full_terminal_names, "");
        artifacts.push(rebase_tokenizer_state_universe(
            "core", core, 0, &[0], monolithic_state_count,
        ));

        // `tdwa_dispatch.cap` is the primitive dispatcher-parent compile, captured
        // before its subgrammars are composed. It therefore begins immediately after
        // the nested dispatcher's fresh reset.
        let dispatch = remap_terminal_labels(
            "dispatch_parent", dispatch_raw, &dispatch_names, &full_terminal_names, "subgrammar0::",
        );
        artifacts.push(rebase_tokenizer_state_universe(
            "dispatch_parent",
            dispatch,
            dispatch_parent_offset,
            &[0, dispatch_reset],
            monolithic_state_count,
        ));

        let mut schema_offset = dispatch_parent_offset + dispatch_state_count as u32;
        for (index, ((artifact, names), &state_count)) in schema_raw
            .into_iter()
            .zip(schema_state_counts.iter())
            .enumerate()
        {
            let artifact = remap_terminal_labels(
                &format!("schema_{index}"),
                artifact,
                &names,
                &full_terminal_names,
                &format!("subgrammar0::subgrammar{index}::"),
            );
            artifacts.push(rebase_tokenizer_state_universe(
                &format!("schema_{index}"),
                artifact,
                schema_offset,
                &[0, dispatch_reset],
                monolithic_state_count,
            ));
            schema_offset += state_count as u32;
        }
        assert_eq!(schema_offset as usize, monolithic_state_count);

        let reconcile_started = Instant::now();
        let reconciled = MappedArtifact::reconcile_vec(artifacts);
        let (all, common_id_map) = reconciled.into_parts();
        eprintln!(
            "MINBOUND reconcile artifacts={} common_tsids={} common_tokens={} ms={:.3}",
            all.len(), common_id_map.num_tsids(), common_id_map.num_internal_tokens(),
            reconcile_started.elapsed().as_secs_f64() * 1000.0,
        );
        assert_eq!(all.len(), 13);
        let monolithic = &all[0];
        let build_parser = std::env::var_os("GLRMASK_MINBOUND_BUILD_PARSER").is_some();
        let all_components_residual = {
            let monolithic = &all[0];
            let run_variant = |name: &str, indices: &[usize]| -> DWA {
                eprintln!("MINBOUND variant={name} indices={indices:?}");
                let components = indices.iter().map(|&index| all[index].clone()).collect::<Vec<_>>();
                let union = union_dwas(&components, &common_id_map);
                let residual = exact_weighted_difference(monolithic, &union);
                let nonempty_final_states = residual
                    .states()
                    .iter()
                    .filter(|state| state.final_weight.as_ref().is_some_and(|weight| !weight.is_empty()))
                    .count();
                let labels = residual
                    .states()
                    .iter()
                    .flat_map(|state| state.transitions.keys().copied())
                    .filter(|&label| label >= 0)
                    .collect::<BTreeSet<_>>();
                eprintln!(
                    "MINBOUND variant_result name={name} component_union_states={} component_union_transitions={} residual_states={} residual_transitions={} residual_final_states={} residual_labels={}",
                    union.num_states(), union.num_transitions(), residual.num_states(), residual.num_transitions(), nonempty_final_states, labels.len(),
                );
                if std::env::var_os("GLRMASK_MINBOUND_VERBOSE_DIAG").is_some() { residual_diagnostics(name, &residual, &common_id_map, &full_terminal_names); }
                residual
            };

            // Partial controls are useful diagnostics; the last variant is the exact
            // primitive-component subtraction requested by the experiment.
            let _ = run_variant("core_plus_dispatch_parent", &[1, 2]);
            let _ = run_variant("core_plus_individual_schemas", &[1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
            run_variant("all_primitive_components", &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        };

        drop(all);
        let full_for_factor = load("full.bin");
        let full_grammar = analyzed(&full_for_factor);
        let trigram_candidates = mb_candidate_trigrams(&all_components_residual);
        let valid_trigrams = mb_valid_candidate_trigrams(
            &full_grammar,
            &full_for_factor.table.control_terminals,
            &trigram_candidates,
        );
        let trigram_filtered = mb_filter_valid_trigrams(&all_components_residual, &valid_trigrams);
        acyclic_path_stats("trigram_filtered", &trigram_filtered);
        eprintln!(
            "MINBOUND substring_result name=valid_trigrams states={} transitions={} final_states={} labels={}",
            trigram_filtered.num_states(), trigram_filtered.num_transitions(),
            trigram_filtered.states().iter().filter(|state| state.final_weight.as_ref().is_some_and(|weight| !weight.is_empty())).count(),
            trigram_filtered.states().iter().flat_map(|state| state.transitions.keys().copied()).filter(|&label| label >= 0).collect::<BTreeSet<_>>().len(),
        );

        let depth_ge_2 = filter_residual_min_terminals(&all_components_residual, 2);
        let cross_component_only = filter_residual_cross_component(&all_components_residual, &full_terminal_names);
        for (name, dwa) in [("depth_ge_2_substring_proxy", &depth_ge_2), ("cross_component_only", &cross_component_only)] {
            let labels = dwa.states().iter().flat_map(|state| state.transitions.keys().copied()).filter(|&label| label >= 0).collect::<BTreeSet<_>>();
            let finals = dwa.states().iter().filter(|state| state.final_weight.as_ref().is_some_and(|weight| !weight.is_empty())).count();
            eprintln!("MINBOUND ideal_result name={name} states={} transitions={} final_states={} labels={}", dwa.num_states(), dwa.num_transitions(), finals, labels.len());
        }

        if build_parser {
            drop(trigram_candidates);
            drop(valid_trigrams);
            drop(depth_ge_2);
            drop(cross_component_only);
            drop(all_components_residual);
            parser_from_residual("valid_trigrams", &trigram_filtered, &full_for_factor, &common_id_map, &vocab);
        }
    }

    // ---- Phase 1 restricted-walk probe (prepared-static-linker redesign) ----
    // Runs the STANDARD terminal-DWA construction (unified L2P trie walk) on a
    // merged tokenizer with narrowed inputs: start states = component i's
    // commit states, vocab = full or T_i-restricted. Measurement only; the old
    // boundary path is untouched. Report: /tmp/grammars25-redesign/E-phase1-walk.md

    // Phase 1 probe setup helpers (shared by both probe tests).
    fn phase1_load(root: &std::path::Path, name: &str) -> Constraint {
        let bytes = std::fs::read(root.join(name)).unwrap();
        Constraint::load(&bytes).unwrap()
    }

    // Mirror the MINBOUND deferred-data restoration (exprs + rules + metadata).
    fn phase1_restore(component: &mut Constraint, label: &str) {
        if component.tokenizer.terminal_exprs().is_none()
            && let Some(exprs) = component.retained_terminal_exprs().map(|exprs| exprs.to_vec())
        {
            Arc::make_mut(&mut component.tokenizer).restore_terminal_exprs(Some(exprs))
                .expect("restore component terminal exprs");
        }
        let inline_rules = component.table.rules.len();
        let retained = component.retained_table_rules().expect("decode retained rules").len();
        if inline_rules != retained {
            component.table.rules =
                component.retained_table_rules().expect("decode retained rules").to_vec();
        }
        component
            .materialize_composition_metadata_for_compilation()
            .expect("materialize composition metadata");
        eprintln!(
            "PHASE1 restore {label} inline_rules={inline_rules} retained_rules={retained} exprs={}",
            component.tokenizer.terminal_exprs().is_some(),
        );
    }

    struct Phase1Composed {
        table: ComposedTable,
        tokenizer: Tokenizer,
        tokenizer_offsets: Vec<u32>,
        terminal_names: Vec<String>,
        ignore_canonical: Option<u32>,
        global_ignores: bool,
        scoped_ignores: BitSet,
    }

    // Existing composition code only: table splice + control elimination +
    // disjoint-union tokenizer. No boundary discovery runs here.
    fn phase1_compose_low_level(
        tag: &str,
        parent: &Constraint,
        children: &[CompiledSubgrammarInput<'_>],
    ) -> Phase1Composed {
        let global_ignores = component_ignores_are_globally_erasable(parent, children);
        let table_inputs: Vec<SubgrammarTableInput> = children
            .iter()
            .map(|child| SubgrammarTableInput {
                placeholder_terminal: child.placeholder_terminal,
                additional_placeholder_terminals: &[],
                table: &child.constraint.table,
                ignore_terminal: (!global_ignores)
                    .then_some(child.constraint.ignore_terminal)
                    .flatten(),
                start_nullable: child.constraint.table.embedded_start_nullable(),
            })
            .collect();
        let started = Instant::now();
        let mut composed = compose_subgrammar_tables(
            &parent.table,
            (!global_ignores).then_some(parent.ignore_terminal).flatten(),
            &table_inputs,
        )
        .expect("compose tables");
        eliminate_composed_runtime_controls(&mut composed).expect("eliminate controls");
        let table_ms = started.elapsed().as_secs_f64() * 1000.0;
        let terminal_names = merged_terminal_display_names(parent, children);
        let started = Instant::now();
        let mut tokenizer_inputs: Vec<(&Tokenizer, u32)> =
            Vec::with_capacity(children.len() + 1);
        tokenizer_inputs.push((&parent.tokenizer, composed.terminal_offsets[0]));
        for (index, child) in children.iter().enumerate() {
            tokenizer_inputs
                .push((&child.constraint.tokenizer, composed.terminal_offsets[index + 1]));
        }
        let (mut merged, tokenizer_offsets) =
            Tokenizer::disjoint_union_with_terminal_offsets(&tokenizer_inputs);
        if merged.terminal_exprs().is_none() {
            let all: Vec<&Constraint> = std::iter::once(parent)
                .chain(children.iter().map(|child| child.constraint))
                .collect();
            if let Some(exprs) = merged_retained_terminal_exprs(
                &all,
                &composed.terminal_offsets,
                composed.table.num_terminals,
            ) {
                merged.restore_terminal_exprs(Some(exprs)).expect("restore merged exprs");
            }
        }
        let tokenizer_ms = started.elapsed().as_secs_f64() * 1000.0;
        let ignores = merged_ignore_terminals(
            parent,
            children,
            &composed.terminal_offsets,
            global_ignores,
        );
        assert_eq!(
            tokenizer_offsets[0], 1,
            "merged state 0 must be the fresh reset fan-out"
        );
        eprintln!(
            "PHASE1 compose_{tag} global_ignores={global_ignores} lr={} terms={} lexstates={} table_ms={table_ms:.3} tokenizer_ms={tokenizer_ms:.3} tok_offsets={tokenizer_offsets:?} term_offsets={:?} reset_roots={:?} reset_closure_0={:?}",
            composed.table.num_states,
            composed.table.num_terminals,
            merged.num_states(),
            composed.terminal_offsets,
            merged.deterministic_dispatch_roots().map(|roots| roots.to_vec()),
            merged.execute_from_state_end_only(&[], 0).into_vec(),
        );
        Phase1Composed {
            table: composed,
            tokenizer: merged,
            tokenizer_offsets,
            terminal_names,
            ignore_canonical: ignores.canonical,
            global_ignores,
            scoped_ignores: ignores.scoped,
        }
    }

    fn phase1_grammar(
        table: &crate::compiler::glr::table::GLRTable,
        names: &[String],
    ) -> AnalyzedGrammar {
        let augmented_start =
            table.rules.first().expect("composed table has augmented start").lhs;
        AnalyzedGrammar::from_composed_rules(
            table.rules.clone(),
            table.num_terminals,
            names.to_vec(),
            table.nonterminal_display_names.clone(),
            augmented_start,
        )
    }

    const PHASE1_SELECTED10_SHORT: [&str; 10] = [
        "o31994",
        "kb_620_Normalized",
        "sil-kit-participant-configuration",
        "o9792",
        "o9896",
        "o16060",
        "kb_678_Normalized",
        "o83390",
        "taurus",
        "kb_1104_Normalized",
    ];

    #[test]
    #[ignore]
    fn phase1_restricted_walk_selected10() {
        use std::fs;
        use std::path::Path;






        struct Phase1Oracle {
            b_mapped: MappedArtifact<DWA>,
            residual: DWA,
            id_map: InternalIdMap,
            tokens_all: BTreeSet<u32>,
        }

        // MINBOUND B-A oracle recomputed through the existing families pipeline.
        fn phase1_oracle(
            tag: &str,
            composed: &Phase1Composed,
            components: &[(&Tokenizer, &crate::compiler::glr::table::GLRTable, &[String], Option<u32>)],
            vocab: &Vocab,
        ) -> Phase1Oracle {
            let label = format!("phase1_B_{tag}");
            let b = build_terminal_dwa_parts(
                &label,
                &composed.tokenizer,
                &composed.table.table,
                &composed.terminal_names,
                composed.ignore_canonical,
                vocab,
            );
            let mut parts = vec![b];
            for (index, (tokenizer, table, names, ignore)) in components.iter().enumerate() {
                let label = format!("phase1_A_{tag}_{index}");
                let ignore = composed.global_ignores.then_some(*ignore).flatten();
                let a = build_terminal_dwa_parts(label.as_str(), tokenizer, table, names, ignore, vocab);
                let a = offset_terminal_labels(
                    label.as_str(),
                    a,
                    composed.table.terminal_offsets[index],
                );
                let a = rebase_tokenizer_state_universe(
                    label.as_str(),
                    a,
                    composed.tokenizer_offsets[index],
                    &[0],
                    composed.tokenizer.num_states() as usize,
                );
                parts.push(a);
            }
            let started = Instant::now();
            let reconciled = MappedArtifact::reconcile_vec(parts);
            let (all, common) = reconciled.into_parts();
            let a = union_dwas(&all[1..], &common);
            let residual = exact_weighted_difference(&all[0], &a);
            let diff_ms = started.elapsed().as_secs_f64() * 1000.0;
            let crossed =
                filter_residual_cross_outer_component(&residual, &composed.table.terminal_offsets);
            let tokens_all = accepted_original_tokens(&residual, &common);
            eprintln!(
                "PHASE1 oracle_{tag} b_states={} b_trans={} residual_states={} residual_trans={} crossed_states={} crossed_trans={} tokens={} reconcile_union_diff_ms={diff_ms:.3}",
                all[0].num_states(),
                all[0].num_transitions(),
                residual.num_states(),
                residual.num_transitions(),
                crossed.num_states(),
                crossed.num_transitions(),
                tokens_all.len(),
            );
            let b_mapped = MappedArtifact::new(all[0].clone(), common.clone());
            Phase1Oracle { b_mapped, residual, id_map: common, tokens_all }
        }

        // T_i = tokens for which some commit state of L_i completes a terminal
        // strictly before the end of the token (byte-level, parser-agnostic).
        fn phase1_compute_ti(
            tag: &str,
            tokenizer: &Tokenizer,
            vocab: &Vocab,
        ) -> BTreeSet<u32> {
            let started = Instant::now();
            let starts: Vec<u32> = (0..tokenizer.num_states()).collect();
            let ti: BTreeSet<u32> = vocab
                .entries_map()
                .par_iter()
                .filter_map(|(&token, bytes)| {
                    if bytes.is_empty() {
                        return None;
                    }
                    let groups = tokenizer.execute_summary_groups_from_states(bytes, &starts);
                    let resets_inside = groups.iter().any(|(_, matches, _)| {
                        matches.iter().any(|&(_, width)| width < bytes.len())
                    });
                    resets_inside.then_some(token)
                })
                .collect();
            eprintln!(
                "PHASE1 ti_{tag} tokens={} of={} states={} ms={:.3}",
                ti.len(),
                vocab.entries_map().len(),
                starts.len(),
                started.elapsed().as_secs_f64() * 1000.0,
            );
            ti
        }

        // Restricted-equivalence state map (§2.4.3 vehicle): singleton classes
        // over exactly the kept raw states; the analysis then partitions only
        // this universe (token side still comes from the passed vocab).
        fn phase1_restricted_state_map(num_states: usize, keep: &[bool]) -> ManyToOneIdMap {
            let mut original_to_internal = vec![u32::MAX; num_states];
            let mut representatives = Vec::new();
            for (raw, &selected) in keep.iter().enumerate() {
                if selected {
                    original_to_internal[raw] = representatives.len() as u32;
                    representatives.push(raw as u32);
                }
            }
            ManyToOneIdMap::from_singleton_original_to_internal_with_representatives(
                original_to_internal,
                representatives,
            )
        }

        // The standard construction: unified L2P trie walk on the merged
        // tokenizer, all terminals active, optional start-state subset.
        fn phase1_run_walk(
            tag: &str,
            tokenizer: &Tokenizer,
            vocab: &Vocab,
            grammar: &AnalyzedGrammar,
            disallowed: &BTreeMap<u32, BitSet>,
            ignore_terminal: Option<u32>,
            seed_filter: Option<&[bool]>,
            initial_state_map: Option<&ManyToOneIdMap>,
            nwa_crossing: Option<(&[u32], usize)>,
        ) -> Option<
            crate::compiler::stages::id_map_and_terminal_dwa::types::LocalIdMapTerminalDwa,
        > {
            use crate::compiler::stages::id_map_and_terminal_dwa as tdwa;
            if vocab.entries_map().is_empty() {
                eprintln!("PHASE1 walk_{tag} SKIPPED_empty_vocab");
                return None;
            }
            let num_terms = grammar.num_terminals as usize;
            let coloring = TerminalColoring::identity(num_terms);
            let always_allowed = tdwa::grammar_helpers::compute_always_allowed_follows(grammar);
            let active = vec![true; num_terms];
            let flat: Arc<[u32]> = Arc::from(tdwa::l1::build_flat_transition_table(tokenizer));
            let seeded = seed_filter.map_or(tokenizer.num_states() as usize, |keep| {
                keep.iter().filter(|&&selected| selected).count()
            });
            let started = Instant::now();
            // Phase 2 step 1 verification knob: route Step 1 through the
            // shared-equivalence path (computed fresh here for exactness
            // comparison; production code computes it once per link).
            let use_shared =
                std::env::var("PHASE1_SHARED").map(|value| value != "0").unwrap_or(false);
            let shared = use_shared.then(|| {
                tdwa::l2p::compute_shared_l2p_equivalence(
                    "phase1_probe",
                    tokenizer,
                    vocab,
                    ignore_terminal,
                    grammar,
                    &active,
                    disallowed,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    Some(&flat),
                    None,
                    initial_state_map,
                )
                .expect("shared equivalence must compute")
            });
            // Phase 2 step 2 verification knob: NWA-level crossing filter
            // inside the entry point (before determinize/minimize).
            let nwa_filter_enabled = std::env::var("PHASE1_NWAFILT")
                .map(|value| value != "0")
                .unwrap_or(false);
            let nwa_ownership = if nwa_filter_enabled {
                nwa_crossing.map(|(terminal_offsets, _)| {
                    tdwa::scope::BoundaryOwnership::flat(
                        terminal_offsets,
                        grammar.num_terminals,
                    )
                    .expect("phase1 crossing ownership")
                })
            } else {
                None
            };
            let nwa_filter = match (nwa_crossing, nwa_ownership.as_ref()) {
                (Some((_, start_component)), Some(ownership)) => {
                    Some(tdwa::l2p::L2pCrossingFilter {
                        ownership,
                        start_component: tdwa::scope::ImmediateComponentId(
                            start_component as u32,
                        ),
                    })
                }
                _ => None,
            };
            let shard_options = if shared.is_some() || nwa_filter.is_some() {
                Some(tdwa::l2p::L2pShardBuildOptions {
                    shared_equivalence: shared.as_ref(),
                    skip_ti_discovery: shared.is_some(),
                    ti_candidate_groups: None,
                    crossing_filter: nwa_filter,
                    skip_core_compact: false,
                    follow_transparent: None,
                    initial_state_domain_is_exact: false,
                })
            } else {
                None
            };
            let result = tdwa::l2p::build_l2p_id_map_and_terminal_dwa_mode(
                "phase1_probe",
                tokenizer,
                vocab,
                &coloring,
                false,
                ignore_terminal,
                grammar,
                &always_allowed,
                &active,
                disallowed,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                Some(&flat),
                None,
                initial_state_map,
                false,
                seed_filter,
                shard_options.as_ref(),
            )
            .expect("restricted walk must produce a DWA");
            eprintln!(
                "PHASE1 walk_{tag} vocab={} seeded={} shared={} nwafilt={} wall_ms={:.3} id_map_ms={:.3} shared_id_map_ms={:.3} dwa_ms={:.3} compact_ms={:.3} dwa_states={} dwa_trans={} tsids={} itokens={} acyclic={}",
                vocab.entries_map().len(),
                seeded,
                shared.is_some(),
                nwa_filter.is_some(),
                started.elapsed().as_secs_f64() * 1000.0,
                result.profile.id_map_ms,
                shared.as_ref().map_or(0.0, |shared| shared.id_map_ms),
                result.profile.terminal_dwa_ms,
                result.profile.compact_ms,
                result.dwa.num_states(),
                result.dwa.num_transitions(),
                result.id_map.num_tsids(),
                result.id_map.num_internal_tokens(),
                result.dwa.is_acyclic(),
            );
            Some(result)
        }

        // Raw seen-flag product: keep only paths that emit a terminal
        // owned by a component != i. Shared by the DWA-level filter and the
        // NWA-vs-DWA equivalence check (which minimizes with `minimize_owned`
        // instead of the acyclic minimizer).
        fn phase1_crossing_product(
            dwa: &DWA,
            terminal_offsets: &[u32],
            start_component: usize,
        ) -> DWA {
            let mut states = vec![DWAState::default()];
            let mut ids = FxHashMap::<(u32, bool), u32>::default();
            let mut payloads = vec![(dwa.start_state(), false)];
            ids.insert((dwa.start_state(), false), 0);
            let mut queue = VecDeque::from([0u32]);
            while let Some(out) = queue.pop_front() {
                let (source, seen) = payloads[out as usize];
                let source_state = &dwa.states()[source as usize];
                if seen {
                    states[out as usize].final_weight = source_state.final_weight.clone();
                }
                for (&label, (target, weight)) in &source_state.transitions {
                    assert!(
                        label >= 0 && label != DEFAULT_LABEL,
                        "crossing filter: non-terminal label {label}",
                    );
                    let component = terminal_offsets
                        .partition_point(|&offset| offset <= label as u32)
                        .saturating_sub(1);
                    let next_seen = seen || component != start_component;
                    let key = (*target, next_seen);
                    let next = if let Some(&id) = ids.get(&key) {
                        id
                    } else {
                        let id = states.len() as u32;
                        ids.insert(key, id);
                        states.push(DWAState::default());
                        payloads.push((*target, next_seen));
                        queue.push_back(id);
                        id
                    };
                    states[out as usize].transitions.insert(label, (next, weight.clone()));
                }
            }
            DWA::from_parts(states, 0)
        }

        // Structural id-map equality (Step-1 coordinates must be identical
        // across walks for same-coordinate DWA comparison).
        fn phase1_assert_same_id_map(tag: &str, a: &InternalIdMap, b: &InternalIdMap) {
            assert_eq!(
                a.tokenizer_states.original_to_internal,
                b.tokenizer_states.original_to_internal,
                "{tag}: tokenizer_states.original_to_internal differs",
            );
            assert_eq!(
                a.tokenizer_states.internal_to_originals,
                b.tokenizer_states.internal_to_originals,
                "{tag}: tokenizer_states.internal_to_originals differs",
            );
            assert_eq!(
                a.tokenizer_states.representative_original_ids,
                b.tokenizer_states.representative_original_ids,
                "{tag}: tokenizer_states.representative_original_ids differs",
            );
            assert_eq!(
                a.vocab_tokens.original_to_internal, b.vocab_tokens.original_to_internal,
                "{tag}: vocab_tokens.original_to_internal differs",
            );
            assert_eq!(
                a.vocab_tokens.internal_to_originals, b.vocab_tokens.internal_to_originals,
                "{tag}: vocab_tokens.internal_to_originals differs",
            );
            assert_eq!(
                a.vocab_tokens.representative_original_ids,
                b.vocab_tokens.representative_original_ids,
                "{tag}: vocab_tokens.representative_original_ids differs",
            );
            assert_eq!(
                a.deferred_vocab_singleton_original_ids,
                b.deferred_vocab_singleton_original_ids,
                "{tag}: deferred_vocab_singleton_original_ids differs",
            );
        }

        // Keep only paths that emit a terminal owned by a component != i.
        fn phase1_crossing_from(
            tag: &str,
            dwa: &DWA,
            terminal_offsets: &[u32],
            start_component: usize,
        ) -> DWA {
            let started = Instant::now();
            let raw = phase1_crossing_product(dwa, terminal_offsets, start_component);
            let raw_states = raw.num_states();
            let raw_trans = raw.num_transitions();
            let acyclic = raw.is_acyclic();
            let minimized = if acyclic {
                crate::automata::weighted_u32::minimize_acyclic::minimize_acyclic_owned(raw)
            } else {
                minimize_owned(raw)
            };
            eprintln!(
                "PHASE1 crossing_{tag} raw_states={raw_states} raw_trans={raw_trans} states={} trans={} acyclic={acyclic} ms={:.3}",
                minimized.num_states(),
                minimized.num_transitions(),
                started.elapsed().as_secs_f64() * 1000.0,
            );
            minimized
        }

        fn phase1_commit_range(
            tokenizer_offsets: &[u32],
            num_states: u32,
            index: usize,
            total: usize,
        ) -> Vec<bool> {
            let mut keep = vec![false; total];
            let start = tokenizer_offsets[index] as usize;
            for raw in start..start + num_states as usize {
                keep[raw] = true;
            }
            keep
        }

        fn phase1_commit_internal(
            id_map: &InternalIdMap,
            commit_raw: &[bool],
        ) -> Vec<bool> {
            id_map
                .tokenizer_states
                .internal_to_originals
                .iter()
                .map(|members| {
                    members.iter().any(|raw| commit_raw.get(*raw as usize).copied().unwrap_or(false))
                })
                .collect()
        }

        // Accepted original tokens, optionally restricted to a TSID subset.
        // Fixpoint propagation (no acyclicity assumption); matches
        // accepted_original_tokens on acyclic inputs.
        fn phase1_accepted_tokens(
            dwa: &DWA,
            id_map: &InternalIdMap,
            keep_internal_tsid: Option<&[bool]>,
        ) -> BTreeSet<u32> {
            let n = dwa.num_states() as usize;
            let mut ops = ScopedWeightOpCache::default();
            let initial = match keep_internal_tsid {
                None => Weight::all(),
                Some(keep) => {
                    let all_tokens =
                        RangeSetBlaze::from_iter([0..=id_map.max_internal_token_id()]);
                    let mut initial = Weight::empty();
                    for (tsid, selected) in keep.iter().enumerate() {
                        if *selected {
                            let one = Weight::from_token_set_for_tsid(tsid as u32, all_tokens.clone());
                            initial = ops.union(&initial, &one);
                        }
                    }
                    initial
                }
            };
            let mut reach = vec![Weight::empty(); n];
            reach[dwa.start_state() as usize] = initial;
            let mut queue = VecDeque::from([dwa.start_state()]);
            let mut queued = vec![false; n];
            queued[dwa.start_state() as usize] = true;
            let mut relaxations = 0usize;
            while let Some(source) = queue.pop_front() {
                queued[source as usize] = false;
                let source_support = reach[source as usize].clone();
                if source_support.is_empty() {
                    continue;
                }
                for &(target, ref edge_weight) in
                    dwa.states()[source as usize].transitions.values()
                {
                    let support = ops.intersection(&source_support, edge_weight);
                    if support.is_empty() {
                        continue;
                    }
                    relaxations += 1;
                    if relaxations > 20_000_000 {
                        panic!("phase1 token propagation did not converge");
                    }
                    let grown = ops.union(&reach[target as usize], &support);
                    if grown != reach[target as usize] {
                        reach[target as usize] = grown;
                        if !queued[target as usize] {
                            queued[target as usize] = true;
                            queue.push_back(target);
                        }
                    }
                }
            }
            let mut accepted = Weight::empty();
            for (index, state) in dwa.states().iter().enumerate() {
                if reach[index].is_empty() {
                    continue;
                }
                if let Some(final_weight) = state.final_weight.as_ref() {
                    let support = ops.intersection(&reach[index], final_weight);
                    accepted = ops.union(&accepted, &support);
                }
            }
            let mut originals = BTreeSet::new();
            for (_, internal_tokens) in accepted.raw_range_values() {
                for range in internal_tokens.ranges() {
                    for internal_token in range {
                        if let Some(ids) = id_map
                            .vocab_tokens
                            .internal_to_originals
                            .get(internal_token as usize)
                        {
                            originals.extend(ids.iter().copied());
                        }
                    }
                }
            }
            originals
        }

        fn phase1_topo(dwa: &DWA) -> Option<Vec<u32>> {
            let n = dwa.num_states() as usize;
            let mut indegree = vec![0usize; n];
            for state in dwa.states() {
                for &(target, _) in state.transitions.values() {
                    indegree[target as usize] += 1;
                }
            }
            let mut queue = VecDeque::new();
            for (state, &degree) in indegree.iter().enumerate() {
                if degree == 0 {
                    queue.push_back(state as u32);
                }
            }
            let mut topo = Vec::with_capacity(n);
            while let Some(source) = queue.pop_front() {
                topo.push(source);
                for &(target, _) in dwa.states()[source as usize].transitions.values() {
                    indegree[target as usize] -= 1;
                    if indegree[target as usize] == 0 {
                        queue.push_back(target);
                    }
                }
            }
            (topo.len() == n).then_some(topo)
        }

        // Distinct accepting label-paths (saturating); "cyclic" if not a DAG.
        fn phase1_count_paths(dwa: &DWA) -> String {
            let Some(topo) = phase1_topo(dwa) else {
                return "cyclic".to_string();
            };
            let n = dwa.num_states() as usize;
            let mut dp = vec![0u128; n];
            dp[dwa.start_state() as usize] = 1;
            for source in topo {
                for &(target, _) in dwa.states()[source as usize].transitions.values() {
                    dp[target as usize] = dp[target as usize].saturating_add(dp[source as usize]);
                }
            }
            let mut total = 0u128;
            for (index, state) in dwa.states().iter().enumerate() {
                if state.final_weight.as_ref().is_some_and(|weight| !weight.is_empty()) {
                    total = total.saturating_add(dp[index]);
                }
            }
            total.to_string()
        }

        fn phase1_token_diff_report(
            tag: &str,
            crossing: &BTreeSet<u32>,
            oracle: &BTreeSet<u32>,
            vocab: &Vocab,
        ) {
            let only_cross: Vec<u32> = crossing.difference(oracle).copied().collect();
            let only_oracle: Vec<u32> = oracle.difference(crossing).copied().collect();
            eprintln!(
                "PHASE1 gate_{tag} crossing={} oracle={} both={} crossing_only={} oracle_only={}",
                crossing.len(),
                oracle.len(),
                crossing.intersection(oracle).count(),
                only_cross.len(),
                only_oracle.len(),
            );
            let show = |ids: &[u32]| -> Vec<String> {
                ids.iter()
                    .take(8)
                    .map(|id| {
                        let text = vocab
                            .entries_map()
                            .get(id)
                            .map(|bytes| String::from_utf8_lossy(bytes).into_owned())
                            .unwrap_or_else(|| "?".to_string());
                        format!("{id}:{text:?}")
                    })
                    .collect()
            };
            if !only_cross.is_empty() {
                eprintln!("PHASE1 gate_{tag} crossing_only_examples={:?}", show(&only_cross));
            }
            if !only_oracle.is_empty() {
                eprintln!("PHASE1 gate_{tag} ORACLE_ONLY_EXAMPLES={:?}", show(&only_oracle));
            }
        }

        // Deliverable 3 pattern: templates over occurring terminals + standard
        // parser-DWA constructor over the composed (control-eliminated) table.
        fn phase1_parser_from_crossing(
            tag: &str,
            crossing: &DWA,
            table: &crate::compiler::glr::table::GLRTable,
            grammar: &AnalyzedGrammar,
            vocab: &Vocab,
            id_map: &InternalIdMap,
        ) {
            let mut selected = vec![false; table.num_terminals as usize];
            for state in crossing.states() {
                for &label in state.transitions.keys() {
                    if label >= 0 && (label as usize) < selected.len() {
                        selected[label as usize] = true;
                    }
                }
            }
            let active = selected.iter().filter(|&&value| value).count();
            let started = Instant::now();
            let (templates, _, _) = build_composition_templates(table, grammar, &selected);
            let template_ms = started.elapsed().as_secs_f64() * 1000.0;
            let started = Instant::now();
            let parser = build_parser_dwa_from_terminal_dwa_with_precomputed_templates(
                table,
                grammar,
                &TerminalAutomaton::Dwa(crossing.clone()),
                &templates,
                vocab,
                id_map,
                false,
            );
            let parser_ms = started.elapsed().as_secs_f64() * 1000.0;
            let acyclic = parser.is_acyclic();
            let minimized = if acyclic {
                let minimized =
                    crate::automata::weighted_u32::minimize_acyclic::minimize_acyclic_owned(
                        parser.clone(),
                    );
                format!("{}states/{}trans", minimized.num_states(), minimized.num_transitions())
            } else {
                "cyclic".to_string()
            };
            eprintln!(
                "PHASE1 parser_{tag} active_terms={active} crossing_states={} crossing_trans={} template_ms={template_ms:.3} parser_ms={parser_ms:.3} parser_states={} parser_trans={} acyclic={acyclic} min={minimized}",
                crossing.num_states(),
                crossing.num_transitions(),
                parser.num_states(),
                parser.num_transitions(),
            );
        }

        // DynamicDirect cross-check (UNION over live alternatives): every token
        // the exact dynamic backend admits at `core_prefix` (= full prefix, parsed
        // by core as plain JS) must be accepted by the core walk from core-alone
        // live states there, OR by the dispatch walk from dispatch-alone live
        // states over `disp_prefix` (the post-CALL suffix; fresh-spawn model for
        // positions before any CALL byte). A union violation is a REAL walk
        // miss (walks over-approximate parser-agnostically) → blocker.
        fn phase1_dynamic_crosscheck(
            tag: &str,
            core_prefix: &[u8],
            disp_prefix: &[u8],
            dyn_constraint: &Constraint,
            core: &Constraint,
            dispatch: &Constraint,
            vocab: &Vocab,
            core_walk: (&DWA, &InternalIdMap, &[bool]),
            disp_walk: (&DWA, &InternalIdMap, &[bool]),
            tokenizer_offsets: &[u32],
            crossing_tokens: &BTreeSet<u32>,
            ti_tokens: &BTreeSet<u32>,
            spot_tokens: &[u32],
            nd_crossing: &BTreeSet<u32>,
        ) {
            let started = Instant::now();
            let mut dyn_state = dyn_constraint.start();
            dyn_state.commit_bytes(core_prefix).expect("dynamic commit prefix");
            assert!(!dyn_state.is_rejected(), "dynamic rejected prefix");
            let mask = dyn_state.mask();
            let mut dyn_tokens = BTreeSet::new();
            for (word_index, &word) in mask.iter().enumerate() {
                let mut live = word;
                while live != 0 {
                    let bit = live.trailing_zeros() as usize;
                    dyn_tokens.insert((word_index * 32 + bit) as u32);
                    live &= live - 1;
                }
            }
            let live_for = |solo: &Constraint, prefix: &[u8], offset: u32, commit: &[bool]| -> Vec<bool> {
                let mut live_commit = vec![false; commit.len()];
                let mut solo_state = solo.start();
                if solo_state.commit_bytes(prefix).is_err() || solo_state.is_rejected() {
                    return live_commit;
                }
                for (local, _) in solo_state.state.entries.iter() {
                    let merged_state = offset + local;
                    if commit.get(merged_state as usize).copied().unwrap_or(false) {
                        live_commit[merged_state as usize] = true;
                    }
                }
                live_commit
            };
            let (core_dwa, core_id_map, commit_core) = core_walk;
            let (disp_dwa, disp_id_map, commit_disp) = disp_walk;
            let live_core = live_for(core, core_prefix, tokenizer_offsets[0], commit_core);
            let live_disp = live_for(dispatch, disp_prefix, tokenizer_offsets[1], commit_disp);
            let live_core_count = live_core.iter().filter(|&&selected| selected).count();
            let live_disp_count = live_disp.iter().filter(|&&selected| selected).count();
            let live_core_states: Vec<usize> = live_core
                .iter()
                .enumerate()
                .filter_map(|(state, &selected)| selected.then_some(state))
                .collect();
            let live_disp_states: Vec<usize> = live_disp
                .iter()
                .enumerate()
                .filter_map(|(state, &selected)| selected.then_some(state))
                .collect();
            eprintln!(
                "PHASE1 dyncheck_{tag} solo_live_core={live_core_states:?} solo_live_disp={live_disp_states:?}"
            );
            let keep_core = phase1_commit_internal(core_id_map, &live_core);
            let keep_disp = phase1_commit_internal(disp_id_map, &live_disp);
            let acc_core = phase1_accepted_tokens(core_dwa, core_id_map, Some(&keep_core));
            let acc_disp = phase1_accepted_tokens(disp_dwa, disp_id_map, Some(&keep_disp));
            let union_acc: BTreeSet<u32> =
                acc_core.union(&acc_disp).copied().collect();
            let violations: Vec<u32> = dyn_tokens.difference(&union_acc).copied().collect();
            eprintln!(
                "PHASE1 dyncheck_{tag} prefix={:?} dyn_tokens={} live_core={} live_disp={} acc_core={} acc_disp={} union={} violations={} ms={:.3}",
                String::from_utf8_lossy(core_prefix),
                dyn_tokens.len(),
                live_core_count,
                live_disp_count,
                acc_core.len(),
                acc_disp.len(),
                union_acc.len(),
                violations.len(),
                started.elapsed().as_secs_f64() * 1000.0,
            );
            let viol_in_crossing =
                violations.iter().filter(|token| crossing_tokens.contains(token)).count();
            let viol_in_ti =
                violations.iter().filter(|token| ti_tokens.contains(token)).count();
            let viol_in_nd =
                violations.iter().filter(|token| nd_crossing.contains(token)).count();
            eprintln!(
                "PHASE1 dyncheck_{tag} viol_in_crossing={viol_in_crossing} viol_in_ti={viol_in_ti} viol_in_ndcrossing={viol_in_nd}"
            );
            for token in violations.iter().take(8) {
                let bytes =
                    vocab.entries_map().get(token).map(Vec::as_slice).unwrap_or(&[]);
                eprintln!(
                    "PHASE1 dyncheck_{tag} VIOLATION t{token}={:?}",
                    String::from_utf8_lossy(bytes),
                );
            }
            for token in spot_tokens {
                let bytes =
                    vocab.entries_map().get(token).map(Vec::as_slice).unwrap_or(&[]);
                eprintln!(
                    "PHASE1 dynspot_{tag} t{token}={:?} dyn={} core_walk={} disp_walk={}",
                    String::from_utf8_lossy(bytes),
                    dyn_tokens.contains(token),
                    acc_core.contains(token),
                    acc_disp.contains(token),
                );
            }
            // True composed live keys (vs the solo-model live_core/live_disp).
            let mut dyn_keys: Vec<u32> =
                dyn_state.state.entries.iter().map(|(key, _)| *key).collect();
            dyn_keys.sort();
            dyn_keys.dedup();
            eprintln!(
                "PHASE1 dyncheck_{tag} dyn_live_keys={} first={:?}",
                dyn_keys.len(),
                dyn_keys.iter().take(40).collect::<Vec<_>>(),
            );
            // Discriminator: are violations missed even UNRESTRICTED (= walk
            // modeling gap) or only under the live-state restriction (= live
            // model / solo-vs-composed artifact)?
            let acc_core_all = phase1_accepted_tokens(core_dwa, core_id_map, None);
            let acc_disp_all = phase1_accepted_tokens(disp_dwa, disp_id_map, None);
            let viol_unrestr: Vec<u32> = violations
                .iter()
                .copied()
                .filter(|token| {
                    !acc_core_all.contains(token) && !acc_disp_all.contains(token)
                })
                .collect();
            eprintln!(
                "PHASE1 dyncheck_{tag} viol_unrestricted_miss={} of_violations={}",
                viol_unrestr.len(),
                violations.len(),
            );
            let viol_in_core_all =
                violations.iter().filter(|token| acc_core_all.contains(token)).count();
            let viol_in_disp_all =
                violations.iter().filter(|token| acc_disp_all.contains(token)).count();
            eprintln!(
                "PHASE1 dyncheck_{tag} viol_in_core_all={viol_in_core_all} viol_in_disp_all={viol_in_disp_all}"
            );
            // Solo-mask attribution: can each violation be admitted WITHOUT any
            // crossing (core-solo or disp-solo alone)? Both-no = composed-only
            // admission = TRUE crossing missed by the walk (blocker candidate).
            let mut core_solo_state = core.start();
            let core_solo_ok = core_solo_state.commit_bytes(core_prefix).is_ok()
                && !core_solo_state.is_rejected();
            let core_solo_mask =
                core_solo_ok.then(|| core_solo_state.mask()).unwrap_or_default();
            let mut disp_solo_state = dispatch.start();
            let disp_solo_ok = disp_solo_state.commit_bytes(disp_prefix).is_ok()
                && !disp_solo_state.is_rejected();
            let disp_solo_mask =
                disp_solo_ok.then(|| disp_solo_state.mask()).unwrap_or_default();
            let solo_admits = |mask: &[u32], token: u32| -> bool {
                mask.get((token / 32) as usize)
                    .is_some_and(|word| word & (1 << (token % 32)) != 0)
            };
            let mut viol_core_solo = 0usize;
            let mut viol_disp_solo = 0usize;
            let mut viol_neither_solo = Vec::new();
            for token in &violations {
                let in_core = solo_admits(&core_solo_mask, *token);
                let in_disp = solo_admits(&disp_solo_mask, *token);
                viol_core_solo += in_core as usize;
                viol_disp_solo += in_disp as usize;
                if !in_core && !in_disp {
                    viol_neither_solo.push(*token);
                }
            }
            eprintln!(
                "PHASE1 dyncheck_{tag} core_solo_ok={core_solo_ok} disp_solo_ok={disp_solo_ok} viol_core_solo={viol_core_solo} viol_disp_solo={viol_disp_solo} viol_neither_solo={}",
                viol_neither_solo.len(),
            );
            for token in viol_neither_solo.iter().take(8) {
                let bytes =
                    vocab.entries_map().get(token).map(Vec::as_slice).unwrap_or(&[]);
                eprintln!(
                    "PHASE1 dyncheck_{tag} NEITHER_SOLO t{token}={:?}",
                    String::from_utf8_lossy(bytes),
                );
            }
            for token in viol_unrestr.iter().take(8) {
                let bytes =
                    vocab.entries_map().get(token).map(Vec::as_slice).unwrap_or(&[]);
                eprintln!(
                    "PHASE1 dyncheck_{tag} UNRESTR_MISS t{token}={:?}",
                    String::from_utf8_lossy(bytes),
                );
            }
        }

        fn phase1_identity_id_map(tokenizer: &Tokenizer, vocab: &Vocab) -> InternalIdMap {
            let num_states = tokenizer.num_states() as usize;
            let tokenizer_states = ManyToOneIdMap {
                original_to_internal: (0..num_states as u32).collect(),
                internal_to_originals: (0..num_states as u32).map(|state| vec![state]).collect(),
                representative_original_ids: (0..num_states as u32).collect(),
            };
            let max_token = vocab.max_token_id() as usize;
            let mut original_to_internal = vec![u32::MAX; max_token + 1];
            let mut internal_to_originals = Vec::new();
            let mut representative_original_ids = Vec::new();
            for (&token, _) in vocab.entries_map().iter() {
                let internal = internal_to_originals.len() as u32;
                original_to_internal[token as usize] = internal;
                internal_to_originals.push(vec![token]);
                representative_original_ids.push(token);
            }
            InternalIdMap {
                tokenizer_states,
                vocab_tokens: ManyToOneIdMap {
                    original_to_internal,
                    internal_to_originals,
                    representative_original_ids,
                },
                deferred_vocab_singleton_original_ids: None,
            }
        }

        // Probe-side replication of the standard L2P post-id_map pipeline with
        // a caller-supplied id_map (bypass variants b1/b2). Same walk, same
        // postprocess order; TI replay/expansion omitted (the b2 gate trips on
        // any divergence). Returns the compacted DWA + id_map with per-stage ms.
        fn phase1_run_walk_replica(
            tag: &str,
            tokenizer: &Tokenizer,
            vocab: &Vocab,
            grammar: &AnalyzedGrammar,
            disallowed: &BTreeMap<u32, BitSet>,
            ignore_terminal: Option<u32>,
            seed_filter: Option<&[bool]>,
            id_map: &InternalIdMap,
        ) -> Option<(DWA, InternalIdMap)> {
            use crate::compiler::possible_matches::PossibleMatchesComputer;
            use crate::compiler::stages::id_map_and_terminal_dwa as tdwa;
            use crate::ds::vocab_prefix_tree::VocabPrefixTree;
            let num_terms = grammar.num_terminals as usize;
            let coloring = TerminalColoring::identity(num_terms);
            let always_allowed = tdwa::grammar_helpers::compute_always_allowed_follows(grammar);
            let active = vec![true; num_terms];
            let flat: Arc<[u32]> = Arc::from(tdwa::l1::build_flat_transition_table(tokenizer));
            let started = Instant::now();
            let internal_vocab = tdwa::l2p::nwa_builder::internal_vocab_entries(vocab, id_map);
            if internal_vocab.is_empty() {
                eprintln!("PHASE1 replica_{tag} SKIPPED_empty_vocab");
                return None;
            }
            let full_tree = VocabPrefixTree::build_owned(
                internal_vocab.iter().map(|(id, bytes)| (*id as usize, bytes.clone())).collect(),
            );
            let trie_ms = started.elapsed().as_secs_f64() * 1000.0;
            let mut possible_matches = PossibleMatchesComputer::new(tokenizer);
            let mut nwa = NWA::new(id_map.num_tsids(), id_map.max_internal_token_id());
            let leaf_state = nwa.add_state();
            nwa.set_final_weight(leaf_state, Weight::all());
            let start_state = nwa.add_state();
            nwa.start_states_mut().push(start_state);
            let seed_started = Instant::now();
            let roots = match seed_filter {
                Some(keep) => tdwa::l2p::nwa_builder::seed_root_nodes_filtered(
                    tokenizer, &mut nwa, start_state, id_map, keep,
                ),
                None => tdwa::l2p::nwa_builder::seed_root_nodes(tokenizer, &mut nwa, start_state, id_map),
            };
            let seed_ms = seed_started.elapsed().as_secs_f64() * 1000.0;
            let flat_opt: Option<&Arc<[u32]>> = Some(&flat);
            let build_profile = tdwa::l2p::nwa_builder::build_nwa_via_trie_walk(
                tokenizer,
                &coloring,
                false,
                ignore_terminal,
                &mut nwa,
                leaf_state,
                id_map.num_tsids(),
                &full_tree.root,
                &roots,
                &mut possible_matches,
                flat_opt.map(AsRef::as_ref),
                &active,
            );
            let nwa_after_build = nwa.states().len();
            let step_started = Instant::now();
            tdwa::l2p::postprocess::collapse_always_allowed(
                &mut nwa,
                &always_allowed,
                grammar.num_terminals as usize,
            );
            let collapse_ms = step_started.elapsed().as_secs_f64() * 1000.0;
            let step_started = Instant::now();
            tdwa::l2p::postprocess::apply_disallowed_follow_constraints(
                &mut nwa,
                disallowed,
                grammar.num_terminals as usize,
                ignore_terminal,
                None,
            );
            let disallowed_ms = step_started.elapsed().as_secs_f64() * 1000.0;
            let step_started = Instant::now();
            tdwa::l2p::postprocess::prune_non_coreachable_states(&mut nwa);
            let prune_ms = step_started.elapsed().as_secs_f64() * 1000.0;
            let step_started = Instant::now();
            tdwa::l2p::postprocess::canonicalize_acyclic_nwa(&mut nwa);
            let canon_ms = step_started.elapsed().as_secs_f64() * 1000.0;
            let nwa_after_canon = nwa.states().len();
            let step_started = Instant::now();
            let det = determinize(&nwa).expect("replica determinization");
            let det_ms = step_started.elapsed().as_secs_f64() * 1000.0;
            let det_states = det.num_states();
            let det_trans = det.num_transitions();
            let step_started = Instant::now();
            let dwa = minimize_owned(det);
            let min_ms = step_started.elapsed().as_secs_f64() * 1000.0;
            let step_started = Instant::now();
            let mut mapped = MappedArtifact::new(dwa, id_map.clone());
            mapped.compact_dimensions_fast();
            let (dwa, new_id_map) = mapped.into_parts();
            let compact_ms = step_started.elapsed().as_secs_f64() * 1000.0;
            eprintln!(
                "PHASE1 replica_{tag} vocab={} tsids={} itokens={} trie_ms={trie_ms:.3} seed_ms={seed_ms:.3} walk_ms={:.3} flush_ms={:.3} exec_calls={} matches={} match_adds={} nwa={nwa_after_build}->{nwa_after_canon} collapse_ms={collapse_ms:.3} disallowed_ms={disallowed_ms:.3} prune_ms={prune_ms:.3} canon_ms={canon_ms:.3} det_ms={det_ms:.3} det_states={det_states} det_trans={det_trans} min_ms={min_ms:.3} compact_ms={compact_ms:.3} dwa_states={} dwa_trans={} wall_ms={:.3}",
                vocab.entries_map().len(),
                id_map.num_tsids(),
                id_map.num_internal_tokens(),
                build_profile.trie_walk_ms,
                build_profile.flush_ms,
                build_profile.trie_execute_calls,
                build_profile.trie_matches,
                build_profile.match_transition_additions,
                dwa.num_states(),
                dwa.num_transitions(),
                started.elapsed().as_secs_f64() * 1000.0,
            );
            Some((dwa, new_id_map))
        }

        fn phase1_dispatcher_literal_parent_source() -> String {
            let mut source = String::from("start suffix;\n");
            for index in 0..PHASE1_SELECTED10_SHORT.len() {
                source.push_str(&format!(
                    "t TOOL_ARGS_SLOT_{index} ::= @token({});\n",
                    128_320 + PHASE1_SELECTED10_SHORT.len() as u32 + index as u32
                ));
            }
            source.push_str("nt suffix ::=\n    ");
            for index in 0..PHASE1_SELECTED10_SHORT.len() {
                if index != 0 {
                    source.push_str("\n  | ");
                }
                source.push_str(&format!(r#"".tool_{index}(" TOOL_ARGS_SLOT_{index} ")""#));
            }
            source.push_str(";\n");
            source
        }

        fn phase1_validate_unified_walk(
            tag: &str,
            walk_dwa: &DWA,
            walk_id_map: &InternalIdMap,
            b_mapped: &MappedArtifact<DWA>,
        ) {
            if !walk_dwa.is_acyclic() || !b_mapped.artifact().is_acyclic() {
                eprintln!(
                    "PHASE1 validate_{tag} SKIPPED_cyclic walk_acyclic={} b_acyclic={}",
                    walk_dwa.is_acyclic(),
                    b_mapped.artifact().is_acyclic(),
                );
                return;
            }
            let (b_dwa, b_id_map) = b_mapped.clone().into_parts();
            let reconciled = MappedArtifact::reconcile_vec(vec![
                MappedArtifact::new(walk_dwa.clone(), walk_id_map.clone()),
                MappedArtifact::new(b_dwa, b_id_map),
            ]);
            let (all, _) = reconciled.into_parts();
            for (name, first, second) in [("walk_minus_B", &all[0], &all[1]), ("B_minus_walk", &all[1], &all[0])] {
                let diff = exact_weighted_difference(first, second);
                let finals = diff
                    .states()
                    .iter()
                    .filter(|state| {
                        state.final_weight.as_ref().is_some_and(|weight| !weight.is_empty())
                    })
                    .count();
                eprintln!(
                    "PHASE1 validate_{tag}_{name} states={} trans={} nonempty_finals={finals}",
                    diff.num_states(),
                    diff.num_transitions(),
                );
            }
        }

        // ---------- inputs ----------
        let root = std::env::var("PHASE1_DIR").unwrap_or_else(|_| {
            "/Users/isaacbreen/Projects2/temp/2026-09/glrmask-selected10-cache-v29".to_string()
        });
        let root = Path::new(&root).to_path_buf();
        let vocab_path = std::env::var("PHASE1_VOCAB")
            .unwrap_or_else(|_| root.join("vocab_dump.bin").to_string_lossy().into_owned());
        let dump_dir = std::env::var("PHASE1_DUMP_DIR")
            .unwrap_or_else(|_| "/tmp/grammars25-redesign".to_string());
        let only = std::env::var("PHASE1_ONLY").unwrap_or_default();
        if !only.is_empty() && only != "core" && only != "schema" {
            panic!("PHASE1_ONLY must be core|schema, got {only:?}");
        }
        let skip_oracle = std::env::var_os("PHASE1_SKIP_ORACLE").is_some();
        let vocab = load_vocab(&vocab_path);
        eprintln!(
            "PHASE1 setup vocab_tokens={} dir={} dump_dir={dump_dir} skip_oracle={skip_oracle}",
            vocab.entries_map().len(),
            root.display(),
        );

        // ---------- outer composition: core + dispatch, i = core ----------
        if only.is_empty() || only == "core" {
            let mut core = phase1_load(&root, "core.bin");
            let dispatch_name = std::env::var("PHASE1_DISPATCH")
                .unwrap_or_else(|_| "dispatch-literal.bin".to_string());
            let mut dispatch = phase1_load(&root, &dispatch_name);
            phase1_restore(&mut core, "core");
            phase1_restore(&mut dispatch, "dispatch");
            let placeholder = terminal(&core, "PROGRAMMATIC_TOOL_SUFFIX");
            let child = CompiledSubgrammarInput {
                placeholder_terminal: placeholder,
                additional_placeholder_terminals: &[],
                constraint: &dispatch,
            };
            let composed = phase1_compose_low_level("outer", &core, &[child]);
            let grammar = phase1_grammar(&composed.table.table, &composed.terminal_names);
            let disallowed = crate::compiler::pipeline::compute_disallowed_follows(&grammar);
            let commit_core = phase1_commit_range(
                &composed.tokenizer_offsets,
                core.tokenizer.num_states(),
                0,
                composed.tokenizer.num_states() as usize,
            );
            let commit_dispatch = phase1_commit_range(
                &composed.tokenizer_offsets,
                dispatch.tokenizer.num_states(),
                1,
                composed.tokenizer.num_states() as usize,
            );

            let dump_path = format!("{dump_dir}/minbound-tokens.txt");
            let (oracle_tokens_all, oracle_core, oracle_dispatch, b_mapped): (
                BTreeSet<u32>,
                Option<BTreeSet<u32>>,
                Option<BTreeSet<u32>>,
                Option<MappedArtifact<DWA>>,
            ) = if skip_oracle {
                let text = fs::read_to_string(&dump_path)
                    .expect("oracle dump missing; run once without PHASE1_SKIP_ORACLE");
                (
                    text.split_whitespace().map(|value| value.parse::<u32>().unwrap()).collect(),
                    None,
                    None,
                    None,
                )
            } else {
                let oracle = phase1_oracle(
                    "outer",
                    &composed,
                    &[
                        (
                            &core.tokenizer,
                            &core.table,
                            &core.terminal_display_names[..],
                            core.ignore_terminal,
                        ),
                        (
                            &dispatch.tokenizer,
                            &dispatch.table,
                            &dispatch.terminal_display_names[..],
                            dispatch.ignore_terminal,
                        ),
                    ],
                    &vocab,
                );
                let text = oracle
                    .tokens_all
                    .iter()
                    .map(|token| token.to_string())
                    .collect::<Vec<_>>()
                    .join(" ");
                fs::write(&dump_path, format!("{text}\n")).expect("write oracle dump");
                let keep_core = phase1_commit_internal(&oracle.id_map, &commit_core);
                let keep_dispatch = phase1_commit_internal(&oracle.id_map, &commit_dispatch);
                let restricted_core =
                    phase1_accepted_tokens(&oracle.residual, &oracle.id_map, Some(&keep_core));
                let restricted_dispatch =
                    phase1_accepted_tokens(&oracle.residual, &oracle.id_map, Some(&keep_dispatch));
                let both = restricted_core.intersection(&restricted_dispatch).count();
                let neither = oracle
                    .tokens_all
                    .difference(&restricted_core)
                    .filter(|token| !restricted_dispatch.contains(token))
                    .count();
                eprintln!(
                    "PHASE1 oracle_outer restricted_to_core={} restricted_to_dispatch={} both={both} neither={neither} all={}",
                    restricted_core.len(),
                    restricted_dispatch.len(),
                    oracle.tokens_all.len(),
                );
                (
                    oracle.tokens_all,
                    Some(restricted_core),
                    Some(restricted_dispatch),
                    Some(oracle.b_mapped),
                )
            };

            let ti_core_path = format!("{dump_dir}/phase1-ti-outer-core.txt");
            let ti_core = if skip_oracle {
                let text = fs::read_to_string(&ti_core_path).expect("T dump missing");
                text.split_whitespace().map(|value| value.parse::<u32>().unwrap()).collect()
            } else {
                let ti = phase1_compute_ti("outer_core", &core.tokenizer, &vocab);
                let text =
                    ti.iter().map(|token| token.to_string()).collect::<Vec<_>>().join(" ");
                fs::write(&ti_core_path, format!("{text}\n")).expect("write T dump");
                ti
            };
            let ti_vocab = Vocab::new(
                vocab
                    .entries_map()
                    .iter()
                    .filter(|(token, _)| ti_core.contains(token))
                    .map(|(&token, bytes)| (token, bytes.clone()))
                    .collect(),
            );

            if let Some(w0) =
                phase1_run_walk("outer_all_full", &composed.tokenizer, &vocab, &grammar, &disallowed, composed.ignore_canonical, None, None, None)
                && let Some(b) = b_mapped.as_ref()
            {
                phase1_validate_unified_walk("outer", &w0.dwa, &w0.id_map, b);
            }
            let wa = phase1_run_walk(
                "outer_core_full",
                &composed.tokenizer,
                &vocab,
                &grammar,
                &disallowed,
                composed.ignore_canonical,
                Some(&commit_core),
            None,
                Some((&composed.table.terminal_offsets, 0))
            )
            .expect("core full-vocab walk");
            let wb = phase1_run_walk(
                "outer_core_ti",
                &composed.tokenizer,
                &ti_vocab,
                &grammar,
                &disallowed,
                composed.ignore_canonical,
                Some(&commit_core),
            None,
                Some((&composed.table.terminal_offsets, 0))
            );
            let xa = phase1_crossing_from("outer_core_full", &wa.dwa, &composed.table.terminal_offsets, 0);
            let wa_total = phase1_accepted_tokens(&wa.dwa, &wa.id_map, None);
            eprintln!("PHASE1 walktokens_outer_core_full={}", wa_total.len());
            let ta = phase1_accepted_tokens(&xa, &wa.id_map, None);
            eprintln!(
                "PHASE1 tokens_outer_core_full={} paths={}",
                ta.len(),
                phase1_count_paths(&xa),
            );
            eprintln!(
                "PHASE1 ticheck_outer crossing_not_in_T={}",
                ta.difference(&ti_core).count(),
            );
            let mut ti_full_core = BTreeSet::new();
            let mut ti_cross_core = BTreeSet::new();
            if let Some(wb) = wb {
                let xb = phase1_crossing_from(
                    "outer_core_ti",
                    &wb.dwa,
                    &composed.table.terminal_offsets,
                    0,
                );
                let tb = phase1_accepted_tokens(&xb, &wb.id_map, None);
                eprintln!(
                    "PHASE1 tokens_outer_core_ti={} paths={}",
                    tb.len(),
                    phase1_count_paths(&xb),
                );
                phase1_token_diff_report("outer_a_vs_b", &ta, &tb, &vocab);
                ti_full_core = phase1_accepted_tokens(&wb.dwa, &wb.id_map, None);
                ti_cross_core = tb;
            }
            if let Some(restricted) = oracle_core.as_ref() {
                phase1_token_diff_report("outer_core_vs_oracle_i", &ta, restricted, &vocab);
            }
            phase1_token_diff_report("outer_core_vs_oracle_all", &ta, &oracle_tokens_all, &vocab);
            phase1_parser_from_crossing(
                "outer_core_full",
                &xa,
                &composed.table.table,
                &grammar,
                &vocab,
                &wa.id_map,
            );

            // Causality: the same seeded walk WITHOUT table disallowed-follows
            // must expose the raw lexer crossings (proves X_core emptiness comes
            // from the composed table, not from the merged reset/fan-out).
            let nodisallow =
                std::env::var("PHASE1_NODISALLOW").map(|value| value != "0").unwrap_or(true);
            let mut nd_crossing_std = BTreeSet::new();
            if nodisallow {
                let empty_disallowed = BTreeMap::new();
                if let Some((dwa_nd, id_nd)) = phase1_run_walk_replica(
                    "outer_core_nodisallow",
                    &composed.tokenizer,
                    &vocab,
                    &grammar,
                    &empty_disallowed,
                    composed.ignore_canonical,
                    Some(&commit_core),
                    &wa.id_map,
                ) {
                    let x_nd = phase1_crossing_from(
                        "outer_core_nodisallow",
                        &dwa_nd,
                        &composed.table.terminal_offsets,
                        0,
                    );
                    let t_nd = phase1_accepted_tokens(&x_nd, &id_nd, None);
                    eprintln!(
                        "PHASE1 tokens_outer_core_nodisallow={} paths={}",
                        t_nd.len(),
                        phase1_count_paths(&x_nd),
                    );
                    for spot in [2358u32, 2313, 6226, 1287, 28937, 17289, 22715] {
                        if t_nd.contains(&spot) {
                            let bytes = vocab
                                .entries_map()
                                .get(&spot)
                                .map(Vec::as_slice)
                                .unwrap_or(&[]);
                            eprintln!(
                                "PHASE1 nodisallow_contains t{spot}={:?}",
                                String::from_utf8_lossy(bytes),
                            );
                        }
                    }
                }
                // Faithful version through the STANDARD walk entry point (the
                // replica over-accepts; see b2 gate): same seeds, empty table.
                let empty_disallowed = BTreeMap::new();
                if let Some(w_nd) = phase1_run_walk(
                    "outer_core_nodisallow_std",
                    &composed.tokenizer,
                    &vocab,
                    &grammar,
                    &empty_disallowed,
                    composed.ignore_canonical,
                    Some(&commit_core),
                    None,
                    Some((&composed.table.terminal_offsets, 0))
                ) {
                    let x_nd = phase1_crossing_from(
                        "outer_core_nodisallow_std",
                        &w_nd.dwa,
                        &composed.table.terminal_offsets,
                        0,
                    );
                    let t_nd = phase1_accepted_tokens(&x_nd, &w_nd.id_map, None);
                    eprintln!(
                        "PHASE1 tokens_outer_core_nodisallow_std={} paths={}",
                        t_nd.len(),
                        phase1_count_paths(&x_nd),
                    );
                    nd_crossing_std = t_nd.clone();
                    for spot in [2358u32, 2313, 6226, 1287, 28937, 17289, 22715] {
                        if t_nd.contains(&spot) {
                            let bytes = vocab
                                .entries_map()
                                .get(&spot)
                                .map(Vec::as_slice)
                                .unwrap_or(&[]);
                            eprintln!(
                                "PHASE1 nodisallow_std_contains t{spot}={:?}",
                                String::from_utf8_lossy(bytes),
                            );
                        }
                    }
                }
            }

            // Same outer composition, i = dispatch (the nonempty direction).
            let ti_dispatch_path = format!("{dump_dir}/phase1-ti-outer-dispatch.txt");
            let ti_dispatch = if skip_oracle {
                let text = fs::read_to_string(&ti_dispatch_path).expect("T dump missing");
                text.split_whitespace().map(|value| value.parse::<u32>().unwrap()).collect()
            } else {
                let ti = phase1_compute_ti("outer_dispatch", &dispatch.tokenizer, &vocab);
                let text =
                    ti.iter().map(|token| token.to_string()).collect::<Vec<_>>().join(" ");
                fs::write(&ti_dispatch_path, format!("{text}\n")).expect("write T dump");
                ti
            };
            let ti_vocab_dispatch = Vocab::new(
                vocab
                    .entries_map()
                    .iter()
                    .filter(|(token, _)| ti_dispatch.contains(token))
                    .map(|(&token, bytes)| (token, bytes.clone()))
                    .collect(),
            );
            let wa_dispatch = phase1_run_walk(
                "outer_dispatch_full",
                &composed.tokenizer,
                &vocab,
                &grammar,
                &disallowed,
                composed.ignore_canonical,
                Some(&commit_dispatch),
            None,
                Some((&composed.table.terminal_offsets, 1))
            )
            .expect("dispatch full-vocab walk");
            let wb_dispatch = phase1_run_walk(
                "outer_dispatch_ti",
                &composed.tokenizer,
                &ti_vocab_dispatch,
                &grammar,
                &disallowed,
                composed.ignore_canonical,
                Some(&commit_dispatch),
            None,
                Some((&composed.table.terminal_offsets, 1))
            );
            let xa_dispatch = phase1_crossing_from(
                "outer_dispatch_full",
                &wa_dispatch.dwa,
                &composed.table.terminal_offsets,
                1,
            );
            let wa_dispatch_total =
                phase1_accepted_tokens(&wa_dispatch.dwa, &wa_dispatch.id_map, None);
            eprintln!("PHASE1 walktokens_outer_dispatch_full={}", wa_dispatch_total.len());
            let ta_dispatch = phase1_accepted_tokens(&xa_dispatch, &wa_dispatch.id_map, None);
            eprintln!(
                "PHASE1 tokens_outer_dispatch_full={} paths={}",
                ta_dispatch.len(),
                phase1_count_paths(&xa_dispatch),
            );
            eprintln!(
                "PHASE1 ticheck_outer_dispatch crossing_not_in_T={}",
                ta_dispatch.difference(&ti_dispatch).count(),
            );
            // NWA-vs-DWA filter proof (Phase 2 step 2): the DWA-level
            // product minimized with the full minimizer must be
            // weighted-language-equal to the acyclic-minimized form. In a
            // non-NWAFILT run this proves the 41-vs-26 size delta is a
            // minimizer artifact, not a language difference; under NWAFILT
            // it additionally proves the walk DWA is a re-filter fixpoint
            // (no non-crossing accepting paths survived the NWA filter).
            let owned_min = minimize_owned(xa_dispatch.clone());
            let fwd = find_difference(&xa_dispatch, &owned_min)
                .expect("crossing minimizer-equivalence check failed");
            let bwd = find_difference(&owned_min, &xa_dispatch)
                .expect("crossing minimizer-equivalence check failed");
            eprintln!(
                "PHASE1 nwaequiv_outer_dispatch walk_states={} walk_trans={} acyclic_states={} acyclic_trans={} owned_states={} owned_trans={} fwd_none={} bwd_none={}",
                wa_dispatch.dwa.num_states(),
                wa_dispatch.dwa.num_transitions(),
                xa_dispatch.num_states(),
                xa_dispatch.num_transitions(),
                owned_min.num_states(),
                owned_min.num_transitions(),
                fwd.is_none(),
                bwd.is_none(),
            );
            assert!(
                fwd.is_none() && bwd.is_none(),
                "DWA-filtered crossing language differs across minimizers",
            );
            // Grammar-factor filter measurement (Phase 2 step 3): apply the
            // exact factor oracle to the dispatch crossing DWA and report
            // terminal/token counts + time. Production does NOT wire this in
            // (see F report); this is measurement only.
            if std::env::var("PHASE1_FACTOR").map(|value| value != "0").unwrap_or(false) {
                let mut factor_zero_width = composed.table.table.control_terminals.clone();
                factor_zero_width
                    .extend(composed.table.table.skip_terminals.iter().copied());
                factor_zero_width
                    .extend(composed.scoped_ignores.iter().map(|terminal| terminal as u32));
                let count_terms = |dwa: &DWA| {
                    let mut selected =
                        vec![false; composed.table.table.num_terminals as usize];
                    for state in dwa.states() {
                        for &label in state.transitions.keys() {
                            if label >= 0 && (label as usize) < selected.len() {
                                selected[label as usize] = true;
                            }
                        }
                    }
                    selected.iter().filter(|slot| **slot).count()
                };
                let terms_before = count_terms(&xa_dispatch);
                let started = Instant::now();
                let filtered =
                    mb_filter_exact_factor_lazy(&xa_dispatch, &grammar, &factor_zero_width);
                let filter_ms = started.elapsed().as_secs_f64() * 1000.0;
                let terms_after = count_terms(&filtered);
                let tokens_after =
                    phase1_accepted_tokens(&filtered, &wa_dispatch.id_map, None);
                let dropped: Vec<u32> =
                    ta_dispatch.difference(&tokens_after).copied().collect();
                eprintln!(
                    "PHASE1 factor_outer_dispatch states_before={} states_after={} terms_before={terms_before} terms_after={terms_after} tokens_before={} tokens_after={} dropped={dropped:?} ms={filter_ms:.3}",
                    xa_dispatch.num_states(),
                    filtered.num_states(),
                    ta_dispatch.len(),
                    tokens_after.len(),
                );
            }
            // Decisive same-coordinate NWA-vs-DWA proof (Phase 2 step 2).
            // Requires GLRMASK_L2P_SKIP_CORE_COMPACT=1 (both walks in Step-1
            // coordinates) and PHASE1_NWAFILT=1 (wa_dispatch is NWA-filtered):
            // the NWA-filtered walk DWA must be weighted-language-equal to
            // the DWA-level product (minimized with `minimize_owned`) of a
            // separately-run unfiltered walk over the identical id_map.
            let run_nwa_equiv =
                std::env::var("PHASE1_NWA_EQUIV").map(|value| value != "0").unwrap_or(false);
            let nwafilt_on =
                std::env::var("PHASE1_NWAFILT").map(|value| value != "0").unwrap_or(false);
            if run_nwa_equiv && nwafilt_on {
                let w_unfilt = phase1_run_walk(
                    "outer_dispatch_full_unfilt",
                    &composed.tokenizer,
                    &vocab,
                    &grammar,
                    &disallowed,
                    composed.ignore_canonical,
                    Some(&commit_dispatch),
                    None,
                    None,
                )
                .expect("unfiltered dispatch walk for NWA equivalence");
                phase1_assert_same_id_map(
                    "outer_dispatch_nwaequiv",
                    &wa_dispatch.id_map,
                    &w_unfilt.id_map,
                );
                let raw_unfilt = phase1_crossing_product(
                    &w_unfilt.dwa,
                    &composed.table.terminal_offsets,
                    1,
                );
                let dwa_owned = minimize_owned(raw_unfilt);
                let fwd_eq = find_difference(&wa_dispatch.dwa, &dwa_owned)
                    .expect("NWA-vs-DWA equivalence check failed");
                let bwd_eq = find_difference(&dwa_owned, &wa_dispatch.dwa)
                    .expect("NWA-vs-DWA equivalence check failed");
                eprintln!(
                    "PHASE1 nwaequiv_proof_outer_dispatch nwa_states={} nwa_trans={} dwa_states={} dwa_trans={} fwd_none={} bwd_none={}",
                    wa_dispatch.dwa.num_states(),
                    wa_dispatch.dwa.num_transitions(),
                    dwa_owned.num_states(),
                    dwa_owned.num_transitions(),
                    fwd_eq.is_none(),
                    bwd_eq.is_none(),
                );
                assert!(
                    fwd_eq.is_none() && bwd_eq.is_none(),
                    "NWA-filtered crossing language differs from DWA-filtered crossing language",
                );
            }
            let mut ti_full_outer = BTreeSet::new();
            let mut ti_cross_outer = BTreeSet::new();
            if let Some(wb) = wb_dispatch {
                let xb = phase1_crossing_from(
                    "outer_dispatch_ti",
                    &wb.dwa,
                    &composed.table.terminal_offsets,
                    1,
                );
                let tb = phase1_accepted_tokens(&xb, &wb.id_map, None);
                eprintln!(
                    "PHASE1 tokens_outer_dispatch_ti={} paths={}",
                    tb.len(),
                    phase1_count_paths(&xb),
                );
                phase1_token_diff_report("outer_dispatch_a_vs_b", &ta_dispatch, &tb, &vocab);
                ti_full_outer = phase1_accepted_tokens(&wb.dwa, &wb.id_map, None);
                ti_cross_outer = tb;
                // TI determinization anatomy: same inputs through the replica to
                // expose the determinized (pre-minimize) size.
                if std::env::var("PHASE1_TIB2").map(|value| value != "0").unwrap_or(true) {
                    if let Some((dwa_tib2, id_tib2)) = phase1_run_walk_replica(
                        "outer_dispatch_ti_b2reuse",
                        &composed.tokenizer,
                        &ti_vocab_dispatch,
                        &grammar,
                        &disallowed,
                        composed.ignore_canonical,
                        Some(&commit_dispatch),
                        &wb.id_map,
                    ) {
                        let x_tib2 = phase1_crossing_from(
                            "outer_dispatch_ti_b2reuse",
                            &dwa_tib2,
                            &composed.table.terminal_offsets,
                            1,
                        );
                        let t_tib2 = phase1_accepted_tokens(&x_tib2, &id_tib2, None);
                        phase1_token_diff_report(
                            "outer_dispatch_tib2_vs_a",
                            &t_tib2,
                            &ta_dispatch,
                            &vocab,
                        );
                    }
                }
            }
            if let Some(restricted) = oracle_dispatch.as_ref() {
                phase1_token_diff_report(
                    "outer_dispatch_vs_oracle_i",
                    &ta_dispatch,
                    restricted,
                    &vocab,
                );
            }
            phase1_token_diff_report(
                "outer_dispatch_vs_oracle_all",
                &ta_dispatch,
                &oracle_tokens_all,
                &vocab,
            );
            phase1_parser_from_crossing(
                "outer_dispatch_full",
                &xa_dispatch,
                &composed.table.table,
                &grammar,
                &vocab,
                &wa_dispatch.id_map,
            );

            let bypass =
                std::env::var("PHASE1_BYPASS").map(|value| value != "0").unwrap_or(true);
            let run_b1 =
                std::env::var("PHASE1_B1").map(|value| value != "0").unwrap_or(true);
            if bypass {
                // b2: standard id_map reused (analysis amortized) — must reproduce.
                if let Some((dwa_b2, id_b2)) = phase1_run_walk_replica(
                    "outer_dispatch_b2reuse",
                    &composed.tokenizer,
                    &vocab,
                    &grammar,
                    &disallowed,
                    composed.ignore_canonical,
                    Some(&commit_dispatch),
                    &wa_dispatch.id_map,
                ) {
                    let x_b2 = phase1_crossing_from(
                        "outer_dispatch_b2reuse",
                        &dwa_b2,
                        &composed.table.terminal_offsets,
                        1,
                    );
                    let t_b2 = phase1_accepted_tokens(&x_b2, &id_b2, None);
                    phase1_token_diff_report("outer_dispatch_b2_vs_a", &t_b2, &ta_dispatch, &vocab);
                    let f_b2 = phase1_accepted_tokens(&dwa_b2, &id_b2, None);
                    phase1_token_diff_report(
                        "outer_dispatch_b2_full_vs_a",
                        &f_b2,
                        &wa_dispatch_total,
                        &vocab,
                    );
                    eprintln!(
                        "PHASE1 b2check_outer_dispatch dwa_states={} vs_a={} dwa_trans={} vs_a={}",
                        dwa_b2.num_states(),
                        wa_dispatch.dwa.num_states(),
                        dwa_b2.num_transitions(),
                        wa_dispatch.dwa.num_transitions(),
                    );
                    phase1_parser_from_crossing(
                        "outer_dispatch_b2reuse",
                        &x_b2,
                        &composed.table.table,
                        &grammar,
                        &vocab,
                        &id_b2,
                    );
                }
                // b1: identity id_map — no equivalence analysis at all.
                if run_b1 {
                    let id_b1 = phase1_identity_id_map(&composed.tokenizer, &vocab);
                    if let Some((dwa_b1, id_b1c)) = phase1_run_walk_replica(
                        "outer_dispatch_b1ident",
                        &composed.tokenizer,
                        &vocab,
                        &grammar,
                        &disallowed,
                        composed.ignore_canonical,
                        Some(&commit_dispatch),
                        &id_b1,
                    ) {
                        let x_b1 = phase1_crossing_from(
                            "outer_dispatch_b1ident",
                            &dwa_b1,
                            &composed.table.terminal_offsets,
                            1,
                        );
                        let t_b1 = phase1_accepted_tokens(&x_b1, &id_b1c, None);
                        phase1_token_diff_report(
                            "outer_dispatch_b1_vs_a",
                            &t_b1,
                            &ta_dispatch,
                            &vocab,
                        );
                        let f_b1 = phase1_accepted_tokens(&dwa_b1, &id_b1c, None);
                        phase1_token_diff_report(
                            "outer_dispatch_b1_full_vs_a",
                            &f_b1,
                            &wa_dispatch_total,
                            &vocab,
                        );
                        phase1_parser_from_crossing(
                            "outer_dispatch_b1ident",
                            &x_b1,
                            &composed.table.table,
                            &grammar,
                            &vocab,
                            &id_b1c,
                        );
                    }
                }
            }

            // b3: equivalence restricted to Commit_i states × TI vocab (the
            // §2.4.3 vehicle): same standard walk entry point, subset
            // initial_state_map; must reproduce the TI walk exactly.
            let run_b3 =
                std::env::var("PHASE1_B3").map(|value| value != "0").unwrap_or(true);
            if run_b3 {
                let state_map_dispatch = phase1_restricted_state_map(
                    composed.tokenizer.num_states() as usize,
                    &commit_dispatch,
                );
                if let Some(w_b3) = phase1_run_walk(
                    "outer_dispatch_b3restr",
                    &composed.tokenizer,
                    &ti_vocab_dispatch,
                    &grammar,
                    &disallowed,
                    composed.ignore_canonical,
                    Some(&commit_dispatch),
                    Some(&state_map_dispatch),
                    Some((&composed.table.terminal_offsets, 1))
                ) {
                    let x_b3 = phase1_crossing_from(
                        "outer_dispatch_b3restr",
                        &w_b3.dwa,
                        &composed.table.terminal_offsets,
                        1,
                    );
                    let t_b3 = phase1_accepted_tokens(&x_b3, &w_b3.id_map, None);
                    phase1_token_diff_report(
                        "outer_dispatch_b3_vs_ti",
                        &t_b3,
                        &ti_cross_outer,
                        &vocab,
                    );
                    let f_b3 = phase1_accepted_tokens(&w_b3.dwa, &w_b3.id_map, None);
                    phase1_token_diff_report(
                        "outer_dispatch_b3_full_vs_ti",
                        &f_b3,
                        &ti_full_outer,
                        &vocab,
                    );
                    phase1_parser_from_crossing(
                        "outer_dispatch_b3restr",
                        &x_b3,
                        &composed.table.table,
                        &grammar,
                        &vocab,
                        &w_b3.id_map,
                    );
                }
                // b3 for the small component: equivalence over 1286 core states.
                let state_map_core = phase1_restricted_state_map(
                    composed.tokenizer.num_states() as usize,
                    &commit_core,
                );
                if let Some(w_b3c) = phase1_run_walk(
                    "outer_core_b3restr",
                    &composed.tokenizer,
                    &ti_vocab,
                    &grammar,
                    &disallowed,
                    composed.ignore_canonical,
                    Some(&commit_core),
                    Some(&state_map_core),
                    Some((&composed.table.terminal_offsets, 0))
                ) {
                    let x_b3c = phase1_crossing_from(
                        "outer_core_b3restr",
                        &w_b3c.dwa,
                        &composed.table.terminal_offsets,
                        0,
                    );
                    let t_b3c = phase1_accepted_tokens(&x_b3c, &w_b3c.id_map, None);
                    phase1_token_diff_report(
                        "outer_core_b3_vs_ti",
                        &t_b3c,
                        &ti_cross_core,
                        &vocab,
                    );
                    let f_b3c = phase1_accepted_tokens(&w_b3c.dwa, &w_b3c.id_map, None);
                    phase1_token_diff_report(
                        "outer_core_b3_full_vs_ti",
                        &f_b3c,
                        &ti_full_core,
                        &vocab,
                    );
                }
            }

            // DynamicDirect cross-checks at a core position and a dispatch position.
            let run_dyn =
                std::env::var("PHASE1_DYNCHECK").map(|value| value != "0").unwrap_or(true);
            if run_dyn {
            let dyn_started = Instant::now();
            let dyn_placeholder = terminal(&core, "PROGRAMMATIC_TOOL_SUFFIX");
            let dyn_composed = compose_constraints_owned_parent_segmented(
                core.clone(),
                &[CompiledSubgrammarInput {
                    placeholder_terminal: dyn_placeholder,
                    additional_placeholder_terminals: &[],
                    constraint: &dispatch,
                }],
                &vocab,
                SegmentedBoundaryBackend::Dynamic,
            )
            .expect("dynamic compose")
            .constraint;
            eprintln!(
                "PHASE1 dyn_compose_ms={:.3}",
                dyn_started.elapsed().as_secs_f64() * 1000.0,
            );
            let dyn_spots: &[u32] = &[2358, 2313, 6226, 1287, 28937, 17289, 22715, 1237, 340, 16297];
            phase1_dynamic_crosscheck(
                "outer_core_pos",
                b"const x = tools",
                b"",
                &dyn_composed,
                &core,
                &dispatch,
                &vocab,
                (&wa.dwa, &wa.id_map, &commit_core),
                (&wa_dispatch.dwa, &wa_dispatch.id_map, &commit_dispatch),
                &composed.tokenizer_offsets,
                &ta_dispatch,
                &ti_core,
                dyn_spots,
                &nd_crossing_std,
            );
            phase1_dynamic_crosscheck(
                "outer_disp_pos",
                b"const x = tools.tool_5({",
                b".tool_5({",
                &dyn_composed,
                &core,
                &dispatch,
                &vocab,
                (&wa.dwa, &wa.id_map, &commit_core),
                (&wa_dispatch.dwa, &wa_dispatch.id_map, &commit_dispatch),
                &composed.tokenizer_offsets,
                &ta_dispatch,
                &ti_dispatch,
                dyn_spots,
                &nd_crossing_std,
            );
            }
        }

        // ---------- dispatch composition: parent + 10 schemas, i = schema 0 ----------
        if only.is_empty() || only == "schema" {
            let parent_source = phase1_dispatcher_literal_parent_source();
            let parent_started = Instant::now();
            let parent =
                Constraint::compile(crate::Grammar::glrm(&parent_source), &vocab).expect("compile parent");
            eprintln!(
                "PHASE1 dispatcher_parent_compile_ms={:.3}",
                parent_started.elapsed().as_secs_f64() * 1000.0,
            );
            assert!(
                parent.tokenizer.terminal_exprs().is_some(),
                "fresh parent must carry terminal exprs inline",
            );
            let mut schemas = Vec::with_capacity(PHASE1_SELECTED10_SHORT.len());
            for (index, short) in PHASE1_SELECTED10_SHORT.iter().enumerate() {
                let mut schema = phase1_load(&root, &format!("schema-{index:02}-{short}.bin"));
                phase1_restore(&mut schema, short);
                schemas.push(schema);
            }
            let child_inputs: Vec<CompiledSubgrammarInput<'_>> = schemas
                .iter()
                .enumerate()
                .map(|(index, schema)| CompiledSubgrammarInput {
                    placeholder_terminal: terminal(&parent, &format!("TOOL_ARGS_SLOT_{index}")),
                    additional_placeholder_terminals: &[],
                    constraint: schema,
                })
                .collect();
            let composed = phase1_compose_low_level("dispatch", &parent, &child_inputs);
            let grammar = phase1_grammar(&composed.table.table, &composed.terminal_names);
            let disallowed = crate::compiler::pipeline::compute_disallowed_follows(&grammar);
            // Component 1 = schema 0 (component 0 is the dispatcher parent).
            let commit_schema0 = phase1_commit_range(
                &composed.tokenizer_offsets,
                schemas[0].tokenizer.num_states(),
                1,
                composed.tokenizer.num_states() as usize,
            );

            let dump_path = format!("{dump_dir}/minbound-dispatch-tokens.txt");
            let (oracle_tokens_all, oracle_schema0, b_mapped): (
                BTreeSet<u32>,
                Option<BTreeSet<u32>>,
                Option<MappedArtifact<DWA>>,
            ) = if skip_oracle {
                let text = fs::read_to_string(&dump_path)
                    .expect("dispatch oracle dump missing; run once without PHASE1_SKIP_ORACLE");
                (
                    text.split_whitespace().map(|value| value.parse::<u32>().unwrap()).collect(),
                    None,
                    None,
                )
            } else {
                let mut components: Vec<(
                    &Tokenizer,
                    &crate::compiler::glr::table::GLRTable,
                    &[String],
                    Option<u32>,
                )> = Vec::with_capacity(schemas.len() + 1);
                components.push((
                    &parent.tokenizer,
                    &parent.table,
                    &parent.terminal_display_names[..],
                    parent.ignore_terminal,
                ));
                for schema in &schemas {
                    components.push((
                        &schema.tokenizer,
                        &schema.table,
                        &schema.terminal_display_names[..],
                        schema.ignore_terminal,
                    ));
                }
                let oracle = phase1_oracle("dispatch", &composed, &components, &vocab);
                let text = oracle
                    .tokens_all
                    .iter()
                    .map(|token| token.to_string())
                    .collect::<Vec<_>>()
                    .join(" ");
                fs::write(&dump_path, format!("{text}\n")).expect("write dispatch oracle dump");
                let keep = phase1_commit_internal(&oracle.id_map, &commit_schema0);
                let restricted = phase1_accepted_tokens(&oracle.residual, &oracle.id_map, Some(&keep));
                eprintln!(
                    "PHASE1 oracle_dispatch restricted_to_schema0={} all={}",
                    restricted.len(),
                    oracle.tokens_all.len(),
                );
                (oracle.tokens_all, Some(restricted), Some(oracle.b_mapped))
            };

            let ti_schema0_path = format!("{dump_dir}/phase1-ti-dispatch-schema0.txt");
            let ti_schema0: BTreeSet<u32> = if skip_oracle {
                let text = fs::read_to_string(&ti_schema0_path).expect("T dump missing");
                text.split_whitespace().map(|value| value.parse::<u32>().unwrap()).collect()
            } else {
                let ti = phase1_compute_ti("dispatch_schema0", &schemas[0].tokenizer, &vocab);
                let text =
                    ti.iter().map(|token| token.to_string()).collect::<Vec<_>>().join(" ");
                fs::write(&ti_schema0_path, format!("{text}\n")).expect("write T dump");
                ti
            };
            let ti_vocab = Vocab::new(
                vocab
                    .entries_map()
                    .iter()
                    .filter(|(token, _)| ti_schema0.contains(token))
                    .map(|(&token, bytes)| (token, bytes.clone()))
                    .collect(),
            );

            if let Some(w0) =
                phase1_run_walk("dispatch_all_full", &composed.tokenizer, &vocab, &grammar, &disallowed, composed.ignore_canonical, None, None, None)
                && let Some(b) = b_mapped.as_ref()
            {
                phase1_validate_unified_walk("dispatch", &w0.dwa, &w0.id_map, b);
            }
            let wa = phase1_run_walk(
                "dispatch_schema0_full",
                &composed.tokenizer,
                &vocab,
                &grammar,
                &disallowed,
                composed.ignore_canonical,
                Some(&commit_schema0),
            None,
                Some((&composed.table.terminal_offsets, 1))
            )
            .expect("schema0 full-vocab walk");
            let wb = phase1_run_walk(
                "dispatch_schema0_ti",
                &composed.tokenizer,
                &ti_vocab,
                &grammar,
                &disallowed,
                composed.ignore_canonical,
                Some(&commit_schema0),
            None,
                Some((&composed.table.terminal_offsets, 1))
            );
            let xa =
                phase1_crossing_from("dispatch_schema0_full", &wa.dwa, &composed.table.terminal_offsets, 1);
            let wa_total = phase1_accepted_tokens(&wa.dwa, &wa.id_map, None);
            eprintln!("PHASE1 walktokens_dispatch_schema0_full={}", wa_total.len());
            let ta = phase1_accepted_tokens(&xa, &wa.id_map, None);
            eprintln!(
                "PHASE1 tokens_dispatch_schema0_full={} paths={}",
                ta.len(),
                phase1_count_paths(&xa),
            );
            eprintln!(
                "PHASE1 ticheck_dispatch crossing_not_in_T={}",
                ta.difference(&ti_schema0).count(),
            );
            // Same minimizer-artifact proof as the outer section, for schema0.
            let owned_min_schema = minimize_owned(xa.clone());
            let fwd_schema = find_difference(&xa, &owned_min_schema)
                .expect("schema crossing minimizer-equivalence check failed");
            let bwd_schema = find_difference(&owned_min_schema, &xa)
                .expect("schema crossing minimizer-equivalence check failed");
            eprintln!(
                "PHASE1 nwaequiv_dispatch_schema0 walk_states={} walk_trans={} acyclic_states={} acyclic_trans={} owned_states={} owned_trans={} fwd_none={} bwd_none={}",
                wa.dwa.num_states(),
                wa.dwa.num_transitions(),
                xa.num_states(),
                xa.num_transitions(),
                owned_min_schema.num_states(),
                owned_min_schema.num_transitions(),
                fwd_schema.is_none(),
                bwd_schema.is_none(),
            );
            assert!(
                fwd_schema.is_none() && bwd_schema.is_none(),
                "schema DWA-filtered crossing language differs across minimizers",
            );
            let mut ti_full_schema = BTreeSet::new();
            let mut ti_cross_schema = BTreeSet::new();
            if let Some(wb) = wb {
                let xb = phase1_crossing_from(
                    "dispatch_schema0_ti",
                    &wb.dwa,
                    &composed.table.terminal_offsets,
                    1,
                );
                let tb = phase1_accepted_tokens(&xb, &wb.id_map, None);
                eprintln!(
                    "PHASE1 tokens_dispatch_schema0_ti={} paths={}",
                    tb.len(),
                    phase1_count_paths(&xb),
                );
                phase1_token_diff_report("dispatch_a_vs_b", &ta, &tb, &vocab);
                ti_full_schema = phase1_accepted_tokens(&wb.dwa, &wb.id_map, None);
                ti_cross_schema = tb;
            }
            if let Some(restricted) = oracle_schema0.as_ref() {
                phase1_token_diff_report("dispatch_schema0_vs_oracle_i", &ta, restricted, &vocab);
            }
            phase1_token_diff_report("dispatch_schema0_vs_oracle_all", &ta, &oracle_tokens_all, &vocab);
            phase1_parser_from_crossing(
                "dispatch_schema0_full",
                &xa,
                &composed.table.table,
                &grammar,
                &vocab,
                &wa.id_map,
            );

            let bypass =
                std::env::var("PHASE1_BYPASS").map(|value| value != "0").unwrap_or(true);
            let run_b1 =
                std::env::var("PHASE1_B1").map(|value| value != "0").unwrap_or(true);
            if bypass {
                if let Some((dwa_b2, id_b2)) = phase1_run_walk_replica(
                    "dispatch_schema0_b2reuse",
                    &composed.tokenizer,
                    &vocab,
                    &grammar,
                    &disallowed,
                    composed.ignore_canonical,
                    Some(&commit_schema0),
                    &wa.id_map,
                ) {
                    let x_b2 = phase1_crossing_from(
                        "dispatch_schema0_b2reuse",
                        &dwa_b2,
                        &composed.table.terminal_offsets,
                        1,
                    );
                    let t_b2 = phase1_accepted_tokens(&x_b2, &id_b2, None);
                    phase1_token_diff_report("dispatch_schema0_b2_vs_a", &t_b2, &ta, &vocab);
                    let f_b2 = phase1_accepted_tokens(&dwa_b2, &id_b2, None);
                    phase1_token_diff_report(
                        "dispatch_schema0_b2_full_vs_a",
                        &f_b2,
                        &wa_total,
                        &vocab,
                    );
                    eprintln!(
                        "PHASE1 b2check_dispatch_schema0 dwa_states={} vs_a={} dwa_trans={} vs_a={}",
                        dwa_b2.num_states(),
                        wa.dwa.num_states(),
                        dwa_b2.num_transitions(),
                        wa.dwa.num_transitions(),
                    );
                    phase1_parser_from_crossing(
                        "dispatch_schema0_b2reuse",
                        &x_b2,
                        &composed.table.table,
                        &grammar,
                        &vocab,
                        &id_b2,
                    );
                }
                if run_b1 {
                    let id_b1 = phase1_identity_id_map(&composed.tokenizer, &vocab);
                    if let Some((dwa_b1, id_b1c)) = phase1_run_walk_replica(
                        "dispatch_schema0_b1ident",
                        &composed.tokenizer,
                        &vocab,
                        &grammar,
                        &disallowed,
                        composed.ignore_canonical,
                        Some(&commit_schema0),
                        &id_b1,
                    ) {
                        let x_b1 = phase1_crossing_from(
                            "dispatch_schema0_b1ident",
                            &dwa_b1,
                            &composed.table.terminal_offsets,
                            1,
                        );
                        let t_b1 = phase1_accepted_tokens(&x_b1, &id_b1c, None);
                        phase1_token_diff_report("dispatch_schema0_b1_vs_a", &t_b1, &ta, &vocab);
                        let f_b1 = phase1_accepted_tokens(&dwa_b1, &id_b1c, None);
                        phase1_token_diff_report(
                            "dispatch_schema0_b1_full_vs_a",
                            &f_b1,
                            &wa_total,
                            &vocab,
                        );
                        phase1_parser_from_crossing(
                            "dispatch_schema0_b1ident",
                            &x_b1,
                            &composed.table.table,
                            &grammar,
                            &vocab,
                            &id_b1c,
                        );
                    }
                }
            }

            // b3: equivalence restricted to Commit_i states × TI vocab (the
            // §2.4.3 vehicle); must reproduce the TI walk exactly.
            let run_b3_schema =
                std::env::var("PHASE1_B3").map(|value| value != "0").unwrap_or(true);
            if run_b3_schema {
                let state_map_schema0 = phase1_restricted_state_map(
                    composed.tokenizer.num_states() as usize,
                    &commit_schema0,
                );
                if let Some(w_b3) = phase1_run_walk(
                    "dispatch_schema0_b3restr",
                    &composed.tokenizer,
                    &ti_vocab,
                    &grammar,
                    &disallowed,
                    composed.ignore_canonical,
                    Some(&commit_schema0),
                    Some(&state_map_schema0),
                    Some((&composed.table.terminal_offsets, 1))
                ) {
                    let x_b3 = phase1_crossing_from(
                        "dispatch_schema0_b3restr",
                        &w_b3.dwa,
                        &composed.table.terminal_offsets,
                        1,
                    );
                    let t_b3 = phase1_accepted_tokens(&x_b3, &w_b3.id_map, None);
                    phase1_token_diff_report(
                        "dispatch_schema0_b3_vs_ti",
                        &t_b3,
                        &ti_cross_schema,
                        &vocab,
                    );
                    let f_b3 = phase1_accepted_tokens(&w_b3.dwa, &w_b3.id_map, None);
                    phase1_token_diff_report(
                        "dispatch_schema0_b3_full_vs_ti",
                        &f_b3,
                        &ti_full_schema,
                        &vocab,
                    );
                    phase1_parser_from_crossing(
                        "dispatch_schema0_b3restr",
                        &x_b3,
                        &composed.table.table,
                        &grammar,
                        &vocab,
                        &w_b3.id_map,
                    );
                }
            }
        }

        eprintln!("PHASE1 done");
    }

    // ---- Phase 1 reset-semantics probe: WHY is core→dispatch empty? ----
    // No walks here: merged-tokenizer reset targets, byte traces of crossing
    // candidates from a CALL-able core state, and the composed table's allowed
    // cross-component terminal pairs. Fast (seconds).
    #[test]
    #[ignore]
    fn phase1_reset_semantics_selected10() {
        fn owner(tokenizer_offsets: &[u32], state: u32) -> String {
            if state == 0 {
                return "reset".to_string();
            }
            let component =
                tokenizer_offsets.partition_point(|&offset| offset <= state).saturating_sub(1);
            format!("c{component}")
        }

        fn term_owner(term_offsets: &[u32], terminal: u32) -> usize {
            term_offsets.partition_point(|&offset| offset <= terminal).saturating_sub(1)
        }

        fn short_name(names: &[String], id: u32) -> String {
            let name = names.get(id as usize).map(String::as_str).unwrap_or("?");
            if name.len() > 44 { format!("{}…", &name[..40]) } else { name.to_string() }
        }

        // Full byte trace: longest matches (with end states + owners) over the
        // whole token, the live end set, and the post-first-reset suffix rerun.
        fn trace(
            tag: &str,
            tokenizer: &Tokenizer,
            bytes: &[u8],
            start: u32,
            tokenizer_offsets: &[u32],
            term_offsets: &[u32],
            names: &[String],
        ) -> (Vec<(u32, usize)>, Vec<u32>) {
            let result = tokenizer.execute_from_state(bytes, start);
            let mut matches: Vec<(u32, usize, u32)> = result
                .matches
                .iter()
                .map(|matched| (matched.id, matched.width, matched.end_state))
                .collect();
            matches.sort();
            let mut match_text = Vec::new();
            for (id, width, end) in &matches {
                match_text.push(format!(
                    "t{id}[c{}]={}@{}→{}[{}]",
                    term_owner(term_offsets, *id),
                    short_name(names, *id),
                    width,
                    end,
                    owner(tokenizer_offsets, *end),
                ));
            }
            let mut ends: Vec<u32> = result.end_state.iter().copied().collect();
            ends.sort();
            let end_text: Vec<String> =
                ends.iter().map(|state| format!("{state}[{}]", owner(tokenizer_offsets, *state))).collect();
            eprintln!(
                "RST trace_{tag} bytes={:?} start={}[{}] matches=[{}] live=[{}]",
                String::from_utf8_lossy(bytes),
                start,
                owner(tokenizer_offsets, start),
                match_text.join(" "),
                end_text.join(" "),
            );
            // First reset strictly inside the token, if any: rerun the suffix
            // from the merged reset to show exactly where the reset goes.
            let post_ids: Vec<u32>;
            if let Some(&(_, width, _)) =
                matches.iter().filter(|(_, width, _)| *width < bytes.len()).min_by_key(|(_, width, _)| *width)
            {
                let suffix = &bytes[width..];
                let rerun = tokenizer.execute_from_state(suffix, 0);
                let mut post: Vec<String> = rerun
                    .matches
                    .iter()
                    .map(|matched| {
                        format!(
                            "t{}[c{}]={}@{}→{}[{}]",
                            matched.id,
                            term_owner(term_offsets, matched.id),
                            short_name(names, matched.id),
                            matched.width,
                            matched.end_state,
                            owner(tokenizer_offsets, matched.end_state),
                        )
                    })
                    .collect();
                post.sort();
                post_ids = rerun.matches.iter().map(|matched| matched.id).collect();
                let mut post_ends: Vec<u32> = rerun.end_state.iter().copied().collect();
                post_ends.sort();
                eprintln!(
                    "RST reset_{tag} at_width={width} suffix={:?} post_matches=[{}] post_live={:?}",
                    String::from_utf8_lossy(suffix),
                    post.join(" "),
                    post_ends,
                );
            } else {
                post_ids = Vec::new();
                eprintln!("RST reset_{tag} none_inside_token");
            }
            (matches.iter().map(|(id, width, _)| (*id, *width)).collect(), post_ids)
        }

        let root = std::env::var("PHASE1_DIR").unwrap_or_else(|_| {
            "/Users/isaacbreen/Projects2/temp/2026-09/glrmask-selected10-cache-v29".to_string()
        });
        let root = std::path::Path::new(&root).to_path_buf();
        let vocab_path = std::env::var("PHASE1_VOCAB")
            .unwrap_or_else(|_| root.join("vocab_dump.bin").to_string_lossy().into_owned());
        let vocab = load_vocab(&vocab_path);
        let mut core = phase1_load(&root, "core.bin");
        let dispatch_name = std::env::var("PHASE1_DISPATCH")
            .unwrap_or_else(|_| "dispatch-literal.bin".to_string());
        let mut dispatch = phase1_load(&root, &dispatch_name);
        phase1_restore(&mut core, "core");
        phase1_restore(&mut dispatch, "dispatch");
        let placeholder = terminal(&core, "PROGRAMMATIC_TOOL_SUFFIX");
        let child = CompiledSubgrammarInput {
            placeholder_terminal: placeholder,
            additional_placeholder_terminals: &[],
            constraint: &dispatch,
        };
        let composed = phase1_compose_low_level("outer", &core, &[child]);
        let grammar = phase1_grammar(&composed.table.table, &composed.terminal_names);
        let disallowed = crate::compiler::pipeline::compute_disallowed_follows(&grammar);
        let allowed = |first: u32, second: u32| -> bool {
            !disallowed.get(&first).is_some_and(|set| set.contains(second as usize))
        };

        // (a) Reset targets in the merged tokenizer.
        let mut reset_states: Vec<u32> =
            composed.tokenizer.deterministic_reset_states().iter().copied().collect();
        reset_states.sort();
        let mut closure: Vec<u32> =
            composed.tokenizer.execute_from_state_end_only(&[], 0).into_vec();
        closure.sort();
        eprintln!(
            "RST reset_states={reset_states:?} closure_0={closure:?} dispatch_roots={:?}",
            composed.tokenizer.deterministic_dispatch_roots().map(|roots| roots.to_vec()),
        );
        for state in &closure {
            eprintln!("RST closure_member state={state} owner={}", owner(&composed.tokenizer_offsets, *state));
        }

        // (a2) Concrete single-byte transition targets from the merged reset:
        // where a post-reset byte lands, per owning component.
        for byte in [b'{', b';', b'.', b'(', b')', b',', b' ', b'"', b'\n', b'\t', b'\r'] {
            let stepped = composed.tokenizer.execute_from_state(&[byte], 0);
            let mut ends: Vec<u32> = stepped.end_state.iter().copied().collect();
            ends.sort();
            let end_text: Vec<String> = ends
                .iter()
                .map(|state| format!("{state}[{}]", owner(&composed.tokenizer_offsets, *state)))
                .collect();
            let mut stepped_matches: Vec<String> = stepped
                .matches
                .iter()
                .map(|matched| {
                    format!(
                        "t{}[c{}]@{}",
                        matched.id,
                        term_owner(&composed.table.terminal_offsets, matched.id),
                        matched.width,
                    )
                })
                .collect();
            stepped_matches.sort();
            eprintln!(
                "RST fanout byte={:?} live=[{}] matches=[{}]",
                byte as char,
                end_text.join(" "),
                stepped_matches.join(" "),
            );
        }

        // (b) CALL-able core state: live lexer states of a core-alone commit
        // of `const x = tools` (runtime resets included), relabelled to merged.
        let prefix = b"const x = tools";
        let mut core_state = core.start();
        let commit_result = core_state.commit_bytes(prefix);
        let mut call_states: Vec<u32> = core_state
            .state
            .entries
            .iter()
            .map(|(local, _)| composed.tokenizer_offsets[0] + local)
            .collect();
        call_states.sort();
        call_states.dedup();
        call_states.truncate(3);
        eprintln!(
            "RST core_prefix commit_ok={} rejected={} live_local={:?} call_states={call_states:?}",
            commit_result.is_ok(),
            core_state.is_rejected(),
            core_state.state.entries.iter().map(|(local, _)| local).collect::<Vec<_>>(),
        );
        let sentinel_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            core_state.commit_token(128_300)
        }));
        eprintln!(
            "RST core_prefix sentinel_128300_commit={:?}",
            sentinel_result.as_ref().map(|result| result.is_ok()),
        );

        // (c) Vocabulary search for crossing-shaped candidates.
        let patterns: &[(&str, fn(&[u8]) -> bool)] = &[
            ("tools_prefix", |bytes| bytes.starts_with(b"tools")),
            ("dot_tool", |bytes| bytes.windows(6).any(|window| window == b".tool_")),
            ("comma_lbrace", |bytes| bytes.starts_with(b",{")),
            ("comma_space_lbrace", |bytes| bytes.starts_with(b", {")),
            ("lparen_lbrace", |bytes| bytes.starts_with(b"({")),
            ("lparen_space_lbrace", |bytes| bytes.starts_with(b"( {")),
            ("semi_lbrace", |bytes| bytes.starts_with(b";{")),
            ("rparen_lbrace", |bytes| bytes.starts_with(b"){")),
        ];
        for (name, predicate) in patterns {
            let mut hits: Vec<(u32, String)> = vocab
                .entries_map()
                .iter()
                .filter(|(_, bytes)| predicate(bytes))
                .map(|(&token, bytes)| (token, format!("{:?}", String::from_utf8_lossy(bytes))))
                .collect();
            hits.sort();
            eprintln!(
                "RST vocab_{name} count={} examples={:?}",
                hits.len(),
                hits.iter().take(10).collect::<Vec<_>>(),
            );
        }

        // (d) Traces of candidates from the CALL-able core state(s).
        let mut traced = 0usize;
        for (&token, bytes) in vocab.entries_map().iter() {
            let interesting = bytes.starts_with(b",{")
                || bytes.starts_with(b"({")
                || bytes.starts_with(b"( {")
                || bytes.starts_with(b";{")
                || bytes.starts_with(b"){")
                || bytes.starts_with(b"tools.");
            if !interesting || traced >= 8 {
                continue;
            }
            for (index, &state) in call_states.iter().enumerate() {
                let (matches, post_ids) = trace(
                    &format!("cand{traced}_t{token}_s{index}"),
                    &composed.tokenizer,
                    bytes,
                    state,
                    &composed.tokenizer_offsets,
                    &composed.table.terminal_offsets,
                    &composed.terminal_names,
                );
                // Cross-component (reset-match → post-match) pairs: table verdict?
                for (reset_id, width) in &matches {
                    if *width >= bytes.len() {
                        continue;
                    }
                    for post_id in &post_ids {
                        let owner_a = term_owner(&composed.table.terminal_offsets, *reset_id);
                        let owner_b = term_owner(&composed.table.terminal_offsets, *post_id);
                        if owner_a != owner_b {
                            eprintln!(
                                "RST paircheck_t{token} t{reset_id}[c{owner_a}]={} → t{post_id}[c{owner_b}]={} allowed={}",
                                short_name(&composed.terminal_names, *reset_id),
                                short_name(&composed.terminal_names, *post_id),
                                allowed(*reset_id, *post_id),
                            );
                        }
                    }
                }
            }
            traced += 1;
        }
        // Union-violation tokens for contrast: t280 `;\n`, t2313 `({\n`.
        for token in [280u32, 2313] {
            if let Some(bytes) = vocab.entries_map().get(&token) {
                for (index, &state) in call_states.iter().enumerate() {
                    trace(
                        &format!("viol_t{token}_s{index}"),
                        &composed.tokenizer,
                        bytes,
                        state,
                        &composed.tokenizer_offsets,
                        &composed.table.terminal_offsets,
                        &composed.terminal_names,
                    );
                }
            }
        }
        // The dispatch→core direction for contrast: oracle token 1237 `");`.
        if let Some(bytes) = vocab.entries_map().get(&1237) {
            trace(
                "oracle1237_from_dispatch_init",
                &composed.tokenizer,
                bytes,
                composed.tokenizer_offsets[1],
                &composed.tokenizer_offsets,
                &composed.table.terminal_offsets,
                &composed.terminal_names,
            );
        }

        // (e) Which cross-component terminal pairs does the composed table allow?
        let (core_terms, disp_terms) =
            (composed.table.terminal_offsets[0], composed.table.terminal_offsets[1]);
        let num_terms = composed.table.table.num_terminals;
        let mut core_to_disp = Vec::new();
        for first in core_terms..disp_terms {
            for second in disp_terms..num_terms {
                if allowed(first, second) {
                    core_to_disp.push((first, second));
                }
            }
        }
        let mut disp_to_core = Vec::new();
        for first in disp_terms..num_terms {
            for second in core_terms..disp_terms {
                if allowed(first, second) {
                    disp_to_core.push((first, second));
                }
            }
        }
        eprintln!(
            "RST pairs core_to_disp_allowed={} disp_to_core_allowed={}",
            core_to_disp.len(),
            disp_to_core.len(),
        );
        for (first, second) in core_to_disp.iter().take(20) {
            eprintln!(
                "RST pair_c2d t{first}={} → t{second}={}",
                short_name(&composed.terminal_names, *first),
                short_name(&composed.terminal_names, *second),
            );
        }
        for (first, second) in disp_to_core.iter().take(20) {
            eprintln!(
                "RST pair_d2c t{first}={} → t{second}={}",
                short_name(&composed.terminal_names, *first),
                short_name(&composed.terminal_names, *second),
            );
        }
        // The anchor pair: core `tools` → dispatch `.tool_0(`.
        let tools_terminals: Vec<u32> = (core_terms..disp_terms)
            .filter(|terminal| {
                composed.terminal_names.get(*terminal as usize).is_some_and(|name| name.contains("tools"))
            })
            .collect();
        let tool0_terminals: Vec<u32> = (disp_terms..num_terms)
            .filter(|terminal| {
                composed.terminal_names.get(*terminal as usize).is_some_and(|name| name.contains("tool_0"))
            })
            .collect();
        eprintln!(
            "RST anchor tools_terminals={tools_terminals:?} tool0_terminals={tool0_terminals:?}"
        );
        for first in &tools_terminals {
            for second in &tool0_terminals {
                eprintln!(
                    "RST anchor_pair t{first}={} → t{second}={} allowed={}",
                    short_name(&composed.terminal_names, *first),
                    short_name(&composed.terminal_names, *second),
                    allowed(*first, *second),
                );
            }
        }

        // (f) DynamicDirect verdicts on crossing-shaped candidates at a
        // CALL-capable core position and inside dispatch.
        let dyn_composed = compose_constraints_owned_parent_segmented(
            core.clone(),
            &[CompiledSubgrammarInput {
                placeholder_terminal: placeholder,
                additional_placeholder_terminals: &[],
                constraint: &dispatch,
            }],
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("dynamic compose")
        .constraint;
        for prefix in [b"const x = tools".as_slice(), b"const x = tools.tool_5({".as_slice()] {
            let mut dyn_state = dyn_composed.start();
            dyn_state.commit_bytes(prefix).expect("dynamic commit prefix");
            let mask = dyn_state.mask();
            let admits = |token: u32| -> bool {
                mask.get((token / 32) as usize)
                    .is_some_and(|word| word & (1 << (token % 32)) != 0)
            };
            for token in [2358u32, 2313, 6226, 1287, 28937, 17289, 22715, 1237, 340, 16297] {
                let bytes =
                    vocab.entries_map().get(&token).map(Vec::as_slice).unwrap_or(&[]);
                eprintln!(
                    "RST dynmask prefix={:?} t{token}={:?} admits={}",
                    String::from_utf8_lossy(prefix),
                    String::from_utf8_lossy(bytes),
                    admits(token),
                );
            }
        }
        eprintln!("RST done");
    }
