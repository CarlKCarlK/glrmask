    use super::super::dfa::DFA;
    use super::super::Lexer;
    use super::{build_regex, build_regex_local_small_product, build_regex_monolithic, build_regex_partitioned_with_adaptive};
use super::partition::{try_product_union_components};
    use super::component::{compile_product_component_dfa, compile_product_component_dfa_direct};
    use super::factor::factor_regex_expr;
    use crate::automata::lexer::ast::Expr;
    use crate::automata::lexer::regex::parse_regex;
    use crate::automata::lexer::tokenizer::Tokenizer;
    use crate::ds::bitset::BitSet;
    use crate::ds::u8set::U8Set;
    use crate::Vocab;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};
    use std::sync::Arc;

    fn byte_expr(byte: u8) -> Expr {
        Expr::U8Seq(vec![byte])
    }

    fn byte_choice(bytes: &[u8]) -> Expr {
        Expr::Choice(bytes.iter().copied().map(byte_expr).collect())
    }

    #[test]
    fn bounded_suffix_preflight_finds_expensive_nested_component() {
        let inner = Expr::Seq(vec![
            byte_expr(b'<'),
            Expr::Repeat {
                expr: Box::new(byte_expr(b'a')),
                min: 0,
                max: Some(20_000),
            },
            byte_expr(b'>'),
        ]);
        let own = super::mapping::direct_bounded_suffix_state_count_estimate(&inner)
            .expect("direct bounded suffix should have an exact state estimate");
        assert!(own > 10_000);

        let wrapped = Expr::Intersect {
            expr: Box::new(Expr::Choice(vec![byte_expr(b'x'), inner.clone()])),
            intersect: Box::new(Expr::Repeat {
                expr: Box::new(byte_expr(b'a')),
                min: 0,
                max: None,
            }),
        };
        assert_eq!(super::mapping::direct_bounded_suffix_state_count_estimate(&wrapped), None);
        assert_eq!(super::mapping::max_direct_bounded_suffix_state_count_estimate(&wrapped), Some(own));
    }

    #[test]
    fn lazy_zero_min_repeat_suffix_accepts_last_non_sentinel_bound() {
        let expr_for = |max| {
            Expr::Seq(vec![
                byte_expr(b'['),
                Expr::Repeat {
                    expr: Box::new(byte_expr(b'a')),
                    min: 0,
                    max: Some(max),
                },
                byte_expr(b'z'),
            ])
        };

        assert!(super::repeat_suffix::LazyZeroMinRepeatSuffixComponent::from_expr(
            &expr_for((u32::MAX - 1) as usize),
        )
        .is_some());
        assert!(super::repeat_suffix::LazyZeroMinRepeatSuffixComponent::from_expr(
            &expr_for(u32::MAX as usize),
        )
        .is_none());
    }

    #[test]
    fn lazy_zero_min_repeat_suffix_accepts_empty_literal_prefix() {
        let expr = Expr::Seq(vec![
            Expr::Repeat {
                expr: Box::new(byte_expr(b'a')),
                min: 0,
                max: Some(super::virtual_repeat::VIRTUAL_BINARY_REPEAT_MIN_BOUND),
            },
            Expr::U8Class(U8Set::from_bytes(b"bc")),
        ]);
        let mut component = super::repeat_suffix::LazyZeroMinRepeatSuffixComponent::from_expr(&expr)
            .expect("zero-prefix lazy repeat+suffix component");
        assert_eq!(component.prefix_len(), 0);
        let start = component.start_state();
        assert!(!component.is_accepting(start));
        assert!(component.has_future(start));
        let suffix_target = component
            .step_uncached(start, b'b')
            .expect("zero copies may enter the suffix immediately");
        assert!(component.is_accepting(suffix_target));
        let trace = super::mapping::zero_min_repeat_suffix_component_trace(&expr)
            .expect("zero-prefix structural trace must rebuild from the expression");
        assert_eq!(trace.prefix_len, 0);
    }

    fn terminal_matches(expr: Expr, input: &[u8]) -> bool {
        let regex = build_regex(std::slice::from_ref(&expr));
        let tokenizer = Tokenizer {
            dfa: regex.dfa,
            num_terminals: 1,
            packed_runtime_transitions: None,
            packed_runtime_transition_segments: Arc::from([]),
            compressed_transition_segments: Arc::from([]),
            packed_runtime_metadata: None,
            packed_runtime_metadata_segments: Arc::from([]),
            packed_compressed_transition_segments: Arc::from([]),
            virtual_unit_repeat: None,
            virtual_repeat_intersections: Vec::new(),
            virtual_residuals: Vec::new(),
            exprs: Some(Arc::from(vec![expr].into_boxed_slice())),
            terminal_residual_coordinates: None,
            singleton_epsilon_closures: std::sync::OnceLock::new(),
            matched_terminals_cache: std::sync::OnceLock::new(),
            initial_byte_frontiers: std::sync::OnceLock::new(),
            all_self_loop_bytes_cache: std::sync::OnceLock::new(),
            transition_count_cache: std::sync::OnceLock::new(),
            forced_minimized_state_count_cache: std::sync::OnceLock::new(),
            scalar_deterministic_dispatch_cache: std::sync::OnceLock::new(),
            sorted_dispatch_roots_cache: std::sync::OnceLock::new(),
            state_first_bytes_cache: std::sync::OnceLock::new(),
        };
        let exec = tokenizer.execute_from_state(input, tokenizer.initial_state());
        exec.matches
            .iter()
            .any(|matched| matched.id == 0 && matched.width == input.len())
    }

    fn dfa_accepts(dfa: &DFA, input: &[u8]) -> bool {
        let mut state = 0u32;
        for &byte in input {
            let Some(next) = dfa.step(state, byte) else {
                return false;
            };
            state = next;
        }
        dfa.finalizers(state).contains(0)
    }

    fn enumerate_inputs(alphabet: &[u8], max_len: usize) -> Vec<Vec<u8>> {
        fn extend(out: &mut Vec<Vec<u8>>, prefix: &mut Vec<u8>, alphabet: &[u8], max_len: usize) {
            out.push(prefix.clone());
            if prefix.len() == max_len {
                return;
            }
            for &byte in alphabet {
                prefix.push(byte);
                extend(out, prefix, alphabet, max_len);
                prefix.pop();
            }
        }

        let mut out = Vec::new();
        extend(&mut out, &mut Vec::new(), alphabet, max_len);
        out
    }

    fn assert_dfa_observation_equivalent(left: &DFA, right: &DFA) {
        let mut seen = HashSet::new();
        let mut queue = VecDeque::from([(Some(0u32), Some(0u32))]);
        while let Some((left_state, right_state)) = queue.pop_front() {
            if !seen.insert((left_state, right_state)) {
                continue;
            }
            let left_accepting = left_state
                .is_some_and(|state| !left.finalizers(state).is_empty());
            let right_accepting = right_state
                .is_some_and(|state| !right.finalizers(state).is_empty());
            assert_eq!(left_accepting, right_accepting, "acceptance mismatch at {left_state:?}/{right_state:?}");

            let left_future = left_state
                .is_some_and(|state| left.possible_future_group_ids(state).contains(0));
            let right_future = right_state
                .is_some_and(|state| right.possible_future_group_ids(state).contains(0));
            assert_eq!(left_future, right_future, "future mismatch at {left_state:?}/{right_state:?}");

            for byte in 0u8..=255 {
                let next = (
                    left_state.and_then(|state| left.step(state, byte)),
                    right_state.and_then(|state| right.step(state, byte)),
                );
                if next != (None, None) && !seen.contains(&next) {
                    queue.push_back(next);
                }
            }
        }
    }

    #[test]
    fn unique_dense_exclusion_skips_atomic_rhs_languages() {
        assert!(!super::plan::unique_dense_exclusion_rhs_is_compound(&Expr::U8Seq(
            b"literal".to_vec()
        )));
        assert!(!super::plan::unique_dense_exclusion_rhs_is_compound(&Expr::U8Class(
            U8Set::from_bytes(b"ab")
        )));
        assert!(!super::plan::unique_dense_exclusion_rhs_is_compound(&Expr::Epsilon));
        let dfa = super::nfa::compile_expr_to_dfa(&Expr::U8Seq(b"dfa".to_vec()));
        assert!(!super::plan::unique_dense_exclusion_rhs_is_compound(&Expr::Dfa(
            Arc::new(dfa)
        )));

        assert!(super::plan::unique_dense_exclusion_rhs_is_compound(&Expr::Choice(vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"b".to_vec()),
        ])));
        assert!(super::plan::unique_dense_exclusion_rhs_is_compound(&Expr::Seq(vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"b".to_vec()),
        ])));
    }

    #[test]
    fn dense_binary_exclusion_matches_general_exclusion() {
        let left_expr = Expr::Choice(vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"ab".to_vec()),
            Expr::U8Seq(b"ac".to_vec()),
            Expr::U8Seq(b"ba".to_vec()),
            Expr::Seq(vec![
                Expr::U8Seq(b"c".to_vec()),
                Expr::Repeat {
                    expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"ab"))),
                    min: 0,
                    max: Some(2),
                },
            ]),
        ]);
        let right_expr = Expr::Choice(vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"ab".to_vec()),
            Expr::U8Seq(b"c".to_vec()),
            Expr::U8Seq(b"caa".to_vec()),
        ]);
        let expr = Expr::Exclude {
            expr: Box::new(left_expr.clone()),
            exclude: Box::new(right_expr.clone()),
        };
        let expected = compile_product_component_dfa(&expr);
        let left = super::nfa::compile_expr_to_dfa(&left_expr);
        let right = super::nfa::compile_expr_to_dfa(&right_expr);
        let direct = super::product::build_dense_binary_exclusion_dfa(
            &left,
            &right,
            super::nfa::expr_u8set(&expr),
        )
        .expect("dense binary exclusion should apply");

        assert_dfa_observation_equivalent(&expected, &direct);
        for input in enumerate_inputs(b"abc", 4) {
            assert_eq!(
                dfa_accepts(&expected, &input),
                dfa_accepts(&direct, &input),
                "language mismatch for {input:?}",
            );
        }
    }

    fn expression_graph_symbols() -> Vec<Expr> {
        vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"b".to_vec()),
            Expr::Choice(vec![Expr::Epsilon, Expr::U8Seq(b"c".to_vec())]),
            Expr::Choice(vec![Expr::U8Seq(b"a".to_vec()), Expr::U8Seq(b"ab".to_vec())]),
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"d".to_vec())),
                min: 0,
                max: Some(2),
            },
        ]
    }

    #[test]
    fn local_small_product_compilation_matches_default_product() {
        let expressions = vec![
            Expr::U8Seq(b"alpha".to_vec()),
            Expr::Seq(vec![
                Expr::U8Seq(b"b".to_vec()),
                Expr::Repeat {
                    expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"cd"))),
                    min: 1,
                    max: Some(4),
                },
            ]),
            Expr::Choice(vec![
                Expr::U8Seq(b"cat".to_vec()),
                Expr::U8Seq(b"car".to_vec()),
            ]),
            Expr::Repeat {
                expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"xy"))),
                min: 0,
                max: Some(3),
            },
        ];
        let ordinary = build_regex(&expressions).dfa;
        let local = build_regex_local_small_product(&expressions).dfa;
        assert_dfa_observation_equivalent(&ordinary, &local);
    }

    #[test]
    fn grouped_expression_graph_matches_byte_nfa_with_nullable_overlap_and_cycles() {
        use crate::automata::unweighted_u32::nfa::NFA as GraphNfa;

        let mut graph = GraphNfa::new_empty();
        for _ in 0..7 {
            graph.add_state();
        }
        graph.start_states = vec![0, 1];
        graph.add_epsilon(0, 2);
        graph.add_epsilon(2, 0);
        graph.add_transition(0, 0, 3);
        graph.add_transition(1, 3, 3);
        graph.add_transition(2, 2, 4);
        graph.add_transition(3, 4, 3);
        graph.add_transition(3, 2, 5);
        graph.add_transition(3, 2, 6);
        graph.add_epsilon(4, 3);
        graph.add_transition(5, 1, 6);
        graph.set_accepting(6);

        let symbols = expression_graph_symbols();
        let grouped = super::expression_graph::try_compile_expression_labeled_nfa_direct(
            &graph,
            &symbols,
            true,
        )
        .unwrap()
        .expect("forced direct compiler should select the graph");
        let reference = super::expression_graph::compile_expression_labeled_nfa_via_byte_nfa(&graph, &symbols)
            .expect("byte-NFA reference should compile");
        assert_dfa_observation_equivalent(&grouped, &reference);
    }

    #[test]
    fn grouped_expression_graph_seeded_differential() {
        use crate::automata::unweighted_u32::nfa::NFA as GraphNfa;

        let symbols = expression_graph_symbols();
        let mut rng = StdRng::seed_from_u64(0xD1EC_7E67_2026_0731);
        for case in 0..48 {
            let state_count = rng.gen_range(2..=8);
            let mut graph = GraphNfa::new_empty();
            for _ in 0..state_count {
                graph.add_state();
            }
            graph.start_states.push(rng.gen_range(0..state_count) as u32);
            if rng.gen_bool(0.35) {
                let second = rng.gen_range(0..state_count) as u32;
                if !graph.start_states.contains(&second) {
                    graph.start_states.push(second);
                }
            }
            graph.set_accepting(rng.gen_range(0..state_count) as u32);
            for source in 0..state_count as u32 {
                if rng.gen_bool(0.45) {
                    graph.add_epsilon(source, rng.gen_range(0..state_count) as u32);
                }
                for _ in 0..rng.gen_range(1..=3) {
                    graph.add_transition(
                        source,
                        rng.gen_range(0..symbols.len()) as i32,
                        rng.gen_range(0..state_count) as u32,
                    );
                }
            }

            let grouped = super::expression_graph::try_compile_expression_labeled_nfa_direct(
                &graph,
                &symbols,
                true,
            )
            .unwrap()
            .expect("forced direct compiler should select the graph");
            let reference = super::expression_graph::compile_expression_labeled_nfa_via_byte_nfa(&graph, &symbols)
                .expect("byte-NFA reference should compile");
            assert_dfa_observation_equivalent(&grouped, &reference);
            assert_eq!(
                grouped.num_states(),
                reference.num_states(),
                "minimal state count differs in case {case}",
            );
        }
    }

    fn component_pair_for_map_test(
        full_states: usize,
        synthesized_states: usize,
        full_to_synthesized: Vec<u32>,
        protected_residual: bool,
    ) -> super::partition::LexerComponentPair {
        super::partition::LexerComponentPair {
            terminal_ids: vec![0],
            synthesized: DFA::new(synthesized_states),
            full: super::deferred::DeferredDfa::Ready(DFA::new(full_states)),
            full_to_synthesized,
            protected_residual,
        }
    }

    #[test]
    fn partitioned_component_maps_compose_identity_and_protected_offsets() {
        let components = vec![
            component_pair_for_map_test(2, 2, vec![0, 1], false),
            component_pair_for_map_test(3, 2, vec![0, 1, 1], true),
        ];
        let map = super::partition::compose_partitioned_component_state_maps(&components, &[1, 3])
            .expect("complete component maps compose");

        assert_eq!(map, vec![0, 1, 2, 3, 4, 4]);
        assert_eq!(map.len(), 1 + 2 + 3);
    }

    #[test]
    fn partitioned_component_map_composition_rejects_incomplete_or_invalid_maps() {
        let incomplete = vec![component_pair_for_map_test(3, 2, vec![0, 1], true)];
        assert!(super::partition::compose_partitioned_component_state_maps(&incomplete, &[1]).is_none());

        let invalid_target = vec![component_pair_for_map_test(2, 2, vec![0, 2], true)];
        assert!(
            super::partition::compose_partitioned_component_state_maps(&invalid_target, &[1]).is_none()
        );

        let wrong_offset = vec![component_pair_for_map_test(2, 2, vec![0, 1], false)];
        assert!(super::partition::compose_partitioned_component_state_maps(&wrong_offset, &[2]).is_none());
    }

    #[test]
    fn rebuilt_mixed_product_expression_preserves_group_operation_semantics() {
        let components = vec![byte_choice(b"ab"), byte_expr(b'b'), byte_expr(b'a')];
        let exclusions = BTreeMap::from([(0, BTreeSet::from([1]))]);
        let intersections = BTreeMap::from([(0, BTreeSet::from([2]))]);
        let rebuilt = super::plan::rebuild_single_visible_group_expression(
            &components,
            &exclusions,
            &intersections,
        )
        .expect("single visible product reconstruction");

        for input in enumerate_inputs(b"ab", 2) {
            assert_eq!(terminal_matches(rebuilt.clone(), &input), input == b"a");
        }
    }

    #[test]
    fn vocabulary_repeat_horizon_cache_allows_nested_parallel_misses() {
        use rayon::prelude::*;

        let body = Arc::new(super::nfa::compile_expr_to_dfa(&Expr::Choice(vec![
            Expr::U8Seq(b"ab".to_vec()),
            Expr::U8Seq(b"aba".to_vec()),
        ])));
        let vocab = Arc::new(Vocab::new(
            (0..256)
                .map(|token| {
                    let length = token as usize % 32 + 1;
                    (token, b"ab".repeat(length))
                })
                .collect(),
        ));
        let cache = Arc::new(super::repeat_horizon::VocabularyRepeatHorizonCache::new());
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("test thread pool");
        let results = pool.install(|| {
            (0..16)
                .into_par_iter()
                .map(|_| cache.horizon_for_dfa(&body, &vocab))
                .collect::<Vec<_>>()
        });
        assert!(results.iter().all(|result| *result == results[0]));
    }

    #[test]
    fn vocabulary_repeat_horizon_is_suffix_closed_and_residual_aware() {
        let body = Expr::U8Seq(b"ab".to_vec());
        let vocab = Vocab::new(vec![(0, b"xabab".to_vec())]);

        // The token suffix "abab" crosses two body boundaries from the body
        // start state. The analysis must consider that suffix even though it is
        // not itself a vocabulary entry.
        assert_eq!(
            super::repeat_horizon::vocabulary_repeat_boundary_horizon(&body, &vocab),
            Some(2),
        );

        let vocab = Vocab::new(vec![(0, b"babab".to_vec())]);
        // Starting from the reachable residual after an initial 'a', this token
        // completes one pending copy and two further copies.
        assert_eq!(
            super::repeat_horizon::vocabulary_repeat_boundary_horizon(&body, &vocab),
            Some(3),
        );
    }

    #[test]
    fn vocabulary_repeat_horizon_counts_boundaries_before_token_exit() {
        let body = Expr::U8Seq(b"a".to_vec());
        let vocab = Vocab::new(vec![(0, b"aaaaX".to_vec())]);

        assert_eq!(
            super::repeat_horizon::vocabulary_repeat_boundary_horizon(&body, &vocab),
            Some(4),
        );
    }

    #[test]
    fn vocabulary_repeat_horizon_respects_dominance_not_path_count() {
        let body = parse_regex("a+", false);
        let vocab = Vocab::new(vec![(0, b"aaaaaaaa".to_vec())]);

        // Although there are paths that choose a repeat boundary after every
        // byte, the zero-boundary path at the same residual dominates all of
        // them. Consequently an upper repetition counter is unobservable for
        // this body language and the translation displacement is exactly zero.
        assert_eq!(
            super::repeat_horizon::vocabulary_repeat_boundary_horizon(&body, &vocab),
            Some(0),
        );
    }

    #[test]
    fn zero_min_repeat_transport_reserves_live_count_spread() {
        let ambiguous_body = Expr::Choice(vec![
            byte_expr(b'a'),
            Expr::U8Seq(b"aaaaaaaa".to_vec()),
        ]);
        let body_dfa = super::nfa::compile_expr_to_dfa(&ambiguous_body);
        let suffix_dfa = super::nfa::compile_expr_to_dfa(&byte_expr(b'b'));
        let trace = |max| {
            let built = super::repeat_suffix::build_zero_min_repeat_suffix_dominance_dfa_internal(
                &body_dfa,
                &suffix_dfa,
                max,
                true,
            )
            .expect("dominance trace");
            Arc::new(super::mapping::ZeroMinRepeatSuffixComponentTrace {
                dfa: Arc::new(built.dfa),
                prefix_len: 0,
                body_dfa: body_dfa.clone(),
                suffix_dfa: suffix_dfa.clone(),
                max,
                tail_states: built.states,
                tail_state_by_key: built.state_by_key,
            })
        };
        let full_trace = trace(40);
        let synthesized_trace = trace(10);
        let max_live_count_span = full_trace
            .tail_states
            .iter()
            .filter_map(|state| {
                let mut counts = state
                    .body_min_counts
                    .iter()
                    .copied()
                    .filter(|&count| count != u32::MAX);
                let first = counts.next()?;
                let (minimum, maximum) =
                    counts.fold((first, first), |(minimum, maximum), count| {
                        (minimum.min(count), maximum.max(count))
                    });
                Some(maximum - minimum)
            })
            .max()
            .unwrap_or(0);
        assert!(max_live_count_span > 1);

        let vocab = Vocab::new(vec![(0, b"aaaaaaaa".to_vec())]);
        let horizons = super::repeat_horizon::VocabularyRepeatHorizonCache::new();
        assert!(
            super::mapping::zero_min_repeat_suffix_state_map(
                &Expr::Epsilon,
                &Expr::Epsilon,
                full_trace.dfa.as_ref(),
                synthesized_trace.dfa.as_ref(),
                Some(Arc::clone(&full_trace)),
                Some(Arc::clone(&synthesized_trace)),
                vocab.max_token_byte_len(),
                Some(&vocab),
                Some(&horizons),
            )
            .is_none(),
            "the shortened repeat must fail closed when live count spread and one-token displacement exceed its upper-bound headroom",
        );
    }

    #[test]
    fn dense_binary_intersection_matches_component_conjunction_exhaustively() {
        let cases = vec![
            (
                Expr::Seq(vec![
                    Expr::Repeat {
                        expr: Box::new(byte_choice(b"ab")),
                        min: 1,
                        max: Some(4),
                    },
                    Expr::U8Seq(b"b".to_vec()),
                ]),
                Expr::Repeat {
                    expr: Box::new(Expr::U8Seq(b"ab".to_vec())),
                    min: 0,
                    max: Some(3),
                },
            ),
            (
                Expr::Repeat {
                    expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"ab "))),
                    min: 0,
                    max: Some(5),
                },
                Expr::Seq(vec![
                    Expr::Repeat {
                        expr: Box::new(byte_choice(b"ab")),
                        min: 0,
                        max: Some(3),
                    },
                    Expr::U8Seq(b" ".to_vec()),
                ]),
            ),
            (
                Expr::Choice(vec![
                    Expr::U8Seq(b"abba".to_vec()),
                    Expr::U8Seq(b"baab".to_vec()),
                    Expr::U8Seq(b" ".to_vec()),
                ]),
                Expr::Repeat {
                    expr: Box::new(byte_choice(b"ab ")),
                    min: 1,
                    max: Some(4),
                },
            ),
        ];
        let inputs = enumerate_inputs(b"ab ", 6);

        for (left_expr, right_expr) in cases {
            let components = vec![
                super::product::compile_product_component(&left_expr),
                super::product::compile_product_component(&right_expr),
            ];
            let left = components[0].partition_dfa().clone();
            let right = components[1].partition_dfa().clone();
            let (class_map, class_members) =
                super::product::compute_product_equivalence_classes(&components);
            let class_transitions =
                super::product::build_product_class_transitions(&components, &class_map);
            let (product, _, trace) = super::deferred::try_build_dense_binary_intersection_product(
                &components,
                &class_members,
                &class_transitions,
                true,
                false,
            )
            .expect("dense binary intersection");
            let trace = trace.expect("captured dense trace");
            assert_eq!(trace.state_tuples.len(), product.num_states());

            for input in &inputs {
                assert_eq!(
                    dfa_accepts(&product, input),
                    dfa_accepts(&left, input) && dfa_accepts(&right, input),
                    "left={left_expr:?} right={right_expr:?} input={input:?}",
                );
            }
        }
    }

    #[test]
    fn lazy_zero_min_repeat_intersection_matches_materialized_product() {
        let repeat_component = Expr::Seq(vec![
            Expr::U8Seq(b"[".to_vec()),
            Expr::Repeat {
                expr: Box::new(Expr::Choice(vec![
                    Expr::U8Seq(b"a".to_vec()),
                    Expr::U8Seq(b"ab".to_vec()),
                ])),
                min: 0,
                max: Some(1_024),
            },
            Expr::U8Seq(b"]".to_vec()),
        ]);
        let filter_component = Expr::Seq(vec![
            Expr::U8Seq(b"[".to_vec()),
            Expr::Repeat {
                expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"ab"))),
                min: 0,
                max: Some(7),
            },
            Expr::U8Seq(b"]".to_vec()),
        ]);
        let expression = Expr::Intersect {
            expr: Box::new(repeat_component),
            intersect: Box::new(filter_component),
        };

        let eager = super::compile_with_plan(super::plan::build_exclusion_compile_plan(
            std::slice::from_ref(&expression),
        ));
        let plan = super::plan::build_exclusion_compile_plan(std::slice::from_ref(&expression));
        let (lazy, trace) = super::deferred::try_compile_lazy_zero_min_repeat_intersection(&plan, false)
            .expect("eligible repeat intersection must compile lazily");
        assert_eq!(trace.state_tuples.len(), lazy.num_states());
        assert!(matches!(
            &trace.state_lookup,
            super::product::ProductStateLookup::Hash(_)
        ));

        for input in enumerate_inputs(b"[]abx", 9) {
            assert_eq!(
                dfa_state_observation(&lazy, 0, &input),
                dfa_state_observation(&eager, 0, &input),
                "lazy product differed for input {input:?}",
            );
        }
    }

    #[test]
    fn lazy_repeat_selector_prefers_the_unique_giant_operand() {
        let component = |max| {
            Expr::Seq(vec![
                Expr::Repeat {
                    expr: Box::new(byte_expr(b'a')),
                    min: 0,
                    max: Some(max),
                },
                byte_expr(b'b'),
            ])
        };

        let non_giant = component(super::repeat_suffix::LAZY_ZERO_MIN_REPEAT_SUFFIX_MIN_BOUND);
        let giant = component(super::virtual_repeat::VIRTUAL_BINARY_REPEAT_MIN_BOUND);
        let (index, _) = super::deferred::select_lazy_zero_min_repeat_suffix_component(&[
            non_giant.clone(),
            giant.clone(),
        ])
        .expect("the unique giant lazy component must be selected");
        assert_eq!(index, 1);

        let (index, _) = super::deferred::select_lazy_zero_min_repeat_suffix_component(&[
            giant.clone(),
            non_giant,
        ])
        .expect("operand reversal must still select the unique giant component");
        assert_eq!(index, 0);

        assert!(
            super::deferred::select_lazy_zero_min_repeat_suffix_component(&[
                giant,
                component(super::virtual_repeat::VIRTUAL_BINARY_REPEAT_MIN_BOUND + 1),
            ])
            .is_none(),
            "the one-lazy-component lane must not claim two giant coordinates",
        );
    }

    fn finite_dfa_with_partial_nonlive_x_cycle() -> DFA {
        // Language is exactly {"b"}. The x-edge enters a partial dead cycle
        // rather than the canonical 256-byte sink, so product construction
        // must use finalizer/future liveness rather than sink identity alone.
        let mut dfa = DFA::new(3);
        dfa.ensure_group_capacity(1);
        dfa.set_group_u8set(0, U8Set::from_bytes(b"bx"));
        dfa.set_transitions_from_sorted_entries(0, vec![(b'b', 1), (b'x', 2)]);
        dfa.set_transitions_from_sorted_entries(2, vec![(b'x', 2)]);

        let mut start_future = crate::ds::bitset::BitSet::new(1);
        start_future.set(0);
        dfa.overwrite_state_metadata(
            0,
            crate::ds::bitset::BitSet::new(1),
            start_future,
        );
        let mut accepting = crate::ds::bitset::BitSet::new(1);
        accepting.set(0);
        dfa.overwrite_state_metadata(
            1,
            accepting,
            crate::ds::bitset::BitSet::new(1),
        );
        dfa.overwrite_state_metadata(
            2,
            crate::ds::bitset::BitSet::new(1),
            crate::ds::bitset::BitSet::new(1),
        );
        dfa
    }

    #[test]
    fn lazy_giant_intersection_prunes_partial_nonlive_other_cycles() {
        let giant = Expr::Seq(vec![
            Expr::Repeat {
                expr: Box::new(byte_expr(b'x')),
                min: 0,
                max: Some(super::virtual_repeat::VIRTUAL_BINARY_REPEAT_MIN_BOUND),
            },
            byte_expr(b'b'),
        ]);
        let ordinary = Expr::Dfa(Arc::new(finite_dfa_with_partial_nonlive_x_cycle()));

        for reversed in [false, true] {
            let expression = if reversed {
                Expr::Intersect {
                    expr: Box::new(ordinary.clone()),
                    intersect: Box::new(giant.clone()),
                }
            } else {
                Expr::Intersect {
                    expr: Box::new(giant.clone()),
                    intersect: Box::new(ordinary.clone()),
                }
            };
            let eager = super::compile_with_plan(super::plan::build_exclusion_compile_plan(
                std::slice::from_ref(&expression),
            ));
            let plan = super::plan::build_exclusion_compile_plan(std::slice::from_ref(&expression));
            let (mut lazy, mut trace) =
                super::deferred::try_compile_lazy_zero_min_repeat_intersection(&plan, false)
                    .expect("partial dead cycle must not reject the exact lazy product");
            assert_eq!(lazy.num_states(), 2, "only start and accepted 'b' survive");
            assert!(lazy.step(0, b'x').is_none());
            assert!(matches!(
                &trace.state_lookup,
                super::product::ProductStateLookup::Hash(_)
            ));

            for input in [b"".as_slice(), b"b", b"x", b"xx", b"xb"] {
                let semantic_observation = |dfa: &DFA| {
                    let (_, accepting, future) = dfa_state_observation(dfa, 0, input);
                    (accepting, future)
                };
                assert_eq!(
                    semantic_observation(&lazy),
                    semantic_observation(&eager),
                    "lazy product differed for reversed={reversed} input={input:?}",
                );
            }

            // Retained structural traces can be augmented from seed tuples.
            // Seed an impossible residual containing the ordinary dead-cycle
            // state and verify augmentation cannot turn it into a giant-only
            // x-walk after the ordinary coordinate is pruned.
            let ordinary_index = if reversed { 0usize } else { 1usize };
            let mut seed = super::product::ProductStateTuple::new();
            seed.push((0, if ordinary_index == 0 { 2 } else { 0 }));
            seed.push((1, if ordinary_index == 1 { 2 } else { 0 }));
            let seed_state = lazy.num_states() as u32;
            super::mapping::augment_product_dfa_from_seed_tuples(
                &mut lazy,
                &mut trace,
                &[seed],
                &plan.exclusions,
                &plan.intersections,
            );
            assert!(lazy.num_states() > seed_state as usize);
            assert!(lazy.step(seed_state, b'x').is_none());

            let compact = super::deferred::try_compile_compact_zero_min_repeat_intersection_runtime(
                &plan,
                false,
            )
            .expect("compact lazy lane should also prune the partial dead cycle");
            let (compact_dfa, compact_segment) = compact.finish_runtime();
            let compact = Tokenizer::from_parts_with_compressed_transitions(
                compact_dfa,
                1,
                None,
                compact_segment.into_iter().collect(),
            );
            let eager = Tokenizer::from_parts(eager, 1, None);
            for input in [b"".as_slice(), b"b", b"x", b"xx", b"xb"] {
                let semantic_observation = |tokenizer: &Tokenizer| {
                    let (matches, futures, _) = tokenizer_observation(tokenizer, input);
                    (matches, futures)
                };
                assert_eq!(
                    semantic_observation(&compact),
                    semantic_observation(&eager),
                    "compact product differed for reversed={reversed} input={input:?}",
                );
            }
        }
    }

    #[test]
    fn compact_zero_min_repeat_runtime_matches_materialized_product() {
        let repeat_component = Expr::Seq(vec![
            Expr::U8Seq(b"[".to_vec()),
            Expr::Repeat {
                expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"ab"))),
                min: 0,
                max: Some(1_024),
            },
            Expr::U8Seq(b"]".to_vec()),
        ]);
        let filter_component = Expr::Seq(vec![
            Expr::U8Seq(b"[".to_vec()),
            Expr::Repeat {
                expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"ab"))),
                min: 0,
                max: Some(7),
            },
            Expr::U8Seq(b"]".to_vec()),
        ]);
        let expression = Expr::Intersect {
            expr: Box::new(repeat_component),
            intersect: Box::new(filter_component),
        };

        let eager = super::compile_with_plan(super::plan::build_exclusion_compile_plan(
            std::slice::from_ref(&expression),
        ));
        let plan = super::plan::build_exclusion_compile_plan(std::slice::from_ref(&expression));
        let compact = super::deferred::try_compile_compact_zero_min_repeat_intersection_runtime(
            &plan,
            false,
        )
        .expect("eligible deterministic repeat intersection must compile compactly");
        let (compact_dfa, compact_segment) = compact.finish_runtime();
        let compact = Tokenizer::from_parts_with_compressed_transitions(
            compact_dfa,
            1,
            None,
            compact_segment.into_iter().collect(),
        );
        let eager = Tokenizer::from_parts(eager, 1, None);

        for input in enumerate_inputs(b"[]abx", 9) {
            assert_eq!(
                tokenizer_observation(&compact, &input),
                tokenizer_observation(&eager, &input),
                "compact runtime product differed for input {input:?}",
            );
        }
    }

    #[test]
    fn compact_zero_min_repeat_runtime_does_not_allocate_from_billion_bound() {
        let repeat_component = Expr::Seq(vec![
            Expr::U8Seq(b"[".to_vec()),
            Expr::Repeat {
                expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"ab"))),
                min: 0,
                max: Some(1_000_000_000),
            },
            Expr::U8Seq(b"]".to_vec()),
        ]);
        // The intersecting coordinate bounds the reachable residual product to
        // a tiny set. Construction must therefore scale with those reachable
        // residuals, not with the syntactic billion-count upper bound.
        let filter_component = Expr::Seq(vec![
            Expr::U8Seq(b"[".to_vec()),
            Expr::Repeat {
                expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"ab"))),
                min: 0,
                max: Some(7),
            },
            Expr::U8Seq(b"]".to_vec()),
        ]);
        let expression = Expr::Intersect {
            expr: Box::new(repeat_component),
            intersect: Box::new(filter_component),
        };
        let plan = super::plan::build_exclusion_compile_plan(std::slice::from_ref(&expression));
        let compact = super::deferred::try_compile_compact_zero_min_repeat_intersection_runtime(
            &plan,
            false,
        )
        .expect("eligible billion-bound repeat intersection must stay compact");
        assert!(compact.num_states() < 128);
    }

    #[test]
    fn exact_runtime_union_preisolates_nullable_components() {
        let expressions = vec![
            Expr::Epsilon,
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                min: 0,
                max: None,
            },
            Expr::U8Seq(b"b".to_vec()),
        ];
        let partitions = [0, 1, 2];
        let residual_classes = [None, None, None];

        let baseline_regex =
            super::build_regex_partitioned_with_options(&expressions, &partitions, super::PartitionOptions { residual_isolation_classes: Some(&residual_classes), adaptive: Some(false), ..super::PartitionOptions::default() });
        let mut baseline = baseline_regex.into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.clone().into_boxed_slice())),
        );
        assert_eq!(
            baseline.isolate_start_state_and_drain_nullable_terminals(),
            BTreeSet::from([0, 1]),
        );

        let mut preisolated = super::partition_runtime::build_exact_partitioned_runtime_tokenizer(
            &expressions,
            None,
            &partitions,
            &residual_classes,
        );
        assert!(
            preisolated
                .isolate_start_state_and_drain_nullable_terminals()
                .is_empty(),
            "component-local isolation must leave no nullable finalizer at the union root",
        );

        for input in enumerate_inputs(b"abx", 5) {
            assert_eq!(
                tokenizer_observation(&preisolated, &input),
                tokenizer_observation(&baseline, &input),
                "component-local nullable isolation differed for input {input:?}",
            );
        }
    }

    fn dfa_state_observation(
        dfa: &super::DFA,
        mut state: u32,
        input: &[u8],
    ) -> (bool, bool, bool) {
        for &byte in input {
            let Some(next) = dfa.step(state, byte) else {
                return (true, false, false);
            };
            state = next;
        }
        (
            false,
            dfa.finalizers(state).contains(0),
            dfa.possible_future_group_ids(state).contains(0),
        )
    }

    #[test]
    fn layered_bounded_suffix_transport_matches_all_short_observations() {
        const HORIZON: usize = 4;
        let alphabet = b"[]abx";
        let inputs = enumerate_inputs(alphabet, HORIZON);
        let bodies = [
            Expr::U8Class(U8Set::from_bytes(b"ab")),
            Expr::U8Seq(b"ab".to_vec()),
        ];

        for body in bodies {
            let make_expr = |max| {
                factor_regex_expr(Expr::Seq(vec![
                    Expr::U8Seq(b"[".to_vec()),
                    Expr::Repeat {
                        expr: Box::new(body.clone()),
                        min: 2,
                        max: Some(max),
                    },
                    Expr::U8Seq(b"]".to_vec()),
                ]))
            };
            let full_expr = make_expr(24);
            let synthesized_expr = make_expr(16);
            let full = super::component::compile_product_component_materialized_dfa(&full_expr);
            let synthesized =
                super::component::compile_product_component_materialized_dfa(&synthesized_expr);
            let mapping = super::mapping::direct_bounded_suffix_state_map(
                &full_expr,
                &synthesized_expr,
                &full,
                &synthesized,
                HORIZON,
                alphabet,
                None,
                None,
            )
            .expect("direct layered transport");
            assert_eq!(mapping.primary().len(), full.num_states());

            for full_state in 0..full.num_states() as u32 {
                let synthesized_state = mapping.primary()[full_state as usize];
                for input in &inputs {
                    assert_eq!(
                        dfa_state_observation(&full, full_state, input),
                        dfa_state_observation(&synthesized, synthesized_state, input),
                        "body={body:?} full_state={full_state} synthesized_state={synthesized_state} input={input:?}",
                    );
                }
            }
        }
    }

    fn tokenizer_observation(
        tokenizer: &Tokenizer,
        input: &[u8],
    ) -> (Vec<(u32, usize)>, Vec<u32>, bool) {
        let execution = tokenizer.execute_from_state_all_widths(input, tokenizer.initial_state());
        let mut matches = execution
            .matches
            .iter()
            .map(|matched| (matched.id, matched.width))
            .collect::<Vec<_>>();
        matches.sort_unstable();
        matches.dedup();

        let mut futures = execution
            .end_state
            .iter()
            .flat_map(|&state| tokenizer.possible_future_terminals_iter(state))
            .collect::<Vec<_>>();
        futures.sort_unstable();
        futures.dedup();
        (matches, futures, execution.end_state.is_empty())
    }

    fn execute_state_set_observation(
        tokenizer: &Tokenizer,
        roots: &[u32],
        input: &[u8],
    ) -> (Vec<(u32, usize)>, Vec<u32>) {
        let mut matches = Vec::new();
        let mut end_states = Vec::new();
        for &root in roots {
            let execution = tokenizer.execute_from_state_all_widths(input, root);
            matches.extend(
                execution
                    .matches
                    .into_iter()
                    .map(|matched| (matched.id, matched.width)),
            );
            end_states.extend(execution.end_state);
        }
        matches.sort_unstable();
        matches.dedup();
        end_states.sort_unstable();
        end_states.dedup();

        let mut futures = end_states
            .iter()
            .flat_map(|&state| tokenizer.possible_future_terminals_iter(state))
            .collect::<Vec<_>>();
        futures.sort_unstable();
        futures.dedup();
        (matches, futures)
    }

    fn random_small_expr(rng: &mut StdRng, depth: usize) -> Expr {
        let atom = |rng: &mut StdRng| match rng.gen_range(0..4) {
            0 => Expr::U8Seq(vec![b'a' + rng.gen_range(0..3)]),
            1 => {
                let len = rng.gen_range(1..=3);
                Expr::U8Seq(
                    (0..len)
                        .map(|_| b'a' + rng.gen_range(0..3))
                        .collect(),
                )
            }
            2 => {
                let mut bytes = Vec::new();
                for byte in b'a'..=b'c' {
                    if rng.gen_bool(0.5) {
                        bytes.push(byte);
                    }
                }
                if bytes.is_empty() {
                    bytes.push(b'a' + rng.gen_range(0..3));
                }
                Expr::U8Class(U8Set::from_bytes(&bytes))
            }
            _ => Expr::Epsilon,
        };

        if depth == 0 {
            return atom(rng);
        }
        match rng.gen_range(0..9) {
            0..=2 => atom(rng),
            3 => Expr::Choice(vec![
                random_small_expr(rng, depth - 1),
                random_small_expr(rng, depth - 1),
            ]),
            4 => Expr::Seq(vec![
                random_small_expr(rng, depth - 1),
                random_small_expr(rng, depth - 1),
            ]),
            5 => Expr::Repeat {
                expr: Box::new(random_small_expr(rng, depth - 1)),
                min: rng.gen_range(0..=1),
                max: Some(rng.gen_range(1..=3)),
            },
            6 => Expr::Repeat {
                expr: Box::new(atom(rng)),
                min: rng.gen_range(0..=1),
                max: None,
            },
            7 => Expr::Exclude {
                expr: Box::new(random_small_expr(rng, depth - 1)),
                exclude: Box::new(random_small_expr(rng, depth - 1)),
            },
            _ => Expr::Intersect {
                expr: Box::new(random_small_expr(rng, depth - 1)),
                intersect: Box::new(random_small_expr(rng, depth - 1)),
            },
        }
    }

    fn random_group_free_expr(rng: &mut StdRng, depth: usize) -> Expr {
        let atom = |rng: &mut StdRng| match rng.gen_range(0..4) {
            0 => Expr::U8Seq(vec![b'a' + rng.gen_range(0..3)]),
            1 => Expr::U8Seq(
                (0..rng.gen_range(1..=3))
                    .map(|_| b'a' + rng.gen_range(0..3))
                    .collect(),
            ),
            2 => Expr::U8Class(U8Set::from_bytes(match rng.gen_range(0..3) {
                0 => b"ab",
                1 => b"bc",
                _ => b"abc",
            })),
            _ => Expr::Epsilon,
        };

        if depth == 0 {
            return atom(rng);
        }
        match rng.gen_range(0..7) {
            0..=2 => atom(rng),
            3 => Expr::Choice(vec![
                random_group_free_expr(rng, depth - 1),
                random_group_free_expr(rng, depth - 1),
            ]),
            4 => Expr::Seq(vec![
                random_group_free_expr(rng, depth - 1),
                random_group_free_expr(rng, depth - 1),
            ]),
            5 => Expr::Repeat {
                expr: Box::new(random_group_free_expr(rng, depth - 1)),
                min: rng.gen_range(0..=1),
                max: Some(rng.gen_range(1..=3)),
            },
            _ => Expr::Shared(Arc::new(random_group_free_expr(rng, depth - 1))),
        }
    }

    fn expr_contains_dfa(expr: &Expr) -> bool {
        match expr {
            Expr::Dfa(_) => true,
            Expr::Exclude { expr, exclude } => {
                expr_contains_dfa(expr) || expr_contains_dfa(exclude)
            }
            Expr::Intersect { expr, intersect } => {
                expr_contains_dfa(expr) || expr_contains_dfa(intersect)
            }
            Expr::Seq(parts) | Expr::Choice(parts) => parts.iter().any(expr_contains_dfa),
            Expr::Repeat { expr, .. } => expr_contains_dfa(expr),
            Expr::Shared(expr) => expr_contains_dfa(expr),
            Expr::U8Seq(_) | Expr::U8Class(_) | Expr::Epsilon => false,
        }
    }

    fn tokenizer_from_partitioned_exprs(exprs: &[Expr]) -> Tokenizer {
        let partitions = (0..exprs.len() as u32).collect::<Vec<_>>();
        build_regex_partitioned_with_adaptive(exprs, &partitions, false).into_tokenizer(
            exprs.len() as u32,
            Some(Arc::from(exprs.to_vec().into_boxed_slice())),
        )
    }

    #[test]
    fn partitioned_lexer_matches_monolithic_semantics_exhaustively() {
        let shared_tail = Arc::new(Expr::Choice(vec![
            Expr::U8Seq(b"b".to_vec()),
            Expr::U8Seq(b"c".to_vec()),
        ]));
        let exprs = vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::Choice(vec![
                Expr::U8Seq(b"ab".to_vec()),
                Expr::U8Seq(b"ac".to_vec()),
            ]),
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                min: 1,
                max: None,
            },
            Expr::Seq(vec![
                Expr::Repeat {
                    expr: Box::new(Expr::U8Seq(b" ".to_vec())),
                    min: 0,
                    max: Some(2),
                },
                Expr::Shared(Arc::clone(&shared_tail)),
            ]),
            Expr::Exclude {
                expr: Box::new(byte_choice(b"abc")),
                exclude: Box::new(byte_expr(b'b')),
            },
            Expr::Intersect {
                expr: Box::new(byte_choice(b"ab")),
                intersect: Box::new(byte_choice(b"bc")),
            },
            Expr::Seq(vec![
                Expr::U8Seq(b"a".to_vec()),
                Expr::Shared(shared_tail),
            ]),
        ];
        let monolithic = build_regex_monolithic(&exprs).into_tokenizer(
            exprs.len() as u32,
            Some(Arc::from(exprs.clone().into_boxed_slice())),
        );
        let partitionings = [
            (0..exprs.len() as u32).collect::<Vec<_>>(),
            vec![0, 0, 1, 1, 2, 2, 1],
        ];
        let inputs = enumerate_inputs(b"abc ", 5);

        for partitions in partitionings {
            let partitioned = build_regex_partitioned_with_adaptive(
                &exprs,
                &partitions,
                false,
            )
            .into_tokenizer(
                exprs.len() as u32,
                Some(Arc::from(exprs.clone().into_boxed_slice())),
            );
            for input in &inputs {
                assert_eq!(
                    tokenizer_observation(&partitioned, input),
                    tokenizer_observation(&monolithic, input),
                    "partitioned lexer differed for partitions={partitions:?} input={input:?}",
                );
            }
        }
    }

    #[test]
    fn seeded_partitioned_lexer_differential_fuzz() {
        let mut rng = StdRng::seed_from_u64(0xE051_10FA_2026_0710);
        let prefixes = enumerate_inputs(b"abc", 3);
        let suffixes = enumerate_inputs(b"abc", 3);

        for case in 0..48 {
            let expr_count = rng.gen_range(2..=6);
            let exprs = (0..expr_count)
                .map(|_| random_small_expr(&mut rng, 2))
                .collect::<Vec<_>>();
            let monolithic = build_regex_monolithic(&exprs).into_tokenizer(
                exprs.len() as u32,
                Some(Arc::from(exprs.clone().into_boxed_slice())),
            );

            for partition_case in 0..3 {
                let partitions = match partition_case {
                    0 => (0..expr_count as u32).collect::<Vec<_>>(),
                    1 => (0..expr_count)
                        .map(|_| rng.gen_range(0..3))
                        .collect::<Vec<_>>(),
                    _ => vec![0; expr_count],
                };
                let partitioned = build_regex_partitioned_with_adaptive(
                    &exprs,
                    &partitions,
                    false,
                )
                .into_tokenizer(
                    exprs.len() as u32,
                    Some(Arc::from(exprs.clone().into_boxed_slice())),
                );
                let adaptive = build_regex_partitioned_with_adaptive(
                    &exprs,
                    &partitions,
                    true,
                )
                .into_tokenizer(
                    exprs.len() as u32,
                    Some(Arc::from(exprs.clone().into_boxed_slice())),
                );

                for prefix in &prefixes {
                    assert_eq!(
                        tokenizer_observation(&partitioned, prefix),
                        tokenizer_observation(&monolithic, prefix),
                        "top-level mismatch case={case} partitions={partitions:?} prefix={prefix:?} exprs={exprs:?}",
                    );
                    assert_eq!(
                        tokenizer_observation(&adaptive, prefix),
                        tokenizer_observation(&monolithic, prefix),
                        "adaptive top-level mismatch case={case} partitions={partitions:?} prefix={prefix:?} exprs={exprs:?}",
                    );

                    let partitioned_roots = partitioned.execute_from_state_end_only(
                        prefix,
                        partitioned.initial_state(),
                    );
                    let adaptive_roots = adaptive.execute_from_state_end_only(
                        prefix,
                        adaptive.initial_state(),
                    );
                    let monolithic_roots = monolithic.execute_from_state_end_only(
                        prefix,
                        monolithic.initial_state(),
                    );
                    for suffix in &suffixes {
                        assert_eq!(
                            execute_state_set_observation(
                                &partitioned,
                                &partitioned_roots,
                                suffix,
                            ),
                            execute_state_set_observation(
                                &monolithic,
                                &monolithic_roots,
                                suffix,
                            ),
                            "residual mismatch case={case} partitions={partitions:?} prefix={prefix:?} suffix={suffix:?} exprs={exprs:?}",
                        );
                        assert_eq!(
                            execute_state_set_observation(
                                &adaptive,
                                &adaptive_roots,
                                suffix,
                            ),
                            execute_state_set_observation(
                                &monolithic,
                                &monolithic_roots,
                                suffix,
                            ),
                            "adaptive residual mismatch case={case} partitions={partitions:?} prefix={prefix:?} suffix={suffix:?} exprs={exprs:?}",
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn nested_group_op_materialization_reuses_structurally_equal_dfas() {
        let nested = Expr::Exclude {
            expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"abc"))),
            exclude: Box::new(Expr::U8Seq(vec![b'b'])),
        };
        let mut cache = super::plan::NestedGroupOpCache::default();
        let first = super::plan::materialize_nested_group_ops(nested.clone(), &mut cache);
        let second = super::plan::materialize_nested_group_ops(nested, &mut cache);

        assert_eq!(cache.cache_misses, 1);
        assert_eq!(cache.cache_hits, 1);
        match (first, second) {
            (Expr::Dfa(first), Expr::Dfa(second)) => assert!(Arc::ptr_eq(&first, &second)),
            _ => panic!("nested group operation was not materialized to a DFA"),
        }
    }

    #[test]
    fn repeated_subexpression_dfa_materialization_preserves_observations_exhaustively() {
        let repeated = Expr::Seq(vec![
            Expr::Choice(vec![byte_expr(b'a'), byte_expr(b'b')]),
            Expr::Repeat {
                expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"bc"))),
                min: 1,
                max: Some(2),
            },
        ]);
        let exprs = vec![
            Expr::Seq(vec![byte_expr(b'a'), repeated.clone(), byte_expr(b'c')]),
            Expr::Choice(vec![repeated.clone(), Expr::U8Seq(b"cc".to_vec())]),
            Expr::Repeat {
                expr: Box::new(repeated.clone()),
                min: 0,
                max: Some(2),
            },
            Expr::Repeat {
                expr: Box::new(repeated.clone()),
                min: 1,
                max: None,
            },
            Expr::Exclude {
                expr: Box::new(Expr::Choice(vec![repeated.clone(), byte_expr(b'c')])),
                exclude: Box::new(byte_expr(b'c')),
            },
            Expr::Intersect {
                expr: Box::new(Expr::Choice(vec![repeated.clone(), Expr::U8Seq(b"ab".to_vec())])),
                intersect: Box::new(Expr::Choice(vec![repeated, byte_expr(b'b')])),
            },
        ];
        let rewritten = super::plan::materialize_repeated_subexpression_dfas_with_limits(&exprs, 2, 2)
            .expect("the repeated subtree should be materialized");
        assert!(rewritten.iter().any(expr_contains_dfa));

        let original = tokenizer_from_partitioned_exprs(&exprs);
        let materialized = tokenizer_from_partitioned_exprs(&rewritten);
        for input in enumerate_inputs(b"abc", 6) {
            assert_eq!(
                tokenizer_observation(&materialized, &input),
                tokenizer_observation(&original, &input),
                "materialized repeated subtree changed tokenizer observation for input={input:?}",
            );
        }
    }

    #[test]
    fn repeated_subexpression_dfa_materialization_seeded_differential() {
        let mut rng = StdRng::seed_from_u64(0xC5E0_2026_0718);
        let inputs = enumerate_inputs(b"abc", 4);

        for case in 0..32 {
            let repeated = Expr::Seq(vec![
                random_group_free_expr(&mut rng, 2),
                random_group_free_expr(&mut rng, 2),
            ]);
            let exprs = vec![
                Expr::Seq(vec![byte_expr(b'a'), repeated.clone(), byte_expr(b'b')]),
                Expr::Choice(vec![repeated.clone(), random_group_free_expr(&mut rng, 1)]),
                Expr::Repeat {
                    expr: Box::new(repeated.clone()),
                    min: rng.gen_range(0..=1),
                    max: Some(rng.gen_range(1..=3)),
                },
                Expr::Shared(Arc::new(repeated)),
            ];
            let rewritten =
                super::plan::materialize_repeated_subexpression_dfas_with_limits(&exprs, 2, 2)
                    .expect("the seeded repeated subtree should be materialized");
            let original = tokenizer_from_partitioned_exprs(&exprs);
            let materialized = tokenizer_from_partitioned_exprs(&rewritten);

            for input in &inputs {
                assert_eq!(
                    tokenizer_observation(&materialized, input),
                    tokenizer_observation(&original, input),
                    "seeded CSE mismatch case={case} input={input:?} exprs={exprs:?} rewritten={rewritten:?}",
                );
            }
        }
    }

    #[test]
    fn duplicate_product_coordinates_collapse_without_changing_observations() {
        let repeated = Expr::Seq(vec![
            Expr::U8Class(U8Set::from_bytes(b"ab")),
            Expr::Repeat {
                expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"bc"))),
                min: 1,
                max: Some(3),
            },
        ]);
        let expressions = vec![
            repeated.clone(),
            repeated,
            Expr::U8Seq(b"ac".to_vec()),
        ];
        let exclusions = BTreeMap::new();
        let intersections = BTreeMap::new();
        let collapsed = super::product::build_product_dfa(
            &expressions,
            None,
            expressions.len(),
            &exclusions,
            &intersections,
            false,
            true,
            false,
            false,
        )
        .0;
        let uncollapsed = super::product::build_product_dfa(
            &expressions,
            None,
            expressions.len(),
            &exclusions,
            &intersections,
            true,
            true,
            false,
            false,
        )
        .0;
        assert_eq!(collapsed, uncollapsed);

        let (traced_collapsed, _, trace) = super::product::build_product_dfa(
            &expressions,
            None,
            expressions.len(),
            &exclusions,
            &intersections,
            true,
            true,
            false,
            true,
        );
        assert_eq!(collapsed, traced_collapsed);
        let trace = trace.expect("trace-enabled collapsed product must retain its state tuples");
        assert_eq!(trace.state_tuples.len(), traced_collapsed.num_states());
        assert!(
            trace.coordinate_groups.iter().any(|groups| groups.len() > 1),
            "duplicate logical groups should fan out from one physical product coordinate"
        );
    }

    #[test]
    fn duplicate_product_coordinates_preserve_group_operations() {
        let repeated = Expr::Seq(vec![
            Expr::U8Class(U8Set::from_bytes(b"ab")),
            Expr::Repeat {
                expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"bc"))),
                min: 1,
                max: Some(3),
            },
        ]);
        let expressions = vec![
            Expr::Exclude {
                expr: Box::new(repeated.clone()),
                exclude: Box::new(repeated.clone()),
            },
            Expr::Choice(vec![repeated, Expr::U8Seq(b"ac".to_vec())]),
        ];
        let collapsed = super::compile_with_plan_internal(
            super::plan::build_exclusion_compile_plan(&expressions),
            false,
        )
        .0;
        let uncollapsed = super::compile_with_plan_internal(
            super::plan::build_exclusion_compile_plan(&expressions),
            true,
        )
        .0;
        assert_eq!(collapsed, uncollapsed);
    }

    #[test]
    fn duplicate_product_components_share_the_same_dfa() {
        let expr = Expr::Seq(vec![
            Expr::U8Seq(vec![b'"']),
            Expr::Repeat {
                expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"abc"))),
                min: 0,
                max: None,
            },
        ]);
        let shared_expr = Expr::Shared(Arc::new(expr.clone()));
        let (components, cache_hits) = super::product::compile_product_components(&[expr, shared_expr]);

        assert_eq!(cache_hits, 1);
        match (&components[0], &components[1]) {
            (
                super::product::ProductComponent::Materialized(first),
                super::product::ProductComponent::Materialized(second),
            ) => assert!(Arc::ptr_eq(first, second)),
            _ => panic!("expected materialized product components"),
        }
    }

    #[test]
    fn factors_contains_literal_choice_with_common_quoted_string_shell() {
        let char = Expr::U8Class(U8Set::from_bytes(br#"ABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789"#));
        let mk = |s: &[u8]| {
            Expr::Seq(vec![
                Expr::U8Seq(b"\"".to_vec()),
                Expr::Repeat {
                    expr: Box::new(char.clone()),
                    min: 0,
                    max: None,
                },
                Expr::U8Seq(s.to_vec()),
                Expr::Repeat {
                    expr: Box::new(char.clone()),
                    min: 0,
                    max: None,
                },
                Expr::U8Seq(b"\"".to_vec()),
            ])
        };
        let expr = Expr::Seq(vec![
            Expr::U8Seq(b"\"interval\": ".to_vec()),
            Expr::Choice(vec![
                mk(b"INTERVAL_TICK"),
                mk(b"INTERVAL_M1"),
                mk(b"INTERVAL_M2"),
                mk(b"INTERVAL_M3"),
                mk(b"INTERVAL_M4"),
                mk(b"INTERVAL_M5"),
                mk(b"INTERVAL_M6"),
                mk(b"INTERVAL_M10"),
                mk(b"INTERVAL_M15"),
                mk(b"INTERVAL_M20"),
                mk(b"INTERVAL_M30"),
                mk(b"INTERVAL_H1"),
                mk(b"INTERVAL_H2"),
                mk(b"INTERVAL_H4"),
                mk(b"INTERVAL_D1"),
                mk(b"INTERVAL_W1"),
                mk(b"INTERVAL_MN1"),
            ]),
        ]);
        let factored = factor_regex_expr(expr);
        let regex = build_regex(&[factored]);
        assert!(
            regex.num_states() < 500,
            "factored regex should not construct a huge terminal DFA; states={}",
            regex.num_states(),
        );
        let accept = |bytes: &[u8]| {
            let mut state = 0;
            for &b in bytes {
                let Some(next) = regex.step(state, b) else {
                    return false;
                };
                state = next;
            }
            !regex.dfa.finalizers(state).is_empty()
        };
        assert!(accept(br#""interval": "INTERVAL_M1""#));
        assert!(accept(br#""interval": "XXXINTERVAL_M1YYY""#));
        assert!(accept(br#""interval": "INTERVAL_TICK""#));
        assert!(!accept(br#""interval": "NOPE""#));
    }

    #[test]
    fn factors_repeated_choice_edges_shared_by_only_some_arms() {
        let char = Expr::U8Class(U8Set::from_bytes(b"abcdefghijklmnopqrstuvwxyz"));
        let star = Expr::Repeat {
            expr: Box::new(char),
            min: 0,
            max: None,
        };
        let original = Expr::Choice(vec![
            Expr::Seq(vec![Expr::U8Seq(b"list".to_vec()), star.clone()]),
            Expr::Seq(vec![
                star.clone(),
                Expr::U8Seq(b"date".to_vec()),
                star.clone(),
            ]),
            Expr::Seq(vec![
                star.clone(),
                Expr::U8Seq(b"time".to_vec()),
                star.clone(),
            ]),
            Expr::Seq(vec![star.clone(), Expr::U8Seq(b"number".to_vec())]),
        ]);
        let factored = factor_regex_expr(original.clone());

        for input in [
            b"list".as_slice(),
            b"listanything".as_slice(),
            b"xxdateyy".as_slice(),
            b"time".as_slice(),
            b"prefixnumbersuffix".as_slice(),
            b"prefixnumber".as_slice(),
            b"nothing".as_slice(),
        ] {
            assert_eq!(
                terminal_matches(original.clone(), input),
                terminal_matches(factored.clone(), input),
                "subset choice-edge factoring changed acceptance for {input:?}",
            );
        }
    }

    #[test]
    fn factors_repeated_exclusion_rhs_across_choice_arms_exactly() {
        let literal_choice = |a: &[u8], b: &[u8]| {
            Expr::Choice(vec![Expr::U8Seq(a.to_vec()), Expr::U8Seq(b.to_vec())])
        };
        let original = Expr::Choice(vec![
            Expr::Exclude {
                expr: Box::new(literal_choice(b"bad", b"cat")),
                exclude: Box::new(Expr::U8Seq(b"bad".to_vec())),
            },
            Expr::Exclude {
                expr: Box::new(literal_choice(b"bad", b"dog")),
                exclude: Box::new(Expr::U8Seq(b"bad".to_vec())),
            },
            Expr::Exclude {
                expr: Box::new(literal_choice(b"fox", b"yak")),
                exclude: Box::new(Expr::U8Seq(b"yak".to_vec())),
            },
            Expr::U8Seq(b"eel".to_vec()),
        ]);
        let factored = factor_regex_expr(original.clone());
        assert!(
            super::factor::group_op_node_count(&factored) < super::factor::group_op_node_count(&original),
            "repeated exclusion RHS should collapse sibling set-difference nodes",
        );

        for input in [
            b"".as_slice(),
            b"bad".as_slice(),
            b"cat".as_slice(),
            b"dog".as_slice(),
            b"fox".as_slice(),
            b"yak".as_slice(),
            b"eel".as_slice(),
            b"other".as_slice(),
        ] {
            assert_eq!(
                terminal_matches(original.clone(), input),
                terminal_matches(factored.clone(), input),
                "repeated exclusion-RHS factoring changed acceptance for {input:?}",
            );
        }
    }

    #[test]
    fn shared_factoring_cache_preserves_factored_arc_identity() {
        let shared = Arc::new(Expr::Choice(vec![
            Expr::Seq(vec![Expr::U8Seq(b"pre".to_vec()), Expr::U8Seq(b"a".to_vec())]),
            Expr::Seq(vec![Expr::U8Seq(b"pre".to_vec()), Expr::U8Seq(b"b".to_vec())]),
        ]));
        let factored_shared = Arc::new(factor_regex_expr((*shared).clone()));
        let mut cache = rustc_hash::FxHashMap::default();
        cache.insert(Arc::as_ptr(&shared) as usize, Arc::clone(&factored_shared));
        let expression = Expr::Choice(vec![
            Expr::Shared(Arc::clone(&shared)),
            Expr::Seq(vec![Expr::U8Seq(b"x".to_vec()), Expr::Shared(shared)]),
        ]);

        let factored = super::factor::factor_regex_expr_with_shared_cache(expression, &cache);
        let Expr::Choice(arms) = factored else {
            panic!("factoring should retain the outer choice");
        };
        let Expr::Shared(first) = &arms[0] else {
            panic!("cached shared root should remain shared");
        };
        let Expr::Seq(second_parts) = &arms[1] else {
            panic!("second arm should remain a sequence");
        };
        let Expr::Shared(second) = &second_parts[1] else {
            panic!("nested cached root should remain shared");
        };
        assert!(Arc::ptr_eq(first, &factored_shared));
        assert!(Arc::ptr_eq(second, &factored_shared));
        assert!(Arc::ptr_eq(first, second));
    }

    #[test]
    fn factors_aligned_zero_min_unit_repeat_intersection_exactly() {
        let word = U8Set::from_bytes(
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_",
        );
        let letters =
            U8Set::from_bytes(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");
        let expression = Expr::Intersect {
            expr: Box::new(Expr::Repeat {
                expr: Box::new(Expr::U8Class(word)),
                min: 0,
                max: Some(1_000_000_000),
            }),
            intersect: Box::new(Expr::Repeat {
                expr: Box::new(Expr::U8Class(letters)),
                min: 0,
                max: Some(700_000_000),
            }),
        };

        assert_eq!(
            factor_regex_expr(expression),
            Expr::Repeat {
                expr: Box::new(Expr::U8Class(letters)),
                min: 0,
                max: Some(700_000_000),
            },
        );
    }

    #[test]
    fn aligned_unit_repeat_intersection_factoring_preserves_small_language() {
        let left = Expr::Repeat {
            expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"ab"))),
            min: 0,
            max: Some(3),
        };
        let right = Expr::Repeat {
            expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"bc"))),
            min: 0,
            max: Some(2),
        };
        let original = Expr::Intersect {
            expr: Box::new(left),
            intersect: Box::new(right),
        };
        let factored = factor_regex_expr(original.clone());

        for input in enumerate_inputs(b"abc", 4) {
            assert_eq!(
                terminal_matches(original.clone(), &input),
                terminal_matches(factored.clone(), &input),
                "factoring changed acceptance for {input:?}",
            );
        }
    }

    #[test]
    fn factors_aligned_nonzero_unit_repeat_intersection_exactly() {
        let left = Expr::Repeat {
            expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"ab"))),
            min: 2,
            max: Some(5),
        };
        let right = Expr::Repeat {
            expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"bc"))),
            min: 3,
            max: Some(4),
        };
        let original = Expr::Intersect {
            expr: Box::new(left),
            intersect: Box::new(right),
        };
        let factored = factor_regex_expr(original.clone());
        assert_eq!(
            factored,
            Expr::Repeat {
                expr: Box::new(Expr::U8Class(U8Set::single(b'b'))),
                min: 3,
                max: Some(4),
            },
        );
        for input in enumerate_inputs(b"abc", 6) {
            assert_eq!(
                terminal_matches(original.clone(), &input),
                terminal_matches(factored.clone(), &input),
                "nonzero aligned-unit factoring changed acceptance for {input:?}",
            );
        }

        let disjoint = Expr::Intersect {
            expr: Box::new(Expr::Repeat {
                expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"ab"))),
                min: 1,
                max: Some(2),
            }),
            intersect: Box::new(Expr::Repeat {
                expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"bc"))),
                min: 3,
                max: Some(4),
            }),
        };
        assert_eq!(factor_regex_expr(disjoint), Expr::U8Class(U8Set::empty()));
    }

    #[test]
    fn aligned_unit_empty_byte_intersection_normalizes_fully() {
        let expression_for = |min| Expr::Intersect {
            expr: Box::new(Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                min,
                max: Some(super::virtual_repeat::VIRTUAL_BINARY_REPEAT_MIN_BOUND),
            }),
            intersect: Box::new(Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"b".to_vec())),
                min,
                max: Some(super::virtual_repeat::VIRTUAL_BINARY_REPEAT_MIN_BOUND),
            }),
        };
        assert_eq!(factor_regex_expr(expression_for(0)), Expr::Epsilon);
        assert_eq!(
            factor_regex_expr(expression_for(1)),
            Expr::U8Class(U8Set::empty()),
        );
    }

    #[test]
    fn aligned_repeat_intersection_rewrite_fails_closed_for_variable_width_body() {
        let variable_width = Expr::Choice(vec![byte_expr(b'a'), Expr::U8Seq(b"aa".to_vec())]);
        let expression = Expr::Intersect {
            expr: Box::new(Expr::Repeat {
                expr: Box::new(variable_width),
                min: 0,
                max: Some(100),
            }),
            intersect: Box::new(Expr::Repeat {
                expr: Box::new(byte_expr(b'a')),
                min: 0,
                max: Some(100),
            }),
        };

        assert!(matches!(factor_regex_expr(expression), Expr::Intersect { .. }));
    }

    #[test]
    fn sparse_virtual_repeat_finite_product_does_not_flatten_repeat_state_ids() {
        const LONG: usize = 65_536;
        let mut base_dfa = DFA::new(LONG + 1);
        base_dfa.ensure_group_capacity(1);
        base_dfa.set_transitions_from_sorted_entries(0, vec![(b'a', LONG as u32)]);
        let mut finalizers = crate::ds::bitset::BitSet::new(1);
        finalizers.set(0);
        base_dfa.overwrite_state_metadata(
            LONG as u32,
            finalizers,
            crate::ds::bitset::BitSet::new(1),
        );
        let components = vec![
            super::product::ProductComponent::VirtualBoundedRepeat {
                base_dfa: Arc::new(base_dfa),
                min: 1,
                max: 1_000_000_000,
            },
            super::product::ProductComponent::VirtualFixedSequence {
                byte_sets: Arc::from(
                    vec![U8Set::single(b'a'); LONG].into_boxed_slice(),
                ),
                suffix_live: Arc::from(vec![true; LONG + 1].into_boxed_slice()),
            },
        ];
        let (class_map, class_members) = super::product::compute_product_equivalence_classes(&components);
        let class_transitions = super::product::build_product_class_transitions(&components, &class_map);
        let (dfa, direct, trace) =
            super::product::try_build_sparse_virtual_repeat_finite_intersection_product(
                &components,
                &class_members,
                &class_transitions,
                false,
                false,
            )
            .expect("virtual repeat with finite coordinate should use sparse exact product");

        // The synthetic base DFA has 65,537 states while the finite side
        // drives 65,536 completed copies. The old flattened coordinate would
        // need 65,536 * 65,537 = 4,295,032,832, beyond u32::MAX. The structural
        // coordinate keeps only the actually reachable O(LONG) pair states.
        assert!(direct);
        assert!(trace.is_none());
        assert_eq!(dfa.num_states(), LONG + 1);
        let mut state = 0u32;
        for _ in 0..LONG {
            state = dfa.step(state, b'a').expect("accepted finite-prefix step");
        }
        assert!(dfa.finalizers(state).contains(0));
    }

    #[test]
    fn lazy_giant_intersection_materializes_only_budgeted_non_giant_repeat_side() {
        // This root repeat is above the generic product's direct-repeat
        // threshold, so it would normally be represented as a
        // VirtualBoundedRepeat. Its exact DFA is still tiny, however, and the
        // lazy intersection lane may materialize it under the shared budget.
        let small_other = Expr::Repeat {
            expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"ab"))),
            min: 0,
            max: Some(100),
        };
        assert!(super::deferred::compile_lazy_intersection_materialized_other_component(
            &small_other,
            true,
        )
        .is_some());

        // Stay one below the giant-repeat threshold so the right side remains
        // nominally "non-giant", but make its exact bounded-repeat DFA exceed
        // the 100k materialization budget. The specialized fast path must
        // decline instead of turning this optimization into a large eager
        // allocation; the general residual runtime remains the correctness
        // fallback at the dynamic-constraint layer.
        let oversized_other = Expr::Repeat {
            expr: Box::new(Expr::U8Seq({
                let mut bytes = vec![b'a'; 31];
                bytes.push(b'b');
                bytes
            })),
            min: 0,
            max: Some(super::virtual_repeat::VIRTUAL_BINARY_REPEAT_MIN_BOUND - 1),
        };
        assert!(super::deferred::compile_lazy_intersection_materialized_other_component(
            &oversized_other,
            true,
        )
        .is_none());

        let cyclic_other = Expr::Repeat {
            expr: Box::new(byte_expr(b'a')),
            min: 0,
            max: None,
        };
        assert!(super::deferred::compile_lazy_intersection_materialized_other_component(
            &cyclic_other,
            true,
        )
        .is_none());
    }

    #[test]
    fn bounded_repeat_from_base_rejects_layer_and_state_id_overflow() {
        let base = super::nfa::compile_expr_to_dfa(&byte_expr(b'a'));
        assert!(
            super::bounded_repeat::build_bounded_repeat_dfa_from_base(&base, 2, 1).is_none(),
            "an invalid repeat interval must fail closed",
        );
        assert!(
            super::bounded_repeat::build_bounded_repeat_dfa_from_base(&base, 0, usize::MAX).is_none(),
            "max + 1 must fail closed instead of overflowing",
        );
        assert!(
            super::bounded_repeat::build_bounded_repeat_dfa_from_base(&base, 0, u32::MAX as usize).is_none(),
            "layered state IDs must fit in u32",
        );
    }

    #[test]
    fn bounded_repeat_from_base_marks_only_productive_residuals_future_live() {
        let mut base = DFA::new(3);
        base.ensure_group_capacity(1);
        base.set_transitions_from_sorted_entries(0, vec![(b'a', 1), (b'x', 2)]);
        base.set_transitions_from_sorted_entries(2, vec![(b'x', 2)]);
        let mut finalizers = BitSet::new(1);
        finalizers.set(0);
        base.overwrite_state_metadata(1, finalizers, BitSet::new(1));

        let repeated = super::bounded_repeat::build_bounded_repeat_dfa_from_base(&base, 0, 2)
            .expect("small repeat must materialize");
        let dead = repeated.step(0, b'x').expect("dead residual remains represented");
        assert!(
            !repeated.possible_future_group_ids(dead).contains(0),
            "a reachable base residual with no path to a finalizer must not advertise a repeat future",
        );
        let live = repeated.step(0, b'a').expect("accepted copy transition");
        assert!(repeated.finalizers(live).contains(0));
        assert!(repeated.possible_future_group_ids(live).contains(0));
    }

    #[test]
    fn bounded_repeat_suffix_ignores_unproductive_delimiter_transition() {
        let body = byte_expr(b'a');
        let mut supplied_base = DFA::new(3);
        supplied_base.ensure_group_capacity(1);
        supplied_base.set_transitions_from_sorted_entries(0, vec![(b'a', 1), (b'x', 2)]);
        supplied_base.set_transitions_from_sorted_entries(2, vec![(b'x', 2)]);
        let mut finalizers = BitSet::new(1);
        finalizers.set(0);
        supplied_base.overwrite_state_metadata(1, finalizers, BitSet::new(1));

        let mut cache = super::bounded_repeat::RepeatBaseDfaCache::default();
        cache.insert(body.clone(), Arc::new(supplied_base));
        let parts = vec![
            Expr::Repeat {
                expr: Box::new(body.clone()),
                min: 0,
                max: Some(2),
            },
            byte_expr(b'x'),
        ];
        let (with_dead_edge, _) = super::bounded_repeat::build_bounded_repeat_with_suffix_dfa_with_cache(
            &parts,
            Some(&cache),
        )
        .expect("an unproductive body edge on the suffix byte is not a boundary ambiguity");

        let clean_parts = vec![
            Expr::Repeat {
                expr: Box::new(body),
                min: 0,
                max: Some(2),
            },
            byte_expr(b'x'),
        ];
        let clean = super::bounded_repeat::build_bounded_repeat_with_suffix_dfa(&clean_parts)
            .expect("clean prefix-free repeat+suffix must use the direct builder")
            .0;
        for input in enumerate_inputs(b"ax", 4) {
            assert_eq!(
                dfa_accepts(&with_dead_edge, &input),
                dfa_accepts(&clean, &input),
                "dropping an unproductive body transition changed the repeat+suffix language for {input:?}",
            );
        }
        let suffix_target = with_dead_edge
            .step(0, b'x')
            .expect("the delimiter must enter the suffix, not the dead body residual");
        assert!(with_dead_edge.finalizers(suffix_target).contains(0));

        let overflowing = vec![
            Expr::Repeat {
                expr: Box::new(byte_expr(b'a')),
                min: 0,
                max: Some(usize::MAX),
            },
            byte_expr(b'x'),
        ];
        assert!(
            super::bounded_repeat::build_bounded_repeat_with_suffix_dfa(&overflowing).is_none(),
            "repeat+suffix layer arithmetic must fail closed on max + 1 overflow",
        );
    }

    #[test]
    fn factors_same_body_giant_repeats_with_unique_literal_suffix_boundary() {
        let body = factor_regex_expr(Expr::Choice(vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"bc".to_vec()),
        ]));
        let component = |max, suffix: &[u8]| {
            Expr::Seq(vec![
                Expr::U8Seq(b"[".to_vec()),
                Expr::Repeat {
                    expr: Box::new(body.clone()),
                    min: 0,
                    max: Some(max),
                },
                Expr::U8Seq(suffix.to_vec()),
            ])
        };

        let shared_suffix = Expr::Intersect {
            expr: Box::new(component(
                super::virtual_repeat::VIRTUAL_BINARY_REPEAT_MIN_BOUND + 7,
                b"]abca",
            )),
            intersect: Box::new(component(
                super::virtual_repeat::VIRTUAL_BINARY_REPEAT_MIN_BOUND,
                b"]abca",
            )),
        };
        assert_eq!(
            factor_regex_expr(shared_suffix),
            Expr::Seq(vec![
                Expr::U8Seq(b"[".to_vec()),
                Expr::Repeat {
                    expr: Box::new(body.clone()),
                    min: 0,
                    max: Some(super::virtual_repeat::VIRTUAL_BINARY_REPEAT_MIN_BOUND),
                },
                Expr::U8Seq(b"]abca".to_vec()),
            ]),
        );

        let disjoint_suffixes = Expr::Intersect {
            expr: Box::new(component(super::virtual_repeat::VIRTUAL_BINARY_REPEAT_MIN_BOUND, b"]")),
            intersect: Box::new(component(super::virtual_repeat::VIRTUAL_BINARY_REPEAT_MIN_BOUND + 3, b"}")),
        };
        assert_eq!(
            factor_regex_expr(disjoint_suffixes),
            Expr::U8Class(U8Set::empty()),
        );

        // A small materialized analogue is a cheap semantic oracle for the
        // variable-width prefix-free body: no byte string can satisfy the two
        // different uniquely-delimited suffixes.
        let small_left = component(3, b"]");
        let small_right = component(4, b"}");
        let small_left = compile_product_component_dfa(&small_left);
        let small_right = compile_product_component_dfa(&small_right);
        for input in enumerate_inputs(b"[abc]}", 6) {
            assert!(
                !(dfa_accepts(&small_left, &input) && dfa_accepts(&small_right, &input)),
                "unexpected common word for uniquely delimited suffixes: {input:?}",
            );
        }
    }

    #[test]
    fn delimited_repeat_suffix_factoring_handles_only_safe_nonzero_min_results() {
        let giant = super::virtual_repeat::VIRTUAL_BINARY_REPEAT_MIN_BOUND;
        let component = |min, max, suffix: &[u8]| {
            Expr::Seq(vec![
                Expr::Repeat {
                    expr: Box::new(byte_expr(b'a')),
                    min,
                    max: Some(max),
                },
                Expr::U8Seq(suffix.to_vec()),
            ])
        };

        let disjoint_counts = Expr::Intersect {
            expr: Box::new(component(5, giant, b"b")),
            intersect: Box::new(component(1, 3, b"b")),
        };
        assert_eq!(
            factor_regex_expr(disjoint_counts),
            Expr::U8Class(U8Set::empty()),
            "unique body/suffix boundaries make disjoint count intervals exactly empty",
        );

        let different_suffixes = Expr::Intersect {
            expr: Box::new(component(2, giant, b"b")),
            intersect: Box::new(component(3, giant + 1, b"c")),
        };
        assert_eq!(
            factor_regex_expr(different_suffixes),
            Expr::U8Class(U8Set::empty()),
            "different uniquely-delimited literals are disjoint even with positive minima",
        );

        let bounded_merge = Expr::Intersect {
            expr: Box::new(component(2, giant, b"b")),
            intersect: Box::new(component(3, 7, b"b")),
        };
        assert_eq!(
            factor_regex_expr(bounded_merge),
            Expr::Seq(vec![
                Expr::Repeat {
                    expr: Box::new(byte_expr(b'a')),
                    min: 3,
                    max: Some(7),
                },
                Expr::U8Seq(b"b".to_vec()),
            ]),
            "a positive-minimum merged interval is safe when its upper bound is ordinary",
        );

        let unsupported_giant_result = Expr::Intersect {
            expr: Box::new(component(2, giant + 1, b"b")),
            intersect: Box::new(component(3, giant, b"b")),
        };
        assert!(
            matches!(
                factor_regex_expr(unsupported_giant_result),
                Expr::Intersect { .. }
            ),
            "do not rewrite to a positive-minimum giant repeat+suffix before that shape has an exact runtime lane",
        );

        let long_body = Expr::U8Seq(vec![b'a'; 32]);
        let large_ordinary = |min, max| {
            Expr::Seq(vec![
                Expr::Repeat {
                    expr: Box::new(long_body.clone()),
                    min,
                    max: Some(max),
                },
                Expr::U8Seq(b"b".to_vec()),
            ])
        };
        let over_budget_ordinary_result = Expr::Intersect {
            expr: Box::new(large_ordinary(0, giant)),
            intersect: Box::new(large_ordinary(0, giant - 1)),
        };
        assert!(
            matches!(
                factor_regex_expr(over_budget_ordinary_result),
                Expr::Intersect { .. }
            ),
            "a sub-threshold merged repeat must still remain unfactored when its exact layered DFA exceeds the explicit budget",
        );
    }

    #[test]
    fn delimited_repeat_suffix_factoring_rejects_count_shift_and_non_code_bodies() {
        let giant = super::virtual_repeat::VIRTUAL_BINARY_REPEAT_MIN_BOUND;
        let repeat = |body, suffix: &[u8]| {
            Expr::Seq(vec![
                Expr::Repeat {
                    expr: Box::new(body),
                    min: 0,
                    max: Some(giant),
                },
                Expr::U8Seq(suffix.to_vec()),
            ])
        };

        // The left suffix begins with 'a', which can start another body word.
        // Indeed a*ab and a*b overlap by shifting one repetition across the
        // suffix boundary, so the algebraic rewrite must not fire.
        let shifted = Expr::Intersect {
            expr: Box::new(repeat(byte_expr(b'a'), b"ab")),
            intersect: Box::new(repeat(byte_expr(b'a'), b"b")),
        };
        assert!(matches!(factor_regex_expr(shifted), Expr::Intersect { .. }));

        // Structural body equality is not enough: {a, aa} is not prefix-free,
        // so a byte string can carry different factor counts before the same
        // delimiter. Keep the original intersection when the code proof fails.
        let ambiguous = Expr::Choice(vec![byte_expr(b'a'), Expr::U8Seq(b"aa".to_vec())]);
        let non_code = Expr::Intersect {
            expr: Box::new(repeat(ambiguous.clone(), b"b")),
            intersect: Box::new(repeat(ambiguous, b"b")),
        };
        assert!(matches!(factor_regex_expr(non_code), Expr::Intersect { .. }));

        // Even identical arbitrary prefix languages cannot in general be
        // cancelled from an intersection. This rewrite only accepts one fixed
        // literal prefix byte string.
        let nonliteral_prefix = Expr::Choice(vec![Expr::Epsilon, byte_expr(b'p')]);
        let prefixed = |suffix: &[u8]| {
            Expr::Seq(vec![
                nonliteral_prefix.clone(),
                Expr::Repeat {
                    expr: Box::new(byte_expr(b'a')),
                    min: 0,
                    max: Some(giant),
                },
                Expr::U8Seq(suffix.to_vec()),
            ])
        };
        let nonliteral = Expr::Intersect {
            expr: Box::new(prefixed(b"b")),
            intersect: Box::new(prefixed(b"c")),
        };
        assert!(matches!(factor_regex_expr(nonliteral), Expr::Intersect { .. }));
    }

    #[test]
    fn factors_same_prefix_free_body_nonzero_repeat_intersection_exactly() {
        let body = factor_regex_expr(Expr::Choice(vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"bc".to_vec()),
        ]));
        let original = Expr::Intersect {
            expr: Box::new(Expr::Repeat {
                expr: Box::new(body.clone()),
                min: 1,
                max: Some(super::virtual_repeat::VIRTUAL_BINARY_REPEAT_MIN_BOUND),
            }),
            intersect: Box::new(Expr::Repeat {
                expr: Box::new(body.clone()),
                min: 2,
                max: Some(3),
            }),
        };
        let factored = factor_regex_expr(original.clone());
        assert_eq!(
            factored,
            Expr::Repeat {
                expr: Box::new(body),
                min: 2,
                max: Some(3),
            },
        );

        // Keep the giant side only barely above the virtualization threshold
        // so the unfactored expression is still a cheap materialized oracle.
        let original_dfa = build_regex(std::slice::from_ref(&original)).dfa;
        let factored_dfa = build_regex(std::slice::from_ref(&factored)).dfa;
        for input in enumerate_inputs(b"abc", 6) {
            assert_eq!(
                dfa_accepts(&original_dfa, &input),
                dfa_accepts(&factored_dfa, &input),
                "same-body factoring changed acceptance for {input:?}",
            );
        }
    }

    #[test]
    fn same_prefix_free_body_disjoint_repeat_intervals_factor_to_empty() {
        let body = Expr::Choice(vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"bc".to_vec()),
        ]);
        let expression = Expr::Intersect {
            expr: Box::new(Expr::Repeat {
                expr: Box::new(body.clone()),
                min: 4,
                max: Some(super::virtual_repeat::VIRTUAL_BINARY_REPEAT_MIN_BOUND),
            }),
            intersect: Box::new(Expr::Repeat {
                expr: Box::new(body),
                min: 1,
                max: Some(3),
            }),
        };
        assert_eq!(factor_regex_expr(expression), Expr::U8Class(U8Set::empty()));
    }

    #[test]
    fn same_body_nonzero_repeat_factoring_requires_prefix_free_body() {
        let body = Expr::Choice(vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"aa".to_vec()),
        ]);
        let expression = Expr::Intersect {
            expr: Box::new(Expr::Repeat {
                expr: Box::new(body.clone()),
                min: 1,
                max: Some(super::virtual_repeat::VIRTUAL_BINARY_REPEAT_MIN_BOUND),
            }),
            intersect: Box::new(Expr::Repeat {
                expr: Box::new(body),
                min: 2,
                max: Some(3),
            }),
        };
        assert!(matches!(factor_regex_expr(expression), Expr::Intersect { .. }));

        // Disjoint factor-count intervals are not enough without unique
        // decipherability: a^(4097) has both a 4097-copy decomposition and a
        // 4096-copy decomposition when the body language is {"a", "aa"}.
        let body = Expr::Choice(vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"aa".to_vec()),
        ]);
        let disjoint = Expr::Intersect {
            expr: Box::new(Expr::Repeat {
                expr: Box::new(body.clone()),
                min: 1,
                max: Some(super::virtual_repeat::VIRTUAL_BINARY_REPEAT_MIN_BOUND),
            }),
            intersect: Box::new(Expr::Repeat {
                expr: Box::new(body),
                min: super::virtual_repeat::VIRTUAL_BINARY_REPEAT_MIN_BOUND + 1,
                max: Some(super::virtual_repeat::VIRTUAL_BINARY_REPEAT_MIN_BOUND + 1),
            }),
        };
        assert!(matches!(factor_regex_expr(disjoint), Expr::Intersect { .. }));
    }

    #[test]
    fn nested_exclude_in_exclusion_branch_compiles() {
        let nested_residual = Expr::Exclude {
            expr: Box::new(byte_choice(b"ab")),
            exclude: Box::new(byte_expr(b'a')),
        };
        assert!(!terminal_matches(nested_residual.clone(), b"a"));
        assert!(terminal_matches(nested_residual.clone(), b"b"));
        assert!(!terminal_matches(nested_residual.clone(), b"c"));

        let expr = Expr::Exclude {
            expr: Box::new(byte_choice(b"bc")),
            exclude: Box::new(nested_residual),
        };

        assert!(!terminal_matches(expr.clone(), b"a"));
        assert!(!terminal_matches(expr.clone(), b"b"));
        assert!(terminal_matches(expr, b"c"));
    }

    #[test]
    fn nested_intersect_in_exclusion_branch_compiles() {
        let nested_intersection = Expr::Intersect {
            expr: Box::new(byte_choice(b"ab")),
            intersect: Box::new(byte_expr(b'b')),
        };
        assert!(!terminal_matches(nested_intersection.clone(), b"a"));
        assert!(terminal_matches(nested_intersection.clone(), b"b"));
        assert!(!terminal_matches(nested_intersection.clone(), b"c"));

        let expr = Expr::Exclude {
            expr: Box::new(byte_choice(b"bc")),
            exclude: Box::new(nested_intersection),
        };

        assert!(!terminal_matches(expr.clone(), b"a"));
        assert!(!terminal_matches(expr.clone(), b"b"));
        assert!(terminal_matches(expr, b"c"));
    }

    #[test]
    fn product_trace_residuals_keep_empty_complex_terminal_as_dead_coordinate() {
        let live = byte_choice(b"ab");
        let dead = Expr::Exclude {
            expr: Box::new(byte_choice(b"ab")),
            exclude: Box::new(byte_choice(b"ab")),
        };
        let exprs = vec![live, dead];
        let partitions = vec![0u32, 0u32];
        let retained = Arc::from(exprs.clone().into_boxed_slice());
        let tokenizer = super::partition_runtime::build_partitioned_tokenizer_with_product_trace_terminal_residuals(
            &exprs,
            None,
            &partitions,
            None,
            retained,
            Some(false),
            true,
        )
        .expect("empty complex terminal should not force product-trace fallback");

        for input in [b"a".as_slice(), b"b".as_slice()] {
            let exec = tokenizer.execute_from_state(input, tokenizer.initial_state());
            assert!(exec.matches.iter().any(|matched| matched.id == 0));
            assert!(!exec.matches.iter().any(|matched| matched.id == 1));
        }

        let coordinates = tokenizer
            .terminal_residual_coordinates()
            .expect("product-trace tokenizer should retain terminal residual coordinates");
        let dead_dfa = coordinates
            .terminal_dfa(1)
            .expect("dead terminal should retain a residual DFA");
        assert_eq!(dead_dfa.num_states(), 1);
        assert!(dead_dfa.finalizers(0).is_empty());
        assert!(dead_dfa.possible_future_group_ids(0).is_empty());
        for state in 0..tokenizer.num_states() {
            assert!(
                coordinates
                    .row(state)
                    .expect("coordinate row must exist")
                    .iter()
                    .all(|&(terminal, _)| terminal != 1),
                "dead terminal unexpectedly appears in residual row {state}",
            );
        }
    }

    #[test]
    fn product_trace_retains_finite_exclusion_continuation_certificate() {
        let open = Expr::Seq(vec![
            Expr::U8Seq(vec![b'"']),
            Expr::Repeat {
                expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"ab"))),
                min: 0,
                max: None,
            },
            Expr::U8Seq(b"\": ".to_vec()),
        ]);
        let excluded = Expr::U8Seq(b"\"ab\": ".to_vec());
        let exprs = vec![
            Expr::Exclude {
                expr: Box::new(open),
                exclude: Box::new(excluded),
            },
            Expr::U8Seq(b"x".to_vec()),
        ];
        let partitions = vec![0u32, 0u32];
        let tokenizer = super::partition_runtime::build_partitioned_tokenizer_with_product_trace_terminal_residuals(
            &exprs,
            None,
            &partitions,
            None,
            Arc::from(exprs.clone().into_boxed_slice()),
            Some(false),
            false,
        )
        .expect("finite exclusion should retain product-trace coordinates");

        let certificates = (0..tokenizer.num_states())
            .filter_map(|state| tokenizer.terminal_exclusion_continuation(state, 0))
            .collect::<Vec<_>>();
        assert!(!certificates.is_empty(), "expected exclusion continuation metadata");
        assert!(
            certificates
                .iter()
                .any(|certificate| certificate.right_max_remaining == Some(6)),
            "the initial finite exclusion should have an exact six-byte remaining bound: {certificates:?}",
        );
        assert!(
            certificates.iter().any(|certificate| certificate.right_state.is_none()),
            "a non-matching open-name prefix should kill the finite exclusion while LEFT stays live: {certificates:?}",
        );
        assert!(certificates.iter().all(|certificate| {
            certificate
                .right_max_remaining
                .is_none_or(|remaining| remaining <= 6)
        }));
        assert!(tokenizer.terminal_exclusion_left_dfa(0).is_some());
    }

    #[test]
    fn standalone_exact_repeat_matches_only_at_full_length() {
        let expr = Expr::Repeat {
            expr: Box::new(Expr::U8Class(U8Set::single(b' '))),
            min: 16,
            max: Some(16),
        };
        let regex = build_regex(std::slice::from_ref(&expr));
        let tokenizer = Tokenizer {
            dfa: regex.dfa,
            num_terminals: 1,
            packed_runtime_transitions: None,
            packed_runtime_transition_segments: Arc::from([]),
            compressed_transition_segments: Arc::from([]),
            packed_runtime_metadata: None,
            packed_runtime_metadata_segments: Arc::from([]),
            packed_compressed_transition_segments: Arc::from([]),
            virtual_unit_repeat: None,
            virtual_repeat_intersections: Vec::new(),
            virtual_residuals: Vec::new(),
            exprs: Some(Arc::from(vec![expr].into_boxed_slice())),
            terminal_residual_coordinates: None,
            singleton_epsilon_closures: std::sync::OnceLock::new(),
            matched_terminals_cache: std::sync::OnceLock::new(),
            initial_byte_frontiers: std::sync::OnceLock::new(),
            all_self_loop_bytes_cache: std::sync::OnceLock::new(),
            transition_count_cache: std::sync::OnceLock::new(),
            forced_minimized_state_count_cache: std::sync::OnceLock::new(),
            scalar_deterministic_dispatch_cache: std::sync::OnceLock::new(),
            sorted_dispatch_roots_cache: std::sync::OnceLock::new(),
            state_first_bytes_cache: std::sync::OnceLock::new(),
        };

        for len in [1usize, 2, 15] {
            let input = vec![b' '; len];
            let exec = tokenizer.execute_from_state(&input, tokenizer.initial_state());
            assert!(
                !exec.matches.iter().any(|matched| matched.id == 0),
                "exact repeat matched too early at len {len}: {:?}",
                exec.matches,
            );
        }

        let input = vec![b' '; 16];
        let exec = tokenizer.execute_from_state(&input, tokenizer.initial_state());
        assert!(
            exec.matches.iter().any(|matched| matched.id == 0 && matched.width == 16),
            "exact repeat did not match at len 16: {:?}",
            exec.matches,
        );
    }

    #[test]
    fn product_exact_repeat_matches_only_at_full_length() {
        let space = Expr::U8Class(U8Set::single(b' '));
        let exact_repeat = Expr::Repeat {
            expr: Box::new(Expr::U8Class(U8Set::single(b' '))),
            min: 16,
            max: Some(16),
        };

        let regex = build_regex(&[space.clone(), exact_repeat.clone()]);
        let tokenizer = Tokenizer {
            dfa: regex.dfa,
            num_terminals: 2,
            packed_runtime_transitions: None,
            packed_runtime_transition_segments: Arc::from([]),
            compressed_transition_segments: Arc::from([]),
            packed_runtime_metadata: None,
            packed_runtime_metadata_segments: Arc::from([]),
            packed_compressed_transition_segments: Arc::from([]),
            virtual_unit_repeat: None,
            virtual_repeat_intersections: Vec::new(),
            virtual_residuals: Vec::new(),
            exprs: Some(Arc::from(vec![space, exact_repeat].into_boxed_slice())),
            terminal_residual_coordinates: None,
            singleton_epsilon_closures: std::sync::OnceLock::new(),
            matched_terminals_cache: std::sync::OnceLock::new(),
            initial_byte_frontiers: std::sync::OnceLock::new(),
            all_self_loop_bytes_cache: std::sync::OnceLock::new(),
            transition_count_cache: std::sync::OnceLock::new(),
            forced_minimized_state_count_cache: std::sync::OnceLock::new(),
            scalar_deterministic_dispatch_cache: std::sync::OnceLock::new(),
            sorted_dispatch_roots_cache: std::sync::OnceLock::new(),
            state_first_bytes_cache: std::sync::OnceLock::new(),
        };

        for len in [1usize, 2, 15] {
            let input = vec![b' '; len];
            let exec = tokenizer.execute_from_state(&input, tokenizer.initial_state());
            assert!(
                !exec.matches.iter().any(|matched| matched.id == 1),
                "product exact repeat matched too early at len {len}: {:?}",
                exec.matches,
            );
        }

        let input = vec![b' '; 16];
        let exec = tokenizer.execute_from_state(&input, tokenizer.initial_state());
        assert!(
            exec.matches.iter().any(|matched| matched.id == 1 && matched.width == 16),
            "product exact repeat did not match at len 16: {:?}",
            exec.matches,
        );
    }

    #[test]
    fn product_vbr_exact_repeat_matches_only_at_full_length() {
        let space = Expr::U8Class(U8Set::single(b' '));
        let exact_repeat = Expr::Repeat {
            expr: Box::new(Expr::U8Class(U8Set::single(b' '))),
            min: 32,
            max: Some(32),
        };

        let regex = build_regex(&[space.clone(), exact_repeat.clone()]);
        let tokenizer = Tokenizer {
            dfa: regex.dfa,
            num_terminals: 2,
            packed_runtime_transitions: None,
            packed_runtime_transition_segments: Arc::from([]),
            compressed_transition_segments: Arc::from([]),
            packed_runtime_metadata: None,
            packed_runtime_metadata_segments: Arc::from([]),
            packed_compressed_transition_segments: Arc::from([]),
            virtual_unit_repeat: None,
            virtual_repeat_intersections: Vec::new(),
            virtual_residuals: Vec::new(),
            exprs: Some(Arc::from(vec![space, exact_repeat].into_boxed_slice())),
            terminal_residual_coordinates: None,
            singleton_epsilon_closures: std::sync::OnceLock::new(),
            matched_terminals_cache: std::sync::OnceLock::new(),
            initial_byte_frontiers: std::sync::OnceLock::new(),
            all_self_loop_bytes_cache: std::sync::OnceLock::new(),
            transition_count_cache: std::sync::OnceLock::new(),
            forced_minimized_state_count_cache: std::sync::OnceLock::new(),
            scalar_deterministic_dispatch_cache: std::sync::OnceLock::new(),
            sorted_dispatch_roots_cache: std::sync::OnceLock::new(),
            state_first_bytes_cache: std::sync::OnceLock::new(),
        };

        for len in [1usize, 2, 31] {
            let input = vec![b' '; len];
            let exec = tokenizer.execute_from_state(&input, tokenizer.initial_state());
            assert!(
                !exec.matches.iter().any(|matched| matched.id == 1),
                "product VBR exact repeat matched too early at len {len}: {:?}",
                exec.matches,
            );
        }

        let input = vec![b' '; 32];
        let exec = tokenizer.execute_from_state(&input, tokenizer.initial_state());
        assert!(
            exec.matches.iter().any(|matched| matched.id == 1 && matched.width == 32),
            "product VBR exact repeat did not match at len 32: {:?}",
            exec.matches,
        );
    }

    #[test]
    fn glrm_chunk16_terminal_family_keeps_exact_repeat_nonfinal_until_16() {
        let space = Expr::U8Class(U8Set::single(b' '));
        let quote = Expr::U8Seq(vec![b'"']);
        let exact_16 = Expr::Repeat {
            expr: Box::new(Expr::U8Class(U8Set::single(b' '))),
            min: 16,
            max: Some(16),
        };
        let upto_16 = Expr::Repeat {
            expr: Box::new(Expr::U8Class(U8Set::single(b' '))),
            min: 0,
            max: Some(16),
        };
        let upto_close_16 = Expr::Seq(vec![upto_16.clone(), quote.clone()]);
        let upto_3 = Expr::Repeat {
            expr: Box::new(Expr::U8Class(U8Set::single(b' '))),
            min: 0,
            max: Some(3),
        };
        let upto_close_3 = Expr::Seq(vec![upto_3.clone(), quote.clone()]);

        let exprs = vec![
            space.clone(),
            exact_16.clone(),
            upto_16,
            upto_close_16,
            upto_3,
            upto_close_3,
            quote,
        ];
        let regex = build_regex(&exprs);
        let tokenizer = Tokenizer {
            dfa: regex.dfa,
            num_terminals: exprs.len() as u32,
            packed_runtime_transitions: None,
            packed_runtime_transition_segments: Arc::from([]),
            compressed_transition_segments: Arc::from([]),
            packed_runtime_metadata: None,
            packed_runtime_metadata_segments: Arc::from([]),
            packed_compressed_transition_segments: Arc::from([]),
            virtual_unit_repeat: None,
            virtual_repeat_intersections: Vec::new(),
            virtual_residuals: Vec::new(),
            exprs: Some(Arc::from(exprs.into_boxed_slice())),
            terminal_residual_coordinates: None,
            singleton_epsilon_closures: std::sync::OnceLock::new(),
            matched_terminals_cache: std::sync::OnceLock::new(),
            initial_byte_frontiers: std::sync::OnceLock::new(),
            all_self_loop_bytes_cache: std::sync::OnceLock::new(),
            transition_count_cache: std::sync::OnceLock::new(),
            forced_minimized_state_count_cache: std::sync::OnceLock::new(),
            scalar_deterministic_dispatch_cache: std::sync::OnceLock::new(),
            sorted_dispatch_roots_cache: std::sync::OnceLock::new(),
            state_first_bytes_cache: std::sync::OnceLock::new(),
        };

        for len in [1usize, 2, 15] {
            let input = vec![b' '; len];
            let exec = tokenizer.execute_from_state(&input, tokenizer.initial_state());
            assert!(
                !exec.matches.iter().any(|matched| matched.id == 1),
                "GLRM family exact repeat matched too early at len {len}: {:?}",
                exec.matches,
            );
        }

        let input = vec![b' '; 16];
        let exec = tokenizer.execute_from_state(&input, tokenizer.initial_state());
        assert!(
            exec.matches.iter().any(|matched| matched.id == 1 && matched.width == 16),
            "GLRM family exact repeat did not match at len 16: {:?}",
            exec.matches,
        );
    }

    #[test]
    fn multi_dfa_literal_choice_direct_matches_generic_language() {
        let arm_a = Arc::new(super::nfa::compile_expr_to_dfa(&Expr::Choice(vec![
            Expr::U8Seq(b"alpha".to_vec()),
            Expr::U8Seq(b"alpine".to_vec()),
            Expr::U8Seq(b"amber".to_vec()),
        ])));
        let arm_b = Arc::new(super::nfa::compile_expr_to_dfa(&Expr::Choice(vec![
            Expr::U8Seq(b"beta".to_vec()),
            Expr::U8Seq(b"better".to_vec()),
            Expr::U8Seq(b"binary".to_vec()),
        ])));
        let arm_c = Arc::new(super::nfa::compile_expr_to_dfa(&Expr::Repeat {
            expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"xyz"))),
            min: 2,
            max: Some(5),
        }));
        let expression = Expr::Choice(vec![
            Expr::Dfa(arm_a),
            Expr::U8Seq(b"anchor".to_vec()),
            Expr::U8Seq(b"alphabet".to_vec()),
            Expr::Dfa(arm_b),
            Expr::U8Seq(b"zebra".to_vec()),
            Expr::Dfa(arm_c),
        ]);
        let direct = super::component::build_dfas_with_literal_choice(&expression)
            .expect("multi-DFA plus literal choice should compile directly");
        let generic = super::nfa::compile_expr_to_dfa(&expression);
        assert_dfa_observation_equivalent(&direct, &generic);
    }

    #[test]
    fn product_vbr_with_literal_prefix_and_no_suffix_uses_direct_path() {
        let expression = Expr::Seq(vec![
            Expr::U8Seq(b"pre".to_vec()),
            Expr::Repeat {
                expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"ab"))),
                min: 2,
                max: Some(5),
            },
        ]);

        let direct = super::component::compile_product_component_dfa_direct(&expression)
            .expect("literal prefix followed by bounded repeat should compile directly")
            .0;
        let generic = super::component::compile_product_component_dfa(&expression);
        assert_dfa_observation_equivalent(&direct, &generic);
        assert_eq!(direct.num_states(), 3 + (5 + 1) * 2);
    }

    #[test]
    fn repeated_bounded_repeat_bodies_share_one_local_base_dfa() {
        let body = Expr::Seq(vec![
            Expr::U8Class(U8Set::from_bytes(b"ab")),
            Expr::U8Class(U8Set::from_bytes(b"cd")),
        ]);
        let expressions = vec![
            Expr::Repeat {
                expr: Box::new(body.clone()),
                min: 1,
                max: Some(4),
            },
            Expr::Seq(vec![
                Expr::U8Seq(b"pre".to_vec()),
                Expr::Repeat {
                    expr: Box::new(body.clone()),
                    min: 2,
                    max: Some(5),
                },
            ]),
        ];
        let expression_refs = expressions.iter().collect::<Vec<_>>();
        let cache = super::product::build_repeat_base_dfa_cache(&expression_refs, false);
        assert_eq!(cache.len(), 1);
        assert!(cache.contains_key(&body));

        for expression in &expressions {
            let cached = super::component::compile_product_component_materialized_dfa_with_options_and_cache(
                expression,
                false,
                Some(&cache),
            );
            let uncached =
                super::component::compile_product_component_materialized_dfa_with_options(expression, false);
            assert_dfa_observation_equivalent(&cached, &uncached);
        }
    }

    #[test]
    fn product_vbr_with_literal_prefix_uses_direct_bounded_repeat_tail() {
        let quote = Expr::U8Seq(vec![b'"']);
        let spaces = Expr::Repeat {
            expr: Box::new(Expr::U8Class(U8Set::single(b' '))),
            min: 0,
            max: Some(32),
        };
        let expr = Expr::Seq(vec![quote.clone(), spaces, quote]);

        let Some((dfa, _)) = super::component::compile_product_component_dfa_direct(&expr) else {
            panic!("prefixed bounded repeat did not use direct product component path");
        };
        assert!(
            dfa.num_states() <= 80,
            "direct prefixed bounded repeat DFA unexpectedly large: {} states",
            dfa.num_states(),
        );

        let tokenizer = Tokenizer {
            dfa,
            num_terminals: 1,
            packed_runtime_transitions: None,
            packed_runtime_transition_segments: Arc::from([]),
            compressed_transition_segments: Arc::from([]),
            packed_runtime_metadata: None,
            packed_runtime_metadata_segments: Arc::from([]),
            packed_compressed_transition_segments: Arc::from([]),
            virtual_unit_repeat: None,
            virtual_repeat_intersections: Vec::new(),
            virtual_residuals: Vec::new(),
            exprs: Some(Arc::from(vec![expr].into_boxed_slice())),
            terminal_residual_coordinates: None,
            singleton_epsilon_closures: std::sync::OnceLock::new(),
            matched_terminals_cache: std::sync::OnceLock::new(),
            initial_byte_frontiers: std::sync::OnceLock::new(),
            all_self_loop_bytes_cache: std::sync::OnceLock::new(),
            transition_count_cache: std::sync::OnceLock::new(),
            forced_minimized_state_count_cache: std::sync::OnceLock::new(),
            scalar_deterministic_dispatch_cache: std::sync::OnceLock::new(),
            sorted_dispatch_roots_cache: std::sync::OnceLock::new(),
            state_first_bytes_cache: std::sync::OnceLock::new(),
        };

        for len in [0usize, 1, 31, 32] {
            let mut input = Vec::with_capacity(len + 2);
            input.push(b'"');
            input.extend(std::iter::repeat(b' ').take(len));
            input.push(b'"');
            let exec = tokenizer.execute_from_state(&input, tokenizer.initial_state());
            assert!(
                exec.matches
                    .iter()
                    .any(|matched| matched.id == 0 && matched.width == input.len()),
                "prefixed bounded repeat did not match length {len}: {:?}",
                exec.matches,
            );
        }
    }

    #[test]
    fn product_vbr_with_literal_prefix_and_regex_suffix_matches() {
        let quote = Expr::U8Seq(vec![b'"']);
        let word = Expr::U8Class(U8Set::single(b'a'));
        let space = Expr::U8Class(U8Set::single(b' '));
        let word_run = Expr::Repeat {
            expr: Box::new(word.clone()),
            min: 1,
            max: None,
        };
        let space_run = Expr::Repeat {
            expr: Box::new(space),
            min: 1,
            max: None,
        };
        let pair = Expr::Seq(vec![word_run.clone(), space_run]);
        let repeated_pairs = Expr::Repeat {
            expr: Box::new(pair),
            min: 0,
            max: Some(49),
        };
        let expr = Expr::Seq(vec![quote, repeated_pairs, word_run]);

        let Some((dfa, _)) = super::component::compile_product_component_dfa_direct(&expr) else {
            panic!("prefixed bounded repeat with regex suffix did not use direct path");
        };
        assert!(
            dfa.num_states() <= 400,
            "direct prefixed bounded repeat with regex suffix unexpectedly large: {} states",
            dfa.num_states(),
        );

        let tokenizer = Tokenizer {
            dfa,
            num_terminals: 1,
            packed_runtime_transitions: None,
            packed_runtime_transition_segments: Arc::from([]),
            compressed_transition_segments: Arc::from([]),
            packed_runtime_metadata: None,
            packed_runtime_metadata_segments: Arc::from([]),
            packed_compressed_transition_segments: Arc::from([]),
            virtual_unit_repeat: None,
            virtual_repeat_intersections: Vec::new(),
            virtual_residuals: Vec::new(),
            exprs: Some(Arc::from(vec![expr].into_boxed_slice())),
            terminal_residual_coordinates: None,
            singleton_epsilon_closures: std::sync::OnceLock::new(),
            matched_terminals_cache: std::sync::OnceLock::new(),
            initial_byte_frontiers: std::sync::OnceLock::new(),
            all_self_loop_bytes_cache: std::sync::OnceLock::new(),
            transition_count_cache: std::sync::OnceLock::new(),
            forced_minimized_state_count_cache: std::sync::OnceLock::new(),
            scalar_deterministic_dispatch_cache: std::sync::OnceLock::new(),
            sorted_dispatch_roots_cache: std::sync::OnceLock::new(),
            state_first_bytes_cache: std::sync::OnceLock::new(),
        };

        for input in [b"\"a".as_slice(), b"\"aa", b"\"a a", b"\"aa  aaa"] {
            let exec = tokenizer.execute_from_state(input, tokenizer.initial_state());
            assert!(
                exec.matches
                    .iter()
                    .any(|matched| matched.id == 0 && matched.width == input.len()),
                "prefixed bounded repeat with suffix did not match {:?}: {:?}",
                std::str::from_utf8(input).unwrap(),
                exec.matches,
            );
        }

        let exec = tokenizer.execute_from_state(b"\"a ", tokenizer.initial_state());
        assert!(
            !exec
                .matches
                .iter()
                .any(|matched| matched.id == 0 && matched.width == 3),
            "prefixed bounded repeat with suffix matched trailing space: {:?}",
            exec.matches,
        );
    }

    #[test]
    fn prefixed_bounded_repeat_with_regex_suffix_uses_direct_path_without_repeat_cutoff() {
        let quote = Expr::U8Seq(vec![b'"']);
        let word = Expr::U8Class(U8Set::single(b'a'));
        let space = Expr::U8Class(U8Set::single(b' '));
        let word_run = Expr::Repeat {
            expr: Box::new(word.clone()),
            min: 1,
            max: None,
        };
        let space_run = Expr::Repeat {
            expr: Box::new(space),
            min: 1,
            max: None,
        };
        let pair = Expr::Seq(vec![word_run.clone(), space_run]);
        let repeated_pairs = Expr::Repeat {
            expr: Box::new(pair),
            min: 0,
            max: Some(29),
        };
        let expr = Expr::Seq(vec![quote, repeated_pairs, word_run]);

        let Some((dfa, _)) = super::component::compile_product_component_dfa_direct(&expr) else {
            panic!("prefixed bounded repeat with regex suffix did not use direct path");
        };
        assert!(
            dfa.num_states() <= 300,
            "direct prefixed bounded repeat with regex suffix unexpectedly large: {} states",
            dfa.num_states(),
        );

        let repeated_pairs = Expr::Repeat {
            expr: Box::new(Expr::Seq(vec![
                Expr::Repeat {
                    expr: Box::new(Expr::U8Class(U8Set::single(b'a'))),
                    min: 1,
                    max: None,
                },
                Expr::Repeat {
                    expr: Box::new(Expr::U8Class(U8Set::single(b' '))),
                    min: 1,
                    max: None,
                },
            ])),
            min: 0,
            max: Some(2),
        };
        let expr = Expr::Seq(vec![
            Expr::U8Seq(vec![b'"']),
            repeated_pairs,
            Expr::Repeat {
                expr: Box::new(Expr::U8Class(U8Set::single(b'a'))),
                min: 1,
                max: None,
            },
        ]);
        let Some((dfa, _)) = super::component::compile_product_component_dfa_direct(&expr) else {
            panic!("small prefixed bounded repeat with regex suffix did not use direct path");
        };
        assert!(
            dfa.num_states() <= 40,
            "small direct prefixed bounded repeat with regex suffix unexpectedly large: {} states",
            dfa.num_states(),
        );
    }

    fn prefixed_optional_word_list_expr(max_pairs: usize) -> Expr {
        let nonspace_plus = Expr::Repeat {
            expr: Box::new(Expr::U8Class(U8Set::single(b'a'))),
            min: 1,
            max: None,
        };
        let space_plus = Expr::Repeat {
            expr: Box::new(Expr::U8Class(U8Set::single(b' '))),
            min: 1,
            max: None,
        };
        let body = Expr::Seq(vec![nonspace_plus.clone(), space_plus]);
        let repeated = Expr::Repeat {
            expr: Box::new(body),
            min: 0,
            max: Some(max_pairs),
        };

        Expr::Seq(vec![
            Expr::U8Seq(vec![b'"']),
            Expr::Choice(vec![
                Expr::Epsilon,
                Expr::Seq(vec![repeated, nonspace_plus]),
            ]),
        ])
    }

    #[test]
    fn prefixed_optional_choice_uses_direct_component_path_for_bounded_repeat_suffix() {
        let expr = prefixed_optional_word_list_expr(199);

        let Some((dfa, _)) = compile_product_component_dfa_direct(&expr) else {
            panic!("prefixed optional wrapper did not use direct product component path");
        };

        assert!(
            dfa.num_states() < 10_000,
            "prefixed optional direct-path DFA unexpectedly large: {} states",
            dfa.num_states(),
        );
        assert!(dfa.finalizers(1).contains(0));
    }

    #[test]
    fn prefixed_optional_word_list_semantics() {
        let expr = prefixed_optional_word_list_expr(2);
        let regex = build_regex(std::slice::from_ref(&expr));
        let tokenizer = Tokenizer {
            dfa: regex.dfa,
            num_terminals: 1,
            packed_runtime_transitions: None,
            packed_runtime_transition_segments: Arc::from([]),
            compressed_transition_segments: Arc::from([]),
            packed_runtime_metadata: None,
            packed_runtime_metadata_segments: Arc::from([]),
            packed_compressed_transition_segments: Arc::from([]),
            virtual_unit_repeat: None,
            virtual_repeat_intersections: Vec::new(),
            virtual_residuals: Vec::new(),
            exprs: Some(Arc::from(vec![expr].into_boxed_slice())),
            terminal_residual_coordinates: None,
            singleton_epsilon_closures: std::sync::OnceLock::new(),
            matched_terminals_cache: std::sync::OnceLock::new(),
            initial_byte_frontiers: std::sync::OnceLock::new(),
            all_self_loop_bytes_cache: std::sync::OnceLock::new(),
            transition_count_cache: std::sync::OnceLock::new(),
            forced_minimized_state_count_cache: std::sync::OnceLock::new(),
            scalar_deterministic_dispatch_cache: std::sync::OnceLock::new(),
            sorted_dispatch_roots_cache: std::sync::OnceLock::new(),
            state_first_bytes_cache: std::sync::OnceLock::new(),
        };

        for input in [b"\"".as_slice(), b"\"a", b"\"a a", b"\"a  a"] {
            let exec = tokenizer.execute_from_state(input, tokenizer.initial_state());
            assert!(
                exec.matches
                    .iter()
                    .any(|matched| matched.id == 0 && matched.width == input.len()),
                "prefixed optional word-list did not match {:?}: {:?}",
                std::str::from_utf8(input).unwrap(),
                exec.matches,
            );
        }

        let exec = tokenizer.execute_from_state(b"\" a", tokenizer.initial_state());
        assert!(
            !exec
                .matches
                .iter()
                .any(|matched| matched.id == 0 && matched.width == 3),
            "prefixed optional word-list matched leading space unexpectedly: {:?}",
            exec.matches,
        );
    }

    #[test]
    fn prefixed_optional_word_list_with_literal_suffix_uses_direct_exact_path() {
        let base = prefixed_optional_word_list_expr(2);
        let Expr::Seq(mut parts) = base else { unreachable!() };
        parts.push(Expr::U8Seq(vec![b'"']));
        let expr = Expr::Seq(parts);

        let Some((direct, _)) = compile_product_component_dfa_direct(&expr) else {
            panic!("optional-middle bounded repeat did not use direct component path");
        };
        let generic = super::compile_single_expr_dfa(&expr);
        let samples: &[&[u8]] = &[
            br#""""#,
            br#""a""#,
            br#""a a""#,
            br#""a  a""#,
            br#"" a""#,
            br#""a a a""#,
            br#""a a a a""#,
        ];
        for input in samples {
            assert_eq!(
                dfa_accepts(&direct, input),
                dfa_accepts(&generic, input),
                "direct optional-middle path changed language for {input:?}",
            );
        }
        assert!(dfa_accepts(&direct, br#""""#));
        assert!(dfa_accepts(&direct, br#""a""#));
        assert!(!dfa_accepts(&direct, br#"" a""#));
    }

    #[test]
    fn finite_literal_sequence_choice_trie_matches_generic_compilation() {
        let expr = Expr::Seq(vec![
            Expr::U8Seq(b"<".to_vec()),
            Expr::Choice(vec![
                Expr::U8Seq(b"a".to_vec()),
                Expr::Seq(vec![Expr::U8Seq(b"a".to_vec()), Expr::U8Seq(b"b".to_vec())]),
                Expr::U8Seq(b"b".to_vec()),
            ]),
            Expr::Choice(vec![Expr::Epsilon, Expr::U8Seq(b">".to_vec())]),
        ]);

        let direct = super::component::build_finite_literal_language_dfa(&expr)
            .expect("finite concatenated literal language should compile as a trie");
        let generic = super::compile_single_expr_dfa(&expr);

        for input in enumerate_inputs(b"<>abx", 4) {
            assert_eq!(
                dfa_state_observation(&direct, 0, &input),
                dfa_state_observation(&generic, 0, &input),
                "finite-literal trie changed DFA semantics for input {input:?}",
            );
        }
    }

    #[test]
    fn direct_tokenizer_states_for_o35155_regex_groups() {
        let expr_1 = parse_regex(r"(\w+\.)+\d+", false);
        let expr_2 = parse_regex(r"\w+_(\w_)?\d+", false);
        let expr_3 = parse_regex(r"(\w|-){12}", false);
        let expr_5 = parse_regex(r"\d{7,9}", false);

        let regex_1235 = build_regex(&[
            expr_1.clone(),
            expr_2.clone(),
            expr_3.clone(),
            expr_5.clone(),
        ]);
        let regex_125 = build_regex(&[expr_1, expr_2, expr_5]);

        eprintln!(
            "o35155 direct tokenizer states for regex groups 1,2,3,5: states={} transitions={}",
            regex_1235.num_states(),
            regex_1235.num_transitions()
        );
        eprintln!(
            "o35155 direct tokenizer states for regex groups 1,2,5: states={} transitions={}",
            regex_125.num_states(),
            regex_125.num_transitions()
        );
    }

    #[test]
    fn bounded_repeat_regex_suffix_must_fork_at_ambiguous_boundary() {
        // ("a"+)? "a" matches "aa":
        //
        //   optional body "a"+ consumes the first "a"
        //   suffix "a" consumes the second "a"
        //
        // The regex-suffix fast path used to greedily continue the body on the
        // second "a" and drop the valid suffix path.
        let expr = Expr::Seq(vec![
            Expr::Repeat {
                expr: Box::new(Expr::Repeat {
                    expr: Box::new(Expr::U8Seq(vec![b'a'])),
                    min: 1,
                    max: None,
                }),
                min: 0,
                max: Some(1),
            },
            Expr::U8Seq(vec![b'a']),
        ]);
        assert!(terminal_matches(expr, b"aa"));
    }
    #[test]
    fn bounded_repeat_regex_suffix_nullable_class_suffix_finalizes_after_body() {
        // "a" [b]? matches both "a" and "ab".
        //
        // The regex-suffix fast path used to miss the body/suffix boundary at
        // end-of-input when the suffix was nullable.
        let expr = Expr::Seq(vec![
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(vec![b'a'])),
                min: 1,
                max: Some(1),
            },
            Expr::Choice(vec![
                Expr::Epsilon,
                Expr::U8Class(U8Set::single(b'b')),
            ]),
        ]);
        assert!(terminal_matches(expr.clone(), b"a"));
        assert!(terminal_matches(expr, b"ab"));
    }
    #[test]
    fn bounded_repeat_regex_suffix_nullable_suffix_after_optional_body() {
        // ("a")? [b]? matches "a", "b", and "ab".
        //
        // Do not assert the empty string here: zero-length terminals are not a
        // useful lexer regression target and may be intentionally unsupported by
        // terminal_matches / terminal DFA metadata. This test is about preserving
        // non-empty matches when both the repeated body and regex suffix are
        // nullable.
        let expr = Expr::Seq(vec![
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(vec![b'a'])),
                min: 0,
                max: Some(1),
            },
            Expr::Choice(vec![
                Expr::Epsilon,
                Expr::U8Class(U8Set::single(b'b')),
            ]),
        ]);
        assert!(terminal_matches(expr.clone(), b"a"));
        assert!(terminal_matches(expr.clone(), b"b"));
        assert!(terminal_matches(expr, b"ab"));
    }
    #[test]
    fn bounded_repeat_regex_suffix_zero_max_must_not_consume_body() {
        // "a"{0,0} [b] is just [b]. It must not match "ab".
        //
        // The regex-suffix fast path used to start with a live body state even when
        // max == 0.
        let expr = Expr::Seq(vec![
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(vec![b'a'])),
                min: 0,
                max: Some(0),
            },
            Expr::U8Class(U8Set::single(b'b')),
        ]);
        assert!(terminal_matches(expr.clone(), b"b"));
        assert!(!terminal_matches(expr, b"ab"));
    }

    #[test]
    fn build_regex_defaults_to_one_monolithic_dfa() {
        let expressions = vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"ab".to_vec()),
        ];
        let tokenizer = build_regex(&expressions).into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
        assert!(!tokenizer.has_epsilon_transitions());
    }

    #[test]
    fn separate_terminal_partitions_preserve_multiple_live_end_states() {
        let expressions = vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                min: 1,
                max: None,
            },
        ];
        let tokenizer = build_regex_partitioned_with_adaptive(&expressions, &[0, 1], false)
            .into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );

        assert!(tokenizer.has_epsilon_transitions());
        let result = tokenizer.execute_from_state(b"a", tokenizer.initial_state());
        assert_eq!(result.end_state.len(), 2, "both terminal components must remain live");
        assert_eq!(
            result.matches.iter().map(|matched| matched.id).collect::<std::collections::BTreeSet<_>>(),
            std::collections::BTreeSet::from([0, 1]),
        );

        let continued = tokenizer.execute_from_state(b"a", result.end_state[1]);
        assert!(continued.matches.iter().any(|matched| matched.id == 1));
    }

    #[test]
    fn explicit_partition_ids_control_joint_determinization() {
        let expressions = vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"ab".to_vec()),
            Expr::U8Seq(b"z".to_vec()),
        ];
        let regex = super::build_regex_partitioned_with_adaptive(
            &expressions,
            &[7, 7, 9],
            false,
        );
        assert_eq!(
            regex.dfa.states()[0].epsilon_transitions.len(),
            2,
            "two declared partitions must produce two epsilon branches",
        );
        let tokenizer = regex.into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
        let result = tokenizer.execute_from_state(b"a", tokenizer.initial_state());
        assert!(result.matches.iter().any(|matched| matched.id == 0));
        assert!(!result.end_state.is_empty());
    }

    #[test]
    fn protected_residual_components_survive_adaptive_determinization() {
        let expressions = vec![
            Expr::U8Seq(b"same".to_vec()),
            Expr::U8Seq(b"same".to_vec()),
            Expr::U8Seq(b"ordinary-a".to_vec()),
            Expr::U8Seq(b"ordinary-b".to_vec()),
        ];
        let isolation = [Some(100), Some(101), None, None];
        let components = super::partition::compile_partition_components(
            &expressions,
            None,
            &[0, 1, 2, 3],
            Some(&isolation),
        );

        let output = super::partition::adaptively_determinize_components_with_limits(
            components,
            32_768,
            1_000,
            1_000,
            Some(1),
        );

        for protected_terminal in [0usize, 1] {
            let component = output
                .iter()
                .find(|component| component.terminal_ids.contains(&protected_terminal))
                .expect("protected terminal must remain present");
            assert!(component.protected_residual);
            assert_eq!(component.terminal_ids, vec![protected_terminal]);
        }
    }

    #[test]
    fn protected_identical_terminals_keep_independent_live_states() {
        let expressions = vec![
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                min: 1,
                max: Some(4),
            },
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                min: 1,
                max: Some(4),
            },
        ];
        let regex = super::build_regex_partitioned_with_options(&expressions, &[0, 1], super::PartitionOptions { residual_isolation_classes: Some(&[Some(200), Some(201)]), adaptive: Some(true), ..super::PartitionOptions::default() });
        assert_eq!(
            regex.dfa.states()[0].epsilon_transitions.len(),
            2,
            "protected coordinates must not collapse into one adaptive product",
        );
        let tokenizer = regex.into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
        let result = tokenizer.execute_from_state(b"a", tokenizer.initial_state());
        assert_eq!(result.end_state.len(), 2);
        assert_eq!(
            result
                .matches
                .iter()
                .map(|matched| matched.id)
                .collect::<std::collections::BTreeSet<_>>(),
            std::collections::BTreeSet::from([0, 1]),
        );
    }

    #[test]
    fn shared_nested_group_ops_are_initialized_before_parallel_partition_compile() {
        let shared_nested = Expr::Exclude {
            expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"abcdefghijklmnopqrstuvwxyz"))),
            exclude: Box::new(Expr::U8Seq(b"x".to_vec())),
        };
        let expressions = (0..64u8)
            .map(|suffix| {
                Expr::Seq(vec![
                    shared_nested.clone(),
                    Expr::U8Seq(vec![b'0' + suffix % 10]),
                ])
            })
            .collect::<Vec<_>>();
        let partitions = (0..expressions.len() as u32).collect::<Vec<_>>();
        let mut grouped = std::collections::BTreeMap::<u32, Vec<usize>>::new();
        for (terminal, &partition) in partitions.iter().enumerate() {
            grouped.entry(partition).or_default().push(terminal);
        }

        let shared = super::plan::shared_duplicate_nested_group_op_cache(&expressions, &grouped)
            .expect("the repeated nested exclusion should use the shared cache");
        super::plan::prewarm_shared_duplicate_nested_group_ops(&shared);
        assert!(shared.all_entries_initialized());

        let components =
            super::partition::compile_partition_components(&expressions, None, &partitions, None);
        assert_eq!(components.len(), expressions.len());
    }

    #[test]
    #[should_panic(
        expected = "shared nested group-op cache must be prewarmed before parallel compilation"
    )]
    fn shared_nested_group_ops_cannot_initialize_from_partition_workers() {
        let shared_nested = Expr::Exclude {
            expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"abcdefghijklmnopqrstuvwxyz"))),
            exclude: Box::new(Expr::U8Seq(b"x".to_vec())),
        };
        let expressions = (0..64u8)
            .map(|suffix| {
                Expr::Seq(vec![
                    shared_nested.clone(),
                    Expr::U8Seq(vec![b'0' + suffix % 10]),
                ])
            })
            .collect::<Vec<_>>();
        let partitions = (0..expressions.len() as u32).collect::<Vec<_>>();
        let mut grouped = std::collections::BTreeMap::<u32, Vec<usize>>::new();
        for (terminal, &partition) in partitions.iter().enumerate() {
            grouped.entry(partition).or_default().push(terminal);
        }
        let shared = super::plan::shared_duplicate_nested_group_op_cache(&expressions, &grouped)
            .expect("the repeated nested exclusion should use the shared cache");

        let _ = super::partition::compile_terminal_ids_with_shared_duplicate_cache(
            &expressions,
            None,
            &[0],
            Some(&shared),
        );
    }

    #[test]
    fn partitioned_union_transports_exact_possible_futures_without_recompute() {
        let expressions = vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"ab".to_vec()),
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"b".to_vec())),
                min: 1,
                max: None,
            },
            Expr::Epsilon,
        ];
        let regex = build_regex_partitioned_with_adaptive(
            &expressions,
            &[7, 7, 9, 11],
            false,
        );
        let exact = regex.dfa;
        let mut recomputed = exact.clone();
        recomputed.recompute_possible_futures();

        assert_eq!(
            exact, recomputed,
            "transported component futures and epsilon-root union must match the generic fixpoint",
        );
    }

    #[test]
    fn bounded_product_trial_stops_before_cross_pattern_blowup() {
        let expressions = vec![
            parse_regex(r"\w+_(\w_)?\d+", false),
            parse_regex(r"(\w|-){12}", false),
        ];
        let components =
            super::partition::compile_partition_components(&expressions, None, &[0, 1], None);
        assert!(
            try_product_union_components(&components, 32, usize::MAX, None).is_none(),
            "the bounded trial unexpectedly completed within 32 product states",
        );
    }

    #[test]
    fn adaptive_transition_growth_limit_is_inclusive() {
        assert!(super::settings::adaptive_transition_growth_is_acceptable(100, 600, 600));
        assert!(!super::settings::adaptive_transition_growth_is_acceptable(100, 601, 600));
    }

    #[test]
    fn adaptive_transition_growth_rejection_keeps_partition_components() {
        let expressions = vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"ab".to_vec()),
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"b".to_vec())),
                min: 1,
                max: None,
            },
        ];
        let components =
            super::partition::compile_partition_components(&expressions, None, &[0, 1, 2], None);
        let original_terminal_ids = components
            .iter()
            .map(|component| component.terminal_ids.clone())
            .collect::<Vec<_>>();

        let retained = super::partition::adaptively_determinize_components_with_limits(
            components,
            32_768,
            100,
            1,
            None,
        );

        assert_eq!(retained.len(), original_terminal_ids.len());
        assert_eq!(
            retained
                .iter()
                .map(|component| component.terminal_ids.clone())
                .collect::<Vec<_>>(),
            original_terminal_ids,
        );
    }

    #[test]
    fn bounded_adaptive_product_accounts_for_copied_component_states() {
        let expressions = vec![
            Expr::U8Seq(b"abcd".to_vec()),
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                min: 1,
                max: Some(8),
            },
        ];
        let components =
            super::partition::compile_partition_components(&expressions, None, &[0, 1], None);
        let untouched_states = 1usize
            + components
                .iter()
                .map(|component| component.dfa.num_states())
                .sum::<usize>();

        assert!(
            try_product_union_components(
                &components,
                untouched_states,
                usize::MAX,
                Some(1),
            )
            .is_none(),
            "a depth-one prefix plus copied suffix DFAs must not fit inside the untouched-union state count",
        );

        let expanded = try_product_union_components(
            &components,
            untouched_states.saturating_mul(2),
            usize::MAX,
            Some(1),
        )
        .expect("the bounded product should fit once explicit growth is allowed");
        assert!(
            expanded.num_states() > untouched_states,
            "this fixture should expose the bounded prefix + copied-component overhead",
        );
    }

    #[test]
    fn adaptive_bounded_depth_uses_additive_prefix_budget() {
        let expressions = vec![
            Expr::U8Seq(b"abc".to_vec()),
            Expr::U8Seq(b"abd".to_vec()),
        ];
        let components =
            super::partition::compile_partition_components(&expressions, None, &[0, 1], None);
        let untouched_states = 1usize
            + components
                .iter()
                .map(|component| component.dfa.num_states())
                .sum::<usize>();

        let bounded = super::partition::adaptively_determinize_components_with_limits(
            super::partition::compile_partition_components(&expressions, None, &[0, 1], None),
            32_768,
            100,
            600,
            Some(1),
        );
        assert_eq!(
            bounded.len(),
            1,
            "bounded depth should use its explicit additive prefix-state budget rather than the full-product percentage budget",
        );
        assert!(
            bounded[0].dfa.num_states()
                <= untouched_states + super::settings::adaptive_lexer_bounded_overhead_states(),
        );

        let full = super::partition::adaptively_determinize_components_with_limits(
            components,
            32_768,
            100,
            600,
            None,
        );
        let [combined] = full.as_slice() else {
            panic!("full mode should accept the compressing exact product");
        };
        assert!(
            combined.dfa.num_states() < untouched_states,
            "the exact product should be smaller than the untouched epsilon union for this shared-prefix fixture",
        );
        assert!(
            !combined.dfa.has_epsilon_transitions(),
            "the accepted exact product should remain deterministic",
        );
    }

    #[test]
    fn adaptive_policy_does_not_change_per_partition_compilation() {
        let expressions = vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"ab".to_vec()),
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"b".to_vec())),
                min: 1,
                max: None,
            },
            Expr::U8Seq(b"c".to_vec()),
        ];
        let partitions = [7, 7, 9, 11];
        let components =
            super::partition::compile_partition_components(&expressions, None, &partitions, None);
        let expected_terminal_ids = [vec![0, 1], vec![2], vec![3]];

        assert_eq!(components.len(), expected_terminal_ids.len());
        for (component, expected_ids) in components.iter().zip(expected_terminal_ids) {
            assert_eq!(component.terminal_ids, expected_ids);
            let expected = super::partition::compile_terminal_ids(&expressions, None, &expected_ids);
            assert_eq!(component.dfa, expected);
        }
    }

    #[test]
    fn adaptive_prefix_depth_preserves_partitioned_semantics() {
        let expressions = vec![
            Expr::U8Seq(b"abcd".to_vec()),
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                min: 1,
                max: Some(8),
            },
        ];
        let baseline = build_regex_partitioned_with_adaptive(&expressions, &[0, 1], false)
            .into_tokenizer(
                expressions.len() as u32,
                Some(Arc::from(expressions.clone().into_boxed_slice())),
            );
        let components =
            super::partition::compile_partition_components(&expressions, None, &[0, 1], None);
        let prefix_dfa = try_product_union_components(&components, 32_768, usize::MAX, Some(1))
            .expect("depth-one adaptive product should fit");
        assert!(prefix_dfa.states()[0].epsilon_transitions.is_empty());
        assert!(
            prefix_dfa.states()[0]
                .transitions
                .iter()
                .any(|(_, &target)| !prefix_dfa.states()[target as usize].epsilon_transitions.is_empty()),
            "the depth-one product should resume exact component states at its frontier",
        );
        let adaptive = super::Regex { dfa: prefix_dfa }.into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );

        for input in enumerate_inputs(b"abcdx", 8) {
            assert_eq!(
                tokenizer_observation(&adaptive, &input),
                tokenizer_observation(&baseline, &input),
                "depth-one adaptive prefix differed for input {input:?}",
            );
        }
    }

    #[test]
    fn adaptive_final_nfa_determinization_preserves_partitioned_semantics() {
        let expressions = vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"ab".to_vec()),
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                min: 1,
                max: None,
            },
        ];
        let singleton = build_regex_partitioned_with_adaptive(
            &expressions,
            &[0, 1, 2],
            false,
        )
        .into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.clone().into_boxed_slice())),
        );
        let adaptive = build_regex_partitioned_with_adaptive(
            &expressions,
            &[0, 1, 2],
            true,
        )
        .into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );

        for input in enumerate_inputs(b"ab", 4) {
            assert_eq!(
                tokenizer_observation(&adaptive, &input),
                tokenizer_observation(&singleton, &input),
                "adaptive partitioning differed for input {input:?}",
            );
        }
    }

    #[test]
    fn direct_trivial_product_components_match_generic_compilation() {
        let expressions = [
            Expr::U8Seq(b"literal".to_vec()),
            Expr::U8Seq(Vec::new()),
            Expr::U8Class(U8Set::from_bytes(b"abc")),
            Expr::Epsilon,
        ];

        for expr in expressions {
            let direct = compile_product_component_dfa_direct(&expr)
                .expect("trivial expression must have a direct DFA")
                .0;
            let generic = compile_product_component_dfa(&expr);
            assert_eq!(
                direct.group_id_to_u8set(0),
                generic.group_id_to_u8set(0),
                "group byte set differed for {expr:?}",
            );

            let mut inputs = vec![Vec::new()];
            match &expr {
                Expr::U8Seq(bytes) => {
                    for prefix_len in 0..=bytes.len() {
                        let prefix = bytes[..prefix_len].to_vec();
                        inputs.push(prefix.clone());
                        for byte in 0..=u8::MAX {
                            let mut deviated = prefix.clone();
                            deviated.push(byte);
                            inputs.push(deviated);
                        }
                    }
                }
                Expr::U8Class(_) => {
                    inputs.extend((0..=u8::MAX).map(|byte| vec![byte]));
                    inputs.extend((0..=u8::MAX).map(|byte| vec![byte, byte]));
                }
                Expr::Epsilon => inputs.push(vec![0]),
                _ => unreachable!(),
            }

            let observe = |dfa: DFA, input: &[u8]| {
                let tokenizer = Tokenizer::from_parts(dfa, 1, None);
                let states = tokenizer.run(input);
                let mut matched = BTreeSet::new();
                let mut future = BTreeSet::new();
                for state in states {
                    matched.extend(tokenizer.matched_terminals_iter(state));
                    future.extend(tokenizer.possible_future_terminals_iter(state));
                }
                (matched, future)
            };
            for input in inputs {
                assert_eq!(
                    observe(direct.clone(), &input),
                    observe(generic.clone(), &input),
                    "direct DFA behavior differed for {expr:?} on {input:?}",
                );
            }
        }
    }

    #[test]
    fn direct_fixed_sequence_component_matches_generic_compilation() {
        let expression = Expr::Seq(vec![
            Expr::U8Class(U8Set::from_bytes(b"+-")),
            Expr::Shared(Arc::new(Expr::U8Seq(b"ab".to_vec()))),
            Expr::Epsilon,
            Expr::U8Class(U8Set::from_bytes(b"xy")),
        ]);
        let direct = super::component::compile_product_component_dfa_direct(&expression)
            .expect("fixed sequence should compile directly")
            .0;
        let generic = super::component::compile_product_component_dfa(&expression);
        let direct = super::Regex { dfa: direct }.into_tokenizer(1, None);
        let generic = super::Regex { dfa: generic }.into_tokenizer(1, None);

        for input in enumerate_inputs(b"+-abxyz", 4) {
            assert_eq!(
                tokenizer_observation(&direct, &input),
                tokenizer_observation(&generic, &input),
                "direct fixed sequence differed for input {input:?}",
            );
        }
    }

    #[test]
    fn virtual_fixed_sequence_components_preserve_exact_product_layout() {
        let expressions = vec![
            Expr::Seq(vec![
                Expr::U8Class(U8Set::from_bytes(b"+-")),
                Expr::U8Seq(b"alpha".to_vec()),
            ]),
            Expr::U8Seq(b" beta".to_vec()),
            Expr::Seq(vec![
                Expr::U8Class(U8Set::from_bytes(b"xy")),
                Expr::U8Class(U8Set::from_bytes(b"12")),
            ]),
            Expr::Repeat {
                expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"ab"))),
                min: 1,
                max: Some(3),
            },
        ];
        let exclusions = BTreeMap::new();
        let intersections = BTreeMap::new();

        let baseline = super::product::build_product_dfa(
            &expressions,
            None,
            expressions.len(),
            &exclusions,
            &intersections,
            false,
            false,
            false,
            false,
        )
        .0;
        let virtualized = super::product::build_product_dfa(
            &expressions,
            None,
            expressions.len(),
            &exclusions,
            &intersections,
            false,
            true,
            false,
            false,
        )
        .0;

        assert_eq!(virtualized, baseline);
    }

    #[test]
    fn zero_min_repeat_suffix_dominance_matches_generic_ambiguous_boundaries() {
        let expressions = [
            Expr::Seq(vec![
                Expr::Repeat {
                    expr: Box::new(parse_regex("a+", false)),
                    min: 0,
                    max: Some(4),
                },
                parse_regex("a+", false),
            ]),
            Expr::Seq(vec![
                Expr::Repeat {
                    expr: Box::new(parse_regex("a+b+", false)),
                    min: 0,
                    max: Some(4),
                },
                parse_regex("a+", false),
            ]),
            Expr::Seq(vec![
                Expr::Repeat {
                    expr: Box::new(parse_regex("(?:ab|a)", false)),
                    min: 0,
                    max: Some(4),
                },
                parse_regex("b+", false),
            ]),
        ];

        for expression in expressions {
            let direct_dfa = compile_product_component_dfa_direct(&expression)
                .expect("zero-minimum bounded repeat with non-nullable suffix must compile directly")
                .0;
            let direct = super::Regex { dfa: direct_dfa }.into_tokenizer(
                1,
                Some(Arc::from(vec![expression.clone()].into_boxed_slice())),
            );

            let mut nfa = super::nfa::build_regex_nfa(std::slice::from_ref(&expression));
            nfa.condense_epsilon_sccs();
            let generic = super::Regex {
                dfa: nfa.to_minimized_dfa(),
            }
            .into_tokenizer(
                1,
                Some(Arc::from(vec![expression.clone()].into_boxed_slice())),
            );

            for input in enumerate_inputs(b"abx", 8) {
                assert_eq!(
                    tokenizer_observation(&direct, &input),
                    tokenizer_observation(&generic, &input),
                    "dominance quotient differed from generic determinization for expression {expression:?}, input {input:?}",
                );
            }
        }
    }

    #[test]
    fn generic_product_component_fallback_preserves_strict_future_metadata() {
        let expr = Expr::Repeat {
            expr: Box::new(Expr::Epsilon),
            min: 0,
            max: None,
        };
        let product = super::product::compile_product_component(&expr);
        let product = product.partition_dfa();
        let generic = compile_product_component_dfa(&expr);

        assert_eq!(product.finalizers(0), generic.finalizers(0));
        assert_eq!(
            product.possible_future_group_ids(0),
            generic.possible_future_group_ids(0),
        );
        assert!(product.finalizers(0).contains(0));
        assert!(
            !product.possible_future_group_ids(0).contains(0),
            "an epsilon-only terminal is final now but is not reachable after another byte",
        );
    }

    #[test]
    fn deferred_dense_binary_intersection_matches_eager_product() {
        let bounded_string = |max| {
            Expr::Seq(vec![
                Expr::U8Seq(b"\"".to_vec()),
                Expr::Repeat {
                    expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"ab"))),
                    // Keep this regression on the ordinary deferred-product
                    // implementation; zero-minimum repeat intersections have
                    // their own exact lazy-product regression above.
                    min: 1,
                    max: Some(max),
                },
                Expr::U8Seq(b"\"".to_vec()),
            ])
        };
        let expression = Expr::Intersect {
            expr: Box::new(bounded_string(31)),
            intersect: Box::new(bounded_string(17)),
        };

        let eager = super::compile_with_plan(super::plan::build_exclusion_compile_plan(
            std::slice::from_ref(&expression),
        ));
        let (mut deferred, trace) = match super::deferred::try_compile_with_plan_deferred_dense(
            super::plan::build_exclusion_compile_plan(std::slice::from_ref(&expression)),
        ) {
            Ok(prepared) => prepared,
            Err(_) => panic!("binary intersection should admit deferred dense construction"),
        };
        let trace = trace.expect("deferred construction must retain its state tuples");
        deferred
            .attach_dense_runtime_trace(trace)
            .expect("deferred construction must accept its runtime trace");
        let finished = deferred.finish();

        assert_eq!(finished, eager);
    }

    #[test]
    fn retained_dense_runtime_matches_eager_product() {
        let bounded_string = |body: Expr, max| {
            Expr::Seq(vec![
                Expr::U8Seq(b"\"".to_vec()),
                Expr::Repeat {
                    expr: Box::new(body),
                    min: 1,
                    max: Some(max),
                },
                Expr::U8Seq(b"\"".to_vec()),
            ])
        };
        let cases = [
            Expr::Intersect {
                expr: Box::new(bounded_string(
                    Expr::U8Class(U8Set::from_bytes(b"ab")),
                    31,
                )),
                intersect: Box::new(bounded_string(
                    Expr::U8Class(U8Set::from_bytes(b"ab")),
                    17,
                )),
            },
            Expr::Intersect {
                expr: Box::new(bounded_string(
                    Expr::Choice(vec![
                        Expr::U8Seq(b"a".to_vec()),
                        Expr::U8Seq(b"ab".to_vec()),
                    ]),
                    24,
                )),
                intersect: Box::new(bounded_string(
                    Expr::U8Class(U8Set::from_bytes(b"ab")),
                    19,
                )),
            },
        ];

        for expression in cases {
            let eager = Tokenizer::from_parts(
                super::compile_with_plan(super::plan::build_exclusion_compile_plan(
                    std::slice::from_ref(&expression),
                )),
                1,
                None,
            );
            // Runtime retention defaults to multi-worker compilation. Cover
            // both layouts explicitly rather than depending on the caller's
            // global RAYON_NUM_THREADS setting.
            for threads in [1, 2] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .expect("test thread pool");
                let (mut retained, trace) = pool.install(|| {
                    match super::deferred::try_compile_with_plan_deferred_dense_min_pair_cells(
                        super::plan::build_exclusion_compile_plan(std::slice::from_ref(&expression)),
                        0,
                        true,
                    ) {
                        Ok(prepared) => prepared,
                        Err(_) => panic!("binary intersection must admit dense construction"),
                    }
                });
                assert_eq!(trace.is_some(), threads == 1);
                if let Some(trace) = trace {
                    retained.attach_dense_runtime_trace(trace)
                        .expect("single-worker construction must accept its state trace");
                }
                let (dfa, segment) = retained.finish_runtime();
                let runtime = Tokenizer::from_parts_with_compressed_transitions(
                    dfa, 1, None, segment.into_iter().collect(),
                );
                for input in enumerate_inputs(b"\"abx", 8) {
                    assert_eq!(
                        tokenizer_observation(&runtime, &input),
                        tokenizer_observation(&eager, &input),
                        "dense product differed with {threads} workers for expression {expression:?}, input {input:?}",
                    );
                }
            }
        }
    }

    #[test]
    fn adaptive_final_representation_matches_monolithic_for_ignore_and_repeated_terminals() {
        let expressions = vec![
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b" ".to_vec())),
                min: 1,
                max: None,
            },
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                min: 1,
                max: None,
            },
            Expr::U8Seq(b"b".to_vec()),
            Expr::U8Seq(b"c".to_vec()),
        ];
        let monolithic = build_regex_monolithic(&expressions).into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.clone().into_boxed_slice())),
        );
        let adaptive = build_regex_partitioned_with_adaptive(
            &expressions,
            &[0, 1, 2, 3],
            true,
        )
        .into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );

        for input in enumerate_inputs(b" abc", 6) {
            assert_eq!(
                tokenizer_observation(&adaptive, &input),
                tokenizer_observation(&monolithic, &input),
                "adaptive tokenizer differed for input {input:?}",
            );
        }
    }

    #[test]
    fn epsilon_partitioned_tokenizer_round_trips_through_serde() {
        let expressions = vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"ab".to_vec()),
        ];
        let tokenizer = build_regex_partitioned_with_adaptive(
            &expressions,
            &[0, 1],
            false,
        )
        .into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
        let encoded = bincode::serialize(&tokenizer).unwrap();
        let decoded: Tokenizer = bincode::deserialize(&encoded).unwrap();
        assert!(decoded.has_epsilon_transitions());
        assert_eq!(
            tokenizer.execute_from_state(b"a", tokenizer.initial_state()),
            decoded.execute_from_state(b"a", decoded.initial_state()),
        );
    }

    #[test]
    fn fingerprint_product_state_lookup_verifies_canonical_tuple_and_overflow() {
        let mut first = super::product::ProductStateTuple::new();
        first.push((0, 7));
        first.push((3, 11));
        let mut overflow_tuple = super::product::ProductStateTuple::new();
        overflow_tuple.push((1, 5));
        overflow_tuple.push((4, 13));

        let mut state_by_fingerprint = rustc_hash::FxHashMap::default();
        state_by_fingerprint.insert(super::product::product_state_fingerprint(&first), 0);
        let mut overflow = rustc_hash::FxHashMap::default();
        overflow.insert(overflow_tuple.clone(), 1);
        let mut lookup = super::product::ProductStateLookup::Fingerprint {
            state_by_fingerprint,
            canonical_tuples: vec![first.clone(), overflow_tuple.clone()],
            overflow,
        };

        assert_eq!(lookup.get(&first), Some(0));
        assert_eq!(lookup.get(&overflow_tuple), Some(1));

        let mut inserted = super::product::ProductStateTuple::new();
        inserted.push((2, 17));
        lookup.insert(inserted.clone(), 2);
        assert_eq!(lookup.get(&inserted), Some(2));
    }

#[test]
fn factoring_choice_spines_does_not_rewalk_cached_children() {
    fn contains(expr: &Expr, wanted: &Arc<Expr>) -> bool {
        match expr {
            Expr::Shared(inner) => Arc::ptr_eq(inner, wanted) || contains(inner, wanted),
            Expr::Seq(parts) | Expr::Choice(parts) => parts.iter().any(|part| contains(part, wanted)),
            Expr::Repeat { expr, .. } => contains(expr, wanted),
            Expr::Exclude { expr, exclude } => contains(expr, wanted) || contains(exclude, wanted),
            Expr::Intersect { expr, intersect } => contains(expr, wanted) || contains(intersect, wanted),
            _ => false,
        }
    }
    let shared = Arc::new(Expr::Repeat {
        expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"ab"))), min: 1, max: None,
    });
    let cached = Arc::new(factor_regex_expr((*shared).clone()));
    let cache = rustc_hash::FxHashMap::from_iter([(Arc::as_ptr(&shared) as usize, Arc::clone(&cached))]);
    let literal = |s: &str| Expr::U8Seq(s.as_bytes().to_vec());
    for suffix in [false, true] {
        for subset in [false, true] {
            let mut options = ["a", "b"].into_iter().map(|different| {
                if suffix {
                    Expr::Seq(vec![literal(different), Expr::Shared(Arc::clone(&shared)), literal("x")])
                } else {
                    Expr::Seq(vec![literal("x"), Expr::Shared(Arc::clone(&shared)), literal(different)])
                }
            }).collect::<Vec<_>>();
            if subset { options.push(literal("z")); }
            let original = Expr::Choice(options);
            let factored = super::factor_regex_expr_with_shared_cache(original.clone(), &cache);
            assert!(contains(&factored, &cached), "choice rewrite must retain the already factored child");
            for input in ["", "z", "x", "xaa", "xab", "xaba", "xbab", "aax", "abx", "bax", "bbbx", "xc", "cax"] {
                assert_eq!(terminal_matches(original.clone(), input.as_bytes()),
                    terminal_matches(factored.clone(), input.as_bytes()),
                    "suffix={suffix} subset={subset} input={input}");
            }
        }
    }
}
