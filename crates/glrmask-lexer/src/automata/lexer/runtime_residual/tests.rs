use super::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[test]
fn exact_byte_class_expansion_preserves_possible_futures() {
    let mut dfa = DFA::new(3);
    dfa.ensure_group_capacity(1);
    dfa.add_transition(0, 0, 1);
    dfa.add_transition(1, 1, 2);
    let mut finalizers = BitSet::new(1);
    finalizers.set(0);
    dfa.overwrite_state_metadata(2, finalizers, BitSet::new(1));
    dfa.recompute_possible_futures();

    let futures_before = (0..dfa.num_states() as u32)
        .map(|state| dfa.possible_future_group_ids(state).clone())
        .collect::<Vec<_>>();

    expand_exact_byte_classes(&mut dfa, &[vec![b'a', b'b'], vec![b'c', b'd']]);

    for (state, expected) in futures_before.iter().enumerate() {
        assert_eq!(dfa.possible_future_group_ids(state as u32), expected);
    }
    assert_eq!(dfa.step(0, b'a'), Some(1));
    assert_eq!(dfa.step(0, b'b'), Some(1));
    assert_eq!(dfa.step(1, b'c'), Some(2));
    assert_eq!(dfa.step(1, b'd'), Some(2));
}

fn bytes(value: &[u8]) -> Expr {
    Expr::U8Seq(value.to_vec())
}

fn accepts(arena: &mut ResidualArena, mut state: ResidualId, input: &[u8]) -> bool {
    for &byte in input {
        state = arena.step(state, byte).unwrap();
    }
    arena.is_nullable(state)
}

fn materialized_accepts(dfa: &DFA, input: &[u8]) -> bool {
    let mut state = 0;
    for &byte in input {
        let Some(target) = dfa.step(state, byte) else {
            return false;
        };
        state = target;
    }
    dfa.finalizers(state).contains(0)
}

fn random_small_expr(rng: &mut StdRng, depth: usize) -> Expr {
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
            min: rng.gen_range(0..=2),
            max: Some(rng.gen_range(2..=4)),
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

fn all_words(alphabet: &[u8], max_len: usize) -> Vec<Vec<u8>> {
    let mut words = vec![Vec::new()];
    let mut frontier = vec![Vec::new()];
    for _ in 0..max_len {
        let mut next = Vec::new();
        for prefix in frontier {
            for &byte in alphabet {
                let mut word = prefix.clone();
                word.push(byte);
                words.push(word.clone());
                next.push(word);
            }
        }
        frontier = next;
    }
    words
}

#[test]
fn giant_repeat_bound_stays_symbolic() {
    let expr = Expr::Repeat {
        expr: Box::new(bytes(b"ab")),
        min: 3,
        max: Some(1_000_000_000),
    };
    let (mut arena, root) = ResidualArena::from_expr(&expr).unwrap();
    let initial_states = arena.state_count();
    assert!(arena.has_future(root).unwrap());
    assert!(arena.state_count() <= initial_states + 2);

    let mut state = root;
    for _ in 0..3 {
        state = arena.step(state, b'a').unwrap();
        state = arena.step(state, b'b').unwrap();
    }
    assert!(arena.is_nullable(state));
    assert!(arena.has_future(state).unwrap());
    assert!(arena.state_count() < 32);
}

#[test]
fn repeat_suffix_boundary_is_general_derivative_nondeterminism() {
    let expr = Expr::Seq(vec![
        Expr::Repeat {
            expr: Box::new(bytes(b"a")),
            min: 0,
            max: Some(1_000_000_000),
        },
        bytes(b"ab"),
    ]);
    let (mut arena, root) = ResidualArena::from_expr(&expr).unwrap();
    assert!(accepts(&mut arena, root, b"ab"));
    assert!(accepts(&mut arena, root, b"aab"));
    assert!(accepts(&mut arena, root, b"aaaaab"));
    assert!(!accepts(&mut arena, root, b"b"));
    assert!(arena.state_count() < 64);
}

#[test]
fn nullable_repeat_body_does_not_walk_the_bound() {
    let body = Expr::Choice(vec![Expr::Epsilon, bytes(b"a")]);
    let expr = Expr::Repeat {
        expr: Box::new(body),
        min: 500_000_000,
        max: Some(1_000_000_000),
    };
    let (mut arena, root) = ResidualArena::from_expr(&expr).unwrap();
    assert!(arena.is_nullable(root));
    assert!(arena.has_future(root).unwrap());
    assert!(accepts(&mut arena, root, b"aaa"));
    assert!(arena.state_count() < 32);
}

#[test]
fn boolean_residuals_derive_compositionally() {
    let left = Expr::Seq(vec![
        Expr::Repeat {
            expr: Box::new(bytes(b"a")),
            min: 0,
            max: Some(32),
        },
        bytes(b"b"),
    ]);
    let right = Expr::Choice(vec![bytes(b"b"), bytes(b"aab"), bytes(b"c")]);
    let expr = Expr::Intersect {
        expr: Box::new(left),
        intersect: Box::new(right),
    };
    let (mut arena, root) = ResidualArena::from_expr(&expr).unwrap();
    assert!(arena.has_future(root).unwrap());
    assert!(accepts(&mut arena, root, b"b"));
    assert!(accepts(&mut arena, root, b"aab"));
    assert!(!accepts(&mut arena, root, b"ab"));
    assert!(!accepts(&mut arena, root, b"c"));
}

#[test]
fn embedded_dfa_epsilon_closure_is_a_compositional_residual() {
    let mut dfa = DFA::new(4);
    dfa.ensure_group_capacity(1);
    dfa.add_epsilon_transition(0, 1);
    dfa.add_transition(1, b'a', 2);
    dfa.add_epsilon_transition(2, 3);
    let mut accepting = BitSet::new(1);
    accepting.set(0);
    dfa.overwrite_state_metadata(3, accepting, BitSet::new(1));

    let expr = Expr::Dfa(Arc::new(dfa));
    let (mut arena, root) = ResidualArena::from_expr(&expr)
        .expect("epsilon-bearing embedded DFA must stay in the general residual algebra");
    assert!(!arena.is_nullable(root));
    assert!(arena.has_future(root).unwrap());
    assert!(accepts(&mut arena, root, b"a"));
    assert!(!accepts(&mut arena, root, b""));
    assert!(!accepts(&mut arena, root, b"aa"));
}

#[test]
fn boolean_liveness_ceiling_is_error_not_dead() {
    let expr = Expr::Intersect {
        expr: Box::new(Expr::Repeat {
            expr: Box::new(bytes(b"a")),
            min: 100,
            max: Some(100),
        }),
        intersect: Box::new(Expr::Repeat {
            expr: Box::new(bytes(b"aa")),
            min: 50,
            max: Some(50),
        }),
    };
    let (mut arena, root) = ResidualArena::from_expr(&expr).unwrap();
    let error = arena
        .has_future_with_budget(root, 8, 64)
        .expect_err("a deliberately tiny resource ceiling must not become a false dead result");
    assert!(error.contains("budget"), "unexpected liveness error: {error}");
    assert!(arena.has_future_with_budget(root, 256, 512).unwrap());
}

#[test]
fn boolean_liveness_does_not_retain_dense_transition_rows() {
    let expr = Expr::Intersect {
        expr: Box::new(Expr::Repeat {
            expr: Box::new(bytes(b"a")),
            min: 2,
            max: Some(4),
        }),
        intersect: Box::new(bytes(b"aa")),
    };
    let (mut arena, root) = ResidualArena::from_expr(&expr).unwrap();
    assert!(arena.transitions.iter().all(Option::is_none));
    assert!(arena.has_future(root).unwrap());
    assert!(
        arena.transitions.iter().all(Option::is_none),
        "exact liveness must keep derivative caching query-local rather than retaining dense rows",
    );

    // Normal runtime stepping deliberately keeps the dense hot-path cache.
    assert_ne!(arena.step(root, b'a').unwrap(), arena.empty);
    assert!(arena.transitions[root as usize].is_some());
}

#[test]
fn boolean_liveness_budget_charges_recursive_sparse_derivatives() {
    let expr = Expr::Intersect {
        expr: Box::new(Expr::Choice(vec![bytes(b"a"), bytes(b"b"), bytes(b"c")])),
        intersect: Box::new(Expr::Choice(vec![bytes(b"a"), bytes(b"b")])),
    };
    let (mut arena, root) = ResidualArena::from_expr(&expr).unwrap();
    let error = arena
        .has_future_with_budget(root, 16, 1)
        .expect_err("recursive sparse derivative work must consume the transition budget");
    assert!(error.contains("transition budget"), "unexpected error: {error}");
    assert!(arena.has_future_with_budget(root, 16, 32).unwrap());
}

#[test]
fn giant_repeat_liveness_solves_embedded_body_once() {
    let mut body = DFA::new(3);
    body.ensure_group_capacity(1);
    body.add_transition(0, b'a', 1);
    body.add_transition(1, b'b', 2);
    let mut accepting = BitSet::new(1);
    accepting.set(0);
    body.overwrite_state_metadata(2, accepting, BitSet::new(1));
    // Deliberately leave derived future metadata stale: the residual
    // engine must reason from the DFA graph, not from precomputed labels.

    let expr = Expr::Repeat {
        expr: Box::new(Expr::Dfa(Arc::new(body))),
        min: 1_000_000_000,
        max: Some(1_000_000_000),
    };
    let (mut arena, root) = ResidualArena::from_expr(&expr).unwrap();
    assert!(
        arena.has_future_with_budget(root, 8, 16).unwrap(),
        "repeat liveness must solve the body language once rather than walk the billion-copy counter",
    );
}

#[test]
fn sigma_star_identity_eliminates_trivial_boolean_counter_search() {
    let mut body = DFA::new(2);
    body.ensure_group_capacity(1);
    body.add_transition(0, b'a', 1);
    let mut accepting = BitSet::new(1);
    accepting.set(0);
    body.overwrite_state_metadata(1, accepting, BitSet::new(1));

    let counted = Expr::Repeat {
        expr: Box::new(Expr::Dfa(Arc::new(body))),
        min: 1_000_000_000,
        max: Some(1_000_000_000),
    };
    let sigma_star = Expr::Repeat {
        expr: Box::new(Expr::U8Class(U8Set::all())),
        min: 0,
        max: None,
    };
    let expr = Expr::Intersect {
        expr: Box::new(counted),
        intersect: Box::new(sigma_star.clone()),
    };
    let (mut arena, root) = ResidualArena::from_expr(&expr).unwrap();
    assert!(
        arena.has_future_with_budget(root, 4, 4).unwrap(),
        "intersection with sigma-star must simplify before walking the billion-copy counter",
    );

    let choice = Expr::Choice(vec![bytes(b"literal"), sigma_star.clone()]);
    let (mut arena, root) = ResidualArena::from_expr(&choice).unwrap();
    assert_eq!(root, arena.sigma_star);
    assert!(accepts(&mut arena, root, b"anything\0goes"));

    let excluded = Expr::Exclude {
        expr: Box::new(bytes(b"literal")),
        exclude: Box::new(sigma_star),
    };
    let (arena, root) = ResidualArena::from_expr(&excluded).unwrap();
    assert!(arena.is_empty(root));
}

#[test]
fn seeded_residual_algebra_matches_materialized_dfa() {
    let mut rng = StdRng::seed_from_u64(0x5E51_DA1A_2026_0826);
    let words = all_words(b"abc", 4);
    for case in 0..256 {
        let expr = random_small_expr(&mut rng, 3);
        let dfa = super::super::compile::compile_terminal_expr_dfa(&expr);
        let (mut arena, root) = ResidualArena::from_expr(&expr)
            .unwrap_or_else(|| panic!("residual compilation failed for case {case}: {expr:?}"));

        for word in &words {
            assert_eq!(
                accepts(&mut arena, root, word),
                materialized_accepts(&dfa, word),
                "residual/materialized language mismatch in case {case}, expr={expr:?}, word={word:?}",
            );
        }

        assert_eq!(
            arena.has_future(root).unwrap(),
            dfa.possible_future_group_ids(0).contains(0),
            "residual/materialized root liveness mismatch in case {case}, expr={expr:?}",
        );
    }
}

#[test]
fn sequence_liveness_skips_hard_nullable_siblings() {
    let hard_nullable = Expr::Repeat {
        expr: Box::new(Expr::Intersect {
            expr: Box::new(Expr::Repeat {
                expr: Box::new(bytes(b"a")),
                min: 100,
                max: Some(100),
            }),
            intersect: Box::new(Expr::Repeat {
                expr: Box::new(bytes(b"aa")),
                min: 50,
                max: Some(50),
            }),
        }),
        min: 0,
        max: Some(1_000_000_000),
    };
    let expr = Expr::Seq(vec![hard_nullable, bytes(b"z")]);
    let (mut arena, root) = ResidualArena::from_expr(&expr).unwrap();
    assert!(
        arena.has_future_with_budget(root, 0, 0).unwrap(),
        "the nonnullable literal proves a positive sequence word without solving the nullable sibling",
    );
}

#[test]
fn runtime_future_bit_is_conservative_until_exact_boundary_check() {
    // The whole intersection accepts "c", but after consuming 'a' its
    // residual is exactly b intersect c: syntactically nonempty, semantically dead.
    let expr = Expr::Intersect {
        expr: Box::new(Expr::Choice(vec![bytes(b"ab"), bytes(b"c")])),
        intersect: Box::new(Expr::Choice(vec![bytes(b"ac"), bytes(b"c")])),
    };
    let allocator = Arc::new(VirtualStateAllocator::new(2).unwrap());
    let owners = Arc::new(VirtualRuntimeStateOwners::new(2, &[1]).unwrap());
    let runtime =
        VirtualResidualRuntime::new(&expr, 0, 0, 1, 2, 1, allocator, owners).unwrap();
    let dead_prefix = runtime
        .step(1, b'a')
        .expect("syntactic derivative is retained until the exact boundary check");
    assert!(runtime.futures(dead_prefix).unwrap().contains(0));
    assert_eq!(runtime.exact_has_future(dead_prefix).unwrap(), Some(false));

    let accepting = runtime.step(1, b'c').unwrap();
    assert!(runtime.finalizers(accepting).unwrap().contains(0));
    assert_eq!(runtime.exact_has_future(accepting).unwrap(), Some(false));
}

#[test]
fn empty_boolean_root_is_conservative_until_exact_boundary_check() {
    let a_star = || Expr::Repeat {
        expr: Box::new(bytes(b"a")),
        min: 0,
        max: None,
    };
    let expr = Expr::Intersect {
        expr: Box::new(Expr::Seq(vec![a_star(), bytes(b"b")])),
        intersect: Box::new(Expr::Seq(vec![a_star(), bytes(b"c")])),
    };
    let allocator = Arc::new(VirtualStateAllocator::new(2).unwrap());
    let owners = Arc::new(VirtualRuntimeStateOwners::new(2, &[1]).unwrap());
    let runtime =
        VirtualResidualRuntime::new(&expr, 0, 0, 1, 2, 1, allocator, owners).unwrap();
    assert!(runtime.root_has_future());
    assert!(runtime.futures(1).unwrap().contains(0));
    assert_eq!(runtime.exact_has_future(1).unwrap(), Some(false));
    assert_eq!(
        runtime.step(1, b'a'),
        Some(1),
        "a syntactically continuing dead Boolean residual may remain as a conservative proxy until exact boundary pruning",
    );
}

#[test]
fn runtime_construction_does_not_force_boolean_liveness_search() {
    let expr = Expr::Intersect {
        expr: Box::new(Expr::Repeat {
            expr: Box::new(bytes(b"a")),
            min: 100,
            max: Some(100),
        }),
        intersect: Box::new(Expr::Repeat {
            expr: Box::new(bytes(b"aa")),
            min: 50,
            max: Some(50),
        }),
    };
    let allocator = Arc::new(VirtualStateAllocator::new(2).unwrap());
    let owners = Arc::new(VirtualRuntimeStateOwners::new(2, &[1]).unwrap());
    let runtime =
        VirtualResidualRuntime::new(&expr, 0, 0, 1, 2, 1, allocator, owners).unwrap();
    assert!(runtime.root_has_future());

    let store = runtime.store.lock().unwrap();
    assert_eq!(
        store.arena.nonempty_cache[store.root as usize],
        None,
        "constructing the runtime must not eagerly solve a hard Boolean liveness problem",
    );
}

fn bounded_code_body() -> Expr {
    Expr::Choice(vec![bytes(b"a"), bytes(b"bc")])
}

fn bounded_code_envelope_expr(min: usize, max: usize) -> Expr {
    Expr::Seq(vec![
        bytes(b"<"),
        Expr::Repeat {
            expr: Box::new(bounded_code_body()),
            min,
            max: Some(max),
        },
        bytes(b">"),
    ])
}

fn bounded_code_envelope_with_body(body: Expr, min: usize, max: usize) -> Expr {
    Expr::Seq(vec![
        bytes(b"<"),
        Expr::Repeat {
            expr: Box::new(body),
            min,
            max: Some(max),
        },
        bytes(b">"),
    ])
}

fn exact_code_count_pattern(count: usize) -> Expr {
    let mut parts = Vec::with_capacity(count + 2);
    parts.push(bytes(b"<"));
    parts.extend((0..count).map(|_| bounded_code_body()));
    parts.push(bytes(b">"));
    Expr::Seq(parts)
}

#[test]
fn dynamic_direct_coordinate_matches_owned_runtime_state_semantics() {
    let unbounded = Expr::Seq(vec![
        bytes(b"<"),
        Expr::Repeat {
            expr: Box::new(bounded_code_body()),
            min: 0,
            max: None,
        },
        bytes(b">")
    ]);
    let expr = Expr::Intersect {
        expr: Box::new(unbounded),
        intersect: Box::new(bounded_code_envelope_expr(0, 4)),
    };
    let allocator = Arc::new(VirtualStateAllocator::new(2).unwrap());
    let owners = Arc::new(VirtualRuntimeStateOwners::new(2, &[1]).unwrap());
    let runtime = VirtualResidualRuntime::new_dynamic(
        &expr, 0, 0, 1, 2, 1, allocator, owners,
    )
    .expect("bounded-code dynamic runtime should certify a direct coordinate");

    for input in [
        &b"<>"[..],
        &b"<a>"[..],
        &b"<bc>"[..],
        &b"<abc>"[..],
        &b"<aaaa>"[..],
        &b"<aaaaa>"[..],
        &b"<b>"[..],
    ] {
        let mut direct = runtime
            .direct_coordinate_for_state(1)
            .expect("dynamic runtime root should expose a direct coordinate");
        let mut state = Some(1u32);
        for &byte in input {
            let direct_next = runtime.direct_coordinate_step(direct, byte);
            let state_next = state.and_then(|source| runtime.step(source, byte));
            assert_eq!(direct_next.is_some(), state_next.is_some(), "step liveness mismatch for {input:?} at byte {byte:?}");
            let (Some(next_direct), Some(next_state)) = (direct_next, state_next) else {
                break;
            };
            direct = next_direct;
            state = Some(next_state);
            let direct_accepting = runtime
                .direct_coordinate_accepting(direct)
                .expect("coordinate belongs to this runtime");
            assert_eq!(direct_accepting, runtime.accepting_now(next_state).unwrap());
            let direct_future = runtime
                .direct_coordinate_has_future(direct)
                .expect("coordinate belongs to this runtime");
            assert_eq!(direct_future, runtime.exact_has_future(next_state).unwrap().unwrap());
            assert_eq!(runtime.state_for_direct_coordinate(direct), Some(next_state));
        }
    }
}

#[test]
fn coordinate_runtime_retains_bounded_slice_radius_after_prefix() {
    let unbounded = Expr::Seq(vec![
        bytes(b"<"),
        Expr::Repeat {
            expr: Box::new(bounded_code_body()),
            min: 0,
            max: None,
        },
        bytes(b">")
    ]);
    let expr = Expr::Intersect {
        expr: Box::new(unbounded),
        intersect: Box::new(bounded_code_envelope_expr(0, 4)),
    };
    let allocator = Arc::new(VirtualStateAllocator::new(2).unwrap());
    let owners = Arc::new(VirtualRuntimeStateOwners::new(2, &[1]).unwrap());
    let runtime = VirtualResidualRuntime::new_dynamic(
        &expr, 0, 0, 1, 2, 1, allocator, owners,
    )
    .expect("bounded-code dynamic runtime should certify a direct coordinate");

    let mut byte_to_class = [1u8; 256];
    byte_to_class[b'a' as usize] = 0;
    let class_count = 2usize;
    let transitions = [1u32, 2, 1, 2, 2, 2];
    let accepting = [false, true, false];
    let productive = [true, true, false];

    let after_open = runtime.step(1, b'<').expect("opening delimiter should be live");
    assert_eq!(
        runtime.parser_transparent_byte_dfa_repeat_radius(
            after_open, 0, class_count, &byte_to_class, &transitions,
            &accepting, &productive, 8, 16 * 1024,
        ),
        Some(4),
    );
    let after_one = runtime.step(after_open, b'a').expect("first body atom should be live");
    assert_eq!(
        runtime.parser_transparent_byte_dfa_repeat_radius(
            after_one, 0, class_count, &byte_to_class, &transitions,
            &accepting, &productive, 8, 16 * 1024,
        ),
        Some(3),
    );
}

#[test]
fn slice_radius_totality_rejects_missing_multibyte_atom() {
    // The existential completion relation contains only the successful
    // a-loop; it omits the body word bc, which the pattern cannot read.
    let pattern = Expr::Seq(vec![
        bytes(b"<"),
        Expr::Repeat { expr: Box::new(bytes(b"a")), min: 0, max: None },
        bytes(b">"),
    ]);
    let expr = Expr::Intersect {
        expr: Box::new(pattern),
        intersect: Box::new(bounded_code_envelope_expr(0, 4)),
    };
    let runtime = VirtualResidualRuntime::new_dynamic(
        &expr, 0, 0, 1, 2, 1,
        Arc::new(VirtualStateAllocator::new(2).unwrap()),
        Arc::new(VirtualRuntimeStateOwners::new(2, &[1]).unwrap()),
    ).expect("bounded-code test runtime");
    let source = runtime.step(1, b'<').expect("live prefix");
    assert!(runtime.step(source, b'b').is_none());

    // Exact atom language a|bc, repeated by the slice DFA. First accepting
    // boundaries are complete body code words, including the missing bc.
    let mut classes = [3u8; 256];
    classes[b'a' as usize] = 0;
    classes[b'b' as usize] = 1;
    classes[b'c' as usize] = 2;
    let transitions = [1, 2, 3, 3, 1, 2, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3];
    let accepting = [false, true, false, false];
    let productive = [true, true, true, false];
    assert_eq!(
        runtime.parser_transparent_byte_dfa_repeat_radius(
            source, 0, 4, &classes, &transitions, &accepting, &productive, 8, 16 * 1024,
        ),
        Some(0),
        "an existing self-loop cannot certify an absent body-code transition",
    );
}

#[test]
fn slice_radius_totality_rejects_space_after_last_allowed_word() {
    for max_words in 1usize..=4 {
        let word = || Expr::Repeat {
            expr: Box::new(bytes(b"a")), min: 1, max: None,
        };
        let pattern = Expr::Seq(vec![
            bytes(b"<"), word(),
            Expr::Repeat {
                expr: Box::new(Expr::Seq(vec![bytes(b" "), word()])),
                min: 0, max: Some(max_words - 1),
            },
            bytes(b">"),
        ]);
        let expr = Expr::Intersect {
            expr: Box::new(pattern),
            intersect: Box::new(bounded_code_envelope_with_body(
                Expr::Choice(vec![bytes(b"a"), bytes(b" ")]), 0, 32,
            )),
        };
        let runtime = VirtualResidualRuntime::new_dynamic(
            &expr, 0, 0, 1, 2, 1,
            Arc::new(VirtualStateAllocator::new(2).unwrap()),
            Arc::new(VirtualRuntimeStateOwners::new(2, &[1]).unwrap()),
        ).expect("word-count bounded-code runtime");
        let prefix = format!("<{}", vec!["a"; max_words].join(" "));
        let mut source = 1u32;
        for byte in prefix.bytes() {
            source = runtime.step(source, byte).expect("valid last-word prefix");
        }
        assert!(runtime.step(source, b' ').is_none(), "word bound must reject another separator");
        let mut broad_classes = [1u8; 256];
        broad_classes[b'a' as usize] = 0;
        broad_classes[b' ' as usize] = 0;
        let transitions = [1, 2, 1, 2, 2, 2];
        let accepting = [false, true, false];
        let productive = [true, true, false];
        assert_eq!(
            runtime.parser_transparent_byte_dfa_repeat_radius(
                source, 0, 2, &broad_classes, &transitions,
                &accepting, &productive, 32, 16 * 1024,
            ),
            Some(0), "broad atom set includes the forbidden separator; max_words={max_words}",
        );
        // A narrower alphabet really is total and invariant, and should
        // retain the fast positive radius up to the character envelope.
        let mut narrow_classes = [1u8; 256];
        narrow_classes[b'a' as usize] = 0;
        assert_eq!(
            runtime.parser_transparent_byte_dfa_repeat_radius(
                source, 0, 2, &narrow_classes, &transitions,
                &accepting, &productive, 32, 16 * 1024,
            ),
            Some((32 - (2 * max_words - 1)) as u32),
        );
    }
}

#[test]
fn direct_coordinate_dense_mask_key_refines_full_finite_projection() {
    let unbounded = Expr::Seq(vec![
        bytes(b"<"),
        Expr::Repeat {
            expr: Box::new(bounded_code_body()),
            min: 0,
            max: None,
        },
        bytes(b">")
    ]);
    let expr = Expr::Intersect {
        expr: Box::new(unbounded),
        intersect: Box::new(bounded_code_envelope_expr(0, 80)),
    };
    let allocator = Arc::new(VirtualStateAllocator::new(2).unwrap());
    let owners = Arc::new(VirtualRuntimeStateOwners::new(2, &[1]).unwrap());
    let runtime = Arc::new(
        VirtualResidualRuntime::new_dynamic(
            &expr, 0, 0, 1, 2, 1, allocator, owners,
        )
        .expect("bounded-code dynamic runtime should certify a direct coordinate"),
    );
    let max_token_len = 8usize;
    let (_, _, _, projection) = Arc::clone(&runtime)
        .build_finite_mask_projection(max_token_len, 10_000)
        .expect("fixture should admit a finite mask projection");

    let mut coordinate = runtime
        .direct_coordinate_for_state(1)
        .expect("dynamic runtime root should expose a direct coordinate");
    let mut by_dense = FxHashMap::<(u32, u32), u32>::default();
    let mut exact_states = 0usize;
    let mut alias_count = 0usize;
    for &byte in std::iter::once(&b'<').chain(std::iter::repeat_n(&b'a', 70)) {
        coordinate = runtime
            .direct_coordinate_step(coordinate, byte)
            .expect("long valid body prefix should remain live");
        let state = runtime
            .state_for_direct_coordinate(coordinate)
            .expect("reachable direct coordinate should materialize");
        let Some(key) = runtime
            .direct_coordinate_finite_mask_dense_key(coordinate, max_token_len)
        else {
            continue;
        };
        exact_states += 1;
        let projected = projection
            .project(state)
            .expect("reachable exact state must project");
        if let Some(previous) = by_dense.insert(key, projected) {
            alias_count += 1;
            assert_eq!(
                previous, projected,
                "equal dense one-token coordinates must map to one minimized projection state",
            );
        }
    }
    assert!(exact_states > 40, "fixture should traverse the deep exact count interval");
    assert!(alias_count > 20, "dense key should collapse many deep-interior exact counts");
}

#[test]
fn bounded_code_oracle_drops_redundant_unbounded_envelope_pattern() {
    let unbounded = Expr::Seq(vec![
        bytes(b"<"),
        Expr::Repeat {
            expr: Box::new(bounded_code_body()),
            min: 0,
            max: None,
        },
        bytes(b">"),
    ]);
    let expr = Expr::Intersect {
        expr: Box::new(unbounded),
        intersect: Box::new(bounded_code_envelope_expr(2, 4)),
    };
    let mut oracle = BoundedCodeIntersectionOracle::from_expr(&expr)
        .expect("redundant unbounded envelope pattern should certify");
    assert_eq!(oracle.pattern.num_states(), 1);
    assert!(oracle.has_future(oracle.root_coordinate()));

    let materialized = compile_terminal_expr_dfa(&expr);
    assert_eq!(
        oracle.has_future(oracle.root_coordinate()),
        materialized.possible_future_group_ids(0).contains(0),
    );
}

#[test]
fn standalone_bounded_code_mask_component_retains_byte_transitions() {
    let unbounded = Expr::Seq(vec![
        bytes(b"<"),
        Expr::Repeat {
            expr: Box::new(bounded_code_body()),
            min: 0,
            max: None,
        },
        bytes(b">"),
    ]);
    let expr = Expr::Intersect {
        expr: Box::new(unbounded),
        intersect: Box::new(bounded_code_envelope_expr(0, 100)),
    };

    let (dfa, root) = prepare_bounded_code_mask_component(&expr)
        .expect("bounded-code expression should prepare")
        .finish_for_vocab_conservative(8)
        .expect("bounded-code mask component should compile");

    assert!(
        dfa.transition_count() > 0,
        "standalone mask component must materialize its compressed transition sidecar"
    );
    assert!(dfa.possible_future_group_ids(root).contains(0));
    let after_prefix = dfa
        .step(root, b'<')
        .expect("standalone mask component must execute its prefix byte");
    assert!(dfa.possible_future_group_ids(after_prefix).contains(0));
}

#[test]
fn standalone_bounded_code_mask_component_accepts_moderate_large_minimum() {
    let unbounded = Expr::Seq(vec![
        bytes(b"<"),
        Expr::Repeat {
            expr: Box::new(bounded_code_body()),
            min: 0,
            max: None,
        },
        bytes(b">"),
    ]);
    let expr = Expr::Intersect {
        expr: Box::new(unbounded),
        intersect: Box::new(bounded_code_envelope_expr(500, 10_000)),
    };

    let (dfa, root) = prepare_bounded_code_mask_component(&expr)
        .expect("bounded-code expression should prepare")
        .finish_for_vocab_conservative(128)
        .expect("moderate lower bound should stay on the finite mask path");

    assert!(dfa.possible_future_group_ids(root).contains(0));
    assert!(
        dfa.num_states() < 20_000,
        "finite mask coordinate should scale with the lower bound plus one-token stencil, not maxLength: {} states",
        dfa.num_states(),
    );
}

#[test]
fn bounded_code_intersection_oracle_respects_gapped_copy_counts() {
    // The pattern admits exactly two or four code words.  An envelope of
    // exactly three words is therefore dead even though each operand is
    // individually live.  This is the counterexample that rules out a
    // simple min/max-distance liveness approximation.
    let pattern = Expr::Choice(vec![exact_code_count_pattern(2), exact_code_count_pattern(4)]);
    let dead = Expr::Intersect {
        expr: Box::new(pattern.clone()),
        intersect: Box::new(bounded_code_envelope_expr(3, 3)),
    };
    let mut dead_oracle = BoundedCodeIntersectionOracle::from_expr(&dead)
        .expect("prefix-code bounded intersection should certify");
    assert!(!dead_oracle.has_future(dead_oracle.root_coordinate()));

    let live = Expr::Intersect {
        expr: Box::new(pattern),
        intersect: Box::new(bounded_code_envelope_expr(3, 4)),
    };
    let mut live_oracle = BoundedCodeIntersectionOracle::from_expr(&live)
        .expect("prefix-code bounded intersection should certify");
    assert!(live_oracle.has_future(live_oracle.root_coordinate()));
}

#[test]
fn bounded_code_oracle_coalesces_identical_envelope_intervals_exactly() {
    let pattern = exact_code_count_pattern(3);
    let expr = Expr::Intersect {
        expr: Box::new(Expr::Intersect {
            expr: Box::new(pattern),
            intersect: Box::new(bounded_code_envelope_expr(1, 4)),
        }),
        intersect: Box::new(bounded_code_envelope_expr(3, 6)),
    };
    let mut oracle = BoundedCodeIntersectionOracle::from_expr(&expr)
        .expect("identical bounded-code envelopes should coalesce");
    assert_eq!((oracle.min, oracle.max), (3, 4));
    assert!(oracle.has_future(oracle.root_coordinate()));

    let materialized = compile_terminal_expr_dfa(&expr);
    assert!(
        materialized
            .possible_future_group_ids(0)
            .contains(0),
        "materialized intersection must agree that the root has a future",
    );
}

#[test]
fn bounded_code_oracle_coalesces_disjoint_identical_envelopes_to_dead() {
    let pattern = Expr::Choice(vec![exact_code_count_pattern(2), exact_code_count_pattern(4)]);
    let expr = Expr::Intersect {
        expr: Box::new(Expr::Intersect {
            expr: Box::new(pattern),
            intersect: Box::new(bounded_code_envelope_expr(1, 2)),
        }),
        intersect: Box::new(bounded_code_envelope_expr(4, 5)),
    };
    let mut oracle = BoundedCodeIntersectionOracle::from_expr(&expr)
        .expect("disjoint identical envelopes should still admit an exact dead certificate");
    assert_eq!((oracle.min, oracle.max), (4, 2));
    assert!(!oracle.has_future(oracle.root_coordinate()));

    let materialized = compile_terminal_expr_dfa(&expr);
    assert!(
        !materialized
            .possible_future_group_ids(0)
            .contains(0),
        "materialized disjoint intersection must also be dead",
    );
}

#[test]
fn bounded_code_oracle_rejects_ambiguous_code_boundaries() {
    let pattern = Expr::Seq(vec![
        bytes(b"<"),
        Expr::Repeat {
            expr: Box::new(Expr::U8Class(U8Set::all())),
            min: 0,
            max: None,
        },
        bytes(b">"),
    ]);

    // "a" is a prefix of "ab", so greedily treating the first accepting
    // body state as one completed code word would not be exact.
    let non_prefix_free = Expr::Intersect {
        expr: Box::new(pattern.clone()),
        intersect: Box::new(bounded_code_envelope_with_body(
            Expr::Choice(vec![bytes(b"a"), bytes(b"ab")]),
            0,
            4,
        )),
    };
    assert!(BoundedCodeIntersectionOracle::from_expr(&non_prefix_free).is_none());

    // At a code boundary, '>' could either begin another productive body
    // word or begin the fixed suffix. The sidecar deliberately refuses
    // such an envelope rather than choosing one interpretation.
    let suffix_ambiguous = Expr::Intersect {
        expr: Box::new(pattern),
        intersect: Box::new(bounded_code_envelope_with_body(
            Expr::Choice(vec![bytes(b"a"), bytes(b">x")]),
            0,
            4,
        )),
    };
    assert!(BoundedCodeIntersectionOracle::from_expr(&suffix_ambiguous).is_none());
}

#[test]
fn bounded_code_oracle_does_not_materialize_nested_giant_repeats() {
    let pattern = Expr::Seq(vec![
        bytes(b"<"),
        Expr::Repeat {
            expr: Box::new(Expr::U8Class(U8Set::all())),
            min: 0,
            max: None,
        },
        bytes(b">"),
    ]);
    let giant_body = Expr::Repeat {
        expr: Box::new(bytes(b"a")),
        min: 4_096,
        max: Some(4_096),
    };
    let body_giant = Expr::Intersect {
        expr: Box::new(pattern),
        intersect: Box::new(bounded_code_envelope_with_body(giant_body, 0, 5_000)),
    };
    assert!(BoundedCodeIntersectionOracle::from_expr(&body_giant).is_none());

    // A giant bounded repeat can also hide inside the independently
    // compiled pattern operand. Even when simplification could make a
    // particular example cheap (epsilon repeated many times), the oracle
    // must not rely on eagerly discovering that after materialization.
    let giant_pattern = Expr::Choice(vec![
        Expr::Seq(vec![
            bytes(b"<"),
            Expr::Repeat {
                expr: Box::new(Expr::Epsilon),
                min: 0,
                max: Some(4_096),
            },
            bytes(b">"),
        ]),
        bytes(b"x"),
    ]);
    let pattern_giant = Expr::Intersect {
        expr: Box::new(giant_pattern),
        intersect: Box::new(bounded_code_envelope_expr(0, 5_000)),
    };
    assert!(BoundedCodeIntersectionOracle::from_expr(&pattern_giant).is_none());
}

#[test]
fn bounded_code_oracle_matches_materialized_future_at_every_small_prefix() {
    let pattern = Expr::Choice(vec![
        exact_code_count_pattern(1),
        exact_code_count_pattern(3),
        exact_code_count_pattern(4),
    ]);
    let expr = Expr::Intersect {
        expr: Box::new(pattern),
        intersect: Box::new(bounded_code_envelope_expr(1, 4)),
    };
    let materialized = compile_terminal_expr_dfa(&expr);
    let allocator = Arc::new(VirtualStateAllocator::new(2).unwrap());
    let owners = Arc::new(VirtualRuntimeStateOwners::new(2, &[1]).unwrap());
    let runtime =
        VirtualResidualRuntime::new(&expr, 0, 0, 1, 2, 1, allocator, owners).unwrap();
    assert!(runtime.has_bounded_code_liveness_oracle());

    let alphabet = [b'<', b'>', b'a', b'b', b'c'];
    let mut queue = VecDeque::from([(Vec::<u8>::new(), 1u32, 0u32)]);
    let mut seen = FxHashSet::<(u32, u32)>::default();
    seen.insert((1, 0));
    while let Some((prefix, residual_state, materialized_state)) = queue.pop_front() {
        assert_eq!(
            runtime.exact_has_future(residual_state).unwrap(),
            Some(
                materialized
                    .possible_future_group_ids(materialized_state)
                    .contains(0)
            ),
            "future mismatch after prefix {:?}",
            String::from_utf8_lossy(&prefix),
        );
        assert_eq!(
            runtime
                .futures(residual_state)
                .expect("reached residual state must have metadata")
                .contains(0),
            materialized
                .possible_future_group_ids(materialized_state)
                .contains(0),
            "ordinary future metadata mismatch after prefix {:?}",
            String::from_utf8_lossy(&prefix),
        );
        if prefix.len() >= 12 {
            continue;
        }
        for &byte in &alphabet {
            let Some(materialized_target) = materialized.step(materialized_state, byte) else {
                continue;
            };
            let Some(residual_target) = runtime.step(residual_state, byte) else {
                continue;
            };
            if seen.insert((residual_target, materialized_target)) {
                let mut next_prefix = prefix.clone();
                next_prefix.push(byte);
                queue.push_back((next_prefix, residual_target, materialized_target));
            }
        }
    }
}

#[test]
fn bounded_code_sparse_oracle_wire_preserves_runtime_liveness() {
    let expr = Expr::Intersect {
        expr: Box::new(Expr::Choice(vec![
            exact_code_count_pattern(1),
            exact_code_count_pattern(3),
            exact_code_count_pattern(4),
        ])),
        intersect: Box::new(bounded_code_envelope_expr(1, 4)),
    };
    let allocator = Arc::new(VirtualStateAllocator::new(2).unwrap());
    let owners = Arc::new(VirtualRuntimeStateOwners::new(2, &[1]).unwrap());
    let original = VirtualResidualRuntime::new(
        &expr,
        0,
        0,
        1,
        2,
        1,
        Arc::clone(&allocator),
        Arc::clone(&owners),
    )
    .unwrap();
    assert!(original.has_bounded_code_liveness_oracle());
    let wire = original.serialized_bounded_code_oracle();
    assert!(wire.starts_with(&SPARSE_BOUNDED_CODE_ORACLE_MAGIC));

    let loaded = VirtualResidualRuntime::new_preserving_oracle_coordinate_from_oracle_bytes(
        &expr,
        &wire,
        0,
        0,
        1,
        2,
        1,
        allocator,
        owners,
    )
    .expect("BCO2 oracle should restore without rebuilding dense relations");
    assert!(loaded.has_bounded_code_liveness_oracle());

    let alphabet = [b'<', b'>', b'a', b'b', b'c'];
    let mut queue = VecDeque::from([(Vec::<u8>::new(), 1u32, 1u32)]);
    let mut seen = FxHashSet::<(u32, u32)>::default();
    seen.insert((1, 1));
    while let Some((prefix, original_state, loaded_state)) = queue.pop_front() {
        assert_eq!(
            loaded.exact_has_future(loaded_state).unwrap(),
            original.exact_has_future(original_state).unwrap(),
            "BCO2 future mismatch after {:?}",
            String::from_utf8_lossy(&prefix),
        );
        if prefix.len() >= 10 {
            continue;
        }
        for &byte in &alphabet {
            let original_next = original.step(original_state, byte);
            let loaded_next = loaded.step(loaded_state, byte);
            assert_eq!(
                loaded_next.is_some(),
                original_next.is_some(),
                "BCO2 transition mismatch after {:?} + {:?}",
                String::from_utf8_lossy(&prefix),
                byte as char,
            );
            let (Some(original_next), Some(loaded_next)) = (original_next, loaded_next) else {
                continue;
            };
            if seen.insert((original_next, loaded_next)) {
                let mut next_prefix = prefix.clone();
                next_prefix.push(byte);
                queue.push_back((next_prefix, original_next, loaded_next));
            }
        }
    }
}

#[test]
fn master_slice_artifact_round_trips_prepared_runtime_and_rejects_bad_shape() {
    let expr = Expr::Intersect {
        expr: Box::new(Expr::Seq(vec![
            bytes(b"<"),
            Expr::Repeat {
                expr: Box::new(bounded_code_body()),
                min: 0,
                max: None,
            },
            bytes(b">"),
        ])),
        intersect: Box::new(bounded_code_envelope_expr(0, 8)),
    };
    let allocator = Arc::new(VirtualStateAllocator::new(2).unwrap());
    let owners = Arc::new(VirtualRuntimeStateOwners::new(2, &[1]).unwrap());
    let original = VirtualResidualRuntime::new(
        &expr,
        0,
        0,
        1,
        2,
        1,
        Arc::clone(&allocator),
        Arc::clone(&owners),
    )
    .unwrap();

    // A compact DFA for (a|bc)+. State 1 marks an atom boundary and also
    // accepts the next atom, matching the master-slice contract.
    let mut byte_to_class = [0u8; 256];
    for (byte, class) in byte_to_class.iter_mut().enumerate() {
        *class = byte as u8;
    }
    let state_count = 3usize;
    let class_count = 256usize;
    let dead = state_count as u32;
    let mut transitions = vec![dead; state_count * class_count];
    transitions[b'a' as usize] = 1;
    transitions[b'b' as usize] = 2;
    transitions[class_count + b'a' as usize] = 1;
    transitions[class_count + b'b' as usize] = 2;
    transitions[2 * class_count + b'c' as usize] = 1;
    let accepting = [false, true, false];
    let productive = [true, true, true];
    original.prepare_master_slice_artifacts(
        0,
        class_count,
        &byte_to_class,
        &transitions,
        &accepting,
        &productive,
        8,
    );
    let artifact = original
        .master_slice_artifact()
        .expect("master-slice preparation should produce a transfer artifact");
    assert!(!artifact.words.is_empty());
    assert!(!artifact.body_exact.is_empty());
    assert!(!artifact.pattern_targets.is_empty());

    let oracle_bytes = original.serialized_bounded_code_oracle();
    let loaded_allocator = Arc::new(VirtualStateAllocator::new(2).unwrap());
    let loaded_owners = Arc::new(VirtualRuntimeStateOwners::new(2, &[1]).unwrap());
    let loaded = VirtualResidualRuntime::new_preserving_oracle_coordinate_from_oracle_bytes(
        &expr,
        &oracle_bytes,
        0,
        0,
        1,
        2,
        1,
        loaded_allocator,
        loaded_owners,
    )
    .unwrap();
    loaded.restore_master_slice_artifact(artifact.clone()).unwrap();

    let original_store = original.store.lock().unwrap();
    let loaded_store = loaded.store.lock().unwrap();
    assert_eq!(
        loaded_store.body_boundary_future_by_completed,
        original_store.body_boundary_future_by_completed,
    );
    assert_eq!(
        loaded_store.slice_atom_body_exact_cache,
        original_store.slice_atom_body_exact_cache,
    );
    assert_eq!(
        loaded_store.slice_atom_pattern_targets_cache,
        original_store.slice_atom_pattern_targets_cache,
    );
    drop(loaded_store);
    drop(original_store);

    // The compact load path deliberately omits relation powers. Its
    // backwards-DP artifact must also serve the generic (non-body-equal)
    // slice proof and the repeat-radius BFS, not only ordinary liveness.
    let mut compact = VirtualResidualRuntime::compact_liveness_oracle_from_master_slice_artifact(&artifact)
        .expect("valid transferred oracle");
    assert!(compact.exact_powers.is_empty() && compact.prefix_sums.is_empty());
    let coordinate = compact.root_coordinate();
    assert_eq!(compact.has_future_for_slice_proof(coordinate, None), None,
        "missing proof data must decline, not certify a dead language");
    loaded.store.lock().unwrap().liveness_oracle = Some(compact);

    for prefix in [b"<".as_slice(), b"<a", b"<bc", b"<b"] {
        let mut old_state = 1;
        let mut loaded_state = 1;
        for &byte in prefix {
            old_state = original.step(old_state, byte).expect("valid original prefix");
            loaded_state = loaded.step(loaded_state, byte).expect("valid loaded prefix");
        }
        // This alphabet is intentionally not the oracle body (a|bc), so
        // the specialized exact-body proof must fall through to the BFS.
        let first = if prefix == b"<b" { b'c' } else { b'a' };
        let mut repeating = vec![2_u32; 2 * class_count];
        repeating[first as usize] = 1;
        repeating[class_count + b'a' as usize] = 1;
        let repeating_accepting = [false, true];
        let repeating_productive = [true, true];
        let expected_radius = original.parser_transparent_byte_dfa_repeat_radius(
            old_state, 0, class_count, &byte_to_class, &repeating,
            &repeating_accepting, &repeating_productive, 8, 10_000,
        );
        let loaded_radius = loaded.parser_transparent_byte_dfa_repeat_radius(
            loaded_state, 0, class_count, &byte_to_class, &repeating,
            &repeating_accepting, &repeating_productive, 8, 10_000,
        );
        assert!(expected_radius.is_some());
        assert_eq!(loaded_radius, expected_radius, "prefix={prefix:?}");

        let finite_states = 5usize;
        let mut finite = vec![finite_states as u32; finite_states * class_count];
        finite[first as usize] = 1;
        for state in 1..finite_states - 1 {
            finite[state * class_count + b'a' as usize] = (state + 1) as u32;
        }
        let finite_productive = vec![true; finite_states];
        let expected = original.parser_transparent_byte_dfa(
            old_state, 0, class_count, &byte_to_class, &finite,
            &finite_productive, true, 10_000,
        );
        let actual = loaded.parser_transparent_byte_dfa(
            loaded_state, 0, class_count, &byte_to_class, &finite,
            &finite_productive, true, 10_000,
        );
        assert_eq!(actual, expected, "finite slice prefix={prefix:?}");
        assert_eq!(actual, Some(true));

        // Dynamic joint-root and uniform-byte fast paths use different
        // proof entry points. Exercise them against the genuinely compact
        // oracle too, rather than accidentally retaining relation powers.
        let old_coordinate = {
            let store = original.store.lock().unwrap();
            original.oracle_coordinate_for_state_locked(&store, old_state).unwrap()
        };
        let loaded_coordinate = {
            let store = loaded.store.lock().unwrap();
            loaded.oracle_coordinate_for_state_locked(&store, loaded_state).unwrap()
        };
        let direct_expected = original.direct_coordinate_parser_transparent_byte_dfa(
            VirtualResidualDirectCoordinate { runtime_index: original.runtime_index, coordinate: old_coordinate },
            0, class_count, &byte_to_class, &finite, &finite_productive, true, 10_000,
        );
        let direct_actual = loaded.direct_coordinate_parser_transparent_byte_dfa(
            VirtualResidualDirectCoordinate { runtime_index: loaded.runtime_index, coordinate: loaded_coordinate },
            0, class_count, &byte_to_class, &finite, &finite_productive, true, 10_000,
        );
        assert_eq!(direct_actual, direct_expected, "direct finite slice prefix={prefix:?}");
        assert_eq!(direct_actual, Some(true));
        for byte in [b'a', b'c'] {
            let family = U8Set::single(byte);
            for horizon in [1, 3] {
                let expected = original.parser_transparent_byte_family(old_state, family, horizon);
                let actual = loaded.parser_transparent_byte_family(loaded_state, family, horizon);
                assert!(expected.is_some());
                assert_eq!(actual, expected, "byte family prefix={prefix:?} byte={byte} horizon={horizon}");
            }
        }
    }

    let mut malformed = artifact;
    malformed.pattern_states += 1;
    assert!(loaded.restore_master_slice_artifact(malformed).is_err());
}

#[test]
fn bounded_code_oracle_keeps_billion_bound_logarithmic() {
    let expr = Expr::Intersect {
        expr: Box::new(exact_code_count_pattern(2)),
        intersect: Box::new(bounded_code_envelope_expr(0, 1_000_000_000)),
    };
    let mut oracle = BoundedCodeIntersectionOracle::from_expr(&expr)
        .expect("billion-copy prefix-code envelope should certify");
    assert!(oracle.has_future(oracle.root_coordinate()));
    assert!(
        oracle.exact_powers.len() <= 31,
        "doubling table must scale with log2(max), got {} layers",
        oracle.exact_powers.len(),
    );
}

#[test]
fn bounded_code_oracle_sizes_doubling_for_inclusive_power_of_two_ranges() {
    for max in [0usize, 1, 3, 7, 15] {
        let expr = Expr::Intersect {
            expr: Box::new(exact_code_count_pattern(max)),
            intersect: Box::new(bounded_code_envelope_expr(0, max)),
        };
        let mut oracle = BoundedCodeIntersectionOracle::from_expr(&expr)
            .unwrap_or_else(|| panic!("bounded-code oracle should certify max={max}"));
        assert!(
            oracle.has_future(oracle.root_coordinate()),
            "exactly {max} code words must be reachable inside 0..={max}",
        );
    }
}

#[test]
fn bounded_code_oracle_ambiguity_propagates_to_existing_successors() {
    let expr = Expr::Intersect {
        expr: Box::new(exact_code_count_pattern(1)),
        intersect: Box::new(bounded_code_envelope_expr(0, 4)),
    };
    let allocator = Arc::new(VirtualStateAllocator::new(2).unwrap());
    let owners = Arc::new(VirtualRuntimeStateOwners::new(2, &[1]).unwrap());
    let runtime =
        VirtualResidualRuntime::new(&expr, 0, 0, 1, 2, 1, allocator, owners).unwrap();

    let body_boundary = runtime.step(1, b'<').unwrap();
    let after_one = runtime.step(body_boundary, b'a').unwrap();
    assert!(runtime.futures(after_one).unwrap().contains(0));

    {
        let mut store = runtime.store.lock().unwrap();
        let source = VirtualResidualRuntime::residual_for_state(
            &store,
            runtime.root_state,
            body_boundary,
        )
        .unwrap() as usize;
        let target = VirtualResidualRuntime::residual_for_state(
            &store,
            runtime.root_state,
            after_one,
        )
        .unwrap() as usize;
        assert!(matches!(
            store.oracle_coordinates[target],
            BoundedCodeOracleSlot::Exact(_)
        ));
        assert_eq!(store.oracle_futures[target], Some(true));
        store.oracle_coordinates[source] = BoundedCodeOracleSlot::Ambiguous;
    }

    assert_eq!(runtime.step(body_boundary, b'a'), Some(after_one));
    let store = runtime.store.lock().unwrap();
    let target = VirtualResidualRuntime::residual_for_state(
        &store,
        runtime.root_state,
        after_one,
    )
    .unwrap() as usize;
    assert_eq!(
        store.oracle_coordinates[target],
        BoundedCodeOracleSlot::Ambiguous
    );
    assert_eq!(store.oracle_futures[target], None);
}

#[test]
fn bounded_code_runtime_liveness_does_not_fall_back_to_boolean_search() {
    let expr = Expr::Intersect {
        expr: Box::new(Expr::Choice(vec![
            exact_code_count_pattern(2),
            exact_code_count_pattern(4),
        ])),
        intersect: Box::new(bounded_code_envelope_expr(3, 3)),
    };
    let allocator = Arc::new(VirtualStateAllocator::new(2).unwrap());
    let owners = Arc::new(VirtualRuntimeStateOwners::new(2, &[1]).unwrap());
    let runtime =
        VirtualResidualRuntime::new(&expr, 0, 0, 1, 2, 1, allocator, owners).unwrap();
    assert!(runtime.has_bounded_code_liveness_oracle());
    assert_eq!(runtime.exact_has_future(1).unwrap(), Some(false));
    assert!(
        runtime.futures(1).unwrap().is_empty(),
        "certified exact liveness must be visible through ordinary tokenizer future metadata",
    );
    let store = runtime.store.lock().unwrap();
    assert_eq!(
        store.arena.nonempty_cache[store.root as usize],
        None,
        "certified bounded-code liveness must not invoke generic Boolean reachability",
    );
}
