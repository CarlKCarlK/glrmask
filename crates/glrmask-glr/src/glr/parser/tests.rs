use super::{
    DisjointComponentActionProvider, ParserComponentTableSource, ParserGSS,
    ScopedParserSymbol, ScopedSubgrammarLink, close_provider_control_stacks,
    advance_provider_control_closed_stacks,
    materialize_control_eliminated_scoped_provider_table,
    ParserActionProvider,
    advance_concrete_stacks_reference,
    advance_stacks_with_provider,
    advance_stacks_disjoint_top_terminals_bounded,
    normalized_concrete_stacks,
    advance_stacks,
    apply_guarded_stack_shifts,
    apply_guarded_stack_shifts_as_predecessor_remap,
    apply_guarded_stack_shifts_fast,
    apply_guarded_stack_shifts_to_vstack,
    stack_admissible_terminals,
    stack_may_advance_on,
    stack_may_advance_on_any,
    stacks_finished,
    stacks_finished_with_provider,
    GLRTableActionProvider,
    stack_may_advance_disjoint_top_terminals_bounded,
    try_advance_bounded_concrete_paths,
    try_advance_mixed_top_replace_wave,
    try_advance_uniform_deterministic_frontier,
    try_advance_pop1_reduce_guarded_stackshift_wave,
    try_advance_pop1_reduce_plus_stackshift_wave,
    try_advance_pop1_stackshift_shift_wave,
    try_advance_single_active_pop1_stackshift_wave,
    GssSemanticKeyInterner,
    ProvidedAction,
    ProvidedActionRef,
    merge_into,
    reduce_sources_from_isolated,
};
use crate::compiler::glr::accumulator::TerminalsDisallowed;
use crate::compiler::glr::analysis::EOF;
use crate::compiler::glr::table::testing::build_test_table;
use crate::compiler::glr::table::{
    Action, AdmissionPolicy, GLRTable, GuardedStackShift, StackShift, StackShiftGuard,
};
use crate::ds::bitset::BitSet;
use crate::ds::leveled_gss::Merge;
use crate::grammar::flat::TerminalID;
use rustc_hash::FxHashSet;
use smallvec::SmallVec;

#[test]
fn provider_may_advance_fastpath_matches_exact_reference() {
    use super::stack_may_advance_on_with_provider;
    // Exact old expression, kept as the reference (unchanged semantics).
    fn exact<P: ParserActionProvider>(
        provider: &P,
        stack: &ParserGSS,
        symbol: P::Symbol,
    ) -> bool {
        if stack.is_empty() {
            return false;
        }
        !advance_stacks_with_provider(
            provider,
            close_provider_control_stacks(provider, stack),
            symbol,
        )
        .is_empty()
    }

    fn check<P: ParserActionProvider>(
        provider: &P,
        stack: &ParserGSS,
        symbol: P::Symbol,
        expect: bool,
        case: &str,
    ) {
        assert_eq!(
            stack_may_advance_on_with_provider(provider, stack, symbol),
            expect,
            "optimized predicate disagrees with expected ({case})"
        );
        assert_eq!(
            exact(provider, stack, symbol),
            expect,
            "reference disagrees with expected (fixture wrong? {case})"
        );
        assert_eq!(
            super::stack_may_advance_on_any_with_provider(provider, stack, [symbol]),
            expect,
            "batched singleton differs ({case})"
        );
        assert_eq!(
            super::stack_may_advance_on_any_with_provider(provider, stack, [symbol, symbol]),
            expect,
            "batched duplicate differs ({case})"
        );
        assert!(!super::stack_may_advance_on_any_with_provider(
            provider, stack, std::iter::empty::<P::Symbol>(),
        ));
        let mut admitted = Vec::new();
        super::for_each_admitted_symbol_with_provider(
            provider, stack, [(0u32, symbol),(1u32, symbol)],
            |key| admitted.push(key),
        );
        assert_eq!(admitted, if expect { vec![0,1] } else { vec![] },
            "batched exact admission differs ({case})");
        let mut predicate_calls = 0;
        let found = super::find_admitted_symbol_with_provider(
            provider, stack, [(0u32, symbol), (1u32, symbol)],
            |&key| { predicate_calls += 1; key == 0 },
        );
        assert_eq!(found, expect.then_some(0), "lazy witness differs ({case})");
        assert_eq!(predicate_calls, usize::from(expect),
            "predicate must run only for admitted symbols and stop at a witness ({case})");
        let found = super::find_admitted_symbol_with_provider(
            provider, stack, [(0u32, symbol), (1u32, symbol)], |&key| key == 1,
        );
        assert_eq!(found, expect.then_some(1), "later lazy witness differs ({case})");
        assert!(super::find_admitted_symbol_with_provider(
            provider, stack, [(0u32, symbol)], |_| false,
        ).is_none(), "parser admission alone must not satisfy the predicate ({case})");
    }

    // Plain zero-pop shift: sufficient fast path, true.
    let plain = build_test_table(
        2,
        2,
        &[&[(0, Action::Shift(1, false))], &[]],
        &[&[], &[]],
    );
    let provider = GLRTableActionProvider::new(&plain);
    let stack = ParserGSS::from_single_stack(vec![0], TerminalsDisallowed::new());
    check(&provider, &stack, 0, true, "plain-true");
    check(&provider, &stack, 1, false, "plain-false");

    // Split accept+shift: shift alternative suffices, true.
    let split = build_test_table(
        3,
        2,
        &[&[(
            0,
            Action::Split {
                shift: Some((1, false)),
                reduces: vec![(0, 1)],
                accept: true,
            },
        )], &[], &[]],
        &[&[(0, (2, false))], &[], &[]],
    );
    let provider = GLRTableActionProvider::new(&split);
    check(&provider, &stack, 0, true, "split");

    // Accept only: no shift, false.
    let accept_only = build_test_table(2, 2, &[&[(0, Action::Accept)], &[]], &[&[], &[]]);
    let provider = GLRTableActionProvider::new(&accept_only);
    check(&provider, &stack, 0, false, "accept-only");

    // Empty input: false without touching the provider.
    let empty = ParserGSS::empty();
    check(&provider, &empty, 0, false, "empty");

    // Skip stays on the reference path (fast path requires nonempty
    // pushes); both must agree it advances.
    let skip = build_test_table(2, 2, &[&[(0, Action::Skip)], &[]], &[&[], &[]]);
    let provider = GLRTableActionProvider::new(&skip);
    check(&provider, &stack, 0, true, "skip");

    // Zero-pop StackShifts with mapped pushes: true.
    let shifts = build_test_table(
        3,
        2,
        &[&[(
            0,
            Action::StackShifts(vec![
                StackShift { pop: 1, pushes: vec![2] },
                StackShift { pop: 0, pushes: vec![1] },
            ]),
        )], &[], &[]],
        &[&[], &[], &[]],
    );
    let provider = GLRTableActionProvider::new(&shifts);
    check(&provider, &stack, 0, true, "zeropop-shift");

    // Nonzero-pop shift: the fallback decides. NOTE: this fixture's lone
    // single-stack CANNOT pop (pop(1) on [0] saturates at the root, so
    // the reference admits it) — the fast path must not claim pop>0.
    let pop1 = build_test_table(
        2,
        2,
        &[&[(0, Action::StackShifts(vec![StackShift { pop: 1, pushes: vec![1] }]))], &[]],
        &[&[], &[]],
    );
    let provider = GLRTableActionProvider::new(&pop1);
    let lone = ParserGSS::from_single_stack(vec![0], TerminalsDisallowed::new());
    check(&provider, &lone, 0, true, "pop1-saturated");

    // Reduce-to-shift via feasible goto: fallback computes true.
    let red = build_test_table(
        3,
        2,
        &[&[(0, Action::Reduce(0, 1))], &[(0, Action::Shift(2, false))], &[]],
        &[&[(0, (1, false))], &[], &[]],
    );
    let provider = GLRTableActionProvider::new(&red);
    let deep = ParserGSS::from_single_stack(vec![0, 0], TerminalsDisallowed::new());
    check(&provider, &deep, 0, true, "reduce-shift");

    // Infeasible reduction (no goto): false.
    let nogoto = build_test_table(
        2,
        2,
        &[&[(0, Action::Reduce(0, 1))], &[]],
        &[&[], &[]],
    );
    let provider = GLRTableActionProvider::new(&nogoto);
    check(&provider, &deep, 0, false, "infeasible-reduce");

    // Guarded shift, guard matches: fallback true; guard fails: false.
    let guarded = build_test_table(
        3,
        2,
        &[&[(
            0,
            Action::GuardedStackShifts(vec![GuardedStackShift {
                guards: vec![StackShiftGuard { pop: 1, states: vec![0] }],
                pop: 1,
                pushes: vec![2],
            }]),
        )], &[], &[]],
        &[&[], &[], &[]],
    );
    let provider = GLRTableActionProvider::new(&guarded);
    check(&provider, &deep, 0, true, "guarded-pass");
    let bad = ParserGSS::from_single_stack(vec![1, 0], TerminalsDisallowed::new());
    check(&provider, &bad, 0, false, "guarded-fail");

    // Identity via disjoint-provider ignore terminal: fast path true.
    let ws_parent = build_test_table(2, 2, &[&[(0, Action::Shift(1, false))], &[]], &[&[], &[]]);
    struct WsSource<'a> {
        table: &'a GLRTable,
    }
    impl ParserComponentTableSource for WsSource<'_> {
        fn component_count(&self) -> usize {
            1
        }
        fn component_table(&self, component: u32) -> Option<&GLRTable> {
            (component == 0).then_some(self.table)
        }
        fn component_ignore_terminal(&self, component: u32) -> Option<TerminalID> {
            (component == 0).then_some(1)
        }
    }
    let ws_source = WsSource { table: &ws_parent };
    let ws_links: [ScopedSubgrammarLink; 0] = [];
    let ws_provider =
        DisjointComponentActionProvider::new(&ws_source, &ws_links).unwrap();
    let ws_top = ws_provider.scoped_state(0, 0).unwrap();
    let ws_stack =
        ParserGSS::from_single_stack(vec![ws_top], TerminalsDisallowed::new());
    check(
        &ws_provider,
        &ws_stack,
        ScopedParserSymbol::Terminal { component: 0, terminal: 1 },
        true,
        "identity-ignore",
    );

    // Unmapped PUSH target on a valid top: provider denies the mapping,
    // so the alternative is infeasible — both must agree false.
    struct DenyTargetProvider<'a> {
        inner: GLRTableActionProvider<'a>,
    }
    impl ParserActionProvider for DenyTargetProvider<'_> {
        type Symbol = TerminalID;
        fn action(&self, state: u32, symbol: TerminalID) -> Option<ProvidedAction<'_>> {
            self.inner.action(state, symbol)
        }
        fn scope_state(&self, _scope: u32, _local: u32) -> Option<u32> {
            None
        }
        fn goto_target(
            &self,
            scope: u32,
            goto_from: u32,
            nt: u32,
        ) -> Option<(u32, bool)> {
            self.inner.goto_target(scope, goto_from, nt)
        }
        fn state_count_hint(&self) -> usize {
            self.inner.state_count_hint()
        }
    }
    let deny_table = build_test_table(
        2,
        2,
        &[&[(0, Action::Shift(1, false))], &[]],
        &[&[], &[]],
    );
    let deny_inner = GLRTableActionProvider::new(&deny_table);
    let deny_provider = DenyTargetProvider { inner: deny_inner };
    check(&deny_provider, &stack, 0, false, "unmapped-push");

    // Unknown-scope top (provider.action None): false, no wrong short-circuit.
    let parent = build_test_table(2, 2, &[&[(0, Action::Shift(1, false))], &[]], &[&[], &[]]);
    let child = build_test_table(2, 2, &[&[(0, Action::Shift(1, false))], &[]], &[&[], &[]]);
    let components = [&parent, &child];
    let links: [ScopedSubgrammarLink; 0] = [];
    let provider = DisjointComponentActionProvider::new(&components, &links).unwrap();
    let foreign = ParserGSS::from_single_stack(vec![u32::MAX - 1], TerminalsDisallowed::new());
    check(
        &provider,
        &foreign,
        ScopedParserSymbol::Terminal { component: 0, terminal: 0 },
        false,
        "unknown-scope",
    );
    // Same disjoint provider, valid mapped shift: true.
    let p0 = provider.scoped_state(0, 0).unwrap();
    let ok = ParserGSS::from_single_stack(vec![p0], TerminalsDisallowed::new());
    check(
        &provider,
        &ok,
        ScopedParserSymbol::Terminal { component: 0, terminal: 0 },
        true,
        "disjoint-mapped-shift",
    );
}

#[test]
fn disjoint_component_provider_parses_across_link_without_composed_table() {
    let slot = 0;
    let parent_tail = 1;
    let child_token = 0;
    let parent = build_test_table(
        3,
        2,
        &[
            &[(slot, Action::Shift(1, false))],
            &[(parent_tail, Action::Shift(2, false))],
            &[],
        ],
        &[&[], &[], &[]],
    );
    let child = build_test_table(
        2,
        1,
        &[
            &[(child_token, Action::Shift(1, false))],
            &[(EOF, Action::Accept)],
        ],
        &[&[], &[]],
    );
    let components = [&parent, &child];
    let links = [ScopedSubgrammarLink {
        parent_component: 0,
        slot_terminal: slot,
        child_component: 1,
        child_start: 0,
        return_pop: 2,
        child_start_nullable: false,
    }];
    let provider = DisjointComponentActionProvider::new(&components, &links).unwrap();
    let p0 = provider.scoped_state(0, 0).unwrap();
    let p1 = provider.scoped_state(0, 1).unwrap();
    let p2 = provider.scoped_state(0, 2).unwrap();
    let c0 = provider.scoped_state(1, 0).unwrap();
    let c1 = provider.scoped_state(1, 1).unwrap();
    let start = ParserGSS::from_single_stack(vec![p0], TerminalsDisallowed::new());

    let called = advance_stacks_with_provider(
        &provider,
        start,
        ScopedParserSymbol::Entry { link: 0 },
    );
    assert_eq!(called.single_top_value(), Some(c0));
    assert!(called
        .to_stacks(8)
        .unwrap()
        .iter()
        .any(|(stack, _)| stack == &vec![p0, p1, c0]));

    let child_advanced = advance_stacks_with_provider(
        &provider,
        called,
        ScopedParserSymbol::Terminal {
            component: 1,
            terminal: child_token,
        },
    );
    assert_eq!(child_advanced.single_top_value(), Some(c1));

    let returned = advance_stacks_with_provider(
        &provider,
        child_advanced,
        ScopedParserSymbol::Finish { component: 1 },
    );
    assert_eq!(returned.single_top_value(), Some(p1));

    let finished = advance_stacks_with_provider(
        &provider,
        returned,
        ScopedParserSymbol::Terminal {
            component: 0,
            terminal: parent_tail,
        },
    );
    assert_eq!(finished.single_top_value(), Some(p2));
}

#[test]
fn materialized_scoped_provider_table_preserves_call_return_stack_language() {
    let slot = 0;
    let parent_tail = 1;
    let child_token = 0;
    let parent = build_test_table(
        3,
        2,
        &[
            &[(slot, Action::Shift(1, false))],
            &[(parent_tail, Action::Shift(2, false))],
            &[],
        ],
        &[&[], &[], &[]],
    );
    let child = build_test_table(
        2,
        1,
        &[
            &[(child_token, Action::Shift(1, false))],
            &[(EOF, Action::Accept)],
        ],
        &[&[], &[]],
    );
    let components = [&parent, &child];
    let links = [ScopedSubgrammarLink {
        parent_component: 0,
        slot_terminal: slot,
        child_component: 1,
        child_start: 0,
        return_pop: 2,
        child_start_nullable: false,
    }];
    let provider = DisjointComponentActionProvider::new(&components, &links).unwrap();
    let p0 = provider.scoped_state(0, 0).unwrap();
    let p2 = provider.scoped_state(0, 2).unwrap();
    let terminal_symbols = vec![
        SmallVec::from_slice(&[ScopedParserSymbol::Terminal {
            component: 1,
            terminal: child_token,
        }]),
        SmallVec::from_slice(&[ScopedParserSymbol::Terminal {
            component: 0,
            terminal: parent_tail,
        }]),
    ];
    let table = materialize_control_eliminated_scoped_provider_table(
        &provider,
        &terminal_symbols,
    )
    .unwrap();
    assert!(table.control_terminals.is_empty());
    assert_eq!(table.num_states, provider.state_count_hint() as u32);
    assert_eq!(table.num_terminals, terminal_symbols.len() as u32);

    let start = ParserGSS::from_single_stack(vec![p0], TerminalsDisallowed::new());
    let provider_after_child = advance_provider_control_closed_stacks(
        &provider,
        &close_provider_control_stacks(&provider, &start),
        terminal_symbols[0][0],
    );
    let table_after_child = advance_stacks(&table, &start, 0);
    assert!(!provider_after_child.is_empty());
    assert!(!table_after_child.is_empty());

    let provider_finished = advance_provider_control_closed_stacks(
        &provider,
        &provider_after_child,
        terminal_symbols[1][0],
    );
    let table_finished = advance_stacks(&table, &table_after_child, 1);
    assert!(!provider_finished.is_empty());
    assert!(!table_finished.is_empty());
    assert_eq!(table_finished.single_top_value(), Some(p2));
}

#[test]
fn materialized_scoped_provider_table_preserves_midword_call_and_return() {
    let x = 0;
    let slot = 1;
    let z = 2;
    let y = 0;
    let child_nt = 0;
    let parent = build_test_table(
        4,
        3,
        &[
            &[(x, Action::Shift(1, false))],
            &[(slot, Action::Shift(2, false))],
            &[(z, Action::Shift(3, false))],
            &[],
        ],
        &[&[], &[], &[], &[]],
    );
    let child = build_test_table(
        3,
        1,
        &[
            &[(y, Action::Shift(1, false))],
            &[(EOF, Action::Reduce(child_nt, 1))],
            &[(EOF, Action::Accept)],
        ],
        &[&[(child_nt, (2, false))], &[], &[]],
    );
    let components = [&parent, &child];
    let links = [ScopedSubgrammarLink {
        parent_component: 0,
        slot_terminal: slot,
        child_component: 1,
        child_start: 0,
        return_pop: 2,
        child_start_nullable: false,
    }];
    let provider = DisjointComponentActionProvider::new(&components, &links).unwrap();
    let p0 = provider.scoped_state(0, 0).unwrap();
    let p3 = provider.scoped_state(0, 3).unwrap();
    let terminal_symbols = vec![
        SmallVec::from_slice(&[ScopedParserSymbol::Terminal {
            component: 0,
            terminal: x,
        }]),
        SmallVec::from_slice(&[ScopedParserSymbol::Terminal {
            component: 1,
            terminal: y,
        }]),
        SmallVec::from_slice(&[ScopedParserSymbol::Terminal {
            component: 0,
            terminal: z,
        }]),
    ];
    let table = materialize_control_eliminated_scoped_provider_table(
        &provider,
        &terminal_symbols,
    )
    .unwrap();

    let mut provider_stack =
        ParserGSS::from_single_stack(vec![p0], TerminalsDisallowed::new());
    let mut table_stack = provider_stack.clone();
    for (terminal, symbol) in terminal_symbols.iter().enumerate() {
        provider_stack = advance_provider_control_closed_stacks(
            &provider,
            &provider_stack,
            symbol[0],
        );
        table_stack = advance_stacks(&table, &table_stack, terminal as u32);
        assert!(!provider_stack.is_empty(), "provider rejected terminal {terminal}");
        assert!(!table_stack.is_empty(), "table rejected terminal {terminal}");
    }
    assert_eq!(
        normalized_concrete_stacks(&provider_stack),
        normalized_concrete_stacks(&table_stack),
        "provider/table visible-word result diverged",
    );
    assert_eq!(table_stack.single_top_value(), Some(p3));
}

#[test]
fn materialized_scoped_provider_table_preserves_nullable_finish_branch() {
    let slot = 0;
    let tail = 1;
    let parent = build_test_table(
        3,
        2,
        &[
            &[(slot, Action::Shift(1, false))],
            &[(tail, Action::Shift(2, false))],
            &[],
        ],
        &[&[], &[], &[]],
    );
    let child = build_test_table(1, 0, &[&[]], &[&[]]);
    let components = [&parent, &child];
    let links = [ScopedSubgrammarLink {
        parent_component: 0,
        slot_terminal: slot,
        child_component: 1,
        child_start: 0,
        return_pop: 2,
        child_start_nullable: true,
    }];
    let provider = DisjointComponentActionProvider::new(&components, &links).unwrap();
    let p0 = provider.scoped_state(0, 0).unwrap();
    let p2 = provider.scoped_state(0, 2).unwrap();
    let terminal_symbols = vec![SmallVec::from_slice(&[ScopedParserSymbol::Terminal {
        component: 0,
        terminal: tail,
    }])];
    let table = materialize_control_eliminated_scoped_provider_table(
        &provider,
        &terminal_symbols,
    )
    .unwrap();
    let start = ParserGSS::from_single_stack(vec![p0], TerminalsDisallowed::new());
    let provider_finished = advance_provider_control_closed_stacks(
        &provider,
        &close_provider_control_stacks(&provider, &start),
        terminal_symbols[0][0],
    );
    let table_finished = advance_stacks(&table, &start, 0);
    assert_eq!(
        normalized_concrete_stacks(&provider_finished),
        normalized_concrete_stacks(&table_finished),
    );
    assert_eq!(table_finished.single_top_value(), Some(p2));
}

#[test]
fn disjoint_local_guarded_shift_respects_component_scope() {
    let token = 0;
    let parent = build_test_table(1, 0, &[&[]], &[&[]]);
    let child = build_test_table(
        3,
        1,
        &[
            &[],
            &[(token, Action::GuardedStackShifts(vec![GuardedStackShift {
                guards: vec![StackShiftGuard { pop: 1, states: vec![0] }],
                pop: 1,
                pushes: vec![2],
            }]))],
            &[],
        ],
        &[&[], &[], &[]],
    );
    let components = [&parent, &child];
    let provider = DisjointComponentActionProvider::new(&components, &[]).unwrap();
    let p0 = provider.scoped_state(0, 0).unwrap();
    let c0 = provider.scoped_state(1, 0).unwrap();
    let c1 = provider.scoped_state(1, 1).unwrap();
    let c2 = provider.scoped_state(1, 2).unwrap();

    let matching = ParserGSS::from_single_stack(vec![c0, c1], TerminalsDisallowed::new());
    let shifted = advance_stacks_with_provider(
        &provider,
        matching,
        ScopedParserSymbol::Terminal {
            component: 1,
            terminal: token,
        },
    );
    assert_eq!(shifted.single_top_value(), Some(c2));

    let cross_scope =
        ParserGSS::from_single_stack(vec![p0, c1], TerminalsDisallowed::new());
    let rejected = advance_stacks_with_provider(
        &provider,
        cross_scope,
        ScopedParserSymbol::Terminal {
            component: 1,
            terminal: token,
        },
    );
    assert!(rejected.is_empty());
}

#[test]
fn disjoint_provider_preserves_reductions_under_entry_and_finish() {
    let slot = 0;
    let child_token = 0;
    let nt = 0;
    let parent = build_test_table(
        4,
        1,
        &[
            &[],
            &[(slot, Action::Reduce(nt, 1))],
            &[(slot, Action::Shift(3, false))],
            &[],
        ],
        &[&[(nt, (2, false))], &[], &[], &[]],
    );
    let child = build_test_table(
        3,
        1,
        &[
            &[(child_token, Action::Shift(1, false))],
            &[(EOF, Action::Reduce(nt, 1))],
            &[(EOF, Action::Accept)],
        ],
        &[&[(nt, (2, false))], &[], &[]],
    );
    let components = [&parent, &child];
    let links = [ScopedSubgrammarLink {
        parent_component: 0,
        slot_terminal: slot,
        child_component: 1,
        child_start: 0,
        return_pop: 2,
        child_start_nullable: false,
    }];
    let provider = DisjointComponentActionProvider::new(&components, &links).unwrap();
    let p0 = provider.scoped_state(0, 0).unwrap();
    let p1 = provider.scoped_state(0, 1).unwrap();
    let p2 = provider.scoped_state(0, 2).unwrap();
    let p3 = provider.scoped_state(0, 3).unwrap();
    let c0 = provider.scoped_state(1, 0).unwrap();
    let start = ParserGSS::from_single_stack(vec![p0, p1], TerminalsDisallowed::new());

    let called = advance_stacks_with_provider(
        &provider,
        start,
        ScopedParserSymbol::Entry { link: 0 },
    );
    assert_eq!(called.single_top_value(), Some(c0));
    assert!(called
        .to_stacks(8)
        .unwrap()
        .iter()
        .any(|(stack, _)| stack == &vec![p0, p2, p3, c0]));

    let child_advanced = advance_stacks_with_provider(
        &provider,
        called,
        ScopedParserSymbol::Terminal {
            component: 1,
            terminal: child_token,
        },
    );
    let returned = advance_stacks_with_provider(
        &provider,
        child_advanced,
        ScopedParserSymbol::Finish { component: 1 },
    );
    assert_eq!(returned.single_top_value(), Some(p3));
}

#[test]
fn disjoint_provider_nullable_child_can_return_immediately() {
    let slot = 0;
    let parent = build_test_table(
        2,
        1,
        &[&[(slot, Action::Shift(1, false))], &[]],
        &[&[], &[]],
    );
    let child = build_test_table(1, 0, &[&[]], &[&[]]);
    let components = [&parent, &child];
    let links = [ScopedSubgrammarLink {
        parent_component: 0,
        slot_terminal: slot,
        child_component: 1,
        child_start: 0,
        return_pop: 2,
        child_start_nullable: true,
    }];
    let provider = DisjointComponentActionProvider::new(&components, &links).unwrap();
    let p0 = provider.scoped_state(0, 0).unwrap();
    let p1 = provider.scoped_state(0, 1).unwrap();
    let c0 = provider.scoped_state(1, 0).unwrap();
    let start = ParserGSS::from_single_stack(vec![p0], TerminalsDisallowed::new());
    let closed = close_provider_control_stacks(&provider, &start);
    let stacks = closed.to_stacks(16).unwrap();
    assert!(stacks.iter().any(|(stack, _)| stack == &vec![p0, p1]));
    assert!(stacks
        .iter()
        .any(|(stack, _)| stack == &vec![p0, p1, c0]));
}

#[test]
fn uniform_deterministic_frontier_preserves_divergent_lower_prefixes() {
    let token = 0;
    let nt0 = 0;
    let nt1 = 1;
    let mut action_rows = vec![Vec::<(u32, Action)>::new(); 10];
    action_rows[9].push((token, Action::Reduce(nt0, 1)));
    action_rows[6].push((token, Action::Reduce(nt1, 1)));
    action_rows[7].push((token, Action::Shift(8, false)));
    let action_refs = action_rows
        .iter()
        .map(Vec::as_slice)
        .collect::<Vec<_>>();

    let mut goto_rows = vec![Vec::<(u32, (u32, bool))>::new(); 10];
    goto_rows[4].push((nt0, (6, true)));
    goto_rows[1].push((nt1, (7, true)));
    let goto_refs = goto_rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let table = build_test_table(10, 1, &action_refs, &goto_refs);

    let acc = TerminalsDisallowed::new();
    let before = ParserGSS::from_stacks(&[
        (vec![0, 2, 1, 4, 9], acc.clone()),
        (vec![0, 3, 1, 4, 9], acc.clone()),
        (vec![0, 5, 1, 4, 9], acc),
    ]);
    let fast = try_advance_uniform_deterministic_frontier(&table, &before, token)
        .expect("uniform reduction chain should be structural");
    let expected = advance_concrete_stacks_reference(&table, &before, token);

    assert_eq!(
        normalized_concrete_stacks(&fast),
        normalized_concrete_stacks(&expected),
    );
    assert_eq!(
        normalized_concrete_stacks(&fast),
        vec![
            (vec![0, 2, 7, 8], TerminalsDisallowed::new()),
            (vec![0, 3, 7, 8], TerminalsDisallowed::new()),
            (vec![0, 5, 7, 8], TerminalsDisallowed::new()),
        ],
    );
}

#[test]
fn concrete_advance_reference_applies_whole_stack_effect_atomically() {
    let token = 0;
    let table = build_test_table(
        2,
        1,
        &[
            &[],
            &[(
                token,
                Action::StackShifts(vec![StackShift {
                    pop: 2,
                    pushes: vec![7],
                }]),
            )],
        ],
        &[&[], &[]],
    );
    let before = ParserGSS::from_single_stack(
        vec![0, 1],
        TerminalsDisallowed::new(),
    );
    let expected = ParserGSS::from_single_stack(
        vec![7],
        TerminalsDisallowed::new(),
    );

    assert_eq!(
        advance_concrete_stacks_reference(&table, &before, token),
        expected,
    );
}

#[test]
fn concrete_advance_reference_merges_accumulators_at_reduce_closure_join() {
    let token = 0;
    let nt = 0;
    let table = build_test_table(
        6,
        1,
        &[
            &[],
            &[],
            &[(token, Action::Reduce(nt, 1))],
            &[(token, Action::Reduce(nt, 1))],
            &[(token, Action::Shift(5, false))],
            &[],
        ],
        &[&[(nt, (4, false))], &[], &[], &[], &[], &[]],
    );
    let left_acc = TerminalsDisallowed::new().with_insert(10, 20);
    let right_acc = TerminalsDisallowed::new().with_insert(11, 21);
    let before = ParserGSS::from_stacks(&[
        (vec![0, 2], left_acc.clone()),
        (vec![0, 3], right_acc.clone()),
    ]);
    let expected = ParserGSS::from_single_stack(
        vec![0, 4, 5],
        left_acc.merge(&right_acc),
    );

    assert_eq!(
        advance_concrete_stacks_reference(&table, &before, token),
        expected,
    );
}

#[test]
fn advance_stacks_matches_reduce_fanout_collapse_fast_path() {
    let token = 0;
    let nt = 0;
    let table = build_test_table(
        5,
        1,
        &[
            &[],
            &[],
            &[(token, Action::StackShifts(vec![StackShift { pop: 2, pushes: vec![7] }]))],
            &[(token, Action::StackShifts(vec![StackShift { pop: 2, pushes: vec![7] }]))],
            &[(token, Action::Reduce(nt, 1))],
        ],
        &[
            &[(nt, (2, false))],
            &[(nt, (3, false))],
            &[],
            &[],
            &[],
        ],
    );

    let acc = TerminalsDisallowed::new();
    let before = ParserGSS::from_stacks(&[
        (vec![0, 4], acc.clone()),
        (vec![1, 4], acc),
    ]);
    let expected = ParserGSS::from_single_stack(vec![7], TerminalsDisallowed::new());

    assert_eq!(advance_stacks(&table, &before, token), expected);
}

#[test]
fn advance_stacks_selective_pure_frontier_shift_keeps_only_actionable_top() {
    let token = 0;
    let mut action_rows = vec![Vec::new(); 134];
    action_rows[131] = vec![(
        token,
        Action::StackShifts(vec![StackShift {
            pop: 0,
            pushes: vec![96],
        }]),
    )];
    let action_refs: Vec<&[(u32, Action)]> =
        action_rows.iter().map(|row| row.as_slice()).collect();
    let goto_rows = vec![Vec::new(); 134];
    let goto_refs: Vec<&[(u32, (u32, bool))]> =
        goto_rows.iter().map(|row| row.as_slice()).collect();
    let table = build_test_table(134, 1, &action_refs, &goto_refs);

    let acc = TerminalsDisallowed::new();
    let before = ParserGSS::from_stacks(&[
        (vec![0_u32, 1, 17, 47, 74, 131], acc.clone()),
        (vec![0_u32, 1, 17, 47, 74, 132], acc.clone()),
        (vec![0_u32, 1, 17, 47, 74, 133], acc),
    ]);
    let expected = ParserGSS::from_single_stack(
        vec![0_u32, 1, 17, 47, 74, 131, 96],
        TerminalsDisallowed::new(),
    );

    assert_eq!(advance_stacks(&table, &before, token), expected);
}

#[test]
fn single_active_pop1_stackshift_wave_discards_dead_top_without_cross_product() {
    let token = 0;
    let mut action_rows = vec![Vec::new(); 500];
    action_rows[452] = vec![(
        token,
        Action::StackShifts(vec![StackShift {
            pop: 1,
            pushes: vec![384, 411],
        }]),
    )];
    let action_refs: Vec<&[(u32, Action)]> =
        action_rows.iter().map(|row| row.as_slice()).collect();
    let goto_rows = vec![Vec::new(); 500];
    let goto_refs: Vec<&[(u32, (u32, bool))]> =
        goto_rows.iter().map(|row| row.as_slice()).collect();
    let table = build_test_table(500, 1, &action_refs, &goto_refs);

    let acc = TerminalsDisallowed::new();
    let before = ParserGSS::from_stacks(&[
        (vec![0, 1, 186, 40, 85, 123, 452], acc.clone()),
        (vec![0, 1, 202, 40, 85, 123, 452], acc.clone()),
        (vec![0, 1, 322, 92, 150, 426], acc),
    ]);
    let expected = ParserGSS::from_stacks(&[
        (
            vec![0, 1, 186, 40, 85, 123, 384, 411],
            TerminalsDisallowed::new(),
        ),
        (
            vec![0, 1, 202, 40, 85, 123, 384, 411],
            TerminalsDisallowed::new(),
        ),
    ]);

    let fast = try_advance_single_active_pop1_stackshift_wave(&table, &before, token)
        .expect("single live stack-shift branch should be structural");
    let mut fast_stacks = fast
        .to_stacks(16)
        .expect("bounded structural result should enumerate");
    let mut expected_stacks = expected
        .to_stacks(16)
        .expect("bounded expected result should enumerate");
    fast_stacks.sort_by(|left, right| left.0.cmp(&right.0));
    expected_stacks.sort_by(|left, right| left.0.cmp(&right.0));
    assert_eq!(fast_stacks, expected_stacks);

    let mut actual_stacks = advance_stacks(&table, &before, token)
        .to_stacks(16)
        .expect("bounded advanced result should enumerate");
    actual_stacks.sort_by(|left, right| left.0.cmp(&right.0));
    assert_eq!(actual_stacks, expected_stacks);
}

#[test]
fn pop1_reduce_and_guarded_shift_wave_remaps_predecessors_structurally() {
    let token = 0;
    let nt = 0;
    let mut action_rows = vec![Vec::new(); 400];
    action_rows[133] = vec![(
        token,
        Action::GuardedStackShifts(vec![GuardedStackShift {
            guards: vec![
                StackShiftGuard {
                    pop: 1,
                    states: vec![322],
                },
                StackShiftGuard {
                    pop: 2,
                    states: vec![10],
                },
            ],
            pop: 2,
            pushes: vec![327],
        }]),
    )];
    action_rows[137] = vec![(token, Action::Reduce(nt, 1))];
    action_rows[266] = vec![(token, Action::Shift(351, true))];
    action_rows[284] = vec![(token, Action::Shift(360, true))];
    let action_refs: Vec<&[(u32, Action)]> =
        action_rows.iter().map(|row| row.as_slice()).collect();

    let mut goto_rows = vec![Vec::new(); 400];
    goto_rows[168] = vec![(nt, (266, true))];
    goto_rows[181] = vec![(nt, (284, true))];
    let goto_refs: Vec<&[(u32, (u32, bool))]> =
        goto_rows.iter().map(|row| row.as_slice()).collect();
    let table = build_test_table(400, 1, &action_refs, &goto_refs);

    let acc = TerminalsDisallowed::new();
    let before = ParserGSS::from_stacks(&[
        (vec![0, 10, 168, 137], acc.clone()),
        (vec![0, 10, 181, 137], acc.clone()),
        (vec![0, 10, 322, 133], acc),
    ]);
    let expected = ParserGSS::from_stacks(&[
        (vec![0, 10, 351], TerminalsDisallowed::new()),
        (vec![0, 10, 360], TerminalsDisallowed::new()),
        (vec![0, 10, 327], TerminalsDisallowed::new()),
    ]);

    let bounded = try_advance_bounded_concrete_paths(&table, &before, token)
        .expect("small mixed paths should be interpreted concretely");
    assert!(bounded.semantically_eq(&expected, 16).unwrap());

    let fast = try_advance_pop1_reduce_guarded_stackshift_wave(&table, &before, token)
        .expect("mixed wave should be handled structurally");
    assert!(fast.semantically_eq(&expected, 16).unwrap());
    assert!(advance_stacks(&table, &before, token)
        .semantically_eq(&expected, 16)
        .unwrap());
}

#[test]
fn pop1_reduce_plus_stackshift_wave_fast_path_matches_snowplow_shape() {
    let token = 0;
    let nt = 0;
    let mut action_rows = vec![Vec::new(); 989];
    action_rows[655] = vec![(
        token,
        Action::StackShifts(vec![StackShift { pop: 1, pushes: vec![975] }]),
    )];
    action_rows[659] = vec![(
        token,
        Action::StackShifts(vec![
            StackShift { pop: 1, pushes: vec![654] },
            StackShift { pop: 1, pushes: vec![988] },
        ]),
    )];
    action_rows[987] = vec![(token, Action::Reduce(nt, 1))];
    let action_refs: Vec<&[(u32, Action)]> =
        action_rows.iter().map(|row| row.as_slice()).collect();

    let mut goto_rows = vec![Vec::new(); 989];
    goto_rows[87] = vec![(nt, (659, true))];
    let goto_refs: Vec<&[(u32, (u32, bool))]> =
        goto_rows.iter().map(|row| row.as_slice()).collect();

    let table = build_test_table(989, 1, &action_refs, &goto_refs);
    let acc = TerminalsDisallowed::new();
    let before = ParserGSS::from_stacks(&[
        (vec![0_u32, 87, 987], acc.clone()),
        (vec![0_u32, 87, 655], acc),
    ]);
    let expected = ParserGSS::from_stacks(&[
        (vec![0_u32, 87, 975], TerminalsDisallowed::new()),
        (vec![0_u32, 654], TerminalsDisallowed::new()),
        (vec![0_u32, 988], TerminalsDisallowed::new()),
    ]);

    let mut fast_stacks = try_advance_pop1_reduce_plus_stackshift_wave(&table, &before, token)
        .expect("fast path should match this wave")
        .to_stacks(4_096).expect("stack enumeration exceeded explicit limit");
    let mut expected_stacks = expected.to_stacks(4_096).expect("stack enumeration exceeded explicit limit");
    fast_stacks.sort_by(|left, right| left.0.cmp(&right.0));
    expected_stacks.sort_by(|left, right| left.0.cmp(&right.0));
    assert_eq!(fast_stacks, expected_stacks);

    let mut actual_stacks = advance_stacks(&table, &before, token).to_stacks(4_096).expect("stack enumeration exceeded explicit limit");
    actual_stacks.sort_by(|left, right| left.0.cmp(&right.0));
    assert_eq!(actual_stacks, expected_stacks);
}

#[test]
fn pop1_reduce_plus_stackshift_wave_rejects_cross_product_base() {
    let token = 0;
    let nt = 0;
    let mut action_rows = vec![Vec::new(); 989];
    action_rows[655] = vec![(
        token,
        Action::StackShifts(vec![StackShift { pop: 1, pushes: vec![975] }]),
    )];
    action_rows[659] = vec![(
        token,
        Action::StackShifts(vec![
            StackShift { pop: 1, pushes: vec![654] },
            StackShift { pop: 1, pushes: vec![988] },
        ]),
    )];
    action_rows[987] = vec![(token, Action::Reduce(nt, 1))];
    let action_refs: Vec<&[(u32, Action)]> =
        action_rows.iter().map(|row| row.as_slice()).collect();

    let mut goto_rows = vec![Vec::new(); 989];
    goto_rows[87] = vec![(nt, (659, true))];
    let goto_refs: Vec<&[(u32, (u32, bool))]> =
        goto_rows.iter().map(|row| row.as_slice()).collect();

    let table = build_test_table(989, 1, &action_refs, &goto_refs);
    let acc = TerminalsDisallowed::new();
    let before = ParserGSS::from_stacks(&[
        (vec![0_u32, 87, 987], acc.clone()),
        (vec![1_u32, 87, 655], acc),
    ]);

    assert_eq!(
        try_advance_pop1_reduce_plus_stackshift_wave(&table, &before, token),
        None
    );
}

#[test]
fn may_advance_consults_admission_rows_not_execution_actions() {
    let token = 0;
    let mut table = build_test_table(
        2,
        1,
        &[&[], &[(token, Action::Shift(1, false))]],
        &[&[], &[]],
    );
    table.advance[1].clear(token as usize);

    let stack = ParserGSS::from_single_stack(vec![1], TerminalsDisallowed::new());
    assert!(table.action(1, token).is_some());
    assert!(!stack_may_advance_on(&table, &stack, token));

    let mut terminals = BitSet::new(1);
    terminals.set(token as usize);
    assert!(!stack_may_advance_on_any(&table, &stack, &terminals));
}

#[test]
fn may_advance_rechecks_guarded_stack_shifts_against_concrete_stack() {
    let token = 0;
    let mut table = build_test_table(
        3,
        1,
        &[
            &[],
            &[],
            &[(
                token,
                Action::GuardedStackShifts(vec![GuardedStackShift {
                    guards: vec![StackShiftGuard {
                        pop: 1,
                        states: vec![0],
                    }],
                    pop: 2,
                    pushes: vec![7],
                }]),
            )],
        ],
        &[&[], &[], &[]],
    );
    table.admission_policy = AdmissionPolicy::ExactSimulation;

    let stack = ParserGSS::from_single_stack(vec![1, 2], TerminalsDisallowed::new());

    assert!(table.advance_row_allows(2, token));
    assert!(advance_stacks(&table, &stack, token).is_empty());
    assert!(!stack_may_advance_on(&table, &stack, token));

    let mut terminals = BitSet::new(1);
    terminals.set(token as usize);
    assert!(!stack_may_advance_on_any(&table, &stack, &terminals));
}

#[test]
fn row_presence_admission_does_not_recheck_lowered_guarded_effects() {
    let token = 0;
    let table = build_test_table(
        3,
        1,
        &[
            &[],
            &[],
            &[(
                token,
                Action::GuardedStackShifts(vec![GuardedStackShift {
                    guards: vec![StackShiftGuard {
                        pop: 1,
                        states: vec![0],
                    }],
                    pop: 2,
                    pushes: vec![7],
                }]),
            )],
        ],
        &[&[], &[], &[]],
    );
    let stack = ParserGSS::from_single_stack(vec![1, 2], TerminalsDisallowed::new());

    assert_eq!(table.admission_policy, AdmissionPolicy::RowPresenceExact);
    assert!(table.advance_row_allows(2, token));
    assert!(stack_may_advance_on(&table, &stack, token));

    let mut terminals = BitSet::new(1);
    terminals.set(token as usize);
    assert!(stack_may_advance_on_any(&table, &stack, &terminals));
}

#[test]
fn guarded_predecessor_remap_preserves_branched_floor_correlation() {
    let acc = TerminalsDisallowed::new();
    let before = ParserGSS::from_stacks(&[
        (vec![0_u32, 353, 669, 800], acc.clone()),
        (vec![0_u32, 353, 711, 800], acc.clone()),
        (vec![0_u32, 353, 753, 800], acc),
    ]);
    let shifts = vec![
        GuardedStackShift {
            guards: vec![
                StackShiftGuard {
                    pop: 1,
                    states: vec![669],
                },
                StackShiftGuard {
                    pop: 2,
                    states: vec![353, 1503],
                },
            ],
            pop: 2,
            pushes: vec![1115],
        },
        GuardedStackShift {
            guards: vec![
                StackShiftGuard {
                    pop: 1,
                    states: vec![711],
                },
                StackShiftGuard {
                    pop: 2,
                    states: vec![353, 1504],
                },
            ],
            pop: 2,
            pushes: vec![1168],
        },
        GuardedStackShift {
            guards: vec![
                StackShiftGuard {
                    pop: 1,
                    states: vec![753],
                },
                StackShiftGuard {
                    pop: 2,
                    states: vec![353, 1505],
                },
            ],
            pop: 2,
            pushes: vec![1221],
        },
        GuardedStackShift {
            guards: vec![StackShiftGuard {
                pop: 1,
                states: vec![1348],
            }],
            pop: 2,
            pushes: vec![1457, 1497],
        },
        GuardedStackShift {
            guards: vec![
                StackShiftGuard {
                    pop: 1,
                    states: vec![1547],
                },
                StackShiftGuard {
                    pop: 3,
                    states: vec![1457],
                },
            ],
            pop: 3,
            pushes: vec![1498, 1497],
        },
    ];

    let fast = apply_guarded_stack_shifts_as_predecessor_remap(&before, &shifts)
        .expect("predecessor remap shape should be recognized");
    let general = apply_guarded_stack_shifts(before, &shifts, None);
    assert!(fast.semantically_eq(&general, 16).unwrap());

    let mut stacks = fast
        .to_stacks(16)
        .expect("three-path result should remain bounded");
    stacks.sort_by(|left, right| left.0.cmp(&right.0));
    assert_eq!(
        stacks
            .into_iter()
            .map(|(stack, _)| stack)
            .collect::<Vec<_>>(),
        vec![
            vec![0, 353, 1115],
            vec![0, 353, 1168],
            vec![0, 353, 1221],
        ],
    );
}

#[test]
fn bounded_guarded_effects_handle_machine_schema_branched_floor() {
    let acc = TerminalsDisallowed::new();
    let before = ParserGSS::from_stacks(&[
        (
            vec![0, 1, 10, 40, 87, 124, 85, 123, 186, 40, 85, 123, 322, 133],
            acc.clone(),
        ),
        (
            vec![0, 1, 10, 40, 87, 124, 85, 123, 202, 40, 85, 123, 322, 133],
            acc.clone(),
        ),
        (
            vec![
                0, 1, 10, 40, 87, 124, 85, 123, 322, 92, 150, 92, 250, 343, 426, 133,
            ],
            acc,
        ),
    ]);
    let shifts = vec![
        GuardedStackShift {
            guards: vec![
                StackShiftGuard {
                    pop: 1,
                    states: vec![50],
                },
                StackShiftGuard {
                    pop: 2,
                    states: vec![1],
                },
            ],
            pop: 2,
            pushes: vec![74],
        },
        GuardedStackShift {
            guards: vec![
                StackShiftGuard {
                    pop: 1,
                    states: vec![322],
                },
                StackShiftGuard {
                    pop: 2,
                    states: vec![123],
                },
            ],
            pop: 2,
            pushes: vec![327],
        },
        GuardedStackShift {
            guards: vec![
                StackShiftGuard {
                    pop: 1,
                    states: vec![426],
                },
                StackShiftGuard {
                    pop: 2,
                    states: vec![250],
                },
            ],
            pop: 2,
            pushes: vec![343, 342],
        },
        GuardedStackShift {
            guards: vec![
                StackShiftGuard {
                    pop: 1,
                    states: vec![426],
                },
                StackShiftGuard {
                    pop: 2,
                    states: vec![343],
                },
                StackShiftGuard {
                    pop: 3,
                    states: vec![250],
                },
            ],
            pop: 3,
            pushes: vec![343, 342],
        },
    ];
    let expected = ParserGSS::from_stacks(&[
        (
            vec![0, 1, 10, 40, 87, 124, 85, 123, 186, 40, 85, 123, 327],
            TerminalsDisallowed::new(),
        ),
        (
            vec![0, 1, 10, 40, 87, 124, 85, 123, 202, 40, 85, 123, 327],
            TerminalsDisallowed::new(),
        ),
        (
            vec![
                0, 1, 10, 40, 87, 124, 85, 123, 322, 92, 150, 92, 250, 343, 342,
            ],
            TerminalsDisallowed::new(),
        ),
    ]);

    let fast = apply_guarded_stack_shifts_fast(&before, &shifts, None)
        .expect("small branched floor should use bounded guarded effects");
    assert!(fast.semantically_eq(&expected, 16).unwrap());
}

#[test]
fn guarded_stack_shift_advance_distributes_over_merged_branched_floor() {
    let token = 0;
    let mut action_rows = vec![Vec::new(); 329];
    action_rows[74] = vec![(
        token,
        Action::GuardedStackShifts(vec![
            GuardedStackShift {
                guards: vec![StackShiftGuard {
                    pop: 1,
                    states: vec![171],
                }],
                pop: 2,
                pushes: vec![213, 265],
            },
            GuardedStackShift {
                guards: vec![StackShiftGuard {
                    pop: 1,
                    states: vec![323],
                }],
                pop: 2,
                pushes: vec![328, 370],
            },
        ]),
    )];
    let action_refs: Vec<&[(u32, Action)]> =
        action_rows.iter().map(Vec::as_slice).collect();
    let goto_rows = vec![Vec::new(); 329];
    let goto_refs: Vec<&[(u32, (u32, bool))]> =
        goto_rows.iter().map(Vec::as_slice).collect();
    let table = build_test_table(329, 1, &action_refs, &goto_refs);

    let acc = TerminalsDisallowed::new();
    let left = ParserGSS::from_single_stack(vec![0, 171, 74], acc.clone());
    let right = ParserGSS::from_single_stack(vec![0, 323, 74], acc);
    let merged = left.merge(&right);

    let expected = advance_stacks(&table, &left, token)
        .merge(&advance_stacks(&table, &right, token));
    let actual = advance_stacks(&table, &merged, token);

    let mut expected_stacks = expected.to_stacks(4_096).expect("stack enumeration exceeded explicit limit");
    let mut actual_stacks = actual.to_stacks(4_096).expect("stack enumeration exceeded explicit limit");
    expected_stacks.sort_by(|a, b| a.0.cmp(&b.0));
    actual_stacks.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(actual_stacks, expected_stacks);
    assert!(stack_may_advance_on(&table, &merged, token));
}

#[test]
fn stack_shift_advance_distributes_over_merged_branched_floor() {
    let token = 0;
    let mut action_rows = vec![Vec::new(); 75];
    action_rows[74] = vec![(
        token,
        Action::StackShifts(vec![StackShift {
            pop: 2,
            pushes: vec![265],
        }]),
    )];
    let action_refs: Vec<&[(u32, Action)]> =
        action_rows.iter().map(Vec::as_slice).collect();
    let goto_rows = vec![Vec::new(); 75];
    let goto_refs: Vec<&[(u32, (u32, bool))]> =
        goto_rows.iter().map(Vec::as_slice).collect();
    let table = build_test_table(75, 1, &action_refs, &goto_refs);

    let acc = TerminalsDisallowed::new();
    let left = ParserGSS::from_single_stack(vec![0, 171, 74], acc.clone());
    let right = ParserGSS::from_single_stack(vec![0, 323, 74], acc);
    let merged = left.merge(&right);

    let expected = advance_stacks(&table, &left, token)
        .merge(&advance_stacks(&table, &right, token));
    let actual = advance_stacks(&table, &merged, token);

    let mut expected_stacks = expected.to_stacks(4_096).expect("stack enumeration exceeded explicit limit");
    let mut actual_stacks = actual.to_stacks(4_096).expect("stack enumeration exceeded explicit limit");
    expected_stacks.sort_by(|a, b| a.0.cmp(&b.0));
    actual_stacks.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(actual_stacks, expected_stacks);
}

fn assert_advance_distributes_over_merge(
    table: &GLRTable,
    left: &ParserGSS,
    right: &ParserGSS,
    token: u32,
    case: &str,
) {
    let merged = left.merge(right);
    let expected = advance_stacks(table, left, token)
        .merge(&advance_stacks(table, right, token));
    let actual = advance_stacks(table, &merged, token);
    let concrete_reference = advance_concrete_stacks_reference(table, &merged, token);

    let mut expected_stacks = expected.to_stacks(4_096).expect("stack enumeration exceeded explicit limit");
    let mut actual_stacks = actual.to_stacks(4_096).expect("stack enumeration exceeded explicit limit");
    let mut reference_stacks = concrete_reference.to_stacks(4_096).expect("stack enumeration exceeded explicit limit");
    expected_stacks.sort_by(|a, b| a.0.cmp(&b.0));
    actual_stacks.sort_by(|a, b| a.0.cmp(&b.0));
    reference_stacks.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(actual_stacks, expected_stacks, "{case}");
    assert_eq!(actual_stacks, reference_stacks, "{case}");
}

fn branched_floor_pair(common_suffix_len: usize) -> (ParserGSS, ParserGSS) {
    let suffix = [10_u32, 11, 12];
    let mut left = vec![0, 1];
    left.extend_from_slice(&suffix[..common_suffix_len]);
    let mut right = vec![0, 2];
    right.extend_from_slice(&suffix[..common_suffix_len]);
    let acc = TerminalsDisallowed::new();
    (
        ParserGSS::from_single_stack(left, acc.clone()),
        ParserGSS::from_single_stack(right, acc),
    )
}

fn assert_advance_matches_concrete_reference_case(
    table: &GLRTable,
    before: &ParserGSS,
    token: u32,
    case: &str,
) {
    let actual = advance_stacks(table, before, token);
    let expected = advance_concrete_stacks_reference(table, before, token);
    assert_eq!(
        super::normalized_concrete_stacks(&actual),
        super::normalized_concrete_stacks(&expected),
        "{case}: before={:?}",
        before.to_stacks(4_096).expect("stack enumeration exceeded explicit limit"),
    );
}

#[test]
fn generated_mixed_pop1_frontier_actions_match_concrete_reference() {
    let token = 0;
    let nt0 = 0;
    let nt1 = 1;
    let actions = vec![
        None,
        Some(Action::Shift(40, false)),
        Some(Action::Shift(41, true)),
        Some(Action::StackShifts(vec![StackShift {
            pop: 0,
            pushes: vec![42],
        }])),
        Some(Action::StackShifts(vec![StackShift {
            pop: 1,
            pushes: vec![43],
        }])),
        Some(Action::StackShifts(vec![StackShift {
            pop: 1,
            pushes: vec![44, 45],
        }])),
        Some(Action::StackShifts(vec![StackShift {
            pop: 2,
            pushes: vec![46],
        }])),
        Some(Action::StackShifts(vec![
            StackShift {
                pop: 1,
                pushes: vec![47],
            },
            StackShift {
                pop: 2,
                pushes: vec![48],
            },
        ])),
        Some(Action::Reduce(nt0, 1)),
        Some(Action::Reduce(nt1, 1)),
        Some(Action::Split {
            shift: Some((49, false)),
            reduces: vec![(nt0, 1)],
            accept: false,
        }),
        Some(Action::Split {
            shift: None,
            reduces: vec![(nt0, 1), (nt1, 1)],
            accept: false,
        }),
        Some(Action::GuardedStackShifts(vec![GuardedStackShift {
            guards: vec![StackShiftGuard {
                pop: 1,
                states: vec![10],
            }],
            pop: 2,
            pushes: vec![52],
        }])),
    ];

    for (left_index, left_action) in actions.iter().enumerate() {
        for (right_index, right_action) in actions.iter().enumerate() {
            let mut action_rows = vec![Vec::new(); 64];
            if let Some(action) = left_action {
                action_rows[20].push((token, action.clone()));
            }
            if let Some(action) = right_action {
                action_rows[21].push((token, action.clone()));
            }
            action_rows[30].push((token, Action::Shift(50, false)));
            action_rows[31].push((
                token,
                Action::StackShifts(vec![StackShift {
                    pop: 1,
                    pushes: vec![51],
                }]),
            ));
            let action_refs: Vec<&[(u32, Action)]> =
                action_rows.iter().map(Vec::as_slice).collect();

            let mut goto_rows = vec![Vec::new(); 64];
            goto_rows[10].push((nt0, (30, false)));
            goto_rows[10].push((nt1, (31, false)));
            let goto_refs: Vec<&[(u32, (u32, bool))]> =
                goto_rows.iter().map(Vec::as_slice).collect();
            let table = build_test_table(64, 1, &action_refs, &goto_refs);

            let left_acc = TerminalsDisallowed::new().with_insert(60, 0);
            let right_acc = TerminalsDisallowed::new().with_insert(61, 0);
            let before = ParserGSS::from_stacks(&[
                (vec![0, 10, 20], left_acc),
                (vec![0, 10, 21], right_acc),
            ]);
            let case = format!(
                "left_index={left_index} left={left_action:?} right_index={right_index} right={right_action:?}",
            );
            assert_advance_matches_concrete_reference_case(&table, &before, token, &case);
        }
    }
}

#[test]
fn mixed_top_replace_wave_handles_machine_schema_step333_shape() {
    let token = 0;
    let nt0 = 0;
    let nt1 = 1;
    let mut action_rows = vec![Vec::new(); 400];
    action_rows[355].push((token, Action::Shift(264, true)));
    action_rows[363].push((token, Action::Reduce(nt0, 1)));
    action_rows[373].push((token, Action::Reduce(nt1, 1)));
    action_rows[193].push((token, Action::Shift(280, true)));
    action_rows[208].push((token, Action::Shift(299, true)));
    let action_refs: Vec<&[(u32, Action)]> =
        action_rows.iter().map(Vec::as_slice).collect();

    let mut goto_rows = vec![Vec::new(); 400];
    goto_rows[10].push((nt0, (193, false)));
    goto_rows[10].push((nt1, (208, false)));
    let goto_refs: Vec<&[(u32, (u32, bool))]> =
        goto_rows.iter().map(Vec::as_slice).collect();
    let table = build_test_table(400, 1, &action_refs, &goto_refs);

    let acc = TerminalsDisallowed::new();
    let before = ParserGSS::from_stacks(&[
        (vec![0, 1, 10, 327], acc.clone()),
        (vec![0, 2, 10, 327], acc.clone()),
        (vec![0, 1, 10, 342], acc.clone()),
        (vec![0, 1, 10, 355], acc.clone()),
        (vec![0, 2, 10, 355], acc.clone()),
        (vec![0, 1, 10, 363], acc.clone()),
        (vec![0, 2, 10, 363], acc.clone()),
        (vec![0, 1, 10, 373], acc.clone()),
        (vec![0, 2, 10, 373], acc),
    ]);
    let expected = ParserGSS::from_stacks(&[
        (vec![0, 1, 10, 264], TerminalsDisallowed::new()),
        (vec![0, 2, 10, 264], TerminalsDisallowed::new()),
        (vec![0, 1, 10, 280], TerminalsDisallowed::new()),
        (vec![0, 2, 10, 280], TerminalsDisallowed::new()),
        (vec![0, 1, 10, 299], TerminalsDisallowed::new()),
        (vec![0, 2, 10, 299], TerminalsDisallowed::new()),
    ]);

    let remapped = try_advance_mixed_top_replace_wave(&table, &before, token)
        .expect("machine schema step333 shape should be a direct top remap");
    assert!(remapped.semantically_eq(&expected, 16).unwrap());
    assert!(advance_stacks(&table, &before, token)
        .semantically_eq(&expected, 16)
        .unwrap());
}

#[test]
fn mixed_top_replace_wave_handles_reduce_plus_dead_frontier() {
    let token = 0;
    let nt = 0;
    let mut action_rows = vec![Vec::new(); 200];
    action_rows[99].push((token, Action::Reduce(nt, 1)));
    action_rows[112].push((token, Action::Shift(170, true)));
    let action_refs: Vec<&[(u32, Action)]> =
        action_rows.iter().map(Vec::as_slice).collect();
    let mut goto_rows = vec![Vec::new(); 200];
    goto_rows[56].push((nt, (112, false)));
    let goto_refs: Vec<&[(u32, (u32, bool))]> =
        goto_rows.iter().map(Vec::as_slice).collect();
    let table = build_test_table(200, 1, &action_refs, &goto_refs);

    let acc = TerminalsDisallowed::new();
    let before = ParserGSS::from_stacks(&[
        (vec![0, 1, 40, 56, 99], acc.clone()),
        (vec![0, 1, 40, 56, 83], acc),
    ]);
    let expected = ParserGSS::from_single_stack(
        vec![0, 1, 40, 56, 170],
        TerminalsDisallowed::new(),
    );

    let fast = try_advance_mixed_top_replace_wave(&table, &before, token)
        .expect("reduce plus dead frontier should be a direct top remap");
    assert!(fast.semantically_eq(&expected, 8).unwrap());
    assert!(advance_stacks(&table, &before, token)
        .semantically_eq(&expected, 8)
        .unwrap());
}

#[test]
fn bounded_deterministic_reduce_paths_handle_predecessor_dependent_chains() {
    let token = 0;
    let nt_value = 0;
    let nt_regular = 1;
    let nt_deep = 2;
    let mut action_rows = vec![Vec::new(); 500];
    action_rows[90].push((token, Action::Reduce(nt_value, 1)));
    action_rows[335].push((token, Action::Reduce(nt_regular, 1)));
    action_rows[336].push((token, Action::Shift(337, true)));
    action_rows[446].push((token, Action::Reduce(nt_deep, 1)));
    action_rows[343].push((
        token,
        Action::GuardedStackShifts(vec![
            GuardedStackShift {
                guards: vec![StackShiftGuard {
                    pop: 2,
                    states: vec![59],
                }],
                pop: 3,
                pushes: vec![107],
            },
            GuardedStackShift {
                guards: vec![StackShiftGuard {
                    pop: 2,
                    states: vec![80],
                }],
                pop: 3,
                pushes: vec![119],
            },
            GuardedStackShift {
                guards: vec![StackShiftGuard {
                    pop: 2,
                    states: vec![92],
                }],
                pop: 3,
                pushes: vec![135],
            },
        ]),
    ));
    let action_refs: Vec<&[(u32, Action)]> =
        action_rows.iter().map(Vec::as_slice).collect();

    let mut goto_rows = vec![Vec::new(); 500];
    goto_rows[389].push((nt_value, (335, true)));
    goto_rows[426].push((nt_value, (446, true)));
    goto_rows[138].push((nt_regular, (336, false)));
    goto_rows[250].push((nt_deep, (343, false)));
    let goto_refs: Vec<&[(u32, (u32, bool))]> =
        goto_rows.iter().map(Vec::as_slice).collect();
    let table = build_test_table(500, 1, &action_refs, &goto_refs);

    let acc_a = TerminalsDisallowed::new().with_insert(60, 0);
    let acc_b = TerminalsDisallowed::new().with_insert(61, 0);
    let before = ParserGSS::from_stacks(&[
        (vec![0, 1, 168, 138, 389, 90], acc_a.clone()),
        (vec![0, 1, 181, 138, 389, 90], acc_a.clone()),
        (vec![0, 1, 198, 138, 389, 90], acc_a.clone()),
        (vec![0, 1, 322, 91, 59, 250, 426, 90], acc_b.clone()),
        (vec![0, 1, 322, 91, 80, 250, 426, 90], acc_b.clone()),
        (vec![0, 1, 322, 91, 92, 250, 426, 90], acc_b.clone()),
    ]);
    let expected = ParserGSS::from_stacks(&[
        (vec![0, 1, 168, 138, 337], acc_a.clone()),
        (vec![0, 1, 181, 138, 337], acc_a.clone()),
        (vec![0, 1, 198, 138, 337], acc_a),
        (vec![0, 1, 322, 91, 107], acc_b.clone()),
        (vec![0, 1, 322, 91, 119], acc_b.clone()),
        (vec![0, 1, 322, 91, 135], acc_b),
    ]);

    let fast = try_advance_bounded_concrete_paths(&table, &before, token)
        .expect("small predecessor-dependent reduce chains should remain bounded");
    assert!(fast.semantically_eq(&expected, 16).unwrap());
    assert!(advance_stacks(&table, &before, token)
        .semantically_eq(&expected, 16)
        .unwrap());
}

#[test]
fn bounded_concrete_paths_handle_guarded_shift_and_pure_shift() {
    let token = 0;
    let mut action_rows = vec![Vec::new(); 64];
    action_rows[10].push((
        token,
        Action::GuardedStackShifts(vec![GuardedStackShift {
            guards: vec![StackShiftGuard {
                pop: 1,
                states: vec![1],
            }],
            pop: 2,
            pushes: vec![20],
        }]),
    ));
    action_rows[11].push((token, Action::Shift(30, true)));
    let action_refs = action_rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let goto_rows = vec![Vec::new(); 64];
    let goto_refs = goto_rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let table = build_test_table(64, 1, &action_refs, &goto_refs);

    let before = ParserGSS::from_stacks(&[
        (vec![0, 1, 10], TerminalsDisallowed::new()),
        (vec![0, 2, 11], TerminalsDisallowed::new()),
    ]);
    let expected = ParserGSS::from_stacks(&[
        (vec![0, 20], TerminalsDisallowed::new()),
        (vec![0, 2, 30], TerminalsDisallowed::new()),
    ]);

    let fast = try_advance_bounded_concrete_paths(&table, &before, token)
        .expect("small guarded/shift frontier should stay bounded");
    assert!(fast.semantically_eq(&expected, 8).unwrap());
    assert!(advance_stacks(&table, &before, token)
        .semantically_eq(&expected, 8)
        .unwrap());
}

#[test]
fn top_local_stack_effects_preserve_upper_branch_accumulators() {
    let token = 0;
    let mut action_rows = vec![Vec::new(); 64];
    action_rows[10].push((
        token,
        Action::StackShifts(vec![StackShift {
            pop: 1,
            pushes: vec![20, 30],
        }]),
    ));
    action_rows[11].push((token, Action::Shift(40, false)));
    let action_refs: Vec<&[(u32, Action)]> =
        action_rows.iter().map(Vec::as_slice).collect();
    let goto_rows = vec![Vec::new(); 64];
    let goto_refs: Vec<&[(u32, (u32, bool))]> =
        goto_rows.iter().map(Vec::as_slice).collect();
    let table = build_test_table(64, 1, &action_refs, &goto_refs);

    let acc_a = TerminalsDisallowed::new().with_insert(60, 0);
    let acc_b = TerminalsDisallowed::new().with_insert(61, 0);
    let before = ParserGSS::from_stacks(&[
        (vec![0, 1, 10], acc_a.clone()),
        (vec![0, 2, 11], acc_b.clone()),
    ]);
    let expected = ParserGSS::from_stacks(&[
        (vec![0, 1, 20, 30], acc_a),
        (vec![0, 2, 11, 40], acc_b),
    ]);

    let fast = try_advance_pop1_stackshift_shift_wave(&table, &before, token)
        .expect("top-local effects should traverse upper accumulator branches");
    assert!(fast.semantically_eq(&expected, 8).unwrap());
    assert!(advance_stacks(&table, &before, token)
        .semantically_eq(&expected, 8)
        .unwrap());
}

#[test]
fn top_local_stack_effect_wave_handles_machine_schema_step333_followup() {
    let token = 0;
    let mut action_rows = vec![Vec::new(); 500];
    action_rows[264].push((
        token,
        Action::StackShifts(vec![StackShift {
            pop: 1,
            pushes: vec![171, 88],
        }]),
    ));
    action_rows[280].push((
        token,
        Action::StackShifts(vec![StackShift {
            pop: 1,
            pushes: vec![185, 288],
        }]),
    ));
    action_rows[299].push((
        token,
        Action::StackShifts(vec![StackShift {
            pop: 1,
            pushes: vec![201, 288],
        }]),
    ));
    action_rows[426].push((token, Action::Shift(92, false)));
    action_rows[322].push((token, Action::Shift(92, false)));
    let action_refs: Vec<&[(u32, Action)]> =
        action_rows.iter().map(Vec::as_slice).collect();
    let goto_rows = vec![Vec::new(); 500];
    let goto_refs: Vec<&[(u32, (u32, bool))]> =
        goto_rows.iter().map(Vec::as_slice).collect();
    let table = build_test_table(500, 1, &action_refs, &goto_refs);

    let acc = TerminalsDisallowed::new();
    let before = ParserGSS::from_stacks(&[
        (vec![0, 1, 10, 186, 40, 85, 123, 264], acc.clone()),
        (vec![0, 1, 10, 202, 40, 85, 123, 264], acc.clone()),
        (vec![0, 1, 10, 186, 40, 85, 123, 280], acc.clone()),
        (vec![0, 1, 10, 202, 40, 85, 123, 280], acc.clone()),
        (vec![0, 1, 10, 186, 40, 85, 123, 299], acc.clone()),
        (vec![0, 1, 10, 202, 40, 85, 123, 299], acc.clone()),
        (vec![0, 1, 10, 322, 92, 150, 250, 343, 426], acc.clone()),
        (vec![0, 1, 10, 186, 40, 85, 123, 322], acc.clone()),
        (vec![0, 1, 10, 202, 40, 85, 123, 322], acc),
    ]);
    let expected = ParserGSS::from_stacks(&[
        (vec![0, 1, 10, 322, 92, 150, 250, 343, 426, 92], TerminalsDisallowed::new()),
        (vec![0, 1, 10, 186, 40, 85, 123, 322, 92], TerminalsDisallowed::new()),
        (vec![0, 1, 10, 202, 40, 85, 123, 322, 92], TerminalsDisallowed::new()),
        (vec![0, 1, 10, 186, 40, 85, 123, 171, 88], TerminalsDisallowed::new()),
        (vec![0, 1, 10, 202, 40, 85, 123, 171, 88], TerminalsDisallowed::new()),
        (vec![0, 1, 10, 186, 40, 85, 123, 185, 288], TerminalsDisallowed::new()),
        (vec![0, 1, 10, 202, 40, 85, 123, 185, 288], TerminalsDisallowed::new()),
        (vec![0, 1, 10, 186, 40, 85, 123, 201, 288], TerminalsDisallowed::new()),
        (vec![0, 1, 10, 202, 40, 85, 123, 201, 288], TerminalsDisallowed::new()),
    ]);

    let fast = try_advance_pop1_stackshift_shift_wave(&table, &before, token)
        .expect("machine schema follow-up should be top-local");
    assert!(fast.semantically_eq(&expected, 32).unwrap());
    assert!(advance_stacks(&table, &before, token)
        .semantically_eq(&expected, 32)
        .unwrap());
}

#[test]
fn generated_top_action_advance_distributes_over_branched_floor_merge() {
    let token = 0;
    for common_suffix_len in 1..=3 {
        let (left, right) = branched_floor_pair(common_suffix_len);
        let top = 9 + common_suffix_len as u32;
        let mut actions = vec![
            Action::Shift(40, false),
            Action::Shift(40, true),
        ];

        for pop in 0..=common_suffix_len + 1 {
            actions.push(Action::StackShifts(vec![StackShift {
                pop: pop as u32,
                pushes: vec![40],
            }]));
            actions.push(Action::StackShifts(vec![StackShift {
                pop: pop as u32,
                pushes: Vec::new(),
            }]));
        }
        actions.push(Action::StackShifts(vec![
            StackShift {
                pop: 0,
                pushes: vec![40],
            },
            StackShift {
                pop: common_suffix_len as u32 + 1,
                pushes: vec![41],
            },
        ]));
        actions.push(Action::StackShifts(vec![
            StackShift {
                pop: common_suffix_len as u32 + 1,
                pushes: vec![40],
            },
            StackShift {
                pop: 0,
                pushes: vec![41],
            },
        ]));
        actions.push(Action::StackShifts(vec![
            StackShift {
                pop: common_suffix_len as u32,
                pushes: vec![40, 42],
            },
            StackShift {
                pop: common_suffix_len as u32,
                pushes: vec![41, 42],
            },
        ]));

        actions.push(Action::GuardedStackShifts(vec![
            GuardedStackShift {
                guards: vec![StackShiftGuard {
                    pop: common_suffix_len as u32,
                    states: vec![1],
                }],
                pop: common_suffix_len as u32 + 1,
                pushes: vec![40],
            },
            GuardedStackShift {
                guards: vec![StackShiftGuard {
                    pop: common_suffix_len as u32,
                    states: vec![2],
                }],
                pop: common_suffix_len as u32 + 1,
                pushes: vec![41],
            },
        ]));

        for (action_index, action) in actions.into_iter().enumerate() {
            let mut action_rows = vec![Vec::new(); 64];
            action_rows[top as usize] = vec![(token, action)];
            let action_refs: Vec<&[(u32, Action)]> =
                action_rows.iter().map(Vec::as_slice).collect();
            let goto_rows = vec![Vec::new(); 64];
            let goto_refs: Vec<&[(u32, (u32, bool))]> =
                goto_rows.iter().map(Vec::as_slice).collect();
            let table = build_test_table(64, 1, &action_refs, &goto_refs);
            let case = format!(
                "common_suffix_len={common_suffix_len} action_index={action_index} action={:?}",
                table.action(top, token),
            );
            assert_advance_distributes_over_merge(&table, &left, &right, token, &case);
        }
    }
}

#[test]
fn generated_reduce_chain_advance_distributes_over_branched_floor_merge() {
    let token = 0;
    let nt = 0;
    for common_suffix_len in 1..=3 {
        let (left, right) = branched_floor_pair(common_suffix_len);
        let top = 9 + common_suffix_len as u32;
        for reduce_len in 1..=common_suffix_len {
            for is_replace in [false, true] {
                let mut action_rows = vec![Vec::new(); 64];
                action_rows[top as usize] =
                    vec![(token, Action::Reduce(nt, reduce_len as u32))];
                action_rows[50] = vec![(token, Action::Shift(60, false))];
                action_rows[51] = vec![(token, Action::Shift(60, false))];

                let left_stacks = left.to_stacks(4_096).expect("stack enumeration exceeded explicit limit");
                let right_stacks = right.to_stacks(4_096).expect("stack enumeration exceeded explicit limit");
                let left_values = &left_stacks[0].0;
                let right_values = &right_stacks[0].0;
                let left_goto_from = left_values[left_values.len() - reduce_len - 1];
                let right_goto_from = right_values[right_values.len() - reduce_len - 1];

                let mut goto_rows = vec![Vec::new(); 64];
                goto_rows[left_goto_from as usize].push((nt, (50, is_replace)));
                if right_goto_from != left_goto_from {
                    goto_rows[right_goto_from as usize].push((nt, (51, is_replace)));
                }

                let action_refs: Vec<&[(u32, Action)]> =
                    action_rows.iter().map(Vec::as_slice).collect();
                let goto_refs: Vec<&[(u32, (u32, bool))]> =
                    goto_rows.iter().map(Vec::as_slice).collect();
                let table = build_test_table(64, 1, &action_refs, &goto_refs);
                let case = format!(
                    "reduce common_suffix_len={common_suffix_len} reduce_len={reduce_len} is_replace={is_replace} left_goto={left_goto_from} right_goto={right_goto_from}",
                );
                assert_advance_distributes_over_merge(&table, &left, &right, token, &case);
            }
        }
    }
}

#[test]
fn exact_admission_rejects_union_reduce_with_no_real_goto() {
    let token = 0;
    let nt = 0;
    let mut table = build_test_table(
        3,
        1,
        &[&[], &[], &[(token, Action::Reduce(nt, 1))]],
        &[&[], &[], &[]],
    );
    table.admission_policy = AdmissionPolicy::ExactSimulation;

    let stack = ParserGSS::from_single_stack(vec![0, 2], TerminalsDisallowed::new());

    assert!(!stack_may_advance_on(&table, &stack, token));
}

#[test]
fn exact_admission_accepts_reduce_goto_then_shift_path() {
    let token = 0;
    let nt = 0;
    let mut table = build_test_table(
        5,
        1,
        &[
            &[],
            &[],
            &[(token, Action::Reduce(nt, 1))],
            &[(token, Action::Shift(4, false))],
            &[],
        ],
        &[&[(nt, (3, false))], &[], &[], &[], &[]],
    );
    table.admission_policy = AdmissionPolicy::ExactSimulation;

    let stack = ParserGSS::from_single_stack(vec![0, 2], TerminalsDisallowed::new());

    assert!(stack_may_advance_on(&table, &stack, token));
}

#[test]
fn exact_admission_any_uses_same_exactness_as_single_terminal() {
    let token = 0;
    let nt = 0;
    let mut table = build_test_table(
        5,
        2,
        &[
            &[],
            &[],
            &[(token, Action::Reduce(nt, 1))],
            &[(token, Action::Shift(4, false))],
            &[],
        ],
        &[&[(nt, (3, false))], &[], &[], &[], &[]],
    );
    table.admission_policy = AdmissionPolicy::ExactSimulation;
    let stack = ParserGSS::from_single_stack(vec![0, 2], TerminalsDisallowed::new());

    let mut terminals = BitSet::new(3);
    terminals.set(token as usize);
    assert_eq!(
        stack_may_advance_on(&table, &stack, token),
        stack_may_advance_on_any(&table, &stack, &terminals)
    );
}

fn assert_exact_any_matches_disjunction(
    table: &GLRTable,
    stack: &ParserGSS,
    terminals: &BitSet,
) {
    let disjunction = terminals.iter_ones().any(|bit| match bit {
        bit if bit == table.num_terminals as usize => stack_may_advance_on(table, stack, EOF),
        bit if bit < table.num_terminals as usize => {
            stack_may_advance_on(table, stack, bit as u32)
        }
        _ => false,
    });

    assert_eq!(
        stack_may_advance_on_any(table, stack, terminals),
        disjunction
    );
}

fn assert_admissible_set_matches_individual(
    table: &GLRTable,
    stack: &ParserGSS,
    terminals: &BitSet,
) {
    let admitted = stack_admissible_terminals(table, stack, terminals);
    let mut expected = BitSet::new(terminals.len());
    for bit in terminals.iter_ones() {
        let terminal = if bit == table.num_terminals as usize {
            EOF
        } else if bit < table.num_terminals as usize {
            bit as u32
        } else {
            continue;
        };
        if stack_may_advance_on(table, stack, terminal) {
            expected.set(bit);
        }
    }
    assert_eq!(admitted, expected);
}

#[test]
fn exact_admission_any_does_not_mix_lookahead_reductions() {
    let reduce_token = 0;
    let shift_token = 1;
    let nt = 0;
    let mut table = build_test_table(
        5,
        2,
        &[
            &[],
            &[],
            &[(reduce_token, Action::Reduce(nt, 1))],
            &[(shift_token, Action::Shift(4, false))],
            &[],
        ],
        &[&[(nt, (3, false))], &[], &[], &[], &[]],
    );
    table.admission_policy = AdmissionPolicy::ExactSimulation;

    let stack = ParserGSS::from_single_stack(vec![0, 2], TerminalsDisallowed::new());
    let mut terminals = BitSet::new(3);
    terminals.set(reduce_token as usize);
    terminals.set(shift_token as usize);

    assert!(!stack_may_advance_on(&table, &stack, reduce_token));
    assert!(!stack_may_advance_on(&table, &stack, shift_token));
    assert_exact_any_matches_disjunction(&table, &stack, &terminals);
    assert_admissible_set_matches_individual(&table, &stack, &terminals);
    assert!(!stack_may_advance_on_any(&table, &stack, &terminals));
}

#[test]
fn exact_admission_any_batches_reduce_frontier_by_terminal_set() {
    let token_a = 0;
    let token_b = 1;
    let nt = 0;
    let mut table = build_test_table(
        5,
        2,
        &[
            &[],
            &[],
            &[
                (token_a, Action::Reduce(nt, 1)),
                (token_b, Action::Reduce(nt, 1)),
            ],
            &[(token_b, Action::Shift(4, false))],
            &[],
        ],
        &[&[(nt, (3, false))], &[], &[], &[], &[]],
    );
    table.admission_policy = AdmissionPolicy::ExactSimulation;

    let stack = ParserGSS::from_single_stack(vec![0, 2], TerminalsDisallowed::new());
    let mut terminals = BitSet::new(3);
    terminals.set(token_a as usize);
    terminals.set(token_b as usize);

    assert!(!stack_may_advance_on(&table, &stack, token_a));
    assert!(stack_may_advance_on(&table, &stack, token_b));
    assert_exact_any_matches_disjunction(&table, &stack, &terminals);
    assert_admissible_set_matches_individual(&table, &stack, &terminals);
    assert!(stack_may_advance_on_any(&table, &stack, &terminals));
}

#[test]
fn exact_admission_any_preserves_guarded_shift_checks() {
    let token = 0;
    let other_token = 1;
    let mut table = build_test_table(
        3,
        2,
        &[
            &[],
            &[],
            &[(
                token,
                Action::GuardedStackShifts(vec![GuardedStackShift {
                    guards: vec![StackShiftGuard {
                        pop: 1,
                        states: vec![0],
                    }],
                    pop: 2,
                    pushes: vec![7],
                }]),
            )],
        ],
        &[&[], &[], &[]],
    );
    table.admission_policy = AdmissionPolicy::ExactSimulation;

    let mut terminals = BitSet::new(3);
    terminals.set(token as usize);
    terminals.set(other_token as usize);

    let rejected = ParserGSS::from_single_stack(vec![1, 2], TerminalsDisallowed::new());
    assert_exact_any_matches_disjunction(&table, &rejected, &terminals);
    assert_admissible_set_matches_individual(&table, &rejected, &terminals);
    assert!(!stack_may_advance_on_any(&table, &rejected, &terminals));

    let accepted = ParserGSS::from_single_stack(vec![0, 2], TerminalsDisallowed::new());
    assert_exact_any_matches_disjunction(&table, &accepted, &terminals);
    assert_admissible_set_matches_individual(&table, &accepted, &terminals);
    assert!(stack_may_advance_on_any(&table, &accepted, &terminals));
}

#[test]
fn exact_admission_any_handles_eof_acceptance() {
    let token = 0;
    let mut table = build_test_table(1, 1, &[&[(EOF, Action::Accept)]], &[&[]]);
    table.admission_policy = AdmissionPolicy::ExactSimulation;
    let stack = ParserGSS::from_single_stack(vec![0], TerminalsDisallowed::new());

    let mut non_eof = BitSet::new(2);
    non_eof.set(token as usize);
    assert_exact_any_matches_disjunction(&table, &stack, &non_eof);
    assert_admissible_set_matches_individual(&table, &stack, &non_eof);
    assert!(!stack_may_advance_on_any(&table, &stack, &non_eof));

    let mut eof = BitSet::new(2);
    eof.set(table.num_terminals as usize);
    assert_exact_any_matches_disjunction(&table, &stack, &eof);
    assert_admissible_set_matches_individual(&table, &stack, &eof);
    assert!(stack_may_advance_on_any(&table, &stack, &eof));
}

/// A nullable-skip / trivia reduction leaves an EOF `Reduce` on the top
/// state. That is NOT completion when the reduction has no feasible goto
/// continuation to an accepting state.
#[test]
fn eof_reduce_without_root_accept_is_incomplete() {
    let table = build_test_table(
        3,
        1,
        &[&[], &[(EOF, Action::Reduce(0, 1))], &[(EOF, Action::Accept)]],
        &[&[], &[], &[]],
    );
    let stack = ParserGSS::from_single_stack(vec![0, 1], TerminalsDisallowed::new());
    assert!(
        !stacks_finished(&table, &stack),
        "an EOF reduce with no goto continuation must not count as completion",
    );
}

/// The same EOF `Reduce` DOES complete when its reduction chain has a
/// feasible goto to an accepting state.
#[test]
fn eof_reduction_chain_reaches_root_accept() {
    let table = build_test_table(
        3,
        1,
        &[&[], &[(EOF, Action::Reduce(0, 1))], &[(EOF, Action::Accept)]],
        &[&[(0, (2, false))], &[], &[]],
    );
    let stack = ParserGSS::from_single_stack(vec![0, 1], TerminalsDisallowed::new());
    assert!(
        stacks_finished(&table, &stack),
        "an EOF reduce chain that reaches an accepting state must complete",
    );
}

/// A nullable/empty root accepts EOF directly from the start state.
#[test]
fn nullable_empty_root_eof_accepts() {
    let table = build_test_table(1, 1, &[&[(EOF, Action::Accept)]], &[&[]]);
    let stack = ParserGSS::from_single_stack(vec![0], TerminalsDisallowed::new());
    assert!(stacks_finished(&table, &stack));
}

/// A valid reduce branch to Accept completes even when an ambiguous sibling
/// reduce branch is dead.
#[test]
fn ambiguous_eof_split_with_dead_branch_still_completes() {
    let table = build_test_table(
        4,
        1,
        &[
            &[],
            &[(EOF, Action::Split {
                shift: None,
                reduces: vec![(0, 1), (1, 1)],
                accept: false,
            })],
            &[(EOF, Action::Accept)],
            &[],
        ],
        &[&[(0, (2, false))], &[], &[], &[]],
    );
    let stack = ParserGSS::from_single_stack(vec![0, 1], TerminalsDisallowed::new());
    assert!(
        stacks_finished(&table, &stack),
        "an ambiguous EOF split completes when one reduce branch reaches Accept",
    );
}

/// An EOF shift is ordinary progress, not completion: `may_advance` must
/// stay true while the completion predicate rejects it.
#[test]
fn eof_shift_may_advance_but_is_not_completion() {
    let mut table =
        build_test_table(2, 1, &[&[(EOF, Action::Shift(1, false))], &[]], &[&[], &[]]);
    table.admission_policy = AdmissionPolicy::ExactSimulation;
    let stack = ParserGSS::from_single_stack(vec![0], TerminalsDisallowed::new());
    assert!(
        stack_may_advance_on(&table, &stack, EOF),
        "an EOF shift is an ordinary advance",
    );
    assert!(
        !stacks_finished(&table, &stack),
        "an EOF shift alone must not count as completion",
    );
}

/// An EOF skip (scoped-ignore shape) is ordinary progress, not completion.
#[test]
fn eof_skip_may_advance_but_is_not_completion() {
    let mut table = build_test_table(1, 1, &[&[(EOF, Action::Skip)]], &[&[]]);
    table.admission_policy = AdmissionPolicy::ExactSimulation;
    let stack = ParserGSS::from_single_stack(vec![0], TerminalsDisallowed::new());
    assert!(
        stack_may_advance_on(&table, &stack, EOF),
        "an EOF skip is an ordinary advance",
    );
    assert!(
        !stacks_finished(&table, &stack),
        "an EOF skip alone must not count as completion",
    );
}

/// A split with only a shift branch (no accepting branch) is not
/// completion, even though the shift is an ordinary advance.
#[test]
fn eof_split_shift_only_is_not_completion() {
    let mut table = build_test_table(
        2,
        1,
        &[
            &[],
            &[(EOF, Action::Split {
                shift: Some((1, false)),
                reduces: Vec::new(),
                accept: false,
            })],
        ],
        &[&[], &[]],
    );
    table.admission_policy = AdmissionPolicy::ExactSimulation;
    let stack = ParserGSS::from_single_stack(vec![0, 1], TerminalsDisallowed::new());
    assert!(
        stack_may_advance_on(&table, &stack, EOF),
        "an EOF split shift is an ordinary advance",
    );
    assert!(
        !stacks_finished(&table, &stack),
        "an EOF split with only a shift branch must not count as completion",
    );
}

#[test]
fn provider_eof_suite_mirrors_concrete_semantics() {
    // 1. Orphan reduce false: reduce with no goto continuation does not complete
    let table1 = build_test_table(
        3,
        1,
        &[&[], &[(EOF, Action::Reduce(0, 1))], &[(EOF, Action::Accept)]],
        &[&[], &[], &[]],
    );
    let provider1 = GLRTableActionProvider::new(&table1);
    let stack1 = ParserGSS::from_single_stack(vec![0, 1], TerminalsDisallowed::new());
    assert!(!stacks_finished_with_provider(&provider1, &stack1, EOF));

    // 2. Reduce chain Accept true: reduce reaches Accept via goto
    let table2 = build_test_table(
        3,
        1,
        &[&[], &[(EOF, Action::Reduce(0, 1))], &[(EOF, Action::Accept)]],
        &[&[(0, (2, false))], &[], &[]],
    );
    let provider2 = GLRTableActionProvider::new(&table2);
    let stack2 = ParserGSS::from_single_stack(vec![0, 1], TerminalsDisallowed::new());
    assert!(stacks_finished_with_provider(&provider2, &stack2, EOF));

    // 3. Nullable empty root true
    let table3 = build_test_table(1, 1, &[&[(EOF, Action::Accept)]], &[&[]]);
    let provider3 = GLRTableActionProvider::new(&table3);
    let stack3 = ParserGSS::from_single_stack(vec![0], TerminalsDisallowed::new());
    assert!(stacks_finished_with_provider(&provider3, &stack3, EOF));

    // 4. Ambiguous dead branch + Accept true
    let table4 = build_test_table(
        4,
        1,
        &[
            &[],
            &[(EOF, Action::Split {
                shift: None,
                reduces: vec![(0, 1), (1, 1)],
                accept: false,
            })],
            &[(EOF, Action::Accept)],
            &[],
        ],
        &[&[(0, (2, false))], &[], &[], &[]],
    );
    let provider4 = GLRTableActionProvider::new(&table4);
    let stack4 = ParserGSS::from_single_stack(vec![0, 1], TerminalsDisallowed::new());
    assert!(stacks_finished_with_provider(&provider4, &stack4, EOF));

    // 5. EOF shift/skip/stackshift false
    let table_shift = build_test_table(2, 1, &[&[(EOF, Action::Shift(1, false))], &[]], &[&[], &[]]);
    let provider_shift = GLRTableActionProvider::new(&table_shift);
    let stack_shift = ParserGSS::from_single_stack(vec![0], TerminalsDisallowed::new());
    assert!(!stacks_finished_with_provider(&provider_shift, &stack_shift, EOF));

    let table_skip = build_test_table(1, 1, &[&[(EOF, Action::Skip)]], &[&[]]);
    let provider_skip = GLRTableActionProvider::new(&table_skip);
    let stack_skip = ParserGSS::from_single_stack(vec![0], TerminalsDisallowed::new());
    assert!(!stacks_finished_with_provider(&provider_skip, &stack_skip, EOF));

    let table_stackshift = build_test_table(
        2,
        1,
        &[&[(EOF, Action::StackShifts(vec![StackShift { pop: 0, pushes: vec![1] }]))], &[]],
        &[&[], &[]],
    );
    let provider_stackshift = GLRTableActionProvider::new(&table_stackshift);
    let stack_stackshift = ParserGSS::from_single_stack(vec![0], TerminalsDisallowed::new());
    assert!(!stacks_finished_with_provider(&provider_stackshift, &stack_stackshift, EOF));

    // 6. No action / empty stack false
    let table_no_action = build_test_table(1, 1, &[&[]], &[&[]]);
    let provider_no_action = GLRTableActionProvider::new(&table_no_action);
    let stack_no_action = ParserGSS::from_single_stack(vec![0], TerminalsDisallowed::new());
    assert!(!stacks_finished_with_provider(&provider_no_action, &stack_no_action, EOF));
    assert!(!stacks_finished_with_provider(&provider_no_action, &ParserGSS::empty(), EOF));

    // 7. Infeasible pop false (reduce length > stack depth)
    let table_infeasible = build_test_table(
        2,
        1,
        &[&[(EOF, Action::Reduce(0, 5))], &[(EOF, Action::Accept)]],
        &[&[(0, (1, false))], &[]],
    );
    let provider_infeasible = GLRTableActionProvider::new(&table_infeasible);
    let stack_infeasible = ParserGSS::from_single_stack(vec![0], TerminalsDisallowed::new());
    assert!(!stacks_finished_with_provider(&provider_infeasible, &stack_infeasible, EOF));

    // 8. Reduction cycle terminates false
    let table_cycle = build_test_table(
        2,
        1,
        &[&[(EOF, Action::Reduce(0, 1))], &[]],
        &[&[(0, (0, false))], &[]],
    );
    let provider_cycle = GLRTableActionProvider::new(&table_cycle);
    let stack_cycle = ParserGSS::from_single_stack(vec![0], TerminalsDisallowed::new());
    assert!(!stacks_finished_with_provider(&provider_cycle, &stack_cycle, EOF));
}

#[test]
fn provider_split_with_accept_shift_and_reduce_preserves_shifts_and_accepts() {
    let token = 0;
    let nt = 0;
    // States:
    // State 0 (base): goto on nt -> State 2
    // State 1 (top): Split with accept + shift(3) + reduce(nt, 1)
    // State 2: reduction goto target, shifts 4 on token
    // State 3: direct shift target
    // State 4: reduction-derived shift target
    let table = build_test_table(
        5,
        1,
        &[
            &[],
            &[(
                token,
                Action::Split {
                    shift: Some((3, false)),
                    reduces: vec![(nt, 1)],
                    accept: true,
                },
            )],
            &[(token, Action::Shift(4, false))],
            &[],
            &[],
        ],
        &[
            &[(nt, (2, false))],
            &[],
            &[],
            &[],
            &[],
        ],
    );
    let provider = GLRTableActionProvider::new(&table);
    let stack = ParserGSS::from_single_stack(vec![0, 1], TerminalsDisallowed::new());

    // 1. Advance mode: verifies both direct shift (state 3) and reduction-derived shift (state 4) are preserved
    let advanced = advance_stacks_with_provider(&provider, stack.clone(), token);
    let paths = advanced.to_stacks(32).expect("should enumerate paths");
    let top_states: FxHashSet<u32> = paths.iter().map(|(s, _)| *s.last().unwrap()).collect();
    assert!(
        top_states.contains(&3),
        "Advance must preserve direct shift from Action::Split; top_states={:?}",
        top_states
    );
    assert!(
        top_states.contains(&4),
        "Advance must preserve reduction-derived shift from Action::Split; top_states={:?}",
        top_states
    );
    assert_eq!(top_states.len(), 2);

    // 2. Completion mode: verifies accept: true causes stacks_finished_with_provider to accept
    assert!(
        stacks_finished_with_provider(&provider, &stack, token),
        "Completion must accept when Action::Split has accept: true"
    );
}

#[test]
fn provider_eof_non_acceptance_variants() {
    // EOF extra_stack_shifts alone does not count as acceptance
    struct ExtraStackShiftsProvider;
    impl ParserActionProvider for ExtraStackShiftsProvider {
        type Symbol = TerminalID;
        fn action(&self, _state: u32, _symbol: TerminalID) -> Option<ProvidedAction<'_>> {
            Some(ProvidedAction {
                action: ProvidedActionRef::Identity,
                reduction_scope: 0,
                extra_stack_shifts: smallvec::smallvec![StackShift {
                    pop: 0,
                    pushes: vec![1],
                }],
            })
        }
        fn scope_state(&self, _scope: u32, local: u32) -> Option<u32> {
            Some(local)
        }
        fn goto_target(&self, _scope: u32, _from: u32, _nt: u32) -> Option<(u32, bool)> {
            None
        }
        fn state_count_hint(&self) -> usize {
            2
        }
    }

    let stack = ParserGSS::from_single_stack(vec![0], TerminalsDisallowed::new());
    assert!(
        !stacks_finished_with_provider(&ExtraStackShiftsProvider, &stack, EOF),
        "extra_stack_shifts must not count as acceptance in Completion mode"
    );

    // EOF Identity alone does not count as acceptance
    struct IdentityProvider;
    impl ParserActionProvider for IdentityProvider {
        type Symbol = TerminalID;
        fn action(&self, _state: u32, _symbol: TerminalID) -> Option<ProvidedAction<'_>> {
            Some(ProvidedAction {
                action: ProvidedActionRef::Identity,
                reduction_scope: 0,
                extra_stack_shifts: SmallVec::new(),
            })
        }
        fn scope_state(&self, _scope: u32, local: u32) -> Option<u32> {
            Some(local)
        }
        fn goto_target(&self, _scope: u32, _from: u32, _nt: u32) -> Option<(u32, bool)> {
            None
        }
        fn state_count_hint(&self) -> usize {
            1
        }
    }

    assert!(
        !stacks_finished_with_provider(&IdentityProvider, &stack, EOF),
        "Identity must not count as acceptance in Completion mode"
    );
}

#[test]
fn provider_nullable_growing_stack_cycle_bounded_witness() {
    let nt = 0;
    let table = build_test_table(
        2,
        1,
        &[
            &[(EOF, Action::Reduce(nt, 0))],
            &[(EOF, Action::Reduce(nt, 0))],
        ],
        &[
            &[(nt, (1, false))],
            &[(nt, (1, false))],
        ],
    );
    let provider = GLRTableActionProvider::new(&table);
    let mut key_interner = GssSemanticKeyInterner::<u32, TerminalsDisallowed>::new();
    let mut visited = FxHashSet::<u32>::default();

    let mut current = ParserGSS::from_single_stack(vec![0], TerminalsDisallowed::new());
    let mut keys = Vec::new();

    for _ in 0..10 {
        let mut next = ParserGSS::empty();
        for state in current.peek_values() {
            let provided = provider.action(state, EOF).unwrap();
            let isolated = current.isolate(Some(state));
            if let ProvidedActionRef::Local { action, .. } = provided.action {
                action.for_each_reduce(|nt, len| {
                    for (goto_from, base) in
                        reduce_sources_from_isolated(&isolated, len as usize)
                    {
                        if let Some((target, is_replace)) =
                            provider.goto_target(provided.reduction_scope, goto_from, nt)
                        {
                            let branch = if is_replace {
                                base.popn(1).push(target)
                            } else {
                                base.push(target)
                            };
                            let key = key_interner.key(&branch);
                            assert!(
                                visited.insert(key),
                                "key must be fresh because stack depth increased"
                            );
                            keys.push(key);
                            merge_into(&mut next, branch);
                        }
                    }
                });
            }
        }
        assert!(!next.is_empty());
        current = next;
    }
    // NOTE: This test is explicitly diagnostic. Passing this probe confirms that whole-stack
    // deduplication alone does not terminate nullable growing-stack cycles; it does NOT imply
    // fixed correctness of unbounded cycle termination.
    assert_eq!(keys.len(), 10);
    let unique_keys: FxHashSet<_> = keys.iter().copied().collect();
    assert_eq!(
        unique_keys.len(),
        10,
        "whole-stack dedup generated unique keys for all growing stacks; dedup does not terminate growing-stack cycles"
    );
}

#[test]
fn provider_eof_reduction_exposing_control_transition_accepts() {
    // Test synthetic chain: reduce -> control -> Accept on EOF.
    // Stack starts at [0, 10].
    // State 10 on EOF: Reduce(nt=0, len=1).
    // Goto from 0 on nt=0: State 1.
    // State 1 has NO EOF action, but exposes control symbol CTRL (999).
    // On CTRL (999): State 1 shifts to State 2.
    // State 2 on EOF: Action::Accept.
    const CTRL_SYM: u32 = 999;
    static ACTION_REDUCE: Action = Action::Reduce(0, 1);
    static ACTION_SHIFT_CTRL: Action = Action::Shift(2, false);
    static ACTION_ACCEPT: Action = Action::Accept;

    struct ReduceControlAcceptProvider;
    impl ParserActionProvider for ReduceControlAcceptProvider {
        type Symbol = u32;
        fn action(&self, state: u32, symbol: u32) -> Option<ProvidedAction<'_>> {
            match (state, symbol) {
                (10, EOF) => Some(ProvidedAction {
                    action: ProvidedActionRef::Local { scope: 0, action: &ACTION_REDUCE },
                    reduction_scope: 0,
                    extra_stack_shifts: SmallVec::new(),
                }),
                (1, CTRL_SYM) => Some(ProvidedAction {
                    action: ProvidedActionRef::Local { scope: 0, action: &ACTION_SHIFT_CTRL },
                    reduction_scope: 0,
                    extra_stack_shifts: SmallVec::new(),
                }),
                (2, EOF) => Some(ProvidedAction {
                    action: ProvidedActionRef::Local { scope: 0, action: &ACTION_ACCEPT },
                    reduction_scope: 0,
                    extra_stack_shifts: SmallVec::new(),
                }),
                _ => None,
            }
        }
        fn control_symbols(&self, state: u32, out: &mut SmallVec<[Self::Symbol; 4]>) {
            if state == 1 {
                out.push(CTRL_SYM);
            }
        }
        fn scope_state(&self, _scope: u32, local: u32) -> Option<u32> {
            Some(local)
        }
        fn goto_target(&self, _scope: u32, goto_from: u32, nt: u32) -> Option<(u32, bool)> {
            if goto_from == 0 && nt == 0 {
                Some((1, false))
            } else {
                None
            }
        }
        fn state_count_hint(&self) -> usize {
            16
        }
    }

    let provider = ReduceControlAcceptProvider;
    let stack = ParserGSS::from_single_stack(vec![0, 10], TerminalsDisallowed::new());

    // Completion-mode traversal closes controls on the next reduction frontier,
    // so post-reduction control transitions are explored and reach Accept.
    let finished = stacks_finished_with_provider(&provider, &stack, EOF);
    assert!(
        finished,
        "Completion-mode traversal closes controls on the next reduction frontier, exposing post-reduction control transitions to Accept"
    );

    // Retain separate post-reduction frontier direct verification:
    let post_reduce_stack = ParserGSS::from_single_stack(vec![0, 1], TerminalsDisallowed::new());
    let closed_post = close_provider_control_stacks(&provider, &post_reduce_stack);
    assert!(
        stacks_finished_with_provider(&provider, &closed_post, EOF),
        "When post-reduction frontier is control-closed, it reaches State 2 which accepts on EOF"
    );
}

#[test]
fn provider_eof_reduction_two_stage_control_accepts() {
    // Chain: reduce1 -> control1 -> reduce2 -> control2 -> Accept
    // [0, 10] -> Reduce(nt0=0, len=1) -> [0, 1]
    // [0, 1] on CTRL1 (991) -> shifts 2 -> [0, 1, 2]
    // [0, 1, 2] on EOF -> Reduce(nt1=1, len=2) -> base [0], goto nt1=1 -> [0, 3]
    // [0, 3] on CTRL2 (992) -> shifts 4 -> [0, 3, 4]
    // [0, 3, 4] on EOF -> Accept
    const CTRL1: u32 = 991;
    const CTRL2: u32 = 992;
    static REDUCE_1: Action = Action::Reduce(0, 1);
    static SHIFT_CTRL1: Action = Action::Shift(2, false);
    static REDUCE_2: Action = Action::Reduce(1, 2);
    static SHIFT_CTRL2: Action = Action::Shift(4, false);
    static ACCEPT_ACT: Action = Action::Accept;

    struct TwoStageProvider;
    impl ParserActionProvider for TwoStageProvider {
        type Symbol = u32;
        fn action(&self, state: u32, symbol: u32) -> Option<ProvidedAction<'_>> {
            match (state, symbol) {
                (10, EOF) => Some(ProvidedAction {
                    action: ProvidedActionRef::Local { scope: 0, action: &REDUCE_1 },
                    reduction_scope: 0,
                    extra_stack_shifts: SmallVec::new(),
                }),
                (1, CTRL1) => Some(ProvidedAction {
                    action: ProvidedActionRef::Local { scope: 0, action: &SHIFT_CTRL1 },
                    reduction_scope: 0,
                    extra_stack_shifts: SmallVec::new(),
                }),
                (2, EOF) => Some(ProvidedAction {
                    action: ProvidedActionRef::Local { scope: 0, action: &REDUCE_2 },
                    reduction_scope: 0,
                    extra_stack_shifts: SmallVec::new(),
                }),
                (3, CTRL2) => Some(ProvidedAction {
                    action: ProvidedActionRef::Local { scope: 0, action: &SHIFT_CTRL2 },
                    reduction_scope: 0,
                    extra_stack_shifts: SmallVec::new(),
                }),
                (4, EOF) => Some(ProvidedAction {
                    action: ProvidedActionRef::Local { scope: 0, action: &ACCEPT_ACT },
                    reduction_scope: 0,
                    extra_stack_shifts: SmallVec::new(),
                }),
                _ => None,
            }
        }
        fn control_symbols(&self, state: u32, out: &mut SmallVec<[Self::Symbol; 4]>) {
            match state {
                1 => out.push(CTRL1),
                3 => out.push(CTRL2),
                _ => {}
            }
        }
        fn scope_state(&self, _scope: u32, local: u32) -> Option<u32> {
            Some(local)
        }
        fn goto_target(&self, _scope: u32, goto_from: u32, nt: u32) -> Option<(u32, bool)> {
            match (goto_from, nt) {
                (0, 0) => Some((1, false)),
                (0, 1) => Some((3, false)),
                _ => None,
            }
        }
        fn state_count_hint(&self) -> usize {
            16
        }
    }

    let provider = TwoStageProvider;
    let stack = ParserGSS::from_single_stack(vec![0, 10], TerminalsDisallowed::new());
    assert!(
        stacks_finished_with_provider(&provider, &stack, EOF),
        "Two consecutive reduce/control stages should complete to Accept"
    );
}

#[test]
fn provider_eof_reduction_exposing_control_transition_no_accept_rejects() {
    // Negative case: reduce -> control -> non-accepting state
    const CTRL_SYM: u32 = 999;
    static ACTION_REDUCE: Action = Action::Reduce(0, 1);
    static ACTION_SHIFT_CTRL: Action = Action::Shift(2, false);

    struct NoAcceptProvider;
    impl ParserActionProvider for NoAcceptProvider {
        type Symbol = u32;
        fn action(&self, state: u32, symbol: u32) -> Option<ProvidedAction<'_>> {
            match (state, symbol) {
                (10, EOF) => Some(ProvidedAction {
                    action: ProvidedActionRef::Local { scope: 0, action: &ACTION_REDUCE },
                    reduction_scope: 0,
                    extra_stack_shifts: SmallVec::new(),
                }),
                (1, CTRL_SYM) => Some(ProvidedAction {
                    action: ProvidedActionRef::Local { scope: 0, action: &ACTION_SHIFT_CTRL },
                    reduction_scope: 0,
                    extra_stack_shifts: SmallVec::new(),
                }),
                // State 2 has no EOF action
                _ => None,
            }
        }
        fn control_symbols(&self, state: u32, out: &mut SmallVec<[Self::Symbol; 4]>) {
            if state == 1 {
                out.push(CTRL_SYM);
            }
        }
        fn scope_state(&self, _scope: u32, local: u32) -> Option<u32> {
            Some(local)
        }
        fn goto_target(&self, _scope: u32, goto_from: u32, nt: u32) -> Option<(u32, bool)> {
            if goto_from == 0 && nt == 0 {
                Some((1, false))
            } else {
                None
            }
        }
        fn state_count_hint(&self) -> usize {
            16
        }
    }

    let provider = NoAcceptProvider;
    let stack = ParserGSS::from_single_stack(vec![0, 10], TerminalsDisallowed::new());
    assert!(
        !stacks_finished_with_provider(&provider, &stack, EOF),
        "reduce -> control -> non-accepting state must reject"
    );
}

#[test]
fn provider_scoped_composition_eof_control_closure_accepts() {
    let parent_table = build_test_table(
        2,
        1,
        &[&[(0, Action::Shift(1, false))], &[(EOF, Action::Accept)]],
        &[&[], &[]],
    );
    let child_table = build_test_table(
        1,
        1,
        &[&[(EOF, Action::Accept)]],
        &[&[]],
    );
    let tables: [&GLRTable; 2] = [&parent_table, &child_table];
    let links = [ScopedSubgrammarLink {
        parent_component: 0,
        slot_terminal: 0,
        child_component: 1,
        child_start: 0,
        return_pop: 1,
        child_start_nullable: true,
    }];
    let provider = DisjointComponentActionProvider::new(&tables, &links).unwrap();

    let child_start_scoped = provider.scoped_state(1, 0).unwrap();
    let stack = ParserGSS::from_single_stack(
        vec![provider.scoped_state(0, 0).unwrap(), child_start_scoped],
        TerminalsDisallowed::new(),
    );
    let parent_eof = ScopedParserSymbol::Terminal { component: 0, terminal: EOF };
    assert!(stacks_finished_with_provider(&provider, &stack, parent_eof));
}

#[test]
fn advance_stacks_materializes_single_concrete_path_for_split() {
    let token = 0;
    let nt = 0;
    let table = build_test_table(
        6,
        1,
        &[
            &[],
            &[],
            &[(token, Action::Split {
                shift: Some((3, false)),
                reduces: vec![(nt, 1)],
                accept: false,
            })],
            &[],
            &[(token, Action::Shift(5, false))],
            &[],
        ],
        &[
            &[(nt, (4, false))],
            &[],
            &[],
            &[],
            &[],
            &[],
        ],
    );

    let acc = TerminalsDisallowed::new();
    let before = ParserGSS::from_stacks(&[
        (vec![0, 1], acc.clone()),
        (vec![0, 2], acc.clone()),
    ])
    .popn(1)
    .push(2);
    let expected = ParserGSS::from_stacks(&[
        (vec![0, 2, 3], acc.clone()),
        (vec![0, 4, 5], acc),
    ]);

    let mut actual_stacks = advance_stacks(&table, &before, token).to_stacks(4_096).expect("stack enumeration exceeded explicit limit");
    let mut expected_stacks = expected.to_stacks(4_096).expect("stack enumeration exceeded explicit limit");
    actual_stacks.sort_by(|left, right| left.0.cmp(&right.0));
    expected_stacks.sort_by(|left, right| left.0.cmp(&right.0));

    assert_eq!(actual_stacks, expected_stacks);
}

#[test]
fn disjoint_top_terminal_admission_preserves_earlier_success() {
    let token_a = 0;
    let token_b = 1;
    let nt = 0;
    let mut table = build_test_table(
        6,
        2,
        &[
            &[],
            &[],
            &[(token_a, Action::Reduce(nt, 1))],
            &[(token_a, Action::Shift(5, false))],
            &[],
            &[],
        ],
        &[&[(nt, (3, false))], &[], &[], &[], &[], &[]],
    );
    table.admission_policy = AdmissionPolicy::ExactSimulation;
    let stack = ParserGSS::from_stacks(&[
        (vec![0, 2], TerminalsDisallowed::new()),
        (vec![1, 4], TerminalsDisallowed::new()),
    ]);

    assert_eq!(
        stack_may_advance_disjoint_top_terminals_bounded(
            &table,
            &stack,
            &[(2, token_a), (4, token_b)],
        ),
        Some(true),
    );
}

#[test]
fn disjoint_top_terminal_advance_matches_pointwise_union() {
    let token_a = 0;
    let token_b = 1;
    let nt_a = 0;
    let nt_b = 1;
    let table = build_test_table(
        8,
        2,
        &[
            &[],
            &[],
            &[(token_a, Action::Reduce(nt_a, 1))],
            &[(token_a, Action::Shift(6, false))],
            &[(token_b, Action::Reduce(nt_b, 1))],
            &[(token_b, Action::Shift(7, false))],
            &[],
            &[],
        ],
        &[
            &[(nt_a, (3, false))],
            &[(nt_b, (5, false))],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
        ],
    );
    let acc = TerminalsDisallowed::new();
    let stack = ParserGSS::from_stacks(&[
        (vec![0, 2], acc.clone()),
        (vec![1, 4], acc),
    ]);

    let actual = advance_stacks_disjoint_top_terminals_bounded(
        &table,
        &stack,
        &[(2, token_a), (4, token_b)],
    )
    .expect("bounded disjoint advance should apply");
    let left = advance_stacks(&table, &stack.isolate(Some(2)), token_a);
    let right = advance_stacks(&table, &stack.isolate(Some(4)), token_b);
    let expected = left.merge(&right);
    assert!(
        actual
            .semantically_eq(&expected, 32)
            .expect("small exact stack languages should fit"),
        "actual={:?} expected={:?}",
        actual.to_stacks(32),
        expected.to_stacks(32),
    );
}

#[test]
fn indexed_guarded_vstack_matches_linear_guarded_vstack() {
    let token = 0;
    let mut table = build_test_table(
        1,
        1,
        &[&[(
            token,
            Action::GuardedStackShifts(vec![
                GuardedStackShift {
                    guards: vec![
                        StackShiftGuard {
                            pop: 1,
                            states: vec![10, 20],
                        },
                        StackShiftGuard {
                            pop: 2,
                            states: vec![1],
                        },
                    ],
                    pop: 3,
                    pushes: vec![50],
                },
                GuardedStackShift {
                    guards: vec![
                        StackShiftGuard {
                            pop: 1,
                            states: vec![10],
                        },
                        StackShiftGuard {
                            pop: 2,
                            states: vec![2],
                        },
                    ],
                    pop: 3,
                    pushes: vec![51],
                },
                GuardedStackShift {
                    guards: vec![StackShiftGuard {
                        pop: 1,
                        states: vec![10, 20],
                    }],
                    pop: 2,
                    pushes: vec![52],
                },
                GuardedStackShift {
                    guards: vec![
                        StackShiftGuard {
                            pop: 1,
                            states: vec![30],
                        },
                        StackShiftGuard {
                            pop: 2,
                            states: vec![1],
                        },
                    ],
                    pop: 3,
                    pushes: vec![53],
                },
            ]),
        )]],
        &[&[]],
    );
    table.rebuild_guarded_shift_index();

    let shifts = match table.action(0, token) {
        Some(Action::GuardedStackShifts(shifts)) => shifts,
        other => panic!("expected guarded stack shifts, got {other:?}"),
    };
    let index = table
        .guarded_shift_index(0, token)
        .expect("expected guarded shift index");

    let stack_a = ParserGSS::from_single_stack(vec![0, 1, 10, 99], TerminalsDisallowed::new());
    let stack_b = ParserGSS::from_single_stack(vec![0, 2, 10, 99], TerminalsDisallowed::new());

    for stack in [&stack_a, &stack_b] {
        let vstack = stack.try_virtual_stack().expect("expected virtual stack");
        let mut indexed = apply_guarded_stack_shifts_to_vstack(&vstack, shifts, Some(index)).to_stacks(4_096).expect("stack enumeration exceeded explicit limit");
        let mut linear = apply_guarded_stack_shifts_to_vstack(&vstack, shifts, None).to_stacks(4_096).expect("stack enumeration exceeded explicit limit");
        indexed.sort_by(|left, right| left.0.cmp(&right.0));
        linear.sort_by(|left, right| left.0.cmp(&right.0));
        assert_eq!(indexed, linear);
    }
}
