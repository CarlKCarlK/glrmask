use super::{
    add_completed_lr0_reductions, build_experimental_core_merged_table, build_lalr_table,
    build_lalr_table_impl, build_lr1_table, compute_lalr_item_lookaheads,
    build_lr0_item_sets, build_lr1_item_sets,
    build_lr1_item_sets_with_preclosure_reuse, build_table,
    build_table_with_default_construction, grouped_item_lookahead_counts,
    finish_table_with_early_identity_quotient, initialize_pending_and_goto,
    lr1_core_key, pending_table_has_conflict, refine_experimental_core_partition,
    selected_glr_table_construction, slr_reductions_would_conflict,
    try_build_direct_regular_table, try_build_direct_regular_table_reference,
    union_experimental_core_rows, ExperimentalCoreCompatibilitySig, Item,
    BitSet, FxHashMap, LR1ItemCore, LR1ItemSet,
};
use crate::compiler::glr::accumulator::TerminalsDisallowed;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::glr::parser::{
    advance_stacks, stack_may_advance_on, stacks_finished, ParserGSS,
};
use crate::compiler::glr::table::{Action, AdmissionPolicy, GLRTable, GlrTableConstruction};
use crate::grammar::flat::{
    DirectRegularAutomaton, DirectRegularState, GrammarDef, Rule, Symbol, Terminal,
};
use std::collections::{BTreeMap, VecDeque};

fn multi_lookahead_grammar() -> AnalyzedGrammar {
    let grammar = GrammarDef {
        rules: vec![
            Rule {
                lhs: 0,
                rhs: vec![Symbol::Nonterminal(1), Symbol::Terminal(1)],
            },
            Rule {
                lhs: 0,
                rhs: vec![Symbol::Nonterminal(1), Symbol::Terminal(2)],
            },
            Rule {
                lhs: 0,
                rhs: vec![Symbol::Nonterminal(1), Symbol::Terminal(3)],
            },
            Rule {
                lhs: 1,
                rhs: vec![Symbol::Terminal(0)],
            },
        ],
        start: 0,
        terminals: vec![
            Terminal::Literal { id: 0, bytes: b"x".to_vec() },
            Terminal::Literal { id: 1, bytes: b"a".to_vec() },
            Terminal::Literal { id: 2, bytes: b"b".to_vec() },
            Terminal::Literal { id: 3, bytes: b"c".to_vec() },
        ],
        ..GrammarDef::default()
    };
    AnalyzedGrammar::from_grammar_def(&grammar)
}

fn assert_early_lalr_identity_quotient_exact(grammar: &AnalyzedGrammar) {
    let mut ordinary = build_lalr_table_impl(grammar, false, false);
    let mut early = build_lalr_table_impl(grammar, true, false);
    let fixed_point = build_lalr_table_impl(grammar, true, true);
    ordinary.merge_identical_rows();
    early.merge_identical_rows();

    assert_eq!(ordinary.num_states, early.num_states);
    assert_eq!(ordinary.action, early.action);
    assert_eq!(ordinary.goto, early.goto);
    assert_eq!(ordinary.rules, early.rules);
    assert_eq!(ordinary.forwarded_shifts, early.forwarded_shifts);
    assert_eq!(ordinary.num_terminals, early.num_terminals);
    assert_eq!(ordinary.num_rules, early.num_rules);
    assert_eq!(ordinary.construction, early.construction);
    assert_eq!(ordinary.admission_policy, early.admission_policy);

    assert_eq!(ordinary.num_states, fixed_point.num_states);
    assert_eq!(ordinary.action, fixed_point.action);
    assert_eq!(ordinary.goto, fixed_point.goto);
    assert_eq!(ordinary.rules, fixed_point.rules);
    assert_eq!(ordinary.forwarded_shifts, fixed_point.forwarded_shifts);
    assert_eq!(ordinary.num_terminals, fixed_point.num_terminals);
    assert_eq!(ordinary.num_rules, fixed_point.num_rules);
    assert_eq!(ordinary.construction, fixed_point.construction);
    assert_eq!(ordinary.admission_policy, fixed_point.admission_policy);
}

#[test]
fn early_lalr_identity_quotient_matches_ordinary_row_fixed_point() {
    assert_early_lalr_identity_quotient_exact(&multi_lookahead_grammar());
    assert_early_lalr_identity_quotient_exact(&mysterious_conflict_grammar());
    assert_early_lalr_identity_quotient_exact(&generated_unit_dag_grammar(5, 3, true, true));
}

#[test]
fn lr1_preclosure_kernel_reuse_preserves_canonical_item_sets_and_transitions() {
    for grammar in [
        multi_lookahead_grammar(),
        recursive_ambiguous_grammar(),
        template_like_grammar(),
    ] {
        let (reference_sets, reference_transitions) =
            build_lr1_item_sets_with_preclosure_reuse(&grammar, false);
        let (reused_sets, reused_transitions) =
            build_lr1_item_sets_with_preclosure_reuse(&grammar, true);
        assert_eq!(reused_sets, reference_sets);
        assert_eq!(reused_transitions, reference_transitions);
    }
}

fn direct_regular_grammar() -> AnalyzedGrammar {
    let mut grammar = GrammarDef {
        rules: vec![
            Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            },
            Rule {
                lhs: 0,
                rhs: vec![Symbol::Nonterminal(0), Symbol::Terminal(1)],
            },
            Rule {
                lhs: 1,
                rhs: vec![Symbol::Nonterminal(0), Symbol::Terminal(2)],
            },
            Rule {
                lhs: 2,
                rhs: vec![Symbol::Nonterminal(0), Symbol::Terminal(2)],
            },
            Rule {
                lhs: 3,
                rhs: vec![Symbol::Nonterminal(1), Symbol::Terminal(3)],
            },
            Rule {
                lhs: 3,
                rhs: vec![Symbol::Nonterminal(2), Symbol::Terminal(4)],
            },
        ],
        start: 3,
        terminals: (0..5)
            .map(|id| Terminal::Literal {
                id,
                bytes: vec![b'a' + id as u8],
            })
            .collect(),
        direct_regular_automaton: Some(DirectRegularAutomaton {
            states: vec![
                DirectRegularState {
                    transitions: BTreeMap::from([(0, vec![1])]),
                    ..DirectRegularState::default()
                },
                DirectRegularState {
                    transitions: BTreeMap::from([(1, vec![1]), (2, vec![2])]),
                    ..DirectRegularState::default()
                },
                DirectRegularState {
                    transitions: BTreeMap::from([(3, vec![3]), (4, vec![3])]),
                    ..DirectRegularState::default()
                },
                DirectRegularState {
                    is_accepting: true,
                    ..DirectRegularState::default()
                },
            ],
            start_states: vec![0],
        }),
        ..GrammarDef::default()
    };
    let direct_regular_automaton = grammar.direct_regular_automaton.take();
    let mut analyzed = AnalyzedGrammar::from_grammar_def(&grammar);
    analyzed.direct_regular_automaton = direct_regular_automaton;
    analyzed
}

fn direct_regular_epsilon_cycle_grammar() -> AnalyzedGrammar {
    AnalyzedGrammar::from_grammar_def(&GrammarDef {
        rules: vec![Rule {
            lhs: 0,
            rhs: vec![Symbol::Terminal(0)],
        }],
        start: 0,
        terminals: (0..3)
            .map(|id| Terminal::Literal {
                id,
                bytes: vec![b'a' + id as u8],
            })
            .collect(),
        direct_regular_automaton: Some(DirectRegularAutomaton {
            states: vec![
                DirectRegularState {
                    transitions: BTreeMap::from([(0, vec![3])]),
                    epsilons: vec![1, 2],
                    is_accepting: false,
                },
                DirectRegularState {
                    transitions: BTreeMap::from([(1, vec![1])]),
                    epsilons: vec![2],
                    is_accepting: false,
                },
                DirectRegularState {
                    transitions: BTreeMap::from([(2, vec![3])]),
                    epsilons: vec![1],
                    is_accepting: true,
                },
                DirectRegularState {
                    is_accepting: true,
                    ..DirectRegularState::default()
                },
            ],
            start_states: vec![0, 1],
        }),
        ..GrammarDef::default()
    })
}

fn large_left_linear_grammar() -> AnalyzedGrammar {
    let mut rules = vec![Rule {
        lhs: 0,
        rhs: vec![Symbol::Terminal(0)],
    }];
    for state in 1..=64u32 {
        rules.push(Rule {
            lhs: state,
            rhs: vec![Symbol::Nonterminal(state - 1), Symbol::Terminal(0)],
        });
    }
    AnalyzedGrammar::from_grammar_def(&GrammarDef {
        rules,
        start: 64,
        terminals: vec![Terminal::Literal {
            id: 0,
            bytes: b"x".to_vec(),
        }],
        ..GrammarDef::default()
    })
}

fn mysterious_conflict_grammar() -> AnalyzedGrammar {
    let grammar = GrammarDef {
        rules: vec![
            Rule {
                lhs: 0,
                rhs: vec![
                    Symbol::Terminal(0),
                    Symbol::Nonterminal(1),
                    Symbol::Terminal(3),
                ],
            },
            Rule {
                lhs: 0,
                rhs: vec![
                    Symbol::Terminal(1),
                    Symbol::Nonterminal(1),
                    Symbol::Terminal(4),
                ],
            },
            Rule {
                lhs: 0,
                rhs: vec![
                    Symbol::Terminal(0),
                    Symbol::Nonterminal(2),
                    Symbol::Terminal(4),
                ],
            },
            Rule {
                lhs: 0,
                rhs: vec![
                    Symbol::Terminal(1),
                    Symbol::Nonterminal(2),
                    Symbol::Terminal(3),
                ],
            },
            Rule {
                lhs: 1,
                rhs: vec![Symbol::Terminal(2)],
            },
            Rule {
                lhs: 2,
                rhs: vec![Symbol::Terminal(2)],
            },
        ],
        start: 0,
        terminals: vec![
            Terminal::Literal { id: 0, bytes: b"a".to_vec() },
            Terminal::Literal { id: 1, bytes: b"b".to_vec() },
            Terminal::Literal { id: 2, bytes: b"c".to_vec() },
            Terminal::Literal { id: 3, bytes: b"d".to_vec() },
            Terminal::Literal { id: 4, bytes: b"e".to_vec() },
        ],
        ..GrammarDef::default()
    };
    AnalyzedGrammar::from_grammar_def(&grammar)
}

fn terminal(id: u32, byte: u8) -> Terminal {
    Terminal::Literal {
        id,
        bytes: vec![byte],
    }
}

fn analyzed(rules: Vec<Rule>, start: u32, num_terminals: u32) -> AnalyzedGrammar {
    let grammar = GrammarDef {
        rules,
        start,
        terminals: (0..num_terminals)
            .map(|id| terminal(id, b'a'.wrapping_add(id as u8)))
            .collect(),
        ..GrammarDef::default()
    };
    AnalyzedGrammar::from_grammar_def(&grammar)
}

fn unit_chain_grammar() -> AnalyzedGrammar {
    analyzed(
        vec![
            Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(1), Symbol::Terminal(2)] },
            Rule { lhs: 1, rhs: vec![Symbol::Nonterminal(2)] },
            Rule { lhs: 2, rhs: vec![Symbol::Nonterminal(3)] },
            Rule { lhs: 3, rhs: vec![Symbol::Terminal(0)] },
            Rule { lhs: 3, rhs: vec![Symbol::Terminal(1)] },
        ],
        0,
        3,
    )
}

fn ambiguous_unit_chain_grammar() -> AnalyzedGrammar {
    analyzed(
        vec![
            Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(1), Symbol::Terminal(1)] },
            Rule { lhs: 1, rhs: vec![Symbol::Nonterminal(2)] },
            Rule { lhs: 1, rhs: vec![Symbol::Nonterminal(3)] },
            Rule { lhs: 2, rhs: vec![Symbol::Nonterminal(4)] },
            Rule { lhs: 3, rhs: vec![Symbol::Nonterminal(4)] },
            Rule { lhs: 4, rhs: vec![Symbol::Terminal(0)] },
        ],
        0,
        2,
    )
}

fn nullable_unit_chain_grammar() -> AnalyzedGrammar {
    analyzed(
        vec![
            Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(1), Symbol::Terminal(2)] },
            Rule { lhs: 1, rhs: Vec::new() },
            Rule { lhs: 1, rhs: vec![Symbol::Nonterminal(2)] },
            Rule { lhs: 2, rhs: vec![Symbol::Nonterminal(3)] },
            Rule { lhs: 3, rhs: vec![Symbol::Terminal(0)] },
            Rule { lhs: 3, rhs: vec![Symbol::Terminal(1)] },
        ],
        0,
        3,
    )
}

fn recursive_ambiguous_grammar() -> AnalyzedGrammar {
    analyzed(
        vec![
            Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(1)] },
            Rule { lhs: 1, rhs: vec![Symbol::Nonterminal(2)] },
            Rule { lhs: 1, rhs: vec![Symbol::Nonterminal(3)] },
            Rule { lhs: 2, rhs: vec![Symbol::Terminal(0)] },
            Rule {
                lhs: 2,
                rhs: vec![
                    Symbol::Terminal(1),
                    Symbol::Nonterminal(1),
                    Symbol::Terminal(2),
                ],
            },
            Rule { lhs: 3, rhs: vec![Symbol::Terminal(0)] },
            Rule {
                lhs: 3,
                rhs: vec![
                    Symbol::Terminal(1),
                    Symbol::Nonterminal(1),
                    Symbol::Terminal(2),
                ],
            },
        ],
        0,
        3,
    )
}

fn template_like_grammar() -> AnalyzedGrammar {
    analyzed(
        vec![
            Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(1)] },
            Rule {
                lhs: 1,
                rhs: vec![
                    Symbol::Terminal(0),
                    Symbol::Terminal(1),
                    Symbol::Nonterminal(2),
                    Symbol::Terminal(2),
                ],
            },
            Rule { lhs: 2, rhs: vec![Symbol::Nonterminal(3)] },
            Rule {
                lhs: 2,
                rhs: vec![
                    Symbol::Nonterminal(2),
                    Symbol::Terminal(3),
                    Symbol::Nonterminal(3),
                ],
            },
            Rule { lhs: 3, rhs: vec![Symbol::Nonterminal(4)] },
            Rule { lhs: 3, rhs: vec![Symbol::Nonterminal(5)] },
            Rule { lhs: 4, rhs: vec![Symbol::Terminal(0)] },
            Rule {
                lhs: 4,
                rhs: vec![
                    Symbol::Terminal(0),
                    Symbol::Terminal(1),
                    Symbol::Nonterminal(2),
                    Symbol::Terminal(2),
                ],
            },
            Rule { lhs: 5, rhs: vec![Symbol::Terminal(0)] },
            Rule {
                lhs: 5,
                rhs: vec![
                    Symbol::Terminal(0),
                    Symbol::Terminal(1),
                    Symbol::Nonterminal(2),
                    Symbol::Terminal(2),
                ],
            },
        ],
        0,
        4,
    )
}

fn core_merged_tables_before_and_after_unit_lowering(
    grammar: &AnalyzedGrammar,
) -> (GLRTable, GLRTable, usize) {
    let (item_sets, transitions) = build_lr1_item_sets(grammar);
    let raw = build_experimental_core_merged_table(grammar, &item_sets, &transitions)
        .expect("test grammar must support core merging");

    let mut reference = raw.clone();
    reference.prune_unreachable_states();
    reference.rebuild_guarded_shift_index();

    let mut lowered = raw;
    let report = lowered.collapse_sr_unit_reductions_for_correctness_oracle();
    assert!(!report.aborted, "unit lowering aborted: {report:?}");
    lowered.extend_advance_rows_from_actions();
    lowered.prune_unreachable_states();
    lowered.rebuild_guarded_shift_index();
    let changed_states = report.changed_original_states.len();
    (reference, lowered, changed_states)
}

fn assert_bounded_parser_bisimulation(
    name: &str,
    grammar: &AnalyzedGrammar,
    max_depth: usize,
) -> usize {
    let (reference, lowered, changed_states) =
        core_merged_tables_before_and_after_unit_lowering(grammar);
    assert_eq!(reference.num_terminals, lowered.num_terminals);
    let start = ParserGSS::from_single_stack(vec![0], TerminalsDisallowed::new());
    let mut queue = VecDeque::from([(Vec::<u32>::new(), start.clone(), start)]);
    let mut visited_prefixes = 0usize;

    while let Some((prefix, left, right)) = queue.pop_front() {
        visited_prefixes += 1;
        assert_eq!(
            stacks_finished(&reference, &left),
            stacks_finished(&lowered, &right),
            "completion mismatch for {name} at prefix {prefix:?}",
        );
        if prefix.len() == max_depth {
            continue;
        }

        for terminal in 0..reference.num_terminals {
            assert_eq!(
                stack_may_advance_on(&reference, &left, terminal),
                stack_may_advance_on(&lowered, &right, terminal),
                "admission mismatch for {name} at prefix {prefix:?}, terminal {terminal}",
            );
            let left_next = advance_stacks(&reference, &left, terminal);
            let right_next = advance_stacks(&lowered, &right, terminal);
            assert_eq!(
                left_next.is_empty(),
                right_next.is_empty(),
                "recognition mismatch for {name} at prefix {prefix:?}, terminal {terminal}",
            );
            if !left_next.is_empty() {
                let mut next_prefix = prefix.clone();
                next_prefix.push(terminal);
                queue.push_back((next_prefix, left_next, right_next));
            }
        }
    }
    assert!(visited_prefixes > 1, "{name} test did not explore any successors");
    changed_states
}

fn generated_unit_dag_grammar(
    chain_len: usize,
    branches: usize,
    nullable: bool,
    recursive_tail: bool,
) -> AnalyzedGrammar {
    assert!(chain_len > 0);
    assert!(branches > 0);
    let mut rules = vec![Rule {
        lhs: 0,
        rhs: vec![Symbol::Nonterminal(1), Symbol::Terminal(2)],
    }];
    if nullable {
        rules.push(Rule { lhs: 1, rhs: Vec::new() });
    }
    let first_chain_nt = 2u32;
    for branch in 0..branches {
        let branch_start = first_chain_nt + (branch * chain_len) as u32;
        rules.push(Rule {
            lhs: 1,
            rhs: vec![Symbol::Nonterminal(branch_start)],
        });
        for level in 0..chain_len {
            let current = branch_start + level as u32;
            if level + 1 < chain_len {
                rules.push(Rule {
                    lhs: current,
                    rhs: vec![Symbol::Nonterminal(current + 1)],
                });
            } else {
                rules.push(Rule {
                    lhs: current,
                    rhs: vec![Symbol::Terminal((branch % 2) as u32)],
                });
                if recursive_tail {
                    rules.push(Rule {
                        lhs: current,
                        rhs: vec![
                            Symbol::Terminal((branch % 2) as u32),
                            Symbol::Nonterminal(1),
                        ],
                    });
                }
            }
        }
    }
    analyzed(rules, 0, 3)
}

#[test]
fn core_merged_unit_lowering_is_bisimilar_on_generated_unit_dags() {
    let mut cases = 0usize;
    let mut changed_cases = 0usize;
    for chain_len in 1..=5 {
        for branches in 1..=3 {
            for nullable in [false, true] {
                for recursive_tail in [false, true] {
                    let name = format!(
                        "generated-chain-{chain_len}-branches-{branches}-nullable-{nullable}-recursive-{recursive_tail}"
                    );
                    let grammar = generated_unit_dag_grammar(
                        chain_len,
                        branches,
                        nullable,
                        recursive_tail,
                    );
                    let changed = assert_bounded_parser_bisimulation(&name, &grammar, 7);
                    cases += 1;
                    changed_cases += usize::from(changed > 0);
                }
            }
        }
    }
    assert_eq!(cases, 60);
    assert!(
        changed_cases >= 20,
        "generated gate was too vacuous: only {changed_cases}/{cases} grammars changed"
    );
}

#[test]
fn core_merged_unit_lowering_is_bisimilar_on_small_grammar_families() {
    let cases = [
        ("multi-lookahead", multi_lookahead_grammar(), 5usize),
        ("mysterious-conflict", mysterious_conflict_grammar(), 5),
        ("unit-chain", unit_chain_grammar(), 6),
        ("ambiguous-unit-chain", ambiguous_unit_chain_grammar(), 6),
        ("nullable-unit-chain", nullable_unit_chain_grammar(), 6),
        ("recursive-ambiguous", recursive_ambiguous_grammar(), 7),
        ("template-like", template_like_grammar(), 8),
    ];
    for (name, grammar, depth) in cases {
        let _ = assert_bounded_parser_bisimulation(name, &grammar, depth);
    }
}

#[test]
fn grouped_lr1_items_merge_multiple_lookaheads_on_one_core() {
    let grammar = multi_lookahead_grammar();
    let counts = grouped_item_lookahead_counts(&grammar);

    assert!(
        counts
            .iter()
            .flatten()
            .any(|&(rule, dot, _stack_depth, lookahead_count)| {
                rule == 4 && dot == 1 && lookahead_count == 3
            }),
        "{counts:?}"
    );
}

#[test]
fn grouped_lr1_items_still_emit_expected_lowered_shift_actions() {
    let grammar = multi_lookahead_grammar();
    let table = build_table(&grammar);

    assert!(table.action.iter().any(|row| {
        matches!(row.get(&1), Some(Action::Shift(_, true)))
            && matches!(row.get(&2), Some(Action::Shift(_, true)))
            && matches!(row.get(&3), Some(Action::Shift(_, true)))
    }));
}

#[test]
fn default_build_uses_core_merged_exact_admission() {
    let grammar = multi_lookahead_grammar();
    let table = build_table(&grammar);

    assert_eq!(
        table.construction,
        GlrTableConstruction::ExperimentalCoreMerged
    );
    assert_eq!(table.admission_policy, AdmissionPolicy::ExactSimulation);
}

#[test]
fn direct_regular_reused_closure_workspace_matches_reference() {
    let grammar = direct_regular_epsilon_cycle_grammar();
    let direct = try_build_direct_regular_table(&grammar)
        .expect("workspace direct table should build");
    let reference = try_build_direct_regular_table_reference(&grammar)
        .expect("closure-reference direct table should build");
    assert_eq!(direct.action, reference.action);
    assert_eq!(direct.advance, reference.advance);
    assert_eq!(direct.num_states, reference.num_states);
}

#[test]
fn direct_regular_dag_rows_match_reference() {
    let grammar = direct_regular_grammar();
    let ((initial, initial_advance, _), rows) = super::direct_regular_dag_rows(&grammar)
        .expect("regular fixture should be an epsilon DAG");
    let mut action = Vec::with_capacity(rows.len() + 1);
    let mut advance = Vec::with_capacity(rows.len() + 1);
    action.push(initial);
    advance.push(initial_advance);
    for (row, advance_row, _) in rows {
        action.push(row);
        advance.push(advance_row);
    }
    let reference = try_build_direct_regular_table_reference(&grammar)
        .expect("closure-reference direct table should build");
    assert_eq!(action, reference.action);
    assert_eq!(advance, reference.advance);
}

#[test]
fn direct_regular_dag_rows_reject_epsilon_cycles() {
    assert!(super::direct_regular_dag_rows(&direct_regular_epsilon_cycle_grammar()).is_none());
}

#[test]
fn direct_regular_table_matches_legacy_parser() {
    let grammar = direct_regular_grammar();
    let direct = try_build_direct_regular_table(&grammar).expect("regular table should build");
    let mut reference_grammar = grammar.clone();
    reference_grammar.direct_regular_automaton = None;
    let reference = build_table_with_default_construction(
        &reference_grammar,
        GlrTableConstruction::LegacyRowBisim,
    );

    let start = ParserGSS::from_single_stack(vec![0], TerminalsDisallowed::new());
    let mut queue = VecDeque::from([(Vec::<u32>::new(), start.clone(), start)]);
    let mut visited = 0usize;
    while let Some((prefix, left, right)) = queue.pop_front() {
        visited += 1;
        assert_eq!(
            stacks_finished(&reference, &left),
            stacks_finished(&direct, &right),
            "completion mismatch at {prefix:?}",
        );
        if prefix.len() == 6 {
            continue;
        }
        for terminal in 0..reference.num_terminals {
            assert_eq!(
                stack_may_advance_on(&reference, &left, terminal),
                stack_may_advance_on(&direct, &right, terminal),
                "admission mismatch at {prefix:?} on {terminal}",
            );
            let left_next = advance_stacks(&reference, &left, terminal);
            let right_next = advance_stacks(&direct, &right, terminal);
            assert_eq!(
                left_next.is_empty(),
                right_next.is_empty(),
                "recognition mismatch at {prefix:?} on {terminal}",
            );
            if !left_next.is_empty() {
                let mut next = prefix.clone();
                next.push(terminal);
                queue.push_back((next, left_next, right_next));
            }
        }
    }
    assert!(visited > 4);
}

#[test]
fn large_left_linear_grammar_prefers_row_bisim() {
    let grammar = large_left_linear_grammar();
    assert_eq!(
        selected_glr_table_construction(
            &grammar,
            GlrTableConstruction::ExperimentalCoreMerged,
        ),
        GlrTableConstruction::LegacyRowBisim,
    );
}

#[test]
fn very_large_legacy_grammar_prefers_lalr() {
    let rules = (0..40_000)
        .map(|_| Rule {
            lhs: 0,
            rhs: vec![Symbol::Terminal(0)],
        })
        .collect();
    let grammar = AnalyzedGrammar::from_grammar_def(&GrammarDef {
        rules,
        start: 0,
        terminals: vec![Terminal::Literal {
            id: 0,
            bytes: b"x".to_vec(),
        }],
        ..GrammarDef::default()
    });
    assert_eq!(
        selected_glr_table_construction(
            &grammar,
            GlrTableConstruction::LegacyRowBisim,
        ),
        GlrTableConstruction::Lalr,
    );

    let many_terminals = analyzed(
        (0..40_000)
            .map(|_| Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            })
            .collect(),
        0,
        513,
    );
    assert_eq!(
        selected_glr_table_construction(
            &many_terminals,
            GlrTableConstruction::LegacyRowBisim,
        ),
        GlrTableConstruction::LegacyRowBisim,
    );
}

#[test]
fn pushdown_grammar_keeps_core_merged_default() {
    let grammar = multi_lookahead_grammar();
    assert_eq!(
        selected_glr_table_construction(
            &grammar,
            GlrTableConstruction::ExperimentalCoreMerged,
        ),
        GlrTableConstruction::ExperimentalCoreMerged,
    );
}

#[test]
fn legacy_row_bisim_can_be_requested_as_default() {
    let grammar = multi_lookahead_grammar();
    let table = build_table_with_default_construction(
        &grammar,
        GlrTableConstruction::LegacyRowBisim,
    );

    assert_eq!(table.construction, GlrTableConstruction::LegacyRowBisim);
    assert_eq!(table.admission_policy, AdmissionPolicy::RowPresenceExact);
}

#[test]
fn lalr_builds_real_lr0_based_table() {
    let grammar = multi_lookahead_grammar();
    let table = build_lalr_table(&grammar);

    assert_eq!(table.construction, GlrTableConstruction::Lalr);
    assert_eq!(table.admission_policy, AdmissionPolicy::ExactSimulation);
    assert!(!table.has_ambiguity(), "{:#?}", table.ambiguous_actions());
}

#[test]
fn lalr_nullable_unit_chain_matches_legacy_recognition() {
    let grammar = nullable_unit_chain_grammar();
    let legacy = build_table_with_default_construction(
        &grammar,
        GlrTableConstruction::LegacyRowBisim,
    );
    let lalr = build_lalr_table(&grammar);
    let start = ParserGSS::from_single_stack(vec![0], TerminalsDisallowed::new());
    let mut queue = VecDeque::from([(Vec::<u32>::new(), start.clone(), start)]);
    let mut visited = 0usize;

    while let Some((prefix, left, right)) = queue.pop_front() {
        visited += 1;
        assert_eq!(
            stacks_finished(&legacy, &left),
            stacks_finished(&lalr, &right),
            "completion mismatch at {prefix:?}",
        );
        if prefix.len() == 6 {
            continue;
        }
        for terminal in 0..legacy.num_terminals {
            assert_eq!(
                stack_may_advance_on(&legacy, &left, terminal),
                stack_may_advance_on(&lalr, &right, terminal),
                "admission mismatch at {prefix:?} on {terminal}",
            );
            let left_next = advance_stacks(&legacy, &left, terminal);
            let right_next = advance_stacks(&lalr, &right, terminal);
            assert_eq!(
                left_next.is_empty(),
                right_next.is_empty(),
                "recognition mismatch at {prefix:?} on {terminal}",
            );
            if !left_next.is_empty() {
                let mut next = prefix.clone();
                next.push(terminal);
                queue.push_back((next, left_next, right_next));
            }
        }
    }
    assert!(visited > 4);
}

#[test]
fn lalr_matches_legacy_on_generated_nullable_grammars() {
    for chain_len in 1..=5 {
        for branches in 1..=3 {
                let recursive_tail = false;
                let grammar = generated_unit_dag_grammar(chain_len, branches, true, false);
                let legacy = build_table_with_default_construction(
                    &grammar,
                    GlrTableConstruction::LegacyRowBisim,
                );
                let lalr = build_lalr_table(&grammar);
                let start = ParserGSS::from_single_stack(
                    vec![0],
                    TerminalsDisallowed::new(),
                );
                let mut queue = VecDeque::from([(
                    Vec::<u32>::new(),
                    start.clone(),
                    start,
                )]);
                while let Some((prefix, left, right)) = queue.pop_front() {
                    assert_eq!(
                        stacks_finished(&legacy, &left),
                        stacks_finished(&lalr, &right),
                        "completion mismatch chain={chain_len} branches={branches} recursive={recursive_tail} prefix={prefix:?}",
                    );
                    if prefix.len() == 7 {
                        continue;
                    }
                    for terminal in 0..legacy.num_terminals {
                        assert_eq!(
                            stack_may_advance_on(&legacy, &left, terminal),
                            stack_may_advance_on(&lalr, &right, terminal),
                            "admission mismatch chain={chain_len} branches={branches} recursive={recursive_tail} prefix={prefix:?} terminal={terminal}",
                        );
                        let left_next = advance_stacks(&legacy, &left, terminal);
                        let right_next = advance_stacks(&lalr, &right, terminal);
                        assert_eq!(
                            left_next.is_empty(),
                            right_next.is_empty(),
                            "recognition mismatch chain={chain_len} branches={branches} recursive={recursive_tail} prefix={prefix:?} terminal={terminal}",
                        );
                        if !left_next.is_empty() {
                            let mut next = prefix.clone();
                            next.push(terminal);
                            queue.push_back((next, left_next, right_next));
                        }
                    }
                }
        }
    }
}

#[test]
fn flat_lr0_kernels_preserve_every_state_and_transition() {
    use super::{build_lr0_item_sets_flat, build_lr0_item_sets_tree};
    let mut grammars = vec![multi_lookahead_grammar(), mysterious_conflict_grammar(),
        recursive_ambiguous_grammar(), template_like_grammar(), large_left_linear_grammar(),
        unit_chain_grammar(), ambiguous_unit_chain_grammar(), nullable_unit_chain_grammar()];
    for n in 1..=7 {
        for branches in 1..=3 {
            for recursive in [false, true] {
                grammars.push(generated_unit_dag_grammar(n, branches, true, recursive));
            }
        }
    }
    for grammar in grammars {
        let (a, ta) = build_lr0_item_sets_tree(&grammar);
        let (b, tb) = build_lr0_item_sets_flat(&grammar);
        assert_eq!(ta, tb);
        assert_eq!(a.len(), b.len());
        for (a, b) in a.iter().zip(&b) {
            assert_eq!(a.kernel, b.kernel);
            assert_eq!(a.closure, b.closure);
        }
    }
}

#[test]
fn slr_conflict_preflight_matches_materialized_reductions() {
    for grammar in [
        multi_lookahead_grammar(),
        mysterious_conflict_grammar(),
        recursive_ambiguous_grammar(),
        template_like_grammar(),
    ] {
        let (states, transitions) = build_lr0_item_sets(&grammar);
        let (mut pending, _, _) = initialize_pending_and_goto(&transitions);
        add_completed_lr0_reductions(&grammar, &states, None, &mut pending);
        let materialized = pending_table_has_conflict(&pending);
        let preflight = slr_reductions_would_conflict(&grammar, &states, &transitions);
        assert_eq!(preflight, materialized);
    }
}

#[test]
fn conflict_free_slr_fast_path_matches_lalr_admission_and_recognition() {
    let mut checked = 0usize;
    for grammar in [
        nullable_unit_chain_grammar(),
        generated_unit_dag_grammar(4, 2, true, false),
        generated_unit_dag_grammar(5, 3, true, false),
    ] {
        let (states, transitions) = build_lr0_item_sets(&grammar);
        if slr_reductions_would_conflict(&grammar, &states, &transitions) {
            continue;
        }
        checked += 1;

        let (mut slr_pending, slr_goto, slr_forwarded) =
            initialize_pending_and_goto(&transitions);
        add_completed_lr0_reductions(&grammar, &states, None, &mut slr_pending);
        let slr = finish_table_with_early_identity_quotient(
            &grammar,
            slr_pending,
            slr_goto,
            slr_forwarded,
            GlrTableConstruction::Lalr,
            AdmissionPolicy::ExactSimulation,
            false,
        );

        let (mut lalr_pending, lalr_goto, lalr_forwarded) =
            initialize_pending_and_goto(&transitions);
        let lookaheads = compute_lalr_item_lookaheads(&grammar, &states, &transitions);
        add_completed_lr0_reductions(
            &grammar,
            &states,
            Some(&lookaheads),
            &mut lalr_pending,
        );
        let lalr = finish_table_with_early_identity_quotient(
            &grammar,
            lalr_pending,
            lalr_goto,
            lalr_forwarded,
            GlrTableConstruction::Lalr,
            AdmissionPolicy::ExactSimulation,
            false,
        );

        let start = ParserGSS::from_single_stack(vec![0], TerminalsDisallowed::new());
        let mut queue = VecDeque::from([(Vec::<u32>::new(), start.clone(), start)]);
        while let Some((prefix, left, right)) = queue.pop_front() {
            assert_eq!(
                stacks_finished(&slr, &left),
                stacks_finished(&lalr, &right),
                "completion mismatch at {prefix:?}",
            );
            if prefix.len() == 7 {
                continue;
            }
            for terminal in 0..slr.num_terminals {
                assert_eq!(
                    stack_may_advance_on(&slr, &left, terminal),
                    stack_may_advance_on(&lalr, &right, terminal),
                    "admission mismatch at {prefix:?} on {terminal}",
                );
                let left_next = advance_stacks(&slr, &left, terminal);
                let right_next = advance_stacks(&lalr, &right, terminal);
                assert_eq!(
                    left_next.is_empty(),
                    right_next.is_empty(),
                    "recognition mismatch at {prefix:?} on {terminal}",
                );
                if !left_next.is_empty() {
                    let mut next = prefix.clone();
                    next.push(terminal);
                    queue.push_back((next, left_next, right_next));
                }
            }
        }
    }
    assert!(checked > 0, "test corpus must contain an SLR conflict-free grammar");
}

#[test]
fn lalr_exposes_classic_lr1_not_lalr_conflict() {
    let grammar = mysterious_conflict_grammar();
    let table = build_lalr_table(&grammar);

    assert_eq!(table.construction, GlrTableConstruction::Lalr);
    assert!(table.has_ambiguity(), "expected GLR split from LALR merge");
}

#[test]
fn lalr_conflict_preserves_legacy_recognition_on_bounded_prefixes() {
    let grammar = mysterious_conflict_grammar();
    let legacy = build_table_with_default_construction(
        &grammar,
        GlrTableConstruction::LegacyRowBisim,
    );
    let lalr = build_lalr_table(&grammar);
    let start = ParserGSS::from_single_stack(vec![0], TerminalsDisallowed::new());
    let mut queue = VecDeque::from([(Vec::<u32>::new(), start.clone(), start)]);
    let mut visited = 0usize;

    while let Some((prefix, left, right)) = queue.pop_front() {
        visited += 1;
        assert_eq!(
            stacks_finished(&legacy, &left),
            stacks_finished(&lalr, &right),
            "completion mismatch at {prefix:?}",
        );
        if prefix.len() == 5 {
            continue;
        }
        for terminal in 0..legacy.num_terminals {
            assert_eq!(
                stack_may_advance_on(&legacy, &left, terminal),
                stack_may_advance_on(&lalr, &right, terminal),
                "admission mismatch at {prefix:?} on {terminal}",
            );
            let left_next = advance_stacks(&legacy, &left, terminal);
            let right_next = advance_stacks(&lalr, &right, terminal);
            assert_eq!(
                left_next.is_empty(),
                right_next.is_empty(),
                "recognition mismatch at {prefix:?} on {terminal}",
            );
            if !left_next.is_empty() {
                let mut next = prefix.clone();
                next.push(terminal);
                queue.push_back((next, left_next, right_next));
            }
        }
    }
    assert!(visited > 4);
}

#[test]
fn probe_normal_grammar_compiler_nullable_recursive_growing_stack() {
    // Grammar:
    // Rule 0: S -> A (nt 0 -> nt 1)
    // Rule 1: A -> A B (nt 1 -> nt 1, nt 2) -- recursive with net push
    // Rule 2: B -> epsilon (nt 2 -> empty) -- nullable
    // Rule 3: A -> 'x' (nt 1 -> terminal 0)
    let grammar = analyzed(
        vec![
            Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(1)] },
            Rule { lhs: 1, rhs: vec![Symbol::Nonterminal(1), Symbol::Nonterminal(2)] },
            Rule { lhs: 2, rhs: Vec::new() },
            Rule { lhs: 1, rhs: vec![Symbol::Terminal(0)] },
        ],
        0,
        1,
    );
    let table = super::build_table(&grammar);
    eprintln!(
        "[GRAMMAR_SCC_PROBE] compiled: num_states={} num_rules={} construction={:?}",
        table.num_states, table.num_rules, table.construction
    );
    for (state, row) in table.action.iter().enumerate() {
        for (terminal, action) in row.iter() {
            eprintln!("[GRAMMAR_SCC_PROBE] state={state} term={terminal} action={action:?}");
        }
    }
    for (state, row) in table.goto.iter().enumerate() {
        for (nt, (target, replace)) in row.iter() {
            eprintln!("[GRAMMAR_SCC_PROBE] goto state={state} nt={nt} target={target} replace={replace}");
        }
    }
}

fn refine_experimental_core_partition_btree_reference(
    table: &GLRTable,
    core_keys: &[Vec<Item>],
) -> Vec<u32> {
    let mut class_by_core: BTreeMap<Vec<Item>, u32> = BTreeMap::new();
    let mut partition = Vec::with_capacity(core_keys.len());
    for key in core_keys {
        let next = class_by_core.len() as u32;
        partition.push(*class_by_core.entry(key.clone()).or_insert(next));
    }

    loop {
        let mut sig_to_class: BTreeMap<ExperimentalCoreCompatibilitySig, u32> = BTreeMap::new();
        let mut next_partition = Vec::with_capacity(partition.len());
        for state in 0..table.num_states as usize {
            let sig = ExperimentalCoreCompatibilitySig::new(table, state, partition[state], &partition);
            let next = sig_to_class.len() as u32;
            next_partition.push(*sig_to_class.entry(sig).or_insert(next));
        }
        if next_partition == partition {
            return partition;
        }
        partition = next_partition;
    }
}

fn js_like_statement_grammar() -> AnalyzedGrammar {
    analyzed(
        vec![
            Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(1)] },
            Rule { lhs: 1, rhs: vec![Symbol::Nonterminal(2), Symbol::Nonterminal(1)] },
            Rule { lhs: 1, rhs: Vec::new() },
            Rule { lhs: 2, rhs: vec![Symbol::Nonterminal(3)] },
            Rule { lhs: 2, rhs: vec![Symbol::Nonterminal(4)] },
            Rule { lhs: 2, rhs: vec![Symbol::Nonterminal(5)] },
            Rule { lhs: 2, rhs: vec![Symbol::Nonterminal(6)] },
            Rule {
                lhs: 3,
                rhs: vec![
                    Symbol::Terminal(2),
                    Symbol::Terminal(0),
                    Symbol::Terminal(3),
                    Symbol::Nonterminal(7),
                    Symbol::Terminal(4),
                ],
            },
            Rule {
                lhs: 3,
                rhs: vec![
                    Symbol::Terminal(2),
                    Symbol::Terminal(0),
                    Symbol::Terminal(4),
                ],
            },
            Rule {
                lhs: 4,
                rhs: vec![
                    Symbol::Terminal(5),
                    Symbol::Terminal(6),
                    Symbol::Nonterminal(7),
                    Symbol::Terminal(7),
                    Symbol::Nonterminal(2),
                ],
            },
            Rule {
                lhs: 4,
                rhs: vec![
                    Symbol::Terminal(5),
                    Symbol::Terminal(6),
                    Symbol::Nonterminal(7),
                    Symbol::Terminal(7),
                    Symbol::Nonterminal(2),
                    Symbol::Terminal(8),
                    Symbol::Nonterminal(2),
                ],
            },
            Rule {
                lhs: 5,
                rhs: vec![Symbol::Nonterminal(7), Symbol::Terminal(4)],
            },
            Rule {
                lhs: 5,
                rhs: vec![Symbol::Terminal(13), Symbol::Nonterminal(7), Symbol::Terminal(4)],
            },
            Rule {
                lhs: 5,
                rhs: vec![Symbol::Terminal(4)],
            },
            Rule {
                lhs: 6,
                rhs: vec![Symbol::Terminal(9), Symbol::Nonterminal(1), Symbol::Terminal(10)],
            },
            Rule { lhs: 7, rhs: vec![Symbol::Nonterminal(8)] },
            Rule {
                lhs: 8,
                rhs: vec![Symbol::Nonterminal(8), Symbol::Terminal(11), Symbol::Nonterminal(9)],
            },
            Rule { lhs: 8, rhs: vec![Symbol::Nonterminal(9)] },
            Rule {
                lhs: 9,
                rhs: vec![Symbol::Nonterminal(9), Symbol::Terminal(12), Symbol::Nonterminal(10)],
            },
            Rule { lhs: 9, rhs: vec![Symbol::Nonterminal(10)] },
            Rule { lhs: 10, rhs: vec![Symbol::Terminal(0)] },
            Rule { lhs: 10, rhs: vec![Symbol::Terminal(1)] },
            Rule {
                lhs: 10,
                rhs: vec![Symbol::Terminal(6), Symbol::Nonterminal(7), Symbol::Terminal(7)],
            },
            Rule {
                lhs: 10,
                rhs: vec![
                    Symbol::Terminal(0),
                    Symbol::Terminal(6),
                    Symbol::Nonterminal(7),
                    Symbol::Terminal(7),
                ],
            },
        ],
        0,
        14,
    )
}

#[test]
fn experimental_core_partition_fast_path_matches_btree_reference() {
    let mut grammars = vec![
        multi_lookahead_grammar(),
        mysterious_conflict_grammar(),
        recursive_ambiguous_grammar(),
        template_like_grammar(),
        large_left_linear_grammar(),
        unit_chain_grammar(),
        ambiguous_unit_chain_grammar(),
        nullable_unit_chain_grammar(),
        js_like_statement_grammar(),
    ];
    for n in 1..=5 {
        for branches in 1..=3 {
            for recursive in [false, true] {
                grammars.push(generated_unit_dag_grammar(n, branches, true, recursive));
            }
        }
    }

    for (idx, grammar) in grammars.into_iter().enumerate() {
        let (item_sets, transitions) = build_lr1_item_sets(&grammar);
        let canonical = build_lr1_table(&grammar, &item_sets, &transitions);
        let core_keys = item_sets.iter().map(lr1_core_key).collect::<Vec<_>>();

        let btree_partition =
            refine_experimental_core_partition_btree_reference(&canonical, &core_keys);
        let fast_partition = refine_experimental_core_partition(&canonical, &core_keys);

        assert_eq!(
            fast_partition, btree_partition,
            "partition mismatch on grammar index {idx}",
        );

        let fast_table =
            build_experimental_core_merged_table(&grammar, &item_sets, &transitions);
        let btree_table = union_experimental_core_rows(canonical.clone(), &btree_partition);

        assert_eq!(
            fast_table.as_ref().map(|t| t.num_states),
            btree_table.as_ref().map(|t| t.num_states),
            "num_states mismatch on grammar index {idx}",
        );
        assert_eq!(
            fast_table.as_ref().map(|t| &t.action),
            btree_table.as_ref().map(|t| &t.action),
            "action table mismatch on grammar index {idx}",
        );
        assert_eq!(
            fast_table.as_ref().map(|t| &t.goto),
            btree_table.as_ref().map(|t| &t.goto),
            "goto table mismatch on grammar index {idx}",
        );
        assert_eq!(
            fast_table.as_ref().map(|t| &t.forwarded_shifts),
            btree_table.as_ref().map(|t| &t.forwarded_shifts),
            "forwarded_shifts mismatch on grammar index {idx}",
        );
    }
}

#[test]
fn direct_core_table_is_default_and_reference_override_is_explicit() {
    for value in [None, Some(""), Some("0"), Some("false"), Some("off")] {
        assert!(super::direct_core_table_enabled_for_override(value));
    }
    for value in ["1", "true", "yes", "on", " TRUE "] {
        assert!(!super::direct_core_table_enabled_for_override(Some(value)));
    }
}

#[test]
fn emitted_lr1_lookaheads_keep_first_empty_request_and_exact_subsets() {
    for len in [0, 1, 64, 65, 257] {
        let mut emitted = FxHashMap::default();
        let empty = BitSet::new(len);
        assert!(super::emitted_lr1_lookaheads_changed(&mut emitted, 11, &empty));
        assert!(!super::emitted_lr1_lookaheads_changed(&mut emitted, 11, &empty));
        for bit in 0..len {
            let mut request = BitSet::new(len);
            request.set(bit);
            assert!(super::emitted_lr1_lookaheads_changed(&mut emitted, 11, &request));
            assert!(!super::emitted_lr1_lookaheads_changed(&mut emitted, 11, &request));
        }
        assert_eq!(emitted[&11].count_ones(), len);
        // A different nonterminal has not emitted anything yet.
        assert!(super::emitted_lr1_lookaheads_changed(&mut emitted, 12, &empty));
    }
}

#[test]
fn emitted_lr1_closure_preserves_zero_lookahead_entry_cores() {
    let grammar = AnalyzedGrammar::from_grammar_def(&GrammarDef {
        rules: vec![
            Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(1)] },
            Rule { lhs: 1, rhs: vec![Symbol::Nonterminal(2)] },
            Rule { lhs: 1, rhs: vec![Symbol::Nonterminal(3)] },
            Rule { lhs: 2, rhs: vec![Symbol::Terminal(0)] },
            Rule { lhs: 3, rhs: vec![Symbol::Terminal(1)] },
        ],
        start: 0,
        terminals: vec![
            Terminal::Literal { id: 0, bytes: b"a".to_vec() },
            Terminal::Literal { id: 1, bytes: b"b".to_vec() },
        ],
        ..GrammarDef::default()
    });
    let rule = grammar.rules.iter().position(|r| matches!(r.rhs.as_slice(),
        [Symbol::Nonterminal(nt)] if grammar.rules_by_lhs[*nt as usize].len() > 1
    )).expect("multi-production nonterminal in singleton suffix");
    let mut seed = LR1ItemSet::new();
    for depth in [7, 9] {
        seed.insert(LR1ItemCore::new(rule as u32, 0, depth),
                    BitSet::new(grammar.num_terminals as usize + 1));
    }
    let suffix = super::rule_suffix_first_sets(&grammar);
    let reference = super::lr1_closure::<false>(seed.clone(), &grammar, &suffix);
    let candidate = super::lr1_closure::<true>(seed, &grammar, &suffix);
    assert!(candidate.len() > 2, "first empty emission must create entry cores");
    assert_eq!(candidate, reference);
}

#[test]
fn lr1_emission_cache_is_default_with_explicit_reference_override() {
    for disabled in [None, Some(""), Some("0"), Some("false"), Some("off")] {
        assert!(super::lr1_emission_cache_enabled_for_override(disabled));
    }
    for disabled in ["1", "true", "yes", "on", " TRUE "] {
        assert!(!super::lr1_emission_cache_enabled_for_override(Some(disabled)));
    }
}

#[test]
fn cached_lr1_emissions_preserve_canonical_items_edges_and_complete_tables() {
    let mut grammars = vec![
        multi_lookahead_grammar(), mysterious_conflict_grammar(),
        recursive_ambiguous_grammar(), template_like_grammar(),
        large_left_linear_grammar(), unit_chain_grammar(),
        ambiguous_unit_chain_grammar(), nullable_unit_chain_grammar(),
        js_like_statement_grammar(),
    ];
    for depth in 1..=5 {
        for branches in 1..=3 {
            for nullable in [false, true] {
                for recursive in [false, true] {
                    grammars.push(generated_unit_dag_grammar(depth, branches, nullable, recursive));
                }
            }
        }
    }
    for (index, grammar) in grammars.iter().enumerate() {
        for reuse in [false, true] {
            let (old_items, old_edges) = super::build_lr1_item_sets_impl::<false>(grammar, reuse);
            let (new_items, new_edges) = super::build_lr1_item_sets_impl::<true>(grammar, reuse);
            assert_eq!(new_items, old_items, "canonical items grammar={index} reuse={reuse}");
            assert_eq!(new_edges, old_edges, "canonical edges grammar={index} reuse={reuse}");
            let reference = super::build_core_merged_table_from_items(grammar, &old_items, &old_edges).unwrap();
            let candidate = super::build_core_merged_table_from_items(grammar, &new_items, &new_edges).unwrap();
            assert_eq!(bincode::serialize(&candidate).unwrap(), bincode::serialize(&reference).unwrap(),
                       "complete table grammar={index} reuse={reuse}");
        }
    }
}

#[test]
fn direct_core_table_matches_materialized_canonical_rows() {
    let mut grammars = vec![
        multi_lookahead_grammar(), mysterious_conflict_grammar(),
        recursive_ambiguous_grammar(), template_like_grammar(),
        large_left_linear_grammar(), unit_chain_grammar(),
        ambiguous_unit_chain_grammar(), nullable_unit_chain_grammar(),
        js_like_statement_grammar(),
    ];
    for depth in 1..=5 {
        for branches in 1..=3 {
            for nullable in [false, true] {
                for recursive in [false, true] {
                    grammars.push(generated_unit_dag_grammar(
                        depth, branches, nullable, recursive,
                    ));
                }
            }
        }
    }
    let mut reduced_cases = 0;
    for (index, grammar) in grammars.iter().enumerate() {
        let (items, transitions) = build_lr1_item_sets(grammar);
        let canonical = build_lr1_table(grammar, &items, &transitions);
        let cores = items.iter().map(lr1_core_key).collect::<Vec<_>>();
        let partition = refine_experimental_core_partition(&canonical, &cores);
        // Check the signature correspondence independently of the shared
        // partition loop, including identity and final quotient targets.
        for targets in [(0..items.len() as u32).collect::<Vec<_>>(), partition] {
            for (state, row) in transitions.iter().enumerate() {
                assert_eq!(
                    ExperimentalCoreCompatibilitySig::from_transitions(
                        row, targets[state], &targets,
                    ),
                    ExperimentalCoreCompatibilitySig::new(
                        &canonical, state, targets[state], &targets,
                    ),
                    "signature grammar={index} state={state}",
                );
            }
        }
        let reference = super::build_core_merged_table_materialized(grammar, &items, &transitions)
            .expect("materialized table");
        let direct = super::build_core_merged_table_from_items(grammar, &items, &transitions)
            .expect("direct table");
        assert_eq!(direct.action, reference.action, "actions grammar={index}");
        assert_eq!(direct.goto, reference.goto, "gotos grammar={index}");
        for (left, right) in direct.action.iter().zip(&reference.action) {
            let (super::ActionRow::Sparse(left), super::ActionRow::Sparse(right)) = (left, right)
                else { panic!("fresh core tables must retain sparse rows"); };
            assert_eq!(std::mem::discriminant(left), std::mem::discriminant(right),
                       "runtime action-row representation grammar={index}");
        }
        for (left, right) in direct.goto.iter().zip(&reference.goto) {
            assert_eq!(std::mem::discriminant(left), std::mem::discriminant(right),
                       "runtime goto-row representation grammar={index}");
        }
        assert_eq!(direct.advance, reference.advance, "admission grammar={index}");
        assert_eq!(direct.unconditional_advance, reference.unconditional_advance);
        assert_eq!(direct.forwarded_shifts, reference.forwarded_shifts);
        let direct_bytes = bincode::serialize(&direct).unwrap();
        let reference_bytes = bincode::serialize(&reference).unwrap();
        assert!(direct_bytes == reference_bytes,
                "complete serialized table grammar={index}, first difference={:?}, lengths={}/{}",
                direct_bytes.iter().zip(&reference_bytes).position(|(a, b)| a != b),
                direct_bytes.len(), reference_bytes.len());
        reduced_cases += usize::from(direct.num_states < canonical.num_states);
    }
    assert!(reduced_cases > 0, "must exercise canonical states merged into fewer rows");
}

#[test]
fn direct_core_table_preserves_replace_and_forwarded_shift_flags() {
    let grammar = js_like_statement_grammar();
    let (items, original) = build_lr1_item_sets(&grammar);
    for salt in 0..4 {
        let mut transitions = original.clone();
        for (state, row) in transitions.iter_mut().enumerate() {
            for (offset, (_, (_, replace, forwarded))) in row.iter_mut().enumerate() {
                *replace = (state + offset + salt) % 3 == 0;
                *forwarded = (state + offset + salt) % 2 == 0;
            }
        }
        let reference = super::build_core_merged_table_materialized(&grammar, &items, &transitions)
            .expect("materialized flagged table");
        let direct = super::build_core_merged_table_from_items(&grammar, &items, &transitions)
            .expect("direct flagged table");
        assert_eq!(direct.action, reference.action);
        assert_eq!(direct.goto, reference.goto);
        assert_eq!(direct.advance, reference.advance);
        assert_eq!(direct.forwarded_shifts, reference.forwarded_shifts);
        let direct_bytes = bincode::serialize(&direct).unwrap();
        let reference_bytes = bincode::serialize(&reference).unwrap();
        assert!(direct_bytes == reference_bytes,
                "flagged complete table salt={salt}, first difference={:?}",
                direct_bytes.iter().zip(&reference_bytes).position(|(a, b)| a != b));
    }
    assert!(super::build_core_merged_table_from_items(&grammar, &items, &[]).is_none());
}
