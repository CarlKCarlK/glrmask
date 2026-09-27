use super::*;
use crate::glr::accumulator::TerminalsDisallowed;
use crate::glr::parser::{ParserGSS, advance_stacks};
use std::sync::Mutex;

static ENV_LOCK: Mutex<()> = Mutex::new(());

struct EnvVarGuard {
    name: &'static str,
    previous: Option<String>,
}

impl EnvVarGuard {
    fn set(name: &'static str, value: &str) -> Self {
        let previous = std::env::var(name).ok();
        unsafe {
            std::env::set_var(name, value);
        }
        Self { name, previous }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        unsafe {
            if let Some(previous) = &self.previous {
                std::env::set_var(self.name, previous);
            } else {
                std::env::remove_var(self.name);
            }
        }
    }
}

fn default_unit_inline_budget() -> UnitInlineBudget {
    UnitInlineBudget {
        started_at: std::time::Instant::now(),
        max_ms: DEFAULT_UNIT_INLINE_WORK_MAX_WALL_MS,
        max_iterations: DEFAULT_UNIT_INLINE_WORK_MAX_ITERATIONS,
        max_cells: DEFAULT_UNIT_INLINE_WORK_MAX_CELLS,
        max_synthetic_states: DEFAULT_UNIT_INLINE_WORK_MAX_SYNTHETIC_STATES,
        max_stack_effect_visits: DEFAULT_UNIT_INLINE_WORK_MAX_STACK_EFFECT_VISITS,
        iterations: std::sync::atomic::AtomicUsize::new(0),
        cells: std::sync::atomic::AtomicUsize::new(0),
        synthetic_states: std::sync::atomic::AtomicUsize::new(0),
        stack_effect_visits: std::sync::atomic::AtomicUsize::new(0),
        abort_code: std::sync::atomic::AtomicU8::new(ABORT_NONE),
    }
}

#[test]
fn control_predecessor_propagation_matches_synchronous_reference() {
    fn next(seed: &mut u64) -> u64 {
        *seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *seed
    }

    fn random_pushes(seed: &mut u64, nstates: u32) -> Vec<u32> {
        let len = (next(seed) % 3 + 1) as usize;
        (0..len)
            .map(|_| (next(seed) % u64::from(nstates)) as u32)
            .collect()
    }

    fn random_action(seed: &mut u64, nstates: u32) -> Action {
        match next(seed) % 8 {
            0 => Action::Shift(
                (next(seed) % u64::from(nstates)) as u32,
                next(seed) & 1 != 0,
            ),
            1 => Action::ReplaceShifts(
                (0..(next(seed) % 3 + 1))
                    .map(|_| (next(seed) % u64::from(nstates)) as u32)
                    .collect(),
            ),
            2 => Action::StackShifts(
                (0..(next(seed) % 3 + 1))
                    .map(|_| StackShift {
                        pop: (next(seed) % 4) as u32,
                        pushes: random_pushes(seed, nstates),
                    })
                    .collect(),
            ),
            3 => Action::GuardedStackShifts(
                (0..(next(seed) % 3 + 1))
                    .map(|_| GuardedStackShift {
                        guards: Vec::new(),
                        pop: (next(seed) % 4) as u32,
                        pushes: random_pushes(seed, nstates),
                    })
                    .collect(),
            ),
            4 => Action::Split {
                shift: Some((
                    (next(seed) % u64::from(nstates)) as u32,
                    next(seed) & 1 != 0,
                )),
                reduces: Vec::new(),
                accept: false,
            },
            5 => Action::Skip,
            6 => Action::Reduce((next(seed) % 4) as u32, (next(seed) % 4) as u32),
            _ => Action::Accept,
        }
    }

    let mut seed = 0xD1CE_F00D_5EED_BAADu64;
    for case in 0..256 {
        let nstates = (next(&mut seed) % 9 + 2) as usize;
        let mut action = vec![ActionRow::default(); nstates];
        let mut goto = vec![GotoRow::default(); nstates];

        for state in 0..nstates {
            for terminal in 0..4 {
                if next(&mut seed) % 3 != 0 {
                    action[state].insert(
                        terminal,
                        random_action(&mut seed, nstates as u32),
                    );
                }
            }
            for nonterminal in 0..4 {
                if next(&mut seed) & 1 != 0 {
                    goto[state].insert(
                        nonterminal,
                        (
                            (next(&mut seed) % nstates as u64) as u32,
                            next(&mut seed) & 1 != 0,
                        ),
                    );
                }
            }
        }

        let table = GLRTable {
            action,
            goto,
            num_states: nstates as u32,
            num_terminals: 4,
            num_rules: 0,
            rules: Vec::new(),
            nonterminal_display_names: Vec::new(),
            embedded_start: Default::default(),
            construction: GlrTableConstruction::LegacyRowBisim,
            admission_policy: AdmissionPolicy::RowPresenceExact,
            advance: Vec::new(),
            unconditional_advance: Vec::new(),
            forwarded_shifts: FxHashSet::default(),
            control_terminals: Default::default(),
            skip_terminals: Default::default(),
            guarded_shift_index: Vec::new(),
            direct_regular_wide_frontiers: Vec::new(),
        };

        let expected = build_control_elimination_predecessors_synchronous_reference(&table)
            .unwrap_or_else(|error| panic!("reference failed for case {case}: {error}"));
        let actual = build_control_elimination_predecessors(&table)
            .unwrap_or_else(|error| panic!("optimized failed for case {case}: {error}"));
        assert_eq!(actual, expected, "case {case}");

        let mut origins = (0..nstates as u32)
            .filter(|_| next(&mut seed) & 3 == 0)
            .collect::<Vec<_>>();
        if origins.is_empty() {
            origins.push((next(&mut seed) % nstates as u64) as u32);
        }
        origins.sort_unstable();
        origins.dedup();
        let demand = build_control_elimination_predecessors_demand(&table, &origins)
            .unwrap_or_else(|error| panic!("demand solver failed for case {case}: {error}"));
        let mut required = vec![false; nstates];
        let mut queue = VecDeque::from(origins.clone());
        while let Some(state) = queue.pop_front() {
            if std::mem::replace(&mut required[state as usize], true) {
                continue;
            }
            queue.extend(expected[state as usize].iter().copied());
        }
        for state in 0..nstates {
            if required[state] || !demand[state].is_empty() {
                assert_eq!(
                    demand[state], expected[state],
                    "materialized demand row differs for case {case}, state {state}, origins {origins:?}",
                );
            }
        }
    }
}

fn guarded_effect(pop: u32, guard_count: u32) -> GuardedStackShift {
    GuardedStackShift {
        guards: (0..guard_count)
            .map(|guard_pop| StackShiftGuard {
                pop: guard_pop,
                states: vec![0],
            })
            .collect(),
        pop: pop.max(guard_count.saturating_sub(1)),
        pushes: vec![1],
    }
}

fn table_with_stack_shifts(
    shifts: Vec<StackShift>,
    goto_rows: &[(u32, &[(NonterminalID, (u32, bool))])],
) -> GLRTable {
    let num_states = 8;
    let mut action = vec![ActionRow::default(); num_states];
    action[0].insert(0, Action::StackShifts(shifts));

    let mut goto = vec![GotoRow::default(); num_states];
    for &(state, row) in goto_rows {
        for &(nt, target) in row {
            goto[state as usize].insert(nt, target);
        }
    }

    GLRTable {
        action,
        goto,
        num_states: num_states as u32,
        num_terminals: 1,
        num_rules: 0,
        rules: Vec::new(),
        nonterminal_display_names: Vec::new(),
        embedded_start: Default::default(),
        construction: GlrTableConstruction::LegacyRowBisim,
        admission_policy: AdmissionPolicy::RowPresenceExact,
        advance: Vec::new(),
        unconditional_advance: Vec::new(),
        forwarded_shifts: FxHashSet::default(),
        control_terminals: Default::default(),
        skip_terminals: Default::default(),
        guarded_shift_index: Vec::new(),
        direct_regular_wide_frontiers: Vec::new(),
    }
}

fn stack_shifts_at_start(table: &GLRTable) -> Vec<StackShift> {
    match table.action(0, 0).expect("expected action at state 0 terminal 0") {
        Action::StackShifts(shifts) => shifts.clone(),
        action => panic!("expected stack shifts, got {action:?}"),
    }
}

#[test]
fn contextual_state_sharing_compacts_ambiguous_top_with_distinct_stack_contexts() {
    // Two reachable parser paths have different caller states (1 and 2)
    // but structurally corresponding tops (3 and 4). Sharing 3~4 should
    // turn the GSS frontier from two top LR IDs into one while retaining
    // the caller distinction one frame below as an exact guard.
    let mut action = vec![ActionRow::default(); 6];
    action[0].insert(10, Action::Shift(1, false));
    action[0].insert(11, Action::Shift(2, false));
    action[1].insert(20, Action::Shift(3, false));
    action[2].insert(21, Action::Shift(4, false));
    action[3].insert(0, Action::Shift(5, true));
    action[4].insert(0, Action::Shift(5, true));
    action[5].insert(1, Action::Shift(5, true));
    let mut table = GLRTable {
        action,
        goto: vec![GotoRow::default(); 6],
        num_states: 6,
        num_terminals: 22,
        num_rules: 0,
        rules: Vec::new(),
        nonterminal_display_names: Vec::new(),
        embedded_start: Default::default(),
        construction: GlrTableConstruction::LegacyRowBisim,
        admission_policy: AdmissionPolicy::ExactSimulation,
        advance: Vec::new(),
        unconditional_advance: Vec::new(),
        forwarded_shifts: FxHashSet::default(),
        control_terminals: Default::default(),
        skip_terminals: Default::default(),
        guarded_shift_index: Vec::new(),
        direct_regular_wide_frontiers: Vec::new(),
    };
    table.rebuild_advance_rows_from_actions();
    let baseline = table.clone();
    let mapping = table.share_context_distinguishable_states_exact(&[vec![3, 4]]);

    assert_eq!(mapping[3], mapping[4]);
    assert!(table.num_states < baseline.num_states);

    let acc = TerminalsDisallowed::new();
    let baseline_before = ParserGSS::from_stacks(&[
        (vec![0, 1, 3], acc.clone()),
        (vec![0, 2, 4], acc.clone()),
    ]);
    let shared_before = ParserGSS::from_stacks(&[
        (
            vec![mapping[0], mapping[1], mapping[3]],
            acc.clone(),
        ),
        (vec![mapping[0], mapping[2], mapping[4]], acc),
    ]);
    assert_eq!(baseline_before.peek_values().len(), 2);
    assert_eq!(shared_before.peek_values().len(), 1);

    let baseline_after = advance_stacks(&baseline, &baseline_before, 0);
    let shared_after = advance_stacks(&table, &shared_before, 0);
    let mut expected = baseline_after
        .to_stacks(32)
        .unwrap()
        .into_iter()
        .map(|(stack, acc)| {
            (
                stack
                    .into_iter()
                    .map(|state| mapping[state as usize])
                    .collect::<Vec<_>>(),
                acc,
            )
        })
        .collect::<Vec<_>>();
    let mut actual = shared_after.to_stacks(32).unwrap();
    expected.sort_by(|left, right| left.0.cmp(&right.0));
    actual.sort_by(|left, right| left.0.cmp(&right.0));
    assert_eq!(actual, expected);
}

#[test]
fn state_local_stack_effect_exhaustion_does_not_abort_parent_budget() {
    let parent = default_unit_inline_budget();
    let child = parent.child_with_stack_effect_visit_limit(2);

    assert!(child.record_stack_effect_visit());
    assert!(child.record_stack_effect_visit());
    assert!(!child.record_stack_effect_visit());
    assert_eq!(
        child.abort_code.load(std::sync::atomic::Ordering::Relaxed),
        ABORT_STACK_EFFECT_VISITS,
    );

    parent.fold_in_isolated_state(&child);

    assert_eq!(parent.stack_effect_visits(), 3);
    assert!(!parent.is_aborted());
}

#[test]
fn state_local_work_still_counts_towards_global_stack_effect_limit() {
    let mut parent = default_unit_inline_budget();
    parent.max_stack_effect_visits = 4;

    let first = parent.child_with_stack_effect_visit_limit(2);
    assert!(first.record_stack_effect_visit());
    assert!(first.record_stack_effect_visit());
    assert!(!first.record_stack_effect_visit());
    parent.fold_in_isolated_state(&first);
    assert!(!parent.is_aborted());

    let second = parent.child_with_stack_effect_visit_limit(2);
    assert!(second.record_stack_effect_visit());
    assert!(second.record_stack_effect_visit());
    parent.fold_in_isolated_state(&second);

    assert!(parent.is_aborted());
    assert_eq!(parent.report().reason, Some("stack_effect_visits"));
}

#[test]
fn core_merged_stack_effect_lowering_accepts_only_unguarded_effects() {
    let _guard = ENV_LOCK.lock().unwrap();
    let _env = EnvVarGuard::set(MAX_GUARDED_STACK_EFFECTS_ENV, "1000000");
    let mut table = table_with_stack_shifts(Vec::new(), &[]);
    table.construction = GlrTableConstruction::ExperimentalCoreMerged;

    let unguarded = vec![GuardedStackShift {
        guards: Vec::new(),
        pop: 2,
        pushes: vec![2],
    }];
    assert!(matches!(
        stack_effect_action(&table, unguarded),
        Some(Action::StackShifts(shifts)) if shifts.len() == 1
    ));

    assert!(stack_effect_action(&table, vec![guarded_effect(1, 1)]).is_none());
}

#[test]
fn legacy_stack_effect_lowering_still_accepts_guarded_effects() {
    let _guard = ENV_LOCK.lock().unwrap();
    let _env = EnvVarGuard::set(MAX_GUARDED_STACK_EFFECTS_ENV, "1000000");
    let table = table_with_stack_shifts(Vec::new(), &[]);

    assert!(matches!(
        stack_effect_action(&table, vec![guarded_effect(1, 1)]),
        Some(Action::GuardedStackShifts(shifts)) if shifts.len() == 1
    ));
}

#[test]
fn row_fingerprint_is_independent_of_sparse_row_insertion_order() {
    let mut action_left = ActionRow::default();
    action_left.insert(3, Action::Reduce(7, 1));
    action_left.insert(1, Action::Shift(4, false));
    action_left.insert(9, Action::Accept);
    let mut action_right = ActionRow::default();
    action_right.insert(9, Action::Accept);
    action_right.insert(1, Action::Shift(4, false));
    action_right.insert(3, Action::Reduce(7, 1));

    let mut goto_left = GotoRow::default();
    goto_left.insert(5, (2, false));
    goto_left.insert(1, (6, true));
    let mut goto_right = GotoRow::default();
    goto_right.insert(1, (6, true));
    goto_right.insert(5, (2, false));

    assert!(rows_equal(
        &action_left,
        &goto_left,
        None,
        &action_right,
        &goto_right,
        None,
    ));
    assert_eq!(
        row_fingerprint(&action_left, &goto_left, None),
        row_fingerprint(&action_right, &goto_right, None),
    );
}

#[test]
fn incremental_row_merge_matches_full_quotient_on_small_generated_tables() {
    fn next(seed: &mut u64) -> u64 {
        *seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *seed
    }

    fn action(seed: &mut u64, nstates: u32) -> Action {
        match next(seed) % 5 {
            0 => Action::Shift((next(seed) % nstates as u64) as u32, next(seed) & 1 != 0),
            1 => Action::Reduce((next(seed) % 3) as u32, (next(seed) % 3) as u32),
            2 => Action::Split {
                shift: (next(seed) & 1 != 0)
                    .then(|| ((next(seed) % nstates as u64) as u32, next(seed) & 1 != 0)),
                reduces: vec![((next(seed) % 3) as u32, (next(seed) % 3) as u32)],
                accept: next(seed) & 1 != 0,
            },
            3 => Action::StackShifts(vec![
                StackShift {
                    pop: (next(seed) % 3) as u32,
                    pushes: vec![(next(seed) % nstates as u64) as u32],
                },
                StackShift {
                    pop: (next(seed) % 3) as u32,
                    pushes: vec![(next(seed) % nstates as u64) as u32],
                },
            ]),
            _ => Action::Accept,
        }
    }

    let mut seed = 0xA3B1_C2D3_E4F5_6789u64;
    for _case in 0..128 {
        let nstates = 6usize;
        let mut action_rows = vec![ActionRow::default(); nstates];
        let mut goto_rows = vec![GotoRow::default(); nstates];
        for state in 0..nstates {
            for terminal in 0..3 {
                if next(&mut seed) % 4 != 0 {
                    action_rows[state].insert(terminal, action(&mut seed, nstates as u32));
                }
            }
            for nonterminal in 0..3 {
                if next(&mut seed) & 1 != 0 {
                    goto_rows[state].insert(
                        nonterminal,
                        ((next(&mut seed) % nstates as u64) as u32, next(&mut seed) & 1 != 0),
                    );
                }
            }
        }

        let mut base = GLRTable {
            action: action_rows,
            goto: goto_rows,
            num_states: nstates as u32,
            num_terminals: 3,
            num_rules: 0,
            rules: Vec::new(),
            nonterminal_display_names: Vec::new(),
            embedded_start: Default::default(),
            construction: GlrTableConstruction::LegacyRowBisim,
            admission_policy: AdmissionPolicy::RowPresenceExact,
            advance: Vec::new(),
            unconditional_advance: Vec::new(),
            forwarded_shifts: FxHashSet::default(),
            control_terminals: Default::default(),
            skip_terminals: Default::default(),
            guarded_shift_index: Vec::new(),
            direct_regular_wide_frontiers: Vec::new(),
        };
        base.merge_identical_rows();
        if base.num_states == 0 {
            continue;
        }

        let dirty = (next(&mut seed) % base.num_states as u64) as u32;
        let terminal = (next(&mut seed) % 3) as u32;
        base.action[dirty as usize].insert(terminal, action(&mut seed, base.num_states));

        let mut expected = base.clone();
        expected.merge_identical_rows();
        let mut incremental = base;
        incremental.merge_identical_rows_from_dirty(&[dirty]);

        assert_eq!(incremental.num_states, expected.num_states);
        assert_eq!(incremental.action, expected.action);
        assert_eq!(incremental.goto, expected.goto);
        assert_eq!(incremental.advance, expected.advance);
        assert_eq!(incremental.forwarded_shifts, expected.forwarded_shifts);
    }
}

#[test]
fn incremental_row_merge_matches_full_quotient_for_noncanonical_stack_shifts() {
    let mut action = vec![ActionRow::default(); 4];
    action[0].insert(
        0,
        Action::StackShifts(vec![
            StackShift {
                pop: 2,
                pushes: vec![2],
            },
            StackShift {
                pop: 1,
                pushes: vec![3],
            },
        ]),
    );
    // Initially distinct from state 0, so the starting table is already
    // fully exact-minimized before the local mutation below.
    action[1].insert(
        0,
        Action::StackShifts(vec![
            StackShift {
                pop: 1,
                pushes: vec![3],
            },
            StackShift {
                pop: 2,
                pushes: vec![3],
            },
        ]),
    );
    action[2].insert(0, Action::Shift(2, false));
    action[3].insert(0, Action::Accept);

    let table = GLRTable {
        action,
        goto: vec![GotoRow::default(); 4],
        num_states: 4,
        num_terminals: 1,
        num_rules: 0,
        rules: Vec::new(),
        nonterminal_display_names: Vec::new(),
        embedded_start: Default::default(),
        construction: GlrTableConstruction::LegacyRowBisim,
        admission_policy: AdmissionPolicy::RowPresenceExact,
        advance: Vec::new(),
        unconditional_advance: Vec::new(),
        forwarded_shifts: FxHashSet::default(),
        control_terminals: Default::default(),
        skip_terminals: Default::default(),
        guarded_shift_index: Vec::new(),
        direct_regular_wide_frontiers: Vec::new(),
    };

    let mut base = table.clone();
    base.merge_identical_rows();
    assert_eq!(base.num_states, 4);

    // Stack-effect equality remaps and normalizes alternatives. The two
    // rows are exact equals under the identity map, while their direct
    // Action hashes differ because their alternative order differs.
    base.action[1].insert(
        0,
        Action::StackShifts(vec![
            StackShift {
                pop: 1,
                pushes: vec![3],
            },
            StackShift {
                pop: 2,
                pushes: vec![2],
            },
        ]),
    );
    let identity = (0..base.num_states).collect::<Vec<_>>();
    assert!(rows_equal_after_remap(
        &base.action[0],
        &base.goto[0],
        None,
        &base.action[1],
        &base.goto[1],
        None,
        &identity,
    ));
    assert_ne!(
        row_fingerprint(&base.action[0], &base.goto[0], None),
        row_fingerprint(&base.action[1], &base.goto[1], None),
    );

    let mut expected = base.clone();
    expected.merge_identical_rows();
    assert_eq!(expected.num_states, 3);

    let mut incremental = base;
    assert!(incremental.merge_identical_rows_from_dirty(&[1]));
    assert_eq!(incremental.num_states, expected.num_states);
    assert_eq!(incremental.action, expected.action);
    assert_eq!(incremental.goto, expected.goto);
    assert_eq!(incremental.advance, expected.advance);
    assert_eq!(incremental.forwarded_shifts, expected.forwarded_shifts);
}

#[test]
fn unit_reduction_inlining_budget_abort_keeps_original_table() {
    let _guard = ENV_LOCK.lock().unwrap();
    let _env = EnvVarGuard::set(UNIT_INLINE_WORK_MAX_STACK_EFFECT_VISITS_ENV, "1");

    let mut action = vec![ActionRow::default(); 5];
    action[2].insert(
        0,
        Action::Split {
            shift: Some((4, false)),
            reduces: vec![(10, 1)],
            accept: false,
        },
    );
    action[3].insert(0, Action::Shift(4, false));

    let mut goto = vec![GotoRow::default(); 5];
    goto[1].insert(10, (3, true));

    let mut table = GLRTable {
        action,
        goto,
        num_states: 5,
        num_terminals: 1,
        num_rules: 0,
        rules: Vec::new(),
        nonterminal_display_names: Vec::new(),
        embedded_start: Default::default(),
        construction: GlrTableConstruction::LegacyRowBisim,
        admission_policy: AdmissionPolicy::RowPresenceExact,
        advance: Vec::new(),
        unconditional_advance: Vec::new(),
        forwarded_shifts: FxHashSet::default(),
        control_terminals: Default::default(),
        skip_terminals: Default::default(),
        guarded_shift_index: Vec::new(),
        direct_regular_wide_frontiers: Vec::new(),
    };
    let original_action = format!("{:?}", table.action);
    let original_goto = format!("{:?}", table.goto);
    let original_num_states = table.num_states;

    let report = table.collapse_sr_unit_reductions_with_compatible_gotos();

    assert!(report.aborted);
    assert_eq!(report.reason, Some("stack_effect_visits"));
    assert_eq!(table.num_states, original_num_states);
    assert_eq!(format!("{:?}", table.action), original_action);
    assert_eq!(format!("{:?}", table.goto), original_goto);
}

#[test]
fn canonicalizes_stack_shift_predecessor_to_goto_superset() {
    let mut table = table_with_stack_shifts(
        vec![
            StackShift {
                pop: 1,
                pushes: vec![1, 3, 4],
            },
            StackShift {
                pop: 1,
                pushes: vec![2, 3, 4],
            },
        ],
        &[
            (1, &[(10, (20, true)), (11, (21, false))]),
            (2, &[(10, (20, true))]),
        ],
    );

    table.canonicalize_stack_shift_predecessors();

    assert_eq!(
        stack_shifts_at_start(&table),
        vec![StackShift {
            pop: 1,
            pushes: vec![1, 3, 4],
        }]
    );
}

#[test]
fn leaves_protected_stack_shift_terminal_unchanged() {
    let mut table = table_with_stack_shifts(
        vec![
            StackShift {
                pop: 1,
                pushes: vec![1, 3, 4],
            },
            StackShift {
                pop: 1,
                pushes: vec![2, 3, 4],
            },
        ],
        &[
            (1, &[(10, (20, true)), (11, (21, false))]),
            (2, &[(10, (20, true))]),
        ],
    );
    let mut protected = BitSet::new(table.num_terminals as usize);
    protected.set(0);

    table.canonicalize_stack_shift_predecessors_except(&protected);

    assert_eq!(
        stack_shifts_at_start(&table),
        vec![
            StackShift {
                pop: 1,
                pushes: vec![1, 3, 4],
            },
            StackShift {
                pop: 1,
                pushes: vec![2, 3, 4],
            },
        ]
    );
}

#[test]
fn leaves_stack_shift_predecessors_unchanged_when_canonicalization_is_disabled() {
    let mut table = table_with_stack_shifts(
        vec![
            StackShift {
                pop: 1,
                pushes: vec![1, 3, 4],
            },
            StackShift {
                pop: 1,
                pushes: vec![2, 3, 4],
            },
        ],
        &[
            (1, &[(10, (20, true)), (11, (21, false))]),
            (2, &[(10, (20, true))]),
        ],
    );

    table.canonicalize_stack_shift_predecessors_with_enabled(
        false,
        &BitSet::new(table.num_terminals as usize),
    );

    assert_eq!(
        stack_shifts_at_start(&table),
        vec![
            StackShift {
                pop: 1,
                pushes: vec![1, 3, 4],
            },
            StackShift {
                pop: 1,
                pushes: vec![2, 3, 4],
            },
        ]
    );
}

#[test]
fn does_not_canonicalize_stack_shift_predecessors_when_shared_goto_target_differs() {
    let mut table = table_with_stack_shifts(
        vec![
            StackShift {
                pop: 1,
                pushes: vec![1, 3, 4],
            },
            StackShift {
                pop: 1,
                pushes: vec![2, 3, 4],
            },
        ],
        &[
            (1, &[(10, (20, true)), (11, (21, false))]),
            (2, &[(10, (22, true))]),
        ],
    );

    table.canonicalize_stack_shift_predecessors();

    assert_eq!(
        stack_shifts_at_start(&table),
        vec![
            StackShift {
                pop: 1,
                pushes: vec![1, 3, 4],
            },
            StackShift {
                pop: 1,
                pushes: vec![2, 3, 4],
            },
        ]
    );
}

#[test]
fn does_not_canonicalize_empty_goto_row_to_nonempty_superset() {
    let mut table = table_with_stack_shifts(
        vec![
            StackShift {
                pop: 1,
                pushes: vec![1, 3, 4],
            },
            StackShift {
                pop: 1,
                pushes: vec![2, 3, 4],
            },
        ],
        &[(1, &[(10, (20, true))])],
    );

    table.canonicalize_stack_shift_predecessors();

    assert_eq!(
        stack_shifts_at_start(&table),
        vec![
            StackShift {
                pop: 1,
                pushes: vec![1, 3, 4],
            },
            StackShift {
                pop: 1,
                pushes: vec![2, 3, 4],
            },
        ]
    );
}

#[test]
fn canonicalizes_buried_middle_stack_shift_predecessor_to_goto_superset() {
    let mut table = table_with_stack_shifts(
        vec![
            StackShift {
                pop: 1,
                pushes: vec![9, 1, 3, 4],
            },
            StackShift {
                pop: 1,
                pushes: vec![9, 2, 3, 4],
            },
        ],
        &[
            (1, &[(10, (20, true)), (11, (21, false))]),
            (2, &[(10, (20, true))]),
        ],
    );

    table.canonicalize_stack_shift_predecessors();

    assert_eq!(
        stack_shifts_at_start(&table),
        vec![StackShift {
            pop: 1,
            pushes: vec![9, 1, 3, 4],
        }]
    );
}

#[test]
fn does_not_canonicalize_top_pushed_state_even_when_goto_rows_are_compatible() {
    let mut table = table_with_stack_shifts(
        vec![
            StackShift {
                pop: 1,
                pushes: vec![9, 3, 1],
            },
            StackShift {
                pop: 1,
                pushes: vec![9, 3, 2],
            },
        ],
        &[
            (1, &[(10, (20, true)), (11, (21, false))]),
            (2, &[(10, (20, true))]),
        ],
    );

    table.canonicalize_stack_shift_predecessors();

    assert_eq!(
        stack_shifts_at_start(&table),
        vec![
            StackShift {
                pop: 1,
                pushes: vec![9, 3, 1],
            },
            StackShift {
                pop: 1,
                pushes: vec![9, 3, 2],
            },
        ]
    );
}

#[test]
fn reduce_frame_allows_origin_dependent_multiple_goto_targets() {
    let mut table = table_with_stack_shifts(Vec::new(), &[
        (1, &[(10, (3, false))]),
        (2, &[(10, (4, false))]),
    ]);
    table.num_states = 6;
    table.action.resize(6, ActionRow::default());
    table.goto.resize(6, GotoRow::default());

    let mut predecessors = vec![PredecessorSet::new(); 6];
    predecessors[5] = smallvec![1, 2];
    let budget = default_unit_inline_budget();

    let result = apply_reduce_to_frame(
        &table,
        &predecessors,
        5,
        StackEffectFrame {
            pop: 0,
            pushes: Vec::new(),
            guards: Vec::new(),
        },
        10,
        1,
        &mut FxHashMap::default(),
        &budget,
    );

    let Some(ReduceFrameResult::Frames { frames, origin_dependent }) = result else {
        panic!("expected frames");
    };
    assert!(origin_dependent);
    assert_eq!(
        frames,
        vec![
            StackEffectFrame {
                pop: 1,
                pushes: vec![3],
                guards: vec![StackShiftGuard {
                    pop: 1,
                    states: vec![1],
                }],
            },
            StackEffectFrame {
                pop: 1,
                pushes: vec![4],
                guards: vec![StackShiftGuard {
                    pop: 1,
                    states: vec![2],
                }],
            },
        ]
    );
}

#[test]
fn reduce_frame_allows_origin_dependent_single_goto_target() {
    let mut table = table_with_stack_shifts(Vec::new(), &[
        (1, &[(10, (3, false))]),
        (2, &[(10, (3, false))]),
    ]);
    table.num_states = 6;
    table.action.resize(6, ActionRow::default());
    table.goto.resize(6, GotoRow::default());

    let mut predecessors = vec![PredecessorSet::new(); 6];
    predecessors[5] = smallvec![1, 2];
    let budget = default_unit_inline_budget();

    let result = apply_reduce_to_frame(
        &table,
        &predecessors,
        5,
        StackEffectFrame {
            pop: 0,
            pushes: Vec::new(),
            guards: Vec::new(),
        },
        10,
        1,
        &mut FxHashMap::default(),
        &budget,
    );

    let Some(ReduceFrameResult::Frames { frames, origin_dependent }) = result else {
        panic!("expected frames");
    };
    assert!(origin_dependent);
    assert_eq!(
        frames,
        vec![
            StackEffectFrame {
                pop: 1,
                pushes: vec![3],
                guards: Vec::new(),
            }
        ]
    );
}

#[test]
fn in_place_action_target_remap_matches_copy_remap() {
    let mapping = vec![3, 0, 2, 1, 4, 5];
    let actions = vec![
        Action::Shift(1, false),
        Action::Reduce(7, 2),
        Action::Split {
            shift: Some((3, true)),
            reduces: vec![(4, 1), (5, 2)],
            accept: false,
        },
        Action::StackShifts(vec![
            StackShift {
                pop: 1,
                pushes: vec![0, 3],
            },
            StackShift {
                pop: 1,
                pushes: vec![0, 1],
            },
        ]),
        Action::GuardedStackShifts(vec![GuardedStackShift {
            guards: vec![StackShiftGuard {
                pop: 1,
                states: vec![0, 3],
            }],
            pop: 2,
            pushes: vec![1, 4],
        }]),
        Action::Accept,
    ];
    for action in actions {
        let expected = remap_action_targets(&action, &mapping);
        let mut actual = action;
        remap_action_targets_in_place(&mut actual, &mapping);
        assert_eq!(actual, expected);
    }
}

#[test]
fn trivial_origin_dependent_one_push_shortcut_matches_generic_refusal() {
    let mut action = vec![ActionRow::default(); 6];
    action[3].insert(0, Action::Shift(5, true));
    action[4].insert(0, Action::Shift(5, true));
    let mut goto = vec![GotoRow::default(); 6];
    goto[1].insert(10, (3, false));
    goto[2].insert(10, (4, false));
    let table = GLRTable {
        action,
        goto,
        num_states: 6,
        num_terminals: 1,
        num_rules: 0,
        rules: Vec::new(),
        nonterminal_display_names: Vec::new(),
        embedded_start: Default::default(),
        construction: GlrTableConstruction::LegacyRowBisim,
        admission_policy: AdmissionPolicy::RowPresenceExact,
        advance: Vec::new(),
        unconditional_advance: Vec::new(),
        forwarded_shifts: FxHashSet::default(),
        control_terminals: Default::default(),
        skip_terminals: Default::default(),
        guarded_shift_index: Vec::new(),
        direct_regular_wide_frontiers: Vec::new(),
    };
    let mut predecessors = vec![PredecessorSet::new(); 6];
    predecessors[0] = smallvec![1, 2];
    let reduce = Action::Reduce(10, 1);
    let mut shortcut_reads = Vec::new();
    assert!(is_trivial_origin_dependent_one_push(
        &table,
        &predecessors,
        0,
        0,
        &reduce,
        &mut shortcut_reads,
    ));
    assert_eq!(shortcut_reads, vec![((0, 0), (3, 0)), ((0, 0), (4, 0))]);

    let budget = default_unit_inline_budget();
    let generic = try_inline_action_to_stack_shifts_detailed(
        &table,
        &predecessors,
        0,
        0,
        &reduce,
        &mut FxHashMap::default(),
        &mut Vec::new(),
        &budget,
    );
    assert!(generic.action.is_none());
    assert!(matches!(
        generic.outcome,
        StackInlineOutcomeKind::OriginDependentOnePush
    ));
}

#[test]
fn trivial_origin_dependent_one_push_shortcut_rejects_nonreplace_continuation() {
    let mut action = vec![ActionRow::default(); 4];
    action[3].insert(0, Action::Shift(3, false));
    let mut goto = vec![GotoRow::default(); 4];
    goto[1].insert(10, (3, false));
    let table = GLRTable {
        action,
        goto,
        num_states: 4,
        num_terminals: 1,
        num_rules: 0,
        rules: Vec::new(),
        nonterminal_display_names: Vec::new(),
        embedded_start: Default::default(),
        construction: GlrTableConstruction::LegacyRowBisim,
        admission_policy: AdmissionPolicy::RowPresenceExact,
        advance: Vec::new(),
        unconditional_advance: Vec::new(),
        forwarded_shifts: FxHashSet::default(),
        control_terminals: Default::default(),
        skip_terminals: Default::default(),
        guarded_shift_index: Vec::new(),
        direct_regular_wide_frontiers: Vec::new(),
    };
    let mut predecessors = vec![PredecessorSet::new(); 4];
    predecessors[0].push(1);
    assert!(!is_trivial_origin_dependent_one_push(
        &table,
        &predecessors,
        0,
        0,
        &Action::Reduce(10, 1),
        &mut Vec::new(),
    ));
}

#[test]
fn inline_action_to_stack_shifts_keeps_multishift_replacement_reduce_chain() {
    let mut action = vec![ActionRow::default(); 5];
    action[2].insert(
        0,
        Action::Split {
            shift: Some((4, false)),
            reduces: vec![(10, 1)],
            accept: false,
        },
    );
    action[3].insert(0, Action::Shift(4, false));

    let mut goto = vec![GotoRow::default(); 5];
    goto[1].insert(10, (3, true));

    let table = GLRTable {
        action,
        goto,
        num_states: 5,
        num_terminals: 1,
        num_rules: 0,
        rules: Vec::new(),
        nonterminal_display_names: Vec::new(),
        embedded_start: Default::default(),
        construction: GlrTableConstruction::LegacyRowBisim,
        admission_policy: AdmissionPolicy::RowPresenceExact,
        advance: Vec::new(),
        unconditional_advance: Vec::new(),
        forwarded_shifts: FxHashSet::default(),
        control_terminals: Default::default(),
        skip_terminals: Default::default(),
        guarded_shift_index: Vec::new(),
        direct_regular_wide_frontiers: Vec::new(),
    };
    let mut predecessors = vec![PredecessorSet::new(); 5];
    predecessors[2].push(1);
    let budget = default_unit_inline_budget();

    let action = table.action(2, 0).expect("expected split action");
    let result = try_inline_action_to_stack_shifts(
        &table,
        &predecessors,
        2,
        0,
        action,
        &mut FxHashMap::default(),
        &mut Vec::new(),
        &budget,
    );

    let Some(Action::StackShifts(shifts)) = result else {
        panic!("expected multi-stack-shift action, got {result:?}");
    };
    assert_eq!(
        shifts,
        vec![
            StackShift {
                pop: 0,
                pushes: vec![4],
            },
            StackShift {
                pop: 2,
                pushes: vec![3, 4],
            },
        ]
    );
}

#[test]
fn inline_action_to_stack_shifts_handles_replace_shift_and_replace_goto() {
    let mut action = vec![ActionRow::default(); 6];
    action[2].insert(
        0,
        Action::Split {
            shift: Some((4, true)),
            reduces: vec![(10, 1)],
            accept: false,
        },
    );
    action[3].insert(0, Action::Shift(5, true));

    let mut goto = vec![GotoRow::default(); 6];
    goto[1].insert(10, (3, true));

    let table = GLRTable {
        action,
        goto,
        num_states: 6,
        num_terminals: 1,
        num_rules: 0,
        rules: Vec::new(),
        nonterminal_display_names: Vec::new(),
        embedded_start: Default::default(),
        construction: GlrTableConstruction::LegacyRowBisim,
        admission_policy: AdmissionPolicy::RowPresenceExact,
        advance: Vec::new(),
        unconditional_advance: Vec::new(),
        forwarded_shifts: FxHashSet::default(),
        control_terminals: Default::default(),
        skip_terminals: Default::default(),
        guarded_shift_index: Vec::new(),
        direct_regular_wide_frontiers: Vec::new(),
    };
    let mut predecessors = vec![PredecessorSet::new(); 6];
    predecessors[2].push(1);
    let budget = default_unit_inline_budget();

    let action = table.action(2, 0).expect("expected split action");
    let result = try_inline_action_to_stack_shifts(
        &table,
        &predecessors,
        2,
        0,
        action,
        &mut FxHashMap::default(),
        &mut Vec::new(),
        &budget,
    );

    let Some(Action::StackShifts(shifts)) = result else {
        panic!("expected replacement stack shifts, got {result:?}");
    };
    assert_eq!(
        shifts,
        vec![
            StackShift {
                pop: 1,
                pushes: vec![4],
            },
            StackShift {
                pop: 2,
                pushes: vec![5],
            },
        ]
    );
}

#[test]
fn inline_action_to_stack_shifts_guards_divergent_replace_gotos_by_predecessor() {
    let mut action = vec![ActionRow::default(); 9];
    action[2].insert(0, Action::Reduce(10, 1));
    action[3].insert(0, Action::Shift(7, false));
    action[4].insert(0, Action::Shift(8, false));

    let mut goto = vec![GotoRow::default(); 9];
    goto[1].insert(10, (3, true));
    goto[6].insert(10, (4, true));

    let table = GLRTable {
        action,
        goto,
        num_states: 9,
        num_terminals: 1,
        num_rules: 0,
        rules: Vec::new(),
        nonterminal_display_names: Vec::new(),
        embedded_start: Default::default(),
        construction: GlrTableConstruction::LegacyRowBisim,
        admission_policy: AdmissionPolicy::RowPresenceExact,
        advance: Vec::new(),
        unconditional_advance: Vec::new(),
        forwarded_shifts: FxHashSet::default(),
        control_terminals: Default::default(),
        skip_terminals: Default::default(),
        guarded_shift_index: Vec::new(),
        direct_regular_wide_frontiers: Vec::new(),
    };
    let mut predecessors = vec![PredecessorSet::new(); 9];
    predecessors[2].extend_from_slice(&[1, 6]);
    let budget = default_unit_inline_budget();

    let action = table.action(2, 0).expect("expected reduce action");
    let result = try_inline_action_to_stack_shifts(
        &table,
        &predecessors,
        2,
        0,
        action,
        &mut FxHashMap::default(),
        &mut Vec::new(),
        &budget,
    );

    let Some(Action::GuardedStackShifts(shifts)) = result else {
        panic!("expected guarded replacement stack shifts, got {result:?}");
    };
    assert_eq!(
        shifts,
        vec![
            GuardedStackShift {
                guards: vec![StackShiftGuard {
                    pop: 1,
                    states: vec![1],
                }],
                pop: 2,
                pushes: vec![3, 7],
            },
            GuardedStackShift {
                guards: vec![StackShiftGuard {
                    pop: 1,
                    states: vec![6],
                }],
                pop: 2,
                pushes: vec![4, 8],
            },
        ]
    );
}

#[test]
fn unit_reduce_destination_without_predecessors_is_not_inlineable() {
    let table = GLRTable {
        action: vec![ActionRow::default(); 2],
        goto: vec![GotoRow::default(); 2],
        num_states: 2,
        num_terminals: 1,
        num_rules: 0,
        rules: Vec::new(),
        nonterminal_display_names: Vec::new(),
        embedded_start: Default::default(),
        construction: GlrTableConstruction::ExperimentalCoreMerged,
        admission_policy: AdmissionPolicy::RowPresenceExact,
        advance: Vec::new(),
        unconditional_advance: Vec::new(),
        forwarded_shifts: FxHashSet::default(),
        control_terminals: Default::default(),
        skip_terminals: Default::default(),
        guarded_shift_index: Vec::new(),
        direct_regular_wide_frontiers: Vec::new(),
    };
    let predecessors = vec![PredecessorSet::new(); 2];

    assert_eq!(unit_reduce_destination(&table, &predecessors, 1, 10), None);
}

#[test]
fn compatible_goto_unit_destination_still_refuses_replace_goto() {
    let action = vec![ActionRow::default(); 4];
    let mut goto = vec![GotoRow::default(); 4];
    goto[1].insert(10, (3, true));

    let table = GLRTable {
        action,
        goto,
        num_states: 4,
        num_terminals: 1,
        num_rules: 0,
        rules: Vec::new(),
        nonterminal_display_names: Vec::new(),
        embedded_start: Default::default(),
        construction: GlrTableConstruction::LegacyRowBisim,
        admission_policy: AdmissionPolicy::RowPresenceExact,
        advance: Vec::new(),
        unconditional_advance: Vec::new(),
        forwarded_shifts: FxHashSet::default(),
        control_terminals: Default::default(),
        skip_terminals: Default::default(),
        guarded_shift_index: Vec::new(),
        direct_regular_wide_frontiers: Vec::new(),
    };
    let mut predecessors = vec![PredecessorSet::new(); 4];
    predecessors[2].push(1);

    assert_eq!(unit_reduce_destination(&table, &predecessors, 2, 10), None);
}

#[test]
fn suffix_quotient_collapses_same_pop_stack_shift_fanout() {
    let token0 = 0;
    let token1 = 1;
    let mut action = vec![ActionRow::default(); 8];
    action[0].insert(
        token0,
        Action::StackShifts(vec![
            StackShift {
                pop: 1,
                pushes: vec![1, 2],
            },
            StackShift {
                pop: 1,
                pushes: vec![3, 4],
            },
        ]),
    );
    action[2].insert(
        token1,
        Action::StackShifts(vec![
            StackShift {
                pop: 1,
                pushes: vec![5],
            },
            StackShift {
                pop: 2,
                pushes: vec![6],
            },
        ]),
    );
    action[4].insert(
        token1,
        Action::StackShifts(vec![StackShift {
            pop: 2,
            pushes: vec![7],
        }]),
    );

    let mut table = GLRTable {
        action,
        goto: vec![GotoRow::default(); 8],
        num_states: 8,
        num_terminals: 2,
        num_rules: 0,
        rules: Vec::new(),
        nonterminal_display_names: Vec::new(),
        embedded_start: Default::default(),
        construction: GlrTableConstruction::LegacyRowBisim,
        admission_policy: AdmissionPolicy::RowPresenceExact,
        advance: Vec::new(),
        unconditional_advance: Vec::new(),
        forwarded_shifts: FxHashSet::default(),
        control_terminals: Default::default(),
        skip_terminals: Default::default(),
        guarded_shift_index: Vec::new(),
        direct_regular_wide_frontiers: Vec::new(),
    };
    table.rebuild_advance_rows_from_actions();

    table.quotient_recognizer_stack_suffixes();

    assert!(matches!(table.action(0, token0), Some(Action::Shift(_, true))));
    assert!(
        table.ambiguous_actions().is_empty(),
        "{:#?}",
        table.ambiguous_actions()
    );
}


#[test]
fn suffix_quotient_builds_synthetic_rows_through_reductions() {
    let produce = 0;
    let consume = 1;
    let mut action = vec![ActionRow::default(); 9];
    action[0].insert(
        produce,
        Action::StackShifts(vec![
            StackShift { pop: 0, pushes: vec![1] },
            StackShift { pop: 0, pushes: vec![2] },
        ]),
    );
    action[1].insert(consume, Action::Reduce(10, 1));
    action[2].insert(consume, Action::Reduce(11, 1));
    action[3].insert(consume, Action::Shift(7, false));
    action[4].insert(consume, Action::Shift(8, false));

    let mut goto = vec![GotoRow::default(); 9];
    goto[0].insert(10, (3, false));
    goto[0].insert(11, (4, false));

    let mut table = GLRTable {
        action,
        goto,
        num_states: 9,
        num_terminals: 2,
        num_rules: 0,
        rules: Vec::new(),
        nonterminal_display_names: Vec::new(),
        embedded_start: Default::default(),
        construction: GlrTableConstruction::LegacyRowBisim,
        admission_policy: AdmissionPolicy::RowPresenceExact,
        advance: Vec::new(),
        unconditional_advance: Vec::new(),
        forwarded_shifts: FxHashSet::default(),
        control_terminals: Default::default(),
        skip_terminals: Default::default(),
        guarded_shift_index: Vec::new(),
        direct_regular_wide_frontiers: Vec::new(),
    };
    table.rebuild_advance_rows_from_actions();

    table.quotient_recognizer_stack_suffixes();

    let synthetic = match table.action(0, produce) {
        Some(Action::StackShifts(producer_shifts)) => {
            assert_eq!(producer_shifts.len(), 1);
            assert_eq!(producer_shifts[0].pop, 0);
            assert_eq!(producer_shifts[0].pushes.len(), 1);
            producer_shifts[0].pushes[0]
        }
        Some(Action::Shift(target, replace)) => {
            assert!(!replace);
            *target
        }
        other => panic!("expected producer to be rewritten to one target action, got {other:?}"),
    };
    let Some(Action::GuardedStackShifts(shifts)) = table.action(synthetic, consume) else {
        panic!("expected synthetic row to compile reductions into guarded stack effects");
    };
    assert!(!shifts.is_empty());
    assert!(shifts.iter().all(|shift| shift.pop == 1));
    assert!(shifts.iter().all(|shift| shift.guards.len() == 1));
    assert!(shifts.iter().all(|shift| shift.guards[0].pop == 1));
    assert!(shifts.iter().all(|shift| shift.guards[0].states == vec![0]));
}

#[test]
fn suffix_quotient_preserves_guarded_stack_shift_guards() {
    let token = 0;
    let guard = StackShiftGuard {
        pop: 1,
        states: vec![9],
    };
    let mut action = vec![ActionRow::default(); 12];
    action[0].insert(
        token,
        Action::GuardedStackShifts(vec![
            GuardedStackShift {
                guards: vec![guard.clone()],
                pop: 1,
                pushes: vec![1, 2],
            },
            GuardedStackShift {
                guards: vec![guard.clone()],
                pop: 1,
                pushes: vec![3, 4],
            },
        ]),
    );

    let mut table = GLRTable {
        action,
        goto: vec![GotoRow::default(); 12],
        num_states: 12,
        num_terminals: 1,
        num_rules: 0,
        rules: Vec::new(),
        nonterminal_display_names: Vec::new(),
        embedded_start: Default::default(),
        construction: GlrTableConstruction::LegacyRowBisim,
        admission_policy: AdmissionPolicy::RowPresenceExact,
        advance: Vec::new(),
        unconditional_advance: Vec::new(),
        forwarded_shifts: FxHashSet::default(),
        control_terminals: Default::default(),
        skip_terminals: Default::default(),
        guarded_shift_index: Vec::new(),
        direct_regular_wide_frontiers: Vec::new(),
    };
    table.rebuild_advance_rows_from_actions();

    table.quotient_recognizer_stack_suffixes();

    let Some(Action::GuardedStackShifts(shifts)) = table.action(0, token) else {
        panic!("expected one guarded stack-shift action");
    };
    assert_eq!(shifts.len(), 1);
    assert_eq!(shifts[0].guards.len(), 1);
    assert_eq!(shifts[0].guards[0].pop, guard.pop);
    assert!(!shifts[0].guards[0].states.is_empty());
    assert_eq!(shifts[0].pop, 1);
    assert_eq!(shifts[0].pushes.len(), 1);
    assert!(
        table.ambiguous_actions().is_empty(),
        "{:#?}",
        table.ambiguous_actions()
    );
}

#[test]
fn suffix_quotient_rolls_back_when_nested_construction_hits_depth_bound() {
    let outer_suffixes = vec![vec![10, 1], vec![10, 2]];

    let mut table = GLRTable {
        action: vec![ActionRow::default(); 11],
        goto: vec![GotoRow::default(); 11],
        num_states: 11,
        num_terminals: 0,
        num_rules: 0,
        rules: Vec::new(),
        nonterminal_display_names: Vec::new(),
        embedded_start: Default::default(),
        construction: GlrTableConstruction::LegacyRowBisim,
        admission_policy: AdmissionPolicy::RowPresenceExact,
        advance: Vec::new(),
        unconditional_advance: Vec::new(),
        forwarded_shifts: FxHashSet::default(),
        control_terminals: Default::default(),
        skip_terminals: Default::default(),
        guarded_shift_index: Vec::new(),
        direct_regular_wide_frontiers: Vec::new(),
    };
    table.goto[1].insert(0, (3, false));
    table.goto[2].insert(0, (4, false));
    table.rebuild_advance_rows_from_actions();

    let original_num_states = table.num_states;
    let original_action = table.action.clone();
    let original_goto = table.goto.clone();
    let original_advance = table.advance.clone();

    let mut quotient = SuffixQuotient {
        suffix_to_state: FxHashMap::default(),
        failed_suffixes: FxHashSet::default(),
        max_states: 4096,
        max_alts: 8,
        max_width: 8,
        max_depth: 1,
        build_depth: 0,
        created_states: 0,
    };

    assert_eq!(
        quotient.ensure_suffix_state(&mut table, outer_suffixes.clone()),
        Err(())
    );
    assert_eq!(table.num_states, original_num_states);
    assert_eq!(table.action, original_action);
    assert_eq!(table.goto, original_goto);
    assert_eq!(table.advance, original_advance);
    assert_eq!(quotient.created_states, 0);
    assert_eq!(quotient.build_depth, 0);
    assert!(quotient.failed_suffixes.contains(&outer_suffixes));
    assert!(quotient
        .suffix_to_state
        .values()
        .all(|&state| state < original_num_states));
}

#[test]
fn suffix_quotient_rolls_back_nested_created_states_on_outer_failure() {
    let outer_suffixes = vec![vec![10, 1], vec![10, 2]];

    let mut table = GLRTable {
        action: vec![ActionRow::default(); 11],
        goto: vec![GotoRow::default(); 11],
        num_states: 11,
        num_terminals: 0,
        num_rules: 0,
        rules: Vec::new(),
        nonterminal_display_names: Vec::new(),
        embedded_start: Default::default(),
        construction: GlrTableConstruction::LegacyRowBisim,
        admission_policy: AdmissionPolicy::RowPresenceExact,
        advance: Vec::new(),
        unconditional_advance: Vec::new(),
        forwarded_shifts: FxHashSet::default(),
        control_terminals: Default::default(),
        skip_terminals: Default::default(),
        guarded_shift_index: Vec::new(),
        direct_regular_wide_frontiers: Vec::new(),
    };
    table.goto[1].insert(0, (3, false));
    table.goto[2].insert(0, (4, false));
    table.goto[1].insert(1, (5, false));
    table.goto[2].insert(1, (6, false));
    table.rebuild_advance_rows_from_actions();

    let original_num_states = table.num_states;
    let original_action_len = table.action.len();
    let original_goto_len = table.goto.len();
    let original_advance_len = table.advance.len();

    let mut quotient = SuffixQuotient {
        suffix_to_state: FxHashMap::default(),
        failed_suffixes: FxHashSet::default(),
        max_states: 2,
        max_alts: 8,
        max_width: 8,
        max_depth: 128,
        build_depth: 0,
        created_states: 0,
    };

    assert_eq!(
        quotient.ensure_suffix_state(&mut table, outer_suffixes.clone()),
        Err(())
    );
    assert_eq!(table.num_states, original_num_states);
    assert_eq!(table.action.len(), original_action_len);
    assert_eq!(table.goto.len(), original_goto_len);
    assert_eq!(table.advance.len(), original_advance_len);
    assert_eq!(quotient.created_states, 0);
    assert!(quotient.failed_suffixes.contains(&outer_suffixes));
    assert!(
        quotient
            .suffix_to_state
            .values()
            .all(|&state| state < original_num_states)
    );
}
