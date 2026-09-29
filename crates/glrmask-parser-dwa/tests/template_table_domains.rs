//! Exact relation-domain differential checks against the built-in LR engine.
//! Unlike viable-prefix-only tests, these include wrong predecessor stacks and
//! insufficient stack depths, where row-presence admission is not sufficient.
#![cfg(feature = "internal-api")]

use glrmask_glr::__private::glr::{
    accumulator::TerminalsDisallowed,
    parser::{ParserGSS, advance_stacks, stack_may_advance_on},
    table::{Action, AdmissionPolicy, GLRTable, GuardedStackShift, StackShift, StackShiftGuard, testing::build_test_table},
};
use glrmask_parser_dwa::__private::templates::{
    admissibility::TemplateDomain,
    characterize::characterize_selected_terminals_for_terminal_count,
    compile_dfa::{Templates, specialize_template_dfa_defaults_for_commit_split_input, try_split_commit_template_dfas},
};

fn domain(table: &GLRTable) -> TemplateDomain {
    let chars = characterize_selected_terminals_for_terminal_count(table, 1, &[true]);
    let raw = Templates::from_characterizations(&chars).by_terminal.remove(&0).unwrap();
    let specialized = specialize_template_dfa_defaults_for_commit_split_input(&raw);
    let split = try_split_commit_template_dfas(&specialized).expect("bounded acyclic relation must split");
    TemplateDomain::compile(&split).unwrap()
}

fn stacks(alphabet: u32, max_depth: usize) -> Vec<Vec<u32>> {
    let mut all = vec![Vec::new()]; let mut layer = vec![Vec::new()];
    for _ in 0..max_depth {
        let mut next = Vec::new();
        for stack in layer {
            for top in 0..alphabet { let mut s = stack.clone(); s.push(top); next.push(s); }
        }
        all.extend(next.iter().cloned()); layer = next;
    }
    all
}

#[test]
fn acyclic_table_domains_match_exact_admission_and_advance_for_all_small_stacks() {
    let all = stacks(8, 4); // 4681 complete bottom-first stacks.
    for len in [0, 1, 2] {
        for replace in [false, true] {
            let guarded = Action::GuardedStackShifts(vec![
                GuardedStackShift { pop: 1, pushes: vec![7], guards: vec![StackShiftGuard { pop: 1, states: vec![0, 2, 4] }] },
                GuardedStackShift { pop: 2, pushes: vec![6, 7], guards: vec![StackShiftGuard { pop: 1, states: vec![1, 3] }, StackShiftGuard { pop: 2, states: vec![0] }] },
            ]);
            let actions = [
                vec![], vec![(0, Action::Reduce(0, len))],
                vec![(0, Action::Split { shift: Some((7, replace)), reduces: vec![(0,len),(1,1)], accept: false })],
                vec![(0, Action::StackShifts(vec![StackShift { pop: 2, pushes: vec![7] },StackShift { pop: 1, pushes: vec![6,7] }]))],
                vec![(0, guarded.clone())],
                vec![(0, Action::ReplaceShifts(vec![6,7].into()))],
                vec![(0, Action::Skip)], vec![(0, Action::Shift(7, replace))],
            ];
            // Every goto exits to a consuming endpoint, so reduction closure
            // cannot loop even on stacks not reachable from the grammar start.
            let gotos = [
                vec![(0,(6,replace)),(1,(4,false))], vec![(0,(7,replace))],
                vec![(0,(6,replace)),(1,(4,replace))], vec![],
                vec![(0,(7,replace))], vec![], vec![], vec![],
            ];
            let mut table = build_test_table(8,1,&actions.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                &gotos.iter().map(Vec::as_slice).collect::<Vec<_>>());
            table.admission_policy = AdmissionPolicy::ExactSimulation;
            let compiled = domain(&table);
            let wire = compiled.to_bytes().unwrap(); let compiled = TemplateDomain::from_bytes(&wire).unwrap();
            for stack in &all {
                let gss = ParserGSS::from_single_stack(stack.clone(),TerminalsDisallowed::new());
                let expected = !advance_stacks(&table,&gss,0).is_empty();
                assert_eq!(stack_may_advance_on(&table,&gss,0),expected,
                    "LR self-check len={len} replace={replace} stack={stack:?}");
                assert_eq!(compiled.matches_top_first(stack.iter().rev().copied()),expected,
                    "template len={len} replace={replace} stack={stack:?}");
            }
        }
    }
}

#[test]
fn default_specialization_retains_feasible_guarded_alternatives_after_reduction() {
    let actions = [
        vec![], vec![(0,Action::Reduce(0,2))], vec![],
        vec![(0,Action::GuardedStackShifts(vec![
            GuardedStackShift { guards: vec![StackShiftGuard { pop:1,states:vec![0]}], pop:1,pushes:vec![4]},
            GuardedStackShift { guards: vec![StackShiftGuard { pop:2,states:vec![2]}], pop:2,pushes:vec![5]},
        ]))],vec![],vec![],
    ];
    let gotos = [vec![(0,(3,false))],vec![],vec![(0,(3,true))],vec![],vec![],vec![]];
    let mut table = build_test_table(6,1,&actions.iter().map(Vec::as_slice).collect::<Vec<_>>(),
        &gotos.iter().map(Vec::as_slice).collect::<Vec<_>>());
    table.admission_policy=AdmissionPolicy::ExactSimulation;
    let compiled=domain(&table);
    for stack in stacks(6,4) {
        let gss=ParserGSS::from_single_stack(stack.clone(),TerminalsDisallowed::new());
        assert_eq!(compiled.matches_top_first(stack.iter().rev().copied()),!advance_stacks(&table,&gss,0).is_empty(),
            "stack={stack:?}");
    }
}
