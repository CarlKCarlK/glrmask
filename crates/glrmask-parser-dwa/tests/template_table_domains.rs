//! Exact relation-domain differential checks against the built-in LR engine.
//! Unlike viable-prefix-only tests, these include wrong predecessor stacks and
//! insufficient stack depths, where row-presence admission is not sufficient.
#![cfg(feature = "internal-api")]

use glrmask_artifact::CommitTemplateDfas;
use std::collections::BTreeSet;
use glrmask_glr::__private::glr::labels::{DEFAULT_LABEL, negative_to_positive_label};
use glrmask_glr::__private::glr::{
    accumulator::TerminalsDisallowed,
    parser::{ParserGSS, advance_stacks, advance_stacks_owned, advance_stacks_profiled, stack_may_advance_on},
    table::{Action, AdmissionPolicy, GLRTable, GuardedStackShift, StackShift, StackShiftGuard, testing::build_test_table},
};
use glrmask_parser_dwa::__private::templates::{
    admissibility::TemplateDomain,
    characterize::characterize_selected_terminals_for_terminal_count,
    compile_dfa::{Templates, specialize_template_dfa_defaults_for_commit_split_input, try_split_commit_template_dfas},
};

fn split_template(table: &GLRTable) -> CommitTemplateDfas {
    let chars = characterize_selected_terminals_for_terminal_count(table, 1, &[true]);
    let raw = Templates::from_characterizations(&chars).by_terminal.remove(&0).unwrap();
    let specialized = specialize_template_dfa_defaults_for_commit_split_input(&raw);
    try_split_commit_template_dfas(&specialized).expect("bounded acyclic relation must split")
}

fn domain(table: &GLRTable) -> TemplateDomain {
    TemplateDomain::compile(&split_template(table)).unwrap()
}

// Independent literal stack transformer. It models the template relation as
// concrete words, not through the optimized GSS evaluator or domain compiler.
fn literal_outputs(template: &CommitTemplateDfas, original: &[u32]) -> BTreeSet<Vec<u32>> {
    let mut work = vec![(0u8, template.pop.start_state, original.to_vec())];
    let mut visited = BTreeSet::new();
    let mut outputs = BTreeSet::new();
    while let Some((phase, id, stack)) = work.pop() {
        if !visited.insert((phase,id,stack.clone())) { continue; }
        let dfa = [&template.pop,&template.read,&template.push][phase as usize];
        let state = &dfa.states[id as usize];
        if state.is_accepting { outputs.insert(stack.clone()); }
        match phase {
            0 => {
                if let Some(&top) = stack.last()
                    && let Some(&target) = state.transitions.get(&(top as i32)).or_else(|| state.transitions.get(&DEFAULT_LABEL))
                {
                    let mut next=stack.clone(); next.pop(); work.push((0,target,next));
                }
                if let Some(Some(target))=template.pop_to_read.get(id as usize) { work.push((1,*target,stack.clone())); }
                if let Some(Some(target))=template.pop_to_push.get(id as usize) { work.push((2,*target,stack)); }
            }
            1 => {
                if let Some(&top)=stack.last() && let Some(&target)=state.transitions.get(&(top as i32)) {
                    work.push((1,target,stack.clone()));
                }
                if let Some(Some(target))=template.read_to_push.get(id as usize) { work.push((2,*target,stack)); }
            }
            2 => for (&label,&target) in &state.transitions {
                let mut next=stack.clone(); next.push(negative_to_positive_label(label) as u32); work.push((2,target,next));
            },
            _=>unreachable!(),
        }
    }
    outputs
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
            let split = split_template(&table);
            let compiled = TemplateDomain::compile(&split).unwrap();
            let wire = compiled.to_bytes().unwrap(); let compiled = TemplateDomain::from_bytes(&wire).unwrap();
            for stack in &all {
                let gss = ParserGSS::from_single_stack(stack.clone(),TerminalsDisallowed::new());
                let advanced = advance_stacks(&table,&gss,0);
                let expected = !advanced.is_empty();
                let expected_stacks: BTreeSet<Vec<u32>> = advanced.to_stacks(10_000).unwrap().into_iter().map(|entry| entry.0).collect();
                assert_eq!(literal_outputs(&split, stack), expected_stacks,
                    "output relation len={len} replace={replace} stack={stack:?}");
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

#[test]
fn read_compression_does_not_enable_shadowed_default_stack_effects() {
    use glrmask_finite_automata::unweighted_u32::dfa::DFA;
    use glrmask_glr::__private::glr::labels::encode_negative_label;
    let mut dfa = DFA::new();
    let a = dfa.add_state(); let b = dfa.add_state(); let accept = dfa.add_state();
    let fallback = dfa.add_state();
    dfa.add_transition(dfa.start_state, 7, a);
    dfa.add_transition(a, encode_negative_label(7), b);
    dfa.add_transition(b, encode_negative_label(20), accept);
    dfa.add_transition(dfa.start_state, DEFAULT_LABEL, fallback);
    dfa.add_transition(fallback, encode_negative_label(30), accept);
    dfa.add_transition(dfa.start_state, encode_negative_label(40), accept);
    dfa.set_accepting(accept, true);
    let split = try_split_commit_template_dfas(&dfa).unwrap();
    for bottom in [0, 7, 9] {
        assert_eq!(literal_outputs(&split, &[bottom,7]),
            BTreeSet::from([vec![bottom,7,20], vec![bottom,7,40]]));
        assert_eq!(literal_outputs(&split, &[bottom,9]),
            BTreeSet::from([vec![bottom,30], vec![bottom,9,40]]));
    }
    assert_eq!(literal_outputs(&split, &[]), BTreeSet::from([vec![40]]));
}

#[test]
fn mixed_guarded_shift_and_nonreplacing_reduce_keeps_both_paths() {
    // Minimized from Codecov: the fast combined predecessor-remap handles
    // replacing gotos only. A non-replacing goto is unsupported, NOT dead.
    // The relation is distributive: [0,1,2] -> [0,7], [0,3] -> [0,5].
    for chained in [false, true] {
        let actions = [
            vec![], vec![],
            vec![(0, Action::GuardedStackShifts(vec![GuardedStackShift {
                guards: vec![StackShiftGuard { pop: 1, states: vec![1] }],
                pop: 2, pushes: vec![7],
            }]))],
            vec![(0, Action::Reduce(0,1))],
            vec![(0, if chained { Action::Reduce(1,1) } else { Action::Shift(5,true) })],
            vec![], vec![(0,Action::Shift(5,true))], vec![],
        ];
        let gotos = [vec![(0,(4,false)),(1,(6,false))],vec![],vec![],vec![],vec![],vec![],vec![],vec![]];
        let table=build_test_table(8,1,&actions.iter().map(Vec::as_slice).collect::<Vec<_>>(),
            &gotos.iter().map(Vec::as_slice).collect::<Vec<_>>());
        let expected=BTreeSet::from([vec![0,5],vec![0,7]]);
        let split=split_template(&table);
        let mut literal=literal_outputs(&split,&[0,1,2]);
        literal.extend(literal_outputs(&split,&[0,3]));
        assert_eq!(literal,expected,"literal relation chained={chained}");
        for reverse in [false,true] {
            let mut inputs=vec![(vec![0,1,2],TerminalsDisallowed::new()),(vec![0,3],TerminalsDisallowed::new())];
            if reverse { inputs.reverse(); }
            let gss=ParserGSS::from_stacks(&inputs);
            for (kind, result) in [
                ("borrowed", advance_stacks(&table, &gss, 0)),
                ("owned", advance_stacks_owned(&table, gss.clone(), 0)),
                ("profiled", advance_stacks_profiled(&table, &gss, 0).0),
            ] {
                let actual: BTreeSet<_> = result.to_stacks(100).unwrap()
                    .into_iter().map(|(stack, _)| stack).collect();
                assert_eq!(actual, expected,
                    "combined LR kernel {kind} chained={chained} reverse={reverse}");
            }
        }
    }
}


#[test]
fn fallible_selected_characterization_rejects_cycles_without_panicking_or_fallback() {
    use glrmask_parser_dwa::__private::templates::characterize::
        try_characterize_selected_terminals_for_terminal_count;

    // This reduction crosses its pushed goto prefix, consumes another input
    // predecessor, and may repeat until it finds state 2 and reaches a shift.
    // Unlike a pure no-output epsilon loop, it has unbounded productive stack
    // inspection, so the characterizer must decline acyclic certification.
    let actions = [vec![], vec![(0, Action::Reduce(0, 2))], vec![],
        vec![(0, Action::Shift(4, false))], vec![]];
    let gotos = [vec![(0, (1, false))], vec![], vec![(0, (3, false))], vec![], vec![]];
    let cyclic = build_test_table(5, 1,
        &actions.iter().map(Vec::as_slice).collect::<Vec<_>>(),
        &gotos.iter().map(Vec::as_slice).collect::<Vec<_>>());
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(||
        try_characterize_selected_terminals_for_terminal_count(&cyclic, 1, &[true])));
    let error = result.expect("unsupported reduction cycles must return an error")
        .expect_err("a cyclic relation must not be truncated or accepted");
    assert!(error.contains("terminal 0") && error.contains("reduction cycle"), "{error}");
    assert!(try_characterize_selected_terminals_for_terminal_count(&cyclic, 1, &[false])
        .unwrap().is_empty(), "an unselected relation requires no characterization");
    assert!(try_characterize_selected_terminals_for_terminal_count(&cyclic, 1, &[]).is_err());

    let actions = [vec![(0, Action::Shift(1, false))], vec![(0, Action::Skip)]];
    let acyclic = build_test_table(2, 1,
        &actions.iter().map(Vec::as_slice).collect::<Vec<_>>(), &[&[], &[]]);
    assert_eq!(try_characterize_selected_terminals_for_terminal_count(&acyclic, 1, &[true]).unwrap(),
        characterize_selected_terminals_for_terminal_count(&acyclic, 1, &[true]),
        "the fallible API must preserve the ordinary bounded relation exactly");
}
