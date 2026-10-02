use std::collections::BTreeSet;
use super::{CommitTemplateDfas, DFA, DEFAULT_LABEL, encode_negative_label, negative_to_positive_label};
use super as link_program;

// Deliberately independent, literal interpreter for small test graphs. No
// production GSS, signed-word recombination or linker transformation is used.
fn outputs(program: &CommitTemplateDfas, input: &[u32]) -> BTreeSet<Vec<u32>> {
    let mut todo = vec![(0u8, program.pop.start_state, input.to_vec())];
    let mut seen = BTreeSet::new(); let mut result = BTreeSet::new();
    while let Some((phase, state, stack)) = todo.pop() {
        assert!(seen.len() < 65536, "probe interpreter budget");
        if !seen.insert((phase, state, stack.clone())) { continue; }
        let graph = match phase { 0 => &program.pop, 1 => &program.read, _ => &program.push };
        let row = &graph.states[state as usize];
        if row.is_accepting { result.insert(stack.clone()); }
        match phase {
            0 => {
                if let Some(target) = program.pop_to_read.get(state as usize).copied().flatten() {
                    todo.push((1, target, stack.clone()));
                }
                if let Some(target) = program.pop_to_push.get(state as usize).copied().flatten() {
                    todo.push((2, target, stack.clone()));
                }
                if let Some(&top) = stack.last() {
                    if let Some(&target) = row.transitions.get(&(top as i32)).or_else(|| row.transitions.get(&DEFAULT_LABEL)) {
                        let mut rest = stack; rest.pop(); todo.push((0, target, rest));
                    }
                }
            }
            1 => {
                if let Some(target) = program.read_to_push.get(state as usize).copied().flatten() {
                    todo.push((2, target, stack.clone()));
                }
                if let Some(&top) = stack.last() {
                    if let Some(&target) = row.transitions.get(&(top as i32)) { todo.push((1, target, stack)); }
                }
            }
            _ => for (&label, &target) in &row.transitions {
                let mut next = stack.clone(); next.push(negative_to_positive_label(label) as u32);
                todo.push((2, target, next));
            }
        }
    }
    result
}

fn rename(program: &CommitTemplateDfas, offset: u32) -> CommitTemplateDfas {
    let mut p = program.clone();
    for dfa in [&mut p.pop, &mut p.read] { for row in &mut dfa.states {
        row.transitions = row.transitions.iter().map(|(&label, &target)|
            (if label == DEFAULT_LABEL { label } else { label + offset as i32 }, target)).collect();
    }}
    // The independent oracle explicitly restricts each DEFAULT to this
    // component; the production implementation checks its sidecar at runtime.
    for row in &mut p.pop.states {
        if let Some(target) = row.transitions.remove(&DEFAULT_LABEL) {
            for symbol in offset..offset + 3 {
                row.transitions.entry(symbol as i32).or_insert(target);
            }
        }
    }
    for row in &mut p.push.states {
        row.transitions = row.transitions.iter().map(|(&label, &target)|
            (encode_negative_label(negative_to_positive_label(label) as u32 + offset), target)).collect();
    }
    p
}

fn populations(alphabet: u32, max_depth: usize) -> Vec<Vec<u32>> {
    let mut all = vec![Vec::new()]; let mut layer = all.clone();
    for _ in 0..max_depth {
        let mut next = Vec::new();
        for stack in layer { for top in 0..alphabet {
            let mut extension = stack.clone(); extension.push(top); next.push(extension);
        }}
        all.extend(next.iter().cloned()); layer = next;
    }
    all
}

fn next(seed: &mut u64) -> u32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*seed >> 32) as u32
}

fn random_program(seed: &mut u64) -> CommitTemplateDfas {
    let mut graphs = Vec::new();
    for phase in 0..3 {
        let n = 1 + next(seed) % 5; let mut graph = DFA::new();
        while graph.states.len() < n as usize { graph.add_state(); }
        for state in 0..n {
            graph.set_accepting(state, next(seed) % 4 == 0);
            if state + 1 == n { continue; }
            for symbol in 0..3 {
                if next(seed) % 3 != 0 { continue; }
                let target = state + 1 + next(seed) % (n - state - 1);
                let label = if phase == 2 { encode_negative_label(symbol) } else { symbol as i32 };
                graph.add_transition(state, label, target);
            }
            if phase == 0 && next(seed) % 3 == 0 {
                let target = state + 1 + next(seed) % (n - state - 1);
                graph.add_transition(state, DEFAULT_LABEL, target);
            }
        }
        graphs.push(graph);
    }
    let push = graphs.pop().unwrap(); let read = graphs.pop().unwrap(); let pop = graphs.pop().unwrap();
    let mut links = |count: usize, target_count: usize| (0..count).map(|_| {
        let value = next(seed); (value % 3 == 0).then_some((value / 3) % target_count as u32)
    }).collect();
    CommitTemplateDfas { pop_to_read: links(pop.states.len(), read.states.len()),
        pop_to_push: links(pop.states.len(), push.states.len()), read_to_push: links(read.states.len(), push.states.len()),
        pop, read, push }
}

#[test]
fn randomized_relations_match_literal_phase_execution() {
    let mut seed = 503460u64; let stacks = populations(6, 3); let mut checks = 0usize;
    for case in 0..512 {
        let source = random_program(&mut seed);
        super::super::compile_domain(&source).unwrap();
        let renamed = rename(&source, 2);
        let scoped = link_program::compile(&[link_program::scoped(&source, 2, 3).unwrap()])
            .unwrap_or_else(|error| panic!("scope case {case}: {error}, source={source:?}"));
        let domain = super::super::compile_domain(&scoped).unwrap();
        let preserved = link_program::scoped_template(&source, 2, 3).unwrap();
        let preserved_domain = super::super::compile_domain(&preserved).unwrap();
        let mut called = link_program::action_nfa(&source).unwrap(); link_program::append_push(&mut called, 5);
        let called = link_program::compile(&[called])
            .unwrap_or_else(|error| panic!("append case {case}: {error}, source={source:?}"));
        for stack in &stacks {
            let expected = if stack.last().is_some_and(|top| (2..5).contains(top)) {
                outputs(&renamed, stack)
            } else { BTreeSet::new() };
            assert_eq!(outputs(&scoped, stack), expected, "scope case={case}, stack={stack:?}, source={source:?}");
            assert_eq!(outputs(&preserved, stack), expected, "preserved scope case={case}, stack={stack:?}");
            assert_eq!(domain.matches_top_first(stack.iter().rev().copied()), !expected.is_empty());
            assert_eq!(preserved_domain.matches_top_first(stack.iter().rev().copied()), !expected.is_empty());
            let expected = outputs(&source, stack).into_iter().map(|mut word| { word.push(5); word }).collect();
            assert_eq!(outputs(&called, stack), expected, "call case={case}, stack={stack:?}, source={source:?}");
            checks += 1;
        }
    }
    println!("PASS random_graphs=512 stack_inputs={} output_and_admission_checks={checks}", stacks.len());
}
