use std::{collections::{BTreeSet, VecDeque}, sync::Arc};
use glrmask_artifact::CommitTemplateDfas;
use glrmask_finite_automata::unweighted_u32::dfa::DFA;
use glrmask_glr::__private::glr::labels::{DEFAULT_LABEL, encode_negative_label, negative_to_positive_label};
#[path = "../src/compiler/template_follow_support.rs"]
mod support;

fn reject() -> CommitTemplateDfas {
    CommitTemplateDfas { pop: DFA::new(), read: DFA::new(), push: DFA::new(),
        pop_to_read: vec![], pop_to_push: vec![], read_to_push: vec![] }
}

fn rewrite(pop: &[i32], push: &[u32]) -> CommitTemplateDfas {
    let mut p = reject(); let mut at = 0;
    for &label in pop { let next = p.pop.add_state(); p.pop.add_transition(at, label, next); at = next; }
    p.pop_to_push.resize(p.pop.states.len(), None); p.pop_to_push[at as usize] = Some(0);
    at = 0;
    for &label in push { let next = p.push.add_state(); p.push.add_transition(at, encode_negative_label(label), next); at = next; }
    p.push.set_accepting(at, true); p
}

// Independent literal phase interpreter: no production GSS, domain automata,
// signed-word rewriting, or static support approximation is used here.
fn execute(p: &CommitTemplateDfas, input: &[u32]) -> BTreeSet<Vec<u32>> {
    let mut todo = vec![(0u8, p.pop.start_state, input.to_vec())];
    let mut seen = BTreeSet::new(); let mut result = BTreeSet::new();
    while let Some((phase, state, stack)) = todo.pop() {
        assert!(seen.len() < 65536);
        if !seen.insert((phase, state, stack.clone())) { continue; }
        let row = match phase { 0 => &p.pop.states[state as usize], 1 => &p.read.states[state as usize], _ => &p.push.states[state as usize] };
        if row.is_accepting { result.insert(stack.clone()); }
        match phase {
            0 => {
                if let Some(next) = p.pop_to_read.get(state as usize).copied().flatten() { todo.push((1, next, stack.clone())); }
                if let Some(next) = p.pop_to_push.get(state as usize).copied().flatten() { todo.push((2, next, stack.clone())); }
                if let Some(&top) = stack.last() {
                    if let Some(&next) = row.transitions.get(&(top as i32)).or_else(|| row.transitions.get(&DEFAULT_LABEL)) {
                        let mut rest = stack; rest.pop(); todo.push((0, next, rest));
                    }
                }
            }
            1 => {
                if let Some(next) = p.read_to_push.get(state as usize).copied().flatten() { todo.push((2, next, stack.clone())); }
                if let Some(&top) = stack.last() {
                    if let Some(&next) = row.transitions.get(&(top as i32)) { todo.push((1, next, stack)); }
                }
            }
            _ => for (&label, &next) in &row.transitions {
                let mut rest = stack.clone(); rest.push(negative_to_positive_label(label) as u32); todo.push((2, next, rest));
            }
        }
    }
    result
}

fn closure(programs: &[Option<Arc<CommitTemplateDfas>>], mut reached: BTreeSet<Vec<u32>>) -> BTreeSet<Vec<u32>> {
    let mut todo = VecDeque::from_iter(reached.iter().cloned());
    while let Some(stack) = todo.pop_front() {
        for control in &programs[4..] {
            for next in execute(control.as_ref().unwrap(), &stack) {
                assert_eq!(next.len(), stack.len(), "test controls preserve depth, so this is a finite exhaustive closure");
                if reached.insert(next.clone()) { todo.push_back(next); }
            }
        }
    }
    reached
}

fn random(s: &mut u64) -> u32 { *s = s.wrapping_mul(6364136223846793005).wrapping_add(1); (*s >> 32) as u32 }

#[test]
fn template_follow_upper_bound_never_prunes_literal_control_paths() {
    let mut stacks = vec![vec![]]; let mut layer = vec![vec![]];
    for _ in 0..4 { let mut next = vec![]; for stack in layer { for top in 0..3 {
        let mut s = stack.clone(); s.push(top); next.push(s);
    }} stacks.extend(next.iter().cloned()); layer = next; }
    let mut rng = 10379; let mut checks = 0usize; let mut exclusions = 0usize;
    for case in 0..256 {
        let mut programs = vec![];
        for _ in 0..4 {
            let pop = (0..random(&mut rng) % 3).map(|_| if random(&mut rng) % 4 == 0 { DEFAULT_LABEL } else { (random(&mut rng) % 3) as i32 }).collect::<Vec<_>>();
            let push = (0..random(&mut rng) % 3).map(|_| random(&mut rng) % 3).collect::<Vec<_>>();
            programs.push(Some(Arc::new(rewrite(&pop, &push))));
        }
        for _ in 0..2 { programs.push(Some(Arc::new(rewrite(&[(random(&mut rng) % 3) as i32], &[random(&mut rng) % 3])))); }
        for budget in [0, 48, 96, 1024, 32_000_000] {
            let rows = support::disallowed(&programs, 4, budget).unwrap();
            exclusions += rows.iter().map(Vec::len).sum::<usize>();
            for stack in &stacks { for first in 0..4 {
                let after = closure(&programs, execute(programs[first].as_ref().unwrap(), stack));
                for &second in &rows[first] {
                    assert!(!after.iter().any(|s| !execute(programs[second as usize].as_ref().unwrap(), s).is_empty()),
                        "unsound exclusion case={case} budget={budget} input={stack:?} first={first} second={second}");
                    checks += 1;
                }
            }}
        }
    }
    assert!(exclusions > 0);
    // An accepting empty PUSH / POP-only relation retains an unknown lower
    // top (or empty stack); it must not invent a narrower output support.
    let programs = vec![Some(Arc::new(rewrite(&[1], &[]))), Some(Arc::new(rewrite(&[], &[2])))];
    assert!(support::disallowed(&programs, 2, 32000).unwrap()[0].is_empty());
    println!("PASS template-only follow exclusions: literal_checks={checks} excluded_pairs={exclusions}, including exhausted-budget widening");
}
