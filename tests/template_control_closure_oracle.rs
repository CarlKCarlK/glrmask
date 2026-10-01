//! Independent finite-state/literal oracle for the shared signed-NWA solver.
use std::collections::{BTreeMap, VecDeque};
use glrmask_weight::Weight;
use glrmask_weighted_automata::weighted::{nwa::NWA, dwa::DWA};
use glrmask_parser_dwa::__private::{resolve_negatives::resolve_negative_codes_in_nwa,
    parser_dwa::normalize_weighted_stack_predicate_for_symbol_count};
use range_set_blaze::RangeSetBlaze;
const DEFAULT: i32 = i32::MAX - 1;
fn push(v: u32) -> i32 { i32::MIN + v as i32 }
fn weight(bits: u8) -> Weight {
    Weight::from_uniform(0..=0, (0..2u32).filter(|&id| bits & (1 << id) != 0).collect::<RangeSetBlaze<u32>>())
}
fn bits(w: &Weight) -> u8 {
    (0..2u32).fold(0, |n, id| n | if w.tokens_for_tsid(0).contains(id) { 1 << id } else { 0 })
}
fn add(nwa: &mut NWA, from: u32, to: u32, ops: &[i32], mask: u8) {
    if ops.is_empty() { nwa.add_epsilon(from, to, weight(mask)); return; }
    let mut at = from;
    for (i, &op) in ops.iter().enumerate() {
        let next = if i + 1 == ops.len() { to } else { nwa.add_state() };
        nwa.add_transition(at, op, next, weight(mask)); at = next;
    }
}
fn compile(controls: &[(Vec<i32>, u8)], ordinary: &[(Vec<i32>, u8)]) -> DWA {
    let mut nwa = NWA::new(1, 1); let start = nwa.add_state(); let end = nwa.add_state();
    nwa.set_start_states(vec![start]); nwa.set_final_weight(end, weight(3));
    for (ops, mask) in controls { add(&mut nwa, start, start, ops, *mask); }
    for (ops, mask) in ordinary { add(&mut nwa, start, end, ops, *mask); }
    resolve_negative_codes_in_nwa(&mut nwa, false);
    assert!(nwa.states().iter().all(|row| row.transitions.keys().all(|&label| label >= 0)));
    normalize_weighted_stack_predicate_for_symbol_count(3, &nwa)
}
fn accepts(dwa: &DWA, input: &[u32]) -> u8 {
    let mut at = dwa.start_state(); let mut live = 3;
    let mut accepted = dwa.states()[at as usize].final_weight.as_ref().map_or(0, bits);
    for &top in input.iter().rev() {
        let row = &dwa.states()[at as usize];
        let Some((next, w)) = row.transitions.get(&(top as i32)).or_else(|| row.transitions.get(&DEFAULT)) else { break; };
        live &= bits(w); at = *next;
        accepted |= live & dwa.states()[at as usize].final_weight.as_ref().map_or(0, bits);
    }
    accepted
}
fn execute(ops: &[i32], input: &[u32]) -> Option<Vec<u32>> {
    let mut stack = input.to_vec();
    for &op in ops {
        if op < 0 { stack.push(op.wrapping_sub(i32::MIN) as u32); }
        else if stack.pop() != Some(op as u32) { return None; }
    }
    Some(stack)
}
fn literal(controls: &[(Vec<i32>, u8)], ordinary: &[(Vec<i32>, u8)], input: &[u32]) -> u8 {
    // Controls in randomized tests preserve depth. This closure is genuinely
    // finite (at most 3^depth states), not a bounded trace approximation.
    let mut reached = BTreeMap::from([(input.to_vec(), 3u8)]);
    let mut queue = VecDeque::from([input.to_vec()]); let mut accepted = 0;
    while let Some(stack) = queue.pop_front() {
        let live = reached[&stack];
        for (ops, mask) in ordinary { if execute(ops, &stack).is_some() { accepted |= live & mask; } }
        for (ops, mask) in controls {
            if let Some(next) = execute(ops, &stack) {
                assert_eq!(next.len(), input.len());
                let slot = reached.entry(next.clone()).or_default(); let add = live & mask;
                if add & !*slot != 0 { *slot |= add; queue.push_back(next); }
            }
        }
    }
    accepted
}
fn rand(s: &mut u64) -> u32 { *s = s.wrapping_mul(6364136223846793005).wrapping_add(1); (*s >> 32) as u32 }
#[test]
fn cyclic_control_programs_match_a_literal_weighted_closure() {
    let pop_star = compile(&[(vec![1], 3)], &[(vec![0], 3)]);
    for depth in 0..128 { let mut input = vec![0]; input.resize(depth + 1, 1); assert_eq!(accepts(&pop_star, &input), 3); }
    assert_eq!(accepts(&pop_star, &[]), 0); assert_eq!(accepts(&pop_star, &[2, 1, 1]), 0);
    let push_star = compile(&[(vec![push(1)], 3)], &[(vec![1, 1, 0], 3)]);
    for depth in 0..12 { let mut input = vec![0]; input.resize(depth + 1, 1); assert_eq!(accepts(&push_star, &input), if depth <= 2 { 3 } else { 0 }); }
    assert_eq!(accepts(&push_star, &[]), 0);
    let mut stacks = vec![vec![]]; let mut layer = vec![vec![]];
    for _ in 0..5 { let mut next = vec![]; for stack in layer { for top in 0..3 {
        let mut s = stack.clone(); s.push(top); next.push(s);
    }} stacks.extend(next.iter().cloned()); layer = next; }
    let mut rng = 7397; let mut comparisons = 0;
    for case in 0..256 {
        let mut controls = vec![];
        for _ in 0..(1 + rand(&mut rng) % 6) {
            let n = rand(&mut rng) % 3; let mut ops = vec![];
            for _ in 0..n { ops.push((rand(&mut rng) % 3) as i32); }
            for _ in 0..n { ops.push(push(rand(&mut rng) % 3)); }
            controls.push((ops, (1 + rand(&mut rng) % 3) as u8));
        }
        let mut ordinary = vec![];
        for _ in 0..(1 + rand(&mut rng) % 3) {
            let mut ops = vec![]; for _ in 0..(rand(&mut rng) % 4) { ops.push((rand(&mut rng) % 3) as i32); }
            for _ in 0..(rand(&mut rng) % 3) { ops.push(push(rand(&mut rng) % 3)); }
            ordinary.push((ops, (1 + rand(&mut rng) % 3) as u8));
        }
        let dwa = compile(&controls, &ordinary);
        for stack in &stacks {
            assert_eq!(accepts(&dwa, stack), literal(&controls, &ordinary, stack), "case={case} stack={stack:?} controls={controls:?} ordinary={ordinary:?}");
            comparisons += 1;
        }
    }
    println!("PASS control-star weighted fixed-point comparisons={comparisons}; unbounded pop/push witnesses passed");
}
