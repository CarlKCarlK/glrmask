//! Execute a small template relation inside the existing flat commit frontier.
//!
//! POP and READ each have one route for a concrete input stack. Phase links
//! contribute independent output alternatives along that route. Small PUSH
//! languages use the already-derived exact suffix cache. No LR action, parser
//! callback, GSS allocation, or separate lexer/commit engine is involved.
//!
//! All limits are optimization budgets. A decline clears partial scratch output
//! and leaves the caller's stack untouched; the ordinary template/GSS evaluator
//! then executes the complete relation, including large shared PUSH languages.

use super::{FlatActionScratch, FlatInlineStack, FLAT_ACTION_MAX_STEPS, LINEAR_STACK_RESERVE};
use crate::runtime::artifact::FastCommitTemplateDfas;

#[inline]
fn charge(left: &mut usize, amount: usize) -> Option<()> {
    *left = left.checked_sub(amount)?;
    Some(())
}

#[inline]
fn emit(base: &[u32], suffix: &[u32], scratch: &mut FlatActionScratch, left: &mut usize) -> Option<()> {
    let len = base.len().checked_add(suffix.len())?;
    if len > LINEAR_STACK_RESERVE { return None; }
    // Bound copying as well as graph visits. The output buffer never spills.
    charge(left, 1 + len)?;
    let mut stack = FlatInlineStack::new();
    stack.extend_from_slice(base);
    stack.extend_from_slice(suffix);
    scratch.push_complete(stack).then_some(())
}

#[inline]
fn push_outputs(
    template: &FastCommitTemplateDfas,
    entry: u32,
    base: &[u32],
    scratch: &mut FlatActionScratch,
    left: &mut usize,
) -> Option<()> {
    // None means preparation declined, not that the language is empty.
    let language = template.push_suffixes.get(entry as usize)?.as_ref()?;
    for suffix in language.words() {
        emit(base, suffix, scratch, left)?;
    }
    Some(())
}

fn read_outputs(
    template: &FastCommitTemplateDfas,
    mut state: u32,
    base: &[u32],
    scratch: &mut FlatActionScratch,
    left: &mut usize,
) -> Option<()> {
    loop {
        charge(left, 1)?;
        let row = template.read.states.get(state as usize)?;
        if row.default_target.is_some() { return None; }
        if row.is_accepting { emit(base, &[], scratch, left)?; }
        if let Some(push) = template.read_to_push.get(state as usize).copied().flatten() {
            push_outputs(template, push, base, scratch, left)?;
        }
        // Epsilon phase links and empty READ words work even on an empty
        // stack. Only an actual READ edge needs a visible input symbol.
        let Some(&top) = base.last() else { break; };
        let Some(next) = row.transitions.get(top as i32) else { break; };
        state = next;
    }
    Some(())
}

fn evaluate(
    template: &FastCommitTemplateDfas,
    source: &[u32],
    scratch: &mut FlatActionScratch,
) -> Option<bool> {
    if source.len() > LINEAR_STACK_RESERVE { return None; }
    let mut left = FLAT_ACTION_MAX_STEPS;
    let mut keep = source.len();
    let mut state = template.pop.start_state;
    loop {
        charge(&mut left, 1)?;
        let row = template.pop.states.get(state as usize)?;
        let base = &source[..keep];
        if row.is_accepting { emit(base, &[], scratch, &mut left)?; }
        if let Some(push) = template.pop_to_push.get(state as usize).copied().flatten() {
            push_outputs(template, push, base, scratch, &mut left)?;
        }
        if let Some(read) = template.pop_to_read.get(state as usize).copied().flatten() {
            read_outputs(template, read, base, scratch, &mut left)?;
        }
        let Some(&top) = base.last() else { break; };
        // Resolve explicit shadowing BEFORE testing destination productivity:
        // an explicit dead edge still excludes that symbol from DEFAULT.
        let Some(next) = row.transitions.get(top as i32).or(row.default_target) else { break; };
        state = next;
        keep -= 1;
    }
    Some(!scratch.complete.is_empty())
}

pub(super) fn apply(
    template: &FastCommitTemplateDfas,
    source: &[u32],
    scratch: &mut FlatActionScratch,
) -> Option<bool> {
    scratch.clear();
    let result = evaluate(template, source, scratch);
    if result.is_none() { scratch.clear(); }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automata::unweighted_u32::dfa::{DFA, DFAState};
    use crate::compiler::glr::labels::{DEFAULT_LABEL, encode_negative_label, negative_to_positive_label};
    use crate::runtime::CommitTemplateDfas;
    use std::collections::{BTreeMap, BTreeSet};

    fn graph(n: usize) -> DFA {
        DFA { start_state: 0, states: (0..n).map(|_| DFAState { is_accepting: false, transitions: BTreeMap::new() }).collect() }
    }
    fn blank(n: usize) -> CommitTemplateDfas {
        CommitTemplateDfas { pop: graph(n), read: graph(n), push: graph(n), pop_to_read: Vec::new(), pop_to_push: Vec::new(), read_to_push: Vec::new() }
    }
    // Literal stack-language oracle: no fast rows, suffix cache, shared GSS,
    // admission certificate, or production traversal helper is used here.
    fn literal(t: &CommitTemplateDfas, input: &[u32]) -> BTreeSet<Vec<u32>> {
        let mut pending = vec![(0u8, t.pop.start_state, input.to_vec())];
        let mut output = BTreeSet::new();
        while let Some((phase, id, stack)) = pending.pop() {
            let g = match phase { 0 => &t.pop, 1 => &t.read, _ => &t.push };
            let row = &g.states[id as usize];
            if row.is_accepting { output.insert(stack.clone()); }
            if phase == 0 {
                if let Some(r) = t.pop_to_read.get(id as usize).copied().flatten() { pending.push((1, r, stack.clone())); }
                if let Some(p) = t.pop_to_push.get(id as usize).copied().flatten() { pending.push((2, p, stack.clone())); }
            } else if phase == 1 {
                if let Some(p) = t.read_to_push.get(id as usize).copied().flatten() { pending.push((2, p, stack.clone())); }
            }
            match phase {
                0 => if let Some(&top) = stack.last() {
                    if let Some(&next) = row.transitions.get(&(top as i32)).or_else(||row.transitions.get(&DEFAULT_LABEL)) {
                        let mut s = stack; s.pop(); pending.push((0, next, s));
                    }
                },
                1 => if let Some(&top) = stack.last() {
                    if let Some(&next) = row.transitions.get(&(top as i32)) { pending.push((1, next, stack)); }
                },
                _ => for (&label, &next) in &row.transitions {
                    let mut s = stack.clone(); s.push(u32::try_from(negative_to_positive_label(label)).expect("valid PUSH label")); pending.push((2, next, s));
                },
            }
        }
        output
    }
    fn outputs(s: &FlatActionScratch) -> BTreeSet<Vec<u32>> { s.complete.iter().map(|x|x.to_vec()).collect() }
    fn next(x: &mut u64) -> u32 { *x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (*x >> 32) as u32 }

    #[test]
    fn generated_acyclic_flat_relations_match_literal_oracle() {
        let mut seed = 608894u64;
        let mut checked = 0;
        let mut declined = 0;
        let mut words = vec![Vec::new()];
        for n in 1..=3 {
            for code in 0..3usize.pow(n) {
                let mut c = code; let mut w = Vec::new();
                for _ in 0..n { w.push((c % 3) as u32); c /= 3; }
                words.push(w);
            }
        }
        for _ in 0..512 {
            let mut t = blank(4);
            for (phase, g) in [&mut t.pop, &mut t.read, &mut t.push].into_iter().enumerate() {
                for i in 0..4 {
                    g.states[i].is_accepting = next(&mut seed) % 3 == 0;
                    if i == 3 { continue; }
                    for label in 0..3 {
                        if next(&mut seed) % 2 == 0 {
                            let target = (i + 1 + next(&mut seed) as usize % (3-i)) as u32;
                            let l = if phase == 2 { encode_negative_label(label) } else { label as i32 };
                            g.states[i].transitions.insert(l, target);
                        }
                    }
                    if phase == 0 && next(&mut seed) % 2 == 0 {
                        g.states[i].transitions.insert(DEFAULT_LABEL, (i+1) as u32);
                    }
                }
            }
            for links in [&mut t.pop_to_read, &mut t.pop_to_push, &mut t.read_to_push] {
                for _ in 0..(next(&mut seed)%5) {
                    links.push((next(&mut seed)%2 == 0).then(|| next(&mut seed)%4));
                }
            }
            let fast = FastCommitTemplateDfas::from_template(&t);
            for word in &words {
                let expected = literal(&t, word);
                let mut scratch = FlatActionScratch::default();
                match apply(&fast, word, &mut scratch) {
                    Some(nonempty) => { assert_eq!(nonempty, !expected.is_empty()); assert_eq!(outputs(&scratch), expected); checked += 1; }
                    None => { assert!(scratch.complete.is_empty()); declined += 1; }
                }
                assert!(!scratch.complete.spilled() && !scratch.pending.spilled());
            }
        }
        assert!(checked > 8_000, "only {checked} exact evaluations, {declined} conservative declines");
    }

    #[test]
    fn dead_default_shadow_and_read_epsilon_on_empty_are_exact() {
        let mut t = blank(3);
        t.pop.states[0].transitions.insert(DEFAULT_LABEL, 1);
        t.pop.states[0].transitions.insert(1, 2); // Explicit dead edge.
        t.pop.states[1].is_accepting = true;
        t.pop_to_read = vec![Some(0)];
        t.read.states[0].is_accepting = true;
        let fast = FastCommitTemplateDfas::from_template(&t);
        for source in [vec![], vec![0,1], vec![0,2]] {
            let mut scratch = FlatActionScratch::default();
            assert!(apply(&fast, &source, &mut scratch).is_some());
            assert_eq!(outputs(&scratch), literal(&t, &source));
        }
    }

    #[test]
    fn exhausted_work_and_stack_budgets_decline_transactionally() {
        let mut t = blank(FLAT_ACTION_MAX_STEPS + 2);
        t.pop_to_read = vec![Some(0)];
        for i in 0..FLAT_ACTION_MAX_STEPS + 1 {
            t.read.states[i].is_accepting = true;
            t.read.states[i].transitions.insert(0, (i+1) as u32);
        }
        let fast = FastCommitTemplateDfas::from_template(&t);
        let source = vec![0]; let mut scratch = FlatActionScratch::default();
        assert!(apply(&fast, &source, &mut scratch).is_none());
        assert!(scratch.complete.is_empty()); assert_eq!(source, [0]);
        let mut t = blank(2);
        t.pop_to_push = vec![Some(0)];
        t.push.states[0].transitions.insert(encode_negative_label(1), 1);
        t.push.states[1].is_accepting = true;
        assert!(apply(&FastCommitTemplateDfas::from_template(&t), &[0; LINEAR_STACK_RESERVE], &mut scratch).is_none());
        assert!(scratch.complete.is_empty());
    }

    #[test]
    fn too_many_outputs_decline_without_partial_language() {
        let mut t = blank(2);
        t.pop_to_push = vec![Some(0)];
        for symbol in 0..32 { t.push.states[0].transitions.insert(encode_negative_label(symbol), 1); }
        t.push.states[1].is_accepting = true;
        let fast = FastCommitTemplateDfas::from_template(&t);
        let mut scratch = FlatActionScratch::default();
        assert!(apply(&fast, &[0], &mut scratch).is_none());
        assert!(scratch.complete.is_empty());
        assert!(!scratch.complete.spilled());
    }
}
