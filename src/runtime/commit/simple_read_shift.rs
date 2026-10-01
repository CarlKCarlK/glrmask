//! Exact local READ(t), PUSH(u) shortcut for the ordinary template interpreter.
//!
//! This recognizes a relation containing exactly one output on the current
//! known top: the input stack followed by one symbol. The parallel POP branch
//! must be structurally empty after explicit-before-DEFAULT selection. Neither
//! POP nor READ entry may accept or emit an epsilon PUSH branch; the selected
//! READ leaf and PUSH chain must have no other outputs. Thus no unknown suffix,
//! alternative stack path, allocation-based simulation, or LR action is needed.
//! A non-matching shape leaves the complete existing interpreter authoritative.

use crate::compiler::glr::labels::{is_negative_label, negative_to_positive_label};
#[cfg(test)]
use crate::compiler::glr::parser::ParserGSS;
use crate::ds::leveled_gss::{LeveledGSS, Merge};
use crate::runtime::artifact::FastTemplateTransitionRow;
use crate::runtime::{CommitTemplateDfas, FastCommitTemplateDfas};

// An optional derived plan, never the authoritative parser. A large root
// retains the existing interpreter instead of allocating another large table.
const MAX_READ_SHIFT_ENTRIES: usize = 512;

#[derive(Debug, Clone)]
pub(crate) struct PreparedReadShift {
    by_top: FastTemplateTransitionRow,
}

impl PreparedReadShift {
    pub(crate) fn prepare(template: &CommitTemplateDfas) -> Option<Box<Self>> {
        let pop_id = template.pop.start_state as usize;
        let pop = template.pop.states.get(pop_id)?;
        let read_id = template.pop_to_read.get(pop_id).copied().flatten()?;
        if pop.is_accepting
            || template
                .pop_to_push
                .get(pop_id)
                .copied()
                .flatten()
                .is_some()
        {
            return None;
        }
        let read = template.read.states.get(read_id as usize)?;
        if read.is_accepting
            || read.transitions.len() > MAX_READ_SHIFT_ENTRIES
            || template
                .read_to_push
                .get(read_id as usize)
                .copied()
                .flatten()
                .is_some()
        {
            return None;
        }
        let mut entries = Vec::new();
        for (&top, &target) in &read.transitions {
            if top < 0 || top == crate::compiler::glr::labels::DEFAULT_LABEL {
                return None;
            }
            if let Some(symbol) = exact_push_for_top(template, pop_id, top, target) {
                entries.push((top, symbol));
            }
        }
        if entries.is_empty() {
            return None;
        }
        let count = entries.len();
        Some(Box::new(Self {
            by_top: FastTemplateTransitionRow::from_entries(entries, count),
        }))
    }
}

// Construction-only proof of the selected local relation. It deliberately
// recognizes the same narrow shape as the earlier on-demand shortcut. No
// graph traversal is deferred into commitment or cold mask generation.
fn exact_push_for_top(
    template: &CommitTemplateDfas,
    pop_id: usize,
    top: i32,
    target: u32,
) -> Option<u32> {
    use crate::compiler::glr::labels::DEFAULT_LABEL;
    let read_tail = template.read.states.get(target as usize)?;
    if read_tail.is_accepting || !read_tail.transitions.is_empty() {
        return None;
    }
    let pop = template.pop.states.get(pop_id)?;
    if let Some(&other) = pop
        .transitions
        .get(&top)
        .or_else(|| pop.transitions.get(&DEFAULT_LABEL))
    {
        let dead = template.pop.states.get(other as usize)?;
        if dead.is_accepting
            || !dead.transitions.is_empty()
            || template
                .pop_to_read
                .get(other as usize)
                .copied()
                .flatten()
                .is_some()
            || template
                .pop_to_push
                .get(other as usize)
                .copied()
                .flatten()
                .is_some()
        {
            return None;
        }
    }
    let push_id = template
        .read_to_push
        .get(target as usize)
        .copied()
        .flatten()?;
    let push = template.push.states.get(push_id as usize)?;
    if push.is_accepting || push.transitions.len() != 1 {
        return None;
    }
    let (&label, &end) = push.transitions.first_key_value()?;
    if !is_negative_label(label) {
        return None;
    }
    let end = template.push.states.get(end as usize)?;
    if !end.is_accepting || !end.transitions.is_empty() {
        return None;
    }
    u32::try_from(negative_to_positive_label(label)).ok()
}

#[cfg(test)]
std::thread_local! {
    static SUCCESSES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    pub(crate) static MASK_SUCCESSES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[inline]
pub(crate) fn apply<A>(
    _template: &CommitTemplateDfas,
    fast: &FastCommitTemplateDfas,
    input: &LeveledGSS<u32, A>,
) -> Option<LeveledGSS<u32, A>>
where
    A: Merge + Eq + std::hash::Hash,
{
    let plan = fast.read_shift.as_ref()?;
    let top = i32::try_from(input.single_exclusive_top_value()?).ok()?;
    let symbol = plan.by_top.get(top)?;
    // This is the same GSS primitive used by the shared ordinary Shift path.
    // The borrowed wrapper invokes it before cloning an interpreter input.
    #[cfg(test)]
    SUCCESSES.with(|count| count.set(count.get() + 1));
    Some(input.push(symbol))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automata::unweighted_u32::dfa::DFA;
    use crate::compiler::glr::accumulator::TerminalsDisallowed;
    use crate::compiler::glr::labels::{DEFAULT_LABEL, encode_negative_label};
    use std::collections::BTreeSet;

    fn fixture(top: u32, pushed: u32) -> CommitTemplateDfas {
        let graph = || {
            let mut d = DFA::new();
            for _ in 1..4 {
                d.add_state();
            }
            d
        };
        let mut t = CommitTemplateDfas {
            pop: graph(),
            read: graph(),
            push: graph(),
            pop_to_read: vec![None; 4],
            pop_to_push: vec![None; 4],
            read_to_push: vec![None; 4],
        };
        t.pop.start_state = 2;
        t.pop.states[2].transitions.insert(top as i32, 3); // explicit dead blocker
        t.pop.states[2].transitions.insert(DEFAULT_LABEL, 1);
        t.pop.states[1].is_accepting = true; // OTHER tops are popped instead
        t.pop_to_read[2] = Some(1);
        t.read.states[1].transitions.insert(top as i32, 2);
        t.read_to_push[2] = Some(1);
        t.push.states[1]
            .transitions
            .insert(encode_negative_label(pushed), 2);
        t.push.states[2].is_accepting = true;
        t
    }

    fn literal(t: &CommitTemplateDfas, input: &ParserGSS) -> ParserGSS {
        let mut result = Vec::new();
        for (word, acc) in input.to_stacks(4096).unwrap() {
            let mut todo = vec![(0usize, t.pop.start_state, word)];
            let mut seen = BTreeSet::new();
            while let Some((phase, id, word)) = todo.pop() {
                if !seen.insert((phase, id, word.clone())) {
                    continue;
                }
                let row = &[&t.pop, &t.read, &t.push][phase].states[id as usize];
                if row.is_accepting {
                    result.push((word.clone(), acc.clone()));
                }
                if phase == 0 {
                    if let Some(Some(to)) = t.pop_to_read.get(id as usize) {
                        todo.push((1, *to, word.clone()));
                    }
                    if let Some(Some(to)) = t.pop_to_push.get(id as usize) {
                        todo.push((2, *to, word.clone()));
                    }
                } else if phase == 1 {
                    if let Some(Some(to)) = t.read_to_push.get(id as usize) {
                        todo.push((2, *to, word.clone()));
                    }
                }
                if phase == 2 {
                    for (&label, &to) in &row.transitions {
                        let mut next = word.clone();
                        next.push(negative_to_positive_label(label) as u32);
                        todo.push((phase, to, next));
                    }
                } else if let Some(&top) = word.last() {
                    if let Some(&to) = row.transitions.get(&(top as i32)).or_else(|| {
                        if phase == 0 {
                            row.transitions.get(&DEFAULT_LABEL)
                        } else {
                            None
                        }
                    }) {
                        let mut next = word.clone();
                        if phase == 0 {
                            next.pop();
                        }
                        todo.push((phase, to, next));
                    }
                }
            }
        }
        ParserGSS::from_stacks(&result)
    }

    fn check(t: &CommitTemplateDfas, input: &ParserGSS) -> Option<ParserGSS> {
        let before = input.to_stacks(4096).unwrap();
        let fast = FastCommitTemplateDfas::from_template(t);
        let actual = apply(t, &fast, input);
        if let Some(actual) = &actual {
            assert_eq!(
                actual.semantically_eq(&literal(t, input), 65536),
                Some(true)
            );
        }
        assert_eq!(before, input.to_stacks(4096).unwrap());
        actual
    }

    #[test]
    fn simple_read_shift_annotation_erasure_commutes_with_push() {
        let t = fixture(5, 7);
        let fast = FastCommitTemplateDfas::from_template(&t);
        let inputs = [
            ParserGSS::from_single_stack(vec![0, 5], TerminalsDisallowed::new()),
            ParserGSS::from_single_stack(
                vec![0, 1, 2, 5],
                TerminalsDisallowed::new().with_insert(0, 3),
            ),
            ParserGSS::from_stacks(&[
                (vec![0, 1, 5], TerminalsDisallowed::new()),
                (vec![0, 2, 5], TerminalsDisallowed::new().with_insert(1, 4)),
            ]),
        ];
        let mut used = 0;
        for input in inputs {
            let plain = input.apply(|_| ());
            let expected = literal(&t, &input).apply(|_| ());
            let actual = apply(&t, &fast, &plain).expect("unit GSS must use exact prepared shift");
            used += 1;
            assert_eq!(actual.semantically_eq(&expected, 65536), Some(true));
            assert_eq!(
                plain.semantically_eq(&input.apply(|_| ()), 65536),
                Some(true)
            );
        }
        assert_eq!(used, 3);
    }

    #[test]
    fn simple_read_shift_is_reached_inside_shared_dynamic_walk() {
        use crate::template_parser::{
            LexerDefinition, ParserDefinition, ParserProgram, StackLabel, StackTemplate,
            TemplateBuildOptions, TerminalPattern,
        };
        use crate::{Constraint, Optimization, Vocab};
        let tokens = [
            "a", "b", "aa", "ab", "ba", "bb", "aab", "aba", "aabb", "aaba", "abab", "aaabbb",
        ];
        let vocab = Vocab::new(
            tokens
                .iter()
                .enumerate()
                .map(|(id, word)| (id as u32, word.as_bytes().to_vec()))
                .collect(),
        );
        let program = ParserProgram::new(ParserDefinition {
            stack_symbol_count: 2,
            terminals: vec![
                StackTemplate::read_top_and_push([0, 1], [1]),
                StackTemplate::rewrite([StackLabel::Symbol(1)], []),
            ],
            completion: StackTemplate::read_top_and_push([0], []),
        })
        .unwrap();
        let lexer = LexerDefinition::new(vec![
            TerminalPattern::literal(b"a".to_vec()),
            TerminalPattern::literal(b"b".to_vec()),
        ]);
        let fresh = program
            .compile_with(
                &lexer,
                &vocab,
                TemplateBuildOptions::default().optimization(Optimization::FastBuild),
            )
            .unwrap();
        let loaded = Constraint::load(fresh.save()).unwrap();
        let external =
            Constraint::load_with_vocab(fresh.save_with_external_vocab().unwrap(), &vocab).unwrap();
        for constraint in [&fresh, &loaded, &external] {
            assert!(constraint.table.as_lr().is_none());
            MASK_SUCCESSES.with(|count| count.set(0));
            let mut state = constraint.start();
            let mut mask = vec![0u32; constraint.mask_len()];
            let mut balance = 0i32;
            for committed in [None, Some(0u32), Some(1u32)] {
                if let Some(token) = committed {
                    state.commit_token(token).unwrap();
                    balance += if token == 0 { 1 } else { -1 };
                }
                state.fill_mask(&mut mask);
                assert_eq!(state.is_accepting(), balance == 0);
                for (id, word) in tokens.iter().enumerate() {
                    let mut depth = balance;
                    let expected = word.bytes().all(|byte| {
                        depth += if byte == b'a' { 1 } else { -1 };
                        depth >= 0
                    });
                    let actual = mask[id / 32] & (1u32 << (id % 32)) != 0;
                    assert_eq!(actual, expected, "balance={balance}, token={word}");
                }
            }
            assert!(
                MASK_SUCCESSES.with(|count| count.get()) > 0,
                "the real shared dynamic walker must execute the unit-annotation shortcut"
            );
        }
    }

    #[test]
    fn simple_read_shift_preparation_is_bounded_and_keeps_exact_fallback() {
        let mut t = fixture(0, 7);
        t.pop.states[2].transitions.clear();
        t.pop.states[2].transitions.insert(DEFAULT_LABEL, 1);
        t.read.states[1].transitions.clear();
        for top in 0..MAX_READ_SHIFT_ENTRIES as i32 {
            t.pop.states[2].transitions.insert(top, 3);
            t.read.states[1].transitions.insert(top, 2);
        }
        let fast = FastCommitTemplateDfas::from_template(&t);
        let plan = fast
            .read_shift
            .as_ref()
            .expect("at-budget inventory must be used");
        let mut entries = Vec::new();
        plan.by_top
            .for_each(|top, pushed| entries.push((top, pushed)));
        assert_eq!(entries.len(), MAX_READ_SHIFT_ENTRIES);
        for top in [0, 7, 63, MAX_READ_SHIFT_ENTRIES as u32 - 1] {
            let input = ParserGSS::from_single_stack(vec![0, top], TerminalsDisallowed::new());
            let result = apply(&t, &fast, &input).unwrap();
            assert_eq!(
                result.semantically_eq(&literal(&t, &input), 65536),
                Some(true)
            );
        }
        let top = MAX_READ_SHIFT_ENTRIES as i32;
        t.pop.states[2].transitions.insert(top, 3);
        t.read.states[1].transitions.insert(top, 2);
        let over_budget = FastCommitTemplateDfas::from_template(&t);
        assert!(over_budget.read_shift.is_none());
        let input = ParserGSS::from_single_stack(vec![0, top as u32], TerminalsDisallowed::new());
        assert!(apply(&t, &over_budget, &input).is_none());
        assert_eq!(
            literal(&t, &input).semantically_eq(&input.push(7), 65536),
            Some(true)
        );
    }

    #[test]
    fn simple_read_shift_requires_complete_validated_graphs() {
        let mut t = fixture(5, 7);
        // The selected local branch would otherwise look eligible, but an
        // unreachable malformed/cyclic row must not bypass whole preparation.
        t.push.states[3]
            .transitions
            .insert(encode_negative_label(2), 3);
        let fast = FastCommitTemplateDfas::from_template(&t);
        assert!(fast.read_shift.is_none());
        t.push.states[3].transitions.clear();
        t.read.states[3].transitions.insert(DEFAULT_LABEL, 0);
        let fast = FastCommitTemplateDfas::from_template(&t);
        assert!(fast.read_shift.is_none());
    }

    #[test]
    fn simple_read_shift_matches_literal_with_shared_floors_and_accumulators() {
        for top in 0..16u32 {
            for pushed in [0, 1, 7, 63, 257] {
                let t = fixture(top, pushed);
                let left = TerminalsDisallowed::new().with_insert(2, 5);
                let right = TerminalsDisallowed::new().with_insert(4, 7);
                let sources = [
                    ParserGSS::from_single_stack(vec![top], left.clone()),
                    ParserGSS::from_single_stack(vec![0, 1, top], right.clone()),
                    ParserGSS::from_stacks(&[(vec![0, 1], left), (vec![0, 2], right)]).push(top),
                ];
                for source in sources {
                    let result =
                        check(&t, &source).expect("exact READ/PUSH fixture must use shortcut");
                    assert_eq!(
                        result.semantically_eq(&source.push(pushed), 65536),
                        Some(true)
                    );
                }
                assert!(
                    check(
                        &t,
                        &ParserGSS::from_single_stack(vec![top + 1], TerminalsDisallowed::new())
                    )
                    .is_none()
                );
            }
        }
    }

    #[test]
    fn simple_read_shift_declines_every_extra_phase_output() {
        let source = ParserGSS::from_single_stack(vec![0, 5], TerminalsDisallowed::new());
        for change in 0..10 {
            let mut t = fixture(5, 7);
            match change {
                0 => t.pop.states[2].is_accepting = true,
                1 => t.pop_to_push[2] = Some(1),
                2 => {
                    t.pop.states[2].transitions.remove(&5);
                } // DEFAULT is no longer shadowed
                3 => t.pop.states[3].is_accepting = true,
                4 => t.read.states[1].is_accepting = true,
                5 => t.read_to_push[1] = Some(1),
                6 => t.read.states[2].is_accepting = true,
                7 => {
                    t.read.states[2].transitions.insert(5, 3);
                    t.read.states[3].is_accepting = true;
                }
                8 => t.push.states[1].is_accepting = true,
                _ => {
                    t.push.states[1]
                        .transitions
                        .insert(encode_negative_label(8), 3);
                    t.push.states[3].is_accepting = true;
                }
            }
            assert!(check(&t, &source).is_none(), "variant {change}");
        }
    }

    #[test]
    fn simple_read_shift_declines_unknown_top_and_long_push() {
        let t = fixture(5, 7);
        assert!(check(&t, &ParserGSS::empty()).is_none());
        let branches = ParserGSS::from_stacks(&[
            (vec![0, 5], TerminalsDisallowed::new()),
            (vec![0, 6], TerminalsDisallowed::new()),
        ]);
        assert!(check(&t, &branches).is_none());
        let mut longer = t;
        longer.push.states[2].is_accepting = false;
        longer.push.states[2]
            .transitions
            .insert(encode_negative_label(8), 3);
        longer.push.states[3].is_accepting = true;
        assert!(
            check(
                &longer,
                &ParserGSS::from_single_stack(vec![0, 5], TerminalsDisallowed::new())
            )
            .is_none()
        );
    }

    #[test]
    fn simple_read_shift_is_reached_by_both_shared_advance_wrappers() {
        use crate::template_parser::{
            LexerDefinition, ParserDefinition, ParserProgram, StackTemplate, TemplateBuildOptions,
            TerminalPattern,
        };
        use crate::{Constraint, Optimization, Vocab};

        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"aa".to_vec())]);
        let parser = ParserProgram::new(ParserDefinition {
            stack_symbol_count: 2,
            terminals: vec![StackTemplate::read_top_and_push([0, 1], [1])],
            completion: StackTemplate::read_top_and_push([1], []),
        })
        .unwrap();
        let lexer = LexerDefinition::new(vec![TerminalPattern::literal(b"a".to_vec())]);
        for optimization in [Optimization::FastBuild, Optimization::FastRuntime] {
            let fresh = parser
                .compile_with(
                    &lexer,
                    &vocab,
                    TemplateBuildOptions::default().optimization(optimization),
                )
                .unwrap();
            let loaded = Constraint::load(fresh.save()).unwrap();
            let external =
                Constraint::load_with_vocab(fresh.save_with_external_vocab().unwrap(), &vocab)
                    .unwrap();
            for constraint in [&fresh, &loaded, &external] {
                assert!(constraint.table.as_lr().is_none());
                let input = ParserGSS::from_single_stack(vec![0], TerminalsDisallowed::new());
                let expected = input.push(1);
                SUCCESSES.with(|count| count.set(0));
                let borrowed = super::super::advance_parser_stacks(constraint, &input, 0);
                assert_eq!(
                    SUCCESSES.with(|count| count.get()),
                    1,
                    "borrowed shared parser primitive must reach the shortcut"
                );
                assert_eq!(borrowed.semantically_eq(&expected, 65536), Some(true));
                let owned = super::super::advance_parser_stacks_owned(constraint, input, 0);
                assert_eq!(
                    SUCCESSES.with(|count| count.get()),
                    2,
                    "owned shared parser primitive must reach the shortcut"
                );
                assert_eq!(owned.semantically_eq(&expected, 65536), Some(true));

                let mut state = constraint.start();
                SUCCESSES.with(|count| count.set(0));
                for _ in 0..3 {
                    let mut mask = vec![0; constraint.mask_len()];
                    state.fill_mask(&mut mask);
                    assert_ne!(mask[0] & 1, 0);
                    state.commit_token(0).unwrap();
                    assert!(state.is_accepting());
                }
                assert!(
                    SUCCESSES.with(|count| count.get()) > 0,
                    "actual public commit path must reach the shortcut, not only direct unit calls"
                );
            }
        }
    }
}
