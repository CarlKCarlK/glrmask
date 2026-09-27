//! Lower ordinary expressions to byte NFAs and share deterministic literal prefixes.

use crate::automata::lexer::ast::Expr;
use crate::automata::lexer::dfa::DFA;
use crate::automata::lexer::nfa::NFA;
use crate::ds::u8set::U8Set;
use rustc_hash::FxHashMap;
use std::collections::HashMap;
use super::bounded_repeat::build_bounded_repeat_dfa;
use super::factor::seq_from_parts;

fn expr_accepts_empty(expr: &Expr) -> bool {
    match expr {
        Expr::U8Seq(bytes) => bytes.is_empty(),
        Expr::U8Class(_) => false,
        Expr::Dfa(dfa) => !dfa.finalizers(0).is_empty(),
        Expr::Intersect { expr, intersect } => {
            expr_accepts_empty(expr) && expr_accepts_empty(intersect)
        }
        Expr::Seq(parts) => parts.iter().all(expr_accepts_empty),
        Expr::Choice(options) => options.iter().any(expr_accepts_empty),
        Expr::Exclude { expr, exclude } => expr_accepts_empty(expr) && !expr_accepts_empty(exclude),
        Expr::Repeat { expr: _, min, .. } => *min == 0,
        Expr::Shared(inner) => expr_accepts_empty(inner),
        Expr::Epsilon => true,
    }
}

pub(crate) fn expr_u8set(expr: &Expr) -> U8Set {
    match expr {
        Expr::U8Seq(bytes) => U8Set::from_bytes(bytes),
        Expr::U8Class(set) => *set,
        Expr::Dfa(dfa) => {
            let mut set = U8Set::empty();
            for state in dfa.states() {
                for (byte, _) in state.transitions.iter() {
                    set.insert(byte);
                }
            }
            set
        }
        Expr::Seq(parts) | Expr::Choice(parts) => parts
            .iter()
            .fold(U8Set::empty(), |acc, part| acc | expr_u8set(part)),
        Expr::Intersect { expr, intersect } => expr_u8set(expr).intersection(&expr_u8set(intersect)),
        Expr::Exclude { expr, .. } => expr_u8set(expr),
        Expr::Repeat { expr, .. } => expr_u8set(expr),
        Expr::Shared(inner) => expr_u8set(inner),
        Expr::Epsilon => U8Set::empty(),
    }
}

fn highest_power_of_two_leq(value: usize) -> usize {
    debug_assert!(value > 0);
    1usize << (usize::BITS - value.leading_zeros() - 1)
}

struct RepeatCompiler<'expr, 'nfa> {
    expr: &'expr Expr,
    nfa: &'nfa mut NFA,
    power_cache: HashMap<(usize, u32), u32>,
    upto_cache: HashMap<(usize, u32), u32>,
}

impl<'expr, 'nfa> RepeatCompiler<'expr, 'nfa> {
    fn new(expr: &'expr Expr, nfa: &'nfa mut NFA) -> Self {
        Self {
            expr,
            nfa,
            power_cache: HashMap::new(),
            upto_cache: HashMap::new(),
        }
    }

    fn compile_power(&mut self, copies: usize, end: u32) -> u32 {
        debug_assert!(copies.is_power_of_two());

        if let Some(&start) = self.power_cache.get(&(copies, end)) {
            return start;
        }

        let start = if copies == 1 {
            let start = self.nfa.add_state();
            append_compiled_expr(self.expr, self.nfa, start, end);
            start
        } else {
            let half = copies / 2;
            let suffix_start = self.compile_power(half, end);
            self.compile_power(half, suffix_start)
        };

        self.power_cache.insert((copies, end), start);
        start
    }

    fn compile_exact(&mut self, copies: usize, end: u32) -> u32 {
        if copies == 0 {
            return end;
        }

        let largest_power = highest_power_of_two_leq(copies);
        let suffix_start = self.compile_exact(copies - largest_power, end);
        self.compile_power(largest_power, suffix_start)
    }

    fn compile_upto(&mut self, copies: usize, end: u32) -> u32 {
        if copies == 0 {
            return end;
        }

        if let Some(&start) = self.upto_cache.get(&(copies, end)) {
            return start;
        }

        let largest_power = highest_power_of_two_leq(copies);
        let split = self.nfa.add_state();

        let smaller_start = self.compile_upto(largest_power - 1, end);
        self.nfa.add_epsilon(split, smaller_start);

        let suffix_start = self.compile_upto(copies - largest_power, end);
        let power_start = self.compile_power(largest_power, suffix_start);
        self.nfa.add_epsilon(split, power_start);

        self.upto_cache.insert((copies, end), split);
        split
    }
}

fn append_byte_sequence_expr(bytes: &[u8], nfa: &mut NFA, start: u32, end: u32) {
    let mut state = start;
    for (index, &byte) in bytes.iter().enumerate() {
        let next = if index + 1 == bytes.len() {
            end
        } else {
            nfa.add_state()
        };
        nfa.add_transition(state, byte, next);
        state = next;
    }

    if bytes.is_empty() {
        nfa.add_epsilon(start, end);
    }
}

fn append_dfa_expr(dfa: &DFA, nfa: &mut NFA, start: u32, end: u32) {
    let mut state_map = Vec::with_capacity(dfa.num_states());
    for _ in 0..dfa.num_states() {
        state_map.push(nfa.add_state());
    }
    nfa.add_epsilon(start, state_map[0]);

    for (state_id, state) in dfa.states().iter().enumerate() {
        let mapped_state = state_map[state_id];
        for (byte, &target) in state.transitions.iter() {
            nfa.add_transition(mapped_state, byte, state_map[target as usize]);
        }
        if !state.finalizers.is_empty() {
            nfa.add_epsilon(mapped_state, end);
        }
    }
}

fn append_sequence_expr(parts: &[Expr], nfa: &mut NFA, start: u32, end: u32) {
    let mut state = start;
    for (index, part) in parts.iter().enumerate() {
        let next = if index + 1 == parts.len() {
            end
        } else {
            nfa.add_state()
        };
        append_compiled_expr(part, nfa, state, next);
        state = next;
    }

    if parts.is_empty() {
        nfa.add_epsilon(start, end);
    }
}

fn append_choice_expr(options: &[Expr], nfa: &mut NFA, start: u32, end: u32) {
    if options.is_empty() {
        nfa.add_epsilon(start, end);
        return;
    }

    for option in options {
        append_compiled_expr(option, nfa, start, end);
    }
}

pub(super) const DIRECT_BOUNDED_REPEAT_THRESHOLD: usize = 32;

pub(super) fn compile_expr_to_dfa(expr: &Expr) -> DFA {
    let mut nfa = build_regex_nfa_impl(std::slice::from_ref(expr));
    nfa.condense_epsilon_sccs();
    nfa.to_minimized_dfa()
}

fn append_bounded_repeat_expr(expr: &Expr, min: usize, max: usize, nfa: &mut NFA, start: u32, end: u32) {
    if max < min {
        return;
    }

    if let Some(dfa) = build_bounded_repeat_dfa(expr, min, max) {
        append_dfa_expr(&dfa, nfa, start, end);
        return;
    }

    let mut repeat_compiler = RepeatCompiler::new(expr, nfa);
    let optional = max - min;
    let tail_start = repeat_compiler.compile_upto(optional, end);
    let repeat_start = repeat_compiler.compile_exact(min, tail_start);
    repeat_compiler.nfa.add_epsilon(start, repeat_start);
}

fn append_unbounded_repeat_expr(
    expr: &Expr,
    min: usize,
    nfa: &mut NFA,
    start: u32,
    end: u32,
) {
    let mut current = start;
    for _ in 0..min {
        let next = nfa.add_state();
        append_compiled_expr(expr, nfa, current, next);
        current = next;
    }

    if current == start {
        let fresh = nfa.add_state();
        nfa.add_epsilon(start, fresh);
        current = fresh;
    }

    nfa.add_epsilon(current, end);
    let loop_state = nfa.add_state();
    append_compiled_expr(expr, nfa, current, loop_state);
    nfa.add_epsilon(loop_state, current);
    if expr_accepts_empty(expr) {
        nfa.add_epsilon(loop_state, end);
    }
}

pub(super) fn append_compiled_expr(expr: &Expr, nfa: &mut NFA, start: u32, end: u32) {
    match expr {
        Expr::U8Seq(bytes) => append_byte_sequence_expr(bytes, nfa, start, end),
        Expr::U8Class(set) => {
            nfa.add_u8set_transition(start, *set, end);
        }
        Expr::Dfa(dfa) => append_dfa_expr(dfa, nfa, start, end),
        Expr::Intersect { .. } => {
            unreachable!("nested Expr::Intersect must be lowered before NFA compilation")
        }
        Expr::Seq(parts) => append_sequence_expr(parts, nfa, start, end),
        Expr::Choice(options) => append_choice_expr(options, nfa, start, end),
        Expr::Exclude { .. } => {
            unreachable!("nested Expr::Exclude must be lowered before NFA compilation")
        }
        Expr::Repeat { expr, min, max } => match max {
            Some(max) => append_bounded_repeat_expr(expr, *min, *max, nfa, start, end),
            None => append_unbounded_repeat_expr(expr, *min, nfa, start, end),
        },
        Expr::Shared(inner) => append_compiled_expr(inner, nfa, start, end),
        Expr::Epsilon => nfa.add_epsilon(start, end),
    }
}

/// Compile multiple expressions into a single NFA (without determinization).
///
/// Each expression's index becomes its group ID. Equal literal/class prefixes
/// are shared recursively. The previous implementation factored at most one
/// prefix common to *every* terminal, which left large families of line-like
/// terminals as thousands of duplicated suffix chains after their first byte.
pub(super) fn build_regex_nfa(exprs: &[Expr]) -> NFA {
    build_regex_nfa_impl(exprs)
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum DeterministicPrefixAtom {
    Byte(u8),
    Class(U8Set),
}

fn deterministic_prefix(expr: &Expr) -> (Vec<DeterministicPrefixAtom>, Expr) {
    fn collect(expr: &Expr, atoms: &mut Vec<DeterministicPrefixAtom>) -> Expr {
        match expr {
            Expr::Shared(inner) => collect(inner, atoms),
            Expr::U8Seq(bytes) => {
                atoms.extend(bytes.iter().copied().map(DeterministicPrefixAtom::Byte));
                Expr::Epsilon
            }
            Expr::U8Class(bytes) => {
                atoms.push(DeterministicPrefixAtom::Class(*bytes));
                Expr::Epsilon
            }
            Expr::Seq(parts) => {
                for (index, part) in parts.iter().enumerate() {
                    let remainder = collect(part, atoms);
                    if !matches!(remainder, Expr::Epsilon) {
                        let mut tail = Vec::with_capacity(parts.len() - index);
                        tail.push(remainder);
                        tail.extend_from_slice(&parts[index + 1..]);
                        return seq_from_parts(tail);
                    }
                }
                Expr::Epsilon
            }
            Expr::Dfa(_)
            | Expr::Intersect { .. }
            | Expr::Choice(_)
            | Expr::Exclude { .. }
            | Expr::Repeat { .. }
            | Expr::Epsilon => expr.clone(),
        }
    }

    let mut atoms = Vec::new();
    let remainder = collect(expr, &mut atoms);
    (atoms, remainder)
}

fn append_prefix_shared_expressions(
    nfa: &mut NFA,
    expressions: Vec<(u32, Expr)>,
) {
    let mut transitions = FxHashMap::<(u32, DeterministicPrefixAtom), u32>::default();
    let mut finals = Vec::<(u32, u32)>::new();
    let mut opaque = Vec::<(u32, u32, Expr)>::new();

    for (group_id, expr) in expressions {
        let (prefix, remainder) = deterministic_prefix(&expr);
        let mut state = 0u32;
        for atom in prefix {
            let key = (state, atom.clone());
            state = if let Some(&target) = transitions.get(&key) {
                target
            } else {
                let target = nfa.add_state();
                match &atom {
                    DeterministicPrefixAtom::Byte(byte) => {
                        append_compiled_expr(&Expr::U8Seq(vec![*byte]), nfa, state, target);
                    }
                    DeterministicPrefixAtom::Class(bytes) => {
                        append_compiled_expr(&Expr::U8Class(*bytes), nfa, state, target);
                    }
                }
                transitions.insert(key, target);
                target
            };
        }
        if matches!(remainder, Expr::Epsilon) {
            finals.push((state, group_id));
        } else {
            opaque.push((state, group_id, remainder));
        }
    }

    for (state, group_id) in finals {
        nfa.add_finalizer(state, group_id);
    }
    for (state, group_id, expr) in opaque {
        let accept = nfa.add_state();
        append_compiled_expr(&expr, nfa, state, accept);
        nfa.add_finalizer(accept, group_id);
    }
}

fn build_regex_nfa_impl(exprs: &[Expr]) -> NFA {
    let optimized_exprs = exprs
        .iter()
        .cloned()
        .map(Expr::optimize)
        .enumerate()
        .map(|(group_id, expr)| (group_id as u32, expr))
        .collect::<Vec<_>>();

    let mut nfa = NFA::new(1);
    append_prefix_shared_expressions(&mut nfa, optimized_exprs);
    nfa
}
