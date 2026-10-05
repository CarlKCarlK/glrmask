//! Native finite-program construction.
//!
//! Source DEFAULT is wildcard union. The constructed deterministic graph uses
//! explicit-before-DEFAULT fallback. Fallback-aware bottom-up reduction retains
//! explicit dead exceptions. The split result is subsequently subjected to the
//! shared mandatory ValidatedTemplate preparation by its caller.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::Arc;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::automata::unweighted_u32::{
    dfa::{DFA, DFAState},
    determinize::determinize,
    minimize_acyclic::minimize_acyclic,
    nfa::NFA,
};
use crate::compiler::glr::labels::{
    DEFAULT_LABEL, encode_negative_label, is_negative_label,
};
use crate::runtime::CommitTemplateDfas;

use super::super::characterize::{StackMatcher, TerminalCharacterization};
use super::super::compile_dfa::{
    Templates, specialize_template_dfa_defaults_for_commit_split_input,
    try_split_commit_template_dfas,
};

pub struct ProgramCompiler {
    validate: bool,
    minimize: bool,
}

impl Default for ProgramCompiler {
    fn default() -> Self {
        Self::new()
    }
}

impl ProgramCompiler {
    pub fn new() -> Self {
        Self {
            validate: super::super::compiler_template_validation_enabled()
                || super::flag("GLRMASK_VALIDATE_TEMPLATE_QUOTIENT"),
            minimize: !super::flag("GLRMASK_SKIP_TEMPLATE_MINIMIZE"),
        }
    }

    pub fn compile(
        &self,
        characterization: &TerminalCharacterization,
    ) -> Result<CommitTemplateDfas, String> {
        let nfa = compact_nfa(characterization)?;
        validate_nfa(&nfa)?;

        let input_edges = nfa.states.iter().map(|state| {
            state.epsilons.len()
                + state.transitions.values().map(Vec::len).sum::<usize>()
        }).sum::<usize>();
        let budget = nfa.states.len().saturating_add(input_edges)
            .saturating_mul(32).max(4_096);

        let specialized = match determinize_union(&nfa, budget) {
            Ok(dfa) => {
                if self.minimize {
                    let reduced = reduce_fallback_acyclic(&dfa)?;
                    if self.validate {
                        assert_eq!(
                            fallback_mismatch(&dfa, &reduced), None,
                            "fallback-aware acyclic reduction changed the action language"
                        );
                    }
                    reduced
                } else {
                    dfa
                }
            }
            Err(AttemptError::Budget) => {
                // A bounded speculative optimization, not an acceptance bound.
                // Reuse the same compact NFA and synchronously run the exact
                // staged automata algorithm. No LR runtime or truncation exists.
                let raw = determinize(&nfa);
                let raw = if self.minimize { minimize_acyclic(&raw) } else { raw };
                specialize_template_dfa_defaults_for_commit_split_input(&raw)
            }
        };

        if self.validate {
            // Independent established builder, including its own NFA and raw
            // minimization. This work is deliberately absent in normal builds.
            let input = BTreeMap::from([(0, characterization.clone())]);
            let raw = Templates::dfas_from_characterizations(&input)
                .remove(&0).expect("reference singleton template");
            let reference = specialize_template_dfa_defaults_for_commit_split_input(&raw);
            assert_eq!(
                fallback_mismatch(&reference, &specialized), None,
                "native fused compiler changed the complete concrete action language"
            );
        }
        let split = dense_split(&specialized)?;
        if self.validate {
            let reference = try_split_commit_template_dfas(&specialized)
                .expect("native split input must satisfy established phase discipline");
            assert_same_split(&split, &reference);
        }
        Ok(split)
    }

    /// Consume retained Static raw inputs without re-characterizing them.
    ///
    /// If DEFAULT union is structurally scalar, specialization is only a
    /// reachable renumbering. The splitter's deterministic discovery order is
    /// invariant under that renumbering, so no specialized graph is needed.
    pub fn compile_raw(&self, raw: &DFA) -> Result<CommitTemplateDfas, String> {
        topological(raw)?;
        let scalar = raw.states.iter().all(|state| {
            state.transitions.get(&DEFAULT_LABEL).is_none_or(|default| {
                state.transitions.iter().all(|(&label, target)| {
                    label == DEFAULT_LABEL || label < 0 || target == default
                })
            })
        });
        let split = if scalar {
            dense_split(raw)?
        } else {
            let specialized =
                specialize_template_dfa_defaults_for_commit_split_input(raw);
            dense_split(&specialized)?
        };
        if self.validate {
            let specialized =
                specialize_template_dfa_defaults_for_commit_split_input(raw);
            let reference = try_split_commit_template_dfas(&specialized)
                .ok_or("retained template violates POP/READ/PUSH phase discipline")?;
            assert_same_split(&split, &reference);
        }
        Ok(split)
    }
}

fn checked_state(nfa: &mut NFA) -> Result<u32, String> {
    if nfa.states.len() >= u32::MAX as usize {
        return Err("native template NFA exceeds its state coordinate".into());
    }
    Ok(nfa.add_state())
}

fn check_symbol(symbol: u32) -> Result<(), String> {
    if symbol >= DEFAULT_LABEL as u32 {
        return Err(format!("template stack symbol {symbol} uses a reserved coordinate"));
    }
    Ok(())
}

fn matcher_edges(
    nfa: &mut NFA, from: u32, matcher: &StackMatcher, target: u32,
) -> Result<(), String> {
    match matcher {
        StackMatcher::Any => nfa.add_transition(from, DEFAULT_LABEL, target),
        StackMatcher::State(symbol) => {
            check_symbol(*symbol)?;
            nfa.add_transition(from, *symbol as i32, target);
        }
        StackMatcher::States(symbols) => {
            for &symbol in symbols {
                check_symbol(symbol)?;
                nfa.add_transition(from, symbol as i32, target);
            }
        }
    }
    Ok(())
}

struct Builder<'a> {
    nfa: NFA,
    accept: u32,
    pushes: FxHashMap<(u32, u32), u32>,
    pops: FxHashMap<(u32, &'a StackMatcher), u32>,
}

impl<'a> Builder<'a> {
    fn push_tail(&mut self, pushes: &[u32]) -> Result<u32, String> {
        let mut target = self.accept;
        for &symbol in pushes.iter().rev() {
            check_symbol(symbol)?;
            let key = (target, symbol);
            target = if let Some(&state) = self.pushes.get(&key) {
                state
            } else {
                let state = checked_state(&mut self.nfa)?;
                self.nfa.add_transition(state, encode_negative_label(symbol), target);
                self.pushes.insert(key, state);
                state
            };
        }
        Ok(target)
    }

    fn pop_path(
        &mut self, source: u32, pop: &'a [StackMatcher], mut target: u32,
    ) -> Result<(), String> {
        let Some((first, suffix)) = pop.split_first() else {
            self.nfa.add_epsilon(source, target);
            return Ok(());
        };
        for matcher in suffix.iter().rev() {
            let key = (target, matcher);
            target = if let Some(&state) = self.pops.get(&key) {
                state
            } else {
                let state = checked_state(&mut self.nfa)?;
                matcher_edges(&mut self.nfa, state, matcher, target)?;
                self.pops.insert(key, state);
                state
            };
        }
        matcher_edges(&mut self.nfa, source, first, target)
    }

    fn escape(
        &mut self, source: u32, pop: &'a [StackMatcher], pushes: &[u32],
    ) -> Result<(), String> {
        let tail = self.push_tail(pushes)?;
        self.pop_path(source, pop, tail)
    }
}

fn compact_nfa(c: &TerminalCharacterization) -> Result<NFA, String> {
    let mut nfa = NFA::new();
    let mut nts = BTreeMap::new();
    for &nt in &c.all_nts {
        nts.insert(nt, checked_state(&mut nfa)?);
    }
    let accept = checked_state(&mut nfa)?;
    nfa.set_accepting(accept);
    let mut builder = Builder {
        nfa,
        accept,
        pushes: FxHashMap::default(),
        pops: FxHashMap::default(),
    };
    for escape in &c.escapes {
        builder.escape(0, &escape.pop, &escape.pushes)?;
    }
    for reduce in &c.reduces {
        if let Some(&target) = nts.get(&reduce.nonterminal) {
            builder.pop_path(0, &reduce.pop, target)?;
        }
    }
    for escape in &c.nt_escapes {
        if let Some(&source) = nts.get(&escape.source_nonterminal) {
            builder.escape(source, &escape.pop, &escape.pushes)?;
        }
    }
    for reduce in &c.nt_rereduces {
        if let (Some(&source), Some(&target)) = (
            nts.get(&reduce.source_nonterminal),
            nts.get(&reduce.target_nonterminal),
        ) {
            builder.pop_path(source, &reduce.pop, target)?;
        }
    }
    Ok(builder.nfa)
}

fn validate_nfa(nfa: &NFA) -> Result<(), String> {
    if nfa.start_states.iter().any(|&state| state as usize >= nfa.states.len())
        || nfa.states.iter().any(|state| {
            state.epsilons.iter().chain(state.transitions.values().flatten())
                .any(|&target| target as usize >= nfa.states.len())
        })
    {
        return Err("native template NFA contains an invalid state reference".into());
    }
    if !nfa.compute_is_acyclic() {
        return Err("native template NFA is cyclic".into());
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AttemptError {
    Budget,
}

fn spend(left: &mut usize, work: usize) -> Result<(), AttemptError> {
    *left = left.checked_sub(work).ok_or(AttemptError::Budget)?;
    Ok(())
}

struct Closure {
    marks: Vec<u32>,
    epoch: u32,
    work: Vec<u32>,
}

impl Closure {
    fn new(states: usize) -> Self {
        Self { marks: vec![0; states], epoch: 0, work: Vec::new() }
    }

    fn close(
        &mut self, nfa: &NFA, seeds: &[u32], budget: &mut usize,
    ) -> Result<Vec<u32>, AttemptError> {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.marks.fill(0);
            self.epoch = 1;
        }
        self.work.clear();
        spend(budget, seeds.len())?;
        for &state in seeds {
            if self.marks[state as usize] != self.epoch {
                self.marks[state as usize] = self.epoch;
                self.work.push(state);
            }
        }
        let mut head = 0;
        while head < self.work.len() {
            let state = self.work[head] as usize;
            spend(budget, 1 + nfa.states[state].epsilons.len())?;
            for &target in &nfa.states[state].epsilons {
                if self.marks[target as usize] != self.epoch {
                    self.marks[target as usize] = self.epoch;
                    self.work.push(target);
                }
            }
            head += 1;
        }
        self.work.sort_unstable();
        Ok(self.work.clone())
    }
}

fn union_sorted(
    left: &[u32], right: &[u32], budget: &mut usize,
) -> Result<Vec<u32>, AttemptError> {
    spend(budget, left.len().saturating_add(right.len()))?;
    let mut out = Vec::with_capacity(left.len().saturating_add(right.len()));
    let (mut i, mut j) = (0, 0);
    while i < left.len() && j < right.len() {
        match left[i].cmp(&right[j]) {
            std::cmp::Ordering::Less => {
                out.push(left[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                out.push(right[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                out.push(left[i]);
                i += 1;
                j += 1;
            }
        }
    }
    out.extend_from_slice(&left[i..]);
    out.extend_from_slice(&right[j..]);
    Ok(out)
}

fn intern_subset(
    key: Vec<u32>,
    ids: &mut FxHashMap<Arc<[u32]>, u32>,
    subsets: &mut Vec<Arc<[u32]>>,
    dfa: &mut DFA,
    budget: &mut usize,
) -> Result<u32, AttemptError> {
    spend(budget, key.len().saturating_add(1))?;
    if let Some(&id) = ids.get(key.as_slice()) {
        return Ok(id);
    }
    if dfa.states.len() >= u32::MAX as usize {
        return Err(AttemptError::Budget);
    }
    let key: Arc<[u32]> = key.into();
    let id = dfa.add_state();
    ids.insert(Arc::clone(&key), id);
    subsets.push(key);
    Ok(id)
}

fn determinize_union(nfa: &NFA, mut budget: usize) -> Result<DFA, AttemptError> {
    let mut closure = Closure::new(nfa.states.len());
    let initial = closure.close(nfa, &nfa.start_states, &mut budget)?;
    let mut dfa = DFA::default();
    let mut ids = FxHashMap::<Arc<[u32]>, u32>::default();
    let mut subsets = Vec::new();
    dfa.start_state = intern_subset(initial, &mut ids, &mut subsets, &mut dfa, &mut budget)?;

    let mut buckets = FxHashMap::<i32, Vec<u32>>::default();
    let mut labels = Vec::new();
    let mut defaults = Vec::new();
    let mut head = 0;
    while head < subsets.len() {
        let subset = Arc::clone(&subsets[head]);
        let source = head as u32;
        dfa.states[head].is_accepting =
            subset.iter().any(|&state| nfa.states[state as usize].is_accepting);
        defaults.clear();
        for &state in subset.iter() {
            spend(&mut budget, 1)?;
            for (&label, targets) in &nfa.states[state as usize].transitions {
                spend(&mut budget, targets.len().saturating_add(1))?;
                if targets.is_empty() {
                    continue;
                }
                if label == DEFAULT_LABEL {
                    defaults.extend_from_slice(targets);
                } else {
                    let bucket = buckets.entry(label).or_default();
                    if bucket.is_empty() {
                        labels.push(label);
                    }
                    bucket.extend_from_slice(targets);
                }
            }
        }

        let default_closure = closure.close(nfa, &defaults, &mut budget)?;
        labels.sort_unstable();
        if !default_closure.is_empty() {
            labels.push(DEFAULT_LABEL);
            labels.sort_unstable();
        }
        for &label in &labels {
            let key = if label == DEFAULT_LABEL {
                default_closure.clone()
            } else {
                let explicit = closure.close(
                    nfa, buckets.get(&label).expect("touched label"), &mut budget,
                )?;
                if label >= 0 && !default_closure.is_empty() {
                    union_sorted(&explicit, &default_closure, &mut budget)?
                } else {
                    explicit
                }
            };
            if key.is_empty() {
                continue;
            }
            let target = intern_subset(
                key, &mut ids, &mut subsets, &mut dfa, &mut budget,
            )?;
            dfa.add_transition(source, label, target);
        }
        for label in labels.drain(..) {
            if label != DEFAULT_LABEL {
                buckets.get_mut(&label).expect("persistent label bucket").clear();
            }
        }
        head += 1;
    }
    Ok(dfa)
}

fn topological(dfa: &DFA) -> Result<Vec<usize>, String> {
    if dfa.states.is_empty() {
        return if dfa.start_state == 0 {
            Ok(Vec::new())
        } else {
            Err("empty native DFA has nonzero start".into())
        };
    }
    if dfa.start_state as usize >= dfa.states.len() {
        return Err("native DFA start lies outside its graph".into());
    }
    let mut incoming = vec![0usize; dfa.states.len()];
    for state in &dfa.states {
        for &target in state.transitions.values() {
            let Some(degree) = incoming.get_mut(target as usize) else {
                return Err("native DFA edge lies outside its graph".into());
            };
            *degree = degree.checked_add(1).ok_or("native DFA indegree overflow")?;
        }
    }
    let mut order = incoming.iter().enumerate()
        .filter_map(|(state, &degree)| (degree == 0).then_some(state))
        .collect::<Vec<_>>();
    let mut head = 0;
    while head < order.len() {
        for &target in dfa.states[order[head]].transitions.values() {
            incoming[target as usize] -= 1;
            if incoming[target as usize] == 0 {
                order.push(target as usize);
            }
        }
        head += 1;
    }
    if order.len() != dfa.states.len() {
        return Err("native template DFA is cyclic".into());
    }
    Ok(order)
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct Signature {
    accepting: bool,
    default: u32,
    edges: Vec<(i32, u32)>,
}

fn reduce_fallback_acyclic(dfa: &DFA) -> Result<DFA, String> {
    let order = topological(dfa)?;
    if order.is_empty() {
        return Ok(DFA::new());
    }
    let dead = Signature { accepting: false, default: 0, edges: Vec::new() };
    let mut ids = FxHashMap::default();
    ids.insert(dead.clone(), 0u32);
    let mut rows = vec![dead];
    let mut classes = vec![0u32; dfa.states.len()];

    for &source in order.iter().rev() {
        let row = &dfa.states[source];
        let default = row.transitions.get(&DEFAULT_LABEL)
            .map_or(0, |&target| classes[target as usize]);
        let edges = row.transitions.iter().filter_map(|(&label, &target)| {
            if label == DEFAULT_LABEL {
                return None;
            }
            let target = classes[target as usize];
            // Negative missing edges reject. Concrete missing edges fall back.
            // Keep productive explicit edges even when equal to DEFAULT, since
            // they may enable a cheap exact READ compression.
            if target == 0 && (label < 0 || default == 0) {
                None
            } else {
                Some((label, target))
            }
        }).collect();
        let signature = Signature {
            accepting: row.is_accepting,
            default,
            edges,
        };
        let class = if let Some(&class) = ids.get(&signature) {
            class
        } else {
            let class = u32::try_from(rows.len())
                .map_err(|_| "native template class coordinate overflow")?;
            ids.insert(signature.clone(), class);
            rows.push(signature);
            class
        };
        classes[source] = class;
    }

    let start = classes[dfa.start_state as usize];
    if start == 0 {
        return Ok(DFA::new());
    }
    let needs_dead = rows.iter().skip(1)
        .any(|row| row.edges.iter().any(|&(_, target)| target == 0));
    let good_count = rows.len() - 1;
    let dead_id = u32::try_from(good_count)
        .map_err(|_| "native template state coordinate overflow")?;
    let mut result = DFA {
        states: vec![DFAState::default(); good_count + usize::from(needs_dead)],
        start_state: start - 1,
    };
    for (class, row) in rows.into_iter().enumerate().skip(1) {
        let state = &mut result.states[class - 1];
        state.is_accepting = row.accepting;
        for (label, target) in row.edges {
            state.transitions.insert(
                label, if target == 0 { dead_id } else { target - 1 },
            );
        }
        if row.default != 0 {
            state.transitions.insert(DEFAULT_LABEL, row.default - 1);
        }
    }
    Ok(result)
}

fn ensure(
    old: u32, original: &DFA, graph: &mut DFA, mapping: &mut [Option<u32>],
    copy_accepting: bool,
) -> u32 {
    if let Some(state) = mapping[old as usize] {
        return state;
    }
    let state = graph.add_state();
    if copy_accepting {
        graph.states[state as usize].is_accepting =
            original.states[old as usize].is_accepting;
    }
    mapping[old as usize] = Some(state);
    state
}

fn enqueue(
    state: u32, phase: usize, marks: &mut [u8], queue: &mut VecDeque<(u32, usize)>,
) {
    let bit = 1u8 << phase;
    if marks[state as usize] & bit == 0 {
        marks[state as usize] |= bit;
        queue.push_back((state, phase));
    }
}

fn dense_split(dfa: &DFA) -> Result<CommitTemplateDfas, String> {
    if dfa.states.is_empty() {
        return Ok(CommitTemplateDfas {
            pop: DFA::new(),
            ..CommitTemplateDfas::default()
        });
    }
    if dfa.start_state as usize >= dfa.states.len()
        || dfa.states.iter().any(|state| {
            state.transitions.values().any(|&target| target as usize >= dfa.states.len())
        })
    {
        return Err("native split input has invalid state references".into());
    }
    let n = dfa.states.len();
    let mut result = CommitTemplateDfas::default();
    let mut pop = vec![None; n];
    let mut push = vec![None; n];
    let mut read_sources = vec![None; n];
    let mut read_targets = vec![None; n];
    result.pop.start_state = ensure(
        dfa.start_state, dfa, &mut result.pop, &mut pop, true,
    );
    let mut marks = vec![0u8; n];
    let mut queue = VecDeque::new();
    enqueue(dfa.start_state, 0, &mut marks, &mut queue);
    let mut dead_pop = None;

    while let Some((old_id, phase)) = queue.pop_front() {
        let old = &dfa.states[old_id as usize];
        if phase == 0 {
            let source = ensure(old_id, dfa, &mut result.pop, &mut pop, true);
            result.pop_to_read.resize(result.pop.states.len(), None);
            result.pop_to_push.resize(result.pop.states.len(), None);
            for (&label, &target) in &old.transitions {
                if is_negative_label(label) {
                    let entry = ensure(old_id, dfa, &mut result.push, &mut push, true);
                    result.pop_to_push[source as usize] = Some(entry);
                    enqueue(old_id, 1, &mut marks, &mut queue);
                    continue;
                }
                let target_row = &dfa.states[target as usize];
                let read_target = if label != DEFAULT_LABEL
                    && label >= 0
                    && !target_row.is_accepting
                    && target_row.transitions.len() == 1
                {
                    target_row.transitions.first_key_value().and_then(|(&pushed, &next)| {
                        (pushed == encode_negative_label(label as u32)).then_some(next)
                    })
                } else {
                    None
                };
                if let Some(after) = read_target {
                    let read_source = ensure(
                        old_id, dfa, &mut result.read, &mut read_sources, false,
                    );
                    let read_target = ensure(
                        after, dfa, &mut result.read, &mut read_targets, false,
                    );
                    result.read_to_push.resize(result.read.states.len(), None);
                    result.pop_to_read[source as usize] = Some(read_source);
                    result.read.add_transition(read_source, label, read_target);
                    if old.transitions.contains_key(&DEFAULT_LABEL) {
                        let dead = *dead_pop.get_or_insert_with(|| result.pop.add_state());
                        result.pop_to_read.resize(result.pop.states.len(), None);
                        result.pop_to_push.resize(result.pop.states.len(), None);
                        result.pop.add_transition(source, label, dead);
                    }
                    let entry = ensure(after, dfa, &mut result.push, &mut push, true);
                    result.read_to_push[read_target as usize] = Some(entry);
                    enqueue(after, 2, &mut marks, &mut queue);
                } else {
                    let next = ensure(target, dfa, &mut result.pop, &mut pop, true);
                    result.pop_to_read.resize(result.pop.states.len(), None);
                    result.pop_to_push.resize(result.pop.states.len(), None);
                    result.pop.add_transition(source, label, next);
                    enqueue(target, 0, &mut marks, &mut queue);
                }
            }
        } else {
            let source = ensure(old_id, dfa, &mut result.push, &mut push, true);
            for (&label, &target) in &old.transitions {
                if !is_negative_label(label) {
                    if phase == 1 {
                        continue;
                    }
                    return Err("template violates POP/READ/PUSH phase discipline".into());
                }
                let next = ensure(target, dfa, &mut result.push, &mut push, true);
                result.push.add_transition(source, label, next);
                enqueue(target, 2, &mut marks, &mut queue);
            }
        }
    }
    Ok(result)
}

fn assert_same_split(left: &CommitTemplateDfas, right: &CommitTemplateDfas) {
    assert_eq!(left.pop, right.pop);
    assert_eq!(left.read, right.read);
    assert_eq!(left.push, right.push);
    assert_eq!(left.pop_to_read, right.pop_to_read);
    assert_eq!(left.pop_to_push, right.pop_to_push);
    assert_eq!(left.read_to_push, right.read_to_push);
}

fn accepting(dfa: &DFA, state: Option<u32>) -> bool {
    state.and_then(|state| dfa.states.get(state as usize))
        .is_some_and(|state| state.is_accepting)
}

fn advance(dfa: &DFA, state: Option<u32>, label: i32) -> Option<u32> {
    let row = dfa.states.get(state? as usize)?;
    row.transitions.get(&label).copied().or_else(|| {
        (label >= 0).then(|| row.transitions.get(&DEFAULT_LABEL).copied()).flatten()
    })
}

/// Complete finite product over fallback DFAs. One fresh concrete positive
/// label represents every locally unmentioned stack symbol.
fn fallback_mismatch(left: &DFA, right: &DFA) -> Option<Vec<i32>> {
    let start = (Some(left.start_state), Some(right.start_state));
    let mut seen = FxHashSet::default();
    seen.insert(start);
    let mut nodes = vec![(start, None::<(usize, i32)>)];
    let mut head = 0;
    while head < nodes.len() {
        let ((a, b), _) = nodes[head];
        if accepting(left, a) != accepting(right, b) {
            let mut word = Vec::new();
            let mut at = head;
            while let Some((parent, label)) = nodes[at].1 {
                word.push(label);
                at = parent;
            }
            word.reverse();
            return Some(word);
        }
        let mut labels = BTreeSet::new();
        for (graph, state) in [(left, a), (right, b)] {
            if let Some(row) = state.and_then(|state| graph.states.get(state as usize)) {
                labels.extend(row.transitions.keys().copied()
                    .filter(|&label| label != DEFAULT_LABEL));
            }
        }
        let mut other = 0i32;
        while labels.contains(&other) || other == DEFAULT_LABEL {
            other = other.checked_add(1)
                .expect("finite graph leaves a concrete symbol available");
        }
        labels.insert(other);
        for label in labels {
            let next = (advance(left, a, label), advance(right, b, label));
            if seen.insert(next) {
                nodes.push((next, Some((head, label))));
            }
        }
        head += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::templates::admissibility::TemplateDomain;
    use crate::templates::characterize::{
        InitialEscape, InitialReduce, NtEscape, NtRereduce,
    };

    #[test]
    fn fallback_reduction_keeps_explicit_dead_shadow() {
        let mut dfa = DFA::new();
        let yes = dfa.add_state();
        let no = dfa.add_state();
        dfa.set_accepting(yes, true);
        dfa.add_transition(0, DEFAULT_LABEL, yes);
        dfa.add_transition(0, 7, no);
        let reduced = reduce_fallback_acyclic(&dfa).unwrap();
        assert_eq!(fallback_mismatch(&dfa, &reduced), None);
        let root = &reduced.states[reduced.start_state as usize];
        assert!(root.transitions.contains_key(&7));
        let split = dense_split(&reduced).unwrap();
        let domain = TemplateDomain::compile(&split).unwrap();
        assert!(!domain.matches_top_first([7]));
        assert!(domain.matches_top_first([8]));
    }

    #[test]
    fn generated_compact_programs_match_complete_reference_products() {
        let compiler = ProgramCompiler { validate: true, minimize: true };
        for seed in 0..96u32 {
            let c = TerminalCharacterization {
                escapes: vec![
                    InitialEscape {
                        pop: vec![StackMatcher::State(seed % 7)],
                        pushes: vec![seed % 5, (seed + 1) % 5],
                    },
                    InitialEscape {
                        pop: vec![StackMatcher::Any, StackMatcher::States(vec![1, 3, 6])],
                        pushes: vec![seed % 3],
                    },
                    InitialEscape {
                        pop: Vec::new(),
                        pushes: if seed % 3 == 0 { Vec::new() } else { vec![2] },
                    },
                ],
                reduces: vec![InitialReduce {
                    pop: vec![StackMatcher::State((seed + 2) % 7)],
                    nonterminal: 0,
                }],
                nt_escapes: vec![NtEscape {
                    source_nonterminal: 1,
                    pop: vec![StackMatcher::State(seed % 7)],
                    pushes: vec![4],
                }],
                nt_rereduces: vec![NtRereduce {
                    source_nonterminal: 0,
                    pop: vec![StackMatcher::Any],
                    target_nonterminal: 1,
                }],
                all_nts: BTreeSet::from([0, 1]),
            };
            let program = compiler.compile(&c).unwrap();
            TemplateDomain::compile(&program).unwrap();
        }
    }

    #[test]
    fn speculative_budget_is_not_a_language_or_acceptance_bound() {
        let c = TerminalCharacterization {
            escapes: vec![
                InitialEscape { pop: vec![StackMatcher::Any], pushes: vec![1] },
                InitialEscape { pop: vec![StackMatcher::State(2)], pushes: vec![2] },
            ],
            reduces: Vec::new(),
            nt_escapes: Vec::new(),
            nt_rereduces: Vec::new(),
            all_nts: BTreeSet::new(),
        };
        let nfa = compact_nfa(&c).unwrap();
        assert_eq!(determinize_union(&nfa, 0), Err(AttemptError::Budget));
        let direct = determinize_union(&nfa, 100_000).unwrap();
        let staged = specialize_template_dfa_defaults_for_commit_split_input(
            &minimize_acyclic(&determinize(&nfa))
        );
        assert_eq!(fallback_mismatch(&direct, &staged), None);
    }

    #[test]
    fn wildcard_union_never_applies_to_signed_pushes() {
        let mut nfa = NFA::new();
        let yes = nfa.add_state();
        nfa.set_accepting(yes);
        nfa.add_transition(0, DEFAULT_LABEL, yes);
        let dfa = determinize_union(&nfa, 1_000).unwrap();
        assert_eq!(advance(&dfa, Some(dfa.start_state), encode_negative_label(0)), None);
        assert!(accepting(&dfa, advance(&dfa, Some(dfa.start_state), 0)));
    }

    #[test]
    fn dense_split_matches_established_split_and_default_blockers() {
        for top in 0..32u32 {
            let mut dfa = DFA::new();
            let restore = dfa.add_state();
            let tail = dfa.add_state();
            let yes = dfa.add_state();
            let fallback = dfa.add_state();
            dfa.add_transition(0, top as i32, restore);
            dfa.add_transition(restore, encode_negative_label(top), tail);
            dfa.add_transition(tail, encode_negative_label(63), yes);
            dfa.add_transition(0, DEFAULT_LABEL, fallback);
            dfa.add_transition(fallback, encode_negative_label(62), yes);
            dfa.set_accepting(yes, true);
            let actual = dense_split(&dfa).unwrap();
            let reference = try_split_commit_template_dfas(&dfa).unwrap();
            assert_same_split(&actual, &reference);
            let root = &actual.pop.states[actual.pop.start_state as usize];
            let dead = root.transitions[&(top as i32)] as usize;
            assert!(!actual.pop.states[dead].is_accepting);
            assert!(actual.pop.states[dead].transitions.is_empty());
        }
    }

    #[test]
    fn retained_scalar_path_avoids_specialization_without_changing_split_layout() {
        let compiler = ProgramCompiler { validate: true, minimize: true };
        for top in 0..16u32 {
            let mut dfa = DFA::new();
            let restore = dfa.add_state();
            let yes = dfa.add_state();
            dfa.add_transition(0, top as i32, restore);
            dfa.add_transition(0, DEFAULT_LABEL, restore);
            dfa.add_transition(restore, encode_negative_label(top), yes);
            dfa.set_accepting(yes, true);
            let actual = compiler.compile_raw(&dfa).unwrap();
            let reference = try_split_commit_template_dfas(
                &specialize_template_dfa_defaults_for_commit_split_input(&dfa)
            ).unwrap();
            assert_same_split(&actual, &reference);
        }
    }

    #[test]
    fn invalid_targets_cycles_and_post_push_pop_are_rejected() {
        let compiler = ProgramCompiler { validate: false, minimize: true };
        let mut bad = DFA::new();
        bad.add_transition(0, 0, 99);
        assert!(compiler.compile_raw(&bad).is_err());
        let mut cyclic = DFA::new();
        cyclic.add_transition(0, 0, 0);
        assert!(compiler.compile_raw(&cyclic).is_err());
        let mut phases = DFA::new();
        let pushed = phases.add_state();
        let yes = phases.add_state();
        phases.add_transition(0, encode_negative_label(1), pushed);
        phases.add_transition(pushed, 2, yes);
        phases.set_accepting(yes, true);
        assert!(compiler.compile_raw(&phases).is_err());
    }

    #[test]
    fn acyclic_reduction_is_iterative_on_deep_inputs() {
        let mut dfa = DFA::new();
        let mut at = 0;
        for _ in 0..30_000 {
            let next = dfa.add_state();
            dfa.add_transition(at, DEFAULT_LABEL, next);
            at = next;
        }
        dfa.set_accepting(at, true);
        let reduced = reduce_fallback_acyclic(&dfa).unwrap();
        assert_eq!(reduced.states.len(), 30_001);
        assert!(reduced.compute_is_acyclic());
    }
}
