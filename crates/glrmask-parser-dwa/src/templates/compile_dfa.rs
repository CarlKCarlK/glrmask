//! Template-DFA compilation from terminal characterizations.
//!
//! Builds each template as a lightweight NFA (fresh intermediate states per
//! path, epsilon-connected to NT nodes) and then determinizes + minimizes to
//! produce an acyclic unweighted DFA.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::time::Instant;

use crate::automata::unweighted_u32::dfa::DFA as UnweightedDfa;
use crate::automata::unweighted_u32::determinize::determinize;
use crate::automata::unweighted_u32::minimize_acyclic::minimize_acyclic as minimize_dfa;
use crate::automata::unweighted_u32::nfa::NFA;
use crate::automata::weighted::nwa::{NWA, NWAState};
use crate::compiler::glr::labels::{
    DEFAULT_LABEL,
    encode_negative_label,
    encode_positive_label,
    is_negative_label,
};
use crate::compiler::glr::table::{Action, GLRTable};
use crate::compiler::stages::templates::characterize::{StackMatcher, TerminalCharacterization};
use crate::compiler::stages::templates::commit_template_dfas_enabled;
use crate::ds::weight::Weight;
use crate::grammar::flat::TerminalID;
use crate::runtime::CommitTemplateDfas;

#[derive(Debug, Clone, Copy, Default)]
pub struct TemplateCompileProfile {
    pub build_nfa_ms: f64,
    pub determinize_ms: f64,
    pub minimize_ms: f64,
    pub fanout_ms: f64,
    pub validation_ms: f64,
    pub total_ms: f64,
    pub wall_ms: f64,
    pub num_terminals: usize,
    pub unique_characterizations: usize,
    pub compiled_characterizations: usize,
    pub quotient_hits: usize,
    pub max_characterization_multiplicity: usize,
    pub minimize_skipped: bool,
    pub total_nfa_states: usize,
    pub max_nfa_states: usize,
    pub total_nfa_transitions: usize,
    pub max_nfa_transitions: usize,
    pub total_dfa_states: usize,
    pub max_dfa_states: usize,
    pub total_dfa_transitions: usize,
    pub max_dfa_transitions: usize,
    pub total_premin_dfa_states: usize,
    pub max_premin_dfa_states: usize,
    pub total_premin_dfa_transitions: usize,
    pub max_premin_dfa_transitions: usize,
}

impl TemplateCompileProfile {
    pub fn avg_nfa_states(&self) -> f64 {
        average(self.total_nfa_states, self.num_terminals)
    }

    pub fn avg_nfa_transitions(&self) -> f64 {
        average(self.total_nfa_transitions, self.num_terminals)
    }

    pub fn avg_dfa_states(&self) -> f64 {
        average(self.total_dfa_states, self.num_terminals)
    }

    pub fn avg_dfa_transitions(&self) -> f64 {
        average(self.total_dfa_transitions, self.num_terminals)
    }

    pub fn avg_premin_dfa_states(&self) -> f64 {
        average(self.total_premin_dfa_states, self.num_terminals)
    }

    pub fn avg_premin_dfa_transitions(&self) -> f64 {
        average(self.total_premin_dfa_transitions, self.num_terminals)
    }

    fn observe_compilation(&mut self, sample: &TemplateCompilationSample, multiplicity: usize) {
        self.build_nfa_ms += sample.build_nfa_ms;
        self.determinize_ms += sample.determinize_ms;
        self.minimize_ms += sample.minimize_ms;
        self.total_ms += sample.total_ms();
        self.compiled_characterizations += 1;
        self.num_terminals += multiplicity;
        self.total_nfa_states += sample.nfa_states * multiplicity;
        self.max_nfa_states = self.max_nfa_states.max(sample.nfa_states);
        self.total_nfa_transitions += sample.nfa_transitions * multiplicity;
        self.max_nfa_transitions = self.max_nfa_transitions.max(sample.nfa_transitions);
        self.total_dfa_states += sample.dfa_states * multiplicity;
        self.max_dfa_states = self.max_dfa_states.max(sample.dfa_states);
        self.total_dfa_transitions += sample.dfa_transitions * multiplicity;
        self.max_dfa_transitions = self.max_dfa_transitions.max(sample.dfa_transitions);
        self.total_premin_dfa_states += sample.premin_dfa_states * multiplicity;
        self.max_premin_dfa_states = self.max_premin_dfa_states.max(sample.premin_dfa_states);
        self.total_premin_dfa_transitions += sample.premin_dfa_transitions * multiplicity;
        self.max_premin_dfa_transitions =
            self.max_premin_dfa_transitions.max(sample.premin_dfa_transitions);
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct TemplateCompilationSample {
    build_nfa_ms: f64,
    determinize_ms: f64,
    minimize_ms: f64,
    nfa_states: usize,
    nfa_transitions: usize,
    dfa_states: usize,
    dfa_transitions: usize,
    premin_dfa_states: usize,
    premin_dfa_transitions: usize,
}

impl TemplateCompilationSample {
    fn total_ms(&self) -> f64 {
        self.build_nfa_ms + self.determinize_ms + self.minimize_ms
    }
}

fn elapsed_ms(started_at: Instant) -> f64 {
    started_at.elapsed().as_secs_f64() * 1000.0
}

fn average(total: usize, count: usize) -> f64 {
    if count == 0 {
        0.0
    } else {
        total as f64 / count as f64
    }
}

fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name)
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            !matches!(normalized.as_str(), "" | "0" | "false" | "no" | "off")
        })
        .unwrap_or(false)
}

fn skip_template_minimization_enabled() -> bool {
    env_flag_enabled("GLRMASK_SKIP_TEMPLATE_MINIMIZE")
}

fn template_quotient_validation_enabled() -> bool {
    commit_template_dfas_enabled() || env_flag_enabled("GLRMASK_VALIDATE_TEMPLATE_QUOTIENT")
}

fn nfa_size(nfa: &NFA) -> (usize, usize) {
    let transitions = nfa
        .states
        .iter()
        .map(|state| {
            state
                .transitions
                .values()
                .map(Vec::len)
                .sum::<usize>()
                + state.epsilons.len()
        })
        .sum();
    (nfa.states.len(), transitions)
}

fn dfa_size(dfa: &UnweightedDfa) -> (usize, usize) {
    let transitions = dfa
        .states
        .iter()
        .map(|state| state.transitions.len())
        .sum();
    (dfa.states.len(), transitions)
}

fn dfa_to_nwa_skeleton(dfa: &UnweightedDfa) -> NWA {
    let states = dfa
        .states
        .iter()
        .map(|state| NWAState {
            final_weight: state.is_accepting.then(Weight::empty),
            transitions: state
                .transitions
                .iter()
                .map(|(&label, &target)| (label, vec![(target, Weight::empty())]))
                .collect(),
            epsilons: Vec::new(),
        })
        .collect();

    NWA::from_parts(
        states,
        vec![dfa.start_state],
    )
}


fn nfa_epsilon_closure(nfa: &NFA, seeds: impl IntoIterator<Item = u32>) -> BTreeSet<u32> {
    let mut closure = BTreeSet::new();
    let mut worklist = VecDeque::new();
    for state in seeds {
        if closure.insert(state) {
            worklist.push_back(state);
        }
    }
    while let Some(state) = worklist.pop_front() {
        let Some(node) = nfa.states.get(state as usize) else {
            continue;
        };
        for &target in &node.epsilons {
            if closure.insert(target) {
                worklist.push_back(target);
            }
        }
    }
    closure
}

fn nfa_accepts_at(nfa: &NFA, states: &BTreeSet<u32>) -> bool {
    states.iter().any(|&state| nfa.is_accepting(state))
}

fn nfa_outgoing_labels(nfa: &NFA, states: &BTreeSet<u32>, labels: &mut BTreeSet<i32>) {
    for &state in states {
        if let Some(node) = nfa.states.get(state as usize) {
            labels.extend(node.transitions.keys().copied());
        }
    }
}

fn nfa_advance(nfa: &NFA, states: &BTreeSet<u32>, label: i32) -> BTreeSet<u32> {
    let targets = states.iter().flat_map(|&state| {
        nfa.states
            .get(state as usize)
            .and_then(|node| node.transitions.get(&label))
            .into_iter()
            .flatten()
            .copied()
    });
    nfa_epsilon_closure(nfa, targets)
}

/// Exact NFA-vs-DFA language comparison, including epsilon closure. The
/// product state is finite: `(epsilon-closed NFA subset, optional DFA state)`.
fn find_nfa_dfa_language_mismatch(nfa: &NFA, dfa: &UnweightedDfa) -> Option<Vec<i32>> {
    let nfa_start = nfa_epsilon_closure(nfa, nfa.start_states.iter().copied());
    let dfa_start = Some(dfa.start_state);
    let mut seen = BTreeSet::from([(nfa_start.clone(), dfa_start)]);
    let mut worklist = VecDeque::from([(nfa_start, dfa_start, Vec::new())]);

    while let Some((nfa_states, dfa_state, witness)) = worklist.pop_front() {
        if nfa_accepts_at(nfa, &nfa_states) != dfa_accepts_at(dfa, dfa_state) {
            return Some(witness);
        }
        let mut labels = BTreeSet::new();
        nfa_outgoing_labels(nfa, &nfa_states, &mut labels);
        add_outgoing_labels(dfa, dfa_state, &mut labels);
        for label in labels {
            let next = (
                nfa_advance(nfa, &nfa_states, label),
                dfa_target(dfa, dfa_state, label),
            );
            if seen.insert(next.clone()) {
                let mut next_witness = witness.clone();
                next_witness.push(label);
                worklist.push_back((next.0, next.1, next_witness));
            }
        }
    }
    None
}

fn default_specialization_local_alphabet(
    original: &UnweightedDfa,
    original_states: &BTreeSet<u32>,
    specialized: &UnweightedDfa,
    specialized_state: Option<u32>,
) -> BTreeSet<i32> {
    let mut labels = BTreeSet::new();
    for &state_id in original_states {
        if let Some(state) = original.states.get(state_id as usize) {
            labels.extend(
                state
                    .transitions
                    .keys()
                    .copied()
                    .filter(|&label| label != DEFAULT_LABEL),
            );
        }
    }
    if let Some(state) = specialized_state
        .and_then(|state_id| specialized.states.get(state_id as usize))
    {
        labels.extend(
            state
                .transitions
                .keys()
                .copied()
                .filter(|&label| label != DEFAULT_LABEL),
        );
    }

    // At this product state, every unmentioned nonnegative stack symbol follows
    // exactly the same DEFAULT transitions. One fresh representative therefore
    // completes the quotient of the infinite concrete stack alphabet.
    let mut other = 0i32;
    while labels.contains(&other) {
        other = other
            .checked_add(1)
            .expect("finite local template alphabet must leave a stack label unused");
    }
    labels.insert(other);
    labels
}

fn original_default_semantic_advance(
    dfa: &UnweightedDfa,
    states: &BTreeSet<u32>,
    label: i32,
) -> BTreeSet<u32> {
    let mut targets = BTreeSet::new();
    for &state_id in states {
        let Some(state) = dfa.states.get(state_id as usize) else {
            continue;
        };
        if let Some(&target) = state.transitions.get(&label) {
            targets.insert(target);
        }
        if label >= 0
            && let Some(&default_target) = state.transitions.get(&DEFAULT_LABEL)
        {
            targets.insert(default_target);
        }
    }
    targets
}

fn specialized_default_semantic_advance(
    dfa: &UnweightedDfa,
    state_id: Option<u32>,
    label: i32,
) -> Option<u32> {
    let state = dfa.states.get(state_id? as usize)?;
    state
        .transitions
        .get(&label)
        .copied()
        .or_else(|| (label >= 0).then(|| state.transitions.get(&DEFAULT_LABEL).copied()).flatten())
}

/// Compare the original wildcard-DFA semantics with the deterministic commit
/// specialization. Original DEFAULT and an explicit positive edge are both
/// viable; specialized runtime lookup uses the determinized explicit edge and
/// falls back to DEFAULT only when no concrete edge exists.
fn find_default_specialization_mismatch(
    original: &UnweightedDfa,
    specialized: &UnweightedDfa,
) -> Option<Vec<i32>> {
    let original_start = BTreeSet::from([original.start_state]);
    let specialized_start = Some(specialized.start_state);
    let mut seen = BTreeSet::from([(original_start.clone(), specialized_start)]);
    let mut worklist = VecDeque::from([(original_start, specialized_start, Vec::new())]);

    while let Some((original_states, specialized_state, witness)) = worklist.pop_front() {
        let original_accepts = original_states
            .iter()
            .any(|&state| dfa_accepts_at(original, Some(state)));
        if original_accepts != dfa_accepts_at(specialized, specialized_state) {
            return Some(witness);
        }
        let alphabet = default_specialization_local_alphabet(
            original,
            &original_states,
            specialized,
            specialized_state,
        );
        for label in alphabet {
            let next = (
                original_default_semantic_advance(original, &original_states, label),
                specialized_default_semantic_advance(specialized, specialized_state, label),
            );
            if seen.insert(next.clone()) {
                let mut next_witness = witness.clone();
                next_witness.push(label);
                worklist.push_back((next.0, next.1, next_witness));
            }
        }
    }
    None
}

fn specialize_template_dfa_defaults_for_commit_determinized(dfa: &UnweightedDfa) -> UnweightedDfa {
    // DEFAULT in the source is a wildcard *union*, whereas the runtime's
    // deterministic lookup uses DEFAULT only if an explicit edge is absent.
    // This must be resolved on whole reachable subsets, not on each original
    // state's local explicit alphabet: another member of a subset can supply
    // an explicit edge and thereby hide this member's wildcard contribution.
    let mut specialized = UnweightedDfa::new();
    let start_subset = BTreeSet::from([dfa.start_state]);
    let mut ids = BTreeMap::from([(start_subset.clone(), specialized.start_state)]);
    let mut pending = VecDeque::from([start_subset]);

    while let Some(subset) = pending.pop_front() {
        let from = ids[&subset];
        let mut explicit: BTreeMap<i32, BTreeSet<u32>> = BTreeMap::new();
        let mut defaults = BTreeSet::new();
        for &state_id in &subset {
            let Some(state) = dfa.states.get(state_id as usize) else {
                continue;
            };
            specialized.states[from as usize].is_accepting |= state.is_accepting;
            for (&label, &target) in &state.transitions {
                if label == DEFAULT_LABEL {
                    defaults.insert(target);
                } else {
                    explicit.entry(label).or_default().insert(target);
                }
            }
        }

        // Every concrete nonnegative symbol not explicitly mentioned below
        // has the same successor subset: exactly `defaults`. Negative labels
        // encode PUSH, so they must never inherit wildcard POP transitions.
        for (&label, targets) in &mut explicit {
            if label >= 0 {
                targets.extend(defaults.iter().copied());
            }
        }
        if !defaults.is_empty() {
            explicit.insert(DEFAULT_LABEL, defaults);
        }
        for (label, targets) in explicit {
            let to = if let Some(&id) = ids.get(&targets) {
                id
            } else {
                let id = specialized.add_state();
                ids.insert(targets.clone(), id);
                pending.push_back(targets);
                id
            };
            specialized.add_transition(from, label, to);
        }
    }

    if template_quotient_validation_enabled()
        && let Some(witness) = find_default_specialization_mismatch(dfa, &specialized)
    {
        panic!(
            "commit DEFAULT specialization changed concrete action semantics; witness: {witness:?}"
        );
    }
    specialized
}

pub fn specialize_template_dfa_defaults_for_commit(
    dfa: &UnweightedDfa,
) -> UnweightedDfa {
    let determinized = specialize_template_dfa_defaults_for_commit_determinized(dfa);
    if skip_template_minimization_enabled() {
        determinized
    } else {
        minimize_dfa(&determinized)
    }
}

pub fn specialize_template_dfa_defaults_for_commit_split_input(
    dfa: &UnweightedDfa,
) -> UnweightedDfa {
    specialize_template_dfa_defaults_for_commit_determinized(dfa)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum CommitTemplatePhase {
    Pop,
    PushEntry,
    PushAfter,
}

fn ensure_pop_state(
    old_state: u32,
    old_dfa: &UnweightedDfa,
    pop: &mut UnweightedDfa,
    pop_to_read: &mut Vec<Option<u32>>,
    pop_to_push: &mut Vec<Option<u32>>,
    pop_map: &mut BTreeMap<u32, u32>,
) -> u32 {
    if let Some(&state) = pop_map.get(&old_state) {
        return state;
    }
    let state = pop.add_state();
    pop_to_read.resize(pop.states.len(), None);
    pop_to_push.resize(pop.states.len(), None);
    if let Some(old) = old_dfa.states.get(old_state as usize) {
        pop.states[state as usize].is_accepting = old.is_accepting;
    }
    pop_map.insert(old_state, state);
    state
}

fn ensure_push_state(
    old_state: u32,
    old_dfa: &UnweightedDfa,
    push: &mut UnweightedDfa,
    push_map: &mut BTreeMap<u32, u32>,
) -> u32 {
    if let Some(&state) = push_map.get(&old_state) {
        return state;
    }
    let state = push.add_state();
    if let Some(old) = old_dfa.states.get(old_state as usize) {
        push.states[state as usize].is_accepting = old.is_accepting;
    }
    push_map.insert(old_state, state);
    state
}

fn ensure_read_source_state(
    old_state: u32,
    read: &mut UnweightedDfa,
    read_to_push: &mut Vec<Option<u32>>,
    read_source_map: &mut BTreeMap<u32, u32>,
) -> u32 {
    if let Some(&state) = read_source_map.get(&old_state) {
        return state;
    }
    let state = read.add_state();
    read_to_push.resize(read.states.len(), None);
    read_source_map.insert(old_state, state);
    state
}

fn ensure_read_target_state(
    old_state: u32,
    read: &mut UnweightedDfa,
    read_to_push: &mut Vec<Option<u32>>,
    read_target_map: &mut BTreeMap<u32, u32>,
) -> u32 {
    if let Some(&state) = read_target_map.get(&old_state) {
        return state;
    }
    let state = read.add_state();
    read_to_push.resize(read.states.len(), None);
    read_target_map.insert(old_state, state);
    state
}

fn pure_same_label_push_target(dfa: &UnweightedDfa, old_state: u32, label: i32) -> Option<u32> {
    let state = dfa.states.get(old_state as usize)?;
    if state.is_accepting || state.transitions.len() != 1 {
        return None;
    }
    let (&push_label, &target) = state.transitions.iter().next()?;
    (push_label == encode_negative_label(label as u32)).then_some(target)
}


/// Reconstruct the action-word language represented by a split commit
/// transducer.
///
/// Pop and push transitions retain their original labels. Phase links are
/// epsilon transitions. A read transition labelled `x` is the compressed form
/// of the original two-symbol word `x, -x`: inspect the top stack symbol and
/// push the same symbol back. Expanding every read edge this way gives an NFA
/// over exactly the unsplit template alphabet.
pub fn recombine_split_commit_template_language(split: &CommitTemplateDfas) -> NFA {
    let mut nfa = NFA::new_empty();
    let pop_offset = 0u32;
    let read_offset = split.pop.states.len() as u32;
    let push_offset = read_offset + split.read.states.len() as u32;
    let fixed_states = split.pop.states.len() + split.read.states.len() + split.push.states.len();
    for _ in 0..fixed_states {
        nfa.add_state();
    }
    nfa.start_states = vec![pop_offset + split.pop.start_state];

    for (state_id, state) in split.pop.states.iter().enumerate() {
        let from = pop_offset + state_id as u32;
        if state.is_accepting {
            nfa.set_accepting(from);
        }
        for (&label, &target) in &state.transitions {
            nfa.add_transition(from, label, pop_offset + target);
        }
        if let Some(Some(target)) = split.pop_to_read.get(state_id) {
            nfa.add_epsilon(from, read_offset + *target);
        }
        if let Some(Some(target)) = split.pop_to_push.get(state_id) {
            nfa.add_epsilon(from, push_offset + *target);
        }
    }

    for (state_id, state) in split.read.states.iter().enumerate() {
        let from = read_offset + state_id as u32;
        if state.is_accepting {
            nfa.set_accepting(from);
        }
        for (&label, &target) in &state.transitions {
            assert!(
                label >= 0 && label != DEFAULT_LABEL,
                "split commit read DFA contains non-concrete read label {label}"
            );
            let intermediate = nfa.add_state();
            nfa.add_transition(from, label, intermediate);
            nfa.add_transition(
                intermediate,
                encode_negative_label(label as u32),
                read_offset + target,
            );
        }
        if let Some(Some(target)) = split.read_to_push.get(state_id) {
            nfa.add_epsilon(from, push_offset + *target);
        }
    }

    for (state_id, state) in split.push.states.iter().enumerate() {
        let from = push_offset + state_id as u32;
        if state.is_accepting {
            nfa.set_accepting(from);
        }
        for (&label, &target) in &state.transitions {
            assert!(
                is_negative_label(label),
                "split commit push DFA contains non-push label {label}"
            );
            nfa.add_transition(from, label, push_offset + target);
        }
    }

    nfa
}

/// Reconstruct the unsplit stack-effect language carried by a persisted split
/// commit template. This is primarily useful to test whether the serialized
/// runtime template can double as an exact composition-time parser template.
pub fn recombine_split_commit_template_dfa(split: &CommitTemplateDfas) -> UnweightedDfa {
    determinize(&recombine_split_commit_template_language(split))
}

fn find_split_commit_language_mismatch(
    unsplit: &UnweightedDfa,
    split: &CommitTemplateDfas,
) -> Option<Vec<i32>> {
    let recombined = recombine_split_commit_template_language(split);
    find_nfa_dfa_language_mismatch(&recombined, unsplit)
}

pub fn try_split_commit_template_dfas(
    dfa: &UnweightedDfa,
) -> Option<CommitTemplateDfas> {
    let mut pop = UnweightedDfa::default();
    let mut read = UnweightedDfa::default();
    let mut push = UnweightedDfa::default();
    let mut pop_to_read = Vec::new();
    let mut pop_to_push = Vec::new();
    let mut read_to_push = Vec::new();
    let mut pop_map = BTreeMap::new();
    let mut push_map = BTreeMap::new();
    let mut read_source_map = BTreeMap::new();
    let mut read_target_map = BTreeMap::new();
    // READ compression removes an explicit POP edge from its row. When the
    // row also has DEFAULT, a rejecting explicit edge must continue to shadow
    // that fallback. Otherwise x / push(x) becomes READ(x) *plus* DEFAULT(x),
    // admitting an unrelated stack effect. One shared dead POP state suffices.
    let mut dead_pop_state = None;

    let start = ensure_pop_state(
        dfa.start_state,
        dfa,
        &mut pop,
        &mut pop_to_read,
        &mut pop_to_push,
        &mut pop_map,
    );
    pop.start_state = start;

    let mut worklist = VecDeque::from([(dfa.start_state, CommitTemplatePhase::Pop)]);
    let mut visited = BTreeSet::new();

    while let Some((old_state, phase)) = worklist.pop_front() {
        if !visited.insert((old_state, phase)) {
            continue;
        }
        let Some(old) = dfa.states.get(old_state as usize) else {
            continue;
        };

        match phase {
            CommitTemplatePhase::Pop => {
                let pop_state = ensure_pop_state(
                    old_state,
                    dfa,
                    &mut pop,
                    &mut pop_to_read,
                    &mut pop_to_push,
                    &mut pop_map,
                );
                for (&label, &target) in &old.transitions {
                    if is_negative_label(label) {
                        let push_state = ensure_push_state(old_state, dfa, &mut push, &mut push_map);
                        pop_to_push[pop_state as usize] = Some(push_state);
                        worklist.push_back((old_state, CommitTemplatePhase::PushEntry));
                        continue;
                    }

                    if label != DEFAULT_LABEL
                        && label >= 0
                        && let Some(post_read_target) =
                            pure_same_label_push_target(dfa, target, label)
                    {
                        let read_source = ensure_read_source_state(
                            old_state,
                            &mut read,
                            &mut read_to_push,
                            &mut read_source_map,
                        );
                        pop_to_read[pop_state as usize] = Some(read_source);
                        let read_target = ensure_read_target_state(
                            post_read_target,
                            &mut read,
                            &mut read_to_push,
                            &mut read_target_map,
                        );
                        read.add_transition(read_source, label, read_target);
                        if old.transitions.contains_key(&DEFAULT_LABEL) {
                            let dead = *dead_pop_state.get_or_insert_with(|| {
                                let dead = pop.add_state();
                                pop_to_read.resize(pop.states.len(), None);
                                pop_to_push.resize(pop.states.len(), None);
                                dead
                            });
                            pop.add_transition(pop_state, label, dead);
                        }
                        let push_target =
                            ensure_push_state(post_read_target, dfa, &mut push, &mut push_map);
                        read_to_push[read_target as usize] = Some(push_target);
                        worklist.push_back((post_read_target, CommitTemplatePhase::PushAfter));
                        continue;
                    }

                    if label == DEFAULT_LABEL || label >= 0 {
                        let target_state = ensure_pop_state(
                            target,
                            dfa,
                            &mut pop,
                            &mut pop_to_read,
                            &mut pop_to_push,
                            &mut pop_map,
                        );
                        pop.add_transition(pop_state, label, target_state);
                        worklist.push_back((target, CommitTemplatePhase::Pop));
                    }
                }
            }
            CommitTemplatePhase::PushEntry | CommitTemplatePhase::PushAfter => {
                let push_state = ensure_push_state(old_state, dfa, &mut push, &mut push_map);
                for (&label, &target) in &old.transitions {
                    if !is_negative_label(label) {
                        if phase == CommitTemplatePhase::PushEntry {
                            continue;
                        }
                        return None;
                    }
                    let target_state = ensure_push_state(target, dfa, &mut push, &mut push_map);
                    push.add_transition(push_state, label, target_state);
                    worklist.push_back((target, CommitTemplatePhase::PushAfter));
                }
            }
        }
    }

    let split = CommitTemplateDfas {
        pop,
        read,
        push,
        pop_to_read,
        pop_to_push,
        read_to_push,
    };
    if find_split_commit_language_mismatch(dfa, &split).is_some() {
        return None;
    }
    Some(split)
}


fn compile_template_with_profile(
    characterization: &TerminalCharacterization,
) -> (UnweightedDfa, NWA, TemplateCompilationSample) {
    compile_template_with_profile_and_minimize(
        characterization,
        skip_template_minimization_enabled(),
    )
}

fn compile_template_with_profile_and_minimize(
    characterization: &TerminalCharacterization,
    skip_minimize: bool,
) -> (UnweightedDfa, NWA, TemplateCompilationSample) {
    let (dfa, sample) = compile_template_dfa_with_profile(characterization, skip_minimize);
    let skeleton = dfa_to_nwa_skeleton(&dfa);
    (dfa, skeleton, sample)
}

fn compile_template_dfa_with_profile(
    characterization: &TerminalCharacterization,
    skip_minimize: bool,
) -> (UnweightedDfa, TemplateCompilationSample) {
    let build_nfa_started_at = Instant::now();
    let nfa = build_template_nfa(characterization);
    let build_nfa_ms = elapsed_ms(build_nfa_started_at);
    let (nfa_states, nfa_transitions) = nfa_size(&nfa);

    let determinize_started_at = Instant::now();
    let determinized = determinize(&nfa);
    if template_quotient_validation_enabled()
        && let Some(witness) = find_nfa_dfa_language_mismatch(&nfa, &determinized)
    {
        panic!("template determinization changed the NFA language; witness: {witness:?}");
    }
    let determinize_ms = elapsed_ms(determinize_started_at);
    let (premin_dfa_states, premin_dfa_transitions) = dfa_size(&determinized);

    let minimize_started_at = Instant::now();
    let dfa = if skip_minimize {
        determinized.clone()
    } else {
        minimize_dfa(&determinized)
    };
    if template_quotient_validation_enabled()
        && let Some(witness) = find_dfa_language_mismatch(&determinized, &dfa)
    {
        panic!("template minimization changed the DFA language; witness: {witness:?}");
    }
    let minimize_ms = if skip_minimize {
        0.0
    } else {
        elapsed_ms(minimize_started_at)
    };
    let (dfa_states, dfa_transitions) = dfa_size(&dfa);

    (
        dfa,
        TemplateCompilationSample {
            build_nfa_ms,
            determinize_ms,
            minimize_ms,
            nfa_states,
            nfa_transitions,
            dfa_states,
            dfa_transitions,
            premin_dfa_states,
            premin_dfa_transitions,
        },
    )
}

#[derive(Debug, Clone, Default)]
pub struct Templates {
    pub by_terminal: BTreeMap<TerminalID, UnweightedDfa>,
    pub by_terminal_nwa: BTreeMap<TerminalID, NWA>,
}

impl Templates {
    pub fn from_terminal_dfas(
        by_terminal: BTreeMap<TerminalID, UnweightedDfa>,
    ) -> Self {
        let by_terminal_nwa = by_terminal
            .iter()
            .map(|(&terminal, dfa)| (terminal, dfa_to_nwa_skeleton(dfa)))
            .collect();
        Self {
            by_terminal,
            by_terminal_nwa,
        }
    }

    /// Build exact templates for a constant-depth direct-regular parser table.
    ///
    /// Every terminal action replaces the current parser top with one of a
    /// finite set of targets. Its template language is therefore exactly the
    /// set of two-label words `old_top, -new_top`. Construct that DFA directly,
    /// sharing middle states by target set, instead of passing through generic
    /// stack-effect characterization, NFA construction, determinization and
    /// minimization.
    pub fn from_direct_regular_table(
        table: &GLRTable,
        num_terminals: u32,
    ) -> Option<Self> {
        let mut targets_by_terminal = (0..num_terminals)
            .map(|_| BTreeMap::<u32, Vec<u32>>::new())
            .collect::<Vec<_>>();
        for (source, row) in table.action.iter().enumerate() {
            for (terminal, action) in row {
                if terminal >= num_terminals {
                    continue;
                }
                let targets = targets_by_terminal[terminal as usize]
                    .entry(source as u32)
                    .or_default();
                match action {
                    Action::Shift(target, true) => targets.push(*target),
                    Action::StackShifts(shifts) => {
                        for shift in shifts {
                            if shift.pop != 1 || shift.pushes.len() != 1 {
                                return None;
                            }
                            targets.push(shift.pushes[0]);
                        }
                    }
                    _ => return None,
                }
            }
        }

        let mut by_terminal = BTreeMap::new();
        let mut by_terminal_nwa = BTreeMap::new();
        for (terminal, targets_by_source) in targets_by_terminal.into_iter().enumerate() {
            let terminal = terminal as u32;
            let mut dfa = UnweightedDfa::new();
            if !targets_by_source.is_empty() {
                let accept = dfa.add_state();
                dfa.set_accepting(accept, true);
                let mut middle_by_targets = BTreeMap::<Vec<u32>, u32>::new();
                for (source, mut targets) in targets_by_source {
                    targets.sort_unstable();
                    targets.dedup();
                    let middle = if let Some(&middle) = middle_by_targets.get(&targets) {
                        middle
                    } else {
                        let middle = dfa.add_state();
                        for &target in &targets {
                            dfa.add_transition(middle, encode_negative_label(target), accept);
                        }
                        middle_by_targets.insert(targets, middle);
                        middle
                    };
                    dfa.add_transition(dfa.start_state, encode_positive_label(source), middle);
                }
            }
            let skeleton = dfa_to_nwa_skeleton(&dfa);
            by_terminal.insert(terminal, dfa);
            by_terminal_nwa.insert(terminal, skeleton);
        }

        Some(Self {
            by_terminal,
            by_terminal_nwa,
        })
    }

    pub fn from_characterizations(
        characterizations: &BTreeMap<TerminalID, TerminalCharacterization>,
    ) -> Self {
        Self::from_characterizations_profiled(characterizations).0
    }

    pub fn from_characterizations_profiled(
        characterizations: &BTreeMap<TerminalID, TerminalCharacterization>,
    ) -> (Self, TemplateCompileProfile) {
        Self::from_characterizations_with_outputs(characterizations, true)
    }

    /// Compile the same exact terminal DFAs without producing unused NWA
    /// skeletons. Runtime parser installation consumes only these DFAs;
    /// composition callers retain the existing dual-output constructor.
    pub fn dfas_from_characterizations(
        characterizations: &BTreeMap<TerminalID, TerminalCharacterization>,
    ) -> BTreeMap<TerminalID, UnweightedDfa> {
        Self::from_characterizations_with_outputs(characterizations, false).0.by_terminal
    }

    fn from_characterizations_with_outputs(
        characterizations: &BTreeMap<TerminalID, TerminalCharacterization>,
        include_skeletons: bool,
    ) -> (Self, TemplateCompileProfile) {
        use rayon::prelude::*;

        let total_started_at = Instant::now();
        let skip_minimize = skip_template_minimization_enabled();

        let mut grouped = BTreeMap::<&TerminalCharacterization, Vec<TerminalID>>::new();
        for (&terminal, characterization) in characterizations {
            grouped.entry(characterization).or_default().push(terminal);
        }
        let groups: Vec<(&TerminalCharacterization, Vec<TerminalID>)> = grouped.into_iter().collect();

        let build = |(characterization, terminals): &(&TerminalCharacterization, Vec<TerminalID>)| {
            let (dfa, sample) = compile_template_dfa_with_profile(*characterization, skip_minimize);
            let skeleton = include_skeletons.then(|| dfa_to_nwa_skeleton(&dfa));
            (terminals.clone(), dfa, skeleton, sample)
        };
        let compiled: Vec<(Vec<TerminalID>, UnweightedDfa, Option<NWA>, TemplateCompilationSample)> =
            if super::macro_parallelism_disabled() {
                groups.iter().map(build).collect()
            } else {
                groups.par_iter().map(build).collect()
            };
        super::report_macro_item_timings(
            "template_compile_characterizations",
            &compiled
                .iter()
                .map(|(_, _, _, sample)| sample.total_ms())
                .collect::<Vec<_>>(),
        );

        let mut profile = TemplateCompileProfile {
            unique_characterizations: groups.len(),
            max_characterization_multiplicity: groups
                .iter()
                .map(|(_, terminals)| terminals.len())
                .max()
                .unwrap_or(0),
            quotient_hits: characterizations.len().saturating_sub(groups.len()),
            minimize_skipped: skip_minimize,
            ..TemplateCompileProfile::default()
        };

        let mut by_terminal = BTreeMap::new();
        let mut by_terminal_nwa = BTreeMap::new();
        let fanout_started_at = Instant::now();
        for (terminals, dfa, skeleton, sample) in compiled {
            profile.observe_compilation(&sample, terminals.len());
            for terminal in terminals {
                by_terminal.insert(terminal, dfa.clone());
                if let Some(skeleton) = &skeleton {
                    by_terminal_nwa.insert(terminal, skeleton.clone());
                }
            }
        }
        profile.fanout_ms = elapsed_ms(fanout_started_at);
        profile.total_ms += profile.fanout_ms;

        let validation_started_at = Instant::now();
        if template_quotient_validation_enabled() {
            validate_template_quotient(&groups, &by_terminal, &by_terminal_nwa, include_skeletons);
        }
        profile.validation_ms = elapsed_ms(validation_started_at);
        profile.total_ms += profile.validation_ms;
        profile.wall_ms = elapsed_ms(total_started_at);

        (
            Self {
                by_terminal,
                by_terminal_nwa,
            },
            profile,
        )
    }
}

fn dfa_accepts_at(dfa: &UnweightedDfa, state: Option<u32>) -> bool {
    state
        .and_then(|state| dfa.states.get(state as usize))
        .is_some_and(|state| state.is_accepting)
}

fn dfa_target(dfa: &UnweightedDfa, state: Option<u32>, label: i32) -> Option<u32> {
    state
        .and_then(|state| dfa.states.get(state as usize))
        .and_then(|state| state.transitions.get(&label).copied())
}

fn add_outgoing_labels(dfa: &UnweightedDfa, state: Option<u32>, labels: &mut BTreeSet<i32>) {
    if let Some(state) = state.and_then(|state| dfa.states.get(state as usize)) {
        labels.extend(state.transitions.keys().copied());
    }
}

fn find_dfa_language_mismatch(
    left: &UnweightedDfa,
    right: &UnweightedDfa,
) -> Option<Vec<i32>> {
    let mut seen = BTreeSet::<(Option<u32>, Option<u32>)>::new();
    let mut worklist = VecDeque::<(Option<u32>, Option<u32>, Vec<i32>)>::new();

    let start = (Some(left.start_state), Some(right.start_state));
    seen.insert(start);
    worklist.push_back((start.0, start.1, Vec::new()));

    while let Some((left_state, right_state, witness)) = worklist.pop_front() {
        if dfa_accepts_at(left, left_state) != dfa_accepts_at(right, right_state) {
            return Some(witness);
        }

        let mut labels = BTreeSet::new();
        add_outgoing_labels(left, left_state, &mut labels);
        add_outgoing_labels(right, right_state, &mut labels);

        for label in labels {
            let next = (
                dfa_target(left, left_state, label),
                dfa_target(right, right_state, label),
            );
            if seen.insert(next) {
                let mut next_witness = witness.clone();
                next_witness.push(label);
                worklist.push_back((next.0, next.1, next_witness));
            }
        }
    }

    None
}

fn nwa_skeleton_matches_dfa(dfa: &UnweightedDfa, skeleton: &NWA) -> bool {
    let expected = dfa_to_nwa_skeleton(dfa);
    expected.start_states() == skeleton.start_states() && expected.states() == skeleton.states()
}

fn validate_template_quotient(
    groups: &[(&TerminalCharacterization, Vec<TerminalID>)],
    by_terminal: &BTreeMap<TerminalID, UnweightedDfa>,
    by_terminal_nwa: &BTreeMap<TerminalID, NWA>,
    include_skeletons: bool,
) {
    if !include_skeletons {
        assert!(by_terminal_nwa.is_empty(), "DFA-only construction emitted an unused skeleton");
    }
    let skip_minimize = skip_template_minimization_enabled();
    for (characterization, terminals) in groups {
        let representative = terminals[0];
        let representative_dfa = by_terminal.get(&representative).unwrap_or_else(|| {
            panic!("missing template DFA for representative terminal {representative}")
        });

        for &terminal in terminals {
            let cached = by_terminal
                .get(&terminal)
                .unwrap_or_else(|| panic!("missing template DFA for terminal {terminal}"));
            assert_eq!(
                cached, representative_dfa,
                "template quotient fanout mismatch for terminal {terminal} and representative {representative}"
            );
            if include_skeletons {
                let cached_skeleton = by_terminal_nwa.get(&terminal).unwrap_or_else(|| {
                    panic!("missing template NWA skeleton for terminal {terminal}")
                });
                assert!(
                    nwa_skeleton_matches_dfa(cached, cached_skeleton),
                    "template NWA skeleton is not the DFA skeleton for terminal {terminal}"
                );
            }
        }

        if skip_minimize {
            let (old_minimized, _, _) =
                compile_template_with_profile_and_minimize(characterization, false);
            if let Some(witness) = find_dfa_language_mismatch(representative_dfa, &old_minimized) {
                panic!(
                    "template minimization-skip mismatch for representative terminal {representative}; witness label path: {:?}",
                    witness
                );
            }
        }
    }
}

fn build_nonterminal_nodes(
    nfa: &mut NFA,
    characterization: &TerminalCharacterization,
) -> BTreeMap<u32, u32> {
    let mut nonterminal_nodes = BTreeMap::new();
    for &nonterminal in &characterization.all_nts {
        let state = nfa.add_state();
        nonterminal_nodes.insert(nonterminal, state);
    }
    nonterminal_nodes
}

/// A shared DEFAULT-labeled pop chain ending at `target`.
///
/// `chain[i]` is an NFA state such that there is a sequence of `i+1`
/// consecutive DEFAULT transitions from `chain[i]` to `target`. That is:
/// - `chain[0]` has a DEFAULT transition to `target` (one pop).
/// - `chain[i]` has a DEFAULT transition to `chain[i - 1]` (i+1 pops).
///
/// A caller wanting `k` pops leading to `target` (`k >= 1`) directs its
/// positive transition to `chain[k - 1]`, reusing all DEFAULT-pop states
/// shared by other reduces targeting the same nonterminal. This keeps
/// the template NFA size at O(num_nonterminals × max_pop_count) instead
/// of O(total_reduces × avg_pop_count).
struct PopChain {
    states: Vec<u32>,
}

struct PopChainPool {
    chains: BTreeMap<u32, PopChain>,
}

impl PopChainPool {
    fn new() -> Self {
        Self {
            chains: BTreeMap::new(),
        }
    }

    /// Return the NFA state that has a chain of `pop_count` DEFAULT transitions
    /// terminating at the nonterminal node `target_state`, extending the shared
    /// chain for `target_nt` as needed. Requires `pop_count >= 1`.
    fn entry_state(
        &mut self,
        nfa: &mut NFA,
        target_nt: u32,
        target_state: u32,
        pop_count: usize,
    ) -> u32 {
        debug_assert!(pop_count >= 1);
        let chain = self.chains.entry(target_nt).or_insert_with(|| PopChain {
            states: Vec::new(),
        });
        while chain.states.len() < pop_count {
            let idx = chain.states.len();
            let predecessor = if idx == 0 {
                target_state
            } else {
                chain.states[idx - 1]
            };
            let new_state = nfa.add_state();
            nfa.add_transition(new_state, DEFAULT_LABEL, predecessor);
            chain.states.push(new_state);
        }
        chain.states[pop_count - 1]
    }
}

fn add_positive_transition_chain_shared(
    nfa: &mut NFA,
    pool: &mut PopChainPool,
    from: u32,
    revealed_state: u32,
    pop_count: usize,
    target_nt: u32,
    target_state: u32,
) {
    if pop_count == 0 {
        nfa.add_epsilon(from, target_state);
        return;
    }
    if pop_count == 1 {
        nfa.add_transition(from, encode_positive_label(revealed_state), target_state);
        return;
    }
    let entry = pool.entry_state(nfa, target_nt, target_state, pop_count - 1);
    nfa.add_transition(from, encode_positive_label(revealed_state), entry);
}

fn add_matcher_transition(nfa: &mut NFA, from: u32, matcher: &StackMatcher, to: u32) {
    match matcher {
        StackMatcher::Any => {
            nfa.add_transition(from, DEFAULT_LABEL, to);
        }
        StackMatcher::State(state) => {
            nfa.add_transition(from, encode_positive_label(*state), to);
        }
        StackMatcher::States(states) => {
            for &state in states {
                nfa.add_transition(from, encode_positive_label(state), to);
            }
        }
    }
}

fn add_pop_pattern_path(nfa: &mut NFA, from: u32, pop: &[StackMatcher], to: u32) {
    if pop.is_empty() {
        nfa.add_epsilon(from, to);
        return;
    }

    let mut current = from;
    for (index, matcher) in pop.iter().enumerate() {
        let next = if index + 1 == pop.len() {
            to
        } else {
            nfa.add_state()
        };
        add_matcher_transition(nfa, current, matcher, next);
        current = next;
    }
}

fn simple_exact_then_any(pop: &[StackMatcher]) -> Option<(u32, usize)> {
    let (first, rest) = pop.split_first()?;
    let StackMatcher::State(first_state) = first else {
        return None;
    };

    if rest.iter().all(|matcher| matches!(matcher, StackMatcher::Any)) {
        Some((*first_state, pop.len()))
    } else {
        None
    }
}

fn add_reduce_pattern_path(
    nfa: &mut NFA,
    pool: &mut PopChainPool,
    from: u32,
    pop: &[StackMatcher],
    target_nt: u32,
    target_state: u32,
) {
    if let Some((first_state, pop_count)) = simple_exact_then_any(pop) {
        add_positive_transition_chain_shared(
            nfa,
            pool,
            from,
            first_state,
            pop_count,
            target_nt,
            target_state,
        );
    } else {
        add_pop_pattern_path(nfa, from, pop, target_state);
    }
}

fn resolve_pos_target(
    nfa: &mut NFA,
    pos_target_cache: &mut BTreeMap<Vec<u32>, u32>,
    suffix_trie: &mut BTreeMap<(u32, u32), u32>,
    accept_root: u32,
    pushes: &[u32],
) -> u32 {
    if let Some(&cached) = pos_target_cache.get(pushes) {
        return cached;
    }
    let mut cur = accept_root;
    for &push_state in pushes.iter().rev() {
        let key = (cur, push_state);
        cur = if let Some(&existing) = suffix_trie.get(&key) {
            existing
        } else {
            let state = nfa.add_state();
            nfa.add_transition(state, encode_negative_label(push_state), cur);
            suffix_trie.insert(key, state);
            state
        };
    }
    pos_target_cache.insert(pushes.to_vec(), cur);
    cur
}

fn add_escape_pattern_path(
    nfa: &mut NFA,
    pos_target_cache: &mut BTreeMap<Vec<u32>, u32>,
    suffix_trie: &mut BTreeMap<(u32, u32), u32>,
    emitted_escapes: &mut BTreeSet<(u32, Vec<StackMatcher>, Vec<u32>)>,
    accept_root: u32,
    from: u32,
    pop: &[StackMatcher],
    pushes: &[u32],
) {
    if !emitted_escapes.insert((from, pop.to_vec(), pushes.to_vec())) {
        return;
    }
    let pos_target = resolve_pos_target(nfa, pos_target_cache, suffix_trie, accept_root, pushes);
    add_pop_pattern_path(nfa, from, pop, pos_target);
}

/// Build an unweighted NFA from a terminal characterization.
///
/// Each shift/reduce/escape/re-reduce path gets its own fresh intermediate
/// states, connected to the shared start state (via epsilon) and to shared
/// NT-node states.
fn build_template_nfa(characterization: &TerminalCharacterization) -> NFA {
    let mut nfa = NFA::new();
    let start = 0u32; // NFA::new() creates state 0 as start

    let nonterminal_nodes = build_nonterminal_nodes(&mut nfa, characterization);
    let mut pool = PopChainPool::new();

    // Shared escape-chain tail.
    //
    // An "escape chain" is the sequence
    //     positive(revealed_state) → negative(pushes[0]) → … → negative(pushes[n]) → accepting
    // emitted for every `(escape)` and `(nt_escape)` entry in the
    // characterization. Rather than materialise a distinct entry node per
    // signature and splice the source via an epsilon, each source adds its
    // positive transition directly to a shared "pos-target" state that
    // represents the state reached just after firing `positive(revealed)`.
    // The pos-target state is cached per `pushes` (the `revealed` component
    // differs per caller but never affects the negative-chain tail).
    //
    // A source dedup set eliminates duplicate positive transitions when the
    // characterization repeats `(source, revealed, pushes)` tuples.

    // Suffix trie over *reversed* push sequences, all rooted at a single
    // shared accepting state. If two signatures share a common `pushes`
    // suffix, they share the corresponding NFA states and negative
    // transitions. For `(pushes = [p0, p1, …, pn])`, the trie walk starts at
    // the shared `accept_root` and consumes `pn, pn-1, …, p0` in reverse;
    // the state reached after consuming all pushes is the pos-target that
    // the caller's positive transition points at.
    //
    // Key: `(child_state, push_label)` → `parent_state` such that
    // `parent_state` has a `negative(push_label)` transition to `child_state`.
    let accept_root = nfa.add_state();
    nfa.set_accepting(accept_root);
    let mut suffix_trie: BTreeMap<(u32, u32), u32> = BTreeMap::new();

    // Cache of pos-target states keyed by `pushes`.
    let mut pos_target_cache: BTreeMap<Vec<u32>, u32> = BTreeMap::new();

    // Dedup set for emitted `(source, revealed, pushes)` positive transitions.
    // Keying includes `pushes` rather than `pos_target` because two distinct
    // `pushes` sequences may resolve (under suffix sharing) to the same
    // `pos_target`, yet still represent logically distinct escapes; we dedupe
    // purely to avoid inserting the same transition twice when the
    // characterization contains exact duplicates.
    let mut emitted_escapes: BTreeSet<(u32, Vec<StackMatcher>, Vec<u32>)> = BTreeSet::new();

    // Initial escapes: start → positive(initial_state) → [extra DEFAULT pops] → [shared suffix tail] → accept_root
    for escape in &characterization.escapes {
        add_escape_pattern_path(
            &mut nfa,
            &mut pos_target_cache,
            &mut suffix_trie,
            &mut emitted_escapes,
            accept_root,
            start,
            &escape.pop,
            &escape.pushes,
        );
    }

    for reduce in &characterization.reduces {
        let Some(&target_nonterminal_state) = nonterminal_nodes.get(&reduce.nonterminal) else {
            continue;
        };

        add_reduce_pattern_path(
            &mut nfa,
            &mut pool,
            start,
            &reduce.pop,
            reduce.nonterminal,
            target_nonterminal_state,
        );
    }

    // NT escapes: source_nt_node → positive(revealed) → [shared suffix tail] → accept_root.
    // The suffix tail is shared across every `(source, revealed, pushes)` that
    // agrees on the `pushes` tail; the positive transition is added directly
    // from the source, with dedup against exact `(source, revealed, pushes)`
    // duplicates.
    for nt_escape in &characterization.nt_escapes {
        let Some(&source_state) = nonterminal_nodes.get(&nt_escape.source_nonterminal) else {
            continue;
        };
        add_escape_pattern_path(
            &mut nfa,
            &mut pos_target_cache,
            &mut suffix_trie,
            &mut emitted_escapes,
            accept_root,
            source_state,
            &nt_escape.pop,
            &nt_escape.pushes,
        );
    }

    for nt_rereduce in &characterization.nt_rereduces {
        let (Some(&source_state), Some(&target_state)) =
            (
                nonterminal_nodes.get(&nt_rereduce.source_nonterminal),
                nonterminal_nodes.get(&nt_rereduce.target_nonterminal),
            )
        else {
            continue;
        };

        add_reduce_pattern_path(
            &mut nfa,
            &mut pool,
            source_state,
            &nt_rereduce.pop,
            nt_rereduce.target_nonterminal,
            target_state,
        );
    }

    nfa
}

#[cfg(test)]
mod tests {
    use super::{
        specialize_template_dfa_defaults_for_commit_determinized,
        find_nfa_dfa_language_mismatch,
        find_default_specialization_mismatch,
        find_split_commit_language_mismatch, try_split_commit_template_dfas,
    };
    use crate::automata::unweighted_u32::determinize::determinize;
    use crate::automata::unweighted_u32::dfa::DFA as UnweightedDfa;
    use crate::automata::unweighted_u32::nfa::NFA;
    use crate::compiler::glr::labels::{
        DEFAULT_LABEL, encode_negative_label,
    };

    #[test]
    fn dfa_only_output_matches_complete_template_construction() {
        use super::{Templates, build_template_nfa, validate_template_quotient};
        use crate::compiler::stages::templates::characterize::{
            InitialEscape, InitialReduce, NtEscape, NtRereduce,
            StackMatcher, TerminalCharacterization,
        };
        use std::collections::{BTreeMap, BTreeSet};

        for seed in 0..64u32 {
            let mut characterization = TerminalCharacterization {
                escapes: vec![
                    InitialEscape {
                        pop: vec![StackMatcher::State(seed % 7)],
                        pushes: vec![seed % 5, (seed + 1) % 5],
                    },
                    InitialEscape {
                        pop: vec![StackMatcher::Any, StackMatcher::States(vec![1, 3, 6])],
                        pushes: vec![seed % 3],
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
            if seed % 3 == 0 {
                characterization.escapes.push(InitialEscape {
                    pop: Vec::new(), pushes: Vec::new(),
                });
            }
            let mut different = characterization.clone();
            different.escapes[0].pushes.push(7);
            let input = BTreeMap::from([
                (0, characterization.clone()),
                (4, different.clone()),
                (12, characterization.clone()),
                (101, characterization),
            ]);
            let complete = Templates::from_characterizations(&input);
            let (dfa_only, profile) = Templates::from_characterizations_with_outputs(&input, false);
            assert!(dfa_only.by_terminal_nwa.is_empty());
            assert_eq!(complete.by_terminal, dfa_only.by_terminal);
            assert_eq!(Templates::dfas_from_characterizations(&input), complete.by_terminal);
            assert_eq!(profile.unique_characterizations, 2);
            assert_eq!(profile.max_characterization_multiplicity, 3);
            let mut grouped = BTreeMap::<&TerminalCharacterization, Vec<u32>>::new();
            for (&terminal, characterization) in &input {
                grouped.entry(characterization).or_default().push(terminal);
                let nfa = build_template_nfa(characterization);
                assert_eq!(find_nfa_dfa_language_mismatch(&nfa, &dfa_only.by_terminal[&terminal]), None);
            }
            let groups = grouped.into_iter().collect::<Vec<_>>();
            validate_template_quotient(&groups, &dfa_only.by_terminal, &dfa_only.by_terminal_nwa, false);
            validate_template_quotient(&groups, &complete.by_terminal, &complete.by_terminal_nwa, true);
        }
    }

    #[test]
    fn dfa_only_empty_inventory_preserves_no_outputs() {
        let input = std::collections::BTreeMap::new();
        let full = super::Templates::from_characterizations(&input);
        let only = super::Templates::dfas_from_characterizations(&input);
        assert!(full.by_terminal.is_empty());
        assert!(full.by_terminal_nwa.is_empty());
        assert_eq!(only, full.by_terminal);
    }

    fn mixed_phase_commit_dfa() -> UnweightedDfa {
        let mut dfa = UnweightedDfa::new();

        // Read compression candidate: 7, -7, -20.
        let after_pop_for_read = dfa.add_state();
        let after_read = dfa.add_state();
        let read_accept = dfa.add_state();
        dfa.add_transition(dfa.start_state, 7, after_pop_for_read);
        dfa.add_transition(after_pop_for_read, encode_negative_label(7), after_read);
        dfa.add_transition(after_read, encode_negative_label(20), read_accept);
        dfa.set_accepting(read_accept, true);

        // Ordinary default-pop then push path: DEFAULT, -30.
        let after_default_pop = dfa.add_state();
        let default_accept = dfa.add_state();
        dfa.add_transition(dfa.start_state, DEFAULT_LABEL, after_default_pop);
        dfa.add_transition(
            after_default_pop,
            encode_negative_label(30),
            default_accept,
        );
        dfa.set_accepting(default_accept, true);

        // Push-entry path with no pop: -40.
        let direct_push_accept = dfa.add_state();
        dfa.add_transition(
            dfa.start_state,
            encode_negative_label(40),
            direct_push_accept,
        );
        dfa.set_accepting(direct_push_accept, true);

        dfa
    }

    #[test]
    fn split_commit_transducer_preserves_mixed_phase_action_language() {
        let dfa = mixed_phase_commit_dfa();
        let split = try_split_commit_template_dfas(&dfa)
            .expect("mixed-phase commit DFA should be splittable");
        assert_eq!(find_split_commit_language_mismatch(&dfa, &split), None);
    }

    #[test]
    fn read_compression_keeps_explicit_default_shadow() {
        // For top 7: preserve 7 and push 20. For OTHER tops: replace by 30.
        // There is also an unconditional push 40. Literal action-word
        // equivalence alone cannot detect losing the exclusion of 7 from
        // DEFAULT when the explicit pop(7),push(7) pair becomes READ(7).
        let dfa = mixed_phase_commit_dfa();
        let split = try_split_commit_template_dfas(&dfa).unwrap();
        let root = &split.pop.states[split.pop.start_state as usize];
        let &shadow = root.transitions.get(&7)
            .expect("READ-compressed label must still shadow DEFAULT");
        let dead = &split.pop.states[shadow as usize];
        assert!(!dead.is_accepting && dead.transitions.is_empty());
        assert!(split.pop_to_read[shadow as usize].is_none());
        assert!(split.pop_to_push[shadow as usize].is_none());
        assert!(root.transitions.contains_key(&DEFAULT_LABEL));
        let read = split.pop_to_read[split.pop.start_state as usize].unwrap();
        assert!(split.read.states[read as usize].transitions.contains_key(&7));
        assert_eq!(find_split_commit_language_mismatch(&dfa, &split), None);
    }

    #[test]
    fn split_commit_equivalence_checker_returns_a_corruption_witness() {
        let dfa = mixed_phase_commit_dfa();
        let mut split = try_split_commit_template_dfas(&dfa)
            .expect("mixed-phase commit DFA should be splittable");
        let link = split
            .read_to_push
            .iter_mut()
            .find(|link| link.is_some())
            .expect("test DFA must produce a read-to-push link");
        *link = None;

        let witness = find_split_commit_language_mismatch(&dfa, &split)
            .expect("corrupt split must differ from the unsplit DFA");
        assert_eq!(
            witness,
            vec![7, encode_negative_label(7), encode_negative_label(20)]
        );
    }

    #[test]
    fn unsupported_post_push_pop_declines_without_panicking() {
        let mut dfa = UnweightedDfa::new();
        let after_push = dfa.add_state();
        let accepted = dfa.add_state();
        dfa.add_transition(
            dfa.start_state,
            encode_negative_label(7),
            after_push,
        );
        dfa.add_transition(after_push, 9, accepted);
        dfa.set_accepting(accepted, true);

        assert!(try_split_commit_template_dfas(&dfa).is_none());
    }

    #[test]
    fn nfa_dfa_equivalence_checker_detects_corrupted_acceptance() {
        let mut nfa = NFA::new();
        let branch = nfa.add_state();
        let accepted = nfa.add_state();
        nfa.add_epsilon(nfa.start_states[0], branch);
        nfa.add_transition(branch, 5, accepted);
        nfa.set_accepting(accepted);

        let determinized = determinize(&nfa);
        assert_eq!(find_nfa_dfa_language_mismatch(&nfa, &determinized), None);

        let mut corrupted = determinized.clone();
        for state in &mut corrupted.states {
            state.is_accepting = false;
        }
        assert_eq!(find_nfa_dfa_language_mismatch(&nfa, &corrupted), Some(vec![5]));
    }

    #[test]
    fn default_specialization_checker_detects_lost_default_branch() {
        let mut original = UnweightedDfa::new();
        let explicit_accept = original.add_state();
        let default_accept = original.add_state();
        original.add_transition(original.start_state, 7, explicit_accept);
        original.add_transition(original.start_state, DEFAULT_LABEL, default_accept);
        original.set_accepting(explicit_accept, true);
        original.set_accepting(default_accept, true);

        let specialized = specialize_template_dfa_defaults_for_commit_determinized(&original);
        assert_eq!(
            find_default_specialization_mismatch(&original, &specialized),
            None
        );

        let mut corrupted = specialized.clone();
        corrupted.states[corrupted.start_state as usize]
            .transitions
            .remove(&DEFAULT_LABEL);
        assert!(find_default_specialization_mismatch(&original, &corrupted).is_some());
    }

    #[test]
    fn default_specialization_preserves_cross_subset_wildcards() {
        // After consuming 1, both a and b are live. Symbol 2 is explicit only
        // at a, but b's DEFAULT must contribute on 2 as well. Specializing
        // each original state before ordinary determinization loses -20.
        let mut original = UnweightedDfa::new();
        let a = original.add_state();
        let b = original.add_state();
        let a_push = original.add_state();
        let b_push = original.add_state();
        let accepted = original.add_state();
        original.add_transition(0, 1, a);
        original.add_transition(0, DEFAULT_LABEL, b);
        original.add_transition(a, 2, a_push);
        original.add_transition(b, DEFAULT_LABEL, b_push);
        original.add_transition(a_push, encode_negative_label(10), accepted);
        original.add_transition(b_push, encode_negative_label(20), accepted);
        original.set_accepting(accepted, true);

        // Demonstrate the exact pre-fix failure using the old local expansion.
        let mut locally_expanded = NFA::new_empty();
        locally_expanded.states = vec![Default::default(); original.states.len()];
        locally_expanded.start_states = vec![original.start_state];
        for (source, state) in original.states.iter().enumerate() {
            if state.is_accepting { locally_expanded.set_accepting(source as u32); }
            for (&label, &target) in &state.transitions {
                locally_expanded.add_transition(source as u32, label, target);
                if label >= 0 && label != DEFAULT_LABEL
                    && let Some(&wildcard) = state.transitions.get(&DEFAULT_LABEL)
                {
                    locally_expanded.add_transition(source as u32, label, wildcard);
                }
            }
        }
        assert_eq!(find_default_specialization_mismatch(&original, &determinize(&locally_expanded)),
            Some(vec![1, 2, encode_negative_label(20)]));

        let specialized = specialize_template_dfa_defaults_for_commit_determinized(&original);
        assert_eq!(find_default_specialization_mismatch(&original, &specialized), None);
        assert!(specialized.compute_is_acyclic());
        for last in [10, 20] {
            let mut cursor = Some(specialized.start_state);
            for label in [1, 2, encode_negative_label(last)] {
                cursor = super::specialized_default_semantic_advance(&specialized, cursor, label);
            }
            assert!(super::dfa_accepts_at(&specialized, cursor));
        }
        let split = try_split_commit_template_dfas(&specialized)
            .expect("correct specialization retains the pop-then-push phase discipline");
        assert_eq!(find_split_commit_language_mismatch(&specialized, &split), None);
    }

    #[test]
    fn default_specialization_does_not_treat_push_as_wildcard_pop() {
        let mut original = UnweightedDfa::new();
        let accepted = original.add_state();
        original.add_transition(0, DEFAULT_LABEL, accepted);
        original.set_accepting(accepted, true);
        let specialized = specialize_template_dfa_defaults_for_commit_determinized(&original);
        assert_eq!(find_default_specialization_mismatch(&original, &specialized), None);
        assert_eq!(super::specialized_default_semantic_advance(
            &specialized, Some(specialized.start_state), encode_negative_label(0)), None);
    }

    #[test]
    fn default_specialization_preserves_generated_acyclic_relations() {
        // Deterministic generation; no new dependency and no test-order/env
        // dependence. The independent product checker covers all words, not
        // just sampled accepting paths, for each finite graph.
        let alphabet = [0, 1, 2, DEFAULT_LABEL, encode_negative_label(0), encode_negative_label(2)];
        for seed in 1..=128u64 {
            let mut random = seed;
            let mut next = || {
                random = random.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                random
            };
            let mut original = UnweightedDfa::new();
            for _ in 1..9 { original.add_state(); }
            for state in 0..9u32 {
                original.set_accepting(state, next() >> 61 == 0 || state == 8);
                if state == 8 { continue; }
                for &label in &alphabet {
                    if next() >> 62 == 0 { continue; }
                    let target = state + 1 + (next() % (8 - state) as u64) as u32;
                    original.add_transition(state, label, target);
                }
            }
            let specialized = specialize_template_dfa_defaults_for_commit_determinized(&original);
            assert_eq!(find_default_specialization_mismatch(&original, &specialized), None,
                "seed={seed}");
            assert!(specialized.compute_is_acyclic(), "seed={seed}");
        }
    }

}
