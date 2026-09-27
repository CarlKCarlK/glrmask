//! Reuse and transport parser templates across component bindings.

use crate::automata::lexer::Lexer;
use super::{
    AnalyzedGrammar, Arc, BTreeMap, BTreeSet, BitSet, BoundaryTokenDiscovery, ComposedTable,
    Constraint, DEFAULT_LABEL, ExprByteSummary, FxHashMap, Instant, NWA, NWAState, Templates,
    U8Set, UnweightedDfa, VecDeque, Vocab, Weight, build_commit_templates_from_raw_templates,
    characterize_selected_terminals, characterize_terminal_action_state_seeds_for_terminal_count,
    characterize_terminal_nt_predecessor_seeds_for_terminal_count, compose_profile_enabled,
    defer_boundary_commit_templates, encode_negative_label, encode_positive_label,
    expr_byte_summary, is_negative_label, macro_parallelism_disabled, minimize_unweighted_dfa,
    negative_to_positive_label, report_macro_item_timings,
    specialize_template_dfa_defaults_for_commit_split_input, try_split_commit_template_dfas,
};
use rayon::prelude::*;

pub(super) fn remap_composition_template_label(label: i32, state_relation: &[Vec<u32>]) -> Option<i32> {
    if label == DEFAULT_LABEL {
        return Some(label);
    }
    let (local_state, negative) = if is_negative_label(label) {
        (negative_to_positive_label(label) as u32, true)
    } else if label >= 0 {
        (label as u32, false)
    } else {
        return None;
    };
    let mapped = state_relation.get(local_state as usize)?;
    if mapped.len() != 1 {
        return None;
    }
    Some(if negative {
        encode_negative_label(mapped[0])
    } else {
        encode_positive_label(mapped[0])
    })
}

pub(super) fn transport_composition_template_dfa(
    mut dfa: UnweightedDfa,
    state_relation: &[Vec<u32>],
) -> Option<UnweightedDfa> {
    for state in &mut dfa.states {
        let old = std::mem::take(&mut state.transitions);
        let mut mapped = BTreeMap::new();
        for (label, target) in old {
            let mapped_label = remap_composition_template_label(label, state_relation)?;
            if let Some(previous) = mapped.insert(mapped_label, target)
                && previous != target
            {
                return None;
            }
        }
        state.transitions = mapped;
    }
    Some(dfa)
}

/// Transport a cached composition-template DFA and construct its NWA skeleton
/// in the same transition traversal.  The eager composition fast path needs
/// both representations immediately, so rebuilding the skeleton afterward
/// would just walk and allocate every transported transition a second time.
pub(crate) fn transport_composition_template_dfa_with_skeleton(
    mut dfa: UnweightedDfa,
    state_relation: &[Vec<u32>],
) -> Option<(UnweightedDfa, NWA)> {
    let mut nwa_states = Vec::with_capacity(dfa.states.len());
    for state in &mut dfa.states {
        let old = std::mem::take(&mut state.transitions);
        let mut mapped = BTreeMap::new();
        let mut nwa_transitions = BTreeMap::new();
        for (label, target) in old {
            let mapped_label = remap_composition_template_label(label, state_relation)?;
            if let Some(previous) = mapped.insert(mapped_label, target)
                && previous != target
            {
                return None;
            }
            if let Some(previous) = nwa_transitions.insert(
                mapped_label,
                vec![(target, Weight::empty())],
            ) && previous[0].0 != target
            {
                return None;
            }
        }
        state.transitions = mapped;
        nwa_states.push(NWAState {
            final_weight: state.is_accepting.then(Weight::empty),
            transitions: nwa_transitions,
            epsilons: Vec::new(),
        });
    }
    let start_state = dfa.start_state;
    let nwa = NWA::from_parts(nwa_states, vec![start_state]);
    Some((dfa, nwa))
}

/// Cached-only transport used by the eager changed-parent-template finish path.
/// That path already requires every selected unchanged terminal to have a
/// transportable cache entry; returning `None` preserves its existing fallback
/// behavior while avoiding a second DFA->NWA traversal on the all-hit path.
pub(super) fn try_rebuild_cached_transported_component_templates(
    composed_table: &ComposedTable,
    components: &[&Constraint],
    active_terminals: &[bool],
) -> Option<Templates> {
    let mut per_component = Vec::with_capacity(components.len());
    for (component_index, component) in components.iter().copied().enumerate() {
        let terminal_offset = composed_table.terminal_offsets[component_index];
        let relation = &composed_table.state_relations[component_index];
        let selected = (0..component.table.num_terminals as usize)
            .filter_map(|local_terminal| {
                let global_terminal = terminal_offset as usize + local_terminal;
                active_terminals
                    .get(global_terminal)
                    .copied()
                    .unwrap_or(false)
                    .then_some(local_terminal)
            })
            .collect::<Vec<_>>();
        let transport = |&local_terminal: &usize| {
                let dfa = component
                    .composition_parser_templates_by_terminal
                    .get(local_terminal)
                    .and_then(Option::as_ref)
                    .cloned()?;
                let (dfa, nwa) =
                    transport_composition_template_dfa_with_skeleton(dfa, relation)?;
                Some((terminal_offset + local_terminal as u32, dfa, nwa))
        };
        let transported = if macro_parallelism_disabled() {
            let mut timings = Vec::with_capacity(selected.len());
            let result = selected
                .iter()
                .map(|item| {
                    let started = Instant::now();
                    let result = transport(item);
                    timings.push(started.elapsed().as_secs_f64() * 1000.0);
                    result
                })
                .collect::<Vec<_>>();
            report_macro_item_timings("compose_cached_template_transport", &timings);
            result
        } else {
            selected.par_iter().map(transport).collect::<Vec<_>>()
        };
        if transported.iter().any(Option::is_none) {
            return None;
        }
        per_component.extend(transported.into_iter().flatten());
    }

    let mut by_terminal = BTreeMap::new();
    let mut by_terminal_nwa = BTreeMap::new();
    for (terminal, dfa, nwa) in per_component {
        by_terminal.insert(terminal, dfa);
        by_terminal_nwa.insert(terminal, nwa);
    }
    Some(Templates {
        by_terminal,
        by_terminal_nwa,
    })
}

pub(super) fn unweighted_dfa_difference(left: &UnweightedDfa, right: &UnweightedDfa) -> UnweightedDfa {
    type Pair = (u32, Option<u32>);
    let start: Pair = (left.start_state, Some(right.start_state));
    let mut output = UnweightedDfa::new();
    let mut state_by_pair = FxHashMap::<Pair, u32>::default();
    state_by_pair.insert(start, output.start_state);
    let mut queue = VecDeque::from([start]);

    while let Some((left_state, right_state)) = queue.pop_front() {
        let output_state = state_by_pair[&(left_state, right_state)];
        let left_node = &left.states[left_state as usize];
        let right_accepting = right_state
            .and_then(|state| right.states.get(state as usize))
            .is_some_and(|state| state.is_accepting);
        output.set_accepting(output_state, left_node.is_accepting && !right_accepting);
        for (&label, &left_target) in &left_node.transitions {
            let right_target = right_state
                .and_then(|state| right.states.get(state as usize))
                .and_then(|state| state.transitions.get(&label).copied());
            let pair = (left_target, right_target);
            let target = if let Some(&target) = state_by_pair.get(&pair) {
                target
            } else {
                let target = output.add_state();
                state_by_pair.insert(pair, target);
                queue.push_back(pair);
                target
            };
            output.add_transition(output_state, label, target);
        }
    }
    output
}

pub(super) fn unweighted_dfa_language_is_empty(dfa: &UnweightedDfa) -> bool {
    let mut seen = vec![false; dfa.states.len()];
    let mut stack = vec![dfa.start_state];
    while let Some(state) = stack.pop() {
        let Some(seen_state) = seen.get_mut(state as usize) else {
            continue;
        };
        if *seen_state {
            continue;
        }
        *seen_state = true;
        let node = &dfa.states[state as usize];
        if node.is_accepting {
            return false;
        }
        stack.extend(node.transitions.values().copied());
    }
    true
}

pub(super) fn unweighted_dfa_shortest_word(dfa: &UnweightedDfa) -> Option<Vec<i32>> {
    if dfa.states.is_empty() {
        return None;
    }
    let mut queue = VecDeque::from([dfa.start_state]);
    let mut previous = vec![None::<(u32, i32)>; dfa.states.len()];
    let mut seen = vec![false; dfa.states.len()];
    seen[dfa.start_state as usize] = true;
    while let Some(state) = queue.pop_front() {
        if dfa.states[state as usize].is_accepting {
            let mut word = Vec::new();
            let mut current = state;
            while current != dfa.start_state {
                let (parent, label) = previous[current as usize]?;
                word.push(label);
                current = parent;
            }
            word.reverse();
            return Some(word);
        }
        for (&label, &target) in &dfa.states[state as usize].transitions {
            if !seen[target as usize] {
                seen[target as usize] = true;
                previous[target as usize] = Some((state, label));
                queue.push_back(target);
            }
        }
    }
    None
}

pub(super) fn trim_unweighted_dfa_productive(dfa: UnweightedDfa) -> UnweightedDfa {
    if dfa.states.is_empty() {
        return dfa;
    }
    let mut predecessors = vec![Vec::<u32>::new(); dfa.states.len()];
    let mut productive = vec![false; dfa.states.len()];
    let mut queue = VecDeque::<u32>::new();
    for (state_id, state) in dfa.states.iter().enumerate() {
        if state.is_accepting {
            productive[state_id] = true;
            queue.push_back(state_id as u32);
        }
        for &target in state.transitions.values() {
            predecessors[target as usize].push(state_id as u32);
        }
    }
    while let Some(target) = queue.pop_front() {
        for &source in &predecessors[target as usize] {
            if !productive[source as usize] {
                productive[source as usize] = true;
                queue.push_back(source);
            }
        }
    }
    if !productive.get(dfa.start_state as usize).copied().unwrap_or(false) {
        return UnweightedDfa::new();
    }
    let mut remap = vec![u32::MAX; dfa.states.len()];
    let mut states = Vec::with_capacity(productive.iter().filter(|&&value| value).count());
    for (old, state) in dfa.states.iter().enumerate() {
        if productive[old] {
            remap[old] = states.len() as u32;
            states.push(crate::automata::unweighted_u32::dfa::DFAState {
                is_accepting: state.is_accepting,
                transitions: BTreeMap::new(),
            });
        }
    }
    for (old, state) in dfa.states.iter().enumerate() {
        if !productive[old] {
            continue;
        }
        let new = remap[old] as usize;
        for (&label, &target) in &state.transitions {
            let mapped = remap[target as usize];
            if mapped != u32::MAX {
                states[new].transitions.insert(label, mapped);
            }
        }
    }
    UnweightedDfa { states, start_state: remap[dfa.start_state as usize] }
}

pub(super) fn rebuild_transported_component_templates(
    composed_table: &ComposedTable,
    components: &[&Constraint],
    active_terminals: &[bool],
) -> BTreeMap<u32, UnweightedDfa> {
    let profile_transport =
        std::env::var_os("GLRMASK_PROFILE_TEMPLATE_TRANSPORT").is_some();
    let mut result = BTreeMap::new();
    for (component_index, component) in components.iter().copied().enumerate() {
        let component_started = Instant::now();
        let terminal_offset = composed_table.terminal_offsets[component_index];
        let mut selected = vec![false; component.table.num_terminals as usize];
        for (local, selected_slot) in selected.iter_mut().enumerate() {
            let global = terminal_offset as usize + local;
            *selected_slot = active_terminals.get(global).copied().unwrap_or(false);
        }
        if !selected.iter().any(|&value| value) {
            continue;
        }
        let mut missing = vec![false; selected.len()];
        let relation = &composed_table.state_relations[component_index];
        let cached_started = Instant::now();
        let transport_cached = |(local_terminal, &is_selected): (usize, &bool)| {
                is_selected.then(|| {
                    let transported = component
                        .composition_parser_templates_by_terminal
                        .get(local_terminal)
                        .and_then(Option::as_ref)
                        .cloned()
                        .and_then(|dfa| transport_composition_template_dfa(dfa, relation));
                    (local_terminal, transported)
                })
        };
        let cached_results = if macro_parallelism_disabled() {
            let mut timings = Vec::new();
            let result = selected
                .iter()
                .enumerate()
                .filter_map(|item| {
                    let started = Instant::now();
                    let result = transport_cached(item);
                    timings.push(started.elapsed().as_secs_f64() * 1000.0);
                    result
                })
                .collect::<Vec<_>>();
            report_macro_item_timings("compose_cached_template_transport_selected", &timings);
            result
        } else {
            selected
                .par_iter()
                .enumerate()
                .filter_map(transport_cached)
                .collect::<Vec<_>>()
        };
        let cached_ms = cached_started.elapsed().as_secs_f64() * 1000.0;
        let insert_started = Instant::now();
        let mut cache_hits = 0usize;
        let mut cache_states = 0usize;
        let mut cache_transitions = 0usize;
        for (local_terminal, transported) in cached_results {
            if let Some(transported) = transported {
                cache_hits += 1;
                cache_states += transported.states.len();
                cache_transitions += transported
                    .states
                    .iter()
                    .map(|state| state.transitions.len())
                    .sum::<usize>();
                let global_terminal = terminal_offset + local_terminal as u32;
                result.insert(global_terminal, transported);
            } else {
                missing[local_terminal] = true;
            }
        }
        let insert_ms = insert_started.elapsed().as_secs_f64() * 1000.0;
        if !missing.iter().any(|&value| value) {
            if profile_transport {
                eprintln!(
                    "[glrmask/profile][composition_template_transport] component={} selected={} cache_hits={} cache_misses=0 cache_states={} cache_transitions={} cached_ms={cached_ms:.3} insert_ms={insert_ms:.3} fallback_ms=0.000 total_ms={:.3}",
                    component_index,
                    selected.iter().filter(|&&value| value).count(),
                    cache_hits,
                    cache_states,
                    cache_transitions,
                    component_started.elapsed().as_secs_f64() * 1000.0,
                );
            }
            continue;
        }
        let fallback_started = Instant::now();
        let Some(augmented_start) = component.table.rules.first().map(|rule| rule.lhs) else {
            continue;
        };
        let analyzed = AnalyzedGrammar::from_composed_rules(
            component.table.rules.clone(),
            component.table.num_terminals,
            component.terminal_display_names().to_vec(),
            component.table.nonterminal_display_names.clone(),
            augmented_start,
        );
        let characterizations = characterize_selected_terminals(&component.table, &analyzed, &missing);
        let templates = Templates::from_characterizations(&characterizations);
        for (local_terminal, old_template) in templates.by_terminal {
            let global_terminal = terminal_offset + local_terminal;
            let Some(transported) = transport_composition_template_dfa(
                old_template,
                &composed_table.state_relations[component_index],
            ) else {
                continue;
            };
            result.insert(global_terminal, transported);
        }
        if profile_transport {
            eprintln!(
                "[glrmask/profile][composition_template_transport] component={} selected={} cache_hits={} cache_misses={} cache_states={} cache_transitions={} cached_ms={cached_ms:.3} insert_ms={insert_ms:.3} fallback_ms={:.3} total_ms={:.3}",
                component_index,
                selected.iter().filter(|&&value| value).count(),
                cache_hits,
                missing.iter().filter(|&&value| value).count(),
                cache_states,
                cache_transitions,
                fallback_started.elapsed().as_secs_f64() * 1000.0,
                component_started.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }
    result
}

pub(super) fn build_complete_composed_parser_template_cache(
    composed_table: &ComposedTable,
    components: &[&Constraint],
    analyzed: &AnalyzedGrammar,
    scoped_ignore_terminals: &BitSet,
) -> Vec<Option<UnweightedDfa>> {
    let terminal_count = analyzed.num_terminals as usize;
    let parent_terminal_end = composed_table
        .terminal_offsets
        .get(1)
        .copied()
        .unwrap_or(analyzed.num_terminals) as usize;

    // The linker rewrites parent continuation semantics. Ordinary child rows
    // are copied and their LR-state labels are transported through the exact
    // state relation; scoped child ignores are the child-side exception.
    let mut fresh = vec![false; terminal_count];
    fresh[..parent_terminal_end.min(terminal_count)].fill(true);
    for terminal in scoped_ignore_terminals.iter() {
        if let Some(slot) = fresh.get_mut(terminal) {
            *slot = true;
        }
    }

    let mut reuse = vec![false; terminal_count];
    for terminal in parent_terminal_end.min(terminal_count)..terminal_count {
        reuse[terminal] = !fresh[terminal];
    }
    let transported = rebuild_transported_component_templates(
        composed_table,
        components,
        &reuse,
    );

    let characterizations =
        characterize_selected_terminals(&composed_table.table, analyzed, &fresh);
    let fresh_templates = Templates::from_characterizations(&characterizations);
    let mut result = vec![None; terminal_count];
    for (terminal, dfa) in transported {
        if let Some(slot) = result.get_mut(terminal as usize) {
            *slot = Some(dfa);
        }
    }
    for (terminal, dfa) in fresh_templates.by_terminal {
        if let Some(slot) = result.get_mut(terminal as usize) {
            *slot = Some(dfa);
        }
    }

    // A small number of child templates can fail direct transport when their
    // standalone stack-effect language mentions a start/accept state whose
    // composition relation is one-to-many. Re-characterize only those missing
    // terminals in the already-composed table so the persisted cache is truly
    // complete for future nested composition.
    let missing = result.iter().map(Option::is_none).collect::<Vec<_>>();
    if missing.iter().any(|&value| value) {
        let missing_characterizations =
            characterize_selected_terminals(&composed_table.table, analyzed, &missing);
        let missing_templates = Templates::from_characterizations(&missing_characterizations);
        for (terminal, dfa) in missing_templates.by_terminal {
            if let Some(slot) = result.get_mut(terminal as usize) {
                *slot = Some(dfa);
            }
        }
    }
    result
}


#[derive(Debug, Clone)]
pub(super) struct ConcreteBoundaryDeltaEntry {
    pub(super) old_terminal: u32,
    pub(super) old_template: UnweightedDfa,
    pub(super) delta_terminal: u32,
    pub(super) delta_template: UnweightedDfa,
}

#[derive(Debug, Clone)]
pub(super) struct ConcreteBoundaryDeltaPlan {
    pub(super) original_num_terminals: u32,
    pub(super) synthetic_num_terminals: u32,
    pub(super) by_global_terminal: BTreeMap<u32, ConcreteBoundaryDeltaEntry>,
    /// Active terminals whose transported standalone template and composed
    /// template were compared exactly. An empty standalone language is represented
    /// by an ordinary empty DFA; a missing transported template is unknown and
    /// therefore conservative, never interpreted as the empty language.
    pub(super) compared_terminals: BTreeSet<u32>,
    /// Active terminals for which we could not prove `Old ⊆ New` after
    /// transporting the cached component template into composed parser
    /// coordinates. Any local path touching one of these stays on the full
    /// composed-template lane.
    pub(super) unsafe_terminals: BTreeSet<u32>,
}

pub(super) fn prepare_concrete_boundary_delta_plan(
    composed_table: &ComposedTable,
    components: &[&Constraint],
    active_terminals: &[bool],
    composed_templates: &Templates,
    original_num_terminals: u32,
) -> ConcreteBoundaryDeltaPlan {
    let old_templates = rebuild_transported_component_templates(
        composed_table,
        components,
        active_terminals,
    );
    let mut by_global_terminal = BTreeMap::new();
    let mut compared_terminals = BTreeSet::new();
    let mut unsafe_terminals = BTreeSet::new();
    let mut next_terminal = original_num_terminals;

    for (terminal, &active) in active_terminals.iter().enumerate() {
        if !active {
            continue;
        }
        let terminal = terminal as u32;
        let Some(new) = composed_templates.by_terminal.get(&terminal) else {
            unsafe_terminals.insert(terminal);
            continue;
        };
        let Some(old) = old_templates.get(&terminal) else {
            // Characterization produces an explicit empty DFA when the standalone
            // terminal has no parser action. Missing here therefore means template
            // transport failed (for example because the state relation is not
            // functionally representable), not Old=∅. Keep the full conservative lane.
            unsafe_terminals.insert(terminal);
            continue;
        };
        compared_terminals.insert(terminal);

        let removed = unweighted_dfa_difference(old, new);
        if !unweighted_dfa_language_is_empty(&removed) {
            unsafe_terminals.insert(terminal);
            continue;
        }
        let delta = trim_unweighted_dfa_productive(unweighted_dfa_difference(new, old));
        if unweighted_dfa_language_is_empty(&delta) {
            continue;
        }
        let old_terminal = next_terminal;
        let delta_terminal = next_terminal + 1;
        next_terminal += 2;
        by_global_terminal.insert(
            terminal,
            ConcreteBoundaryDeltaEntry {
                old_terminal,
                old_template: old.clone(),
                delta_terminal,
                delta_template: delta,
            },
        );
    }

    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_boundary_delta_plan] deltas={} unsafe={} unsafe_ids={:?}",
            by_global_terminal.len(),
            unsafe_terminals.len(),
            unsafe_terminals.iter().take(64).copied().collect::<Vec<_>>(),
        );
    }

    ConcreteBoundaryDeltaPlan {
        original_num_terminals,
        synthetic_num_terminals: next_terminal,
        by_global_terminal,
        compared_terminals,
        unsafe_terminals,
    }
}

pub(super) fn finish_eager_concrete_boundary_delta_plan(
    precomputed: EagerConcreteDeltaPrecompute,
    active_terminals: &[bool],
    original_num_terminals: u32,
) -> ConcreteBoundaryDeltaPlan {
    let mut by_global_terminal = BTreeMap::new();
    let mut next_terminal = original_num_terminals;
    for (terminal, (old_template, delta_template)) in precomputed.delta_templates {
        if !active_terminals
            .get(terminal as usize)
            .copied()
            .unwrap_or(false)
        {
            continue;
        }
        let old_terminal = next_terminal;
        let delta_terminal = next_terminal + 1;
        next_terminal += 2;
        by_global_terminal.insert(
            terminal,
            ConcreteBoundaryDeltaEntry {
                old_terminal,
                old_template,
                delta_terminal,
                delta_template,
            },
        );
    }
    let unsafe_terminals = precomputed
        .unsafe_terminals
        .into_iter()
        .filter(|&terminal| {
            active_terminals
                .get(terminal as usize)
                .copied()
                .unwrap_or(false)
        })
        .collect::<BTreeSet<_>>();
    let compared_terminals = active_terminals
        .iter()
        .enumerate()
        .filter_map(|(terminal, &active)| {
            (active && !unsafe_terminals.contains(&(terminal as u32)))
                .then_some(terminal as u32)
        })
        .collect();
    ConcreteBoundaryDeltaPlan {
        original_num_terminals,
        synthetic_num_terminals: next_terminal,
        by_global_terminal,
        compared_terminals,
        unsafe_terminals,
    }
}


/// Exact reset/base-case support for a one-terminal parser-template delta.
///
/// `collect_one_byte_seed_relations*` covers arbitrary lexer states but only
/// one-byte model tokens.  This companion relation covers arbitrary-length
/// model tokens that can complete one selected grammar terminal exactly at the
/// lexer reset.  The parser word is still length one, so it is governed by the
/// same Old/New/Delta factorization.
pub(super) fn boundary_delta_reset_relations(
    components: &[&Constraint],
    terminal_offsets: &[u32],
    vocab: &Vocab,
    selected_terminals: &[bool],
    control_terminals: &BTreeSet<u32>,
) -> BTreeMap<Vec<u32>, BTreeMap<u32, BTreeSet<u32>>> {
    debug_assert_eq!(components.len(), terminal_offsets.len());
    let started_at = Instant::now();

    let mut selected_by_component = Vec::<BitSet>::with_capacity(components.len());
    for (component_index, component) in components.iter().enumerate() {
        let terminal_offset = terminal_offsets[component_index];
        let mut selected = BitSet::new(component.tokenizer.num_terminals() as usize);
        for local in 0..component.tokenizer.num_terminals() {
            let global = terminal_offset + local;
            if selected_terminals
                .get(global as usize)
                .copied()
                .unwrap_or(false)
                && !control_terminals.contains(&global)
            {
                selected.set(local as usize);
            }
        }
        selected_by_component.push(selected);
    }

    let mut pairs = Vec::<(u32, u32)>::new();
    let mut fallback_components = Vec::<usize>::new();
    let mut cached_components = 0usize;
    for (component_index, component) in components.iter().enumerate() {
        let selected = &selected_by_component[component_index];
        if selected.is_empty() {
            continue;
        }
        if component.composition_reset_tokens_by_terminal.len()
            == component.tokenizer.num_terminals() as usize
        {
            cached_components += 1;
            let terminal_offset = terminal_offsets[component_index];
            for local_terminal in selected.iter() {
                if let Some(tokens) = component
                    .composition_reset_tokens_by_terminal
                    .get(local_terminal)
                {
                    pairs.extend(
                        tokens
                            .iter()
                            .copied()
                            .map(|token| (terminal_offset + local_terminal as u32, token)),
                    );
                }
            }
        } else {
            fallback_components.push(component_index);
        }
    }

    // Old/unprepared artifacts remain exact. Restrict their vocabulary scan
    // only with byte-language facts that are necessary for an exact terminal
    // match: first byte, last byte, and bytes reachable somewhere in the
    // terminal expression. These summaries are deliberately conservative
    // (opaque/missing expressions become all-bytes), and every survivor still
    // runs through the authoritative tokenizer below before publication.
    //
    // Do not use `possible_matches` here. That structure indexes runtime
    // delayed-match queries and is not a superset of all reset-complete tokens.
    let mut candidate_scan_components = Vec::<(usize, Vec<u32>)>::new();
    let mut candidate_tokens = 0usize;
    for component_index in fallback_components {
        let component = components[component_index];
        let selected = &selected_by_component[component_index];
        let summaries = selected
            .iter()
            .map(|local_terminal| {
                component
                    .retained_terminal_expr(local_terminal as u32)
                    .map(expr_byte_summary)
                    .unwrap_or(ExprByteSummary {
                        nullable: false,
                        first: U8Set::all(),
                        last: U8Set::all(),
                        reachable: U8Set::all(),
                    })
            })
            .collect::<Vec<_>>();
        let candidates = vocab
            .entries_map()
            .iter()
            .filter_map(|(&token_id, bytes)| {
                let (&first, rest) = bytes.split_first()?;
                let last = rest.last().copied().unwrap_or(first);
                summaries
                    .iter()
                    .any(|summary| {
                        summary.first.contains(first)
                            && summary.last.contains(last)
                            && bytes.iter().all(|&byte| summary.reachable.contains(byte))
                    })
                    .then_some(token_id)
            })
            .collect::<Vec<_>>();
        candidate_tokens += candidates.len();
        candidate_scan_components.push((component_index, candidates));
    }

    let mut candidate_pairs = candidate_scan_components
        .par_iter()
        .flat_map_iter(|(component_index, candidates)| {
            let component_index = *component_index;
            let component = components[component_index];
            let selected = &selected_by_component[component_index];
            candidates.iter().filter_map(move |&token_id| {
                let bytes = vocab.entries_map().get(&token_id)?;
                if bytes.is_empty() {
                    return None;
                }
                let (_, matches) = component
                    .tokenizer
                    .execute_summary_from_state(bytes, component.tokenizer.start_state());
                let terminals = matches
                    .into_iter()
                    .filter_map(|(local_terminal, width)| {
                        (width == bytes.len() && selected.contains(local_terminal as usize))
                            .then_some(terminal_offsets[component_index] + local_terminal)
                    })
                    .collect::<Vec<_>>();
                (!terminals.is_empty()).then_some((token_id, terminals))
            })
        })
        .flat_map_iter(|(token_id, terminals)| {
            terminals.into_iter().map(move |terminal| (terminal, token_id))
        })
        .collect::<Vec<_>>();
    pairs.append(&mut candidate_pairs);

    let mut result = BTreeMap::<Vec<u32>, BTreeMap<u32, BTreeSet<u32>>>::new();
    for (terminal, token) in pairs {
        result
            .entry(vec![terminal])
            .or_default()
            .entry(0)
            .or_default()
            .insert(token);
    }
    if compose_profile_enabled() {
        let token_cells = result
            .values()
            .flat_map(|by_state| by_state.values())
            .map(BTreeSet::len)
            .sum::<usize>();
        eprintln!(
            "[glrmask/profile][constraint_boundary_delta_reset_relations] terminals={} token_cells={} cached_components={} candidate_components={} candidate_tokens={} ms={:.3}",
            result.len(),
            token_cells,
            cached_components,
            candidate_scan_components.len(),
            candidate_tokens,
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    result
}

pub(super) fn merge_one_terminal_relations(
    into: &mut BTreeMap<Vec<u32>, BTreeMap<u32, BTreeSet<u32>>>,
    from: BTreeMap<Vec<u32>, BTreeMap<u32, BTreeSet<u32>>>,
) {
    for (sequence, by_state) in from {
        let target_by_state = into.entry(sequence).or_default();
        for (state, tokens) in by_state {
            target_by_state.entry(state).or_default().extend(tokens);
        }
    }
}

/// Rewrite one-terminal boundary seeds into the exact additive parser-template
/// novelty supplied by composition.  A transported component normally already
/// supplies Old_t, so a safe changed terminal contributes only
/// Delta_t = New_t \\ Old_t and an unchanged compared terminal contributes
/// nothing.  The exception is support shadowed by the normalized component
/// parser's start-final acceptance: prefix-final subtraction removed Old_t's
/// outgoing branch for that exact (lexer-state, model-token) support cell, so
/// the boundary must retain full New_t there.
pub(super) fn factor_one_terminal_seed_relations(
    relations: BTreeMap<Vec<u32>, BTreeMap<u32, BTreeSet<u32>>>,
    plan: &ConcreteBoundaryDeltaPlan,
    components: &[&Constraint],
    tokenizer_state_offsets: &[u32],
    terminal_offsets: &[u32],
) -> BTreeMap<Vec<u32>, BTreeMap<u32, BTreeSet<u32>>> {
    debug_assert_eq!(components.len(), tokenizer_state_offsets.len());
    debug_assert_eq!(components.len(), terminal_offsets.len());

    let owner_for_terminal = |terminal: u32| -> Option<usize> {
        let component = terminal_offsets
            .partition_point(|&offset| offset <= terminal)
            .checked_sub(1)?;
        let local = terminal.checked_sub(*terminal_offsets.get(component)?)?;
        (local < components.get(component)?.tokenizer.num_terminals()).then_some(component)
    };

    let scoped_ignore_shadow_is_separately_supplied = |terminal: u32| -> bool {
        if std::env::var_os("GLRMASK_EXPERIMENT_SCOPED_IGNORE_SHADOW_REUSE").is_none()
            || std::env::var_os("GLRMASK_EXPERIMENT_COMPONENT_SCOPED_IGNORE_TOP_ACCEPT").is_none()
        {
            return false;
        }
        let Some(component_index) = owner_for_terminal(terminal) else {
            return false;
        };
        let Some(ignore) = components[component_index].ignore_terminal else {
            return false;
        };
        terminal_offsets[component_index] + ignore == terminal
    };

    let support_shadowed_at_component_start =
        |terminal: u32, raw_state: u32, original_token: u32| -> Option<bool> {
            let component_index = owner_for_terminal(terminal)?;
            let component = *components.get(component_index)?;
            let state_offset = *tokenizer_state_offsets.get(component_index)?;
            let local_state = if raw_state == 0 {
                component.tokenizer.start_state()
            } else {
                let local = raw_state.checked_sub(state_offset)?;
                (local < component.tokenizer.num_states()).then_some(local)?
            };
            let internal_token = component.original_token_internal_at(original_token)?;
            if internal_token == u32::MAX {
                return None;
            }
            let start_final = component
                .parser_dwa
                .states()
                .get(component.parser_dwa.start_state() as usize)?
                .final_weight
                .as_ref();
            let Some(start_final) = start_final else {
                return Some(false);
            };
            Some(
                component
                    .internal_tsids_for_state(local_state)
                    .iter()
                    .copied()
                    .any(|tsid| start_final.tokens_for_tsid(tsid).contains(internal_token)),
            )
        };

    fn insert_relation(
        factored: &mut BTreeMap<Vec<u32>, BTreeMap<u32, BTreeSet<u32>>>,
        label: u32,
        state: u32,
        token: u32,
    ) {
        factored
            .entry(vec![label])
            .or_default()
            .entry(state)
            .or_default()
            .insert(token);
    }

    let mut factored = BTreeMap::<Vec<u32>, BTreeMap<u32, BTreeSet<u32>>>::new();
    let mut promoted_shadow_cells = 0usize;
    let mut promoted_shadow_tokens = BTreeSet::<u32>::new();
    let mut promoted_shadow_by_terminal = BTreeMap::<u32, BTreeSet<u32>>::new();
    let mut reused_scoped_ignore_shadow_cells = 0usize;
    let mut dropped_cells = 0usize;
    let mut delta_cells = 0usize;
    let mut conservative_cells = 0usize;

    for (sequence, by_state) in relations {
        if sequence.len() != 1 {
            // This helper is deliberately only the n=1 base case of the
            // first-delta decomposition.  Preserve any future non-unit caller
            // conservatively rather than silently changing its language.
            factored.insert(sequence, by_state);
            continue;
        }
        let terminal = sequence[0];
        let entry = plan.by_global_terminal.get(&terminal);
        let unsafe_or_unclassified = plan.unsafe_terminals.contains(&terminal)
            || (!plan.compared_terminals.contains(&terminal) && entry.is_none());

        for (state, tokens) in by_state {
            for token in tokens {
                if unsafe_or_unclassified {
                    conservative_cells += 1;
                    insert_relation(&mut factored, terminal, state, token);
                    continue;
                }
                match support_shadowed_at_component_start(terminal, state, token) {
                    Some(true) => {
                        if scoped_ignore_shadow_is_separately_supplied(terminal) {
                            reused_scoped_ignore_shadow_cells += 1;
                            continue;
                        }
                        promoted_shadow_cells += 1;
                        promoted_shadow_tokens.insert(token);
                        promoted_shadow_by_terminal.entry(terminal).or_default().insert(token);
                        insert_relation(&mut factored, terminal, state, token);
                    }
                    Some(false) => {
                        if let Some(entry) = entry {
                            delta_cells += 1;
                            insert_relation(&mut factored, entry.delta_terminal, state, token);
                        } else {
                            // Compared and absent from the delta map means
                            // Old_t == New_t; the component already supplies it.
                            dropped_cells += 1;
                        }
                    }
                    None => {
                        // If ownership/coordinate provenance cannot be proved,
                        // keep full New_t.  Correctness beats the optimization.
                        conservative_cells += 1;
                        insert_relation(&mut factored, terminal, state, token);
                    }
                }
            }
        }
    }

    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_boundary_factor_one_terminal] rows={} promoted_shadow_cells={} promoted_shadow_tokens={} promoted_shadow_by_terminal={:?} reused_scoped_ignore_shadow_cells={} delta_cells={} dropped_cells={} conservative_cells={}",
            factored.len(),
            promoted_shadow_cells,
            promoted_shadow_tokens.len(),
            promoted_shadow_by_terminal.iter().map(|(&terminal, tokens)| (terminal, tokens.len())).collect::<Vec<_>>(),
            reused_scoped_ignore_shadow_cells,
            delta_cells,
            dropped_cells,
            conservative_cells,
        );
    }
    factored
}

pub(super) fn unweighted_dfa_to_template_nwa(dfa: &UnweightedDfa) -> NWA {
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
        .collect::<Vec<_>>();
    NWA::from_parts(states, vec![dfa.start_state])
}

pub(super) fn install_concrete_boundary_delta_templates(
    templates: &mut Templates,
    plan: &ConcreteBoundaryDeltaPlan,
) {
    for entry in plan.by_global_terminal.values() {
        templates
            .by_terminal
            .insert(entry.old_terminal, entry.old_template.clone());
        templates.by_terminal_nwa.insert(
            entry.old_terminal,
            unweighted_dfa_to_template_nwa(&entry.old_template),
        );
        templates
            .by_terminal
            .insert(entry.delta_terminal, entry.delta_template.clone());
        templates.by_terminal_nwa.insert(
            entry.delta_terminal,
            unweighted_dfa_to_template_nwa(&entry.delta_template),
        );
    }
}

pub(super) fn profile_current_boundary_template_delta(
    composed_table: &ComposedTable,
    components: &[&Constraint],
    active_terminals: &[bool],
    composed_templates: &Templates,
    boundary_paths: &BoundaryTokenDiscovery,
    tokenizer_state_offsets: &[u32],
) {
    if std::env::var_os("GLRMASK_PROFILE_CURRENT_BOUNDARY_TEMPLATE_DELTA").is_none() {
        return;
    }
    let started = Instant::now();
    let old_templates = rebuild_transported_component_templates(
        composed_table,
        components,
        active_terminals,
    );
    let mut changed = BTreeSet::<u32>::new();
    let mut incomparable = Vec::<u32>::new();
    let mut compared = 0usize;
    let mut old_states = 0usize;
    let mut new_states = 0usize;
    let mut delta_states = 0usize;
    let mut delta_transitions = 0usize;
    let mut minimized_delta_states = 0usize;
    let mut minimized_delta_transitions = 0usize;
    for (&terminal, old) in &old_templates {
        let Some(new) = composed_templates.by_terminal.get(&terminal) else {
            continue;
        };
        compared += 1;
        let removed = unweighted_dfa_difference(old, new);
        if !unweighted_dfa_language_is_empty(&removed) {
            if compose_profile_enabled() {
                eprintln!(
                    "[glrmask/profile][current_boundary_template_removed] terminal={} name={} old_states={} new_states={} witness={:?}",
                    terminal,
                    composed_templates
                        .by_terminal
                        .contains_key(&terminal)
                        .then(|| terminal.to_string())
                        .unwrap_or_default(),
                    old.states.len(),
                    new.states.len(),
                    unweighted_dfa_shortest_word(&removed),
                );
            }
            incomparable.push(terminal);
            continue;
        }
        let delta = trim_unweighted_dfa_productive(unweighted_dfa_difference(new, old));
        if !unweighted_dfa_language_is_empty(&delta) {
            changed.insert(terminal);
            delta_states += delta.states.len();
            delta_transitions += delta.states.iter().map(|state| state.transitions.len()).sum::<usize>();
            let minimized = minimize_unweighted_dfa(&delta);
            minimized_delta_states += minimized.states.len();
            minimized_delta_transitions += minimized
                .states
                .iter()
                .map(|state| state.transitions.len())
                .sum::<usize>();
        }
        old_states += old.states.len();
        new_states += new.states.len();
    }

    let terminal_component = |terminal: u32| -> usize {
        composed_table
            .terminal_offsets
            .partition_point(|&offset| offset <= terminal)
            .saturating_sub(1)
    };
    let state_component = |state: u32| -> Option<usize> {
        if state == 0 {
            None
        } else {
            Some(
                tokenizer_state_offsets
                    .partition_point(|&offset| offset <= state)
                    .saturating_sub(1),
            )
        }
    };
    let mut witnesses_with_cross = 0usize;
    let mut witnesses_with_local = 0usize;
    let mut local_all_one = 0usize;
    let mut local_with_zero = 0usize;
    let mut local_with_one = 0usize;
    let mut local_with_multi = 0usize;
    let mut local_only_all_one = 0usize;
    for witness in &boundary_paths.witnesses {
        let mut reach = vec![BTreeSet::<(Option<usize>, bool, u8)>::new(); witness.nodes.len()];
        for component in witness.start_states.iter().copied().map(state_component) {
            reach[0].insert((component, false, 0));
        }
        let mut order = (0..witness.nodes.len()).collect::<Vec<_>>();
        order.sort_unstable_by_key(|&node| witness.nodes[node].key.offset);
        for source in order {
            if !witness.good[source] || reach[source].is_empty() {
                continue;
            }
            let source_reach = reach[source].clone();
            for edge in witness.nodes[source]
                .outgoing
                .iter()
                .filter(|edge| witness.good[edge.target])
            {
                let next_component = terminal_component(edge.terminal);
                let add = u8::from(changed.contains(&edge.terminal));
                for &(last_component, crossed, count) in &source_reach {
                    let next_crossed = crossed
                        || last_component.is_some_and(|component| component != next_component);
                    reach[edge.target].insert((
                        Some(next_component),
                        next_crossed,
                        count.saturating_add(add).min(2),
                    ));
                }
            }
        }
        let mut local_counts = BTreeSet::<u8>::new();
        let mut has_cross = false;
        for (node, &accepting) in witness.accepting.iter().enumerate() {
            if !accepting {
                continue;
            }
            for &(_, crossed, count) in &reach[node] {
                if crossed {
                    has_cross = true;
                } else {
                    local_counts.insert(count);
                }
            }
        }
        if has_cross {
            witnesses_with_cross += 1;
        }
        if !local_counts.is_empty() {
            witnesses_with_local += 1;
            local_with_zero += usize::from(local_counts.contains(&0));
            local_with_one += usize::from(local_counts.contains(&1));
            local_with_multi += usize::from(local_counts.contains(&2));
            if local_counts.len() == 1 && local_counts.contains(&1) {
                local_all_one += 1;
                if !has_cross {
                    local_only_all_one += 1;
                }
            }
        }
    }
    eprintln!(
        "[glrmask/profile][current_boundary_template_delta] compared={} old_templates={} changed={} incomparable={} old_states={} new_states={} delta_states={} delta_transitions={} minimized_delta_states={} minimized_delta_transitions={} witnesses={} witnesses_with_cross={} witnesses_with_local={} local_all_one={} local_only_all_one={} local_with_zero={} local_with_one={} local_with_multi={} changed_ids={:?} incomparable_ids={:?} total_ms={:.3}",
        compared,
        old_templates.len(),
        changed.len(),
        incomparable.len(),
        old_states,
        new_states,
        delta_states,
        delta_transitions,
        minimized_delta_states,
        minimized_delta_transitions,
        boundary_paths.witnesses.len(),
        witnesses_with_cross,
        witnesses_with_local,
        local_all_one,
        local_only_all_one,
        local_with_zero,
        local_with_one,
        local_with_multi,
        changed,
        incomparable,
        started.elapsed().as_secs_f64() * 1000.0,
    );
}

pub(super) fn build_composition_templates(
    table: &crate::compiler::glr::table::GLRTable,
    analyzed: &AnalyzedGrammar,
    selected: &[bool],
) -> (
    Templates,
    Vec<Option<Arc<crate::runtime::CommitTemplateDfas>>>,
    f64,
) {
    let started_at = Instant::now();
    let characterize_started_at = Instant::now();
    let characterizations = characterize_selected_terminals(table, analyzed, selected);
    let characterize_ms = characterize_started_at.elapsed().as_secs_f64() * 1000.0;
    let templates_started_at = Instant::now();
    let templates = Templates::from_characterizations(&characterizations);
    let from_characterizations_ms = templates_started_at.elapsed().as_secs_f64() * 1000.0;
    let commit_started_at = Instant::now();
    let mut template_dfas_by_terminal = vec![None; analyzed.num_terminals as usize];
    if !defer_boundary_commit_templates() {
        let split = |(&terminal, dfa): (&u32, &UnweightedDfa)| {
                let commit_dfa = specialize_template_dfa_defaults_for_commit_split_input(dfa);
                try_split_commit_template_dfas(&commit_dfa)
                    .map(|split| (terminal, Arc::new(split)))
        };
        let split_templates = if macro_parallelism_disabled() {
            let mut timings = Vec::with_capacity(templates.by_terminal.len());
            let result = templates
                .by_terminal
                .iter()
                .filter_map(|item| {
                    let started = Instant::now();
                    let result = split(item);
                    timings.push(started.elapsed().as_secs_f64() * 1000.0);
                    result
                })
                .collect::<Vec<_>>();
            report_macro_item_timings("compose_template_commit_splits", &timings);
            result
        } else {
            templates
                .by_terminal
                .par_iter()
                .filter_map(split)
                .collect::<Vec<_>>()
        };
        for (terminal, split) in split_templates {
            if let Some(slot) = template_dfas_by_terminal.get_mut(terminal as usize) {
                *slot = Some(split);
            }
        }
    }
    let commit_ms = commit_started_at.elapsed().as_secs_f64() * 1000.0;
    let total_ms = started_at.elapsed().as_secs_f64() * 1000.0;
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_composition_templates] selected={} characterized={} characterize_ms={characterize_ms:.3} from_characterizations_ms={from_characterizations_ms:.3} commit_ms={commit_ms:.3} total_ms={total_ms:.3}",
            selected.iter().filter(|&&value| value).count(),
            characterizations.len(),
        );
    }
    (templates, template_dfas_by_terminal, total_ms)
}

/// Reuse cached component parser-template characterizations and apply only the
/// stack effects introduced by the current subgrammar splice.
///
/// This fast path is deliberately conservative.  It is enabled only when the
/// parent LR-state coordinate is unchanged, every selected parent terminal has
/// a cached standalone characterization, and composition does not rewrite an
/// existing parent action/forwarded-shift cell.  New actions in appended child
/// states and new boundary-nonterminal goto predecessors are additive and can
/// therefore be characterized independently, unioned into the cached parent
/// characterization, and recompiled only for genuinely changed terminals.
pub(super) struct CachedCompositionTemplatePlan {
    pub(super) reused_dfas: BTreeMap<u32, UnweightedDfa>,
    pub(super) characterization_deltas: BTreeMap<
        u32,
        crate::compiler::stages::templates::characterize::TerminalCharacterization,
    >,
    pub(super) fresh_characterizations: BTreeMap<
        u32,
        crate::compiler::stages::templates::characterize::TerminalCharacterization,
    >,
    pub(super) num_terminals: usize,
}

pub(super) struct EagerChangedParentTemplates {
    pub(super) templates: Templates,
    pub(super) changed_parent: Vec<bool>,
    pub(super) delta_templates: BTreeMap<u32, (UnweightedDfa, UnweightedDfa)>,
    pub(super) unsafe_parent: BTreeSet<u32>,
    pub(super) build_ms: f64,
}

pub(super) struct EagerConcreteDeltaPrecompute {
    pub(super) delta_templates: BTreeMap<u32, (UnweightedDfa, UnweightedDfa)>,
    pub(super) unsafe_terminals: BTreeSet<u32>,
}

impl CachedCompositionTemplatePlan {
    pub(super) fn materialize(self, parent: &Constraint) -> Templates {
        let total_started_at = Instant::now();
        let reused_started_at = Instant::now();
        let mut templates = Templates::from_terminal_dfas(self.reused_dfas);
        let reused_ms = reused_started_at.elapsed().as_secs_f64() * 1000.0;
        let patch_assembly_started_at = Instant::now();
        let mut patched_characterizations = BTreeMap::new();
        for (terminal, delta) in self.characterization_deltas {
            let mut patched = parent
                .composition_parser_characterizations_by_terminal
                .get(terminal as usize)
                .and_then(Option::as_ref)
                .expect("cached composition plan requires its parent characterization")
                .clone();
            patched.escapes.extend(delta.escapes);
            patched.reduces.extend(delta.reduces);
            patched.nt_escapes.extend(delta.nt_escapes);
            patched.nt_rereduces.extend(delta.nt_rereduces);
            patched.all_nts.extend(delta.all_nts);
            patched.escapes.sort();
            patched.escapes.dedup();
            patched.reduces.sort();
            patched.reduces.dedup();
            patched.nt_escapes.sort();
            patched.nt_escapes.dedup();
            patched.nt_rereduces.sort();
            patched.nt_rereduces.dedup();
            patched_characterizations.insert(terminal, patched);
        }
        let patch_assembly_ms = patch_assembly_started_at.elapsed().as_secs_f64() * 1000.0;
        let patch_compile_started_at = Instant::now();
        let patched = Templates::from_characterizations(&patched_characterizations);
        let patch_compile_ms = patch_compile_started_at.elapsed().as_secs_f64() * 1000.0;
        templates.by_terminal.extend(patched.by_terminal);
        templates.by_terminal_nwa.extend(patched.by_terminal_nwa);
        let fresh_started_at = Instant::now();
        let fresh = Templates::from_characterizations(&self.fresh_characterizations);
        let fresh_ms = fresh_started_at.elapsed().as_secs_f64() * 1000.0;
        templates.by_terminal.extend(fresh.by_terminal);
        templates.by_terminal_nwa.extend(fresh.by_terminal_nwa);
        if compose_profile_enabled() {
            eprintln!(
                "[glrmask/profile][constraint_cached_template_materialize] reused_ms={reused_ms:.3} patch_assembly_ms={patch_assembly_ms:.3} patch_compile_ms={patch_compile_ms:.3} fresh_ms={fresh_ms:.3} total_ms={:.3}",
                total_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        templates
    }

    pub(super) fn raw_template_cache(templates: &Templates, num_terminals: usize) -> Vec<Option<UnweightedDfa>> {
        let mut raw = vec![None; num_terminals];
        for (&terminal, dfa) in &templates.by_terminal {
            if let Some(slot) = raw.get_mut(terminal as usize) {
                *slot = Some(dfa.clone());
            }
        }
        raw
    }
}

pub(super) fn changed_parent_template_candidate_terminals(
    composed_table: &ComposedTable,
    parent: &Constraint,
    num_terminals: u32,
) -> BitSet {
    let parent_terminal_end = composed_table
        .terminal_offsets
        .get(1)
        .copied()
        .unwrap_or(num_terminals)
        .min(num_terminals) as usize;
    let mut candidates = BitSet::new(num_terminals as usize);
    if std::env::var_os("GLRMASK_EXPERIMENT_LINKER_PARENT_TEMPLATE_CANDIDATES").is_some() {
        for &terminal in &composed_table.appended_parent_action_terminals {
            if (terminal as usize) < parent_terminal_end {
                candidates.set(terminal as usize);
            }
        }
    } else {
        for row in composed_table
            .table
            .action
            .iter()
            .skip(parent.table.num_states as usize)
        {
            for (terminal, _) in row.iter() {
                if (terminal as usize) < parent_terminal_end {
                    candidates.set(terminal as usize);
                }
            }
        }
    }
    for row in &composed_table.table.goto {
        for &boundary_nonterminal in &composed_table.boundary_nonterminals {
            let Some(&(top_state, _)) = row.get(&boundary_nonterminal) else {
                continue;
            };
            if let Some(action_row) = composed_table.table.action.get(top_state as usize) {
                for (terminal, _) in action_row.iter() {
                    if (terminal as usize) < parent_terminal_end {
                        candidates.set(terminal as usize);
                    }
                }
            }
        }
    }
    candidates
}

pub(super) fn try_build_changed_parent_templates_for_terminal_count(
    composed_table: &ComposedTable,
    components: &[&Constraint],
    num_terminals: u32,
    scoped_ignore_terminals: &BitSet,
    need_concrete_delta: bool,
) -> Option<EagerChangedParentTemplates> {
    let started_at = Instant::now();
    let parent = *components.first()?;
    let parent_terminal_end = composed_table
        .terminal_offsets
        .get(1)
        .copied()
        .unwrap_or(num_terminals)
        .min(num_terminals) as usize;
    // A cached parent template can change only if the splice introduces a new
    // action for that terminal in an appended state, or if a new boundary
    // nonterminal goto exposes a predecessor whose top state admits it.
    let candidate_started_at = Instant::now();
    let candidates = changed_parent_template_candidate_terminals(
        composed_table,
        parent,
        num_terminals,
    );
    if std::env::var_os("GLRMASK_VALIDATE_LINKER_PARENT_TEMPLATE_CANDIDATES").is_some()
        && std::env::var_os("GLRMASK_EXPERIMENT_LINKER_PARENT_TEMPLATE_CANDIDATES").is_some()
    {
        let mut reference = BTreeSet::new();
        for row in composed_table
            .table
            .action
            .iter()
            .skip(parent.table.num_states as usize)
        {
            for (terminal, _) in row.iter() {
                if (terminal as usize) < parent_terminal_end {
                    reference.insert(terminal);
                }
            }
        }
        assert!(
            composed_table
                .appended_parent_action_terminals
                .iter()
                .all(|terminal| reference.contains(terminal)),
            "linker-published appended parent action terminals differ from completed-table scan",
        );
    }
    let candidate_ms = candidate_started_at.elapsed().as_secs_f64() * 1000.0;
    let mut selected = vec![false; num_terminals as usize];
    for terminal in candidates.iter() {
        if terminal >= parent_terminal_end {
            continue;
        }
        let terminal_u32 = terminal as u32;
        let cached = parent
            .composition_parser_characterizations_by_terminal
            .get(terminal)
            .is_some_and(Option::is_some);
        if !cached
            || scoped_ignore_terminals.contains(terminal)
            || composed_table.control_terminals.contains(&terminal_u32)
        {
            continue;
        }
        let existing_rows_unchanged = (0..parent.table.num_states).all(|state| {
            composed_table.table.action(state, terminal_u32)
                == parent.table.action(state, terminal_u32)
                && composed_table
                    .table
                    .forwarded_shifts
                    .contains(&(state, terminal_u32))
                    == parent
                        .table
                        .forwarded_shifts
                        .contains(&(state, terminal_u32))
        });
        selected[terminal] = existing_rows_unchanged;
    }
    let plan = try_prepare_cached_composition_template_plan_for_terminal_count(
        composed_table,
        components,
        num_terminals,
        None,
        &selected,
        scoped_ignore_terminals,
    )?;
    if !plan.fresh_characterizations.is_empty() {
        return None;
    }

    let mut changed_parent = vec![false; num_terminals as usize];
    let mut patched_characterizations = BTreeMap::new();
    for (terminal, delta) in plan.characterization_deltas {
        let mut patched = parent
            .composition_parser_characterizations_by_terminal
            .get(terminal as usize)
            .and_then(Option::as_ref)?
            .clone();
        patched.escapes.extend(delta.escapes);
        patched.reduces.extend(delta.reduces);
        patched.nt_escapes.extend(delta.nt_escapes);
        patched.nt_rereduces.extend(delta.nt_rereduces);
        patched.all_nts.extend(delta.all_nts);
        patched.escapes.sort();
        patched.escapes.dedup();
        patched.reduces.sort();
        patched.reduces.dedup();
        patched.nt_escapes.sort();
        patched.nt_escapes.dedup();
        patched.nt_rereduces.sort();
        patched.nt_rereduces.dedup();
        changed_parent[terminal as usize] = true;
        patched_characterizations.insert(terminal, patched);
    }
    let template_compile_started_at = Instant::now();
    let templates = Templates::from_characterizations(&patched_characterizations);
    let template_compile_ms = template_compile_started_at.elapsed().as_secs_f64() * 1000.0;
    // The exact New\Old template delta is independent of lexical boundary
    // discovery. Compute it in this eager worker while discovery runs, rather
    // than serializing another full DFA-difference pass after the 140 boundary
    // tokens are known.
    let delta_started_at = Instant::now();
    let delta_results = if need_concrete_delta {
        let build_delta = |(terminal, &changed): (usize, &bool)| {
                if !changed {
                    return None;
                }
                let terminal = terminal as u32;
                let old = parent
                    .composition_parser_templates_by_terminal
                    .get(terminal as usize)
                    .and_then(Option::as_ref);
                let new = templates.by_terminal.get(&terminal);
                let (Some(old), Some(new)) = (old, new) else {
                    return Some((terminal, None, true));
                };
                let removed = unweighted_dfa_difference(old, new);
                if !unweighted_dfa_language_is_empty(&removed) {
                    return Some((terminal, None, true));
                }
                let delta = trim_unweighted_dfa_productive(unweighted_dfa_difference(new, old));
                let delta = (!unweighted_dfa_language_is_empty(&delta))
                    .then(|| (old.clone(), delta));
                Some((terminal, delta, false))
        };
        if macro_parallelism_disabled() {
            let mut timings = Vec::new();
            let result = changed_parent
                .iter()
                .enumerate()
                .filter_map(|item| {
                    let started = Instant::now();
                    let result = build_delta(item);
                    timings.push(started.elapsed().as_secs_f64() * 1000.0);
                    result
                })
                .collect::<Vec<_>>();
            report_macro_item_timings("compose_changed_parent_template_deltas", &timings);
            result
        } else {
            changed_parent
                .par_iter()
                .enumerate()
                .filter_map(build_delta)
                .collect::<Vec<_>>()
        }
    } else {
        Vec::new()
    };
    let delta_ms = delta_started_at.elapsed().as_secs_f64() * 1000.0;
    let mut delta_templates = BTreeMap::new();
    let mut unsafe_parent = BTreeSet::new();
    for (terminal, delta, unsafe_terminal) in delta_results {
        if unsafe_terminal {
            unsafe_parent.insert(terminal);
        } else if let Some(delta) = delta {
            delta_templates.insert(terminal, delta);
        }
    }
    let build_ms = started_at.elapsed().as_secs_f64() * 1000.0;
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_eager_changed_parent_templates] candidates={} selected_parent={} changed_parent={} changed_ids={:?} states={} candidate_ms={candidate_ms:.3} template_compile_ms={template_compile_ms:.3} delta_ms={delta_ms:.3} build_ms={build_ms:.3}",
            candidates.count_ones(),
            selected.iter().filter(|&&active| active).count(),
            changed_parent.iter().filter(|&&changed| changed).count(),
            changed_parent.iter().enumerate().filter_map(|(terminal, &changed)| changed.then_some(terminal)).collect::<Vec<_>>(),
            templates
                .by_terminal
                .values()
                .map(|dfa| dfa.num_states())
                .sum::<usize>(),
        );
    }
    Some(EagerChangedParentTemplates {
        templates,
        changed_parent,
        delta_templates,
        unsafe_parent,
        build_ms,
    })
}

pub(super) fn finish_eager_changed_parent_templates(
    mut eager: EagerChangedParentTemplates,
    composed_table: &ComposedTable,
    components: &[&Constraint],
    active_terminals: &[bool],
    pretransported: Option<Templates>,
) -> Option<(
    Templates,
    Vec<Option<Arc<crate::runtime::CommitTemplateDfas>>>,
    f64,
    EagerConcreteDeltaPrecompute,
)> {
    let finish_started_at = Instant::now();
    if eager.changed_parent.len() != active_terminals.len() {
        return None;
    }
    eager
        .templates
        .by_terminal
        .retain(|terminal, _| active_terminals[*terminal as usize]);
    eager
        .templates
        .by_terminal_nwa
        .retain(|terminal, _| active_terminals[*terminal as usize]);

    let mut transport_selected = active_terminals.to_vec();
    for (terminal, &changed) in eager.changed_parent.iter().enumerate() {
        if changed {
            transport_selected[terminal] = false;
        }
    }
    let expected_transport = transport_selected.iter().filter(|&&selected| selected).count();
    let transported = if let Some(mut transported) = pretransported {
        transported
            .by_terminal
            .retain(|terminal, _| transport_selected.get(*terminal as usize).copied().unwrap_or(false));
        transported
            .by_terminal_nwa
            .retain(|terminal, _| transport_selected.get(*terminal as usize).copied().unwrap_or(false));
        transported
    } else {
        try_rebuild_cached_transported_component_templates(
            composed_table,
            components,
            &transport_selected,
        )?
    };
    if transported.by_terminal.len() != expected_transport {
        return None;
    }
    if std::env::var_os("GLRMASK_VALIDATE_PAIRED_TEMPLATE_TRANSPORT").is_some() {
        let reference = Templates::from_terminal_dfas(transported.by_terminal.clone());
        assert_eq!(
            transported.by_terminal_nwa.len(),
            reference.by_terminal_nwa.len(),
            "paired cached-template transport changed NWA template count",
        );
        for (&terminal, nwa) in &transported.by_terminal_nwa {
            let reference_nwa = reference
                .by_terminal_nwa
                .get(&terminal)
                .expect("reference NWA skeleton missing transported terminal");
            assert_eq!(
                nwa.start_states(),
                reference_nwa.start_states(),
                "paired cached-template transport changed NWA starts for terminal {terminal}",
            );
            assert_eq!(
                nwa.states(),
                reference_nwa.states(),
                "paired cached-template transport changed NWA states for terminal {terminal}",
            );
        }
    }
    eager.templates.by_terminal.extend(transported.by_terminal);
    eager
        .templates
        .by_terminal_nwa
        .extend(transported.by_terminal_nwa);
    if eager.templates.by_terminal.len()
        != active_terminals.iter().filter(|&&active| active).count()
    {
        return None;
    }

    let template_dfas_by_terminal = if defer_boundary_commit_templates() {
        vec![None; active_terminals.len()]
    } else {
        let raw_templates = CachedCompositionTemplatePlan::raw_template_cache(
            &eager.templates,
            active_terminals.len(),
        );
        build_commit_templates_from_raw_templates(&raw_templates)
    };
    let finish_ms = finish_started_at.elapsed().as_secs_f64() * 1000.0;
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_eager_changed_parent_finish] active={} changed_active={} transported={} finish_ms={finish_ms:.3} eager_cpu_ms={:.3}",
            active_terminals.iter().filter(|&&active| active).count(),
            eager
                .changed_parent
                .iter()
                .zip(active_terminals)
                .filter(|(changed, active)| **changed && **active)
                .count(),
            expected_transport,
            eager.build_ms,
        );
    }
    Some((
        eager.templates,
        template_dfas_by_terminal,
        eager.build_ms + finish_ms,
        EagerConcreteDeltaPrecompute {
            delta_templates: eager.delta_templates,
            unsafe_terminals: eager.unsafe_parent,
        },
    ))
}

pub(super) fn try_prepare_cached_composition_template_plan(
    composed_table: &ComposedTable,
    components: &[&Constraint],
    analyzed: &AnalyzedGrammar,
    selected: &[bool],
    scoped_ignore_terminals: &BitSet,
) -> Option<CachedCompositionTemplatePlan> {
    try_prepare_cached_composition_template_plan_for_terminal_count(
        composed_table,
        components,
        analyzed.num_terminals,
        Some(analyzed),
        selected,
        scoped_ignore_terminals,
    )
}

pub(super) fn try_prepare_cached_composition_template_plan_for_terminal_count(
    composed_table: &ComposedTable,
    components: &[&Constraint],
    num_terminals: u32,
    analyzed_for_fresh: Option<&AnalyzedGrammar>,
    selected: &[bool],
    scoped_ignore_terminals: &BitSet,
) -> Option<CachedCompositionTemplatePlan> {
    let total_started_at = Instant::now();
    if std::env::var_os("GLRMASK_DISABLE_CACHED_BOUNDARY_TEMPLATES").is_some() {
        return None;
    }
    let parent = *components.first()?;
    let parent_relation = composed_table.state_relations.first()?;
    if parent_relation
        .iter()
        .enumerate()
        .any(|(local, targets)| targets.as_slice() != [local as u32])
    {
        return None;
    }
    if selected.len() != num_terminals as usize {
        return None;
    }
    let parent_terminal_end = composed_table
        .terminal_offsets
        .get(1)
        .copied()
        .unwrap_or(num_terminals)
        .min(num_terminals);
    if parent_terminal_end > parent.table.num_terminals {
        return None;
    }

    let action_seed_started_at = Instant::now();
    let mut action_seeds = vec![Vec::<u32>::new(); num_terminals as usize];
    for terminal in 0..parent_terminal_end {
        if !selected[terminal as usize] {
            continue;
        }
        parent
            .composition_parser_characterizations_by_terminal
            .get(terminal as usize)
            .and_then(Option::as_ref)?;
        for state in 0..parent.table.num_states {
            if composed_table.table.action(state, terminal) != parent.table.action(state, terminal)
                || composed_table.table.forwarded_shifts.contains(&(state, terminal))
                    != parent.table.forwarded_shifts.contains(&(state, terminal))
            {
                return None;
            }
        }
        for state in parent.table.num_states..composed_table.table.num_states {
            let action = composed_table.table.action(state, terminal);
            let forwarded = composed_table.table.forwarded_shifts.contains(&(state, terminal));
            if forwarded && action.is_none() {
                return None;
            }
            if action.is_some() {
                action_seeds[terminal as usize].push(state);
            }
        }
    }
    let action_seed_ms = action_seed_started_at.elapsed().as_secs_f64() * 1000.0;
    let action_characterize_started_at = Instant::now();
    let action_deltas = characterize_terminal_action_state_seeds_for_terminal_count(
        &composed_table.table,
        num_terminals,
        &action_seeds,
    );
    let action_characterize_ms =
        action_characterize_started_at.elapsed().as_secs_f64() * 1000.0;

    let nt_seed_started_at = Instant::now();
    let mut nt_predecessor_seeds =
        vec![Vec::<(u32, u32, u32, bool)>::new(); num_terminals as usize];
    for (revealed_state, row) in composed_table.table.goto.iter().enumerate() {
        for &boundary_nonterminal in &composed_table.boundary_nonterminals {
            let Some(&(top_state, goto_replace)) = row.get(&boundary_nonterminal) else {
                continue;
            };
            for terminal in 0..parent_terminal_end {
                if selected[terminal as usize]
                    && composed_table.table.action(top_state, terminal).is_some()
                {
                    nt_predecessor_seeds[terminal as usize].push((
                        top_state,
                        revealed_state as u32,
                        boundary_nonterminal,
                        goto_replace,
                    ));
                }
            }
        }
    }
    for seeds in &mut nt_predecessor_seeds {
        seeds.sort_unstable();
        seeds.dedup();
    }
    let nt_seed_ms = nt_seed_started_at.elapsed().as_secs_f64() * 1000.0;
    let nt_characterize_started_at = Instant::now();
    let nt_deltas = characterize_terminal_nt_predecessor_seeds_for_terminal_count(
        &composed_table.table,
        num_terminals,
        &nt_predecessor_seeds,
    );
    let nt_characterize_ms =
        nt_characterize_started_at.elapsed().as_secs_f64() * 1000.0;

    let patch_started_at = Instant::now();
    let mut characterization_deltas = BTreeMap::new();
    let mut changed_parent = vec![false; selected.len()];
    for terminal in 0..parent_terminal_end {
        if !selected[terminal as usize] {
            continue;
        }
        let action_delta = action_deltas.get(&terminal);
        let nt_delta = nt_deltas.get(&terminal);
        let has_delta = [action_delta, nt_delta].into_iter().flatten().any(|delta| {
            !delta.escapes.is_empty()
                || !delta.reduces.is_empty()
                || !delta.nt_escapes.is_empty()
                || !delta.nt_rereduces.is_empty()
        });
        if !has_delta {
            continue;
        }
        let mut delta = crate::compiler::stages::templates::characterize::TerminalCharacterization {
            escapes: Vec::new(),
            reduces: Vec::new(),
            nt_escapes: Vec::new(),
            nt_rereduces: Vec::new(),
            all_nts: BTreeSet::new(),
        };
        for part in [action_delta, nt_delta].into_iter().flatten() {
            // These seeded characterizations are additive by construction.
            // Keep only the tiny delta here; cloning/merging the cached parent
            // characterization belongs to parser-template materialization.
            delta.escapes.extend(part.escapes.iter().cloned());
            delta.reduces.extend(part.reduces.iter().cloned());
            delta.nt_escapes.extend(part.nt_escapes.iter().cloned());
            delta
                .nt_rereduces
                .extend(part.nt_rereduces.iter().cloned());
            delta.all_nts.extend(part.all_nts.iter().copied());
        }
        delta.escapes.sort();
        delta.escapes.dedup();
        delta.reduces.sort();
        delta.reduces.dedup();
        delta.nt_escapes.sort();
        delta.nt_escapes.dedup();
        delta.nt_rereduces.sort();
        delta.nt_rereduces.dedup();
        changed_parent[terminal as usize] = true;
        characterization_deltas.insert(terminal, delta);
    }
    let patch_ms = patch_started_at.elapsed().as_secs_f64() * 1000.0;

    let mut reuse_selected = selected.to_vec();
    for (terminal, changed) in changed_parent.iter().copied().enumerate() {
        if changed {
            reuse_selected[terminal] = false;
        }
    }
    let mut fresh_selected = vec![false; selected.len()];
    for terminal in scoped_ignore_terminals.iter() {
        if terminal < selected.len() && selected[terminal] {
            reuse_selected[terminal] = false;
            fresh_selected[terminal] = true;
        }
    }
    for &terminal in &composed_table.control_terminals {
        if let Some(&true) = selected.get(terminal as usize) {
            reuse_selected[terminal as usize] = false;
            fresh_selected[terminal as usize] = true;
        }
    }

    let transport_started_at = Instant::now();
    let reused_dfas = rebuild_transported_component_templates(
        composed_table,
        components,
        &reuse_selected,
    );
    let transport_ms = transport_started_at.elapsed().as_secs_f64() * 1000.0;
    for (terminal, &active) in selected.iter().enumerate() {
        if active
            && !changed_parent[terminal]
            && !reused_dfas.contains_key(&(terminal as u32))
        {
            fresh_selected[terminal] = true;
        }
    }
    let fresh_started_at = Instant::now();
    let fresh_characterizations = if fresh_selected.iter().any(|&fresh| fresh) {
        characterize_selected_terminals(
            &composed_table.table,
            analyzed_for_fresh?,
            &fresh_selected,
        )
    } else {
        BTreeMap::new()
    };
    let fresh_ms = fresh_started_at.elapsed().as_secs_f64() * 1000.0;
    if selected.iter().enumerate().any(|(terminal, &active)| {
        active
            && !reused_dfas.contains_key(&(terminal as u32))
            && !characterization_deltas.contains_key(&(terminal as u32))
            && !fresh_characterizations.contains_key(&(terminal as u32))
    }) {
        return None;
    }

    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_cached_template_plan] selected={} action_seed_ms={action_seed_ms:.3} action_characterize_ms={action_characterize_ms:.3} nt_seed_ms={nt_seed_ms:.3} nt_characterize_ms={nt_characterize_ms:.3} patch_ms={patch_ms:.3} transport_ms={transport_ms:.3} fresh_ms={fresh_ms:.3} reused={} patched={} fresh={} total_ms={:.3}",
            selected.iter().filter(|&&value| value).count(),
            reused_dfas.len(),
            characterization_deltas.len(),
            fresh_characterizations.len(),
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Some(CachedCompositionTemplatePlan {
        reused_dfas,
        characterization_deltas,
        fresh_characterizations,
        num_terminals: num_terminals as usize,
    })
}

pub(super) fn try_build_cached_composition_templates_for_terminal_count(
    composed_table: &ComposedTable,
    components: &[&Constraint],
    num_terminals: u32,
    selected: &[bool],
    scoped_ignore_terminals: &BitSet,
) -> Option<(
    Templates,
    Vec<Option<Arc<crate::runtime::CommitTemplateDfas>>>,
    f64,
)> {
    let started_at = Instant::now();
    let plan = try_prepare_cached_composition_template_plan_for_terminal_count(
        composed_table,
        components,
        num_terminals,
        None,
        selected,
        scoped_ignore_terminals,
    )?;
    let changed_parent = plan.characterization_deltas.len();
    let fresh = plan.fresh_characterizations.len();
    let reused = plan.reused_dfas.len();
    debug_assert_eq!(plan.num_terminals, num_terminals as usize);

    let templates = plan.materialize(components[0]);
    let template_dfas_by_terminal = if defer_boundary_commit_templates() {
        vec![None; num_terminals as usize]
    } else {
        let raw_templates = CachedCompositionTemplatePlan::raw_template_cache(
            &templates,
            num_terminals as usize,
        );
        build_commit_templates_from_raw_templates(&raw_templates)
    };
    let total_ms = started_at.elapsed().as_secs_f64() * 1000.0;
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_cached_composition_templates_count_only] selected={} reused={} changed_parent={} fresh={} raw_templates={} commit_templates={} total_ms={total_ms:.3}",
            selected.iter().filter(|&&value| value).count(),
            reused,
            changed_parent,
            fresh,
            templates.by_terminal.len(),
            template_dfas_by_terminal.iter().flatten().count(),
        );
    }
    Some((templates, template_dfas_by_terminal, total_ms))
}

pub(super) fn try_build_cached_composition_templates(
    composed_table: &ComposedTable,
    components: &[&Constraint],
    analyzed: &AnalyzedGrammar,
    selected: &[bool],
    scoped_ignore_terminals: &BitSet,
) -> Option<(
    Templates,
    Vec<Option<Arc<crate::runtime::CommitTemplateDfas>>>,
    f64,
)> {
    let started_at = Instant::now();
    let plan = try_prepare_cached_composition_template_plan(
        composed_table,
        components,
        analyzed,
        selected,
        scoped_ignore_terminals,
    )?;
    let changed_parent = plan.characterization_deltas.len();
    let fresh = plan.fresh_characterizations.len();
    let reused = plan.reused_dfas.len();
    debug_assert_eq!(plan.num_terminals, analyzed.num_terminals as usize);

    let templates = plan.materialize(components[0]);
    let template_dfas_by_terminal = if defer_boundary_commit_templates() {
        vec![None; analyzed.num_terminals as usize]
    } else {
        let raw_templates = CachedCompositionTemplatePlan::raw_template_cache(
            &templates,
            analyzed.num_terminals as usize,
        );
        build_commit_templates_from_raw_templates(&raw_templates)
    };

    if std::env::var_os("GLRMASK_VALIDATE_CACHED_BOUNDARY_TEMPLATES").is_some() {
        let (reference, _, _) =
            build_composition_templates(&composed_table.table, analyzed, selected);
        for (terminal, &active) in selected.iter().enumerate() {
            if !active {
                continue;
            }
            let terminal = terminal as u32;
            let candidate = templates
                .by_terminal
                .get(&terminal)
                .expect("cached-template fast path must cover selected terminal");
            let expected = reference
                .by_terminal
                .get(&terminal)
                .expect("reference template build must cover selected terminal");
            let candidate_only = unweighted_dfa_difference(candidate, expected);
            let expected_only = unweighted_dfa_difference(expected, candidate);
            assert!(
                unweighted_dfa_language_is_empty(&candidate_only)
                    && unweighted_dfa_language_is_empty(&expected_only),
                "cached composition template differs from full characterization for terminal {terminal}: candidate_only={:?} expected_only={:?}",
                unweighted_dfa_shortest_word(&candidate_only),
                unweighted_dfa_shortest_word(&expected_only),
            );
        }
    }
    let total_ms = started_at.elapsed().as_secs_f64() * 1000.0;
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_cached_composition_templates] selected={} reused={} changed_parent={} fresh={} raw_templates={} commit_templates={} total_ms={total_ms:.3}",
            selected.iter().filter(|&&value| value).count(),
            reused,
            changed_parent,
            fresh,
            templates.by_terminal.len(),
            template_dfas_by_terminal.iter().flatten().count(),
        );
    }
    Some((templates, template_dfas_by_terminal, total_ms))
}
