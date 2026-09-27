//! Determinize expression-labeled graphs without expanding every expression to bytes.

use crate::automata::lexer::ast::Expr;
use crate::automata::lexer::dfa::DFA;
use crate::automata::lexer::nfa::NFA;

use crate::ds::bitset::BitSet;
use rustc_hash::FxHashMap;
use std::hash::Hash;
use std::sync::Arc;
use std::time::Instant;
use super::component::compile_terminal_expr_dfa;
use super::nfa::append_compiled_expr;
use rayon::prelude::*;

fn direct_expression_graph_byte_classes(local_dfas: &[DFA]) -> (Vec<u8>, Vec<Vec<u8>>) {
    let mut classes = vec![0u8; 256];
    let mut num_classes = 1usize;
    for dfa in local_dfas {
        for state in 0..dfa.num_states() as u32 {
            if num_classes == 256 {
                break;
            }
            let mut keys = FxHashMap::<(u8, u32), u8>::default();
            let mut refined = vec![0u8; 256];
            let mut next_class = 0u16;
            for byte in 0u8..=255 {
                let key = (classes[byte as usize], dfa.step(state, byte).unwrap_or(u32::MAX));
                let class = *keys.entry(key).or_insert_with(|| {
                    let class = next_class as u8;
                    next_class += 1;
                    class
                });
                refined[byte as usize] = class;
            }
            if refined != classes {
                classes = refined;
                num_classes = next_class as usize;
            }
        }
    }
    let mut members = vec![Vec::<u8>::new(); num_classes];
    for byte in 0u8..=255 {
        members[classes[byte as usize] as usize].push(byte);
    }
    (classes, members)
}

fn direct_expression_graph_class_transitions(
    local_dfas: &[DFA],
    class_members: &[Vec<u8>],
) -> Vec<Vec<Box<[(u8, u32)]>>> {
    local_dfas
        .iter()
        .map(|dfa| {
            (0..dfa.num_states() as u32)
                .map(|state| {
                    let mut transitions = Vec::new();
                    for (class, members) in class_members.iter().enumerate() {
                        let target = dfa.step(state, members[0]);
                        debug_assert!(members
                            .iter()
                            .all(|&byte| dfa.step(state, byte) == target));
                        if let Some(target) = target {
                            transitions.push((class as u8, target));
                        }
                    }
                    transitions.into_boxed_slice()
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

pub(super) fn expand_direct_expression_graph_classes(dfa: &mut DFA, class_members: &[Vec<u8>]) {
    let expanded = dfa
        .states()
        .iter()
        .map(|state| {
            let capacity = state
                .transitions
                .iter()
                .map(|(class, _)| class_members[class as usize].len())
                .sum();
            let mut entries = Vec::with_capacity(capacity);
            for (class, &target) in state.transitions.iter() {
                entries.extend(
                    class_members[class as usize]
                        .iter()
                        .map(|&byte| (byte, target)),
                );
            }
            entries.sort_unstable_by_key(|entry| entry.0);
            crate::ds::char_transitions::CharTransitions::from_sorted_entries(entries)
        })
        .collect::<Vec<_>>();
    for (state, transitions) in dfa.states_mut().iter_mut().zip(expanded) {
        state.transitions = transitions;
    }
}

struct DirectExpressionGraphEdge {
    symbol: usize,
    targets: Box<[u32]>,
    config_base: u32,
}

struct DirectReadyExpansion {
    configs: Box<[u32]>,
    accepting: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct DirectGroupedConfig {
    symbol: u32,
    local_state: u32,
    edge_set: u32,
}

#[derive(Clone, Debug)]
struct DirectGroupedState {
    accepting: bool,
    groups: Box<[DirectGroupedConfig]>,
    hash: u64,
}

impl DirectGroupedState {
    fn new(accepting: bool, groups: Box<[DirectGroupedConfig]>) -> Self {
        let mut hasher = rustc_hash::FxHasher::default();
        std::hash::Hash::hash(&accepting, &mut hasher);
        std::hash::Hash::hash(&groups, &mut hasher);
        let hash = std::hash::Hasher::finish(&hasher);
        Self {
            accepting,
            groups,
            hash,
        }
    }
}

impl PartialEq for DirectGroupedState {
    fn eq(&self, other: &Self) -> bool {
        self.hash == other.hash
            && self.accepting == other.accepting
            && self.groups == other.groups
    }
}

impl Eq for DirectGroupedState {}

impl std::hash::Hash for DirectGroupedState {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_u64(self.hash);
    }
}

#[derive(Clone)]
struct DirectGroupedContinuation {
    accepting: bool,
    groups: Box<[DirectGroupedConfig]>,
}

struct DirectEdgeSetInterner {
    sets: Vec<Arc<[u32]>>,
    map: FxHashMap<Arc<[u32]>, u32>,
    union_cache: FxHashMap<Arc<[u32]>, u32>,
    merge_buffer: Vec<u32>,
    union_cache_hits: usize,
    union_cache_misses: usize,
}

impl DirectEdgeSetInterner {
    fn new() -> Self {
        Self {
            sets: Vec::new(),
            map: FxHashMap::default(),
            union_cache: FxHashMap::default(),
            merge_buffer: Vec::new(),
            union_cache_hits: 0,
            union_cache_misses: 0,
        }
    }

    fn intern_sorted(&mut self, edges: &[u32]) -> u32 {
        debug_assert!(!edges.is_empty());
        debug_assert!(edges.windows(2).all(|pair| pair[0] < pair[1]));
        if let Some(&existing) = self.map.get(edges) {
            return existing;
        }
        let id = self.sets.len() as u32;
        let edges: Arc<[u32]> = Arc::from(edges);
        self.map.insert(Arc::clone(&edges), id);
        self.sets.push(edges);
        id
    }

    fn get(&self, id: u32) -> &[u32] {
        &self.sets[id as usize]
    }

    fn union(&mut self, ids: &[u32]) -> u32 {
        debug_assert!(!ids.is_empty());
        debug_assert!(ids.windows(2).all(|pair| pair[0] < pair[1]));
        if ids.len() == 1 {
            return ids[0];
        }
        if let Some(&existing) = self.union_cache.get(ids) {
            self.union_cache_hits += 1;
            return existing;
        }
        self.union_cache_misses += 1;
        self.merge_buffer.clear();
        for &id in ids {
            self.merge_buffer
                .extend_from_slice(&self.sets[id as usize]);
        }
        self.merge_buffer.sort_unstable();
        self.merge_buffer.dedup();
        let result = if let Some(&existing) = self.map.get(self.merge_buffer.as_slice()) {
            existing
        } else {
            let id = self.sets.len() as u32;
            let edges: Arc<[u32]> = Arc::from(self.merge_buffer.as_slice());
            self.map.insert(Arc::clone(&edges), id);
            self.sets.push(edges);
            id
        };
        self.union_cache.insert(Arc::from(ids), result);
        result
    }
}

fn group_direct_configs(
    configs: &[u32],
    config_edge: &[u32],
    config_local_state: &[u32],
    edges: &[DirectExpressionGraphEdge],
    edge_sets: &mut DirectEdgeSetInterner,
) -> Box<[DirectGroupedConfig]> {
    let mut entries = configs
        .iter()
        .map(|&config| {
            let edge = config_edge[config as usize];
            let edge_data = &edges[edge as usize];
            (
                edge_data.symbol as u32,
                config_local_state[config as usize],
                edge,
            )
        })
        .collect::<Vec<_>>();
    entries.sort_unstable();
    entries.dedup();
    let mut groups = Vec::new();
    let mut position = 0usize;
    while position < entries.len() {
        let (symbol, local_state, _) = entries[position];
        let start = position;
        position += 1;
        while position < entries.len()
            && entries[position].0 == symbol
            && entries[position].1 == local_state
        {
            position += 1;
        }
        let edge_ids = entries[start..position]
            .iter()
            .map(|entry| entry.2)
            .collect::<Vec<_>>();
        let edge_set = edge_sets.intern_sorted(&edge_ids);
        groups.push(DirectGroupedConfig {
            symbol,
            local_state,
            edge_set,
        });
    }
    groups.into_boxed_slice()
}

fn canonicalize_direct_group_fragments(
    fragments: &mut Vec<DirectGroupedConfig>,
    edge_sets: &mut DirectEdgeSetInterner,
    edge_set_ids: &mut Vec<u32>,
    output: &mut Vec<DirectGroupedConfig>,
) {
    fragments.sort_unstable();
    output.clear();
    let mut position = 0usize;
    while position < fragments.len() {
        let symbol = fragments[position].symbol;
        let local_state = fragments[position].local_state;
        edge_set_ids.clear();
        while position < fragments.len()
            && fragments[position].symbol == symbol
            && fragments[position].local_state == local_state
        {
            edge_set_ids.push(fragments[position].edge_set);
            position += 1;
        }
        edge_set_ids.sort_unstable();
        edge_set_ids.dedup();
        output.push(DirectGroupedConfig {
            symbol,
            local_state,
            edge_set: edge_sets.union(edge_set_ids),
        });
    }
}

fn compile_direct_expression_graph_grouped(
    graph: &crate::automata::unweighted_u32::nfa::NFA,
    edges: &[DirectExpressionGraphEdge],
    config_edge: &[u32],
    config_local_state: &[u32],
    ready_expansions: &[DirectReadyExpansion],
    initial_accepting: bool,
    initial_configs: &[u32],
    local_dfas: &[DFA],
    local_class_transitions: &[Vec<Box<[(u8, u32)]>>],
    class_members: &[Vec<u8>],
    profile: bool,
) -> DFA {
    let setup_started = profile.then(Instant::now);
    let mut edge_sets = DirectEdgeSetInterner::new();
    let ready_groups = ready_expansions
        .iter()
        .map(|expansion| {
            group_direct_configs(
                &expansion.configs,
                config_edge,
                config_local_state,
                edges,
                &mut edge_sets,
            )
        })
        .collect::<Vec<_>>();
    let mut fragments = Vec::<DirectGroupedConfig>::new();
    let mut edge_set_ids = Vec::<u32>::new();
    let mut canonical_groups = Vec::<DirectGroupedConfig>::new();
    let grouped_continuations = edges
        .iter()
        .map(|edge| {
            fragments.clear();
            let mut accepting = false;
            for &target in edge.targets.iter() {
                accepting |= ready_expansions[target as usize].accepting;
                fragments.extend_from_slice(&ready_groups[target as usize]);
            }
            canonicalize_direct_group_fragments(
                &mut fragments,
                &mut edge_sets,
                &mut edge_set_ids,
                &mut canonical_groups,
            );
            DirectGroupedContinuation {
                accepting,
                groups: canonical_groups.clone().into_boxed_slice(),
            }
        })
        .collect::<Vec<_>>();
    let initial_groups = group_direct_configs(
        initial_configs,
        config_edge,
        config_local_state,
        edges,
        &mut edge_sets,
    );
    let setup_ms = setup_started.map_or(0.0, |started| {
        started.elapsed().as_secs_f64() * 1000.0
    });

    let determinize_started = profile.then(Instant::now);
    let initial = Arc::new(DirectGroupedState::new(initial_accepting, initial_groups));
    let mut dfa = DFA::new(1);
    dfa.ensure_group_capacity(1);
    let state_capacity = config_edge.len().max(edges.len() * 8);
    let mut state_map = FxHashMap::<Arc<DirectGroupedState>, u32>::default();
    state_map.reserve(state_capacity);
    state_map.insert(Arc::clone(&initial), 0);
    let mut states = Vec::<Arc<DirectGroupedState>>::with_capacity(state_capacity);
    states.push(initial);
    let mut accepting_states = Vec::with_capacity(state_capacity);
    accepting_states.push(initial_accepting);
    let mut worklist = Vec::with_capacity(state_capacity);
    worklist.push(0u32);

    let num_classes = class_members.len();
    let mut class_fragments = (0..num_classes)
        .map(|_| Vec::<DirectGroupedConfig>::new())
        .collect::<Vec<_>>();
    let mut class_accepting = vec![false; num_classes];
    let mut class_used = vec![false; num_classes];
    let mut used_classes = Vec::<u8>::with_capacity(num_classes);
    let mut target_groups = Vec::<DirectGroupedConfig>::new();
    let mut state_hits = 0usize;
    let mut state_misses = 0usize;
    let mut source_groups = 0usize;
    let mut direct_group_contributions = 0usize;
    let mut continuation_group_contributions = 0usize;
    let mut scan_duration = std::time::Duration::ZERO;
    let mut canonicalize_duration = std::time::Duration::ZERO;
    let mut lookup_duration = std::time::Duration::ZERO;

    while let Some(source_state) = worklist.pop() {
        let source = Arc::clone(&states[source_state as usize]);
        if profile {
            source_groups += source.groups.len();
        }
        let scan_started = profile.then(Instant::now);
        for group in source.groups.iter().copied() {
            let symbol = group.symbol as usize;
            let local_dfa = &local_dfas[symbol];
            for &(class, next_local) in
                local_class_transitions[symbol][group.local_state as usize].iter()
            {
                let class_index = class as usize;
                if !class_used[class_index] {
                    class_used[class_index] = true;
                    used_classes.push(class);
                }
                if !local_class_transitions[symbol][next_local as usize].is_empty() {
                    class_fragments[class_index].push(DirectGroupedConfig {
                        symbol: group.symbol,
                        local_state: next_local,
                        edge_set: group.edge_set,
                    });
                    if profile {
                        direct_group_contributions += 1;
                    }
                }
                if !local_dfa.finalizers(next_local).is_empty() {
                    for &edge in edge_sets.get(group.edge_set) {
                        let continuation = &grouped_continuations[edge as usize];
                        class_accepting[class_index] |= continuation.accepting;
                        class_fragments[class_index].extend_from_slice(&continuation.groups);
                        if profile {
                            continuation_group_contributions += continuation.groups.len();
                        }
                    }
                }
            }
        }
        if let Some(started) = scan_started {
            scan_duration += started.elapsed();
        }

        used_classes.sort_unstable();
        let mut transitions = Vec::with_capacity(used_classes.len());
        for &class in &used_classes {
            let class_index = class as usize;
            let canonicalize_started = profile.then(Instant::now);
            canonicalize_direct_group_fragments(
                &mut class_fragments[class_index],
                &mut edge_sets,
                &mut edge_set_ids,
                &mut target_groups,
            );
            if let Some(started) = canonicalize_started {
                canonicalize_duration += started.elapsed();
            }
            let accepting = class_accepting[class_index];
            class_fragments[class_index].clear();
            class_accepting[class_index] = false;
            class_used[class_index] = false;
            if target_groups.is_empty() && !accepting {
                continue;
            }

            let lookup_started = profile.then(Instant::now);
            let key = DirectGroupedState::new(
                accepting,
                std::mem::take(&mut target_groups).into_boxed_slice(),
            );
            let target = if let Some(&existing) = state_map.get(&key) {
                if profile {
                    state_hits += 1;
                }
                target_groups = key.groups.into_vec();
                existing
            } else {
                if profile {
                    state_misses += 1;
                }
                let target = dfa.add_state();
                let key = Arc::new(key);
                state_map.insert(Arc::clone(&key), target);
                states.push(key);
                accepting_states.push(accepting);
                worklist.push(target);
                target_groups = Vec::new();
                target
            };
            if let Some(started) = lookup_started {
                lookup_duration += started.elapsed();
            }
            transitions.push((class, target));
        }
        used_classes.clear();
        dfa.set_transitions_from_sorted_entries(source_state, transitions);
    }
    set_direct_expression_graph_finalizers(&mut dfa, &accepting_states);
    let determinize_ms = determinize_started.map_or(0.0, |started| {
        started.elapsed().as_secs_f64() * 1000.0
    });
    let pre_states = dfa.num_states();
    let pre_transitions = dfa.transition_count();
    let minimize_started = profile.then(Instant::now);
    let mut minimized = dfa.minimize_owned_reachable();
    let minimize_ms = minimize_started.map_or(0.0, |started| {
        started.elapsed().as_secs_f64() * 1000.0
    });
    let post_class_transitions = minimized.transition_count();
    let expand_started = profile.then(Instant::now);
    expand_direct_expression_graph_classes(&mut minimized, class_members);
    let expand_ms = expand_started.map_or(0.0, |started| {
        started.elapsed().as_secs_f64() * 1000.0
    });
    if profile {
        eprintln!(
            "[glrmask/profile][tokenizer] grouped_expression_graph edge_sets={} union_cache_hits={} union_cache_misses={} pre_states={} pre_class_transitions={} post_states={} post_class_transitions={} post_byte_transitions={} avg_source_groups={:.2} direct_group_contributions={} continuation_group_contributions={} state_hits={} state_misses={} setup_ms={:.3} scan_ms={:.3} canonicalize_ms={:.3} lookup_ms={:.3} determinize_ms={:.3} minimize_ms={:.3} expand_ms={:.3}",
            edge_sets.sets.len(),
            edge_sets.union_cache_hits,
            edge_sets.union_cache_misses,
            pre_states,
            pre_transitions,
            minimized.num_states(),
            post_class_transitions,
            minimized.transition_count(),
            source_groups as f64 / pre_states.max(1) as f64,
            direct_group_contributions,
            continuation_group_contributions,
            state_hits,
            state_misses,
            setup_ms,
            scan_duration.as_secs_f64() * 1000.0,
            canonicalize_duration.as_secs_f64() * 1000.0,
            lookup_duration.as_secs_f64() * 1000.0,
            determinize_ms,
            minimize_ms,
            expand_ms,
        );
    }
    minimized
}

fn direct_expression_graph_ready_expansions(
    graph: &crate::automata::unweighted_u32::nfa::NFA,
    outgoing_edges: &[Vec<u32>],
    edges: &[DirectExpressionGraphEdge],
    local_dfas: &[DFA],
    local_class_transitions: &[Vec<Box<[(u8, u32)]>>],
) -> Vec<DirectReadyExpansion> {
    let num_states = graph.states.len();
    let mut marks = vec![0u32; num_states];
    let mut generation = 0u32;
    let mut stack = Vec::<u32>::new();
    let mut configs = Vec::<u32>::new();
    let mut result = Vec::with_capacity(num_states);

    for root in 0..num_states {
        generation = generation.wrapping_add(1);
        if generation == 0 {
            marks.fill(0);
            generation = 1;
        }
        stack.clear();
        configs.clear();
        stack.push(root as u32);
        marks[root] = generation;
        let mut accepting = false;
        while let Some(state) = stack.pop() {
            let graph_state = &graph.states[state as usize];
            accepting |= graph_state.is_accepting;
            for &edge_id in &outgoing_edges[state as usize] {
                let edge = &edges[edge_id as usize];
                let local_dfa = &local_dfas[edge.symbol];
                if !local_class_transitions[edge.symbol][0].is_empty() {
                    configs.push(edge.config_base);
                }
                if !local_dfa.finalizers(0).is_empty() {
                    for &target in edge.targets.iter() {
                        let mark = &mut marks[target as usize];
                        if *mark != generation {
                            *mark = generation;
                            stack.push(target);
                        }
                    }
                }
            }
            for &target in &graph_state.epsilons {
                let mark = &mut marks[target as usize];
                if *mark != generation {
                    *mark = generation;
                    stack.push(target);
                }
            }
        }
        configs.sort_unstable();
        configs.dedup();
        result.push(DirectReadyExpansion {
            configs: configs.clone().into_boxed_slice(),
            accepting,
        });
    }
    result
}

fn union_direct_ready_expansions(
    targets: impl IntoIterator<Item = u32>,
    expansions: &[DirectReadyExpansion],
    config_marks: &mut [u32],
    generation: &mut u32,
    configs: &mut Vec<u32>,
) -> bool {
    *generation = generation.wrapping_add(1);
    if *generation == 0 {
        config_marks.fill(0);
        *generation = 1;
    }
    configs.clear();
    let mut accepting = false;
    for target in targets {
        let expansion = &expansions[target as usize];
        accepting |= expansion.accepting;
        for &config in expansion.configs.iter() {
            let mark = &mut config_marks[config as usize];
            if *mark != *generation {
                *mark = *generation;
                configs.push(config);
            }
        }
    }
    configs.sort_unstable();
    accepting
}

fn set_direct_expression_graph_finalizers(dfa: &mut DFA, accepting: &[bool]) {
    for (state, &is_accepting) in accepting.iter().enumerate() {
        let mut finalizers = BitSet::new(1);
        if is_accepting {
            finalizers.set(0);
        }
        // The owned minimizer recomputes strict possible-future metadata after
        // quotienting. Computing it on the pre-minimized graph would be a full
        // redundant graph traversal.
        dfa.overwrite_state_metadata(state as u32, finalizers, BitSet::new(1));
    }
}

pub(super) fn try_compile_expression_labeled_nfa_direct(
    graph: &crate::automata::unweighted_u32::nfa::NFA,
    symbols: &[Expr],
    force: bool,
) -> Result<Option<DFA>, String> {
    let edge_group_count = graph
        .states
        .iter()
        .map(|state| state.transitions.len())
        .sum::<usize>();
    if !force && (graph.states.len() < 128 || edge_group_count < 64) {
        return Ok(None);
    }

    let profile = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    let total_started = profile.then(Instant::now);
    let local_started = profile.then(Instant::now);
    let local_dfas = symbols
        .par_iter()
        .map(compile_terminal_expr_dfa)
        .collect::<Vec<_>>();
    let local_ms = local_started.map_or(0.0, |started| {
        started.elapsed().as_secs_f64() * 1000.0
    });
    if local_dfas
        .iter()
        .any(|dfa| dfa.num_states() == 0 || dfa.epsilon_transition_count() != 0)
    {
        return Ok(None);
    }

    let (_byte_class_map, class_members) = direct_expression_graph_byte_classes(&local_dfas);
    let local_class_transitions =
        direct_expression_graph_class_transitions(&local_dfas, &class_members);

    let setup_started = profile.then(Instant::now);
    let mut outgoing_edges = vec![Vec::<u32>::new(); graph.states.len()];
    let mut edges = Vec::<DirectExpressionGraphEdge>::with_capacity(edge_group_count);
    let mut config_edge = Vec::<u32>::new();
    let mut config_local_state = Vec::<u32>::new();
    for (source, state) in graph.states.iter().enumerate() {
        for (&label, targets) in &state.transitions {
            let symbol = usize::try_from(label)
                .map_err(|_| format!("NFA has invalid negative symbol label {label}"))?;
            let local_dfa = local_dfas.get(symbol).ok_or_else(|| {
                format!(
                    "NFA symbol label {label} is out of range for {} symbols",
                    symbols.len()
                )
            })?;
            let config_base = u32::try_from(config_edge.len())
                .map_err(|_| "direct expression graph configuration overflow".to_string())?;
            let edge_id = u32::try_from(edges.len())
                .map_err(|_| "direct expression graph edge overflow".to_string())?;
            let mut checked_targets = Vec::with_capacity(targets.len());
            for &target in targets {
                if target as usize >= graph.states.len() {
                    return Err(format!("NFA target state {target} is out of range"));
                }
                checked_targets.push(target);
            }
            checked_targets.sort_unstable();
            checked_targets.dedup();
            for local_state in 0..local_dfa.num_states() {
                config_edge.push(edge_id);
                config_local_state.push(local_state as u32);
            }
            edges.push(DirectExpressionGraphEdge {
                symbol,
                targets: checked_targets.into_boxed_slice(),
                config_base,
            });
            outgoing_edges[source].push(edge_id);
        }
    }
    if config_edge.is_empty() {
        return Ok(None);
    }

    for &start in &graph.start_states {
        if start as usize >= graph.states.len() {
            return Err(format!("NFA start state {start} is out of range"));
        }
    }

    let ready_expansions = direct_expression_graph_ready_expansions(
        graph,
        &outgoing_edges,
        &edges,
        &local_dfas,
        &local_class_transitions,
    );
    let mut config_marks = vec![0u32; config_edge.len()];
    let mut generation = 0u32;
    let mut initial_configs = Vec::<u32>::new();
    let initial_accepting = union_direct_ready_expansions(
        graph.start_states.iter().copied(),
        &ready_expansions,
        &mut config_marks,
        &mut generation,
        &mut initial_configs,
    );
    let setup_ms = setup_started.map_or(0.0, |started| {
        started.elapsed().as_secs_f64() * 1000.0
    });

    let result = compile_direct_expression_graph_grouped(
        graph,
        &edges,
        &config_edge,
        &config_local_state,
        &ready_expansions,
        initial_accepting,
        &initial_configs,
        &local_dfas,
        &local_class_transitions,
        &class_members,
        profile,
    );
    if let Some(started) = total_started {
        eprintln!(
            "[glrmask/profile][tokenizer] grouped_expression_graph_total graph_states={} graph_edges={} symbols={} classes={} local_states={} configs={} local_ms={:.3} outer_setup_ms={:.3} total_ms={:.3}",
            graph.states.len(),
            edge_group_count,
            symbols.len(),
            class_members.len(),
            local_dfas.iter().map(DFA::num_states).sum::<usize>(),
            config_edge.len(),
            local_ms,
            setup_ms,
            started.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Ok(Some(result))
}

pub(super) fn compile_expression_labeled_nfa_via_byte_nfa(
    graph: &crate::automata::unweighted_u32::nfa::NFA,
    symbols: &[Expr],
) -> Result<DFA, String> {
    // State zero is a synthetic root so multiple starts remain representable.
    let mut nfa = NFA::new(1);
    let state_map = (0..graph.states.len())
        .map(|_| nfa.add_state())
        .collect::<Vec<_>>();

    for &start in &graph.start_states {
        let Some(&mapped) = state_map.get(start as usize) else {
            return Err(format!("NFA start state {start} is out of range"));
        };
        nfa.add_epsilon(0, mapped);
    }

    for (state_id, state) in graph.states.iter().enumerate() {
        let mapped_from = state_map[state_id];
        if state.is_accepting {
            nfa.add_finalizer(mapped_from, 0);
        }

        for &target in &state.epsilons {
            let Some(&mapped_target) = state_map.get(target as usize) else {
                return Err(format!("NFA epsilon target {target} is out of range"));
            };
            nfa.add_epsilon(mapped_from, mapped_target);
        }

        for (&label, targets) in &state.transitions {
            let symbol_index = usize::try_from(label)
                .map_err(|_| format!("NFA has invalid negative symbol label {label}"))?;
            let symbol = symbols.get(symbol_index).ok_or_else(|| {
                format!(
                    "NFA symbol label {label} is out of range for {} symbols",
                    symbols.len()
                )
            })?;
            for &target in targets {
                let Some(&mapped_target) = state_map.get(target as usize) else {
                    return Err(format!("NFA target state {target} is out of range"));
                };
                append_compiled_expr(symbol, &mut nfa, mapped_from, mapped_target);
            }
        }
    }

    nfa.condense_epsilon_sccs();
    Ok(nfa.to_minimized_dfa())
}

/// Compile an explicit expression-labelled graph into one byte-level lexer DFA.
///
/// Large graphs are composed directly from independently compiled edge DFAs,
/// including nullable edge languages. This preserves the graph's factorization
/// and avoids constructing
/// a global epsilon NFA whose powerset immediately has to rediscover it.
pub fn compile_expression_labeled_nfa(
    graph: &crate::automata::unweighted_u32::nfa::NFA,
    symbols: &[Expr],
) -> Result<DFA, String> {
    if graph.start_states.is_empty() {
        return Err("expression-labelled NFA has no start state".to_string());
    }
    if let Some(dfa) = try_compile_expression_labeled_nfa_direct(graph, symbols, false)? {
        return Ok(dfa);
    }
    compile_expression_labeled_nfa_via_byte_nfa(graph, symbols)
}
