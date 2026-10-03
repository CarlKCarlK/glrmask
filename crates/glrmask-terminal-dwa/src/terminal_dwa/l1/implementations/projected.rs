//! Exact projected-residual L1 compiler.
//!
//! A state represents one terminal's residual language from one lexer
//! configuration. Every non-dead state is accepting: surviving to a model-token
//! boundary is exactly `finalizer || possible_future`. All `(raw state, terminal)`
//! residuals are roots; minimization therefore never prunes by reachability from
//! one distinguished start state.

use std::collections::{BTreeMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Instant;

use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHasher};

use super::{BuildInput, LocalIdMapTerminalDwa, common};
use crate::compiler::stages::equiv_types::ManyToOneIdMap;
use crate::automata::lexer::tokenizer::{SingletonEpsilonClosures, TerminalResidualCoordinates};
use crate::automata::lexer::{DFA, Lexer};
use crate::terminal_dwa::l1::implementations::support::{DEAD, Scanner};
use crate::Vocab;

const UNKNOWN: u32 = u32::MAX - 1;

enum ConfigStates {
    One(u32),
    Many(Arc<[u32]>),
}

impl ConfigStates {
    fn len(&self) -> usize {
        match self {
            Self::One(_) => 1,
            Self::Many(states) => states.len(),
        }
    }
}

struct Projected<'a> {
    input: BuildInput<'a>,
    active: Vec<Vec<u32>>,
    active_by_group: Vec<Box<[u64]>>,
    terminals: Vec<u32>,
    terminal_groups: Vec<usize>,
    group_for_terminal: Vec<usize>,
    configs: Vec<(u32, ConfigStates)>,
    singleton_ids: FxHashMap<u64, u32>,
    ids: FxHashMap<(u32, Arc<[u32]>), u32>,
    transitions: Vec<Vec<(u8, u32)>>,
    singleton_closures: Arc<SingletonEpsilonClosures>,
    root_closure_states: usize,
    root_memberships: usize,
}

impl<'a> Projected<'a> {
    fn new(input: BuildInput<'a>) -> Self {
        let active = (0..input.tokenizer.num_states())
            .map(|state| {
                super::super::collect_active_terminal_signature(
                    input.tokenizer,
                    state,
                    input.active_terminals,
                )
            })
            .collect::<Vec<_>>();
        let terminals = input
            .active_terminals
            .iter()
            .enumerate()
            .filter_map(|(terminal, &active)| active.then_some(terminal as u32))
            .collect::<Vec<_>>();
        let mut terminal_index = vec![usize::MAX; input.active_terminals.len()];
        for (index, &terminal) in terminals.iter().enumerate() {
            terminal_index[terminal as usize] = index;
        }
        let words = (input.tokenizer.num_states() as usize).div_ceil(64);
        let mut active_by_terminal = (0..terminals.len())
            .map(|_| vec![0u64; words].into_boxed_slice())
            .collect::<Vec<_>>();
        for (state, signature) in active.iter().enumerate() {
            for &terminal in signature {
                let index = terminal_index[terminal as usize];
                active_by_terminal[index][state / 64] |= 1u64 << (state % 64);
            }
        }

        // Terminal identity is irrelevant to the residual language once two
        // terminals have exactly the same active-state predicate. Quotienting
        // those predicates before residual construction prevents duplicated
        // `(terminal, configuration)` subautomata while retaining every
        // terminal in the emitted signature.
        let mut group_ids = FxHashMap::<Vec<u64>, usize>::default();
        let mut active_by_group = Vec::<Box<[u64]>>::new();
        let mut group_for_terminal = vec![usize::MAX; input.active_terminals.len()];
        let mut terminal_groups = Vec::with_capacity(terminals.len());
        for (&terminal, active_states) in terminals.iter().zip(active_by_terminal) {
            let group = if let Some(&group) = group_ids.get(active_states.as_ref()) {
                group
            } else {
                let group = active_by_group.len();
                group_ids.insert(active_states.to_vec(), group);
                active_by_group.push(active_states);
                group
            };
            group_for_terminal[terminal as usize] = group;
            terminal_groups.push(group);
        }
        Self {
            input,
            active,
            active_by_group,
            terminals,
            terminal_groups,
            group_for_terminal,
            configs: Vec::new(),
            singleton_ids: FxHashMap::default(),
            ids: FxHashMap::default(),
            transitions: Vec::new(),
            singleton_closures: input.tokenizer.all_singleton_epsilon_closures(),
            root_closure_states: 0,
            root_memberships: 0,
        }
    }

    fn state_active(&self, state: u32, group: u32) -> bool {
        self.active_by_group[group as usize][state as usize / 64]
            & (1u64 << (state as usize % 64))
            != 0
    }

    fn intern_singleton(&mut self, group: u32, state: u32) -> u32 {
        let key = (u64::from(group) << 32) | u64::from(state);
        if let Some(&id) = self.singleton_ids.get(&key) {
            return id;
        }
        let id = self.configs.len() as u32;
        self.singleton_ids.insert(key, id);
        self.configs.push((group, ConfigStates::One(state)));
        self.transitions.push(Vec::new());
        id
    }

    fn intern_filtered(&mut self, group: u32, states: Vec<u32>) -> u32 {
        if states.is_empty() {
            return DEAD;
        }
        debug_assert!(states.windows(2).all(|pair| pair[0] < pair[1]));
        if states.len() == 1 {
            return self.intern_singleton(group, states[0]);
        }
        let key = (group, Arc::<[u32]>::from(states));
        if let Some(&id) = self.ids.get(&key) {
            return id;
        }
        let id = self.configs.len() as u32;
        self.ids.insert(key.clone(), id);
        self.configs
            .push((group, ConfigStates::Many(Arc::clone(&key.1))));
        self.transitions.push(Vec::new());
        id
    }

    fn intern(&mut self, group: u32, mut states: Vec<u32>) -> u32 {
        states.retain(|&state| self.state_active(state, group));
        self.intern_filtered(group, states)
    }

    /// Project one epsilon closure onto all active terminal-predicate groups.
    /// Store only live memberships; dense `raw_state × group` root matrices are
    /// overwhelmingly DEAD on the large synthesized tokenizers.
    fn root_sparse_row(&mut self, raw: u32) -> Vec<(u32, u32)> {
        let closure_len = self.singleton_closures[raw as usize].len();
        self.root_closure_states += closure_len;
        if closure_len == 1 {
            let state = self.singleton_closures[raw as usize][0];
            let terminals = self.active[state as usize].clone();
            let mut row = Vec::<(u32, u32)>::with_capacity(terminals.len());
            for terminal in terminals {
                let group = self.group_for_terminal[terminal as usize];
                debug_assert_ne!(group, usize::MAX);
                row.push((group as u32, self.intern_singleton(group as u32, state)));
            }
            row.sort_unstable_by_key(|&(group, _)| group);
            row.dedup_by_key(|entry| entry.0);
            self.root_memberships += row.len();
            return row;
        }

        let closure = self.singleton_closures[raw as usize].to_vec();
        let mut grouped = (0..self.active_by_group.len())
            .map(|_| Vec::<u32>::new())
            .collect::<Vec<_>>();
        for state in closure {
            for &terminal in &self.active[state as usize] {
                let group = self.group_for_terminal[terminal as usize];
                debug_assert_ne!(group, usize::MAX);
                if grouped[group].last().copied() != Some(state) {
                    grouped[group].push(state);
                    self.root_memberships += 1;
                }
            }
        }
        let mut row = Vec::<(u32, u32)>::new();
        for (group, states) in grouped.iter_mut().enumerate() {
            if states.is_empty() {
                continue;
            }
            let projected = self.intern_filtered(group as u32, std::mem::take(states));
            if projected != DEAD {
                row.push((group as u32, projected));
            }
        }
        row
    }

    fn step(&mut self, state: u32, byte: u8, roots: &SparseRoots) -> u32 {
        let group = self.configs[state as usize].0;
        match &self.configs[state as usize].1 {
            ConfigStates::One(source) => {
                let target = self.input.flat_trans[*source as usize * 256 + byte as usize];
                if target == DEAD {
                    DEAD
                } else {
                    roots.get(target, group)
                }
            }
            ConfigStates::Many(config) => {
                let config = Arc::clone(config);
                self.intern(
                    group,
                    self.input.tokenizer.step_all(config.as_ref(), byte).to_vec(),
                )
            }
        }
    }
}

/// Compact raw-root relation for projected residual L1. Rows are sorted by
/// terminal-predicate group and contain only live projected states. Large p1/p2
/// tokenizers have O(10^5) raw states and O(10^2) groups but fewer than one live
/// root membership per raw state on average, so CSR avoids tens of megabytes of
/// DEAD entries and substantially improves cache locality under parallel builds.
struct SparseRoots {
    offsets: Box<[u32]>,
    entries: Vec<(u32, u32)>,
}

struct DirectLocalResidual {
    original_to_local: Vec<u32>,
    transitions: Vec<Vec<(u8, u32)>>,
    elapsed_ms: f64,
}

fn build_direct_local_residual(
    dfa: &DFA,
    terminal_group: u32,
    bytes: &[u8],
    symbol_for_representative: &[u8; 256],
) -> DirectLocalResidual {
    let started = Instant::now();
    let mut original_to_live = vec![DEAD; dfa.num_states()];
    let mut live_count = 0u32;
    for state in 0..dfa.num_states() as u32 {
        if dfa.finalizers(state).contains(terminal_group as usize)
            || dfa
                .possible_future_group_ids(state)
                .contains(terminal_group as usize)
        {
            original_to_live[state as usize] = live_count;
            live_count += 1;
        }
    }

    let large_dfa = dfa.transition_count() >= 500_000;
    let row_capacity = if large_dfa && dfa.num_states() != 0 {
        let average_source_edges = dfa.transition_count().div_ceil(dfa.num_states());
        average_source_edges
            .saturating_mul(bytes.len())
            .div_ceil(256)
            .max(1)
    } else {
        0
    };
    let mut live_transitions = vec![Vec::<(u8, u32)>::new(); live_count as usize];
    for state in 0..dfa.num_states() as u32 {
        let source = original_to_live[state as usize];
        if source == DEAD {
            continue;
        }
        let row = &mut live_transitions[source as usize];
        if row_capacity != 0 {
            row.reserve(row_capacity);
        }
        for (byte, target_state) in dfa.transitions(state) {
            let symbol = symbol_for_representative[byte as usize];
            if symbol == u8::MAX {
                continue;
            }
            let target = original_to_live[target_state as usize];
            if target != DEAD {
                row.push((symbol, target));
            }
        }
        debug_assert!(row.windows(2).all(|pair| pair[0].0 < pair[1].0));
    }

    // Dense terminal DFAs are poor local-minimization candidates: on the hot
    // cases they collapse by only a handful of states while scanning tens of
    // thousands of edges. Passing them through is exact because the final
    // cross-terminal minimizer still computes the full right congruence.
    let live_edge_count = live_transitions.iter().map(Vec::len).sum::<usize>();
    let skip_dense_edges = live_edge_count >= 5_000;
    if skip_dense_edges {
        let mut original_to_local = vec![DEAD; dfa.num_states()];
        for (original, &live) in original_to_live.iter().enumerate() {
            if live != DEAD {
                original_to_local[original] = live;
            }
        }
        return DirectLocalResidual {
            original_to_local,
            transitions: live_transitions,
            elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        };
    }

    let minimized = minimize(&live_transitions, bytes);
    let mut original_to_local = vec![DEAD; dfa.num_states()];
    for (original, &live) in original_to_live.iter().enumerate() {
        if live != DEAD {
            original_to_local[original] = minimized.classes[live as usize];
        }
    }
    let mut transitions = vec![Vec::<(u8, u32)>::new(); minimized.state_count];
    for source in 0..minimized.state_count {
        for (symbol, column) in minimized.columns.iter().enumerate() {
            let target = column[source];
            if target != DEAD {
                transitions[source].push((symbol as u8, target));
            }
        }
    }
    DirectLocalResidual {
        original_to_local,
        transitions,
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
    }
}

impl SparseRoots {
    fn from_rows(rows: Vec<Vec<(u32, u32)>>) -> Self {
        let mut offsets = Vec::with_capacity(rows.len() + 1);
        let total = rows.iter().map(Vec::len).sum::<usize>();
        let mut entries = Vec::with_capacity(total);
        offsets.push(0);
        for row in rows {
            debug_assert!(row.windows(2).all(|pair| pair[0].0 < pair[1].0));
            entries.extend(row);
            offsets.push(entries.len() as u32);
        }
        Self { offsets: offsets.into_boxed_slice(), entries }
    }

    #[inline]
    fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    #[inline]
    fn row(&self, raw: usize) -> &[(u32, u32)] {
        let start = self.offsets[raw] as usize;
        let end = self.offsets[raw + 1] as usize;
        &self.entries[start..end]
    }

    #[inline]
    fn get(&self, raw: u32, group: u32) -> u32 {
        let row = self.row(raw as usize);
        match row {
            [] => DEAD,
            [(only_group, state)] => {
                if *only_group == group { *state } else { DEAD }
            }
            _ => row
                .binary_search_by_key(&group, |&(candidate, _)| candidate)
                .ok()
                .map_or(DEAD, |index| row[index].1),
        }
    }

    fn remap_states(&mut self, classes: &[u32]) {
        for (_, state) in &mut self.entries {
            *state = classes[*state as usize];
        }
    }
}

/// Construct the same live-prefix residual union directly from the
/// single-terminal DFAs retained by tokenizer construction. Terminal identity
/// labels roots, but is not part of the residual language: the ordinary global
/// minimization below remains free to merge equivalent states across terminals.
fn build_direct_terminal_residual_machine<'a>(
    input: BuildInput<'a>,
    bytes: &[u8],
) -> Option<(Projected<'a>, SparseRoots)> {
    let coordinates = input.tokenizer.terminal_residual_coordinates()?;
    build_direct_terminal_residual_machine_with_coordinates(input, bytes, coordinates)
}

fn build_direct_terminal_residual_machine_with_coordinates<'a>(
    input: BuildInput<'a>,
    bytes: &[u8],
    coordinates: &TerminalResidualCoordinates,
) -> Option<(Projected<'a>, SparseRoots)> {
    let profile = std::env::var_os("GLRMASK_PROFILE_L1_IMPLEMENTATIONS").is_some();
    let total_started = profile.then(Instant::now);
    if coordinates.len() != input.tokenizer.num_states() as usize
        || coordinates.terminal_dfa_count() < input.active_terminals.len()
    {
        return None;
    }

    let terminals = input
        .active_terminals
        .iter()
        .enumerate()
        .filter_map(|(terminal, &active)| active.then_some(terminal as u32))
        .collect::<Vec<_>>();
    let mut group_for_terminal = vec![usize::MAX; input.active_terminals.len()];
    let mut terminal_groups = Vec::with_capacity(terminals.len());
    for (group, &terminal) in terminals.iter().enumerate() {
        group_for_terminal[terminal as usize] = group;
        terminal_groups.push(group);
    }

    let mut symbol_for_representative = [u8::MAX; 256];
    for (symbol, &byte) in bytes.iter().enumerate() {
        symbol_for_representative[byte as usize] = symbol as u8;
    }
    let local_total_work_ms: f64;
    let local_max_ms: f64;
    let (configs, transitions, state_maps, index_ms, edges_ms) = {
        let local_started = profile.then(Instant::now);
        // Rayon scheduling is material when this branch contains only a few
        // tiny terminal residual DFAs (common for structural JSON terminals).
        // Choose from actual residual work, not partition/schema identity: keep
        // small unions on the calling worker and retain parallel construction
        // for larger local automata where it amortizes scheduling.
        const SEQUENTIAL_DIRECT_LOCAL_MAX_TERMINALS: usize = 4;
        const SEQUENTIAL_DIRECT_LOCAL_MAX_TRANSITIONS: usize = 4_096;
        let direct_local_transition_work = terminals
            .iter()
            .filter_map(|&terminal| coordinates.terminal_dfa_and_group(terminal))
            .map(|(dfa, _)| dfa.transition_count())
            .sum::<usize>();
        let sequential_direct_locals = terminals.len() <= SEQUENTIAL_DIRECT_LOCAL_MAX_TERMINALS
            && direct_local_transition_work <= SEQUENTIAL_DIRECT_LOCAL_MAX_TRANSITIONS;
        let build_local = |terminal: u32| {
            coordinates
                .terminal_dfa_and_group(terminal)
                .map(|(dfa, group)| {
                    build_direct_local_residual(dfa, group, bytes, &symbol_for_representative)
                })
        };
        let locals = if sequential_direct_locals {
            terminals
                .iter()
                .copied()
                .map(build_local)
                .collect::<Vec<_>>()
        } else {
            terminals
                .par_iter()
                .copied()
                .map(build_local)
                .collect::<Vec<_>>()
        };
        if locals.iter().any(Option::is_none) {
            return None;
        }
        let locals = locals.into_iter().map(Option::unwrap).collect::<Vec<_>>();
        local_total_work_ms = locals.iter().map(|local| local.elapsed_ms).sum();
        local_max_ms = locals.iter().map(|local| local.elapsed_ms).fold(0.0f64, f64::max);
        if profile
            && let Some((index, slowest)) = locals
                .iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| left.elapsed_ms.total_cmp(&right.elapsed_ms))
        {
            let terminal = terminals[index];
            if let Some((dfa, _)) = coordinates.terminal_dfa_and_group(terminal) {
                eprintln!(
                    "[glrmask/profile][l1_direct_local_max] partition={} terminal={} source_states={} source_edges={} reduced_states={} reduced_edges={} elapsed_ms={:.3}",
                    input.partition_label,
                    terminal,
                    dfa.num_states(),
                    dfa.transition_count(),
                    slowest.transitions.len(),
                    slowest.transitions.iter().map(Vec::len).sum::<usize>(),
                    slowest.elapsed_ms,
                );
            }
        }
        if std::env::var_os("GLRMASK_PROFILE_L1_DIRECT_LOCALS").is_some() {
            for (index, local) in locals.iter().enumerate() {
                let terminal = terminals[index];
                let Some((dfa, _)) = coordinates.terminal_dfa_and_group(terminal) else {
                    continue;
                };
                if dfa.transition_count() < 500 {
                    continue;
                }
                eprintln!(
                    "[glrmask/profile][l1_direct_local] partition={} terminal={} source_states={} source_edges={} output_states={} output_edges={} elapsed_ms={:.3}",
                    input.partition_label,
                    terminal,
                    dfa.num_states(),
                    dfa.transition_count(),
                    local.transitions.len(),
                    local.transitions.iter().map(Vec::len).sum::<usize>(),
                    local.elapsed_ms,
                );
            }
        }

        let concat_started = profile.then(Instant::now);
        let mut configs = Vec::<(u32, ConfigStates)>::new();
        let mut transitions = Vec::<Vec<(u8, u32)>>::new();
        let mut state_maps = vec![None::<Vec<u32>>; input.active_terminals.len()];
        for (group, (&terminal, local)) in terminals.iter().zip(locals).enumerate() {
            let base = configs.len() as u32;
            let DirectLocalResidual {
                original_to_local,
                transitions: mut local_transitions,
                elapsed_ms: _,
            } = local;
            let map = original_to_local
                .into_iter()
                .map(|state| if state == DEAD { DEAD } else { base + state })
                .collect::<Vec<_>>();
            state_maps[terminal as usize] = Some(map);
            configs.extend(
                (0..local_transitions.len())
                    .map(|state| (group as u32, ConfigStates::One(state as u32))),
            );
            for row in &mut local_transitions {
                for (_, target) in row {
                    *target += base;
                }
            }
            transitions.extend(local_transitions);
        }
        (
            configs,
            transitions,
            state_maps,
            local_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
            concat_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
        )
    };

    let roots_started = profile.then(Instant::now);
    let singleton_closures = input.tokenizer.all_singleton_epsilon_closures();
    let mut root_closure_states = 0usize;
    let mut root_memberships = 0usize;
    let raw_states = input.tokenizer.num_states() as usize;
    let mut root_offsets = Vec::with_capacity(raw_states + 1);
    let mut root_entries = Vec::<(u32, u32)>::with_capacity(raw_states);
    let mut row = Vec::<(u32, u32)>::new();
    root_offsets.push(0);
    for raw in 0..input.tokenizer.num_states() {
        let closure = &singleton_closures[raw as usize];
        root_closure_states += closure.len();
        row.clear();
        for &closure_state in closure.iter() {
            for &(terminal, residual_state) in coordinates.row(closure_state)? {
                let terminal = terminal as usize;
                if !input.active_terminals.get(terminal).copied().unwrap_or(false) {
                    continue;
                }
                let group = *group_for_terminal.get(terminal)?;
                if group == usize::MAX {
                    return None;
                }
                let map = state_maps.get(terminal)?.as_ref()?;
                let projected = *map.get(residual_state as usize)?;
                if projected == DEAD {
                    continue;
                }
                if let Some(existing) = row.iter().find(|(candidate, _)| *candidate == group as u32) {
                    // A deterministic terminal residual coordinate must not map
                    // one epsilon-closed raw state to two distinct residuals of
                    // the same terminal. Fail closed to the established path.
                    if existing.1 != projected {
                        return None;
                    }
                    continue;
                }
                row.push((group as u32, projected));
                root_memberships += 1;
            }
        }
        row.sort_unstable_by_key(|&(group, _)| group);
        root_entries.extend_from_slice(&row);
        root_offsets.push(root_entries.len() as u32);
    }
    let roots_ms = roots_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

    let finish_started = profile.then(Instant::now);
    let words = (input.tokenizer.num_states() as usize).div_ceil(64);
    let projected = Projected {
        input,
        active: Vec::new(),
        active_by_group: (0..terminals.len())
            .map(|_| vec![0u64; words].into_boxed_slice())
            .collect(),
        terminals,
        terminal_groups,
        group_for_terminal,
        configs,
        singleton_ids: FxHashMap::default(),
        ids: FxHashMap::default(),
        transitions,
        singleton_closures,
        root_closure_states,
        root_memberships,
    };
    let roots = SparseRoots {
        offsets: root_offsets.into_boxed_slice(),
        entries: root_entries,
    };
    if profile {
        eprintln!(
            "[glrmask/profile][l1_direct_construct] partition={} terminals={} states={} edges={} root_memberships={} local_minimize={} local_total_work_ms={:.3} local_max_ms={:.3} index_ms={:.3} edges_ms={:.3} roots_ms={:.3} finish_ms={:.3} total_ms={:.3}",
            input.partition_label,
            projected.terminals.len(),
            projected.configs.len(),
            projected.transitions.iter().map(Vec::len).sum::<usize>(),
            projected.root_memberships,
            true,
            local_total_work_ms,
            local_max_ms,
            index_ms,
            edges_ms,
            roots_ms,
            finish_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
            total_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
        );
    }
    Some((projected, roots))
}

struct Minimized {
    classes: Vec<u32>,
    /// Exact byte-equivalence columns, stored target-contiguously for reverse
    /// preimage scans.
    columns: Vec<Box<[u32]>>,
    byte_class: [u8; 256],
    state_count: usize,
    rounds: usize,
}

fn minimize(transitions: &[Vec<(u8, u32)>], alphabet: &[u8]) -> Minimized {
    minimize_seeded(transitions, alphabet, None)
}

fn minimize_direct_residual_union(
    transitions: &[Vec<(u8, u32)>],
    alphabet: &[u8],
) -> Minimized {
    let profile = std::env::var_os("GLRMASK_PROFILE_L1_IMPLEMENTATIONS").is_some();
    let seed_started = profile.then(Instant::now);
    const ROUNDS: usize = 2;
    // Safe finite-depth fingerprints. Start with one hash for every live state
    // (depth zero), then hash each sparse transition row using only the symbol
    // and the previous-round target hash. Language-equivalent states receive
    // identical hashes at every round. Hash collisions merely leave unrelated
    // states in the same seed block; the exact Hopcroft pass below splits them.
    let mut hashes = vec![0x9e37_79b9_7f4a_7c15u64; transitions.len()];
    let mut class_counts = Vec::with_capacity(ROUNDS);
    // Rayon setup dominates these two fingerprint rounds for tiny direct
    // residual unions. Keep genuinely small machines on the calling worker;
    // larger unions retain parallel hashing. The work metric counts exactly
    // the states and live edges visited by each round.
    const SEQUENTIAL_SEED_WORK_LIMIT: usize = 64;
    let transition_work = transitions.len()
        + transitions.iter().map(Vec::len).sum::<usize>();
    for _ in 0..ROUNDS {
        let hash_row = |row: &Vec<(u8, u32)>| {
            let mut hash = 0x243f_6a88_85a3_08d3u64 ^ row.len() as u64;
            for &(symbol, target) in row {
                let value = hashes[target as usize]
                    ^ (u64::from(symbol) + 1).wrapping_mul(0x9e37_79b1_85eb_ca87);
                hash ^= value.wrapping_mul(0xc2b2_ae3d_27d4_eb4f);
                hash = hash
                    .rotate_left(27)
                    .wrapping_mul(0x1656_67b1_9e37_79f9)
                    .wrapping_add(0x85eb_ca77_c2b2_ae63);
            }
            hash
        };
        hashes = if transition_work <= SEQUENTIAL_SEED_WORK_LIMIT {
            transitions.iter().map(hash_row).collect()
        } else {
            transitions.par_iter().map(hash_row).collect()
        };
        if profile {
            let mut unique = FxHashMap::<u64, ()>::default();
            for &hash in &hashes {
                unique.insert(hash, ());
            }
            class_counts.push(unique.len());
        }
    }
    let mut ids = FxHashMap::<u64, u32>::default();
    let mut next = 0u32;
    let classes = hashes
        .into_iter()
        .map(|hash| {
            *ids.entry(hash).or_insert_with(|| {
                let id = next;
                next += 1;
                id
            })
        })
        .collect::<Vec<_>>();
    let seed_ms = seed_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let exact_started = profile.then(Instant::now);
    let minimized = minimize_seeded_symbol_target_csr(transitions, alphabet, &classes);
    if profile {
        eprintln!(
            "[glrmask/profile][l1_direct_symbol_seed] states={} edges={} requested_rounds={} class_counts={:?} seed_ms={:.3} exact_ms={:.3} final_states={}",
            transitions.len(),
            transitions.iter().map(Vec::len).sum::<usize>(),
            ROUNDS,
            class_counts,
            seed_ms,
            exact_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
            minimized.state_count,
        );
    }
    minimized
}

fn minimize_seeded(
    transitions: &[Vec<(u8, u32)>],
    alphabet: &[u8],
    initial_classes: Option<&[u32]>,
) -> Minimized {
    let profile = std::env::var_os("GLRMASK_PROFILE_L1_IMPLEMENTATIONS").is_some();
    let total_started = profile.then(Instant::now);
    // The input alphabet is already quotiented by exact raw tokenizer columns.
    // That equality is preserved by epsilon closure and terminal projection,
    // so no second projected-column quotient is necessary.
    let mut byte_class = [0u8; 256];
    for (symbol, &byte) in alphabet.iter().enumerate() {
        byte_class[byte as usize] = symbol as u8;
    }

    // Every represented state is accepting; DEAD is the one implicit rejecting
    // sink. Keep the large projected automaton sparse. Its literal-heavy rows
    // contain very few live edges, and materializing `states × alphabet` here
    // would dominate both time and memory before minimization collapses it.
    let live = transitions.len();
    let blocks_started = profile.then(Instant::now);
    let mut class = initial_classes.map_or_else(|| vec![0u32; live], <[u32]>::to_vec);
    debug_assert_eq!(class.len(), live);
    let block_count = class.iter().copied().max().map_or(0usize, |value| value as usize + 1);
    let mut blocks = vec![Vec::<u32>::new(); block_count.max(usize::from(live != 0))];
    if initial_classes.is_none() && live != 0 {
        blocks[0].reserve(live);
    }
    for (state, &block) in class.iter().enumerate() {
        blocks[block as usize].push(state as u32);
    }
    debug_assert!(blocks.iter().all(|block| !block.is_empty()));
    let blocks_ms = blocks_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

    let reverse_started = profile.then(Instant::now);
    let mut incoming_counts = vec![0u32; live];
    let mut block_incoming = vec![vec![0u32; alphabet.len()].into_boxed_slice(); blocks.len()];
    for row in transitions {
        for &(symbol, target) in row {
            incoming_counts[target as usize] += 1;
            block_incoming[class[target as usize] as usize][symbol as usize] += 1;
        }
    }
    let mut incoming_offsets = vec![0u32; live + 1];
    for target in 0..live {
        incoming_offsets[target + 1] = incoming_offsets[target] + incoming_counts[target];
    }
    let mut next = incoming_offsets[..live].to_vec();
    let mut incoming = vec![(0u8, 0u32); incoming_offsets[live] as usize];
    for (source, row) in transitions.iter().enumerate() {
        for &(symbol, target) in row {
            let slot = &mut next[target as usize];
            incoming[*slot as usize] = (symbol, source as u32);
            *slot += 1;
        }
    }
    for target in 0..live {
        let start = incoming_offsets[target] as usize;
        let end = incoming_offsets[target + 1] as usize;
        incoming[start..end].sort_unstable();
    }
    let reverse_ms = reverse_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

    // Hopcroft over all residual roots. With no seed, one live block is
    // sufficient because the implicit rejecting sink's predecessor set is the
    // complement of the represented live predecessors. With a seed, refinement
    // computes the coarsest right congruence that respects those initial blocks.
    let queue_started = profile.then(Instant::now);
    let mut queue = VecDeque::<(u32, usize)>::new();
    let mut queued = vec![vec![0u8; alphabet.len()].into_boxed_slice(); blocks.len()];
    for block in 0..blocks.len() {
        for symbol in 0..alphabet.len() {
            if block_incoming[block][symbol] != 0 {
                queue.push_back((block as u32, symbol));
                queued[block][symbol] = 1;
            }
        }
    }
    let mut marked = vec![false; live];
    let mut affected_members = vec![Vec::<u32>::new(); blocks.len()];
    let mut affected_blocks = Vec::<u32>::new();
    let mut pops = 0usize;
    let queue_ms = queue_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

    let refine_started = profile.then(Instant::now);
    while let Some((splitter, symbol)) = queue.pop_front() {
        queued[splitter as usize][symbol] = 0;
        pops += 1;
        let symbol = symbol as u8;
        for &target in &blocks[splitter as usize] {
            let start = incoming_offsets[target as usize] as usize;
            let end = incoming_offsets[target as usize + 1] as usize;
            let edges = &incoming[start..end];
            let first = edges.partition_point(|&(edge_symbol, _)| edge_symbol < symbol);
            let last = edges.partition_point(|&(edge_symbol, _)| edge_symbol <= symbol);
            for &(_, source) in &edges[first..last] {
                let block = class[source as usize];
                let members = &mut affected_members[block as usize];
                if members.is_empty() {
                    affected_blocks.push(block);
                }
                members.push(source);
            }
        }
        for block_id in affected_blocks.drain(..) {
            let block_id = block_id as usize;
            let mut sources = std::mem::take(&mut affected_members[block_id]);
            if sources.len() == blocks[block_id].len() {
                sources.clear();
                affected_members[block_id] = sources;
                continue;
            }
            for &source in &sources {
                marked[source as usize] = true;
            }
            let old = std::mem::take(&mut blocks[block_id]);
            let mut inside = Vec::with_capacity(sources.len());
            let mut outside = Vec::with_capacity(old.len() - sources.len());
            for state in old {
                if marked[state as usize] {
                    inside.push(state);
                } else {
                    outside.push(state);
                }
            }
            for &source in &sources {
                marked[source as usize] = false;
            }
            blocks[block_id] = outside;
            let new_id = blocks.len();
            for &state in &inside {
                class[state as usize] = new_id as u32;
            }
            blocks.push(inside);

            let mut inside_incoming = vec![0u32; alphabet.len()];
            for &state in &blocks[new_id] {
                let start = incoming_offsets[state as usize] as usize;
                let end = incoming_offsets[state as usize + 1] as usize;
                for &(incoming_symbol, _) in &incoming[start..end] {
                    inside_incoming[incoming_symbol as usize] += 1;
                }
            }
            let mut outside_incoming = std::mem::take(&mut block_incoming[block_id]).into_vec();
            for symbol in 0..alphabet.len() {
                outside_incoming[symbol] -= inside_incoming[symbol];
            }
            block_incoming[block_id] = outside_incoming.into_boxed_slice();
            block_incoming.push(inside_incoming.into_boxed_slice());
            sources.clear();
            affected_members[block_id] = sources;
            queued.push(vec![0u8; alphabet.len()].into_boxed_slice());
            affected_members.push(Vec::new());

            for other_symbol in 0..alphabet.len() {
                if queued[block_id][other_symbol] != 0 {
                    if block_incoming[new_id][other_symbol] != 0 {
                        queue.push_back((new_id as u32, other_symbol));
                        queued[new_id][other_symbol] = 1;
                    }
                } else {
                    let smaller = if blocks[block_id].len() <= blocks[new_id].len() {
                        block_id
                    } else {
                        new_id
                    };
                    if block_incoming[smaller][other_symbol] != 0
                        && queued[smaller][other_symbol] == 0
                    {
                        queue.push_back((smaller as u32, other_symbol));
                        queued[smaller][other_symbol] = 1;
                    }
                }
            }
        }
    }
    let refine_ms = refine_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

    let output_started = profile.then(Instant::now);
    let representatives = blocks
        .iter()
        .map(|members| *members.iter().min().expect("non-empty DFA block") as usize)
        .collect::<Vec<_>>();
    let classes = class;
    let mut columns = vec![vec![DEAD; representatives.len()]; alphabet.len()];
    for (new_state, &representative) in representatives.iter().enumerate() {
        for &(symbol, target) in &transitions[representative] {
            columns[symbol as usize][new_state] = classes[target as usize];
        }
    }
    let output_ms = output_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    if profile && live >= 1000 {
        eprintln!(
            "[glrmask/profile][l1_minimize_seeded] states={} edges={} alphabet={} initial_blocks={} final_blocks={} pops={} blocks_ms={:.3} reverse_ms={:.3} queue_ms={:.3} refine_ms={:.3} output_ms={:.3} total_ms={:.3}",
            live,
            transitions.iter().map(Vec::len).sum::<usize>(),
            alphabet.len(),
            block_count.max(usize::from(live != 0)),
            representatives.len(),
            pops,
            blocks_ms,
            reverse_ms,
            queue_ms,
            refine_ms,
            output_ms,
            total_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
        );
    }
    Minimized {
        classes,
        columns: columns.into_iter().map(Vec::into_boxed_slice).collect(),
        byte_class,
        state_count: representatives.len(),
        rounds: pops,
    }
}

fn minimize_seeded_symbol_target_csr(
    transitions: &[Vec<(u8, u32)>],
    alphabet: &[u8],
    initial_classes: &[u32],
) -> Minimized {
    let profile = std::env::var_os("GLRMASK_PROFILE_L1_IMPLEMENTATIONS").is_some();
    let total_started = profile.then(Instant::now);
    let live = transitions.len();
    debug_assert_eq!(initial_classes.len(), live);

    let mut byte_class = [0u8; 256];
    for (symbol, &byte) in alphabet.iter().enumerate() {
        byte_class[byte as usize] = symbol as u8;
    }

    let mut class = initial_classes.to_vec();
    let block_count = class
        .iter()
        .copied()
        .max()
        .map_or(0usize, |value| value as usize + 1);
    let mut blocks = vec![Vec::<u32>::new(); block_count.max(usize::from(live != 0))];
    let mut position_in_block = vec![0u32; live];
    for (state, &block) in class.iter().enumerate() {
        position_in_block[state] = blocks[block as usize].len() as u32;
        blocks[block as usize].push(state as u32);
    }

    let reverse_started = profile.then(Instant::now);
    let symbols = alphabet.len();
    debug_assert!(symbols <= 256);
    let cells = symbols.saturating_mul(live);
    let target_symbol_words = symbols.div_ceil(64);
    let mut target_symbol_bits = vec![0u64; live.saturating_mul(target_symbol_words)];
    for row in transitions {
        for &(symbol, target) in row {
            if target_symbol_words != 0 {
                let word = symbol as usize / 64;
                let bit = symbol as usize % 64;
                target_symbol_bits[target as usize * target_symbol_words + word] |= 1u64 << bit;
            }
        }
    }
    // Keep one compact rank byte per `(symbol, target)` cell, but store counts
    // and predecessor offsets only for non-empty cells. This preserves O(1)
    // splitter lookup while avoiding several dense u32 arrays of
    // `alphabet.len() * live` entries.
    let mut target_symbol_counts = vec![0u32; live];
    for target in 0..live {
        let start = target * target_symbol_words;
        let end = start + target_symbol_words;
        target_symbol_counts[target] = target_symbol_bits[start..end]
            .iter()
            .map(|word| word.count_ones())
            .sum();
    }
    let mut target_symbol_offsets = vec![0u32; live + 1];
    for target in 0..live {
        target_symbol_offsets[target + 1] =
            target_symbol_offsets[target] + target_symbol_counts[target];
    }
    let mut target_symbols = vec![(0u8, 0u32); target_symbol_offsets[live] as usize];
    // Rank among the non-empty symbols for this target. With fewer than 256
    // symbols, rank 255 is free as an absent sentinel, so the refinement hot
    // loop can avoid a separate presence-bit lookup. A full 256-symbol
    // alphabet can legitimately use rank 255 and therefore retains the exact
    // `target_symbol_bits` presence check.
    let mut pair_rank = vec![u8::MAX; cells];
    for target in 0..live {
        let bits_start = target * target_symbol_words;
        let mut rank = 0usize;
        for word_index in 0..target_symbol_words {
            let mut bits = target_symbol_bits[bits_start + word_index];
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                let symbol = word_index * 64 + bit;
                debug_assert!(symbol < symbols);
                debug_assert!(rank <= u8::MAX as usize);
                let pair = target_symbol_offsets[target] as usize + rank;
                target_symbols[pair].0 = symbol as u8;
                pair_rank[symbol * live + target] = rank as u8;
                rank += 1;
                bits &= bits - 1;
            }
        }
        debug_assert_eq!(rank, target_symbol_counts[target] as usize);
    }
    for row in transitions {
        for &(symbol, target) in row {
            let rank = pair_rank[symbol as usize * live + target as usize];
            let pair = target_symbol_offsets[target as usize] as usize + rank as usize;
            target_symbols[pair].1 += 1;
        }
    }
    let pair_count = target_symbols.len();
    let mut predecessor_offsets = vec![0u32; pair_count + 1];
    for (pair, &(_, count)) in target_symbols.iter().enumerate() {
        predecessor_offsets[pair + 1] = predecessor_offsets[pair] + count;
    }
    let mut next = predecessor_offsets[..pair_count].to_vec();
    let mut predecessors = vec![0u32; predecessor_offsets[pair_count] as usize];
    for (source, row) in transitions.iter().enumerate() {
        for &(symbol, target) in row {
            let rank = pair_rank[symbol as usize * live + target as usize];
            let pair = target_symbol_offsets[target as usize] as usize + rank as usize;
            let slot = &mut next[pair];
            predecessors[*slot as usize] = source as u32;
            *slot += 1;
        }
    }
    drop(next);
    drop(target_symbol_counts);
    let target_symbol_bits = (symbols == 256).then_some(target_symbol_bits);
    let reverse_ms = reverse_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

    // There can be at most `live` blocks. Keep their per-symbol incoming counts
    // contiguous so splits append one zeroed row instead of allocating and
    // boxing a fresh row ~once per final equivalence class.
    let mut block_incoming = Vec::<u32>::with_capacity(live.saturating_mul(symbols));
    block_incoming.resize(blocks.len().saturating_mul(symbols), 0);
    for row in transitions {
        for &(symbol, target) in row {
            let block = class[target as usize] as usize;
            block_incoming[block * symbols + symbol as usize] += 1;
        }
    }
    // A `(u32, usize)` queue entry occupies 16 bytes on 64-bit targets. The
    // symbol fits in one byte, so pack it into the low byte of a u64 and keep
    // the block ID in the upper bits. FIFO order and refinement semantics are
    // unchanged while large worklists move half as many bytes.
    let mut queue = VecDeque::<u64>::new();
    // Input symbols are u8-valued, so four words cover every possible symbol.
    // Inline masks avoid one allocation per block and make queue membership a
    // compact bit test instead of a pointer-chasing byte lookup.
    let mut queued = vec![[0u64; 4]; blocks.len()];
    for block in 0..blocks.len() {
        for symbol in 0..symbols {
            if block_incoming[block * symbols + symbol] != 0 {
                queue.push_back(((block as u64) << 8) | symbol as u64);
                queued[block][symbol / 64] |= 1u64 << (symbol % 64);
            }
        }
    }
    let mut affected_members = vec![Vec::<u32>::new(); blocks.len()];
    let mut affected_blocks = Vec::<u32>::new();
    let mut affected_mark = vec![false; live];
    let mut pops = 0usize;

    let refine_started = profile.then(Instant::now);
    while let Some(work) = queue.pop_front() {
        let splitter = (work >> 8) as u32;
        let symbol = (work & 0xff) as usize;
        queued[splitter as usize][symbol / 64] &= !(1u64 << (symbol % 64));
        pops += 1;
        for &target in &blocks[splitter as usize] {
            let cell = symbol * live + target as usize;
            let rank = if symbols < 256 {
                let rank = pair_rank[cell];
                if rank == u8::MAX {
                    continue;
                }
                rank
            } else {
                let word = symbol / 64;
                let bit = 1u64 << (symbol % 64);
                if target_symbol_bits.as_ref().expect("full alphabet presence bits")
                    [target as usize * target_symbol_words + word]
                    & bit
                    == 0
                {
                    continue;
                }
                pair_rank[cell]
            };
            let pair = target_symbol_offsets[target as usize] as usize + rank as usize;
            let start = predecessor_offsets[pair] as usize;
            let end = predecessor_offsets[pair + 1] as usize;
            for &source in &predecessors[start..end] {
                let block = class[source as usize];
                let members = &mut affected_members[block as usize];
                if members.is_empty() {
                    affected_blocks.push(block);
                }
                members.push(source);
            }
        }
        for block_id in affected_blocks.drain(..) {
            let block_id = block_id as usize;
            let mut sources = std::mem::take(&mut affected_members[block_id]);
            if sources.len() == blocks[block_id].len() {
                sources.clear();
                affected_members[block_id] = sources;
                continue;
            }
            let new_id = blocks.len();
            // For one fixed input symbol a deterministic DFA has at most one
            // outgoing edge per source, so `sources` contains no duplicates.
            if sources.len() <= blocks[block_id].len() - sources.len() {
                // The affected side is the smaller half. Remove just those
                // states from the old block; this is cheaper than rescanning
                // the whole block to materialize its complement.
                let mut inside = Vec::with_capacity(sources.len());
                for &source in &sources {
                    let source = source as usize;
                    let position = position_in_block[source] as usize;
                    debug_assert_eq!(blocks[block_id][position] as usize, source);
                    let removed = blocks[block_id].swap_remove(position);
                    debug_assert_eq!(removed as usize, source);
                    if position < blocks[block_id].len() {
                        let moved = blocks[block_id][position] as usize;
                        position_in_block[moved] = position as u32;
                    }
                    position_in_block[source] = inside.len() as u32;
                    class[source] = new_id as u32;
                    inside.push(source as u32);
                }
                blocks.push(inside);
                sources.clear();
                affected_members[block_id] = sources;
            } else {
                // The affected side is the larger half. Keep it under the old
                // block id and move the smaller complement instead. `sources`
                // already materializes the whole affected side, so replace the
                // old block with it directly and recycle the old block buffer
                // as this block's predecessor scratch.
                for &source in &sources {
                    affected_mark[source as usize] = true;
                }
                let mut old_block = std::mem::take(&mut blocks[block_id]);
                let mut complement = Vec::with_capacity(old_block.len() - sources.len());
                for &state in &old_block {
                    if !affected_mark[state as usize] {
                        complement.push(state);
                    }
                }
                debug_assert_eq!(complement.len() + sources.len(), old_block.len());
                for (position, &source) in sources.iter().enumerate() {
                    let source = source as usize;
                    debug_assert_eq!(class[source] as usize, block_id);
                    affected_mark[source] = false;
                    position_in_block[source] = position as u32;
                }
                for (position, &state) in complement.iter().enumerate() {
                    let state = state as usize;
                    class[state] = new_id as u32;
                    position_in_block[state] = position as u32;
                }
                blocks[block_id] = sources;
                blocks.push(complement);
                old_block.clear();
                affected_members[block_id] = old_block;
            }

            block_incoming.resize((new_id + 1).saturating_mul(symbols), 0);
            let new_base = new_id * symbols;
            let old_base = block_id * symbols;
            let mut touched = [0u64; 4];
            for &state in &blocks[new_id] {
                let start = target_symbol_offsets[state as usize] as usize;
                let end = target_symbol_offsets[state as usize + 1] as usize;
                for &(symbol, count) in &target_symbols[start..end] {
                    let symbol = symbol as usize;
                    block_incoming[new_base + symbol] += count;
                    touched[symbol / 64] |= 1u64 << (symbol % 64);
                }
            }
            for (word_index, &word) in touched.iter().enumerate() {
                let mut bits = word;
                while bits != 0 {
                    let bit = bits.trailing_zeros() as usize;
                    let symbol = word_index * 64 + bit;
                    block_incoming[old_base + symbol] -= block_incoming[new_base + symbol];
                    bits &= bits - 1;
                }
            }
            queued.push([0u64; 4]);
            affected_members.push(Vec::new());

            for other_symbol in 0..symbols {
                let word = other_symbol / 64;
                let bit = 1u64 << (other_symbol % 64);
                if queued[block_id][word] & bit != 0 {
                    if block_incoming[new_base + other_symbol] != 0 {
                        queue.push_back(((new_id as u64) << 8) | other_symbol as u64);
                        queued[new_id][word] |= bit;
                    }
                } else {
                    let smaller = if blocks[block_id].len() <= blocks[new_id].len() {
                        block_id
                    } else {
                        new_id
                    };
                    if block_incoming[smaller * symbols + other_symbol] != 0
                        && queued[smaller][word] & bit == 0
                    {
                        queue.push_back(((smaller as u64) << 8) | other_symbol as u64);
                        queued[smaller][word] |= bit;
                    }
                }
            }
        }
    }
    let refine_ms = refine_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

    let representatives = blocks
        .iter()
        .map(|members| *members.iter().min().expect("non-empty DFA block") as usize)
        .collect::<Vec<_>>();
    let mut columns = vec![vec![DEAD; representatives.len()]; alphabet.len()];
    for (new_state, &representative) in representatives.iter().enumerate() {
        for &(symbol, target) in &transitions[representative] {
            columns[symbol as usize][new_state] = class[target as usize];
        }
    }
    if profile {
        eprintln!(
            "[glrmask/profile][l1_minimize_symbol_target_csr] states={} edges={} alphabet={} initial_blocks={} final_blocks={} pops={} reverse_ms={:.3} refine_ms={:.3} total_ms={:.3}",
            live,
            transitions.iter().map(Vec::len).sum::<usize>(),
            alphabet.len(),
            block_count,
            representatives.len(),
            pops,
            reverse_ms,
            refine_ms,
            total_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
        );
    }
    Minimized {
        classes: class,
        columns: columns.into_iter().map(Vec::into_boxed_slice).collect(),
        byte_class,
        state_count: representatives.len(),
        rounds: pops,
    }
}

#[derive(Default)]
struct GroupedMinimizeStats {
    local_states: usize,
    local_rounds: usize,
    local_ms: f64,
    local_max_group_ms: f64,
    global_ms: f64,
    dag_groups: usize,
    dag_states: usize,
    hopcroft_groups: usize,
    hopcroft_states: usize,
}

/// Exact two-level minimization for the projected residual automaton.
/// Transitions never change terminal group, so language equivalence can first
/// be computed independently inside each disconnected group. Quotienting those
/// components cannot remove any cross-group equivalence; a second minimization
/// of their much smaller disjoint union performs exactly those remaining
/// cross-group merges.

/// Exact congruence for a deterministic component whose only cycles are
/// literal self-loops.  Removing self-loops yields a DAG, so child behavior is
/// already known in reverse topological order.  Equal structural signatures
/// are safe to merge.  This may intentionally over-distinguish states (for
/// example, a self-loop versus an edge to a language-equivalent descendant);
/// the final global minimizer recovers any such missed equivalences.
fn minimize_dag_with_self_loops(
    transitions: &[Vec<(u8, u32)>],
    states: &[u32],
    local_of_global: &[u32],
) -> Option<(Vec<u32>, Vec<u32>)> {
    let n = states.len();
    let mut indegree = vec![0u32; n];
    let mut outgoing = vec![Vec::<u32>::new(); n];
    for (local, &global) in states.iter().enumerate() {
        for &(_, target) in &transitions[global as usize] {
            if target == global {
                continue;
            }
            let target_local = local_of_global[target as usize];
            debug_assert_ne!(target_local, u32::MAX);
            outgoing[local].push(target_local);
            indegree[target_local as usize] += 1;
        }
    }
    let mut queue = VecDeque::<u32>::new();
    for (state, &degree) in indegree.iter().enumerate() {
        if degree == 0 {
            queue.push_back(state as u32);
        }
    }
    let mut topo = Vec::<u32>::with_capacity(n);
    while let Some(state) = queue.pop_front() {
        topo.push(state);
        for &target in &outgoing[state as usize] {
            indegree[target as usize] -= 1;
            if indegree[target as usize] == 0 {
                queue.push_back(target);
            }
        }
    }
    if topo.len() != n {
        return None;
    }

    // Non-self targets are later in topological order, hence already assigned
    // when processing the order backwards.  u32::MAX denotes SELF and cannot
    // collide with an ordinary local class ID.
    let mut class = vec![u32::MAX; n];
    let mut class_ids = FxHashMap::<Vec<(u8, u32)>, u32>::default();
    let mut representatives = Vec::<u32>::new();
    for &local in topo.iter().rev() {
        let global = states[local as usize];
        let mut signature = Vec::with_capacity(transitions[global as usize].len());
        for &(symbol, target) in &transitions[global as usize] {
            let target_class = if target == global {
                u32::MAX
            } else {
                let target_local = local_of_global[target as usize];
                let target_class = class[target_local as usize];
                debug_assert_ne!(target_class, u32::MAX);
                target_class
            };
            signature.push((symbol, target_class));
        }
        let next = representatives.len() as u32;
        let state_class = *class_ids.entry(signature).or_insert_with(|| {
            representatives.push(global);
            next
        });
        class[local as usize] = state_class;
    }
    Some((class, representatives))
}


/// Exact congruence reduction for a deterministic component by its SCC DAG.
/// Most projected terminal residuals are a large DAG feeding a tiny cyclic core.
/// Singleton SCCs are structurally hashed bottom-up. Nontrivial SCCs are refined
/// only against themselves, treating already-reduced successor SCC classes as
/// fixed observations. The result may over-distinguish across SCC boundaries;
/// the final cross-group minimizer recovers any missed language equivalences.
struct FlatLocalTransitions {
    offsets: Vec<u32>,
    edges: Vec<(u8, u32)>,
}

impl FlatLocalTransitions {
    #[inline]
    fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    #[inline]
    fn row(&self, state: usize) -> &[(u8, u32)] {
        let start = self.offsets[state] as usize;
        let end = self.offsets[state + 1] as usize;
        &self.edges[start..end]
    }

    #[inline]
    fn edge_count(&self) -> usize {
        self.edges.len()
    }
}

#[inline]
fn transition_symbol_mask(row: &[(u8, u32)]) -> [u64; 4] {
    let mut mask = [0u64; 4];
    for &(symbol, _) in row {
        mask[symbol as usize >> 6] |= 1u64 << (symbol & 63);
    }
    mask
}

fn minimize_scc_dag(transitions: &FlatLocalTransitions) -> (Vec<u32>, Vec<u32>, usize, usize) {
    let n = transitions.len();
    if n == 0 {
        return (Vec::new(), Vec::new(), 0, 0);
    }
    let profiling = std::env::var_os("GLRMASK_PROFILE_L1_IMPLEMENTATIONS").is_some();
    let phase_started = profiling.then(Instant::now);

    // Kosaraju, ignoring literal self-loops (they do not affect SCC membership).
    let mut reverse = vec![Vec::<u32>::new(); n];
    for source in 0..n {
        for &(_, target) in transitions.row(source) {
            if target as usize != source {
                reverse[target as usize].push(source as u32);
            }
        }
    }
    let mut seen = vec![false; n];
    let mut order = Vec::<u32>::with_capacity(n);
    for root in 0..n {
        if seen[root] {
            continue;
        }
        seen[root] = true;
        let mut stack = vec![(root as u32, 0usize)];
        while let Some((state, edge_index)) = stack.pop() {
            let row = transitions.row(state as usize);
            let mut next_index = edge_index;
            let mut descended = false;
            while next_index < row.len() {
                let target = row[next_index].1;
                next_index += 1;
                if target == state || seen[target as usize] {
                    continue;
                }
                stack.push((state, next_index));
                seen[target as usize] = true;
                stack.push((target, 0));
                descended = true;
                break;
            }
            if !descended {
                order.push(state);
            }
        }
    }

    seen.fill(false);
    let mut component_of = vec![u32::MAX; n];
    let mut components = Vec::<Vec<u32>>::new();
    for &root in order.iter().rev() {
        if seen[root as usize] {
            continue;
        }
        let component = components.len() as u32;
        seen[root as usize] = true;
        let mut members = Vec::<u32>::new();
        let mut stack = vec![root];
        while let Some(state) = stack.pop() {
            component_of[state as usize] = component;
            members.push(state);
            for &source in &reverse[state as usize] {
                if !seen[source as usize] {
                    seen[source as usize] = true;
                    stack.push(source);
                }
            }
        }
        components.push(members);
    }

    let decompose_ms = phase_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let condensation_started = profiling.then(Instant::now);

    // Process sink SCCs first so every edge leaving the current SCC already has
    // a final reduced class.
    let mut successors = vec![Vec::<u32>::new(); components.len()];
    let mut predecessors = vec![Vec::<u32>::new(); components.len()];
    for source in 0..n {
        let source_component = component_of[source];
        for &(_, target) in transitions.row(source) {
            let target_component = component_of[target as usize];
            if target_component != source_component {
                successors[source_component as usize].push(target_component);
            }
        }
    }
    for edges in &mut successors {
        edges.sort_unstable();
        edges.dedup();
    }
    for (source, edges) in successors.iter().enumerate() {
        for &target in edges {
            predecessors[target as usize].push(source as u32);
        }
    }
    let mut remaining_successors = successors.iter().map(Vec::len).collect::<Vec<_>>();
    let mut ready = VecDeque::<u32>::new();
    for (component, &remaining) in remaining_successors.iter().enumerate() {
        if remaining == 0 {
            ready.push_back(component as u32);
        }
    }

    let condensation_ms = condensation_started.map_or(0.0, |started| {
        started.elapsed().as_secs_f64() * 1000.0
    });
    let solve_started = profiling.then(Instant::now);
    let mut candidate_tests = 0usize;
    let mut candidate_states = 0usize;
    let mut candidate_matches = 0usize;
    let mut max_candidate_bucket = 0usize;

    let mut class = vec![u32::MAX; n];
    let mut representatives = Vec::<u32>::new();
    // Canonical quotient transition row for every solved language class.
    // Exact row lookup handles acyclic states without physical self-loops.
    // For a state with a physical self-loop on symbol b, any equivalent class C
    // must itself transition on b to C; index those candidates by b so the
    // fixed-point test only scans a tiny exact candidate set.
    let mut class_rows = Vec::<Vec<(u8, u32)>>::new();
    let mut row_ids = FxHashMap::<Vec<(u8, u32)>, u32>::default();
    let mut self_classes_by_shape =
        FxHashMap::<(u8, [u64; 4]), Vec<u32>>::default();
    let mut cyclic_components = 0usize;
    let mut cyclic_states = 0usize;
    let mut processed_components = 0usize;

    while let Some(component) = ready.pop_front() {
        processed_components += 1;
        let members = &components[component as usize];
        if members.len() == 1 {
            let state = members[0];
            let source_row = transitions.row(state as usize);

            // A singleton SCC may be language-equivalent to an already-solved
            // descendant. Without a physical self-loop its quotient row is fully
            // known, so exact row interning suffices. With a self-loop, candidate
            // C must semantically self-loop on that same symbol; test only those
            // indexed candidates and interpret physical SELF as C.
            let symbol_mask = transition_symbol_mask(source_row);
            let best_self_symbol = source_row
                .iter()
                .filter_map(|&(symbol, target)| (target == state).then_some(symbol))
                .min_by_key(|&symbol| {
                    self_classes_by_shape
                        .get(&(symbol, symbol_mask))
                        .map_or(0, Vec::len)
                });
            let state_class = if let Some(self_symbol) = best_self_symbol {
                let mut matched = None;
                let candidates = self_classes_by_shape
                    .get(&(self_symbol, symbol_mask))
                    .map(Vec::as_slice)
                    .unwrap_or(&[]);
                candidate_states += 1;
                candidate_tests += candidates.len();
                max_candidate_bucket = max_candidate_bucket.max(candidates.len());
                'candidate: for &candidate in candidates {
                    let candidate_row = &class_rows[candidate as usize];
                    if candidate_row.len() != source_row.len() {
                        continue;
                    }
                    for (&(source_symbol, source_target), &(candidate_symbol, candidate_target)) in
                        source_row.iter().zip(candidate_row)
                    {
                        if source_symbol != candidate_symbol {
                            continue 'candidate;
                        }
                        let source_target_class = if source_target == state {
                            candidate
                        } else {
                            let target_class = class[source_target as usize];
                            debug_assert_ne!(target_class, u32::MAX);
                            target_class
                        };
                        if source_target_class != candidate_target {
                            continue 'candidate;
                        }
                    }
                    matched = Some(candidate);
                    break;
                }
                if let Some(candidate) = matched {
                    candidate_matches += 1;
                    candidate
                } else {
                    let new_class = class_rows.len() as u32;
                    let row = source_row
                        .iter()
                        .map(|&(symbol, target)| {
                            let target_class = if target == state {
                                new_class
                            } else {
                                let target_class = class[target as usize];
                                debug_assert_ne!(target_class, u32::MAX);
                                target_class
                            };
                            (symbol, target_class)
                        })
                        .collect::<Vec<_>>();
                    let row_mask = transition_symbol_mask(&row);
                    for &(symbol, target_class) in &row {
                        if target_class == new_class {
                            self_classes_by_shape
                                .entry((symbol, row_mask))
                                .or_default()
                                .push(new_class);
                        }
                    }
                    row_ids.entry(row.clone()).or_insert(new_class);
                    class_rows.push(row);
                    representatives.push(state);
                    new_class
                }
            } else {
                let row = source_row
                    .iter()
                    .map(|&(symbol, target)| {
                        let target_class = class[target as usize];
                        debug_assert_ne!(target_class, u32::MAX);
                        (symbol, target_class)
                    })
                    .collect::<Vec<_>>();
                if let Some(&existing) = row_ids.get(&row) {
                    existing
                } else {
                    let new_class = class_rows.len() as u32;
                    row_ids.insert(row.clone(), new_class);
                    class_rows.push(row);
                    representatives.push(state);
                    new_class
                }
            };
            class[state as usize] = state_class;
        } else {
            cyclic_components += 1;
            cyclic_states += members.len();
            let mut member_index = vec![u32::MAX; n];
            for (index, &state) in members.iter().enumerate() {
                member_index[state as usize] = index as u32;
            }

            // Standard deterministic partition refinement only inside this tiny
            // SCC. External successor classes are fixed observations.
            let mut local_classes = vec![0u32; members.len()];
            loop {
                let mut ids = FxHashMap::<Vec<(u8, u64)>, u32>::default();
                let mut next_classes = vec![0u32; members.len()];
                for (local, &state) in members.iter().enumerate() {
                    let mut signature =
                        Vec::<(u8, u64)>::with_capacity(transitions.row(state as usize).len());
                    for &(symbol, target) in transitions.row(state as usize) {
                        let target_local = member_index[target as usize];
                        let behavior = if target_local != u32::MAX {
                            (1u64 << 63) | u64::from(local_classes[target_local as usize])
                        } else {
                            let target_class = class[target as usize];
                            debug_assert_ne!(target_class, u32::MAX);
                            u64::from(target_class)
                        };
                        signature.push((symbol, behavior));
                    }
                    let next = ids.len() as u32;
                    next_classes[local] = *ids.entry(signature).or_insert(next);
                }
                if next_classes == local_classes {
                    break;
                }
                local_classes = next_classes;
            }
            let local_count = local_classes
                .iter()
                .copied()
                .max()
                .map_or(0usize, |value| value as usize + 1);
            let base = class_rows.len() as u32;
            let mut local_representatives = vec![u32::MAX; local_count];
            for (local, &state) in members.iter().enumerate() {
                let local_class = local_classes[local] as usize;
                class[state as usize] = base + local_class as u32;
                if local_representatives[local_class] == u32::MAX {
                    local_representatives[local_class] = state;
                }
            }
            for (local_class, &representative) in local_representatives.iter().enumerate() {
                let global_class = base + local_class as u32;
                let row = transitions
                    .row(representative as usize)
                    .iter()
                    .map(|&(symbol, target)| {
                        let target_class = class[target as usize];
                        debug_assert_ne!(target_class, u32::MAX);
                        (symbol, target_class)
                    })
                    .collect::<Vec<_>>();
                let row_mask = transition_symbol_mask(&row);
                for &(symbol, target_class) in &row {
                    if target_class == global_class {
                        self_classes_by_shape
                            .entry((symbol, row_mask))
                            .or_default()
                            .push(global_class);
                    }
                }
                row_ids.entry(row.clone()).or_insert(global_class);
                class_rows.push(row);
                representatives.push(representative);
            }
        }

        for &predecessor in &predecessors[component as usize] {
            let remaining = &mut remaining_successors[predecessor as usize];
            *remaining -= 1;
            if *remaining == 0 {
                ready.push_back(predecessor);
            }
        }
    }
    debug_assert_eq!(processed_components, components.len());
    debug_assert!(class.iter().all(|&value| value != u32::MAX));
    debug_assert_eq!(class_rows.len(), representatives.len());
    if profiling {
        let solve_ms = solve_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        eprintln!(
            "[glrmask/profile][l1_projected_scc_reduce] states={} edges={} classes={} sccs={} cyclic_sccs={} cyclic_states={} decompose_ms={:.3} condensation_ms={:.3} solve_ms={:.3} candidate_states={} candidate_tests={} candidate_matches={} max_candidate_bucket={}",
            n,
            transitions.edge_count(),
            representatives.len(),
            components.len(),
            cyclic_components,
            cyclic_states,
            decompose_ms,
            condensation_ms,
            solve_ms,
            candidate_states,
            candidate_tests,
            candidate_matches,
            max_candidate_bucket,
        );
    }
    (class, representatives, cyclic_components, cyclic_states)
}

fn minimize_grouped_local(
    transitions: &[Vec<(u8, u32)>],
    groups: &[u32],
    alphabet: &[u8],
) -> (Minimized, GroupedMinimizeStats) {
    debug_assert_eq!(transitions.len(), groups.len());
    let group_count = groups
        .iter()
        .copied()
        .max()
        .map_or(0usize, |group| group as usize + 1);
    let mut states_by_group = vec![Vec::<u32>::new(); group_count];
    for (state, &group) in groups.iter().enumerate() {
        states_by_group[group as usize].push(state as u32);
    }

    let local_started = Instant::now();
    let mut reduced_of_state = vec![u32::MAX; transitions.len()];
    let mut local_of_global = vec![u32::MAX; transitions.len()];
    let mut reduced_representatives = Vec::<u32>::new();
    let mut local_rounds = 0usize;
    let mut dag_groups = 0usize;
    let mut dag_states = 0usize;
    let mut hopcroft_groups = 0usize;
    let mut hopcroft_states = 0usize;
    let mut local_max_group_ms = 0.0f64;
    let dag_enabled = std::env::var("GLRMASK_L1_PROJECTED_DAG_MINIMIZE")
        .map(|value| {
            let value = value.trim();
            value.is_empty() || (value != "0" && !value.eq_ignore_ascii_case("false"))
        })
        .unwrap_or(true);
    let projected_edge_count = transitions.iter().map(Vec::len).sum::<usize>();
    let use_scc = std::env::var("GLRMASK_L1_PROJECTED_SCC_MINIMIZE")
        .map(|value| {
            let value = value.trim();
            value.is_empty() || (value != "0" && !value.eq_ignore_ascii_case("false"))
        })
        .unwrap_or_else(|_| {
            // Dense projected residual graphs are the regime where the local
            // components are large DAGs feeding tiny SCC cores.  SCC-DAG
            // reduction avoids repeated global refinement there; on sparse
            // residual graphs ordinary grouped minimization has lower overhead.
            transitions.len() >= 5_000
                && projected_edge_count >= transitions.len().saturating_mul(10)
        });

    for (group_id, states) in states_by_group
        .iter()
        .enumerate()
        .filter(|(_, states)| !states.is_empty())
    {
        let group_started = Instant::now();
        for (local, &global) in states.iter().enumerate() {
            local_of_global[global as usize] = local as u32;
        }
        // Large projected terminal components in the p90 shapes are almost
        // DAGs but contain one tiny SCC. Probing them with a full DAG walk and
        // then rebuilding the same graph for SCC reduction doubles the edge
        // traffic. Both reducers are exact, so route large components directly
        // to SCC when that kernel is enabled; retain the cheaper DAG probe for
        // small components.
        let direct_scc = use_scc && states.len() >= 128;
        let dag_result = (!direct_scc && dag_enabled)
            .then(|| minimize_dag_with_self_loops(transitions, states, &local_of_global))
            .flatten();
        let (local_classes, representatives) = if let Some(result) = dag_result {
            dag_groups += 1;
            dag_states += states.len();
            result
        } else {
            hopcroft_groups += 1;
            hopcroft_states += states.len();
            let local_classes = if use_scc {
                let edge_capacity = states
                    .iter()
                    .map(|&global| transitions[global as usize].len())
                    .sum::<usize>();
                let mut offsets = Vec::with_capacity(states.len() + 1);
                let mut edges = Vec::with_capacity(edge_capacity);
                offsets.push(0);
                for &global in states {
                    for &(symbol, target) in &transitions[global as usize] {
                        debug_assert_eq!(groups[target as usize], groups[global as usize]);
                        let local_target = local_of_global[target as usize];
                        debug_assert_ne!(local_target, u32::MAX);
                        edges.push((symbol, local_target));
                    }
                    offsets.push(edges.len() as u32);
                }
                let local_transitions = FlatLocalTransitions { offsets, edges };
                let (classes, _representatives, cyclic_components, cyclic_states) =
                    minimize_scc_dag(&local_transitions);
                if std::env::var_os("GLRMASK_PROFILE_L1_IMPLEMENTATIONS").is_some() {
                    eprintln!(
                        "[glrmask/profile][l1_projected_group_scc] group={} states={} edges={} cyclic_sccs={} cyclic_states={} mode=scc_dag",
                        group_id,
                        states.len(),
                        local_transitions.edge_count(),
                        cyclic_components,
                        cyclic_states,
                    );
                }
                classes
            } else {
                let mut local_transitions = Vec::with_capacity(states.len());
                for &global in states {
                    let mut row = Vec::with_capacity(transitions[global as usize].len());
                    for &(symbol, target) in &transitions[global as usize] {
                        debug_assert_eq!(groups[target as usize], groups[global as usize]);
                        let local_target = local_of_global[target as usize];
                        debug_assert_ne!(local_target, u32::MAX);
                        row.push((symbol, local_target));
                    }
                    local_transitions.push(row);
                }
                let local_minimized = minimize(&local_transitions, alphabet);
                local_rounds += local_minimized.rounds;
                local_minimized.classes
            };
            let state_count = local_classes
                .iter()
                .copied()
                .max()
                .map_or(0usize, |class| class as usize + 1);
            let mut representatives = vec![u32::MAX; state_count];
            for (local, &global) in states.iter().enumerate() {
                let class = local_classes[local] as usize;
                if representatives[class] == u32::MAX {
                    representatives[class] = global;
                }
            }
            (local_classes, representatives)
        };

        let base = reduced_representatives.len() as u32;
        for (local, &global) in states.iter().enumerate() {
            reduced_of_state[global as usize] = base + local_classes[local];
        }
        reduced_representatives.extend(representatives);
        for &global in states {
            local_of_global[global as usize] = u32::MAX;
        }
        local_max_group_ms =
            local_max_group_ms.max(group_started.elapsed().as_secs_f64() * 1000.0);
    }
    let local_ms = local_started.elapsed().as_secs_f64() * 1000.0;

    let global_started = Instant::now();
    let mut reduced_transitions = Vec::with_capacity(reduced_representatives.len());
    for &representative in &reduced_representatives {
        reduced_transitions.push(
            transitions[representative as usize]
                .iter()
                .map(|&(symbol, target)| (symbol, reduced_of_state[target as usize]))
                .collect::<Vec<_>>(),
        );
    }
    let reduced = minimize(&reduced_transitions, alphabet);
    let global_ms = global_started.elapsed().as_secs_f64() * 1000.0;
    let classes = reduced_of_state
        .into_iter()
        .map(|local_class| reduced.classes[local_class as usize])
        .collect::<Vec<_>>();
    (
        Minimized {
            classes,
            columns: reduced.columns,
            byte_class: reduced.byte_class,
            state_count: reduced.state_count,
            rounds: local_rounds + reduced.rounds,
        },
        GroupedMinimizeStats {
            local_states: reduced_representatives.len(),
            local_rounds,
            local_ms,
            local_max_group_ms,
            global_ms,
            dag_groups,
            dag_states,
            hopcroft_groups,
            hopcroft_states,
        },
    )
}

fn minimize_grouped(
    transitions: &[Vec<(u8, u32)>],
    groups: &[u32],
    alphabet: &[u8],
) -> (Minimized, GroupedMinimizeStats) {
    let seeded = std::env::var("GLRMASK_L1_PROJECTED_GROUPED_SEEDED")
        .ok()
        .is_some_and(|value| {
            let value = value.trim();
            value.is_empty() || (value != "0" && !value.eq_ignore_ascii_case("false"))
        });
    if seeded {
        minimize_grouped_seeded(transitions, groups, alphabet)
    } else {
        minimize_grouped_local(transitions, groups, alphabet)
    }
}

fn minimize_grouped_seeded(
    transitions: &[Vec<(u8, u32)>],
    groups: &[u32],
    alphabet: &[u8],
) -> (Minimized, GroupedMinimizeStats) {
    debug_assert_eq!(transitions.len(), groups.len());
    let local_started = Instant::now();
    let local = minimize_seeded(transitions, alphabet, Some(groups));
    let local_ms = local_started.elapsed().as_secs_f64() * 1000.0;

    let mut representatives = vec![u32::MAX; local.state_count];
    for (state, &class) in local.classes.iter().enumerate() {
        if representatives[class as usize] == u32::MAX {
            representatives[class as usize] = state as u32;
        }
    }
    debug_assert!(representatives.iter().all(|&state| state != u32::MAX));

    let global_started = Instant::now();
    let mut reduced_transitions = Vec::with_capacity(representatives.len());
    for &representative in &representatives {
        reduced_transitions.push(
            transitions[representative as usize]
                .iter()
                .map(|&(symbol, target)| (symbol, local.classes[target as usize]))
                .collect::<Vec<_>>(),
        );
    }
    let reduced = minimize(&reduced_transitions, alphabet);
    let global_ms = global_started.elapsed().as_secs_f64() * 1000.0;
    let classes = local
        .classes
        .iter()
        .map(|&local_class| reduced.classes[local_class as usize])
        .collect::<Vec<_>>();
    (
        Minimized {
            classes,
            columns: reduced.columns,
            byte_class: reduced.byte_class,
            state_count: reduced.state_count,
            rounds: local.rounds + reduced.rounds,
        },
        GroupedMinimizeStats {
            local_states: representatives.len(),
            local_rounds: local.rounds,
            local_ms,
            local_max_group_ms: local_ms,
            global_ms,
            dag_groups: 0,
            dag_states: 0,
            hopcroft_groups: 0,
            hopcroft_states: 0,
        },
    )
}

fn unique_vocab(input: BuildInput<'_>) -> (Vec<Vec<u32>>, Vec<Arc<[u8]>>) {
    let mut tokens = Vec::<Arc<[u8]>>::with_capacity(input.vocab.len());
    let mut aliases = Vec::<Vec<u32>>::with_capacity(input.vocab.len());

    let mut push = |id: u32, bytes: &Arc<[u8]>| {
        if tokens
            .last()
            .is_some_and(|token| token.as_ref() == bytes.as_ref())
        {
            aliases
                .last_mut()
                .expect("duplicate token has predecessor")
                .push(id);
        } else {
            tokens.push(Arc::clone(bytes));
            aliases.push(vec![id]);
        }
    };

    if let Some(parent) = input.subset_parent_order {
        // Split-L1 is a subset of an already sorted parent vocabulary. Filtering
        // that order is sufficient; do not construct another L1IdentityVocabOrder
        // (and especially do not build its packed-bucket metadata).
        let mut included = vec![false; parent.original_to_internal.len()];
        for (id, _) in input.vocab.iter() {
            included[id as usize] = true;
        }
        for (id, bytes) in parent.token_entries_sorted.iter() {
            if included[*id as usize] {
                push(*id, bytes);
            }
        }
    } else {
        let order = super::super::prepared_l1_identity_vocab_order(input.vocab);
        for (id, bytes) in order.token_entries_sorted.iter() {
            push(*id, bytes);
        }
    }
    debug_assert_eq!(
        aliases.iter().map(Vec::len).sum::<usize>(),
        input.vocab.len()
    );
    (aliases, tokens)
}

/// Quotient vocabulary bytes by their exact raw tokenizer transition column.
/// Equality here is stronger than needed but cheap to prove: identical direct
/// transitions from every raw state imply identical `step_all` results from
/// every epsilon-closed configuration.
fn quotient_input_bytes(input: BuildInput<'_>, bytes: &[u8]) -> (Vec<u8>, [u8; 256]) {
    let states = input.tokenizer.num_states() as usize;
    let mut column_ids = FxHashMap::<Vec<u32>, u8>::default();
    let mut representatives = Vec::<u8>::new();
    let mut representative = [0u8; 256];
    for &byte in bytes {
        let column = if let Some(by_byte) = input.transitions_by_byte {
            by_byte[byte as usize * states..(byte as usize + 1) * states].to_vec()
        } else {
            (0..states)
                .map(|state| input.flat_trans[state * 256 + byte as usize])
                .collect::<Vec<_>>()
        };
        let class = if let Some(&class) = column_ids.get(&column) {
            class
        } else {
            let class = representatives.len() as u8;
            column_ids.insert(column, class);
            representatives.push(byte);
            class
        };
        representative[byte as usize] = representatives[class as usize];
    }
    (representatives, representative)
}

/// Exact finite-vocabulary base case for one-byte partitions.
///
/// This is the same single-group semantics without constructing the closure of
/// the residual language beyond the only byte each token can consume. It is a
/// base case of the algorithm, not a fallback to the production builder.
fn build_one_byte(
    input: BuildInput<'_>,
    aliases: &[Vec<u32>],
    tokens: &[Arc<[u8]>],
    total: Instant,
) -> Option<LocalIdMapTerminalDwa> {
    let scan = Instant::now();
    let mut scanner = Scanner::new(input);
    let mut raw_to_start = Vec::with_capacity(input.tokenizer.num_states() as usize);
    for raw in 0..input.tokenizer.num_states() {
        raw_to_start.push(scanner.start(raw));
    }
    let mut raw_by_start = vec![Vec::<u32>::new(); scanner.configs.len()];
    for (raw, &start) in raw_to_start.iter().enumerate() {
        raw_by_start[start as usize].push(raw as u32);
    }

    let token_bytes = tokens.iter().map(|token| token[0]).collect::<Vec<_>>();
    let (byte_representatives, byte_representative) = quotient_input_bytes(input, &token_bytes);
    let mut representative_index = [usize::MAX; 256];
    for (index, &byte) in byte_representatives.iter().enumerate() {
        representative_index[byte as usize] = index;
    }

    let mut state_class = vec![0u32; input.tokenizer.num_states() as usize];
    let mut row_ids = FxHashMap::<Vec<u32>, u32>::default();
    let mut rows = Vec::<Vec<u32>>::new();
    for (start, raw_states) in raw_by_start.iter().enumerate() {
        if raw_states.is_empty() {
            continue;
        }
        let singleton = (scanner.configs[start].len() == 1).then_some(scanner.configs[start][0]);
        let representative_signatures = byte_representatives
            .iter()
            .map(|&byte| {
                if let Some(state) = singleton {
                    let target = input.flat_trans[state as usize * 256 + byte as usize];
                    if target == DEAD {
                        0
                    } else {
                        scanner.signature(raw_to_start[target as usize])
                    }
                } else {
                    let target = scanner.step(start as u32, byte);
                    scanner.signature(target)
                }
            })
            .collect::<Vec<_>>();
        let row = token_bytes
            .iter()
            .map(|&byte| {
                let representative = byte_representative[byte as usize];
                representative_signatures[representative_index[representative as usize]]
            })
            .collect::<Vec<_>>();
        let next = rows.len() as u32;
        let class = *row_ids.entry(row.clone()).or_insert_with(|| {
            rows.push(row);
            next
        });
        for &raw in raw_states {
            state_class[raw as usize] = class;
        }
    }
    let scan_ms = scan.elapsed().as_secs_f64() * 1000.0;
    let finished = common::finish(
        input,
        aliases,
        &scanner.signatures,
        state_class,
        rows,
        scan_ms,
        || total.elapsed().as_secs_f64() * 1000.0,
    )?;
    if std::env::var_os("GLRMASK_PROFILE_L1_IMPLEMENTATIONS").is_some() {
        eprintln!(
            "[glrmask/profile][l1_single_one_byte] partition={} raw_states={} tokens={} byte_classes={} configs={} signatures={} state_classes={} token_classes={} scan_ms={:.3} compact_ms={:.3} build_ms={:.3} total_ms={:.3}",
            input.partition_label,
            input.tokenizer.num_states(),
            tokens.len(),
            byte_representatives.len(),
            scanner.configs.len(),
            scanner.signatures.len(),
            finished.state_classes,
            finished.token_classes,
            scan_ms,
            finished.compact_ms,
            finished.build_ms,
            total.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Some(finished.artifact)
}

/// Deterministic reverse subset construction for the all-live DFA.
///
/// For a byte string `x`, a subset state denotes
/// `A_x = { q | delta(q, x) is live }`. The empty suffix starts at every live
/// state, and prepending `b` is the exact preimage
/// `A_{bx} = { q | delta(q, b) in A_x }`. Walking a token backwards therefore
/// yields its complete acceptance column over every minimized residual state.
struct ReverseColumn {
    offsets: Box<[u32]>,
    sources: Box<[u32]>,
    live_sources: Box<[u64]>,
    live_targets: Box<[u64]>,
    live_target_count: usize,
}

struct ReverseSubsets<'a> {
    columns: Vec<ReverseColumn>,
    forward_columns: &'a [Box<[u32]>],
    byte_class: &'a [u8; 256],
    state_count: usize,
    sets: Vec<Arc<[u64]>>,
    // The subset itself can be tens of kilobytes. Hash it exactly once, then
    // key the table by that compact fingerprint. Any fingerprint collision is
    // resolved by exact slice comparison against `sets`, so this is still a
    // fully exact interner rather than a probabilistic shortcut.
    ids: FxHashMap<u64, u32>,
    id_collisions: FxHashMap<u64, Vec<u32>>,
    cache: Vec<u32>,
    symbol_count: usize,
    computed_transitions: usize,
    target_visits: usize,
    predecessor_visits: usize,
    force_source_scan: bool,
}

impl<'a> ReverseSubsets<'a> {
    fn new(
        columns: &'a [Box<[u32]>],
        byte_class: &'a [u8; 256],
        state_count: usize,
        force_source_scan: bool,
    ) -> Self {
        let words = state_count.div_ceil(64);
        let mut all = vec![u64::MAX; words];
        if let Some(last) = all.last_mut() {
            let remainder = state_count % 64;
            if remainder != 0 {
                *last = (1u64 << remainder) - 1;
            }
        }
        let build_column = |column: &Box<[u32]>| {
                let mut counts = vec![0u32; state_count];
                let mut live_sources = vec![0u64; words];
                let mut live_targets = vec![0u64; words];
                for (source, &target) in column.iter().enumerate() {
                    if target != DEAD {
                        counts[target as usize] += 1;
                        live_sources[source / 64] |= 1u64 << (source % 64);
                        live_targets[target as usize / 64] |= 1u64 << (target as usize % 64);
                    }
                }
                let live_target_count = live_targets
                    .iter()
                    .map(|word| word.count_ones() as usize)
                    .sum();
                let mut offsets = vec![0u32; state_count + 1];
                for target in 0..state_count {
                    offsets[target + 1] = offsets[target] + counts[target];
                }
                let mut next = offsets[..state_count].to_vec();
                let mut sources = vec![0u32; offsets[state_count] as usize];
                for (source, &target) in column.iter().enumerate() {
                    if target != DEAD {
                        let slot = &mut next[target as usize];
                        sources[*slot as usize] = source as u32;
                        *slot += 1;
                    }
                }
                ReverseColumn {
                    offsets: offsets.into_boxed_slice(),
                    sources: sources.into_boxed_slice(),
                    live_sources: live_sources.into_boxed_slice(),
                    live_targets: live_targets.into_boxed_slice(),
                    live_target_count,
                }
            };
        let reverse_columns: Vec<ReverseColumn> = columns.iter().map(build_column).collect();
        let symbol_count = reverse_columns.len();
        let mut result = Self {
            columns: reverse_columns,
            forward_columns: columns,
            byte_class,
            state_count,
            sets: Vec::new(),
            ids: FxHashMap::default(),
            id_collisions: FxHashMap::default(),
            cache: Vec::new(),
            symbol_count,
            computed_transitions: 0,
            target_visits: 0,
            predecessor_visits: 0,
            force_source_scan,
        };
        let start = result.intern(all);
        debug_assert_eq!(start, 0);
        result
    }

    #[inline]
    fn subset_hash(set: &[u64]) -> u64 {
        let mut hasher = FxHasher::default();
        set.hash(&mut hasher);
        hasher.finish()
    }

    fn intern(&mut self, set: Vec<u64>) -> u32 {
        let hash = Self::subset_hash(&set);
        self.intern_hashed(set, hash)
    }

    fn intern_hashed(&mut self, set: Vec<u64>, hash: u64) -> u32 {
        if let Some(&id) = self.ids.get(&hash) {
            if self.sets[id as usize].as_ref() == set.as_slice() {
                return id;
            }
            if let Some(collisions) = self.id_collisions.get(&hash) {
                for &collision_id in collisions {
                    if self.sets[collision_id as usize].as_ref() == set.as_slice() {
                        return collision_id;
                    }
                }
            }
        }
        let id = self.sets.len() as u32;
        let set: Arc<[u64]> = Arc::from(set.into_boxed_slice());
        if self.ids.contains_key(&hash) {
            self.id_collisions.entry(hash).or_default().push(id);
        } else {
            self.ids.insert(hash, id);
        }
        self.sets.push(set);
        self.cache
            .resize(self.cache.len() + self.symbol_count, UNKNOWN);
        id
    }

    #[inline]
    fn contains(set: &[u64], state: u32) -> bool {
        set[state as usize / 64] & (1u64 << (state as usize % 64)) != 0
    }

    fn visit_targets(set: &[u64], state_count: usize, included: bool, mut visit: impl FnMut(usize)) {
        for (word_index, &set_word) in set.iter().enumerate() {
            let mut word = if included { set_word } else { !set_word };
            if word_index + 1 == set.len() && !state_count.is_multiple_of(64) {
                word &= (1u64 << (state_count % 64)) - 1;
            }
            while word != 0 {
                let bit = word.trailing_zeros() as usize;
                visit(word_index * 64 + bit);
                word &= word - 1;
            }
        }
    }

    fn prepend(&mut self, suffix: u32, byte: u8) -> u32 {
        let symbol = self.byte_class[byte as usize] as usize;
        self.prepend_symbol(suffix, symbol)
    }

    fn prepend_symbol(&mut self, suffix: u32, symbol: usize) -> u32 {
        let cache_index = suffix as usize * self.symbol_count + symbol;
        let cached = self.cache[cache_index];
        if cached != UNKNOWN {
            return cached;
        }
        let suffix_set = &self.sets[suffix as usize];
        let column = &self.columns[symbol];
        if self.force_source_scan {
            let forward = &self.forward_columns[symbol];
            let mut predecessor = vec![0u64; suffix_set.len()];
            let mut source_visits = 0usize;
            for (word_index, &live_word) in column.live_sources.iter().enumerate() {
                let mut word = live_word;
                while word != 0 {
                    let bit = word.trailing_zeros() as usize;
                    let source = word_index * 64 + bit;
                    let target = forward[source];
                    debug_assert_ne!(target, DEAD);
                    source_visits += 1;
                    if Self::contains(suffix_set, target) {
                        predecessor[source / 64] |= 1u64 << (source % 64);
                    }
                    word &= word - 1;
                }
            }
            self.computed_transitions += 1;
            self.predecessor_visits += source_visits;
            let target = self.intern(predecessor);
            self.cache[cache_index] = target;
            return target;
        }
        // Only targets with at least one predecessor in this symbol column can
        // affect the preimage. Count the suffix intersection with that sparse
        // live-target set, then enumerate whichever side of the live targets is
        // smaller. This preserves the exact preimage while avoiding visits to
        // dead target rows and avoids choosing the wrong complement based on
        // unrelated states that have no predecessor for this symbol.
        let included_targets = suffix_set
            .iter()
            .zip(column.live_targets.iter())
            .map(|(&set, &live)| (set & live).count_ones() as usize)
            .sum::<usize>();
        let use_included = included_targets <= column.live_target_count - included_targets;
        let mut predecessor = if use_included {
            vec![0u64; suffix_set.len()]
        } else {
            column.live_sources.to_vec()
        };
        let mut target_visits = 0usize;
        let mut predecessor_visits = 0usize;
        for (word_index, (&set_word, &live_word)) in suffix_set
            .iter()
            .zip(column.live_targets.iter())
            .enumerate()
        {
            let mut selected = if use_included {
                set_word & live_word
            } else {
                !set_word & live_word
            };
            while selected != 0 {
                let bit_index = selected.trailing_zeros() as usize;
                let target = word_index * 64 + bit_index;
                target_visits += 1;
                let start = column.offsets[target] as usize;
                let end = column.offsets[target + 1] as usize;
                predecessor_visits += end - start;
                for &source in &column.sources[start..end] {
                    let word = &mut predecessor[source as usize / 64];
                    let bit = 1u64 << (source as usize % 64);
                    if use_included {
                        *word |= bit;
                    } else {
                        *word &= !bit;
                    }
                }
                selected &= selected - 1;
            }
        }
        self.computed_transitions += 1;
        self.target_visits += target_visits;
        self.predecessor_visits += predecessor_visits;
        let target = self.intern(predecessor);
        self.cache[cache_index] = target;
        target
    }

    fn token(&mut self, token: &[u8]) -> u32 {
        token
            .iter()
            .rev()
            .fold(0, |suffix, &byte| self.prepend(suffix, byte))
    }

    /// Complete the reverse subset machine over every minimized input-byte
    /// class and freeze it as one dense row-major table.  Median projected-L1
    /// shapes have only tens to low hundreds of reverse states; once frozen,
    /// vocabulary classification becomes plain array indexing with no hash
    /// interning or lazy-transition branches in the token loop.
    fn try_complete_dense(&mut self, max_states: usize) -> Option<Box<[u32]>> {
        let symbols = self.columns.len();
        let mut state = 0usize;
        while state < self.sets.len() {
            if self.sets.len() > max_states {
                return None;
            }
            for symbol in 0..symbols {
                self.prepend_symbol(state as u32, symbol);
                if self.sets.len() > max_states {
                    return None;
                }
            }
            state += 1;
        }
        debug_assert_eq!(symbols, self.symbol_count);
        debug_assert!(self.cache.iter().all(|&target| target != UNKNOWN));
        Some(self.cache.clone().into_boxed_slice())
    }
}

#[inline(always)]
fn intern_dense_subset_class(
    subset: u32,
    subset_to_class: &mut Vec<u32>,
    class_subsets: &mut Vec<u32>,
) -> u32 {
    let index = subset as usize;
    if index >= subset_to_class.len() {
        subset_to_class.resize(index + 1, u32::MAX);
    }
    let class = subset_to_class[index];
    if class != u32::MAX {
        return class;
    }
    let class = class_subsets.len() as u32;
    subset_to_class[index] = class;
    class_subsets.push(subset);
    class
}

const PROJECTED_CELL_BUDGET: usize = 4_000_000;
const PROJECTED_SUBSET_CELL_BUDGET: usize = 1_500_000;

fn projected_cell_budget(input: BuildInput<'_>) -> usize {
    let (env_name, default) = if input.subset_parent_order.is_some() {
        (
            "GLRMASK_L1_SINGLE_PROJECTED_SUBSET_CELL_BUDGET",
            PROJECTED_SUBSET_CELL_BUDGET,
        )
    } else {
        (
            "GLRMASK_L1_SINGLE_PROJECTED_CELL_BUDGET",
            PROJECTED_CELL_BUDGET,
        )
    };
    std::env::var(env_name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn projected_shape(input: BuildInput<'_>) -> (usize, usize) {
    let mut relevant = [false; 256];
    for (_, token) in input.vocab.iter() {
        for &byte in token {
            relevant[byte as usize] = true;
        }
    }
    let vocab_bytes = relevant.iter().filter(|&&used| used).count();
    let memberships = (0..input.tokenizer.num_states())
        .map(|state| {
            super::super::collect_active_terminal_signature(
                input.tokenizer,
                state,
                input.active_terminals,
            )
            .len()
        })
        .sum();
    (memberships, vocab_bytes)
}

/// Return whether the legacy experimental auto-selector would choose projected.
/// Production does not call this: projected is the unconditional default.
pub(super) fn should_use_projected(input: BuildInput<'_>) -> bool {
    let generic_epsilon = input.tokenizer.has_epsilon_transitions()
        && !input.tokenizer.has_scalar_deterministic_dispatch();
    if !generic_epsilon {
        return true;
    }
    let (memberships, vocab_bytes) = projected_shape(input);
    memberships.saturating_mul(vocab_bytes) <= projected_cell_budget(input)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProjectedKernel {
    Finite,
    Residual,
}

fn projected_kernel(input: BuildInput<'_>) -> ProjectedKernel {
    if std::env::var("GLRMASK_L1_PROJECTED_RESIDUAL_PARTITIONS")
        .ok()
        .is_some_and(|scope| {
            scope
                .split(',')
                .map(str::trim)
                .any(|partition| partition == input.partition_label)
        })
    {
        return ProjectedKernel::Residual;
    }
    if std::env::var("GLRMASK_L1_PROJECTED_FINITE_PARTITIONS")
        .ok()
        .is_some_and(|scope| {
            scope
                .split(',')
                .map(str::trim)
                .any(|partition| partition == input.partition_label)
        })
    {
        return ProjectedKernel::Finite;
    }
    let p2_finite_state_limit = std::env::var("GLRMASK_P2_FINITE_MAX_TOKENIZER_STATES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        // Residual p2 became substantially cheaper after the dense-reverse,
        // reverse-trie, and prepared-vocabulary work.  Revalidation on the
        // current canonical corpus shows finite still wins for the tiniest
        // lexers, but the old <=160 crossover is stale; above ~40 states the
        // residual kernel is now overwhelmingly faster.  Keep the env override
        // for exact diagnostics and older-machine experiments.
        .unwrap_or(40);
    let p2_finite = input.partition_label == "p2"
        && input.subset_parent_order.is_none()
        && input.tokenizer.num_states() <= p2_finite_state_limit
        && std::env::var("GLRMASK_P2_FINITE_PROJECTED")
            .map(|value| {
                let value = value.trim();
                value.is_empty() || (value != "0" && !value.eq_ignore_ascii_case("false"))
            })
            .unwrap_or(true);
    if p2_finite {
        return ProjectedKernel::Finite;
    }
    if let Ok(value) = std::env::var("GLRMASK_L1_PROJECTED_KERNEL") {
        return match value.trim().to_ascii_lowercase().as_str() {
            "finite" | "vocab" | "trie" => ProjectedKernel::Finite,
            "residual" | "dfa" => ProjectedKernel::Residual,
            "" | "auto" => projected_kernel_auto(input),
            other => panic!(
                "unknown GLRMASK_L1_PROJECTED_KERNEL={other:?}; expected auto, finite, or residual"
            ),
        };
    }
    projected_kernel_auto(input)
}


fn profile_vector_projection_shape(input: BuildInput<'_>) {
    let Ok(scope) = std::env::var("GLRMASK_L1_PROJECTED_VECTOR_PROFILE") else {
        return;
    };
    let scope = scope.trim();
    if !scope.is_empty() && scope != "1" && scope != input.partition_label {
        return;
    }
    let started = Instant::now();
    let mut scanner = Scanner::new(input);
    let mut raw_roots = Vec::with_capacity(input.tokenizer.num_states() as usize);
    for raw in 0..input.tokenizer.num_states() {
        raw_roots.push(scanner.start(raw));
    }
    let mut vocab_bytes = input.vocab.relevant_bytes().iter().copied().collect::<Vec<_>>();
    vocab_bytes.sort_unstable();
    let (bytes, _) = quotient_input_bytes(input, &vocab_bytes);
    let mut transitions = Vec::<Vec<(u8, u32)>>::new();
    let mut state = 0usize;
    while state < scanner.configs.len() {
        let mut row = Vec::new();
        for (symbol, &byte) in bytes.iter().enumerate() {
            let target = scanner.step(state as u32, byte);
            if target != DEAD {
                row.push((symbol as u8, target));
            }
        }
        transitions.push(row);
        state += 1;
    }
    let build_ms = started.elapsed().as_secs_f64() * 1000.0;
    let outputs = (0..scanner.configs.len())
        .map(|state| scanner.signature(state as u32))
        .collect::<Vec<_>>();
    let minimize_started = Instant::now();
    let minimized = minimize_seeded(&transitions, &bytes, Some(&outputs));
    let minimize_ms = minimize_started.elapsed().as_secs_f64() * 1000.0;
    let mut root_classes = raw_roots
        .iter()
        .map(|&root| minimized.classes[root as usize])
        .collect::<Vec<_>>();
    root_classes.sort_unstable();
    root_classes.dedup();
    eprintln!(
        "[glrmask/profile][l1_projected_vector] partition={} configs={} edges={} output_signatures={} minimized_states={} raw_root_classes={} build_ms={:.3} minimize_ms={:.3} total_ms={:.3}",
        input.partition_label,
        transitions.len(),
        transitions.iter().map(Vec::len).sum::<usize>(),
        scanner.signatures.len(),
        minimized.state_count,
        root_classes.len(),
        build_ms,
        minimize_ms,
        started.elapsed().as_secs_f64() * 1000.0,
    );
}

fn projected_kernel_auto(input: BuildInput<'_>) -> ProjectedKernel {
    // p1 is best treated as one adaptive projected algorithm rather than two
    // externally selected kernels. Start from residual projection; its cheap
    // root-membership precheck switches to finite projection before residual
    // state growth, and finite has a bounded-work escape back to residual when
    // the vocabulary trie is the pathological side. Keeping the decision here
    // unconditional avoids a second, weaker shape heuristic fighting that
    // measured-work handoff.
    if input.partition_label == "p1" && input.subset_parent_order.is_none() {
        return ProjectedKernel::Residual;
    }
    let active_terminals = input.active_terminals.iter().filter(|&&active| active).count();
    let vocab_tokens = input.vocab.len();
    let vocab_bytes = input.vocab.relevant_bytes().len();

    // p9 sits directly on the generic narrow-family crossover in the current
    // vocabulary split.  The 17.5k-token p9 family with only ~65 active
    // terminals is still a small residual problem, while finite projection can
    // spend hundreds of milliseconds scanning its state×vocabulary product.
    // Keep genuinely wide p9 families (notably the 300+ terminal Kubernetes
    // shapes) on the existing finite/auto policy; their finite scan is cheap.
    if input.partition_label == "p9" && active_terminals <= 96 {
        return ProjectedKernel::Residual;
    }

    // Both kernels are exact projections of the same finite-vocabulary L1
    // relation. Residual projection is strongest for small terminal families,
    // tiny vocabularies, and very wide vocabularies where enumerating trie
    // behavior creates a large state×token matrix. Finite projection wins when
    // many terminal residuals share behavior over a moderate vocabulary, most
    // notably split-L1 subsets.
    if active_terminals <= 64
        || vocab_tokens <= 32
        || vocab_tokens > 18_000
        || (active_terminals >= 120 && vocab_tokens > 3_000 && vocab_bytes > 128)
    {
        ProjectedKernel::Residual
    } else {
        ProjectedKernel::Finite
    }
}

/// Build L1 only through projected representations. `quotient` remains an
/// explicit diagnostic/reference implementation but is never used here.
pub(super) fn build(input: BuildInput<'_>) -> Option<LocalIdMapTerminalDwa> {
    profile_vector_projection_shape(input);
    let kernel = projected_kernel(input);
    if std::env::var_os("GLRMASK_PROFILE_L1_IMPLEMENTATIONS").is_some() {
        eprintln!(
            "[glrmask/profile][l1_projected_plan] partition={} active_terminals={} vocab_tokens={} vocab_bytes={} kernel={:?}",
            input.partition_label,
            input.active_terminals.iter().filter(|&&active| active).count(),
            input.vocab.len(),
            input.vocab.relevant_bytes().len(),
            kernel,
        );
    }
    match kernel {
        ProjectedKernel::Finite => build_finite_projected(input),
        ProjectedKernel::Residual => build_binary(input),
    }
}


#[derive(Default)]
struct FiniteTrieNode {
    token: Option<usize>,
    edge_token: usize,
    edge_start: u32,
    edge_end: u32,
    subtree_start: u32,
    subtree_end: u32,
    subtree_bytes: [u64; 4],
    children: Vec<u32>,
}

struct FiniteTrie {
    nodes: Vec<FiniteTrieNode>,
}

struct ReverseVocabTrie {
    // Nodes are topological: every parent index is smaller than its child.
    // Keeping the fields split avoids padding a ~170k-node p2 trie and makes
    // the hot parent/byte scan denser.
    parents: Box<[u32]>,
    bytes: Box<[u8]>,
    token_node: Box<[u32]>,
}

struct FiniteVocabProjection {
    aliases: Vec<Vec<u32>>,
    /// `(original_token_id, unique_token_index)` sorted by original token ID.
    /// Full partition vocabularies cache this once so projected L1 can build
    /// sorted reverse ID maps without per-grammar class sorting.
    original_order: Option<Box<[(u32, u32)]>>,
    tokens: Vec<Arc<[u8]>>,
    /// Sorted byte alphabet appearing anywhere in `tokens`. This is a pure
    /// vocabulary property and is reused by the residual kernel on every
    /// grammar instead of rescanning all token bytes.
    vocab_bytes: Box<[u8]>,
    trie: FiniteTrie,
    reverse_trie: Option<ReverseVocabTrie>,
}

impl crate::vocab::VocabDerivedArtifact for FiniteVocabProjection {}

fn collect_vocab_bytes(tokens: &[Arc<[u8]>]) -> Box<[u8]> {
    let mut relevant = [false; 256];
    for token in tokens {
        for &byte in token.iter() {
            relevant[byte as usize] = true;
        }
    }
    relevant
        .iter()
        .enumerate()
        .filter_map(|(byte, &used)| used.then_some(byte as u8))
        .collect::<Vec<_>>()
        .into_boxed_slice()
}


fn small_prepared_vocab_enabled(input: BuildInput<'_>) -> bool {
    let max_tokenizer_states = std::env::var(
        "GLRMASK_L1_SMALL_PREPARED_VOCAB_MAX_TOKENIZER_STATES",
    )
    .ok()
    .and_then(|value| value.parse::<u32>().ok())
    .unwrap_or(4_000);
    input.subset_parent_order.is_none()
        && input.tokenizer.num_states() <= max_tokenizer_states
}

fn preordered_vocab_map_enabled(input: BuildInput<'_>) -> bool {
    std::env::var("GLRMASK_L1_PREORDERED_VOCAB_MAP")
        .map(|value| {
            let value = value.trim();
            value.is_empty() || (value != "0" && !value.eq_ignore_ascii_case("false"))
        })
        .unwrap_or_else(|_| small_prepared_vocab_enabled(input))
}

fn residual_prepared_vocab_min_tokens(input: BuildInput<'_>) -> usize {
    std::env::var("GLRMASK_L1_RESIDUAL_PREPARED_VOCAB_MIN_TOKENS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or_else(|| {
            if small_prepared_vocab_enabled(input) {
                10_000
            } else {
                50_000
            }
        })
}

fn build_reverse_vocab_trie(tokens: &[Arc<[u8]>]) -> ReverseVocabTrie {
    let mut order = (0..tokens.len() as u32).collect::<Vec<_>>();
    order.sort_unstable_by(|&left, &right| {
        tokens[left as usize]
            .iter()
            .rev()
            .cmp(tokens[right as usize].iter().rev())
            .then_with(|| left.cmp(&right))
    });

    let edge_count = tokens.iter().map(|token| token.len()).sum::<usize>();
    let mut parents = Vec::with_capacity(edge_count + 1);
    let mut bytes = Vec::with_capacity(edge_count + 1);
    let mut token_node = vec![u32::MAX; tokens.len()];
    parents.push(u32::MAX);
    bytes.push(0);

    let mut path = vec![0u32];
    let mut previous: Option<u32> = None;
    for token_index in order {
        let token = &tokens[token_index as usize];
        let lcp = previous.map_or(0, |previous| {
            tokens[previous as usize]
                .iter()
                .rev()
                .zip(token.iter().rev())
                .take_while(|(left, right)| left == right)
                .count()
        });
        path.truncate(lcp + 1);
        let mut parent = *path.last().expect("reverse vocabulary trie has root");
        for &byte in token[..token.len() - lcp].iter().rev() {
            let node = parents.len() as u32;
            parents.push(parent);
            bytes.push(byte);
            path.push(node);
            parent = node;
        }
        debug_assert_eq!(token_node[token_index as usize], u32::MAX);
        token_node[token_index as usize] = parent;
        previous = Some(token_index);
    }
    // Reorder the immutable trie breadth-first.  Parents remain before children,
    // while nodes at the same depth become contiguous; this layout is markedly
    // friendlier to the compile-time parent-state scan than DFS/LCP order.
    let mut depths = vec![0usize; parents.len()];
    let mut max_depth = 0usize;
    for node in 1..parents.len() {
        let parent = parents[node] as usize;
        let depth = depths[parent] + 1;
        depths[node] = depth;
        max_depth = max_depth.max(depth);
    }
    let mut counts = vec![0usize; max_depth + 1];
    for &depth in &depths {
        counts[depth] += 1;
    }
    let mut level_offsets = Vec::with_capacity(counts.len() + 1);
    level_offsets.push(0u32);
    for &count in &counts {
        level_offsets.push(level_offsets.last().copied().unwrap() + count as u32);
    }
    let mut cursors = level_offsets[..counts.len()]
        .iter()
        .map(|&offset| offset as usize)
        .collect::<Vec<_>>();
    let mut old_to_new = vec![0u32; parents.len()];
    for (old, &depth) in depths.iter().enumerate() {
        let new = cursors[depth];
        cursors[depth] += 1;
        old_to_new[old] = new as u32;
    }
    debug_assert_eq!(old_to_new[0], 0);
    let mut reordered_parents = vec![u32::MAX; parents.len()];
    let mut reordered_bytes = vec![0u8; bytes.len()];
    for old in 1..parents.len() {
        let new = old_to_new[old] as usize;
        reordered_parents[new] = old_to_new[parents[old] as usize];
        reordered_bytes[new] = bytes[old];
    }
    for node in &mut token_node {
        *node = old_to_new[*node as usize];
    }
    ReverseVocabTrie {
        parents: reordered_parents.into_boxed_slice(),
        bytes: reordered_bytes.into_boxed_slice(),
        token_node: token_node.into_boxed_slice(),
    }
}

fn reverse_vocab_trie_min_tokens() -> usize {
    std::env::var("GLRMASK_L1_REVERSE_TRIE_MIN_TOKENS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        // Prepared p1/p4 vocabulary shards are already large enough for their
        // shared suffixes to amortize the compact reverse trie.  Building the
        // trie happens during vocabulary preparation, outside per-grammar
        // compile latency, while the residual kernel reuses it on every build.
        .unwrap_or(10_000)
}

fn build_finite_vocab_projection(vocab: &Vocab) -> Arc<FiniteVocabProjection> {
    if let Some(cached) = vocab.vocab_derived_cache_get::<FiniteVocabProjection>() {
        return cached;
    }
    let order = super::super::prepared_l1_identity_vocab_order(vocab);
    let mut tokens = Vec::<Arc<[u8]>>::with_capacity(vocab.len());
    let mut aliases = Vec::<Vec<u32>>::with_capacity(vocab.len());
    for (id, bytes) in order.token_entries_sorted.iter() {
        if tokens
            .last()
            .is_some_and(|token| token.as_ref() == bytes.as_ref())
        {
            aliases
                .last_mut()
                .expect("duplicate token has predecessor")
                .push(*id);
        } else {
            tokens.push(Arc::clone(bytes));
            aliases.push(vec![*id]);
        }
    }
    debug_assert_eq!(aliases.iter().map(Vec::len).sum::<usize>(), vocab.len());
    let mut original_order = aliases
        .iter()
        .enumerate()
        .flat_map(|(unique, originals)| {
            originals
                .iter()
                .copied()
                .map(move |original| (original, unique as u32))
        })
        .collect::<Vec<_>>();
    original_order.sort_unstable_by_key(|&(original, _)| original);
    let vocab_bytes = collect_vocab_bytes(&tokens);
    let trie = FiniteTrie::build(&tokens);
    let reverse_trie =
        (tokens.len() >= reverse_vocab_trie_min_tokens()).then(|| build_reverse_vocab_trie(&tokens));
    let projection = Arc::new(FiniteVocabProjection {
        aliases,
        original_order: Some(original_order.into_boxed_slice()),
        tokens,
        vocab_bytes,
        trie,
        reverse_trie,
    });
    vocab.vocab_derived_cache_set(Arc::clone(&projection));
    projection
}

pub(crate) fn prepare_finite_vocab_projection(vocab: &Vocab) {
    // The finite projected kernel is not just a large-p2 fallback: it is the
    // fast projected representation for several medium/large p1 shapes too.
    // This trie is a pure vocabulary artifact, so build it during vocabulary
    // preparation instead of charging the first grammar that selects finite L1.
    // For small partitions the same construction is sub-millisecond and often
    // never selected, so retaining another trie is counterproductive. p1/p2
    // full partitions are comfortably above this threshold.
    if vocab.len() >= 10_000 {
        let _ = build_finite_vocab_projection(vocab);
    }
}

fn finite_vocab_projection(input: BuildInput<'_>) -> (Arc<FiniteVocabProjection>, bool, f64) {
    let started = Instant::now();
    if input.subset_parent_order.is_none() {
        let cached = input
            .vocab
            .vocab_derived_cache_get::<FiniteVocabProjection>();
        if let Some(cached) = cached {
            return (cached, true, started.elapsed().as_secs_f64() * 1000.0);
        }
        let projection = build_finite_vocab_projection(input.vocab);
        return (
            projection,
            false,
            started.elapsed().as_secs_f64() * 1000.0,
        );
    }
    let (aliases, tokens) = unique_vocab(input);
    let vocab_bytes = collect_vocab_bytes(&tokens);
    let trie = FiniteTrie::build(&tokens);
    let reverse_trie =
        (tokens.len() >= reverse_vocab_trie_min_tokens()).then(|| build_reverse_vocab_trie(&tokens));
    (
        Arc::new(FiniteVocabProjection {
            aliases,
            original_order: None,
            tokens,
            vocab_bytes,
            trie,
            reverse_trie,
        }),
        false,
        started.elapsed().as_secs_f64() * 1000.0,
    )
}

impl FiniteTrie {
    /// Build a compressed radix trie directly from the byte-sorted unique
    /// vocabulary. Edge labels are slices of one vocabulary token, so the trie
    /// allocates no copied edge bytes and carries no unrelated subtree metadata.
    fn build(tokens: &[Arc<[u8]>]) -> Self {
        fn lcp(first: &[u8], last: &[u8], from: usize) -> usize {
            let mut index = from;
            let end = first.len().min(last.len());
            while index < end && first[index] == last[index] {
                index += 1;
            }
            index
        }

        fn add_range(
            tokens: &[Arc<[u8]>],
            begin: usize,
            end: usize,
            parent_prefix_len: usize,
            nodes: &mut Vec<FiniteTrieNode>,
        ) -> u32 {
            debug_assert!(begin < end);
            let first = tokens[begin].as_ref();
            let last = tokens[end - 1].as_ref();
            let prefix_len = lcp(first, last, parent_prefix_len);
            let token = (first.len() == prefix_len).then_some(begin);
            let node = nodes.len() as u32;
            nodes.push(FiniteTrieNode {
                token,
                edge_token: begin,
                edge_start: parent_prefix_len as u32,
                edge_end: prefix_len as u32,
                subtree_start: begin as u32,
                subtree_end: end as u32,
                subtree_bytes: [0; 4],
                children: Vec::new(),
            });
            let mut cursor = begin + usize::from(token.is_some());
            while cursor < end {
                let byte = tokens[cursor][prefix_len];
                let child_begin = cursor;
                cursor += 1;
                while cursor < end && tokens[cursor][prefix_len] == byte {
                    cursor += 1;
                }
                let child = add_range(tokens, child_begin, cursor, prefix_len, nodes);
                nodes[node as usize].children.push(child);
            }
            let children = nodes[node as usize].children.clone();
            let mut subtree_bytes = [0u64; 4];
            for child in children {
                let child_node = &nodes[child as usize];
                let edge = &tokens[child_node.edge_token]
                    [child_node.edge_start as usize..child_node.edge_end as usize];
                for &byte in edge {
                    subtree_bytes[byte as usize >> 6] |= 1u64 << (byte & 63);
                }
                for (dst, &src) in subtree_bytes.iter_mut().zip(child_node.subtree_bytes.iter()) {
                    *dst |= src;
                }
            }
            nodes[node as usize].subtree_bytes = subtree_bytes;
            node
        }

        if tokens.is_empty() {
            return Self { nodes: vec![FiniteTrieNode::default()] };
        }
        let mut nodes = Vec::new();
        let root = add_range(tokens, 0, tokens.len(), 0, &mut nodes);
        debug_assert_eq!(root, 0);
        // If all tokens have a non-empty common prefix, `add_range` puts that
        // prefix on node zero. Split it into an empty root so the first cached
        // transition is still one common vocabulary bucket.
        if nodes[0].edge_end != 0 {
            let old = std::mem::take(&mut nodes);
            let mut shifted = Vec::with_capacity(old.len() + 1);
            shifted.push(FiniteTrieNode {
                children: vec![1],
                ..FiniteTrieNode::default()
            });
            shifted.extend(old.into_iter().map(|mut node| {
                for child in &mut node.children {
                    *child += 1;
                }
                node
            }));
            nodes = shifted;
        }
        Self { nodes }
    }

    #[inline]
    fn edge<'a>(&self, node: u32, tokens: &'a [Arc<[u8]>]) -> &'a [u8] {
        let node = &self.nodes[node as usize];
        &tokens[node.edge_token][node.edge_start as usize..node.edge_end as usize]
    }
}



struct FrozenFiniteScanner {
    width: usize,
    transitions: Box<[u32]>,
    byte_slot: [u16; 256],
    signatures: Box<[u32]>,
    self_loops: Box<[crate::ds::u8set::U8Set]>,
}

impl FrozenFiniteScanner {
    fn build(scanner: &mut Scanner<'_>, input: BuildInput<'_>) -> Self {
        let mut vocab_bytes = input.vocab.relevant_bytes().iter().copied().collect::<Vec<_>>();
        vocab_bytes.sort_unstable();
        let (representatives, byte_representative) = quotient_input_bytes(input, &vocab_bytes);
        let width = representatives.len();
        let mut representative_slot = [u16::MAX; 256];
        for (slot, &byte) in representatives.iter().enumerate() {
            representative_slot[byte as usize] = slot as u16;
        }
        let mut byte_slot = [u16::MAX; 256];
        for &byte in &vocab_bytes {
            byte_slot[byte as usize] = representative_slot[byte_representative[byte as usize] as usize];
        }

        let mut transitions = Vec::<u32>::new();
        let mut state = 0usize;
        while state < scanner.configs.len() {
            transitions.reserve(width);
            for &byte in &representatives {
                transitions.push(scanner.step(state as u32, byte));
            }
            state += 1;
        }
        let num_states = scanner.configs.len();
        debug_assert_eq!(transitions.len(), num_states * width);
        let signatures = (0..num_states)
            .map(|state| scanner.signature(state as u32))
            .collect::<Vec<_>>();
        let mut self_loops = vec![crate::ds::u8set::U8Set::empty(); num_states];
        for state in 0..num_states {
            let row = &transitions[state * width..(state + 1) * width];
            for &byte in &vocab_bytes {
                let slot = byte_slot[byte as usize] as usize;
                if row[slot] == state as u32 {
                    self_loops[state].insert(byte);
                }
            }
        }
        Self {
            width,
            transitions: transitions.into_boxed_slice(),
            byte_slot,
            signatures: signatures.into_boxed_slice(),
            self_loops: self_loops.into_boxed_slice(),
        }
    }

    #[inline]
    fn step(&self, state: u32, byte: u8) -> u32 {
        if state == DEAD {
            return DEAD;
        }
        let slot = self.byte_slot[byte as usize];
        debug_assert_ne!(slot, u16::MAX);
        self.transitions[state as usize * self.width + slot as usize]
    }

    #[inline]
    fn step_bytes(&self, mut state: u32, bytes: &[u8]) -> u32 {
        for &byte in bytes {
            state = self.step(state, byte);
            if state == DEAD {
                break;
            }
        }
        state
    }

    #[inline]
    fn signature(&self, state: u32) -> u32 {
        if state == DEAD { 0 } else { self.signatures[state as usize] }
    }
}

fn collect_finite_profile_frozen(
    scanner: &FrozenFiniteScanner,
    trie: &FiniteTrie,
    tokens: &[Arc<[u8]>],
    node: u32,
    config: u32,
    out: &mut Vec<ProfileRun>,
    pair_visits: &mut usize,
    uniform_subtrees: &mut usize,
    uniform_tokens: &mut usize,
) {
    *pair_visits += 1;
    let current = &trie.nodes[node as usize];
    let loops = scanner.self_loops[config as usize];
    let covered = crate::ds::u8set::U8Set::from_words(current.subtree_bytes).is_subset(&loops);
    if covered {
        let signature = scanner.signature(config);
        push_profile_run(out, current.subtree_start, current.subtree_end, signature);
        *uniform_subtrees += 1;
        *uniform_tokens += (current.subtree_end - current.subtree_start) as usize;
        return;
    }
    if let Some(token) = current.token {
        push_profile_run(
            out,
            token as u32,
            token as u32 + 1,
            scanner.signature(config),
        );
    }
    for &child in &current.children {
        let target = scanner.step_bytes(config, trie.edge(child, tokens));
        if target != DEAD {
            collect_finite_profile_frozen(
                scanner,
                trie,
                tokens,
                child,
                target,
                out,
                pair_visits,
                uniform_subtrees,
                uniform_tokens,
            );
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ProfileRun {
    start: u32,
    end: u32,
    signature: u32,
}

#[inline]
fn push_profile_run(out: &mut Vec<ProfileRun>, start: u32, end: u32, signature: u32) {
    if signature == 0 || start == end {
        return;
    }
    if let Some(last) = out.last_mut()
        && last.end == start
        && last.signature == signature
    {
        last.end = end;
        return;
    }
    out.push(ProfileRun { start, end, signature });
}

fn collect_finite_profile(
    scanner: &mut Scanner<'_>,
    trie: &FiniteTrie,
    tokens: &[Arc<[u8]>],
    self_loops: &[crate::ds::u8set::U8Set],
    node: u32,
    config: u32,
    out: &mut Vec<ProfileRun>,
    pair_visits: &mut usize,
    uniform_subtrees: &mut usize,
    uniform_tokens: &mut usize,
    max_profile_runs: usize,
    max_pair_visits: usize,
) -> bool {
    *pair_visits += 1;
    if *pair_visits > max_pair_visits {
        return false;
    }
    let current = &trie.nodes[node as usize];
    if let Some(state) = scanner.singleton_state(config) {
        let loops = self_loops[state as usize];
        let covered = crate::ds::u8set::U8Set::from_words(current.subtree_bytes).is_subset(&loops);
        if covered {
            let signature = scanner.signature(config);
            push_profile_run(
                out,
                current.subtree_start,
                current.subtree_end,
                signature,
            );
            *uniform_subtrees += 1;
            *uniform_tokens += (current.subtree_end - current.subtree_start) as usize;
            return out.len() <= max_profile_runs;
        }
    }
    if let Some(token) = current.token {
        let signature = scanner.signature(config);
        push_profile_run(out, token as u32, token as u32 + 1, signature);
        if out.len() > max_profile_runs {
            return false;
        }
    }
    for &child in &current.children {
        let target = scanner.step_bytes(config, trie.edge(child, tokens));
        if target != DEAD {
            if !collect_finite_profile(
                scanner, trie, tokens, self_loops, child, target, out, pair_visits,
                uniform_subtrees, uniform_tokens, max_profile_runs, max_pair_visits,
            ) {
                return false;
            }
        }
    }
    true
}

#[derive(Clone, Copy)]
struct FiniteRowEvent {
    position: u32,
    row: u32,
    signature: u32,
}

/// Collision-free canonical ID for a mutable vector of finite-row signatures.
/// Leaves and internal pairs are hash-consed, so the root ID is equal iff the
/// entire vector is equal. One row update costs O(log rows) and never copies the
/// full state-class vector.
struct CanonicalSignatureVector {
    base: usize,
    tree: Vec<u32>,
    leaf_ids: FxHashMap<u32, u32>,
    pair_ids: FxHashMap<(u32, u32), u32>,
    next_id: u32,
}

impl CanonicalSignatureVector {
    fn new(width: usize) -> Self {
        let base = width.max(1).next_power_of_two();
        let mut this = Self {
            base,
            tree: vec![0; base * 2],
            leaf_ids: FxHashMap::default(),
            pair_ids: FxHashMap::default(),
            next_id: 0,
        };
        let zero = this.intern_leaf(0);
        this.tree[base..].fill(zero);
        for node in (1..base).rev() {
            let left = this.tree[node * 2];
            let right = this.tree[node * 2 + 1];
            this.tree[node] = this.intern_pair(left, right);
        }
        this
    }

    fn alloc_id(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn intern_leaf(&mut self, value: u32) -> u32 {
        if let Some(&id) = self.leaf_ids.get(&value) {
            return id;
        }
        let id = self.alloc_id();
        self.leaf_ids.insert(value, id);
        id
    }

    fn intern_pair(&mut self, left: u32, right: u32) -> u32 {
        if let Some(&id) = self.pair_ids.get(&(left, right)) {
            return id;
        }
        let id = self.alloc_id();
        self.pair_ids.insert((left, right), id);
        id
    }

    fn set(&mut self, index: usize, value: u32) {
        debug_assert!(index < self.base);
        let mut node = self.base + index;
        let leaf = self.intern_leaf(value);
        if self.tree[node] == leaf {
            return;
        }
        self.tree[node] = leaf;
        node /= 2;
        while node != 0 {
            let left = self.tree[node * 2];
            let right = self.tree[node * 2 + 1];
            let id = self.intern_pair(left, right);
            if self.tree[node] == id {
                break;
            }
            self.tree[node] = id;
            node /= 2;
        }
    }

    fn root(&self) -> u32 {
        self.tree[1]
    }
}

fn finite_compact_runs(
    root_token: Option<usize>,
    class_fingerprints: &[Vec<u32>],
    profiles: &[Arc<[ProfileRun]>],
    token_count: usize,
    materialize_compact_rows: bool,
) -> (Vec<u32>, Vec<usize>, Vec<Vec<u32>>, usize, usize) {
    let mut rows = Vec::<Vec<ProfileRun>>::with_capacity(class_fingerprints.len());
    let mut referenced_runs = 0usize;
    for fingerprint in class_fingerprints {
        let mut row = Vec::<ProfileRun>::new();
        let mut offset = 0usize;
        if let Some(token) = root_token {
            let signature = fingerprint[0];
            push_profile_run(&mut row, token as u32, token as u32 + 1, signature);
            offset = 1;
        }
        for &profile in &fingerprint[offset..] {
            for run in profiles[profile as usize].iter() {
                push_profile_run(&mut row, run.start, run.end, run.signature);
            }
        }
        referenced_runs += row.len();
        rows.push(row);
    }

    // For a small state-class × token matrix, direct materialization is much
    // cheaper than maintaining the persistent segment-tree signature vector.
    // p6/p10 hard-schema shapes are only O(10^6) cells, while the run sweep was
    // written for 80k-token p2 and pays hashing/tree-update overhead per event.
    let dense_cells = rows.len().saturating_mul(token_count);
    let dense_limit = std::env::var("GLRMASK_L1_FINITE_DENSE_COMPACT_MAX_CELLS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(2_000_000);
    if dense_cells <= dense_limit
        && std::env::var_os("GLRMASK_DISABLE_L1_FINITE_DENSE_COMPACT").is_none()
    {
        let mut columns = (0..token_count)
            .map(|_| vec![0u32; rows.len()])
            .collect::<Vec<_>>();
        for (row_index, row) in rows.iter().enumerate() {
            for run in row {
                for token in run.start as usize..run.end as usize {
                    columns[token][row_index] = run.signature;
                }
            }
        }
        let mut class_ids = FxHashMap::<Vec<u32>, u32>::default();
        let mut token_class = vec![0u32; token_count];
        let mut token_reps = Vec::<usize>::new();
        for (token, column) in columns.iter().enumerate() {
            let class = if let Some(&class) = class_ids.get(column) {
                class
            } else {
                let class = token_reps.len() as u32;
                class_ids.insert(column.clone(), class);
                token_reps.push(token);
                class
            };
            token_class[token] = class;
        }
        let mut compact_rows = Vec::new();
        if materialize_compact_rows {
            compact_rows = (0..rows.len())
                .map(|_| vec![0u32; token_reps.len()])
                .collect::<Vec<_>>();
            for (class, &token) in token_reps.iter().enumerate() {
                for row in 0..rows.len() {
                    compact_rows[row][class] = columns[token][row];
                }
            }
        }
        return (token_class, token_reps, compact_rows, 0, referenced_runs);
    }

    // Only the sweep paths consume boundary events.  Dense compaction above
    // materializes the small matrix directly, so constructing two events for
    // every run before knowing which path wins is pure allocation/fill work.
    let mut events = Vec::<FiniteRowEvent>::with_capacity(referenced_runs.saturating_mul(2));
    for (row_index, row) in rows.iter().enumerate() {
        for run in row {
            events.push(FiniteRowEvent {
                position: run.start,
                row: row_index as u32,
                signature: run.signature,
            });
            events.push(FiniteRowEvent {
                position: run.end,
                row: row_index as u32,
                signature: 0,
            });
        }
    }

    let use_exact_fingerprint_sweep = std::env::var("GLRMASK_L1_FINITE_EXACT_FINGERPRINT_SWEEP")
        .map(|value| {
            let value = value.trim();
            value.is_empty() || (value != "0" && !value.eq_ignore_ascii_case("false"))
        })
        // This is exact: the 128-bit fingerprint is only an index and every
        // candidate reuse is verified against the complete column vector.
        // Revalidation on the vocab-only path also makes it faster than the
        // legacy persistent-tree sweep, so use the same default as Static.
        .unwrap_or(true);
    if use_exact_fingerprint_sweep {
        // Event positions are vocabulary indices in a compact bounded domain.
        // Counting-sort them in O(events + tokens) instead of comparison-sorting
        // millions of `(position,row,kind)` tuples. End events occupy the first
        // part of each position bucket, preserving the required end-before-start
        // semantics for adjacent runs.
        let mut end_counts = vec![0usize; token_count + 1];
        let mut start_counts = vec![0usize; token_count + 1];
        for event in &events {
            if event.signature == 0 {
                end_counts[event.position as usize] += 1;
            } else {
                start_counts[event.position as usize] += 1;
            }
        }
        let mut offsets = vec![0usize; token_count + 2];
        for position in 0..=token_count {
            offsets[position + 1] = offsets[position]
                + end_counts[position]
                + start_counts[position];
        }
        let mut end_cursor = offsets[..=token_count].to_vec();
        let mut start_cursor = (0..=token_count)
            .map(|position| offsets[position] + end_counts[position])
            .collect::<Vec<_>>();
        let mut ordered = vec![FiniteRowEvent {
            position: 0,
            row: 0,
            signature: 0,
        }; events.len()];
        for event in events.drain(..) {
            let position = event.position as usize;
            let cursor = if event.signature == 0 {
                &mut end_cursor[position]
            } else {
                &mut start_cursor[position]
            };
            ordered[*cursor] = event;
            *cursor += 1;
        }
        events = ordered;
        #[inline(always)]
        fn mix64(mut value: u64) -> u64 {
            value ^= value >> 30;
            value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
            value ^= value >> 27;
            value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
            value ^ (value >> 31)
        }
        #[inline(always)]
        fn cell_hash(row: u32, signature: u32, seed: u64) -> u64 {
            if signature == 0 {
                0
            } else {
                mix64(seed ^ ((row as u64) << 32) ^ signature as u64)
            }
        }

        // Maintain a cheap 128-bit fingerprint of the current exact token
        // column. Fingerprints are only an index: every reuse is verified by
        // comparing the complete row vector, so hash collisions cannot affect
        // correctness. This removes the O(log rows) hash-consed segment-tree
        // update paid for every run boundary on large finite p2 matrices.
        let mut current = vec![0u32; class_fingerprints.len()];
        let mut fingerprint_a = 0u64;
        let mut fingerprint_b = 0u64;
        let mut buckets = FxHashMap::<(u64, u64), Vec<u32>>::default();
        let mut class_vectors = Vec::<Box<[u32]>>::new();
        let mut token_class = vec![0u32; token_count];
        let mut token_reps = Vec::<usize>::new();
        let mut position = 0usize;
        let mut event_index = 0usize;
        while position < token_count {
            while event_index < events.len() && events[event_index].position as usize == position {
                let event = events[event_index];
                let row = event.row as usize;
                let old = current[row];
                if old != event.signature {
                    fingerprint_a ^= cell_hash(event.row, old, 0x243f_6a88_85a3_08d3)
                        ^ cell_hash(event.row, event.signature, 0x243f_6a88_85a3_08d3);
                    fingerprint_b ^= cell_hash(event.row, old, 0x1319_8a2e_0370_7344)
                        ^ cell_hash(event.row, event.signature, 0x1319_8a2e_0370_7344);
                    current[row] = event.signature;
                }
                event_index += 1;
            }
            let next_position = events
                .get(event_index)
                .map_or(token_count, |event| event.position as usize)
                .min(token_count);
            debug_assert!(next_position > position || event_index < events.len());
            if next_position > position {
                let key = (fingerprint_a, fingerprint_b);
                let candidates = buckets.entry(key).or_default();
                let class = if let Some(class) = candidates
                    .iter()
                    .copied()
                    .find(|&class| class_vectors[class as usize].as_ref() == current.as_slice())
                {
                    class
                } else {
                    let class = class_vectors.len() as u32;
                    class_vectors.push(current.clone().into_boxed_slice());
                    candidates.push(class);
                    token_reps.push(position);
                    class
                };
                token_class[position..next_position].fill(class);
                position = next_position;
            }
        }
        let mut compact_rows = Vec::new();
        if materialize_compact_rows {
            compact_rows = (0..class_fingerprints.len())
                .map(|_| vec![0u32; class_vectors.len()])
                .collect::<Vec<_>>();
            for (class, column) in class_vectors.iter().enumerate() {
                for (row, &signature) in column.iter().enumerate() {
                    compact_rows[row][class] = signature;
                }
            }
        }
        return (
            token_class,
            token_reps,
            compact_rows,
            events.len(),
            referenced_runs,
        );
    }

    // Legacy diagnostic path retains its comparison sort.
    events.sort_unstable_by_key(|event| {
        (event.position, event.row, u8::from(event.signature != 0))
    });
    let mut vector = CanonicalSignatureVector::new(class_fingerprints.len());
    let mut class_by_root = FxHashMap::<u32, u32>::default();
    let mut token_class = vec![0u32; token_count];
    let mut token_reps = Vec::<usize>::new();
    let mut position = 0usize;
    let mut event_index = 0usize;
    while position < token_count {
        while event_index < events.len() && events[event_index].position as usize == position {
            let event = events[event_index];
            vector.set(event.row as usize, event.signature);
            event_index += 1;
        }
        let next_position = events
            .get(event_index)
            .map_or(token_count, |event| event.position as usize)
            .min(token_count);
        debug_assert!(next_position > position || event_index < events.len());
        if next_position > position {
            let root = vector.root();
            let class = if let Some(&class) = class_by_root.get(&root) {
                class
            } else {
                let class = token_reps.len() as u32;
                class_by_root.insert(root, class);
                token_reps.push(position);
                class
            };
            token_class[position..next_position].fill(class);
            position = next_position;
        }
    }

    let mut reps_sorted = token_reps
        .iter()
        .enumerate()
        .map(|(class, &token)| (token, class))
        .collect::<Vec<_>>();
    reps_sorted.sort_unstable();
    let mut compact_rows = Vec::new();
    if materialize_compact_rows {
        compact_rows = Vec::with_capacity(rows.len());
        for row in &rows {
            let mut compact = vec![0u32; token_reps.len()];
            let mut run_index = 0usize;
            for &(token, class) in &reps_sorted {
                while run_index < row.len() && row[run_index].end as usize <= token {
                    run_index += 1;
                }
                if let Some(run) = row.get(run_index)
                    && run.start as usize <= token
                    && token < run.end as usize
                {
                    compact[class] = run.signature;
                }
            }
            compact_rows.push(compact);
        }
    }
    (token_class, token_reps, compact_rows, events.len(), referenced_runs)
}

/// Exact finite-vocabulary L1 projection.
///
/// For one first radix bucket, a raw lexer state matters only through the lexer
/// configuration reached after that bucket's common prefix. Equal targets share
/// the complete computation over the remaining finite suffix set. The resulting
/// per-bucket profile IDs form an exact short fingerprint for the raw state;
/// only fingerprints that survive as state classes are materialized into rows.
fn build_finite_projected(input: BuildInput<'_>) -> Option<LocalIdMapTerminalDwa> {
    build_finite_projected_impl(input, false)
}

fn build_finite_projected_unbounded(input: BuildInput<'_>) -> Option<LocalIdMapTerminalDwa> {
    build_finite_projected_impl(input, true)
}

fn build_finite_projected_impl(
    input: BuildInput<'_>,
    bypass_adaptive_budget: bool,
) -> Option<LocalIdMapTerminalDwa> {
    if input.vocab.is_empty() {
        return None;
    }
    let total = Instant::now();
    let (finite_vocab, finite_vocab_cache_hit, prep_ms) = finite_vocab_projection(input);
    let aliases = finite_vocab.aliases.as_slice();
    let tokens = finite_vocab.tokens.as_slice();
    let trie = &finite_vocab.trie;

    let scan_started = Instant::now();
    let mut scanner = Scanner::new(input);
    let mut starts = BTreeMap::<u32, Vec<u32>>::new();
    if let Some(state_map) = input.initial_state_map {
        debug_assert_eq!(
            state_map.original_to_internal.len(),
            input.tokenizer.num_states() as usize,
        );
        // `initial_state_map` is an exact certified quotient for this L1 branch.
        // Finite projection historically ignored it and constructed a projected
        // Scanner root for every raw tokenizer state, only to rediscover the
        // same equivalence after vocabulary replay.  Evaluate one representative
        // per certified class, then fan the resulting finite state class back to
        // every raw member of that class.
        for (class, &representative) in state_map.representative_original_ids.iter().enumerate() {
            if representative == u32::MAX {
                continue;
            }
            let start = scanner.start(representative);
            starts
                .entry(start)
                .or_default()
                .extend_from_slice(&state_map.internal_to_originals[class]);
        }
        // An incomplete quotient conservatively retains unmapped raw states.
        // Under an exact boundary domain, however, unmapped means excluded.
        if !input.initial_state_domain_is_exact {
            for (raw, &class) in state_map.original_to_internal.iter().enumerate() {
                if class == u32::MAX {
                    starts
                        .entry(scanner.start(raw as u32))
                        .or_default()
                        .push(raw as u32);
                }
            }
        }
    } else {
        for raw in 0..input.tokenizer.num_states() {
            starts.entry(scanner.start(raw)).or_default().push(raw);
        }
    }

    let root_token = trie.nodes[0].token;
    let root_children = trie.nodes[0].children.clone();
    let mut state_class = vec![0u32; input.tokenizer.num_states() as usize];
    let mut class_fingerprints = Vec::<Vec<u32>>::new();
    let mut class_ids = FxHashMap::<Vec<u32>, u32>::default();

    let mut profiles = vec![Arc::<[ProfileRun]>::from([])];
    let mut profile_ids = FxHashMap::<Arc<[ProfileRun]>, u32>::default();
    let mut bucket_cache = FxHashMap::<(u32, u32), u32>::default();
    let mut cache_hits = 0usize;
    let mut pair_visits = 0usize;
    let mut uniform_subtrees = 0usize;
    let mut uniform_tokens = 0usize;
    let self_loops = input.tokenizer.all_self_loop_bytes();
    let adaptive_budget = !bypass_adaptive_budget
        && input.subset_parent_order.is_none()
        && matches!(input.partition_label, "p1" | "p2");
    let (profile_run_budget, pair_visit_budget) = if adaptive_budget {
        let (runs_env, pairs_env, default_pairs) = if input.partition_label == "p1" {
            (
                "GLRMASK_P1_FINITE_MAX_PROFILE_RUNS",
                "GLRMASK_P1_FINITE_MAX_PAIR_VISITS",
                50_000,
            )
        } else {
            (
                "GLRMASK_P2_FINITE_MAX_PROFILE_RUNS",
                "GLRMASK_P2_FINITE_MAX_PAIR_VISITS",
                150_000,
            )
        };
        (
            std::env::var(runs_env)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(2_048),
            std::env::var(pairs_env)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(default_pairs),
        )
    } else {
        (usize::MAX, usize::MAX)
    };
    let mut profile_run_count = 0usize;

    let parallel_profile_override = std::env::var("GLRMASK_L1_FINITE_PARALLEL_PROFILES").ok();
    let force_parallel_profiles = parallel_profile_override
        .as_deref()
        .is_some_and(|value| value.trim().eq_ignore_ascii_case("force"));
    let parallel_profiles_enabled = parallel_profile_override
        .as_deref()
        .map(|value| {
            let value = value.trim();
            value.is_empty()
                || value.eq_ignore_ascii_case("force")
                || (value != "0" && !value.eq_ignore_ascii_case("false"))
        })
        .unwrap_or(true);
    let parallel_profiles = (bypass_adaptive_budget || force_parallel_profiles)
        && input.subset_parent_order.is_none()
        && input.vocab.len() >= 10_000
        && input.tokenizer.num_states() >= 10_000
        && rayon::current_num_threads() > 1
        && parallel_profiles_enabled;

    if parallel_profiles {
        let freeze_started = Instant::now();
        let frozen = FrozenFiniteScanner::build(&mut scanner, input);
        let freeze_ms = freeze_started.elapsed().as_secs_f64() * 1000.0;

        // Enumerate the small set of unique top-level radix jobs before doing
        // any suffix work. The historical finite scanner used one mutable
        // interner throughout the walk, which serialized all 60M+ pair visits
        // even though equal `(child,target)` jobs are independent once the
        // projected config graph has been frozen.
        let mut key_ids = FxHashMap::<(u32, u32), usize>::default();
        let mut keys = Vec::<(u32, u32)>::new();
        let mut work = Vec::<(u32, Vec<u32>, Vec<usize>)>::with_capacity(starts.len());
        for (start, raw_states) in starts {
            let mut ids = Vec::with_capacity(root_children.len());
            for &child in &root_children {
                let target = frozen.step_bytes(start, trie.edge(child, tokens));
                if target == DEAD {
                    ids.push(usize::MAX);
                    continue;
                }
                let key = (child, target);
                let id = if let Some(&id) = key_ids.get(&key) {
                    cache_hits += 1;
                    id
                } else {
                    let id = keys.len();
                    key_ids.insert(key, id);
                    keys.push(key);
                    id
                };
                ids.push(id);
            }
            work.push((start, raw_states, ids));
        }

        let results = keys
            .par_iter()
            .map(|&(child, target)| {
                let mut values = Vec::<ProfileRun>::new();
                let mut visits = 0usize;
                let mut uniform = 0usize;
                let mut uniform_token_count = 0usize;
                collect_finite_profile_frozen(
                    &frozen,
                    trie,
                    tokens,
                    child,
                    target,
                    &mut values,
                    &mut visits,
                    &mut uniform,
                    &mut uniform_token_count,
                );
                (
                    Arc::<[ProfileRun]>::from(values),
                    visits,
                    uniform,
                    uniform_token_count,
                )
            })
            .collect::<Vec<_>>();

        if std::env::var_os("GLRMASK_PROFILE_L1_IMPLEMENTATIONS").is_some() {
            let mut visit_counts = results.iter().map(|(_, visits, _, _)| *visits).collect::<Vec<_>>();
            visit_counts.sort_unstable();
            let quantile = |num: usize, den: usize| -> usize {
                if visit_counts.is_empty() { return 0; }
                visit_counts[(visit_counts.len() - 1) * num / den]
            };
            let max = visit_counts.last().copied().unwrap_or(0);
            let total = visit_counts.iter().sum::<usize>();
            let top_1pct = visit_counts.iter().rev().take((visit_counts.len() / 100).max(1)).sum::<usize>();
            eprintln!(
                "[glrmask/profile][l1_finite_parallel_job_weights] partition={} jobs={} total_visits={} p50={} p90={} p99={} max={} top1pct_visits={} top1pct_pct={:.2}",
                input.partition_label,
                visit_counts.len(),
                total,
                quantile(50, 100),
                quantile(90, 100),
                quantile(99, 100),
                max,
                top_1pct,
                100.0 * top_1pct as f64 / total.max(1) as f64,
            );
        }
        let mut key_profile = vec![0u32; keys.len()];
        for (key, (values, visits, uniform, uniform_token_count)) in
            results.into_iter().enumerate()
        {
            pair_visits += visits;
            uniform_subtrees += uniform;
            uniform_tokens += uniform_token_count;
            let profile = if values.is_empty() {
                0
            } else if let Some(&profile) = profile_ids.get(&values) {
                profile
            } else {
                let profile = profiles.len() as u32;
                profile_ids.insert(Arc::clone(&values), profile);
                profiles.push(values);
                profile
            };
            key_profile[key] = profile;
            bucket_cache.insert(keys[key], profile);
        }

        for (start, raw_states, ids) in work {
            let mut fingerprint =
                Vec::with_capacity(root_children.len() + usize::from(root_token.is_some()));
            if root_token.is_some() {
                fingerprint.push(frozen.signature(start));
            }
            for id in ids {
                fingerprint.push(if id == usize::MAX { 0 } else { key_profile[id] });
            }
            let class = if let Some(&class) = class_ids.get(&fingerprint) {
                class
            } else {
                let class = class_fingerprints.len() as u32;
                class_ids.insert(fingerprint.clone(), class);
                class_fingerprints.push(fingerprint);
                class
            };
            for raw in raw_states {
                state_class[raw as usize] = class;
            }
        }
        if std::env::var_os("GLRMASK_PROFILE_L1_IMPLEMENTATIONS").is_some() {
            eprintln!(
                "[glrmask/profile][l1_finite_parallel_profiles] partition={} configs={} byte_classes={} jobs={} cache_hits={} freeze_ms={:.3}",
                input.partition_label,
                frozen.signatures.len(),
                frozen.width,
                keys.len(),
                cache_hits,
                freeze_ms,
            );
        }
    } else {
        for (start, raw_states) in starts {
            let mut fingerprint =
                Vec::with_capacity(root_children.len() + usize::from(root_token.is_some()));
            if root_token.is_some() {
                fingerprint.push(scanner.signature(start));
            }
            for &child in &root_children {
                let target = scanner.step_bytes(start, trie.edge(child, tokens));
                if target == DEAD {
                    fingerprint.push(0);
                    continue;
                }
                let key = (child, target);
                let profile = if let Some(&profile) = bucket_cache.get(&key) {
                    cache_hits += 1;
                    profile
                } else {
                    let mut values = Vec::new();
                    let remaining_runs = profile_run_budget.saturating_sub(profile_run_count);
                    if !collect_finite_profile(
                        &mut scanner,
                        trie,
                        tokens,
                        self_loops.as_ref(),
                        child,
                        target,
                        &mut values,
                        &mut pair_visits,
                        &mut uniform_subtrees,
                        &mut uniform_tokens,
                        remaining_runs,
                        pair_visit_budget,
                    ) {
                        if std::env::var_os("GLRMASK_PROFILE_L1_IMPLEMENTATIONS").is_some() {
                            eprintln!(
                                "[glrmask/profile][l1_finite_budget_abort] partition={} profile_runs={} pair_visits={} max_profile_runs={} max_pair_visits={} action=residual",
                                input.partition_label, profile_run_count + values.len(), pair_visits,
                                profile_run_budget, pair_visit_budget,
                            );
                        }
                        return build_binary_without_finite_switch(input);
                    }
                    let values: Arc<[ProfileRun]> = Arc::from(values);
                    let profile = if values.is_empty() {
                        0
                    } else if let Some(&profile) = profile_ids.get(&values) {
                        profile
                    } else {
                        let profile = profiles.len() as u32;
                        profile_run_count += values.len();
                        profile_ids.insert(Arc::clone(&values), profile);
                        profiles.push(values);
                        profile
                    };
                    if profile_run_count > profile_run_budget {
                        return build_binary_without_finite_switch(input);
                    }
                    bucket_cache.insert(key, profile);
                    profile
                };
                fingerprint.push(profile);
            }

            let class = if let Some(&class) = class_ids.get(&fingerprint) {
                class
            } else {
                let class = class_fingerprints.len() as u32;
                class_ids.insert(fingerprint.clone(), class);
                class_fingerprints.push(fingerprint);
                class
            };
            for raw in raw_states {
                state_class[raw as usize] = class;
            }
        }
    }
    let scan_ms = scan_started.elapsed().as_secs_f64() * 1000.0;

    let materialize_started = Instant::now();
    let use_run_sweep = std::env::var_os("GLRMASK_DISABLE_L1_FINITE_RUN_SWEEP").is_none();
    let (finished, run_sweep_events, referenced_runs) = if use_run_sweep {
        let compact_started = Instant::now();
        let (token_class, _token_reps, compact_rows, events, referenced_runs) =
            finite_compact_runs(
                root_token,
                &class_fingerprints,
                &profiles,
                aliases.len(),
                true,
            );
        let compact_ms = compact_started.elapsed().as_secs_f64() * 1000.0;
        let preordered_vocab = preordered_vocab_map_enabled(input)
            .then(|| finite_vocab.original_order.as_deref())
            .flatten();
        let finished = common::finish_compacted(
            input,
            aliases,
            &scanner.signatures,
            state_class,
            compact_rows,
            token_class,
            preordered_vocab,
            prep_ms + scan_ms,
            compact_ms,
            || total.elapsed().as_secs_f64() * 1000.0,
        )?;
        (finished, events, referenced_runs)
    } else {
        let mut rows = Vec::with_capacity(class_fingerprints.len());
        for fingerprint in &class_fingerprints {
            let mut row = vec![0u32; aliases.len()];
            let mut offset = 0usize;
            if let Some(token) = root_token {
                row[token] = fingerprint[0];
                offset = 1;
            }
            for &profile in &fingerprint[offset..] {
                for run in profiles[profile as usize].iter() {
                    row[run.start as usize..run.end as usize].fill(run.signature);
                }
            }
            rows.push(row);
        }
        let materialize_ms = materialize_started.elapsed().as_secs_f64() * 1000.0;
        let finished = common::finish(
            input,
            aliases,
            &scanner.signatures,
            state_class,
            rows,
            prep_ms + scan_ms + materialize_ms,
            || total.elapsed().as_secs_f64() * 1000.0,
        )?;
        (finished, 0, 0)
    };
    let materialize_ms = materialize_started.elapsed().as_secs_f64() * 1000.0;
    if std::env::var_os("GLRMASK_PROFILE_L1_IMPLEMENTATIONS").is_some() {
        eprintln!(
            "[glrmask/profile][l1_projected] partition={} raw_states={} tokens={} trie_nodes={} configs={} signatures={} bucket_cache={} cache_hits={} pair_visits={} uniform_subtrees={} uniform_tokens={} profile_runs={} profiles={} state_classes={} token_classes={} run_sweep={} run_events={} referenced_runs={} vocab_cache_hit={} prep_ms={:.3} scan_ms={:.3} materialize_ms={:.3} compact_ms={:.3} build_ms={:.3} total_ms={:.3}",
            input.partition_label,
            input.tokenizer.num_states(),
            aliases.len(),
            trie.nodes.len(),
            scanner.configs.len(),
            scanner.signatures.len(),
            bucket_cache.len(),
            cache_hits,
            pair_visits,
            uniform_subtrees,
            uniform_tokens,
            profiles.iter().map(|profile| profile.len()).sum::<usize>(),
            profiles.len(),
            finished.state_classes,
            finished.token_classes,
            use_run_sweep,
            run_sweep_events,
            referenced_runs,
            finite_vocab_cache_hit,
            prep_ms,
            scan_ms,
            materialize_ms,
            finished.compact_ms,
            finished.build_ms,
            total.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Some(finished.artifact)
}

fn projected_limit_exceeded(input: BuildInput<'_>, states: usize, limit: usize) -> bool {
    if states <= limit {
        return false;
    }
    if std::env::var_os("GLRMASK_PROFILE_L1_IMPLEMENTATIONS").is_some() {
        eprintln!(
            "[glrmask/profile][l1_single_projected_abort] partition={} projected_states={} limit={} action=reject",
            input.partition_label, states, limit,
        );
    }
    true
}

fn projected_root_membership_precheck(input: BuildInput<'_>, threshold: usize) -> (usize, bool) {
    let mut memberships = 0usize;
    let mut visit = |state: u32| {
        memberships += super::super::collect_active_terminal_signature(
            input.tokenizer,
            state,
            input.active_terminals,
        )
        .len();
        memberships > threshold
    };
    if let Some(state_map) = input.initial_state_map {
        for &state in state_map
            .representative_original_ids
            .iter()
            .filter(|&&state| state != u32::MAX)
        {
            if visit(state) {
                return (memberships, true);
            }
        }
    } else {
        for state in 0..input.tokenizer.num_states() {
            if visit(state) {
                return (memberships, true);
            }
        }
    }
    (memberships, false)
}

fn residual_finite_switch_states(input: BuildInput<'_>) -> usize {
    if input.subset_parent_order.is_some()
        || input.partition_label == "single_terminal_global"
    {
        // The global one-terminal path has exactly one terminal signature. Its
        // residual projection stays compact even when the underlying tokenizer
        // has many states, while finite projection expands the full
        // state×vocabulary relation and can be orders of magnitude slower.
        // Keep this already-specialized path on the exact residual kernel.
        return usize::MAX;
    }
    let (env_name, default) = if input.partition_label == "p1" {
        // The precheck compares this with the number of live
        // (state,terminal-group) roots. Above a few thousand roots the residual
        // DFA can grow much faster than the finite-vocabulary projection. The
        // finite kernel has its own work budget and falls back to residual if
        // its trie traversal is the pathological side instead.
        ("GLRMASK_P1_RESIDUAL_FINITE_SWITCH_STATES", 3_500)
    } else if input.partition_label == "p5" && input.vocab.len() >= 4_000 {
        // Large p5 residuals can occasionally explode to O(100k) projected
        // roots even though ordinary/max-tail shapes remain around O(10k).
        // The existing membership precheck catches that topology before
        // constructing the residual automata and hands it to the exact finite
        // projected kernel instead.
        ("GLRMASK_P5_RESIDUAL_FINITE_SWITCH_STATES", 50_000)
    } else if input.vocab.len() >= 50_000 {
        // The original 9K handoff predated the sparse-root, grouped-minimize,
        // dense-reverse, and reverse-trie residual improvements. Revalidation
        // on the 128K ordinary corpus shows the residual kernel remains cheaper
        // through the ~20K projected-state regime; switching earlier throws
        // away useful residual work and makes the p2 tail substantially slower.
        ("GLRMASK_L1_RESIDUAL_FINITE_SWITCH_STATES", 20_000)
    } else {
        return usize::MAX;
    };
    std::env::var(env_name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

/// Vocab-only equivalence has a materially different residual/finite crossover
/// from Static because it never materializes the projected transition artifact.
/// Keep Static's established handoffs above unchanged and apply the larger
/// residual budget only to the dedicated vocabulary-equivalence calculation.
fn vocab_only_residual_finite_switch_states(input: BuildInput<'_>) -> usize {
    // On very large tokenizer machines the finite kernel's state×vocabulary
    // scan is itself the pathological side.  The dedicated vocab-only path
    // does not need a projected transition artifact, and the residual kernel
    // remains substantially cheaper for the large p1/p2/p5 families while
    // producing the same exact token partition.  Keep the ordinary small and
    // medium-tokenizer crossovers below unchanged; this guard only removes the
    // stale finite safety handoff in the large-state regime.
    if input.subset_parent_order.is_none()
        && matches!(input.partition_label, "p1" | "p2" | "p5")
        && input.tokenizer.num_states() > 50_000
    {
        return usize::MAX;
    }
    let (env_name, default) = if input.subset_parent_order.is_none()
        && input.partition_label == "p1"
        && input.vocab.len() >= 10_000
        && input.tokenizer.terminal_residual_coordinates().is_some()
    {
        // Vocab-only p1 has the same shifted crossover as p2/p5. Static's
        // 3.5k residual ceiling is tuned for materializing a downstream DWA;
        // here we only need the token quotient. Restrict the larger budget to
        // tokenizers exposing exact terminal-residual coordinates: those are
        // the direct-mask families where the residual construction is cheap.
        // Ordinary tokenizers keep Static's established crossover. Keep a
        // 200k projected-state safety ceiling so genuinely explosive residual
        // machines still fall back to the exact finite kernel.
        ("GLRMASK_VOCAB_P1_RESIDUAL_FINITE_SWITCH_STATES", 200_000)
    } else if input.subset_parent_order.is_none()
        && input.partition_label == "p2"
        && input.vocab.len() >= 50_000
    {
        // Revalidated on the canonical seed-7 sweep after the sparse minimizer
        // and live-target reverse-classifier changes: all observed p2 machines
        // in the 20k..178k projected-state range are substantially faster as
        // residual projections. Retain a finite safety handoff above that band.
        ("GLRMASK_VOCAB_P2_RESIDUAL_FINITE_SWITCH_STATES", 200_000)
    } else if input.subset_parent_order.is_none()
        && input.partition_label == "p5"
        && input.vocab.len() >= 4_000
    {
        // Tail-focused revalidation found the same crossover shift for p5:
        // machines throughout the observed 143k..178k range are neutral or
        // faster as residual projections, while >200k machines still fall back
        // to finite. This policy is intentionally vocab-only.
        ("GLRMASK_VOCAB_P5_RESIDUAL_FINITE_SWITCH_STATES", 200_000)
    } else {
        return residual_finite_switch_states(input);
    };
    std::env::var(env_name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn build_binary(input: BuildInput<'_>) -> Option<LocalIdMapTerminalDwa> {
    build_binary_impl(input, true)
}

fn build_binary_without_finite_switch(input: BuildInput<'_>) -> Option<LocalIdMapTerminalDwa> {
    build_binary_impl(input, false)
}

fn build_binary_impl(input: BuildInput<'_>, allow_finite_switch: bool) -> Option<LocalIdMapTerminalDwa> {
    if input.vocab.is_empty() {
        return None;
    }
    let total = Instant::now();
    let finite_switch_states = if allow_finite_switch { residual_finite_switch_states(input) } else { usize::MAX };
    let setup_started = Instant::now();
    let use_finite_precheck = allow_finite_switch && std::env::var("GLRMASK_L1_RESIDUAL_FINITE_PRECHECK")
        .map(|value| {
            let value = value.trim();
            value.is_empty() || value == "1" || value.eq_ignore_ascii_case("true")
        })
        .unwrap_or(true);
    if use_finite_precheck && finite_switch_states != usize::MAX {
        let p2_unbounded_max_tokenizer_states = std::env::var(
            "GLRMASK_P2_RESIDUAL_FINITE_UNBOUNDED_MAX_TOKENIZER_STATES",
        )
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(50_000);
        // The unbounded finite p2 handoff is a win for the compact Kubernetes
        // p99 topology, where a few thousand tokenizer states fan out into a
        // huge residual-root set.  Very large tokenizer machines are the
        // opposite shape: on o21135-37 (~78k local states), the finite scan is
        // itself enormous and can take seconds longer than residual.  Do not
        // even pay the bounded finite probe there; retain the exact residual
        // kernel directly.
        if input.partition_label == "p2"
            && input.subset_parent_order.is_none()
            && input.tokenizer.num_states() > p2_unbounded_max_tokenizer_states
        {
            if std::env::var_os("GLRMASK_PROFILE_L1_IMPLEMENTATIONS").is_some() {
                eprintln!(
                    "[glrmask/profile][l1_residual_finite_precheck] partition=p2 tokenizer_states={} max_unbounded_tokenizer_states={} selected=false reason=large_tokenizer_residual",
                    input.tokenizer.num_states(),
                    p2_unbounded_max_tokenizer_states,
                );
            }
        } else {
        let unbounded_p2_memberships = (input.partition_label == "p2").then(|| {
            std::env::var("GLRMASK_P2_RESIDUAL_FINITE_UNBOUNDED_MEMBERSHIPS")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(50_000)
        });
        // For p2, probe far enough to distinguish a merely-large residual
        // (which should keep the existing bounded finite attempt) from the
        // pathological 100k+ root regime where falling back to residual is
        // much more expensive than letting finite projection complete.
        let probe_threshold = unbounded_p2_memberships
            .map(|threshold| threshold.max(finite_switch_states))
            .unwrap_or(finite_switch_states);
        let precheck_started = Instant::now();
        let (memberships, exceeds_probe_threshold) =
            projected_root_membership_precheck(input, probe_threshold);
        let exceeds_threshold = exceeds_probe_threshold || memberships > finite_switch_states;
        let exceeds_unbounded_p2 = unbounded_p2_memberships
            .is_some_and(|threshold| exceeds_probe_threshold || memberships > threshold);
        let precheck_ms = precheck_started.elapsed().as_secs_f64() * 1000.0;
        if std::env::var_os("GLRMASK_PROFILE_L1_IMPLEMENTATIONS").is_some() {
            eprintln!(
                "[glrmask/profile][l1_residual_finite_precheck] partition={} memberships={} threshold={} probe_threshold={} selected={} unbounded={} precheck_ms={:.3}",
                input.partition_label,
                memberships,
                finite_switch_states,
                probe_threshold,
                exceeds_threshold,
                exceeds_unbounded_p2,
                precheck_ms,
            );
        }
        if exceeds_threshold {
            if exceeds_unbounded_p2 {
                return build_finite_projected_unbounded(input);
            }
            return build_finite_projected(input);
        }
        }
    }
    // Full-partition vocabularies at or above the finite-preparation threshold
    // already have exact deduplicated tokens/aliases cached before grammar
    // compilation. Reuse that immutable artifact rather than rebuilding the
    // same vectors on every residual compile. Split-L1 keeps its subset-specific
    // filtered order.
    let reuse_prepared_vocab = std::env::var("GLRMASK_L1_RESIDUAL_REUSE_PREPARED_VOCAB")
        .map(|value| {
            let value = value.trim();
            value.is_empty() || (value != "0" && !value.eq_ignore_ascii_case("false"))
        })
        .unwrap_or(true);
    let prepared_vocab_min_tokens = residual_prepared_vocab_min_tokens(input);
    let vocab_projection_started = Instant::now();
    let prepared_vocab = (reuse_prepared_vocab
        && input.subset_parent_order.is_none()
        && input.vocab.len() >= prepared_vocab_min_tokens)
        .then(|| build_finite_vocab_projection(input.vocab));
    let vocab_projection_ms = vocab_projection_started.elapsed().as_secs_f64() * 1000.0;
    let owned_vocab;
    let (aliases, tokens): (&[Vec<u32>], &[Arc<[u8]>]) = if let Some(prepared) = prepared_vocab.as_ref() {
        (prepared.aliases.as_slice(), prepared.tokens.as_slice())
    } else {
        owned_vocab = unique_vocab(input);
        (owned_vocab.0.as_slice(), owned_vocab.1.as_slice())
    };
    if tokens.iter().all(|token| token.len() == 1) {
        return build_one_byte(input, aliases, tokens, total);
    }
    let byte_setup_started = Instant::now();
    let vocab_bytes = if prepared_vocab.is_some() {
        let mut bytes = input.vocab.relevant_bytes().iter().copied().collect::<Vec<_>>();
        bytes.sort_unstable();
        bytes
    } else {
        let mut relevant = [false; 256];
        for token in tokens {
            for &byte in token.iter() {
                relevant[byte as usize] = true;
            }
        }
        relevant
            .iter()
            .enumerate()
            .filter_map(|(byte, &used)| used.then_some(byte as u8))
            .collect::<Vec<_>>()
    };
    let (bytes, input_byte_representative) = quotient_input_bytes(input, &vocab_bytes);
    let byte_setup_ms = byte_setup_started.elapsed().as_secs_f64() * 1000.0;
    let setup_ms = setup_started.elapsed().as_secs_f64() * 1000.0;

    let projected_started = Instant::now();
    let broad_id_map_p2_direct = input.id_map_only
        && input.partition_label == "p2"
        && input.active_terminals.iter().filter(|&&active| active).count() >= 64;
    let direct_terminal_residuals = std::env::var("GLRMASK_L1_DIRECT_TERMINAL_RESIDUALS")
        .map(|value| {
            let value = value.trim();
            value.is_empty() || (value != "0" && !value.eq_ignore_ascii_case("false"))
        })
        .unwrap_or(
            input.id_map_only
                && (input.partition_label == "p1" || broad_id_map_p2_direct),
        )
        || std::env::var("GLRMASK_L1_DIRECT_TERMINAL_RESIDUALS_PARTITIONS")
            .ok()
            .is_some_and(|scope| {
                scope
                    .split(',')
                    .map(str::trim)
                    .any(|label| label == input.partition_label)
            });
    let projected_new_started = Instant::now();
    let direct_machine = direct_terminal_residuals
        .then(|| build_direct_terminal_residual_machine(input, &bytes))
        .flatten();
    let direct_selected = direct_machine.is_some();
    let (mut projected, direct_roots) = if let Some((projected, roots)) = direct_machine {
        (projected, Some(roots))
    } else {
        (Projected::new(input), None)
    };
    let projected_new_ms = projected_new_started.elapsed().as_secs_f64() * 1000.0;
    let projected_roots_started = Instant::now();
    let mut roots = if let Some(roots) = direct_roots {
        roots
    } else {
        // `initial_state_map` is already an exact certified quotient for this
        // L1 branch. Construct projected rows only for quotient representatives,
        // then fan their sparse memberships back to the raw coordinate.
        let root_rows = if let Some(state_map) = input.initial_state_map {
            debug_assert_eq!(
                state_map.original_to_internal.len(),
                input.tokenizer.num_states() as usize
            );
            let representative_rows = state_map
                .representative_original_ids
                .iter()
                .map(|&raw| {
                    assert_ne!(raw, u32::MAX, "L1 state quotient has an unmapped representative");
                    projected.root_sparse_row(raw)
                })
                .collect::<Vec<_>>();
            state_map
                .original_to_internal
                .iter()
                .enumerate()
                .map(|(raw, &class)| {
                    if class == u32::MAX {
                        if input.initial_state_domain_is_exact {
                            Vec::new()
                        } else {
                            projected.root_sparse_row(raw as u32)
                        }
                    } else {
                        representative_rows[class as usize].clone()
                    }
                })
                .collect::<Vec<_>>()
        } else {
            (0..input.tokenizer.num_states())
                .map(|raw| projected.root_sparse_row(raw))
                .collect::<Vec<_>>()
        };
        SparseRoots::from_rows(root_rows)
    };
    let projected_roots_ms = projected_roots_started.elapsed().as_secs_f64() * 1000.0;
    let projected_expand_started = Instant::now();
    if !direct_selected && projected.configs.len() > finite_switch_states {
        if std::env::var_os("GLRMASK_PROFILE_L1_IMPLEMENTATIONS").is_some() {
            eprintln!(
                "[glrmask/profile][l1_residual_finite_switch] partition={} projected_states={} threshold={} phase=roots action=finite",
                input.partition_label,
                projected.configs.len(),
                finite_switch_states,
            );
        }
        return build_finite_projected(input);
    }
    let limit = std::env::var("GLRMASK_L1_SINGLE_MAX_STATES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(2_000_000usize);
    if !direct_selected && projected_limit_exceeded(input, projected.configs.len(), limit) {
        // None denotes an empty relation, not a failed build. Finite projection
        // may already have sent us here after its own budget was exhausted.
        // Report a typed failure rather than retrying or returning partial data.
        glrmask_invariant::__private::fail_compilation_resource_limit(format!(
            "projected L1 partition {:?} has {} states, exceeding GLRMASK_L1_SINGLE_MAX_STATES={}",
            input.partition_label, projected.configs.len(), limit,
        ));
    }
    let expanded = if direct_selected {
        projected.configs.len()
    } else {
        let mut queue = VecDeque::from_iter(0..projected.configs.len() as u32);
        let mut expanded = 0usize;
        while let Some(state) = queue.pop_front() {
            let mut row = Vec::new();
            for (symbol, &byte) in bytes.iter().enumerate() {
                let before = projected.configs.len();
                let target = projected.step(state, byte, &roots);
                if target != DEAD {
                    row.push((symbol as u8, target));
                }
                if projected.configs.len() > before {
                    queue.extend(before as u32..projected.configs.len() as u32);
                    if projected.configs.len() > finite_switch_states {
                        if std::env::var_os("GLRMASK_PROFILE_L1_IMPLEMENTATIONS").is_some() {
                            eprintln!(
                                "[glrmask/profile][l1_residual_finite_switch] partition={} projected_states={} threshold={} action=finite",
                                input.partition_label,
                                projected.configs.len(),
                                finite_switch_states,
                            );
                        }
                        return build_finite_projected(input);
                    }
                    if projected_limit_exceeded(input, projected.configs.len(), limit) {
                        return build_finite_projected(input);
                    }
                }
            }
            projected.transitions[state as usize] = row;
            expanded += 1;
        }
        expanded
    };
    let projected_expand_ms = projected_expand_started.elapsed().as_secs_f64() * 1000.0;
    let projected_ms = projected_started.elapsed().as_secs_f64() * 1000.0;
    if std::env::var_os("GLRMASK_PROFILE_L1_IMPLEMENTATIONS").is_some() {
        eprintln!(
            "[glrmask/profile][l1_projected_residual_breakdown] partition={} direct_terminal_residuals={} new_ms={:.3} roots_ms={:.3} expand_ms={:.3} total_project_ms={:.3}",
            input.partition_label,
            direct_selected,
            projected_new_ms,
            projected_roots_ms,
            projected_expand_ms,
            projected_ms,
        );
    }

    let minimize_started = Instant::now();
    let groups = projected
        .configs
        .iter()
        .map(|(group, _)| *group)
        .collect::<Vec<_>>();
    let use_grouped_minimize = std::env::var("GLRMASK_L1_PROJECTED_GROUPED_MINIMIZE")
        .map(|value| {
            let value = value.trim();
            value.is_empty() || (value != "0" && !value.eq_ignore_ascii_case("false"))
        })
        .unwrap_or(true);
    let (mut minimized, grouped_minimize) = if direct_selected {
        (minimize_direct_residual_union(&projected.transitions, &bytes), None)
    } else if use_grouped_minimize {
        let (minimized, stats) = minimize_grouped(&projected.transitions, &groups, &bytes);
        (minimized, Some(stats))
    } else {
        (minimize(&projected.transitions, &bytes), None)
    };
    for &byte in &vocab_bytes {
        minimized.byte_class[byte as usize] =
            minimized.byte_class[input_byte_representative[byte as usize] as usize];
    }
    let minimize_ms = minimize_started.elapsed().as_secs_f64() * 1000.0;
    roots.remap_states(&minimized.classes);

    let traverse_started = Instant::now();
    let root_vectors_started = Instant::now();
    // Full residual-root vectors exactly preclassify raw lexer states.  Roots
    // are immutable from here on, so intern borrowed slices and retain only one
    // raw representative per distinct vector.  The previous path cloned every
    // 200+ entry vector into both the hash table and a second owned store.
    let mut root_vector_ids = FxHashMap::<&[(u32, u32)], u32>::default();
    let mut root_vector_reps = Vec::<u32>::new();
    let mut raw_root_class = vec![0u32; roots.len()];
    for raw in 0..roots.len() {
        let root_row = roots.row(raw);
        raw_root_class[raw] = if let Some(&class) = root_vector_ids.get(root_row) {
            class
        } else {
            let class = root_vector_reps.len() as u32;
            root_vector_ids.insert(root_row, class);
            root_vector_reps.push(raw as u32);
            class
        };
    }

    let mut used_classes = roots.entries.iter().map(|&(_, state)| state).collect::<Vec<_>>();
    used_classes.sort_unstable();
    used_classes.dedup();
    let mut used_index = vec![usize::MAX; minimized.state_count];
    for (index, &state) in used_classes.iter().enumerate() {
        used_index[state as usize] = index;
    }
    let root_vectors_ms = root_vectors_started.elapsed().as_secs_f64() * 1000.0;

    let reverse_started = Instant::now();
    let force_reverse_source_scan = std::env::var("GLRMASK_L1_RESIDUAL_REVERSE_SOURCE_SCAN")
        .ok()
        .is_some_and(|scope| {
            let scope = scope.trim();
            scope.is_empty()
                || scope == "1"
                || scope.split(',').map(str::trim).any(|label| label == input.partition_label)
        });
    let mut reverse = ReverseSubsets::new(
        &minimized.columns,
        &minimized.byte_class,
        minimized.state_count,
        force_reverse_source_scan,
    );
    let mut final_subset_to_class = Vec::<u32>::new();
    let mut class_subsets = Vec::<u32>::new();
    let mut token_class = vec![0u32; aliases.len()];
    let token_bytes = tokens.iter().map(|token| token.len()).sum::<usize>();
    let dense_reverse_enabled = std::env::var("GLRMASK_L1_RESIDUAL_DENSE_REVERSE")
        .map(|value| {
            let value = value.trim();
            value.is_empty() || (value != "0" && !value.eq_ignore_ascii_case("false"))
        })
        .unwrap_or(input.partition_label == "p2" && input.subset_parent_order.is_none());
    let use_reverse_trie = std::env::var("GLRMASK_L1_RESIDUAL_REVERSE_TRIE")
        .map(|value| {
            let value = value.trim();
            value.is_empty() || (value != "0" && !value.eq_ignore_ascii_case("false"))
        })
        .unwrap_or(true);
    let reverse_trie = use_reverse_trie
        .then_some(())
        .and_then(|()| prepared_vocab.as_ref())
        .and_then(|prepared| prepared.reverse_trie.as_ref());
    let dense_reverse_max_states = std::env::var("GLRMASK_L1_RESIDUAL_DENSE_REVERSE_MAX_STATES")
        .ok()
        .and_then(|value| value.parse().ok())
        // Dense completion is a speculative acceleration for the lazy exact reverse
        // walk. Compact p2 residuals routinely close in the low hundreds of states;
        // completing those machines removes mutable subset interning from the much
        // larger vocabulary traversal. Stop at 192 so larger machines still fall
        // back before speculative completion grows materially.
        .unwrap_or(192usize);
    let reverse_complete_started = Instant::now();
    let dense_reverse = dense_reverse_enabled
        .then(|| reverse.try_complete_dense(dense_reverse_max_states))
        .flatten();
    let reverse_complete_ms = reverse_complete_started.elapsed().as_secs_f64() * 1000.0;
    let reverse_classify_started = Instant::now();

    match (dense_reverse.as_ref(), reverse_trie) {
        (Some(dense), Some(trie)) => {
            debug_assert_eq!(trie.parents.len(), trie.bytes.len());
            let symbols = reverse.symbol_count;
            let mut subset_at_node = vec![0u32; trie.parents.len()];
            for node in 1..trie.parents.len() {
                let parent = trie.parents[node] as usize;
                let symbol = reverse.byte_class[trie.bytes[node] as usize] as usize;
                subset_at_node[node] = dense[subset_at_node[parent] as usize * symbols + symbol];
            }
            for (token_index, &node) in trie.token_node.iter().enumerate() {
                let subset = subset_at_node[node as usize];
                token_class[token_index] = intern_dense_subset_class(
                    subset,
                    &mut final_subset_to_class,
                    &mut class_subsets,
                );
            }
        }
        (Some(dense), None) => {
            let symbols = reverse.symbol_count;
            for (token_index, token) in tokens.iter().enumerate() {
                let subset = token.iter().rev().fold(0u32, |subset, &byte| {
                    let symbol = reverse.byte_class[byte as usize] as usize;
                    dense[subset as usize * symbols + symbol]
                });
                token_class[token_index] = intern_dense_subset_class(
                    subset,
                    &mut final_subset_to_class,
                    &mut class_subsets,
                );
            }
        }
        (None, Some(trie)) => {
            debug_assert_eq!(trie.parents.len(), trie.bytes.len());
            let mut subset_at_node = vec![0u32; trie.parents.len()];
            for node in 1..trie.parents.len() {
                let parent = trie.parents[node] as usize;
                subset_at_node[node] = reverse.prepend(subset_at_node[parent], trie.bytes[node]);
            }
            for (token_index, &node) in trie.token_node.iter().enumerate() {
                let subset = subset_at_node[node as usize];
                token_class[token_index] = intern_dense_subset_class(
                    subset,
                    &mut final_subset_to_class,
                    &mut class_subsets,
                );
            }
        }
        (None, None) => {
            for (token_index, token) in tokens.iter().enumerate() {
                let subset = reverse.token(token);
                token_class[token_index] = intern_dense_subset_class(
                    subset,
                    &mut final_subset_to_class,
                    &mut class_subsets,
                );
            }
        }
    }
    let reverse_classify_ms = reverse_classify_started.elapsed().as_secs_f64() * 1000.0;
    let reverse_ms = reverse_started.elapsed().as_secs_f64() * 1000.0;
    let incidence_started = Instant::now();
    let mut live_token_classes = vec![Vec::<u32>::new(); used_classes.len()];
    for (token_class, &subset) in class_subsets.iter().enumerate() {
        let set = &reverse.sets[subset as usize];
        for (used, &state) in used_classes.iter().enumerate() {
            if ReverseSubsets::contains(set, state) {
                live_token_classes[used].push(token_class as u32);
            }
        }
    }
    let incidence_ms = incidence_started.elapsed().as_secs_f64() * 1000.0;

    let signature_started = Instant::now();
    // Each minimized residual state is observable only through the sorted set
    // of residual token classes that can reach it.  Intern those exact token
    // profiles once.  A root relation is then determined exactly by one profile
    // ID per terminal-predicate group, because those groups partition the fixed
    // terminal identities.  Classify all root vectors using this compact key and
    // materialize the full (terminal, token-class) relation only for one
    // representative of each final L1 state class.
    let mut token_profile_ids = FxHashMap::<&[u32], u32>::default();
    let mut profile_for_used_state = vec![0u32; live_token_classes.len()];
    let mut next_profile = 1u32; // zero denotes the empty/dead profile
    for (used, token_classes) in live_token_classes.iter().enumerate() {
        if token_classes.is_empty() {
            continue;
        }
        let profile = if let Some(&profile) = token_profile_ids.get(token_classes.as_slice()) {
            profile
        } else {
            let profile = next_profile;
            next_profile += 1;
            token_profile_ids.insert(token_classes.as_slice(), profile);
            profile
        };
        profile_for_used_state[used] = profile;
    }

    let group_count = projected.active_by_group.len();
    let mut state_class_ids = FxHashMap::<Box<[(u32, u32)]>, u32>::default();
    let mut state_class_representative_root = Vec::<u32>::new();
    let mut root_to_state_class = vec![0u32; root_vector_reps.len()];
    let mut key = Vec::<(u32, u32)>::new();
    for (root_class, &raw_rep) in root_vector_reps.iter().enumerate() {
        key.clear();
        for &(group, state) in roots.row(raw_rep as usize) {
            let profile = profile_for_used_state[used_index[state as usize]];
            if profile != 0 {
                key.push((group, profile));
            }
        }
        let class = if let Some(&class) = state_class_ids.get(key.as_slice()) {
            class
        } else {
            let class = state_class_representative_root.len() as u32;
            state_class_ids.insert(key.clone().into_boxed_slice(), class);
            state_class_representative_root.push(root_class as u32);
            class
        };
        root_to_state_class[root_class] = class;
    }

    let mut sparse_rows = Vec::<Vec<(u32, u32)>>::with_capacity(state_class_representative_root.len());
    let mut sparse_pair_visits = 0usize;
    let mut group_state = vec![DEAD; group_count];
    for &root_class in &state_class_representative_root {
        let raw_rep = root_vector_reps[root_class as usize];
        let root_row = roots.row(raw_rep as usize);
        for &(group, state) in root_row {
            group_state[group as usize] = state;
        }
        let mut sparse = Vec::<(u32, u32)>::new();
        for (terminal_index, &group) in projected.terminal_groups.iter().enumerate() {
            let state = group_state[group];
            if state == DEAD {
                continue;
            }
            for &token in &live_token_classes[used_index[state as usize]] {
                sparse.push((projected.terminals[terminal_index], token));
            }
        }
        for &(group, _) in root_row {
            group_state[group as usize] = DEAD;
        }
        sparse_pair_visits += sparse.len();
        sparse_rows.push(sparse);
    }

    // The sparse relation is already the final semantic row. No dense
    // token-class matrix or terminal-set signature interning is necessary.
    let signature_updates = 0usize;
    let state_class = raw_root_class
        .into_iter()
        .map(|class| root_to_state_class[class as usize])
        .collect::<Vec<_>>();
    let sparse_row_count = sparse_rows.len();
    let signature_count = 0usize;
    let signature_ms = signature_started.elapsed().as_secs_f64() * 1000.0;
    let traverse_ms = traverse_started.elapsed().as_secs_f64() * 1000.0;
    let finish_started = Instant::now();
    let preordered_vocab = preordered_vocab_map_enabled(input)
        .then(|| {
            prepared_vocab
                .as_ref()
                .and_then(|prepared| prepared.original_order.as_deref())
        })
        .flatten();
    let finished = common::finish_sparse_terminal_rows(
        input,
        aliases,
        state_class,
        sparse_rows,
        token_class,
        preordered_vocab,
        projected_ms + minimize_ms + traverse_ms,
        0.0,
        || total.elapsed().as_secs_f64() * 1000.0,
    )?;
    let finish_ms = finish_started.elapsed().as_secs_f64() * 1000.0;
    if std::env::var_os("GLRMASK_PROFILE_L1_IMPLEMENTATIONS").is_some() {
        let live_edges = projected.transitions.iter().map(Vec::len).sum::<usize>();
        let max_live_edges = projected.transitions.iter().map(Vec::len).max().unwrap_or(0);
        let singleton_configs = projected
            .configs
            .iter()
            .filter(|(_, states)| states.len() == 1)
            .count();
        let total_config_states = projected
            .configs
            .iter()
            .map(|(_, states)| states.len())
            .sum::<usize>();
        let max_config_states = projected
            .configs
            .iter()
            .map(|(_, states)| states.len())
            .max()
            .unwrap_or(0);
        eprintln!(
            "[glrmask/profile][l1_single] partition={} raw_states={} terminals={} terminal_groups={} vocab_bytes={} input_byte_classes={} minimized_bytes={} minimize_rounds={} grouped_local_states={} grouped_local_rounds={} grouped_local_ms={:.3} grouped_local_max_group_ms={:.3} grouped_global_ms={:.3} dag_groups={} dag_states={} hopcroft_groups={} hopcroft_states={} projected_states={} live_edges={} max_live_edges={} singleton_configs={} total_config_states={} max_config_states={} expanded={} minimized_states={} root_closure_states={} root_memberships={} root_classes={} root_vectors={} residual_token_classes={} sparse_rows={} sparse_pairs={} signature_updates={} reverse_states={} reverse_transitions={} reverse_target_visits={} reverse_predecessor_visits={} token_bytes={} signatures={} state_classes={} token_classes={} vocab_projection_ms={:.3} byte_setup_ms={:.3} setup_ms={:.3} project_ms={:.3} minimize_ms={:.3} root_vectors_ms={:.3} reverse_ms={:.3} reverse_complete_ms={:.3} reverse_classify_ms={:.3} incidence_ms={:.3} signature_ms={:.3} traverse_ms={:.3} finish_ms={:.3} compact_ms={:.3} build_ms={:.3} total_ms={:.3}",
            input.partition_label,
            input.tokenizer.num_states(),
            projected.terminals.len(),
            projected.active_by_group.len(),
            vocab_bytes.len(),
            bytes.len(),
            minimized.columns.len(),
            minimized.rounds,
            grouped_minimize.as_ref().map_or(0, |stats| stats.local_states),
            grouped_minimize.as_ref().map_or(0, |stats| stats.local_rounds),
            grouped_minimize.as_ref().map_or(0.0, |stats| stats.local_ms),
            grouped_minimize.as_ref().map_or(0.0, |stats| stats.local_max_group_ms),
            grouped_minimize.as_ref().map_or(0.0, |stats| stats.global_ms),
            grouped_minimize.as_ref().map_or(0, |stats| stats.dag_groups),
            grouped_minimize.as_ref().map_or(0, |stats| stats.dag_states),
            grouped_minimize.as_ref().map_or(0, |stats| stats.hopcroft_groups),
            grouped_minimize.as_ref().map_or(0, |stats| stats.hopcroft_states),
            projected.configs.len(),
            live_edges,
            max_live_edges,
            singleton_configs,
            total_config_states,
            max_config_states,
            expanded,
            minimized.state_count,
            projected.root_closure_states,
            projected.root_memberships,
            used_classes.len(),
            root_vector_reps.len(),
            class_subsets.len(),
            sparse_row_count,
            sparse_pair_visits,
            signature_updates,
            reverse.sets.len(),
            reverse.computed_transitions,
            reverse.target_visits,
            reverse.predecessor_visits,
            token_bytes,
            signature_count,
            finished.state_classes,
            finished.token_classes,
            vocab_projection_ms,
            byte_setup_ms,
            setup_ms,
            projected_ms,
            minimize_ms,
            root_vectors_ms,
            reverse_ms,
            reverse_complete_ms,
            reverse_classify_ms,
            incidence_ms,
            signature_ms,
            traverse_ms,
            finish_ms,
            finished.compact_ms,
            finished.build_ms,
            total.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Some(finished.artifact)
}


#[derive(Clone, Debug)]
pub struct L1VocabEquivResult {
    pub vocab_map: ManyToOneIdMap,
    pub state_classes: usize,
    pub token_classes: usize,
    pub prep_ms: f64,
    pub prep_cpu_ms: f64,
    pub scan_ms: f64,
    pub scan_cpu_ms: f64,
    pub compact_ms: f64,
    pub compact_cpu_ms: f64,
    pub total_wall_ms: f64,
    pub total_cpu_ms: f64,
    pub run_sweep_events: usize,
    pub referenced_runs: usize,
    pub kernel: &'static str,
}

pub fn build_projected_vocab_equivalence(input: BuildInput<'_>) -> Option<L1VocabEquivResult> {
    let active_terminals = input.active_terminals.iter().filter(|&&active| active).count();
    // Vocab-only equivalence has a different cost model from Static: there is
    // no downstream DWA construction to amortize an expensive finite scan.
    // Wide L1 families benefit from the finite trie scan; narrow families can
    // be much cheaper through residual projection even when Static's adaptive
    // residual path would switch back to finite for downstream reasons.
    if input.subset_parent_order.is_none()
        && input.partition_label == "p1"
        && active_terminals <= 96
        && input.vocab.len() >= 10_000
    {
        return build_binary_vocab_only_forced(input);
    }
    if input.subset_parent_order.is_none()
        && input.partition_label == "p2"
        && active_terminals <= 500
        && input.vocab.len() >= 50_000
    {
        // Static's residual->finite handoff is tuned for the downstream DWA
        // build. For vocab-only equivalence, medium-width p2 families can be
        // dramatically cheaper as residual projections even when they cross
        // that handoff. Use the existing cheap root-membership probe to keep
        // genuinely explosive residuals on the finite path.
        const VOCAB_ONLY_P2_RESIDUAL_MAX_ROOT_MEMBERSHIPS: usize = 20_000;
        let (memberships, exceeded) = projected_root_membership_precheck(
            input,
            VOCAB_ONLY_P2_RESIDUAL_MAX_ROOT_MEMBERSHIPS,
        );
        if !exceeded && memberships <= VOCAB_ONLY_P2_RESIDUAL_MAX_ROOT_MEMBERSHIPS {
            return build_binary_vocab_only_forced(input);
        }
    }
    if input.subset_parent_order.is_none() && input.partition_label == "p4" {
        // Vocab-only P4 (Unicode-only alpha) is substantially cheaper through
        // exact residual projection even for wide terminal families.  The old
        // active-terminal crossover forced a full finite-vocabulary scan in
        // precisely the cases where the residual relation often collapses to a
        // tiny quotient.
        return build_binary_vocab_only_forced(input);
    }
    // VocabPartition does not need the projected state-transition artifact.
    // When exact terminal-residual coordinates are available, start p5 on the
    // residual kernel and let the vocab-only residual/finite safety ceiling
    // decide whether the resulting machine is still economical. Static keeps
    // its separate, lower crossover in `residual_finite_switch_states`.
    let direct_terminal_residuals_available = input
        .tokenizer
        .terminal_residual_coordinates()
        .is_some()
        && std::env::var("GLRMASK_L1_DIRECT_TERMINAL_RESIDUALS")
            .map(|value| {
                let value = value.trim();
                value.is_empty() || (value != "0" && !value.eq_ignore_ascii_case("false"))
            })
            .unwrap_or(true);
    let force_p5_residual = input.partition_label == "p5"
        && (direct_terminal_residuals_available
            || std::env::var("GLRMASK_VOCAB_P5_FORCE_RESIDUAL")
                .map(|value| {
                    let value = value.trim();
                    value.is_empty() || (value != "0" && !value.eq_ignore_ascii_case("false"))
                })
                .unwrap_or(false));
    let kernel = if force_p5_residual {
        ProjectedKernel::Residual
    } else {
        projected_kernel(input)
    };
    match kernel {
        ProjectedKernel::Finite => build_finite_projected_vocab_only(input, false),
        ProjectedKernel::Residual => build_binary_vocab_only(input),
    }
}

fn build_finite_projected_vocab_only(
    input: BuildInput<'_>,
    bypass_adaptive_budget: bool,
) -> Option<L1VocabEquivResult> {
    if input.vocab.is_empty() {
        return None;
    }
    let profile = std::env::var_os("GLRMASK_PROFILE_L1_IMPLEMENTATIONS").is_some();
    let total_timer = Instant::now();
    let prep_timer = Instant::now();
    let (finite_vocab, _finite_vocab_cache_hit, _prep_ms) = finite_vocab_projection(input);
    let prep_wall_ms = prep_timer.elapsed().as_secs_f64() * 1000.0;
    let prep_cpu_ms = 0.0;
    let aliases = finite_vocab.aliases.as_slice();
    let tokens = finite_vocab.tokens.as_slice();
    let trie = &finite_vocab.trie;

    let scan_timer = Instant::now();
    let mut scanner = Scanner::new(input);
    let mut starts = BTreeMap::<u32, Vec<u32>>::new();
    if let Some(state_map) = input.initial_state_map {
        for (class, &representative) in state_map.representative_original_ids.iter().enumerate() {
            if representative == u32::MAX {
                continue;
            }
            let start = scanner.start(representative);
            starts
                .entry(start)
                .or_default()
                .extend_from_slice(&state_map.internal_to_originals[class]);
        }
        if !input.initial_state_domain_is_exact {
            for (raw, &class) in state_map.original_to_internal.iter().enumerate() {
                if class == u32::MAX {
                    starts
                        .entry(scanner.start(raw as u32))
                        .or_default()
                        .push(raw as u32);
                }
            }
        }
    } else {
        for raw in 0..input.tokenizer.num_states() {
            starts.entry(scanner.start(raw)).or_default().push(raw);
        }
    }

    let root_token = trie.nodes[0].token;
    let root_children = trie.nodes[0].children.clone();
    let mut state_class = vec![0u32; input.tokenizer.num_states() as usize];
    let mut class_fingerprints = Vec::<Vec<u32>>::new();
    let mut class_ids = FxHashMap::<Vec<u32>, u32>::default();

    let mut profiles = vec![Arc::<[ProfileRun]>::from([])];
    let mut profile_ids = FxHashMap::<Arc<[ProfileRun]>, u32>::default();
    let mut bucket_cache = FxHashMap::<(u32, u32), u32>::default();
    let mut cache_hits = 0usize;
    let mut pair_visits = 0usize;
    let mut uniform_subtrees = 0usize;
    let mut uniform_tokens = 0usize;
    let self_loops = input.tokenizer.all_self_loop_bytes();
    let adaptive_budget = !bypass_adaptive_budget
        && input.subset_parent_order.is_none()
        && matches!(input.partition_label, "p1" | "p2");
    let (profile_run_budget, pair_visit_budget) = if adaptive_budget {
        let (runs_env, pairs_env, default_pairs) = if input.partition_label == "p1" {
            ("GLRMASK_P1_FINITE_MAX_PROFILE_RUNS", "GLRMASK_P1_FINITE_MAX_PAIR_VISITS", 50_000)
        } else {
            ("GLRMASK_P2_FINITE_MAX_PROFILE_RUNS", "GLRMASK_P2_FINITE_MAX_PAIR_VISITS", 150_000)
        };
        (
            std::env::var(runs_env).ok().and_then(|v| v.parse().ok()).unwrap_or(2_048),
            std::env::var(pairs_env).ok().and_then(|v| v.parse().ok()).unwrap_or(default_pairs),
        )
    } else {
        (usize::MAX, usize::MAX)
    };
    let mut profile_run_count = 0usize;

    let parallel_profile_override = std::env::var("GLRMASK_L1_FINITE_PARALLEL_PROFILES").ok();
    let force_parallel_profiles = parallel_profile_override
        .as_deref()
        .is_some_and(|value| value.trim().eq_ignore_ascii_case("force"));
    let parallel_profiles_enabled = parallel_profile_override
        .as_deref()
        .map(|value| {
            let value = value.trim();
            value.is_empty()
                || value.eq_ignore_ascii_case("force")
                || (value != "0" && !value.eq_ignore_ascii_case("false"))
        })
        .unwrap_or(true);
    let parallel_profiles = (bypass_adaptive_budget || force_parallel_profiles)
        && input.subset_parent_order.is_none()
        && input.vocab.len() >= 10_000
        && input.tokenizer.num_states() >= 10_000
        && rayon::current_num_threads() > 1
        && parallel_profiles_enabled;

    if parallel_profiles {
        let frozen = FrozenFiniteScanner::build(&mut scanner, input);
        let mut key_ids = FxHashMap::<(u32, u32), usize>::default();
        let mut keys = Vec::<(u32, u32)>::new();
        let mut work = Vec::<(u32, Vec<u32>, Vec<usize>)>::with_capacity(starts.len());
        for (start, raw_states) in starts {
            let mut ids = Vec::with_capacity(root_children.len());
            for &child in &root_children {
                let target = frozen.step_bytes(start, trie.edge(child, tokens));
                if target == DEAD {
                    ids.push(usize::MAX);
                    continue;
                }
                let key = (child, target);
                let id = if let Some(&id) = key_ids.get(&key) {
                    cache_hits += 1;
                    id
                } else {
                    let id = keys.len();
                    key_ids.insert(key, id);
                    keys.push(key);
                    id
                };
                ids.push(id);
            }
            work.push((start, raw_states, ids));
        }

        let results = keys
            .par_iter()
            .map(|&(child, target)| {
                let mut values = Vec::<ProfileRun>::new();
                let mut visits = 0usize;
                let mut uniform = 0usize;
                let mut uniform_token_count = 0usize;
                collect_finite_profile_frozen(
                    &frozen,
                    trie,
                    tokens,
                    child,
                    target,
                    &mut values,
                    &mut visits,
                    &mut uniform,
                    &mut uniform_token_count,
                );
                (Arc::<[ProfileRun]>::from(values), visits, uniform, uniform_token_count)
            })
            .collect::<Vec<_>>();

        let mut key_profile = vec![0u32; keys.len()];
        for (key, (values, visits, uniform, uniform_token_count)) in results.into_iter().enumerate() {
            pair_visits += visits;
            uniform_subtrees += uniform;
            uniform_tokens += uniform_token_count;
            let profile = if values.is_empty() {
                0
            } else if let Some(&profile) = profile_ids.get(&values) {
                profile
            } else {
                let profile = profiles.len() as u32;
                profile_ids.insert(Arc::clone(&values), profile);
                profiles.push(values);
                profile
            };
            key_profile[key] = profile;
            bucket_cache.insert(keys[key], profile);
        }

        for (start, raw_states, ids) in work {
            let mut fingerprint =
                Vec::with_capacity(root_children.len() + usize::from(root_token.is_some()));
            if root_token.is_some() {
                fingerprint.push(frozen.signature(start));
            }
            for id in ids {
                fingerprint.push(if id == usize::MAX { 0 } else { key_profile[id] });
            }
            let class = if let Some(&class) = class_ids.get(&fingerprint) {
                class
            } else {
                let class = class_fingerprints.len() as u32;
                class_ids.insert(fingerprint.clone(), class);
                class_fingerprints.push(fingerprint);
                class
            };
            for raw in raw_states {
                state_class[raw as usize] = class;
            }
        }
    } else {
        for (start, raw_states) in starts {
            let mut fingerprint =
                Vec::with_capacity(root_children.len() + usize::from(root_token.is_some()));
            if root_token.is_some() {
                fingerprint.push(scanner.signature(start));
            }
            for &child in &root_children {
                let target = scanner.step_bytes(start, trie.edge(child, tokens));
                if target == DEAD {
                    fingerprint.push(0);
                    continue;
                }
                let key = (child, target);
                let profile = if let Some(&profile) = bucket_cache.get(&key) {
                    cache_hits += 1;
                    profile
                } else {
                    let mut values = Vec::new();
                    let remaining_runs = profile_run_budget.saturating_sub(profile_run_count);
                    if !collect_finite_profile(
                        &mut scanner,
                        trie,
                        tokens,
                        self_loops.as_ref(),
                        child,
                        target,
                        &mut values,
                        &mut pair_visits,
                        &mut uniform_subtrees,
                        &mut uniform_tokens,
                        remaining_runs,
                        pair_visit_budget,
                    ) {
                        return build_binary_vocab_only(input);
                    }
                    let values: Arc<[ProfileRun]> = Arc::from(values);
                    let profile = if values.is_empty() {
                        0
                    } else if let Some(&profile) = profile_ids.get(&values) {
                        profile
                    } else {
                        let profile = profiles.len() as u32;
                        profile_run_count += values.len();
                        profile_ids.insert(Arc::clone(&values), profile);
                        profiles.push(values);
                        profile
                    };
                    if profile_run_count > profile_run_budget {
                        return build_binary_vocab_only(input);
                    }
                    bucket_cache.insert(key, profile);
                    profile
                };
                fingerprint.push(profile);
            }

            let class = if let Some(&class) = class_ids.get(&fingerprint) {
                class
            } else {
                let class = class_fingerprints.len() as u32;
                class_ids.insert(fingerprint.clone(), class);
                class_fingerprints.push(fingerprint);
                class
            };
            for raw in raw_states {
                state_class[raw as usize] = class;
            }
        }
    }
    let scan_wall_ms = scan_timer.elapsed().as_secs_f64() * 1000.0;
    let scan_cpu_ms = 0.0;
    let _ = (cache_hits, pair_visits, uniform_subtrees, uniform_tokens);

    let compact_timer = Instant::now();
    let (token_class, _token_reps, _compact_rows, run_sweep_events, referenced_runs) =
        finite_compact_runs(
            root_token,
            &class_fingerprints,
            &profiles,
            aliases.len(),
            false,
        );
    let preordered_vocab = preordered_vocab_map_enabled(input)
        .then(|| finite_vocab.original_order.as_deref())
        .flatten();
    let token_classes = token_class.iter().copied().max().map_or(0, |class| class + 1);
    let vocab_tokens = if input.id_map_only {
        common::grouped_vocab_id_map(
            aliases,
            &token_class,
            token_classes,
            preordered_vocab,
        )
    } else {
        common::direct_vocab_id_map(
            input.vocab.max_token_id(),
            aliases,
            &token_class,
            token_classes,
            preordered_vocab,
        )
    };
    let compact_wall_ms = compact_timer.elapsed().as_secs_f64() * 1000.0;
    let compact_cpu_ms = 0.0;
    let total_wall_ms = total_timer.elapsed().as_secs_f64() * 1000.0;
    let total_cpu_ms = 0.0;

    Some(L1VocabEquivResult {
        vocab_map: vocab_tokens,
        state_classes: class_fingerprints.len(),
        token_classes: token_classes as usize,
        prep_ms: prep_wall_ms,
        prep_cpu_ms,
        scan_ms: scan_wall_ms,
        scan_cpu_ms,
        compact_ms: compact_wall_ms,
        compact_cpu_ms,
        total_wall_ms,
        total_cpu_ms,
        run_sweep_events,
        referenced_runs,
        kernel: "finite_projected",
    })
}

fn build_binary_vocab_only(input: BuildInput<'_>) -> Option<L1VocabEquivResult> {
    build_binary_vocab_only_with_switch(input, vocab_only_residual_finite_switch_states(input))
}

fn build_binary_vocab_only_forced(input: BuildInput<'_>) -> Option<L1VocabEquivResult> {
    build_binary_vocab_only_with_switch(input, usize::MAX)
}

fn build_binary_vocab_only_with_switch(
    input: BuildInput<'_>,
    finite_switch_states: usize,
) -> Option<L1VocabEquivResult> {
    if input.vocab.is_empty() {
        return None;
    }
    let profile = std::env::var_os("GLRMASK_PROFILE_L1_IMPLEMENTATIONS").is_some();
    let total_timer = Instant::now();
    let setup_timer = Instant::now();
    let (prepared_vocab, _, _) =
        if input.subset_parent_order.is_none() && input.vocab.len() >= 10_000 {
            let (projection, hit, ms) = finite_vocab_projection(input);
            (Some(projection), hit, ms)
        } else {
            (None, false, 0.0)
        };
    let (aliases_storage, tokens_storage);
    let (aliases, tokens) = if let Some(prepared) = prepared_vocab.as_ref() {
        (prepared.aliases.as_slice(), prepared.tokens.as_slice())
    } else {
        let (a, t) = unique_vocab(input);
        aliases_storage = a;
        tokens_storage = t;
        (aliases_storage.as_slice(), tokens_storage.as_slice())
    };
    let byte_setup_started = Instant::now();
    let vocab_bytes_storage;
    let vocab_bytes: &[u8] = if let Some(prepared) = prepared_vocab.as_ref() {
        prepared.vocab_bytes.as_ref()
    } else {
        vocab_bytes_storage = collect_vocab_bytes(tokens);
        vocab_bytes_storage.as_ref()
    };
    let (bytes, input_byte_representative) = quotient_input_bytes(input, &vocab_bytes);
    let prep_wall_ms = setup_timer.elapsed().as_secs_f64() * 1000.0;
    let prep_cpu_ms = 0.0;

    let scan_timer = Instant::now();
    let direct_terminal_residuals = input
        .tokenizer
        .terminal_residual_coordinates()
        .is_some()
        && std::env::var("GLRMASK_L1_DIRECT_TERMINAL_RESIDUALS")
            .map(|value| {
                let value = value.trim();
                value.is_empty() || (value != "0" && !value.eq_ignore_ascii_case("false"))
            })
            .unwrap_or(true);
    let direct_started = profile.then(Instant::now);
    let direct_machine = if direct_terminal_residuals {
        build_direct_terminal_residual_machine(input, &bytes)
    } else {
        None
    };
    let direct_ms = direct_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let direct_selected = direct_machine.is_some();
    let (mut projected, direct_roots) = if let Some((projected, roots)) = direct_machine {
        (projected, Some(roots))
    } else {
        (Projected::new(input), None)
    };
    let mut roots = if let Some(roots) = direct_roots {
        roots
    } else {
        let root_rows = if let Some(state_map) = input.initial_state_map {
            let representative_rows = state_map
                .representative_original_ids
                .iter()
                .map(|&raw| {
                    assert_ne!(raw, u32::MAX, "L1 state quotient has an unmapped representative");
                    projected.root_sparse_row(raw)
                })
                .collect::<Vec<_>>();
            state_map
                .original_to_internal
                .iter()
                .enumerate()
                .map(|(raw, &class)| {
                    if class == u32::MAX {
                        projected.root_sparse_row(raw as u32)
                    } else {
                        representative_rows[class as usize].clone()
                    }
                })
                .collect::<Vec<_>>()
        } else {
            (0..input.tokenizer.num_states())
                .map(|raw| projected.root_sparse_row(raw))
                .collect::<Vec<_>>()
        };
        SparseRoots::from_rows(root_rows)
    };
    // Direct terminal residual construction changes how we obtain the exact
    // residual machine, not the residual-vs-finite cost crossover. Preserve
    // the existing finite handoff when the resulting direct machine is already
    // beyond that crossover; otherwise enabling direct residuals can turn an
    // intentionally-finite branch into a much slower residual scan.
    if projected.configs.len() > finite_switch_states {
        return build_finite_projected_vocab_only(input, true);
    }
    let limit = std::env::var("GLRMASK_L1_SINGLE_MAX_STATES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(2_000_000usize);
    if projected_limit_exceeded(input, projected.configs.len(), limit) {
        return build_finite_projected_vocab_only(input, true);
    }
    let expand_started = profile.then(Instant::now);
    if !direct_selected {
        let mut queue = VecDeque::from_iter(0..projected.configs.len() as u32);
        while let Some(state) = queue.pop_front() {
            let mut row = Vec::new();
            for (symbol, &byte) in bytes.iter().enumerate() {
                let before = projected.configs.len();
                let target = projected.step(state, byte, &roots);
                if target != DEAD {
                    row.push((symbol as u8, target));
                }
                if projected.configs.len() > before {
                    queue.extend(before as u32..projected.configs.len() as u32);
                    if projected.configs.len() > finite_switch_states {
                        return build_finite_projected_vocab_only(input, true);
                    }
                    if projected_limit_exceeded(input, projected.configs.len(), limit) {
                        return build_finite_projected_vocab_only(input, true);
                    }
                }
            }
            projected.transitions[state as usize] = row;
        }
    }
    let expand_ms = expand_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let groups = projected
        .configs
        .iter()
        .map(|(group, _)| *group)
        .collect::<Vec<_>>();
    let use_grouped_minimize = std::env::var("GLRMASK_L1_PROJECTED_GROUPED_MINIMIZE")
        .map(|value| {
            let value = value.trim();
            value.is_empty() || (value != "0" && !value.eq_ignore_ascii_case("false"))
        })
        .unwrap_or(true);
    let minimize_started = profile.then(Instant::now);
    let (mut minimized, _) = if direct_selected {
        (minimize_direct_residual_union(&projected.transitions, &bytes), None)
    } else if use_grouped_minimize {
        let (minimized, stats) = minimize_grouped(&projected.transitions, &groups, &bytes);
        (minimized, Some(stats))
    } else {
        (minimize(&projected.transitions, &bytes), None)
    };
    let minimize_ms = minimize_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let reverse_setup_started = profile.then(Instant::now);
    for &byte in vocab_bytes {
        minimized.byte_class[byte as usize] =
            minimized.byte_class[input_byte_representative[byte as usize] as usize];
    }
    roots.remap_states(&minimized.classes);
    let force_source_scan = false;
    let mut reverse = ReverseSubsets::new(&minimized.columns, &minimized.byte_class, minimized.state_count, force_source_scan);
    let mut final_subset_to_class = Vec::<u32>::new();
    let mut class_subsets = Vec::<u32>::new();
    let mut token_class = vec![0u32; aliases.len()];
    let dense_reverse_enabled = std::env::var("GLRMASK_L1_RESIDUAL_DENSE_REVERSE")
        .map(|value| {
            let value = value.trim();
            value.is_empty() || (value != "0" && !value.eq_ignore_ascii_case("false"))
        })
        .unwrap_or(input.partition_label == "p2" && input.subset_parent_order.is_none());
    let use_reverse_trie = std::env::var("GLRMASK_L1_RESIDUAL_REVERSE_TRIE")
        .map(|value| {
            let value = value.trim();
            value.is_empty() || (value != "0" && !value.eq_ignore_ascii_case("false"))
        })
        .unwrap_or(true);
    let reverse_trie = use_reverse_trie
        .then_some(())
        .and_then(|()| prepared_vocab.as_ref())
        .and_then(|prepared| prepared.reverse_trie.as_ref());
    let dense_reverse_max_states = std::env::var("GLRMASK_L1_RESIDUAL_DENSE_REVERSE_MAX_STATES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(192usize);
    let dense_reverse = dense_reverse_enabled
        .then(|| reverse.try_complete_dense(dense_reverse_max_states))
        .flatten();
    let reverse_setup_ms =
        reverse_setup_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let token_started = profile.then(Instant::now);
    match (dense_reverse.as_ref(), reverse_trie) {
        (Some(dense), Some(trie)) => {
            debug_assert_eq!(trie.parents.len(), trie.bytes.len());
            let symbols = reverse.symbol_count;
            let mut subset_at_node = vec![0u32; trie.parents.len()];
            for node in 1..trie.parents.len() {
                let parent = trie.parents[node] as usize;
                let symbol = reverse.byte_class[trie.bytes[node] as usize] as usize;
                subset_at_node[node] = dense[subset_at_node[parent] as usize * symbols + symbol];
            }
            for (token_index, &node) in trie.token_node.iter().enumerate() {
                let subset = subset_at_node[node as usize];
                token_class[token_index] = intern_dense_subset_class(
                    subset,
                    &mut final_subset_to_class,
                    &mut class_subsets,
                );
            }
        }
        (Some(dense), None) => {
            let symbols = reverse.symbol_count;
            for (token_index, token) in tokens.iter().enumerate() {
                let subset = token.iter().rev().fold(0u32, |subset, &byte| {
                    let symbol = reverse.byte_class[byte as usize] as usize;
                    dense[subset as usize * symbols + symbol]
                });
                token_class[token_index] = intern_dense_subset_class(
                    subset,
                    &mut final_subset_to_class,
                    &mut class_subsets,
                );
            }
        }
        (None, Some(trie)) => {
            debug_assert_eq!(trie.parents.len(), trie.bytes.len());
            let mut subset_at_node = vec![0u32; trie.parents.len()];
            for node in 1..trie.parents.len() {
                let parent = trie.parents[node] as usize;
                subset_at_node[node] = reverse.prepend(subset_at_node[parent], trie.bytes[node]);
            }
            for (token_index, &node) in trie.token_node.iter().enumerate() {
                let subset = subset_at_node[node as usize];
                token_class[token_index] = intern_dense_subset_class(
                    subset,
                    &mut final_subset_to_class,
                    &mut class_subsets,
                );
            }
        }
            (None, None) => {
                for (token_index, token) in tokens.iter().enumerate() {
                    let subset = reverse.token(token);
                    token_class[token_index] = intern_dense_subset_class(
                        subset,
                        &mut final_subset_to_class,
                        &mut class_subsets,
                    );
                }
            }
        }
    let token_ms = token_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let scan_wall_ms = scan_timer.elapsed().as_secs_f64() * 1000.0;
    if profile {
        eprintln!(
            "[glrmask/profile][l1_residual_projected_scan] partition={} direct_selected={} direct_ms={direct_ms:.3} expand_ms={expand_ms:.3} minimize_ms={minimize_ms:.3} reverse_setup_ms={reverse_setup_ms:.3} token_ms={token_ms:.3} scan_ms={scan_wall_ms:.3}",
            input.partition_label,
            direct_selected,
        );
    }
    let scan_cpu_ms = 0.0;

    let compact_timer = Instant::now();
    let preordered_vocab = preordered_vocab_map_enabled(input)
        .then(|| prepared_vocab.as_ref().and_then(|p| p.original_order.as_deref()))
        .flatten();
    let token_classes = token_class.iter().copied().max().map_or(0, |class| class + 1);
    let vocab_tokens = if input.id_map_only {
        common::grouped_vocab_id_map(
            aliases,
            &token_class,
            token_classes,
            preordered_vocab,
        )
    } else {
        common::direct_vocab_id_map(
            input.vocab.max_token_id(),
            aliases,
            &token_class,
            token_classes,
            preordered_vocab,
        )
    };
    let compact_wall_ms = compact_timer.elapsed().as_secs_f64() * 1000.0;
    let compact_cpu_ms = 0.0;
    let total_wall_ms = total_timer.elapsed().as_secs_f64() * 1000.0;
    let total_cpu_ms = 0.0;

    Some(L1VocabEquivResult {
        vocab_map: vocab_tokens,
        state_classes: minimized.state_count,
        token_classes: token_classes as usize,
        prep_ms: prep_wall_ms,
        prep_cpu_ms,
        scan_ms: scan_wall_ms,
        scan_cpu_ms,
        compact_ms: compact_wall_ms,
        compact_cpu_ms,
        total_wall_ms,
        total_cpu_ms,
        run_sweep_events: 0,
        referenced_runs: 0,
        kernel: "residual_projected",
    })
}


#[cfg(test)]
mod finite_run_sweep_tests {
    use super::*;

    #[test]
    fn reverse_subset_interner_resolves_fingerprint_collisions_exactly() {
        let columns = vec![vec![0u32, 1].into_boxed_slice()];
        let byte_class = [0u8; 256];
        let mut reverse = ReverseSubsets::new(&columns, &byte_class, 2, false);
        let forced_hash = (0u64..)
            .find(|hash| !reverse.ids.contains_key(hash))
            .expect("there is always an unused fingerprint");

        let left = reverse.intern_hashed(vec![0b01], forced_hash);
        let right = reverse.intern_hashed(vec![0b10], forced_hash);
        assert_ne!(left, right);
        assert_eq!(reverse.intern_hashed(vec![0b01], forced_hash), left);
        assert_eq!(reverse.intern_hashed(vec![0b10], forced_hash), right);
        assert_eq!(reverse.sets[left as usize].as_ref(), &[0b01]);
        assert_eq!(reverse.sets[right as usize].as_ref(), &[0b10]);
    }

    #[test]
    fn sparse_symbol_target_minimizer_handles_all_256_symbol_ranks() {
        let alphabet = (0u16..=255).map(|byte| byte as u8).collect::<Vec<_>>();
        let mut transitions = vec![Vec::<(u8, u32)>::new(); 4];
        // State 0 sends every byte to state 2, so `(255, 2)` occupies rank 255
        // in the target-major sparse predecessor index. State 1 differs only on
        // that final byte and must therefore split from state 0.
        transitions[0] = alphabet.iter().copied().map(|byte| (byte, 2)).collect();
        transitions[1] = alphabet
            .iter()
            .copied()
            .map(|byte| (byte, if byte == 255 { 3 } else { 2 }))
            .collect();
        let initial = [0, 0, 1, 2];

        let generic = minimize_seeded(&transitions, &alphabet, Some(&initial));
        let sparse = minimize_seeded_symbol_target_csr(&transitions, &alphabet, &initial);
        assert_eq!(generic.state_count, sparse.state_count);
        for left in 0..transitions.len() {
            for right in 0..transitions.len() {
                assert_eq!(
                    generic.classes[left] == generic.classes[right],
                    sparse.classes[left] == sparse.classes[right],
                    "equivalence mismatch for states {left} and {right}",
                );
            }
        }
    }

    #[test]
    fn sparse_symbol_target_minimizer_moves_smaller_complement_exactly() {
        let alphabet = vec![0u8];
        let mut transitions = vec![Vec::<(u8, u32)>::new(); 5];
        // States 0..=3 begin in one block. On symbol 0, three of them point
        // into splitter state 4 while state 3 does not. The affected side is
        // therefore 3/4 of the block, exercising the complement-moving path.
        transitions[0].push((0, 4));
        transitions[1].push((0, 4));
        transitions[2].push((0, 4));
        let initial = [0, 0, 0, 0, 1];

        let generic = minimize_seeded(&transitions, &alphabet, Some(&initial));
        let sparse = minimize_seeded_symbol_target_csr(&transitions, &alphabet, &initial);
        assert_eq!(generic.state_count, sparse.state_count);
        for left in 0..transitions.len() {
            for right in 0..transitions.len() {
                assert_eq!(
                    generic.classes[left] == generic.classes[right],
                    sparse.classes[left] == sparse.classes[right],
                    "equivalence mismatch for states {left} and {right}",
                );
            }
        }
    }


    #[test]
    fn run_sweep_matches_dense_token_vector_equivalence() {
        const ROWS: usize = 17;
        const TOKENS: usize = 137;
        let mut dense = vec![vec![0u32; TOKENS]; ROWS];
        for (row, values) in dense.iter_mut().enumerate() {
            for (token, value) in values.iter_mut().enumerate() {
                // Piecewise structure with many adjacent boundaries and repeated
                // vectors across non-adjacent token intervals.
                let block = token / (2 + row % 7);
                *value = if (block + row) % 5 == 0 {
                    0
                } else {
                    1 + ((block * 3 + row * 11) % 9) as u32
                };
            }
        }

        let mut profiles = vec![Arc::<[ProfileRun]>::from([])];
        let mut fingerprints = Vec::with_capacity(ROWS);
        for values in &dense {
            let mut runs = Vec::new();
            let mut start = 0usize;
            while start < TOKENS {
                let signature = values[start];
                let mut end = start + 1;
                while end < TOKENS && values[end] == signature {
                    end += 1;
                }
                push_profile_run(&mut runs, start as u32, end as u32, signature);
                start = end;
            }
            let profile = profiles.len() as u32;
            profiles.push(Arc::from(runs));
            fingerprints.push(vec![profile]);
        }

        let (classes, reps, compact_rows, events, referenced_runs) =
            finite_compact_runs(None, &fingerprints, &profiles, TOKENS, true);
        // Small matrices intentionally take the exact dense compactor and
        // therefore do not materialize sweep events. Both compaction paths must
        // preserve the same token-vector equivalence below.
        assert!(events > 0 || ROWS * TOKENS <= 2_000_000);
        assert!(referenced_runs > 0);
        assert_eq!(compact_rows.len(), ROWS);
        assert_eq!(compact_rows[0].len(), reps.len());

        for left in 0..TOKENS {
            for right in 0..TOKENS {
                let dense_equal = dense.iter().all(|row| row[left] == row[right]);
                assert_eq!(
                    classes[left] == classes[right],
                    dense_equal,
                    "token equivalence mismatch for {left} and {right}",
                );
            }
        }
        for (class, &representative) in reps.iter().enumerate() {
            for row in 0..ROWS {
                assert_eq!(compact_rows[row][class], dense[row][representative]);
            }
        }
    }
}
