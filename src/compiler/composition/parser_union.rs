//! Combine weighted parser automata while preserving DEFAULT semantics.

use super::{
    Arc, BTreeMap, BTreeSet, BoundaryTokenDiscovery, BoundaryTokenNodeKey, Constraint,
    DEFAULT_LABEL, DWA, DWAState, DirectComponentCoordinateMaps, FxHashMap, FxHashSet, Instant,
    InternalIdMap, ManyToOneIdMap, MappedArtifact, NWA, NWAState, ParserDefaultDomain,
    ParserDwaComponent, PossibleMatches, ScopedWeightOpCache, SmallVec, VecDeque, Weight,
    WeightRefs, build_direct_component_coordinate_maps, component_parser_nwa,
    compose_profile_enabled, determinize, encode_positive_label, find_difference,
    is_negative_label, macro_parallelism_disabled, remap_weights_with_maps,
    report_macro_item_timings, reverse_hashcons_owned, strip_unscoped_ignore_identity,
};
use rayon::prelude::*;

pub(super) fn boundary_discovery_good_signature(
    discovery: &BoundaryTokenDiscovery,
) -> BTreeSet<(u32, Vec<u32>, Vec<(BoundaryTokenNodeKey, bool, Vec<(u32, BoundaryTokenNodeKey)>)>)> {
    discovery
        .witnesses
        .iter()
        .map(|witness| {
            let mut rows = witness
                .nodes
                .iter()
                .enumerate()
                .filter_map(|(node_id, node)| {
                    witness.good[node_id].then(|| {
                        let mut outgoing = node
                            .outgoing
                            .iter()
                            .filter(|edge| witness.good[edge.target])
                            .map(|edge| (edge.terminal, witness.nodes[edge.target].key))
                            .collect::<Vec<_>>();
                        outgoing.sort_unstable();
                        outgoing.dedup();
                        (node.key, witness.accepting[node_id], outgoing)
                    })
                })
                .collect::<Vec<_>>();
            rows.sort_unstable();
            (
                witness.token_id,
                witness.start_states.clone(),
                rows,
            )
        })
        .collect()
}

#[derive(Clone)]
pub(super) struct RawTransitionRun {
    pub(super) start: i32,
    pub(super) end: i32,
    pub(super) targets: Vec<(u32, Weight)>,
}

#[derive(Clone)]
pub(super) struct RawCompressedState {
    pub(super) runs: Vec<RawTransitionRun>,
    pub(super) default_targets: Option<Vec<(u32, Weight)>>,
    pub(super) final_weight: Option<Weight>,
    pub(super) deterministic: bool,
}

#[derive(Clone)]
pub(super) struct RawCompressedAutomaton {
    pub(super) states: Vec<RawCompressedState>,
    pub(super) start_states: Vec<u32>,
}

pub(super) fn subtract_weight_support_from_raw_automata(
    automata: &mut [RawCompressedAutomaton],
    claimed: &Weight,
) {
    if claimed.is_empty() {
        return;
    }
    for automaton in automata {
        for state in &mut automaton.states {
            for run in &mut state.runs {
                for (_, weight) in &mut run.targets {
                    *weight = weight.difference(claimed);
                }
                run.targets.retain(|(_, weight)| !weight.is_empty());
            }
            state.runs.retain(|run| !run.targets.is_empty());
            if let Some(targets) = &mut state.default_targets {
                for (_, weight) in targets.iter_mut() {
                    *weight = weight.difference(claimed);
                }
                targets.retain(|(_, weight)| !weight.is_empty());
                if targets.is_empty() {
                    state.default_targets = None;
                }
            }
            if let Some(final_weight) = state.final_weight.take() {
                let remaining = final_weight.difference(claimed);
                state.final_weight = (!remaining.is_empty()).then_some(remaining);
            }
        }
    }
}

pub(super) fn same_raw_targets(left: &[(u32, Weight)], right: &[(u32, Weight)]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|((left_target, left_weight), (right_target, right_weight))| {
                left_target == right_target && left_weight.ptr_key() == right_weight.ptr_key()
            })
}

impl RawCompressedAutomaton {
    pub(super) fn from_nwa(automaton: NWA) -> Self {
        let (states, start_states) = automaton.into_parts();
        let states = states
            .into_par_iter()
            .map(|state| {
                debug_assert!(state.epsilons.is_empty());
                let mut runs = Vec::<RawTransitionRun>::new();
                let mut default_targets = None;
                let mut deterministic = true;
                for (label, targets) in state.transitions {
                    deterministic &= targets.len() <= 1;
                    if targets.is_empty() {
                        continue;
                    }
                    if label == DEFAULT_LABEL {
                        default_targets = Some(targets);
                        continue;
                    }
                    if let Some(last) = runs.last_mut()
                        && last.end.checked_add(1) == Some(label)
                        && same_raw_targets(&last.targets, &targets)
                    {
                        last.end = label;
                    } else {
                        runs.push(RawTransitionRun {
                            start: label,
                            end: label,
                            targets,
                        });
                    }
                }
                RawCompressedState {
                    runs,
                    default_targets,
                    final_weight: state.final_weight,
                    deterministic,
                }
            })
            .collect();
        Self {
            states,
            start_states,
        }
    }

    pub(super) fn from_dwa_preserving_defaults(automaton: &DWA) -> Self {
        let states = automaton
            .states()
            .par_iter()
            .map(|state| {
                let mut runs = Vec::<RawTransitionRun>::new();
                let mut default_targets = None;
                for (&label, (target, weight)) in &state.transitions {
                    let targets = vec![(*target, weight.clone())];
                    if label == DEFAULT_LABEL {
                        default_targets = Some(targets);
                        continue;
                    }
                    if let Some(last) = runs.last_mut()
                        && last.end.checked_add(1) == Some(label)
                        && same_raw_targets(&last.targets, &targets)
                    {
                        last.end = label;
                    } else {
                        runs.push(RawTransitionRun {
                            start: label,
                            end: label,
                            targets,
                        });
                    }
                }
                RawCompressedState {
                    runs,
                    default_targets,
                    final_weight: state.final_weight.clone(),
                    deterministic: true,
                }
            })
            .collect();
        Self {
            states,
            start_states: vec![automaton.start_state()],
        }
    }

    pub(super) fn to_nwa(&self) -> NWA {
        let states = self
            .states
            .iter()
            .map(|state| {
                let mut transitions = BTreeMap::<i32, Vec<(u32, Weight)>>::new();
                for run in &state.runs {
                    for label in run.start..=run.end {
                        transitions.insert(label, run.targets.clone());
                    }
                }
                if let Some(default_targets) = &state.default_targets {
                    transitions.insert(DEFAULT_LABEL, default_targets.clone());
                }
                NWAState {
                    final_weight: state.final_weight.clone(),
                    transitions,
                    epsilons: Vec::new(),
                }
            })
            .collect();
        NWA::from_parts(states, self.start_states.clone())
    }
}

impl WeightRefs for RawCompressedAutomaton {
     fn weight_refs(&self) -> Vec<&Weight> {
        let mut weights = Vec::new();
        for state in &self.states {
            if let Some(weight) = &state.final_weight {
                weights.push(weight);
            }
            for run in &state.runs {
                weights.extend(run.targets.iter().map(|(_, weight)| weight));
            }
            if let Some(targets) = &state.default_targets {
                weights.extend(targets.iter().map(|(_, weight)| weight));
            }
        }
        weights
    }

     fn weight_refs_mut(&mut self) -> Vec<&mut Weight> {
        let mut weights = Vec::new();
        for state in &mut self.states {
            if let Some(weight) = &mut state.final_weight {
                weights.push(weight);
            }
            for run in &mut state.runs {
                weights.extend(run.targets.iter_mut().map(|(_, weight)| weight));
            }
            if let Some(targets) = &mut state.default_targets {
                weights.extend(targets.iter_mut().map(|(_, weight)| weight));
            }
        }
        weights
    }
}

/// Exact union/determinization specialized for epsilon-free acyclic component
/// parser automata. Reachable singleton raw states are copied directly;
/// synthetic subset states
/// are created only for genuinely overlapping branches (normally at the merged
/// root). This remains exact when a future table relation introduces additional
/// label overlap, rather than falling off a disjoint-alphabet fast path.
pub(super) fn supports_overlap_local_union(automata: &[NWA]) -> bool {
    automata.iter().all(|automaton| {
        automaton.is_acyclic()
            && automaton
                .states()
                .iter()
                .all(|state| state.epsilons.is_empty())
    })
}

pub(super) fn determinize_epsilon_free_component_union(
    automata: Vec<NWA>,
    default_positive_label_count: Option<u32>,
) -> Option<(DWA, usize)> {
    if !supports_overlap_local_union(&automata) {
        return None;
    }
    if compose_profile_enabled() {
        let shapes = automata
            .iter()
            .map(|automaton| (automaton.num_states(), automaton.num_transitions(), automaton.start_states().len()))
            .collect::<Vec<_>>();
        eprintln!("[glrmask/profile][constraint_overlap_local_inputs] shapes={shapes:?}");
    }

    let mut raw_states = Vec::new();
    let mut raw_owners = Vec::<usize>::new();
    let mut raw_locals = Vec::<u32>::new();
    let mut starts = Vec::new();
    for (automaton_index, automaton) in automata.into_iter().enumerate() {
        let offset = raw_states.len() as u32;
        let (states, start_states) = automaton.into_parts();
        raw_owners.extend(std::iter::repeat_n(automaton_index, states.len()));
        raw_locals.extend(0..states.len() as u32);
        starts.extend(start_states.into_iter().map(|state| offset + state));
        for mut appended in states {
            for targets in appended.transitions.values_mut() {
                for (target, _) in targets {
                    *target += offset;
                }
            }
            raw_states.push(appended);
        }
    }
    if starts.is_empty() {
        return Some((DWA::new(0, 0), 0));
    }

    type ResidualSubset = SmallVec<[(u32, Weight); 4]>;
    // Semantic key, not a storage-address key.  Weight has structural Eq and a
    // cached structural Hash, so equal residual languages intern to the same
    // determinized state regardless of allocator/interner lifetime ordering.
    type ResidualSubsetKey = SmallVec<[(u32, Weight); 4]>;

    #[derive(Clone)]
    enum PendingUnionWeight {
        Immediate(Weight),
        Deferred(usize),
    }

    #[derive(Clone, Copy)]
    struct DeferredWeightPatchRun {
        state: u32,
        start: i32,
        end: i32,
        job: usize,
    }

    fn append_pending_transition_run(
        output_state: u32,
        start: i32,
        end: i32,
        target: u32,
        pending_weight: PendingUnionWeight,
        output_transitions: &mut Vec<(i32, (u32, Weight))>,
        deferred_patches: &mut Vec<DeferredWeightPatchRun>,
        insert_default_first: bool,
    ) {
        debug_assert!(start <= end);
        let mut append_entries = |weight: &Weight| {
            if insert_default_first {
                debug_assert_eq!(start, end);
                output_transitions.insert(0, (start, (target, weight.clone())));
            } else {
                for label in start..=end {
                    output_transitions.push((label, (target, weight.clone())));
                }
            }
        };
        match pending_weight {
            PendingUnionWeight::Immediate(weight) => append_entries(&weight),
            PendingUnionWeight::Deferred(job) => {
                append_entries(&Weight::empty());
                if let Some(previous) = deferred_patches.last_mut()
                    && previous.state == output_state
                    && previous.job == job
                    && previous.end.checked_add(1) == Some(start)
                {
                    previous.end = end;
                } else {
                    deferred_patches.push(DeferredWeightPatchRun {
                        state: output_state,
                        start,
                        end,
                        job,
                    });
                }
            }
        }
    }


    fn normalize_subset(entries: Vec<(u32, Weight)>) -> ResidualSubset {
        let mut by_state = FxHashMap::<u32, Weight>::default();
        for (state, weight) in entries {
            if weight.is_empty() {
                continue;
            }
            by_state
                .entry(state)
                .and_modify(|existing| *existing = existing.union(&weight))
                .or_insert(weight);
        }
        let mut normalized = by_state.into_iter().collect::<ResidualSubset>();
        normalized.sort_unstable_by_key(|(state, _)| *state);
        normalized
    }

    fn explicit_transition_runs<'a>(
        source: &'a NWAState,
    ) -> Vec<(i32, i32, &'a Vec<(u32, Weight)>)> {
        let mut runs = Vec::<(i32, i32, &'a Vec<(u32, Weight)>)>::new();
        for (&label, targets) in source
            .transitions
            .iter()
            .filter(|(label, _)| **label != DEFAULT_LABEL)
        {
            if let Some((_, end, previous_targets)) = runs.last_mut()
                && end.checked_add(1) == Some(label)
                && same_raw_targets(previous_targets, targets)
            {
                *end = label;
            } else {
                runs.push((label, label, targets));
            }
        }
        runs
    }

    fn intern_singleton(
        raw_state: u32,
        singleton_states: &mut [u32],
        singleton_count: &mut usize,
        states: &mut Vec<DWAState>,
        queue: &mut VecDeque<(u32, ResidualSubset)>,
    ) -> u32 {
        let slot = &mut singleton_states[raw_state as usize];
        if *slot != u32::MAX {
            return *slot;
        }
        let created = states.len() as u32;
        states.push(DWAState::default());
        *slot = created;
        *singleton_count += 1;
        let mut subset = ResidualSubset::new();
        subset.push((raw_state, Weight::all()));
        queue.push_back((created, subset));
        created
    }

    fn finish_overlap_transition(
        mut contributions: SmallVec<[(u32, Weight); 4]>,
        weight_ops: &mut ScopedWeightOpCache,
        defer_support_unions: bool,
        deferred_union_ids: &mut FxHashMap<(usize, usize), usize>,
        deferred_union_jobs: &mut Vec<(Weight, Weight)>,
        singleton_states: &mut [u32],
        singleton_count: &mut usize,
        states: &mut Vec<DWAState>,
        queue: &mut VecDeque<(u32, ResidualSubset)>,
        subset_states: &mut FxHashMap<ResidualSubsetKey, u32>,
    ) -> Option<(u32, PendingUnionWeight)> {
        let finish_subset = |
            normalized: ResidualSubset,
            singleton_states: &mut [u32],
            singleton_count: &mut usize,
            states: &mut Vec<DWAState>,
            queue: &mut VecDeque<(u32, ResidualSubset)>,
            subset_states: &mut FxHashMap<ResidualSubsetKey, u32>,
        | {
            if normalized.len() == 1 {
                return intern_singleton(
                    normalized[0].0,
                    singleton_states,
                    singleton_count,
                    states,
                    queue,
                );
            }
            let key = normalized
                .iter()
                .map(|(state, weight)| (*state, weight.clone()))
                .collect::<ResidualSubsetKey>();
            if let Some(&existing) = subset_states.get(&key) {
                existing
            } else {
                let created = states.len() as u32;
                states.push(DWAState::default());
                subset_states.insert(key, created);
                queue.push_back((created, normalized));
                created
            }
        };

        match contributions.len() {
            0 => None,
            1 => {
                let (target, edge_weight) = contributions.pop().unwrap();
                let target = intern_singleton(
                    target,
                    singleton_states,
                    singleton_count,
                    states,
                    queue,
                );
                Some((target, PendingUnionWeight::Immediate(edge_weight)))
            }
            2 => {
                let (right_target, right_weight) = contributions.pop().unwrap();
                let (left_target, left_weight) = contributions.pop().unwrap();

                // The target residual subset is determined entirely by the
                // non-empty contributions.  Support outside this edge is
                // unobservable after the edge is taken, so constructing the
                // expensive union is not required to discover graph topology.
                // This matches the eager normal form used here today, where
                // Weight::complement() deliberately contributes no finite
                // outside-support normalization.
                let pending_weight = if left_weight.is_full() || right_weight.is_full() {
                    PendingUnionWeight::Immediate(Weight::all())
                } else if left_weight.ptr_key() == right_weight.ptr_key() {
                    PendingUnionWeight::Immediate(left_weight.clone())
                } else if defer_support_unions {
                    let left_key = left_weight.ptr_key();
                    let right_key = right_weight.ptr_key();
                    let key = if left_key <= right_key {
                        (left_key, right_key)
                    } else {
                        (right_key, left_key)
                    };
                    let job = if let Some(&job) = deferred_union_ids.get(&key) {
                        job
                    } else {
                        let job = deferred_union_jobs.len();
                        deferred_union_ids.insert(key, job);
                        deferred_union_jobs.push((left_weight.clone(), right_weight.clone()));
                        job
                    };
                    PendingUnionWeight::Deferred(job)
                } else {
                    PendingUnionWeight::Immediate(weight_ops.union(&left_weight, &right_weight))
                };

                if left_target == right_target {
                    let target = intern_singleton(
                        left_target,
                        singleton_states,
                        singleton_count,
                        states,
                        queue,
                    );
                    return Some((target, pending_weight));
                }

                let mut normalized = ResidualSubset::new();
                if left_target < right_target {
                    normalized.push((left_target, left_weight));
                    normalized.push((right_target, right_weight));
                } else {
                    normalized.push((right_target, right_weight));
                    normalized.push((left_target, left_weight));
                }
                let target = finish_subset(
                    normalized,
                    singleton_states,
                    singleton_count,
                    states,
                    queue,
                    subset_states,
                );
                Some((target, pending_weight))
            }
            _ => {
                contributions.sort_unstable_by_key(|(target, _)| *target);
                let mut next_subset = ResidualSubset::new();
                for (target, contribution) in contributions {
                    if let Some((last_target, existing)) = next_subset.last_mut()
                        && *last_target == target
                    {
                        *existing = weight_ops.union(existing, &contribution);
                    } else {
                        next_subset.push((target, contribution));
                    }
                }
                let edge_weight =
                    weight_ops.union_all(next_subset.iter().map(|(_, weight)| weight));
                let edge_complement = edge_weight.complement();
                let normalized = if edge_complement.is_empty() {
                    next_subset
                } else {
                    next_subset
                        .into_iter()
                        .map(|(state, weight)| {
                            let residual = weight_ops.union(&weight, &edge_complement);
                            (state, residual)
                        })
                        .collect::<ResidualSubset>()
                };
                let target = finish_subset(
                    normalized,
                    singleton_states,
                    singleton_count,
                    states,
                    queue,
                    subset_states,
                );
                Some((target, PendingUnionWeight::Immediate(edge_weight)))
            }
        }
    }

    let preallocate_raw_singletons =
        std::env::var_os("GLRMASK_EXPERIMENT_PREALLOCATE_RAW_SINGLETONS").is_some();
    let mut preallocated_nondeterministic = Vec::<u32>::new();
    let (mut states, mut singleton_states, mut singleton_count) = if preallocate_raw_singletons {
        let states = raw_states
            .iter()
            .enumerate()
            .map(|(raw_state, source)| {
                if source.transitions.values().any(|targets| targets.len() > 1) {
                    preallocated_nondeterministic.push(raw_state as u32);
                    return DWAState::default();
                }
                let source_has_default = source.transitions.contains_key(&DEFAULT_LABEL);
                let transitions = source
                    .transitions
                    .iter()
                    .filter_map(|(&label, targets)| {
                        let (target, edge_weight) = targets.first()?;
                        if edge_weight.is_empty()
                            && !(source_has_default && label >= 0 && label != DEFAULT_LABEL)
                        {
                            return None;
                        }
                        Some((label, (*target, edge_weight.clone())))
                    })
                    .collect();
                let final_weight = source
                    .final_weight
                    .as_ref()
                    .filter(|weight| !weight.is_empty())
                    .cloned();
                DWAState {
                    transitions,
                    final_weight,
                }
            })
            .collect::<Vec<_>>();
        let singleton_states = (0..raw_states.len() as u32).collect::<Vec<_>>();
        let singleton_count = raw_states.len();
        (states, singleton_states, singleton_count)
    } else {
        (
            Vec::<DWAState>::new(),
            vec![u32::MAX; raw_states.len()],
            0usize,
        )
    };
    let mut dead_shadow_state = None::<u32>;
    let mut queue = VecDeque::<(u32, ResidualSubset)>::new();
    if preallocate_raw_singletons {
        for raw_state in preallocated_nondeterministic.iter().copied() {
            let mut subset = ResidualSubset::new();
            subset.push((raw_state, Weight::all()));
            queue.push_back((raw_state, subset));
        }
        if compose_profile_enabled() {
            eprintln!(
                "[glrmask/profile][constraint_raw_singleton_preallocation] raw_states={} nondeterministic_rows={}",
                raw_states.len(),
                preallocated_nondeterministic.len(),
            );
        }
    }
    let initial_subset = normalize_subset(
        starts
            .into_iter()
            .map(|state| (state, Weight::all()))
            .collect(),
    );
    let start_state = if initial_subset.len() == 1 {
        intern_singleton(
            initial_subset[0].0,
            &mut singleton_states,
            &mut singleton_count,
            &mut states,
            &mut queue,
        )
    } else {
        let state = states.len() as u32;
        states.push(DWAState::default());
        queue.push_back((state, initial_subset.clone()));
        state
    };
    let mut subset_states = FxHashMap::<ResidualSubsetKey, u32>::default();
    if initial_subset.len() > 1 {
        subset_states.insert(
            initial_subset
                .iter()
                .map(|(state, weight)| (*state, weight.clone()))
                .collect::<ResidualSubsetKey>(),
            start_state,
        );
    }

    let mut weight_ops = ScopedWeightOpCache::default();
    let mut profiled_singletons = 0usize;
    let mut profiled_pair_subsets = 0usize;
    let mut profiled_wide_subsets = 0usize;
    let mut profiled_max_subset = 0usize;
    let mut profiled_explicit_labels = 0usize;
    let mut profiled_pair_left_only_labels = 0usize;
    let mut profiled_pair_right_only_labels = 0usize;
    let mut profiled_pair_both_labels = 0usize;
    let mut profiled_pair_left_prefix_full = 0usize;
    let mut profiled_pair_right_prefix_full = 0usize;
    let mut profiled_pair_cached_default_uses = 0usize;
    let mut profiled_pair_owner_counts = BTreeMap::<(usize, usize), usize>::new();
    let mut profiled_boundary_pair_states = FxHashMap::<u32, usize>::default();
    let mut profiled_pair_left_residuals = FxHashSet::<(u32, usize)>::default();
    let mut profiled_pair_right_residuals = FxHashSet::<(u32, usize)>::default();
    let mut profiled_pair_left_raw_states = FxHashSet::<u32>::default();
    let mut profiled_pair_right_raw_states = FxHashSet::<u32>::default();
    type ResidualEdgeIntersections = SmallVec<[(usize, Weight); 8]>;
    let mut cached_residual_rows =
        FxHashMap::<(u32, usize), Arc<ResidualEdgeIntersections>>::default();
    let use_residual_row_cache =
        std::env::var_os("GLRMASK_DISABLE_DIRECT_UNION_RESIDUAL_ROW_CACHE").is_none();
    let use_interval_pair_runs =
        std::env::var_os("GLRMASK_DISABLE_DIRECT_UNION_INTERVAL_RUNS").is_none();
    let defer_support_unions =
        std::env::var_os("GLRMASK_DISABLE_DIRECT_UNION_DEFER_SUPPORT_UNIONS").is_none()
            && rayon::current_num_threads() > 1;
    let mut deferred_union_ids = FxHashMap::<(usize, usize), usize>::default();
    let mut deferred_union_jobs = Vec::<(Weight, Weight)>::new();
    let mut deferred_weight_patches = Vec::<DeferredWeightPatchRun>::new();
    let mut deferred_final_union_ids = FxHashMap::<SmallVec<[usize; 4]>, usize>::default();
    let mut deferred_final_union_jobs = Vec::<SmallVec<[Weight; 4]>>::new();
    let mut deferred_final_weight_patches = Vec::<(u32, usize)>::new();
    let profiled_local_intersection_lookups = std::cell::Cell::new(0usize);
    let profiled_global_intersection_lookups = std::cell::Cell::new(0usize);
    let profiled_row_cache_build_intersections = std::cell::Cell::new(0usize);
    let residual_prefetch_min_raw_states = std::env::var(
        "GLRMASK_DIRECT_UNION_RESIDUAL_PREFETCH_MIN_RAW_STATES",
    )
    .ok()
    .and_then(|value| value.parse::<usize>().ok())
    .unwrap_or(65_536);
    let use_parallel_residual_row_prefetch = use_residual_row_cache
        && raw_states.len() >= residual_prefetch_min_raw_states
        && std::env::var_os("GLRMASK_DISABLE_PARALLEL_RESIDUAL_ROW_PREFETCH").is_none()
        && rayon::current_num_threads() > 1;
    let residual_row_prefetch_batch = std::env::var("GLRMASK_RESIDUAL_ROW_PREFETCH_BATCH")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(8192)
        .max(1);
    let mut residual_row_prefetch_remaining = 0usize;
    let mut profiled_residual_row_prefetch_batches = 0usize;
    let mut profiled_residual_row_prefetch_rows = 0usize;
    let mut profiled_residual_row_prefetch_intersection_jobs = 0usize;
    let mut profiled_residual_row_prefetch_ms = 0.0f64;
    while let Some((output_state, subset)) = queue.pop_front() {
        if use_parallel_residual_row_prefetch && residual_row_prefetch_remaining == 0 {
            let started_at = Instant::now();
            let covered_states = (queue.len() + 1).min(residual_row_prefetch_batch);
            let mut missing = FxHashMap::<(u32, usize), (u32, Weight)>::default();
            for candidate in std::iter::once(&subset)
                .chain(queue.iter().map(|(_, subset)| subset))
                .take(covered_states)
            {
                if candidate.len() != 2 {
                    continue;
                }
                for (raw_state, prefix) in candidate {
                    let key = (*raw_state, prefix.ptr_key());
                    if !cached_residual_rows.contains_key(&key) {
                        missing
                            .entry(key)
                            .or_insert_with(|| (*raw_state, prefix.clone()));
                    }
                }
            }
            let jobs = missing.into_iter().collect::<Vec<_>>();
            let dedup_intersection_jobs =
                std::env::var_os("GLRMASK_DISABLE_DEDUP_RESIDUAL_PREFETCH_JOBS").is_none();
            let prepared = if dedup_intersection_jobs {
                // Read source rows in parallel, then intern exact
                // (residual-prefix, source-edge-weight) algebra jobs across the
                // whole frontier batch.  The row cache itself is keyed by raw
                // state because different rows can expose different labels,
                // but their Weight intersections are often identical.
                let row_specs = jobs
                    .par_iter()
                    .map(|(key, (raw_state, prefix))| {
                        let source = &raw_states[*raw_state as usize];
                        let mut edges = SmallVec::<[(usize, Weight); 8]>::new();
                        for targets in source.transitions.values() {
                            for (_, edge_weight) in targets {
                                let edge_key = edge_weight.ptr_key();
                                if edges.iter().any(|(key, _)| *key == edge_key) {
                                    continue;
                                }
                                edges.push((edge_key, edge_weight.clone()));
                            }
                        }
                        edges.sort_unstable_by_key(|(key, _)| *key);
                        (*key, prefix.clone(), edges)
                    })
                    .collect::<Vec<_>>();
                let mut intersection_job_ids = FxHashMap::<(usize, usize), usize>::default();
                let mut intersection_jobs = Vec::<(Weight, Weight)>::new();
                let mut indexed_rows = Vec::<((u32, usize), SmallVec<[(usize, usize); 8]>)>::with_capacity(row_specs.len());
                for (row_key, prefix, edges) in row_specs {
                    let prefix_key = prefix.ptr_key();
                    let mut indexed = SmallVec::<[(usize, usize); 8]>::new();
                    for (edge_key, edge_weight) in edges {
                        let pair_key = if prefix_key <= edge_key {
                            (prefix_key, edge_key)
                        } else {
                            (edge_key, prefix_key)
                        };
                        let job = if let Some(&job) = intersection_job_ids.get(&pair_key) {
                            job
                        } else {
                            let job = intersection_jobs.len();
                            intersection_job_ids.insert(pair_key, job);
                            intersection_jobs.push((prefix.clone(), edge_weight));
                            job
                        };
                        indexed.push((edge_key, job));
                    }
                    indexed_rows.push((row_key, indexed));
                }
                profiled_residual_row_prefetch_intersection_jobs += intersection_jobs.len();
                let intersections = intersection_jobs
                    .par_iter()
                    .map(|(left, right)| left.intersection_uncached(right))
                    .collect::<Vec<_>>();
                indexed_rows
                    .into_iter()
                    .map(|(row_key, indexed)| {
                        let row = indexed
                            .into_iter()
                            .map(|(edge_key, job)| (edge_key, intersections[job].clone()))
                            .collect::<ResidualEdgeIntersections>();
                        (row_key, Arc::new(row))
                    })
                    .collect::<Vec<_>>()
            } else {
                jobs
                    .par_iter()
                    .map(|(key, (raw_state, prefix))| {
                        let source = &raw_states[*raw_state as usize];
                        let mut weight_ops = ScopedWeightOpCache::default();
                        let mut edge_weights = ResidualEdgeIntersections::new();
                        for targets in source.transitions.values() {
                            for (_, edge_weight) in targets {
                                let edge_key = edge_weight.ptr_key();
                                if edge_weights.iter().any(|(key, _)| *key == edge_key) {
                                    continue;
                                }
                                edge_weights.push((
                                    edge_key,
                                    weight_ops.intersection(prefix, edge_weight),
                                ));
                            }
                        }
                        edge_weights.sort_unstable_by_key(|(key, _)| *key);
                        (*key, Arc::new(edge_weights))
                    })
                    .collect::<Vec<_>>()
            };
            profiled_row_cache_build_intersections.set(
                profiled_row_cache_build_intersections.get()
                    + prepared.iter().map(|(_, row)| row.len()).sum::<usize>(),
            );
            profiled_residual_row_prefetch_rows += prepared.len();
            for (key, row) in prepared {
                cached_residual_rows.entry(key).or_insert(row);
            }
            profiled_residual_row_prefetch_batches += 1;
            profiled_residual_row_prefetch_ms += started_at.elapsed().as_secs_f64() * 1000.0;
            residual_row_prefetch_remaining = covered_states;
        }
        if use_parallel_residual_row_prefetch {
            residual_row_prefetch_remaining = residual_row_prefetch_remaining.saturating_sub(1);
        }
        profiled_max_subset = profiled_max_subset.max(subset.len());
        match subset.len() {
            1 => profiled_singletons += 1,
            2 => profiled_pair_subsets += 1,
            _ => profiled_wide_subsets += 1,
        }
        // Singleton states are always interned with an all-weight prefix. When
        // the raw row is already deterministic, copy it directly instead of
        // rebuilding the same row through label grouping, hash maps, and
        // weight normalization. Only genuine overlap subsets need the general
        // determinization path below.
        if subset.len() == 1 && subset[0].1.is_full() {
            let raw_state = subset[0].0;
            let source = &raw_states[raw_state as usize];
            if source.transitions.values().all(|targets| targets.len() <= 1) {
                let mut output_transitions = Vec::with_capacity(source.transitions.len());
                let final_weight = source
                    .final_weight
                    .as_ref()
                    .filter(|weight| !weight.is_empty())
                    .cloned();
                let source_has_default = source.transitions.contains_key(&DEFAULT_LABEL);
                for (&label, targets) in &source.transitions {
                    let Some((target, edge_weight)) = targets.first() else {
                        continue;
                    };
                    if edge_weight.is_empty()
                        && !(source_has_default && label >= 0 && label != DEFAULT_LABEL)
                    {
                        continue;
                    }
                    let target = intern_singleton(
                        *target,
                        &mut singleton_states,
                        &mut singleton_count,
                        &mut states,
                        &mut queue,
                    );
                    output_transitions.push((label, (target, edge_weight.clone())));
                }
                states[output_state as usize] = DWAState {
                    transitions: output_transitions.into_iter().collect(),
                    final_weight,
                };
                continue;
            }
        }

        let mut final_parts = SmallVec::<[Weight; 4]>::new();
        for (raw_state, prefix_weight) in &subset {
            let source = &raw_states[*raw_state as usize];
            if let Some(final_weight) = &source.final_weight {
                let contribution = weight_ops.intersection(prefix_weight, final_weight);
                if !contribution.is_empty() {
                    final_parts.push(contribution);
                }
            }
        }
        // Keep wildcard transitions symbolic. Process one label at a time so
        // synthetic rows do not allocate a nested label->target hash table and
        // sort it again afterward. Raw rows are deterministic in the common
        // path; the small target vector also handles genuine NWA overlap.
        let final_weight = match final_parts.len() {
            0 => None,
            1 => final_parts.pop(),
            _ if defer_support_unions => {
                final_parts.sort_unstable_by_key(Weight::ptr_key);
                let key = final_parts
                    .iter()
                    .map(Weight::ptr_key)
                    .collect::<SmallVec<[usize; 4]>>();
                let job = if let Some(&job) = deferred_final_union_ids.get(&key) {
                    job
                } else {
                    let job = deferred_final_union_jobs.len();
                    deferred_final_union_ids.insert(key, job);
                    deferred_final_union_jobs.push(final_parts);
                    job
                };
                deferred_final_weight_patches.push((output_state, job));
                None
            }
            _ => {
                let weight = weight_ops.union_all(final_parts.iter());
                (!weight.is_empty()).then_some(weight)
            }
        };
        let mut output_transitions = Vec::<(i32, (u32, Weight))>::new();

        if subset.len() == 2 {
            let owner_pair = (
                raw_owners[subset[0].0 as usize],
                raw_owners[subset[1].0 as usize],
            );
            *profiled_pair_owner_counts.entry(owner_pair).or_default() += 1;
            if owner_pair.1 == 2 {
                *profiled_boundary_pair_states
                    .entry(raw_locals[subset[1].0 as usize])
                    .or_default() += 1;
            } else if owner_pair.0 == 2 {
                *profiled_boundary_pair_states
                    .entry(raw_locals[subset[0].0 as usize])
                    .or_default() += 1;
            }
            let left_source = &raw_states[subset[0].0 as usize];
            let right_source = &raw_states[subset[1].0 as usize];
            let left_prefix = &subset[0].1;
            let right_prefix = &subset[1].1;
            profiled_pair_left_raw_states.insert(subset[0].0);
            profiled_pair_right_raw_states.insert(subset[1].0);
            profiled_pair_left_residuals.insert((subset[0].0, left_prefix.ptr_key()));
            profiled_pair_right_residuals.insert((subset[1].0, right_prefix.ptr_key()));
            let mut prepare_residual_row = |
                raw_state: u32,
                prefix: &Weight,
                source: &crate::automata::weighted_u32::nwa::NWAState,
                weight_ops: &mut ScopedWeightOpCache,
            | -> Option<Arc<ResidualEdgeIntersections>> {
                if !use_residual_row_cache {
                    return None;
                }
                let key = (raw_state, prefix.ptr_key());
                if let Some(cached) = cached_residual_rows.get(&key) {
                    return Some(Arc::clone(cached));
                }
                let mut edge_weights = ResidualEdgeIntersections::new();
                for targets in source.transitions.values() {
                    for (_, edge_weight) in targets {
                        let edge_key = edge_weight.ptr_key();
                        if edge_weights.iter().any(|(key, _)| *key == edge_key) {
                            continue;
                        }
                        profiled_row_cache_build_intersections.set(
                            profiled_row_cache_build_intersections.get() + 1,
                        );
                        edge_weights.push((
                            edge_key,
                            weight_ops.intersection(prefix, edge_weight),
                        ));
                    }
                }
                edge_weights.sort_unstable_by_key(|(key, _)| *key);
                let cached = Arc::new(edge_weights);
                cached_residual_rows.insert(key, Arc::clone(&cached));
                Some(cached)
            };
            let left_residual_row = prepare_residual_row(
                subset[0].0,
                left_prefix,
                left_source,
                &mut weight_ops,
            );
            let right_residual_row = prepare_residual_row(
                subset[1].0,
                right_prefix,
                right_source,
                &mut weight_ops,
            );
            let restricted_intersection = |
                cached: Option<&ResidualEdgeIntersections>,
                prefix: &Weight,
                edge_weight: &Weight,
                weight_ops: &mut ScopedWeightOpCache,
            | -> Weight {
                if let Some(cached) = cached {
                    profiled_local_intersection_lookups.set(
                        profiled_local_intersection_lookups.get() + 1,
                    );
                    let key = edge_weight.ptr_key();
                    let index = cached
                        .binary_search_by_key(&key, |(cached_key, _)| *cached_key)
                        .expect("residual row cache must cover every source edge weight");
                    cached[index].1.clone()
                } else {
                    profiled_global_intersection_lookups.set(
                        profiled_global_intersection_lookups.get() + 1,
                    );
                    weight_ops.intersection(prefix, edge_weight)
                }
            };
            if left_prefix.is_full() {
                profiled_pair_left_prefix_full += 1;
            }
            if right_prefix.is_full() {
                profiled_pair_right_prefix_full += 1;
            }
            let left_default = left_source.transitions.get(&DEFAULT_LABEL);
            let right_default = right_source.transitions.get(&DEFAULT_LABEL);
            let has_symbolic_default = default_positive_label_count.is_some()
                && (left_default.is_some() || right_default.is_some());
            let build_default_contributions = |
                targets: Option<&Vec<(u32, Weight)>>,
                cached: Option<&ResidualEdgeIntersections>,
                prefix: &Weight,
                weight_ops: &mut ScopedWeightOpCache,
            | {
                let mut contributions = SmallVec::<[(u32, Weight); 2]>::new();
                if let Some(targets) = targets {
                    for (target, edge_weight) in targets {
                        let contribution = restricted_intersection(
                            cached,
                            prefix,
                            edge_weight,
                            weight_ops,
                        );
                        if !contribution.is_empty() {
                            contributions.push((*target, contribution));
                        }
                    }
                }
                contributions
            };
            let left_default_contributions = build_default_contributions(
                left_default,
                left_residual_row.as_deref(),
                left_prefix,
                &mut weight_ops,
            );
            let right_default_contributions = build_default_contributions(
                right_default,
                right_residual_row.as_deref(),
                right_prefix,
                &mut weight_ops,
            );
            if use_interval_pair_runs {
                // Exact run-wise version of the label merge below.  Every
                // boundary is a start/end+1 of a maximal source run (plus 0,
                // where DEFAULT begins to apply).  Therefore both raw target
                // vectors and both residual-prefix intersections are constant
                // throughout each interval.  Solve the weighted successor once
                // per interval, then expand that already-computed ordinary DWA
                // transition across its explicit labels.
                let left_runs = explicit_transition_runs(left_source);
                let right_runs = explicit_transition_runs(right_source);
                let mut boundaries = Vec::<i64>::with_capacity(
                    2 * (left_runs.len() + right_runs.len()) + 1,
                );
                for (start, end, _) in left_runs.iter().chain(right_runs.iter()) {
                    boundaries.push(i64::from(*start));
                    boundaries.push(i64::from(*end) + 1);
                }
                if has_symbolic_default {
                    // Negative explicit labels never fall through to DEFAULT;
                    // non-negative labels do.  No source run can cross the
                    // excluded DEFAULT_LABEL itself, but 0 must still be an
                    // interval boundary for the fallback rule.
                    boundaries.push(0);
                }
                boundaries.sort_unstable();
                boundaries.dedup();

                let mut left_position = 0usize;
                let mut right_position = 0usize;
                for interval in boundaries.windows(2) {
                    let interval_start = interval[0];
                    let interval_end = interval[1] - 1;
                    if interval_start > interval_end
                        || interval_start < i64::from(i32::MIN)
                        || interval_end > i64::from(i32::MAX)
                    {
                        continue;
                    }
                    let label = interval_start as i32;
                    while left_position < left_runs.len()
                        && left_runs[left_position].1 < label
                    {
                        left_position += 1;
                    }
                    while right_position < right_runs.len()
                        && right_runs[right_position].1 < label
                    {
                        right_position += 1;
                    }
                    let left_explicit = left_runs.get(left_position).and_then(|run| {
                        (run.0 <= label && label <= run.1).then_some(run.2)
                    });
                    let right_explicit = right_runs.get(right_position).and_then(|run| {
                        (run.0 <= label && label <= run.1).then_some(run.2)
                    });
                    if left_explicit.is_none() && right_explicit.is_none() {
                        // A gap covered only by DEFAULT is represented by the
                        // one symbolic DEFAULT transition below, never by a
                        // redundant explicit row.
                        continue;
                    }
                    let span = (interval_end - interval_start + 1) as usize;
                    profiled_explicit_labels += span;
                    let left_targets = left_explicit.or_else(|| {
                        (label >= 0).then_some(left_default).flatten()
                    });
                    let right_targets = right_explicit.or_else(|| {
                        (label >= 0).then_some(right_default).flatten()
                    });
                    match (left_targets.is_some(), right_targets.is_some()) {
                        (true, true) => profiled_pair_both_labels += span,
                        (true, false) => profiled_pair_left_only_labels += span,
                        (false, true) => profiled_pair_right_only_labels += span,
                        (false, false) => unreachable!(
                            "run interval has an explicit label on at least one side"
                        ),
                    }

                    let mut contributions = SmallVec::<[(u32, Weight); 4]>::new();
                    if let Some(targets) = left_explicit {
                        for (target, edge_weight) in targets {
                            let contribution = restricted_intersection(
                                left_residual_row.as_deref(),
                                left_prefix,
                                edge_weight,
                                &mut weight_ops,
                            );
                            if !contribution.is_empty() {
                                contributions.push((*target, contribution));
                            }
                        }
                    } else if left_targets.is_some() {
                        profiled_pair_cached_default_uses += span;
                        contributions.extend(left_default_contributions.iter().cloned());
                    }
                    if let Some(targets) = right_explicit {
                        for (target, edge_weight) in targets {
                            let contribution = restricted_intersection(
                                right_residual_row.as_deref(),
                                right_prefix,
                                edge_weight,
                                &mut weight_ops,
                            );
                            if !contribution.is_empty() {
                                contributions.push((*target, contribution));
                            }
                        }
                    } else if right_targets.is_some() {
                        profiled_pair_cached_default_uses += span;
                        contributions.extend(right_default_contributions.iter().cloned());
                    }

                    if let Some((target, edge_weight)) = finish_overlap_transition(
                        contributions,
                        &mut weight_ops,
                        defer_support_unions,
                        &mut deferred_union_ids,
                        &mut deferred_union_jobs,
                        &mut singleton_states,
                        &mut singleton_count,
                        &mut states,
                        &mut queue,
                        &mut subset_states,
                    ) {
                        append_pending_transition_run(
                            output_state,
                            interval_start as i32,
                            interval_end as i32,
                            target,
                            edge_weight,
                            &mut output_transitions,
                            &mut deferred_weight_patches,
                            false,
                        );
                    } else if label >= 0 && has_symbolic_default {
                        let dead = *dead_shadow_state.get_or_insert_with(|| {
                            let dead = states.len() as u32;
                            states.push(DWAState::default());
                            dead
                        });
                        for explicit_label in (interval_start as i32)..=(interval_end as i32) {
                            output_transitions.push((
                                explicit_label,
                                (dead, Weight::empty()),
                            ));
                        }
                    }
                }
            } else {
            let mut left = left_source
                .transitions
                .iter()
                .filter(|(label, _)| **label != DEFAULT_LABEL)
                .peekable();
            let mut right = right_source
                .transitions
                .iter()
                .filter(|(label, _)| **label != DEFAULT_LABEL)
                .peekable();

            loop {
                let left_label = left.peek().map(|(label, _)| **label);
                let right_label = right.peek().map(|(label, _)| **label);
                let Some(label) = (match (left_label, right_label) {
                    (Some(left), Some(right)) => Some(left.min(right)),
                    (Some(left), None) => Some(left),
                    (None, Some(right)) => Some(right),
                    (None, None) => None,
                }) else {
                    break;
                };
                profiled_explicit_labels += 1;
                let left_targets = if left_label == Some(label) {
                    left.next().map(|(_, targets)| targets)
                } else if label >= 0 {
                    left_default
                } else {
                    None
                };
                let right_targets = if right_label == Some(label) {
                    right.next().map(|(_, targets)| targets)
                } else if label >= 0 {
                    right_default
                } else {
                    None
                };
                match (left_targets.is_some(), right_targets.is_some()) {
                    (true, true) => profiled_pair_both_labels += 1,
                    (true, false) => profiled_pair_left_only_labels += 1,
                    (false, true) => profiled_pair_right_only_labels += 1,
                    (false, false) => unreachable!("merged explicit label must exist on at least one side"),
                }
                let mut contributions = SmallVec::<[(u32, Weight); 4]>::new();
                if left_label == Some(label) {
                    if let Some(targets) = left_targets {
                        for (target, edge_weight) in targets {
                            let contribution = restricted_intersection(
                                left_residual_row.as_deref(),
                                left_prefix,
                                edge_weight,
                                &mut weight_ops,
                            );
                            if !contribution.is_empty() {
                                contributions.push((*target, contribution));
                            }
                        }
                    }
                } else if left_targets.is_some() {
                    profiled_pair_cached_default_uses += 1;
                    contributions.extend(left_default_contributions.iter().cloned());
                }
                if right_label == Some(label) {
                    if let Some(targets) = right_targets {
                        for (target, edge_weight) in targets {
                            let contribution = restricted_intersection(
                                right_residual_row.as_deref(),
                                right_prefix,
                                edge_weight,
                                &mut weight_ops,
                            );
                            if !contribution.is_empty() {
                                contributions.push((*target, contribution));
                            }
                        }
                    }
                } else if right_targets.is_some() {
                    profiled_pair_cached_default_uses += 1;
                    contributions.extend(right_default_contributions.iter().cloned());
                }
                if let Some((target, edge_weight)) = finish_overlap_transition(
                    contributions,
                    &mut weight_ops,
                    defer_support_unions,
                    &mut deferred_union_ids,
                    &mut deferred_union_jobs,
                    &mut singleton_states,
                    &mut singleton_count,
                    &mut states,
                    &mut queue,
                    &mut subset_states,
                ) {
                    append_pending_transition_run(
                        output_state,
                        label,
                        label,
                        target,
                        edge_weight,
                        &mut output_transitions,
                        &mut deferred_weight_patches,
                        false,
                    );
                } else if label >= 0 && has_symbolic_default {
                    let dead = *dead_shadow_state.get_or_insert_with(|| {
                        let dead = states.len() as u32;
                        states.push(DWAState::default());
                        dead
                    });
                    output_transitions.push((label, (dead, Weight::empty())));
                }
            }
            }

            if has_symbolic_default {
                let mut contributions = SmallVec::<[(u32, Weight); 4]>::new();
                contributions.extend(left_default_contributions.iter().cloned());
                contributions.extend(right_default_contributions.iter().cloned());
                if let Some((target, edge_weight)) = finish_overlap_transition(
                    contributions,
                    &mut weight_ops,
                    defer_support_unions,
                    &mut deferred_union_ids,
                    &mut deferred_union_jobs,
                    &mut singleton_states,
                    &mut singleton_count,
                    &mut states,
                    &mut queue,
                    &mut subset_states,
                ) {
                    append_pending_transition_run(
                        output_state,
                        DEFAULT_LABEL,
                        DEFAULT_LABEL,
                        target,
                        edge_weight,
                        &mut output_transitions,
                        &mut deferred_weight_patches,
                        true,
                    );
                }
            }
            states[output_state as usize] = DWAState {
                transitions: output_transitions.into_iter().collect(),
                final_weight,
            };
            continue;
        }

        let mut explicit_labels = SmallVec::<[i32; 32]>::new();
        for (raw_state, _) in &subset {
            explicit_labels.extend(
                raw_states[*raw_state as usize]
                    .transitions
                    .keys()
                    .copied()
                    .filter(|&label| label != DEFAULT_LABEL),
            );
        }
        explicit_labels.sort_unstable();
        explicit_labels.dedup();
        profiled_explicit_labels += explicit_labels.len();

        let include_default = default_positive_label_count.is_some()
            && subset.iter().any(|(raw_state, _)| {
                raw_states[*raw_state as usize]
                    .transitions
                    .contains_key(&DEFAULT_LABEL)
            });
        let labels = explicit_labels
            .iter()
            .copied()
            .chain(include_default.then_some(DEFAULT_LABEL));
        for label in labels {
            let mut contributions = SmallVec::<[(u32, Weight); 4]>::new();
            for (raw_state, prefix_weight) in &subset {
                let source = &raw_states[*raw_state as usize];
                let targets = if label == DEFAULT_LABEL {
                    source.transitions.get(&DEFAULT_LABEL)
                } else {
                    source.transitions.get(&label).or_else(|| {
                        (label >= 0)
                            .then(|| source.transitions.get(&DEFAULT_LABEL))
                            .flatten()
                    })
                };
                let Some(targets) = targets else {
                    continue;
                };
                for (target, edge_weight) in targets {
                    let contribution = weight_ops.intersection(prefix_weight, edge_weight);
                    if !contribution.is_empty() {
                        contributions.push((*target, contribution));
                    }
                }
            }
            if let Some((target, edge_weight)) = finish_overlap_transition(
                contributions,
                &mut weight_ops,
                defer_support_unions,
                &mut deferred_union_ids,
                &mut deferred_union_jobs,
                &mut singleton_states,
                &mut singleton_count,
                &mut states,
                &mut queue,
                &mut subset_states,
            ) {
                append_pending_transition_run(
                    output_state,
                    label,
                    label,
                    target,
                    edge_weight,
                    &mut output_transitions,
                    &mut deferred_weight_patches,
                    label == DEFAULT_LABEL,
                );
            } else if label != DEFAULT_LABEL && label >= 0 && include_default {
                let dead = *dead_shadow_state.get_or_insert_with(|| {
                    let dead = states.len() as u32;
                    states.push(DWAState::default());
                    dead
                });
                output_transitions.push((label, (dead, Weight::empty())));
            }
        }
        states[output_state as usize] = DWAState {
            transitions: output_transitions.into_iter().collect(),
            final_weight,
        };
    }

    let deferred_union_started_at = Instant::now();
    let deferred_union_results = if defer_support_unions {
        deferred_union_jobs
            .par_iter()
            .map(|(left, right)| Weight::union_all_direct([left, right]))
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let deferred_union_ms = deferred_union_started_at.elapsed().as_secs_f64() * 1000.0;
    let deferred_final_union_started_at = Instant::now();
    let deferred_final_union_results = if defer_support_unions {
        deferred_final_union_jobs
            .par_iter()
            .map(|weights| Weight::union_all_direct(weights.iter()))
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let deferred_final_union_ms =
        deferred_final_union_started_at.elapsed().as_secs_f64() * 1000.0;

    let deferred_patch_started_at = Instant::now();
    if defer_support_unions {
        for patch in &deferred_weight_patches {
            let weight = &deferred_union_results[patch.job];
            debug_assert!(!weight.is_empty());
            for (_, (_, edge_weight)) in states[patch.state as usize]
                .transitions
                .range_mut(patch.start..=patch.end)
            {
                *edge_weight = weight.clone();
            }
        }
        for &(state, job) in &deferred_final_weight_patches {
            let weight = deferred_final_union_results[job].clone();
            debug_assert!(!weight.is_empty());
            states[state as usize].final_weight = Some(weight);
        }
    }
    let deferred_patch_ms = deferred_patch_started_at.elapsed().as_secs_f64() * 1000.0;

    let synthetic_states = states.len().saturating_sub(singleton_count);
    if compose_profile_enabled() {
        let left_residual_row_labels = profiled_pair_left_residuals
            .iter()
            .map(|(state, _)| raw_states[*state as usize].transitions.len())
            .sum::<usize>();
        let right_residual_row_labels = profiled_pair_right_residuals
            .iter()
            .map(|(state, _)| raw_states[*state as usize].transitions.len())
            .sum::<usize>();
        let count_distinct_row_weights = |state: u32| {
            raw_states[state as usize]
                .transitions
                .values()
                .flat_map(|targets| targets.iter().map(|(_, weight)| weight.ptr_key()))
                .collect::<FxHashSet<_>>()
                .len()
        };
        let count_explicit_row_runs = |state: u32| {
            let source = &raw_states[state as usize];
            let mut labels = 0usize;
            let mut runs = 0usize;
            let mut previous: Option<(i32, &Vec<(u32, Weight)>)> = None;
            for (&label, targets) in source
                .transitions
                .iter()
                .filter(|(label, _)| **label != DEFAULT_LABEL)
            {
                labels += 1;
                let continues = previous.is_some_and(|(previous_label, previous_targets)| {
                    previous_label.checked_add(1) == Some(label)
                        && same_raw_targets(previous_targets, targets)
                });
                if !continues {
                    runs += 1;
                }
                previous = Some((label, targets));
            }
            (labels, runs)
        };
        let (left_residual_explicit_labels, left_residual_runs) = profiled_pair_left_residuals
            .iter()
            .map(|(state, _)| count_explicit_row_runs(*state))
            .fold((0usize, 0usize), |(labels, runs), next| {
                (labels + next.0, runs + next.1)
            });
        let (right_residual_explicit_labels, right_residual_runs) = profiled_pair_right_residuals
            .iter()
            .map(|(state, _)| count_explicit_row_runs(*state))
            .fold((0usize, 0usize), |(labels, runs), next| {
                (labels + next.0, runs + next.1)
            });
        let left_residual_row_weights = profiled_pair_left_residuals
            .iter()
            .map(|(state, _)| count_distinct_row_weights(*state))
            .sum::<usize>();
        let right_residual_row_weights = profiled_pair_right_residuals
            .iter()
            .map(|(state, _)| count_distinct_row_weights(*state))
            .sum::<usize>();
        eprintln!(
            "[glrmask/profile][constraint_overlap_pair_owners] counts={profiled_pair_owner_counts:?}"
        );
        let mut boundary_pair_states = profiled_boundary_pair_states.into_iter().collect::<Vec<_>>();
        boundary_pair_states.sort_unstable_by_key(|&(state, count)| (std::cmp::Reverse(count), state));
        boundary_pair_states.truncate(24);
        eprintln!(
            "[glrmask/profile][constraint_overlap_boundary_state_pairs] top={boundary_pair_states:?}"
        );
        eprintln!(
            "[glrmask/profile][constraint_overlap_local_shape] raw_states={} result_states={} singleton_states={} raw_singletons_preallocated={} pair_subsets={} wide_subsets={} max_subset={} explicit_labels={} pair_left_only_labels={} pair_right_only_labels={} pair_both_labels={} pair_left_prefix_full={} pair_right_prefix_full={} pair_cached_default_uses={} left_raw_states={} right_raw_states={} left_residual_rows={} right_residual_rows={} left_residual_row_labels={} right_residual_row_labels={} left_residual_explicit_labels={} right_residual_explicit_labels={} left_residual_runs={} right_residual_runs={} left_residual_row_weights={} right_residual_row_weights={} cached_residual_rows={} cached_residual_edge_weights={} local_intersection_lookups={} global_intersection_lookups={} row_cache_build_intersections={} residual_prefetch_batches={} residual_prefetch_rows={} residual_prefetch_intersection_jobs={} residual_prefetch_ms={:.3} union_cache_entries={} intersection_cache_entries={} deferred_union_jobs={} deferred_patch_runs={} deferred_final_union_jobs={} deferred_final_patches={} deferred_union_ms={deferred_union_ms:.3} deferred_final_union_ms={deferred_final_union_ms:.3} deferred_patch_ms={deferred_patch_ms:.3}",
            raw_states.len(),
            states.len(),
            profiled_singletons,
            preallocate_raw_singletons,
            profiled_pair_subsets,
            profiled_wide_subsets,
            profiled_max_subset,
            profiled_explicit_labels,
            profiled_pair_left_only_labels,
            profiled_pair_right_only_labels,
            profiled_pair_both_labels,
            profiled_pair_left_prefix_full,
            profiled_pair_right_prefix_full,
            profiled_pair_cached_default_uses,
            profiled_pair_left_raw_states.len(),
            profiled_pair_right_raw_states.len(),
            profiled_pair_left_residuals.len(),
            profiled_pair_right_residuals.len(),
            left_residual_row_labels,
            right_residual_row_labels,
            left_residual_explicit_labels,
            right_residual_explicit_labels,
            left_residual_runs,
            right_residual_runs,
            left_residual_row_weights,
            right_residual_row_weights,
            cached_residual_rows.len(),
            cached_residual_rows.values().map(|row| row.len()).sum::<usize>(),
            profiled_local_intersection_lookups.get(),
            profiled_global_intersection_lookups.get(),
            profiled_row_cache_build_intersections.get(),
            profiled_residual_row_prefetch_batches,
            profiled_residual_row_prefetch_rows,
            profiled_residual_row_prefetch_intersection_jobs,
            profiled_residual_row_prefetch_ms,
            weight_ops.union_entry_count(),
            weight_ops.intersection_entry_count(),
            deferred_union_jobs.len(),
            deferred_weight_patches.len(),
            deferred_final_union_jobs.len(),
            deferred_final_weight_patches.len(),
        );
    }
    Some((DWA::from_parts(states, start_state), synthetic_states))
}


pub(super) struct UnmappedComponentParserArtifact {
    pub(super) automaton: NWA,
    pub(super) possible_matches: PossibleMatches,
}

pub(super) fn prepare_unmapped_component_parser_artifacts(
    components: &[ParserDwaComponent<'_>],
    terminal_offsets: &[u32],
    default_domains: &[Option<ParserDefaultDomain>],
    strip_scoped_ignore_identity: bool,
) -> Result<Vec<UnmappedComponentParserArtifact>, String> {
    if components.len() != terminal_offsets.len() || components.len() != default_domains.len() {
        return Err("component/parser terminal-offset/default-domain count mismatch".into());
    }
    let build = |index: usize| {
            let component = components[index];
            let terminal_offset = terminal_offsets[index];
            let default_domain = &default_domains[index];
            let possible_matches = component_possible_matches(&component, terminal_offset)?;
            let mut automaton = component_parser_nwa(&component, default_domain.as_ref())?;
            if strip_scoped_ignore_identity {
                let ignore_weight = component
                    .constraint
                    .ignore_terminal
                    .and_then(|ignore| possible_matches.get(&(terminal_offset + ignore)));
                strip_unscoped_ignore_identity(&mut automaton, ignore_weight);
            }
            Ok(UnmappedComponentParserArtifact {
                automaton,
                possible_matches,
            })
    };
    if macro_parallelism_disabled() {
        let mut timings = Vec::with_capacity(components.len());
        let result = (0..components.len())
            .map(|index| {
                let started = Instant::now();
                let result = build(index);
                timings.push(started.elapsed().as_secs_f64() * 1000.0);
                result
            })
            .collect();
        report_macro_item_timings("compose_component_parser_artifacts", &timings);
        result
    } else {
        (0..components.len()).into_par_iter().map(build).collect()
    }
}

pub(super) fn prepare_unmapped_component_parser_automata(
    components: &[ParserDwaComponent<'_>],
    default_domains: &[Option<ParserDefaultDomain>],
    strip_scoped_ignore_identity: bool,
) -> Result<Vec<NWA>, String> {
    if components.len() != default_domains.len() {
        return Err("component/parser default-domain count mismatch".into());
    }
    let build = |index: usize| {
            let component = components[index];
            let default_domain = &default_domains[index];
            let mut automaton = component_parser_nwa(&component, default_domain.as_ref())?;
            if strip_scoped_ignore_identity {
                let ignore_weight = component
                    .constraint
                    .ignore_terminal
                    .and_then(|ignore| component.constraint.possible_matches.get(&ignore));
                strip_unscoped_ignore_identity(&mut automaton, ignore_weight);
            }
            Ok(automaton)
    };
    if macro_parallelism_disabled() {
        let mut timings = Vec::with_capacity(components.len());
        let result = (0..components.len())
            .map(|index| {
                let started = Instant::now();
                let result = build(index);
                timings.push(started.elapsed().as_secs_f64() * 1000.0);
                result
            })
            .collect();
        report_macro_item_timings("compose_component_parser_automata", &timings);
        result
    } else {
        (0..components.len()).into_par_iter().map(build).collect()
    }
}

/// Exact deferred union of already-compiled component parser DWAs.
///
/// The semantic union is represented by multiple NWA start components; no
/// parser row, parser weight, or mapped transition is copied while constructing
/// the view.  In the overwhelmingly common composition case every local LR
/// state has exactly one composed-state image, so retain that relation as one
/// flat `u32` per local state.  The eventual component+boundary parser consumer
/// may materialize the view and determinize the combined NWA in one pass.
///
/// This intentionally makes the *component union itself* the cheap operation:
/// expensive parser-state label transport and weight-coordinate publication are
/// properties of later realization, not of forming the union language.
pub(super) enum DeferredParserStateRelation {
    Singleton(Vec<u32>),
    General(Vec<Vec<u32>>),
}

impl DeferredParserStateRelation {
    pub(super) fn from_relation(relation: &[Vec<u32>]) -> Self {
        if relation.iter().all(|targets| targets.len() == 1) {
            Self::Singleton(relation.iter().map(|targets| targets[0]).collect())
        } else {
            Self::General(relation.to_vec())
        }
    }

    pub(super) fn materialize(&self) -> std::borrow::Cow<'_, [Vec<u32>]> {
        match self {
            Self::Singleton(targets) => std::borrow::Cow::Owned(
                targets.iter().map(|&target| vec![target]).collect(),
            ),
            Self::General(relation) => std::borrow::Cow::Borrowed(relation),
        }
    }

    pub(super) fn is_singleton(&self) -> bool {
        matches!(self, Self::Singleton(_))
    }
}

pub(super) struct DeferredComponentParserUnionComponent<'a> {
    pub(super) constraint: &'a Constraint,
    pub(super) parser_state_relation: DeferredParserStateRelation,
    pub(super) tokenizer_state_offset: u32,
    pub(super) terminal_offset: u32,
    pub(super) default_domain: Option<ParserDefaultDomain>,
}

pub(super) struct DeferredComponentParserUnionView<'a> {
    pub(super) components: Vec<DeferredComponentParserUnionComponent<'a>>,
    pub(super) strip_scoped_ignore_identity: bool,
}

pub(super) enum ComponentParserUnionWork<'a> {
    Deferred(DeferredComponentParserUnionView<'a>),
    Materialized(Vec<NWA>),
}

impl ComponentParserUnionWork<'_> {
    pub(super) fn materialize_automata(self) -> Result<Vec<NWA>, String> {
        match self {
            Self::Deferred(view) => view.materialize_automata(),
            Self::Materialized(automata) => Ok(automata),
        }
    }

    pub(super) fn view_shape(&self) -> Option<(usize, usize)> {
        match self {
            Self::Deferred(view) => Some((
                view.component_count(),
                view.singleton_relation_count(),
            )),
            Self::Materialized(_) => None,
        }
    }
}

impl<'a> DeferredComponentParserUnionView<'a> {
    pub(super) fn new(
        constraints: &[&'a Constraint],
        parser_state_relations: &[Vec<Vec<u32>>],
        tokenizer_state_offsets: &[u32],
        terminal_offsets: &[u32],
        default_domains: &[Option<ParserDefaultDomain>],
        strip_scoped_ignore_identity: bool,
    ) -> Option<Self> {
        if constraints.len() != parser_state_relations.len()
            || constraints.len() != tokenizer_state_offsets.len()
            || constraints.len() != terminal_offsets.len()
            || constraints.len() != default_domains.len()
            || (std::env::var_os("GLRMASK_EXPERIMENT_COMPONENT_SCOPED_IGNORE_TOP_ACCEPT").is_some()
                && std::env::var_os("GLRMASK_EXPERIMENT_EXTERNALIZE_COMPONENT_TOP_ACCEPT").is_none())
        {
            return None;
        }
        Some(Self {
            components: constraints
                .iter()
                .copied()
                .zip(parser_state_relations)
                .zip(tokenizer_state_offsets.iter().copied())
                .zip(terminal_offsets.iter().copied())
                .zip(default_domains.iter().cloned())
                .map(
                    |((((constraint, relation), tokenizer_state_offset), terminal_offset), default_domain)| DeferredComponentParserUnionComponent {
                    constraint,
                    parser_state_relation: DeferredParserStateRelation::from_relation(
                        relation,
                    ),
                    tokenizer_state_offset,
                    terminal_offset,
                    default_domain,
                },
                )
                .collect(),
            strip_scoped_ignore_identity,
        })
    }

    pub(super) fn materialize_automata(self) -> Result<Vec<NWA>, String> {
        let strip_scoped_ignore_identity = self.strip_scoped_ignore_identity;
        let build = |component: DeferredComponentParserUnionComponent<'a>| {
                let relation = component.parser_state_relation.materialize();
                let parser_component = ParserDwaComponent {
                    constraint: component.constraint,
                    parser_state_relation: relation.as_ref(),
                    tokenizer_state_offset: component.tokenizer_state_offset,
                    terminal_offset: component.terminal_offset,
                    composed_table: None,
                };
                let mut automaton =
                    component_parser_nwa(&parser_component, component.default_domain.as_ref())?;
                if strip_scoped_ignore_identity {
                    let ignore_weight = component
                        .constraint
                        .ignore_terminal
                        .and_then(|ignore| component.constraint.possible_matches.get(&ignore));
                    strip_unscoped_ignore_identity(&mut automaton, ignore_weight);
                }
                Ok(automaton)
        };
        if macro_parallelism_disabled() {
            let mut timings = Vec::with_capacity(self.components.len());
            let result = self
                .components
                .into_iter()
                .map(|component| {
                    let started = Instant::now();
                    let result = build(component);
                    timings.push(started.elapsed().as_secs_f64() * 1000.0);
                    result
                })
                .collect();
            report_macro_item_timings("compose_deferred_component_parser_materialize", &timings);
            result
        } else {
            self.components.into_par_iter().map(build).collect()
        }
    }

    pub(super) fn component_count(&self) -> usize {
        self.components.len()
    }

    pub(super) fn singleton_relation_count(&self) -> usize {
        self.components
            .iter()
            .filter(|component| component.parser_state_relation.is_singleton())
            .count()
    }
}

pub(super) fn prepare_unmapped_component_possible_matches(
    components: &[ParserDwaComponent<'_>],
    terminal_offsets: &[u32],
) -> Result<Vec<PossibleMatches>, String> {
    if components.len() != terminal_offsets.len() {
        return Err("component/parser terminal-offset count mismatch".into());
    }
    let build = |index: usize| {
            let component = components[index];
            let terminal_offset = terminal_offsets[index];
            if component.constraint.possible_matches_complete {
                component_possible_matches(&component, terminal_offset)
            } else {
                // The explicit segmented runtime does not use a retained
                // dynamic component's absent static possible-matches table for
                // local masking. B is discovered from the exact tokenizer and
                // terminal expressions, while that component's A is evaluated
                // by its own dynamic backend. Keep the unified artifact's
                // optional static summary sparse rather than compiling the
                // dynamic component into a static quotient.
                Ok(PossibleMatches::new())
            }
    };
    if macro_parallelism_disabled() {
        let mut timings = Vec::with_capacity(components.len());
        let result = (0..components.len())
            .map(|index| {
                let started = Instant::now();
                let result = build(index);
                timings.push(started.elapsed().as_secs_f64() * 1000.0);
                result
            })
            .collect();
        report_macro_item_timings("compose_component_possible_matches", &timings);
        result
    } else {
        (0..components.len()).into_par_iter().map(build).collect()
    }
}

#[derive(Debug)]
pub(super) struct BoundaryRefinementPlan {
    pub(super) common_map: InternalIdMap,
    pub(super) component_token_map: Option<Vec<Vec<u32>>>,
    pub(super) boundary_tsid_map: Vec<Vec<u32>>,
    pub(super) boundary_token_map: Vec<Vec<u32>>,
}

pub(super) struct PreparedOwnedComponentArtifacts {
    /// Per-component maps into `id_map`. Parser automata are materialized only
    /// at the final parser-union boundary.
    pub(super) automata_maps: Vec<DirectComponentCoordinateMaps>,
    pub(super) possible_matches: PossibleMatches,
    pub(super) id_map: InternalIdMap,
    pub(super) boundary_tsid_map: Option<Vec<Vec<u32>>>,
    pub(super) boundary_token_map: Option<Vec<Vec<u32>>>,
    pub(super) token_mask_caches: Option<crate::runtime::InternalTokenMaskPrebuild>,
    pub(super) token_mask_cache_ms: f64,
    pub(super) remap_ms: f64,
}

pub(super) fn prebuild_segmented_token_mask_caches(
    id_map: &InternalIdMap,
) -> (Option<crate::runtime::InternalTokenMaskPrebuild>, f64) {
    if std::env::var_os("GLRMASK_EXPERIMENT_PREBUILD_SEGMENTED_TOKEN_MASK_CACHES").is_none() {
        return (None, 0.0);
    }
    let started_at = Instant::now();
    let caches = crate::runtime::InternalTokenMaskPrebuild::build(
        &id_map.vocab_tokens.original_to_internal,
        &id_map.vocab_tokens.internal_to_originals,
    );
    (Some(caches), started_at.elapsed().as_secs_f64() * 1000.0)
}

pub(super) fn prepare_deferred_component_artifacts(
    possible_matches_by_component: Vec<PossibleMatches>,
    component_maps: Vec<DirectComponentCoordinateMaps>,
    base_to_common_tokens: Option<&[Vec<u32>]>,
    common_tsid_count: usize,
) -> Result<(Vec<DirectComponentCoordinateMaps>, PossibleMatches, f64), String> {
    if possible_matches_by_component.len() != component_maps.len() {
        return Err("component artifact/map count mismatch".into());
    }
    let started_at = Instant::now();
    let prepare = |(mut possible_matches, mut maps): (
        PossibleMatches,
        DirectComponentCoordinateMaps,
    )| {
            if let Some(base_to_common) = base_to_common_tokens {
                maps.local_to_global_tokens =
                    compose_local_id_map(&maps.local_to_global_tokens, base_to_common);
            }
            let mut weights = possible_matches.weight_refs_mut();
            remap_weights_with_maps(
                &mut weights,
                &maps.local_to_global_tsids,
                &maps.local_to_global_tokens,
                common_tsid_count,
            );
            drop(weights);
            Ok::<_, String>((maps, possible_matches))
    };
    let prepared = if macro_parallelism_disabled() {
        let mut timings = Vec::with_capacity(possible_matches_by_component.len());
        let result = possible_matches_by_component
            .into_iter()
            .zip(component_maps)
            .map(|item| {
                let started = Instant::now();
                let result = prepare(item);
                timings.push(started.elapsed().as_secs_f64() * 1000.0);
                result
            })
            .collect::<Result<Vec<_>, String>>()?;
        report_macro_item_timings("compose_deferred_component_remap", &timings);
        result
    } else {
        possible_matches_by_component
            .into_par_iter()
            .zip(component_maps.into_par_iter())
            .map(prepare)
            .collect::<Result<Vec<_>, String>>()?
    };
    let mut maps = Vec::with_capacity(prepared.len());
    let mut possible_matches = PossibleMatches::new();
    for (map, component_possible_matches) in prepared {
        maps.push(map);
        for (terminal, weight) in component_possible_matches {
            possible_matches
                .entry(terminal)
                .and_modify(|existing| *existing = existing.union(&weight))
                .or_insert(weight);
        }
    }
    Ok((
        maps,
        possible_matches,
        started_at.elapsed().as_secs_f64() * 1000.0,
    ))
}

pub(super) fn build_boundary_refinement_plan(
    component_map: InternalIdMap,
    boundary_map: &InternalIdMap,
    component_token_coordinate_is_singleton: bool,
) -> Option<BoundaryRefinementPlan> {
    let boundary_tsid_map = boundary_map
        .tokenizer_states
        .internal_to_originals
        .par_iter()
        .map(|originals| {
            let mut mapped = Vec::new();
            for &original in originals {
                let component_tsid = component_map
                    .tokenizer_states
                    .original_to_internal
                    .get(original as usize)
                    .copied()
                    .unwrap_or(u32::MAX);
                if component_tsid == u32::MAX {
                    return None;
                }
                mapped.push(component_tsid);
            }
            mapped.sort_unstable();
            mapped.dedup();
            Some(mapped)
        })
        .collect::<Option<Vec<_>>>()?;

    // A singleton component token coordinate already refines every possible
    // boundary partition: each original model token is its own component
    // class. Keep that authoritative coordinate verbatim and construct only
    // the tiny boundary -> component map. Besides avoiding an O(|vocab|)
    // refinement proof, this tells the component remapper that there is no
    // additional component->common token transform to compose.
    if component_token_coordinate_is_singleton {
        let mut boundary_token_map =
            vec![Vec::<u32>::new(); boundary_map.num_internal_tokens() as usize];
        for (boundary_token, originals) in boundary_map
            .vocab_tokens
            .internal_to_originals
            .iter()
            .enumerate()
        {
            let destinations = &mut boundary_token_map[boundary_token];
            for &original in originals {
                let component_token = component_map
                    .vocab_tokens
                    .original_to_internal
                    .get(original as usize)
                    .copied()
                    .unwrap_or(u32::MAX);
                if component_token == u32::MAX {
                    return None;
                }
                destinations.push(component_token);
            }
            destinations.sort_unstable();
            destinations.dedup();
        }
        return Some(BoundaryRefinementPlan {
            common_map: component_map,
            component_token_map: None,
            boundary_tsid_map,
            boundary_token_map,
        });
    }

    let original_count = component_map
        .vocab_tokens
        .original_to_internal
        .len()
        .max(boundary_map.vocab_tokens.original_to_internal.len());
    let mut original_to_common = vec![u32::MAX; original_count];
    let mut common_to_originals = Vec::<Vec<u32>>::new();
    let mut representatives = Vec::<u32>::new();
    let mut component_token_map =
        vec![Vec::<u32>::new(); component_map.num_internal_tokens() as usize];
    let mut boundary_token_map =
        vec![Vec::<u32>::new(); boundary_map.num_internal_tokens() as usize];

    for (component_token, originals) in component_map
        .vocab_tokens
        .internal_to_originals
        .iter()
        .enumerate()
    {
        let mut by_boundary = BTreeMap::<u32, Vec<u32>>::new();
        for &original in originals {
            let boundary_token = boundary_map
                .vocab_tokens
                .original_to_internal
                .get(original as usize)
                .copied()
                .unwrap_or(u32::MAX);
            by_boundary.entry(boundary_token).or_default().push(original);
        }
        for (boundary_token, grouped_originals) in by_boundary {
            let common = common_to_originals.len() as u32;
            component_token_map[component_token].push(common);
            if boundary_token != u32::MAX {
                boundary_token_map[boundary_token as usize].push(common);
            }
            for &original in &grouped_originals {
                original_to_common[original as usize] = common;
            }
            representatives.push(grouped_originals[0]);
            common_to_originals.push(grouped_originals);
        }
    }
    for (boundary_token, originals) in boundary_map
        .vocab_tokens
        .internal_to_originals
        .iter()
        .enumerate()
    {
        let right_only = originals
            .iter()
            .copied()
            .filter(|&original| {
                component_map
                    .vocab_tokens
                    .original_to_internal
                    .get(original as usize)
                    .copied()
                    .unwrap_or(u32::MAX)
                    == u32::MAX
            })
            .collect::<Vec<_>>();
        if right_only.is_empty() {
            continue;
        }
        let common = common_to_originals.len() as u32;
        boundary_token_map[boundary_token].push(common);
        for &original in &right_only {
            original_to_common[original as usize] = common;
        }
        representatives.push(right_only[0]);
        common_to_originals.push(right_only);
    }
    Some(BoundaryRefinementPlan {
        common_map: InternalIdMap {
            tokenizer_states: component_map.tokenizer_states,
            vocab_tokens: ManyToOneIdMap {
                original_to_internal: original_to_common,
                internal_to_originals: common_to_originals,
                representative_original_ids: representatives,
            },
            deferred_vocab_singleton_original_ids: None,
        },
        component_token_map: Some(component_token_map),
        boundary_tsid_map,
        boundary_token_map,
    })
}

/// Recover the private boundary TSID for every global raw tokenizer state from
/// the refinement map used by the flattened linker. `boundary_tsid_map[b]`
/// lists the common TSID classes represented by private boundary class `b`.
/// Segmented runtime needs the inverse raw-state lookup that the compact
/// boundary `InternalIdMap` deliberately omits.
pub(super) fn segmented_boundary_state_to_private_tsid(
    common_state_map: &ManyToOneIdMap,
    boundary_tsid_map: &[Vec<u32>],
) -> Result<Vec<u32>, String> {
    let common_count = common_state_map.internal_to_originals.len();
    let mut common_to_boundary = vec![u32::MAX; common_count];
    for (boundary_tsid, common_tsids) in boundary_tsid_map.iter().enumerate() {
        for &common_tsid in common_tsids {
            let Some(slot) = common_to_boundary.get_mut(common_tsid as usize) else {
                return Err(format!(
                    "boundary TSID {boundary_tsid} maps outside common tokenizer coordinate: {common_tsid}"
                ));
            };
            if *slot != u32::MAX && *slot != boundary_tsid as u32 {
                return Err(format!(
                    "common tokenizer TSID {common_tsid} belongs to multiple boundary TSIDs: {} and {boundary_tsid}",
                    *slot,
                ));
            }
            *slot = boundary_tsid as u32;
        }
    }
    let mut raw_to_boundary = Vec::with_capacity(common_state_map.original_to_internal.len());
    for (raw_state, &common_tsid) in common_state_map.original_to_internal.iter().enumerate() {
        let boundary_tsid = common_to_boundary
            .get(common_tsid as usize)
            .copied()
            .unwrap_or(u32::MAX);
        if boundary_tsid == u32::MAX {
            return Err(format!(
                "global tokenizer state {raw_state} / common TSID {common_tsid} has no private boundary TSID"
            ));
        }
        raw_to_boundary.push(boundary_tsid);
    }
    Ok(raw_to_boundary)
}

pub(super) fn full_nwa_topological_order(nwa: &NWA) -> Option<Vec<u32>> {
    let state_count = nwa.num_states() as usize;
    let mut indegree = vec![0usize; state_count];
    for state in nwa.states() {
        for (target, _) in &state.epsilons {
            *indegree.get_mut(*target as usize)? += 1;
        }
        for targets in state.transitions.values() {
            for (target, _) in targets {
                *indegree.get_mut(*target as usize)? += 1;
            }
        }
    }
    let mut queue = VecDeque::new();
    for (state, &degree) in indegree.iter().enumerate() {
        if degree == 0 {
            queue.push_back(state as u32);
        }
    }
    let mut order = Vec::with_capacity(state_count);
    while let Some(source) = queue.pop_front() {
        order.push(source);
        let state = &nwa.states()[source as usize];
        for target in state
            .epsilons
            .iter()
            .map(|(target, _)| *target)
            .chain(
                state
                    .transitions
                    .values()
                    .flat_map(|targets| targets.iter().map(|(target, _)| *target)),
            )
        {
            let degree = &mut indegree[target as usize];
            *degree -= 1;
            if *degree == 0 {
                queue.push_back(target);
            }
        }
    }
    (order.len() == state_count).then_some(order)
}


pub(super) fn reverse_hashcons_positive_acyclic_nwa_fast(nwa: NWA) -> NWA {
    reverse_hashcons_acyclic_nwa_fast_with_reverse_topo(nwa, None, false)
}

pub(super) fn reverse_hashcons_positive_acyclic_nwa_fast_with_reverse_topo(
    nwa: NWA,
    precomputed_reverse_topo: Option<Vec<u32>>,
) -> NWA {
    reverse_hashcons_acyclic_nwa_fast_with_reverse_topo(nwa, precomputed_reverse_topo, false)
}

pub(super) fn reverse_hashcons_signed_acyclic_nwa_fast(nwa: NWA) -> NWA {
    reverse_hashcons_acyclic_nwa_fast_with_reverse_topo(nwa, None, true)
}

pub(super) fn reverse_hashcons_acyclic_nwa_fast_with_reverse_topo(
    nwa: NWA,
    precomputed_reverse_topo: Option<Vec<u32>>,
    allow_signed_labels: bool,
) -> NWA {
    use rayon::prelude::*;
    use rustc_hash::FxHasher;
    use smallvec::SmallVec;
    use std::hash::{Hash, Hasher};

    fn fingerprint_state(source: &NWAState, old_to_new: &[u32]) -> u64 {
        let mut hasher = FxHasher::default();
        source
            .final_weight
            .as_ref()
            .map(Weight::ptr_key)
            .hash(&mut hasher);
        source.transitions.len().hash(&mut hasher);
        for (&label, targets) in &source.transitions {
            label.hash(&mut hasher);
            targets.len().hash(&mut hasher);
            for (target, weight) in targets {
                old_to_new[*target as usize].hash(&mut hasher);
                weight.ptr_key().hash(&mut hasher);
            }
        }
        source.epsilons.len().hash(&mut hasher);
        for (target, weight) in &source.epsilons {
            old_to_new[*target as usize].hash(&mut hasher);
            weight.ptr_key().hash(&mut hasher);
        }
        hasher.finish()
    }

    // Fingerprints are only a candidate index. Exact structural comparison is
    // the equivalence proof, so hash collisions cannot merge distinct states.
    fn equivalent_old_states(
        source: &NWAState,
        representative: &NWAState,
        old_to_new: &[u32],
    ) -> bool {
        if source.final_weight.as_ref().map(Weight::ptr_key)
            != representative.final_weight.as_ref().map(Weight::ptr_key)
            || source.transitions.len() != representative.transitions.len()
            || source.epsilons.len() != representative.epsilons.len()
        {
            return false;
        }
        for ((&source_label, source_targets), (&rep_label, rep_targets)) in
            source.transitions.iter().zip(representative.transitions.iter())
        {
            if source_label != rep_label || source_targets.len() != rep_targets.len() {
                return false;
            }
            for ((source_target, source_weight), (rep_target, rep_weight)) in
                source_targets.iter().zip(rep_targets.iter())
            {
                if old_to_new[*source_target as usize] != old_to_new[*rep_target as usize]
                    || source_weight.ptr_key() != rep_weight.ptr_key()
                {
                    return false;
                }
            }
        }
        source
            .epsilons
            .iter()
            .zip(representative.epsilons.iter())
            .all(
                |((source_target, source_weight), (rep_target, rep_weight))| {
                    old_to_new[*source_target as usize] == old_to_new[*rep_target as usize]
                        && source_weight.ptr_key() == rep_weight.ptr_key()
                },
            )
    }

    let profile = compose_profile_enabled();
    let total_started = profile.then(Instant::now);
    let topo_started = profile.then(Instant::now);
    let old_state_count = nwa.num_states() as usize;
    let reverse_order = if let Some(order) = precomputed_reverse_topo
        && order.len() == old_state_count
        && order.iter().all(|&state| (state as usize) < old_state_count)
    {
        order
    } else {
        let Some(order) = full_nwa_topological_order(&nwa) else {
            return nwa;
        };
        order.into_iter().rev().collect()
    };
    let topo_ms = topo_started.map(|started| started.elapsed().as_secs_f64() * 1000.0);
    let (old_states, old_starts) = nwa.into_parts();
    let mut old_to_new = vec![u32::MAX; old_state_count];
    let mut representatives = Vec::<u32>::new();
    let mut buckets = FxHashMap::<u64, SmallVec<[u32; 2]>>::default();
    let classify_started = profile.then(Instant::now);

    for old_id in reverse_order {
        let source = &old_states[old_id as usize];
        debug_assert!(allow_signed_labels
            || source
                .transitions
                .keys()
                .all(|&label| !is_negative_label(label)));
        let fingerprint = fingerprint_state(source, &old_to_new);
        let existing = buckets.get(&fingerprint).and_then(|candidates| {
            candidates.iter().copied().find(|&candidate| {
                equivalent_old_states(
                    source,
                    &old_states[representatives[candidate as usize] as usize],
                    &old_to_new,
                )
            })
        });
        let new_id = if let Some(existing) = existing {
            existing
        } else {
            let created = representatives.len() as u32;
            representatives.push(old_id);
            buckets.entry(fingerprint).or_default().push(created);
            created
        };
        old_to_new[old_id as usize] = new_id;
    }
    let classify_ms = classify_started.map(|started| started.elapsed().as_secs_f64() * 1000.0);

    let renumber_by_old_id = std::env::var_os(
        "GLRMASK_EXPERIMENT_HASHCONS_RENUMBER_SKIP_SORT",
    )
    .is_some();
    if renumber_by_old_id {
        let mut class_by_old_id = representatives
            .iter()
            .copied()
            .enumerate()
            .map(|(class_id, old_id)| (old_id, class_id as u32))
            .collect::<Vec<_>>();
        class_by_old_id.sort_unstable_by_key(|(old_id, _)| *old_id);
        let mut class_to_sorted = vec![u32::MAX; representatives.len()];
        for (sorted_id, &(_, class_id)) in class_by_old_id.iter().enumerate() {
            class_to_sorted[class_id as usize] = sorted_id as u32;
        }
        for mapped in &mut old_to_new {
            *mapped = class_to_sorted[*mapped as usize];
        }
        representatives = class_by_old_id
            .into_iter()
            .map(|(old_id, _)| old_id)
            .collect();
    }

    // Move the representative states into the quotient and drop all duplicate
    // states on the Rayon workers. This avoids paying the discarded graph's
    // large serial destructor tail on the composition critical path.
    let move_started = profile.then(Instant::now);
    let mut representative_new_id = vec![u32::MAX; old_state_count];
    for (new_id, &old_id) in representatives.iter().enumerate() {
        representative_new_id[old_id as usize] = new_id as u32;
    }
    let mut moved = old_states
        .into_par_iter()
        .enumerate()
        .filter_map(|(old_id, mut state)| {
            let new_id = representative_new_id[old_id];
            if new_id == u32::MAX {
                return None;
            }
            for targets in state.transitions.values_mut() {
                for (target, _) in targets {
                    *target = old_to_new[*target as usize];
                }
            }
            for (target, _) in &mut state.epsilons {
                *target = old_to_new[*target as usize];
            }
            Some((new_id, state))
        })
        .collect::<Vec<_>>();
    if !renumber_by_old_id
        || moved.windows(2).any(|pair| pair[0].0 >= pair[1].0)
    {
        moved.par_sort_unstable_by_key(|(new_id, _)| *new_id);
    }
    let states = moved.into_iter().map(|(_, state)| state).collect::<Vec<_>>();
    let move_ms = move_started.map(|started| started.elapsed().as_secs_f64() * 1000.0);

    let mut starts = old_starts
        .into_iter()
        .map(|state| old_to_new[state as usize])
        .collect::<Vec<_>>();
    starts.sort_unstable();
    starts.dedup();
    if let (Some(topo_ms), Some(classify_ms), Some(move_ms), Some(total_started)) =
        (topo_ms, classify_ms, move_ms, total_started)
    {
        eprintln!(
            "[glrmask/profile][fast_reverse_hashcons_nwa] signed={} input_states={} output_states={} topo_ms={topo_ms:.3} classify_ms={classify_ms:.3} move_drop_ms={move_ms:.3} total_ms={:.3}",
            allow_signed_labels,
            old_state_count,
            states.len(),
            total_started.elapsed().as_secs_f64() * 1000.0,
        );
    }
    NWA::from_parts(states, starts)
}

pub(super) fn reverse_hashcons_positive_acyclic_nwa(nwa: NWA) -> NWA {
    #[derive(Hash, PartialEq, Eq)]
    struct Key {
        final_weight: Option<usize>,
        transitions: Vec<(i32, Vec<(u32, usize)>)>,
        epsilons: Vec<(u32, usize)>,
    }

    let Some(order) = full_nwa_topological_order(&nwa) else {
        return nwa;
    };
    let old_state_count = nwa.num_states() as usize;
    let (old_states, old_starts) = nwa.into_parts();
    let mut old_to_new = vec![u32::MAX; old_state_count];
    let mut states = Vec::new();
    let mut ids = FxHashMap::<Key, u32>::default();

    for old_id in order.into_iter().rev() {
        let source = &old_states[old_id as usize];
        debug_assert!(source.transitions.keys().all(|&label| !is_negative_label(label)));
        let transitions = source
            .transitions
            .iter()
            .map(|(&label, targets)| {
                (
                    label,
                    targets
                        .iter()
                        .map(|(target, weight)| {
                            let mapped = old_to_new[*target as usize];
                            debug_assert_ne!(mapped, u32::MAX);
                            (mapped, weight.ptr_key())
                        })
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();
        let epsilons = source
            .epsilons
            .iter()
            .map(|(target, weight)| {
                let mapped = old_to_new[*target as usize];
                debug_assert_ne!(mapped, u32::MAX);
                (mapped, weight.ptr_key())
            })
            .collect::<Vec<_>>();
        let key = Key {
            final_weight: source.final_weight.as_ref().map(Weight::ptr_key),
            transitions,
            epsilons,
        };
        let new_id = if let Some(&existing) = ids.get(&key) {
            existing
        } else {
            let created = states.len() as u32;
            let mut state = source.clone();
            for targets in state.transitions.values_mut() {
                for (target, _) in targets {
                    *target = old_to_new[*target as usize];
                }
            }
            for (target, _) in &mut state.epsilons {
                *target = old_to_new[*target as usize];
            }
            states.push(state);
            ids.insert(key, created);
            created
        };
        old_to_new[old_id as usize] = new_id;
    }

    let mut starts = old_starts
        .into_iter()
        .map(|state| old_to_new[state as usize])
        .collect::<Vec<_>>();
    starts.sort_unstable();
    starts.dedup();
    NWA::from_parts(states, starts)
}

pub(super) fn compose_local_id_map(
    local_to_base: &[Vec<u32>],
    base_to_common: &[Vec<u32>],
) -> Vec<Vec<u32>> {
    local_to_base
        .iter()
        .map(|base_ids| {
            let mut common = base_ids
                .iter()
                .filter_map(|&base| base_to_common.get(base as usize))
                .flat_map(|ids| ids.iter().copied())
                .collect::<Vec<_>>();
            common.sort_unstable();
            common.dedup();
            common
        })
        .collect()
}

pub(super) fn remap_unmapped_component_artifacts(
    artifacts: Vec<UnmappedComponentParserArtifact>,
    component_maps: Vec<DirectComponentCoordinateMaps>,
    base_to_common_tokens: Option<&[Vec<u32>]>,
    common_tsid_count: usize,
) -> Result<(Vec<NWA>, PossibleMatches, f64), String> {
    if artifacts.len() != component_maps.len() {
        return Err("component artifact/map count mismatch".into());
    }
    let started_at = Instant::now();
    let remap = |(artifact, maps): (
        UnmappedComponentParserArtifact,
        DirectComponentCoordinateMaps,
    )| {
            let token_map = base_to_common_tokens.map_or(
                maps.local_to_global_tokens.clone(),
                |base_to_common| {
                    compose_local_id_map(&maps.local_to_global_tokens, base_to_common)
                },
            );
            let mut pair = (artifact.automaton, artifact.possible_matches);
            if compose_profile_enabled() {
                eprintln!(
                    "[glrmask/profile][constraint_component_weight_ref_shape] parser_refs={} possible_match_refs={}",
                    pair.0.weight_refs().len(),
                    pair.1.weight_refs().len(),
                );
            }
            let mut weights = pair.weight_refs_mut();
            remap_weights_with_maps(
                &mut weights,
                &maps.local_to_global_tsids,
                &token_map,
                common_tsid_count,
            );
            Ok::<_, String>(pair)
    };
    let remapped = if macro_parallelism_disabled() {
        let mut timings = Vec::with_capacity(artifacts.len());
        let result = artifacts
            .into_iter()
            .zip(component_maps)
            .map(|item| {
                let started = Instant::now();
                let result = remap(item);
                timings.push(started.elapsed().as_secs_f64() * 1000.0);
                result
            })
            .collect::<Result<Vec<_>, String>>()?;
        report_macro_item_timings("compose_component_artifact_remap", &timings);
        result
    } else {
        artifacts
            .into_par_iter()
            .zip(component_maps.into_par_iter())
            .map(remap)
            .collect::<Result<Vec<_>, String>>()?
    };
    let mut automata = Vec::with_capacity(remapped.len());
    let mut possible_matches = PossibleMatches::new();
    for (automaton, component_possible_matches) in remapped {
        automata.push(automaton);
        for (terminal, weight) in component_possible_matches {
            possible_matches
                .entry(terminal)
                .and_modify(|existing| *existing = existing.union(&weight))
                .or_insert(weight);
        }
    }
    Ok((
        automata,
        possible_matches,
        started_at.elapsed().as_secs_f64() * 1000.0,
    ))
}

pub(super) fn component_possible_matches(
    component: &ParserDwaComponent<'_>,
    terminal_offset: u32,
) -> Result<PossibleMatches, String> {
    if !component.constraint.possible_matches_complete {
        return Err(
            "cannot compose a constraint with incomplete possible_matches; the dynamic possible-matches fallback is forbidden for constraint composition"
                .into(),
        );
    }
    Ok(component
        .constraint
        .runtime_possible_match_terminals()
        .filter_map(|terminal| {
            component
                .constraint
                .runtime_possible_match_weight(terminal)
                .map(|weight| (terminal_offset + terminal, weight.to_weight()))
        })
        .collect())
}

pub(super) fn compose_component_parser_dwas_and_possible_matches(
    components: &[ParserDwaComponent<'_>],
    terminal_offsets: &[u32],
    default_domains: &[Option<ParserDefaultDomain>],
    merged_tokenizer_state_count: usize,
    original_token_ids: &[u32],
    strip_scoped_ignore_identity: bool,
) -> Result<(MappedArtifact<(DWA, PossibleMatches)>, BTreeMap<i32, Weight>), String> {
    if components.is_empty() {
        return Err("cannot compose zero parser DWAs".into());
    }
    if terminal_offsets.len() != components.len() || default_domains.len() != components.len() {
        return Err(format!(
            "terminal-offset/default-domain count ({}/{}) does not match component count {}",
            terminal_offsets.len(),
            default_domains.len(),
            components.len(),
        ));
    }
    let total_started_at = Instant::now();
    let coordinate_started_at = Instant::now();
    let (id_map, component_maps) = build_direct_component_coordinate_maps(
        components,
        merged_tokenizer_state_count,
        original_token_ids,
    )?;
    let coordinate_ms = coordinate_started_at.elapsed().as_secs_f64() * 1000.0;
    let global_tsid_count = id_map.num_tsids() as usize;
    struct PreparedComponentArtifact {
        artifact: (NWA, PossibleMatches),
        parser_nwa_ms: f64,
        possible_matches_ms: f64,
        remap_ms: f64,
        component_index: usize,
        local_tsids: usize,
        tsid_fanout: usize,
        local_tokens: usize,
        token_fanout: usize,
    }
    let jobs = components
        .iter()
        .copied()
        .zip(terminal_offsets.iter().copied())
        .zip(default_domains.iter())
        .zip(component_maps)
        .enumerate()
        .collect::<Vec<_>>();
    let prepare = |(component_index, (((component, terminal_offset), default_domain), coordinate_maps)): (
        usize,
        (((ParserDwaComponent<'_>, u32), &Option<ParserDefaultDomain>), DirectComponentCoordinateMaps),
    )| {
            let started_at = Instant::now();
            let mut parser_nwa = component_parser_nwa(&component, default_domain.as_ref())?;
            let parser_nwa_ms = started_at.elapsed().as_secs_f64() * 1000.0;
            let started_at = Instant::now();
            let possible_matches = component_possible_matches(&component, terminal_offset)?;
            let possible_matches_ms = started_at.elapsed().as_secs_f64() * 1000.0;
            if strip_scoped_ignore_identity {
                let ignore_weight = component
                    .constraint
                    .ignore_terminal
                    .and_then(|ignore| possible_matches.get(&(terminal_offset + ignore)));
                // The standalone parser's globally-erased trivia identity must
                // not leak into other scopes. The composed boundary parser
                // reintroduces any state-dependent visible terminal behavior
                // directly from the composed LR table.
                strip_unscoped_ignore_identity(&mut parser_nwa, ignore_weight);
            }
            let mut artifact = (parser_nwa, possible_matches);
            let started_at = Instant::now();
            let mut weights = artifact.weight_refs_mut();
            remap_weights_with_maps(
                &mut weights,
                &coordinate_maps.local_to_global_tsids,
                &coordinate_maps.local_to_global_tokens,
                global_tsid_count,
            );
            drop(weights);
            let remap_ms = started_at.elapsed().as_secs_f64() * 1000.0;
            Ok::<_, String>(PreparedComponentArtifact {
                artifact,
                parser_nwa_ms,
                possible_matches_ms,
                remap_ms,
                component_index,
                local_tsids: coordinate_maps.local_to_global_tsids.len(),
                tsid_fanout: coordinate_maps
                    .local_to_global_tsids
                    .iter()
                    .map(Vec::len)
                    .sum(),
                local_tokens: coordinate_maps.local_to_global_tokens.len(),
                token_fanout: coordinate_maps
                    .local_to_global_tokens
                    .iter()
                    .map(Vec::len)
                    .sum(),
            })
    };
    let prepared = if macro_parallelism_disabled() {
        let mut timings = Vec::with_capacity(jobs.len());
        let result = jobs
            .into_iter()
            .map(|item| {
                let started = Instant::now();
                let result = prepare(item);
                timings.push(started.elapsed().as_secs_f64() * 1000.0);
                result
            })
            .collect::<Result<Vec<_>, String>>()?;
        report_macro_item_timings("compose_component_parser_prepare", &timings);
        result
    } else {
        jobs.into_par_iter()
            .map(prepare)
            .collect::<Result<Vec<_>, String>>()?
    };
    let parser_nwa_ms = prepared
        .iter()
        .map(|prepared| prepared.parser_nwa_ms)
        .fold(0.0, f64::max);
    let possible_matches_ms = prepared
        .iter()
        .map(|prepared| prepared.possible_matches_ms)
        .fold(0.0, f64::max);
    let remap_ms = prepared
        .iter()
        .map(|prepared| prepared.remap_ms)
        .fold(0.0, f64::max);
    if compose_profile_enabled() {
        for prepared in &prepared {
            eprintln!(
                "[glrmask/profile][constraint_component_remap] component={} local_tsids={} global_tsids={} tsid_fanout={} local_tokens={} global_tokens={} token_fanout={} parser_nwa_ms={:.3} remap_ms={:.3}",
                prepared.component_index,
                prepared.local_tsids,
                global_tsid_count,
                prepared.tsid_fanout,
                prepared.local_tokens,
                id_map.num_internal_tokens(),
                prepared.token_fanout,
                prepared.parser_nwa_ms,
                prepared.remap_ms,
            );
        }
    }
    let artifacts = prepared
        .into_iter()
        .map(|prepared| prepared.artifact)
        .collect::<Vec<_>>();

    let automata = artifacts
        .iter()
        .map(|(automaton, _)| automaton)
        .collect::<Vec<_>>();
    let union_nwa_states = automata
        .iter()
        .map(|automaton| automaton.num_states())
        .sum::<u32>();
    let mut possible_matches = PossibleMatches::new();
    for (_, component_possible_matches) in &artifacts {
        for (&terminal, weight) in component_possible_matches {
            possible_matches
                .entry(terminal)
                .and_modify(|existing| *existing = existing.union(weight))
                .or_insert_with(|| weight.clone());
        }
    }

    let build_generic = || -> Result<(DWA, f64, f64), String> {
        let append_started_at = Instant::now();
        let mut union = NWA::new(id_map.num_tsids(), id_map.max_internal_token_id());
        let mut starts = Vec::new();
        for automaton in &automata {
            let body = union.append_with_body(automaton);
            starts.extend(body.start_states);
        }
        union.set_start_states(starts);
        let append_ms = append_started_at.elapsed().as_secs_f64() * 1000.0;
        let determinize_started_at = Instant::now();
        let dwa = determinize(&union).map_err(|error| error.to_string())?;
        let determinize_ms = determinize_started_at.elapsed().as_secs_f64() * 1000.0;
        Ok((dwa, append_ms, determinize_ms))
    };

    let direct_started_at = Instant::now();
    let direct = if std::env::var_os("GLRMASK_COMPOSE_GENERIC_COMPONENT_UNION").is_some() {
        None
    } else {
        determinize_epsilon_free_component_union(
            automata.iter().map(|automaton| (*automaton).clone()).collect(),
            None,
        )
    };
    let direct_ms = direct_started_at.elapsed().as_secs_f64() * 1000.0;
    let (dwa, union_path, synthetic_states, append_ms, determinize_ms) =
        if let Some((direct_dwa, synthetic_states)) = direct {
            if std::env::var_os("GLRMASK_VALIDATE_COMPOSE_COMPONENT_DIRECT_UNION").is_some() {
                let (reference, _, _) = build_generic()?;
                let difference = find_difference(&direct_dwa, &reference)
                    .map_err(|error| error.to_string())?;
                assert_eq!(
                    difference, None,
                    "direct component parser-DWA union differs from generic determinization",
                );
                eprintln!(
                    "[glrmask/validate][compose_component_direct_union] raw_states={} synthetic_states={} exact=true",
                    union_nwa_states,
                    synthetic_states,
                );
            }
            (direct_dwa, "direct", synthetic_states, 0.0, 0.0)
        } else {
            let (generic, append_ms, determinize_ms) = build_generic()?;
            (generic, "generic", 0, append_ms, determinize_ms)
        };
    if compose_profile_enabled() {
        let start_count = automata
            .iter()
            .map(|automaton| automaton.start_states().len())
            .sum::<usize>();
        let epsilon_edges = automata
            .iter()
            .flat_map(|automaton| automaton.states())
            .map(|state| state.epsilons.len())
            .sum::<usize>();
        let nondeterministic_rows = automata
            .iter()
            .flat_map(|automaton| automaton.states())
            .flat_map(|state| state.transitions.values())
            .filter(|targets| targets.len() > 1)
            .count();
        eprintln!(
            "[glrmask/profile][constraint_component_reuse] components={} global_tsids={} global_tokens={} coordinate_ms={coordinate_ms:.3} parser_nwa_ms={parser_nwa_ms:.3} possible_matches_ms={possible_matches_ms:.3} remap_ms={remap_ms:.3} union_path={} direct_ms={direct_ms:.3} append_ms={append_ms:.3} determinize_ms={determinize_ms:.3} starts={} epsilon_edges={} nondeterministic_rows={} union_nwa_states={} synthetic_states={} result_states={} total_ms={:.3}",
            components.len(),
            id_map.num_tsids(),
            id_map.num_internal_tokens(),
            union_path,
            start_count,
            epsilon_edges,
            nondeterministic_rows,
            union_nwa_states,
            synthetic_states,
            dwa.num_states(),
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Ok((MappedArtifact::new((dwa, possible_matches), id_map), BTreeMap::new()))
}

pub(super) fn explicit_parser_nwa(
    dwa: &DWA,
    num_parser_states: u32,
    extra_positive_labels: &[i32],
) -> NWA {
    let mut nwa = NWA::new(0, 0);
    for _ in dwa.states() {
        nwa.add_state();
    }
    nwa.set_start_states(vec![dwa.start_state()]);
    for (source, state) in dwa.states().iter().enumerate() {
        if let Some(final_weight) = &state.final_weight {
            nwa.set_final_weight(source as u32, final_weight.clone());
        }
        let explicit_positive = state
            .transitions
            .keys()
            .filter_map(|&label| {
                (label >= 0 && label != DEFAULT_LABEL).then_some(label as u32)
            })
            .collect::<BTreeSet<_>>();
        for (&label, (target, weight)) in &state.transitions {
            if label != DEFAULT_LABEL {
                nwa.add_transition(source as u32, label, *target, weight.clone());
            }
        }
        if let Some((target, weight)) = state.transitions.get(&DEFAULT_LABEL) {
            for parser_state in 0..num_parser_states {
                if !explicit_positive.contains(&parser_state) {
                    nwa.add_transition(
                        source as u32,
                        encode_positive_label(parser_state),
                        *target,
                        weight.clone(),
                    );
                }
            }
            for &label in extra_positive_labels {
                if label >= 0 && !state.transitions.contains_key(&label) {
                    nwa.add_transition(source as u32, label, *target, weight.clone());
                }
            }
        }
    }
    nwa
}

pub(super) fn parser_nwa_preserve_defaults(dwa: &DWA) -> NWA {
    let states = dwa
        .states()
        .iter()
        .map(|state| crate::automata::weighted_u32::nwa::NWAState {
            final_weight: state.final_weight.clone(),
            transitions: state
                .transitions
                .iter()
                .map(|(&label, (target, weight))| {
                    (label, vec![(*target, weight.clone())])
                })
                .collect(),
            epsilons: Vec::new(),
        })
        .collect::<Vec<_>>();
    NWA::from_parts(states, vec![dwa.start_state()])
}

pub(super) fn pair_boundary_into_component_refinement(
    component_artifacts: MappedArtifact<(DWA, PossibleMatches)>,
    boundary: MappedArtifact<DWA>,
) -> Option<MappedArtifact<((DWA, PossibleMatches), DWA)>> {
    let total_started_at = Instant::now();
    let ((mut component_artifact, component_map), (mut boundary_dwa, boundary_map)) =
        (component_artifacts.into_parts(), boundary.into_parts());

    // The component map covers the full raw tokenizer-state domain. Preserve
    // that TSID partition exactly and map each compact boundary TSID directly
    // to the component classes represented by its raw states.
    if component_map.tokenizer_states.original_to_internal.len()
        < boundary_map.tokenizer_states.original_to_internal.len()
    {
        return None;
    }
    let tsid_started_at = Instant::now();
    let component_tsid_count = component_map.num_tsids() as usize;
    let boundary_tsid_map = boundary_map
        .tokenizer_states
        .internal_to_originals
        .par_iter()
        .map(|originals| {
            let mut seen = vec![false; component_tsid_count];
            let mut mapped = Vec::new();
            for &original in originals {
                let component_tsid = component_map
                    .tokenizer_states
                    .original_to_internal
                    .get(original as usize)
                    .copied()
                    .unwrap_or(u32::MAX);
                if component_tsid == u32::MAX {
                    return None;
                }
                if !std::mem::replace(&mut seen[component_tsid as usize], true) {
                    mapped.push(component_tsid);
                }
            }
            mapped.sort_unstable();
            Some(mapped)
        })
        .collect::<Option<Vec<_>>>()?;
    let component_tsid_map = (0..component_tsid_count as u32)
        .map(|tsid| vec![tsid])
        .collect::<Vec<_>>();
    let tsid_ms = tsid_started_at.elapsed().as_secs_f64() * 1000.0;

    // If the component token partition already refines the boundary token
    // partition, their exact common refinement is the component partition
    // itself. Preserve that coordinate verbatim: the component parser DWA and
    // possible-matches need no token remap at all; only the boundary artifact
    // is lifted into the finer component classes.
    //
    // Treat "absent from the boundary map" as a partition cell of its own. A
    // component class containing both selected and unselected originals would
    // therefore be split by the boundary view and must use the generic meet
    // construction below.
    let direct_token_refinement_started_at = Instant::now();
    let component_refines_boundary = component_map
        .vocab_tokens
        .internal_to_originals
        .iter()
        .all(|originals| {
            let mut boundary_class = None::<u32>;
            for &original in originals {
                let class = boundary_map
                    .vocab_tokens
                    .original_to_internal
                    .get(original as usize)
                    .copied()
                    .unwrap_or(u32::MAX);
                match boundary_class {
                    None => boundary_class = Some(class),
                    Some(existing) if existing == class => {}
                    Some(_) => return false,
                }
            }
            true
        });
    let boundary_covered_by_component = boundary_map
        .vocab_tokens
        .internal_to_originals
        .iter()
        .flatten()
        .all(|&original| {
            component_map
                .vocab_tokens
                .original_to_internal
                .get(original as usize)
                .copied()
                .is_some_and(|token| token != u32::MAX)
        });
    if component_refines_boundary && boundary_covered_by_component {
        let mut boundary_token_map =
            vec![Vec::<u32>::new(); boundary_map.num_internal_tokens() as usize];
        for (boundary_token, originals) in boundary_map
            .vocab_tokens
            .internal_to_originals
            .iter()
            .enumerate()
        {
            let destinations = &mut boundary_token_map[boundary_token];
            for &original in originals {
                let component_token = component_map
                    .vocab_tokens
                    .original_to_internal
                    .get(original as usize)
                    .copied()
                    .unwrap_or(u32::MAX);
                debug_assert_ne!(component_token, u32::MAX);
                destinations.push(component_token);
            }
            destinations.sort_unstable();
            destinations.dedup();
        }
        let token_ms = direct_token_refinement_started_at.elapsed().as_secs_f64() * 1000.0;
        let boundary_started_at = Instant::now();
        let mut boundary_weights = boundary_dwa.weight_refs_mut();
        remap_weights_with_maps(
            &mut boundary_weights,
            &boundary_tsid_map,
            &boundary_token_map,
            component_map.num_tsids() as usize,
        );
        let boundary_ms = boundary_started_at.elapsed().as_secs_f64() * 1000.0;
        if compose_profile_enabled() {
            eprintln!(
                "[glrmask/profile][constraint_boundary_direct_reconcile] mode=component_token_refinement component_tsids={} boundary_tsids={} common_tsids={} component_tokens={} boundary_tokens={} common_tokens={} tsid_ms={tsid_ms:.3} token_ms={token_ms:.3} component_remap_ms=0.000 boundary_remap_ms={boundary_ms:.3} total_ms={:.3}",
                component_map.num_tsids(),
                boundary_map.num_tsids(),
                component_map.num_tsids(),
                component_map.num_internal_tokens(),
                boundary_map.num_internal_tokens(),
                component_map.num_internal_tokens(),
                total_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        return Some(MappedArtifact::new(
            (component_artifact, boundary_dwa),
            component_map,
        ));
    }

    // Preserve component-major token locality. Only component classes touched
    // by a boundary class are split; all untouched originals remain together.
    // This is the exact pair refinement ordered by (component_class,
    // boundary_class), followed by boundary-only classes.
    let token_started_at = Instant::now();
    let original_count = component_map
        .vocab_tokens
        .original_to_internal
        .len()
        .max(boundary_map.vocab_tokens.original_to_internal.len());
    let mut original_to_common = vec![u32::MAX; original_count];
    let mut common_to_originals = Vec::<Vec<u32>>::new();
    let mut representatives = Vec::<u32>::new();
    let mut component_token_map =
        vec![Vec::<u32>::new(); component_map.num_internal_tokens() as usize];
    let mut boundary_token_map =
        vec![Vec::<u32>::new(); boundary_map.num_internal_tokens() as usize];

    for (component_token, originals) in component_map
        .vocab_tokens
        .internal_to_originals
        .iter()
        .enumerate()
    {
        let mut by_boundary = BTreeMap::<u32, Vec<u32>>::new();
        for &original in originals {
            let boundary_token = boundary_map
                .vocab_tokens
                .original_to_internal
                .get(original as usize)
                .copied()
                .unwrap_or(u32::MAX);
            by_boundary.entry(boundary_token).or_default().push(original);
        }
        for (boundary_token, grouped_originals) in by_boundary {
            let common = common_to_originals.len() as u32;
            component_token_map[component_token].push(common);
            if boundary_token != u32::MAX {
                boundary_token_map[boundary_token as usize].push(common);
            }
            for &original in &grouped_originals {
                original_to_common[original as usize] = common;
            }
            representatives.push(grouped_originals[0]);
            common_to_originals.push(grouped_originals);
        }
    }
    for (boundary_token, originals) in boundary_map
        .vocab_tokens
        .internal_to_originals
        .iter()
        .enumerate()
    {
        let right_only = originals
            .iter()
            .copied()
            .filter(|&original| {
                component_map
                    .vocab_tokens
                    .original_to_internal
                    .get(original as usize)
                    .copied()
                    .unwrap_or(u32::MAX)
                    == u32::MAX
            })
            .collect::<Vec<_>>();
        if right_only.is_empty() {
            continue;
        }
        let common = common_to_originals.len() as u32;
        boundary_token_map[boundary_token].push(common);
        for &original in &right_only {
            original_to_common[original as usize] = common;
        }
        representatives.push(right_only[0]);
        common_to_originals.push(right_only);
    }
    let common_map = InternalIdMap {
        tokenizer_states: component_map.tokenizer_states.clone(),
        vocab_tokens: ManyToOneIdMap {
            original_to_internal: original_to_common,
            internal_to_originals: common_to_originals,
            representative_original_ids: representatives,
        },
        deferred_vocab_singleton_original_ids: None,
    };
    let token_ms = token_started_at.elapsed().as_secs_f64() * 1000.0;

    let component_started_at = Instant::now();
    let mut component_weights = component_artifact.weight_refs_mut();
    remap_weights_with_maps(
        &mut component_weights,
        &component_tsid_map,
        &component_token_map,
        common_map.num_tsids() as usize,
    );
    let component_ms = component_started_at.elapsed().as_secs_f64() * 1000.0;
    let boundary_started_at = Instant::now();
    let mut boundary_weights = boundary_dwa.weight_refs_mut();
    remap_weights_with_maps(
        &mut boundary_weights,
        &boundary_tsid_map,
        &boundary_token_map,
        common_map.num_tsids() as usize,
    );
    let boundary_ms = boundary_started_at.elapsed().as_secs_f64() * 1000.0;

    if compose_profile_enabled() {
        let split_component_classes = component_token_map
            .iter()
            .filter(|destinations| destinations.len() > 1)
            .count();
        eprintln!(
            "[glrmask/profile][constraint_boundary_direct_reconcile] component_tsids={} boundary_tsids={} common_tsids={} component_tokens={} boundary_tokens={} common_tokens={} split_component_classes={} tsid_ms={tsid_ms:.3} token_ms={token_ms:.3} component_remap_ms={component_ms:.3} boundary_remap_ms={boundary_ms:.3} total_ms={:.3}",
            component_map.num_tsids(),
            boundary_map.num_tsids(),
            common_map.num_tsids(),
            component_map.num_internal_tokens(),
            boundary_map.num_internal_tokens(),
            common_map.num_internal_tokens(),
            split_component_classes,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Some(MappedArtifact::new(
        (component_artifact, boundary_dwa),
        common_map,
    ))
}


pub(super) fn profile_exact_boundary_parser_delta(
    component_dwa: &DWA,
    boundary_dwa: &DWA,
    num_parser_states: u32,
) {
    if std::env::var_os("GLRMASK_PROFILE_EXACT_BOUNDARY_PARSER_DELTA").is_none() {
        return;
    }
    let total_started = Instant::now();
    let extra_positive_labels = component_dwa
        .states()
        .iter()
        .chain(boundary_dwa.states())
        .flat_map(|state| state.transitions.keys().copied())
        .filter(|&label| label >= num_parser_states as i32 && label != DEFAULT_LABEL)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let explicit_started = Instant::now();
    let component_explicit = determinize(&explicit_parser_nwa(
        component_dwa,
        num_parser_states,
        &extra_positive_labels,
    ));
    let boundary_explicit = determinize(&explicit_parser_nwa(
        boundary_dwa,
        num_parser_states,
        &extra_positive_labels,
    ));
    let explicit_ms = explicit_started.elapsed().as_secs_f64() * 1000.0;
    let (Ok(component_explicit), Ok(boundary_explicit)) = (component_explicit, boundary_explicit) else {
        eprintln!("[glrmask/profile][constraint_exact_boundary_parser_delta] failed=explicitize explicit_ms={explicit_ms:.3}");
        return;
    };
    if !component_explicit.is_acyclic() || !boundary_explicit.is_acyclic() {
        eprintln!(
            "[glrmask/profile][constraint_exact_boundary_parser_delta] failed=cyclic component_states={} boundary_states={} explicit_ms={explicit_ms:.3}",
            component_explicit.num_states(),
            boundary_explicit.num_states(),
        );
        return;
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    struct Key {
        boundary: u32,
        component: u32,
        boundary_weight: usize,
        component_weight: usize,
    }
    let difference_started = Instant::now();
    let mut ops = ScopedWeightOpCache::default();
    let all = Weight::all();
    let empty = Weight::empty();
    let mut states = vec![DWAState::default()];
    let mut payloads = vec![(
        boundary_explicit.start_state(),
        Some(component_explicit.start_state()),
        all.clone(),
        all.clone(),
    )];
    let mut ids = FxHashMap::<Key, u32>::default();
    ids.insert(
        Key {
            boundary: boundary_explicit.start_state(),
            component: component_explicit.start_state(),
            boundary_weight: all.ptr_key(),
            component_weight: all.ptr_key(),
        },
        0,
    );
    let mut queue = VecDeque::from([0u32]);
    while let Some(out_state) = queue.pop_front() {
        let (boundary_state, component_state, boundary_prefix, component_prefix) =
            payloads[out_state as usize].clone();
        let boundary_row = &boundary_explicit.states()[boundary_state as usize];
        if let Some(boundary_final) = boundary_row.final_weight.as_ref() {
            let boundary_accept = ops.intersection(&boundary_prefix, boundary_final);
            let component_accept = component_state
                .and_then(|state| component_explicit.states()[state as usize].final_weight.as_ref())
                .map(|final_weight| ops.intersection(&component_prefix, final_weight))
                .unwrap_or_else(Weight::empty);
            let residual = ops.difference(&boundary_accept, &component_accept);
            if !residual.is_empty() {
                states[out_state as usize].final_weight = Some(residual);
            }
        }
        for (&label, (boundary_target, boundary_edge)) in &boundary_row.transitions {
            let next_boundary_weight = ops.intersection(&boundary_prefix, boundary_edge);
            if next_boundary_weight.is_empty() {
                continue;
            }
            let (next_component, next_component_weight) = if let Some(component_state) = component_state {
                if let Some((target, edge)) = component_explicit.states()[component_state as usize]
                    .transitions
                    .get(&label)
                {
                    let support = ops.intersection(&component_prefix, edge);
                    if support.is_empty() {
                        (None, empty.clone())
                    } else {
                        (Some(*target), support)
                    }
                } else {
                    (None, empty.clone())
                }
            } else {
                (None, empty.clone())
            };
            let key = Key {
                boundary: *boundary_target,
                component: next_component.unwrap_or(u32::MAX),
                boundary_weight: next_boundary_weight.ptr_key(),
                component_weight: next_component_weight.ptr_key(),
            };
            let target = if let Some(&target) = ids.get(&key) {
                target
            } else {
                let target = states.len() as u32;
                ids.insert(key, target);
                states.push(DWAState::default());
                payloads.push((
                    *boundary_target,
                    next_component,
                    next_boundary_weight,
                    next_component_weight,
                ));
                queue.push_back(target);
                target
            };
            states[out_state as usize]
                .transitions
                .insert(label, (target, Weight::all()));
        }
    }
    let raw = DWA::from_parts(states, 0);
    let raw_states = raw.num_states();
    let raw_transitions = raw.num_transitions();
    let raw_finals = raw
        .states()
        .iter()
        .filter(|state| state.final_weight.as_ref().is_some_and(|weight| !weight.is_empty()))
        .count();
    let minimized = reverse_hashcons_owned(raw);
    let minimized_states = minimized.num_states();
    let minimized_transitions = minimized.num_transitions();
    let minimized_finals = minimized
        .states()
        .iter()
        .filter(|state| state.final_weight.as_ref().is_some_and(|weight| !weight.is_empty()))
        .count();
    eprintln!(
        "[glrmask/profile][constraint_exact_boundary_parser_delta] component_input_states={} component_input_transitions={} boundary_input_states={} boundary_input_transitions={} explicit_component_states={} explicit_component_transitions={} explicit_boundary_states={} explicit_boundary_transitions={} extra_labels={} explicit_ms={explicit_ms:.3} raw_states={} raw_transitions={} raw_finals={} minimized_states={} minimized_transitions={} minimized_finals={} difference_ms={:.3} total_ms={:.3}",
        component_dwa.num_states(),
        component_dwa.num_transitions(),
        boundary_dwa.num_states(),
        boundary_dwa.num_transitions(),
        component_explicit.num_states(),
        component_explicit.num_transitions(),
        boundary_explicit.num_states(),
        boundary_explicit.num_transitions(),
        extra_positive_labels.len(),
        raw_states,
        raw_transitions,
        raw_finals,
        minimized_states,
        minimized_transitions,
        minimized_finals,
        difference_started.elapsed().as_secs_f64() * 1000.0,
        total_started.elapsed().as_secs_f64() * 1000.0,
    );
}

pub(super) fn union_boundary_parser_dwa(
    component_artifacts: MappedArtifact<(DWA, PossibleMatches)>,
    boundary: MappedArtifact<DWA>,
    num_parser_states: u32,
) -> Result<MappedArtifact<(DWA, PossibleMatches)>, String> {
    let total_started_at = Instant::now();
    let pair_started_at = Instant::now();
    let paired = if std::env::var_os("GLRMASK_COMPOSE_GENERIC_BOUNDARY_RECONCILE").is_some() {
        component_artifacts.pair_forced_common(boundary)
    } else {
        pair_boundary_into_component_refinement(component_artifacts, boundary)
            .expect("component map must cover the composed boundary coordinate domain")
    };
    let pair_ms = pair_started_at.elapsed().as_secs_f64() * 1000.0;
    let (((component_dwa, possible_matches), boundary_dwa), id_map) = paired.into_parts();
    profile_exact_boundary_parser_delta(&component_dwa, &boundary_dwa, num_parser_states);
    let explicit_started_at = Instant::now();
    let component_nwa = parser_nwa_preserve_defaults(&component_dwa);
    let boundary_nwa = parser_nwa_preserve_defaults(&boundary_dwa);
    let explicit_ms = explicit_started_at.elapsed().as_secs_f64() * 1000.0;
    if compose_profile_enabled() {
        let count_defaults = |dwa: &DWA| {
            dwa.states()
                .iter()
                .filter(|state| state.transitions.contains_key(&DEFAULT_LABEL))
                .count()
        };
        eprintln!(
            "[glrmask/profile][constraint_parser_union_defaults] component_default_states={} boundary_default_states={} component_transitions={} boundary_transitions={}",
            count_defaults(&component_dwa),
            count_defaults(&boundary_dwa),
            component_dwa.num_transitions(),
            boundary_dwa.num_transitions(),
        );
    }
    let automata = [&component_nwa, &boundary_nwa];

    let build_generic = || -> Result<(DWA, f64, f64), String> {
        let append_started_at = Instant::now();
        let extra_positive_labels = component_dwa
            .states()
            .iter()
            .chain(boundary_dwa.states())
            .flat_map(|state| state.transitions.keys().copied())
            .filter(|&label| label >= num_parser_states as i32 && label != DEFAULT_LABEL)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let generic_component_nwa = explicit_parser_nwa(
            &component_dwa,
            num_parser_states,
            &extra_positive_labels,
        );
        let generic_boundary_nwa = explicit_parser_nwa(
            &boundary_dwa,
            num_parser_states,
            &extra_positive_labels,
        );
        let mut union = NWA::new(id_map.num_tsids(), id_map.max_internal_token_id());
        let component_body = union.append_with_body(&generic_component_nwa);
        let boundary_body = union.append_with_body(&generic_boundary_nwa);
        union.set_start_states(
            component_body
                .start_states
                .into_iter()
                .chain(boundary_body.start_states)
                .collect(),
        );
        let append_ms = append_started_at.elapsed().as_secs_f64() * 1000.0;
        let determinize_started_at = Instant::now();
        let parser_dwa = determinize(&union).map_err(|error| error.to_string())?;
        let determinize_ms = determinize_started_at.elapsed().as_secs_f64() * 1000.0;
        Ok((parser_dwa, append_ms, determinize_ms))
    };

    let direct_started_at = Instant::now();
    let direct = if std::env::var_os("GLRMASK_COMPOSE_GENERIC_BOUNDARY_UNION").is_some() {
        None
    } else {
        determinize_epsilon_free_component_union(
            automata.iter().map(|automaton| (*automaton).clone()).collect(),
            Some(num_parser_states),
        )
    };
    let direct_ms = direct_started_at.elapsed().as_secs_f64() * 1000.0;
    let (parser_dwa, union_path, synthetic_states, append_ms, determinize_ms) =
        if let Some((direct_dwa, synthetic_states)) = direct {
            if std::env::var_os("GLRMASK_VALIDATE_COMPOSE_BOUNDARY_DIRECT_UNION").is_some() {
                let (reference, _, _) = build_generic()?;
                let extra_positive_labels = component_dwa
                    .states()
                    .iter()
                    .chain(boundary_dwa.states())
                    .flat_map(|state| state.transitions.keys().copied())
                    .filter(|&label| label >= num_parser_states as i32 && label != DEFAULT_LABEL)
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect::<Vec<_>>();
                let direct_explicit_nwa =
                    explicit_parser_nwa(&direct_dwa, num_parser_states, &extra_positive_labels);
                let direct_explicit =
                    determinize(&direct_explicit_nwa).map_err(|error| error.to_string())?;
                let difference = find_difference(&direct_explicit, &reference)
                    .map_err(|error| error.to_string())?;
                assert_eq!(
                    difference, None,
                    "direct boundary parser-DWA union differs from generic determinization",
                );
                eprintln!(
                    "[glrmask/validate][compose_boundary_direct_union] raw_states={} synthetic_states={} exact=true",
                    component_nwa.num_states() + boundary_nwa.num_states(),
                    synthetic_states,
                );
            }
            (direct_dwa, "direct", synthetic_states, 0.0, 0.0)
        } else {
            let (generic, append_ms, determinize_ms) = build_generic()?;
            (generic, "generic", 0, append_ms, determinize_ms)
        };
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_parser_union] component_states={} boundary_states={} raw_states={} result_states={} result_transitions={} pair_ms={pair_ms:.3} explicit_ms={explicit_ms:.3} union_path={} direct_ms={direct_ms:.3} synthetic_states={} append_ms={append_ms:.3} determinize_ms={determinize_ms:.3} total_ms={:.3}",
            component_dwa.num_states(),
            boundary_dwa.num_states(),
            component_nwa.num_states() + boundary_nwa.num_states(),
            parser_dwa.num_states(),
            parser_dwa.num_transitions(),
            union_path,
            synthetic_states,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Ok(MappedArtifact::new(
        (parser_dwa, possible_matches),
        id_map,
    ))
}
