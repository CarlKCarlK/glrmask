//! Weighted determinization for acyclic NWAs.
//!
//! Cyclic inputs are rejected and must be handled by the caller.

use std::collections::{BTreeMap, VecDeque, hash_map::Entry as HashMapEntry};
use std::time::Instant;
use std::sync::Arc;

use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use super::dwa::DWA;
use super::equivalence::find_difference;
use super::nwa::NWA;
use crate::ds::weight::{SharedTokenSet, ScopedWeightOpCache, Weight, WeightIntersectionIndex};
use crate::GlrMaskError;

const MAX_INDEXED_FINAL_PATH_RANGES: usize = 8;
const MIN_INDEXED_FINAL_WEIGHT_RANGES: usize = 32;

/// Outgoing edges from one NWA state, grouped by the exact shared `Weight`.
///
/// For a path weight `p`, every edge in a group contributes `p ∩ w`, so this
/// computes that intersection once and distributes the immutable result to the
/// group's distinct labels and destinations.
#[derive(Clone)]
struct WeightGroupedEdge {
    label: i32,
    dst: u32,
    /// This source label has exactly one target and that target is epsilon-free.
    /// A one-entry determinized subset can emit it without the label staging map.
    direct_singleton: bool,
}

#[derive(Clone)]
struct WeightGroupedTransitions {
    weight: Weight,
    edges: Vec<WeightGroupedEdge>,
}

#[derive(Debug)]
enum PreparedDirectLabel {
    Raw(i32, SmallVec<[(u32, Weight); 1]>),
    Multi {
        label: i32,
        next_key: Vec<(u32, Weight)>,
        edge_weight: Weight,
        contribution_count: usize,
        point_entries: bool,
    },
}

fn prepare_direct_label_parallel(
    nwa: &NWA,
    direct_single_target_enabled: bool,
    label: i32,
    mut target_contributions: SmallVec<[(u32, Weight); 1]>,
) -> PreparedDirectLabel {
    if target_contributions.len() <= 1
        || !direct_single_target_enabled
        || !target_contributions.iter().all(|(dst, _)| {
            nwa.states()
                .get(*dst as usize)
                .is_some_and(|state| state.epsilons.is_empty())
        })
    {
        return PreparedDirectLabel::Raw(label, target_contributions);
    }

    let point_entries = target_contributions
        .iter()
        .all(|(_, weight)| weight.single_tsid_shared_entry().is_some());
    let contribution_count = target_contributions.len();
    if point_entries {
        let (next_key, edge_weight) = aggregate_direct_point_entries(target_contributions);
        return PreparedDirectLabel::Multi {
            label,
            next_key,
            edge_weight,
            contribution_count,
            point_entries: true,
        };
    }

    target_contributions.sort_unstable_by_key(|(dst, _)| *dst);
    let mut next_key = Vec::<(u32, Weight)>::new();
    let mut index = 0usize;
    while index < target_contributions.len() {
        let dst = target_contributions[index].0;
        let mut end = index + 1;
        while end < target_contributions.len() && target_contributions[end].0 == dst {
            end += 1;
        }
        let combined = if end == index + 1 {
            target_contributions[index].1.clone()
        } else {
            Weight::union_all(
                target_contributions[index..end]
                    .iter()
                    .map(|(_, weight)| weight),
            )
        };
        next_key.push((dst, combined));
        index = end;
    }
    let edge_weight = if next_key.len() == 1 {
        next_key[0].1.clone()
    } else {
        Weight::union_all(next_key.iter().map(|(_, weight)| weight))
    };
    PreparedDirectLabel::Multi {
        label,
        next_key,
        edge_weight,
        contribution_count,
        point_entries: false,
    }
}

fn build_weight_grouped_transitions(nwa: &NWA) -> Vec<Vec<WeightGroupedTransitions>> {
    nwa.states()
        .iter()
        .map(|state| {
            let mut groups = FxHashMap::<usize, usize>::default();
            let mut result = Vec::<WeightGroupedTransitions>::new();
            for (&label, targets) in &state.transitions {
                let label_has_single_target = targets.len() == 1;
                for (dst, weight) in targets {
                    let direct_singleton_edge = label_has_single_target
                        && nwa.states()[*dst as usize].epsilons.is_empty();
                    let key = weight.ptr_key();
                    let index = if let Some(&existing) = groups.get(&key) {
                        existing
                    } else {
                        let created = result.len();
                        groups.insert(key, created);
                        result.push(WeightGroupedTransitions {
                            weight: weight.clone(),
                            edges: Vec::new(),
                        });
                        created
                    };
                    result[index].edges.push(WeightGroupedEdge {
                        label,
                        dst: *dst,
                        direct_singleton: direct_singleton_edge,
                    });
                }
            }
            result
        })
        .collect()
}

fn union_state_weight(weights: &mut FxHashMap<u32, Weight>, state_id: u32, add: Weight) {
    if add.is_empty() {
        return;
    }

    match weights.entry(state_id) {
        HashMapEntry::Occupied(mut occupied) => {
            let existing = occupied.get_mut();
            *existing = existing.union(&add);
        }
        HashMapEntry::Vacant(vacant) => {
            vacant.insert(add);
        }
    }
}

fn final_weight_intersection(
    path_weight: &Weight,
    state_final: &Weight,
    final_weight_indices: &FxHashMap<usize, WeightIntersectionIndex>,
    max_indexed_path_ranges: usize,
) -> Weight {
    if path_weight.outer_range_count() <= max_indexed_path_ranges {
        if let Some(index) = final_weight_indices.get(&state_final.ptr_key()) {
            return path_weight.intersection_with_index(index);
        }
    }
    path_weight.intersection(state_final)
}

fn subset_final_weight(
    nwa: &NWA,
    subset_entries: &[(u32, Weight)],
    final_weight_indices: &FxHashMap<usize, WeightIntersectionIndex>,
    max_indexed_path_ranges: usize,
) -> Weight {
    subset_entries.iter().fold(Weight::empty(), |final_weight, (state_id, path_weight)| {
        let Some(state_final) = nwa.states()[*state_id as usize].final_weight.as_ref() else {
            return final_weight;
        };

        final_weight.union(&final_weight_intersection(
            path_weight,
            state_final,
            final_weight_indices,
            max_indexed_path_ranges,
        ))
    })
}

#[derive(Default)]
struct FinalWeightProfile {
    subsets: usize,
    subset_entries: usize,
    final_entries: usize,
    nonempty_contributions: usize,
    max_final_entries: usize,
    intersection_ms: f64,
    union_ms: f64,
}

impl FinalWeightProfile {
    fn merge(&mut self, other: Self) {
        self.subsets += other.subsets;
        self.subset_entries += other.subset_entries;
        self.final_entries += other.final_entries;
        self.nonempty_contributions += other.nonempty_contributions;
        self.max_final_entries = self.max_final_entries.max(other.max_final_entries);
        self.intersection_ms += other.intersection_ms;
        self.union_ms += other.union_ms;
    }
}

fn subset_final_weight_profiled(
    nwa: &NWA,
    subset_entries: &[(u32, Weight)],
    final_weight_indices: &FxHashMap<usize, WeightIntersectionIndex>,
    max_indexed_path_ranges: usize,
) -> (Weight, FinalWeightProfile) {
    let mut profile = FinalWeightProfile {
        subsets: 1,
        subset_entries: subset_entries.len(),
        ..FinalWeightProfile::default()
    };
    let mut final_weight = Weight::empty();

    for (state_id, path_weight) in subset_entries {
        let Some(state_final) = nwa.states()[*state_id as usize].final_weight.as_ref() else {
            continue;
        };
        profile.final_entries += 1;
        profile.max_final_entries = profile.max_final_entries.max(profile.final_entries);

        let intersection_started_at = Instant::now();
        let contribution = final_weight_intersection(
            path_weight,
            state_final,
            final_weight_indices,
            max_indexed_path_ranges,
        );
        profile.intersection_ms += intersection_started_at.elapsed().as_secs_f64() * 1000.0;
        if contribution.is_empty() {
            continue;
        }
        profile.nonempty_contributions += 1;

        let union_started_at = Instant::now();
        final_weight = final_weight.union(&contribution);
        profile.union_ms += union_started_at.elapsed().as_secs_f64() * 1000.0;
    }

    (final_weight, profile)
}

fn seed_start_subset(nwa: &NWA) -> FxHashMap<u32, Weight> {
    let mut start_subset = FxHashMap::default();
    for &state_id in nwa.start_states() {
        union_state_weight(&mut start_subset, state_id, Weight::all());
    }
    start_subset
}

/// Exact determinization for an NWA whose structural accepted language has
/// labelled depth at most two.
///
/// Rather than constructing residual subsets, evaluate the accepted weight of
/// every word of length 0, 1, or 2 directly. The resulting prefix DWA stores
/// the exact one-label word weight as the depth-1 final weight, and the union
/// of all accepted words below a first label on its start edge. Two-label word
/// weights live on the second edge into one shared all-final leaf.
///
/// The caller must establish the structural depth bound. This function keeps
/// an acyclicity check because the reverse language DP requires a DAG.
pub fn determinize_depth2(nwa: &NWA) -> Result<DWA, GlrMaskError> {
    if !nwa.is_acyclic() {
        return Err(GlrMaskError::Compilation(
            "depth-2 weighted determinization supports only acyclic NWAs".into(),
        ));
    }

    let profile = determinize_profile_enabled();
    let total_started_at = profile.then(Instant::now);
    type Word = SmallVec<[i32; 2]>;

    fn union_word_weight(language: &mut BTreeMap<Word, Weight>, word: Word, add: Weight) {
        if add.is_empty() {
            return;
        }
        match language.entry(word) {
            std::collections::btree_map::Entry::Occupied(mut occupied) => {
                *occupied.get_mut() = occupied.get().union(&add);
            }
            std::collections::btree_map::Entry::Vacant(vacant) => {
                vacant.insert(add);
            }
        }
    }

    let mut in_degree = vec![0u32; nwa.states().len()];
    for state in nwa.states() {
        for (dst, _) in &state.epsilons {
            in_degree[*dst as usize] += 1;
        }
        for targets in state.transitions.values() {
            for (dst, _) in targets {
                in_degree[*dst as usize] += 1;
            }
        }
    }
    let mut queue = VecDeque::new();
    for (state_id, degree) in in_degree.iter().enumerate() {
        if *degree == 0 {
            queue.push_back(state_id);
        }
    }
    let mut topo_order = Vec::with_capacity(nwa.states().len());
    while let Some(state_id) = queue.pop_front() {
        topo_order.push(state_id);
        let state = &nwa.states()[state_id];
        for (dst, _) in &state.epsilons {
            in_degree[*dst as usize] -= 1;
            if in_degree[*dst as usize] == 0 {
                queue.push_back(*dst as usize);
            }
        }
        for targets in state.transitions.values() {
            for (dst, _) in targets {
                in_degree[*dst as usize] -= 1;
                if in_degree[*dst as usize] == 0 {
                    queue.push_back(*dst as usize);
                }
            }
        }
    }
    debug_assert_eq!(topo_order.len(), nwa.states().len());

    let dp_started_at = profile.then(Instant::now);
    let mut languages = vec![BTreeMap::<Word, Weight>::new(); nwa.states().len()];
    let mut weight_cache = ScopedWeightOpCache::default();
    let mut dp_word_contributions = 0usize;
    let mut max_state_words = 0usize;
    for state_id in topo_order.into_iter().rev() {
        let state = &nwa.states()[state_id];
        let mut contributions = BTreeMap::<Word, SmallVec<[Weight; 4]>>::new();
        if let Some(final_weight) = &state.final_weight {
            contributions
                .entry(Word::new())
                .or_default()
                .push(final_weight.clone());
        }
        for (dst, edge_weight) in &state.epsilons {
            for (word, suffix_weight) in &languages[*dst as usize] {
                dp_word_contributions += 1;
                let contribution = weight_cache.intersection(edge_weight, suffix_weight);
                if !contribution.is_empty() {
                    contributions
                        .entry(word.clone())
                        .or_default()
                        .push(contribution);
                }
            }
        }
        for (&label, targets) in &state.transitions {
            for (dst, edge_weight) in targets {
                for (suffix, suffix_weight) in &languages[*dst as usize] {
                    if suffix.len() >= 2 {
                        let contribution = weight_cache.intersection(edge_weight, suffix_weight);
                        if !contribution.is_empty() {
                            return Err(GlrMaskError::Compilation(
                                "depth-2 determinization received an accepted path deeper than two labels".into(),
                            ));
                        }
                        continue;
                    }
                    dp_word_contributions += 1;
                    let contribution = weight_cache.intersection(edge_weight, suffix_weight);
                    if contribution.is_empty() {
                        continue;
                    }
                    let mut word = Word::with_capacity(suffix.len() + 1);
                    word.push(label);
                    word.extend_from_slice(suffix);
                    contributions.entry(word).or_default().push(contribution);
                }
            }
        }
        let language = contributions
            .into_iter()
            .filter_map(|(word, weights)| {
                let weight = Weight::union_all(weights.iter());
                (!weight.is_empty()).then_some((word, weight))
            })
            .collect::<BTreeMap<_, _>>();
        max_state_words = max_state_words.max(language.len());
        languages[state_id] = language;
    }
    let dp_ms = dp_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    let start_merge_started_at = profile.then(Instant::now);
    let mut accepted = BTreeMap::<Word, Weight>::new();
    for &start_state in nwa.start_states() {
        for (word, weight) in &languages[start_state as usize] {
            union_word_weight(&mut accepted, word.clone(), weight.clone());
        }
    }
    let start_merge_ms = start_merge_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    let accepted_word_count = accepted.len();
    let build_started_at = profile.then(Instant::now);
    let mut by_first = BTreeMap::<i32, (Weight, BTreeMap<i32, Weight>)>::new();
    let mut empty_weight = Weight::empty();
    for (word, weight) in accepted {
        match word.as_slice() {
            [] => empty_weight = empty_weight.union(&weight),
            [first] => {
                let entry = by_first
                    .entry(*first)
                    .or_insert_with(|| (Weight::empty(), BTreeMap::new()));
                entry.0 = entry.0.union(&weight);
            }
            [first, second] => {
                let entry = by_first
                    .entry(*first)
                    .or_insert_with(|| (Weight::empty(), BTreeMap::new()));
                entry
                    .1
                    .entry(*second)
                    .and_modify(|existing| *existing = existing.union(&weight))
                    .or_insert(weight);
            }
            _ => unreachable!("depth-2 language contains an overlong word"),
        }
    }

    let mut dwa = DWA::new(0, 0);
    if !empty_weight.is_empty() {
        dwa.set_final_weight(dwa.start_state(), empty_weight);
    }
    let mut shared_leaf = None::<u32>;
    for (first_label, (first_final, second_words)) in by_first {
        let prefix_weight = Weight::union_all(
            std::iter::once(&first_final).chain(second_words.values()),
        );
        let first_state = dwa.add_state();
        dwa.add_transition(dwa.start_state(), first_label, first_state, prefix_weight);
        if !first_final.is_empty() {
            dwa.set_final_weight(first_state, first_final);
        }
        if !second_words.is_empty() {
            let leaf = *shared_leaf.get_or_insert_with(|| {
                let state = dwa.add_state();
                dwa.set_final_weight(state, Weight::all());
                state
            });
            for (second_label, word_weight) in second_words {
                dwa.add_transition(first_state, second_label, leaf, word_weight);
            }
        }
    }
    let build_ms = build_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    if let Some(total_started_at) = total_started_at {
        eprintln!(
            "[glrmask/profile][determinize_depth2] nwa_states={} dp_word_contributions={} max_state_words={} accepted_words={} intersection_cache_entries={} dp_ms={:.3} start_merge_ms={:.3} build_ms={:.3} total_ms={:.3}",
            nwa.states().len(),
            dp_word_contributions,
            max_state_words,
            accepted_word_count,
            weight_cache.intersection_entry_count(),
            dp_ms,
            start_merge_ms,
            build_ms,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }

    Ok(dwa)
}

/// Aggregate direct epsilon-free point-entry contributions without repeatedly
/// rebuilding whole weight maps. Each contribution carries exactly one TSID;
/// sorting lets us reduce token sets locally for a `(destination, TSID)` pair
/// and construct the canonical per-destination and edge weights in one pass.
fn aggregate_direct_point_entries(
    target_contributions: SmallVec<[(u32, Weight); 1]>,
) -> (Vec<(u32, Weight)>, Weight) {
    let mut points: Vec<(u32, u32, SharedTokenSet)> = target_contributions
        .into_iter()
        .map(|(dst, weight)| {
            let (tsid, tokens) = weight
                .single_tsid_shared_entry()
                .expect("point-entry aggregation requires point weights");
            (dst, tsid, tokens)
        })
        .collect();

    let mut edge_entries: Vec<(u32, SharedTokenSet)> = points
        .iter()
        .map(|(_, tsid, tokens)| (*tsid, std::sync::Arc::clone(tokens)))
        .collect();
    edge_entries.sort_unstable_by_key(|(tsid, _)| *tsid);
    let edge_weight = Weight::union_sorted_point_entries(edge_entries);

    points.sort_unstable_by_key(|(dst, tsid, _)| (*dst, *tsid));
    let mut next_key = Vec::new();
    let mut group_start = 0usize;
    while group_start < points.len() {
        let dst = points[group_start].0;
        let mut group_end = group_start + 1;
        while group_end < points.len() && points[group_end].0 == dst {
            group_end += 1;
        }
        let weight = Weight::union_sorted_point_entries(
            points[group_start..group_end]
                .iter()
                .map(|(_, tsid, tokens)| (*tsid, std::sync::Arc::clone(tokens))),
        );
        debug_assert!(!weight.is_empty());
        next_key.push((dst, weight));
        group_start = group_end;
    }

    (next_key, edge_weight)
}

fn intern_determinized_subset(
    next_key: Vec<(u32, Weight)>,
    subset_map: &mut FxHashMap<Arc<[(u32, Weight)]>, u32>,
    worklist: &mut VecDeque<(u32, Arc<[(u32, Weight)]>)>,
    dwa: &mut DWA,
) -> u32 {
    if let Some(existing) = subset_map.get(next_key.as_slice()).copied() {
        return existing;
    }

    let new_id = dwa.add_state();
    let shared_key: Arc<[(u32, Weight)]> = next_key.into();
    subset_map.insert(Arc::clone(&shared_key), new_id);
    worklist.push_back((new_id, shared_key));
    new_id
}

/// Intern a one-entry subset through a pointer-identity front cache. The
/// structural `subset_map` remains authoritative on every cache miss.
fn intern_determinized_singleton(
    dst: u32,
    weight: &Weight,
    singleton_subsets: &mut FxHashMap<(u32, usize), u32>,
    subset_map: &mut FxHashMap<Arc<[(u32, Weight)]>, u32>,
    worklist: &mut VecDeque<(u32, Arc<[(u32, Weight)]>)>,
    dwa: &mut DWA,
) -> (u32, bool) {
    let singleton_key = (dst, weight.ptr_key());
    if let Some(existing) = singleton_subsets.get(&singleton_key).copied() {
        return (existing, true);
    }

    let state = intern_determinized_subset(
        vec![(dst, weight.clone())],
        subset_map,
        worklist,
        dwa,
    );
    singleton_subsets.insert(singleton_key, state);
    (state, false)
}

fn determinize_profile_enabled() -> bool {
    std::env::var("GLRMASK_PROFILE_DETERMINIZE")
        .map(|value| value == "1")
        .unwrap_or(false)
}

/// Workspace union builders must share this choice with generic determinization.
/// Raw weighted language is invariant under normalization, but a parser's
/// structural DEFAULT_LABEL fallback can distinguish the resulting rows.
#[doc(hidden)]
pub fn normalize_direct_singletons_enabled(nwa_states: usize) -> bool {
    if std::env::var_os("GLRMASK_DISABLE_DETERMINIZE_NORMALIZE_SINGLETONS").is_some() {
        return false;
    }
    if std::env::var_os("GLRMASK_DETERMINIZE_NORMALIZE_SINGLETONS").is_some() {
        return true;
    }
    let min_states = std::env::var("GLRMASK_DETERMINIZE_NORMALIZE_SINGLETON_MIN_STATES")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(16_384);
    nwa_states >= min_states
}

pub fn determinize(nwa: &NWA) -> Result<DWA, GlrMaskError> {
    let profile = determinize_profile_enabled();
    let normalize_direct_singletons = normalize_direct_singletons_enabled(nwa.states().len());
    let dwa = determinize_impl_with_options(
        nwa,
        true,
        true,
        profile,
        normalize_direct_singletons,
    )?;

    if normalize_direct_singletons
        && std::env::var_os("GLRMASK_VALIDATE_DETERMINIZE_NORMALIZE_SINGLETONS").is_some()
    {
        let reference = determinize_impl_with_options(nwa, true, true, false, false)?;
        if let Some(word) = find_difference(&dwa, &reference)? {
            return Err(GlrMaskError::Compilation(format!(
                "normalized-singleton determinization differs on labels {word:?}"
            )));
        }
        if profile {
            eprintln!("[glrmask/profile][determinize_normalized_singleton_equivalence] result=equivalent");
        }
    }

    if std::env::var_os("GLRMASK_ASSERT_GROUPED_DETERMINIZE_EQUIVALENCE").is_some() {
        let reference = determinize_impl_with_options(
            nwa,
            true,
            false,
            false,
            normalize_direct_singletons,
        )?;
        match find_difference(&dwa, &reference)? {
            Some(word) => {
                return Err(GlrMaskError::Compilation(format!(
                    "grouped-weight determinization differs from the ordinary path on labels {word:?}"
                )));
            }
            None if profile => eprintln!(
                "[glrmask/profile][determinize_grouped_weight_equivalence] result=equivalent"
            ),
            None => {}
        }
    }

    Ok(dwa)
}

fn determinize_impl(
    nwa: &NWA,
    direct_single_target_enabled: bool,
) -> Result<DWA, GlrMaskError> {
    determinize_impl_with_options(
        nwa,
        direct_single_target_enabled,
        true,
        determinize_profile_enabled(),
        normalize_direct_singletons_enabled(nwa.states().len()),
    )
}

fn determinize_impl_with_options(
    nwa: &NWA,
    direct_single_target_enabled: bool,
    group_transition_weights: bool,
    profile: bool,
    normalize_direct_singletons: bool,
) -> Result<DWA, GlrMaskError> {
    if !nwa.is_acyclic() {
        return Err(GlrMaskError::Compilation(
            "weighted determinization currently supports only acyclic NWAs".into(),
        ));
    }

    let weight_group_build_started_at = profile.then(Instant::now);
    let weight_grouped_transitions = group_transition_weights.then(|| build_weight_grouped_transitions(nwa));
    let weight_group_build_ms = weight_group_build_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    fn canonicalize(subset: &FxHashMap<u32, Weight>) -> Vec<(u32, Weight)> {
        let mut entries: Vec<_> = subset
            .iter()
            .filter_map(|(&state_id, weight)| (!weight.is_empty()).then_some((state_id, weight.clone())))
            .collect();
        entries.sort_by_key(|(state_id, _)| *state_id);
        entries
    }

    fn epsilon_closure(nwa: &NWA, seed: FxHashMap<u32, Weight>) -> FxHashMap<u32, Weight> {
        // Fast path: single-state seed with no epsilon transitions (99.6% of calls)
        if seed.len() == 1 {
            let (&state_id, _) = seed.iter().next().unwrap();
            if let Some(state) = nwa.states().get(state_id as usize) {
                if state.epsilons.is_empty() {
                    return seed;
                }
            }
        }

        let mut closure = seed;
        let mut queue: VecDeque<u32> = closure.keys().copied().collect();

        while let Some(state_id) = queue.pop_front() {
            let Some(current_weight) = closure.get(&state_id).cloned() else {
                continue;
            };
            let Some(state) = nwa.states().get(state_id as usize) else {
                continue;
            };
            for (dst, edge_weight) in &state.epsilons {
                let contribution = current_weight.intersection(edge_weight);
                if contribution.is_empty() {
                    continue;
                }
                let existing = closure.get(dst).cloned().unwrap_or_else(Weight::empty);
                if !contribution.is_subset(&existing) {
                    closure.insert(*dst, existing.union(&contribution));
                    queue.push_back(*dst);
                }
            }
        }

        closure
    }

    let mut dwa = DWA::new(0, 0);
    let start_id = dwa.start_state();

    let start_subset = epsilon_closure(nwa, seed_start_subset(nwa));

    if start_subset.is_empty() {
        return Ok(dwa);
    }

    let mut subset_map: FxHashMap<Arc<[(u32, Weight)]>, u32> = FxHashMap::default();
    // This is an identity-only front cache. A miss always falls back through
    // `subset_map`, which retains structural equality as the source of truth.
    let mut singleton_subsets: FxHashMap<(u32, usize), u32> = FxHashMap::default();
    let normalized_singleton_weight = Weight::all();
    let mut worklist: VecDeque<(u32, Arc<[(u32, Weight)]>)> = VecDeque::new();
    let start_entries: Arc<[(u32, Weight)]> = canonicalize(&start_subset).into();
    subset_map.insert(Arc::clone(&start_entries), start_id);
    worklist.push_back((start_id, start_entries));

    // Almost every label has one surviving destination. Keep that common
    // case inline instead of allocating a nested hash map and a Vec per label.
    let mut raw_targets: FxHashMap<i32, SmallVec<[(u32, Weight); 1]>> = FxHashMap::default();
    // All operands in the expansion loop remain owned by the NWA or subset_map
    // for the lifetime of this determinization, so a local exact cache avoids
    // thread-local memo overhead on the heavily reused intersection pairs.
    let mut scoped_determinize_weight_cache = ScopedWeightOpCache::default();
    let mut profile_subset_entries = 0usize;
    let mut profile_max_subset_entries = 0usize;
    let mut profile_raw_transition_visits = 0usize;
    let mut profile_weight_group_visits = 0usize;
    let mut profile_labels = 0usize;
    let mut profile_target_contributions = 0usize;
    let mut profile_single_contribution_labels = 0usize;
    let mut profile_single_contribution_no_epsilon_labels = 0usize;
    let mut profile_direct_single_target_labels = 0usize;
    let mut profile_direct_singleton_cache_hits = 0usize;
    let mut profile_direct_singleton_cache_misses = 0usize;
    let mut profile_direct_singleton_fast_path_groups = 0usize;
    let mut profile_direct_singleton_fast_path_labels = 0usize;
    let mut profile_multi_contribution_single_target_no_epsilon_labels = 0usize;
    let mut profile_multi_contribution_single_target_no_epsilon_contributions = 0usize;
    let mut profile_direct_multi_target_labels = 0usize;
    let mut profile_multi_contribution_all_no_epsilon_labels = 0usize;
    let mut profile_multi_contribution_all_no_epsilon_contributions = 0usize;
    let mut profile_expand_ms = 0.0;
    let mut profile_combine_ms = 0.0;
    let mut profile_edge_union_ms = 0.0;
    let mut profile_closure_ms = 0.0;
    let mut profile_normalize_ms = 0.0;
    let mut profile_canonicalize_ms = 0.0;
    let mut profile_subset_lookup_ms = 0.0;
    let mut profile_direct_point_aggregate_ms = 0.0;
    let mut profile_direct_point_aggregate_labels = 0usize;
    let mut profile_direct_point_aggregate_contributions = 0usize;
    let mut profile_direct_general_aggregate_ms = 0.0;
    let mut profile_direct_general_single_dst_labels = 0usize;
    let mut profile_direct_general_single_dst_contributions = 0usize;

    // A large ready frontier benefits from preparing label aggregations in
    // parallel, but forcing waves on tiny NWAs disables the very cheap direct
    // singleton path and can be a net loss.  Make the proven large-NWA path
    // automatic while retaining the historical force-on knob and an explicit
    // production kill switch.
    let parallel_direct_waves_forced =
        std::env::var_os("GLRMASK_DETERMINIZE_PARALLEL_DIRECT_WAVES").is_some();
    let parallel_direct_waves_disabled =
        std::env::var_os("GLRMASK_DISABLE_DETERMINIZE_PARALLEL_DIRECT_WAVES").is_some();
    let parallel_direct_waves = !parallel_direct_waves_disabled
        && rayon::current_num_threads() > 1
        && (parallel_direct_waves_forced || nwa.states().len() >= 128);
    let wave_max = std::env::var("GLRMASK_DETERMINIZE_WAVE_MAX")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(1_000);

    while !worklist.is_empty() {
        // Only states already discovered at wave start are expanded. Newly
        // interned states are deferred to the next wave, preserving the exact
        // fixed point while exposing aggregation work across the whole ready
        // frontier.
        let wave_len = if parallel_direct_waves {
            worklist.len().min(wave_max)
        } else {
            1
        };
        let mut raw_target_batches = Vec::new();
        for _ in 0..wave_len {
            let Some((from_state, subset_entries)) = worklist.pop_front() else {
                break;
            };
            if profile {
                profile_subset_entries += subset_entries.len();
                profile_max_subset_entries = profile_max_subset_entries.max(subset_entries.len());
            }
            // Final weight computation is deferred to after the main loop
            // and parallelized across all states.
            let expand_started_at = profile.then(Instant::now);

            if let Some(weight_grouped_transitions) = &weight_grouped_transitions {
                if direct_single_target_enabled
                    && subset_entries.len() == 1
                    && !parallel_direct_waves
                {
                    let (nwa_state_id, path_weight) = &subset_entries[0];
                    for group in &weight_grouped_transitions[*nwa_state_id as usize] {
                        if profile {
                            profile_weight_group_visits += 1;
                            profile_raw_transition_visits += group.edges.len();
                        }
                        let next_weight = scoped_determinize_weight_cache
                            .intersection(path_weight, &group.weight);
                        if next_weight.is_empty() {
                            continue;
                        }

                        let mut emitted_direct_singleton = false;
                        for edge in &group.edges {
                            if edge.direct_singleton {
                                emitted_direct_singleton = true;
                                if profile {
                                    profile_labels += 1;
                                    profile_target_contributions += 1;
                                    profile_single_contribution_labels += 1;
                                    profile_single_contribution_no_epsilon_labels += 1;
                                    profile_direct_single_target_labels += 1;
                                    profile_direct_singleton_fast_path_labels += 1;
                                }
                                let subset_lookup_started_at = profile.then(Instant::now);
                                let (to_state, cache_hit) = intern_determinized_singleton(
                                    edge.dst,
                                    if normalize_direct_singletons {
                                        &normalized_singleton_weight
                                    } else {
                                        &next_weight
                                    },
                                    &mut singleton_subsets,
                                    &mut subset_map,
                                    &mut worklist,
                                    &mut dwa,
                                );
                                if let Some(subset_lookup_started_at) = subset_lookup_started_at {
                                    profile_subset_lookup_ms +=
                                        subset_lookup_started_at.elapsed().as_secs_f64() * 1000.0;
                                }
                                if profile {
                                    if cache_hit {
                                        profile_direct_singleton_cache_hits += 1;
                                    } else {
                                        profile_direct_singleton_cache_misses += 1;
                                    }
                                }
                                dwa.add_transition(from_state, edge.label, to_state, next_weight.clone());
                            } else {
                                raw_targets
                                    .entry(edge.label)
                                    .or_default()
                                    .push((edge.dst, next_weight.clone()));
                            }
                        }
                        if emitted_direct_singleton && profile {
                            profile_direct_singleton_fast_path_groups += 1;
                        }
                    }
                } else {
                    for (nwa_state_id, path_weight) in subset_entries.iter() {
                        for group in &weight_grouped_transitions[*nwa_state_id as usize] {
                            if profile {
                                profile_weight_group_visits += 1;
                                profile_raw_transition_visits += group.edges.len();
                            }
                            let next_weight = scoped_determinize_weight_cache
                                .intersection(path_weight, &group.weight);
                            if next_weight.is_empty() {
                                continue;
                            }
                            for edge in &group.edges {
                                raw_targets
                                    .entry(edge.label)
                                    .or_default()
                                    .push((edge.dst, next_weight.clone()));
                            }
                        }
                    }
                }
            } else {
                for (nwa_state_id, path_weight) in subset_entries.iter() {
                    let state = &nwa.states()[*nwa_state_id as usize];
                    for (&label, targets) in &state.transitions {
                        for (dst, trans_weight) in targets {
                            if profile {
                                profile_raw_transition_visits += 1;
                            }
                            let next_weight = scoped_determinize_weight_cache
                                .intersection(path_weight, trans_weight);
                            if next_weight.is_empty() {
                                continue;
                            }
                            raw_targets.entry(label).or_default().push((*dst, next_weight));
                        }
                    }
                }
            }

            if let Some(expand_started_at) = expand_started_at {
                profile_expand_ms += expand_started_at.elapsed().as_secs_f64() * 1000.0;
            }

            raw_target_batches.extend(
                raw_targets
                    .drain()
                    .map(|(label, contributions)| (from_state, label, contributions)),
            );
        }

        let parallel_direct_labels =
            (parallel_direct_waves
                || std::env::var_os("GLRMASK_DETERMINIZE_PARALLEL_DIRECT_LABELS").is_some())
                && raw_target_batches.len() >= 2
                && rayon::current_num_threads() > 1;
        let prepared_labels = if parallel_direct_labels {
            raw_target_batches
                .into_par_iter()
                .map(|(from_state, label, contributions)| {
                    (
                        from_state,
                        prepare_direct_label_parallel(
                            nwa,
                            direct_single_target_enabled,
                            label,
                            contributions,
                        ),
                    )
                })
                .collect::<Vec<_>>()
        } else {
            raw_target_batches
                .into_iter()
                .map(|(from_state, label, contributions)| {
                    (from_state, PreparedDirectLabel::Raw(label, contributions))
                })
                .collect::<Vec<_>>()
        };

        for (from_state, prepared_label) in prepared_labels {
            if let PreparedDirectLabel::Multi {
                label,
                next_key,
                edge_weight,
                contribution_count,
                point_entries,
            } = prepared_label
            {
                if profile {
                    profile_labels += 1;
                    profile_target_contributions += contribution_count;
                    profile_multi_contribution_all_no_epsilon_labels += 1;
                    profile_multi_contribution_all_no_epsilon_contributions += contribution_count;
                    if next_key.len() == 1 {
                        profile_multi_contribution_single_target_no_epsilon_labels += 1;
                        profile_multi_contribution_single_target_no_epsilon_contributions +=
                            contribution_count;
                    }
                    if !point_entries {
                        profile_direct_multi_target_labels += 1;
                    }
                }
                debug_assert!(!edge_weight.is_empty());
                let subset_lookup_started_at = profile.then(Instant::now);
                let to_state = intern_determinized_subset(
                    next_key,
                    &mut subset_map,
                    &mut worklist,
                    &mut dwa,
                );
                if let Some(subset_lookup_started_at) = subset_lookup_started_at {
                    profile_subset_lookup_ms +=
                        subset_lookup_started_at.elapsed().as_secs_f64() * 1000.0;
                }
                dwa.add_transition(from_state, label, to_state, edge_weight);
                continue;
            }
            let PreparedDirectLabel::Raw(label, target_contributions) = prepared_label else {
                unreachable!();
            };
            if profile {
                profile_labels += 1;
                profile_target_contributions += target_contributions.len();
                if target_contributions.len() == 1 {
                    profile_single_contribution_labels += 1;
                    let (dst, _) = &target_contributions[0];
                    if nwa
                        .states()
                        .get(*dst as usize)
                        .is_some_and(|state| state.epsilons.is_empty())
                    {
                        profile_single_contribution_no_epsilon_labels += 1;
                    }
                } else {
                    let dst = target_contributions[0].0;
                    let all_no_epsilon = target_contributions.iter().all(|(candidate, _)| {
                        nwa
                            .states()
                            .get(*candidate as usize)
                            .is_some_and(|state| state.epsilons.is_empty())
                    });
                    if all_no_epsilon {
                        profile_multi_contribution_all_no_epsilon_labels += 1;
                        profile_multi_contribution_all_no_epsilon_contributions +=
                            target_contributions.len();
                    }
                    if target_contributions.iter().all(|(candidate, _)| *candidate == dst)
                        && all_no_epsilon
                    {
                        profile_multi_contribution_single_target_no_epsilon_labels += 1;
                        profile_multi_contribution_single_target_no_epsilon_contributions +=
                            target_contributions.len();
                    }
                }
            }
            if target_contributions.is_empty() {
                continue;
            }

            let direct_no_epsilon_targets = direct_single_target_enabled
                && target_contributions.iter().all(|(dst, _)| {
                    nwa
                        .states()
                        .get(*dst as usize)
                        .is_some_and(|state| state.epsilons.is_empty())
                });
            if direct_no_epsilon_targets {
                if target_contributions.len() == 1 {
                    if profile {
                        profile_direct_single_target_labels += 1;
                    }
                    let (dst, edge_weight) = target_contributions.into_iter().next().unwrap();
                    let subset_lookup_started_at = profile.then(Instant::now);
                    let (to_state, cache_hit) = intern_determinized_singleton(
                        dst,
                        if normalize_direct_singletons {
                            &normalized_singleton_weight
                        } else {
                            &edge_weight
                        },
                        &mut singleton_subsets,
                        &mut subset_map,
                        &mut worklist,
                        &mut dwa,
                    );
                    if profile {
                        if cache_hit {
                            profile_direct_singleton_cache_hits += 1;
                        } else {
                            profile_direct_singleton_cache_misses += 1;
                        }
                    }
                    if let Some(subset_lookup_started_at) = subset_lookup_started_at {
                        profile_subset_lookup_ms +=
                            subset_lookup_started_at.elapsed().as_secs_f64() * 1000.0;
                    }
                    dwa.add_transition(from_state, label, to_state, edge_weight);
                    continue;
                }

                let direct_point_entries = target_contributions
                    .iter()
                    .all(|(_, weight)| weight.single_tsid_shared_entry().is_some());
                if direct_point_entries {
                    let contribution_count = target_contributions.len();
                    let direct_point_started_at = profile.then(Instant::now);
                    let (mut next_key, edge_weight) = aggregate_direct_point_entries(target_contributions);
                    if normalize_direct_singletons && next_key.len() == 1 {
                        next_key[0].1 = normalized_singleton_weight.clone();
                    }
                    if let Some(started_at) = direct_point_started_at {
                        profile_direct_point_aggregate_ms +=
                            started_at.elapsed().as_secs_f64() * 1000.0;
                        profile_direct_point_aggregate_labels += 1;
                        profile_direct_point_aggregate_contributions += contribution_count;
                    }
                    debug_assert!(!edge_weight.is_empty());
                    let subset_lookup_started_at = profile.then(Instant::now);
                    let to_state = intern_determinized_subset(
                        next_key,
                        &mut subset_map,
                        &mut worklist,
                        &mut dwa,
                    );
                    if let Some(subset_lookup_started_at) = subset_lookup_started_at {
                        profile_subset_lookup_ms +=
                            subset_lookup_started_at.elapsed().as_secs_f64() * 1000.0;
                    }
                    dwa.add_transition(from_state, label, to_state, edge_weight);
                    continue;
                }

                if profile {
                    profile_direct_multi_target_labels += 1;
                }

                // Exact common case: every contribution lands in the same NWA
                // destination. The determinized subset then contains exactly one
                // entry and its path weight is also the transition edge weight.
                // Build that union once instead of repeatedly unioning by
                // destination and then unioning the destination weights again.
                let single_destination = target_contributions
                    .first()
                    .map(|(dst, _)| *dst)
                    .filter(|&dst| {
                        target_contributions
                            .iter()
                            .all(|(candidate, _)| *candidate == dst)
                    });
                if let Some(dst) = single_destination {
                    let contribution_count = target_contributions.len();
                    let aggregate_started_at = profile.then(Instant::now);
                    let edge_weight = Weight::union_all(
                        target_contributions.iter().map(|(_, weight)| weight),
                    );
                    if let Some(started_at) = aggregate_started_at {
                        profile_direct_general_aggregate_ms +=
                            started_at.elapsed().as_secs_f64() * 1000.0;
                        profile_direct_general_single_dst_labels += 1;
                        profile_direct_general_single_dst_contributions += contribution_count;
                    }
                    debug_assert!(!edge_weight.is_empty());
                    let subset_lookup_started_at = profile.then(Instant::now);
                    let to_state = intern_determinized_subset(
                        vec![(
                            dst,
                            if normalize_direct_singletons {
                                normalized_singleton_weight.clone()
                            } else {
                                edge_weight.clone()
                            },
                        )],
                        &mut subset_map,
                        &mut worklist,
                        &mut dwa,
                    );
                    if let Some(subset_lookup_started_at) = subset_lookup_started_at {
                        profile_subset_lookup_ms +=
                            subset_lookup_started_at.elapsed().as_secs_f64() * 1000.0;
                    }
                    dwa.add_transition(from_state, label, to_state, edge_weight);
                    continue;
                }

                let direct_general_started_at = profile.then(Instant::now);
                let mut sorted_targets = target_contributions;
                sorted_targets.sort_unstable_by_key(|(dst, _)| *dst);
                let mut next_key: Vec<(u32, Weight)> =
                    Vec::with_capacity(sorted_targets.len());
                for (dst, weight) in sorted_targets {
                    if let Some((last_dst, last_weight)) = next_key.last_mut() {
                        if *last_dst == dst {
                            *last_weight = last_weight.union(&weight);
                            continue;
                        }
                    }
                    next_key.push((dst, weight));
                }
                let edge_weight = Weight::union_all(next_key.iter().map(|(_, weight)| weight));
                if let Some(started_at) = direct_general_started_at {
                    profile_direct_general_aggregate_ms +=
                        started_at.elapsed().as_secs_f64() * 1000.0;
                }
                debug_assert!(!edge_weight.is_empty());

                let subset_lookup_started_at = profile.then(Instant::now);
                let to_state = intern_determinized_subset(
                    next_key,
                    &mut subset_map,
                    &mut worklist,
                    &mut dwa,
                );
                if let Some(subset_lookup_started_at) = subset_lookup_started_at {
                    profile_subset_lookup_ms +=
                        subset_lookup_started_at.elapsed().as_secs_f64() * 1000.0;
                }
                dwa.add_transition(from_state, label, to_state, edge_weight);
                continue;
            }

            let combine_started_at = profile.then(Instant::now);
            let mut target_subset: FxHashMap<u32, Weight> = FxHashMap::default();
            if target_contributions.len() == 1 {
                let (dst, weight) = target_contributions.into_iter().next().unwrap();
                target_subset.insert(dst, weight);
            } else {
                // Keep the first contribution per destination in the compact
                // target map. On the first repeat, move that destination's
                // operands into a side bucket and union them once at the end.
                // This preserves the exact subset construction while avoiding
                // quadratic growth from repeatedly rebuilding wide weights.
                let mut duplicate_weights: Option<FxHashMap<u32, SmallVec<[Weight; 2]>>> = None;
                for (dst, weight) in target_contributions {
                    if let Some(duplicates) = duplicate_weights.as_mut() {
                        if let Some(weights) = duplicates.get_mut(&dst) {
                            weights.push(weight);
                            continue;
                        }
                    }

                    match target_subset.entry(dst) {
                        HashMapEntry::Vacant(vacant) => {
                            vacant.insert(weight);
                        }
                        HashMapEntry::Occupied(occupied) => {
                            let previous = occupied.remove();
                            let mut weights = SmallVec::<[Weight; 2]>::new();
                            weights.push(previous);
                            weights.push(weight);
                            let duplicates = duplicate_weights.get_or_insert_with(FxHashMap::default);
                            let replaced = duplicates.insert(dst, weights);
                            debug_assert!(replaced.is_none());
                        }
                    }
                }
                if let Some(duplicates) = duplicate_weights {
                    for (dst, weights) in duplicates {
                        target_subset.insert(dst, Weight::union_all(weights.iter()));
                    }
                }
            }

            if let Some(combine_started_at) = combine_started_at {
                profile_combine_ms += combine_started_at.elapsed().as_secs_f64() * 1000.0;
            }
            if target_subset.is_empty() {
                continue;
            }

            let edge_union_started_at = profile.then(Instant::now);
            let edge_weight = Weight::union_all(target_subset.values());
            if let Some(edge_union_started_at) = edge_union_started_at {
                profile_edge_union_ms += edge_union_started_at.elapsed().as_secs_f64() * 1000.0;
            }
            if edge_weight.is_empty() {
                continue;
            }

            let closure_started_at = profile.then(Instant::now);
            let expanded = epsilon_closure(nwa, target_subset);
            if let Some(closure_started_at) = closure_started_at {
                profile_closure_ms += closure_started_at.elapsed().as_secs_f64() * 1000.0;
            }
            if expanded.is_empty() {
                continue;
            }

            let normalize_started_at = profile.then(Instant::now);
            let edge_complement = edge_weight.complement();
            let normalized: FxHashMap<u32, Weight> = if edge_complement.is_empty() {
                expanded
            } else {
                expanded
                    .into_iter()
                    .filter_map(|(state_id, weight)| {
                        let normalized_weight = weight.union(&edge_complement);
                        (!normalized_weight.is_empty()).then_some((state_id, normalized_weight))
                    })
                    .collect()
            };
            if let Some(normalize_started_at) = normalize_started_at {
                profile_normalize_ms += normalize_started_at.elapsed().as_secs_f64() * 1000.0;
            }
            let canonicalize_started_at = profile.then(Instant::now);
            let next_key = canonicalize(&normalized);
            if let Some(canonicalize_started_at) = canonicalize_started_at {
                profile_canonicalize_ms += canonicalize_started_at.elapsed().as_secs_f64() * 1000.0;
            }
            if next_key.is_empty() {
                continue;
            }
            let subset_lookup_started_at = profile.then(Instant::now);
            let to_state = intern_determinized_subset(
                next_key,
                &mut subset_map,
                &mut worklist,
                &mut dwa,
            );
            if let Some(subset_lookup_started_at) = subset_lookup_started_at {
                profile_subset_lookup_ms += subset_lookup_started_at.elapsed().as_secs_f64() * 1000.0;
            }

            dwa.add_transition(from_state, label, to_state, edge_weight);
        }
    }

    if profile {
        eprintln!(
            "[glrmask/profile][determinize_scoped_weight_cache] intersections={}",
            scoped_determinize_weight_cache.intersection_entry_count(),
        );
    }

    // Compute final weights in parallel after the main loop.
    // The subset_map already stores all (entries, state_id) pairs.
    // Sparse path weights recur against a few wide final weights. Index those
    // wide maps once so each sparse range can seek directly to its overlaps.
    let final_weight_indices: FxHashMap<usize, WeightIntersectionIndex> = nwa
        .states()
        .iter()
        .filter_map(|state| state.final_weight.as_ref())
        .filter(|weight| weight.outer_range_count() >= MIN_INDEXED_FINAL_WEIGHT_RANGES)
        .map(|weight| (weight.ptr_key(), weight.intersection_index()))
        .collect();
    let final_weights_started_at = profile.then(Instant::now);
    let mut final_weight_profile = FinalWeightProfile::default();
    let final_weights: Vec<(u32, Weight)> = if profile {
        let profiled: Vec<(u32, Weight, FinalWeightProfile)> = subset_map
            .par_iter()
            .map(|(entries, &state_id)| {
                let (fw, stats) = subset_final_weight_profiled(
                    nwa,
                    entries,
                    &final_weight_indices,
                    MAX_INDEXED_FINAL_PATH_RANGES,
                );
                (state_id, fw, stats)
            })
            .collect();
        profiled
            .into_iter()
            .filter_map(|(state_id, fw, stats)| {
                final_weight_profile.merge(stats);
                (!fw.is_empty()).then_some((state_id, fw))
            })
            .collect()
    } else {
        subset_map
            .par_iter()
            .filter_map(|(entries, &state_id)| {
                let fw = subset_final_weight(
                    nwa,
                    entries,
                    &final_weight_indices,
                    MAX_INDEXED_FINAL_PATH_RANGES,
                );
                (!fw.is_empty()).then_some((state_id, fw))
            })
            .collect()
    };
    for (state_id, fw) in final_weights {
        dwa.set_final_weight(state_id, fw);
    }
    let final_weights_ms = final_weights_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    if profile {
        let mut final_group_entries = 0usize;
        let mut final_group_count = 0usize;
        let mut final_subsets_with_reused_group = 0usize;
        let mut max_final_groups_per_subset = 0usize;
        let mut max_final_entries_per_group = 0usize;
        for entries in subset_map.keys() {
            let mut groups = FxHashMap::<usize, usize>::default();
            for (state_id, _) in entries.iter() {
                let Some(state_final) = nwa.states()[*state_id as usize].final_weight.as_ref() else {
                    continue;
                };
                *groups.entry(state_final.ptr_key()).or_default() += 1;
            }
            let final_entry_count: usize = groups.values().sum();
            if final_entry_count > 0 {
                final_group_entries += final_entry_count;
                final_group_count += groups.len();
                max_final_groups_per_subset = max_final_groups_per_subset.max(groups.len());
                let max_group = groups.values().copied().max().unwrap_or(0);
                max_final_entries_per_group = max_final_entries_per_group.max(max_group);
                final_subsets_with_reused_group += usize::from(max_group > 1);
            }
        }
        eprintln!(
            "[glrmask/profile][determinize_final_groups] entries={} groups={} saved_intersections={} subsets_with_reused_group={} max_groups_per_subset={} max_entries_per_group={}",
            final_group_entries,
            final_group_count,
            final_group_entries.saturating_sub(final_group_count),
            final_subsets_with_reused_group,
            max_final_groups_per_subset,
            max_final_entries_per_group,
        );

        let max_weight_dim = dwa.states().iter()
            .filter_map(|s| s.final_weight.as_ref())
            .map(Weight::num_ranges)
            .max()
            .unwrap_or(0);
        let mut final_pairs = FxHashSet::default();
        let mut final_path_weights = FxHashSet::default();
        let mut final_state_weights = FxHashSet::default();
        let mut final_path_outer_ranges = 0usize;
        let mut final_state_outer_ranges = 0usize;
        let mut max_final_path_outer_ranges = 0usize;
        let mut max_final_state_outer_ranges = 0usize;
        for entries in subset_map.keys() {
            for (state_id, path_weight) in entries.iter() {
                let Some(state_final) = nwa.states()[*state_id as usize].final_weight.as_ref() else {
                    continue;
                };
                let path_key = path_weight.ptr_key();
                let state_key = state_final.ptr_key();
                final_pairs.insert((path_key, state_key));
                final_path_weights.insert(path_key);
                final_state_weights.insert(state_key);
                let path_ranges = path_weight.outer_range_count();
                let state_ranges = state_final.outer_range_count();
                final_path_outer_ranges += path_ranges;
                final_state_outer_ranges += state_ranges;
                max_final_path_outer_ranges = max_final_path_outer_ranges.max(path_ranges);
                max_final_state_outer_ranges = max_final_state_outer_ranges.max(state_ranges);
            }
        }
        eprintln!(
            "[glrmask/profile][determinize_final_shape] pairs={} unique_pairs={} unique_path_weights={} unique_state_weights={} path_outer_ranges={} state_outer_ranges={} max_path_outer_ranges={} max_state_outer_ranges={}",
            final_weight_profile.final_entries,
            final_pairs.len(),
            final_path_weights.len(),
            final_state_weights.len(),
            final_path_outer_ranges,
            final_state_outer_ranges,
            max_final_path_outer_ranges,
            max_final_state_outer_ranges,
        );

        eprintln!(
            "[glrmask/profile][determinize] nwa_states={} dwa_states={} subset_map_entries={} max_weight_dim={} subset_entries={} max_subset_entries={} raw_transition_visits={} weight_grouped_transitions={} weight_group_build_ms={:.3} weight_group_visits={} labels={} target_contributions={} single_contribution_labels={} single_contribution_no_epsilon_labels={} direct_single_target_labels={} direct_singleton_cache_hits={} direct_singleton_cache_misses={} direct_singleton_fast_path_groups={} direct_singleton_fast_path_labels={} multi_contribution_single_target_no_epsilon_labels={} multi_contribution_single_target_no_epsilon_contributions={} direct_multi_target_labels={} multi_contribution_all_no_epsilon_labels={} multi_contribution_all_no_epsilon_contributions={} expand_ms={:.3} combine_ms={:.3} edge_union_ms={:.3} closure_ms={:.3} normalize_ms={:.3} canonicalize_ms={:.3} subset_lookup_ms={:.3} direct_point_aggregate_ms={:.3} direct_point_aggregate_labels={} direct_point_aggregate_contributions={} direct_general_aggregate_ms={:.3} direct_general_single_dst_labels={} direct_general_single_dst_contributions={} final_weights_ms={:.3} final_subsets={} final_subset_entries={} final_entries={} final_nonempty_contributions={} final_max_entries={} final_intersection_ms={:.3} final_union_ms={:.3}",
            nwa.states().len(),
            dwa.states().len(),
            subset_map.len(),
            max_weight_dim,
            profile_subset_entries,
            profile_max_subset_entries,
            profile_raw_transition_visits,
            group_transition_weights,
            weight_group_build_ms,
            profile_weight_group_visits,
            profile_labels,
            profile_target_contributions,
            profile_single_contribution_labels,
            profile_single_contribution_no_epsilon_labels,
            profile_direct_single_target_labels,
            profile_direct_singleton_cache_hits,
            profile_direct_singleton_cache_misses,
            profile_direct_singleton_fast_path_groups,
            profile_direct_singleton_fast_path_labels,
            profile_multi_contribution_single_target_no_epsilon_labels,
            profile_multi_contribution_single_target_no_epsilon_contributions,
            profile_direct_multi_target_labels,
            profile_multi_contribution_all_no_epsilon_labels,
            profile_multi_contribution_all_no_epsilon_contributions,
            profile_expand_ms,
            profile_combine_ms,
            profile_edge_union_ms,
            profile_closure_ms,
            profile_normalize_ms,
            profile_canonicalize_ms,
            profile_subset_lookup_ms,
            profile_direct_point_aggregate_ms,
            profile_direct_point_aggregate_labels,
            profile_direct_point_aggregate_contributions,
            profile_direct_general_aggregate_ms,
            profile_direct_general_single_dst_labels,
            profile_direct_general_single_dst_contributions,
            final_weights_ms,
            final_weight_profile.subsets,
            final_weight_profile.subset_entries,
            final_weight_profile.final_entries,
            final_weight_profile.nonempty_contributions,
            final_weight_profile.max_final_entries,
            final_weight_profile.intersection_ms,
            final_weight_profile.union_ms,
        );
    }

    Ok(dwa)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weighted_u32::equivalence::find_difference;
    use range_set_blaze::RangeSetBlaze;

    fn tokens(values: impl IntoIterator<Item = u32>) -> Weight {
        Weight::from_per_tsid_token_sets(std::iter::once((
            0,
            RangeSetBlaze::from_iter(values.into_iter().map(|value| value..=value)),
        )))
    }

    #[test]
    fn determinize_unions_duplicate_label_target_contributions() {
        let mut nwa = NWA::new(1, 2);
        let start = nwa.add_state();
        let accept = nwa.add_state();
        nwa.set_start_states(vec![start]);
        nwa.add_transition(start, 7, accept, tokens([0]));
        nwa.add_transition(start, 7, accept, tokens([1]));
        nwa.set_final_weight(accept, tokens([0, 1]));

        let fast = determinize_impl(&nwa, true).unwrap();
        let generic = determinize_impl(&nwa, false).unwrap();
        let grouped = determinize_impl_with_options(&nwa, true, true, false, false).unwrap();
        assert_eq!(find_difference(&fast, &generic).unwrap(), None);
        assert_eq!(find_difference(&grouped, &generic).unwrap(), None);
        assert_eq!(fast.eval_word(&[7]), tokens([0, 1]));
    }

    #[test]
    fn depth2_language_dp_matches_generic_with_epsilon_paths() {
        let mut nwa = NWA::new(1, 4);
        let start = nwa.add_state();
        let left = nwa.add_state();
        let right = nwa.add_state();
        let middle = nwa.add_state();
        let accept = nwa.add_state();
        nwa.set_start_states(vec![start]);

        nwa.add_epsilon(start, left, tokens([0, 1, 2]));
        nwa.add_epsilon(start, right, tokens([1, 2, 3]));
        nwa.add_transition(left, 7, middle, tokens([0, 1]));
        nwa.add_transition(right, 7, middle, tokens([2, 3]));
        nwa.set_final_weight(middle, tokens([0, 2]));
        nwa.add_transition(middle, 8, accept, tokens([1, 2, 3]));
        nwa.set_final_weight(accept, tokens([1, 2]));

        let depth2 = determinize_depth2(&nwa).unwrap();
        let generic = determinize(&nwa).unwrap();
        assert_eq!(find_difference(&depth2, &generic).unwrap(), None);
        assert_eq!(depth2.eval_word(&[7]), tokens([0, 2]));
        assert_eq!(depth2.eval_word(&[7, 8]), tokens([1, 2]));
    }

    #[test]
    fn depth2_language_dp_rejects_accepted_three_label_path() {
        let mut nwa = NWA::new(1, 1);
        let states: Vec<u32> = (0..4).map(|_| nwa.add_state()).collect();
        nwa.set_start_states(vec![states[0]]);
        nwa.add_transition(states[0], 1, states[1], tokens([0]));
        nwa.add_transition(states[1], 2, states[2], tokens([0]));
        nwa.add_transition(states[2], 3, states[3], tokens([0]));
        nwa.set_final_weight(states[3], tokens([0]));

        let error = determinize_depth2(&nwa).unwrap_err();
        assert!(error.to_string().contains("deeper than two labels"), "{error}");
    }

    #[test]
    fn determinize_batches_many_repeated_destinations_exactly() {
        let mut nwa = NWA::new(1, 7);
        let start = nwa.add_state();
        let left = nwa.add_state();
        let right = nwa.add_state();
        nwa.set_start_states(vec![start]);

        for token in 0..=4 {
            nwa.add_transition(start, 7, left, tokens([token]));
        }
        for token in 5..=6 {
            nwa.add_transition(start, 7, right, tokens([token]));
        }
        nwa.set_final_weight(left, tokens(0..=4));
        nwa.set_final_weight(right, tokens(5..=6));

        let dwa = determinize(&nwa).unwrap();
        assert_eq!(dwa.eval_word(&[7]), tokens(0..=6));
    }

    #[test]
    fn direct_epsilon_free_subsets_match_generic_across_acyclic_cases() {
        for case in 0u32..32 {
            let mut nwa = NWA::new(1, 8);
            let states: Vec<u32> = (0..5).map(|_| nwa.add_state()).collect();
            nwa.set_start_states(vec![states[0]]);

            for from in 0..4usize {
                for to in (from + 1)..5usize {
                    if (case + (from * 7 + to * 11) as u32) % 3 == 0 {
                        continue;
                    }
                    let label = ((case + (from * 3 + to) as u32) % 4) as i32;
                    let first = (case + (from * 5 + to) as u32) % 6;
                    let second = (first + 1 + case % 3) % 7;
                    nwa.add_transition(states[from], label, states[to], tokens([first, second]));
                    if (case + from as u32 + to as u32) % 5 == 0 {
                        let extra = (second + 2) % 8;
                        nwa.add_transition(states[from], label, states[to], tokens([extra]));
                    }
                }
            }

            for state in 0..5usize {
                if (case + state as u32) % 2 == 0 {
                    let first = (case + state as u32) % 7;
                    nwa.set_final_weight(states[state], tokens([first]));
                }
            }

            // Exercise generic fallback as well: a transition into this state
            // must retain epsilon closure, while the remaining direct targets
            // may use the optimized path.
            if case % 4 == 0 {
                nwa.add_epsilon(states[1], states[4], tokens([case % 6]));
            }

            let fast = determinize_impl(&nwa, true).unwrap();
            let generic = determinize_impl(&nwa, false).unwrap();
            let grouped = determinize_impl_with_options(&nwa, true, true, false, false).unwrap();
            assert_eq!(
                find_difference(&fast, &generic).unwrap(),
                None,
                "case {case}",
            );
            assert_eq!(
                find_difference(&grouped, &generic).unwrap(),
                None,
                "grouped case {case}",
            );
        }
    }

    #[test]
    fn point_entry_aggregation_matches_generic_determinization() {
        let mut nwa = NWA::new(1, 8);
        let start = nwa.add_state();
        let first_accept = nwa.add_state();
        let second_accept = nwa.add_state();
        nwa.set_start_states(vec![start]);
        nwa.add_transition(start, 5, first_accept, tokens([0, 2]));
        nwa.add_transition(start, 5, first_accept, tokens([1, 3]));
        nwa.add_transition(start, 5, second_accept, tokens([2, 4]));
        nwa.add_transition(start, 5, second_accept, tokens([5]));
        nwa.set_final_weight(first_accept, tokens([0, 1, 2, 3]));
        nwa.set_final_weight(second_accept, tokens([2, 4, 5]));

        let fast = determinize_impl(&nwa, true).unwrap();
        let generic = determinize_impl(&nwa, false).unwrap();
        let grouped = determinize_impl_with_options(&nwa, true, true, false, false).unwrap();
        assert_eq!(find_difference(&fast, &generic).unwrap(), None);
        assert_eq!(find_difference(&grouped, &generic).unwrap(), None);
        assert_eq!(fast.eval_word(&[5]), tokens([0, 1, 2, 3, 4, 5]));
    }

    #[test]
    fn grouped_singleton_edges_mix_direct_and_epsilon_fallback_exactly() {
        let mut nwa = NWA::new(1, 4);
        let start = nwa.add_state();
        let first_accept = nwa.add_state();
        let second_accept = nwa.add_state();
        let epsilon_source = nwa.add_state();
        let epsilon_accept = nwa.add_state();
        nwa.set_start_states(vec![start]);

        // The three transition labels share one exact Weight group. Two are
        // directly emit-able; the third must keep the epsilon-closure fallback.
        let shared = tokens([0, 1]);
        nwa.add_transition(start, 7, first_accept, shared.clone());
        nwa.add_transition(start, 8, second_accept, shared.clone());
        nwa.add_transition(start, 9, epsilon_source, shared);
        nwa.add_epsilon(epsilon_source, epsilon_accept, tokens([0, 1]));
        nwa.set_final_weight(first_accept, tokens([0, 1]));
        nwa.set_final_weight(second_accept, tokens([0, 1]));
        nwa.set_final_weight(epsilon_accept, tokens([0, 1]));

        let grouped = determinize_impl_with_options(&nwa, true, true, false, false).unwrap();
        let generic = determinize_impl_with_options(&nwa, false, false, false, false).unwrap();
        assert_eq!(find_difference(&grouped, &generic).unwrap(), None);
        assert_eq!(grouped.eval_word(&[7]), tokens([0, 1]));
        assert_eq!(grouped.eval_word(&[8]), tokens([0, 1]));
        assert_eq!(grouped.eval_word(&[9]), tokens([0, 1]));
    }

    #[test]
    fn direct_single_target_path_matches_generic_determinization() {
        let mut nwa = NWA::new(1, 4);
        let start = nwa.add_state();
        let first_accept = nwa.add_state();
        let second_accept = nwa.add_state();
        nwa.set_start_states(vec![start]);
        nwa.add_transition(start, 7, first_accept, tokens([0, 1]));
        nwa.add_transition(start, 8, second_accept, tokens([2]));
        nwa.set_final_weight(first_accept, tokens([0, 1]));
        nwa.set_final_weight(second_accept, tokens([2]));

        let fast = determinize_impl(&nwa, true).unwrap();
        let generic = determinize_impl(&nwa, false).unwrap();

        assert_eq!(find_difference(&fast, &generic).unwrap(), None);
        assert_eq!(fast.eval_word(&[7]), tokens([0, 1]));
        assert_eq!(fast.eval_word(&[8]), tokens([2]));

        let mut multi_destination = NWA::new(1, 4);
        let start = multi_destination.add_state();
        let first_accept = multi_destination.add_state();
        let second_accept = multi_destination.add_state();
        multi_destination.set_start_states(vec![start]);
        multi_destination.add_transition(start, 9, first_accept, tokens([0]));
        multi_destination.add_transition(start, 9, second_accept, tokens([1]));
        multi_destination.set_final_weight(first_accept, tokens([0]));
        multi_destination.set_final_weight(second_accept, tokens([1]));

        let fast = determinize_impl(&multi_destination, true).unwrap();
        let generic = determinize_impl(&multi_destination, false).unwrap();
        assert_eq!(find_difference(&fast, &generic).unwrap(), None);
        assert_eq!(fast.eval_word(&[9]), tokens([0, 1]));
    }
}
