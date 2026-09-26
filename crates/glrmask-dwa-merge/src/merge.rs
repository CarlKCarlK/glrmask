//! Merge multiple (InternalIdMap, DWA) pairs into one.
//!
//! Handles both overlapping vocabs (e.g., L1 + L2+ from the same partition)
//! and disjoint vocabs (e.g., different character-type partitions) uniformly
//! via composite-key refinement.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::Arc;
use std::time::Instant;

use range_set_blaze::RangeSetBlaze;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::automata::weighted::determinize::{determinize, normalize_direct_singletons_enabled};
use crate::automata::weighted::equivalence::find_difference;
use crate::automata::weighted::dwa::{DWA, DWAState};
use crate::automata::weighted::minimize::minimize_owned;
use crate::automata::weighted::minimize_token_deterministic_nwa::{
    minimize_token_deterministic_nwa_owned, quotient_disjoint_source_nwa_owned,
};
use crate::automata::weighted::nwa::NWA;
use crate::compiler::glr::labels::DEFAULT_LABEL;
use crate::compiler::stages::equiv_types::{InternalIdMap, ManyToOneIdMap};
use crate::compiler::stages::mapped_artifact::MappedArtifact;
use crate::ds::weight::{ScopedWeightOpCache, Weight};

use super::types::{LocalIdMapTerminalDwa, TerminalDwaPhaseProfile, compile_profile_enabled};

type RemapCache<T> = FxHashMap<usize, T>;
type CompositeClassKey = SmallVec<[u32; 16]>;

enum FastGlobalTokenMaps {
    Direct(Vec<Vec<u32>>),
    General(Vec<Vec<Vec<u32>>>),
}

fn merged_terminal_dwa_phase_enabled(variable: &str, default: bool) -> bool {
    std::env::var(variable)
        .map(|value| {
            let trimmed = value.trim();
            !trimmed.is_empty() && trimmed != "0" && !trimmed.eq_ignore_ascii_case("false")
        })
        .unwrap_or(default)
}

fn minimize_merged_terminal_dwa_enabled(default: bool) -> bool {
    merged_terminal_dwa_phase_enabled("GLRMASK_MINIMIZE_MERGED_TERMINAL_DWA", default)
}

fn compact_merged_terminal_dwa_enabled(default: bool) -> bool {
    merged_terminal_dwa_phase_enabled("GLRMASK_COMPACT_MERGED_TERMINAL_DWA", default)
}

fn assert_direct_global_terminal_dwa_merge_equivalence_enabled() -> bool {
    std::env::var_os("GLRMASK_ASSERT_DIRECT_GLOBAL_TERMINAL_DWA_MERGE_EQUIVALENCE").is_some()
}

fn minimize_token_deterministic_terminal_nwa_enabled() -> bool {
    std::env::var_os("GLRMASK_EXPERIMENTAL_MINIMIZE_TOKEN_DETERMINISTIC_TERMINAL_NWA").is_some()
}

fn quotient_token_deterministic_terminal_nwa_by_source_enabled() -> bool {
    std::env::var_os("GLRMASK_EXPERIMENTAL_QUOTIENT_TOKEN_NWA_BY_SOURCE").is_some()
}

fn refine_token_deterministic_terminal_nwa_after_source_quotient_enabled() -> bool {
    std::env::var_os("GLRMASK_EXPERIMENTAL_REFINE_TOKEN_NWA_AFTER_SOURCE_QUOTIENT").is_some()
}

fn fast_disjoint_terminal_nwa_id_map_enabled() -> bool {
    std::env::var_os("GLRMASK_EXPERIMENTAL_FAST_DISJOINT_TERMINAL_NWA_ID_MAP").is_some()
}

fn primary_token_nwa_tsid_source() -> Option<usize> {
    std::env::var("GLRMASK_EXPERIMENTAL_ORDER_TOKEN_NWA_TSID_BY_SOURCE")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
}

fn select_primary_token_nwa_tsid_source(inputs: &[LocalIdMapTerminalDwa]) -> Option<usize> {
    primary_token_nwa_tsid_source().or_else(|| {
        inputs
            .iter()
            .enumerate()
            .max_by_key(|(_, input)| input.id_map.num_tsids())
            .map(|(index, _)| index)
    })
}

fn token_nwa_tsid_source_order(source_count: usize) -> Option<Vec<usize>> {
    let value = std::env::var("GLRMASK_EXPERIMENTAL_TOKEN_NWA_TSID_ORDER").ok()?;
    let order: Vec<usize> = value
        .split(',')
        .map(|part| part.trim().parse::<usize>().ok())
        .collect::<Option<_>>()?;
    let mut seen = vec![false; source_count];
    if order.len() != source_count
        || order.iter().any(|&source| source >= source_count || std::mem::replace(&mut seen[source], true))
    {
        return None;
    }
    Some(order)
}

#[derive(Default)]
struct TokenNwaRemapDetail {
    unique_weights: usize,
    full_weights: usize,
    compact_entries: usize,
    local_tsid_visits: usize,
    global_tsid_expansions: usize,
    max_local_entry_span: usize,
    max_global_entry_expansion: usize,
}

fn profile_token_nwa_remap_detail(
    dwa: &DWA,
    local_to_global_tsids: &[Vec<u32>],
) -> TokenNwaRemapDetail {
    let mut detail = TokenNwaRemapDetail::default();
    let mut seen = FxHashSet::<usize>::default();
    let mut inspect = |weight: &Weight| {
        if weight.is_empty() || !seen.insert(weight.ptr_key()) {
            return;
        }
        detail.unique_weights += 1;
        if weight.is_full() {
            detail.full_weights += 1;
            detail.global_tsid_expansions += local_to_global_tsids.iter().map(Vec::len).sum::<usize>();
            return;
        }
        let Some(entries) = weight.compact_entries() else {
            return;
        };
        detail.compact_entries += entries.len();
        for (start, end, _) in entries {
            let span = end.saturating_sub(start) as usize + 1;
            detail.local_tsid_visits += span;
            detail.max_local_entry_span = detail.max_local_entry_span.max(span);
            let mut entry_expansion = 0usize;
            for local_tsid in start..=end {
                let expansion = local_to_global_tsids
                    .get(local_tsid as usize)
                    .map_or(0, Vec::len);
                detail.global_tsid_expansions += expansion;
                entry_expansion += expansion;
            }
            detail.max_global_entry_expansion = detail.max_global_entry_expansion.max(entry_expansion);
        }
    };
    for state in dwa.states() {
        if let Some(final_weight) = &state.final_weight {
            inspect(final_weight);
        }
        for (_, weight) in state.transitions.values() {
            inspect(weight);
        }
    }
    detail
}

/// Return whether each source owns a disjoint set of original vocabulary tokens.
///
/// After remapping, this is stronger than a representation detail: every
/// non-empty weight from a source is contained in that source's global token
/// domain.  Consequently intersections of contributions from different
/// sources are empty, which lets `direct_union_disjoint_token_domain_dwas`
/// distribute the weighted path intersection over the branch union exactly.
fn inputs_have_disjoint_token_domains(inputs: &[LocalIdMapTerminalDwa], max_token_id: u32) -> bool {
    let mut owner = vec![u32::MAX; max_token_id as usize + 1];

    for (input_index, input) in inputs.iter().enumerate() {
        for (original_token, &local_token) in input
            .id_map
            .vocab_tokens
            .original_to_internal
            .iter()
            .enumerate()
        {
            if local_token == u32::MAX {
                continue;
            }
            let slot = &mut owner[original_token];
            if *slot != u32::MAX {
                return false;
            }
            *slot = input_index as u32;
        }
    }

    true
}

/// Token-class remapping used while merging id-map-local weights.
///
/// Disjoint global vocabularies are laid out as contiguous source blocks, so
/// their token mapping is an affine shift rather than a per-token relation.
enum TokenRemap<'a> {
    Explicit(&'a [Vec<u32>]),
    Offset { offset: u32, local_token_count: u32 },
}

impl TokenRemap<'_> {
    fn map_tokens(
        &self,
        token_key: usize,
        tokens: &RangeSetBlaze<u32>,
        token_cache: &mut RemapCache<Arc<RangeSetBlaze<u32>>>,
    ) -> Arc<RangeSetBlaze<u32>> {
        if let Some(existing) = token_cache.get(&token_key) {
            return Arc::clone(existing);
        }
        let mapped = match self {
            Self::Offset { offset, .. } => RangeSetBlaze::from_iter(tokens.ranges().map(|range| {
                let start = (*range.start())
                    .checked_add(*offset)
                    .expect("global token offset overflow");
                let end = (*range.end())
                    .checked_add(*offset)
                    .expect("global token offset overflow");
                start..=end
            })),
            Self::Explicit(local_to_global_tokens) => {
                let mut result = RangeSetBlaze::new();
                for local_token in tokens.iter() {
                    if let Some(globals) = local_to_global_tokens.get(local_token as usize) {
                        for &global_token in globals {
                            result.insert(global_token);
                        }
                    }
                }
                result
            }
        };
        let mapped = Arc::new(mapped);
        token_cache.insert(token_key, Arc::clone(&mapped));
        mapped
    }

    fn all_global_tokens(&self) -> RangeSetBlaze<u32> {
        match self {
            Self::Offset { offset, local_token_count } => {
                if *local_token_count == 0 {
                    RangeSetBlaze::new()
                } else {
                    RangeSetBlaze::from_iter(std::iter::once(
                        *offset..=offset
                            .checked_add(*local_token_count - 1)
                            .expect("global token offset overflow"),
                    ))
                }
            }
            Self::Explicit(local_to_global_tokens) => {
                let mut all_global_tokens = RangeSetBlaze::new();
                for globals in *local_to_global_tokens {
                    for &global_token in globals {
                        all_global_tokens.insert(global_token);
                    }
                }
                all_global_tokens
            }
        }
    }
}

fn build_local_to_global_tsid_runs(
    local_to_global_tsids: &[Vec<u32>],
) -> Vec<SmallVec<[(u32, u32); 1]>> {
    local_to_global_tsids
        .iter()
        .map(|globals| {
            let mut runs = SmallVec::<[(u32, u32); 1]>::new();
            for &global_tsid in globals {
                if let Some((_, end)) = runs.last_mut() {
                    if global_tsid == end.saturating_add(1) {
                        *end = global_tsid;
                        continue;
                    }
                }
                runs.push((global_tsid, global_tsid));
            }
            runs
        })
        .collect()
}

/// Remap a deterministic weighted automaton directly, preserving the same
/// branch-local token domain handling as `remap_nwa_with_maps`.
fn remap_dwa_with_maps(
    dwa: &mut DWA,
    local_to_global_tsids: &[Vec<u32>],
    local_to_global_tokens: &[Vec<u32>],
    global_tsid_count: usize,
) {
    remap_dwa_with_token_remap(
        dwa,
        local_to_global_tsids,
        TokenRemap::Explicit(local_to_global_tokens),
        global_tsid_count,
    );
}

fn remap_dwa_with_token_offset(
    dwa: &mut DWA,
    local_to_global_tsids: &[Vec<u32>],
    token_offset: u32,
    local_token_count: u32,
    global_tsid_count: usize,
) {
    remap_dwa_with_token_remap(
        dwa,
        local_to_global_tsids,
        TokenRemap::Offset {
            offset: token_offset,
            local_token_count,
        },
        global_tsid_count,
    );
}

fn remap_dwa_with_token_remap(
    dwa: &mut DWA,
    local_to_global_tsids: &[Vec<u32>],
    token_remap: TokenRemap<'_>,
    global_tsid_count: usize,
) {
    let profile = compile_profile_enabled();
    let local_to_global_tsid_runs = build_local_to_global_tsid_runs(local_to_global_tsids);
    let mut weight_cache = RemapCache::<Weight>::default();
    let mut token_cache = RemapCache::<Arc<RangeSetBlaze<u32>>>::default();
    let mut weights_seen = 0usize;
    let input_states = dwa.num_states();
    let input_transitions = dwa.num_transitions();

    for state in dwa.states_mut() {
        if let Some(final_weight) = state.final_weight.as_mut() {
            weights_seen += 1;
            *final_weight = remap_weight_cached_with_tsid_runs(
                final_weight,
                local_to_global_tsids,
                &local_to_global_tsid_runs,
                &token_remap,
                global_tsid_count,
                &mut weight_cache,
                &mut token_cache,
            );
            if final_weight.is_empty() {
                state.final_weight = None;
            }
        }

        for (_, weight) in state.transitions.values_mut() {
            weights_seen += 1;
            *weight = remap_weight_cached_with_tsid_runs(
                weight,
                local_to_global_tsids,
                &local_to_global_tsid_runs,
                &token_remap,
                global_tsid_count,
                &mut weight_cache,
                &mut token_cache,
            );
        }
        state.transitions.retain(|_, (_, weight)| !weight.is_empty());
    }
    if profile {
        eprintln!(
            "[glrmask/profile][terminal_weight_remap] states={} transitions={} weights_seen={} unique_weights={} weight_cache_hits={} unique_token_sets={}",
            input_states,
            input_transitions,
            weights_seen,
            weight_cache.len(),
            weights_seen.saturating_sub(weight_cache.len()),
            token_cache.len(),
        );
    }
}


fn contiguous_tsid_blocks(local_to_global_tsids: &[Vec<u32>]) -> Option<Vec<(u32, u32)>> {
    let mut blocks = Vec::with_capacity(local_to_global_tsids.len());
    let mut expected_start = 0u32;
    for globals in local_to_global_tsids {
        let start = *globals.first()?;
        let end = *globals.last()?;
        if start != expected_start
            || globals
                .iter()
                .enumerate()
                .any(|(offset, &global)| start.checked_add(offset as u32) != Some(global))
        {
            return None;
        }
        expected_start = end.checked_add(1)?;
        blocks.push((start, end));
    }
    Some(blocks)
}

fn remap_dwa_with_contiguous_tsid_blocks_offset(
    dwa: &mut DWA,
    blocks: &[(u32, u32)],
    token_offset: u32,
    local_token_count: u32,
) {
    let token_remap = TokenRemap::Offset {
        offset: token_offset,
        local_token_count,
    };
    let mut weight_cache = RemapCache::<Weight>::default();
    let mut token_cache = RemapCache::<Arc<RangeSetBlaze<u32>>>::default();
    for state in dwa.states_mut() {
        if let Some(final_weight) = state.final_weight.as_mut() {
            *final_weight = remap_weight_with_contiguous_tsid_blocks_cached(
                final_weight,
                blocks,
                &token_remap,
                &mut weight_cache,
                &mut token_cache,
            );
            if final_weight.is_empty() {
                state.final_weight = None;
            }
        }
        for (_, weight) in state.transitions.values_mut() {
            *weight = remap_weight_with_contiguous_tsid_blocks_cached(
                weight,
                blocks,
                &token_remap,
                &mut weight_cache,
                &mut token_cache,
            );
        }
        state.transitions.retain(|_, (_, weight)| !weight.is_empty());
    }
}

fn remap_weight_with_contiguous_tsid_blocks_cached(
    weight: &Weight,
    blocks: &[(u32, u32)],
    token_remap: &TokenRemap<'_>,
    cache: &mut RemapCache<Weight>,
    token_cache: &mut RemapCache<Arc<RangeSetBlaze<u32>>>,
) -> Weight {
    let key = weight.ptr_key();
    if let Some(existing) = cache.get(&key) {
        return existing.clone();
    }
    let remapped = remap_weight_with_contiguous_tsid_blocks(weight, blocks, token_remap, token_cache);
    cache.insert(key, remapped.clone());
    remapped
}

fn remap_weight_with_contiguous_tsid_blocks(
    weight: &Weight,
    blocks: &[(u32, u32)],
    token_remap: &TokenRemap<'_>,
    token_cache: &mut RemapCache<Arc<RangeSetBlaze<u32>>>,
) -> Weight {
    use crate::ds::weight::finalize_weight_map;
    use range_set_blaze::RangeMapBlaze;

    if weight.is_empty() || blocks.is_empty() {
        return Weight::empty();
    }
    let mut map = RangeMapBlaze::<u32, Arc<RangeSetBlaze<u32>>>::new();
    if weight.is_full() {
        let tokens = Arc::new(token_remap.all_global_tokens());
        if tokens.is_empty() {
            return Weight::empty();
        }
        map.extend_simple(std::iter::once((blocks[0].0..=blocks.last().unwrap().1, tokens)));
        return finalize_weight_map(map);
    }
    let Some(entries) = weight.compact_entries() else {
        return weight.clone();
    };
    for (start, end, tokens) in entries {
        let Some((global_start, _)) = blocks.get(start as usize).copied() else {
            continue;
        };
        let Some((_, global_end)) = blocks.get(end as usize).copied() else {
            continue;
        };
        let token_key = Arc::as_ptr(&tokens) as usize;
        let mapped_tokens = token_remap.map_tokens(token_key, tokens.as_ref(), token_cache);
        if !mapped_tokens.is_empty() {
            map.extend_simple(std::iter::once((global_start..=global_end, mapped_tokens)));
        }
    }
    finalize_weight_map(map)
}

/// Deterministically union automata whose global token domains are disjoint.
///
/// For a word `x`, source `i` contributes `W_i(x)`, the intersection of its
/// edge and final weights.  Disjoint source domains imply
/// `(⋃ A_i) ∩ (⋃ B_i) = ⋃ (A_i ∩ B_i)`, so the subset construction need only
/// remember one deterministic state per source rather than a weighted subset.
/// Missing component states represent dead paths.  The output is exactly the
/// ordinary NWA-union determinization, before minimization.
fn direct_union_disjoint_token_domain_dwas(inputs: &[DWA]) -> DWA {
    assert!(!inputs.is_empty());

    const DEAD: u32 = u32::MAX;

    let mut output = DWA::new(0, 0);
    let start_tuple: Vec<u32> = inputs.iter().map(DWA::start_state).collect();
    let mut state_ids = FxHashMap::<Vec<u32>, u32>::default();
    let mut worklist = VecDeque::<Vec<u32>>::new();
    state_ids.insert(start_tuple.clone(), output.start_state());
    worklist.push_back(start_tuple);

    while let Some(component_states) = worklist.pop_front() {
        let from_state = state_ids[&component_states];

        let final_weight = Weight::union_all(
            component_states
                .iter()
                .enumerate()
                .filter_map(|(input_index, &state_id)| {
                    (state_id != DEAD)
                        .then(|| inputs[input_index].states()[state_id as usize].final_weight.as_ref())
                        .flatten()
                }),
        );
        if !final_weight.is_empty() {
            output.set_final_weight(from_state, final_weight);
        }

        let mut by_label = FxHashMap::<i32, Vec<(usize, u32, Weight)>>::default();
        for (input_index, &state_id) in component_states.iter().enumerate() {
            if state_id == DEAD {
                continue;
            }
            for (&label, &(target, ref weight)) in &inputs[input_index].states()[state_id as usize].transitions {
                by_label
                    .entry(label)
                    .or_default()
                    .push((input_index, target, weight.clone()));
            }
        }

        for (label, transitions) in by_label {
            let edge_weight = Weight::union_all(transitions.iter().map(|(_, _, weight)| weight));
            if edge_weight.is_empty() {
                continue;
            }

            let mut target_tuple = vec![DEAD; inputs.len()];
            for (input_index, target, _) in transitions {
                target_tuple[input_index] = target;
            }

            let to_state = if let Some(&existing) = state_ids.get(&target_tuple) {
                existing
            } else {
                let new_state = output.add_state();
                state_ids.insert(target_tuple.clone(), new_state);
                worklist.push_back(target_tuple);
                new_state
            };
            output.add_transition(from_state, label, to_state, edge_weight);
        }
    }

    debug_assert!(output.is_acyclic());
    output
}

fn contiguous_token_offset(local_to_global: &[u32]) -> Option<u32> {
    let offset = *local_to_global.first()?;
    if offset == u32::MAX {
        return None;
    }
    local_to_global
        .iter()
        .enumerate()
        .all(|(local, &global)| {
            global != u32::MAX && offset.checked_add(local as u32) == Some(global)
        })
        .then_some(offset)
}

/// Build the exact deterministic union without subset determinization whenever
/// the source vocabularies are disjoint.  The generic NWA route remains the
/// fallback for overlapping domains.
fn try_merge_disjoint_token_domain_dwas(
    inputs: &[LocalIdMapTerminalDwa],
    global_id_map: &InternalIdMap,
    direct_local_to_global_token_maps: Option<&Vec<Vec<u32>>>,
    max_token_id: u32,
) -> Option<DWA> {
    if !inputs_have_disjoint_token_domains(inputs, max_token_id) {
        return None;
    }

    let profiling = compile_profile_enabled();
    let mut build_maps_ms = 0.0;
    let mut clone_ms = 0.0;
    let mut remap_ms = 0.0;
    let mut remapped = Vec::with_capacity(inputs.len());
    for (input_index, input) in inputs.iter().enumerate() {
        let started_at = profiling.then(Instant::now);
        let tsid_map = build_local_to_global_tsid_map(&input.id_map, global_id_map);
        let direct_token_map = direct_local_to_global_token_maps
            .and_then(|maps| maps.get(input_index));
        let token_offset = direct_token_map.and_then(|map| contiguous_token_offset(map));
        if profiling {
            let tsid_pairs = tsid_map.iter().map(Vec::len).sum::<usize>();
            let tsid_one_to_one = tsid_map.iter().filter(|targets| targets.len() == 1).count();
            let tsid_max_fanout = tsid_map.iter().map(Vec::len).max().unwrap_or(0);
            eprintln!(
                "[glrmask/profile][direct_disjoint_terminal_dwa_maps] input={} local_tsids={} tsid_pairs={} tsid_one_to_one={} tsid_max_fanout={} local_tokens={} token_offset={:?}",
                input_index,
                tsid_map.len(),
                tsid_pairs,
                tsid_one_to_one,
                tsid_max_fanout,
                input.id_map.num_internal_tokens(),
                token_offset,
            );
        }
        if let Some(started_at) = started_at {
            build_maps_ms += started_at.elapsed().as_secs_f64() * 1000.0;
        }
        let started_at = profiling.then(Instant::now);
        let mut dwa = input.dwa.clone();
        if let Some(started_at) = started_at {
            clone_ms += started_at.elapsed().as_secs_f64() * 1000.0;
        }
        let started_at = profiling.then(Instant::now);
        if let Some(token_offset) = token_offset {
            remap_dwa_with_token_offset(
                &mut dwa,
                &tsid_map,
                token_offset,
                input.id_map.num_internal_tokens(),
                global_id_map.num_tsids() as usize,
            );
        } else {
            let token_map = direct_token_map
                .map(|direct_map| build_direct_local_to_global_token_map(direct_map))
                .unwrap_or_else(|| build_local_to_global_token_map(&input.id_map, global_id_map));
            remap_dwa_with_maps(
                &mut dwa,
                &tsid_map,
                &token_map,
                global_id_map.num_tsids() as usize,
            );
        }
        if let Some(started_at) = started_at {
            remap_ms += started_at.elapsed().as_secs_f64() * 1000.0;
        }
        remapped.push(dwa);
    }
    let union_started_at = profiling.then(Instant::now);
    let result = direct_union_disjoint_token_domain_dwas(&remapped);
    if let Some(union_started_at) = union_started_at {
        eprintln!(
            "[glrmask/profile][direct_disjoint_terminal_dwa_union] inputs={} build_maps_ms={:.3} clone_ms={:.3} remap_ms={:.3} union_ms={:.3} states={} transitions={}",
            inputs.len(),
            build_maps_ms,
            clone_ms,
            remap_ms,
            union_started_at.elapsed().as_secs_f64() * 1000.0,
            result.num_states(),
            result.num_transitions(),
        );
    }

    Some(result)
}

/// Merge local branch outputs for a single partition into one compacted DWA.
pub fn merge_local_id_maps_and_terminal_dwas(
    inputs: Vec<LocalIdMapTerminalDwa>,
    num_tokenizer_states: usize,
    max_token_id: u32,
) -> LocalIdMapTerminalDwa {
    assert!(!inputs.is_empty(), "merge_local_id_maps_and_terminal_dwas called with empty inputs");

    if inputs.len() == 1 {
        let mut input = inputs.into_iter().next().unwrap();
        input.profile = TerminalDwaPhaseProfile::default();
        return input;
    }

    let input_refs: Vec<&LocalIdMapTerminalDwa> = inputs.iter().collect();
    let id_map_refs: Vec<&InternalIdMap> = input_refs.iter().map(|input| &input.id_map).collect();
    let (global_id_map, direct_local_to_global_token_maps) =
        build_unified_global_id_map(&id_map_refs, num_tokenizer_states, max_token_id);

    let pre_minimize = try_merge_disjoint_token_domain_dwas(
        &inputs,
        &global_id_map,
        direct_local_to_global_token_maps.as_ref(),
        max_token_id,
    )
    .unwrap_or_else(|| {
        let mut global_nwa = NWA::new(
            global_id_map.num_tsids(),
            global_id_map.max_internal_token_id(),
        );
        let mut global_body = global_nwa.body();

        for local in &inputs {
            let mut nwa = local.dwa.to_nwa();
            let tsid_map = build_local_to_global_tsid_map(&local.id_map, &global_id_map);
            let token_map = build_local_to_global_token_map(&local.id_map, &global_id_map);
            remap_nwa_with_maps(
                &mut nwa,
                &tsid_map,
                &token_map,
                global_id_map.num_tsids() as usize,
            );
            global_body = global_nwa.union_in_place(&nwa, &global_body);
        }
        global_nwa.set_start_states(global_body.start_states);
        determinize(&global_nwa).expect("merge terminal NWA determinization failed")
    });
    let compact_started_at = Instant::now();
    let mut mapped_dwa = MappedArtifact::new(minimize_owned(pre_minimize), global_id_map);
    mapped_dwa.compact_dimensions_fast();
    let compact_ms = compact_started_at.elapsed().as_secs_f64() * 1000.0;
    let (dwa, id_map) = mapped_dwa.into_parts();

    LocalIdMapTerminalDwa {
        id_map,
        dwa,
        profile: TerminalDwaPhaseProfile {
            compact_ms,
            ..TerminalDwaPhaseProfile::default()
        },
    }
}

/// Union mapped deterministic weighted automata in a common ID space.
///
/// The merge machinery is independent of how the input DWAs were produced;
/// parser DWAs use the same TSID/token-labelled weights as terminal DWAs.  This
/// wrapper keeps parser construction from depending on the terminal-specific
/// transport struct while reusing the exact remap/union/determinize path.
pub fn merge_mapped_dwas(
    inputs: Vec<MappedArtifact<DWA>>,
    num_tokenizer_states: usize,
    max_token_id: u32,
) -> MappedArtifact<DWA> {
    assert!(!inputs.is_empty(), "merge_mapped_dwas called with empty inputs");
    let locals = inputs
        .into_iter()
        .map(|input| {
            let (dwa, id_map) = input.into_parts();
            LocalIdMapTerminalDwa {
                id_map,
                dwa,
                profile: TerminalDwaPhaseProfile::default(),
            }
        })
        .collect();
    let merged = merge_local_id_maps_and_terminal_dwas(
        locals,
        num_tokenizer_states,
        max_token_id,
    );
    MappedArtifact::new(merged.dwa, merged.id_map)
}

fn immediate_dwa_accepting_edges(dwa: &DWA) -> Option<Vec<(i32, Weight)>> {
    if dwa.states().len() != 2 {
        return None;
    }
    let start_id = dwa.start_state() as usize;
    let final_id = 1usize.checked_sub(start_id)?;
    let start = &dwa.states()[start_id];
    let final_state = &dwa.states()[final_id];
    if start
        .final_weight
        .as_ref()
        .is_some_and(|weight| !weight.is_empty())
        || !final_state.transitions.is_empty()
    {
        return None;
    }
    let final_weight = final_state.final_weight.as_ref()?;
    if final_weight.is_empty() || start.transitions.is_empty() {
        return None;
    }

    let mut accepting = Vec::with_capacity(start.transitions.len());
    for (&label, (target, edge_weight)) in &start.transitions {
        if label < 0 || *target as usize != final_id || edge_weight.is_empty() {
            return None;
        }
        let weight = edge_weight.intersection(final_weight);
        if weight.is_empty() {
            return None;
        }
        accepting.push((label, weight));
    }
    Some(accepting)
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct ExactWeightedStateSignature {
    final_weight: Option<usize>,
    transitions: Vec<(i32, usize, usize)>,
}

/// Merge only states with byte-for-byte identical weighted behavior modulo
/// already-merged children. This is linear-ish hash-consing for an acyclic DWA,
/// not the compatibility search performed by the full weighted minimizer.
fn exact_quotient_acyclic_dwa(dwa: DWA) -> DWA {
    if dwa.states().is_empty() || !dwa.is_acyclic() {
        return dwa;
    }

    fn visit(state: usize, dwa: &DWA, seen: &mut [bool], order: &mut Vec<usize>) {
        if seen[state] {
            return;
        }
        seen[state] = true;
        for (target, weight) in dwa.states()[state].transitions.values() {
            let target = *target as usize;
            if !weight.is_empty() && target < dwa.states().len() {
                visit(target, dwa, seen, order);
            }
        }
        order.push(state);
    }

    let mut seen = vec![false; dwa.states().len()];
    let mut order = Vec::with_capacity(dwa.states().len());
    visit(dwa.start_state() as usize, &dwa, &mut seen, &mut order);

    let mut class_of = vec![usize::MAX; dwa.states().len()];
    let mut signature_to_class = FxHashMap::<ExactWeightedStateSignature, usize>::default();
    let mut representatives = Vec::<usize>::new();
    for &state_id in &order {
        let state = &dwa.states()[state_id];
        let signature = ExactWeightedStateSignature {
            final_weight: state.final_weight.as_ref().map(Weight::ptr_key),
            transitions: state
                .transitions
                .iter()
                .filter_map(|(&label, (target, weight))| {
                    (!weight.is_empty()).then_some((
                        label,
                        class_of[*target as usize],
                        weight.ptr_key(),
                    ))
                })
                .collect(),
        };
        let class = if let Some(&class) = signature_to_class.get(&signature) {
            class
        } else {
            let class = representatives.len();
            signature_to_class.insert(signature, class);
            representatives.push(state_id);
            class
        };
        class_of[state_id] = class;
    }

    let mut states = vec![DWAState::default(); representatives.len()];
    for (class, &representative) in representatives.iter().enumerate() {
        let source = &dwa.states()[representative];
        states[class].final_weight = source.final_weight.clone();
        states[class].transitions = source
            .transitions
            .iter()
            .filter_map(|(&label, (target, weight))| {
                (!weight.is_empty()).then_some((
                    label,
                    (class_of[*target as usize] as u32, weight.clone()),
                ))
            })
            .collect();
    }
    DWA::from_parts(states, class_of[dwa.start_state() as usize] as u32)
}

/// Fold a depth-one parser language into another deterministic parser DWA.
///
/// For a label accepted by both inputs, the merged start edge reaches a
/// decorated copy of the other destination. The copy admits the immediate
/// branch in its final weight, while every continuation edge is intersected
/// with the other branch's incoming weight. This preserves the token-path
/// correlation exactly, even when the two source weight domains overlap.
fn try_graft_immediate_parser_dwa(immediate: &DWA, mut other: DWA) -> Option<DWA> {
    let accepting = immediate_dwa_accepting_edges(immediate)?;
    let start_id = other.start_state();
    let original_start = other.states().get(start_id as usize)?.clone();
    let mut decorated_by_key = FxHashMap::<(u32, usize, usize), u32>::default();
    let mut missing = Vec::<(i32, Weight)>::new();
    for (label, add_weight) in accepting {
        let existing = original_start
            .transitions
            .get(&label)
            .or_else(|| original_start.transitions.get(&DEFAULT_LABEL))
            .cloned();
        if let Some((target, edge_weight)) = existing {
            let key = (target, edge_weight.ptr_key(), add_weight.ptr_key());
            let decorated = if let Some(&existing) = decorated_by_key.get(&key) {
                existing
            } else {
                let source_state = if target == start_id {
                    &original_start
                } else {
                    other.states().get(target as usize)?
                };
                let mut decorated_state = source_state.clone();
                decorated_state.final_weight = Some(match decorated_state.final_weight.take() {
                    Some(final_weight) => final_weight.union(&add_weight),
                    None => add_weight.clone(),
                });
                decorated_state.transitions.retain(|_, (_, continuation_weight)| {
                    *continuation_weight = continuation_weight.intersection(&edge_weight);
                    !continuation_weight.is_empty()
                });
                let new_state = other.add_state();
                other.states_mut()[new_state as usize] = decorated_state;
                decorated_by_key.insert(key, new_state);
                new_state
            };
            other.add_transition(
                start_id,
                label,
                decorated,
                edge_weight.union(&add_weight),
            );
        } else {
            missing.push((label, add_weight));
        }
    }

    if !missing.is_empty() {
        let sink = other.add_state();
        other.set_final_weight(sink, Weight::all());
        for (label, weight) in missing {
            other.add_transition(start_id, label, sink, weight);
        }
    }
    Some(other)
}


/// Exact weighted union specialized for two deterministic, epsilon-free DWAs.
///
/// When generic determinization normalizes singleton subsets `(q, w)` to
/// `(q, all)`, their source rows can be moved directly. Otherwise singleton
/// residuals remain explicit. This distinction preserves structural fallback
/// row selection, not just raw weighted language. Paired subsets always retain
/// at most one state from each input.
fn union_two_parser_dwas_direct(mut left: DWA, mut right: DWA) -> DWA {
    type PairKey = (u32, u32, Weight, Weight);
    const ABSENT: u32 = u32::MAX;

    fn intern_pair(
        left_state: u32,
        right_state: u32,
        left_weight: Weight,
        right_weight: Weight,
        source_state_count: u32,
        pair_states: &mut Vec<DWAState>,
        pairs: &mut FxHashMap<PairKey, u32>,
        worklist: &mut VecDeque<(u32, u32, u32, Weight, Weight)>,
    ) -> u32 {
        let key = (
            left_state,
            right_state,
            left_weight.clone(),
            right_weight.clone(),
        );
        if let Some(&state) = pairs.get(&key) {
            return state;
        }
        let state = source_state_count + pair_states.len() as u32;
        pair_states.push(DWAState::default());
        pairs.insert(key, state);
        worklist.push_back((
            state,
            left_state,
            right_state,
            left_weight,
            right_weight,
        ));
        state
    }

    let left_start = left.start_state();
    let right_start = right.start_state();
    let mut left_states = std::mem::take(left.states_mut());
    let mut right_states = std::mem::take(right.states_mut());
    let right_offset = left_states.len() as u32;
    let source_state_count = right_offset + right_states.len() as u32;
    // NWA::union_in_place appends each source without adding states. Use the
    // same size and override policy as the reference path.
    let normalize_singletons = normalize_direct_singletons_enabled(source_state_count as usize);
    // Generic expansion omits empty-weight edges. Copied singleton rows must
    // do the same, since an explicit empty edge would shadow DEFAULT_LABEL.
    for row in left_states.iter_mut().chain(right_states.iter_mut()) {
        row.transitions.retain(|_, (_, weight)| !weight.is_empty());
    }
    let mut pair_states = Vec::<DWAState>::new();

    let mut pairs = FxHashMap::<PairKey, u32>::default();
    let mut worklist = VecDeque::<(u32, u32, u32, Weight, Weight)>::new();
    let start = intern_pair(
        left_start,
        right_start,
        Weight::all(),
        Weight::all(),
        source_state_count,
        &mut pair_states,
        &mut pairs,
        &mut worklist,
    );
    let mut weight_ops = ScopedWeightOpCache::default();

    while let Some((out_state, left_id, right_id, left_residual, right_residual)) =
        worklist.pop_front()
    {
        let left_state = (left_id != ABSENT).then(|| &left_states[left_id as usize]);
        let right_state = (right_id != ABSENT).then(|| &right_states[right_id as usize]);
        let mut output = DWAState::default();

        if let Some(final_weight) = left_state.and_then(|state| state.final_weight.as_ref()) {
            let contribution = weight_ops.intersection(&left_residual, final_weight);
            if !contribution.is_empty() {
                output.final_weight = Some(contribution);
            }
        }
        if let Some(final_weight) = right_state.and_then(|state| state.final_weight.as_ref()) {
            let contribution = weight_ops.intersection(&right_residual, final_weight);
            if !contribution.is_empty() {
                output.final_weight = Some(match output.final_weight.take() {
                    Some(existing) => weight_ops.union(&existing, &contribution),
                    None => contribution,
                });
            }
        }

        let mut left_transitions = left_state.into_iter().flat_map(|state| state.transitions.iter()).peekable();
        let mut right_transitions = right_state.into_iter().flat_map(|state| state.transitions.iter()).peekable();
        loop {
            let label = match (left_transitions.peek(), right_transitions.peek()) {
                (Some((left_label, _)), Some((right_label, _))) => {
                    Some((**left_label).min(**right_label))
                }
                (Some((left_label, _)), None) => Some(**left_label),
                (None, Some((right_label, _))) => Some(**right_label),
                (None, None) => None,
            };
            let Some(label) = label else { break };

            let left_edge = if left_transitions
                .peek()
                .is_some_and(|(candidate, _)| **candidate == label)
            {
                left_transitions.next().map(|(_, edge)| edge)
            } else {
                None
            };
            let right_edge = if right_transitions
                .peek()
                .is_some_and(|(candidate, _)| **candidate == label)
            {
                right_transitions.next().map(|(_, edge)| edge)
            } else {
                None
            };

            let left_contribution = left_edge.and_then(|(target, weight)| {
                let contribution = weight_ops.intersection(&left_residual, weight);
                (!contribution.is_empty()).then_some((*target, contribution))
            });
            let right_contribution = right_edge.and_then(|(target, weight)| {
                let contribution = weight_ops.intersection(&right_residual, weight);
                (!contribution.is_empty()).then_some((*target, contribution))
            });

            let transition = match (left_contribution, right_contribution) {
                (None, None) => None,
                (Some((target, weight)), None) => {
                    let out_target = if normalize_singletons { target } else {
                        intern_pair(target, ABSENT, weight.clone(), Weight::empty(),
                            source_state_count, &mut pair_states, &mut pairs, &mut worklist)
                    };
                    Some((out_target, weight))
                }
                (None, Some((target, weight))) => {
                    let out_target = if normalize_singletons { right_offset + target } else {
                        intern_pair(ABSENT, target, Weight::empty(), weight.clone(),
                            source_state_count, &mut pair_states, &mut pairs, &mut worklist)
                    };
                    Some((out_target, weight))
                }
                (Some((left_target, left_weight)), Some((right_target, right_weight))) => {
                    let edge_weight = weight_ops.union(&left_weight, &right_weight);
                    let target = intern_pair(
                        left_target,
                        right_target,
                        left_weight,
                        right_weight,
                        source_state_count,
                        &mut pair_states,
                        &mut pairs,
                        &mut worklist,
                    );
                    Some((target, edge_weight))
                }
            };
            if let Some((target, weight)) = transition {
                output.transitions.insert(label, (target, weight));
            }
        }
        pair_states[(out_state - source_state_count) as usize] = output;
    }

    // Move source rows into the union with stable singleton IDs. Only right-side
    // source targets need the fixed offset; left-side IDs already match.
    for state in &mut right_states {
        for (target, _) in state.transitions.values_mut() {
            *target += right_offset;
        }
    }
    let mut states = left_states;
    states.reserve(right_states.len() + pair_states.len());
    states.extend(right_states);
    states.extend(pair_states);

    // Not every copied singleton row is reachable from the paired start. Compact
    // by moving the live rows; no BTreeMap/Weight cloning is needed.
    let mut reachable = vec![false; states.len()];
    let mut reachable_queue = VecDeque::new();
    reachable[start as usize] = true;
    reachable_queue.push_back(start);
    while let Some(state_id) = reachable_queue.pop_front() {
        for (target, _) in states[state_id as usize].transitions.values() {
            let target_idx = *target as usize;
            if !reachable[target_idx] {
                reachable[target_idx] = true;
                reachable_queue.push_back(*target);
            }
        }
    }

    let mut remap = vec![u32::MAX; states.len()];
    let mut next_state = 0u32;
    for (state_id, &is_reachable) in reachable.iter().enumerate() {
        if is_reachable {
            remap[state_id] = next_state;
            next_state += 1;
        }
    }
    let new_start = remap[start as usize];
    let mut compact = Vec::with_capacity(next_state as usize);
    for (state_id, mut state) in states.into_iter().enumerate() {
        if !reachable[state_id] {
            continue;
        }
        for (target, _) in state.transitions.values_mut() {
            let mapped = remap[*target as usize];
            debug_assert_ne!(mapped, u32::MAX);
            *target = mapped;
        }
        compact.push(state);
    }

    DWA::from_parts(compact, new_start)
}

/// Merge parser-family DWAs without paying for an unnecessary second global
/// weighted minimization.
///
/// A depth-one family is grafted exactly into the other family, including the
/// overlapping-weight case. Other shapes retain the ordinary generic merge as
/// a fallback. Parser-family artifacts are immediately reconciled with possible
/// matches and finalized, so globally minimizing them here can dominate compile
/// time without being required for correctness.
pub fn merge_mapped_parser_dwas(
    inputs: Vec<MappedArtifact<DWA>>,
    num_tokenizer_states: usize,
    max_token_id: u32,
) -> MappedArtifact<DWA> {
    assert!(
        !inputs.is_empty(),
        "merge_mapped_parser_dwas called with empty inputs"
    );
    if inputs.len() == 1 {
        return inputs.into_iter().next().unwrap();
    }

    if inputs.len() == 2 {
        let total_started_at = Instant::now();
        let mut iter = inputs.into_iter();
        let left = iter.next().unwrap();
        let right = iter.next().unwrap();
        let left_is_immediate = immediate_dwa_accepting_edges(left.artifact()).is_some();
        let right_is_immediate = immediate_dwa_accepting_edges(right.artifact()).is_some();

        // Reconcile with the deeper family first so refinements of each L2P
        // class remain contiguous. Then form the exact deterministic union.
        // Do not run the generic weighted minimizer here: parser DWAs interpret
        // DEFAULT_LABEL as fallback, while that minimizer treats labels as an
        // ordinary alphabet and can change the parser language.
        let reconcile_started_at = Instant::now();
        let reconciled = if left_is_immediate && !right_is_immediate {
            let paired = right.pair_forced_common(left);
            let ((right, left), id_map) = paired.into_parts();
            MappedArtifact::new(vec![left, right], id_map)
        } else if right_is_immediate && !left_is_immediate {
            let paired = left.pair_forced_common(right);
            let ((left, right), id_map) = paired.into_parts();
            MappedArtifact::new(vec![left, right], id_map)
        } else {
            MappedArtifact::reconcile_vec(vec![left, right])
        };
        let reconcile_ms = reconcile_started_at.elapsed().as_secs_f64() * 1000.0;
        let (dwas, common_id_map) = reconciled.into_parts();

        let source_states: usize = dwas.iter().map(|dwa| dwa.states().len()).sum();
        // Large normalized unions are dominated by copying and re-expanding
        // singleton rows. Small unions keep the existing builder unless an
        // explicit diagnostic asks for the residual-aware direct path.
        let direct_by_default = source_states >= 16_384
            && normalize_direct_singletons_enabled(source_states);
        let use_direct = std::env::var_os("GLRMASK_DISABLE_DIRECT_TWO_PARSER_UNION").is_none()
            && (direct_by_default
                || std::env::var_os("GLRMASK_EXPERIMENTAL_DIRECT_TWO_PARSER_UNION").is_some());
        if use_direct {
            let reference = if std::env::var_os("GLRMASK_VALIDATE_DIRECT_TWO_PARSER_UNION").is_some() {
                let mut reference_nwa = NWA::new(
                    common_id_map.num_tsids(),
                    common_id_map.max_internal_token_id(),
                );
                let mut body = reference_nwa.body();
                for source in &dwas {
                    body = reference_nwa.union_in_place(&source.to_nwa(), &body);
                }
                reference_nwa.set_start_states(body.start_states);
                Some(
                    determinize(&reference_nwa)
                        .expect("parser-family reference NWA union must determinize"),
                )
            } else {
                None
            };
            let union_started_at = Instant::now();
            let mut iter = dwas.into_iter();
            let dwa = union_two_parser_dwas_direct(
                iter.next().expect("two parser DWAs have a left input"),
                iter.next().expect("two parser DWAs have a right input"),
            );
            let union_ms = union_started_at.elapsed().as_secs_f64() * 1000.0;
            if let Some(reference) = reference {
                let difference = find_difference(&dwa, &reference)
                    .expect("direct parser-family union equivalence check failed");
                assert!(
                    difference.is_none(),
                    "direct parser-family union differs from generic union on labels {difference:?}",
                );
            }
            if compile_profile_enabled() {
                eprintln!(
                    "[glrmask/profile][parser_dwa_merge] inputs=2 mode=direct_two_dwa reconcile_ms={:.3} union_ms={:.3} states={} transitions={} total_ms={:.3}",
                    reconcile_ms,
                    union_ms,
                    dwa.num_states(),
                    dwa.num_transitions(),
                    total_started_at.elapsed().as_secs_f64() * 1000.0,
                );
            }
            return MappedArtifact::new(dwa, common_id_map);
        }

        let union_started_at = Instant::now();
        let mut union = NWA::new(
            common_id_map.num_tsids(),
            common_id_map.max_internal_token_id(),
        );
        let mut body = union.body();
        for dwa in dwas {
            body = union.union_in_place(&dwa.to_nwa(), &body);
        }
        union.set_start_states(body.start_states);
        let dwa = determinize(&union).expect("parser-family NWA union must determinize");
        let union_ms = union_started_at.elapsed().as_secs_f64() * 1000.0;
        if compile_profile_enabled() {
            eprintln!(
                "[glrmask/profile][parser_dwa_merge] inputs=2 mode=exact_raw_union reconcile_ms={:.3} union_ms={:.3} states={} transitions={} total_ms={:.3}",
                reconcile_ms,
                union_ms,
                dwa.num_states(),
                dwa.num_transitions(),
                total_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        return MappedArtifact::new(dwa, common_id_map);
    }

    let locals: Vec<LocalIdMapTerminalDwa> = inputs
        .into_iter()
        .map(|input| {
            let (dwa, id_map) = input.into_parts();
            LocalIdMapTerminalDwa {
                id_map,
                dwa,
                profile: TerminalDwaPhaseProfile::default(),
            }
        })
        .collect();
    let input_refs: Vec<&LocalIdMapTerminalDwa> = locals.iter().collect();
    let id_map_refs: Vec<&InternalIdMap> =
        input_refs.iter().map(|input| &input.id_map).collect();
    let total_started_at = Instant::now();
    let (global_id_map, direct_local_to_global_token_maps) =
        build_unified_global_id_map(&id_map_refs, num_tokenizer_states, max_token_id);

    if let Some(dwa) = try_merge_disjoint_token_domain_dwas(
        &locals,
        &global_id_map,
        direct_local_to_global_token_maps.as_ref(),
        max_token_id,
    ) {
        if compile_profile_enabled() {
            eprintln!(
                "[glrmask/profile][parser_dwa_merge] inputs={} mode=direct_disjoint states={} transitions={} total_ms={:.3}",
                locals.len(),
                dwa.num_states(),
                dwa.num_transitions(),
                total_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        return MappedArtifact::new(dwa, global_id_map);
    }

    // Overlapping source domains need the generic weighted union and
    // minimization used by the ordinary merge path.
    let inputs = locals
        .into_iter()
        .map(|local| MappedArtifact::new(local.dwa, local.id_map))
        .collect();
    merge_mapped_dwas(inputs, num_tokenizer_states, max_token_id)
}

/// Keep an immediate parser family as a top-state acceptance overlay instead
/// of materializing its union with the deeper parser DWA.
///
/// The returned DWA is the non-immediate family in the exact common ID space;
/// `top_accept[label]` is unioned after consuming the first stack symbol. This
/// is semantically the same union that `try_graft_immediate_parser_dwa` builds,
/// but avoids destination decoration and the subsequent quotient.
pub fn merge_mapped_parser_dwas_with_top_accept(
    inputs: Vec<MappedArtifact<DWA>>,
    num_tokenizer_states: usize,
    max_token_id: u32,
) -> (MappedArtifact<DWA>, BTreeMap<i32, Weight>) {
    assert!(
        !inputs.is_empty(),
        "merge_mapped_parser_dwas_with_top_accept called with empty inputs"
    );
    if inputs.len() == 1 {
        return (inputs.into_iter().next().unwrap(), BTreeMap::new());
    }

    if inputs.len() > 2 {
        let input_count = inputs.len();
        let immediate_count = inputs
            .iter()
            .filter(|input| immediate_dwa_accepting_edges(input.artifact()).is_some())
            .count();
        if immediate_count > 0 && immediate_count < input_count {
            let total_started_at = Instant::now();
            let mut primaries = Vec::with_capacity(input_count - immediate_count);
            let mut overlay: Option<MappedArtifact<BTreeMap<i32, Weight>>> = None;
            for input in inputs {
                if immediate_dwa_accepting_edges(input.artifact()).is_some() {
                    let (dwa, id_map) = input.into_parts();
                    let mapped = MappedArtifact::new(
                        immediate_dwa_accepting_edges(&dwa)
                            .expect("immediate parser shape was checked above")
                            .into_iter()
                            .collect::<BTreeMap<_, _>>(),
                        id_map,
                    );
                    overlay = Some(match overlay.take() {
                        None => mapped,
                        Some(existing) => {
                            let ((mut left, right), id_map) =
                                existing.pair_forced_common(mapped).into_parts();
                            for (label, weight) in right {
                                left.entry(label)
                                    .and_modify(|existing| *existing = existing.union(&weight))
                                    .or_insert(weight);
                            }
                            MappedArtifact::new(left, id_map)
                        }
                    });
                } else {
                    primaries.push(input);
                }
            }
            let primary = if primaries.len() == 1 {
                primaries.pop().unwrap()
            } else {
                merge_mapped_parser_dwas(primaries, num_tokenizer_states, max_token_id)
            };
            let overlay = overlay.expect("at least one immediate parser family was counted");
            let ((primary_dwa, top_accept), common_id_map) =
                primary.pair_forced_common(overlay).into_parts();
            if compile_profile_enabled() {
                eprintln!(
                    "[glrmask/profile][parser_dwa_merge] inputs={} mode=multi_top_accept_overlay overlay_families={} primary_families={} overlay_labels={} states={} transitions={} total_ms={:.3}",
                    input_count,
                    immediate_count,
                    input_count - immediate_count,
                    top_accept.len(),
                    primary_dwa.num_states(),
                    primary_dwa.num_transitions(),
                    total_started_at.elapsed().as_secs_f64() * 1000.0,
                );
            }
            return (
                MappedArtifact::new(primary_dwa, common_id_map),
                top_accept,
            );
        }
    }

    if inputs.len() == 2 {
        let total_started_at = Instant::now();
        let mut iter = inputs.into_iter();
        let left = iter.next().unwrap();
        let right = iter.next().unwrap();
        let left_is_immediate = immediate_dwa_accepting_edges(left.artifact()).is_some();
        let right_is_immediate = immediate_dwa_accepting_edges(right.artifact()).is_some();

        if left_is_immediate || right_is_immediate {
            let (primary, immediate) = if left_is_immediate {
                (right, left)
            } else {
                (left, right)
            };
            let (immediate_dwa, immediate_id_map) = immediate.into_parts();
            let top_accept: BTreeMap<i32, Weight> =
                immediate_dwa_accepting_edges(&immediate_dwa)
                    .expect("immediate parser shape was checked above")
                    .into_iter()
                    .collect();
            let reconciled = primary
                .pair_forced_common(MappedArtifact::new(top_accept, immediate_id_map));
            let ((primary_dwa, top_accept), common_id_map) = reconciled.into_parts();
            if compile_profile_enabled() {
                eprintln!(
                    "[glrmask/profile][parser_dwa_merge] inputs=2 mode=top_accept_overlay overlay_labels={} states={} transitions={} total_ms={:.3}",
                    top_accept.len(),
                    primary_dwa.num_states(),
                    primary_dwa.num_transitions(),
                    total_started_at.elapsed().as_secs_f64() * 1000.0,
                );
            }
            return (
                MappedArtifact::new(primary_dwa, common_id_map),
                top_accept,
            );
        }

        return (
            merge_mapped_parser_dwas(
                vec![left, right],
                num_tokenizer_states,
                max_token_id,
            ),
            BTreeMap::new(),
        );
    }

    (
        merge_mapped_parser_dwas(inputs, num_tokenizer_states, max_token_id),
        BTreeMap::new(),
    )
}

/// Build the ordinary global NWA union and determinize it. This is used only
/// as an opt-in semantic oracle for the specialized disjoint-domain merge.
fn generic_global_union_determinization(
    inputs: &[LocalIdMapTerminalDwa],
    global_id_map: &InternalIdMap,
    direct_local_to_global_token_maps: Option<&Vec<Vec<u32>>>,
) -> DWA {
    let mut global_nwa = NWA::new(
        global_id_map.num_tsids(),
        global_id_map.max_internal_token_id(),
    );
    let mut global_body = global_nwa.body();
    for (input_index, input) in inputs.iter().enumerate() {
        let mut nwa = input.dwa.to_nwa();
        let tsid_map = build_local_to_global_tsid_map(&input.id_map, global_id_map);
        let token_map = direct_local_to_global_token_maps
            .and_then(|maps| maps.get(input_index))
            .map(|direct_map| build_direct_local_to_global_token_map(direct_map))
            .unwrap_or_else(|| build_local_to_global_token_map(&input.id_map, global_id_map));
        remap_nwa_with_maps(
            &mut nwa,
            &tsid_map,
            &token_map,
            global_id_map.num_tsids() as usize,
        );
        global_body = global_nwa.union_in_place(&nwa, &global_body);
    }
    global_nwa.set_start_states(global_body.start_states);
    determinize(&global_nwa).expect("global terminal NWA reference determinization failed")
}

fn prune_unreachable_nwa_with_sources(
    nwa: NWA,
    state_sources: Vec<Option<usize>>,
) -> (NWA, Vec<Option<usize>>) {
    assert_eq!(state_sources.len(), nwa.states().len());
    let mut reachable = vec![false; nwa.states().len()];
    let mut stack = nwa.start_states().to_vec();
    while let Some(state_id) = stack.pop() {
        let index = state_id as usize;
        if index >= reachable.len() || reachable[index] {
            continue;
        }
        reachable[index] = true;
        let state = &nwa.states()[index];
        for branches in state.transitions.values() {
            stack.extend(branches.iter().map(|(target, _)| *target));
        }
        stack.extend(state.epsilons.iter().map(|(target, _)| *target));
    }

    let mut old_to_new = vec![u32::MAX; reachable.len()];
    let mut states = Vec::new();
    let mut sources = Vec::new();
    for (old, &is_reachable) in reachable.iter().enumerate() {
        if is_reachable {
            old_to_new[old] = states.len() as u32;
            states.push(nwa.states()[old].clone());
            sources.push(state_sources[old]);
        }
    }
    for state in &mut states {
        for branches in state.transitions.values_mut() {
            for (target, _) in branches {
                *target = old_to_new[*target as usize];
            }
        }
        for (target, _) in &mut state.epsilons {
            *target = old_to_new[*target as usize];
        }
    }
    let starts = nwa
        .start_states()
        .iter()
        .map(|start| old_to_new[*start as usize])
        .collect();
    (NWA::from_parts(states, starts), sources)
}


/// Build a global token-deterministic NWA when every source owns a disjoint
/// vocabulary domain. Multiple branches for one label are then deterministic
/// with respect to the source token, so parser construction can consume them
/// directly without materializing a product DWA.
pub fn try_merge_id_maps_and_token_deterministic_nwa(
    inputs: &[LocalIdMapTerminalDwa],
    num_tokenizer_states: usize,
    max_token_id: u32,
) -> Option<(NWA, InternalIdMap, TerminalDwaPhaseProfile)> {
    try_merge_id_maps_and_token_deterministic_nwa_impl(
        inputs,
        num_tokenizer_states,
        max_token_id,
        false,
    )
}

/// Build the same exact token-deterministic NWA union when the caller already
/// proved that every input owns a disjoint original-vocabulary domain.
///
/// Generic callers must use the checked entry point above. This entry point
/// avoids rediscovering the partition invariant by scanning the whole vocab.
pub fn try_merge_id_maps_and_token_deterministic_nwa_proven_disjoint(
    inputs: &[LocalIdMapTerminalDwa],
    num_tokenizer_states: usize,
    max_token_id: u32,
) -> Option<(NWA, InternalIdMap, TerminalDwaPhaseProfile)> {
    try_merge_id_maps_and_token_deterministic_nwa_impl(
        inputs,
        num_tokenizer_states,
        max_token_id,
        true,
    )
}

fn try_merge_id_maps_and_token_deterministic_nwa_impl(
    inputs: &[LocalIdMapTerminalDwa],
    num_tokenizer_states: usize,
    max_token_id: u32,
    token_domains_proven_disjoint: bool,
) -> Option<(NWA, InternalIdMap, TerminalDwaPhaseProfile)> {
    if inputs.len() < 2 {
        return None;
    }
    if token_domains_proven_disjoint {
        if std::env::var_os("GLRMASK_ASSERT_PROVEN_DISJOINT_TOKEN_NWA_DOMAINS").is_some() {
            assert!(
                inputs_have_disjoint_token_domains(inputs, max_token_id),
                "caller-proven token-NWA merge received overlapping vocabulary domains",
            );
        }
    } else if !inputs_have_disjoint_token_domains(inputs, max_token_id) {
        return None;
    }

    let total_started_at = Instant::now();
    let primary_tsid_source = select_primary_token_nwa_tsid_source(inputs)
        .filter(|source| *source < inputs.len());
    let id_map_refs: Vec<&InternalIdMap> = inputs.iter().map(|input| &input.id_map).collect();
    let id_map_started_at = Instant::now();
    let (global_id_map, direct_local_to_global_token_maps, precomputed_tsid_maps) =
        if token_domains_proven_disjoint || fast_disjoint_terminal_nwa_id_map_enabled() {
            let (id_map, direct_maps, tsid_maps) = build_unified_global_id_map_disjoint_fast(
                &id_map_refs,
                num_tokenizer_states,
                max_token_id,
                primary_tsid_source,
            );
            (id_map, Some(direct_maps), Some(tsid_maps))
        } else {
            let (id_map, direct_maps) =
                build_unified_global_id_map(&id_map_refs, num_tokenizer_states, max_token_id);
            (id_map, direct_maps, None)
        };
    let id_map_ms = id_map_started_at.elapsed().as_secs_f64() * 1000.0;
    let all_tsid_maps: Vec<Vec<Vec<u32>>> = precomputed_tsid_maps.unwrap_or_else(|| {
        inputs
            .iter()
            .map(|input| build_local_to_global_tsid_map(&input.id_map, &global_id_map))
            .collect()
    });
    let primary_tsid_blocks = primary_tsid_source
        .and_then(|source| contiguous_tsid_blocks(&all_tsid_maps[source]));

    let remap_started_at = Instant::now();
    let profiling = compile_profile_enabled();
    let mut global_nwa = NWA::new(
        global_id_map.num_tsids(),
        global_id_map.max_internal_token_id(),
    );
    let mut state_sources = Vec::<Option<usize>>::new();
    let mut combined_start: Option<u32> = None;
    let tsid_map_ms = 0.0;
    let mut token_map_ms = 0.0;
    let mut clone_ms = 0.0;
    let mut remap_ms = 0.0;
    let mut to_nwa_ms = 0.0;
    let mut append_ms = 0.0;
    let mut fuse_start_ms = 0.0;
    for (input_index, input) in inputs.iter().enumerate() {
        let tsid_map = &all_tsid_maps[input_index];
        let remap_detail = std::env::var_os("GLRMASK_PROFILE_TOKEN_NWA_REMAP_DETAIL")
            .is_some()
            .then(|| profile_token_nwa_remap_detail(&input.dwa, tsid_map));
        let started_at = profiling.then(Instant::now);
        let direct_token_map = direct_local_to_global_token_maps
            .as_ref()
            .and_then(|maps| maps.get(input_index));
        let token_offset = direct_token_map.and_then(|map| contiguous_token_offset(map));
        if let Some(started_at) = started_at {
            token_map_ms += started_at.elapsed().as_secs_f64() * 1000.0;
        }
        let started_at = profiling.then(Instant::now);
        let mut dwa = input.dwa.clone();
        if let Some(started_at) = started_at {
            clone_ms += started_at.elapsed().as_secs_f64() * 1000.0;
        }
        let started_at = profiling.then(Instant::now);
        if primary_tsid_source == Some(input_index) {
            if let (Some(token_offset), Some(blocks)) = (token_offset, primary_tsid_blocks.as_deref()) {
                remap_dwa_with_contiguous_tsid_blocks_offset(
                    &mut dwa,
                    blocks,
                    token_offset,
                    input.id_map.num_internal_tokens(),
                );
            } else if let Some(token_offset) = token_offset {
                remap_dwa_with_token_offset(
                    &mut dwa,
                    tsid_map,
                    token_offset,
                    input.id_map.num_internal_tokens(),
                    global_id_map.num_tsids() as usize,
                );
            } else {
                let token_map = direct_token_map
                    .map(|map| build_direct_local_to_global_token_map(map))
                    .unwrap_or_else(|| build_local_to_global_token_map(&input.id_map, &global_id_map));
                remap_dwa_with_maps(
                    &mut dwa,
                    tsid_map,
                    &token_map,
                    global_id_map.num_tsids() as usize,
                );
            }
        } else if let Some(token_offset) = token_offset {
            remap_dwa_with_token_offset(
                &mut dwa,
                tsid_map,
                token_offset,
                input.id_map.num_internal_tokens(),
                global_id_map.num_tsids() as usize,
            );
        } else {
            let token_map = direct_token_map
                .map(|map| build_direct_local_to_global_token_map(map))
                .unwrap_or_else(|| build_local_to_global_token_map(&input.id_map, &global_id_map));
            remap_dwa_with_maps(
                &mut dwa,
                tsid_map,
                &token_map,
                global_id_map.num_tsids() as usize,
            );
        }
        if let Some(started_at) = started_at {
            let elapsed_ms = started_at.elapsed().as_secs_f64() * 1000.0;
            remap_ms += elapsed_ms;
            let tsid_pairs = tsid_map.iter().map(Vec::len).sum::<usize>();
            let tsid_max_fanout = tsid_map.iter().map(Vec::len).max().unwrap_or(0);
            let tsid_runs = tsid_map
                .iter()
                .map(|globals| {
                    globals
                        .iter()
                        .enumerate()
                        .filter(|(index, global)| {
                            *index == 0 || **global != globals[*index - 1].saturating_add(1)
                        })
                        .count()
                })
                .sum::<usize>();
            eprintln!(
                "[glrmask/profile][token_deterministic_terminal_nwa_input] input={} states={} transitions={} local_tsids={} tsid_pairs={} tsid_runs={} tsid_max_fanout={} remap_ms={:.3}",
                input_index,
                input.dwa.num_states(),
                input.dwa.num_transitions(),
                input.id_map.num_tsids(),
                tsid_pairs,
                tsid_runs,
                tsid_max_fanout,
                elapsed_ms,
            );
        }
        if let Some(detail) = remap_detail {
            eprintln!(
                "[glrmask/profile][token_nwa_remap_detail] input={} unique_weights={} full_weights={} compact_entries={} local_tsid_visits={} global_tsid_expansions={} max_local_entry_span={} max_global_entry_expansion={}",
                input_index,
                detail.unique_weights,
                detail.full_weights,
                detail.compact_entries,
                detail.local_tsid_visits,
                detail.global_tsid_expansions,
                detail.max_local_entry_span,
                detail.max_global_entry_expansion,
            );
        }
        let started_at = profiling.then(Instant::now);
        let source_nwa = dwa.to_nwa();
        if let Some(started_at) = started_at {
            to_nwa_ms += started_at.elapsed().as_secs_f64() * 1000.0;
        }
        let started_at = profiling.then(Instant::now);
        let before_append = global_nwa.num_states() as usize;
        let body = global_nwa.append_with_body(&source_nwa);
        if let Some(started_at) = started_at {
            append_ms += started_at.elapsed().as_secs_f64() * 1000.0;
        }
        state_sources.extend(std::iter::repeat_n(Some(input_index), global_nwa.num_states() as usize - before_append));
        debug_assert_eq!(body.start_states.len(), 1);
        let source_start = body.start_states[0];
        let started_at = profiling.then(Instant::now);
        if let Some(combined_start) = combined_start {
            let source = global_nwa.states()[source_start as usize].clone();
            if let Some(final_weight) = source.final_weight {
                let target = &mut global_nwa.states_mut()[combined_start as usize].final_weight;
                *target = Some(match target.take() {
                    Some(existing) => existing.union(&final_weight),
                    None => final_weight,
                });
            }
            for (label, branches) in source.transitions {
                for (target, weight) in branches {
                    global_nwa.add_transition(combined_start, label, target, weight);
                }
            }
            assert!(source.epsilons.is_empty(), "source DWA conversion must not create epsilon edges");
        } else {
            combined_start = Some(source_start);
            state_sources[source_start as usize] = None;
        }
        if let Some(started_at) = started_at {
            fuse_start_ms += started_at.elapsed().as_secs_f64() * 1000.0;
        }
    }
    let started_at = profiling.then(Instant::now);
    global_nwa.set_start_states(vec![combined_start.expect("at least one input")]);
    let (mut global_nwa, state_sources) = prune_unreachable_nwa_with_sources(global_nwa, state_sources);
    let prune_ms = started_at.map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0).unwrap_or(0.0);
    let mut source_quotient_ms = 0.0;
    let mut refine_ms = 0.0;
    if quotient_token_deterministic_terminal_nwa_by_source_enabled() {
        let started_at = profiling.then(Instant::now);
        global_nwa = quotient_disjoint_source_nwa_owned(global_nwa, &state_sources)
            .expect("token-deterministic source quotient failed");
        if let Some(started_at) = started_at {
            source_quotient_ms = started_at.elapsed().as_secs_f64() * 1000.0;
        }
        if refine_token_deterministic_terminal_nwa_after_source_quotient_enabled() {
            let started_at = profiling.then(Instant::now);
            global_nwa = minimize_token_deterministic_nwa_owned(global_nwa)
                .expect("token-deterministic refinement after source quotient failed");
            if let Some(started_at) = started_at {
                refine_ms = started_at.elapsed().as_secs_f64() * 1000.0;
            }
        }
    } else if minimize_token_deterministic_terminal_nwa_enabled() {
        let started_at = profiling.then(Instant::now);
        global_nwa = minimize_token_deterministic_nwa_owned(global_nwa)
            .expect("token-deterministic terminal NWA minimization failed");
        if let Some(started_at) = started_at {
            refine_ms = started_at.elapsed().as_secs_f64() * 1000.0;
        }
    }
    debug_assert!(global_nwa.is_acyclic());
    debug_assert!(global_nwa.states().iter().all(|state| state.epsilons.is_empty()));
    let remap_and_union_ms = remap_started_at.elapsed().as_secs_f64() * 1000.0;

    if assert_direct_global_terminal_dwa_merge_equivalence_enabled() {
        let reference = generic_global_union_determinization(
            inputs,
            &global_id_map,
            direct_local_to_global_token_maps.as_ref(),
        );
        let determinized = determinize(&global_nwa)
            .expect("token-deterministic terminal NWA assertion determinization failed");
        let difference = find_difference(&determinized, &reference)
            .expect("token-deterministic terminal NWA equivalence requires acyclic inputs");
        assert!(
            difference.is_none(),
            "token-deterministic terminal NWA differs from generic NWA union on labels {:?}",
            difference,
        );
        if profiling {
            eprintln!("[glrmask/profile][token_deterministic_terminal_nwa_equivalence] result=equivalent");
        }
    }

    if profiling {
        eprintln!(
            "[glrmask/profile][token_deterministic_terminal_nwa_merge] inputs={} id_map_ms={:.3} remap_and_union_ms={:.3} tsid_map_ms={:.3} token_map_ms={:.3} clone_ms={:.3} remap_ms={:.3} to_nwa_ms={:.3} append_ms={:.3} fuse_start_ms={:.3} prune_ms={:.3} source_quotient_ms={:.3} refine_ms={:.3} states={} transitions={} total_ms={:.3}",
            inputs.len(),
            id_map_ms,
            remap_and_union_ms,
            tsid_map_ms,
            token_map_ms,
            clone_ms,
            remap_ms,
            to_nwa_ms,
            append_ms,
            fuse_start_ms,
            prune_ms,
            source_quotient_ms,
            refine_ms,
            global_nwa.num_states(),
            global_nwa.num_transitions(),
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }

    Some((
        global_nwa,
        global_id_map,
        TerminalDwaPhaseProfile {
            id_map_ms,
            global_merge_ms: total_started_at.elapsed().as_secs_f64() * 1000.0,
            ..TerminalDwaPhaseProfile::default()
        },
    ))
}


/// Merge already-compacted partition outputs into one global DWA.
///
/// Partition-local merges have already minimized their local outputs. When
/// global token domains are disjoint, an exact deterministic source-product is
/// used and remains unminimized/uncompacted by default; this avoids constructing
/// a large subset DWA only to erase it immediately. Overlapping domains retain
/// the generic determinize-minimize-compact route. `GLRMASK_MINIMIZE_MERGED_TERMINAL_DWA`
/// and `GLRMASK_COMPACT_MERGED_TERMINAL_DWA` explicitly override either default.
pub fn merge_id_maps_and_terminal_dwas(
    inputs: Vec<LocalIdMapTerminalDwa>,
    num_tokenizer_states: usize,
    max_token_id: u32,
) -> LocalIdMapTerminalDwa {
    merge_id_maps_and_terminal_dwas_impl(inputs, num_tokenizer_states, max_token_id, false)
}

/// Merge terminal-DWA pieces whose vocabulary domains are proven disjoint by
/// their caller. Generic callers should use `merge_id_maps_and_terminal_dwas`.
pub fn merge_id_maps_and_terminal_dwas_proven_disjoint(
    inputs: Vec<LocalIdMapTerminalDwa>,
    num_tokenizer_states: usize,
    max_token_id: u32,
) -> LocalIdMapTerminalDwa {
    merge_id_maps_and_terminal_dwas_impl(inputs, num_tokenizer_states, max_token_id, true)
}

fn merge_id_maps_and_terminal_dwas_impl(
    inputs: Vec<LocalIdMapTerminalDwa>,
    num_tokenizer_states: usize,
    max_token_id: u32,
    token_domains_proven_disjoint: bool,
) -> LocalIdMapTerminalDwa {
    assert!(!inputs.is_empty(), "merge_id_maps_and_terminal_dwas called with empty inputs");

    if inputs.len() == 1 {
        let mut input = inputs.into_iter().next().unwrap();
        input.profile = TerminalDwaPhaseProfile::default();
        return input;
    }

    if let Some(merged) = try_merge_immediate_terminal_dwas(
        &inputs,
        num_tokenizer_states,
        max_token_id,
        token_domains_proven_disjoint,
    ) {
        return merged;
    }

    let total_started_at = Instant::now();

    let build_unified_global_id_map_started_at = Instant::now();
    let id_map_refs: Vec<&InternalIdMap> = inputs.iter().map(|input| &input.id_map).collect();
    let (global_id_map, direct_local_to_global_token_maps) =
        build_unified_global_id_map(&id_map_refs, num_tokenizer_states, max_token_id);
    let build_unified_global_id_map_ms =
        build_unified_global_id_map_started_at.elapsed().as_secs_f64() * 1000.0;

    let remap_and_union_started_at = Instant::now();
    // The direct product is exact when token domains are disjoint. Unlike the
    // generic NWA route it keeps source-local deterministic contexts separate,
    // avoiding a large subset expansion that global minimization later erases.
    let direct_global = try_merge_disjoint_token_domain_dwas(
        &inputs,
        &global_id_map,
        direct_local_to_global_token_maps.as_ref(),
        max_token_id,
);
    let used_direct_global = direct_global.is_some();
    let mut global_nwa = NWA::new(
        global_id_map.num_tsids(),
        global_id_map.max_internal_token_id(),
    );
    let mut global_body = global_nwa.body();
    let profiling = compile_profile_enabled();
    let mut to_nwa_ms = 0.0;
    let mut build_tsid_map_ms = 0.0;
    let mut build_token_map_ms = 0.0;
    let mut remap_nwa_ms = 0.0;
    let mut union_ms = 0.0;

    if direct_global.is_none() {
        for (input_idx, input) in inputs.iter().enumerate() {
            let started_at = profiling.then(Instant::now);
            let mut nwa = input.dwa.to_nwa();
            if let Some(started_at) = started_at {
                to_nwa_ms += started_at.elapsed().as_secs_f64() * 1000.0;
            }

            let started_at = profiling.then(Instant::now);
            let tsid_map = build_local_to_global_tsid_map(&input.id_map, &global_id_map);
            if let Some(started_at) = started_at {
                build_tsid_map_ms += started_at.elapsed().as_secs_f64() * 1000.0;
            }

            let started_at = profiling.then(Instant::now);
            let token_map = direct_local_to_global_token_maps
                .as_ref()
                .and_then(|maps| maps.get(input_idx))
                .map(|direct_map| build_direct_local_to_global_token_map(direct_map))
                .unwrap_or_else(|| build_local_to_global_token_map(&input.id_map, &global_id_map));
            if let Some(started_at) = started_at {
                build_token_map_ms += started_at.elapsed().as_secs_f64() * 1000.0;
            }

            let started_at = profiling.then(Instant::now);
            remap_nwa_with_maps(
                &mut nwa,
                &tsid_map,
                &token_map,
                global_id_map.num_tsids() as usize,
            );
            if let Some(started_at) = started_at {
                remap_nwa_ms += started_at.elapsed().as_secs_f64() * 1000.0;
            }

            let started_at = profiling.then(Instant::now);
            global_body = global_nwa.union_in_place(&nwa, &global_body);
            if let Some(started_at) = started_at {
                union_ms += started_at.elapsed().as_secs_f64() * 1000.0;
            }
        }
        global_nwa.set_start_states(global_body.start_states);
    }
    let remap_and_union_ms = remap_and_union_started_at.elapsed().as_secs_f64() * 1000.0;
    let nwa_states_before_determinize = global_nwa.num_states();
    let nwa_transitions_before_determinize = global_nwa.num_transitions();

    let determinize_started_at = Instant::now();
    let det = direct_global.unwrap_or_else(|| {
        determinize(&global_nwa).expect("merge terminal NWA determinization failed")
    });
    let determinize_ms = determinize_started_at.elapsed().as_secs_f64() * 1000.0;
    let det_states = det.num_states();
    let det_transitions = det.num_transitions();

    if used_direct_global && assert_direct_global_terminal_dwa_merge_equivalence_enabled() {
        let reference = generic_global_union_determinization(
            &inputs,
            &global_id_map,
            direct_local_to_global_token_maps.as_ref(),
        );
        let difference = find_difference(&det, &reference)
            .expect("direct global terminal DWA equivalence requires acyclic inputs");
        assert!(
            difference.is_none(),
            "direct global terminal DWA differs from generic NWA union on labels {:?}",
            difference,
        );
        if profiling {
            eprintln!("[glrmask/profile][direct_global_terminal_dwa_equivalence] result=equivalent");
        }
    }

    // A direct product is already exact. Its slightly larger topology is much
    // cheaper to construct than generic subset expansion and remains suitable
    // for the downstream joint reconciliation pass, so do not minimize it by
    // default. Explicit environment values still override this strategy.
    let minimize_enabled = minimize_merged_terminal_dwa_enabled(!used_direct_global);
    let minimize_started_at = Instant::now();
    let dwa = if minimize_enabled {
        minimize_owned(det)
    } else {
        det
    };
    let minimize_ms = if minimize_enabled {
        minimize_started_at.elapsed().as_secs_f64() * 1000.0
    } else {
        0.0
    };
    let mut mapped_dwa = MappedArtifact::new(dwa, global_id_map);
    let before_compact_stats = profiling.then(|| mapped_dwa.artifact().stats());
    let before_compact_range_counts = profiling.then(|| mapped_dwa.interned_range_counts());
    let before_num_tsids = profiling.then(|| mapped_dwa.id_map().num_tsids());
    let before_num_tokens = profiling.then(|| mapped_dwa.id_map().num_internal_tokens());
    let compact_enabled = compact_merged_terminal_dwa_enabled(!used_direct_global);
    let (compact_report, compact_ms) = if compact_enabled {
        let compact_started_at = Instant::now();
        let compact_report = if profiling {
            mapped_dwa.compact_dimensions_fast_with_stats()
        } else {
            mapped_dwa.compact_dimensions_fast()
        };
        let compact_ms = compact_started_at.elapsed().as_secs_f64() * 1000.0;
        (Some(compact_report), compact_ms)
    } else {
        (None, 0.0)
    };
    let after_compact_stats = profiling.then(|| mapped_dwa.artifact().stats());
    let after_compact_range_counts = profiling.then(|| mapped_dwa.interned_range_counts());
    let total_ms = total_started_at.elapsed().as_secs_f64() * 1000.0;
    let (dwa, id_map) = mapped_dwa.into_parts();

    if profiling {
        let before_compact_stats = before_compact_stats.unwrap();
        let before_compact_range_counts = before_compact_range_counts.unwrap();
        let before_num_tsids = before_num_tsids.unwrap();
        let before_num_tokens = before_num_tokens.unwrap();
        let after_compact_stats = after_compact_stats.unwrap();
        let after_compact_range_counts = after_compact_range_counts.unwrap();
        let profile_stats = compact_report.and_then(|report| report.profile_stats);
        eprintln!(
            "[glrmask/profile][terminal_dwa_global_merge] inputs={} build_unified_global_id_map_ms={:.3} remap_and_union_ms={:.3} to_nwa_ms={:.3} build_tsid_map_ms={:.3} build_token_map_ms={:.3} remap_nwa_ms={:.3} union_ms={:.3} determinize_ms={:.3} minimize_ms={:.3} compact_ms={:.3} total_ms={:.3} nwa_states_before_determinize={} nwa_transitions_before_determinize={} det_states={} det_transitions={} dwa_states_before_compact={} dwa_transitions_before_compact={} dwa_states_after_compact={} dwa_transitions_after_compact={}",
            inputs.len(),
            build_unified_global_id_map_ms,
            remap_and_union_ms,
            to_nwa_ms,
            build_tsid_map_ms,
            build_token_map_ms,
            remap_nwa_ms,
            union_ms,
            determinize_ms,
            minimize_ms,
            compact_ms,
            total_ms,
            nwa_states_before_determinize,
            nwa_transitions_before_determinize,
            det_states,
            det_transitions,
            before_compact_stats.states,
            before_compact_stats.transitions,
            after_compact_stats.states,
            after_compact_stats.transitions,
        );
        eprintln!(
            "[glrmask/profile][merged_terminal_dwa] minimize_enabled={} compact_enabled={} states_before_compact={} transitions_before_compact={} interned_ranges_before_compact={} token_ranges_before_compact={} states_after_compact={} transitions_after_compact={} interned_ranges_after_compact={} token_ranges_after_compact={} tsids_before_compact={} tsids_after_compact={} tokens_before_compact={} tokens_after_compact={} compact_ms={:.3}",
            minimize_enabled,
            compact_enabled,
            before_compact_stats.states,
            before_compact_stats.transitions,
            before_compact_stats.interned_ranges,
            before_compact_range_counts.token_ranges,
            after_compact_stats.states,
            after_compact_stats.transitions,
            after_compact_stats.interned_ranges,
            after_compact_range_counts.token_ranges,
            before_num_tsids,
            profile_stats.map(|stats| stats.tsids_after).unwrap_or(before_num_tsids as usize),
            before_num_tokens,
            profile_stats.map(|stats| stats.tokens_after).unwrap_or(before_num_tokens as usize),
            compact_ms,
        );
    }

    LocalIdMapTerminalDwa {
        id_map,
        dwa,
        profile: TerminalDwaPhaseProfile {
            compact_ms,
            global_merge_ms: total_ms,
            ..TerminalDwaPhaseProfile::default()
        },
    }
}

/// Merge a family of depth-one terminal DWAs directly.
///
/// The generic global merge converts the inputs to an NWA, determinizes a
/// large set of one-edge alternatives, then minimizes the result back to two
/// states. For an immediate family the exact union is simply the union of the
/// remapped acceptance weights on each start label.
fn try_merge_immediate_terminal_dwas(
    inputs: &[LocalIdMapTerminalDwa],
    num_tokenizer_states: usize,
    max_token_id: u32,
    token_domains_proven_disjoint: bool,
) -> Option<LocalIdMapTerminalDwa> {
    if inputs.len() < 2
        || inputs
            .iter()
            .any(|input| immediate_dwa_accepting_edges(&input.dwa).is_none())
    {
        return None;
    }

    let total_started_at = Instant::now();
    let id_map_started_at = Instant::now();
    let id_map_refs: Vec<&InternalIdMap> = inputs.iter().map(|input| &input.id_map).collect();
    // These inputs are the compiler's disjoint vocabulary partitions. Preserve
    // that proof instead of scanning every original-token map again, and retain
    // the local→global TSID maps produced while constructing the global classes.
    let (global_id_map, token_maps, local_to_global_tsid_maps) =
        build_unified_global_id_map_fast(
            &id_map_refs,
            num_tokenizer_states,
            max_token_id,
            None,
            token_domains_proven_disjoint,
        );
    let id_map_ms = id_map_started_at.elapsed().as_secs_f64() * 1000.0;

    let remap_started_at = Instant::now();
    let mut weights_by_label = BTreeMap::<i32, Vec<Weight>>::new();
    for (input_index, input) in inputs.iter().enumerate() {
        let accepting = immediate_dwa_accepting_edges(&input.dwa)?;
        let tsid_map = &local_to_global_tsid_maps[input_index];
        let tsid_runs = build_local_to_global_tsid_runs(tsid_map);
        let direct_token_map_storage;
        let token_map = match &token_maps {
            FastGlobalTokenMaps::Direct(maps) => {
                direct_token_map_storage =
                    build_direct_local_to_global_token_map(&maps[input_index]);
                direct_token_map_storage.as_slice()
            }
            FastGlobalTokenMaps::General(maps) => maps[input_index].as_slice(),
        };
        let token_remap = TokenRemap::Explicit(token_map);
        let mut weight_cache = RemapCache::<Weight>::default();
        let mut token_cache = RemapCache::<Arc<RangeSetBlaze<u32>>>::default();
        for (label, weight) in accepting {
            let weight = remap_weight_cached_with_tsid_runs(
                &weight,
                tsid_map,
                &tsid_runs,
                &token_remap,
                global_id_map.num_tsids() as usize,
                &mut weight_cache,
                &mut token_cache,
            );
            if !weight.is_empty() {
                weights_by_label.entry(label).or_default().push(weight);
            }
        }
    }
    let remap_ms = remap_started_at.elapsed().as_secs_f64() * 1000.0;

    let union_started_at = Instant::now();
    let mut dwa = DWA::new(global_id_map.num_tsids(), global_id_map.max_internal_token_id());
    let final_state = dwa.add_state();
    dwa.set_final_weight(final_state, Weight::all());
    for (label, weights) in weights_by_label {
        let weight = Weight::union_all(weights.iter());
        if !weight.is_empty() {
            dwa.add_transition(dwa.start_state(), label, final_state, weight);
        }
    }
    let union_ms = union_started_at.elapsed().as_secs_f64() * 1000.0;
    let total_ms = total_started_at.elapsed().as_secs_f64() * 1000.0;

    if compile_profile_enabled() {
        eprintln!(
            "[glrmask/profile][immediate_terminal_dwa_merge] inputs={} id_map_ms={:.3} remap_ms={:.3} union_ms={:.3} states={} transitions={} tsids={} tokens={} total_ms={:.3}",
            inputs.len(),
            id_map_ms,
            remap_ms,
            union_ms,
            dwa.num_states(),
            dwa.num_transitions(),
            global_id_map.num_tsids(),
            global_id_map.num_internal_tokens(),
            total_ms,
        );
    }

    Some(LocalIdMapTerminalDwa {
        id_map: global_id_map,
        dwa,
        profile: TerminalDwaPhaseProfile {
            global_merge_ms: total_ms,
            ..TerminalDwaPhaseProfile::default()
        },
    })
}

/// Fast exact global map construction for an already-proven disjoint token
/// partition. The input order and original-state order are deterministic, so
/// first-seen class IDs are stable for this merge and avoid a costly sort and
/// relabel pass used only for cross-order canonicalization.
fn build_unified_global_id_map_fast(
    inputs: &[&InternalIdMap],
    num_tokenizer_states: usize,
    max_token_id: u32,
    primary_source: Option<usize>,
    require_direct_token_maps: bool,
) -> (InternalIdMap, FastGlobalTokenMaps, Vec<Vec<Vec<u32>>>) {
    let mut composite_to_class = FxHashMap::<SmallVec<[u32; 16]>, u32>::default();
    let mut state_o2i = vec![0u32; num_tokenizer_states];
    let mut state_i2o: Vec<Vec<u32>> = Vec::new();
    let mut state_reps: Vec<u32> = Vec::new();
    let mut class_keys = Vec::<SmallVec<[u32; 16]>>::new();

    for state in 0..num_tokenizer_states {
        let mut composite = SmallVec::<[u32; 16]>::with_capacity(inputs.len());
        composite.extend(
            inputs
                .iter()
                .map(|input| input.tokenizer_states.original_to_internal[state]),
        );
        let next_id = state_i2o.len() as u32;
        let class = if let Some(&existing) = composite_to_class.get(&composite) {
            existing
        } else {
            let class = next_id;
            class_keys.push(composite.clone());
            composite_to_class.insert(composite, class);
            state_i2o.push(Vec::new());
            state_reps.push(state as u32);
            class
        };
        state_o2i[state] = class;
        state_i2o[class as usize].push(state as u32);
    }

    let source_order = token_nwa_tsid_source_order(inputs.len()).or_else(|| {
        primary_source
            .filter(|source| *source < inputs.len())
            .map(|primary| {
                std::iter::once(primary)
                    .chain((0..inputs.len()).filter(move |&source| source != primary))
                    .collect()
            })
    });
    if let Some(source_order) = source_order {
        let mut order: Vec<usize> = (0..class_keys.len()).collect();
        order.sort_unstable_by(|&left, &right| {
            for &source in &source_order {
                let comparison = class_keys[left][source].cmp(&class_keys[right][source]);
                if !comparison.is_eq() {
                    return comparison;
                }
            }
            left.cmp(&right)
        });
        let mut old_to_new = vec![0u32; class_keys.len()];
        for (new_id, &old_id) in order.iter().enumerate() {
            old_to_new[old_id] = new_id as u32;
        }
        for class in &mut state_o2i {
            *class = old_to_new[*class as usize];
        }
        let mut new_keys = Vec::with_capacity(class_keys.len());
        let mut new_i2o = Vec::with_capacity(state_i2o.len());
        let mut new_reps = Vec::with_capacity(state_reps.len());
        for old_id in order {
            new_keys.push(std::mem::take(&mut class_keys[old_id]));
            new_i2o.push(std::mem::take(&mut state_i2o[old_id]));
            new_reps.push(state_reps[old_id]);
        }
        class_keys = new_keys;
        state_i2o = new_i2o;
        state_reps = new_reps;
    }

    let mut local_to_global_tsids: Vec<Vec<Vec<u32>>> = inputs
        .iter()
        .map(|input| vec![Vec::new(); input.num_tsids() as usize])
        .collect();
    for (global_tsid, composite) in class_keys.iter().enumerate() {
        for (input_index, &local_tsid) in composite.iter().enumerate() {
            local_to_global_tsids[input_index][local_tsid as usize].push(global_tsid as u32);
        }
    }

    let (vocab_tokens, token_maps) = if require_direct_token_maps {
        let (vocab_tokens, direct_maps) =
            build_unified_global_token_id_map_assume_disjoint(inputs, max_token_id);
        (vocab_tokens, FastGlobalTokenMaps::Direct(direct_maps))
    } else {
        let use_two_source_fast_path = std::env::var("GLRMASK_TWO_SOURCE_TOKEN_MERGE")
            .map(|value| {
                let trimmed = value.trim();
                trimmed.is_empty()
                    || (trimmed != "0" && !trimmed.eq_ignore_ascii_case("false"))
            })
            .unwrap_or(true);
        let (vocab_tokens, general_maps) = if use_two_source_fast_path {
            build_unified_global_token_id_map_at_most_two(inputs, max_token_id)
                .unwrap_or_else(|| {
                    build_unified_global_token_id_map_sparse_generic(inputs, max_token_id)
                })
        } else {
            build_unified_global_token_id_map_sparse_generic(inputs, max_token_id)
        };
        (vocab_tokens, FastGlobalTokenMaps::General(general_maps))
    };
    (
        InternalIdMap {
            tokenizer_states: ManyToOneIdMap {
                original_to_internal: state_o2i,
                internal_to_originals: state_i2o,
                representative_original_ids: state_reps,
            },
            vocab_tokens,
            deferred_vocab_singleton_original_ids: None,
        },
        token_maps,
        local_to_global_tsids,
    )
}

/// Exact sparse token-map merge for the compiler's ordinary-L1 plus split-L1
/// layout. Character partitions are disjoint, so an original token appears in
/// at most the ordinary and split artifact for one partition. Keep those two
/// memberships inline instead of allocating one `SmallVec` per overlapping
/// token. If an arbitrary caller supplies a third membership, return `None`
/// and let the fully generic builder handle it.
fn build_unified_global_token_id_map_at_most_two(
    inputs: &[&InternalIdMap],
    max_token_id: u32,
) -> Option<(ManyToOneIdMap, Vec<Vec<Vec<u32>>>)> {
    let use_packed = std::env::var("GLRMASK_PACKED_TWO_SOURCE_TOKEN_MERGE")
        .map(|value| {
            let trimmed = value.trim();
            trimmed.is_empty() || (trimmed != "0" && !trimmed.eq_ignore_ascii_case("false"))
        })
        .unwrap_or(true);
    if use_packed {
        build_unified_global_token_id_map_at_most_two_packed(inputs, max_token_id)
            .or_else(|| build_unified_global_token_id_map_at_most_two_wide(inputs, max_token_id))
    } else {
        build_unified_global_token_id_map_at_most_two_wide(inputs, max_token_id)
    }
}

fn build_unified_global_token_id_map_at_most_two_wide(
    inputs: &[&InternalIdMap],
    max_token_id: u32,
) -> Option<(ManyToOneIdMap, Vec<Vec<Vec<u32>>>)> {
    let token_count = max_token_id as usize + 1;
    let mut first_input = vec![u16::MAX; token_count];
    let mut first_local_class = vec![u32::MAX; token_count];
    let mut second_input = vec![u16::MAX; token_count];
    let mut second_local_class = vec![u32::MAX; token_count];

    for (input_index, input) in inputs.iter().enumerate() {
        if input_index >= u16::MAX as usize {
            return None;
        }
        for (local_class, originals) in input
            .vocab_tokens
            .internal_to_originals
            .iter()
            .enumerate()
        {
            for &token_id in originals {
                let token_index = token_id as usize;
                debug_assert!(token_index < token_count);
                if first_input[token_index] == u16::MAX {
                    first_input[token_index] = input_index as u16;
                    first_local_class[token_index] = local_class as u32;
                } else if second_input[token_index] == u16::MAX {
                    second_input[token_index] = input_index as u16;
                    second_local_class[token_index] = local_class as u32;
                } else {
                    return None;
                }
            }
        }
    }

    let mut unique_counts = inputs
        .iter()
        .map(|input| vec![0usize; input.num_internal_tokens() as usize])
        .collect::<Vec<_>>();
    let mut overlap_group_by_token = vec![u32::MAX; token_count];
    let mut overlap_group_by_membership =
        FxHashMap::<(u64, u64), u32>::default();
    let mut overlap_memberships = Vec::<(u64, u64)>::new();
    let mut overlap_counts = Vec::<usize>::new();

    for token_index in 0..token_count {
        if first_input[token_index] == u16::MAX {
            continue;
        }
        if second_input[token_index] == u16::MAX {
            unique_counts[first_input[token_index] as usize]
                [first_local_class[token_index] as usize] += 1;
            continue;
        }
        let membership = (
            ((first_input[token_index] as u64) << 32)
                | first_local_class[token_index] as u64,
            ((second_input[token_index] as u64) << 32)
                | second_local_class[token_index] as u64,
        );
        let group = if let Some(&group) = overlap_group_by_membership.get(&membership) {
            group
        } else {
            let group = overlap_memberships.len() as u32;
            overlap_group_by_membership.insert(membership, group);
            overlap_memberships.push(membership);
            overlap_counts.push(0);
            group
        };
        overlap_group_by_token[token_index] = group;
        overlap_counts[group as usize] += 1;
    }

    let mut token_i2o = Vec::<Vec<u32>>::new();
    let mut token_reps = Vec::<u32>::new();
    let mut local_to_global = inputs
        .iter()
        .map(|input| vec![Vec::<u32>::new(); input.num_internal_tokens() as usize])
        .collect::<Vec<_>>();
    let mut unique_global_class = inputs
        .iter()
        .map(|input| vec![u32::MAX; input.num_internal_tokens() as usize])
        .collect::<Vec<_>>();

    for input_index in 0..inputs.len() {
        for local_class in 0..unique_counts[input_index].len() {
            let count = unique_counts[input_index][local_class];
            if count == 0 {
                continue;
            }
            let global_class = token_i2o.len() as u32;
            unique_global_class[input_index][local_class] = global_class;
            local_to_global[input_index][local_class].push(global_class);
            token_i2o.push(Vec::with_capacity(count));
            token_reps.push(u32::MAX);
        }
    }

    let overlap_global_base = token_i2o.len() as u32;
    for (group, &(first, second)) in overlap_memberships.iter().enumerate() {
        let global_class = overlap_global_base + group as u32;
        token_i2o.push(Vec::with_capacity(overlap_counts[group]));
        token_reps.push(u32::MAX);
        for packed in [first, second] {
            let input_index = (packed >> 32) as usize;
            let local_class = packed as u32 as usize;
            local_to_global[input_index][local_class].push(global_class);
        }
    }

    let mut token_o2i = vec![u32::MAX; token_count];
    for token_index in 0..token_count {
        if first_input[token_index] == u16::MAX {
            continue;
        }
        let group = overlap_group_by_token[token_index];
        let global_class = if group == u32::MAX {
            unique_global_class[first_input[token_index] as usize]
                [first_local_class[token_index] as usize]
        } else {
            overlap_global_base + group
        };
        debug_assert_ne!(global_class, u32::MAX);
        token_o2i[token_index] = global_class;
        let originals = &mut token_i2o[global_class as usize];
        if originals.is_empty() {
            token_reps[global_class as usize] = token_index as u32;
        }
        originals.push(token_index as u32);
    }

    debug_assert!(token_i2o.iter().all(|originals| !originals.is_empty()));
    Some((
        ManyToOneIdMap {
            original_to_internal: token_o2i,
            internal_to_originals: token_i2o,
            representative_original_ids: token_reps,
        },
        local_to_global,
    ))
}

fn build_unified_global_token_id_map_at_most_two_packed(
    inputs: &[&InternalIdMap],
    max_token_id: u32,
) -> Option<(ManyToOneIdMap, Vec<Vec<Vec<u32>>>)> {
    const INPUT_BITS: u32 = 8;
    const CLASS_BITS: u32 = 32 - INPUT_BITS;
    const CLASS_MASK: u32 = (1u32 << CLASS_BITS) - 1;
    const MAX_INPUT: usize = (1usize << INPUT_BITS) - 1;
    let pack = |input: usize, class: usize| -> Option<u32> {
        (input < MAX_INPUT && class < CLASS_MASK as usize)
            .then_some(((input as u32) << CLASS_BITS) | class as u32)
    };
    let unpack_input = |packed: u32| (packed >> CLASS_BITS) as usize;
    let unpack_class = |packed: u32| (packed & CLASS_MASK) as usize;

    let token_count = max_token_id as usize + 1;
    let mut first_membership = vec![u32::MAX; token_count];
    let mut second_membership = vec![u32::MAX; token_count];

    for (input_index, input) in inputs.iter().enumerate() {
        for (local_class, originals) in input
            .vocab_tokens
            .internal_to_originals
            .iter()
            .enumerate()
        {
            let packed = pack(input_index, local_class)?;
            for &token_id in originals {
                let token_index = token_id as usize;
                debug_assert!(token_index < token_count);
                if first_membership[token_index] == u32::MAX {
                    first_membership[token_index] = packed;
                } else if second_membership[token_index] == u32::MAX {
                    second_membership[token_index] = packed;
                } else {
                    return None;
                }
            }
        }
    }

    let mut unique_counts = inputs
        .iter()
        .map(|input| vec![0usize; input.num_internal_tokens() as usize])
        .collect::<Vec<_>>();
    let mut overlap_group_by_membership =
        FxHashMap::<(u32, u32), u32>::default();
    let mut overlap_memberships = Vec::<(u32, u32)>::new();
    let mut overlap_counts = Vec::<usize>::new();

    for token_index in 0..token_count {
        let first = first_membership[token_index];
        if first == u32::MAX {
            continue;
        }
        let second = second_membership[token_index];
        if second == u32::MAX {
            unique_counts[unpack_input(first)][unpack_class(first)] += 1;
            continue;
        }
        let membership = (first, second);
        let group = if let Some(&group) = overlap_group_by_membership.get(&membership) {
            group
        } else {
            let group = overlap_memberships.len() as u32;
            overlap_group_by_membership.insert(membership, group);
            overlap_memberships.push(membership);
            overlap_counts.push(0);
            group
        };
        overlap_counts[group as usize] += 1;
    }

    let mut token_i2o = Vec::<Vec<u32>>::new();
    let mut token_reps = Vec::<u32>::new();
    let mut local_to_global = inputs
        .iter()
        .map(|input| vec![Vec::<u32>::new(); input.num_internal_tokens() as usize])
        .collect::<Vec<_>>();
    let mut unique_global_class = inputs
        .iter()
        .map(|input| vec![u32::MAX; input.num_internal_tokens() as usize])
        .collect::<Vec<_>>();

    for input_index in 0..inputs.len() {
        for local_class in 0..unique_counts[input_index].len() {
            let count = unique_counts[input_index][local_class];
            if count == 0 {
                continue;
            }
            let global_class = token_i2o.len() as u32;
            unique_global_class[input_index][local_class] = global_class;
            local_to_global[input_index][local_class].push(global_class);
            token_i2o.push(Vec::with_capacity(count));
            token_reps.push(u32::MAX);
        }
    }

    let overlap_global_base = token_i2o.len() as u32;
    for (group, &(first, second)) in overlap_memberships.iter().enumerate() {
        let global_class = overlap_global_base + group as u32;
        token_i2o.push(Vec::with_capacity(overlap_counts[group]));
        token_reps.push(u32::MAX);
        for packed in [first, second] {
            local_to_global[unpack_input(packed)][unpack_class(packed)].push(global_class);
        }
    }

    let mut token_o2i = vec![u32::MAX; token_count];
    for token_index in 0..token_count {
        let first = first_membership[token_index];
        if first == u32::MAX {
            continue;
        }
        let second = second_membership[token_index];
        let global_class = if second == u32::MAX {
            unique_global_class[unpack_input(first)][unpack_class(first)]
        } else {
            overlap_global_base + overlap_group_by_membership[&(first, second)]
        };
        debug_assert_ne!(global_class, u32::MAX);
        token_o2i[token_index] = global_class;
        let originals = &mut token_i2o[global_class as usize];
        if originals.is_empty() {
            token_reps[global_class as usize] = token_index as u32;
        }
        originals.push(token_index as u32);
    }

    debug_assert!(token_i2o.iter().all(|originals| !originals.is_empty()));
    Some((
        ManyToOneIdMap {
            original_to_internal: token_o2i,
            internal_to_originals: token_i2o,
            representative_original_ids: token_reps,
        },
        local_to_global,
    ))
}

fn build_unified_global_id_map_disjoint_fast(
    inputs: &[&InternalIdMap],
    num_tokenizer_states: usize,
    max_token_id: u32,
    primary_source: Option<usize>,
) -> (InternalIdMap, Vec<Vec<u32>>, Vec<Vec<Vec<u32>>>) {
    let (id_map, token_maps, tsid_maps) = build_unified_global_id_map_fast(
        inputs,
        num_tokenizer_states,
        max_token_id,
        primary_source,
        true,
    );
    let FastGlobalTokenMaps::Direct(direct_maps) = token_maps else {
        panic!("disjoint token domains must produce direct token maps");
    };
    (id_map, direct_maps, tsid_maps)
}

fn build_unified_global_token_id_map_assume_disjoint(
    inputs: &[&InternalIdMap],
    max_token_id: u32,
) -> (ManyToOneIdMap, Vec<Vec<u32>>) {
    let mut token_o2i = vec![u32::MAX; max_token_id as usize + 1];
    let token_class_count: usize = inputs
        .iter()
        .map(|input| input.vocab_tokens.internal_to_originals.len())
        .sum();
    let mut token_i2o = Vec::with_capacity(token_class_count);
    let mut token_reps = Vec::with_capacity(token_class_count);
    let mut direct_local_to_global_token_maps = Vec::with_capacity(inputs.len());

    for input in inputs {
        let mut local_to_global = vec![u32::MAX; input.num_internal_tokens() as usize];
        for (local_class, originals) in input.vocab_tokens.internal_to_originals.iter().enumerate() {
            if originals.is_empty() {
                continue;
            }
            let global_class = token_i2o.len() as u32;
            local_to_global[local_class] = global_class;
            token_reps.push(input.vocab_tokens.representative_original_ids[local_class]);
            token_i2o.push(originals.clone());
            for &token_id in originals {
                token_o2i[token_id as usize] = global_class;
            }
        }
        direct_local_to_global_token_maps.push(local_to_global);
    }

    (
        ManyToOneIdMap {
            original_to_internal: token_o2i,
            internal_to_originals: token_i2o,
            representative_original_ids: token_reps,
        },
        direct_local_to_global_token_maps,
    )
}

/// Fast exact global map construction for an already-proven disjoint token
/// partition. The input order and original-state order are deterministic, so
/// first-seen class IDs are stable for this merge and avoid a costly sort and
/// relabel pass used only for cross-order canonicalization.
fn build_unified_global_id_map(
    inputs: &[&InternalIdMap],
    num_tokenizer_states: usize,
    max_token_id: u32,
) -> (InternalIdMap, Option<Vec<Vec<u32>>>) {
    let mut composite_to_class = FxHashMap::<CompositeClassKey, u32>::default();
    let mut state_o2i = vec![0u32; num_tokenizer_states];
    let mut state_i2o: Vec<Vec<u32>> = Vec::new();
    let mut state_reps: Vec<u32> = Vec::new();

    for state in 0..num_tokenizer_states {
        let composite: CompositeClassKey = inputs
            .iter()
            .map(|input| input.tokenizer_states.original_to_internal[state])
            .collect();
        let next_id = state_i2o.len() as u32;
        let class = *composite_to_class.entry(composite).or_insert_with(|| {
            state_i2o.push(Vec::new());
            state_reps.push(state as u32);
            next_id
        });
        state_o2i[state] = class;
        state_i2o[class as usize].push(state as u32);
    }

    reorder_classes(composite_to_class, &mut state_o2i, &mut state_i2o, &mut state_reps);

    let (vocab_tokens, direct_local_to_global_token_maps) =
        build_unified_global_token_id_map_disjoint(inputs, max_token_id)
            .unwrap_or_else(|| (build_unified_global_token_id_map_generic(inputs, max_token_id), None));
    (
        InternalIdMap {
            tokenizer_states: ManyToOneIdMap {
                original_to_internal: state_o2i,
                internal_to_originals: state_i2o,
                representative_original_ids: state_reps,
            },
            vocab_tokens,
            deferred_vocab_singleton_original_ids: None,
        },
        direct_local_to_global_token_maps,
    )
}

/// Build only the unified exact ID map used by terminal-DWA merging, without
/// remapping or constructing any automata.
pub fn merge_internal_id_maps_only(
    inputs: &[&InternalIdMap],
    num_tokenizer_states: usize,
    max_token_id: u32,
) -> InternalIdMap {
    assert!(!inputs.is_empty(), "merge_internal_id_maps_only called with empty inputs");
    if inputs.len() == 1 {
        return inputs[0].clone();
    }
    build_unified_global_id_map(inputs, num_tokenizer_states, max_token_id).0
}

fn build_unified_global_token_id_map_sparse_generic(
    inputs: &[&InternalIdMap],
    max_token_id: u32,
) -> (ManyToOneIdMap, Vec<Vec<Vec<u32>>>) {
    type MembershipKey = SmallVec<[u64; 4]>;

    let token_count = max_token_id as usize + 1;
    let mut first_input = vec![u16::MAX; token_count];
    let mut first_local_class = vec![u32::MAX; token_count];
    let mut overlap_index = vec![u32::MAX; token_count];
    let mut overlap_memberships = Vec::<MembershipKey>::new();

    for (input_index, input) in inputs.iter().enumerate() {
        assert!(input_index < u16::MAX as usize, "too many immediate merge inputs");
        for (local_class, originals) in input
            .vocab_tokens
            .internal_to_originals
            .iter()
            .enumerate()
        {
            for &token_id in originals {
                let token_index = token_id as usize;
                debug_assert!(token_index < token_count);
                if first_input[token_index] == u16::MAX {
                    first_input[token_index] = input_index as u16;
                    first_local_class[token_index] = local_class as u32;
                    continue;
                }

                let membership_index = if overlap_index[token_index] == u32::MAX {
                    let membership_index = overlap_memberships.len() as u32;
                    let mut membership = MembershipKey::new();
                    membership.push(
                        ((first_input[token_index] as u64) << 32)
                            | first_local_class[token_index] as u64,
                    );
                    overlap_memberships.push(membership);
                    overlap_index[token_index] = membership_index;
                    membership_index
                } else {
                    overlap_index[token_index]
                };
                overlap_memberships[membership_index as usize]
                    .push(((input_index as u64) << 32) | local_class as u64);
            }
        }
    }

    let mut unique_counts = inputs
        .iter()
        .map(|input| vec![0usize; input.num_internal_tokens() as usize])
        .collect::<Vec<_>>();
    for token_index in 0..token_count {
        if first_input[token_index] != u16::MAX && overlap_index[token_index] == u32::MAX {
            unique_counts[first_input[token_index] as usize]
                [first_local_class[token_index] as usize] += 1;
        }
    }

    let mut token_i2o = Vec::<Vec<u32>>::new();
    let mut token_reps = Vec::<u32>::new();
    let mut local_to_global = inputs
        .iter()
        .map(|input| vec![Vec::<u32>::new(); input.num_internal_tokens() as usize])
        .collect::<Vec<_>>();
    let mut unique_global_class = inputs
        .iter()
        .map(|input| vec![u32::MAX; input.num_internal_tokens() as usize])
        .collect::<Vec<_>>();

    for input_index in 0..inputs.len() {
        for local_class in 0..unique_counts[input_index].len() {
            if unique_counts[input_index][local_class] == 0 {
                continue;
            }
            let global_class = token_i2o.len() as u32;
            unique_global_class[input_index][local_class] = global_class;
            local_to_global[input_index][local_class].push(global_class);
            token_i2o.push(Vec::with_capacity(unique_counts[input_index][local_class]));
            token_reps.push(u32::MAX);
        }
    }

    let mut overlap_class_by_membership = FxHashMap::<MembershipKey, u32>::default();
    let mut token_o2i = vec![u32::MAX; token_count];
    for token_index in 0..token_count {
        if first_input[token_index] == u16::MAX {
            continue;
        }
        let global_class = if overlap_index[token_index] == u32::MAX {
            unique_global_class[first_input[token_index] as usize]
                [first_local_class[token_index] as usize]
        } else {
            let membership = &overlap_memberships[overlap_index[token_index] as usize];
            if let Some(&existing) = overlap_class_by_membership.get(membership) {
                existing
            } else {
                let global_class = token_i2o.len() as u32;
                overlap_class_by_membership.insert(membership.clone(), global_class);
                token_i2o.push(Vec::new());
                token_reps.push(u32::MAX);
                for &packed in membership {
                    let input_index = (packed >> 32) as usize;
                    let local_class = packed as u32 as usize;
                    local_to_global[input_index][local_class].push(global_class);
                }
                global_class
            }
        };
        debug_assert_ne!(global_class, u32::MAX);
        token_o2i[token_index] = global_class;
        let originals = &mut token_i2o[global_class as usize];
        if originals.is_empty() {
            token_reps[global_class as usize] = token_index as u32;
        }
        originals.push(token_index as u32);
    }

    debug_assert!(token_i2o.iter().all(|originals| !originals.is_empty()));
    (
        ManyToOneIdMap {
            original_to_internal: token_o2i,
            internal_to_originals: token_i2o,
            representative_original_ids: token_reps,
        },
        local_to_global,
    )
}

fn build_unified_global_token_id_map_generic(
    inputs: &[&InternalIdMap],
    max_token_id: u32,
) -> ManyToOneIdMap {
    let mut token_composite_to_class = FxHashMap::<CompositeClassKey, u32>::default();
    let mut token_o2i = vec![u32::MAX; max_token_id as usize + 1];
    let mut token_i2o: Vec<Vec<u32>> = Vec::new();
    let mut token_reps: Vec<u32> = Vec::new();

    for token_id in 0..=max_token_id {
        let composite: CompositeClassKey = inputs
            .iter()
            .map(|input| {
                input
                    .vocab_tokens
                    .original_to_internal
                    .get(token_id as usize)
                    .copied()
                    .unwrap_or(u32::MAX)
            })
            .collect();
        if composite.iter().all(|&value| value == u32::MAX) {
            continue;
        }

        let next_id = token_i2o.len() as u32;
        let class = *token_composite_to_class
            .entry(composite)
            .or_insert_with(|| {
                token_i2o.push(Vec::new());
                token_reps.push(token_id);
                next_id
            });
        token_o2i[token_id as usize] = class;
        token_i2o[class as usize].push(token_id);
    }

    reorder_classes_with_sentinel(
        token_composite_to_class,
        &mut token_o2i,
        &mut token_i2o,
        &mut token_reps,
        u32::MAX,
    );

    ManyToOneIdMap {
        original_to_internal: token_o2i,
        internal_to_originals: token_i2o,
        representative_original_ids: token_reps,
    }
}

fn build_unified_global_token_id_map_disjoint(
    inputs: &[&InternalIdMap],
    max_token_id: u32,
) -> Option<(ManyToOneIdMap, Option<Vec<Vec<u32>>>)> {
    let mut owner_by_token = vec![usize::MAX; max_token_id as usize + 1];

    for (input_idx, input) in inputs.iter().enumerate() {
        for (token_id, &local_class) in input.vocab_tokens.original_to_internal.iter().enumerate() {
            if local_class == u32::MAX {
                continue;
            }
            let owner = &mut owner_by_token[token_id];
            if *owner != usize::MAX && *owner != input_idx {
                return None;
            }
            *owner = input_idx;
        }
    }

    let mut token_o2i = vec![u32::MAX; max_token_id as usize + 1];
    let mut token_i2o: Vec<Vec<u32>> = Vec::new();
    let mut token_reps: Vec<u32> = Vec::new();
    let mut direct_local_to_global_token_maps = Vec::with_capacity(inputs.len());

    for input in inputs {
        let mut local_to_global = vec![u32::MAX; input.num_internal_tokens() as usize];

        for (local_class, originals) in input.vocab_tokens.internal_to_originals.iter().enumerate() {
            if originals.is_empty() {
                continue;
            }

            let global_class = token_i2o.len() as u32;
            local_to_global[local_class] = global_class;
            token_reps.push(input.vocab_tokens.representative_original_ids[local_class]);
            token_i2o.push(originals.clone());
            for &token_id in originals {
                token_o2i[token_id as usize] = global_class;
            }
        }

        direct_local_to_global_token_maps.push(local_to_global);
    }

    Some((
        ManyToOneIdMap {
            original_to_internal: token_o2i,
            internal_to_originals: token_i2o,
            representative_original_ids: token_reps,
        },
        Some(direct_local_to_global_token_maps),
    ))
}

fn build_direct_local_to_global_token_map(local_to_global: &[u32]) -> Vec<Vec<u32>> {
    local_to_global
        .iter()
        .map(|&global_class| {
            if global_class == u32::MAX {
                Vec::new()
            } else {
                vec![global_class]
            }
        })
        .collect()
}

fn reorder_classes(
    composite_to_class: FxHashMap<CompositeClassKey, u32>,
    o2i: &mut [u32],
    i2o: &mut Vec<Vec<u32>>,
    reps: &mut Vec<u32>,
) {
    let num_classes = i2o.len();
    if num_classes <= 1 {
        return;
    }

    let mut sorted: Vec<(CompositeClassKey, u32)> = composite_to_class.into_iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));

    let mut old_to_new = vec![0u32; num_classes];
    for (new_id, (_, old_id)) in sorted.iter().enumerate() {
        old_to_new[*old_id as usize] = new_id as u32;
    }

    for val in o2i.iter_mut() {
        *val = old_to_new[*val as usize];
    }

    let mut new_i2o = vec![Vec::new(); num_classes];
    let mut new_reps = vec![0u32; num_classes];
    for (new_id, (_, old_id)) in sorted.iter().enumerate() {
        new_i2o[new_id] = std::mem::take(&mut i2o[*old_id as usize]);
        new_reps[new_id] = reps[*old_id as usize];
    }
    *i2o = new_i2o;
    *reps = new_reps;
}

fn reorder_classes_with_sentinel(
    composite_to_class: FxHashMap<CompositeClassKey, u32>,
    o2i: &mut [u32],
    i2o: &mut Vec<Vec<u32>>,
    reps: &mut Vec<u32>,
    sentinel: u32,
) {
    let num_classes = i2o.len();
    if num_classes <= 1 {
        return;
    }

    let mut sorted: Vec<(CompositeClassKey, u32)> = composite_to_class.into_iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));

    let mut old_to_new = vec![0u32; num_classes];
    for (new_id, (_, old_id)) in sorted.iter().enumerate() {
        old_to_new[*old_id as usize] = new_id as u32;
    }

    for val in o2i.iter_mut() {
        if *val != sentinel {
            *val = old_to_new[*val as usize];
        }
    }

    let mut new_i2o = vec![Vec::new(); num_classes];
    let mut new_reps = vec![0u32; num_classes];
    for (new_id, (_, old_id)) in sorted.iter().enumerate() {
        new_i2o[new_id] = std::mem::take(&mut i2o[*old_id as usize]);
        new_reps[new_id] = reps[*old_id as usize];
    }
    *i2o = new_i2o;
    *reps = new_reps;
}

/// Map local TSIDs to the global composite TSIDs.
///
/// `build_unified_global_id_map` constructs each global TSID from the tuple of
/// all input-local TSIDs. Therefore any representative original tokenizer
/// state of a global class identifies the unique local class that owns it.
/// Iterating global representatives emits each destination once, in ascending
/// global-ID order, avoiding the old per-input `BTreeSet` aggregation.
fn build_local_to_global_tsid_map(
    local_id_map: &InternalIdMap,
    global_id_map: &InternalIdMap,
) -> Vec<Vec<u32>> {
    let mut local_to_global = vec![Vec::new(); local_id_map.num_tsids() as usize];
    for (global_tsid, &representative_state) in global_id_map
        .tokenizer_states
        .representative_original_ids
        .iter()
        .enumerate()
    {
        let local_tsid = local_id_map.tokenizer_states.original_to_internal
            [representative_state as usize];
        if local_tsid == u32::MAX {
            #[cfg(debug_assertions)]
            for &original_state in
                &global_id_map.tokenizer_states.internal_to_originals[global_tsid]
            {
                debug_assert_eq!(
                    local_id_map.tokenizer_states.original_to_internal
                        [original_state as usize],
                    u32::MAX,
                    "a global composite TSID must refine the local unmapped sentinel",
                );
            }
            continue;
        }
        let local_tsid = local_tsid as usize;
        local_to_global[local_tsid].push(global_tsid as u32);

        #[cfg(debug_assertions)]
        for &original_state in &global_id_map.tokenizer_states.internal_to_originals[global_tsid] {
            debug_assert_eq!(
                local_id_map.tokenizer_states.original_to_internal[original_state as usize],
                local_tsid as u32,
                "a global composite TSID must refine every local TSID map",
            );
        }
    }
    local_to_global
}

/// Map local internal token classes to global internal token classes.
fn build_local_to_global_token_map(
    local_id_map: &InternalIdMap,
    global_id_map: &InternalIdMap,
) -> Vec<Vec<u32>> {
    let num_local = local_id_map.num_internal_tokens() as usize;
    let mut local_to_global = vec![BTreeSet::new(); num_local];

    for (orig, &local_class) in local_id_map
        .vocab_tokens
        .original_to_internal
        .iter()
        .enumerate()
    {
        if local_class == u32::MAX {
            continue;
        }
        let global_class = global_id_map
            .vocab_tokens
            .original_to_internal
            .get(orig)
            .copied()
            .unwrap_or(u32::MAX);
        if global_class == u32::MAX {
            continue;
        }
        local_to_global[local_class as usize].insert(global_class);
    }

    local_to_global
        .into_iter()
        .map(|s| s.into_iter().collect())
        .collect()
}

/// Remap all weights in an NWA from local TSID/token space to global space.
fn remap_nwa_with_maps(
    nwa: &mut NWA,
    local_to_global_tsids: &[Vec<u32>],
    local_to_global_tokens: &[Vec<u32>],
    global_tsid_count: usize,
) {
    let mut weight_cache = RemapCache::<Weight>::default();
    let mut token_cache = RemapCache::<Arc<RangeSetBlaze<u32>>>::default();

    for state in  nwa.states_mut() {
        if let Some(final_weight) = state.final_weight.as_mut() {
            *final_weight = remap_weight_cached(
                final_weight,
                local_to_global_tsids,
                local_to_global_tokens,
                global_tsid_count,
                &mut weight_cache,
                &mut token_cache,
            );
            if final_weight.is_empty() {
                state.final_weight = None;
            }
        }

        for targets in state.transitions.values_mut() {
            for (_, weight) in targets.iter_mut() {
                *weight = remap_weight_cached(
                    weight,
                    local_to_global_tsids,
                    local_to_global_tokens,
                    global_tsid_count,
                    &mut weight_cache,
                    &mut token_cache,
                );
            }
            targets.retain(|(_, weight)| !weight.is_empty());
        }
        state.transitions.retain(|_, targets| !targets.is_empty());

        for (_, weight) in state.epsilons.iter_mut() {
            *weight = remap_weight_cached(
                weight,
                local_to_global_tsids,
                local_to_global_tokens,
                global_tsid_count,
                &mut weight_cache,
                &mut token_cache,
            );
        }
        state.epsilons.retain(|(_, weight)| !weight.is_empty());
    }

}

fn remap_weight_cached(
    weight: &Weight,
    local_to_global_tsids: &[Vec<u32>],
    local_to_global_tokens: &[Vec<u32>],
    global_tsid_count: usize,
    cache: &mut RemapCache<Weight>,
    token_cache: &mut RemapCache<Arc<RangeSetBlaze<u32>>>,
) -> Weight {
    remap_weight_cached_with_token_remap(
        weight,
        local_to_global_tsids,
        &TokenRemap::Explicit(local_to_global_tokens),
        global_tsid_count,
        cache,
        token_cache,
    )
}

fn remap_weight_cached_with_tsid_runs(
    weight: &Weight,
    local_to_global_tsids: &[Vec<u32>],
    local_to_global_tsid_runs: &[SmallVec<[(u32, u32); 1]>],
    token_remap: &TokenRemap<'_>,
    global_tsid_count: usize,
    cache: &mut RemapCache<Weight>,
    token_cache: &mut RemapCache<Arc<RangeSetBlaze<u32>>>,
) -> Weight {
    let ptr = weight.ptr_key();
    if let Some(cached) = cache.get(&ptr) {
        return cached.clone();
    }
    let remapped = remap_weight_with_tsid_runs(
        weight,
        local_to_global_tsids,
        local_to_global_tsid_runs,
        token_remap,
        global_tsid_count,
        token_cache,
    );
    cache.insert(ptr, remapped.clone());
    remapped
}

fn remap_weight_with_tsid_runs(
    weight: &Weight,
    local_to_global_tsids: &[Vec<u32>],
    local_to_global_tsid_runs: &[SmallVec<[(u32, u32); 1]>],
    token_remap: &TokenRemap<'_>,
    global_tsid_count: usize,
    token_cache: &mut RemapCache<Arc<RangeSetBlaze<u32>>>,
) -> Weight {
    if weight.is_empty() {
        return weight.clone();
    }
    if weight.is_full() {
        let all_global_tokens = token_remap.all_global_tokens();
        let Some(last_global_tsid) = global_tsid_count.checked_sub(1).map(|count| count as u32) else {
            return Weight::empty();
        };
        return (!all_global_tokens.is_empty())
            .then(|| Weight::from_uniform(0..=last_global_tsid, all_global_tokens))
            .unwrap_or_else(Weight::empty);
    }

    let mut intervals = SmallVec::<[(u32, u32, Arc<RangeSetBlaze<u32>>); 16]>::new();
    for (start, end, tokens) in weight.range_entries() {
        let token_key = Arc::as_ptr(&tokens) as usize;
        let mapped_tokens = token_remap.map_tokens(token_key, tokens.as_ref(), token_cache);
        if mapped_tokens.is_empty() {
            continue;
        }
        for local_tsid in start..=end {
            let Some(runs) = local_to_global_tsid_runs.get(local_tsid as usize) else {
                continue;
            };
            for &(global_start, global_end) in runs {
                intervals.push((global_start, global_end, Arc::clone(&mapped_tokens)));
            }
        }
    }
    if intervals.is_empty() {
        return Weight::empty();
    }
    intervals.sort_unstable_by_key(|(start, _, _)| *start);

    // A global refinement class has one local class for this source, so these
    // intervals are disjoint. Keep the old generic path as an exact fallback
    // if a caller ever supplies a non-partition map.
    for pair in intervals.windows(2) {
        if pair[0].1 >= pair[1].0 {
            return remap_weight_with_token_remap(
                weight,
                local_to_global_tsids,
                token_remap,
                global_tsid_count,
                token_cache,
            );
        }
    }

    use crate::ds::weight::finalize_weight_map;
    use range_set_blaze::RangeMapBlaze;
    let mut map = RangeMapBlaze::<u32, Arc<RangeSetBlaze<u32>>>::new();
    let mut run_start = intervals[0].0;
    let mut run_end = intervals[0].1;
    let mut run_tokens = Arc::clone(&intervals[0].2);
    for (start, end, tokens) in intervals.into_iter().skip(1) {
        if start == run_end.saturating_add(1)
            && (Arc::ptr_eq(&run_tokens, &tokens) || run_tokens.as_ref() == tokens.as_ref())
        {
            run_end = end;
        } else {
            map.extend_simple(std::iter::once((run_start..=run_end, run_tokens)));
            run_start = start;
            run_end = end;
            run_tokens = tokens;
        }
    }
    map.extend_simple(std::iter::once((run_start..=run_end, run_tokens)));
    finalize_weight_map(map)
}

fn remap_weight_cached_with_token_remap(
    weight: &Weight,
    local_to_global_tsids: &[Vec<u32>],
    token_remap: &TokenRemap<'_>,
    global_tsid_count: usize,
    cache: &mut RemapCache<Weight>,
    token_cache: &mut RemapCache<Arc<RangeSetBlaze<u32>>>,
) -> Weight {
    let ptr = weight.ptr_key();
    if let Some(cached) = cache.get(&ptr) {
        return cached.clone();
    }
    let remapped = remap_weight_with_token_remap(
        weight,
        local_to_global_tsids,
        token_remap,
        global_tsid_count,
        token_cache,
    );
    cache.insert(ptr, remapped.clone());
    remapped
}

fn remap_weight_with_token_remap(
    weight: &Weight,
    local_to_global_tsids: &[Vec<u32>],
    token_remap: &TokenRemap<'_>,
    global_tsid_count: usize,
    token_cache: &mut RemapCache<Arc<RangeSetBlaze<u32>>>,
) -> Weight {
    if weight.is_empty() {
        return weight.clone();
    }

    if weight.is_full() {
        let all_global_tokens = token_remap.all_global_tokens();
        let Some(last_global_tsid) = global_tsid_count.checked_sub(1).map(|count| count as u32) else {
            return Weight::empty();
        };
        if all_global_tokens.is_empty() {
            return Weight::empty();
        }
        // Every global TSID contains exactly one local TSID for this source.
        // Hence local `all` covers the complete global TSID domain even when
        // individual local classes are interleaved in that global ordering.
        return Weight::from_uniform(0..=last_global_tsid, all_global_tokens);
    }

    let Some(entries) = weight.compact_entries() else {
        return weight.clone();
    };
    use crate::ds::weight::{finalize_weight_map, shared_rangeset};

    let mut tokens_by_global_tsid = Vec::<(u32, Arc<RangeSetBlaze<u32>>)>::new();
    for (start, end, tokens) in entries {
        let token_key = Arc::as_ptr(&tokens) as usize;
        let mapped_tokens = token_remap.map_tokens(token_key, tokens.as_ref(), token_cache);

        for local_tsid in start..=end {
            let Some(global_tsids) = local_to_global_tsids.get(local_tsid as usize) else {
                continue;
            };
            for &global_tsid in global_tsids {
                if (global_tsid as usize) < global_tsid_count {
                    tokens_by_global_tsid.push((global_tsid, Arc::clone(&mapped_tokens)));
                }
            }
        }
    }

    if tokens_by_global_tsid.is_empty() {
        return Weight::empty();
    }
    tokens_by_global_tsid.sort_unstable_by_key(|(global_tsid, _)| *global_tsid);

    let mut merged_by_global_tsid = Vec::<(u32, Arc<RangeSetBlaze<u32>>)>::new();
    for (global_tsid, tokens) in tokens_by_global_tsid {
        if let Some((last_tsid, last_tokens)) = merged_by_global_tsid.last_mut() {
            if *last_tsid == global_tsid {
                if !Arc::ptr_eq(last_tokens, &tokens) && last_tokens.as_ref() != tokens.as_ref() {
                    *last_tokens = shared_rangeset(last_tokens.as_ref() | tokens.as_ref());
                }
                continue;
            }
        }
        merged_by_global_tsid.push((global_tsid, tokens));
    }

    use range_set_blaze::RangeMapBlaze;
    let mut map = RangeMapBlaze::<u32, Arc<RangeSetBlaze<u32>>>::new();
    let mut run_start: Option<u32> = None;
    let mut run_end = 0u32;
    let mut run_tokens: Option<Arc<RangeSetBlaze<u32>>> = None;
    for (global_tsid, tokens) in merged_by_global_tsid {
        if let Some(current) = &run_tokens {
            if global_tsid == run_end.wrapping_add(1)
                && (Arc::ptr_eq(current, &tokens) || current.as_ref() == tokens.as_ref())
            {
                run_end = global_tsid;
                continue;
            }
            map.extend_simple(std::iter::once((
                run_start.expect("run start must exist")..=run_end,
                Arc::clone(current),
            )));
        }
        run_start = Some(global_tsid);
        run_end = global_tsid;
        run_tokens = Some(tokens);
    }
    if let Some(tokens) = run_tokens {
        map.extend_simple(std::iter::once((
            run_start.expect("run start must exist")..=run_end,
            tokens,
        )));
    }
    finalize_weight_map(map)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn singleton_weight(token: u32) -> Weight {
        Weight::from_per_tsid_token_sets(std::iter::once((
            0,
            RangeSetBlaze::from_iter(std::iter::once(token..=token)),
        )))
    }

    fn token_range_weight(start: u32, end: u32) -> Weight {
        Weight::from_per_tsid_token_sets(std::iter::once((
            0,
            RangeSetBlaze::from_iter(std::iter::once(start..=end)),
        )))
    }

    fn generic_union(inputs: &[DWA]) -> DWA {
        let mut nwa = NWA::new(1, 2);
        let mut body = nwa.body();
        for input in inputs {
            let branch = input.to_nwa();
            body = nwa.union_in_place(&branch, &body);
        }
        nwa.set_start_states(body.start_states);
        determinize(&nwa).expect("test NWA should determinize")
    }

    #[test]
    fn direct_two_parser_union_keeps_correlated_residuals_and_singleton_rows() {
        let mut left = DWA::new(1, 2);
        let l1 = left.add_state();
        let l2 = left.add_state();
        left.add_transition(0, 7, l1, token_range_weight(0, 1));
        left.add_transition(l1, 8, l2, token_range_weight(0, 2));
        left.set_final_weight(l1, singleton_weight(0));
        left.set_final_weight(l2, singleton_weight(0));
        let mut right = DWA::new(1, 2);
        let r1 = right.add_state();
        let r2 = right.add_state();
        right.add_transition(0, 7, r1, token_range_weight(1, 2));
        right.add_transition(r1, 9, r2, token_range_weight(0, 2));
        right.set_final_weight(0, singleton_weight(1));
        right.set_final_weight(r1, singleton_weight(2));
        right.set_final_weight(r2, singleton_weight(2));

        let direct = union_two_parser_dwas_direct(left.clone(), right.clone());
        let generic = generic_union(&[left.clone(), right.clone()]);
        for word in all_words(&[7, 8, 9, DEFAULT_LABEL], 5) {
            let expected = left.eval_word(&word).union(&right.eval_word(&word));
            assert_eq!(direct.eval_word(&word), expected, "direct word={word:?}");
            assert_eq!(generic.eval_word(&word), expected, "generic word={word:?}");
        }
        assert!(find_difference(&direct, &generic).unwrap().is_none());
    }

    #[test]
    fn direct_two_parser_union_matches_generated_weighted_automata() {
        // All operations are exact finite-set union/intersection. Include
        // start finality, empty/full weights, and the raw fallback label used
        // by the existing generic parser-family union. Its input contract is
        // acyclic: the generic determinizer explicitly rejects cyclic NWAs.
        fn next(seed: &mut u64) -> u64 {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            *seed
        }
        fn weight(seed: &mut u64) -> Weight {
            let bits = next(seed);
            if bits % 11 == 0 { return Weight::all(); }
            Weight::from_per_tsid_token_sets((0..2).map(|tsid| {
                let tokens = (0..3).filter(|token| bits & (1 << (tsid * 3 + token)) != 0);
                (tsid, RangeSetBlaze::from_iter(tokens))
            }))
        }
        fn machine(seed: &mut u64) -> DWA {
            let mut result = DWA::new(2, 2);
            for _ in 1..4 { result.add_state(); }
            for state in 0..4 {
                if next(seed) & 1 != 0 { result.set_final_weight(state, weight(seed)); }
                for label in [0, 7, 8, DEFAULT_LABEL] {
                    if state == 3 { continue; }
                    if next(seed) % 3 == 0 { continue; }
                    let target = state + 1 + (next(seed) % u64::from(3 - state)) as u32;
                    result.add_transition(state, label, target, weight(seed));
                }
            }
            result
        }
        let words = all_words(&[0, 7, 8, DEFAULT_LABEL], 4);
        for case in 1..=96 {
            let mut seed = 0x9e3779b97f4a7c15_u64.wrapping_mul(case);
            let left = machine(&mut seed);
            let right = machine(&mut seed);
            let direct = union_two_parser_dwas_direct(left.clone(), right.clone());
            let mut nwa = NWA::new(2, 2);
            let mut body = nwa.body();
            for source in [&left, &right] {
                body = nwa.union_in_place(&source.to_nwa(), &body);
            }
            nwa.set_start_states(body.start_states);
            let generic = determinize(&nwa).expect("finite test weights must determinize");
            for word in &words {
                let expected = left.eval_word(word).union(&right.eval_word(word));
                assert_eq!(direct.eval_word(word), expected, "case={case} word={word:?}");
                assert_eq!(generic.eval_word(word), expected, "generic case={case} word={word:?}");
            }
            assert!(find_difference(&direct, &generic).unwrap().is_none(), "case={case}");
        }
    }

    #[test]
    fn direct_two_parser_union_preserves_generic_fallback_row_selection() {
        fn fallback_eval(dwa: &DWA, word: &[i32]) -> Weight {
            let mut state = dwa.start_state();
            let mut path = Weight::all();
            for label in word {
                let row = &dwa.states()[state as usize];
                let Some((target, weight)) = row.transitions.get(label)
                    .or_else(|| row.transitions.get(&DEFAULT_LABEL)) else {
                        return Weight::empty();
                    };
                path = path.intersection(weight);
                state = *target;
            }
            dwa.states()[state as usize].final_weight.as_ref()
                .map(|weight| path.intersection(weight)).unwrap_or_else(Weight::empty)
        }
        for padded in [false, true] {
        for boundary in 0..32 {
            let mut left = DWA::new(1, 2);
            let l1 = left.add_state();
            let l2 = left.add_state();
            left.add_transition(0, 7, l1, token_range_weight(0, 1));
            left.add_transition(l1, DEFAULT_LABEL, l2, token_range_weight(0, 1));
            if boundary & 1 != 0 { left.add_transition(l1, 8, l2, singleton_weight(2)); }
            if boundary & 16 != 0 { left.add_transition(l1, 8, l2, Weight::empty()); }
            left.set_final_weight(l2, token_range_weight(0, 2));
            let mut right = DWA::new(1, 2);
            let r1 = right.add_state();
            let r2 = right.add_state();
            right.add_transition(0, if boundary & 8 == 0 { 7 } else { 10 }, r1,
                token_range_weight(1, 2));
            if boundary & 2 != 0 { right.add_transition(r1, 8, r2, singleton_weight(1)); }
            if boundary & 4 != 0 { right.add_transition(r1, DEFAULT_LABEL, r2, singleton_weight(2)); }
            right.set_final_weight(r2, Weight::all());
            // Exercise both sides of the generic singleton normalization
            // threshold without mutating process-wide environment variables.
            if padded {
                while left.num_states() < 16_384 { left.add_state(); }
            }
            let direct = union_two_parser_dwas_direct(left.clone(), right.clone());
            let generic = generic_union(&[left, right]);
            for word in all_words(&[7, 8, 9], 3) {
                assert_eq!(fallback_eval(&direct, &word), fallback_eval(&generic, &word),
                    "padded={padded} boundary={boundary} word={word:?}");
            }
        }
        }
    }

    fn all_words(labels: &[i32], max_len: usize) -> Vec<Vec<i32>> {
        let mut words = vec![Vec::new()];
        let mut frontier = vec![Vec::new()];
        for _ in 0..max_len {
            let mut next = Vec::new();
            for prefix in frontier {
                for &label in labels {
                    let mut word = prefix.clone();
                    word.push(label);
                    words.push(word.clone());
                    next.push(word);
                }
            }
            frontier = next;
        }
        words
    }

    fn id_map_with_tsid_partition(
        original_to_internal: Vec<u32>,
        representative_original_ids: Vec<u32>,
    ) -> InternalIdMap {
        let num_tsids = representative_original_ids.len() as u32;
        InternalIdMap {
            tokenizer_states: ManyToOneIdMap::from_original_to_internal_with_representatives(
                original_to_internal,
                num_tsids,
                representative_original_ids,
            ),
            vocab_tokens: ManyToOneIdMap::from_original_to_internal_allowing_unmapped(
                Vec::new(),
                0,
            ),
            deferred_vocab_singleton_original_ids: None,
        }
    }

    fn reference_local_to_global_tsid_map(
        local_id_map: &InternalIdMap,
        global_id_map: &InternalIdMap,
    ) -> Vec<Vec<u32>> {
        let mut result = vec![std::collections::BTreeSet::new(); local_id_map.num_tsids() as usize];
        for (original_state, &local_tsid) in local_id_map
            .tokenizer_states
            .original_to_internal
            .iter()
            .enumerate()
        {
            if local_tsid == u32::MAX {
                continue;
            }
            result[local_tsid as usize]
                .insert(global_id_map.tokenizer_states.original_to_internal[original_state]);
        }
        result
            .into_iter()
            .map(|targets| targets.into_iter().collect())
            .collect()
    }

    fn id_map_with_vocab_partition(
        original_to_internal: Vec<u32>,
        num_classes: u32,
    ) -> InternalIdMap {
        InternalIdMap {
            tokenizer_states: ManyToOneIdMap::from_original_to_internal_with_representatives(
                vec![0],
                1,
                vec![0],
            ),
            vocab_tokens: ManyToOneIdMap::from_original_to_internal_allowing_unmapped(
                original_to_internal,
                num_classes,
            ),
            deferred_vocab_singleton_original_ids: None,
        }
    }

    #[test]
    fn sparse_overlapping_token_map_matches_generic_partition() {
        let first = id_map_with_vocab_partition(
            vec![0, 0, u32::MAX, 1, 1],
            2,
        );
        let second = id_map_with_vocab_partition(
            vec![u32::MAX, 0, 0, 1, u32::MAX],
            2,
        );
        let inputs = [&first, &second];
        let generic = build_unified_global_token_id_map_generic(&inputs, 4);
        let (sparse, local_to_global) =
            build_unified_global_token_id_map_sparse_generic(&inputs, 4);
        let (two_source, two_source_local_to_global) =
            build_unified_global_token_id_map_at_most_two(&inputs, 4)
                .expect("two inputs cannot create a third membership");
        let (wide, wide_local_to_global) =
            build_unified_global_token_id_map_at_most_two_wide(&inputs, 4)
                .expect("two inputs cannot create a third membership");
        let (packed, packed_local_to_global) =
            build_unified_global_token_id_map_at_most_two_packed(&inputs, 4)
                .expect("fixture fits packed membership bounds");

        let mut generic_groups = generic.internal_to_originals.clone();
        let mut sparse_groups = sparse.internal_to_originals.clone();
        let mut two_source_groups = two_source.internal_to_originals.clone();
        generic_groups.sort_unstable();
        sparse_groups.sort_unstable();
        two_source_groups.sort_unstable();
        assert_eq!(sparse_groups, generic_groups);
        assert_eq!(two_source_groups, generic_groups);
        assert_eq!(two_source.original_to_internal, sparse.original_to_internal);
        assert_eq!(two_source_local_to_global, local_to_global);
        assert_eq!(packed.original_to_internal, wide.original_to_internal);
        assert_eq!(packed.internal_to_originals, wide.internal_to_originals);
        assert_eq!(
            packed.representative_original_ids,
            wide.representative_original_ids,
        );
        assert_eq!(packed_local_to_global, wide_local_to_global);

        for (input_index, input) in inputs.iter().enumerate() {
            for (local_class, originals) in
                input.vocab_tokens.internal_to_originals.iter().enumerate()
            {
                for &token_id in originals {
                    let global_class = sparse.original_to_internal[token_id as usize];
                    assert!(
                        local_to_global[input_index][local_class].contains(&global_class),
                        "input={input_index} local_class={local_class} token={token_id} global={global_class}",
                    );
                }
            }
        }
    }

    #[test]
    fn assume_disjoint_token_map_matches_checked_disjoint() {
        let first = id_map_with_vocab_partition(
            vec![0, 0, u32::MAX, u32::MAX, 1, 1],
            2,
        );
        let second = id_map_with_vocab_partition(
            vec![u32::MAX, u32::MAX, 0, 0, u32::MAX, u32::MAX],
            1,
        );
        let inputs = [&first, &second];
        let (assumed, assumed_maps) =
            build_unified_global_token_id_map_assume_disjoint(&inputs, 5);
        let (checked, checked_maps) = build_unified_global_token_id_map_disjoint(&inputs, 5)
            .expect("fixture token domains are disjoint");

        assert_eq!(assumed.original_to_internal, checked.original_to_internal);
        assert_eq!(assumed.internal_to_originals, checked.internal_to_originals);
        assert_eq!(
            assumed.representative_original_ids,
            checked.representative_original_ids
        );
        assert_eq!(assumed_maps, checked_maps.expect("checked path returns direct maps"));
    }

    #[test]
    fn two_source_token_map_rejects_third_membership() {
        let first = id_map_with_vocab_partition(vec![0], 1);
        let second = id_map_with_vocab_partition(vec![0], 1);
        let third = id_map_with_vocab_partition(vec![0], 1);
        assert!(
            build_unified_global_token_id_map_at_most_two(
                &[&first, &second, &third],
                0,
            )
            .is_none()
        );
    }

    #[test]
    fn representative_tsid_map_matches_set_reference() {
        let global = id_map_with_tsid_partition(vec![0, 0, 1, 1, 2, 2], vec![0, 2, 4]);
        let local = id_map_with_tsid_partition(vec![0, 0, 1, 1, 0, 0], vec![0, 2]);

        assert_eq!(
            build_local_to_global_tsid_map(&local, &global),
            reference_local_to_global_tsid_map(&local, &global),
        );
    }

    #[test]
    fn representative_tsid_map_skips_global_classes_unmapped_locally() {
        let global = id_map_with_tsid_partition(vec![0, 0, 1, 1, 2, 2], vec![0, 2, 4]);
        let local = id_map_with_tsid_partition(
            vec![0, 0, u32::MAX, u32::MAX, 1, 1],
            vec![0, 4],
        );

        assert_eq!(
            build_local_to_global_tsid_map(&local, &global),
            vec![vec![0], vec![2]],
        );
        assert_eq!(
            build_local_to_global_tsid_map(&local, &global),
            reference_local_to_global_tsid_map(&local, &global),
        );
    }

    #[test]
    fn contiguous_token_offset_rejects_unmapped_classes() {
        assert_eq!(contiguous_token_offset(&[9, 10, 11]), Some(9));
        assert_eq!(contiguous_token_offset(&[u32::MAX]), None);
        assert_eq!(contiguous_token_offset(&[9, u32::MAX]), None);
    }

    #[test]
    fn affine_token_remap_matches_explicit_token_map() {
        let weight = Weight::from_per_tsid_token_sets([
            (0, RangeSetBlaze::from_iter([0..=2, 5..=5])),
            (1, RangeSetBlaze::from_iter([1..=4])),
        ]);
        let tsid_map = vec![vec![1, 3], vec![2]];
        let explicit = vec![vec![7], vec![8], vec![9], vec![10], vec![11], vec![12]];
        let mut explicit_cache = RemapCache::default();
        let mut explicit_tokens = RemapCache::default();
        let explicit_result = remap_weight_cached_with_token_remap(
            &weight,
            &tsid_map,
            &TokenRemap::Explicit(&explicit),
            4,
            &mut explicit_cache,
            &mut explicit_tokens,
        );
        let mut offset_cache = RemapCache::default();
        let mut offset_tokens = RemapCache::default();
        let offset_result = remap_weight_cached_with_token_remap(
            &weight,
            &tsid_map,
            &TokenRemap::Offset {
                offset: 7,
                local_token_count: 6,
            },
            4,
            &mut offset_cache,
            &mut offset_tokens,
        );
        assert_eq!(explicit_result, offset_result);
    }

    #[test]
    fn interval_remap_matches_generic_and_overlapping_fallback() {
        let weight = Weight::from_per_tsid_token_sets([
            (0, RangeSetBlaze::from_iter([0..=1])),
            (1, RangeSetBlaze::from_iter([2..=3])),
        ]);
        let token_remap = TokenRemap::Offset {
            offset: 10,
            local_token_count: 4,
        };

        for tsid_map in [vec![vec![2, 4], vec![0, 1, 3]], vec![vec![0], vec![0]]] {
            let runs = build_local_to_global_tsid_runs(&tsid_map);
            let mut generic_tokens = RemapCache::default();
            let generic = remap_weight_with_token_remap(
                &weight,
                &tsid_map,
                &token_remap,
                5,
                &mut generic_tokens,
            );
            let mut interval_tokens = RemapCache::default();
            let interval = remap_weight_with_tsid_runs(
                &weight,
                &tsid_map,
                &runs,
                &token_remap,
                5,
                &mut interval_tokens,
            );
            assert_eq!(interval, generic);
        }
    }

    #[test]
    fn full_weight_remap_covers_uniform_global_tsid_domain() {
        let tsid_map = vec![vec![2, 4], vec![0, 1, 3]];
        let mut cache = RemapCache::default();
        let mut token_cache = RemapCache::default();
        let remapped = remap_weight_cached_with_token_remap(
            &Weight::all(),
            &tsid_map,
            &TokenRemap::Offset {
                offset: 10,
                local_token_count: 3,
            },
            5,
            &mut cache,
            &mut token_cache,
        );
        let expected = Weight::from_uniform(0..=4, RangeSetBlaze::from_iter([10..=12]));
        assert_eq!(remapped, expected);
    }

    #[test]
    fn direct_disjoint_union_matches_generic_subset_determinization() {
        let left_weight = singleton_weight(0);
        let mut left = DWA::new(1, 1);
        let left_mid = left.add_state();
        let left_end = left.add_state();
        left.add_transition(left.start_state(), 7, left_mid, left_weight.clone());
        left.add_transition(left_mid, 8, left_end, left_weight.clone());
        left.set_final_weight(left_mid, left_weight.clone());
        left.set_final_weight(left_end, left_weight);

        let right_weight = singleton_weight(1);
        let mut right = DWA::new(1, 1);
        let right_mid = right.add_state();
        let right_end = right.add_state();
        right.add_transition(right.start_state(), 7, right_mid, right_weight.clone());
        right.add_transition(right_mid, 9, right_end, right_weight.clone());
        right.set_final_weight(right_mid, right_weight.clone());
        right.set_final_weight(right_end, right_weight);

        let inputs = vec![left, right];
        let direct = direct_union_disjoint_token_domain_dwas(&inputs);
        let generic = generic_union(&inputs);

        for word in all_words(&[7, 8, 9], 3) {
            assert_eq!(
                direct.eval_word(&word),
                generic.eval_word(&word),
                "word={word:?}"
            );
        }
    }

    #[test]
    fn immediate_parser_graft_preserves_overlapping_path_correlation() {
        let mut immediate = DWA::new(1, 2);
        let immediate_final = immediate.add_state();
        immediate.add_transition(
            immediate.start_state(),
            7,
            immediate_final,
            token_range_weight(0, 1),
        );
        immediate.set_final_weight(immediate_final, Weight::all());

        let mut other = DWA::new(1, 2);
        let middle = other.add_state();
        let end = other.add_state();
        other.add_transition(
            other.start_state(),
            7,
            middle,
            token_range_weight(1, 2),
        );
        // Token 0 occurs on the continuation edge, but it did not take the
        // other automaton's first edge. A naive merged edge/final update would
        // incorrectly allow token 0 to continue through label 8.
        other.add_transition(middle, 8, end, token_range_weight(0, 2));
        other.set_final_weight(middle, singleton_weight(1));
        other.set_final_weight(end, Weight::all());

        let generic = generic_union(&[immediate.clone(), other.clone()]);
        let grafted = try_graft_immediate_parser_dwa(&immediate, other.clone())
            .expect("depth-one parser DWA should graft");
        let quotient = exact_quotient_acyclic_dwa(grafted);
        let overlay: BTreeMap<i32, Weight> = immediate_dwa_accepting_edges(&immediate)
            .expect("depth-one parser DWA should form an overlay")
            .into_iter()
            .collect();

        for word in all_words(&[7, 8], 3) {
            let overlay_result = match word.as_slice() {
                [label] => other
                    .eval_word(&word)
                    .union(overlay.get(label).unwrap_or(&Weight::empty())),
                _ => other.eval_word(&word),
            };
            assert_eq!(
                quotient.eval_word(&word),
                generic.eval_word(&word),
                "word={word:?}"
            );
            assert_eq!(overlay_result, generic.eval_word(&word), "overlay word={word:?}");
        }
        assert_eq!(quotient.eval_word(&[7]), token_range_weight(0, 1));
        assert_eq!(quotient.eval_word(&[7, 8]), token_range_weight(1, 2));
    }

    #[test]
    fn immediate_parser_family_is_overlayed_before_two_deep_families_merge() {
        let mut left = DWA::new(1, 2);
        let left_mid = left.add_state();
        let left_end = left.add_state();
        left.add_transition(left.start_state(), 7, left_mid, token_range_weight(0, 1));
        left.add_transition(left_mid, 8, left_end, token_range_weight(0, 1));
        left.set_final_weight(left_end, Weight::all());

        let mut right = DWA::new(1, 2);
        let right_mid = right.add_state();
        let right_end = right.add_state();
        right.add_transition(right.start_state(), 7, right_mid, token_range_weight(1, 2));
        right.add_transition(right_mid, 9, right_end, token_range_weight(1, 2));
        right.set_final_weight(right_end, Weight::all());

        let mut immediate = DWA::new(1, 2);
        let immediate_final = immediate.add_state();
        immediate.add_transition(immediate.start_state(), 10, immediate_final, singleton_weight(2));
        immediate.set_final_weight(immediate_final, Weight::all());

        let generic = generic_union(&[left.clone(), right.clone(), immediate.clone()]);
        let id_map = id_map_with_vocab_partition(vec![0, 1, 2], 3);
        let (merged, overlay) = merge_mapped_parser_dwas_with_top_accept(
            vec![
                MappedArtifact::new(left, id_map.clone()),
                MappedArtifact::new(right, id_map.clone()),
                MappedArtifact::new(immediate, id_map),
            ],
            1,
            2,
        );

        for word in all_words(&[7, 8, 9, 10], 3) {
            let mut actual = merged.artifact().eval_word(&word);
            if let [label] = word.as_slice() {
                if let Some(weight) = overlay.get(label) {
                    actual = actual.union(weight);
                }
            }
            assert_eq!(actual, generic.eval_word(&word), "word={word:?}");
        }
    }

    #[test]
    fn multiple_immediate_parser_families_stay_as_top_accept_overlay() {
        let mut deep = DWA::new(1, 2);
        let deep_mid = deep.add_state();
        let deep_end = deep.add_state();
        deep.add_transition(deep.start_state(), 7, deep_mid, token_range_weight(1, 2));
        deep.add_transition(deep_mid, 8, deep_end, token_range_weight(0, 2));
        deep.set_final_weight(deep_mid, singleton_weight(1));
        deep.set_final_weight(deep_end, Weight::all());

        let mut first = DWA::new(1, 2);
        let first_final = first.add_state();
        first.add_transition(first.start_state(), 7, first_final, singleton_weight(0));
        first.set_final_weight(first_final, Weight::all());

        let mut second = DWA::new(1, 2);
        let second_final = second.add_state();
        second.add_transition(second.start_state(), 9, second_final, singleton_weight(2));
        second.set_final_weight(second_final, Weight::all());

        let generic = generic_union(&[deep.clone(), first.clone(), second.clone()]);
        let id_map = id_map_with_vocab_partition(vec![0, 1, 2], 3);
        let (merged, overlay) = merge_mapped_parser_dwas_with_top_accept(
            vec![
                MappedArtifact::new(deep, id_map.clone()),
                MappedArtifact::new(first, id_map.clone()),
                MappedArtifact::new(second, id_map),
            ],
            1,
            2,
        );

        for word in all_words(&[7, 8, 9], 3) {
            let mut actual = merged.artifact().eval_word(&word);
            if let [label] = word.as_slice() {
                if let Some(weight) = overlay.get(label) {
                    actual = actual.union(weight);
                }
            }
            assert_eq!(actual, generic.eval_word(&word), "word={word:?}");
        }
    }
}
