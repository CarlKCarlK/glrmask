//! Template bundle assembly into a weighted NWA.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::time::Instant;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::sync::Arc;

use crate::automata::unweighted_u32::dfa::DFA as UnweightedDfa;
use crate::automata::unweighted_u32::nfa::NFA as UnweightedNfa;
use crate::automata::unweighted_u32::determinize::determinize as unweighted_determinize;
use crate::automata::unweighted_u32::minimize_acyclic::minimize_acyclic as unweighted_minimize;
use crate::automata::weighted::dwa::DWA;
use crate::automata::weighted::minimize::minimize;
use crate::automata::weighted::nwa::{NWA, NWAState};
use crate::grammar::flat::TerminalID;
use crate::compiler::stages::templates::compile_dfa::Templates;
use crate::ds::weight::{SharedTokenSet, Weight};

type SubsetKey = SmallVec<[u64; 4]>;
type LabelTargets = SmallVec<[(i32, u32, u32); 8]>;
const SUBSET_BLOCK_BITS: usize = 8;
const SUBSET_BLOCK_MASK: u64 = (1u64 << SUBSET_BLOCK_BITS) - 1;

/// A per-weight terminal group either reuses an immutable template DFA or owns
/// the union required for a multi-terminal group. Keeping singleton groups
/// borrowed avoids cloning their full DFA only to read it once while building a
/// deterministic bundle.
enum BundleGroupDfa<'a> {
    Borrowed(&'a UnweightedDfa),
    Cached(Arc<UnweightedDfa>),
    Owned(UnweightedDfa),
}

/// Immutable topology cache shared by all bundles in one parser-DWA build.
/// The key is the exact sorted terminal set in one equal-weight group. The
/// weights themselves deliberately are not part of the cache: each bundle
/// still performs its own weighted product/determinization.
#[derive(Default)]
pub(crate) struct BundleGroupDfaCache {
    multi_terminal_groups: FxHashMap<Vec<TerminalID>, Arc<UnweightedDfa>>,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct BundleGroupDfaCachePlanStats {
    pub(crate) multi_terminal_group_occurrences: usize,
    pub(crate) repeated_groups: usize,
    pub(crate) repeated_group_occurrences: usize,
    pub(crate) repeated_terminal_occurrences: usize,
    pub(crate) max_repeated_group_terminals: usize,
}

pub(crate) struct BundleGroupDfaCachePlan {
    repeated: Vec<Vec<TerminalID>>,
    pub(crate) stats: BundleGroupDfaCachePlanStats,
}

impl BundleGroupDfaCache {
    pub(crate) fn len(&self) -> usize {
        self.multi_terminal_groups.len()
    }
}

impl BundleGroupDfa<'_> {
    #[inline]
    fn dfa(&self) -> &UnweightedDfa {
        match self {
            Self::Borrowed(dfa) => dfa,
            Self::Cached(dfa) => dfa.as_ref(),
            Self::Owned(dfa) => dfa,
        }
    }
}

fn checked_usize_to_u32(value: usize, what: &str) -> u32 {
    u32::try_from(value).unwrap_or_else(|_| panic!("{what} exceeds u32::MAX"))
}

fn checked_u32_add(lhs: u32, rhs: u32, what: &str) -> u32 {
    lhs.checked_add(rhs)
        .unwrap_or_else(|| panic!("{what} exceeds u32::MAX"))
}

fn empty_bundle_nwa() -> NWA {
    let mut nwa = NWA::new(0, 0);
    let start_state = nwa.add_state();
    nwa.start_states_mut().push(start_state);
    nwa
}

fn instantiate_weighted_nwa_from_skeleton(skeleton: &NWA, weight: &Weight) -> NWA {
    let mut bundle = skeleton.clone();
    for state in  bundle.states_mut() {
        if state.final_weight.is_some() {
            state.final_weight = Some(weight.clone());
        }
        for targets in state.transitions.values_mut() {
            for (_, edge_weight) in targets {
                *edge_weight = weight.clone();
            }
        }
        for (_, epsilon_weight) in &mut state.epsilons {
            *epsilon_weight = weight.clone();
        }
    }

    bundle
}

fn clear_subset_key(key: &mut SubsetKey) {
    for word in key.iter_mut() {
        *word = 0;
    }
}

fn collect_label_targets(
    groups: &[(&Weight, BundleGroupDfa<'_>)],
    product_state: &[(u32, u32)],
    label_targets: &mut LabelTargets,
) {
    label_targets.clear();
    for &(group_id, dfa_state) in product_state {
        let dfa = groups[group_id as usize].1.dfa();
        for (&label, &target) in &dfa.states[dfa_state as usize].transitions {
            label_targets.push((label, group_id, target));
        }
    }
    // Preserve the former BTreeMap's ascending-label traversal, while retaining
    // each product state's canonical group order within a label.
    label_targets.sort_unstable_by_key(|&(label, group_id, _)| (label, group_id));
}

fn set_subset_key_bit(key: &mut SubsetKey, index: usize) {
    let word_index = index / 64;
    let bit_index = index % 64;
    key[word_index] |= 1u64 << bit_index;
}

fn cached_subset_union(
    cache: &mut FxHashMap<SubsetKey, Weight>,
    subset_key: &SubsetKey,
    subset: &[usize],
    group_weights: &[Weight],
    subset_unions: Option<&SubsetUnionIndex>,
    single_tsid_entries: Option<&[(u32, SharedTokenSet)]>,
) -> Weight {
    match subset {
        [] => return Weight::empty(),
        [index] => return group_weights[*index].clone(),
        _ => {}
    }

    if let Some(existing) = cache.get(subset_key) {
        return existing.clone();
    }

    let result = if let Some(single_tsid_entries) = single_tsid_entries {
        Weight::union_single_tsid_shared_entries(
            subset
                .iter()
                .map(|&index| {
                    let (tsid, tokens) = &single_tsid_entries[index];
                    (*tsid, Arc::clone(tokens))
                }),
        )
    } else if subset.len() >= SUBSET_BLOCK_BITS {
        if let Some(subset_unions) = subset_unions {
            subset_unions.union(subset_key)
        } else {
            Weight::union_all(subset.iter().map(|&index| &group_weights[index]))
        }
    } else {
        Weight::union_all(subset.iter().map(|&index| &group_weights[index]))
    };
    cache.insert(subset_key.clone(), result.clone());
    result
}

struct SubsetUnionIndex {
    block_unions: Vec<Box<[Weight]>>,
    segment_base: usize,
    segment_unions: Vec<Weight>,
}

impl SubsetUnionIndex {
    fn new(group_weights: &[Weight]) -> Self {
        Self {
            block_unions: build_subset_block_unions(group_weights),
            segment_base: group_weights.len().next_power_of_two(),
            segment_unions: build_subset_segment_unions(group_weights),
        }
    }

    fn union(&self, subset_key: &SubsetKey) -> Weight {
        let block_parts = self.block_parts(subset_key);
        let segment_parts = self.segment_parts(subset_key);
        if segment_parts.len() < block_parts.len() {
            Weight::union_all(segment_parts)
        } else {
            Weight::union_all(block_parts)
        }
    }

    fn block_parts<'a>(&'a self, subset_key: &SubsetKey) -> SmallVec<[&'a Weight; 8]> {
        let mut parts = SmallVec::<[&Weight; 8]>::new();
        for (block_index, block_table) in self.block_unions.iter().enumerate() {
            let bit_offset = block_index * SUBSET_BLOCK_BITS;
            let word_index = bit_offset / 64;
            let bit_index = bit_offset % 64;
            let mask = ((subset_key[word_index] >> bit_index) & SUBSET_BLOCK_MASK) as usize;
            if mask != 0 {
                parts.push(&block_table[mask]);
            }
        }
        parts
    }

    fn segment_parts<'a>(&'a self, subset_key: &SubsetKey) -> SmallVec<[&'a Weight; 8]> {
        let mut parts = SmallVec::<[&Weight; 8]>::new();
        for (word_index, &word) in subset_key.iter().enumerate() {
            let mut remaining = word;
            while remaining != 0 {
                let start_bit = remaining.trailing_zeros() as usize;
                let shifted = remaining >> start_bit;
                let run_len = (!shifted).trailing_zeros().min(64 - start_bit as u32) as usize;
                let start = word_index * 64 + start_bit;
                self.push_segment_range_parts(start, start + run_len, &mut parts);
                let run_mask = if run_len == 64 {
                    u64::MAX
                } else {
                    ((1u64 << run_len) - 1) << start_bit
                };
                remaining &= !run_mask;
            }
        }
        parts
    }

    fn push_segment_range_parts<'a>(
        &'a self,
        mut start: usize,
        mut end: usize,
        parts: &mut SmallVec<[&'a Weight; 8]>,
    ) {
        start += self.segment_base;
        end += self.segment_base;
        while start < end {
            if start % 2 == 1 {
                let weight = &self.segment_unions[start];
                if !weight.is_empty() {
                    parts.push(weight);
                }
                start += 1;
            }
            if end % 2 == 1 {
                end -= 1;
                let weight = &self.segment_unions[end];
                if !weight.is_empty() {
                    parts.push(weight);
                }
            }
            start /= 2;
            end /= 2;
        }
    }
}

fn build_subset_block_unions(group_weights: &[Weight]) -> Vec<Box<[Weight]>> {
    group_weights
        .chunks(SUBSET_BLOCK_BITS)
        .map(|chunk| {
            let table_len = 1usize << chunk.len();
            let mut unions = vec![Weight::empty(); table_len];
            for mask in 1..table_len {
                let bit = mask.trailing_zeros() as usize;
                let prev = mask & (mask - 1);
                unions[mask] = unions[prev].union(&chunk[bit]);
            }
            unions.into_boxed_slice()
        })
        .collect()
}

fn build_subset_segment_unions(group_weights: &[Weight]) -> Vec<Weight> {
    let base = group_weights.len().next_power_of_two();
    let mut tree = vec![Weight::empty(); base * 2];
    for (index, weight) in group_weights.iter().enumerate() {
        tree[base + index] = weight.clone();
    }
    for index in (1..base).rev() {
        tree[index] = tree[index * 2].union(&tree[index * 2 + 1]);
    }
    tree
}

#[derive(Clone, Debug, Default)]
pub struct BundleBuildProfile {
    pub input_terminals: usize,
    pub nonempty_terminals: usize,
    pub weight_groups: usize,
    pub single_entry_weights: usize,
    pub single_tsid_weights: usize,
    pub total_weight_outer_ranges: usize,
    pub singleton_groups: usize,
    pub multi_terminal_groups: usize,
    pub largest_weight_group: usize,
    pub build_group_dfas_ms: f64,
    pub union_groups_ms: f64,
    pub slowest_group_terminals: usize,
    pub slowest_group_dfa_states: usize,
    pub slowest_group_dfa_transitions: usize,
    pub slowest_group_ms: f64,
    pub group_dfa_cache_hits: usize,
    pub group_dfa_cache_misses: usize,
    pub determinize_bundle_ms: f64,
    pub determinize_pop_state_ms: f64,
    pub determinize_alive_groups_ms: f64,
    pub determinize_effective_weights_ms: f64,
    pub determinize_final_weight_ms: f64,
    pub determinize_collect_labels_ms: f64,
    pub determinize_next_state_ms: f64,
    pub determinize_edge_weight_ms: f64,
    pub determinize_state_lookup_ms: f64,
    pub determinize_add_transition_ms: f64,
    pub determinize_states_visited: usize,
    pub determinize_labels_processed: usize,
    pub determinize_transitions_added: usize,
    pub determinize_worklist_peak: usize,
    pub determinize_cache_entries: usize,
    pub determinize_edge_subset_total: usize,
    pub determinize_edge_subset_max: usize,
    pub determinize_edge_cache_hits: usize,
    pub determinize_edge_cache_hit_subset_total: usize,
    pub determinize_edge_cache_misses: usize,
    pub determinize_edge_cache_miss_subset_total: usize,
    pub minimize_ms: f64,
    pub dwa_to_nwa_ms: f64,
    pub result_dwa_states: usize,
    pub result_dwa_transitions: usize,
    pub result_nwa_states: usize,
    pub result_nwa_transitions: usize,
    pub total_ms: f64,
    pub used_single_terminal_fast_path: bool,
    pub minimize_skipped: bool,
}

fn elapsed_ms(started_at: Instant) -> f64 {
    started_at.elapsed().as_secs_f64() * 1000.0
}

fn minimize_template_bundles_enabled() -> bool {
    std::env::var("GLRMASK_MINIMIZE_TEMPLATE_BUNDLES")
        .map(|value| value == "1")
        .unwrap_or(false)
}

fn template_bundle_subset_union_index_enabled() -> bool {
    std::env::var("GLRMASK_TEMPLATE_BUNDLE_SUBSET_UNION_INDEX")
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

fn compute_unweighted_dfa_transition_count(dfa: &UnweightedDfa) -> usize {
    dfa.states.iter().map(|state| state.transitions.len()).sum()
}

impl Templates {
    fn build_single_terminal_bundle(
        &self,
        terminal_weights: &BTreeMap<TerminalID, Weight>,
    ) -> Option<NWA> {
        let (&terminal, weight) = terminal_weights.iter().next()?;
        if terminal_weights.len() != 1 {
            return None;
        }
        if weight.is_empty() {
            return Some(empty_bundle_nwa());
        }
        Some(
            self.by_terminal_nwa
                .get(&terminal)
                .map(|template_nwa| instantiate_weighted_nwa_from_skeleton(template_nwa, weight))
                .unwrap_or_else(empty_bundle_nwa),
        )
    }

    fn group_terminals_by_weight<'a>(
        &'a self,
        terminal_weights: &'a BTreeMap<TerminalID, Weight>,
    ) -> Vec<(&'a Weight, Vec<TerminalID>)> {
        let mut weight_groups: HashMap<&Weight, Vec<TerminalID>> = HashMap::new();
        for (&terminal, weight) in terminal_weights {
            if weight.is_empty() || !self.by_terminal.contains_key(&terminal) {
                continue;
            }
            weight_groups.entry(weight).or_default().push(terminal);
        }

        // Terminal IDs come from a BTreeMap, so each group's first terminal is
        // stable. This gives the product construction a deterministic order
        // without traversing every group DFA to allocate a structural sort key.
        let mut groups = weight_groups.into_iter().collect::<Vec<_>>();
        groups.sort_unstable_by_key(|(_, terminals)| terminals[0]);
        groups
    }

    pub(crate) fn plan_bundle_group_dfa_cache(
        &self,
        bundles: &[&BTreeMap<TerminalID, Weight>],
    ) -> BundleGroupDfaCachePlan {
        let mut counts = BTreeMap::<Vec<TerminalID>, usize>::new();
        let mut multi_terminal_group_occurrences = 0usize;
        for bundle in bundles {
            for (_, terminals) in self.group_terminals_by_weight(bundle) {
                if terminals.len() > 1 {
                    multi_terminal_group_occurrences += 1;
                    *counts.entry(terminals).or_default() += 1;
                }
            }
        }
        let mut stats = BundleGroupDfaCachePlanStats {
            multi_terminal_group_occurrences,
            ..BundleGroupDfaCachePlanStats::default()
        };
        let mut repeated = Vec::new();
        for (terminals, count) in counts {
            if count <= 1 {
                continue;
            }
            stats.repeated_groups += 1;
            stats.repeated_group_occurrences += count;
            stats.repeated_terminal_occurrences += terminals.len() * count;
            stats.max_repeated_group_terminals =
                stats.max_repeated_group_terminals.max(terminals.len());
            repeated.push(terminals);
        }
        BundleGroupDfaCachePlan { repeated, stats }
    }

    pub(crate) fn build_bundle_group_dfa_cache(
        &self,
        bundles: &[&BTreeMap<TerminalID, Weight>],
    ) -> BundleGroupDfaCache {
        self.build_bundle_group_dfa_cache_from_plan(self.plan_bundle_group_dfa_cache(bundles))
    }

    pub(crate) fn build_bundle_group_dfa_cache_from_plan(
        &self,
        plan: BundleGroupDfaCachePlan,
    ) -> BundleGroupDfaCache {
        let repeated = plan.repeated;
        let build_entry = |terminals: Vec<TerminalID>| {
                let merged = union_unweighted_dfas(
                    terminals
                        .iter()
                        .filter_map(|terminal| self.by_terminal.get(terminal)),
                );
                (terminals, Arc::new(merged))
            };
        let entries = if super::macro_parallelism_disabled() {
            let mut timings = Vec::with_capacity(repeated.len());
            let entries = repeated
                .into_iter()
                .map(|terminals| {
                    let started = Instant::now();
                    let entry = build_entry(terminals);
                    timings.push(elapsed_ms(started));
                    entry
                })
                .collect::<Vec<_>>();
            super::report_macro_item_timings("template_bundle_group_dfa_cache", &timings);
            entries
        } else {
            repeated.into_par_iter().map(build_entry).collect::<Vec<_>>()
        };
        BundleGroupDfaCache {
            multi_terminal_groups: entries.into_iter().collect(),
        }
    }

    fn build_group_dfas_profiled<'a>(
        &'a self,
        weight_groups: &'a [(&'a Weight, Vec<TerminalID>)],
        group_cache: Option<&BundleGroupDfaCache>,
        profile: &mut BundleBuildProfile,
    ) -> Vec<(&'a Weight, BundleGroupDfa<'a>)> {
        let build_started_at = Instant::now();
        let mut group_dfas = Vec::with_capacity(weight_groups.len());
        for (weight, terminals) in weight_groups {
            profile.nonempty_terminals += terminals.len();
            profile.largest_weight_group = profile.largest_weight_group.max(terminals.len());
            if terminals.len() == 1 {
                profile.singleton_groups += 1;
                if let Some(template) = self.by_terminal.get(&terminals[0]) {
                    group_dfas.push((*weight, BundleGroupDfa::Borrowed(template)));
                }
                continue;
            }

            profile.multi_terminal_groups += 1;
            if let Some(cached) = group_cache
                .and_then(|cache| cache.multi_terminal_groups.get(terminals))
            {
                profile.group_dfa_cache_hits += 1;
                group_dfas.push((*weight, BundleGroupDfa::Cached(Arc::clone(cached))));
                continue;
            }

            profile.group_dfa_cache_misses += usize::from(group_cache.is_some());
            let group_started_at = Instant::now();
            let merged = union_unweighted_dfas(
                terminals.iter().filter_map(|terminal| self.by_terminal.get(terminal)),
            );
            let group_ms = elapsed_ms(group_started_at);
            profile.union_groups_ms += group_ms;

            if group_ms > profile.slowest_group_ms {
                profile.slowest_group_ms = group_ms;
                profile.slowest_group_terminals = terminals.len();
                profile.slowest_group_dfa_states = merged.states.len();
                profile.slowest_group_dfa_transitions = compute_unweighted_dfa_transition_count(&merged);
            }

            group_dfas.push((*weight, BundleGroupDfa::Owned(merged)));
        }
        profile.build_group_dfas_ms = elapsed_ms(build_started_at);
        group_dfas
    }

    fn build_group_dfas<'a>(
        &'a self,
        weight_groups: &'a [(&'a Weight, Vec<TerminalID>)],
        group_cache: Option<&BundleGroupDfaCache>,
    ) -> Vec<(&'a Weight, BundleGroupDfa<'a>)> {
        let mut group_dfas = Vec::with_capacity(weight_groups.len());
        for (weight, terminals) in weight_groups {
            if terminals.len() == 1 {
                if let Some(template) = self.by_terminal.get(&terminals[0]) {
                    group_dfas.push((*weight, BundleGroupDfa::Borrowed(template)));
                }
            } else if let Some(cached) = group_cache
                .and_then(|cache| cache.multi_terminal_groups.get(terminals))
            {
                group_dfas.push((*weight, BundleGroupDfa::Cached(Arc::clone(cached))));
            } else {
                let merged = union_unweighted_dfas(
                    terminals.iter().filter_map(|terminal| self.by_terminal.get(terminal)),
                );
                group_dfas.push((*weight, BundleGroupDfa::Owned(merged)));
            }
        }
        group_dfas
    }

    pub fn build_bundle_profiled(
        &self,
        terminal_weights: &BTreeMap<TerminalID, Weight>,
    ) -> (NWA, BundleBuildProfile) {
        self.build_bundle_profiled_with_group_cache(terminal_weights, None)
    }

    pub(crate) fn build_bundle_profiled_cached(
        &self,
        terminal_weights: &BTreeMap<TerminalID, Weight>,
        group_cache: &BundleGroupDfaCache,
    ) -> (NWA, BundleBuildProfile) {
        self.build_bundle_profiled_with_group_cache(terminal_weights, Some(group_cache))
    }

    fn build_bundle_profiled_with_group_cache(
        &self,
        terminal_weights: &BTreeMap<TerminalID, Weight>,
        group_cache: Option<&BundleGroupDfaCache>,
    ) -> (NWA, BundleBuildProfile) {
        let total_started_at = Instant::now();
        let mut profile = BundleBuildProfile {
            input_terminals: terminal_weights.len(),
            ..BundleBuildProfile::default()
        };

        if let Some(bundle) = self.build_single_terminal_bundle(terminal_weights) {
            profile.used_single_terminal_fast_path = true;
            profile.result_nwa_states = bundle.states().len();
            profile.result_nwa_transitions = NWA::num_transitions(&bundle);
            profile.total_ms = elapsed_ms(total_started_at);
            return (bundle, profile);
        }

        let weight_groups = self.group_terminals_by_weight(terminal_weights);
        profile.weight_groups = weight_groups.len();
        for (weight, _) in &weight_groups {
            profile.total_weight_outer_ranges += weight.outer_range_count();
            if weight.single_compact_entry_parts().is_some() {
                profile.single_entry_weights += 1;
            }
            if weight.single_tsid_shared_entry().is_some() {
                profile.single_tsid_weights += 1;
            }
        }
        let group_dfas = self.build_group_dfas_profiled(&weight_groups, group_cache, &mut profile);

        // STICKY NOTE: NEVER REMOVE THIS NOTE.
        // These parser bundles must be determinized before they are converted
        // back to NWAs and stitched into the parser DWA. Negative-resolution
        // relies on this determinized boundary: if bundles remain nondeterministic
        // or merely factored, the next bundle has to combine with all unresolved
        // alternatives from the previous bundle, and adjacent bundles explode
        // combinatorially. If a large grammar cannot determinize the first bundle,
        // that is a compiler-scaling problem to solve by reducing/quotienting the
        // deterministic product, not by skipping this determinization step.
        let determinize_started_at = Instant::now();
        let (bundle_dwa, determinize_profile) = determinize_bundle_groups_profiled(&group_dfas);
        profile.determinize_bundle_ms = elapsed_ms(determinize_started_at);
        profile.determinize_pop_state_ms = determinize_profile.pop_state_ms;
        profile.determinize_alive_groups_ms = determinize_profile.alive_groups_ms;
        profile.determinize_effective_weights_ms = determinize_profile.effective_weights_ms;
        profile.determinize_final_weight_ms = determinize_profile.final_weight_ms;
        profile.determinize_collect_labels_ms = determinize_profile.collect_labels_ms;
        profile.determinize_next_state_ms = determinize_profile.next_state_ms;
        profile.determinize_edge_weight_ms = determinize_profile.edge_weight_ms;
        profile.determinize_state_lookup_ms = determinize_profile.state_lookup_ms;
        profile.determinize_add_transition_ms = determinize_profile.add_transition_ms;
        profile.determinize_states_visited = determinize_profile.states_visited;
        profile.determinize_labels_processed = determinize_profile.labels_processed;
        profile.determinize_transitions_added = determinize_profile.transitions_added;
        profile.determinize_worklist_peak = determinize_profile.worklist_peak;
        profile.determinize_cache_entries = determinize_profile.cache_entries;
        profile.determinize_edge_subset_total = determinize_profile.edge_subset_total;
        profile.determinize_edge_subset_max = determinize_profile.edge_subset_max;
        profile.determinize_edge_cache_hits = determinize_profile.edge_cache_hits;
        profile.determinize_edge_cache_hit_subset_total = determinize_profile.edge_cache_hit_subset_total;
        profile.determinize_edge_cache_misses = determinize_profile.edge_cache_misses;
        profile.determinize_edge_cache_miss_subset_total = determinize_profile.edge_cache_miss_subset_total;
        profile.result_dwa_states = bundle_dwa.states().len();
        profile.result_dwa_transitions = DWA::num_transitions(&bundle_dwa);

        let minimize_started_at = Instant::now();
        profile.minimize_skipped = !minimize_template_bundles_enabled();
        let minimized = if profile.weight_groups > 1 && !profile.minimize_skipped {
            minimize(&bundle_dwa)
        } else {
            bundle_dwa
        };
        profile.minimize_ms = elapsed_ms(minimize_started_at);
        profile.result_dwa_states = minimized.states().len();
        profile.result_dwa_transitions = DWA::num_transitions(&minimized);

        let to_nwa_started_at = Instant::now();
        let nwa = dwa_to_nwa(&minimized);
        profile.dwa_to_nwa_ms = elapsed_ms(to_nwa_started_at);
        profile.result_nwa_states = nwa.states().len();
        profile.result_nwa_transitions = NWA::num_transitions(&nwa);
        profile.total_ms = elapsed_ms(total_started_at);

        (nwa, profile)
    }

    /// Assemble a weighted NWA for one bundle of (terminal, weight) entries.
    ///
    /// Pipeline: group by weight, merge each group, determinize the product,
    /// then convert back to an NWA. This determinization is correctness-critical
    /// for keeping negative-resolution local between bundle boundaries; see the
    /// sticky note in `build_bundle_profiled` above before changing it.
    /// Bundle minimization is skipped by default
    /// because parser-DWA composition reuses these fragments directly and the
    /// minimization cost dominates compile time on large schemas. Set
    /// `GLRMASK_MINIMIZE_TEMPLATE_BUNDLES=1` to restore the old behavior.
    pub fn build_bundle(
        &self,
        terminal_weights: &BTreeMap<TerminalID, Weight>,
    ) -> NWA {
        self.build_bundle_with_group_cache(terminal_weights, None)
    }

    pub(crate) fn build_bundle_cached(
        &self,
        terminal_weights: &BTreeMap<TerminalID, Weight>,
        group_cache: &BundleGroupDfaCache,
    ) -> NWA {
        self.build_bundle_with_group_cache(terminal_weights, Some(group_cache))
    }

    fn build_bundle_with_group_cache(
        &self,
        terminal_weights: &BTreeMap<TerminalID, Weight>,
        group_cache: Option<&BundleGroupDfaCache>,
    ) -> NWA {
        if let Some(bundle) = self.build_single_terminal_bundle(terminal_weights) {
            return bundle;
        }

        if std::env::var_os("GLRMASK_EXPERIMENT_SMALL_BUNDLE_UNGROUPED").is_some()
            && terminal_weights.len() <= 8
            && !minimize_template_bundles_enabled()
        {
            let direct_groups = terminal_weights
                .iter()
                .filter_map(|(&terminal, weight)| {
                    (!weight.is_empty())
                        .then(|| {
                            self.by_terminal
                                .get(&terminal)
                                .map(|dfa| (weight, BundleGroupDfa::Borrowed(dfa)))
                        })
                        .flatten()
                })
                .collect::<Vec<_>>();
            if direct_groups.len() > 1
                && std::env::var_os("GLRMASK_EXPERIMENT_SMALL_BUNDLE_FIXED_NWA").is_some()
                && let Some(nwa) = determinize_bundle_groups_serial_small_fixed_nwa(&direct_groups)
            {
                return nwa;
            }
        }

        let weight_groups = self.group_terminals_by_weight(terminal_weights);
        let group_dfas = self.build_group_dfas(&weight_groups, group_cache);
        if std::env::var_os("GLRMASK_EXPERIMENT_SMALL_BUNDLE_FIXED_NWA").is_some()
            && !minimize_template_bundles_enabled()
            && let Some(nwa) = determinize_bundle_groups_serial_small_fixed_nwa(&group_dfas)
        {
            return nwa;
        }
        let bundle_dwa = determinize_bundle_groups(&group_dfas);
        let minimized = if weight_groups.len() > 1 && minimize_template_bundles_enabled() {
            minimize(&bundle_dwa)
        } else {
            bundle_dwa
        };
        dwa_to_nwa(&minimized)
    }
}

#[derive(Clone, Debug, Default)]
struct DeterminizeBundleProfile {
    pop_state_ms: f64,
    alive_groups_ms: f64,
    effective_weights_ms: f64,
    final_weight_ms: f64,
    collect_labels_ms: f64,
    next_state_ms: f64,
    edge_weight_ms: f64,
    state_lookup_ms: f64,
    add_transition_ms: f64,
    states_visited: usize,
    labels_processed: usize,
    transitions_added: usize,
    worklist_peak: usize,
    cache_entries: usize,
    edge_subset_total: usize,
    edge_subset_max: usize,
    edge_cache_hits: usize,
    edge_cache_hit_subset_total: usize,
    edge_cache_misses: usize,
    edge_cache_miss_subset_total: usize,
}

fn determinize_bundle_groups_profiled(
    groups: &[(&Weight, BundleGroupDfa<'_>)],
) -> (DWA, DeterminizeBundleProfile) {
    use crate::automata::weighted_u32::dwa::DWA;

    let mut profile = DeterminizeBundleProfile::default();

    let n = groups.len();
    if n == 0 {
        return (DWA::new(0, 0), profile);
    }

    let group_weights: Vec<Weight> = groups
        .iter()
        .map(|(weight, _)| (*weight).clone())
        .collect();
    let single_tsid_entries = group_weights
        .iter()
        .map(Weight::single_tsid_shared_entry)
        .collect::<Option<Vec<_>>>();

    let mut subset_union_cache: FxHashMap<SubsetKey, Weight> = FxHashMap::default();
    let subset_unions = template_bundle_subset_union_index_enabled().then(|| SubsetUnionIndex::new(&group_weights));

    let start_key: Vec<(u32, u32)> = groups
        .iter()
        .enumerate()
        .map(|(group_id, (_, dfa))| {
            (
                checked_usize_to_u32(group_id, "bundle group id"),
                dfa.dfa().start_state,
            )
        })
        .collect();

    let mut dwa = DWA::new(0, 0);
    let mut state_map: FxHashMap<Vec<(u32, u32)>, u32> = FxHashMap::default();
    let mut singleton_state_map: FxHashMap<(u32, u32), u32> = FxHashMap::default();
    let mut worklist: VecDeque<(u32, Vec<(u32, u32)>)> = VecDeque::new();

    if let [singleton] = start_key.as_slice() {
        singleton_state_map.insert(*singleton, 0);
    } else {
        state_map.insert(start_key.clone(), 0);
    }
    worklist.push_back((0, start_key));
    profile.worklist_peak = worklist.len();

    let mut label_targets = LabelTargets::new();
    let key_words = n.div_ceil(64);
    let mut final_groups = SmallVec::<[usize; 8]>::new();
    let mut final_key = SubsetKey::from_elem(0, key_words);
    let mut edge_groups = SmallVec::<[usize; 8]>::new();
    let mut edge_key = SubsetKey::from_elem(0, key_words);

    while let Some((dwa_state, product_state)) = worklist.pop_front() {
        profile.states_visited += 1;
        let state_started_at = Instant::now();
        profile.pop_state_ms += elapsed_ms(state_started_at);

        let alive_started_at = Instant::now();
        let _alive_groups = product_state.len();
        profile.alive_groups_ms += elapsed_ms(alive_started_at);

        let effective_started_at = Instant::now();
        profile.effective_weights_ms += elapsed_ms(effective_started_at);

        let final_started_at = Instant::now();
        final_groups.clear();
        clear_subset_key(&mut final_key);
        for &(group_id, dfa_state) in &product_state {
            let group_id = group_id as usize;
            if groups[group_id].1.dfa().states[dfa_state as usize].is_accepting {
                final_groups.push(group_id);
                set_subset_key_bit(&mut final_key, group_id);
            }
        }
        let final_w = cached_subset_union(
            &mut subset_union_cache,
            &final_key,
            &final_groups,
            &group_weights,
            subset_unions.as_ref(),
            single_tsid_entries.as_deref(),
        );
        if !final_w.is_empty() {
            dwa.set_final_weight(dwa_state, final_w);
        }
        profile.final_weight_ms += elapsed_ms(final_started_at);

        let labels_started_at = Instant::now();
        collect_label_targets(groups, &product_state, &mut label_targets);
        profile.collect_labels_ms += elapsed_ms(labels_started_at);

        let mut label_start = 0usize;
        while label_start < label_targets.len() {
            let label = label_targets[label_start].0;
            let mut label_end = label_start + 1;
            while label_end < label_targets.len() && label_targets[label_end].0 == label {
                label_end += 1;
            }
            profile.labels_processed += 1;

            let next_state_started_at = Instant::now();
            edge_groups.clear();
            clear_subset_key(&mut edge_key);
            let singleton_target = (label_end == label_start + 1).then(|| {
                let (_, group_id, target) = label_targets[label_start];
                (group_id, target)
            });
            let mut next_state = singleton_target
                .is_none()
                .then(|| Vec::with_capacity(label_end - label_start));
            for &(_, group_id, target) in &label_targets[label_start..label_end] {
                let group_id = group_id as usize;
                edge_groups.push(group_id);
                set_subset_key_bit(&mut edge_key, group_id);
                if let Some(next_state) = next_state.as_mut() {
                    next_state.push((group_id as u32, target));
                }
            }
            profile.next_state_ms += elapsed_ms(next_state_started_at);

            let edge_subset_len = edge_groups.len();
            profile.edge_subset_total += edge_subset_len;
            profile.edge_subset_max = profile.edge_subset_max.max(edge_subset_len);
            if subset_union_cache.contains_key(&edge_key) {
                profile.edge_cache_hits += 1;
                profile.edge_cache_hit_subset_total += edge_subset_len;
            } else {
                profile.edge_cache_misses += 1;
                profile.edge_cache_miss_subset_total += edge_subset_len;
            }

            let edge_weight_started_at = Instant::now();
            let edge_w = cached_subset_union(
                &mut subset_union_cache,
                &edge_key,
                &edge_groups,
                &group_weights,
                subset_unions.as_ref(),
                single_tsid_entries.as_deref(),
            );
            if edge_w.is_empty() {
                profile.edge_weight_ms += elapsed_ms(edge_weight_started_at);
                label_start = label_end;
                continue;
            }
            profile.edge_weight_ms += elapsed_ms(edge_weight_started_at);

            let lookup_started_at = Instant::now();
            let to_dwa = if let Some(singleton_target) = singleton_target {
                if let Some(&existing) = singleton_state_map.get(&singleton_target) {
                    existing
                } else {
                    let new_id = dwa.add_state();
                    singleton_state_map.insert(singleton_target, new_id);
                    worklist.push_back((new_id, vec![singleton_target]));
                    profile.worklist_peak = profile.worklist_peak.max(worklist.len());
                    new_id
                }
            } else {
                let next_state = next_state.expect("non-singleton bundle state is populated");
                if let Some(&existing) = state_map.get(&next_state) {
                    existing
                } else {
                    let new_id = dwa.add_state();
                    state_map.insert(next_state.clone(), new_id);
                    worklist.push_back((new_id, next_state));
                    profile.worklist_peak = profile.worklist_peak.max(worklist.len());
                    new_id
                }
            };
            profile.state_lookup_ms += elapsed_ms(lookup_started_at);

            let add_transition_started_at = Instant::now();
            dwa.add_transition(dwa_state, label, to_dwa, edge_w);
            profile.add_transition_ms += elapsed_ms(add_transition_started_at);
            profile.transitions_added += 1;
            label_start = label_end;
        }
    }

    profile.cache_entries = subset_union_cache.len();

    (dwa, profile)
}


fn determinize_bundle_groups_parallel_small(groups: &[(&Weight, BundleGroupDfa<'_>)]) -> Option<DWA> {
    use crate::automata::weighted_u32::dwa::DWA;

    let started_at = Instant::now();
    let n = groups.len();
    if n == 0 || n > 8 || rayon::current_num_threads() <= 1 {
        return None;
    }

    // For a small bundle there are only 2^n possible supports. Materialize
    // their exact Weight union once; workers then carry a u16 support mask
    // rather than performing Weight algebra in the product-state hot loop.
    let subset_count = 1usize << n;
    let mut subset_weights = Vec::with_capacity(subset_count);
    subset_weights.push(Weight::empty());
    for mask in 1..subset_count {
        let bit = mask.trailing_zeros() as usize;
        let previous = mask & (mask - 1);
        subset_weights.push(subset_weights[previous].union(groups[bit].0));
    }

    #[derive(Debug)]
    struct ScannedEdge {
        label: i32,
        next: Vec<(u32, u32)>,
        support_mask: u16,
    }
    #[derive(Debug)]
    struct ScannedState {
        from: u32,
        final_mask: u16,
        edges: Vec<ScannedEdge>,
    }

    let start_key = groups
        .iter()
        .enumerate()
        .map(|(group_id, (_, dfa))| {
            (
                checked_usize_to_u32(group_id, "bundle group id"),
                dfa.dfa().start_state,
            )
        })
        .collect::<Vec<_>>();

    let mut dwa = DWA::new(0, 0);
    let mut state_map = FxHashMap::<Vec<(u32, u32)>, u32>::default();
    let mut singleton_state_map = FxHashMap::<(u32, u32), u32>::default();
    if let [singleton] = start_key.as_slice() {
        singleton_state_map.insert(*singleton, 0);
    } else {
        state_map.insert(start_key.clone(), 0);
    }
    let mut frontier = vec![(0u32, start_key)];
    let mut waves = 0usize;
    let mut max_frontier = 0usize;

    while !frontier.is_empty() {
        waves += 1;
        max_frontier = max_frontier.max(frontier.len());
        let scanned = frontier
            .par_iter()
            .map_init(LabelTargets::new, |label_targets, (from, product_state)| {
                let mut final_mask = 0u16;
                for &(group_id, dfa_state) in product_state {
                    if groups[group_id as usize].1.dfa().states[dfa_state as usize].is_accepting {
                        final_mask |= 1u16 << group_id;
                    }
                }

                collect_label_targets(groups, product_state, label_targets);
                let mut edges = Vec::with_capacity(label_targets.len());
                let mut label_start = 0usize;
                while label_start < label_targets.len() {
                    let label = label_targets[label_start].0;
                    let mut label_end = label_start + 1;
                    while label_end < label_targets.len() && label_targets[label_end].0 == label {
                        label_end += 1;
                    }
                    let mut support_mask = 0u16;
                    let mut next = Vec::with_capacity(label_end - label_start);
                    for &(_, group_id, target) in &label_targets[label_start..label_end] {
                        support_mask |= 1u16 << group_id;
                        next.push((group_id, target));
                    }
                    edges.push(ScannedEdge {
                        label,
                        next,
                        support_mask,
                    });
                    label_start = label_end;
                }
                ScannedState {
                    from: *from,
                    final_mask,
                    edges,
                }
            })
            .collect::<Vec<_>>();

        let mut next_frontier = Vec::<(u32, Vec<(u32, u32)>)>::new();
        for state in scanned {
            if state.final_mask != 0 {
                let weight = subset_weights[state.final_mask as usize].clone();
                if !weight.is_empty() {
                    dwa.set_final_weight(state.from, weight);
                }
            }
            for edge in state.edges {
                let edge_weight = subset_weights[edge.support_mask as usize].clone();
                if edge_weight.is_empty() {
                    continue;
                }
                let target = if let [singleton] = edge.next.as_slice() {
                    if let Some(&existing) = singleton_state_map.get(singleton) {
                        existing
                    } else {
                        let new_id = dwa.add_state();
                        singleton_state_map.insert(*singleton, new_id);
                        next_frontier.push((new_id, edge.next));
                        new_id
                    }
                } else if let Some(&existing) = state_map.get(&edge.next) {
                    existing
                } else {
                    let new_id = dwa.add_state();
                    state_map.insert(edge.next.clone(), new_id);
                    next_frontier.push((new_id, edge.next));
                    new_id
                };
                dwa.add_transition(state.from, edge.label, target, edge_weight);
            }
        }
        frontier = next_frontier;
    }

    if std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
    {
        eprintln!(
            "[glrmask/profile][parallel_bundle_determinize] groups={} subset_weights={} states={} transitions={} waves={} max_frontier={} total_ms={:.3}",
            n,
            subset_weights.len(),
            dwa.num_states(),
            dwa.num_transitions(),
            waves,
            max_frontier,
            elapsed_ms(started_at),
        );
    }
    Some(dwa)
}




fn determinize_bundle_groups_serial_small_fixed_nwa(
    groups: &[(&Weight, BundleGroupDfa<'_>)],
) -> Option<NWA> {
    const DEAD: u32 = u32::MAX;
    let n = groups.len();
    if n == 0 || n > 8 {
        return None;
    }

    let subset_count = 1usize << n;
    let mut subset_weights = Vec::with_capacity(subset_count);
    subset_weights.push(Weight::empty());
    for mask in 1..subset_count {
        let bit = mask.trailing_zeros() as usize;
        let previous = mask & (mask - 1);
        subset_weights.push(subset_weights[previous].union(groups[bit].0));
    }

    let mut start_key = [DEAD; 8];
    for (group_id, (_, dfa)) in groups.iter().enumerate() {
        start_key[group_id] = dfa.dfa().start_state;
    }

    let mut states = vec![NWAState::default()];
    let mut state_map = FxHashMap::<[u32; 8], u32>::default();
    state_map.insert(start_key, 0);
    let mut worklist = VecDeque::from([(0u32, start_key)]);
    let mut label_targets = LabelTargets::new();

    while let Some((from, product_state)) = worklist.pop_front() {
        let mut final_mask = 0u16;
        label_targets.clear();
        for group_id in 0..n {
            let dfa_state = product_state[group_id];
            if dfa_state == DEAD {
                continue;
            }
            let dfa = groups[group_id].1.dfa();
            let node = &dfa.states[dfa_state as usize];
            if node.is_accepting {
                final_mask |= 1u16 << group_id;
            }
            for (&label, &target) in &node.transitions {
                label_targets.push((label, group_id as u32, target));
            }
        }
        if final_mask != 0 {
            let weight = subset_weights[final_mask as usize].clone();
            if !weight.is_empty() {
                states[from as usize].final_weight = Some(weight);
            }
        }

        label_targets.sort_unstable_by_key(|&(label, group_id, _)| (label, group_id));
        let mut label_start = 0usize;
        while label_start < label_targets.len() {
            let label = label_targets[label_start].0;
            let mut label_end = label_start + 1;
            while label_end < label_targets.len() && label_targets[label_end].0 == label {
                label_end += 1;
            }
            let mut support_mask = 0u16;
            let mut next = [DEAD; 8];
            for &(_, group_id, target) in &label_targets[label_start..label_end] {
                support_mask |= 1u16 << group_id;
                next[group_id as usize] = target;
            }
            let edge_weight = subset_weights[support_mask as usize].clone();
            if !edge_weight.is_empty() {
                let target = if let Some(&existing) = state_map.get(&next) {
                    existing
                } else {
                    let new_id = checked_usize_to_u32(states.len(), "bundle NWA state id");
                    states.push(NWAState::default());
                    state_map.insert(next, new_id);
                    worklist.push_back((new_id, next));
                    new_id
                };
                states[from as usize]
                    .transitions
                    .insert(label, vec![(target, edge_weight)]);
            }
            label_start = label_end;
        }
    }

    Some(NWA::from_parts(states, vec![0]))
}
fn determinize_bundle_groups_serial_small_fixed(
    groups: &[(&Weight, BundleGroupDfa<'_>)],
) -> Option<DWA> {
    use crate::automata::weighted_u32::dwa::DWA;

    const DEAD: u32 = u32::MAX;
    let n = groups.len();
    if n == 0 || n > 8 {
        return None;
    }

    // The support semiring has at most 2^8 exact group unions. Compute each
    // once and keep product-state topology in a fixed key so the hot loop never
    // allocates Vec<(group,state)> keys merely to probe the state table.
    let subset_count = 1usize << n;
    let mut subset_weights = Vec::with_capacity(subset_count);
    subset_weights.push(Weight::empty());
    for mask in 1..subset_count {
        let bit = mask.trailing_zeros() as usize;
        let previous = mask & (mask - 1);
        subset_weights.push(subset_weights[previous].union(groups[bit].0));
    }

    let mut start_key = [DEAD; 8];
    for (group_id, (_, dfa)) in groups.iter().enumerate() {
        start_key[group_id] = dfa.dfa().start_state;
    }

    let mut dwa = DWA::new(0, 0);
    let mut state_map = FxHashMap::<[u32; 8], u32>::default();
    state_map.insert(start_key, 0);
    let mut worklist = VecDeque::from([(0u32, start_key)]);
    let mut label_targets = LabelTargets::new();

    while let Some((from, product_state)) = worklist.pop_front() {
        let mut final_mask = 0u16;
        label_targets.clear();
        for group_id in 0..n {
            let dfa_state = product_state[group_id];
            if dfa_state == DEAD {
                continue;
            }
            let dfa = groups[group_id].1.dfa();
            let node = &dfa.states[dfa_state as usize];
            if node.is_accepting {
                final_mask |= 1u16 << group_id;
            }
            for (&label, &target) in &node.transitions {
                label_targets.push((label, group_id as u32, target));
            }
        }
        if final_mask != 0 {
            let weight = subset_weights[final_mask as usize].clone();
            if !weight.is_empty() {
                dwa.set_final_weight(from, weight);
            }
        }

        label_targets.sort_unstable_by_key(|&(label, group_id, _)| (label, group_id));
        let mut label_start = 0usize;
        while label_start < label_targets.len() {
            let label = label_targets[label_start].0;
            let mut label_end = label_start + 1;
            while label_end < label_targets.len() && label_targets[label_end].0 == label {
                label_end += 1;
            }
            let mut support_mask = 0u16;
            let mut next = [DEAD; 8];
            for &(_, group_id, target) in &label_targets[label_start..label_end] {
                support_mask |= 1u16 << group_id;
                next[group_id as usize] = target;
            }
            let edge_weight = subset_weights[support_mask as usize].clone();
            if !edge_weight.is_empty() {
                let target = if let Some(&existing) = state_map.get(&next) {
                    existing
                } else {
                    let new_id = dwa.add_state();
                    state_map.insert(next, new_id);
                    worklist.push_back((new_id, next));
                    new_id
                };
                dwa.add_transition(from, label, target, edge_weight);
            }
            label_start = label_end;
        }
    }
    Some(dwa)
}

fn determinize_bundle_groups_serial_small(groups: &[(&Weight, BundleGroupDfa<'_>)]) -> Option<DWA> {
    use crate::automata::weighted_u32::dwa::DWA;

    let n = groups.len();
    if n == 0 || n > 8 {
        return None;
    }
    let subset_count = 1usize << n;
    let mut subset_weights = Vec::with_capacity(subset_count);
    subset_weights.push(Weight::empty());
    for mask in 1..subset_count {
        let bit = mask.trailing_zeros() as usize;
        let previous = mask & (mask - 1);
        subset_weights.push(subset_weights[previous].union(groups[bit].0));
    }

    let start_key = groups
        .iter()
        .enumerate()
        .map(|(group_id, (_, dfa))| {
            (
                checked_usize_to_u32(group_id, "bundle group id"),
                dfa.dfa().start_state,
            )
        })
        .collect::<Vec<_>>();

    let mut dwa = DWA::new(0, 0);
    let mut state_map = FxHashMap::<Vec<(u32, u32)>, u32>::default();
    let mut singleton_state_map = FxHashMap::<(u32, u32), u32>::default();
    if let [singleton] = start_key.as_slice() {
        singleton_state_map.insert(*singleton, 0);
    } else {
        state_map.insert(start_key.clone(), 0);
    }
    let mut worklist = VecDeque::from([(0u32, start_key)]);
    let mut label_targets = LabelTargets::new();

    while let Some((from, product_state)) = worklist.pop_front() {
        let mut final_mask = 0u16;
        for &(group_id, dfa_state) in &product_state {
            if groups[group_id as usize].1.dfa().states[dfa_state as usize].is_accepting {
                final_mask |= 1u16 << group_id;
            }
        }
        if final_mask != 0 {
            let weight = subset_weights[final_mask as usize].clone();
            if !weight.is_empty() {
                dwa.set_final_weight(from, weight);
            }
        }

        collect_label_targets(groups, &product_state, &mut label_targets);
        let mut label_start = 0usize;
        while label_start < label_targets.len() {
            let label = label_targets[label_start].0;
            let mut label_end = label_start + 1;
            while label_end < label_targets.len() && label_targets[label_end].0 == label {
                label_end += 1;
            }
            let mut support_mask = 0u16;
            let mut next = Vec::with_capacity(label_end - label_start);
            for &(_, group_id, target) in &label_targets[label_start..label_end] {
                support_mask |= 1u16 << group_id;
                next.push((group_id, target));
            }
            let edge_weight = subset_weights[support_mask as usize].clone();
            if !edge_weight.is_empty() {
                let target = if let [singleton] = next.as_slice() {
                    if let Some(&existing) = singleton_state_map.get(singleton) {
                        existing
                    } else {
                        let new_id = dwa.add_state();
                        singleton_state_map.insert(*singleton, new_id);
                        worklist.push_back((new_id, next));
                        new_id
                    }
                } else if let Some(&existing) = state_map.get(&next) {
                    existing
                } else {
                    let new_id = dwa.add_state();
                    state_map.insert(next.clone(), new_id);
                    worklist.push_back((new_id, next));
                    new_id
                };
                dwa.add_transition(from, label, target, edge_weight);
            }
            label_start = label_end;
        }
    }
    Some(dwa)
}

fn determinize_bundle_groups(groups: &[(&Weight, BundleGroupDfa<'_>)]) -> DWA {
    use crate::automata::weighted_u32::dwa::DWA;

    let n = groups.len();
    if n == 0 {
        return DWA::new(0, 0);
    }
    if std::env::var_os("GLRMASK_EXPERIMENT_SMALL_BUNDLE_FIXED_KEY").is_some()
        && let Some(dwa) = determinize_bundle_groups_serial_small_fixed(groups)
    {
        return dwa;
    }
    if std::env::var_os("GLRMASK_EXPERIMENT_SMALL_BUNDLE_SUBSET_WEIGHTS").is_some()
        && let Some(dwa) = determinize_bundle_groups_serial_small(groups)
    {
        return dwa;
    }
    if std::env::var_os("GLRMASK_EXPERIMENT_PARALLEL_BUNDLE_DETERMINIZE").is_some()
        && rayon::current_thread_index().is_none()
        && let Some(dwa) = determinize_bundle_groups_parallel_small(groups)
    {
        return dwa;
    }

    let group_weights: Vec<Weight> = groups
        .iter()
        .map(|(weight, _)| (*weight).clone())
        .collect();
    let single_tsid_entries = group_weights
        .iter()
        .map(Weight::single_tsid_shared_entry)
        .collect::<Option<Vec<_>>>();

    let mut subset_union_cache: FxHashMap<SubsetKey, Weight> = FxHashMap::default();
    let subset_unions = template_bundle_subset_union_index_enabled().then(|| SubsetUnionIndex::new(&group_weights));

    let start_key: Vec<(u32, u32)> = groups
        .iter()
        .enumerate()
        .map(|(group_id, (_, dfa))| {
            (
                checked_usize_to_u32(group_id, "bundle group id"),
                dfa.dfa().start_state,
            )
        })
        .collect();

    let mut dwa = DWA::new(0, 0);
    let mut state_map: FxHashMap<Vec<(u32, u32)>, u32> = FxHashMap::default();
    let mut singleton_state_map: FxHashMap<(u32, u32), u32> = FxHashMap::default();
    let mut worklist: VecDeque<(u32, Vec<(u32, u32)>)> = VecDeque::new();

    if let [singleton] = start_key.as_slice() {
        singleton_state_map.insert(*singleton, 0);
    } else {
        state_map.insert(start_key.clone(), 0);
    }
    worklist.push_back((0, start_key));

    let mut label_targets = LabelTargets::new();
    let key_words = n.div_ceil(64);
    let mut final_groups = SmallVec::<[usize; 8]>::new();
    let mut final_key = SubsetKey::from_elem(0, key_words);
    let mut edge_groups = SmallVec::<[usize; 8]>::new();
    let mut edge_key = SubsetKey::from_elem(0, key_words);

    while let Some((dwa_state, product_state)) = worklist.pop_front() {

        final_groups.clear();
        clear_subset_key(&mut final_key);
        for &(group_id, dfa_state) in &product_state {
            let group_id = group_id as usize;
            if groups[group_id].1.dfa().states[dfa_state as usize].is_accepting {
                final_groups.push(group_id);
                set_subset_key_bit(&mut final_key, group_id);
            }
        }
        let final_w = cached_subset_union(
            &mut subset_union_cache,
            &final_key,
            &final_groups,
            &group_weights,
            subset_unions.as_ref(),
            single_tsid_entries.as_deref(),
        );
        if !final_w.is_empty() {
            dwa.set_final_weight(dwa_state, final_w);
        }

        collect_label_targets(groups, &product_state, &mut label_targets);

        let mut label_start = 0usize;
        while label_start < label_targets.len() {
            let label = label_targets[label_start].0;
            let mut label_end = label_start + 1;
            while label_end < label_targets.len() && label_targets[label_end].0 == label {
                label_end += 1;
            }

            edge_groups.clear();
            clear_subset_key(&mut edge_key);
            let singleton_target = (label_end == label_start + 1).then(|| {
                let (_, group_id, target) = label_targets[label_start];
                (group_id, target)
            });
            let mut next_state = singleton_target
                .is_none()
                .then(|| Vec::with_capacity(label_end - label_start));
            for &(_, group_id, target) in &label_targets[label_start..label_end] {
                let group_id = group_id as usize;
                edge_groups.push(group_id);
                set_subset_key_bit(&mut edge_key, group_id);
                if let Some(next_state) = next_state.as_mut() {
                    next_state.push((group_id as u32, target));
                }
            }

            let edge_w = cached_subset_union(
                &mut subset_union_cache,
                &edge_key,
                &edge_groups,
                &group_weights,
                subset_unions.as_ref(),
                single_tsid_entries.as_deref(),
            );
            if edge_w.is_empty() {
                label_start = label_end;
                continue;
            }

            let to_dwa = if let Some(singleton_target) = singleton_target {
                if let Some(&existing) = singleton_state_map.get(&singleton_target) {
                    existing
                } else {
                    let new_id = dwa.add_state();
                    singleton_state_map.insert(singleton_target, new_id);
                    worklist.push_back((new_id, vec![singleton_target]));
                    new_id
                }
            } else {
                let next_state = next_state.expect("non-singleton bundle state is populated");
                if let Some(&existing) = state_map.get(&next_state) {
                    existing
                } else {
                    let new_id = dwa.add_state();
                    state_map.insert(next_state.clone(), new_id);
                    worklist.push_back((new_id, next_state));
                    new_id
                }
            };

            dwa.add_transition(dwa_state, label, to_dwa, edge_w);
            label_start = label_end;
        }
    }

    dwa
}

/// Union multiple unweighted DFAs into one DFA via NFA union + determinize + minimize.
fn union_unweighted_dfas<'a>(dfas: impl Iterator<Item = &'a UnweightedDfa>) -> UnweightedDfa {
    let mut nfa = UnweightedNfa::new_empty();
    let shared_start = nfa.add_state();
    nfa.start_states.push(shared_start);

    for dfa in dfas {
        if dfa.states.is_empty() {
            continue;
        }
        let offset = checked_usize_to_u32(nfa.states.len(), "bundle-union NFA offset");
        for _ in &dfa.states {
            nfa.add_state();
        }
        // Epsilon from shared start to this DFA's start.
        nfa.add_epsilon(
            shared_start,
            checked_u32_add(offset, dfa.start_state, "bundle-union start state"),
        );
        for (state_id, state) in dfa.states.iter().enumerate() {
            let from = checked_u32_add(
                offset,
                checked_usize_to_u32(state_id, "bundle-union DFA state id"),
                "bundle-union source state",
            );
            if state.is_accepting {
                nfa.set_accepting(from);
            }
            for (&label, &target) in &state.transitions {
                nfa.add_transition(
                    from,
                    label,
                    checked_u32_add(offset, target, "bundle-union target state"),
                );
            }
        }
    }

    let det = unweighted_determinize(&nfa);
    unweighted_minimize(&det)
}

fn dwa_to_nwa(dwa: &DWA) -> NWA {
    let states = dwa
        .states()
        .iter()
        .map(|state| NWAState {
            final_weight: state.final_weight.clone(),
            transitions: state
                .transitions
                .iter()
                .map(|(&label, (target, weight))| (label, vec![(*target, weight.clone())]))
                .collect(),
            epsilons: Vec::new(),
        })
        .collect();

    NWA::from_parts(
        states,
        vec![dwa.start_state()],
    )
}

#[cfg(test)]
mod tests {
    use range_set_blaze::RangeSetBlaze;

    use super::*;

    fn weight(tokens: std::ops::RangeInclusive<u32>) -> Weight {
        Weight::from_token_set_for_tsid(0, RangeSetBlaze::from_iter([tokens]))
    }

    fn eval_bundle_product(
        groups: &[(&Weight, BundleGroupDfa<'_>)],
        word: &[i32],
    ) -> Weight {
        let mut product_state = groups
            .iter()
            .enumerate()
            .map(|(group_id, (_, dfa))| (group_id, dfa.dfa().start_state))
            .collect::<Vec<_>>();
        let mut accumulated = Weight::all();

        for &label in word {
            let mut next_state = Vec::new();
            let mut edge_weights = SmallVec::<[&Weight; 4]>::new();
            for &(group_id, dfa_state) in &product_state {
                let dfa = groups[group_id].1.dfa();
                let Some(&target) = dfa.states[dfa_state as usize].transitions.get(&label) else {
                    continue;
                };
                next_state.push((group_id, target));
                edge_weights.push(groups[group_id].0);
            }
            let edge_weight = Weight::union_all(edge_weights);
            accumulated = accumulated.intersection(&edge_weight);
            if accumulated.is_empty() {
                return accumulated;
            }
            product_state = next_state;
        }

        let final_weight = Weight::union_all(product_state.iter().filter_map(
            |&(group_id, dfa_state)| {
                groups[group_id].1.dfa().states[dfa_state as usize]
                    .is_accepting
                    .then_some(groups[group_id].0)
            },
        ));
        accumulated.intersection(&final_weight)
    }

    fn visit_words(
        alphabet: &[i32],
        remaining: usize,
        word: &mut Vec<i32>,
        visit: &mut impl FnMut(&[i32]),
    ) {
        visit(word);
        if remaining == 0 {
            return;
        }
        for &label in alphabet {
            word.push(label);
            visit_words(alphabet, remaining - 1, word, visit);
            word.pop();
        }
    }

    #[test]
    fn repeated_group_dfa_cache_preserves_bundle_structure() {
        fn template(first: i32, second: i32) -> UnweightedDfa {
            let mut dfa = UnweightedDfa::new();
            let middle = dfa.add_state();
            let accept = dfa.add_state();
            dfa.add_transition(0, first, middle);
            dfa.add_transition(middle, second, accept);
            dfa.set_accepting(accept, true);
            dfa
        }

        let templates = Templates {
            by_terminal: BTreeMap::from([
                (0, template(1, 10)),
                (1, template(1, 11)),
                (2, template(2, 12)),
            ]),
            by_terminal_nwa: BTreeMap::new(),
        };
        let first_shared = weight(0..=7);
        let first_other = weight(8..=15);
        let second_shared = weight(16..=23);
        let second_other = weight(24..=31);
        let first = BTreeMap::from([
            (0, first_shared.clone()),
            (1, first_shared),
            (2, first_other),
        ]);
        let second = BTreeMap::from([
            (0, second_shared.clone()),
            (1, second_shared),
            (2, second_other),
        ]);
        let cache = templates.build_bundle_group_dfa_cache(&[&first, &second]);
        assert_eq!(cache.len(), 1, "only terminal group [0,1] repeats");

        for bundle in [&first, &second] {
            let uncached = templates.build_bundle(bundle);
            let cached = templates.build_bundle_cached(bundle, &cache);
            assert_eq!(cached.start_states(), uncached.start_states());
            assert_eq!(cached.states(), uncached.states());
            let (_, profile) = templates.build_bundle_profiled_cached(bundle, &cache);
            assert_eq!(profile.group_dfa_cache_hits, 1);
            assert_eq!(profile.group_dfa_cache_misses, 0);
        }
    }

    #[test]
    #[ignore]
    fn repeated_group_dfa_cache_benchmark_probe() {
        fn long_template(terminal: i32) -> UnweightedDfa {
            let mut dfa = UnweightedDfa::new();
            let mut state = 0u32;
            for depth in 0..30i32 {
                let next = dfa.add_state();
                dfa.add_transition(state, 1000 + depth, next);
                state = next;
            }
            let accept = dfa.add_state();
            dfa.add_transition(state, 2000 + terminal, accept);
            dfa.set_accepting(accept, true);
            dfa
        }

        let templates = Templates {
            by_terminal: (0..16u32)
                .map(|terminal| (terminal, long_template(terminal as i32)))
                .collect(),
            by_terminal_nwa: BTreeMap::new(),
        };
        let bundles = (0..20u32)
            .map(|index| {
                let shared = weight((index * 32)..=(index * 32 + 15));
                (0..16u32)
                    .map(|terminal| (terminal, shared.clone()))
                    .collect::<BTreeMap<_, _>>()
            })
            .collect::<Vec<_>>();
        let refs = bundles.iter().collect::<Vec<_>>();

        let uncached_started = Instant::now();
        let mut uncached_union_ms = 0.0;
        for bundle in &bundles {
            let (_, profile) = templates.build_bundle_profiled(bundle);
            uncached_union_ms += profile.union_groups_ms;
        }
        let uncached_ms = elapsed_ms(uncached_started);

        let cache_started = Instant::now();
        let cache = templates.build_bundle_group_dfa_cache(&refs);
        let cache_build_ms = elapsed_ms(cache_started);
        assert_eq!(cache.len(), 1);
        let cached_started = Instant::now();
        let mut cached_union_ms = 0.0;
        let mut cache_hits = 0usize;
        for bundle in &bundles {
            let (_, profile) = templates.build_bundle_profiled_cached(bundle, &cache);
            cached_union_ms += profile.union_groups_ms;
            cache_hits += profile.group_dfa_cache_hits;
        }
        let cached_ms = elapsed_ms(cached_started);
        assert_eq!(cache_hits, bundles.len());
        assert_eq!(cached_union_ms, 0.0);
        eprintln!(
            "repeated group DFA cache benchmark: bundles={} terminals_per_group=16 uncached_ms={uncached_ms:.3} uncached_union_ms={uncached_union_ms:.3} cache_build_ms={cache_build_ms:.3} cached_bundle_ms={cached_ms:.3} cached_total_ms={:.3}",
            bundles.len(),
            cache_build_ms + cached_ms,
        );
    }

    #[test]
    fn collect_label_targets_is_sorted() {
        let mut first = UnweightedDfa::new();
        first.add_transition(0, 2, 20);
        first.add_transition(0, 1, 10);
        let mut second = UnweightedDfa::new();
        second.add_transition(0, 3, 31);
        second.add_transition(0, 1, 11);
        let weight = Weight::all();
        let groups = vec![
            (&weight, BundleGroupDfa::Owned(first)),
            (&weight, BundleGroupDfa::Owned(second)),
        ];
        let state = vec![(0, 0), (1, 0)];
        let mut targets = LabelTargets::new();

        collect_label_targets(&groups, &state, &mut targets);

        assert_eq!(targets.as_slice(), &[(1, 0, 10), (1, 1, 11), (2, 0, 20), (3, 1, 31)]);
    }

    #[test]
    fn singleton_bundle_state_interner_preserves_weighted_product() {
        let mut first = UnweightedDfa::new();
        let first_after_shared = first.add_state();
        let first_after_singleton = first.add_state();
        first.add_transition(0, 1, first_after_shared);
        first.add_transition(first_after_shared, 2, first_after_singleton);
        first.add_transition(first_after_shared, 5, first_after_shared);
        first.set_accepting(first_after_shared, true);
        first.set_accepting(first_after_singleton, true);

        let mut second = UnweightedDfa::new();
        let second_after_shared = second.add_state();
        let second_after_singleton = second.add_state();
        second.add_transition(0, 1, second_after_shared);
        second.add_transition(second_after_shared, 3, second_after_singleton);
        second.add_transition(second_after_shared, 5, second_after_shared);
        second.set_accepting(second_after_singleton, true);

        let mut third = UnweightedDfa::new();
        let third_after_singleton = third.add_state();
        third.add_transition(0, 4, third_after_singleton);
        third.set_accepting(third_after_singleton, true);

        let first_weight = weight(0..=15);
        let second_weight = weight(8..=23);
        let third_weight = weight(20..=31);
        let groups = vec![
            (&first_weight, BundleGroupDfa::Owned(first)),
            (&second_weight, BundleGroupDfa::Owned(second)),
            (&third_weight, BundleGroupDfa::Owned(third)),
        ];

        let optimized = determinize_bundle_groups(&groups);
        let (profiled, _) = determinize_bundle_groups_profiled(&groups);
        let pool = |threads| rayon::ThreadPoolBuilder::new()
            .num_threads(threads).build().expect("test thread pool");
        assert!(pool(1).install(|| determinize_bundle_groups_parallel_small(&groups)).is_none());
        let parallel = pool(2).install(|| determinize_bundle_groups_parallel_small(&groups))
            .expect("small bundle must support parallel determinization with two workers");
        let mut word = Vec::new();
        visit_words(&[1, 2, 3, 4, 5], 4, &mut word, &mut |word| {
            let expected = eval_bundle_product(&groups, word);
            assert_eq!(optimized.eval_word(word), expected, "optimized word={word:?}");
            assert_eq!(profiled.eval_word(word), expected, "profiled word={word:?}");
            assert_eq!(parallel.eval_word(word), expected, "parallel word={word:?}");
        });
    }
}
