#[path = "finite_read_support.rs"]
mod finite_read_support;
pub use finite_read_support::FiniteParserReadSupport;

use std::collections::{hash_map::Entry, BTreeMap, BTreeSet, VecDeque};
use std::sync::Arc;
use std::hash::{Hash, Hasher};
use std::time::Instant;

use rustc_hash::{FxHashMap, FxHashSet, FxHasher};
use smallvec::SmallVec;

use crate::Vocab;
use crate::automata::weighted::dwa::{DWA, DWAState};
use crate::automata::weighted::equivalence::find_difference;
use crate::automata::weighted::minimize::minimize;
use crate::automata::weighted::nwa::{NWA, NWAState, NwaBody};
use crate::automata::weighted::terminal_automaton::TerminalAutomaton;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::glr::labels::{DEFAULT_LABEL, is_negative_label, negative_to_positive_label};
use crate::compiler::glr::table::{
    Action, AdmissionPolicy, GLRTable, GlrTableConstruction,
};
use crate::grammar::flat::TerminalID;
use crate::compiler::stages::equiv_types::InternalIdMap;
use crate::compiler::stages::resolve_negatives::{
    apply_finality_fixpoint, resolve_negative_codes_in_nwa,
};
use crate::compiler::stages::templates::Templates;
use crate::ds::bitset::BitSet;
use crate::ds::weight::{ScopedWeightOpCache, Weight};

fn compile_profile_enabled() -> bool {
    std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
}

type TerminalBundle = BTreeMap<TerminalID, Weight>;
type BundleSignature = Vec<(TerminalID, Weight)>;
type TargetContribs = SmallVec<[(u32, Weight); 4]>;
type DeferredFinalEntries = SmallVec<[(u32, Weight); 4]>;
type FinalPathWeights = SmallVec<[Weight; 4]>;
type FinalGroups = SmallVec<[(Weight, FinalPathWeights); 4]>;

struct ParallelSupportScanScratch {
    weight_ops: ScopedWeightOpCache,
    sparse_only: bool,
    flat_scan: bool,
    flat: Vec<(i32, u32, Weight)>,
    dense: Vec<TargetContribs>,
    dense_touched: Vec<bool>,
    touched_dense: Vec<usize>,
    default: TargetContribs,
    sparse: FxHashMap<i32, TargetContribs>,
}

impl ParallelSupportScanScratch {
    fn new(dense_label_limit: usize) -> Self {
        let sparse_only = std::env::var_os("GLRMASK_EXPERIMENT_PARSER_SUPPORT_SPARSE_SCAN").is_some();
        let flat_scan = std::env::var_os("GLRMASK_EXPERIMENT_PARSER_SUPPORT_PARALLEL_FLAT_SCAN").is_some();
        Self {
            weight_ops: ScopedWeightOpCache::default(),
            sparse_only,
            flat_scan,
            flat: Vec::with_capacity(64),
            dense: if sparse_only || flat_scan {
                Vec::new()
            } else {
                (0..dense_label_limit).map(|_| TargetContribs::new()).collect()
            },
            dense_touched: if sparse_only || flat_scan {
                Vec::new()
            } else {
                vec![false; dense_label_limit]
            },
            touched_dense: Vec::new(),
            default: TargetContribs::new(),
            sparse: FxHashMap::default(),
        }
    }

    #[inline]
    fn push(&mut self, label: i32, target: u32, weight: Weight) {
        if self.flat_scan {
            self.flat.push((label, target, weight));
        } else if !self.sparse_only && label >= 0 && (label as usize) < self.dense.len() {
            let index = label as usize;
            if !self.dense_touched[index] {
                self.dense_touched[index] = true;
                self.touched_dense.push(index);
            }
            self.dense[index].push((target, weight));
        } else if label == DEFAULT_LABEL {
            self.default.push((target, weight));
        } else {
            self.sparse.entry(label).or_default().push((target, weight));
        }
    }

    fn take_labels(&mut self) -> Vec<(i32, TargetContribs)> {
        if self.flat_scan {
            self.flat.sort_unstable_by_key(|(label, target, _)| (*label, *target));
            let mut labels = Vec::<(i32, TargetContribs)>::new();
            for (label, target, weight) in self.flat.drain(..) {
                if let Some((last_label, contribs)) = labels.last_mut()
                    && *last_label == label
                {
                    contribs.push((target, weight));
                } else {
                    let mut contribs = TargetContribs::new();
                    contribs.push((target, weight));
                    labels.push((label, contribs));
                }
            }
            return labels;
        }
        let mut labels = Vec::with_capacity(
            self.touched_dense.len() + usize::from(!self.default.is_empty()) + self.sparse.len(),
        );
        for index in self.touched_dense.drain(..) {
            self.dense_touched[index] = false;
            labels.push((index as i32, std::mem::take(&mut self.dense[index])));
        }
        if !self.default.is_empty() {
            labels.push((DEFAULT_LABEL, std::mem::take(&mut self.default)));
        }
        labels.extend(self.sparse.drain());
        labels
    }
}

const PROFILE_PARSER_DWA_DETERMINIZE_DETAIL_ENV: &str =
    "GLRMASK_PROFILE_PARSER_DWA_DETERMINIZE_DETAIL";

#[inline]
fn add_target_contribution(contribs: &mut TargetContribs, target: u32, add: Weight) {
    if add.is_empty() {
        return;
    }

    if let Some((_, existing)) = contribs.iter_mut().find(|(existing_target, _)| *existing_target == target) {
        *existing = existing.union(&add);
    } else {
        contribs.push((target, add));
    }
}

#[derive(Default)]
struct ParserDwaDeterminizeDetail {
    states_processed: usize,
    outgoing_transitions_scanned: usize,
    intersection_calls: usize,
    nonempty_intersections: usize,
    target_contribution_pushes: usize,
    target_contribution_merges: usize,
    target_contrib_len_before_sum: usize,
    target_contrib_len_after_sum: usize,
    target_contrib_len_before_max: usize,
    target_contrib_len_after_max: usize,
    subset_key_constructions: usize,
    subset_intern_hits: usize,
    subset_intern_misses: usize,
    closure_cache_hits: usize,
    closure_cache_misses: usize,
    intersection_scan_ms: f64,
    label_processing_ms: f64,
    labels_processed: usize,
    label_contribs_sum: usize,
    label_contribs_max: usize,
    contribution_sort_ms: f64,
    edge_weight_union_ms: f64,
    closure_key_ms: f64,
    closure_lookup_ms: f64,
    local_epsilon_closure_miss_ms: f64,
    post_closure_subset_key_ms: f64,
    subset_map_lookup_ms: f64,
    add_transition_ms: f64,
    final_weight_states: usize,
    final_weight_entries: usize,
    final_weight_entries_max: usize,
    final_weight_signature_distinct: usize,
    final_weight_signature_hit_potential: usize,
    final_grouping_ms: f64,
    final_path_union_ms: f64,
    final_intersection_ms: f64,
    final_output_union_ms: f64,
    union_cache_hits: usize,
    union_cache_misses: usize,
    union_cache_key_len_sum: usize,
    union_cache_key_len_max: usize,
    union_cache_ms: f64,
    fallback_labels_expanded: usize,
    fallback_contrib_entries_duplicated: usize,
}

impl ParserDwaDeterminizeDetail {
    fn enabled() -> bool {
        std::env::var(PROFILE_PARSER_DWA_DETERMINIZE_DETAIL_ENV)
            .map(|value| {
                let normalized = value.trim().to_ascii_lowercase();
                matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
            })
            .unwrap_or(false)
    }

    fn record_target_contrib_len(&mut self, before: usize, after: usize) {
        self.target_contrib_len_before_sum += before;
        self.target_contrib_len_after_sum += after;
        self.target_contrib_len_before_max = self.target_contrib_len_before_max.max(before);
        self.target_contrib_len_after_max = self.target_contrib_len_after_max.max(after);
    }

    fn emit(&self, name: &str) {
        eprintln!(
            "[glrmask/profile][parser_dwa_determinize_detail] name={} states_processed={} outgoing_transitions_scanned={} intersection_calls={} nonempty_intersections={} target_contribution_pushes={} target_contribution_merges={} target_contrib_len_before_sum={} target_contrib_len_after_sum={} target_contrib_len_before_max={} target_contrib_len_after_max={} subset_key_constructions={} subset_intern_hits={} subset_intern_misses={} closure_cache_hits={} closure_cache_misses={} intersection_scan_ms={:.3} label_processing_ms={:.3} final_weight_states={} final_weight_entries={} final_weight_entries_max={} final_weight_signature_distinct={} final_weight_signature_hit_potential={} fallback_labels_expanded={} fallback_contrib_entries_duplicated={}",
            name,
            self.states_processed,
            self.outgoing_transitions_scanned,
            self.intersection_calls,
            self.nonempty_intersections,
            self.target_contribution_pushes,
            self.target_contribution_merges,
            self.target_contrib_len_before_sum,
            self.target_contrib_len_after_sum,
            self.target_contrib_len_before_max,
            self.target_contrib_len_after_max,
            self.subset_key_constructions,
            self.subset_intern_hits,
            self.subset_intern_misses,
            self.closure_cache_hits,
            self.closure_cache_misses,
            self.intersection_scan_ms,
            self.label_processing_ms,
            self.final_weight_states,
            self.final_weight_entries,
            self.final_weight_entries_max,
            self.final_weight_signature_distinct,
            self.final_weight_signature_hit_potential,
            self.fallback_labels_expanded,
            self.fallback_contrib_entries_duplicated,
        );
        eprintln!(
            "[glrmask/profile][parser_dwa_determinize_fine] name={} labels_processed={} label_contribs_sum={} label_contribs_max={} contribution_sort_ms={:.3} edge_weight_union_ms={:.3} closure_key_ms={:.3} closure_lookup_ms={:.3} local_epsilon_closure_miss_ms={:.3} post_closure_subset_key_ms={:.3} subset_map_lookup_ms={:.3} add_transition_ms={:.3} final_grouping_ms={:.3} final_path_union_ms={:.3} final_intersection_ms={:.3} final_output_union_ms={:.3} union_cache_hits={} union_cache_misses={} union_cache_key_len_sum={} union_cache_key_len_max={} union_cache_ms={:.3}",
            name,
            self.labels_processed,
            self.label_contribs_sum,
            self.label_contribs_max,
            self.contribution_sort_ms,
            self.edge_weight_union_ms,
            self.closure_key_ms,
            self.closure_lookup_ms,
            self.local_epsilon_closure_miss_ms,
            self.post_closure_subset_key_ms,
            self.subset_map_lookup_ms,
            self.add_transition_ms,
            self.final_grouping_ms,
            self.final_path_union_ms,
            self.final_intersection_ms,
            self.final_output_union_ms,
            self.union_cache_hits,
            self.union_cache_misses,
            self.union_cache_key_len_sum,
            self.union_cache_key_len_max,
            self.union_cache_ms,
        );
    }
}

#[inline]
fn add_target_contribution_profiled(
    contribs: &mut TargetContribs,
    target: u32,
    add: Weight,
    mut detail: Option<&mut ParserDwaDeterminizeDetail>,
) {
    if detail.is_none() {
        add_target_contribution(contribs, target, add);
        return;
    }

    if add.is_empty() {
        return;
    }

    let before = contribs.len();
    if let Some((_, existing)) = contribs
        .iter_mut()
        .find(|(existing_target, _)| *existing_target == target)
    {
        *existing = existing.union(&add);
        if let Some(detail) = detail.as_mut() {
            detail.target_contribution_merges += 1;
        }
    } else {
        contribs.push((target, add));
        if let Some(detail) = detail.as_mut() {
            detail.target_contribution_pushes += 1;
        }
    }
    if let Some(detail) = detail {
        detail.record_target_contrib_len(before, contribs.len());
    }
}

#[inline]
fn push_target_contribution_profiled(
    contribs: &mut TargetContribs,
    target: u32,
    add: Weight,
    detail: Option<&mut ParserDwaDeterminizeDetail>,
) {
    if add.is_empty() {
        return;
    }
    let before = contribs.len();
    contribs.push((target, add));
    if let Some(detail) = detail {
        detail.target_contribution_pushes += 1;
        detail.record_target_contrib_len(before, contribs.len());
    }
}

#[inline]
fn merge_sorted_target_contributions(
    contribs: &mut TargetContribs,
    mut detail: Option<&mut ParserDwaDeterminizeDetail>,
) {
    if contribs.len() < 2 {
        return;
    }
    let mut write = 0usize;
    for read in 1..contribs.len() {
        if contribs[write].0 == contribs[read].0 {
            let merged = contribs[write].1.union(&contribs[read].1);
            contribs[write].1 = merged;
            if let Some(detail) = detail.as_mut() {
                detail.target_contribution_merges += 1;
            }
        } else {
            write += 1;
            if write != read {
                contribs[write] = contribs[read].clone();
            }
        }
    }
    contribs.truncate(write + 1);
}

fn extend_target_contribs(dst: &mut TargetContribs, src: &TargetContribs) {
    for (target, weight) in src {
        add_target_contribution(dst, *target, weight.clone());
    }
}

#[derive(Debug, Clone, Copy)]
struct Branch {
    target: u32,
    bundle_id: usize,
}

#[derive(Debug, Clone)]
struct StateSummary {
    final_weight: Option<Weight>,
    epsilon_branches: Vec<(u32, Weight)>,
    branches: Vec<Branch>,
}

#[derive(Debug, Clone)]
struct StateSummaries {
    states: Vec<StateSummary>,
    start_states: Vec<u32>,
    unique_bundles: Vec<TerminalBundle>,
    bundle_accepts: Vec<bool>,
}

#[derive(Clone, Default)]
pub struct PrebuiltParserBundleCache {
    by_signature: FxHashMap<BundleSignature, Arc<NWA>>,
}

impl PrebuiltParserBundleCache {
    pub fn len(&self) -> usize {
        self.by_signature.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_signature.is_empty()
    }
}

#[derive(Debug, Clone)]
struct DeterminizedDwaWithSupports {
    dwa: DWA,
    supports: Vec<Vec<u32>>,
}

#[derive(Debug, Clone)]
struct CachedClosure {
    to_state: u32,
    edge_weight: Weight,
}

struct ParserSingletonSubsetCache {
    primary: Option<Vec<Option<(usize, u32)>>>,
    overflow: FxHashMap<(u32, usize), u32>,
}

impl ParserSingletonSubsetCache {
    fn new(num_nwa_states: usize) -> Self {
        let dense = num_nwa_states >= std::env::var("GLRMASK_PARSER_SINGLETON_DENSE_MIN_NWA_STATES")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(16_384)
            && std::env::var_os("GLRMASK_DISABLE_PARSER_SINGLETON_DENSE_CACHE").is_none();
        Self {
            primary: dense.then(|| vec![None; num_nwa_states]),
            overflow: FxHashMap::default(),
        }
    }

    #[inline]
    fn get(&self, state: u32, weight_key: usize) -> Option<u32> {
        if let Some(primary) = &self.primary
            && let Some((primary_weight, dwa_state)) = primary[state as usize]
            && primary_weight == weight_key
        {
            return Some(dwa_state);
        }
        self.overflow.get(&(state, weight_key)).copied()
    }

    #[inline]
    fn insert(&mut self, state: u32, weight_key: usize, dwa_state: u32) {
        if let Some(primary) = &mut self.primary {
            let slot = &mut primary[state as usize];
            match *slot {
                None => {
                    *slot = Some((weight_key, dwa_state));
                    return;
                }
                Some((existing_weight, _)) if existing_weight == weight_key => {
                    *slot = Some((weight_key, dwa_state));
                    return;
                }
                Some(_) => {}
            }
        }
        self.overflow.insert((state, weight_key), dwa_state);
    }
}

fn elapsed_ms(started_at: Instant) -> f64 {
    started_at.elapsed().as_secs_f64() * 1000.0
}

fn skip_parser_dwa_minimization_env_override() -> Option<bool> {
    std::env::var("GLRMASK_SKIP_PARSER_DWA_MINIMIZE")
        .ok()
        .map(|value| {
            let trimmed = value.trim();
            !(trimmed.is_empty()
                || trimmed == "0"
                || trimmed.eq_ignore_ascii_case("false"))
        })
}

#[inline]
fn should_skip_parser_dwa_minimization(
    _pre_minimize_states: usize,
    _pre_minimize_transitions: usize,
) -> bool {
    // Parser-DWA minimization is behavior-preserving but comparatively expensive
    // on the large-schema tail path.  The preceding construction already shares
    // continuation subgraphs and applies default/fallback normalization, so the
    // unminimized DWA is small enough for the runtime fast-transition cache.
    // Keep an escape hatch for size-sensitive experiments.
    skip_parser_dwa_minimization_env_override().unwrap_or(true)
}

#[derive(Default)]
struct ParserNwaBuildProfile {
    state_prep_ms: f64,
    compose_state_ms: f64,
    parser_nwa_build_ms: f64,
}

#[derive(Default)]
struct ParserDwaComposeDetailProfile {
    total_states: usize,
    productive_states: usize,
    total_branches: usize,
    productive_branches: usize,
    unique_bundles: usize,
    accepting_bundles: usize,
    state_init_ms: f64,
    branch_walk_ms: f64,
    memo_hit_clone_ms: f64,
    fragment_build_ms: f64,
    epsilon_link_ms: f64,
    bundle_profile_total_ms: f64,
    bundle_profile_build_group_dfas_ms: f64,
    bundle_profile_union_groups_ms: f64,
    bundle_profile_determinize_ms: f64,
    bundle_profile_minimize_ms: f64,
    bundle_profile_dwa_to_nwa_ms: f64,
    memo_hits: usize,
    memo_misses: usize,
    bundle_cache_builds: usize,
    bundle_profile_result_dwa_states: usize,
    bundle_profile_result_dwa_transitions: usize,
    bundle_profile_result_nwa_states: usize,
    bundle_profile_result_nwa_transitions: usize,
    epsilon_edges_added: usize,
    fragment_start_states_total: usize,
}

fn parser_dwa_compose_detail_enabled() -> bool {
    std::env::var("GLRMASK_PROFILE_PARSER_DWA_COMPOSE_DETAIL")
        .map(|value| value == "1")
        .unwrap_or(false)
}

fn group_terminal_edges_by_target(
    terminal_automaton: &TerminalAutomaton,
    num_terminals: u32,
    state_id: u32,
) -> BTreeMap<u32, TerminalBundle> {
    let mut bundles_by_target = BTreeMap::<u32, TerminalBundle>::new();
    let mut add = |target: u32, label: i32, weight: &Weight| {
        if label < 0 || label as u32 >= num_terminals || weight.is_empty() {
            return;
        }
        bundles_by_target
            .entry(target)
            .or_default()
            .entry(label as TerminalID)
            .and_modify(|existing| *existing = existing.union(weight))
            .or_insert_with(|| weight.clone());
    };

    match terminal_automaton {
        TerminalAutomaton::Dwa(dwa) => {
            let Some(state) = dwa.states().get(state_id as usize) else {
                return bundles_by_target;
            };
            for (&label, (target, weight)) in &state.transitions {
                add(*target, label, weight);
            }
        }
        TerminalAutomaton::TokenDeterministicNwa(nwa) => {
            let Some(state) = nwa.states().get(state_id as usize) else {
                return bundles_by_target;
            };
            assert!(
                state.epsilons.is_empty(),
                "token-deterministic terminal NWA must not contain epsilon edges",
            );
            for (&label, branches) in &state.transitions {
                for (target, weight) in branches {
                    add(*target, label, weight);
                }
            }
        }
        TerminalAutomaton::EpsilonNwa(nwa) => {
            let Some(state) = nwa.states().get(state_id as usize) else {
                return bundles_by_target;
            };
            for (&label, branches) in &state.transitions {
                for (target, weight) in branches {
                    add(*target, label, weight);
                }
            }
        }
    }

    bundles_by_target
}

fn terminal_state_final_weight(
    terminal_automaton: &TerminalAutomaton,
    state_id: usize,
) -> Option<Weight> {
    match terminal_automaton {
        TerminalAutomaton::Dwa(dwa) => dwa
            .states()
            .get(state_id)
            .and_then(|state| state.final_weight.clone()),
        TerminalAutomaton::TokenDeterministicNwa(nwa)
        | TerminalAutomaton::EpsilonNwa(nwa) => nwa
            .states()
            .get(state_id)
            .and_then(|state| state.final_weight.clone()),
    }
}

fn terminal_state_epsilon_branches(
    terminal_automaton: &TerminalAutomaton,
    state_id: usize,
) -> Vec<(u32, Weight)> {
    match terminal_automaton {
        TerminalAutomaton::EpsilonNwa(nwa) => nwa
            .states()
            .get(state_id)
            .map(|state| state.epsilons.clone())
            .unwrap_or_default(),
        TerminalAutomaton::Dwa(_) | TerminalAutomaton::TokenDeterministicNwa(_) => Vec::new(),
    }
}

fn bundle_signature(bundle: &TerminalBundle) -> BundleSignature {
    bundle
        .iter()
        .map(|(&terminal, weight)| (terminal, weight.clone()))
        .collect()
}

fn terminal_template_has_acceptance(template: &NWA) -> bool {
    template.states().iter().any(|state| state.final_weight.is_some())
}

fn terminal_bundle_has_acceptance(bundle: &TerminalBundle, templates: &Templates) -> bool {
    bundle.iter().any(|(&terminal, weight)| {
        !weight.is_empty()
            && templates
                .by_terminal_nwa
                .get(&terminal)
                .is_some_and(terminal_template_has_acceptance)
    })
}

fn build_state_summaries(
    terminal_automaton: &TerminalAutomaton,
    num_terminals: u32,
    templates: &Templates,
) -> StateSummaries {
    let state_count = terminal_automaton.num_states();
    let mut branches_by_state: Vec<Vec<Branch>> = Vec::with_capacity(state_count);
    let mut bundle_ids_by_signature: FxHashMap<BundleSignature, usize> = FxHashMap::default();
    let mut unique_bundles: Vec<TerminalBundle> = Vec::new();

    for state_id in 0..state_count {
        let bundles_by_target =
            group_terminal_edges_by_target(terminal_automaton, num_terminals, state_id as u32);
        let mut branches = Vec::with_capacity(bundles_by_target.len());
        for (target, bundle) in bundles_by_target {
            let signature = bundle_signature(&bundle);
            let bundle_id = if let Some(&bundle_id) = bundle_ids_by_signature.get(&signature) {
                bundle_id
            } else {
                let bundle_id = unique_bundles.len();
                bundle_ids_by_signature.insert(signature, bundle_id);
                unique_bundles.push(bundle);
                bundle_id
            };
            branches.push(Branch { target, bundle_id });
        }
        branches_by_state.push(branches);
    }

    let bundle_accepts: Vec<bool> = unique_bundles
        .iter()
        .map(|bundle| terminal_bundle_has_acceptance(bundle, templates))
        .collect();

    let states = (0..state_count)
        .map(|state_id| StateSummary {
            final_weight: terminal_state_final_weight(terminal_automaton, state_id),
            epsilon_branches: terminal_state_epsilon_branches(terminal_automaton, state_id),
            branches: std::mem::take(&mut branches_by_state[state_id]),
        })
        .collect();

    StateSummaries {
        states,
        start_states: terminal_automaton.start_states(),
        unique_bundles,
        bundle_accepts,
    }
}

/// Prebuild deterministic multi-terminal parser bundles whose terminal set is
/// disjoint from `excluded_terminals`. The cache is keyed by the full weighted
/// terminal-bundle signature, so a later parser-NWA build can verify exact
/// identity before reuse. This is compile-time-only state.
pub fn prebuild_parser_bundle_cache_excluding_terminals(
    terminal_automaton: &TerminalAutomaton,
    num_terminals: u32,
    templates: &Templates,
    excluded_terminals: &[bool],
) -> PrebuiltParserBundleCache {
    let summaries = build_state_summaries(terminal_automaton, num_terminals, templates);
    let productive = compute_productive_terminal_states(&summaries);
    let mut used = vec![false; summaries.unique_bundles.len()];
    for (state_id, state) in summaries.states.iter().enumerate() {
        if !productive[state_id] {
            continue;
        }
        for branch in &state.branches {
            let target_idx = branch.target as usize;
            if productive.get(target_idx).copied().unwrap_or(false)
                && summaries.bundle_accepts.get(branch.bundle_id).copied().unwrap_or(false)
                && summaries.unique_bundles[branch.bundle_id].len() > 1
            {
                used[branch.bundle_id] = true;
            }
        }
    }
    use rayon::prelude::*;
    let selected = summaries
        .unique_bundles
        .iter()
        .enumerate()
        .filter_map(|(bundle_id, bundle)| {
            (used[bundle_id]
                && !bundle.keys().any(|&terminal| {
                    excluded_terminals
                        .get(terminal as usize)
                        .copied()
                        .unwrap_or(true)
                }))
            .then_some(bundle)
        })
        .collect::<Vec<_>>();
    let repeated_group_cache = templates.build_bundle_group_dfa_cache(&selected);
    let build_bundle = |bundle: &&BTreeMap<TerminalID, Weight>| {
            (
                bundle_signature(bundle),
                Arc::new(templates.build_bundle_cached(bundle, &repeated_group_cache)),
            )
        };
    let built = if crate::templates::macro_parallelism_disabled() {
        let mut timings = Vec::with_capacity(selected.len());
        let built = selected
            .iter()
            .map(|bundle| {
                let started = Instant::now();
                let built = build_bundle(bundle);
                timings.push(started.elapsed().as_secs_f64() * 1000.0);
                built
            })
            .collect::<Vec<_>>();
        crate::templates::report_macro_item_timings("parser_bundle_prebuild", &timings);
        built
    } else {
        selected.par_iter().map(build_bundle).collect::<Vec<_>>()
    };
    PrebuiltParserBundleCache {
        by_signature: built.into_iter().collect(),
    }
}

fn union_final_weight(slot: &mut Option<Weight>, add: Weight) -> bool {
    if add.is_empty() {
        return false;
    }

    match slot {
        Some(existing) => {
            let updated = existing.union(&add);
            if updated != *existing {
                *existing = updated;
                true
            } else {
                false
            }
        }
        None => {
            *slot = Some(add);
            true
        }
    }
}

fn parser_state_label(label: i32, num_parser_states: u32) -> Option<u32> {
    if label >= 0 && (label as u32) < num_parser_states {
        Some(label as u32)
    } else {
        None
    }
}

fn top_row_action_is_unconditionally_applicable(_action: &Action) -> bool {
    // RowPresenceExact actions originate from a parser row whose terminal
    // domain is an exact admission set. Guarded stack shifts are a lowered
    // representation of that row's already-valid action, not a weaker
    // admission predicate. Their guards select the exact stack effect; they do
    // not make the terminal cease to be admissible from the row's top state.
    true
}

fn immediate_completion_weights_by_terminal(
    terminal_automaton: &TerminalAutomaton,
    grammar: &AnalyzedGrammar,
) -> BTreeMap<TerminalID, Weight> {
    let mut complete_by_terminal = BTreeMap::<TerminalID, Weight>::new();
    for start_state in terminal_automaton.start_states() {
        for (target, bundle) in
            group_terminal_edges_by_target(
                terminal_automaton,
                grammar.num_terminals,
                start_state,
            )
        {
            let Some(target_final) = terminal_state_final_weight(terminal_automaton, target as usize)
            else {
                continue;
            };
            for (terminal, edge_weight) in bundle {
                let complete = edge_weight.intersection(&target_final);
                if complete.is_empty() {
                    continue;
                }
                complete_by_terminal
                    .entry(terminal)
                    .and_modify(|existing| *existing = existing.union(&complete))
                    .or_insert(complete);
            }
        }
    }
    complete_by_terminal
}

fn immediate_acceptance_certificate_parts(
    terminal_automaton: &TerminalAutomaton,
    grammar: &AnalyzedGrammar,
    table: &GLRTable,
) -> BTreeMap<i32, Vec<Weight>> {
    if table.admission_policy != AdmissionPolicy::RowPresenceExact {
        return BTreeMap::new();
    }
    let complete_by_terminal = immediate_completion_weights_by_terminal(terminal_automaton, grammar);
    table
        .action
        .iter()
        .enumerate()
        .filter_map(|(parser_top, row)| {
            let parts = row
                .iter()
                .filter_map(|(terminal, action)| {
                    (top_row_action_is_unconditionally_applicable(action))
                        .then(|| complete_by_terminal.get(&terminal).cloned())
                        .flatten()
                })
                .collect::<Vec<_>>();
            (!parts.is_empty()).then_some((parser_top as i32, parts))
        })
        .collect()
}

fn immediate_acceptance_certificates(
    terminal_automaton: &TerminalAutomaton,
    grammar: &AnalyzedGrammar,
    table: &GLRTable,
) -> Vec<Weight> {
    let parts = immediate_acceptance_certificate_parts(terminal_automaton, grammar, table);
    let use_signature_cache =
        std::env::var_os("GLRMASK_DISABLE_IMMEDIATE_ACCEPT_SIGNATURE_CACHE").is_none();
    if !use_signature_cache {
        return (0..table.num_states)
            .map(|parser_top| {
                parts
                    .get(&(parser_top as i32))
                    .map(|weights| Weight::union_all(weights.iter()))
                    .unwrap_or_else(Weight::empty)
            })
            .collect();
    }

    // Parser rows frequently repeat the same set of immediately-accepted
    // terminals.  The old path recomputed the identical multi-way token-set
    // union once per parser state.  Intern the exact pointer signature and
    // materialize each union once; terminal completion weights remain alive for
    // this whole scope, so pointer identity is stable and exact.
    let mut union_by_signature = FxHashMap::<Vec<usize>, Weight>::default();
    let mut nonempty_rows = 0usize;
    let mut part_refs = 0usize;
    let result = (0..table.num_states)
        .map(|parser_top| {
            let Some(weights) = parts.get(&(parser_top as i32)) else {
                return Weight::empty();
            };
            nonempty_rows += 1;
            part_refs += weights.len();
            let mut signature = weights.iter().map(Weight::ptr_key).collect::<Vec<_>>();
            signature.sort_unstable();
            signature.dedup();
            if let Some(cached) = union_by_signature.get(&signature) {
                return cached.clone();
            }
            let union = Weight::union_all(weights.iter());
            union_by_signature.insert(signature, union.clone());
            union
        })
        .collect();
    if compile_profile_enabled() {
        eprintln!(
            "[glrmask/profile][immediate_accept_signature_cache] nonempty_rows={} part_refs={} distinct_signatures={} reused_rows={}",
            nonempty_rows,
            part_refs,
            union_by_signature.len(),
            nonempty_rows.saturating_sub(union_by_signature.len()),
        );
    }
    result
}

pub fn try_build_immediate_parser_top_accept_parts(
    terminal_automaton: &TerminalAutomaton,
    grammar: &AnalyzedGrammar,
    table: &GLRTable,
) -> Option<BTreeMap<i32, Vec<Weight>>> {
    terminal_automaton_is_immediate_completion(terminal_automaton, grammar, table)
        .then(|| immediate_acceptance_certificate_parts(terminal_automaton, grammar, table))
}

pub fn try_build_immediate_terminal_completion_weights(
    terminal_automaton: &TerminalAutomaton,
    grammar: &AnalyzedGrammar,
    table: &GLRTable,
) -> Option<BTreeMap<TerminalID, Weight>> {
    terminal_automaton_is_immediate_completion(terminal_automaton, grammar, table)
        .then(|| immediate_completion_weights_by_terminal(terminal_automaton, grammar))
}

#[cfg(test)]
fn direct_regular_action_targets(action: &Action) -> Option<SmallVec<[u32; 4]>> {
    match action {
        Action::Shift(target, true) => Some(SmallVec::from_slice(&[*target])),
        Action::StackShifts(shifts) => {
            let mut targets = SmallVec::<[u32; 4]>::new();
            for shift in shifts {
                if shift.pop != 1 || shift.pushes.len() != 1 {
                    return None;
                }
                targets.push(shift.pushes[0]);
            }
            targets.sort_unstable();
            targets.dedup();
            (!targets.is_empty()).then_some(targets)
        }
        _ => None,
    }
}

/// Compute exact token acceptance for a constant-depth direct-regular parser by
/// solving the weighted product of terminal-automaton states and parser-top
/// states. All product edges are epsilon edges because the sole parser-stack
/// symbol has already selected the product's parser-top coordinate.
pub fn try_build_direct_regular_parser_top_accept_parts(
    terminal_automaton: &TerminalAutomaton,
    grammar: &AnalyzedGrammar,
    table: &GLRTable,
) -> Option<BTreeMap<i32, Vec<Weight>>> {
    let automaton = grammar.direct_regular_automaton.as_ref()?;
    if table.admission_policy != AdmissionPolicy::RowPresenceExact
        || table.num_states == 0
        || automaton.states.is_empty()
        || automaton.start_states.is_empty()
    {
        return None;
    }
    let started_at = Instant::now();
    let terminal_state_count = terminal_automaton.num_states();
    if terminal_state_count == 0 {
        return None;
    }

    // Runtime parser state 0 is synthetic. Direct-regular NFA state N maps to
    // parser state N+1, matching direct GLR table construction. Working from
    // the sparse NFA rather than its epsilon-closed action rows avoids copying
    // a suffix frontier into every preceding state of skip-chain grammars.
    let parser_state_count = automaton.states.len().checked_add(1)?;
    if parser_state_count != table.num_states as usize {
        return None;
    }
    let product_state_count = terminal_state_count.checked_mul(parser_state_count)?;
    if product_state_count > u32::MAX as usize {
        return None;
    }
    let product_state = |terminal_state: usize, parser_state: usize| -> u32 {
        (terminal_state * parser_state_count + parser_state) as u32
    };

    let mut product = NWA::new(0, 0);
    for _ in 0..product_state_count {
        product.add_state();
    }

    // A final terminal-automaton state means one grammar terminal has
    // completed. Any live direct-regular parser state is a valid post-commit
    // continuation, exactly as in the previous table-product construction.
    for terminal_state in 0..terminal_state_count {
        if let Some(final_weight) = terminal_state_final_weight(terminal_automaton, terminal_state)
            && !final_weight.is_empty()
        {
            for parser_state in 0..parser_state_count {
                product.set_final_weight(
                    product_state(terminal_state, parser_state),
                    final_weight.clone(),
                );
            }
        }

        // Terminal-automaton epsilon edges do not change parser position.
        for (target, weight) in
            terminal_state_epsilon_branches(terminal_automaton, terminal_state)
        {
            if weight.is_empty() || target as usize >= terminal_state_count {
                continue;
            }
            for parser_state in 0..parser_state_count {
                product.add_epsilon(
                    product_state(terminal_state, parser_state),
                    product_state(target as usize, parser_state),
                    weight.clone(),
                );
            }
        }
    }

    // Parser epsilon edges do not change terminal-automaton state. State 0 has
    // epsilon edges to every direct-regular start state; NFA state N is N+1.
    let all = Weight::all();
    for terminal_state in 0..terminal_state_count {
        let terminal_product_base = terminal_state * parser_state_count;
        for &start in &automaton.start_states {
            if start as usize >= automaton.states.len() {
                return None;
            }
            product.add_epsilon(
                (terminal_product_base) as u32,
                (terminal_product_base + start as usize + 1) as u32,
                all.clone(),
            );
        }
        for (source, state) in automaton.states.iter().enumerate() {
            for &target in &state.epsilons {
                if target as usize >= automaton.states.len() {
                    return None;
                }
                product.add_epsilon(
                    (terminal_product_base + source + 1) as u32,
                    (terminal_product_base + target as usize + 1) as u32,
                    all.clone(),
                );
            }
        }
    }

    // Index the sparse direct-regular terminal edges by grammar terminal.
    let mut parser_sources_by_terminal =
        vec![Vec::<(u32, SmallVec<[u32; 4]>)>::new(); grammar.num_terminals as usize];
    for (source, state) in automaton.states.iter().enumerate() {
        for (&terminal, targets) in &state.transitions {
            if terminal >= grammar.num_terminals || targets.is_empty() {
                return None;
            }
            let mut mapped = SmallVec::<[u32; 4]>::new();
            for &target in targets {
                if target as usize >= automaton.states.len() {
                    return None;
                }
                mapped.push(target + 1);
            }
            mapped.sort_unstable();
            mapped.dedup();
            parser_sources_by_terminal[terminal as usize].push((source as u32 + 1, mapped));
        }
    }

    // Match terminal-automaton labelled edges with sparse parser-NFA edges.
    for terminal_state in 0..terminal_state_count {
        for (target, bundle) in
            group_terminal_edges_by_target(
                terminal_automaton,
                grammar.num_terminals,
                terminal_state as u32,
            )
        {
            if target as usize >= terminal_state_count {
                return None;
            }
            for (terminal, edge_weight) in bundle {
                if edge_weight.is_empty() {
                    continue;
                }
                for (source, parser_targets) in
                    &parser_sources_by_terminal[terminal as usize]
                {
                    let from = product_state(terminal_state, *source as usize);
                    for &parser_target in parser_targets {
                        product.add_epsilon(
                            from,
                            product_state(target as usize, parser_target as usize),
                            edge_weight.clone(),
                        );
                    }
                }
            }
        }
    }

    apply_finality_fixpoint(&mut product);
    let terminal_starts = terminal_automaton.start_states();
    let mut parts = BTreeMap::new();
    for parser_top in 0..parser_state_count {
        let weights = terminal_starts
            .iter()
            .filter_map(|&terminal_start| {
                product
                    .states()
                    .get(product_state(terminal_start as usize, parser_top) as usize)
                    .and_then(|state| state.final_weight.clone())
                    .filter(|weight| !weight.is_empty())
            })
            .collect::<Vec<_>>();
        if !weights.is_empty() {
            parts.insert(parser_top as i32, weights);
        }
    }

    if compile_profile_enabled() {
        eprintln!(
            "[glrmask/profile][direct_regular_parser_sparse_product] terminal_states={} parser_states={} product_states={} product_edges={} labels={} part_refs={} unique_weights={} total_ms={:.3}",
            terminal_state_count,
            parser_state_count,
            product_state_count,
            product.num_transitions(),
            parts.len(),
            parts.values().map(Vec::len).sum::<usize>(),
            parts
                .values()
                .flatten()
                .map(Weight::ptr_key)
                .collect::<rustc_hash::FxHashSet<_>>()
                .len(),
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Some(parts)
}

#[cfg(test)]
fn try_build_direct_regular_parser_top_accept_parts_table_product_reference(
    terminal_automaton: &TerminalAutomaton,
    grammar: &AnalyzedGrammar,
    table: &GLRTable,
) -> Option<BTreeMap<i32, Vec<Weight>>> {
    if grammar.direct_regular_automaton.is_none()
        || table.admission_policy != AdmissionPolicy::RowPresenceExact
        || table.num_states == 0
    {
        return None;
    }
    let started_at = Instant::now();
    let parser_state_count = table.num_states as usize;
    let terminal_state_count = terminal_automaton.num_states();
    if terminal_state_count == 0 {
        return None;
    }

    let mut parser_sources_by_terminal =
        vec![Vec::<(u32, SmallVec<[u32; 4]>)>::new(); grammar.num_terminals as usize];
    for (source, row) in table.action.iter().enumerate() {
        for (terminal, action) in row {
            if terminal >= grammar.num_terminals {
                continue;
            }
            let targets = direct_regular_action_targets(action)?;
            if targets.iter().any(|&target| target >= table.num_states) {
                return None;
            }
            parser_sources_by_terminal[terminal as usize].push((source as u32, targets));
        }
    }

    let product_state = |terminal_state: usize, parser_state: u32| -> u32 {
        (terminal_state * parser_state_count + parser_state as usize) as u32
    };
    let product_state_count = terminal_state_count.checked_mul(parser_state_count)?;
    let mut product = NWA::new(0, 0);
    for _ in 0..product_state_count {
        product.add_state();
    }

    for terminal_state in 0..terminal_state_count {
        if let Some(final_weight) = terminal_state_final_weight(terminal_automaton, terminal_state)
            && !final_weight.is_empty()
        {
            for parser_state in 0..table.num_states {
                product.set_final_weight(
                    product_state(terminal_state, parser_state),
                    final_weight.clone(),
                );
            }
        }

        for (target, weight) in
            terminal_state_epsilon_branches(terminal_automaton, terminal_state)
        {
            if weight.is_empty() || target as usize >= terminal_state_count {
                continue;
            }
            for parser_state in 0..table.num_states {
                product.add_epsilon(
                    product_state(terminal_state, parser_state),
                    product_state(target as usize, parser_state),
                    weight.clone(),
                );
            }
        }

        for (target, bundle) in
            group_terminal_edges_by_target(
                terminal_automaton,
                grammar.num_terminals,
                terminal_state as u32,
            )
        {
            if target as usize >= terminal_state_count {
                return None;
            }
            for (terminal, edge_weight) in bundle {
                if edge_weight.is_empty() {
                    continue;
                }
                for (source, parser_targets) in
                    &parser_sources_by_terminal[terminal as usize]
                {
                    let from = product_state(terminal_state, *source);
                    for &parser_target in parser_targets {
                        product.add_epsilon(
                            from,
                            product_state(target as usize, parser_target),
                            edge_weight.clone(),
                        );
                    }
                }
            }
        }
    }

    apply_finality_fixpoint(&mut product);
    let terminal_starts = terminal_automaton.start_states();
    let mut parts = BTreeMap::new();
    for parser_top in 0..table.num_states {
        let weights = terminal_starts
            .iter()
            .filter_map(|&terminal_start| {
                product
                    .states()
                    .get(product_state(terminal_start as usize, parser_top) as usize)
                    .and_then(|state| state.final_weight.clone())
                    .filter(|weight| !weight.is_empty())
            })
            .collect::<Vec<_>>();
        if !weights.is_empty() {
            parts.insert(parser_top as i32, weights);
        }
    }

    if compile_profile_enabled() {
        eprintln!(
            "[glrmask/profile][direct_regular_parser_product] terminal_states={} parser_states={} product_states={} product_edges={} labels={} part_refs={} unique_weights={} total_ms={:.3}",
            terminal_state_count,
            parser_state_count,
            product_state_count,
            product.num_transitions(),
            parts.len(),
            parts.values().map(Vec::len).sum::<usize>(),
            parts
                .values()
                .flatten()
                .map(Weight::ptr_key)
                .collect::<rustc_hash::FxHashSet<_>>()
                .len(),
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Some(parts)
}

fn terminal_automaton_is_immediate_completion(
    terminal_automaton: &TerminalAutomaton,
    grammar: &AnalyzedGrammar,
    table: &GLRTable,
) -> bool {
    if table.admission_policy != AdmissionPolicy::RowPresenceExact {
        return false;
    }
    let TerminalAutomaton::Dwa(dwa) = terminal_automaton else {
        return false;
    };
    let Some(start) = dwa.states().get(dwa.start_state() as usize) else {
        return false;
    };

    if start
        .final_weight
        .as_ref()
        .is_some_and(|weight| !weight.is_empty())
    {
        return false;
    }

    let mut saw_edge = false;
    for (&label, (target, edge_weight)) in &start.transitions {
        if edge_weight.is_empty() {
            continue;
        }
        if label < 0 || label as u32 >= grammar.num_terminals {
            return false;
        }
        let Some(target_state) = dwa.states().get(*target as usize) else {
            return false;
        };
        if target_state
            .transitions
            .values()
            .any(|(_, weight)| !weight.is_empty())
        {
            return false;
        }
        let Some(target_final) = target_state.final_weight.as_ref() else {
            return false;
        };
        if !edge_weight.is_subset(target_final) {
            return false;
        }
        saw_edge = true;
    }
    saw_edge
}

pub fn try_build_immediate_parser_dwa(
    terminal_automaton: &TerminalAutomaton,
    grammar: &AnalyzedGrammar,
    table: &GLRTable,
) -> Option<DWA> {
    if !terminal_automaton_is_immediate_completion(terminal_automaton, grammar, table) {
        return None;
    }
    let certificates = immediate_acceptance_certificates(terminal_automaton, grammar, table);
    let mut parser_dwa = DWA::new(0, 0);
    let final_state = parser_dwa.add_state();
    parser_dwa.set_final_weight(final_state, Weight::all());
    for (parser_top, weight) in certificates.into_iter().enumerate() {
        if !weight.is_empty() {
            parser_dwa.add_transition(0, parser_top as i32, final_state, weight);
        }
    }
    Some(parser_dwa)
}

fn collapse_immediate_acceptance_certificates(
    parser_dwa: &mut DWA,
    terminal_automaton: &TerminalAutomaton,
    grammar: &AnalyzedGrammar,
    table: &GLRTable,
) -> usize {
    if parser_dwa.states().is_empty() {
        return 0;
    }
    let certificates = immediate_acceptance_certificates(terminal_automaton, grammar, table);
    let start_state = parser_dwa.start_state();
    let mut rewrites = Vec::<(i32, Weight)>::new();
    for (&label, (_target, edge_weight)) in
        &parser_dwa.states()[start_state as usize].transitions
    {
        let Some(parser_top) = parser_state_label(label, table.num_states) else {
            continue;
        };
        if edge_weight.is_subset(&certificates[parser_top as usize]) {
            rewrites.push((label, edge_weight.clone()));
        }
    }
    if rewrites.is_empty() {
        return 0;
    }

    let sink = parser_dwa.add_state();
    parser_dwa.set_final_weight(sink, Weight::union_all(rewrites.iter().map(|(_, w)| w)));
    for (label, _) in &rewrites {
        let (target, _) = parser_dwa.states_mut()[start_state as usize]
            .transitions
            .get_mut(label)
            .expect("certified start transition disappeared");
        *target = sink;
    }
    rewrites.len()
}

fn trim_unreachable_dwa(mut dwa: DWA) -> DWA {
    if dwa.states().is_empty() {
        return dwa;
    }
    let old_start = dwa.start_state() as usize;
    // Consume the state's owned BTreeMaps/weights instead of cloning the whole
    // DWA and then cloning every reachable row a second time.
    let old_states = std::mem::take(dwa.states_mut());
    let mut reachable = vec![false; old_states.len()];
    let mut queue = VecDeque::from([old_start]);
    reachable[old_start] = true;
    while let Some(state_id) = queue.pop_front() {
        for (target, weight) in old_states[state_id].transitions.values() {
            let target = *target as usize;
            if weight.is_empty() || target >= old_states.len() || reachable[target] {
                continue;
            }
            reachable[target] = true;
            queue.push_back(target);
        }
    }

    let mut remap = vec![u32::MAX; old_states.len()];
    let live_count = reachable.iter().filter(|&&live| live).count();
    let mut next_id = 0u32;
    for (old_id, &live) in reachable.iter().enumerate() {
        if live {
            remap[old_id] = next_id;
            next_id += 1;
        }
    }
    let mut new_states = Vec::with_capacity(live_count);
    for (old_id, state) in old_states.into_iter().enumerate() {
        if reachable[old_id] {
            new_states.push(state);
        }
    }
    if rayon::current_num_threads() > 1 && new_states.len() >= 16_384 {
        use rayon::prelude::*;
        new_states.par_iter_mut().for_each(|state| {
            state.transitions.retain(|_, (target, weight)| {
                if weight.is_empty() || (*target as usize) >= remap.len() {
                    return false;
                }
                let mapped = remap[*target as usize];
                if mapped == u32::MAX {
                    return false;
                }
                *target = mapped;
                true
            });
        });
    } else {
        for state in &mut new_states {
            state.transitions.retain(|_, (target, weight)| {
                if weight.is_empty() || (*target as usize) >= remap.len() {
                    return false;
                }
                let mapped = remap[*target as usize];
                if mapped == u32::MAX {
                    return false;
                }
                *target = mapped;
                true
            });
        }
    }
    DWA::from_parts(new_states, remap[old_start])
}

/// Push final weights from transition-free leaf states into their incoming
/// edges, then share one `final = all` sink. Runtime evaluation already
/// intersects the accumulated path weight with the destination final weight,
/// so this is an exact weighted normalization rather than a language change.
fn collapse_final_leaf_targets(mut dwa: DWA) -> DWA {
    if dwa.states().is_empty() {
        return dwa;
    }
    let leaf_finals: Vec<Option<Weight>> = dwa
        .states()
        .iter()
        .map(|state| {
            (state.transitions.is_empty())
                .then(|| state.final_weight.clone())
                .flatten()
                .filter(|weight| !weight.is_empty())
        })
        .collect();
    if leaf_finals.iter().all(Option::is_none) {
        return dwa;
    }

    let sink = dwa.add_state();
    dwa.set_final_weight(sink, Weight::all());
    let changed_count = if rayon::current_num_threads() > 1 && sink as usize >= 16_384 {
        use rayon::prelude::*;
        dwa.states_mut()[..sink as usize]
            .par_iter_mut()
            .map_init(ScopedWeightOpCache::default, |weight_ops, state| {
                let mut changed = 0usize;
                state.transitions.retain(|_, (target, edge_weight)| {
                    let Some(final_weight) = leaf_finals
                        .get(*target as usize)
                        .and_then(Option::as_ref)
                    else {
                        return true;
                    };
                    let pushed = weight_ops.intersection(edge_weight, final_weight);
                    changed += 1;
                    if pushed.is_empty() {
                        false
                    } else {
                        *target = sink;
                        *edge_weight = pushed;
                        true
                    }
                });
                changed
            })
            .sum::<usize>()
    } else {
        let mut changed = 0usize;
        let mut weight_ops = ScopedWeightOpCache::default();
        for state_id in 0..sink as usize {
            dwa.states_mut()[state_id].transitions.retain(|_, (target, edge_weight)| {
                let Some(final_weight) = leaf_finals
                    .get(*target as usize)
                    .and_then(Option::as_ref)
                else {
                    return true;
                };
                let pushed = weight_ops.intersection(edge_weight, final_weight);
                changed += 1;
                if pushed.is_empty() {
                    false
                } else {
                    *target = sink;
                    *edge_weight = pushed;
                    true
                }
            });
        }
        changed
    };
    if changed_count == 0 {
        dwa.states_mut().pop();
        return dwa;
    }
    trim_unreachable_dwa(dwa)
}

enum PossibleOutgoingIds {
    Empty,
    All,
    Some(BitSet),
}

fn build_possible_outgoing_ids_by_state(
    parser_nwa: &NWA,
    state_supports: &[Vec<u32>],
    num_parser_states: u32,
) -> Vec<PossibleOutgoingIds> {
    enum OutgoingIds {
        Empty,
        All,
        Some(Vec<u32>),
    }

    let num_parser_states = num_parser_states as usize;
    let all_parser_states = BitSet::all(num_parser_states);
    let parallel = rayon::current_num_threads() > 1
        && parser_nwa.states().len()
            >= std::env::var("GLRMASK_POSSIBLE_OUTGOING_PARALLEL_MIN_NWA_STATES")
                .ok()
                .and_then(|value| value.trim().parse::<usize>().ok())
                .unwrap_or(4_096)
        && std::env::var_os("GLRMASK_DISABLE_PARALLEL_POSSIBLE_OUTGOING").is_none();

    let summarize_state = |state: &crate::automata::weighted::nwa::NWAState| {
        let mut ids = Vec::new();
        for &label in state.transitions.keys() {
            if label == DEFAULT_LABEL {
                return OutgoingIds::All;
            }
            if let Some(parser_state_id) = parser_state_label(label, num_parser_states as u32) {
                ids.push(parser_state_id);
            }
        }
        if ids.is_empty() {
            OutgoingIds::Empty
        } else {
            OutgoingIds::Some(ids)
        }
    };

    let state_outgoing_ids: Vec<OutgoingIds> = if parallel {
        use rayon::prelude::*;
        parser_nwa.states().par_iter().map(summarize_state).collect()
    } else {
        parser_nwa.states().iter().map(summarize_state).collect()
    };

    let summarize_support = |support: &Vec<u32>| {
        if support.len() == 1 {
            let state_id = support[0] as usize;
            return match state_outgoing_ids.get(state_id) {
                Some(OutgoingIds::Empty) => PossibleOutgoingIds::Empty,
                Some(OutgoingIds::All) => PossibleOutgoingIds::All,
                Some(OutgoingIds::Some(ids)) => {
                    let mut bitset = BitSet::new(num_parser_states);
                    for &parser_state_id in ids {
                        bitset.set(parser_state_id as usize);
                    }
                    if bitset == all_parser_states {
                        PossibleOutgoingIds::All
                    } else {
                        PossibleOutgoingIds::Some(bitset)
                    }
                }
                None => PossibleOutgoingIds::Empty,
            };
        }

        let mut ids = BitSet::new(num_parser_states);
        for &state_id in support {
            let Some(state_ids) = state_outgoing_ids.get(state_id as usize) else {
                continue;
            };
            match state_ids {
                OutgoingIds::Empty => {}
                OutgoingIds::All => return PossibleOutgoingIds::All,
                OutgoingIds::Some(state_ids) => {
                    for &parser_state_id in state_ids {
                        ids.set(parser_state_id as usize);
                    }
                    if ids == all_parser_states {
                        break;
                    }
                }
            }
        }
        if ids.is_empty() {
            PossibleOutgoingIds::Empty
        } else if ids == all_parser_states {
            PossibleOutgoingIds::All
        } else {
            PossibleOutgoingIds::Some(ids)
        }
    };

    if parallel && state_supports.len() >= 1_024 {
        use rayon::prelude::*;
        state_supports.par_iter().map(summarize_support).collect()
    } else {
        state_supports.iter().map(summarize_support).collect()
    }
}

fn local_epsilon_closure(
    nwa: &NWA,
    weight_by_state: &mut Vec<Option<Weight>>,
    closure_queue: &mut VecDeque<u32>,
    seed: &mut FxHashMap<u32, Weight>,
) {
    let mut seed_states: Vec<u32> = Vec::new();
    for (&state_id, weight) in seed.iter() {
        weight_by_state[state_id as usize] = Some(weight.clone());
        closure_queue.push_back(state_id);
        seed_states.push(state_id);
    }
    if seed.len() == 1 {
        let state_id = seed_states[0];
        if let Some(state) = nwa.states().get(state_id as usize) {
            if state.epsilons.is_empty() {
                closure_queue.clear();
                for &s in &seed_states {
                    weight_by_state[s as usize] = None;
                }
                return;
            }
        }
    }
    while let Some(state_id) = closure_queue.pop_front() {
        let Some(current_weight) = weight_by_state[state_id as usize].clone() else {
            continue;
        };
        let Some(state) = nwa.states().get(state_id as usize) else {
            continue;
        };
        for (target, edge_weight) in &state.epsilons {
            let contribution = current_weight.intersection(edge_weight);
            if contribution.is_empty() {
                continue;
            }
            let target_idx = *target as usize;
            if let Some(existing) = &weight_by_state[target_idx] {
                if !contribution.is_subset(existing) {
                    weight_by_state[target_idx] = Some(existing.union(&contribution));
                    closure_queue.push_back(*target);
                    if !seed_states.contains(target) {
                        seed_states.push(*target);
                    }
                }
            } else {
                weight_by_state[target_idx] = Some(contribution);
                closure_queue.push_back(*target);
                seed_states.push(*target);
            }
        }
    }
    seed.clear();
    for &s in &seed_states {
        if let Some(w) = weight_by_state[s as usize].take() {
            seed.insert(s, w);
        }
    }
}

fn local_epsilon_closure_canonical(
    nwa: &NWA,
    weight_by_state: &mut [Option<Weight>],
    closure_queue: &mut VecDeque<u32>,
    seeds: &[(u32, Weight)],
    touched_states: &mut Vec<u32>,
    canonical: &mut Vec<(u32, Weight)>,
    weight_ops: &mut ScopedWeightOpCache,
    use_weight_cache: bool,
) {
    debug_assert!(closure_queue.is_empty());
    touched_states.clear();
    canonical.clear();

    for (state_id, weight) in seeds {
        let slot = &mut weight_by_state[*state_id as usize];
        debug_assert!(slot.is_none());
        *slot = Some(weight.clone());
        closure_queue.push_back(*state_id);
        touched_states.push(*state_id);
    }

    while let Some(state_id) = closure_queue.pop_front() {
        let Some(current_weight) = weight_by_state[state_id as usize].clone() else {
            continue;
        };
        let Some(state) = nwa.states().get(state_id as usize) else {
            continue;
        };
        for (target, edge_weight) in &state.epsilons {
            let contribution = if use_weight_cache {
                weight_ops.intersection(&current_weight, edge_weight)
            } else {
                current_weight.intersection(edge_weight)
            };
            if contribution.is_empty() {
                continue;
            }
            let target_idx = *target as usize;
            if let Some(existing) = &weight_by_state[target_idx] {
                if !contribution.is_subset(existing) {
                    weight_by_state[target_idx] = Some(if use_weight_cache {
                        weight_ops.union(existing, &contribution)
                    } else {
                        existing.union(&contribution)
                    });
                    closure_queue.push_back(*target);
                }
            } else {
                weight_by_state[target_idx] = Some(contribution);
                closure_queue.push_back(*target);
                touched_states.push(*target);
            }
        }
    }

    touched_states.sort_unstable();
    for &state_id in touched_states.iter() {
        if let Some(weight) = weight_by_state[state_id as usize].take()
            && !weight.is_empty()
        {
            canonical.push((state_id, weight));
        }
    }
}

fn determinize_with_supports_mode(
    nwa: &NWA,
    dense_positive_label_limit: Option<u32>,
    defer_edge_unions_override: Option<bool>,
    normalize_singletons_override: Option<bool>,
    normalize_subsets_override: Option<bool>,
) -> DeterminizedDwaWithSupports {
    fn subset_key(entries: &[(u32, Weight)]) -> Vec<(u32, usize)> {
        entries.iter().map(|(sid, w)| (*sid, w.ptr_key())).collect()
    }

      #[derive(Default)]
      struct UnionAllCache {
        entries: FxHashMap<SmallVec<[usize; 16]>, Weight>,
        ordered_keys: bool,
        profile_enabled: bool,
        hits: usize,
        misses: usize,
        key_len_sum: usize,
        key_len_max: usize,
        total_ms: f64,
    }

    impl UnionAllCache {
        fn record_elapsed(&mut self, started: Option<Instant>) {
            if let Some(started) = started {
                self.total_ms += elapsed_ms(started);
            }
        }

        fn union_all<'a>(&mut self, weights: impl IntoIterator<Item = &'a Weight>) -> Weight {
            let started = self.profile_enabled.then(Instant::now);
            let mut meaningful = SmallVec::<[&Weight; 8]>::new();
            for weight in weights {
                if weight.is_full() {
                    self.record_elapsed(started);
                    return Weight::all();
                }
                if !weight.is_empty() {
                    meaningful.push(weight);
                }
            }

            if meaningful.is_empty() {
                self.record_elapsed(started);
                return Weight::empty();
            }
            if meaningful.len() == 1 {
                self.record_elapsed(started);
                return meaningful[0].clone();
            }

            let mut key: SmallVec<[usize; 16]> =
                meaningful.iter().map(|weight| weight.ptr_key()).collect();
            // Contributions are already in deterministic target-state order.
            // Using that exact sequence as the cache key preserves correctness:
            // a different order merely misses the cache and recomputes the exact
            // union. Canonical sorting only increases sharing, while costing more
            // than the rare extra miss on parser-DWA workloads.
            if !self.ordered_keys {
                key.sort_unstable();
                key.dedup();
            }
            self.key_len_sum += key.len();
            self.key_len_max = self.key_len_max.max(key.len());

            if key.len() == 1 {
                self.record_elapsed(started);
                return meaningful[0].clone();
            }

            if let Some(weight) = self.entries.get(&key) {
                let weight = weight.clone();
                self.hits += 1;
                self.record_elapsed(started);
                return weight;
            }

            self.misses += 1;
            let direct_union =
                std::env::var_os("GLRMASK_DISABLE_PARSER_SUPPORT_DIRECT_UNION").is_none()
                    && (std::env::var_os("GLRMASK_PARSER_SUPPORT_DIRECT_UNION").is_some()
                        || meaningful.len() >= 5);
            let weight = if direct_union {
                Weight::union_all_direct(meaningful.into_iter())
            } else {
                Weight::union_all(meaningful.into_iter())
            };
            self.entries.insert(key, weight.clone());
            self.record_elapsed(started);
            weight
        }
    }

    #[derive(Clone)]
    enum DeferredEdgeWeight {
        Immediate(Weight),
        Job(usize),
    }

    #[derive(Clone)]
    struct DeferredClosure {
        to_state: u32,
        edge_weight: DeferredEdgeWeight,
    }

    struct DeferredTransition {
        from_state: u32,
        label: i32,
        to_state: u32,
        edge_weight: DeferredEdgeWeight,
    }
    let num_nwa_states = nwa.states().len();

    // Use flat arrays for epsilon closure when NWA is small enough.
    // weight_by_state[i] = Some(weight) means state i is in the closure.
    let mut weight_by_state: Vec<Option<Weight>> = vec![None; num_nwa_states];
    let mut closure_queue: VecDeque<u32> = VecDeque::new();
    // Reusable buffer for canonicalized entries.
    let mut canon_buf: Vec<(u32, Weight)> = Vec::new();

    // Epsilon closure using flat arrays instead of FxHashMap.
    let epsilon_closure = |weight_by_state: &mut Vec<Option<Weight>>,
                           closure_queue: &mut VecDeque<u32>,
                           seed: &mut FxHashMap<u32, Weight>| {
        // Initialize flat array from seed.
        let mut seed_states: Vec<u32> = Vec::new();
        for (&state_id, weight) in seed.iter() {
            weight_by_state[state_id as usize] = Some(weight.clone());
            closure_queue.push_back(state_id);
            seed_states.push(state_id);
        }

        // Fast path: single seed with no epsilons.
        if seed.len() == 1 {
            let state_id = seed_states[0];
            if let Some(state) = nwa.states().get(state_id as usize) {
                if state.epsilons.is_empty() {
                    // Clean up and return early — seed is already populated.
                    closure_queue.clear();
                    for &s in &seed_states {
                        weight_by_state[s as usize] = None;
                    }
                    return;
                }
            }
        }

        while let Some(state_id) = closure_queue.pop_front() {
            let Some(current_weight) = weight_by_state[state_id as usize].clone() else {
                continue;
            };
            let Some(state) = nwa.states().get(state_id as usize) else {
                continue;
            };
            for (target, edge_weight) in &state.epsilons {
                let contribution = current_weight.intersection(edge_weight);
                if contribution.is_empty() {
                    continue;
                }
                let target_idx = *target as usize;
                if let Some(existing) = &weight_by_state[target_idx] {
                    if !contribution.is_subset(existing) {
                        weight_by_state[target_idx] = Some(existing.union(&contribution));
                        closure_queue.push_back(*target);
                    }
                } else {
                    weight_by_state[target_idx] = Some(contribution);
                    closure_queue.push_back(*target);
                    seed_states.push(*target);
                }
            }
        }

        // Write results back to seed map.
        seed.clear();
        for &s in &seed_states {
            if let Some(w) = weight_by_state[s as usize].take() {
                seed.insert(s, w);
            }
        }
    };

    // Canonicalize from FxHashMap into reusable buffer.
    let canonicalize_into =
        |map: &FxHashMap<u32, Weight>, buf: &mut Vec<(u32, Weight)>| {
            buf.clear();
            for (&state_id, weight) in map.iter() {
                if !weight.is_empty() {
                    buf.push((state_id, weight.clone()));
                }
            }
            buf.sort_unstable_by_key(|(state_id, _)| *state_id);
        };

    let mut dwa = DWA::new(0, 0);
    let mut supports = vec![Vec::new()];

    let mut start_subset = FxHashMap::default();
    for &state_id in nwa.start_states() {
        let existing = start_subset.get(&state_id).cloned().unwrap_or_else(Weight::empty);
        start_subset.insert(state_id, existing.union(&Weight::all()));
    }
    epsilon_closure(&mut weight_by_state, &mut closure_queue, &mut start_subset);
    if start_subset.is_empty() {
        return DeterminizedDwaWithSupports { dwa, supports };
    }

    canonicalize_into(&start_subset, &mut canon_buf);
    supports[0] = canon_buf.iter().map(|(state_id, _)| *state_id).collect();

    let normalize_singletons = normalize_singletons_override.unwrap_or_else(|| {
        if std::env::var_os("GLRMASK_DISABLE_PARSER_SUPPORT_NORMALIZE_SINGLETONS").is_some() {
            return false;
        }
        if std::env::var_os("GLRMASK_PARSER_SUPPORT_NORMALIZE_SINGLETONS").is_some() {
            return true;
        }
        let min_states = std::env::var("GLRMASK_PARSER_SUPPORT_NORMALIZE_SINGLETON_MIN_NWA_STATES")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(4_096);
        nwa.states().len() >= min_states
    });
    let normalized_singleton_weight = Weight::all();
    let normalized_singleton_key = normalized_singleton_weight.ptr_key();
    let normalize_subsets = normalize_subsets_override.unwrap_or_else(|| {
        std::env::var_os("GLRMASK_PARSER_SUPPORT_NORMALIZE_SUBSETS").is_some()
            && std::env::var_os("GLRMASK_DISABLE_PARSER_SUPPORT_NORMALIZE_SUBSETS").is_none()
    });

    let mut subset_map: FxHashMap<Vec<(u32, usize)>, u32> = FxHashMap::default();
    let mut singleton_subsets = ParserSingletonSubsetCache::new(nwa.states().len());
    let start_key = subset_key(&canon_buf);
    subset_map.insert(start_key, dwa.start_state());
    if let [(state_id, weight)] = canon_buf.as_slice() {
        singleton_subsets.insert(*state_id, weight.ptr_key(), dwa.start_state());
    }
    let mut worklist: VecDeque<(u32, Vec<(u32, Weight)>)> = VecDeque::new();
    worklist.push_back((dwa.start_state(), canon_buf.clone()));

    let dense_label_limit = dense_positive_label_limit.map(|n| n as usize).unwrap_or(0);
    let mut dense_raw_targets: Vec<TargetContribs> =
        (0..dense_label_limit).map(|_| TargetContribs::new()).collect();
    let mut default_raw_targets: TargetContribs = TargetContribs::new();
    let mut sparse_raw_targets: FxHashMap<i32, TargetContribs> = FxHashMap::default();
    let mut touched_dense_labels: Vec<usize> = Vec::new();
    let mut dense_label_touched: Vec<bool> = vec![false; dense_label_limit];
    let mut default_touched = false;
    let mut intersection_cache = ScopedWeightOpCache::default();
    let use_epsilon_closure_weight_cache =
        std::env::var_os("GLRMASK_DISABLE_PARSER_EPSILON_CLOSURE_WEIGHT_CACHE").is_none();
    // Memoize local epsilon-closure outputs keyed by pre-closure weighted subsets.
    let mut closure_cache: FxHashMap<Vec<(u32, usize)>, CachedClosure> = FxHashMap::default();
    let mut singleton_closure_cache: FxHashMap<(u32, usize), CachedClosure> = FxHashMap::default();
    let mut key_buf: Vec<(u32, usize)> = Vec::new();
    let mut closure_touched_states: Vec<u32> = Vec::new();
    let mut closure_canon: Vec<(u32, Weight)> = Vec::new();
    let use_flat_canonical_closure = std::env::var("GLRMASK_DISABLE_FLAT_CANONICAL_EPSILON_CLOSURE")
        .map(|value| {
            let value = value.trim();
            value.is_empty() || value == "0" || value.eq_ignore_ascii_case("false")
        })
        .unwrap_or(true);
    let mut detail =
        ParserDwaDeterminizeDetail::enabled().then(ParserDwaDeterminizeDetail::default);
    let mut profiled_component_pairs = detail.as_ref().map(|_| FxHashSet::<(u32, usize)>::default());
    let mut profiled_component_entries = 0usize;
    let mut profiled_unique_component_transition_scans = 0usize;
    let component_row_cache_enabled = detail.is_none()
        && nwa.states().len() >= std::env::var("GLRMASK_PARSER_SUPPORT_COMPONENT_CACHE_MIN_NWA_STATES")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(16_384)
        && std::env::var_os("GLRMASK_DISABLE_PARSER_SUPPORT_COMPONENT_CACHE").is_none();
    let mut component_row_cache =
        FxHashMap::<(u32, usize), Arc<Vec<(i32, u32, Weight)>>>::default();
    let mut union_cache = UnionAllCache {
        ordered_keys: std::env::var_os("GLRMASK_DISABLE_ORDERED_UNION_CACHE_KEY").is_none(),
        profile_enabled: detail.is_some(),
        ..UnionAllCache::default()
    };

    let defer_edge_unions = defer_edge_unions_override
        .unwrap_or_else(|| parser_support_defer_edge_unions_enabled(nwa.states().len()));
    let mut deferred_union_ids = FxHashMap::<SmallVec<[usize; 16]>, usize>::default();
    let mut deferred_union_jobs = Vec::<SmallVec<[Weight; 8]>>::new();
    let mut deferred_closure_cache =
        FxHashMap::<Vec<(u32, usize)>, DeferredClosure>::default();
    let mut deferred_singleton_closure_cache =
        FxHashMap::<(u32, usize), DeferredClosure>::default();
    let mut deferred_transitions = Vec::<DeferredTransition>::new();
    let mut deferred_hits = 0usize;
    let mut deferred_misses = 0usize;
    let mut deferred_key_len_sum = 0usize;
    let mut deferred_key_len_max = 0usize;

    // Deferred final weight computation: store subset entries for each DWA state
    // and compute final weights in parallel after the main loop.
    let mut deferred_final_entries: Vec<(u32, DeferredFinalEntries)> = Vec::new();

    // The expensive support determinization is a graph fixed point only at the
    // target-subset interning boundary. Expanding already-interned states is
    // embarrassingly parallel: each state independently scans its NWA support
    // and computes label -> weighted-target contributions. Keep interning and
    // transition insertion serial/deterministic, but batch the scan phase so the
    // millions of weight intersections can use the full compile pool.
    let parallel_frontier_scan = detail.is_none()
        && rayon::current_num_threads() > 1
        && nwa.states().len() >= std::env::var("GLRMASK_PARSER_SUPPORT_PARALLEL_MIN_NWA_STATES")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(8_192)
        && std::env::var_os("GLRMASK_DISABLE_PARSER_SUPPORT_PARALLEL_FRONTIER").is_none();
    let parallel_frontier_min = std::env::var("GLRMASK_PARSER_SUPPORT_PARALLEL_MIN_FRONTIER")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|&value| value > 1)
        .unwrap_or(16);
    let parallel_frontier_wave = std::env::var("GLRMASK_PARSER_SUPPORT_PARALLEL_WAVE")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(2_048);
    let fast_support_profile = compile_profile_enabled() && detail.is_none();
    let mut fast_wave_count = 0usize;
    let mut fast_wave_states = 0usize;
    let mut fast_component_refs = 0usize;
    let mut fast_component_rows_computed = 0usize;
    let mut fast_component_row_edges = 0usize;
    let mut fast_component_key_ms = 0.0f64;
    let mut fast_component_compute_ms = 0.0f64;
    let mut fast_state_aggregate_ms = 0.0f64;
    let mut fast_serial_scan_ms = 0.0f64;
    let mut fast_label_process_ms = 0.0f64;
    let mut fast_deferred_union_ms = 0.0f64;
    let mut fast_transition_materialize_ms = 0.0f64;
    let mut fast_final_weights_ms = 0.0f64;
    let mut fast_labels = 0usize;
    let mut fast_label_contribs = 0usize;
    let mut fast_singleton_no_epsilon = 0usize;
    let mut fast_closure_hits = 0usize;
    let mut fast_closure_misses = 0usize;
    let mut fast_subset_hits = 0usize;
    let mut fast_subset_misses = 0usize;
    struct PrescannedSupportState {
        from_state: u32,
        subset_entries: Vec<(u32, Weight)>,
        cached_transitions: Vec<DeferredTransition>,
        pending_labels: Vec<(i32, TargetContribs)>,
        parallel_closure_hits: usize,
        parallel_subset_hits: usize,
    }
    let mut prescanned_states: VecDeque<PrescannedSupportState> = VecDeque::new();
    let mut fast_parallel_closure_hits = 0usize;
    let mut fast_parallel_subset_hits = 0usize;

    let mut defer_edge_weight = |contribs: &TargetContribs| -> DeferredEdgeWeight {
        debug_assert!(!contribs.is_empty());
        if contribs.iter().any(|(_, weight)| weight.is_full()) {
            return DeferredEdgeWeight::Immediate(Weight::all());
        }
        if contribs.len() == 1 {
            return DeferredEdgeWeight::Immediate(contribs[0].1.clone());
        }
        let key: SmallVec<[usize; 16]> =
            contribs.iter().map(|(_, weight)| weight.ptr_key()).collect();
        deferred_key_len_sum += key.len();
        deferred_key_len_max = deferred_key_len_max.max(key.len());
        if let Some(&job) = deferred_union_ids.get(&key) {
            deferred_hits += 1;
            return DeferredEdgeWeight::Job(job);
        }
        deferred_misses += 1;
        let job = deferred_union_jobs.len();
        deferred_union_ids.insert(key, job);
        deferred_union_jobs.push(
            contribs
                .iter()
                .map(|(_, weight)| weight.clone())
                .collect::<SmallVec<[Weight; 8]>>(),
        );
        DeferredEdgeWeight::Job(job)
    };

    while !worklist.is_empty() || !prescanned_states.is_empty() {
        if prescanned_states.is_empty()
            && parallel_frontier_scan
            && worklist.len() >= parallel_frontier_min
        {
            use rayon::prelude::*;
            let wave_len = worklist.len().min(parallel_frontier_wave);
            let wave = worklist.drain(..wave_len).collect::<Vec<_>>();
            if fast_support_profile {
                fast_wave_count += 1;
                fast_wave_states += wave.len();
                fast_component_refs += wave.iter().map(|(_, entries)| entries.len()).sum::<usize>();
            }

            // Materialize each distinct weighted NWA row at most once globally.
            // The wave itself often reuses the same component hundreds of times,
            // so computing missing rows in parallel before state aggregation turns
            // the expensive intersection work into a compact, cacheable frontier.
            if component_row_cache_enabled {
                let key_started = fast_support_profile.then(Instant::now);
                let mut missing = FxHashMap::<(u32, usize), (u32, Weight)>::default();
                for (_, subset_entries) in &wave {
                    for (nwa_state_id, path_weight) in subset_entries {
                        let key = (*nwa_state_id, path_weight.ptr_key());
                        if !component_row_cache.contains_key(&key) {
                            missing.entry(key).or_insert_with(|| (*nwa_state_id, path_weight.clone()));
                        }
                    }
                }
                if let Some(started) = key_started {
                    fast_component_key_ms += elapsed_ms(started);
                }
                let compute_started = fast_support_profile.then(Instant::now);
                let computed = missing
                    .into_iter()
                    .collect::<Vec<_>>()
                    .into_par_iter()
                    .map_init(ScopedWeightOpCache::default, |intersection_cache, (key, (nwa_state_id, path_weight))| {
                        let state = &nwa.states()[nwa_state_id as usize];
                        let mut row = Vec::new();
                        for (&label, targets) in &state.transitions {
                            for (target, transition_weight) in targets {
                                let next_weight = intersection_cache.intersection(&path_weight, transition_weight);
                                if !next_weight.is_empty() {
                                    row.push((label, *target, next_weight));
                                }
                            }
                        }
                        (key, Arc::new(row))
                    })
                    .collect::<Vec<_>>();
                if let Some(started) = compute_started {
                    fast_component_compute_ms += elapsed_ms(started);
                }
                if fast_support_profile {
                    fast_component_rows_computed += computed.len();
                    fast_component_row_edges += computed.iter().map(|(_, row)| row.len()).sum::<usize>();
                }
                for (key, row) in computed {
                    component_row_cache.entry(key).or_insert(row);
                }
            }

            let component_rows = &component_row_cache;
            let closure_cache_ref = &deferred_closure_cache;
            let singleton_closure_cache_ref = &deferred_singleton_closure_cache;
            let singleton_subsets_ref = &singleton_subsets;
            let aggregate_started = fast_support_profile.then(Instant::now);
            let scanned = wave
                .into_par_iter()
                .map_init(
                    || ParallelSupportScanScratch::new(dense_label_limit),
                    |scratch, (from_state, subset_entries)| {
                    for (nwa_state_id, path_weight) in &subset_entries {
                        if component_row_cache_enabled {
                            let key = (*nwa_state_id, path_weight.ptr_key());
                            let row = component_rows
                                .get(&key)
                                .expect("parallel support component row must be precomputed");
                            for (label, target, next_weight) in row.iter() {
                                scratch.push(*label, *target, next_weight.clone());
                            }
                            continue;
                        }

                        let state = &nwa.states()[*nwa_state_id as usize];
                        for (&label, targets) in &state.transitions {
                            for (target, transition_weight) in targets {
                                let next_weight = scratch
                                    .weight_ops
                                    .intersection(path_weight, transition_weight);
                                if !next_weight.is_empty() {
                                    scratch.push(label, *target, next_weight);
                                }
                            }
                        }
                    }

                    let mut cached_transitions = Vec::new();
                    let mut pending_labels = Vec::new();
                    let mut parallel_closure_hits = 0usize;
                    let mut parallel_subset_hits = 0usize;
                    for (label, mut contribs) in scratch.take_labels() {
                        if contribs.len() > 1 {
                            contribs.sort_unstable_by_key(|(state_id, _)| *state_id);
                            merge_sorted_target_contributions(&mut contribs, None);
                        }

                        // Existing direct singleton states can be resolved
                        // against the immutable wave-start cache in parallel.
                        // Only genuine singleton misses need serial interning.
                        if let [(state_id, weight)] = contribs.as_slice()
                            && nwa.states()[*state_id as usize].epsilons.is_empty()
                        {
                            let subset_weight_key = if normalize_singletons {
                                normalized_singleton_key
                            } else {
                                weight.ptr_key()
                            };
                            if let Some(to_state) =
                                singleton_subsets_ref.get(*state_id, subset_weight_key)
                            {
                                cached_transitions.push(DeferredTransition {
                                    from_state,
                                    label,
                                    to_state,
                                    edge_weight: DeferredEdgeWeight::Immediate(weight.clone()),
                                });
                                parallel_subset_hits += 1;
                            } else {
                                pending_labels.push((label, contribs));
                            }
                            continue;
                        }

                        let cached = match contribs.as_slice() {
                            [(state_id, weight)] => singleton_closure_cache_ref
                                .get(&(*state_id, weight.ptr_key()))
                                .cloned(),
                            _ => {
                                let key = contribs
                                    .iter()
                                    .map(|(state_id, weight)| (*state_id, weight.ptr_key()))
                                    .collect::<Vec<_>>();
                                closure_cache_ref.get(&key).cloned()
                            }
                        };
                        if let Some(cached) = cached {
                            cached_transitions.push(DeferredTransition {
                                from_state,
                                label,
                                to_state: cached.to_state,
                                edge_weight: cached.edge_weight,
                            });
                            parallel_closure_hits += 1;
                        } else {
                            pending_labels.push((label, contribs));
                        }
                    }
                    pending_labels.sort_unstable_by_key(|(label, _)| *label);
                    cached_transitions.sort_unstable_by_key(|transition| transition.label);
                    PrescannedSupportState {
                        from_state,
                        subset_entries,
                        cached_transitions,
                        pending_labels,
                        parallel_closure_hits,
                        parallel_subset_hits,
                    }
                })
                .collect::<Vec<_>>();
            if let Some(started) = aggregate_started {
                fast_state_aggregate_ms += elapsed_ms(started);
            }
            if fast_support_profile {
                fast_parallel_closure_hits += scanned
                    .iter()
                    .map(|state| state.parallel_closure_hits)
                    .sum::<usize>();
                fast_parallel_subset_hits += scanned
                    .iter()
                    .map(|state| state.parallel_subset_hits)
                    .sum::<usize>();
                fast_closure_hits += scanned
                    .iter()
                    .map(|state| state.parallel_closure_hits)
                    .sum::<usize>();
                fast_subset_hits += scanned
                    .iter()
                    .map(|state| state.parallel_subset_hits)
                    .sum::<usize>();
            }
            prescanned_states.extend(scanned);
        }

        let (from_state, subset_entries, prescanned_cached, prescanned_labels) =
            if let Some(state) = prescanned_states.pop_front() {
                (
                    state.from_state,
                    state.subset_entries,
                    Some(state.cached_transitions),
                    Some(state.pending_labels),
                )
            } else {
                let (from_state, subset_entries) = worklist
                    .pop_front()
                    .expect("support determinizer worklist unexpectedly empty");
                (from_state, subset_entries, None, None)
            };

        if let Some(detail) = detail.as_mut() {
            detail.states_processed += 1;
        }

        // Save subset entries for deferred parallel final weight computation.
        // Only save entries whose NWA states have final weights.
        let has_finals: DeferredFinalEntries = subset_entries.iter()
            .filter(|(nwa_state_id, _)| nwa.states()[*nwa_state_id as usize].final_weight.is_some())
            .map(|(id, w)| (*id, w.clone()))
            .collect();
        if !has_finals.is_empty() {
            deferred_final_entries.push((from_state, has_finals));
        }

        if prescanned_labels.is_none() {
            if fast_support_profile {
                fast_component_refs += subset_entries.len();
            }
            let scan_started = (detail.is_some() || fast_support_profile).then(Instant::now);
            for (nwa_state_id, path_weight) in &subset_entries {
                let state = &nwa.states()[*nwa_state_id as usize];
                if let Some(component_pairs) = profiled_component_pairs.as_mut() {
                    profiled_component_entries += 1;
                    if component_pairs.insert((*nwa_state_id, path_weight.ptr_key())) {
                        profiled_unique_component_transition_scans +=
                            state.transitions.values().map(Vec::len).sum::<usize>();
                    }
                }

                let cache_key = (*nwa_state_id, path_weight.ptr_key());
                let cached_row = component_row_cache_enabled.then(|| {
                    if let Some(row) = component_row_cache.get(&cache_key) {
                        return Arc::clone(row);
                    }
                    let mut row = Vec::new();
                    for (&label, targets) in &state.transitions {
                        for (target, transition_weight) in targets {
                            let next_weight =
                                intersection_cache.intersection(path_weight, transition_weight);
                            if !next_weight.is_empty() {
                                row.push((label, *target, next_weight));
                            }
                        }
                    }
                    let row = Arc::new(row);
                    component_row_cache.insert(cache_key, Arc::clone(&row));
                    row
                });

                if let Some(row) = cached_row {
                    for (label, target, next_weight) in row.iter() {
                        let target_weights = if *label >= 0 && (*label as usize) < dense_label_limit {
                            let label_idx = *label as usize;
                            if !dense_label_touched[label_idx] {
                                dense_label_touched[label_idx] = true;
                                touched_dense_labels.push(label_idx);
                            }
                            &mut dense_raw_targets[label_idx]
                        } else if *label == DEFAULT_LABEL {
                            default_touched = true;
                            &mut default_raw_targets
                        } else {
                            sparse_raw_targets.entry(*label).or_default()
                        };
                        target_weights.push((*target, next_weight.clone()));
                    }
                    continue;
                }

                for (&label, targets) in &state.transitions {
                    for (target, transition_weight) in targets {
                        if let Some(detail) = detail.as_mut() {
                            detail.outgoing_transitions_scanned += 1;
                            detail.intersection_calls += 1;
                        }
                        let next_weight =
                            intersection_cache.intersection(path_weight, transition_weight);
                        if next_weight.is_empty() {
                            continue;
                        }
                        if let Some(detail) = detail.as_mut() {
                            detail.nonempty_intersections += 1;
                        }

                        let target_weights = if label >= 0 && (label as usize) < dense_label_limit {
                            let label_idx = label as usize;
                            if !dense_label_touched[label_idx] {
                                dense_label_touched[label_idx] = true;
                                touched_dense_labels.push(label_idx);
                            }
                            &mut dense_raw_targets[label_idx]
                        } else if label == DEFAULT_LABEL {
                            default_touched = true;
                            &mut default_raw_targets
                        } else {
                            sparse_raw_targets.entry(label).or_default()
                        };
                        push_target_contribution_profiled(
                            target_weights,
                            *target,
                            next_weight,
                            detail.as_mut(),
                        );
                    }
                }
            }
            if let Some(started_at) = scan_started {
                let ms = elapsed_ms(started_at);
                if let Some(detail) = detail.as_mut() {
                    detail.intersection_scan_ms += ms;
                } else if fast_support_profile {
                    fast_serial_scan_ms += ms;
                }
            }
        }

        if let Some(cached) = prescanned_cached {
            if fast_support_profile {
                fast_labels += cached.len();
            }
            deferred_transitions.extend(cached);
        }

        let mut pre_closure_key: Vec<(u32, usize)> = Vec::new();
        let label_started = (detail.is_some() || fast_support_profile).then(Instant::now);

        let mut process_label = |label: i32, mut contribs: TargetContribs| {
            if contribs.is_empty() {
                return;
            }

            debug_assert!(contribs.iter().all(|(_, weight)| !weight.is_empty()));
            if fast_support_profile {
                fast_labels += 1;
                fast_label_contribs += contribs.len();
            }

            if let Some(detail) = detail.as_mut() {
                detail.labels_processed += 1;
                detail.label_contribs_sum += contribs.len();
                detail.label_contribs_max = detail.label_contribs_max.max(contribs.len());
            }
            let sort_started = detail.as_ref().map(|_| Instant::now());
            if contribs.len() > 1 {
                contribs.sort_unstable_by_key(|(state_id, _)| *state_id);
                merge_sorted_target_contributions(&mut contribs, detail.as_mut());
            }
            if let (Some(detail), Some(started_at)) = (detail.as_mut(), sort_started) {
                detail.contribution_sort_ms += elapsed_ms(started_at);
            }

            if contribs.len() == 1 {
                let (only_state, only_weight) = &contribs[0];
                if nwa.states()[*only_state as usize].epsilons.is_empty() {
                    if fast_support_profile {
                        fast_singleton_no_epsilon += 1;
                    }
                    let singleton_key = (
                        *only_state,
                        if normalize_singletons {
                            normalized_singleton_key
                        } else {
                            only_weight.ptr_key()
                        },
                    );
                    let subset_lookup_started = detail.as_ref().map(|_| Instant::now());
                    let to_state = if let Some(existing) = singleton_subsets.get(singleton_key.0, singleton_key.1) {
                        if let Some(detail) = detail.as_mut() {
                            detail.subset_intern_hits += 1;
                        }
                        if fast_support_profile {
                            fast_subset_hits += 1;
                        }
                        existing
                    } else {
                        if let Some(detail) = detail.as_mut() {
                            detail.subset_intern_misses += 1;
                        }
                        if fast_support_profile {
                            fast_subset_misses += 1;
                        }
                        let new_state = dwa.add_state();
                        subset_map.insert(vec![singleton_key], new_state);
                        singleton_subsets.insert(singleton_key.0, singleton_key.1, new_state);
                        worklist.push_back((
                            new_state,
                            vec![(
                                *only_state,
                                if normalize_singletons {
                                    normalized_singleton_weight.clone()
                                } else {
                                    only_weight.clone()
                                },
                            )],
                        ));
                        supports.push(vec![*only_state]);
                        new_state
                    };
                    if let (Some(detail), Some(started_at)) =
                        (detail.as_mut(), subset_lookup_started)
                    {
                        detail.subset_map_lookup_ms += elapsed_ms(started_at);
                    }
                    if defer_edge_unions {
                        deferred_transitions.push(DeferredTransition {
                            from_state,
                            label,
                            to_state,
                            edge_weight: DeferredEdgeWeight::Immediate(only_weight.clone()),
                        });
                    } else {
                        let add_transition_started = detail.as_ref().map(|_| Instant::now());
                        dwa.add_transition(from_state, label, to_state, only_weight.clone());
                        if let (Some(detail), Some(started_at)) =
                            (detail.as_mut(), add_transition_started)
                        {
                            detail.add_transition_ms += elapsed_ms(started_at);
                        }
                    }
                    return;
                }
            }

            let closure_key_started = detail.as_ref().map(|_| Instant::now());
            let singleton_closure_key = match contribs.as_slice() {
                [(state_id, weight)] => Some((*state_id, weight.ptr_key())),
                _ => None,
            };
            if singleton_closure_key.is_none() {
                pre_closure_key.clear();
                pre_closure_key.extend(contribs.iter().map(|(sid, w)| (*sid, w.ptr_key())));
                if let Some(detail) = detail.as_mut() {
                    detail.subset_key_constructions += 1;
                }
            }
            if let (Some(detail), Some(started_at)) = (detail.as_mut(), closure_key_started) {
                detail.closure_key_ms += elapsed_ms(started_at);
            }

            let closure_lookup_started = detail.as_ref().map(|_| Instant::now());
            if defer_edge_unions {
                let cached = match singleton_closure_key {
                    Some(key) => deferred_singleton_closure_cache.get(&key).cloned(),
                    None => deferred_closure_cache.get(&pre_closure_key).cloned(),
                };
                if let (Some(detail), Some(started_at)) = (detail.as_mut(), closure_lookup_started) {
                    detail.closure_lookup_ms += elapsed_ms(started_at);
                }
                if let Some(cached) = cached {
                    if let Some(detail) = detail.as_mut() {
                        detail.closure_cache_hits += 1;
                    }
                    if fast_support_profile {
                        fast_closure_hits += 1;
                    }
                    deferred_transitions.push(DeferredTransition {
                        from_state,
                        label,
                        to_state: cached.to_state,
                        edge_weight: cached.edge_weight,
                    });
                    return;
                }
            } else {
                let cached = match singleton_closure_key {
                    Some(key) => singleton_closure_cache.get(&key).cloned(),
                    None => closure_cache.get(&pre_closure_key).cloned(),
                };
                if let (Some(detail), Some(started_at)) = (detail.as_mut(), closure_lookup_started) {
                    detail.closure_lookup_ms += elapsed_ms(started_at);
                }
                if let Some(cached) = cached {
                    if let Some(detail) = detail.as_mut() {
                        detail.closure_cache_hits += 1;
                    }
                    if fast_support_profile {
                        fast_closure_hits += 1;
                    }
                    let add_transition_started = detail.as_ref().map(|_| Instant::now());
                    dwa.add_transition(from_state, label, cached.to_state, cached.edge_weight);
                    if let (Some(detail), Some(started_at)) =
                        (detail.as_mut(), add_transition_started)
                    {
                        detail.add_transition_ms += elapsed_ms(started_at);
                    }
                    return;
                }
            }

            if let Some(detail) = detail.as_mut() {
                detail.closure_cache_misses += 1;
            }
            if fast_support_profile {
                fast_closure_misses += 1;
            }
            let deferred_edge_weight = defer_edge_unions.then(|| defer_edge_weight(&contribs));
            let edge_weight = if defer_edge_unions {
                None
            } else {
                let edge_weight_started = detail.as_ref().map(|_| Instant::now());
                let edge_weight = union_cache.union_all(contribs.iter().map(|(_, weight)| weight));
                if let (Some(detail), Some(started_at)) = (detail.as_mut(), edge_weight_started) {
                    detail.edge_weight_union_ms += elapsed_ms(started_at);
                }
                if edge_weight.is_empty() {
                    return;
                }
                Some(edge_weight)
            };
            let closure_started = detail.as_ref().map(|_| Instant::now());
            let mut owned_canon = Vec::new();
            if use_flat_canonical_closure {
                local_epsilon_closure_canonical(
                    nwa,
                    &mut weight_by_state,
                    &mut closure_queue,
                    &contribs,
                    &mut closure_touched_states,
                    &mut closure_canon,
                    &mut intersection_cache,
                    use_epsilon_closure_weight_cache,
                );
            } else {
                let mut target_subset: FxHashMap<u32, Weight> = contribs
                    .iter()
                    .map(|(state_id, weight)| (*state_id, weight.clone()))
                    .collect();
                local_epsilon_closure(
                    nwa,
                    &mut weight_by_state,
                    &mut closure_queue,
                    &mut target_subset,
                );
                owned_canon = target_subset
                    .iter()
                    .filter(|(_, weight)| !weight.is_empty())
                    .map(|(state_id, weight)| (*state_id, weight.clone()))
                    .collect();
                owned_canon.sort_unstable_by_key(|(state_id, _)| *state_id);
            }
            if let (Some(detail), Some(started_at)) = (detail.as_mut(), closure_started) {
                detail.local_epsilon_closure_miss_ms += elapsed_ms(started_at);
            }
            let raw_canon = if use_flat_canonical_closure {
                closure_canon.as_slice()
            } else {
                owned_canon.as_slice()
            };
            if raw_canon.is_empty() {
                return;
            }

            // Weighted subset normalization: factor the union of residuals onto
            // the incoming edge. For E=union(w_i), replacing each residual w_i
            // by w_i union !E is exact because E intersect (w_i union !E)=w_i.
            // The singleton case reduces to the cheaper `(q,w)->(q,all)` rule.
            let mut normalized_canon_owned = Vec::<(u32, Weight)>::new();
            let mut normalized_closure_edge_weight: Option<Weight> = None;
            let canon = if normalize_subsets && raw_canon.len() > 1 {
                let factored_edge = Weight::union_all_direct(raw_canon.iter().map(|(_, weight)| weight));
                if factored_edge.is_empty() {
                    return;
                }
                let outside = factored_edge.complement();
                normalized_canon_owned.reserve(raw_canon.len());
                if outside.is_empty() {
                    normalized_canon_owned.extend(raw_canon.iter().cloned());
                } else {
                    normalized_canon_owned.extend(
                        raw_canon
                            .iter()
                            .map(|(state_id, weight)| (*state_id, weight.union(&outside))),
                    );
                }
                normalized_closure_edge_weight = Some(factored_edge);
                normalized_canon_owned.as_slice()
            } else {
                raw_canon
            };

            let subset_lookup_started = detail.as_ref().map(|_| Instant::now());
            let to_state = if let [(only_state, only_weight)] = canon {
                let singleton_key = (
                    *only_state,
                    if normalize_singletons {
                        normalized_singleton_key
                    } else {
                        only_weight.ptr_key()
                    },
                );
                if normalize_singletons {
                    // The closure residual belongs on the incoming DWA edge if
                    // the singleton state itself is canonicalized to `(q, all)`.
                    normalized_closure_edge_weight = Some(only_weight.clone());
                }
                if let Some(existing) = singleton_subsets.get(singleton_key.0, singleton_key.1) {
                    if let Some(detail) = detail.as_mut() {
                        detail.subset_intern_hits += 1;
                    }
                    if fast_support_profile {
                        fast_subset_hits += 1;
                    }
                    existing
                } else {
                    if let Some(detail) = detail.as_mut() {
                        detail.subset_intern_misses += 1;
                    }
                    if fast_support_profile {
                        fast_subset_misses += 1;
                    }
                    let new_state = dwa.add_state();
                    subset_map.insert(vec![singleton_key], new_state);
                    singleton_subsets.insert(singleton_key.0, singleton_key.1, new_state);
                    worklist.push_back((
                        new_state,
                        vec![(
                            *only_state,
                            if normalize_singletons {
                                normalized_singleton_weight.clone()
                            } else {
                                only_weight.clone()
                            },
                        )],
                    ));
                    supports.push(vec![*only_state]);
                    new_state
                }
            } else {
                let subset_key_started = detail.as_ref().map(|_| Instant::now());
                key_buf.clear();
                key_buf.extend(canon.iter().map(|(sid, w)| (*sid, w.ptr_key())));
                if let Some(detail) = detail.as_mut() {
                    detail.subset_key_constructions += 1;
                }
                if let (Some(detail), Some(started_at)) = (detail.as_mut(), subset_key_started) {
                    detail.post_closure_subset_key_ms += elapsed_ms(started_at);
                }
                if let Some(existing) = subset_map.get(&key_buf).copied() {
                    if let Some(detail) = detail.as_mut() {
                        detail.subset_intern_hits += 1;
                    }
                    if fast_support_profile {
                        fast_subset_hits += 1;
                    }
                    existing
                } else {
                    if let Some(detail) = detail.as_mut() {
                        detail.subset_intern_misses += 1;
                    }
                    if fast_support_profile {
                        fast_subset_misses += 1;
                    }
                    let new_state = dwa.add_state();
                    subset_map.insert(key_buf.clone(), new_state);
                    worklist.push_back((new_state, canon.to_vec()));
                    supports.push(canon.iter().map(|(sid, _)| *sid).collect());
                    new_state
                }
            };
            if let (Some(detail), Some(started_at)) = (detail.as_mut(), subset_lookup_started) {
                detail.subset_map_lookup_ms += elapsed_ms(started_at);
            }
            if defer_edge_unions {
                let edge_weight = if let Some(weight) = normalized_closure_edge_weight.clone() {
                    DeferredEdgeWeight::Immediate(weight)
                } else {
                    deferred_edge_weight
                        .expect("deferred parser support edge must retain its union job")
                };
                let cached = DeferredClosure {
                    to_state,
                    edge_weight: edge_weight.clone(),
                };
                if let Some(key) = singleton_closure_key {
                    deferred_singleton_closure_cache.insert(key, cached);
                } else {
                    deferred_closure_cache.insert(pre_closure_key.clone(), cached);
                }
                deferred_transitions.push(DeferredTransition {
                    from_state,
                    label,
                    to_state,
                    edge_weight,
                });
            } else {
                let edge_weight = normalized_closure_edge_weight
                    .clone()
                    .or(edge_weight)
                    .expect("eager support edge union must exist");
                let cached = CachedClosure {
                    to_state,
                    edge_weight: edge_weight.clone(),
                };
                if let Some(key) = singleton_closure_key {
                    singleton_closure_cache.insert(key, cached);
                } else {
                    closure_cache.insert(pre_closure_key.clone(), cached);
                }
                let add_transition_started = detail.as_ref().map(|_| Instant::now());
                dwa.add_transition(from_state, label, to_state, edge_weight);
                if let (Some(detail), Some(started_at)) = (detail.as_mut(), add_transition_started) {
                    detail.add_transition_ms += elapsed_ms(started_at);
                }
            }
        };

        if let Some(labels) = prescanned_labels {
            for (label, contribs) in labels {
                process_label(label, contribs);
            }
        } else {
            for label_idx in touched_dense_labels.drain(..) {
                dense_label_touched[label_idx] = false;
                process_label(label_idx as i32, std::mem::take(&mut dense_raw_targets[label_idx]));
            }

            if default_touched {
                default_touched = false;
                process_label(DEFAULT_LABEL, std::mem::take(&mut default_raw_targets));
            }

            for (label, contribs) in sparse_raw_targets.drain() {
                process_label(label, contribs);
            }
        }
        if let Some(started_at) = label_started {
            let ms = elapsed_ms(started_at);
            if let Some(detail) = detail.as_mut() {
                detail.label_processing_ms += ms;
            } else if fast_support_profile {
                fast_label_process_ms += ms;
            }
        }
    }

    if defer_edge_unions {
        use rayon::prelude::*;
        let union_started = (detail.is_some() || fast_support_profile).then(Instant::now);
        let union_results = deferred_union_jobs
            .par_iter()
            .map(|weights| Weight::union_all_direct(weights.iter()))
            .collect::<Vec<_>>();
        if let Some(started_at) = union_started {
            let ms = elapsed_ms(started_at);
            if let Some(detail) = detail.as_mut() {
                detail.edge_weight_union_ms += ms;
                detail.union_cache_ms += ms;
            } else if fast_support_profile {
                fast_deferred_union_ms += ms;
            }
        }
        let add_started = (detail.is_some() || fast_support_profile).then(Instant::now);
        let resolved = deferred_transitions
            .into_par_iter()
            .map(|transition| {
                let edge_weight = match transition.edge_weight {
                    DeferredEdgeWeight::Immediate(weight) => weight,
                    DeferredEdgeWeight::Job(job) => union_results[job].clone(),
                };
                debug_assert!(!edge_weight.is_empty());
                (
                    transition.from_state,
                    transition.label,
                    transition.to_state,
                    edge_weight,
                )
            })
            .collect::<Vec<_>>();

        let mut rows: Vec<Vec<(i32, (u32, Weight))>> =
            (0..dwa.states().len()).map(|_| Vec::new()).collect();
        for (from_state, label, to_state, edge_weight) in resolved {
            rows[from_state as usize].push((label, (to_state, edge_weight)));
        }
        dwa.states_mut()
            .par_iter_mut()
            .zip(rows.into_par_iter())
            .for_each(|(state, row)| {
                if !row.is_empty() {
                    state.transitions = row.into_iter().collect();
                }
            });
        if let Some(started_at) = add_started {
            let ms = elapsed_ms(started_at);
            if let Some(detail) = detail.as_mut() {
                detail.add_transition_ms += ms;
            } else if fast_support_profile {
                fast_transition_materialize_ms += ms;
            }
        }
    }

    let mut final_signature_ids: FxHashMap<Vec<(usize, Vec<usize>)>, usize> = FxHashMap::default();
    let mut final_signature_groups: Vec<FinalGroups> = Vec::new();
    let mut final_jobs: Vec<(u32, usize)> = Vec::with_capacity(deferred_final_entries.len());
    let final_grouping_started = (detail.is_some() || fast_support_profile).then(Instant::now);

    let build_signature = |entries: &DeferredFinalEntries| {
        let mut groups: SmallVec<[(usize, SmallVec<[usize; 4]>); 4]> = SmallVec::new();
        for (nwa_state_id, path_weight) in entries {
            let Some(state_final) = nwa.states()[*nwa_state_id as usize].final_weight.as_ref() else {
                continue;
            };
            let final_key = state_final.ptr_key();
            if let Some((_, path_keys)) = groups
                .iter_mut()
                .find(|(existing_final_key, _)| *existing_final_key == final_key)
            {
                path_keys.push(path_weight.ptr_key());
            } else {
                groups.push((final_key, smallvec::smallvec![path_weight.ptr_key()]));
            }
        }
        groups.sort_unstable_by_key(|(final_key, _)| *final_key);
        groups
            .into_iter()
            .map(|(final_key, mut path_keys)| {
                path_keys.sort_unstable();
                path_keys.dedup();
                (final_key, path_keys.into_vec())
            })
            .collect::<Vec<_>>()
    };

    let parallel_signature_grouping = detail.is_none()
        && rayon::current_num_threads() > 1
        && deferred_final_entries.len() >= 4_096
        && std::env::var_os("GLRMASK_DISABLE_PARSER_FINAL_PARALLEL_SIGNATURES").is_none();
    let mut prepared_signatures = if parallel_signature_grouping {
        use rayon::prelude::*;
        deferred_final_entries
            .par_iter()
            .map(|(_, entries)| build_signature(entries))
            .collect::<Vec<_>>()
    } else {
        deferred_final_entries
            .iter()
            .map(|(_, entries)| build_signature(entries))
            .collect::<Vec<_>>()
    };

    if let Some(detail) = detail.as_mut() {
        detail.final_weight_entries = deferred_final_entries
            .iter()
            .map(|(_, entries)| entries.len())
            .sum();
        detail.final_weight_entries_max = deferred_final_entries
            .iter()
            .map(|(_, entries)| entries.len())
            .max()
            .unwrap_or(0);
    }

    for (index, (state_id, entries)) in deferred_final_entries.iter().enumerate() {
        let signature = std::mem::take(&mut prepared_signatures[index]);
        let signature_id = match final_signature_ids.entry(signature) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => {
                // Only materialize owned Weight groups for a genuinely new
                // signature. Most parser-final signatures are repeats.
                let mut groups: SmallVec<[(usize, Weight, FinalPathWeights); 4]> = SmallVec::new();
                for (nwa_state_id, path_weight) in entries {
                    let Some(state_final) = nwa.states()[*nwa_state_id as usize].final_weight.as_ref() else {
                        continue;
                    };
                    let final_key = state_final.ptr_key();
                    if let Some((_, _, path_weights)) = groups
                        .iter_mut()
                        .find(|(existing_final_key, _, _)| *existing_final_key == final_key)
                    {
                        path_weights.push(path_weight.clone());
                    } else {
                        groups.push((
                            final_key,
                            state_final.clone(),
                            smallvec::smallvec![path_weight.clone()],
                        ));
                    }
                }
                groups.sort_unstable_by_key(|(final_key, _, _)| *final_key);
                for (_, _, path_weights) in &mut groups {
                    path_weights.sort_unstable_by_key(Weight::ptr_key);
                    path_weights.dedup_by_key(|weight| weight.ptr_key());
                }
                let signature_id = final_signature_groups.len();
                final_signature_groups.push(
                    groups
                        .into_iter()
                        .map(|(_, state_final, path_weights)| (state_final, path_weights))
                        .collect(),
                );
                entry.insert(signature_id);
                signature_id
            }
        };
        final_jobs.push((*state_id, signature_id));
    }
    if let Some(started_at) = final_grouping_started {
        let ms = elapsed_ms(started_at);
        if let Some(detail) = detail.as_mut() {
            detail.final_grouping_ms += ms;
        }
    }
    if let Some(detail) = detail.as_mut() {
        detail.final_weight_states = final_jobs.len();
        detail.final_weight_signature_distinct = final_signature_groups.len();
        detail.final_weight_signature_hit_potential =
            final_jobs.len().saturating_sub(final_signature_groups.len());
    }

    // Compute final weights in parallel once per distinct final-weight signature.
    let fast_final_started = fast_support_profile.then(Instant::now);
    {
        use rayon::prelude::*;
        let detail_enabled = detail.is_some();
        let final_weights_by_signature: Vec<Option<Weight>> = {
            let intern_started_at = Instant::now();
            let mut component_ids = FxHashMap::<(usize, Vec<usize>), usize>::default();
            let mut components = Vec::<(Weight, SmallVec<[Weight; 4]>)>::new();
            let signature_components: Vec<SmallVec<[usize; 8]>> = final_signature_groups
                .iter()
                .map(|groups| {
                    groups
                        .iter()
                        .map(|(final_w, path_weights)| {
                            let key = (
                                final_w.ptr_key(),
                                path_weights.iter().map(Weight::ptr_key).collect::<Vec<_>>(),
                            );
                            if let Some(&component_id) = component_ids.get(&key) {
                                component_id
                            } else {
                                let component_id = components.len();
                                component_ids.insert(key, component_id);
                                components.push((final_w.clone(), path_weights.clone()));
                                component_id
                            }
                        })
                        .collect::<SmallVec<[usize; 8]>>()
                })
                .collect::<Vec<_>>();
            let intern_ms = elapsed_ms(intern_started_at);
            let parallel_min_components = std::env::var("GLRMASK_PARSER_FINAL_PARALLEL_MIN_COMPONENTS")
                .ok()
                .and_then(|value| value.trim().parse::<usize>().ok())
                .unwrap_or(512);
            let rayon_workers = rayon::current_num_threads();
            let parallel_min_signatures = std::env::var("GLRMASK_PARSER_FINAL_PARALLEL_MIN_SIGNATURES")
                .ok()
                .and_then(|value| value.trim().parse::<usize>().ok())
                // Above 48 workers, continuing to scale the threshold makes
                // larger machines *less* likely to parallelize the same exact
                // signature workload. Cap at the 48-worker crossover instead.
                .unwrap_or_else(|| 128usize.max(rayon_workers.saturating_mul(4).min(192)));
            let has_parallel_workers = rayon_workers > 1;
            let use_parallel_final_components = has_parallel_workers
                && components.len() >= parallel_min_components;
            let use_parallel_final_signatures = has_parallel_workers
                && signature_components.len() >= parallel_min_signatures;
            let compute_component = |weight_ops: &mut ScopedWeightOpCache,
                                     (final_w, path_weights): &(Weight, SmallVec<[Weight; 4]>)| {
                let path_started = detail_enabled.then(Instant::now);
                let path_union = weight_ops.union_all(path_weights.iter());
                let path_ms = path_started.map(elapsed_ms).unwrap_or(0.0);
                let intersection_started = detail_enabled.then(Instant::now);
                let contribution = weight_ops.intersection(&path_union, final_w);
                let intersection_ms = intersection_started.map(elapsed_ms).unwrap_or(0.0);
                (
                    (!contribution.is_empty()).then_some(contribution),
                    path_ms,
                    intersection_ms,
                )
            };
            let component_results: Vec<(Option<Weight>, f64, f64)> =
                if use_parallel_final_components {
                    components
                        .par_iter()
                        .map_init(ScopedWeightOpCache::default, compute_component)
                        .collect()
                } else {
                    let mut weight_ops = ScopedWeightOpCache::default();
                    components
                        .iter()
                        .map(|component| compute_component(&mut weight_ops, component))
                        .collect()
                };
            if let Some(detail) = detail.as_mut() {
                detail.final_path_union_ms +=
                    component_results.iter().map(|(_, ms, _)| *ms).sum::<f64>();
                detail.final_intersection_ms +=
                    component_results.iter().map(|(_, _, ms)| *ms).sum::<f64>();
            }
            let output_started_at = Instant::now();
            let compute_signature = |component_ids: &SmallVec<[usize; 8]>| {
                let weights = component_ids
                    .iter()
                    .filter_map(|&component_id| component_results[component_id].0.as_ref());
                let direct_final_union =
                    std::env::var_os("GLRMASK_DISABLE_PARSER_FINAL_DIRECT_UNION").is_none()
                        && (std::env::var_os("GLRMASK_PARSER_FINAL_DIRECT_UNION").is_some()
                            || component_ids.len() >= 5);
                let weight = if direct_final_union {
                    Weight::union_all_direct(weights)
                } else {
                    Weight::union_all(weights)
                };
                (!weight.is_empty()).then_some(weight)
            };
            let results = if use_parallel_final_signatures {
                signature_components
                    .par_iter()
                    .map(compute_signature)
                    .collect::<Vec<_>>()
            } else {
                signature_components
                    .iter()
                    .map(compute_signature)
                    .collect::<Vec<_>>()
            };
            if std::env::var_os("GLRMASK_VALIDATE_INTERNED_FINAL_GROUPS").is_some() {
                let reference = final_signature_groups
                    .iter()
                    .map(|final_groups| {
                        let contributions = final_groups
                            .iter()
                            .filter_map(|(final_w, path_weights)| {
                                let path_union = Weight::union_all(path_weights.iter());
                                let contribution = path_union.intersection(final_w);
                                (!contribution.is_empty()).then_some(contribution)
                            })
                            .collect::<SmallVec<[Weight; 4]>>();
                        let result = Weight::union_all(contributions.iter());
                        (!result.is_empty()).then_some(result)
                    })
                    .collect::<Vec<_>>();
                assert_eq!(
                    results, reference,
                    "interned parser final-weight components changed the weighted language",
                );
            }
            if let Some(detail) = detail.as_mut() {
                detail.final_output_union_ms += elapsed_ms(output_started_at);
            }
            if compile_profile_enabled() {
                let total_components = signature_components.iter().map(|ids| ids.len()).sum::<usize>();
                eprintln!(
                    "[glrmask/profile][parser_final_group_intern] signatures={} total_components={} unique_components={} intern_ms={:.3}",
                    signature_components.len(),
                    total_components,
                    components.len(),
                    intern_ms,
                );
            }
            results
        };
        for (state_id, signature_id) in final_jobs {
            if let Some(weight) = &final_weights_by_signature[signature_id] {
                dwa.set_final_weight(state_id, weight.clone());
            }
        }
    }
    if let Some(started) = fast_final_started {
        fast_final_weights_ms = elapsed_ms(started);
    }

    if let Some(detail) = detail.as_mut() {
        if defer_edge_unions {
            detail.union_cache_hits = deferred_hits;
            detail.union_cache_misses = deferred_misses;
            detail.union_cache_key_len_sum = deferred_key_len_sum;
            detail.union_cache_key_len_max = deferred_key_len_max;
        } else {
            detail.union_cache_hits = union_cache.hits;
            detail.union_cache_misses = union_cache.misses;
            detail.union_cache_key_len_sum = union_cache.key_len_sum;
            detail.union_cache_key_len_max = union_cache.key_len_max;
            detail.union_cache_ms = union_cache.total_ms;
        }
    }

    if fast_support_profile {
        eprintln!(
            "[glrmask/profile][parser_support_fast] nwa_states={} dwa_states={} waves={} wave_states={} component_refs={} component_rows={} component_row_edges={} component_cache_entries={} labels={} label_contribs={} singleton_no_epsilon={} closure_hits={} parallel_closure_hits={} closure_misses={} subset_hits={} parallel_subset_hits={} subset_misses={} component_key_ms={:.3} component_compute_ms={:.3} state_aggregate_ms={:.3} serial_scan_ms={:.3} label_process_ms={:.3} deferred_union_ms={:.3} transition_materialize_ms={:.3} final_weights_ms={:.3}",
            nwa.states().len(),
            dwa.states().len(),
            fast_wave_count,
            fast_wave_states,
            fast_component_refs,
            fast_component_rows_computed,
            fast_component_row_edges,
            component_row_cache.len(),
            fast_labels,
            fast_label_contribs,
            fast_singleton_no_epsilon,
            fast_closure_hits,
            fast_parallel_closure_hits,
            fast_closure_misses,
            fast_subset_hits,
            fast_parallel_subset_hits,
            fast_subset_misses,
            fast_component_key_ms,
            fast_component_compute_ms,
            fast_state_aggregate_ms,
            fast_serial_scan_ms,
            fast_label_process_ms,
            fast_deferred_union_ms,
            fast_transition_materialize_ms,
            fast_final_weights_ms,
        );
    }

    if let Some(detail) = detail {
        detail.emit("support");
        if let Some(component_pairs) = profiled_component_pairs {
            eprintln!(
                "[glrmask/profile][parser_support_components] component_entries={} unique_component_pairs={} pair_reuse={} unique_component_transition_scans={} actual_transition_scans={}",
                profiled_component_entries,
                component_pairs.len(),
                profiled_component_entries.saturating_sub(component_pairs.len()),
                profiled_unique_component_transition_scans,
                detail.outgoing_transitions_scanned,
            );
        }
    }

    DeterminizedDwaWithSupports { dwa, supports }
}


// Compilation may use a larger exact private observation algebra than the
// compact runtime format. Keep the common <=16-row case inline and allocate
// only the actually used rows for larger compile-time domains. Runtime
// SmallBoundaryDwa publication remains capped at its existing 16-row format.
const FAST_BOUNDARY_TSID_LIMIT: usize = 64;
type FastBoundaryWeightValue = SmallVec<[u64; 16]>;
type FastBoundaryWeightId = u32;
type FastBoundaryContribs = SmallVec<[(u32, FastBoundaryWeightId); 4]>;

#[derive(Clone, Copy)]
struct FiniteCompileLimits {
    states: usize,
    edges: usize,
    weights: usize,
    words: usize,
    work: usize,
}

impl Default for FiniteCompileLimits {
    fn default() -> Self {
        Self { states: 200_000, edges: 4_000_000, weights: 500_000,
               words: 8 * 1024 * 1024, work: 20_000_000 }
    }
}

struct FastBoundaryWeightInterner {
    // Only the new finite-output route enables this policy. A failed private
    // computation is discarded before any graph is published. Returning the
    // existing empty ID after exhaustion is an internal unwind aid, not a
    // partial mask result: callers must and do return None on failure.
    limits: Option<FiniteCompileLimits>,
    failed: bool,
    work: usize,

    values: Vec<FastBoundaryWeightValue>,
    ids: FxHashMap<FastBoundaryWeightValue, FastBoundaryWeightId>,
    intersections: FxHashMap<(FastBoundaryWeightId, FastBoundaryWeightId), FastBoundaryWeightId>,
    unions: FxHashMap<(FastBoundaryWeightId, FastBoundaryWeightId), FastBoundaryWeightId>,
    differences: FxHashMap<(FastBoundaryWeightId, FastBoundaryWeightId), FastBoundaryWeightId>,
    tsid_count: usize,
    token_count: usize,
    all_token_mask: u64,
    borrowed_intern: bool,
    source_last_cache_enabled: bool,
    last_source_ptr: usize,
    last_source_id: FastBoundaryWeightId,
}

impl FastBoundaryWeightInterner {
    fn new(tsid_count: usize, token_count: usize) -> Option<Self> {
        if tsid_count == 0
            || tsid_count > FAST_BOUNDARY_TSID_LIMIT
            || token_count == 0
            || token_count > 64
        {
            return None;
        }
        let all_token_mask = if token_count == 64 {
            u64::MAX
        } else {
            (1u64 << token_count) - 1
        };
        let empty: FastBoundaryWeightValue = smallvec::smallvec![0u64; tsid_count];
        let mut all = empty.clone();
        all[..tsid_count].fill(all_token_mask);
        let mut ids = FxHashMap::default();
        ids.insert(empty.clone(), 0);
        ids.insert(all.clone(), 1);
        Some(Self {
            limits: None,
            failed: false,
            work: 0,
            values: vec![empty, all],
            ids,
            intersections: FxHashMap::default(),
            unions: FxHashMap::default(),
            differences: FxHashMap::default(),
            tsid_count,
            token_count,
            all_token_mask,
            borrowed_intern: std::env::var_os("GLRMASK_BOUNDARY_BORROWED_WEIGHT_INTERN").is_some(),
            source_last_cache_enabled: std::env::var_os(
                "GLRMASK_EXPERIMENT_SMALL_BOUNDARY_SOURCE_LAST_CACHE",
            )
            .is_some(),
            last_source_ptr: usize::MAX,
            last_source_id: 0,
        })
    }

    #[inline]
    fn empty_id(&self) -> FastBoundaryWeightId {
        0
    }

    #[inline]
    fn all_id(&self) -> FastBoundaryWeightId {
        1
    }

    fn intern(&mut self, value: FastBoundaryWeightValue) -> FastBoundaryWeightId {
        if let Some(&id) = self.ids.get(&value) {
            return id;
        }
        if let Some(limits) = self.limits {
            if self.failed || self.values.len() >= limits.weights
                || (self.values.len() + 1).saturating_mul(self.tsid_count) > limits.words {
                self.failed = true;
                return 0;
            }
        }
        let id = self.values.len() as u32;
        self.values.push(value.clone());
        self.ids.insert(value, id);
        id
    }

    // Look up a borrowed stack buffer before allocating an owned vector. Exact
    // values, insertion order, weight IDs and resource-decline semantics match
    // the original intern() path.
    fn intern_slice(&mut self, value:&[u64])->FastBoundaryWeightId {
        if let Some(&id)=self.ids.get(value){return id;}
        if let Some(limits)=self.limits{
            if self.failed || self.values.len()>=limits.weights
                ||(self.values.len()+1).saturating_mul(self.tsid_count)>limits.words{
                self.failed=true;return 0;
            }
        }
        let id=self.values.len()as u32;
        let owned=FastBoundaryWeightValue::from_slice(value);
        self.values.push(owned.clone());self.ids.insert(owned,id);id
    }

    #[inline]
    fn allow_work(&mut self, amount: usize, states: usize, edges: usize) -> bool {
        if let Some(limits) = self.limits {
            self.work = self.work.saturating_add(amount);
            self.failed |= self.work > limits.work || states > limits.states || edges > limits.edges;
        }
        !self.failed
    }

    fn compact_runtime_weights(&self) -> Option<Vec<[u64; 16]>> {
        if self.tsid_count > 16 { return None; }
        Some(self.values.iter().map(|value| {
            let mut row = [0u64; 16];
            row[..self.tsid_count].copy_from_slice(value);
            row
        }).collect())
    }

    fn source_weight_id(
        &mut self,
        weight: &Weight,
        by_ptr: &mut FxHashMap<usize, FastBoundaryWeightId>,
        source_tsid_map: Option<&[u32]>,
    ) -> Option<FastBoundaryWeightId> {
        if weight.is_empty() {
            return Some(self.empty_id());
        }
        if weight.is_full() {
            return Some(self.all_id());
        }
        let ptr = weight.ptr_key();
        if source_tsid_map.is_none()
            && self.source_last_cache_enabled
            && ptr == self.last_source_ptr
        {
            return Some(self.last_source_id);
        }
        if let Some(&id) = by_ptr.get(&ptr) {
            if source_tsid_map.is_none() && self.source_last_cache_enabled {
                self.last_source_ptr = ptr;
                self.last_source_id = id;
            }
            return Some(id);
        }
        let mut value: FastBoundaryWeightValue = smallvec::smallvec![0u64; self.tsid_count];
        for (start, end, tokens) in weight.range_entries() {
            if source_tsid_map.is_none() && end as usize >= self.tsid_count {
                return None;
            }
            if let Some(map) = source_tsid_map
                && end as usize >= map.len()
            {
                return None;
            }
            let mut mask = 0u64;
            for range in tokens.ranges() {
                for token in range {
                    if token as usize >= self.token_count || token >= 64 {
                        return None;
                    }
                    mask |= 1u64 << token;
                }
            }
            for source_tsid in start..=end {
                let target_tsid = source_tsid_map
                    .map(|map| map[source_tsid as usize])
                    .unwrap_or(source_tsid);
                if target_tsid as usize >= self.tsid_count {
                    return None;
                }
                let slot = &mut value[target_tsid as usize];
                // The quotient contract requires every source TSID in one
                // class to agree on every source weight. A mismatch means the
                // caller supplied an invalid quotient, so abandon the fast path.
                if *slot != 0 && *slot != mask {
                    return None;
                }
                *slot = mask;
            }
        }
        let id = self.intern(value);
        by_ptr.insert(ptr, id);
        if source_tsid_map.is_none() && self.source_last_cache_enabled {
            self.last_source_ptr = ptr;
            self.last_source_id = id;
        }
        Some(id)
    }

    #[inline]
    fn intersection(
        &mut self,
        left: FastBoundaryWeightId,
        right: FastBoundaryWeightId,
    ) -> FastBoundaryWeightId {
        if left == 0 || right == 0 {
            return 0;
        }
        if left == 1 {
            return right;
        }
        if right == 1 || left == right {
            return left;
        }
        let key = if left <= right { (left, right) } else { (right, left) };
        if let Some(&id) = self.intersections.get(&key) {
            return id;
        }
        let id=if self.borrowed_intern{
            let mut buffer=[0u64;FAST_BOUNDARY_TSID_LIMIT];
            for (slot,(&a,&b)) in buffer[..self.tsid_count].iter_mut()
                .zip(self.values[left as usize].iter().zip(&self.values[right as usize])){
                *slot=a & b;
            }
            self.intern_slice(&buffer[..self.tsid_count])
        }else{
        let mut value: FastBoundaryWeightValue = smallvec::smallvec![0u64; self.tsid_count];
        for (slot, (&a, &b)) in value
            .iter_mut()
            .zip(self.values[left as usize].iter().zip(&self.values[right as usize]))
            .take(self.tsid_count)
        {
            *slot = a & b;
        }
        self.intern(value)
        };
        if self.limits.is_none() || self.intersections.len() < 262_144 {
            self.intersections.insert(key, id);
        }
        id
    }

    #[inline]
    fn union(
        &mut self,
        left: FastBoundaryWeightId,
        right: FastBoundaryWeightId,
    ) -> FastBoundaryWeightId {
        if left == 0 {
            return right;
        }
        if right == 0 || left == right {
            return left;
        }
        if left == 1 || right == 1 {
            return 1;
        }
        let key = if left <= right { (left, right) } else { (right, left) };
        if let Some(&id) = self.unions.get(&key) {
            return id;
        }
        let id=if self.borrowed_intern{
            let mut buffer=[0u64;FAST_BOUNDARY_TSID_LIMIT];
            for (slot,(&a,&b)) in buffer[..self.tsid_count].iter_mut()
                .zip(self.values[left as usize].iter().zip(&self.values[right as usize])){
                *slot=a | b;
            }
            self.intern_slice(&buffer[..self.tsid_count])
        }else{
        let mut value: FastBoundaryWeightValue = smallvec::smallvec![0u64; self.tsid_count];
        for (slot, (&a, &b)) in value
            .iter_mut()
            .zip(self.values[left as usize].iter().zip(&self.values[right as usize]))
            .take(self.tsid_count)
        {
            *slot = a | b;
        }
        self.intern(value)
        };
        if self.limits.is_none() || self.unions.len() < 262_144 {
            self.unions.insert(key, id);
        }
        id
    }

    #[inline]
    fn difference(
        &mut self,
        left: FastBoundaryWeightId,
        right: FastBoundaryWeightId,
    ) -> FastBoundaryWeightId {
        if left == 0 || right == 1 || left == right {
            return 0;
        }
        if right == 0 {
            return left;
        }
        let key = (left, right);
        if let Some(&id) = self.differences.get(&key) {
            return id;
        }
        let id=if self.borrowed_intern{
            let mut buffer=[0u64;FAST_BOUNDARY_TSID_LIMIT];
            for (slot,(&a,&b)) in buffer[..self.tsid_count].iter_mut()
                .zip(self.values[left as usize].iter().zip(&self.values[right as usize])){
                *slot=a & !b;
            }
            self.intern_slice(&buffer[..self.tsid_count])
        }else{
        let mut value: FastBoundaryWeightValue = smallvec::smallvec![0u64; self.tsid_count];
        for (slot, (&a, &b)) in value
            .iter_mut()
            .zip(self.values[left as usize].iter().zip(&self.values[right as usize]))
            .take(self.tsid_count)
        {
            *slot = a & !b;
        }
        self.intern(value)
        };
        if self.limits.is_none() || self.differences.len() < 262_144 {
            self.differences.insert(key, id);
        }
        id
    }

    fn is_subset(
        &self,
        left: FastBoundaryWeightId,
        right: FastBoundaryWeightId,
    ) -> bool {
        if left == 0 || right == 1 || left == right {
            return true;
        }
        if right == 0 {
            return false;
        }
        self.values[left as usize]
            .iter()
            .zip(&self.values[right as usize])
            .take(self.tsid_count)
            .all(|(&a, &b)| a & !b == 0)
    }

    fn to_weight(&self, id: FastBoundaryWeightId) -> Weight {
        if id == 0 {
            return Weight::empty();
        }
        if id == 1 {
            return Weight::all();
        }
        let value = &self.values[id as usize];
        Weight::from_per_tsid_token_sets(
            value[..self.tsid_count]
                .iter()
                .enumerate()
                .filter_map(|(tsid, &mask)| {
                    if mask == 0 {
                        return None;
                    }
                    let tokens = range_set_blaze::RangeSetBlaze::from_iter(
                        (0..self.token_count as u32)
                            .filter(|&token| mask & (1u64 << token) != 0),
                    );
                    Some((tsid as u32, tokens))
                }),
        )
    }
}

#[derive(Clone)]
struct FastBoundaryNwaState {
    epsilons: Vec<(u32, FastBoundaryWeightId)>,
    transitions: Vec<(i32, SmallVec<[(u32, FastBoundaryWeightId); 1]>)>,
    final_weight: FastBoundaryWeightId,
}

#[path = "finite_template_program.rs"]
mod finite_template_program;
#[path = "finite_top_support.rs"]
mod finite_top_support;
#[path = "finite_weighted_top.rs"]
mod finite_weighted_top;
#[path = "finite_weight_support.rs"]
mod finite_weight_support;
#[path="finite_requotient.rs"]
mod finite_requotient;
#[cfg(feature = "internal-api")]
pub use finite_template_program::{FiniteTemplateInstance, FiniteTemplateProgram,
    FiniteTemplateProgramProfile, normalize_finite_template_program};



#[derive(Clone, Default)]
enum FastBoundaryQueryRow {
    #[default]
    Empty,
    One((u32, i32), FastBoundaryWeightId),
    Many(FxHashMap<(u32, i32), FastBoundaryWeightId>),
}

impl FastBoundaryQueryRow {
    fn merge(
        &mut self,
        key: (u32, i32),
        add: FastBoundaryWeightId,
        interner: &mut FastBoundaryWeightInterner,
    ) -> bool {
        if add == 0 {
            return false;
        }
        match self {
            Self::Empty => {
                *self = Self::One(key, add);
                true
            }
            Self::One(existing_key, existing) if *existing_key == key => {
                let merged = interner.union(*existing, add);
                if merged == *existing {
                    false
                } else {
                    *existing = merged;
                    true
                }
            }
            Self::One(existing_key, existing) => {
                let previous_key = *existing_key;
                let previous_weight = *existing;
                let mut entries = FxHashMap::with_capacity_and_hasher(4, Default::default());
                entries.insert(previous_key, previous_weight);
                entries.insert(key, add);
                *self = Self::Many(entries);
                true
            }
            Self::Many(entries) => {
                match entries.entry(key) {
                    Entry::Vacant(entry) => {
                        entry.insert(add);
                        true
                    }
                    Entry::Occupied(mut entry) => {
                        let merged = interner.union(*entry.get(), add);
                        if merged == *entry.get() {
                            false
                        } else {
                            entry.insert(merged);
                            true
                        }
                    }
                }
            }
        }
    }

    #[inline]
    fn get(&self, key: &(u32, i32)) -> FastBoundaryWeightId {
        match self {
            Self::Empty => 0,
            Self::One(existing_key, weight) => (*existing_key == *key).then_some(*weight).unwrap_or(0),
            Self::Many(entries) => entries.get(key).copied().unwrap_or(0),
        }
    }

    fn for_each(&self, mut f: impl FnMut((u32, i32), FastBoundaryWeightId)) {
        match self {
            Self::Empty => {}
            Self::One(key, weight) => f(*key, *weight),
            Self::Many(entries) => {
                for (&key, &weight) in entries {
                    f(key, weight);
                }
            }
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Empty => 0,
            Self::One(_, _) => 1,
            Self::Many(entries) => entries.len(),
        }
    }
}

#[derive(Clone, Default)]
enum FastBoundaryDerivedRow {
    #[default]
    Empty,
    One(u32, FastBoundaryWeightId),
    Many(FxHashMap<u32, FastBoundaryWeightId>),
}

impl FastBoundaryDerivedRow {
    fn merge(
        &mut self,
        target: u32,
        add: FastBoundaryWeightId,
        interner: &mut FastBoundaryWeightInterner,
    ) -> bool {
        if add == 0 {
            return false;
        }
        match self {
            Self::Empty => {
                *self = Self::One(target, add);
                true
            }
            Self::One(existing_target, existing) if *existing_target == target => {
                let merged = interner.union(*existing, add);
                if merged == *existing {
                    false
                } else {
                    *existing = merged;
                    true
                }
            }
            Self::One(existing_target, existing) => {
                let previous_target = *existing_target;
                let previous_weight = *existing;
                let mut entries = FxHashMap::with_capacity_and_hasher(4, Default::default());
                entries.insert(previous_target, previous_weight);
                entries.insert(target, add);
                *self = Self::Many(entries);
                true
            }
            Self::Many(entries) => match entries.entry(target) {
                Entry::Vacant(entry) => {
                    entry.insert(add);
                    true
                }
                Entry::Occupied(mut entry) => {
                    let merged = interner.union(*entry.get(), add);
                    if merged == *entry.get() {
                        false
                    } else {
                        entry.insert(merged);
                        true
                    }
                }
            },
        }
    }

    #[inline]
    fn get(&self, target: u32) -> FastBoundaryWeightId {
        match self {
            Self::Empty => 0,
            Self::One(existing_target, weight) => (*existing_target == target).then_some(*weight).unwrap_or(0),
            Self::Many(entries) => entries.get(&target).copied().unwrap_or(0),
        }
    }

    fn for_each(&self, mut f: impl FnMut(u32, FastBoundaryWeightId)) {
        match self {
            Self::Empty => {}
            Self::One(target, weight) => f(*target, *weight),
            Self::Many(entries) => {
                for (&target, &weight) in entries {
                    f(target, weight);
                }
            }
        }
    }

    fn into_entries(self) -> Vec<(u32, FastBoundaryWeightId)> {
        match self {
            Self::Empty => Vec::new(),
            Self::One(target, weight) => vec![(target, weight)],
            Self::Many(entries) => entries.into_iter().collect(),
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Empty => 0,
            Self::One(_, _) => 1,
            Self::Many(entries) => entries.len(),
        }
    }
}

fn fast_boundary_topological_order(states: &[FastBoundaryNwaState]) -> Option<Vec<u32>> {
    let n = states.len();
    let mut indegree = vec![0usize; n];
    for state in states {
        for (_, branches) in &state.transitions {
            for (target, weight) in branches {
                if *weight != 0 && (*target as usize) < n {
                    indegree[*target as usize] += 1;
                }
            }
        }
        for (target, weight) in &state.epsilons {
            if *weight != 0 && (*target as usize) < n {
                indegree[*target as usize] += 1;
            }
        }
    }
    let mut queue = VecDeque::new();
    for (state, degree) in indegree.iter().enumerate() {
        if *degree == 0 {
            queue.push_back(state as u32);
        }
    }
    let mut order = Vec::with_capacity(n);
    while let Some(source) = queue.pop_front() {
        order.push(source);
        let state = &states[source as usize];
        for (_, branches) in &state.transitions {
            for (target, weight) in branches {
                if *weight == 0 || (*target as usize) >= n {
                    continue;
                }
                indegree[*target as usize] -= 1;
                if indegree[*target as usize] == 0 {
                    queue.push_back(*target);
                }
            }
        }
        for (target, weight) in &state.epsilons {
            if *weight == 0 || (*target as usize) >= n {
                continue;
            }
            indegree[*target as usize] -= 1;
            if indegree[*target as usize] == 0 {
                queue.push_back(*target);
            }
        }
    }
    (order.len() == n).then_some(order)
}

/// A private permutation/rank certificate. It does not merge states or
/// change labels. Reusers check current nonzero edges against its ranks;
/// newly derived cancellation edges are checked before insertion.
struct CheckedNativeTopology {
    order: Vec<u32>,
    ranks: Vec<u32>,
}
impl CheckedNativeTopology {
    fn from_order(order: Vec<u32>) -> Option<Self> {
        let mut ranks = vec![u32::MAX; order.len()];
        for (rank, &q) in order.iter().enumerate() {
            let slot = ranks.get_mut(q as usize)?;
            if *slot != u32::MAX { return None; }
            *slot = u32::try_from(rank).ok()?;
        }
        Some(Self { order, ranks })
    }
    fn permits(&self, from: usize, to: u32) -> bool {
        match (self.ranks.get(from), self.ranks.get(to as usize)) {
            (Some(a), Some(b)) => a < b,
            _ => false,
        }
    }
    fn certifies(&self, states: &[FastBoundaryNwaState]) -> bool {
        self.order.len() == states.len() && states.iter().enumerate().all(|(q, row)| {
            row.epsilons.iter().chain(row.transitions.iter().flat_map(|(_, bs)| bs.iter()))
                .all(|&(target, weight)| (target as usize) < states.len()
                    && (weight == 0 || self.permits(q, target)))
        })
    }
}

fn fast_boundary_cancellations_worklist(
    states: &[FastBoundaryNwaState],
    interner: &mut FastBoundaryWeightInterner,
) -> Option<NativeCancellationResult> {
    let n = states.len();
    let mut query_weights = vec![FastBoundaryQueryRow::default(); n];
    let mut derived = vec![FastBoundaryDerivedRow::default(); n];
    let mut worklist = VecDeque::<(u32, u32, i32)>::new();
    let cancellation_started = Instant::now();
    let mut worklist_pops = 0usize;

    let queue_query = |query_weights: &mut [FastBoundaryQueryRow],
                       worklist: &mut VecDeque<(u32, u32, i32)>,
                       current: u32,
                       source: u32,
                       label: i32,
                       add: FastBoundaryWeightId,
                       interner: &mut FastBoundaryWeightInterner| {
        if (current as usize) < query_weights.len()
            && query_weights[current as usize].merge((source, label), add, interner)
        {
            worklist.push_back((current, source, label));
        }
    };

    for source in 0..n as u32 {
        for (label, branches) in &states[source as usize].transitions {
            if !is_negative_label(*label) {
                continue;
            }
            let positive = negative_to_positive_label(*label);
            for (target, weight) in branches {
                if *weight != 0 {
                    queue_query(
                        &mut query_weights,
                        &mut worklist,
                        *target,
                        source,
                        positive,
                        *weight,
                        interner,
                    );
                }
            }
        }
    }

    while let Some((current, source, positive_label)) = worklist.pop_front() {
        // The existing pop counter gives a bounded cancellation-work budget
        // without a full policy check on every inner query. At most 255 query
        // pops can occur after a limit is crossed; nothing is published until
        // the final exhaustion check. Mask-word caps still apply per insertion.
        if worklist_pops & 255 == 0 && !interner.allow_work(256, states.len(), 0) { return None; }
        worklist_pops += 1;
        if current as usize >= n || source as usize >= n {
            continue;
        }
        let query = query_weights[current as usize].get(&(source, positive_label));
        if query == 0 {
            continue;
        }

        let mut local_updates = Vec::with_capacity(derived[current as usize].len());
        derived[current as usize].for_each(|target, edge_weight| {
            local_updates.push((target, interner.intersection(query, edge_weight)));
        });
        for (target, propagated) in local_updates {
            queue_query(
                &mut query_weights,
                &mut worklist,
                target,
                source,
                positive_label,
                propagated,
                interner,
            );
        }

        for (label, branches) in &states[current as usize].transitions {
            if *label != positive_label && *label != DEFAULT_LABEL {
                continue;
            }
            for (target, edge_weight) in branches {
                if *target as usize >= n {
                    continue;
                }
                let add = interner.intersection(query, *edge_weight);
                if add == 0
                    || !derived[source as usize].merge(*target, add, interner)
                {
                    continue;
                }
                let derived_weight = derived[source as usize].get(*target);
                let mut upstream_updates = Vec::with_capacity(query_weights[source as usize].len());
                query_weights[source as usize].for_each(|(upstream_source, upstream_label), upstream_weight| {
                    let propagated = interner.intersection(upstream_weight, derived_weight);
                    if propagated != 0 {
                        upstream_updates.push((upstream_source, upstream_label, propagated));
                    }
                });
                for (upstream_source, upstream_label, propagated) in upstream_updates {
                    queue_query(
                        &mut query_weights,
                        &mut worklist,
                        *target,
                        upstream_source,
                        upstream_label,
                        propagated,
                        interner,
                    );
                }
            }
        }

        for (target, edge_weight) in &states[current as usize].epsilons {
            if *target as usize >= n {
                continue;
            }
            let propagated = interner.intersection(query, *edge_weight);
            queue_query(
                &mut query_weights,
                &mut worklist,
                *target,
                source,
                positive_label,
                propagated,
                interner,
            );
        }
    }

    let cancellation_ms = elapsed_ms(cancellation_started);
    let query_entries = query_weights.iter().map(FastBoundaryQueryRow::len).sum::<usize>();
    let max_query_entries = query_weights.iter().map(FastBoundaryQueryRow::len).max().unwrap_or(0);
    let derived_entries = derived.iter().map(FastBoundaryDerivedRow::len).sum::<usize>();
    let max_derived_entries = derived.iter().map(FastBoundaryDerivedRow::len).max().unwrap_or(0);
    Some(NativeCancellationResult { derived, worklist_pops, query_entries, max_query_entries,
        derived_entries, max_derived_entries, cancellation_ms })
}

#[path = "finite_cancellation.rs"]
mod finite_cancellation;

struct NativeCancellationResult {
    derived: Vec<FastBoundaryDerivedRow>,
    worklist_pops: usize,
    query_entries: usize,
    max_query_entries: usize,
    derived_entries: usize,
    max_derived_entries: usize,
    cancellation_ms: f64,
}

fn fast_boundary_resolve_negative_codes(
    states: &mut [FastBoundaryNwaState],
    interner: &mut FastBoundaryWeightInterner,
) -> Option<()> {
    fast_boundary_resolve_negative_codes_with_topology(states, interner, None)
}

fn fast_boundary_resolve_negative_codes_with_topology(
    states: &mut [FastBoundaryNwaState],
    interner: &mut FastBoundaryWeightInterner,
    topology: Option<&CheckedNativeTopology>,
) -> Option<()> {
    let n = states.len();
    if topology.is_some_and(|order| !order.certifies(states)) { return None; }
    let memo = crate::optimized_env_flag("GLRMASK_BOUNDARY_MEMO_CANCELLATIONS");
    let result = if memo {
        let started = Instant::now();
        let (derived, stats) = finite_cancellation::compute_with_topology(states, interner, topology)?;
        let cancellation_ms = elapsed_ms(started);
        if std::env::var_os("GLRMASK_VALIDATE_BOUNDARY_MEMO_CANCELLATIONS").is_some() {
            let reference=fast_boundary_cancellations_worklist(states,interner)
                .expect("native cancellation reference exhausted its budget; equality not established");
            for (source,(left,right)) in reference.derived.iter().zip(&derived).enumerate(){
                let mut l=Vec::new();let mut r=Vec::new();
                left.for_each(|target,w|l.push((target,w)));
                right.for_each(|target,w|r.push((target,w)));
                l.sort_unstable();r.sort_unstable();
                assert_eq!(l,r,"memo cancellation changed derived epsilon weights at source {source}");
            }
            eprintln!("[glrmask/validate][native_cancellation_summary] exact_derived_rows=true states={n}");
        }
        if compile_profile_enabled() {
            eprintln!("[glrmask/profile][native_cancellation_summary] states={n} ms={cancellation_ms:.3} stats={stats:?}");
        }
        NativeCancellationResult {
            derived_entries: derived.iter().map(FastBoundaryDerivedRow::len).sum(),
            max_derived_entries: derived.iter().map(FastBoundaryDerivedRow::len).max().unwrap_or(0),
            derived, worklist_pops: 0, query_entries: 0, max_query_entries: 0, cancellation_ms,
        }
    } else { fast_boundary_cancellations_worklist(states, interner)? };
    let NativeCancellationResult { derived, worklist_pops, query_entries, max_query_entries,
        derived_entries, max_derived_entries, cancellation_ms } = result;
    for (source, row) in derived.into_iter().enumerate() {
        for (target, weight) in row.into_entries() {
            if weight != 0 {
                // Cancellation connects endpoints of an original forward path.
                // Check that certificate consequence rather than re-sorting.
                if topology.is_some_and(|order| !order.permits(source, target)) { return None; }
                states[source].epsilons.push((target, weight));
            }
        }
    }

    let finality_started = Instant::now();
    let owned_topo;
    let topo = if let Some(topology) = topology { topology.order.as_slice() } else {
        owned_topo = fast_boundary_topological_order(states)?;
        owned_topo.as_slice()
    };
    let mut finals = states.iter().map(|state| state.final_weight).collect::<Vec<_>>();
    for &source in topo.iter().rev() {
        let mut final_weight = finals[source as usize];
        for (target, edge_weight) in &states[source as usize].epsilons {
            if (*target as usize) < n {
                let contribution = interner.intersection(*edge_weight, finals[*target as usize]);
                final_weight = interner.union(final_weight, contribution);
            }
        }
        for (label, branches) in &states[source as usize].transitions {
            if *label != DEFAULT_LABEL && !is_negative_label(*label) {
                continue;
            }
            for (target, edge_weight) in branches {
                if (*target as usize) < n {
                    let contribution =
                        interner.intersection(*edge_weight, finals[*target as usize]);
                    final_weight = interner.union(final_weight, contribution);
                }
            }
        }
        finals[source as usize] = final_weight;
    }
    for (state, final_weight) in states.iter_mut().zip(finals) {
        state.final_weight = final_weight;
        state.transitions.retain(|(label, _)| !is_negative_label(*label));
    }

    let finality_ms = elapsed_ms(finality_started);
    let prune_started = Instant::now();
    let mut terminal = states
        .iter()
        .map(|state| {
            state.final_weight != 0
                && state.epsilons.is_empty()
                && !state
                    .transitions
                    .iter()
                    .any(|(label, branches)| *label != DEFAULT_LABEL && !branches.is_empty())
                && !state
                    .transitions
                    .iter()
                    .any(|(label, branches)| *label == DEFAULT_LABEL && !branches.is_empty())
        })
        .collect::<Vec<_>>();
    let mut dependents = vec![Vec::<usize>::new(); n];
    let mut remaining = vec![usize::MAX; n];
    let mut terminal_queue = VecDeque::new();
    for (state, &is_terminal) in terminal.iter().enumerate() {
        if is_terminal {
            terminal_queue.push_back(state);
        }
    }
    for state_id in 0..n {
        if terminal[state_id] {
            continue;
        }
        let state = &states[state_id];
        let candidate = state.final_weight != 0
            && state.epsilons.is_empty()
            && !state
                .transitions
                .iter()
                .any(|(label, branches)| *label != DEFAULT_LABEL && !branches.is_empty());
        if !candidate {
            continue;
        }
        let default = state
            .transitions
            .iter()
            .find_map(|(label, branches)| (*label == DEFAULT_LABEL).then_some(branches));
        let Some(default) = default else {
            terminal[state_id] = true;
            terminal_queue.push_back(state_id);
            continue;
        };
        if default
            .iter()
            .any(|(_, weight)| !interner.is_subset(*weight, state.final_weight))
        {
            continue;
        }
        let mut count = 0usize;
        for (target, _) in default {
            let target = *target as usize;
            if target >= n {
                count += 1;
            } else if !terminal[target] {
                dependents[target].push(state_id);
                count += 1;
            }
        }
        remaining[state_id] = count;
        if count == 0 {
            terminal[state_id] = true;
            terminal_queue.push_back(state_id);
        }
    }
    while let Some(done) = terminal_queue.pop_front() {
        for dependent in dependents[done].clone() {
            if terminal[dependent] {
                continue;
            }
            if remaining[dependent] == usize::MAX || remaining[dependent] == 0 {
                continue;
            }
            remaining[dependent] -= 1;
            if remaining[dependent] == 0 {
                terminal[dependent] = true;
                terminal_queue.push_back(dependent);
            }
        }
    }
    for state in states.iter_mut() {
        let final_weight = state.final_weight;
        for (label, branches) in &mut state.transitions {
            if *label == DEFAULT_LABEL && final_weight != 0 {
                branches.retain(|(target, edge_weight)| {
                    (*target as usize) >= n
                        || !terminal[*target as usize]
                        || !interner.is_subset(*edge_weight, final_weight)
                });
            }
        }
        state.transitions.retain(|(_, branches)| !branches.is_empty());
    }
    let prune_ms = elapsed_ms(prune_started);
    if compile_profile_enabled() {
        eprintln!(
            "[glrmask/profile][fast_boundary_resolve_detail] memo={memo} states={} worklist_pops={} query_entries={} max_query_entries={} derived_entries={} max_derived_entries={} weights={} intersection_pairs={} union_pairs={} cancellation_ms={cancellation_ms:.3} finality_ms={finality_ms:.3} prune_ms={prune_ms:.3} total_ms={:.3}",
            states.len(),
            worklist_pops,
            query_entries,
            max_query_entries,
            derived_entries,
            max_derived_entries,
            interner.values.len(),
            interner.intersections.len(),
            interner.unions.len(),
            cancellation_ms + finality_ms + prune_ms,
        );
    }
    Some(())
}

fn resolve_negative_codes_small_boundary_impl(
    nwa: &NWA,
    tsid_count: usize,
    token_count: usize,
) -> Option<NWA> {
    let total_started = Instant::now();
    let mut interner = FastBoundaryWeightInterner::new(tsid_count, token_count)?;
    let mut source_weight_ids = FxHashMap::<usize, FastBoundaryWeightId>::default();
    let convert_started = Instant::now();
    let mut states = Vec::with_capacity(nwa.states().len());
    for state in nwa.states() {
        let final_weight = match state.final_weight.as_ref() {
            Some(weight) => interner.source_weight_id(weight, &mut source_weight_ids, None)?,
            None => 0,
        };
        let mut epsilons = Vec::with_capacity(state.epsilons.len());
        for (target, weight) in &state.epsilons {
            let weight = interner.source_weight_id(weight, &mut source_weight_ids, None)?;
            if weight != 0 {
                epsilons.push((*target, weight));
            }
        }
        let mut transitions = Vec::with_capacity(state.transitions.len());
        for (&label, branches) in &state.transitions {
            let mut out = SmallVec::with_capacity(branches.len());
            for (target, weight) in branches {
                let weight = interner.source_weight_id(weight, &mut source_weight_ids, None)?;
                if weight != 0 {
                    out.push((*target, weight));
                }
            }
            if !out.is_empty() {
                transitions.push((label, out));
            }
        }
        states.push(FastBoundaryNwaState {
            epsilons,
            transitions,
            final_weight,
        });
    }
    let convert_ms = elapsed_ms(convert_started);
    let resolve_started = Instant::now();
    fast_boundary_resolve_negative_codes(&mut states, &mut interner)?;
    let resolve_ms = elapsed_ms(resolve_started);
    let materialize_started = Instant::now();
    let mut weight_cache = vec![None::<Weight>; interner.values.len()];
    if !weight_cache.is_empty() {
        weight_cache[0] = Some(Weight::empty());
    }
    if weight_cache.len() > 1 {
        weight_cache[1] = Some(Weight::all());
    }
    let mut materialized_weight = |id: FastBoundaryWeightId| {
        if let Some(weight) = weight_cache[id as usize].as_ref() {
            return weight.clone();
        }
        let weight = interner.to_weight(id);
        weight_cache[id as usize] = Some(weight.clone());
        weight
    };
    let mut output = NWA::new(0, 0);
    for _ in 0..states.len() {
        output.add_state();
    }
    output.set_start_states(nwa.start_states().to_vec());
    for (state_id, state) in states.into_iter().enumerate() {
        if state.final_weight != 0 {
            output.set_final_weight(state_id as u32, materialized_weight(state.final_weight));
        }
        for (target, weight) in state.epsilons {
            output.add_epsilon(state_id as u32, target, materialized_weight(weight));
        }
        for (label, branches) in state.transitions {
            for (target, weight) in branches {
                output.add_transition(
                    state_id as u32,
                    label,
                    target,
                    materialized_weight(weight),
                );
            }
        }
    }
    let materialize_ms = elapsed_ms(materialize_started);
    if compile_profile_enabled() {
        eprintln!(
            "[glrmask/profile][small_boundary_signed_resolution] states={} source_weights={} fast_weights={} convert_ms={convert_ms:.3} resolve_ms={resolve_ms:.3} materialize_ms={materialize_ms:.3} total_ms={:.3}",
            nwa.states().len(),
            source_weight_ids.len(),
            interner.values.len(),
            elapsed_ms(total_started),
        );
    }
    Some(output)
}

pub fn resolve_negative_codes_small_boundary(
    nwa: &NWA,
    tsid_count: usize,
    token_count: usize,
) -> Option<NWA> {
    resolve_negative_codes_small_boundary_impl(nwa, tsid_count, token_count)
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct SmallBoundaryDwaState {
    pub transitions: Vec<(i32, u32, u32)>,
    pub final_weight: u32,
}

type FastBoundaryDwaState = SmallBoundaryDwaState;

/// Exact deterministic parser DWA specialized for a small private boundary
/// coordinate.  `weights[id][tsid]` is the bit mask of private token classes
/// admitted by that weight at the given private TSID.  This changes only the
/// weight representation: state topology, DEFAULT semantics, and deterministic
/// parser-state transitions are the same as the ordinary weighted DWA.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SmallBoundaryDwa {
    pub states: Vec<SmallBoundaryDwaState>,
    pub weights: Vec<[u64; 16]>,
    pub tsid_count: u8,
    pub token_count: u8,
}

impl SmallBoundaryDwa {
    #[inline]
    pub fn start_state(&self) -> u32 { 0 }

    #[inline]
    pub fn num_states(&self) -> u32 { self.states.len() as u32 }

    pub fn num_transitions(&self) -> usize {
        self.states.iter().map(|state| state.transitions.len()).sum()
    }

    #[inline]
    pub fn weight_mask(&self, weight_id: u32, tsid: u32) -> u64 {
        if tsid >= self.tsid_count as u32 {
            return 0;
        }
        self.weights
            .get(weight_id as usize)
            .map_or(0, |weight| weight[tsid as usize])
    }

    #[inline]
    pub fn all_token_mask(&self) -> u64 {
        if self.token_count == 64 { u64::MAX } else { (1u64 << self.token_count) - 1 }
    }

    /// Reference/materialization path used by artifact serialization and exact
    /// differential validation. Runtime compact evaluation does not call this.
    pub fn to_generic_dwa(&self) -> DWA {
        let to_weight = |id: u32| {
            if id == 0 {
                return Weight::empty();
            }
            if id == 1 {
                return Weight::all();
            }
            let Some(value) = self.weights.get(id as usize) else {
                return Weight::empty();
            };
            Weight::from_per_tsid_token_sets(
                value[..self.tsid_count as usize]
                    .iter()
                    .enumerate()
                    .filter_map(|(tsid, &mask)| {
                        if mask == 0 {
                            return None;
                        }
                        let tokens = range_set_blaze::RangeSetBlaze::from_iter(
                            (0..self.token_count as u32)
                                .filter(|&token| mask & (1u64 << token) != 0),
                        );
                        Some((tsid as u32, tokens))
                    }),
            )
        };
        let states = self.states.iter().map(|state| {
            let transitions = state.transitions.iter().filter(|(_, _, weight)| *weight != 0)
                .map(|&(label, target, weight)| (label, (target, to_weight(weight))))
                .collect();
            let final_weight = (state.final_weight != 0).then(|| to_weight(state.final_weight));
            DWAState { transitions, final_weight }
        }).collect();
        DWA::from_parts(states, 0)
    }
}

fn fast_boundary_epsilon_closure(
    nwa: &[FastBoundaryNwaState],
    interner: &mut FastBoundaryWeightInterner,
    seed: &[(u32, FastBoundaryWeightId)],
    weight_by_state: &mut [FastBoundaryWeightId],
    queue: &mut VecDeque<u32>,
    touched: &mut Vec<u32>,
) -> Vec<(u32, FastBoundaryWeightId)> {
    touched.clear();
    queue.clear();
    for &(state, weight) in seed {
        if weight == 0 {
            continue;
        }
        let slot = &mut weight_by_state[state as usize];
        if *slot == 0 {
            *slot = weight;
            touched.push(state);
            queue.push_back(state);
        } else {
            let merged = interner.union(*slot, weight);
            if merged != *slot {
                *slot = merged;
                queue.push_back(state);
            }
        }
    }
    while let Some(state) = queue.pop_front() {
        let current = weight_by_state[state as usize];
        for &(target, edge_weight) in &nwa[state as usize].epsilons {
            let contribution = interner.intersection(current, edge_weight);
            if contribution == 0 {
                continue;
            }
            let slot = &mut weight_by_state[target as usize];
            if *slot == 0 {
                *slot = contribution;
                touched.push(target);
                queue.push_back(target);
            } else if !interner.is_subset(contribution, *slot) {
                let merged = interner.union(*slot, contribution);
                if merged != *slot {
                    *slot = merged;
                    queue.push_back(target);
                }
            }
        }
    }
    touched.sort_unstable();
    let mut result = Vec::with_capacity(touched.len());
    for &state in touched.iter() {
        let weight = std::mem::replace(&mut weight_by_state[state as usize], 0);
        if weight != 0 {
            result.push((state, weight));
        }
    }
    result
}

fn fast_boundary_singleton_state(
    nwa_state: u32,
    singleton_states: &mut [u32],
    subset_map: &mut FxHashMap<Vec<(u32, FastBoundaryWeightId)>, u32>,
    out_states: &mut Vec<FastBoundaryDwaState>,
    supports: &mut Vec<Vec<u32>>,
    worklist: &mut VecDeque<(u32, Vec<(u32, FastBoundaryWeightId)>)>,
    all_weight: FastBoundaryWeightId,
) -> u32 {
    let existing = singleton_states[nwa_state as usize];
    if existing != u32::MAX {
        return existing;
    }
    let state = out_states.len() as u32;
    let subset = vec![(nwa_state, all_weight)];
    subset_map.insert(subset.clone(), state);
    singleton_states[nwa_state as usize] = state;
    out_states.push(FastBoundaryDwaState::default());
    supports.push(vec![nwa_state]);
    worklist.push_back((state, subset));
    state
}

/// Canonicalize target contributions without allocating a second frontier.
#[inline]
fn compact_fast_boundary_contribs(
    contribs: &mut FastBoundaryContribs,
    interner: &mut FastBoundaryWeightInterner,
) {
    if contribs.len() < 2 { return; }
    contribs.sort_unstable_by_key(|(state, _)| *state);
    let mut written = 0usize;
    for read in 0..contribs.len() {
        let (state, weight) = contribs[read];
        if written != 0 && contribs[written - 1].0 == state {
            let previous = contribs[written - 1].1;
            contribs[written - 1].1 = interner.union(previous, weight);
        } else {
            contribs[written] = (state, weight);
            written += 1;
        }
    }
    contribs.truncate(written);
}

/// Keep common per-label frontier storage, but do not pin exceptional buffers
/// across the entire determinization. This changes storage, not frontier keys.
#[inline]
fn recycle_fast_boundary_contribs(contribs: &mut FastBoundaryContribs) {
    if contribs.capacity() > 128 {
        *contribs = FastBoundaryContribs::new();
    } else {
        contribs.clear();
    }
}

#[cfg(test)]
#[test]
fn recycled_boundary_contribs_preserve_small_buffers_and_release_large_ones() {
    let mut buffer = FastBoundaryContribs::new();
    for i in 0..64 { buffer.push((i, 1)); }
    let pointer = buffer.as_ptr();
    let capacity = buffer.capacity();
    let snapshot = buffer.to_vec();
    recycle_fast_boundary_contribs(&mut buffer);
    assert!(buffer.is_empty());
    assert_eq!(buffer.as_ptr(), pointer);
    assert_eq!(buffer.capacity(), capacity);
    buffer.push((999, 2));
    assert_eq!(snapshot[0], (0, 1));
    assert_eq!(snapshot.len(), 64);
    for i in 0..512 { buffer.push((i, 3)); }
    assert!(buffer.capacity() > 128);
    recycle_fast_boundary_contribs(&mut buffer);
    assert!(buffer.is_empty());
    assert!(buffer.capacity() <= 128);
}

#[cfg(test)]
#[test]
fn inplace_boundary_contribs_match_independent_grouped_unions_without_reallocation() {
    let mut pool = FastBoundaryWeightInterner::new(1, 8).unwrap();
    let weights = (0..256u64).map(|v| pool.intern(smallvec::smallvec![v])).collect::<Vec<_>>();
    let mut seed = 73455u64;
    let mut next = || { seed=seed.wrapping_mul(6364136223846793005).wrapping_add(1); (seed>>32)as usize };
    for len in 0..128 {
        for _ in 0..16 {
            let mut input = FastBoundaryContribs::new();
            let mut expected = BTreeMap::<u32,u64>::new();
            for _ in 0..len {
                let target = (next()%24)as u32;
                let bits = next()%256;
                input.push((target,weights[bits]));
                *expected.entry(target).or_default() |= bits as u64;
            }
            let pointer = input.as_ptr();
            let capacity = input.capacity();
            compact_fast_boundary_contribs(&mut input, &mut pool);
            assert_eq!(input.as_ptr(),pointer);
            assert_eq!(input.capacity(),capacity);
            let actual=input.iter().map(|&(q,w)|(q,pool.values[w as usize][0])).collect::<Vec<_>>();
            assert_eq!(actual,expected.into_iter().collect::<Vec<_>>());
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn fast_boundary_process_contribs(
    contribs: &mut FastBoundaryContribs,
    nwa: &[FastBoundaryNwaState],
    interner: &mut FastBoundaryWeightInterner,
    weight_by_state: &mut [FastBoundaryWeightId],
    closure_queue: &mut VecDeque<u32>,
    closure_touched: &mut Vec<u32>,
    singleton_states: &mut [u32],
    subset_map: &mut FxHashMap<Vec<(u32, FastBoundaryWeightId)>, u32>,
    singleton_closure_cache: &mut FxHashMap<
        (u32, FastBoundaryWeightId),
        (u32, FastBoundaryWeightId),
    >,
    closure_cache: &mut FxHashMap<
        Vec<(u32, FastBoundaryWeightId)>,
        (u32, FastBoundaryWeightId),
    >,
    out_states: &mut Vec<FastBoundaryDwaState>,
    supports: &mut Vec<Vec<u32>>,
    worklist: &mut VecDeque<(u32, Vec<(u32, FastBoundaryWeightId)>)>,
) -> Option<(u32, FastBoundaryWeightId)> {
    if contribs.is_empty() {
        return None;
    }
    compact_fast_boundary_contribs(contribs, interner);

    if let [(state, weight)] = contribs.as_slice()
        && nwa[*state as usize].epsilons.is_empty()
    {
        let to_state = fast_boundary_singleton_state(
            *state,
            singleton_states,
            subset_map,
            out_states,
            supports,
            worklist,
            interner.all_id(),
        );
        return Some((to_state, *weight));
    }

    let singleton_key = match contribs.as_slice() {
        [(state, weight)] => Some((*state, *weight)),
        _ => None,
    };
    if let Some(key) = singleton_key {
        if let Some(&cached) = singleton_closure_cache.get(&key) {
            return Some(cached);
        }
    } else if let Some(&cached) = closure_cache.get(contribs.as_slice()) {
        return Some(cached);
    }

    let mut incoming_edge_weight = interner.empty_id();
    for (_, weight) in contribs.iter() {
        incoming_edge_weight = interner.union(incoming_edge_weight, *weight);
    }
    if incoming_edge_weight == 0 {
        return None;
    }
    let closure = fast_boundary_epsilon_closure(
        nwa,
        interner,
        contribs.as_slice(),
        weight_by_state,
        closure_queue,
        closure_touched,
    );
    if closure.is_empty() {
        return None;
    }

    let (to_state, edge_weight) = if let [(state, residual)] = closure.as_slice() {
        let to_state = fast_boundary_singleton_state(
            *state,
            singleton_states,
            subset_map,
            out_states,
            supports,
            worklist,
            interner.all_id(),
        );
        (to_state, *residual)
    } else if let Some(&existing) = subset_map.get(closure.as_slice()) {
        (existing, incoming_edge_weight)
    } else {
        let state = out_states.len() as u32;
        subset_map.insert(closure.clone(), state);
        supports.push(closure.iter().map(|(state, _)| *state).collect());
        out_states.push(FastBoundaryDwaState::default());
        worklist.push_back((state, closure));
        (state, incoming_edge_weight)
    };
    let cached = (to_state, edge_weight);
    if let Some(key) = singleton_key {
        singleton_closure_cache.insert(key, cached);
    } else {
        // Cache keys must outlive the scratch buffer. Retain common capacities
        // for future labels/subsets; move unusually large allocations instead.
        let key = if contribs.capacity() > 128 {
            std::mem::take(contribs).into_vec()
        } else {
            contribs.to_vec()
        };
        closure_cache.insert(key, cached);
    }
    Some(cached)
}

/// Exact support determinizer specialized for a boundary weight coordinate with
/// at most 16 TSID classes and 64 token classes. The graph/subset algorithm is
/// the ordinary weighted determinization; only the temporary semiring
/// representation changes from RangeMap/RangeSet weights to interned fixed-size
/// bit arrays.


#[derive(Clone)]
enum FastPossibleOutgoingIds {
    Empty,
    All,
    Small(SmallVec<[u32; 16]>),
    Bits(BitSet),
    Shared(std::sync::Arc<[u32]>),
}

impl FastPossibleOutgoingIds {
    #[inline]
    fn count(&self, num_parser_states: u32) -> usize {
        match self {
            Self::Empty => 0,
            Self::All => num_parser_states as usize,
            Self::Small(ids) => ids.len(),
            Self::Bits(ids) => ids.count_ones(),
            Self::Shared(ids) => ids.len(),
        }
    }

    #[inline]
    fn for_each(&self, num_parser_states: u32, mut f: impl FnMut(u32)) {
        match self {
            Self::Empty => {}
            Self::All => {
                for parser_state in 0..num_parser_states {
                    f(parser_state);
                }
            }
            Self::Small(ids) => {
                for &parser_state in ids {
                    f(parser_state);
                }
            }
            Self::Shared(ids) => { for &parser_state in ids.iter() { f(parser_state); } }
            Self::Bits(ids) => {
                for parser_state in ids.iter_ones() {
                    f(parser_state as u32);
                }
            }
        }
    }
}

fn fast_boundary_possible_outgoing_ids(
    nwa:&[FastBoundaryNwaState], supports:&[Vec<u32>], num_parser_states:u32,
)->Vec<FastPossibleOutgoingIds>{
    if crate::optimized_env_flag("GLRMASK_BOUNDARY_POSSIBLE_SET_CACHE"){
        let result=fast_boundary_possible_outgoing_ids_cached(nwa,supports,num_parser_states);
        if std::env::var_os("GLRMASK_VALIDATE_BOUNDARY_GUARD_SET_CACHE").is_some(){
            validate_guard_observer(nwa,supports,num_parser_states,&result);
        }
        result
    }else{fast_boundary_possible_outgoing_ids_reference(nwa,supports,num_parser_states)}
}

/// Equality is only over row observations, not NWA states. In particular the
/// topology and zero-weight guards observed by the normalizer remain intact.
fn fast_boundary_possible_outgoing_ids_cached(
    nwa:&[FastBoundaryNwaState], supports:&[Vec<u32>], num_parser_states:u32,
)->Vec<FastPossibleOutgoingIds>{
    fast_boundary_possible_outgoing_ids_cached_with_budget(nwa,supports,num_parser_states,65_536,1_048_576)
}

fn fast_boundary_possible_outgoing_ids_cached_with_budget(
    nwa:&[FastBoundaryNwaState],supports:&[Vec<u32>],num_parser_states:u32,
    cache_cap:usize,member_budget:usize,
)->Vec<FastPossibleOutgoingIds>{
    let started=Instant::now();
    let mut classes=vec![Vec::<u32>::new(),Vec::<u32>::new()]; // empty, ALL
    let mut index=FxHashMap::<Vec<u32>,u32>::default();
    let mut state_class=Vec::with_capacity(nwa.len());
    for state in nwa {
        let mut ids=Vec::new();let mut all=false;
        for &(label,_) in &state.transitions {
            if label==DEFAULT_LABEL{all=true;break;}
            if let Some(q)=parser_state_label(label,num_parser_states){ids.push(q);}
        }
        if all{state_class.push(1);continue;}
        if ids.is_empty(){state_class.push(0);continue;}
        ids.sort_unstable();ids.dedup();
        let id=if let Some(&id)=index.get(&ids){id}else{
            let id=classes.len() as u32;classes.push(ids.clone());index.insert(ids,id);id
        };
        state_class.push(id);
    }
    let mut cache=FxHashMap::<(bool,SmallVec<[u32;8]>),FastPossibleOutgoingIds>::default();
    let mut hits=0usize;
    let mut retained_members=0usize;
    let output=supports.iter().map(|support|{
        let mut key=SmallVec::<[u32;8]>::new();
        for &state in support {
            let c=state_class.get(state as usize).copied().unwrap_or(0);
            if c==1{return FastPossibleOutgoingIds::All;}
            if c!=0{key.push(c);}
        }
        if key.is_empty(){return FastPossibleOutgoingIds::Empty;}
        key.sort_unstable();key.dedup();
        // Preserve the reference's singleton explicit-full versus multi-row
        // wildcard convention. They have different fallback dispatch paths.
        let cache_key=(support.len()==1,key);
        if let Some(value)=cache.get(&cache_key){hits+=1;return value.clone();}
        let mut ids=Vec::new();
        for &c in &cache_key.1{ids.extend_from_slice(&classes[c as usize]);}
        if cache_key.1.len()>1{ids.sort_unstable();ids.dedup();}
        let value=if !cache_key.0&&ids.len()==num_parser_states as usize{
            FastPossibleOutgoingIds::All
        }else if ids.len()<=16{
            FastPossibleOutgoingIds::Small(ids.into_iter().collect())
        }else{FastPossibleOutgoingIds::Shared(std::sync::Arc::from(ids))};
        let members=cache_key.1.len().saturating_add(value.count(num_parser_states));
        if cache.len()<cache_cap && members<=member_budget.saturating_sub(retained_members){
            retained_members+=members;cache.insert(cache_key,value.clone());
        }
        value
    }).collect::<Vec<_>>();
    if compile_profile_enabled(){eprintln!("[glrmask/profile][guard_observer_cache] raw_states={} classes={} supports={} entries={} hits={} elapsed_ms={:.3}",nwa.len(),classes.len(),supports.len(),cache.len(),hits,elapsed_ms(started));}
    output
}

/// Independent diagnostic observer using raw label-key set union. Branch
/// vector contents are intentionally irrelevant, including zero/empty guards.
fn validate_guard_observer(
    nwa:&[FastBoundaryNwaState],supports:&[Vec<u32>],alphabet:u32,
    actual:&[FastPossibleOutgoingIds],
){
    assert_eq!(supports.len(),actual.len());
    for (row,(support,actual)) in supports.iter().zip(actual).enumerate(){
        let mut keys=BTreeSet::new();let mut wildcard=false;
        for &q in support { if let Some(state)=nwa.get(q as usize){
            for &(label,_) in &state.transitions {
                if label==DEFAULT_LABEL{wildcard=true;}
                else if label>=0 && (label as u32)<alphabet{keys.insert(label as u32);}
            }
        }}
        let full=wildcard || (support.len()!=1 && !keys.is_empty() && keys.len()==alphabet as usize);
        let expected=if full{(0..alphabet).collect::<Vec<_>>()}else{keys.into_iter().collect()};
        let mut found=Vec::new();actual.for_each(alphabet,|q|found.push(q));
        assert_eq!(found,expected,"raw guard set differs at support {row}");
        assert_eq!(actual.count(alphabet),expected.len(),"guard count differs");
        assert_eq!(matches!(actual,FastPossibleOutgoingIds::All),full,"wildcard convention differs");
    }
}

fn fast_boundary_possible_outgoing_ids_reference(
    nwa: &[FastBoundaryNwaState],
    supports: &[Vec<u32>],
    num_parser_states: u32,
) -> Vec<FastPossibleOutgoingIds> {
    enum Row {
        Empty,
        All,
        Some(SmallVec<[u32; 8]>),
    }
    let mut rows = Vec::with_capacity(nwa.len());
    for state in nwa {
        let mut ids = SmallVec::<[u32; 8]>::new();
        let mut all = false;
        for (label, _) in &state.transitions {
            if *label == DEFAULT_LABEL {
                all = true;
                break;
            }
            if let Some(parser_state) = parser_state_label(*label, num_parser_states) {
                ids.push(parser_state);
            }
        }
        if all {
            rows.push(Row::All);
        } else if ids.is_empty() {
            rows.push(Row::Empty);
        } else {
            ids.sort_unstable();
            ids.dedup();
            rows.push(Row::Some(ids));
        }
    }

    const SMALL_LIMIT: usize = 64;
    supports
        .iter()
        .map(|support| {
            if support.len() == 1 {
                return match rows.get(support[0] as usize) {
                    Some(Row::All) => FastPossibleOutgoingIds::All,
                    Some(Row::Some(ids)) => {
                        FastPossibleOutgoingIds::Small(ids.iter().copied().collect())
                    }
                    Some(Row::Empty) | None => FastPossibleOutgoingIds::Empty,
                };
            }

            let mut small = SmallVec::<[u32; 16]>::new();
            let mut bits = None::<BitSet>;
            for &state in support {
                match rows.get(state as usize) {
                    Some(Row::All) => return FastPossibleOutgoingIds::All,
                    Some(Row::Some(row)) => {
                        if let Some(bitset) = bits.as_mut() {
                            for &parser_state in row {
                                bitset.set(parser_state as usize);
                            }
                        } else {
                            small.extend(row.iter().copied());
                            if small.len() > SMALL_LIMIT {
                                small.sort_unstable();
                                small.dedup();
                                if small.len() > SMALL_LIMIT {
                                    let mut bitset = BitSet::new(num_parser_states as usize);
                                    for parser_state in small.drain(..) {
                                        bitset.set(parser_state as usize);
                                    }
                                    bits = Some(bitset);
                                }
                            }
                        }
                    }
                    Some(Row::Empty) | None => {}
                }
            }
            if let Some(bitset) = bits {
                let count = bitset.count_ones();
                if count == 0 {
                    FastPossibleOutgoingIds::Empty
                } else if count == num_parser_states as usize {
                    FastPossibleOutgoingIds::All
                } else {
                    FastPossibleOutgoingIds::Bits(bitset)
                }
            } else {
                small.sort_unstable();
                small.dedup();
                if small.is_empty() {
                    FastPossibleOutgoingIds::Empty
                } else if small.len() == num_parser_states as usize {
                    FastPossibleOutgoingIds::All
                } else {
                    FastPossibleOutgoingIds::Small(small)
                }
            }
        })
        .collect()
}

fn fast_boundary_default_step(
    state: &FastBoundaryDwaState,
    parser_state: u32,
    shared_target: &mut Option<u32>,
    default_weight: &mut FastBoundaryWeightId,
    interner: &mut FastBoundaryWeightInterner,
) -> bool {
    let Some((_, target, weight)) = state
        .transitions
        .iter()
        .find(|(label, _, _)| *label == parser_state as i32)
        .copied()
    else {
        return false;
    };
    if shared_target.is_some_and(|existing| existing != target) {
        return false;
    }
    *shared_target = Some(target);
    *default_weight = interner.intersection(*default_weight, weight);
    *default_weight != 0
}

fn optimize_fast_boundary_defaults(
    states: &mut [FastBoundaryDwaState],
    possible_by_state: &[FastPossibleOutgoingIds],
    num_parser_states: u32,
    interner: &mut FastBoundaryWeightInterner,
) {
    loop {
        let mut changed = false;
        for (state_id, possible) in possible_by_state.iter().enumerate() {
            let possible_count = possible.count(num_parser_states);
            if possible_count < 2 {
                continue;
            }
            let actual_positive = states[state_id]
                .transitions
                .iter()
                .filter(|(label, _, _)| parser_state_label(*label, num_parser_states).is_some())
                .count();
            if actual_positive != possible_count {
                continue;
            }
            let mut shared_target = None::<u32>;
            let mut default_weight = interner.all_id();
            let mut valid = !matches!(possible, FastPossibleOutgoingIds::Empty);
            if valid {
                possible.for_each(num_parser_states, |parser_state| {
                    if valid
                        && !fast_boundary_default_step(
                            &states[state_id],
                            parser_state,
                            &mut shared_target,
                            &mut default_weight,
                            interner,
                        )
                    {
                        valid = false;
                    }
                });
            }
            let Some(target) = shared_target else { continue; };
            if !valid || default_weight == 0 {
                continue;
            }
            if let Some((_, existing_target, existing_weight)) = states[state_id]
                .transitions.iter_mut().find(|(label, _, _)| *label == DEFAULT_LABEL)
            {
                if *existing_target == target {
                    let updated = interner.union(*existing_weight, default_weight);
                    changed |= updated != *existing_weight;
                    *existing_weight = updated;
                }
            } else {
                states[state_id].transitions.push((DEFAULT_LABEL, target, default_weight));
                changed = true;
            }
        }

        for state_id in 0..states.len() {
            let Some((_, default_target, default_weight)) = states[state_id]
                .transitions.iter().find(|(label, _, _)| *label == DEFAULT_LABEL).copied()
            else { continue; };
            let target_final = states.get(default_target as usize).map_or(0, |state| state.final_weight);
            let lifted = interner.intersection(default_weight, target_final);
            if lifted == 0 { continue; }
            let updated_final = interner.union(states[state_id].final_weight, lifted);
            changed |= updated_final != states[state_id].final_weight;
            states[state_id].final_weight = updated_final;
            for (_, _, weight) in &mut states[state_id].transitions {
                let updated = interner.difference(*weight, lifted);
                changed |= updated != *weight;
                *weight = updated;
            }
            states[state_id].transitions.retain(|(_, _, weight)| *weight != 0);
        }

        for state_id in 0..states.len() {
            let Some((_, default_target, default_weight)) = states[state_id]
                .transitions.iter().find(|(label, _, _)| *label == DEFAULT_LABEL).copied()
            else { continue; };
            for (label, target, weight) in &mut states[state_id].transitions {
                if *label == DEFAULT_LABEL || *target != default_target { continue; }
                let updated = interner.difference(*weight, default_weight);
                changed |= updated != *weight;
                *weight = updated;
            }
            states[state_id].transitions.retain(|(_, _, weight)| *weight != 0);
        }
        if !changed { break; }
    }
}


fn fast_boundary_add_target(
    contribs: &mut FastBoundaryContribs,
    target: u32,
    weight: FastBoundaryWeightId,
    interner: &mut FastBoundaryWeightInterner,
) {
    if weight == 0 { return; }
    if let Some((_, existing)) = contribs.iter_mut().find(|(state, _)| *state == target) {
        *existing = interner.union(*existing, weight);
    } else {
        contribs.push((target, weight));
    }
}

fn determinize_fast_boundary_with_fallbacks(
    input: &[FastBoundaryDwaState],
    possible_by_state: &[FastPossibleOutgoingIds],
    num_parser_states: u32,
    interner: &mut FastBoundaryWeightInterner,
) -> Vec<FastBoundaryDwaState> {
    let started_at = Instant::now();
    if input.is_empty() {
        return vec![FastBoundaryDwaState::default()];
    }
    let all = interner.all_id();
    let mut result = vec![FastBoundaryDwaState::default()];
    let mut normalized_singletons = FxHashMap::<u32, u32>::default();
    normalized_singletons.insert(0, 0);
    let mut subset_map = FxHashMap::<Vec<(u32, FastBoundaryWeightId)>, u32>::default();
    subset_map.insert(vec![(0, all)], 0);
    let mut worklist = VecDeque::from([(0u32, vec![(0u32, all)])]);

    let dense_limit = num_parser_states as usize;
    let mut dense = (0..dense_limit).map(|_| FastBoundaryContribs::new()).collect::<Vec<_>>();
    let mut dense_touched = vec![false; dense_limit];
    let mut touched_dense = Vec::<usize>::new();
    let mut default = FastBoundaryContribs::new();
    let mut sparse = FxHashMap::<i32, FastBoundaryContribs>::default();
    let mut default_all = FastBoundaryContribs::new();
    let mut singleton_rows = 0usize;
    let mut complex_rows = 0usize;

    while let Some((from_state, subset)) = worklist.pop_front() {
        if !interner.allow_work(1, result.len(), 0) { return Vec::new(); }
        let mut final_weight = 0;
        for &(state_id, path_weight) in &subset {
            let state_final = input[state_id as usize].final_weight;
            if state_final != 0 {
                let contribution = interner.intersection(path_weight, state_final);
                final_weight = interner.union(final_weight, contribution);
            }
        }
        result[from_state as usize].final_weight = final_weight;

        if let [(input_state, path_weight)] = subset.as_slice()
            && *path_weight == all
            && !input[*input_state as usize]
                .transitions.iter().any(|(label, _, _)| *label == DEFAULT_LABEL)
        {
            singleton_rows += 1;
            let mut rewritten = Vec::with_capacity(input[*input_state as usize].transitions.len());
            for &(label, input_target, weight) in &input[*input_state as usize].transitions {
                if weight == 0 { continue; }
                let target = if let Some(&existing) = normalized_singletons.get(&input_target) {
                    existing
                } else {
                    let created = result.len() as u32;
                    result.push(FastBoundaryDwaState::default());
                    normalized_singletons.insert(input_target, created);
                    subset_map.insert(vec![(input_target, all)], created);
                    worklist.push_back((created, vec![(input_target, all)]));
                    created
                };
                rewritten.push((label, target, weight));
            }
            result[from_state as usize].transitions = rewritten;
            continue;
        }
        complex_rows += 1;
        default_all.clear();

        for &(input_state, path_weight) in &subset {
            let state = &input[input_state as usize];
            for &(label, target, transition_weight) in &state.transitions {
                if label == DEFAULT_LABEL { continue; }
                let next = interner.intersection(path_weight, transition_weight);
                if next == 0 { continue; }
                if label >= 0 && (label as usize) < dense_limit {
                    let index = label as usize;
                    if !dense_touched[index] {
                        dense_touched[index] = true;
                        touched_dense.push(index);
                    }
                    fast_boundary_add_target(&mut dense[index], target, next, interner);
                } else {
                    fast_boundary_add_target(sparse.entry(label).or_default(), target, next, interner);
                }
            }
            let Some((_, default_target, default_weight)) = state
                .transitions.iter().find(|(label, _, _)| *label == DEFAULT_LABEL).copied()
            else { continue; };
            let fallback_weight = interner.intersection(path_weight, default_weight);
            if fallback_weight == 0 { continue; }
            fast_boundary_add_target(&mut default, default_target, fallback_weight, interner);

            // DEFAULT is an additive wildcard branch. It participates both in
            // each explicit row and in the parser states known reachable from
            // this NWA support state.
            for &(label, _, _) in &state.transitions {
                if label == DEFAULT_LABEL { continue; }
                if label >= 0 && (label as usize) < dense_limit {
                    let index = label as usize;
                    if !dense_touched[index] {
                        dense_touched[index] = true;
                        touched_dense.push(index);
                    }
                    fast_boundary_add_target(&mut dense[index], default_target, fallback_weight, interner);
                } else {
                    fast_boundary_add_target(sparse.entry(label).or_default(), default_target, fallback_weight, interner);
                }
            }
            if let Some(possible) = possible_by_state.get(input_state as usize) {
                match possible {
                    FastPossibleOutgoingIds::All => {
                        fast_boundary_add_target(&mut default_all, default_target, fallback_weight, interner);
                    }
                    FastPossibleOutgoingIds::Empty => {}
                    FastPossibleOutgoingIds::Small(_) | FastPossibleOutgoingIds::Bits(_) | FastPossibleOutgoingIds::Shared(_) => {
                        possible.for_each(num_parser_states, |parser_state| {
                            let parser_state = parser_state as usize;
                            if !dense_touched[parser_state] {
                                dense_touched[parser_state] = true;
                                touched_dense.push(parser_state);
                            }
                            fast_boundary_add_target(
                                &mut dense[parser_state],
                                default_target,
                                fallback_weight,
                                interner,
                            );
                        });
                    }
                }
            }
        }

        let process = |label: i32,
                           mut contribs: FastBoundaryContribs,
                           result: &mut Vec<FastBoundaryDwaState>,
                           normalized_singletons: &mut FxHashMap<u32, u32>,
                           subset_map: &mut FxHashMap<Vec<(u32, FastBoundaryWeightId)>, u32>,
                           worklist: &mut VecDeque<(u32, Vec<(u32, FastBoundaryWeightId)>)>,
                           interner: &mut FastBoundaryWeightInterner| {
            if contribs.is_empty() { return; }
            contribs.sort_unstable_by_key(|(state, _)| *state);
            let mut edge_weight = 0;
            for (_, weight) in &contribs {
                edge_weight = interner.union(edge_weight, *weight);
            }
            if edge_weight == 0 { return; }
            let target = if let [(only_state, _)] = contribs.as_slice() {
                if let Some(&existing) = normalized_singletons.get(only_state) {
                    existing
                } else {
                    let created = result.len() as u32;
                    result.push(FastBoundaryDwaState::default());
                    normalized_singletons.insert(*only_state, created);
                    subset_map.insert(vec![(*only_state, all)], created);
                    worklist.push_back((created, vec![(*only_state, all)]));
                    created
                }
            } else if let Some(&existing) = subset_map.get(contribs.as_slice()) {
                existing
            } else {
                let created = result.len() as u32;
                result.push(FastBoundaryDwaState::default());
                subset_map.insert(contribs.clone().into_vec(), created);
                worklist.push_back((created, contribs.into_vec()));
                created
            };
            result[from_state as usize].transitions.push((label, target, edge_weight));
        };

        touched_dense.sort_unstable();
        for label in touched_dense.drain(..) {
            dense_touched[label] = false;
            if !default_all.is_empty() {
                for &(target, weight) in &default_all {
                    fast_boundary_add_target(&mut dense[label], target, weight, interner);
                }
            }
            process(
                label as i32, std::mem::take(&mut dense[label]), &mut result,
                &mut normalized_singletons, &mut subset_map, &mut worklist, interner,
            );
        }
        if !default.is_empty() {
            process(
                DEFAULT_LABEL, std::mem::take(&mut default), &mut result,
                &mut normalized_singletons, &mut subset_map, &mut worklist, interner,
            );
        }
        let mut sparse_rows = sparse.drain().collect::<Vec<_>>();
        sparse_rows.sort_unstable_by_key(|(label, _)| *label);
        for (label, contribs) in sparse_rows {
            process(
                label, contribs, &mut result, &mut normalized_singletons,
                &mut subset_map, &mut worklist, interner,
            );
        }
    }
    if compile_profile_enabled() {
        eprintln!(
            "[glrmask/profile][fast_boundary_fallback] input_states={} output_states={} singleton_rows={} complex_rows={} total_ms={:.3}",
            input.len(), result.len(), singleton_rows, complex_rows, elapsed_ms(started_at),
        );
    }
    result
}

fn subtract_fast_boundary_finals(
    states: &mut [FastBoundaryDwaState],
    interner: &mut FastBoundaryWeightInterner,
) {
    for state in states {
        let final_weight = state.final_weight;
        if final_weight == 0 { continue; }
        for (_, _, weight) in &mut state.transitions {
            *weight = interner.difference(*weight, final_weight);
        }
        state.transitions.retain(|(_, _, weight)| *weight != 0);
    }
}

/// Compile-only bit graph. Unlike the fixed 16-row runtime wire this supports
/// all 64 private rows and never materializes generic weights per edge.
#[derive(Debug, Clone)]
pub struct FiniteBoundaryDwa {
    pub states: Vec<SmallBoundaryDwaState>,
    pub weights: Vec<Box<[u64]>>,
    pub rows: usize,
    pub token_count: usize,
}

impl FiniteBoundaryDwa {
    /// Differential-validation bridge only. The actual compiler can hand this
    /// graph to the finite minimizer without materializing this intermediate.
    pub fn to_generic_dwa(&self) -> DWA {
        let mut weights = vec![None; self.weights.len()];
        let mut decode = |id: u32| -> Weight {
            weights[id as usize].get_or_insert_with(|| {
                if id == 0 { return Weight::empty(); }
                if id == 1 { return Weight::all(); }
                Weight::from_per_tsid_token_sets(self.weights[id as usize].iter().enumerate()
                    .filter_map(|(row, &bits)| (bits != 0).then(|| (row as u32,
                        (0..self.token_count as u32).filter(|&bit| bits & (1u64 << bit) != 0).collect()))))
            }).clone()
        };
        let states = self.states.iter().map(|state| {
            let mut result = DWAState::default();
            if state.final_weight != 0 { result.final_weight = Some(decode(state.final_weight)); }
            for &(label, target, weight) in &state.transitions {
                // Keep explicit empty guards as explicit edges if present.
                result.transitions.insert(label, (target, decode(weight)));
            }
            result
        }).collect();
        DWA::from_parts(states, 0)
    }
}

enum SmallBoundaryDeterminizeOutput {
    Finite(FiniteBoundaryDwa),
    Generic(DeterminizedDwaWithSupports),
    Compact(SmallBoundaryDwa),
}

fn determinize_preconverted_small_boundary_output(
    fast_nwa: &[FastBoundaryNwaState],
    start_states: &[u32],
    dense_positive_label_limit: u32,
    interner: &mut FastBoundaryWeightInterner,
    source_weight_count: usize,
    conversion_ms: f64,
    total_started_at: Instant,
    compact_output: bool,
    finite_output: bool,
) -> Option<SmallBoundaryDeterminizeOutput> {

    let state_count = fast_nwa.len();
    let mut weight_by_state = vec![interner.empty_id(); state_count];
    let mut closure_queue = VecDeque::<u32>::new();
    let mut closure_touched = Vec::<u32>::new();
    let mut start = start_states
        .iter()
        .copied()
        .map(|state| (state, interner.all_id()))
        .collect::<Vec<_>>();
    start.sort_unstable_by_key(|(state, _)| *state);
    start.dedup_by_key(|(state, _)| *state);
    let start = fast_boundary_epsilon_closure(
        fast_nwa,
        interner,
        &start,
        &mut weight_by_state,
        &mut closure_queue,
        &mut closure_touched,
    );
    if !interner.allow_work(0, fast_nwa.len(), 0) { return None; }
    if start.is_empty() {
        return Some(if finite_output {
            SmallBoundaryDeterminizeOutput::Finite(FiniteBoundaryDwa {
                states: vec![SmallBoundaryDwaState::default()],
                weights: interner.values.iter().map(|value| value.to_vec().into_boxed_slice()).collect(),
                rows: interner.tsid_count,
                token_count: interner.token_count,
            })
        } else if compact_output {
            SmallBoundaryDeterminizeOutput::Compact(SmallBoundaryDwa {
                states: vec![SmallBoundaryDwaState::default()],
                weights: interner.compact_runtime_weights()?,
                tsid_count: interner.tsid_count as u8,
                token_count: interner.token_count as u8,
            })
        } else {
            SmallBoundaryDeterminizeOutput::Generic(DeterminizedDwaWithSupports {
                dwa: DWA::new(0, 0),
                supports: vec![Vec::new()],
            })
        });
    }

    let mut out_states = vec![FastBoundaryDwaState::default()];
    let mut supports = vec![start.iter().map(|(state, _)| *state).collect::<Vec<_>>()];
    let mut subset_map = FxHashMap::<Vec<(u32, FastBoundaryWeightId)>, u32>::default();
    subset_map.insert(start.clone(), 0);
    let mut singleton_states = vec![u32::MAX; state_count];
    if let [(state, weight)] = start.as_slice()
        && *weight == interner.all_id()
    {
        singleton_states[*state as usize] = 0;
    }
    let mut worklist = VecDeque::from([(0u32, start)]);
    let mut singleton_closure_cache = FxHashMap::<
        (u32, FastBoundaryWeightId),
        (u32, FastBoundaryWeightId),
    >::default();
    let mut closure_cache = FxHashMap::<
        Vec<(u32, FastBoundaryWeightId)>,
        (u32, FastBoundaryWeightId),
    >::default();

    let dense_limit = dense_positive_label_limit as usize;
    let mut dense = (0..dense_limit)
        .map(|_| FastBoundaryContribs::new())
        .collect::<Vec<_>>();
    let mut dense_touched = vec![false; dense_limit];
    let mut touched_dense = Vec::<usize>::new();
    let mut default = FastBoundaryContribs::new();
    let mut sparse = FxHashMap::<i32, FastBoundaryContribs>::default();
    let determinize_started_at = Instant::now();

    let mut finite_edge_count = 0usize;
    while let Some((from_state, subset)) = worklist.pop_front() {
        if !interner.allow_work(1, out_states.len(), finite_edge_count) { return None; }
        let mut final_weight = interner.empty_id();
        for &(nwa_state, path_weight) in &subset {
            let state_final = fast_nwa[nwa_state as usize].final_weight;
            if state_final != 0 {
                let contribution = interner.intersection(path_weight, state_final);
                final_weight = interner.union(final_weight, contribution);
            }
        }
        out_states[from_state as usize].final_weight = final_weight;

        for &(nwa_state, path_weight) in &subset {
            for (label, branches) in &fast_nwa[nwa_state as usize].transitions {
                for &(target, edge_weight) in branches {
                    let contribution = interner.intersection(path_weight, edge_weight);
                    if contribution == 0 {
                        continue;
                    }
                    if *label >= 0 && (*label as usize) < dense_limit {
                        let index = *label as usize;
                        if !dense_touched[index] {
                            dense_touched[index] = true;
                            touched_dense.push(index);
                        }
                        dense[index].push((target, contribution));
                    } else if *label == DEFAULT_LABEL {
                        default.push((target, contribution));
                    } else {
                        sparse.entry(*label).or_default().push((target, contribution));
                    }
                }
            }
        }

        // NWA transition rows are ordered, so this is already stable in the
        // common case. Sorting the touched label IDs makes subset discovery
        // independent of source-row insertion details.
        touched_dense.sort_unstable();
        for label in touched_dense.drain(..) {
            dense_touched[label] = false;
            if let Some((to_state, edge_weight)) = fast_boundary_process_contribs(
                &mut dense[label],
                fast_nwa,
                interner,
                &mut weight_by_state,
                &mut closure_queue,
                &mut closure_touched,
                &mut singleton_states,
                &mut subset_map,
                &mut singleton_closure_cache,
                &mut closure_cache,
                &mut out_states,
                &mut supports,
                &mut worklist,
            ) {
                out_states[from_state as usize]
                    .transitions
                    .push((label as i32, to_state, edge_weight));
            }
            recycle_fast_boundary_contribs(&mut dense[label]);
        }
        if !default.is_empty() {
            if let Some((to_state, edge_weight)) = fast_boundary_process_contribs(
                &mut default,
                fast_nwa,
                interner,
                &mut weight_by_state,
                &mut closure_queue,
                &mut closure_touched,
                &mut singleton_states,
                &mut subset_map,
                &mut singleton_closure_cache,
                &mut closure_cache,
                &mut out_states,
                &mut supports,
                &mut worklist,
            ) {
                out_states[from_state as usize]
                    .transitions
                    .push((DEFAULT_LABEL, to_state, edge_weight));
            }
            recycle_fast_boundary_contribs(&mut default);
        }
        if !sparse.is_empty() {
            let mut sparse_rows = sparse.drain().collect::<Vec<_>>();
            sparse_rows.sort_unstable_by_key(|(label, _)| *label);
            for (label, mut contribs) in sparse_rows {
                if let Some((to_state, edge_weight)) = fast_boundary_process_contribs(
                    &mut contribs,
                    &fast_nwa,
                    interner,
                    &mut weight_by_state,
                    &mut closure_queue,
                    &mut closure_touched,
                    &mut singleton_states,
                    &mut subset_map,
                    &mut singleton_closure_cache,
                    &mut closure_cache,
                    &mut out_states,
                    &mut supports,
                    &mut worklist,
                ) {
                    out_states[from_state as usize]
                        .transitions
                        .push((label, to_state, edge_weight));
                }
            }
        }
        finite_edge_count = finite_edge_count.saturating_add(out_states[from_state as usize].transitions.len());
        if !interner.allow_work(subset.len(), out_states.len(), finite_edge_count) { return None; }
    }
    let determinize_ms = elapsed_ms(determinize_started_at);
    let compact_post_started_at = Instant::now();
    let compact_fallback = finite_output
        || std::env::var_os("GLRMASK_EXPERIMENT_SMALL_BOUNDARY_COMPACT_FALLBACK").is_some();
    if compact_fallback
        || std::env::var_os("GLRMASK_EXPERIMENT_SMALL_BOUNDARY_COMPACT_POST").is_some()
    {
        let possible_started_at = Instant::now();
        let possible = fast_boundary_possible_outgoing_ids(fast_nwa, &supports, dense_positive_label_limit);
        if std::env::var_os("GLRMASK_VALIDATE_FAST_BOUNDARY_SPARSE_POSSIBLE").is_some() {
            let reference = build_possible_outgoing_ids_by_fast_boundary_state(
                fast_nwa,
                &supports,
                dense_positive_label_limit,
            );
            assert_eq!(possible.len(), reference.len());
            for (state_id, (candidate, reference)) in
                possible.iter().zip(reference.iter()).enumerate()
            {
                let mut candidate_bits = BitSet::new(dense_positive_label_limit as usize);
                match candidate {
                    FastPossibleOutgoingIds::Empty => {}
                    FastPossibleOutgoingIds::All => {
                        candidate_bits = BitSet::all(dense_positive_label_limit as usize);
                    }
                    FastPossibleOutgoingIds::Small(ids) => {
                        for &id in ids { candidate_bits.set(id as usize); }
                    }
                    FastPossibleOutgoingIds::Bits(ids) => candidate_bits = ids.clone(),
                    FastPossibleOutgoingIds::Shared(ids) => { for &id in ids.iter() { candidate_bits.set(id as usize); } },
                }
                let mut reference_bits = BitSet::new(dense_positive_label_limit as usize);
                match reference {
                    PossibleOutgoingIds::Empty => {}
                    PossibleOutgoingIds::All => {
                        reference_bits = BitSet::all(dense_positive_label_limit as usize);
                    }
                    PossibleOutgoingIds::Some(ids) => reference_bits = ids.clone(),
                }
                assert_eq!(
                    candidate_bits,
                    reference_bits,
                    "sparse possible-outgoing mismatch at deterministic state {state_id}",
                );
            }
            eprintln!(
                "[glrmask/validate][fast_boundary_sparse_possible] exact=true states={}",
                possible.len(),
            );
        }
        let possible_ms = elapsed_ms(possible_started_at);
        let default_started_at = Instant::now();
        if !finite_output || std::env::var_os("GLRMASK_EXPERIMENT_LAZY_DIRECT_DISABLE_DEFAULT_OPT").is_none() {
            optimize_fast_boundary_defaults(&mut out_states, &possible, dense_positive_label_limit, interner);
        }
        let default_ms = elapsed_ms(default_started_at);
        let subtract_started_at = Instant::now();
        subtract_fast_boundary_finals(&mut out_states, interner);
        let subtract_ms = elapsed_ms(subtract_started_at);
        let fallback_started_at = Instant::now();
        if compact_fallback {
            out_states = determinize_fast_boundary_with_fallbacks(
                &out_states, &possible, dense_positive_label_limit, interner,
            );
        }
        let fallback_ms = elapsed_ms(fallback_started_at);
        if compile_profile_enabled() {
            eprintln!(
                "[glrmask/profile][fast_boundary_compact_post] possible_ms={possible_ms:.3} default_ms={default_ms:.3} subtract_ms={subtract_ms:.3} fallback_ms={fallback_ms:.3}"
            );
        }
    }
    let compact_post_ms = elapsed_ms(compact_post_started_at);

    if finite_output {
        let edges = out_states.iter().map(|state| state.transitions.len()).sum();
        if !interner.allow_work(0, out_states.len(), edges) { return None; }
        return Some(SmallBoundaryDeterminizeOutput::Finite(FiniteBoundaryDwa {
            states: out_states,
            weights: interner.values.iter().map(|value| value.to_vec().into_boxed_slice()).collect(),
            rows: interner.tsid_count,
            token_count: interner.token_count,
        }));
    }

    if compact_output {
        if compile_profile_enabled() {
            eprintln!(
                "[glrmask/profile][parser_support_small_boundary] nwa_states={} dwa_states={} fast_weights={} source_weights={} closure_cache={} singleton_closure_cache={} conversion_ms={conversion_ms:.3} determinize_ms={determinize_ms:.3} compact_post_ms={compact_post_ms:.3} materialize_ms=0.000 compact_output=true total_ms={:.3}",
                fast_nwa.len(),
                out_states.len(),
                interner.values.len(),
                source_weight_count,
                closure_cache.len(),
                singleton_closure_cache.len(),
                total_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        return Some(SmallBoundaryDeterminizeOutput::Compact(SmallBoundaryDwa {
            states: out_states,
            weights: interner.compact_runtime_weights()?,
            tsid_count: interner.tsid_count as u8,
            token_count: interner.token_count as u8,
        }));
    }

    let materialize_started_at = Instant::now();
    let parallel_materialize = std::env::var_os(
        "GLRMASK_EXPERIMENT_SMALL_BOUNDARY_PARALLEL_MATERIALIZE",
    )
    .is_some()
        && rayon::current_num_threads() > 1;
    let dwa = if parallel_materialize {
        use rayon::prelude::*;
        let mut used_weight = vec![false; interner.values.len()];
        for state in &out_states {
            if state.final_weight != 0 {
                used_weight[state.final_weight as usize] = true;
            }
            for &(_, _, weight) in &state.transitions {
                if weight != 0 {
                    used_weight[weight as usize] = true;
                }
            }
        }
        let used_ids = used_weight
            .iter()
            .enumerate()
            .filter_map(|(id, &used)| used.then_some(id))
            .collect::<Vec<_>>();
        let converted = used_ids
            .par_iter()
            .map(|&id| (id, interner.to_weight(id as FastBoundaryWeightId)))
            .collect::<Vec<_>>();
        let mut weights = vec![None::<Weight>; interner.values.len()];
        for (id, weight) in converted {
            weights[id] = Some(weight);
        }
        let states = out_states
            .into_par_iter()
            .map(|state| DWAState {
                transitions: state
                    .transitions
                    .into_iter()
                    .filter(|(_, _, weight)| *weight != 0)
                    .map(|(label, target, weight)| {
                        (
                            label,
                            (
                                target,
                                weights[weight as usize]
                                    .as_ref()
                                    .expect("used boundary edge weight was not materialized")
                                    .clone(),
                            ),
                        )
                    })
                    .collect(),
                final_weight: (state.final_weight != 0).then(|| {
                    weights[state.final_weight as usize]
                        .as_ref()
                        .expect("used boundary final weight was not materialized")
                        .clone()
                }),
            })
            .collect::<Vec<_>>();
        if compile_profile_enabled() {
            eprintln!(
                "[glrmask/profile][small_boundary_materialize_used_weights] used={} interned={}",
                used_ids.len(),
                interner.values.len(),
            );
        }
        DWA::from_parts(states, 0)
    } else {
        let mut weight_cache = vec![None::<Weight>; interner.values.len()];
        weight_cache[0] = Some(Weight::empty());
        if weight_cache.len() > 1 {
            weight_cache[1] = Some(Weight::all());
        }
        let mut materialized_weight = |id: FastBoundaryWeightId| {
            if let Some(weight) = weight_cache[id as usize].as_ref() {
                return weight.clone();
            }
            let weight = interner.to_weight(id);
            weight_cache[id as usize] = Some(weight.clone());
            weight
        };
        let mut dwa = DWA::new(0, 0);
        for _ in 1..out_states.len() {
            dwa.add_state();
        }
        for (state_id, state) in out_states.into_iter().enumerate() {
            if state.final_weight != 0 {
                dwa.set_final_weight(state_id as u32, materialized_weight(state.final_weight));
            }
            for (label, target, edge_weight) in state.transitions {
                dwa.add_transition(
                    state_id as u32,
                    label,
                    target,
                    materialized_weight(edge_weight),
                );
            }
        }
        dwa
    };
    let materialize_ms = elapsed_ms(materialize_started_at);
    if compile_profile_enabled() {
        eprintln!(
            "[glrmask/profile][parser_support_small_boundary] nwa_states={} dwa_states={} fast_weights={} source_weights={} closure_cache={} singleton_closure_cache={} conversion_ms={conversion_ms:.3} determinize_ms={determinize_ms:.3} compact_post_ms={compact_post_ms:.3} materialize_ms={materialize_ms:.3} total_ms={:.3}",
            fast_nwa.len(),
            dwa.states().len(),
            interner.values.len(),
            source_weight_count,
            closure_cache.len(),
            singleton_closure_cache.len(),
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Some(SmallBoundaryDeterminizeOutput::Generic(
        DeterminizedDwaWithSupports { dwa, supports },
    ))
}

fn determinize_preconverted_small_boundary(
    fast_nwa: &[FastBoundaryNwaState],
    start_states: &[u32],
    dense_positive_label_limit: u32,
    interner: &mut FastBoundaryWeightInterner,
    source_weight_count: usize,
    conversion_ms: f64,
    total_started_at: Instant,
) -> Option<DeterminizedDwaWithSupports> {
    match determinize_preconverted_small_boundary_output(
        fast_nwa,
        start_states,
        dense_positive_label_limit,
        interner,
        source_weight_count,
        conversion_ms,
        total_started_at,
        false,
        false,
    )? {
        SmallBoundaryDeterminizeOutput::Finite(_) => unreachable!("generic output requested"),
        SmallBoundaryDeterminizeOutput::Generic(result) => Some(result),
        SmallBoundaryDeterminizeOutput::Compact(_) => unreachable!("generic output requested"),
    }
}

fn determinize_preconverted_small_boundary_compact(
    fast_nwa: &[FastBoundaryNwaState],
    start_states: &[u32],
    dense_positive_label_limit: u32,
    interner: &mut FastBoundaryWeightInterner,
    source_weight_count: usize,
    conversion_ms: f64,
    total_started_at: Instant,
) -> Option<SmallBoundaryDwa> {
    match determinize_preconverted_small_boundary_output(
        fast_nwa,
        start_states,
        dense_positive_label_limit,
        interner,
        source_weight_count,
        conversion_ms,
        total_started_at,
        true,
        false,
    )? {
        SmallBoundaryDeterminizeOutput::Finite(_) => unreachable!("compact output requested"),
        SmallBoundaryDeterminizeOutput::Compact(result) => Some(result),
        SmallBoundaryDeterminizeOutput::Generic(_) => unreachable!("compact output requested"),
    }
}

fn fast_boundary_reverse_hashcons_positive(
    states: Vec<FastBoundaryNwaState>,
    start_states: &[u32],
) -> Option<(Vec<FastBoundaryNwaState>, Vec<u32>)> {
    fn fingerprint(state: &FastBoundaryNwaState, old_to_new: &[u32]) -> u64 {
        let mut hasher = FxHasher::default();
        state.final_weight.hash(&mut hasher);
        state.transitions.len().hash(&mut hasher);
        for (label, branches) in &state.transitions {
            label.hash(&mut hasher);
            branches.len().hash(&mut hasher);
            for (target, weight) in branches {
                old_to_new[*target as usize].hash(&mut hasher);
                weight.hash(&mut hasher);
            }
        }
        state.epsilons.len().hash(&mut hasher);
        for (target, weight) in &state.epsilons {
            old_to_new[*target as usize].hash(&mut hasher);
            weight.hash(&mut hasher);
        }
        hasher.finish()
    }
    fn equivalent(
        left: &FastBoundaryNwaState,
        right: &FastBoundaryNwaState,
        old_to_new: &[u32],
    ) -> bool {
        left.final_weight == right.final_weight
            && left.transitions.len() == right.transitions.len()
            && left.epsilons.len() == right.epsilons.len()
            && left.transitions.iter().zip(&right.transitions).all(
                |((left_label, left_branches), (right_label, right_branches))| {
                    left_label == right_label
                        && left_branches.len() == right_branches.len()
                        && left_branches.iter().zip(right_branches).all(
                            |((left_target, left_weight), (right_target, right_weight))| {
                                old_to_new[*left_target as usize]
                                    == old_to_new[*right_target as usize]
                                    && left_weight == right_weight
                            },
                        )
                },
            )
            && left.epsilons.iter().zip(&right.epsilons).all(
                |((left_target, left_weight), (right_target, right_weight))| {
                    old_to_new[*left_target as usize] == old_to_new[*right_target as usize]
                        && left_weight == right_weight
                },
            )
    }

    let started = Instant::now();
    let order = fast_boundary_topological_order(&states)?;
    let n = states.len();
    let mut old_to_new = vec![u32::MAX; n];
    let mut representatives = Vec::<u32>::new();
    let mut buckets = FxHashMap::<u64, SmallVec<[u32; 2]>>::default();
    for old_id in order.into_iter().rev() {
        let state = &states[old_id as usize];
        debug_assert!(state
            .transitions
            .iter()
            .all(|(label, _)| !is_negative_label(*label)));
        let fp = fingerprint(state, &old_to_new);
        let existing = buckets.get(&fp).and_then(|candidates| {
            candidates.iter().copied().find(|&candidate| {
                equivalent(
                    state,
                    &states[representatives[candidate as usize] as usize],
                    &old_to_new,
                )
            })
        });
        let id = if let Some(existing) = existing {
            existing
        } else {
            let id = representatives.len() as u32;
            representatives.push(old_id);
            buckets.entry(fp).or_default().push(id);
            id
        };
        old_to_new[old_id as usize] = id;
    }

    let mut source = states.into_iter().map(Some).collect::<Vec<_>>();
    let mut output = Vec::with_capacity(representatives.len());
    for &old_id in &representatives {
        let mut state = source[old_id as usize]
            .take()
            .expect("fast boundary representative state must exist");
        for (_, branches) in &mut state.transitions {
            for (target, _) in branches {
                *target = old_to_new[*target as usize];
            }
        }
        for (target, _) in &mut state.epsilons {
            *target = old_to_new[*target as usize];
        }
        output.push(state);
    }
    let mut starts = start_states
        .iter()
        .filter_map(|&state| old_to_new.get(state as usize).copied())
        .collect::<Vec<_>>();
    starts.sort_unstable();
    starts.dedup();
    if compile_profile_enabled() {
        eprintln!(
            "[glrmask/profile][fast_boundary_hashcons] input_states={} output_states={} total_ms={:.3}",
            n,
            output.len(),
            elapsed_ms(started),
        );
    }
    Some((output, starts))
}

fn build_possible_outgoing_ids_by_fast_boundary_state(
    parser_nwa: &[FastBoundaryNwaState],
    state_supports: &[Vec<u32>],
    num_parser_states: u32,
) -> Vec<PossibleOutgoingIds> {
    #[derive(Clone)]
    enum OutgoingIds {
        Empty,
        All,
        Some(Vec<u32>),
    }
    let n = num_parser_states as usize;
    let all_parser_states = BitSet::all(n);
    let summarize = |state: &FastBoundaryNwaState| {
        let mut ids = Vec::new();
        for (label, branches) in &state.transitions {
            if branches.is_empty() {
                continue;
            }
            if *label == DEFAULT_LABEL {
                return OutgoingIds::All;
            }
            if let Some(parser_state_id) = parser_state_label(*label, num_parser_states) {
                ids.push(parser_state_id);
            }
        }
        if ids.is_empty() {
            OutgoingIds::Empty
        } else {
            ids.sort_unstable();
            ids.dedup();
            OutgoingIds::Some(ids)
        }
    };
    let state_outgoing = if rayon::current_num_threads() > 1 && parser_nwa.len() >= 4_096 {
        use rayon::prelude::*;
        parser_nwa.par_iter().map(summarize).collect::<Vec<_>>()
    } else {
        parser_nwa.iter().map(summarize).collect::<Vec<_>>()
    };
    let summarize_support = |support: &Vec<u32>| {
        if support.len() == 1 {
            return match state_outgoing.get(support[0] as usize) {
                Some(OutgoingIds::Empty) | None => PossibleOutgoingIds::Empty,
                Some(OutgoingIds::All) => PossibleOutgoingIds::All,
                Some(OutgoingIds::Some(ids)) => {
                    let mut set = BitSet::new(n);
                    for &id in ids {
                        set.set(id as usize);
                    }
                    if set == all_parser_states {
                        PossibleOutgoingIds::All
                    } else {
                        PossibleOutgoingIds::Some(set)
                    }
                }
            };
        }
        let mut set = BitSet::new(n);
        for &state in support {
            match state_outgoing.get(state as usize) {
                Some(OutgoingIds::All) => return PossibleOutgoingIds::All,
                Some(OutgoingIds::Some(ids)) => {
                    for &id in ids {
                        set.set(id as usize);
                    }
                    if set == all_parser_states {
                        return PossibleOutgoingIds::All;
                    }
                }
                Some(OutgoingIds::Empty) | None => {}
            }
        }
        if set.is_empty() {
            PossibleOutgoingIds::Empty
        } else if set == all_parser_states {
            PossibleOutgoingIds::All
        } else {
            PossibleOutgoingIds::Some(set)
        }
    };
    if rayon::current_num_threads() > 1 && state_supports.len() >= 1_024 {
        use rayon::prelude::*;
        state_supports.par_iter().map(summarize_support).collect()
    } else {
        state_supports.iter().map(summarize_support).collect()
    }
}

/// Exact small-coordinate publication from a signed parser NWA. The signed
/// graph is converted once to the fixed-width boundary semiring; cancellation,
/// finality and support determinization all consume that same representation.
/// The returned runtime artifact is still the ordinary deterministic positive
/// parser DWA.
pub fn normalize_signed_weighted_parser_stack_nwa_small_boundary(
    table: &GLRTable,
    signed_nwa: &NWA,
    tsid_count: usize,
    token_count: usize,
) -> Option<DWA> {
    normalize_signed_weighted_parser_stack_nwa_small_boundary_for_parser_state_count(
        table.num_states,
        signed_nwa,
        tsid_count,
        token_count,
    )
}

pub fn normalize_signed_weighted_parser_stack_nwa_small_boundary_for_parser_state_count(
    num_parser_states: u32,
    signed_nwa: &NWA,
    tsid_count: usize,
    token_count: usize,
) -> Option<DWA> {
    if tsid_count == 0 || tsid_count > FAST_BOUNDARY_TSID_LIMIT || token_count == 0 || token_count > 64 {
        return None;
    }
    let total_started = Instant::now();
    let convert_started = Instant::now();
    let mut interner = FastBoundaryWeightInterner::new(tsid_count, token_count)?;
    let mut source_weight_ids = FxHashMap::<usize, FastBoundaryWeightId>::default();
    let mut fast_nwa = Vec::with_capacity(signed_nwa.states().len());
    for state in signed_nwa.states() {
        let final_weight = match state.final_weight.as_ref() {
            Some(weight) => interner.source_weight_id(weight, &mut source_weight_ids, None)?,
            None => 0,
        };
        let mut epsilons = Vec::with_capacity(state.epsilons.len());
        for (target, weight) in &state.epsilons {
            let id = interner.source_weight_id(weight, &mut source_weight_ids, None)?;
            if id != 0 {
                epsilons.push((*target, id));
            }
        }
        let mut transitions = Vec::with_capacity(state.transitions.len());
        for (&label, branches) in &state.transitions {
            let mut row = SmallVec::with_capacity(branches.len());
            for (target, weight) in branches {
                let id = interner.source_weight_id(weight, &mut source_weight_ids, None)?;
                if id != 0 {
                    row.push((*target, id));
                }
            }
            if !row.is_empty() {
                transitions.push((label, row));
            }
        }
        fast_nwa.push(FastBoundaryNwaState {
            epsilons,
            transitions,
            final_weight,
        });
    }
    let convert_ms = elapsed_ms(convert_started);
    let resolve_started = Instant::now();
    fast_boundary_resolve_negative_codes(&mut fast_nwa, &mut interner)?;
    let resolve_ms = elapsed_ms(resolve_started);
    let hashcons_started = Instant::now();
    let (fast_nwa, fast_start_states) =
        fast_boundary_reverse_hashcons_positive(fast_nwa, signed_nwa.start_states())?;
    let hashcons_ms = elapsed_ms(hashcons_started);

    let determinize_started = Instant::now();
    let determinized = determinize_preconverted_small_boundary(
        &fast_nwa,
        &fast_start_states,
        num_parser_states,
        &mut interner,
        source_weight_ids.len(),
        convert_ms,
        total_started,
    )?;
    let determinize_ms = elapsed_ms(determinize_started);
    let mut parser_dwa = determinized.dwa;
    let compact_post = std::env::var_os("GLRMASK_EXPERIMENT_SMALL_BOUNDARY_COMPACT_FALLBACK").is_some()
        || std::env::var_os("GLRMASK_EXPERIMENT_SMALL_BOUNDARY_COMPACT_POST").is_some();
    if compact_post {
        if compile_profile_enabled() {
            eprintln!(
                "[glrmask/profile][small_boundary_signed_fused] signed_states={} positive_states={} source_weights={} fast_weights={} convert_ms={convert_ms:.3} resolve_ms={resolve_ms:.3} hashcons_ms={hashcons_ms:.3} compact_publication=true output_states={} output_transitions={} total_ms={:.3}",
                signed_nwa.states().len(),
                fast_nwa.len(),
                source_weight_ids.len(),
                interner.values.len(),
                parser_dwa.num_states(),
                parser_dwa.num_transitions(),
                elapsed_ms(total_started),
            );
        }
        return Some(parser_dwa);
    }

    let possible_started = Instant::now();
    let possible_by_state = build_possible_outgoing_ids_by_fast_boundary_state(
        &fast_nwa,
        &determinized.supports,
        num_parser_states,
    );
    let possible_ms = elapsed_ms(possible_started);
    let default_started = Instant::now();
    optimize_parser_dwa_defaults(&mut parser_dwa, &possible_by_state, num_parser_states);
    let default_ms = elapsed_ms(default_started);
    let subtract_started = Instant::now();
    subtract_final_weights_from_outgoing_dwa(&mut parser_dwa);
    let subtract_ms = elapsed_ms(subtract_started);
    let fallback_started = Instant::now();
    parser_dwa = determinize_parser_dwa_with_fallbacks(
        &parser_dwa,
        &possible_by_state,
        num_parser_states,
    );
    let fallback_ms = elapsed_ms(fallback_started);
    if compile_profile_enabled() {
        eprintln!(
            "[glrmask/profile][small_boundary_signed_fused] signed_states={} positive_states={} source_weights={} fast_weights={} convert_ms={convert_ms:.3} resolve_ms={resolve_ms:.3} hashcons_ms={hashcons_ms:.3} determinize_ms={determinize_ms:.3} possible_ms={possible_ms:.3} default_ms={default_ms:.3} subtract_ms={subtract_ms:.3} fallback_ms={fallback_ms:.3} output_states={} output_transitions={} total_ms={:.3}",
            signed_nwa.states().len(),
            fast_nwa.len(),
            source_weight_ids.len(),
            interner.values.len(),
            parser_dwa.num_states(),
            parser_dwa.num_transitions(),
            elapsed_ms(total_started),
        );
    }
    Some(parser_dwa)
}

fn determinize_with_supports_small_boundary(
    nwa: &NWA,
    dense_positive_label_limit: u32,
    tsid_count: usize,
    token_count: usize,
    source_tsid_map: Option<&[u32]>,
) -> Option<DeterminizedDwaWithSupports> {
    if std::env::var_os("GLRMASK_DISABLE_PARSER_SUPPORT_NORMALIZE_SINGLETONS").is_some()
        || (std::env::var_os("GLRMASK_PARSER_SUPPORT_NORMALIZE_SINGLETONS").is_none()
            && nwa.states().len()
                < std::env::var("GLRMASK_PARSER_SUPPORT_NORMALIZE_SINGLETON_MIN_NWA_STATES")
                    .ok()
                    .and_then(|value| value.trim().parse::<usize>().ok())
                    .unwrap_or(4_096))
        || (std::env::var_os("GLRMASK_PARSER_SUPPORT_NORMALIZE_SUBSETS").is_some()
            && std::env::var_os("GLRMASK_DISABLE_PARSER_SUPPORT_NORMALIZE_SUBSETS").is_none())
    {
        return None;
    }
    let total_started_at = Instant::now();
    let mut interner = FastBoundaryWeightInterner::new(tsid_count, token_count)?;
    let mut source_weight_ids = FxHashMap::<usize, FastBoundaryWeightId>::default();
    let mut fast_nwa = Vec::with_capacity(nwa.states().len());
    for state in nwa.states() {
        let final_weight = match state.final_weight.as_ref() {
            Some(weight) => interner.source_weight_id(weight, &mut source_weight_ids, source_tsid_map)?,
            None => interner.empty_id(),
        };
        let mut epsilons = Vec::with_capacity(state.epsilons.len());
        for (target, weight) in &state.epsilons {
            let weight = interner.source_weight_id(weight, &mut source_weight_ids, source_tsid_map)?;
            if weight != 0 {
                epsilons.push((*target, weight));
            }
        }
        let mut transitions = Vec::with_capacity(state.transitions.len());
        for (&label, branches) in &state.transitions {
            let mut fast_branches = SmallVec::with_capacity(branches.len());
            for (target, weight) in branches {
                let weight = interner.source_weight_id(weight, &mut source_weight_ids, source_tsid_map)?;
                if weight != 0 {
                    fast_branches.push((*target, weight));
                }
            }
            if !fast_branches.is_empty() {
                transitions.push((label, fast_branches));
            }
        }
        fast_nwa.push(FastBoundaryNwaState {
            epsilons,
            transitions,
            final_weight,
        });
    }
    let conversion_ms = elapsed_ms(total_started_at);
    determinize_preconverted_small_boundary(
        &fast_nwa,
        nwa.start_states(),
        dense_positive_label_limit,
        &mut interner,
        source_weight_ids.len(),
        conversion_ms,
        total_started_at,
    )
}

/// Exact small-boundary deterministic parser DWA without generic `Weight`
/// materialization.  This is the same positive deterministic automaton produced
/// by the ordinary small-boundary path, retaining the compact TSID×token-mask
/// weight algebra as its runtime representation.
pub fn normalize_weighted_parser_stack_nwa_small_boundary_compact_for_parser_state_count(
    nwa: &NWA,
    dense_positive_label_limit: u32,
    tsid_count: usize,
    token_count: usize,
    source_tsid_map: Option<&[u32]>,
) -> Option<SmallBoundaryDwa> {
    if std::env::var_os("GLRMASK_DISABLE_PARSER_SUPPORT_NORMALIZE_SINGLETONS").is_some()
        || (std::env::var_os("GLRMASK_PARSER_SUPPORT_NORMALIZE_SINGLETONS").is_none()
            && nwa.states().len()
                < std::env::var("GLRMASK_PARSER_SUPPORT_NORMALIZE_SINGLETON_MIN_NWA_STATES")
                    .ok()
                    .and_then(|value| value.trim().parse::<usize>().ok())
                    .unwrap_or(4_096))
        || (std::env::var_os("GLRMASK_PARSER_SUPPORT_NORMALIZE_SUBSETS").is_some()
            && std::env::var_os("GLRMASK_DISABLE_PARSER_SUPPORT_NORMALIZE_SUBSETS").is_none())
    {
        return None;
    }
    let total_started_at = Instant::now();
    let mut interner = FastBoundaryWeightInterner::new(tsid_count, token_count)?;
    let mut source_weight_ids = FxHashMap::<usize, FastBoundaryWeightId>::default();
    let mut fast_nwa = Vec::with_capacity(nwa.states().len());
    for state in nwa.states() {
        let final_weight = match state.final_weight.as_ref() {
            Some(weight) => interner.source_weight_id(weight, &mut source_weight_ids, source_tsid_map)?,
            None => interner.empty_id(),
        };
        let mut epsilons = Vec::with_capacity(state.epsilons.len());
        for (target, weight) in &state.epsilons {
            let weight = interner.source_weight_id(weight, &mut source_weight_ids, source_tsid_map)?;
            if weight != 0 {
                epsilons.push((*target, weight));
            }
        }
        let mut transitions = Vec::with_capacity(state.transitions.len());
        for (&label, branches) in &state.transitions {
            let mut fast_branches = SmallVec::with_capacity(branches.len());
            for (target, weight) in branches {
                let weight = interner.source_weight_id(weight, &mut source_weight_ids, source_tsid_map)?;
                if weight != 0 {
                    fast_branches.push((*target, weight));
                }
            }
            if !fast_branches.is_empty() {
                transitions.push((label, fast_branches));
            }
        }
        fast_nwa.push(FastBoundaryNwaState {
            epsilons,
            transitions,
            final_weight,
        });
    }
    let conversion_ms = elapsed_ms(total_started_at);
    determinize_preconverted_small_boundary_compact(
        &fast_nwa,
        nwa.start_states(),
        dense_positive_label_limit,
        &mut interner,
        source_weight_ids.len(),
        conversion_ms,
        total_started_at,
    )
}

/// Retain the exact ordinary positive compiler graph in private bit coordinates.
pub fn normalize_weighted_parser_stack_nwa_finite_for_parser_state_count(
    nwa: &NWA, parser_states: u32, rows: usize, token_count: usize,
    source_tsid_map: Option<&[u32]>,
) -> Option<FiniteBoundaryDwa> {
    normalize_finite_impl(nwa, parser_states, rows, token_count, source_tsid_map,
                         false, FiniteCompileLimits::default())
}

pub fn normalize_signed_parser_stack_nwa_finite_for_parser_state_count(
    nwa: &NWA, parser_states: u32, rows: usize, token_count: usize,
    source_tsid_map: Option<&[u32]>,
) -> Option<FiniteBoundaryDwa> {
    normalize_finite_impl(nwa, parser_states, rows, token_count, source_tsid_map,
                         true, FiniteCompileLimits::default())
}

fn normalize_finite_impl(
    nwa: &NWA, dense_positive_label_limit: u32, tsid_count: usize, token_count: usize,
    source_tsid_map: Option<&[u32]>, signed: bool, limits: FiniteCompileLimits,
) -> Option<FiniteBoundaryDwa> {
    normalize_finite_with_read_context_impl(nwa,dense_positive_label_limit,tsid_count,token_count,
        source_tsid_map,signed,limits,None)
}

/// The context is a necessary rejection domain certified by the caller's
/// intact parser effects. No state quotient or speculative DEFAULT policy is
/// changed by this entry point. The original entry point remains the control.
pub fn normalize_signed_parser_stack_nwa_finite_with_read_context(
    nwa:&NWA,parser_states:u32,rows:usize,token_count:usize,
    source_tsid_map:Option<&[u32]>,context:&FiniteParserReadSupport,
)->Option<FiniteBoundaryDwa>{
    normalize_finite_with_read_context_impl(nwa,parser_states,rows,token_count,
        source_tsid_map,true,FiniteCompileLimits::default(),Some(context))
}

fn normalize_finite_with_read_context_impl(
    nwa: &NWA, dense_positive_label_limit: u32, tsid_count: usize, token_count: usize,
    source_tsid_map: Option<&[u32]>, signed: bool, limits: FiniteCompileLimits,
    read_context: Option<&FiniteParserReadSupport>,
) -> Option<FiniteBoundaryDwa> {
    if std::env::var_os("GLRMASK_DISABLE_PARSER_SUPPORT_NORMALIZE_SINGLETONS").is_some()
        || (std::env::var_os("GLRMASK_PARSER_SUPPORT_NORMALIZE_SINGLETONS").is_none()
            && nwa.states().len()
                < std::env::var("GLRMASK_PARSER_SUPPORT_NORMALIZE_SINGLETON_MIN_NWA_STATES")
                    .ok()
                    .and_then(|value| value.trim().parse::<usize>().ok())
                    .unwrap_or(4_096))
        || (std::env::var_os("GLRMASK_PARSER_SUPPORT_NORMALIZE_SUBSETS").is_some()
            && std::env::var_os("GLRMASK_DISABLE_PARSER_SUPPORT_NORMALIZE_SUBSETS").is_none())
    {
        return None;
    }
    let n = nwa.states().len();
    if n == 0 || n > limits.states || nwa.num_transitions() > limits.edges
        || dense_positive_label_limit == 0 || dense_positive_label_limit as usize > limits.states
        || dense_positive_label_limit >= DEFAULT_LABEL as u32
        || limits.weights < 2 || limits.words < tsid_count * 2
        || nwa.start_states().iter().any(|&state| state as usize >= n)
        || !nwa.is_acyclic() {
        return None;
    }
    for state in nwa.states() {
        for (&label, branches) in &state.transitions {
            let valid = label == DEFAULT_LABEL || (0..dense_positive_label_limit as i32).contains(&label)
                || (signed && is_negative_label(label)
                    && (0..dense_positive_label_limit as i32).contains(&negative_to_positive_label(label)));
            if !valid || branches.iter().any(|(target, _)| *target as usize >= n) { return None; }
        }
        if state.epsilons.iter().any(|(target, _)| *target as usize >= n) { return None; }
    }
    let total_started_at = Instant::now();
    let mut interner = FastBoundaryWeightInterner::new(tsid_count, token_count)?;
    interner.limits = Some(limits);
    let mut source_weight_ids = FxHashMap::<usize, FastBoundaryWeightId>::default();
    let mut fast_nwa = Vec::with_capacity(nwa.states().len());
    for state in nwa.states() {
        let final_weight = match state.final_weight.as_ref() {
            Some(weight) => interner.source_weight_id(weight, &mut source_weight_ids, source_tsid_map)?,
            None => interner.empty_id(),
        };
        let mut epsilons = Vec::with_capacity(state.epsilons.len());
        for (target, weight) in &state.epsilons {
            let weight = interner.source_weight_id(weight, &mut source_weight_ids, source_tsid_map)?;
            epsilons.push((*target, weight));
        }
        let mut transitions = Vec::with_capacity(state.transitions.len());
        for (&label, branches) in &state.transitions {
            let mut fast_branches = SmallVec::with_capacity(branches.len());
            for (target, weight) in branches {
                let weight = interner.source_weight_id(weight, &mut source_weight_ids, source_tsid_map)?;
                if weight != 0 {
                    fast_branches.push((*target, weight));
                }
            }
            if fast_branches.is_empty() && (!signed || !branches.is_empty()) {
                fast_branches.push((0, 0));
            }
            transitions.push((label, fast_branches));
        }
        fast_nwa.push(FastBoundaryNwaState {
            epsilons,
            transitions,
            final_weight,
        });
    }
    if !interner.allow_work(0, n, nwa.num_transitions()) { return None; }
    if signed {
        // Keep logical state identities intact: no early hash-consing.
        fast_boundary_resolve_negative_codes(&mut fast_nwa, &mut interner)?;
    }
    if !interner.allow_work(0, n, 0) { return None; }
    if let Some(context)=read_context {
        let started=Instant::now();
        let profile=finite_read_support::restrict(&mut fast_nwa,nwa.start_states(),context)?;
        if compile_profile_enabled() || std::env::var_os("GLRMASK_PROFILE_COMPOSE").is_some(){
            eprintln!("[glrmask/profile][native_predecessor_support] ms={:.3} profile={profile:?}",elapsed_ms(started));
        }
    }
    // Diagnostic only: exact row sharing preserves the weighted positive NWA,
    // including its local label guards. It can expose more opportunities to
    // the contextual DEFAULT normalizer, so accepted-language equivalence
    // alone is NOT an acceptance certificate for this experimental path.
    let canonical_starts;
    let start_states = if signed
        && std::env::var_os("GLRMASK_BOUNDARY_NATIVE_GUARD_HASHCONS").is_some()
    {
        let (canonical, starts) = fast_boundary_reverse_hashcons_positive(
            fast_nwa, nwa.start_states(),
        )?;
        fast_nwa = canonical;
        canonical_starts = starts;
        canonical_starts.as_slice()
    } else {
        nwa.start_states()
    };
    let conversion_ms = elapsed_ms(total_started_at);
    match determinize_preconverted_small_boundary_output(
        &fast_nwa, start_states, dense_positive_label_limit,
        &mut interner, source_weight_ids.len(), conversion_ms, total_started_at,
        false, true,
    )? {
        SmallBoundaryDeterminizeOutput::Finite(result) => Some(result),
        _ => unreachable!("finite compile-time output requested"),
    }
}

fn parser_support_defer_edge_unions_enabled(nwa_states: usize) -> bool {
    std::env::var_os("GLRMASK_DISABLE_PARSER_SUPPORT_DEFER_EDGE_UNIONS").is_none()
        && rayon::current_num_threads() > 1
        && (std::env::var_os("GLRMASK_PARSER_SUPPORT_DEFER_EDGE_UNIONS").is_some()
            || nwa_states >= 512)
}

fn determinize_with_supports(
    nwa: &NWA,
    dense_positive_label_limit: Option<u32>,
) -> DeterminizedDwaWithSupports {
    determinize_with_supports_mode(nwa, dense_positive_label_limit, None, None, None)
}

fn determinize_parser_dwa_with_fallbacks_impl(
    dwa: &DWA,
    possible_by_state: &[PossibleOutgoingIds],
    num_parser_states: u32,
    normalize_singletons: bool,
) -> DWA {
    fn subset_key(entries: &[(u32, Weight)]) -> Vec<(u32, usize)> {
        entries.iter().map(|(sid, w)| (*sid, w.ptr_key())).collect()
    }

    let dense_label_limit = num_parser_states as usize;
    let fixed_singleton_ids = normalize_singletons
        && dwa.states().len() >= std::env::var("GLRMASK_FALLBACK_FIXED_SINGLETON_MIN_STATES")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(16_384)
        && std::env::var_os("GLRMASK_DISABLE_FALLBACK_FIXED_SINGLETON_IDS").is_none();
    let mut result = if fixed_singleton_ids {
        DWA::from_parts(vec![DWAState::default(); dwa.states().len()], dwa.start_state())
    } else {
        DWA::new(0, 0)
    };
    let mut fixed_singleton_scheduled = fixed_singleton_ids.then(|| vec![false; dwa.states().len()]);

    let mut start_subset = FxHashMap::default();
    start_subset.insert(dwa.start_state(), Weight::all());

    let mut canon_buf: Vec<(u32, Weight)> = start_subset
        .iter()
        .map(|(state_id, weight)| (*state_id, weight.clone()))
        .collect();
    canon_buf.sort_unstable_by_key(|(state_id, _)| *state_id);

    let mut subset_map: FxHashMap<Vec<(u32, usize)>, u32> = FxHashMap::default();
    // A singleton weighted subset `(q, w)` denotes `w ∩ L(q)`. The incoming
    // transition already carries `w`, so the destination can always be the
    // same canonical state for `L(q)` with residual `all`. Keeping `w` in the
    // state key duplicates rows for different accumulated token sets and was
    // the dominant cost of fallback determinization on large grammars.
    let mut normalized_singleton_subsets: FxHashMap<u32, u32> = FxHashMap::default();
    let mut weighted_singleton_subsets: FxHashMap<(u32, usize), u32> = FxHashMap::default();
    let normalized_singleton_weight = Weight::all();
    let normalized_singleton_key = normalized_singleton_weight.ptr_key();
    let start_key = subset_key(&canon_buf);
    subset_map.insert(start_key, result.start_state());
    if let [(state_id, weight)] = canon_buf.as_slice() {
        if normalize_singletons && weight.is_full() {
            if fixed_singleton_ids {
                debug_assert_eq!(*state_id, result.start_state());
                fixed_singleton_scheduled.as_mut().unwrap()[*state_id as usize] = true;
            } else {
                normalized_singleton_subsets.insert(*state_id, result.start_state());
            }
        } else {
            weighted_singleton_subsets
                .insert((*state_id, weight.ptr_key()), result.start_state());
        }
    }
    let mut worklist: VecDeque<(u32, Vec<(u32, Weight)>)> = VecDeque::new();
    worklist.push_back((result.start_state(), canon_buf.clone()));

    let mut dense_raw_targets: Vec<TargetContribs> =
        (0..dense_label_limit).map(|_| TargetContribs::new()).collect();
    let mut default_raw_targets: TargetContribs = TargetContribs::new();
    let mut sparse_raw_targets: FxHashMap<i32, TargetContribs> = FxHashMap::default();
    let mut touched_dense_labels: Vec<usize> = Vec::new();
    let mut dense_label_touched: Vec<bool> = vec![false; dense_label_limit];
    let mut default_touched = false;
    let mut dense_default_all_raw_targets: TargetContribs = TargetContribs::new();
    let mut intersection_cache = ScopedWeightOpCache::default();
    let mut key_buf: Vec<(u32, usize)> = Vec::new();
    let mut final_contributions: Vec<Weight> = Vec::new();
    let mut detail =
        ParserDwaDeterminizeDetail::enabled().then(ParserDwaDeterminizeDetail::default);

    struct PreparedFallbackSingletonRow {
        from_state: u32,
        state: DWAState,
        targets: Vec<u32>,
    }
    let parallel_fixed_singletons = fixed_singleton_ids
        && detail.is_none()
        && rayon::current_num_threads() > 1
        && std::env::var_os("GLRMASK_DISABLE_FALLBACK_PARALLEL_SINGLETON_ROWS").is_none();
    let fallback_parallel_min = std::env::var("GLRMASK_FALLBACK_PARALLEL_MIN_FRONTIER")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|&value| value > 1)
        .unwrap_or(16);
    let fallback_parallel_wave = std::env::var("GLRMASK_FALLBACK_PARALLEL_WAVE")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(2_048);
    let fallback_profile = compile_profile_enabled() && detail.is_none();
    let fallback_started = fallback_profile.then(Instant::now);
    let mut fallback_parallel_waves = 0usize;
    let mut fallback_parallel_rows = 0usize;
    let mut prepared_singleton_rows = VecDeque::<PreparedFallbackSingletonRow>::new();
    let mut serial_pending = VecDeque::<(u32, Vec<(u32, Weight)>)>::new();

    while !worklist.is_empty() || !serial_pending.is_empty() || !prepared_singleton_rows.is_empty() {
        if prepared_singleton_rows.is_empty()
            && serial_pending.is_empty()
            && parallel_fixed_singletons
            && worklist.len() >= fallback_parallel_min
        {
            use rayon::prelude::*;
            let wave_len = worklist.len().min(fallback_parallel_wave);
            let wave = worklist.drain(..wave_len).collect::<Vec<_>>();
            let mut eligible = Vec::<(u32, u32)>::new();
            for (from_state, subset_entries) in wave {
                let eligible_state = match subset_entries.as_slice() {
                    [(dwa_state_id, path_weight)]
                        if path_weight.is_full()
                            && from_state == *dwa_state_id
                            && !dwa.states()[*dwa_state_id as usize]
                                .transitions
                                .contains_key(&DEFAULT_LABEL) => Some(*dwa_state_id),
                    _ => None,
                };
                if let Some(dwa_state_id) = eligible_state {
                    eligible.push((from_state, dwa_state_id));
                } else {
                    serial_pending.push_back((from_state, subset_entries));
                }
            }
            if !eligible.is_empty() {
                fallback_parallel_waves += 1;
                fallback_parallel_rows += eligible.len();
                let prepared = eligible
                    .into_par_iter()
                    .map(|(from_state, dwa_state_id)| {
                        let mut state = dwa.states()[dwa_state_id as usize].clone();
                        state.transitions.retain(|_, (_, weight)| !weight.is_empty());
                        let targets = state
                            .transitions
                            .values()
                            .map(|(target, _)| *target)
                            .collect::<Vec<_>>();
                        PreparedFallbackSingletonRow {
                            from_state,
                            state,
                            targets,
                        }
                    })
                    .collect::<Vec<_>>();
                prepared_singleton_rows.extend(prepared);
            }
        }

        if let Some(prepared) = prepared_singleton_rows.pop_front() {
            let scheduled = fixed_singleton_scheduled
                .as_mut()
                .expect("parallel singleton fallback requires fixed state IDs");
            for input_target in prepared.targets {
                if !scheduled[input_target as usize] {
                    scheduled[input_target as usize] = true;
                    worklist.push_back((
                        input_target,
                        vec![(input_target, normalized_singleton_weight.clone())],
                    ));
                }
            }
            result.states_mut()[prepared.from_state as usize] = prepared.state;
            continue;
        }

        let (from_state, subset_entries) = serial_pending
            .pop_front()
            .or_else(|| worklist.pop_front())
            .expect("fallback determinizer work queues unexpectedly empty");
        dense_default_all_raw_targets.clear();
        if let Some(detail) = detail.as_mut() {
            detail.states_processed += 1;
        }

        final_contributions.clear();
        let scan_started = detail.as_ref().map(|_| Instant::now());
        for (state_id, path_weight) in &subset_entries {
            let Some(state_final) = dwa.states()[*state_id as usize].final_weight.as_ref() else {
                continue;
            };
            if let Some(detail) = detail.as_mut() {
                detail.intersection_calls += 1;
            }
            let contribution = intersection_cache.intersection(path_weight, state_final);
            if !contribution.is_empty() {
                if let Some(detail) = detail.as_mut() {
                    detail.nonempty_intersections += 1;
                }
                final_contributions.push(contribution);
            }
        }
        let final_weight = Weight::union_all(final_contributions.iter());
        if !final_weight.is_empty() {
            result.set_final_weight(from_state, final_weight);
        }

        // The fallback pass overwhelmingly visits singleton subsets whose input
        // state has no DEFAULT edge. In that case every output transition stays
        // singleton, so staging contributions by label and running the generic
        // subset path is pure overhead.
        if normalize_singletons
            && let [(dwa_state_id, path_weight)] = subset_entries.as_slice()
            && path_weight.is_full()
            && let Some(state) = dwa.states().get(*dwa_state_id as usize)
            && !state.transitions.contains_key(&DEFAULT_LABEL)
        {
            // Preserve the already-sorted row allocation and rewrite only its
            // targets/weights. This avoids millions of individual BTreeMap
            // insertions in the dominant singleton fallback path.
            let mut rewritten = state.transitions.clone();
            let mut remove = Vec::new();
            for (&label, (target, transition_weight)) in &mut rewritten {
                if let Some(detail) = detail.as_mut() {
                    detail.outgoing_transitions_scanned += 1;
                }
                let input_target = *target;
                if transition_weight.is_empty() {
                    remove.push(label);
                    continue;
                }
                if let Some(detail) = detail.as_mut() {
                    detail.nonempty_intersections += 1;
                }
                let to_state = if fixed_singleton_ids {
                    let scheduled = fixed_singleton_scheduled.as_mut().unwrap();
                    if !scheduled[input_target as usize] {
                        scheduled[input_target as usize] = true;
                        worklist.push_back((
                            input_target,
                            vec![(input_target, normalized_singleton_weight.clone())],
                        ));
                    }
                    input_target
                } else if let Some(existing) =
                    normalized_singleton_subsets.get(&input_target).copied()
                {
                    if let Some(detail) = detail.as_mut() {
                        detail.subset_intern_hits += 1;
                    }
                    existing
                } else {
                    if let Some(detail) = detail.as_mut() {
                        detail.subset_intern_misses += 1;
                    }
                    let new_state = result.add_state();
                    subset_map.insert(vec![(input_target, normalized_singleton_key)], new_state);
                    normalized_singleton_subsets.insert(input_target, new_state);
                    worklist.push_back((
                        new_state,
                        vec![(input_target, normalized_singleton_weight.clone())],
                    ));
                    new_state
                };
                *target = to_state;
            }
            for label in remove {
                rewritten.remove(&label);
            }
            result.states_mut()[from_state as usize].transitions = rewritten;
            continue;
        }

        for (dwa_state_id, path_weight) in &subset_entries {
            let state = &dwa.states()[*dwa_state_id as usize];

            for (&label, (target, transition_weight)) in &state.transitions {
                if label == DEFAULT_LABEL {
                    continue;
                }
                if let Some(detail) = detail.as_mut() {
                    detail.outgoing_transitions_scanned += 1;
                    detail.intersection_calls += 1;
                }
                let next_weight =
                    intersection_cache.intersection(path_weight, transition_weight);
                if next_weight.is_empty() {
                    continue;
                }
                if let Some(detail) = detail.as_mut() {
                    detail.nonempty_intersections += 1;
                }

                let target_weights = if label >= 0 && (label as usize) < dense_label_limit {
                    let label_idx = label as usize;
                    if !dense_label_touched[label_idx] {
                        dense_label_touched[label_idx] = true;
                        touched_dense_labels.push(label_idx);
                    }
                    &mut dense_raw_targets[label_idx]
                } else {
                    sparse_raw_targets.entry(label).or_default()
                };
                add_target_contribution_profiled(target_weights, *target, next_weight, detail.as_mut());
            }

            let Some((default_target, default_weight)) = state.transitions.get(&DEFAULT_LABEL) else {
                continue;
            };

            if let Some(detail) = detail.as_mut() {
                detail.outgoing_transitions_scanned += 1;
                detail.intersection_calls += 1;
            }
            let fallback_weight = intersection_cache.intersection(path_weight, default_weight);
            if fallback_weight.is_empty() {
                continue;
            }
            if let Some(detail) = detail.as_mut() {
                detail.nonempty_intersections += 1;
            }

            default_touched = true;
            add_target_contribution_profiled(
                &mut default_raw_targets,
                *default_target,
                fallback_weight.clone(),
                detail.as_mut(),
            );

            for &label in state.transitions.keys() {
                if label == DEFAULT_LABEL {
                    continue;
                }
                if let Some(detail) = detail.as_mut() {
                    detail.fallback_labels_expanded += 1;
                    detail.fallback_contrib_entries_duplicated += 1;
                }
                if label >= 0 && (label as usize) < dense_label_limit {
                    let label_idx = label as usize;
                    if !dense_label_touched[label_idx] {
                        dense_label_touched[label_idx] = true;
                        touched_dense_labels.push(label_idx);
                    }
                    let target_weights = &mut dense_raw_targets[label_idx];
                    add_target_contribution_profiled(
                        target_weights,
                        *default_target,
                        fallback_weight.clone(),
                        detail.as_mut(),
                    );
                } else {
                    let target_weights = sparse_raw_targets.entry(label).or_default();
                    add_target_contribution_profiled(
                        target_weights,
                        *default_target,
                        fallback_weight.clone(),
                        detail.as_mut(),
                    );
                }
            }

            match possible_by_state.get(*dwa_state_id as usize) {
                Some(PossibleOutgoingIds::All) => {
                    if let Some(detail) = detail.as_mut() {
                        detail.fallback_labels_expanded += dense_label_limit;
                        detail.fallback_contrib_entries_duplicated += 1;
                    }
                    add_target_contribution_profiled(
                        &mut dense_default_all_raw_targets,
                        *default_target,
                        fallback_weight.clone(),
                        detail.as_mut(),
                    );
                }
                Some(PossibleOutgoingIds::Some(ids)) => {
                    for parser_state_id in ids.iter_ones() {
                        if let Some(detail) = detail.as_mut() {
                            detail.fallback_labels_expanded += 1;
                            detail.fallback_contrib_entries_duplicated += 1;
                        }
                        let label_idx = parser_state_id;
                        if !dense_label_touched[label_idx] {
                            dense_label_touched[label_idx] = true;
                            touched_dense_labels.push(label_idx);
                        }
                        let target_weights = &mut dense_raw_targets[label_idx];
                        add_target_contribution_profiled(
                            target_weights,
                            *default_target,
                            fallback_weight.clone(),
                            detail.as_mut(),
                        );
                    }
                }
                Some(PossibleOutgoingIds::Empty) | None => {}
            }
        }
        if let (Some(detail), Some(started_at)) = (detail.as_mut(), scan_started) {
            detail.intersection_scan_ms += elapsed_ms(started_at);
        }

        let label_started = detail.as_ref().map(|_| Instant::now());
        let mut process_label = |label: i32, mut contribs: TargetContribs| {
            if contribs.is_empty() {
                return;
            }

            debug_assert!(contribs.iter().all(|(_, weight)| !weight.is_empty()));
            contribs.sort_unstable_by_key(|(state_id, _)| *state_id);

            let edge_weight = Weight::union_all(contribs.iter().map(|(_, weight)| weight));
            if edge_weight.is_empty() {
                return;
            }

            let to_state = if let [(only_state, only_weight)] = contribs.as_slice() {
                if normalize_singletons {
                    if fixed_singleton_ids {
                        let scheduled = fixed_singleton_scheduled.as_mut().unwrap();
                        if !scheduled[*only_state as usize] {
                            scheduled[*only_state as usize] = true;
                            worklist.push_back((
                                *only_state,
                                vec![(*only_state, normalized_singleton_weight.clone())],
                            ));
                        }
                        *only_state
                    } else if let Some(existing) =
                        normalized_singleton_subsets.get(only_state).copied()
                    {
                        if let Some(detail) = detail.as_mut() {
                            detail.subset_intern_hits += 1;
                        }
                        existing
                    } else {
                        if let Some(detail) = detail.as_mut() {
                            detail.subset_intern_misses += 1;
                        }
                        let new_state = result.add_state();
                        subset_map.insert(vec![(*only_state, normalized_singleton_key)], new_state);
                        normalized_singleton_subsets.insert(*only_state, new_state);
                        worklist.push_back((
                            new_state,
                            vec![(*only_state, normalized_singleton_weight.clone())],
                        ));
                        new_state
                    }
                } else {
                    let singleton_key = (*only_state, only_weight.ptr_key());
                    if let Some(existing) = weighted_singleton_subsets.get(&singleton_key).copied() {
                        if let Some(detail) = detail.as_mut() {
                            detail.subset_intern_hits += 1;
                        }
                        existing
                    } else {
                        if let Some(detail) = detail.as_mut() {
                            detail.subset_intern_misses += 1;
                        }
                        let new_state = result.add_state();
                        subset_map.insert(vec![singleton_key], new_state);
                        weighted_singleton_subsets.insert(singleton_key, new_state);
                        worklist.push_back((new_state, contribs.into_iter().collect()));
                        new_state
                    }
                }
            } else {
                key_buf.clear();
                key_buf.extend(contribs.iter().map(|(sid, w)| (*sid, w.ptr_key())));
                if let Some(detail) = detail.as_mut() {
                    detail.subset_key_constructions += 1;
                }
                if let Some(existing) = subset_map.get(&key_buf).copied() {
                    if let Some(detail) = detail.as_mut() {
                        detail.subset_intern_hits += 1;
                    }
                    existing
                } else {
                    if let Some(detail) = detail.as_mut() {
                        detail.subset_intern_misses += 1;
                    }
                    let new_state = result.add_state();
                    subset_map.insert(key_buf.clone(), new_state);
                    let next_entries: Vec<(u32, Weight)> = contribs.into_iter().collect();
                    worklist.push_back((new_state, next_entries));
                    new_state
                }
            };

            result.add_transition(from_state, label, to_state, edge_weight);
        };

        for label_idx in touched_dense_labels.drain(..) {
            dense_label_touched[label_idx] = false;
            if !dense_default_all_raw_targets.is_empty() {
                extend_target_contribs(
                    &mut dense_raw_targets[label_idx],
                    &dense_default_all_raw_targets,
                );
            }
            process_label(
                label_idx as i32,
                std::mem::take(&mut dense_raw_targets[label_idx]),
            );
        }
        if default_touched {
            default_touched = false;
            process_label(DEFAULT_LABEL, std::mem::take(&mut default_raw_targets));
        }
        for (label, contribs) in sparse_raw_targets.drain() {
            process_label(label, contribs);
        }
        if let (Some(detail), Some(started_at)) = (detail.as_mut(), label_started) {
            detail.label_processing_ms += elapsed_ms(started_at);
        }
    }

    if let Some(started) = fallback_started {
        eprintln!(
            "[glrmask/profile][fallback_fast] input_states={} output_states={} parallel_waves={} parallel_rows={} total_ms={:.3}",
            dwa.states().len(),
            result.states().len(),
            fallback_parallel_waves,
            fallback_parallel_rows,
            elapsed_ms(started),
        );
    }
    if let Some(detail) = detail {
        detail.emit("fallback");
    }

    result
}

fn determinize_parser_dwa_with_fallbacks(
    dwa: &DWA,
    possible_by_state: &[PossibleOutgoingIds],
    num_parser_states: u32,
) -> DWA {
    determinize_parser_dwa_with_fallbacks_impl(
        dwa,
        possible_by_state,
        num_parser_states,
        true,
    )
}

fn optimize_parser_dwa_defaults(
    dwa: &mut DWA,
    possible_by_state: &[PossibleOutgoingIds],
    num_parser_states: u32,
) {
    loop {
        let mut changed = false;

        for (state_id, possible_ids) in possible_by_state.iter().enumerate() {
            let possible_count = match possible_ids {
                PossibleOutgoingIds::Empty => 0,
                PossibleOutgoingIds::All => num_parser_states as usize,
                PossibleOutgoingIds::Some(ids) => ids.count_ones(),
            };
            if possible_count < 2 {
                continue;
            }

            let state = &dwa.states()[state_id];

            let mut actual_positive = BitSet::new(num_parser_states as usize);
            for &label in state.transitions.keys() {
                if let Some(ps) = parser_state_label(label, num_parser_states) {
                    actual_positive.set(ps as usize);
                }
            }
            match possible_ids {
                PossibleOutgoingIds::Empty => continue,
                PossibleOutgoingIds::All => {
                    if actual_positive.count_ones() != num_parser_states as usize {
                        continue;
                    }
                }
                PossibleOutgoingIds::Some(ids) => {
                    if actual_positive != *ids {
                        continue;
                    }
                }
            }

            let mut shared_target: Option<u32> = None;
            let mut default_weight: Option<Weight> = None;
            let mut valid = true;

            match possible_ids {
                PossibleOutgoingIds::Empty => continue,
                PossibleOutgoingIds::All => {
                    for ps in 0..num_parser_states {
                        let label = ps as i32;
                        let Some((target, weight)) = state.transitions.get(&label) else {
                            valid = false;
                            break;
                        };
                        match shared_target {
                            Some(existing) if existing != *target => {
                                valid = false;
                                break;
                            }
                            None => shared_target = Some(*target),
                            _ => {}
                        }
                        default_weight = Some(match default_weight {
                            Some(existing) => existing.intersection(weight),
                            None => weight.clone(),
                        });
                    }
                }
                PossibleOutgoingIds::Some(ids) => {
                    for ps in ids.iter_ones() {
                        let label = ps as i32;
                        let Some((target, weight)) = state.transitions.get(&label) else {
                            valid = false;
                            break;
                        };
                        match shared_target {
                            Some(existing) if existing != *target => {
                                valid = false;
                                break;
                            }
                            None => shared_target = Some(*target),
                            _ => {}
                        }
                        default_weight = Some(match default_weight {
                            Some(existing) => existing.intersection(weight),
                            None => weight.clone(),
                        });
                    }
                }
            }

            let Some(target) = shared_target else { continue };
            let Some(default_weight) = default_weight else { continue };
            if !valid || default_weight.is_empty() {
                continue;
            }

            let state = &mut dwa.states_mut()[state_id];
            let entry = state.transitions.entry(DEFAULT_LABEL);
            match entry {
                std::collections::btree_map::Entry::Occupied(mut occ) => {
                    let (existing_target, existing_weight) = occ.get_mut();
                    if *existing_target == target {
                        let updated = existing_weight.union(&default_weight);
                        if updated != *existing_weight {
                            *existing_weight = updated;
                            changed = true;
                        }
                    }
                }
                std::collections::btree_map::Entry::Vacant(vac) => {
                    vac.insert((target, default_weight));
                    changed = true;
                }
            }
        }

        for state_id in 0..dwa.states().len() {
            let Some((default_target, default_weight)) =
                dwa.states()[state_id].transitions.get(&DEFAULT_LABEL).cloned()
            else {
                continue;
            };

            let target_final = dwa.states()[default_target as usize].final_weight.clone();
            let Some(target_final) = target_final else { continue };
            let lifted = default_weight.intersection(&target_final);
            if lifted.is_empty() {
                continue;
            }

            if union_final_weight(&mut dwa.states_mut()[state_id].final_weight, lifted.clone()) {
                changed = true;
            }

            let state = &mut dwa.states_mut()[state_id];
            let mut to_remove = Vec::new();
            for (&label, (_, weight)) in state.transitions.iter_mut() {
                let new_weight = weight.difference(&lifted);
                if new_weight != *weight {
                    *weight = new_weight;
                    changed = true;
                }
                if weight.is_empty() {
                    to_remove.push(label);
                }
            }
            for label in to_remove {
                state.transitions.remove(&label);
            }
        }

        for state_id in 0..dwa.states().len() {
            let Some(&(default_target, ref default_weight)) =
                dwa.states()[state_id].transitions.get(&DEFAULT_LABEL)
            else {
                continue;
            };
            let default_target = default_target;
            let default_weight = default_weight.clone();

            let state = &mut dwa.states_mut()[state_id];
            let mut to_remove = Vec::new();
            for (&label, (target, weight)) in state.transitions.iter_mut() {
                if label == DEFAULT_LABEL {
                    continue;
                }
                if *target != default_target {
                    continue;
                }
                let new_weight = weight.difference(&default_weight);
                if new_weight != *weight {
                    *weight = new_weight;
                    changed = true;
                }
                if weight.is_empty() {
                    to_remove.push(label);
                }
            }
            for label in to_remove {
                state.transitions.remove(&label);
            }
        }

        if !changed {
            break;
        }
    }
}

fn subtract_final_weights_from_outgoing_dwa_impl(dwa: &mut DWA, parallel: bool) {
    if std::env::var_os("GLRMASK_PROFILE_FINAL_SUBTRACTION_DETAIL").is_some() {
        let mut pairs = FxHashSet::<(usize, usize)>::default();
        let mut calls = 0usize;
        let mut states_with_final = 0usize;
        let mut final_weights = FxHashSet::<usize>::default();
        let mut edge_weights = FxHashSet::<usize>::default();
        for state in dwa.states() {
            let Some(final_weight) = state.final_weight.as_ref() else {
                continue;
            };
            if final_weight.is_empty() {
                continue;
            }
            states_with_final += 1;
            final_weights.insert(final_weight.ptr_key());
            for (_, weight) in state.transitions.values() {
                calls += 1;
                edge_weights.insert(weight.ptr_key());
                pairs.insert((weight.ptr_key(), final_weight.ptr_key()));
            }
        }
        eprintln!(
            "[glrmask/profile][final_subtraction_detail] states={} states_with_final={} calls={} unique_pairs={} pair_reuse={} unique_final_weights={} unique_edge_weights={}",
            dwa.states().len(),
            states_with_final,
            calls,
            pairs.len(),
            calls.saturating_sub(pairs.len()),
            final_weights.len(),
            edge_weights.len(),
        );
    }
    if parallel {
        use rayon::prelude::*;

        let global_pair_cache = dwa.states().len()
            >= std::env::var("GLRMASK_FINAL_SUBTRACTION_GLOBAL_CACHE_MIN_STATES")
                .ok()
                .and_then(|value| value.trim().parse::<usize>().ok())
                .unwrap_or(16_384)
            && std::env::var_os("GLRMASK_DISABLE_FINAL_SUBTRACTION_GLOBAL_CACHE").is_none();
        if global_pair_cache {
            // The same edge/final weight pair recurs across many states. Build
            // the exact difference once per live pointer pair, in parallel, then
            // make the multi-million-edge rewrite a lookup-only pass. All operand
            // Arcs remain live for this scope, so pointer identity is stable.
            let mut unique_pairs =
                FxHashMap::<(usize, usize), (Weight, Weight)>::default();
            for state in dwa.states() {
                let Some(final_weight) = state.final_weight.as_ref() else {
                    continue;
                };
                if final_weight.is_empty() {
                    continue;
                }
                for (_, edge_weight) in state.transitions.values() {
                    unique_pairs
                        .entry((edge_weight.ptr_key(), final_weight.ptr_key()))
                        .or_insert_with(|| (edge_weight.clone(), final_weight.clone()));
                }
            }
            let differences = unique_pairs
                .into_par_iter()
                .map(|(key, (edge_weight, final_weight))| {
                    (key, edge_weight.difference(&final_weight))
                })
                .collect::<FxHashMap<_, _>>();
            dwa.states_mut().par_iter_mut().for_each(|state| {
                let Some(final_weight) = state.final_weight.as_ref() else {
                    return;
                };
                if final_weight.is_empty() {
                    return;
                }
                let final_key = final_weight.ptr_key();
                state.transitions.retain(|_, (_, weight)| {
                    let key = (weight.ptr_key(), final_key);
                    let new_weight = differences
                        .get(&key)
                        .expect("final-subtraction difference pair was not precomputed");
                    if new_weight != weight {
                        *weight = new_weight.clone();
                    }
                    !weight.is_empty()
                });
            });
            return;
        }

        dwa.states_mut().par_iter_mut().for_each_init(
            ScopedWeightOpCache::default,
            |weight_ops, state| {
                let Some(final_weight) = state.final_weight.clone() else {
                    return;
                };
                if final_weight.is_empty() {
                    return;
                }
                state.transitions.retain(|_, (_, weight)| {
                    let new_weight = weight_ops.difference(weight, &final_weight);
                    if new_weight != *weight {
                        *weight = new_weight;
                    }
                    !weight.is_empty()
                });
            },
        );
        return;
    }

    let mut weight_ops = ScopedWeightOpCache::default();
    for state_id in 0..dwa.states().len() {
        let Some(final_weight) = dwa.states()[state_id].final_weight.clone() else {
            continue;
        };
        if final_weight.is_empty() {
            continue;
        }
        let state = &mut dwa.states_mut()[state_id];
        let mut to_remove = Vec::new();
        for (&label, (_, weight)) in state.transitions.iter_mut() {
            let new_weight = weight_ops.difference(weight, &final_weight);
            if new_weight != *weight {
                *weight = new_weight;
            }
            if weight.is_empty() {
                to_remove.push(label);
            }
        }
        for label in to_remove {
            state.transitions.remove(&label);
        }
    }
}

fn subtract_final_weights_from_outgoing_dwa(dwa: &mut DWA) {
    subtract_final_weights_from_outgoing_dwa_impl(dwa, rayon::current_num_threads() > 1);
}

fn dwa_to_nwa(dwa: &DWA) -> NWA {
    let mut nwa = NWA::new(0, 0);
    *nwa.states_mut() = vec![crate::automata::weighted::nwa::NWAState::default(); dwa.states().len()];
    nwa.set_start_states(vec![dwa.start_state()]);

    for (state_id, state) in dwa.states().iter().enumerate() {
        if let Some(final_weight) = state.final_weight.clone() {
            nwa.states_mut()[state_id].final_weight = Some(final_weight);
        }
        for (&label, (target, weight)) in &state.transitions {
            nwa.states_mut()[state_id]
                .transitions
                .entry(label)
                .or_default()
                .push((*target, weight.clone()));
        }
    }

    nwa
}

fn compute_productive_terminal_states(summaries: &StateSummaries) -> Vec<bool> {
    let states = &summaries.states;
    let mut reverse_edges: Vec<Vec<u32>> = vec![Vec::new(); states.len()];
    let mut productive = vec![false; states.len()];
    let mut worklist = VecDeque::new();

    for (state_id, state) in states.iter().enumerate() {
        if state
            .final_weight
            .as_ref()
            .is_some_and(|weight| !weight.is_empty())
        {
            productive[state_id] = true;
            worklist.push_back(state_id as u32);
        }

        for (target, weight) in &state.epsilon_branches {
            if !weight.is_empty() && (*target as usize) < states.len() {
                reverse_edges[*target as usize].push(state_id as u32);
            }
        }

        for branch in &state.branches {
            if (branch.target as usize) < states.len()
                && summaries
                    .bundle_accepts
                    .get(branch.bundle_id)
                    .copied()
                    .unwrap_or(false)
            {
                reverse_edges[branch.target as usize].push(state_id as u32);
            }
        }
    }

    while let Some(target) = worklist.pop_front() {
        for &source in &reverse_edges[target as usize] {
            let source_idx = source as usize;
            if !productive[source_idx] {
                productive[source_idx] = true;
                worklist.push_back(source);
            }
        }
    }

    productive
}

fn append_weighted_template_redirecting_finals(
    arena: &mut NWA,
    template: &NWA,
    weight: &Weight,
    continuation_state: u32,
) -> NwaBody {
    if std::env::var_os("GLRMASK_EXPERIMENT_BULK_FRAGMENT_APPEND").is_some() {
        let offset = arena.states().len() as u32;
        let starts = template
            .start_states()
            .iter()
            .map(|state| offset + *state)
            .collect::<Vec<_>>();
        let states = arena.states_mut();
        states.reserve(template.states().len());
        for source in template.states() {
            let mut appended = source.clone();
            for targets in appended.transitions.values_mut() {
                for (target, edge_weight) in targets {
                    *target += offset;
                    *edge_weight = weight.clone();
                }
            }
            for (target, epsilon_weight) in &mut appended.epsilons {
                *target += offset;
                *epsilon_weight = weight.clone();
            }
            if appended.final_weight.take().is_some() {
                appended.epsilons.push((continuation_state, weight.clone()));
            }
            states.push(appended);
        }
        return NwaBody { start_states: starts };
    }

    let offset = arena.states().len() as u32;
    let body = arena.append_with_body(template);
    let appended_len = template.states().len();

    for state_id in offset as usize..offset as usize + appended_len {
        let state = &mut arena.states_mut()[state_id];
        for targets in state.transitions.values_mut() {
            for (_, edge_weight) in targets {
                *edge_weight = weight.clone();
            }
        }
        for (_, epsilon_weight) in &mut state.epsilons {
            *epsilon_weight = weight.clone();
        }
    }

    for state_id in offset as usize..offset as usize + appended_len {
        if arena.states_mut()[state_id].final_weight.take().is_some() {
            arena.add_epsilon(state_id as u32, continuation_state, weight.clone());
        }
    }

    body
}

/// Internal compiler splice for an already weighted deterministic terminal bundle.
#[doc(hidden)]
pub fn append_bundle_redirecting_finals(
    arena: &mut NWA,
    bundle: &NWA,
    continuation_state: u32,
) -> NwaBody {
    if std::env::var_os("GLRMASK_EXPERIMENT_BULK_FRAGMENT_APPEND").is_some() {
        let offset = arena.states().len() as u32;
        let starts = bundle
            .start_states()
            .iter()
            .map(|state| offset + *state)
            .collect::<Vec<_>>();
        let states = arena.states_mut();
        states.reserve(bundle.states().len());
        for source in bundle.states() {
            let mut appended = source.clone();
            for targets in appended.transitions.values_mut() {
                for (target, _) in targets {
                    *target += offset;
                }
            }
            for (target, _) in &mut appended.epsilons {
                *target += offset;
            }
            if let Some(final_weight) = appended.final_weight.take()
                && !final_weight.is_empty()
            {
                appended.epsilons.push((continuation_state, final_weight));
            }
            states.push(appended);
        }
        return NwaBody { start_states: starts };
    }

    let offset = arena.states().len() as u32;
    let body = arena.append_with_body(bundle);
    let appended_len = bundle.states().len();

    for state_id in offset as usize..offset as usize + appended_len {
        let Some(final_weight) = arena.states_mut()[state_id].final_weight.take() else {
            continue;
        };
        if !final_weight.is_empty() {
            arena.add_epsilon(state_id as u32, continuation_state, final_weight);
        }
    }

    body
}

fn append_branch_fragment(
    arena: &mut NWA,
    summaries: &StateSummaries,
    templates: &Templates,
    built_bundle_cache: &mut [Option<Arc<NWA>>],
    bundle_id: usize,
    continuation_state: u32,
    preserve_bundle_nondeterminism: bool,
    compose_detail: Option<&mut ParserDwaComposeDetailProfile>,
) -> Option<NwaBody> {
    let bundle = summaries.unique_bundles.get(bundle_id)?;
    if !summaries.bundle_accepts.get(bundle_id).copied().unwrap_or(false) {
        return None;
    }

    if bundle.len() == 1 {
        let (&terminal, weight) = bundle.iter().next().expect("len checked");
        if weight.is_empty() {
            return None;
        }
        let template = templates.by_terminal_nwa.get(&terminal)?;
        return Some(append_weighted_template_redirecting_finals(
            arena,
            template,
            weight,
            continuation_state,
        ));
    }

    // The ordinary parser compiler determinizes every multi-terminal bundle so
    // the subsequent negative-code cancellation step sees one stable local
    // relation. The optional fast path below is *compile-time only*: it keeps
    // the exact union of weighted terminal-template NWAs nondeterministic while
    // this larger parser NWA is assembled, avoiding a local DFA product that is
    // immediately converted back into an NWA. Callers using this path MUST run
    // the ordinary negative-code resolution before publishing the automaton to
    // runtime. Runtime parser automata must contain no negative/PUSH labels.
    if preserve_bundle_nondeterminism {
        let mut starts = Vec::new();
        for (&terminal, weight) in bundle {
            if weight.is_empty() {
                continue;
            }
            let template = templates.by_terminal_nwa.get(&terminal)?;
            let body = append_weighted_template_redirecting_finals(
                arena,
                template,
                weight,
                continuation_state,
            );
            starts.extend(body.start_states);
        }
        starts.sort_unstable();
        starts.dedup();
        return (!starts.is_empty()).then_some(NwaBody { start_states: starts });
    }

    // STICKY NOTE: keep parser bundles eagerly determinized here.
    //
    // It is tempting to leave multi-terminal bundles nondeterministic or factored so
    // this stage can avoid a large deterministic bundle build. Do not do that. These
    // bundles are the unit on which downstream negative-resolution operates. If a
    // bundle is left nondeterministic, negative-resolution has to distribute one
    // bundle alternatives against the next bundle alternatives, which recreates
    // the same cross-product later and can become a combinatorial explosion between
    // adjacent bundles. Eager determinization pays that cost once, locally, and gives
    // negative-resolution a stable deterministic object to compose.
    //
    // NEVER remove this note without replacing it with an equally explicit invariant
    // explaining why parser-bundle determinization is required. We have repeatedly
    // rediscovered this and incorrectly proposed removing determinization. If the
    // first multi-terminal bundle cannot be determinized, the fix is to reduce the
    // bundle/grammar/compiler state space, not to pass a nondeterministic bundle
    // downstream.
    if built_bundle_cache[bundle_id].is_none() {
        if let Some(detail) = compose_detail {
            let (bundle_nwa, bundle_profile) = templates.build_bundle_profiled(bundle);
            detail.bundle_profile_total_ms += bundle_profile.total_ms;
            detail.bundle_profile_build_group_dfas_ms += bundle_profile.build_group_dfas_ms;
            detail.bundle_profile_union_groups_ms += bundle_profile.union_groups_ms;
            detail.bundle_profile_determinize_ms += bundle_profile.determinize_bundle_ms;
            detail.bundle_profile_minimize_ms += bundle_profile.minimize_ms;
            detail.bundle_profile_dwa_to_nwa_ms += bundle_profile.dwa_to_nwa_ms;
            detail.bundle_profile_result_dwa_states += bundle_profile.result_dwa_states;
            detail.bundle_profile_result_dwa_transitions += bundle_profile.result_dwa_transitions;
            detail.bundle_profile_result_nwa_states += bundle_profile.result_nwa_states;
            detail.bundle_profile_result_nwa_transitions += bundle_profile.result_nwa_transitions;
            eprintln!(
                "[glrmask/profile][parser_bundle] bundle_id={} terminals={} weight_groups={} single_entry_weights={} single_tsid_weights={} total_weight_outer_ranges={} build_group_dfas_ms={:.3} union_groups_ms={:.3} determinize_bundle_ms={:.3} det_pop_ms={:.3} det_alive_ms={:.3} det_final_ms={:.3} det_collect_labels_ms={:.3} det_next_state_ms={:.3} det_edge_weight_ms={:.3} det_lookup_ms={:.3} det_add_transition_ms={:.3} det_states={} det_labels={} det_transitions={} det_edge_subset_total={} det_edge_subset_max={} det_edge_cache_hits={} det_edge_cache_misses={} minimize_ms={:.3} minimize_skipped={} dwa_to_nwa_ms={:.3} total_ms={:.3} result_dwa_states={} result_dwa_transitions={} result_nwa_states={} result_nwa_transitions={}",
                bundle_id,
                bundle_profile.input_terminals,
                bundle_profile.weight_groups,
                bundle_profile.single_entry_weights,
                bundle_profile.single_tsid_weights,
                bundle_profile.total_weight_outer_ranges,
                bundle_profile.build_group_dfas_ms,
                bundle_profile.union_groups_ms,
                bundle_profile.determinize_bundle_ms,
                bundle_profile.determinize_pop_state_ms,
                bundle_profile.determinize_alive_groups_ms,
                bundle_profile.determinize_final_weight_ms,
                bundle_profile.determinize_collect_labels_ms,
                bundle_profile.determinize_next_state_ms,
                bundle_profile.determinize_edge_weight_ms,
                bundle_profile.determinize_state_lookup_ms,
                bundle_profile.determinize_add_transition_ms,
                bundle_profile.determinize_states_visited,
                bundle_profile.determinize_labels_processed,
                bundle_profile.determinize_transitions_added,
                bundle_profile.determinize_edge_subset_total,
                bundle_profile.determinize_edge_subset_max,
                bundle_profile.determinize_edge_cache_hits,
                bundle_profile.determinize_edge_cache_misses,
                bundle_profile.minimize_ms,
                bundle_profile.minimize_skipped,
                bundle_profile.dwa_to_nwa_ms,
                bundle_profile.total_ms,
                bundle_profile.result_dwa_states,
                bundle_profile.result_dwa_transitions,
                bundle_profile.result_nwa_states,
                bundle_profile.result_nwa_transitions,
            );
            built_bundle_cache[bundle_id] = Some(Arc::new(bundle_nwa));
        } else {
            built_bundle_cache[bundle_id] = Some(Arc::new(templates.build_bundle(bundle)));
        }
    }
    let bundle_nwa = built_bundle_cache[bundle_id]
        .as_ref()
        .expect("bundle cache entry just initialized");
    Some(append_bundle_redirecting_finals(
        arena,
        bundle_nwa.as_ref(),
        continuation_state,
    ))
}

fn build_parser_nwa_from_terminal_dwa(
    terminal_dwa: &TerminalAutomaton,
    grammar: &AnalyzedGrammar,
    templates: &Templates,
    table: &GLRTable,
) -> Option<(NWA, ParserNwaBuildProfile)> {
    build_parser_nwa_from_terminal_dwa_for_terminal_count(
        terminal_dwa,
        grammar.num_terminals,
        templates,
        Some(table),
        false,
        None,
        true,
    )
}

fn build_parser_nwa_from_terminal_dwa_for_terminal_count(
    terminal_dwa: &TerminalAutomaton,
    num_terminals: u32,
    templates: &Templates,
    table: Option<&GLRTable>,
    preserve_bundle_nondeterminism: bool,
    prebuilt_bundle_cache: Option<&PrebuiltParserBundleCache>,
    allow_parallel: bool,
) -> Option<(NWA, ParserNwaBuildProfile)> {
    let total_started_at = Instant::now();
    let state_prep_started_at = Instant::now();
    let summaries = build_state_summaries(terminal_dwa, num_terminals, templates);
    let productive = compute_productive_terminal_states(&summaries);
    let state_prep_ms = elapsed_ms(state_prep_started_at);
    let states = &summaries.states;
    if std::env::var_os("GLRMASK_PROFILE_FUTURE_TEMPLATE_PRUNE").is_some()
        && let Some(table) = table
    {
        let mut contexts = 0usize;
        let mut skipped_epsilon = 0usize;
        let mut future_terminal_total = 0usize;
        let mut allowed_top_total = 0usize;
        let mut accepting_negative_edges = 0usize;
        let mut future_invalid_edges = 0usize;
        let mut fully_prunable_weighted_edges = 0usize;
        let mut restricted_weighted_edges = 0usize;
        let mut unchanged_weighted_edges = 0usize;
        let mut current_terminal_refs = 0usize;
        let mut original_outer_ranges = 0usize;
        let mut restricted_outer_ranges = 0usize;
        for state in states {
            for branch in &state.branches {
                let Some(target) = states.get(branch.target as usize) else { continue };
                if !target.epsilon_branches.is_empty() {
                    skipped_epsilon += 1;
                    continue;
                }
                let mut future_terminals = std::collections::BTreeSet::<TerminalID>::new();
                for future_branch in &target.branches {
                    if let Some(bundle) = summaries.unique_bundles.get(future_branch.bundle_id) {
                        future_terminals.extend(bundle.keys().copied());
                    }
                }
                if future_terminals.is_empty() {
                    continue;
                }
                contexts += 1;
                future_terminal_total += future_terminals.len();
                let mut allowed_tops = vec![false; table.num_states as usize];
                for top in 0..table.num_states {
                    allowed_tops[top as usize] = future_terminals
                        .iter()
                        .any(|&terminal| table.action(top, terminal).is_some());
                }
                allowed_top_total += allowed_tops.iter().filter(|&&allowed| allowed).count();
                let Some(current_bundle) = summaries.unique_bundles.get(branch.bundle_id) else {
                    continue;
                };
                current_terminal_refs += current_bundle.len();
                for (&terminal, branch_weight) in current_bundle {
                    let Some(dfa) = templates.by_terminal.get(&terminal) else { continue };
                    for dfa_state in &dfa.states {
                        for (&label, &target_state) in &dfa_state.transitions {
                            if !is_negative_label(label)
                                || !dfa.states.get(target_state as usize).is_some_and(|state| state.is_accepting)
                            {
                                continue;
                            }
                            accepting_negative_edges += 1;
                            let top = negative_to_positive_label(label) as usize;
                            if top < allowed_tops.len() && allowed_tops[top] {
                                continue;
                            }
                            future_invalid_edges += 1;
                            original_outer_ranges += branch_weight.outer_range_count();
                            let restricted = match target.final_weight.as_ref() {
                                Some(final_weight) => branch_weight.intersection(final_weight),
                                None => Weight::empty(),
                            };
                            restricted_outer_ranges += restricted.outer_range_count();
                            if restricted.is_empty() {
                                fully_prunable_weighted_edges += 1;
                            } else if restricted == *branch_weight {
                                unchanged_weighted_edges += 1;
                            } else {
                                restricted_weighted_edges += 1;
                            }
                        }
                    }
                }
            }
        }
        eprintln!(
            "[glrmask/profile][future_template_prune_potential] contexts={} skipped_epsilon={} current_terminal_refs={} avg_future_terminals={:.2} avg_allowed_tops={:.2} parser_states={} accepting_negative_edges={} future_invalid_edges={} fully_prunable_weighted_edges={} restricted_weighted_edges={} unchanged_weighted_edges={} original_outer_ranges={} restricted_outer_ranges={}",
            contexts,
            skipped_epsilon,
            current_terminal_refs,
            if contexts == 0 { 0.0 } else { future_terminal_total as f64 / contexts as f64 },
            if contexts == 0 { 0.0 } else { allowed_top_total as f64 / contexts as f64 },
            table.num_states,
            accepting_negative_edges,
            future_invalid_edges,
            fully_prunable_weighted_edges,
            restricted_weighted_edges,
            unchanged_weighted_edges,
            original_outer_ranges,
            restricted_outer_ranges,
        );
    }
    let compose_detail_enabled = parser_dwa_compose_detail_enabled();
    let hybrid_nondeterministic_min_terminals = std::env::var(
        "GLRMASK_COMPILE_NONDETERMINISTIC_BUNDLE_MIN_TERMINALS",
    )
    .ok()
    .and_then(|value| value.trim().parse::<usize>().ok())
    .filter(|&value| value >= 2);
    let preserve_bundle = |len: usize| {
        preserve_bundle_nondeterminism
            || hybrid_nondeterministic_min_terminals.is_some_and(|min| len >= min)
    };
    let mut compose_detail = ParserDwaComposeDetailProfile {
        total_states: states.len(),
        productive_states: productive.iter().filter(|&&is_productive| is_productive).count(),
        total_branches: states
            .iter()
            .map(|state| state.epsilon_branches.len() + state.branches.len())
            .sum(),
        productive_branches: 0,
        unique_bundles: summaries.unique_bundles.len(),
        accepting_bundles: summaries.bundle_accepts.iter().filter(|&&accepts| accepts).count(),
        ..ParserDwaComposeDetailProfile::default()
    };

    let productive_start_states: Vec<u32> = summaries
        .start_states
        .iter()
        .copied()
        .filter(|state| productive.get(*state as usize).copied().unwrap_or(false))
        .collect();
    if productive_start_states.is_empty() {
        return None;
    }

    let graph_started_at = Instant::now();
    let mut arena = NWA::new(0, 0);
    let mut continuation_states = vec![u32::MAX; states.len()];

    let state_init_started_at = Instant::now();
    for (state_id, state) in states.iter().enumerate() {
        if !productive[state_id] {
            continue;
        }
        let continuation_state = arena.add_state();
        continuation_states[state_id] = continuation_state;
        if let Some(final_weight) = state
            .final_weight
            .as_ref()
            .filter(|weight| !weight.is_empty())
        {
            arena.set_final_weight(continuation_state, final_weight.clone());
        }
    }
    compose_detail.state_init_ms = elapsed_ms(state_init_started_at);

    let mut branch_fragment_memo: FxHashMap<(usize, u32), NwaBody> = FxHashMap::default();
    let mut used_multi_bundle = vec![false; summaries.unique_bundles.len()];
    for (state_id, state) in states.iter().enumerate() {
        if !productive[state_id] {
            continue;
        }
        for branch in &state.branches {
            let target_idx = branch.target as usize;
            if productive.get(target_idx).copied().unwrap_or(false)
                && summaries
                    .bundle_accepts
                    .get(branch.bundle_id)
                    .copied()
                    .unwrap_or(false)
                && summaries.unique_bundles[branch.bundle_id].len() > 1
                && !preserve_bundle(summaries.unique_bundles[branch.bundle_id].len())
            {
                used_multi_bundle[branch.bundle_id] = true;
            }
        }
    }

    use rayon::prelude::*;

    let mut built_bundle_cache: Vec<Option<Arc<NWA>>> = summaries
        .unique_bundles
        .iter()
        .map(|bundle| {
            prebuilt_bundle_cache
                .and_then(|cache| cache.by_signature.get(&bundle_signature(bundle)))
                .cloned()
        })
        .collect();
    let prebuilt_bundle_hits = built_bundle_cache.iter().filter(|entry| entry.is_some()).count();
    let bundle_prebuild_started_at = Instant::now();
    let mut repeated_group_cache_ms = 0.0f64;
    if !compose_detail_enabled && !preserve_bundle_nondeterminism {
        let repeated_group_cache_started_at = Instant::now();
        let used_bundles = summaries
            .unique_bundles
            .iter()
            .enumerate()
            .filter_map(|(bundle_id, bundle)| {
                (used_multi_bundle[bundle_id] && built_bundle_cache[bundle_id].is_none())
                    .then_some(bundle)
            })
            .collect::<Vec<_>>();
        let cache_plan = templates.plan_bundle_group_dfa_cache(&used_bundles);
        let cache_plan_stats = cache_plan.stats;
        const LOW_REUSE_LARGE_GROUP_TERMINALS: usize = 256;
        let parallel_bundle_prebuild = allow_parallel
            && !crate::templates::macro_parallelism_disabled()
            && rayon::current_num_threads() > 1;
        // A single very large group used by only two bundles is cheaper to
        // construct independently inside the parallel bundle prebuild than to
        // serialize both bundles behind one eagerly materialized shared union.
        // Keep the cache for smaller groups and for serial compilation, where
        // duplicate construction cannot overlap.
        let skip_low_reuse_large_group_cache = parallel_bundle_prebuild
            && cache_plan_stats.repeated_groups == 1
            && cache_plan_stats.repeated_group_occurrences == 2
            && cache_plan_stats.max_repeated_group_terminals >= LOW_REUSE_LARGE_GROUP_TERMINALS;
        if compile_profile_enabled() {
            eprintln!(
                "[glrmask/profile][parser_repeated_group_cache_plan] bundles={} multi_group_occurrences={} repeated_groups={} repeated_occurrences={} repeated_terminal_occurrences={} max_repeated_group_terminals={} skip_low_reuse_large={}",
                used_bundles.len(),
                cache_plan_stats.multi_terminal_group_occurrences,
                cache_plan_stats.repeated_groups,
                cache_plan_stats.repeated_group_occurrences,
                cache_plan_stats.repeated_terminal_occurrences,
                cache_plan_stats.max_repeated_group_terminals,
                skip_low_reuse_large_group_cache,
            );
        }
        let repeated_group_cache = (!skip_low_reuse_large_group_cache)
            .then(|| templates.build_bundle_group_dfa_cache_from_plan(cache_plan));
        repeated_group_cache_ms = elapsed_ms(repeated_group_cache_started_at);
        let coarse_parallel_bundle = if std::env::var_os(
            "GLRMASK_EXPERIMENT_PARALLEL_BUNDLE_DETERMINIZE",
        )
        .is_some()
            && allow_parallel
            && !crate::templates::macro_parallelism_disabled()
            && rayon::current_num_threads() > 1
        {
            summaries
                .unique_bundles
                .iter()
                .enumerate()
                .filter(|(bundle_id, bundle)| used_multi_bundle[*bundle_id] && bundle.len() > 1)
                .max_by_key(|(_, bundle)| bundle.len())
                .map(|(bundle_id, bundle)| {
                    let started = Instant::now();
                    let built = Arc::new(match repeated_group_cache.as_ref() {
                        Some(cache) => templates.build_bundle_cached(bundle, cache),
                        None => templates.build_bundle(bundle),
                    });
                    if compile_profile_enabled() {
                        eprintln!(
                            "[glrmask/profile][parser_bundle_coarse_parallel] bundle_id={} terminals={} states={} transitions={} total_ms={:.3}",
                            bundle_id,
                            bundle.len(),
                            built.num_states(),
                            built.num_transitions(),
                            elapsed_ms(started),
                        );
                    }
                    (bundle_id, built)
                })
        } else {
            None
        };
        let coarse_parallel_bundle_id = coarse_parallel_bundle.as_ref().map(|(id, _)| *id);
        let profile_bundle_prebuild =
            std::env::var_os("GLRMASK_PROFILE_BUNDLE_PREBUILD_DETAIL").is_some();
        let build_bundle = |(bundle_id, bundle): (usize, &BTreeMap<TerminalID, Weight>)| {
            if !(used_multi_bundle[bundle_id]
                && built_bundle_cache[bundle_id].is_none()
                && Some(bundle_id) != coarse_parallel_bundle_id)
            {
                return (None, 0.0f64);
            }
            let started = Instant::now();
            let built = Arc::new(match repeated_group_cache.as_ref() {
                Some(cache) => templates.build_bundle_cached(bundle, cache),
                None => templates.build_bundle(bundle),
            });
            let ms = elapsed_ms(started);
            (Some(built), ms)
        };
        let built_with_timings = if allow_parallel
            && !crate::templates::macro_parallelism_disabled()
        {
            summaries
                .unique_bundles
                .par_iter()
                .enumerate()
                .map(build_bundle)
                .collect::<Vec<_>>()
        } else {
            summaries
                .unique_bundles
                .iter()
                .enumerate()
                .map(build_bundle)
                .collect::<Vec<_>>()
        };
        crate::templates::report_macro_item_timings(
            "parser_bundle_prebuild_all",
            &built_with_timings
                .iter()
                .filter_map(|(built, ms)| built.as_ref().map(|_| *ms))
                .collect::<Vec<_>>(),
        );
        if profile_bundle_prebuild {
            let mut rows = built_with_timings
                .iter()
                .enumerate()
                .filter_map(|(bundle_id, (built, ms))| {
                    built.as_ref().map(|built| (
                        bundle_id,
                        summaries.unique_bundles[bundle_id].len(),
                        *ms,
                        built.num_states(),
                        built.num_transitions(),
                    ))
                })
                .collect::<Vec<_>>();
            rows.sort_by(|left, right| right.2.total_cmp(&left.2));
            for (bundle_id, terminals, ms, states, transitions) in rows.into_iter().take(12) {
                eprintln!(
                    "[glrmask/profile][parser_bundle_prebuild_item] id={} terminals={} states={} transitions={} ms={:.3}",
                    bundle_id, terminals, states, transitions, ms,
                );
            }
        }
        for (slot, (built, _)) in built_bundle_cache.iter_mut().zip(built_with_timings) {
            if built.is_some() {
                *slot = built;
            }
        }
        if let Some((bundle_id, built)) = coarse_parallel_bundle {
            built_bundle_cache[bundle_id] = Some(built);
        }
    }
    let bundle_prebuild_ms = elapsed_ms(bundle_prebuild_started_at);

    let branch_walk_started_at = Instant::now();
    let parallel_fragment_assembly = std::env::var_os(
        "GLRMASK_EXPERIMENT_PARALLEL_FRAGMENT_ASSEMBLY",
    )
    .is_some()
        && !crate::templates::macro_parallelism_disabled()
        && !compose_detail_enabled
        && !preserve_bundle_nondeterminism
        && hybrid_nondeterministic_min_terminals.is_none()
        && allow_parallel
        && rayon::current_num_threads() > 1;

    let parallel_fragments_done = if parallel_fragment_assembly {
        enum FragmentSource<'a> {
            WeightedTemplate(&'a NWA, &'a Weight),
            Bundle(&'a NWA),
        }
        struct FragmentTask<'a> {
            from: u32,
            target_continuation: u32,
            source: FragmentSource<'a>,
            offset: u32,
        }

        let mut specs = Vec::<(u32, u32, usize)>::new();
        let mut keys = FxHashSet::<(usize, u32)>::default();
        let mut unique = true;
        for (state_id, state) in states.iter().enumerate() {
            if !productive[state_id] {
                continue;
            }
            let from = continuation_states[state_id];
            for branch in &state.branches {
                let target_idx = branch.target as usize;
                if !productive.get(target_idx).copied().unwrap_or(false)
                    || !summaries
                        .bundle_accepts
                        .get(branch.bundle_id)
                        .copied()
                        .unwrap_or(false)
                {
                    continue;
                }
                if !keys.insert((branch.bundle_id, branch.target)) {
                    unique = false;
                    break;
                }
                specs.push((from, continuation_states[target_idx], branch.bundle_id));
            }
            if !unique {
                break;
            }
        }

        if unique {
            let base_offset = arena.states().len() as u32;
            let mut next_offset = base_offset;
            let mut tasks = Vec::<FragmentTask<'_>>::with_capacity(specs.len());
            let mut valid = true;
            for (from, target_continuation, bundle_id) in specs {
                let bundle = &summaries.unique_bundles[bundle_id];
                let source = if bundle.len() == 1 {
                    let (&terminal, weight) = bundle.iter().next().expect("len checked");
                    if weight.is_empty() {
                        valid = false;
                        break;
                    }
                    let Some(template) = templates.by_terminal_nwa.get(&terminal) else {
                        valid = false;
                        break;
                    };
                    FragmentSource::WeightedTemplate(template, weight)
                } else {
                    let Some(source) = built_bundle_cache[bundle_id].as_deref() else {
                        valid = false;
                        break;
                    };
                    FragmentSource::Bundle(source)
                };
                let len = match source {
                    FragmentSource::WeightedTemplate(template, _) => template.states().len(),
                    FragmentSource::Bundle(bundle) => bundle.states().len(),
                };
                let Some(after) = next_offset.checked_add(len as u32) else {
                    valid = false;
                    break;
                };
                tasks.push(FragmentTask {
                    from,
                    target_continuation,
                    source,
                    offset: next_offset,
                });
                next_offset = after;
            }

            if valid {
                let built = tasks
                    .par_iter()
                    .map(|task| {
                        let (source, override_weight) = match task.source {
                            FragmentSource::WeightedTemplate(source, weight) => {
                                (source, Some(weight))
                            }
                            FragmentSource::Bundle(source) => (source, None),
                        };
                        let starts = source
                            .start_states()
                            .iter()
                            .map(|state| task.offset + *state)
                            .collect::<Vec<_>>();
                        let mut output = Vec::<NWAState>::with_capacity(source.states().len());
                        for source_state in source.states() {
                            let mut appended = source_state.clone();
                            for targets in appended.transitions.values_mut() {
                                for (target, edge_weight) in targets {
                                    *target += task.offset;
                                    if let Some(weight) = override_weight {
                                        *edge_weight = weight.clone();
                                    }
                                }
                            }
                            for (target, epsilon_weight) in &mut appended.epsilons {
                                *target += task.offset;
                                if let Some(weight) = override_weight {
                                    *epsilon_weight = weight.clone();
                                }
                            }
                            if let Some(final_weight) = appended.final_weight.take() {
                                let continuation_weight = override_weight
                                    .cloned()
                                    .unwrap_or(final_weight);
                                if !continuation_weight.is_empty() {
                                    appended
                                        .epsilons
                                        .push((task.target_continuation, continuation_weight));
                                }
                            }
                            output.push(appended);
                        }
                        (task.from, starts, output)
                    })
                    .collect::<Vec<_>>();
                let fragment_count = built.len();
                let total_states = built.iter().map(|(_, _, states)| states.len()).sum::<usize>();
                {
                    let arena_states = arena.states_mut();
                    arena_states.reserve(total_states);
                    let mut links = Vec::with_capacity(fragment_count);
                    for (from, starts, states) in built {
                        arena_states.extend(states);
                        links.push((from, starts));
                    }
                    for (from, starts) in links {
                        for start in starts {
                            arena_states[from as usize]
                                .epsilons
                                .push((start, Weight::all()));
                        }
                    }
                }
                if compile_profile_enabled() {
                    eprintln!(
                        "[glrmask/profile][parser_parallel_fragment_assembly] fragments={} states={} ms={:.3}",
                        fragment_count,
                        total_states,
                        elapsed_ms(branch_walk_started_at),
                    );
                }
                true
            } else {
                false
            }
        } else {
            false
        }
    } else {
        false
    };

    for (state_id, state) in states.iter().enumerate() {
        if !productive[state_id] {
            continue;
        }
        let from = continuation_states[state_id];
        assert_ne!(from, u32::MAX, "missing parser-DWA continuation state");

        for (target, weight) in &state.epsilon_branches {
            let target_idx = *target as usize;
            if weight.is_empty() || !productive.get(target_idx).copied().unwrap_or(false) {
                continue;
            }
            let target_continuation = continuation_states[target_idx];
            assert_ne!(
                target_continuation,
                u32::MAX,
                "missing parser-DWA epsilon target continuation state",
            );
            arena.add_epsilon(from, target_continuation, weight.clone());
            compose_detail.productive_branches += 1;
            compose_detail.epsilon_edges_added += 1;
        }

        if parallel_fragments_done {
            continue;
        }

        for branch in &state.branches {
            let target_idx = branch.target as usize;
            if !productive.get(target_idx).copied().unwrap_or(false)
                || !summaries
                    .bundle_accepts
                    .get(branch.bundle_id)
                    .copied()
                    .unwrap_or(false)
            {
                continue;
            }
            compose_detail.productive_branches += 1;

            let target_continuation = continuation_states[target_idx];
            assert_ne!(
                target_continuation,
                u32::MAX,
                "missing parser-DWA target continuation state",
            );
            let fragment_key = (branch.bundle_id, branch.target);
            let fragment = if let Some(existing) = branch_fragment_memo.get(&fragment_key) {
                if compose_detail_enabled {
                    let memo_hit_started_at = Instant::now();
                    compose_detail.memo_hits += 1;
                    let cloned = existing.clone();
                    compose_detail.memo_hit_clone_ms += elapsed_ms(memo_hit_started_at);
                    cloned
                } else {
                    existing.clone()
                }
            } else {
                if compose_detail_enabled {
                    compose_detail.memo_misses += 1;
                    if built_bundle_cache[branch.bundle_id].is_none()
                        && summaries.unique_bundles[branch.bundle_id].len() > 1
                    {
                        compose_detail.bundle_cache_builds += 1;
                    }
                }
                let fragment_build_started_at = Instant::now();
                let Some(body) = append_branch_fragment(
                    &mut arena,
                    &summaries,
                    &templates,
                    &mut built_bundle_cache,
                    branch.bundle_id,
                    target_continuation,
                    preserve_bundle(summaries.unique_bundles[branch.bundle_id].len()),
                    compose_detail_enabled.then_some(&mut compose_detail),
                ) else {
                    continue;
                };
                compose_detail.fragment_build_ms += elapsed_ms(fragment_build_started_at);
                branch_fragment_memo.insert(fragment_key, body.clone());
                body
            };

            let epsilon_link_started_at = Instant::now();
            let fragment_start_states_len = fragment.start_states.len();
            for start in fragment.start_states {
                arena.add_epsilon(from, start, Weight::all());
                compose_detail.epsilon_edges_added += 1;
            }
            compose_detail.fragment_start_states_total += fragment_start_states_len;
            compose_detail.epsilon_link_ms += elapsed_ms(epsilon_link_started_at);
        }
    }
    compose_detail.branch_walk_ms = elapsed_ms(branch_walk_started_at);

    let parser_start_states: Vec<u32> = productive_start_states
        .into_iter()
        .map(|state| continuation_states[state as usize])
        .collect();
    assert!(
        parser_start_states.iter().all(|state| *state != u32::MAX),
        "missing parser-DWA start continuation state",
    );
    arena.set_start_states(parser_start_states);
    let compose_state_ms = elapsed_ms(graph_started_at);
    if compile_profile_enabled() && !compose_detail_enabled {
        eprintln!(
            "[glrmask/profile][parser_dwa_build_phases] state_prep_ms={state_prep_ms:.3} repeated_group_cache_ms={repeated_group_cache_ms:.3} bundle_prebuild_ms={bundle_prebuild_ms:.3} prebuilt_bundle_hits={prebuilt_bundle_hits} branch_walk_ms={:.3} graph_total_ms={compose_state_ms:.3} total_ms={:.3}",
            elapsed_ms(branch_walk_started_at),
            elapsed_ms(total_started_at),
        );
    }

    if compose_detail_enabled {
        eprintln!(
            "[glrmask/profile][parser_dwa_compose] total_states={} productive_states={} total_branches={} productive_branches={} unique_bundles={} accepting_bundles={} state_init_ms={:.3} branch_walk_ms={:.3} memo_hit_clone_ms={:.3} fragment_build_ms={:.3} epsilon_link_ms={:.3} memo_hits={} memo_misses={} bundle_cache_builds={} epsilon_edges_added={} fragment_start_states_total={}",
            compose_detail.total_states,
            compose_detail.productive_states,
            compose_detail.total_branches,
            compose_detail.productive_branches,
            compose_detail.unique_bundles,
            compose_detail.accepting_bundles,
            compose_detail.state_init_ms,
            compose_detail.branch_walk_ms,
            compose_detail.memo_hit_clone_ms,
            compose_detail.fragment_build_ms,
            compose_detail.epsilon_link_ms,
            compose_detail.memo_hits,
            compose_detail.memo_misses,
            compose_detail.bundle_cache_builds,
            compose_detail.epsilon_edges_added,
            compose_detail.fragment_start_states_total,
        );
        eprintln!(
            "[glrmask/profile][parser_dwa_compose_bundles] bundle_cache_builds={} bundle_profile_total_ms={:.3} build_group_dfas_ms={:.3} union_groups_ms={:.3} determinize_bundle_ms={:.3} minimize_ms={:.3} dwa_to_nwa_ms={:.3} result_dwa_states_total={} result_dwa_transitions_total={} result_nwa_states_total={} result_nwa_transitions_total={}",
            compose_detail.bundle_cache_builds,
            compose_detail.bundle_profile_total_ms,
            compose_detail.bundle_profile_build_group_dfas_ms,
            compose_detail.bundle_profile_union_groups_ms,
            compose_detail.bundle_profile_determinize_ms,
            compose_detail.bundle_profile_minimize_ms,
            compose_detail.bundle_profile_dwa_to_nwa_ms,
            compose_detail.bundle_profile_result_dwa_states,
            compose_detail.bundle_profile_result_dwa_transitions,
            compose_detail.bundle_profile_result_nwa_states,
            compose_detail.bundle_profile_result_nwa_transitions,
        );
    }

    Some((
        arena,
        ParserNwaBuildProfile {
            state_prep_ms,
            compose_state_ms,
            parser_nwa_build_ms: elapsed_ms(total_started_at),
        },
    ))
}

/// Build the exact parser-stack NWA induced by a terminal automaton without
/// resolving negative stack-effect labels or determinizing/minimizing it.
///
/// Composition can carry this representation directly into a later union and
/// perform those global normalization steps once, after all parser languages
/// have been combined.
pub fn build_parser_nwa_from_terminal_dwa_with_precomputed_templates(
    terminal_dwa: &TerminalAutomaton,
    grammar: &AnalyzedGrammar,
    templates: &Templates,
    table: &GLRTable,
) -> Option<NWA> {
    build_parser_nwa_from_terminal_dwa(terminal_dwa, grammar, templates, table)
        .map(|(nwa, _)| nwa)
}

/// Count-only parser-NWA construction for callers which already own the GLR
/// table and precomputed terminal templates. The parser-NWA composition itself
/// consults the analyzed grammar only to reject terminal labels outside the
/// grammar terminal domain; all stack behavior comes from `templates` and
/// `table`.
pub fn build_parser_nwa_from_terminal_dwa_with_precomputed_templates_for_terminal_count(
    terminal_dwa: &TerminalAutomaton,
    num_terminals: u32,
    templates: &Templates,
    table: &GLRTable,
) -> Option<NWA> {
    build_parser_nwa_from_terminal_dwa_for_terminal_count(
        terminal_dwa,
        num_terminals,
        templates,
        Some(table),
        false,
        None,
        true,
    )
    .map(|(nwa, _)| nwa)
}

/// Count-only parser-NWA construction with temporary compile-time bundle
/// nondeterminism. This is language-exact at the bundle-union level, but the
/// returned NWA still contains the ordinary signed stack-effect alphabet and
/// MUST be passed through compile-time negative resolution before any runtime
/// artifact is constructed. It is not a runtime signed-stack representation.
pub fn build_parser_nwa_from_terminal_dwa_with_precomputed_templates_for_terminal_count_no_table(
    terminal_dwa: &TerminalAutomaton,
    num_terminals: u32,
    templates: &Templates,
    preserve_bundle_nondeterminism: bool,
) -> Option<NWA> {
    build_parser_nwa_from_terminal_dwa_for_terminal_count(
        terminal_dwa,
        num_terminals,
        templates,
        None,
        preserve_bundle_nondeterminism,
        None,
        true,
    )
    .map(|(nwa, _)| nwa)
}

pub fn build_parser_nwa_from_terminal_dwa_with_precomputed_templates_for_terminal_count_no_table_with_bundle_cache(
    terminal_dwa: &TerminalAutomaton,
    num_terminals: u32,
    templates: &Templates,
    prebuilt_bundle_cache: &PrebuiltParserBundleCache,
) -> Option<NWA> {
    build_parser_nwa_from_terminal_dwa_for_terminal_count(
        terminal_dwa,
        num_terminals,
        templates,
        None,
        false,
        Some(prebuilt_bundle_cache),
        false,
    )
    .map(|(nwa, _)| nwa)
}

pub fn build_parser_nwa_from_terminal_dwa_with_precomputed_templates_for_terminal_count_nondeterministic_bundles(
    terminal_dwa: &TerminalAutomaton,
    num_terminals: u32,
    templates: &Templates,
    table: &GLRTable,
) -> Option<NWA> {
    build_parser_nwa_from_terminal_dwa_for_terminal_count(
        terminal_dwa,
        num_terminals,
        templates,
        Some(table),
        true,
        None,
        true,
    )
    .map(|(nwa, _)| nwa)
}

// Exact compile-time parser-stack domain algebra used by the compiled-subgrammar linker.
fn determinize_boolean_domain_with_supports(domain: &NWA) -> DeterminizedDwaWithSupports {
    fn epsilon_closure(domain: &NWA, seeds: &[u32]) -> Vec<u32> {
        let mut seen = FxHashSet::<u32>::default();
        let mut stack = seeds.to_vec();
        while let Some(state) = stack.pop() {
            if !seen.insert(state) {
                continue;
            }
            let Some(node) = domain.states().get(state as usize) else {
                continue;
            };
            for (target, weight) in &node.epsilons {
                debug_assert!(weight.is_full() || weight.is_empty());
                if !weight.is_empty() {
                    stack.push(*target);
                }
            }
        }
        let mut closure = seen.into_iter().collect::<Vec<_>>();
        closure.sort_unstable();
        closure
    }

    let mut start_seeds = domain.start_states().to_vec();
    start_seeds.sort_unstable();
    start_seeds.dedup();
    let start = epsilon_closure(domain, &start_seeds);
    let mut dwa = DWA::new(0, 0);
    let mut supports = vec![start.clone()];
    if start.is_empty() {
        return DeterminizedDwaWithSupports { dwa, supports };
    }

    let mut subset_to_state = FxHashMap::<Vec<u32>, u32>::default();
    let mut subsets = Vec::<Vec<u32>>::new();
    subset_to_state.insert(start.clone(), dwa.start_state());
    subsets.push(start);
    let mut queue = VecDeque::from([dwa.start_state()]);

    while let Some(source_id) = queue.pop_front() {
        let subset = subsets[source_id as usize].clone();
        if subset.iter().any(|&state| {
            domain.states()[state as usize]
                .final_weight
                .as_ref()
                .is_some_and(|weight| !weight.is_empty())
        }) {
            dwa.set_final_weight(source_id, Weight::all());
        }

        // IMPORTANT: DEFAULT_LABEL is still an ordinary symbolic label here.
        // It becomes a parser-state fallback only after this subset's NWA
        // support has been recorded and PossibleOutgoingIds can be derived.
        let mut targets_by_label = BTreeMap::<i32, Vec<u32>>::new();
        for &state in &subset {
            let node = &domain.states()[state as usize];
            for (&label, targets) in &node.transitions {
                if is_negative_label(label) {
                    panic!("boolean parser-domain determinization requires negative-free NWA");
                }
                let seeds = targets_by_label.entry(label).or_default();
                for (target, weight) in targets {
                    debug_assert!(weight.is_full() || weight.is_empty());
                    if !weight.is_empty() {
                        seeds.push(*target);
                    }
                }
            }
        }

        for (label, mut seeds) in targets_by_label {
            seeds.sort_unstable();
            seeds.dedup();
            let closure = epsilon_closure(domain, &seeds);
            if closure.is_empty() {
                continue;
            }
            let target = if let Some(&existing) = subset_to_state.get(&closure) {
                existing
            } else {
                let target = dwa.add_state();
                subset_to_state.insert(closure.clone(), target);
                supports.push(closure.clone());
                subsets.push(closure);
                queue.push_back(target);
                target
            };
            dwa.add_transition(source_id, label, target, Weight::all());
        }
    }

    DeterminizedDwaWithSupports { dwa, supports }
}

pub fn determinize_boolean_parser_stack_domain_nwa(
    table: &GLRTable,
    domain: &NWA,
) -> DWA {
    let determinized = determinize_boolean_domain_with_supports(domain);
    let mut result = determinized.dwa;
    let possible_by_state = build_possible_outgoing_ids_by_state(
        domain,
        &determinized.supports,
        table.num_states,
    );
    if std::env::var_os("GLRMASK_EXPERIMENT_DISABLE_BOOLEAN_DOMAIN_DEFAULT_OPT").is_none() {
        optimize_parser_dwa_defaults(&mut result, &possible_by_state, table.num_states);
    }
    subtract_final_weights_from_outgoing_dwa_impl(&mut result, false);
    determinize_parser_dwa_with_fallbacks(&result, &possible_by_state, table.num_states)
}

pub fn normalize_parser_stack_domain_nwa(table: &GLRTable, domain: &NWA) -> DWA {
    minimize(&determinize_boolean_parser_stack_domain_nwa(table, domain))
}

/// Normalize a boolean parser-stack NWA while preserving its explicit rows.
///
/// This is the exact standalone form needed when several independently
/// supported parser domains will be combined later. Synthesizing DEFAULT rows
/// here would discard the support provenance required by that later union.
pub fn normalize_parser_stack_domain_nwa_preserving_explicit(
    table: &GLRTable,
    domain: &NWA,
) -> DWA {
    let determinized = determinize_boolean_domain_with_supports(domain);
    let mut result = determinized.dwa;
    let possible_by_state = build_possible_outgoing_ids_by_state(
        domain,
        &determinized.supports,
        table.num_states,
    );
    subtract_final_weights_from_outgoing_dwa_impl(&mut result, false);
    let result = determinize_parser_dwa_with_fallbacks(
        &result,
        &possible_by_state,
        table.num_states,
    );
    minimize(&result)
}

/// Normalize an already-positive weighted parser-stack NWA into the ordinary
/// runtime parser DWA representation.
///
/// This is the post-negative-resolution half of
/// `build_parser_dwa_from_terminal_dwa_with_precomputed_templates`: callers
/// that have composed parser-stack effects directly can reuse the exact same
/// support-aware DEFAULT/finality semantics without first reconstructing a
/// terminal automaton.

fn normalize_weighted_parser_stack_nwa_impl(
    num_parser_states: u32,
    parser_nwa: &NWA,
    small_boundary_coordinate: Option<(usize, usize)>,
    source_tsid_map: Option<&[u32]>,
) -> DWA {
    let profile = std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some();
    let total_started_at = profile.then(Instant::now);
    let determinize_started_at = Instant::now();
    let determinized = small_boundary_coordinate
        .and_then(|(tsids, tokens)| {
            determinize_with_supports_small_boundary(
                parser_nwa,
                num_parser_states,
                tsids,
                tokens,
                source_tsid_map,
            )
        })
        .unwrap_or_else(|| determinize_with_supports(parser_nwa, Some(num_parser_states)));
    let determinize_ms = elapsed_ms(determinize_started_at);
    let mut parser_dwa = determinized.dwa;
    let compact_fallback = small_boundary_coordinate.is_some()
        && std::env::var_os("GLRMASK_EXPERIMENT_SMALL_BOUNDARY_COMPACT_FALLBACK").is_some();
    let possible_started_at = Instant::now();
    let possible_by_state = if compact_fallback {
        Vec::new()
    } else {
        build_possible_outgoing_ids_by_state(parser_nwa, &determinized.supports, num_parser_states)
    };
    let possible_ms = elapsed_ms(possible_started_at);
    let compact_post = small_boundary_coordinate.is_some()
        && (compact_fallback
            || std::env::var_os("GLRMASK_EXPERIMENT_SMALL_BOUNDARY_COMPACT_POST").is_some());
    let default_started_at = Instant::now();
    if !compact_post
        && std::env::var_os("GLRMASK_EXPERIMENT_LAZY_DIRECT_DISABLE_DEFAULT_OPT").is_none()
    {
        optimize_parser_dwa_defaults(&mut parser_dwa, &possible_by_state, num_parser_states);
    }
    let default_ms = elapsed_ms(default_started_at);
    let subtract_started_at = Instant::now();
    if !compact_post {
        subtract_final_weights_from_outgoing_dwa(&mut parser_dwa);
    }
    let subtract_ms = elapsed_ms(subtract_started_at);
    let fallback_started_at = Instant::now();
    if !compact_fallback {
        parser_dwa = determinize_parser_dwa_with_fallbacks(
            &parser_dwa,
            &possible_by_state,
            num_parser_states,
        );
    }
    let fallback_ms = elapsed_ms(fallback_started_at);
    let minimize_started_at = Instant::now();
    let pre_minimize_states = parser_dwa.num_states();
    let pre_minimize_transitions = parser_dwa.num_transitions();
    if should_skip_parser_dwa_minimization(parser_dwa.states().len(), parser_dwa.num_transitions()) {
        if profile {
            eprintln!(
                "[glrmask/profile][normalize_weighted_parser_stack_nwa] nwa_states={} pre_minimize_states={} pre_minimize_transitions={} post_states={} post_transitions={} minimize_skipped=true determinize_ms={determinize_ms:.3} possible_ms={possible_ms:.3} default_ms={default_ms:.3} subtract_ms={subtract_ms:.3} fallback_ms={fallback_ms:.3} minimize_ms=0.000 total_ms={:.3}",
                parser_nwa.num_states(),
                pre_minimize_states,
                pre_minimize_transitions,
                parser_dwa.num_states(),
                parser_dwa.num_transitions(),
                total_started_at.map_or(0.0, elapsed_ms),
            );
        }
        parser_dwa
    } else {
        let minimized = minimize(&parser_dwa);
        let minimize_ms = elapsed_ms(minimize_started_at);
        if profile {
            eprintln!(
                "[glrmask/profile][normalize_weighted_parser_stack_nwa] nwa_states={} pre_minimize_states={} pre_minimize_transitions={} post_states={} post_transitions={} minimize_skipped=false determinize_ms={determinize_ms:.3} possible_ms={possible_ms:.3} default_ms={default_ms:.3} subtract_ms={subtract_ms:.3} fallback_ms={fallback_ms:.3} minimize_ms={minimize_ms:.3} total_ms={:.3}",
                parser_nwa.num_states(),
                pre_minimize_states,
                pre_minimize_transitions,
                minimized.num_states(),
                minimized.num_transitions(),
                total_started_at.map_or(0.0, elapsed_ms),
            );
        }
        minimized
    }
}

pub fn normalize_weighted_parser_stack_nwa(table: &GLRTable, parser_nwa: &NWA) -> DWA {
    normalize_weighted_parser_stack_nwa_impl(table.num_states, parser_nwa, None, None)
}

pub fn normalize_weighted_parser_stack_nwa_for_parser_state_count(
    num_parser_states: u32,
    parser_nwa: &NWA,
) -> DWA {
    normalize_weighted_parser_stack_nwa_impl(num_parser_states, parser_nwa, None, None)
}

pub fn normalize_weighted_parser_stack_nwa_small_boundary(
    table: &GLRTable,
    parser_nwa: &NWA,
    num_tsids: usize,
    num_tokens: usize,
) -> DWA {
    let candidate = normalize_weighted_parser_stack_nwa_impl(
        table.num_states,
        parser_nwa,
        Some((num_tsids, num_tokens)),
        None,
    );
    if std::env::var_os("GLRMASK_VALIDATE_SMALL_BOUNDARY_WEIGHT_DETERMINIZER").is_some() {
        let reference = normalize_weighted_parser_stack_nwa_impl(
            table.num_states,
            parser_nwa,
            None,
            None,
        );
        assert_eq!(
            find_difference(&candidate, &reference).unwrap(),
            None,
            "small-boundary weight determinizer accepts behavior absent from generic normalizer",
        );
        assert_eq!(
            find_difference(&reference, &candidate).unwrap(),
            None,
            "small-boundary weight determinizer omits behavior accepted by generic normalizer",
        );
        eprintln!(
            "[glrmask/validate][small_boundary_weight_determinizer] candidate_states={} reference_states={} exact=true",
            candidate.num_states(),
            reference.num_states(),
        );
    }
    candidate
}

pub fn normalize_weighted_parser_stack_nwa_small_boundary_with_tsid_map(
    table: &GLRTable,
    parser_nwa: &NWA,
    num_tsids: usize,
    num_tokens: usize,
    source_tsid_map: &[u32],
) -> DWA {
    normalize_weighted_parser_stack_nwa_impl(
        table.num_states,
        parser_nwa,
        Some((num_tsids, num_tokens)),
        Some(source_tsid_map),
    )
}

pub fn normalize_weighted_parser_stack_nwa_small_boundary_for_parser_state_count(
    num_parser_states: u32,
    parser_nwa: &NWA,
    num_tsids: usize,
    num_tokens: usize,
    source_tsid_map: Option<&[u32]>,
) -> DWA {
    normalize_weighted_parser_stack_nwa_impl(
        num_parser_states,
        parser_nwa,
        Some((num_tsids, num_tokens)),
        source_tsid_map,
    )
}

pub fn universal_parser_stack_domain_dwa() -> DWA {
    let mut result = DWA::new(0, 1);
    result.set_final_weight(0, Weight::all());
    result
}

pub fn universal_parser_stack_domain_nwa() -> NWA {
    let mut result = NWA::new(0, 0);
    let start = result.add_state();
    result.set_start_states(vec![start]);
    result.set_final_weight(start, Weight::all());
    result
}

/// Exact positive-NWA preimage. Unlike the DWA wrapper below, this deliberately
/// does not determinize or normalize the result. That makes it suitable for a
/// backward dynamic program over a finite terminal suffix DAG: a later
/// terminal template can cancel its pushed states directly against this
/// positive NWA, and determinization can be deferred until the completed root
/// predicates are weighted and unioned.
pub fn build_terminal_bundle_preimage_domain_nwa(
    table: &GLRTable,
    templates: &Templates,
    terminals: &[TerminalID],
    target_domain: &NWA,
) -> Option<NWA> {
    if terminals.is_empty() {
        return None;
    }
    let terminal_weights = terminals
        .iter()
        .copied()
        .map(|terminal| (terminal, Weight::all()))
        .collect::<BTreeMap<_, _>>();
    if terminal_weights
        .keys()
        .any(|terminal| !templates.by_terminal_nwa.contains_key(terminal))
    {
        return None;
    }
    let mut bundle = templates.build_bundle(&terminal_weights);
    for state in bundle.states_mut() {
        for targets in state.transitions.values_mut() {
            for (_, weight) in targets {
                if weight.is_empty() {
                    *weight = Weight::all();
                }
            }
        }
        for (_, weight) in &mut state.epsilons {
            if weight.is_empty() {
                *weight = Weight::all();
            }
        }
        if state.final_weight.as_ref().is_some_and(Weight::is_empty) {
            state.final_weight = Some(Weight::all());
        }
    }

    let mut arena = NWA::new(0, 0);
    let bundle_offset = arena.states().len() as u32;
    let bundle_body = arena.append_with_body(&bundle);
    let bundle_finals = bundle
        .states()
        .iter()
        .enumerate()
        .filter_map(|(local, state)| {
            state
                .final_weight
                .as_ref()
                .is_some_and(|weight| !weight.is_empty())
                .then_some(bundle_offset + local as u32)
        })
        .collect::<Vec<_>>();
    if bundle_finals.is_empty() {
        return None;
    }
    let target_body = arena.append_with_body(target_domain);
    for source in bundle_finals {
        let Some(final_weight) = arena.states_mut()[source as usize].final_weight.take() else {
            continue;
        };
        if final_weight.is_empty() {
            continue;
        }
        for &target_start in &target_body.start_states {
            arena.add_epsilon(source, target_start, final_weight.clone());
        }
    }
    arena.set_start_states(bundle_body.start_states);
    resolve_negative_codes_in_nwa(
        &mut arena,
        table.construction == GlrTableConstruction::ExperimentalCoreMerged,
    );
    Some(arena)
}


#[derive(Debug, Clone, Copy, Default)]
pub struct ParserStackPreimageProfile {
    pub bundle_ms: f64,
    pub concatenate_ms: f64,
    pub resolve_ms: f64,
    pub normalize_ms: f64,
    pub total_ms: f64,
    pub bundle_states: usize,
    pub concatenated_states: usize,
    pub result_states: usize,
}


fn advance_boolean_parser_domain_state(
    domain: &DWA,
    state: u32,
    parser_state: u32,
) -> Option<u32> {
    let row = domain.states().get(state as usize)?;
    if row
        .final_weight
        .as_ref()
        .is_some_and(|weight| !weight.is_empty())
    {
        // Parser-stack languages are prefix languages: once the target domain
        // has accepted the visible top-of-stack prefix, any deeper pushed or
        // pre-existing stack suffix is irrelevant. Treat final states as
        // absorbing while consuming the terminal effect's remaining pushes.
        return Some(state);
    }
    let label = parser_state as i32;
    let (target, weight) = row
        .transitions
        .get(&label)
        .or_else(|| row.transitions.get(&DEFAULT_LABEL))?;
    debug_assert!(weight.is_full() || weight.is_empty());
    (!weight.is_empty()).then_some(*target)
}

fn direct_negative_suffix_residuals(
    bundle: &NWA,
    bundle_state: u32,
    domain: &DWA,
    domain_state: u32,
    memo: &mut FxHashMap<(u32, u32), Option<Vec<u32>>>,
) -> Option<Vec<u32>> {
    if let Some(cached) = memo.get(&(bundle_state, domain_state)) {
        return cached.clone();
    }
    let node = bundle.states().get(bundle_state as usize)?;
    let mut residuals = Vec::<u32>::new();
    if node
        .final_weight
        .as_ref()
        .is_some_and(|weight| !weight.is_empty())
    {
        residuals.push(domain_state);
    }
    for (&label, targets) in &node.transitions {
        if !is_negative_label(label) {
            // Stack-effect templates are in read-then-push normal form. Once a
            // push is encountered, seeing another read would require a more
            // general transducer product; decline this direct fast path rather
            // than weakening semantics.
            memo.insert((bundle_state, domain_state), None);
            return None;
        }
        let parser_state = negative_to_positive_label(label) as u32;
        for (target, weight) in targets {
            debug_assert!(weight.is_full() || weight.is_empty());
            if weight.is_empty() {
                continue;
            }
            // Pushes are observed from the final stack top downward, which is
            // the reverse of the negative-edge order in the stack effect. The
            // generic cancellation solver establishes downstream cancellations
            // first and then propagates upstream queries through the derived
            // epsilons. Mirror that algebra directly: resolve the child's later
            // pushes first, then consume this push from each residual domain.
            let Some(child) = direct_negative_suffix_residuals(
                bundle,
                *target,
                domain,
                domain_state,
                memo,
            ) else {
                memo.insert((bundle_state, domain_state), None);
                return None;
            };
            for child_domain in child {
                if let Some(next_domain) = advance_boolean_parser_domain_state(
                    domain,
                    child_domain,
                    parser_state,
                ) {
                    residuals.push(next_domain);
                }
            }
        }
    }
    for (target, weight) in &node.epsilons {
        debug_assert!(weight.is_full() || weight.is_empty());
        if weight.is_empty() {
            continue;
        }
        let Some(child) = direct_negative_suffix_residuals(
            bundle,
            *target,
            domain,
            domain_state,
            memo,
        ) else {
            memo.insert((bundle_state, domain_state), None);
            return None;
        };
        residuals.extend(child);
    }
    residuals.sort_unstable();
    residuals.dedup();
    memo.insert((bundle_state, domain_state), Some(residuals.clone()));
    Some(residuals)
}

/// Algebraic boolean preimage for a read-then-push stack-effect bundle.
///
/// Positive labels are reads from the pre-token parser stack and remain in the
/// resulting automaton. Negative labels are pushes. Because `target_domain` is
/// already deterministic, a pushed parser state is consumed by one ordinary
/// DWA transition (explicit label or DEFAULT fallback); the remaining target
/// state is the exact residual language. This is the relational composition
/// performed by negative-code cancellation, but without materializing the
/// concatenated graph or running a global fixpoint.
pub fn build_prebuilt_terminal_bundle_preimage_domain_dwa_direct_profiled(
    table: &GLRTable,
    bundle: &NWA,
    target_domain: &DWA,
) -> (Option<DWA>, ParserStackPreimageProfile) {
    let total_started_at = Instant::now();
    let mut profile = ParserStackPreimageProfile {
        bundle_states: bundle.states().len(),
        ..ParserStackPreimageProfile::default()
    };
    if bundle.start_states().is_empty() || target_domain.states().is_empty() {
        profile.total_ms = elapsed_ms(total_started_at);
        return (None, profile);
    }

    let build_started_at = Instant::now();
    let target_nwa = target_domain.to_nwa();
    let mut result = NWA::new(0, 0);
    let bundle_offset = 0u32;
    let bundle_body = result.append_with_body(bundle);
    debug_assert_eq!(bundle_offset, 0);
    let target_offset = result.states().len() as u32;
    let target_body = result.append_with_body(&target_nwa);
    debug_assert_eq!(target_body.start_states.len(), 1);

    let target_start = target_domain.start_state();
    let mut memo = FxHashMap::<(u32, u32), Option<Vec<u32>>>::default();
    let bundle_len = bundle.states().len();
    for source in 0..bundle_len {
        let original = &bundle.states()[source];
        let mut positive_transitions = BTreeMap::new();
        let mut epsilons = Vec::<(u32, Weight)>::new();

        if original
            .final_weight
            .as_ref()
            .is_some_and(|weight| !weight.is_empty())
        {
            epsilons.push((target_offset + target_start, Weight::all()));
        }
        for (&label, targets) in &original.transitions {
            if is_negative_label(label) {
                let parser_state = negative_to_positive_label(label) as u32;
                for (target, weight) in targets {
                    debug_assert!(weight.is_full() || weight.is_empty());
                    if weight.is_empty() {
                        continue;
                    }
                    let Some(residuals) = direct_negative_suffix_residuals(
                        bundle,
                        *target,
                        target_domain,
                        target_start,
                        &mut memo,
                    ) else {
                        profile.total_ms = elapsed_ms(total_started_at);
                        return (None, profile);
                    };
                    epsilons.extend(residuals.into_iter().filter_map(|state| {
                        advance_boolean_parser_domain_state(
                            target_domain,
                            state,
                            parser_state,
                        )
                        .map(|residual| (target_offset + residual, Weight::all()))
                    }));
                }
            } else {
                positive_transitions.insert(label, targets.clone());
            }
        }
        // Bundle epsilons belong to the still-reading prefix unless their
        // target immediately enters a negative suffix. Keeping them is exact;
        // any negative transitions at the target are processed when that state
        // is reached by epsilon closure during determinization.
        epsilons.extend(original.epsilons.iter().cloned());
        epsilons.sort_unstable_by_key(|(target, _)| *target);
        epsilons.dedup_by(|left, right| left.0 == right.0 && left.1 == right.1);
        let state = &mut result.states_mut()[source];
        state.transitions = positive_transitions;
        state.epsilons = epsilons;
        state.final_weight = None;
    }
    result.set_start_states(bundle_body.start_states);
    // Negative cancellation is now algebraic, but parser stack languages also
    // project finality backward through DEFAULT/epsilon edges. Preserve that
    // exact stack-prefix semantics before ordinary boolean determinization.
    apply_finality_fixpoint(&mut result);
    profile.concatenate_ms = elapsed_ms(build_started_at);
    profile.concatenated_states = result.states().len();

    let normalize_started_at = Instant::now();
    let domain = normalize_parser_stack_domain_nwa(table, &result);
    profile.normalize_ms = elapsed_ms(normalize_started_at);
    profile.result_states = domain.states().len();
    profile.total_ms = elapsed_ms(total_started_at);

    (Some(domain), profile)
}


pub fn build_boolean_terminal_bundle_nwa(
    templates: &Templates,
    terminals: &[TerminalID],
) -> Option<NWA> {
    if terminals.is_empty()
        || terminals
            .iter()
            .any(|terminal| !templates.by_terminal_nwa.contains_key(terminal))
    {
        return None;
    }
    let terminal_weights = terminals
        .iter()
        .copied()
        .map(|terminal| (terminal, Weight::all()))
        .collect::<BTreeMap<_, _>>();
    let mut bundle = templates.build_bundle(&terminal_weights);
    for state in bundle.states_mut() {
        for targets in state.transitions.values_mut() {
            for (_, weight) in targets {
                if weight.is_empty() {
                    *weight = Weight::all();
                }
            }
        }
        for (_, weight) in &mut state.epsilons {
            if weight.is_empty() {
                *weight = Weight::all();
            }
        }
        if state.final_weight.as_ref().is_some_and(Weight::is_empty) {
            state.final_weight = Some(Weight::all());
        }
    }
    Some(bundle)
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum LazyBooleanDomainExpr {
    Empty,
    Universal,
    Read { label: i32, child: u32 },
    Union { left: u32, right: u32 },
}

/// Lazy compile-time algebra for boolean parser-stack prefix languages.
///
/// Unlike `SharedBooleanParserDomains`, this representation deliberately does
/// not determinize unions. DEFAULT remains an ordinary symbolic stack matcher
/// until the final NWA normalization, preserving the parser-support semantics
/// used by the generic compiler. The DAG is only an intermediate linking form;
/// runtime artifacts remain ordinary DWAs.
pub struct LazyBooleanParserDomains {
    nodes: Vec<LazyBooleanDomainExpr>,
    prefix_final: Vec<bool>,
    reads: FxHashMap<(i32, u32), u32>,
    unions: FxHashMap<(u32, u32), u32>,
    advance_memo: FxHashMap<(u32, u32), u32>,
    default_advance_memo: FxHashMap<u32, u32>,
    prefix_finalized: FxHashSet<u32>,
}

impl Default for LazyBooleanParserDomains {
    fn default() -> Self {
        Self::new()
    }
}

impl LazyBooleanParserDomains {
    pub const EMPTY: u32 = 0;
    pub const UNIVERSAL: u32 = 1;

    pub fn new() -> Self {
        Self {
            nodes: vec![LazyBooleanDomainExpr::Empty, LazyBooleanDomainExpr::Universal],
            prefix_final: vec![false, true],
            reads: FxHashMap::default(),
            unions: FxHashMap::default(),
            advance_memo: FxHashMap::default(),
            default_advance_memo: FxHashMap::default(),
            prefix_finalized: FxHashSet::from_iter([Self::EMPTY, Self::UNIVERSAL]),
        }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty_root(&self, root: u32) -> bool {
        root == Self::EMPTY
    }

    pub fn is_universal_root(&self, root: u32) -> bool {
        root == Self::UNIVERSAL
    }

    /// Concrete parser-state labels whose derivative can differ from the
    /// wildcard-only derivative. DEFAULT itself is deliberately excluded.
    pub fn explicit_labels(&self, root: u32) -> Vec<i32> {
        let mut labels = BTreeSet::<i32>::new();
        let mut seen = FxHashSet::<u32>::default();
        let mut stack = vec![root];
        while let Some(node) = stack.pop() {
            if !seen.insert(node) {
                continue;
            }
            match self.nodes[node as usize] {
                LazyBooleanDomainExpr::Empty | LazyBooleanDomainExpr::Universal => {}
                LazyBooleanDomainExpr::Read { label, child } => {
                    if label != DEFAULT_LABEL {
                        debug_assert!(!is_negative_label(label));
                        labels.insert(label);
                    }
                    stack.push(child);
                }
                LazyBooleanDomainExpr::Union { left, right } => {
                    stack.push(left);
                    stack.push(right);
                }
            }
        }
        labels.into_iter().collect()
    }

    /// Derivative for any concrete parser state not represented by an explicit
    /// read in the expression. Only symbolic DEFAULT reads survive.
    pub fn advance_default(&mut self, root: u32) -> u32 {
        if root == Self::EMPTY || self.prefix_final[root as usize] {
            return root;
        }
        if let Some(&cached) = self.default_advance_memo.get(&root) {
            return cached;
        }
        let expr = self.nodes[root as usize];
        let result = match expr {
            LazyBooleanDomainExpr::Empty => Self::EMPTY,
            LazyBooleanDomainExpr::Universal => Self::UNIVERSAL,
            LazyBooleanDomainExpr::Read { label, child } => {
                if label == DEFAULT_LABEL { child } else { Self::EMPTY }
            }
            LazyBooleanDomainExpr::Union { left, right } => {
                let left = self.advance_default(left);
                let right = self.advance_default(right);
                self.union(left, right)
            }
        };
        self.default_advance_memo.insert(root, result);
        result
    }

    pub fn read(&mut self, label: i32, child: u32) -> u32 {
        if child == Self::EMPTY {
            return Self::EMPTY;
        }
        let key = (label, child);
        if let Some(&existing) = self.reads.get(&key) {
            return existing;
        }
        let id = self.nodes.len() as u32;
        self.nodes.push(LazyBooleanDomainExpr::Read { label, child });
        self.prefix_final.push(false);
        self.reads.insert(key, id);
        id
    }

    pub fn union(&mut self, left: u32, right: u32) -> u32 {
        if left == right || right == Self::EMPTY {
            return left;
        }
        if left == Self::EMPTY {
            return right;
        }
        if left == Self::UNIVERSAL || right == Self::UNIVERSAL {
            return Self::UNIVERSAL;
        }
        let (left, right) = if left < right { (left, right) } else { (right, left) };
        if let Some(&existing) = self.unions.get(&(left, right)) {
            return existing;
        }
        let id = self.nodes.len() as u32;
        self.nodes.push(LazyBooleanDomainExpr::Union { left, right });
        self.prefix_final.push(
            self.prefix_final[left as usize] || self.prefix_final[right as usize],
        );
        self.unions.insert((left, right), id);
        id
    }

    pub fn union_all(&mut self, roots: impl IntoIterator<Item = u32>) -> u32 {
        roots.into_iter().fold(Self::EMPTY, |acc, root| self.union(acc, root))
    }

    /// Consume one concrete parser state from the top of the represented stack
    /// language. DEFAULT is a symbolic wildcard here: final parser-DWA fallback
    /// normalization happens only when the expression is exported to an NWA.
    pub fn advance(&mut self, root: u32, parser_state: u32) -> u32 {
        if root == Self::EMPTY || self.prefix_final[root as usize] {
            return root;
        }
        if let Some(&cached) = self.advance_memo.get(&(root, parser_state)) {
            return cached;
        }
        let expr = self.nodes[root as usize];
        let result = match expr {
            LazyBooleanDomainExpr::Empty => Self::EMPTY,
            LazyBooleanDomainExpr::Universal => Self::UNIVERSAL,
            LazyBooleanDomainExpr::Read { label, child } => {
                if label == DEFAULT_LABEL || label == parser_state as i32 {
                    child
                } else {
                    Self::EMPTY
                }
            }
            LazyBooleanDomainExpr::Union { left, right } => {
                let left = self.advance(left, parser_state);
                let right = self.advance(right, parser_state);
                self.union(left, right)
            }
        };
        self.advance_memo.insert((root, parser_state), result);
        result
    }

    /// Apply parser-stack prefix finality after a complete bundle
    /// preimage has been assembled. Epsilon/union finality is represented by
    /// the union node's `prefix_final` bit as it is built. The only remaining
    /// backward-finality edge in this negative-free expression is DEFAULT.
    ///
    /// Finalized roots remain structurally intact: their final bit makes them
    /// absorbing for subsequent pushed parser states, while positive-read
    /// provenance is preserved until export/normalization.
    pub fn finalize_prefix_domain(&mut self, root: u32) -> bool {
        if self.prefix_finalized.contains(&root) {
            return self.prefix_final[root as usize];
        }
        let expr = self.nodes[root as usize];
        let final_value = match expr {
            LazyBooleanDomainExpr::Empty => false,
            LazyBooleanDomainExpr::Universal => true,
            LazyBooleanDomainExpr::Read { label, child } => {
                let child_final = self.finalize_prefix_domain(child);
                label == DEFAULT_LABEL && child_final
            }
            LazyBooleanDomainExpr::Union { left, right } => {
                self.finalize_prefix_domain(left) || self.finalize_prefix_domain(right)
            }
        };
        self.prefix_final[root as usize] = final_value;
        self.prefix_finalized.insert(root);
        final_value
    }

    fn negative_suffix_root(
        &mut self,
        bundle: &NWA,
        bundle_state: u32,
        target_root: u32,
        memo: &mut FxHashMap<(u32, u32), Option<u32>>,
    ) -> Option<u32> {
        if let Some(cached) = memo.get(&(bundle_state, target_root)) {
            return *cached;
        }
        let node = bundle.states().get(bundle_state as usize)?;
        let mut result = if node
            .final_weight
            .as_ref()
            .is_some_and(|weight| !weight.is_empty())
        {
            target_root
        } else {
            Self::EMPTY
        };
        for (&label, targets) in &node.transitions {
            if !is_negative_label(label) {
                memo.insert((bundle_state, target_root), None);
                return None;
            }
            let parser_state = negative_to_positive_label(label) as u32;
            for (target, weight) in targets {
                debug_assert!(weight.is_full() || weight.is_empty());
                if weight.is_empty() {
                    continue;
                }
                let Some(child) = self.negative_suffix_root(
                    bundle,
                    *target,
                    target_root,
                    memo,
                ) else {
                    memo.insert((bundle_state, target_root), None);
                    return None;
                };
                let residual = self.advance(child, parser_state);
                result = self.union(result, residual);
            }
        }
        for (target, weight) in &node.epsilons {
            debug_assert!(weight.is_full() || weight.is_empty());
            if weight.is_empty() {
                continue;
            }
            let Some(child) = self.negative_suffix_root(bundle, *target, target_root, memo) else {
                memo.insert((bundle_state, target_root), None);
                return None;
            };
            result = self.union(result, child);
        }
        memo.insert((bundle_state, target_root), Some(result));
        Some(result)
    }

    fn preimage_state(
        &mut self,
        bundle: &NWA,
        bundle_state: u32,
        target_root: u32,
        positive_memo: &mut FxHashMap<(u32, u32), Option<u32>>,
        negative_memo: &mut FxHashMap<(u32, u32), Option<u32>>,
    ) -> Option<u32> {
        if let Some(cached) = positive_memo.get(&(bundle_state, target_root)) {
            return *cached;
        }
        let node = bundle.states().get(bundle_state as usize)?;
        let mut result = if node
            .final_weight
            .as_ref()
            .is_some_and(|weight| !weight.is_empty())
        {
            target_root
        } else {
            Self::EMPTY
        };
        for (&label, targets) in &node.transitions {
            for (target, weight) in targets {
                debug_assert!(weight.is_full() || weight.is_empty());
                if weight.is_empty() {
                    continue;
                }
                let branch = if is_negative_label(label) {
                    let parser_state = negative_to_positive_label(label) as u32;
                    let Some(child) = self.negative_suffix_root(
                        bundle,
                        *target,
                        target_root,
                        negative_memo,
                    ) else {
                        positive_memo.insert((bundle_state, target_root), None);
                        return None;
                    };
                    self.advance(child, parser_state)
                } else {
                    let Some(child) = self.preimage_state(
                        bundle,
                        *target,
                        target_root,
                        positive_memo,
                        negative_memo,
                    ) else {
                        positive_memo.insert((bundle_state, target_root), None);
                        return None;
                    };
                    self.read(label, child)
                };
                result = self.union(result, branch);
            }
        }
        for (target, weight) in &node.epsilons {
            debug_assert!(weight.is_full() || weight.is_empty());
            if weight.is_empty() {
                continue;
            }
            let Some(child) = self.preimage_state(
                bundle,
                *target,
                target_root,
                positive_memo,
                negative_memo,
            ) else {
                positive_memo.insert((bundle_state, target_root), None);
                return None;
            };
            result = self.union(result, child);
        }
        positive_memo.insert((bundle_state, target_root), Some(result));
        Some(result)
    }

    pub fn preimage_bundle(&mut self, bundle: &NWA, target_root: u32) -> Option<u32> {
        let mut positive_memo = FxHashMap::default();
        let mut negative_memo = FxHashMap::default();
        let mut result = Self::EMPTY;
        for &start in bundle.start_states() {
            let root = self.preimage_state(
                bundle,
                start,
                target_root,
                &mut positive_memo,
                &mut negative_memo,
            )?;
            result = self.union(result, root);
        }
        if std::env::var_os("GLRMASK_EXPERIMENT_FINALIZE_LAZY_PREIMAGE").is_some() {
            self.finalize_prefix_domain(result);
        }
        Some(result)
    }

    /// Export several support-weighted roots as one shared positive NWA.
    /// Expression nodes reachable from multiple roots are materialized once;
    /// only the fresh global start carries support weights. This keeps the
    /// lazy DAG's structural sharing intact through the final weighted parser
    /// normalization.
    pub fn to_weighted_nwa(&self, roots: &[(u32, Weight)]) -> NWA {
        let mut reachable = FxHashSet::<u32>::default();
        let mut stack = roots
            .iter()
            .filter_map(|(root, weight)| (!weight.is_empty()).then_some(*root))
            .collect::<Vec<_>>();
        while let Some(node) = stack.pop() {
            if !reachable.insert(node) {
                continue;
            }
            match self.nodes[node as usize] {
                LazyBooleanDomainExpr::Empty | LazyBooleanDomainExpr::Universal => {}
                LazyBooleanDomainExpr::Read { child, .. } => stack.push(child),
                LazyBooleanDomainExpr::Union { left, right } => {
                    stack.push(left);
                    stack.push(right);
                }
            }
        }
        let mut ordered = reachable.into_iter().collect::<Vec<_>>();
        ordered.sort_unstable();
        let mut remap = FxHashMap::<u32, u32>::default();
        let mut nwa = NWA::new(0, 0);
        let global_start = nwa.add_state();
        for node in &ordered {
            remap.insert(*node, nwa.add_state());
        }
        nwa.set_start_states(vec![global_start]);
        for (root, weight) in roots {
            if weight.is_empty() || *root == Self::EMPTY {
                continue;
            }
            if let Some(&target) = remap.get(root) {
                nwa.add_epsilon(global_start, target, weight.clone());
            }
        }
        for node in ordered {
            let from = remap[&node];
            match self.nodes[node as usize] {
                LazyBooleanDomainExpr::Empty => {}
                LazyBooleanDomainExpr::Universal => nwa.set_final_weight(from, Weight::all()),
                LazyBooleanDomainExpr::Read { label, child } => {
                    if self.prefix_final[node as usize] {
                        nwa.set_final_weight(from, Weight::all());
                    }
                    nwa.add_transition(from, label, remap[&child], Weight::all());
                }
                LazyBooleanDomainExpr::Union { left, right } => {
                    if self.prefix_final[node as usize] {
                        nwa.set_final_weight(from, Weight::all());
                    }
                    nwa.add_epsilon(from, remap[&left], Weight::all());
                    nwa.add_epsilon(from, remap[&right], Weight::all());
                }
            }
        }
        nwa
    }

    pub fn to_nwa(&self, root: u32) -> NWA {
        let mut reachable = FxHashSet::<u32>::default();
        let mut stack = vec![root];
        while let Some(node) = stack.pop() {
            if !reachable.insert(node) {
                continue;
            }
            match self.nodes[node as usize] {
                LazyBooleanDomainExpr::Empty | LazyBooleanDomainExpr::Universal => {}
                LazyBooleanDomainExpr::Read { child, .. } => stack.push(child),
                LazyBooleanDomainExpr::Union { left, right } => {
                    stack.push(left);
                    stack.push(right);
                }
            }
        }
        let mut ordered = reachable.into_iter().collect::<Vec<_>>();
        ordered.sort_unstable();
        let mut remap = FxHashMap::<u32, u32>::default();
        let mut nwa = NWA::new(0, 0);
        for node in &ordered {
            remap.insert(*node, nwa.add_state());
        }
        nwa.set_start_states(vec![remap[&root]]);
        for node in ordered {
            let from = remap[&node];
            match self.nodes[node as usize] {
                LazyBooleanDomainExpr::Empty => {}
                LazyBooleanDomainExpr::Universal => nwa.set_final_weight(from, Weight::all()),
                LazyBooleanDomainExpr::Read { label, child } => {
                    if self.prefix_final[node as usize] {
                        nwa.set_final_weight(from, Weight::all());
                    }
                    nwa.add_transition(from, label, remap[&child], Weight::all());
                }
                LazyBooleanDomainExpr::Union { left, right } => {
                    if self.prefix_final[node as usize] {
                        nwa.set_final_weight(from, Weight::all());
                    }
                    nwa.add_epsilon(from, remap[&left], Weight::all());
                    nwa.add_epsilon(from, remap[&right], Weight::all());
                }
            }
        }
        nwa
    }
}


#[derive(Clone)]
struct SharedBooleanDomainNode {
    explicit: BTreeMap<i32, u32>,
    default: Option<u32>,
    accepting: bool,
}

/// Canonical shared DAG for boolean parser-stack prefix predicates.
///
/// Node 0 is the empty language and node 1 is the universal/already-accepted
/// prefix language. All other nodes are hash-consed deterministic rows over
/// parser-state labels plus DEFAULT fallback. This is intentionally a compile-
/// time representation: exported parser DWAs remain ordinary runtime DWAs.
pub struct SharedBooleanParserDomains {
    nodes: Vec<SharedBooleanDomainNode>,
    interner: FxHashMap<(bool, Vec<(i32, u32)>, Option<u32>), u32>,
    union_memo: FxHashMap<(u32, u32), u32>,
    derivative_rows: Vec<Option<Arc<Vec<(i32, u32)>>>>,
    prefix_finality_memo: FxHashMap<u32, u32>,
}

impl Default for SharedBooleanParserDomains {
    fn default() -> Self {
        Self::new()
    }
}

impl SharedBooleanParserDomains {
    pub const EMPTY: u32 = 0;
    pub const UNIVERSAL: u32 = 1;

    pub fn new() -> Self {
        let empty = SharedBooleanDomainNode {
            explicit: BTreeMap::new(),
            default: None,
            accepting: false,
        };
        let universal = SharedBooleanDomainNode {
            explicit: BTreeMap::new(),
            default: None,
            accepting: true,
        };
        let mut interner = FxHashMap::default();
        interner.insert((false, Vec::new(), None), Self::EMPTY);
        interner.insert((true, Vec::new(), None), Self::UNIVERSAL);
        Self {
            nodes: vec![empty, universal],
            interner,
            union_memo: FxHashMap::default(),
            derivative_rows: vec![Some(Arc::new(Vec::new())), Some(Arc::new(Vec::new()))],
            prefix_finality_memo: FxHashMap::default(),
        }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn row_shape_stats(&self) -> (usize, usize, usize, usize, usize, usize) {
        let mut explicit_total = 0usize;
        let mut defaults = 0usize;
        let mut zero = 0usize;
        let mut one = 0usize;
        let mut two = 0usize;
        let mut max_explicit = 0usize;
        for node in &self.nodes {
            let len = node.explicit.len();
            explicit_total += len;
            defaults += usize::from(node.default.is_some());
            zero += usize::from(len == 0);
            one += usize::from(len == 1);
            two += usize::from(len == 2);
            max_explicit = max_explicit.max(len);
        }
        (explicit_total, defaults, zero, one, two, max_explicit)
    }

    pub fn is_empty_root(&self, root: u32) -> bool {
        root == Self::EMPTY
    }

    pub fn is_universal_root(&self, root: u32) -> bool {
        root == Self::UNIVERSAL
    }

    pub fn explicit_labels(&self, root: u32) -> Vec<i32> {
        if root == Self::EMPTY || root == Self::UNIVERSAL {
            return Vec::new();
        }
        self.nodes[root as usize].explicit.keys().copied().collect()
    }

    pub fn advance_default(&self, root: u32) -> u32 {
        if root == Self::EMPTY || root == Self::UNIVERSAL {
            return root;
        }
        self.nodes[root as usize].default.unwrap_or(Self::EMPTY)
    }

    fn make_node(&mut self, mut explicit: BTreeMap<i32, u32>, default: Option<u32>) -> u32 {
        let default = default.filter(|&target| target != Self::EMPTY);
        // DEFAULT remains an additive wildcard branch in this compile-time
        // representation, matching the positive NWA before parser fallback
        // normalization.  Prefix finality still projects backward through a
        // DEFAULT edge.
        if default == Some(Self::UNIVERSAL)
            && std::env::var_os("GLRMASK_EXPERIMENT_DEFER_SHARED_DOMAIN_DEFAULT_FINALITY").is_none()
        {
            return Self::UNIVERSAL;
        }
        explicit.retain(|_, target| {
            if *target == Self::EMPTY {
                return false;
            }
            if Some(*target) == default {
                return false;
            }
            true
        });
        if explicit.is_empty() && default.is_none() {
            return Self::EMPTY;
        }
        let row = explicit.iter().map(|(&label, &target)| (label, target)).collect::<Vec<_>>();
        let key = (false, row, default);
        if let Some(&existing) = self.interner.get(&key) {
            return existing;
        }
        let id = self.nodes.len() as u32;
        self.nodes.push(SharedBooleanDomainNode {
            explicit,
            default,
            accepting: false,
        });
        self.derivative_rows.push(None);
        self.interner.insert(key, id);
        id
    }

    pub fn union(&mut self, left: u32, right: u32) -> u32 {
        if left == right || right == Self::EMPTY {
            return left;
        }
        if left == Self::EMPTY {
            return right;
        }
        if left == Self::UNIVERSAL || right == Self::UNIVERSAL {
            return Self::UNIVERSAL;
        }
        let key = if left < right { (left, right) } else { (right, left) };
        if let Some(&cached) = self.union_memo.get(&key) {
            return cached;
        }
        let left_node = self.nodes[left as usize].clone();
        let right_node = self.nodes[right as usize].clone();
        debug_assert!(!left_node.accepting && !right_node.accepting);

        let default = match (left_node.default, right_node.default) {
            (Some(left), Some(right)) => Some(self.union(left, right)),
            (Some(left), None) => Some(left),
            (None, Some(right)) => Some(right),
            (None, None) => None,
        };
        let mut labels = BTreeSet::new();
        labels.extend(left_node.explicit.keys().copied());
        labels.extend(right_node.explicit.keys().copied());
        let mut explicit = BTreeMap::new();
        for label in labels {
            // Do not fold DEFAULT into explicit labels here.  DEFAULT is a
            // symbolic wildcard NWA branch, so a concrete derivative unions
            // the matching explicit branch with the wildcard branch later.
            let left_child = left_node.explicit.get(&label).copied();
            let right_child = right_node.explicit.get(&label).copied();
            let child = match (left_child, right_child) {
                (Some(left), Some(right)) => self.union(left, right),
                (Some(left), None) => left,
                (None, Some(right)) => right,
                (None, None) => Self::EMPTY,
            };
            explicit.insert(label, child);
        }
        let result = self.make_node(explicit, default);
        self.union_memo.insert(key, result);
        result
    }

    pub fn union_all(&mut self, roots: impl IntoIterator<Item = u32>) -> u32 {
        roots
            .into_iter()
            .fold(Self::EMPTY, |combined, root| self.union(combined, root))
    }

    /// Concrete derivatives for every explicit positive label in a shared
    /// domain row. Each derivative already includes the row's additive DEFAULT
    /// branch, so callers can apply the returned entries as sparse overrides on
    /// top of `advance_default(root)`.
    pub fn explicit_derivatives(&mut self, root: u32) -> Arc<Vec<(i32, u32)>> {
        if let Some(cached) = self
            .derivative_rows
            .get(root as usize)
            .and_then(|slot| slot.as_ref())
        {
            return Arc::clone(cached);
        }
        let node = self.nodes[root as usize].clone();
        let wildcard = node.default.unwrap_or(Self::EMPTY);
        let mut row = Vec::with_capacity(node.explicit.len());
        for (label, explicit) in node.explicit {
            let child = self.union(explicit, wildcard);
            if child != Self::EMPTY {
                row.push((label, child));
            }
        }
        let row = Arc::new(row);
        self.derivative_rows[root as usize] = Some(Arc::clone(&row));
        row
    }

    pub fn advance(&mut self, root: u32, parser_state: u32) -> u32 {
        if root == Self::EMPTY || root == Self::UNIVERSAL {
            return root;
        }
        let row = self.explicit_derivatives(root);
        match row.binary_search_by_key(&(parser_state as i32), |(label, _)| *label) {
            Ok(index) => row[index].1,
            Err(_) => self.advance_default(root),
        }
    }

    pub fn derivative_row_cache_len(&self) -> usize {
        self.derivative_rows.iter().filter(|row| row.is_some()).count()
    }

    /// Exact preimage through one or more grammar terminals whose only
    /// GLR action is `Skip`: the parser stack is unchanged, but the current
    /// top state must be one where at least one of those terminals is legal.
    pub fn preimage_identity_skip(&mut self, target_root: u32, allowed_states: &[u32]) -> u32 {
        if target_root == Self::EMPTY || allowed_states.is_empty() {
            return Self::EMPTY;
        }
        let mut explicit = BTreeMap::new();
        for &parser_state in allowed_states {
            let child = self.advance(target_root, parser_state);
            if child != Self::EMPTY {
                explicit.insert(parser_state as i32, child);
            }
        }
        self.make_node(explicit, None)
    }

    /// Apply parser-stack prefix finality after a complete relational
    /// preimage has been assembled. Finality propagates through DEFAULT stack
    /// reads (negative/push edges have already been cancelled algebraically),
    /// but never backward through concrete positive reads.
    ///
    /// This must not run while individual template fragments are still being
    /// composed: doing so can erase positive-read provenance before the full
    /// stack effect is known. That was the failure mode of the old eager
    /// `DEFAULT -> UNIVERSAL` simplification in `make_node`.
    pub fn finalize_prefix_domain(&mut self, root: u32) -> u32 {
        if root == Self::EMPTY || root == Self::UNIVERSAL {
            return root;
        }
        if let Some(&cached) = self.prefix_finality_memo.get(&root) {
            return cached;
        }
        let node = self.nodes[root as usize].clone();
        let mut explicit = BTreeMap::new();
        for (label, child) in node.explicit {
            let child = self.finalize_prefix_domain(child);
            if child != Self::EMPTY {
                explicit.insert(label, child);
            }
        }
        let default = node.default.map(|child| self.finalize_prefix_domain(child));
        let result = if default == Some(Self::UNIVERSAL) {
            // A DEFAULT edge can consume an arbitrary deeper parser-stack state.
            // If its target has already accepted the visible prefix, the source
            // itself is prefix-final; from this residual point every deeper
            // suffix is irrelevant.
            Self::UNIVERSAL
        } else {
            self.make_node(explicit, default)
        };
        self.prefix_finality_memo.insert(root, result);
        result
    }

    fn prepend(&mut self, label: i32, child: u32) -> u32 {
        if child == Self::EMPTY {
            return Self::EMPTY;
        }
        if label == DEFAULT_LABEL {
            return self.make_node(BTreeMap::new(), Some(child));
        }
        let mut explicit = BTreeMap::new();
        explicit.insert(label, child);
        self.make_node(explicit, None)
    }

    fn negative_suffix_root(
        &mut self,
        bundle: &NWA,
        bundle_state: u32,
        target_root: u32,
        memo: &mut FxHashMap<(u32, u32), Option<u32>>,
    ) -> Option<u32> {
        if let Some(cached) = memo.get(&(bundle_state, target_root)) {
            return *cached;
        }
        let node = bundle.states().get(bundle_state as usize)?;
        let mut result = if node
            .final_weight
            .as_ref()
            .is_some_and(|weight| !weight.is_empty())
        {
            target_root
        } else {
            Self::EMPTY
        };
        for (&label, targets) in &node.transitions {
            if !is_negative_label(label) {
                memo.insert((bundle_state, target_root), None);
                return None;
            }
            let parser_state = negative_to_positive_label(label) as u32;
            for (target, weight) in targets {
                debug_assert!(weight.is_full() || weight.is_empty());
                if weight.is_empty() {
                    continue;
                }
                let Some(child) = self.negative_suffix_root(
                    bundle,
                    *target,
                    target_root,
                    memo,
                ) else {
                    memo.insert((bundle_state, target_root), None);
                    return None;
                };
                let residual = self.advance(child, parser_state);
                result = self.union(result, residual);
            }
        }
        for (target, weight) in &node.epsilons {
            debug_assert!(weight.is_full() || weight.is_empty());
            if weight.is_empty() {
                continue;
            }
            let Some(child) = self.negative_suffix_root(bundle, *target, target_root, memo) else {
                memo.insert((bundle_state, target_root), None);
                return None;
            };
            result = self.union(result, child);
        }
        memo.insert((bundle_state, target_root), Some(result));
        Some(result)
    }

    fn preimage_state(
        &mut self,
        bundle: &NWA,
        bundle_state: u32,
        target_root: u32,
        positive_memo: &mut FxHashMap<(u32, u32), Option<u32>>,
        negative_memo: &mut FxHashMap<(u32, u32), Option<u32>>,
    ) -> Option<u32> {
        if let Some(cached) = positive_memo.get(&(bundle_state, target_root)) {
            return *cached;
        }
        let node = bundle.states().get(bundle_state as usize)?;
        let mut result = if node
            .final_weight
            .as_ref()
            .is_some_and(|weight| !weight.is_empty())
        {
            target_root
        } else {
            Self::EMPTY
        };
        for (&label, targets) in &node.transitions {
            for (target, weight) in targets {
                debug_assert!(weight.is_full() || weight.is_empty());
                if weight.is_empty() {
                    continue;
                }
                let branch = if is_negative_label(label) {
                    let parser_state = negative_to_positive_label(label) as u32;
                    let Some(child) = self.negative_suffix_root(
                        bundle,
                        *target,
                        target_root,
                        negative_memo,
                    ) else {
                        positive_memo.insert((bundle_state, target_root), None);
                        return None;
                    };
                    self.advance(child, parser_state)
                } else {
                    let Some(child) = self.preimage_state(
                        bundle,
                        *target,
                        target_root,
                        positive_memo,
                        negative_memo,
                    ) else {
                        positive_memo.insert((bundle_state, target_root), None);
                        return None;
                    };
                    self.prepend(label, child)
                };
                result = self.union(result, branch);
            }
        }
        for (target, weight) in &node.epsilons {
            debug_assert!(weight.is_full() || weight.is_empty());
            if weight.is_empty() {
                continue;
            }
            let Some(child) = self.preimage_state(
                bundle,
                *target,
                target_root,
                positive_memo,
                negative_memo,
            ) else {
                positive_memo.insert((bundle_state, target_root), None);
                return None;
            };
            result = self.union(result, child);
        }
        positive_memo.insert((bundle_state, target_root), Some(result));
        Some(result)
    }

    pub fn preimage_bundle(&mut self, bundle: &NWA, target_root: u32) -> Option<u32> {
        let mut positive_memo = FxHashMap::default();
        let mut negative_memo = FxHashMap::default();
        let mut result = Self::EMPTY;
        for &start in bundle.start_states() {
            let root = self.preimage_state(
                bundle,
                start,
                target_root,
                &mut positive_memo,
                &mut negative_memo,
            )?;
            result = self.union(result, root);
        }
        if std::env::var_os("GLRMASK_EXPERIMENT_FINALIZE_SHARED_PREIMAGE").is_some() {
            result = self.finalize_prefix_domain(result);
        }
        Some(result)
    }

    pub fn to_dwa(&self, root: u32) -> DWA {
        let mut output = DWA::new(0, 0);
        let mut remap = FxHashMap::<u32, u32>::default();
        remap.insert(root, output.start_state());
        let mut queue = VecDeque::from([root]);
        while let Some(source) = queue.pop_front() {
            let output_source = remap[&source];
            let node = &self.nodes[source as usize];
            if node.accepting {
                output.set_final_weight(output_source, Weight::all());
            }
            for (&label, &target) in &node.explicit {
                let output_target = if let Some(&existing) = remap.get(&target) {
                    existing
                } else {
                    let created = output.add_state();
                    remap.insert(target, created);
                    queue.push_back(target);
                    created
                };
                output.add_transition(output_source, label, output_target, Weight::all());
            }
            if let Some(target) = node.default {
                let output_target = if let Some(&existing) = remap.get(&target) {
                    existing
                } else {
                    let created = output.add_state();
                    remap.insert(target, created);
                    queue.push_back(target);
                    created
                };
                output.add_transition(
                    output_source,
                    DEFAULT_LABEL,
                    output_target,
                    Weight::all(),
                );
            }
        }
        output
    }

    /// Export the shared compile-time row DAG without interpreting DEFAULT as
    /// deterministic fallback.  In this representation DEFAULT is an additive
    /// wildcard NWA branch; parser-specific support/fallback normalization is
    /// deliberately deferred until after the complete graph is assembled.
    pub fn to_nwa(&self, root: u32) -> NWA {
        let mut output = NWA::new(0, 0);
        let start = output.add_state();
        output.set_start_states(vec![start]);
        let mut remap = FxHashMap::<u32, u32>::default();
        remap.insert(root, start);
        let mut queue = VecDeque::from([root]);
        while let Some(source) = queue.pop_front() {
            let output_source = remap[&source];
            let node = &self.nodes[source as usize];
            if node.accepting {
                output.set_final_weight(output_source, Weight::all());
            }
            for (&label, &target) in &node.explicit {
                let output_target = if let Some(&existing) = remap.get(&target) {
                    existing
                } else {
                    let created = output.add_state();
                    remap.insert(target, created);
                    queue.push_back(target);
                    created
                };
                output.add_transition(output_source, label, output_target, Weight::all());
            }
            if let Some(target) = node.default {
                let output_target = if let Some(&existing) = remap.get(&target) {
                    existing
                } else {
                    let created = output.add_state();
                    remap.insert(target, created);
                    queue.push_back(target);
                    created
                };
                output.add_transition(
                    output_source,
                    DEFAULT_LABEL,
                    output_target,
                    Weight::all(),
                );
            }
        }
        output
    }
}




pub fn build_parser_dwa_from_terminal_dwa_with_precomputed_templates(
    table: &GLRTable,
    grammar: &AnalyzedGrammar,
    terminal_dwa: &TerminalAutomaton,
    templates: &Templates,
    _vocab: &Vocab,
    _id_map: &InternalIdMap,
    collapse_immediate_acceptance: bool,
) -> DWA {
    let num_parser_states = table.num_states;
    let total_started_at = Instant::now();
    let minimize_skipped = false;
    let profiling_enabled = compile_profile_enabled();
    let (terminal_dwa_transition_count, terminal_dwa_interned_ranges) = if profiling_enabled {
        let stats = terminal_dwa.stats();
        (stats.transitions, stats.interned_ranges)
    } else {
        (0, 0)
    };
    let Some((mut parser_nwa, parser_nwa_profile)) = build_parser_nwa_from_terminal_dwa(terminal_dwa, grammar, templates, table) else {
        if profiling_enabled {
            eprintln!(
                "[glrmask/profile][parser_dwa_detail] terminal_dwa_states={} terminal_dwa_transitions={} terminal_dwa_interned_ranges={} parser_nwa_built=false pre_minimize_states=0 pre_minimize_transitions=0 post_minimize_states=0 post_minimize_transitions=0 minimize_skipped={} state_prep_ms=0.000 compose_state_ms=0.000 parser_nwa_build_ms=0.000 resolve_negative_ms=0.000 support_determinize_ms=0.000 possible_outgoing_ms=0.000 default_opt_ms=0.000 subtract_final_ms=0.000 fallback_determinize_ms=0.000 minimize_ms=0.000 total_ms={:.3}",
                terminal_dwa.num_states(),
                terminal_dwa_transition_count,
                terminal_dwa_interned_ranges,
                minimize_skipped,
                elapsed_ms(total_started_at),
            );
        }
        return DWA::new(0, 0);
    };

    let resolve_negative_started_at = Instant::now();
    resolve_negative_codes_in_nwa(
        &mut parser_nwa,
        table.construction == GlrTableConstruction::ExperimentalCoreMerged,
    );
    let resolve_negative_ms = elapsed_ms(resolve_negative_started_at);

    let support_determinize_started_at = Instant::now();
    let determinized = determinize_with_supports(&parser_nwa, Some(num_parser_states));
    let support_determinize_ms = elapsed_ms(support_determinize_started_at);
    if std::env::var_os("GLRMASK_VALIDATE_PARSER_SUPPORT_NORMALIZE_SINGLETONS").is_some()
        && std::env::var_os("GLRMASK_PARSER_SUPPORT_NORMALIZE_SINGLETONS").is_some()
    {
        let reference = determinize_with_supports_mode(
            &parser_nwa,
            Some(num_parser_states),
            None,
            Some(false),
            Some(false),
        );
        let difference = find_difference(&determinized.dwa, &reference.dwa)
            .expect("parser support singleton-normalization equivalence checker failed");
        assert!(
            difference.is_none(),
            "normalized parser support DWA differs from weighted-singleton reference on labels {difference:?}",
        );
        eprintln!(
            "[glrmask/validate][parser_support_normalize_singletons] normalized_states={} reference_states={} result=equivalent",
            determinized.dwa.num_states(),
            reference.dwa.num_states(),
        );
    }
    if std::env::var_os("GLRMASK_VALIDATE_PARSER_SUPPORT_NORMALIZE_SUBSETS").is_some()
        && std::env::var_os("GLRMASK_PARSER_SUPPORT_NORMALIZE_SUBSETS").is_some()
    {
        let reference = determinize_with_supports_mode(
            &parser_nwa,
            Some(num_parser_states),
            None,
            Some(false),
            Some(false),
        );
        let difference = find_difference(&determinized.dwa, &reference.dwa)
            .expect("parser support subset-normalization equivalence checker failed");
        assert!(
            difference.is_none(),
            "normalized parser support subsets differ from reference on labels {difference:?}",
        );
        eprintln!(
            "[glrmask/validate][parser_support_normalize_subsets] normalized_states={} reference_states={} result=equivalent",
            determinized.dwa.num_states(),
            reference.dwa.num_states(),
        );
    }
    if std::env::var_os("GLRMASK_VALIDATE_PARSER_SUPPORT_DEFER_EDGE_UNIONS").is_some()
        && parser_support_defer_edge_unions_enabled(parser_nwa.states().len())
    {
        let reference = determinize_with_supports_mode(
            &parser_nwa,
            Some(num_parser_states),
            Some(false),
            None,
            None,
        );
        assert_eq!(
            determinized.supports, reference.supports,
            "deferred parser support unions changed NWA support sets",
        );
        let difference = find_difference(&determinized.dwa, &reference.dwa)
            .expect("parser support equivalence checker failed");
        assert!(
            difference.is_none(),
            "deferred parser support unions changed the weighted language: {difference:?}",
        );
        eprintln!(
            "[glrmask/profile][parser_support_deferred_union_equivalence] result=equivalent"
        );
    }
    let mut parser_dwa_pre_minimize = determinized.dwa;

    let guaranteed_read_started_at = Instant::now();
    let immediate_read_rewrites = if collapse_immediate_acceptance {
        collapse_immediate_acceptance_certificates(
            &mut parser_dwa_pre_minimize,
            terminal_dwa,
            grammar,
            table,
        )
    } else {
        0
    };
    let guaranteed_read_rewrites = immediate_read_rewrites;
    let guaranteed_read_ms = elapsed_ms(guaranteed_read_started_at);

    let possible_outgoing_started_at = Instant::now();
    let possible_by_state = build_possible_outgoing_ids_by_state(
        &parser_nwa,
        &determinized.supports,
        num_parser_states,
    );
    let possible_outgoing_ms = elapsed_ms(possible_outgoing_started_at);

    let default_opt_started_at = Instant::now();
    optimize_parser_dwa_defaults(
        &mut parser_dwa_pre_minimize,
        &possible_by_state,
        num_parser_states,
    );
    let default_opt_ms = elapsed_ms(default_opt_started_at);

    let subtract_final_started_at = Instant::now();
    let validate_parallel_subtraction =
        std::env::var_os("GLRMASK_VALIDATE_PARALLEL_FINAL_SUBTRACTION").is_some();
    let serial_reference = validate_parallel_subtraction.then(|| parser_dwa_pre_minimize.clone());
    subtract_final_weights_from_outgoing_dwa(&mut parser_dwa_pre_minimize);
    if let Some(mut serial_reference) = serial_reference {
        subtract_final_weights_from_outgoing_dwa_impl(&mut serial_reference, false);
        assert_eq!(
            parser_dwa_pre_minimize.start_state(),
            serial_reference.start_state(),
            "parallel final subtraction changed the DWA start state",
        );
        assert_eq!(
            parser_dwa_pre_minimize.states().len(),
            serial_reference.states().len(),
            "parallel final subtraction changed the DWA state count",
        );
        for (parallel_state, serial_state) in parser_dwa_pre_minimize
            .states()
            .iter()
            .zip(serial_reference.states())
        {
            assert_eq!(
                parallel_state.final_weight, serial_state.final_weight,
                "parallel final subtraction changed a final weight",
            );
            assert_eq!(
                parallel_state.transitions, serial_state.transitions,
                "parallel final subtraction changed a transition row",
            );
        }
    }
    let subtract_final_ms = elapsed_ms(subtract_final_started_at);

    let fallback_determinize_started_at = Instant::now();
    let normalized_fallback_started_at = Instant::now();
    let normalized_fallback = determinize_parser_dwa_with_fallbacks(
        &parser_dwa_pre_minimize,
        &possible_by_state,
        num_parser_states,
    );
    let normalized_fallback_ms = elapsed_ms(normalized_fallback_started_at);
    if std::env::var_os("GLRMASK_VALIDATE_FALLBACK_SINGLETON_NORMALIZATION").is_some() {
        let reference_started_at = Instant::now();
        let reference = determinize_parser_dwa_with_fallbacks_impl(
            &parser_dwa_pre_minimize,
            &possible_by_state,
            num_parser_states,
            false,
        );
        let reference_ms = elapsed_ms(reference_started_at);
        let equivalence_started_at = Instant::now();
        let difference = find_difference(&normalized_fallback, &reference)
            .expect("fallback singleton validation requires acyclic parser DWAs");
        let equivalence_ms = elapsed_ms(equivalence_started_at);
        assert!(
            difference.is_none(),
            "normalized parser fallback DWA differs from the legacy weighted-singleton DWA on labels {:?}",
            difference,
        );
        if profiling_enabled {
            eprintln!(
                "[glrmask/profile][fallback_singleton_normalization] normalized_states={} normalized_transitions={} reference_states={} reference_transitions={} normalized_ms={normalized_fallback_ms:.3} reference_ms={reference_ms:.3} equivalence_ms={equivalence_ms:.3} result=equivalent",
                normalized_fallback.num_states(),
                normalized_fallback.num_transitions(),
                reference.num_states(),
                reference.num_transitions(),
            );
        }
    }
    parser_dwa_pre_minimize = normalized_fallback;
    if collapse_immediate_acceptance {
        parser_dwa_pre_minimize = collapse_final_leaf_targets(parser_dwa_pre_minimize);
    }
    let fallback_determinize_ms = elapsed_ms(fallback_determinize_started_at);

    let pre_minimize_state_count = parser_dwa_pre_minimize.states().len();
    let pre_minimize_transition_count = parser_dwa_pre_minimize.num_transitions();
    let minimize_skipped = should_skip_parser_dwa_minimization(
        pre_minimize_state_count,
        pre_minimize_transition_count,
    );
    let (minimized, minimize_ms, post_minimize_state_count, post_minimize_transition_count) =
        if minimize_skipped {
            (
                parser_dwa_pre_minimize,
                0.0,
                pre_minimize_state_count,
                pre_minimize_transition_count,
            )
        } else {
            let minimize_started_at = Instant::now();
            let minimized = minimize(&parser_dwa_pre_minimize);
            let minimize_ms = elapsed_ms(minimize_started_at);
            let post_minimize_state_count = minimized.states().len();
            let post_minimize_transition_count = minimized.num_transitions();
            (
                minimized,
                minimize_ms,
                post_minimize_state_count,
                post_minimize_transition_count,
            )
        };

    if profiling_enabled {
        eprintln!(
            "[glrmask/profile][parser_dwa_detail] terminal_dwa_states={} terminal_dwa_transitions={} terminal_dwa_interned_ranges={} parser_nwa_states={} parser_nwa_start_states={} pre_minimize_states={} pre_minimize_transitions={} post_minimize_states={} post_minimize_transitions={} minimize_skipped={} state_prep_ms={:.3} compose_state_ms={:.3} parser_nwa_build_ms={:.3} resolve_negative_ms={:.3} support_determinize_ms={:.3} guaranteed_read_rewrites={} guaranteed_read_ms={:.3} possible_outgoing_ms={:.3} default_opt_ms={:.3} subtract_final_ms={:.3} fallback_determinize_ms={:.3} minimize_ms={:.3} total_ms={:.3}",
            terminal_dwa.num_states(),
            terminal_dwa_transition_count,
            terminal_dwa_interned_ranges,
            parser_nwa.states().len(),
            parser_nwa.start_states().len(),
            pre_minimize_state_count,
            pre_minimize_transition_count,
            post_minimize_state_count,
            post_minimize_transition_count,
            minimize_skipped,
            parser_nwa_profile.state_prep_ms,
            parser_nwa_profile.compose_state_ms,
            parser_nwa_profile.parser_nwa_build_ms,
            resolve_negative_ms,
            support_determinize_ms,
            guaranteed_read_rewrites,
            guaranteed_read_ms,
            possible_outgoing_ms,
            default_opt_ms,
            subtract_final_ms,
            fallback_determinize_ms,
            minimize_ms,
            elapsed_ms(total_started_at),
        );
    }

    minimized
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use range_set_blaze::RangeSetBlaze;
    use rustc_hash::FxHashMap;

    use super::{
        PossibleOutgoingIds, build_parser_nwa_from_terminal_dwa,
        collapse_final_leaf_targets, determinize_parser_dwa_with_fallbacks,
        determinize_with_supports, immediate_acceptance_certificates,
        local_epsilon_closure, local_epsilon_closure_canonical,
        try_build_direct_regular_parser_top_accept_parts,
        try_build_direct_regular_parser_top_accept_parts_table_product_reference,
        try_build_immediate_parser_top_accept_parts,
        subtract_final_weights_from_outgoing_dwa_impl,
    };
    use crate::automata::weighted::dwa::DWA;
    use crate::automata::weighted::nwa::NWA;
    use crate::automata::weighted::terminal_automaton::TerminalAutomaton;
    use crate::compiler::glr::analysis::AnalyzedGrammar;
    use crate::compiler::glr::labels::DEFAULT_LABEL;
    use crate::compiler::glr::table::testing::build_test_table;
    use crate::compiler::glr::table::Action;
    use crate::compiler::stages::resolve_negatives::resolve_negative_codes_in_nwa;
    use crate::compiler::stages::templates::Templates;
    use crate::ds::weight::Weight;
    use crate::grammar::flat::{
        DirectRegularAutomaton, GrammarDef, Rule, Symbol, Terminal,
    };

    #[test]
    fn extended_boundary_weight_rows_preserve_exact_finite_algebra() {
        for rows in [1usize, 16, 17, 64] {
            let mut interner = super::FastBoundaryWeightInterner::new(rows, 64).unwrap();
            let domain = Weight::from_uniform(0..=rows as u32 - 1, RangeSetBlaze::from_iter([0..=63]));
            let mut sources = vec![Weight::empty(), Weight::all()];
            for i in 0..24u32 {
                sources.push(Weight::from_per_tsid_token_sets((0..rows as u32).filter_map(|row| {
                    let tokens = (0..64).filter(|&token| (token + 3 * row + i) % 11 < (i % 7) + 1).collect::<RangeSetBlaze<u32>>();
                    (!tokens.is_empty()).then_some((row, tokens))
                })));
            }
            let mut by_ptr = rustc_hash::FxHashMap::default();
            let ids = sources.iter().map(|weight| interner.source_weight_id(weight, &mut by_ptr, None).unwrap()).collect::<Vec<_>>();
            for (a, &left) in sources.iter().zip(&ids) {
                assert_eq!(domain.intersection(&interner.to_weight(left)), domain.intersection(a));
                for (b, &right) in sources.iter().zip(&ids) {
                    let union = interner.union(left, right);
                    let intersection = interner.intersection(left, right);
                    let difference = interner.difference(left, right);
                    assert_eq!(domain.intersection(&interner.to_weight(union)), domain.intersection(&a.union(b)));
                    assert_eq!(domain.intersection(&interner.to_weight(intersection)), domain.intersection(&a.intersection(b)));
                    assert_eq!(domain.intersection(&interner.to_weight(difference)), domain.intersection(a).difference(&domain.intersection(b)));
                }
            }
            assert_eq!(interner.compact_runtime_weights().is_some(), rows <= 16,
                "large compile domains must not be truncated into the 16-row runtime format");
        }
        assert!(super::FastBoundaryWeightInterner::new(65, 64).is_none());
    }

    fn weight(tokens: std::ops::RangeInclusive<u32>) -> Weight {
        Weight::from_token_set_for_tsid(0, RangeSetBlaze::from_iter([tokens]))
    }

    fn eval_with_default(dwa: &DWA, word: &[i32]) -> Weight {
        let mut state_id = dwa.start_state();
        let mut accumulated = Weight::all();
        for &label in word {
            let Some((target, edge_weight)) = dwa.states()[state_id as usize]
                .transitions
                .get(&label)
                .or_else(|| dwa.states()[state_id as usize].transitions.get(&DEFAULT_LABEL))
            else {
                return Weight::empty();
            };
            accumulated = accumulated.intersection(edge_weight);
            if accumulated.is_empty() {
                return accumulated;
            }
            state_id = *target;
        }
        dwa.states()[state_id as usize]
            .final_weight
            .as_ref()
            .map_or_else(Weight::empty, |final_weight| {
                accumulated.intersection(final_weight)
            })
    }

    #[test]
    fn flat_canonical_epsilon_closure_matches_map_reference() {
        fn next_u32(state: &mut u64) -> u32 {
            *state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (*state >> 32) as u32
        }

        const STATE_COUNT: usize = 64;
        let mut random = 0x8f4d_3a2b_1907_65ceu64;
        let mut nwa = NWA::new(0, 0);
        for _ in 0..STATE_COUNT {
            nwa.add_state();
        }
        for source in 0..STATE_COUNT - 1 {
            let remaining = STATE_COUNT - source - 1;
            let edge_count = 1 + next_u32(&mut random) as usize % remaining.min(4);
            for _ in 0..edge_count {
                let target = source + 1 + next_u32(&mut random) as usize % remaining;
                let start = next_u32(&mut random) % 24;
                let end = (start + next_u32(&mut random) % 8).min(31);
                nwa.add_epsilon(source as u32, target as u32, weight(start..=end));
            }
        }

        for case in 0..256 {
            let mut seeds = FxHashMap::<u32, Weight>::default();
            let seed_count = 1 + next_u32(&mut random) as usize % 8;
            for _ in 0..seed_count {
                let state = next_u32(&mut random) as usize % STATE_COUNT;
                let start = next_u32(&mut random) % 24;
                let end = (start + next_u32(&mut random) % 8).min(31);
                let add = weight(start..=end);
                seeds
                    .entry(state as u32)
                    .and_modify(|existing| *existing = existing.union(&add))
                    .or_insert(add);
            }

            let mut reference = seeds.clone();
            let mut reference_weights = vec![None; STATE_COUNT];
            let mut reference_queue = VecDeque::new();
            local_epsilon_closure(
                &nwa,
                &mut reference_weights,
                &mut reference_queue,
                &mut reference,
            );
            let mut reference_canonical = reference.into_iter().collect::<Vec<_>>();
            reference_canonical.sort_unstable_by_key(|(state, _)| *state);

            let mut seed_canonical = seeds.into_iter().collect::<Vec<_>>();
            seed_canonical.sort_unstable_by_key(|(state, _)| *state);
            let mut flat_weights = vec![None; STATE_COUNT];
            let mut flat_queue = VecDeque::new();
            let mut touched = Vec::new();
            let mut flat_canonical = Vec::new();
            local_epsilon_closure_canonical(
                &nwa,
                &mut flat_weights,
                &mut flat_queue,
                &seed_canonical,
                &mut touched,
                &mut flat_canonical,
                &mut super::ScopedWeightOpCache::default(),
                true,
            );

            assert_eq!(
                flat_canonical, reference_canonical,
                "epsilon-closure mismatch in generated case {case}",
            );
            assert!(flat_weights.iter().all(Option::is_none));
            assert!(flat_queue.is_empty());
        }
    }

    #[test]
    fn immediate_acceptance_parts_union_matches_combined_certificates() {
        let mut terminal_dwa = DWA::new(1, 31);
        let accept_a = terminal_dwa.add_state();
        let accept_b = terminal_dwa.add_state();
        let accept_c = terminal_dwa.add_state();
        terminal_dwa.set_final_weight(accept_a, Weight::all());
        terminal_dwa.set_final_weight(accept_b, Weight::all());
        terminal_dwa.set_final_weight(accept_c, Weight::all());
        terminal_dwa.add_transition(
            terminal_dwa.start_state(),
            0,
            accept_a,
            weight(0..=7),
        );
        terminal_dwa.add_transition(
            terminal_dwa.start_state(),
            1,
            accept_b,
            weight(4..=15),
        );
        terminal_dwa.add_transition(
            terminal_dwa.start_state(),
            2,
            accept_c,
            weight(12..=23),
        );
        let terminal_automaton = TerminalAutomaton::Dwa(terminal_dwa);

        let grammar = AnalyzedGrammar::from_grammar_def(&GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            start: 0,
            terminals: (0..3)
                .map(|id| Terminal::Literal {
                    id,
                    bytes: vec![b'a' + id as u8],
                })
                .collect(),
            ..GrammarDef::default()
        });
        let table = build_test_table(
            3,
            3,
            &[
                &[
                    (0, Action::Shift(0, true)),
                    (1, Action::Shift(1, true)),
                ],
                &[
                    (1, Action::Shift(1, true)),
                    (2, Action::Shift(2, true)),
                ],
                &[],
            ],
            &[&[], &[], &[]],
        );

        let parts = try_build_immediate_parser_top_accept_parts(
            &terminal_automaton,
            &grammar,
            &table,
        )
        .expect("test terminal automaton is an immediate-completion family");
        let combined = immediate_acceptance_certificates(&terminal_automaton, &grammar, &table);

        assert_eq!(parts.get(&0).map(Vec::len), Some(2));
        assert_eq!(parts.get(&1).map(Vec::len), Some(2));
        assert!(!parts.contains_key(&2));
        for parser_top in 0..table.num_states {
            let parts_union = parts
                .get(&(parser_top as i32))
                .map(|weights| Weight::union_all(weights.iter()))
                .unwrap_or_else(Weight::empty);
            assert_eq!(parts_union, combined[parser_top as usize]);
        }
    }

    #[test]
    fn direct_regular_parser_product_matches_generic_parser_dwa() {
        let mut terminal_dwa = DWA::new(1, 31);
        let after_zero = terminal_dwa.add_state();
        let accept = terminal_dwa.add_state();
        terminal_dwa.set_final_weight(after_zero, weight(0..=7));
        terminal_dwa.set_final_weight(accept, Weight::all());
        terminal_dwa.add_transition(
            terminal_dwa.start_state(),
            0,
            after_zero,
            weight(0..=15),
        );
        terminal_dwa.add_transition(after_zero, 0, after_zero, weight(8..=12));
        terminal_dwa.add_transition(after_zero, 1, accept, weight(4..=20));
        terminal_dwa.add_transition(
            terminal_dwa.start_state(),
            2,
            accept,
            weight(16..=23),
        );
        let terminal_automaton = TerminalAutomaton::Dwa(terminal_dwa);

        let grammar = AnalyzedGrammar::from_grammar_def(&GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            start: 0,
            terminals: (0..3)
                .map(|id| Terminal::Literal {
                    id,
                    bytes: vec![b'a' + id as u8],
                })
                .collect(),
            direct_regular_automaton: Some(DirectRegularAutomaton {
                states: vec![
                    crate::grammar::flat::DirectRegularState {
                        is_accepting: false,
                        transitions: [
                            (0, vec![0]),
                            (1, vec![1]),
                            (2, vec![1]),
                        ]
                        .into_iter()
                        .collect(),
                        epsilons: Vec::new(),
                    },
                    crate::grammar::flat::DirectRegularState {
                        is_accepting: true,
                        transitions: Default::default(),
                        epsilons: Vec::new(),
                    },
                ],
                start_states: vec![0],
            }),
            ..GrammarDef::default()
        });
        let table = build_test_table(
            3,
            3,
            &[
                &[
                    (0, Action::Shift(1, true)),
                    (1, Action::Shift(2, true)),
                    (2, Action::Shift(2, true)),
                ],
                &[
                    (0, Action::Shift(1, true)),
                    (1, Action::Shift(2, true)),
                    (2, Action::Shift(2, true)),
                ],
                &[],
            ],
            &[&[], &[], &[]],
        );
        let templates = Templates::from_direct_regular_table(&table, grammar.num_terminals)
            .expect("test table has direct-regular actions");
        let (mut generic_nwa, _) = build_parser_nwa_from_terminal_dwa(
            &terminal_automaton,
            &grammar,
            &templates,
            &table,
        )
        .expect("generic parser NWA should build for direct templates");
        resolve_negative_codes_in_nwa(&mut generic_nwa, false);
        let generic = determinize_with_supports(&generic_nwa, Some(table.num_states)).dwa;
        let direct = try_build_direct_regular_parser_top_accept_parts(
            &terminal_automaton,
            &grammar,
            &table,
        )
        .expect("sparse direct product should accept the direct parser metadata");
        let table_reference =
            try_build_direct_regular_parser_top_accept_parts_table_product_reference(
                &terminal_automaton,
                &grammar,
                &table,
            )
            .expect("table-product reference should accept the direct parser table");

        for parser_top in 0..table.num_states {
            let direct_weight = direct
                .get(&(parser_top as i32))
                .map(|weights| Weight::union_all(weights.iter()))
                .unwrap_or_else(Weight::empty);
            let reference_weight = table_reference
                .get(&(parser_top as i32))
                .map(|weights| Weight::union_all(weights.iter()))
                .unwrap_or_else(Weight::empty);
            assert_eq!(
                direct_weight,
                reference_weight,
                "sparse and table products differ at parser top {parser_top}",
            );
            let mut generic_prefix_weight = generic
                .states()
                .get(generic.start_state() as usize)
                .and_then(|state| state.final_weight.clone())
                .unwrap_or_else(Weight::empty);
            generic_prefix_weight = generic_prefix_weight.union(
                &generic.eval_word(&[parser_top as i32]),
            );
            assert_eq!(
                direct_weight,
                generic_prefix_weight,
                "direct product mismatch at parser top {parser_top}",
            );
        }
    }

    #[test]
    fn parallel_final_subtraction_matches_serial_rows() {
        let mut source = DWA::new(1, 31);
        let left = source.add_state();
        let right = source.add_state();
        source.set_final_weight(source.start_state(), weight(4..=11));
        source.add_transition(source.start_state(), 1, left, weight(0..=15));
        source.add_transition(source.start_state(), 2, right, weight(8..=20));
        source.set_final_weight(left, weight(0..=3));
        source.add_transition(left, 3, right, weight(0..=9));
        source.set_final_weight(right, weight(16..=23));
        source.add_transition(right, 4, left, weight(12..=27));

        let mut serial = source.clone();
        let mut parallel = source;
        subtract_final_weights_from_outgoing_dwa_impl(&mut serial, false);
        subtract_final_weights_from_outgoing_dwa_impl(&mut parallel, true);

        assert_eq!(serial.start_state(), parallel.start_state());
        assert_eq!(serial.states().len(), parallel.states().len());
        for (serial_state, parallel_state) in serial.states().iter().zip(parallel.states()) {
            assert_eq!(serial_state.final_weight, parallel_state.final_weight);
            assert_eq!(serial_state.transitions, parallel_state.transitions);
        }
    }

    #[test]
    fn fallback_determinization_reuses_singleton_source_states() {
        let mut source = DWA::new(1, 31);
        let middle = source.add_state();
        let leaf = source.add_state();
        source.add_transition(source.start_state(), 10, middle, weight(0..=15));
        source.add_transition(source.start_state(), 11, middle, weight(8..=23));
        source.set_final_weight(middle, weight(4..=19));
        source.add_transition(middle, 12, leaf, weight(2..=27));
        source.set_final_weight(leaf, weight(6..=25));

        let possible = (0..source.states().len())
            .map(|_| PossibleOutgoingIds::Empty)
            .collect::<Vec<_>>();
        let determinized = determinize_parser_dwa_with_fallbacks(&source, &possible, 32);

        for word in [
            vec![],
            vec![10],
            vec![11],
            vec![10, 12],
            vec![11, 12],
            vec![12],
        ] {
            assert_eq!(
                determinized.eval_word(&word),
                source.eval_word(&word),
                "word={word:?}",
            );
        }
        assert_eq!(determinized.states().len(), source.states().len());
    }

    #[test]
    fn fallback_determinization_combines_explicit_and_default_branches() {
        let mut source = DWA::new(1, 31);
        let explicit = source.add_state();
        let fallback = source.add_state();
        source.add_transition(source.start_state(), 7, explicit, weight(0..=15));
        source.add_transition(
            source.start_state(),
            DEFAULT_LABEL,
            fallback,
            weight(8..=23),
        );
        source.set_final_weight(explicit, weight(0..=5));
        source.set_final_weight(fallback, weight(12..=27));

        let possible = vec![
            PossibleOutgoingIds::All,
            PossibleOutgoingIds::Empty,
            PossibleOutgoingIds::Empty,
        ];
        let determinized = determinize_parser_dwa_with_fallbacks(&source, &possible, 32);

        let explicit_result = weight(0..=5);
        let fallback_result = weight(12..=23);
        assert_eq!(
            eval_with_default(&determinized, &[7]),
            explicit_result.union(&fallback_result),
        );
        assert_eq!(eval_with_default(&determinized, &[8]), fallback_result);
        assert_eq!(determinized.eval_word(&[DEFAULT_LABEL]), weight(12..=23));
    }

    #[test]
    fn final_leaf_weights_are_pushed_into_shared_sink_edges() {
        let mut dwa = DWA::new(1, 5);
        let left = dwa.add_state();
        let right = dwa.add_state();
        dwa.add_transition(0, 10, left, weight(0..=5));
        dwa.add_transition(0, 11, right, weight(0..=5));
        dwa.set_final_weight(left, weight(0..=2));
        dwa.set_final_weight(right, weight(3..=4));

        let collapsed = collapse_final_leaf_targets(dwa);

        assert_eq!(collapsed.states().len(), 2);
        assert_eq!(collapsed.eval_word(&[10]), weight(0..=2));
        assert_eq!(collapsed.eval_word(&[11]), weight(3..=4));
        let targets: Vec<u32> = collapsed.states()[collapsed.start_state() as usize]
            .transitions
            .values()
            .map(|(target, _)| *target)
            .collect();
        assert_eq!(targets.len(), 2);
        assert!(targets.iter().all(|target| *target == targets[0]));
        assert!(collapsed.states()[targets[0] as usize]
            .final_weight
            .as_ref()
            .is_some_and(Weight::is_full));
    }

    #[test]
    fn nonleaf_continuations_are_not_shortened() {
        let mut dwa = DWA::new(1, 5);
        let middle = dwa.add_state();
        let leaf = dwa.add_state();
        dwa.add_transition(0, 10, middle, weight(0..=5));
        dwa.add_transition(middle, 11, leaf, weight(0..=5));
        dwa.set_final_weight(leaf, weight(1..=3));

        let collapsed = collapse_final_leaf_targets(dwa);

        assert_eq!(collapsed.states().len(), 3);
        assert!(collapsed.eval_word(&[10]).is_empty());
        assert_eq!(collapsed.eval_word(&[10, 11]), weight(1..=3));
    }
}

#[cfg(test)]
#[test]
fn finite_native_resource_declines_without_publishing_or_mutating() {
    let mut input = NWA::new(0, 0);
    input.add_state(); input.add_state(); input.set_start_states(vec![0]);
    input.add_transition(0, 0, 1, Weight::from_token_set_for_tsid(0, [1,3].into_iter().collect()));
    input.set_final_weight(1, Weight::all());
    let snapshot = format!("{input:?}");
    // Call after tiny-graph singleton gate is made explicit by a large arena;
    // no environment mutation is needed for this independent decline check.
    for _ in 0..4096 { input.add_state(); }
    let limits = FiniteCompileLimits { weights: 2, ..Default::default() };
    assert!(normalize_finite_impl(&input, 4, 1, 4, None, true, limits).is_none());
    assert_eq!(input.states()[0].transitions[&0][0].0, 1);
    let mut cyclic = input.clone(); cyclic.add_transition(1, 1, 0, Weight::all());
    assert!(normalize_finite_impl(&cyclic, 4, 1, 4, None, true, Default::default()).is_none());
    let mut malformed = input.clone(); malformed.set_start_states(vec![u32::MAX]);
    assert!(normalize_finite_impl(&malformed, 4, 1, 4, None, true, Default::default()).is_none());
    assert!(!snapshot.is_empty());
}

#[cfg(test)]
#[test]
fn borrowed_finite_weight_intern_preserves_ids_algebra_and_resource_limits(){
 let mut seed=7654321u64;let mut next=||{seed=seed.wrapping_mul(6364136223846793005).wrapping_add(1);seed};
 for rows in [1,16,17,44,64]{for bits in [1,17,63,64]{for cap in [3,32,100_000]{
  let mut a=FastBoundaryWeightInterner::new(rows,bits).unwrap();let mut b=FastBoundaryWeightInterner::new(rows,bits).unwrap();
  a.borrowed_intern=false;b.borrowed_intern=true;
  a.limits=Some(FiniteCompileLimits{weights:cap,..Default::default()});b.limits=a.limits;
  for _ in 0..32{let value=(0..rows).map(|_|next()&a.all_token_mask).collect::<FastBoundaryWeightValue>();
   let x=a.intern(value.clone());let y=b.intern(value);assert_eq!(x,y);}
  for _ in 0..2000{let x=next()as usize%a.values.len();let y=next()as usize%a.values.len();let op=next()%3;
   let l=match op{0=>a.intersection(x as u32,y as u32),1=>a.union(x as u32,y as u32),_=>a.difference(x as u32,y as u32)};
   let r=match op{0=>b.intersection(x as u32,y as u32),1=>b.union(x as u32,y as u32),_=>b.difference(x as u32,y as u32)};
   assert_eq!(l,r);assert_eq!(a.failed,b.failed);assert_eq!(a.values.len(),b.values.len());assert_eq!(a.values[l as usize],b.values[r as usize]);
  }
  assert_eq!(a.values,b.values);assert_eq!(a.intersections,b.intersections);assert_eq!(a.unions,b.unions);assert_eq!(a.differences,b.differences);
 }}}
}


#[cfg(test)]
mod checked_topology_reuse_tests {
    use super::*;

    fn pool() -> FastBoundaryWeightInterner {
        let mut p = FastBoundaryWeightInterner::new(2, 64).unwrap();
        for word in [1u64, 3, 5, 10, 31, 63] {
            p.intern(smallvec::smallvec![word, word << 1]);
        }
        p
    }
    fn rows(states: &[FastBoundaryNwaState], p: &FastBoundaryWeightInterner)
        -> Vec<(Weight, Vec<(u32, Weight)>, Vec<(i32, Vec<(u32, Weight)>)>)>
    {
        states.iter().map(|s| (
            p.to_weight(s.final_weight),
            s.epsilons.iter().map(|&(q,w)|(q,p.to_weight(w))).collect(),
            s.transitions.iter().map(|(l,bs)|(*l,bs.iter().map(|&(q,w)|(q,p.to_weight(w))).collect())).collect(),
        )).collect()
    }
    #[test]
    fn reused_topology_matches_fresh_cancellation_support_and_normalizer() {
        let mut seed = 912743u64;
        let mut next = || { seed=seed.wrapping_mul(6364136223846793005).wrapping_add(1); (seed>>32) as usize };
        let domain = FiniteParserReadSupport::new_checked(3, 0,
            &[vec![(0,1),(1,1),(2,2)],vec![(0,1)],vec![(2,2)]], &[true,true,true], true).unwrap();
        for case in 0..512 {
            let n = 4 + next()%10;
            let mut permutation = (0..n as u32).collect::<Vec<_>>();
            for q in (1..n).rev() { let r=next()%(q+1); permutation.swap(q,r); }
            let mut original = (0..n).map(|_| FastBoundaryNwaState {
                final_weight:0, epsilons:Vec::new(), transitions:Vec::new(),
            }).collect::<Vec<_>>();
            for q in 0..n {
                let row=&mut original[permutation[q] as usize];
                row.final_weight=(next()%8) as u32;
                let mut labels=vec![0,1,2,DEFAULT_LABEL,crate::compiler::glr::labels::encode_negative_label(0),crate::compiler::glr::labels::encode_negative_label(1)];
                labels.sort_unstable();
                for label in labels {
                    let mut branches=SmallVec::new();
                    for target in q+1..n { if next()%5==0 { branches.push((permutation[target],(next()%8)as u32)); } }
                    if !branches.is_empty() || next()%3==0 { row.transitions.push((label,branches)); }
                }
                for target in q+1..n { if next()%5==0 {row.epsilons.push((permutation[target],(next()%8)as u32));} }
            }
            let order=CheckedNativeTopology::from_order(fast_boundary_topological_order(&original).unwrap()).unwrap();
            assert!(order.certifies(&original));
            let mut left_pool=pool();let mut right_pool=pool();
            let (left,_) = finite_cancellation::compute(&original,&mut left_pool).unwrap();
            let (right,_) = finite_cancellation::compute_with_topology(&original,&mut right_pool,Some(&order)).unwrap();
            for (a,b) in left.iter().zip(&right) {
                let mut x=Vec::new();let mut y=Vec::new();
                a.for_each(|q,w|x.push((q,left_pool.to_weight(w))));
                b.for_each(|q,w|y.push((q,right_pool.to_weight(w))));
                x.sort_unstable_by_key(|p|p.0);y.sort_unstable_by_key(|p|p.0);assert_eq!(x,y,"derived case{case}");
            }
            let mut left=original.clone();let mut right=original;
            fast_boundary_resolve_negative_codes(&mut left,&mut left_pool).unwrap();
            fast_boundary_resolve_negative_codes_with_topology(&mut right,&mut right_pool,Some(&order)).unwrap();
            assert_eq!(rows(&left,&left_pool),rows(&right,&right_pool),"resolved case{case}");
            assert!(order.certifies(&right));
            finite_read_support::restrict(&mut left,&[permutation[0]],&domain).unwrap();
            finite_read_support::restrict_with_topology(&mut right,&[permutation[0]],&domain,Some(&order)).unwrap();
            assert_eq!(rows(&left,&left_pool),rows(&right,&right_pool),"support case{case}");
            let mut normalized=Vec::new();
            for (source,pool) in [(&left,&mut left_pool),(&right,&mut right_pool)] {
                match determinize_preconverted_small_boundary_output(source,&[permutation[0]],3,pool,8,0.,Instant::now(),false,true).unwrap() {
                    SmallBoundaryDeterminizeOutput::Finite(x)=>normalized.push(x.to_generic_dwa()), _=>unreachable!(),
                }
            }
            let result=crate::parser_equivalence::compare_parser_mask_prefix_languages(&normalized[0],&normalized[1],3,100_000).unwrap();
            assert!(result.difference.is_none(),"normalized case{case}: {:?}",result.difference);
        }
    }
    #[test]
    fn reused_topology_declines_bad_permutations_and_incompatible_graphs() {
        assert!(CheckedNativeTopology::from_order(vec![0,0]).is_none());
        assert!(CheckedNativeTopology::from_order(vec![0,2]).is_none());
        let order=CheckedNativeTopology::from_order(vec![0,1]).unwrap();
        let mut graph=vec![FastBoundaryNwaState{final_weight:0,epsilons:vec![(1,1)],transitions:vec![]},
            FastBoundaryNwaState{final_weight:1,epsilons:vec![(0,0)],transitions:vec![]}];
        assert!(order.certifies(&graph));
        graph[1].epsilons[0].1=1;
        assert!(!order.certifies(&graph));
        let mut p=pool();let before=rows(&graph,&p);
        assert!(fast_boundary_resolve_negative_codes_with_topology(&mut graph,&mut p,Some(&order)).is_none());
        assert_eq!(before,rows(&graph,&p));
        graph[1].epsilons[0]=(u32::MAX,0);
        assert!(!order.certifies(&graph));
    }
}

#[cfg(test)]
#[test]
fn exact_guard_class_union_preserves_empty_rows_full_explicit_and_wildcards(){
    fn observe(v:&FastPossibleOutgoingIds,n:u32)->(bool,bool,usize,Vec<u32>){
        let mut out=Vec::new();v.for_each(n,|q|out.push(q));
        (matches!(v,FastPossibleOutgoingIds::All),matches!(v,FastPossibleOutgoingIds::Empty),v.count(n),out)
    }
    let mut seed=8128u64;let mut next=||{seed=seed.wrapping_mul(6364136223846793005).wrapping_add(1);(seed>>32)as usize};
    for alphabet in [1u32,4,17,65,129]{
        for _case in 0..128 {
            let mut nwa=Vec::new();
            for q in 0..24u32 {
                let mut transitions=Vec::new();
                for label in 0..alphabet {
                    if q==0||next()%4==0{
                        let branches=match next()%3{0=>SmallVec::new(),1=>smallvec::smallvec![(0,0)],_=>smallvec::smallvec![(0,1)]};
                        transitions.push((label as i32,branches));
                    }
                }
                if q==1{transitions.push((DEFAULT_LABEL,SmallVec::new()));}
                nwa.push(FastBoundaryNwaState{final_weight:0,epsilons:Vec::new(),transitions});
            }
            let mut supports=vec![vec![],vec![0],vec![0,0],vec![1],vec![999]];
            for _ in 0..100{supports.push((0..next()%10).map(|_|next()as u32%26).collect());}
            let old=fast_boundary_possible_outgoing_ids_reference(&nwa,&supports,alphabet);
            let new=fast_boundary_possible_outgoing_ids_cached(&nwa,&supports,alphabet);
            assert_eq!(old.len(),new.len());
            for(a,b)in old.iter().zip(&new){assert_eq!(observe(a,alphabet),observe(b,alphabet));}
            validate_guard_observer(&nwa,&supports,alphabet,&new);
            for(cap,words)in [(0,0),(1,0),(1,16),(4,128)]{
                let bounded=fast_boundary_possible_outgoing_ids_cached_with_budget(&nwa,&supports,alphabet,cap,words);
                for(a,b)in old.iter().zip(&bounded){assert_eq!(observe(a,alphabet),observe(b,alphabet));}
                validate_guard_observer(&nwa,&supports,alphabet,&bounded);
            }
            assert!(!matches!(new[1],FastPossibleOutgoingIds::All));
            assert!(matches!(new[2],FastPossibleOutgoingIds::All));
            assert!(matches!(new[3],FastPossibleOutgoingIds::All));
        }
    }
}
