//! Product DFA states, byte classes, construction and trace metadata.

use crate::automata::lexer::ast::Expr;
use crate::automata::lexer::dfa::DFA;
use crate::automata::lexer::tokenizer::CompressedTransitionSegment;
use crate::ds::bitset::BitSet;
use crate::ds::u8set::U8Set;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::hash::Hash;
use std::sync::Arc;
use std::time::Instant;
use super::bounded_repeat::{
    RepeatBaseDfaCache,
    cached_direct_bounded_repeat_base_dfa,
    compile_direct_bounded_repeat_base_dfa_unconditionally,
};
use super::deferred::try_build_dense_binary_intersection_product;
use super::dfa_analysis::{
    dfa_transition_count,
    product_component_has_finite_language,
    refine_u8_partitions,
};
use super::factor::unwrap_shared;
use super::component::{
    collect_fixed_sequence_byte_sets,
    compile_product_component_materialized_dfa_with_options_and_cache,
    mark_state_accepting,
};
use super::mapping::{ZeroMinRepeatSuffixComponentTrace, zero_min_repeat_suffix_component_trace};
use super::plan::expr_profile_summary;
use super::repeat_suffix::compute_dfa_byte_equivalence_classes;
use rayon::prelude::*;

pub(super) type ProductStateTuple = SmallVec<[(u32, u32); 12]>;

const PRODUCT_STATE_FINGERPRINT_SEED: u64 = 0x9e37_79b9_7f4a_7c15;

#[inline]
fn extend_product_state_fingerprint(fingerprint: u64, group: u32, state: u32) -> u64 {
    let pair = ((group as u64) << 32) | state as u64;
    fingerprint
        .rotate_left(27)
        .wrapping_add(pair.wrapping_mul(0x9e37_79b1_85eb_ca87))
        .wrapping_mul(0xc2b2_ae3d_27d4_eb4f)
}

pub(super) fn product_state_fingerprint(tuple: &ProductStateTuple) -> u64 {
    tuple.iter().fold(
        PRODUCT_STATE_FINGERPRINT_SEED,
        |fingerprint, &(group, state)| extend_product_state_fingerprint(fingerprint, group, state),
    )
}

pub(super) struct ProductComponentProfileLabel {
    pub(super) name: String,
    pub(super) origin: &'static str,
    pub(super) shared: bool,
}

struct ProductGrowthTrieNode {
    children: HashMap<u32, usize>,
}

impl ProductGrowthTrieNode {
    fn new() -> Self {
        Self {
            children: HashMap::new(),
        }
    }
}

struct ProductGrowthRecorder {
    nodes: Vec<ProductGrowthTrieNode>,
    prefix_counts: Vec<usize>,
    dense_states: Vec<u32>,
}

impl ProductGrowthRecorder {
    fn new(num_groups: usize) -> Self {
        Self {
            nodes: vec![ProductGrowthTrieNode::new()],
            prefix_counts: vec![0; num_groups],
            dense_states: vec![0; num_groups],
        }
    }

    fn record(&mut self, num_groups: usize, state_tuple: &ProductStateTuple) {
        self.dense_states.fill(0);
        for &(group_id, state) in state_tuple {
            let group_index = group_id as usize;
            if group_index < num_groups {
                self.dense_states[group_index] = state.saturating_add(1);
            }
        }

        let mut node_index = 0usize;
        for (depth, &state) in self.dense_states.iter().enumerate() {
            let next_index = if let Some(&existing) = self.nodes[node_index].children.get(&state) {
                existing
            } else {
                let new_index = self.nodes.len();
                self.nodes.push(ProductGrowthTrieNode::new());
                self.nodes[node_index].children.insert(state, new_index);
                self.prefix_counts[depth] += 1;
                new_index
            };
            node_index = next_index;
        }
    }

    fn prefix_counts(&self) -> &[usize] {
        &self.prefix_counts
    }
}

pub(super) fn product_component_state_flags(component: &ProductComponent, state: u32) -> (bool, bool) {
    match component {
        ProductComponent::Materialized(dfa)
        | ProductComponent::MaterializedZeroMinRepeatSuffix { dfa, .. } => (
            dfa.finalizers(state).contains(0),
            dfa.possible_future_group_ids(state).contains(0),
        ),
        ProductComponent::VirtualFixedSequence {
            byte_sets,
            suffix_live,
        } => {
            let position = state as usize;
            (
                position == byte_sets.len(),
                position < byte_sets.len()
                    && suffix_live.get(position).copied().unwrap_or(false),
            )
        }
        ProductComponent::VirtualBoundedRepeat { base_dfa, min, max } => {
            let base_state_count = base_dfa.num_states() as u32;
            let copy_count = state / base_state_count;
            let base_state = state % base_state_count;
            (base_state == 0 && copy_count >= *min, copy_count < *max)
        }
    }
}

fn product_state_metadata_with_layout(
    components: &[ProductComponent],
    coordinate_groups: &[Vec<usize>],
    num_groups: usize,
    state_tuple: &ProductStateTuple,
) -> (BitSet, BitSet) {
    let mut finalizers = BitSet::new(num_groups);
    let mut future = BitSet::new(num_groups);

    for &(coordinate_id, state) in state_tuple {
        let coordinate = coordinate_id as usize;
        let (accepting, can_continue) =
            product_component_state_flags(&components[coordinate], state);
        for &group in &coordinate_groups[coordinate] {
            if accepting {
                finalizers.set(group);
            }
            if can_continue {
                future.set(group);
            }
        }
    }

    (finalizers, future)
}

fn product_state_single_visible_finalizer_with_layout(
    components: &[ProductComponent],
    coordinate_groups: &[Vec<usize>],
    num_groups: usize,
    state_tuple: &ProductStateTuple,
    exclusions: &BTreeMap<u32, BTreeSet<u32>>,
    intersections: &BTreeMap<u32, BTreeSet<u32>>,
) -> BitSet {
    let mut accepting = vec![false; num_groups];
    for &(coordinate_id, state) in state_tuple {
        let coordinate = coordinate_id as usize;
        let is_accepting = product_component_state_flags(&components[coordinate], state).0;
        for &group in &coordinate_groups[coordinate] {
            accepting[group] = is_accepting;
        }
    }

    let mut visible_accepting = accepting.first().copied().unwrap_or(false);
    if visible_accepting
        && exclusions
            .get(&0)
            .is_some_and(|blocked| blocked.iter().any(|&group| accepting[group as usize]))
    {
        visible_accepting = false;
    }
    if visible_accepting
        && intersections
            .get(&0)
            .is_some_and(|required| required.iter().any(|&group| !accepting[group as usize]))
    {
        visible_accepting = false;
    }

    let mut finalizers = BitSet::new(1);
    if visible_accepting {
        finalizers.set(0);
    }
    finalizers
}

pub(super) fn identity_product_coordinate_groups(num_groups: usize) -> Vec<Vec<usize>> {
    (0..num_groups).map(|group| vec![group]).collect()
}

pub(super) fn product_state_metadata(
    components: &[ProductComponent],
    state_tuple: &ProductStateTuple,
) -> (BitSet, BitSet) {
    let coordinate_groups = identity_product_coordinate_groups(components.len());
    product_state_metadata_with_layout(
        components,
        &coordinate_groups,
        components.len(),
        state_tuple,
    )
}

pub(super) fn product_state_single_visible_finalizer(
    components: &[ProductComponent],
    state_tuple: &ProductStateTuple,
    exclusions: &BTreeMap<u32, BTreeSet<u32>>,
    intersections: &BTreeMap<u32, BTreeSet<u32>>,
) -> BitSet {
    let coordinate_groups = identity_product_coordinate_groups(components.len());
    product_state_single_visible_finalizer_with_layout(
        components,
        &coordinate_groups,
        components.len(),
        state_tuple,
        exclusions,
        intersections,
    )
}

pub(super) fn set_single_group_futures_from_class_graph(
    dfa: &mut DFA,
    class_transitions: &[Vec<(u8, u32)>],
) {
    let states = dfa.num_states();
    debug_assert_eq!(states, class_transitions.len());
    let mut predecessor_counts = vec![0u32; states];
    let mut seen_target_epoch = vec![0u32; states];
    let mut epoch = 1u32;
    let mut edge_count = 0usize;
    for transitions in class_transitions {
        for &(_, target) in transitions {
            if seen_target_epoch[target as usize] == epoch {
                continue;
            }
            seen_target_epoch[target as usize] = epoch;
            predecessor_counts[target as usize] += 1;
            edge_count += 1;
        }
        epoch = epoch.checked_add(1).expect("product state epoch overflow");
    }

    let mut offsets = vec![0usize; states + 1];
    for state in 0..states {
        offsets[state + 1] = offsets[state] + predecessor_counts[state] as usize;
    }
    debug_assert_eq!(offsets[states], edge_count);
    let mut write_offsets = offsets[..states].to_vec();
    let mut predecessors = vec![0u32; edge_count];
    seen_target_epoch.fill(0);
    epoch = 1;
    for (source, transitions) in class_transitions.iter().enumerate() {
        for &(_, target) in transitions {
            if seen_target_epoch[target as usize] == epoch {
                continue;
            }
            seen_target_epoch[target as usize] = epoch;
            let slot = &mut write_offsets[target as usize];
            predecessors[*slot] = source as u32;
            *slot += 1;
        }
        epoch = epoch.checked_add(1).expect("product state epoch overflow");
    }

    let mut can_reach_final = vec![false; states];
    let mut queue = VecDeque::<u32>::new();
    for state in 0..states {
        if dfa.finalizers(state as u32).contains(0) {
            can_reach_final[state] = true;
            queue.push_back(state as u32);
        }
    }
    while let Some(target) = queue.pop_front() {
        let target = target as usize;
        for &source in &predecessors[offsets[target]..offsets[target + 1]] {
            if !can_reach_final[source as usize] {
                can_reach_final[source as usize] = true;
                queue.push_back(source);
            }
        }
    }

    for (source, transitions) in class_transitions.iter().enumerate() {
        let mut future = BitSet::new(1);
        if transitions
            .iter()
            .any(|&(_, target)| can_reach_final[target as usize])
        {
            future.set(0);
        }
        dfa.set_possible_future_group_ids(source as u32, future);
    }
}

/// Compute single-group futures from a class graph without first rebuilding a
/// CSR predecessor graph. The deferred dense intersection already stores one
/// compact transition row per state. A linked reverse graph can be assembled
/// in one pass, and the reverse reachability walk itself identifies exactly
/// the states with an outgoing edge to a final-reachable state.
pub(super) fn compute_single_group_futures_from_class_graph_csr(
    accepting: &[bool],
    transition_offsets: &[u32],
    transition_targets: &[u32],
    cooperative_yield_interval: usize,
) -> Vec<bool> {
    let states = accepting.len();
    debug_assert_eq!(transition_offsets.len(), states + 1);

    let mut predecessor_heads = vec![u32::MAX; states];
    let mut predecessor_sources = Vec::<u32>::new();
    let mut predecessor_next = Vec::<u32>::new();
    let mut seen_target_source = vec![u32::MAX; states];

    for source in 0..states {
        if cooperative_yield_interval != 0 && source % cooperative_yield_interval == 0 {
            let _ = rayon::yield_now();
        }
        let source_u32 = source as u32;
        let row_start = transition_offsets[source] as usize;
        let row_end = transition_offsets[source + 1] as usize;
        for &target in &transition_targets[row_start..row_end] {
            let target = target as usize;
            if seen_target_source[target] == source_u32 {
                continue;
            }
            seen_target_source[target] = source_u32;
            let edge = predecessor_sources.len() as u32;
            predecessor_sources.push(source_u32);
            predecessor_next.push(predecessor_heads[target]);
            predecessor_heads[target] = edge;
        }
    }

    let mut can_reach_final = vec![false; states];
    let mut has_future = vec![false; states];
    let mut queue = VecDeque::<u32>::new();
    for (state, &is_accepting) in accepting.iter().enumerate() {
        if is_accepting {
            can_reach_final[state] = true;
            queue.push_back(state as u32);
        }
    }

    let mut processed_targets = 0usize;
    while let Some(target) = queue.pop_front() {
        if cooperative_yield_interval != 0
            && processed_targets % cooperative_yield_interval == 0
        {
            let _ = rayon::yield_now();
        }
        processed_targets += 1;
        let mut edge = predecessor_heads[target as usize];
        while edge != u32::MAX {
            let source = predecessor_sources[edge as usize] as usize;
            has_future[source] = true;
            if !can_reach_final[source] {
                can_reach_final[source] = true;
                queue.push_back(source as u32);
            }
            edge = predecessor_next[edge as usize];
        }
    }
    has_future
}

pub(super) fn set_single_group_futures_from_class_graph_csr(
    dfa: &mut DFA,
    transition_offsets: &[u32],
    transition_targets: &[u32],
    cooperative_yield_interval: usize,
) {
    let accepting = (0..dfa.num_states())
        .map(|state| dfa.finalizers(state as u32).contains(0))
        .collect::<Vec<_>>();
    let has_future = compute_single_group_futures_from_class_graph_csr(
        &accepting,
        transition_offsets,
        transition_targets,
        cooperative_yield_interval,
    );
    for (state, &future) in has_future.iter().enumerate() {
        let mut future_groups = BitSet::new(1);
        if future {
            future_groups.set(0);
        }
        dfa.set_possible_future_group_ids(state as u32, future_groups);
    }
}

pub(super) fn explicit_dead_sink_state(dfa: &DFA) -> Option<u32> {
    for (state_id, state) in dfa.states().iter().enumerate() {
        if !state.finalizers.is_empty() {
            continue;
        }

        let mut transition_count = 0usize;
        let mut loops_to_self = true;
        for (_, &target) in state.transitions.iter() {
            transition_count += 1;
            if target != state_id as u32 {
                loops_to_self = false;
                break;
            }
        }

        if loops_to_self && transition_count == 256 {
            return Some(state_id as u32);
        }
    }

    None
}

#[derive(Clone)]
pub(super) enum ProductComponent {
    Materialized(Arc<DFA>),
    MaterializedZeroMinRepeatSuffix {
        dfa: Arc<DFA>,
        trace: Arc<ZeroMinRepeatSuffixComponentTrace>,
    },
    VirtualFixedSequence {
        byte_sets: Arc<[U8Set]>,
        suffix_live: Arc<[bool]>,
    },
    VirtualBoundedRepeat {
        base_dfa: Arc<DFA>,
        min: u32,
        max: u32,
    },
}

impl ProductComponent {
    pub(super) fn materialized_dfa(&self) -> Option<&DFA> {
        match self {
            Self::Materialized(dfa) | Self::MaterializedZeroMinRepeatSuffix { dfa, .. } => {
                Some(dfa)
            }
            Self::VirtualFixedSequence { .. } | Self::VirtualBoundedRepeat { .. } => None,
        }
    }

    pub(super) fn materialized_dfa_arc(&self) -> Option<Arc<DFA>> {
        match self {
            Self::Materialized(dfa) | Self::MaterializedZeroMinRepeatSuffix { dfa, .. } => {
                Some(Arc::clone(dfa))
            }
            Self::VirtualFixedSequence { .. } | Self::VirtualBoundedRepeat { .. } => None,
        }
    }

    /// Exact one-terminal DFA carrying the same residual state numbering used
    /// by this product coordinate.  This lets compile-time residual analyses
    /// consume the normal fast virtual fixed-sequence representation without
    /// forcing tokenizer construction to materialize those components.
    pub(super) fn terminal_residual_dfa_arc(&self) -> Option<Arc<DFA>> {
        match self {
            Self::Materialized(dfa) | Self::MaterializedZeroMinRepeatSuffix { dfa, .. } => {
                Some(Arc::clone(dfa))
            }
            Self::VirtualFixedSequence { byte_sets, .. } => {
                let mut dfa = DFA::new(byte_sets.len() + 1);
                dfa.ensure_group_capacity(1);
                let mut support = U8Set::empty();
                for (position, bytes) in byte_sets.iter().enumerate() {
                    support = support.union(bytes);
                    dfa.set_transitions_from_sorted_entries(
                        position as u32,
                        bytes
                            .iter()
                            .map(|byte| (byte, position as u32 + 1))
                            .collect(),
                    );
                }
                dfa.set_group_u8set(0, support);
                mark_state_accepting(&mut dfa, byte_sets.len() as u32);
                dfa.recompute_possible_futures();
                Some(Arc::new(dfa))
            }
            Self::VirtualBoundedRepeat { .. } => None,
        }
    }

    pub(super) fn zero_min_repeat_suffix_trace(&self) -> Option<Arc<ZeroMinRepeatSuffixComponentTrace>> {
        match self {
            Self::MaterializedZeroMinRepeatSuffix { trace, .. } => Some(Arc::clone(trace)),
            _ => None,
        }
    }
}

pub(super) struct ProductBuildTrace {
    pub(super) components: Vec<ProductComponent>,
    /// Logical product groups represented by each physical product coordinate.
    /// Normally this is the identity layout. Trace-enabled vocab partition
    /// builds may collapse duplicate coordinates exactly as the ordinary lexer
    /// does, while retaining this fan-out map for terminal residual recovery.
    pub(super) coordinate_groups: Vec<Vec<usize>>,
    pub(super) state_tuples: ProductStateTuples,
    pub(super) state_lookup: ProductStateLookup,
    pub(super) direct_single_visible_group: bool,
}

pub(super) enum ProductStateTuples {
    Generic(Vec<ProductStateTuple>),
    DenseBinary(Vec<(u32, u32)>),
}

impl ProductStateTuples {
    pub(super) fn len(&self) -> usize {
        match self {
            Self::Generic(tuples) => tuples.len(),
            Self::DenseBinary(pairs) => pairs.len(),
        }
    }

    pub(super) fn tuple(&self, state: usize) -> ProductStateTuple {
        match self {
            Self::Generic(tuples) => tuples[state].clone(),
            Self::DenseBinary(pairs) => {
                let (left, right) = pairs[state];
                let mut tuple = ProductStateTuple::new();
                tuple.push((0, left));
                tuple.push((1, right));
                tuple
            }
        }
    }

    pub(super) fn push(&mut self, tuple: ProductStateTuple) {
        match self {
            Self::Generic(tuples) => tuples.push(tuple),
            Self::DenseBinary(pairs)
                if tuple.len() == 2 && tuple[0].0 == 0 && tuple[1].0 == 1 =>
            {
                pairs.push((tuple[0].1, tuple[1].1));
            }
            Self::DenseBinary(pairs) => {
                let mut tuples = Vec::with_capacity(pairs.len() + 1);
                tuples.extend(pairs.drain(..).map(|(left, right)| {
                    let mut tuple = ProductStateTuple::new();
                    tuple.push((0, left));
                    tuple.push((1, right));
                    tuple
                }));
                tuples.push(tuple);
                *self = Self::Generic(tuples);
            }
        }
    }
}

pub(super) enum ProductStateLookup {
    Hash(FxHashMap<ProductStateTuple, u32>),
    Fingerprint {
        state_by_fingerprint: FxHashMap<u64, u32>,
        canonical_tuples: Vec<ProductStateTuple>,
        overflow: FxHashMap<ProductStateTuple, u32>,
    },
    DenseBinary {
        right_states: usize,
        state_by_pair: Vec<u32>,
        overflow: FxHashMap<ProductStateTuple, u32>,
    },
}

impl ProductStateLookup {
    pub(super) fn dense_pair_index(tuple: &ProductStateTuple, right_states: usize) -> Option<usize> {
        if tuple.len() != 2 || tuple[0].0 != 0 || tuple[1].0 != 1 {
            return None;
        }
        (tuple[0].1 as usize)
            .checked_mul(right_states)?
            .checked_add(tuple[1].1 as usize)
    }

    pub(super) fn get(&self, tuple: &ProductStateTuple) -> Option<u32> {
        match self {
            Self::Hash(states) => states.get(tuple).copied(),
            Self::Fingerprint {
                state_by_fingerprint,
                canonical_tuples,
                overflow,
            } => {
                let fingerprint = product_state_fingerprint(tuple);
                state_by_fingerprint
                    .get(&fingerprint)
                    .copied()
                    .filter(|&state| canonical_tuples[state as usize] == *tuple)
                    .or_else(|| overflow.get(tuple).copied())
            }
            Self::DenseBinary {
                right_states,
                state_by_pair,
                overflow,
            } => Self::dense_pair_index(tuple, *right_states)
                .and_then(|index| state_by_pair.get(index).copied())
                .filter(|&state| state != u32::MAX)
                .or_else(|| overflow.get(tuple).copied()),
        }
    }

    pub(super) fn insert(&mut self, tuple: ProductStateTuple, state: u32) {
        match self {
            Self::Hash(states) => {
                states.insert(tuple, state);
            }
            Self::Fingerprint {
                state_by_fingerprint,
                canonical_tuples,
                overflow,
            } => {
                debug_assert_eq!(canonical_tuples.len(), state as usize);
                let fingerprint = product_state_fingerprint(&tuple);
                if state_by_fingerprint.contains_key(&fingerprint) {
                    overflow.insert(tuple.clone(), state);
                } else {
                    state_by_fingerprint.insert(fingerprint, state);
                }
                canonical_tuples.push(tuple);
            }
            Self::DenseBinary {
                right_states,
                state_by_pair,
                overflow,
            } => {
                if let Some(index) = Self::dense_pair_index(&tuple, *right_states)
                    && let Some(slot) = state_by_pair.get_mut(index)
                {
                    *slot = state;
                } else {
                    overflow.insert(tuple, state);
                }
            }
        }
    }
}

pub(super) enum ProductComponentClassTransitions {
    Materialized(Vec<Vec<(u8, u32)>>),
    VirtualFixedSequence(Vec<Vec<(u8, u32)>>),
    VirtualBoundedRepeat(Vec<Vec<(u8, u32)>>),
}

fn product_class_transition_entry_count(
    transitions: &[ProductComponentClassTransitions],
) -> usize {
    transitions
        .iter()
        .map(|component| match component {
            ProductComponentClassTransitions::Materialized(rows)
            | ProductComponentClassTransitions::VirtualFixedSequence(rows)
            | ProductComponentClassTransitions::VirtualBoundedRepeat(rows) => {
                rows.iter().map(Vec::len).sum::<usize>()
            }
        })
        .sum()
}

impl ProductComponent {
    pub(super) fn partition_dfa(&self) -> &DFA {
        match self {
            ProductComponent::Materialized(dfa)
            | ProductComponent::MaterializedZeroMinRepeatSuffix { dfa, .. } => dfa,
            ProductComponent::VirtualFixedSequence { .. } => {
                panic!("virtual fixed sequences do not materialize a partition DFA")
            }
            ProductComponent::VirtualBoundedRepeat { base_dfa, .. } => base_dfa,
        }
    }

    pub(super) fn profile_state_count(&self) -> usize {
        match self {
            ProductComponent::Materialized(dfa)
            | ProductComponent::MaterializedZeroMinRepeatSuffix { dfa, .. } => dfa.num_states(),
            ProductComponent::VirtualFixedSequence { byte_sets, .. } => byte_sets.len() + 1,
            ProductComponent::VirtualBoundedRepeat { base_dfa, .. } => base_dfa.num_states(),
        }
    }

    pub(super) fn profile_transition_count(&self) -> usize {
        match self {
            ProductComponent::Materialized(dfa)
            | ProductComponent::MaterializedZeroMinRepeatSuffix { dfa, .. } => {
                dfa_transition_count(dfa)
            }
            ProductComponent::VirtualFixedSequence { byte_sets, .. } => {
                byte_sets.iter().map(U8Set::len).sum()
            }
            ProductComponent::VirtualBoundedRepeat { base_dfa, .. } => {
                dfa_transition_count(base_dfa)
            }
        }
    }

    pub(super) fn dead_state(&self) -> Option<u32> {
        match self {
            ProductComponent::Materialized(dfa)
            | ProductComponent::MaterializedZeroMinRepeatSuffix { dfa, .. } => {
                explicit_dead_sink_state(dfa)
            }
            ProductComponent::VirtualFixedSequence { .. } => None,
            ProductComponent::VirtualBoundedRepeat { base_dfa, .. } => explicit_dead_sink_state(base_dfa),
        }
    }
}

pub(super) fn compile_product_component_with_options(
    expr: &Expr,
    preserve_coordinates: bool,
    virtual_fixed_sequences: bool,
    repeat_base_cache: Option<&RepeatBaseDfaCache>,
) -> ProductComponent {
    if preserve_coordinates
        && let Some(trace) = zero_min_repeat_suffix_component_trace(expr)
    {
        let dfa = Arc::clone(&trace.dfa);
        return ProductComponent::MaterializedZeroMinRepeatSuffix {
            dfa,
            trace: Arc::new(trace),
        };
    }
    if !preserve_coordinates && virtual_fixed_sequences {
        let mut byte_sets = Vec::new();
        if collect_fixed_sequence_byte_sets(expr, &mut byte_sets) {
            let mut suffix_live = vec![false; byte_sets.len() + 1];
            suffix_live[byte_sets.len()] = true;
            for position in (0..byte_sets.len()).rev() {
                suffix_live[position] = !byte_sets[position].is_empty() && suffix_live[position + 1];
            }
            return ProductComponent::VirtualFixedSequence {
                byte_sets: Arc::from(byte_sets.into_boxed_slice()),
                suffix_live: Arc::from(suffix_live.into_boxed_slice()),
            };
        }
    }
    match expr {
        Expr::Shared(inner) => compile_product_component_with_options(
            inner,
            preserve_coordinates,
            virtual_fixed_sequences,
            repeat_base_cache,
        ),
        Expr::Repeat {
            expr: repeat_expr,
            min,
            max: Some(max),
        } => {
            if let Some(base_dfa) = cached_direct_bounded_repeat_base_dfa(
                repeat_expr,
                *max,
                repeat_base_cache,
            ) {
                return ProductComponent::VirtualBoundedRepeat {
                    base_dfa,
                    min: *min as u32,
                    max: *max as u32,
                };
            }

            ProductComponent::Materialized(Arc::new(
                compile_product_component_materialized_dfa_with_options_and_cache(
                    expr,
                    preserve_coordinates,
                    repeat_base_cache,
                ),
            ))
        }
        _ => ProductComponent::Materialized(Arc::new(
            compile_product_component_materialized_dfa_with_options_and_cache(
                expr,
                preserve_coordinates,
                repeat_base_cache,
            ),
        )),
    }
}

pub(super) fn compile_product_component(expr: &Expr) -> ProductComponent {
    compile_product_component_with_options(expr, false, true, None)
}

pub(super) fn certify_supplied_dfa_state_homomorphism(
    full: &DFA,
    full_compressed: Option<&CompressedTransitionSegment>,
    synthesized: &DFA,
    full_to_synthesized: &[u32],
) -> bool {
    if full.num_groups() != synthesized.num_groups()
        || full_to_synthesized.len() != full.num_states()
        || full_to_synthesized
            .iter()
            .any(|&state| state as usize >= synthesized.num_states())
    {
        return false;
    }

    let diagnostics = std::env::var_os("GLRMASK_PROFILE_SYNTH_CERT").is_some();
    let mut metadata_mismatch_states = 0usize;
    let mut epsilon_mismatch_states = 0usize;
    let mut transition_mismatch_states = 0usize;
    let mut mismatched_synthesized_states = FxHashSet::<u32>::default();
    let synthesized_dense_len = match synthesized.num_states().checked_mul(256) {
        Some(len) => len,
        None => return false,
    };
    let mut synthesized_dense = vec![u32::MAX; synthesized_dense_len];
    let mut synthesized_transition_counts = vec![0usize; synthesized.num_states()];
    for (state, dfa_state) in synthesized.states().iter().enumerate() {
        synthesized_transition_counts[state] = dfa_state.transitions.len();
        let row = &mut synthesized_dense[state * 256..(state + 1) * 256];
        for (byte, &target) in dfa_state.transitions.iter() {
            row[byte as usize] = target;
        }
    }

    let mut mapped_epsilon = Vec::<u32>::new();
    for full_state in 0..full.num_states() as u32 {
        let synthesized_state = full_to_synthesized[full_state as usize];
        if full.finalizers(full_state) != synthesized.finalizers(synthesized_state)
            || full.possible_future_group_ids(full_state)
                != synthesized.possible_future_group_ids(synthesized_state)
        {
            if !diagnostics {
                return false;
            }
            metadata_mismatch_states += 1;
            mismatched_synthesized_states.insert(synthesized_state);
        }

        mapped_epsilon.clear();
        mapped_epsilon.extend(
            full.states()[full_state as usize]
                .epsilon_transitions
                .iter()
                .map(|&target| full_to_synthesized[target as usize]),
        );
        mapped_epsilon.sort_unstable();
        mapped_epsilon.dedup();
        if mapped_epsilon
            != synthesized.states()[synthesized_state as usize].epsilon_transitions
        {
            if !diagnostics {
                return false;
            }
            epsilon_mismatch_states += 1;
            mismatched_synthesized_states.insert(synthesized_state);
        }

        let synthesized_row = &synthesized_dense
            [synthesized_state as usize * 256..(synthesized_state as usize + 1) * 256];
        let full_transition_count = match full_compressed {
            Some(segment) if segment.contains_state(full_state) => {
                segment.transition_count(full_state)
            }
            _ => full.states()[full_state as usize].transitions.len(),
        };
        let mut transition_mismatch = full_transition_count
            != synthesized_transition_counts[synthesized_state as usize];
        let transitions_match = match full_compressed {
            Some(segment) if segment.contains_state(full_state) => segment.transitions_satisfy(
                full_state,
                |byte, target| {
                    full_to_synthesized[target as usize] == synthesized_row[byte as usize]
                },
            ),
            _ => full.states()[full_state as usize]
                .transitions
                .iter()
                .all(|(byte, &target)| {
                    full_to_synthesized[target as usize] == synthesized_row[byte as usize]
                }),
        };
        transition_mismatch |= !transitions_match;
        if transition_mismatch && !diagnostics {
            return false;
        }
        if transition_mismatch {
            transition_mismatch_states += 1;
            mismatched_synthesized_states.insert(synthesized_state);
        }
    }
    if diagnostics {
        eprintln!(
            "[glrmask/profile][synth_cert] full_states={} synthesized_states={} metadata_mismatch_states={} epsilon_mismatch_states={} transition_mismatch_states={} mismatched_synthesized_states={}",
            full.num_states(),
            synthesized.num_states(),
            metadata_mismatch_states,
            epsilon_mismatch_states,
            transition_mismatch_states,
            mismatched_synthesized_states.len(),
        );
    }
    metadata_mismatch_states == 0
        && epsilon_mismatch_states == 0
        && transition_mismatch_states == 0
}

pub(super) fn deterministic_component_homomorphism_state_map(
    full: &DFA,
    synthesized: &DFA,
) -> Option<Vec<u32>> {
    if full.num_groups() != synthesized.num_groups() {
        return None;
    }
    let mut mapping = vec![u32::MAX; full.num_states()];
    let mut worklist = VecDeque::from([(0u32, 0u32)]);
    mapping[0] = 0;

    while let Some((full_state, synthesized_state)) = worklist.pop_front() {
        if full.finalizers(full_state) != synthesized.finalizers(synthesized_state)
            || full.possible_future_group_ids(full_state)
                != synthesized.possible_future_group_ids(synthesized_state)
        {
            return None;
        }
        for byte in 0u16..=255 {
            let full_target = full.step(full_state, byte as u8);
            let synthesized_target = synthesized.step(synthesized_state, byte as u8);
            match (full_target, synthesized_target) {
                (None, None) => {}
                (Some(full_target), Some(synthesized_target)) => {
                    let slot = &mut mapping[full_target as usize];
                    if *slot == u32::MAX {
                        *slot = synthesized_target;
                        worklist.push_back((full_target, synthesized_target));
                    } else if *slot != synthesized_target {
                        return None;
                    }
                }
                _ => return None,
            }
        }
    }
    mapping.iter().all(|&state| state != u32::MAX).then_some(mapping)
}

fn count_bounded_repeat_base_uses(expr: &Expr, counts: &mut FxHashMap<Expr, usize>) {
    match unwrap_shared(expr) {
        Expr::Repeat {
            expr,
            max: Some(_),
            ..
        } => {
            let body = unwrap_shared(expr);
            *counts.entry(body.clone()).or_default() += 1;
            count_bounded_repeat_base_uses(body, counts);
        }
        Expr::Seq(parts) | Expr::Choice(parts) => {
            for part in parts {
                count_bounded_repeat_base_uses(part, counts);
            }
        }
        Expr::Exclude { expr, exclude } => {
            count_bounded_repeat_base_uses(expr, counts);
            count_bounded_repeat_base_uses(exclude, counts);
        }
        Expr::Intersect { expr, intersect } => {
            count_bounded_repeat_base_uses(expr, counts);
            count_bounded_repeat_base_uses(intersect, counts);
        }
        Expr::Shared(_) => unreachable!("unwrap_shared removes Shared"),
        Expr::U8Seq(_) | Expr::U8Class(_) | Expr::Dfa(_) | Expr::Epsilon => {}
        Expr::Repeat { max: None, expr, .. } => {
            count_bounded_repeat_base_uses(expr, counts);
        }
    }
}

const LOCAL_SMALL_PRODUCT_MAX_COORDINATES: usize = 32;

fn local_small_product_work_enabled(component_count: usize, local_small_product: bool) -> bool {
    local_small_product && component_count <= LOCAL_SMALL_PRODUCT_MAX_COORDINATES
}

pub(super) fn build_repeat_base_dfa_cache(
    exprs: &[&Expr],
    local_small_product: bool,
) -> RepeatBaseDfaCache {
    let mut counts = FxHashMap::<Expr, usize>::default();
    for expr in exprs {
        count_bounded_repeat_base_uses(expr, &mut counts);
    }
    let repeated = counts
        .into_iter()
        .filter_map(|(expr, uses)| (uses >= 2).then_some(expr))
        .collect::<Vec<_>>();
    let compile = |expr: Expr| {
        compile_direct_bounded_repeat_base_dfa_unconditionally(&expr)
            .map(|dfa| (expr, Arc::new(dfa)))
    };
    if local_small_product_work_enabled(exprs.len(), local_small_product) {
        repeated.into_iter().filter_map(compile).collect()
    } else {
        repeated.into_par_iter().filter_map(compile).collect()
    }
}

#[derive(Debug)]
pub(super) struct ProductComponentCompileProfile {
    first_group_index: usize,
    uses: usize,
    compile_ms: f64,
    states: usize,
    transitions: usize,
}

pub(super) fn compile_product_components_profiled(
    exprs: &[Expr],
    profile_detail: bool,
    preserve_coordinates: bool,
    virtual_fixed_sequences: bool,
    local_small_product: bool,
) -> (
    Vec<ProductComponent>,
    usize,
    Option<Vec<ProductComponentCompileProfile>>,
    Vec<usize>,
) {
    let mut unique_exprs = Vec::<&Expr>::new();
    let mut unique_first_group_indices = Vec::<usize>::new();
    let mut component_indices = Vec::with_capacity(exprs.len());
    let mut index_by_expr = FxHashMap::<&Expr, usize>::default();

    for (group_index, expr) in exprs.iter().enumerate() {
        let expr = unwrap_shared(expr);
        let index = if let Some(&index) = index_by_expr.get(expr) {
            index
        } else {
            let index = unique_exprs.len();
            unique_exprs.push(expr);
            unique_first_group_indices.push(group_index);
            index_by_expr.insert(expr, index);
            index
        };
        component_indices.push(index);
    }

    let repeat_base_cache = build_repeat_base_dfa_cache(&unique_exprs, local_small_product);
    let repeat_base_cache = (!repeat_base_cache.is_empty()).then_some(&repeat_base_cache);

    let compile_component = |(_index, expr): (usize, &&Expr)| {
        if profile_detail {
            let started_at = Instant::now();
            let component = compile_product_component_with_options(
                expr,
                preserve_coordinates,
                virtual_fixed_sequences,
                repeat_base_cache,
            );
            (
                component,
                Some(started_at.elapsed().as_secs_f64() * 1000.0),
            )
        } else {
            (
                compile_product_component_with_options(
                    expr,
                    preserve_coordinates,
                    virtual_fixed_sequences,
                    repeat_base_cache,
                ),
                None,
            )
        }
    };
    let compiled: Vec<(ProductComponent, Option<f64>)> =
        if local_small_product_work_enabled(unique_exprs.len(), local_small_product) {
            unique_exprs.iter().enumerate().map(compile_component).collect()
        } else {
            unique_exprs.par_iter().enumerate().map(compile_component).collect()
        };
    let (unique_components, compile_times): (Vec<_>, Vec<_>) = compiled.into_iter().unzip();
    let cache_hits = exprs.len() - unique_components.len();
    let profiles = profile_detail.then(|| {
        let mut uses = vec![0usize; unique_components.len()];
        for &index in &component_indices {
            uses[index] += 1;
        }
        unique_components
            .iter()
            .zip(&compile_times)
            .enumerate()
            .map(|(index, (component, compile_ms))| ProductComponentCompileProfile {
                first_group_index: unique_first_group_indices[index],
                uses: uses[index],
                compile_ms: compile_ms.unwrap_or_default(),
                states: component.profile_state_count(),
                transitions: component.profile_transition_count(),
            })
            .collect::<Vec<_>>()
    });
    let components = component_indices
        .iter()
        .copied()
        .map(|index| unique_components[index].clone())
        .collect();
    (components, cache_hits, profiles, component_indices)
}

pub(super) fn compile_product_components(exprs: &[Expr]) -> (Vec<ProductComponent>, usize) {
    let (components, cache_hits, _, _) =
        compile_product_components_profiled(exprs, false, false, true, false);
    (components, cache_hits)
}

fn product_coordinate_layout(
    logical_components: Vec<ProductComponent>,
    component_indices: &[usize],
    collapse_duplicates: bool,
) -> (Vec<ProductComponent>, Vec<Vec<usize>>) {
    if !collapse_duplicates {
        let coordinate_groups = (0..logical_components.len())
            .map(|group| vec![group])
            .collect();
        return (logical_components, coordinate_groups);
    }

    let coordinate_count = component_indices
        .iter()
        .copied()
        .max()
        .map_or(0, |max_index| max_index + 1);
    let mut components = vec![None; coordinate_count];
    let mut coordinate_groups = vec![Vec::new(); coordinate_count];
    for (group, (component, &coordinate)) in logical_components
        .into_iter()
        .zip(component_indices)
        .enumerate()
    {
        coordinate_groups[coordinate].push(group);
        if components[coordinate].is_none() {
            components[coordinate] = Some(component);
        }
    }
    let components = components
        .into_iter()
        .map(|component| component.expect("every product coordinate has a component"))
        .collect();
    (components, coordinate_groups)
}

pub(super) fn build_product_dfa(
    exprs: &[Expr],
    profile_labels: Option<&[ProductComponentProfileLabel]>,
    visible_groups: usize,
    exclusions: &BTreeMap<u32, BTreeSet<u32>>,
    intersections: &BTreeMap<u32, BTreeSet<u32>>,
    capture_trace: bool,
    virtual_fixed_sequences: bool,
    local_small_product: bool,
    collapse_traced_duplicate_coordinates: bool,
) -> (DFA, bool, Option<ProductBuildTrace>) {
    let profile_trace = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TRACE").is_some();
    let profile_detail = profile_trace
        || std::env::var_os("GLRMASK_PROFILE_TOKENIZER_DETAIL").is_some();
    let profile_timing = profile_detail
        || std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    let profile_started_at = Instant::now();
    let component_compile_started_at = Instant::now();
    let preserve_component_coordinates = capture_trace && !collapse_traced_duplicate_coordinates;
    let (logical_components, component_cache_hits, component_profiles, component_indices) =
        compile_product_components_profiled(
            exprs,
            profile_detail,
            preserve_component_coordinates,
            virtual_fixed_sequences,
            local_small_product,
        );
    let component_compile_ms = profile_timing
        .then(|| component_compile_started_at.elapsed().as_secs_f64() * 1000.0);
    if profile_timing {
        eprintln!(
            "[glrmask/profile][tokenizer] product_component_cache groups={} unique_components={} cache_hits={}",
            exprs.len(),
            exprs.len() - component_cache_hits,
            component_cache_hits,
        );
    }
    if profile_detail {
        eprintln!(
            "[glrmask/profile][tokenizer] product_components groups={} unique_components={} cache_hits={} compile_components_ms={:.3}",
            logical_components.len(),
            logical_components.len() - component_cache_hits,
            component_cache_hits,
            profile_started_at.elapsed().as_secs_f64() * 1000.0
        );
        let mut ranked = component_profiles
            .as_ref()
            .expect("detail profiling records unique component profiles")
            .iter()
            .enumerate()
            .collect::<Vec<_>>();
        ranked.sort_unstable_by(|(_, left), (_, right)| {
            right.compile_ms.total_cmp(&left.compile_ms)
        });
        let report_count = if profile_trace {
            ranked.len()
        } else {
            ranked.len().min(20)
        };
        for (rank, (unique_index, component)) in ranked.into_iter().take(report_count).enumerate() {
            let index = component.first_group_index;
            let label = profile_labels
                .and_then(|labels| labels.get(index))
                .map(|label| {
                    format!(
                        " name={:?} origin={} shared={} expr={:?}",
                        label.name,
                        label.origin,
                        label.shared,
                        expr_profile_summary(&exprs[index]),
                    )
                })
                .unwrap_or_else(|| format!(" expr={:?}", expr_profile_summary(&exprs[index])));
            eprintln!(
                "[glrmask/profile][tokenizer/component-rank] rank={} unique_index={} first_group_index={} uses={} states={} transitions={} compile_ms={:.3}{}",
                rank + 1,
                unique_index,
                index,
                component.uses,
                component.states,
                component.transitions,
                component.compile_ms,
                label,
            );
        }
        let omitted = component_profiles
            .as_ref()
            .map_or(0, |profiles| profiles.len().saturating_sub(report_count));
        if omitted > 0 {
            eprintln!(
                "[glrmask/profile][tokenizer/component-rank] omitted={} set GLRMASK_PROFILE_TOKENIZER_TRACE=1 for exhaustive component output",
                omitted,
            );
        }
    }
    let num_groups = logical_components.len();
    let collapse_duplicates = component_cache_hits > 0
        && visible_groups > 1
        && (!capture_trace || collapse_traced_duplicate_coordinates)
        && !profile_trace;
    let (components, coordinate_groups) = product_coordinate_layout(
        logical_components,
        &component_indices,
        collapse_duplicates,
    );
    let num_coordinates = components.len();
    let direct_single_visible_group = visible_groups == 1
        && num_groups > 1
        && (!exclusions.is_empty() || !intersections.is_empty());
    let component_dead_states: Vec<Option<u32>> = components
        .iter()
        .map(ProductComponent::dead_state)
        .collect();
    let equivalence_classes_started_at = Instant::now();
    let (class_map, class_members) = compute_product_equivalence_classes(&components);
    let equivalence_classes_ms = profile_timing
        .then(|| equivalence_classes_started_at.elapsed().as_secs_f64() * 1000.0);
    let num_classes = class_members.len();
    let class_transition_started_at = Instant::now();
    let component_class_transitions = build_product_class_transitions(&components, &class_map);
    let class_transition_ms = profile_timing
        .then(|| class_transition_started_at.elapsed().as_secs_f64() * 1000.0);
    if direct_single_visible_group
        && pure_binary_intersection(exclusions, intersections)
        && let Some(result) = try_build_dense_binary_intersection_product(
            &components,
            &class_members,
            &component_class_transitions,
            capture_trace,
            profile_timing,
        )
    {
        return result;
    }
    if direct_single_visible_group
        && pure_binary_intersection(exclusions, intersections)
        && let Some(result) = try_build_sparse_virtual_repeat_finite_intersection_product(
            &components,
            &class_members,
            &component_class_transitions,
            capture_trace,
            profile_timing,
        )
    {
        return result;
    }
    let mut dfa = DFA::new(1);
    dfa.ensure_group_capacity(if direct_single_visible_group {
        1
    } else {
        num_groups
    });

    assert!(num_groups <= u32::MAX as usize, "too many product DFA groups");
    let mut start_tuple = ProductStateTuple::with_capacity(num_coordinates);
    for coordinate_id in 0..num_coordinates {
        start_tuple.push((coordinate_id as u32, 0u32));
    }
    let (start_finalizers, start_future) = if direct_single_visible_group {
        (
            product_state_single_visible_finalizer_with_layout(
                &components,
                &coordinate_groups,
                num_groups,
                &start_tuple,
                exclusions,
                intersections,
            ),
            BitSet::new(1),
        )
    } else {
        product_state_metadata_with_layout(&components, &coordinate_groups, num_groups, &start_tuple)
    };
    dfa.overwrite_state_metadata(0, start_finalizers, start_future);

    const PRODUCT_STATE_FINGERPRINT_MIN_CLASS_TRANSITIONS: usize = 20_000;
    let fingerprint_state_lookup = capture_trace
        && std::env::var_os("GLRMASK_DISABLE_PRODUCT_STATE_FINGERPRINT").is_none()
        && (std::env::var_os("GLRMASK_EXPERIMENT_PRODUCT_STATE_FINGERPRINT").is_some()
            || product_class_transition_entry_count(&component_class_transitions)
                >= PRODUCT_STATE_FINGERPRINT_MIN_CLASS_TRANSITIONS);
    let mut state_map = FxHashMap::<ProductStateTuple, u32>::default();
    let mut state_by_fingerprint = FxHashMap::<u64, u32>::default();
    let mut fingerprint_collisions = FxHashMap::<ProductStateTuple, u32>::default();
    let mut fingerprint_state_tuples = fingerprint_state_lookup.then(|| vec![start_tuple.clone()]);
    let mut worklist = VecDeque::new();
    let mut pending_class_transitions = vec![Vec::<(u8, u32)>::new()];
    // Pre-allocated buffers for class transition tuples (reused across states)
    let mut class_buffers: Vec<ProductStateTuple> = (0..num_classes)
        .map(|_| ProductStateTuple::new())
        .collect();
    let mut class_fingerprints = vec![PRODUCT_STATE_FINGERPRINT_SEED; num_classes];
    let mut class_active = vec![false; num_classes];
    let mut used_classes = Vec::<usize>::new();
    let mut growth_recorder = profile_trace.then(|| ProductGrowthRecorder::new(num_coordinates));
    let mut state_tuples = capture_trace.then(|| vec![start_tuple.clone()]);
    if fingerprint_state_lookup {
        state_by_fingerprint.insert(product_state_fingerprint(&start_tuple), 0);
    } else {
        state_map.insert(start_tuple.clone(), 0);
    }
    if let Some(recorder) = growth_recorder.as_mut() {
        recorder.record(num_coordinates, &start_tuple);
    }
    worklist.push_back((0, start_tuple));

    let product_state_expand_started_at = Instant::now();
    while let Some((current_state, state_tuple)) = worklist.pop_front() {
        for &(group_id, component_state) in &state_tuple {
            let group_index = group_id as usize;

            match (&components[group_index], &component_class_transitions[group_index]) {
                (
                    ProductComponent::Materialized(_)
                    | ProductComponent::MaterializedZeroMinRepeatSuffix { .. },
                    ProductComponentClassTransitions::Materialized(class_transitions),
                ) => {
                    for &(class_id, target) in &class_transitions[component_state as usize] {
                        let class_index = class_id as usize;
                        if !class_active[class_index] {
                            class_active[class_index] = true;
                            used_classes.push(class_index);
                            class_fingerprints[class_index] = PRODUCT_STATE_FINGERPRINT_SEED;
                        }
                        if component_dead_states[group_index] == Some(target) {
                            continue;
                        }

                        class_buffers[class_index].push((group_id, target));
                        if fingerprint_state_lookup {
                            class_fingerprints[class_index] = extend_product_state_fingerprint(
                                class_fingerprints[class_index],
                                group_id,
                                target,
                            );
                        }
                    }
                }
                (
                    ProductComponent::VirtualFixedSequence { .. },
                    ProductComponentClassTransitions::VirtualFixedSequence(class_transitions),
                ) => {
                    for &(class_id, target) in &class_transitions[component_state as usize] {
                        let class_index = class_id as usize;
                        if !class_active[class_index] {
                            class_active[class_index] = true;
                            used_classes.push(class_index);
                            class_fingerprints[class_index] = PRODUCT_STATE_FINGERPRINT_SEED;
                        }
                        class_buffers[class_index].push((group_id, target));
                        if fingerprint_state_lookup {
                            class_fingerprints[class_index] = extend_product_state_fingerprint(
                                class_fingerprints[class_index],
                                group_id,
                                target,
                            );
                        }
                    }
                }
                (
                    ProductComponent::VirtualBoundedRepeat { base_dfa, max, .. },
                    ProductComponentClassTransitions::VirtualBoundedRepeat(base_class_transitions),
                ) => {
                    let base_state_count = base_dfa.num_states() as u32;
                    let copy_count = component_state / base_state_count;
                    if copy_count >= *max {
                        continue;
                    }

                    let base_state = component_state % base_state_count;
                    if base_dfa.finalizers(base_state).contains(0) {
                        continue;
                    }
                    for &(class_id, target_base) in &base_class_transitions[base_state as usize] {
                        let class_index = class_id as usize;
                        if !class_active[class_index] {
                            class_active[class_index] = true;
                            used_classes.push(class_index);
                            class_fingerprints[class_index] = PRODUCT_STATE_FINGERPRINT_SEED;
                        }
                        if component_dead_states[group_index] == Some(target_base) {
                            continue;
                        }

                        let target = if base_dfa.finalizers(target_base).contains(0) {
                            (copy_count + 1) * base_state_count
                        } else {
                            copy_count * base_state_count + target_base
                        };

                        class_buffers[class_index].push((group_id, target));
                        if fingerprint_state_lookup {
                            class_fingerprints[class_index] = extend_product_state_fingerprint(
                                class_fingerprints[class_index],
                                group_id,
                                target,
                            );
                        }
                    }
                }
                _ => unreachable!("component and class-transition kinds must match"),
            }
        }

        let mut class_transitions = Vec::with_capacity(used_classes.len());
        for &class_index in &used_classes {
            let next_tuple = &class_buffers[class_index];
            let existing = if fingerprint_state_lookup {
                let fingerprint = class_fingerprints[class_index];
                state_by_fingerprint
                    .get(&fingerprint)
                    .copied()
                    .filter(|&state| {
                        fingerprint_state_tuples
                            .as_ref()
                            .expect("fingerprint lookup retains canonical tuples")[state as usize]
                            == *next_tuple
                    })
                    .or_else(|| fingerprint_collisions.get(next_tuple).copied())
            } else {
                state_map.get(next_tuple).copied()
            };
            let next_state = if let Some(existing) = existing {
                existing
            } else {
                let new_state = dfa.add_state();
                let (finalizers, future) = if direct_single_visible_group {
                    (
                        product_state_single_visible_finalizer_with_layout(
                            &components,
                            &coordinate_groups,
                            num_groups,
                            next_tuple,
                            exclusions,
                            intersections,
                        ),
                        BitSet::new(1),
                    )
                } else {
                    product_state_metadata_with_layout(&components, &coordinate_groups, num_groups, next_tuple)
                };
                dfa.overwrite_state_metadata(new_state, finalizers, future);
                if fingerprint_state_lookup {
                    let fingerprint = class_fingerprints[class_index];
                    if state_by_fingerprint.contains_key(&fingerprint) {
                        fingerprint_collisions.insert(next_tuple.clone(), new_state);
                    } else {
                        state_by_fingerprint.insert(fingerprint, new_state);
                    }
                    let tuples = fingerprint_state_tuples
                        .as_mut()
                        .expect("fingerprint lookup retains canonical tuples");
                    debug_assert_eq!(tuples.len(), new_state as usize);
                    tuples.push(next_tuple.clone());
                } else {
                    state_map.insert(next_tuple.clone(), new_state);
                }
                if let Some(state_tuples) = state_tuples.as_mut() {
                    debug_assert_eq!(state_tuples.len(), new_state as usize);
                    state_tuples.push(next_tuple.clone());
                }
                if let Some(recorder) = growth_recorder.as_mut() {
                    recorder.record(num_coordinates, next_tuple);
                }
                pending_class_transitions.push(Vec::new());
                worklist.push_back((new_state, next_tuple.clone()));
                new_state
            };
            class_transitions.push((class_index as u8, next_state));
            class_buffers[class_index].clear();
            class_fingerprints[class_index] = PRODUCT_STATE_FINGERPRINT_SEED;
            class_active[class_index] = false;
        }
        used_classes.clear();
        pending_class_transitions[current_state as usize] = class_transitions;
    }
    let product_state_expand_ms = profile_timing
        .then(|| product_state_expand_started_at.elapsed().as_secs_f64() * 1000.0);

    let direct_future_started_at = Instant::now();
    if direct_single_visible_group {
        set_single_group_futures_from_class_graph(&mut dfa, &pending_class_transitions);
    }
    let direct_future_ms = profile_timing
        .then(|| direct_future_started_at.elapsed().as_secs_f64() * 1000.0);

    if profile_trace {
        if let Some(recorder) = growth_recorder.as_ref() {
            let mut states_before = 0usize;
            for (index, states_after) in recorder.prefix_counts().iter().copied().enumerate() {
                let label = profile_labels
                    .and_then(|labels| labels.get(index))
                    .map(|label| {
                        format!(
                            " name={:?} origin={} shared={}",
                            label.name,
                            label.origin,
                            label.shared
                        )
                    })
                    .unwrap_or_else(|| format!(" expr={:?}", expr_profile_summary(&exprs[index])));
                eprintln!(
                    "[glrmask/profile][tokenizer/product-growth] component_index={} states_before={} states_after={} delta_states={}{}",
                    index,
                    states_before,
                    states_after,
                    states_after.saturating_sub(states_before),
                    label
                );
                states_before = states_after;
            }
        }
        eprintln!(
            "[glrmask/profile][tokenizer] product_reachable states={} classes={} construct_ms={:.3}",
            dfa.num_states(),
            num_classes,
            profile_started_at.elapsed().as_secs_f64() * 1000.0
        );
    }

    let byte_expand_started_at = Instant::now();
    let expand_byte_row = |class_transitions: Vec<(u8, u32)>| {
        let byte_capacity: usize = class_transitions
            .iter()
            .map(|(class_id, _)| class_members[*class_id as usize].len())
            .sum();
        const DENSE_BYTE_EXPANSION_THRESHOLD: usize = 96;

        let transitions = if byte_capacity >= DENSE_BYTE_EXPANSION_THRESHOLD {
            // Byte-equivalence classes are disjoint but need not be
            // contiguous, so expanding classes in class-ID order does
            // not generally produce byte-sorted output.  For dense rows,
            // scatter targets into the fixed byte alphabet and scan it
            // once. This avoids a large per-state comparison sort.
            let mut target_by_byte = [u32::MAX; 256];
            for (class_id, target) in class_transitions {
                for &byte in &class_members[class_id as usize] {
                    target_by_byte[byte as usize] = target;
                }
            }
            target_by_byte
                .into_iter()
                .enumerate()
                .filter_map(|(byte, target)| {
                    (target != u32::MAX).then_some((byte as u8, target))
                })
                .collect()
        } else {
            let mut transitions = Vec::with_capacity(byte_capacity);
            for (class_id, target) in class_transitions {
                for &byte in &class_members[class_id as usize] {
                    transitions.push((byte, target));
                }
            }
            if transitions.len() > 1 {
                transitions.sort_unstable_by_key(|entry| entry.0);
            }
            transitions
        };
        crate::ds::char_transitions::CharTransitions::from_sorted_entries(transitions)
    };
    let expanded_transitions: Vec<crate::ds::char_transitions::CharTransitions<u32>> =
        if local_small_product_work_enabled(num_coordinates, local_small_product) {
            pending_class_transitions
                .into_iter()
                .map(expand_byte_row)
                .collect()
        } else {
            pending_class_transitions
                .into_par_iter()
                .map(expand_byte_row)
                .collect()
        };
    let byte_expand_ms = profile_timing
        .then(|| byte_expand_started_at.elapsed().as_secs_f64() * 1000.0);

    for (state, transitions) in dfa.states_mut().iter_mut().zip(expanded_transitions) {
        state.transitions = transitions;
    }

    if profile_timing {
        eprintln!(
            "[glrmask/profile][tokenizer] product_phases groups={} coordinates={} classes={} direct_single_visible_group={} component_compile_ms={:.3} equivalence_classes_ms={:.3} class_transition_ms={:.3} product_state_expand_ms={:.3} direct_future_ms={:.3} byte_expand_ms={:.3}",
            num_groups,
            num_coordinates,
            num_classes,
            direct_single_visible_group,
            component_compile_ms.unwrap_or_default(),
            equivalence_classes_ms.unwrap_or_default(),
            class_transition_ms.unwrap_or_default(),
            product_state_expand_ms.unwrap_or_default(),
            direct_future_ms.unwrap_or_default(),
            byte_expand_ms.unwrap_or_default(),
        );
    }

    let state_lookup = if fingerprint_state_lookup {
        ProductStateLookup::Fingerprint {
            state_by_fingerprint,
            canonical_tuples: fingerprint_state_tuples
                .expect("fingerprint lookup retains canonical tuples"),
            overflow: fingerprint_collisions,
        }
    } else {
        ProductStateLookup::Hash(state_map)
    };
    let trace = state_tuples.map(|state_tuples| ProductBuildTrace {
        components,
        coordinate_groups,
        state_tuples: ProductStateTuples::Generic(state_tuples),
        state_lookup,
        direct_single_visible_group,
    });
    (dfa, direct_single_visible_group, trace)
}

pub(super) fn compute_product_equivalence_classes(components: &[ProductComponent]) -> (Vec<u8>, Vec<Vec<u8>>) {
    let mut partitions = vec![U8Set::all()];
    let mut seen_sets = FxHashSet::default();

    for component in components {
        match component {
            ProductComponent::Materialized(dfa)
            | ProductComponent::MaterializedZeroMinRepeatSuffix { dfa, .. }
            | ProductComponent::VirtualBoundedRepeat { base_dfa: dfa, .. } => {
                for state in dfa.states() {
                    let mut bytes_by_target = FxHashMap::<u32, U8Set>::default();
                    for (byte, &target) in state.transitions.iter() {
                        bytes_by_target
                            .entry(target)
                            .and_modify(|set| {
                                set.insert(byte);
                            })
                            .or_insert_with(|| U8Set::single(byte));
                    }

                    for byte_set in bytes_by_target.into_values() {
                        if seen_sets.insert(byte_set) {
                            partitions = refine_u8_partitions(partitions, byte_set);
                        }
                    }
                }
            }
            ProductComponent::VirtualFixedSequence { byte_sets, .. } => {
                for &byte_set in byte_sets.iter() {
                    if seen_sets.insert(byte_set) {
                        partitions = refine_u8_partitions(partitions, byte_set);
                    }
                }
            }
        }
    }

    let mut class_map = vec![0u8; 256];
    let mut class_members = vec![Vec::new(); partitions.len()];
    for (class_id, partition) in partitions.iter().enumerate() {
        for byte in partition.iter() {
            class_map[byte as usize] = class_id as u8;
            class_members[class_id].push(byte);
        }
    }

    (class_map, class_members)
}

pub(super) fn build_product_class_transitions_for_dfa(dfa: &DFA, class_map: &[u8]) -> Vec<Vec<(u8, u32)>> {
    let class_count = class_map
        .iter()
        .copied()
        .max()
        .map_or(0usize, |class| class as usize + 1);
    dfa.states()
        .par_iter()
        .map(|state| {
            let mut target_by_class = [u32::MAX; 256];
            for (byte, &target) in state.transitions.iter() {
                let class = class_map[byte as usize] as usize;
                let slot = &mut target_by_class[class];
                if *slot == u32::MAX {
                    *slot = target;
                } else {
                    debug_assert_eq!(
                        *slot, target,
                        "lexer byte-equivalence class must have one target per DFA state",
                    );
                }
            }
            (0..class_count)
                .filter_map(|class| {
                    let target = target_by_class[class];
                    (target != u32::MAX).then_some((class as u8, target))
                })
                .collect()
        })
        .collect()
}

pub(super) fn build_product_class_transitions(
    components: &[ProductComponent],
    class_map: &[u8],
) -> Vec<ProductComponentClassTransitions> {
    components
        .iter()
        .map(|component| match component {
            ProductComponent::Materialized(dfa)
            | ProductComponent::MaterializedZeroMinRepeatSuffix { dfa, .. } => {
                ProductComponentClassTransitions::Materialized(build_product_class_transitions_for_dfa(
                    dfa, class_map,
                ))
            }
            ProductComponent::VirtualFixedSequence { byte_sets, .. } => {
                let mut transitions = Vec::with_capacity(byte_sets.len() + 1);
                for (position, bytes) in byte_sets.iter().enumerate() {
                    let mut classes = bytes
                        .iter()
                        .map(|byte| class_map[byte as usize])
                        .collect::<Vec<_>>();
                    classes.sort_unstable();
                    classes.dedup();
                    transitions.push(
                        classes
                            .into_iter()
                            .map(|class| (class, position as u32 + 1))
                            .collect(),
                    );
                }
                transitions.push(Vec::new());
                ProductComponentClassTransitions::VirtualFixedSequence(transitions)
            }
            ProductComponent::VirtualBoundedRepeat { base_dfa, .. } => {
                ProductComponentClassTransitions::VirtualBoundedRepeat(
                    build_product_class_transitions_for_dfa(base_dfa, class_map),
                )
            }
        })
        .collect()
}

fn finite_product_component_class_row(
    component: &ProductComponent,
    class_transitions: &ProductComponentClassTransitions,
    state: u32,
    out: &mut Vec<(u8, u32)>,
) -> Option<()> {
    out.clear();
    match (component, class_transitions) {
        (
            ProductComponent::Materialized(_)
            | ProductComponent::MaterializedZeroMinRepeatSuffix { .. },
            ProductComponentClassTransitions::Materialized(rows),
        ) => out.extend_from_slice(rows.get(state as usize)?),
        (
            ProductComponent::VirtualFixedSequence { .. },
            ProductComponentClassTransitions::VirtualFixedSequence(rows),
        ) => out.extend_from_slice(rows.get(state as usize)?),
        _ => return None,
    }
    Some(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct SparseVirtualRepeatResidual {
    completed: u32,
    base_state: u32,
}

fn sparse_virtual_repeat_class_row(
    component: &ProductComponent,
    class_transitions: &ProductComponentClassTransitions,
    state: SparseVirtualRepeatResidual,
    out: &mut Vec<(u8, SparseVirtualRepeatResidual)>,
) -> Option<()> {
    let ProductComponent::VirtualBoundedRepeat { base_dfa, max, .. } = component else {
        return None;
    };
    let ProductComponentClassTransitions::VirtualBoundedRepeat(base_rows) = class_transitions else {
        return None;
    };
    out.clear();
    if state.completed >= *max {
        return Some(());
    }
    if base_dfa.finalizers(state.base_state).contains(0) {
        return Some(());
    }
    let dead = explicit_dead_sink_state(base_dfa);
    for &(class, target_base) in base_rows.get(state.base_state as usize)? {
        if dead == Some(target_base) {
            continue;
        }
        let target = if base_dfa.finalizers(target_base).contains(0) {
            SparseVirtualRepeatResidual {
                completed: state.completed.checked_add(1)?,
                base_state: 0,
            }
        } else {
            SparseVirtualRepeatResidual {
                completed: state.completed,
                base_state: target_base,
            }
        };
        out.push((class, target));
    }
    Some(())
}

fn sparse_virtual_repeat_is_accepting(
    component: &ProductComponent,
    state: SparseVirtualRepeatResidual,
) -> Option<bool> {
    let ProductComponent::VirtualBoundedRepeat { min, .. } = component else {
        return None;
    };
    Some(state.base_state == 0 && state.completed >= *min)
}

/// Exact sparse product for a pure binary intersection containing one virtual
/// bounded-repeat coordinate and one finite-language coordinate.  The generic
/// product builder intentionally keeps partial tuples so it can represent
/// unions/exclusions as well as intersections; for a giant repeat that can
/// leave a repeat-only tuple alive after the other coordinate dies and expand
/// once per declared repetition.  Here intersection semantics let us require
/// both coordinates to transition on every byte and discard targets that can
/// no longer accept, so the finite coordinate's live DAG bounds discovery
/// independently of the repeat maximum.
pub(super) fn try_build_sparse_virtual_repeat_finite_intersection_product(
    components: &[ProductComponent],
    class_members: &[Vec<u8>],
    component_class_transitions: &[ProductComponentClassTransitions],
    capture_trace: bool,
    profile_timing: bool,
) -> Option<(DFA, bool, Option<ProductBuildTrace>)> {
    if components.len() != 2 || component_class_transitions.len() != 2 {
        return None;
    }
    let repeat_index = match (&components[0], &components[1]) {
        (ProductComponent::VirtualBoundedRepeat { .. }, ProductComponent::VirtualBoundedRepeat { .. }) => {
            return None;
        }
        (ProductComponent::VirtualBoundedRepeat { .. }, _) => 0usize,
        (_, ProductComponent::VirtualBoundedRepeat { .. }) => 1usize,
        _ => return None,
    };
    let finite_index = 1 - repeat_index;
    if !product_component_has_finite_language(&components[finite_index]) {
        return None;
    }

    let started_at = profile_timing.then(Instant::now);
    let start_repeat = SparseVirtualRepeatResidual {
        completed: 0,
        base_state: 0,
    };
    let start = (start_repeat, 0u32);
    let mut pair_to_state = FxHashMap::<(SparseVirtualRepeatResidual, u32), u32>::default();
    pair_to_state.insert(start, 0);
    let mut pairs = vec![start];
    let mut pending_class_transitions = vec![Vec::<(u8, u32)>::new()];
    let start_accepting = sparse_virtual_repeat_is_accepting(
        &components[repeat_index],
        start_repeat,
    )
    .expect("identified virtual repeat component must expose repeat acceptance")
        && product_component_state_flags(&components[finite_index], 0).0;
    let mut accepting = vec![start_accepting];
    let mut repeat_row = Vec::<(u8, SparseVirtualRepeatResidual)>::new();
    let mut finite_row = Vec::<(u8, u32)>::new();

    let mut cursor = 0usize;
    while cursor < pairs.len() {
        let (repeat_state, finite_state) = pairs[cursor];
        sparse_virtual_repeat_class_row(
            &components[repeat_index],
            &component_class_transitions[repeat_index],
            repeat_state,
            &mut repeat_row,
        )
        .expect("identified virtual repeat component must expose an exact class row");
        finite_product_component_class_row(
            &components[finite_index],
            &component_class_transitions[finite_index],
            finite_state,
            &mut finite_row,
        )
        .expect("finite intersection component must expose an exact class row");

        let mut repeat_position = 0usize;
        let mut finite_position = 0usize;
        let mut row = Vec::<(u8, u32)>::new();
        while repeat_position < repeat_row.len() && finite_position < finite_row.len() {
            let (repeat_class, repeat_target) = repeat_row[repeat_position];
            let (finite_class, finite_target) = finite_row[finite_position];
            if repeat_class < finite_class {
                repeat_position += 1;
                continue;
            }
            if finite_class < repeat_class {
                finite_position += 1;
                continue;
            }
            repeat_position += 1;
            finite_position += 1;

            let finite_flags =
                product_component_state_flags(&components[finite_index], finite_target);
            if !finite_flags.0 && !finite_flags.1 {
                continue;
            }
            let repeat_accepting = sparse_virtual_repeat_is_accepting(
                &components[repeat_index],
                repeat_target,
            )
            .expect("identified virtual repeat component must expose repeat acceptance");
            let pair = (repeat_target, finite_target);
            let target = if let Some(&target) = pair_to_state.get(&pair) {
                target
            } else {
                let target = u32::try_from(pairs.len())
                    .expect("sparse exact product exceeded the DFA u32 state-id domain");
                pair_to_state.insert(pair, target);
                pairs.push(pair);
                pending_class_transitions.push(Vec::new());
                accepting.push(repeat_accepting && finite_flags.0);
                target
            };
            row.push((repeat_class, target));
        }
        pending_class_transitions[cursor] = row;
        cursor += 1;
    }

    let mut dfa = DFA::new(1);
    dfa.ensure_group_capacity(1);
    while dfa.num_states() < pairs.len() {
        dfa.add_state();
    }
    for (state, &is_accepting) in accepting.iter().enumerate() {
        let mut finalizers = BitSet::new(1);
        if is_accepting {
            finalizers.set(0);
        }
        dfa.overwrite_state_metadata(state as u32, finalizers, BitSet::new(1));
    }
    set_single_group_futures_from_class_graph(&mut dfa, &pending_class_transitions);

    let expanded = pending_class_transitions
        .into_iter()
        .map(|row| {
            let mut transitions = Vec::new();
            for (class, target) in row {
                transitions.extend(
                    class_members[class as usize]
                        .iter()
                        .copied()
                        .map(|byte| (byte, target)),
                );
            }
            if transitions.len() > 1 {
                transitions.sort_unstable_by_key(|entry| entry.0);
            }
            crate::ds::char_transitions::CharTransitions::from_sorted_entries(transitions)
        })
        .collect::<Vec<_>>();
    for (state, transitions) in dfa.states_mut().iter_mut().zip(expanded) {
        state.transitions = transitions;
    }
    if let Some(started_at) = started_at {
        eprintln!(
            "[glrmask/profile][tokenizer] sparse_virtual_repeat_finite_intersection repeat_component={} states={} classes={} total_ms={:.3}",
            repeat_index,
            dfa.num_states(),
            class_members.len(),
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    let _ = capture_trace;
    Some((dfa, true, None))
}

pub(super) fn pure_binary_intersection(
    exclusions: &BTreeMap<u32, BTreeSet<u32>>,
    intersections: &BTreeMap<u32, BTreeSet<u32>>,
) -> bool {
    exclusions.is_empty()
        && intersections.len() == 1
        && intersections
            .get(&0)
            .is_some_and(|required| required.len() == 1 && required.contains(&1))
}

/// Exact single-group DFA for `left \\ right`.
///
/// Unlike the ordinary two-component product, the RHS is allowed to die while
/// the LHS remains live.  A dedicated sentinel represents that permanently-dead
/// RHS coordinate, so reachable states fit in one dense
/// `(left_state, right_state_or_dead)` table with no tuple hashing.
pub(super) fn build_dense_binary_exclusion_dfa(left: &DFA, right: &DFA, support: U8Set) -> Option<DFA> {
    const MAX_DENSE_PAIR_CELLS: usize = 40_000_000;

    if left.num_states() == 0
        || right.num_states() == 0
        || left.num_groups() != 1
        || right.num_groups() != 1
        || left.has_epsilon_transitions()
        || right.has_epsilon_transitions()
    {
        return None;
    }

    let left_states = left.num_states();
    let right_states = right.num_states();
    let right_dead_sentinel = u32::try_from(right_states).ok()?;
    let right_stride = right_states.checked_add(1)?;
    let pair_cells = left_states.checked_mul(right_stride)?;
    if pair_cells == 0 || pair_cells > MAX_DENSE_PAIR_CELLS {
        return None;
    }

    let profile = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    let started_at = profile.then(Instant::now);
    let class_started_at = profile.then(Instant::now);
    let (class_map, class_members) = compute_dfa_byte_equivalence_classes(&[left, right]);
    let class_ms = class_started_at
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let left_transitions = build_product_class_transitions_for_dfa(left, &class_map);
    let right_transitions = build_product_class_transitions_for_dfa(right, &class_map);
    let left_dead = explicit_dead_sink_state(left);
    let right_dead = explicit_dead_sink_state(right);
    if left_dead == Some(0) {
        let mut dfa = DFA::new(1);
        dfa.ensure_group_capacity(1);
        dfa.set_group_u8set(0, support);
        return Some(dfa);
    }
    let right_start = if right_dead == Some(0) {
        right_dead_sentinel
    } else {
        0
    };

    let pair_index = |left_state: u32, right_state: u32| -> Option<usize> {
        (left_state as usize)
            .checked_mul(right_stride)?
            .checked_add(right_state as usize)
    };

    let mut state_by_pair = vec![u32::MAX; pair_cells];
    state_by_pair[pair_index(0, right_start)?] = 0;
    let mut pairs = vec![(0u32, right_start)];
    let mut accepting = Vec::<bool>::new();
    let mut row_offsets = Vec::<u32>::with_capacity(256);
    let mut row_classes = Vec::<u8>::new();
    let mut row_targets = Vec::<u32>::new();
    row_offsets.push(0);

    let is_accepting = |left_state: u32, right_state: u32| {
        if !left.finalizers(left_state).contains(0) {
            return false;
        }
        right_state == right_dead_sentinel || !right.finalizers(right_state).contains(0)
    };

    let construct_started_at = profile.then(Instant::now);
    let mut cursor = 0usize;
    while cursor < pairs.len() {
        let (left_state, right_state) = pairs[cursor];
        accepting.push(is_accepting(left_state, right_state));
        let left_row = left_transitions.get(left_state as usize)?;
        let right_row = if right_state == right_dead_sentinel {
            None
        } else {
            Some(right_transitions.get(right_state as usize)?)
        };
        let mut right_index = 0usize;
        for &(class, left_target) in left_row {
            if left_target == u32::MAX || left_dead == Some(left_target) {
                continue;
            }

            let right_target = if let Some(right_row) = right_row {
                while right_index < right_row.len() && right_row[right_index].0 < class {
                    right_index += 1;
                }
                if right_index < right_row.len() && right_row[right_index].0 == class {
                    let target = right_row[right_index].1;
                    if target == u32::MAX || right_dead == Some(target) {
                        right_dead_sentinel
                    } else {
                        target
                    }
                } else {
                    right_dead_sentinel
                }
            } else {
                right_dead_sentinel
            };

            let index = pair_index(left_target, right_target)?;
            let target = if state_by_pair[index] != u32::MAX {
                state_by_pair[index]
            } else {
                let target = u32::try_from(pairs.len()).ok()?;
                state_by_pair[index] = target;
                pairs.push((left_target, right_target));
                target
            };
            row_classes.push(class);
            row_targets.push(target);
        }
        row_offsets.push(u32::try_from(row_targets.len()).ok()?);
        cursor += 1;
    }
    let construct_ms = construct_started_at
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

    let future_started_at = profile.then(Instant::now);
    let has_future = compute_single_group_futures_from_class_graph_csr(
        &accepting,
        &row_offsets,
        &row_targets,
        0,
    );
    let future_ms = future_started_at
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

    let mut dfa = DFA::new(accepting.len());
    dfa.ensure_group_capacity(1);
    dfa.set_group_u8set(0, support);
    for state in 0..dfa.num_states() {
        let mut finalizers = BitSet::new(1);
        if accepting[state] {
            finalizers.set(0);
        }
        let mut future = BitSet::new(1);
        if has_future[state] {
            future.set(0);
        }
        dfa.overwrite_state_metadata(state as u32, finalizers, future);
    }

    let expand_started_at = profile.then(Instant::now);
    for state in 0..dfa.num_states() {
        let start = row_offsets[state] as usize;
        let end = row_offsets[state + 1] as usize;
        let byte_capacity = row_classes[start..end]
            .iter()
            .map(|class| class_members[*class as usize].len())
            .sum();
        let mut transitions = Vec::<(u8, u32)>::with_capacity(byte_capacity);
        for (&class, &target) in row_classes[start..end]
            .iter()
            .zip(&row_targets[start..end])
        {
            transitions.extend(
                class_members[class as usize]
                    .iter()
                    .copied()
                    .map(|byte| (byte, target)),
            );
        }
        if transitions.len() > 1 {
            transitions.sort_unstable_by_key(|entry| entry.0);
        }
        dfa.set_transitions_from_sorted_entries(state as u32, transitions);
    }
    let expand_ms = expand_started_at
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

    if let Some(started_at) = started_at {
        eprintln!(
            "[glrmask/profile][tokenizer] dense_binary_exclusion left_states={} right_states={} pair_cells={} reachable_states={} classes={} class_ms={:.3} construct_ms={:.3} future_ms={:.3} expand_ms={:.3} total_ms={:.3}",
            left_states,
            right_states,
            pair_cells,
            dfa.num_states(),
            class_members.len(),
            class_ms,
            construct_ms,
            future_ms,
            expand_ms,
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Some(dfa)
}
