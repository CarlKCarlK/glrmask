//! Deferred dense/virtual product construction without eager runtime materialization.

use crate::automata::lexer::ast::Expr;
use crate::automata::lexer::dfa::DFA;
use crate::automata::lexer::tokenizer::{CompressedTransitionEntries, CompressedTransitionSegment};
use crate::ds::bitset::BitSet;
use crate::ds::u8set::U8Set;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::hash::Hash;
use std::sync::Arc;
use std::time::Instant;
use super::bounded_repeat::build_bounded_repeat_dfa_from_base;
use super::dfa_analysis::{
    dfa_transition_count,
    product_component_has_finite_language,
    single_group_dfa_state_is_live,
};
use super::factor::unwrap_shared;
use super::nfa::expr_u8set;
use super::plan::{ExclusionCompilePlan, build_exclusion_compile_plan};
use super::product::{
    ProductBuildTrace,
    ProductComponent,
    ProductComponentClassTransitions,
    ProductStateLookup,
    ProductStateTuple,
    ProductStateTuples,
    build_product_class_transitions,
    build_product_class_transitions_for_dfa,
    compile_product_component_with_options,
    compile_product_components_profiled,
    compute_product_equivalence_classes,
    compute_single_group_futures_from_class_graph_csr,
    identity_product_coordinate_groups,
    pure_binary_intersection,
    set_single_group_futures_from_class_graph,
    set_single_group_futures_from_class_graph_csr,
};
use super::repeat_suffix::{
    LazyZeroMinRepeatSuffixComponent,
    ZeroMinRepeatSuffixState,
    compute_lazy_zero_min_repeat_product_equivalence_classes,
};
use super::virtual_repeat::VIRTUAL_BINARY_REPEAT_MIN_BOUND;
use rayon::prelude::*;

pub(super) enum DeferredDfa {
    Ready(DFA),
    ReadyCompressed {
        dfa: DFA,
        segment: CompressedTransitionSegment,
    },
    DenseBinary(DeferredDenseBinaryIntersectionProduct),
}

impl DeferredDfa {
    pub(super) fn num_states(&self) -> usize {
        match self {
            Self::Ready(dfa) => dfa.num_states(),
            Self::ReadyCompressed { dfa, .. } => dfa.num_states(),
            Self::DenseBinary(product) => product.num_states(),
        }
    }

    pub(super) fn initial_is_nonnullable_without_epsilon(&self) -> bool {
        match self {
            Self::Ready(dfa) => {
                dfa.finalizers(0).is_empty() && !dfa.has_epsilon_transitions()
            }
            Self::ReadyCompressed { dfa, .. } => {
                dfa.finalizers(0).is_empty() && !dfa.has_epsilon_transitions()
            }
            Self::DenseBinary(product) => !product.initial_is_accepting(),
        }
    }

    pub(super) fn ensure_group_capacity(&mut self, num_groups: usize) {
        match self {
            Self::Ready(dfa) => dfa.ensure_group_capacity(num_groups),
            Self::ReadyCompressed { dfa, .. } => dfa.ensure_group_capacity(num_groups),
            Self::DenseBinary(product) => product.ensure_group_capacity(num_groups),
        }
    }

    pub(super) fn set_group_u8set(&mut self, group_id: u32, set: U8Set) {
        match self {
            Self::Ready(dfa) => dfa.set_group_u8set(group_id, set),
            Self::ReadyCompressed { dfa, .. } => dfa.set_group_u8set(group_id, set),
            Self::DenseBinary(product) => product.set_group_u8set(group_id, set),
        }
    }

    pub(super) fn attach_dense_runtime_trace(&mut self, trace: ProductBuildTrace) -> Option<()> {
        match self {
            Self::Ready(_) | Self::ReadyCompressed { .. } => Some(()),
            Self::DenseBinary(product) => product.attach_runtime_trace(trace),
        }
    }

    pub(super) fn has_deferred_runtime_materialization(&self) -> bool {
        match self {
            Self::Ready(_) => false,
            Self::ReadyCompressed { .. } => true,
            Self::DenseBinary(product) => product.deferred_transition_rows.is_some(),
        }
    }

    pub(super) fn finish(self) -> DFA {
        match self {
            Self::Ready(dfa) => dfa,
            Self::ReadyCompressed { mut dfa, segment } => {
                segment.materialize_into_dfa(&mut dfa);
                dfa
            }
            Self::DenseBinary(product) => product.finish(),
        }
    }

    pub(super) fn finish_runtime(self) -> (DFA, Option<CompressedTransitionSegment>) {
        match self {
            Self::Ready(dfa) => (dfa, None),
            Self::ReadyCompressed { dfa, segment } => (dfa, Some(segment)),
            Self::DenseBinary(product) => {
                let (dfa, segment) = product.finish_compressed();
                (dfa, Some(segment))
            }
        }
    }
}

pub(super) struct DeferredDenseBinaryIntersectionProduct {
    metadata: DFA,
    accepting: Vec<bool>,
    pending_class_transition_offsets: Vec<u32>,
    pending_class_transition_classes: Vec<u8>,
    pending_class_transition_targets: Vec<u32>,
    deferred_transition_rows: Option<DeferredDenseBinaryTransitionRows>,
    class_members: Vec<Vec<u8>>,
    left_states: usize,
    right_states: usize,
    pair_cells: usize,
    num_classes: usize,
    cooperative_yield_interval: usize,
    discovery_ms: f64,
    started_at: Option<Instant>,
}

struct DeferredDenseBinaryTransitionRows {
    pairs: Vec<(u32, u32)>,
    state_by_pair: Vec<u32>,
    left_transitions: Vec<Vec<(u8, u32)>>,
    right_transitions: Vec<Vec<(u8, u32)>>,
    left_dead: Option<u32>,
    right_dead: Option<u32>,
}

impl DeferredDenseBinaryIntersectionProduct {
    fn num_states(&self) -> usize {
        self.accepting.len()
    }

    fn initial_is_accepting(&self) -> bool {
        self.accepting.first().copied().unwrap_or(false)
    }

    fn ensure_group_capacity(&mut self, num_groups: usize) {
        self.metadata.ensure_group_capacity(num_groups);
    }

    fn set_group_u8set(&mut self, group_id: u32, set: U8Set) {
        self.metadata.set_group_u8set(group_id, set);
    }

    fn materialize_metadata_parts(
        metadata: DFA,
        accepting: Vec<bool>,
        cooperative_yield_interval: usize,
    ) -> DFA {
        if metadata.num_states() == accepting.len() {
            return metadata;
        }
        let num_groups = metadata.num_groups();
        let mut dfa = DFA::new(accepting.len());
        dfa.ensure_group_capacity(num_groups);
        for group in 0..num_groups {
            dfa.set_group_u8set(
                group as u32,
                *metadata.group_id_to_u8set(group as u32),
            );
        }
        if num_groups != 0 {
            for (state, &is_accepting) in accepting.iter().enumerate() {
                if cooperative_yield_interval != 0
                    && state % cooperative_yield_interval == 0
                {
                    let _ = rayon::yield_now();
                }
                if is_accepting {
                    let mut finalizers = BitSet::new(num_groups);
                    finalizers.set(0);
                    dfa.overwrite_state_metadata(
                        state as u32,
                        finalizers,
                        BitSet::new(num_groups),
                    );
                }
            }
        }
        dfa
    }

    fn attach_runtime_trace(&mut self, trace: ProductBuildTrace) -> Option<()> {
        let ProductStateTuples::DenseBinary(pairs) = trace.state_tuples else {
            return None;
        };
        let ProductStateLookup::DenseBinary {
            right_states,
            state_by_pair,
            overflow,
        } = trace.state_lookup
        else {
            return None;
        };
        if right_states != self.right_states || !overflow.is_empty() || pairs.len() != self.num_states()
        {
            return None;
        }
        let Some(rows) = self.deferred_transition_rows.as_mut() else {
            return Some(());
        };
        rows.pairs = pairs;
        rows.state_by_pair = state_by_pair;
        Some(())
    }

    fn materialize_pending_class_transitions(&mut self) -> Option<()> {
        let Some(rows) = self.deferred_transition_rows.take() else {
            return Some(());
        };
        let state_count = self.pending_class_transition_offsets.len().checked_sub(1)?;
        if rows.pairs.len() != state_count
            || rows.state_by_pair.len() != self.pair_cells
        {
            return None;
        }
        if self.pending_class_transition_offsets.len() != rows.pairs.len() + 1 {
            return None;
        }
        let total_entries = *self.pending_class_transition_offsets.last()? as usize;
        self.pending_class_transition_classes = Vec::with_capacity(total_entries);
        self.pending_class_transition_targets = Vec::with_capacity(total_entries);
        for (state, &(left_state, right_state)) in rows.pairs.iter().enumerate() {
            if self.cooperative_yield_interval != 0
                && state % self.cooperative_yield_interval == 0
            {
                let _ = rayon::yield_now();
            }
            let expected_start = self.pending_class_transition_offsets[state] as usize;
            if self.pending_class_transition_classes.len() != expected_start
                || self.pending_class_transition_targets.len() != expected_start
            {
                return None;
            }
            let left_row = &rows.left_transitions[left_state as usize];
            let right_row = &rows.right_transitions[right_state as usize];
            let mut left_index = 0usize;
            let mut right_index = 0usize;
            while left_index < left_row.len() && right_index < right_row.len() {
                let (left_class, left_target) = left_row[left_index];
                let (right_class, right_target) = right_row[right_index];
                if left_class < right_class {
                    left_index += 1;
                    continue;
                }
                if right_class < left_class {
                    right_index += 1;
                    continue;
                }
                left_index += 1;
                right_index += 1;
                if left_target == u32::MAX
                    || right_target == u32::MAX
                    || rows.left_dead == Some(left_target)
                    || rows.right_dead == Some(right_target)
                {
                    continue;
                }
                let pair_index = (left_target as usize)
                    .checked_mul(self.right_states)?
                    .checked_add(right_target as usize)?;
                let target = *rows.state_by_pair.get(pair_index)?;
                if target == u32::MAX {
                    return None;
                }
                self.pending_class_transition_classes.push(left_class);
                self.pending_class_transition_targets.push(target);
            }
            if self.pending_class_transition_classes.len()
                != self.pending_class_transition_offsets[state + 1] as usize
                || self.pending_class_transition_targets.len()
                    != self.pending_class_transition_offsets[state + 1] as usize
            {
                return None;
            }
        }
        debug_assert_eq!(self.pending_class_transition_classes.len(), total_entries);
        debug_assert_eq!(self.pending_class_transition_targets.len(), total_entries);
        Some(())
    }

    fn finish(mut self) -> DFA {
        let profile_timing = self.started_at.is_some();
        let metadata = std::mem::take(&mut self.metadata);
        let accepting = std::mem::take(&mut self.accepting);
        let cooperative_yield_interval = self.cooperative_yield_interval;
        let ((mut dfa, metadata_ms), (transition_ok, transition_rows_ms)) = rayon::join(
            || {
                let started_at = profile_timing.then(Instant::now);
                let dfa = Self::materialize_metadata_parts(
                    metadata,
                    accepting,
                    cooperative_yield_interval,
                );
                let ms = started_at
                    .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
                (dfa, ms)
            },
            || {
                let started_at = profile_timing.then(Instant::now);
                let result = self.materialize_pending_class_transitions();
                let ms = started_at
                    .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
                (result, ms)
            },
        );
        transition_ok.expect("deferred dense transition rows must materialize");
        let future_started_at = profile_timing.then(Instant::now);
        set_single_group_futures_from_class_graph_csr(
            &mut dfa,
            &self.pending_class_transition_offsets,
            &self.pending_class_transition_targets,
            self.cooperative_yield_interval,
        );
        let future_ms = future_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

        let expansion_started_at = profile_timing.then(Instant::now);
        let parallel_expansion = std::env::var("GLRMASK_PARALLEL_DEFERRED_BYTE_EXPANSION")
            .map(|value| {
                let value = value.trim();
                !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
            })
            .unwrap_or(false);
        let dense_expansion_threshold = std::env::var(
            "GLRMASK_DEFERRED_DENSE_BYTE_EXPANSION_THRESHOLD",
        )
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(16);
        let mut byte_to_class = [0u8; 256];
        for (class, bytes) in self.class_members.iter().enumerate() {
            for &byte in bytes {
                byte_to_class[byte as usize] = class as u8;
            }
        }
        let expand = |classes: &[u8], targets: &[u32], target_by_class: &mut [u32]| {
            debug_assert_eq!(classes.len(), targets.len());
            let byte_capacity: usize = classes
                .iter()
                .map(|class| self.class_members[*class as usize].len())
                .sum();
            let entries = if byte_capacity >= dense_expansion_threshold {
                target_by_class.fill(u32::MAX);
                for (&class, &target) in classes.iter().zip(targets) {
                    target_by_class[class as usize] = target;
                }
                let mut entries = Vec::with_capacity(byte_capacity);
                for (byte, &class) in byte_to_class.iter().enumerate() {
                    let target = target_by_class[class as usize];
                    if target != u32::MAX {
                        entries.push((byte as u8, target));
                    }
                }
                entries
            } else {
                let mut entries = Vec::with_capacity(byte_capacity);
                for (&class, &target) in classes.iter().zip(targets) {
                    for &byte in &self.class_members[class as usize] {
                        entries.push((byte, target));
                    }
                }
                if entries.len() > 1 {
                    entries.sort_unstable_by_key(|entry| entry.0);
                }
                entries
            };
            crate::ds::char_transitions::CharTransitions::from_sorted_entries(entries)
        };
        let expanded_transitions: Vec<crate::ds::char_transitions::CharTransitions<u32>> =
            if parallel_expansion {
                (0..dfa.num_states())
                    .into_par_iter()
                    .map_init(
                        || vec![u32::MAX; self.num_classes],
                        |target_by_class, state| {
                            let row_start = self.pending_class_transition_offsets[state] as usize;
                            let row_end =
                                self.pending_class_transition_offsets[state + 1] as usize;
                            expand(
                                &self.pending_class_transition_classes[row_start..row_end],
                                &self.pending_class_transition_targets[row_start..row_end],
                                target_by_class,
                            )
                        },
                    )
                    .collect()
            } else {
                let mut target_by_class = vec![u32::MAX; self.num_classes];
                (0..dfa.num_states())
                    .map(|state| {
                        let row_start = self.pending_class_transition_offsets[state] as usize;
                        let row_end = self.pending_class_transition_offsets[state + 1] as usize;
                        expand(
                            &self.pending_class_transition_classes[row_start..row_end],
                            &self.pending_class_transition_targets[row_start..row_end],
                            &mut target_by_class,
                        )
                    })
                    .collect()
            };
        for (state, transitions) in dfa.states_mut().iter_mut().zip(expanded_transitions) {
            state.transitions = transitions;
        }
        let expansion_ms = expansion_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

        if let Some(started_at) = self.started_at {
            eprintln!(
                "[glrmask/profile][tokenizer] dense_binary_intersection left_states={} right_states={} pair_cells={} reachable_states={} class_entries={} classes={} discovery_ms={:.3} transition_rows_ms={:.3} metadata_ms={:.3} future_ms={:.3} expansion_ms={:.3} total_ms={:.3}",
                self.left_states,
                self.right_states,
                self.pair_cells,
                dfa.num_states(),
                self.pending_class_transition_classes.len(),
                self.num_classes,
                self.discovery_ms,
                transition_rows_ms,
                metadata_ms,
                future_ms,
                expansion_ms,
                started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }

        dfa
    }

    fn finish_compressed(mut self) -> (DFA, CompressedTransitionSegment) {
        let profile_timing = self.started_at.is_some();
        let transition_rows_started_at = profile_timing.then(Instant::now);
        self.materialize_pending_class_transitions()
            .expect("deferred dense transition rows must materialize");
        let transition_rows_ms = transition_rows_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let future_started_at = profile_timing.then(Instant::now);
        let has_future = compute_single_group_futures_from_class_graph_csr(
            &self.accepting,
            &self.pending_class_transition_offsets,
            &self.pending_class_transition_targets,
            self.cooperative_yield_interval,
        );
        let future_ms = future_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let metadata_started_at = profile_timing.then(Instant::now);
        debug_assert_eq!(self.metadata.num_groups(), 1);
        let group_u8set = *self.metadata.group_id_to_u8set(0);
        let dfa = DFA::new_with_single_group_metadata(
            &self.accepting,
            &has_future,
            group_u8set,
        );
        let metadata_ms = metadata_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let mut byte_to_class = [0u8; 256];
        let class_members = self
            .class_members
            .into_iter()
            .enumerate()
            .map(|(class, bytes)| {
                for &byte in &bytes {
                    byte_to_class[byte as usize] = class as u8;
                }
                bytes.into_boxed_slice()
            })
            .collect::<Vec<_>>();
        let expanded_transition_count = self
            .pending_class_transition_classes
            .iter()
            .map(|class| class_members[*class as usize].len())
            .sum();
        if let Some(started_at) = self.started_at {
            eprintln!(
                "[glrmask/profile][tokenizer] dense_binary_intersection left_states={} right_states={} pair_cells={} reachable_states={} class_entries={} classes={} discovery_ms={:.3} transition_rows_ms={:.3} metadata_ms={:.3} future_ms={:.3} expansion_ms=0.000 compressed=true total_ms={:.3}",
                self.left_states,
                self.right_states,
                self.pair_cells,
                dfa.num_states(),
                self.pending_class_transition_classes.len(),
                self.num_classes,
                self.discovery_ms,
                transition_rows_ms,
                metadata_ms,
                future_ms,
                started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        let segment = CompressedTransitionSegment {
            state_offset: 0,
            state_count: dfa.num_states() as u32,
            byte_to_class: Arc::from(byte_to_class.to_vec().into_boxed_slice()),
            class_members: Arc::from(class_members),
            row_offsets: Arc::from(self.pending_class_transition_offsets),
            entries: CompressedTransitionEntries::from_parts(
                self.pending_class_transition_classes,
                self.pending_class_transition_targets,
            ),
            expanded_transition_count,
        };
        (dfa, segment)
    }
}

fn try_discover_dense_binary_intersection_product(
    components: &[ProductComponent],
    class_members: &[Vec<u8>],
    component_class_transitions: &[ProductComponentClassTransitions],
    capture_trace: bool,
    defer_transition_rows: bool,
    defer_metadata: bool,
    profile_timing: bool,
) -> Option<(
    DeferredDenseBinaryIntersectionProduct,
    Option<ProductBuildTrace>,
)> {
    const MAX_DENSE_PAIR_CELLS: usize = 40_000_000;

    if components.len() != 2 || component_class_transitions.len() != 2 {
        return None;
    }
    let left = components[0].materialized_dfa()?;
    let right = components[1].materialized_dfa()?;
    let left_states = left.num_states();
    let right_states = right.num_states();
    let pair_cells = left_states.checked_mul(right_states)?;
    if pair_cells == 0 || pair_cells > MAX_DENSE_PAIR_CELLS {
        return None;
    }
    let num_classes = class_members.len();
    let cooperative_yield_interval = (defer_transition_rows || defer_metadata)
        .then(|| {
            std::env::var("GLRMASK_DEFERRED_DENSE_YIELD_INTERVAL")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(0)
        })
        .unwrap_or(0);
    let ProductComponentClassTransitions::Materialized(left_transitions) =
        &component_class_transitions[0]
    else {
        return None;
    };
    let ProductComponentClassTransitions::Materialized(right_transitions) =
        &component_class_transitions[1]
    else {
        return None;
    };
    let left_dead = components[0].dead_state();
    let right_dead = components[1].dead_state();

    let started_at = profile_timing.then(Instant::now);
    let discovery_started_at = profile_timing.then(Instant::now);
    let mut state_by_pair = vec![u32::MAX; pair_cells];
    state_by_pair[0] = 0;
    let mut pairs = vec![(0u32, 0u32)];
    let mut pending_class_transition_offsets = Vec::<u32>::new();
    pending_class_transition_offsets.push(0);
    let mut pending_class_transition_classes = Vec::<u8>::new();
    let mut pending_class_transition_targets = Vec::<u32>::new();
    let mut deferred_transition_count = 0usize;
    let mut metadata = DFA::new(1);
    metadata.ensure_group_capacity(1);
    let is_accepting = |left_state: u32, right_state: u32| {
        left.finalizers(left_state).contains(0) && right.finalizers(right_state).contains(0)
    };
    let mut accepting = vec![is_accepting(0, 0)];
    if accepting[0] && !defer_metadata {
        let mut finalizers = BitSet::new(1);
        finalizers.set(0);
        metadata.overwrite_state_metadata(0, finalizers, BitSet::new(1));
    }

    let mut cursor = 0usize;
    while cursor < pairs.len() {
        let (left_state, right_state) = pairs[cursor];
        let left_row = &left_transitions[left_state as usize];
        let right_row = &right_transitions[right_state as usize];
        let mut left_index = 0usize;
        let mut right_index = 0usize;
        while left_index < left_row.len() && right_index < right_row.len() {
            let (left_class, left_target) = left_row[left_index];
            let (right_class, right_target) = right_row[right_index];
            if left_class < right_class {
                left_index += 1;
                continue;
            }
            if right_class < left_class {
                right_index += 1;
                continue;
            }
            left_index += 1;
            right_index += 1;
            if left_target == u32::MAX
                || right_target == u32::MAX
                || left_dead == Some(left_target)
                || right_dead == Some(right_target)
            {
                continue;
            }
            let pair_index = (left_target as usize)
                .checked_mul(right_states)?
                .checked_add(right_target as usize)?;
            let target = if state_by_pair[pair_index] != u32::MAX {
                state_by_pair[pair_index]
            } else {
                let target = u32::try_from(pairs.len()).ok()?;
                state_by_pair[pair_index] = target;
                pairs.push((left_target, right_target));
                let target_accepting = is_accepting(left_target, right_target);
                accepting.push(target_accepting);
                if !defer_metadata {
                    let added = metadata.add_state();
                    debug_assert_eq!(added, target);
                    if target_accepting {
                        let mut finalizers = BitSet::new(1);
                        finalizers.set(0);
                        metadata.overwrite_state_metadata(
                            target,
                            finalizers,
                            BitSet::new(1),
                        );
                    }
                }
                debug_assert_eq!(accepting.len() - 1, target as usize);
                target
            };
            if defer_transition_rows {
                deferred_transition_count += 1;
            } else {
                pending_class_transition_classes.push(left_class);
                pending_class_transition_targets.push(target);
            }
        }
        pending_class_transition_offsets.push(u32::try_from(if defer_transition_rows {
            deferred_transition_count
        } else {
            pending_class_transition_classes.len()
        }).ok()?);
        cursor += 1;
    }
    let discovery_ms = discovery_started_at
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

    let trace = capture_trace.then(|| ProductBuildTrace {
            components: components.to_vec(),
            coordinate_groups: identity_product_coordinate_groups(components.len()),
            state_tuples: ProductStateTuples::DenseBinary(pairs),
            state_lookup: ProductStateLookup::DenseBinary {
                right_states,
                state_by_pair,
                overflow: FxHashMap::default(),
            },
            direct_single_visible_group: true,
    });

    Some((
        DeferredDenseBinaryIntersectionProduct {
            metadata,
            accepting,
            pending_class_transition_offsets,
            pending_class_transition_classes,
            pending_class_transition_targets,
            deferred_transition_rows: defer_transition_rows.then(|| {
                DeferredDenseBinaryTransitionRows {
                    pairs: Vec::new(),
                    state_by_pair: Vec::new(),
                    left_transitions: left_transitions.clone(),
                    right_transitions: right_transitions.clone(),
                    left_dead,
                    right_dead,
                }
            }),
            class_members: class_members.to_vec(),
            left_states,
            right_states,
            pair_cells,
            num_classes,
            cooperative_yield_interval,
            discovery_ms,
            started_at,
        },
        trace,
    ))
}

/// Exact dense compiler for a single visible terminal defined as the
/// intersection of two materialized deterministic components.
///
/// The ordinary product builder stores each reachable tuple in a hash table.
/// For a binary intersection every live product state is exactly one pair and
/// any transition with a dead component is dead for the logical terminal. A
/// dense `(left_state, right_state) -> product_state` table therefore provides
/// the same construction with constant-time array lookup and no tuple hashing.
pub(super) fn try_build_dense_binary_intersection_product(
    components: &[ProductComponent],
    class_members: &[Vec<u8>],
    component_class_transitions: &[ProductComponentClassTransitions],
    capture_trace: bool,
    profile_timing: bool,
) -> Option<(DFA, bool, Option<ProductBuildTrace>)> {
    let (deferred, trace) = try_discover_dense_binary_intersection_product(
        components,
        class_members,
        component_class_transitions,
        capture_trace,
        false,
        false,
        profile_timing,
    )?;
    Some((deferred.finish(), true, trace))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct CompactZeroMinRepeatTailState {
    body: Option<(u32, u32)>,
    suffix_state: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum CompactZeroMinRepeatState {
    Prefix(u32),
    Tail(CompactZeroMinRepeatTailState),
}

fn compact_zero_min_repeat_tail(
    state: &ZeroMinRepeatSuffixState,
) -> Option<CompactZeroMinRepeatTailState> {
    let mut body_iter = state
        .body_min_counts
        .iter()
        .enumerate()
        .filter(|(_, count)| **count != u32::MAX);
    let body = body_iter
        .next()
        .map(|(body_state, &completed)| (body_state as u32, completed));
    if body_iter.next().is_some() || state.suffix_states.len() > 1 {
        return None;
    }
    let suffix_state = state.suffix_states.first().copied();
    (body.is_some() || suffix_state.is_some()).then_some(CompactZeroMinRepeatTailState {
        body,
        suffix_state,
    })
}

fn compact_zero_min_repeat_start(
    lazy: &LazyZeroMinRepeatSuffixComponent,
) -> Option<CompactZeroMinRepeatState> {
    if lazy.prefix.is_empty() {
        compact_zero_min_repeat_tail(lazy.tail_states.first()?)
            .map(CompactZeroMinRepeatState::Tail)
    } else {
        Some(CompactZeroMinRepeatState::Prefix(0))
    }
}

fn compact_zero_min_repeat_step(
    lazy: &LazyZeroMinRepeatSuffixComponent,
    state: CompactZeroMinRepeatState,
    byte: u8,
) -> Result<Option<CompactZeroMinRepeatState>, ()> {
    let tail = match state {
        CompactZeroMinRepeatState::Prefix(position) => {
            let position = position as usize;
            if lazy.prefix.get(position).copied() != Some(byte) {
                return Ok(None);
            }
            let next = position + 1;
            if next < lazy.prefix.len() {
                return Ok(Some(CompactZeroMinRepeatState::Prefix(next as u32)));
            }
            let tail = compact_zero_min_repeat_tail(lazy.tail_states.first().ok_or(())?)
                .map(CompactZeroMinRepeatState::Tail)
                .map(Some)
                .ok_or(())?;
            return Ok(tail);
        }
        CompactZeroMinRepeatState::Tail(tail) => tail,
    };

    let mut body_candidates = SmallVec::<[(u32, u32); 2]>::new();
    let mut suffix_candidates = SmallVec::<[u32; 2]>::new();

    if let Some((body_state, completed)) = tail.body
        && let Some(target) = lazy.body_dfa.step(body_state, byte)
    {
        if lazy.body_dfa.finalizers(target).contains(0) {
            let completed = completed.saturating_add(1);
            suffix_candidates.push(0);
            if completed < lazy.max as u32 {
                body_candidates.push((0, completed));
            }
        }
        if lazy.body_dfa.possible_future_group_ids(target).contains(0) {
            body_candidates.push((target, completed));
        }
    }
    if let Some(suffix_state) = tail.suffix_state
        && let Some(target) = lazy.suffix_dfa.step(suffix_state, byte)
    {
        if lazy.suffix_dfa.finalizers(target).contains(0)
            || lazy
                .suffix_dfa
                .possible_future_group_ids(target)
                .contains(0)
        {
            suffix_candidates.push(target);
        }
    }

    body_candidates.sort_unstable();
    body_candidates.dedup();
    suffix_candidates.sort_unstable();
    suffix_candidates.dedup();
    if body_candidates.is_empty() && suffix_candidates.is_empty() {
        return Ok(None);
    }
    if body_candidates.len() > 1 || suffix_candidates.len() > 1 {
        return Err(());
    }
    Ok(Some(CompactZeroMinRepeatState::Tail(
        CompactZeroMinRepeatTailState {
            body: body_candidates.first().copied(),
            suffix_state: suffix_candidates.first().copied(),
        },
    )))
}

fn compact_zero_min_repeat_is_accepting(
    lazy: &LazyZeroMinRepeatSuffixComponent,
    state: CompactZeroMinRepeatState,
) -> bool {
    match state {
        CompactZeroMinRepeatState::Prefix(_) => false,
        CompactZeroMinRepeatState::Tail(tail) => tail
            .suffix_state
            .is_some_and(|suffix_state| lazy.suffix_dfa.finalizers(suffix_state).contains(0)),
    }
}

/// Build the exact runtime-only form of a large zero-minimum bounded-repeat
/// intersection when every reachable lazy residual has at most one body
/// state/count and at most one suffix state. This is the common maxLength string
/// shape. The helper is deliberately fail-closed: any transition requiring a
/// genuine residual set returns `None`, and the general lazy product remains
/// authoritative.
pub(super) fn select_lazy_zero_min_repeat_suffix_component(
    expressions: &[Expr],
) -> Option<(usize, LazyZeroMinRepeatSuffixComponent)> {
    let giant_indices = expressions
        .iter()
        .enumerate()
        .filter_map(|(index, expr)| {
            expression_contains_large_bounded_repeat(expr).then_some(index)
        })
        .collect::<SmallVec<[usize; 2]>>();
    match giant_indices.as_slice() {
        [index] => LazyZeroMinRepeatSuffixComponent::from_expr(&expressions[*index])
            .map(|lazy| (*index, lazy)),
        [] => expressions.iter().enumerate().find_map(|(index, expr)| {
            LazyZeroMinRepeatSuffixComponent::from_expr(expr).map(|lazy| (index, lazy))
        }),
        _ => None,
    }
}

pub(super) fn try_compile_compact_zero_min_repeat_intersection_runtime(
    plan: &ExclusionCompilePlan,
    profile_timing: bool,
) -> Option<DeferredDfa> {
    if plan.visible_groups != 1
        || plan.compiled_exprs.len() != 2
        || !pure_binary_intersection(&plan.exclusions, &plan.intersections)
    {
        return None;
    }

    let started_at = profile_timing.then(Instant::now);
    let (lazy_index, lazy) =
        select_lazy_zero_min_repeat_suffix_component(&plan.compiled_exprs)?;
    let other_index = 1 - lazy_index;
    let other_component = compile_lazy_intersection_materialized_other_component(
        &plan.compiled_exprs[other_index],
        false,
    )?;
    let other_dfa = other_component.materialized_dfa()?;
    let other_dead = other_component.dead_state();
    let (class_map, class_members) =
        compute_lazy_zero_min_repeat_product_equivalence_classes(&lazy, other_dfa);
    let other_transitions = build_product_class_transitions_for_dfa(other_dfa, &class_map);

    let start_lazy = compact_zero_min_repeat_start(&lazy)?;
    let start = (start_lazy, 0u32);
    let mut state_by_pair = FxHashMap::<(CompactZeroMinRepeatState, u32), u32>::default();
    state_by_pair.insert(start, 0);
    let mut pairs = vec![start];
    let mut accepting = vec![compact_zero_min_repeat_is_accepting(&lazy, start_lazy)
        && other_dfa.finalizers(0).contains(0)];
    // The whole point of this runtime form is that storage is proportional to
    // reachable product residuals, not to the syntactic repetition bound.
    // Reserving `max` here made a billion-bound repeat try to reserve billions
    // of offsets even when only a handful of residuals survive the intersecting
    // automaton.
    let mut row_offsets = Vec::<u32>::new();
    row_offsets.push(0);
    let mut row_classes = Vec::<u8>::new();
    let mut row_targets = Vec::<u32>::new();

    let discovery_started_at = profile_timing.then(Instant::now);
    let mut cursor = 0usize;
    while cursor < pairs.len() {
        let (lazy_state, other_state) = pairs[cursor];
        for &(class, other_target) in &other_transitions[other_state as usize] {
            if other_dead == Some(other_target)
                || !single_group_dfa_state_is_live(other_dfa, other_target)
            {
                continue;
            }
            let representative = *class_members[class as usize].first()?;
            let lazy_target = match compact_zero_min_repeat_step(
                &lazy,
                lazy_state,
                representative,
            ) {
                Ok(Some(target)) => target,
                Ok(None) => continue,
                Err(()) => return None,
            };
            let key = (lazy_target, other_target);
            let target = if let Some(&target) = state_by_pair.get(&key) {
                target
            } else {
                let target = u32::try_from(pairs.len()).ok()?;
                state_by_pair.insert(key, target);
                pairs.push(key);
                accepting.push(
                    compact_zero_min_repeat_is_accepting(&lazy, lazy_target)
                        && other_dfa.finalizers(other_target).contains(0),
                );
                target
            };
            row_classes.push(class);
            row_targets.push(target);
        }
        row_offsets.push(u32::try_from(row_classes.len()).ok()?);
        cursor += 1;
    }
    let discovery_ms = discovery_started_at
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

    let future_started_at = profile_timing.then(Instant::now);
    let has_future = compute_single_group_futures_from_class_graph_csr(
        &accepting,
        &row_offsets,
        &row_targets,
        0,
    );
    let future_ms = future_started_at
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let dfa = DFA::new_with_single_group_metadata(
        &accepting,
        &has_future,
        expr_u8set(&plan.compiled_exprs[0]),
    );

    let mut byte_to_class = [0u8; 256];
    let class_members = class_members
        .into_iter()
        .enumerate()
        .map(|(class, bytes)| {
            for &byte in &bytes {
                byte_to_class[byte as usize] = class as u8;
            }
            bytes.into_boxed_slice()
        })
        .collect::<Vec<_>>();
    let expanded_transition_count = row_classes
        .iter()
        .map(|class| class_members[*class as usize].len())
        .sum();
    let segment = CompressedTransitionSegment {
        state_offset: 0,
        state_count: dfa.num_states() as u32,
        byte_to_class: Arc::from(byte_to_class.to_vec().into_boxed_slice()),
        class_members: Arc::from(class_members),
        row_offsets: Arc::from(row_offsets),
        entries: CompressedTransitionEntries::from_parts(row_classes, row_targets),
        expanded_transition_count,
    };
    if let Some(started_at) = started_at {
        eprintln!(
            "[glrmask/profile][tokenizer] compact_zero_min_repeat_runtime lazy_component={} product_states={} classes={} body_states={} suffix_states={} repeat_max={} discovery_ms={:.3} future_ms={:.3} total_ms={:.3}",
            lazy_index,
            dfa.num_states(),
            segment.class_members.len(),
            lazy.body_dfa.num_states(),
            lazy.suffix_dfa.num_states(),
            lazy.max,
            discovery_ms,
            future_ms,
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Some(DeferredDfa::ReadyCompressed { dfa, segment })
}

pub(super) fn try_compile_lazy_zero_min_repeat_intersection(
    plan: &ExclusionCompilePlan,
    profile_timing: bool,
) -> Option<(DFA, ProductBuildTrace)> {
    if plan.visible_groups != 1
        || plan.compiled_exprs.len() != 2
        || !pure_binary_intersection(&plan.exclusions, &plan.intersections)
    {
        return None;
    }

    let started_at = profile_timing.then(Instant::now);
    let (lazy_index, mut lazy) =
        select_lazy_zero_min_repeat_suffix_component(&plan.compiled_exprs)?;
    let other_index = 1 - lazy_index;
    let other_component = compile_lazy_intersection_materialized_other_component(
        &plan.compiled_exprs[other_index],
        true,
    )?;
    let other_dfa = other_component.materialized_dfa()?;
    let other_dead = other_component.dead_state();

    let (class_map, class_members) =
        compute_lazy_zero_min_repeat_product_equivalence_classes(&lazy, other_dfa);
    lazy.prepare_classes(class_members.len());
    let other_transitions = build_product_class_transitions_for_dfa(other_dfa, &class_map);

    let mut dfa = DFA::new(1);
    dfa.ensure_group_capacity(1);
    dfa.set_group_u8set(0, expr_u8set(&plan.compiled_exprs[0]));
    let start_accepting = lazy.is_accepting(0) && other_dfa.finalizers(0).contains(0);
    if start_accepting {
        let mut finalizers = BitSet::new(1);
        finalizers.set(0);
        dfa.overwrite_state_metadata(0, finalizers, BitSet::new(1));
    }

    let mut pairs = vec![(0u32, 0u32)];
    let mut state_by_lazy_other = FxHashMap::<(u32, u32), u32>::default();
    state_by_lazy_other.insert((0, 0), 0);
    let mut pending_class_transitions = vec![Vec::<(u8, u32)>::new()];
    let discovery_started_at = profile_timing.then(Instant::now);
    let mut cursor = 0usize;
    while cursor < pairs.len() {
        let pair = pairs[cursor];
        let (lazy_state, other_state) = if lazy_index == 0 {
            (pair.0, pair.1)
        } else {
            (pair.1, pair.0)
        };
        let mut row = Vec::with_capacity(other_transitions[other_state as usize].len());
        for &(class, other_target) in &other_transitions[other_state as usize] {
            if other_dead == Some(other_target)
                || !single_group_dfa_state_is_live(other_dfa, other_target)
            {
                continue;
            }
            let representative = *class_members[class as usize].first()?;
            let Some(lazy_target) = lazy.step_class(lazy_state, class, representative) else {
                continue;
            };
            let target_pair = if lazy_index == 0 {
                (lazy_target, other_target)
            } else {
                (other_target, lazy_target)
            };
            let lookup_key = (lazy_target, other_target);
            let target = if let Some(&existing) = state_by_lazy_other.get(&lookup_key) {
                existing
            } else {
                let target = u32::try_from(pairs.len()).ok()?;
                state_by_lazy_other.insert(lookup_key, target);
                pairs.push(target_pair);
                pending_class_transitions.push(Vec::new());
                let added = dfa.add_state();
                debug_assert_eq!(added, target);
                let accepting = lazy.is_accepting(lazy_target)
                    && other_dfa.finalizers(other_target).contains(0);
                if accepting {
                    let mut finalizers = BitSet::new(1);
                    finalizers.set(0);
                    dfa.overwrite_state_metadata(target, finalizers, BitSet::new(1));
                }
                target
            };
            row.push((class, target));
        }
        pending_class_transitions[cursor] = row;
        cursor += 1;
    }
    let discovery_ms = discovery_started_at
        .map(|started| started.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    let future_started_at = profile_timing.then(Instant::now);
    set_single_group_futures_from_class_graph(&mut dfa, &pending_class_transitions);
    let future_ms = future_started_at
        .map(|started| started.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);
    let expansion_started_at = profile_timing.then(Instant::now);
    let expanded_transitions = pending_class_transitions
        .into_par_iter()
        .map(|class_transitions| {
            let byte_capacity = class_transitions
                .iter()
                .map(|(class, _)| class_members[*class as usize].len())
                .sum();
            let mut transitions = Vec::with_capacity(byte_capacity);
            for (class, target) in class_transitions {
                for &byte in &class_members[class as usize] {
                    transitions.push((byte, target));
                }
            }
            if transitions.len() > 1 {
                transitions.sort_unstable_by_key(|entry| entry.0);
            }
            crate::ds::char_transitions::CharTransitions::from_sorted_entries(transitions)
        })
        .collect::<Vec<_>>();
    for (state, transitions) in dfa.states_mut().iter_mut().zip(expanded_transitions) {
        state.transitions = transitions;
    }
    let expansion_ms = expansion_started_at
        .map(|started| started.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    let component_started_at = profile_timing.then(Instant::now);
    let body_states = lazy.body_dfa.num_states();
    let suffix_states = lazy.suffix_dfa.num_states();
    let repeat_max = lazy.max;
    let lazy_component = lazy.into_product_component(&class_members);
    let lazy_states = lazy_component
        .materialized_dfa()
        .map(DFA::num_states)?;
    let other_states = other_dfa.num_states();
    let components = if lazy_index == 0 {
        vec![lazy_component, other_component]
    } else {
        vec![other_component, lazy_component]
    };
    let component_ms = component_started_at
        .map(|started| started.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);
    let lookup_started_at = profile_timing.then(Instant::now);
    let mut sparse_lookup = FxHashMap::<ProductStateTuple, u32>::default();
    sparse_lookup.reserve(pairs.len());
    for (state, &(left, right)) in pairs.iter().enumerate() {
        let mut tuple = ProductStateTuple::new();
        tuple.push((0, left));
        tuple.push((1, right));
        sparse_lookup.insert(tuple, state as u32);
    }
    let trace = ProductBuildTrace {
        components,
        coordinate_groups: identity_product_coordinate_groups(2),
        state_tuples: ProductStateTuples::DenseBinary(pairs),
        state_lookup: ProductStateLookup::Hash(sparse_lookup),
        direct_single_visible_group: true,
    };
    let lookup_ms = lookup_started_at
        .map(|started| started.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    if let Some(started_at) = started_at {
        eprintln!(
            "[glrmask/profile][tokenizer] lazy_zero_min_repeat_intersection lazy_component={} lazy_states={} other_states={} product_states={} classes={} body_states={} suffix_states={} repeat_max={} discovery_ms={:.3} future_ms={:.3} expansion_ms={:.3} component_ms={:.3} lookup_ms={:.3} total_ms={:.3}",
            lazy_index,
            lazy_states,
            other_states,
            dfa.num_states(),
            class_members.len(),
            body_states,
            suffix_states,
            repeat_max,
            discovery_ms,
            future_ms,
            expansion_ms,
            component_ms,
            lookup_ms,
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Some((dfa, trace))
}

pub(super) fn try_compile_with_plan_deferred_dense(
    plan: ExclusionCompilePlan,
) -> Result<(DeferredDfa, Option<ProductBuildTrace>), ExclusionCompilePlan> {
    try_compile_with_plan_deferred_dense_min_pair_cells(plan, 0, false)
}

pub(super) fn try_compile_with_plan_deferred_dense_min_pair_cells(
    plan: ExclusionCompilePlan,
    min_pair_cells: usize,
    retain_transition_rows_during_discovery: bool,
) -> Result<(DeferredDfa, Option<ProductBuildTrace>), ExclusionCompilePlan> {
    if plan.visible_groups != 1
        || plan.compiled_exprs.len() != 2
        || !pure_binary_intersection(&plan.exclusions, &plan.intersections)
    {
        return Err(plan);
    }

    let profile_detail = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TRACE").is_some()
        || std::env::var_os("GLRMASK_PROFILE_TOKENIZER_DETAIL").is_some();
    let profile_timing = profile_detail
        || std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    // For giant nested repeats this is a correctness/safety lane, not merely a
    // performance optimization. A debug override must not route an expression
    // that validation accepted back into eager repeat materialization.
    let requires_lazy_giant = plan
        .compiled_exprs
        .iter()
        .any(expression_contains_large_bounded_repeat);
    let lazy_zero_min_repeat_product = requires_lazy_giant
        || std::env::var_os("GLRMASK_DISABLE_LAZY_ZERO_MIN_REPEAT_PRODUCT").is_none();
    if retain_transition_rows_during_discovery
        && lazy_zero_min_repeat_product
        && let Some(runtime) =
            try_compile_compact_zero_min_repeat_intersection_runtime(&plan, profile_timing)
    {
        return Ok((runtime, None));
    }
    if lazy_zero_min_repeat_product
        && let Some((dfa, trace)) =
            try_compile_lazy_zero_min_repeat_intersection(&plan, profile_timing)
    {
        return Ok((DeferredDfa::Ready(dfa), Some(trace)));
    }
    let (components, component_cache_hits, _, _) =
        compile_product_components_profiled(
            &plan.compiled_exprs,
            profile_detail,
            true,
            true,
            false,
        );
    let pair_cells = components
        .first()
        .and_then(ProductComponent::materialized_dfa)
        .and_then(|left| {
            components
                .get(1)
                .and_then(ProductComponent::materialized_dfa)
                .and_then(|right| left.num_states().checked_mul(right.num_states()))
        });
    if pair_cells.is_none_or(|pair_cells| pair_cells < min_pair_cells) {
        return Err(plan);
    }
    if profile_timing {
        eprintln!(
            "[glrmask/profile][tokenizer] deferred_product_component_cache groups={} unique_components={} cache_hits={}",
            plan.compiled_exprs.len(),
            plan.compiled_exprs.len() - component_cache_hits,
            component_cache_hits,
        );
    }
    let (class_map, class_members) = compute_product_equivalence_classes(&components);
    let component_class_transitions = build_product_class_transitions(&components, &class_map);
    let defer_runtime_materialization =
        std::env::var("GLRMASK_DEFER_DENSE_RUNTIME_MATERIALIZATION")
            .map(|value| {
                let value = value.trim();
                !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
            })
            .unwrap_or_else(|_| rayon::current_num_threads() > 1);
    let retain_transition_rows_during_discovery =
        retain_transition_rows_during_discovery && defer_runtime_materialization;
    let Some((mut deferred, trace)) = try_discover_dense_binary_intersection_product(
        &components,
        &class_members,
        &component_class_transitions,
        !retain_transition_rows_during_discovery,
        defer_runtime_materialization && !retain_transition_rows_during_discovery,
        defer_runtime_materialization,
        profile_timing,
    ) else {
        return Err(plan);
    };

    deferred.ensure_group_capacity(1);
    deferred.set_group_u8set(0, expr_u8set(&plan.compiled_exprs[0]));
    Ok((DeferredDfa::DenseBinary(deferred), trace))
}

/// Return whether one terminal expression has the exact pure binary-product
/// shape supported by the compressed runtime tokenizer builder.
pub fn expression_supports_deferred_dense_runtime(expr: &Expr) -> bool {
    // `build_exclusion_compile_plan([expr])` can be expensive because it
    // materializes nested group operations before returning its outer plan.
    // A pure binary intersection is possible only when the top-level group-op
    // chain contributes exactly one intersection and no exclusions. Nested
    // group operations inside either operand are materialized into that
    // operand and cannot create another outer plan component. Reject every
    // other shape before doing the expensive exact plan construction; all
    // plausible candidates still go through the historical proof below.
    fn outer_group_op_counts(expr: &Expr) -> (usize, usize) {
        match expr {
            Expr::Exclude { expr, .. } => {
                let (exclusions, intersections) = outer_group_op_counts(expr);
                (exclusions + 1, intersections)
            }
            Expr::Intersect { expr, .. } => {
                let (exclusions, intersections) = outer_group_op_counts(expr);
                (exclusions, intersections + 1)
            }
            Expr::Shared(inner)
                if matches!(inner.as_ref(), Expr::Exclude { .. } | Expr::Intersect { .. }) =>
            {
                outer_group_op_counts(inner)
            }
            _ => (0, 0),
        }
    }
    if outer_group_op_counts(expr) != (0, 1) {
        return false;
    }
    let plan = build_exclusion_compile_plan(std::slice::from_ref(expr));
    plan.visible_groups == 1
        && plan.compiled_exprs.len() == 2
        && pure_binary_intersection(&plan.exclusions, &plan.intersections)
}

/// Whether the exact general residual runtime has the bounded-code liveness
/// certificate needed to make this expression a safe protected dynamic
/// component. This is an internal representation-policy predicate; it does not
/// change the accepted language.
#[doc(hidden)]
pub fn expression_supports_bounded_code_residual_runtime(expr: &Expr) -> bool {
    crate::automata::lexer::runtime_residual::expression_supports_bounded_code_liveness_oracle(expr)
}

/// Cheap shape-only prefilter used by Static compilation before the preserving
/// residual constructor performs the full exact oracle certification.
#[doc(hidden)]
pub fn expression_may_support_bounded_code_residual_runtime(expr: &Expr) -> bool {
    crate::automata::lexer::runtime_residual::expression_may_support_bounded_code_liveness_oracle(expr)
}

/// Whether this expression contains any bounded repetition large enough that
/// falling through to the ordinary repeat compiler could allocate in direct
/// proportion to the declared bound.
#[doc(hidden)]
pub fn expression_contains_large_bounded_repeat(expr: &Expr) -> bool {
    match unwrap_shared(expr) {
        Expr::Repeat { expr, max, .. } => {
            max.is_some_and(|max| max >= VIRTUAL_BINARY_REPEAT_MIN_BOUND)
                || expression_contains_large_bounded_repeat(expr)
        }
        Expr::Seq(parts) | Expr::Choice(parts) => {
            parts.iter().any(expression_contains_large_bounded_repeat)
        }
        Expr::Intersect { expr, intersect } => {
            expression_contains_large_bounded_repeat(expr)
                || expression_contains_large_bounded_repeat(intersect)
        }
        Expr::Exclude { expr, exclude } => {
            expression_contains_large_bounded_repeat(expr)
                || expression_contains_large_bounded_repeat(exclude)
        }
        Expr::U8Seq(_) | Expr::U8Class(_) | Expr::Dfa(_) | Expr::Epsilon => false,
        Expr::Shared(_) => unreachable!("unwrap_shared removes Shared"),
    }
}

const LAZY_INTERSECTION_OTHER_MATERIALIZED_STATE_BUDGET: usize = 100_000;

const LAZY_INTERSECTION_OTHER_MATERIALIZED_TRANSITION_BUDGET: usize = 1_000_000;

/// Compile the ordinary side of the lazy zero-min-repeat/suffix intersection
/// lane without ever expanding a giant bounded repeat. The generic product
/// compiler represents deterministic bounded roots with `VirtualBoundedRepeat`
/// once their bound reaches the direct-repeat threshold, even when the exact
/// materialized DFA would still be tiny. That representation choice should not
/// force the lazy intersection lane to reject an otherwise cheap component.
///
/// Keep the ordinary coordinate independent of the giant repeat bound. Its
/// accepted language must be finite, so its live DFA is acyclic and bounds the
/// length of every synchronized product walk. For a non-giant virtual repeat,
/// the new layered expansion is additionally limited by its exact
/// `(max + 1) * base_states` footprint and a conservative
/// `max * base_transitions` bound. Validation calls this same helper, so an
/// expression cannot be admitted under a weaker proof than compilation uses.
pub(super) fn compile_lazy_intersection_materialized_other_component(
    expr: &Expr,
    preserve_coordinates: bool,
) -> Option<ProductComponent> {
    if expression_contains_large_bounded_repeat(expr) {
        return None;
    }
    let component =
        compile_product_component_with_options(expr, preserve_coordinates, true, None);
    let component = match component {
        component @ (ProductComponent::Materialized(_)
        | ProductComponent::MaterializedZeroMinRepeatSuffix { .. }) => Some(component),
        ProductComponent::VirtualBoundedRepeat {
            base_dfa,
            min,
            max,
        } => {
            let total_states = (max as usize + 1).checked_mul(base_dfa.num_states())?;
            let total_transitions = (max as usize).checked_mul(dfa_transition_count(&base_dfa))?;
            if total_states > LAZY_INTERSECTION_OTHER_MATERIALIZED_STATE_BUDGET
                || total_transitions > LAZY_INTERSECTION_OTHER_MATERIALIZED_TRANSITION_BUDGET
            {
                return None;
            }
            let mut dfa = build_bounded_repeat_dfa_from_base(
                base_dfa.as_ref(),
                min as usize,
                max as usize,
            )?;
            dfa.ensure_group_capacity(1);
            dfa.set_group_u8set(0, expr_u8set(expr));
            Some(ProductComponent::Materialized(Arc::new(dfa)))
        }
        ProductComponent::VirtualFixedSequence { .. } => None,
    }?;
    product_component_has_finite_language(&component).then_some(component)
}
