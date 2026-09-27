//! Assemble terminal partitions and adaptively combine compatible lexer components.

use crate::automata::lexer::ast::Expr;
use crate::automata::lexer::dfa::DFA;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::ds::bitset::BitSet;
use crate::ds::u8set::U8Set;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::hash::Hash;
use std::sync::Arc;
use std::time::Instant;
use super::{Regex, compile_with_plan};
use super::deferred::DeferredDfa;
use super::dfa_analysis::{dfa_transition_count, refine_u8_partitions};
use super::plan::{
    NestedGroupOpCache,
    SharedDuplicateNestedGroupOpCache,
    build_exclusion_compile_plan,
    build_exclusion_compile_plan_with_labels,
    build_exclusion_compile_plan_with_labels_and_cache,
    expr_profile_summary,
    materialize_repeated_subexpression_dfas,
    prewarm_shared_duplicate_nested_group_ops,
    shared_duplicate_nested_group_op_cache,
};
use super::product::{
    ProductStateTuple,
    build_product_class_transitions_for_dfa,
    explicit_dead_sink_state,
};
use super::settings::{
    adaptive_lexer_bounded_overhead_states,
    adaptive_lexer_growth_percent,
    adaptive_lexer_max_depth,
    adaptive_lexer_state_limit,
    adaptive_lexer_transition_growth_percent,
    adaptive_transition_growth_is_acceptable,
};
use rayon::prelude::*;

pub(super) fn compile_terminal_ids(
    exprs: &[Expr],
    visible_labels: Option<&[String]>,
    terminal_ids: &[usize],
) -> DFA {
    compile_terminal_ids_with_shared_duplicate_cache(exprs, visible_labels, terminal_ids, None)
}

pub(super) fn compile_terminal_ids_with_shared_duplicate_cache(
    exprs: &[Expr],
    visible_labels: Option<&[String]>,
    terminal_ids: &[usize],
    shared_duplicates: Option<&Arc<SharedDuplicateNestedGroupOpCache>>,
) -> DFA {
    let profile = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    let total_started_at = profile.then(Instant::now);
    let clone_started_at = profile.then(Instant::now);
    let local_exprs = terminal_ids
        .iter()
        .map(|&terminal| exprs[terminal].clone())
        .collect::<Vec<_>>();
    let local_labels = visible_labels.map(|labels| {
        terminal_ids
            .iter()
            .map(|&terminal| labels[terminal].clone())
            .collect::<Vec<_>>()
    });
    let clone_ms = clone_started_at
        .map(|started| started.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);
    let mut nested_group_op_cache = NestedGroupOpCache {
        shared_duplicates: shared_duplicates.cloned(),
        ..NestedGroupOpCache::default()
    };
    let plan_started_at = profile.then(Instant::now);
    let plan = build_exclusion_compile_plan_with_labels_and_cache(
        &local_exprs,
        local_labels.as_deref(),
        &mut nested_group_op_cache,
    );
    let plan_ms = plan_started_at
        .map(|started| started.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);
    let compile_started_at = profile.then(Instant::now);
    let dfa = compile_with_plan(plan);
    if let Some(total_started_at) = total_started_at {
        eprintln!(
            "[glrmask/profile][tokenizer] partition_plan_detail terminals={} clone_ms={:.3} plan_ms={:.3} nested_entries={} nested_hits={} nested_misses={} nested_compiled_ms={:.3} nested_max_compile_ms={:.3} compile_ms={:.3} total_ms={:.3}",
            terminal_ids.len(),
            clone_ms,
            plan_ms,
            nested_group_op_cache.compiled.len(),
            nested_group_op_cache.cache_hits,
            nested_group_op_cache.cache_misses,
            nested_group_op_cache.compiled_ms,
            nested_group_op_cache.max_compile_ms,
            compile_started_at
                .map(|started| started.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0),
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    dfa
}

pub(super) struct LexerComponent {
    pub(super) terminal_ids: Vec<usize>,
    pub(super) dfa: DFA,
    pub(super) protected_residual: bool,
}

pub(super) struct LexerComponentPair {
    pub(super) terminal_ids: Vec<usize>,
    pub(super) synthesized: DFA,
    pub(super) full: DeferredDfa,
    pub(super) full_to_synthesized: Vec<u32>,
    pub(super) protected_residual: bool,
}

pub(super) fn isolate_component_nullable_start(dfa: DFA, num_terminals: usize) -> (DFA, BTreeSet<u32>) {
    let mut tokenizer = Regex { dfa }.into_tokenizer(num_terminals as u32, None);
    let nullable = tokenizer.isolate_start_state_and_drain_nullable_terminals();
    (tokenizer.dfa, nullable)
}

pub struct CompiledPartitionedExpressionPair {
    pub synthesized: Regex,
    pub full: Regex,
    pub full_to_synthesized: Vec<u32>,
}

pub struct PreparedPartitionedExpressionPair {
    pub synthesized: Regex,
    pub(super) full: DeferredPartitionedRegex,
    pub full_to_synthesized: Vec<u32>,
    pub synthesized_expressions: Vec<Expr>,
}

impl PreparedPartitionedExpressionPair {
    pub fn full_num_states(&self) -> usize {
        self.full.num_states()
    }

    pub fn finish_full(self) -> CompiledPartitionedExpressionPair {
        CompiledPartitionedExpressionPair {
            synthesized: self.synthesized,
            full: self.full.finish(),
            full_to_synthesized: self.full_to_synthesized,
        }
    }

    pub fn into_parts(
        self,
    ) -> (Regex, DeferredPartitionedRegex, Vec<u32>, Vec<Expr>) {
        (
            self.synthesized,
            self.full,
            self.full_to_synthesized,
            self.synthesized_expressions,
        )
    }
}

pub struct DeferredPartitionedRegex {
    pub(super) components: Vec<LexerComponentPair>,
    pub(super) total_groups: usize,
}

impl DeferredPartitionedRegex {
    pub fn num_states(&self) -> usize {
        1 + self
            .components
            .iter()
            .map(|component| component.full.num_states())
            .sum::<usize>()
    }

    pub fn has_deferred_runtime_materialization(&self) -> bool {
        self.components
            .iter()
            .any(|component| component.full.has_deferred_runtime_materialization())
    }

    pub fn finish(self) -> Regex {
        let components = self
            .components
            .into_iter()
            .map(|component| LexerComponent {
                terminal_ids: component.terminal_ids,
                dfa: component.full.finish(),
                protected_residual: component.protected_residual,
            })
            .collect::<Vec<_>>();
        Regex {
            dfa: combine_lexer_components_under_epsilon_root(components, self.total_groups),
        }
    }

    pub fn finish_runtime_tokenizer(
        self,
        num_terminals: u32,
        expressions: Arc<[Expr]>,
    ) -> Tokenizer {
        let mut combined = DFA::new(1);
        combined.ensure_group_capacity(self.total_groups);
        let mut root_futures = BitSet::new(self.total_groups);
        let mut compressed_segments = Vec::new();

        for component in self.components {
            let terminal_ids = component.terminal_ids;
            let (component_dfa, compressed) = component.full.finish_runtime();
            debug_assert_eq!(component_dfa.num_groups(), terminal_ids.len());
            for local_group in component_dfa.possible_future_group_ids(0).iter() {
                root_futures.set(terminal_ids[local_group]);
            }
            let offset = combined.append_rebased_component(component_dfa, &terminal_ids);
            combined.add_epsilon_transition(0, offset);
            if let Some(mut segment) = compressed {
                segment.state_offset = offset;
                compressed_segments.push(segment);
            }
        }
        combined.set_possible_future_group_ids(0, root_futures);
        Tokenizer::from_parts_with_compressed_transitions(
            combined,
            num_terminals,
            Some(expressions),
            compressed_segments,
        )
    }
}

pub(super) fn compile_partition_components(
    exprs: &[Expr],
    visible_labels: Option<&[String]>,
    partitions: &[u32],
    residual_isolation_classes: Option<&[Option<u32>]>,
) -> Vec<LexerComponent> {
    let profile = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    let profile_top = profile
        || std::env::var_os("GLRMASK_PROFILE_DYNAMIC_TOKENIZER_TOP").is_some();
    let rewrite_started_at = Instant::now();
    let rewritten_exprs = materialize_repeated_subexpression_dfas(exprs);
    let rewrite_ms = rewrite_started_at.elapsed().as_secs_f64() * 1000.0;
    let exprs = rewritten_exprs.as_deref().unwrap_or(exprs);
    let setup_started_at = Instant::now();
    let mut grouped = BTreeMap::<u32, Vec<usize>>::new();
    for (terminal, &partition) in partitions.iter().enumerate() {
        grouped.entry(partition).or_default().push(terminal);
    }
    if let Some(classes) = residual_isolation_classes {
        assert_eq!(
            classes.len(),
            exprs.len(),
            "one residual isolation class entry is required per terminal",
        );
        for terminal_ids in grouped.values() {
            let mut class = None;
            let mut has_unprotected = false;
            for &terminal in terminal_ids {
                match classes[terminal] {
                    Some(current) => match class {
                        Some(previous) => assert_eq!(
                            previous, current,
                            "one lexer partition cannot contain distinct protected residual classes",
                        ),
                        None => class = Some(current),
                    },
                    None => has_unprotected = true,
                }
            }
            assert!(
                class.is_none() || !has_unprotected,
                "one lexer partition cannot mix protected and unprotected residual coordinates",
            );
        }
    }
    let shared_duplicates = shared_duplicate_nested_group_op_cache(exprs, &grouped);
    let prewarm_started_at = Instant::now();
    if let Some(shared_duplicates) = &shared_duplicates {
        prewarm_shared_duplicate_nested_group_ops(shared_duplicates);
    }
    let prewarm_ms = prewarm_started_at.elapsed().as_secs_f64() * 1000.0;
    let setup_ms = setup_started_at.elapsed().as_secs_f64() * 1000.0 - prewarm_ms;

    let compile_started_at = Instant::now();
    let components: Vec<LexerComponent> = grouped
        .into_iter()
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|(partition, terminal_ids)| {
            let started_at = Instant::now();
            let dfa = compile_terminal_ids_with_shared_duplicate_cache(
                exprs,
                visible_labels,
                &terminal_ids,
                shared_duplicates.as_ref(),
            );
            if profile {
                let single_label = (terminal_ids.len() == 1)
                    .then(|| {
                        visible_labels
                            .and_then(|labels| labels.get(terminal_ids[0]))
                            .map(String::as_str)
                    })
                    .flatten();
                let single_expr = (terminal_ids.len() == 1)
                    .then(|| expr_profile_summary(&exprs[terminal_ids[0]]));
                eprintln!(
                    "[glrmask/profile][tokenizer] partition_compile partition={} terminals={} terminal_ids={:?} single_label={:?} single_expr={:?} states={} transitions={} total_ms={:.3}",
                    partition,
                    terminal_ids.len(),
                    terminal_ids,
                    single_label,
                    single_expr,
                    dfa.num_states(),
                    dfa_transition_count(&dfa),
                    started_at.elapsed().as_secs_f64() * 1000.0,
                );
            }
            let protected_residual = residual_isolation_classes.is_some_and(|classes| {
                terminal_ids
                    .first()
                    .is_some_and(|&terminal| classes[terminal].is_some())
            });
            LexerComponent {
                terminal_ids,
                dfa,
                protected_residual,
            }
        })
        .collect();
    if profile_top {
        eprintln!(
            "[glrmask/profile][tokenizer_partition_components_top] terminals={} components={} rewrite_ms={:.3} setup_ms={:.3} prewarm_ms={:.3} compile_wall_ms={:.3} total_ms={:.3}",
            exprs.len(),
            components.len(),
            rewrite_ms,
            setup_ms,
            prewarm_ms,
            compile_started_at.elapsed().as_secs_f64() * 1000.0,
            rewrite_ms + setup_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    components
}

fn lexer_component_product_metadata(
    components: &[LexerComponent],
    group_offsets: &[usize],
    state_tuple: &ProductStateTuple,
    total_groups: usize,
) -> (BitSet, BitSet) {
    let mut finalizers = BitSet::new(total_groups);
    let mut futures = BitSet::new(total_groups);

    for &(component_id, component_state) in state_tuple {
        let component_index = component_id as usize;
        let component = &components[component_index].dfa;
        let offset = group_offsets[component_index];
        for group in component.finalizers(component_state).iter() {
            finalizers.set(offset + group);
        }
        for group in component
            .possible_future_group_ids(component_state)
            .iter()
        {
            futures.set(offset + group);
        }
    }

    (finalizers, futures)
}

fn compute_lexer_component_equivalence_classes(
    components: &[LexerComponent],
) -> (Vec<u8>, Vec<Vec<u8>>) {
    let mut partitions = vec![U8Set::all()];
    let mut seen_sets = FxHashSet::default();

    for component in components {
        for state in component.dfa.states() {
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

/// Attempt one exact prefix determinization of the final union of independently
/// compiled lexer partitions. The sparse product tuple contains only component
/// states still live after the consumed bytes. Product construction stops at
/// `max_depth` consumed bytes and reconnects each frontier tuple to exact copies
/// of its live component states with epsilon edges. Thus adaptive
/// determinization can coalesce a bounded prefix without forcing unrelated
/// long-running terminals into the product for their entire lifetime.
///
/// Construction also stops before allocating the first product state beyond
/// `state_limit`; callers can then preserve the original epsilon-NFA unchanged.
/// `None` retains the historical full-product behavior.
pub(super) fn try_product_union_components(
    components: &[LexerComponent],
    state_limit: usize,
    transition_limit: usize,
    max_depth: Option<usize>,
) -> Option<DFA> {
    assert!(state_limit > 0, "adaptive lexer state limit must be positive");
    assert!(transition_limit > 0, "adaptive lexer transition limit must be positive");
    debug_assert!(components
        .iter()
        .all(|component| !component.dfa.has_epsilon_transitions()));

    // A bounded product is not just its determinized prefix: after the cutoff
    // it appends exact copies of the component DFAs and reconnects frontier
    // tuples to them.  Account for those copies *before* constructing the
    // prefix so `state_limit` / `transition_limit` bound the final DFA rather
    // than only the speculative prefix.  The previous implementation applied
    // the state cap only to `combined` and then appended every component,
    // allowing a nominal 100% growth limit to produce a larger lexer.
    let copied_states = if max_depth.is_some() {
        components
            .iter()
            .map(|component| component.dfa.num_states())
            .sum::<usize>()
    } else {
        0
    };
    let copied_transitions = if max_depth.is_some() {
        components
            .iter()
            .map(|component| dfa_transition_count(&component.dfa))
            .sum::<usize>()
    } else {
        0
    };
    let product_state_limit = state_limit.checked_sub(copied_states)?;
    if product_state_limit == 0 {
        return None;
    }
    let product_transition_limit = transition_limit.checked_sub(copied_transitions)?;

    let profile = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_DETAIL").is_some()
        || std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    let setup_started_at = Instant::now();
    let total_groups = components
        .iter()
        .map(|component| component.dfa.num_groups())
        .sum::<usize>();
    let mut group_offsets = Vec::with_capacity(components.len());
    let mut group_offset = 0usize;
    for component in components {
        group_offsets.push(group_offset);
        group_offset += component.dfa.num_groups();
    }
    let component_dead_states = components
        .iter()
        .map(|component| explicit_dead_sink_state(&component.dfa))
        .collect::<Vec<_>>();
    let class_started_at = Instant::now();
    let (class_map, class_members) = compute_lexer_component_equivalence_classes(components);
    let class_ms = class_started_at.elapsed().as_secs_f64() * 1000.0;
    let class_transitions_started_at = Instant::now();
    let component_class_transitions = components
        .iter()
        .map(|component| build_product_class_transitions_for_dfa(&component.dfa, &class_map))
        .collect::<Vec<_>>();
    let class_transitions_ms = class_transitions_started_at.elapsed().as_secs_f64() * 1000.0;
    let setup_ms = setup_started_at.elapsed().as_secs_f64() * 1000.0;
    let num_classes = class_members.len();

    let mut combined = DFA::new(1);
    combined.ensure_group_capacity(total_groups);
    for (component_index, component) in components.iter().enumerate() {
        let offset = group_offsets[component_index];
        for local_group in 0..component.dfa.num_groups() {
            combined.set_group_u8set(
                (offset + local_group) as u32,
                *component.dfa.group_id_to_u8set(local_group as u32),
            );
        }
    }

    let mut start = ProductStateTuple::with_capacity(components.len());
    for component_id in 0..components.len() {
        start.push((component_id as u32, 0));
    }
    let (finalizers, futures) =
        lexer_component_product_metadata(components, &group_offsets, &start, total_groups);
    combined.overwrite_state_metadata(0, finalizers, futures);

    #[derive(Clone, PartialEq, Eq, Hash)]
    enum ProductStateKey {
        Full(ProductStateTuple),
        Bounded(usize, ProductStateTuple),
    }
    let state_key = |depth: usize, tuple: ProductStateTuple| match max_depth {
        Some(_) => ProductStateKey::Bounded(depth, tuple),
        None => ProductStateKey::Full(tuple),
    };

    let mut state_map = FxHashMap::<ProductStateKey, u32>::default();
    state_map.insert(state_key(0, start.clone()), 0);
    let mut worklist = VecDeque::from([(0u32, start, 0usize)]);
    let mut pending_class_transitions = vec![Vec::<(u8, u32)>::new()];
    let mut frontier_states = Vec::<(u32, ProductStateTuple)>::new();
    let mut class_buffers = (0..num_classes)
        .map(|_| ProductStateTuple::new())
        .collect::<Vec<_>>();
    let mut class_active = vec![false; num_classes];
    let mut used_classes = Vec::<usize>::new();
    let mut projected_byte_transitions = 0usize;

    let state_expand_started_at = Instant::now();
    while let Some((combined_state, state_tuple, depth)) = worklist.pop_front() {
        if max_depth.is_some_and(|max_depth| depth >= max_depth) {
            frontier_states.push((combined_state, state_tuple));
            continue;
        }
        for &(component_id, component_state) in &state_tuple {
            let component_index = component_id as usize;
            for &(class_id, target) in
                &component_class_transitions[component_index][component_state as usize]
            {
                let class_index = class_id as usize;
                if !class_active[class_index] {
                    class_active[class_index] = true;
                    used_classes.push(class_index);
                }
                if component_dead_states[component_index] == Some(target) {
                    continue;
                }
                class_buffers[class_index].push((component_id, target));
            }
        }

        let mut transitions = Vec::with_capacity(used_classes.len());
        for &class_index in &used_classes {
            projected_byte_transitions = projected_byte_transitions
                .saturating_add(class_members[class_index].len());
            if projected_byte_transitions > product_transition_limit {
                return None;
            }
            let next_tuple = &class_buffers[class_index];
            let next_depth = depth + 1;
            let next_key = state_key(next_depth, next_tuple.clone());
            let target = if let Some(&existing) = state_map.get(&next_key) {
                existing
            } else {
                if combined.num_states() >= product_state_limit {
                    return None;
                }
                let new_state = combined.add_state();
                let (finalizers, futures) = lexer_component_product_metadata(
                    components,
                    &group_offsets,
                    next_tuple,
                    total_groups,
                );
                combined.overwrite_state_metadata(new_state, finalizers, futures);
                state_map.insert(next_key, new_state);
                pending_class_transitions.push(Vec::new());
                worklist.push_back((new_state, next_tuple.clone(), next_depth));
                new_state
            };
            transitions.push((class_index as u8, target));
            class_buffers[class_index].clear();
            class_active[class_index] = false;
        }
        used_classes.clear();
        pending_class_transitions[combined_state as usize] = transitions;
    }
    let state_expand_ms = state_expand_started_at.elapsed().as_secs_f64() * 1000.0;

    let byte_expand_started_at = Instant::now();
    let expanded_transitions: Vec<crate::ds::char_transitions::CharTransitions<u32>> =
        pending_class_transitions
            .into_par_iter()
            .map(|class_transitions| {
                let byte_capacity = class_transitions
                    .iter()
                    .map(|(class_id, _)| class_members[*class_id as usize].len())
                    .sum::<usize>();
                const DENSE_BYTE_EXPANSION_THRESHOLD: usize = 96;
                let transitions = if byte_capacity >= DENSE_BYTE_EXPANSION_THRESHOLD {
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
            })
            .collect();
    for (state, transitions) in combined.states_mut().iter_mut().zip(expanded_transitions) {
        state.transitions = transitions;
    }

    if max_depth.is_some() {
        // Append one exact copy of every independently compiled component. A
        // product frontier can then resume from the precise component states in
        // its sparse tuple instead of continuing cross-component
        // determinization.
        let mut component_offsets = Vec::with_capacity(components.len());
        for (component_index, component) in components.iter().enumerate() {
            let component_offset = combined.num_states() as u32;
            component_offsets.push(component_offset);
            for _ in 0..component.dfa.num_states() {
                combined.add_state();
            }

            let group_offset = group_offsets[component_index];
            for (state_index, state) in component.dfa.states().iter().enumerate() {
                let mapped_state = component_offset + state_index as u32;
                combined.set_transitions_from_sorted_entries(
                    mapped_state,
                    state
                        .transitions
                        .iter()
                        .map(|(byte, &target)| (byte, component_offset + target))
                        .collect(),
                );
                for &target in &state.epsilon_transitions {
                    combined.add_epsilon_transition(mapped_state, component_offset + target);
                }

                let mut finalizers = BitSet::new(total_groups);
                let mut futures = BitSet::new(total_groups);
                for local_group in state.finalizers.iter() {
                    finalizers.set(group_offset + local_group);
                }
                for local_group in component
                    .dfa
                    .possible_future_group_ids(state_index as u32)
                    .iter()
                {
                    futures.set(group_offset + local_group);
                }
                combined.overwrite_state_metadata(mapped_state, finalizers, futures);
            }
        }

        for (frontier_state, state_tuple) in frontier_states {
            // The frontier is an epsilon branching state, like the root of the
            // ordinary partition union. Its exact component children carry
            // acceptance. Duplicating finalizers on the branch state creates a
            // second accepting path for the same terminal and is observably
            // different to the original epsilon-NFA for downstream state-set
            // analyses. Strict futures may remain cached on the branch state,
            // just as they are on the ordinary epsilon-union root.
            let futures = combined.possible_future_group_ids(frontier_state).clone();
            combined.overwrite_state_metadata(
                frontier_state,
                BitSet::new(total_groups),
                futures,
            );
            for (component_id, component_state) in state_tuple {
                combined.add_epsilon_transition(
                    frontier_state,
                    component_offsets[component_id as usize] + component_state,
                );
            }
        }

        debug_assert!(combined.num_states() <= state_limit);
        debug_assert!(dfa_transition_count(&combined) <= transition_limit);
    }
    let byte_expand_ms = byte_expand_started_at.elapsed().as_secs_f64() * 1000.0;

    if profile {
        eprintln!(
            "[glrmask/profile][tokenizer] adaptive_product states={} classes={} max_depth={:?} setup_ms={:.3} class_ms={:.3} class_transitions_ms={:.3} state_expand_ms={:.3} byte_expand_ms={:.3}",
            combined.num_states(),
            num_classes,
            max_depth,
            setup_ms,
            class_ms,
            class_transitions_ms,
            state_expand_ms,
            byte_expand_ms,
        );
    }

    Some(combined)
}

pub(super) fn adaptively_determinize_components(
    inputs: Vec<LexerComponent>,
    state_limit: usize,
) -> Vec<LexerComponent> {
    adaptively_determinize_components_with_limits(
        inputs,
        state_limit,
        adaptive_lexer_growth_percent(),
        adaptive_lexer_transition_growth_percent(),
        adaptive_lexer_max_depth(),
    )
}

pub(super) fn adaptively_determinize_components_with_limits(
    inputs: Vec<LexerComponent>,
    state_limit: usize,
    growth_percent: usize,
    transition_growth_percent: usize,
    max_depth: Option<usize>,
) -> Vec<LexerComponent> {
    if inputs.iter().any(|component| component.protected_residual) {
        let mut protected = Vec::new();
        let mut ordinary = Vec::new();
        for component in inputs {
            if component.protected_residual {
                protected.push(component);
            } else {
                ordinary.push(component);
            }
        }

        let mut output = if ordinary.len() >= 2 {
            adaptively_determinize_components_with_limits(
                ordinary,
                state_limit,
                growth_percent,
                transition_growth_percent,
                max_depth,
            )
        } else {
            ordinary
        };
        output.append(&mut protected);
        output.sort_unstable_by_key(|component| {
            component.terminal_ids.first().copied().unwrap_or(usize::MAX)
        });
        return output;
    }

    let profile = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_DETAIL").is_some()
        || std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    let started_at = Instant::now();
    let input_states = inputs.iter().map(|batch| batch.dfa.num_states()).sum::<usize>();
    let input_transitions = inputs
        .iter()
        .map(|batch| dfa_transition_count(&batch.dfa))
        .sum::<usize>();
    let input_batches = inputs.len();
    let terminals = inputs
        .iter()
        .map(|batch| batch.terminal_ids.len())
        .sum::<usize>();

    // Compare against the representation we would actually return if adaptive
    // determinization is rejected: the disjoint component DFAs plus their one
    // epsilon-union root.  Using only `input_states` made a 100% growth cap one
    // state stricter for full products while, paradoxically, bounded products
    // could exceed it after appending their component copies.
    let baseline_states = input_states.saturating_add(usize::from(input_batches > 1));
    let percentage_growth_limit = baseline_states
        .saturating_mul(growth_percent)
        .saturating_add(99)
        / 100;
    let effective_state_limit = match max_depth {
        None => state_limit.min(percentage_growth_limit).max(1),
        Some(_) => state_limit
            .min(
                baseline_states.saturating_add(adaptive_lexer_bounded_overhead_states()),
            )
            .max(1),
    };
    let transition_limit = input_transitions
        .saturating_mul(transition_growth_percent)
        / 100;
    let attempt_started_at = Instant::now();
    let candidate = try_product_union_components(
        &inputs,
        effective_state_limit,
        transition_limit.max(1),
        max_depth,
    );
    let attempt_ms = attempt_started_at.elapsed().as_secs_f64() * 1000.0;
    let candidate_transitions = candidate.as_ref().map(dfa_transition_count);
    let accepted = candidate.as_ref().is_some_and(|dfa| {
        dfa.num_states() <= effective_state_limit
            && adaptive_transition_growth_is_acceptable(
                input_transitions,
                dfa_transition_count(dfa),
                transition_growth_percent,
            )
    });
    let terminal_ids = inputs
        .iter()
        .flat_map(|component| component.terminal_ids.iter().copied())
        .collect::<Vec<_>>();
    let output_states = candidate.as_ref().map_or(input_states, DFA::num_states);
    let output_transitions = candidate_transitions.unwrap_or(input_transitions);

    if profile {
        eprintln!(
            "[glrmask/profile][tokenizer] adaptive_determinize partitions={} terminals={} output_components={} input_states={} baseline_states={} output_states={} input_transitions={} output_transitions={} accepted={} max_states={} effective_state_limit={} max_depth={:?} bounded_overhead_states={} max_growth_percent={} max_transition_growth_percent={} attempt_ms={:.3} total_ms={:.3}",
            input_batches,
            terminals,
            if accepted { 1 } else { input_batches },
            input_states,
            baseline_states,
            output_states,
            input_transitions,
            output_transitions,
            accepted,
            state_limit,
            effective_state_limit,
            max_depth,
            adaptive_lexer_bounded_overhead_states(),
            growth_percent,
            transition_growth_percent,
            attempt_ms,
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }

    match candidate {
        Some(dfa) if accepted => vec![LexerComponent {
            terminal_ids,
            dfa,
            protected_residual: false,
        }],
        _ => inputs,
    }
}

fn remap_component_groups(
    component: &DFA,
    terminal_ids: &[usize],
    total_groups: usize,
) -> DFA {
    debug_assert_eq!(component.num_groups(), terminal_ids.len());
    let mut remapped = DFA::new(component.num_states());
    remapped.ensure_group_capacity(total_groups);

    for (local_group, &terminal_id) in terminal_ids.iter().enumerate() {
        remapped.set_group_u8set(
            terminal_id as u32,
            *component.group_id_to_u8set(local_group as u32),
        );
    }

    for (state_index, state) in component.states().iter().enumerate() {
        remapped.set_transitions_from_sorted_entries(
            state_index as u32,
            state.transitions.iter().map(|(byte, &target)| (byte, target)).collect(),
        );
        for &target in &state.epsilon_transitions {
            remapped.add_epsilon_transition(state_index as u32, target);
        }

        let mut finalizers = BitSet::new(total_groups);
        let mut futures = BitSet::new(total_groups);
        for (local_group, &terminal_id) in terminal_ids.iter().enumerate() {
            if state.finalizers.contains(local_group) {
                finalizers.set(terminal_id);
            }
            if component
                .possible_future_group_ids(state_index as u32)
                .contains(local_group)
            {
                futures.set(terminal_id);
            }
        }
        remapped.overwrite_state_metadata(state_index as u32, finalizers, futures);
    }

    remapped
}

pub(super) fn compile_terminal_partitions(
    exprs: &[Expr],
    visible_labels: Option<&[String]>,
    partitions: &[u32],
    residual_isolation_classes: Option<&[Option<u32>]>,
    adaptive: bool,
) -> DFA {
    let profile = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some()
        || std::env::var_os("GLRMASK_PROFILE_DYNAMIC_TOKENIZER_TOP").is_some();
    let total_started_at = Instant::now();
    assert_eq!(exprs.len(), partitions.len(), "one lexer partition id is required per terminal");
    if let Some(classes) = residual_isolation_classes {
        assert_eq!(
            exprs.len(),
            classes.len(),
            "one residual isolation class entry is required per terminal",
        );
    }
    if let Some(labels) = visible_labels {
        assert_eq!(exprs.len(), labels.len(), "one profile label is required per terminal");
    }
    if exprs.is_empty() {
        return compile_with_plan(build_exclusion_compile_plan(exprs));
    }

    let num_partitions = partitions.iter().copied().collect::<BTreeSet<_>>().len();
    if num_partitions == 1 {
        return compile_with_plan(build_exclusion_compile_plan_with_labels(exprs, visible_labels));
    }

    // Every partition is compiled exactly as declared, independently of the
    // adaptive policy. Adaptive determinization is a second, generic step over
    // the disjoint deterministic components of the combined epsilon-NFA.
    let partition_compile_started_at = Instant::now();
    let mut components = compile_partition_components(
        exprs,
        visible_labels,
        partitions,
        residual_isolation_classes,
    );
    let partition_compile_ms = partition_compile_started_at.elapsed().as_secs_f64() * 1000.0;

    let adaptive_started_at = Instant::now();
    if adaptive {
        components =
            adaptively_determinize_components(components, adaptive_lexer_state_limit());
    }
    let adaptive_ms = adaptive_started_at.elapsed().as_secs_f64() * 1000.0;

    if let [component] = components.as_slice() {
        return remap_component_groups(&component.dfa, &component.terminal_ids, exprs.len());
    }

    let combine_started_at = Instant::now();
    let mut combined = DFA::new(1);
    combined.ensure_group_capacity(exprs.len());
    let mut root_futures = BitSet::new(exprs.len());

    for batch in components {
        for local_group in batch.dfa.possible_future_group_ids(0).iter() {
            root_futures.set(batch.terminal_ids[local_group]);
        }
        let offset = combined.append_rebased_component(batch.dfa, &batch.terminal_ids);
        combined.add_epsilon_transition(0, offset);
    }
    let combine_ms = combine_started_at.elapsed().as_secs_f64() * 1000.0;

    let futures_started_at = Instant::now();
    // Components are disjoint below a new epsilon-only root. Their strict
    // possible-future sets remain exact after local->global terminal remapping.
    // The root's strict futures are exactly the union of the component start
    // states' strict futures; no generic epsilon fixpoint is needed.
    combined.set_possible_future_group_ids(0, root_futures);
    let futures_ms = futures_started_at.elapsed().as_secs_f64() * 1000.0;
    if profile {
        eprintln!(
            "[glrmask/profile][tokenizer] partitioned_build terminals={} partitions={} partition_compile_ms={:.3} adaptive_ms={:.3} combine_ms={:.3} futures_ms={:.3} total_ms={:.3}",
            exprs.len(),
            num_partitions,
            partition_compile_ms,
            adaptive_ms,
            combine_ms,
            futures_ms,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    combined
}

pub(super) fn combine_synthesized_component_pairs_under_epsilon_root(
    components: &[LexerComponentPair],
    total_groups: usize,
) -> (DFA, Vec<u32>) {
    let total_states = 1usize
        + components
            .iter()
            .map(|component| component.synthesized.num_states())
            .sum::<usize>();
    let mut combined = DFA::new(total_states);
    combined.ensure_group_capacity(total_groups);
    let mut root_futures = BitSet::new(total_groups);
    let mut offsets = Vec::with_capacity(components.len());

    let mut offset = 1u32;
    for component_pair in components {
        offsets.push(offset);
        let terminal_ids = &component_pair.terminal_ids;
        let component = &component_pair.synthesized;
        debug_assert_eq!(component.num_groups(), terminal_ids.len());
        combined.add_epsilon_transition(0, offset);

        for (local_group, &terminal_id) in terminal_ids.iter().enumerate() {
            combined.set_group_u8set(
                terminal_id as u32,
                *component.group_id_to_u8set(local_group as u32),
            );
        }
        for local_group in component.possible_future_group_ids(0).iter() {
            root_futures.set(terminal_ids[local_group]);
        }

        for (state_index, state) in component.states().iter().enumerate() {
            let mapped_state = offset + state_index as u32;
            combined.set_transitions_from_sorted_entries(
                mapped_state,
                state
                    .transitions
                    .iter()
                    .map(|(byte, &target)| (byte, offset + target))
                    .collect(),
            );
            for &target in &state.epsilon_transitions {
                combined.add_epsilon_transition(mapped_state, offset + target);
            }

            let mut finalizers = BitSet::new(total_groups);
            let mut futures = BitSet::new(total_groups);
            for local_group in state.finalizers.iter() {
                finalizers.set(terminal_ids[local_group]);
            }
            for local_group in component.possible_future_group_ids(state_index as u32).iter() {
                futures.set(terminal_ids[local_group]);
            }
            combined.overwrite_state_metadata(mapped_state, finalizers, futures);
        }
        offset += component.num_states() as u32;
    }
    combined.set_possible_future_group_ids(0, root_futures);
    (combined, offsets)
}

pub(super) fn compose_partitioned_component_state_maps(
    components: &[LexerComponentPair],
    synthesized_offsets: &[u32],
) -> Option<Vec<u32>> {
    if components.len() != synthesized_offsets.len() {
        return None;
    }
    let full_state_count = 1usize.checked_add(
        components
            .iter()
            .map(|component| component.full.num_states())
            .sum::<usize>(),
    )?;
    let mut full_to_synthesized = Vec::with_capacity(full_state_count);
    full_to_synthesized.push(0);
    let mut expected_synthesized_offset = 1u32;

    for (component, &synthesized_offset) in components.iter().zip(synthesized_offsets) {
        if synthesized_offset != expected_synthesized_offset
            || component.full_to_synthesized.len() != component.full.num_states()
            || component
                .full_to_synthesized
                .iter()
                .any(|&state| state as usize >= component.synthesized.num_states())
        {
            return None;
        }
        full_to_synthesized.extend(
            component
                .full_to_synthesized
                .iter()
                .map(|&state| synthesized_offset.checked_add(state))
                .collect::<Option<Vec<_>>>()?,
        );
        expected_synthesized_offset = expected_synthesized_offset
            .checked_add(component.synthesized.num_states() as u32)?;
    }

    (full_to_synthesized.len() == full_state_count).then_some(full_to_synthesized)
}

pub(super) fn combine_lexer_components_under_epsilon_root(
    components: Vec<LexerComponent>,
    total_groups: usize,
) -> DFA {
    let mut combined = DFA::new(1);
    combined.ensure_group_capacity(total_groups);
    let mut root_futures = BitSet::new(total_groups);

    for lexer_component in components {
        let terminal_ids = lexer_component.terminal_ids;
        let component = lexer_component.dfa;
        debug_assert_eq!(component.num_groups(), terminal_ids.len());
        for local_group in component.possible_future_group_ids(0).iter() {
            root_futures.set(terminal_ids[local_group]);
        }
        let offset = combined.append_rebased_component(component, &terminal_ids);
        combined.add_epsilon_transition(0, offset);
    }
    combined.set_possible_future_group_ids(0, root_futures);
    combined
}
