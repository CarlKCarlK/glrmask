//! Compile one product component, using exact finite-language shortcuts when possible.

use crate::automata::lexer::ast::Expr;
use crate::automata::lexer::dfa::DFA;
use crate::ds::bitset::BitSet;
use crate::ds::u8set::U8Set;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;
use super::compile_with_plan;
use super::bounded_repeat::{
    RepeatBaseDfaCache,
    build_bounded_repeat_dfa_with_cache,
    build_bounded_repeat_with_regex_suffix_with_options_and_cache,
    build_bounded_repeat_with_suffix_dfa_with_cache,
    build_prefixed_bounded_repeat_with_suffix_dfa_with_options_and_cache,
};
use super::dfa_analysis::dfa_transition_count;
use super::expression_graph::expand_direct_expression_graph_classes;
use super::factor::unwrap_shared;
use super::nfa::expr_u8set;
use super::plan::build_exclusion_compile_plan;
use super::product::{build_product_class_transitions_for_dfa, explicit_dead_sink_state};
use super::repeat_suffix::compute_dfa_byte_equivalence_classes;

fn expr_is_epsilon_only(expr: &Expr) -> bool {
    match expr {
        Expr::Epsilon => true,
        Expr::U8Seq(bytes) => bytes.is_empty(),
        Expr::Seq(parts) => parts.iter().all(expr_is_epsilon_only),
        Expr::Shared(inner) => expr_is_epsilon_only(inner),
        Expr::U8Class(_)
        | Expr::Dfa(_)
        | Expr::Choice(_)
        | Expr::Exclude { .. }
        | Expr::Intersect { .. }
        | Expr::Repeat { .. } => false,
    }
}

fn optional_choice_non_epsilon(expr: &Expr) -> Option<&Expr> {
    let options = match expr {
        Expr::Shared(inner) => return optional_choice_non_epsilon(inner),
        Expr::Choice(options) if options.len() == 2 => options,
        _ => return None,
    };

    if expr_is_epsilon_only(&options[0]) {
        Some(&options[1])
    } else if expr_is_epsilon_only(&options[1]) {
        Some(&options[0])
    } else {
        None
    }
}

pub(super) fn optional_tail_parts(expr: &Expr) -> Option<Vec<Expr>> {
    let non_epsilon = optional_choice_non_epsilon(expr)?;
    match non_epsilon {
        Expr::Shared(inner) => optional_tail_parts(inner).or_else(|| Some(vec![inner.as_ref().clone()])),
        Expr::Seq(parts) => Some(parts.clone()),
        other => Some(vec![other.clone()]),
    }
}

pub(super) fn mark_state_accepting(dfa: &mut DFA, state_id: u32) {
    dfa.ensure_group_capacity(1);

    let mut finalizers = dfa.finalizers(state_id).clone();
    finalizers.set(0);
    let mut future = dfa.possible_future_group_ids(state_id).clone();
    future.set(0);
    dfa.overwrite_state_metadata(state_id, finalizers, future);
}

pub(super) fn collect_fixed_sequence_byte_sets(expr: &Expr, byte_sets: &mut Vec<U8Set>) -> bool {
    match expr {
        Expr::U8Seq(bytes) => {
            byte_sets.extend(bytes.iter().copied().map(U8Set::from_byte));
            true
        }
        Expr::U8Class(bytes) => {
            byte_sets.push(*bytes);
            true
        }
        Expr::Seq(parts) => parts
            .iter()
            .all(|part| collect_fixed_sequence_byte_sets(part, byte_sets)),
        Expr::Shared(inner) => collect_fixed_sequence_byte_sets(inner, byte_sets),
        Expr::Epsilon => true,
        Expr::Dfa(_)
        | Expr::Choice(_)
        | Expr::Exclude { .. }
        | Expr::Intersect { .. }
        | Expr::Repeat { .. } => false,
    }
}

fn build_fixed_sequence_dfa(expr: &Expr) -> Option<DFA> {
    let mut byte_sets = Vec::new();
    collect_fixed_sequence_byte_sets(expr, &mut byte_sets).then(|| {
        let mut dfa = DFA::new(byte_sets.len() + 1);
        dfa.ensure_group_capacity(1);
        dfa.set_group_u8set(0, expr_u8set(expr));
        for (state, bytes) in byte_sets.into_iter().enumerate() {
            dfa.set_transitions_from_sorted_entries(
                state as u32,
                bytes
                    .iter()
                    .map(|byte| (byte, state as u32 + 1))
                    .collect(),
            );
        }
        let final_state = (dfa.num_states() - 1) as u32;
        mark_state_accepting(&mut dfa, final_state);
        dfa.recompute_possible_futures();
        dfa
    })
}

const MAX_FINITE_LITERAL_ALTERNATIVES: usize = 4096;

const MAX_FINITE_LITERAL_BYTES: usize = 1 << 20;

fn collect_finite_literal_language(expr: &Expr) -> Option<Vec<Vec<u8>>> {
    match unwrap_shared(expr) {
        Expr::U8Seq(bytes) => Some(vec![bytes.clone()]),
        Expr::Epsilon => Some(vec![Vec::new()]),
        Expr::Choice(options) => {
            let mut out = Vec::new();
            let mut total_bytes = 0usize;
            for option in options {
                let alternatives = collect_finite_literal_language(option)?;
                if out.len().saturating_add(alternatives.len()) > MAX_FINITE_LITERAL_ALTERNATIVES {
                    return None;
                }
                for literal in alternatives {
                    total_bytes = total_bytes.saturating_add(literal.len());
                    if total_bytes > MAX_FINITE_LITERAL_BYTES {
                        return None;
                    }
                    out.push(literal);
                }
            }
            Some(out)
        }
        Expr::Seq(parts) => {
            let mut prefixes = vec![Vec::<u8>::new()];
            for part in parts {
                let suffixes = collect_finite_literal_language(part)?;
                let next_count = prefixes.len().checked_mul(suffixes.len())?;
                if next_count > MAX_FINITE_LITERAL_ALTERNATIVES {
                    return None;
                }
                let mut next = Vec::with_capacity(next_count);
                let mut total_bytes = 0usize;
                for prefix in &prefixes {
                    for suffix in &suffixes {
                        let len = prefix.len().checked_add(suffix.len())?;
                        total_bytes = total_bytes.checked_add(len)?;
                        if total_bytes > MAX_FINITE_LITERAL_BYTES {
                            return None;
                        }
                        let mut literal = Vec::with_capacity(len);
                        literal.extend_from_slice(prefix);
                        literal.extend_from_slice(suffix);
                        next.push(literal);
                    }
                }
                prefixes = next;
            }
            Some(prefixes)
        }
        _ => None,
    }
}

pub(super) fn build_finite_literal_language_dfa(expr: &Expr) -> Option<DFA> {
    if std::env::var_os("GLRMASK_DISABLE_FINITE_LITERAL_TRIE_DIRECT").is_some() {
        return None;
    }
    let profile_timing = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    let started_at = profile_timing.then(Instant::now);
    let literals = collect_finite_literal_language(expr)?;
    if literals.len() <= 1 {
        return None;
    }

    #[derive(Default)]
    struct Node {
        children: BTreeMap<u8, usize>,
        accepting: bool,
    }

    let mut trie = vec![Node::default()];
    for literal in &literals {
        let mut node = 0usize;
        for &byte in literal {
            let next = if let Some(&next) = trie[node].children.get(&byte) {
                next
            } else {
                let next = trie.len();
                trie.push(Node::default());
                trie[node].children.insert(byte, next);
                next
            };
            node = next;
        }
        trie[node].accepting = true;
    }

    let trie_nodes = trie.len();
    let mut dfa = DFA::new(trie_nodes);
    dfa.ensure_group_capacity(1);
    dfa.set_group_u8set(0, expr_u8set(expr));
    for (node, trie_node) in trie.iter().enumerate() {
        dfa.set_transitions_from_sorted_entries(
            node as u32,
            trie_node
                .children
                .iter()
                .map(|(&byte, &target)| (byte, target as u32))
                .collect(),
        );
        if trie_node.accepting {
            mark_state_accepting(&mut dfa, node as u32);
        }
    }
    dfa.recompute_possible_futures();
    let minimized = dfa.minimize_owned_reachable();
    if profile_timing {
        eprintln!(
            "[glrmask/profile][tokenizer] finite_literal_trie_direct alternatives={} trie_nodes={} final_states={} final_transitions={} total_ms={:.3}",
            literals.len(),
            trie_nodes,
            minimized.num_states(),
            dfa_transition_count(&minimized),
            started_at.unwrap().elapsed().as_secs_f64() * 1000.0,
        );
    }
    Some(minimized)
}

/// Exact union of already-deterministic single-group DFAs and fixed byte
/// strings. General Choice compilation lowers every DFA arm back into an NFA
/// and redeterminizes the union. Here the determinized state is simply the
/// sparse tuple of still-live DFA states plus one literal-trie DFA state.
/// The product stays on the global byte-class alphabet through minimization and
/// expands classes back to bytes only for the final compact automaton.
pub(super) fn build_dfas_with_literal_choice(expr: &Expr) -> Option<DFA> {
    let profile_timing = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    let total_started_at = profile_timing.then(Instant::now);
    if std::env::var_os("GLRMASK_DISABLE_DFA_LITERAL_CHOICE_DIRECT").is_some() {
        return None;
    }
    let Expr::Choice(options) = unwrap_shared(expr) else {
        return None;
    };
    let mut dfas = Vec::<Arc<DFA>>::new();
    let mut literals = Vec::<&[u8]>::new();
    for option in options {
        match unwrap_shared(option) {
            Expr::Dfa(dfa) => dfas.push(Arc::clone(dfa)),
            Expr::U8Seq(bytes) => literals.push(bytes.as_slice()),
            Expr::Epsilon => literals.push(&[]),
            _ => return None,
        }
    }
    if dfas.is_empty() || literals.is_empty() {
        return None;
    }
    if dfas
        .iter()
        .any(|dfa| dfa.num_groups() != 1 || dfa.has_epsilon_transitions())
    {
        return None;
    }

    #[derive(Default)]
    struct LiteralTrieNode {
        children: BTreeMap<u8, usize>,
        accepting: bool,
    }
    let mut trie = vec![LiteralTrieNode::default()];
    for literal in literals {
        let mut node = 0usize;
        for &byte in literal {
            let next = if let Some(&next) = trie[node].children.get(&byte) {
                next
            } else {
                let next = trie.len();
                trie.push(LiteralTrieNode::default());
                trie[node].children.insert(byte, next);
                next
            };
            node = next;
        }
        trie[node].accepting = true;
    }
    let trie_nodes = trie.len();

    let mut literal_dfa = DFA::new(trie.len());
    literal_dfa.ensure_group_capacity(1);
    literal_dfa.set_group_u8set(0, expr_u8set(expr));
    for (node, trie_node) in trie.iter().enumerate() {
        literal_dfa.set_transitions_from_sorted_entries(
            node as u32,
            trie_node
                .children
                .iter()
                .map(|(&byte, &target)| (byte, target as u32))
                .collect(),
        );
        let mut finalizers = BitSet::new(1);
        if trie_node.accepting {
            finalizers.set(0);
        }
        literal_dfa.overwrite_state_metadata(node as u32, finalizers, BitSet::new(1));
    }

    let mut components = dfas.iter().map(Arc::as_ref).collect::<Vec<_>>();
    components.push(&literal_dfa);
    let class_started_at = profile_timing.then(Instant::now);
    let (class_map, class_members) = compute_dfa_byte_equivalence_classes(&components);
    let class_ms = class_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    if class_members.len() > u8::MAX as usize + 1 {
        return None;
    }
    let class_count = class_members.len();
    let component_class_transitions = components
        .iter()
        .map(|dfa| {
            build_product_class_transitions_for_dfa(dfa, &class_map)
                .into_iter()
                .map(|row| {
                    let mut dense = vec![u32::MAX; class_count];
                    for (class, target) in row {
                        dense[class as usize] = target;
                    }
                    dense
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let dead_states = components
        .iter()
        .map(|dfa| explicit_dead_sink_state(dfa))
        .collect::<Vec<_>>();
    let start = std::iter::repeat_n(0u32, components.len())
        .collect::<SmallVec<[u32; 4]>>();
    let mut result = DFA::new(1);
    result.ensure_group_capacity(1);
    result.set_group_u8set(0, expr_u8set(expr));
    let mut state_by_tuple = FxHashMap::<SmallVec<[u32; 4]>, u32>::default();
    state_by_tuple.insert(start.clone(), 0);
    let mut queue = VecDeque::from([(0u32, start)]);
    const MAX_DIRECT_CHOICE_STATES: usize = 32_768;
    const MAX_DIRECT_CHOICE_CLASS_TRANSITIONS: usize = 2_000_000;
    let mut transition_count = 0usize;
    let construct_started_at = profile_timing.then(Instant::now);

    while let Some((result_state, tuple)) = queue.pop_front() {
        let accepting = components.iter().enumerate().any(|(index, dfa)| {
            let state = tuple[index];
            state != u32::MAX && dfa.finalizers(state).contains(0)
        });
        let mut finalizers = BitSet::new(1);
        if accepting {
            finalizers.set(0);
        }
        result.overwrite_state_metadata(result_state, finalizers, BitSet::new(1));

        let mut transitions = Vec::<(u8, u32)>::new();
        for class in 0..class_count {
            let mut next = SmallVec::<[u32; 4]>::with_capacity(components.len());
            let mut any_live = false;
            for (index, dfa) in components.iter().enumerate() {
                let state = tuple[index];
                if state == u32::MAX {
                    next.push(u32::MAX);
                    continue;
                }
                let target = component_class_transitions[index][state as usize][class];
                if target != u32::MAX && dead_states[index] != Some(target) {
                    next.push(target);
                    any_live = true;
                } else {
                    next.push(u32::MAX);
                }
            }
            if !any_live {
                continue;
            }
            let target = if let Some(&target) = state_by_tuple.get(&next) {
                target
            } else {
                if result.num_states() >= MAX_DIRECT_CHOICE_STATES {
                    return None;
                }
                let target = result.add_state();
                state_by_tuple.insert(next.clone(), target);
                queue.push_back((target, next));
                target
            };
            transitions.push((class as u8, target));
        }
        transition_count = transition_count.saturating_add(transitions.len());
        if transition_count > MAX_DIRECT_CHOICE_CLASS_TRANSITIONS {
            return None;
        }
        result.set_transitions_from_sorted_entries(result_state, transitions);
    }
    let construct_ms = construct_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

    let raw_states = result.num_states();
    let raw_class_transitions = dfa_transition_count(&result);
    let futures_started_at = profile_timing.then(Instant::now);
    result.recompute_possible_futures();
    let futures_ms = futures_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let minimize_started_at = profile_timing.then(Instant::now);
    let mut minimized = result.minimize_owned_reachable();
    let minimize_ms = minimize_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let final_class_transitions = dfa_transition_count(&minimized);
    let expand_started_at = profile_timing.then(Instant::now);
    expand_direct_expression_graph_classes(&mut minimized, &class_members);
    let expand_ms = expand_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    if profile_timing {
        eprintln!(
            "[glrmask/profile][tokenizer] dfa_literal_choice_direct dfa_arms={} trie_nodes={} classes={} raw_states={} raw_class_transitions={} final_states={} final_class_transitions={} final_byte_transitions={} class_ms={:.3} construct_ms={:.3} futures_ms={:.3} minimize_ms={:.3} expand_ms={:.3} total_ms={:.3}",
            dfas.len(),
            trie_nodes,
            class_members.len(),
            raw_states,
            raw_class_transitions,
            minimized.num_states(),
            final_class_transitions,
            dfa_transition_count(&minimized),
            class_ms,
            construct_ms,
            futures_ms,
            minimize_ms,
            expand_ms,
            total_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
        );
    }
    Some(minimized)
}

fn compile_product_component_dfa_direct_with_options_and_cache(
    expr: &Expr,
    preserve_coordinates: bool,
    cache: Option<&RepeatBaseDfaCache>,
) -> Option<(DFA, bool)> {
    match expr {
        Expr::Shared(inner) => {
            compile_product_component_dfa_direct_with_options_and_cache(
                inner,
                preserve_coordinates,
                cache,
            )
        }
        Expr::U8Seq(bytes) => {
            let mut dfa = DFA::new(bytes.len() + 1);
            dfa.ensure_group_capacity(1);
            dfa.set_group_u8set(0, U8Set::from_bytes(bytes));
            for (index, &byte) in bytes.iter().enumerate() {
                dfa.add_transition(index as u32, byte, index as u32 + 1);
            }
            mark_state_accepting(&mut dfa, bytes.len() as u32);
            dfa.recompute_possible_futures();
            Some((dfa, false))
        }
        Expr::U8Class(bytes) => {
            let mut dfa = DFA::new(2);
            dfa.ensure_group_capacity(1);
            dfa.set_group_u8set(0, *bytes);
            dfa.set_transitions_from_sorted_entries(
                0,
                bytes.iter().map(|byte| (byte, 1)).collect(),
            );
            mark_state_accepting(&mut dfa, 1);
            dfa.recompute_possible_futures();
            Some((dfa, false))
        }
        Expr::Epsilon => {
            let mut dfa = DFA::new(1);
            dfa.ensure_group_capacity(1);
            dfa.set_group_u8set(0, U8Set::empty());
            mark_state_accepting(&mut dfa, 0);
            dfa.recompute_possible_futures();
            Some((dfa, false))
        }
        Expr::Dfa(dfa) => Some((dfa.as_ref().clone(), true)),
        Expr::Choice(_) => {
            if let Some(dfa) = build_dfas_with_literal_choice(expr) {
                return Some((dfa, false));
            }
            let non_epsilon = optional_choice_non_epsilon(expr)?;
            let (mut dfa, needs_future_recompute) =
                compile_product_component_dfa_direct_with_options_and_cache(
                    non_epsilon,
                    preserve_coordinates,
                    cache,
                )?;
            mark_state_accepting(&mut dfa, 0);
            Some((dfa, needs_future_recompute))
        }
        Expr::Repeat {
            expr,
            min,
            max: Some(max),
        } => build_bounded_repeat_dfa_with_cache(expr, *min, *max, cache)
            .map(|dfa| (dfa, false)),
        Expr::Seq(parts) => build_fixed_sequence_dfa(expr)
            .map(|dfa| (dfa, false))
            .or_else(|| build_finite_literal_language_dfa(expr).map(|dfa| (dfa, false)))
            .or_else(|| build_bounded_repeat_with_suffix_dfa_with_cache(parts, cache))
            .or_else(|| {
                build_bounded_repeat_with_regex_suffix_with_options_and_cache(
                    parts,
                    preserve_coordinates,
                    cache,
                )
            })
            .or_else(|| {
                build_prefixed_bounded_repeat_with_suffix_dfa_with_options_and_cache(
                    parts,
                    preserve_coordinates,
                    cache,
                )
            }),
        _ => None,
    }
}

fn compile_product_component_dfa_direct_with_options(
    expr: &Expr,
    preserve_coordinates: bool,
) -> Option<(DFA, bool)> {
    compile_product_component_dfa_direct_with_options_and_cache(
        expr,
        preserve_coordinates,
        None,
    )
}

pub(super) fn compile_product_component_dfa_direct(expr: &Expr) -> Option<(DFA, bool)> {
    compile_product_component_dfa_direct_with_options(expr, false)
}

pub(super) fn compile_product_component_dfa(expr: &Expr) -> DFA {
    compile_with_plan(build_exclusion_compile_plan(std::slice::from_ref(expr)))
}

/// Compile one logical terminal definition through the full expression plan.
///
/// Unlike `compile_single_expr_dfa`, this lowers nested exclusions and
/// intersections before NFA/DFA construction. Definition-level analyses must
/// use this entry point because terminal expressions are not guaranteed to be
/// primitive product leaves.
pub fn compile_terminal_expr_dfa(expr: &Expr) -> DFA {
    compile_product_component_dfa(expr)
}

pub(super) fn compile_product_component_materialized_dfa_with_options_and_cache(
    expr: &Expr,
    preserve_coordinates: bool,
    cache: Option<&RepeatBaseDfaCache>,
) -> DFA {
    let profile_timing = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    let direct_started_at = profile_timing.then(Instant::now);
    let dfa = if let Some((mut dfa, needs_future_recompute)) =
        compile_product_component_dfa_direct_with_options_and_cache(
            expr,
            preserve_coordinates,
            cache,
        )
    {
        let direct_ms = direct_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        dfa.ensure_group_capacity(1);
        dfa.set_group_u8set(0, expr_u8set(expr));
        if needs_future_recompute {
            dfa.recompute_possible_futures();
        }
        if profile_timing {
            eprintln!(
                "[glrmask/profile][tokenizer] product_component_path path=direct states={} transitions={} direct_ms={:.3}",
                dfa.num_states(),
                dfa_transition_count(&dfa),
                direct_ms,
            );
        }
        dfa
    } else {
        let direct_ms = direct_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        // The generic NFA->DFA path may still carry conservative future-group
        // metadata until the ordinary single-expression compile/minimize pass
        // recomputes it. Product construction consumes that metadata directly,
        // so retain the exact old fallback rather than using the raw DFA here.
        let fallback_started_at = profile_timing.then(Instant::now);
        let dfa = compile_product_component_dfa(expr);
        if profile_timing {
            eprintln!(
                "[glrmask/profile][tokenizer] product_component_path path=fallback states={} transitions={} direct_attempt_ms={:.3} fallback_ms={:.3}",
                dfa.num_states(),
                dfa_transition_count(&dfa),
                direct_ms,
                fallback_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
            );
        }
        dfa
    };
    dfa
}

pub(super) fn compile_product_component_materialized_dfa_with_options(
    expr: &Expr,
    preserve_coordinates: bool,
) -> DFA {
    compile_product_component_materialized_dfa_with_options_and_cache(
        expr,
        preserve_coordinates,
        None,
    )
}

pub(super) fn compile_product_component_materialized_dfa(expr: &Expr) -> DFA {
    compile_product_component_materialized_dfa_with_options(expr, false)
}
