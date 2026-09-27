//! Construct exact bounded-repeat DFAs, including literal and regular suffixes.

use crate::automata::lexer::ast::Expr;
use crate::automata::lexer::dfa::DFA;
use crate::ds::bitset::BitSet;
use rustc_hash::FxHashMap;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;
use super::dfa_analysis::{
    dfa_is_nonnullable_and_prefix_free,
    dfa_transition_count,
    productive_dfa_states,
};
use super::factor::unwrap_shared;
use super::component::{mark_state_accepting, optional_tail_parts};
use super::nfa::{DIRECT_BOUNDED_REPEAT_THRESHOLD, compile_expr_to_dfa};
use super::repeat_suffix::build_zero_min_repeat_suffix_dominance_dfa_internal;

pub(super) type RepeatBaseDfaCache = FxHashMap<Expr, Arc<DFA>>;

pub(super) fn cached_direct_bounded_repeat_base_dfa_unconditionally(
    expr: &Expr,
    cache: Option<&RepeatBaseDfaCache>,
) -> Option<Arc<DFA>> {
    let expr = unwrap_shared(expr);
    if let Some(cached) = cache.and_then(|cache| cache.get(expr)) {
        return Some(Arc::clone(cached));
    }
    compile_direct_bounded_repeat_base_dfa_unconditionally(expr).map(Arc::new)
}

pub(super) fn cached_direct_bounded_repeat_base_dfa(
    expr: &Expr,
    max: usize,
    cache: Option<&RepeatBaseDfaCache>,
) -> Option<Arc<DFA>> {
    if max < DIRECT_BOUNDED_REPEAT_THRESHOLD {
        return None;
    }
    cached_direct_bounded_repeat_base_dfa_unconditionally(expr, cache)
}

pub(super) fn compile_direct_bounded_repeat_base_dfa_unconditionally(expr: &Expr) -> Option<DFA> {
    let base_dfa = compile_expr_to_dfa(expr);
    if base_dfa.num_states() == 0 || !dfa_is_nonnullable_and_prefix_free(&base_dfa) {
        return None;
    }

    Some(base_dfa)
}



pub(super) fn build_bounded_repeat_dfa_with_cache(
    expr: &Expr,
    min: usize,
    max: usize,
    cache: Option<&RepeatBaseDfaCache>,
) -> Option<DFA> {
    let base_dfa = cached_direct_bounded_repeat_base_dfa(expr, max, cache)?;
    build_bounded_repeat_dfa_from_base(base_dfa.as_ref(), min, max)
}

pub(super) fn build_bounded_repeat_dfa(expr: &Expr, min: usize, max: usize) -> Option<DFA> {
    build_bounded_repeat_dfa_with_cache(expr, min, max, None)
}

pub(super) fn build_bounded_repeat_dfa_from_base(base_dfa: &DFA, min: usize, max: usize) -> Option<DFA> {
    if min > max {
        return None;
    }

    let base_states = base_dfa.states();
    let base_state_count = base_states.len();
    let layers = max.checked_add(1)?;
    let total_states = layers.checked_mul(base_state_count)?;
    u32::try_from(total_states).ok()?;
    let productive = productive_dfa_states(base_dfa);
    let mut dfa = DFA::new(total_states);
    dfa.ensure_group_capacity(1);

    for copies_done in 0..=max {
        for (state_id, state) in base_states.iter().enumerate() {
            let mapped_state = (copies_done * base_state_count + state_id) as u32;
            let mut finalizers = crate::ds::bitset::BitSet::new(1);
            let mut future = crate::ds::bitset::BitSet::new(1);
            if state_id == 0 && copies_done >= min {
                finalizers.set(0);
            }
            if copies_done < max && productive[state_id] {
                future.set(0);
            }
            dfa.overwrite_state_metadata(mapped_state, finalizers, future);

            if copies_done == max || !base_dfa.finalizers(state_id as u32).is_empty() {
                continue;
            }

            let mut transitions = Vec::with_capacity(state.transitions.len());
            for (byte, &target) in state.transitions.iter() {
                let mapped_target = if !base_dfa.finalizers(target).is_empty() {
                    ((copies_done + 1) * base_state_count) as u32
                } else {
                    (copies_done * base_state_count + target as usize) as u32
                };
                transitions.push((byte, mapped_target));
            }
            dfa.set_transitions_from_sorted_entries(mapped_state, transitions);
        }
    }

    Some(dfa)
}

/// Collects all bytes from a slice of suffix expressions that are all U8Seq.
/// Returns None if any expression is not a simple byte sequence.
pub(super) fn collect_suffix_bytes(exprs: &[Expr]) -> Option<Vec<u8>> {
    let mut bytes = Vec::new();
    for expr in exprs {
        match expr {
            Expr::U8Seq(b) => bytes.extend_from_slice(b),
            Expr::Shared(inner) => match inner.as_ref() {
                Expr::U8Seq(b) => bytes.extend_from_slice(b),
                _ => return None,
            },
            _ => return None,
        }
    }
    if bytes.is_empty() { None } else { Some(bytes) }
}

/// Builds a DFA for `Seq([Repeat{expr, min, max}, suffix_bytes...])` directly,
/// avoiding NFA→DFA determinization. Works when the first suffix byte does not
/// overlap with the repeat expression's start-state transitions (e.g., closing
/// quote `"` after JSON string chars that exclude `"`).
pub(super) fn build_bounded_repeat_with_suffix_dfa_with_cache(
    parts: &[Expr],
    cache: Option<&RepeatBaseDfaCache>,
) -> Option<(DFA, bool)> {
    if parts.len() < 2 {
        return None;
    }

    // Extract repeat parameters, unwrapping Shared if needed.
    let first = match &parts[0] {
        Expr::Shared(inner) => inner.as_ref(),
        other => other,
    };
    let (repeat_expr, min, max) = match first {
        Expr::Repeat {
            expr,
            min,
            max: Some(max),
        } => (expr.as_ref(), *min, *max),
        _ => return None,
    };

    let suffix_bytes = collect_suffix_bytes(&parts[1..])?;
    let base_dfa = cached_direct_bounded_repeat_base_dfa_unconditionally(repeat_expr, cache)?;

    let base_states = base_dfa.states();
    let base_state_count = base_states.len();
    let layers = max.checked_add(1)?;
    let repeat_state_count = layers.checked_mul(base_state_count)?;
    let suffix_len = suffix_bytes.len();
    let total_states = repeat_state_count.checked_add(suffix_len)?;
    u32::try_from(total_states).ok()?;
    let productive = productive_dfa_states(&base_dfa);

    // The suffix boundary is ambiguous only when its first byte can begin a
    // body word. A syntactic start transition into an unproductive residual is
    // irrelevant to the body language and may be discarded below.
    if base_states[0]
        .transitions
        .get(suffix_bytes[0])
        .is_some_and(|&target| productive[target as usize])
    {
        return None;
    }

    let mut dfa = DFA::new(total_states);
    dfa.ensure_group_capacity(1);
    let first_suffix_state = repeat_state_count as u32;

    for copies_done in 0..=max {
        for (state_id, state) in base_states.iter().enumerate() {
            let mapped_state = (copies_done * base_state_count + state_id) as u32;
            // No finalizers on repeat states — only the suffix chain end finalizes.
            let finalizers = crate::ds::bitset::BitSet::new(1);
            let mut future = crate::ds::bitset::BitSet::new(1);

            let is_accepting_pos = state_id == 0 && copies_done >= min;
            if (copies_done < max && productive[state_id]) || is_accepting_pos {
                future.set(0);
            }
            dfa.overwrite_state_metadata(mapped_state, finalizers, future);

            // At max copies or at a base-DFA finalizer state: no repeat transitions,
            // but accepting positions still get the suffix entry transition.
            if copies_done == max || !base_dfa.finalizers(state_id as u32).is_empty() {
                if is_accepting_pos {
                    dfa.set_transitions_from_sorted_entries(
                        mapped_state,
                        vec![(suffix_bytes[0], first_suffix_state)],
                    );
                }
                continue;
            }

            // Build transitions: repeat transitions + optional suffix entry.
            let extra = if is_accepting_pos { 1 } else { 0 };
            let mut transitions = Vec::with_capacity(state.transitions.len() + extra);
            for (byte, &target) in state.transitions.iter() {
                if !productive[target as usize] {
                    continue;
                }
                let mapped_target = if !base_dfa.finalizers(target).is_empty() {
                    ((copies_done + 1) * base_state_count) as u32
                } else {
                    (copies_done * base_state_count + target as usize) as u32
                };
                transitions.push((byte, mapped_target));
            }
            if is_accepting_pos {
                let pos = transitions.partition_point(|&(b, _)| b < suffix_bytes[0]);
                transitions.insert(pos, (suffix_bytes[0], first_suffix_state));
            }
            dfa.set_transitions_from_sorted_entries(mapped_state, transitions);
        }
    }

    // Build suffix chain: each state transitions on the NEXT suffix byte.
    for i in 0..suffix_len {
        let suffix_state = (repeat_state_count + i) as u32;
        if i + 1 < suffix_len {
            let next_suffix = (repeat_state_count + i + 1) as u32;
            let mut future = crate::ds::bitset::BitSet::new(1);
            future.set(0);
            dfa.overwrite_state_metadata(
                suffix_state,
                crate::ds::bitset::BitSet::new(1),
                future,
            );
            dfa.set_transitions_from_sorted_entries(
                suffix_state,
                vec![(suffix_bytes[i + 1], next_suffix)],
            );
        } else {
            // Last suffix state: finalizer, no transitions, no future.
            let mut finalizers = crate::ds::bitset::BitSet::new(1);
            finalizers.set(0);
            dfa.overwrite_state_metadata(
                suffix_state,
                finalizers,
                crate::ds::bitset::BitSet::new(1),
            );
        }
    }

    Some((dfa, false))
}

pub(super) fn build_bounded_repeat_with_suffix_dfa(parts: &[Expr]) -> Option<(DFA, bool)> {
    build_bounded_repeat_with_suffix_dfa_with_cache(parts, None)
}



pub(super) fn build_bounded_repeat_with_regex_suffix_with_options_and_cache(
    parts: &[Expr],
    preserve_coordinates: bool,
    cache: Option<&RepeatBaseDfaCache>,
) -> Option<(DFA, bool)> {
    let profile_timing = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    let total_started_at = profile_timing.then(Instant::now);
    if parts.len() < 2 {
        return None;
    }

    // Flatten one level of nested Seq: Seq([Seq([a, b]), c]) → [a, b, c]
    let flat_parts: Vec<&Expr>;
    let parts_ref: &[&Expr] = {
        let first_unwrapped = match &parts[0] {
            Expr::Shared(inner) => inner.as_ref(),
            other => other,
        };
        if let Expr::Seq(inner_parts) = first_unwrapped {
            flat_parts = inner_parts.iter().chain(parts[1..].iter()).collect();
            &flat_parts
        } else {
            flat_parts = parts.iter().collect();
            &flat_parts
        }
    };

    if parts_ref.len() < 2 {
        return None;
    }

    let first = match parts_ref[0] {
        Expr::Shared(inner) => inner.as_ref(),
        other => other,
    };
    let (repeat_expr, min, max) = match first {
        Expr::Repeat {
            expr,
            min,
            max: Some(max),
        } => (expr.as_ref(), *min, *max),
        _ => return None,
    };

    // With max == 0 the body is not allowed to consume anything. The compact
    // product construction starts with a live body state, so it would otherwise
    // permit one body occurrence before the suffix. Let the general path handle
    // this exact zero-repeat case.
    if max == 0 {
        return None;
    }

    let body_started_at = profile_timing.then(Instant::now);
    let body_dfa = cache
        .and_then(|cache| cache.get(unwrap_shared(repeat_expr)))
        .map(|dfa| dfa.as_ref().clone())
        .unwrap_or_else(|| compile_expr_to_dfa(repeat_expr));
    let body_ms = body_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    if body_dfa.num_states() == 0 || !body_dfa.finalizers(0).is_empty() {
        return None;
    }

    let suffix_expr = if parts_ref.len() == 2 {
        parts_ref[1].clone()
    } else {
        Expr::Seq(parts_ref[1..].iter().map(|e| (*e).clone()).collect())
    };
    let suffix_started_at = profile_timing.then(Instant::now);
    let suffix_dfa = compile_expr_to_dfa(&suffix_expr);
    let suffix_ms = suffix_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    if suffix_dfa.num_states() == 0 {
        return None;
    }

    // Nullable suffixes require considering a body/suffix boundary without
    // consuming another byte. This compact construction only starts fresh
    // suffix paths while processing a byte, so it can miss finalization at
    // end-of-input. Fall back to the general construction.
    if !suffix_dfa.finalizers(0).is_empty() {
        return None;
    }

    // For a zero-minimum repeat, paths that reach the same body residual with
    // different completed-copy counts are ordered by language inclusion. The
    // smallest count has at least as much remaining repeat budget and may take
    // every suffix boundary available to a larger count, so larger counts are
    // redundant. Track only that minimum count per body state, plus the exact
    // set of live suffix states. This is an exact quotient of the ordinary NFA
    // subset construction and handles ambiguous body/suffix boundaries without
    // materializing the enormous unrolled-repeat powerset.
    // The dense minimum-count vector makes this quotient exceptionally fast
    // for the small repeat bodies that cause large counter products, but its
    // per-state work is quadratic in a very large body DFA. Large-body,
    // low-repeat wrappers are better served by the existing compact product
    // below (or the generic fallback when genuinely ambiguous).
    const DOMINANCE_MAX_BODY_STATES: usize = 256;
    if min == 0
        && body_dfa.num_states() <= DOMINANCE_MAX_BODY_STATES
        && let Some(built) = build_zero_min_repeat_suffix_dominance_dfa_internal(
            &body_dfa,
            &suffix_dfa,
            max,
            preserve_coordinates,
        )
    {
        let dfa = built.dfa;
        if let Some(total_started_at) = total_started_at {
            eprintln!(
                "[glrmask/profile][tokenizer] bounded_repeat_regex_suffix_dominance body_states={} suffix_states={} max={} final_states={} final_transitions={} body_ms={:.3} suffix_ms={:.3} total_ms={:.3}",
                body_dfa.num_states(),
                suffix_dfa.num_states(),
                max,
                dfa.num_states(),
                dfa_transition_count(&dfa),
                body_ms,
                suffix_ms,
                total_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        return Some((dfa, false));
    }

    let max_product =
        (body_dfa.num_states() + 1) * (suffix_dfa.num_states() + 1) * (max + 1);
    if max_product > 500_000 {
        return None;
    }

    let body_dead = body_dfa.num_states() as u32;
    let suffix_dead = suffix_dfa.num_states() as u32;

    let mut state_map: FxHashMap<(u32, u32, u32), u32> = FxHashMap::default();
    let mut worklist: VecDeque<(u32, (u32, u32, u32))> = VecDeque::new();
    let mut dfa = DFA::new(1);
    dfa.ensure_group_capacity(1);

    let start_suffix = if min == 0 { 0u32 } else { suffix_dead };
    let start_key = (0u32, start_suffix, 0u32);
    state_map.insert(start_key, 0);
    worklist.push_back((0, start_key));

    {
        let is_accept = start_suffix < suffix_dead
            && !suffix_dfa.finalizers(start_suffix).is_empty();
        let mut finalizers = BitSet::new(1);
        let mut future = BitSet::new(1);
        if is_accept {
            finalizers.set(0);
        }
        future.set(0);
        dfa.overwrite_state_metadata(0, finalizers, future);
    }

    let construct_started_at = profile_timing.then(Instant::now);
    while let Some((dfa_state, (b, s, c))) = worklist.pop_front() {
        let body_is_accept = b < body_dead && !body_dfa.finalizers(b).is_empty();

        let mut transitions = Vec::new();
        for byte_val in 0u16..=255 {
            let x = byte_val as u8;

            let b_next = if b < body_dead {
                body_dfa.step(b, x).map_or(body_dead, |t| t)
            } else {
                body_dead
            };

            // If the body is accepting before this byte, there is an implicit
            // boundary here: either continue the repeated body with `x`, or
            // finish the body and let the suffix consume `x`.
            //
            // This product state tracks only one body state and one suffix
            // state. If both paths are live, keeping only the body path is a
            // lossy greedy approximation.
            //
            // Minimal counterexample:
            //
            //   ("a"+)? "a"
            //
            // on input "aa". At the second "a", the body can continue, but the
            // suffix can also start. Dropping the suffix path falsely rejects.
            if body_is_accept && b_next != body_dead {
                let new_c = c + 1;
                if new_c >= min as u32 && new_c <= max as u32 {
                    let fresh_s = suffix_dfa
                        .step(0, x)
                        .map_or(suffix_dead, |t| t);
                    if fresh_s != suffix_dead {
                        return None;
                    }
                }
            }

            let (final_b, final_s, final_c) =
                if body_is_accept && b_next == body_dead {
                    let new_c = c + 1;
                    let new_b = if new_c < max as u32 {
                        body_dfa.step(0, x).map_or(body_dead, |t| t)
                    } else {
                        body_dead
                    };
                    let old_s_next = if s < suffix_dead {
                        suffix_dfa.step(s, x).map_or(suffix_dead, |t| t)
                    } else {
                        suffix_dead
                    };
                    let fresh_s = if new_c >= min as u32 {
                        suffix_dfa.step(0, x).map_or(suffix_dead, |t| t)
                    } else {
                        suffix_dead
                    };
                    let new_s = match (old_s_next < suffix_dead, fresh_s < suffix_dead) {
                        (true, true) if old_s_next != fresh_s => return None,
                        (true, _) => old_s_next,
                        (_, true) => fresh_s,
                        _ => suffix_dead,
                    };
                    (new_b, new_s, new_c)
                } else {
                    let s_next = if s < suffix_dead {
                        suffix_dfa.step(s, x).map_or(suffix_dead, |t| t)
                    } else {
                        suffix_dead
                    };
                    (b_next, s_next, c)
                };

            if final_b == body_dead && final_s == suffix_dead {
                continue;
            }

            let target_key = (final_b, final_s, final_c);
            let target_dfa_state =
                if let Some(&existing) = state_map.get(&target_key) {
                    existing
                } else {
                    let new_state = dfa.add_state();
                    let accept = final_s < suffix_dead
                        && !suffix_dfa.finalizers(final_s).is_empty()
                        && final_c >= min as u32;
                    let has_future = final_b < body_dead || final_s < suffix_dead;
                    let mut finalizers = BitSet::new(1);
                    let mut future = BitSet::new(1);
                    if accept {
                        finalizers.set(0);
                    }
                    if has_future {
                        future.set(0);
                    }
                    dfa.overwrite_state_metadata(new_state, finalizers, future);
                    state_map.insert(target_key, new_state);
                    worklist.push_back((new_state, target_key));
                    new_state
                };

            transitions.push((x, target_dfa_state));
        }

        if transitions.len() > 1 {
            transitions.sort_unstable_by_key(|e| e.0);
        }
        dfa.set_transitions_from_sorted_entries(dfa_state, transitions);
    }

    let construct_ms = construct_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let pre_minimize_states = dfa.num_states();
    let pre_minimize_transitions = dfa_transition_count(&dfa);
    let minimize_started_at = profile_timing.then(Instant::now);
    let dfa = dfa.minimize();
    let minimize_ms = minimize_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    if let Some(total_started_at) = total_started_at {
        eprintln!(
            "[glrmask/profile][tokenizer] bounded_repeat_regex_suffix body_states={} suffix_states={} max={} pre_minimize_states={} pre_minimize_transitions={} final_states={} final_transitions={} body_ms={:.3} suffix_ms={:.3} construct_ms={:.3} minimize_ms={:.3} total_ms={:.3}",
            body_dfa.num_states(),
            suffix_dfa.num_states(),
            max,
            pre_minimize_states,
            pre_minimize_transitions,
            dfa.num_states(),
            dfa_transition_count(&dfa),
            body_ms,
            suffix_ms,
            construct_ms,
            minimize_ms,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Some((dfa, false))
}



pub(super) fn prepend_literal_prefix_to_dfa(prefix_bytes: &[u8], tail_dfa: DFA) -> Option<DFA> {
    if prefix_bytes.is_empty() {
        return Some(tail_dfa);
    }

    let total_states = prefix_bytes.len().checked_add(tail_dfa.num_states())?;
    let tail_offset = prefix_bytes.len() as u32;
    let mut dfa = DFA::new(total_states);
    dfa.ensure_group_capacity(tail_dfa.num_groups());

    for (i, &byte) in prefix_bytes.iter().enumerate() {
        let mut future = BitSet::new(tail_dfa.num_groups());
        if tail_dfa.num_groups() > 0 {
            future.set(0);
        }
        dfa.overwrite_state_metadata(i as u32, BitSet::new(tail_dfa.num_groups()), future);
        let target = if i + 1 == prefix_bytes.len() {
            tail_offset
        } else {
            (i + 1) as u32
        };
        dfa.set_transitions_from_sorted_entries(i as u32, vec![(byte, target)]);
    }

    for state_id in 0..tail_dfa.num_states() {
        let mapped_state = tail_offset + state_id as u32;
        dfa.overwrite_state_metadata(
            mapped_state,
            tail_dfa.finalizers(state_id as u32).clone(),
            tail_dfa.possible_future_group_ids(state_id as u32).clone(),
        );
        let transitions = tail_dfa.states()[state_id]
            .transitions
            .iter()
            .map(|(byte, &target)| (byte, tail_offset + target))
            .collect();
        dfa.set_transitions_from_sorted_entries(mapped_state, transitions);
    }

    Some(dfa)
}

fn build_bounded_repeat_with_regex_suffix_with_options(
    parts: &[Expr],
    preserve_coordinates: bool,
) -> Option<(DFA, bool)> {
    build_bounded_repeat_with_regex_suffix_with_options_and_cache(
        parts,
        preserve_coordinates,
        None,
    )
}

fn add_disjoint_literal_alternative_from_start(
    dfa: &mut DFA,
    literal: &[u8],
) -> Option<()> {
    if literal.is_empty() {
        mark_state_accepting(dfa, 0);
        return Some(());
    }
    // This helper deliberately handles only a branch whose first byte is not
    // already live from the existing start state.  Under that proof the new
    // literal path is disjoint after its first byte, so grafting a private
    // chain is exactly a DFA union and needs no determinization.
    if dfa.step(0, literal[0]).is_some() {
        return None;
    }
    dfa.ensure_group_capacity(1);
    let mut source = 0u32;
    for (index, &byte) in literal.iter().enumerate() {
        let target = dfa.add_state();
        let is_last = index + 1 == literal.len();
        let mut finalizers = BitSet::new(1);
        let mut futures = BitSet::new(1);
        if is_last {
            finalizers.set(0);
        } else {
            futures.set(0);
        }
        dfa.overwrite_state_metadata(target, finalizers, futures);
        dfa.add_transition(source, byte, target);
        source = target;
    }
    Some(())
}

pub(super) fn build_prefixed_bounded_repeat_with_suffix_dfa_with_options_and_cache(
    parts: &[Expr],
    preserve_coordinates: bool,
    cache: Option<&RepeatBaseDfaCache>,
) -> Option<(DFA, bool)> {
    let mut flat_parts = Vec::new();
    for part in parts {
        match part {
            Expr::Shared(inner) => match inner.as_ref() {
                Expr::Seq(inner_parts) => flat_parts.extend(inner_parts.iter().cloned()),
                _ => flat_parts.push(part.clone()),
            },
            Expr::Seq(inner_parts) => flat_parts.extend(inner_parts.iter().cloned()),
            _ => flat_parts.push(part.clone()),
        }
    }

    let parts = flat_parts.as_slice();
    if parts.len() < 2 {
        return None;
    }

    // A literal prefix followed directly by a bounded repeat is the same
    // structural family as the suffix-bearing forms below, but it has no
    // boundary ambiguity to resolve. Compile the repeat directly and prepend
    // the fixed bytes instead of falling back through an unrolled NFA.
    let final_part = match parts.last()? {
        Expr::Shared(inner) => inner.as_ref(),
        other => other,
    };
    if let Expr::Repeat {
        expr,
        min,
        max: Some(max),
    } = final_part
    {
        let prefix_bytes = collect_suffix_bytes(&parts[..parts.len() - 1])?;
        let base_dfa = cached_direct_bounded_repeat_base_dfa_unconditionally(expr, cache)?;
        let tail_dfa = build_bounded_repeat_dfa_from_base(base_dfa.as_ref(), *min, *max)?;
        return prepend_literal_prefix_to_dfa(&prefix_bytes, tail_dfa)
            .map(|dfa| (dfa, false));
    }

    for repeat_index in 1..parts.len() - 1 {
        let repeat_expr = match &parts[repeat_index] {
            Expr::Shared(inner) => inner.as_ref(),
            other => other,
        };
        let Expr::Repeat { .. } = repeat_expr else {
            continue;
        };

        let prefix_bytes = collect_suffix_bytes(&parts[..repeat_index])?;
        let tail_parts: Vec<Expr> = parts[repeat_index..].to_vec();
        let (tail_dfa, needs_future_recompute) =
            build_bounded_repeat_with_suffix_dfa_with_cache(&tail_parts, cache)
                .or_else(|| {
                    build_bounded_repeat_with_regex_suffix_with_options_and_cache(
                        &tail_parts,
                        preserve_coordinates,
                        cache,
                    )
                })?;
        let dfa = prepend_literal_prefix_to_dfa(&prefix_bytes, tail_dfa)?;
        return Some((dfa, needs_future_recompute));
    }

    // Fixed literal prefix + optional repeat tail + fixed literal suffix.
    // JSON property terminals frequently lower to this shape.  Compile the
    // non-empty arm with the existing bounded-repeat fast path, then graft the
    // epsilon arm's literal suffix directly when its first byte is disjoint
    // from the non-empty arm at the tail start.  The byte-disjointness check is
    // a complete determinism proof for this representation rewrite; otherwise
    // fail closed to the generic compiler.
    for optional_index in 1..parts.len().saturating_sub(1) {
        let Some(mut tail_parts) = optional_tail_parts(&parts[optional_index]) else {
            continue;
        };
        if tail_parts.len() < 2 {
            continue;
        }
        let Some(prefix_bytes) = collect_suffix_bytes(&parts[..optional_index]) else {
            continue;
        };
        let Some(suffix_bytes) = collect_suffix_bytes(&parts[optional_index + 1..]) else {
            continue;
        };
        tail_parts.extend_from_slice(&parts[optional_index + 1..]);
        let Some((mut tail_dfa, needs_future_recompute)) =
            build_bounded_repeat_with_suffix_dfa_with_cache(&tail_parts, cache)
                .or_else(|| {
                    build_bounded_repeat_with_regex_suffix_with_options_and_cache(
                        &tail_parts,
                        preserve_coordinates,
                        cache,
                    )
                })
        else {
            continue;
        };
        if add_disjoint_literal_alternative_from_start(&mut tail_dfa, &suffix_bytes).is_none() {
            continue;
        }
        let dfa = prepend_literal_prefix_to_dfa(&prefix_bytes, tail_dfa)?;
        return Some((dfa, needs_future_recompute));
    }

    if parts.len() == 2 {
        let prefix_bytes = collect_suffix_bytes(&parts[..1])?;
        let tail_parts = optional_tail_parts(&parts[1])?;
        if tail_parts.len() >= 2 {
            let (tail_dfa, needs_future_recompute) =
                build_bounded_repeat_with_suffix_dfa_with_cache(&tail_parts, cache)
                    .or_else(|| {
                        build_bounded_repeat_with_regex_suffix_with_options_and_cache(
                            &tail_parts,
                            preserve_coordinates,
                            cache,
                        )
                    })?;
            let mut dfa = prepend_literal_prefix_to_dfa(&prefix_bytes, tail_dfa)?;
            mark_state_accepting(&mut dfa, prefix_bytes.len() as u32);
            return Some((dfa, needs_future_recompute));
        }
    }

    None
}



fn build_prefixed_bounded_repeat_with_suffix_dfa_with_options(
    parts: &[Expr],
    preserve_coordinates: bool,
) -> Option<(DFA, bool)> {
    build_prefixed_bounded_repeat_with_suffix_dfa_with_options_and_cache(
        parts,
        preserve_coordinates,
        None,
    )
}
