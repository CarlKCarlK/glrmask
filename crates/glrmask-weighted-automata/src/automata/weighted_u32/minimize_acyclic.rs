//! Acyclic weighted DWA minimization via weight pushing and height-layered coloring.
//!
//! The pipeline pushes weights backward to discard dead token flow, groups
//! states by topological height, colors each height bucket subject to
//! compatibility constraints, and reconstructs the minimized automaton from the
//! merged buckets.
use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use range_set_blaze::RangeSetBlaze;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use super::dwa::{DWA, DWAState};
use super::equivalence::find_difference;
use crate::ds::weight::{SharedTokenSet, Weight, WeightIntersectionIndex};

type Label = i32;

/// Ordering policy for exact pointwise merge groups during acyclic weighted
/// minimization. The policy affects only representation choices among already
/// compatible classes; it does not alter the accepted weighted language.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PointwiseClassOrder {
    /// Preserve the partition-refinement class order used historically.
    Stable,
    /// Place denser partial behavior functions first so their constraints are
    /// absorbed before smaller compatible fragments are grouped greedily.
    DescendingDomain,
}

const UNMAPPED: u32 = u32::MAX;
fn weighted_dwa_minimize_profile_enabled() -> bool {
    std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
}

fn reconstruction_token_range_coalescing_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("GLRMASK_WEIGHT_UNION_COALESCE_TOKEN_RANGES")
            .map(|value| {
                let value = value.trim();
                !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
            })
            .unwrap_or(false)
    })
}

fn mapped_target(old_to_new: &[u32], target: u32) -> Option<u32> {
    let mapped = old_to_new.get(target as usize).copied().unwrap_or(UNMAPPED);
    (mapped != UNMAPPED).then_some(mapped)
}
fn compute_reachable_from_start(dwa: &DWA, start_state: usize) -> Vec<bool> {
    let mut reachable = vec![false; dwa.states().len()];
    if start_state >= dwa.states().len() {
        return reachable;
    }

    let mut stack = vec![start_state];
    while let Some(state_id) = stack.pop() {
        if reachable[state_id] {
            continue;
        }

        reachable[state_id] = true;
        for (target, _) in dwa.states()[state_id].transitions.values() {
            let target = *target as usize;
            if target < dwa.states().len() && !reachable[target] {
                stack.push(target);
            }
        }
    }

    reachable
}

fn weight_body_id(weight: &Weight) -> usize {
    weight.ptr_key()
}

fn intersection_memo_key(left: &Weight, right: &Weight) -> (usize, usize) {
    let left_id = weight_body_id(left);
    let right_id = weight_body_id(right);
    if left_id <= right_id {
        (left_id, right_id)
    } else {
        (right_id, left_id)
    }
}

fn memoized_intersection(
    cache: &mut FxHashMap<(usize, usize), Weight>,
    left: &Weight,
    right: &Weight,
    index: Option<&WeightIntersectionIndex>,
) -> Weight {
    if left.is_empty() || right.is_empty() {
        return Weight::empty();
    }
    if left.is_full() {
        return right.clone();
    }
    if right.is_full() {
        return left.clone();
    }

    let key = intersection_memo_key(left, right);
    if let Some(existing) = cache.get(&key) {
        return existing.clone();
    }

    let value = index.map_or_else(|| left.intersection(right), |index| left.intersection_with_index(index));
    cache.insert(key, value.clone());
    value
}

// Push weights backward before any topological analysis.

/// Push weights: intersect each transition weight with the backward-reachable
/// set of its target state.  This ensures transitions only carry tokens that
/// can actually reach acceptance, enabling more state merges.
///
/// Returns (changed, topo_order, reachable_sets) so callers can reuse them.
pub fn push_weights(dwa: &mut DWA) -> (bool, Option<Vec<usize>>, Vec<Weight>) {
    let profile_enabled = weighted_dwa_minimize_profile_enabled();
    let n = dwa.states().len();
    if n == 0 {
        return (false, Some(Vec::new()), Vec::new());
    }

    // 1. Topological order (Kahn's algorithm)
    let topo_started_at = profile_enabled.then(Instant::now);
    let Some(topo) = compute_topo_order(dwa) else {
        return (false, None, Vec::new()); // cyclic
    };
    let topo_ms = topo_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    // 2+3 combined: backward reachable sets + push transition weights.
    // In reverse topo order, each state's targets have already been processed,
    // so we compute reachable[u] and push transitions in a single pass.
    let mut reachable: Vec<Weight> = vec![Weight::empty(); n];
    let mut intersection_cache = FxHashMap::default();
    let mut incoming_transition_counts = vec![0usize; n];
    for state in dwa.states() {
        for (target, _) in state.transitions.values() {
            if (*target as usize) < n {
                incoming_transition_counts[*target as usize] += 1;
            }
        }
    }
    let mut intersection_indexes = FxHashMap::<usize, WeightIntersectionIndex>::default();
    let mut indexed_intersection_uses = 0usize;
    let mut changed = false;
    let mut target_full = 0usize;
    let mut target_empty = 0usize;
    let mut target_partial = 0usize;
    let mut reachable_parts = 0usize;
    let mut pushed_transitions = 0usize;
    let mut intersection_ms = 0.0;
    let mut union_ms = 0.0;
    let mut apply_ms = 0.0;
    let mut union_size_histogram = [0usize; 7];
    let mut max_union_size = 0usize;
    let mut union_key_occurrences = 0usize;
    let mut union_key_repeats = 0usize;
    let mut union_keys_seen = FxHashSet::<Vec<usize>>::default();

    for &u in topo.iter().rev() {
        let state = &dwa.states()[u];
        let mut state_reachable_parts: Vec<Weight> =
            Vec::with_capacity(state.transitions.len() + 1);
        let mut acc_full = false;
        if let Some(final_weight) = &state.final_weight {
            if final_weight.is_full() {
                acc_full = true;
            } else if !final_weight.is_empty() {
                state_reachable_parts.push(final_weight.clone());
            }
        }
        let mut pushed: Vec<(Label, u32, Option<Weight>)> = Vec::new();
        for (&lbl, (target, w)) in &state.transitions {
            let t = *target as usize;
            if t >= n {
                continue;
            }
            if reachable[t].is_full() {
                target_full += 1;
                if !acc_full && !w.is_empty() {
                    if w.is_full() {
                        acc_full = true;
                        state_reachable_parts.clear();
                    } else {
                        state_reachable_parts.push(w.clone());
                    }
                }
                // w ∩ all = w, no push needed
            } else if reachable[t].is_empty() {
                target_empty += 1;
                // w ∩ empty = empty, remove transition
                pushed.push((lbl, *target, None));
                // Contributes nothing to acc
            } else {
                target_partial += 1;
                let intersection_started_at = profile_enabled.then(Instant::now);
                let index = (incoming_transition_counts[t] >= 8).then(|| {
                    let key = reachable[t].ptr_key();
                    intersection_indexes
                        .entry(key)
                        .or_insert_with(|| reachable[t].intersection_index())
                });
                if index.is_some() {
                    indexed_intersection_uses += 1;
                }
                let new_w = memoized_intersection(
                    &mut intersection_cache,
                    w,
                    &reachable[t],
                    index.as_deref(),
                );
                if let Some(started_at) = intersection_started_at {
                    intersection_ms += started_at.elapsed().as_secs_f64() * 1000.0;
                }
                if !acc_full && !new_w.is_empty() {
                    if new_w.is_full() {
                        acc_full = true;
                        state_reachable_parts.clear();
                    } else {
                        state_reachable_parts.push(new_w.clone());
                    }
                }
                if new_w != *w {
                    pushed.push((lbl, *target, if new_w.is_empty() { None } else { Some(new_w) }));
                }
            }
        }
        reachable[u] = if acc_full {
            Weight::all()
        } else {
            reachable_parts += state_reachable_parts.len();
            if profile_enabled {
                let union_size = state_reachable_parts.len();
                max_union_size = max_union_size.max(union_size);
                let bucket = match union_size {
                    0 => 0,
                    1 => 1,
                    2 => 2,
                    3 => 3,
                    4 => 4,
                    5..=16 => 5,
                    _ => 6,
                };
                union_size_histogram[bucket] += 1;
                if union_size >= 2 {
                    let mut key: Vec<usize> = state_reachable_parts
                        .iter()
                        .map(weight_body_id)
                        .collect();
                    key.sort_unstable();
                    key.dedup();
                    union_key_occurrences += 1;
                    if !union_keys_seen.insert(key) {
                        union_key_repeats += 1;
                    }
                }
            }
            let union_started_at = profile_enabled.then(Instant::now);
            let result = Weight::union_all(state_reachable_parts.iter());
            if let Some(started_at) = union_started_at {
                union_ms += started_at.elapsed().as_secs_f64() * 1000.0;
            }
            result
        };

        pushed_transitions += pushed.len();
        let apply_started_at = profile_enabled.then(Instant::now);
        for (lbl, target, new_w_opt) in pushed {
            if let Some(new_w) = new_w_opt {
                dwa.states_mut()[u].transitions.insert(lbl, (target, new_w));
            } else {
                dwa.states_mut()[u].transitions.remove(&lbl);
            }
            changed = true;
        }
        if let Some(started_at) = apply_started_at {
            apply_ms += started_at.elapsed().as_secs_f64() * 1000.0;
        }
    }

    if profile_enabled {
        eprintln!(
            "[glrmask/profile][weighted_dwa_minimize_push] states={} topo_ms={:.3} target_full={} target_empty={} target_partial={} intersection_ms={:.3} intersection_cache_entries={} intersection_indexes={} indexed_intersection_uses={} reachable_parts={} union_ms={:.3} union_sizes=[{},{},{},{},{},{},{}] max_union_size={} union_key_occurrences={} union_unique_keys={} union_key_repeats={} pushed_transitions={} apply_ms={:.3}",
            n,
            topo_ms,
            target_full,
            target_empty,
            target_partial,
            intersection_ms,
            intersection_cache.len(),
            intersection_indexes.len(),
            indexed_intersection_uses,
            reachable_parts,
            union_ms,
            union_size_histogram[0],
            union_size_histogram[1],
            union_size_histogram[2],
            union_size_histogram[3],
            union_size_histogram[4],
            union_size_histogram[5],
            union_size_histogram[6],
            max_union_size,
            union_key_occurrences,
            union_keys_seen.len(),
            union_key_repeats,
            pushed_transitions,
            apply_ms,
        );
    }

    (changed, Some(topo), reachable)
}

// Topological analysis.

fn compute_topo_order(dwa: &DWA) -> Option<Vec<usize>> {
    let n = dwa.states().len();
    let mut in_degree = vec![0u32; n];
    for state in dwa.states() {
        for (_, (target, _)) in &state.transitions {
            let t = *target as usize;
            if t < n {
                in_degree[t] += 1;
            }
        }
    }

    let mut queue: Vec<usize> = in_degree
        .iter()
        .enumerate()
        .filter(|(_, d)| **d == 0)
        .map(|(i, _)| i)
        .collect();
    let mut head = 0;
    let mut topo = Vec::with_capacity(n);

    while head < queue.len() {
        let u = queue[head];
        head += 1;
        topo.push(u);
        for (_, (target, _)) in &dwa.states()[u].transitions {
            let t = *target as usize;
            if t < n {
                in_degree[t] -= 1;
                if in_degree[t] == 0 {
                    queue.push(t);
                }
            }
        }
    }

    if topo.len() == n {
        Some(topo)
    } else {
        None // cyclic
    }
}

/// Recover the live path domain of an already path-conditioned DWA from its
/// incoming edge weights. Determinization emits cumulative path weights on
/// edges; for such a DWA, all incoming weights to a state describe exactly the
/// token contexts in which that residual is reachable.

/// Exact reverse structural hash-cons for an acyclic weighted DWA.
///
/// States are merged only when their complete weighted suffix representation is
/// identical after targets have already been hash-consed: same final weight and
/// the same ordered `label -> (target, weight)` row. This performs no weight
/// pushing or compatibility reasoning; it is a pure DAG quotient and therefore
/// preserves the weighted language exactly.
pub fn reverse_hashcons_acyclic_owned(dwa: DWA) -> DWA {
    if dwa.states().len() <= 1 {
        return dwa;
    }
    let Some(topo) = compute_topo_order(&dwa) else {
        return dwa;
    };

    type Signature = (Option<Weight>, Vec<(Label, u32, Weight)>);
    let old_start = dwa.start_state() as usize;
    let mut old_to_new = vec![UNMAPPED; dwa.states().len()];
    let mut intern = FxHashMap::<Signature, u32>::default();
    let mut new_states = Vec::<DWAState>::new();

    for &old_id in topo.iter().rev() {
        let old = &dwa.states()[old_id];
        let transitions = old
            .transitions
            .iter()
            .map(|(&label, (target, weight))| {
                let mapped = old_to_new[*target as usize];
                debug_assert_ne!(mapped, UNMAPPED, "reverse topo target must be mapped");
                (label, mapped, weight.clone())
            })
            .collect::<Vec<_>>();
        let signature = (old.final_weight.clone(), transitions.clone());
        if let Some(&existing) = intern.get(&signature) {
            old_to_new[old_id] = existing;
            continue;
        }

        let new_id = new_states.len() as u32;
        let state = DWAState {
            transitions: transitions
                .into_iter()
                .map(|(label, target, weight)| (label, (target, weight)))
                .collect(),
            final_weight: old.final_weight.clone(),
        };
        new_states.push(state);
        intern.insert(signature, new_id);
        old_to_new[old_id] = new_id;
    }

    let start_state = old_to_new.get(old_start).copied().unwrap_or(UNMAPPED);
    debug_assert_ne!(start_state, UNMAPPED);
    DWA::from_parts(new_states, start_state)
}

fn compute_forward_path_domains(dwa: &DWA) -> Option<(Vec<usize>, Vec<Weight>)> {
    let topo = compute_topo_order(dwa)?;
    let n = dwa.states().len();
    let mut incoming = vec![SmallVec::<[Weight; 2]>::new(); n];
    for state in dwa.states() {
        for (target, weight) in state.transitions.values() {
            let target = *target as usize;
            if target < n && !weight.is_empty() {
                incoming[target].push(weight.clone());
            }
        }
    }
    let mut domains = vec![Weight::empty(); n];
    let start = dwa.start_state() as usize;
    if start < n {
        domains[start] = Weight::all();
    }
    for state in 0..n {
        if state == start || incoming[state].is_empty() {
            continue;
        }
        // Determinized residuals commonly have many identical incoming path
        // domains. Pointer-dedup keeps the union proportional to distinct
        // domains without hashing the full weight value.
        incoming[state].sort_unstable_by_key(Weight::ptr_key);
        incoming[state].dedup_by_key(|weight| weight.ptr_key());
        domains[state] = Weight::union_all(incoming[state].iter());
    }
    Some((topo, domains))
}

/// Needed sets: for each state, the set of tokens that can flow from that
/// state to any accepting state.  Computed in topological order (leaves first).
fn compute_heights(dwa: &DWA, topo: &[usize]) -> Vec<usize> {
    let n = dwa.states().len();
    let mut heights = vec![0usize; n];
    // Process in reverse topo order so children are resolved before parents
    for &u in topo.iter().rev() {
        heights[u] = dwa.states()[u]
            .transitions
            .values()
            .filter_map(|(target, _)| {
                let t = *target as usize;
                (t < n).then(|| heights[t] + 1)
            })
            .max()
            .unwrap_or(0);
    }
    heights
}

#[derive(Clone)]
struct ProductiveTransition {
    label: Label,
    target: u32,
    weight: Weight,
}

fn compute_productive_transitions(dwa: &DWA, needed: &[Weight]) -> Vec<Vec<ProductiveTransition>> {
    let n = dwa.states().len();
    let mut result = Vec::with_capacity(n);

    for state in dwa.states() {
        let mut transitions = Vec::with_capacity(state.transitions.len());
        for (&label, (target, weight)) in &state.transitions {
            let t = *target as usize;
            if t >= n {
                continue;
            }
            // After push_weights, all remaining transitions are already productive
            // (w_pushed = w_orig ∩ reachable[t], and needed == reachable from push).
            // The intersection w_pushed ∩ needed[t] = w_pushed, so just clone.
            if needed[t].is_empty() {
                continue;
            }
            transitions.push(ProductiveTransition {
                label,
                target: *target,
                weight: weight.clone(),
            });
        }
        result.push(transitions);
    }

    result
}

#[cfg(debug_assertions)]
fn debug_assert_pushed_weights_within_needed(dwa: &DWA, needed: &[Weight]) {
    debug_assert_eq!(dwa.states().len(), needed.len());

    let n = dwa.states().len();
    for (state_id, state) in dwa.states().iter().enumerate() {
        let source_needed = &needed[state_id];
        if let Some(final_weight) = &state.final_weight {
            debug_assert!(
                final_weight.is_subset(source_needed),
                "pushed DWA invariant violated: final weight at state {state_id} is not contained in needed[state]",
            );
        }

        for (&label, (target, weight)) in &state.transitions {
            let target_id = *target as usize;
            if target_id >= n {
                continue;
            }
            debug_assert!(
                weight.is_subset(source_needed),
                "pushed DWA invariant violated: transition weight at state {state_id}, label {label} is not contained in needed[source]",
            );
            debug_assert!(
                weight.is_subset(&needed[target_id]),
                "pushed DWA invariant violated: transition weight at state {state_id}, label {label} is not contained in needed[target]",
            );
        }
    }
}

// Incompatibility graph and coloring.

/// Check if `w_a.intersection(domain) == w_b.intersection(domain)` without
/// allocating intermediate Weight objects.  Three-way merge scan over the
/// tsid dimension; at each atomic sub-interval the token-level agreement is
/// verified with a zero-allocation sweep (`token_sets_agree_on_domain`).
fn weights_equal_on_domain(w_a: &Weight, w_b: &Weight, domain: &Weight) -> bool {
    // Fast path: identical Arc ⇒ trivially equal on any domain.
    if w_a.storage_ptr_eq(w_b) {
        return true;
    }
    if domain.is_empty() {
        return true;
    }

    let mut a_iter = w_a.raw_range_values();
    let mut b_iter = w_b.raw_range_values();
    let mut a_entry = a_iter.next();
    let mut b_entry = b_iter.next();

    for (d_range, d_tokens) in domain.raw_range_values() {
        let d_lo = *d_range.start();
        let d_hi = *d_range.end();

        if d_tokens.is_empty() {
            continue;
        }

        // Skip entries fully before this domain range.
        while a_entry.as_ref().is_some_and(|(r, _)| *r.end() < d_lo) {
            a_entry = a_iter.next();
        }
        while b_entry.as_ref().is_some_and(|(r, _)| *r.end() < d_lo) {
            b_entry = b_iter.next();
        }

        let mut pos = d_lo;
        while pos <= d_hi {
            // Determine w_a's token set at `pos` and how far it extends.
            let (a_tokens, a_bound): (Option<&Arc<RangeSetBlaze<u32>>>, u32) = match &a_entry {
                Some((r, tokens)) if *r.start() <= pos => (Some(tokens), *r.end()),
                Some((r, _)) => (None, r.start() - 1), // gap; r.start()>pos≥1
                None => (None, d_hi),
            };
            let (b_tokens, b_bound): (Option<&Arc<RangeSetBlaze<u32>>>, u32) = match &b_entry {
                Some((r, tokens)) if *r.start() <= pos => (Some(tokens), *r.end()),
                Some((r, _)) => (None, r.start() - 1),
                None => (None, d_hi),
            };

            let sub_end = d_hi.min(a_bound).min(b_bound);

            // Check: a_tokens ∩ d_tokens  ==  b_tokens ∩ d_tokens
            match (a_tokens, b_tokens) {
                (None, None) => { /* both empty → agree */ }
                (Some(at), None) => {
                    if !at.as_ref().is_disjoint(d_tokens.as_ref()) {
                        return false;
                    }
                }
                (None, Some(bt)) => {
                    if !bt.as_ref().is_disjoint(d_tokens.as_ref()) {
                        return false;
                    }
                }
                (Some(at), Some(bt)) => {
                    if !Arc::ptr_eq(at, bt) && at.as_ref() != bt.as_ref() {
                        if !token_sets_agree_on_domain(
                            at.as_ref(),
                            bt.as_ref(),
                            d_tokens.as_ref(),
                        ) {
                            return false;
                        }
                    }
                }
            }

            if sub_end == u32::MAX {
                break;
            }
            pos = sub_end + 1;
            while a_entry.as_ref().is_some_and(|(r, _)| *r.end() < pos) {
                a_entry = a_iter.next();
            }
            while b_entry.as_ref().is_some_and(|(r, _)| *r.end() < pos) {
                b_entry = b_iter.next();
            }
        }
    }

    true
}

/// Zero-allocation check: `a ∩ d == b ∩ d` for `RangeSetBlaze` values.
/// Sweep-scans the sorted ranges of `a`, `b`, and `d` in parallel.
fn token_sets_agree_on_domain(
    a: &RangeSetBlaze<u32>,
    b: &RangeSetBlaze<u32>,
    d: &RangeSetBlaze<u32>,
) -> bool {
    let mut a_ranges = a.ranges().peekable();
    let mut b_ranges = b.ranges().peekable();

    for d_range in d.ranges() {
        let d_lo = *d_range.start();
        let d_hi = *d_range.end();

        while a_ranges.peek().is_some_and(|r| *r.end() < d_lo) {
            a_ranges.next();
        }
        while b_ranges.peek().is_some_and(|r| *r.end() < d_lo) {
            b_ranges.next();
        }

        let mut pos = d_lo;
        while pos <= d_hi {
            let (in_a, a_end) = match a_ranges.peek() {
                Some(r) if *r.start() <= pos => (true, *r.end()),
                Some(r) => (false, r.start() - 1),
                None => (false, d_hi),
            };
            let (in_b, b_end) = match b_ranges.peek() {
                Some(r) if *r.start() <= pos => (true, *r.end()),
                Some(r) => (false, r.start() - 1),
                None => (false, d_hi),
            };

            if in_a != in_b {
                return false;
            }

            let sub_end = d_hi.min(a_end).min(b_end);
            if sub_end == u32::MAX {
                break;
            }
            pos = sub_end + 1;
            while a_ranges.peek().is_some_and(|r| *r.end() < pos) {
                a_ranges.next();
            }
            while b_ranges.peek().is_some_and(|r| *r.end() < pos) {
                b_ranges.next();
            }
        }
    }

    true
}

fn token_sets_intersect_three(
    a: &RangeSetBlaze<u32>,
    b: &RangeSetBlaze<u32>,
    c: &RangeSetBlaze<u32>,
) -> bool {
    let mut a_ranges = a.ranges().peekable();
    let mut b_ranges = b.ranges().peekable();
    let mut c_ranges = c.ranges().peekable();

    loop {
        let (Some(a_range), Some(b_range), Some(c_range)) =
            (a_ranges.peek(), b_ranges.peek(), c_ranges.peek())
        else {
            return false;
        };

        let start = (*a_range.start()).max(*b_range.start()).max(*c_range.start());
        let end = (*a_range.end()).min(*b_range.end()).min(*c_range.end());
        if start <= end {
            return true;
        }

        let min_end = (*a_range.end()).min(*b_range.end()).min(*c_range.end());
        if *a_range.end() == min_end {
            a_ranges.next();
        }
        if *b_range.end() == min_end {
            b_ranges.next();
        }
        if *c_range.end() == min_end {
            c_ranges.next();
        }
    }
}

fn token_sets_agree_on_domain_intersection(
    a: &RangeSetBlaze<u32>,
    b: &RangeSetBlaze<u32>,
    left: &RangeSetBlaze<u32>,
    right: &RangeSetBlaze<u32>,
) -> bool {
    let mut a_ranges = a.ranges().peekable();
    let mut b_ranges = b.ranges().peekable();
    let mut left_ranges = left.ranges().peekable();
    let mut right_ranges = right.ranges().peekable();

    loop {
        while let (Some(left_range), Some(right_range)) = (left_ranges.peek(), right_ranges.peek()) {
            if *left_range.end() < *right_range.start() {
                left_ranges.next();
            } else if *right_range.end() < *left_range.start() {
                right_ranges.next();
            } else {
                break;
            }
        }

        let (Some(left_range), Some(right_range)) = (left_ranges.peek(), right_ranges.peek()) else {
            return true;
        };

        let d_lo = (*left_range.start()).max(*right_range.start());
        let d_hi = (*left_range.end()).min(*right_range.end());
        while a_ranges.peek().is_some_and(|r| *r.end() < d_lo) {
            a_ranges.next();
        }
        while b_ranges.peek().is_some_and(|r| *r.end() < d_lo) {
            b_ranges.next();
        }

        let mut pos = d_lo;
        while pos <= d_hi {
            let (in_a, a_end) = match a_ranges.peek() {
                Some(r) if *r.start() <= pos => (true, *r.end()),
                Some(r) => (false, r.start() - 1),
                None => (false, d_hi),
            };
            let (in_b, b_end) = match b_ranges.peek() {
                Some(r) if *r.start() <= pos => (true, *r.end()),
                Some(r) => (false, r.start() - 1),
                None => (false, d_hi),
            };

            if in_a != in_b {
                return false;
            }

            let sub_end = d_hi.min(a_end).min(b_end);
            if sub_end == u32::MAX {
                break;
            }
            pos = sub_end + 1;
            while a_ranges.peek().is_some_and(|r| *r.end() < pos) {
                a_ranges.next();
            }
            while b_ranges.peek().is_some_and(|r| *r.end() < pos) {
                b_ranges.next();
            }
        }

        let left_ended = *left_range.end() == d_hi;
        let right_ended = *right_range.end() == d_hi;
        if left_ended {
            left_ranges.next();
        }
        if right_ended {
            right_ranges.next();
        }
    }
}

fn weight_is_disjoint_from_domain_intersection(
    weight: &Weight,
    left_domain: &Weight,
    right_domain: &Weight,
) -> bool {
    let mut weight_iter = weight.raw_range_values();
    let mut weight_entry = weight_iter.next();
    let mut left_iter = left_domain.raw_range_values();
    let mut right_iter = right_domain.raw_range_values();
    let mut left_entry = left_iter.next();
    let mut right_entry = right_iter.next();

    loop {
        while let (Some((left_range, _)), Some((right_range, _))) = (&left_entry, &right_entry) {
            if *left_range.end() < *right_range.start() {
                left_entry = left_iter.next();
            } else if *right_range.end() < *left_range.start() {
                right_entry = right_iter.next();
            } else {
                break;
            }
        }

        let (Some((left_range, left_tokens)), Some((right_range, right_tokens))) =
            (&left_entry, &right_entry)
        else {
            return true;
        };

        let d_lo = (*left_range.start()).max(*right_range.start());
        let d_hi = (*left_range.end()).min(*right_range.end());
        while weight_entry.as_ref().is_some_and(|(r, _)| *r.end() < d_lo) {
            weight_entry = weight_iter.next();
        }

        let mut pos = d_lo;
        while pos <= d_hi {
            let (weight_tokens, weight_end): (Option<&Arc<RangeSetBlaze<u32>>>, u32) = match &weight_entry {
                Some((r, tokens)) if *r.start() <= pos => (Some(tokens), *r.end()),
                Some((r, _)) => (None, r.start() - 1),
                None => (None, d_hi),
            };

            let sub_end = d_hi.min(weight_end);
            if let Some(weight_tokens) = weight_tokens {
                if token_sets_intersect_three(
                    weight_tokens.as_ref(),
                    left_tokens.as_ref(),
                    right_tokens.as_ref(),
                ) {
                    return false;
                }
            }

            if sub_end == u32::MAX {
                break;
            }
            pos = sub_end + 1;
            while weight_entry.as_ref().is_some_and(|(r, _)| *r.end() < pos) {
                weight_entry = weight_iter.next();
            }
        }

        let left_ended = *left_range.end() == d_hi;
        let right_ended = *right_range.end() == d_hi;
        if left_ended {
            left_entry = left_iter.next();
        }
        if right_ended {
            right_entry = right_iter.next();
        }
    }
}

fn weights_equal_on_domain_intersection(
    w_a: &Weight,
    w_b: &Weight,
    left_domain: &Weight,
    right_domain: &Weight,
) -> bool {
    if w_a.storage_ptr_eq(w_b) {
        return true;
    }

    let mut a_iter = w_a.raw_range_values();
    let mut b_iter = w_b.raw_range_values();
    let mut left_iter = left_domain.raw_range_values();
    let mut right_iter = right_domain.raw_range_values();
    let mut a_entry = a_iter.next();
    let mut b_entry = b_iter.next();
    let mut left_entry = left_iter.next();
    let mut right_entry = right_iter.next();

    loop {
        while let (Some((left_range, _)), Some((right_range, _))) = (&left_entry, &right_entry) {
            if *left_range.end() < *right_range.start() {
                left_entry = left_iter.next();
            } else if *right_range.end() < *left_range.start() {
                right_entry = right_iter.next();
            } else {
                break;
            }
        }

        let (Some((left_range, left_tokens)), Some((right_range, right_tokens))) =
            (&left_entry, &right_entry)
        else {
            return true;
        };

        if left_tokens.as_ref().is_disjoint(right_tokens.as_ref()) {
            let left_ended = *left_range.end() <= *right_range.end();
            let right_ended = *right_range.end() <= *left_range.end();
            if left_ended {
                left_entry = left_iter.next();
            }
            if right_ended {
                right_entry = right_iter.next();
            }
            continue;
        }

        let d_lo = (*left_range.start()).max(*right_range.start());
        let d_hi = (*left_range.end()).min(*right_range.end());
        while a_entry.as_ref().is_some_and(|(r, _)| *r.end() < d_lo) {
            a_entry = a_iter.next();
        }
        while b_entry.as_ref().is_some_and(|(r, _)| *r.end() < d_lo) {
            b_entry = b_iter.next();
        }

        let mut pos = d_lo;
        while pos <= d_hi {
            let (a_tokens, a_bound): (Option<&Arc<RangeSetBlaze<u32>>>, u32) = match &a_entry {
                Some((r, tokens)) if *r.start() <= pos => (Some(tokens), *r.end()),
                Some((r, _)) => (None, r.start() - 1),
                None => (None, d_hi),
            };
            let (b_tokens, b_bound): (Option<&Arc<RangeSetBlaze<u32>>>, u32) = match &b_entry {
                Some((r, tokens)) if *r.start() <= pos => (Some(tokens), *r.end()),
                Some((r, _)) => (None, r.start() - 1),
                None => (None, d_hi),
            };

            let sub_end = d_hi.min(a_bound).min(b_bound);
            match (a_tokens, b_tokens) {
                (None, None) => {}
                (Some(at), None) => {
                    if token_sets_intersect_three(
                        at.as_ref(),
                        left_tokens.as_ref(),
                        right_tokens.as_ref(),
                    ) {
                        return false;
                    }
                }
                (None, Some(bt)) => {
                    if token_sets_intersect_three(
                        bt.as_ref(),
                        left_tokens.as_ref(),
                        right_tokens.as_ref(),
                    ) {
                        return false;
                    }
                }
                (Some(at), Some(bt)) => {
                    if !Arc::ptr_eq(at, bt) && at.as_ref() != bt.as_ref() {
                        if !token_sets_agree_on_domain_intersection(
                            at.as_ref(),
                            bt.as_ref(),
                            left_tokens.as_ref(),
                            right_tokens.as_ref(),
                        ) {
                            return false;
                        }
                    }
                }
            }

            if sub_end == u32::MAX {
                break;
            }
            pos = sub_end + 1;
            while a_entry.as_ref().is_some_and(|(r, _)| *r.end() < pos) {
                a_entry = a_iter.next();
            }
            while b_entry.as_ref().is_some_and(|(r, _)| *r.end() < pos) {
                b_entry = b_iter.next();
            }
        }

        let left_ended = *left_range.end() == d_hi;
        let right_ended = *right_range.end() == d_hi;
        if left_ended {
            left_entry = left_iter.next();
        }
        if right_ended {
            right_entry = right_iter.next();
        }
    }
}

/// Check if two candidate states can be merged.
///
/// States are compatible if:
/// 1. Their needed sets don't overlap, OR
/// 2. On the overlapping domain, they have identical final weights and
///    identical transition targets (after remapping through old_to_new).
/// 3. Even when disjoint, transitions on the same label must go to the same
///    target (since the DWA can only store one target per label).
fn are_compatible(
    u: usize,
    v: usize,
    dwa: &DWA,
    needed: &[Weight],
    old_to_new: &[u32],
    productive_transitions: &[Vec<ProductiveTransition>],
    known_overlapping: bool,
) -> bool {
    let needed_u = &needed[u];
    let needed_v = &needed[v];

    let domain_disjoint = if known_overlapping {
        false
    } else {
        needed_u.is_disjoint(needed_v)
    };

    // Check transitions — do target conflict detection first (cheap).
    let n = dwa.states().len();
    let trans_u = &productive_transitions[u];
    let trans_v = &productive_transitions[v];

    // Quick target-conflict check first (no weight ops needed)
    {
        let mut idx_u = 0usize;
        let mut idx_v = 0usize;
        while idx_u < trans_u.len() || idx_v < trans_v.len() {
            let (entry_u, entry_v) = match (trans_u.get(idx_u), trans_v.get(idx_v)) {
                (Some(u_entry), Some(v_entry)) => {
                    if u_entry.label == v_entry.label {
                        idx_u += 1;
                        idx_v += 1;
                        (Some(u_entry), Some(v_entry))
                    } else if u_entry.label < v_entry.label {
                        idx_u += 1;
                        (Some(u_entry), None)
                    } else {
                        idx_v += 1;
                        (None, Some(v_entry))
                    }
                }
                (Some(u_entry), None) => { idx_u += 1; (Some(u_entry), None) }
                (None, Some(v_entry)) => { idx_v += 1; (None, Some(v_entry)) }
                (None, None) => break,
            };

            let target_u = entry_u.and_then(|e| ((e.target as usize) < n).then_some(e.target));
            let target_v = entry_v.and_then(|e| ((e.target as usize) < n).then_some(e.target));
            let mapped_u = target_u.and_then(|t| mapped_target(old_to_new, t));
            let mapped_v = target_v.and_then(|t| mapped_target(old_to_new, t));
            match (mapped_u, mapped_v) {
                (Some(mu), Some(mv)) if mu != mv => {
                    let has_u = entry_u.is_some_and(|e| !e.weight.is_empty());
                    let has_v = entry_v.is_some_and(|e| !e.weight.is_empty());
                    if has_u || has_v {
                        return false;
                    }
                }
                _ => {}
            }
        }
    }

    // If domains are disjoint we're done — no overlap weight checks needed.
    if domain_disjoint {
        return true;
    }

    // Check if all transition weights are identical (Arc-equal or value-equal).
    // This is a fast path that avoids computing the overlap weight entirely.
    {
        let mut all_equal = true;
        let fw_u = dwa.states()[u].final_weight.as_ref();
        let fw_v = dwa.states()[v].final_weight.as_ref();
        match (fw_u, fw_v) {
            (Some(wu), Some(wv)) if wu == wv => {}
            (None, None) => {}
            _ => { all_equal = false; }
        }
        if all_equal {
            let mut idx_u = 0usize;
            let mut idx_v = 0usize;
            while idx_u < trans_u.len() || idx_v < trans_v.len() {
                let (eu, ev) = match (trans_u.get(idx_u), trans_v.get(idx_v)) {
                    (Some(a), Some(b)) => {
                        if a.label == b.label { idx_u += 1; idx_v += 1; (Some(a), Some(b)) }
                        else if a.label < b.label { idx_u += 1; (Some(a), None) }
                        else { idx_v += 1; (None, Some(b)) }
                    }
                    (Some(a), None) => { idx_u += 1; (Some(a), None) }
                    (None, Some(b)) => { idx_v += 1; (None, Some(b)) }
                    (None, None) => break,
                };
                let w_u = eu.map(|e| &e.weight);
                let w_v = ev.map(|e| &e.weight);
                match (w_u, w_v) {
                    (Some(wu), Some(wv)) if wu == wv => {}
                    (None, None) => {}
                    _ => { all_equal = false; break; }
                }
            }
        }
        if all_equal {
            return true;
        }
    }

    // Slow path: compute overlap and check weight equality on the intersection
    let overlap = needed_u.intersection(needed_v);

    // Check final weights on the overlapping domain
    {
        let fw_u = dwa.states()[u].final_weight.as_ref();
        let fw_v = dwa.states()[v].final_weight.as_ref();
        match (fw_u, fw_v) {
            (Some(wu), Some(wv)) => {
                if !weights_equal_on_domain(wu, wv, &overlap) {
                    return false;
                }
            }
            (Some(fw), None) | (None, Some(fw)) => {
                if !fw.is_disjoint(&overlap) {
                    return false;
                }
            }
            (None, None) => {}
        }
    }

    // Slow path: check overlap weights per transition label.
    // Target conflicts were already caught in pass 1 above.
    {
        let mut idx_u = 0usize;
        let mut idx_v = 0usize;
        while idx_u < trans_u.len() || idx_v < trans_v.len() {
            let (entry_u, entry_v) = match (trans_u.get(idx_u), trans_v.get(idx_v)) {
                (Some(a), Some(b)) => {
                    if a.label == b.label { idx_u += 1; idx_v += 1; (Some(a), Some(b)) }
                    else if a.label < b.label { idx_u += 1; (Some(a), None) }
                    else { idx_v += 1; (None, Some(b)) }
                }
                (Some(a), None) => { idx_u += 1; (Some(a), None) }
                (None, Some(b)) => { idx_v += 1; (None, Some(b)) }
                (None, None) => break,
            };

            let w_u_full = entry_u.and_then(|e| ((e.target as usize) < n).then_some(&e.weight));
            let w_v_full = entry_v.and_then(|e| ((e.target as usize) < n).then_some(&e.weight));

            // Fast path: if both full weights are equal, overlap restrictions are too.
            match (w_u_full, w_v_full) {
                (Some(wu), Some(wv)) if wu == wv => continue,
                (None, None) => continue,
                _ => {}
            }

            let u_disjoint = w_u_full.map_or(true, |w| w.is_disjoint(&overlap));
            let v_disjoint = w_v_full.map_or(true, |w| w.is_disjoint(&overlap));

            if u_disjoint && v_disjoint {
                continue;
            }
            if u_disjoint != v_disjoint {
                return false;
            }

            // Both non-empty on overlap → check equality.
            if !weights_equal_on_domain(w_u_full.unwrap(), w_v_full.unwrap(), &overlap) {
                return false;
            }

            // Equal on overlap → targets must agree (re-check for this specific case).
            let target_u = entry_u.and_then(|e| ((e.target as usize) < n).then_some(e.target));
            let target_v = entry_v.and_then(|e| ((e.target as usize) < n).then_some(e.target));
            let mapped_u = target_u.and_then(|t| mapped_target(old_to_new, t));
            let mapped_v = target_v.and_then(|t| mapped_target(old_to_new, t));
            match (mapped_u, mapped_v) {
                (Some(mu), Some(mv)) if mu != mv => return false,
                (Some(_), None) | (None, Some(_)) => return false,
                _ => {}
            }
        }
    }

    true
}

#[derive(Clone)]
struct ClassProfile {
    targets: Vec<(Label, u32)>,
    weights: Vec<(Label, Weight)>,
    final_weight: Option<Weight>,
}

fn build_class_profile(
    rep: usize,
    old_to_new: &[u32],
    productive_transitions: &[Vec<ProductiveTransition>],
    dwa: &DWA,
) -> ClassProfile {
    let mut targets = Vec::with_capacity(productive_transitions[rep].len());
    let mut weights = Vec::with_capacity(productive_transitions[rep].len());
    for pt in &productive_transitions[rep] {
        if let Some(mapped) = mapped_target(old_to_new, pt.target) {
            targets.push((pt.label, mapped));
        }
        weights.push((pt.label, pt.weight.clone()));
    }
    targets.sort_unstable_by_key(|(label, _)| *label);
    weights.sort_unstable_by_key(|(label, _)| *label);
    ClassProfile {
        targets,
        weights,
        final_weight: dwa.states()[rep].final_weight.clone(),
    }
}

/// Sparse pointwise behavior used to compare one group against a candidate
/// without rescanning all overlapping members.
#[derive(Clone, Eq, PartialEq, Hash)]
struct PointwiseBehavior {
    final_active: bool,
    transitions: Vec<(Label, u32)>,
}

#[derive(Default)]
struct PointwiseBehaviorInterner {
    ids: FxHashMap<PointwiseBehavior, u32>,
    values: Vec<PointwiseBehavior>,
}

impl PointwiseBehaviorInterner {
    fn intern(&mut self, final_active: bool, transitions: Vec<(Label, u32)>) -> u32 {
        let behavior = PointwiseBehavior { final_active, transitions };
        if let Some(&id) = self.ids.get(&behavior) {
            return id;
        }
        let id = self.values.len() as u32;
        self.values.push(behavior.clone());
        self.ids.insert(behavior, id);
        id
    }

    fn get(&self, id: u32) -> &PointwiseBehavior {
        &self.values[id as usize]
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
struct TokenBehaviorRange {
    start: u32,
    end: u32,
    behavior: u32,
}

#[derive(Clone, Eq, PartialEq, Hash)]
struct PointwiseRegionBuildKey {
    domain_tokens: usize,
    final_tokens: usize,
    transitions: SmallVec<[(Label, u32, usize); 4]>,
}

#[derive(Default)]
struct PointwiseRegionBuildCache {
    entries: FxHashMap<PointwiseRegionBuildKey, Option<Arc<Vec<TokenBehaviorRange>>>>,
    hits: usize,
    misses: usize,
}

struct DirectOverlayCache {
    region_values: Vec<Arc<Vec<TokenBehaviorRange>>>,
    region_ids_by_ptr: FxHashMap<usize, u32>,
    slots: Vec<Option<(u64, u32)>>,
    hits: usize,
    misses: usize,
    replacements: usize,
}

struct PointwiseRegionInterner {
    // Keep the immutable region itself as the hash key. `Arc<Vec<_>>` hashes
    // and compares by its contents, so lookup can still borrow a fresh Vec,
    // but newly interned regions no longer need a second full Vec clone solely
    // for map ownership.
    regions: FxHashMap<Arc<Vec<TokenBehaviorRange>>, ()>,
    direct_overlay: Option<DirectOverlayCache>,
}

impl Default for PointwiseRegionInterner {
    fn default() -> Self {
        let requested_slots = std::env::var("GLRMASK_WEIGHTED_MINIMIZE_DIRECT_OVERLAY_SLOTS")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(0);
        let slot_count = if requested_slots == 0 {
            0
        } else {
            requested_slots.next_power_of_two()
        };
        Self::with_direct_overlay_slots(slot_count)
    }
}

impl PointwiseRegionInterner {
    fn with_direct_overlay_slots(slot_count: usize) -> Self {
        debug_assert!(slot_count == 0 || slot_count.is_power_of_two());
        Self {
            regions: FxHashMap::default(),
            direct_overlay: (slot_count != 0).then(|| DirectOverlayCache {
                region_values: Vec::new(),
                region_ids_by_ptr: FxHashMap::default(),
                slots: vec![None; slot_count],
                hits: 0,
                misses: 0,
                replacements: 0,
            }),
        }
    }

    fn intern(&mut self, ranges: Vec<TokenBehaviorRange>) -> Arc<Vec<TokenBehaviorRange>> {
        if let Some((existing, _)) = self.regions.get_key_value(&ranges) {
            return Arc::clone(existing);
        }
        let ranges = Arc::new(ranges);
        self.regions.insert(Arc::clone(&ranges), ());
        if let Some(cache) = self.direct_overlay.as_mut() {
            let id = cache.region_values.len() as u32;
            cache
                .region_ids_by_ptr
                .insert(Arc::as_ptr(&ranges) as usize, id);
            cache.region_values.push(Arc::clone(&ranges));
        }
        ranges
    }

    fn overlay_compatible(
        &mut self,
        left: &Arc<Vec<TokenBehaviorRange>>,
        right: &Arc<Vec<TokenBehaviorRange>>,
    ) -> Arc<Vec<TokenBehaviorRange>> {
        if Arc::ptr_eq(left, right) {
            return Arc::clone(left);
        }
        let Some(cache) = self.direct_overlay.as_mut() else {
            return self.intern(overlay_compatible_token_behavior_ranges(
                left.as_ref(),
                right.as_ref(),
            ));
        };

        // Some callers merge externally constructed profile regions rather
        // than regions produced by this interner. They have no local compact
        // ID, so simply use the ordinary exact overlay path for that operation.
        // The hot minimizer path interns its regions first and still gets the
        // direct-cache fast path.
        let (Some(&left_id), Some(&right_id)) = (
            cache.region_ids_by_ptr.get(&(Arc::as_ptr(left) as usize)),
            cache.region_ids_by_ptr.get(&(Arc::as_ptr(right) as usize)),
        ) else {
            return self.intern(overlay_compatible_token_behavior_ranges(
                left.as_ref(),
                right.as_ref(),
            ));
        };
        let (low, high) = if left_id <= right_id {
            (left_id, right_id)
        } else {
            (right_id, left_id)
        };
        let key = (u64::from(low) << 32) | u64::from(high);
        let mixed = key
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .rotate_left(23)
            ^ key.rotate_right(17);
        let slot_index = mixed as usize & (cache.slots.len() - 1);
        if let Some((cached_key, result_id)) = cache.slots[slot_index] {
            if cached_key == key {
                cache.hits += 1;
                return Arc::clone(&cache.region_values[result_id as usize]);
            }
            cache.replacements += 1;
        }

        cache.misses += 1;
        let result = self.intern(overlay_compatible_token_behavior_ranges(
            left.as_ref(),
            right.as_ref(),
        ));
        let cache = self
            .direct_overlay
            .as_mut()
            .expect("direct overlay cache must remain enabled");
        let result_id = *cache
            .region_ids_by_ptr
            .get(&(Arc::as_ptr(&result) as usize))
            .expect("newly interned pointwise region must have a compact ID");
        cache.slots[slot_index] = Some((key, result_id));
        result
    }

    fn direct_overlay_stats(&self) -> (usize, usize, usize, usize) {
        self.direct_overlay.as_ref().map_or((0, 0, 0, 0), |cache| {
            (
                cache.slots.len(),
                cache.hits,
                cache.misses,
                cache.replacements,
            )
        })
    }
}

#[derive(Default)]
struct PointwiseProfile {
    /// Sorted by TSID. Each behavior region is immutable and shared by every
    /// TSID whose weight behavior is identical.
    by_tsid: Vec<(u32, Arc<Vec<TokenBehaviorRange>>)>,
}

/// One constant pointwise behavior over an inclusive TSID interval.
///
/// The existing pointwise path expands this interval into one entry per TSID.
/// That is exact but pathological when a weight is constant across thousands
/// of compacted tokenizer states. The range form preserves the same partial
/// behavior function without that expansion.
#[derive(Clone)]
struct PointwiseTsidRange {
    start: u32,
    end: u32,
    region: Arc<Vec<TokenBehaviorRange>>,
}

#[derive(Default)]
struct PointwiseRangeProfile {
    /// Sorted, disjoint TSID intervals. Adjacent intervals with the same
    /// region are coalesced.
    by_tsid_range: Vec<PointwiseTsidRange>,
}

#[derive(Default)]
struct PointwiseRangeBehaviorMap {
    /// Sorted, disjoint TSID intervals encoding the exact union behavior of a
    /// merge group.
    ranges: Vec<PointwiseTsidRange>,
}

#[derive(Default)]
struct PointwiseRangeMergeGroup {
    targets_by_label: FxHashMap<Label, u32>,
    behavior_by_tsid: PointwiseRangeBehaviorMap,
    member_classes: Vec<usize>,
}

fn pointwise_tsid_ranges_enabled() -> bool {
    std::env::var("GLRMASK_POINTWISE_TSID_RANGES")
        .map(|value| {
            let value = value.trim();
            !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
        })
        .unwrap_or(false)
}

fn pointwise_tsid_ranges_auto_enabled() -> bool {
    std::env::var("GLRMASK_POINTWISE_TSID_RANGES")
        .map(|value| value.trim().eq_ignore_ascii_case("auto"))
        .unwrap_or(false)
}

const POINTWISE_TSID_RANGE_MIN_COMPRESSION: usize = 4;

fn push_pointwise_tsid_range(
    ranges: &mut Vec<PointwiseTsidRange>,
    start: u32,
    end: u32,
    region: Arc<Vec<TokenBehaviorRange>>,
) {
    if start > end {
        return;
    }
    if let Some(previous) = ranges.last_mut() {
        if Arc::ptr_eq(&previous.region, &region)
            && previous.end != u32::MAX
            && previous.end + 1 == start
        {
            previous.end = end;
            return;
        }
    }
    ranges.push(PointwiseTsidRange { start, end, region });
}

const MAX_DENSE_POINTWISE_TSID_SLOTS: usize = 65_536;

#[derive(Clone, Copy)]
enum PointwiseBehaviorMapLayout {
    Sparse,
    Dense { slots: usize },
}

enum PointwiseBehaviorMap {
    Sparse(FxHashMap<u32, Arc<Vec<TokenBehaviorRange>>>),
    Dense(Vec<Option<Arc<Vec<TokenBehaviorRange>>>>),
}

impl Default for PointwiseBehaviorMap {
    fn default() -> Self {
        Self::Sparse(FxHashMap::default())
    }
}

impl PointwiseBehaviorMap {
    fn new(layout: PointwiseBehaviorMapLayout) -> Self {
        match layout {
            PointwiseBehaviorMapLayout::Sparse => Self::Sparse(FxHashMap::default()),
            PointwiseBehaviorMapLayout::Dense { slots } => Self::Dense(vec![None; slots]),
        }
    }

    fn get(&self, tsid: u32) -> Option<&Arc<Vec<TokenBehaviorRange>>> {
        match self {
            Self::Sparse(entries) => entries.get(&tsid),
            Self::Dense(entries) => entries.get(tsid as usize).and_then(Option::as_ref),
        }
    }

    fn merge_profile(
        &mut self,
        profile: &PointwiseProfile,
        regions: &mut PointwiseRegionInterner,
    ) {
        for (tsid, ranges) in &profile.by_tsid {
            match self {
                Self::Sparse(entries) => match entries.get_mut(tsid) {
                    Some(existing) if Arc::ptr_eq(existing, ranges) => {}
                    Some(existing) => {
                        *existing = regions.overlay_compatible(existing, ranges);
                    }
                    None => {
                        entries.insert(*tsid, Arc::clone(ranges));
                    }
                },
                Self::Dense(entries) => {
                    let entry = entries
                        .get_mut(*tsid as usize)
                        .expect("dense pointwise behavior map must cover every profile TSID");
                    match entry {
                        Some(existing) if Arc::ptr_eq(existing, ranges) => {}
                        Some(existing) => {
                            *existing = regions.overlay_compatible(existing, ranges);
                        }
                        None => {
                            *entry = Some(Arc::clone(ranges));
                        }
                    }
                }
            }
        }
    }

    fn region_entry_count(&self) -> usize {
        match self {
            Self::Sparse(entries) => entries.values().map(|ranges| ranges.len()).sum(),
            Self::Dense(entries) => entries
                .iter()
                .flatten()
                .map(|ranges| ranges.len())
                .sum(),
        }
    }
}

impl PointwiseRangeBehaviorMap {
    fn profile_compatible(&self, profile: &PointwiseRangeProfile) -> bool {
        let mut existing_index = 0usize;
        let mut profile_index = 0usize;
        while existing_index < self.ranges.len() && profile_index < profile.by_tsid_range.len() {
            let existing = &self.ranges[existing_index];
            let add = &profile.by_tsid_range[profile_index];
            if existing.end < add.start {
                existing_index += 1;
                continue;
            }
            if add.end < existing.start {
                profile_index += 1;
                continue;
            }
            if !Arc::ptr_eq(&existing.region, &add.region)
                && !token_behavior_ranges_compatible(existing.region.as_ref(), add.region.as_ref())
            {
                return false;
            }
            if existing.end <= add.end {
                existing_index += 1;
            }
            if add.end <= existing.end {
                profile_index += 1;
            }
        }
        true
    }

    fn merge_profile(
        &mut self,
        profile: &PointwiseRangeProfile,
        regions: &mut PointwiseRegionInterner,
    ) {
        let mut merged = Vec::with_capacity(self.ranges.len() + profile.by_tsid_range.len());
        let mut existing_index = 0usize;
        let mut profile_index = 0usize;
        let mut existing = self.ranges.get(existing_index).cloned();
        let mut add = profile.by_tsid_range.get(profile_index).cloned();

        loop {
            match (existing.take(), add.take()) {
                (None, None) => break,
                (Some(range), None) => {
                    push_pointwise_tsid_range(&mut merged, range.start, range.end, range.region);
                    existing_index += 1;
                    existing = self.ranges.get(existing_index).cloned();
                }
                (None, Some(range)) => {
                    push_pointwise_tsid_range(&mut merged, range.start, range.end, range.region);
                    profile_index += 1;
                    add = profile.by_tsid_range.get(profile_index).cloned();
                }
                (Some(mut left), Some(mut right)) => {
                    if left.end < right.start {
                        push_pointwise_tsid_range(&mut merged, left.start, left.end, left.region);
                        existing_index += 1;
                        existing = self.ranges.get(existing_index).cloned();
                        add = Some(right);
                        continue;
                    }
                    if right.end < left.start {
                        push_pointwise_tsid_range(&mut merged, right.start, right.end, right.region);
                        profile_index += 1;
                        existing = Some(left);
                        add = profile.by_tsid_range.get(profile_index).cloned();
                        continue;
                    }

                    if left.start < right.start {
                        push_pointwise_tsid_range(
                            &mut merged,
                            left.start,
                            right.start - 1,
                            Arc::clone(&left.region),
                        );
                        left.start = right.start;
                    } else if right.start < left.start {
                        push_pointwise_tsid_range(
                            &mut merged,
                            right.start,
                            left.start - 1,
                            Arc::clone(&right.region),
                        );
                        right.start = left.start;
                    }

                    debug_assert!(
                        Arc::ptr_eq(&left.region, &right.region)
                            || token_behavior_ranges_compatible(
                                left.region.as_ref(),
                                right.region.as_ref(),
                            ),
                        "range profiles must be checked for compatibility before merging",
                    );
                    let end = left.end.min(right.end);
                    let region = if Arc::ptr_eq(&left.region, &right.region) {
                        left.region.clone()
                    } else {
                        regions.overlay_compatible(&left.region, &right.region)
                    };
                    push_pointwise_tsid_range(&mut merged, left.start, end, region);

                    if left.end == end {
                        existing_index += 1;
                        existing = self.ranges.get(existing_index).cloned();
                    } else {
                        left.start = end + 1;
                        existing = Some(left);
                    }
                    if right.end == end {
                        profile_index += 1;
                        add = profile.by_tsid_range.get(profile_index).cloned();
                    } else {
                        right.start = end + 1;
                        add = Some(right);
                    }
                }
            }
        }
        self.ranges = merged;
    }

    fn region_entry_count(&self) -> usize {
        self.ranges.iter().map(|range| range.region.len()).sum()
    }
}

#[derive(Default)]
struct PointwiseMergeGroup {
    targets_by_label: FxHashMap<Label, u32>,
    /// Exact partial behavior function of all members already in this group.
    behavior_by_tsid: PointwiseBehaviorMap,
    member_classes: Vec<usize>,
}

struct HybridColoring {
    coloring: Vec<usize>,
    direct_builders: Option<Vec<MergedStateBuilder>>,
}

impl HybridColoring {
    fn ordinary(coloring: Vec<usize>) -> Self {
        Self {
            coloring,
            direct_builders: None,
        }
    }
}

fn range_set_contains(tokens: &RangeSetBlaze<u32>, value: u32) -> bool {
    tokens.contains(value)
}

fn profile_target_for_label(profile: &ClassProfile, label: Label) -> Option<u32> {
    profile
        .targets
        .binary_search_by_key(&label, |(candidate, _)| *candidate)
        .ok()
        .map(|index| profile.targets[index].1)
}

fn push_token_behavior_range(
    ranges: &mut Vec<TokenBehaviorRange>,
    start: u32,
    end: u32,
    behavior: u32,
) {
    if start > end {
        return;
    }
    if let Some(previous) = ranges.last_mut() {
        if previous.behavior == behavior
            && previous.end != u32::MAX
            && previous.end + 1 == start
        {
            previous.end = end;
            return;
        }
    }
    ranges.push(TokenBehaviorRange { start, end, behavior });
}

fn add_tsid_boundary_if_overlapping(
    boundaries: &mut Vec<u64>,
    domain_start: u32,
    domain_end: u32,
    range_start: u32,
    range_end: u32,
) {
    let start = domain_start.max(range_start);
    let end = domain_end.min(range_end);
    if start <= end {
        boundaries.push(u64::from(start));
        boundaries.push(u64::from(end) + 1);
    }
}

fn build_token_behavior_region(
    domain_tokens: &RangeSetBlaze<u32>,
    final_tokens: Option<&RangeSetBlaze<u32>>,
    active_transitions: &[(Label, u32, &RangeSetBlaze<u32>)],
    behaviors: &mut PointwiseBehaviorInterner,
    regions: &mut PointwiseRegionInterner,
    build_cache: &mut PointwiseRegionBuildCache,
) -> Option<Arc<Vec<TokenBehaviorRange>>> {
    // Token sets are immutable and retained by the DWA/needed weights for this
    // whole coloring pass, so their addresses form an exact local cache key.
    let key = PointwiseRegionBuildKey {
        domain_tokens: domain_tokens as *const RangeSetBlaze<u32> as usize,
        final_tokens: final_tokens
            .map(|tokens| tokens as *const RangeSetBlaze<u32> as usize)
            .unwrap_or(0),
        transitions: active_transitions
            .iter()
            .map(|(label, target, tokens)| {
                (*label, *target, *tokens as *const RangeSetBlaze<u32> as usize)
            })
            .collect(),
    };
    if let Some(existing) = build_cache.entries.get(&key) {
        build_cache.hits += 1;
        return existing.clone();
    }
    build_cache.misses += 1;
    let mut boundaries = Vec::<u64>::new();
    for range in domain_tokens.ranges() {
        boundaries.push(u64::from(*range.start()));
        boundaries.push(u64::from(*range.end()) + 1);
    }
    if let Some(tokens) = final_tokens {
        for range in tokens.ranges() {
            boundaries.push(u64::from(*range.start()));
            boundaries.push(u64::from(*range.end()) + 1);
        }
    }
    for (_, _, tokens) in active_transitions {
        for range in tokens.ranges() {
            boundaries.push(u64::from(*range.start()));
            boundaries.push(u64::from(*range.end()) + 1);
        }
    }
    boundaries.sort_unstable();
    boundaries.dedup();

    let mut token_ranges = Vec::new();
    for pair in boundaries.windows(2) {
        let start64 = pair[0];
        let next = pair[1];
        if start64 > u64::from(u32::MAX) || start64 >= next {
            continue;
        }
        let start = start64 as u32;
        let end = (next - 1) as u32;
        if !range_set_contains(domain_tokens, start) {
            continue;
        }
        let final_active = final_tokens.is_some_and(|tokens| range_set_contains(tokens, start));
        let mut transitions = Vec::new();
        for (label, target, tokens) in active_transitions {
            if range_set_contains(tokens, start) {
                transitions.push((*label, *target));
            }
        }
        debug_assert!(final_active || !transitions.is_empty());
        if !final_active && transitions.is_empty() {
            return None;
        }
        let behavior = behaviors.intern(final_active, transitions);
        push_token_behavior_range(&mut token_ranges, start, end, behavior);
    }
    let result = (!token_ranges.is_empty()).then(|| regions.intern(token_ranges));
    build_cache.entries.insert(key, result.clone());
    result
}

#[inline]
fn pointwise_cursor_at<'a, I>(
    iter: &mut std::iter::Peekable<I>,
    pos: u32,
    window_end_exclusive: u64,
) -> (Option<&'a RangeSetBlaze<u32>>, u64)
where
    I: Iterator<Item = (std::ops::RangeInclusive<u32>, &'a Arc<RangeSetBlaze<u32>>)>,
{
    while iter
        .peek()
        .is_some_and(|(range, _)| *range.end() < pos)
    {
        iter.next();
    }
    let Some((range, tokens)) = iter.peek() else {
        return (None, window_end_exclusive);
    };
    let start = *range.start();
    let end = *range.end();
    if start <= pos {
        let shared: &'a Arc<RangeSetBlaze<u32>> = *tokens;
        (
            Some(shared.as_ref()),
            (u64::from(end) + 1).min(window_end_exclusive),
        )
    } else {
        (None, u64::from(start).min(window_end_exclusive))
    }
}

/// Exact k-way monotone sweep over the sorted outer TSID ranges. Unlike the
/// boundary-materializing path, this never copies source range lists, clips
/// them into temporary vectors, or sorts/deduplicates change points.
fn build_pointwise_profile_monotone(
    domain: &Weight,
    profile: &ClassProfile,
    behaviors: &mut PointwiseBehaviorInterner,
    regions: &mut PointwiseRegionInterner,
    build_cache: &mut PointwiseRegionBuildCache,
) -> Option<PointwiseProfile> {
    if domain.is_full()
        || profile.final_weight.as_ref().is_some_and(Weight::is_full)
        || profile.weights.iter().any(|(_, weight)| weight.is_full())
    {
        return None;
    }

    let transitions: SmallVec<[(Label, u32, &Weight); 4]> = profile
        .weights
        .iter()
        .map(|(label, weight)| Some((*label, profile_target_for_label(profile, *label)?, weight)))
        .collect::<Option<_>>()?;
    let mut final_iter = profile
        .final_weight
        .as_ref()
        .map(|weight| weight.raw_range_values().peekable());
    let mut transition_iters: SmallVec<[_; 4]> = transitions
        .iter()
        .map(|(_, _, weight)| weight.raw_range_values().peekable())
        .collect();

    let mut by_tsid = Vec::new();
    for (domain_start, domain_end, domain_tokens) in domain.range_entries() {
        let window_end_exclusive = u64::from(domain_end) + 1;
        let mut pos = domain_start;
        loop {
            let (final_tokens, mut next_change) = if let Some(iter) = final_iter.as_mut() {
                pointwise_cursor_at(iter, pos, window_end_exclusive)
            } else {
                (None, window_end_exclusive)
            };
            let mut active_transitions: SmallVec<[(Label, u32, &RangeSetBlaze<u32>); 4]> =
                SmallVec::new();
            for (index, iter) in transition_iters.iter_mut().enumerate() {
                let (tokens, transition_change) =
                    pointwise_cursor_at(iter, pos, window_end_exclusive);
                next_change = next_change.min(transition_change);
                if let Some(tokens) = tokens {
                    let (label, target, _) = transitions[index];
                    active_transitions.push((label, target, tokens));
                }
            }
            debug_assert!(next_change > u64::from(pos));
            if next_change <= u64::from(pos) {
                return None;
            }
            let interval_end = (next_change - 1) as u32;
            let region = build_token_behavior_region(
                domain_tokens.as_ref(),
                final_tokens,
                active_transitions.as_slice(),
                behaviors,
                regions,
                build_cache,
            )?;
            for tsid in pos..=interval_end {
                by_tsid.push((tsid, Arc::clone(&region)));
            }
            if next_change >= window_end_exclusive {
                break;
            }
            pos = next_change as u32;
        }
    }
    Some(PointwiseProfile { by_tsid })
}

/// Materialize a complete observable behavior function. A profile is constant
/// over large TSID intervals, so build its token behavior once per interval
/// then share that immutable region for every TSID in the interval.
fn build_pointwise_profile(
    domain: &Weight,
    profile: &ClassProfile,
    behaviors: &mut PointwiseBehaviorInterner,
    regions: &mut PointwiseRegionInterner,
    build_cache: &mut PointwiseRegionBuildCache,
    monotone_pointwise: bool,
) -> Option<PointwiseProfile> {
    if monotone_pointwise {
        return build_pointwise_profile_monotone(
            domain, profile, behaviors, regions, build_cache,
        );
    }
    if domain.is_full()
        || profile.final_weight.as_ref().is_some_and(Weight::is_full)
        || profile.weights.iter().any(|(_, weight)| weight.is_full())
    {
        return None;
    }

    let transitions: SmallVec<[(Label, u32, &Weight); 4]> = profile
        .weights
        .iter()
        .map(|(label, weight)| Some((*label, profile_target_for_label(profile, *label)?, weight)))
        .collect::<Option<_>>()?;

    // The domain is fragmented into many small compact windows. Collect each
    // source's full sorted TSID range list ONCE up front; re-scanning every
    // transition's complete range list per window would be O(windows * ranges).
    // Monotone cursors below then advance forward as windows advance, making the
    // per-window range gathering O(ranges + overlaps) overall.
    let final_full: SmallVec<[(u32, u32, &RangeSetBlaze<u32>); 8]> = profile
        .final_weight
        .as_ref()
        .map(|fw| {
            fw.raw_range_values()
                .map(|(r, tokens)| (*r.start(), *r.end(), tokens.as_ref()))
                .collect()
        })
        .unwrap_or_default();
    let transition_full: SmallVec<
        [SmallVec<[(u32, u32, &RangeSetBlaze<u32>); 8]>; 4],
    > = transitions
        .iter()
        .map(|(_, _, weight)| {
            weight
                .raw_range_values()
                .map(|(r, tokens)| (*r.start(), *r.end(), tokens.as_ref()))
                .collect()
        })
        .collect();

    let mut by_tsid = Vec::new();
    let mut final_scan = 0usize;
    let mut transition_scan: SmallVec<[usize; 4]> =
        std::iter::repeat_n(0usize, transitions.len()).collect();
    for (domain_start, domain_end, domain_tokens) in domain.range_entries() {
        // Build the TSID boundaries and, in the same pass, collect each source's
        // token-set ranges clipped to the domain window. The clipped range lists
        // let a coordinated sweep read the active token set per interval with a
        // monotone cursor instead of a binary-search `get` per interval per
        // transition, which is the dominant color-step cost.
        let mut boundaries: SmallVec<[u64; 32]> = SmallVec::new();
        boundaries.push(u64::from(domain_start));
        boundaries.push(u64::from(domain_end) + 1);
        let mut final_ranges: SmallVec<[(u32, u32, &RangeSetBlaze<u32>); 8]> = SmallVec::new();
        // Advance the monotone scan cursor past ranges ending before this window;
        // windows are visited left to right so earlier ranges are never revisited.
        while final_scan < final_full.len() && final_full[final_scan].1 < domain_start {
            final_scan += 1;
        }
        let mut peek = final_scan;
        while peek < final_full.len() && final_full[peek].0 <= domain_end {
            let (range_start, range_end, tokens) = final_full[peek];
            let start = domain_start.max(range_start);
            let end = domain_end.min(range_end);
            if start <= end {
                boundaries.push(u64::from(start));
                boundaries.push(u64::from(end) + 1);
                final_ranges.push((start, end, tokens));
            }
            peek += 1;
        }
        let mut transition_ranges: SmallVec<
            [SmallVec<[(u32, u32, &RangeSetBlaze<u32>); 8]>; 4],
        > = SmallVec::new();
        transition_ranges.reserve(transitions.len());
        for (index, full) in transition_full.iter().enumerate() {
            let cursor = &mut transition_scan[index];
            while *cursor < full.len() && full[*cursor].1 < domain_start {
                *cursor += 1;
            }
            let mut clipped: SmallVec<[(u32, u32, &RangeSetBlaze<u32>); 8]> = SmallVec::new();
            let mut peek = *cursor;
            while peek < full.len() && full[peek].0 <= domain_end {
                let (range_start, range_end, tokens) = full[peek];
                let start = domain_start.max(range_start);
                let end = domain_end.min(range_end);
                if start <= end {
                    boundaries.push(u64::from(start));
                    boundaries.push(u64::from(end) + 1);
                    clipped.push((start, end, tokens));
                }
                peek += 1;
            }
            transition_ranges.push(clipped);
        }
        boundaries.sort_unstable();
        boundaries.dedup();

        // Only transitions with at least one clipped range in this window can
        // ever be active in an interval; skipping the empties keeps the sweep
        // inner loop proportional to the live transitions rather than the full
        // transition count. Indices stay ascending so push order is unchanged.
        let active_indices: SmallVec<[usize; 4]> = transition_ranges
            .iter()
            .enumerate()
            .filter(|(_, ranges)| !ranges.is_empty())
            .map(|(index, _)| index)
            .collect();

        // Monotone cursors: TSID intervals are visited left to right, and every
        // clipped range list is sorted, so each cursor only advances forward.
        let mut final_cursor = 0usize;
        let mut transition_cursors: SmallVec<[usize; 4]> =
            std::iter::repeat_n(0usize, transitions.len()).collect();
        for pair in boundaries.windows(2) {
            let start64 = pair[0];
            let next = pair[1];
            if start64 > u64::from(u32::MAX) || start64 >= next {
                continue;
            }
            let tsid_start = start64 as u32;
            let tsid_end = (next - 1) as u32;
            while final_cursor < final_ranges.len() && final_ranges[final_cursor].1 < tsid_start {
                final_cursor += 1;
            }
            let final_tokens = final_ranges
                .get(final_cursor)
                .filter(|(start, _, _)| *start <= tsid_start)
                .map(|(_, _, tokens)| *tokens);
            let mut active_transitions: SmallVec<[(Label, u32, &RangeSetBlaze<u32>); 4]> =
                SmallVec::new();
            for &index in &active_indices {
                let (label, target, _) = &transitions[index];
                let ranges = &transition_ranges[index];
                let cursor = &mut transition_cursors[index];
                while *cursor < ranges.len() && ranges[*cursor].1 < tsid_start {
                    *cursor += 1;
                }
                if let Some((start, _, tokens)) = ranges.get(*cursor) {
                    if *start <= tsid_start {
                        active_transitions.push((*label, *target, *tokens));
                    }
                }
            }
            let region = build_token_behavior_region(
                domain_tokens.as_ref(),
                final_tokens,
                active_transitions.as_slice(),
                behaviors,
                regions,
                build_cache,
            )?;
            for tsid in tsid_start..=tsid_end {
                by_tsid.push((tsid, Arc::clone(&region)));
            }
        }
    }
    Some(PointwiseProfile { by_tsid })
}

/// Range-compressed equivalent of [`build_pointwise_profile`].
///
/// Every point in one emitted TSID interval observes the same final and
/// transition token sets, so the token behavior region is constant throughout
/// that interval. Keeping the interval intact is exact.
fn build_pointwise_range_profile(
    domain: &Weight,
    profile: &ClassProfile,
    behaviors: &mut PointwiseBehaviorInterner,
    regions: &mut PointwiseRegionInterner,
    build_cache: &mut PointwiseRegionBuildCache,
) -> Option<PointwiseRangeProfile> {
    if domain.is_full()
        || profile.final_weight.as_ref().is_some_and(Weight::is_full)
        || profile.weights.iter().any(|(_, weight)| weight.is_full())
    {
        return None;
    }

    let transitions: Vec<(Label, u32, &Weight)> = profile
        .weights
        .iter()
        .map(|(label, weight)| Some((*label, profile_target_for_label(profile, *label)?, weight)))
        .collect::<Option<_>>()?;
    let mut by_tsid_range = Vec::new();
    for (domain_start, domain_end, domain_tokens) in domain.compact_entries()? {
        let mut boundaries = vec![u64::from(domain_start), u64::from(domain_end) + 1];
        if let Some(final_weight) = &profile.final_weight {
            for (tsid_range, _) in final_weight.raw_range_values() {
                add_tsid_boundary_if_overlapping(
                    &mut boundaries,
                    domain_start,
                    domain_end,
                    *tsid_range.start(),
                    *tsid_range.end(),
                );
            }
        }
        for (_, _, weight) in &transitions {
            for (tsid_range, _) in weight.raw_range_values() {
                add_tsid_boundary_if_overlapping(
                    &mut boundaries,
                    domain_start,
                    domain_end,
                    *tsid_range.start(),
                    *tsid_range.end(),
                );
            }
        }
        boundaries.sort_unstable();
        boundaries.dedup();

        for pair in boundaries.windows(2) {
            let start64 = pair[0];
            let next = pair[1];
            if start64 > u64::from(u32::MAX) || start64 >= next {
                continue;
            }
            let tsid_start = start64 as u32;
            let tsid_end = (next - 1) as u32;
            let final_tokens = profile
                .final_weight
                .as_ref()
                .and_then(|weight| weight.token_set_for_tsid_ref(tsid_start))
                .map(|tokens| tokens.as_ref());
            let mut active_transitions = Vec::new();
            for (label, target, weight) in &transitions {
                if let Some(tokens) = weight.token_set_for_tsid_ref(tsid_start) {
                    active_transitions.push((*label, *target, tokens.as_ref()));
                }
            }
            let region = build_token_behavior_region(
                domain_tokens.as_ref(),
                final_tokens,
                &active_transitions,
                behaviors,
                regions,
                build_cache,
            )?;
            push_pointwise_tsid_range(&mut by_tsid_range, tsid_start, tsid_end, region);
        }
    }
    Some(PointwiseRangeProfile { by_tsid_range })
}

fn token_behavior_ranges_compatible(
    left: &[TokenBehaviorRange],
    right: &[TokenBehaviorRange],
) -> bool {
    let mut left_index = 0usize;
    let mut right_index = 0usize;
    while left_index < left.len() && right_index < right.len() {
        let left_range = left[left_index];
        let right_range = right[right_index];
        if left_range.end < right_range.start {
            left_index += 1;
            continue;
        }
        if right_range.end < left_range.start {
            right_index += 1;
            continue;
        }
        if left_range.behavior != right_range.behavior {
            return false;
        }
        if left_range.end <= right_range.end {
            left_index += 1;
        }
        if right_range.end <= left_range.end {
            right_index += 1;
        }
    }
    true
}

/// Overlay two compatible sparse functions. Equal behavior is required where
/// both inputs are defined; the output preserves their exact union.
fn overlay_compatible_token_behavior_ranges(
    existing: &[TokenBehaviorRange],
    add: &[TokenBehaviorRange],
) -> Vec<TokenBehaviorRange> {
    let mut result = Vec::with_capacity(existing.len() + add.len());
    let mut existing_index = 0usize;
    let mut add_index = 0usize;
    let mut current_existing = existing.get(existing_index).copied();
    let mut current_add = add.get(add_index).copied();

    loop {
        match (current_existing, current_add) {
            (None, None) => break,
            (Some(range), None) => {
                push_token_behavior_range(&mut result, range.start, range.end, range.behavior);
                existing_index += 1;
                current_existing = existing.get(existing_index).copied();
            }
            (None, Some(range)) => {
                push_token_behavior_range(&mut result, range.start, range.end, range.behavior);
                add_index += 1;
                current_add = add.get(add_index).copied();
            }
            (Some(mut left), Some(mut right)) => {
                if left.end < right.start {
                    push_token_behavior_range(&mut result, left.start, left.end, left.behavior);
                    existing_index += 1;
                    current_existing = existing.get(existing_index).copied();
                    continue;
                }
                if right.end < left.start {
                    push_token_behavior_range(&mut result, right.start, right.end, right.behavior);
                    add_index += 1;
                    current_add = add.get(add_index).copied();
                    continue;
                }

                debug_assert_eq!(left.behavior, right.behavior);
                if left.start < right.start {
                    push_token_behavior_range(&mut result, left.start, right.start - 1, left.behavior);
                    left.start = right.start;
                } else if right.start < left.start {
                    push_token_behavior_range(&mut result, right.start, left.start - 1, right.behavior);
                    right.start = left.start;
                }

                let end = left.end.min(right.end);
                push_token_behavior_range(&mut result, left.start, end, left.behavior);
                if left.end == end {
                    existing_index += 1;
                    current_existing = existing.get(existing_index).copied();
                } else {
                    left.start = end + 1;
                    current_existing = Some(left);
                }
                if right.end == end {
                    add_index += 1;
                    current_add = add.get(add_index).copied();
                } else {
                    right.start = end + 1;
                    current_add = Some(right);
                }
            }
        }
    }
    result
}

fn pointwise_profile_compatible(group: &PointwiseMergeGroup, profile: &PointwiseProfile) -> bool {
    profile.by_tsid.iter().all(|(tsid, ranges)| {
        group.behavior_by_tsid.get(*tsid).is_none_or(|existing| {
            Arc::ptr_eq(existing, ranges)
                || token_behavior_ranges_compatible(existing.as_ref(), ranges.as_ref())
        })
    })
}

fn merge_pointwise_profile_into_group(
    group: &mut PointwiseMergeGroup,
    profile: &PointwiseProfile,
    regions: &mut PointwiseRegionInterner,
) {
    group.behavior_by_tsid.merge_profile(profile, regions);
}
/// Exact greedy grouping using one sparse partial behavior function per group.
/// The class order and target-map restriction are identical to the memberwise
/// path; only the witness representation changes.
fn try_build_and_color_pointwise(
    candidates: &[usize],
    class_coloring: &[usize],
    class_needed_union: &[Weight],
    class_profiles: &[ClassProfile],
    pointwise_class_order: PointwiseClassOrder,
    profile_enabled: bool,
    monotone_pointwise: bool,
    direct_overlay_slots: usize,
) -> Option<HybridColoring> {
    let started_at = Instant::now();
    let mut interner = PointwiseBehaviorInterner::default();
    let mut regions = PointwiseRegionInterner::with_direct_overlay_slots(direct_overlay_slots);
    let mut region_build_cache = PointwiseRegionBuildCache::default();
    let profile_started_at = Instant::now();
    let mut pointwise_profiles = Vec::with_capacity(class_profiles.len());
    for (domain, profile) in class_needed_union.iter().zip(class_profiles) {
        pointwise_profiles.push(build_pointwise_profile(
            domain,
            profile,
            &mut interner,
            &mut regions,
            &mut region_build_cache,
            monotone_pointwise,
        )?);
    }
    let profile_build_ms = profile_started_at.elapsed().as_secs_f64() * 1000.0;

    let profile_entries = pointwise_profiles
        .iter()
        .map(|profile| profile.by_tsid.len())
        .sum::<usize>();
    let max_tsid = pointwise_profiles
        .iter()
        .flat_map(|profile| profile.by_tsid.iter().map(|(tsid, _)| *tsid))
        .max();
    let behavior_map_layout = max_tsid
        .and_then(|max_tsid| usize::try_from(max_tsid).ok()?.checked_add(1))
        .filter(|&slots| slots <= MAX_DENSE_POINTWISE_TSID_SLOTS && slots <= profile_entries)
        .map(|slots| PointwiseBehaviorMapLayout::Dense { slots })
        .unwrap_or(PointwiseBehaviorMapLayout::Sparse);

    let merge_started_at = Instant::now();
    let mut groups = Vec::<PointwiseMergeGroup>::new();
    let mut group_attempts = 0usize;
    let mut target_rejects = 0usize;
    let mut behavior_rejects = 0usize;
    let mut class_order = (0..class_profiles.len()).collect::<Vec<_>>();
    if pointwise_class_order == PointwiseClassOrder::DescendingDomain {
        class_order.sort_unstable_by_key(|&class| {
            std::cmp::Reverse((
                pointwise_profiles[class].by_tsid.len(),
                class_profiles[class].targets.len(),
                class,
            ))
        });
    }
    for class in class_order {
        let class_profile = &class_profiles[class];
        let pointwise_profile = &pointwise_profiles[class];
        let mut placed = false;
        for group in &mut groups {
            group_attempts += 1;
            if !targets_compatible_with_group_map(&class_profile.targets, &group.targets_by_label) {
                target_rejects += 1;
                continue;
            }
            if !pointwise_profile_compatible(group, pointwise_profile) {
                behavior_rejects += 1;
                continue;
            }
            #[cfg(debug_assertions)]
            debug_assert!(memberwise_group_compatible(
                &class_needed_union[class],
                class_profile,
                &group.member_classes,
                class_needed_union,
                class_profiles,
            ));
            for (label, target) in &class_profile.targets {
                group.targets_by_label.entry(*label).or_insert(*target);
            }
            merge_pointwise_profile_into_group(group, pointwise_profile, &mut regions);
            group.member_classes.push(class);
            placed = true;
            break;
        }
        if !placed {
            let mut targets_by_label = FxHashMap::default();
            targets_by_label.reserve(class_profile.targets.len());
            for (label, target) in &class_profile.targets {
                targets_by_label.insert(*label, *target);
            }
            let mut group = PointwiseMergeGroup {
                targets_by_label,
                behavior_by_tsid: PointwiseBehaviorMap::new(behavior_map_layout),
                member_classes: vec![class],
            };
            merge_pointwise_profile_into_group(&mut group, pointwise_profile, &mut regions);
            groups.push(group);
        }
    }
    let merge_ms = merge_started_at.elapsed().as_secs_f64() * 1000.0;

    let mut class_to_group = vec![0usize; class_profiles.len()];
    for (group_id, group) in groups.iter().enumerate() {
        for &class in &group.member_classes {
            class_to_group[class] = group_id;
        }
    }
    let coloring = class_coloring
        .iter()
        .map(|class| class_to_group[*class])
        .collect();

    if profile_enabled {
        let (direct_overlay_slots, direct_overlay_hits, direct_overlay_misses, direct_overlay_replacements) =
            regions.direct_overlay_stats();
        let region_entries = groups
            .iter()
            .map(|group| {
                group.behavior_by_tsid.region_entry_count()
            })
            .sum::<usize>();
        eprintln!(
            "[glrmask/profile][weighted_dwa_minimize_pointwise] candidates={} classes={} groups={} behaviors={} interned_regions={} regions={} region_build_cache_entries={} region_build_cache_hits={} region_build_cache_misses={} direct_overlay_slots={} direct_overlay_hits={} direct_overlay_misses={} direct_overlay_replacements={} profile_build_ms={:.3} merge_ms={:.3} total_ms={:.3} group_attempts={} target_rejects={} behavior_rejects={}",
            candidates.len(),
            class_profiles.len(),
            groups.len(),
            interner.ids.len(),
            regions.regions.len(),
            region_entries,
            region_build_cache.entries.len(),
            region_build_cache.hits,
            region_build_cache.misses,
            direct_overlay_slots,
            direct_overlay_hits,
            direct_overlay_misses,
            direct_overlay_replacements,
            profile_build_ms,
            merge_ms,
            started_at.elapsed().as_secs_f64() * 1000.0,
            group_attempts,
            target_rejects,
            behavior_rejects,
        );
    }
    // Pointwise coloring has already materialized the exact union behavior of
    // every merge group. Reusing that summary is cheaper than walking every
    // member state again and rebuilding the same final/transition unions. Keep
    // the historical environment variable as a diagnostic kill switch, but
    // make the proven direct reconstruction path the production default.
    let direct_reconstruct = std::env::var("GLRMASK_WEIGHTED_MINIMIZE_DIRECT_POINTWISE_RECONSTRUCT")
        .ok()
        .map(|value| {
            let value = value.trim();
            !matches!(value.to_ascii_lowercase().as_str(), "0" | "false" | "no" | "off")
        })
        .unwrap_or(true);
    let direct_builders =
        direct_reconstruct.then(|| pointwise_groups_to_builders(&groups, &interner));
    Some(HybridColoring {
        coloring,
        direct_builders,
    })
}

/// Exact greedy pointwise coloring using TSID intervals rather than one map
/// entry per TSID. This is equivalent to [`try_build_and_color_pointwise`]; it
/// changes only the representation of a partial behavior function.
fn try_build_and_color_pointwise_ranges(
    candidates: &[usize],
    class_coloring: &[usize],
    class_needed_union: &[Weight],
    class_profiles: &[ClassProfile],
    pointwise_class_order: PointwiseClassOrder,
    profile_enabled: bool,
    monotone_pointwise: bool,
) -> Option<Vec<usize>> {
    let started_at = Instant::now();
    let mut interner = PointwiseBehaviorInterner::default();
    let mut regions = PointwiseRegionInterner::default();
    let mut region_build_cache = PointwiseRegionBuildCache::default();
    let profile_started_at = Instant::now();
    let mut pointwise_profiles = Vec::with_capacity(class_profiles.len());
    for (domain, profile) in class_needed_union.iter().zip(class_profiles) {
        pointwise_profiles.push(build_pointwise_range_profile(
            domain,
            profile,
            &mut interner,
            &mut regions,
            &mut region_build_cache,
        )?);
    }
    let profile_build_ms = profile_started_at.elapsed().as_secs_f64() * 1000.0;

    let profile_ranges = pointwise_profiles
        .iter()
        .map(|profile| profile.by_tsid_range.len())
        .sum::<usize>();
    let profile_tsid_cells = pointwise_profiles
        .iter()
        .flat_map(|profile| profile.by_tsid_range.iter())
        .map(|range| (range.end as usize - range.start as usize) + 1)
        .sum::<usize>();

    if pointwise_tsid_ranges_auto_enabled()
        && profile_ranges.saturating_mul(POINTWISE_TSID_RANGE_MIN_COMPRESSION)
            > profile_tsid_cells
    {
        if profile_enabled {
            eprintln!(
                "[glrmask/profile][weighted_dwa_minimize_pointwise_ranges_fallback] candidates={} classes={} profile_ranges={} profile_tsid_cells={} profile_build_ms={:.3}",
                candidates.len(),
                class_profiles.len(),
                profile_ranges,
                profile_tsid_cells,
                profile_build_ms,
            );
        }
        return try_build_and_color_pointwise(
            candidates,
            class_coloring,
            class_needed_union,
            class_profiles,
            pointwise_class_order,
            profile_enabled,
            monotone_pointwise,
            0,
        )
        .map(|result| result.coloring);
    }

    let merge_started_at = Instant::now();
    let mut groups = Vec::<PointwiseRangeMergeGroup>::new();
    let mut group_attempts = 0usize;
    let mut target_rejects = 0usize;
    let mut behavior_rejects = 0usize;
    let mut class_order = (0..class_profiles.len()).collect::<Vec<_>>();
    if pointwise_class_order == PointwiseClassOrder::DescendingDomain {
        class_order.sort_unstable_by_key(|&class| {
            std::cmp::Reverse((
                pointwise_profiles[class].by_tsid_range.len(),
                class_profiles[class].targets.len(),
                class,
            ))
        });
    }
    for class in class_order {
        let class_profile = &class_profiles[class];
        let pointwise_profile = &pointwise_profiles[class];
        let mut placed = false;
        for group in &mut groups {
            group_attempts += 1;
            if !targets_compatible_with_group_map(&class_profile.targets, &group.targets_by_label) {
                target_rejects += 1;
                continue;
            }
            if !group.behavior_by_tsid.profile_compatible(pointwise_profile) {
                behavior_rejects += 1;
                continue;
            }
            #[cfg(debug_assertions)]
            debug_assert!(memberwise_group_compatible(
                &class_needed_union[class],
                class_profile,
                &group.member_classes,
                class_needed_union,
                class_profiles,
            ));
            for (label, target) in &class_profile.targets {
                group.targets_by_label.entry(*label).or_insert(*target);
            }
            group
                .behavior_by_tsid
                .merge_profile(pointwise_profile, &mut regions);
            group.member_classes.push(class);
            placed = true;
            break;
        }
        if !placed {
            let mut targets_by_label = FxHashMap::default();
            targets_by_label.reserve(class_profile.targets.len());
            for (label, target) in &class_profile.targets {
                targets_by_label.insert(*label, *target);
            }
            let mut group = PointwiseRangeMergeGroup {
                targets_by_label,
                behavior_by_tsid: PointwiseRangeBehaviorMap::default(),
                member_classes: vec![class],
            };
            group
                .behavior_by_tsid
                .merge_profile(pointwise_profile, &mut regions);
            groups.push(group);
        }
    }
    let merge_ms = merge_started_at.elapsed().as_secs_f64() * 1000.0;

    let mut class_to_group = vec![0usize; class_profiles.len()];
    for (group_id, group) in groups.iter().enumerate() {
        for &class in &group.member_classes {
            class_to_group[class] = group_id;
        }
    }
    let coloring = class_coloring
        .iter()
        .map(|class| class_to_group[*class])
        .collect();

    if profile_enabled {
        let (direct_overlay_slots, direct_overlay_hits, direct_overlay_misses, direct_overlay_replacements) =
            regions.direct_overlay_stats();
        let group_tsid_ranges = groups
            .iter()
            .map(|group| group.behavior_by_tsid.ranges.len())
            .sum::<usize>();
        let region_entries = groups
            .iter()
            .map(|group| group.behavior_by_tsid.region_entry_count())
            .sum::<usize>();
        eprintln!(
            "[glrmask/profile][weighted_dwa_minimize_pointwise_ranges] candidates={} classes={} groups={} behaviors={} interned_regions={} profile_ranges={} profile_tsid_cells={} group_tsid_ranges={} regions={} region_build_cache_entries={} region_build_cache_hits={} region_build_cache_misses={} direct_overlay_slots={} direct_overlay_hits={} direct_overlay_misses={} direct_overlay_replacements={} profile_build_ms={:.3} merge_ms={:.3} total_ms={:.3} group_attempts={} target_rejects={} behavior_rejects={}",
            candidates.len(),
            class_profiles.len(),
            groups.len(),
            interner.ids.len(),
            regions.regions.len(),
            profile_ranges,
            profile_tsid_cells,
            group_tsid_ranges,
            region_entries,
            region_build_cache.entries.len(),
            region_build_cache.hits,
            region_build_cache.misses,
            direct_overlay_slots,
            direct_overlay_hits,
            direct_overlay_misses,
            direct_overlay_replacements,
            profile_build_ms,
            merge_ms,
            started_at.elapsed().as_secs_f64() * 1000.0,
            group_attempts,
            target_rejects,
            behavior_rejects,
        );
    }
    Some(coloring)
}

fn sorted_targets_compatible(class_targets: &[(Label, u32)], group_targets: &[(Label, u32)]) -> bool {
    let mut class_idx = 0;
    let mut group_idx = 0;

    while class_idx < class_targets.len() && group_idx < group_targets.len() {
        let (class_label, class_target) = class_targets[class_idx];
        let (group_label, group_target) = group_targets[group_idx];
        if class_label == group_label {
            if class_target != group_target {
                return false;
            }
            class_idx += 1;
            group_idx += 1;
        } else if class_label < group_label {
            class_idx += 1;
        } else {
            group_idx += 1;
        }
    }

    true
}

fn sorted_weights_compatible_on_domain_intersection(
    class_weights: &[(Label, Weight)],
    group_weights: &[(Label, Weight)],
    left_domain: &Weight,
    right_domain: &Weight,
) -> bool {
    let mut class_idx = 0;
    let mut group_idx = 0;

    while class_idx < class_weights.len() && group_idx < group_weights.len() {
        let (class_label, class_weight) = &class_weights[class_idx];
        let (group_label, group_weight) = &group_weights[group_idx];
        if class_label == group_label {
            if class_weight != group_weight {
                let class_disjoint = weight_is_disjoint_from_domain_intersection(
                    class_weight,
                    left_domain,
                    right_domain,
                );
                let group_disjoint = weight_is_disjoint_from_domain_intersection(
                    group_weight,
                    left_domain,
                    right_domain,
                );
                if class_disjoint != group_disjoint {
                    return false;
                }
                if !class_disjoint
                    && !weights_equal_on_domain_intersection(
                        class_weight,
                        group_weight,
                        left_domain,
                        right_domain,
                    )
                {
                    return false;
                }
            }
            class_idx += 1;
            group_idx += 1;
        } else if class_label < group_label {
            if !weight_is_disjoint_from_domain_intersection(class_weight, left_domain, right_domain) {
                return false;
            }
            class_idx += 1;
        } else {
            if !weight_is_disjoint_from_domain_intersection(group_weight, left_domain, right_domain) {
                return false;
            }
            group_idx += 1;
        }
    }

    for (_, class_weight) in &class_weights[class_idx..] {
        if !weight_is_disjoint_from_domain_intersection(class_weight, left_domain, right_domain) {
            return false;
        }
    }
    for (_, group_weight) in &group_weights[group_idx..] {
        if !weight_is_disjoint_from_domain_intersection(group_weight, left_domain, right_domain) {
            return false;
        }
    }

    true
}

/// Compare two sparse label→weight profiles on one already-materialized domain.
///
/// The group path compares many labels against the same overlap. Materializing
/// that overlap once is exact and avoids recomputing its TSID/token intersection
/// for every label.
fn sorted_weights_compatible_on_domain(
    class_weights: &[(Label, Weight)],
    group_weights: &[(Label, Weight)],
    domain: &Weight,
) -> bool {
    let mut class_idx = 0;
    let mut group_idx = 0;

    while class_idx < class_weights.len() && group_idx < group_weights.len() {
        let (class_label, class_weight) = &class_weights[class_idx];
        let (group_label, group_weight) = &group_weights[group_idx];
        if class_label == group_label {
            if class_weight != group_weight {
                let class_disjoint = class_weight.is_disjoint(domain);
                let group_disjoint = group_weight.is_disjoint(domain);
                if class_disjoint != group_disjoint {
                    return false;
                }
                if !class_disjoint && !weights_equal_on_domain(class_weight, group_weight, domain) {
                    return false;
                }
            }
            class_idx += 1;
            group_idx += 1;
        } else if class_label < group_label {
            if !class_weight.is_disjoint(domain) {
                return false;
            }
            class_idx += 1;
        } else {
            if !group_weight.is_disjoint(domain) {
                return false;
            }
            group_idx += 1;
        }
    }

    for (_, class_weight) in &class_weights[class_idx..] {
        if !class_weight.is_disjoint(domain) {
            return false;
        }
    }
    for (_, group_weight) in &group_weights[group_idx..] {
        if !group_weight.is_disjoint(domain) {
            return false;
        }
    }

    true
}

fn targets_compatible_with_group_map(
    class_targets: &[(Label, u32)],
    group_targets_by_label: &FxHashMap<Label, u32>,
) -> bool {
    class_targets.iter().all(|(label, target)| {
        group_targets_by_label
            .get(label)
            .is_none_or(|group_target| *group_target == *target)
    })
}

fn final_weights_compatible_on_domain_intersection(
    class_final_weight: Option<&Weight>,
    member_final_weight: Option<&Weight>,
    class_domain: &Weight,
    member_domain: &Weight,
) -> bool {
    match (class_final_weight, member_final_weight) {
        (Some(class_fw), Some(member_fw)) => weights_equal_on_domain_intersection(
            class_fw,
            member_fw,
            class_domain,
            member_domain,
        ),
        (Some(class_fw), None) => {
            weight_is_disjoint_from_domain_intersection(class_fw, class_domain, member_domain)
        }
        (None, Some(member_fw)) => {
            weight_is_disjoint_from_domain_intersection(member_fw, class_domain, member_domain)
        }
        (None, None) => true,
    }
}

fn final_weights_compatible_on_domain(
    class_final_weight: Option<&Weight>,
    group_final_weight: Option<&Weight>,
    domain: &Weight,
) -> bool {
    match (class_final_weight, group_final_weight) {
        (Some(class_weight), Some(group_weight)) => {
            weights_equal_on_domain(class_weight, group_weight, domain)
        }
        (Some(weight), None) | (None, Some(weight)) => weight.is_disjoint(domain),
        (None, None) => true,
    }
}

fn merge_sorted_targets(existing: &mut Vec<(Label, u32)>, add: &[(Label, u32)]) {
    if add.is_empty() {
        return;
    }
    if existing.is_empty() {
        existing.extend_from_slice(add);
        return;
    }

    let mut merged = Vec::with_capacity(existing.len() + add.len());
    let mut existing_idx = 0;
    let mut add_idx = 0;
    while existing_idx < existing.len() && add_idx < add.len() {
        let existing_entry = existing[existing_idx];
        let add_entry = add[add_idx];
        if existing_entry.0 == add_entry.0 {
            debug_assert_eq!(existing_entry.1, add_entry.1);
            merged.push(existing_entry);
            existing_idx += 1;
            add_idx += 1;
        } else if existing_entry.0 < add_entry.0 {
            merged.push(existing_entry);
            existing_idx += 1;
        } else {
            merged.push(add_entry);
            add_idx += 1;
        }
    }
    merged.extend_from_slice(&existing[existing_idx..]);
    merged.extend_from_slice(&add[add_idx..]);
    *existing = merged;
}

fn merge_sorted_weights(existing: &mut Vec<(Label, Weight)>, add: &[(Label, Weight)]) {
    if add.is_empty() {
        return;
    }
    if existing.is_empty() {
        existing.extend(add.iter().cloned());
        return;
    }

    let mut merged = Vec::with_capacity(existing.len() + add.len());
    let mut existing_idx = 0;
    let mut add_idx = 0;
    while existing_idx < existing.len() && add_idx < add.len() {
        let (existing_label, existing_weight) = &existing[existing_idx];
        let (add_label, add_weight) = &add[add_idx];
        if existing_label == add_label {
            let merged_weight = if existing_weight == add_weight {
                existing_weight.clone()
            } else {
                existing_weight.union(add_weight)
            };
            merged.push((*existing_label, merged_weight));
            existing_idx += 1;
            add_idx += 1;
        } else if existing_label < add_label {
            merged.push((*existing_label, existing_weight.clone()));
            existing_idx += 1;
        } else {
            merged.push((*add_label, add_weight.clone()));
            add_idx += 1;
        }
    }
    merged.extend(existing[existing_idx..].iter().cloned());
    merged.extend(add[add_idx..].iter().cloned());
    *existing = merged;
}

fn tsid_coverage_disjoint(
    left: &Option<RangeSetBlaze<u32>>,
    right: &Option<RangeSetBlaze<u32>>,
) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => left.is_disjoint(right),
        _ => false,
    }
}

fn merge_tsid_coverage(
    target: &mut Option<RangeSetBlaze<u32>>,
    add: &Option<RangeSetBlaze<u32>>,
) {
    match (&mut *target, add) {
        (None, _) | (_, None) => *target = None,
        (Some(target_set), Some(add_set)) => {
            *target_set = target_set.clone() | add_set.clone();
        }
    }
}

const TSID_MEMBER_INDEX_ENUMERATION_LIMIT: usize = 256;

fn enumerate_tsid_coverage_limited(
    coverage: &Option<RangeSetBlaze<u32>>,
    limit: usize,
    mut visit: impl FnMut(u32),
) -> bool {
    let Some(coverage) = coverage else {
        return false;
    };
    let mut count = 0usize;
    for range in coverage.ranges() {
        let mut tsid = *range.start();
        loop {
            count += 1;
            if count > limit {
                return false;
            }
            visit(tsid);
            if tsid == *range.end() {
                break;
            }
            tsid += 1;
        }
    }
    true
}

/// The indexed memberwise path is considerably cheaper when each incoming
/// class has a small TSID footprint.  Broad classes cannot use that index and
/// can force repeated scans of a large group, so after several such probes we
/// promote the group to the exact pointwise summary used by the summary path.
/// Dense indexed probes can be just as expensive.  Promote compact profiles
/// after enough actual overlap checks, rather than using wall-clock time: the
/// latter is host-load dependent and can misclassify very wide profiles whose
/// eager summary would be more expensive than their indexed checks.  Both
/// representations encode the same compatibility condition.
const SUMMARY_PROMOTION_MIN_MEMBERS: usize = 64;
const SUMMARY_PROMOTION_BROAD_PROBES: usize = 4;
const SUMMARY_WORK_PROMOTION_MAX_PROFILE_WEIGHTS: usize = 128;
const SUMMARY_PROMOTION_MEMBERWISE_OVERLAP_CHECKS: usize = 4_096;

/// A summary is an exact union over a stable prefix of a merge group.  Updating
/// that union one member at a time repeatedly normalizes the same growing
/// weights, which is quadratic for wide terminal-DWA groups.  Keep subsequent
/// members as an exact bounded suffix and rebuild the immutable snapshot in
/// batches.  Compatibility checks cover both pieces, so this changes only the
/// construction schedule, not the merge relation.
const SUMMARY_SNAPSHOT_BATCH_SIZE: usize = 64;

struct ExactGroupSummary {
    needed_union: Weight,
    merged_final_weight: Option<Weight>,
    transition_weights: Vec<(Label, Weight)>,
}

struct OverlapMergeGroup {
    targets_by_label: FxHashMap<Label, u32>,
    needed_tsid_coverage: Option<RangeSetBlaze<u32>>,
    indexed_members_by_tsid: FxHashMap<u32, Vec<usize>>,
    unindexed_member_classes: Vec<usize>,
    member_classes: Vec<usize>,
    broad_probe_count: usize,
    max_profile_weights: usize,
    memberwise_overlap_checks: usize,
    /// Exact aggregate for every member before `summary_pending_classes`.
    summary: Option<ExactGroupSummary>,
    /// Exact suffix not yet folded into the immutable aggregate.
    summary_pending_classes: Vec<usize>,
}

fn should_promote_group_summary(group: &OverlapMergeGroup) -> bool {
    group.member_classes.len() >= SUMMARY_PROMOTION_MIN_MEMBERS
        && (group.broad_probe_count >= SUMMARY_PROMOTION_BROAD_PROBES
            || (group.max_profile_weights <= SUMMARY_WORK_PROMOTION_MAX_PROFILE_WEIGHTS
                && group.memberwise_overlap_checks
                    >= SUMMARY_PROMOTION_MEMBERWISE_OVERLAP_CHECKS))
}

fn build_exact_group_summary(
    member_classes: &[usize],
    class_needed_union: &[Weight],
    class_profiles: &[ClassProfile],
) -> ExactGroupSummary {
    let needed_union = Weight::union_all(
        member_classes
            .iter()
            .map(|&class| &class_needed_union[class]),
    );

    let final_weights: Vec<&Weight> = member_classes
        .iter()
        .filter_map(|&class| class_profiles[class].final_weight.as_ref())
        .collect();
    let merged_final_weight =
        (!final_weights.is_empty()).then(|| Weight::union_all(final_weights));

    let mut weights_by_label: BTreeMap<Label, Vec<&Weight>> = BTreeMap::new();
    for &class in member_classes {
        for (label, weight) in &class_profiles[class].weights {
            weights_by_label.entry(*label).or_default().push(weight);
        }
    }
    let transition_weights = weights_by_label
        .into_iter()
        .map(|(label, weights)| (label, Weight::union_all(weights)))
        .collect();

    ExactGroupSummary {
        needed_union,
        merged_final_weight,
        transition_weights,
    }
}

fn update_exact_group_summary(
    summary: &mut ExactGroupSummary,
    needed: &Weight,
    profile: &ClassProfile,
) {
    if summary.needed_union != *needed {
        summary.needed_union = summary.needed_union.union(needed);
    }
    if let Some(final_weight) = &profile.final_weight {
        summary.merged_final_weight = Some(match summary.merged_final_weight.take() {
            Some(existing) if existing == *final_weight => existing,
            Some(existing) => existing.union(final_weight),
            None => final_weight.clone(),
        });
    }
    merge_sorted_weights(&mut summary.transition_weights, &profile.weights);
}

fn memberwise_group_compatible(
    class_domain: &Weight,
    class_profile: &ClassProfile,
    member_classes: &[usize],
    class_needed_union: &[Weight],
    class_profiles: &[ClassProfile],
) -> bool {
    member_classes.iter().all(|&member_class| {
        let member_domain = &class_needed_union[member_class];
        class_domain.is_disjoint(member_domain)
            || (final_weights_compatible_on_domain_intersection(
                class_profile.final_weight.as_ref(),
                class_profiles[member_class].final_weight.as_ref(),
                class_domain,
                member_domain,
            ) && sorted_weights_compatible_on_domain_intersection(
                &class_profile.weights,
                &class_profiles[member_class].weights,
                class_domain,
                member_domain,
            ))
    })
}

/// Hybrid coloring: partition refinement to reduce candidates into classes,
/// then graph coloring among class representatives.
///
/// States within the same partition-refinement class have identical transition
/// structure (same labels, mapped targets, weights) and are guaranteed compatible.
/// We only need to check pairwise compatibility among classes, reducing O(N²) to O(K²)
/// where K is the number of classes (typically K << N).
///
/// To ensure correctness, we compute the union of needed sets for each class
/// and use those when checking inter-class compatibility. This guarantees that
/// if two class representatives are deemed compatible, ALL pairs across the
/// two classes are compatible (since transitions/weights are identical within
/// a class, only the needed-set overlap domain varies).
fn build_and_color_pairwise_greedy(
    dwa: &DWA,
    candidates: &[usize],
    needed: &[Weight],
    old_to_new: &[u32],
    productive_transitions: &[Vec<ProductiveTransition>],
) -> Vec<usize> {
    // A merge group is valid when every pair of members agrees on the
    // intersection of its live domains and has no conflicting label target.
    // Checking the full clique directly is exact: pairwise agreement of these
    // partial deterministic behaviors implies one well-defined union behavior
    // for the whole group. For small height buckets this avoids constructing
    // the much larger pointwise profile representation merely to discover that
    // almost every successful merge is domain-disjoint.
    let mut groups = Vec::<SmallVec<[usize; 16]>>::new();
    let mut coloring = Vec::with_capacity(candidates.len());
    for &candidate in candidates {
        let group = groups
            .iter()
            .position(|members| {
                members.iter().all(|&member| {
                    are_compatible(
                        candidate,
                        member,
                        dwa,
                        needed,
                        old_to_new,
                        productive_transitions,
                        false,
                    )
                })
            })
            .unwrap_or_else(|| {
                groups.push(SmallVec::new());
                groups.len() - 1
            });
        groups[group].push(candidate);
        coloring.push(group);
    }
    coloring
}

fn build_and_color_hybrid(
    dwa: &DWA,
    candidates: &[usize],
    needed: &[Weight],
    old_to_new: &[u32],
    productive_transitions: &[Vec<ProductiveTransition>],
    pointwise_class_order: PointwiseClassOrder,
    monotone_pointwise: bool,
) -> HybridColoring {
    let profile_enabled = weighted_dwa_minimize_profile_enabled();
    // Pairwise clique checks are excellent for genuinely sparse small buckets,
    // but become much more expensive than the pointwise representation once
    // every state is live over a moderately broad TSID domain. The old hard
    // <=64 cutoff sent exactly those broad-domain buckets down an O(n^2)-ish
    // path. Counting the finite TSID projection directly from compressed
    // weight ranges is cheap and lets us retain pairwise for the sparse shape
    // it was intended for.
    const SMALL_PAIRWISE_ALWAYS_CANDIDATES: usize = 16;
    const SMALL_PAIRWISE_MAX_CANDIDATES: usize = 64;
    const SMALL_PAIRWISE_MAX_AVG_TSID_COVERAGE: usize = 8;
    let small_pairwise_disabled =
        std::env::var_os("GLRMASK_WEIGHTED_MINIMIZE_DISABLE_PAIRWISE_SMALL").is_some();
    let small_pairwise_sparse = if candidates.len() <= SMALL_PAIRWISE_ALWAYS_CANDIDATES
        && !small_pairwise_disabled
    {
        true
    } else if candidates.len() <= SMALL_PAIRWISE_MAX_CANDIDATES
        && !small_pairwise_disabled
    {
        let max_total = candidates
            .len()
            .saturating_mul(SMALL_PAIRWISE_MAX_AVG_TSID_COVERAGE);
        let mut total = 0usize;
        let mut sparse = true;
        for &candidate in candidates {
            let Some(coverage) = needed[candidate].tsid_coverage_len() else {
                sparse = false;
                break;
            };
            total = total.saturating_add(coverage);
            if total > max_total {
                sparse = false;
                break;
            }
        }
        sparse
    } else {
        false
    };
    if small_pairwise_sparse {
        return HybridColoring::ordinary(build_and_color_pairwise_greedy(
            dwa,
            candidates,
            needed,
            old_to_new,
            productive_transitions,
        ));
    }
    let broad_small_bucket = candidates.len() <= SMALL_PAIRWISE_MAX_CANDIDATES
        && !small_pairwise_disabled;
    let total_started_at = Instant::now();

    // Step 1: Partition refinement to get fine-grained classes.
    let partition_refine_started_at = Instant::now();
    let class_coloring = partition_refine_coloring_raw(candidates, dwa, old_to_new);
    let partition_refine_ms = partition_refine_started_at.elapsed().as_secs_f64() * 1000.0;
    let num_classes = class_coloring.iter().max().map(|&c| c + 1).unwrap_or(0);
    if num_classes <= 1 {
        if profile_enabled {
            eprintln!(
                "[glrmask/profile][weighted_dwa_minimize_hybrid] candidates={} classes={} groups={} partition_refine_ms={:.3} class_union_ms={:.3} class_profiles_ms={:.3} greedy_merge_ms={:.3} map_ms={:.3} total_ms={:.3}",
                candidates.len(),
                num_classes,
                num_classes,
                partition_refine_ms,
                0.0,
                0.0,
                0.0,
                0.0,
                total_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        return HybridColoring::ordinary(class_coloring);
    }

    // Step 2: Pick one representative state per class and compute the union
    // of needed sets for each class.
    let class_union_started_at = Instant::now();
    let mut class_rep_state: Vec<usize> = vec![usize::MAX; num_classes];
    let mut class_needed_union: Vec<Weight> = Vec::with_capacity(num_classes);
    class_needed_union.resize_with(num_classes, Weight::empty);
    for (idx, &class) in class_coloring.iter().enumerate() {
        let state_id = candidates[idx];
        if class_rep_state[class] == usize::MAX {
            class_rep_state[class] = state_id;
        }
        class_needed_union[class] = class_needed_union[class].union(&needed[state_id]);
    }
    let class_union_ms = class_union_started_at.elapsed().as_secs_f64() * 1000.0;

    let class_profiles_started_at = Instant::now();
    let class_profiles: Vec<ClassProfile> = class_rep_state
        .iter()
        .map(|&rep| build_class_profile(rep, old_to_new, productive_transitions, dwa))
        .collect();
    let class_profiles_ms = class_profiles_started_at.elapsed().as_secs_f64() * 1000.0;
    if profile_enabled {
        let mut coverage_cells = 0usize;
        let mut max_coverage_cells = 0usize;
        let mut coverage_over_256 = 0usize;
        for weight in &class_needed_union {
            let count = weight
                .tsid_coverage()
                .as_ref()
                .map(|set| set.ranges().map(|range| (*range.end() as usize - *range.start() as usize) + 1).sum())
                .unwrap_or(usize::MAX);
            coverage_cells = coverage_cells.saturating_add(count);
            max_coverage_cells = max_coverage_cells.max(count);
            coverage_over_256 += usize::from(count > 256);
        }
        let total_profile_weight_ranges: usize = class_profiles
            .iter()
            .flat_map(|profile| profile.weights.iter())
            .map(|(_, weight)| weight.outer_range_count())
            .sum();
        let total_profile_weight_cells: usize = class_profiles
            .iter()
            .flat_map(|profile| profile.weights.iter())
            .map(|(_, weight)| weight.tsid_coverage().map(|set| set.ranges().map(|range| (*range.end() as usize - *range.start() as usize) + 1).sum()).unwrap_or(usize::MAX))
            .fold(0usize, usize::saturating_add);
        eprintln!(
            "[glrmask/profile][weighted_dwa_minimize_hybrid_shape] candidates={} classes={} coverage_cells={} max_coverage_cells={} coverage_over_256={} total_profile_weight_ranges={} total_profile_weight_cells={}",
            candidates.len(), num_classes, coverage_cells, max_coverage_cells, coverage_over_256, total_profile_weight_ranges, total_profile_weight_cells,
        );
    }

    // Each class is a partial behavior function over its pushed-needed
    // TSID/token domain. When that representation is finite, compare against
    // the exact union function of each group instead of revisiting all members.
    // A full sentinel cannot be enumerated, so it retains the generic path.
    let estimated_pointwise_work = class_profiles
        .iter()
        .flat_map(|profile| profile.weights.iter())
        .map(|(_, weight)| weight.outer_range_count())
        .sum::<usize>();
    let force_overlap_indexed_merge = std::env::var("GLRMASK_WEIGHTED_MINIMIZE_FORCE_OVERLAP_INDEXED")
        .ok()
        .is_some_and(|value| {
            let value = value.trim();
            value.is_empty() || value == "1" || value.eq_ignore_ascii_case("true")
        });
    let prefer_overlap_indexed_merge = force_overlap_indexed_merge
        || (num_classes <= 64 && estimated_pointwise_work >= 8_192);
    let pointwise_coloring = if prefer_overlap_indexed_merge {
        None
    } else if pointwise_tsid_ranges_enabled() {
        try_build_and_color_pointwise_ranges(
            candidates,
            &class_coloring,
            &class_needed_union,
            &class_profiles,
            pointwise_class_order,
            profile_enabled,
            monotone_pointwise,
        )
        .map(HybridColoring::ordinary)
    } else {
        const DIRECT_OVERLAY_MIN_LARGE_CANDIDATES: usize = 96;
        let default_overlay =
            broad_small_bucket || candidates.len() >= DIRECT_OVERLAY_MIN_LARGE_CANDIDATES;
        let requested_overlay_slots = std::env::var("GLRMASK_WEIGHTED_MINIMIZE_DIRECT_OVERLAY_SLOTS")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(if default_overlay { 256 } else { 0 });
        let direct_overlay_slots = if requested_overlay_slots == 0 {
            0
        } else {
            requested_overlay_slots.next_power_of_two()
        };
        try_build_and_color_pointwise(
            candidates,
            &class_coloring,
            &class_needed_union,
            &class_profiles,
            pointwise_class_order,
            profile_enabled,
            monotone_pointwise,
            direct_overlay_slots,
        )
    };
    if profile_enabled && prefer_overlap_indexed_merge {
        eprintln!(
            "[glrmask/profile][weighted_dwa_minimize_overlap_choice] candidates={} classes={} estimated_pointwise_work={} strategy=overlap_indexed",
            candidates.len(), num_classes, estimated_pointwise_work,
        );
    }
    if let Some(coloring) = pointwise_coloring {
        if profile_enabled {
            eprintln!(
                "[glrmask/profile][weighted_dwa_minimize_pointwise_preamble] candidates={} classes={} partition_refine_ms={:.3} class_union_ms={:.3} class_profiles_ms={:.3}",
                candidates.len(),
                num_classes,
                partition_refine_ms,
                class_union_ms,
                class_profiles_ms,
            );
        }
        return coloring;
    }

    // The generic fallback needs per-class TSID coverage for its indexed
    // overlap probes. Do not build it on the normal pointwise-success path.
    let class_tsid_coverage: Vec<Option<RangeSetBlaze<u32>>> = class_needed_union
        .iter()
        .map(Weight::tsid_coverage)
        .collect();
    let classes_with_final_weight = class_profiles
        .iter()
        .filter(|profile| profile.final_weight.is_some())
        .count();
    let min_targets = class_profiles
        .iter()
        .map(|profile| profile.targets.len())
        .min()
        .unwrap_or(0);
    let max_targets = class_profiles
        .iter()
        .map(|profile| profile.targets.len())
        .max()
        .unwrap_or(0);
    let avg_targets = class_profiles
        .iter()
        .map(|profile| profile.targets.len() as f64)
        .sum::<f64>()
        / num_classes as f64;
    let min_weights = class_profiles
        .iter()
        .map(|profile| profile.weights.len())
        .min()
        .unwrap_or(0);
    let max_weights = class_profiles
        .iter()
        .map(|profile| profile.weights.len())
        .max()
        .unwrap_or(0);
    let avg_weights = class_profiles
        .iter()
        .map(|profile| profile.weights.len() as f64)
        .sum::<f64>()
        / num_classes as f64;

    // Step 3: Greedy merge of classes, handling both disjoint and overlapping
    // needed sets. Instead of building an O(K²) incompatibility graph, we check
    // each class against only the small number of existing groups (~14 for kb_684).
    //
    let greedy_merge_started_at = Instant::now();
    let mut groups: Vec<OverlapMergeGroup> = Vec::new();
    let mut group_attempts = 0usize;
    let mut target_checks = 0usize;
    let mut target_rejects = 0usize;
    let mut target_check_ms = 0.0;
    let mut disjoint_checks = 0usize;
    let mut disjoint_true = 0usize;
    let mut disjoint_check_ms = 0.0;
    let mut final_weight_checks = 0usize;
    let mut final_weight_rejects = 0usize;
    let mut final_weight_check_ms = 0.0;
    let mut transition_weight_checks = 0usize;
    let mut transition_weight_rejects = 0usize;
    let mut transition_weight_check_ms = 0.0;
    let mut group_update_ms = 0.0;
    let mut memberwise_indexed_probes = 0usize;
    let mut memberwise_broad_probes = 0usize;
    let mut memberwise_member_scans = 0usize;
    let mut memberwise_overlap_checks = 0usize;
    let mut summary_checks = 0usize;
    let mut summary_promotions = 0usize;
    let mut summary_work_promotions = 0usize;
    let mut summary_snapshot_rebuilds = 0usize;
    let mut summary_pending_member_scans = 0usize;
    let mut overlap_candidate_marks = vec![0u32; num_classes];
    let mut overlap_candidate_mark = 0u32;
    let mut overlap_candidate_members = Vec::<usize>::new();

    for class in 0..num_classes {
        let cn = &class_needed_union[class];
        let class_profile = &class_profiles[class];

        let mut placed = false;
        for g in &mut groups {
            group_attempts += 1;

            target_checks += 1;
            let target_check_started_at = Instant::now();
            let targets_compatible =
                targets_compatible_with_group_map(&class_profile.targets, &g.targets_by_label);
            target_check_ms += target_check_started_at.elapsed().as_secs_f64() * 1000.0;
            if !targets_compatible {
                target_rejects += 1;
                continue;
            }

            disjoint_checks += 1;
            let disjoint_check_started_at = Instant::now();
            let compatible = if let Some(summary) = g.summary.as_ref() {
                summary_checks += 1;
                let is_disjoint = cn.is_disjoint(&summary.needed_union);
                if is_disjoint {
                    disjoint_true += 1;
                    true
                } else {
                    let overlap = cn.intersection(&summary.needed_union);
                    debug_assert!(!overlap.is_empty());

                    final_weight_checks += 1;
                    let final_weight_check_started_at = Instant::now();
                    let mut summary_compatible = final_weights_compatible_on_domain(
                        class_profile.final_weight.as_ref(),
                        summary.merged_final_weight.as_ref(),
                        &overlap,
                    );
                    final_weight_check_ms +=
                        final_weight_check_started_at.elapsed().as_secs_f64() * 1000.0;
                    if !summary_compatible {
                        final_weight_rejects += 1;
                    }

                    if summary_compatible {
                        transition_weight_checks += 1;
                        let transition_weight_check_started_at = Instant::now();
                        summary_compatible = sorted_weights_compatible_on_domain(
                            &class_profile.weights,
                            &summary.transition_weights,
                            &overlap,
                        );
                        transition_weight_check_ms +=
                            transition_weight_check_started_at.elapsed().as_secs_f64() * 1000.0;
                        if !summary_compatible {
                            transition_weight_rejects += 1;
                        }
                    }

                    // The immutable summary covers the stable prefix.  Check the
                    // bounded suffix memberwise until the next snapshot rebuild.
                    // This is exact: a candidate is compatible with the whole
                    // group iff it is compatible with the prefix union and every
                    // suffix member on their respective overlap domains.
                    if summary_compatible {
                        for &member_class in &g.summary_pending_classes {
                            summary_pending_member_scans += 1;
                            memberwise_member_scans += 1;
                            let member_needed = &class_needed_union[member_class];
                            if cn.is_disjoint(member_needed) {
                                continue;
                            }
                            memberwise_overlap_checks += 1;

                            final_weight_checks += 1;
                            let final_weight_check_started_at = Instant::now();
                            let final_weight_ok = final_weights_compatible_on_domain_intersection(
                                class_profile.final_weight.as_ref(),
                                class_profiles[member_class].final_weight.as_ref(),
                                cn,
                                member_needed,
                            );
                            final_weight_check_ms +=
                                final_weight_check_started_at.elapsed().as_secs_f64() * 1000.0;
                            if !final_weight_ok {
                                final_weight_rejects += 1;
                                summary_compatible = false;
                                break;
                            }

                            transition_weight_checks += 1;
                            let transition_weight_check_started_at = Instant::now();
                            let transition_weights_ok =
                                sorted_weights_compatible_on_domain_intersection(
                                    &class_profile.weights,
                                    &class_profiles[member_class].weights,
                                    cn,
                                    member_needed,
                                );
                            transition_weight_check_ms += transition_weight_check_started_at
                                .elapsed()
                                .as_secs_f64()
                                * 1000.0;
                            if !transition_weights_ok {
                                transition_weight_rejects += 1;
                                summary_compatible = false;
                                break;
                            }
                        }
                    }

                    #[cfg(debug_assertions)]
                    debug_assert_eq!(
                        summary_compatible,
                        memberwise_group_compatible(
                            cn,
                            class_profile,
                            &g.member_classes,
                            &class_needed_union,
                            &class_profiles,
                        ),
                        "group summary plus pending suffix must be equivalent to checking every member",
                    );
                    summary_compatible
                }
            } else {
                overlap_candidate_members.clear();
                overlap_candidate_mark = overlap_candidate_mark.wrapping_add(1);
                if overlap_candidate_mark == 0 {
                    overlap_candidate_marks.fill(0);
                    overlap_candidate_mark = 1;
                }

                for &member_class in &g.unindexed_member_classes {
                    overlap_candidate_marks[member_class] = overlap_candidate_mark;
                    overlap_candidate_members.push(member_class);
                }
                let indexed = enumerate_tsid_coverage_limited(
                    &class_tsid_coverage[class],
                    TSID_MEMBER_INDEX_ENUMERATION_LIMIT,
                    |tsid| {
                        if let Some(members) = g.indexed_members_by_tsid.get(&tsid) {
                            for &member_class in members {
                                if overlap_candidate_marks[member_class] != overlap_candidate_mark {
                                    overlap_candidate_marks[member_class] = overlap_candidate_mark;
                                    overlap_candidate_members.push(member_class);
                                }
                            }
                        }
                    },
                );

                let scan_members: &[usize] = if indexed {
                    memberwise_indexed_probes += 1;
                    if overlap_candidate_members.is_empty() {
                        disjoint_true += 1;
                    }
                    &overlap_candidate_members
                } else {
                    memberwise_broad_probes += 1;
                    g.broad_probe_count += 1;
                    if tsid_coverage_disjoint(
                        &class_tsid_coverage[class],
                        &g.needed_tsid_coverage,
                    ) {
                        disjoint_true += 1;
                        &[]
                    } else {
                        &g.member_classes
                    }
                };

                let mut saw_overlap = false;
                let mut memberwise_compatible = true;
                for &member_class in scan_members {
                    memberwise_member_scans += 1;
                    let member_needed = &class_needed_union[member_class];
                    if cn.is_disjoint(member_needed) {
                        continue;
                    }
                    saw_overlap = true;
                    g.memberwise_overlap_checks += 1;
                    memberwise_overlap_checks += 1;

                    final_weight_checks += 1;
                    let final_weight_check_started_at = Instant::now();
                    let final_weight_ok = final_weights_compatible_on_domain_intersection(
                        class_profile.final_weight.as_ref(),
                        class_profiles[member_class].final_weight.as_ref(),
                        cn,
                        member_needed,
                    );
                    final_weight_check_ms +=
                        final_weight_check_started_at.elapsed().as_secs_f64() * 1000.0;
                    if !final_weight_ok {
                        final_weight_rejects += 1;
                        memberwise_compatible = false;
                        break;
                    }

                    transition_weight_checks += 1;
                    let transition_weight_check_started_at = Instant::now();
                    let transition_weights_ok = sorted_weights_compatible_on_domain_intersection(
                        &class_profile.weights,
                        &class_profiles[member_class].weights,
                        cn,
                        member_needed,
                    );
                    transition_weight_check_ms +=
                        transition_weight_check_started_at.elapsed().as_secs_f64() * 1000.0;
                    if !transition_weights_ok {
                        transition_weight_rejects += 1;
                        memberwise_compatible = false;
                        break;
                    }
                }
                if !saw_overlap && !scan_members.is_empty() {
                    disjoint_true += 1;
                }
                memberwise_compatible
            };
            if g.summary.is_none() && should_promote_group_summary(g) {
                let promoted_by_work = g.broad_probe_count < SUMMARY_PROMOTION_BROAD_PROBES;
                g.summary = Some(build_exact_group_summary(
                    &g.member_classes,
                    &class_needed_union,
                    &class_profiles,
                ));
                summary_promotions += 1;
                summary_work_promotions += usize::from(promoted_by_work);
            }
            disjoint_check_ms += disjoint_check_started_at.elapsed().as_secs_f64() * 1000.0;
            if !compatible {
                continue;
            }

            // Compatible — merge into this group
            let group_update_started_at = Instant::now();
            for (label, target) in &class_profile.targets {
                if let Some(existing_target) = g.targets_by_label.get(label) {
                    debug_assert_eq!(*existing_target, *target);
                } else {
                    g.targets_by_label.insert(*label, *target);
                }
            }
            merge_tsid_coverage(
                &mut g.needed_tsid_coverage,
                &class_tsid_coverage[class],
            );
            if !enumerate_tsid_coverage_limited(
                &class_tsid_coverage[class],
                TSID_MEMBER_INDEX_ENUMERATION_LIMIT,
                |tsid| g.indexed_members_by_tsid.entry(tsid).or_default().push(class),
            ) {
                g.unindexed_member_classes.push(class);
            }
            g.member_classes.push(class);
            g.max_profile_weights = g.max_profile_weights.max(class_profile.weights.len());
            if g.summary.is_some() {
                g.summary_pending_classes.push(class);
                if g.summary_pending_classes.len() >= SUMMARY_SNAPSHOT_BATCH_SIZE {
                    g.summary = Some(build_exact_group_summary(
                        &g.member_classes,
                        &class_needed_union,
                        &class_profiles,
                    ));
                    g.summary_pending_classes.clear();
                    summary_snapshot_rebuilds += 1;
                }
            } else if should_promote_group_summary(g) {
                g.summary = Some(build_exact_group_summary(
                    &g.member_classes,
                    &class_needed_union,
                    &class_profiles,
                ));
                summary_promotions += 1;
                summary_work_promotions += usize::from(
                    g.broad_probe_count < SUMMARY_PROMOTION_BROAD_PROBES,
                );
            }
            group_update_ms += group_update_started_at.elapsed().as_secs_f64() * 1000.0;
            placed = true;
            break;
        }

        if !placed {
            let mut targets_by_label = FxHashMap::default();
            targets_by_label.reserve(class_profile.targets.len());
            for (label, target) in &class_profile.targets {
                targets_by_label.insert(*label, *target);
            }

            groups.push(OverlapMergeGroup {
                targets_by_label,
                needed_tsid_coverage: class_tsid_coverage[class].clone(),
                indexed_members_by_tsid: {
                    let mut by_tsid = FxHashMap::default();
                    if enumerate_tsid_coverage_limited(
                        &class_tsid_coverage[class],
                        TSID_MEMBER_INDEX_ENUMERATION_LIMIT,
                        |tsid| by_tsid.entry(tsid).or_insert_with(Vec::new).push(class),
                    ) {
                        by_tsid
                    } else {
                        FxHashMap::default()
                    }
                },
                unindexed_member_classes: {
                    if enumerate_tsid_coverage_limited(
                        &class_tsid_coverage[class],
                        TSID_MEMBER_INDEX_ENUMERATION_LIMIT,
                        |_| {},
                    ) {
                        Vec::new()
                    } else {
                        vec![class]
                    }
                },
                member_classes: vec![class],
                broad_probe_count: 0,
                max_profile_weights: class_profile.weights.len(),
                memberwise_overlap_checks: 0,
                summary: None,
                summary_pending_classes: Vec::new(),
            });
        }
    }

    let greedy_merge_ms = greedy_merge_started_at.elapsed().as_secs_f64() * 1000.0;
    let min_group_size = groups
        .iter()
        .map(|group| group.member_classes.len())
        .min()
        .unwrap_or(0);
    let max_group_size = groups
        .iter()
        .map(|group| group.member_classes.len())
        .max()
        .unwrap_or(0);

    // Step 4: Map each candidate through class -> merged color.
    let map_started_at = Instant::now();
    let mut class_to_group = vec![0usize; num_classes];
    for (gid, g) in groups.iter().enumerate() {
        for &c in &g.member_classes {
            class_to_group[c] = gid;
        }
    }
    let nc = candidates.len();
    let mut coloring = vec![0usize; nc];
    for (idx, &class) in class_coloring.iter().enumerate() {
        coloring[idx] = class_to_group[class];
    }

    let map_ms = map_started_at.elapsed().as_secs_f64() * 1000.0;

    if profile_enabled {
        eprintln!(
            "[glrmask/profile][weighted_dwa_minimize_hybrid] candidates={} classes={} groups={} partition_refine_ms={:.3} class_union_ms={:.3} class_profiles_ms={:.3} greedy_merge_ms={:.3} map_ms={:.3} total_ms={:.3}",
            candidates.len(),
            num_classes,
            groups.len(),
            partition_refine_ms,
            class_union_ms,
            class_profiles_ms,
            greedy_merge_ms,
            map_ms,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
        eprintln!(
            "[glrmask/profile][weighted_dwa_minimize_hybrid_detail] candidates={} classes={} groups={} group_attempts={} target_checks={} target_rejects={} target_check_ms={:.3} disjoint_checks={} disjoint_true={} disjoint_check_ms={:.3} final_weight_checks={} final_weight_rejects={} final_weight_check_ms={:.3} transition_weight_checks={} transition_weight_rejects={} transition_weight_check_ms={:.3} group_update_ms={:.3} memberwise_indexed_probes={} memberwise_broad_probes={} memberwise_member_scans={} memberwise_overlap_checks={} summary_checks={} summary_promotions={} summary_work_promotions={} summary_snapshot_rebuilds={} summary_pending_member_scans={} classes_with_final_weight={} min_targets={} max_targets={} avg_targets={:.3} min_weights={} max_weights={} avg_weights={:.3} min_group_size={} max_group_size={}",
            candidates.len(),
            num_classes,
            groups.len(),
            group_attempts,
            target_checks,
            target_rejects,
            target_check_ms,
            disjoint_checks,
            disjoint_true,
            disjoint_check_ms,
            final_weight_checks,
            final_weight_rejects,
            final_weight_check_ms,
            transition_weight_checks,
            transition_weight_rejects,
            transition_weight_check_ms,
            group_update_ms,
            memberwise_indexed_probes,
            memberwise_broad_probes,
            memberwise_member_scans,
            memberwise_overlap_checks,
            summary_checks,
            summary_promotions,
            summary_work_promotions,
            summary_snapshot_rebuilds,
            summary_pending_member_scans,
            classes_with_final_weight,
            min_targets,
            max_targets,
            avg_targets,
            min_weights,
            max_weights,
            avg_weights,
            min_group_size,
            max_group_size,
        );
    }

    HybridColoring::ordinary(coloring)
}

fn partition_refine_coloring_raw(
    candidates: &[usize],
    dwa: &DWA,
    old_to_new: &[u32],
) -> Vec<usize> {
    use std::hash::{Hash, Hasher};
    use rustc_hash::FxHasher;

    let nc = candidates.len();
    let mut hash_groups: rustc_hash::FxHashMap<u64, Vec<usize>> =
        rustc_hash::FxHashMap::default();

    for idx in 0..nc {
        let c = candidates[idx];
        let mut hasher = FxHasher::default();
        dwa.states()[c].final_weight.hash(&mut hasher);
        // Hash raw transitions (BTreeMap iterates in label order)
        for (&label, (target, weight)) in &dwa.states()[c].transitions {
            let Some(mapped) = mapped_target(old_to_new, *target) else {
                continue;
            };
            label.hash(&mut hasher);
            mapped.hash(&mut hasher);
            weight.hash(&mut hasher);
        }
        dwa.states()[c].transitions.len().hash(&mut hasher);

        let sig = hasher.finish();
        hash_groups.entry(sig).or_default().push(idx);
    }

    let mut colors = vec![0usize; nc];
    let mut next_color = 0;

    for group in hash_groups.values() {
        if group.len() == 1 {
            colors[group[0]] = next_color;
            next_color += 1;
            continue;
        }

        // Verify within group for hash collisions
        let mut sub_groups: Vec<Vec<usize>> = Vec::new();
        'outer: for &idx in group {
            let c = candidates[idx];
            for sub in &mut sub_groups {
                let rep = candidates[sub[0]];
                if states_raw_equal(c, rep, dwa, old_to_new) {
                    sub.push(idx);
                    continue 'outer;
                }
            }
            sub_groups.push(vec![idx]);
        }

        for sub in &sub_groups {
            let color = next_color;
            next_color += 1;
            for &idx in sub {
                colors[idx] = color;
            }
        }
    }

    colors
}

/// Check if two states have identical raw signatures (no needed restriction).
fn states_raw_equal(
    u: usize,
    v: usize,
    dwa: &DWA,
    old_to_new: &[u32],
) -> bool {
    if dwa.states()[u].final_weight != dwa.states()[v].final_weight {
        return false;
    }
    let su = &dwa.states()[u];
    let sv = &dwa.states()[v];
    if su.transitions.len() != sv.transitions.len() {
        return false;
    }
    // BTreeMap iterates in key order, so we can zip
    for ((&lu, (tu, wu)), (&lv, (tv, wv))) in su.transitions.iter().zip(sv.transitions.iter()) {
        if lu != lv { return false; }
        if mapped_target(old_to_new, *tu) != mapped_target(old_to_new, *tv) { return false; }
        if wu != wv { return false; }
    }
    true
}

fn try_all_compatible_height_0_coloring(
    candidates: &[usize],
    dwa: &DWA,
    _needed: &[Weight],
) -> Option<Vec<usize>> {
    // After push_weights, all leaf states (h=0, no transitions) are always
    // mutually compatible.  Proof:
    //
    //   For leaves, needed[u] = final_weight[u] because there are no outgoing
    //   transitions.
    //
    //   As we merge leaf states into one witness group, witness_domain and
    //   witness_final stay equal: both are the union of the same member final
    //   weights, so they grow in lockstep as each new leaf is added.
    //
    //   The compatibility check compares:
    //     witness_final ∩ overlap  vs  candidate_final ∩ overlap
    //   where overlap = witness_domain ∩ needed_candidate.
    //
    //   Since witness = witness_domain = witness_final:
    //     W ∩ (W ∩ F) = W ∩ F = F ∩ (W ∩ F)
    //   (by idempotency of set intersection), so both sides are always equal.
    //
    // Therefore we can merge all h=0 candidates into a single color without
    // checking pairwise compatibility.

    if candidates.len() <= 1 {
        return None;
    }
    if !candidates
        .iter()
        .all(|&id| dwa.states()[id].transitions.is_empty())
    {
        return None;
    }

    Some(vec![0; candidates.len()])
}

// Merge and reconstruct.

struct MergedStateBuilder {
    final_weights_pending: Vec<Weight>,
    transitions_pending: rustc_hash::FxHashMap<Label, (u32, Vec<Weight>)>,
}

impl Default for MergedStateBuilder {
    fn default() -> Self {
        Self {
            final_weights_pending: Vec::new(),
            transitions_pending: rustc_hash::FxHashMap::default(),
        }
    }
}

impl MergedStateBuilder {
    fn add_final_weight(&mut self, weight: &Weight) {
        self.final_weights_pending.push(weight.clone());
    }

    fn add_transition(&mut self, label: Label, target: u32, weight: Weight) {
        use std::collections::hash_map::Entry;
        match self.transitions_pending.entry(label) {
            Entry::Occupied(mut entry) => {
                let (existing_target, pending_weights) = entry.get_mut();
                debug_assert_eq!(*existing_target, target);
                pending_weights.push(weight);
            }
            Entry::Vacant(entry) => {
                entry.insert((target, vec![weight]));
            }
        }
    }

}

/// Materialize the exact union behavior already computed by pointwise coloring
/// directly into one builder per merge group.
///
/// Pointwise coloring maintains, for every `(tsid, token)` in a group's live
/// domain, the complete deterministic behavior: finality plus labelled mapped
/// targets. Re-expanding all member states after that proof and unioning their
/// weights again is redundant. Rebuilding weights from this partial function is
/// exactly the same union, expressed once per output group instead of once per
/// input member.
#[cfg(test)]
fn pointwise_groups_to_builders_reference(
    groups: &[PointwiseMergeGroup],
    interner: &PointwiseBehaviorInterner,
) -> Vec<MergedStateBuilder> {
    groups
        .iter()
        .map(|group| {
            let mut entries = match &group.behavior_by_tsid {
                PointwiseBehaviorMap::Sparse(entries) => entries
                    .iter()
                    .map(|(&tsid, region)| (tsid, region))
                    .collect::<Vec<_>>(),
                PointwiseBehaviorMap::Dense(entries) => entries
                    .iter()
                    .enumerate()
                    .filter_map(|(tsid, region)| {
                        region.as_ref().map(|region| (tsid as u32, region))
                    })
                    .collect::<Vec<_>>(),
            };
            entries.sort_unstable_by_key(|(tsid, _)| *tsid);

            let mut final_entries = Vec::<(u32, RangeSetBlaze<u32>)>::new();
            let mut transition_entries = BTreeMap::<Label, Vec<(u32, RangeSetBlaze<u32>)>>::new();

            for (tsid, region) in entries {
                let mut final_ranges = Vec::<std::ops::RangeInclusive<u32>>::new();
                let mut ranges_by_label =
                    BTreeMap::<Label, Vec<std::ops::RangeInclusive<u32>>>::new();
                for token_range in region.iter() {
                    let behavior = interner.get(token_range.behavior);
                    if behavior.final_active {
                        final_ranges.push(token_range.start..=token_range.end);
                    }
                    for &(label, target) in &behavior.transitions {
                        debug_assert_eq!(group.targets_by_label.get(&label), Some(&target));
                        ranges_by_label
                            .entry(label)
                            .or_default()
                            .push(token_range.start..=token_range.end);
                    }
                }
                if !final_ranges.is_empty() {
                    final_entries.push((tsid, RangeSetBlaze::from_iter(final_ranges)));
                }
                for (label, ranges) in ranges_by_label {
                    transition_entries
                        .entry(label)
                        .or_default()
                        .push((tsid, RangeSetBlaze::from_iter(ranges)));
                }
            }

            let mut builder = MergedStateBuilder::default();
            let final_weight = Weight::from_per_tsid_token_sets(final_entries);
            if !final_weight.is_empty() {
                builder.add_final_weight(&final_weight);
            }
            for (label, entries) in transition_entries {
                let weight = Weight::from_per_tsid_token_sets(entries);
                if weight.is_empty() {
                    continue;
                }
                let target = *group
                    .targets_by_label
                    .get(&label)
                    .expect("pointwise behavior label must have a merged target");
                builder.add_transition(label, target, weight);
            }
            builder
        })
        .collect()
}

/// Decoded token sets for one unique pointwise behavior region.
///
/// `final_tokens` is `None` when no token in the region is final-active.
/// `transitions` carries one entry per distinct label in the region; each
/// entry stores the mapped target alongside the token set. The target is
/// stored so every cache reuse can re-assert the existing
/// `targets_by_label` invariant without consulting another interner.
struct DecodedPointwiseRegion {
    final_tokens: Option<SharedTokenSet>,
    transitions: Vec<(Label, u32, SharedTokenSet)>,
}

fn decode_pointwise_region(
    region: &[TokenBehaviorRange],
    interner: &PointwiseBehaviorInterner,
) -> DecodedPointwiseRegion {
    let mut final_ranges = Vec::<std::ops::RangeInclusive<u32>>::new();
    let mut ranges_by_label = BTreeMap::<Label, (u32, Vec<std::ops::RangeInclusive<u32>>)>::new();
    for token_range in region.iter() {
        let behavior = interner.get(token_range.behavior);
        if behavior.final_active {
            final_ranges.push(token_range.start..=token_range.end);
        }
        for &(label, target) in &behavior.transitions {
            match ranges_by_label.entry(label) {
                std::collections::btree_map::Entry::Occupied(mut entry) => {
                    debug_assert_eq!(entry.get().0, target);
                    entry.get_mut().1.push(token_range.start..=token_range.end);
                }
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert((target, vec![token_range.start..=token_range.end]));
                }
            }
        }
    }
    let final_tokens = (!final_ranges.is_empty())
        .then(|| crate::ds::weight::shared_rangeset(RangeSetBlaze::from_iter(final_ranges)));
    let transitions = ranges_by_label
        .into_iter()
        .map(|(label, (target, ranges))| {
            (
                label,
                target,
                crate::ds::weight::shared_rangeset(RangeSetBlaze::from_iter(ranges)),
            )
        })
        .collect();
    DecodedPointwiseRegion {
        final_tokens,
        transitions,
    }
}

/// Materialize the exact union behavior already computed by pointwise coloring
/// directly into one builder per merge group.
///
/// Pointwise coloring maintains, for every `(tsid, token)` in a group's live
/// domain, the complete deterministic behavior: finality plus labelled mapped
/// targets. Re-expanding all member states after that proof and unioning their
/// weights again is redundant. Rebuilding weights from this partial function is
/// exactly the same union, expressed once per output group instead of once per
/// input member.
///
/// Regions are shared `Arc`s: every TSID with identical behavior points at
/// the same allocation. This decodes each unique region once per call
/// (keyed by the region pointer, which stays valid because `groups` is
/// borrowed for the whole call and the behavior interner is fixed for the
/// same call) and reuses the decoded token sets for every TSID occurrence
/// via [`Weight::from_per_tsid_shared`]. The per-TSID entries, TSID sort,
/// group targets, and coordinate IDs are unchanged.
fn pointwise_groups_to_builders(
    groups: &[PointwiseMergeGroup],
    interner: &PointwiseBehaviorInterner,
) -> Vec<MergedStateBuilder> {
    // Call-scoped only: groups hold every original Arc alive for the whole
    // call and `interner` is fixed for the same call, so a pointer key cannot
    // observe ABA reuse or a cross-context region.
    let mut decoded_by_region_ptr: FxHashMap<usize, DecodedPointwiseRegion> = FxHashMap::default();
    if std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
    {
        let materialize_started_at = Instant::now();
        let builders =
            pointwise_groups_to_builders_cached(groups, interner, &mut decoded_by_region_ptr);
        let materialize_ms = materialize_started_at.elapsed().as_secs_f64() * 1000.0;
        let region_occurrences = groups
            .iter()
            .map(|group| match &group.behavior_by_tsid {
                PointwiseBehaviorMap::Sparse(entries) => entries.len(),
                PointwiseBehaviorMap::Dense(entries) => entries.iter().flatten().count(),
            })
            .sum::<usize>();
        let (mut output_final_weights, mut output_transition_weights) = (0usize, 0usize);
        for builder in &builders {
            output_final_weights += builder.final_weights_pending.len();
            output_transition_weights += builder
                .transitions_pending
                .values()
                .map(|(_, pending)| pending.len())
                .sum::<usize>();
        }
        eprintln!(
            "[glrmask/profile][weighted_dwa_minimize_pointwise_materialize] groups={} region_occurrences={} unique_decodes={} output_final_weights={} output_transition_weights={} materialize_ms={:.3}",
            groups.len(),
            region_occurrences,
            decoded_by_region_ptr.len(),
            output_final_weights,
            output_transition_weights,
            materialize_ms,
        );
        return builders;
    }
    pointwise_groups_to_builders_cached(groups, interner, &mut decoded_by_region_ptr)
}

fn pointwise_groups_to_builders_cached(
    groups: &[PointwiseMergeGroup],
    interner: &PointwiseBehaviorInterner,
    decoded_by_region_ptr: &mut FxHashMap<usize, DecodedPointwiseRegion>,
) -> Vec<MergedStateBuilder> {
    groups
        .iter()
        .map(|group| {
            let mut entries = match &group.behavior_by_tsid {
                PointwiseBehaviorMap::Sparse(entries) => entries
                    .iter()
                    .map(|(&tsid, region)| (tsid, region))
                    .collect::<Vec<_>>(),
                PointwiseBehaviorMap::Dense(entries) => entries
                    .iter()
                    .enumerate()
                    .filter_map(|(tsid, region)| {
                        region.as_ref().map(|region| (tsid as u32, region))
                    })
                    .collect::<Vec<_>>(),
            };
            entries.sort_unstable_by_key(|(tsid, _)| *tsid);

            let mut final_entries = Vec::<(u32, SharedTokenSet)>::new();
            let mut transition_entries = BTreeMap::<Label, Vec<(u32, SharedTokenSet)>>::new();

            for (tsid, region) in entries {
                let region_ptr = Arc::as_ptr(region) as usize;
                let decoded = match decoded_by_region_ptr.entry(region_ptr) {
                    std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        entry.insert(decode_pointwise_region(region, interner))
                    }
                };
                if let Some(final_tokens) = decoded.final_tokens.as_ref() {
                    final_entries.push((tsid, Arc::clone(final_tokens)));
                }
                for (label, target, tokens) in &decoded.transitions {
                    debug_assert_eq!(group.targets_by_label.get(label), Some(target));
                    transition_entries
                        .entry(*label)
                        .or_default()
                        .push((tsid, Arc::clone(tokens)));
                }
            }

            let mut builder = MergedStateBuilder::default();
            let final_weight = Weight::from_per_tsid_shared(final_entries);
            if !final_weight.is_empty() {
                builder.add_final_weight(&final_weight);
            }
            for (label, entries) in transition_entries {
                let weight = Weight::from_per_tsid_shared(entries);
                if weight.is_empty() {
                    continue;
                }
                let target = *group
                    .targets_by_label
                    .get(&label)
                    .expect("pointwise behavior label must have a merged target");
                builder.add_transition(label, target, weight);
            }
            builder
        })
        .collect()
}

/// Build a reconstructed weight in one exact multiway sweep.
///
/// `Weight::union_all` already has a dedicated event-sweep implementation for
/// wide unions. Feeding it a tree of bounded intermediate unions adds repeated
/// sorting, allocation, and range coalescing without reducing the final work.
fn batch_build_weight(pending: Vec<Weight>) -> Weight {
    if reconstruction_token_range_coalescing_enabled() {
        Weight::union_all_reconstruction(pending.iter())
    } else {
        Weight::union_all(pending.iter())
    }
}

fn merge_state_into_builder(
    old_id: usize,
    color: usize,
    dwa: &DWA,
    old_to_new: &[u32],
    builders: &mut [MergedStateBuilder],
) {
    let builder = &mut builders[color];
    let old_state = &dwa.states()[old_id];

    // Union final weights
    if let Some(fw) = &old_state.final_weight {
        builder.add_final_weight(fw);
    }

    // Merge transitions
    let n = dwa.states().len();
    for (&label, (target_raw, w_orig)) in &old_state.transitions {
        let t = *target_raw as usize;
        if t >= n {
            continue;
        }
        let target_new = old_to_new[t];
        if target_new == UNMAPPED {
            continue;
        }
        // After push_weights, w_orig is already restricted to reachable[target].
        // The merged target's needed = union(reachable[s] for all s merged into target_new),
        // which is a superset of reachable[target]. So w_orig ∩ merged_needed = w_orig.
        if !w_orig.is_empty() {
            builder.add_transition(label, target_new, w_orig.clone());
        }
    }
}

fn merge_productive_state_into_builder(
    old_id: usize,
    color: usize,
    dwa: &DWA,
    productive_transitions: &[Vec<ProductiveTransition>],
    old_to_new: &[u32],
    builders: &mut [MergedStateBuilder],
) {
    let builder = &mut builders[color];
    let old_state = &dwa.states()[old_id];
    if let Some(final_weight) = &old_state.final_weight {
        builder.add_final_weight(final_weight);
    }
    for transition in &productive_transitions[old_id] {
        let target = mapped_target(old_to_new, transition.target)
            .expect("productive transition target must be colored first");
        builder.add_transition(transition.label, target, transition.weight.clone());
    }
}

fn reconstruct_dwa(start_old: usize, old_to_new: &[u32], builders: Vec<MergedStateBuilder>) -> DWA {
    let profile_enabled = weighted_dwa_minimize_profile_enabled();
    let mut final_pending_weight_count = 0usize;
    let mut max_final_pending_weight_count = 0usize;
    let mut final_batches_over_16 = 0usize;
    let mut transition_batch_count = 0usize;
    let mut transition_pending_weight_count = 0usize;
    let mut max_transition_pending_weight_count = 0usize;
    let mut transition_batches_over_16 = 0usize;
    let mut final_union_ms = 0.0;
    let mut transition_union_ms = 0.0;
    let mut insert_ms = 0.0;
    let states: Vec<DWAState> = builders
        .into_iter()
        .map(|b| {
            let mut state = DWAState::default();
            final_pending_weight_count += b.final_weights_pending.len();
            max_final_pending_weight_count = max_final_pending_weight_count.max(b.final_weights_pending.len());
            final_batches_over_16 += usize::from(b.final_weights_pending.len() > 16);
            let final_union_started_at = Instant::now();
            let final_weight = batch_build_weight(b.final_weights_pending);
            final_union_ms += final_union_started_at.elapsed().as_secs_f64() * 1000.0;
            if !final_weight.is_empty() {
                state.final_weight = Some(final_weight);
            }
            for (lbl, (target, pending_weights)) in b.transitions_pending {
                transition_batch_count += 1;
                transition_pending_weight_count += pending_weights.len();
                max_transition_pending_weight_count = max_transition_pending_weight_count.max(pending_weights.len());
                transition_batches_over_16 += usize::from(pending_weights.len() > 16);
                let transition_union_started_at = Instant::now();
                let weight = batch_build_weight(pending_weights);
                transition_union_ms += transition_union_started_at.elapsed().as_secs_f64() * 1000.0;
                if !weight.is_empty() {
                    let insert_started_at = Instant::now();
                    state.transitions.insert(lbl, (target, weight));
                    insert_ms += insert_started_at.elapsed().as_secs_f64() * 1000.0;
                }
            }
            state
        })
        .collect();

    if profile_enabled {
        eprintln!(
            "[glrmask/profile][weighted_dwa_minimize_reconstruct] output_states={} final_pending_weights={} max_final_pending_weights={} final_batches_over_16={} final_union_ms={:.3} transition_batches={} transition_pending_weights={} max_transition_pending_weights={} transition_batches_over_16={} transition_union_ms={:.3} insert_ms={:.3}",
            states.len(),
            final_pending_weight_count,
            max_final_pending_weight_count,
            final_batches_over_16,
            final_union_ms,
            transition_batch_count,
            transition_pending_weight_count,
            max_transition_pending_weight_count,
            transition_batches_over_16,
            transition_union_ms,
            insert_ms,
        );
    }

    let start_new = old_to_new[start_old];
    DWA::from_parts(
        states,
        if start_new == UNMAPPED { 0 } else { start_new },
    )
}

fn canonical_dead_dwa() -> DWA {
    DWA::new(0, 0)
}

/// Exact small-input path that never materializes a transient pushed DWA.
///
/// The ordinary pipeline computes each productive transition, writes it back
/// into the state's BTreeMap, scans those maps again to build productive
/// profiles, and finally clones the same weights into reconstruction builders.
/// For small height buckets we already use exact pairwise coloring, so the
/// pushed maps are not needed for partition refinement. Keep the productive
/// transitions in one compact side vector and reconstruct directly from it.
fn try_minimize_small_pairwise_direct(dwa: &DWA) -> Option<DWA> {
    // The quadratic work is governed by the largest non-leaf height bucket,
    // not by the total number of states.  Structural L2P DWAs commonly have a
    // large collection of height-0 leaves plus several modest upper buckets;
    // rejecting those at 128 total states needlessly falls back to the generic
    // push/materialize/rescan pipeline even though every pairwise-coloring
    // bucket remains small.  Keep a conservative total-size guard for now, but
    // admit the common 129-192-state regime while retaining the independent
    // MAX_HEIGHT_BUCKET bound below.
    const MAX_INPUT_STATES: usize = 192;
    const MAX_HEIGHT_BUCKET: usize = 64;

    let n = dwa.states().len();
    if n == 0 || n > MAX_INPUT_STATES {
        return None;
    }
    let started_at = Instant::now();
    let topo = compute_topo_order(dwa)?;

    let mut incoming_transition_counts = vec![0usize; n];
    for state in dwa.states() {
        for (target, _) in state.transitions.values() {
            if (*target as usize) < n {
                incoming_transition_counts[*target as usize] += 1;
            }
        }
    }

    let mut needed = vec![Weight::empty(); n];
    let mut productive_transitions = vec![Vec::<ProductiveTransition>::new(); n];
    let mut intersection_cache = FxHashMap::<(usize, usize), Weight>::default();
    let mut intersection_indexes = FxHashMap::<usize, WeightIntersectionIndex>::default();
    for &source in topo.iter().rev() {
        let state = &dwa.states()[source];
        let mut reachable_parts = Vec::<Weight>::with_capacity(state.transitions.len() + 1);
        let mut reachable_is_full = false;
        if let Some(final_weight) = &state.final_weight {
            if final_weight.is_full() {
                reachable_is_full = true;
            } else if !final_weight.is_empty() {
                reachable_parts.push(final_weight.clone());
            }
        }

        let mut productive = Vec::with_capacity(state.transitions.len());
        for (&label, (target, weight)) in &state.transitions {
            let target_index = *target as usize;
            if target_index >= n || needed[target_index].is_empty() {
                continue;
            }
            let pushed_weight = if needed[target_index].is_full() {
                weight.clone()
            } else {
                let index = (incoming_transition_counts[target_index] >= 8).then(|| {
                    let key = needed[target_index].ptr_key();
                    intersection_indexes
                        .entry(key)
                        .or_insert_with(|| needed[target_index].intersection_index())
                });
                memoized_intersection(
                    &mut intersection_cache,
                    weight,
                    &needed[target_index],
                    index.as_deref(),
                )
            };
            if pushed_weight.is_empty() {
                continue;
            }
            if !reachable_is_full {
                if pushed_weight.is_full() {
                    reachable_is_full = true;
                    reachable_parts.clear();
                } else {
                    reachable_parts.push(pushed_weight.clone());
                }
            }
            productive.push(ProductiveTransition {
                label,
                target: *target,
                weight: pushed_weight,
            });
        }
        productive_transitions[source] = productive;
        needed[source] = if reachable_is_full {
            Weight::all()
        } else {
            Weight::union_all(reachable_parts.iter())
        };
    }

    let start_state = dwa.start_state() as usize;
    if start_state >= n || needed[start_state].is_empty() {
        return Some(canonical_dead_dwa());
    }

    let mut heights = vec![0usize; n];
    for &source in topo.iter().rev() {
        heights[source] = productive_transitions[source]
            .iter()
            .map(|transition| heights[transition.target as usize] + 1)
            .max()
            .unwrap_or(0);
    }
    let max_height = heights.iter().copied().max().unwrap_or(0);

    let mut reachable_from_start = vec![false; n];
    let mut stack = vec![start_state];
    while let Some(source) = stack.pop() {
        if reachable_from_start[source] {
            continue;
        }
        reachable_from_start[source] = true;
        stack.extend(
            productive_transitions[source]
                .iter()
                .map(|transition| transition.target as usize),
        );
    }

    let mut states_by_height = vec![Vec::<usize>::new(); max_height + 1];
    for state in 0..n {
        if reachable_from_start[state] && !needed[state].is_empty() {
            states_by_height[heights[state]].push(state);
        }
    }
    // Height 0 is handled by the all-compatible leaf fast path below and
    // never enters pairwise coloring. Bound only the non-leaf buckets that can
    // actually pay O(bucket^2) compatibility work. Applying this limit to the
    // leaf bucket made 65+ leaf automata do all small-direct preparation, reject,
    // and then repeat the same weight-pushing work in the generic minimizer.
    if states_by_height
        .iter()
        .skip(1)
        .any(|bucket| bucket.len() > MAX_HEIGHT_BUCKET)
    {
        return None;
    }

    let mut old_to_new = vec![UNMAPPED; n];
    let mut new_states = Vec::<MergedStateBuilder>::new();
    for (height, candidates) in states_by_height.iter().enumerate() {
        if candidates.is_empty() {
            continue;
        }
        let coloring = if height == 0
            && candidates
                .iter()
                .all(|&state| productive_transitions[state].is_empty())
        {
            vec![0usize; candidates.len()]
        } else {
            build_and_color_pairwise_greedy(
                dwa,
                candidates,
                &needed,
                &old_to_new,
                &productive_transitions,
            )
        };
        let base_new_id = new_states.len() as u32;
        let num_colors = coloring.iter().copied().max().map_or(0, |color| color + 1);
        for (index, &color) in coloring.iter().enumerate() {
            old_to_new[candidates[index]] = base_new_id + color as u32;
        }
        new_states.extend((0..num_colors).map(|_| MergedStateBuilder::default()));
        let builders = &mut new_states[base_new_id as usize..];
        for (index, &color) in coloring.iter().enumerate() {
            merge_productive_state_into_builder(
                candidates[index],
                color,
                dwa,
                &productive_transitions,
                &old_to_new,
                builders,
            );
        }
    }

    let minimized = reconstruct_dwa(start_state, &old_to_new, new_states);
    if weighted_dwa_minimize_profile_enabled() {
        eprintln!(
            "[glrmask/profile][weighted_dwa_minimize_small_direct] input_states={} input_transitions={} output_states={} output_transitions={} total_ms={:.3}",
            dwa.num_states(),
            dwa.num_transitions(),
            minimized.num_states(),
            minimized.num_transitions(),
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Some(minimized)
}

// Public API.

/// Minimize an acyclic DWA using weight pushing + graph-coloring.
///
/// Falls back to the caller's DWA unchanged if the input is cyclic.
pub fn minimize_acyclic(dwa: &DWA) -> DWA {
    minimize_acyclic_owned(dwa.clone())
}

pub fn minimize_acyclic_owned(pushed: DWA) -> DWA {
    minimize_acyclic_owned_with_pointwise_class_order(pushed, PointwiseClassOrder::Stable)
}

/// Minimize a DWA produced from an already backward-pushed, path-conditioned
/// automaton. Determinization carries cumulative path domains on incoming
/// edges, so those domains can replace a second backward-needed union pass.
pub fn minimize_acyclic_owned_path_conditioned(pushed: DWA) -> DWA {
    minimize_acyclic_owned_impl(pushed, PointwiseClassOrder::Stable, true)
}

pub fn minimize_acyclic_owned_with_pointwise_class_order(
    pushed: DWA,
    pointwise_class_order: PointwiseClassOrder,
) -> DWA {
    minimize_acyclic_owned_impl(pushed, pointwise_class_order, false)
}

fn minimize_acyclic_owned_impl(
    mut pushed: DWA,
    pointwise_class_order: PointwiseClassOrder,
    use_forward_domains: bool,
) -> DWA {
    if pushed.states().is_empty() {
        return pushed;
    }
    if std::env::var_os("GLRMASK_WEIGHTED_MINIMIZE_DISABLE_SMALL_DIRECT").is_none()
        && let Some(minimized) = try_minimize_small_pairwise_direct(&pushed)
    {
        return minimized;
    }

    let strict_forward_reference =
        (use_forward_domains
            && std::env::var_os("GLRMASK_ASSERT_WEIGHTED_MINIMIZE_FORWARD_DOMAINS").is_some())
        .then(|| pushed.clone());
    let profile_enabled = weighted_dwa_minimize_profile_enabled();
    let total_started_at = Instant::now();
    let input_states = pushed.num_states();
    let input_transitions = pushed.num_transitions();

    let push_started_at = Instant::now();
    let (topo, needed) = if use_forward_domains {
        match compute_forward_path_domains(&pushed) {
            Some(result) => result,
            None => return pushed,
        }
    } else {
        let (_, topo_from_push, reachable_from_push) = push_weights(&mut pushed);
        let topo = match topo_from_push {
            Some(t) => t,
            None => return pushed,
        };
        (topo, reachable_from_push)
    };
    let push_ms = push_started_at.elapsed().as_secs_f64() * 1000.0;

    let start_state = pushed.start_state() as usize;
    #[cfg(debug_assertions)]
    debug_assert_pushed_weights_within_needed(&pushed, &needed);
    if needed[start_state].is_empty() {
        return canonical_dead_dwa();
    }
    let productive_transitions_started_at = Instant::now();
    let productive_transitions = compute_productive_transitions(&pushed, &needed);
    let productive_transitions_ms =
        productive_transitions_started_at.elapsed().as_secs_f64() * 1000.0;
    let heights_started_at = Instant::now();
    let heights = compute_heights(&pushed, &topo);
    let heights_ms = heights_started_at.elapsed().as_secs_f64() * 1000.0;
    let max_height = heights.iter().max().copied().unwrap_or(0);

    let n = pushed.states().len();

    let reachable_from_start_started_at = Instant::now();
    let reachable_from_start = compute_reachable_from_start(&pushed, start_state);
    let reachable_from_start_ms =
        reachable_from_start_started_at.elapsed().as_secs_f64() * 1000.0;

    // Group states by height (only reachable states with non-empty needed sets)
    let group_by_height_started_at = Instant::now();
    let mut states_by_height: Vec<Vec<usize>> = vec![vec![]; max_height + 1];
    for (id, &h) in heights.iter().enumerate() {
        if !reachable_from_start[id] {
            continue;
        }
        if needed[id].is_empty() {
            continue;
        }
        states_by_height[h].push(id);
    }
    let group_by_height_ms = group_by_height_started_at.elapsed().as_secs_f64() * 1000.0;
    let active_states = states_by_height.iter().map(Vec::len).sum::<usize>();
    let max_bucket_size = states_by_height.iter().map(Vec::len).max().unwrap_or(0);

    // Bottom-up: color and merge
    let mut old_to_new = vec![UNMAPPED; n];
    let mut new_states: Vec<MergedStateBuilder> = Vec::new();
    let mut color_ms_total = 0.0;
    let mut merge_ms_total = 0.0;

    for h in 0..=max_height {
        let candidates = &states_by_height[h];
        if candidates.is_empty() {
            continue;
        }
        let height_started_at = Instant::now();
        let mut height_color_ms = 0.0;
        let mut fast_height0 = false;

        if h == 0 {
            let color_started_at = Instant::now();
            let all_compatible = try_all_compatible_height_0_coloring(candidates, &pushed, &needed).is_some();
            height_color_ms = color_started_at.elapsed().as_secs_f64() * 1000.0;
            if all_compatible {
                fast_height0 = true;
                let merge_started_at = Instant::now();
                let base_new_id = new_states.len() as u32;
                let num_colors = 1usize;

                for &candidate in candidates {
                    old_to_new[candidate] = base_new_id;
                }

                new_states.extend((0..num_colors).map(|_| MergedStateBuilder::default()));

                let builders = &mut new_states[base_new_id as usize..];
                for &candidate in candidates {
                    merge_state_into_builder(
                        candidate,
                        0,
                        &pushed,
                        &old_to_new,
                        builders,
                    );
                }
                let height_merge_ms = merge_started_at.elapsed().as_secs_f64() * 1000.0;
                color_ms_total += height_color_ms;
                merge_ms_total += height_merge_ms;
                if profile_enabled {
                    eprintln!(
                        "[glrmask/profile][weighted_dwa_minimize_height] height={} candidates={} colors={} fast_height0={} color_ms={:.3} merge_ms={:.3} total_ms={:.3}",
                        h,
                        candidates.len(),
                        num_colors,
                        fast_height0,
                        height_color_ms,
                        height_merge_ms,
                        height_started_at.elapsed().as_secs_f64() * 1000.0,
                    );
                }
                continue;
            }
        }

        let color_started_at = Instant::now();
        let coloring_result = build_and_color_hybrid(
            &pushed,
            candidates,
            &needed,
            &old_to_new,
            &productive_transitions,
            pointwise_class_order,
            use_forward_domains
                && std::env::var_os("GLRMASK_DISABLE_MONOTONE_POINTWISE").is_none(),
        );
        let coloring = &coloring_result.coloring;
        height_color_ms += color_started_at.elapsed().as_secs_f64() * 1000.0;

        let merge_started_at = Instant::now();
        let base_new_id = new_states.len() as u32;
        let num_colors = coloring.iter().max().map(|&c| c + 1).unwrap_or(0);

        // Map old states to new merged states
        for (idx, &color) in coloring.iter().enumerate() {
            old_to_new[candidates[idx]] = base_new_id + color as u32;
        }

        if let Some(direct_builders) = coloring_result.direct_builders {
            debug_assert_eq!(direct_builders.len(), num_colors);
            new_states.extend(direct_builders);
        } else {
            // Extend builders
            new_states.extend((0..num_colors).map(|_| MergedStateBuilder::default()));

            // Merge states into builders
            let builders = &mut new_states[base_new_id as usize..];
            for (idx, &color) in coloring.iter().enumerate() {
                merge_state_into_builder(
                    candidates[idx],
                    color,
                    &pushed,
                    &old_to_new,
                    builders,
                );
            }
        }
        let height_merge_ms = merge_started_at.elapsed().as_secs_f64() * 1000.0;
        color_ms_total += height_color_ms;
        merge_ms_total += height_merge_ms;

        if profile_enabled {
            eprintln!(
                "[glrmask/profile][weighted_dwa_minimize_height] height={} candidates={} colors={} fast_height0={} color_ms={:.3} merge_ms={:.3} total_ms={:.3}",
                h,
                candidates.len(),
                num_colors,
                fast_height0,
                height_color_ms,
                height_merge_ms,
                height_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }

    let reconstruct_started_at = Instant::now();
    let minimized = reconstruct_dwa(start_state, &old_to_new, new_states);
    let reconstruct_ms = reconstruct_started_at.elapsed().as_secs_f64() * 1000.0;

    if profile_enabled {
        eprintln!(
            "[glrmask/profile][weighted_dwa_minimize] input_states={} input_transitions={} push_ms={:.3} productive_transitions_ms={:.3} heights_ms={:.3} reachable_from_start_ms={:.3} group_by_height_ms={:.3} color_ms_total={:.3} merge_ms_total={:.3} reconstruct_ms={:.3} max_height={} active_states={} max_bucket_size={} output_states={} output_transitions={} total_ms={:.3}",
            input_states,
            input_transitions,
            push_ms,
            productive_transitions_ms,
            heights_ms,
            reachable_from_start_ms,
            group_by_height_ms,
            color_ms_total,
            merge_ms_total,
            reconstruct_ms,
            max_height,
            active_states,
            max_bucket_size,
            minimized.num_states(),
            minimized.num_transitions(),
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }

    if let Some(reference) = strict_forward_reference {
        assert_eq!(
            find_difference(&minimized, &reference)
                .expect("forward-domain minimizer equivalence check failed"),
            None,
            "forward-domain minimizer changed weighted language",
        );
    }

    minimized
}

#[cfg(test)]
mod tests ;
