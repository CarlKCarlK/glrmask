//! Lazy zero-minimum repeat/suffix states and dominance-aware materialization.

use crate::automata::lexer::ast::Expr;
use crate::automata::lexer::dfa::DFA;

use crate::ds::bitset::BitSet;
use crate::ds::u8set::U8Set;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use std::collections::VecDeque;
use std::sync::Arc;
use super::bounded_repeat::collect_suffix_bytes;
use super::dfa_analysis::{dfa_transition_count, refine_u8_partitions};
use super::factor::{seq_from_parts, unwrap_shared};
use super::mapping::ZeroMinRepeatSuffixComponentTrace;
use super::nfa::{compile_expr_to_dfa, expr_u8set};
use super::product::ProductComponent;

/// TODO: replace this compact product with an exact subset construction if we
/// need to support ambiguous body/suffix boundaries or nullable suffixes in the
/// fast path. Until then, this function must return None for any case requiring
/// multiple simultaneous boundary choices.
///
/// Builds a DFA for `Seq([Repeat{body, min, max}, suffix_exprs...])` using a
/// product construction of body_DFA × suffix_DFA × completion_counter.
///
/// Handles cases where the suffix is a regex (not just bytes) and/or the body
/// is not prefix-free, which `build_bounded_repeat_with_suffix_dfa` cannot handle.
/// Avoids the exponential NFA→DFA blowup that occurs with unrolled bounded repeats.
///
/// The product state is `(body_state, suffix_state, counter)`:
///   - body tracks progress through the repeat body expression
///   - suffix tracks the suffix match (started at body boundaries when counter >= min)
///   - counter tracks completed body repetitions (0..max)
///
/// This is a fast path, not a general NFA subset construction. It must return
/// `None` whenever the compact state would need to represent multiple live
/// body/suffix boundary choices.
///
/// At body completion, the counter increments and the suffix may start. If two
/// live paths cannot be represented by one `(body_state, suffix_state, counter)`
/// tuple, this function falls back to the general compiler.
#[derive(Clone, PartialEq, Eq, Hash)]
pub(super) struct ZeroMinRepeatSuffixState {
    /// Minimum completed-copy count reaching each body DFA residual. A smaller
    /// count dominates every larger count at the same residual.
    pub(super) body_min_counts: Box<[u32]>,
    /// Exact live suffix residuals. Count is irrelevant once the suffix starts.
    pub(super) suffix_states: Box<[u32]>,
}

pub(super) fn close_zero_min_repeat_suffix_state(
    body_min_counts: &mut [u32],
    suffix_states: &mut Vec<u32>,
    body_dfa: &DFA,
    suffix_dfa: &DFA,
    max: usize,
) {
    let mut completed_boundary = u32::MAX;
    for (body_state, &completed) in body_min_counts.iter().enumerate() {
        if completed == u32::MAX {
            continue;
        }
        if body_dfa.finalizers(body_state as u32).contains(0) {
            completed_boundary = completed_boundary.min(completed.saturating_add(1));
        }
    }

    if completed_boundary != u32::MAX {
        suffix_states.push(0);
        if completed_boundary < max as u32 {
            body_min_counts[0] = body_min_counts[0].min(completed_boundary);
        }
    }

    // Keep only residuals that can consume at least one more byte toward a
    // body match. Acceptance at the current body state has already spawned the
    // repeat/suffix boundary above.
    for (body_state, completed) in body_min_counts.iter_mut().enumerate() {
        if *completed != u32::MAX
            && !body_dfa
                .possible_future_group_ids(body_state as u32)
                .contains(0)
        {
            *completed = u32::MAX;
        }
    }

    suffix_states.retain(|&state| {
        suffix_dfa.finalizers(state).contains(0)
            || suffix_dfa.possible_future_group_ids(state).contains(0)
    });
    suffix_states.sort_unstable();
    suffix_states.dedup();
}

fn set_zero_min_repeat_suffix_metadata(
    dfa: &mut DFA,
    state_id: u32,
    state: &ZeroMinRepeatSuffixState,
    suffix_dfa: &DFA,
) {
    let mut finalizers = BitSet::new(1);
    if state
        .suffix_states
        .iter()
        .any(|&suffix_state| suffix_dfa.finalizers(suffix_state).contains(0))
    {
        finalizers.set(0);
    }
    let mut future = BitSet::new(1);
    if state.body_min_counts.iter().any(|&count| count != u32::MAX)
        || state
            .suffix_states
            .iter()
            .any(|&suffix_state| {
                suffix_dfa
                    .possible_future_group_ids(suffix_state)
                    .contains(0)
            })
    {
        future.set(0);
    }
    dfa.overwrite_state_metadata(state_id, finalizers, future);
}

pub(super) struct ZeroMinRepeatSuffixBuild {
    pub(super) dfa: DFA,
    pub(super) states: Vec<ZeroMinRepeatSuffixState>,
    pub(super) state_by_key: FxHashMap<ZeroMinRepeatSuffixState, u32>,
}

/// Lazily interns the exact dominance states for a zero-minimum bounded
/// repeat followed by a non-nullable suffix. Product construction can then
/// visit only component residuals that survive the other intersection
/// coordinate instead of materializing every repeat-count layer first.
pub(super) struct LazyZeroMinRepeatSuffixComponent {
    pub(super) prefix: Vec<u8>,
    pub(super) body_dfa: DFA,
    pub(super) suffix_dfa: DFA,
    pub(super) max: usize,
    pub(super) group_u8set: U8Set,
    pub(super) tail_states: Vec<ZeroMinRepeatSuffixState>,
    pub(super) tail_state_by_key: FxHashMap<ZeroMinRepeatSuffixState, u32>,
    pub(super) class_count: usize,
    pub(super) class_targets: Vec<u32>,
}

pub(super) const LAZY_ZERO_MIN_REPEAT_SUFFIX_MIN_BOUND: usize = 1_024;

impl LazyZeroMinRepeatSuffixComponent {
    pub(super) fn from_expr(expr: &Expr) -> Option<Self> {
        let unwrapped = unwrap_shared(expr);
        let Expr::Seq(parts) = unwrapped else {
            return None;
        };
        let mut flat_parts = Vec::<Expr>::new();
        for part in parts {
            match unwrap_shared(part) {
                Expr::Seq(inner) => flat_parts.extend(inner.iter().cloned()),
                _ => flat_parts.push(part.clone()),
            }
        }

        for repeat_index in 0..flat_parts.len().saturating_sub(1) {
            let Expr::Repeat {
                expr: body,
                min: 0,
                max: Some(max),
            } = unwrap_shared(&flat_parts[repeat_index])
            else {
                continue;
            };
            // The lazy product is intended for very large bounded residuals
            // whose eager constituent DFA would dominate compilation. Smaller
            // repeats are already cheap to materialize and preserve better
            // downstream state locality through the ordinary product path.
            if *max < LAZY_ZERO_MIN_REPEAT_SUFFIX_MIN_BOUND {
                continue;
            }
            // `u32::MAX` is the unreachable sentinel in body_min_counts, so it
            // cannot also be a real completed-copy count. Reject it (and any
            // larger usize bound) before compiling either body or suffix.
            if *max >= u32::MAX as usize {
                return None;
            }
            let prefix = if repeat_index == 0 {
                Vec::new()
            } else {
                collect_suffix_bytes(&flat_parts[..repeat_index])?
            };
            let suffix_expr = seq_from_parts(flat_parts[repeat_index + 1..].to_vec());
            let body_dfa = compile_expr_to_dfa(body);
            let suffix_dfa = compile_expr_to_dfa(&suffix_expr);
            if body_dfa.num_states() == 0
                || body_dfa.num_states() > 256
                || body_dfa.finalizers(0).contains(0)
                || suffix_dfa.num_states() == 0
                || suffix_dfa.finalizers(0).contains(0)
            {
                continue;
            }

            let mut start_body = vec![u32::MAX; body_dfa.num_states()];
            start_body[0] = 0;
            let mut start_suffix = vec![0u32];
            close_zero_min_repeat_suffix_state(
                &mut start_body,
                &mut start_suffix,
                &body_dfa,
                &suffix_dfa,
                *max,
            );
            let start = ZeroMinRepeatSuffixState {
                body_min_counts: start_body.into_boxed_slice(),
                suffix_states: start_suffix.into_boxed_slice(),
            };
            let mut tail_state_by_key = FxHashMap::default();
            tail_state_by_key.insert(start.clone(), 0);
            return Some(Self {
                prefix,
                body_dfa,
                suffix_dfa,
                max: *max,
                group_u8set: expr_u8set(expr),
                tail_states: vec![start],
                tail_state_by_key,
                class_count: 0,
                class_targets: Vec::new(),
            });
        }
        None
    }

    pub(super) fn prefix_len(&self) -> usize {
        self.prefix.len()
    }

    pub(super) fn num_states(&self) -> usize {
        self.prefix.len() + self.tail_states.len()
    }

    pub(super) fn start_state(&self) -> u32 {
        0
    }

    pub(super) fn prepare_classes(&mut self, class_count: usize) {
        debug_assert_eq!(self.class_count, 0);
        debug_assert!(self.class_targets.is_empty());
        self.class_count = class_count;
        self.class_targets = vec![u32::MAX; self.num_states().saturating_mul(class_count)];
    }

    pub(super) fn is_accepting(&self, state: u32) -> bool {
        let Some(tail) = (state as usize)
            .checked_sub(self.prefix.len())
            .and_then(|index| self.tail_states.get(index))
        else {
            return false;
        };
        tail.suffix_states
            .iter()
            .any(|&suffix_state| self.suffix_dfa.finalizers(suffix_state).contains(0))
    }

    pub(super) fn has_future(&self, state: u32) -> bool {
        if (state as usize) < self.prefix.len() {
            return true;
        }
        let tail = &self.tail_states[state as usize - self.prefix.len()];
        tail.body_min_counts.iter().any(|&count| count != u32::MAX)
            || tail.suffix_states.iter().any(|&suffix_state| {
                self.suffix_dfa
                    .possible_future_group_ids(suffix_state)
                    .contains(0)
            })
    }

    pub(super) fn step_class(&mut self, state: u32, class: u8, byte: u8) -> Option<u32> {
        let cache_index = (state as usize)
            .checked_mul(self.class_count)?
            .checked_add(class as usize)?;
        let cached = *self.class_targets.get(cache_index)?;
        // `u32::MAX - 1` is the cached dead-transition sentinel. The real DFA
        // state space is bounded far below it by the surrounding product.
        const DEAD: u32 = u32::MAX - 1;
        if cached == DEAD {
            return None;
        }
        if cached != u32::MAX {
            return Some(cached);
        }
        let target = self.step_uncached(state, byte).unwrap_or(DEAD);
        self.class_targets[cache_index] = target;
        if target == DEAD {
            None
        } else {
            Some(target)
        }
    }

    pub(super) fn step_uncached(&mut self, state: u32, byte: u8) -> Option<u32> {
        let state = state as usize;
        if state < self.prefix.len() {
            if self.prefix[state] != byte {
                return None;
            }
            return Some((state + 1) as u32);
        }

        let tail_index = state - self.prefix.len();
        let next = {
            let current = self.tail_states.get(tail_index)?;
            let mut next_body = vec![u32::MAX; self.body_dfa.num_states()];
            for (body_state, &completed) in current.body_min_counts.iter().enumerate() {
                if completed == u32::MAX {
                    continue;
                }
                if let Some(target) = self.body_dfa.step(body_state as u32, byte) {
                    let target_count = &mut next_body[target as usize];
                    *target_count = (*target_count).min(completed);
                }
            }
            let mut next_suffix = Vec::with_capacity(current.suffix_states.len());
            for &suffix_state in current.suffix_states.iter() {
                if let Some(target) = self.suffix_dfa.step(suffix_state, byte) {
                    next_suffix.push(target);
                }
            }
            close_zero_min_repeat_suffix_state(
                &mut next_body,
                &mut next_suffix,
                &self.body_dfa,
                &self.suffix_dfa,
                self.max,
            );
            if next_body.iter().all(|&count| count == u32::MAX) && next_suffix.is_empty() {
                return None;
            }
            ZeroMinRepeatSuffixState {
                body_min_counts: next_body.into_boxed_slice(),
                suffix_states: next_suffix.into_boxed_slice(),
            }
        };
        let local = if let Some(&local) = self.tail_state_by_key.get(&next) {
            local
        } else {
            let local = u32::try_from(self.tail_states.len()).ok()?;
            self.tail_state_by_key.insert(next.clone(), local);
            self.tail_states.push(next);
            self.class_targets
                .extend(std::iter::repeat_n(u32::MAX, self.class_count));
            local
        };
        Some(self.prefix.len() as u32 + local)
    }

    pub(super) fn into_product_component(
        self,
        class_members: &[Vec<u8>],
    ) -> ProductComponent {
        let mut dfa = DFA::new(self.num_states());
        dfa.ensure_group_capacity(1);
        dfa.set_group_u8set(0, self.group_u8set);
        for state in 0..self.num_states() {
            let mut finalizers = BitSet::new(1);
            if self.is_accepting(state as u32) {
                finalizers.set(0);
            }
            let mut future = BitSet::new(1);
            if self.has_future(state as u32) {
                future.set(0);
            }
            dfa.overwrite_state_metadata(state as u32, finalizers, future);
            let mut transitions = Vec::new();
            let row_start = state * self.class_count;
            for class in 0..self.class_count {
                let target = self.class_targets[row_start + class];
                if target >= u32::MAX - 1 {
                    continue;
                }
                transitions.extend(
                    class_members[class]
                        .iter()
                        .copied()
                        .map(|byte| (byte, target)),
                );
            }
            transitions.sort_unstable_by_key(|entry| entry.0);
            dfa.set_transitions_from_sorted_entries(state as u32, transitions);
        }
        let dfa = Arc::new(dfa);
        let trace = ZeroMinRepeatSuffixComponentTrace {
            dfa: Arc::clone(&dfa),
            prefix_len: self.prefix.len(),
            body_dfa: self.body_dfa,
            suffix_dfa: self.suffix_dfa,
            max: self.max,
            tail_states: self.tail_states,
            tail_state_by_key: self.tail_state_by_key,
        };
        ProductComponent::MaterializedZeroMinRepeatSuffix {
            dfa,
            trace: Arc::new(trace),
        }
    }
}

pub(super) fn compute_dfa_byte_equivalence_classes(dfas: &[&DFA]) -> (Vec<u8>, Vec<Vec<u8>>) {
    let mut partitions = vec![U8Set::all()];
    let mut seen_sets = FxHashSet::default();

    for dfa in dfas {
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

pub(super) fn compute_lazy_zero_min_repeat_product_equivalence_classes(
    lazy: &LazyZeroMinRepeatSuffixComponent,
    other: &DFA,
) -> (Vec<u8>, Vec<Vec<u8>>) {
    let (_, class_members) = compute_dfa_byte_equivalence_classes(&[
        &lazy.body_dfa,
        &lazy.suffix_dfa,
        other,
    ]);
    let mut partitions = class_members
        .into_iter()
        .map(|bytes| {
            let mut set = U8Set::empty();
            for byte in bytes {
                set.insert(byte);
            }
            set
        })
        .collect::<Vec<_>>();
    for &byte in &lazy.prefix {
        partitions = refine_u8_partitions(partitions, U8Set::single(byte));
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

pub(super) fn build_zero_min_repeat_suffix_dominance_dfa_internal(
    body_dfa: &DFA,
    suffix_dfa: &DFA,
    max: usize,
    preserve_coordinates: bool,
) -> Option<ZeroMinRepeatSuffixBuild> {
    if max == 0
        || body_dfa.num_states() == 0
        || suffix_dfa.num_states() == 0
        || body_dfa.finalizers(0).contains(0)
        || suffix_dfa.finalizers(0).contains(0)
    {
        return None;
    }

    let mut start_body = vec![u32::MAX; body_dfa.num_states()];
    start_body[0] = 0;
    let mut start_suffix = vec![0u32];
    close_zero_min_repeat_suffix_state(
        &mut start_body,
        &mut start_suffix,
        body_dfa,
        suffix_dfa,
        max,
    );
    let start = ZeroMinRepeatSuffixState {
        body_min_counts: start_body.into_boxed_slice(),
        suffix_states: start_suffix.into_boxed_slice(),
    };

    let mut dfa = DFA::new(1);
    dfa.ensure_group_capacity(1);
    set_zero_min_repeat_suffix_metadata(&mut dfa, 0, &start, suffix_dfa);
    let mut state_map = FxHashMap::<ZeroMinRepeatSuffixState, u32>::default();
    state_map.insert(start.clone(), 0);
    let mut states = vec![start.clone()];
    let mut worklist = VecDeque::from([(0u32, start)]);
    let (byte_to_class, class_members) =
        compute_dfa_byte_equivalence_classes(&[body_dfa, suffix_dfa]);
    // Dominance states are dense only as an externally visible coordinate.
    // In practice their live body frontier is tiny (typically one or two
    // residuals), while the body DFA can have dozens or hundreds of states.
    // Reuse one dense scratch row and touch only live residuals while stepping;
    // materialize a boxed dense row only for a live class edge that must be
    // interned. This preserves the historical state key exactly while avoiding
    // one allocation + full body scan for every state/class attempt.
    let mut next_body_scratch = vec![u32::MAX; body_dfa.num_states()];

    while let Some((state_id, state)) = worklist.pop_front() {
        let live_body = state
            .body_min_counts
            .iter()
            .enumerate()
            .filter_map(|(body_state, &completed)| {
                (completed != u32::MAX).then_some((body_state, completed))
            })
            .collect::<SmallVec<[(usize, u32); 8]>>();
        let mut target_by_class = vec![u32::MAX; class_members.len()];
        for (class, members) in class_members.iter().enumerate() {
            let byte = members[0];
            let mut touched_body = SmallVec::<[usize; 8]>::new();
            for &(body_state, completed) in &live_body {
                if let Some(target) = body_dfa.step(body_state as u32, byte) {
                    let target = target as usize;
                    let target_count = &mut next_body_scratch[target];
                    if *target_count == u32::MAX {
                        *target_count = completed;
                        touched_body.push(target);
                    } else {
                        *target_count = (*target_count).min(completed);
                    }
                }
            }

            let mut next_suffix = SmallVec::<[u32; 4]>::new();
            for &suffix_state in state.suffix_states.iter() {
                if let Some(target) = suffix_dfa.step(suffix_state, byte) {
                    next_suffix.push(target);
                }
            }

            let mut completed_boundary = u32::MAX;
            for &body_state in &touched_body {
                if body_dfa.finalizers(body_state as u32).contains(0) {
                    completed_boundary = completed_boundary.min(
                        next_body_scratch[body_state].saturating_add(1),
                    );
                }
            }
            if completed_boundary != u32::MAX {
                next_suffix.push(0);
                if completed_boundary < max as u32 {
                    if next_body_scratch[0] == u32::MAX {
                        next_body_scratch[0] = completed_boundary;
                        touched_body.push(0);
                    } else {
                        next_body_scratch[0] =
                            next_body_scratch[0].min(completed_boundary);
                    }
                }
            }

            touched_body.retain(|body_state| {
                let body_state = *body_state;
                if body_dfa
                    .possible_future_group_ids(body_state as u32)
                    .contains(0)
                {
                    true
                } else {
                    next_body_scratch[body_state] = u32::MAX;
                    false
                }
            });
            next_suffix.retain(|suffix_state| {
                let suffix_state = *suffix_state;
                suffix_dfa.finalizers(suffix_state).contains(0)
                    || suffix_dfa
                        .possible_future_group_ids(suffix_state)
                        .contains(0)
            });
            next_suffix.sort_unstable();
            next_suffix.dedup();

            if touched_body.is_empty() && next_suffix.is_empty() {
                continue;
            }

            let next = ZeroMinRepeatSuffixState {
                body_min_counts: next_body_scratch.clone().into_boxed_slice(),
                suffix_states: next_suffix.into_vec().into_boxed_slice(),
            };
            for &body_state in &touched_body {
                next_body_scratch[body_state] = u32::MAX;
            }
            let target = if let Some(&target) = state_map.get(&next) {
                target
            } else {
                let target = dfa.add_state();
                set_zero_min_repeat_suffix_metadata(&mut dfa, target, &next, suffix_dfa);
                state_map.insert(next.clone(), target);
                states.push(next.clone());
                worklist.push_back((target, next));
                target
            };
            target_by_class[class] = target;
        }
        let transitions = byte_to_class
            .iter()
            .enumerate()
            .filter_map(|(byte, &class)| {
                let target = target_by_class[class as usize];
                (target != u32::MAX).then_some((byte as u8, target))
            })
            .collect();
        dfa.set_transitions_from_sorted_entries(state_id, transitions);
    }

    // Hopcroft-style minimization is counterproductive for broad direct
    // residual DFAs: a few thousand states with dense byte rows can cost
    // seconds even when almost no states merge. Downstream product/DWA
    // construction is already designed to consume unminimized deterministic
    // components. Keep minimization for compact results where it is cheap and
    // useful, but preserve the exact direct DFA as-is above that threshold.
    let transitions = dfa_transition_count(&dfa);
    if !preserve_coordinates && dfa.num_states() <= 2_048 && transitions <= 100_000 {
        dfa = dfa.minimize();
        states.clear();
        state_map.clear();
    }
    Some(ZeroMinRepeatSuffixBuild {
        dfa,
        states,
        state_by_key: state_map,
    })
}
