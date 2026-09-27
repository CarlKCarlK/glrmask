//! Shared finite-state language and metadata analyses.

use crate::automata::lexer::dfa::DFA;
use crate::ds::u8set::U8Set;
use std::collections::VecDeque;
use super::product::ProductComponent;

pub(super) fn productive_dfa_states(dfa: &DFA) -> Vec<bool> {
    let mut reverse_edges = vec![Vec::new(); dfa.num_states()];
    for (state_id, state) in dfa.states().iter().enumerate() {
        for (_, &target) in state.transitions.iter() {
            reverse_edges[target as usize].push(state_id as u32);
        }
    }

    let mut productive = vec![false; dfa.num_states()];
    let mut stack = Vec::new();
    for state_id in 0..dfa.num_states() as u32 {
        if !dfa.finalizers(state_id).is_empty() {
            productive[state_id as usize] = true;
            stack.push(state_id);
        }
    }

    while let Some(state_id) = stack.pop() {
        for &pred in &reverse_edges[state_id as usize] {
            if !productive[pred as usize] {
                productive[pred as usize] = true;
                stack.push(pred);
            }
        }
    }

    productive
}

pub(super) fn dfa_is_nonnullable_and_prefix_free(dfa: &DFA) -> bool {
    if !dfa.finalizers(0).is_empty() {
        return false;
    }

    let productive = productive_dfa_states(dfa);
    for state in dfa.states() {
        if state.finalizers.is_empty() {
            continue;
        }
        for (_, &target) in state.transitions.iter() {
            if productive[target as usize] {
                return false;
            }
        }
    }

    true
}

pub(super) fn dfa_transition_count(dfa: &DFA) -> usize {
    dfa.transition_count()
}

/// Whether the language accepted by one already-materializable product
/// component is finite.  We only care about states that are both reachable
/// from the component root and can still participate in an accepting word;
/// that live subgraph accepts an infinite language iff it contains a cycle.
///
/// This is a resource-safety proof for pairing an exact virtual bounded-repeat
/// coordinate with an ordinary product coordinate.  If the ordinary language
/// is finite, its live DAG bounds every product walk independently of the
/// repeat's declared (possibly billion-scale) upper bound.
pub(super) fn single_group_dfa_state_is_live(dfa: &DFA, state: u32) -> bool {
    dfa.finalizers(state).contains(0) || dfa.possible_future_group_ids(state).contains(0)
}

fn single_group_dfa_has_finite_language(dfa: &DFA) -> bool {
    if dfa.num_states() == 0 {
        return true;
    }

    let is_live = |state: u32| single_group_dfa_state_is_live(dfa, state);
    if !is_live(0) {
        return true;
    }

    let mut reachable = vec![false; dfa.num_states()];
    let mut stack = vec![0u32];
    reachable[0] = true;
    while let Some(state) = stack.pop() {
        for (_, &target) in dfa.states()[state as usize].transitions.iter() {
            if is_live(target) && !reachable[target as usize] {
                reachable[target as usize] = true;
                stack.push(target);
            }
        }
    }

    let mut indegree = vec![0u32; dfa.num_states()];
    let mut live_count = 0usize;
    for state in 0..dfa.num_states() as u32 {
        if !reachable[state as usize] {
            continue;
        }
        live_count += 1;
        for (_, &target) in dfa.states()[state as usize].transitions.iter() {
            if reachable[target as usize] {
                indegree[target as usize] = indegree[target as usize].saturating_add(1);
            }
        }
    }
    let mut queue = VecDeque::new();
    for state in 0..dfa.num_states() as u32 {
        if reachable[state as usize] && indegree[state as usize] == 0 {
            queue.push_back(state);
        }
    }
    let mut removed = 0usize;
    while let Some(state) = queue.pop_front() {
        removed += 1;
        for (_, &target) in dfa.states()[state as usize].transitions.iter() {
            if !reachable[target as usize] {
                continue;
            }
            indegree[target as usize] -= 1;
            if indegree[target as usize] == 0 {
                queue.push_back(target);
            }
        }
    }
    removed == live_count
}

/// Exact maximum accepted suffix length from every live state of a
/// deterministic one-group DFA. Returns `None` if the live subgraph contains a
/// cycle or inconsistent future metadata, proving that no finite bound is
/// available. Dead states are represented by `None` entries.
pub(super) fn single_group_dfa_max_remaining(dfa: &DFA) -> Option<Vec<Option<u32>>> {
    fn visit(
        dfa: &DFA,
        state: u32,
        colors: &mut [u8],
        memo: &mut [Option<u32>],
    ) -> Option<u32> {
        let index = state as usize;
        match *colors.get(index)? {
            1 => return None,
            2 => return *memo.get(index)?,
            _ => {}
        }
        if !single_group_dfa_state_is_live(dfa, state) {
            colors[index] = 2;
            return None;
        }
        colors[index] = 1;
        let mut best = dfa.finalizers(state).contains(0).then_some(0u32);
        for (_, target) in dfa.transitions(state) {
            if !single_group_dfa_state_is_live(dfa, target) {
                continue;
            }
            let remaining = visit(dfa, target, colors, memo)?.checked_add(1)?;
            best = Some(best.map_or(remaining, |current| current.max(remaining)));
        }
        let best = best?;
        memo[index] = Some(best);
        colors[index] = 2;
        Some(best)
    }

    let mut colors = vec![0u8; dfa.num_states()];
    let mut memo = vec![None; dfa.num_states()];
    for state in 0..dfa.num_states() as u32 {
        if single_group_dfa_state_is_live(dfa, state) && colors[state as usize] == 0 {
            visit(dfa, state, &mut colors, &mut memo)?;
        }
    }
    Some(memo)
}

pub(super) fn product_component_has_finite_language(component: &ProductComponent) -> bool {
    match component {
        ProductComponent::Materialized(dfa)
        | ProductComponent::MaterializedZeroMinRepeatSuffix { dfa, .. } => {
            single_group_dfa_has_finite_language(dfa)
        }
        ProductComponent::VirtualFixedSequence { .. } => true,
        ProductComponent::VirtualBoundedRepeat { .. } => false,
    }
}

pub(super) fn refine_u8_partitions(partitions: Vec<U8Set>, split: U8Set) -> Vec<U8Set> {
    let mut next_partitions = Vec::with_capacity(partitions.len() * 2);
    for partition in partitions {
        let intersection = partition.intersection(&split);
        let difference = partition.difference(&split);
        if !intersection.is_empty() {
            next_partitions.push(intersection);
        }
        if !difference.is_empty() {
            next_partitions.push(difference);
        }
    }
    next_partitions
}
